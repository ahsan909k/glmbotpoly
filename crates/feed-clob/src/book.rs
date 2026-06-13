//! Pure L2 book state for one token: snapshot replace, level deltas, top
//! extraction, and integrity predicates. No IO, no clocks.
//!
//! Internals are `BTreeMap<Decimal, Decimal>` keyed by raw price — tick
//! regime flips are a non-event for book state (every 0.01-grid price is on
//! the 0.001 grid). [`core_types::Price`]/[`core_types::Size`] validation
//! happens at the boundary ([`BookState::to_snapshot`] /
//! [`BookState::top_of_book`]), where the published sorting contract (bids
//! best-first/highest, asks best-first/lowest — core-types `book.rs`) is
//! produced from the map order.

use std::collections::BTreeMap;

use core_types::{
    BookLevel, BookSnapshot, ConditionId, Decimal, Price, Side, Size, TickSize, TimestampMs,
    TokenId, TopOfBook,
};

/// A level that cannot survive the core-types boundary. Reaching this from
/// [`BookState::to_snapshot`] means an invalid level got past the apply-time
/// filter — treated upstream as desync evidence, never a panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BookError {
    /// Price outside (0, 1) or finer than the 4-dp venue grid.
    #[error("level price {0} is not a valid venue price")]
    BadPrice(Decimal),
    /// Negative size.
    #[error("level size {0} is negative")]
    BadSize(Decimal),
}

/// True when a wire price could be a real venue level: inside the open
/// (0, 1) interval and on the finest (4-dp) venue grid.
fn valid_price(price: Decimal) -> bool {
    let normalized = price.normalize();
    normalized > Decimal::ZERO && normalized < Decimal::ONE && normalized.scale() <= 4
}

fn build_level(price: Decimal, size: Decimal) -> Result<BookLevel, BookError> {
    Ok(BookLevel {
        price: Price::on_grid(price, TickSize::T00001).map_err(|_| BookError::BadPrice(price))?,
        size: Size::new(size).map_err(|_| BookError::BadSize(size))?,
    })
}

/// Result of applying one level change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeOutcome {
    /// Input invalid (bad price/size) — dropped, state untouched.
    Rejected,
    /// Applied; `consumed` opposite-side levels were removed by implied
    /// consumption (a trade at the touch — see [`BookState::apply_change`]).
    Applied {
        /// Opposite-side levels removed to keep the book uncrossed.
        consumed: usize,
    },
}

/// L2 book state for one token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BookState {
    /// Bid levels, price → displayed size. Best bid = max key.
    bids: BTreeMap<Decimal, Decimal>,
    /// Ask levels, price → displayed size. Best ask = min key.
    asks: BTreeMap<Decimal, Decimal>,
}

impl BookState {
    /// Empty book.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the whole book from a venue snapshot. Levels with an
    /// invalid price or a negative size are dropped and counted (the
    /// returned number; zero-size levels are a benign "no level" encoding
    /// and skipped silently). Prices are normalized so `0.50` and `0.5`
    /// can never coexist as distinct levels.
    pub fn apply_snapshot(
        &mut self,
        bids: &[(Decimal, Decimal)],
        asks: &[(Decimal, Decimal)],
    ) -> usize {
        self.bids.clear();
        self.asks.clear();
        let mut rejected = 0;
        for (side_levels, map) in [(bids, &mut self.bids), (asks, &mut self.asks)] {
            for &(price, size) in side_levels {
                if size.is_zero() {
                    continue;
                }
                if !valid_price(price) || size.is_sign_negative() {
                    rejected += 1;
                    continue;
                }
                map.insert(price.normalize(), size.normalize());
            }
        }
        rejected
    }

    /// Applies one `price_change` level update: `size` is the new TOTAL
    /// displayed size at `price` (zero removes the level); removing a level
    /// that does not exist is a valid no-op.
    ///
    /// A change that would cross the book **uncrosses it by implied
    /// consumption**: a trade at the touch arrives as a level update on the
    /// aggressor's side carrying the post-trade venue tops, while the
    /// consumed opposite level's removal is never sent as a delta — it only
    /// materializes in the trade-driven `book` snapshot moments later
    /// (live-verified 2026-06-12). Dropping the opposite-side levels at or
    /// better than the new price reproduces exactly the venue state its own
    /// carried `best_bid`/`best_ask` report, and the snapshot then confirms
    /// wholesale.
    pub fn apply_change(&mut self, side: Side, price: Decimal, size: Decimal) -> ChangeOutcome {
        let price = price.normalize();
        let (own, opposite) = match side {
            Side::Buy => (&mut self.bids, &mut self.asks),
            Side::Sell => (&mut self.asks, &mut self.bids),
        };
        if size.is_zero() {
            own.remove(&price);
            return ChangeOutcome::Applied { consumed: 0 };
        }
        if !valid_price(price) || size.is_sign_negative() {
            return ChangeOutcome::Rejected;
        }
        own.insert(price, size.normalize());
        let consumed_prices: Vec<Decimal> = match side {
            // A bid at `price` consumes asks priced at or below it…
            Side::Buy => opposite.range(..=price).map(|(p, _)| *p).collect(),
            // …an ask at `price` consumes bids priced at or above it.
            Side::Sell => opposite.range(price..).map(|(p, _)| *p).collect(),
        };
        for consumed in &consumed_prices {
            opposite.remove(consumed);
        }
        ChangeOutcome::Applied {
            consumed: consumed_prices.len(),
        }
    }

    /// Best bid as (price, size), if any.
    #[must_use]
    pub fn best_bid(&self) -> Option<(Decimal, Decimal)> {
        self.bids.last_key_value().map(|(p, s)| (*p, *s))
    }

    /// Best ask as (price, size), if any.
    #[must_use]
    pub fn best_ask(&self) -> Option<(Decimal, Decimal)> {
        self.asks.first_key_value().map(|(p, s)| (*p, *s))
    }

    /// True when best bid ≥ best ask. Delta application keeps the book
    /// uncrossed by construction ([`BookState::apply_change`]), so this can
    /// only be true after a crossed venue *snapshot* — defensive evidence
    /// that the venue state itself is garbage.
    #[must_use]
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => bid >= ask,
            _ => false,
        }
    }

    /// True when both sides are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// (bid levels, ask levels) counts.
    #[must_use]
    pub fn depth(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }

    /// Builds the validated top-of-book view (no full-Vec allocation).
    ///
    /// # Errors
    /// [`BookError`] when a top level fails core-types validation — possible
    /// only if an invalid level slipped past the apply-time filter; treated
    /// upstream as desync evidence.
    pub fn top_of_book(&self, ts: TimestampMs) -> Result<TopOfBook, BookError> {
        let bid = self
            .best_bid()
            .map(|(p, s)| build_level(p, s))
            .transpose()?;
        let ask = self
            .best_ask()
            .map(|(p, s)| build_level(p, s))
            .transpose()?;
        Ok(TopOfBook { bid, ask, ts })
    }

    /// Builds the full validated [`BookSnapshot`] honoring the documented
    /// sorting contract: bids best (highest) first, asks best (lowest)
    /// first.
    ///
    /// # Errors
    /// [`BookError`] when any level fails core-types validation (see
    /// [`BookState::top_of_book`]).
    pub fn to_snapshot(
        &self,
        token_id: &TokenId,
        condition_id: &ConditionId,
        ts: TimestampMs,
        seq_hash: Option<String>,
    ) -> Result<BookSnapshot, BookError> {
        let bids = self
            .bids
            .iter()
            .rev()
            .map(|(&p, &s)| build_level(p, s))
            .collect::<Result<Vec<_>, _>>()?;
        let asks = self
            .asks
            .iter()
            .map(|(&p, &s)| build_level(p, s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BookSnapshot {
            token_id: token_id.clone(),
            condition_id: condition_id.clone(),
            bids,
            asks,
            ts,
            seq_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const TS: TimestampMs = TimestampMs::from_millis(1_700_000_000_000);

    fn token() -> TokenId {
        TokenId::new("1").unwrap()
    }

    fn condition() -> ConditionId {
        ConditionId::new(format!("0x{}", "00".repeat(32))).unwrap()
    }

    fn populated() -> BookState {
        let mut book = BookState::new();
        let rejected = book.apply_snapshot(
            &[
                (dec!(0.48), dec!(100)),
                (dec!(0.47), dec!(50)),
                (dec!(0.49), dec!(10)),
            ],
            &[(dec!(0.53), dec!(75)), (dec!(0.52), dec!(200))],
        );
        assert_eq!(rejected, 0);
        book
    }

    #[test]
    fn snapshot_sorts_per_contract() {
        let book = populated();
        let snap = book.to_snapshot(&token(), &condition(), TS, None).unwrap();
        let bid_prices: Vec<Decimal> = snap.bids.iter().map(|l| l.price.as_decimal()).collect();
        let ask_prices: Vec<Decimal> = snap.asks.iter().map(|l| l.price.as_decimal()).collect();
        // Bids best (highest) first; asks best (lowest) first.
        assert_eq!(bid_prices, vec![dec!(0.49), dec!(0.48), dec!(0.47)]);
        assert_eq!(ask_prices, vec![dec!(0.52), dec!(0.53)]);
        assert_eq!(snap.top().bid.unwrap().size.as_decimal(), dec!(10));
        assert_eq!(snap.top().ask.unwrap().size.as_decimal(), dec!(200));
    }

    #[test]
    fn snapshot_rejects_garbage_levels_and_skips_zero_sizes() {
        let mut book = BookState::new();
        let rejected = book.apply_snapshot(
            &[
                (dec!(0.48), dec!(100)),  // good
                (dec!(0), dec!(5)),       // price 0 — invalid
                (dec!(1), dec!(5)),       // price 1 — invalid
                (dec!(0.12345), dec!(5)), // 5 dp — off any venue grid
                (dec!(0.40), dec!(-3)),   // negative size
                (dec!(0.41), dec!(0)),    // zero size — benign skip
            ],
            &[],
        );
        assert_eq!(rejected, 4);
        assert_eq!(book.depth(), (1, 0));
        assert_eq!(book.best_bid(), Some((dec!(0.48), dec!(100))));
    }

    /// Applied with no implied consumption.
    fn clean(outcome: ChangeOutcome) -> bool {
        outcome == ChangeOutcome::Applied { consumed: 0 }
    }

    #[test]
    fn change_updates_removes_and_rejects() {
        let mut book = populated();
        // Update an existing level's total size.
        assert!(clean(book.apply_change(Side::Buy, dec!(0.48), dec!(60))));
        assert_eq!(book.best_bid(), Some((dec!(0.49), dec!(10))));
        // Remove the best bid via size 0.
        assert!(clean(book.apply_change(Side::Buy, dec!(0.49), dec!(0))));
        assert_eq!(book.best_bid(), Some((dec!(0.48), dec!(60))));
        // Removing a level that does not exist is a valid no-op.
        assert!(clean(book.apply_change(Side::Sell, dec!(0.99), dec!(0))));
        // New ask level inside the spread becomes best.
        assert!(clean(book.apply_change(Side::Sell, dec!(0.51), dec!(5))));
        assert_eq!(book.best_ask(), Some((dec!(0.51), dec!(5))));
        // Garbage is rejected without touching state.
        assert_eq!(
            book.apply_change(Side::Buy, dec!(0), dec!(5)),
            ChangeOutcome::Rejected
        );
        assert_eq!(
            book.apply_change(Side::Buy, dec!(0.5), dec!(-1)),
            ChangeOutcome::Rejected
        );
        assert_eq!(book.depth(), (2, 3));
    }

    #[test]
    fn price_normalization_prevents_duplicate_levels() {
        let mut book = BookState::new();
        assert!(clean(book.apply_change(Side::Buy, dec!(0.50), dec!(10))));
        assert!(clean(book.apply_change(Side::Buy, dec!(0.5000), dec!(20))));
        assert_eq!(book.depth(), (1, 0));
        assert_eq!(book.best_bid(), Some((dec!(0.5), dec!(20))));
        // Removal through a differently-scaled price hits the same level.
        assert!(clean(book.apply_change(Side::Buy, dec!(0.5), dec!(0))));
        assert!(book.is_empty());
    }

    #[test]
    fn crossing_change_consumes_opposite_levels() {
        // The live-verified trade-at-the-touch pattern: a bid arriving AT
        // the best ask means that ask was just consumed by a trade (its
        // removal never comes as a delta — only in the next snapshot).
        let mut book = populated();
        assert_eq!(
            book.apply_change(Side::Buy, dec!(0.52), dec!(5)),
            ChangeOutcome::Applied { consumed: 1 }
        );
        assert!(!book.is_crossed());
        assert_eq!(book.best_bid(), Some((dec!(0.52), dec!(5))));
        assert_eq!(book.best_ask(), Some((dec!(0.53), dec!(75))));
        // An aggressive ask sweeping through several bid levels consumes
        // them all.
        assert_eq!(
            book.apply_change(Side::Sell, dec!(0.48), dec!(7)),
            ChangeOutcome::Applied { consumed: 3 } // bids 0.52, 0.49, 0.48
        );
        assert!(!book.is_crossed());
        assert_eq!(book.best_bid(), Some((dec!(0.47), dec!(50))));
        assert_eq!(book.best_ask(), Some((dec!(0.48), dec!(7))));
    }

    #[test]
    fn crossed_venue_snapshot_is_detected() {
        // Deltas can no longer cross the book; only a venue snapshot can —
        // defensive evidence the venue state itself is garbage.
        let mut book = BookState::new();
        book.apply_snapshot(&[(dec!(0.60), dec!(5))], &[(dec!(0.50), dec!(5))]);
        assert!(book.is_crossed());
        // One-sided and empty books never count as crossed.
        let mut one_sided = BookState::new();
        assert!(!one_sided.is_crossed());
        assert!(clean(one_sided.apply_change(Side::Buy, dec!(0.9), dec!(1))));
        assert!(!one_sided.is_crossed());
    }

    #[test]
    fn top_of_book_handles_empty_sides() {
        let book = BookState::new();
        let top = book.top_of_book(TS).unwrap();
        assert!(top.bid.is_none());
        assert!(top.ask.is_none());
        assert_eq!(top.ts, TS);

        let populated = populated();
        let top = populated.top_of_book(TS).unwrap();
        assert_eq!(top.bid.unwrap().price.as_decimal(), dec!(0.49));
        assert_eq!(top.ask.unwrap().price.as_decimal(), dec!(0.52));
        assert_eq!(top.mid(), Some(dec!(0.505)));
    }

    #[test]
    fn three_dp_prices_coexist_with_two_dp_after_tick_flip() {
        let mut book = BookState::new();
        assert!(clean(book.apply_change(Side::Sell, dec!(0.97), dec!(10))));
        assert!(clean(book.apply_change(Side::Sell, dec!(0.965), dec!(5))));
        assert!(clean(book.apply_change(Side::Sell, dec!(0.9999), dec!(1))));
        let snap = book.to_snapshot(&token(), &condition(), TS, None).unwrap();
        let prices: Vec<Decimal> = snap.asks.iter().map(|l| l.price.as_decimal()).collect();
        assert_eq!(prices, vec![dec!(0.965), dec!(0.97), dec!(0.9999)]);
    }

    #[test]
    fn snapshot_replace_drops_prior_state() {
        let mut book = populated();
        let rejected = book.apply_snapshot(&[(dec!(0.10), dec!(1))], &[(dec!(0.90), dec!(2))]);
        assert_eq!(rejected, 0);
        assert_eq!(book.depth(), (1, 1));
        assert_eq!(book.best_bid(), Some((dec!(0.1), dec!(1))));
        assert_eq!(book.best_ask(), Some((dec!(0.9), dec!(2))));
    }

    #[test]
    fn content_equality_is_value_based() {
        let a = populated();
        let mut b = populated();
        assert_eq!(a, b);
        assert!(clean(b.apply_change(Side::Buy, dec!(0.30), dec!(1))));
        assert_ne!(a, b);
        assert!(clean(b.apply_change(Side::Buy, dec!(0.30), dec!(0))));
        assert_eq!(a, b, "quiet-vs-broken check relies on value equality");
    }

    #[test]
    fn randomized_applies_match_naive_model() {
        // Tiny deterministic LCG; no rand crate (CLAUDE.md §3).
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u32
        };
        let mut book = BookState::new();
        let mut model: std::collections::HashMap<(bool, Decimal), Decimal> =
            std::collections::HashMap::new();
        for _ in 0..2_000 {
            let is_bid = next() % 2 == 0;
            let price = Decimal::new(i64::from(next() % 99 + 1), 2); // 0.01..=0.99
            let size = Decimal::from(next() % 5); // 0..=4, zero = remove
            let side = if is_bid { Side::Buy } else { Side::Sell };
            assert_ne!(
                book.apply_change(side, price, size),
                ChangeOutcome::Rejected
            );
            if size.is_zero() {
                model.remove(&(is_bid, price));
            } else {
                model.insert((is_bid, price), size);
                // Mirror the implied consumption: the new level removes
                // every opposite-side level at or better than its price.
                model.retain(|&(model_bid, model_price), _| {
                    if model_bid == is_bid {
                        return true;
                    }
                    if is_bid {
                        model_price > price // surviving asks are above the bid
                    } else {
                        model_price < price // surviving bids are below the ask
                    }
                });
            }
        }
        let (bid_count, ask_count) = book.depth();
        assert_eq!(
            bid_count,
            model.keys().filter(|(is_bid, _)| *is_bid).count()
        );
        assert_eq!(
            ask_count,
            model.keys().filter(|(is_bid, _)| !*is_bid).count()
        );
        let snap = book.to_snapshot(&token(), &condition(), TS, None).unwrap();
        for level in &snap.bids {
            assert_eq!(
                model.get(&(true, level.price.as_decimal())).copied(),
                Some(level.size.as_decimal())
            );
        }
        for level in &snap.asks {
            assert_eq!(
                model.get(&(false, level.price.as_decimal())).copied(),
                Some(level.size.as_decimal())
            );
        }
    }
}
