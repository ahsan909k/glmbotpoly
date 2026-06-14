//! A per-token local view of the live Polymarket book, fed from the real
//! `feed-clob` event stream (`Event::Book` / `Event::TopOfBook`).
//!
//! It is *read* market data — paper never owns the real book. We keep the
//! latest full depth (for queue position and marketable walks) plus the
//! freshest top (for the post-only cross check), and apply only minimal local
//! depletion when a paper *taker* consumes displayed depth, so two simultaneous
//! paper takers can't both eat the same shares (the next snapshot overwrites the
//! depletion anyway). §9: when the two sources disagree we always pick the
//! interpretation that makes us fill *less*.

use core_types::{BookLevel, BookSnapshot, Decimal, OrderQty, Price, Side, Size, TimestampMs};

/// Dust tolerance for "did the marketable order fully fill" — absorbs the
/// `remaining/price * price` rounding when converting a dollar budget to shares.
const DUST: Decimal = Decimal::from_parts(1, 0, 0, false, 6); // 0.000001

/// The plan produced by walking the book for a marketable order. Read-only:
/// the engine commits it by emitting fills and calling [`SimBook::consume`].
#[derive(Debug, Clone, PartialEq)]
pub struct MarketableWalk {
    /// `(level price, shares taken)` per consumed level, best price first.
    pub fills: Vec<(Price, Size)>,
    /// True when the whole order (full notional for BUY, full share count for
    /// SELL) was satisfied within the displayed depth and worst-price limit.
    pub fully_filled: bool,
}

impl MarketableWalk {
    /// Total shares this walk would fill.
    #[must_use]
    pub fn total_shares(&self) -> Size {
        self.fills.iter().map(|(_, s)| *s).sum()
    }
}

/// A local mirror of one outcome token's order book.
#[derive(Debug, Clone, Default)]
pub struct SimBook {
    /// Bid levels, best (highest price) first — from the last `Event::Book`.
    bids: Vec<BookLevel>,
    /// Ask levels, best (lowest price) first — from the last `Event::Book`.
    asks: Vec<BookLevel>,
    /// Timestamp of the last full snapshot.
    book_ts: TimestampMs,
    /// Freshest best bid from an `Event::TopOfBook`, if newer than the snapshot.
    top_bid: Option<BookLevel>,
    /// Freshest best ask from an `Event::TopOfBook`, if newer than the snapshot.
    top_ask: Option<BookLevel>,
    /// Timestamp of the last top-of-book update.
    top_ts: TimestampMs,
}

impl SimBook {
    /// An empty book.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces depth wholesale from a venue snapshot (the ground truth).
    pub fn apply_snapshot(&mut self, snap: &BookSnapshot) {
        self.bids = snap.bids.clone();
        self.asks = snap.asks.clone();
        self.book_ts = snap.ts;
    }

    /// Refreshes the top from a `TopOfBook` update (used for the cross check).
    pub fn apply_top(&mut self, bid: Option<BookLevel>, ask: Option<BookLevel>, ts: TimestampMs) {
        self.top_bid = bid;
        self.top_ask = ask;
        self.top_ts = ts;
    }

    /// Best bid: the fresher of the snapshot top and the `TopOfBook` update
    /// wins outright (so a fresh empty snapshot overrides a stale top).
    #[must_use]
    pub fn best_bid(&self) -> Option<BookLevel> {
        if self.top_ts > self.book_ts {
            self.top_bid
        } else {
            self.bids.first().copied()
        }
    }

    /// Best ask: see [`SimBook::best_bid`].
    #[must_use]
    pub fn best_ask(&self) -> Option<BookLevel> {
        if self.top_ts > self.book_ts {
            self.top_ask
        } else {
            self.asks.first().copied()
        }
    }

    /// Total displayed size at exactly `price` on `side` — the pessimistic
    /// queue position a resting order joins *behind* at activation. If a fresher
    /// top reports the same level larger, we take the larger (more queue ahead ⇒
    /// fewer fills ⇒ conservative).
    #[must_use]
    pub fn displayed_at(&self, side: Side, price: Price) -> Size {
        let levels = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };
        let mut sz = levels
            .iter()
            .find(|l| l.price == price)
            .map_or(Size::ZERO, |l| l.size);
        if self.top_ts > self.book_ts {
            let top = match side {
                Side::Buy => self.top_bid,
                Side::Sell => self.top_ask,
            };
            if let Some(l) = top
                && l.price == price
            {
                sz = size_max(sz, l.size);
            }
        }
        sz
    }

    /// Plans a marketable walk for `qty` on `side` against the opposite book,
    /// respecting the `worst` price limit. Read-only.
    #[must_use]
    pub fn walk_marketable(&self, side: Side, worst: Price, qty: OrderQty) -> MarketableWalk {
        match (side, qty) {
            // BUY consumes asks (lowest first) up to a dollar budget; we will
            // not pay above `worst`.
            (Side::Buy, OrderQty::Notional(budget)) => {
                self.walk_buy_notional(worst.as_decimal(), budget.as_decimal())
            }
            // SELL consumes bids (highest first) up to a share count; we will
            // not sell below `worst`.
            (Side::Sell, OrderQty::Shares(shares)) => {
                self.walk_sell_shares(worst.as_decimal(), shares.as_decimal())
            }
            // Mismatched qty kinds never reach here (the engine rejects them at
            // placement, mirroring live's `require_*`). Empty plan, not filled.
            _ => MarketableWalk {
                fills: Vec::new(),
                fully_filled: false,
            },
        }
    }

    fn walk_buy_notional(&self, worst: Decimal, budget: Decimal) -> MarketableWalk {
        let mut remaining = budget;
        let mut fills = Vec::new();
        for lvl in &self.asks {
            if remaining <= DUST {
                break;
            }
            let a = lvl.price.as_decimal();
            if a > worst {
                break; // past the slippage limit — stop, do not fill worse.
            }
            // Shares affordable with the remaining budget at this price (a > 0
            // always, by the Price invariant), capped by displayed depth.
            let affordable = remaining / a;
            let take = min_dec(affordable, lvl.size.as_decimal());
            if take <= Decimal::ZERO {
                break;
            }
            fills.push((lvl.price, Size::new(take).unwrap_or(Size::ZERO)));
            remaining -= take * a;
        }
        MarketableWalk {
            fully_filled: remaining <= DUST,
            fills,
        }
    }

    fn walk_sell_shares(&self, worst: Decimal, shares: Decimal) -> MarketableWalk {
        let mut remaining = shares;
        let mut fills = Vec::new();
        for lvl in &self.bids {
            if remaining <= DUST {
                break;
            }
            let b = lvl.price.as_decimal();
            if b < worst {
                break; // below the slippage limit — stop, do not sell cheaper.
            }
            let take = min_dec(remaining, lvl.size.as_decimal());
            if take <= Decimal::ZERO {
                break;
            }
            fills.push((lvl.price, Size::new(take).unwrap_or(Size::ZERO)));
            remaining -= take;
        }
        MarketableWalk {
            fully_filled: remaining <= DUST,
            fills,
        }
    }

    /// Locally depletes displayed depth after a paper taker consumed it (until
    /// the next snapshot replaces the book), so simultaneous paper takers can't
    /// double-eat the same shares.
    pub fn consume(&mut self, side: Side, price: Price, size: Size) {
        let levels = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        if let Some(idx) = levels.iter().position(|l| l.price == price) {
            let remaining = levels[idx].size.saturating_sub(size);
            if remaining.is_zero() {
                levels.remove(idx);
            } else {
                levels[idx].size = remaining;
            }
        }
    }
}

/// The larger of two sizes by value.
fn size_max(a: Size, b: Size) -> Size {
    if a.as_decimal() >= b.as_decimal() {
        a
    } else {
        b
    }
}

/// The smaller of two decimals.
fn min_dec(a: Decimal, b: Decimal) -> Decimal {
    if a <= b { a } else { b }
}

#[cfg(test)]
mod tests {
    use core_types::{ConditionId, Dollars, RoundDir, TickSize, TokenId};
    use rust_decimal::dec;

    use super::*;

    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn lvl(p: Decimal, s: Decimal) -> BookLevel {
        BookLevel {
            price: px(p),
            size: sz(s),
        }
    }

    fn snapshot(bids: &[(Decimal, Decimal)], asks: &[(Decimal, Decimal)], ts: i64) -> BookSnapshot {
        BookSnapshot {
            token_id: TokenId::new("1").unwrap(),
            condition_id: ConditionId::new(format!("0x{}", "00".repeat(32))).unwrap(),
            bids: bids.iter().map(|(p, s)| lvl(*p, *s)).collect(),
            asks: asks.iter().map(|(p, s)| lvl(*p, *s)).collect(),
            ts: TimestampMs::from_millis(ts),
            seq_hash: None,
        }
    }

    fn loaded() -> SimBook {
        let mut b = SimBook::new();
        b.apply_snapshot(&snapshot(
            &[(dec!(0.49), dec!(100)), (dec!(0.48), dec!(200))],
            &[(dec!(0.51), dec!(80)), (dec!(0.52), dec!(150))],
            1_000,
        ));
        b
    }

    #[test]
    fn tops_and_displayed_at() {
        let b = loaded();
        assert_eq!(b.best_bid().map(|l| l.price), Some(px(dec!(0.49))));
        assert_eq!(b.best_ask().map(|l| l.price), Some(px(dec!(0.51))));
        assert_eq!(b.displayed_at(Side::Buy, px(dec!(0.49))), sz(dec!(100)));
        assert_eq!(b.displayed_at(Side::Sell, px(dec!(0.52))), sz(dec!(150)));
        // No level there → zero queue.
        assert_eq!(b.displayed_at(Side::Buy, px(dec!(0.40))), Size::ZERO);
    }

    #[test]
    fn fresher_top_overrides_and_widens_queue() {
        let mut b = loaded();
        // A newer top shows the best bid grown to 250 and best ask moved.
        b.apply_top(
            Some(lvl(dec!(0.49), dec!(250))),
            Some(lvl(dec!(0.53), dec!(10))),
            TimestampMs::from_millis(2_000),
        );
        assert_eq!(b.best_ask().map(|l| l.price), Some(px(dec!(0.53))));
        // Conservative: queue at our level is the larger (250, not 100).
        assert_eq!(b.displayed_at(Side::Buy, px(dec!(0.49))), sz(dec!(250)));
        // A stale top does NOT override a fresher snapshot.
        b.apply_snapshot(&snapshot(
            &[(dec!(0.50), dec!(5))],
            &[(dec!(0.55), dec!(5))],
            3_000,
        ));
        assert_eq!(b.best_bid().map(|l| l.price), Some(px(dec!(0.50))));
    }

    #[test]
    fn buy_walk_respects_limit_and_depth() {
        let b = loaded();
        // $50 to spend, limit 0.52. 0.51 level (80 sh) absorbs 50/0.51 ≈ 98 sh?
        // No — 80 sh costs 40.80; remaining 9.20 buys 9.20/0.52 ≈ 17.69 at 0.52.
        let walk = b.walk_marketable(
            Side::Buy,
            px(dec!(0.52)),
            OrderQty::Notional(Dollars::new(dec!(50))),
        );
        assert!(walk.fully_filled);
        assert_eq!(walk.fills[0].0, px(dec!(0.51)));
        assert_eq!(walk.fills[1].0, px(dec!(0.52)));

        // Tight limit 0.51 → only the first level; $50 can't all be spent there
        // (80 sh × 0.51 = 40.80 < 50) → not fully filled.
        let walk = b.walk_marketable(
            Side::Buy,
            px(dec!(0.51)),
            OrderQty::Notional(Dollars::new(dec!(50))),
        );
        assert!(!walk.fully_filled);
        assert_eq!(walk.fills.len(), 1);
    }

    #[test]
    fn sell_walk_respects_limit_and_depth() {
        let b = loaded();
        // Sell 120 shares, floor 0.48. 0.49 bid (100) + 0.48 bid (20 of 200).
        let walk = b.walk_marketable(Side::Sell, px(dec!(0.48)), OrderQty::Shares(sz(dec!(120))));
        assert!(walk.fully_filled);
        assert_eq!(walk.total_shares(), sz(dec!(120)));
        assert_eq!(walk.fills[0], (px(dec!(0.49)), sz(dec!(100))));
        assert_eq!(walk.fills[1], (px(dec!(0.48)), sz(dec!(20))));

        // Floor 0.49 → only 100 shares available → not fully filled.
        let walk = b.walk_marketable(Side::Sell, px(dec!(0.49)), OrderQty::Shares(sz(dec!(120))));
        assert!(!walk.fully_filled);
        assert_eq!(walk.total_shares(), sz(dec!(100)));
    }

    #[test]
    fn consume_depletes_then_removes() {
        let mut b = loaded();
        b.consume(Side::Sell, px(dec!(0.51)), sz(dec!(30)));
        assert_eq!(b.displayed_at(Side::Sell, px(dec!(0.51))), sz(dec!(50)));
        b.consume(Side::Sell, px(dec!(0.51)), sz(dec!(50)));
        assert_eq!(b.displayed_at(Side::Sell, px(dec!(0.51))), Size::ZERO);
        // Best ask is now the next level.
        assert_eq!(b.best_ask().map(|l| l.price), Some(px(dec!(0.52))));
    }
}
