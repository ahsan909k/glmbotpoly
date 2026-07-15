//! Client-side self-match / self-trade prevention (CLAUDE.md §7/§11), shared by
//! all three taker paths (momentum, late-window, model).
//!
//! Our quote manager only ever rests **BUY** orders on both outcome tokens. By
//! the §7 mirror (`BUY Up @ p ≡ SELL Down @ 1−p`), a live resting BUY on
//! `outcome.opposite()` at the complement price `1 − p` is economically an offer
//! to *sell* `outcome` at `p`, so it surfaces in `outcome`'s ask book at price
//! `p`. A taker FAK BUY that lifted it would be a self-match (a wash trade paying
//! taker fees on both legs). [`filter_asks`] removes (or shrinks) exactly those
//! ask levels before any taker walks the book.
//!
//! Pure and sans-IO. Naturally window-scoped: the quoter's [`RestingView`] holds
//! only its single active window, and each resting order is additionally matched
//! on its [`WindowId`], so the filter is a no-op for any window the quoter is not
//! currently quoting (and when `resting` is `None`) — which is exactly when a
//! self-match is impossible.
//!
//! Paper venues cannot reproduce the collision (a paper FAK never matches our own
//! resting paper orders — disjoint books), so this guard is proven by the unit
//! tests below and validated live.

use core_types::{BookLevel, Outcome, Size, WindowId};

use crate::quote_manager::RestingView;

/// Removes or shrinks ask levels that are our own §7-mirrored resting liquidity,
/// returning the ask depth a taker may safely lift.
///
/// For each ask at price `p`, subtract the remaining size
/// (`original_size − filled_size`) of our live resting BUY orders on
/// `outcome.opposite()` at `p.complement()` in the same `window`. A level fully
/// covered by our own liquidity is dropped; a partially-covered one is shrunk to
/// the external remainder. When `resting` is `None` (no active quoter window) the
/// asks are returned unchanged.
#[must_use]
pub fn filter_asks(
    outcome: Outcome,
    asks: &[BookLevel],
    window: WindowId,
    resting: Option<&RestingView>,
) -> Vec<BookLevel> {
    let Some(view) = resting else {
        return asks.to_vec();
    };
    let opp = outcome.opposite();
    asks.iter()
        .filter_map(|lvl| {
            let mirror_price = lvl.price.complement();
            let ours: Size = view
                .live_orders()
                .filter(|o| o.window == window && o.outcome == opp && o.price == mirror_price)
                .map(|o| o.original_size.saturating_sub(o.filled_size))
                .fold(Size::ZERO, |acc, s| acc + s);
            let remaining = lvl.size.saturating_sub(ours);
            if remaining.is_zero() {
                None // the whole level is ours → skip it
            } else {
                Some(BookLevel {
                    price: lvl.price,
                    size: remaining,
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quote_manager::{ClientId, RestingView};
    use core_types::{
        Asset, OrderId, OrderState, OrderUpdate, Price, Series, Side, Size, TickSize, TimestampMs,
        TokenId, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    const TICK: TickSize = TickSize::T001;

    fn window(open_ms: i64) -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(open_ms),
        }
    }

    fn price(d: rust_decimal::Decimal) -> Price {
        Price::on_grid(d, TICK).expect("grid price")
    }

    fn ask(p: rust_decimal::Decimal, sz: rust_decimal::Decimal) -> BookLevel {
        BookLevel {
            price: price(p),
            size: Size::new(sz).expect("size"),
        }
    }

    fn oid(seq: u64) -> OrderId {
        OrderId::new(format!("o{seq}")).expect("oid")
    }

    /// Places one live resting order in a view (via the `Accepted` path) with the
    /// given outcome, price, and size. Level/seq are arbitrary but unique.
    fn place(
        view: &mut RestingView,
        win: WindowId,
        outcome: Outcome,
        p: rust_decimal::Decimal,
        sz: rust_decimal::Decimal,
        seq: u64,
    ) {
        let cid = ClientId {
            open_ms: win.open_time.as_millis(),
            outcome,
            level: seq as u32,
            seq,
        };
        view.record_pending(oid(seq), cid, win, price(p), Size::new(sz).expect("size"));
    }

    /// Builds a full [`OrderUpdate`] carrying a new lifecycle state and cumulative fill.
    fn update(
        win: WindowId,
        seq: u64,
        state: OrderState,
        filled: rust_decimal::Decimal,
    ) -> OrderUpdate {
        OrderUpdate {
            order_id: oid(seq),
            window: win,
            token_id: TokenId::new("111").expect("token"),
            side: Side::Buy,
            state,
            price: price(dec!(0.70)),
            original_size: Size::new(dec!(20)).unwrap(),
            filled_size: Size::new(filled).expect("size"),
            reject_reason: None,
            ts_venue: None,
            ts_local: TimestampMs::from_millis(1),
        }
    }

    /// No resting view (or an empty one) ⇒ asks returned unchanged.
    #[test]
    fn none_or_empty_is_a_no_op() {
        let asks = [ask(dec!(0.30), dec!(50)), ask(dec!(0.31), dec!(40))];
        assert_eq!(filter_asks(Outcome::Up, &asks, window(0), None), asks);
        let empty = RestingView::new();
        assert_eq!(
            filter_asks(Outcome::Up, &asks, window(0), Some(&empty)),
            asks
        );
    }

    /// An ask fully covered by our mirrored resting BUY is skipped entirely — the
    /// taker sees `NoAsks` at that price. (Buy Up @ 0.30 ⇄ our Down ask @ 0.70.)
    #[test]
    fn full_cover_skips_the_level() {
        let win = window(0);
        let mut view = RestingView::new();
        // Our resting BUY Down @ 0.70 = a mirrored ask on Up @ 0.30, size 50.
        place(&mut view, win, Outcome::Down, dec!(0.70), dec!(50), 1);
        let asks = [ask(dec!(0.30), dec!(50))];
        let out = filter_asks(Outcome::Up, &asks, win, Some(&view));
        assert!(out.is_empty(), "level fully ours ⇒ removed");
    }

    /// A partially-covered ask is shrunk to the external remainder.
    #[test]
    fn partial_cover_shrinks_the_level() {
        let win = window(0);
        let mut view = RestingView::new();
        place(&mut view, win, Outcome::Down, dec!(0.70), dec!(20), 1);
        let asks = [ask(dec!(0.30), dec!(50))];
        let out = filter_asks(Outcome::Up, &asks, win, Some(&view));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].price.as_decimal(), dec!(0.30));
        assert_eq!(out[0].size, Size::new(dec!(30)).unwrap()); // 50 − 20
    }

    /// Only the *remaining* (unfilled) size of our resting order shields depth.
    #[test]
    fn only_unfilled_size_shields() {
        let win = window(0);
        let mut view = RestingView::new();
        // Record a resting BUY Down @ 0.70 for 20, then fold in a partial fill of 15.
        place(&mut view, win, Outcome::Down, dec!(0.70), dec!(20), 1);
        view.apply_order_update(&update(win, 1, OrderState::Open, dec!(0)));
        view.apply_order_update(&update(win, 1, OrderState::PartiallyFilled, dec!(15)));
        let asks = [ask(dec!(0.30), dec!(50))];
        let out = filter_asks(Outcome::Up, &asks, win, Some(&view));
        assert_eq!(out[0].size, Size::new(dec!(45)).unwrap()); // 50 − (20 − 15)
    }

    /// A resting order on a *different* window / the *same* outcome / a *non-complement*
    /// price does not shield.
    #[test]
    fn scoping_non_matches_are_no_ops() {
        let win = window(0);
        let other = window(300_000);
        let asks = [ask(dec!(0.30), dec!(50))];

        // Different window.
        let mut v1 = RestingView::new();
        place(&mut v1, other, Outcome::Down, dec!(0.70), dec!(50), 1);
        assert_eq!(
            filter_asks(Outcome::Up, &asks, win, Some(&v1))[0].size,
            Size::new(dec!(50)).unwrap()
        );

        // Same outcome as the taker (a bid on Up, not the mirror).
        let mut v2 = RestingView::new();
        place(&mut v2, win, Outcome::Up, dec!(0.30), dec!(50), 2);
        assert_eq!(
            filter_asks(Outcome::Up, &asks, win, Some(&v2))[0].size,
            Size::new(dec!(50)).unwrap()
        );

        // Opposite outcome but a non-complement price (0.65 ≠ 1 − 0.30).
        let mut v3 = RestingView::new();
        place(&mut v3, win, Outcome::Down, dec!(0.65), dec!(50), 3);
        assert_eq!(
            filter_asks(Outcome::Up, &asks, win, Some(&v3))[0].size,
            Size::new(dec!(50)).unwrap()
        );
    }

    /// Symmetric: a taker BUY Down is shielded by our resting BUY Up at the
    /// complement.
    #[test]
    fn symmetric_for_a_down_taker() {
        let win = window(0);
        let mut view = RestingView::new();
        // Our resting BUY Up @ 0.40 = a mirrored ask on Down @ 0.60.
        place(&mut view, win, Outcome::Up, dec!(0.40), dec!(30), 1);
        let asks = [ask(dec!(0.60), dec!(30))];
        let out = filter_asks(Outcome::Down, &asks, win, Some(&view));
        assert!(out.is_empty());
    }
}
