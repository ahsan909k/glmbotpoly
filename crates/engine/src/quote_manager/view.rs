//! [`RestingView`]: the manager's own view of the orders it has resting at the
//! venue, for the active window.
//!
//! The `engine` crate cannot depend on `venue-live`'s canonical `OrderStore`
//! (§4 inward-arrows), so the manager keeps its own view. It is *simpler* than
//! `OrderStore` because the manager placed every order and knows its full
//! attribution at placement time: `record_pending` is fed straight from the
//! `Accepted` of a place, and subsequent lifecycle changes fold in from the
//! venue's [`OrderUpdate`](core_types::OrderUpdate) stream.
//!
//! Pure and sans-IO: mutated by recording placements and folding venue events;
//! it never reads a clock or a venue.
//!
//! ## Strict-overlap rule (split-cycle)
//!
//! A cancelled order is marked [`OrderState::PendingCancel`] and **kept in its
//! slot** — blocking any placement into that slot — until its terminal
//! `Canceled` update folds in and frees the slot. Only then will a later converge
//! place the replacement, so the old and the new never rest on the same side at
//! once.

use std::collections::HashMap;

use core_types::{OrderState, OrderUpdate, Outcome, Price, Size, WindowId};

use super::client_id::ClientId;
use core_types::OrderId;

/// One order the manager placed (and has not yet seen terminalize), with
/// everything the planner needs to diff it against the desired set and to cancel
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestingOrder {
    /// Venue order id (the correlation key, from the place `Accepted`).
    pub order_id: OrderId,
    /// Our attribution, exactly what we sent in the `client_id`.
    pub client_id: ClientId,
    /// Window the order trades.
    pub window: WindowId,
    /// Outcome ladder (`client_id.outcome`).
    pub outcome: Outcome,
    /// Ladder level (`client_id.level`).
    pub level: u32,
    /// The on-grid price we placed at (post-normalize). What the diff compares.
    pub price: Price,
    /// Original size placed.
    pub original_size: Size,
    /// Cumulative filled, folded from `OrderUpdate.filled_size` (informational —
    /// the diff keys on price, not remaining size).
    pub filled_size: Size,
    /// Latest known lifecycle state.
    pub state: OrderState,
}

impl RestingOrder {
    /// True while the order occupies (or is about to occupy) a slot on the book.
    /// Terminal orders are dropped from the view, so a held [`RestingOrder`] is
    /// always live by construction.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.state == OrderState::PendingNew || self.state.is_working()
    }

    /// True once a cancel has been issued for this order (it still holds its slot
    /// until the terminal `Canceled` arrives).
    #[must_use]
    pub fn is_pending_cancel(&self) -> bool {
        self.state == OrderState::PendingCancel
    }
}

/// The manager's view of its resting orders for the active window.
///
/// - `by_id` is the authority (folds order updates).
/// - `slot` maps `(outcome, level)` to the id we consider the live order for that
///   slot. A slot is "occupied" (blocks placement) whenever it points at a live
///   order — including a [`OrderState::PendingCancel`] one that is still draining.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestingView {
    by_id: HashMap<OrderId, RestingOrder>,
    slot: HashMap<(Outcome, u32), OrderId>,
}

impl RestingView {
    /// An empty view.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a freshly-placed order (called with the place `Accepted`). The
    /// `(outcome, level)` come from the `client_id`. Inserts it as
    /// [`OrderState::PendingNew`] and points the slot at it.
    pub fn record_pending(
        &mut self,
        order_id: OrderId,
        client_id: ClientId,
        window: WindowId,
        price: Price,
        size: Size,
    ) {
        let outcome = client_id.outcome;
        let level = client_id.level;
        self.slot.insert((outcome, level), order_id.clone());
        self.by_id.insert(
            order_id.clone(),
            RestingOrder {
                order_id,
                client_id,
                window,
                outcome,
                level,
                price,
                original_size: size,
                filled_size: Size::ZERO,
                state: OrderState::PendingNew,
            },
        );
    }

    /// Marks an order [`OrderState::PendingCancel`] the moment a cancel is issued,
    /// so a racing trigger does not double-cancel it. Best-effort: honours the
    /// transition table (a still-`PendingNew` order cannot move to
    /// `PendingCancel`, but it keeps its slot regardless and terminalizes via its
    /// eventual `Canceled` update).
    pub fn mark_pending_cancel(&mut self, id: &OrderId) {
        if let Some(order) = self.by_id.get_mut(id)
            && order.state.can_transition_to(OrderState::PendingCancel)
        {
            order.state = OrderState::PendingCancel;
        }
    }

    /// Folds one venue order update (keyed on `order_id`). Updates the cumulative
    /// fill (monotone) and the lifecycle state (honouring the transition table);
    /// a terminal state drops the order from `by_id` and frees its slot.
    pub fn apply_order_update(&mut self, u: &OrderUpdate) {
        let Some(order) = self.by_id.get_mut(&u.order_id) else {
            return; // not ours / already dropped — ignore.
        };
        if u.filled_size.as_decimal() > order.filled_size.as_decimal() {
            order.filled_size = u.filled_size;
        }
        if order.state != u.state && order.state.can_transition_to(u.state) {
            order.state = u.state;
        }
        let terminal = order.state.is_terminal();
        let slot_key = (order.outcome, order.level);
        if terminal {
            self.by_id.remove(&u.order_id);
            if self.slot.get(&slot_key) == Some(&u.order_id) {
                self.slot.remove(&slot_key);
            }
        }
    }

    /// The live order currently occupying `(outcome, level)`, if any.
    #[must_use]
    pub fn resting_at(&self, outcome: Outcome, level: u32) -> Option<&RestingOrder> {
        let id = self.slot.get(&(outcome, level))?;
        self.by_id.get(id)
    }

    /// Every live order (all held orders are live by construction).
    pub fn live_orders(&self) -> impl Iterator<Item = &RestingOrder> {
        self.by_id.values()
    }

    /// The ids of every live order.
    #[must_use]
    pub fn live_ids(&self) -> Vec<OrderId> {
        self.by_id.keys().cloned().collect()
    }

    /// True when no orders are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Number of live orders held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// True when at least one held order is not already cancelling — i.e. there
    /// is something a fresh cancel could still act on. Once every order is
    /// [`OrderState::PendingCancel`], a cancel-all / cancel-market would be a
    /// redundant no-op, so the planner stops re-issuing.
    #[must_use]
    pub fn has_live_cancelable(&self) -> bool {
        self.by_id.values().any(|o| !o.is_pending_cancel())
    }

    /// Drops every order for `window` (called on window Closed/Resolved — the
    /// venue terminalizes them on close, and this is the hard reset).
    pub fn drop_window(&mut self, window: WindowId) {
        self.by_id.retain(|_, o| o.window != window);
        let by_id = &self.by_id;
        self.slot.retain(|_, id| by_id.contains_key(id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{Asset, RoundDir, Series, TickSize, TimestampMs, TokenId, WindowDuration};
    use rust_decimal::dec;

    const OPEN_MS: i64 = 1_781_000_000_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }
    fn px(d: rust_decimal::Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).expect("price")
    }
    fn sz(d: rust_decimal::Decimal) -> Size {
        Size::new(d).expect("size")
    }
    fn oid(s: &str) -> OrderId {
        OrderId::new(s).expect("oid")
    }
    fn cid(outcome: Outcome, level: u32, seq: u64) -> ClientId {
        ClientId {
            open_ms: OPEN_MS,
            outcome,
            level,
            seq,
        }
    }
    fn update(id: &OrderId, state: OrderState, filled: rust_decimal::Decimal) -> OrderUpdate {
        OrderUpdate {
            order_id: id.clone(),
            window: window(),
            token_id: TokenId::new("111").expect("token"),
            side: core_types::Side::Buy,
            state,
            price: px(dec!(0.49)),
            original_size: sz(dec!(10)),
            filled_size: sz(filled),
            reject_reason: None,
            ts_venue: None,
            ts_local: TimestampMs::from_millis(OPEN_MS),
        }
    }

    #[test]
    fn record_sets_slot_and_state() {
        let mut v = RestingView::new();
        v.record_pending(
            oid("o1"),
            cid(Outcome::Up, 0, 0),
            window(),
            px(dec!(0.49)),
            sz(dec!(10)),
        );
        let order = v.resting_at(Outcome::Up, 0).expect("present");
        assert_eq!(order.order_id, oid("o1"));
        assert_eq!(order.state, OrderState::PendingNew);
        assert_eq!(order.price, px(dec!(0.49)));
        assert_eq!(v.len(), 1);
        assert!(v.has_live_cancelable());
    }

    #[test]
    fn order_update_advances_then_terminal_drops_and_frees_slot() {
        let mut v = RestingView::new();
        v.record_pending(
            oid("o1"),
            cid(Outcome::Up, 0, 0),
            window(),
            px(dec!(0.49)),
            sz(dec!(10)),
        );
        // PendingNew -> Open (allowed).
        v.apply_order_update(&update(&oid("o1"), OrderState::Open, dec!(0)));
        assert_eq!(
            v.resting_at(Outcome::Up, 0).expect("present").state,
            OrderState::Open
        );
        // Open -> PartiallyFilled, cumulative fill monotone.
        v.apply_order_update(&update(&oid("o1"), OrderState::PartiallyFilled, dec!(4)));
        assert_eq!(
            v.resting_at(Outcome::Up, 0).expect("present").filled_size,
            sz(dec!(4))
        );
        // Terminal Canceled drops it and frees the slot.
        v.apply_order_update(&update(&oid("o1"), OrderState::Canceled, dec!(4)));
        assert!(v.resting_at(Outcome::Up, 0).is_none());
        assert!(v.is_empty());
    }

    #[test]
    fn pending_cancel_blocks_slot_until_terminal_frees_it() {
        let mut v = RestingView::new();
        v.record_pending(
            oid("o1"),
            cid(Outcome::Up, 0, 0),
            window(),
            px(dec!(0.49)),
            sz(dec!(10)),
        );
        v.apply_order_update(&update(&oid("o1"), OrderState::Open, dec!(0)));
        v.mark_pending_cancel(&oid("o1"));
        let order = v
            .resting_at(Outcome::Up, 0)
            .expect("still occupying the slot");
        assert!(order.is_pending_cancel());
        assert!(
            !v.has_live_cancelable(),
            "nothing fresh to cancel once pending"
        );
        // The terminal Canceled finally frees the slot.
        v.apply_order_update(&update(&oid("o1"), OrderState::Canceled, dec!(0)));
        assert!(v.resting_at(Outcome::Up, 0).is_none());
    }

    #[test]
    fn replacement_uses_a_new_id_in_the_same_slot_after_free() {
        let mut v = RestingView::new();
        v.record_pending(
            oid("o1"),
            cid(Outcome::Up, 0, 0),
            window(),
            px(dec!(0.49)),
            sz(dec!(10)),
        );
        v.apply_order_update(&update(&oid("o1"), OrderState::Open, dec!(0)));
        v.mark_pending_cancel(&oid("o1"));
        v.apply_order_update(&update(&oid("o1"), OrderState::Canceled, dec!(0)));
        // Slot free → place the replacement (new seq → new id).
        v.record_pending(
            oid("o2"),
            cid(Outcome::Up, 0, 1),
            window(),
            px(dec!(0.47)),
            sz(dec!(10)),
        );
        let order = v.resting_at(Outcome::Up, 0).expect("replacement present");
        assert_eq!(order.order_id, oid("o2"));
        assert_eq!(order.price, px(dec!(0.47)));
        assert_eq!(order.client_id.seq, 1);
    }

    #[test]
    fn unknown_order_update_is_ignored() {
        let mut v = RestingView::new();
        v.apply_order_update(&update(&oid("ghost"), OrderState::Canceled, dec!(0)));
        assert!(v.is_empty());
    }

    #[test]
    fn drop_window_clears_everything() {
        let mut v = RestingView::new();
        v.record_pending(
            oid("o1"),
            cid(Outcome::Up, 0, 0),
            window(),
            px(dec!(0.49)),
            sz(dec!(10)),
        );
        v.record_pending(
            oid("o2"),
            cid(Outcome::Down, 0, 1),
            window(),
            px(dec!(0.49)),
            sz(dec!(10)),
        );
        v.drop_window(window());
        assert!(v.is_empty());
        assert!(v.resting_at(Outcome::Up, 0).is_none());
        assert!(v.resting_at(Outcome::Down, 0).is_none());
    }
}
