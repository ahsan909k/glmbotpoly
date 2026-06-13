//! The in-memory order store and the interim REST-poll reconcile diff.
//!
//! On each successful `place`, [`LiveVenue`](crate::LiveVenue) records the
//! order's window/token context here. [`OrderStore::reconcile`] then diffs a
//! fresh open-orders poll against that context to emit
//! [`OrderUpdate`]s — a fill-and-cancel feedback signal until the real-time
//! `/ws/user` push (a later task) replaces it. Pure and clock-injected, so the
//! diff is fully unit-testable.

use std::collections::HashMap;

use core_types::{
    OrderId, OrderState, OrderUpdate, Outcome, Price, Side, Size, TimestampMs, TokenId, WindowId,
};

use crate::port::RawOpenOrder;

/// The context [`LiveVenue`](crate::LiveVenue) keeps for one order it placed, so
/// a polled open-order (which keys by id/token) can be turned back into a fully
/// populated [`OrderUpdate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedOrder {
    /// Window the order trades.
    pub window: WindowId,
    /// Outcome token.
    pub token_id: TokenId,
    /// Which outcome that token is.
    pub outcome: Outcome,
    /// Order side.
    pub side: Side,
    /// Limit price.
    pub price: Price,
    /// Original size in shares.
    pub original_size: Size,
    /// Last state we emitted for this order.
    pub last_state: OrderState,
    /// Last cumulative filled size we emitted.
    pub last_filled: Size,
}

/// Tracks the orders this adapter placed, keyed by venue order id.
#[derive(Debug, Default)]
pub struct OrderStore {
    orders: HashMap<OrderId, TrackedOrder>,
}

impl OrderStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Begins (or replaces) tracking an order.
    pub fn track(&mut self, order_id: OrderId, tracked: TrackedOrder) {
        self.orders.insert(order_id, tracked);
    }

    /// Marks an order terminal (used by the cancel paths so the reconcile diff
    /// does not later re-report a known cancellation). No-op for unknown ids.
    pub fn mark_terminal(&mut self, order_id: &OrderId, state: OrderState) {
        if let Some(t) = self.orders.get_mut(order_id) {
            t.last_state = state;
        }
    }

    /// Number of tracked orders (for diagnostics/tests).
    #[must_use]
    pub fn len(&self) -> usize {
        self.orders.len()
    }

    /// True when nothing is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Diffs a fresh open-orders poll against tracked state and returns the
    /// [`OrderUpdate`]s to emit, updating tracked state in place. `now` stamps
    /// `ts_local`. Polled orders not in the store are ignored (we cannot map an
    /// unknown order to a window — adopting them is a job for the WS-reconcile
    /// task); the caller logs them.
    #[must_use]
    pub fn reconcile(&mut self, polled: &[RawOpenOrder], now: TimestampMs) -> Vec<OrderUpdate> {
        let polled_by_id: HashMap<&OrderId, &RawOpenOrder> =
            polled.iter().map(|p| (&p.order_id, p)).collect();

        let mut updates = Vec::new();
        for (id, tracked) in &mut self.orders {
            if tracked.last_state.is_terminal() {
                continue;
            }
            let (new_state, new_filled) = match polled_by_id.get(id) {
                Some(po) => (
                    open_status_to_state(&po.status, po.size_matched, tracked.original_size),
                    po.size_matched,
                ),
                None => {
                    // Left the book entirely → terminal. Fully filled if our last
                    // known fill covers the original, else cancelled.
                    let filled = tracked.last_filled;
                    let state = if !tracked.original_size.is_zero()
                        && filled.as_decimal() >= tracked.original_size.as_decimal()
                    {
                        OrderState::Filled
                    } else {
                        OrderState::Canceled
                    };
                    (state, filled)
                }
            };

            let state_changed = new_state != tracked.last_state;
            let fill_changed = new_filled != tracked.last_filled;
            if !state_changed && !fill_changed {
                continue;
            }
            if state_changed && !tracked.last_state.can_transition_to(new_state) {
                // An illegal transition from the wire indicates a missed message;
                // the caller logs and a future full reconcile resyncs.
                tracing::warn!(
                    target: "venue::live",
                    order_id = %id,
                    from = ?tracked.last_state,
                    to = ?new_state,
                    "skipping illegal order-state transition from poll"
                );
                continue;
            }

            updates.push(OrderUpdate {
                order_id: id.clone(),
                window: tracked.window,
                token_id: tracked.token_id.clone(),
                side: tracked.side,
                state: new_state,
                price: tracked.price,
                original_size: tracked.original_size,
                filled_size: new_filled,
                reject_reason: None,
                ts_venue: None,
                ts_local: now,
            });
            tracked.last_state = new_state;
            tracked.last_filled = new_filled;
        }
        updates
    }
}

/// Maps a venue open-order status string + fill progress to an [`OrderState`].
fn open_status_to_state(status: &str, filled: Size, original: Size) -> OrderState {
    match status.to_ascii_lowercase().as_str() {
        "canceled" | "cancelled" => OrderState::Canceled,
        "matched" => {
            if !original.is_zero() && filled.as_decimal() >= original.as_decimal() {
                OrderState::Filled
            } else {
                OrderState::PartiallyFilled
            }
        }
        // live / unmatched / delayed / anything else: still working.
        _ => {
            if filled.is_zero() {
                OrderState::Open
            } else {
                OrderState::PartiallyFilled
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::{Asset, RoundDir, Series, TickSize, WindowDuration};
    use rust_decimal::dec;

    use super::*;

    fn oid(s: &str) -> OrderId {
        OrderId::new(s).unwrap()
    }
    fn size(d: rust_decimal::Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn price() -> Price {
        Price::quantize(dec!(0.50), TickSize::T001, RoundDir::Down).unwrap()
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_000_000),
        }
    }
    fn tracked(state: OrderState, filled: rust_decimal::Decimal) -> TrackedOrder {
        TrackedOrder {
            window: window(),
            token_id: TokenId::new("123").unwrap(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: price(),
            original_size: size(dec!(10)),
            last_state: state,
            last_filled: size(filled),
        }
    }
    fn open(id: &str, status: &str, matched: rust_decimal::Decimal) -> RawOpenOrder {
        RawOpenOrder {
            order_id: oid(id),
            status: status.to_owned(),
            original_size: size(dec!(10)),
            size_matched: size(matched),
        }
    }
    fn now() -> TimestampMs {
        TimestampMs::from_millis(2_000_000)
    }

    #[test]
    fn open_then_partial_fill_emits_update() {
        let mut store = OrderStore::new();
        store.track(oid("o1"), tracked(OrderState::Open, dec!(0)));
        // Poll shows 3 of 10 matched.
        let ups = store.reconcile(&[open("o1", "LIVE", dec!(3))], now());
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].state, OrderState::PartiallyFilled);
        assert_eq!(ups[0].filled_size, size(dec!(3)));
        // No change on the next identical poll.
        assert!(
            store
                .reconcile(&[open("o1", "LIVE", dec!(3))], now())
                .is_empty()
        );
        // More fill → another update.
        let ups = store.reconcile(&[open("o1", "MATCHED", dec!(10))], now());
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].state, OrderState::Filled);
    }

    #[test]
    fn vanished_order_with_full_fill_is_filled_else_canceled() {
        // Filled: last fill covered the original, then it left the book.
        let mut store = OrderStore::new();
        store.track(oid("o1"), tracked(OrderState::PartiallyFilled, dec!(10)));
        let ups = store.reconcile(&[], now());
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].state, OrderState::Filled);

        // Cancelled: it left the book with only a partial fill.
        let mut store = OrderStore::new();
        store.track(oid("o2"), tracked(OrderState::PartiallyFilled, dec!(4)));
        let ups = store.reconcile(&[], now());
        assert_eq!(ups[0].state, OrderState::Canceled);
        assert_eq!(ups[0].filled_size, size(dec!(4)));
    }

    #[test]
    fn terminal_orders_are_not_reprocessed() {
        let mut store = OrderStore::new();
        store.track(oid("o1"), tracked(OrderState::Filled, dec!(10)));
        assert!(store.reconcile(&[], now()).is_empty());
        assert!(
            store
                .reconcile(&[open("o1", "LIVE", dec!(0))], now())
                .is_empty()
        );
    }

    #[test]
    fn mark_terminal_suppresses_later_vanish_event() {
        let mut store = OrderStore::new();
        store.track(oid("o1"), tracked(OrderState::Open, dec!(0)));
        store.mark_terminal(&oid("o1"), OrderState::Canceled);
        assert!(store.reconcile(&[], now()).is_empty());
    }

    #[test]
    fn pendingnew_transitions_to_open() {
        let mut store = OrderStore::new();
        store.track(oid("o1"), tracked(OrderState::PendingNew, dec!(0)));
        let ups = store.reconcile(&[open("o1", "LIVE", dec!(0))], now());
        assert_eq!(ups.len(), 1);
        assert_eq!(ups[0].state, OrderState::Open);
    }
}
