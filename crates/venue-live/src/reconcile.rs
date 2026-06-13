//! The interim REST open-orders-poll reconcile driver.
//!
//! Polls [`ClobPort::open_orders`] on an interval, diffs against the order store
//! ([`OrderStore::reconcile`]), and publishes the resulting [`OrderUpdate`]s
//! onto the venue event channel. This is the stand-in fill/cancel feedback until
//! the real-time `/ws/user` push (a later task) replaces it behind the same
//! channel. The loop exits when the event receiver is dropped.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;
use venue_api::VenueEvent;

use crate::port::ClobPort;
use crate::store::OrderStore;

/// Runs the poll/diff/publish loop until the event channel's receiver is
/// dropped. `interval` is the poll cadence.
pub async fn reconcile_loop<P: ClobPort>(
    backend: Arc<P>,
    store: Arc<Mutex<OrderStore>>,
    events: mpsc::Sender<VenueEvent>,
    interval: Duration,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        if events.is_closed() {
            break;
        }
        match backend.open_orders().await {
            Ok(polled) => {
                let now = timeutil::wall_now();
                // Lock is released before any await (no lock held across .await).
                let updates = {
                    let mut store = store.lock().unwrap_or_else(|e| e.into_inner());
                    store.reconcile(&polled, now)
                };
                for update in updates {
                    if events
                        .try_send(VenueEvent::Order(Arc::new(update)))
                        .is_err()
                    {
                        tracing::warn!(
                            target: "venue::live",
                            "order-event channel full or closed; reconcile backlog"
                        );
                        break;
                    }
                }
            }
            Err(error) => {
                tracing::warn!(target: "venue::live", %error, "open-orders poll failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::{
        Asset, Outcome, RoundDir, Series, Side, Size, TickSize, TimestampMs, TokenId,
        WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    use super::*;
    use crate::fake::FakeClobPort;
    use crate::port::RawOpenOrder;
    use crate::store::TrackedOrder;
    use core_types::{OrderId, OrderState, Price};

    fn tracked() -> TrackedOrder {
        TrackedOrder {
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(1_000_000),
            },
            token_id: TokenId::new("123").unwrap(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: Price::quantize(dec!(0.50), TickSize::T001, RoundDir::Down).unwrap(),
            original_size: Size::new(dec!(10)).unwrap(),
            last_state: OrderState::Open,
            last_filled: Size::ZERO,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn loop_emits_updates_then_exits_when_receiver_dropped() {
        let backend = Arc::new(FakeClobPort::new());
        // First poll: order partially filled. Second poll: gone (→ canceled,
        // partial). Subsequent polls: empty.
        backend.push_open_orders(vec![RawOpenOrder {
            order_id: OrderId::new("o1").unwrap(),
            status: "LIVE".to_owned(),
            original_size: Size::new(dec!(10)).unwrap(),
            size_matched: Size::new(dec!(4)).unwrap(),
        }]);
        backend.push_open_orders(vec![]); // vanished

        let store = Arc::new(Mutex::new(OrderStore::new()));
        store
            .lock()
            .unwrap()
            .track(OrderId::new("o1").unwrap(), tracked());

        let (tx, mut rx) = mpsc::channel(16);
        let handle = tokio::spawn(reconcile_loop(
            Arc::clone(&backend),
            Arc::clone(&store),
            tx,
            Duration::from_millis(100),
        ));

        // First tick → PartiallyFilled (4/10).
        let first = rx.recv().await.expect("first update");
        let VenueEvent::Order(u) = first else {
            panic!("expected order update")
        };
        assert_eq!(u.state, OrderState::PartiallyFilled);
        assert_eq!(u.filled_size, Size::new(dec!(4)).unwrap());

        // Second tick → vanished with partial fill → Canceled.
        let second = rx.recv().await.expect("second update");
        let VenueEvent::Order(u) = second else {
            panic!("expected order update")
        };
        assert_eq!(u.state, OrderState::Canceled);

        // Dropping the receiver stops the loop.
        drop(rx);
        // Let the loop observe the closed channel on its next tick.
        tokio::time::advance(Duration::from_millis(200)).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
    }
}
