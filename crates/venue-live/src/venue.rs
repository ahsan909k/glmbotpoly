//! [`LiveVenue`]: the [`VenuePort`]/[`VenueEvents`] implementation, generic over
//! the [`ClobPort`] network backend.
//!
//! All venue-agnostic logic lives here — translating orders via
//! [`convert::build`], chunking batches to the venue max, mapping raw acks and
//! cancel reports into the public `venue_api` types, and tracking placed orders
//! for the reconcile poll. The SDK never appears; tests drive it through
//! [`FakeClobPort`](crate::FakeClobPort).

use std::sync::{Arc, Mutex};

use core_types::{ConditionId, NewOrder, OrderId, OrderQty, OrderState, Size};
use tokio::sync::mpsc;
use venue_api::{
    Accepted, BatchPlaced, CancelReport, NotCanceled, PlaceRejection, RejectReason, VenueError,
    VenueEvent, VenueEvents, VenuePort, Wallet,
};

use crate::convert::{self, BuiltOrder};
use crate::error::classify_reject;
use crate::params::LiveParams;
use crate::port::{ClobPort, RawAck, RawCancel};
use crate::store::{OrderStore, TrackedOrder};

/// Maximum orders per batch request (CLAUDE.md §7). `place_batch` chunks at this
/// size; the venue itself also enforces a max and is mapped to
/// [`VenueError::BatchTooLarge`](venue_api::VenueError::BatchTooLarge).
pub const MAX_BATCH: usize = 15;

/// The live execution adapter, generic over its network backend `P`.
///
/// Construct via `connect` (gated, network-capable) or `dry_run` (signs but
/// never submits). The `with_backend` constructor is crate-internal so external
/// code can only obtain a network-capable adapter through the gated path —
/// preserving the §11 safety invariant.
pub struct LiveVenue<P: ClobPort> {
    backend: Arc<P>,
    params: LiveParams,
    store: Arc<Mutex<OrderStore>>,
    event_tx: mpsc::Sender<VenueEvent>,
    event_rx: Option<mpsc::Receiver<VenueEvent>>,
}

impl<P: ClobPort> LiveVenue<P> {
    /// Builds an adapter over an arbitrary backend. Crate-internal: the gated
    /// `connect` / `dry_run` constructors call this after their checks.
    pub(crate) fn with_backend(backend: P, params: LiveParams) -> Self {
        let (event_tx, event_rx) = mpsc::channel(params.event_channel_capacity.max(1));
        Self {
            backend: Arc::new(backend),
            params,
            store: Arc::new(Mutex::new(OrderStore::new())),
            event_tx,
            event_rx: Some(event_rx),
        }
    }

    /// The handles the reconcile loop needs: a backend clone, the shared store,
    /// and an event sender. The caller spawns [`crate::reconcile_loop`] with
    /// these.
    #[must_use]
    pub(crate) fn reconcile_handles(
        &self,
    ) -> (Arc<P>, Arc<Mutex<OrderStore>>, mpsc::Sender<VenueEvent>) {
        (
            Arc::clone(&self.backend),
            Arc::clone(&self.store),
            self.event_tx.clone(),
        )
    }

    fn lock_store(&self) -> std::sync::MutexGuard<'_, OrderStore> {
        self.store.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Records a successfully-placed resting order so the reconcile poll can
    /// emit its fills/cancels. Marketable orders never rest, so they are not
    /// tracked (their feedback arrives via the WS task / immediate ack).
    fn track_placed(&self, order: &NewOrder, accepted: &Accepted) {
        if order.tif.is_marketable() {
            return;
        }
        let OrderQty::Shares(original_size) = order.qty else {
            return;
        };
        self.lock_store().track(
            accepted.order_id.clone(),
            TrackedOrder {
                window: order.window,
                token_id: order.token_id.clone(),
                outcome: order.outcome,
                side: order.side,
                price: order.price,
                original_size,
                last_state: accepted.state,
                last_filled: Size::ZERO,
            },
        );
    }

    fn note_canceled(&self, report: &CancelReport) {
        let mut store = self.lock_store();
        for id in &report.canceled {
            store.mark_terminal(id, OrderState::Canceled);
        }
    }
}

impl<P: ClobPort> VenuePort for LiveVenue<P> {
    async fn place(&self, order: &NewOrder) -> Result<Accepted, VenueError> {
        let now = timeutil::wall_now();
        let built = convert::build(order, now, &self.params).map_err(VenueError::from)?;
        let ack = self
            .backend
            .submit(&built)
            .await
            .map_err(VenueError::from)?;
        let accepted = accept_single(ack)?;
        self.track_placed(order, &accepted);
        Ok(accepted)
    }

    async fn place_batch(&self, orders: &[NewOrder]) -> Result<BatchPlaced, VenueError> {
        let now = timeutil::wall_now();
        // A build error becomes a per-order rejection, never a whole-request
        // failure — one malformed order must not sink the rest of the batch.
        let built: Vec<Result<BuiltOrder, PlaceRejection>> = orders
            .iter()
            .map(|o| {
                convert::build(o, now, &self.params).map_err(|e| PlaceRejection {
                    client_id: o.client_id.clone(),
                    reason: RejectReason::Other(e.to_string()),
                    raw: e.to_string(),
                })
            })
            .collect();

        let to_submit: Vec<BuiltOrder> = built
            .iter()
            .filter_map(|r| r.as_ref().ok().cloned())
            .collect();

        let mut acks: Vec<RawAck> = Vec::with_capacity(to_submit.len());
        for chunk in to_submit.chunks(MAX_BATCH) {
            let chunk_acks = self
                .backend
                .submit_batch(chunk)
                .await
                .map_err(VenueError::from)?;
            if chunk_acks.len() != chunk.len() {
                return Err(VenueError::Transport(format!(
                    "venue returned {} acks for a batch of {}",
                    chunk_acks.len(),
                    chunk.len()
                )));
            }
            acks.extend(chunk_acks);
        }

        let mut ack_iter = acks.into_iter();
        let mut results = Vec::with_capacity(orders.len());
        for (slot, order) in built.into_iter().zip(orders.iter()) {
            match slot {
                Err(rejection) => results.push(Err(rejection)),
                Ok(_) => {
                    let ack = ack_iter.next().ok_or_else(|| {
                        VenueError::Transport("batch ack count mismatch".to_owned())
                    })?;
                    let result = accept_batch(ack);
                    if let Ok(accepted) = &result {
                        self.track_placed(order, accepted);
                    }
                    results.push(result);
                }
            }
        }
        Ok(BatchPlaced { results })
    }

    async fn cancel(&self, id: &OrderId) -> Result<CancelReport, VenueError> {
        let raw = self
            .backend
            .cancel_one(id)
            .await
            .map_err(VenueError::from)?;
        let report = cancel_report(raw);
        self.note_canceled(&report);
        Ok(report)
    }

    async fn cancel_market(&self, market: &ConditionId) -> Result<CancelReport, VenueError> {
        let raw = self
            .backend
            .cancel_market(market)
            .await
            .map_err(VenueError::from)?;
        let report = cancel_report(raw);
        self.note_canceled(&report);
        Ok(report)
    }

    async fn cancel_all(&self) -> Result<CancelReport, VenueError> {
        let raw = self.backend.cancel_all().await.map_err(VenueError::from)?;
        let report = cancel_report(raw);
        self.note_canceled(&report);
        Ok(report)
    }

    async fn balances(&self) -> Result<Wallet, VenueError> {
        self.backend.balances().await.map_err(VenueError::from)
    }
}

impl<P: ClobPort> VenueEvents for LiveVenue<P> {
    fn take_event_rx(&mut self) -> Option<mpsc::Receiver<VenueEvent>> {
        self.event_rx.take()
    }
}

/// Maps a single-order ack into an [`Accepted`] or a typed rejection.
fn accept_single(ack: RawAck) -> Result<Accepted, VenueError> {
    if !ack.success {
        return Err(VenueError::Rejected(classify_reject(
            ack.error.as_deref().unwrap_or_default(),
        )));
    }
    let order_id = ack
        .order_id
        .ok_or_else(|| VenueError::Transport("venue accepted an order without an id".to_owned()))?;
    Ok(Accepted {
        client_id: ack.client_id,
        order_id,
        state: status_to_state(&ack.status),
    })
}

/// Maps one ack inside a batch into a [`Result<Accepted, PlaceRejection>`].
fn accept_batch(ack: RawAck) -> Result<Accepted, PlaceRejection> {
    if !ack.success {
        let raw = ack.error.unwrap_or_default();
        return Err(PlaceRejection {
            client_id: ack.client_id,
            reason: classify_reject(&raw),
            raw,
        });
    }
    match ack.order_id {
        Some(order_id) => Ok(Accepted {
            client_id: ack.client_id,
            order_id,
            state: status_to_state(&ack.status),
        }),
        None => Err(PlaceRejection {
            client_id: ack.client_id,
            reason: RejectReason::Other("venue accepted an order without an id".to_owned()),
            raw: ack.status,
        }),
    }
}

/// Maps a venue order-status string to an [`OrderState`] for an ack.
fn status_to_state(status: &str) -> OrderState {
    match status.to_ascii_lowercase().as_str() {
        "matched" => OrderState::Filled,
        "canceled" | "cancelled" => OrderState::Canceled,
        "delayed" => OrderState::PendingNew,
        // live / unmatched / anything else: resting and working.
        _ => OrderState::Open,
    }
}

/// Maps a raw cancel response into the venue-agnostic [`CancelReport`].
fn cancel_report(raw: RawCancel) -> CancelReport {
    let not_canceled = raw
        .not_canceled
        .into_iter()
        .map(|(order_id, reason)| NotCanceled {
            order_id,
            reason: classify_reject(&reason),
            raw: reason,
        })
        .collect();
    CancelReport {
        canceled: raw.canceled,
        not_canceled,
    }
}

#[cfg(test)]
mod tests {
    use core_types::{
        Asset, Dollars, Outcome, Price, RoundDir, Series, Side, TickSize, TimeInForce, TimestampMs,
        TokenId, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    use super::*;
    use crate::convert::OrderClass;
    use crate::fake::{CancelCall, FakeClobPort};
    use crate::params::SigType;
    use crate::port::RawAck;

    fn params() -> LiveParams {
        LiveParams {
            sig_type: SigType::DepositWallet,
            funder: Some(format!("0x{}", "ab".repeat(20))),
            ..LiveParams::default()
        }
    }

    fn venue() -> (LiveVenue<FakeClobPort>, Arc<FakeClobPort>) {
        let backend = FakeClobPort::new();
        let venue = LiveVenue::with_backend(backend, params());
        let handle = Arc::clone(&venue.backend);
        (venue, handle)
    }

    fn price(d: rust_decimal::Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
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

    fn order(side: Side, qty: OrderQty, tif: TimeInForce, client: &str) -> NewOrder {
        NewOrder {
            client_id: Some(client.to_owned()),
            window: window(),
            token_id: TokenId::new("123").unwrap(),
            outcome: Outcome::Up,
            side,
            price: price(dec!(0.40)),
            qty,
            tif,
        }
    }

    fn shares(d: rust_decimal::Decimal) -> OrderQty {
        OrderQty::Shares(Size::new(d).unwrap())
    }

    #[tokio::test]
    async fn place_gtc_constructs_post_only_limit_and_tracks() {
        let (venue, fake) = venue();
        let o = order(
            Side::Buy,
            shares(dec!(10)),
            TimeInForce::Gtc { post_only: true },
            "c0",
        );
        let accepted = venue.place(&o).await.unwrap();
        assert_eq!(accepted.client_id.as_deref(), Some("c0"));
        assert_eq!(accepted.state, OrderState::Open);

        let submitted = fake.submitted();
        assert_eq!(submitted.len(), 1);
        let b = &submitted[0];
        assert_eq!(b.token_id.as_str(), "123");
        assert_eq!(b.side, Side::Buy);
        assert_eq!(b.price, price(dec!(0.40)));
        assert_eq!(b.amount, shares(dec!(10)));
        assert_eq!(
            b.class,
            OrderClass::Limit {
                post_only: true,
                expiration: None
            }
        );
        assert_eq!(b.sig_type, SigType::DepositWallet);
        assert_eq!(b.funder, params().funder);
        // Resting orders are tracked for the reconcile poll.
        assert_eq!(venue.lock_store().len(), 1);
    }

    #[tokio::test]
    async fn place_gtd_floors_expiration_inside_the_adapter() {
        let (venue, fake) = venue();
        // Desired expiry in the past → must be floored to ≥ now + 60s.
        let o = order(
            Side::Sell,
            shares(dec!(7)),
            TimeInForce::Gtd {
                expires_at: TimestampMs::from_millis(0),
                post_only: true,
            },
            "c1",
        );
        venue.place(&o).await.unwrap();
        let b = &fake.submitted()[0];
        let OrderClass::Limit {
            expiration: Some(exp),
            ..
        } = b.class
        else {
            panic!("expected GTD limit");
        };
        let now = timeutil::wall_now();
        assert!(
            exp.as_millis() >= now.as_millis() + 60_000,
            "GTD expiration must be floored to at least now + 60s"
        );
    }

    #[tokio::test]
    async fn place_marketable_buy_is_dollars_sell_is_shares() {
        let (venue, fake) = venue();
        venue
            .place(&order(
                Side::Buy,
                OrderQty::Notional(Dollars::new(dec!(25))),
                TimeInForce::Fak,
                "buy",
            ))
            .await
            .unwrap();
        venue
            .place(&order(
                Side::Sell,
                shares(dec!(10)),
                TimeInForce::Fok,
                "sell",
            ))
            .await
            .unwrap();

        let s = fake.submitted();
        assert_eq!(s[0].class, OrderClass::Marketable { fok: false });
        assert_eq!(s[0].amount, OrderQty::Notional(Dollars::new(dec!(25))));
        assert_eq!(s[1].class, OrderClass::Marketable { fok: true });
        assert_eq!(s[1].amount, shares(dec!(10)));
        // Marketable orders never rest → not tracked.
        assert_eq!(venue.lock_store().len(), 0);
    }

    #[tokio::test]
    async fn place_batch_chunks_at_fifteen_and_preserves_order() {
        let (venue, fake) = venue();
        let orders: Vec<NewOrder> = (0..16)
            .map(|i| {
                order(
                    Side::Buy,
                    shares(dec!(10)),
                    TimeInForce::Gtc { post_only: true },
                    &format!("c{i}"),
                )
            })
            .collect();
        let batch = venue.place_batch(&orders).await.unwrap();
        assert_eq!(batch.results.len(), 16);
        assert!(batch.results.iter().all(Result::is_ok));
        // 16 orders → chunks of [15, 1].
        assert_eq!(fake.batch_calls(), vec![15, 1]);
        // Client ids preserved positionally.
        for (i, r) in batch.results.iter().enumerate() {
            assert_eq!(
                r.as_ref().unwrap().client_id.as_deref(),
                Some(&*format!("c{i}"))
            );
        }
    }

    #[tokio::test]
    async fn rejected_ack_maps_to_typed_reject() {
        let (venue, fake) = venue();
        fake.push_ack(RawAck {
            client_id: Some("c0".to_owned()),
            success: false,
            order_id: None,
            status: "unmatched".to_owned(),
            error: Some("invalid post-only order: order crosses book".to_owned()),
        });
        let err = venue
            .place(&order(
                Side::Buy,
                shares(dec!(10)),
                TimeInForce::Gtc { post_only: true },
                "c0",
            ))
            .await
            .unwrap_err();
        assert_eq!(err, VenueError::Rejected(RejectReason::CrossedBook));
    }

    #[tokio::test]
    async fn cancel_reports_partial_failures() {
        let (venue, fake) = venue();
        fake.push_cancel(RawCancel {
            canceled: vec![OrderId::new("a").unwrap()],
            not_canceled: vec![(
                OrderId::new("b").unwrap(),
                "order canceled in the CTF exchange contract".to_owned(),
            )],
        });
        let report = venue.cancel(&OrderId::new("a").unwrap()).await.unwrap();
        assert!(!report.all_canceled());
        assert_eq!(report.canceled, vec![OrderId::new("a").unwrap()]);
        assert_eq!(report.not_canceled.len(), 1);
        assert_eq!(report.not_canceled[0].reason, RejectReason::AlreadyGone);
        assert_eq!(
            fake.cancel_calls(),
            vec![CancelCall::One(OrderId::new("a").unwrap())]
        );
    }

    #[tokio::test]
    async fn cancel_all_and_market_route_to_backend() {
        let (venue, fake) = venue();
        venue.cancel_all().await.unwrap();
        let cid = ConditionId::new(format!("0x{}", "cd".repeat(32))).unwrap();
        venue.cancel_market(&cid).await.unwrap();
        assert_eq!(
            fake.cancel_calls(),
            vec![CancelCall::All, CancelCall::Market(cid)]
        );
    }

    #[tokio::test]
    async fn balances_are_forwarded() {
        let (venue, fake) = venue();
        let wallet = Wallet {
            collateral_available: Dollars::new(dec!(123.45)),
            collateral_total: Dollars::new(dec!(200)),
            positions: vec![],
        };
        fake.set_balances(wallet.clone());
        assert_eq!(venue.balances().await.unwrap(), wallet);
    }

    #[tokio::test]
    async fn event_rx_is_taken_once() {
        let (mut venue, _fake) = venue();
        assert!(venue.take_event_rx().is_some());
        assert!(venue.take_event_rx().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn reconcile_handles_drive_the_event_stream() {
        use std::time::Duration;

        let (mut venue, fake) = venue();
        let mut rx = venue.take_event_rx().unwrap();

        // Place a resting order (tracked), then script a poll showing it filled.
        let accepted = venue
            .place(&order(
                Side::Buy,
                shares(dec!(10)),
                TimeInForce::Gtc { post_only: true },
                "c0",
            ))
            .await
            .unwrap();
        fake.push_open_orders(vec![crate::port::RawOpenOrder {
            order_id: accepted.order_id.clone(),
            status: "MATCHED".to_owned(),
            original_size: Size::new(dec!(10)).unwrap(),
            size_matched: Size::new(dec!(10)).unwrap(),
        }]);

        let (backend, store, tx) = venue.reconcile_handles();
        let handle = tokio::spawn(crate::reconcile_loop(
            backend,
            store,
            tx,
            Duration::from_millis(100),
        ));

        let VenueEvent::Order(update) = rx.recv().await.expect("an order update") else {
            panic!("expected an order update");
        };
        assert_eq!(update.order_id, accepted.order_id);
        assert_eq!(update.state, OrderState::Filled);

        drop(rx);
        tokio::time::advance(Duration::from_millis(200)).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
    }
}
