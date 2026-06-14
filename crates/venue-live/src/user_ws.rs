//! The authenticated `/ws/user` connection task: connect → subscribe →
//! reconcile → stream → reconnect, forever.
//!
//! Hand-rolled over feed-util's [`Transport`]/[`Backoff`] seams (the generic
//! price-tick driver cannot express the on-connect REST reconcile or the
//! request/response auth subscription), following feed-clob's `run_connection`.
//! It is the only IO in the user-channel path: the [`OrderStore`] and the wire
//! parser stay pure, so this loop is driven offline by `feed_util::fake` +
//! [`FakeClobPort`](crate::FakeClobPort) under paused time.
//!
//! On every (re)connect it runs a full REST open-orders reconcile **before**
//! streaming, so a fill that happened during the gap is recovered and ordered
//! ahead of new events (CLAUDE.md §11 / the user-channel docs). A low-frequency
//! safety reconcile runs while connected as defence in depth.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use config::Secrets;
use core_types::{ConditionId, TimestampMs};
use feed_util::{Backoff, Connection, Transport, WsFrame};
use tokio::sync::{mpsc, watch};
use venue_api::VenueEvent;

use crate::error::VenueLiveError;
use crate::params::LiveParams;
use crate::port::ClobPort;
use crate::store::{OrderStore, StoreEffect};
use crate::user_wire::{
    IgnoredReason, PING_TEXT, ParsedUserFrame, WireUserEvent, parse_user_frame, subscribe_message,
};

/// Repeated-warning suppression window for skipped frames.
const WARN_SUPPRESS_MS: i64 = 10_000;

/// The L2 API credentials the user-channel subscription needs. Holds the
/// exposed secret strings (it must serialize them into the subscribe message),
/// so it redacts itself in `Debug` and is never serializable.
#[derive(Clone)]
pub struct UserWsCreds {
    api_key: String,
    secret: String,
    passphrase: String,
}

impl UserWsCreds {
    /// Wraps already-obtained credential values.
    #[must_use]
    pub fn new(api_key: String, secret: String, passphrase: String) -> Self {
        Self {
            api_key,
            secret,
            passphrase,
        }
    }

    /// Extracts the three L2 credentials from [`Secrets`]. The §11 arming gate
    /// already requires all of them, so `connect` has them by construction; this
    /// fails closed if a caller reaches here without them (e.g. the SDK derived
    /// them and they were never set as secrets).
    ///
    /// # Errors
    /// [`VenueLiveError::MissingCredentials`] if any of the three is absent.
    pub fn from_secrets(secrets: &Secrets) -> Result<Self, VenueLiveError> {
        let (Some(key), Some(secret), Some(pass)) = (
            secrets.pm_api_key.as_ref(),
            secrets.pm_api_secret.as_ref(),
            secrets.pm_api_passphrase.as_ref(),
        ) else {
            return Err(VenueLiveError::MissingCredentials);
        };
        Ok(Self::new(
            key.expose().to_owned(),
            secret.expose().to_owned(),
            pass.expose().to_owned(),
        ))
    }

    pub(crate) fn api_key(&self) -> &str {
        &self.api_key
    }
    pub(crate) fn secret(&self) -> &str {
        &self.secret
    }
    pub(crate) fn passphrase(&self) -> &str {
        &self.passphrase
    }
}

impl std::fmt::Debug for UserWsCreds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserWsCreds")
            .field("api_key", &"[REDACTED]")
            .field("secret", &"[REDACTED]")
            .field("passphrase", &"[REDACTED]")
            .finish()
    }
}

/// Everything the user-channel task needs.
pub(crate) struct UserWsArgs<T, P, F> {
    /// Network/timing parameters (URL, ping, dead-after, backoff, safety
    /// reconcile interval, empty-markets policy).
    pub params: LiveParams,
    /// The L2 credentials for the subscribe message.
    pub creds: UserWsCreds,
    /// REST backend for the reconcile poll.
    pub backend: Arc<P>,
    /// The canonical order store (shared with `LiveVenue`).
    pub store: Arc<Mutex<OrderStore>>,
    /// The venue event channel (shared with `LiveVenue::take_event_rx`).
    pub events: mpsc::Sender<VenueEvent>,
    /// WebSocket transport (real or scripted fake).
    pub transport: T,
    /// Desired subscription set (condition ids); a change reconnects to
    /// resubscribe.
    pub markets_rx: watch::Receiver<Vec<ConditionId>>,
    /// Shutdown signal (a dropped sender is also a shutdown).
    pub shutdown_rx: watch::Receiver<bool>,
    /// Wall-clock source (stamps effects + drives the dead-socket watchdog).
    pub now_fn: F,
    /// Backoff jitter seed; `None` seeds from the clock.
    pub backoff_seed: Option<u64>,
}

/// Why a streaming episode ended.
enum EpisodeEnd {
    /// Reconnect (reason logged).
    Reconnect(String),
    /// Stop the task (shutdown signalled, or the event channel closed).
    Stop,
}

/// Per-reason repeated-warning suppression (local sibling of feed-util's gate).
struct WarnGate {
    last: HashMap<IgnoredReason, TimestampMs>,
}

impl WarnGate {
    fn new() -> Self {
        Self {
            last: HashMap::new(),
        }
    }

    fn should_warn(&mut self, reason: IgnoredReason, now: TimestampMs) -> bool {
        match self.last.get(&reason) {
            Some(&at) if now.signed_duration_since(at).as_millis() < WARN_SUPPRESS_MS => false,
            _ => {
                self.last.insert(reason, now);
                true
            }
        }
    }
}

/// Runs the user channel until shutdown (or the event consumer goes away).
pub(crate) async fn run<T, P, F>(args: UserWsArgs<T, P, F>)
where
    T: Transport,
    P: ClobPort,
    F: Fn() -> TimestampMs + Send,
{
    let UserWsArgs {
        params,
        creds,
        backend,
        store,
        events,
        mut transport,
        mut markets_rx,
        mut shutdown_rx,
        now_fn,
        backoff_seed,
    } = args;
    let seed = backoff_seed.unwrap_or_else(|| now_fn().as_millis() as u64);
    let mut backoff = Backoff::new(params.ws_backoff, seed);
    let mut warn_gate = WarnGate::new();

    loop {
        if *shutdown_rx.borrow() || shutdown_rx.has_changed().is_err() {
            return;
        }
        let markets = markets_rx.borrow_and_update().clone();
        if markets.is_empty() && !params.subscribe_all_when_empty {
            tracing::info!(
                target: "venue::live",
                "user channel: no markets to subscribe; waiting for a non-empty set"
            );
            tokio::select! {
                _ = markets_rx.changed() => continue,
                _ = shutdown_rx.changed() => return,
            }
        }
        let subscribe = subscribe_message(
            creds.api_key(),
            creds.secret(),
            creds.passphrase(),
            markets.iter(),
        );

        let connect = tokio::select! {
            result = transport.connect(&params.user_ws_url, params.ws_connect_timeout) => result,
            _ = shutdown_rx.changed() => continue,
        };
        let mut conn = match connect {
            Ok(conn) => conn,
            Err(error) => {
                let delay = backoff.next_delay();
                tracing::warn!(
                    target: "venue::live", %error,
                    retry_in_ms = delay.as_millis() as u64, "user channel connect failed"
                );
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
                continue;
            }
        };
        if let Err(error) = conn.send_text(&subscribe).await {
            tracing::warn!(target: "venue::live", %error, "user channel subscribe send failed");
            conn.close().await;
            sleep_or_shutdown(backoff.next_delay(), &mut shutdown_rx).await;
            continue;
        }
        tracing::info!(
            target: "venue::live", markets = markets.len(),
            empty_is_all = markets.is_empty(), "user channel connected and subscribed"
        );

        // On-connect reconcile, before streaming: recover any fill missed during
        // the gap and order it ahead of new events.
        if reconcile_now(&backend, &store, &events, &now_fn)
            .await
            .is_err()
        {
            return;
        }

        let end = stream_episode(StreamCtx {
            conn: &mut conn,
            params: &params,
            backend: &backend,
            store: &store,
            events: &events,
            markets_rx: &mut markets_rx,
            shutdown_rx: &mut shutdown_rx,
            now_fn: &now_fn,
            backoff: &mut backoff,
            warn_gate: &mut warn_gate,
        })
        .await;
        conn.close().await;
        match end {
            EpisodeEnd::Stop => return,
            EpisodeEnd::Reconnect(reason) => {
                let delay = backoff.next_delay();
                tracing::warn!(
                    target: "venue::live", %reason,
                    retry_in_ms = delay.as_millis() as u64, "user channel ended — reconnecting"
                );
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
            }
        }
    }
}

/// Borrowed context for one streaming episode.
struct StreamCtx<'a, C, P, F> {
    conn: &'a mut C,
    params: &'a LiveParams,
    backend: &'a Arc<P>,
    store: &'a Arc<Mutex<OrderStore>>,
    events: &'a mpsc::Sender<VenueEvent>,
    markets_rx: &'a mut watch::Receiver<Vec<ConditionId>>,
    shutdown_rx: &'a mut watch::Receiver<bool>,
    now_fn: &'a F,
    backoff: &'a mut Backoff,
    warn_gate: &'a mut WarnGate,
}

async fn stream_episode<C, P, F>(ctx: StreamCtx<'_, C, P, F>) -> EpisodeEnd
where
    C: Connection,
    P: ClobPort,
    F: Fn() -> TimestampMs + Send,
{
    let StreamCtx {
        conn,
        params,
        backend,
        store,
        events,
        markets_rx,
        shutdown_rx,
        now_fn,
        backoff,
        warn_gate,
    } = ctx;
    let dead_after_ms = params.ws_dead_after.as_millis() as i64;
    let mut ping = tokio::time::interval(params.ws_ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // First safety tick is one interval out (the on-connect reconcile just ran).
    let mut safety = tokio::time::interval_at(
        tokio::time::Instant::now() + params.safety_reconcile_interval,
        params.safety_reconcile_interval,
    );
    safety.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_inbound = now_fn();
    let mut backoff_reset = false;
    let mut markets_open = true;

    loop {
        tokio::select! {
            frame = conn.recv() => {
                match frame {
                    None => return EpisodeEnd::Reconnect("peer closed".to_owned()),
                    Some(Err(error)) => {
                        return EpisodeEnd::Reconnect(format!("socket error: {error}"));
                    }
                    Some(Ok(frame)) => {
                        let now = now_fn();
                        last_inbound = now;
                        let text = match frame {
                            WsFrame::Text(text) => text,
                            WsFrame::Binary(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                            WsFrame::Ping => continue,
                        };
                        match handle_text(&text, now, store, warn_gate) {
                            HandledFrame::Effects { effects, any_ok } => {
                                if any_ok && !backoff_reset {
                                    backoff.reset();
                                    backoff_reset = true;
                                }
                                if send_effects(effects, events).await.is_err() {
                                    return EpisodeEnd::Stop;
                                }
                            }
                            HandledFrame::Nothing => {}
                        }
                    }
                }
            }
            _ = ping.tick() => {
                let now = now_fn();
                if now.signed_duration_since(last_inbound).as_millis() >= dead_after_ms {
                    return EpisodeEnd::Reconnect(format!(
                        "dead socket: no inbound frames for {dead_after_ms} ms"
                    ));
                }
                if let Err(error) = conn.send_text(PING_TEXT).await {
                    return EpisodeEnd::Reconnect(format!("ping send failed: {error}"));
                }
            }
            _ = safety.tick() => {
                if reconcile_now(backend, store, events, now_fn).await.is_err() {
                    return EpisodeEnd::Stop;
                }
            }
            changed = markets_rx.changed(), if markets_open => {
                match changed {
                    Ok(()) => return EpisodeEnd::Reconnect("subscription markets changed".to_owned()),
                    Err(_) => markets_open = false,
                }
            }
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    return EpisodeEnd::Stop;
                }
            }
        }
    }
}

/// What a parsed text frame produced.
enum HandledFrame {
    /// Store effects to publish, and whether any event parsed successfully
    /// (drives the backoff reset).
    Effects {
        effects: Vec<StoreEffect>,
        any_ok: bool,
    },
    /// Ack/Pong/ignored — nothing to publish.
    Nothing,
}

/// Parses one text frame and applies its events to the store under one lock.
fn handle_text(
    text: &str,
    now: TimestampMs,
    store: &Arc<Mutex<OrderStore>>,
    warn_gate: &mut WarnGate,
) -> HandledFrame {
    match parse_user_frame(text, now) {
        ParsedUserFrame::Ack => {
            tracing::debug!(target: "venue::live", "user channel ack frame");
            HandledFrame::Nothing
        }
        ParsedUserFrame::Pong => HandledFrame::Nothing,
        ParsedUserFrame::Ignored(reason) => {
            if warn_gate.should_warn(reason, now) {
                tracing::warn!(
                    target: "venue::live", reason = %reason,
                    preview = preview(text), "skipping user-channel frame"
                );
            }
            HandledFrame::Nothing
        }
        ParsedUserFrame::Events(events) => {
            let mut effects = Vec::new();
            let mut any_ok = false;
            let mut store = store.lock().unwrap_or_else(|e| e.into_inner());
            for event in events {
                match event {
                    Ok(WireUserEvent::Order(order)) => {
                        any_ok = true;
                        effects.extend(store.apply_order(&order, now));
                    }
                    Ok(WireUserEvent::Trade(trade)) => {
                        any_ok = true;
                        effects.extend(store.apply_trade(&trade, now));
                    }
                    Err(reason) => {
                        if warn_gate.should_warn(reason, now) {
                            tracing::warn!(
                                target: "venue::live", reason = %reason,
                                preview = preview(text), "skipping user-channel event"
                            );
                        }
                    }
                }
            }
            HandledFrame::Effects { effects, any_ok }
        }
    }
}

/// Runs a full REST reconcile and publishes the corrections. Returns `Err(())`
/// when the event channel has closed (the task should stop).
async fn reconcile_now<P, F>(
    backend: &Arc<P>,
    store: &Arc<Mutex<OrderStore>>,
    events: &mpsc::Sender<VenueEvent>,
    now_fn: &F,
) -> Result<(), ()>
where
    P: ClobPort,
    F: Fn() -> TimestampMs,
{
    match backend.open_orders().await {
        Ok(polled) => {
            let now = now_fn();
            let effects = {
                let mut store = store.lock().unwrap_or_else(|e| e.into_inner());
                store.reconcile(&polled, now)
            };
            send_effects(effects, events).await
        }
        Err(error) => {
            tracing::warn!(target: "venue::live", %error, "user-channel reconcile poll failed");
            Ok(())
        }
    }
}

/// Publishes store effects in order. Returns `Err(())` if the channel closed.
async fn send_effects(
    effects: Vec<StoreEffect>,
    events: &mpsc::Sender<VenueEvent>,
) -> Result<(), ()> {
    for effect in effects {
        let event = match effect {
            StoreEffect::Order(update) => VenueEvent::Order(Arc::new(update)),
            StoreEffect::Fill(fill) => VenueEvent::Fill(Arc::new(fill)),
        };
        if events.send(event).await.is_err() {
            return Err(());
        }
    }
    Ok(())
}

async fn sleep_or_shutdown(delay: Duration, shutdown_rx: &mut watch::Receiver<bool>) {
    tokio::select! {
        () = tokio::time::sleep(delay) => {}
        _ = shutdown_rx.changed() => {}
    }
}

fn preview(text: &str) -> String {
    const MAX: usize = 160;
    if text.len() <= MAX {
        return text.to_owned();
    }
    let cut = text
        .char_indices()
        .take_while(|(i, _)| *i < MAX)
        .last()
        .map_or(0, |(i, c)| i + c.len_utf8());
    format!("{}…", &text[..cut])
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use core_types::{
        Asset, Decimal, OrderId, OrderState, Outcome, Price, RoundDir, Series, Side, Size,
        TickSize, WindowDuration, WindowId,
    };
    use feed_util::WsFrame;
    use feed_util::fake::{ConnHandle, FakeTransport, script};
    use rust_decimal::dec;

    use super::*;
    use crate::fake::FakeClobPort;
    use crate::port::RawOpenOrder;
    use crate::store::TrackedOrder;

    const NOW: TimestampMs = TimestampMs::from_millis(1_700_000_000_000);

    fn token() -> core_types::TokenId {
        core_types::TokenId::new("123").unwrap()
    }
    fn cid() -> ConditionId {
        ConditionId::new(format!("0x{}", "ab".repeat(32))).unwrap()
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
    fn limit() -> Price {
        Price::quantize(dec!(0.40), TickSize::T001, RoundDir::Down).unwrap()
    }

    fn tracking_store(id: &str) -> Arc<Mutex<OrderStore>> {
        let mut store = OrderStore::new();
        store.set_default_fee_rate(dec!(0.07));
        store.track(
            OrderId::new(id).unwrap(),
            TrackedOrder {
                window: window(),
                token_id: token(),
                outcome: Outcome::Up,
                side: Side::Buy,
                price: limit(),
                original_size: Size::new(dec!(10)).unwrap(),
                last_state: OrderState::Open,
                last_filled: Size::ZERO,
            },
        );
        Arc::new(Mutex::new(store))
    }

    fn raw_open(id: &str, status: &str, matched: Decimal) -> RawOpenOrder {
        RawOpenOrder {
            order_id: OrderId::new(id).unwrap(),
            status: status.to_owned(),
            original_size: Size::new(dec!(10)).unwrap(),
            size_matched: Size::new(matched).unwrap(),
            token_id: token(),
            side: Side::Buy,
            price: dec!(0.40),
            condition_id: cid(),
        }
    }

    fn order_update_frame(id: &str, matched: Decimal) -> WsFrame {
        WsFrame::Text(format!(
            r#"{{"event_type":"order","type":"UPDATE","id":"{id}","market":"{}","asset_id":"123",
                "side":"BUY","original_size":"10","price":"0.40","size_matched":"{matched}"}}"#,
            cid()
        ))
    }

    fn maker_trade_frame(trade_id: &str, maker_order: &str, amount: Decimal) -> WsFrame {
        WsFrame::Text(format!(
            r#"{{"event_type":"trade","id":"{trade_id}","taker_order_id":"counterparty","market":"{}",
                "asset_id":"123","side":"SELL","size":"{amount}","price":"0.40","status":"MATCHED",
                "maker_orders":[{{"order_id":"{maker_order}","asset_id":"123","matched_amount":"{amount}","price":"0.40"}}]}}"#,
            cid()
        ))
    }

    struct Harness {
        handles: VecDeque<ConnHandle>,
        event_rx: mpsc::Receiver<VenueEvent>,
        markets_tx: watch::Sender<Vec<ConditionId>>,
        shutdown_tx: watch::Sender<bool>,
        join: tokio::task::JoinHandle<()>,
    }

    fn spawn(
        store: Arc<Mutex<OrderStore>>,
        backend: Arc<FakeClobPort>,
        ok_attempts: &[bool],
    ) -> Harness {
        let (transport, handles, _attempt_at): (FakeTransport, _, _) = script(ok_attempts);
        let (event_tx, event_rx) = mpsc::channel(64);
        let (markets_tx, markets_rx) = watch::channel(vec![cid()]);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let join = tokio::spawn(run(UserWsArgs {
            params: LiveParams::default(),
            creds: UserWsCreds::new("key".into(), "sec".into(), "pass".into()),
            backend,
            store,
            events: event_tx,
            transport,
            markets_rx,
            shutdown_rx,
            now_fn: || NOW,
            backoff_seed: Some(1),
        }));
        Harness {
            handles,
            event_rx,
            markets_tx,
            shutdown_tx,
            join,
        }
    }

    async fn recv_subscribe(handle: &mut ConnHandle) -> String {
        tokio::time::timeout(Duration::from_secs(5), handle.sent_rx.recv())
            .await
            .expect("subscribe within timeout")
            .expect("a subscribe frame")
    }

    async fn recv_event(rx: &mut mpsc::Receiver<VenueEvent>) -> VenueEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event within timeout")
            .expect("an event")
    }

    #[tokio::test(start_paused = true)]
    async fn connect_subscribes_then_reconciles_then_streams() {
        let store = tracking_store("o1");
        let backend = Arc::new(FakeClobPort::new());
        backend.push_open_orders(vec![raw_open("o1", "LIVE", dec!(0))]); // on-connect: no-op
        let mut h = spawn(store, Arc::clone(&backend), &[true]);
        let mut conn = h.handles.pop_front().unwrap();

        let subscribe = recv_subscribe(&mut conn).await;
        assert!(subscribe.contains("\"apiKey\":\"key\""), "got {subscribe}");
        assert!(subscribe.contains("\"type\":\"user\""));
        assert!(subscribe.contains(cid().as_str()));

        // Stream an order update → a PartiallyFilled order event.
        conn.frame_tx
            .send(order_update_frame("o1", dec!(4)))
            .unwrap();
        let VenueEvent::Order(u) = recv_event(&mut h.event_rx).await else {
            panic!("expected an order update");
        };
        assert_eq!(u.state, OrderState::PartiallyFilled);
        assert_eq!(u.filled_size, Size::new(dec!(4)).unwrap());

        let _ = h.shutdown_tx.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(1), h.join).await;
    }

    #[tokio::test(start_paused = true)]
    async fn reconnect_recovers_missed_fill_then_dedups_the_real_trade() {
        let store = tracking_store("o1");
        let backend = Arc::new(FakeClobPort::new());
        // conn1 on-connect: nothing filled. conn2 on-connect: 6/10 filled (the
        // fill happened while we were disconnected).
        backend.push_open_orders(vec![raw_open("o1", "LIVE", dec!(0))]);
        backend.push_open_orders(vec![raw_open("o1", "LIVE", dec!(6))]);
        let mut h = spawn(store, Arc::clone(&backend), &[true, true]);

        let mut conn1 = h.handles.pop_front().unwrap();
        let _ = recv_subscribe(&mut conn1).await;
        // Drop conn1 → driver sees a clean peer close → reconnect after backoff.
        drop(conn1);
        tokio::time::advance(Duration::from_secs(1)).await;

        let mut conn2 = h.handles.pop_front().unwrap();
        let _ = recv_subscribe(&mut conn2).await;

        // The reconnect's on-connect reconcile recovers the missed fill: a
        // synthetic maker Fill for the 6 shares, then the OrderUpdate correction.
        let VenueEvent::Fill(fill) = recv_event(&mut h.event_rx).await else {
            panic!("expected a synthetic fill");
        };
        assert_eq!(fill.size, Size::new(dec!(6)).unwrap());
        assert_eq!(fill.trade_id, None);
        let VenueEvent::Order(u) = recv_event(&mut h.event_rx).await else {
            panic!("expected the correction order update");
        };
        assert_eq!(u.state, OrderState::PartiallyFilled);
        assert_eq!(u.filled_size, Size::new(dec!(6)).unwrap());

        // The real trade for those same 6 shares arrives → no second Fill.
        conn2
            .frame_tx
            .send(maker_trade_frame("t-late", "o1", dec!(6)))
            .unwrap();
        // Give the task a chance to process; assert no Fill follows.
        tokio::time::advance(Duration::from_millis(10)).await;
        match tokio::time::timeout(Duration::from_millis(50), h.event_rx.recv()).await {
            Err(_) => {}                         // timed out — nothing emitted, as expected
            Ok(Some(VenueEvent::Order(_))) => {} // a no-op order update is acceptable
            Ok(other) => panic!("expected no fill for already-covered shares, got {other:?}"),
        }

        let _ = h.shutdown_tx.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(1), h.join).await;
    }

    #[tokio::test(start_paused = true)]
    async fn duplicate_trade_over_the_wire_emits_one_fill() {
        let store = tracking_store("o1");
        let backend = Arc::new(FakeClobPort::new());
        backend.push_open_orders(vec![raw_open("o1", "LIVE", dec!(0))]);
        let mut h = spawn(store, Arc::clone(&backend), &[true]);
        let mut conn = h.handles.pop_front().unwrap();
        let _ = recv_subscribe(&mut conn).await;

        // Same trade frame twice.
        conn.frame_tx
            .send(maker_trade_frame("t1", "o1", dec!(4)))
            .unwrap();
        conn.frame_tx
            .send(maker_trade_frame("t1", "o1", dec!(4)))
            .unwrap();

        // First delivery: exactly one Fill (+ its order update).
        let VenueEvent::Fill(fill) = recv_event(&mut h.event_rx).await else {
            panic!("expected a fill");
        };
        assert_eq!(fill.size, Size::new(dec!(4)).unwrap());
        let VenueEvent::Order(_) = recv_event(&mut h.event_rx).await else {
            panic!("expected the fill's order update");
        };
        // The duplicate produced nothing.
        tokio::time::advance(Duration::from_millis(10)).await;
        assert!(
            tokio::time::timeout(Duration::from_millis(50), h.event_rx.recv())
                .await
                .is_err(),
            "the duplicate trade must emit nothing"
        );

        let _ = h.shutdown_tx.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(1), h.join).await;
    }

    #[tokio::test(start_paused = true)]
    async fn ping_is_sent_on_the_interval() {
        let store = tracking_store("o1");
        let backend = Arc::new(FakeClobPort::new());
        let mut h = spawn(store, Arc::clone(&backend), &[true]);
        let mut conn = h.handles.pop_front().unwrap();
        let _ = recv_subscribe(&mut conn).await;

        tokio::time::advance(LiveParams::default().ws_ping_interval).await;
        let next = tokio::time::timeout(Duration::from_secs(5), conn.sent_rx.recv())
            .await
            .expect("a ping")
            .expect("ping text");
        assert_eq!(next, PING_TEXT);

        let _ = h.shutdown_tx.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(1), h.join).await;
    }

    #[tokio::test(start_paused = true)]
    async fn markets_change_triggers_resubscribe() {
        let store = tracking_store("o1");
        let backend = Arc::new(FakeClobPort::new());
        let mut h = spawn(store, Arc::clone(&backend), &[true, true]);
        let mut conn1 = h.handles.pop_front().unwrap();
        let _ = recv_subscribe(&mut conn1).await;

        // Change the desired subscription set → reconnect + resubscribe.
        let other = ConditionId::new(format!("0x{}", "cd".repeat(32))).unwrap();
        h.markets_tx.send(vec![other.clone()]).unwrap();
        tokio::time::advance(Duration::from_secs(1)).await;

        let mut conn2 = h.handles.pop_front().unwrap();
        let subscribe = recv_subscribe(&mut conn2).await;
        assert!(subscribe.contains(other.as_str()), "got {subscribe}");

        let _ = h.shutdown_tx.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(1), h.join).await;
    }
}
