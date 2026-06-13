//! One connection task per window: connect → subscribe → PING → stream →
//! reconnect forever. The generic feed-util driver cannot express this
//! lifecycle (dynamic short-lived connections, machine-requested recycles,
//! L2 events instead of price ticks), so the loop is hand-rolled over the
//! shared transport/backoff seams (Decisions Log 2026-06-12).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    DurationMs, Event, MarketInfo, MarketLifecycleEvent, TimestampMs, WindowId, WindowLifecycle,
};
use feed_util::{Backoff, Connection, Transport};
use tokio::sync::{mpsc, watch};

use crate::machine::{ConnMachine, MachineOutput, MachineParams, MachineStatus};
use crate::wire::{IgnoredReason, ParsedFrame, parse_frame};
use crate::{ClobParams, FeedError};

/// A connection with no inbound frames at all (data, acks, or PONGs) for
/// this many ping intervals is assumed half-open and torn down.
const DEAD_SOCKET_PINGS: u32 = 3;

/// Repeated-warning suppression window for skipped frames.
const WARN_SUPPRESS: DurationMs = DurationMs::from_millis(10_000);

/// Watchdog cadence: staleness scan, coalescer flush, dead-socket check,
/// status push.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(1);

/// Point-in-time status of one window connection (assembled by the driver,
/// aggregated into [`crate::ClobStatus`] by the supervisor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowFeedStatus {
    /// The window.
    pub window: WindowId,
    /// Gamma event slug (`polymarket.com/event/{slug}` — operator
    /// verification).
    pub event_slug: String,
    /// Window close time.
    pub close_time: TimestampMs,
    /// A connection is currently up.
    pub connected: bool,
    /// Completed connection episodes (reconnects = episodes − 1, counting
    /// machine-requested recycles and forced ones alike).
    pub episodes: u64,
    /// Machine view: health, phase, per-token books, drift/anomaly counts.
    pub machine: MachineStatus,
}

/// Everything one connection task needs.
pub(crate) struct ConnArgs<T, F> {
    pub params: ClobParams,
    pub market: Arc<MarketInfo>,
    pub transport: T,
    pub now_fn: F,
    pub bus_tx: mpsc::Sender<Event>,
    /// Market lifecycle notices → supervisor (dedup) → scheduler.
    pub market_tx: mpsc::Sender<MarketLifecycleEvent>,
    /// Window lifecycle phases from the supervisor.
    pub lifecycle_rx: watch::Receiver<WindowLifecycle>,
    /// Forced recycle (operator demo / `ClobCommand::RecycleWindow`).
    pub recycle_rx: mpsc::Receiver<()>,
    pub status_tx: watch::Sender<WindowFeedStatus>,
    pub shutdown_rx: watch::Receiver<bool>,
    /// Backoff jitter seed; `None` seeds from the clock.
    pub backoff_seed: Option<u64>,
}

/// Why a streaming episode ended.
enum EpisodeEnd {
    /// Reconnect (with the reason logged).
    Reconnect(String),
    /// Graceful shutdown.
    Shutdown,
}

/// Per-reason repeated-warning suppression (local sibling of feed-util's
/// gate — this crate's reason set is small enough not to widen that API).
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
            Some(&at) if now.signed_duration_since(at) < WARN_SUPPRESS => false,
            _ => {
                self.last.insert(reason, now);
                true
            }
        }
    }
}

/// Runs one window's connection until shutdown.
///
/// # Errors
/// [`FeedError::BusClosed`] when the bus (or the supervisor's market-event
/// channel) is gone — the process is shutting down or mis-wired. Connection
/// trouble is logged and retried forever.
pub(crate) async fn run_connection<T, F>(args: ConnArgs<T, F>) -> Result<(), FeedError>
where
    T: Transport,
    F: Fn() -> TimestampMs + Send,
{
    let ConnArgs {
        params,
        market,
        mut transport,
        now_fn,
        bus_tx,
        market_tx,
        mut lifecycle_rx,
        mut recycle_rx,
        status_tx,
        mut shutdown_rx,
        backoff_seed,
    } = args;
    let window = market.window;
    let subscribe = crate::wire::subscribe_message([&market.tokens.up, &market.tokens.down]);
    let machine_params = MachineParams {
        book_stale_after: params.book_stale_after,
        publish_interval: params.publish_interval,
    };
    let mut machine = ConnMachine::new(
        Arc::clone(&market),
        machine_params,
        *lifecycle_rx.borrow(),
        now_fn(),
    );
    let seed = backoff_seed.unwrap_or_else(|| now_fn().as_millis() as u64);
    let mut backoff = Backoff::new(params.backoff, seed);
    let mut warn_gate = WarnGate::new();
    let mut episodes = 0u64;
    let mut outputs = Vec::new();

    loop {
        // A dropped shutdown sender means the supervisor is gone — same as
        // an explicit shutdown (and prevents a closed-watch spin).
        if *shutdown_rx.borrow() || shutdown_rx.has_changed().is_err() {
            return Ok(());
        }
        machine.on_lifecycle(*lifecycle_rx.borrow_and_update());
        let connect = tokio::select! {
            result = transport.connect(&params.url, params.connect_timeout) => result,
            _ = shutdown_rx.changed() => continue,
        };
        let mut conn = match connect {
            Ok(conn) => conn,
            Err(error) => {
                let delay = backoff.next_delay();
                tracing::warn!(
                    target: "feed", feed = "clob", window = %window,
                    error = %error, retry_in_ms = delay.as_millis() as u64,
                    "connect failed"
                );
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
                continue;
            }
        };
        if let Err(error) = conn.send_text(&subscribe).await {
            tracing::warn!(
                target: "feed", feed = "clob", window = %window,
                error = %error, "subscribe send failed"
            );
            conn.close().await;
            sleep_or_shutdown(backoff.next_delay(), &mut shutdown_rx).await;
            continue;
        }
        episodes += 1;
        let now = now_fn();
        machine.on_connected(now);
        tracing::info!(
            target: "feed", feed = "clob", window = %window, episode = episodes,
            "connected and subscribed"
        );
        push_status(&status_tx, &market, true, episodes, &machine, now);

        let end = stream_episode(StreamCtx {
            conn: &mut conn,
            machine: &mut machine,
            market: &market,
            params: &params,
            now_fn: &now_fn,
            bus_tx: &bus_tx,
            market_tx: &market_tx,
            lifecycle_rx: &mut lifecycle_rx,
            recycle_rx: &mut recycle_rx,
            status_tx: &status_tx,
            shutdown_rx: &mut shutdown_rx,
            backoff: &mut backoff,
            warn_gate: &mut warn_gate,
            episodes,
            outputs: &mut outputs,
        })
        .await?;
        conn.close().await;
        let now = now_fn();
        machine.on_disconnected(now, &mut outputs);
        flush_outputs(&mut outputs, &bus_tx, &market_tx).await?;
        push_status(&status_tx, &market, false, episodes, &machine, now);
        match end {
            EpisodeEnd::Shutdown => return Ok(()),
            EpisodeEnd::Reconnect(reason) => {
                let delay = backoff.next_delay();
                tracing::warn!(
                    target: "feed", feed = "clob", window = %window,
                    reason = %reason, retry_in_ms = delay.as_millis() as u64,
                    "connection ended — reconnecting"
                );
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
            }
        }
    }
}

/// Borrowed context for one streaming episode (keeps the select loop's
/// signature manageable).
struct StreamCtx<'a, C, F> {
    conn: &'a mut C,
    machine: &'a mut ConnMachine,
    market: &'a Arc<MarketInfo>,
    params: &'a ClobParams,
    now_fn: &'a F,
    bus_tx: &'a mpsc::Sender<Event>,
    market_tx: &'a mpsc::Sender<MarketLifecycleEvent>,
    lifecycle_rx: &'a mut watch::Receiver<WindowLifecycle>,
    recycle_rx: &'a mut mpsc::Receiver<()>,
    status_tx: &'a watch::Sender<WindowFeedStatus>,
    shutdown_rx: &'a mut watch::Receiver<bool>,
    backoff: &'a mut Backoff,
    warn_gate: &'a mut WarnGate,
    episodes: u64,
    outputs: &'a mut Vec<MachineOutput>,
}

async fn stream_episode<C, F>(ctx: StreamCtx<'_, C, F>) -> Result<EpisodeEnd, FeedError>
where
    C: Connection,
    F: Fn() -> TimestampMs + Send,
{
    let StreamCtx {
        conn,
        machine,
        market,
        params,
        now_fn,
        bus_tx,
        market_tx,
        lifecycle_rx,
        recycle_rx,
        status_tx,
        shutdown_rx,
        backoff,
        warn_gate,
        episodes,
        outputs,
    } = ctx;
    let window = market.window;
    let dead_after = params.ping_interval.saturating_mul(DEAD_SOCKET_PINGS);
    let mut ping = tokio::time::interval(params.ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut watchdog = tokio::time::interval(WATCHDOG_INTERVAL);
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_inbound = now_fn();
    let mut backoff_reset = false;
    // Closed control channels stay "ready" forever — fuse them so the
    // select never spins (they only close at supervisor teardown, which is
    // followed by the shutdown signal).
    let mut lifecycle_open = true;
    let mut recycle_open = true;

    loop {
        tokio::select! {
            frame = conn.recv() => {
                let now = now_fn();
                match frame {
                    None => return Ok(EpisodeEnd::Reconnect("peer closed".to_owned())),
                    Some(Err(error)) => {
                        return Ok(EpisodeEnd::Reconnect(format!("socket error: {error}")));
                    }
                    Some(Ok(frame)) => {
                        last_inbound = now;
                        let text = match frame {
                            feed_util::WsFrame::Text(text) => text,
                            // Binary frames parsed defensively as UTF-8.
                            feed_util::WsFrame::Binary(bytes) => {
                                String::from_utf8_lossy(&bytes).into_owned()
                            }
                            // Server protocol ping: liveness only.
                            feed_util::WsFrame::Ping => continue,
                        };
                        match parse_frame(&text, now) {
                            ParsedFrame::Ack => {
                                tracing::debug!(
                                    target: "feed", feed = "clob", window = %window,
                                    "ack frame"
                                );
                            }
                            ParsedFrame::Pong => {}
                            ParsedFrame::Ignored(reason) => {
                                if warn_gate.should_warn(reason, now) {
                                    tracing::warn!(
                                        target: "feed", feed = "clob", window = %window,
                                        reason = %reason,
                                        preview = preview(&text),
                                        "skipping frame"
                                    );
                                }
                            }
                            ParsedFrame::Events(events) => {
                                let mut any_ok = false;
                                for event in events {
                                    match event {
                                        Ok(event) => {
                                            any_ok = true;
                                            machine.on_event(event, now, outputs);
                                        }
                                        Err(reason) => {
                                            if warn_gate.should_warn(reason, now) {
                                                tracing::warn!(
                                                    target: "feed", feed = "clob",
                                                    window = %window, reason = %reason,
                                                    preview = preview(&text),
                                                    "skipping event in frame"
                                                );
                                            }
                                        }
                                    }
                                }
                                if any_ok && !backoff_reset {
                                    backoff.reset();
                                    backoff_reset = true;
                                }
                                if let Some(end) =
                                    drain(outputs, bus_tx, market_tx, window).await?
                                {
                                    return Ok(end);
                                }
                            }
                        }
                    }
                }
            }
            _ = ping.tick() => {
                if let Err(error) = conn.send_text(crate::wire::PING_TEXT).await {
                    return Ok(EpisodeEnd::Reconnect(format!("ping send failed: {error}")));
                }
            }
            _ = watchdog.tick() => {
                let now = now_fn();
                let silent = now.signed_duration_since(last_inbound);
                if silent.as_millis() >= dead_after.as_millis() as i64 {
                    return Ok(EpisodeEnd::Reconnect(format!(
                        "dead socket: no inbound frames for {silent}"
                    )));
                }
                machine.on_tick(now, outputs);
                if let Some(end) = drain(outputs, bus_tx, market_tx, window).await? {
                    return Ok(end);
                }
                push_status(status_tx, market, true, episodes, machine, now);
            }
            changed = lifecycle_rx.changed(), if lifecycle_open => {
                match changed {
                    Ok(()) => machine.on_lifecycle(*lifecycle_rx.borrow_and_update()),
                    Err(_) => lifecycle_open = false,
                }
            }
            recycle = recycle_rx.recv(), if recycle_open => {
                match recycle {
                    Some(()) => return Ok(EpisodeEnd::Reconnect("forced recycle".to_owned())),
                    None => recycle_open = false,
                }
            }
            changed = shutdown_rx.changed() => {
                // A dropped sender (supervisor gone) is a shutdown too.
                if changed.is_err() || *shutdown_rx.borrow() {
                    return Ok(EpisodeEnd::Shutdown);
                }
            }
        }
    }
}

/// Executes machine outputs in order. Returns the episode end when the
/// machine requested a recycle.
async fn drain(
    outputs: &mut Vec<MachineOutput>,
    bus_tx: &mpsc::Sender<Event>,
    market_tx: &mpsc::Sender<MarketLifecycleEvent>,
    window: WindowId,
) -> Result<Option<EpisodeEnd>, FeedError> {
    let mut end = None;
    for output in outputs.drain(..) {
        match output {
            MachineOutput::Publish(event) => {
                bus_tx.send(event).await.map_err(|_| FeedError::BusClosed)?;
            }
            MachineOutput::Market(event) => {
                market_tx
                    .send(event)
                    .await
                    .map_err(|_| FeedError::BusClosed)?;
            }
            MachineOutput::Recycle(reason) => {
                if end.is_none() {
                    end = Some(EpisodeEnd::Reconnect(format!(
                        "machine recycle: {reason:?}"
                    )));
                }
            }
            MachineOutput::Drift { token } => {
                tracing::debug!(
                    target: "feed", feed = "clob", window = %window, token = %token.as_str(),
                    "derived book drifted from venue snapshot (repaired by replace)"
                );
            }
        }
    }
    Ok(end)
}

async fn flush_outputs(
    outputs: &mut Vec<MachineOutput>,
    bus_tx: &mpsc::Sender<Event>,
    market_tx: &mpsc::Sender<MarketLifecycleEvent>,
) -> Result<(), FeedError> {
    for output in outputs.drain(..) {
        match output {
            MachineOutput::Publish(event) => {
                bus_tx.send(event).await.map_err(|_| FeedError::BusClosed)?;
            }
            MachineOutput::Market(event) => {
                market_tx
                    .send(event)
                    .await
                    .map_err(|_| FeedError::BusClosed)?;
            }
            // No live connection — a recycle request is moot.
            MachineOutput::Recycle(_) | MachineOutput::Drift { .. } => {}
        }
    }
    Ok(())
}

fn push_status(
    status_tx: &watch::Sender<WindowFeedStatus>,
    market: &Arc<MarketInfo>,
    connected: bool,
    episodes: u64,
    machine: &ConnMachine,
    now: TimestampMs,
) {
    // A dropped receiver just means nobody is watching — never an error.
    let _ = status_tx.send(WindowFeedStatus {
        window: market.window,
        event_slug: market.event_slug.clone(),
        close_time: market.close_time,
        connected,
        episodes,
        machine: machine.status(now),
    });
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
        text.to_owned()
    } else {
        let cut = text
            .char_indices()
            .take_while(|(i, _)| *i < MAX)
            .last()
            .map_or(0, |(i, c)| i + c.len_utf8());
        format!("{}…", &text[..cut])
    }
}
