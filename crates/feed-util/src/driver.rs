//! The generic async shell around the machine: owns the connection lifecycle
//! (connect → subscribe → stream → reconnect with jittered backoff), the
//! optional client keepalive, the watchdog tick, the runtime command channel,
//! and all bus sends. Everything protocol-specific — connect messages, frame
//! parsing, keepalive text, command semantics — lives behind [`FeedProtocol`];
//! feed-rtds and feed-binance are thin protocol impls over this one loop.
//! Transport trouble is never fatal — the driver reconnects forever; only a
//! dropped bus (process teardown) errors.
//!
//! Logging: tracing targets must be string literals, so one generic driver
//! cannot emit per-feed targets — every event here goes out under target
//! `"feed"` with a `feed = <name>` field (protocol/wire-level logs in the
//! feed crates keep their own targets).

use std::collections::HashMap;
use std::hash::Hash;
use std::time::Duration;

use core_types::{DurationMs, Event, FeedHealth, TimestampMs};
use tokio::sync::{mpsc, watch};
use tokio::time::MissedTickBehavior;

use crate::backoff::{Backoff, BackoffParams};
use crate::error::FeedError;
use crate::machine::{FeedMachine, KeyStatus, Output, StreamKey};
use crate::transport::{Connection, Transport, WsFrame};

/// Watchdog cadence: staleness scans + status snapshots.
const WATCHDOG_TICK: Duration = Duration::from_secs(1);

/// A tracked stream stale for this many of its own staleness thresholds —
/// while other traffic proves the connection alive — means its server-side
/// subscription decayed (observed live on RTDS 2026-06-12); the connection is
/// recycled to resubscribe. Bounded retry: one recycle at most every
/// `STALE_RECYCLE_MULTIPLE × stale_after` because anchors refresh on every
/// reconnect.
const STALE_RECYCLE_MULTIPLE: i64 = 6;

/// Minimum spacing between repeated warnings for the same skip reason.
const WARN_EVERY: DurationMs = DurationMs::from_millis(10_000);

/// Optional client-initiated keepalive (RTDS wants a text `"PING"` every
/// 5 s; Binance forbids unsolicited messages — its server pings us instead,
/// answered automatically by the transport).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keepalive {
    /// Send cadence.
    pub interval: Duration,
    /// The text frame to send.
    pub text: String,
}

/// Connection parameters (mapped from `feeds.*` config by the binary — feed
/// crates depend only on core-types and this crate).
#[derive(Debug, Clone, PartialEq)]
pub struct DriverParams {
    /// WebSocket endpoint URL.
    pub url: String,
    /// TCP+TLS+WS handshake deadline.
    pub connect_timeout: Duration,
    /// Reconnect backoff curve.
    pub backoff: BackoffParams,
    /// A connection with no inbound frames at all (data, acks, or server
    /// pings) for this long is assumed half-open and torn down.
    pub dead_after: Duration,
    /// Client keepalive, if the venue wants one.
    pub keepalive: Option<Keepalive>,
}

/// Connection phase, for status displays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnState {
    /// Dialing (also the initial state).
    #[default]
    Connecting,
    /// Streaming.
    Connected,
    /// Between attempts, sleeping the backoff delay.
    Backoff,
}

impl std::fmt::Display for ConnState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connecting => f.write_str("connecting"),
            Self::Connected => f.write_str("connected"),
            Self::Backoff => f.write_str("backoff"),
        }
    }
}

/// Point-in-time feed snapshot, pushed over the status watch channel after
/// every watchdog tick and connection transition.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedStatus<K> {
    /// When the snapshot was taken.
    pub at: TimestampMs,
    /// Connection phase.
    pub connection: ConnState,
    /// Successful re-connections since boot (first connect not counted).
    pub reconnects: u32,
    /// Per-stream state (consumers compute display ages against their own
    /// clock so a frozen snapshot never shows a frozen age).
    pub keys: Vec<KeyStatus<K>>,
}

// Manual impl: a derive would demand `K: Default` for an always-empty Vec.
impl<K> Default for FeedStatus<K> {
    fn default() -> Self {
        Self {
            at: TimestampMs::from_millis(0),
            connection: ConnState::default(),
            reconnects: 0,
            keys: Vec::new(),
        }
    }
}

/// One parsed price observation handed from the protocol to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceObs<K> {
    /// Which tracked stream it belongs to.
    pub key: K,
    /// Wire-exact value.
    pub value: core_types::Decimal,
    /// Source timestamp from the wire (a feed with no wire timestamp passes
    /// the `now` it was handed — see [`FeedProtocol::handle_frame`]).
    pub ts_exchange: TimestampMs,
}

/// What the protocol made of one inbound text frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameOutcome<K, R> {
    /// A subscription acknowledgement (logged at debug).
    Ack,
    /// Venue keepalive chatter (server PONGs etc.) — nothing to do.
    Keepalive,
    /// One or more price observations (a backfill carries many).
    Prices(Vec<PriceObs<K>>),
    /// Skipped frame, warn-gated with the reason.
    Ignored(R),
    /// Skipped frame that is *normal* traffic (e.g. untracked symbols under
    /// an all-symbols subscription) — logged at trace, never warned.
    IgnoredQuiet(R),
}

/// One step of a runtime command, returned by
/// [`FeedProtocol::on_command`] in execution order (machine changes are
/// listed before the wire sends they relate to, mirroring the original RTDS
/// driver's ordering).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandAction<K> {
    /// Start tracking a stream with its staleness threshold.
    Track {
        /// The stream.
        key: K,
        /// Its staleness threshold.
        stale_after: DurationMs,
    },
    /// Stop tracking a stream (silently — not an outage).
    Untrack(K),
    /// Send a text frame to the venue.
    Send(String),
}

/// Everything a feed must define about its venue: identity, what to track,
/// what to say on connect, and how to read frames. One instance lives for
/// the driver's whole life (across reconnects).
pub trait FeedProtocol: Send {
    /// Stream-key type (carries source/asset/kind identity).
    type Key: StreamKey;
    /// Skip-reason type for warn-gating.
    type Reason: Copy + Eq + Hash + std::fmt::Display + Send;
    /// Runtime command type (`feed-rtds` subscribes/unsubscribes at runtime;
    /// a feed without runtime control uses an uninhabited type).
    type Command: Send;

    /// Short feed name for the `feed = …` log field (`"rtds"`, `"binance"`).
    fn name(&self) -> &'static str;

    /// The tracked streams and their staleness thresholds. Read once at
    /// startup to seed the machine; runtime changes go through
    /// [`Self::on_command`].
    fn keys(&self) -> Vec<(Self::Key, DurationMs)>;

    /// Messages to send immediately after every (re)connect, in order.
    /// Empty for venues that subscribe via the URL.
    fn connect_messages(&self) -> Vec<String>;

    /// Classifies one inbound text frame. `now` is the same instant the
    /// driver will stamp as `ts_local` on any resulting ticks — a venue
    /// whose payload carries no source timestamp uses it for `ts_exchange`
    /// so the two are consistent.
    fn handle_frame(
        &mut self,
        text: &str,
        now: TimestampMs,
    ) -> FrameOutcome<Self::Key, Self::Reason>;

    /// Translates one runtime command into ordered driver actions.
    fn on_command(&mut self, command: Self::Command) -> Vec<CommandAction<Self::Key>>;
}

/// Everything [`run`] needs. Construct one per feed instance.
pub struct DriverArgs<P: FeedProtocol, T, F> {
    /// Connection parameters.
    pub params: DriverParams,
    /// The venue protocol.
    pub protocol: P,
    /// The transport (real: [`crate::WsTransport`]; tests script a fake).
    pub transport: T,
    /// Wall-clock source; the only clock surface in the crate.
    pub now_fn: F,
    /// The internal bus. Sends are awaited — backpressure is explicit.
    pub bus_tx: mpsc::Sender<Event>,
    /// Runtime commands; `None` when the feed has no runtime control.
    pub command_rx: Option<mpsc::Receiver<P::Command>>,
    /// Optional status snapshots (CLI table, dashboard later).
    pub status_tx: Option<watch::Sender<FeedStatus<P::Key>>>,
    /// Flip to `true` for graceful shutdown.
    pub shutdown_rx: watch::Receiver<bool>,
    /// Backoff jitter seed; `None` seeds from the clock. Tests inject one
    /// for determinism.
    pub backoff_seed: Option<u64>,
}

/// Runs the feed until shutdown.
///
/// # Errors
/// [`FeedError::BusClosed`] when every bus receiver is gone. Connection
/// trouble is logged and retried forever, never returned.
pub async fn run<P, T, F>(args: DriverArgs<P, T, F>) -> Result<(), FeedError>
where
    P: FeedProtocol,
    T: Transport,
    F: Fn() -> TimestampMs + Send,
{
    let DriverArgs {
        params,
        mut protocol,
        mut transport,
        now_fn,
        bus_tx,
        mut command_rx,
        status_tx,
        mut shutdown_rx,
        backoff_seed,
    } = args;

    let feed = protocol.name();
    let seed = backoff_seed.unwrap_or_else(|| now_fn().as_millis() as u64 ^ 0x9E37_79B9_7F4A_7C15);
    let mut machine = FeedMachine::new(protocol.keys(), now_fn());
    let mut backoff = Backoff::new(params.backoff, seed);
    let mut out: Vec<Output> = Vec::new();
    let mut warn_gate = WarnGate::default();
    let mut reconnects: u32 = 0;
    let mut first_connect = true;

    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }
        push_status(
            &status_tx,
            &machine,
            ConnState::Connecting,
            reconnects,
            now_fn(),
        );
        let mut conn = tokio::select! {
            res = transport.connect(&params.url, params.connect_timeout) => match res {
                Ok(conn) => conn,
                Err(error) => {
                    tracing::warn!(target: "feed", feed, %error, url = %params.url, "connect failed");
                    push_status(&status_tx, &machine, ConnState::Backoff, reconnects, now_fn());
                    if sleep_backoff(feed, &mut backoff, &mut shutdown_rx).await {
                        return Ok(());
                    }
                    continue;
                }
            },
            () = wait_shutdown(&mut shutdown_rx) => return Ok(()),
        };

        if first_connect {
            first_connect = false;
        } else {
            reconnects = reconnects.saturating_add(1);
        }
        tracing::info!(target: "feed", feed, url = %params.url, reconnects, "connected");
        machine.on_connected(now_fn());

        let mut subscribed = true;
        for message in protocol.connect_messages() {
            if let Err(error) = conn.send_text(&message).await {
                tracing::warn!(target: "feed", feed, %error, "subscribe send failed");
                subscribed = false;
                break;
            }
        }

        let disconnect_reason = if subscribed {
            push_status(
                &status_tx,
                &machine,
                ConnState::Connected,
                reconnects,
                now_fn(),
            );
            stream_episode(StreamCtx {
                conn: &mut conn,
                protocol: &mut protocol,
                machine: &mut machine,
                out: &mut out,
                warn_gate: &mut warn_gate,
                backoff: &mut backoff,
                now_fn: &now_fn,
                bus_tx: &bus_tx,
                command_rx: &mut command_rx,
                status_tx: &status_tx,
                shutdown_rx: &mut shutdown_rx,
                keepalive: params.keepalive.as_ref(),
                dead_after: params.dead_after,
                reconnects,
            })
            .await?
        } else {
            Some("subscribe send failed".to_owned())
        };

        conn.close().await;
        let Some(reason) = disconnect_reason else {
            tracing::info!(target: "feed", feed, "shutdown requested");
            return Ok(());
        };
        tracing::warn!(target: "feed", feed, reason = %reason, "disconnected — reconnecting");
        machine.on_disconnected(now_fn(), &mut out);
        flush(feed, &mut out, &bus_tx).await?;
        push_status(
            &status_tx,
            &machine,
            ConnState::Backoff,
            reconnects,
            now_fn(),
        );
        if sleep_backoff(feed, &mut backoff, &mut shutdown_rx).await {
            return Ok(());
        }
    }
}

/// Borrowed context for one connection episode (keeps [`stream_episode`]'s
/// signature sane).
struct StreamCtx<'a, P: FeedProtocol, C, F> {
    conn: &'a mut C,
    protocol: &'a mut P,
    machine: &'a mut FeedMachine<P::Key>,
    out: &'a mut Vec<Output>,
    warn_gate: &'a mut WarnGate<P::Reason>,
    backoff: &'a mut Backoff,
    now_fn: &'a F,
    bus_tx: &'a mpsc::Sender<Event>,
    command_rx: &'a mut Option<mpsc::Receiver<P::Command>>,
    status_tx: &'a Option<watch::Sender<FeedStatus<P::Key>>>,
    shutdown_rx: &'a mut watch::Receiver<bool>,
    keepalive: Option<&'a Keepalive>,
    dead_after: Duration,
    reconnects: u32,
}

/// Streams one connected episode. Returns `Ok(Some(reason))` on disconnect
/// (reconnect), `Ok(None)` on shutdown, `Err` only on a dropped bus.
async fn stream_episode<P, C, F>(ctx: StreamCtx<'_, P, C, F>) -> Result<Option<String>, FeedError>
where
    P: FeedProtocol,
    C: Connection,
    F: Fn() -> TimestampMs,
{
    let StreamCtx {
        conn,
        protocol,
        machine,
        out,
        warn_gate,
        backoff,
        now_fn,
        bus_tx,
        command_rx,
        status_tx,
        shutdown_rx,
        keepalive,
        dead_after,
        reconnects,
    } = ctx;
    let feed = protocol.name();

    let mut keepalive_timer = keepalive.map(|k| {
        let mut interval = tokio::time::interval(k.interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        interval.reset(); // don't fire immediately after subscribing
        interval
    });
    let mut watchdog = tokio::time::interval(WATCHDOG_TICK);
    watchdog.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut last_frame = tokio::time::Instant::now();
    let mut proven = false; // backoff resets on the first real data, not on connect

    loop {
        tokio::select! {
            frame = conn.recv() => {
                let text = match frame {
                    Some(Ok(WsFrame::Text(text))) => text,
                    // Binary frames are parsed defensively as (lossy) UTF-8 —
                    // genuine UTF-8 parses normally, junk classifies as
                    // malformed through the protocol's own reason.
                    Some(Ok(WsFrame::Binary(bytes))) => String::from_utf8_lossy(&bytes).into_owned(),
                    // Server pings are liveness evidence only (the transport
                    // already queued the pong reply).
                    Some(Ok(WsFrame::Ping)) => {
                        last_frame = tokio::time::Instant::now();
                        continue;
                    }
                    Some(Err(error)) => return Ok(Some(format!("socket error: {error}"))),
                    None => return Ok(Some("peer closed".to_owned())),
                };
                last_frame = tokio::time::Instant::now();
                let now = now_fn();
                match protocol.handle_frame(&text, now) {
                    FrameOutcome::Ack => {
                        tracing::debug!(target: "feed", feed, "subscription ack");
                    }
                    FrameOutcome::Keepalive => {}
                    FrameOutcome::Prices(observations) => {
                        if !proven {
                            proven = true;
                            backoff.reset();
                        }
                        for obs in observations {
                            machine.on_price(obs.key, obs.value, obs.ts_exchange, now, out);
                        }
                        flush(feed, out, bus_tx).await?;
                    }
                    FrameOutcome::IgnoredQuiet(reason) => {
                        tracing::trace!(target: "feed", feed, %reason, "skipped frame");
                    }
                    FrameOutcome::Ignored(reason) => warn_gate.warn(feed, reason, &text, now),
                }
            }
            () = tick_opt(&mut keepalive_timer) => {
                if let Some(k) = keepalive
                    && let Err(error) = conn.send_text(&k.text).await
                {
                    return Ok(Some(format!("keepalive send failed: {error}")));
                }
            }
            _ = watchdog.tick() => {
                let now = now_fn();
                machine.on_tick(now, out);
                flush(feed, out, bus_tx).await?;
                if last_frame.elapsed() >= dead_after {
                    return Ok(Some(format!(
                        "no frames for {dead_after:?} — assuming dead socket"
                    )));
                }
                // Self-heal a decayed server-side subscription: one stream
                // starved while others stream means resubscribing is the fix
                // — recycle the connection (anchors refresh on reconnect, so
                // recycles are spaced by the full threshold).
                if let Some(breach) = machine.worst_breach(now, STALE_RECYCLE_MULTIPLE) {
                    return Ok(Some(format!(
                        "stream starved for {}ms (threshold {}ms) — recycling connection to resubscribe",
                        breach.age.as_millis(),
                        breach.threshold.as_millis()
                    )));
                }
                push_status(status_tx, machine, ConnState::Connected, reconnects, now);
            }
            cmd = recv_command(command_rx) => match cmd {
                Some(command) => {
                    for action in protocol.on_command(command) {
                        match action {
                            CommandAction::Track { key, stale_after } => {
                                tracing::info!(target: "feed", feed, stream = %key, "subscribing (runtime)");
                                machine.subscribe(key, stale_after, now_fn());
                            }
                            CommandAction::Untrack(key) => {
                                tracing::info!(target: "feed", feed, stream = %key, "unsubscribing (runtime)");
                                machine.unsubscribe(key);
                            }
                            CommandAction::Send(message) => {
                                if let Err(error) = conn.send_text(&message).await {
                                    return Ok(Some(format!("command send failed: {error}")));
                                }
                            }
                        }
                    }
                }
                None => {
                    tracing::debug!(target: "feed", feed, "command channel closed — continuing without runtime control");
                    *command_rx = None;
                }
            },
            () = wait_shutdown(shutdown_rx) => return Ok(None),
        }
    }
}

/// Ticks the keepalive timer, or never resolves when the feed has none.
async fn tick_opt(timer: &mut Option<tokio::time::Interval>) {
    match timer {
        Some(interval) => {
            interval.tick().await;
        }
        None => std::future::pending().await,
    }
}

/// Drains machine outputs to the bus (and the log).
async fn flush(
    feed: &'static str,
    out: &mut Vec<Output>,
    bus_tx: &mpsc::Sender<Event>,
) -> Result<(), FeedError> {
    for output in out.drain(..) {
        let event = match output {
            Output::Publish(tick) => Event::PriceTick(tick),
            Output::Health(health) => {
                match health {
                    FeedHealth::Stale {
                        source,
                        asset,
                        kind,
                        age,
                    } => tracing::warn!(
                        target: "feed",
                        feed,
                        ?source,
                        %asset,
                        ?kind,
                        age_ms = age.as_millis(),
                        "stream STALE"
                    ),
                    FeedHealth::Recovered {
                        source,
                        asset,
                        kind,
                        gap,
                    } => tracing::info!(
                        target: "feed",
                        feed,
                        ?source,
                        %asset,
                        ?kind,
                        gap_ms = gap.as_millis(),
                        "stream recovered"
                    ),
                }
                Event::FeedHealth(health)
            }
        };
        bus_tx.send(event).await.map_err(|_| FeedError::BusClosed)?;
    }
    Ok(())
}

/// Publishes a status snapshot, if anyone is watching.
fn push_status<K: StreamKey>(
    status_tx: &Option<watch::Sender<FeedStatus<K>>>,
    machine: &FeedMachine<K>,
    connection: ConnState,
    reconnects: u32,
    at: TimestampMs,
) {
    if let Some(tx) = status_tx {
        tx.send_replace(FeedStatus {
            at,
            connection,
            reconnects,
            keys: machine.status(),
        });
    }
}

/// Sleeps the next backoff delay; `true` means shutdown arrived mid-sleep.
async fn sleep_backoff(
    feed: &'static str,
    backoff: &mut Backoff,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> bool {
    let delay = backoff.next_delay();
    tracing::info!(
        target: "feed",
        feed,
        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
        "reconnect backoff"
    );
    tokio::select! {
        () = tokio::time::sleep(delay) => false,
        () = wait_shutdown(shutdown_rx) => true,
    }
}

/// Resolves when shutdown is requested (or the control side vanished, which
/// we treat the same — better to stop than run unsupervised).
async fn wait_shutdown(shutdown_rx: &mut watch::Receiver<bool>) {
    let _ = shutdown_rx.wait_for(|&stop| stop).await;
}

/// Receives one command, or never resolves when no channel is attached.
async fn recv_command<C>(rx: &mut Option<mpsc::Receiver<C>>) -> Option<C> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// Rate-limits skip warnings per reason: first occurrence logs immediately
/// (with a frame preview), repeats collapse into one warning per
/// [`WARN_EVERY`] carrying the suppressed count.
#[derive(Debug)]
struct WarnGate<R> {
    gates: HashMap<R, Gate>,
}

// Manual impl: a derive would demand `R: Default`.
impl<R> Default for WarnGate<R> {
    fn default() -> Self {
        Self {
            gates: HashMap::new(),
        }
    }
}

#[derive(Debug)]
struct Gate {
    last_emit: TimestampMs,
    suppressed: u64,
}

impl<R: Copy + Eq + Hash + std::fmt::Display> WarnGate<R> {
    fn warn(&mut self, feed: &'static str, reason: R, raw: &str, now: TimestampMs) {
        match self.gates.get_mut(&reason) {
            None => {
                tracing::warn!(
                    target: "feed",
                    feed,
                    %reason,
                    preview = %preview(raw),
                    "skipping unparseable frame"
                );
                self.gates.insert(
                    reason,
                    Gate {
                        last_emit: now,
                        suppressed: 0,
                    },
                );
            }
            Some(gate) => {
                if now.signed_duration_since(gate.last_emit) >= WARN_EVERY {
                    tracing::warn!(
                        target: "feed",
                        feed,
                        %reason,
                        preview = %preview(raw),
                        suppressed = gate.suppressed,
                        "skipping unparseable frames"
                    );
                    gate.last_emit = now;
                    gate.suppressed = 0;
                } else {
                    gate.suppressed = gate.suppressed.saturating_add(1);
                }
            }
        }
    }
}

/// Short loggable preview of a frame (harness precedent).
fn preview(raw: &str) -> String {
    raw.trim().chars().take(160).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warn_gate_suppresses_within_window_and_reports_count() {
        // The gate is pure bookkeeping; we assert its state transitions (the
        // tracing side effects are observed manually / in integration runs).
        let mut gate: WarnGate<&'static str> = WarnGate::default();
        let t0 = TimestampMs::from_millis(0);
        gate.warn("test", "unknown topic", "{}", t0);
        assert_eq!(gate.gates["unknown topic"].suppressed, 0);

        for i in 1..=5 {
            gate.warn("test", "unknown topic", "{}", TimestampMs::from_millis(i));
        }
        assert_eq!(gate.gates["unknown topic"].suppressed, 5);

        // Past the window: emits again and resets the counter.
        gate.warn(
            "test",
            "unknown topic",
            "{}",
            TimestampMs::from_millis(10_001),
        );
        let g = &gate.gates["unknown topic"];
        assert_eq!(g.suppressed, 0);
        assert_eq!(g.last_emit, TimestampMs::from_millis(10_001));

        // Distinct reasons gate independently.
        gate.warn(
            "test",
            "malformed JSON",
            "junk",
            TimestampMs::from_millis(10_002),
        );
        assert_eq!(gate.gates.len(), 2);
    }

    #[test]
    fn preview_truncates_long_frames() {
        let long = "x".repeat(500);
        assert_eq!(preview(&long).chars().count(), 160);
        assert_eq!(preview("  short  "), "short");
    }
}
