//! The per-series fleet manager: consumes scheduler window-lifecycle
//! announcements, runs one connection task per window (pre-connecting the
//! successor `presubscribe_lead` before its open so rollover is gap-free by
//! construction), lets closed windows linger for their `market_resolved`,
//! dedups market lifecycle notices across connections (delivery scope is
//! venue-undocumented — broadcast and pair-scoped both handled), and
//! forwards them to the scheduler.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    DurationMs, Event, MarketInfo, MarketLifecycleEvent, TimestampMs, WindowId, WindowLifecycle,
};
use feed_util::Transport;
use tokio::sync::{mpsc, watch};

use crate::driver::{ConnArgs, WindowFeedStatus, run_connection};
use crate::machine::{ConnMachine, MachineParams};
use crate::{ClobParams, FeedError};

/// Supervisor scan cadence: start due connections, expire lingerers, reap
/// finished tasks, push status.
const SCAN_INTERVAL: Duration = Duration::from_secs(1);

/// Dedup TTL for market lifecycle notices (`new_market` may be broadcast to
/// every connection; `market_resolved` reaches at least the subscribed one).
const DEDUP_TTL: DurationMs = DurationMs::from_millis(600_000);

/// Dedup map sweep threshold.
const DEDUP_SWEEP_LEN: usize = 1_024;

/// Runtime commands into the supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClobCommand {
    /// Drop one window's connection once (it reconnects with backoff) — the
    /// live forced-reconnect continuity demo.
    RecycleWindow(WindowId),
}

/// Point-in-time status of the whole clob feed (one entry per live
/// connection task, sorted by window).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClobStatus {
    /// Per-window connection status.
    pub windows: Vec<WindowFeedStatus>,
}

/// Everything [`run`] needs. Construct one per feed instance.
pub struct ClobArgs<TF, F> {
    /// Connection parameters.
    pub params: ClobParams,
    /// Makes one transport per connection task (real:
    /// `|| feed_util::WsTransport`; tests script fakes).
    pub transport_factory: TF,
    /// Wall-clock source; the only clock surface in the crate.
    pub now_fn: F,
    /// The internal bus. Sends are awaited — backpressure is explicit.
    pub bus_tx: mpsc::Sender<Event>,
    /// Scheduler `Event::Window` announcements, forwarded by the binary
    /// (bus fan-out stays an orchestrator concern).
    pub window_rx: mpsc::Receiver<(Arc<MarketInfo>, WindowLifecycle)>,
    /// Deduplicated market lifecycle notices → scheduler `market_rx`.
    /// `None` when no scheduler is attached (events are still deduped and
    /// logged).
    pub market_tx: Option<mpsc::Sender<MarketLifecycleEvent>>,
    /// Runtime commands; `None` until a control plane exists.
    pub command_rx: Option<mpsc::Receiver<ClobCommand>>,
    /// Optional status snapshots (CLI ladder, dashboard later).
    pub status_tx: Option<watch::Sender<ClobStatus>>,
    /// Flip to `true` for graceful shutdown.
    pub shutdown_rx: watch::Receiver<bool>,
    /// Backoff jitter seed; `None` seeds from the clock. Tests inject one
    /// for determinism.
    pub backoff_seed: Option<u64>,
}

/// One spawned connection task.
struct ConnHandle {
    market: Arc<MarketInfo>,
    join: tokio::task::JoinHandle<Result<(), FeedError>>,
    lifecycle_tx: watch::Sender<WindowLifecycle>,
    recycle_tx: mpsc::Sender<()>,
    shutdown_tx: watch::Sender<bool>,
    status_rx: watch::Receiver<WindowFeedStatus>,
    /// Hard teardown time once the window closed (waiting for
    /// `market_resolved`).
    linger_until: Option<TimestampMs>,
}

/// A window known but not yet due for connection.
struct Pending {
    market: Arc<MarketInfo>,
    lifecycle: WindowLifecycle,
    connect_at: TimestampMs,
}

/// TTL'd seen-set for market lifecycle notices, keyed
/// (condition id, event kind).
struct Dedup {
    seen: HashMap<(String, u8), TimestampMs>,
}

impl Dedup {
    fn new() -> Self {
        Self {
            seen: HashMap::new(),
        }
    }

    fn fresh(&mut self, key: (String, u8), now: TimestampMs) -> bool {
        if self.seen.len() >= DEDUP_SWEEP_LEN {
            // Lazy sweep; a sustained flood of >1024 distinct markets per
            // TTL would keep the map large, which is harmless.
            self.seen
                .retain(|_, &mut at| now.signed_duration_since(at) < DEDUP_TTL);
        }
        match self.seen.get(&key) {
            Some(&at) if now.signed_duration_since(at) < DEDUP_TTL => false,
            _ => {
                self.seen.insert(key, now);
                true
            }
        }
    }
}

/// Mutable supervisor state, borrowed by the select-loop handlers.
struct State<TF, F> {
    params: ClobParams,
    transport_factory: TF,
    now_fn: F,
    bus_tx: mpsc::Sender<Event>,
    market_tx: Option<mpsc::Sender<MarketLifecycleEvent>>,
    market_evt_tx: mpsc::Sender<MarketLifecycleEvent>,
    backoff_seed: Option<u64>,
    conns: BTreeMap<WindowId, ConnHandle>,
    pending: BTreeMap<WindowId, Pending>,
    closing: Vec<tokio::task::JoinHandle<Result<(), FeedError>>>,
    dedup: Dedup,
}

/// Runs the clob feed until shutdown.
///
/// # Errors
/// [`FeedError::BusClosed`] when the bus receiver is gone — the process is
/// shutting down or mis-wired. Connection trouble never returns.
pub async fn run<TF, T, F>(args: ClobArgs<TF, F>) -> Result<(), FeedError>
where
    TF: FnMut() -> T + Send,
    T: Transport + 'static,
    F: Fn() -> TimestampMs + Send + Sync + Clone + 'static,
{
    let ClobArgs {
        params,
        transport_factory,
        now_fn,
        bus_tx,
        window_rx,
        market_tx,
        command_rx,
        status_tx,
        mut shutdown_rx,
        backoff_seed,
    } = args;
    // Internal funnel: every connection task's market notices land here for
    // dedup. The supervisor keeps one sender so the channel never closes.
    let (market_evt_tx, mut market_evt_rx) = mpsc::channel::<MarketLifecycleEvent>(64);
    let mut state = State {
        params,
        transport_factory,
        now_fn,
        bus_tx,
        market_tx,
        market_evt_tx,
        backoff_seed,
        conns: BTreeMap::new(),
        pending: BTreeMap::new(),
        closing: Vec::new(),
        dedup: Dedup::new(),
    };
    let mut window_rx = Some(window_rx);
    let mut command_rx = command_rx;
    let mut scan = tokio::time::interval(SCAN_INTERVAL);
    scan.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let result = loop {
        tokio::select! {
            maybe = recv_opt(&mut window_rx) => match maybe {
                Some((market, lifecycle)) => state.on_window(market, lifecycle),
                None => {
                    tracing::warn!(
                        target: "feed", feed = "clob",
                        "window lifecycle source closed — no new windows will be tracked"
                    );
                    window_rx = None;
                }
            },
            maybe = market_evt_rx.recv() => {
                // Never `None`: the supervisor holds a sender.
                if let Some(event) = maybe
                    && let Err(error) = state.on_market_event(event).await
                {
                    break Err(error);
                }
            }
            maybe = recv_opt(&mut command_rx) => match maybe {
                Some(command) => state.on_command(command),
                None => command_rx = None,
            },
            _ = scan.tick() => {
                if let Err(error) = state.scan(status_tx.as_ref()).await {
                    break Err(error);
                }
            }
            changed = shutdown_rx.changed() => {
                // A dropped sender (binary gone) is a shutdown too.
                if changed.is_err() || *shutdown_rx.borrow() {
                    break Ok(());
                }
            }
        }
    };
    state.shutdown_all().await;
    result
}

async fn recv_opt<T>(rx: &mut Option<mpsc::Receiver<T>>) -> Option<T> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

impl<TF, T, F> State<TF, F>
where
    TF: FnMut() -> T + Send,
    T: Transport + 'static,
    F: Fn() -> TimestampMs + Send + Sync + Clone + 'static,
{
    /// One scheduler window-lifecycle announcement.
    fn on_window(&mut self, market: Arc<MarketInfo>, lifecycle: WindowLifecycle) {
        let now = (self.now_fn)();
        let window = market.window;
        if let Some(handle) = self.conns.get_mut(&window) {
            // Phase update for a live connection; watch send only fails if
            // the task died, which the scan reaps.
            let _ = handle.lifecycle_tx.send(lifecycle);
            match lifecycle {
                WindowLifecycle::Closed => {
                    let deadline = now.saturating_add(self.params.resolution_linger);
                    handle.linger_until = Some(handle.linger_until.unwrap_or(deadline));
                }
                WindowLifecycle::Resolved { .. } => self.teardown(window, "resolved"),
                _ => {}
            }
            return;
        }
        match lifecycle {
            WindowLifecycle::Discovered | WindowLifecycle::Open | WindowLifecycle::Closing => {
                let connect_at = std::cmp::max(
                    now,
                    market
                        .window
                        .open_time
                        .saturating_add(DurationMs::from_millis(
                            -self.params.presubscribe_lead.as_millis(),
                        )),
                );
                if connect_at <= now {
                    self.spawn_connection(market, lifecycle);
                } else {
                    tracing::debug!(
                        target: "feed", feed = "clob", window = %window,
                        connect_at = connect_at.as_millis(),
                        "window pending — will connect at open − presubscribe_lead"
                    );
                    self.pending.insert(
                        window,
                        Pending {
                            market,
                            lifecycle,
                            connect_at,
                        },
                    );
                }
            }
            // Never connected and already past close: nothing useful left
            // to subscribe for (the scheduler's discovery refresh owns
            // resolution recovery beyond the linger budget).
            WindowLifecycle::Closed | WindowLifecycle::Resolved { .. } => {
                self.pending.remove(&window);
            }
        }
    }

    fn spawn_connection(&mut self, market: Arc<MarketInfo>, lifecycle: WindowLifecycle) {
        let now = (self.now_fn)();
        let window = market.window;
        let (lifecycle_tx, lifecycle_rx) = watch::channel(lifecycle);
        let (recycle_tx, recycle_rx) = mpsc::channel(1);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let machine_params = MachineParams {
            book_stale_after: self.params.book_stale_after,
            publish_interval: self.params.publish_interval,
        };
        let initial_status = WindowFeedStatus {
            window,
            event_slug: market.event_slug.clone(),
            close_time: market.close_time,
            connected: false,
            episodes: 0,
            machine: ConnMachine::new(Arc::clone(&market), machine_params, lifecycle, now)
                .status(now),
        };
        let (status_tx, status_rx) = watch::channel(initial_status);
        tracing::info!(
            target: "feed", feed = "clob", window = %window,
            slug = %market.event_slug, "starting window connection"
        );
        let join = tokio::spawn(run_connection(ConnArgs {
            params: self.params.clone(),
            market: Arc::clone(&market),
            transport: (self.transport_factory)(),
            now_fn: self.now_fn.clone(),
            bus_tx: self.bus_tx.clone(),
            market_tx: self.market_evt_tx.clone(),
            lifecycle_rx,
            recycle_rx,
            status_tx,
            shutdown_rx,
            backoff_seed: self.backoff_seed,
        }));
        self.conns.insert(
            window,
            ConnHandle {
                market,
                join,
                lifecycle_tx,
                recycle_tx,
                shutdown_tx,
                status_rx,
                linger_until: None,
            },
        );
    }

    fn teardown(&mut self, window: WindowId, why: &str) {
        if let Some(handle) = self.conns.remove(&window) {
            tracing::info!(
                target: "feed", feed = "clob", window = %window, why,
                "tearing down window connection"
            );
            let _ = handle.shutdown_tx.send(true);
            self.closing.push(handle.join);
        }
        self.pending.remove(&window);
    }

    /// One deduplicated market lifecycle notice from any connection.
    async fn on_market_event(&mut self, event: MarketLifecycleEvent) -> Result<(), FeedError> {
        let now = (self.now_fn)();
        let (condition, kind) = match &event {
            MarketLifecycleEvent::NewMarket { condition_id, .. } => (condition_id.clone(), 0u8),
            MarketLifecycleEvent::MarketResolved { condition_id, .. } => {
                (condition_id.clone(), 1u8)
            }
        };
        if !self.dedup.fresh((condition.as_str().to_owned(), kind), now) {
            return Ok(());
        }
        match &event {
            MarketLifecycleEvent::NewMarket { slug, .. } => {
                tracing::debug!(
                    target: "feed", feed = "clob", slug = %slug,
                    "new_market notice (refresh hint)"
                );
            }
            MarketLifecycleEvent::MarketResolved { winning_token, .. } => {
                tracing::info!(
                    target: "feed", feed = "clob",
                    condition = condition.as_str(),
                    winning_token = winning_token.as_str(),
                    "market_resolved notice"
                );
            }
        }
        if let Some(tx) = &self.market_tx
            && tx.send(event.clone()).await.is_err()
        {
            tracing::warn!(
                target: "feed", feed = "clob",
                "scheduler market channel closed — no longer forwarding"
            );
            self.market_tx = None;
        }
        // Our own window resolved: its connection has nothing left to do
        // (the scheduler's Resolved announcement would tear it down anyway —
        // this also covers runs with no scheduler attached).
        if let MarketLifecycleEvent::MarketResolved { .. } = &event {
            let resolved = self
                .conns
                .iter()
                .find(|(_, handle)| handle.market.condition_id == condition)
                .map(|(window, _)| *window);
            if let Some(window) = resolved {
                self.teardown(window, "market_resolved seen");
            }
        }
        Ok(())
    }

    fn on_command(&mut self, command: ClobCommand) {
        match command {
            ClobCommand::RecycleWindow(window) => {
                if let Some(handle) = self.conns.get(&window) {
                    tracing::info!(
                        target: "feed", feed = "clob", window = %window,
                        "forced recycle requested"
                    );
                    // Cap-1 channel: a recycle already in flight is enough.
                    let _ = handle.recycle_tx.try_send(());
                } else {
                    tracing::warn!(
                        target: "feed", feed = "clob", window = %window,
                        "forced recycle for unknown window — ignored"
                    );
                }
            }
        }
    }

    /// Periodic housekeeping.
    async fn scan(
        &mut self,
        status_tx: Option<&watch::Sender<ClobStatus>>,
    ) -> Result<(), FeedError> {
        let now = (self.now_fn)();
        // Start pending connections that came due.
        let due: Vec<WindowId> = self
            .pending
            .iter()
            .filter(|(_, p)| p.connect_at <= now)
            .map(|(w, _)| *w)
            .collect();
        for window in due {
            if let Some(pending) = self.pending.remove(&window) {
                self.spawn_connection(pending.market, pending.lifecycle);
            }
        }
        // Expire lingerers: armed deadline, or close_time + linger as the
        // belt-and-braces fallback when the Closed announcement was missed.
        let expired: Vec<WindowId> = self
            .conns
            .iter()
            .filter(|(_, handle)| {
                let fallback = handle
                    .market
                    .close_time
                    .saturating_add(self.params.resolution_linger);
                handle.linger_until.unwrap_or(fallback) <= now
            })
            .map(|(window, _)| *window)
            .collect();
        for window in expired {
            self.teardown(window, "resolution linger expired");
        }
        // Reap tasks that were asked to stop.
        let mut still_closing = Vec::new();
        for join in self.closing.drain(..) {
            if join.is_finished() {
                match join.await {
                    Ok(Ok(())) => {}
                    // The bus can close while we are tearing down — benign.
                    Ok(Err(error)) => tracing::debug!(
                        target: "feed", feed = "clob", error = %error,
                        "closing connection task ended with error"
                    ),
                    Err(join_error) => tracing::error!(
                        target: "feed", feed = "clob", error = %join_error,
                        "closing connection task panicked"
                    ),
                }
            } else {
                still_closing.push(join);
            }
        }
        self.closing = still_closing;
        // A live connection task exiting on its own is fatal wiring trouble
        // (drivers reconnect forever; only a dead bus stops them).
        let dead: Vec<WindowId> = self
            .conns
            .iter()
            .filter(|(_, handle)| handle.join.is_finished())
            .map(|(window, _)| *window)
            .collect();
        for window in dead {
            if let Some(handle) = self.conns.remove(&window) {
                match handle.join.await {
                    Ok(Err(error)) => {
                        tracing::error!(
                            target: "feed", feed = "clob", window = %window, error = %error,
                            "connection task died"
                        );
                        return Err(error);
                    }
                    Ok(Ok(())) => tracing::warn!(
                        target: "feed", feed = "clob", window = %window,
                        "connection task exited unexpectedly"
                    ),
                    Err(join_error) => {
                        tracing::error!(
                            target: "feed", feed = "clob", window = %window, error = %join_error,
                            "connection task panicked"
                        );
                        return Err(FeedError::BusClosed);
                    }
                }
            }
        }
        // Status snapshot.
        if let Some(tx) = status_tx {
            let windows = self
                .conns
                .values()
                .map(|handle| handle.status_rx.borrow().clone())
                .collect();
            let _ = tx.send(ClobStatus { windows });
        }
        Ok(())
    }

    /// Graceful shutdown: stop every task and wait for them.
    async fn shutdown_all(mut self) {
        let windows: Vec<WindowId> = self.conns.keys().copied().collect();
        for window in windows {
            self.teardown(window, "shutdown");
        }
        for join in self.closing.drain(..) {
            let _ = join.await;
        }
    }
}
