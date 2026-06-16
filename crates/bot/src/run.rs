//! `bot run` — the main run mode (CLAUDE.md §5): one command starts paper trading
//! on all enabled series concurrently under a supervision tree.
//!
//! Topology (the `bot dashboard` wiring, with the real engine in place of the
//! trivial quoter): scheduler + feed-clob + feed-rtds + feed-binance + the §11
//! clock-skew monitor publish onto one bus; a multi-asset [`ModelRuntime`] prices
//! every active window; the single-gateway [`RiskManager`] (owning the quote
//! manager + the momentum and late-window takers, behind its breakers) is the
//! only thing that reaches the [`PaperVenue`]; a [`journal::Recorder`] captures
//! every event; and the axum dashboard serves it all (analytics ride free through
//! [`DashboardHandle::project`]). Real data, paper money.
//!
//! What this mode adds over the smoke subcommands:
//! - **Supervision.** Every long-running task is restarted with backoff if it
//!   dies, while the rest keep running. A critical dependency's death triggers an
//!   immediate cancel-all (the §11 evacuation) and gates new order flow until the
//!   dependency is back (the [`RiskManager`]'s own staleness breakers backstop it).
//! - **Resilient startup self-check.** The bot refuses to *trade* until the clock
//!   is sane (the skew monitor's verdict), discovery has a current window for every
//!   enabled series (the scheduler's `Open` announcements), and the feeds are
//!   healthy (first ticks on the bus) — but it stays up retrying and auto-arms once
//!   healthy. It never exits on a slow/failed self-check.
//! - **Graceful shutdown** on Ctrl-C *and* SIGTERM: stop strategies → cancel all
//!   open orders (draining until zero remain) → flush the journal → exit.
//! - **Resource discipline.** Bounded channels with explicit overflow policy, and
//!   a periodic RSS report + settled-inventory prune so a 24/7 session is memory
//!   stable.
//!
//! [`risk_params`] and the sub-bundle mappers are the §4 `config → engine`
//! boundary maps (deferred by every prior engine task to "the engine-bus-wiring
//! task" — this is it); the engine never depends on `config`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use config::{AppConfig, EngineParams, RunConfig, Secrets};
use core_types::{
    BreakerKind, ControlEvent, Dollars, Event, MarketInfo, MarketLifecycleEvent, Mode, OrderId,
    PriceSource, RiskEvent, Series, TimestampMs, WindowId, WindowLifecycle,
};
use dashboard::{ControlRequest, DashboardHandle};
use discovery::DiscoveryService;
use engine::{
    InventoryEffect, InventoryManager, LateWindowTakerParams, MomentumTakerParams,
    NormalizerParams, QuoteManagerParams, QuoteParams, RiskManager, RiskParams,
};
use feed_binance::{BinanceArgs, BinanceParams, BinanceSub};
use feed_clob::{ClobArgs, ClobParams};
use feed_rtds::{FeedSub, RtdsArgs, RtdsParams};
use feed_util::{Backoff, BackoffParams, WsTransport};
use journal::Recorder;
use scheduler::{SchedulerArgs, Timing};
use timeutil::{
    NtpOffsetSource, NtpParams, SkewMonitorArgs, SkewParams, SystemClock, run_skew_monitor,
    wall_now,
};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use venue_api::{VenueEvent, VenueEvents, VenuePort};
use venue_paper::PaperVenue;

use crate::boot::journal_params;
use crate::control::{ControlPlane, Decision, RuntimeControlState, VenueAction};
use crate::dashboard::params_view;
use crate::feed::{binance_params, clob_params, rtds_params};
use crate::model_runtime::ModelRuntime;
use crate::paper::paper_params;
use crate::timecfg::{ntp_params, skew_params, std_duration};

// ---- cadences + capacities -------------------------------------------------

/// WebSocket broadcast backlog per dashboard subscriber.
const WS_BROADCAST_CAP: usize = 1_024;
/// Central bus capacity (await-send backpressure; a retained sender keeps it open).
const BUS_CAP: usize = 256;
/// Window-announcement (→ clob) and market-lifecycle (clob → scheduler) capacity.
const WINDOW_CAP: usize = 64;
/// Control-request channel capacity (dashboard → loop).
const CONTROL_CAP: usize = 16;
/// Wallet/ledger/risk sampling cadence (drives the equity curve + risk panel).
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
/// Model recompute + publish cadence (covers τ-decay between ticks).
const MODEL_HEARTBEAT: Duration = Duration::from_millis(100);
/// Readiness re-evaluation cadence while not yet armed.
const READINESS_PERIOD: Duration = Duration::from_secs(1);
/// Most current windows cached for the clob re-seed (≈2 per series × 6).
const MAX_CACHED_WINDOWS: usize = 32;
/// Settled-inventory retention: prune settled windows whose open is older than
/// this (well beyond the longest window + linger so nothing live is dropped).
const INVENTORY_RETENTION_MS: i64 = 2 * 3_600_000;
/// How long to wait at shutdown for resting paper orders to terminalize.
const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
/// Assumed memory page size for the Linux RSS report (good enough for a trend).
#[cfg(target_os = "linux")]
const PAGE_KB: u64 = 4;

// ============================================================================
// config → engine parameter boundary maps (§4; the engine never depends on config)
// ============================================================================

/// Maps `config.engine.defaults` into the quoting calculator's params. The four
/// fields without a config key (`cancel_rtt_secs`, `atm_band`, the vol-spike
/// pair) keep the engine default — their config exposure is deferred per the
/// [`QuoteParams`] docs.
#[must_use]
pub(crate) fn quote_params(e: &EngineParams) -> QuoteParams {
    QuoteParams {
        min_edge: e.min_edge,
        k1_vol_multiplier: e.k1_vol_multiplier,
        expected_hold_secs: e.expected_hold_secs,
        gamma_inventory_skew: e.gamma_inventory_skew,
        touch_size: e.touch_size,
        ladder_levels: e.ladder_levels,
        ladder_size_per_level: e.ladder_size_per_level,
        ladder_tick_offset: e.ladder_tick_offset,
        pair_cost_threshold: e.pair_cost_threshold,
        no_atm_final_secs: e.no_atm_final_secs,
        no_passive_final_secs: e.no_passive_final_secs,
        soft_cap_excess: e.soft_cap_excess_shares,
        hard_cap_excess: e.hard_cap_excess_shares,
        max_worst_case_loss: e.max_worst_case_loss_per_window,
        ..QuoteParams::default()
    }
}

/// Maps `config.engine.defaults` into the quote-manager's convergence params
/// (`min_requote_interval_ms` / `max_batch` have no config key — engine default).
#[must_use]
pub(crate) fn quote_manager_params(e: &EngineParams) -> QuoteManagerParams {
    QuoteManagerParams {
        reprice_threshold_theta: e.reprice_threshold_theta,
        cancel_market_theta: e.cancel_market_theta,
        ..QuoteManagerParams::default()
    }
}

/// Maps `config.engine.defaults` into the momentum taker's params (the signal
/// lookback/sigma-mult and the taker-rebate fraction have no config key yet).
#[must_use]
pub(crate) fn momentum_params(e: &EngineParams) -> MomentumTakerParams {
    MomentumTakerParams {
        momentum_buffer: e.taker_momentum_buffer,
        budget_per_window: e.taker_budget_per_window,
        cooldown_ms: e.taker_cooldown_ms.as_millis(),
        ..MomentumTakerParams::default()
    }
}

/// Maps `config.engine.defaults` into the late-window taker's params (every field
/// has a config key).
#[must_use]
pub(crate) fn late_window_params(e: &EngineParams) -> LateWindowTakerParams {
    LateWindowTakerParams {
        tau_threshold_secs: e.late_window_tau_secs,
        certainty_threshold: e.late_certainty_threshold,
        price_cap: e.late_taker_price_cap,
        budget_per_window: e.taker_budget_per_window,
        cooldown_ms: e.taker_cooldown_ms.as_millis(),
    }
}

/// The order normalizer's params. `size_decimals` has no config key (the venue's
/// true share precision must be verified live, per the normalizer docs), so the
/// engine default (whole shares) stands.
#[must_use]
pub(crate) fn normalizer_params(_e: &EngineParams) -> NormalizerParams {
    NormalizerParams::default()
}

/// Maps `config.risk` + `config.engine.defaults` into the [`RiskManager`]'s full
/// parameter bundle. `engine_restart_cooldown_ms` has no config key — the engine
/// default (a few seconds after the last 425/503) stands.
#[must_use]
pub(crate) fn risk_params(config: &AppConfig) -> RiskParams {
    let r = &config.risk;
    let e = &config.engine.defaults;
    RiskParams {
        feed_staleness_ms: r.feed_staleness_ms.as_millis(),
        daily_stop_loss: r.daily_stop_loss,
        max_open_notional: r.max_open_notional,
        sanity_bound: r.sanity_bound,
        sanity_bound_duration_ms: r.sanity_bound_duration_ms.as_millis(),
        error_breaker_max_errors: r.error_breaker_max_errors,
        error_breaker_window_ms: r.error_breaker_window_ms.as_millis(),
        engine_restart_cooldown_ms: RiskParams::default().engine_restart_cooldown_ms,
        quoter_enabled: true,
        momentum_enabled: true,
        late_window_enabled: true,
        quote_manager: quote_manager_params(e),
        quote: quote_params(e),
        normalizer: normalizer_params(e),
        momentum: momentum_params(e),
        late_window: late_window_params(e),
    }
}

/// The per-series worst-case-loss caps the [`RiskManager`] enforces, mapped from
/// each enabled series' resolved engine params.
#[must_use]
pub(crate) fn series_caps(config: &AppConfig, enabled: &[Series]) -> HashMap<Series, Dollars> {
    enabled
        .iter()
        .map(|&s| (s, config.engine.resolved(s).max_worst_case_loss_per_window))
        .collect()
}

// ============================================================================
// supervision
// ============================================================================

/// Which supervised task this is. Critical tasks (the feeds + scheduler) trigger
/// a cancel-all-first on death; the skew monitor does not (its death only stops
/// clock monitoring).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    Scheduler,
    Clob,
    Rtds,
    Binance,
    Skew,
}

impl Task {
    const fn name(self) -> &'static str {
        match self {
            Task::Scheduler => "scheduler",
            Task::Clob => "clob",
            Task::Rtds => "rtds",
            Task::Binance => "binance",
            Task::Skew => "skew",
        }
    }

    /// Whether the engine's order flow depends on this task (so its death must
    /// cancel-all and gate new orders until it recovers).
    const fn critical(self) -> bool {
        !matches!(self, Task::Skew)
    }
}

/// Per-task restart-with-backoff state.
struct SupState {
    backoff: Backoff,
    restart_at: Option<Instant>,
    spawned_at: Instant,
}

impl SupState {
    fn new(params: BackoffParams, seed: u64, now: Instant) -> Self {
        Self {
            backoff: Backoff::new(params, seed),
            restart_at: None,
            spawned_at: now,
        }
    }

    /// Whether this task's scheduled restart is due at `now`.
    fn due(&self, now: Instant) -> bool {
        self.restart_at.is_some_and(|at| now >= at)
    }

    /// Records that the task exited at `now`: resets the backoff if it ran
    /// healthy at least `stable`, then schedules the next restart. Returns the
    /// delay until that restart.
    fn schedule_restart(&mut self, now: Instant, stable: Duration) -> Duration {
        if now.duration_since(self.spawned_at) >= stable {
            self.backoff.reset();
        }
        let delay = self.backoff.next_delay();
        self.restart_at = Some(now + delay);
        delay
    }

    /// Marks the task freshly (re)spawned at `now`.
    fn restarted(&mut self, now: Instant) {
        self.restart_at = None;
        self.spawned_at = now;
    }
}

// ============================================================================
// readiness (the resilient startup self-check, derived from bus events)
// ============================================================================

/// The startup self-check state, folded from the bus: clock sanity comes from the
/// §11 skew monitor's `ClockSkew` breaker, discovery from the scheduler's `Open`
/// announcements, feed health from the first ticks of each feed.
#[derive(Debug, Default)]
struct Readiness {
    fast_feed: bool,
    chainlink: bool,
    book: bool,
    open_series: HashSet<Series>,
    clockskew_tripped: bool,
}

impl Readiness {
    /// Folds one bus event into the readiness flags.
    fn track(&mut self, event: &Event) {
        match event {
            Event::PriceTick(t) => match t.source {
                PriceSource::BinanceDirect => self.fast_feed = true,
                PriceSource::ChainlinkRtds => self.chainlink = true,
                PriceSource::BinanceRtds => {}
            },
            Event::Book(_) => self.book = true,
            Event::Window {
                market,
                lifecycle: WindowLifecycle::Open,
            } => {
                self.open_series.insert(market.window.series);
            }
            Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew,
            }) => self.clockskew_tripped = true,
            Event::Risk(RiskEvent::BreakerCleared {
                breaker: BreakerKind::ClockSkew,
            }) => self.clockskew_tripped = false,
            _ => {}
        }
    }

    fn clock_ok(&self, run: &RunConfig, boot_elapsed: Duration) -> bool {
        !run.require_clock_check
            || (boot_elapsed.as_millis() as i64 >= run.clock_check_grace_ms.as_millis()
                && !self.clockskew_tripped)
    }

    fn discovery_ok(&self, enabled: &[Series], run: &RunConfig) -> bool {
        !run.require_discovery_check || enabled.iter().all(|s| self.open_series.contains(s))
    }

    fn feeds_ok(&self) -> bool {
        self.fast_feed && self.chainlink && self.book
    }

    /// Whether all three gates are satisfied — the bot may begin trading.
    fn trade_ready(&self, enabled: &[Series], run: &RunConfig, boot_elapsed: Duration) -> bool {
        self.clock_ok(run, boot_elapsed) && self.discovery_ok(enabled, run) && self.feeds_ok()
    }

    /// A human-readable list of what is still missing (for the loud warning).
    fn missing(&self, enabled: &[Series], run: &RunConfig, boot_elapsed: Duration) -> Vec<&str> {
        let mut m = Vec::new();
        if !self.clock_ok(run, boot_elapsed) {
            m.push("clock");
        }
        if !self.discovery_ok(enabled, run) {
            m.push("discovery (current windows)");
        }
        if !self.fast_feed {
            m.push("binance feed");
        }
        if !self.chainlink {
            m.push("chainlink feed");
        }
        if !self.book {
            m.push("clob book");
        }
        m
    }
}

/// The window cache used to re-seed a restarted clob (so it connects to the
/// current windows immediately rather than waiting for the next announcement).
fn cache_window(
    cache: &mut HashMap<WindowId, (Arc<MarketInfo>, WindowLifecycle)>,
    market: &Arc<MarketInfo>,
    lifecycle: WindowLifecycle,
) {
    if matches!(lifecycle, WindowLifecycle::Resolved { .. }) {
        cache.remove(&market.window);
        return;
    }
    cache.insert(market.window, (Arc::clone(market), lifecycle));
    if cache.len() > MAX_CACHED_WINDOWS
        && let Some(oldest) = cache
            .keys()
            .min_by_key(|w| w.open_time.as_millis())
            .copied()
    {
        cache.remove(&oldest);
    }
}

// ============================================================================
// entry point
// ============================================================================

/// Builds the runtime and runs the main paper-trading orchestrator until a
/// shutdown signal (Ctrl-C or SIGTERM).
pub fn execute(
    config: &AppConfig,
    secrets: &Secrets,
    series: Option<Series>,
) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run(config, secrets, series))
}

#[allow(clippy::too_many_lines, reason = "the orchestrator owns every channel")]
async fn run(
    config: &AppConfig,
    secrets: &Secrets,
    series_filter: Option<Series>,
) -> anyhow::Result<()> {
    let series_list: Vec<Series> = match series_filter {
        Some(s) => vec![s],
        None => config.engine.enabled_series(),
    };
    if series_list.is_empty() {
        anyhow::bail!("no enabled series to trade (enable some in config or pass --series)");
    }
    let bind = config.dashboard.bind;
    let token = secrets
        .dashboard_token
        .as_ref()
        .map(|s| s.expose().to_owned());

    tracing::info!(
        target: "run",
        %bind,
        series = ?series_list.iter().map(|s| s.key()).collect::<Vec<_>>(),
        "bot run starting (real data, paper money; ctrl-c / SIGTERM to stop)"
    );

    // Restore prior inventory + order state from the journal (§3/§9); the live
    // engine wiring that *seeds* the engine from it is a documented follow-up.
    let _restored = crate::boot::rebuild_and_log(config);

    // Control plane (single venue owner) + dashboard handle.
    let mut control = ControlPlane::new(config, secrets);
    let control_state: RuntimeControlState = control.state.clone();
    let (req_tx, mut req_rx) = mpsc::channel::<ControlRequest>(CONTROL_CAP);
    let handle = DashboardHandle::new(WS_BROADCAST_CAP, wall_now()).with_request_sink(req_tx);
    handle.set_session(Mode::Paper, true, wall_now());
    handle.set_params(params_view(config), wall_now());
    handle.set_control_state(control.snapshot(), wall_now());

    // Journal recorder.
    let recorder = Recorder::spawn(journal_params(config), wall_now)
        .context("spawning the journal recorder")?;

    // Dashboard server (bind up front so a bind failure surfaces immediately).
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding dashboard to {bind}"))?;
    let local = listener.local_addr().unwrap_or(bind);
    let (server_shutdown_tx, server_shutdown_rx) = oneshot::channel::<()>();
    let server_handle = handle.clone();
    let server_token = token.clone();
    let server_task = tokio::spawn(async move {
        dashboard::serve_with_listener(server_handle, listener, server_token, async move {
            let _ = server_shutdown_rx.await;
        })
        .await
    });
    tracing::info!(target: "run", %local, "dashboard listening (waiting to arm — self-check pending)");

    // The bus + its retained senders (so `bus_rx.recv()` never closes mid-run).
    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(BUS_CAP);
    let bus_keepalive = bus_tx.clone();
    let model_tx = bus_tx.clone();

    // Cross-task channels. `window_tx`/`sched_market_tx` are re-created when their
    // consumer (clob / scheduler) restarts; `clob_market_rx` is the stable hub
    // input that forwards clob's market-lifecycle events to the current scheduler.
    let (mut window_tx, window_rx) =
        mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(WINDOW_CAP);
    let (clob_market_tx, mut clob_market_rx) = mpsc::channel::<MarketLifecycleEvent>(WINDOW_CAP);
    let (mut sched_market_tx, sched_market_rx) = mpsc::channel::<MarketLifecycleEvent>(WINDOW_CAP);
    let (shutdown_tx, _shutdown_rx0) = watch::channel(false);

    // The engine, model, venue, and the orchestrator's own inventory (for the
    // dashboard's settlements — distinct from the risk manager's internal one).
    let mut risk = RiskManager::new(risk_params(config), series_caps(config, &series_list));
    let mut model = ModelRuntime::new(config, &series_list).context("building model runtime")?;
    let mut venue = PaperVenue::spawn(paper_params(config), wall_now);
    let mut venue_rx = venue
        .take_event_rx()
        .context("paper venue event stream already taken")?;
    let mut inventory = InventoryManager::new();
    let mut working: HashSet<OrderId> = HashSet::new();
    let mut current_windows: HashMap<WindowId, (Arc<MarketInfo>, WindowLifecycle)> = HashMap::new();

    // Supervision state.
    let backoff = BackoffParams {
        initial: std_duration(config.run.supervision_initial_backoff_ms),
        max: std_duration(config.run.supervision_max_backoff_ms),
        multiplier: config.run.supervision_backoff_multiplier,
    };
    let stable = Duration::from_secs(config.run.stable_secs.max(1).unsigned_abs());
    let boot_at = Instant::now();
    let mut sched_sup = SupState::new(backoff, 0x51, boot_at);
    let mut clob_sup = SupState::new(backoff, 0xC1, boot_at);
    let mut rtds_sup = SupState::new(backoff, 0x52, boot_at);
    let mut binance_sup = SupState::new(backoff, 0xB1, boot_at);
    let mut skew_sup = SupState::new(backoff, 0x53, boot_at);

    // Initial spawns.
    let mut sched_handle = build_discovery(config).map(|service| {
        spawn_scheduler(
            Timing::from_config(&config.scheduler),
            series_list.clone(),
            service,
            bus_tx.clone(),
            sched_market_rx,
            shutdown_tx.subscribe(),
        )
    });
    if sched_handle.is_none() {
        sched_sup.restart_at = Some(boot_at);
    }
    let mut clob_handle = Some(spawn_clob(
        clob_params(&config.feeds),
        bus_tx.clone(),
        window_rx,
        clob_market_tx.clone(),
        shutdown_tx.subscribe(),
    ));
    let mut rtds_handle = Some(spawn_rtds(
        rtds_params(&config.feeds),
        bus_tx.clone(),
        shutdown_tx.subscribe(),
    ));
    let mut binance_handle = Some(spawn_binance(
        binance_params(&config.feeds),
        bus_tx.clone(),
        shutdown_tx.subscribe(),
    ));
    let mut skew_handle = Some(spawn_skew(
        skew_params(&config.clock),
        std_duration(config.clock.check_interval_ms),
        ntp_params(&config.clock),
        bus_tx.clone(),
        shutdown_tx.subscribe(),
    ));

    // Timers.
    let mut sample = interval(SAMPLE_PERIOD);
    let mut heartbeat = interval(MODEL_HEARTBEAT);
    let mut readiness_tick = interval(READINESS_PERIOD);
    let mut risk_tick = interval(std_duration(config.run.risk_tick_ms));
    let mut rss_tick = interval(Duration::from_secs(
        config.run.rss_report_secs.max(1).unsigned_abs(),
    ));

    let mut readiness = Readiness::default();
    let mut armed = false;
    let mut next_warn_at = boot_at + std_duration(config.run.readiness_timeout_ms);

    loop {
        // Order flow is allowed only once armed and while every critical
        // dependency is alive (a dead feed/scheduler gates placement until it
        // restarts — the §11 "cancel-all for any engine whose dependencies died"
        // is the immediate cancel-all in `handle_exit`; this is the standing gate).
        let order_path_open = armed
            && !control_state.is_halted()
            && sched_handle.is_some()
            && clob_handle.is_some()
            && rtds_handle.is_some()
            && binance_handle.is_some();

        let restart_deadline = [&sched_sup, &clob_sup, &rtds_sup, &binance_sup, &skew_sup]
            .iter()
            .filter_map(|s| s.restart_at)
            .min()
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3_600));

        tokio::select! {
            res = join_opt(&mut sched_handle) => {
                sched_handle = None;
                handle_exit(Task::Scheduler, res, &mut sched_sup, &venue, stable).await;
            }
            res = join_opt(&mut clob_handle) => {
                clob_handle = None;
                handle_exit(Task::Clob, res, &mut clob_sup, &venue, stable).await;
            }
            res = join_opt(&mut rtds_handle) => {
                rtds_handle = None;
                handle_exit(Task::Rtds, res, &mut rtds_sup, &venue, stable).await;
            }
            res = join_opt(&mut binance_handle) => {
                binance_handle = None;
                handle_exit(Task::Binance, res, &mut binance_sup, &venue, stable).await;
            }
            res = join_opt(&mut skew_handle) => {
                skew_handle = None;
                handle_exit(Task::Skew, res, &mut skew_sup, &venue, stable).await;
            }
            () = tokio::time::sleep_until(restart_deadline) => {
                let now = Instant::now();
                // Scheduler: rebuild discovery + a fresh market channel.
                if sched_handle.is_none() && sched_sup.due(now) {
                    match build_discovery(config) {
                        Some(service) => {
                            let (tx, rx) = mpsc::channel::<MarketLifecycleEvent>(WINDOW_CAP);
                            sched_market_tx = tx;
                            sched_handle = Some(spawn_scheduler(
                                Timing::from_config(&config.scheduler),
                                series_list.clone(),
                                service,
                                bus_tx.clone(),
                                rx,
                                shutdown_tx.subscribe(),
                            ));
                            sched_sup.restarted(now);
                            tracing::info!(target: "run", "scheduler restarted");
                        }
                        None => {
                            sched_sup.restart_at =
                                Some(now + sched_sup.backoff.next_delay());
                            tracing::warn!(target: "run", "scheduler restart deferred (discovery build failed)");
                        }
                    }
                }
                // Clob: a fresh window channel + re-seed the current windows.
                if clob_handle.is_none() && clob_sup.due(now) {
                    let (tx, rx) = mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(WINDOW_CAP);
                    window_tx = tx;
                    clob_handle = Some(spawn_clob(
                        clob_params(&config.feeds),
                        bus_tx.clone(),
                        rx,
                        clob_market_tx.clone(),
                        shutdown_tx.subscribe(),
                    ));
                    clob_sup.restarted(now);
                    for (m, lc) in current_windows.values() {
                        let _ = window_tx.try_send((Arc::clone(m), *lc));
                    }
                    tracing::info!(
                        target: "run", windows = current_windows.len(),
                        "clob restarted + re-seeded"
                    );
                }
                if rtds_handle.is_none() && rtds_sup.due(now) {
                    rtds_handle = Some(spawn_rtds(
                        rtds_params(&config.feeds),
                        bus_tx.clone(),
                        shutdown_tx.subscribe(),
                    ));
                    rtds_sup.restarted(now);
                    tracing::info!(target: "run", "rtds restarted");
                }
                if binance_handle.is_none() && binance_sup.due(now) {
                    binance_handle = Some(spawn_binance(
                        binance_params(&config.feeds),
                        bus_tx.clone(),
                        shutdown_tx.subscribe(),
                    ));
                    binance_sup.restarted(now);
                    tracing::info!(target: "run", "binance restarted");
                }
                if skew_handle.is_none() && skew_sup.due(now) {
                    skew_handle = Some(spawn_skew(
                        skew_params(&config.clock),
                        std_duration(config.clock.check_interval_ms),
                        ntp_params(&config.clock),
                        bus_tx.clone(),
                        shutdown_tx.subscribe(),
                    ));
                    skew_sup.restarted(now);
                    tracing::info!(target: "run", "skew monitor restarted");
                }
            }
            maybe = bus_rx.recv() => match maybe {
                Some(event) => {
                    let now = wall_now();
                    recorder.record(&event);
                    if let Event::Window { market, lifecycle } = &event {
                        let _ = window_tx.try_send((Arc::clone(market), *lifecycle));
                        cache_window(&mut current_windows, market, *lifecycle);
                    }
                    venue.on_bus_event(&event).await;
                    handle.project(Mode::Paper, &event, now);
                    model.on_event(&event);
                    readiness.track(&event);
                    // The §11 safety events (ClockSkew from the skew monitor) reach
                    // the risk manager even before arming; market-data quoting does
                    // not. Re-feeding the manager's own published breaker kinds is a
                    // no-op (its core ignores them), so no double-react.
                    let safety = matches!(event, Event::Risk(_) | Event::Control(_));
                    if order_path_open || safety {
                        risk.on_event(&event, &venue, now).await;
                        drain_risk(&mut risk, &recorder, &handle, now);
                    }
                    for effect in inventory.on_event(&event) {
                        publish_inventory_effect(&handle, &recorder, effect, now);
                    }
                }
                None => unreachable!("a bus sender is retained for the loop's lifetime"),
            },
            maybe = venue_rx.recv() => {
                if let Some(ve) = maybe {
                    handle_venue_event(
                        &ve, &venue, &mut risk, &mut inventory, &mut working,
                        &recorder, &handle, wall_now(),
                    ).await;
                }
            }
            ev = clob_market_rx.recv() => {
                if let Some(ev) = ev {
                    // The market hub: forward clob's deduped market-lifecycle event
                    // to the current scheduler (drop+count on a full channel).
                    let _ = sched_market_tx.try_send(ev);
                }
            }
            maybe = req_rx.recv() => {
                if let Some(req) = maybe {
                    apply_control(
                        req, &mut control, &venue, &mut risk, &recorder, &handle, wall_now(),
                    ).await;
                }
            }
            _ = readiness_tick.tick(), if !armed => {
                let now_inst = Instant::now();
                let boot_elapsed = now_inst.duration_since(boot_at);
                if readiness.trade_ready(&series_list, &config.run, boot_elapsed) {
                    armed = true;
                    let now = wall_now();
                    tracing::info!(target: "run", "ARMED — startup self-check passed; trading enabled");
                    // Replay the current windows so the risk manager knows the
                    // active window for each series from the first armed tick.
                    for (m, lc) in current_windows.values() {
                        risk.on_event(
                            &Event::Window { market: Arc::clone(m), lifecycle: *lc },
                            &venue, now,
                        ).await;
                    }
                    drain_risk(&mut risk, &recorder, &handle, now);
                } else if now_inst >= next_warn_at {
                    next_warn_at = now_inst + std_duration(config.run.readiness_timeout_ms);
                    tracing::warn!(
                        target: "run",
                        missing = ?readiness.missing(&series_list, &config.run, boot_elapsed),
                        "still NOT trading — startup self-check unsatisfied (staying up, retrying)"
                    );
                }
            }
            _ = risk_tick.tick() => {
                if order_path_open {
                    let now = wall_now();
                    risk.on_tick(&venue, now).await;
                    drain_risk(&mut risk, &recorder, &handle, now);
                }
            }
            _ = heartbeat.tick() => {
                model.on_heartbeat(&model_tx, wall_now());
            }
            _ = sample.tick() => {
                let now = wall_now();
                let wallet = venue.balances().await.unwrap_or_default();
                handle.set_wallet(Mode::Paper, wallet, now);
                handle.set_paper_ledger(Mode::Paper, venue.ledger_snapshot(), now);
                handle.set_risk(Mode::Paper, risk.state_snapshot(), now);
            }
            _ = rss_tick.tick() => {
                let cutoff = TimestampMs::from_millis(
                    wall_now().as_millis().saturating_sub(INVENTORY_RETENTION_MS),
                );
                let pruned_inv = inventory.prune_settled_before(cutoff);
                let pruned_risk = risk.prune_settled_before(cutoff);
                tracing::info!(
                    target: "run",
                    rss_kb = read_rss_kb(),
                    inventory_windows = inventory.len(),
                    cached_windows = current_windows.len(),
                    working_orders = working.len(),
                    pruned_inv,
                    pruned_risk,
                    armed,
                    "resource report"
                );
            }
            r = shutdown_signal() => {
                r.context("listening for shutdown signal")?;
                tracing::info!(target: "run", "shutdown signal — halting strategies, cancelling orders");
                break;
            }
        }
    }

    // ---- graceful shutdown (§5): stop strategies → cancel-all → flush → exit --
    armed = false;
    let _ = armed; // strategies are no longer driven below; keep the intent explicit.
    handle.set_session(Mode::Paper, false, wall_now());

    // Cancel every resting paper order, then drain the venue stream until the
    // working set is empty (proves "zero open paper orders"), bounded by a timeout.
    let report = venue.cancel_all().await;
    tracing::info!(
        target: "run",
        canceled = report.as_ref().map(|r| r.canceled.len()).unwrap_or(0),
        "cancel-all issued at shutdown"
    );
    let drained = drain_open_orders(&mut venue_rx, &mut working).await;
    if working.is_empty() {
        tracing::info!(target: "run", drained, "all paper orders terminalized — zero open orders");
    } else {
        tracing::warn!(
            target: "run",
            still_open = working.len(),
            "shutdown drain timed out with orders still open"
        );
    }
    venue.shutdown();

    // Signal the supervised tasks, release the bus senders, then drain + join.
    let _ = shutdown_tx.send(true);
    drop(bus_tx);
    drop(bus_keepalive);
    drop(model_tx);
    let mut bus_open = true;
    while sched_handle.is_some()
        || clob_handle.is_some()
        || rtds_handle.is_some()
        || binance_handle.is_some()
        || skew_handle.is_some()
    {
        tokio::select! {
            _ = join_opt(&mut sched_handle), if sched_handle.is_some() => { sched_handle = None; }
            _ = join_opt(&mut clob_handle), if clob_handle.is_some() => { clob_handle = None; }
            _ = join_opt(&mut rtds_handle), if rtds_handle.is_some() => { rtds_handle = None; }
            _ = join_opt(&mut binance_handle), if binance_handle.is_some() => { binance_handle = None; }
            _ = join_opt(&mut skew_handle), if skew_handle.is_some() => { skew_handle = None; }
            maybe = bus_rx.recv(), if bus_open => { bus_open = maybe.is_some(); }
        }
    }

    // Stop the dashboard server.
    let _ = server_shutdown_tx.send(());
    match server_task.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(target: "run", error = %e, "dashboard server error"),
        Err(e) => tracing::warn!(target: "run", error = %e, "dashboard server task panicked"),
    }

    // Flush + finalize the journal.
    match recorder.finish() {
        Ok(stats) => tracing::info!(
            target: "run",
            records = stats.records,
            indexed = stats.indexed,
            dropped = stats.dropped,
            "journal flushed"
        ),
        Err(e) => tracing::warn!(target: "run", error = %e, "journal flush error"),
    }
    tracing::info!(target: "run", "bot run shut down cleanly");
    Ok(())
}

// ============================================================================
// helpers
// ============================================================================

/// Builds an interval whose missed ticks are delayed (not bursted).
fn interval(period: Duration) -> tokio::time::Interval {
    let mut i = tokio::time::interval(period);
    i.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    i
}

/// Awaits an `Option<JoinHandle>`, pending forever when `None` so a dead/awaiting
/// task's select arm simply never fires (no poll-after-completion).
async fn join_opt(task: &mut Option<JoinHandle<()>>) -> Result<(), tokio::task::JoinError> {
    match task.as_mut() {
        Some(t) => t.await,
        None => std::future::pending().await,
    }
}

/// Logs a supervised task's exit, performs the cancel-all-first for a critical
/// dependency, and schedules the restart with backoff.
async fn handle_exit(
    task: Task,
    res: Result<(), tokio::task::JoinError>,
    sup: &mut SupState,
    venue: &PaperVenue,
    stable: Duration,
) {
    match res {
        Ok(()) => tracing::warn!(target: "run", task = task.name(), "supervised task exited"),
        Err(e) if e.is_panic() => {
            tracing::error!(target: "run", task = task.name(), "supervised task PANICKED");
        }
        Err(_) => tracing::warn!(target: "run", task = task.name(), "supervised task cancelled"),
    }
    if task.critical() {
        let report = venue.cancel_all().await;
        tracing::warn!(
            target: "run",
            task = task.name(),
            ok = report.is_ok(),
            "cancel-all-first on dependency death; order flow gated until recovery"
        );
    }
    let delay = sup.schedule_restart(Instant::now(), stable);
    tracing::info!(
        target: "run",
        task = task.name(),
        backoff_ms = delay.as_millis(),
        "scheduling restart"
    );
}

/// Drains the risk manager's published breaker events onto the journal + dashboard.
fn drain_risk(
    risk: &mut RiskManager,
    recorder: &Recorder,
    handle: &DashboardHandle,
    now: TimestampMs,
) {
    for ev in risk.take_published() {
        recorder.record(&ev);
        handle.project(Mode::Paper, &ev, now);
    }
}

/// Handles one venue order/fill event: drives the risk manager (authoritative for
/// its inventory/notional), tracks the working-order set, journals + projects the
/// item, and folds fills into the orchestrator's inventory.
#[allow(
    clippy::too_many_arguments,
    reason = "the orchestrator threads its components explicitly, as the dashboard loop does"
)]
async fn handle_venue_event(
    ve: &VenueEvent,
    venue: &PaperVenue,
    risk: &mut RiskManager,
    inventory: &mut InventoryManager,
    working: &mut HashSet<OrderId>,
    recorder: &Recorder,
    handle: &DashboardHandle,
    now: TimestampMs,
) {
    risk.on_venue_event(ve, venue, now).await;
    drain_risk(risk, recorder, handle, now);
    match ve {
        VenueEvent::Order(u) => {
            if u.state.is_terminal() {
                working.remove(&u.order_id);
            } else {
                working.insert(u.order_id.clone());
            }
            let event = Event::OrderUpdate(Arc::clone(u));
            recorder.record(&event);
            handle.project(Mode::Paper, &event, now);
        }
        VenueEvent::Fill(f) => {
            let event = Event::Fill(Arc::clone(f));
            recorder.record(&event);
            handle.project(Mode::Paper, &event, now);
            for effect in inventory.on_event(&event) {
                publish_inventory_effect(handle, recorder, effect, now);
            }
        }
        VenueEvent::Connectivity { connected } => {
            handle.set_ws_connected(Mode::Paper, *connected, now);
        }
    }
}

/// Publishes an inventory effect as the matching bus event to the journal + dashboard.
fn publish_inventory_effect(
    handle: &DashboardHandle,
    recorder: &Recorder,
    effect: InventoryEffect,
    now: TimestampMs,
) {
    let event = match effect {
        InventoryEffect::Snapshot(snapshot) => Event::Inventory(Arc::new(snapshot)),
        InventoryEffect::Settled(summary) => Event::Settlement(Arc::new(summary)),
    };
    recorder.record(&event);
    handle.project(Mode::Paper, &event, now);
}

/// Applies one control-plane request: performs the venue side effect, drives the
/// risk manager with the control events (the authoritative breaker source via
/// `take_published`), journals + projects, fills the capital readback, and replies.
async fn apply_control(
    req: ControlRequest,
    control: &mut ControlPlane,
    venue: &PaperVenue,
    risk: &mut RiskManager,
    recorder: &Recorder,
    handle: &DashboardHandle,
    now: TimestampMs,
) {
    let Decision {
        mut outcome,
        events,
        venue_action,
    } = control.decide(req.command, req.origin, now);

    let capital_change = matches!(
        venue_action,
        Some(VenueAction::SetCapital(_) | VenueAction::AdjustCapital(_))
    );
    match venue_action {
        Some(VenueAction::CancelAll) => {
            let _ = venue.cancel_all().await;
        }
        Some(VenueAction::SetCapital(amount)) => venue.set_capital(amount),
        Some(VenueAction::AdjustCapital(delta)) => venue.adjust_capital(Dollars::new(delta)),
        None => {}
    }

    for event in &events {
        match event {
            // The control plane pre-bakes Risk events for the dashboard-only path;
            // here the risk manager mints the authoritative breakers from the
            // Control event, so skip these to avoid duplicates.
            Event::Risk(_) => {}
            Event::Control(_) => {
                recorder.record(event);
                handle.project(Mode::Paper, event, now);
                risk.on_event(event, venue, now).await;
            }
            _ => recorder.record(event),
        }
    }
    drain_risk(risk, recorder, handle, now);

    let capital = venue.balances().await.ok().map(|w| w.collateral_total);
    if capital_change && let Some(amount) = capital {
        let ev = Event::Control(ControlEvent::PaperCapitalSet { amount });
        recorder.record(&ev);
        handle.project(Mode::Paper, &ev, now);
        let wallet = venue.balances().await.unwrap_or_default();
        handle.set_wallet(Mode::Paper, wallet, now);
        handle.set_paper_ledger(Mode::Paper, venue.ledger_snapshot(), now);
    }

    outcome.state.paper_capital = capital;
    handle.set_control_state(outcome.state.clone(), now);
    if !outcome.accepted() {
        tracing::warn!(target: "control", kind = ?outcome.kind, "control command refused");
    }
    let _ = req.reply.send(outcome);
}

/// Drains the venue event stream after a cancel-all until the working-order set is
/// empty or the drain timeout elapses. Returns the number of events drained.
async fn drain_open_orders(
    venue_rx: &mut mpsc::Receiver<VenueEvent>,
    working: &mut HashSet<OrderId>,
) -> u64 {
    let mut drained = 0u64;
    let deadline = tokio::time::Instant::now() + SHUTDOWN_DRAIN_TIMEOUT;
    while !working.is_empty() {
        tokio::select! {
            maybe = venue_rx.recv() => match maybe {
                Some(VenueEvent::Order(u)) => {
                    drained += 1;
                    if u.state.is_terminal() {
                        working.remove(&u.order_id);
                    }
                }
                Some(_) => drained += 1,
                None => break,
            },
            () = tokio::time::sleep_until(deadline) => break,
        }
    }
    drained
}

/// Reports resident set size in KiB on Linux (dependency-free `/proc/self/statm`),
/// `None` elsewhere. Assumes a 4 KiB page — good enough for a memory trend.
#[cfg(target_os = "linux")]
fn read_rss_kb() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident_pages * PAGE_KB)
}

/// RSS reporting is Linux-only (the deployment target); the dev box reports `None`.
#[cfg(not(target_os = "linux"))]
fn read_rss_kb() -> Option<u64> {
    None
}

/// Awaits Ctrl-C or (on unix) SIGTERM — the graceful-shutdown trigger.
async fn shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).context("installing SIGTERM handler")?;
        tokio::select! {
            r = tokio::signal::ctrl_c() => r.context("listening for ctrl-c")?,
            _ = term.recv() => {}
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .context("listening for ctrl-c")
    }
}

// ---- spawn builders (each wraps a driver, logging on failure) --------------

/// Builds a production discovery service from config, logging on a build failure.
fn build_discovery(config: &AppConfig) -> Option<DiscoveryService<discovery::HttpClient>> {
    match DiscoveryService::from_config(&config.feeds, &config.discovery) {
        Ok(s) => Some(s),
        Err(error) => {
            tracing::error!(target: "run", %error, "discovery service build failed");
            None
        }
    }
}

fn spawn_scheduler(
    timing: Timing,
    series: Vec<Series>,
    service: DiscoveryService<discovery::HttpClient>,
    bus_tx: mpsc::Sender<Event>,
    market_rx: mpsc::Receiver<MarketLifecycleEvent>,
    shutdown_rx: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(error) = scheduler::run(SchedulerArgs {
            timing,
            series,
            refresher: service,
            now_fn: wall_now,
            bus_tx,
            market_rx: Some(market_rx),
            status_tx: None,
            shutdown_rx,
        })
        .await
        {
            tracing::error!(target: "run", task = "scheduler", %error, "driver failed");
        }
    })
}

fn spawn_clob(
    params: ClobParams,
    bus_tx: mpsc::Sender<Event>,
    window_rx: mpsc::Receiver<(Arc<MarketInfo>, WindowLifecycle)>,
    market_tx: mpsc::Sender<MarketLifecycleEvent>,
    shutdown_rx: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(error) = feed_clob::run(ClobArgs {
            params,
            transport_factory: || WsTransport,
            now_fn: wall_now,
            bus_tx,
            window_rx,
            market_tx: Some(market_tx),
            command_rx: None,
            status_tx: None,
            shutdown_rx,
            backoff_seed: None,
        })
        .await
        {
            tracing::error!(target: "run", task = "clob", %error, "feed failed");
        }
    })
}

fn spawn_rtds(
    params: RtdsParams,
    bus_tx: mpsc::Sender<Event>,
    shutdown_rx: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(error) = feed_rtds::run(RtdsArgs {
            params,
            subscriptions: FeedSub::all(),
            transport: WsTransport,
            now_fn: wall_now,
            bus_tx,
            command_rx: None,
            status_tx: None,
            shutdown_rx,
            backoff_seed: None,
        })
        .await
        {
            tracing::error!(target: "run", task = "rtds", %error, "feed failed");
        }
    })
}

fn spawn_binance(
    params: BinanceParams,
    bus_tx: mpsc::Sender<Event>,
    shutdown_rx: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(error) = feed_binance::run(BinanceArgs {
            params,
            subscriptions: BinanceSub::all(),
            transport: WsTransport,
            now_fn: wall_now,
            bus_tx,
            status_tx: None,
            shutdown_rx,
            backoff_seed: None,
        })
        .await
        {
            tracing::error!(target: "run", task = "binance", %error, "feed failed");
        }
    })
}

fn spawn_skew(
    params: SkewParams,
    check_interval: Duration,
    ntp: NtpParams,
    bus_tx: mpsc::Sender<Event>,
    shutdown_rx: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(error) = run_skew_monitor(SkewMonitorArgs {
            params,
            check_interval,
            source: NtpOffsetSource::new(ntp, SystemClock::new()),
            clock: SystemClock::new(),
            bus_tx,
            shutdown_rx,
        })
        .await
        {
            tracing::warn!(target: "run", task = "skew", %error, "skew monitor failed");
        }
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use core_types::{Asset, WindowDuration};

    fn series(asset: Asset, duration: WindowDuration) -> Series {
        Series { asset, duration }
    }

    // ---- config → engine boundary maps (anti-drift) ------------------------

    #[test]
    fn risk_params_maps_config_and_equals_default_for_defaults() {
        let c = AppConfig::default();
        let p = risk_params(&c);
        // Spot-check the explicit mappings.
        assert_eq!(p.feed_staleness_ms, c.risk.feed_staleness_ms.as_millis());
        assert_eq!(p.daily_stop_loss, c.risk.daily_stop_loss);
        assert_eq!(p.max_open_notional, c.risk.max_open_notional);
        assert_eq!(p.error_breaker_max_errors, c.risk.error_breaker_max_errors);
        assert_eq!(p.quote.min_edge, c.engine.defaults.min_edge);
        assert_eq!(
            p.quote.soft_cap_excess,
            c.engine.defaults.soft_cap_excess_shares
        );
        assert_eq!(
            p.quote.hard_cap_excess,
            c.engine.defaults.hard_cap_excess_shares
        );
        assert_eq!(
            p.quote_manager.reprice_threshold_theta,
            c.engine.defaults.reprice_threshold_theta
        );
        assert_eq!(
            p.momentum.budget_per_window,
            c.engine.defaults.taker_budget_per_window
        );
        assert_eq!(
            p.momentum.cooldown_ms,
            c.engine.defaults.taker_cooldown_ms.as_millis()
        );
        assert_eq!(
            p.late_window.tau_threshold_secs,
            c.engine.defaults.late_window_tau_secs
        );
        assert_eq!(
            p.late_window.price_cap,
            c.engine.defaults.late_taker_price_cap
        );
        assert!(p.quoter_enabled && p.momentum_enabled && p.late_window_enabled);
        // The strongest anti-drift pin: the committed defaults must map exactly
        // onto the engine defaults, so the whole bundle equals RiskParams::default().
        assert_eq!(p, RiskParams::default());
    }

    #[test]
    fn series_caps_maps_every_enabled_series() {
        let c = AppConfig::default();
        let enabled = c.engine.enabled_series();
        let caps = series_caps(&c, &enabled);
        assert_eq!(caps.len(), enabled.len());
        for s in &enabled {
            assert_eq!(
                caps[s],
                c.engine.resolved(*s).max_worst_case_loss_per_window
            );
        }
    }

    // ---- readiness gate ----------------------------------------------------

    fn btc5() -> Series {
        series(Asset::Btc, WindowDuration::M5)
    }

    #[test]
    fn readiness_requires_all_three_gates() {
        let run = RunConfig::default();
        let enabled = vec![btc5()];
        let grace = std_duration(run.clock_check_grace_ms);

        // Nothing seen, before the clock grace → not ready; clock + discovery +
        // feeds are all missing.
        let empty = Readiness::default();
        let before = Duration::from_millis(0);
        assert!(!empty.trade_ready(&enabled, &run, before));
        let miss = empty.missing(&enabled, &run, before);
        assert!(miss.contains(&"clock"));
        assert!(miss.contains(&"discovery (current windows)"));
        assert!(miss.contains(&"binance feed"));

        // Feeds up, an Open window seen, past the clock grace, no skew → ready.
        let mut r = Readiness {
            fast_feed: true,
            chainlink: true,
            book: true,
            open_series: HashSet::from([btc5()]),
            ..Readiness::default()
        };
        assert!(r.trade_ready(&enabled, &run, grace));

        // A tripped ClockSkew un-arms (refuse to trade until the clock recovers).
        r.clockskew_tripped = true;
        assert!(!r.trade_ready(&enabled, &run, grace));
        assert!(r.missing(&enabled, &run, grace).contains(&"clock"));

        // Before the grace elapses, the clock is not yet "ok" even un-tripped.
        r.clockskew_tripped = false;
        assert!(!r.trade_ready(&enabled, &run, Duration::from_millis(0)));
    }

    #[test]
    fn readiness_gates_can_be_disabled() {
        let run = RunConfig {
            require_clock_check: false,
            require_discovery_check: false,
            ..RunConfig::default()
        };
        let enabled = vec![btc5()];
        // No Open window, before grace, but both gates disabled → feeds alone arm.
        let r = Readiness {
            fast_feed: true,
            chainlink: true,
            book: true,
            ..Readiness::default()
        };
        assert!(r.trade_ready(&enabled, &run, Duration::from_millis(0)));
    }

    #[test]
    fn readiness_tracks_bus_events() {
        let mut r = Readiness::default();
        r.track(&Event::Risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::ClockSkew,
        }));
        assert!(r.clockskew_tripped);
        r.track(&Event::Risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::ClockSkew,
        }));
        assert!(!r.clockskew_tripped);
    }

    // ---- window cache (clob re-seed) ---------------------------------------

    #[test]
    fn cache_window_inserts_and_drops_on_resolution() {
        let mut cache = HashMap::new();
        let m = market(btc5(), 0);
        cache_window(&mut cache, &m, WindowLifecycle::Open);
        assert_eq!(cache.len(), 1);
        cache_window(&mut cache, &m, WindowLifecycle::Closing);
        assert_eq!(cache.len(), 1); // same window, updated lifecycle
        cache_window(
            &mut cache,
            &m,
            WindowLifecycle::Resolved {
                outcome: core_types::Outcome::Up,
            },
        );
        assert!(cache.is_empty(), "resolved windows are dropped");
    }

    // ---- supervision restart math ------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn schedule_restart_escalates_then_resets_after_stable() {
        let params = BackoffParams {
            initial: Duration::from_millis(1_000),
            max: Duration::from_millis(60_000),
            multiplier: 2.0,
        };
        let now = Instant::now();
        let mut sup = SupState::new(params, 7, now);
        let stable = Duration::from_secs(60);

        // Three rapid exits (well under `stable`) escalate the backoff. With the
        // equal-jitter policy each delay ∈ [raw/2, raw], raw = initial·2^n, so by
        // the third the delay exceeds the initial cap.
        let mut last = Duration::ZERO;
        for _ in 0..3 {
            last = sup.schedule_restart(now, stable);
            sup.restarted(now); // pretend it respawned immediately (no healthy run)
        }
        assert!(
            last > params.initial,
            "escalated delay {last:?} should exceed the initial {:?}",
            params.initial
        );

        // A task that ran healthy past `stable` before dying resets the backoff.
        let later = now + Duration::from_secs(120);
        let reset = sup.schedule_restart(later, stable);
        assert!(
            reset <= params.initial,
            "reset delay {reset:?} should be back within the initial {:?}",
            params.initial
        );
    }

    fn market(s: Series, open_ms: i64) -> Arc<MarketInfo> {
        use core_types::{
            ConditionId, FeeParams, ResolutionSource, Size, TickSize, TokenId, TokenPair,
        };
        use rust_decimal::dec;
        Arc::new(MarketInfo {
            window: WindowId {
                series: s,
                open_time: TimestampMs::from_millis(open_ms),
            },
            event_slug: "test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: TimestampMs::from_millis(open_ms + 300_000),
            strike: Some(dec!(60000)),
            tick_size: TickSize::T001,
            min_order_size: Size::new(dec!(5)).unwrap(),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        })
    }
}
