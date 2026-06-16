//! `bot fair` — the live fair-value smoke run (read-only): the full model
//! (§8) for one series' current window rendered next to the Polymarket book
//! mid, with calibration records journaled and the self-captured strike
//! verified post-hoc against the venue's resolved `priceToBeat`.
//!
//! Wiring (the `ladder` + `vol` topologies merged onto one bus): the scheduler
//! drives the chosen series; feed-clob supplies books + `market_resolved`;
//! feed-rtds + feed-binance supply price ticks. A per-asset
//! [`model::VolEstimator`] and [`model::BasisTracker`] feed a per-window
//! [`model::FairValueEngine`]. The engine recomputes on every relevant tick
//! and on a 100 ms heartbeat (covering τ-decay between ticks); each computed
//! [`model::FairValue`] is published as `Event::Model` on the bus (via
//! `try_send` — the loop drains its own bus, so an awaited send could
//! deadlock) and rendered in place every 250 ms.
//!
//! Calibration JSONL lands in `data/calibration/fair-{series}-{stamp}.jsonl`:
//! ~1 Hz model snapshots, strike freeze/revision records, resolution outcomes,
//! and post-hoc verification verdicts. The strike the bot captured live is
//! reconciled against `eventMetadata.priceToBeat`, which the venue publishes a
//! few minutes after a window resolves (verified 2026-06-13) — the first run's
//! verdicts confirm which [`model::StrikeRule`] the venue actually uses.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{
    AnchorSource, Asset, BookSnapshot, Event, MarketInfo, ModelHealth, ModelHealthEvent,
    ModelHealthReason, Outcome, PriceSource, Series, TickKind, TimestampMs, TokenId, WindowId,
    WindowLifecycle,
};
use discovery::{DiscoveryApi, DiscoveryService, HttpClient};
use feed_binance::{BinanceArgs, BinanceSub};
use feed_clob::ClobArgs;
use feed_rtds::{FeedSub, RtdsArgs};
use feed_util::{FeedError, WsTransport};
use model::{
    BasisParams, BasisTracker, FairInputs, FairParams, FairValue, FairValueEngine, HealthMonitor,
    HealthParams, HealthTransition, StrikeOutcome, StrikeQuality, StrikeRule, VolEstimator,
};
use rust_decimal::prelude::ToPrimitive;
use scheduler::{SchedulerArgs, Timing};
use serde::Serialize;
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::discover::{fmt_countdown, fmt_ts};
use crate::feed::{binance_params, clob_params, rtds_params, spawn_jsonl_writer};
use crate::vol::vol_params;

/// Heartbeat: recompute + publish cadence (covers τ-decay between ticks).
const HEARTBEAT: Duration = Duration::from_millis(100);
/// In-place redraw cadence.
const RENDER_PERIOD: Duration = Duration::from_millis(250);
/// Minimum spacing between journaled snapshot records (~1 Hz).
const SNAPSHOT_INTERVAL_MS: i64 = 1_000;
/// Fallback: trigger verification this long after close if no `market_resolved`
/// arrived (the primary trigger is the Resolved announcement).
const VERIFY_FALLBACK_MS: i64 = 90_000;
/// Drop a resolved window's engine/book state this long after close.
const PRUNE_AFTER_CLOSE_MS: i64 = 200_000;
/// Verification polling: attempts and spacing (~4 min total — metadata appears
/// 1–4 min post-close).
const VERIFY_ATTEMPTS: u32 = 8;
const VERIFY_POLL: Duration = Duration::from_secs(30);
/// Relative tolerance for matching our strike to the venue's `priceToBeat`
/// (Gamma numbers decode through f64, so exact equality is too strict).
const VERIFY_TOL: f64 = 1e-9;
/// Recent notable events kept in the footer pane.
const EVENT_PANE_LINES: usize = 8;

type FeedResult = Result<Result<(), FeedError>, tokio::task::JoinError>;
type SchedResult = Result<Result<(), scheduler::SchedulerError>, tokio::task::JoinError>;

/// Builds the runtime and runs the fair-value smoke run until ctrl-c.
pub fn execute(config: &AppConfig, series: Series) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run_fair(config, series))
}

async fn run_fair(config: &AppConfig, series: Series) -> anyhow::Result<()> {
    let service = DiscoveryService::from_config(&config.feeds, &config.discovery)
        .context("building discovery service")?;
    let verify_client = HttpClient::new(&config.feeds, &config.discovery)
        .context("building verification HTTP client")?;

    let journal_path = calibration_path(series, wall_now());
    let (cal_tx, cal_rx) = mpsc::unbounded_channel::<CalRecord>();
    let writer = spawn_jsonl_writer(journal_path.clone(), cal_rx)?;
    tracing::info!(
        target: "fair",
        series = series.key(),
        journal = %journal_path.display(),
        "fair-value smoke run starting (read-only; ctrl-c to stop)"
    );

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (window_tx, window_rx) = mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(64);
    let (market_tx, market_rx) = mpsc::channel(64);
    let (verify_tx, mut verify_rx) = mpsc::channel::<VerifyResult>(16);
    // Each driver subscribes its own receiver; this one is the keep-alive root.
    let (shutdown_tx, _shutdown_rx) = watch::channel(false);

    let mut sched_task: JoinHandle<Result<(), scheduler::SchedulerError>> =
        tokio::spawn(scheduler::run(SchedulerArgs {
            timing: Timing::from_config(&config.scheduler),
            series: vec![series],
            refresher: service,
            now_fn: wall_now,
            bus_tx: bus_tx.clone(),
            market_rx: Some(market_rx),
            status_tx: None,
            shutdown_rx: shutdown_tx.subscribe(),
        }));
    let mut clob_task: JoinHandle<Result<(), FeedError>> = tokio::spawn(feed_clob::run(ClobArgs {
        params: clob_params(&config.feeds),
        transport_factory: || WsTransport,
        now_fn: wall_now,
        bus_tx: bus_tx.clone(),
        window_rx,
        market_tx: Some(market_tx),
        command_rx: None,
        status_tx: None,
        shutdown_rx: shutdown_tx.subscribe(),
        backoff_seed: None,
    }));
    let mut rtds_task: JoinHandle<Result<(), FeedError>> = tokio::spawn(feed_rtds::run(RtdsArgs {
        params: rtds_params(&config.feeds),
        subscriptions: FeedSub::all(),
        transport: WsTransport,
        now_fn: wall_now,
        bus_tx: bus_tx.clone(),
        command_rx: None,
        status_tx: None,
        shutdown_rx: shutdown_tx.subscribe(),
        backoff_seed: None,
    }));
    let mut binance_task: JoinHandle<Result<(), FeedError>> =
        tokio::spawn(feed_binance::run(BinanceArgs {
            params: binance_params(&config.feeds),
            subscriptions: BinanceSub::all(),
            transport: WsTransport,
            now_fn: wall_now,
            bus_tx: bus_tx.clone(),
            status_tx: None,
            shutdown_rx: shutdown_tx.subscribe(),
            backoff_seed: None,
        }));
    // Kept in the loop to self-publish Event::Model (a real orchestrator gives
    // the model its own task; here the loop both produces and drains).
    let model_tx = bus_tx;

    let mut state = FairState::new(config, series)?;
    let mut heartbeat = tokio::time::interval(HEARTBEAT);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut render = tokio::time::interval(RENDER_PERIOD);
    render.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let started = wall_now();

    let mut sched_result: Option<SchedResult> = None;
    let mut feed_results: [Option<FeedResult>; 3] = [None, None, None];

    // Main phase: until ctrl-c, or any driver exits on its own (fatal).
    loop {
        tokio::select! {
            joined = &mut sched_task, if sched_result.is_none() => {
                sched_result = Some(joined);
                tracing::warn!(target: "fair", "scheduler exited — shutting down");
                break;
            }
            joined = &mut clob_task, if feed_results[0].is_none() => {
                feed_results[0] = Some(joined);
                tracing::warn!(target: "fair", "clob feed exited — shutting down");
                break;
            }
            joined = &mut rtds_task, if feed_results[1].is_none() => {
                feed_results[1] = Some(joined);
                tracing::warn!(target: "fair", "rtds feed exited — shutting down");
                break;
            }
            joined = &mut binance_task, if feed_results[2].is_none() => {
                feed_results[2] = Some(joined);
                tracing::warn!(target: "fair", "binance feed exited — shutting down");
                break;
            }
            maybe = bus_rx.recv() => match maybe {
                Some(event) => state.on_event(
                    &event, &window_tx, &cal_tx, &verify_client, &verify_tx,
                ),
                None => unreachable!("the loop holds a bus sender (model_tx)"),
            },
            maybe = verify_rx.recv() => {
                if let Some(result) = maybe {
                    state.on_verify(result, &cal_tx);
                }
            }
            _ = heartbeat.tick() => {
                state.on_heartbeat(&model_tx, &cal_tx, &verify_client, &verify_tx);
            }
            _ = render.tick() => state.draw(wall_now())?,
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "fair", "ctrl-c — shutting down");
                break;
            }
        }
    }

    // Drain phase: signal shutdown, keep draining the bus so no driver
    // deadlocks on a full channel while exiting.
    let _ = shutdown_tx.send(true);
    drop(model_tx);
    let mut bus_open = true;
    while sched_result.is_none() || feed_results.iter().any(Option::is_none) {
        tokio::select! {
            joined = &mut sched_task, if sched_result.is_none() => sched_result = Some(joined),
            joined = &mut clob_task, if feed_results[0].is_none() => feed_results[0] = Some(joined),
            joined = &mut rtds_task, if feed_results[1].is_none() => feed_results[1] = Some(joined),
            joined = &mut binance_task, if feed_results[2].is_none() => {
                feed_results[2] = Some(joined);
            }
            maybe = bus_rx.recv(), if bus_open => bus_open = maybe.is_some(),
        }
    }

    state.draw(wall_now())?;
    let ran_for = wall_now().signed_duration_since(started);
    println!(
        "\nrun summary: {} ticks, {} snapshots published ({} dropped), {} journaled, \
         {} resolutions, verify {}/{} exact ({} mismatch, {} unavailable) over {}",
        state.ticks,
        state.published,
        state.dropped,
        state.journaled,
        state.resolved,
        state.verify_exact,
        state.verify_exact + state.verify_mismatch + state.verify_unavailable,
        state.verify_mismatch,
        state.verify_unavailable,
        fmt_countdown(ran_for),
    );
    println!("calibration journal: {}", journal_path.display());

    match writer.join() {
        Ok(Ok(lines)) => println!("journal: {lines} records written"),
        Ok(Err(error)) => tracing::warn!(target: "fair", %error, "journal writer failed"),
        Err(_) => tracing::warn!(target: "fair", "journal writer panicked"),
    }

    if let Some(result) = sched_result {
        result
            .context("scheduler driver panicked")?
            .context("scheduler driver failed")?;
    }
    for (name, result) in [
        ("clob", feed_results[0].take()),
        ("rtds", feed_results[1].take()),
        ("binance", feed_results[2].take()),
    ] {
        if let Some(result) = result {
            result
                .with_context(|| format!("{name} feed driver panicked"))?
                .with_context(|| format!("{name} feed driver failed"))?;
        }
    }
    Ok(())
}

/// The captured strike candidates handed to a verification task at trigger
/// time.
#[derive(Debug, Clone, Copy, Default)]
struct StrikeCandidates {
    chosen: Option<f64>,
    before: Option<f64>,
    after: Option<f64>,
}

/// One post-hoc verification outcome, sent back to the main loop.
#[derive(Debug, Clone)]
struct VerifyResult {
    window: WindowId,
    slug: String,
    candidates: StrikeCandidates,
    gamma_price_to_beat: Option<f64>,
    gamma_final_price: Option<f64>,
    matched: &'static str,
    verdict: &'static str,
    delta_chosen_rel: Option<f64>,
}

/// Reconciles our captured candidates with the venue's resolved `priceToBeat`.
/// Pure — unit-tested.
fn reconcile(
    window: WindowId,
    slug: String,
    candidates: StrikeCandidates,
    gamma_price_to_beat: Option<f64>,
    gamma_final_price: Option<f64>,
) -> VerifyResult {
    let Some(ptb) = gamma_price_to_beat else {
        return VerifyResult {
            window,
            slug,
            candidates,
            gamma_price_to_beat: None,
            gamma_final_price,
            matched: "none",
            verdict: "unavailable",
            delta_chosen_rel: None,
        };
    };
    let matches = |v: Option<f64>| v.is_some_and(|x| (x - ptb).abs() / ptb.abs() <= VERIFY_TOL);
    let matched = match (matches(candidates.before), matches(candidates.after)) {
        (true, true) => "both",
        (true, false) => "before",
        (false, true) => "after",
        (false, false) => "none",
    };
    let (verdict, delta) = match candidates.chosen {
        Some(chosen) => {
            let delta = (chosen - ptb).abs() / ptb.abs();
            (
                if delta <= VERIFY_TOL {
                    "exact"
                } else {
                    "mismatch"
                },
                Some(delta),
            )
        }
        None => ("no_strike", None),
    };
    VerifyResult {
        window,
        slug,
        candidates,
        gamma_price_to_beat: Some(ptb),
        gamma_final_price,
        matched,
        verdict,
        delta_chosen_rel: delta,
    }
}

/// Polls the venue for the resolved event metadata, then reconciles and
/// reports. Runs as its own task per resolved window.
async fn verify_strike(
    client: HttpClient,
    window: WindowId,
    slug: String,
    candidates: StrikeCandidates,
    verify_tx: mpsc::Sender<VerifyResult>,
) {
    for attempt in 1..=VERIFY_ATTEMPTS {
        match client.event_by_slug(&slug).await {
            Ok(events) => {
                if let Some(meta) = events.first().and_then(|e| e.event_metadata.as_ref()) {
                    let ptb = meta.price_to_beat.and_then(|d| d.to_f64());
                    if ptb.is_some() {
                        let final_price = meta.final_price.and_then(|d| d.to_f64());
                        let result = reconcile(window, slug, candidates, ptb, final_price);
                        let _ = verify_tx.send(result).await;
                        return;
                    }
                }
                tracing::debug!(
                    target: "fair", %slug, attempt,
                    "eventMetadata.priceToBeat not yet published"
                );
            }
            Err(error) => {
                tracing::warn!(target: "fair", %slug, attempt, %error, "verification fetch failed");
            }
        }
        tokio::time::sleep(VERIFY_POLL).await;
    }
    let result = reconcile(window, slug, candidates, None, None);
    let _ = verify_tx.send(result).await;
}

/// Everything the renderer and journaler track.
struct FairState {
    series: Series,
    asset: Asset,
    fair_params: FairParams,
    vol: VolEstimator,
    basis: BasisTracker,
    /// Per-asset composite-health gate. Lives here (outside the per-window
    /// engine) so a window rollover never resets its hysteresis.
    health: HealthMonitor,
    /// Health-tier transitions awaiting publication on the next heartbeat
    /// (the heartbeat is the single `model_tx` owner).
    pending_health: Vec<HealthTransition>,
    sanity_bound: f64,
    sanity_dur_ms: i64,
    engines: HashMap<WindowId, FairValueEngine>,
    markets: HashMap<WindowId, Arc<MarketInfo>>,
    books: HashMap<TokenId, Arc<BookSnapshot>>,
    current: Option<WindowId>,
    lifecycle: Option<WindowLifecycle>,
    next: Option<WindowId>,
    last_fair: Option<FairValue>,
    sanity_since: Option<TimestampMs>,
    sanity_breached: bool,
    last_snapshot_at: Option<TimestampMs>,
    verify_triggered: HashSet<WindowId>,
    resolved_outcomes: HashMap<WindowId, Outcome>,
    events: VecDeque<String>,
    // Counters for the run summary.
    ticks: u64,
    published: u64,
    dropped: u64,
    journaled: u64,
    resolved: u64,
    verify_exact: u64,
    verify_mismatch: u64,
    verify_unavailable: u64,
}

impl FairState {
    fn new(config: &AppConfig, series: Series) -> anyhow::Result<Self> {
        let asset = series.asset;
        let engine_defaults = &config.engine.defaults;
        let vol = VolEstimator::new(
            asset,
            vol_params(engine_defaults, PriceSource::BinanceDirect, TickKind::Mid),
        )
        .with_context(|| format!("building {asset} vol estimator"))?;
        let basis = BasisTracker::new(asset, basis_params(&config.feeds))
            .with_context(|| format!("building {asset} basis tracker"))?;
        let health = HealthMonitor::new(health_params(&config.feeds))
            .with_context(|| format!("building {asset} health monitor"))?;
        Ok(Self {
            series,
            asset,
            fair_params: fair_params(&config.feeds),
            vol,
            basis,
            health,
            pending_health: Vec::new(),
            sanity_bound: config.risk.sanity_bound,
            sanity_dur_ms: config.risk.sanity_bound_duration_ms.as_millis(),
            engines: HashMap::new(),
            markets: HashMap::new(),
            books: HashMap::new(),
            current: None,
            lifecycle: None,
            next: None,
            last_fair: None,
            sanity_since: None,
            sanity_breached: false,
            last_snapshot_at: None,
            verify_triggered: HashSet::new(),
            resolved_outcomes: HashMap::new(),
            events: VecDeque::new(),
            ticks: 0,
            published: 0,
            dropped: 0,
            journaled: 0,
            resolved: 0,
            verify_exact: 0,
            verify_mismatch: 0,
            verify_unavailable: 0,
        })
    }

    fn push_event(&mut self, now: TimestampMs, line: String) {
        if self.events.len() >= EVENT_PANE_LINES {
            self.events.pop_front();
        }
        self.events.push_back(format!("{} {line}", fmt_ts(now)));
    }

    fn on_event(
        &mut self,
        event: &Event,
        window_tx: &mpsc::Sender<(Arc<MarketInfo>, WindowLifecycle)>,
        cal_tx: &mpsc::UnboundedSender<CalRecord>,
        verify_client: &HttpClient,
        verify_tx: &mpsc::Sender<VerifyResult>,
    ) {
        let now = wall_now();
        match event {
            Event::PriceTick(tick) => {
                if tick.asset != self.asset {
                    return;
                }
                self.ticks += 1;
                self.vol.on_tick(tick);
                self.basis.on_tick(tick);
                // Feed every live engine (current + presubscribed next), so a
                // window's strike is captured around its open.
                let mut strike_events = Vec::new();
                for (win, engine) in &mut self.engines {
                    match engine.on_tick(tick) {
                        StrikeOutcome::Frozen { value, offset_ms } => {
                            strike_events
                                .push((*win, strike_record(engine, value, offset_ms, false)));
                        }
                        StrikeOutcome::RevisedCandidate { .. } => {
                            if let Some(k) = engine.strike().strike() {
                                strike_events.push((*win, strike_record(engine, k, 0, true)));
                            }
                        }
                        _ => {}
                    }
                }
                for (win, record) in strike_events {
                    if let CalRecord::Strike { revision, .. } = &record {
                        let slug = self.slug_of(win);
                        self.push_event(
                            now,
                            if *revision {
                                format!("STRIKE REVISED (held) {slug}")
                            } else {
                                format!("STRIKE FROZEN {slug}")
                            },
                        );
                    }
                    self.journal(cal_tx, record);
                }
                self.recompute_current(now);
            }
            Event::Window { market, lifecycle } => {
                // Forward every announcement to feed-clob (idempotent).
                if let Err(error) = window_tx.try_send((Arc::clone(market), *lifecycle)) {
                    tracing::warn!(target: "fair", %error, "window forward dropped");
                }
                if market.window.series != self.series {
                    return;
                }
                self.on_window(now, market, *lifecycle, cal_tx, verify_client, verify_tx);
            }
            Event::Book(snapshot) => {
                self.books
                    .insert(snapshot.token_id.clone(), Arc::clone(snapshot));
            }
            _ => {}
        }
    }

    fn on_window(
        &mut self,
        now: TimestampMs,
        market: &Arc<MarketInfo>,
        lifecycle: WindowLifecycle,
        cal_tx: &mpsc::UnboundedSender<CalRecord>,
        verify_client: &HttpClient,
        verify_tx: &mpsc::Sender<VerifyResult>,
    ) {
        let win = market.window;
        self.markets.insert(win, Arc::clone(market));
        match lifecycle {
            WindowLifecycle::Discovered => {
                self.ensure_engine(now, market);
                if self.current != Some(win) {
                    self.next = Some(win);
                }
            }
            WindowLifecycle::Open => {
                self.ensure_engine(now, market);
                self.current = Some(win);
                self.lifecycle = Some(lifecycle);
                if self.next == Some(win) {
                    self.next = None;
                }
                self.sanity_since = None;
                self.sanity_breached = false;
                self.push_event(now, format!("OPEN {}", market.event_slug));
            }
            WindowLifecycle::Closing | WindowLifecycle::Closed => {
                if self.current == Some(win) {
                    self.lifecycle = Some(lifecycle);
                }
            }
            WindowLifecycle::Resolved { outcome } => {
                self.resolved += 1;
                self.resolved_outcomes.insert(win, outcome);
                self.push_event(now, format!("RESOLVED {} -> {outcome}", market.event_slug));
                self.journal(
                    cal_tx,
                    CalRecord::Outcome {
                        ts: now.as_millis(),
                        window: win.to_string(),
                        slug: market.event_slug.clone(),
                        outcome: outcome.to_string(),
                    },
                );
                self.trigger_verification(now, win, verify_client, verify_tx);
            }
        }
    }

    /// Creates the per-window engine if absent (at Discovered, so strike
    /// capture sees the pre-open print). A window the model cannot price
    /// (unidentified resolution feed) is logged and skipped.
    fn ensure_engine(&mut self, now: TimestampMs, market: &Arc<MarketInfo>) {
        let win = market.window;
        if self.engines.contains_key(&win) {
            return;
        }
        match FairValueEngine::for_market(market, self.fair_params) {
            Ok(engine) => {
                self.engines.insert(win, engine);
            }
            Err(error) => {
                self.push_event(now, format!("UNPRICEABLE {}: {error}", market.event_slug));
                tracing::warn!(target: "fair", window = %win, %error, "window cannot be priced");
            }
        }
    }

    /// Recomputes the current window's fair value, runs the per-asset health
    /// monitor (stamping the gate onto the value), updates the sanity latch,
    /// and stores it for rendering/publishing. A health-tier change is buffered
    /// for the next heartbeat to publish.
    fn recompute_current(&mut self, now: TimestampMs) {
        let Some(win) = self.current else { return };
        let sigma = self.vol.sigma();
        let basis = self.basis.basis_log();
        let mut fair = match self.engines.get_mut(&win) {
            Some(engine) => engine.compute(FairInputs {
                now,
                sigma_1s: sigma,
                basis_log: basis,
            }),
            None => return,
        };
        // The monitor is the single producer of health; stamp its verdict onto
        // the (otherwise placeholder) fair value before it is rendered/published.
        let (health, reason, transition) = self.health.update(fair.health_inputs());
        fair.health = health;
        fair.reason = reason;
        if let Some(t) = transition {
            self.pending_health.push(t);
            let line = health_transition_line(self.asset, t);
            self.push_event(now, line);
        }
        let mid = self.up_mid();
        self.update_sanity(now, fair.p_up, mid);
        self.last_fair = Some(fair);
    }

    fn update_sanity(&mut self, now: TimestampMs, p_up: Option<f64>, mid: Option<f64>) {
        match (p_up, mid) {
            (Some(p), Some(m)) if (p - m).abs() > self.sanity_bound => {
                let since = *self.sanity_since.get_or_insert(now);
                self.sanity_breached = now.as_millis() - since.as_millis() >= self.sanity_dur_ms;
            }
            _ => {
                self.sanity_since = None;
                self.sanity_breached = false;
            }
        }
    }

    /// Mid of the current window's Up token, as `f64`.
    fn up_mid(&self) -> Option<f64> {
        let win = self.current?;
        let market = self.markets.get(&win)?;
        let book = self.books.get(&market.tokens.up)?;
        book.top().mid().and_then(|d| d.to_f64())
    }

    fn on_heartbeat(
        &mut self,
        model_tx: &mpsc::Sender<Event>,
        cal_tx: &mpsc::UnboundedSender<CalRecord>,
        verify_client: &HttpClient,
        verify_tx: &mpsc::Sender<VerifyResult>,
    ) {
        let now = wall_now();
        self.recompute_current(now);
        // Publish the latest snapshot, if a probability exists.
        if let Some(snapshot) = self.last_fair.and_then(|f| f.snapshot()) {
            match model_tx.try_send(Event::Model(snapshot)) {
                Ok(()) => self.published += 1,
                Err(_) => self.dropped += 1,
            }
        }
        // Publish any buffered health-tier transitions (latched — one per
        // episode, rare). Stop on a full channel and retry next heartbeat so a
        // transition is never silently dropped.
        while let Some(&transition) = self.pending_health.first() {
            let event = Event::ModelHealth(ModelHealthEvent {
                asset: self.asset,
                health: transition.to,
                reason: transition.reason,
                ts: transition.at,
            });
            if model_tx.try_send(event).is_ok() {
                self.pending_health.remove(0);
            } else {
                break;
            }
        }
        // Journal a snapshot record at ~1 Hz.
        if self.current.is_some()
            && self
                .last_snapshot_at
                .is_none_or(|t| now.as_millis() - t.as_millis() >= SNAPSHOT_INTERVAL_MS)
            && let Some(record) = self.snapshot_record(now)
        {
            self.last_snapshot_at = Some(now);
            self.journal(cal_tx, record);
        }
        // Fallback verification for windows long past close that never
        // announced a resolution.
        let overdue: Vec<WindowId> = self
            .markets
            .values()
            .filter(|m| {
                now.as_millis() - m.close_time.as_millis() >= VERIFY_FALLBACK_MS
                    && !self.verify_triggered.contains(&m.window)
            })
            .map(|m| m.window)
            .collect();
        for win in overdue {
            self.trigger_verification(now, win, verify_client, verify_tx);
        }
        self.prune(now);
    }

    /// Snapshots the current window's latest fair value as a journal record.
    fn snapshot_record(&self, now: TimestampMs) -> Option<CalRecord> {
        let fair = self.last_fair?;
        let market = self.markets.get(&fair.window)?;
        let mid = self.up_mid();
        let top = self
            .books
            .get(&market.tokens.up)
            .map(|b| b.top())
            .unwrap_or(core_types::TopOfBook {
                bid: None,
                ask: None,
                ts: now,
            });
        let abs_fair_minus_mid = match (fair.p_up, mid) {
            (Some(p), Some(m)) => Some((p - m).abs()),
            _ => None,
        };
        Some(CalRecord::Snapshot(Box::new(SnapshotRecord {
            ts: now.as_millis(),
            window: fair.window.to_string(),
            slug: market.event_slug.clone(),
            tau_secs: fair.tau_secs,
            k: fair.strike.k,
            strike_quality: quality_label(fair.strike.quality),
            s_chainlink: fair.chainlink_price,
            s_fast: fair.fast_price,
            s_anchor: fair.anchor_price,
            anchor: anchor_label(fair.anchor),
            basis_bps: fair.basis_bps,
            divergence_bps: fair.divergence_bps,
            sigma_1s: fair.sigma_1s,
            sigma_tau: fair.sigma_tau,
            z: fair.z,
            p_up: fair.p_up,
            book_bid_up: top.bid.and_then(|l| l.price.as_decimal().to_f64()),
            book_ask_up: top.ask.and_then(|l| l.price.as_decimal().to_f64()),
            book_mid_up: mid,
            abs_fair_minus_mid,
            sanity_breach: self.sanity_breached,
            health: health_label(fair.health),
            reason: reason_label(fair.reason),
            age_chainlink_ms: fair.chainlink_age.map(|a| a.as_millis()),
            age_binance_ms: fair.fast_age.map(|a| a.as_millis()),
        })))
    }

    fn trigger_verification(
        &mut self,
        now: TimestampMs,
        win: WindowId,
        verify_client: &HttpClient,
        verify_tx: &mpsc::Sender<VerifyResult>,
    ) {
        if !self.verify_triggered.insert(win) {
            return;
        }
        let Some(market) = self.markets.get(&win) else {
            return;
        };
        let slug = market.event_slug.clone();
        let candidates = self
            .engines
            .get(&win)
            .map_or_else(StrikeCandidates::default, |engine| {
                let est = engine.strike().estimate();
                StrikeCandidates {
                    chosen: est.k,
                    before: est.before.map(|c| c.value_f64),
                    after: est.after.map(|c| c.value_f64),
                }
            });
        self.push_event(now, format!("VERIFY scheduled {slug}"));
        tokio::spawn(verify_strike(
            verify_client.clone(),
            win,
            slug,
            candidates,
            verify_tx.clone(),
        ));
    }

    fn on_verify(&mut self, result: VerifyResult, cal_tx: &mpsc::UnboundedSender<CalRecord>) {
        let now = wall_now();
        match result.verdict {
            "exact" => self.verify_exact += 1,
            "unavailable" => self.verify_unavailable += 1,
            _ => self.verify_mismatch += 1,
        }
        let summary = match result.gamma_price_to_beat {
            Some(ptb) => format!(
                "VERIFY {} {} (matched {}, priceToBeat {ptb:.4})",
                result.slug, result.verdict, result.matched
            ),
            None => format!("VERIFY {} unavailable", result.slug),
        };
        if result.verdict == "mismatch" {
            tracing::warn!(target: "fair", slug = %result.slug, "strike verification mismatch");
        }
        self.push_event(now, summary);
        self.journal(
            cal_tx,
            CalRecord::Verify {
                ts: now.as_millis(),
                window: result.window.to_string(),
                slug: result.slug,
                gamma_price_to_beat: result.gamma_price_to_beat,
                gamma_final_price: result.gamma_final_price,
                our_chosen: result.candidates.chosen,
                our_before: result.candidates.before,
                our_after: result.candidates.after,
                matched_candidate: result.matched,
                delta_chosen_rel: result.delta_chosen_rel,
                verdict: result.verdict,
            },
        );
    }

    /// Drops engine/book state for windows well past close that are neither the
    /// current nor the next window.
    fn prune(&mut self, now: TimestampMs) {
        let keep_current = self.current;
        let keep_next = self.next;
        let stale: Vec<WindowId> = self
            .markets
            .values()
            .filter(|m| {
                Some(m.window) != keep_current
                    && Some(m.window) != keep_next
                    && now.as_millis() - m.close_time.as_millis() >= PRUNE_AFTER_CLOSE_MS
            })
            .map(|m| m.window)
            .collect();
        for win in stale {
            self.engines.remove(&win);
            if let Some(market) = self.markets.remove(&win) {
                self.books.remove(&market.tokens.up);
                self.books.remove(&market.tokens.down);
            }
        }
    }

    fn slug_of(&self, win: WindowId) -> String {
        self.markets
            .get(&win)
            .map_or_else(|| win.to_string(), |m| m.event_slug.clone())
    }

    fn journal(&mut self, cal_tx: &mpsc::UnboundedSender<CalRecord>, record: CalRecord) {
        if cal_tx.send(record).is_ok() {
            self.journaled += 1;
        }
    }

    /// Full-frame in-place redraw.
    fn draw(&self, now: TimestampMs) -> anyhow::Result<()> {
        let mut frame = String::with_capacity(2_048);
        frame.push_str("\x1b[H\x1b[J");
        let Some(win) = self.current else {
            frame.push_str(&format!(
                "fair [{}] waiting for the current window (discovery + scheduler warming up)…\n",
                self.series.key()
            ));
            return print_frame(&frame);
        };
        let market = match self.markets.get(&win) {
            Some(m) => m,
            None => return print_frame(&frame),
        };
        let closes_in = market.close_time.signed_duration_since(now);
        frame.push_str(&format!(
            "{} {}  polymarket.com/event/{}\n",
            self.series.key(),
            win,
            market.event_slug
        ));
        frame.push_str(&format!(
            "phase {:<10} closes in {:>8}\n",
            self.lifecycle
                .map_or_else(|| "-".to_owned(), |l| format!("{l:?}")),
            fmt_countdown(closes_in),
        ));

        let fair = self.last_fair;
        // Strike line.
        match fair.map(|f| f.strike) {
            Some(est) => {
                let k = est.k.map_or_else(|| "-".to_owned(), |k| format!("{k:.4}"));
                let before = est.before.map_or_else(
                    || "-".to_owned(),
                    |c| format!("{:+}ms", c.offset_ms(market.window.open_time)),
                );
                let after = est.after.map_or_else(
                    || "-".to_owned(),
                    |c| format!("{:+}ms", c.offset_ms(market.window.open_time)),
                );
                frame.push_str(&format!(
                    "strike K {k}  rule {}  quality {}{}  [before {before}, after {after}]\n",
                    rule_label(est.rule),
                    quality_label(est.quality),
                    if est.frozen { " FROZEN" } else { "" },
                ));
            }
            None => frame.push_str("strike  (warming up)\n"),
        }
        // Inputs line.
        match fair {
            Some(f) => frame.push_str(&format!(
                "inputs  chainlink {} ({})  binance {} ({})  S {}  basis {}  div {}\n",
                opt_price(f.chainlink_price),
                opt_age(f.chainlink_age),
                opt_price(f.fast_price),
                opt_age(f.fast_age),
                opt_price(f.anchor_price),
                f.basis_bps
                    .map_or_else(|| "-".to_owned(), |b| format!("{b:+.1}bps")),
                f.divergence_bps
                    .map_or_else(|| "-".to_owned(), |d| format!("{d:.1}bps")),
            )),
            None => frame.push_str("inputs  (warming up)\n"),
        }
        // Model line.
        match fair {
            Some(f) => frame.push_str(&format!(
                "model   sigma_1s {}  sigma_tau {}  z {}  p_up {}  [{}/{} via {}]\n",
                f.sigma_1s
                    .map_or_else(|| "-".to_owned(), |s| format!("{:.2}bps", s * 1e4)),
                f.sigma_tau
                    .map_or_else(|| "-".to_owned(), |s| format!("{s:.4}")),
                f.z.map_or_else(|| "-".to_owned(), |z| format!("{z:+.3}")),
                f.p_up.map_or_else(|| "-".to_owned(), |p| format!("{p:.4}")),
                health_label(f.health),
                reason_label(f.reason),
                anchor_label(f.anchor),
            )),
            None => frame.push_str("model   (warming up)\n"),
        }
        // Book line.
        let mid = self.up_mid();
        let p_up = fair.and_then(|f| f.p_up);
        let gap = match (p_up, mid) {
            (Some(p), Some(m)) => format!("{:.4}", (p - m).abs()),
            _ => "-".to_owned(),
        };
        frame.push_str(&format!(
            "book    Up mid {}  |p_up - mid| {gap}{}\n",
            mid.map_or_else(|| "-".to_owned(), |m| format!("{m:.4}")),
            if self.sanity_breached {
                "  SANITY BREACH"
            } else {
                ""
            },
        ));

        frame.push_str("recent events:\n");
        if self.events.is_empty() {
            frame.push_str("  (none yet)\n");
        }
        for line in &self.events {
            frame.push_str("  ");
            frame.push_str(line);
            frame.push('\n');
        }
        print_frame(&frame)
    }
}

/// Builds a strike journal record from an engine's current capture estimate.
fn strike_record(
    engine: &FairValueEngine,
    value: f64,
    offset_ms: i64,
    revision: bool,
) -> CalRecord {
    let est = engine.strike();
    let estimate = est.estimate();
    let open = engine.window().open_time;
    CalRecord::Strike {
        ts: wall_now().as_millis(),
        window: engine.window().to_string(),
        rule: rule_label(estimate.rule),
        chosen_value: value,
        chosen_offset_ms: offset_ms,
        before_value: estimate.before.map(|c| c.value_f64),
        before_offset_ms: estimate.before.map(|c| c.offset_ms(open)),
        after_value: estimate.after.map(|c| c.value_f64),
        after_offset_ms: estimate.after.map(|c| c.offset_ms(open)),
        revision,
    }
}

/// ~1 Hz model snapshot for the active window. Boxed inside [`CalRecord`]
/// (it is by far the widest variant); the internally-tagged enum still
/// serializes it flat as `{"type":"snapshot", …}`.
#[derive(Debug, Clone, Serialize)]
struct SnapshotRecord {
    ts: i64,
    window: String,
    slug: String,
    tau_secs: f64,
    k: Option<f64>,
    strike_quality: &'static str,
    s_chainlink: Option<f64>,
    s_fast: Option<f64>,
    s_anchor: Option<f64>,
    anchor: &'static str,
    basis_bps: Option<f64>,
    divergence_bps: Option<f64>,
    sigma_1s: Option<f64>,
    sigma_tau: Option<f64>,
    z: Option<f64>,
    p_up: Option<f64>,
    book_bid_up: Option<f64>,
    book_ask_up: Option<f64>,
    book_mid_up: Option<f64>,
    abs_fair_minus_mid: Option<f64>,
    sanity_breach: bool,
    health: &'static str,
    reason: &'static str,
    age_chainlink_ms: Option<i64>,
    age_binance_ms: Option<i64>,
}

/// Calibration records — one tagged JSON object per line.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CalRecord {
    /// ~1 Hz model snapshot for the active window.
    Snapshot(Box<SnapshotRecord>),
    /// Strike freeze (or held revision).
    Strike {
        ts: i64,
        window: String,
        rule: &'static str,
        chosen_value: f64,
        chosen_offset_ms: i64,
        before_value: Option<f64>,
        before_offset_ms: Option<i64>,
        after_value: Option<f64>,
        after_offset_ms: Option<i64>,
        revision: bool,
    },
    /// Window resolution outcome.
    Outcome {
        ts: i64,
        window: String,
        slug: String,
        outcome: String,
    },
    /// Post-hoc strike verification verdict.
    Verify {
        ts: i64,
        window: String,
        slug: String,
        gamma_price_to_beat: Option<f64>,
        gamma_final_price: Option<f64>,
        our_chosen: Option<f64>,
        our_before: Option<f64>,
        our_after: Option<f64>,
        matched_candidate: &'static str,
        delta_chosen_rel: Option<f64>,
        verdict: &'static str,
    },
}

/// Maps `feeds.binance_stale_after_ms` into the basis pairing window; the rest
/// of [`BasisParams`] are model-crate defaults.
pub(crate) fn basis_params(feeds: &config::FeedsConfig) -> BasisParams {
    BasisParams {
        pairing_max_age_ms: feeds.binance_stale_after_ms.as_millis(),
        ..BasisParams::default()
    }
}

/// Maps the `feeds.*` staleness bounds into [`FairParams`] (anchor selection).
pub(crate) fn fair_params(feeds: &config::FeedsConfig) -> FairParams {
    FairParams {
        chainlink_stale_ms: feeds.rtds_stale_after_ms.as_millis(),
        fast_stale_ms: feeds.binance_stale_after_ms.as_millis(),
    }
}

/// Maps `feeds.rtds_stale_after_ms` into the health gate's Chainlink-staleness
/// bound; the divergence tiers and dwell timers stay at the model-crate defaults.
pub(crate) fn health_params(feeds: &config::FeedsConfig) -> HealthParams {
    let stale = feeds.rtds_stale_after_ms.as_millis();
    HealthParams {
        chainlink_stale_ms: stale,
        // Keep the recovery dead zone proportional (the model default is 60% of
        // the stale bound) so a tighter `rtds_stale_after_ms` can never invert
        // `fresh < stale` and refuse to construct the monitor.
        chainlink_fresh_ms: stale * 3 / 5,
        ..HealthParams::default()
    }
}

/// `data/calibration/fair-{series}-{YYYYMMDD-HHMMSS}.jsonl` (latency-report
/// naming precedent).
fn calibration_path(series: Series, started: TimestampMs) -> PathBuf {
    PathBuf::from("data/calibration").join(format!(
        "fair-{}-{}.jsonl",
        series.key(),
        crate::latency::file_stamp(started),
    ))
}

fn rule_label(rule: StrikeRule) -> &'static str {
    match rule {
        StrikeRule::LastAtOrBefore => "last_at_or_before",
        StrikeRule::FirstAtOrAfter => "first_at_or_after",
    }
}

fn quality_label(quality: StrikeQuality) -> &'static str {
    match quality {
        StrikeQuality::Missing => "missing",
        StrikeQuality::Distant => "distant",
        StrikeQuality::Boundary => "boundary",
    }
}

fn anchor_label(anchor: Option<AnchorSource>) -> &'static str {
    match anchor {
        Some(AnchorSource::Chainlink) => "chainlink",
        Some(AnchorSource::BinanceCorrected) => "binance_corrected",
        None => "none",
    }
}

fn health_label(health: ModelHealth) -> &'static str {
    match health {
        ModelHealth::Ready => "ready",
        ModelHealth::Degraded => "degraded",
        ModelHealth::Unreliable => "unreliable",
    }
}

fn reason_label(reason: ModelHealthReason) -> &'static str {
    match reason {
        ModelHealthReason::Nominal => "nominal",
        ModelHealthReason::Warming => "warming",
        ModelHealthReason::ChainlinkStale => "chainlink_stale",
        ModelHealthReason::NoAnchor => "no_anchor",
        ModelHealthReason::DivergenceHard => "divergence_hard",
        ModelHealthReason::FastLeadLost => "fast_lead_lost",
        ModelHealthReason::DivergenceSoft => "divergence_soft",
    }
}

/// One-line summary of a health-tier change for the events pane.
fn health_transition_line(asset: Asset, t: HealthTransition) -> String {
    format!(
        "MODEL HEALTH {asset} {} -> {} ({})",
        health_label(t.from),
        health_label(t.to),
        reason_label(t.reason),
    )
}

fn opt_price(p: Option<f64>) -> String {
    p.map_or_else(|| "-".to_owned(), |v| format!("{v:.2}"))
}

fn opt_age(age: Option<core_types::DurationMs>) -> String {
    age.map_or_else(
        || "-".to_owned(),
        |a| format!("{:.1}s", a.as_millis() as f64 / 1000.0),
    )
}

fn print_frame(frame: &str) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(frame.as_bytes())
        .context("writing fair frame")?;
    stdout.flush().context("flushing fair frame")
}

#[cfg(test)]
mod tests {
    use core_types::WindowDuration;

    use super::*;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_781_290_500_000),
        }
    }

    #[test]
    fn config_maps_into_model_params() {
        let cfg = config::AppConfig::default();
        let bp = basis_params(&cfg.feeds);
        assert_eq!(
            bp.pairing_max_age_ms,
            cfg.feeds.binance_stale_after_ms.as_millis()
        );
        let fp = fair_params(&cfg.feeds);
        assert_eq!(
            fp.chainlink_stale_ms,
            cfg.feeds.rtds_stale_after_ms.as_millis()
        );
        assert_eq!(
            fp.fast_stale_ms,
            cfg.feeds.binance_stale_after_ms.as_millis()
        );
        let hp = health_params(&cfg.feeds);
        assert_eq!(
            hp.chainlink_stale_ms,
            cfg.feeds.rtds_stale_after_ms.as_millis()
        );
        // The mapped bounds keep a recovery dead zone, so the monitor builds.
        assert!(hp.chainlink_fresh_ms < hp.chainlink_stale_ms);
        assert!(HealthMonitor::new(hp).is_ok());
    }

    #[test]
    fn calibration_path_is_stamped_and_namespaced() {
        let path = calibration_path(
            Series {
                asset: Asset::Eth,
                duration: WindowDuration::H1,
            },
            TimestampMs::from_millis(1_781_290_500_000),
        );
        let s = path.to_string_lossy();
        assert!(s.contains("calibration"), "{s}");
        assert!(s.contains("fair-ETH-1h-"), "{s}");
        assert!(s.ends_with(".jsonl"), "{s}");
    }

    #[test]
    fn reconcile_exact_when_chosen_matches_price_to_beat() {
        let candidates = StrikeCandidates {
            chosen: Some(63_788.969_145_188_41),
            before: Some(63_788.969_145_188_41),
            after: Some(63_790.0),
        };
        let r = reconcile(
            window(),
            "slug".to_owned(),
            candidates,
            Some(63_788.969_145_188_41),
            Some(63_757.0),
        );
        assert_eq!(r.verdict, "exact");
        assert_eq!(r.matched, "before");
        assert!(r.delta_chosen_rel.unwrap() < VERIFY_TOL);
    }

    #[test]
    fn reconcile_mismatch_when_chosen_differs() {
        // Our chosen (before) is off; the venue's priceToBeat matches our
        // `after` candidate — the signal to flip the rule.
        let candidates = StrikeCandidates {
            chosen: Some(63_700.0),
            before: Some(63_700.0),
            after: Some(63_788.969_145_188_41),
        };
        let r = reconcile(
            window(),
            "slug".to_owned(),
            candidates,
            Some(63_788.969_145_188_41),
            None,
        );
        assert_eq!(r.verdict, "mismatch");
        assert_eq!(r.matched, "after");
        assert!(r.delta_chosen_rel.unwrap() > VERIFY_TOL);
    }

    #[test]
    fn reconcile_unavailable_without_metadata() {
        let r = reconcile(
            window(),
            "slug".to_owned(),
            StrikeCandidates::default(),
            None,
            None,
        );
        assert_eq!(r.verdict, "unavailable");
        assert_eq!(r.matched, "none");
        assert!(r.delta_chosen_rel.is_none());
    }

    #[test]
    fn cal_records_serialize_with_type_tag() {
        let snap = CalRecord::Outcome {
            ts: 1,
            window: "BTC-5m@1".to_owned(),
            slug: "s".to_owned(),
            outcome: "Up".to_owned(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"type\":\"outcome\""), "{json}");
        let verify = CalRecord::Verify {
            ts: 2,
            window: "BTC-5m@1".to_owned(),
            slug: "s".to_owned(),
            gamma_price_to_beat: Some(1.0),
            gamma_final_price: None,
            our_chosen: Some(1.0),
            our_before: None,
            our_after: None,
            matched_candidate: "before",
            delta_chosen_rel: Some(0.0),
            verdict: "exact",
        };
        let json = serde_json::to_string(&verify).unwrap();
        assert!(json.contains("\"type\":\"verify\""), "{json}");
        assert!(json.contains("\"verdict\":\"exact\""), "{json}");
    }

    #[test]
    fn boxed_snapshot_serializes_flat_with_type_tag() {
        // Boxing the variant must not change the wire shape: still a single
        // flat object tagged "snapshot", carrying health + reason.
        let snap = CalRecord::Snapshot(Box::new(SnapshotRecord {
            ts: 1,
            window: "BTC-5m@1".to_owned(),
            slug: "s".to_owned(),
            tau_secs: 12.5,
            k: Some(63_000.0),
            strike_quality: "boundary",
            s_chainlink: Some(63_010.0),
            s_fast: Some(63_012.0),
            s_anchor: Some(63_011.0),
            anchor: "binance_corrected",
            basis_bps: Some(-4.7),
            divergence_bps: Some(2.0),
            sigma_1s: Some(0.0004),
            sigma_tau: Some(0.0014),
            z: Some(0.31),
            p_up: Some(0.62),
            book_bid_up: Some(0.60),
            book_ask_up: Some(0.63),
            book_mid_up: Some(0.615),
            abs_fair_minus_mid: Some(0.005),
            sanity_breach: false,
            health: "ready",
            reason: "nominal",
            age_chainlink_ms: Some(120),
            age_binance_ms: Some(35),
        }));
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"type\":\"snapshot\""), "{json}");
        assert!(json.contains("\"health\":\"ready\""), "{json}");
        assert!(json.contains("\"reason\":\"nominal\""), "{json}");
        // Flat — no nesting introduced by the Box/newtype.
        assert!(!json.contains("\"Snapshot\""), "{json}");
    }
}
