//! Deterministic replay harness + parameter sweep (feature `replay`, CLAUDE.md
//! §3/§9/§10).
//!
//! [`run_replay`] re-runs the **full engine** — the [`RiskManager`] gateway and
//! the three strategies it owns — over a recorded journal tape, driving a fresh
//! seed-stable [`PaperVenue`], and produces the same kind of analytics output as
//! live paper: the §10 [`SeriesComparison`] table plus a [`ReplaySummary`].
//! [`run_sweep`] runs a grid over four quoting parameters (minimum edge, skew
//! strength, cancel threshold, taker buffer) on one recording and emits a ranked,
//! dashboard-shaped [`SweepReport`] — the operator's tool for tuning on real
//! captured data.
//!
//! # Determinism (the acceptance bar)
//!
//! The **guaranteed** property is replay-vs-replay byte-identical determinism:
//! two [`run_replay`] calls over the same tape with the same seed serialize to
//! identical JSON. This is secured by (1) a single-threaded, `start_paused`
//! `tokio` runtime — no scheduler nondeterminism; (2) a clock driven purely from
//! the tape's `RecordEnvelope::ts_local_ms` (the wall clock never contributes —
//! virtual time only advances when the harness steps it); (3) a required, fixed
//! `PaperParams::rng_seed` (the latency RNG is a seeded xorshift); (4)
//! `try_recv`-only draining that fully drains the venue between clock steps; and
//! (5) the analytics fold, which already folds in a deterministic order.
//!
//! "Produces the same output as live paper" is **approximate**, not bit-for-bit
//! against the original session: live used the wall clock, real async task
//! ordering, and a wall-seeded latency RNG. The replay reproduces it closely
//! (identical recorded inputs, identical engine/venue code, fixed seed, same
//! latency means) — but the exact, testable guarantee is replay-vs-replay
//! determinism. This mirrors how `Analytics::rebuild == live` is *exact* (a pure
//! fold) while engine re-execution is only exactly self-reproducing.
//!
//! # How it re-runs the engine
//!
//! The recorded engine/venue **outputs** (`OrderUpdate`, `Fill`, `Inventory`,
//! `Settlement`, `ControlAudit`) are dropped — the re-run regenerates them. Every
//! other recorded event is a re-run **input**, fed uniformly to both
//! `venue.on_bus_event` (it ignores what it doesn't consume) and `risk.on_event`,
//! exactly as the live `bot run` loop does. The model is **replayed** (recorded
//! `Model`/`ModelHealth` snapshots) rather than re-run — model params aren't
//! swept, so this is faithful and keeps the `model` crate out of the dependency
//! set. Re-running the engine produces fresh orders → the paper venue produces
//! fresh fills (the `VenueEvent` stream) → those fills feed the re-run
//! `InventoryManager` and `Analytics`, plus the inventory-derived settlements.
//! Periodic `risk.on_tick`s are merged into the tape at the configured cadence.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{Decimal, Dollars, Event, Mode, Series, TimestampMs};
use engine::{InventoryEffect, InventoryManager, RiskManager, RiskParams};
use journal::{JournalRecord, RecordEnvelope};
use rust_decimal::prelude::ToPrimitive;
use tokio::sync::mpsc::Receiver;
use venue_api::{VenueEvent, VenueEvents};
use venue_paper::{PaperParams, PaperVenue};

use crate::{Analytics, AnalyticsParams, ComparisonWindow, DayKey, SeriesComparison, SortColumn};

/// Venue event-channel capacity for a replay — large enough that the venue timer
/// task never blocks across a clock step (the harness drains after every step, so
/// the buffer never approaches this).
const REPLAY_CHANNEL_CAP: usize = 65_536;

/// Bounded end-of-tape flush rounds: advance one max-latency grace per round,
/// draining each, until a round produces nothing new.
const FLUSH_ROUNDS: usize = 8;

/// Everything one replay configuration needs. The bot maps `config → these` at
/// its boundary; the crate stays `config`-free (CLAUDE.md §4).
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// The full risk-manager + strategy parameter bundle.
    pub risk: RiskParams,
    /// Per-series worst-case-loss caps the risk manager enforces.
    pub series_caps: HashMap<Series, Dollars>,
    /// Paper-venue parameters. `rng_seed` **must** be `Some` for a deterministic
    /// replay (else [`ReplayError::UnseededPaper`]); zero jitter gives exact-mean
    /// latencies.
    pub paper: PaperParams,
    /// Analytics tuning.
    pub analytics: AnalyticsParams,
    /// Session mode stamped on the analytics output (`Mode::Paper` for a paper
    /// recording).
    pub mode: Mode,
    /// Cadence (ms) at which `risk.on_tick` is fired, merged into the tape
    /// timeline (the live loop's `config.run.risk_tick_ms`, ≈500).
    pub risk_tick_ms: i64,
    /// Which calendar span the output [`SeriesComparison`] aggregates over.
    pub comparison_window: ComparisonWindow,
}

/// Headline counts proving a replay did non-trivial work, plus how much of the
/// tape was dropped as a regenerated output.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ReplaySummary {
    /// Envelopes in the tape.
    pub events_total: u64,
    /// Recorded outputs dropped (the engine/venue regenerate them).
    pub events_dropped: u64,
    /// Re-run fills observed on the venue stream (must be `> 0` for a meaningful
    /// replay).
    pub fills: u64,
    /// Re-run order updates observed on the venue stream.
    pub order_updates: u64,
    /// Windows settled (inventory `Settled` effects fed to analytics).
    pub windows_settled: u64,
    /// Net ledger PnL summed across the comparison rows.
    pub net_pnl: Dollars,
    /// First envelope's local timestamp (ms).
    pub first_event_ms: i64,
    /// Last envelope's local timestamp (ms) — the replay clock's end and the
    /// `today` reference for the comparison query.
    pub last_event_ms: i64,
}

impl ReplaySummary {
    /// A zeroed summary stamped with the tape's bounds.
    fn new(events_total: u64, first_event_ms: i64, last_event_ms: i64) -> Self {
        Self {
            events_total,
            events_dropped: 0,
            fills: 0,
            order_updates: 0,
            windows_settled: 0,
            net_pnl: Dollars::ZERO,
            first_event_ms,
            last_event_ms,
        }
    }
}

/// One replay's result: the dashboard-shaped per-series comparison + a summary.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ReplayOutput {
    /// The §10 series-comparison table for `comparison_window`.
    pub comparison: SeriesComparison,
    /// Headline counts for the run.
    pub summary: ReplaySummary,
}

/// What can go wrong constructing or running a replay.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The tape held no events.
    #[error("empty tape")]
    EmptyTape,
    /// The paper params left `rng_seed` unset — a deterministic replay requires a
    /// fixed seed.
    #[error("paper rng_seed must be Some for a deterministic replay")]
    UnseededPaper,
    /// The `start_paused` tokio runtime could not be built.
    #[error("building the replay runtime: {0}")]
    Runtime(#[from] std::io::Error),
    /// A parallel sweep worker thread panicked, losing one or more grid points.
    #[error("a sweep worker thread failed")]
    Worker,
}

/// Re-runs the full engine over `events` and returns the resulting analytics.
///
/// Builds a current-thread `start_paused` tokio runtime and `block_on`s the
/// deterministic driver, so callers need no async. Two calls with the same tape
/// and config serialize to byte-identical JSON.
///
/// # Errors
/// [`ReplayError::EmptyTape`] if `events` is empty, [`ReplayError::UnseededPaper`]
/// if `cfg.paper.rng_seed` is `None`, or [`ReplayError::Runtime`] if the runtime
/// cannot be built.
pub fn run_replay(
    events: &[RecordEnvelope],
    cfg: &ReplayConfig,
) -> Result<ReplayOutput, ReplayError> {
    if events.is_empty() {
        return Err(ReplayError::EmptyTape);
    }
    if cfg.paper.rng_seed.is_none() {
        return Err(ReplayError::UnseededPaper);
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()?;
    Ok(runtime.block_on(drive(events, cfg)))
}

/// The deterministic re-run loop. Runs on the current-thread `start_paused`
/// runtime built by [`run_replay`]; virtual time advances only via the explicit
/// `advance_to` sleeps below.
async fn drive(events: &[RecordEnvelope], cfg: &ReplayConfig) -> ReplayOutput {
    let base = events.first().map_or(0, |e| e.ts_local_ms);
    let last_ts = events.last().map_or(base, |e| e.ts_local_ms);
    let total = u64::try_from(events.len()).unwrap_or(u64::MAX);
    let mut summary = ReplaySummary::new(total, base, last_ts);

    // A generous channel so the venue timer never blocks across a clock step.
    let paper = PaperParams {
        event_channel_capacity: REPLAY_CHANNEL_CAP,
        ..cfg.paper.clone()
    };
    let start = tokio::time::Instant::now();
    let mut venue = PaperVenue::spawn(paper, move || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
        TimestampMs::from_millis(base.saturating_add(elapsed))
    });
    // The same clock the venue reads, for the harness's own `now`.
    let now = || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
        TimestampMs::from_millis(base.saturating_add(elapsed))
    };

    let Some(mut rx) = venue.take_event_rx() else {
        // A freshly spawned venue always yields its receiver once; degrade
        // defensively to an empty (still well-formed) result.
        return empty_output(cfg, last_ts, summary);
    };

    let mut risk = RiskManager::new(cfg.risk.clone(), cfg.series_caps.clone());
    let mut inv = InventoryManager::new();
    let mut an = Analytics::new(cfg.mode, cfg.analytics);

    let tick_ms = cfg.risk_tick_ms.max(1);
    let mut last_tick = base;
    let mut clock_ms = base;

    for env in events {
        let t = env.ts_local_ms;
        // 1. Fire periodic risk ticks up to `t` (discrete-event merge with the tape).
        while last_tick.saturating_add(tick_ms) <= t {
            last_tick = last_tick.saturating_add(tick_ms);
            advance_to(&mut clock_ms, last_tick).await;
            drain_venue(
                &mut rx,
                &mut risk,
                &venue,
                &mut inv,
                &mut an,
                &mut summary,
                now(),
            )
            .await;
            risk.on_tick(&venue, now()).await;
            let _ = risk.take_published();
            drain_venue(
                &mut rx,
                &mut risk,
                &venue,
                &mut inv,
                &mut an,
                &mut summary,
                now(),
            )
            .await;
        }
        // 2. Advance to the event; drain anything that matured first (ordering).
        advance_to(&mut clock_ms, t).await;
        drain_venue(
            &mut rx,
            &mut risk,
            &venue,
            &mut inv,
            &mut an,
            &mut summary,
            now(),
        )
        .await;
        // 3. Feed inputs; drop recorded outputs.
        if !is_replay_input(&env.rec) {
            summary.events_dropped += 1;
            continue;
        }
        let ev = env.rec.to_event();
        venue.on_bus_event(&ev).await;
        risk.on_event(&ev, &venue, now()).await;
        let _ = risk.take_published();
        match &ev {
            Event::Window { .. } => {
                let _ = an.on_event(&ev);
                for effect in inv.on_event(&ev) {
                    if let InventoryEffect::Settled(summary_s) = effect {
                        summary.windows_settled += 1;
                        let _ = an.on_event(&Event::Settlement(Arc::new(summary_s)));
                    }
                }
            }
            Event::Model(_) => {
                let _ = an.on_event(&ev);
            }
            _ => {}
        }
        // 4. Drain effects produced by feeding the event.
        drain_venue(
            &mut rx,
            &mut risk,
            &venue,
            &mut inv,
            &mut an,
            &mut summary,
            now(),
        )
        .await;
    }

    // End-of-tape flush: let final placement/cancel latencies + resolution settle.
    let grace = cfg
        .paper
        .placement
        .mean_ms
        .as_millis()
        .max(cfg.paper.cancel.mean_ms.as_millis())
        .saturating_add(1)
        .max(1);
    for _ in 0..FLUSH_ROUNDS {
        let before = (
            summary.fills,
            summary.order_updates,
            summary.windows_settled,
        );
        let target = clock_ms.saturating_add(grace);
        advance_to(&mut clock_ms, target).await;
        drain_venue(
            &mut rx,
            &mut risk,
            &venue,
            &mut inv,
            &mut an,
            &mut summary,
            now(),
        )
        .await;
        if (
            summary.fills,
            summary.order_updates,
            summary.windows_settled,
        ) == before
        {
            break;
        }
    }
    venue.shutdown();

    let today = DayKey::from_ts(TimestampMs::from_millis(last_ts));
    let comparison = an.series_comparison_over(cfg.comparison_window, today);
    summary.net_pnl = comparison.rows.iter().map(|r| r.net_pnl).sum();
    ReplayOutput {
        comparison,
        summary,
    }
}

/// A well-formed empty result (the venue receiver was unexpectedly already taken).
fn empty_output(cfg: &ReplayConfig, last_ts: i64, summary: ReplaySummary) -> ReplayOutput {
    let an = Analytics::new(cfg.mode, cfg.analytics);
    let today = DayKey::from_ts(TimestampMs::from_millis(last_ts));
    let comparison = an.series_comparison_over(cfg.comparison_window, today);
    ReplayOutput {
        comparison,
        summary,
    }
}

/// Advances virtual time to `target` via a `start_paused` sleep (auto-advance
/// fires the venue timer's due deadlines and runs its task before returning).
async fn advance_to(clock_ms: &mut i64, target: i64) {
    if target > *clock_ms {
        let delta = u64::try_from(target - *clock_ms).unwrap_or(0);
        tokio::time::sleep(Duration::from_millis(delta)).await;
        *clock_ms = target;
    }
}

/// Drains every ready venue event into the risk manager (authoritative for its
/// inventory/notional), the re-run `InventoryManager`, and analytics — mirroring
/// the live `bot run` loop's `handle_venue_event`.
async fn drain_venue(
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    venue: &PaperVenue,
    inv: &mut InventoryManager,
    an: &mut Analytics,
    summary: &mut ReplaySummary,
    now: TimestampMs,
) {
    while let Ok(ve) = rx.try_recv() {
        risk.on_venue_event(&ve, venue, now).await;
        let _ = risk.take_published();
        match &ve {
            VenueEvent::Order(_) => summary.order_updates += 1,
            VenueEvent::Fill(fill) => {
                summary.fills += 1;
                let ev = Event::Fill(Arc::clone(fill));
                let _ = an.on_event(&ev);
                for effect in inv.on_event(&ev) {
                    if let InventoryEffect::Settled(s) = effect {
                        summary.windows_settled += 1;
                        let _ = an.on_event(&Event::Settlement(Arc::new(s)));
                    }
                }
            }
            VenueEvent::Connectivity { .. } => {}
        }
    }
}

/// Whether a recorded event is a re-run input (true) or a regenerated engine/venue
/// output to drop (false).
fn is_replay_input(rec: &JournalRecord) -> bool {
    !matches!(
        rec,
        JournalRecord::OrderUpdate(_)
            | JournalRecord::Fill(_)
            | JournalRecord::Inventory(_)
            | JournalRecord::Settlement(_)
            | JournalRecord::ControlAudit(_)
    )
}

// ============================================================================
// Parameter sweep
// ============================================================================

/// The four-dimension parameter grid. An empty dimension is filled with the
/// config base value, so the cartesian product never collapses to zero points.
#[derive(Debug, Clone, Default)]
pub struct SweepGrid {
    /// `quote.min_edge` values.
    pub min_edge: Vec<Decimal>,
    /// `quote.gamma_inventory_skew` (skew strength) values.
    pub gamma: Vec<f64>,
    /// `quote_manager.reprice_threshold_theta` (cancel threshold) values;
    /// `cancel_market_theta` is scaled proportionally per point.
    pub cancel_theta: Vec<f64>,
    /// `momentum.momentum_buffer` (taker buffer) values.
    pub taker_buffer: Vec<Decimal>,
}

/// One concrete corner of the grid.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SweepPoint {
    /// `quote.min_edge`.
    pub min_edge: Decimal,
    /// `quote.gamma_inventory_skew`.
    pub gamma: f64,
    /// `quote_manager.reprice_threshold_theta`.
    pub cancel_theta: f64,
    /// `momentum.momentum_buffer`.
    pub taker_buffer: Decimal,
}

/// Across-series headline metrics for one grid point, mirroring the
/// [`SeriesComparisonRow`](crate::SeriesComparisonRow) columns so the dashboard's
/// series view can render a sweep the same way it renders series.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SweepAggregate {
    /// Windows traded across all series.
    pub windows_traded: u32,
    /// Windows-weighted fraction profitable, in `[0, 1]`.
    pub fraction_profitable: f64,
    /// Net ledger PnL.
    pub net_pnl: Dollars,
    /// Net PnL per traded window.
    pub pnl_per_window: Dollars,
    /// Locked-pair PnL (the guaranteed-edge half of the split).
    pub locked_pair_pnl: Dollars,
    /// Inventory PnL (excess + settlement remainder).
    pub inventory_pnl: Dollars,
    /// Taker fees paid (a positive cost).
    pub fees_paid: Dollars,
    /// Estimated maker rebates earned.
    pub rebates_earned: Dollars,
    /// Maker fills.
    pub maker_fills: u64,
    /// Taker fills.
    pub taker_fills: u64,
    /// `Σ price·size` over taker fills.
    pub taker_notional: Dollars,
    /// Sample-weighted average passive-fill 5s markout, `None` with no sample.
    pub avg_markout_5s: Option<f64>,
    /// The orderable rank key under the report's `rank_metric` (compared, never
    /// displayed; a missing optional is `-INF`).
    pub rank_key: f64,
}

/// One sweep grid point's full result.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SweepRow {
    /// The parameter corner.
    pub point: SweepPoint,
    /// The full per-series comparison for this point (the dashboard series view).
    pub comparison: SeriesComparison,
    /// The across-series headline + rank key.
    pub aggregate: SweepAggregate,
}

/// A complete sweep result, rows ranked best-first by `rank_metric`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SweepReport {
    /// The metric the rows are ranked by.
    pub rank_metric: SortColumn,
    /// Total grid points run.
    pub points_total: usize,
    /// Grid points, ranked best-first (descending under `higher_is_better`,
    /// ascending otherwise), with a stable cartesian-index tiebreak.
    pub rows: Vec<SweepRow>,
}

/// Runs the grid over the shared tape and returns a ranked report.
///
/// Each point clones `base`, overrides the four swept fields (scaling
/// `cancel_market_theta` proportionally with `reprice_threshold_theta`), and
/// replays the whole tape. With `parallel`, points fan out across
/// `available_parallelism() − 2` threads, each with its own runtime; per-replay
/// determinism is independent of parallelism, and the final rank sort makes the
/// report identical either way.
///
/// # Errors
/// [`ReplayError::EmptyTape`] / [`ReplayError::UnseededPaper`] as
/// [`run_replay`], or [`ReplayError::Worker`] if a parallel worker panicked.
pub fn run_sweep(
    events: &[RecordEnvelope],
    base: &ReplayConfig,
    grid: &SweepGrid,
    rank_metric: SortColumn,
    parallel: bool,
) -> Result<SweepReport, ReplayError> {
    if events.is_empty() {
        return Err(ReplayError::EmptyTape);
    }
    if base.paper.rng_seed.is_none() {
        return Err(ReplayError::UnseededPaper);
    }

    let base_reprice = base.risk.quote_manager.reprice_threshold_theta;
    let base_cancel = base.risk.quote_manager.cancel_market_theta;
    let ratio = if base_reprice > 0.0 {
        base_cancel / base_reprice
    } else {
        1.0
    };

    let min_edges = or_base(&grid.min_edge, base.risk.quote.min_edge);
    let gammas = or_base(&grid.gamma, base.risk.quote.gamma_inventory_skew);
    let cancels = or_base(&grid.cancel_theta, base_reprice);
    let buffers = or_base(&grid.taker_buffer, base.risk.momentum.momentum_buffer);

    // Fixed nested cartesian order (operator order preserved).
    let mut points =
        Vec::with_capacity(min_edges.len() * gammas.len() * cancels.len() * buffers.len());
    for &min_edge in &min_edges {
        for &gamma in &gammas {
            for &cancel_theta in &cancels {
                for &taker_buffer in &buffers {
                    points.push(SweepPoint {
                        min_edge,
                        gamma,
                        cancel_theta,
                        taker_buffer,
                    });
                }
            }
        }
    }

    let mut results = if parallel && points.len() > 1 {
        run_points_parallel(&points, base, events, ratio)
    } else {
        points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, run_one(p, base, events, ratio)))
            .collect()
    };
    if results.len() != points.len() {
        return Err(ReplayError::Worker);
    }

    // Reassemble in cartesian order, then rank.
    results.sort_by_key(|(i, _)| *i);
    let taker_budget = base.analytics.taker_budget_per_window;
    let mut indexed: Vec<(usize, SweepRow)> = Vec::with_capacity(points.len());
    for (i, res) in results {
        let out = res?;
        let aggregate = aggregate(&out.comparison.rows, rank_metric, taker_budget);
        indexed.push((
            i,
            SweepRow {
                point: points[i],
                comparison: out.comparison,
                aggregate,
            },
        ));
    }

    let higher = rank_metric.higher_is_better();
    indexed.sort_by(|(ia, a), (ib, b)| {
        let ord = a
            .aggregate
            .rank_key
            .partial_cmp(&b.aggregate.rank_key)
            .unwrap_or(std::cmp::Ordering::Equal);
        let ord = if higher { ord.reverse() } else { ord };
        ord.then(ia.cmp(ib))
    });
    let rows = indexed.into_iter().map(|(_, r)| r).collect();
    Ok(SweepReport {
        rank_metric,
        points_total: points.len(),
        rows,
    })
}

/// `values` if non-empty, else a single-element vec holding the base value.
fn or_base<T: Copy>(values: &[T], base: T) -> Vec<T> {
    if values.is_empty() {
        vec![base]
    } else {
        values.to_vec()
    }
}

/// Replays one grid point: clones `base`, overrides the four swept fields, and
/// runs. `cancel_market_theta` scales with `reprice_threshold_theta` (preserving
/// the §8 ordering) and never drops below it.
fn run_one(
    point: &SweepPoint,
    base: &ReplayConfig,
    events: &[RecordEnvelope],
    ratio: f64,
) -> Result<ReplayOutput, ReplayError> {
    let mut cfg = base.clone();
    cfg.risk.quote.min_edge = point.min_edge;
    cfg.risk.quote.gamma_inventory_skew = point.gamma;
    cfg.risk.quote_manager.reprice_threshold_theta = point.cancel_theta;
    cfg.risk.quote_manager.cancel_market_theta =
        (point.cancel_theta * ratio).max(point.cancel_theta);
    cfg.risk.momentum.momentum_buffer = point.taker_buffer;
    run_replay(events, &cfg)
}

/// Fans grid points across `available_parallelism() − 2` threads (strided so
/// load is balanced), each with its own runtime. Returns `(index, result)` pairs
/// (unordered); the caller sorts by index. A panicked worker simply omits its
/// points, which the caller detects via the count.
fn run_points_parallel(
    points: &[SweepPoint],
    base: &ReplayConfig,
    events: &[RecordEnvelope],
    ratio: f64,
) -> Vec<(usize, Result<ReplayOutput, ReplayError>)> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(1))
        .unwrap_or(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                scope.spawn(move || {
                    let mut out = Vec::new();
                    let mut i = tid;
                    while i < points.len() {
                        out.push((i, run_one(&points[i], base, events, ratio)));
                        i += threads;
                    }
                    out
                })
            })
            .collect();
        let mut all = Vec::new();
        for handle in handles {
            if let Ok(part) = handle.join() {
                all.extend(part);
            }
        }
        all
    })
}

/// Folds the per-series rows into one across-series aggregate and computes the
/// rank key for `rank`.
fn aggregate(
    rows: &[crate::SeriesComparisonRow],
    rank: SortColumn,
    taker_budget: Dollars,
) -> SweepAggregate {
    let mut windows_traded = 0u32;
    let mut net_pnl = Dollars::ZERO;
    let mut locked_pair_pnl = Dollars::ZERO;
    let mut inventory_pnl = Dollars::ZERO;
    let mut fees_paid = Dollars::ZERO;
    let mut rebates_earned = Dollars::ZERO;
    let mut maker_fills = 0u64;
    let mut taker_fills = 0u64;
    let mut taker_notional = Dollars::ZERO;
    let mut profitable_weight = 0.0f64; // Σ fraction_i · traded_i
    let mut markout_weighted = 0.0f64; // Σ mean_i · n_i
    let mut markout_n = 0u64;
    for r in rows {
        windows_traded = windows_traded.saturating_add(r.windows_traded);
        net_pnl = net_pnl + r.net_pnl;
        locked_pair_pnl = locked_pair_pnl + r.locked_pair_pnl;
        inventory_pnl = inventory_pnl + r.inventory_pnl;
        fees_paid = fees_paid + r.fees_paid;
        rebates_earned = rebates_earned + r.rebates_earned;
        maker_fills = maker_fills.saturating_add(r.maker_fills);
        taker_fills = taker_fills.saturating_add(r.taker_fills);
        taker_notional = taker_notional + r.taker_notional;
        profitable_weight += r.fraction_profitable * f64::from(r.windows_traded);
        if let Some(mean) = r.markout_5s.mean {
            markout_weighted += mean * count_f64(r.markout_5s.n);
            markout_n = markout_n.saturating_add(r.markout_5s.n);
        }
    }

    let traded_f = f64::from(windows_traded);
    let fraction_profitable = if windows_traded > 0 {
        profitable_weight / traded_f
    } else {
        0.0
    };
    let pnl_per_window = if windows_traded > 0 {
        Dollars::new(net_pnl.as_decimal() / Decimal::from(windows_traded))
    } else {
        Dollars::ZERO
    };
    let avg_markout_5s = if markout_n > 0 {
        Some(markout_weighted / count_f64(markout_n))
    } else {
        None
    };
    let budget_used = {
        let budget = taker_budget.as_decimal();
        if windows_traded > 0 && budget > Decimal::ZERO {
            let per_window = taker_notional.as_decimal() / Decimal::from(windows_traded);
            (per_window / budget).to_f64()
        } else {
            None
        }
    };

    let rank_key = match rank {
        SortColumn::NetPnl => dollars_f64(net_pnl),
        SortColumn::PnlPerWindow => dollars_f64(pnl_per_window),
        SortColumn::FractionProfitable => fraction_profitable,
        SortColumn::WindowsTraded => traded_f,
        SortColumn::AvgMarkout5s => avg_markout_5s.unwrap_or(f64::NEG_INFINITY),
        SortColumn::FeesPaid => dollars_f64(fees_paid),
        SortColumn::RebatesEarned => dollars_f64(rebates_earned),
        SortColumn::LockedPairPnl => dollars_f64(locked_pair_pnl),
        SortColumn::InventoryPnl => dollars_f64(inventory_pnl),
        SortColumn::TakerBudgetUsed => budget_used.unwrap_or(f64::NEG_INFINITY),
        SortColumn::MakerFills => count_f64(maker_fills),
        SortColumn::TakerFills => count_f64(taker_fills),
        // Health does not aggregate meaningfully across a sweep — neutral key.
        SortColumn::Health => 0.0,
    };

    SweepAggregate {
        windows_traded,
        fraction_profitable,
        net_pnl,
        pnl_per_window,
        locked_pair_pnl,
        inventory_pnl,
        fees_paid,
        rebates_earned,
        maker_fills,
        taker_fills,
        taker_notional,
        avg_markout_5s,
        rank_key,
    }
}

/// Dollars → f64 for an orderable (never displayed) sort key.
fn dollars_f64(d: Dollars) -> f64 {
    d.as_decimal().to_f64().unwrap_or(0.0)
}

/// A fill/window count → f64 (counts are tiny relative to the f64 mantissa).
#[allow(
    clippy::cast_precision_loss,
    reason = "counts are small; key is only compared"
)]
fn count_f64(n: u64) -> f64 {
    n as f64
}
