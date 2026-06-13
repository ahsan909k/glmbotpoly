//! The thin async shell around the machines: owns the clock reads, the
//! discovery worker, the bus sends, and all logging. Per-series failures
//! never propagate — the driver only errors when the process around it is
//! structurally gone (bus or worker dead).

use std::future::Future;
use std::time::Duration;

use core_types::{Event, MarketLifecycleEvent, Series, TimestampMs};
use discovery::{DiscoveryApi, DiscoveryError, DiscoveryService, SeriesWindows};
use tokio::sync::{mpsc, watch};

use crate::error::SchedulerError;
use crate::machine::{CoverageWarning, Output, SeriesMachine, Timing};
use crate::status::SchedulerStatus;

/// Capacity of the refresh request/response channels. The machines'
/// in-flight flags guarantee at most one outstanding request per series, so
/// this can never legitimately fill with six series.
const REFRESH_QUEUE_CAP: usize = 16;

/// Sleep cap between ticks. Bounds the damage from wall-clock steps and
/// suspends — after any oversleep the catch-up tick replays missed
/// boundaries; six no-op ticks per second cost nothing.
const MAX_TICK_MS: i64 = 1_000;

/// The narrow discovery seam the driver needs. Implemented by
/// [`DiscoveryService`]; driver tests fake it at the [`SeriesWindows`] level
/// so no wire types are involved.
pub trait Refresh: Send {
    /// Refreshes one series' windows at `now`. Mirrors
    /// [`DiscoveryService::refresh_series`].
    fn refresh(
        &mut self,
        series: Series,
        now: TimestampMs,
    ) -> impl Future<Output = Result<SeriesWindows, DiscoveryError>> + Send;
}

impl<A: DiscoveryApi + Send + Sync> Refresh for DiscoveryService<A> {
    async fn refresh(
        &mut self,
        series: Series,
        now: TimestampMs,
    ) -> Result<SeriesWindows, DiscoveryError> {
        self.refresh_series(series, now).await
    }
}

/// Everything [`run`] needs. Construct one per scheduler instance.
pub struct SchedulerArgs<R, F> {
    /// Timing policy (`expect_resolutions` is overridden from `market_rx`
    /// presence, so the two can never disagree).
    pub timing: Timing,
    /// The series to schedule (the bot passes `engine.enabled_series()`).
    pub series: Vec<Series>,
    /// The discovery seam.
    pub refresher: R,
    /// Wall-clock source; the only clock surface in the crate.
    pub now_fn: F,
    /// The internal bus. Sends are awaited — backpressure is explicit.
    pub bus_tx: mpsc::Sender<Event>,
    /// Market-channel lifecycle events from feed-clob; `None` until that
    /// crate exists (Day 3), which also disables resolution-overdue warnings.
    pub market_rx: Option<mpsc::Receiver<MarketLifecycleEvent>>,
    /// Optional coverage-status snapshots (smoke table, dashboard later).
    pub status_tx: Option<watch::Sender<SchedulerStatus>>,
    /// Flip to `true` for graceful shutdown.
    pub shutdown_rx: watch::Receiver<bool>,
}

/// Runs the scheduler until shutdown.
///
/// # Errors
/// [`SchedulerError::BusClosed`] when every bus receiver is gone;
/// [`SchedulerError::WorkerGone`] if the discovery worker dies. Per-series
/// trouble is logged and retried, never returned (CLAUDE.md §6).
pub async fn run<R, F>(args: SchedulerArgs<R, F>) -> Result<(), SchedulerError>
where
    R: Refresh + 'static,
    F: Fn() -> TimestampMs + Send + 'static,
{
    let SchedulerArgs {
        mut timing,
        series,
        refresher,
        now_fn,
        bus_tx,
        mut market_rx,
        status_tx,
        mut shutdown_rx,
    } = args;
    timing.expect_resolutions = market_rx.is_some();

    let (req_tx, req_rx) = mpsc::channel::<RefreshRequest>(REFRESH_QUEUE_CAP);
    let (res_tx, mut res_rx) = mpsc::channel::<(Series, Option<SeriesWindows>)>(REFRESH_QUEUE_CAP);
    let worker = tokio::spawn(refresh_worker(refresher, req_rx, res_tx));

    let mut machines: Vec<SeriesMachine> = series
        .iter()
        .map(|&s| SeriesMachine::new(s, timing))
        .collect();
    let mut out: Vec<Output> = Vec::new();

    let result = loop {
        let now = now_fn();
        if let Err(err) = tick_all(&mut machines, &mut out, now, &bus_tx, &req_tx).await {
            break Err(err);
        }
        if let Some(tx) = &status_tx {
            tx.send_replace(SchedulerStatus {
                at: now,
                series: machines.iter().map(|m| m.status(now)).collect(),
            });
        }

        let deadline = machines
            .iter()
            .filter_map(SeriesMachine::next_deadline)
            .min();
        let sleep_ms = deadline.map_or(MAX_TICK_MS, |d| {
            d.as_millis()
                .saturating_sub(now.as_millis())
                .clamp(0, MAX_TICK_MS)
        });

        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(sleep_ms.unsigned_abs())) => {}
            res = res_rx.recv() => match res {
                Some((series, payload)) => {
                    let now = now_fn();
                    if let Some(idx) = machines.iter().position(|m| m.series() == series) {
                        machines[idx].on_discovery(payload, now, &mut out);
                        if let Err(err) = handle_outputs(series, &mut out, now, &bus_tx, &req_tx).await {
                            break Err(err);
                        }
                    }
                }
                None => break Err(SchedulerError::WorkerGone),
            },
            ev = recv_market(&mut market_rx) => match ev {
                Some(event) => {
                    let now = now_fn();
                    match offer_market_event(&mut machines, &event, &mut out, now, &bus_tx, &req_tx).await {
                        Ok(claimed) => {
                            if !claimed {
                                tracing::debug!(target: "scheduler", ?event, "market event matched no scheduled series — dropped");
                            }
                        }
                        Err(err) => break Err(err),
                    }
                }
                None => {
                    tracing::warn!(target: "scheduler", "market event channel closed — continuing on wall clock + discovery only");
                    market_rx = None;
                }
            },
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    tracing::info!(target: "scheduler", "shutdown requested");
                    break Ok(());
                }
            }
        }
    };

    drop(req_tx);
    let _ = worker.await;
    result
}

/// Ticks every machine and acts on the outputs.
async fn tick_all(
    machines: &mut [SeriesMachine],
    out: &mut Vec<Output>,
    now: TimestampMs,
    bus_tx: &mpsc::Sender<Event>,
    req_tx: &mpsc::Sender<RefreshRequest>,
) -> Result<(), SchedulerError> {
    for machine in machines.iter_mut() {
        machine.on_tick(now, out);
        handle_outputs(machine.series(), out, now, bus_tx, req_tx).await?;
    }
    Ok(())
}

/// Offers a market event to every machine; `Ok(true)` when one claimed it.
async fn offer_market_event(
    machines: &mut [SeriesMachine],
    event: &MarketLifecycleEvent,
    out: &mut Vec<Output>,
    now: TimestampMs,
    bus_tx: &mpsc::Sender<Event>,
    req_tx: &mpsc::Sender<RefreshRequest>,
) -> Result<bool, SchedulerError> {
    let mut claimed = false;
    for machine in machines.iter_mut() {
        claimed |= machine.on_market_event(event, now, out);
        handle_outputs(machine.series(), out, now, bus_tx, req_tx).await?;
    }
    Ok(claimed)
}

/// One refresh request to the worker.
struct RefreshRequest {
    series: Series,
    now: TimestampMs,
}

/// Owns the [`Refresh`] implementation and serves requests serially.
/// Refreshes are rare (a handful per window per series), so serialization
/// costs nothing; what matters is that HTTP latency lives on this task, not
/// on the tick loop.
async fn refresh_worker<R: Refresh>(
    mut refresher: R,
    mut req_rx: mpsc::Receiver<RefreshRequest>,
    res_tx: mpsc::Sender<(Series, Option<SeriesWindows>)>,
) {
    while let Some(req) = req_rx.recv().await {
        let payload = match refresher.refresh(req.series, req.now).await {
            Ok(windows) => Some(windows),
            Err(error) => {
                tracing::warn!(
                    target: "scheduler",
                    series = req.series.key(),
                    %error,
                    "discovery refresh failed"
                );
                None
            }
        };
        if res_tx.send((req.series, payload)).await.is_err() {
            return; // driver gone — nothing left to serve
        }
    }
}

/// Receives one market event, or never resolves when no channel is attached.
async fn recv_market(
    rx: &mut Option<mpsc::Receiver<MarketLifecycleEvent>>,
) -> Option<MarketLifecycleEvent> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// Acts on a machine's outputs: announcements to the bus (and the log),
/// refresh requests to the worker, warnings to the log.
async fn handle_outputs(
    series: Series,
    out: &mut Vec<Output>,
    now: TimestampMs,
    bus_tx: &mpsc::Sender<Event>,
    req_tx: &mpsc::Sender<RefreshRequest>,
) -> Result<(), SchedulerError> {
    for output in out.drain(..) {
        match output {
            Output::Announce { market, lifecycle } => {
                tracing::info!(
                    target: "scheduler",
                    series = series.key(),
                    window = %market.window,
                    slug = %market.event_slug,
                    lifecycle = ?lifecycle,
                    closes_in_ms = market.close_time.signed_duration_since(now).as_millis(),
                    "window lifecycle"
                );
                bus_tx
                    .send(Event::Window { market, lifecycle })
                    .await
                    .map_err(|_| SchedulerError::BusClosed)?;
            }
            Output::Refresh { reason } => {
                tracing::debug!(
                    target: "scheduler",
                    series = series.key(),
                    reason = ?reason,
                    "requesting discovery refresh"
                );
                if req_tx.try_send(RefreshRequest { series, now }).is_err() {
                    // Should be impossible (≤1 in flight per series); the
                    // machine's watchdog recovers if it ever happens.
                    tracing::error!(
                        target: "scheduler",
                        series = series.key(),
                        "refresh request channel full or closed — relying on watchdog retry"
                    );
                }
            }
            Output::Warn(warning) => log_warning(series, &warning),
        }
    }
    Ok(())
}

/// Coverage failures (§6) log at ERROR; anomalies log at WARN.
fn log_warning(series: Series, warning: &CoverageWarning) {
    match warning {
        CoverageWarning::ContractViolated { window, close } => {
            tracing::error!(
                target: "scheduler",
                series = series.key(),
                window = %window,
                close = close.as_millis(),
                "§6 CONTRACT VIOLATED: next window unknown inside the final lead — retrying with backoff"
            );
        }
        CoverageWarning::NoNextAtClose { window } => {
            tracing::error!(
                target: "scheduler",
                series = series.key(),
                window = %window,
                "window closed with NO known successor — coverage gap, hunting"
            );
        }
        CoverageWarning::ResolutionOverdue { window, closed_at } => {
            tracing::warn!(
                target: "scheduler",
                series = series.key(),
                window = %window,
                closed_at = closed_at.as_millis(),
                "no market_resolved yet for closed window"
            );
        }
        CoverageWarning::ResolutionBeforeClose { window } => {
            tracing::warn!(
                target: "scheduler",
                series = series.key(),
                window = %window,
                "market_resolved arrived before our clock says the window closed"
            );
        }
        CoverageWarning::UnknownWinningToken { window } => {
            tracing::warn!(
                target: "scheduler",
                series = series.key(),
                window = %window,
                "winning token id matches neither outcome token — refusing to fabricate an outcome"
            );
        }
        CoverageWarning::CurrentSuperseded { old, new } => {
            tracing::warn!(
                target: "scheduler",
                series = series.key(),
                old = %old,
                new = %new,
                "discovery reports a different current window — rolled ours"
            );
        }
        CoverageWarning::ParkedEvicted { window } => {
            tracing::warn!(
                target: "scheduler",
                series = series.key(),
                window = %window,
                "parked-resolution cap reached — evicted oldest unresolved window"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::WindowLifecycle;

    use super::*;
    use crate::testutil::{BTC_5M, market, timing};

    /// One minute into an aligned 5m/15m/1h boundary, so synthetic windows
    /// land on the same grid the venue uses.
    const BASE_MS: i64 = 1_800_000_060_000;

    /// Serves perfectly aligned windows computed from `now` — the venue as
    /// it should be, forever.
    struct FakeRefresh;

    impl Refresh for FakeRefresh {
        async fn refresh(
            &mut self,
            series: Series,
            now: TimestampMs,
        ) -> Result<SeriesWindows, DiscoveryError> {
            let dur = series.duration.as_duration().as_millis();
            let open = now.as_millis() - now.as_millis().rem_euclid(dur);
            Ok(SeriesWindows {
                series,
                current: Some(market(series, open)),
                upcoming: vec![market(series, open + dur), market(series, open + 2 * dur)],
                fetched_at: now,
            })
        }
    }

    /// Wall clock locked to the (paused) tokio clock.
    fn paused_now_fn() -> impl Fn() -> TimestampMs + Send {
        let start = tokio::time::Instant::now();
        move || {
            let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
            TimestampMs::from_millis(BASE_MS + elapsed)
        }
    }

    #[tokio::test(start_paused = true)]
    async fn boots_announces_and_rolls_across_a_boundary_then_shuts_down() {
        let (bus_tx, mut bus_rx) = mpsc::channel(256);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (status_tx, status_rx) = watch::channel(SchedulerStatus::default());
        let handle = tokio::spawn(run(SchedulerArgs {
            timing: timing(false),
            series: vec![BTC_5M],
            refresher: FakeRefresh,
            now_fn: paused_now_fn(),
            bus_tx,
            market_rx: None,
            status_tx: Some(status_tx),
            shutdown_rx,
        }));

        // Watch one full boundary go by (boot at +60 s, close at +240 s).
        let deadline = tokio::time::Instant::now() + Duration::from_secs(360);
        let mut seen: Vec<(i64, WindowLifecycle)> = Vec::new();
        loop {
            tokio::select! {
                ev = bus_rx.recv() => match ev {
                    Some(Event::Window { market, lifecycle }) => {
                        seen.push((market.window.open_time.as_millis(), lifecycle));
                    }
                    Some(_) => {}
                    None => break,
                },
                () = tokio::time::sleep_until(deadline) => break,
            }
        }

        let w0 = BASE_MS - 60_000;
        let w1 = w0 + 300_000;
        let w2 = w0 + 600_000;
        // Boot: every fetched window announced, then the current one opens.
        assert_eq!(
            &seen[..4],
            &[
                (w0, WindowLifecycle::Discovered),
                (w1, WindowLifecycle::Discovered),
                (w2, WindowLifecycle::Discovered),
                (w0, WindowLifecycle::Open),
            ],
            "boot sequence"
        );
        // The boundary: Closing, then Closed and the successor's Open
        // back-to-back — zero coverage gap.
        let closing = seen
            .iter()
            .position(|e| *e == (w0, WindowLifecycle::Closing))
            .expect("closing announced");
        let closed = seen
            .iter()
            .position(|e| *e == (w0, WindowLifecycle::Closed))
            .expect("closed announced");
        assert!(closing < closed);
        assert_eq!(
            seen.get(closed + 1),
            Some(&(w1, WindowLifecycle::Open)),
            "the next window opens in the same instant the last one closes"
        );
        // No resolution source attached: nothing may claim Resolved.
        assert!(
            seen.iter()
                .all(|(_, l)| !matches!(l, WindowLifecycle::Resolved { .. })),
            "no Resolved without a market-event source"
        );
        // Status reflects healthy coverage.
        let status = status_rx.borrow().clone();
        assert_eq!(status.series.len(), 1);
        assert!(status.series[0].contract_ok);
        assert!(status.series[0].next_known);

        // Graceful shutdown.
        shutdown_tx.send(true).expect("driver alive");
        let result = handle.await.expect("driver task not panicked");
        assert!(matches!(result, Ok(())));
    }

    #[tokio::test(start_paused = true)]
    async fn dropped_bus_is_a_fatal_bus_closed_error() {
        let (bus_tx, bus_rx) = mpsc::channel(8);
        drop(bus_rx);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let handle = tokio::spawn(run(SchedulerArgs {
            timing: timing(false),
            series: vec![BTC_5M],
            refresher: FakeRefresh,
            now_fn: paused_now_fn(),
            bus_tx,
            market_rx: None,
            status_tx: None,
            shutdown_rx,
        }));
        let result = tokio::time::timeout(Duration::from_secs(60), handle)
            .await
            .expect("driver exits promptly")
            .expect("driver task not panicked");
        assert!(matches!(result, Err(SchedulerError::BusClosed)));
    }
}
