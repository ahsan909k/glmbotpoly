//! `bot schedule` — the live scheduler smoke run (read-only): drives the
//! per-series lifecycle state machines against real Gamma/CLOB discovery for
//! every enabled series, logging each lifecycle announcement and printing a
//! per-series coverage table every 30 seconds. The §11 clock-skew monitor
//! runs alongside on the same bus (periodic NTP rounds; a trip surfaces as a
//! `Risk` event here, and as a cancel-all once the risk manager exists). No
//! market-event source is attached yet (feed-clob is a later task), so
//! windows park unresolved and no `Resolved` events appear — expected until
//! Day 3. Runs until ctrl-c.

use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{DurationMs, Event};
use scheduler::{SchedulerArgs, SchedulerStatus, Timing};
use timeutil::{NtpOffsetSource, SkewMonitorArgs, SystemClock, run_skew_monitor, wall_now};
use tokio::sync::{mpsc, watch};

use crate::discover::{fmt_countdown, fmt_ts};
use crate::timecfg::{ntp_params, skew_params, std_duration};

/// How often the coverage table prints.
const STATUS_PERIOD: Duration = Duration::from_secs(30);

/// Builds the runtime, spawns the scheduler driver + clock-skew monitor over
/// live discovery/NTP, and consumes the bus until ctrl-c.
pub fn execute(config: &AppConfig) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run_smoke(config))
}

async fn run_smoke(config: &AppConfig) -> anyhow::Result<()> {
    let service = discovery::DiscoveryService::from_config(&config.feeds, &config.discovery)
        .context("building discovery service")?;
    let series = config.engine.enabled_series();
    tracing::info!(
        target: "schedule",
        series = ?series.iter().map(|s| s.key()).collect::<Vec<_>>(),
        "scheduler smoke run starting (read-only; ctrl-c to stop)"
    );

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (status_tx, status_rx) = watch::channel(SchedulerStatus::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // §12.6 integration: the clock-skew monitor shares the scheduler's bus
    // and shutdown signal; its trips/clears arrive as Event::Risk below.
    let mut skew_task = tokio::spawn(run_skew_monitor(SkewMonitorArgs {
        params: skew_params(&config.clock),
        check_interval: std_duration(config.clock.check_interval_ms),
        source: NtpOffsetSource::new(ntp_params(&config.clock), SystemClock::new()),
        clock: SystemClock::new(),
        bus_tx: bus_tx.clone(),
        shutdown_rx: shutdown_tx.subscribe(),
    }));

    let mut driver = tokio::spawn(scheduler::run(SchedulerArgs {
        timing: Timing::from_config(&config.scheduler),
        series,
        refresher: service,
        now_fn: wall_now,
        bus_tx,
        market_rx: None,
        status_tx: Some(status_tx),
        shutdown_rx,
    }));

    let mut interval = tokio::time::interval(STATUS_PERIOD);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut announcements: u64 = 0;
    let mut risk_events: u64 = 0;
    let started = wall_now();

    let driver_result = loop {
        tokio::select! {
            joined = &mut driver => break joined,
            maybe = bus_rx.recv() => {
                // The drivers already log their events; here we only count
                // them for the run summary (risk events also print, §11).
                match maybe {
                    Some(Event::Risk(risk)) => {
                        risk_events += 1;
                        println!("RISK EVENT: {risk:?}");
                    }
                    Some(_) => announcements += 1,
                    None => break (&mut driver).await,
                }
            }
            _ = interval.tick() => print_status(&status_rx.borrow()),
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "schedule", "ctrl-c — shutting down");
                let _ = shutdown_tx.send(true);
                // Keep draining the bus so the driver can never deadlock on
                // a full channel while exiting.
                break loop {
                    tokio::select! {
                        joined = &mut driver => break joined,
                        _ = bus_rx.recv() => {}
                    }
                };
            }
        }
    };

    // Stop the skew monitor too (the scheduler may have exited on its own
    // without anyone signalling shutdown).
    let _ = shutdown_tx.send(true);
    match tokio::time::timeout(Duration::from_secs(5), &mut skew_task).await {
        Ok(joined) => {
            if let Ok(Err(error)) = joined {
                tracing::warn!(target: "schedule", %error, "skew monitor exited with error");
            }
        }
        Err(_) => skew_task.abort(),
    }

    let ran_for = wall_now().signed_duration_since(started);
    println!(
        "\nrun summary: {announcements} lifecycle announcements, {risk_events} risk events over {}",
        fmt_countdown(ran_for)
    );
    driver_result
        .context("scheduler driver panicked")?
        .context("scheduler driver failed")?;
    Ok(())
}

/// One row per series: phase, current window, countdown, coverage health.
fn print_status(status: &SchedulerStatus) {
    println!("\ncoverage at {}", fmt_ts(status.at));
    println!(
        "{:<8} {:<12} {:<28} {:>9}  {:<10} {:>7} {:>11}  §6",
        "series", "phase", "current window", "closes", "next", "parked", "refresh"
    );
    for s in &status.series {
        println!(
            "{:<8} {:<12} {:<28} {:>9}  {:<10} {:>7} {:>11}  {}",
            s.series.key(),
            s.phase,
            s.current_slug.as_deref().unwrap_or("-"),
            s.closes_in_ms.map_or_else(
                || "-".to_owned(),
                |ms| fmt_countdown(DurationMs::from_millis(ms))
            ),
            if s.next_known {
                s.next_opens_in_ms.map_or_else(
                    || "known".to_owned(),
                    |ms| format!("in {}", fmt_countdown(DurationMs::from_millis(ms))),
                )
            } else {
                "UNKNOWN".to_owned()
            },
            s.parked,
            s.refresh_age_ms
                .map_or_else(|| "-".to_owned(), |ms| format!("{}s ago", ms / 1000)),
            if s.contract_ok { "ok" } else { "VIOLATED" },
        );
    }
}
