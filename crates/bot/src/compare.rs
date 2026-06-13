//! `bot compare` — the live feed-comparison run (read-only): both feed
//! drivers (RTDS + direct Binance) on one bus, the
//! [`feed_binance::compare::Comparator`] consuming every tick, and a
//! per-asset summary table — tick rates, value bases, exact-value match
//! lags, cross-correlation lead/lag — printed every minute. Health
//! transitions print as they happen. Runs until ctrl-c.
//!
//! This is the §8 groundwork measurement: which feed actually leads, by how
//! much, and what RTDS-Binance actually republishes (its value transitions
//! matching `direct:trade` vs `direct:mid`) — the numbers that calibrate the
//! model's basis correction.

use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{Event, TimestampMs};
use feed_binance::compare::{AssetSummary, Comparator, WINDOW};
use feed_binance::{BinanceArgs, BinanceSub};
use feed_rtds::{FeedSub, RtdsArgs};
use feed_util::{FeedError, WsTransport};
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::discover::fmt_ts;
use crate::feed::{binance_params, print_health, rtds_params};

/// How often the comparison summary prints.
const SUMMARY_PERIOD: Duration = Duration::from_secs(60);

type DriverResult = Result<Result<(), FeedError>, tokio::task::JoinError>;

/// Builds the runtime, spawns both feed drivers onto one bus, and prints the
/// rolling comparison until ctrl-c.
pub fn execute(config: &AppConfig) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run_compare(config))
}

async fn run_compare(config: &AppConfig) -> anyhow::Result<()> {
    let rtds = rtds_params(&config.feeds);
    let binance = binance_params(&config.feeds);
    tracing::info!(
        target: "compare",
        rtds_url = %rtds.url,
        binance_url = %binance.url,
        summary_period_s = SUMMARY_PERIOD.as_secs(),
        window_s = WINDOW.as_millis() / 1_000,
        "feed comparison starting (read-only; ctrl-c to stop)"
    );

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let mut rtds_task: JoinHandle<Result<(), FeedError>> = tokio::spawn(feed_rtds::run(RtdsArgs {
        params: rtds,
        subscriptions: FeedSub::all(),
        transport: WsTransport,
        now_fn: wall_now,
        bus_tx: bus_tx.clone(),
        command_rx: None,
        status_tx: None,
        shutdown_rx: shutdown_rx.clone(),
        backoff_seed: None,
    }));
    let mut binance_task: JoinHandle<Result<(), FeedError>> =
        tokio::spawn(feed_binance::run(BinanceArgs {
            params: binance,
            subscriptions: BinanceSub::all(),
            transport: WsTransport,
            now_fn: wall_now,
            bus_tx,
            status_tx: None,
            shutdown_rx,
            backoff_seed: None,
        }));

    let mut comparator = Comparator::new();
    let mut ticks: u64 = 0;
    let started = wall_now();
    let mut interval = tokio::time::interval(SUMMARY_PERIOD);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    interval.reset(); // no empty summary at t=0

    let mut rtds_result: Option<DriverResult> = None;
    let mut binance_result: Option<DriverResult> = None;

    // Main phase: until ctrl-c, or a driver exits on its own (fatal).
    loop {
        tokio::select! {
            joined = &mut rtds_task, if rtds_result.is_none() => {
                rtds_result = Some(joined);
                tracing::warn!(target: "compare", "rtds driver exited — shutting down");
                break;
            }
            joined = &mut binance_task, if binance_result.is_none() => {
                binance_result = Some(joined);
                tracing::warn!(target: "compare", "binance driver exited — shutting down");
                break;
            }
            maybe = bus_rx.recv() => match maybe {
                Some(Event::PriceTick(tick)) => {
                    ticks += 1;
                    comparator.on_tick(&tick);
                }
                Some(Event::FeedHealth(health)) => print_health(health),
                Some(_) => {}
                None => break,
            },
            _ = interval.tick() => print_summaries(&comparator.summary(wall_now()), wall_now()),
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "compare", "ctrl-c — shutting down");
                break;
            }
        }
    }

    // Drain phase: signal shutdown and keep the bus flowing so neither
    // driver can deadlock on a full channel while exiting.
    let _ = shutdown_tx.send(true);
    let mut bus_open = true;
    while rtds_result.is_none() || binance_result.is_none() {
        tokio::select! {
            joined = &mut rtds_task, if rtds_result.is_none() => rtds_result = Some(joined),
            joined = &mut binance_task, if binance_result.is_none() => binance_result = Some(joined),
            maybe = bus_rx.recv(), if bus_open => bus_open = maybe.is_some(),
        }
    }

    print_summaries(&comparator.summary(wall_now()), wall_now());
    let ran_for = wall_now().signed_duration_since(started);
    println!(
        "\nrun summary: {ticks} price ticks over {}s",
        ran_for.as_millis() / 1000
    );
    for (name, result) in [("rtds", rtds_result), ("binance", binance_result)] {
        result
            .unwrap_or_else(|| unreachable!("drain loop joined both drivers"))
            .with_context(|| format!("{name} feed driver panicked"))?
            .with_context(|| format!("{name} feed driver failed"))?;
    }
    Ok(())
}

/// The per-asset comparison tables (stdout, discover-table precedent).
fn print_summaries(summaries: &[AssetSummary], now: TimestampMs) {
    println!(
        "\n=== feed comparison at {} (rolling {} min window) ===",
        fmt_ts(now),
        WINDOW.as_millis() / 60_000
    );
    if summaries.is_empty() {
        println!("(no ticks buffered yet)");
        return;
    }
    for summary in summaries {
        println!("\n{}", summary.asset.ticker());
        println!(
            "  {:<14} {:>7} {:>9} {:>14} {:>8}",
            "stream", "ticks", "rate", "latest", "age"
        );
        for s in &summary.streams {
            println!(
                "  {:<14} {:>7} {:>8.1}/s {:>14} {:>8}",
                s.id.to_string(),
                s.ticks,
                s.rate_hz,
                s.latest.map_or_else(|| "-".to_owned(), |v| v.to_string()),
                s.age.map_or_else(
                    || "-".to_owned(),
                    |a| format!("{:.1}s", a.as_millis() as f64 / 1000.0)
                ),
            );
        }
        for b in &summary.bases {
            println!(
                "  basis  {:<12} vs {:<12} mean {:+8.2} bps   p50 {:+8.2} bps   (n={})",
                b.a.to_string(),
                b.b.to_string(),
                b.mean_bps,
                b.p50_bps,
                b.samples
            );
        }
        for x in &summary.xcorr {
            println!(
                "  xcorr  {:<12} vs {:<12} {}   (corr {:.3}, n={})",
                x.a.to_string(),
                x.b.to_string(),
                lead_phrase(x.lag_ms),
                x.correlation,
                x.samples
            );
        }
        for m in &summary.match_lags {
            println!(
                "  match  {:<12} -> {:<12} p50 {:>5} ms   p95 {:>5} ms   ({} matched, {:.0}%)",
                m.a.to_string(),
                m.b.to_string(),
                m.p50_ms,
                m.p95_ms,
                m.matches,
                m.match_rate * 100.0
            );
        }
    }
}

/// Renders a signed lag as a lead/trail phrase (positive = first-named
/// stream leads).
fn lead_phrase(lag_ms: i64) -> String {
    match lag_ms {
        0 => "in sync (lag 0 ms)".to_owned(),
        ms if ms > 0 => format!("first leads by {ms} ms"),
        ms => format!("first TRAILS by {} ms", -ms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lead_phrase_signs() {
        assert_eq!(lead_phrase(700), "first leads by 700 ms");
        assert_eq!(lead_phrase(-300), "first TRAILS by 300 ms");
        assert_eq!(lead_phrase(0), "in sync (lag 0 ms)");
    }
}
