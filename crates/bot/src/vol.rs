//! `bot vol` — the live vol-estimator smoke run (read-only): both feed
//! drivers (RTDS + direct Binance) on one bus feeding four
//! [`model::VolEstimator`] lanes — (BTC, ETH) × (chainlink 1 Hz vendor
//! ticks, direct-Binance bookTicker midpoints) — with a per-lane σ_1s table
//! printed every second. Health transitions print as they happen. Runs
//! until ctrl-c.
//!
//! This demonstrates the model::vol integration end to end: the chainlink
//! lanes go Ready within seconds (the RTDS backfill pre-seeds ~60–120
//! one-second bars), the binance lanes warm up over ~`warmup_returns`
//! seconds of live data, and the half-life/floor/cap parameters flow from
//! `config.engine.defaults` at the binary boundary (feed.rs precedent — the
//! model crate itself never sees the config crate, §4).

use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{Asset, Event, PriceSource, TickKind, TimestampMs};
use feed_binance::{BinanceArgs, BinanceSub};
use feed_rtds::{FeedSub, RtdsArgs};
use feed_util::{FeedError, WsTransport};
use model::{Clamp, IgnoreReason, TickOutcome, VolEstimator, VolParams, VolQuality};
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::discover::fmt_ts;
use crate::feed::{binance_params, print_health, rtds_params};

/// How often the per-lane table prints.
const STATUS_PERIOD: Duration = Duration::from_secs(1);

type DriverResult = Result<Result<(), FeedError>, tokio::task::JoinError>;

/// One estimator plus its outcome tallies for the table.
struct Lane {
    estimator: VolEstimator,
    folds: u64,
    reanchors: u64,
    /// Meaningful drops only (bad value / out-of-order / glitch timestamp);
    /// `WrongStream` is the normal cross-lane case and not counted.
    drops: u64,
}

impl Lane {
    fn new(asset: Asset, params: VolParams) -> anyhow::Result<Self> {
        let estimator = VolEstimator::new(asset, params)
            .with_context(|| format!("building {asset} vol estimator"))?;
        Ok(Self {
            estimator,
            folds: 0,
            reanchors: 0,
            drops: 0,
        })
    }

    fn observe(&mut self, outcome: TickOutcome) {
        match outcome {
            TickOutcome::Folded { .. } => self.folds += 1,
            TickOutcome::Reanchored { .. } => self.reanchors += 1,
            TickOutcome::Ignored(IgnoreReason::WrongStream) => {}
            TickOutcome::Ignored(_) => self.drops += 1,
            TickOutcome::Absorbed | TickOutcome::Opened => {}
        }
    }
}

/// Maps the operator's vol parameters from `engine.defaults` onto one input
/// stream. The input selection and the warmup/gap policy are model-crate
/// defaults for now (config exposure is the model-driver task's job). Shared
/// with `bot fair`.
pub(crate) fn vol_params(
    engine: &config::EngineParams,
    source: PriceSource,
    kind: TickKind,
) -> VolParams {
    VolParams {
        source,
        kind,
        half_life_secs: engine.ewma_half_life_secs,
        floor_1s: engine.vol_floor_1s,
        cap_1s: engine.vol_cap_1s,
        ..VolParams::default()
    }
}

/// The four smoke lanes: both assets on the resolution-grade chainlink feed
/// and on the fast direct-Binance mid feed.
fn build_lanes(config: &AppConfig) -> anyhow::Result<Vec<Lane>> {
    let inputs = [
        (PriceSource::ChainlinkRtds, TickKind::Vendor),
        (PriceSource::BinanceDirect, TickKind::Mid),
    ];
    let mut lanes = Vec::with_capacity(4);
    for asset in Asset::ALL {
        for (source, kind) in inputs {
            lanes.push(Lane::new(
                asset,
                vol_params(&config.engine.defaults, source, kind),
            )?);
        }
    }
    Ok(lanes)
}

/// Builds the runtime, spawns both feed drivers onto one bus, and prints the
/// per-lane vol table until ctrl-c.
pub fn execute(config: &AppConfig) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run_vol(config))
}

async fn run_vol(config: &AppConfig) -> anyhow::Result<()> {
    let rtds = rtds_params(&config.feeds);
    let binance = binance_params(&config.feeds);
    let mut lanes = build_lanes(config)?;
    tracing::info!(
        target: "vol",
        rtds_url = %rtds.url,
        binance_url = %binance.url,
        half_life_s = config.engine.defaults.ewma_half_life_secs,
        floor = config.engine.defaults.vol_floor_1s,
        cap = config.engine.defaults.vol_cap_1s,
        "vol estimator smoke run starting (read-only; ctrl-c to stop)"
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

    let mut ticks: u64 = 0;
    let started = wall_now();
    let mut interval = tokio::time::interval(STATUS_PERIOD);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut rtds_result: Option<DriverResult> = None;
    let mut binance_result: Option<DriverResult> = None;

    // Main phase: until ctrl-c, or a driver exits on its own (fatal).
    loop {
        tokio::select! {
            joined = &mut rtds_task, if rtds_result.is_none() => {
                rtds_result = Some(joined);
                tracing::warn!(target: "vol", "rtds driver exited — shutting down");
                break;
            }
            joined = &mut binance_task, if binance_result.is_none() => {
                binance_result = Some(joined);
                tracing::warn!(target: "vol", "binance driver exited — shutting down");
                break;
            }
            maybe = bus_rx.recv() => match maybe {
                Some(Event::PriceTick(tick)) => {
                    ticks += 1;
                    for lane in &mut lanes {
                        let outcome = lane.estimator.on_tick(&tick);
                        lane.observe(outcome);
                    }
                }
                Some(Event::FeedHealth(health)) => print_health(health),
                Some(_) => {}
                None => break,
            },
            _ = interval.tick() => print_lanes(&lanes, wall_now()),
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "vol", "ctrl-c — shutting down");
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

    print_lanes(&lanes, wall_now());
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

/// The per-lane table (stdout, feed-status precedent).
fn print_lanes(lanes: &[Lane], now: TimestampMs) {
    println!("\nvol at {}", fmt_ts(now));
    println!(
        "{:<6} {:<16} {:>12} {:>10} {:>8} {:>6} {:>8} {:>6} {:>9} {:>6}",
        "asset",
        "input",
        "sigma_1s",
        "bps/rt-s",
        "quality",
        "clamp",
        "returns",
        "folds",
        "reanchors",
        "drops"
    );
    for lane in lanes {
        let estimate = lane.estimator.estimate();
        let (sigma, bps) = estimate.sigma_1s.map_or_else(
            || ("-".to_owned(), "-".to_owned()),
            |s| (format!("{s:.3e}"), format!("{:.2}", s * 10_000.0)),
        );
        println!(
            "{:<6} {:<16} {:>12} {:>10} {:>8} {:>6} {:>8} {:>6} {:>9} {:>6}",
            lane.estimator.asset().ticker(),
            input_label(lane.estimator.params().source, lane.estimator.params().kind),
            sigma,
            bps,
            match estimate.quality {
                VolQuality::Ready => "READY",
                VolQuality::NotReady => "warming",
            },
            match estimate.clamp {
                Clamp::Inactive => "-",
                Clamp::Floor => "FLOOR",
                Clamp::Cap => "CAP",
            },
            estimate.returns_observed,
            lane.folds,
            lane.reanchors,
            lane.drops,
        );
    }
}

/// Short stable label for an input stream, e.g. `chainlink:vendor`.
fn input_label(source: PriceSource, kind: TickKind) -> String {
    let source = match source {
        PriceSource::ChainlinkRtds => "chainlink",
        PriceSource::BinanceRtds => "binance-rtds",
        PriceSource::BinanceDirect => "binance",
    };
    let kind = match kind {
        TickKind::Vendor => "vendor",
        TickKind::Mid => "mid",
        TickKind::Trade => "trade",
    };
    format!("{source}:{kind}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_defaults_map_onto_vol_params() {
        let engine = config::EngineParams::default();
        let params = vol_params(&engine, PriceSource::ChainlinkRtds, TickKind::Vendor);
        assert_eq!(params.source, PriceSource::ChainlinkRtds);
        assert_eq!(params.kind, TickKind::Vendor);
        assert!((params.half_life_secs - engine.ewma_half_life_secs).abs() < f64::EPSILON);
        assert!((params.floor_1s - engine.vol_floor_1s).abs() < f64::EPSILON);
        assert!((params.cap_1s - engine.vol_cap_1s).abs() < f64::EPSILON);
        // Non-config fields stay at the model-crate defaults.
        let defaults = VolParams::default();
        assert_eq!(params.warmup_returns, defaults.warmup_returns);
        assert_eq!(params.max_gap_secs, defaults.max_gap_secs);
    }

    #[test]
    fn four_lanes_cover_both_assets_and_inputs() {
        let config = config::AppConfig::default();
        let lanes = build_lanes(&config).unwrap();
        let mut seen: Vec<(Asset, PriceSource, TickKind)> = lanes
            .iter()
            .map(|l| {
                (
                    l.estimator.asset(),
                    l.estimator.params().source,
                    l.estimator.params().kind,
                )
            })
            .collect();
        seen.sort();
        assert_eq!(
            seen,
            vec![
                (Asset::Btc, PriceSource::ChainlinkRtds, TickKind::Vendor),
                (Asset::Btc, PriceSource::BinanceDirect, TickKind::Mid),
                (Asset::Eth, PriceSource::ChainlinkRtds, TickKind::Vendor),
                (Asset::Eth, PriceSource::BinanceDirect, TickKind::Mid),
            ]
        );
    }

    #[test]
    fn lane_counters_classify_outcomes() {
        let config = config::AppConfig::default();
        let mut lane = Lane::new(
            Asset::Btc,
            vol_params(
                &config.engine.defaults,
                PriceSource::BinanceDirect,
                TickKind::Mid,
            ),
        )
        .unwrap();
        lane.observe(TickOutcome::Folded { gap_secs: 1 });
        lane.observe(TickOutcome::Reanchored { gap_secs: 99 });
        lane.observe(TickOutcome::Ignored(IgnoreReason::WrongStream));
        lane.observe(TickOutcome::Ignored(IgnoreReason::BadValue));
        lane.observe(TickOutcome::Absorbed);
        lane.observe(TickOutcome::Opened);
        assert_eq!((lane.folds, lane.reanchors, lane.drops), (1, 1, 1));
    }

    #[test]
    fn input_labels_are_stable() {
        assert_eq!(
            input_label(PriceSource::ChainlinkRtds, TickKind::Vendor),
            "chainlink:vendor"
        );
        assert_eq!(
            input_label(PriceSource::BinanceDirect, TickKind::Mid),
            "binance:mid"
        );
    }
}
