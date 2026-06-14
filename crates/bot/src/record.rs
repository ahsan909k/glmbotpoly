//! `bot record` — capture the full live bus to disk for replay.
//!
//! Wires the scheduler + all three feeds (feed-clob books/prints/lifecycle,
//! feed-rtds + feed-binance price ticks) onto one bus, exactly like the smoke
//! runs but with **no model** — every `core_types::Event` that crosses the bus
//! is handed to a [`journal::Recorder`], which projects it to a
//! [`journal::JournalRecord`] and writes gzip-compressed, size/age-rotated
//! JSONL segments under `--out-dir` (default `data/journal/`). Runs until
//! ctrl-c; days of capture roll across segments by construction.
//!
//! Records **all enabled series** by default (`config.engine`), or just one
//! with `--series`. The recorder never back-pressures the bus: if the disk
//! stalls and its bounded channel fills, events are dropped + counted (a nonzero
//! drop count in the summary means the capture is incomplete).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{Event, MarketInfo, Series, WindowLifecycle};
use discovery::DiscoveryService;
use feed_binance::{BinanceArgs, BinanceSub};
use feed_clob::ClobArgs;
use feed_rtds::{FeedSub, RtdsArgs};
use feed_util::{FeedError, WsTransport};
use journal::{Recorder, RecorderParams};
use scheduler::{SchedulerArgs, Timing};
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::feed::{binance_params, clob_params, rtds_params};

/// How often the status line prints.
const STATUS_PERIOD: Duration = Duration::from_secs(5);

type FeedResult = Result<Result<(), FeedError>, tokio::task::JoinError>;
type SchedResult = Result<Result<(), scheduler::SchedulerError>, tokio::task::JoinError>;

/// Per-category counters for the status line + summary.
#[derive(Default)]
struct Counts {
    total: u64,
    ticks: u64,
    books: u64,
    tops: u64,
    trades: u64,
    windows: u64,
    health: u64,
    other: u64,
}

impl Counts {
    fn tally(&mut self, event: &Event) {
        self.total += 1;
        match event {
            Event::PriceTick(_) => self.ticks += 1,
            Event::Book(_) => self.books += 1,
            Event::TopOfBook { .. } => self.tops += 1,
            Event::LastTrade { .. } => self.trades += 1,
            Event::Window { .. } => self.windows += 1,
            Event::FeedHealth(_) | Event::BookHealth(_) => self.health += 1,
            _ => self.other += 1,
        }
    }
}

/// Builds the runtime and runs the recorder until ctrl-c. `series` records just
/// that series; `None` records every enabled series.
pub fn execute(config: &AppConfig, series: Option<Series>, out_dir: &Path) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run_record(config, series, out_dir.to_path_buf()))
}

async fn run_record(
    config: &AppConfig,
    series: Option<Series>,
    out_dir: PathBuf,
) -> anyhow::Result<()> {
    let series_list: Vec<Series> = match series {
        Some(s) => vec![s],
        None => config.engine.enabled_series(),
    };
    if series_list.is_empty() {
        anyhow::bail!("no series to record (none given and none enabled in config)");
    }

    let service = DiscoveryService::from_config(&config.feeds, &config.discovery)
        .context("building discovery service")?;

    let recorder = Recorder::spawn(
        RecorderParams {
            out_dir: out_dir.clone(),
            ..RecorderParams::default()
        },
        wall_now,
    )
    .with_context(|| format!("starting the journal recorder in {}", out_dir.display()))?;

    let keys: Vec<&'static str> = series_list.iter().map(|s| s.key()).collect();
    tracing::info!(
        target: "record",
        series = ?keys,
        out_dir = %out_dir.display(),
        "journal recorder starting (read-only capture; ctrl-c to stop)"
    );

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (window_tx, window_rx) = mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(64);
    let (market_tx, market_rx) = mpsc::channel(64);
    // Each driver subscribes its own receiver; this one is the keep-alive root.
    let (shutdown_tx, _shutdown_rx) = watch::channel(false);

    let mut sched_task: JoinHandle<Result<(), scheduler::SchedulerError>> =
        tokio::spawn(scheduler::run(SchedulerArgs {
            timing: Timing::from_config(&config.scheduler),
            series: series_list,
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
    // The loop's keep-alive sender (the drivers hold their own clones); dropped
    // in the drain phase so `bus_rx.recv()` can reach `None`.
    let keepalive_tx = bus_tx;

    let mut counts = Counts::default();
    let mut status = tokio::time::interval(STATUS_PERIOD);
    status.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let started = wall_now();

    let mut sched_result: Option<SchedResult> = None;
    let mut feed_results: [Option<FeedResult>; 3] = [None, None, None];

    // Main phase: capture until ctrl-c, or until any driver exits on its own.
    loop {
        tokio::select! {
            joined = &mut sched_task, if sched_result.is_none() => {
                sched_result = Some(joined);
                tracing::warn!(target: "record", "scheduler exited — shutting down");
                break;
            }
            joined = &mut clob_task, if feed_results[0].is_none() => {
                feed_results[0] = Some(joined);
                tracing::warn!(target: "record", "clob feed exited — shutting down");
                break;
            }
            joined = &mut rtds_task, if feed_results[1].is_none() => {
                feed_results[1] = Some(joined);
                tracing::warn!(target: "record", "rtds feed exited — shutting down");
                break;
            }
            joined = &mut binance_task, if feed_results[2].is_none() => {
                feed_results[2] = Some(joined);
                tracing::warn!(target: "record", "binance feed exited — shutting down");
                break;
            }
            maybe = bus_rx.recv() => match maybe {
                Some(event) => {
                    counts.tally(&event);
                    recorder.record(&event);
                    // Forward window announcements to feed-clob (idempotent).
                    if let Event::Window { market, lifecycle } = &event
                        && let Err(error) = window_tx.try_send((Arc::clone(market), *lifecycle))
                    {
                        tracing::warn!(target: "record", %error, "window forward dropped");
                    }
                }
                None => unreachable!("the loop holds a bus sender (keepalive_tx)"),
            },
            _ = status.tick() => print_status(&counts, recorder.dropped(), started),
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "record", "ctrl-c — shutting down");
                break;
            }
        }
    }

    // Drain phase: signal shutdown, keep recording + draining the bus so no
    // driver deadlocks on a full channel while exiting.
    let _ = shutdown_tx.send(true);
    drop(keepalive_tx);
    let mut bus_open = true;
    while sched_result.is_none() || feed_results.iter().any(Option::is_none) {
        tokio::select! {
            joined = &mut sched_task, if sched_result.is_none() => sched_result = Some(joined),
            joined = &mut clob_task, if feed_results[0].is_none() => feed_results[0] = Some(joined),
            joined = &mut rtds_task, if feed_results[1].is_none() => feed_results[1] = Some(joined),
            joined = &mut binance_task, if feed_results[2].is_none() => {
                feed_results[2] = Some(joined);
            }
            maybe = bus_rx.recv(), if bus_open => match maybe {
                Some(event) => {
                    counts.tally(&event);
                    recorder.record(&event);
                }
                None => bus_open = false,
            },
        }
    }

    // Stop the recorder: drain its backlog, flush, finalize the gzip trailers.
    let stats = recorder.finish().context("finalizing the journal")?;
    let ran_for = wall_now().signed_duration_since(started);
    println!(
        "\nrecord summary: {} events captured ({} ticks, {} books, {} tops, {} trades, \
         {} windows, {} health, {} other) → {} records in {} segments, {} dropped, over {}s",
        counts.total,
        counts.ticks,
        counts.books,
        counts.tops,
        counts.trades,
        counts.windows,
        counts.health,
        counts.other,
        stats.records,
        stats.segments,
        stats.dropped,
        ran_for.as_millis() / 1000
    );
    if stats.dropped > 0 {
        tracing::warn!(
            target: "record",
            dropped = stats.dropped,
            "capture is incomplete — the disk could not keep up with the bus"
        );
    }

    // Surface a driver failure (a clean ctrl-c leaves these as Ok).
    join_results(sched_result, feed_results)
}

/// The status line: cumulative counts + drop count + elapsed.
fn print_status(counts: &Counts, dropped: u64, started: core_types::TimestampMs) {
    let secs = wall_now().signed_duration_since(started).as_millis() / 1000;
    println!(
        "[{secs:>5}s] captured {} events  ({} ticks, {} books, {} trades, {} windows)  \
         dropped {}",
        counts.total, counts.ticks, counts.books, counts.trades, counts.windows, dropped
    );
}

/// Propagates the first driver failure, if any (a panic or a returned error).
fn join_results(sched: Option<SchedResult>, feeds: [Option<FeedResult>; 3]) -> anyhow::Result<()> {
    if let Some(joined) = sched {
        joined
            .context("scheduler task panicked")?
            .context("scheduler failed")?;
    }
    for (idx, feed) in feeds.into_iter().enumerate() {
        let name = ["clob", "rtds", "binance"][idx];
        if let Some(joined) = feed {
            joined
                .with_context(|| format!("{name} feed task panicked"))?
                .with_context(|| format!("{name} feed failed"))?;
        }
    }
    Ok(())
}
