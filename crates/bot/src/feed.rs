//! `bot feed` — a live feed smoke run (read-only) for either price feed:
//! RTDS (`--source rtds`, the default: Binance-source + Chainlink-source
//! crypto prices) or the direct Binance WebSocket (`--source binance`:
//! BTCUSDT/ETHUSDT bookTicker midpoints + trade prints). Prints a per-stream
//! status table every second and every health transition as it happens. Kill
//! the network mid-run to watch the §11-style staleness events, the jittered
//! reconnect, and the per-stream recovery. Runs until ctrl-c.
//!
//! `--raw <file>` taps every in/out frame to a JSONL file
//! (`{"ts_local_ms":…,"dir":"in"|"out","text":…}`) — the fixture-capture
//! path for `crates/feed-rtds/tests/fixtures/` and
//! `crates/feed-binance/tests/fixtures/`.
//!
//! Note: direct-Binance `Mid` ticks carry `ts_exchange := ts_local` (the
//! bookTicker payload has no event time), so their `ex-age` column reads
//! ~0 s by construction — it is not a latency measurement.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{Event, FeedHealth, TimestampMs};
use feed_binance::{BinanceArgs, BinanceParams, BinanceSub};
use feed_rtds::{FeedSub, RtdsArgs, RtdsParams};
use feed_util::{
    BackoffParams, FeedError, FeedStatus, TapDir, TapFrame, TapTransport, Transport, WsTransport,
};
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::discover::fmt_ts;
use crate::timecfg::std_duration;

/// How often the side-by-side table prints.
const STATUS_PERIOD: Duration = Duration::from_secs(1);

/// Which feed `bot feed` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedSource {
    /// Polymarket RTDS (Binance + Chainlink topics) — the default.
    #[default]
    Rtds,
    /// Direct Binance WebSocket.
    Binance,
}

impl std::str::FromStr for FeedSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rtds" => Ok(Self::Rtds),
            "binance" => Ok(Self::Binance),
            other => Err(format!(
                "unknown feed source {other:?} (expected rtds or binance)"
            )),
        }
    }
}

impl std::fmt::Display for FeedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rtds => f.write_str("rtds"),
            Self::Binance => f.write_str("binance"),
        }
    }
}

/// Maps the `[feeds]` section to RTDS feed parameters (feed crates depend
/// only on core-types + feed-util, §4 — the translation lives at the binary
/// boundary, timecfg precedent).
pub(crate) fn rtds_params(cfg: &config::FeedsConfig) -> RtdsParams {
    RtdsParams {
        url: cfg.rtds_url.clone(),
        ping_interval: std_duration(cfg.rtds_ping_interval_ms),
        connect_timeout: std_duration(cfg.ws_connect_timeout_ms),
        backoff: backoff_params(cfg),
        stale_after: cfg.rtds_stale_after_ms,
    }
}

/// Maps the `[feeds]` section to direct-Binance feed parameters.
pub(crate) fn binance_params(cfg: &config::FeedsConfig) -> BinanceParams {
    BinanceParams {
        url: cfg.binance_ws_url.clone(),
        connect_timeout: std_duration(cfg.ws_connect_timeout_ms),
        backoff: backoff_params(cfg),
        stale_after: cfg.binance_stale_after_ms,
        trade_stale_after: cfg.binance_trade_stale_after_ms,
    }
}

/// Maps the `[feeds]` section to CLOB market-channel feed parameters
/// (`bot ladder`).
pub(crate) fn clob_params(cfg: &config::FeedsConfig) -> feed_clob::ClobParams {
    feed_clob::ClobParams {
        url: cfg.clob_market_ws_url.clone(),
        ping_interval: std_duration(cfg.clob_ping_interval_ms),
        connect_timeout: std_duration(cfg.ws_connect_timeout_ms),
        backoff: backoff_params(cfg),
        book_stale_after: cfg.clob_book_stale_after_ms,
        presubscribe_lead: cfg.clob_presubscribe_lead_ms,
        publish_interval: cfg.clob_book_publish_interval_ms,
        resolution_linger: cfg.clob_resolution_linger_ms,
    }
}

fn backoff_params(cfg: &config::FeedsConfig) -> BackoffParams {
    BackoffParams {
        initial: std_duration(cfg.reconnect_initial_backoff_ms),
        max: std_duration(cfg.reconnect_max_backoff_ms),
        multiplier: cfg.reconnect_backoff_multiplier,
    }
}

/// One smoke-runnable feed: its stream-key type and how to spawn its driver
/// over a concrete transport.
trait SmokeFeed {
    /// Status/stream key type (rows of the table).
    type Key: Copy + std::fmt::Display + Send + Sync + 'static;

    /// Endpoint description for the boot log.
    fn url(&self) -> &str;

    /// Spawns the driver. Consumes `self` (the two transport types — tapped
    /// and plain — monomorphize separately).
    fn spawn<T: Transport + 'static>(
        self,
        transport: T,
        bus_tx: mpsc::Sender<Event>,
        status_tx: watch::Sender<FeedStatus<Self::Key>>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> JoinHandle<Result<(), FeedError>>;
}

struct RtdsSmoke {
    params: RtdsParams,
}

impl SmokeFeed for RtdsSmoke {
    type Key = FeedSub;

    fn url(&self) -> &str {
        &self.params.url
    }

    fn spawn<T: Transport + 'static>(
        self,
        transport: T,
        bus_tx: mpsc::Sender<Event>,
        status_tx: watch::Sender<FeedStatus<FeedSub>>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> JoinHandle<Result<(), FeedError>> {
        tokio::spawn(feed_rtds::run(RtdsArgs {
            params: self.params,
            subscriptions: FeedSub::all(),
            transport,
            now_fn: wall_now,
            bus_tx,
            command_rx: None,
            status_tx: Some(status_tx),
            shutdown_rx,
            backoff_seed: None,
        }))
    }
}

struct BinanceSmoke {
    params: BinanceParams,
}

impl SmokeFeed for BinanceSmoke {
    type Key = BinanceSub;

    fn url(&self) -> &str {
        &self.params.url
    }

    fn spawn<T: Transport + 'static>(
        self,
        transport: T,
        bus_tx: mpsc::Sender<Event>,
        status_tx: watch::Sender<FeedStatus<BinanceSub>>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> JoinHandle<Result<(), FeedError>> {
        tokio::spawn(feed_binance::run(BinanceArgs {
            params: self.params,
            subscriptions: BinanceSub::all(),
            transport,
            now_fn: wall_now,
            bus_tx,
            status_tx: Some(status_tx),
            shutdown_rx,
            backoff_seed: None,
        }))
    }
}

/// Builds the runtime, spawns the selected feed driver against its live
/// endpoint, and consumes the bus until ctrl-c.
pub fn execute(config: &AppConfig, source: FeedSource, raw: Option<&Path>) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    match source {
        FeedSource::Rtds => runtime.block_on(run_smoke(
            raw,
            source,
            RtdsSmoke {
                params: rtds_params(&config.feeds),
            },
        )),
        FeedSource::Binance => runtime.block_on(run_smoke(
            raw,
            source,
            BinanceSmoke {
                params: binance_params(&config.feeds),
            },
        )),
    }
}

async fn run_smoke<F: SmokeFeed>(
    raw: Option<&Path>,
    source: FeedSource,
    feed: F,
) -> anyhow::Result<()> {
    tracing::info!(
        target: "feed",
        %source,
        url = %feed.url(),
        raw = raw.map(|p| p.display().to_string()),
        "feed smoke run starting (read-only; ctrl-c to stop)"
    );

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (status_tx, status_rx) = watch::channel(FeedStatus::<F::Key>::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // The tap writer runs on its own OS thread (blocking file IO never
    // touches the runtime); it exits when the transport — and with it the
    // tap sender — is dropped at driver shutdown.
    let (mut driver, writer) = match raw {
        Some(path) => {
            let (tap_tx, tap_rx) = mpsc::unbounded_channel::<TapFrame>();
            let writer = spawn_raw_writer(path.to_path_buf(), tap_rx)?;
            let task = feed.spawn(
                TapTransport::new(WsTransport, tap_tx),
                bus_tx,
                status_tx,
                shutdown_rx,
            );
            (task, Some(writer))
        }
        None => (
            feed.spawn(WsTransport, bus_tx, status_tx, shutdown_rx),
            None,
        ),
    };

    let mut interval = tokio::time::interval(STATUS_PERIOD);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut ticks: u64 = 0;
    let mut health_events: u64 = 0;
    let started = wall_now();

    let driver_result = loop {
        tokio::select! {
            joined = &mut driver => break joined,
            maybe = bus_rx.recv() => match maybe {
                Some(Event::PriceTick(tick)) => {
                    ticks += 1;
                    tracing::debug!(
                        target: "feed",
                        source = ?tick.source,
                        asset = %tick.asset,
                        kind = ?tick.kind,
                        value = %tick.value,
                        ts_exchange = tick.ts_exchange.as_millis(),
                        ts_local = tick.ts_local.as_millis(),
                        "tick"
                    );
                }
                Some(Event::FeedHealth(health)) => {
                    health_events += 1;
                    print_health(health);
                }
                Some(_) => {}
                None => break (&mut driver).await,
            },
            _ = interval.tick() => print_status(&status_rx.borrow(), wall_now()),
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "feed", "ctrl-c — shutting down");
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

    if let Some(writer) = writer {
        // The driver has exited, dropping the tap sender; the writer drains
        // what's left and returns.
        match writer.join() {
            Ok(Ok(lines)) => println!("raw capture: {lines} frames written"),
            Ok(Err(error)) => tracing::warn!(target: "feed", %error, "raw capture writer failed"),
            Err(_) => tracing::warn!(target: "feed", "raw capture writer panicked"),
        }
    }

    let ran_for = wall_now().signed_duration_since(started);
    println!(
        "\nrun summary: {ticks} price ticks, {health_events} health events over {}s",
        ran_for.as_millis() / 1000
    );
    driver_result
        .context("feed driver panicked")?
        .context("feed driver failed")?;
    Ok(())
}

/// Spawns the JSONL tap writer thread. Each line is one frame, stamped at
/// write time: `{"ts_local_ms":…,"dir":"in"|"out","text":…}`. Shared with
/// `bot ladder` (multi-connection: frames interleave in one file).
pub(crate) fn spawn_raw_writer(
    path: PathBuf,
    mut tap_rx: mpsc::UnboundedReceiver<TapFrame>,
) -> anyhow::Result<std::thread::JoinHandle<anyhow::Result<u64>>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating raw capture dir {}", parent.display()))?;
    }
    let file = std::fs::File::create(&path)
        .with_context(|| format!("creating raw capture file {}", path.display()))?;
    Ok(std::thread::spawn(move || {
        let mut out = std::io::BufWriter::new(file);
        let mut lines: u64 = 0;
        while let Some(frame) = tap_rx.blocking_recv() {
            let line = serde_json::json!({
                "ts_local_ms": wall_now().as_millis(),
                "dir": match frame.dir { TapDir::In => "in", TapDir::Out => "out" },
                "text": frame.text,
            });
            writeln!(out, "{line}").context("writing raw capture line")?;
            // Flush per line: the capture must survive ctrl-c and crashes.
            out.flush().context("flushing raw capture")?;
            lines += 1;
        }
        Ok(lines)
    }))
}

/// Spawns a JSONL writer thread for any serializable record type — one JSON
/// object per line, flushed per record for crash/ctrl-c safety (the
/// [`spawn_raw_writer`] pattern, generalized). Used by `bot fair` for
/// calibration records. The thread exits when the sender is dropped.
pub(crate) fn spawn_jsonl_writer<T: serde::Serialize + Send + 'static>(
    path: PathBuf,
    mut rx: mpsc::UnboundedReceiver<T>,
) -> anyhow::Result<std::thread::JoinHandle<anyhow::Result<u64>>> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating journal dir {}", parent.display()))?;
    }
    let file = std::fs::File::create(&path)
        .with_context(|| format!("creating journal file {}", path.display()))?;
    Ok(std::thread::spawn(move || {
        let mut out = std::io::BufWriter::new(file);
        let mut lines: u64 = 0;
        while let Some(record) = rx.blocking_recv() {
            let line = serde_json::to_string(&record).context("serializing journal record")?;
            writeln!(out, "{line}").context("writing journal line")?;
            out.flush().context("flushing journal")?;
            lines += 1;
        }
        Ok(lines)
    }))
}

/// One health transition, printed as it happens (RISK-event precedent in
/// `bot schedule`).
pub(crate) fn print_health(health: FeedHealth) {
    match health {
        FeedHealth::Stale {
            source,
            asset,
            kind,
            age,
        } => {
            println!(
                "FEED HEALTH: {source:?} {asset} {kind:?} STALE (no data for {:.1}s)",
                age.as_millis() as f64 / 1000.0
            );
        }
        FeedHealth::Recovered {
            source,
            asset,
            kind,
            gap,
        } => {
            println!(
                "FEED HEALTH: {source:?} {asset} {kind:?} recovered (gap {:.1}s)",
                gap.as_millis() as f64 / 1000.0
            );
        }
    }
}

/// The side-by-side table: one row per stream with ages computed against the
/// local clock at print time.
fn print_status<K: Copy + std::fmt::Display>(status: &FeedStatus<K>, now: TimestampMs) {
    println!(
        "\nfeed at {}  [{} | reconnects {}]",
        fmt_ts(status.at),
        status.connection,
        status.reconnects
    );
    println!(
        "{:<22} {:>14} {:>8} {:>8} {:>7}  state",
        "stream", "last value", "ex-age", "rx-age", "ticks"
    );
    for key in &status.keys {
        println!(
            "{:<22} {:>14} {:>8} {:>8} {:>7}  {}",
            key.sub.to_string(),
            key.last_value
                .map_or_else(|| "-".to_owned(), |v| v.to_string()),
            fmt_age(key.ts_exchange, now),
            fmt_age(key.ts_local, now),
            key.ticks,
            if key.stale { "STALE" } else { "live" },
        );
    }
}

/// Sub-second-resolution age, or `-` before the first tick.
pub(crate) fn fmt_age(since: Option<TimestampMs>, now: TimestampMs) -> String {
    since.map_or_else(
        || "-".to_owned(),
        |ts| {
            let ms = now.signed_duration_since(ts).as_millis();
            format!("{:.1}s", ms as f64 / 1000.0)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feeds_section_maps_one_to_one_for_rtds() {
        let cfg = config::FeedsConfig::default();
        let params = rtds_params(&cfg);
        assert_eq!(params.url, cfg.rtds_url);
        assert_eq!(params.ping_interval, Duration::from_millis(5_000));
        assert_eq!(params.connect_timeout, Duration::from_millis(5_000));
        assert_eq!(params.backoff.initial, Duration::from_millis(250));
        assert_eq!(params.backoff.max, Duration::from_millis(10_000));
        assert!((params.backoff.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(params.stale_after, cfg.rtds_stale_after_ms);
    }

    #[test]
    fn feeds_section_maps_one_to_one_for_binance() {
        let cfg = config::FeedsConfig::default();
        let params = binance_params(&cfg);
        assert_eq!(params.url, cfg.binance_ws_url);
        assert_eq!(
            params.url, "wss://stream.binance.com:9443",
            "base URL, no path"
        );
        assert_eq!(params.connect_timeout, Duration::from_millis(5_000));
        assert_eq!(params.backoff.initial, Duration::from_millis(250));
        assert_eq!(params.backoff.max, Duration::from_millis(10_000));
        assert!((params.backoff.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(params.stale_after, cfg.binance_stale_after_ms);
        assert_eq!(params.trade_stale_after, cfg.binance_trade_stale_after_ms);
    }

    #[test]
    fn feeds_section_maps_one_to_one_for_clob() {
        let cfg = config::FeedsConfig::default();
        let params = clob_params(&cfg);
        assert_eq!(params.url, cfg.clob_market_ws_url);
        assert_eq!(params.ping_interval, Duration::from_millis(5_000));
        assert_eq!(params.connect_timeout, Duration::from_millis(5_000));
        assert_eq!(params.backoff.initial, Duration::from_millis(250));
        assert_eq!(params.backoff.max, Duration::from_millis(10_000));
        assert!((params.backoff.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(params.book_stale_after, cfg.clob_book_stale_after_ms);
        assert_eq!(params.presubscribe_lead, cfg.clob_presubscribe_lead_ms);
        assert_eq!(params.publish_interval, cfg.clob_book_publish_interval_ms);
        assert_eq!(params.resolution_linger, cfg.clob_resolution_linger_ms);
    }

    #[test]
    fn feed_source_parses_and_rejects() {
        assert_eq!("rtds".parse::<FeedSource>(), Ok(FeedSource::Rtds));
        assert_eq!("binance".parse::<FeedSource>(), Ok(FeedSource::Binance));
        assert!("coinbase".parse::<FeedSource>().is_err());
        assert!("RTDS".parse::<FeedSource>().is_err(), "case-sensitive");
        assert_eq!(FeedSource::default(), FeedSource::Rtds);
    }

    #[test]
    fn ages_format_for_display() {
        let now = TimestampMs::from_millis(10_000);
        assert_eq!(fmt_age(None, now), "-");
        assert_eq!(fmt_age(Some(TimestampMs::from_millis(9_500)), now), "0.5s");
        assert_eq!(fmt_age(Some(TimestampMs::from_millis(0)), now), "10.0s");
        // Ahead-of-local exchange clocks render negative, not panic.
        assert_eq!(
            fmt_age(Some(TimestampMs::from_millis(10_200)), now),
            "-0.2s"
        );
    }
}
