//! Binance glue over the generic feed-util driver: the public [`run`] entry
//! point mapping [`BinanceParams`] onto generic driver parameters. The
//! connection lifecycle, watchdog, dead-socket teardown, and starvation
//! recycle all live in feed-util.
//!
//! Keepalive: none — Binance's *server* pings every 20 s and the transport
//! answers automatically (`WsFrame::Ping` surfaces as liveness for the
//! dead-socket watchdog). The venue also closes every connection at the
//! 24-hour mark; that lands on the ordinary reconnect path.

use std::time::Duration;

use core_types::{DurationMs, Event, TimestampMs};
use feed_util::{BackoffParams, DriverArgs, DriverParams, Transport};
use tokio::sync::{mpsc, watch};

use crate::protocol::BinanceProtocol;
use crate::sub::BinanceSub;
use crate::wire::combined_url;
use crate::{FeedError, FeedStatus};

/// A connection with no frames at all for this long is assumed half-open and
/// torn down: 3 × the 20-second server-ping cadence (the pings surface as
/// liveness, so a healthy-but-quiet socket can never trip this).
const DEAD_AFTER: Duration = Duration::from_secs(60);

/// Connection parameters (mapped from `feeds.*` config by the binary — this
/// crate depends only on core-types and feed-util).
#[derive(Debug, Clone, PartialEq)]
pub struct BinanceParams {
    /// Base WebSocket host, e.g. `wss://stream.binance.com:9443` — [`run`]
    /// appends the combined-stream path for its subscriptions.
    pub url: String,
    /// TCP+TLS+WS handshake deadline.
    pub connect_timeout: Duration,
    /// Reconnect backoff curve.
    pub backoff: BackoffParams,
    /// Staleness threshold for the bookTicker streams.
    pub stale_after: DurationMs,
    /// Staleness threshold for the trade streams (validated ≥ `stale_after`
    /// in config — quiet markets legitimately print nothing for seconds).
    pub trade_stale_after: DurationMs,
}

/// Everything [`run`] needs. Construct one per feed instance.
pub struct BinanceArgs<T, F> {
    /// Connection parameters.
    pub params: BinanceParams,
    /// The streams to subscribe (the bot passes [`BinanceSub::all`]); they
    /// ride the connect URL, fixed for the driver's life.
    pub subscriptions: Vec<BinanceSub>,
    /// The transport (real: [`feed_util::WsTransport`]; tests script a fake).
    pub transport: T,
    /// Wall-clock source; the only clock surface in the crate.
    pub now_fn: F,
    /// The internal bus. Sends are awaited — backpressure is explicit.
    pub bus_tx: mpsc::Sender<Event>,
    /// Optional status snapshots (CLI table, dashboard later).
    pub status_tx: Option<watch::Sender<FeedStatus>>,
    /// Flip to `true` for graceful shutdown.
    pub shutdown_rx: watch::Receiver<bool>,
    /// Backoff jitter seed; `None` seeds from the clock. Tests inject one
    /// for determinism.
    pub backoff_seed: Option<u64>,
}

/// Runs the feed until shutdown.
///
/// # Errors
/// [`FeedError::BusClosed`] when every bus receiver is gone. Connection
/// trouble is logged and retried forever, never returned.
pub async fn run<T, F>(args: BinanceArgs<T, F>) -> Result<(), FeedError>
where
    T: Transport,
    F: Fn() -> TimestampMs + Send,
{
    let BinanceArgs {
        params,
        subscriptions,
        transport,
        now_fn,
        bus_tx,
        status_tx,
        shutdown_rx,
        backoff_seed,
    } = args;
    let url = combined_url(&params.url, &subscriptions);
    let protocol = BinanceProtocol {
        subs: subscriptions,
        stale_after: params.stale_after,
        trade_stale_after: params.trade_stale_after,
    };
    feed_util::run(DriverArgs {
        params: DriverParams {
            url,
            connect_timeout: params.connect_timeout,
            backoff: params.backoff,
            dead_after: DEAD_AFTER,
            keepalive: None,
        },
        protocol,
        transport,
        now_fn,
        bus_tx,
        command_rx: None,
        status_tx,
        shutdown_rx,
        backoff_seed,
    })
    .await
}
