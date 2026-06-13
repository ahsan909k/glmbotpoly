//! The latency measurement harness behind `bot latency` (cargo feature
//! `harness`).
//!
//! Measures, per run: REST round-trip distributions against the CLOB host,
//! the CLOB server-time delta, an NTP offset reading, and WebSocket
//! connect/first-message/inter-message-gap numbers for the CLOB market
//! channel and RTDS. The operator runs this from candidate VPS regions; the
//! REST percentiles calibrate `paper.placement_latency`/`cancel_latency`
//! (§9) and the §8 quoting latency premium — manually, never auto-wired.
//!
//! The harness returns a [`LatencyReport`] *value*; the caller (the bot
//! binary) owns console rendering and the file write.

mod report;
mod rest;
mod ws;

use std::time::Duration;

pub use report::{
    ClobTimeReport, LatencyReport, NtpReport, REPORT_SCHEMA_VERSION, RestTargetReport,
    WsProbeReport,
};
pub use ws::{market_subscribe_message, rtds_crypto_subscribe_message};

use crate::clock::Clock;
use crate::offset::{NtpOffsetSource, NtpParams, OffsetSource};

/// One REST URL to probe.
#[derive(Debug, Clone)]
pub struct RestTarget {
    /// Short report label (e.g. `"clob/time"`).
    pub label: String,
    /// Full URL.
    pub url: String,
}

/// CLOB market-channel probe input.
#[derive(Debug, Clone)]
pub struct MarketWsProbe {
    /// Socket URL (`feeds.clob_market_ws_url`).
    pub url: String,
    /// Live token ids to subscribe to (from discovery).
    pub asset_ids: Vec<String>,
}

/// RTDS probe input.
#[derive(Debug, Clone)]
pub struct RtdsWsProbe {
    /// Socket URL (`feeds.rtds_url`).
    pub url: String,
    /// Keepalive interval (`feeds.rtds_ping_interval_ms`; docs require ≤5 s).
    pub ping_interval: Duration,
}

/// Everything one harness run needs (mapped from config by the binary).
#[derive(Debug, Clone)]
pub struct HarnessParams {
    /// Operator host/region label.
    pub label: String,
    /// REST targets to probe.
    pub rest_targets: Vec<RestTarget>,
    /// Timed samples per REST target.
    pub rest_samples: u32,
    /// Untimed warmup requests per REST target (TLS/pool setup).
    pub rest_warmup: u32,
    /// Pause between timed REST samples.
    pub rest_spacing: Duration,
    /// Per-request HTTP deadline.
    pub http_timeout: Duration,
    /// WS handshake (and first-message) deadline.
    pub ws_connect_timeout: Duration,
    /// Per-socket gap-measurement window.
    pub ws_duration: Duration,
    /// Market-channel probe (`None` = skipped, e.g. discovery failed).
    pub market_ws: Option<MarketWsProbe>,
    /// RTDS probe (`None` = skipped).
    pub rtds_ws: Option<RtdsWsProbe>,
    /// NTP probe (`None` = skipped).
    pub ntp: Option<NtpParams>,
    /// CLOB `GET /time` URL (`None` = skipped).
    pub clob_time_url: Option<String>,
}

/// Keepalive cadence for the CLOB market channel. The docs require a text
/// `"PING"` at least every 10 s (connection dropped without it); pinging at
/// half that keeps jitter from ever grazing the server's deadline.
const MARKET_WS_PING: Duration = Duration::from_secs(5);

/// Runs every configured probe sequentially (REST → CLOB time → NTP →
/// market WS → RTDS WS). Never fails as a whole: per-section errors land in
/// the report ([`LatencyReport::all_ok`] tells the caller).
pub async fn run_harness<C: Clock + Clone + 'static>(
    params: HarnessParams,
    clock: &C,
) -> LatencyReport {
    let started_at = clock.wall();
    let mut report = LatencyReport {
        schema_version: REPORT_SCHEMA_VERSION,
        label: params.label.clone(),
        started_at,
        finished_at: started_at,
        ntp: None,
        clob_time: None,
        rest: Vec::new(),
        ws: Vec::new(),
    };

    match reqwest::Client::builder()
        .timeout(params.http_timeout)
        .build()
    {
        Ok(http) => {
            for target in &params.rest_targets {
                tracing::info!(
                    target: "timeutil::harness",
                    url = %target.url,
                    samples = params.rest_samples,
                    "probing rest target"
                );
                report
                    .rest
                    .push(rest::probe_rest_target(&http, target, &params).await);
            }
            if let Some(url) = &params.clob_time_url {
                tracing::info!(target: "timeutil::harness", %url, "probing clob server time");
                report.clob_time = Some(rest::probe_clob_time(&http, url, clock).await);
            }
        }
        Err(error) => {
            // No HTTP client: every REST section reports the same fatal error.
            for target in &params.rest_targets {
                report.rest.push(RestTargetReport {
                    label: target.label.clone(),
                    url: target.url.clone(),
                    samples: 0,
                    errors: 0,
                    stats: None,
                    error: Some(format!("http client build failed: {error}")),
                });
            }
        }
    }

    if let Some(ntp_params) = params.ntp.clone() {
        tracing::info!(
            target: "timeutil::harness",
            servers = ?ntp_params.servers,
            "measuring ntp offset"
        );
        let mut source = NtpOffsetSource::new(ntp_params, clock.clone());
        let m = source.measure().await;
        report.ntp = Some(NtpReport {
            offset_ms: m.offset.map(|o| o.as_millis()),
            samples_used: m.samples_used,
            queries_failed: m.queries_failed,
            min_round_trip_ms: m.min_round_trip.map(|r| r.as_millis()),
        });
    }

    if let Some(market) = &params.market_ws {
        tracing::info!(
            target: "timeutil::harness",
            url = %market.url,
            tokens = market.asset_ids.len(),
            duration = ?params.ws_duration,
            "probing clob market websocket"
        );
        report.ws.push(
            ws::probe_ws(ws::WsProbeSpec {
                label: "clob-market".to_owned(),
                url: market.url.clone(),
                subscribe: market_subscribe_message(&market.asset_ids),
                ping: ("PING".to_owned(), MARKET_WS_PING),
                connect_timeout: params.ws_connect_timeout,
                duration: params.ws_duration,
                activity_dependent: true,
            })
            .await,
        );
    }

    if let Some(rtds) = &params.rtds_ws {
        tracing::info!(
            target: "timeutil::harness",
            url = %rtds.url,
            duration = ?params.ws_duration,
            "probing rtds websocket"
        );
        report.ws.push(
            ws::probe_ws(ws::WsProbeSpec {
                label: "rtds".to_owned(),
                url: rtds.url.clone(),
                subscribe: rtds_crypto_subscribe_message("btcusdt"),
                ping: ("PING".to_owned(), rtds.ping_interval),
                connect_timeout: params.ws_connect_timeout,
                duration: params.ws_duration,
                activity_dependent: false,
            })
            .await,
        );
    }

    report.finished_at = clock.wall();
    report
}
