//! `bot latency` — the VPS-region latency benchmark (read-only; network
//! required): REST round-trips to the CLOB host, an NTP offset reading, the
//! CLOB server-time delta, and inter-message gaps on the CLOB market channel
//! + RTDS, written as a JSON report.
//!
//! The operator runs this from each candidate VPS region; the REST
//! percentiles calibrate `paper.placement_latency`/`paper.cancel_latency`
//! (§9) and the §8 quoting latency premium — manually, never auto-wired.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context;
use config::AppConfig;
use core_types::TimestampMs;
use timeutil::harness::{
    HarnessParams, LatencyReport, MarketWsProbe, RestTarget, RtdsWsProbe, run_harness,
};
use timeutil::{Clock, LatencyStats, SystemClock};

use crate::timecfg::{ntp_params, std_duration};

/// Builds the runtime and runs one full harness pass. Exit code is SUCCESS
/// only when every attempted probe section completed cleanly (the report is
/// written either way).
pub fn execute(
    config: &AppConfig,
    label: Option<&str>,
    out: Option<&Path>,
) -> anyhow::Result<ExitCode> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run(config, label, out))
}

async fn run(
    config: &AppConfig,
    label: Option<&str>,
    out: Option<&Path>,
) -> anyhow::Result<ExitCode> {
    let label = label.unwrap_or("unlabeled");
    let clock = SystemClock::new();
    let clob_base = config.feeds.clob_rest_url.trim_end_matches('/');

    let market_ws = discover_market_probe(config, &clock).await;
    let params = HarnessParams {
        label: label.to_owned(),
        rest_targets: vec![
            RestTarget {
                label: "clob/time".to_owned(),
                url: format!("{clob_base}/time"),
            },
            RestTarget {
                label: "clob/ok".to_owned(),
                url: format!("{clob_base}/ok"),
            },
        ],
        rest_samples: config.latency.rest_samples,
        rest_warmup: config.latency.rest_warmup,
        rest_spacing: std_duration(config.latency.rest_spacing_ms),
        http_timeout: std_duration(config.latency.http_timeout_ms),
        ws_connect_timeout: std_duration(config.feeds.ws_connect_timeout_ms),
        ws_duration: std_duration(config.latency.ws_duration_ms),
        market_ws,
        rtds_ws: Some(RtdsWsProbe {
            url: config.feeds.rtds_url.clone(),
            ping_interval: std_duration(config.feeds.rtds_ping_interval_ms),
        }),
        ntp: Some(ntp_params(&config.clock)),
        clob_time_url: Some(format!("{clob_base}/time")),
    };

    let report = run_harness(params, &clock).await;
    print_summary(&report);

    let path = write_report(&report, out, &config.latency.report_dir)?;
    println!("\nreport written to {}", path.display());

    Ok(if report.all_ok() {
        ExitCode::SUCCESS
    } else {
        println!("one or more probe sections FAILED — see the report");
        ExitCode::FAILURE
    })
}

/// Runs live discovery and picks ONE current window's Up/Down token pair for
/// the market-channel subscription (the first series with a live window —
/// BTC-5m in practice). Subscribing to all six series' tokens floods the
/// socket (~1500 msg/s observed) and the server resets the connection
/// mid-measurement; one pair still ticks plenty for gap stats. Discovery
/// failure (or zero live windows) skips the probe instead of aborting the
/// run — the gap stats are the only loss.
async fn discover_market_probe(config: &AppConfig, clock: &SystemClock) -> Option<MarketWsProbe> {
    let mut service =
        match discovery::DiscoveryService::from_config(&config.feeds, &config.discovery) {
            Ok(service) => service,
            Err(error) => {
                tracing::warn!(
                    target: "latency",
                    error = %error,
                    "discovery unavailable — skipping the market-channel probe"
                );
                return None;
            }
        };
    let snapshot = service.refresh_all(clock.wall()).await;
    for (series, result) in &snapshot.per_series {
        match result {
            Ok(windows) => {
                if let Some(market) = &windows.current {
                    tracing::info!(
                        target: "latency",
                        series = series.key(),
                        window = %market.event_slug,
                        "market-channel probe will subscribe to this window's token pair"
                    );
                    return Some(MarketWsProbe {
                        url: config.feeds.clob_market_ws_url.clone(),
                        asset_ids: vec![
                            market.tokens.up.as_str().to_owned(),
                            market.tokens.down.as_str().to_owned(),
                        ],
                    });
                }
            }
            Err(error) => tracing::warn!(
                target: "latency",
                series = series.key(),
                error = %error,
                "series discovery failed — trying the next series for the ws probe"
            ),
        }
    }
    tracing::warn!(
        target: "latency",
        "no live windows discovered — skipping the market-channel probe"
    );
    None
}

fn print_summary(report: &LatencyReport) {
    println!(
        "\nlatency report — label {:?} (schema v{})",
        report.label, report.schema_version
    );

    match &report.ntp {
        Some(ntp) => match ntp.offset_ms {
            Some(offset) => println!(
                "ntp offset      {offset:+} ms  ({} samples, {} failed, best rtt {} ms)",
                ntp.samples_used,
                ntp.queries_failed,
                ntp.min_round_trip_ms.unwrap_or(-1),
            ),
            None => println!(
                "ntp offset      UNAVAILABLE ({} queries failed) — check UDP/123 egress",
                ntp.queries_failed
            ),
        },
        None => println!("ntp offset      (not probed)"),
    }
    if let Some(ct) = &report.clob_time {
        match (ct.delta_ms, ct.rtt_ms) {
            (Some(delta), Some(rtt)) => println!(
                "clob /time      delta {delta:+} ms vs local (±1000 ms server resolution; rtt {rtt:.1} ms)"
            ),
            _ => println!(
                "clob /time      FAILED: {}",
                ct.error.as_deref().unwrap_or("unknown")
            ),
        }
    }

    println!(
        "\n{:<14} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "probe", "count", "errors", "min", "p50", "p95", "p99", "max"
    );
    for rest in &report.rest {
        match (&rest.stats, &rest.error) {
            (Some(stats), _) => print_stats_row(&rest.label, rest.errors, stats),
            (None, error) => println!(
                "{:<14} FAILED: {}",
                rest.label,
                error.as_deref().unwrap_or("all samples errored")
            ),
        }
    }
    for ws in &report.ws {
        let label = format!("{} gaps", ws.label);
        match (&ws.gaps, &ws.error) {
            (Some(gaps), _) => print_stats_row(&label, 0, gaps),
            (None, Some(error)) => println!("{label:<14} FAILED: {error}"),
            (None, None) => println!("{label:<14} (no messages)"),
        }
        println!(
            "{:<14}   connect {} ms, subscribe->first msg {} ms, {} msgs over {:.0} s{}{}",
            "",
            ws.connect_ms
                .map_or_else(|| "-".to_owned(), |v| format!("{v:.1}")),
            ws.subscribe_to_first_msg_ms
                .map_or_else(|| "-".to_owned(), |v| format!("{v:.1}")),
            ws.messages,
            ws.duration_ms as f64 / 1000.0,
            if ws.activity_dependent {
                " [activity-dependent: gaps reflect venue traffic, not the network]"
            } else {
                ""
            },
            ws.error
                .as_deref()
                .map_or_else(String::new, |e| format!(" [ended early: {e}]")),
        );
    }
}

fn print_stats_row(label: &str, errors: u32, stats: &LatencyStats) {
    println!(
        "{:<14} {:>6} {:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
        label,
        stats.count,
        errors,
        stats.min_ms,
        stats.p50_ms,
        stats.p95_ms,
        stats.p99_ms,
        stats.max_ms
    );
}

/// Writes the report as pretty JSON to `--out`, or to
/// `{report_dir}/latency-{label}-{YYYYMMDD-HHMMSS}.json`.
fn write_report(
    report: &LatencyReport,
    out: Option<&Path>,
    report_dir: &str,
) -> anyhow::Result<PathBuf> {
    let path = out.map_or_else(
        || {
            Path::new(report_dir).join(format!(
                "latency-{}-{}.json",
                sanitize(&report.label),
                file_stamp(report.started_at)
            ))
        },
        Path::to_path_buf,
    );
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating report directory {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(report).context("serializing the latency report")?;
    std::fs::write(&path, json)
        .with_context(|| format!("writing the report to {}", path.display()))?;
    Ok(path)
}

/// Filesystem-safe label: alphanumerics, `-` and `_` pass; everything else
/// becomes `-`.
fn sanitize(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// `YYYYMMDD-HHMMSS` (UTC) from a wall timestamp; raw millis as the fallback
/// outside the formattable range. Shared with `bot fair`'s journal naming.
pub(crate) fn file_stamp(ts: TimestampMs) -> String {
    discovery::map::format_rfc3339_secs(ts).map_or_else(
        |_| ts.as_millis().to_string(),
        |rfc| {
            rfc.chars()
                .filter_map(|c| match c {
                    '0'..='9' => Some(c),
                    'T' => Some('-'),
                    _ => None,
                })
                .collect()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_stamp_is_compact_utc() {
        // 2026-06-11T08:30:00Z
        assert_eq!(
            file_stamp(TimestampMs::from_millis(1_781_166_600_000)),
            "20260611-083000"
        );
    }

    #[test]
    fn labels_are_sanitized_for_filenames() {
        assert_eq!(sanitize("eu-west-2"), "eu-west-2");
        assert_eq!(sanitize("my vps #1!"), "my-vps--1-");
        assert_eq!(sanitize("dev_pc"), "dev_pc");
    }
}
