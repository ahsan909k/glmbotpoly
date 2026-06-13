//! The latency report: everything one harness run learned, as serde types.
//!
//! Schema is versioned — the operator compares reports across VPS regions
//! and over time, so the shape is a contract.

use core_types::TimestampMs;
use serde::{Deserialize, Serialize};

use crate::stats::LatencyStats;

/// Current [`LatencyReport::schema_version`].
pub const REPORT_SCHEMA_VERSION: u32 = 1;

/// One full harness run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyReport {
    /// Report schema version (see [`REPORT_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Operator-supplied host/region label (`--label`).
    pub label: String,
    /// Wall time the run started.
    pub started_at: TimestampMs,
    /// Wall time the run finished.
    pub finished_at: TimestampMs,
    /// NTP offset result (`None` = probe not configured).
    pub ntp: Option<NtpReport>,
    /// CLOB server-time cross-check (`None` = probe not configured).
    pub clob_time: Option<ClobTimeReport>,
    /// REST round-trip distributions, one per target.
    pub rest: Vec<RestTargetReport>,
    /// WebSocket probe results, one per socket.
    pub ws: Vec<WsProbeReport>,
}

impl LatencyReport {
    /// True when every attempted section produced usable measurements.
    ///
    /// REST targets need stats and no fatal error. A WS section is usable
    /// iff it produced gap stats — an early server disconnect *after*
    /// collecting gaps still measured the network (the `error` field records
    /// it), while a silent socket (connected, no data) did not. NTP needs an
    /// offset; the CLOB time probe needs no error.
    #[must_use]
    pub fn all_ok(&self) -> bool {
        let rest_ok = self
            .rest
            .iter()
            .all(|r| r.error.is_none() && r.stats.is_some());
        let ws_ok = self.ws.iter().all(|w| w.gaps.is_some());
        let ntp_ok = self.ntp.as_ref().is_none_or(|n| n.offset_ms.is_some());
        let clob_ok = self.clob_time.as_ref().is_none_or(|c| c.error.is_none());
        rest_ok && ws_ok && ntp_ok && clob_ok
    }
}

/// Aggregated NTP measurement (same data as one skew-monitor round).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtpReport {
    /// Median offset in ms (`true_time − local_wall`); `None` = all failed.
    pub offset_ms: Option<i64>,
    /// Successful samples behind the median.
    pub samples_used: u32,
    /// Failed queries.
    pub queries_failed: u32,
    /// Best observed SNTP round-trip in ms.
    pub min_round_trip_ms: Option<i64>,
}

/// CLOB `GET /time` venue-clock cross-check. 1-second server resolution:
/// report-only, never an alarm input (Decisions Log 2026-06-12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClobTimeReport {
    /// Server unix time, seconds (raw body).
    pub server_unix_secs: Option<i64>,
    /// Local wall clock at receive, unix ms.
    pub local_unix_ms: i64,
    /// `server×1000 − local` in ms (±1000 ms quantization from the server's
    /// 1 s resolution).
    pub delta_ms: Option<i64>,
    /// Measured request round-trip in ms.
    pub rtt_ms: Option<f64>,
    /// Fatal probe error, if any.
    pub error: Option<String>,
}

/// REST round-trip distribution for one target URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestTargetReport {
    /// Short name (e.g. `"clob/time"`).
    pub label: String,
    /// Full URL probed.
    pub url: String,
    /// Timed samples attempted (excluding warmup).
    pub samples: u32,
    /// Samples that errored (excluded from `stats`).
    pub errors: u32,
    /// Round-trip distribution over the successful samples.
    pub stats: Option<LatencyStats>,
    /// Fatal section error (e.g. client build failed), if any.
    pub error: Option<String>,
}

/// One WebSocket probe: connect + subscribe + inter-message gap measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsProbeReport {
    /// Short name (e.g. `"clob-market"`, `"rtds"`).
    pub label: String,
    /// Socket URL.
    pub url: String,
    /// True when gap stats reflect market activity, not network performance
    /// (the CLOB market channel only speaks when the book changes).
    pub activity_dependent: bool,
    /// TCP+TLS+WS handshake duration in ms.
    pub connect_ms: Option<f64>,
    /// Subscribe-send → first data message, in ms.
    pub subscribe_to_first_msg_ms: Option<f64>,
    /// Data messages received during the measurement window.
    pub messages: u32,
    /// Actual measurement duration in ms.
    pub duration_ms: u64,
    /// Inter-message gap distribution.
    pub gaps: Option<LatencyStats>,
    /// Fatal probe error, if any.
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> LatencyReport {
        LatencyReport {
            schema_version: REPORT_SCHEMA_VERSION,
            label: "dev-pc".to_owned(),
            started_at: TimestampMs::from_millis(1_750_000_000_000),
            finished_at: TimestampMs::from_millis(1_750_000_180_000),
            ntp: Some(NtpReport {
                offset_ms: Some(-3),
                samples_used: 6,
                queries_failed: 0,
                min_round_trip_ms: Some(9),
            }),
            clob_time: Some(ClobTimeReport {
                server_unix_secs: Some(1_750_000_000),
                local_unix_ms: 1_750_000_000_123,
                delta_ms: Some(-123),
                rtt_ms: Some(95.2),
                error: None,
            }),
            rest: vec![RestTargetReport {
                label: "clob/time".to_owned(),
                url: "https://clob.polymarket.com/time".to_owned(),
                samples: 50,
                errors: 0,
                stats: crate::stats_from_ms(&mut [90.0, 95.0, 110.0]),
                error: None,
            }],
            ws: vec![WsProbeReport {
                label: "rtds".to_owned(),
                url: "wss://ws-live-data.polymarket.com".to_owned(),
                activity_dependent: false,
                connect_ms: Some(180.0),
                subscribe_to_first_msg_ms: Some(220.0),
                messages: 240,
                duration_ms: 60_000,
                gaps: crate::stats_from_ms(&mut [240.0, 260.0, 250.0]),
                error: None,
            }],
        }
    }

    #[test]
    fn report_serde_round_trips_and_pins_keys() {
        let report = sample_report();
        let json = serde_json::to_string_pretty(&report).expect("serializes");
        // Spot-pin the keys the operator's comparison tooling will rely on.
        assert!(json.contains("\"schema_version\": 1"));
        assert!(json.contains("\"p99_ms\""));
        assert!(json.contains("\"subscribe_to_first_msg_ms\""));
        assert!(json.contains("\"activity_dependent\""));
        let back: LatencyReport = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.label, report.label);
        assert_eq!(back.rest.len(), 1);
        assert!(back.all_ok());
    }

    #[test]
    fn all_ok_fails_on_section_errors() {
        let mut report = sample_report();
        assert!(report.all_ok());
        // A WS section with no gap stats failed (whatever the error says)...
        report.ws[0].gaps = None;
        report.ws[0].error = Some("connect timeout".to_owned());
        assert!(!report.all_ok());

        // ...but an early disconnect AFTER collecting gaps is still usable.
        let mut report = sample_report();
        report.ws[0].error = Some("connection reset mid-measurement".to_owned());
        assert!(report.ws[0].gaps.is_some());
        assert!(report.all_ok());

        let mut report = sample_report();
        report.rest[0].stats = None;
        assert!(!report.all_ok());

        let mut report = sample_report();
        report.ntp = Some(NtpReport {
            offset_ms: None,
            samples_used: 0,
            queries_failed: 6,
            min_round_trip_ms: None,
        });
        assert!(!report.all_ok());
    }
}
