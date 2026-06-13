//! Latency-harness settings (`bot latency`), consumed by `timeutil::harness`.
//!
//! Named `LatencyHarnessConfig` because [`crate::LatencyConfig`] is already
//! the paper venue's simulated-latency distribution.

use core_types::DurationMs;
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Measurement sizes and deadlines for the latency benchmark. The WS connect
/// timeout is deliberately *not* duplicated here — the harness reuses
/// `feeds.ws_connect_timeout_ms`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LatencyHarnessConfig {
    /// Timed samples per REST target.
    pub rest_samples: u32,
    /// Untimed warmup requests per target (TLS/connection-pool setup), so
    /// the timed samples measure the warm-pool RTT the order path sees.
    pub rest_warmup: u32,
    /// Pause between timed REST samples.
    pub rest_spacing_ms: DurationMs,
    /// Per-request HTTP deadline.
    pub http_timeout_ms: DurationMs,
    /// Inter-message gap measurement window per WebSocket.
    pub ws_duration_ms: DurationMs,
    /// Directory reports are written into (created if missing).
    pub report_dir: String,
}

impl Default for LatencyHarnessConfig {
    fn default() -> Self {
        Self {
            rest_samples: 50,
            rest_warmup: 3,
            rest_spacing_ms: DurationMs::from_millis(100),
            http_timeout_ms: DurationMs::from_millis(5_000),
            ws_duration_ms: DurationMs::from_millis(60_000),
            report_dir: "data/latency".to_owned(),
        }
    }
}

impl LatencyHarnessConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            (5..=1_000).contains(&self.rest_samples),
            "latency.rest_samples",
            "must be in [5, 1000] — fewer gives meaningless percentiles",
        );
        v.require(
            self.rest_warmup <= 20 && self.rest_warmup < self.rest_samples,
            "latency.rest_warmup",
            "must be <= 20 and < rest_samples",
        );
        v.require(
            (10..=5_000).contains(&self.rest_spacing_ms.as_millis()),
            "latency.rest_spacing_ms",
            "must be in [10, 5000] — faster would self-induce rate limiting",
        );
        v.require(
            self.http_timeout_ms.as_millis() > 0 && self.http_timeout_ms.as_millis() <= 30_000,
            "latency.http_timeout_ms",
            "must be in (0, 30000]",
        );
        v.require(
            (5_000..=600_000).contains(&self.ws_duration_ms.as_millis()),
            "latency.ws_duration_ms",
            "must be in [5000, 600000] — shorter gives too few gaps to summarize",
        );
        v.require(
            !self.report_dir.trim().is_empty(),
            "latency.report_dir",
            "must not be empty",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        LatencyHarnessConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_every_out_of_range_field() {
        let cfg = LatencyHarnessConfig {
            rest_samples: 4,
            rest_warmup: 21,
            rest_spacing_ms: DurationMs::from_millis(5),
            http_timeout_ms: DurationMs::from_millis(0),
            ws_duration_ms: DurationMs::from_millis(1_000),
            report_dir: "  ".to_owned(),
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let keys: Vec<String> = v
            .into_result()
            .unwrap_err()
            .into_iter()
            .map(|x| x.key)
            .collect();
        for key in [
            "latency.rest_samples",
            "latency.rest_warmup",
            "latency.rest_spacing_ms",
            "latency.http_timeout_ms",
            "latency.ws_duration_ms",
            "latency.report_dir",
        ] {
            assert!(keys.contains(&key.to_owned()), "missing violation: {key}");
        }
    }

    #[test]
    fn warmup_must_stay_below_samples() {
        let cfg = LatencyHarnessConfig {
            rest_samples: 10,
            rest_warmup: 10,
            ..LatencyHarnessConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_err());
    }
}
