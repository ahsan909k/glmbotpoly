//! Clock-discipline settings (CLAUDE.md §11): periodic NTP offset rounds and
//! the clock-skew breaker policy, consumed by `timeutil`.

use core_types::DurationMs;
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// NTP measurement cadence and the skew trip/clear policy.
///
/// Policy summary (Decisions Log 2026-06-12): `|offset| ≥ trip_bound_ms` for
/// `trip_after` consecutive rounds trips the `ClockSkew` breaker (the risk
/// manager reacts with cancel-all); clearing needs `clear_after` consecutive
/// rounds below the lower `clear_bound_ms`. Failed NTP rounds never trip —
/// they warn, escalating while the offset stays stale.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClockConfig {
    /// SNTP servers (`"host"` or `"host:port"`; port defaults to 123).
    pub ntp_servers: Vec<String>,
    /// Queries per server per measurement round.
    pub samples_per_server: u32,
    /// Per-query response deadline.
    pub query_timeout_ms: DurationMs,
    /// Pause between consecutive queries (rate hygiene toward public pools).
    pub query_spacing_ms: DurationMs,
    /// Steady-state cadence between measurement rounds.
    pub check_interval_ms: DurationMs,
    /// `|offset| ≥ this` counts toward a trip.
    pub trip_bound_ms: DurationMs,
    /// `|offset| < this` counts toward a clear; must be < `trip_bound_ms`
    /// (the gap is the anti-flap hysteresis dead zone).
    pub clear_bound_ms: DurationMs,
    /// Consecutive breaching rounds required to trip.
    pub trip_after: u32,
    /// Consecutive in-bound rounds required to clear.
    pub clear_after: u32,
    /// Age of the last successful round past which staleness warnings fire.
    pub stale_warn_ms: DurationMs,
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self {
            ntp_servers: vec![
                "time.cloudflare.com".to_owned(),
                "time.google.com".to_owned(),
                "pool.ntp.org".to_owned(),
            ],
            samples_per_server: 2,
            query_timeout_ms: DurationMs::from_millis(1_500),
            query_spacing_ms: DurationMs::from_millis(150),
            check_interval_ms: DurationMs::from_millis(60_000),
            trip_bound_ms: DurationMs::from_millis(250),
            clear_bound_ms: DurationMs::from_millis(125),
            trip_after: 2,
            clear_after: 3,
            stale_warn_ms: DurationMs::from_millis(300_000),
        }
    }
}

impl ClockConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            !self.ntp_servers.is_empty() && self.ntp_servers.len() <= 8,
            "clock.ntp_servers",
            "must list between 1 and 8 servers",
        );
        for (i, server) in self.ntp_servers.iter().enumerate() {
            v.require(
                !server.is_empty() && !server.chars().any(char::is_whitespace),
                format!("clock.ntp_servers[{i}]"),
                "must be a non-empty hostname (optionally host:port) without whitespace",
            );
        }
        v.require(
            (1..=8).contains(&self.samples_per_server),
            "clock.samples_per_server",
            "must be in [1, 8]",
        );
        v.require(
            self.query_timeout_ms.as_millis() > 0 && self.query_timeout_ms.as_millis() <= 10_000,
            "clock.query_timeout_ms",
            "must be in (0, 10000]",
        );
        v.require(
            (50..=2_000).contains(&self.query_spacing_ms.as_millis()),
            "clock.query_spacing_ms",
            "must be in [50, 2000] — public NTP pools rate-limit aggressive clients",
        );
        v.require(
            (10_000..=3_600_000).contains(&self.check_interval_ms.as_millis()),
            "clock.check_interval_ms",
            "must be in [10000, 3600000]",
        );
        v.require(
            (50..=5_000).contains(&self.trip_bound_ms.as_millis()),
            "clock.trip_bound_ms",
            "must be in [50, 5000] — tighter than NTP jitter would false-trip a cancel-all",
        );
        v.require(
            self.clear_bound_ms.as_millis() > 0
                && self.clear_bound_ms.as_millis() < self.trip_bound_ms.as_millis(),
            "clock.clear_bound_ms",
            "must be in (0, trip_bound_ms) — the gap is the anti-flap hysteresis",
        );
        v.require(
            (1..=10).contains(&self.trip_after),
            "clock.trip_after",
            "must be in [1, 10]",
        );
        v.require(
            (1..=10).contains(&self.clear_after),
            "clock.clear_after",
            "must be in [1, 10]",
        );
        v.require(
            self.stale_warn_ms.as_millis() >= self.check_interval_ms.as_millis(),
            "clock.stale_warn_ms",
            "must be >= check_interval_ms — anything lower warns between healthy rounds",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        ClockConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_inverted_hysteresis_and_bad_servers() {
        let cfg = ClockConfig {
            ntp_servers: vec!["time.cloudflare.com".to_owned(), "bad host".to_owned()],
            clear_bound_ms: DurationMs::from_millis(250), // == trip bound
            ..ClockConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let keys: Vec<String> = v
            .into_result()
            .unwrap_err()
            .into_iter()
            .map(|x| x.key)
            .collect();
        assert!(keys.contains(&"clock.clear_bound_ms".to_owned()));
        assert!(keys.contains(&"clock.ntp_servers[1]".to_owned()));
    }

    #[test]
    fn rejects_out_of_range_cadence_and_bounds() {
        let cfg = ClockConfig {
            ntp_servers: vec![],
            samples_per_server: 0,
            query_timeout_ms: DurationMs::from_millis(0),
            query_spacing_ms: DurationMs::from_millis(10),
            check_interval_ms: DurationMs::from_millis(1_000),
            trip_bound_ms: DurationMs::from_millis(10),
            trip_after: 0,
            clear_after: 11,
            stale_warn_ms: DurationMs::from_millis(500),
            ..ClockConfig::default()
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
            "clock.ntp_servers",
            "clock.samples_per_server",
            "clock.query_timeout_ms",
            "clock.query_spacing_ms",
            "clock.check_interval_ms",
            "clock.trip_bound_ms",
            "clock.trip_after",
            "clock.clear_after",
            "clock.stale_warn_ms",
        ] {
            assert!(keys.contains(&key.to_owned()), "missing violation: {key}");
        }
    }
}
