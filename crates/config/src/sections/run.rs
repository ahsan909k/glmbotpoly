//! Main run-mode (`bot run`) orchestration policy (CLAUDE.md §5).
//!
//! The supervision-tree settings the orchestrator reads at the binary boundary:
//! the resilient startup self-check gates (refuse to *trade* until clocks are
//! sane, discovery has current windows, and feeds are healthy — but keep the
//! process up and retrying), the per-task restart-with-backoff policy, and the
//! periodic risk-tick / resource-report cadences. Everything else the run mode
//! needs comes from the existing sections ([`crate::EngineConfig`],
//! [`crate::RiskConfig`], [`crate::FeedsConfig`], [`crate::ClockConfig`], …),
//! mapped into the engine/feed/timeutil parameter structs at the `bot` boundary
//! (those crates never depend on `config`, §4).

use core_types::DurationMs;
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// `bot run` supervision + startup self-check policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    /// Gate trading on clock sanity. When `true` (default), the bot will not
    /// arm until the §11 clock-skew monitor has run its startup burst without
    /// tripping `ClockSkew` (a skewed clock keeps the bot up but un-armed until
    /// the operator resyncs; a network that blocks NTP warns and proceeds).
    pub require_clock_check: bool,
    /// Gate trading on discovery. When `true` (default), the bot will not arm
    /// until the scheduler has announced a current (`Open`) window for every
    /// enabled series.
    pub require_discovery_check: bool,
    /// How long after boot to wait for the clock-skew monitor's startup burst
    /// to conclude before treating an un-tripped `ClockSkew` as "clock sane".
    /// Must comfortably exceed `clock.trip_after × the 2 s burst spacing` plus a
    /// measurement round.
    pub clock_check_grace_ms: DurationMs,
    /// If the bot has not armed within this long, log a loud "still not trading"
    /// warning listing what is missing — then keep waiting (resilient: the
    /// process never exits on a slow/failed self-check).
    pub readiness_timeout_ms: DurationMs,
    /// A crashed supervised task waits this long before its first restart…
    pub supervision_initial_backoff_ms: DurationMs,
    /// …doubling (by the multiplier) up to this cap.
    pub supervision_max_backoff_ms: DurationMs,
    /// Backoff growth multiplier between successive restarts of the same task.
    pub supervision_backoff_multiplier: f64,
    /// A task that ran healthy at least this long before dying has its restart
    /// backoff reset (so a one-off crash after a long healthy run does not
    /// inherit an escalated delay).
    pub stable_secs: i64,
    /// Cadence of the resource report (RSS + per-window book counts) and the
    /// settled-inventory prune.
    pub rss_report_secs: i64,
    /// Cadence of the risk manager's periodic tick (the §11 staleness / sanity /
    /// error-window recomputes and the quoter's debounced converge).
    pub risk_tick_ms: DurationMs,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            require_clock_check: true,
            require_discovery_check: true,
            clock_check_grace_ms: DurationMs::from_millis(15_000),
            readiness_timeout_ms: DurationMs::from_millis(120_000),
            supervision_initial_backoff_ms: DurationMs::from_millis(1_000),
            supervision_max_backoff_ms: DurationMs::from_millis(30_000),
            supervision_backoff_multiplier: 2.0,
            stable_secs: 60,
            rss_report_secs: 60,
            risk_tick_ms: DurationMs::from_millis(500),
        }
    }
}

impl RunConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            self.clock_check_grace_ms.as_millis() > 0,
            "run.clock_check_grace_ms",
            "must be > 0",
        );
        v.require(
            self.readiness_timeout_ms.as_millis() > 0,
            "run.readiness_timeout_ms",
            "must be > 0",
        );
        v.require(
            self.supervision_initial_backoff_ms.as_millis() > 0,
            "run.supervision_initial_backoff_ms",
            "must be > 0",
        );
        v.require(
            self.supervision_max_backoff_ms.as_millis()
                >= self.supervision_initial_backoff_ms.as_millis(),
            "run.supervision_max_backoff_ms",
            "must be >= run.supervision_initial_backoff_ms",
        );
        v.require(
            self.supervision_backoff_multiplier.is_finite()
                && self.supervision_backoff_multiplier >= 1.0,
            "run.supervision_backoff_multiplier",
            "must be finite and >= 1.0 (a multiplier below 1 would shrink the backoff)",
        );
        v.require(self.stable_secs > 0, "run.stable_secs", "must be > 0");
        v.require(
            self.rss_report_secs > 0,
            "run.rss_report_secs",
            "must be > 0",
        );
        v.require(
            (10..=5_000).contains(&self.risk_tick_ms.as_millis()),
            "run.risk_tick_ms",
            "must be in [10, 5000] — the §11 staleness timer needs a sub-second cadence",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        RunConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_inverted_backoff_and_bad_cadences() {
        let cfg = RunConfig {
            supervision_initial_backoff_ms: DurationMs::from_millis(30_000),
            supervision_max_backoff_ms: DurationMs::from_millis(1_000), // < initial
            supervision_backoff_multiplier: 0.5,                        // < 1
            risk_tick_ms: DurationMs::from_millis(9_000),               // out of range
            stable_secs: 0,
            ..RunConfig::default()
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
            "run.supervision_max_backoff_ms",
            "run.supervision_backoff_multiplier",
            "run.risk_tick_ms",
            "run.stable_secs",
        ] {
            assert!(keys.contains(&key.to_owned()), "missing violation: {key}");
        }
    }
}
