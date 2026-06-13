//! 24/7 window-rollover scheduler settings (CLAUDE.md §6).

use core_types::DurationMs;
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Scheduler timing policy: when to refresh discovery, when to announce the
/// `Closing` phase, how to back off on failure, and the §6 next-window
/// contract floor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Verify/refresh the next window this long before the current close.
    /// Must leave a retry budget on top of [`Self::next_window_lead_ms`].
    pub refresh_lead_ms: DurationMs,
    /// §6 contract: the next window must be known and announced at least
    /// this long before the current close. 60 s is a hard floor, not a
    /// tunable — validation rejects anything lower.
    pub next_window_lead_ms: DurationMs,
    /// Announce the `Closing` lifecycle phase this long before close (the
    /// engine applies its own finer §8 cutoffs within it).
    pub closing_lead_ms: DurationMs,
    /// Discovery retry backoff starts here and doubles per consecutive
    /// failure…
    pub retry_initial_backoff_ms: DurationMs,
    /// …capped at this.
    pub retry_max_backoff_ms: DurationMs,
    /// A closed window with no `market_resolved` warns loudly after this
    /// (only once a market-event source is attached).
    pub resolution_timeout_ms: DurationMs,
    /// Heartbeat: refresh a series even when everything is known if the last
    /// successful refresh is older than this (keeps the hourly series
    /// demonstrably alive and catches venue schedule surprises).
    pub max_refresh_interval_ms: DurationMs,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            refresh_lead_ms: DurationMs::from_millis(120_000),
            next_window_lead_ms: DurationMs::from_millis(60_000),
            closing_lead_ms: DurationMs::from_millis(30_000),
            retry_initial_backoff_ms: DurationMs::from_millis(1_000),
            retry_max_backoff_ms: DurationMs::from_millis(30_000),
            resolution_timeout_ms: DurationMs::from_millis(120_000),
            max_refresh_interval_ms: DurationMs::from_millis(600_000),
        }
    }
}

impl SchedulerConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            self.next_window_lead_ms.as_millis() >= 60_000,
            "scheduler.next_window_lead_ms",
            "must be >= 60000 — §6 requires the next window at least 60 s before close; \
             this is a floor, not a tunable",
        );
        v.require(
            self.refresh_lead_ms.as_millis() >= self.next_window_lead_ms.as_millis() + 30_000,
            "scheduler.refresh_lead_ms",
            "must exceed next_window_lead_ms by >= 30000 — the gap is the retry budget \
             before the §6 deadline",
        );
        v.require(
            self.refresh_lead_ms.as_millis() < 300_000,
            "scheduler.refresh_lead_ms",
            "must be < 300000 — the pre-close refresh must land inside the shortest (5m) window",
        );
        v.require(
            self.closing_lead_ms.as_millis() > 0 && self.closing_lead_ms.as_millis() <= 120_000,
            "scheduler.closing_lead_ms",
            "must be in (0, 120000]",
        );
        v.require(
            self.retry_initial_backoff_ms.as_millis() >= 100
                && self.retry_initial_backoff_ms <= self.retry_max_backoff_ms,
            "scheduler.retry_initial_backoff_ms",
            "must be >= 100 and <= retry_max_backoff_ms",
        );
        v.require(
            self.retry_max_backoff_ms.as_millis() <= 60_000,
            "scheduler.retry_max_backoff_ms",
            "must be <= 60000 — slower retries would burn the §6 retry budget",
        );
        v.require(
            self.resolution_timeout_ms.as_millis() >= 10_000,
            "scheduler.resolution_timeout_ms",
            "must be >= 10000 — resolution normally arrives within seconds of close",
        );
        v.require(
            self.max_refresh_interval_ms.as_millis() >= 60_000,
            "scheduler.max_refresh_interval_ms",
            "must be >= 60000 — a faster heartbeat would hammer Gamma for no benefit",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        SchedulerConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_sub_60s_contract_lead() {
        let cfg = SchedulerConfig {
            next_window_lead_ms: DurationMs::from_millis(59_999),
            ..SchedulerConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let keys: Vec<String> = v
            .into_result()
            .unwrap_err()
            .into_iter()
            .map(|x| x.key)
            .collect();
        assert!(keys.contains(&"scheduler.next_window_lead_ms".to_owned()));
    }

    #[test]
    fn rejects_refresh_lead_without_retry_budget_or_outside_window() {
        // Equal to the contract lead: no retry budget.
        let cfg = SchedulerConfig {
            refresh_lead_ms: DurationMs::from_millis(60_000),
            ..SchedulerConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_err());

        // Longer than the 5m window itself.
        let cfg = SchedulerConfig {
            refresh_lead_ms: DurationMs::from_millis(300_000),
            ..SchedulerConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_err());
    }

    #[test]
    fn rejects_bad_backoff_and_timeouts() {
        let cfg = SchedulerConfig {
            retry_initial_backoff_ms: DurationMs::from_millis(50),
            retry_max_backoff_ms: DurationMs::from_millis(90_000),
            resolution_timeout_ms: DurationMs::from_millis(1_000),
            max_refresh_interval_ms: DurationMs::from_millis(5_000),
            closing_lead_ms: DurationMs::from_millis(0),
            ..SchedulerConfig::default()
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
            "scheduler.retry_initial_backoff_ms",
            "scheduler.retry_max_backoff_ms",
            "scheduler.resolution_timeout_ms",
            "scheduler.max_refresh_interval_ms",
            "scheduler.closing_lead_ms",
        ] {
            assert!(keys.contains(&key.to_owned()), "missing violation: {key}");
        }

        // Initial > max is also rejected even when both are in range.
        let cfg = SchedulerConfig {
            retry_initial_backoff_ms: DurationMs::from_millis(40_000),
            retry_max_backoff_ms: DurationMs::from_millis(30_000),
            ..SchedulerConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_err());
    }
}
