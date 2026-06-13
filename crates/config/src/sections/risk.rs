//! Hard-wired risk limits (CLAUDE.md §11). Every order passes the risk
//! manager; these are its thresholds. Per-window loss caps live with the
//! engine parameters ([`crate::EngineParams::max_worst_case_loss_per_window`]).

use core_types::{Decimal, Dollars, DurationMs};
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Global risk-manager limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiskConfig {
    /// Feed staleness threshold that triggers cancel-all. §11 fixes the
    /// ceiling at 500 ms — configuring it tighter is allowed, looser is not.
    pub feed_staleness_ms: DurationMs,
    /// Daily stop-loss: once cumulative realized PnL for the day reaches
    /// this loss, all trading halts until manual reset.
    pub daily_stop_loss: Dollars,
    /// Global cap on open notional across all windows and series.
    pub max_open_notional: Dollars,
    /// Stand-down threshold on |model fair − book mid| (probability space).
    pub sanity_bound: f64,
    /// How long the sanity bound must be continuously exceeded before quotes
    /// are pulled.
    pub sanity_bound_duration_ms: DurationMs,
    /// API error count that trips the error-rate circuit breaker…
    pub error_breaker_max_errors: u32,
    /// …within this rolling window.
    pub error_breaker_window_ms: DurationMs,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            feed_staleness_ms: DurationMs::from_millis(500),
            daily_stop_loss: Dollars::new(Decimal::from(200)),
            max_open_notional: Dollars::new(Decimal::from(1_000)),
            sanity_bound: 0.10,
            sanity_bound_duration_ms: DurationMs::from_millis(3_000),
            error_breaker_max_errors: 10,
            error_breaker_window_ms: DurationMs::from_millis(60_000),
        }
    }
}

impl RiskConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        let staleness = self.feed_staleness_ms.as_millis();
        v.require(
            staleness > 0 && staleness <= 500,
            "risk.feed_staleness_ms",
            "must be in (0, 500] — §11 fixes the ceiling at 500 ms (tighter ok, looser never)",
        );
        v.require(
            self.daily_stop_loss.as_decimal() > Decimal::ZERO,
            "risk.daily_stop_loss",
            "must be > 0",
        );
        v.require(
            self.max_open_notional.as_decimal() > Decimal::ZERO,
            "risk.max_open_notional",
            "must be > 0",
        );
        v.require(
            self.sanity_bound.is_finite() && self.sanity_bound > 0.0 && self.sanity_bound < 1.0,
            "risk.sanity_bound",
            "must be a probability-space distance in (0, 1)",
        );
        v.require(
            self.sanity_bound_duration_ms.as_millis() > 0,
            "risk.sanity_bound_duration_ms",
            "must be > 0",
        );
        v.require(
            self.error_breaker_max_errors >= 1,
            "risk.error_breaker_max_errors",
            "must be >= 1",
        );
        v.require(
            self.error_breaker_window_ms.as_millis() > 0,
            "risk.error_breaker_window_ms",
            "must be > 0",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        RiskConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn staleness_outside_s11_bounds_rejected() {
        for bad in [0i64, -1, 501, 600] {
            let cfg = RiskConfig {
                feed_staleness_ms: DurationMs::from_millis(bad),
                ..RiskConfig::default()
            };
            let mut v = Violations::default();
            cfg.validate_into(&mut v);
            let violations = v.into_result().unwrap_err();
            assert!(
                violations.iter().any(|x| x.key == "risk.feed_staleness_ms"),
                "staleness {bad} should be rejected"
            );
        }
    }

    #[test]
    fn negative_money_limits_are_rejected() {
        let cfg = RiskConfig {
            daily_stop_loss: Dollars::new(Decimal::from(-5)),
            max_open_notional: Dollars::ZERO,
            ..RiskConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert_eq!(violations.len(), 2);
    }
}
