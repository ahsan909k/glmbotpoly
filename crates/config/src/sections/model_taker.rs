//! Model-taker settings (CLAUDE.md §8): the live PAPER taker driven by the
//! `shadow` model's per-window `p_up` stream. Off by default (anti-surprise); it
//! also requires `[shadow]` to be enabled (its prediction source — enforced in
//! `validate`). Replicates the tested walk-forward recipe (θ = 0.03, $10/window,
//! no edge gate/price cap/cooldown).

use std::collections::BTreeSet;

use core_types::{Asset, Decimal, Dollars, Series, WindowDuration};
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Which taker wins a same-window conflict (the §8 "fortress map" precedence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Driver {
    /// The fee-aware momentum taker.
    Momentum,
    /// The champion-model taker.
    Model,
}

/// The four M5/M15 series the shadow model covers (1h is out of distribution) —
/// the default model-taker allowlist.
fn default_series() -> BTreeSet<Series> {
    BTreeSet::from([
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M15,
        },
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::M5,
        },
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::M15,
        },
    ])
}

/// Configuration for the model taker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelTakerConfig {
    /// Master gate. Off by default — the model taker never fires unless enabled
    /// (and `[shadow]` is enabled to supply predictions).
    pub enable: bool,
    /// Emergency force-off: when `true` the model taker is disabled even if
    /// `enable` is set (a config kill switch; the runtime kill is the §11 global
    /// Manual breaker, which stands down all takers).
    pub kill_switch: bool,
    /// Fire when `|p_up − 0.5| ≥ theta` (the champion θ = 0.03).
    pub theta: f64,
    /// Per-window taker budget (the $10/window "max size per window" cap).
    pub budget_per_window: Dollars,
    /// Minimum finite-feature count (out of 24) required to act on a prediction.
    pub min_finite_count: u32,
    /// A prediction older than this (ms) stands the taker down silently
    /// (the >15 s "stale/missing predictions" rule); consumed by the bot drain.
    pub max_prediction_age_ms: i64,
    /// A cached book older than this (ms) is not fresh → no fire.
    pub max_book_staleness_ms: i64,
    /// The conflict-arbitration window (ms): a losing driver is suppressed on a
    /// window only while the winning driver fired that same window within it.
    pub arbitration_window_ms: i64,
    /// Per-series allowlist (default the four M5/M15 series).
    pub series: BTreeSet<Series>,
    /// Arbitration winner for BTC-conflicts (default momentum — the fortress map).
    pub precedence_btc: Driver,
    /// Arbitration winner for ETH-conflicts (default the model — the fortress map).
    pub precedence_eth: Driver,
}

impl Default for ModelTakerConfig {
    fn default() -> Self {
        Self {
            enable: false,
            kill_switch: false,
            theta: 0.03,
            budget_per_window: Dollars::new(Decimal::from(10)),
            min_finite_count: 24,
            max_prediction_age_ms: 15_000,
            max_book_staleness_ms: 2_000,
            arbitration_window_ms: 3_000,
            series: default_series(),
            precedence_btc: Driver::Momentum,
            precedence_eth: Driver::Model,
        }
    }
}

impl ModelTakerConfig {
    /// Whether the model taker should act on a prediction for `series` — the
    /// master gate, the kill switch, and the per-series allowlist together.
    #[must_use]
    pub fn is_active(&self, series: Series) -> bool {
        self.enable && !self.kill_switch && self.series.contains(&series)
    }

    /// The precedence winner for an asset (the fortress map).
    #[must_use]
    pub fn precedence_for(&self, asset: Asset) -> Driver {
        match asset {
            Asset::Btc => self.precedence_btc,
            Asset::Eth => self.precedence_eth,
        }
    }

    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            self.theta > 0.0 && self.theta < 0.5,
            "model_taker.theta",
            "must be in (0, 0.5)",
        );
        v.require(
            self.budget_per_window.as_decimal() >= Decimal::ZERO,
            "model_taker.budget_per_window",
            "must be ≥ 0",
        );
        v.require(
            self.min_finite_count <= 24,
            "model_taker.min_finite_count",
            "must be ≤ 24 (the feature count)",
        );
        v.require(
            self.max_prediction_age_ms > 0,
            "model_taker.max_prediction_age_ms",
            "must be > 0",
        );
        v.require(
            self.max_book_staleness_ms > 0,
            "model_taker.max_book_staleness_ms",
            "must be > 0",
        );
        v.require(
            self.arbitration_window_ms >= 0,
            "model_taker.arbitration_window_ms",
            "must be ≥ 0",
        );
        if self.enable && !self.kill_switch {
            v.require(
                !self.series.is_empty(),
                "model_taker.series",
                "must list at least one series when the model taker is enabled",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_off() {
        let cfg = ModelTakerConfig::default();
        assert!(!cfg.enable, "model taker defaults off");
        assert_eq!(cfg.series.len(), 4, "four M5/M15 series");
        assert_eq!(cfg.precedence_for(Asset::Btc), Driver::Momentum);
        assert_eq!(cfg.precedence_for(Asset::Eth), Driver::Model);
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn is_active_respects_gate_kill_and_allowlist() {
        let btc5 = Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        };
        let btc1h = Series {
            asset: Asset::Btc,
            duration: WindowDuration::H1,
        };
        let mut cfg = ModelTakerConfig {
            enable: true,
            ..ModelTakerConfig::default()
        };
        assert!(cfg.is_active(btc5), "enabled + allowlisted");
        assert!(!cfg.is_active(btc1h), "1h not in the allowlist");
        cfg.kill_switch = true;
        assert!(!cfg.is_active(btc5), "kill switch forces off");
    }

    #[test]
    fn rejects_bad_values() {
        let cfg = ModelTakerConfig {
            enable: true,
            theta: 0.6,
            min_finite_count: 30,
            max_prediction_age_ms: 0,
            max_book_staleness_ms: 0,
            series: BTreeSet::new(),
            ..ModelTakerConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"model_taker.theta"));
        assert!(keys.contains(&"model_taker.min_finite_count"));
        assert!(keys.contains(&"model_taker.max_prediction_age_ms"));
        assert!(keys.contains(&"model_taker.max_book_staleness_ms"));
        assert!(keys.contains(&"model_taker.series"));
    }
}
