//! Crate-local analytics tuning.
//!
//! All knobs live here with committed [`Default`]s (the `engine`/`model`
//! precedent). The `config::* → AnalyticsParams` boundary map at the `bot`
//! binary is deferred to the wiring task, exactly as every prior engine/model
//! task deferred its own mapping — so analytics carries no `config` dependency
//! and is fully testable with the defaults. [`AnalyticsParams::validate`] is
//! provided for when that mapping lands.

use core_types::{Decimal, Dollars};
use rust_decimal::dec;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Why an [`AnalyticsParams`] was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParamsError {
    /// A count that must be positive was zero.
    #[error("{0} must be > 0")]
    NotPositive(&'static str),
    /// A value that must be non-negative was negative.
    #[error("{0} must be >= 0")]
    Negative(&'static str),
    /// A fee/rebate-rate default fell outside `[0, 1]`.
    #[error("{field} must be in [0, 1], got {value}")]
    RateOutOfRange {
        /// Which field.
        field: &'static str,
        /// The offending value.
        value: Decimal,
    },
    /// A non-finite threshold.
    #[error("{0} must be finite")]
    NotFinite(&'static str),
}

/// Tuning for the per-series adverse-selection alarm (CLAUDE.md §8/§10): a
/// persistently negative mean 5-second markout on passive fills means our
/// resting quotes are being picked off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdverseSelectionParams {
    /// Passive-fill 5s markouts observed before the alarm leaves
    /// `InsufficientSample` — the "persistently" floor that suppresses
    /// small-sample noise.
    pub min_sample: u32,
    /// Rolling-mean 5s markout (probability units) strictly below which the
    /// series is in `Alarm`. `0.0` is the principled boundary (on average we
    /// lose value after a passive fill); journal-calibratable.
    pub negative_threshold: f64,
    /// How many of the most recent 5s markouts the rolling mean averages over.
    pub window: u32,
}

impl Default for AdverseSelectionParams {
    fn default() -> Self {
        Self {
            min_sample: 50,
            negative_threshold: 0.0,
            window: 200,
        }
    }
}

/// Top-level analytics tuning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnalyticsParams {
    /// Adverse-selection alarm tuning.
    pub adverse: AdverseSelectionParams,
    /// Windows a series must have traded before its comparison row drops the
    /// minimum-sample warning (§10 "minimum-sample warning").
    pub min_sample_windows: u32,
    /// Hard cap on a window's retained model-fair snapshot ring — a backstop
    /// sized well above any realistic window's snapshot count (the ring is
    /// dropped wholesale at resolution, so it never persists). On overflow the
    /// oldest snapshot is dropped and LOCF degrades gracefully.
    pub snapshot_ring_max: usize,
    /// Fee rate used for the maker-rebate base when a fill's window was never
    /// seen as a `Window` event (cold replay from mid-journal). The §7
    /// config-default 0.07; the conservation identity is unaffected (it uses
    /// the settlement's own `fees_paid`, not a recomputed fee).
    pub default_fee_rate: Decimal,
    /// Rebate share used when the window's `FeeParams` were never seen (cold
    /// replay). The §7 crypto-category 0.20.
    pub default_rebate_rate: Decimal,
    /// Shared per-window taker notional budget (the §8 `taker_budget_per_window`,
    /// default `$10`). The series-comparison row's taker-budget-usage column
    /// divides mean per-window taker notional by this. Taker spend is **not** on
    /// the bus, so analytics reconstructs the notional from Taker fills; this
    /// budget is the denominator. The `config::EngineParams → AnalyticsParams`
    /// boundary map at the `bot` binary is deferred (the crate-default matches
    /// config). `0` disables the column (yields `None`).
    pub taker_budget_per_window: Dollars,
}

impl Default for AnalyticsParams {
    fn default() -> Self {
        Self {
            adverse: AdverseSelectionParams::default(),
            min_sample_windows: 30,
            snapshot_ring_max: 65_536,
            default_fee_rate: dec!(0.07),
            default_rebate_rate: dec!(0.20),
            taker_budget_per_window: Dollars::new(dec!(10)),
        }
    }
}

impl AnalyticsParams {
    /// Checks the invariants the boundary map will rely on.
    ///
    /// # Errors
    /// [`ParamsError`] when a count is zero, a rate default leaves `[0, 1]`, or
    /// a threshold is non-finite.
    pub fn validate(&self) -> Result<(), ParamsError> {
        if self.adverse.min_sample == 0 {
            return Err(ParamsError::NotPositive("adverse.min_sample"));
        }
        if self.adverse.window == 0 {
            return Err(ParamsError::NotPositive("adverse.window"));
        }
        if !self.adverse.negative_threshold.is_finite() {
            return Err(ParamsError::NotFinite("adverse.negative_threshold"));
        }
        if self.min_sample_windows == 0 {
            return Err(ParamsError::NotPositive("min_sample_windows"));
        }
        if self.snapshot_ring_max == 0 {
            return Err(ParamsError::NotPositive("snapshot_ring_max"));
        }
        // Zero is allowed (= "no budget" → the usage column reports `None`);
        // a negative budget is nonsense.
        if self.taker_budget_per_window.as_decimal() < Decimal::ZERO {
            return Err(ParamsError::Negative("taker_budget_per_window"));
        }
        for (field, value) in [
            ("default_fee_rate", self.default_fee_rate),
            ("default_rebate_rate", self.default_rebate_rate),
        ] {
            if value < Decimal::ZERO || value > Decimal::ONE {
                return Err(ParamsError::RateOutOfRange { field, value });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_consistent() {
        let p = AnalyticsParams::default();
        assert!(p.validate().is_ok());
        // §7 config defaults.
        assert_eq!(p.default_fee_rate, dec!(0.07));
        assert_eq!(p.default_rebate_rate, dec!(0.20));
        // §8 shared per-window taker budget default.
        assert_eq!(p.taker_budget_per_window, Dollars::new(dec!(10)));
        // The rolling window should be able to reach the sample floor.
        assert!(p.adverse.window >= p.adverse.min_sample);
    }

    #[test]
    fn budget_zero_allowed_negative_rejected() {
        // Zero = "no budget" → the usage column will report `None`; still valid.
        let zero = AnalyticsParams {
            taker_budget_per_window: Dollars::ZERO,
            ..AnalyticsParams::default()
        };
        assert!(zero.validate().is_ok());

        let negative = AnalyticsParams {
            taker_budget_per_window: Dollars::new(dec!(-1)),
            ..AnalyticsParams::default()
        };
        assert!(matches!(
            negative.validate(),
            Err(ParamsError::Negative("taker_budget_per_window"))
        ));
    }

    #[test]
    fn validate_rejects_bad_values() {
        let bad_sample = AnalyticsParams {
            adverse: AdverseSelectionParams {
                min_sample: 0,
                ..AdverseSelectionParams::default()
            },
            ..AnalyticsParams::default()
        };
        assert!(matches!(
            bad_sample.validate(),
            Err(ParamsError::NotPositive(_))
        ));

        let bad_rate = AnalyticsParams {
            default_rebate_rate: dec!(1.5),
            ..AnalyticsParams::default()
        };
        assert!(matches!(
            bad_rate.validate(),
            Err(ParamsError::RateOutOfRange { .. })
        ));

        let bad_threshold = AnalyticsParams {
            adverse: AdverseSelectionParams {
                negative_threshold: f64::NAN,
                ..AdverseSelectionParams::default()
            },
            ..AnalyticsParams::default()
        };
        assert!(matches!(
            bad_threshold.validate(),
            Err(ParamsError::NotFinite(_))
        ));
    }
}
