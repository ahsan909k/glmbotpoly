//! Plain parameters for the paper venue.
//!
//! Mirrors the `bot::live::live_params` / `bot::vol::vol_params` convention: the
//! `config` crate's `PaperConfig` is mapped into this plain struct **at the
//! binary boundary**, so `venue-paper` never depends on `config`. Defaults here
//! match `config::PaperConfig::default` so the two cannot silently drift.

use core_types::{Decimal, Dollars, DurationMs};
use rust_decimal::dec;

/// A simulated latency: uniform in `[mean_ms, mean_ms + jitter_ms]`.
///
/// Jitter is one-sided **upward** on purpose (CLAUDE.md §9): a paper action can
/// only ever be slower than the configured mean, never faster — paper must never
/// be more optimistic than reality. `jitter_ms == 0` ⇒ exactly `mean_ms`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencySpec {
    /// Minimum simulated latency.
    pub mean_ms: DurationMs,
    /// Additional uniform random latency on top of `mean_ms`.
    pub jitter_ms: DurationMs,
}

impl Default for LatencySpec {
    fn default() -> Self {
        Self {
            mean_ms: DurationMs::from_millis(150),
            jitter_ms: DurationMs::from_millis(100),
        }
    }
}

/// Configuration for the paper venue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaperParams {
    /// Starting capital of the paper wallet (boot value; runtime-adjustable
    /// later from the dashboard — that seam is `PaperWallet::set_capital`).
    pub starting_capital: Dollars,
    /// Simulated order-placement latency.
    pub placement: LatencySpec,
    /// Simulated cancel round-trip latency.
    pub cancel: LatencySpec,
    /// Fallback taker fee rate, used **only** when a market's own fee params
    /// have not arrived yet (§7: never hardcode 0.07 in logic).
    pub default_fee_rate: Decimal,
    /// Maker rebate share of collected taker fees (crypto category: 20%) —
    /// used by the rebate-estimate accumulator (reported, not yet credited).
    pub maker_rebate_share: Decimal,
    /// Capacity of the order/fill event channel.
    pub event_channel_capacity: usize,
    /// Seed for the deterministic latency RNG. `None` ⇒ a fixed default seed;
    /// the binary seeds it from the clock, tests pin it for reproducibility.
    pub rng_seed: Option<u64>,
}

impl Default for PaperParams {
    fn default() -> Self {
        Self {
            starting_capital: Dollars::new(Decimal::from(10_000)),
            placement: LatencySpec::default(),
            cancel: LatencySpec::default(),
            default_fee_rate: dec!(0.07),
            maker_rebate_share: dec!(0.20),
            event_channel_capacity: 1024,
            rng_seed: None,
        }
    }
}

/// Why a [`PaperParams`] is invalid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PaperParamsError {
    /// A latency mean was below 1 ms — zero-latency paper actions would be more
    /// optimistic than reality (§9).
    #[error("{which} latency mean_ms must be >= 1 (got {got}ms)")]
    LatencyMeanTooLow {
        /// `"placement"` or `"cancel"`.
        which: &'static str,
        /// The offending value in milliseconds.
        got: i64,
    },
    /// A latency jitter was negative.
    #[error("{which} latency jitter_ms must be >= 0 (got {got}ms)")]
    LatencyJitterNegative {
        /// `"placement"` or `"cancel"`.
        which: &'static str,
        /// The offending value in milliseconds.
        got: i64,
    },
    /// Starting capital was not strictly positive.
    #[error("starting_capital must be > 0 (got {0})")]
    NonPositiveCapital(Dollars),
    /// The fallback fee rate was outside `[0, 1)`.
    #[error("default_fee_rate must be in [0, 1) (got {0})")]
    FeeRateOutOfRange(Decimal),
    /// The maker rebate share was outside `[0, 1]`.
    #[error("maker_rebate_share must be in [0, 1] (got {0})")]
    RebateShareOutOfRange(Decimal),
}

impl PaperParams {
    /// Validates the parameters (belt-and-suspenders: the `config` crate already
    /// validates `PaperConfig`, but the boundary mapping is hand-written).
    ///
    /// # Errors
    /// [`PaperParamsError`] for the first violation found.
    pub fn validate(&self) -> Result<(), PaperParamsError> {
        for (which, spec) in [("placement", &self.placement), ("cancel", &self.cancel)] {
            if spec.mean_ms.as_millis() < 1 {
                return Err(PaperParamsError::LatencyMeanTooLow {
                    which,
                    got: spec.mean_ms.as_millis(),
                });
            }
            if spec.jitter_ms.is_negative() {
                return Err(PaperParamsError::LatencyJitterNegative {
                    which,
                    got: spec.jitter_ms.as_millis(),
                });
            }
        }
        if self.starting_capital.as_decimal() <= Decimal::ZERO {
            return Err(PaperParamsError::NonPositiveCapital(self.starting_capital));
        }
        if self.default_fee_rate < Decimal::ZERO || self.default_fee_rate >= Decimal::ONE {
            return Err(PaperParamsError::FeeRateOutOfRange(self.default_fee_rate));
        }
        if self.maker_rebate_share < Decimal::ZERO || self.maker_rebate_share > Decimal::ONE {
            return Err(PaperParamsError::RebateShareOutOfRange(
                self.maker_rebate_share,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(PaperParams::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_latency() {
        let params = PaperParams {
            placement: LatencySpec {
                mean_ms: DurationMs::ZERO,
                jitter_ms: DurationMs::ZERO,
            },
            ..PaperParams::default()
        };
        assert!(matches!(
            params.validate(),
            Err(PaperParamsError::LatencyMeanTooLow {
                which: "placement",
                ..
            })
        ));
    }

    #[test]
    fn rejects_nonpositive_capital() {
        let params = PaperParams {
            starting_capital: Dollars::ZERO,
            ..PaperParams::default()
        };
        assert!(matches!(
            params.validate(),
            Err(PaperParamsError::NonPositiveCapital(_))
        ));
    }

    #[test]
    fn rejects_out_of_range_rates() {
        let bad_fee = PaperParams {
            default_fee_rate: dec!(1.0),
            ..PaperParams::default()
        };
        assert!(matches!(
            bad_fee.validate(),
            Err(PaperParamsError::FeeRateOutOfRange(_))
        ));
        let bad_rebate = PaperParams {
            maker_rebate_share: dec!(1.5),
            ..PaperParams::default()
        };
        assert!(matches!(
            bad_rebate.validate(),
            Err(PaperParamsError::RebateShareOutOfRange(_))
        ));
    }
}
