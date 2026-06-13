//! Per-asset Chainlink-minus-Binance basis tracker (§8).
//!
//! Chainlink (the resolution feed) leads or lags direct Binance (the fast
//! signal) by a small, slowly-drifting offset — measured at ≈ −13.3 bps over a
//! 20-minute window on 2026-06-12, material against our one-tick edges. The
//! fair-value engine anchors on the basis-corrected Binance price when the
//! fast feed is fresh, so it needs a live estimate of that offset.
//!
//! Sans-IO, same bias-corrected EWMA machinery as [`crate::vol`], but tracking
//! a **level** (the log basis `b = ln(P_chainlink / P_binance)`) rather than a
//! variance. The log form is additive in the fair-value `z`
//! (`ln(S_corrected/K) = ln(S_fast/K) + b`) and scale-free; it renders as bps
//! via `(e^b − 1)·10⁴`.
//!
//! One sample folds per live Chainlink tick, paired with the most recent fresh
//! Binance mid. The **backfill guard** is the load-bearing difference from
//! [`crate::strike`]: strike capture *wants* the RTDS backfill burst, but the
//! basis must reject it — pairing a minutes-old Chainlink print against the
//! current Binance mid would inject a huge spurious basis. So only live
//! Chainlink ticks (recent exchange time) fold.

use core_types::{Asset, PriceSource, PriceTick, TickKind, TimestampMs};
use rust_decimal::prelude::ToPrimitive;
use thiserror::Error;

/// A Chainlink tick whose exchange time trails local receipt by more than this
/// is treated as backfill and never folded (see the module docs). Live RTDS
/// updates run ~1 s behind exchange time; backfill prints are minutes old.
const BACKFILL_GUARD_MS: i64 = 5_000;

/// Upper bound on [`BasisParams::max_decay_secs`] (one day), keeping the gap
/// cast in [`BasisTracker::fold`] lossless.
const MAX_DECAY_LIMIT_SECS: u32 = 86_400;

/// Parameters for one [`BasisTracker`].
///
/// `pairing_max_age_ms` mirrors `feeds.binance_stale_after_ms` from config at
/// the binary boundary; the rest are model-crate constants (the vol-estimator
/// precedent — config exposure is the model-driver task's job).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisParams {
    /// EWMA half-life in seconds. Long (the basis drifts slowly), so a brief
    /// run of pairings cannot swing it. Finite, > 0.
    pub half_life_secs: f64,
    /// A Chainlink tick folds only if the most recent Binance mid is no older
    /// than this. Must be > 0.
    pub pairing_max_age_ms: i64,
    /// Samples required before [`BasisTracker::basis_log`] returns a value.
    /// Must be ≥ 1.
    pub warmup_samples: u32,
    /// Longest gap (seconds) the decay accounts for; longer gaps are clamped
    /// so one post-outage sample cannot dominate the estimate. Must be in
    /// `[1, 86 400]`.
    pub max_decay_secs: u32,
}

impl Default for BasisParams {
    fn default() -> Self {
        Self {
            half_life_secs: 120.0,
            pairing_max_age_ms: 2_500,
            warmup_samples: 30,
            max_decay_secs: 30,
        }
    }
}

/// Why a [`BasisTracker`] refused to construct.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum BasisError {
    /// `half_life_secs` was non-finite or ≤ 0.
    #[error("half_life_secs must be finite and > 0, got {0}")]
    BadHalfLife(f64),
    /// `pairing_max_age_ms` was ≤ 0.
    #[error("pairing_max_age_ms must be > 0, got {0}")]
    BadPairingAge(i64),
    /// `warmup_samples` was 0.
    #[error("warmup_samples must be >= 1, got 0")]
    BadWarmup,
    /// `max_decay_secs` was 0 or above [`MAX_DECAY_LIMIT_SECS`].
    #[error("max_decay_secs must be in [1, {MAX_DECAY_LIMIT_SECS}], got {0}")]
    BadMaxDecay(u32),
}

/// Why [`BasisTracker::on_tick`] did not fold a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisIgnoreReason {
    /// Not a stream this tracker pairs (the asset's Chainlink vendor or direct
    /// Binance mid).
    WrongStream,
    /// Price non-finite, zero, negative, or unrepresentable as `f64`.
    BadValue,
    /// A Chainlink backfill print (old exchange time) — must not pair against
    /// the current Binance mid.
    Backfill,
    /// A Chainlink tick arrived before any Binance mid — nothing to pair with.
    NoFast,
    /// The most recent Binance mid is older than `pairing_max_age_ms`.
    FastStale,
}

/// What one tick did to the tracker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasisOutcome {
    /// Tick not used; see the reason.
    Ignored(BasisIgnoreReason),
    /// A Binance mid refreshed the pairing price.
    FastUpdated,
    /// A basis sample folded into the EWMA.
    Folded,
}

/// Point-in-time tracker reading — the diagnostic surface. The engine consumes
/// [`BasisTracker::basis_log`], which gates on warmup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisEstimate {
    /// Log basis `b`; `Some` from the first folded sample (warming included),
    /// `None` before any.
    pub basis_log: Option<f64>,
    /// `b` expressed in basis points (`(e^b − 1)·10⁴`).
    pub bps: Option<f64>,
    /// Samples folded so far.
    pub samples: u64,
    /// Whether warmup is satisfied.
    pub warmed: bool,
}

/// Per-asset EWMA of the Chainlink-minus-Binance log basis. See the module
/// docs for the design.
#[derive(Debug, Clone)]
pub struct BasisTracker {
    asset: Asset,
    params: BasisParams,
    /// Per-second decay `2^(−1/half_life_secs)`.
    lambda: f64,
    /// Latest Binance mid value and its local receive time.
    last_fast: Option<(f64, TimestampMs)>,
    /// Weighted sum of basis samples (numerator).
    acc: f64,
    /// Weight mass in `(0, 1]` after the first fold (denominator).
    wsum: f64,
    /// Folded-sample count (drives warmup).
    n_samples: u64,
    /// Chainlink exchange second of the last fold (decay spacing).
    last_fold_sec: Option<i64>,
}

impl BasisTracker {
    /// Builds a tracker for `asset`, validating `params`.
    ///
    /// # Errors
    /// A [`BasisError`] naming the first invalid parameter.
    pub fn new(asset: Asset, params: BasisParams) -> Result<Self, BasisError> {
        if !params.half_life_secs.is_finite() || params.half_life_secs <= 0.0 {
            return Err(BasisError::BadHalfLife(params.half_life_secs));
        }
        if params.pairing_max_age_ms <= 0 {
            return Err(BasisError::BadPairingAge(params.pairing_max_age_ms));
        }
        if params.warmup_samples == 0 {
            return Err(BasisError::BadWarmup);
        }
        if params.max_decay_secs == 0 || params.max_decay_secs > MAX_DECAY_LIMIT_SECS {
            return Err(BasisError::BadMaxDecay(params.max_decay_secs));
        }
        Ok(Self {
            asset,
            params,
            lambda: 2.0_f64.powf(-1.0 / params.half_life_secs),
            last_fast: None,
            acc: 0.0,
            wsum: 0.0,
            n_samples: 0,
            last_fold_sec: None,
        })
    }

    /// Consumes one price tick.
    pub fn on_tick(&mut self, tick: &PriceTick) -> BasisOutcome {
        if tick.asset != self.asset {
            return BasisOutcome::Ignored(BasisIgnoreReason::WrongStream);
        }
        let Some(value) = tick.value.to_f64() else {
            return BasisOutcome::Ignored(BasisIgnoreReason::BadValue);
        };
        if !value.is_finite() || value <= 0.0 {
            return BasisOutcome::Ignored(BasisIgnoreReason::BadValue);
        }
        match (tick.source, tick.kind) {
            (PriceSource::BinanceDirect, TickKind::Mid) => {
                self.last_fast = Some((value, tick.ts_local));
                BasisOutcome::FastUpdated
            }
            (PriceSource::ChainlinkRtds, TickKind::Vendor) => {
                let lead = tick.ts_local.signed_duration_since(tick.ts_exchange);
                if lead.as_millis() > BACKFILL_GUARD_MS {
                    return BasisOutcome::Ignored(BasisIgnoreReason::Backfill);
                }
                let Some((fast_value, fast_ts)) = self.last_fast else {
                    return BasisOutcome::Ignored(BasisIgnoreReason::NoFast);
                };
                let pair_age = (tick.ts_local.as_millis() - fast_ts.as_millis()).abs();
                if pair_age > self.params.pairing_max_age_ms {
                    return BasisOutcome::Ignored(BasisIgnoreReason::FastStale);
                }
                let b = (value / fast_value).ln();
                self.fold(b, tick.ts_exchange);
                BasisOutcome::Folded
            }
            _ => BasisOutcome::Ignored(BasisIgnoreReason::WrongStream),
        }
    }

    /// Folds one basis sample, time-decayed by the gap since the last fold.
    fn fold(&mut self, b: f64, ts_exchange: TimestampMs) {
        let sec = ts_exchange.as_millis().div_euclid(1_000);
        let decay = match self.last_fold_sec {
            None => 0.0, // first sample: acc := b, wsum := 1 (bias-corrected)
            Some(prev) => {
                // Clamp the gap so a single post-outage sample cannot wipe the
                // estimate; ≥ 1 so same-second folds still decay once.
                let k = (sec - prev).clamp(1, i64::from(self.params.max_decay_secs));
                self.lambda.powi(k as i32)
            }
        };
        self.acc = decay * self.acc + (1.0 - decay) * b;
        self.wsum = decay * self.wsum + (1.0 - decay);
        self.n_samples += 1;
        self.last_fold_sec = Some(sec);
    }

    /// The engine consumption path: the log basis, `Some` only once warmed.
    #[must_use]
    pub fn basis_log(&self) -> Option<f64> {
        if self.n_samples >= u64::from(self.params.warmup_samples) {
            Some(self.acc / self.wsum)
        } else {
            None
        }
    }

    /// Full diagnostic reading (warming sigma visible from the first sample).
    #[must_use]
    pub fn estimate(&self) -> BasisEstimate {
        let log = (self.n_samples > 0).then(|| self.acc / self.wsum);
        BasisEstimate {
            basis_log: log,
            bps: log.map(|b| (b.exp() - 1.0) * 10_000.0),
            samples: self.n_samples,
            warmed: self.n_samples >= u64::from(self.params.warmup_samples),
        }
    }

    /// The asset this tracker pairs.
    #[must_use]
    pub const fn asset(&self) -> Asset {
        self.asset
    }

    /// The parameters in force.
    #[must_use]
    pub const fn params(&self) -> &BasisParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use core_types::TimestampMs;
    use rust_decimal::Decimal;
    use rust_decimal::prelude::FromPrimitive;

    use super::*;

    const T0: i64 = 1_700_000_000_000;

    fn tracker(params: BasisParams) -> BasisTracker {
        BasisTracker::new(Asset::Btc, params).unwrap()
    }

    fn binance_mid(sec: i64, price: f64) -> PriceTick {
        let ts = TimestampMs::from_millis(T0 + sec * 1_000);
        PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Mid,
            value: Decimal::from_f64(price).unwrap(),
            ts_exchange: ts,
            ts_local: ts,
        }
    }

    /// A live Chainlink tick (exchange ≈ local) at `sec`.
    fn chainlink(sec: i64, price: f64) -> PriceTick {
        let ts = TimestampMs::from_millis(T0 + sec * 1_000);
        PriceTick {
            source: PriceSource::ChainlinkRtds,
            asset: Asset::Btc,
            kind: TickKind::Vendor,
            value: Decimal::from_f64(price).unwrap(),
            ts_exchange: ts,
            ts_local: ts,
        }
    }

    #[test]
    fn constant_basis_is_recovered() {
        // Chainlink sits a constant +3 bps above Binance: b = ln(1.0003).
        const RATIO: f64 = 1.000_3;
        let mut t = tracker(BasisParams::default());
        let fast = 63_500.0;
        for sec in 0..200 {
            t.on_tick(&binance_mid(sec, fast));
            t.on_tick(&chainlink(sec, fast * RATIO));
        }
        let est = t.estimate();
        assert!(est.warmed);
        let b = est.basis_log.unwrap();
        assert!((b - RATIO.ln()).abs() < 1e-12, "b {b} vs {}", RATIO.ln());
        let bps = est.bps.unwrap();
        assert!((bps - 3.0).abs() < 0.02, "bps {bps}");
        assert_eq!(t.basis_log(), Some(b));
    }

    #[test]
    fn warmup_gates_the_engine_path() {
        let params = BasisParams {
            warmup_samples: 5,
            ..BasisParams::default()
        };
        let mut t = tracker(params);
        for sec in 0..4 {
            t.on_tick(&binance_mid(sec, 63_500.0));
            assert_eq!(t.on_tick(&chainlink(sec, 63_520.0)), BasisOutcome::Folded);
            // Warming: diagnostic value visible, engine path still None.
            assert!(t.estimate().basis_log.is_some());
            assert_eq!(t.basis_log(), None);
        }
        t.on_tick(&binance_mid(4, 63_500.0));
        t.on_tick(&chainlink(4, 63_520.0));
        assert!(t.estimate().warmed);
        assert!(t.basis_log().is_some());
    }

    #[test]
    fn chainlink_before_any_binance_does_not_fold() {
        let mut t = tracker(BasisParams::default());
        assert_eq!(
            t.on_tick(&chainlink(0, 63_500.0)),
            BasisOutcome::Ignored(BasisIgnoreReason::NoFast)
        );
        assert_eq!(t.estimate().samples, 0);
    }

    #[test]
    fn stale_fast_pairing_is_skipped() {
        let params = BasisParams::default(); // pairing_max_age_ms = 2 500
        let mut t = tracker(params);
        t.on_tick(&binance_mid(0, 63_500.0));
        // Chainlink 3 s later: the mid is older than the 2.5 s pairing window.
        assert_eq!(
            t.on_tick(&chainlink(3, 63_520.0)),
            BasisOutcome::Ignored(BasisIgnoreReason::FastStale)
        );
        assert_eq!(t.estimate().samples, 0);
    }

    #[test]
    fn backfill_chainlink_is_not_folded() {
        let mut t = tracker(BasisParams::default());
        t.on_tick(&binance_mid(100, 63_500.0));
        // A Chainlink backfill print: exchange time 2 minutes behind local.
        let backfill = PriceTick {
            ts_exchange: TimestampMs::from_millis(T0 + 100_000 - 120_000),
            ts_local: TimestampMs::from_millis(T0 + 100_000),
            ..chainlink(100, 63_520.0)
        };
        assert_eq!(
            t.on_tick(&backfill),
            BasisOutcome::Ignored(BasisIgnoreReason::Backfill)
        );
        assert_eq!(t.estimate().samples, 0);
    }

    #[test]
    fn gap_decay_matches_hand_recursion() {
        let params = BasisParams::default();
        let mut t = tracker(params);
        let lambda = 2.0_f64.powf(-1.0 / params.half_life_secs);
        // (binance, chainlink) folds at seconds 0, 1, 6 (a 5 s gap).
        let samples = [
            (63_500.0, 63_520.0, 0_i64),
            (63_510.0, 63_531.0, 1),
            (63_490.0, 63_512.0, 6),
        ];
        for &(fast, cl, sec) in &samples {
            t.on_tick(&binance_mid(sec, fast));
            assert_eq!(t.on_tick(&chainlink(sec, cl)), BasisOutcome::Folded);
        }
        let (mut acc, mut wsum) = (0.0_f64, 0.0_f64);
        let mut prev: Option<i64> = None;
        for &(fast, cl, sec) in &samples {
            let b = (cl / fast).ln();
            let decay = match prev {
                None => 0.0,
                Some(p) => lambda.powi((sec - p).clamp(1, 30) as i32),
            };
            acc = decay * acc + (1.0 - decay) * b;
            wsum = decay * wsum + (1.0 - decay);
            prev = Some(sec);
        }
        let expected = acc / wsum;
        assert!((t.estimate().basis_log.unwrap() - expected).abs() < 1e-15);
    }

    #[test]
    fn long_gap_does_not_wipe_the_estimate() {
        // After a long outage, one fold of a wildly different basis must not
        // dominate: the decay is clamped to max_decay_secs.
        let params = BasisParams {
            warmup_samples: 1,
            ..BasisParams::default()
        };
        let mut t = tracker(params);
        for sec in 0..60 {
            t.on_tick(&binance_mid(sec, 63_500.0));
            t.on_tick(&chainlink(sec, 63_500.0)); // b = 0
        }
        let before = t.basis_log().unwrap();
        assert!(before.abs() < 1e-9);
        // 1-hour gap, then a +30 bps sample. Clamped decay (λ^30) keeps most of
        // the history.
        t.on_tick(&binance_mid(3_660, 63_500.0));
        t.on_tick(&chainlink(3_660, 63_500.0 * 1.003));
        let after = t.basis_log().unwrap();
        let lambda30 = 2.0_f64.powf(-30.0 / 120.0);
        assert!(after < (1.0 - lambda30) * 0.003 + 1e-4, "after {after}");
        assert!(after > 0.0);
    }

    #[test]
    fn wrong_stream_and_bad_value_rejected() {
        let mut t = tracker(BasisParams::default());
        let eth = PriceTick {
            asset: Asset::Eth,
            ..binance_mid(0, 3_500.0)
        };
        assert_eq!(
            t.on_tick(&eth),
            BasisOutcome::Ignored(BasisIgnoreReason::WrongStream)
        );
        // Binance trade (not mid) and Chainlink-RTDS-binance are not paired.
        let trade = PriceTick {
            kind: TickKind::Trade,
            ..binance_mid(0, 63_500.0)
        };
        assert_eq!(
            t.on_tick(&trade),
            BasisOutcome::Ignored(BasisIgnoreReason::WrongStream)
        );
        let zero = PriceTick {
            value: Decimal::ZERO,
            ..binance_mid(0, 1.0)
        };
        assert_eq!(
            t.on_tick(&zero),
            BasisOutcome::Ignored(BasisIgnoreReason::BadValue)
        );
    }

    #[test]
    fn params_validation_rejects_each_bad_field() {
        let ok = BasisParams::default();
        let mk = |p: BasisParams| BasisTracker::new(Asset::Btc, p);
        assert!(mk(ok).is_ok());
        for hl in [0.0, -1.0, f64::NAN] {
            assert!(matches!(
                mk(BasisParams {
                    half_life_secs: hl,
                    ..ok
                })
                .unwrap_err(),
                BasisError::BadHalfLife(_)
            ));
        }
        assert_eq!(
            mk(BasisParams {
                pairing_max_age_ms: 0,
                ..ok
            })
            .unwrap_err(),
            BasisError::BadPairingAge(0)
        );
        assert_eq!(
            mk(BasisParams {
                warmup_samples: 0,
                ..ok
            })
            .unwrap_err(),
            BasisError::BadWarmup
        );
        for gap in [0, 86_401] {
            assert_eq!(
                mk(BasisParams {
                    max_decay_secs: gap,
                    ..ok
                })
                .unwrap_err(),
                BasisError::BadMaxDecay(gap)
            );
        }
    }
}
