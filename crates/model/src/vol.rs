//! Per-asset EWMA volatility of one-second log returns — §8's `σ_1s`.
//!
//! Sans-IO by construction (scheduler-machine precedent): no clock reads, no
//! logging, no allocation on the tick path. Time enters only through tick
//! timestamps; every observable behavior is a return value, so tests are
//! deterministic. Floats are legal in this crate per CLAUDE.md §2.6 —
//! [`Decimal`](rust_decimal::Decimal) prices convert to `f64` at ingestion.
//!
//! One [`VolEstimator`] instance tracks one asset from one configured
//! `(source, kind)` input stream — every timestamp it ever differences comes
//! from a single clock domain. Ticks aggregate into one-second bars keyed by
//! the **exchange** timestamp (last tick in a second is the bar close); each
//! finalized bar folds one variance sample into a bias-corrected,
//! gap-aware EWMA. Until [`VolParams::warmup_returns`] samples have folded,
//! the estimator reports NOT_READY ([`VolQuality::NotReady`]) and the engine
//! path [`VolEstimator::sigma`] returns `None`.

use core_types::{Asset, PriceSource, PriceTick, TickKind};
use rust_decimal::prelude::ToPrimitive;
use thiserror::Error;

/// Longest a tick's exchange timestamp may lead its local receive time.
///
/// Guards the bar high-water mark: one glitch tick stamped far in the future
/// would otherwise silently drop every legitimate tick until the wall clock
/// caught up. Generous versus real skew (the operator's clock was ~1 s off);
/// backfill ticks always have `ts_exchange ≤ ts_local` and are unaffected.
const MAX_EXCHANGE_LEAD_MS: i64 = 10_000;

/// Upper bound on [`VolParams::max_gap_secs`] (one day). Keeps the
/// `i64 → i32`/`f64` gap casts in the fold provably lossless.
const MAX_GAP_LIMIT_SECS: u32 = 86_400;

/// Parameters for one [`VolEstimator`].
///
/// `half_life_secs` / `floor_1s` / `cap_1s` mirror the config crate's
/// `engine.defaults` fields (`ewma_half_life_secs`, `vol_floor_1s`,
/// `vol_cap_1s`) — authoritative values flow from config at the binary
/// boundary (the model crate depends only on core-types, §4). The remaining
/// fields are model-crate defaults; exposing them in config is deferred to
/// the model-driver task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolParams {
    /// Input feed the estimator listens to; all other sources are ignored.
    pub source: PriceSource,
    /// Input observation flavor (e.g. `Mid` for direct-Binance bookTicker).
    pub kind: TickKind,
    /// EWMA half-life in seconds: a sample's weight halves every this many
    /// seconds of bar time. Must be finite and > 0.
    pub half_life_secs: f64,
    /// Floor on the reported σ_1s — never calmer than this. Finite, > 0.
    pub floor_1s: f64,
    /// Cap on the reported σ_1s — never wilder than this. Finite, > floor.
    pub cap_1s: f64,
    /// Folded samples required before the estimator reports Ready. At the
    /// default 1 Hz input cadence this is seconds of data; an RTDS backfill
    /// (~60–120 one-second points) can satisfy it instantly. Must be ≥ 1.
    pub warmup_returns: u32,
    /// Longest bar-to-bar gap (seconds) folded as a sample. Longer gaps
    /// re-anchor the bar chain without touching the EWMA: a single
    /// outage-spanning return would otherwise carry weight ≈ 1 and wipe the
    /// estimate. Staleness trust is FeedHealth/risk territory, not the
    /// estimator's. Must be in `[1, 86 400]`.
    pub max_gap_secs: u32,
}

impl Default for VolParams {
    /// Mirrors `config/default.toml`'s `engine.defaults` (60 s half-life,
    /// floor 2e-5, cap 5e-3) and the 2026-06-12 feed-comparison finding that
    /// direct-Binance `Mid` is the fast signal.
    fn default() -> Self {
        Self {
            source: PriceSource::BinanceDirect,
            kind: TickKind::Mid,
            half_life_secs: 60.0,
            floor_1s: 0.000_02,
            cap_1s: 0.005,
            warmup_returns: 60,
            max_gap_secs: 30,
        }
    }
}

/// Why a [`VolEstimator`] refused to construct. Config-sourced fields are
/// validated by the config crate too; this re-check makes the estimator safe
/// against any construction path.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum VolError {
    /// `half_life_secs` was non-finite or ≤ 0.
    #[error("half_life_secs must be finite and > 0, got {0}")]
    BadHalfLife(f64),
    /// The floor/cap pair was non-finite or not `0 < floor < cap`.
    #[error("vol bounds must satisfy 0 < floor < cap (finite), got floor {floor}, cap {cap}")]
    BadBounds {
        /// Offending floor.
        floor: f64,
        /// Offending cap.
        cap: f64,
    },
    /// `warmup_returns` was 0 (Ready would precede the first sample).
    #[error("warmup_returns must be >= 1, got 0")]
    BadWarmup,
    /// `max_gap_secs` was 0 or above [`MAX_GAP_LIMIT_SECS`].
    #[error("max_gap_secs must be in [1, {MAX_GAP_LIMIT_SECS}], got {0}")]
    BadMaxGap(u32),
}

/// Whether the estimate has warmed up enough to trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolQuality {
    /// Fewer than [`VolParams::warmup_returns`] samples folded — NOT_READY.
    /// A `None` σ keeps the model without a probability, which the health
    /// monitor reports as [`core_types::ModelHealthReason::Warming`].
    NotReady,
    /// Warmed up; the reported sigma is a genuine multi-sample average.
    Ready,
}

/// Which configured bound, if any, the reported sigma is pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Clamp {
    /// The raw estimate lies inside `(floor, cap)` — reported as-is.
    Inactive,
    /// Raw estimate below the floor — reporting [`VolParams::floor_1s`].
    Floor,
    /// Raw estimate above the cap — reporting [`VolParams::cap_1s`].
    Cap,
}

/// Point-in-time estimator reading — the diagnostic surface (dashboard,
/// smoke runs). The engine consumes [`VolEstimator::sigma`] instead, which
/// gates on quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolEstimate {
    /// Clamped σ per √second; `Some` from the first folded sample onward
    /// (warming included), `None` before any sample.
    pub sigma_1s: Option<f64>,
    /// Warmup state.
    pub quality: VolQuality,
    /// Whether `sigma_1s` is pinned at a configured bound.
    pub clamp: Clamp,
    /// Total variance samples folded so far.
    pub returns_observed: u64,
}

/// Why [`VolEstimator::on_tick`] ignored a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IgnoreReason {
    /// Not this estimator's (asset, source, kind) — the normal case when
    /// every bus tick is offered to every estimator.
    WrongStream,
    /// Price was non-finite, zero, negative, or unrepresentable as `f64`.
    BadValue,
    /// The tick's second precedes the open bar — late delivery or a
    /// reconnect backfill replaying already-folded history (dropping it
    /// prevents double-folding).
    OutOfOrder,
    /// `ts_exchange` leads `ts_local` by more than [`MAX_EXCHANGE_LEAD_MS`]
    /// — a glitch timestamp that must not poison the bar high-water mark.
    ExchangeAheadOfLocal,
}

/// What one tick did to the estimator — telemetry for drivers and tests;
/// the machine itself stays log-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TickOutcome {
    /// Tick not consumed; see the reason.
    Ignored(IgnoreReason),
    /// Same-second tick: the open bar's close was overwritten.
    Absorbed,
    /// A bar opened with nothing to fold (first tick, or the finalized bar
    /// had no prior anchor).
    Opened,
    /// The previous bar finalized and one variance sample folded;
    /// `gap_secs` is the seconds spanned by the return (1 = contiguous).
    Folded {
        /// Seconds between the folded bar and its anchor.
        gap_secs: u32,
    },
    /// The previous bar finalized but its gap exceeded
    /// [`VolParams::max_gap_secs`]: the chain re-anchored, EWMA untouched.
    Reanchored {
        /// The oversized gap, saturated at `u32::MAX`.
        gap_secs: u32,
    },
}

/// One closed (or closing) one-second bar.
#[derive(Debug, Clone, Copy)]
struct Bar {
    /// Bar key: `floor(ts_exchange / 1000)`.
    second: i64,
    /// Last observed price within the second.
    close: f64,
}

/// Per-asset EWMA variance of one-second log returns (§8's σ_1s estimator).
///
/// # EWMA scheme
///
/// Per-second decay `λ = 2^(−1 / half_life_secs)`. A return `r` spanning
/// `k` seconds folds as the variance sample `r²/k` (for a driftless
/// diffusion `E[r²/k] = σ²_1s`) with decay `λᵏ`, in **bias-corrected** form:
///
/// ```text
/// acc  ← λᵏ·acc  + (1 − λᵏ)·(r²/k)
/// wsum ← λᵏ·wsum + (1 − λᵏ)          (starts 0, → 1)
/// var  = acc / wsum
/// ```
///
/// The normalization matters: seeding the EWMA with its first sample would
/// leave that single χ²(1)-noisy sample holding weight λ⁶⁰ = ½ after 60
/// folds at the default half-life — half the "warmed-up" estimate. With
/// bias correction the estimate is a properly normalized weighted average
/// from the first sample onward.
#[derive(Debug, Clone)]
pub struct VolEstimator {
    /// Asset this estimator tracks; ticks for others are ignored.
    asset: Asset,
    params: VolParams,
    /// Per-second decay, precomputed: `2^(−1 / half_life_secs)`.
    lambda: f64,
    /// Bar currently receiving ticks.
    open: Option<Bar>,
    /// Last finalized bar — the return-chain anchor.
    prev: Option<Bar>,
    /// Weighted sum of variance samples (numerator).
    acc: f64,
    /// Weight mass in `(0, 1]` after the first fold (denominator).
    wsum: f64,
    /// Folded-sample count (drives warmup).
    n_returns: u64,
}

impl VolEstimator {
    /// Builds an estimator for `asset`, validating `params`.
    ///
    /// # Errors
    /// A [`VolError`] naming the first invalid parameter.
    pub fn new(asset: Asset, params: VolParams) -> Result<Self, VolError> {
        if !params.half_life_secs.is_finite() || params.half_life_secs <= 0.0 {
            return Err(VolError::BadHalfLife(params.half_life_secs));
        }
        if !params.floor_1s.is_finite()
            || !params.cap_1s.is_finite()
            || params.floor_1s <= 0.0
            || params.cap_1s <= params.floor_1s
        {
            return Err(VolError::BadBounds {
                floor: params.floor_1s,
                cap: params.cap_1s,
            });
        }
        if params.warmup_returns == 0 {
            return Err(VolError::BadWarmup);
        }
        if params.max_gap_secs == 0 || params.max_gap_secs > MAX_GAP_LIMIT_SECS {
            return Err(VolError::BadMaxGap(params.max_gap_secs));
        }
        Ok(Self {
            asset,
            params,
            lambda: 2.0_f64.powf(-1.0 / params.half_life_secs),
            open: None,
            prev: None,
            acc: 0.0,
            wsum: 0.0,
            n_returns: 0,
        })
    }

    /// Consumes one price tick. Cheap for foreign ticks (three enum
    /// compares), allocation-free always.
    pub fn on_tick(&mut self, tick: &PriceTick) -> TickOutcome {
        if tick.asset != self.asset
            || tick.source != self.params.source
            || tick.kind != self.params.kind
        {
            return TickOutcome::Ignored(IgnoreReason::WrongStream);
        }
        let Some(value) = tick.value.to_f64() else {
            return TickOutcome::Ignored(IgnoreReason::BadValue);
        };
        if !value.is_finite() || value <= 0.0 {
            return TickOutcome::Ignored(IgnoreReason::BadValue);
        }
        let lead = tick.ts_exchange.signed_duration_since(tick.ts_local);
        if lead.as_millis() > MAX_EXCHANGE_LEAD_MS {
            return TickOutcome::Ignored(IgnoreReason::ExchangeAheadOfLocal);
        }
        // Bars key on the exchange timestamp: this estimator's single
        // (source, kind) input means one clock domain for every difference,
        // and backfill points (1 s apart in exchange time, delivered in one
        // local-time burst) seed bars correctly.
        let second = tick.ts_exchange.as_millis().div_euclid(1_000);
        let Some(open) = self.open else {
            self.open = Some(Bar {
                second,
                close: value,
            });
            return TickOutcome::Opened;
        };
        if second < open.second {
            return TickOutcome::Ignored(IgnoreReason::OutOfOrder);
        }
        if second == open.second {
            self.open = Some(Bar {
                second,
                close: value,
            });
            return TickOutcome::Absorbed;
        }
        // A later second arrived: the open bar is final. Fold (or
        // re-anchor), then open the new bar.
        let outcome = self.finalize(open);
        self.open = Some(Bar {
            second,
            close: value,
        });
        outcome
    }

    /// Folds a finalized bar against the chain anchor (or anchors the chain
    /// if there is none yet). The bar becomes the new anchor either way.
    fn finalize(&mut self, bar: Bar) -> TickOutcome {
        let Some(prev) = self.prev.replace(bar) else {
            return TickOutcome::Opened;
        };
        let k = bar.second - prev.second; // ≥ 1: bars finalize in key order
        if k > i64::from(self.params.max_gap_secs) {
            return TickOutcome::Reanchored {
                gap_secs: u32::try_from(k).unwrap_or(u32::MAX),
            };
        }
        // k ∈ [1, max_gap_secs ≤ 86 400]: every cast below is lossless.
        let (gap_f, gap_i, gap_u) = (k as f64, k as i32, k as u32);
        let r = (bar.close / prev.close).ln();
        let decay = self.lambda.powi(gap_i);
        self.acc = decay * self.acc + (1.0 - decay) * (r * r / gap_f);
        self.wsum = decay * self.wsum + (1.0 - decay);
        self.n_returns += 1;
        TickOutcome::Folded { gap_secs: gap_u }
    }

    /// The engine consumption path: `Some(clamped σ_1s)` **only when Ready**
    /// — a warming estimate can never leak into fair value or spreads.
    #[must_use]
    pub fn sigma(&self) -> Option<f64> {
        let estimate = self.estimate();
        match estimate.quality {
            VolQuality::Ready => estimate.sigma_1s,
            VolQuality::NotReady => None,
        }
    }

    /// Full diagnostic reading (dashboard / smoke-run surface): sigma is
    /// reported from the first folded sample, with quality and clamp state
    /// alongside.
    #[must_use]
    pub fn estimate(&self) -> VolEstimate {
        if self.n_returns == 0 {
            return VolEstimate {
                sigma_1s: None,
                quality: VolQuality::NotReady,
                clamp: Clamp::Inactive,
                returns_observed: 0,
            };
        }
        // acc ≥ 0 and wsum > 0 after any fold, so raw is a finite, non-
        // negative number — no NaN path.
        let raw = (self.acc / self.wsum).sqrt();
        let (sigma, clamp) = if raw < self.params.floor_1s {
            (self.params.floor_1s, Clamp::Floor)
        } else if raw > self.params.cap_1s {
            (self.params.cap_1s, Clamp::Cap)
        } else {
            (raw, Clamp::Inactive)
        };
        let quality = if self.n_returns >= u64::from(self.params.warmup_returns) {
            VolQuality::Ready
        } else {
            VolQuality::NotReady
        };
        VolEstimate {
            sigma_1s: Some(sigma),
            quality,
            clamp,
            returns_observed: self.n_returns,
        }
    }

    /// The asset this estimator tracks.
    #[must_use]
    pub const fn asset(&self) -> Asset {
        self.asset
    }

    /// The parameters this estimator runs with.
    #[must_use]
    pub const fn params(&self) -> &VolParams {
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

    fn estimator(params: VolParams) -> VolEstimator {
        VolEstimator::new(Asset::Btc, params).unwrap()
    }

    /// A tick on the default input (BinanceDirect Mid BTC) at `T0 + sec` with
    /// `ts_exchange == ts_local` (the bookTicker reality).
    fn tick_at(sec: i64, price: f64) -> PriceTick {
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

    /// xorshift64* (compare.rs precedent — `rand` is not on the tree),
    /// mapped to a uniform in [0, 1).
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (bits >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate standard normal via Irwin–Hall: the sum of 12 uniforms
    /// minus 6 has mean 0 and variance exactly 1.
    fn approx_gaussian(state: &mut u64) -> f64 {
        (0..12).map(|_| uniform(state)).sum::<f64>() - 6.0
    }

    /// Feeds a 1 Hz price path; returns the number of folds it produced.
    fn feed_path(est: &mut VolEstimator, start_sec: i64, prices: &[f64]) -> u64 {
        let mut folds = 0;
        for (i, &p) in prices.iter().enumerate() {
            if let TickOutcome::Folded { .. } = est.on_tick(&tick_at(start_sec + i as i64, p)) {
                folds += 1;
            }
        }
        folds
    }

    /// Alternating ±r log-return price path of `n` bars starting at `p0`.
    fn alternating_path(p0: f64, r: f64, n: usize) -> Vec<f64> {
        let mut prices = Vec::with_capacity(n);
        let mut p = p0;
        for i in 0..n {
            prices.push(p);
            p *= if i % 2 == 0 { r.exp() } else { (-r).exp() };
        }
        prices
    }

    #[test]
    fn constant_vol_series_recovers_sigma() {
        // σ_true = 5e-4 per √second; 1 Hz bars from a driftless geometric
        // walk with approx-Gaussian returns. The EWMA's steady-state
        // dispersion: rel-std of σ̂ ≈ ½·√(2(1−λ)/(1+λ)) ≈ 5.4% at HL = 60 s,
        // so 20% per checkpoint ≈ 3.7σ and 5% on the mean of 10 nearly
        // independent checkpoints (corr λ^600 ≈ 1e-3) ≈ 3σ. Deterministic:
        // fixed seed — if a future change to the PRNG draws ever lands
        // outside, adjust the seed, never the math.
        const SIGMA_TRUE: f64 = 5e-4;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut est = estimator(VolParams::default());
        let mut p = 63_500.0_f64;
        let mut folds: u64 = 0;
        let mut checkpoints = Vec::new();
        for sec in 0..6_602_i64 {
            if let TickOutcome::Folded { .. } = est.on_tick(&tick_at(sec, p)) {
                folds += 1;
                if folds.is_multiple_of(600) && folds <= 6_000 {
                    checkpoints.push(est.sigma().expect("ready after 600 folds"));
                }
            }
            p *= (SIGMA_TRUE * approx_gaussian(&mut state)).exp();
        }
        assert_eq!(checkpoints.len(), 10);
        for (i, sigma) in checkpoints.iter().enumerate() {
            let rel_err = (sigma - SIGMA_TRUE).abs() / SIGMA_TRUE;
            assert!(
                rel_err < 0.20,
                "checkpoint {i}: sigma {sigma:.3e} vs true {SIGMA_TRUE:.1e} (rel {rel_err:.3})"
            );
        }
        let mean = checkpoints.iter().sum::<f64>() / checkpoints.len() as f64;
        let rel_err = (mean - SIGMA_TRUE).abs() / SIGMA_TRUE;
        assert!(
            rel_err < 0.05,
            "checkpoint mean {mean:.4e} vs true {SIGMA_TRUE:.1e} (rel {rel_err:.3})"
        );
        let estimate = est.estimate();
        assert_eq!(estimate.quality, VolQuality::Ready);
        assert_eq!(estimate.clamp, Clamp::Inactive);
    }

    #[test]
    fn spike_excess_halves_after_half_life() {
        // Constant |r| baseline → var ≡ r_b² (to f64 noise ~1e-12 relative);
        // 2000 pre-spike folds saturate wsum (1 − wsum ≈ λ^2001 ≈ 1e-10) so
        // the bias-correction drift factor is ≈ 1 and the spike's excess
        // variance decays by exactly λ per second: ½ after one half-life,
        // ¼ after two. Excess is measured on σ² — sqrt is nonlinear and
        // sigma excess would NOT halve at the half-life.
        const R_BASE: f64 = 5e-4;
        const R_SPIKE: f64 = 1e-2; // post-spike σ ≈ 1.2e-3: inside (floor, cap)
        let mut est = estimator(VolParams::default());

        // Prices: 2001 baseline bars, then the spike return, then baseline
        // alternation resumes from the spiked price.
        let mut prices = alternating_path(63_500.0, R_BASE, 2_001);
        let last = *prices.last().unwrap();
        prices.extend(alternating_path(last * R_SPIKE.exp(), R_BASE, 122));

        let var_of = |est: &VolEstimator| {
            let e = est.estimate();
            assert_eq!(e.clamp, Clamp::Inactive);
            e.sigma_1s.unwrap().powi(2)
        };

        // Feed up to (but not including) the tick that folds the spike: the
        // return ln(p[2001]/p[2000]) folds when the tick at sec 2002 lands.
        for (i, &p) in prices.iter().enumerate().take(2_002) {
            est.on_tick(&tick_at(i as i64, p));
        }
        let baseline = var_of(&est);

        let spike_fold = est.on_tick(&tick_at(2_002, prices[2_002]));
        assert_eq!(spike_fold, TickOutcome::Folded { gap_secs: 1 });
        let excess_0 = var_of(&est) - baseline;
        assert!(excess_0 > 0.0, "spike must raise variance");

        for (i, &p) in prices.iter().enumerate().skip(2_003).take(60) {
            est.on_tick(&tick_at(i as i64, p));
        }
        let excess_60 = var_of(&est) - baseline;
        for (i, &p) in prices.iter().enumerate().skip(2_063).take(60) {
            est.on_tick(&tick_at(i as i64, p));
        }
        let excess_120 = var_of(&est) - baseline;

        assert!(
            (excess_60 / excess_0 - 0.5).abs() <= 1e-6,
            "one half-life: {} of the excess remains",
            excess_60 / excess_0
        );
        assert!(
            (excess_120 / excess_0 - 0.25).abs() <= 1e-6,
            "two half-lives: {} of the excess remains",
            excess_120 / excess_0
        );
    }

    #[test]
    fn floor_engages() {
        let params = VolParams::default();
        let mut est = estimator(params);
        // |r| = floor/10 → raw σ ≈ 2e-6, an order below the 2e-5 floor.
        let prices = alternating_path(63_500.0, params.floor_1s / 10.0, 70);
        feed_path(&mut est, 0, &prices);
        let estimate = est.estimate();
        assert_eq!(estimate.quality, VolQuality::Ready);
        assert_eq!(estimate.clamp, Clamp::Floor);
        assert_eq!(est.sigma(), Some(params.floor_1s), "pinned at the floor");
    }

    #[test]
    fn cap_engages() {
        let params = VolParams::default();
        let mut est = estimator(params);
        // |r| = 10·cap → raw σ ≈ 0.05, an order above the 5e-3 cap.
        let prices = alternating_path(63_500.0, params.cap_1s * 10.0, 70);
        feed_path(&mut est, 0, &prices);
        let estimate = est.estimate();
        assert_eq!(estimate.quality, VolQuality::Ready);
        assert_eq!(estimate.clamp, Clamp::Cap);
        assert_eq!(est.sigma(), Some(params.cap_1s), "pinned at the cap");
    }

    #[test]
    fn warmup_not_ready_until_threshold() {
        let params = VolParams {
            warmup_returns: 5,
            ..VolParams::default()
        };
        let mut est = estimator(params);
        let prices = alternating_path(63_500.0, 3e-4, 8);
        let mut folds = 0_u64;
        for (i, &p) in prices.iter().enumerate() {
            if let TickOutcome::Folded { .. } = est.on_tick(&tick_at(i as i64, p)) {
                folds += 1;
                let estimate = est.estimate();
                assert_eq!(estimate.returns_observed, folds);
                assert!(estimate.sigma_1s.is_some(), "warming sigma is visible");
                if folds < 5 {
                    assert_eq!(estimate.quality, VolQuality::NotReady);
                    assert_eq!(est.sigma(), None, "engine path gated while warming");
                } else {
                    assert_eq!(estimate.quality, VolQuality::Ready);
                    assert!(est.sigma().is_some());
                }
            }
        }
        assert_eq!(folds, 6, "8 bars → 6 folds (first two anchor the chain)");
    }

    #[test]
    fn no_data_reports_not_ready() {
        let est = estimator(VolParams::default());
        let estimate = est.estimate();
        assert_eq!(estimate.sigma_1s, None);
        assert_eq!(estimate.quality, VolQuality::NotReady);
        assert_eq!(estimate.returns_observed, 0);
        assert_eq!(est.sigma(), None);
    }

    #[test]
    fn backfill_burst_seeds_and_replay_is_dropped() {
        // 120 backfill points 1 s apart in exchange time, all delivered at
        // one local instant (the RTDS backfill shape): bars key on exchange
        // time, so they seed 118 folds and Ready immediately.
        let mut est = estimator(VolParams::default());
        let prices = alternating_path(63_500.0, 4e-4, 120);
        let local = TimestampMs::from_millis(T0 + 120_000);
        let mut folds = 0_u64;
        for (i, &p) in prices.iter().enumerate() {
            let tick = PriceTick {
                ts_local: local,
                ..tick_at(i as i64, p)
            };
            if let TickOutcome::Folded { .. } = est.on_tick(&tick) {
                folds += 1;
            }
        }
        assert_eq!(folds, 118, "120 bars → 118 folds");
        assert_eq!(
            est.estimate().quality,
            VolQuality::Ready,
            "seeded instantly"
        );

        // A reconnect's backfill replays already-seen seconds: every older-
        // than-open-bar tick is dropped — never double-folded.
        for (i, &p) in prices.iter().enumerate().skip(110).take(9) {
            let replay = PriceTick {
                ts_local: TimestampMs::from_millis(T0 + 125_000),
                ..tick_at(i as i64, p)
            };
            assert_eq!(
                est.on_tick(&replay),
                TickOutcome::Ignored(IgnoreReason::OutOfOrder)
            );
        }
        assert_eq!(est.estimate().returns_observed, 118, "no double-folding");

        // Live data resumes seamlessly after the burst.
        assert_eq!(
            est.on_tick(&tick_at(120, *prices.last().unwrap() * 1.0002)),
            TickOutcome::Folded { gap_secs: 1 }
        );
    }

    #[test]
    fn intra_second_ticks_collapse_to_one_bar() {
        let mut est = estimator(VolParams::default());
        // 100 ticks inside second 0; only the last (4th ms-offset variant
        // below) may matter.
        for i in 0..100_i64 {
            let tick = PriceTick {
                ts_exchange: TimestampMs::from_millis(T0 + i * 10),
                ts_local: TimestampMs::from_millis(T0 + i * 10),
                ..tick_at(0, 63_000.0 + i as f64)
            };
            let outcome = est.on_tick(&tick);
            if i == 0 {
                assert_eq!(outcome, TickOutcome::Opened);
            } else {
                assert_eq!(outcome, TickOutcome::Absorbed);
            }
        }
        let close_0 = 63_099.0; // last value of second 0
        let close_1 = 63_120.0;
        let close_2 = 63_110.0;
        assert_eq!(est.on_tick(&tick_at(1, close_1)), TickOutcome::Opened);
        assert_eq!(
            est.on_tick(&tick_at(2, close_2)),
            TickOutcome::Folded { gap_secs: 1 }
        );
        // Exactly one fold so far, and it used second 0's LAST value:
        // var = r² → raw sigma = |ln(close_1/close_0)|.
        let expected = (close_1 / close_0).ln().abs();
        let estimate = est.estimate();
        assert_eq!(estimate.returns_observed, 1);
        let sigma = estimate.sigma_1s.unwrap();
        assert!(
            (sigma - expected).abs() < 1e-15,
            "sigma {sigma} vs expected {expected}"
        );
    }

    #[test]
    fn gap_folds_r2_over_k_with_lambda_k() {
        let params = VolParams::default();
        let mut est = estimator(params);
        let lambda = 2.0_f64.powf(-1.0 / params.half_life_secs);
        let (p0, p1, p2, p7, p8) = (63_500.0, 63_510.0, 63_490.0, 63_530.0, 63_525.0);

        est.on_tick(&tick_at(0, p0));
        est.on_tick(&tick_at(1, p1));
        // Folds r(p0→p1), k=1, when sec-2's tick finalizes bar 1.
        assert_eq!(
            est.on_tick(&tick_at(2, p2)),
            TickOutcome::Folded { gap_secs: 1 }
        );
        // Bar 2 finalizes on the post-gap tick: r(p1→p2), still k=1.
        assert_eq!(
            est.on_tick(&tick_at(7, p7)),
            TickOutcome::Folded { gap_secs: 1 }
        );
        // Bar 7 finalizes next: the 5-second gap return r(p2→p7).
        assert_eq!(
            est.on_tick(&tick_at(8, p8)),
            TickOutcome::Folded { gap_secs: 5 }
        );

        // Replay the recursion by hand (powi, exactly as the estimator).
        let (mut acc, mut wsum) = (0.0_f64, 0.0_f64);
        for (from, to, k) in [(p0, p1, 1), (p1, p2, 1), (p2, p7, 5)] {
            let r: f64 = (to / from).ln();
            let decay = lambda.powi(k);
            acc = decay * acc + (1.0 - decay) * (r * r / f64::from(k));
            wsum = decay * wsum + (1.0 - decay);
        }
        let expected = (acc / wsum).sqrt();
        let sigma = est.estimate().sigma_1s.unwrap();
        assert!(
            (sigma - expected).abs() / expected < 1e-12,
            "sigma {sigma} vs hand-computed {expected}"
        );
        assert_eq!(est.estimate().returns_observed, 3);
    }

    #[test]
    fn gap_beyond_max_reanchors_without_folding() {
        let params = VolParams::default(); // max_gap_secs = 30
        let mut est = estimator(params);
        est.on_tick(&tick_at(0, 63_500.0));
        est.on_tick(&tick_at(1, 63_510.0));
        est.on_tick(&tick_at(2, 63_505.0));
        // Bar 2 still folds (k=1) when the post-outage tick arrives…
        assert_eq!(
            est.on_tick(&tick_at(40, 63_700.0)),
            TickOutcome::Folded { gap_secs: 1 }
        );
        let before = (est.acc, est.wsum, est.n_returns);

        // …but bar 40 itself is 38 s from its anchor: re-anchor, EWMA frozen.
        assert_eq!(
            est.on_tick(&tick_at(41, 63_705.0)),
            TickOutcome::Reanchored { gap_secs: 38 }
        );
        assert_eq!(
            (est.acc, est.wsum, est.n_returns),
            before,
            "re-anchor must not touch the EWMA"
        );

        // The chain continues from the re-anchored close.
        assert_eq!(
            est.on_tick(&tick_at(42, 63_710.0)),
            TickOutcome::Folded { gap_secs: 1 }
        );
        assert_eq!(est.estimate().returns_observed, before.2 + 1);
    }

    #[test]
    fn wrong_stream_ignored() {
        let mut est = estimator(VolParams::default());
        let wrong_asset = PriceTick {
            asset: Asset::Eth,
            ..tick_at(0, 3_500.0)
        };
        let wrong_source = PriceTick {
            source: PriceSource::ChainlinkRtds,
            ..tick_at(0, 63_500.0)
        };
        let wrong_kind = PriceTick {
            kind: TickKind::Trade,
            ..tick_at(0, 63_500.0)
        };
        for tick in [&wrong_asset, &wrong_source, &wrong_kind] {
            assert_eq!(
                est.on_tick(tick),
                TickOutcome::Ignored(IgnoreReason::WrongStream)
            );
        }
        assert!(est.open.is_none(), "foreign ticks must not open bars");
    }

    #[test]
    fn bad_values_and_future_timestamps_rejected() {
        let mut est = estimator(VolParams::default());
        for value in [Decimal::ZERO, Decimal::from_f64(-1.0).unwrap()] {
            let tick = PriceTick {
                value,
                ..tick_at(0, 1.0)
            };
            assert_eq!(
                est.on_tick(&tick),
                TickOutcome::Ignored(IgnoreReason::BadValue)
            );
        }
        // Exchange clock 11 s ahead of local: rejected (10 s is the bound).
        let glitch = PriceTick {
            ts_exchange: TimestampMs::from_millis(T0 + 11_000),
            ts_local: TimestampMs::from_millis(T0),
            ..tick_at(0, 63_500.0)
        };
        assert_eq!(
            est.on_tick(&glitch),
            TickOutcome::Ignored(IgnoreReason::ExchangeAheadOfLocal)
        );
        // Exactly at the bound is accepted.
        let edge = PriceTick {
            ts_exchange: TimestampMs::from_millis(T0 + 10_000),
            ts_local: TimestampMs::from_millis(T0),
            ..tick_at(0, 63_500.0)
        };
        assert_eq!(est.on_tick(&edge), TickOutcome::Opened);
        // Backfill shape (exchange far BEHIND local) is always fine.
        assert!(est.open.is_some());
    }

    #[test]
    fn params_validation_rejects_each_bad_field() {
        let ok = VolParams::default();
        let mk = |p: VolParams| VolEstimator::new(Asset::Btc, p);
        assert!(mk(ok).is_ok());
        // NaN payloads compare unequal to themselves — match on the variant.
        for hl in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                mk(VolParams {
                    half_life_secs: hl,
                    ..ok
                })
                .unwrap_err(),
                VolError::BadHalfLife(_)
            ));
        }
        for (floor, cap) in [
            (0.0, 0.005),
            (-1e-5, 0.005),
            (f64::NAN, 0.005),
            (2e-5, f64::NAN),
            (2e-5, 2e-5),
            (2e-5, 1e-6),
        ] {
            assert!(matches!(
                mk(VolParams {
                    floor_1s: floor,
                    cap_1s: cap,
                    ..ok
                })
                .unwrap_err(),
                VolError::BadBounds { .. }
            ));
        }
        assert_eq!(
            mk(VolParams {
                warmup_returns: 0,
                ..ok
            })
            .unwrap_err(),
            VolError::BadWarmup
        );
        for gap in [0, 86_401] {
            assert_eq!(
                mk(VolParams {
                    max_gap_secs: gap,
                    ..ok
                })
                .unwrap_err(),
                VolError::BadMaxGap(gap)
            );
        }
        assert!(
            mk(VolParams {
                max_gap_secs: 86_400,
                ..ok
            })
            .is_ok()
        );
    }
}
