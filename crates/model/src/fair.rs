//! The per-window fair-value engine (§8): `p_up = Φ(z)`,
//! `z = ln(S/K) / (σ_1s·√τ)`.
//!
//! Sans-IO. One [`FairValueEngine`] owns one window's strike capture and the
//! latest input prices; the slowly-varying per-asset state (σ from
//! [`crate::vol`], the basis from [`crate::basis`]) lives outside and is passed
//! into [`FairValueEngine::compute`] as [`FairInputs`] — a window rollover must
//! never reset σ or the basis.
//!
//! `compute` is pure given its inputs and the engine's stored prices: it
//! produces a rich [`FairValue`] for rendering and journaling, and
//! [`FairValue::snapshot`] distills the bus-facing [`ModelSnapshot`]. The
//! engine recomputes on every relevant tick and on the driver's 100 ms
//! heartbeat (so τ-decay between ticks is reflected).
//!
//! Anchoring (§8): the basis-corrected direct-Binance price is the fast signal
//! and is used whenever the Binance feed is fresh and the basis has warmed;
//! otherwise the resolution-grade Chainlink price. With neither fresh the model
//! is [`ModelHealth::Unreliable`] and publishes no probability.

use core_types::{
    AnchorSource, Asset, DurationMs, InputAges, MarketInfo, ModelHealth, ModelHealthReason,
    ModelSnapshot, PriceSource, PriceTick, TickKind, TimestampMs, WindowId,
};
use rust_decimal::prelude::ToPrimitive;

use crate::health::HealthInputs;
use crate::math::norm_cdf;
use crate::strike::{StrikeCapture, StrikeError, StrikeEstimate, StrikeOutcome};

/// `z` is clamped to this magnitude before `Φ`. `Φ(±39)` is already exactly
/// `1.0`/`0.0` in `f64`, and a finite `z` always serializes (JSON has no
/// infinity), so the clamp is lossless for `p_up` and keeps the bus snapshot
/// well-formed when a tiny τ blows the ratio up.
const Z_CLAMP: f64 = 39.0;

/// A Chainlink tick whose exchange time trails local receipt by more than this
/// is a backfill print (an *old* price) and must not become the current `S`.
/// Same rationale as [`crate::basis`]'s backfill guard.
const PRICE_LIVENESS_MS: i64 = 5_000;

/// Sentinel age reported for a feed that has never delivered (so the bus
/// snapshot reads "maximally stale" rather than falsely fresh).
const NEVER_SEEN_AGE: DurationMs = DurationMs::from_millis(i64::MAX);

/// Tuning for one [`FairValueEngine`]. The staleness bounds mirror the
/// `feeds.*` config keys at the binary boundary; they decide anchor selection
/// (basis-corrected fast vs Chainlink). The health gate's divergence/staleness
/// thresholds live in [`crate::health::HealthParams`], not here — this engine
/// only *measures* divergence (`FairValue::divergence_bps`); the per-asset
/// [`crate::HealthMonitor`] decides what it means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FairParams {
    /// Chainlink age beyond which the resolution anchor is considered lost
    /// (`feeds.rtds_stale_after_ms`, 5 000). The fast feed alone is never
    /// quotable (§8: Chainlink is ground truth).
    pub chainlink_stale_ms: i64,
    /// Direct-Binance age beyond which the fast signal is unusable
    /// (`feeds.binance_stale_after_ms`, 2 500).
    pub fast_stale_ms: i64,
}

impl Default for FairParams {
    fn default() -> Self {
        Self {
            chainlink_stale_ms: 5_000,
            fast_stale_ms: 2_500,
        }
    }
}

/// The per-asset inputs `compute` needs that live outside the per-window
/// engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FairInputs {
    /// Evaluation time.
    pub now: TimestampMs,
    /// One-second log-return volatility, `Some` only when the estimator is
    /// Ready ([`crate::VolEstimator::sigma`]).
    pub sigma_1s: Option<f64>,
    /// Log basis `ln(P_chainlink/P_binance)`, `Some` only when the tracker has
    /// warmed ([`crate::BasisTracker::basis_log`]).
    pub basis_log: Option<f64>,
}

/// A full fair-value computation — the model crate's rich per-window surface
/// for rendering and calibration journaling. The bus carries only the distilled
/// [`ModelSnapshot`] from [`FairValue::snapshot`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FairValue {
    /// Window this refers to.
    pub window: WindowId,
    /// Underlying asset.
    pub asset: Asset,
    /// When it was computed (`computed_at`).
    pub computed_at: TimestampMs,
    /// Seconds remaining until close (clamped at 0).
    pub tau_secs: f64,
    /// `p_up = Φ(z)`; `None` until strike, σ, and an anchor all exist.
    pub p_up: Option<f64>,
    /// Standardized log-moneyness (clamped); `None` with `p_up`.
    pub z: Option<f64>,
    /// One-second vol used (echoes the input).
    pub sigma_1s: Option<f64>,
    /// Vol over the remaining horizon, `σ_1s·√τ`.
    pub sigma_tau: Option<f64>,
    /// Strike capture diagnostics (both candidates, quality, freeze state).
    pub strike: StrikeEstimate,
    /// Which input the value anchored on; `None` when neither feed is usable.
    pub anchor: Option<AnchorSource>,
    /// The anchor price `S` used.
    pub anchor_price: Option<f64>,
    /// Latest live Chainlink price.
    pub chainlink_price: Option<f64>,
    /// Latest direct-Binance mid.
    pub fast_price: Option<f64>,
    /// Basis-corrected fast price (`fast · e^b`), when computable.
    pub corrected_fast: Option<f64>,
    /// Basis in bps (`(e^b − 1)·10⁴`), when the basis is known.
    pub basis_bps: Option<f64>,
    /// |corrected-fast vs Chainlink| divergence in bps, when both are fresh.
    pub divergence_bps: Option<f64>,
    /// Age of the latest Chainlink price at `computed_at`.
    pub chainlink_age: Option<DurationMs>,
    /// Age of the latest Binance mid at `computed_at`.
    pub fast_age: Option<DurationMs>,
    /// Health gate. **Caller-stamped:** [`compute`](FairValueEngine::compute)
    /// leaves it `Unreliable`/`Warming`; the driver runs the per-asset
    /// [`crate::HealthMonitor`] on [`health_inputs`](FairValue::health_inputs)
    /// and stamps the real tier here before reading [`snapshot`](FairValue::snapshot).
    /// The monitor is the single producer of health (hysteresis lives there).
    pub health: ModelHealth,
    /// Cause behind `health`. Caller-stamped alongside it.
    pub reason: ModelHealthReason,
}

impl FairValue {
    /// Distills the per-asset signals the [`crate::HealthMonitor`] consumes.
    /// Pure — `now` is this value's `computed_at`.
    #[must_use]
    pub fn health_inputs(&self) -> HealthInputs {
        HealthInputs {
            now: self.computed_at,
            has_probability: self.p_up.is_some(),
            anchor: self.anchor,
            chainlink_age: self.chainlink_age,
            divergence_bps: self.divergence_bps,
        }
    }
}

impl FairValue {
    /// Distills the bus-facing snapshot. `Some` only when a probability was
    /// produced (strike, σ, and an anchor all present) — a warming or anchorless
    /// model publishes nothing.
    #[must_use]
    pub fn snapshot(&self) -> Option<ModelSnapshot> {
        let p_up = self.p_up?;
        let z = self.z?;
        let sigma_1s = self.sigma_1s?;
        let sigma_tau = self.sigma_tau?;
        let anchor = self.anchor?;
        Some(ModelSnapshot {
            asset: self.asset,
            window: Some(self.window),
            p_up,
            z,
            sigma_1s,
            sigma_tau,
            basis: self.basis_bps.unwrap_or(0.0),
            anchor,
            health: self.health,
            reason: self.reason,
            input_ages: InputAges {
                chainlink: self.chainlink_age.unwrap_or(NEVER_SEEN_AGE),
                binance: self.fast_age.unwrap_or(NEVER_SEEN_AGE),
            },
            ts: self.computed_at,
        })
    }
}

/// The per-window fair-value engine. See the module docs.
#[derive(Debug, Clone)]
pub struct FairValueEngine {
    window: WindowId,
    asset: Asset,
    close_time: TimestampMs,
    strike: StrikeCapture,
    params: FairParams,
    /// Latest *live* Chainlink price and its local receive time.
    last_chainlink: Option<(f64, TimestampMs)>,
    /// Latest direct-Binance mid and its local receive time.
    last_fast: Option<(f64, TimestampMs)>,
}

impl FairValueEngine {
    /// Builds an engine around a pre-constructed strike capture.
    #[must_use]
    pub const fn new(
        window: WindowId,
        asset: Asset,
        close_time: TimestampMs,
        strike: StrikeCapture,
        params: FairParams,
    ) -> Self {
        Self {
            window,
            asset,
            close_time,
            strike,
            params,
            last_chainlink: None,
            last_fast: None,
        }
    }

    /// Builds an engine for a market, selecting the strike feed/rule from its
    /// resolution source.
    ///
    /// # Errors
    /// [`StrikeError::UnsupportedResolution`] when the market resolves on a
    /// feed the model cannot anchor on.
    pub fn for_market(market: &MarketInfo, params: FairParams) -> Result<Self, StrikeError> {
        let strike = StrikeCapture::for_market(market)?;
        Ok(Self::new(
            market.window,
            market.window.series.asset,
            market.close_time,
            strike,
            params,
        ))
    }

    /// Consumes one price tick: feeds the strike capture and refreshes the
    /// current-price state. Returns the strike capture's outcome (the
    /// salient telemetry — freezes and revisions).
    pub fn on_tick(&mut self, tick: &PriceTick) -> StrikeOutcome {
        let outcome = self.strike.on_tick(tick);
        if tick.asset == self.asset
            && let Some(value) = tick.value.to_f64()
            && value.is_finite()
            && value > 0.0
        {
            match (tick.source, tick.kind) {
                (PriceSource::ChainlinkRtds, TickKind::Vendor) => {
                    // Only live prints set the current S — a backfill print is
                    // an old price, not "now".
                    let lead = tick.ts_local.signed_duration_since(tick.ts_exchange);
                    if lead.as_millis() <= PRICE_LIVENESS_MS {
                        self.last_chainlink = Some((value, tick.ts_local));
                    }
                }
                (PriceSource::BinanceDirect, TickKind::Mid) => {
                    self.last_fast = Some((value, tick.ts_local));
                }
                _ => {}
            }
        }
        outcome
    }

    /// Recomputes the fair value at `inputs.now`.
    pub fn compute(&mut self, inputs: FairInputs) -> FairValue {
        let now = inputs.now;
        let tau = tau_secs(now, self.close_time);
        let strike_est = self.strike.estimate();
        let k = self.strike.strike();

        let chainlink_age = self
            .last_chainlink
            .map(|(_, tl)| now.signed_duration_since(tl));
        let fast_age = self.last_fast.map(|(_, tl)| now.signed_duration_since(tl));
        let chainlink_price = self.last_chainlink.map(|(v, _)| v);
        let fast_price = self.last_fast.map(|(v, _)| v);

        let chainlink_fresh = fresh(chainlink_age, self.params.chainlink_stale_ms);
        let fast_fresh = fresh(fast_age, self.params.fast_stale_ms);

        // Basis-corrected fast price: fast feed fresh AND basis warmed.
        let corrected_fast = match (fast_price, inputs.basis_log, fast_fresh) {
            (Some(f), Some(b), true) => Some(f * b.exp()),
            _ => None,
        };

        let (anchor, anchor_price) = if let Some(cf) = corrected_fast {
            (Some(AnchorSource::BinanceCorrected), Some(cf))
        } else if chainlink_fresh {
            (Some(AnchorSource::Chainlink), chainlink_price)
        } else {
            (None, None)
        };

        let basis_bps = inputs.basis_log.map(|b| (b.exp() - 1.0) * 10_000.0);

        // Divergence needs both the corrected fast price and a fresh Chainlink.
        // It is *measured* here; the per-asset health monitor (not this engine)
        // decides whether a breach is sustained enough to matter.
        let divergence_bps = match (corrected_fast, chainlink_price, chainlink_fresh) {
            (Some(cf), Some(cl), true) => Some((cf / cl).ln().abs() * 10_000.0),
            _ => None,
        };

        let sigma_tau = inputs.sigma_1s.map(|s| s * tau.sqrt());

        let (p_up, z) = match (k, inputs.sigma_1s, anchor_price, sigma_tau) {
            (Some(k), Some(_), Some(s), Some(st)) if k > 0.0 && s > 0.0 => {
                if tau <= 0.0 || st <= 0.0 {
                    // At/after close: the outcome is decided (≥-ties-Up, §6).
                    if s >= k {
                        (Some(1.0), Some(Z_CLAMP))
                    } else {
                        (Some(0.0), Some(-Z_CLAMP))
                    }
                } else {
                    let z = ((s / k).ln() / st).clamp(-Z_CLAMP, Z_CLAMP);
                    (Some(norm_cdf(z)), Some(z))
                }
            }
            _ => (None, None),
        };

        FairValue {
            window: self.window,
            asset: self.asset,
            computed_at: now,
            tau_secs: tau,
            p_up,
            z,
            sigma_1s: inputs.sigma_1s,
            sigma_tau,
            strike: strike_est,
            anchor,
            anchor_price,
            chainlink_price,
            fast_price,
            corrected_fast,
            basis_bps,
            divergence_bps,
            chainlink_age,
            fast_age,
            // Placeholder: the driver stamps the real tier from the per-asset
            // HealthMonitor (see `FairValue::health` / `health_inputs`).
            health: ModelHealth::Unreliable,
            reason: ModelHealthReason::Warming,
        }
    }

    /// The window this engine prices.
    #[must_use]
    pub const fn window(&self) -> WindowId {
        self.window
    }

    /// Read access to the strike capture (rendering / verification).
    #[must_use]
    pub const fn strike(&self) -> &StrikeCapture {
        &self.strike
    }

    /// The parameters in force.
    #[must_use]
    pub const fn params(&self) -> &FairParams {
        &self.params
    }
}

/// Seconds remaining until `close`, clamped at 0 (mirrors
/// `timeutil::tau_secs`; the model crate cannot depend on `timeutil`).
fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.signed_duration_since(now).as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

/// Whether `age` is within `limit_ms` (a future-stamped tick — negative age —
/// is fresh).
fn fresh(age: Option<DurationMs>, limit_ms: i64) -> bool {
    age.is_some_and(|a| a.as_millis() <= limit_ms)
}

#[cfg(test)]
mod tests {
    use core_types::{Series, WindowDuration};
    use rust_decimal::Decimal;
    use rust_decimal::prelude::FromPrimitive;

    use super::*;
    use crate::strike::StrikeParams;

    const OPEN: i64 = 1_781_290_500_000;
    const CLOSE: i64 = OPEN + 300_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN),
        }
    }

    fn engine() -> FairValueEngine {
        FairValueEngine::new(
            window(),
            Asset::Btc,
            TimestampMs::from_millis(CLOSE),
            StrikeCapture::new(
                window(),
                Asset::Btc,
                TimestampMs::from_millis(OPEN),
                StrikeParams::default(),
            ),
            FairParams::default(),
        )
    }

    fn chainlink(off: i64, price: f64) -> PriceTick {
        let ts = TimestampMs::from_millis(OPEN + off);
        PriceTick {
            source: PriceSource::ChainlinkRtds,
            asset: Asset::Btc,
            kind: TickKind::Vendor,
            value: Decimal::from_f64(price).unwrap(),
            ts_exchange: ts,
            ts_local: ts,
        }
    }

    fn binance(off: i64, price: f64) -> PriceTick {
        let ts = TimestampMs::from_millis(OPEN + off);
        PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Mid,
            value: Decimal::from_f64(price).unwrap(),
            ts_exchange: ts,
            ts_local: ts,
        }
    }

    fn inputs(now_off: i64, sigma: Option<f64>, basis: Option<f64>) -> FairInputs {
        FairInputs {
            now: TimestampMs::from_millis(OPEN + now_off),
            sigma_1s: sigma,
            basis_log: basis,
        }
    }

    /// Engine with the strike frozen at `k` (one live print exactly at open).
    fn engine_with_strike(k: f64) -> FairValueEngine {
        let mut e = engine();
        let out = e.on_tick(&chainlink(0, k));
        assert!(
            matches!(out, StrikeOutcome::Frozen { .. }),
            "strike must freeze"
        );
        e
    }

    fn close(got: f64, want: f64, rel: f64) {
        assert!(
            (got - want).abs() <= rel * want.abs().max(1e-12),
            "got {got}, want {want}"
        );
    }

    #[test]
    fn computes_p_up_from_hand_formula() {
        let mut e = engine_with_strike(100.0);
        // Current Chainlink price at mid-window (now = open + 150 s), fresh.
        e.on_tick(&chainlink(150_000, 100.5));
        let fair = e.compute(inputs(150_000, Some(0.001), None));

        let (s, k, sigma, tau) = (100.5_f64, 100.0_f64, 0.001_f64, 150.0_f64);
        let st = sigma * tau.sqrt();
        let z = (s / k).ln() / st;
        assert_eq!(fair.anchor, Some(AnchorSource::Chainlink));
        close(fair.z.unwrap(), z, 1e-9);
        close(fair.sigma_tau.unwrap(), st, 1e-12);
        close(fair.p_up.unwrap(), norm_cdf(z), 1e-12);
        // Health is caller-stamped; the engine exposes the inputs the monitor reads.
        let hi = fair.health_inputs();
        assert!(hi.has_probability);
        assert_eq!(hi.anchor, Some(AnchorSource::Chainlink));
        // The snapshot carries the same numbers.
        let snap = fair.snapshot().unwrap();
        close(snap.p_up, norm_cdf(z), 1e-12);
        assert_eq!(snap.window, Some(window()));
    }

    #[test]
    fn at_the_money_is_half() {
        let mut e = engine_with_strike(100.0);
        e.on_tick(&chainlink(150_000, 100.0));
        let fair = e.compute(inputs(150_000, Some(0.001), None));
        assert_eq!(fair.z, Some(0.0));
        close(fair.p_up.unwrap(), 0.5, 1e-15);
    }

    #[test]
    fn anchor_prefers_corrected_binance_when_fresh_and_basis_warmed() {
        let mut e = engine_with_strike(100.0);
        e.on_tick(&chainlink(150_000, 100.0));
        e.on_tick(&binance(150_000, 100.0));
        // basis_log = +0.001 → corrected fast = 100·e^0.001 ≈ 100.1.
        let fair = e.compute(inputs(150_000, Some(0.001), Some(0.001)));
        assert_eq!(fair.anchor, Some(AnchorSource::BinanceCorrected));
        close(fair.anchor_price.unwrap(), 100.0 * 0.001_f64.exp(), 1e-9);
        close(fair.basis_bps.unwrap(), (0.001_f64.exp() - 1.0) * 1e4, 1e-9);
    }

    #[test]
    fn anchor_falls_back_to_chainlink_when_fast_stale() {
        let mut e = engine_with_strike(100.0);
        e.on_tick(&chainlink(150_000, 100.2));
        e.on_tick(&binance(140_000, 100.0)); // 10 s old at now → stale
        let fair = e.compute(inputs(150_000, Some(0.001), Some(0.001)));
        assert_eq!(fair.anchor, Some(AnchorSource::Chainlink));
        close(fair.anchor_price.unwrap(), 100.2, 1e-12);
        // Fast-stale alone still prices (Chainlink anchor); the monitor treats a
        // *sustained* Chainlink-only anchor as DEGRADED (tested in `health`).
        assert!(fair.health_inputs().has_probability);
        assert_eq!(fair.health_inputs().anchor, Some(AnchorSource::Chainlink));
    }

    #[test]
    fn both_feeds_stale_yields_no_anchor() {
        let mut e = engine_with_strike(100.0);
        e.on_tick(&chainlink(140_000, 100.0)); // 10 s old → chainlink stale
        e.on_tick(&binance(140_000, 100.0));
        let fair = e.compute(inputs(150_000, Some(0.001), Some(0.001)));
        assert_eq!(fair.anchor, None);
        assert_eq!(fair.p_up, None);
        assert!(fair.health_inputs().anchor.is_none(), "no usable anchor");
        assert!(fair.snapshot().is_none(), "no probability → no snapshot");
    }

    #[test]
    fn no_probability_until_strike_and_sigma() {
        let mut e = engine();
        // No strike, no sigma yet.
        e.on_tick(&chainlink(150_000, 100.0)); // updates price but strike never froze
        let fair = e.compute(inputs(150_000, None, None));
        assert!(!fair.health_inputs().has_probability);
        assert_eq!(fair.p_up, None);
        assert!(fair.snapshot().is_none());

        // Strike present but sigma still None → still no probability.
        let mut e2 = engine_with_strike(100.0);
        e2.on_tick(&chainlink(150_000, 100.0));
        let fair2 = e2.compute(inputs(150_000, None, None));
        assert!(!fair2.health_inputs().has_probability);
    }

    #[test]
    fn measures_divergence_bps() {
        // The engine *measures* divergence (corrected-fast vs Chainlink); the
        // per-asset health monitor decides what a sustained breach means.
        let mut e = engine_with_strike(100.0);
        e.on_tick(&chainlink(150_000, 100.0));
        e.on_tick(&binance(150_000, 100.0));
        // basis_log 0.006 → corrected fast = 100·e^0.006, 60 bps above Chainlink.
        let fair = e.compute(inputs(150_000, Some(0.001), Some(0.006)));
        let d = fair.divergence_bps.unwrap();
        close(d, 60.0, 1e-6);
        assert!(d > 50.0);
        // Converged inputs (basis 0, fast == chainlink) → ~zero divergence.
        e.on_tick(&chainlink(151_000, 100.0));
        e.on_tick(&binance(151_000, 100.0));
        let fair = e.compute(inputs(151_000, Some(0.001), Some(0.0)));
        assert!(fair.divergence_bps.unwrap() < 1.0);
    }

    #[test]
    fn tau_zero_saturates_to_outcome() {
        let mut e = engine_with_strike(100.0);
        // At close, S above strike → resolves Up.
        e.on_tick(&chainlink(300_000, 100.4));
        let up = e.compute(inputs(300_000, Some(0.001), None));
        assert_eq!(up.tau_secs, 0.0);
        assert_eq!(up.p_up, Some(1.0));
        assert_eq!(up.z, Some(Z_CLAMP));
        // S below strike (and exactly at, which is Up) → Down only when strictly below.
        let mut e2 = engine_with_strike(100.0);
        e2.on_tick(&chainlink(300_000, 99.6));
        let down = e2.compute(inputs(300_000, Some(0.001), None));
        assert_eq!(down.p_up, Some(0.0));
        assert_eq!(down.z, Some(-Z_CLAMP));
    }

    #[test]
    fn extreme_ratio_clamps_z_and_keeps_snapshot_finite() {
        let mut e = engine_with_strike(100.0);
        // Far in the money with a tiny remaining horizon: z would explode.
        e.on_tick(&chainlink(299_900, 130.0));
        let fair = e.compute(inputs(299_900, Some(0.001), None)); // tau = 0.1 s
        assert_eq!(fair.z, Some(Z_CLAMP));
        assert_eq!(fair.p_up, Some(1.0));
        let snap = fair.snapshot().unwrap();
        // Finite everywhere → serializes (ModelSnapshot's JSON round-trip is
        // pinned in core-types; here we only guard against inf/NaN leaking in).
        assert!(snap.z.is_finite() && snap.p_up.is_finite() && snap.sigma_tau.is_finite());
    }
}
