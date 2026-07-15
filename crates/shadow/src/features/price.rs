//! Per-asset price/vol/basis state → the 10 BASE features.
//!
//! Maintains, per asset, the per-second Binance-mid bar chain and three
//! continuous EWMAs advanced once per completed bar:
//! - `sigma_1s` via a reused [`model::VolEstimator`] (parity by construction with
//!   the engine and the lab's `sigma_1s_from_bars`),
//! - `realized_vol` via [`EwmStd`] over the consecutive-bar log return,
//! - `basis_ewma` via [`EwmMean`] over the per-bar log basis `ln(chainlink/mid)`.
//!
//! At a sample instant the features are read **as of the last completed bar**
//! (all EWMAs fold on bar completion, so they are mutually consistent). This
//! lags the offline reconstruction (whose as-of bar is the current second's
//! close, known only once that second ends) by up to ~1 second — an inherent
//! streaming-vs-batch gap that the nightly LIVE parity guard measures and
//! tolerates. `z`/`p_up_model`/`log_s_k`/`basis_bps` reproduce
//! `dataset._derive_asof_features` exactly (`z` is basis-corrected).

use core_types::{Asset, PriceTick, TimestampMs};
use model::{VolEstimator, VolParams, norm_cdf};
use rust_decimal::prelude::ToPrimitive;

use super::ewma::{EwmMean, EwmStd};

/// `realized_vol` EWMA half-life / warmup (`ewm(halflife=60, min_periods=10)`).
const REALIZED_VOL_HL: f64 = 60.0;
const REALIZED_VOL_MINP: u32 = 10;
/// `basis_ewma` EWMA half-life / warmup (`ewm(halflife=120, min_periods=30)`).
const BASIS_HL: f64 = 120.0;
const BASIS_MINP: u32 = 30;
/// `z` clamp bound (`lm.Z_CLAMP`).
const Z_CLAMP: f64 = 39.0;
/// Longest a tick's exchange timestamp may lead its local receive time (mirrors
/// `model::VolEstimator`), so shadow's bar chain matches the estimator's.
const MAX_EXCHANGE_LEAD_MS: i64 = 10_000;
/// Recent completed-bar mids retained for the PM-staleness lookback + as-of.
const MID_RING_CAP: usize = 16;
/// Number of BASE features.
pub const N_BASE_FEATURES: usize = 10;

/// The most recently completed one-second bar's derived state.
#[derive(Debug, Clone, Copy)]
struct BarSnapshot {
    sec: i64,
    mid: f64,
    /// `ln(mid / prev_bar_mid)` — `None` for the first bar (the `diff()` NaN).
    ret: Option<f64>,
    /// Chainlink LOCF value as of this bar (`None` before the first Chainlink).
    chainlink: Option<f64>,
}

/// Per-asset price feature state (persists across window rollovers).
#[derive(Debug, Clone)]
pub struct PriceState {
    vol: VolEstimator,
    realized_vol: EwmStd,
    basis_ewma: EwmMean,
    /// Current forming bar: `(second, latest_mid)`.
    open: Option<(i64, f64)>,
    /// Close of the previously completed bar (for the next `ret`).
    prev_bar_mid: Option<f64>,
    /// Chainlink last-observed value (LOCF).
    chainlink: Option<f64>,
    /// The last completed bar's snapshot.
    last_bar: Option<BarSnapshot>,
    /// Recent completed-bar `(second, mid)`, newest last.
    mid_ring: std::collections::VecDeque<(i64, f64)>,
}

/// The 10 BASE features plus diagnostics for the journaled prediction.
#[derive(Debug, Clone, Copy)]
pub struct PriceSample {
    /// `[ret, realized_vol, sigma_1s, log_s_k, z, p_up_model, tau_secs, elapsed_secs, basis_bps, basis_ewma]`.
    pub features: [f64; N_BASE_FEATURES],
    /// Second of the as-of bar (`None` if no bar yet).
    pub asof_sec: Option<i64>,
    /// `sigma_1s` folded-return count (a warmup/parity diagnostic).
    pub sigma_returns: u64,
    /// `basis_ewma` folded-sample count.
    pub basis_samples: u32,
}

impl PriceState {
    /// Builds a fresh per-asset state for `asset`.
    ///
    /// # Errors
    /// [`model::VolError`] if the (compile-time-constant, always-valid) default
    /// vol parameters somehow fail validation — unreachable in practice, but
    /// propagated so the runtime path never `unwrap`s.
    pub fn new(asset: Asset) -> Result<Self, model::VolError> {
        Ok(Self {
            vol: VolEstimator::new(asset, VolParams::default())?,
            realized_vol: EwmStd::new(REALIZED_VOL_HL, REALIZED_VOL_MINP),
            basis_ewma: EwmMean::new(BASIS_HL, BASIS_MINP),
            open: None,
            prev_bar_mid: None,
            chainlink: None,
            last_bar: None,
            mid_ring: std::collections::VecDeque::with_capacity(MID_RING_CAP),
        })
    }

    /// Folds one Binance-direct `Mid` tick (the fast-signal feed).
    pub fn on_mid_tick(&mut self, tick: &PriceTick) {
        let Some(value) = valid_price(tick) else {
            return;
        };
        let second = tick.ts_exchange.as_millis().div_euclid(1_000);
        self.advance_bars(second, value);
        // The estimator does its own (identical) bar keying for sigma_1s.
        let _ = self.vol.on_tick(tick);
    }

    /// Folds one Chainlink tick (updates the LOCF basis anchor).
    pub fn on_chainlink_tick(&mut self, tick: &PriceTick) {
        if let Some(value) = valid_price(tick) {
            self.chainlink = Some(value);
        }
    }

    /// Advances the bar chain, completing the open bar when a later second
    /// arrives.
    fn advance_bars(&mut self, second: i64, mid: f64) {
        match self.open {
            None => self.open = Some((second, mid)),
            Some((os, _)) if second < os => {} // out of order — ignore
            Some((os, _)) if second == os => self.open = Some((os, mid)),
            Some((os, om)) => {
                self.complete_bar(os, om);
                self.open = Some((second, mid));
            }
        }
    }

    /// Folds a just-completed bar into the EWMAs and records its snapshot.
    fn complete_bar(&mut self, sec: i64, mid: f64) {
        let ret = self.prev_bar_mid.map(|p| (mid / p).ln());
        self.realized_vol.fold(ret);
        let basis = self.chainlink.map(|c| (c / mid).ln());
        self.basis_ewma.fold(basis);
        self.last_bar = Some(BarSnapshot {
            sec,
            mid,
            ret,
            chainlink: self.chainlink,
        });
        if self.mid_ring.len() == MID_RING_CAP {
            self.mid_ring.pop_front();
        }
        self.mid_ring.push_back((sec, mid));
        self.prev_bar_mid = Some(mid);
    }

    /// The last completed-bar mid at or before `sec` (the `bser` LOCF lookup for
    /// PM-staleness). `None` if the ring holds nothing at or before `sec`.
    #[must_use]
    pub fn mid_asof(&self, sec: i64) -> Option<f64> {
        self.mid_ring
            .iter()
            .rev()
            .find(|(s, _)| *s <= sec)
            .map(|(_, m)| *m)
    }

    /// Samples the 10 BASE features as of the last completed bar. `strike` is the
    /// window's price-to-beat; `tau_secs`/`elapsed_secs` are known at sample time.
    /// When the as-of bar is older than `max_staleness_ms` the price features are
    /// nulled (NaN) — strictly more conservative, matching the lab's as-of null.
    #[must_use]
    pub fn sample(
        &self,
        strike: Option<f64>,
        tau_secs: f64,
        elapsed_secs: f64,
        sample_ts: TimestampMs,
        max_staleness_ms: i64,
    ) -> PriceSample {
        let sigma = self.vol.sigma();
        let mut features = [f64::NAN; N_BASE_FEATURES];
        // tau/elapsed are always known (they do not depend on a feed bar).
        features[6] = tau_secs;
        features[7] = elapsed_secs;

        let Some(bar) = self.last_bar else {
            return PriceSample {
                features,
                asof_sec: None,
                sigma_returns: self.vol.estimate().returns_observed,
                basis_samples: self.basis_ewma.nobs(),
            };
        };
        let stale = sample_ts.as_millis() - bar.sec * 1_000 > max_staleness_ms;
        if stale {
            return PriceSample {
                features,
                asof_sec: Some(bar.sec),
                sigma_returns: self.vol.estimate().returns_observed,
                basis_samples: self.basis_ewma.nobs(),
            };
        }

        let basis_ewma = self.basis_ewma.value();
        features[0] = bar.ret.unwrap_or(f64::NAN);
        features[1] = self.realized_vol.value().unwrap_or(f64::NAN);
        features[2] = sigma.unwrap_or(f64::NAN);
        let log_s_k = match strike {
            Some(k) if k > 0.0 && bar.mid > 0.0 => (bar.mid / k).ln(),
            _ => f64::NAN,
        };
        features[3] = log_s_k;
        // z = clip((log_s_k + basis_ewma.fillna(0)) / (sigma_1s·√tau), ±39).
        let basis_filled = basis_ewma.unwrap_or(0.0);
        let z = match sigma {
            Some(s) if tau_secs > 0.0 => {
                let sigma_tau = s * tau_secs.sqrt();
                ((log_s_k + basis_filled) / sigma_tau).clamp(-Z_CLAMP, Z_CLAMP)
            }
            _ => f64::NAN,
        };
        features[4] = z;
        features[5] = norm_cdf(z);
        features[8] = match bar.chainlink {
            Some(c) if bar.mid > 0.0 => 1e4 * (c / bar.mid - 1.0),
            _ => f64::NAN,
        };
        features[9] = basis_ewma.unwrap_or(f64::NAN);

        PriceSample {
            features,
            asof_sec: Some(bar.sec),
            sigma_returns: self.vol.estimate().returns_observed,
            basis_samples: self.basis_ewma.nobs(),
        }
    }
}

/// Extracts a finite, positive `f64` price from a tick, applying the estimator's
/// exchange-lead guard so the bar chain matches [`model::VolEstimator`].
fn valid_price(tick: &PriceTick) -> Option<f64> {
    let value = tick.value.to_f64()?;
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let lead = tick.ts_exchange.signed_duration_since(tick.ts_local);
    if lead.as_millis() > MAX_EXCHANGE_LEAD_MS {
        return None;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use core_types::{PriceSource, TickKind};
    use rust_decimal::Decimal;
    use rust_decimal::prelude::FromPrimitive;

    use super::*;

    const T0: i64 = 1_700_000_000_000;

    fn mid_tick(sec: i64, price: f64) -> PriceTick {
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

    #[test]
    fn tau_elapsed_always_present_before_warmup() {
        let st = PriceState::new(Asset::Btc).unwrap();
        let s = st.sample(
            Some(60_000.0),
            120.0,
            180.0,
            TimestampMs::from_millis(T0),
            120_000,
        );
        assert_eq!(s.features[6], 120.0);
        assert_eq!(s.features[7], 180.0);
        assert!(s.features[2].is_nan(), "sigma NaN before warmup");
        assert!(s.asof_sec.is_none());
    }

    #[test]
    fn last_completed_bar_drives_log_s_k_and_p_up() {
        let mut st = PriceState::new(Asset::Btc).unwrap();
        // Feed 3 bars; the 3rd tick completes bar 1 (close 60_100).
        st.on_mid_tick(&mid_tick(0, 60_000.0));
        st.on_mid_tick(&mid_tick(1, 60_100.0));
        st.on_mid_tick(&mid_tick(2, 60_050.0)); // completes bar 1 (60_100)
        let sample_ts = TimestampMs::from_millis(T0 + 2_000);
        let s = st.sample(Some(60_000.0), 120.0, 60.0, sample_ts, 120_000);
        // as-of bar = bar 1 (close 60_100). log_s_k = ln(60_100/60_000).
        assert_eq!(s.asof_sec, Some(T0 / 1_000 + 1));
        assert!((s.features[3] - (60_100.0_f64 / 60_000.0).ln()).abs() < 1e-12);
        // sigma still warming (< 60 folds) → z/p_up NaN.
        assert!(s.features[4].is_nan() && s.features[5].is_nan());
    }

    #[test]
    fn stale_bar_nulls_price_features_but_keeps_tau() {
        let mut st = PriceState::new(Asset::Btc).unwrap();
        st.on_mid_tick(&mid_tick(0, 60_000.0));
        st.on_mid_tick(&mid_tick(1, 60_100.0));
        st.on_mid_tick(&mid_tick(2, 60_050.0));
        // Sample 200 s after the last bar → stale (> 120 s).
        let sample_ts = TimestampMs::from_millis(T0 + 202_000);
        let s = st.sample(Some(60_000.0), 60.0, 60.0, sample_ts, 120_000);
        assert!(s.features[3].is_nan(), "log_s_k nulled when stale");
        assert_eq!(s.features[6], 60.0, "tau kept");
    }

    #[test]
    fn mid_asof_is_locf() {
        let mut st = PriceState::new(Asset::Btc).unwrap();
        st.on_mid_tick(&mid_tick(0, 60_000.0));
        st.on_mid_tick(&mid_tick(1, 60_100.0));
        st.on_mid_tick(&mid_tick(3, 60_050.0)); // completes bar 1 at sec 1
        // Only bars 0 and 1 completed (close 60_000, then 60_100).
        let sec0 = T0 / 1_000;
        assert_eq!(st.mid_asof(sec0 + 1), Some(60_100.0));
        assert_eq!(st.mid_asof(sec0 + 2), Some(60_100.0), "LOCF across the gap");
        assert_eq!(st.mid_asof(sec0), Some(60_000.0));
    }
}
