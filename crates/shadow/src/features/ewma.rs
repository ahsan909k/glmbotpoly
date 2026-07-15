//! Pandas-parity incremental EWMA kernels — the two feature EWMAs that are NOT
//! the engine's `sigma_1s` (which is reused from `model::VolEstimator`):
//!
//! - [`EwmMean`] — `basis_ewma`: `ewm(halflife=120, min_periods=30,
//!   ignore_na=True).mean()` over the per-second log basis `ln(chainlink/mid)`.
//! - [`EwmStd`]  — `realized_vol`: `ewm(halflife=60, min_periods=10).std()`
//!   (pandas defaults `adjust=True, bias=False, ignore_na=False`) over the
//!   per-second log return `ln(mid).diff()`.
//!
//! These reproduce pandas' online recurrences exactly (the `.std()` path mirrors
//! `pandas._libs.window.aggregations.ewmcov` with `bias=False`), pinned to 1e-9
//! by golden vectors generated from pandas. This is the #1 feature-parity hazard
//! (Decisions Log): isolating it here lets the nightly guard catch any drift.
//!
//! Both take a halflife and derive the per-step decay `d = 2^(-1/halflife)`
//! (equivalently pandas' `1 - alpha` with `alpha = 1 - exp(-ln2/halflife)`).

/// The per-step decay from a halflife: `d = 2^(-1/halflife)`.
#[inline]
fn decay_from_halflife(halflife: f64) -> f64 {
    2.0_f64.powf(-1.0 / halflife)
}

/// `ewm(..., adjust=True, ignore_na=True).mean()` — the `basis_ewma` kernel.
///
/// Adjusted mean `y = (Σ dⁱ xᵢ) / (Σ dⁱ)` via the running `num/den` recurrence;
/// NaN inputs are ignored entirely (they neither fold nor advance the decay).
/// Reports `None` until `min_periods` non-NaN observations have folded.
#[derive(Debug, Clone)]
pub struct EwmMean {
    decay: f64,
    min_periods: u32,
    num: f64,
    den: f64,
    nobs: u32,
}

impl EwmMean {
    /// Builds a mean EWMA with the given halflife and warmup.
    #[must_use]
    pub fn new(halflife: f64, min_periods: u32) -> Self {
        Self {
            decay: decay_from_halflife(halflife),
            min_periods,
            num: 0.0,
            den: 0.0,
            nobs: 0,
        }
    }

    /// Folds one observation. `None` (or a non-finite value) is a NaN input and
    /// is ignored (`ignore_na=True`).
    pub fn fold(&mut self, x: Option<f64>) {
        let Some(v) = x.filter(|v| v.is_finite()) else {
            return;
        };
        self.num = v + self.decay * self.num;
        self.den = 1.0 + self.decay * self.den;
        self.nobs = self.nobs.saturating_add(1);
    }

    /// The current mean, or `None` before `min_periods` observations.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        if self.nobs >= self.min_periods && self.den > 0.0 {
            Some(self.num / self.den)
        } else {
            None
        }
    }

    /// Number of non-NaN observations folded so far.
    #[must_use]
    pub fn nobs(&self) -> u32 {
        self.nobs
    }
}

/// `ewm(..., adjust=True, bias=False, ignore_na=False).std()` — the
/// `realized_vol` kernel.
///
/// Reproduces `pandas` `ewmcov(x, x, bias=False)` variance online, then reports
/// its square root. `ignore_na=False`: a NaN input still decays the accumulated
/// weights (the standard pandas default). Reports `None` (→ NaN feature) before
/// `min_periods` observations or when the debias denominator is non-positive.
#[derive(Debug, Clone)]
pub struct EwmStd {
    decay: f64,
    min_periods: u32,
    /// Weighted running mean of the observations.
    mean: f64,
    /// Weighted running variance (the `cov(x, x)`).
    cov: f64,
    sum_wt: f64,
    sum_wt2: f64,
    old_wt: f64,
    nobs: u32,
    /// Whether the first observation has been seen (pandas' `mean == mean`).
    has_obs: bool,
}

impl EwmStd {
    /// The adjusted new-observation weight (pandas `adjust=True`).
    const NEW_WT: f64 = 1.0;

    /// Builds a std EWMA with the given halflife and warmup.
    #[must_use]
    pub fn new(halflife: f64, min_periods: u32) -> Self {
        Self {
            decay: decay_from_halflife(halflife),
            min_periods,
            mean: 0.0,
            cov: 0.0,
            sum_wt: 1.0,
            sum_wt2: 1.0,
            old_wt: 1.0,
            nobs: 0,
            has_obs: false,
        }
    }

    /// Folds one observation. `None` (or a non-finite value) is a NaN input:
    /// with `ignore_na=False` it still decays the weights once a first
    /// observation exists.
    pub fn fold(&mut self, x: Option<f64>) {
        let is_obs = matches!(x, Some(v) if v.is_finite());
        if self.has_obs {
            // ignore_na=False → the decay advances on every element.
            self.sum_wt *= self.decay;
            self.sum_wt2 *= self.decay * self.decay;
            self.old_wt *= self.decay;
            if let Some(cur) = x.filter(|v| v.is_finite()) {
                let old_mean = self.mean;
                if self.mean != cur {
                    self.mean = (self.old_wt * old_mean + Self::NEW_WT * cur)
                        / (self.old_wt + Self::NEW_WT);
                }
                self.cov = (self.old_wt
                    * (self.cov + (old_mean - self.mean) * (old_mean - self.mean))
                    + Self::NEW_WT * (cur - self.mean) * (cur - self.mean))
                    / (self.old_wt + Self::NEW_WT);
                self.sum_wt += Self::NEW_WT;
                self.sum_wt2 += Self::NEW_WT * Self::NEW_WT;
                self.old_wt += Self::NEW_WT;
                self.nobs = self.nobs.saturating_add(1);
            }
        } else if is_obs {
            // The first observation seeds the accumulators.
            let cur = x.unwrap_or(f64::NAN);
            self.mean = cur;
            self.cov = 0.0;
            self.sum_wt = 1.0;
            self.sum_wt2 = 1.0;
            self.old_wt = 1.0;
            self.nobs = 1;
            self.has_obs = true;
        }
    }

    /// The current EWMA standard deviation, or `None` before `min_periods`
    /// observations / when the debias denominator is non-positive.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        if self.nobs < self.min_periods {
            return None;
        }
        let numerator = self.sum_wt * self.sum_wt;
        let denominator = numerator - self.sum_wt2;
        if denominator <= 0.0 {
            return None;
        }
        let var = (numerator / denominator) * self.cov;
        if var.is_finite() && var >= 0.0 {
            Some(var.sqrt())
        } else {
            // A tiny negative from float error (or NaN) → treat as missing,
            // matching pandas' NaN std there.
            None
        }
    }

    /// Number of observations folded so far.
    #[must_use]
    pub fn nobs(&self) -> u32 {
        self.nobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors generated from pandas — the exact
    // `pd.Series(x).ewm(...).mean()/.std()` outputs for the input series below.
    // Parity to 1e-9 is the contract; regenerate from pandas if the kernels change.
    const HL_MEAN: f64 = 120.0;
    const HL_STD: f64 = 60.0;

    fn approx(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => (x - y).abs() <= tol,
            _ => false,
        }
    }

    #[test]
    fn mean_matches_pandas_golden() {
        // pd.Series([nan,1,2,nan,3,2.5,2,4])
        //   .ewm(halflife=120, min_periods=3, ignore_na=True).mean()
        let xs: [Option<f64>; 8] = [
            None,
            Some(1.0),
            Some(2.0),
            None,
            Some(3.0),
            Some(2.5),
            Some(2.0),
            Some(4.0),
        ];
        let want: [Option<f64>; 8] = [
            None,
            None,
            None,
            None,
            Some(2.003_850_796_256_31),
            Some(2.128_964_858_756_541_7),
            Some(2.102_873_057_665_522_3),
            Some(2.423_644_331_203_583_6),
        ];
        let mut m = EwmMean::new(HL_MEAN, 3);
        for (i, &x) in xs.iter().enumerate() {
            m.fold(x);
            assert!(
                approx(m.value(), want[i], 1e-9),
                "mean[{i}]: got {:?} want {:?}",
                m.value(),
                want[i]
            );
        }
    }

    #[test]
    fn std_matches_pandas_golden() {
        // pd.Series([nan,0.01,-0.02,0.015,-0.005,0.02,-0.01,0.0,0.03,-0.02])
        //   .ewm(halflife=60, min_periods=3).std()   (adjust=True, bias=False, ignore_na=False)
        let xs: [Option<f64>; 10] = [
            None,
            Some(0.01),
            Some(-0.02),
            Some(0.015),
            Some(-0.005),
            Some(0.02),
            Some(-0.01),
            Some(0.0),
            Some(0.03),
            Some(-0.02),
        ];
        let want: [Option<f64>; 10] = [
            None,
            None,
            None,
            Some(0.018_946_418_380_999_433),
            Some(0.015_787_094_193_704_197),
            Some(0.016_350_703_899_457_846),
            Some(0.015_698_692_470_806_727),
            Some(0.014_303_390_027_694_379),
            Some(0.016_733_355_983_094_216),
            Some(0.017_794_820_222_661_902),
        ];
        let mut s = EwmStd::new(HL_STD, 3);
        for (i, &x) in xs.iter().enumerate() {
            s.fold(x);
            assert!(
                approx(s.value(), want[i], 1e-9),
                "std[{i}]: got {:?} want {:?}",
                s.value(),
                want[i]
            );
        }
    }
}
