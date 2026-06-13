//! The standard normal CDF Φ and its building blocks `erf`/`erfc` — §8's
//! `p_up = Φ(z)`.
//!
//! Hand-rolled per CLAUDE.md §3 (no new dependency for one function; the
//! SNTP port in `timeutil` is the precedent). The implementation is W. J.
//! Cody's rational-Chebyshev `CALERF` (1969), the three-interval scheme used
//! by most libm `erf`/`erfc`: a rational approximation of `erf` on
//! `|x| ≤ 0.46875`, of `erfc` on `0.46875 < |x| ≤ 4`, and an asymptotic
//! rational on `|x| > 4`. It preserves *relative* accuracy (~1e-16) deep into
//! the tail — exactly what late-window `p_up` near 0 or 1 needs, where an
//! absolute-only method would collapse to zero.
//!
//! `Φ(z) = ½·erfc(−z/√2)`: routing the CDF through `erfc` (not `½(1+erf)`)
//! keeps the far-negative tail accurate, since `erfc` of a large positive
//! argument is computed directly rather than as `1 − (almost 1)`.

// Faithful port of Cody's published double-precision coefficients; several
// carry more decimal digits than an f64 mantissa resolves. Trimming them to
// shortest round-trip form would be gratuitous divergence from the reference.
#![allow(clippy::excessive_precision)]

use std::f64::consts::FRAC_1_SQRT_2;

/// Coefficients for the `erf` approximation on `|x| ≤ 0.46875`.
const A: [f64; 5] = [
    3.16112374387056560e00,
    1.13864154151050156e02,
    3.77485237685302021e02,
    3.20937758913846947e03,
    1.85777706184603153e-1,
];
const B: [f64; 4] = [
    2.36012909523441209e01,
    2.44024637934444173e02,
    1.28261652607737228e03,
    2.84423683343917062e03,
];

/// Coefficients for the `erfc` approximation on `0.46875 < |x| ≤ 4`.
const C: [f64; 9] = [
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
    2.15311535474403846e-8,
];
const D: [f64; 8] = [
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
];

/// Coefficients for the asymptotic `erfc` approximation on `|x| > 4`.
const P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
const Q: [f64; 5] = [
    2.56852019228982242e00,
    1.87295284992346047e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];

/// `1/√π`, the leading constant of the asymptotic series.
const SQRPI: f64 = 5.6418958354775628695e-1;
/// First-interval cutoff.
const THRESH: f64 = 0.46875;
/// Argument-reduction grid for the `exp(−x²)` split (1/16 steps).
const SIXTEN: f64 = 16.0;
/// Below this `|x|`, `x²` underflows to zero in the first interval.
const XSMALL: f64 = 1.11e-16;
/// At or above this `|x|`, `erfc(x)` underflows to 0 in `f64`.
const XBIG: f64 = 26.543;

/// The standard normal cumulative distribution function Φ.
///
/// Finite for every finite input and never `NaN`: saturates to exactly `0.0`
/// far in the left tail and `1.0` far in the right tail. `norm_cdf(0.0)` is
/// exactly `0.5`.
#[must_use]
pub fn norm_cdf(z: f64) -> f64 {
    // Φ(z) = ½·erfc(−z/√2). FRAC_1_SQRT_2 is the exactly-rounded 1/√2.
    0.5 * erfc(-z * FRAC_1_SQRT_2)
}

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt`.
#[must_use]
pub fn erf(x: f64) -> f64 {
    calerf(x, true)
}

/// The complementary error function `erfc(x) = 1 − erf(x)`, accurate deep into
/// the tail where `1 − erf(x)` would lose all significance.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    calerf(x, false)
}

/// Cody's `CALERF`: computes `erf` (`want_erf = true`) or `erfc`
/// (`want_erf = false`) over the three intervals. Allocation- and panic-free.
fn calerf(x: f64, want_erf: bool) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();

    // First interval — direct rational for erf on |x| ≤ 0.46875.
    if y <= THRESH {
        let ysq = if y > XSMALL { y * y } else { 0.0 };
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        let erf_val = x * (xnum + A[3]) / (xden + B[3]);
        return if want_erf { erf_val } else { 1.0 - erf_val };
    }

    // Intervals 2 and 3 compute erfc(|x|); the fixup below restores sign and
    // the requested function.
    let erfc_abs = if y <= 4.0 {
        // Second interval — rational for erfc on 0.46875 < |x| ≤ 4.
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        let result = (xnum + C[7]) / (xden + D[7]);
        scaled_exp(y, result)
    } else if y >= XBIG {
        // Tail underflow: erfc(|x|) is 0 to f64 precision.
        0.0
    } else {
        // Third interval — asymptotic rational for erfc on 4 < |x| < XBIG.
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        let mut result = ysq * (xnum + P[4]) / (xden + Q[4]);
        result = (SQRPI - result) / y;
        scaled_exp(y, result)
    };

    // Fixup: `erfc_abs` is erfc(|x|).
    if want_erf {
        // erf(x) = 1 − erfc(|x|), odd in x.
        let erf_abs = (0.5 - erfc_abs) + 0.5;
        if x < 0.0 { -erf_abs } else { erf_abs }
    } else if x < 0.0 {
        // erfc(−|x|) = 2 − erfc(|x|).
        2.0 - erfc_abs
    } else {
        erfc_abs
    }
}

/// Applies the `exp(−y²)` factor split across a 1/16 grid for accuracy:
/// `exp(−ysq²)·exp(−Δ)` with `Δ = (y−ysq)(y+ysq)` avoids cancellation in the
/// exponent (Cody's technique).
fn scaled_exp(y: f64, result: f64) -> f64 {
    let ysq = (y * SIXTEN).trunc() / SIXTEN;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp() * result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts `got` matches `want` to a tight relative tolerance (absolute
    /// near zero).
    fn close(got: f64, want: f64, rel: f64) {
        let tol = rel * want.abs().max(1e-300);
        assert!(
            (got - want).abs() <= tol,
            "got {got:.17e}, want {want:.17e} (|Δ| {:.3e} > tol {tol:.3e})",
            (got - want).abs()
        );
    }

    #[test]
    fn norm_cdf_matches_reference_table() {
        // Reference Φ(z) values (high-precision). Relative tol 1e-12 proves
        // Cody accuracy while tolerating last-digit noise in the literals.
        let table = [
            (0.1_f64, 0.539_827_837_277_028_96),
            (0.5, 0.691_462_461_274_013_1),
            (1.0, 0.841_344_746_068_542_9),
            (1.96, 0.975_002_104_851_779_5),
            (2.0, 0.977_249_868_051_820_8),
            (3.0, 0.998_650_101_968_369_9),
            (-1.0, 0.158_655_253_931_457_05),
            (-2.0, 0.022_750_131_948_179_195),
            (-3.0, 0.001_349_898_031_630_093_2),
        ];
        for (z, want) in table {
            close(norm_cdf(z), want, 1e-12);
        }
    }

    #[test]
    fn norm_cdf_zero_is_exactly_half() {
        assert_eq!(norm_cdf(0.0), 0.5);
    }

    #[test]
    fn norm_cdf_is_accurate_in_the_far_tail() {
        // Relative accuracy where an absolute method would read zero.
        close(norm_cdf(-6.0), 9.865_876_450_376_946e-10, 1e-10);
        close(norm_cdf(-10.0), 7.619_853_024_160_525e-24, 1e-9);
    }

    #[test]
    fn erf_matches_reference_values() {
        assert_eq!(erf(0.0), 0.0);
        close(erf(0.5), 0.520_499_877_813_046_5, 1e-13);
        close(erf(1.0), 0.842_700_792_949_714_9, 1e-13);
        close(erf(2.0), 0.995_322_265_018_952_7, 1e-13);
        close(erf(-1.0), -0.842_700_792_949_714_9, 1e-13);
    }

    #[test]
    fn erf_and_erfc_are_complementary() {
        let mut z = -6.0;
        while z <= 6.0 {
            let sum = erf(z) + erfc(z);
            assert!((sum - 1.0).abs() < 1e-14, "erf+erfc at {z} = {sum}");
            z += 0.013;
        }
    }

    #[test]
    fn norm_cdf_is_symmetric() {
        // Φ(z) + Φ(−z) = 1 to f64 precision across the body.
        let mut z = 0.0;
        while z <= 8.0 {
            let sum = norm_cdf(z) + norm_cdf(-z);
            assert!((sum - 1.0).abs() < 1e-15, "Φ(±{z}) sum = {sum}");
            z += 0.017;
        }
    }

    #[test]
    fn norm_cdf_is_monotone_nondecreasing() {
        let mut prev = norm_cdf(-12.0);
        let mut z = -12.0;
        while z <= 12.0 {
            let cur = norm_cdf(z);
            assert!(cur >= prev, "Φ decreased at {z}: {prev} → {cur}");
            assert!((0.0..=1.0).contains(&cur), "Φ({z}) = {cur} out of [0,1]");
            prev = cur;
            z += 0.01;
        }
    }

    #[test]
    fn extremes_saturate_without_nan() {
        assert_eq!(norm_cdf(40.0), 1.0);
        assert_eq!(norm_cdf(-40.0), 0.0);
        assert_eq!(norm_cdf(f64::INFINITY), 1.0);
        assert_eq!(norm_cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert!(norm_cdf(f64::NAN).is_nan());
    }
}
