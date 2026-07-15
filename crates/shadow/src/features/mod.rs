//! The 24-feature vector the champion `dir10_full` model consumes.
//!
//! Feature order is the model's positional contract (`FULL_FEATURES` — the
//! `model.txt` uses default `Column_i` names, so `Column_i` == `FULL_FEATURES[i]`
//! by position). NaN is preserved (LightGBM native-missing). The blocks are the
//! 10 BASE ([`price`]), 8 DEPTH ([`depth`]), 6 PM ([`pm`]) features.

pub mod depth;
pub mod ewma;
pub mod pm;
pub mod price;

use depth::N_DEPTH_FEATURES;
use pm::{N_PM_FEATURES, PmSample};
use price::{N_BASE_FEATURES, PriceSample};

/// Total feature count.
pub const N_FEATURES: usize = N_BASE_FEATURES + N_DEPTH_FEATURES + N_PM_FEATURES;

/// The 24 feature names in model order (= `learn_walkforward.FULL_FEATURES`).
/// The nightly parity guard and the export meta pin against this list.
pub const FULL_FEATURES: [&str; N_FEATURES] = [
    // BASE (10)
    "ret",
    "realized_vol",
    "sigma_1s",
    "log_s_k",
    "z",
    "p_up_model",
    "tau_secs",
    "elapsed_secs",
    "basis_bps",
    "basis_ewma",
    // DEPTH (8)
    "depth_imb_1",
    "depth_imb_5",
    "depth_imb_10",
    "depth_imb_20",
    "microprice_gap",
    "bid_depth_slope",
    "ask_depth_slope",
    "depth_spread_bps",
    // PM (6)
    "pm_mid",
    "pm_spread",
    "pm_book_imb",
    "pm_staleness_1s",
    "pm_staleness_2s",
    "pm_staleness_3s",
];

/// One assembled feature vector (`f32`, NaN = native-missing), model order.
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector {
    /// The 24 feature values, in [`FULL_FEATURES`] order.
    pub values: [f32; N_FEATURES],
}

impl FeatureVector {
    /// Assembles the 24-vector from the three feature blocks.
    #[must_use]
    pub fn assemble(base: &PriceSample, depth: &[f64; N_DEPTH_FEATURES], pm: &PmSample) -> Self {
        let mut values = [f32::NAN; N_FEATURES];
        for (dst, src) in values[..N_BASE_FEATURES]
            .iter_mut()
            .zip(base.features.iter())
        {
            *dst = *src as f32;
        }
        for (dst, src) in values[N_BASE_FEATURES..N_BASE_FEATURES + N_DEPTH_FEATURES]
            .iter_mut()
            .zip(depth.iter())
        {
            *dst = *src as f32;
        }
        for (dst, src) in values[N_BASE_FEATURES + N_DEPTH_FEATURES..]
            .iter_mut()
            .zip(pm.features.iter())
        {
            *dst = *src as f32;
        }
        Self { values }
    }

    /// How many of the 24 features are finite (a coarse live coverage signal).
    #[must_use]
    pub fn finite_count(&self) -> u32 {
        self.values.iter().filter(|v| v.is_finite()).count() as u32
    }

    /// A stable 64-bit hash of the vector (NaN canonicalized), rendered as hex —
    /// the journaled `features_hash` for integrity/dedup.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        // FNV-1a over each feature's canonical bit pattern (all NaNs → one code).
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &self.values {
            let bits = if v.is_nan() {
                0x7fc0_0000_u32 // canonical quiet NaN
            } else {
                v.to_bits()
            };
            for b in bits.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{h:016x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_features_has_24_unique_names() {
        assert_eq!(FULL_FEATURES.len(), 24);
        let mut sorted = FULL_FEATURES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 24, "feature names must be unique");
    }

    #[test]
    fn assemble_places_blocks_in_order() {
        let base = PriceSample {
            features: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            asof_sec: Some(1),
            sigma_returns: 0,
            basis_samples: 0,
        };
        let depth = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let pm = PmSample {
            features: [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            asof_ms: Some(1000),
            run_secs: Some(5),
        };
        let fv = FeatureVector::assemble(&base, &depth, &pm);
        for i in 0..N_FEATURES {
            assert!(
                (fv.values[i] - (i as f32 + 1.0)).abs() < 1e-6,
                "feature {i}"
            );
        }
        assert_eq!(fv.finite_count(), 24);
    }

    #[test]
    fn hash_is_stable_and_nan_canonical() {
        let mut a = [0.5_f32; N_FEATURES];
        a[3] = f32::NAN;
        let mut b = a;
        b[3] = f32::from_bits(0x7fa0_0000); // a different NaN bit pattern
        let fa = FeatureVector { values: a };
        let fb = FeatureVector { values: b };
        assert_eq!(fa.hash_hex(), fb.hash_hex(), "NaN patterns canonicalize");
        let mut c = a;
        c[0] = 0.6;
        assert_ne!(fa.hash_hex(), FeatureVector { values: c }.hash_hex());
    }
}
