//! A pure-Rust LightGBM `model.txt` loader and predictor — the allowlist-clean
//! "ONNX equivalent" (no inference dependency; CLAUDE.md §3).
//!
//! A binary GBT prediction is the sigmoid of the sum of the reached leaf values
//! across every tree (`shrinkage`/`boost_from_average` are baked into the saved
//! leaf values, so there is nothing to add back — the champion is the plain
//! variant with no `init_score`). Feature values are `f32` (training fed
//! float32) widened to `f64` for the threshold comparisons, matching LightGBM's
//! predictor exactly. Correctness is pinned to 1e-6 against `booster.predict` by
//! the export fixture (`tests/export_parity.rs`).

mod parse;
mod tree;

use std::path::Path;

pub use parse::LgbmError;
use tree::Tree;

/// A loaded LightGBM binary classifier ready for prediction.
#[derive(Debug, Clone)]
pub struct LgbmModel {
    trees: Vec<Tree>,
    /// The `objective=binary sigmoid:X` coefficient.
    sigmoid: f64,
    /// Feature-vector width the model expects (`max_feature_idx + 1`).
    num_features: usize,
    /// Scratch buffer for the widened feature vector (avoids per-predict alloc).
    scratch: Vec<f64>,
}

impl LgbmModel {
    /// Parses a model from `model.txt` text.
    ///
    /// # Errors
    /// [`LgbmError`] on a malformed header/tree, a categorical split, or an
    /// out-of-range index.
    pub fn from_text(text: &str) -> Result<Self, LgbmError> {
        let parsed = parse::parse(text)?;
        Ok(Self {
            trees: parsed.trees,
            sigmoid: parsed.sigmoid,
            num_features: parsed.num_features,
            scratch: vec![0.0; parsed.num_features],
        })
    }

    /// Loads a model from a `model.txt` file.
    ///
    /// # Errors
    /// [`LgbmError::Io`] if the file cannot be read, else any [`from_text`](Self::from_text) error.
    pub fn load(path: &Path) -> Result<Self, LgbmError> {
        let text = std::fs::read_to_string(path).map_err(|e| LgbmError::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        Self::from_text(&text)
    }

    /// The feature-vector width the model expects.
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// The number of trees (boosting rounds).
    #[must_use]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// P(Up) for one feature vector (`f32`, NaN = native-missing).
    ///
    /// Widens each feature to `f64`, sums the reached leaf values over all
    /// trees, and applies the overflow-safe logistic. If `features` is shorter
    /// than [`num_features`](Self::num_features) the missing tail is treated as
    /// `NaN` (native-missing) — a defensive guard; callers pass a full vector.
    #[must_use]
    pub fn predict(&self, features: &[f32]) -> f64 {
        let mut buf = self.scratch.clone();
        for (dst, src) in buf.iter_mut().zip(features.iter()) {
            *dst = f64::from(*src);
        }
        for dst in buf.iter_mut().skip(features.len()) {
            *dst = f64::NAN;
        }
        self.predict_widened(&buf)
    }

    /// Predict from an already-`f64` feature vector (the fixture path; avoids
    /// the widen copy so the test feeds exact values incl. explicit NaN).
    #[must_use]
    pub fn predict_widened(&self, features: &[f64]) -> f64 {
        let raw: f64 = self.trees.iter().map(|t| t.walk(features)).sum();
        sigmoid(self.sigmoid * raw)
    }
}

/// Overflow-safe logistic (matches `model_lab.lib.gbt._sigmoid`).
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny two-tree model exercising `<=` routing, default-left, and
    // missing-type NaN. Feature 0 splits at 0.5 (default-left, missing=NaN=8→
    // wait, decision_type below sets the bits explicitly).
    const TINY: &str = "tree
version=v4
num_class=1
max_feature_idx=1
objective=binary sigmoid:1
feature_names=Column_0 Column_1

Tree=0
num_leaves=2
num_cat=0
split_feature=0
threshold=0.5
decision_type=10
left_child=-1
right_child=-2
leaf_value=-1.0 1.0

Tree=1
num_leaves=2
num_cat=0
split_feature=1
threshold=0.5
decision_type=2
left_child=-1
right_child=-2
leaf_value=0.0 0.5
end of trees
";

    #[test]
    fn parses_header_and_trees() {
        let m = LgbmModel::from_text(TINY).expect("parse");
        assert_eq!(m.num_features(), 2);
        assert_eq!(m.num_trees(), 2);
        assert!((m.sigmoid - 1.0).abs() < 1e-12);
    }

    #[test]
    fn le_routing_goes_left_on_threshold() {
        let m = LgbmModel::from_text(TINY).expect("parse");
        // feature0 <= 0.5 → left leaf (-1.0); feature1 = 0.0 <= 0.5 → left (0.0).
        // raw = -1.0 + 0.0 = -1.0 → sigmoid(-1).
        let p = m.predict(&[0.5, 0.0]);
        assert!((p - (1.0 / (1.0 + 1.0_f64.exp()))).abs() < 1e-12, "p={p}");
        // feature0 > 0.5 → right leaf (1.0); feature1 = 1.0 > 0.5 → right (0.5).
        // raw = 1.5 → sigmoid(1.5).
        let p2 = m.predict(&[0.6, 1.0]);
        let want = 1.0 / (1.0 + (-1.5_f64).exp());
        assert!((p2 - want).abs() < 1e-12, "p2={p2}");
    }

    #[test]
    fn nan_at_nan_node_routes_default_left() {
        let m = LgbmModel::from_text(TINY).expect("parse");
        // Tree 0 decision_type=10 → missing NaN, default_left. NaN feature0 →
        // left leaf (-1.0). Tree 1 decision_type=2 → missing None, so NaN→0.0,
        // 0.0<=0.5 → left leaf (0.0). raw = -1.0.
        let p = m.predict(&[f32::NAN, f32::NAN]);
        assert!((p - (1.0 / (1.0 + 1.0_f64.exp()))).abs() < 1e-12, "p={p}");
    }

    #[test]
    fn rejects_categorical_split() {
        let bad = TINY.replace("num_cat=0\nsplit_feature=0", "num_cat=1\nsplit_feature=0");
        assert!(matches!(
            LgbmModel::from_text(&bad),
            Err(LgbmError::CategoricalSplit(_))
        ));
    }

    #[test]
    fn tree_leaf_count() {
        let m = LgbmModel::from_text(TINY).expect("parse");
        assert_eq!(m.trees[0].num_leaves(), 2);
    }
}
