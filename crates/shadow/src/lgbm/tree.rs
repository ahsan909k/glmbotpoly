//! A single decision tree from a LightGBM `model.txt` and its walk.
//!
//! One tree is a flat array of internal nodes (`num_leaves - 1` of them) plus a
//! leaf-value array. Prediction routes a feature vector from node 0 down to a
//! leaf and returns that leaf's raw score; the model sums the leaf scores across
//! all trees and applies the sigmoid.
//!
//! The routing at each node reproduces LightGBM's `Tree::NumericalDecision`
//! **exactly** (the 1e-6-parity contract, verified by the export fixture):
//! `<=` (not `<`) on the split threshold; per-node missing handling driven by
//! the `decision_type` bitmask (default direction + missing-type None/Zero/NaN);
//! a NaN at a non-NaN-missing node becomes `0.0` before the comparison; feature
//! values widen `f32 → f64` before the compare (training fed float32).

/// A genuine zero, per LightGBM's `Tree::IsZero` (`|x| <= 1e-35`).
const ZERO_THRESHOLD: f64 = 1e-35;

/// How a node treats a missing (NaN / zero) feature — the `decision_type`
/// bits 2–3 (`(dt >> 2) & 3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MissingType {
    /// No special missing handling (0): a NaN is coerced to `0.0` and compared.
    None,
    /// An exact zero (1) routes to the default direction.
    Zero,
    /// A NaN (2) routes to the default direction.
    Nan,
}

impl MissingType {
    fn from_decision_type(dt: i32) -> Self {
        match (dt >> 2) & 3 {
            1 => MissingType::Zero,
            2 => MissingType::Nan,
            _ => MissingType::None,
        }
    }
}

/// One internal (split) node.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    /// Index into the feature vector this node splits on.
    pub split_feature: usize,
    /// Split threshold (`<=` goes left).
    pub threshold: f64,
    /// Whether a missing value routes left (else right).
    pub default_left: bool,
    /// How this node treats a missing value.
    pub missing: MissingType,
    /// Left child: `>= 0` an internal node index, `< 0` a leaf (`!child`).
    pub left: i32,
    /// Right child: same encoding.
    pub right: i32,
}

impl Node {
    /// Builds a node from the parsed per-node arrays, decoding `decision_type`.
    pub(crate) fn new(
        split_feature: usize,
        threshold: f64,
        decision_type: i32,
        left: i32,
        right: i32,
    ) -> Self {
        Self {
            split_feature,
            threshold,
            default_left: (decision_type >> 1) & 1 == 1,
            missing: MissingType::from_decision_type(decision_type),
            left,
            right,
        }
    }

    /// Reproduces LightGBM `NumericalDecision`: returns the chosen child code
    /// (`>= 0` internal node, `< 0` leaf).
    #[inline]
    fn decide(&self, raw: f64) -> i32 {
        // A NaN at a non-NaN-missing node is treated as 0.0 (LightGBM behavior).
        let mut fval = raw;
        if fval.is_nan() && self.missing != MissingType::Nan {
            fval = 0.0;
        }
        let is_missing = match self.missing {
            MissingType::Zero => fval.abs() <= ZERO_THRESHOLD,
            MissingType::Nan => fval.is_nan(),
            MissingType::None => false,
        };
        if is_missing {
            if self.default_left {
                self.left
            } else {
                self.right
            }
        } else if fval <= self.threshold {
            self.left
        } else {
            self.right
        }
    }
}

/// One decision tree: `num_leaves - 1` internal nodes + `num_leaves` leaf values.
#[derive(Debug, Clone)]
pub(crate) struct Tree {
    nodes: Box<[Node]>,
    leaf_values: Box<[f64]>,
}

impl Tree {
    pub(crate) fn new(nodes: Vec<Node>, leaf_values: Vec<f64>) -> Self {
        Self {
            nodes: nodes.into_boxed_slice(),
            leaf_values: leaf_values.into_boxed_slice(),
        }
    }

    /// Routes `features` (already widened to `f64`) to a leaf and returns its
    /// raw score. A tree with no internal nodes (a single-leaf stump) returns
    /// its lone leaf value.
    #[inline]
    #[must_use]
    pub(crate) fn walk(&self, features: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values.first().copied().unwrap_or(0.0);
        }
        let mut node_idx: i32 = 0;
        // Bounded by tree depth (num_leaves - 1); the +1 guards a malformed
        // cyclic tree from spinning (the parser rejects out-of-range children).
        for _ in 0..=self.nodes.len() {
            let node = &self.nodes[node_idx as usize];
            let child = node.decide(features[node.split_feature]);
            if child < 0 {
                // Leaf encoded as the bitwise-NOT of the leaf index.
                return self.leaf_values[(!child) as usize];
            }
            node_idx = child;
        }
        // Unreachable for a well-formed tree; a defensive fallback.
        self.leaf_values.first().copied().unwrap_or(0.0)
    }

    #[cfg(test)]
    pub(crate) fn num_leaves(&self) -> usize {
        self.leaf_values.len()
    }
}
