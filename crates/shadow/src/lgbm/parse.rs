//! Parser for LightGBM's native text model (`model.txt`).
//!
//! The format is line-oriented `key=value`, a header block followed by one
//! `Tree=k` block per boosting round. We read only the fields prediction needs
//! (`objective` sigmoid coefficient, `max_feature_idx`, and per-tree
//! `num_leaves`/`num_cat`/`split_feature`/`threshold`/`decision_type`/
//! `left_child`/`right_child`/`leaf_value`) and validate structure defensively —
//! a categorical split (unexpected for the all-continuous champion), an
//! out-of-range child, or a length mismatch is a hard, named error rather than a
//! silent mis-prediction.

use std::collections::HashMap;

use thiserror::Error;

use super::tree::{Node, Tree};

/// Why a `model.txt` failed to parse into a usable model.
#[derive(Debug, Clone, Error)]
pub enum LgbmError {
    /// The file could not be read.
    #[error("reading model file {path}: {reason}")]
    Io {
        /// The offending path.
        path: String,
        /// The underlying IO error text.
        reason: String,
    },
    /// A required header field was missing.
    #[error("model header missing `{0}`")]
    MissingHeader(&'static str),
    /// A required per-tree field was missing.
    #[error("tree {tree}: missing `{field}`")]
    MissingTreeField {
        /// Which tree (0-based).
        tree: usize,
        /// The absent field name.
        field: &'static str,
    },
    /// A numeric field failed to parse.
    #[error("parsing `{field}`{}: {value:?}", tree.map(|t| format!(" in tree {t}")).unwrap_or_default())]
    BadNumber {
        /// The field name.
        field: &'static str,
        /// The tree index, if the field is per-tree.
        tree: Option<usize>,
        /// The unparseable token.
        value: String,
    },
    /// A tree declared a categorical split — the champion is all-continuous, so
    /// this signals a wrong/incompatible model rather than a supported case.
    #[error("tree {0} has a categorical split (num_cat > 0) — unsupported")]
    CategoricalSplit(usize),
    /// A per-tree array length disagreed with `num_leaves`.
    #[error("tree {tree}: `{field}` has {got} entries, expected {want}")]
    BadArrayLen {
        /// Which tree.
        tree: usize,
        /// The field name.
        field: &'static str,
        /// Observed length.
        got: usize,
        /// Expected length.
        want: usize,
    },
    /// A child index or split feature pointed out of range.
    #[error("tree {tree}: {what} out of range ({value})")]
    OutOfRange {
        /// Which tree.
        tree: usize,
        /// What was out of range.
        what: &'static str,
        /// The offending value.
        value: i64,
    },
    /// No trees were found.
    #[error("model has no trees")]
    NoTrees,
}

/// The parsed model surface prediction needs.
pub(crate) struct ParsedModel {
    pub sigmoid: f64,
    pub num_features: usize,
    pub trees: Vec<Tree>,
}

fn parse_f64_list(
    field: &'static str,
    tree: Option<usize>,
    s: &str,
) -> Result<Vec<f64>, LgbmError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<f64>().map_err(|_| LgbmError::BadNumber {
                field,
                tree,
                value: t.to_owned(),
            })
        })
        .collect()
}

fn parse_i32_list(field: &'static str, tree: usize, s: &str) -> Result<Vec<i32>, LgbmError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<i32>().map_err(|_| LgbmError::BadNumber {
                field,
                tree: Some(tree),
                value: t.to_owned(),
            })
        })
        .collect()
}

/// Extracts the sigmoid coefficient from an `objective=binary sigmoid:X` header
/// (default `1.0` if the token is absent).
fn sigmoid_from_objective(objective: &str) -> f64 {
    objective
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix("sigmoid:"))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0)
}

/// Parses a whole `model.txt` into the prediction surface.
pub(crate) fn parse(text: &str) -> Result<ParsedModel, LgbmError> {
    let mut lines = text.lines().peekable();

    // --- header: key=value until the first `Tree=` -------------------------
    let mut header: HashMap<&str, &str> = HashMap::new();
    while let Some(&line) = lines.peek() {
        if line.starts_with("Tree=") {
            break;
        }
        if let Some((k, v)) = line.split_once('=') {
            header.insert(k.trim(), v.trim());
        }
        lines.next();
    }
    let objective = header
        .get("objective")
        .ok_or(LgbmError::MissingHeader("objective"))?;
    let sigmoid = sigmoid_from_objective(objective);
    let max_feature_idx: usize = header
        .get("max_feature_idx")
        .ok_or(LgbmError::MissingHeader("max_feature_idx"))?
        .parse()
        .map_err(|_| LgbmError::BadNumber {
            field: "max_feature_idx",
            tree: None,
            value: (*header.get("max_feature_idx").unwrap_or(&"")).to_owned(),
        })?;
    let num_features = max_feature_idx + 1;

    // --- trees --------------------------------------------------------------
    let mut trees: Vec<Tree> = Vec::new();
    let mut current: HashMap<&str, &str> = HashMap::new();
    let mut in_tree = false;

    for line in lines {
        if line.starts_with("Tree=") {
            if in_tree {
                let idx = trees.len();
                trees.push(build_tree(&current, idx, num_features)?);
            }
            current = HashMap::new();
            in_tree = true;
            continue;
        }
        if line == "end of trees" {
            break;
        }
        if !in_tree {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            current.insert(k.trim(), v.trim());
        }
    }
    if in_tree {
        let idx = trees.len();
        trees.push(build_tree(&current, idx, num_features)?);
    }

    if trees.is_empty() {
        return Err(LgbmError::NoTrees);
    }
    Ok(ParsedModel {
        sigmoid,
        num_features,
        trees,
    })
}

/// Builds one [`Tree`] from its parsed `key=value` block.
fn build_tree(
    map: &HashMap<&str, &str>,
    idx: usize,
    num_features: usize,
) -> Result<Tree, LgbmError> {
    let get = |field: &'static str| -> Result<&str, LgbmError> {
        map.get(field)
            .copied()
            .ok_or(LgbmError::MissingTreeField { tree: idx, field })
    };
    let num_leaves: usize = get("num_leaves")?
        .parse()
        .map_err(|_| LgbmError::BadNumber {
            field: "num_leaves",
            tree: Some(idx),
            value: get("num_leaves").unwrap_or("").to_owned(),
        })?;
    let num_cat: i64 = get("num_cat")
        .unwrap_or("0")
        .parse()
        .map_err(|_| LgbmError::BadNumber {
            field: "num_cat",
            tree: Some(idx),
            value: get("num_cat").unwrap_or("").to_owned(),
        })?;
    if num_cat > 0 {
        return Err(LgbmError::CategoricalSplit(idx));
    }

    let leaf_values = parse_f64_list("leaf_value", Some(idx), get("leaf_value")?)?;
    if leaf_values.len() != num_leaves {
        return Err(LgbmError::BadArrayLen {
            tree: idx,
            field: "leaf_value",
            got: leaf_values.len(),
            want: num_leaves,
        });
    }

    // A single-leaf tree (constant) has no internal-node arrays.
    if num_leaves <= 1 {
        return Ok(Tree::new(Vec::new(), leaf_values));
    }

    let n_internal = num_leaves - 1;
    let split_feature = parse_usize_list("split_feature", idx, get("split_feature")?)?;
    let threshold = parse_f64_list("threshold", Some(idx), get("threshold")?)?;
    let decision_type = parse_i32_list("decision_type", idx, get("decision_type")?)?;
    let left_child = parse_i32_list("left_child", idx, get("left_child")?)?;
    let right_child = parse_i32_list("right_child", idx, get("right_child")?)?;

    for (field, len) in [
        ("split_feature", split_feature.len()),
        ("threshold", threshold.len()),
        ("decision_type", decision_type.len()),
        ("left_child", left_child.len()),
        ("right_child", right_child.len()),
    ] {
        if len != n_internal {
            return Err(LgbmError::BadArrayLen {
                tree: idx,
                field,
                got: len,
                want: n_internal,
            });
        }
    }

    let mut nodes = Vec::with_capacity(n_internal);
    for i in 0..n_internal {
        if split_feature[i] >= num_features {
            return Err(LgbmError::OutOfRange {
                tree: idx,
                what: "split_feature",
                value: split_feature[i] as i64,
            });
        }
        validate_child(idx, left_child[i], n_internal, num_leaves)?;
        validate_child(idx, right_child[i], n_internal, num_leaves)?;
        nodes.push(Node::new(
            split_feature[i],
            threshold[i],
            decision_type[i],
            left_child[i],
            right_child[i],
        ));
    }
    Ok(Tree::new(nodes, leaf_values))
}

/// A child is either an internal-node index in `[0, n_internal)` or a leaf
/// encoded as `!leaf_idx` with `leaf_idx` in `[0, num_leaves)`.
fn validate_child(
    tree: usize,
    child: i32,
    n_internal: usize,
    num_leaves: usize,
) -> Result<(), LgbmError> {
    if child >= 0 {
        if (child as usize) >= n_internal {
            return Err(LgbmError::OutOfRange {
                tree,
                what: "internal child",
                value: i64::from(child),
            });
        }
    } else {
        let leaf = !child;
        if leaf < 0 || (leaf as usize) >= num_leaves {
            return Err(LgbmError::OutOfRange {
                tree,
                what: "leaf child",
                value: i64::from(child),
            });
        }
    }
    Ok(())
}

fn parse_usize_list(field: &'static str, tree: usize, s: &str) -> Result<Vec<usize>, LgbmError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<usize>().map_err(|_| LgbmError::BadNumber {
                field,
                tree: Some(tree),
                value: t.to_owned(),
            })
        })
        .collect()
}
