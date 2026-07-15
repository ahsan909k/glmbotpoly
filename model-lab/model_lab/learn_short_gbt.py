"""Stage — learn_short_gbt. A GBT challenger on the *short-horizon* dataset.

Trains a deterministic **LightGBM** binary classifier on ``short_horizon.parquet``
to predict the short-horizon forward-direction labels ``fwd_up_10s`` (primary) and
``fwd_up_15s`` (secondary), reusing the same strict walk-forward scheme, purge gaps,
inner-validation round selection, determinism contract, and shuffled-label control
as :mod:`model_lab.learn_gbt`. Unlike that stage (which scores against the *window*
outcome through the harness), here the model is scored on its **own** short-horizon
label — dir-acc, Brier, and a calibration (reliability) table.

**Trade-flow features are joined in from ``feature_set.parquet``.** The short-horizon
dataset carries no signed-aggressor-flow columns (``flow_imb_*`` / ``trade_intensity_*``
live only in the feature-set stage). Because ``feature_set`` is built on the dataset's
15-second grid while ``short_horizon`` is a denser 5-second grid, those six columns are
attached by a **causal backward as-of join** per ``(series, window_open_ms)`` — never an
exact key join — so each row gets the most recent flow value at-or-before its
``sample_ts_ms`` (bounded by a staleness tolerance). ``--no-external-flow`` skips the join.

**Three-way feature ablation** — the "does microstructure add anything" question:

- **base**  — always-available price / vol / model features + the joined flow;
- **depth** — base + Binance ``depth20`` microstructure (book imbalance, slopes, …);
- **full**  — depth + Polymarket top-of-book features (mid, imbalance, staleness).

Depth / PM features exist only where our own recordings cover the timestamp (always NaN
on the ``binance_proxy`` history); LightGBM handles the NaN natively (learned default
branch), so ``depth`` / ``full`` still train on every row. The pairwise **contribution**
analysis (base→depth = Binance-depth lift, depth→full = PM lift) reports the Brier /
dir-acc delta, overall and on the covered subset.

**Two source scopes in one run.** The whole ablation runs over ``all``
(proxy + chainlink, the primary scope) *and* ``chainlink``-only (the clean microstructure
ablation where depth / PM actually exist), both reported side by side.

**Beyond the pooled scorecard** — an **abstention** curve (coverage-vs-accuracy as a
confidence threshold ``|p−0.5|`` rises), and per-regime (volatility, time-remaining,
source, depth-coverage) breakdowns against the naive ``Φ(z)`` reference.

**Artifacts** (``out/learn_short_gbt/``, per ``tag = <scope>_<target>_<variant>``):
``model_<tag>.txt`` (native booster) + ``model_<tag>.json`` (metadata + importances),
``predictions_<tag>.parquet`` (+ ``_shuffled``), ``oos_<tag>.parquet``,
``feature_importance_<tag>.csv``, ``regime_<tag>.csv``, ``abstention_<tag>.csv``,
``reliability_<tag>.csv``; plus once-per-run ``depth_contribution.csv``, ``folds.csv``,
``metrics.json``. It is an **opt-in** stage — LightGBM ships as the ``gbt`` extra and the
core pipeline stays lightgbm-free.

Run::

    python -m model_lab.learn_short_gbt                       # both targets/variants/scopes
    python -m model_lab.learn_short_gbt --targets fwd10 --scopes chainlink   # quick iteration
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import feature_set as fs
from . import short_horizon as sh
from .config import (
    ParquetNotReady, Paths, assert_parquet_ready, resolve_bounds, resolve_paths, stage_parser,
)
from .learn_common import (
    MS_PER_DAY, _fold_metrics, _fold_schedule, _harness_grid, _harness_summary,
    _label_info_ms, _parse_series, score_through_harness,
)
from .lib import gbt
from .lib import math as lm

DEFAULT_SEED = 20260704
DEFAULT_DAYS = 90
DEFAULT_TRAIN_WEEKS = 4
DEFAULT_TEST_DAYS = 7
DEFAULT_MIN_WINDOWS = eh.DEFAULT_MIN_WINDOWS

# GBT hyperparameter defaults — modest depth, strong regularization (== learn_gbt).
DEFAULT_NUM_LEAVES = 15
DEFAULT_MAX_DEPTH = 4
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_MIN_CHILD_SAMPLES = 100
DEFAULT_REG_LAMBDA = 5.0
DEFAULT_REG_ALPHA = 0.0
DEFAULT_MAX_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_INNER_VAL_FRAC = 0.25
DEFAULT_NUM_THREADS = 1
# Round count used when an inner validation split can't be formed (degenerate fold).
FALLBACK_ROUNDS = 200

# Targets — primary = 10-second forward direction, secondary = 15-second.
TARGETS_SHORT: dict[str, dict] = {
    "fwd10": {"label": "fwd_up_10s", "horizon_secs": 10},
    "fwd15": {"label": "fwd_up_15s", "horizon_secs": 15},
}

# --- feature groups (single source of truth) --------------------------------
# Native short-horizon price / vol / model features (always-available; basis_bps /
# basis_ewma are NaN on binance_proxy rows — a documented caveat, GBT handles it).
NATIVE_BASE = ["ret", "realized_vol", "sigma_1s", "log_s_k", "z", "p_up_model",
               "tau_secs", "elapsed_secs", "basis_bps", "basis_ewma"]
# Signed aggressor flow, joined as-of from feature_set.parquet (available everywhere).
FLOW_FEATURES = ["flow_imb_30s", "flow_imb_120s", "flow_imb_300s",
                 "trade_intensity_30s", "trade_intensity_120s", "trade_intensity_300s"]
# Binance depth20 microstructure (present only where our depth recording covers the row).
DEPTH_FEATURES = ["depth_imb_1", "depth_imb_5", "depth_imb_10", "depth_imb_20",
                  "microprice_gap", "bid_depth_slope", "ask_depth_slope", "depth_spread_bps"]
# Polymarket Up-token book (present only where our top_of_book recording covers the row).
PM_FEATURES = ["pm_mid", "pm_spread", "pm_book_imb",
               "pm_staleness_1s", "pm_staleness_2s", "pm_staleness_3s"]
VARIANT_ORDER = ["base", "depth", "full"]
DEFAULT_SCOPES = ("all", "chainlink")

# Abstention sweep — confidence threshold on |p_up − 0.5|.
ABSTENTION_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
# Bounds staleness of the as-of flow join (feature_set 15s grid vs short-horizon 5s grid).
FLOW_ASOF_TOLERANCE_MS = 90_000

# Drift guard (import-time): every feature we name must exist in the producing schema.
_META_FEATS = {"tau_secs", "elapsed_secs"}  # short_horizon "meta" columns, not FEATURE_COLS
assert (set(DEPTH_FEATURES) | set(PM_FEATURES) | (set(NATIVE_BASE) - _META_FEATS)) <= set(sh.FEATURE_COLS), \
    "learn_short_gbt: a native feature is missing from short_horizon.FEATURE_COLS"
assert _META_FEATS <= set(sh.COLUMNS), "learn_short_gbt: a meta feature is missing from short_horizon.COLUMNS"
assert set(FLOW_FEATURES) <= set(fs.FEATURE_NAMES), \
    "learn_short_gbt: a flow feature is missing from feature_set.FEATURE_NAMES"

CAVEATS = [
    "LightGBM determinism is within-machine only (deterministic=True + force_row_wise + "
    "single thread + pinned seeds); the exact lightgbm version is pinned in requirements-gbt.txt.",
    "Three feature-set variants — base (price/vol/model + joined flow), depth (+ Binance depth20), "
    "full (+ Polymarket book). Trees handle NaN natively, so depth/full train on binance_proxy rows "
    "where depth/PM are all-NaN (learned default branch).",
    "Scored on the model's OWN forward-direction label (fwd_up_10s/15s), not the window outcome; the "
    "Φ(z)=p_up_model arm is a naive reference for the same label, and the harness (vs window outcome_up) "
    "is off by default (--with-harness) since it scores a different question.",
    "Signed-flow features are joined from feature_set.parquet by a causal backward as-of join "
    "(feature_set is on the dataset 15s grid, short_horizon on 5s), bounded by a staleness tolerance; "
    "the join is look-ahead-safe (matched feature_set ts ≤ the sample ts). --no-external-flow skips it.",
    "Boosting rounds are chosen by inner forward-chained validation (a purged tail of each training "
    "window), then refit on the full window — the round count is validated, never fit on the test. "
    "base features basis_bps/basis_ewma are NaN on binance_proxy rows (no Chainlink feed).",
]


# ===========================================================================
# data loading + external-flow join
# ===========================================================================
_KEY_COLS = ["series", "window_open_ms", "window_close_ms", "sample_ts_ms"]
_ANALYSIS_COLS = ["label_source", "depth_feat_covered", "pm_feat_covered"]
_LABEL_COLS = ["fwd_up_10s", "fwd_up_15s"]
# Every short-horizon-native column we read (features + regime/coverage + labels + keys).
_SHORT_FEATURE_COLS = NATIVE_BASE + DEPTH_FEATURES + PM_FEATURES
_SHORT_READ_COLS = list(dict.fromkeys(_KEY_COLS + _SHORT_FEATURE_COLS + _ANALYSIS_COLS + _LABEL_COLS))


def _resolve_feature_sets(external_flow: bool) -> dict[str, list[str]]:
    """The three feature sets. ``base`` includes the joined flow only when it is present."""
    base = list(NATIVE_BASE) + (list(FLOW_FEATURES) if external_flow else [])
    return {"base": base, "depth": base + DEPTH_FEATURES, "full": base + DEPTH_FEATURES + PM_FEATURES}


def _load_short_matrix(paths: Paths, *, days: int, series: list[str] | None) -> pd.DataFrame:
    """Read the trailing ``days`` of ``short_horizon.parquet`` (predicate-pushed on
    ``window_open_ms``), optionally filtered by series. Always loads every source — the
    scope loop filters ``label_source`` in memory."""
    import pyarrow.parquet as pq

    path = paths.table("short_horizon")
    assert_parquet_ready(path, label="short_horizon.parquet", min_rows=1)
    opens = pq.read_table(path, columns=["window_open_ms"]).column(0).to_numpy()
    if len(opens) == 0:
        return pd.DataFrame(columns=_SHORT_READ_COLS)
    max_open = int(opens.max())
    filters = None
    if days and days > 0:
        cutoff = max_open - int(days) * MS_PER_DAY
        filters = [("window_open_ms", ">=", int(cutoff))]
    mat = pq.read_table(path, columns=_SHORT_READ_COLS, filters=filters).to_pandas()
    if series is not None:
        mat = mat[mat["series"].isin(series)]
    return mat.reset_index(drop=True)


def _join_external_flow(mat: pd.DataFrame, paths: Paths, *, tolerance_ms: int) -> tuple[pd.DataFrame, dict]:
    """Attach ``feature_set``'s signed-flow columns by a causal backward as-of join per
    ``(series, window_open_ms)`` on ``sample_ts_ms``. The matched feature-set row is the
    latest at-or-before the sample time (``≤`` — look-ahead-safe), bounded by
    ``tolerance_ms``; unmatched rows get NaN flow (LightGBM handles it). Returns
    ``(mat_with_flow, stats)``."""
    import pyarrow.parquet as pq

    path = paths.table("feature_set")
    assert_parquet_ready(path, label="feature_set.parquet", min_rows=1)
    lo, hi = int(mat["window_open_ms"].min()), int(mat["window_open_ms"].max())
    cols = ["series", "window_open_ms", "sample_ts_ms", *FLOW_FEATURES]
    fsdf = pq.read_table(
        path, columns=cols,
        filters=[("window_open_ms", ">=", lo), ("window_open_ms", "<=", hi)],
    ).to_pandas()
    for c in FLOW_FEATURES:
        fsdf[c] = fsdf[c].astype("float64")
    fsdf["sample_ts_ms"] = fsdf["sample_ts_ms"].astype("int64")
    # Sentinel to distinguish "got an as-of match" from "the matched row carried a
    # non-NaN flow value" (feature_set flow is legitimately NaN where the aggTrades
    # coverage / EWMA warmup is thin — see feature_set's expected-NaN handling).
    fsdf["_fs_asof"] = True

    # merge_asof requires both frames sorted by the `on` key.
    left = mat.sort_values("sample_ts_ms", kind="stable").reset_index(drop=True)
    right = fsdf.sort_values("sample_ts_ms", kind="stable").reset_index(drop=True)
    joined = pd.merge_asof(
        left, right, on="sample_ts_ms", by=["series", "window_open_ms"],
        direction="backward", tolerance=int(tolerance_ms),
    )
    n = int(len(joined))
    asof = int(joined["_fs_asof"].fillna(False).to_numpy().sum()) if n else 0
    present = int(joined[FLOW_FEATURES[0]].notna().sum()) if n else 0
    joined = joined.drop(columns=["_fs_asof"])
    stats = {
        "joined": True,
        "n_asof_matched": asof, "frac_asof_matched": (asof / n) if n else None,
        "n_flow_present": present, "frac_flow_present": (present / n) if n else None,
        "tolerance_ms": int(tolerance_ms), "feature_set_rows": int(len(fsdf)),
    }
    return joined, stats


# ===========================================================================
# inner-validation round selection + walk-forward
# ===========================================================================
def _inner_best_iteration(X, y, ts, label_info, *, params, max_boost_round,
                          early_stopping_rounds, inner_val_frac, purge_ms):
    """Pick the boosting-round count by an inner forward-chained validation split: the
    latest ``inner_val_frac`` (by ``ts``) is validation, the inner-train is purged of
    rows whose label reaches within ``purge_ms`` of the inner-val start. Returns
    ``(best_iteration, n_inner_train, n_inner_val)``; falls back to a modest fixed count
    when a split can't be formed."""
    n = len(y)
    n_val = int(round(inner_val_frac * n))
    fallback = min(int(max_boost_round), FALLBACK_ROUNDS)
    if n_val < 1 or n_val >= n:
        return fallback, n, 0
    order = np.argsort(ts, kind="stable")
    val_idx = order[-n_val:]
    val_start_ms = int(ts[val_idx].min())
    cand = order[:-n_val]
    tr_idx = cand[label_info[cand] <= val_start_ms - int(purge_ms)]
    if len(tr_idx) == 0:
        return fallback, 0, int(n_val)
    _, best = gbt.fit(
        X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], params=params,
        max_boost_round=max_boost_round, early_stopping_rounds=early_stopping_rounds,
    )
    return int(best), int(len(tr_idx)), int(n_val)


_OOS_COLS_SHORT = ["series", "window_open_ms", "sample_ts_ms", "p_up", "y_true",
                   "p_up_model", "sigma_1s", "tau_secs", "dur", "label_source",
                   "depth_feat_covered", "pm_feat_covered"]


def _run_walk_forward_short(sub, y, label_info, *, feature_names, train_weeks, test_days,
                            purge_ms, params, max_boost_round, early_stopping_rounds, inner_val_frac):
    """Slide a trailing train window over ``sub``; per fold pick rounds on an inner
    forward-chained val, refit on the full train window, predict OOS test. Parametrized
    by ``feature_names`` (so the base/depth/full variants share one path). Returns
    ``(oos_df, fold_records, mode)`` — ``oos_df`` carries the harness grid keys +
    ``p_up``/``y_true`` and the analysis columns."""
    open_ms = sub["window_open_ms"].to_numpy(np.int64)
    close_ms = sub["window_close_ms"].to_numpy(np.int64)
    ts = sub["sample_ts_ms"].to_numpy(np.int64)
    day = open_ms // MS_PER_DAY
    X = sub[feature_names].to_numpy(dtype=float)
    p_up_model = sub["p_up_model"].to_numpy(dtype=float)
    sigma_1s = sub["sigma_1s"].to_numpy(dtype=float)
    tau_secs = sub["tau_secs"].to_numpy(dtype=float)
    dur = (close_ms - open_ms) / 1000.0
    label_source = sub["label_source"].to_numpy()
    depth_cov = sub["depth_feat_covered"].to_numpy(dtype=bool)
    pm_cov = sub["pm_feat_covered"].to_numpy(dtype=bool)
    train_days = int(train_weeks) * 7
    folds, mode = _fold_schedule(np.unique(day), train_days, test_days)

    oos_parts: list[pd.DataFrame] = []
    records: list[dict] = []
    for i, (train_start, test_start, test_end) in enumerate(folds):
        train_mask = (
            (day >= train_start) & (day < test_start)
            & (label_info <= int(test_start) * MS_PER_DAY - int(purge_ms))
        )
        test_mask = (day >= test_start) & (day < test_end)
        tr = np.flatnonzero(train_mask)
        te = np.flatnonzero(test_mask)
        if len(tr) == 0 or len(te) == 0:
            records.append({
                "fold_idx": i, "test_start_day": int(test_start), "test_end_day": int(test_end),
                "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_windows": 0,
                "best_iteration": np.nan, "n_inner_train": 0, "n_inner_val": 0, "skipped": True,
                "pos_frac": float("nan"), "brier": float("nan"),
                "logloss": float("nan"), "diracc": float("nan"),
            })
            continue
        best, n_it, n_iv = _inner_best_iteration(
            X[tr], y[tr], ts[tr], label_info[tr], params=params,
            max_boost_round=max_boost_round, early_stopping_rounds=early_stopping_rounds,
            inner_val_frac=inner_val_frac, purge_ms=purge_ms,
        )
        booster = gbt.refit(X[tr], y[tr], params=params, num_boost_round=best)
        prob = gbt.predict_proba(
            booster, X[te], num_iteration=best, num_threads=params["num_threads"])
        part = pd.DataFrame({
            "series": sub["series"].to_numpy()[te],
            "window_open_ms": open_ms[te],
            "sample_ts_ms": ts[te],
            "p_up": prob,
            "y_true": y[te],
            "p_up_model": p_up_model[te],
            "sigma_1s": sigma_1s[te],
            "tau_secs": tau_secs[te],
            "dur": dur[te],
            "label_source": label_source[te],
            "depth_feat_covered": depth_cov[te],
            "pm_feat_covered": pm_cov[te],
        })
        oos_parts.append(part)
        m = _fold_metrics(prob, y[te])
        n_win = int(part.groupby(["series", "window_open_ms"]).ngroups)
        records.append({
            "fold_idx": i, "test_start_day": int(test_start), "test_end_day": int(test_end),
            "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_windows": n_win,
            "best_iteration": int(best), "n_inner_train": int(n_it), "n_inner_val": int(n_iv),
            "skipped": False, **m,
        })

    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame(columns=_OOS_COLS_SHORT)
    return oos, records, mode


def _median_best_iter(records: list[dict]) -> int | None:
    vals = [r["best_iteration"] for r in records if not r["skipped"]
            and r["best_iteration"] is not None and np.isfinite(r["best_iteration"])]
    return int(np.median(vals)) if vals else None


def _final_model_short(sub, y, label_info, *, feature_names, feature_set, train_weeks, params,
                       max_boost_round, early_stopping_rounds, inner_val_frac, purge_ms, rounds,
                       target, label, horizon_secs):
    """Fit the deployable booster on the most recent ``train_weeks`` of data, using
    ``rounds`` (the fold-median best_iteration) if given, else a fresh inner-val
    early-stop. Returns ``(booster, meta, num_rounds)``."""
    open_ms = sub["window_open_ms"].to_numpy(np.int64)
    day = open_ms // MS_PER_DAY
    hi = int(day.max())
    start_day = hi - int(train_weeks) * 7 + 1
    mask = day >= start_day
    if int(mask.sum()) == 0:
        mask = np.ones(len(day), dtype=bool)
        start_day = int(day.min())
    idx = np.flatnonzero(mask)
    X = sub[feature_names].to_numpy(dtype=float)[idx]
    yt = y[idx]
    ts = sub["sample_ts_ms"].to_numpy(np.int64)[idx]
    li = label_info[idx]
    num_rounds = rounds
    if num_rounds is None:
        num_rounds, _, _ = _inner_best_iteration(
            X, yt, ts, li, params=params, max_boost_round=max_boost_round,
            early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac,
            purge_ms=purge_ms,
        )
    booster = gbt.refit(X, yt, params=params, num_boost_round=num_rounds)
    importance = gbt.feature_importance(booster, list(feature_names))
    meta = {
        "target": target,
        "label": label,
        "horizon_secs": horizon_secs,
        "feature_set": feature_set,
        "model": "lightgbm_gbdt",
        "params": params,
        "feature_names": list(feature_names),
        "num_boost_round": int(num_rounds),
        "feature_importance": importance,
        "n_train": int(len(idx)),
        "train_span_days": [start_day, hi],
        "train_span_open_ms": [int(open_ms[idx].min()), int(open_ms[idx].max())],
        "class_balance_pos_frac": float(np.mean(yt)),
    }
    return booster, meta, int(num_rounds)


# ===========================================================================
# analyses — own-label scoring, abstention, regime, contribution
# ===========================================================================
def _nan_metrics() -> dict:
    return {"n": 0, "pos_frac": float("nan"), "brier": float("nan"),
            "logloss": float("nan"), "diracc": float("nan")}


def _pair_metrics(prob: np.ndarray, out: np.ndarray) -> dict:
    return {"brier": lm.brier_score(prob, out), "logloss": lm.log_loss(prob, out),
            "diracc": lm.directional_accuracy(prob, out)}


def _analysis_frame_short(oos: pd.DataFrame) -> pd.DataFrame:
    """Rows on which the model-vs-reference analysis is defined: finite ``p_up_model``
    (so Φ(z) exists) and finite ``sigma_1s`` (past warmup), with a tau bucket attached."""
    df = oos.copy()
    finite = np.isfinite(df["p_up_model"].to_numpy(float)) & np.isfinite(df["sigma_1s"].to_numpy(float))
    df = df[finite].reset_index(drop=True)
    if len(df):
        df["tau_bucket"] = eh.tau_bucket(df["tau_secs"].to_numpy(float), df["dur"].to_numpy(float))
    else:
        df["tau_bucket"] = pd.Series(dtype=object)
    return df


def _regime_blocks(af: pd.DataFrame) -> dict:
    """Model (``p_up``) vs reference (``p_up_model`` = Φ(z)) scores split by volatility
    regime (median ``sigma_1s``), time-remaining bucket, source, and depth coverage."""
    if af.empty:
        return {"vol": {}, "tau": {}, "source": {}, "depth_cov": {}}
    out = af["y_true"].to_numpy(float)
    model_p = af["p_up"].to_numpy(float)
    ref_p = af["p_up_model"].to_numpy(float)

    def block(mask) -> dict:
        if not np.any(mask):
            return {"n": 0}
        return {"n": int(mask.sum()),
                "model": _pair_metrics(model_p[mask], out[mask]),
                "reference": _pair_metrics(ref_p[mask], out[mask])}

    sig = af["sigma_1s"].to_numpy(float)
    med = float(np.median(sig))
    vol = {"median_sigma_1s": med, "low": block(sig <= med), "high": block(sig > med)}
    tb = af["tau_bucket"].to_numpy()
    tau = {bucket: block(tb == bucket) for bucket in eh.TAU_ORDER if np.any(tb == bucket)}
    src = af["label_source"].to_numpy()
    source = {s: block(src == s) for s in ("chainlink", "binance_proxy") if np.any(src == s)}
    dcov = af["depth_feat_covered"].to_numpy(bool)
    depth_cov = {}
    if np.any(dcov):
        depth_cov["covered"] = block(dcov)
    if np.any(~dcov):
        depth_cov["uncovered"] = block(~dcov)
    return {"vol": vol, "tau": tau, "source": source, "depth_cov": depth_cov}


def _abstention_curve(oos: pd.DataFrame) -> list[dict]:
    """Coverage-vs-accuracy as the confidence threshold ``|p_up − 0.5| ≥ t`` rises:
    per ``t``, the retained fraction (coverage, monotone non-increasing) and the
    dir-acc / Brier on the retained rows."""
    if oos.empty:
        return []
    p = oos["p_up"].to_numpy(float)
    y = oos["y_true"].to_numpy(float)
    conf = np.abs(p - 0.5)
    n_all = len(y)
    rows: list[dict] = []
    for t in ABSTENTION_THRESHOLDS:
        mask = conf >= t
        n = int(mask.sum())
        rows.append({
            "threshold": float(t),
            "coverage": (n / n_all) if n_all else float("nan"),
            "n": n,
            "diracc": lm.directional_accuracy(p[mask], y[mask]) if n else float("nan"),
            "brier": lm.brier_score(p[mask], y[mask]) if n else float("nan"),
        })
    return rows


def _pair_contribution(oos_a: pd.DataFrame, oos_b: pd.DataFrame, cov_col: str) -> dict:
    """Brier / dir-acc deltas from feature set ``a`` to ``b`` on the shared OOS rows
    (identical keys — the feature set doesn't change row selection), split by ``cov_col``
    into overall / covered / uncovered. Positive ``brier_delta`` = ``b`` is worse."""
    keys = ["series", "window_open_ms", "sample_ts_ms"]
    if oos_a.empty or oos_b.empty:
        return {"overall": {"n": 0}, "covered": {"n": 0}, "uncovered": {"n": 0}}
    j = oos_a[keys + ["p_up", "y_true", cov_col]].merge(
        oos_b[keys + ["p_up"]], on=keys, how="inner", suffixes=("_a", "_b"),
        validate="one_to_one")

    def blk(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0}
        y = sub["y_true"].to_numpy(float)
        pa = sub["p_up_a"].to_numpy(float)
        pb = sub["p_up_b"].to_numpy(float)
        ba, bb = lm.brier_score(pa, y), lm.brier_score(pb, y)
        da, db = lm.directional_accuracy(pa, y), lm.directional_accuracy(pb, y)
        return {"n": int(len(sub)), "brier_a": ba, "brier_b": bb, "brier_delta": bb - ba,
                "diracc_a": da, "diracc_b": db, "diracc_delta": db - da}

    cov = j[cov_col].to_numpy(bool)
    return {"overall": blk(j), "covered": blk(j[cov]), "uncovered": blk(j[~cov])}


def _flatten_regime_rows(regime: dict) -> list[dict]:
    rows: list[dict] = []

    def emit(breakdown: str, bucket: str, blk: dict) -> None:
        if not blk or blk.get("n", 0) == 0:
            return
        rows.append({
            "breakdown": breakdown, "bucket": bucket, "n": blk["n"],
            "model_brier": blk["model"]["brier"], "model_logloss": blk["model"]["logloss"],
            "model_diracc": blk["model"]["diracc"], "ref_brier": blk["reference"]["brier"],
            "ref_logloss": blk["reference"]["logloss"], "ref_diracc": blk["reference"]["diracc"],
        })

    vol = regime.get("vol") or {}
    emit("vol", "low", vol.get("low", {}))
    emit("vol", "high", vol.get("high", {}))
    for bucket, blk in (regime.get("tau") or {}).items():
        emit("tau", bucket, blk)
    for bucket, blk in (regime.get("source") or {}).items():
        emit("source", bucket, blk)
    for bucket, blk in (regime.get("depth_cov") or {}).items():
        emit("depth_cov", bucket, blk)
    return rows


def _flatten_contribution_rows(scope: str, target: str, contributions: dict) -> list[dict]:
    rows: list[dict] = []
    for comparison, blocks in contributions.items():
        for subset in ("overall", "covered", "uncovered"):
            b = blocks.get(subset) or {}
            if b.get("n", 0) == 0:
                continue
            rows.append({
                "scope": scope, "target": target, "comparison": comparison, "subset": subset,
                "n": b["n"], "brier_a": b["brier_a"], "brier_b": b["brier_b"],
                "brier_delta": b["brier_delta"], "diracc_a": b["diracc_a"],
                "diracc_b": b["diracc_b"], "diracc_delta": b["diracc_delta"],
            })
    return rows


# ===========================================================================
# per-(scope, target, variant) driver
# ===========================================================================
def _run_variant_short(paths, sub_scope, scope, target, variant, feature_names, *,
                       train_weeks, test_days, purge_secs, seed, params, max_boost_round,
                       early_stopping_rounds, inner_val_frac, min_windows, health,
                       bench_ctx, run_shuffle):
    """Walk-forward + final model + own-label/abstention/regime analyses + optional
    harness + shuffled control for one (scope, target, variant). Returns
    ``(block, shuffled_block, fold_rows, oos)`` — ``oos`` feeds the depth-contribution."""
    spec = TARGETS_SHORT[target]
    label, horizon = spec["label"], spec["horizon_secs"]
    out_dir = paths.out_dir / "learn_short_gbt"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scope}_{target}_{variant}"

    sub = sub_scope[sub_scope[label].notna()].reset_index(drop=True)
    if sub.empty:
        block = {
            "label": label, "horizon_secs": horizon, "feature_set": variant,
            "feature_names": list(feature_names), "mode": "insufficient",
            "n_folds": 0, "n_oos": 0, "pooled": _nan_metrics(), "reference_pooled": _nan_metrics(),
            "majority_pooled": lm.majority_baseline(np.empty(0)),
            "num_boost_round": 0, "best_iteration_median": None,
            "final_model_file": None, "booster_file": None, "final_model": {},
            "feature_importance": {"gain": {}, "split": {}},
        }
        return block, {}, [], pd.DataFrame(columns=_OOS_COLS_SHORT)

    y = sub[label].astype("float64").to_numpy()
    info = _label_info_ms(sub, horizon)
    purge_ms = int(purge_secs) * 1000

    oos, folds, mode = _run_walk_forward_short(
        sub, y, info, feature_names=feature_names, train_weeks=train_weeks, test_days=test_days,
        purge_ms=purge_ms, params=params, max_boost_round=max_boost_round,
        early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac,
    )
    pooled = _fold_metrics(oos["p_up"].to_numpy(float), oos["y_true"].to_numpy(float)) \
        if len(oos) else _nan_metrics()

    _harness_grid(oos).to_parquet(out_dir / f"predictions_{tag}.parquet", engine="pyarrow", index=False)
    if len(oos):
        oos.to_parquet(out_dir / f"oos_{tag}.parquet", engine="pyarrow", index=False)

    # Deployable model (median fold best_iteration, else a fresh inner-val early-stop).
    booster, model_meta, num_rounds = _final_model_short(
        sub, y, info, feature_names=feature_names, feature_set=variant, train_weeks=train_weeks,
        params=params, max_boost_round=max_boost_round, early_stopping_rounds=early_stopping_rounds,
        inner_val_frac=inner_val_frac, purge_ms=purge_ms, rounds=_median_best_iter(folds),
        target=target, label=label, horizon_secs=horizon,
    )
    model_meta.update({"seed": int(seed), "model_file": f"model_{tag}.txt",
                       "days_span_days": [int(sub["window_open_ms"].min() // MS_PER_DAY),
                                          int(sub["window_open_ms"].max() // MS_PER_DAY)]})
    gbt.save(booster, out_dir / f"model_{tag}.txt", num_iteration=num_rounds)
    (out_dir / f"model_{tag}.json").write_text(
        json.dumps(model_meta, indent=2, default=eh._json_default), encoding="utf-8")

    importance = model_meta["feature_importance"]
    fi_rows = sorted(
        ({"feature": n, "gain": importance["gain"][n], "split": importance["split"][n]}
         for n in feature_names), key=lambda r: r["gain"], reverse=True)
    pd.DataFrame(fi_rows, columns=["feature", "gain", "split"]).to_csv(
        out_dir / f"feature_importance_{tag}.csv", index=False)

    block: dict = {
        "label": label, "horizon_secs": horizon, "feature_set": variant,
        "feature_names": list(feature_names), "mode": mode,
        "n_folds": int(sum(1 for f in folds if not f["skipped"])),
        "n_oos": int(len(oos)),
        "pooled": pooled,
        "reference_pooled": _nan_metrics(),
        # Always-predict-the-common-side no-skill baseline — model skill reads as lift over it.
        "majority_pooled": (lm.majority_baseline(oos["y_true"].to_numpy(float))
                            if len(oos) else lm.majority_baseline(np.empty(0))),
        "num_boost_round": int(num_rounds),
        "best_iteration_median": _median_best_iter(folds),
        "final_model_file": f"model_{tag}.json",
        "booster_file": f"model_{tag}.txt",
        "final_model": {"n_train": model_meta["n_train"], "train_span_days": model_meta["train_span_days"],
                        "class_balance_pos_frac": model_meta["class_balance_pos_frac"]},
        "feature_importance": importance,
    }

    if len(oos):
        # Own-label calibration.
        reliability = lm.reliability_table(oos["p_up"].to_numpy(float), oos["y_true"].to_numpy(float))
        block["own_label_reliability"] = reliability
        pd.DataFrame(reliability).to_csv(out_dir / f"reliability_{tag}.csv", index=False)

        # Regime + naive Φ(z) reference (on the finite-p_up_model analysis frame).
        af = _analysis_frame_short(oos)
        block["reference_pooled"] = (
            _fold_metrics(af["p_up_model"].to_numpy(float), af["y_true"].to_numpy(float))
            if len(af) else _nan_metrics())
        regime = {"n": int(len(af)), "excluded_no_finite": int(len(oos) - len(af)), **_regime_blocks(af)}
        block["regime"] = regime
        pd.DataFrame(_flatten_regime_rows(regime),
                     columns=["breakdown", "bucket", "n", "model_brier", "model_logloss",
                              "model_diracc", "ref_brier", "ref_logloss", "ref_diracc"]).to_csv(
            out_dir / f"regime_{tag}.csv", index=False)

        # Abstention curve.
        abst = _abstention_curve(oos)
        block["abstention"] = abst
        pd.DataFrame(abst, columns=["threshold", "coverage", "n", "diracc", "brier"]).to_csv(
            out_dir / f"abstention_{tag}.csv", index=False)

    if bench_ctx is not None and len(oos):
        hm = score_through_harness(
            paths, bench_ctx, oos, subdir="learn_short_gbt", out_name=f"harness_{tag}",
            model_label=f"short-gbt-{tag}", title=f"short-horizon gbt — {tag}",
            min_windows=min_windows, health=health)
        block["harness"] = _harness_summary(hm)

    fold_rows = [{"scope": scope, "target": target, "variant": variant, "control": False, **f}
                 for f in folds]

    # ---- shuffled-label control (proves the pipeline can't cheat) ----
    shuffled: dict = {}
    if run_shuffle and len(oos):
        rng = np.random.default_rng(seed)
        y_shuf = y[rng.permutation(len(y))]
        oos_s, folds_s, _ = _run_walk_forward_short(
            sub, y_shuf, info, feature_names=feature_names, train_weeks=train_weeks,
            test_days=test_days, purge_ms=purge_ms, params=params, max_boost_round=max_boost_round,
            early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac)
        pooled_s = _fold_metrics(oos_s["p_up"].to_numpy(float), oos_s["y_true"].to_numpy(float)) \
            if len(oos_s) else _nan_metrics()
        p_oos = float(np.mean(oos_s["y_true"].to_numpy(float))) if len(oos_s) else float("nan")
        chance_brier = p_oos * (1.0 - p_oos)
        base_rate_acc = max(p_oos, 1.0 - p_oos)
        _harness_grid(oos_s).to_parquet(
            out_dir / f"predictions_{tag}_shuffled.parquet", engine="pyarrow", index=False)
        reliably_better = None
        if bench_ctx is not None and len(oos_s):
            hm_s = score_through_harness(
                paths, bench_ctx, oos_s, subdir="learn_short_gbt", out_name=f"harness_{tag}_shuffled",
                model_label=f"short-gbt-{tag}-shuffled", title=f"short-horizon gbt shuffled — {tag}",
                min_windows=min_windows, health=health)
            reliably_better = _harness_summary(hm_s)["vs_formula"]["reliably_better"]
        collapsed = bool(
            np.isfinite(pooled_s["brier"]) and np.isfinite(pooled["brier"])
            and pooled_s["brier"] >= chance_brier - 0.03
            and pooled["brier"] < pooled_s["brier"]
            and (reliably_better is not True))
        shuffled = {
            "pooled": pooled_s, "chance_brier": chance_brier, "base_rate_acc": base_rate_acc,
            "real_pooled_brier": pooled["brier"], "real_pooled_diracc": pooled["diracc"],
            "harness_vs_formula_reliably_better": reliably_better, "collapsed": collapsed,
        }
        fold_rows += [{"scope": scope, "target": target, "variant": variant, "control": True, **f}
                      for f in folds_s]

    return block, shuffled, fold_rows, oos


# ===========================================================================
# stage entry
# ===========================================================================
def learn_short_gbt(
    paths: Paths,
    *,
    days: int = DEFAULT_DAYS,
    train_weeks: int = DEFAULT_TRAIN_WEEKS,
    test_days: int = DEFAULT_TEST_DAYS,
    purge_secs: int | None = None,
    seed: int = DEFAULT_SEED,
    targets: tuple[str, ...] = ("fwd10", "fwd15"),
    variants: tuple[str, ...] = ("base", "depth", "full"),
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    series: str | None = None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    health: str = "ready",
    run_harness: bool = False,
    run_shuffle: bool = True,
    external_flow: bool = True,
    flow_tolerance_ms: int = FLOW_ASOF_TOLERANCE_MS,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    reg_alpha: float = DEFAULT_REG_ALPHA,
    max_boost_round: int = DEFAULT_MAX_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    inner_val_frac: float = DEFAULT_INNER_VAL_FRAC,
    num_threads: int = DEFAULT_NUM_THREADS,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> dict:
    """Train + walk-forward-evaluate the short-horizon GBT challenger over each source
    scope; write ``out/learn_short_gbt/``. ``since_ms``/``until_ms`` bound the (optional)
    harness journal read."""
    series_filter = _parse_series(series)
    mat = _load_short_matrix(paths, days=days, series=series_filter)
    flow_stats = {"joined": False, "n_asof_matched": 0, "frac_asof_matched": None,
                  "n_flow_present": 0, "frac_flow_present": None, "tolerance_ms": int(flow_tolerance_ms)}
    if external_flow and not mat.empty:
        mat, flow_stats = _join_external_flow(mat, paths, tolerance_ms=flow_tolerance_ms)
    # Canonical, deterministic row order regardless of the join.
    if not mat.empty:
        mat = mat.sort_values(["series", "window_open_ms", "sample_ts_ms"],
                              kind="stable").reset_index(drop=True)

    feature_sets = _resolve_feature_sets(external_flow)
    params = gbt.default_params(
        seed, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
        min_child_samples=min_child_samples, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        num_threads=num_threads)

    out_dir = paths.out_dir / "learn_short_gbt"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "config": {
            "days": days, "train_weeks": train_weeks, "test_days": test_days,
            "purge_secs_override": purge_secs, "seed": seed, "targets": list(targets),
            "variants": list(variants), "scopes": list(scopes), "series": series,
            "min_windows": min_windows, "health": health, "run_harness": run_harness,
            "external_flow": external_flow,
        },
        "seed": int(seed),
        "params": params,
        "features": feature_sets,
        "external_flow": flow_stats,
        "n_rows_loaded": int(len(mat)),
        "caveats": CAVEATS,
        "scopes": {},
    }
    if mat.empty:
        result["error"] = "short_horizon.parquet had no rows in scope (check --days/--series)."
        (out_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
        return result

    result["date_span"] = {
        "min_open_ms": int(mat["window_open_ms"].min()),
        "max_open_ms": int(mat["window_open_ms"].max()),
        "min_day": int(mat["window_open_ms"].min() // MS_PER_DAY),
        "max_day": int(mat["window_open_ms"].max() // MS_PER_DAY),
        "distinct_days": int(np.unique(mat["window_open_ms"].to_numpy(np.int64) // MS_PER_DAY).size),
    }

    bench_ctx = eh.load_benchmarks(paths, since_ms=since_ms, until_ms=until_ms) if run_harness else None

    all_folds: list[dict] = []
    contrib_rows: list[dict] = []
    for scope in scopes:
        if scope == "all":
            sub_scope = mat
        elif scope in ("chainlink", "binance_proxy"):
            sub_scope = mat[mat["label_source"] == scope].reset_index(drop=True)
        else:
            raise ValueError(f"unknown scope {scope!r} (expected all|chainlink|binance_proxy)")
        scope_result: dict = {"targets": {}, "shuffled_control": {}, "depth_contribution": {}}
        for target in targets:
            if target not in TARGETS_SHORT:
                raise ValueError(f"unknown target {target!r} (expected one of {list(TARGETS_SHORT)})")
            p_secs = purge_secs if purge_secs is not None else TARGETS_SHORT[target]["horizon_secs"]
            scope_result["targets"][target] = {}
            scope_result["shuffled_control"][target] = {}
            variant_oos: dict[str, pd.DataFrame] = {}
            for variant in variants:
                if variant not in feature_sets:
                    raise ValueError(f"unknown variant {variant!r} (expected one of {VARIANT_ORDER})")
                block, shuffled, fold_rows, oos = _run_variant_short(
                    paths, sub_scope, scope, target, variant, feature_sets[variant],
                    train_weeks=train_weeks, test_days=test_days, purge_secs=p_secs, seed=seed,
                    params=params, max_boost_round=max_boost_round,
                    early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac,
                    min_windows=min_windows, health=health, bench_ctx=bench_ctx, run_shuffle=run_shuffle)
                block["purge_secs"] = p_secs
                scope_result["targets"][target][variant] = block
                if shuffled:
                    scope_result["shuffled_control"][target][variant] = shuffled
                all_folds.extend(fold_rows)
                variant_oos[variant] = oos

            contrib: dict = {}
            if "base" in variant_oos and "depth" in variant_oos:
                contrib["base_vs_depth"] = _pair_contribution(
                    variant_oos["base"], variant_oos["depth"], "depth_feat_covered")
            if "depth" in variant_oos and "full" in variant_oos:
                contrib["depth_vs_full"] = _pair_contribution(
                    variant_oos["depth"], variant_oos["full"], "pm_feat_covered")
            if "base" in variant_oos and "full" in variant_oos:
                contrib["base_vs_full"] = _pair_contribution(
                    variant_oos["base"], variant_oos["full"], "depth_feat_covered")
            scope_result["depth_contribution"][target] = contrib
            contrib_rows.extend(_flatten_contribution_rows(scope, target, contrib))
        result["scopes"][scope] = scope_result

    if all_folds:
        fold_cols = ["scope", "target", "variant", "control", "fold_idx", "test_start_day",
                     "test_end_day", "n_train", "n_test", "n_test_windows", "best_iteration",
                     "n_inner_train", "n_inner_val", "skipped", "pos_frac", "brier", "logloss", "diracc"]
        pd.DataFrame(all_folds, columns=fold_cols).to_csv(out_dir / "folds.csv", index=False)
    if contrib_rows:
        pd.DataFrame(contrib_rows, columns=["scope", "target", "comparison", "subset", "n",
                                            "brier_a", "brier_b", "brier_delta", "diracc_a",
                                            "diracc_b", "diracc_delta"]).to_csv(
            out_dir / "depth_contribution.csv", index=False)

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    return result


def _fmt(x, *args) -> str:
    return eh.fmt(x, *args) if x is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "learn_short_gbt")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"trailing days of short_horizon to use (0 = all; default {DEFAULT_DAYS})")
    parser.add_argument("--train-weeks", type=int, default=DEFAULT_TRAIN_WEEKS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--purge-secs", type=int, default=None,
                        help="purge gap at each boundary in seconds (default: the label horizon 10/15)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--targets", default="fwd10,fwd15",
                        help="comma-separated targets: fwd10,fwd15 (default both)")
    parser.add_argument("--variants", default="base,depth,full",
                        help="comma-separated feature sets: base,depth,full (default all three)")
    parser.add_argument("--scopes", default="all,chainlink",
                        help="comma-separated source scopes: all,chainlink,binance_proxy (default all,chainlink)")
    parser.add_argument("--series", default=None, help="comma-separated series filter, e.g. BTC-5m,ETH-5m")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS)
    parser.add_argument("--health", choices=("ready", "all"), default="ready")
    parser.add_argument("--with-harness", action="store_true",
                        help="also score p_up against the window outcome through the harness "
                             "(a slow journal read; a different question than the 10/15s label)")
    parser.add_argument("--no-shuffle-control", action="store_true", help="skip the shuffled-label control")
    parser.add_argument("--no-external-flow", action="store_true",
                        help="skip the feature_set flow join (base = native short-horizon features only)")
    parser.add_argument("--flow-tolerance-ms", type=int, default=FLOW_ASOF_TOLERANCE_MS,
                        help=f"as-of staleness tolerance for the flow join (default {FLOW_ASOF_TOLERANCE_MS})")
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_NUM_LEAVES)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-child-samples", type=int, default=DEFAULT_MIN_CHILD_SAMPLES)
    parser.add_argument("--reg-lambda", type=float, default=DEFAULT_REG_LAMBDA)
    parser.add_argument("--reg-alpha", type=float, default=DEFAULT_REG_ALPHA)
    parser.add_argument("--max-boost-round", type=int, default=DEFAULT_MAX_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--inner-val-frac", type=float, default=DEFAULT_INNER_VAL_FRAC)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    scopes = tuple(s.strip() for s in args.scopes.split(",") if s.strip())

    print(f"[learn_short_gbt] short_horizon={paths.table('short_horizon')}")
    print(f"[learn_short_gbt] out          ={paths.out_dir / 'learn_short_gbt'}")
    try:
        m = learn_short_gbt(
            paths, days=args.days, train_weeks=args.train_weeks, test_days=args.test_days,
            purge_secs=args.purge_secs, seed=args.seed, targets=targets, variants=variants,
            scopes=scopes, series=args.series, min_windows=args.min_windows, health=args.health,
            run_harness=args.with_harness, run_shuffle=not args.no_shuffle_control,
            external_flow=not args.no_external_flow, flow_tolerance_ms=args.flow_tolerance_ms,
            num_leaves=args.num_leaves, max_depth=args.max_depth, learning_rate=args.learning_rate,
            min_child_samples=args.min_child_samples, reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha, max_boost_round=args.max_boost_round,
            early_stopping_rounds=args.early_stopping_rounds, inner_val_frac=args.inner_val_frac,
            num_threads=args.num_threads, since_ms=since_ms, until_ms=until_ms)
    except ParquetNotReady as exc:
        print(f"[learn_short_gbt] {exc}")
        return 1

    if "error" in m:
        print(f"[learn_short_gbt] {m['error']}")
        return 1

    ds = m["date_span"]
    ef = m["external_flow"]
    print(f"[learn_short_gbt] rows={m['n_rows_loaded']:,} over {ds['distinct_days']} days "
          f"(day {ds['min_day']}→{ds['max_day']}), seed={m['seed']}, "
          f"flow-join={'on' if ef['joined'] else 'off'}"
          + (f" ({ef['frac_asof_matched']:.0%} as-of, {ef['frac_flow_present']:.0%} flow present)"
             if ef.get("frac_asof_matched") is not None else ""))
    for scope, sr in m["scopes"].items():
        for target, variants_blk in sr["targets"].items():
            for variant, blk in variants_blk.items():
                p = blk["pooled"]
                print(f"[learn_short_gbt] {scope}/{target}/{variant:<5} "
                      f"[{blk['mode']}, {blk['n_folds']} fold(s), {blk['n_oos']:,} OOS, "
                      f"rounds={blk['num_boost_round']}]: Brier={_fmt(p['brier'])} "
                      f"log-loss={_fmt(p['logloss'])} dir-acc={_fmt(p['diracc'])}")
                mj = blk.get("majority_pooled") or {}
                if mj.get("n"):
                    print(f"[learn_short_gbt]   majority baseline: dir-acc={_fmt(mj['diracc'])} "
                          f"Brier={_fmt(mj['brier'])}  → model lift dir-acc="
                          f"{_fmt(p['diracc'] - mj['diracc'])} Brier={_fmt(mj['brier'] - p['brier'])}")
                abst = blk.get("abstention") or []
                mid = next((r for r in abst if r["threshold"] == 0.05), None)
                if mid:
                    print(f"[learn_short_gbt]   abstain @|p-.5|≥0.05: coverage={_fmt(mid['coverage'])} "
                          f"dir-acc={_fmt(mid['diracc'])}")
                sc = sr["shuffled_control"].get(target, {}).get(variant)
                if sc:
                    ps = sc["pooled"]
                    print(f"[learn_short_gbt]   shuffled control: Brier={_fmt(ps['brier'])} "
                          f"(chance {_fmt(sc['chance_brier'])}) → collapsed={sc['collapsed']}")
            dc = sr["depth_contribution"].get(target, {})
            for comp in ("base_vs_depth", "depth_vs_full", "base_vs_full"):
                ov = (dc.get(comp) or {}).get("overall") or {}
                if ov.get("n", 0):
                    print(f"[learn_short_gbt]   {scope}/{target} {comp}: "
                          f"ΔBrier={_fmt(ov['brier_delta'])} Δdir-acc={_fmt(ov['diracc_delta'])} "
                          f"(n={ov['n']:,})")

    n_models = sum(1 for sr in m["scopes"].values() for vb in sr["targets"].values() for _ in vb)
    print(f"[learn_short_gbt] wrote {paths.out_dir / 'learn_short_gbt'} ({n_models} model(s): "
          "model_*.txt/.json, predictions_*.parquet, feature_importance/regime/abstention/"
          "reliability csv, depth_contribution.csv)")

    primary = scopes[0] if scopes else None
    ok = primary is not None and all(
        blk["n_oos"] > 0
        for target_blk in m["scopes"].get(primary, {}).get("targets", {}).values()
        for blk in target_blk.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
