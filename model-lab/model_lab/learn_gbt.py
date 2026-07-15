"""Stage — learn_gbt. The second challenger: walk-forward gradient-boosted trees.

Trains a **LightGBM** binary classifier on the same 16 engineered features, the
same two targets (``fwd_up_30s`` / ``outcome_up``), the same strict walk-forward
scheme, purge gaps, and shuffled-label control as the logistic stage
(:mod:`model_lab.learn`), and reports through the same evaluation harness. It is an
**opt-in** stage: LightGBM is a compiled OpenMP library (and pulls ``scipy``), so it
ships as an extra — ``pip install -r requirements-gbt.txt`` — and the core pipeline
stays lightgbm-free.

**Two variants of using the formula.** Adding the formula probability ``Φ(z)`` as a
literal input column would be a *no-op* for trees (they are invariant to monotone
transforms of ``z``, which is already a feature). So the formula is injected two
ways and compared, on the **outcome** target:

- **plain** — a binary GBT on the 16 features, with no formula prior;
- **residual** — the formula is supplied as a LightGBM ``init_score`` base margin
  (``logit(Φ(z))``), so the trees learn only the **correction** over the formula.
  The disagreement analysis then *is* the learned residual.

(``fwd_up_30s`` gets the plain variant only — ``Φ(z)`` predicts the window outcome,
not the 30-second direction.)

**Tuning is forward-chained inside the training window, never on the test period.**
Each fold carves an inner forward-chained validation split from the *tail* of its
training window (purged from the inner-train by the label horizon), early-stops the
number of boosting rounds on it, then **refits on the full training window** for
that many rounds and predicts the OOS test block.

**Beyond the harness scorecard** — the GBT stage answers *"does GBT add anything the
formula lacks, and where"*: per-feature importances (gain + split), a volatility-
regime split (low vs high ``sigma_slow``), a time-remaining breakdown, and a
**residual analysis** — where the GBT disagrees with the formula, bucketed by time
remaining × ``|z|``, and who is right there.

**Artifacts** (``out/learn_gbt/``): ``model_<target>_<variant>.txt`` (native
booster) + ``model_<target>_<variant>.json`` (metadata + importances), the OOS
``predictions_<target>_<variant>.parquet`` grids (+ enriched ``oos_*.parquet``),
``harness_<target>_<variant>/`` reports, ``feature_importance_*.csv``,
``regime_*.csv``, ``residual_*.csv``, ``metrics.json`` and ``folds.csv``.

Run::

    python -m model_lab.learn_gbt                          # both targets, last 90 days
    python -m model_lab.learn_gbt --targets outcome --no-harness   # quick iteration
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import eval_harness as eh
from .config import ParquetNotReady, Paths, resolve_bounds, resolve_paths, stage_parser
from .feature_set import FEATURE_NAMES, WINDOW_SECS
from .learn_common import (
    MS_PER_DAY, TARGETS, _fold_metrics, _fold_schedule, _harness_grid, _harness_summary,
    _label_info_ms, _load_matrix, _parse_series, score_through_harness,
)
from .lib import gbt
from .lib import math as lm

DEFAULT_SEED = 20260704
DEFAULT_DAYS = 90
DEFAULT_TRAIN_WEEKS = 4
DEFAULT_TEST_DAYS = 7
DEFAULT_MIN_WINDOWS = eh.DEFAULT_MIN_WINDOWS

# GBT hyperparameter defaults — modest depth, strong regularization.
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

# The formula (Φ(z)) predicts the window outcome, so the formula-anchored residual
# variant applies only to `outcome`; fwd30 gets the plain variant.
VARIANTS_BY_TARGET = {"outcome": ("plain", "residual"), "fwd30": ("plain",)}

# Residual analysis: a GBT/formula disagreement is |p_gbt − Φ(z)| beyond this.
DISAGREE_DELTA = 0.05
# |z| buckets for the residual cross-tab (edges between the buckets).
Z_ABS_EDGES = [0.5, 1.0, 2.0]
Z_ABS_LABELS = ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,inf)"]

CAVEATS = [
    "LightGBM determinism is within-machine only (deterministic=True + force_row_wise + "
    "single thread + pinned seeds); the exact lightgbm version is pinned in requirements-gbt.txt.",
    "The formula-anchored 'residual' variant learns a correction on top of Φ(z) via init_score; "
    "the 'plain' variant is an independent GBT — the residual analysis contrasts each with the formula.",
    "Boosting rounds are chosen by inner forward-chained validation (a purged tail of each training "
    "window), then refit on the full window — the round count is validated, never fit on the test.",
    "Binance-proxy outcomes approximate the true Chainlink resolution (~0.987 agreement); the harness "
    "scores only the real journal windows; regime/residual analyses drop warmup rows with no finite z.",
    "One pooled model per fold across all series; hour_of_day is a plain numeric feature.",
]


# ===========================================================================
# formula prior + inner-validation round selection
# ===========================================================================
def _formula_p(z: np.ndarray) -> np.ndarray:
    """Formula probability Φ(z) (NaN where z is NaN — warmup)."""
    return lm.norm_cdf(np.asarray(z, dtype=float))


def _base_margin(z: np.ndarray) -> np.ndarray:
    """The residual variant's init_score = logit(Φ(z)); warmup (NaN z) → 0 (no prior)."""
    p = np.clip(lm.norm_cdf(np.nan_to_num(np.asarray(z, dtype=float), nan=0.0)), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _inner_best_iteration(X, y, ts, label_info, base, *, params, max_boost_round,
                          early_stopping_rounds, inner_val_frac, purge_ms):
    """Pick the boosting-round count by an inner forward-chained validation split of
    a training block: the latest ``inner_val_frac`` (by ``ts``) is validation, and
    the inner-train is purged of rows whose label reaches within ``purge_ms`` of the
    inner-val start (same rule as the fold purge, but ms-granular). Returns
    ``(best_iteration, n_inner_train, n_inner_val)``; falls back to a modest fixed
    round count when a split can't be formed."""
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
    b_tr = base[tr_idx] if base is not None else None
    b_val = base[val_idx] if base is not None else None
    _, best = gbt.fit(
        X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], params=params,
        max_boost_round=max_boost_round, early_stopping_rounds=early_stopping_rounds,
        init_score_tr=b_tr, init_score_val=b_val,
    )
    return int(best), int(len(tr_idx)), int(n_val)


# ===========================================================================
# walk-forward (GBT)
# ===========================================================================
_OOS_COLS = ["series", "window_open_ms", "sample_ts_ms", "p_up", "y_true",
             "z", "sigma_slow", "seconds_remaining", "dur"]


def _run_walk_forward_gbt(sub, y, label_info, *, variant, train_weeks, test_days, purge_ms,
                          params, max_boost_round, early_stopping_rounds, inner_val_frac):
    """Slide a trailing train window over ``sub``; per fold pick rounds on an inner
    forward-chained val, refit on the full train window, predict OOS test. Returns
    ``(oos_df, fold_records, mode)`` where ``oos_df`` carries the harness grid keys +
    ``p_up``/``y_true`` and the analysis columns (``z``/``sigma_slow``/τ/``dur``)."""
    open_ms = sub["window_open_ms"].to_numpy(np.int64)
    close_ms = sub["window_close_ms"].to_numpy(np.int64)
    ts = sub["sample_ts_ms"].to_numpy(np.int64)
    day = open_ms // MS_PER_DAY
    X = sub[FEATURE_NAMES].to_numpy(dtype=float)
    z = sub["z"].to_numpy(dtype=float)
    sigma_slow = sub["sigma_slow"].to_numpy(dtype=float)
    seconds_remaining = sub["seconds_remaining"].to_numpy(dtype=float)
    dur = (close_ms - open_ms) / 1000.0
    base = _base_margin(z) if variant == "residual" else None
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
        b_tr = base[tr] if base is not None else None
        best, n_it, n_iv = _inner_best_iteration(
            X[tr], y[tr], ts[tr], label_info[tr], b_tr, params=params,
            max_boost_round=max_boost_round, early_stopping_rounds=early_stopping_rounds,
            inner_val_frac=inner_val_frac, purge_ms=purge_ms,
        )
        booster = gbt.refit(X[tr], y[tr], params=params, num_boost_round=best, init_score=b_tr)
        prob = gbt.predict_proba(
            booster, X[te], num_iteration=best,
            init_score=(base[te] if base is not None else None), num_threads=params["num_threads"],
        )
        y_te = y[te]
        part = pd.DataFrame({
            "series": sub["series"].to_numpy()[te],
            "window_open_ms": open_ms[te],
            "sample_ts_ms": ts[te],
            "p_up": prob,
            "y_true": y_te,
            "z": z[te],
            "sigma_slow": sigma_slow[te],
            "seconds_remaining": seconds_remaining[te],
            "dur": dur[te],
        })
        oos_parts.append(part)
        m = _fold_metrics(prob, y_te)
        n_win = int(part.groupby(["series", "window_open_ms"]).ngroups)
        records.append({
            "fold_idx": i, "test_start_day": int(test_start), "test_end_day": int(test_end),
            "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_windows": n_win,
            "best_iteration": int(best), "n_inner_train": int(n_it), "n_inner_val": int(n_iv),
            "skipped": False, **m,
        })

    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame(columns=_OOS_COLS)
    if len(oos):
        oos["p_up_formula"] = _formula_p(oos["z"].to_numpy(dtype=float))
    else:
        oos["p_up_formula"] = pd.Series(dtype=float)
    return oos, records, mode


def _median_best_iter(records: list[dict]) -> int | None:
    vals = [r["best_iteration"] for r in records if not r["skipped"]
            and r["best_iteration"] is not None and np.isfinite(r["best_iteration"])]
    return int(np.median(vals)) if vals else None


def _final_model(sub, y, label_info, *, variant, train_weeks, params, max_boost_round,
                 early_stopping_rounds, inner_val_frac, purge_ms, rounds, target, label, horizon_secs):
    """Fit the deployable booster on the most recent ``train_weeks`` of data, using
    ``rounds`` (the fold median best_iteration) if given, else a fresh inner-val
    early-stop on the final window. Returns ``(booster, meta, num_rounds)``."""
    open_ms = sub["window_open_ms"].to_numpy(np.int64)
    day = open_ms // MS_PER_DAY
    hi = int(day.max())
    start_day = hi - int(train_weeks) * 7 + 1
    mask = day >= start_day
    if int(mask.sum()) == 0:
        mask = np.ones(len(day), dtype=bool)
        start_day = int(day.min())
    idx = np.flatnonzero(mask)
    X = sub[FEATURE_NAMES].to_numpy(dtype=float)[idx]
    yt = y[idx]
    z = sub["z"].to_numpy(dtype=float)[idx]
    ts = sub["sample_ts_ms"].to_numpy(np.int64)[idx]
    li = label_info[idx]
    base = _base_margin(z) if variant == "residual" else None
    num_rounds = rounds
    if num_rounds is None:
        num_rounds, _, _ = _inner_best_iteration(
            X, yt, ts, li, base, params=params, max_boost_round=max_boost_round,
            early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac,
            purge_ms=purge_ms,
        )
    booster = gbt.refit(X, yt, params=params, num_boost_round=num_rounds, init_score=base)
    importance = gbt.feature_importance(booster, list(FEATURE_NAMES))
    meta = {
        "target": target,
        "label": label,
        "horizon_secs": horizon_secs,
        "variant": variant,
        "model": "lightgbm_gbdt",
        "params": params,
        "feature_names": list(FEATURE_NAMES),
        "num_boost_round": int(num_rounds),
        "feature_importance": importance,
        "n_train": int(len(idx)),
        "train_span_days": [start_day, hi],
        "train_span_open_ms": [int(open_ms[idx].min()), int(open_ms[idx].max())],
        "class_balance_pos_frac": float(np.mean(yt)),
    }
    return booster, meta, int(num_rounds)


# ===========================================================================
# GBT-specific analyses (outcome target)
# ===========================================================================
def _pair_metrics(prob: np.ndarray, out: np.ndarray) -> dict:
    return {"brier": lm.brier_score(prob, out), "logloss": lm.log_loss(prob, out),
            "diracc": lm.directional_accuracy(prob, out)}


def _analysis_frame(oos: pd.DataFrame) -> pd.DataFrame:
    """The rows on which the formula-vs-GBT analysis is defined: finite z (so Φ(z)
    exists) and finite sigma_slow (past warmup), with a tau bucket attached."""
    df = oos.copy()
    finite = np.isfinite(df["z"].to_numpy(float)) & np.isfinite(df["sigma_slow"].to_numpy(float))
    df = df[finite].reset_index(drop=True)
    if len(df):
        df["tau_bucket"] = eh.tau_bucket(
            df["seconds_remaining"].to_numpy(float), df["dur"].to_numpy(float))
    else:
        df["tau_bucket"] = pd.Series(dtype=object)
    return df


def _regime_analysis(oos: pd.DataFrame) -> dict:
    """GBT vs formula (Φ(z)) scores split by volatility regime (median sigma_slow
    split) and by time-remaining bucket."""
    df = _analysis_frame(oos)
    excluded = int(len(oos) - len(df))
    if df.empty:
        return {"n": 0, "excluded_no_finite_z": excluded, "vol": {}, "tau": {}}
    out = df["y_true"].to_numpy(float)
    gbt_p = df["p_up"].to_numpy(float)
    f_p = df["p_up_formula"].to_numpy(float)
    med = float(np.median(df["sigma_slow"].to_numpy(float)))

    def block(mask) -> dict:
        if not np.any(mask):
            return {"n": 0}
        return {"n": int(mask.sum()),
                "gbt": _pair_metrics(gbt_p[mask], out[mask]),
                "formula": _pair_metrics(f_p[mask], out[mask])}

    sig = df["sigma_slow"].to_numpy(float)
    vol = {"median_sigma_slow": med, "low": block(sig <= med), "high": block(sig > med)}
    tau = {}
    tb = df["tau_bucket"].to_numpy()
    for bucket in eh.TAU_ORDER:
        m = tb == bucket
        if np.any(m):
            tau[bucket] = block(m)
    return {"n": int(len(df)), "excluded_no_finite_z": excluded, "vol": vol, "tau": tau}


def _residual_analysis(oos: pd.DataFrame, *, delta: float) -> dict:
    """Where the GBT disagrees with the formula (|p_gbt − Φ(z)| > delta), bucketed by
    time-remaining × |z|: how often, by how much, and who is right (Brier + dir-acc)."""
    df = _analysis_frame(oos)
    if df.empty:
        return {"delta": delta, "n_total": 0, "n_disagree": 0, "frac_disagree": float("nan"),
                "overall": {}, "cells": []}
    out = df["y_true"].to_numpy(float)
    gbt_p = df["p_up"].to_numpy(float)
    f_p = df["p_up_formula"].to_numpy(float)
    d = gbt_p - f_p
    disagree = np.abs(d) > float(delta)
    n_total, n_dis = int(len(df)), int(disagree.sum())

    def summary(mask) -> dict:
        if not np.any(mask):
            return {"n": 0}
        g, f = _pair_metrics(gbt_p[mask], out[mask]), _pair_metrics(f_p[mask], out[mask])
        return {"n": int(mask.sum()), "mean_abs_d": float(np.mean(np.abs(d[mask]))),
                "gbt": g, "formula": f, "gbt_better": bool(g["brier"] < f["brier"])}

    z_abs = np.abs(df["z"].to_numpy(float))
    z_bucket_idx = np.digitize(z_abs, Z_ABS_EDGES, right=False)  # 0..len(edges)
    tb = df["tau_bucket"].to_numpy()
    cells: list[dict] = []
    for bucket in eh.TAU_ORDER:
        for zi, zlab in enumerate(Z_ABS_LABELS):
            m = disagree & (tb == bucket) & (z_bucket_idx == zi)
            s = summary(m)
            if s["n"] == 0:
                continue
            cells.append({"tau_bucket": bucket, "z_bucket": zlab, **s})
    return {"delta": float(delta), "n_total": n_total, "n_disagree": n_dis,
            "frac_disagree": (n_dis / n_total) if n_total else float("nan"),
            "overall": summary(disagree), "cells": cells}


def _flatten_regime_rows(regime: dict) -> list[dict]:
    rows: list[dict] = []

    def emit(breakdown: str, bucket: str, blk: dict) -> None:
        if not blk or blk.get("n", 0) == 0:
            return
        rows.append({
            "breakdown": breakdown, "bucket": bucket, "n": blk["n"],
            "gbt_brier": blk["gbt"]["brier"], "gbt_logloss": blk["gbt"]["logloss"],
            "gbt_diracc": blk["gbt"]["diracc"], "formula_brier": blk["formula"]["brier"],
            "formula_logloss": blk["formula"]["logloss"], "formula_diracc": blk["formula"]["diracc"],
        })

    vol = regime.get("vol") or {}
    emit("vol", "low", vol.get("low", {}))
    emit("vol", "high", vol.get("high", {}))
    for bucket, blk in (regime.get("tau") or {}).items():
        emit("tau", bucket, blk)
    return rows


def _flatten_residual_rows(residual: dict) -> list[dict]:
    rows: list[dict] = []
    ov = residual.get("overall") or {}
    if ov.get("n", 0):
        rows.append({"tau_bucket": "ALL", "z_bucket": "ALL", "n": ov["n"],
                     "mean_abs_d": ov["mean_abs_d"], "gbt_brier": ov["gbt"]["brier"],
                     "formula_brier": ov["formula"]["brier"], "gbt_diracc": ov["gbt"]["diracc"],
                     "formula_diracc": ov["formula"]["diracc"], "gbt_better": ov["gbt_better"]})
    for c in residual.get("cells", []):
        rows.append({"tau_bucket": c["tau_bucket"], "z_bucket": c["z_bucket"], "n": c["n"],
                     "mean_abs_d": c["mean_abs_d"], "gbt_brier": c["gbt"]["brier"],
                     "formula_brier": c["formula"]["brier"], "gbt_diracc": c["gbt"]["diracc"],
                     "formula_diracc": c["formula"]["diracc"], "gbt_better": c["gbt_better"]})
    return rows


# ===========================================================================
# per-(target, variant) driver
# ===========================================================================
def _run_variant(paths, mat, target, variant, *, train_weeks, test_days, purge_secs, seed,
                 params, max_boost_round, early_stopping_rounds, inner_val_frac,
                 min_windows, health, bench_ctx, run_shuffle):
    """Walk-forward + final model + analyses + harness + shuffled control for one
    (target, variant). Returns ``(block, shuffled_block, fold_rows)``."""
    spec = TARGETS[target]
    label, horizon = spec["label"], spec["horizon_secs"]
    sub = mat[mat[label].notna()].reset_index(drop=True)
    y = sub[label].to_numpy(dtype=float)
    info = _label_info_ms(sub, horizon)
    purge_ms = int(purge_secs) * 1000

    oos, folds, mode = _run_walk_forward_gbt(
        sub, y, info, variant=variant, train_weeks=train_weeks, test_days=test_days,
        purge_ms=purge_ms, params=params, max_boost_round=max_boost_round,
        early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac,
    )
    pooled = _fold_metrics(oos["p_up"].to_numpy(float), oos["y_true"].to_numpy(float)) \
        if len(oos) else {"n": 0, "pos_frac": float("nan"), "brier": float("nan"),
                          "logloss": float("nan"), "diracc": float("nan")}

    out_dir = paths.out_dir / "learn_gbt"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{target}_{variant}"
    _harness_grid(oos).to_parquet(out_dir / f"predictions_{tag}.parquet", engine="pyarrow", index=False)
    if len(oos):
        oos.to_parquet(out_dir / f"oos_{tag}.parquet", engine="pyarrow", index=False)

    # Deployable model (median fold best_iteration, else a fresh inner-val early-stop).
    booster, model_meta, num_rounds = _final_model(
        sub, y, info, variant=variant, train_weeks=train_weeks, params=params,
        max_boost_round=max_boost_round, early_stopping_rounds=early_stopping_rounds,
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
         for n in FEATURE_NAMES), key=lambda r: r["gain"], reverse=True)
    pd.DataFrame(fi_rows, columns=["feature", "gain", "split"]).to_csv(
        out_dir / f"feature_importance_{tag}.csv", index=False)

    block: dict = {
        "label": label, "horizon_secs": horizon, "variant": variant, "mode": mode,
        "n_folds": int(sum(1 for f in folds if not f["skipped"])),
        "n_oos": int(len(oos)),
        "pooled": pooled,
        "num_boost_round": int(num_rounds),
        "best_iteration_median": _median_best_iter(folds),
        "final_model_file": f"model_{tag}.json",
        "booster_file": f"model_{tag}.txt",
        "final_model": {"n_train": model_meta["n_train"], "train_span_days": model_meta["train_span_days"],
                        "class_balance_pos_frac": model_meta["class_balance_pos_frac"]},
        "feature_importance": importance,
    }
    if target == "fwd30" and len(oos):
        block["own_label_reliability"] = lm.reliability_table(
            oos["p_up"].to_numpy(float), oos["y_true"].to_numpy(float))
    # The formula-vs-GBT regime + residual analyses (outcome only — Φ(z) is the
    # window-outcome prior).
    if target == "outcome" and len(oos):
        regime = _regime_analysis(oos)
        residual = _residual_analysis(oos, delta=DISAGREE_DELTA)
        block["regime"] = regime
        block["residual"] = residual
        pd.DataFrame(_flatten_regime_rows(regime),
                     columns=["breakdown", "bucket", "n", "gbt_brier", "gbt_logloss", "gbt_diracc",
                              "formula_brier", "formula_logloss", "formula_diracc"]).to_csv(
            out_dir / f"regime_{tag}.csv", index=False)
        pd.DataFrame(_flatten_residual_rows(residual),
                     columns=["tau_bucket", "z_bucket", "n", "mean_abs_d", "gbt_brier",
                              "formula_brier", "gbt_diracc", "formula_diracc", "gbt_better"]).to_csv(
            out_dir / f"residual_{tag}.csv", index=False)

    if bench_ctx is not None and len(oos):
        hm = score_through_harness(
            paths, bench_ctx, oos, subdir="learn_gbt", out_name=f"harness_{tag}",
            model_label=f"gbt-{tag}", title=f"gbt challenger — {tag}",
            min_windows=min_windows, health=health)
        block["harness"] = _harness_summary(hm)

    fold_rows = [{"target": target, "variant": variant, "control": False, **f} for f in folds]

    # ---- shuffled-label control (proves the pipeline can't cheat) ----
    shuffled: dict = {}
    if run_shuffle and len(oos):
        rng = np.random.default_rng(seed)
        y_shuf = y[rng.permutation(len(y))]
        oos_s, folds_s, _ = _run_walk_forward_gbt(
            sub, y_shuf, info, variant=variant, train_weeks=train_weeks, test_days=test_days,
            purge_ms=purge_ms, params=params, max_boost_round=max_boost_round,
            early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac)
        pooled_s = _fold_metrics(oos_s["p_up"].to_numpy(float), oos_s["y_true"].to_numpy(float)) \
            if len(oos_s) else {"n": 0, "brier": float("nan"), "logloss": float("nan"),
                                "diracc": float("nan"), "pos_frac": float("nan")}
        p_oos = float(np.mean(oos_s["y_true"].to_numpy(float))) if len(oos_s) else float("nan")
        chance_brier = p_oos * (1.0 - p_oos)
        base_rate_acc = max(p_oos, 1.0 - p_oos)
        _harness_grid(oos_s).to_parquet(
            out_dir / f"predictions_{tag}_shuffled.parquet", engine="pyarrow", index=False)
        reliably_better = None
        if bench_ctx is not None and len(oos_s):
            hm_s = score_through_harness(
                paths, bench_ctx, oos_s, subdir="learn_gbt", out_name=f"harness_{tag}_shuffled",
                model_label=f"gbt-{tag}-shuffled", title=f"gbt shuffled control — {tag}",
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
        fold_rows += [{"target": target, "variant": variant, "control": True, **f} for f in folds_s]

    return block, shuffled, fold_rows


# ===========================================================================
# stage entry
# ===========================================================================
def learn_gbt(
    paths: Paths,
    *,
    days: int = DEFAULT_DAYS,
    train_weeks: int = DEFAULT_TRAIN_WEEKS,
    test_days: int = DEFAULT_TEST_DAYS,
    purge_secs: int | None = None,
    seed: int = DEFAULT_SEED,
    targets: tuple[str, ...] = ("fwd30", "outcome"),
    variants: tuple[str, ...] = ("plain", "residual"),
    source: str = "all",
    series: str | None = None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    health: str = "ready",
    run_harness: bool = True,
    run_shuffle: bool = True,
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
    """Train + walk-forward-evaluate the GBT challenger; write ``out/learn_gbt/``.
    ``since_ms``/``until_ms`` bound the harness journal read."""
    series_filter = _parse_series(series)
    mat = _load_matrix(paths, days=days, source=source, series=series_filter)
    params = gbt.default_params(
        seed, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
        min_child_samples=min_child_samples, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        num_threads=num_threads)

    out_dir = paths.out_dir / "learn_gbt"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "config": {
            "days": days, "train_weeks": train_weeks, "test_days": test_days,
            "purge_secs_override": purge_secs, "seed": seed, "targets": list(targets),
            "variants": list(variants), "source": source, "series": series,
            "min_windows": min_windows, "health": health, "run_harness": run_harness,
        },
        "seed": int(seed),
        "params": params,
        "features": list(FEATURE_NAMES),
        "n_rows_loaded": int(len(mat)),
        "caveats": CAVEATS,
        "targets": {},
        "shuffled_control": {},
    }
    if mat.empty:
        result["error"] = "feature_set.parquet had no rows in scope (check --days/--source/--series)."
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
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r} (expected one of {list(TARGETS)})")
        p_secs = purge_secs if purge_secs is not None else (
            TARGETS[target]["horizon_secs"] or WINDOW_SECS)
        tgt_variants = tuple(v for v in variants if v in VARIANTS_BY_TARGET.get(target, ("plain",)))
        result["targets"][target] = {}
        result["shuffled_control"][target] = {}
        for variant in tgt_variants:
            block, shuffled, fold_rows = _run_variant(
                paths, mat, target, variant, train_weeks=train_weeks, test_days=test_days,
                purge_secs=p_secs, seed=seed, params=params, max_boost_round=max_boost_round,
                early_stopping_rounds=early_stopping_rounds, inner_val_frac=inner_val_frac,
                min_windows=min_windows, health=health, bench_ctx=bench_ctx, run_shuffle=run_shuffle)
            block["purge_secs"] = p_secs
            result["targets"][target][variant] = block
            if shuffled:
                result["shuffled_control"][target][variant] = shuffled
            all_folds.extend(fold_rows)

    if all_folds:
        fold_cols = ["target", "variant", "control", "fold_idx", "test_start_day", "test_end_day",
                     "n_train", "n_test", "n_test_windows", "best_iteration", "n_inner_train",
                     "n_inner_val", "skipped", "pos_frac", "brier", "logloss", "diracc"]
        pd.DataFrame(all_folds, columns=fold_cols).to_csv(out_dir / "folds.csv", index=False)

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    return result


def _fmt(x, *args) -> str:
    return eh.fmt(x, *args) if x is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "learn_gbt")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"trailing days of feature_set to use (0 = all; default {DEFAULT_DAYS})")
    parser.add_argument("--train-weeks", type=int, default=DEFAULT_TRAIN_WEEKS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--purge-secs", type=int, default=None,
                        help="purge gap at each boundary in seconds (default: the label horizon)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--targets", default="fwd30,outcome",
                        help="comma-separated targets: fwd30,outcome (default both)")
    parser.add_argument("--variants", default="plain,residual",
                        help="comma-separated variants: plain,residual (residual = outcome only)")
    parser.add_argument("--source", choices=("all", "chainlink", "binance_proxy"), default="all")
    parser.add_argument("--series", default=None, help="comma-separated series filter, e.g. BTC-5m,ETH-5m")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS)
    parser.add_argument("--health", choices=("ready", "all"), default="ready")
    parser.add_argument("--no-harness", action="store_true", help="skip the harness (and its journal read)")
    parser.add_argument("--no-shuffle-control", action="store_true", help="skip the shuffled-label control")
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

    print(f"[learn_gbt] feature_set={paths.table('feature_set')}")
    print(f"[learn_gbt] out        ={paths.out_dir / 'learn_gbt'}")
    try:
        m = learn_gbt(
            paths, days=args.days, train_weeks=args.train_weeks, test_days=args.test_days,
            purge_secs=args.purge_secs, seed=args.seed, targets=targets, variants=variants,
            source=args.source, series=args.series, min_windows=args.min_windows,
            health=args.health, run_harness=not args.no_harness,
            run_shuffle=not args.no_shuffle_control, num_leaves=args.num_leaves,
            max_depth=args.max_depth, learning_rate=args.learning_rate,
            min_child_samples=args.min_child_samples, reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha, max_boost_round=args.max_boost_round,
            early_stopping_rounds=args.early_stopping_rounds, inner_val_frac=args.inner_val_frac,
            num_threads=args.num_threads, since_ms=since_ms, until_ms=until_ms)
    except ParquetNotReady as exc:
        print(f"[learn_gbt] {exc}")
        return 1

    if "error" in m:
        print(f"[learn_gbt] {m['error']}")
        return 1

    ds = m["date_span"]
    print(f"[learn_gbt] rows={m['n_rows_loaded']:,} over {ds['distinct_days']} days "
          f"(day {ds['min_day']}→{ds['max_day']}), seed={m['seed']}")
    n_models = 0
    for target, variants_blk in m["targets"].items():
        for variant, blk in variants_blk.items():
            n_models += 1
            p = blk["pooled"]
            print(f"[learn_gbt] {target}/{variant:<8} [{blk['mode']}, {blk['n_folds']} fold(s), "
                  f"{blk['n_oos']:,} OOS, rounds={blk['num_boost_round']}]: "
                  f"Brier={_fmt(p['brier'])} log-loss={_fmt(p['logloss'])} dir-acc={_fmt(p['diracc'])}")
            h = blk.get("harness")
            if h:
                vf, vm = h["vs_formula"], h["vs_market"]
                print(f"[learn_gbt]   harness ({h['n_scored']:,} scored): "
                      f"vs formula {_fmt(vf['brier_improve_pct'], 1)}% "
                      f"(reliably_better={vf['reliably_better']}), "
                      f"vs market {_fmt(vm['brier_improve_pct'], 1)}%")
            res = blk.get("residual")
            if res and res.get("overall", {}).get("n"):
                ov = res["overall"]
                print(f"[learn_gbt]   disagrees w/ formula on {res['n_disagree']:,}/{res['n_total']:,} "
                      f"({res['frac_disagree']:.0%}); there GBT Brier={_fmt(ov['gbt']['brier'])} "
                      f"vs formula {_fmt(ov['formula']['brier'])} → gbt_better={ov['gbt_better']}")
            sc = m["shuffled_control"].get(target, {}).get(variant)
            if sc:
                ps = sc["pooled"]
                print(f"[learn_gbt]   shuffled control: Brier={_fmt(ps['brier'])} "
                      f"(chance {_fmt(sc['chance_brier'])}) → collapsed={sc['collapsed']}")
    print(f"[learn_gbt] wrote {paths.out_dir / 'learn_gbt'} ({n_models} model(s): "
          "model_*.txt/.json, predictions_*.parquet, harness_*/, feature_importance/regime/residual csv)")

    ok = all(blk["n_oos"] > 0 for vb in m["targets"].values() for blk in vb.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
