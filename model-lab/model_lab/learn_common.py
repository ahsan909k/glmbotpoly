"""Shared, model-agnostic scaffolding for the walk-forward challenger stages.

The logistic (:mod:`model_lab.learn`) and gradient-boosted-tree
(:mod:`model_lab.learn_gbt`) challengers train *different* models but share the
same evaluation machinery: how the feature matrix is loaded, how walk-forward
folds are scheduled and purged, the per-fold metric primitives, the target
definitions, and how an out-of-sample grid is scored through the evaluation
harness. Those pieces live here so both stages stay in lockstep — a change to the
fold schedule or the purge is made once and both models inherit it.

This module is **pure numpy/pandas** (no model dependency, no lightgbm) — it is
imported by the lightgbm-free core (`learn.py`) as well as the opt-in GBT stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import eval_harness as eh
from .config import Paths, assert_parquet_ready
from .feature_set import FEATURE_NAMES
from .lib import math as lm

MS_PER_DAY = 86_400_000

# target key -> (label column, forward horizon in seconds; None = to window close).
TARGETS: dict[str, dict] = {
    "fwd30": {"label": "fwd_up_30s", "horizon_secs": 30},
    "outcome": {"label": "outcome_up", "horizon_secs": None},
}
# Read only what we need from the (possibly multi-GB) feature matrix.
_KEY_COLS = ["series", "window_open_ms", "window_close_ms", "sample_ts_ms"]
_META_COLS = ["label_source", "in_live_coverage"]
_LABEL_COLS = ["fwd_up_30s", "outcome_up"]
_READ_COLS = _KEY_COLS + _META_COLS + FEATURE_NAMES + _LABEL_COLS


# ===========================================================================
# data loading
# ===========================================================================
def _parse_series(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


def _load_matrix(paths: Paths, *, days: int, source: str, series: list[str] | None) -> pd.DataFrame:
    """Read the trailing ``days`` of ``feature_set.parquet`` (predicate-pushed on
    ``window_open_ms``), optionally filtered by label source and series."""
    import pyarrow.parquet as pq

    path = paths.table("feature_set")
    assert_parquet_ready(path, label="feature_set.parquet", min_rows=1)
    opens = pq.read_table(path, columns=["window_open_ms"]).column(0).to_numpy()
    if len(opens) == 0:
        return pd.DataFrame(columns=_READ_COLS)
    max_open = int(opens.max())
    filters = None
    if days and days > 0:
        cutoff = max_open - int(days) * MS_PER_DAY
        filters = [("window_open_ms", ">=", int(cutoff))]
    tbl = pq.read_table(path, columns=_READ_COLS, filters=filters)
    mat = tbl.to_pandas()
    if source in ("chainlink", "binance_proxy"):
        mat = mat[mat["label_source"] == source]
    elif source != "all":
        raise ValueError(f"unknown --source {source!r} (expected chainlink|binance_proxy|all)")
    if series is not None:
        mat = mat[mat["series"].isin(series)]
    return mat.reset_index(drop=True)


# ===========================================================================
# walk-forward geometry (no look-ahead)
# ===========================================================================
def _fold_schedule(unique_days: np.ndarray, train_days: int, test_days: int):
    """Chronological folds as ``(train_start_day, test_start_day, test_end_day_excl)``.

    Primary mode is a sliding trailing window: the first test block starts
    ``train_days`` after the earliest day and rolls forward by ``test_days``. If the
    span is too short for even one such fold, fall back to a single chronological
    ~80/20-by-day split (so a tiny fixture still exercises the whole path). Returns
    ``(folds, mode)``; ``folds`` may be empty when there are < 2 distinct days.
    """
    lo, hi = int(unique_days.min()), int(unique_days.max())
    first_test = lo + int(train_days)
    if first_test <= hi:
        folds = []
        t = first_test
        while t <= hi:
            folds.append((t - int(train_days), t, min(t + int(test_days), hi + 1)))
            t += int(test_days)
        return folds, "walk_forward"
    n_days = len(unique_days)
    if n_days < 2:
        return [], "insufficient"
    n_test = max(1, round(0.2 * n_days))
    test_start = int(np.sort(unique_days)[n_days - n_test])
    return [(lo, test_start, hi + 1)], "single_split"


def _fold_masks(day: np.ndarray, label_info_ms: np.ndarray,
                train_start: int, test_start: int, test_end: int, purge_ms: int):
    """Train/test boolean masks for one fold. Train = the trailing day-window
    ``[train_start, test_start)`` **minus** rows whose label information reaches into
    the test period (``label_info_ms > test_start_ms − purge_ms``) — the purge that
    guarantees a ≥ label-horizon gap. Test = ``[test_start, test_end)``."""
    test_start_ms = int(test_start) * MS_PER_DAY
    train_mask = (
        (day >= train_start) & (day < test_start)
        & (label_info_ms <= test_start_ms - int(purge_ms))
    )
    test_mask = (day >= test_start) & (day < test_end)
    return train_mask, test_mask


def _fold_metrics(prob: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "pos_frac": float(np.mean(y)) if len(y) else float("nan"),
        "brier": lm.brier_score(prob, y),
        "logloss": lm.log_loss(prob, y),
        "diracc": lm.directional_accuracy(prob, y),
    }


def _label_info_ms(sub: pd.DataFrame, horizon_secs) -> np.ndarray:
    """The wall-clock time at which a row's label becomes known — used by the purge.

    fwd30: ``sample_ts + horizon``. outcome: the window close (resolution time).
    """
    if horizon_secs is None:
        return sub["window_close_ms"].to_numpy(np.int64)
    return sub["sample_ts_ms"].to_numpy(np.int64) + int(horizon_secs) * 1000


# ===========================================================================
# harness reporting
# ===========================================================================
def _harness_grid(oos: pd.DataFrame) -> pd.DataFrame:
    """The strict 4-column harness contract grid."""
    return oos[["series", "window_open_ms", "sample_ts_ms", "p_up"]].copy()


def score_through_harness(paths: Paths, ctx: dict, oos: pd.DataFrame, *, subdir: str,
                          out_name: str, model_label: str, title: str,
                          min_windows: int, health: str) -> dict:
    """Score an OOS grid against a pre-built benchmark context and write
    ``out/<subdir>/<out_name>/`` (one shared journal read across all grids)."""
    df, counts = eh.build_external_frame_from(ctx, _harness_grid(oos))
    metrics = eh.run(
        df, counts=counts, title=title, model_label=model_label, self_baseline=False,
        min_windows=min_windows, health=health,
    )
    eh.write_outputs(paths.out_dir / subdir / out_name, metrics)
    return metrics


def _harness_summary(m: dict) -> dict:
    """Compact roll-up of a harness metrics dict: overall + vs-formula + vs-market."""
    overall = eh.find_scope(m["scope_rows"], "overall") or {}
    day = m["verdict_table"]["day"]

    def bench(b: str) -> dict:
        row = next((r for r in day["rows"] if r["benchmark"] == b and r["period"] == "ALL"), None)
        st = day["stability"].get(b, {})
        return {
            "model_brier": (row or {}).get("model_brier"),
            "bench_brier": (row or {}).get("bench_brier"),
            "brier_improve_pct": (row or {}).get("brier_improve_pct"),
            "diracc_delta_pp": (row or {}).get("diracc_delta_pp"),
            "win_rate": st.get("win_rate"),
            "reliably_better": st.get("reliably_better"),
        }

    return {
        "n_scored": m["counts"].get("joined_snapshots", 0),
        "overall": {k: overall.get(k) for k in
                    ("model_brier", "model_logloss", "model_diracc", "n_windows")},
        "vs_formula": bench("formula"),
        "vs_market": bench("market"),
    }
