"""Stage — learn_walkforward: the definitive walk-forward challenger over the full history.

This is the money-is-the-bar verdict tool. It trains a deterministic LightGBM classifier on
``historical_dataset.parquet`` (the full Telonex-proxy + journal-recorder history, 2025-10 →
2026-07), evaluated by **strict expanding-window walk-forward** — train on everything before a
~2-week test block, test on the next unseen block, slide forward — and scores each block **two
ways**:

  * ``eval_harness`` calibration stats (Brier / log-loss / directional accuracy) against the two
    benchmarks the venue already gives us for free — the **formula** ``p_up_model`` and the
    **market** Up-mid ``pm_mid`` — as *diagnostics*; and
  * **money** at the current venue-regime cost (255 ms effective: 5 ms network + 250 ms taker
    delay, real fees), by simulating a threshold-swept taker over the recorded Polymarket book,
    marked to the official resolution — the **verdict metric**.

The bar to clear is money, not Brier: the hand-coded **momentum trigger** replayed on the *same*
current-regime windows (and the cited ``momentum_current_regime/vps255`` reference,
**+$5,056.69 / 28,411 trades / 64% win / p=0.00 over 24,697 windows / 33 days**). A **shuffled-
label control** must collapse to chance or the result is VOID.

Two label targets are **raced** and one is recommended:
  * ``dir10`` — the short-horizon 10 s mid-direction label ``fwd_up_10s`` (a momentum-style
    signal: the book is about to move); and
  * ``taker_ev`` — a trade-aligned label: would buying **Up** *right now* at the displayed ask
    (``pm_mid + pm_spread/2``), after the exact taker fee, held to resolution, have made money.

Two feature sets are ablated to isolate the microstructure contribution: ``base`` (price / vol /
formula: 10 features) vs ``full`` (+ 8 Binance-depth + 6 Polymarket-book: 24 features). One
training scope over all rows; the **depth_source split-metrics gate** (telonex vs recorder)
reports OOS metrics for each source and **fails loudly on a material gap** (the standing contract
that Telonex historical depth must behave like our own recorder depth).

Discipline: leakage-safe (expanding train, purge ≥ the label horizon at every boundary), chunked
/ bounded memory (float32 features, predicate-pushdown load), resumable (per-(target, variant)
OOS parquet + per-(series, day) money checkpoints). **Research only — nothing is wired to the
Rust bot. Stops after the report.**

Opt-in dependency: LightGBM (via :mod:`model_lab.lib.gbt`), same ``[gbt]`` extra as
``learn_short_gbt``; the core pipeline / ``verify`` stay lightgbm-free.

Run::

    python -m model_lab.learn_walkforward                       # full history, both targets
    python -m model_lab.learn_walkforward --days 120 --no-money # quick calibration-only pass
"""

from __future__ import annotations

import base64
import io
import json
import sys
import zlib
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import backtest_momentum as bm
from . import eval_harness as eh
from . import historical_common as hc
from . import historical_dataset as hd
from . import historical_labels as hlbl
from . import historical_resolutions as hr
from . import money_judge as mj
from . import dataset as ds
from .config import Paths, assert_parquet_ready, resolve_bounds, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .learn_common import MS_PER_DAY, _fold_masks, _fold_metrics, _parse_series
from .lib import gbt
from .lib import math as lm
from .lib import procmem

# --- configuration ----------------------------------------------------------
DEFAULT_SEED = 20260710
DEFAULT_INIT_TRAIN_DAYS = 56  # ~8 weeks before the first test block
DEFAULT_TEST_DAYS = 14  # ~2-week test blocks
CURRENT_REGIME_FROM = date(2026, 6, 5)  # 250 ms taker-delay lock (out/audits/venue_regime_timeline.md)

# Feature groups — every name must exist in historical_dataset.parquet (asserted at import).
BASE_FEATURES = ["ret", "realized_vol", "sigma_1s", "log_s_k", "z", "p_up_model",
                 "tau_secs", "elapsed_secs", "basis_bps", "basis_ewma"]
DEPTH_FEATURES = ["depth_imb_1", "depth_imb_5", "depth_imb_10", "depth_imb_20",
                  "microprice_gap", "bid_depth_slope", "ask_depth_slope", "depth_spread_bps"]
PM_FEATURES = ["pm_mid", "pm_spread", "pm_book_imb",
               "pm_staleness_1s", "pm_staleness_2s", "pm_staleness_3s"]
FULL_FEATURES = BASE_FEATURES + DEPTH_FEATURES + PM_FEATURES
VARIANTS = {"base": BASE_FEATURES, "full": FULL_FEATURES}
VARIANT_ORDER = ["base", "full"]

# Two targets. horizon_secs drives the purge (label-known time); taker_ev is known at resolution.
TARGETS = {
    "dir10": {"label": "fwd_up_10s", "horizon_secs": 10, "kind": "direction"},
    "taker_ev": {"label": "_taker_ev", "horizon_secs": None, "kind": "taker_ev"},
}
TARGET_ORDER = ["dir10", "taker_ev"]

# GBT hyperparameters (modest depth, strong reg, deterministic). Budget kept lean so the full
# expanding walk-forward (many folds x targets x variants) is tractable on a 16 GB box.
GBT_NUM_LEAVES = 15
GBT_MAX_DEPTH = 4
GBT_LR = 0.05
GBT_MIN_CHILD = 100
GBT_REG_LAMBDA = 5.0
GBT_MAX_BOOST_ROUND = 600
GBT_EARLY_STOPPING = 40
GBT_INNER_VAL_FRAC = 0.25
FALLBACK_ROUNDS = 150
# Per-fold training-row cap. The expanding window's last folds hold ~all history (10M+ rows);
# capping bounds each fit's time + memory to a constant while keeping the FULL temporal span
# (rows sampled uniformly across the expanding window, seeded → deterministic).
DEFAULT_MAX_TRAIN_ROWS = 2_000_000

DEPTH_GATE_BRIER_TOL = 0.03  # |Brier_telonex − Brier_recorder| beyond this = material gap → FAIL
DEPTH_GATE_MIN_WINDOWS = 30  # a source needs this many distinct windows to be gated (else advisory)

# Money knobs (current venue regime).
DEFAULT_NETWORK_MS = 5
DEFAULT_VENUE_DELAY_MS = mj.DEFAULT_VENUE_DELAY_MS  # 250
MONEY_TARGETS = ("dir10", "taker_ev")  # money-scored on the FULL variant

CAVEATS = [
    "Money uses the same conservative single-top-of-book taker model as money_judge/backtest_"
    "momentum (pay the displayed ask, mirror Down off the Up bid, skip on a stale/absent book, "
    "$1 min notional, mark to the official resolution — no early exit).",
    "Effective taker latency = 5 ms network + 250 ms venue delay = 255 ms (the current locked "
    "regime). Momentum reference: momentum_current_regime/vps255 = +$5,056.69 over 33 days.",
    "The verdict metric is money on CURRENT-REGIME windows only (>= 2026-06-05); never pooled "
    "across venue-regime boundaries. eval_harness Brier/log-loss are diagnostics, not the bar.",
    "taker_ev label is derived from the dataset's pm_mid/pm_spread (as-of the sample, top-of-"
    "book); the money simulation uses the real recorded book at sample+255 ms. dir10 is the 10 s "
    "forward mid-direction label.",
    "Historical (Telonex, label_source=telonex) rows use a Binance-anchored strike/outcome; the "
    "recorder rows (>= 2026-07-04) use the Chainlink strike. basis_* and Chainlink features are "
    "NaN on telonex rows (LightGBM native missing). No aggregated-flow features (they live only "
    "in feature_set.parquet, not historical_dataset).",
    "Determinism is per-machine (LightGBM deterministic + single-thread + pinned seeds); pin the "
    "lightgbm version to reproduce.",
]

# Import-time drift guard: every feature must exist in the historical_dataset column set.
_HD_COLS = set(hd.HIST_COLUMNS)
assert set(FULL_FEATURES) <= _HD_COLS, \
    f"learn_walkforward: features missing from historical_dataset: {set(FULL_FEATURES) - _HD_COLS}"
for _c in ("pm_mid", "pm_spread", "outcome_up", "fwd_up_10s", "window_close_ms", "tau_secs",
           "label_source", "depth_source"):
    assert _c in _HD_COLS, f"learn_walkforward: required column {_c!r} missing from historical_dataset"


# --- data loading (bounded, float32) ---------------------------------------
_KEY_COLS = ["series", "asset", "window_open_ms", "sample_ts_ms", "window_close_ms"]
_EXTRA_COLS = ["tau_secs", "p_up_model", "pm_mid", "pm_spread", "outcome_up",
               "fwd_up_10s", "label_source", "depth_source"]


def _read_cols() -> list[str]:
    seen: dict[str, None] = {}
    for c in _KEY_COLS + FULL_FEATURES + _EXTRA_COLS:
        seen.setdefault(c, None)
    return list(seen)


def _load_matrix(paths: Paths, *, since_ms: int | None, until_ms: int | None,
                 days: int, series: list[str] | None) -> pd.DataFrame:
    """Load the needed columns from historical_dataset.parquet, features as float32 (NaN kept
    for LightGBM missing), bounded by --since/--until/--days via predicate pushdown."""
    path = paths.table("historical_dataset")
    assert_parquet_ready(path, label="historical_dataset.parquet", min_rows=1)
    filt: list[tuple] = []
    if since_ms is not None:
        filt.append(("window_open_ms", ">=", int(since_ms)))
    if until_ms is not None:
        filt.append(("window_open_ms", "<", int(until_ms)))
    if days > 0:
        max_open = pc.max(pq.read_table(path, columns=["window_open_ms"]).column(0)).as_py()
        if max_open is not None:
            filt.append(("window_open_ms", ">=", int(max_open) - days * MS_PER_DAY))
    tbl = pq.read_table(path, columns=_read_cols(), filters=filt or None)
    mat = tbl.to_pandas()
    if series:
        mat = mat[mat["series"].isin(series)]
    for f in FULL_FEATURES:
        if f in mat.columns:
            mat[f] = mat[f].astype("float32")
    for c in ("series", "asset", "label_source", "depth_source"):  # bound string-column memory
        if c in mat.columns:
            mat[c] = mat[c].astype("category")
    return mat.reset_index(drop=True)


# --- taker-EV label ---------------------------------------------------------
def _taker_ev_label(mat: pd.DataFrame, fee_rate: float) -> np.ndarray:
    """Binary: would buying Up now at the displayed ask (pm_mid + pm_spread/2), after the exact
    taker fee, held to resolution, net > 0. NaN where the book (pm_mid/pm_spread) is absent.

    Note: held to resolution, buying the winner profits for any ask < ~1 (payoff $1 > ask+fee),
    so this label ~= outcome_up except at a near-$1 entry — i.e. taker_ev is effectively the
    'predict the window winner' target, distinct from dir10's short-horizon direction."""
    mid = mat["pm_mid"].to_numpy(dtype=float)
    spread = mat["pm_spread"].to_numpy(dtype=float)
    outcome = mat["outcome_up"].to_numpy(dtype=float)
    ask = np.clip(mid + spread / 2.0, 1e-6, 1.0 - 1e-6)
    # vectorized replica of lm.taker_fee(1, rate, ask): raw = rate*ask*(1-ask); 5-dp half-away
    # round; $0.00001 floor (raw is >0 for ask in (0,1), so the raw==0 branch never fires here).
    raw = fee_rate * ask * (1.0 - ask)
    fee = np.maximum(np.floor(raw * 1e5 + 0.5) / 1e5, 1e-5)
    net = np.where(outcome >= 0.5, 1.0, 0.0) - ask - fee
    label = np.where(net > 0.0, 1.0, 0.0)
    ok = np.isfinite(mid) & np.isfinite(spread) & np.isfinite(outcome)
    label[~ok] = np.nan
    return label


def _prep_target(mat: pd.DataFrame, target: str, fee_rate: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (sub, y, label_info_ms) for a target: rows with a finite label, the 0/1 label, and
    the ms at which the label becomes known (for the purge)."""
    spec = TARGETS[target]
    if spec["kind"] == "taker_ev":
        y_all = _taker_ev_label(mat, fee_rate)
    else:
        y_all = mat[spec["label"]].to_numpy(dtype=float)
    ok = np.isfinite(y_all)
    sub = mat[ok].reset_index(drop=True)
    y = y_all[ok].astype(np.int8)
    if spec["horizon_secs"] is None:
        info = sub["window_close_ms"].to_numpy(dtype="int64")
    else:
        info = sub["sample_ts_ms"].to_numpy(dtype="int64") + int(spec["horizon_secs"]) * 1000
    return sub, y, info


# --- expanding walk-forward -------------------------------------------------
def _expanding_folds(unique_days: np.ndarray, init_train_days: int, test_days: int) -> tuple[list[tuple[int, int, int]], str]:
    """Expanding-window schedule: train = [lo, test_start) (anchored at lo — EXPANDING), test =
    the next ``test_days``. Returns ((train_start, test_start, test_end_excl)…, mode)."""
    days = np.unique(unique_days.astype("int64"))
    if days.size < 2:
        return [], "insufficient"
    lo, hi = int(days.min()), int(days.max())
    first_test = lo + int(init_train_days)
    if first_test > hi:  # too short for even one block → single ~80/20 chronological split
        n_test = max(1, round(0.2 * days.size))
        test_start = int(days[days.size - n_test])
        return [(lo, test_start, hi + 1)], "single_split"
    folds, t = [], first_test
    while t <= hi:
        folds.append((lo, t, min(t + int(test_days), hi + 1)))
        t += int(test_days)
    return folds, "walk_forward"


def _inner_best_iteration(x, y, ts, label_info, *, params, purge_ms) -> int:
    """Forward-chained inner-validation round count: hold out the latest INNER_VAL_FRAC of the
    train window by time (purged), early-stop on it. Falls back to a fixed count if no split."""
    n = len(y)
    n_val = round(GBT_INNER_VAL_FRAC * n)
    fallback = min(GBT_MAX_BOOST_ROUND, FALLBACK_ROUNDS)
    if n_val < 1 or n_val >= n:
        return fallback
    order = np.argsort(ts, kind="stable")
    val_idx = order[-n_val:]
    val_start = int(ts[val_idx].min())
    cand = order[:-n_val]
    tr_idx = cand[label_info[cand] <= val_start - purge_ms]
    if tr_idx.size == 0:
        return fallback
    _, best = gbt.fit(x[tr_idx], y[tr_idx], x[val_idx], y[val_idx], params=params,
                      max_boost_round=GBT_MAX_BOOST_ROUND, early_stopping_rounds=GBT_EARLY_STOPPING)
    return int(best) if best and best > 0 else fallback


_OOS_COLS = ["series", "window_open_ms", "sample_ts_ms", "p_up", "y_true", "p_up_model",
             "pm_mid", "outcome_up", "tau_secs", "window_close_ms", "label_source", "depth_source"]


def _walk_forward(sub: pd.DataFrame, y: np.ndarray, label_info: np.ndarray, *,
                  feature_names: list[str], folds: list[tuple[int, int, int]], purge_ms: int,
                  params: dict, max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS) -> tuple[pd.DataFrame, list[dict]]:
    """Train one GBT per fold on the expanding window (purged, row-capped), predict the block OOS."""
    day = (sub["window_open_ms"].to_numpy(dtype="int64") // MS_PER_DAY)
    ts = sub["sample_ts_ms"].to_numpy(dtype="int64")
    x = sub[feature_names].to_numpy(dtype=np.float32)
    parts, records = [], []
    for fi, (tr_s, te_s, te_e) in enumerate(folds):
        tr_mask, te_mask = _fold_masks(day, label_info, tr_s, te_s, te_e, purge_ms)
        n_te = int(te_mask.sum())
        tr_idx = np.nonzero(tr_mask)[0]
        n_tr = int(tr_idx.size)
        if n_tr == 0 or n_te == 0:
            records.append({"fold": fi, "test_start_day": te_s, "test_end_day": te_e,
                            "n_train": n_tr, "n_test": n_te, "best_iteration": None, "skipped": True})
            continue
        if max_train_rows and n_tr > max_train_rows:  # uniform, seeded subsample of the expanding window
            rng = np.random.default_rng(int(params["seed"]) + fi)
            tr_idx = np.sort(rng.choice(tr_idx, size=max_train_rows, replace=False))
        best = _inner_best_iteration(x[tr_idx], y[tr_idx], ts[tr_idx], label_info[tr_idx],
                                     params=params, purge_ms=purge_ms)
        booster = gbt.refit(x[tr_idx], y[tr_idx], params=params, num_boost_round=best)
        prob = gbt.predict_proba(booster, x[te_mask], num_iteration=best, num_threads=params["num_threads"])
        te = sub[te_mask].reset_index(drop=True)
        part = pd.DataFrame({
            "series": te["series"].to_numpy(), "window_open_ms": te["window_open_ms"].to_numpy(),
            "sample_ts_ms": te["sample_ts_ms"].to_numpy(), "p_up": prob.astype(float),
            "y_true": y[te_mask].astype(float), "p_up_model": te["p_up_model"].to_numpy(dtype=float),
            "pm_mid": te["pm_mid"].to_numpy(dtype=float), "outcome_up": te["outcome_up"].to_numpy(dtype=float),
            "tau_secs": te["tau_secs"].to_numpy(dtype=float),
            "window_close_ms": te["window_close_ms"].to_numpy(dtype="int64"),
            "label_source": te["label_source"].to_numpy(), "depth_source": te["depth_source"].to_numpy(),
        })
        parts.append(part)
        n_win = int(te.groupby(["series", "window_open_ms"], observed=True).ngroups)
        records.append({"fold": fi, "test_start_day": te_s, "test_end_day": te_e, "n_train": n_tr,
                        "n_train_used": int(tr_idx.size), "n_test": n_te, "n_test_windows": n_win,
                        "best_iteration": int(best), "skipped": False})
    oos = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_OOS_COLS)
    return oos, records


def _final_model(sub: pd.DataFrame, y: np.ndarray, label_info: np.ndarray, *, feature_names: list[str],
                 params: dict, rounds: int | None) -> dict:
    """Fit a deployable booster on the full history (row-capped) for inspection — feature
    importances + class balance. Not used for OOS; the walk-forward folds are."""
    n = len(y)
    idx = np.arange(n)
    if n > DEFAULT_MAX_TRAIN_ROWS:
        idx = np.sort(np.random.default_rng(int(params["seed"])).choice(n, DEFAULT_MAX_TRAIN_ROWS, replace=False))
        y, label_info = y[idx], label_info[idx]
    x = sub[feature_names].iloc[idx].to_numpy(dtype=np.float32)
    ts = sub["sample_ts_ms"].to_numpy(dtype="int64")[idx]
    n_rounds = rounds if (rounds and rounds > 0) else _inner_best_iteration(
        x, y, ts, label_info, params=params, purge_ms=0)
    booster = gbt.refit(x, y, params=params, num_boost_round=n_rounds)
    imp = gbt.feature_importance(booster, feature_names)
    return {"num_boost_round": int(n_rounds), "n_train": int(len(y)),
            "class_balance_pos_frac": float(np.mean(y)) if len(y) else float("nan"),
            "feature_importance": imp}


# --- regime annotation ------------------------------------------------------
def _regime_of(open_ms: int) -> str:
    """The venue-regime label a window's open date falls in (most-recent boundary <= date)."""
    d = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).date().isoformat()
    label = "pre-2026-01-07"
    for b in bm.REGIME_BOUNDARIES:
        if d >= b["date"]:
            label = b["short"]
    return label


def _is_current_regime(open_ms: int) -> bool:
    return datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).date() >= CURRENT_REGIME_FROM


# --- eval-harness diagnostics (per block + aggregate) ----------------------
def _block_stats(oos: pd.DataFrame) -> dict:
    """Model / formula / market Brier + directional-accuracy vs the window outcome, on the rows
    where each predictor is defined. Fast (lm primitives) — no eh.run per block."""
    out: dict[str, float | int] = {"n": int(len(oos)),
                                   "n_windows": int(oos.groupby(["series", "window_open_ms"]).ngroups) if len(oos) else 0}
    y = oos["outcome_up"].to_numpy(dtype=float)
    for name, col in (("model", "p_up"), ("formula", "p_up_model"), ("market", "pm_mid")):
        p = oos[col].to_numpy(dtype=float)
        m = np.isfinite(p) & np.isfinite(y)
        out[f"{name}_brier"] = lm.brier_score(p[m], y[m]) if m.any() else float("nan")
        out[f"{name}_diracc"] = lm.directional_accuracy(p[m], y[m]) if m.any() else float("nan")
        out[f"{name}_n"] = int(m.sum())
    return out


def _per_block_table(oos: pd.DataFrame, folds: list[tuple[int, int, int]]) -> list[dict]:
    rows = []
    day = (oos["window_open_ms"].to_numpy(dtype="int64") // MS_PER_DAY) if len(oos) else np.empty(0)
    for fi, (_, te_s, te_e) in enumerate(folds):
        blk = oos[(day >= te_s) & (day < te_e)] if len(oos) else oos
        if not len(blk):
            continue
        open_lo = int(blk["window_open_ms"].min())
        st = _block_stats(blk)
        rows.append({"fold": fi, "test_start": datetime.fromtimestamp(te_s * MS_PER_DAY / 1000.0, tz=timezone.utc).date().isoformat(),
                     "regime": _regime_of(open_lo), "current": _is_current_regime(open_lo), **st})
    return rows


def _aggregate_eval(paths: Paths, oos: pd.DataFrame, *, target: str, out_name: str) -> dict:
    """One eh.run over the whole OOS with formula=p_up_model, market=pm_mid supplied directly
    (no journal). Returns a compact summary + writes the full harness output folder."""
    if not len(oos):
        return {"n": 0}
    dur = ((oos["window_close_ms"].to_numpy(dtype="int64") - oos["window_open_ms"].to_numpy(dtype="int64")) / 1000.0)
    raw = pd.DataFrame({
        "series": oos["series"].to_numpy(), "open_time": oos["window_open_ms"].to_numpy(dtype="int64"),
        "ts": oos["sample_ts_ms"].to_numpy(dtype="int64"),
        "tau": oos["tau_secs"].to_numpy(dtype=float), "dur": dur,
        "outcome_up": oos["outcome_up"].to_numpy(dtype=float), "model": oos["p_up"].to_numpy(dtype=float),
        "formula": oos["p_up_model"].to_numpy(dtype=float), "market": oos["pm_mid"].to_numpy(dtype=float),
        "health": "Ready",
    })
    df = eh._finalize(raw)
    counts = {"prediction_rows": int(len(oos)), "joined_snapshots": int(len(df))}
    metrics = eh.run(df, counts=counts, title=f"learn_walkforward · {target}",
                     model_label=f"gbt_full_{target}", self_baseline=False, health="all")
    hdir = paths.out_dir / out_name / f"harness_{target}"
    hdir.mkdir(parents=True, exist_ok=True)
    eh.write_outputs(hdir, metrics)
    overall = next((r for r in metrics["scope_rows"] if r["scope"] == "overall"), {})
    return {"n": int(len(oos)), "overall": overall, "reliability": metrics.get("reliability", {}),
            "verdict": metrics.get("verdict", "")}


# --- depth_source split-metrics gate ---------------------------------------
def _depth_split_gate(oos: pd.DataFrame) -> dict:
    """OOS metrics split by depth_source (telonex vs recorder). Fails loudly if the model's Brier
    differs materially between the two well-sampled sources (the standing contract that Telonex
    historical depth behaves like our own recorder depth)."""
    res: dict = {"sources": {}, "gap": None, "pass": True, "reason": ""}
    y = oos["outcome_up"].to_numpy(dtype=float)
    p = oos["p_up"].to_numpy(dtype=float)
    src = oos["depth_source"].astype("object").to_numpy()
    wid = oos["window_open_ms"].to_numpy(dtype="int64")
    for s in ("telonex", "recorder"):
        m = (src == s) & np.isfinite(p) & np.isfinite(y)
        n = int(m.sum())
        nwin = int(np.unique(wid[m]).size) if n else 0
        res["sources"][s] = {"n": n, "n_windows": nwin,
                             "brier": lm.brier_score(p[m], y[m]) if n else float("nan"),
                             "diracc": lm.directional_accuracy(p[m], y[m]) if n else float("nan")}
    t, r = res["sources"]["telonex"], res["sources"]["recorder"]
    both = t["n_windows"] >= DEPTH_GATE_MIN_WINDOWS and r["n_windows"] >= DEPTH_GATE_MIN_WINDOWS
    if both and np.isfinite(t["brier"]) and np.isfinite(r["brier"]):
        gap = abs(t["brier"] - r["brier"])
        res["gap"] = float(gap)
        if gap > DEPTH_GATE_BRIER_TOL:
            res["pass"] = False
            res["reason"] = (f"depth_source Brier gap {gap:.4f} > tol {DEPTH_GATE_BRIER_TOL} "
                             f"(telonex {t['brier']:.4f} / recorder {r['brier']:.4f}) — Telonex historical "
                             f"depth does NOT behave like recorder depth; do not pool them untagged.")
        else:
            res["reason"] = f"depth_source Brier gap {gap:.4f} <= tol {DEPTH_GATE_BRIER_TOL} (OK to pool)."
    else:
        res["reason"] = (f"insufficient windows to gate (telonex {t['n_windows']}, recorder "
                         f"{r['n_windows']}; need >= {DEPTH_GATE_MIN_WINDOWS} each) — advisory only.")
    return res


# --- money scoring on current-regime windows -------------------------------
def _fires_from_group(g: pd.DataFrame | None, series_key: str, open_ms: int, up_token: str,
                      outcome_up: int, close_ms: int, price_col: str) -> pd.DataFrame | None:
    """The fires frame _run_taker consumes, for one window, from a pre-grouped OOS slice."""
    if g is None or not len(g):
        return None
    return pd.DataFrame({
        "series": series_key, "window_open_ms": int(open_ms),
        "sample_ts_ms": g["sample_ts_ms"].to_numpy(dtype="int64"),
        "p_up": g[price_col].to_numpy(dtype=float), "up_token": up_token,
        "outcome_up": int(outcome_up), "close_time": int(close_ms),
    })


def _group_current_regime(oos_by_target: dict[str, pd.DataFrame], since_ms: int) -> dict:
    """Per target: {(series, window_open_ms): slice} over current-regime windows only (bounds the
    per-window lookup in the money loop; avoids an O(windows x rows) scan)."""
    groups: dict[str, dict] = {}
    for t, oos in oos_by_target.items():
        cr = oos[oos["window_open_ms"].to_numpy(dtype="int64") >= since_ms]
        groups[t] = {k: v for k, v in cr.groupby(["series", "window_open_ms"], sort=False)}
    return groups


def _sweep_metrics(trades_by_theta: dict[float, list[dict]]) -> tuple[list[dict], dict | None]:
    """Per-threshold metrics + the best (argmax net, tie→higher threshold)."""
    sweep = [{"threshold": th, "taker": mj._metrics_for_trades(trades_by_theta[th])}
             for th in sorted(trades_by_theta)]
    cands = [s for s in sweep if s["taker"]["n_trades"] >= 1]
    best = None
    if cands:
        best = max(cands, key=lambda s: (s["taker"]["net_pnl"], s["threshold"]))
    return sweep, best


def _money_score(paths: Paths, oos_by_target: dict[str, pd.DataFrame], *, since_ms: int | None,
                 network_ms: int, venue_delay_ms: int, fee_rate: float, window_budget: float,
                 trade_budget: float, res_map: dict, out_dir: Path) -> dict:
    """Replay a threshold-swept taker driven by each target's OOS predictions over the current-
    regime windows, plus the formula-as-taker and the momentum trigger on the identical windows.
    Resumable per-(series, day). Effective latency = network + venue delay."""
    effective = network_ms + venue_delay_ms
    since_floor = int(since_ms) if since_ms is not None else 0
    series_keys = hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    strike_tol = ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0
    thresholds = mj.SWEEP_THRESHOLDS
    groups = _group_current_regime(oos_by_target, since_floor)  # pre-grouped per (series, open)
    first_target = TARGET_ORDER[0]
    ckpt_dir = out_dir / "money_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # accumulators
    model_tr: dict[str, dict[float, list]] = {t: {th: [] for th in thresholds} for t in oos_by_target}
    formula_tr: dict[float, list] = {th: [] for th in thresholds}
    momentum_tr: list[dict] = []
    n_windows = 0
    done: set[tuple[str, str]] = set()

    def _load_ckpt(cp: Path):
        nonlocal n_windows
        data = json.loads(cp.read_text(encoding="utf-8"))
        for t in oos_by_target:
            for th_s, rows in data.get("model", {}).get(t, {}).items():
                model_tr[t][float(th_s)].extend(rows)
        for th_s, rows in data.get("formula", {}).items():
            formula_tr[float(th_s)].extend(rows)
        momentum_tr.extend(data.get("momentum", []))
        n_windows += int(data.get("windows", 0))
        done.add((data["series"], data["day"]))

    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            _load_ckpt(cp)
        except (OSError, json.JSONDecodeError, KeyError):
            continue

    for series_key in series_keys:
        prefix = hc.series_prefix(series_key)
        asset = hc.SLUG_PREFIXES[prefix][1]
        symbol = hc.SYMBOL_BY_ASSET[asset]
        for d in hlbl._days_for_series(paths, series_key, since_ms, None):
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
            grid, ring_ts, ring_px = (bm._day_grid_and_ring(paths, symbol, asset, d) if wins
                                      else (pd.DataFrame(), None, None))
            win_by_key: dict[tuple, dict] = {}
            books: dict[str, dict] = {}
            model_rows: list[dict] = []
            day_model = {t: {th: [] for th in thresholds} for t in oos_by_target}
            day_formula = {th: [] for th in thresholds}
            day_momentum: list[dict] = []
            if wins and not grid.empty:
                mids_by_asset = {asset: (ring_ts, ring_px)}
                bars = bm._grid_bars(grid)
                for w in wins:
                    open_ms, close_ms = int(w["window_open_ms"]), int(w["window_close_ms"])
                    official = res_map.get((series_key, open_ms))
                    if official is None:
                        continue
                    st = hc.binance_anchored_strike(bars, open_ms, strike_tol)
                    if not (st["strike"] > 0.0):
                        continue
                    upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
                    if upq.empty:
                        continue
                    up_token = str(upq["token_id"].iloc[0])
                    outcome_up = 1 if official["outcome"] == "Up" else 0
                    key = (series_key, open_ms)
                    win_by_key[key] = {"up_token": up_token, "close_time": close_ms, "outcome_up": outcome_up}
                    books[up_token] = tpm.pm_book_arrays(upq)
                    model_rows.extend(bm._window_model_rows(grid, w, st))
                    # taker per target
                    for t in oos_by_target:
                        fires = _fires_from_group(groups[t].get((series_key, open_ms)), series_key,
                                                  open_ms, up_token, outcome_up, close_ms, "p_up")
                        if fires is not None:
                            for th in thresholds:
                                rows, _n = mj._run_taker(fires, threshold=th, latency_ms=effective,
                                                         fee_rate=fee_rate, window_budget=window_budget,
                                                         trade_budget=trade_budget, books=books)
                                day_model[t][th].extend(rows)
                    # formula-as-taker (primary target's sample grid; the p_up_model column)
                    ffires = _fires_from_group(groups[first_target].get((series_key, open_ms)), series_key,
                                               open_ms, up_token, outcome_up, close_ms, "p_up_model")
                    if ffires is not None:
                        for th in thresholds:
                            rows, _n = mj._run_taker(ffires, threshold=th, latency_ms=effective,
                                                     fee_rate=fee_rate, window_budget=window_budget,
                                                     trade_budget=trade_budget, books=books)
                            day_formula[th].extend(rows)
                if model_rows:
                    day_momentum = mj._run_momentum(model_rows, mids_by_asset, win_by_key, books,
                                                    latency_ms=effective, fee_rate=fee_rate,
                                                    window_budget=window_budget)
            # checkpoint (atomic) + accumulate
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            payload = {"series": series_key, "day": day_str, "windows": len(win_by_key),
                       "model": {t: {str(th): day_model[t][th] for th in thresholds} for t in oos_by_target},
                       "formula": {str(th): day_formula[th] for th in thresholds},
                       "momentum": day_momentum}
            tmp.write_text(json.dumps(payload, default=eh._json_default), encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            for t in oos_by_target:
                for th in thresholds:
                    model_tr[t][th].extend(day_model[t][th])
            for th in thresholds:
                formula_tr[th].extend(day_formula[th])
            momentum_tr.extend(day_momentum)
            n_windows += len(win_by_key)

    # finalize
    out: dict = {"effective_latency_ms": effective, "network_ms": network_ms,
                 "venue_delay_ms": venue_delay_ms, "current_regime_windows": n_windows,
                 "reference_vps255": {"net_pnl": 5056.69, "trades": 28411, "win_rate": 0.640,
                                      "windows": 24697, "days": 33, "p_value": 0.0,
                                      "source": "out/backtests/momentum_current_regime/vps255"},
                 "targets": {}, "momentum": mj._metrics_for_trades(momentum_tr),
                 "never_trading_net": 0.0}
    f_sweep, f_best = _sweep_metrics(formula_tr)
    out["formula"] = {"sweep": f_sweep, "best": f_best}
    for t in oos_by_target:
        sweep, best = _sweep_metrics(model_tr[t])
        best_rows = model_tr[t][best["threshold"]] if best else []
        out["targets"][t] = {"sweep": sweep, "best": best,
                             "per_series": _money_by(best_rows, lambda r: r["series"]),
                             "lift": mj._lift_over_majority(best_rows)}
    return out


def _money_by(rows: list[dict], key_fn) -> list[dict]:
    by: dict = defaultdict(list)
    for r in rows:
        by[key_fn(r)].append(r)
    out = []
    for k in sorted(by):
        m = mj._metrics_for_trades(by[k])
        out.append({"key": k, **m})
    return out


# --- report -----------------------------------------------------------------
def _verdict_text(result: dict) -> str:
    money = result.get("money")
    if not money:
        return "Money scoring was not run (--no-money); only calibration diagnostics are available."
    ref = money["reference_vps255"]["net_pnl"]
    mom = money["momentum"]["net_pnl"]
    parts = [f"Over {money['current_regime_windows']:,} current-regime windows (>= 2026-06-05, 255 ms "
             f"effective), the momentum trigger replayed here nets ${mom:,.2f} "
             f"(cited vps255 reference ${ref:,.2f}); never-trading nets $0."]
    best_t, best_net = None, -1e18
    for t, blk in money["targets"].items():
        b = blk["best"]
        net = b["taker"]["net_pnl"] if b else float("nan")
        parts.append(f"Challenger '{t}' best-threshold taker nets ${net:,.2f}"
                     + (f" (θ={b['threshold']}, {b['taker']['n_trades']:,} trades, "
                        f"win {b['taker']['win_rate']:.0%})" if b else " (no trades)") + ".")
        if b and net > best_net:
            best_net, best_t = net, t
    fb = money["formula"]["best"]
    fnet = fb["taker"]["net_pnl"] if fb else float("nan")
    parts.append(f"Formula-as-taker (p_up_model) nets ${fnet:,.2f}.")
    if best_t is not None:
        beats_mom = best_net > mom
        beats_form = best_net > fnet
        parts.append(f"Recommended target: '{best_t}' (best money). It "
                     f"{'BEATS' if beats_mom else 'does NOT beat'} the momentum trigger and "
                     f"{'beats' if beats_form else 'does not beat'} the formula-taker on current-regime money.")
    sh = result.get("shuffled")
    if sh is not None:
        parts.append("Shuffled-label control " + ("COLLAPSED to chance (real signal is real)."
                     if sh.get("collapsed") else "did NOT collapse — treat the result as VOID."))
    gate = result.get("depth_gate", {})
    if gate.get("pass") is False:
        parts.append("DEPTH-SOURCE GATE FAILED: " + gate.get("reason", ""))
    return " ".join(parts)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_report(paths: Paths, out_dir: Path, result: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    # per-block csv (full variant, primary target)
    pb = result.get("per_block", {})
    for t, rows in pb.items():
        pd.DataFrame(rows).to_csv(out_dir / f"per_block_{t}.csv", index=False)

    money = result.get("money")
    money_rows = []
    if money:
        money_rows.append({"strategy": "momentum (replayed)", "net_pnl": money["momentum"]["net_pnl"],
                           "trades": money["momentum"]["n_trades"], "win_rate": money["momentum"]["win_rate"]})
        money_rows.append({"strategy": "momentum vps255 (cited)", "net_pnl": money["reference_vps255"]["net_pnl"],
                           "trades": money["reference_vps255"]["trades"], "win_rate": money["reference_vps255"]["win_rate"]})
        fb = money["formula"]["best"]
        money_rows.append({"strategy": "formula-taker (best θ)", "net_pnl": fb["taker"]["net_pnl"] if fb else 0.0,
                           "trades": fb["taker"]["n_trades"] if fb else 0, "win_rate": fb["taker"]["win_rate"] if fb else float("nan")})
        for t, blk in money["targets"].items():
            b = blk["best"]
            money_rows.append({"strategy": f"model '{t}' (best θ)", "net_pnl": b["taker"]["net_pnl"] if b else 0.0,
                               "trades": b["taker"]["n_trades"] if b else 0, "win_rate": b["taker"]["win_rate"] if b else float("nan")})
        money_rows.append({"strategy": "never trading", "net_pnl": 0.0, "trades": 0, "win_rate": float("nan")})
        pd.DataFrame(money_rows).to_csv(out_dir / "money_comparison.csv", index=False)

    gate = result.get("depth_gate", {})
    gate_rows = [{"depth_source": s, **v} for s, v in gate.get("sources", {}).items()]
    imp_rows = []
    for t, fm in result.get("final_models", {}).items():
        gain = fm.get("feature_importance", {}).get("gain", {})
        for f, g in sorted(gain.items(), key=lambda kv: -kv[1])[:12]:
            imp_rows.append({"target": t, "feature": f, "gain": g})
    pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv", index=False)

    def tbl(rows, cols):
        return eh._table_html(rows, cols) if rows else "<p class='muted'>(none)</p>"

    money_html = ""
    if money:
        money_html = f"""<h2>Money — current regime (255 ms effective, official resolution)</h2>
<p class="muted">The verdict metric. Challenger taker (threshold-swept) vs the momentum trigger
(replayed on the identical windows + the cited vps255 reference) vs formula-as-taker vs never-trade.</p>
{tbl(money_rows, [("strategy", "strategy"), ("net_pnl", "net PnL $"), ("trades", "trades"), ("win_rate", "win rate")])}"""

    pb_primary = pb.get(TARGET_ORDER[0], [])
    pb_cols = [("test_start", "block start"), ("regime", "regime"), ("current", "current"),
               ("n_windows", "windows"), ("model_brier", "model Brier"), ("formula_brier", "formula Brier"),
               ("market_brier", "market Brier"), ("model_diracc", "model dir-acc")]
    verdict = result.get("verdict", "")
    bug = " bug" if (gate.get("pass") is False or (result.get("shuffled") and not result["shuffled"].get("collapsed"))) else ""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>learn_walkforward</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1040px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.9rem; }}
 td, th {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem; border-radius: 4px; line-height: 1.5; }}
 .verdict.bug {{ border-left-color: #d62728; background: #fdeeee; }}
</style></head><body>
<h1>learn_walkforward — expanding walk-forward challenger (money is the bar)</h1>
<div class="verdict{bug}">{verdict}</div>
{money_html}
<h2>Per-block calibration (primary target '{TARGET_ORDER[0]}', full variant) — diagnostics</h2>
{tbl(pb_primary, pb_cols)}
<h2>depth_source split gate (telonex vs recorder)</h2>
<p class="muted">{gate.get('reason', '')}</p>
{tbl(gate_rows, [("depth_source", "source"), ("n_windows", "windows"), ("n", "rows"), ("brier", "Brier"), ("diracc", "dir-acc")])}
<h2>Feature importances (final full model, top by gain)</h2>
{tbl(imp_rows, [("target", "target"), ("feature", "feature"), ("gain", "gain")])}
<h2>Caveats</h2><ul>{''.join(f'<li>{c}</li>' for c in CAVEATS)}</ul>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


# --- driver -----------------------------------------------------------------
def _params(seed: int) -> dict:
    return gbt.default_params(seed, num_leaves=GBT_NUM_LEAVES, max_depth=GBT_MAX_DEPTH,
                              learning_rate=GBT_LR, min_child_samples=GBT_MIN_CHILD,
                              reg_lambda=GBT_REG_LAMBDA, num_threads=1)


def learn_walkforward(
    paths: Paths, *, since_ms: int | None = None, until_ms: int | None = None, days: int = 0,
    series: list[str] | None = None, init_train_days: int = DEFAULT_INIT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS, seed: int = DEFAULT_SEED, fee_rate: float = mj.CRYPTO_FEE_RATE,
    run_money: bool = True, run_shuffle: bool = True, network_ms: int = DEFAULT_NETWORK_MS,
    venue_delay_ms: int = DEFAULT_VENUE_DELAY_MS, window_budget: float = mj.DEFAULT_WINDOW_BUDGET,
    trade_budget: float = mj.DEFAULT_TRADE_BUDGET, out_name: str = "learn_walkforward",
) -> dict:
    """Train + walk-forward + score. Returns the metrics dict; writes the report."""
    paths.ensure_out()
    out_dir = paths.out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    params = _params(seed)
    mat = _load_matrix(paths, since_ms=since_ms, until_ms=until_ms, days=days, series=series)
    result: dict = {
        "config": {"since_ms": since_ms, "until_ms": until_ms, "days": days, "series": series,
                   "init_train_days": init_train_days, "test_days": test_days, "seed": seed,
                   "fee_rate": fee_rate, "run_money": run_money, "run_shuffle": run_shuffle,
                   "network_ms": network_ms, "venue_delay_ms": venue_delay_ms},
        "features": {"base": BASE_FEATURES, "full": FULL_FEATURES},
        "caveats": CAVEATS, "n_rows_loaded": int(len(mat)),
        "targets": {}, "folds": {}, "per_block": {}, "final_models": {},
        "aggregate_eval": {}, "depth_gate": {},
    }
    if not len(mat):
        result["verdict"] = "No rows loaded for the given bounds."
        _write_report(paths, out_dir, result)
        return result

    unique_days = np.unique(mat["window_open_ms"].to_numpy(dtype="int64") // MS_PER_DAY)
    folds, mode = _expanding_folds(unique_days, init_train_days, test_days)
    result["fold_mode"] = mode
    result["n_folds"] = len(folds)

    oos_full_by_target: dict[str, pd.DataFrame] = {}
    for target in TARGET_ORDER:
        for variant in VARIANT_ORDER:
            feats = VARIANTS[variant]
            tag = f"{target}_{variant}"
            oos_path = out_dir / f"oos_{tag}.parquet"
            if oos_path.exists():  # resume: reuse cached OOS
                oos = pd.read_parquet(oos_path)
                recs = result["folds"].get(tag, [])
            else:
                sub, y, info = _prep_target(mat, target, fee_rate)
                purge_ms = (TARGETS[target]["horizon_secs"] or 300) * 1000
                oos, recs = _walk_forward(sub, y, info, feature_names=feats, folds=folds,
                                          purge_ms=purge_ms, params=params)
                oos.to_parquet(oos_path, index=False)
            result["folds"][tag] = recs
            pooled = _fold_metrics(oos["p_up"].to_numpy(dtype=float), oos["y_true"].to_numpy(dtype=float)) if len(oos) else {}
            result["targets"].setdefault(target, {})[variant] = {"n_oos": int(len(oos)), "pooled_own_label": pooled}
            if variant == "full":
                oos_full_by_target[target] = oos
                result["per_block"][target] = _per_block_table(oos, folds)
                result["depth_gate"][target] = _depth_split_gate(oos)
                result["aggregate_eval"][target] = _aggregate_eval(paths, oos, target=target, out_name=out_name)
                sub, y, info = _prep_target(mat, target, fee_rate)
                med = int(np.median([r["best_iteration"] for r in recs if not r["skipped"] and r["best_iteration"]])) if any(not r["skipped"] for r in recs) else None
                result["final_models"][target] = _final_model(sub, y, info, feature_names=feats, params=params, rounds=med)

    # headline depth gate = primary target's
    result["depth_gate"] = result["depth_gate"].get(TARGET_ORDER[0], {})

    if run_shuffle:  # needs mat (re-runs the walk-forward with permuted labels) → before we free it
        result["shuffled"] = _shuffled_control(mat, folds, params, seed, fee_rate)
    if run_money:
        import gc
        del mat  # the money step works off the OOS frames + Telonex books, not the matrix
        gc.collect()
        res_map = hr.load_resolution_map(paths, hc.DEFAULT_SERIES)
        money_since = _current_regime_since(since_ms)
        result["money"] = _money_score(paths, oos_full_by_target, since_ms=money_since,
                                       network_ms=network_ms, venue_delay_ms=venue_delay_ms,
                                       fee_rate=fee_rate, window_budget=window_budget,
                                       trade_budget=trade_budget, res_map=res_map, out_dir=out_dir)

    result["peak_rss_mb"] = procmem.peak_rss_mb()
    result["verdict"] = _verdict_text(result)
    _write_report(paths, out_dir, result)
    return result


def _current_regime_since(since_ms: int | None) -> int:
    floor = int(datetime(CURRENT_REGIME_FROM.year, CURRENT_REGIME_FROM.month, CURRENT_REGIME_FROM.day,
                         tzinfo=timezone.utc).timestamp()) * 1000
    return max(floor, since_ms) if since_ms is not None else floor


def _shuffled_control(mat: pd.DataFrame, folds, params, seed: int, fee_rate: float) -> dict:
    """Re-run the walk-forward (primary target, full variant) with permuted labels; the model
    must NOT extract signal (pooled Brier ~ chance) or the whole result is void."""
    target = TARGET_ORDER[0]
    sub, y, info = _prep_target(mat, target, fee_rate)
    rng = np.random.default_rng(seed)
    y_shuf = y[rng.permutation(len(y))]
    oos, _ = _walk_forward(sub, y_shuf, info, feature_names=FULL_FEATURES, folds=folds,
                           purge_ms=(TARGETS[target]["horizon_secs"] or 300) * 1000, params=params)
    if not len(oos):
        return {"collapsed": None, "reason": "no shuffled OOS"}
    p = oos["p_up"].to_numpy(dtype=float)
    yt = oos["y_true"].to_numpy(dtype=float)
    brier = lm.brier_score(p, yt)
    pos = float(np.mean(yt))
    chance = pos * (1.0 - pos)
    collapsed = bool(np.isfinite(brier) and brier >= chance - 0.03)
    return {"pooled_brier": float(brier), "chance_brier": float(chance),
            "collapsed": collapsed, "n_oos": int(len(oos))}


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "learn_walkforward")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default all 4)")
    parser.add_argument("--days", type=int, default=0, help="trailing days to load (0 = full history)")
    parser.add_argument("--init-train-days", type=int, default=DEFAULT_INIT_TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--network-ms", type=int, default=DEFAULT_NETWORK_MS)
    parser.add_argument("--venue-delay-ms", type=int, default=DEFAULT_VENUE_DELAY_MS)
    parser.add_argument("--no-money", action="store_true", help="skip money scoring (calibration only)")
    parser.add_argument("--no-shuffle", action="store_true", help="skip the shuffled-label control")
    parser.add_argument("--out-name", default="learn_walkforward")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = _parse_series(args.series)
    print(f"[learn_walkforward] dataset={paths.table('historical_dataset')} out={paths.out_dir / args.out_name} "
          f"latency=net {args.network_ms}+venue {args.venue_delay_ms}={args.network_ms + args.venue_delay_ms}ms")
    result = learn_walkforward(paths, since_ms=since_ms, until_ms=until_ms, days=args.days, series=series,
                               init_train_days=args.init_train_days, test_days=args.test_days, seed=args.seed,
                               run_money=not args.no_money, run_shuffle=not args.no_shuffle,
                               network_ms=args.network_ms, venue_delay_ms=args.venue_delay_ms,
                               out_name=args.out_name)
    print(f"[learn_walkforward] rows={result['n_rows_loaded']:,} folds={result.get('n_folds')} "
          f"mode={result.get('fold_mode')}")
    if result.get("money"):
        m = result["money"]
        print(f"[learn_walkforward] current-regime windows={m['current_regime_windows']:,} "
              f"momentum=${m['momentum']['net_pnl']:.2f}")
        for t, blk in m["targets"].items():
            b = blk["best"]
            print(f"  model '{t}': ${b['taker']['net_pnl']:.2f}" if b else f"  model '{t}': no trades")
    gate = result.get("depth_gate", {})
    if gate.get("pass") is False:
        print(f"[learn_walkforward] DEPTH GATE FAILED: {gate.get('reason')}")
    procmem.report_peak_rss("learn_walkforward", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[learn_walkforward] {result.get('verdict', '')}")
    print(f"[learn_walkforward] wrote report.html + metrics.json to {paths.out_dir / args.out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
