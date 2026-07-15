"""Multi-horizon × per-market accuracy sweep for the short-horizon model.

The production champion (``model_dir10_full``) predicts a single horizon: the
direction of the Binance mid **10 seconds** ahead (``fwd_up_10s``). The operator
wants to know, across a *range* of horizons and across the four markets, **where
the model is most accurate** and how far accuracy can realistically be pushed.

This stage answers that with one honest screening experiment. It reads the
existing ``historical_dataset.parquet`` (the exact 5-second-grid feature matrix
the champion trained on, all four markets, full 18-month history) and, because
the grid is a clean 5 s, reconstructs the forward-direction label at every
5-s-aligned horizon **by a self-join on the ``mid`` column** — verified to
reproduce the stored ``fwd_up_10s`` bit-for-bit. No journal re-read, no
aggTrades pass.

For each ``(market, horizon)`` cell it trains the *same* LightGBM the champion
uses (``lib/gbt`` defaults) on a chronological train split and scores it
out-of-sample on the held-out tail, reporting:

* directional accuracy, Brier, log-loss on the OOS tail,
* the **majority-class baseline** and the model's **lift** over it — the load-
  bearing honesty metric, because the label has a strong Up bias (~58 %) so a
  naive "always Up" already scores ~58 %,
* an **accuracy-vs-coverage (abstention) curve** — accuracy on the subset where
  ``|p-0.5| >= theta`` as theta rises — the realistic route to high accuracy
  (act only when confident),
* for context, the **formula's window-outcome accuracy bucketed by
  time-remaining** (no training) — where genuinely high accuracy already lives.

This is a *single chronological split* screening experiment (not the champion's
16-fold walk-forward): the split bias is shared across horizons, so the relative
ranking is robust, and the winner can get a full walk-forward follow-up. Results
are written to ``out/horizon_sweep/`` and printed as a readable table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import Paths
from .lib import math as lm
from .lib import procmem

# The champion's exact 24 features (order = model_dir10_full.meta.json).
FEATURES = [
    "ret", "realized_vol", "sigma_1s", "log_s_k", "z", "p_up_model",
    "tau_secs", "elapsed_secs", "basis_bps", "basis_ewma",
    "depth_imb_1", "depth_imb_5", "depth_imb_10", "depth_imb_20",
    "microprice_gap", "bid_depth_slope", "ask_depth_slope", "depth_spread_bps",
    "pm_mid", "pm_spread", "pm_book_imb",
    "pm_staleness_1s", "pm_staleness_2s", "pm_staleness_3s",
]

# 5-s-aligned horizons spanning the operator's requested 3-40 s range.
HORIZONS = [5, 10, 15, 20, 25, 30, 35, 40]
MARKETS = ["BTC-5m", "ETH-5m", "BTC-15m", "ETH-15m"]
THETAS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
TAU_BUCKETS = [(0, 20), (20, 60), (60, 150), (150, 10_000)]  # final_20s / final_min / mid / early
MS_PER_DAY = 86_400_000
DEFAULT_SEED = 20260713

_KEY_COLS = ["series", "window_open_ms", "sample_ts_ms", "tau_secs", "mid",
             "label_source", "outcome_up"]


# ===========================================================================
# data + labels
# ===========================================================================
def load_matrix(paths: Paths, *, days: int | None) -> pd.DataFrame:
    """Load the feature matrix (keys + 24 features + mid + outcome) from
    ``historical_dataset.parquet``, optionally restricted to the trailing
    ``days`` by window-open time. Features cast to float32 to bound memory."""
    import pyarrow.parquet as pq

    path = paths.out_dir / "historical_dataset.parquet"
    config.assert_parquet_ready(path, label="historical_dataset.parquet", min_rows=1)
    read_cols = list(dict.fromkeys(_KEY_COLS + FEATURES))
    filters = None
    if days and days > 0:
        opens = pq.read_table(path, columns=["window_open_ms"]).column(0).to_numpy()
        cutoff = int(opens.max()) - int(days) * MS_PER_DAY
        filters = [("window_open_ms", ">=", int(cutoff))]
    tbl = pq.read_table(path, columns=read_cols, filters=filters)
    df = tbl.to_pandas()
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    df = df.sort_values(["series", "window_open_ms", "sample_ts_ms"]).reset_index(drop=True)
    return df


def add_horizon_labels(df: pd.DataFrame, horizons: list[int]) -> dict[int, str]:
    """Append one direction label column per horizon via a shift on the sorted
    5-s grid, masked to rows where the +h sample truly exists in the same window
    (``series``/``window`` unchanged and ts gap == h*1000). Returns {h: colname}."""
    ser = df["series"].to_numpy()
    win = df["window_open_ms"].to_numpy()
    ts = df["sample_ts_ms"].to_numpy(np.int64)
    mid = df["mid"].to_numpy(np.float64)
    n = len(df)
    cols: dict[int, str] = {}
    for h in horizons:
        k = h // 5
        name = f"lab_{h}s"
        fwd_mid = np.empty(n, np.float64)
        fwd_mid[:] = np.nan
        if k < n:
            fwd_mid[: n - k] = mid[k:]
        # alignment: same series+window and exact ts gap.
        ok = np.zeros(n, bool)
        if k < n:
            same = np.zeros(n, bool)
            same[: n - k] = (ser[k:] == ser[: n - k]) & (win[k:] == win[: n - k])
            gap = np.full(n, -1, np.int64)
            gap[: n - k] = ts[k:] - ts[: n - k]
            ok = same & (gap == h * 1000) & ~np.isnan(fwd_mid)
        lab = np.where(fwd_mid >= mid, 1.0, 0.0)  # ties -> Up, matching label convention
        lab[~ok] = np.nan
        df[name] = lab.astype(np.float32)
        cols[h] = name
    df.drop(columns=["mid"], inplace=True)  # no longer needed; frees ~1 col
    return cols


# ===========================================================================
# metrics
# ===========================================================================
def abstention_curve(prob: np.ndarray, y: np.ndarray, thetas: list[float]) -> list[dict]:
    """Accuracy and coverage on the confident subset ``|p-0.5| >= theta``."""
    p = np.asarray(prob, float)
    y = np.asarray(y, float)
    pred = (p >= 0.5).astype(float)
    correct = (pred == y)
    n = len(y)
    out = []
    for t in thetas:
        m = np.abs(p - 0.5) >= t
        cov = int(m.sum())
        acc = float(correct[m].mean()) if cov else float("nan")
        out.append({"theta": t, "coverage": cov,
                    "coverage_frac": (cov / n) if n else float("nan"), "accuracy": acc})
    return out


def formula_outcome_by_tau(df: pd.DataFrame) -> list[dict]:
    """Directional accuracy & Brier of the formula (p_up_model) vs the window
    outcome, bucketed by time-remaining — the 'where high accuracy lives' view."""
    rows = []
    p = df["p_up_model"].to_numpy(np.float64)
    y = df["outcome_up"].to_numpy(np.float64)
    tau = df["tau_secs"].to_numpy(np.float64)
    ok = ~np.isnan(p) & ~np.isnan(y)
    for lo, hi in TAU_BUCKETS:
        m = ok & (tau >= lo) & (tau < hi)
        if not m.any():
            rows.append({"tau_lo": lo, "tau_hi": hi, "n": 0})
            continue
        rows.append({
            "tau_lo": lo, "tau_hi": hi, "n": int(m.sum()),
            "formula_diracc": lm.directional_accuracy(p[m], y[m]),
            "formula_brier": lm.brier_score(p[m], y[m]),
            "base_rate_up": float(np.mean(y[m])),
        })
    return rows


# ===========================================================================
# per-cell training
# ===========================================================================
def _split_days(days: np.ndarray, test_frac: float) -> int:
    u = np.unique(days)
    if len(u) < 2:
        return int(u.max()) + 1  # nothing to test
    n_test = max(1, round(test_frac * len(u)))
    return int(np.sort(u)[len(u) - n_test])


def run_cell(sub: pd.DataFrame, label_col: str, h: int, *, seed: int, cap: int,
             threads: int, max_rounds: int) -> dict | None:
    """Train the champion GBT on the chronological train split of ``sub`` for
    horizon ``h`` and score OOS on the held-out tail. ``sub`` is one market (or
    pooled), already carrying ``label_col``."""
    from .lib import gbt

    day = (sub["window_open_ms"].to_numpy(np.int64) // MS_PER_DAY)
    y_all = sub[label_col].to_numpy(np.float64)
    have = ~np.isnan(y_all)
    if have.sum() < 5000:
        return None
    test_start = _split_days(day[have], test_frac=0.25)
    test_start_ms = test_start * MS_PER_DAY
    label_info = sub["sample_ts_ms"].to_numpy(np.int64) + h * 1000
    train_m = have & (day < test_start) & (label_info <= test_start_ms)  # purge >= horizon
    test_m = have & (day >= test_start)
    if train_m.sum() < 5000 or test_m.sum() < 1000:
        return None

    x = sub[FEATURES].to_numpy(np.float32)
    tr_idx = np.flatnonzero(train_m)
    rng = np.random.default_rng(seed)
    if len(tr_idx) > cap:
        tr_idx = np.sort(rng.choice(tr_idx, size=cap, replace=False))
    # inner-val tail (latest 15% of the sampled train rows by ts) for early stopping.
    tr_ts = sub["sample_ts_ms"].to_numpy(np.int64)[tr_idx]
    order = np.argsort(tr_ts, kind="stable")
    tr_idx = tr_idx[order]
    n_val = max(1000, int(0.15 * len(tr_idx)))
    inner_tr, inner_val = tr_idx[:-n_val], tr_idx[-n_val:]

    params = gbt.default_params(seed, num_threads=threads)
    booster, best = gbt.fit(
        x[inner_tr], y_all[inner_tr], x[inner_val], y_all[inner_val],
        params=params, max_boost_round=max_rounds, early_stopping_rounds=40,
    )
    te_idx = np.flatnonzero(test_m)
    prob = gbt.predict_proba(booster, x[te_idx], num_iteration=best, num_threads=threads)
    y_te = y_all[te_idx]

    maj = lm.majority_baseline(y_te)
    diracc = lm.directional_accuracy(prob, y_te)
    return {
        "horizon_secs": h,
        "n_train": int(len(inner_tr)),
        "n_test": int(len(te_idx)),
        "test_pos_frac": float(np.mean(y_te)),
        "model_diracc": diracc,
        "model_brier": lm.brier_score(prob, y_te),
        "model_logloss": lm.log_loss(prob, y_te),
        "majority_diracc": maj["diracc"],
        "majority_brier": maj["brier"],
        "lift_pp": (diracc - maj["diracc"]) * 100.0,
        "best_iteration": int(best),
        "abstention": abstention_curve(prob, y_te, THETAS),
        "top_features": _top_gain(booster),
    }


def _top_gain(booster, k: int = 8) -> list[list]:
    from .lib import gbt
    imp = gbt.feature_importance(booster, FEATURES)["gain"]
    tot = sum(imp.values()) or 1.0
    ranked = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [[n, round(100.0 * v / tot, 1)] for n, v in ranked]


# ===========================================================================
# orchestration
# ===========================================================================
def sweep(paths: Paths, *, days: int | None, seed: int, cap: int, threads: int,
          max_rounds: int) -> dict:
    t0 = time.time()
    print(f"[horizon_sweep] loading matrix (days={days or 'all'}) ...", flush=True)
    df = load_matrix(paths, days=days)
    print(f"[horizon_sweep] loaded {len(df):,} rows in {time.time()-t0:.0f}s; building labels ...",
          flush=True)
    cols = add_horizon_labels(df, HORIZONS)

    scopes = {m: df[df["series"] == m] for m in MARKETS}
    scopes["ALL"] = df

    results: dict = {"config": {"days": days, "seed": seed, "cap": cap,
                                "threads": threads, "max_rounds": max_rounds,
                                "horizons": HORIZONS},
                     "cells": {}, "formula_outcome_by_tau": {}}
    for scope, sdf in scopes.items():
        print(f"\n[horizon_sweep] === market {scope} ({len(sdf):,} rows) ===", flush=True)
        results["formula_outcome_by_tau"][scope] = formula_outcome_by_tau(sdf)
        results["cells"][scope] = []
        for h in HORIZONS:
            ct = time.time()
            cell = run_cell(sdf, cols[h], h, seed=seed, cap=cap, threads=threads,
                            max_rounds=max_rounds)
            if cell is None:
                print(f"  h={h:>2}s  SKIPPED (insufficient rows)", flush=True)
                continue
            results["cells"][scope].append(cell)
            print(f"  h={h:>2}s  diracc={cell['model_diracc']:.4f}  "
                  f"majority={cell['majority_diracc']:.4f}  lift={cell['lift_pp']:+.2f}pp  "
                  f"brier={cell['model_brier']:.4f}  n_test={cell['n_test']:,}  "
                  f"({time.time()-ct:.0f}s)", flush=True)
            # incremental write so a killed run keeps partial results.
            _write(paths, results)
    results["elapsed_secs"] = round(time.time() - t0, 1)
    _write(paths, results)
    return results


def _write(paths: Paths, results: dict) -> None:
    out = paths.out_dir / "horizon_sweep"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=_jd), encoding="utf-8")
    # flat CSV of cells for quick reading.
    rows = []
    for scope, cells in results["cells"].items():
        for c in cells:
            rows.append({"market": scope, "horizon_secs": c["horizon_secs"],
                         "model_diracc": c["model_diracc"],
                         "majority_diracc": c["majority_diracc"], "lift_pp": c["lift_pp"],
                         "model_brier": c["model_brier"], "n_test": c["n_test"],
                         "test_pos_frac": c["test_pos_frac"]})
    if rows:
        pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)


def _jd(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> int:
    ap = config.stage_parser("Multi-horizon x per-market accuracy sweep.")
    ap.add_argument("--days", type=int, default=0, help="trailing window-days (0 = all history).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cap", type=int, default=1_200_000, help="max train rows per cell.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=400)
    args = ap.parse_args(argv)
    paths = config.resolve_paths(args)

    res = sweep(paths, days=(args.days or None), seed=args.seed, cap=args.cap,
                threads=args.threads, max_rounds=args.max_rounds)
    procmem.report_peak_rss("horizon_sweep", procmem.peak_rss_mb(),
                            getattr(args, "max_rss_mb", config.DEFAULT_MAX_RSS_MB))
    print(f"\n[horizon_sweep] done in {res['elapsed_secs']}s -> "
          f"{paths.out_dir / 'horizon_sweep'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
