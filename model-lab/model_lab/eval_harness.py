"""The evaluation harness — the single source of truth for scoring a model.

Given *any* model's probability-of-Up outputs over a set of resolved windows, this
module scores them with four metrics — **Brier**, **log-loss**, **directional
accuracy**, and a **calibration (reliability) curve** — and always reports them
**side by side against two benchmarks**:

1. the **formula model** probability (``crates/model`` ``p_up``) recomputed at the
   same timestamps, and
2. the **market-implied** probability — the Polymarket Up-token order-book mid
   ``(best_bid + best_ask)/2`` — on the subset of timestamps where recordings
   exist.

It emits **one standardized verdict table per evaluation**: the model vs each
benchmark, broken out **per test period** (calendar day, with a weekly rollup),
with **relative-improvement percentages** and a **stability** summary across
periods (how often, and how consistently, the model beats each benchmark).

This is a **library**, not a stage (it has no CLI of its own — like
``lib/math.py``). Two stages call it: :mod:`model_lab.evaluate` (the general CLI
face, ``python -m model_lab.evaluate``) and :mod:`model_lab.calibration_audit`
(the formula model as a self-baseline). Both build a *canonical eval frame* and
hand it to :func:`run`; every later stage that scores a model must report through
this module so "is this model any good?" is answered the same way everywhere.

Canonical eval frame — one row per scored snapshot::

    series, open_time, ts, tau, tau_bucket, day, week,
    outcome_up, model, formula, market, in_coverage, health

``model`` is the probability of the model under test; ``formula`` is benchmark 1;
``market`` is benchmark 2 (``NaN`` outside coverage). ``day = open_time //
86_400_000`` (UTC, mirrors ``crates/analytics`` ``DayKey``); ``week = day // 7``.
"""

from __future__ import annotations

import base64
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG, never open a window.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Paths, _in_bounds  # noqa: E402
from .io import journal as jio  # noqa: E402
from .lib import math as lm  # noqa: E402

# --- tuning constants -------------------------------------------------------
N_BINS = 10
DEFAULT_MIN_WINDOWS = 50
MS_PER_DAY = 86_400_000
# Don't pair a snapshot with an Up-token mid older than this (guards a window
# whose book went silent). Books on an active pair update ~1/s.
MAX_MID_STALENESS_MS = 120_000
# The formula benchmark "recomputed at the same timestamps" = the nearest earlier
# engine ``p_up`` snapshot (the engine emits ~1/s). Don't reach back further.
FORMULA_STALENESS_MS = 5_000
# A model's per-period edge is "reliable" when it wins a clear majority of the
# trustworthy periods AND its mean improvement points the right way.
RELIABLE_WIN_RATE = 0.6

# The three predictors, and which two are benchmarks the model is scored against.
PREDICTORS = ["model", "formula", "market"]
BENCHMARKS = ["formula", "market"]

# Time-remaining buckets, in display order (early → about to resolve).
TAU_ORDER = ["early", "mid", "final_minute", "final_20s"]
TAU_LABELS = {
    "early": "early (τ > ½·window)",
    "mid": "mid (60s–½·window)",
    "final_minute": "final minute (20–60s)",
    "final_20s": "final 20s (≤ 20s)",
}

# Canonical eval-frame columns (an empty frame still carries these).
CANONICAL_COLS = [
    "series", "open_time", "ts", "tau", "tau_bucket", "day", "week",
    "outcome_up", "model", "formula", "market", "in_coverage", "health",
]


# ===========================================================================
# journal → tidy frames  (moved from calibration_audit — the shared assembly)
# ===========================================================================
def read_journal(
    paths: Paths, *, since_ms: int | None = None, until_ms: int | None = None
) -> tuple[list[dict], list[tuple], dict, dict, dict]:
    """One pass over the journal. Returns ``(model_rows, top_rows, win_meta,
    outcome_resolved, outcome_settle)``.

    ``win_meta[(series, open_time)] = {close_time, up_token, down_token}`` from any
    ``window`` record; ``outcome_resolved`` from the ``Resolved`` lifecycle (fires
    for every resolved window), ``outcome_settle`` from ``settlement`` (traded
    windows only) as a fallback. ``since_ms``/``until_ms`` bound records by time
    (model/book by ts, windows/settlements by open time) to cap memory + scope.
    """
    model_rows: list[dict] = []
    top_rows: list[tuple] = []  # (token_id, ts, mid)
    win_meta: dict[tuple[str, int], dict] = {}
    outcome_resolved: dict[tuple[str, int], str] = {}
    outcome_settle: dict[tuple[str, int], str] = {}

    for rec in jio.read_records(paths.journal_dir):
        kind = rec.get("type")
        if kind == "model":
            window = rec.get("window")
            if not window:
                continue
            ts = rec.get("ts")
            if not _in_bounds(int(ts) if ts is not None else None, since_ms, until_ms):
                continue
            series, open_time = jio.window_key(window)
            model_rows.append(
                {
                    "series": series,
                    "open_time": open_time,
                    "ts": ts,
                    "p_up": jio.to_float(rec.get("p_up")),
                    "health": rec.get("health"),
                }
            )
        elif kind == "top_of_book":
            top = rec.get("top") or {}
            bid = top.get("bid")
            ask = top.get("ask")
            ts = top.get("ts")
            if not isinstance(bid, dict) or not isinstance(ask, dict):
                continue  # one-sided book → no mid
            if not _in_bounds(int(ts) if ts is not None else None, since_ms, until_ms):
                continue
            bid_p = jio.to_float(bid.get("price"))
            ask_p = jio.to_float(ask.get("price"))
            if not (np.isfinite(bid_p) and np.isfinite(ask_p)):
                continue
            top_rows.append((rec.get("token_id"), ts, (bid_p + ask_p) / 2.0))
        elif kind == "window":
            market = rec.get("market") or {}
            key = jio.window_key(market.get("window"))
            if key == ("", 0) or not _in_bounds(key[1], since_ms, until_ms):
                continue
            if key not in win_meta:
                tokens = market.get("tokens") or {}
                win_meta[key] = {
                    "close_time": market.get("close_time"),
                    "up_token": tokens.get("up"),
                    "down_token": tokens.get("down"),
                }
            lifecycle = rec.get("lifecycle")
            if isinstance(lifecycle, dict) and "Resolved" in lifecycle:
                outcome = (lifecycle["Resolved"] or {}).get("outcome")
                if outcome in ("Up", "Down"):
                    outcome_resolved[key] = outcome
        elif kind == "settlement":
            key = jio.window_key(rec.get("window"))
            outcome = rec.get("outcome")
            if key != ("", 0) and outcome in ("Up", "Down") and _in_bounds(key[1], since_ms, until_ms):
                outcome_settle[key] = outcome

    return model_rows, top_rows, win_meta, outcome_resolved, outcome_settle


def windows_frame(win_meta: dict, outcome_resolved: dict, outcome_settle: dict) -> pd.DataFrame:
    """Resolved windows only: ``series, open_time, close_time, up_token,
    down_token, outcome_up, dur`` (outcome from Resolved, else settlement)."""
    rows: list[dict] = []
    for key, meta in win_meta.items():
        close_time = meta.get("close_time")
        outcome = outcome_resolved.get(key) or outcome_settle.get(key)
        if close_time is None or outcome not in ("Up", "Down"):
            continue
        series, open_time = key
        rows.append(
            {
                "series": series,
                "open_time": int(open_time),
                "close_time": int(close_time),
                "up_token": meta.get("up_token"),
                "down_token": meta.get("down_token"),
                "outcome_up": 1 if outcome == "Up" else 0,
                "dur": (int(close_time) - int(open_time)) / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def market_frame(top_rows: list[tuple], windows: pd.DataFrame) -> pd.DataFrame:
    """Up-token tops → ``series, open_time, ts_mid, up_mid``.

    A top is assigned to the window whose Up token it carries *and* whose
    ``[open, close)`` contains its timestamp. Down-token / unmatched tops dropped.
    """
    if windows.empty or not top_rows:
        return pd.DataFrame(columns=["series", "open_time", "ts_mid", "up_mid"])
    up_windows: dict[Any, list[tuple[int, int, str]]] = defaultdict(list)
    for w in windows.itertuples(index=False):
        up_windows[w.up_token].append((w.open_time, w.close_time, w.series))

    rows: list[dict] = []
    for token_id, ts, mid in top_rows:
        cands = up_windows.get(token_id)
        if not cands or ts is None:
            continue
        for open_time, close_time, series in cands:
            if open_time <= ts < close_time:
                rows.append({"series": series, "open_time": open_time, "ts_mid": int(ts), "up_mid": float(mid)})
                break
    return pd.DataFrame(rows, columns=["series", "open_time", "ts_mid", "up_mid"])


def tau_bucket(tau: np.ndarray, dur: np.ndarray) -> np.ndarray:
    """Duration-aware time-remaining bucket (first matching condition wins)."""
    return np.select(
        [tau <= 20.0, tau <= 60.0, tau <= 0.5 * dur],
        ["final_20s", "final_minute", "mid"],
        default="early",
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLS)


def _attach_market(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """As-of join the last Up-token mid at/before each row's ``ts`` (per window),
    NaN if older than :data:`MAX_MID_STALENESS_MS`. Adds a ``market`` column."""
    if market.empty:
        df = df.copy()
        df["market"] = np.nan
        return df
    right = market.sort_values("ts_mid").rename(columns={"ts_mid": "ts", "up_mid": "market"})
    right = right[["series", "open_time", "ts", "market"]].assign(_mid_ts=lambda r: r["ts"])
    merged = pd.merge_asof(
        df.sort_values("ts"), right, on="ts", by=["series", "open_time"], direction="backward"
    )
    stale = ~np.isfinite(merged["_mid_ts"]) | ((merged["ts"] - merged["_mid_ts"]) > MAX_MID_STALENESS_MS)
    merged.loc[stale, "market"] = np.nan
    return merged.drop(columns=["_mid_ts"])


def _attach_formula(df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """As-of join the nearest earlier engine ``p_up`` (+ its ``health``) at each
    row's ``ts`` (per window) — "the formula model recomputed at the same
    timestamps". NaN beyond :data:`FORMULA_STALENESS_MS`. Adds ``formula`` +
    ``health`` columns."""
    if model_df.empty:
        df = df.copy()
        df["formula"] = np.nan
        df["health"] = None
        return df
    right = model_df[["series", "open_time", "ts", "p_up", "health"]].dropna(subset=["ts"]).copy()
    right["ts"] = right["ts"].astype("int64")
    right = right.sort_values("ts").rename(columns={"p_up": "formula"}).assign(_f_ts=lambda r: r["ts"])
    merged = pd.merge_asof(
        df.sort_values("ts"),
        right[["series", "open_time", "ts", "formula", "health", "_f_ts"]],
        on="ts", by=["series", "open_time"], direction="backward",
    )
    stale = ~np.isfinite(merged["_f_ts"]) | ((merged["ts"] - merged["_f_ts"]) > FORMULA_STALENESS_MS)
    merged.loc[stale, "formula"] = np.nan
    merged.loc[stale, "health"] = None
    return merged.drop(columns=["_f_ts"])


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``tau_bucket``, ``day``, ``week``, ``in_coverage`` and cast ``outcome_up``."""
    df = df.copy()
    df["outcome_up"] = df["outcome_up"].astype(float)
    df["tau_bucket"] = tau_bucket(df["tau"].to_numpy(dtype=float), df["dur"].to_numpy(dtype=float))
    df["day"] = (df["open_time"].astype("int64") // MS_PER_DAY).astype("int64")
    df["week"] = (df["day"] // 7).astype("int64")
    df["in_coverage"] = df["market"].notna()
    return df


def build_self_baseline_frame(
    paths: Paths, *, since_ms: int | None = None, until_ms: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """The canonical frame for the **formula model as the model under test**
    (``model = formula = p_up``, ``market = up_mid``). Journal-only, no prereq
    stages — the basis of the calibration audit and ``evaluate``'s default run.
    ``since_ms``/``until_ms`` bound the journal by time."""
    model_rows, top_rows, win_meta, res, settle = read_journal(paths, since_ms=since_ms, until_ms=until_ms)
    windows = windows_frame(win_meta, res, settle)
    market = market_frame(top_rows, windows)
    counts = {
        "resolved_windows": int(len(windows)),
        "model_snapshots": int(len(model_rows)),
        "market_tops": int(len(top_rows)),
    }
    model = pd.DataFrame(model_rows)
    if model.empty or windows.empty:
        counts["joined_snapshots"] = 0
        return _empty_frame(), counts
    df = model.merge(
        windows[["series", "open_time", "close_time", "outcome_up", "dur"]],
        on=["series", "open_time"], how="inner",
    )
    df = df[df["ts"].notna()].copy()
    if df.empty:
        counts["joined_snapshots"] = 0
        return _empty_frame(), counts
    df["ts"] = df["ts"].astype("int64")
    df["tau"] = (df["close_time"] - df["ts"]) / 1000.0
    df = df[df["tau"] > 0.0].copy()
    if df.empty:
        counts["joined_snapshots"] = 0
        return _empty_frame(), counts
    df = _attach_market(df, market)
    df["model"] = df["p_up"]
    df["formula"] = df["p_up"]
    df = _finalize(df)
    counts["joined_snapshots"] = int(len(df))
    return df, counts


def load_benchmarks(
    paths: Paths, *, since_ms: int | None = None, until_ms: int | None = None
) -> dict:
    """Read the journal once and build the reusable benchmark context — resolved
    ``windows``, the ``market`` (Up-token mid) frame, and the engine ``model``
    snapshots. Scoring several prediction grids against the same journal (e.g. a
    learner's OOS grid *and* its shuffled-control grid) should build this **once**
    and reuse it via :func:`build_external_frame_from`, rather than re-reading the
    (potentially multi-GB) journal per grid. ``since_ms``/``until_ms`` bound by time."""
    model_rows, top_rows, win_meta, res, settle = read_journal(paths, since_ms=since_ms, until_ms=until_ms)
    windows = windows_frame(win_meta, res, settle)
    market = market_frame(top_rows, windows)
    return {
        "windows": windows,
        "market": market,
        "model": pd.DataFrame(model_rows),
        "counts": {
            "resolved_windows": int(len(windows)),
            "model_snapshots": int(len(model_rows)),
            "market_tops": int(len(top_rows)),
        },
    }


def build_external_frame_from(ctx: dict, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Like :func:`build_external_frame`, but against a pre-built benchmark context
    from :func:`load_benchmarks` (so one journal read is shared across many grids)."""
    windows = ctx["windows"]
    counts = dict(ctx["counts"])
    counts["prediction_rows"] = int(len(predictions))
    rename = {"window_open_ms": "open_time", "sample_ts_ms": "ts", "p_up": "model"}
    preds = predictions.rename(columns=rename)
    need = {"series", "open_time", "ts", "model"}
    missing = need - set(preds.columns)
    if missing:
        raise ValueError(
            f"predictions missing required column(s) {sorted(missing)}; expected "
            "(series, window_open_ms, sample_ts_ms, p_up)."
        )
    preds = preds[["series", "open_time", "ts", "model"]].copy()
    if preds.empty or windows.empty:
        counts["joined_snapshots"] = 0
        return _empty_frame(), counts
    preds = preds.merge(
        windows[["series", "open_time", "close_time", "outcome_up", "dur"]],
        on=["series", "open_time"], how="inner",
    )
    preds = preds[preds["ts"].notna()].copy()
    if preds.empty:
        counts["joined_snapshots"] = 0
        return _empty_frame(), counts
    preds["ts"] = preds["ts"].astype("int64")
    preds["tau"] = (preds["close_time"] - preds["ts"]) / 1000.0
    preds = preds[preds["tau"] > 0.0].copy()
    if preds.empty:
        counts["joined_snapshots"] = 0
        return _empty_frame(), counts
    preds = _attach_formula(preds, ctx["model"])
    preds = _attach_market(preds, ctx["market"])
    preds = _finalize(preds)
    counts["joined_snapshots"] = int(len(preds))
    return preds, counts


def build_external_frame(
    paths: Paths, predictions: pd.DataFrame, *, since_ms: int | None = None, until_ms: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """The canonical frame for an **external model's predictions** — a grid parquet
    keyed ``(series, window_open_ms, sample_ts_ms)`` + ``p_up``. The formula and
    market benchmarks are recomputed at the same timestamps from the journal.
    ``since_ms``/``until_ms`` bound the journal by time."""
    return build_external_frame_from(load_benchmarks(paths, since_ms=since_ms, until_ms=until_ms), predictions)


# ===========================================================================
# scoring
# ===========================================================================
def _n_windows(df: pd.DataFrame) -> int:
    return int(df.groupby(["series", "open_time"]).ngroups) if len(df) else 0


def _predictor_metrics(df: pd.DataFrame, col: str) -> dict:
    """Brier / log-loss / directional accuracy for one predictor column."""
    d = df.dropna(subset=[col])
    n = int(len(d))
    if n == 0:
        return {"n": 0, "brier": float("nan"), "logloss": float("nan"), "diracc": float("nan")}
    prob = d[col].to_numpy(dtype=float)
    out = d["outcome_up"].to_numpy(dtype=float)
    return {
        "n": n,
        "brier": lm.brier_score(prob, out),
        "logloss": lm.log_loss(prob, out),
        "diracc": lm.directional_accuracy(prob, out),
    }


def _score(df: pd.DataFrame, min_windows: int) -> dict:
    """All three predictors over ``df`` (rows already in a resolved window,
    ``tau > 0``). Carries ``{pred}_brier / _logloss / _diracc`` for each."""
    n_snap = int(len(df))
    nw = _n_windows(df)
    row: dict = {"n_snapshots": n_snap, "n_windows": nw, "low_sample": bool(nw < min_windows)}
    for pred in PREDICTORS:
        m = _predictor_metrics(df, pred)
        row[f"n_{pred}"] = m["n"]
        row[f"{pred}_brier"] = m["brier"]
        row[f"{pred}_logloss"] = m["logloss"]
        row[f"{pred}_diracc"] = m["diracc"]
    return row


def _scope_rows(scored: pd.DataFrame, all_health: pd.DataFrame, min_windows: int) -> list[dict]:
    """The flat list of scored breakdowns, each tagged with ``scope`` + dims."""
    rows: list[dict] = []

    def add(scope: str, df: pd.DataFrame, *, series: str = "", tau_bucket: str = "") -> None:
        rows.append({"scope": scope, "series": series, "tau_bucket": tau_bucket, **_score(df, min_windows)})

    add("overall", scored)
    add("overall_all_health", all_health)
    add("overall_headtohead", scored.dropna(subset=["model", "market"]))
    for series in sorted(scored["series"].dropna().unique()):
        add("series", scored[scored["series"] == series], series=series)
    for bucket in TAU_ORDER:
        g = scored[scored["tau_bucket"] == bucket]
        if not g.empty:
            add("tau_bucket", g, tau_bucket=bucket)
    for series in sorted(scored["series"].dropna().unique()):
        for bucket in TAU_ORDER:
            g = scored[(scored["series"] == series) & (scored["tau_bucket"] == bucket)]
            if not g.empty:
                add("series_tau", g, series=series, tau_bucket=bucket)
    return rows


def _reliability(df: pd.DataFrame, col: str, min_windows: int, n_bins: int = N_BINS) -> list[dict]:
    """Reliability bins for ``col`` with snapshot and distinct-window counts."""
    d = df.dropna(subset=[col])
    if d.empty:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(d[col].to_numpy(dtype=float), edges, right=False) - 1, 0, n_bins - 1)
    d = d.assign(_bin=idx)
    out: list[dict] = []
    for b in range(n_bins):
        g = d[d["_bin"] == b]
        if g.empty:
            continue
        nw = _n_windows(g)
        out.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "bin_mid": float((edges[b] + edges[b + 1]) / 2.0),
                "n_snapshots": int(len(g)),
                "n_windows": nw,
                "mean_pred": float(g[col].mean()),
                "empirical_rate": float(g["outcome_up"].mean()),
                "low_sample": bool(nw < min_windows),
            }
        )
    return out


def _ece(reliability: list[dict]) -> float:
    """Expected calibration error: snapshot-weighted mean |pred − actual|."""
    if not reliability:
        return float("nan")
    n = sum(r["n_snapshots"] for r in reliability)
    if n == 0:
        return float("nan")
    return sum(r["n_snapshots"] * abs(r["mean_pred"] - r["empirical_rate"]) for r in reliability) / n


# ===========================================================================
# the standardized verdict table (model vs each benchmark, per period)
# ===========================================================================
def _improve_pct(model_v: float, bench_v: float) -> float:
    """Relative improvement of a lower-is-better metric; positive = model better."""
    if not (np.isfinite(model_v) and np.isfinite(bench_v)) or bench_v == 0.0:
        return float("nan")
    return (bench_v - model_v) / bench_v * 100.0


def _compare(df: pd.DataFrame, benchmark: str, min_windows: int) -> dict:
    """Score the model against one benchmark on the rows where **both** are
    defined (apples-to-apples). ``coverage`` = that fraction of model rows."""
    total = int(len(df.dropna(subset=["model"])))
    d = df.dropna(subset=["model", benchmark])
    n = int(len(d))
    nw = _n_windows(d)
    if n == 0:
        mb = bb = ml = bl = md = bd = float("nan")
    else:
        out = d["outcome_up"].to_numpy(dtype=float)
        pm = d["model"].to_numpy(dtype=float)
        pb = d[benchmark].to_numpy(dtype=float)
        mb, bb = lm.brier_score(pm, out), lm.brier_score(pb, out)
        ml, bl = lm.log_loss(pm, out), lm.log_loss(pb, out)
        md, bd = lm.directional_accuracy(pm, out), lm.directional_accuracy(pb, out)
    delta_pp = (md - bd) * 100.0 if (np.isfinite(md) and np.isfinite(bd)) else float("nan")
    return {
        "benchmark": benchmark,
        "n_snapshots": n,
        "n_windows": nw,
        "coverage": (n / total) if total else float("nan"),
        "model_brier": mb, "bench_brier": bb, "brier_improve_pct": _improve_pct(mb, bb),
        "model_logloss": ml, "bench_logloss": bl, "logloss_improve_pct": _improve_pct(ml, bl),
        "model_diracc": md, "bench_diracc": bd, "diracc_delta_pp": delta_pp,
        "low_sample": bool(nw < min_windows),
    }


def _period_label(by: str, value: int) -> str:
    days = int(value) * (7 if by == "week" else 1)
    d = datetime.fromtimestamp(days * 86400, tz=timezone.utc).date().isoformat()
    return f"wk {d}" if by == "week" else d


def _verdict_rows(df: pd.DataFrame, min_windows: int, by: str) -> list[dict]:
    """One row per (period, benchmark) + an ``ALL`` aggregate row per benchmark."""
    rows: list[dict] = []
    for value in sorted(df[by].dropna().unique()):
        sl = df[df[by] == value]
        label = _period_label(by, int(value))
        for b in BENCHMARKS:
            rows.append({"by": by, "period_key": int(value), "period": label, **_compare(sl, b, min_windows)})
    for b in BENCHMARKS:
        rows.append({"by": by, "period_key": -1, "period": "ALL", **_compare(df, b, min_windows)})
    return rows


def _stability(rows: list[dict], benchmark: str) -> dict:
    """Across the trustworthy per-period rows for one benchmark: how often and how
    consistently the model beats it (by Brier)."""
    per = [
        r for r in rows
        if r["benchmark"] == benchmark and r["period"] != "ALL"
        and not r["low_sample"] and np.isfinite(r["brier_improve_pct"])
    ]
    n = len(per)
    if n == 0:
        return {
            "periods_evaluated": 0, "periods_model_better": 0, "win_rate": float("nan"),
            "mean_brier_improve_pct": float("nan"), "std_brier_improve_pct": float("nan"),
            "reliably_better": False,
        }
    imps = np.array([r["brier_improve_pct"] for r in per], dtype=float)
    better = int(np.sum(imps > 0.0))
    win_rate = better / n
    mean = float(imps.mean())
    return {
        "periods_evaluated": n,
        "periods_model_better": better,
        "win_rate": win_rate,
        "mean_brier_improve_pct": mean,
        "std_brier_improve_pct": float(imps.std()),
        "reliably_better": bool(win_rate >= RELIABLE_WIN_RATE and mean > 0.0),
    }


# ===========================================================================
# verdict text
# ===========================================================================
def fmt(x: Any, dp: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{dp}f}" if isinstance(x, float) else str(x)


def find_scope(rows: list[dict], scope: str, **dims) -> dict | None:
    for r in rows:
        if r["scope"] == scope and all(r.get(k) == v for k, v in dims.items()):
            return r
    return None


def _bench_row(rows: list[dict], benchmark: str) -> dict | None:
    for r in rows:
        if r["benchmark"] == benchmark and r["period"] == "ALL":
            return r
    return None


def _lean(cmp: dict) -> str:
    imp = cmp.get("brier_improve_pct", float("nan"))
    if not np.isfinite(imp):
        return "cannot be compared (no overlapping data)"
    if imp > 0.5:
        return f"beats the {cmp['benchmark']} (Brier {fmt(cmp['model_brier'])} vs {fmt(cmp['bench_brier'])}, {imp:+.1f}%)"
    if imp < -0.5:
        return f"trails the {cmp['benchmark']} (Brier {fmt(cmp['model_brier'])} vs {fmt(cmp['bench_brier'])}, {imp:+.1f}%)"
    return f"ties the {cmp['benchmark']} (Brier {fmt(cmp['model_brier'])} vs {fmt(cmp['bench_brier'])})"


def _verdict_text(scope_rows: list[dict], vt: dict, model_rel: list[dict], min_windows: int, params: dict) -> str:
    overall = find_scope(scope_rows, "overall") or {}
    nw = overall.get("n_windows", 0)
    n_model = overall.get("n_model", 0)
    if nw == 0 or n_model == 0:
        return "No resolved windows with scored snapshots were found — nothing to evaluate."

    label = params.get("model_label", "model")
    parts: list[str] = [
        f"Over {nw} resolved window(s) and {n_model:,} scored snapshot(s) "
        f"(health={params.get('health')}), {label} scores a Brier of "
        f"{fmt(overall.get('model_brier'))}, log-loss {fmt(overall.get('model_logloss'))}, and "
        f"directional accuracy {fmt(overall.get('model_diracc'))} (Brier/log-loss lower-is-better)."
    ]

    day_rows = vt["day"]["rows"]
    if params.get("self_baseline"):
        parts.append("This is a self-baseline: the model under test IS the formula model, so the "
                     "model-vs-formula comparison is an identity (~0%) — the market comparison is the signal.")
    else:
        fcmp = _bench_row(day_rows, "formula")
        if fcmp:
            parts.append(f"Against the formula model recomputed at the same timestamps, {label} {_lean(fcmp)}.")
    kcmp = _bench_row(day_rows, "market")
    if kcmp and np.isfinite(kcmp.get("model_brier", float("nan"))):
        cov = kcmp.get("coverage", float("nan"))
        cov_txt = f"{cov:.0%} of snapshots" if np.isfinite(cov) else "the recorded subset"
        parts.append(f"On {cov_txt} where a Polymarket mid exists, {label} {_lean(kcmp)}.")
    elif kcmp:
        parts.append("No market order-book (top_of_book) data overlapped these snapshots — model-only scoring.")

    # Stability across daily periods.
    for b in BENCHMARKS:
        if params.get("self_baseline") and b == "formula":
            continue
        st = vt["day"]["stability"][b]
        if st["periods_evaluated"] > 0:
            verb = "reliably beats" if st["reliably_better"] else "does not reliably beat"
            parts.append(
                f"Across {st['periods_evaluated']} trustworthy daily period(s) it {verb} the {b}: "
                f"ahead in {st['periods_model_better']}/{st['periods_evaluated']} "
                f"(win rate {st['win_rate']:.0%}), mean Brier improvement "
                f"{st['mean_brier_improve_pct']:+.1f}% ± {fmt(st['std_brier_improve_pct'], 1)}%."
            )

    # Near-close sharpening (model).
    early = find_scope(scope_rows, "tau_bucket", tau_bucket="early")
    fin = find_scope(scope_rows, "tau_bucket", tau_bucket="final_20s")
    if fin and np.isfinite(fin.get("model_brier", float("nan"))):
        trust = "" if not fin.get("low_sample") else " (⚠ too few windows to trust)"
        early_b = fmt(early.get("model_brier")) if early else "n/a"
        sharpens = (early and np.isfinite(early.get("model_brier", float("nan")))
                    and fin["model_brier"] < early["model_brier"])
        parts.append(
            f"In the final 20 seconds its Brier is {fmt(fin.get('model_brier'))}{trust}, versus {early_b} "
            + ("early — it sharpens as resolution nears." if sharpens else "early (see the time-remaining table).")
        )

    low = [r for r in scope_rows if r["scope"] in ("series", "tau_bucket", "series_tau") and r["low_sample"]]
    if low:
        parts.append(
            f"⚠ {len(low)} breakdown(s) are backed by fewer than {min_windows} distinct windows and should not "
            "be trusted; they are flagged low_sample in the tables."
        )
    else:
        parts.append(f"Every breakdown is backed by at least {min_windows} windows.")
    return " ".join(parts)


# ===========================================================================
# rendering
# ===========================================================================
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f"<p class='muted'>{alt}: not available.</p>"
    return f"<img alt='{alt}' src='data:image/png;base64,{b64}'/>"


_PRED_STYLE = {"model": ("#1f77b4", "o-"), "formula": ("#2ca02c", "^--"), "market": ("#ff7f0e", "s-")}


def _reliability_png(reliability: dict) -> str | None:
    if not any(reliability.get(p) for p in PREDICTORS):
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    for pred in PREDICTORS:
        rel = reliability.get(pred)
        if not rel:
            continue
        rdf = pd.DataFrame(rel)
        color, style = _PRED_STYLE[pred]
        ax.plot(rdf["mean_pred"], rdf["empirical_rate"], style, color=color, label=pred)
    ax.set_xlabel("mean predicted P(Up)")
    ax.set_ylabel("empirical Up rate")
    ax.set_title("Calibration (reliability)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _png(fig)


def _period_png(day_rows: list[dict]) -> str | None:
    """Brier improvement % of the model vs each benchmark, per daily period."""
    per = [r for r in day_rows if r["period"] != "ALL"]
    if not per:
        return None
    periods = sorted({r["period"] for r in per})
    if not periods:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(periods))
    width = 0.38
    for i, b in enumerate(BENCHMARKS):
        color = _PRED_STYLE[b][0]
        by_period = {r["period"]: r.get("brier_improve_pct", float("nan")) for r in per if r["benchmark"] == b}
        vals = [by_period.get(p, float("nan")) for p in periods]
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=color, label=f"vs {b}")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Brier improvement % (↑ = model better)")
    ax.set_title("Model edge per period")
    ax.legend()
    return _png(fig)


def _table_html(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>no rows.</p>"
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = []
    for r in rows:
        cls = " class='low'" if r.get("low_sample") else ""
        cells = []
        for key, _ in columns:
            v = r.get(key)
            if key == "low_sample":
                cells.append("<td>⚠</td>" if v else "<td></td>")
            elif isinstance(v, float):
                cells.append(f"<td>{fmt(v)}</td>")
            else:
                cells.append(f"<td>{'' if v is None else v}</td>")
        body.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _stability_line(vt: dict, benchmark: str) -> str:
    st = vt["day"]["stability"][benchmark]
    if st["periods_evaluated"] == 0:
        return f"<p class='muted'>vs {benchmark}: no trustworthy daily periods (need ≥ min-windows per day).</p>"
    tag = "reliably better" if st["reliably_better"] else "not reliably better"
    return (
        f"<p><b>vs {benchmark}</b>: {tag} — ahead in {st['periods_model_better']}/"
        f"{st['periods_evaluated']} daily periods (win rate {st['win_rate']:.0%}), mean Brier "
        f"improvement {fmt(st['mean_brier_improve_pct'], 1)}% ± {fmt(st['std_brier_improve_pct'], 1)}%.</p>"
    )


_VERDICT_COLS = [
    ("period", "period"), ("n_windows", "windows"), ("coverage", "coverage"),
    ("model_brier", "model Brier"), ("bench_brier", "bench Brier"), ("brier_improve_pct", "Brier Δ%"),
    ("logloss_improve_pct", "log-loss Δ%"), ("diracc_delta_pp", "dir-acc Δpp"), ("low_sample", "⚠"),
]

_SCORE_COLS = [
    ("n_windows", "windows"), ("n_model", "model n"), ("n_market", "market n"),
    ("model_brier", "model Brier"), ("model_logloss", "model log-loss"), ("model_diracc", "model dir-acc"),
    ("formula_brier", "formula Brier"), ("market_brier", "market Brier"),
    ("market_logloss", "market log-loss"), ("low_sample", "⚠"),
]


def _build_html(metrics: dict) -> str:
    params = metrics["params"]
    scope_rows = metrics["scope_rows"]
    vt = metrics["verdict_table"]
    reliability = metrics["reliability"]

    rel_img = _img_tag(_reliability_png(reliability), "reliability curve")
    per_img = _img_tag(_period_png(vt["day"]["rows"]), "per-period edge")

    def verdict_table_html(benchmark: str) -> str:
        rows = [r for r in vt["day"]["rows"] if r["benchmark"] == benchmark]
        rows = sorted(rows, key=lambda r: (r["period"] == "ALL", r["period_key"]))
        return _table_html(rows, _VERDICT_COLS)

    overall_rows = [r for r in scope_rows if r["scope"].startswith("overall")]
    series_rows = [r for r in scope_rows if r["scope"] == "series"]
    tau_rows = [r for r in scope_rows if r["scope"] == "tau_bucket"]

    overall_cols = [("scope", "scope")] + _SCORE_COLS
    series_cols = [("series", "series")] + _SCORE_COLS
    tau_cols = [("tau_bucket", "time left")] + _SCORE_COLS
    rel_cols = [
        ("bin_lo", "bin lo"), ("bin_hi", "bin hi"), ("n_snapshots", "snapshots"),
        ("n_windows", "windows"), ("mean_pred", "mean pred"), ("empirical_rate", "empirical Up"),
        ("low_sample", "⚠"),
    ]

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{params['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 h3 {{ margin-top: 1.2rem; color: #333; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }}
 tr.low td {{ background: #fff5e6; color: #7a5200; }}
 .muted {{ color: #888; }} code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 img {{ max-width: 100%; }} .row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
</style></head><body>
<h1>{params['title']}</h1>
<p class="muted">Model under test: <code>{params['model_label']}</code>. Scored against the formula
model (recomputed at the same timestamps) and the Polymarket order-book mid, per calendar day.
Rows shaded <span style="background:#fff5e6;color:#7a5200;padding:0 4px">amber ⚠</span> are backed by
fewer than {params['min_windows']} distinct windows — too few to trust.</p>

<p class="verdict">{metrics['verdict']}</p>

<div class="row">{rel_img}{per_img}</div>

<h2>Verdict table — model vs formula</h2>
{verdict_table_html("formula")}
{_stability_line(vt, "formula")}

<h2>Verdict table — model vs market</h2>
{verdict_table_html("market")}
{_stability_line(vt, "market")}

<h2>Overall</h2>
{_table_html(overall_rows, overall_cols)}
<p class="muted"><code>overall_headtohead</code> = only where both a model and a market probability exist.</p>

<h2>By series</h2>
{_table_html(series_rows, series_cols)}

<h2>By time remaining</h2>
{_table_html(tau_rows, tau_cols)}

<h2>Reliability — model</h2>
{_table_html(reliability['model'], rel_cols)}

<h2>Reliability — market mid</h2>
{_table_html(reliability['market'], rel_cols)}

<p class="muted">Generated by the model-lab evaluation harness (<code>model_lab.eval_harness</code>).</p>
</body></html>
"""


# ===========================================================================
# orchestration + output
# ===========================================================================
def _empty_vt() -> dict:
    return {"rows": [], "stability": {b: _stability([], b) for b in BENCHMARKS}}


def run(
    df: pd.DataFrame,
    *,
    counts: dict,
    title: str,
    model_label: str,
    self_baseline: bool,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    health: str = "ready",
) -> dict:
    """Score a canonical eval ``df`` and return the full metrics dict (the same
    dict :func:`write_outputs` serializes). The single scoring entry point — every
    stage that evaluates a model reports through here."""
    params = {
        "min_windows": int(min_windows), "health": health, "n_bins": N_BINS,
        "self_baseline": bool(self_baseline), "model_label": model_label, "title": title,
    }
    if df.empty:
        return {
            "params": params, "counts": counts, "scope_rows": [],
            "reliability": {p: [] for p in PREDICTORS},
            "verdict_table": {"day": _empty_vt(), "week": _empty_vt()},
            "verdict": "No resolved windows with scored snapshots were found — nothing to evaluate.",
        }
    if "health" not in df.columns:
        df = df.assign(health="Ready")
    all_health = df
    scored = df if health == "all" else df[df["health"] == "Ready"]
    scored = scored if not scored.empty else df.iloc[0:0]

    scope_rows = _scope_rows(scored, all_health, min_windows)
    reliability = {p: _reliability(scored, p, min_windows) for p in PREDICTORS}
    verdict_table: dict = {}
    for by in ("day", "week"):
        rows = _verdict_rows(scored, min_windows, by)
        verdict_table[by] = {"rows": rows, "stability": {b: _stability(rows, b) for b in BENCHMARKS}}
    verdict = _verdict_text(scope_rows, verdict_table, reliability["model"], min_windows, params)

    return {
        "params": params, "counts": counts, "scope_rows": scope_rows,
        "reliability": reliability, "verdict_table": verdict_table, "verdict": verdict,
    }


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


_SCORES_CSV_COLS = [
    "scope", "series", "tau_bucket", "n_snapshots", "n_windows",
    "n_model", "n_formula", "n_market",
    "model_brier", "model_logloss", "model_diracc",
    "formula_brier", "formula_logloss", "formula_diracc",
    "market_brier", "market_logloss", "market_diracc", "low_sample",
]
_VERDICT_CSV_COLS = [
    "by", "period", "period_key", "benchmark", "n_snapshots", "n_windows", "coverage",
    "model_brier", "bench_brier", "brier_improve_pct",
    "model_logloss", "bench_logloss", "logloss_improve_pct",
    "model_diracc", "bench_diracc", "diracc_delta_pp", "low_sample",
]
_REL_CSV_COLS = [
    "predictor", "bin_lo", "bin_hi", "bin_mid", "n_snapshots", "n_windows",
    "mean_pred", "empirical_rate", "low_sample",
]


def write_outputs(out_dir, metrics: dict) -> None:
    """Write ``metrics.json, scores.csv, verdict_table.csv, reliability.csv,
    report.html`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    pd.DataFrame(metrics["scope_rows"], columns=_SCORES_CSV_COLS).to_csv(out_dir / "scores.csv", index=False)

    vrows: list[dict] = []
    for by in ("day", "week"):
        vrows.extend(metrics["verdict_table"][by]["rows"])
    pd.DataFrame(vrows, columns=_VERDICT_CSV_COLS).to_csv(out_dir / "verdict_table.csv", index=False)

    rel_rows: list[dict] = []
    for pred in PREDICTORS:
        for r in metrics["reliability"][pred]:
            rel_rows.append({"predictor": pred, **r})
    pd.DataFrame(rel_rows, columns=_REL_CSV_COLS).to_csv(out_dir / "reliability.csv", index=False)

    (out_dir / "report.html").write_text(_build_html(metrics), encoding="utf-8")
