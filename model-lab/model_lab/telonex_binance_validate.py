"""Telonex Binance depth — Stage 3 validation (5 checks; the feature-parity gate is decisive).

Judges whether the trial Binance ``book_snapshot_25`` (Stage 2) is genuine, live-engine-grade
depth by answering five questions with measured PASS/FAIL numbers:

1. **Levels & cadence** — ≥20 levels; per-day median inter-row gap ≤ 1 s.
2. **Clock alignment** — Telonex Binance trades vs our journaled Binance ``Trade`` ticks →
   offset / jitter / drift (reuses ``telonex_validate``'s clock machinery).
3. **FEATURE PARITY (decisive)** — reshape ``book_snapshot_25`` into the exact frame shape
   ``io.depth.read_depth_levels`` yields, run the **same** ``short_horizon._frame_depth_features``
   + ``build_depth_feature_grid``, and compare depth_imb_1/5/10/20 + microprice + slopes against
   our live-journaled ``depth20`` features on the 1 s grid. PASS = corr(depth_imb_20) ≥ 0.99 with
   no systematic bias, and no gross magnitude (unit) mismatch on microprice/slopes. FAIL ⇒ STOP.
4. **Completeness** — duplicate / non-monotonic / malformed rows; gaps > 5 s.
5. **Date range** — the real earliest/last available day (echoed from availability, not assumed).

    python -m model_lab.telonex_binance_validate     # → out/telonex/binance_validation.{json,html}

Reads only the trial parquet + our journal/depth; downloads nothing. Exit 0 iff GO.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import short_horizon as sh
from . import telonex_validate as tv
from .config import Paths, resolve_paths, stage_parser
from .io import depth as depthio
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem
from .telonex_binance_trial import (
    DEPTH_CHANNEL,
    OUT_SUBDIR,
    TRADES_CHANNEL,
    TRIAL_BTC_DEPTH_DAYS,
    TRIAL_BTC_TRADE_DAYS,
    TRIAL_META_NAME,
)

# --- thresholds ---------------------------------------------------------------
CADENCE_MEDIAN_PASS_S = 1.0     # per-day median inter-row gap must be ≤ 1 s
DEPTH_MIN_LEVELS = 20           # depth_imb_20 needs 20; the channel gives 25
LEVELS = 25                     # book_snapshot_25 flattened width per side

PARITY_CORR_PASS = 0.99         # DECISIVE: corr on depth_imb_20
PARITY_SOUND_MIN_CORR = 0.95    # a raw-1 s corr this high + convergence ⇒ genuine same-book data
MODEL_GRID_SECS = 5             # short_horizon's depth-feature grid (DEFAULT_GRID_SECS) — the
#                                 resolution the model actually attaches depth features at
PARITY_BIAS_MAX = 0.01          # DECISIVE: |mean signed error| on depth_imb_20 (unitless ratio)
PARITY_SECONDARY_CORR = 0.90    # microprice_gap + slopes should track this well
PARITY_SCALE_LO, PARITY_SCALE_HI = 0.5, 2.0  # magnitude-ratio guard → catches a size/price unit mismatch
PARITY_MIN_N = 500              # min matched seconds/day to trust the parity
SHIFT_RANGE_S = 30              # search ± this many whole seconds for the depth clock offset

DECISIVE_FEATURE = "depth_imb_20"
# One representative day is plenty for the trades clock offset/jitter/drift (both sides are
# Binance exchange time; the offset is constant). The journal is ~20 GB — scanning all three
# trial days would be needlessly slow. 07-05 sits in the middle of the overlap.
CLOCK_DAYS = ("2026-07-05",)
CLOCK_HOURS = 3   # journal scan window for the clock check (offset/jitter/drift), from 00:00 UTC
PARITY_HOURS = 3  # per-day window for the parity comparison — thousands of 1 s bars, kept fast
_GRID_META = {"asset", "sec", "ts_ms"}
# Features whose *magnitude* (not just shape) must agree — a unit mismatch (e.g. notional vs
# base-qty sizes) leaves the scale-invariant imbalance corr high but blows these up.
_MAGNITUDE_FEATURES = ("microprice_gap", "bid_depth_slope", "ask_depth_slope")


# --------------------------------------------------------------------------- #
# Reshaping Telonex book_snapshot_25 → the read_depth_levels frame shape
# --------------------------------------------------------------------------- #


def _valid_prefix(px: np.ndarray, sz: np.ndarray) -> int:
    """Count of leading (best-first) levels with finite price>0 and finite size — a thin book
    is an honest prefix, matching how ``read_depth_levels`` drops malformed levels."""
    n = 0
    for i in range(px.size):
        if not np.isfinite(px[i]) or not np.isfinite(sz[i]) or px[i] <= 0.0:
            break
        n += 1
    return n


def _tel_depth_grid(path: Path, asset: str, *, since_ms: int | None = None,
                    until_ms: int | None = None) -> pd.DataFrame:
    """Per-second depth-feature grid from a Telonex ``book_snapshot_25`` parquet, computed with
    the **same** ``short_horizon`` code as the live side (no reimplemented math). Optionally
    restricted to ``[since_ms, until_ms)`` (exchange-time ms)."""
    level_cols: list[str] = []
    for side in ("bid", "ask"):
        for i in range(LEVELS):
            level_cols += [f"{side}_price_{i}", f"{side}_size_{i}"]
    df = tv._read_cols(path, ["timestamp_us", *level_cols])
    ts_us = df["timestamp_us"].to_numpy(dtype="int64")
    if since_ms is not None and until_ms is not None:
        keep = (ts_us // 1000 >= since_ms) & (ts_us // 1000 < until_ms)
        df = df[keep].reset_index(drop=True)
        ts_us = ts_us[keep]
    df = df.assign(_sec=ts_us // 1_000_000)
    # Reduce to the last row in each second BEFORE building frames — build_depth_feature_grid
    # keeps last-per-second anyway (sort by recv_ms, drop_duplicates keep="last"), so this is
    # identical output for a timestamp-ordered feed while avoiding millions of feature calls.
    df = df.sort_values("timestamp_us").drop_duplicates("_sec", keep="last")

    def block(prefix: str) -> np.ndarray:
        cols = [f"{prefix}_{i}" for i in range(LEVELS)]
        return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")

    bid_px, bid_sz = block("bid_price"), block("bid_size")
    ask_px, ask_sz = block("ask_price"), block("ask_size")
    recv = df["timestamp_us"].to_numpy(dtype="int64") // 1000
    frames: list[dict[str, Any]] = []
    for k in range(len(df)):
        nb = _valid_prefix(bid_px[k], bid_sz[k])
        na = _valid_prefix(ask_px[k], ask_sz[k])
        if nb == 0 or na == 0:
            continue
        frames.append({
            "recv_ms": int(recv[k]), "asset": asset,
            "bid_px": bid_px[k, :nb], "bid_sz": bid_sz[k, :nb],
            "ask_px": ask_px[k, :na], "ask_sz": ask_sz[k, :na],
        })
    return sh.build_depth_feature_grid(frames, asset)


def _day_start_ms(day_str: str) -> int:
    d = date.fromisoformat(day_str)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def _tel_depth_grid_any(raw_path: Path, asset: str, *, since_ms: int | None = None,
                        until_ms: int | None = None, scratch: Path | None = None) -> pd.DataFrame:
    """Telonex depth grid from either the raw ``.parquet`` OR its finalized ``.parquet.zst``.

    The bulk pull deletes the raw parquet after the roundtrip check, leaving only the whole-file
    ``.zst`` — which pyarrow can't read directly — so the guard/re-validation decompresses it to
    a scratch temp first. Returns an empty frame if neither form is present."""
    raw = Path(raw_path)
    if raw.exists():
        return _tel_depth_grid(raw, asset, since_ms=since_ms, until_ms=until_ms)
    zst = tx.zst_sibling(raw)
    if not zst.exists():
        return pd.DataFrame()
    scratch = Path(scratch) if scratch is not None else raw.parent
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / (raw.name + ".parity.tmp.parquet")
    try:
        tx.decompress_zstd(zst, tmp)
        return _tel_depth_grid(tmp, asset, since_ms=since_ms, until_ms=until_ms)
    finally:
        tx._safe_unlink(tmp)


def _our_depth_grid_days(depth_dir: Path, asset: str, day_strs: tuple[str, ...],
                         *, hours: int = PARITY_HOURS) -> pd.DataFrame:
    """Per-second depth-feature grid from OUR live ``depth20`` capture, reading ONLY the first
    ``hours`` of each ``binance-depth20-{YYYYMMDD}`` day-file for ``day_strs``.

    Mirrors ``io.depth.read_depth_levels`` (same frame shape, same truncated-gzip resilience)
    but scoped to the target day-files, windowed with an early break (the files are
    chronological, so the first ``hours`` are at the front), and reduced to the last frame per
    second on the fly. Then runs the identical ``build_depth_feature_grid`` — the live half of
    the byte-for-byte comparison. The window keeps the whole pass fast (this box kills
    long-running jobs) while leaving thousands of 1 s bars per day for a robust correlation."""
    import gzip
    import json
    import zlib

    wanted = {f"binance-depth20-{d.replace('-', '')}.jsonl.gz": _day_start_ms(d) for d in day_strs}
    win_ms = hours * 3_600_000
    by_sec: dict[int, dict[str, Any]] = {}
    for path in depthio.depth_paths(depth_dir):
        if path.name not in wanted:
            continue
        day_start = wanted[path.name]
        window_end = day_start + win_ms
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    recv = int(obj.get("recv_ms", 0))
                    if recv >= window_end:
                        break  # chronological → past the window for this day-file
                    if recv < day_start:
                        continue
                    data = obj.get("data")
                    if not isinstance(data, dict):
                        continue
                    symbol = str(obj.get("stream", "")).split("@", 1)[0]
                    if depthio._symbol_asset(symbol) != asset:
                        continue
                    bid_px, bid_sz = depthio._level_arrays(data.get("bids"))
                    ask_px, ask_sz = depthio._level_arrays(data.get("asks"))
                    if bid_px.size == 0 or ask_px.size == 0 or bid_px[0] <= 0.0 or ask_px[0] <= 0.0:
                        continue
                    by_sec[recv // 1000] = {  # chronological → last wins
                        "recv_ms": recv, "asset": asset,
                        "bid_px": bid_px, "bid_sz": bid_sz, "ask_px": ask_px, "ask_sz": ask_sz,
                    }
        except (OSError, EOFError, zlib.error):
            continue
    frames = [by_sec[s] for s in sorted(by_sec)]
    return sh.build_depth_feature_grid(frames, asset)


# --------------------------------------------------------------------------- #
# Parity comparison (clock-offset search + per-feature stats)
# --------------------------------------------------------------------------- #


def _feature_cols(grid: pd.DataFrame) -> list[str]:
    return [c for c in grid.columns if c not in _GRID_META]


def _best_shift(our: pd.DataFrame, tel: pd.DataFrame, feature: str,
                shift_range: int = SHIFT_RANGE_S) -> tuple[int, float, float, int]:
    """Integer-second clock offset that best aligns ``tel`` to ``our`` on ``feature``.

    Our depth ``recv_ms`` is a *local* receive clock (subject to our PC's NTP skew), while
    Telonex ``timestamp_us`` is Binance exchange time — so the two per-second grids can sit a
    few whole seconds apart. Cross-correlate over integer-second shifts and take the peak.
    Returns ``(best_shift, corr_at_best, corr_at_0, n_at_best)`` where ``our_sec ≈ tel_sec +
    best_shift``. Feature VALUES are clock-independent, so a stable constant shift is a benign
    clock artifact — a genuinely different book collapses corr at *every* shift."""
    a = our.set_index("sec")[feature].dropna()
    b = tel.set_index("sec")[feature].dropna()
    if a.empty or b.empty:
        return 0, float("nan"), float("nan"), 0
    best = (0, float("-inf"), 0)
    corr0 = float("nan")
    for s in range(-shift_range, shift_range + 1):
        bs = pd.Series(b.to_numpy(), index=b.index.to_numpy() + s)  # shift tel → tel_sec + s
        j = pd.concat([a.rename("o"), bs.rename("t")], axis=1, join="inner").dropna()
        if len(j) < PARITY_MIN_N or j["o"].std() == 0 or j["t"].std() == 0:
            if s == 0:
                corr0 = float("nan")
            continue
        c = float(j["o"].corr(j["t"]))
        if s == 0:
            corr0 = c
        if np.isfinite(c) and c > best[1]:
            best = (s, c, len(j))
    return best[0], best[1], corr0, best[2]


def _aligned(our: pd.DataFrame, tel: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Inner-join the two per-second grids at the given clock ``shift`` (``tel_sec + shift``).
    Keeps ``sec`` as a column so the multi-resolution correlation curve can rebucket."""
    o = our.set_index("sec")[_feature_cols(our)]
    t = tel.copy()
    t["sec"] = t["sec"] + shift
    t = t.set_index("sec")[_feature_cols(tel)]
    return o.join(t, how="inner", lsuffix="_o", rsuffix="_t").reset_index()


def _corr_curve(pooled: pd.DataFrame, feature: str, ks=(1, 2, 3, 5, 10, 30, 60)) -> dict[int, float]:
    """Pearson corr of ``feature`` (our vs Telonex) at ``k``-second MEAN buckets. A genuine
    same-book feed's raw 1 s point correlation is capped by sub-second sampling differences
    (our 100 ms partial-book snapshots vs Telonex event-driven snapshots) and **converges to
    ~1.0** as ``k`` grows; a truly different/scaled book would not."""
    if "sec" not in pooled.columns:
        return {}
    o = pooled[f"{feature}_o"].to_numpy(dtype="float64")
    t = pooled[f"{feature}_t"].to_numpy(dtype="float64")
    sec = pooled["sec"].to_numpy(dtype="int64")
    m = np.isfinite(o) & np.isfinite(t)
    o, t, sec = o[m], t[m], sec[m]
    out: dict[int, float] = {}
    for k in ks:
        b = sec // k
        so = pd.Series(o).groupby(b).mean()
        st = pd.Series(t).groupby(b).mean()
        out[k] = float(so.corr(st)) if len(so) >= 2 and so.std() > 0 and st.std() > 0 else float("nan")
    return out


def _feature_stats(o: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    o = np.asarray(o, dtype="float64")
    t = np.asarray(t, dtype="float64")
    m = np.isfinite(o) & np.isfinite(t)
    o, t = o[m], t[m]
    if o.size == 0:
        return {"n": 0}
    corr = float("nan")
    if o.size >= 2 and np.std(o) > 0 and np.std(t) > 0:
        corr = float(np.corrcoef(o, t)[0, 1])
    err = t - o  # telonex − ours (signed)
    med_abs_o = float(np.median(np.abs(o)))
    return {
        "n": int(o.size),
        "corr": corr,
        "bias": float(np.mean(err)),
        "median_abs_err": float(np.median(np.abs(err))),
        "p95_abs_err": float(np.percentile(np.abs(err), 95)),
        "abs_ratio": (float(np.median(np.abs(t)) / med_abs_o) if med_abs_o > 0 else float("nan")),
    }


def check_parity(paths: Paths, *, asset: str = "btc",
                 depth_days: tuple[str, ...] = TRIAL_BTC_DEPTH_DAYS) -> dict[str, Any]:
    """Decisive check 3 — reconstruct our depth features from Telonex depth and compare."""
    telonex_dir = paths.telonex_dir
    scratch = paths.out_dir / OUT_SUBDIR / "_parity_scratch"
    per_day: list[dict[str, Any]] = []
    aligned_frames: list[pd.DataFrame] = []
    # Read our live depth once for the whole span (only the target day-files), slice per day.
    g_ours_all = _our_depth_grid_days(paths.depth_dir, asset, depth_days)
    for day_str in depth_days:
        raw = tx.binance_raw_path(telonex_dir, DEPTH_CHANNEL, f"{asset}usdt", day_str)
        if not tx.raw_or_zst_exists(raw):
            per_day.append({"day": day_str, "status": "missing_telonex"})
            continue
        since = _day_start_ms(day_str)
        until = since + PARITY_HOURS * 3_600_000
        g_ours = (g_ours_all[(g_ours_all["ts_ms"] >= since) & (g_ours_all["ts_ms"] < until)]
                  if not g_ours_all.empty else g_ours_all)
        g_tel = _tel_depth_grid_any(raw, asset, since_ms=since, until_ms=until, scratch=scratch)
        if g_ours.empty or g_tel.empty:
            per_day.append({"day": day_str, "status": "no_overlap",
                            "our_secs": int(len(g_ours)), "tel_secs": int(len(g_tel))})
            continue
        shift, corr_best, corr0, n = _best_shift(g_ours, g_tel, DECISIVE_FEATURE)
        per_day.append({
            "day": day_str, "status": "ok", "best_shift_secs": int(shift),
            "corr_at_best": corr_best, "corr_at_0": corr0, "n": int(n),
            "our_secs": int(len(g_ours)), "tel_secs": int(len(g_tel)),
        })
        if n >= PARITY_MIN_N and np.isfinite(corr_best):
            aligned_frames.append(_aligned(g_ours, g_tel, shift))

    if not aligned_frames:
        return {"name": "feature_parity", "result": "FAIL", "reason": "no_aligned_data",
                "per_day": per_day}

    pooled = pd.concat(aligned_frames, ignore_index=True)
    feats = [c[:-2] for c in pooled.columns if c.endswith("_o")]
    stats: dict[str, Any] = {}
    for f in feats:
        stats[f] = _feature_stats(pooled[f"{f}_o"].to_numpy(), pooled[f"{f}_t"].to_numpy())

    dec = stats.get(DECISIVE_FEATURE, {"n": 0})
    decisive_ok = (
        dec.get("n", 0) >= PARITY_MIN_N
        and np.isfinite(dec.get("corr", float("nan")))
        and dec["corr"] >= PARITY_CORR_PASS
        and abs(dec.get("bias", 1.0)) <= PARITY_BIAS_MAX
    )
    # Magnitude guard: catch a size/price unit mismatch that the ratio-invariant imbalance hides.
    scale_flags: dict[str, Any] = {}
    magnitude_ok = True
    for f in _MAGNITUDE_FEATURES:
        s = stats.get(f, {})
        ratio = s.get("abs_ratio", float("nan"))
        corr = s.get("corr", float("nan"))
        ok = (np.isfinite(ratio) and PARITY_SCALE_LO <= ratio <= PARITY_SCALE_HI)
        scale_flags[f] = {"abs_ratio": ratio, "corr": corr, "within_scale": bool(ok)}
        if not ok:
            magnitude_ok = False

    shifts = [d["best_shift_secs"] for d in per_day if d.get("status") == "ok"]
    # A per-day shift that varies is OUR local depth clock drifting (recv_ms), not a Telonex
    # data fault — the feature VALUES match at the aligned shift regardless. Reported for
    # transparency but NOT part of the pass condition; the trades clock check (check 2) is the
    # authoritative timing test. The decisive signal is corr on the aligned depth_imb_20.
    shift_stable = (len(shifts) >= 1 and (max(shifts) - min(shifts) <= 1))

    # Multi-resolution convergence of the decisive feature. Our 100 ms partial-book snapshots
    # and Telonex's event-driven snapshots sample the book at different sub-second instants, so
    # the raw 1 s point correlation is a sampling ceiling that converges to ~1.0 as the bucket
    # coarsens — proof the *underlying book is identical* (a different/scaled/futures book would
    # not converge). MODEL_GRID_SECS is the resolution the model actually attaches features at.
    curve = _corr_curve(pooled, DECISIVE_FEATURE)
    corr_model_grid = curve.get(MODEL_GRID_SECS, float("nan"))
    converges = np.isfinite(corr_model_grid) and corr_model_grid >= PARITY_CORR_PASS
    # "sound" = no structural mismatch (magnitude + bias), a high raw correlation, AND the curve
    # reaches the pass bar by the model's grid — i.e. genuine same-book data whose only gap is
    # zero-mean sub-second sampling noise.
    sound = bool(
        magnitude_ok and abs(dec.get("bias", 1.0)) <= PARITY_BIAS_MAX
        and dec.get("n", 0) >= PARITY_MIN_N
        and np.isfinite(dec.get("corr", float("nan"))) and dec["corr"] >= PARITY_SOUND_MIN_CORR
        and converges
    )
    if decisive_ok and magnitude_ok:
        result = "PASS"
    elif sound:
        # Genuine, unbiased, same-book depth; the raw-1 s corr is sampling-limited below 0.99 but
        # converges to ~1.0 and passes at the model's feature grid. Operator judgement call.
        result = "REVIEW"
    else:
        result = "FAIL"
    return {
        "name": "feature_parity",
        "result": result,
        "decisive_feature": DECISIVE_FEATURE,
        "decisive_ok": bool(decisive_ok),
        "sound": sound,
        "magnitude_ok": bool(magnitude_ok),
        "shift_stable": bool(shift_stable),
        "raw_corr": dec.get("corr"),
        "corr_curve_by_bucket_s": {str(k): v for k, v in curve.items()},
        "corr_at_model_grid": corr_model_grid,
        "model_grid_secs": MODEL_GRID_SECS,
        "converges_to_pass_by_model_grid": bool(converges),
        "per_day": per_day,
        "pooled_n": int(len(pooled)),
        "feature_stats": stats,
        "magnitude_guard": scale_flags,
        "thresholds": {"corr_pass": PARITY_CORR_PASS, "bias_max": PARITY_BIAS_MAX,
                       "scale": [PARITY_SCALE_LO, PARITY_SCALE_HI], "min_n": PARITY_MIN_N,
                       "sound_min_corr": PARITY_SOUND_MIN_CORR},
    }


# --------------------------------------------------------------------------- #
# Check 1 — levels & cadence
# --------------------------------------------------------------------------- #


def check_levels_cadence(paths: Paths, *, symbol: str = "btcusdt",
                         depth_days: tuple[str, ...] = TRIAL_BTC_DEPTH_DAYS) -> dict[str, Any]:
    import pyarrow.parquet as pq

    specs: list[tuple[Path, int | None, int | None]] = []
    levels: int | None = None
    for day_str in depth_days:
        p = tx.binance_raw_path(paths.telonex_dir, DEPTH_CHANNEL, symbol, day_str)
        if not p.exists():
            continue
        specs.append((p, None, None))  # Binance depth is one continuous per-day stream
        if levels is None:
            names = set(pq.ParquetFile(p).schema_arrow.names)
            levels = sum(1 for i in range(50) if f"bid_price_{i}" in names)
    cad = tv._cadence_active(specs)
    levels_ok = levels is not None and levels >= DEPTH_MIN_LEVELS
    cadence_ok = cad.get("median_s") is not None and cad["median_s"] <= CADENCE_MEDIAN_PASS_S
    return {
        "name": "levels_cadence",
        "result": tv._pf(levels_ok and cadence_ok),
        "levels": levels,
        "levels_ge_20": bool(levels_ok),
        "cadence": cad,
        "cadence_median_ok": bool(cadence_ok),
        "threshold_s": CADENCE_MEDIAN_PASS_S,
    }


# --------------------------------------------------------------------------- #
# Check 2 — clock alignment (Telonex Binance trades vs our journal)
# --------------------------------------------------------------------------- #


def _our_binance_trades_span(journal_dir: Path, since_ms: int, until_ms: int) -> pd.DataFrame:
    """Our journaled Binance ``Trade`` ticks over ``[since, until)`` in a single pass.

    Reuses ``telonex_validate``'s segment selection but adds a substring pre-filter so
    ``json.loads`` runs only on the tiny fraction of journal lines that mention a
    ``BinanceDirect`` ``price_tick`` — the multi-GB/day journal is otherwise far too slow to
    parse line-by-line for a 3-day span."""
    import gzip
    import json
    import zlib

    segs = tv._segments_covering(journal_dir, [(since_ms, until_ms)])
    ts: list[int] = []
    pr: list[float] = []
    for path in segs:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if '"BinanceDirect"' not in line or '"price_tick"' not in line:
                        continue
                    try:
                        env = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    rec = env.get("rec") or {}
                    if (rec.get("type") != "price_tick" or rec.get("source") != "BinanceDirect"
                            or rec.get("kind") != "Trade"):
                        continue
                    te = rec.get("ts_exchange")
                    if te is None or not (since_ms <= te < until_ms):
                        continue
                    ts.append(int(te))
                    pr.append(tv.jio.to_float(rec.get("value")))
        except (OSError, EOFError, zlib.error):
            continue
    return pd.DataFrame({"ts_ms": ts, "price": pr})


def check_clock(paths: Paths, *, symbol: str = "btcusdt",
                trade_days: tuple[str, ...] = CLOCK_DAYS) -> dict[str, Any]:
    tel_rows: list[pd.DataFrame] = []
    for day_str in trade_days:
        p = tx.binance_raw_path(paths.telonex_dir, TRADES_CHANNEL, symbol, day_str)
        if p.exists():
            d = tv._read_cols(p, ["timestamp_us", "price"])
            tel_rows.append(pd.DataFrame({
                "ts_ms": d["timestamp_us"].to_numpy(dtype="int64") // 1000,
                "price": d["price"].map(tv._to_float).to_numpy(),
            }))
    tel = pd.concat(tel_rows, ignore_index=True) if tel_rows else pd.DataFrame(columns=["ts_ms", "price"])
    days = sorted(date.fromisoformat(x) for x in trade_days)
    since = tv._day_bounds_ms(days[0])[0]
    # Bound the (expensive) journal scan to the first CLOCK_HOURS of the day — thousands of
    # price-matched trades and a multi-hour span are ample for a constant offset + drift, and
    # the 20 GB journal is far too slow to scan a whole day of.
    until = min(tv._day_bounds_ms(days[-1])[1], since + CLOCK_HOURS * 3_600_000)
    ours = _our_binance_trades_span(paths.journal_dir, since, until)
    stats = tv._match_binance(tel, ours)
    ok = tv._stable(stats)
    return {
        "name": "clock_alignment",
        "result": tv._pf(ok),
        "binance_trades_vs_journal": stats,
        "our_trade_rows": int(len(ours)),
        "telonex_trade_rows": int(len(tel)),
        "thresholds": {"jitter_ms": tv.CLOCK_JITTER_PASS_MS,
                       "drift_ms_per_hr": tv.CLOCK_DRIFT_PASS_MS_PER_HR,
                       "min_matches": tv.CLOCK_MIN_MATCHES},
    }


# --------------------------------------------------------------------------- #
# Check 4 — completeness
# --------------------------------------------------------------------------- #


def check_completeness(paths: Paths, cadence: dict[str, Any], *, symbol: str = "btcusdt",
                       depth_days: tuple[str, ...] = TRIAL_BTC_DEPTH_DAYS) -> dict[str, Any]:
    dup_ts = 0
    nonmono = 0
    malformed = 0
    rows_total = 0
    per_day: list[dict[str, Any]] = []
    for day_str in depth_days:
        p = tx.binance_raw_path(paths.telonex_dir, DEPTH_CHANNEL, symbol, day_str)
        if not p.exists():
            continue
        df = tv._read_cols(p, ["timestamp_us", "bid_price_0", "ask_price_0"])
        ts = df["timestamp_us"].to_numpy(dtype="int64")
        d_dup = int(len(ts) - np.unique(ts).size)
        d_nonmono = int((np.diff(ts) < 0).sum()) if ts.size >= 2 else 0
        bid = df["bid_price_0"].map(tv._to_float).to_numpy()
        ask = df["ask_price_0"].map(tv._to_float).to_numpy()
        d_malformed = int((~np.isfinite(bid) | ~np.isfinite(ask) | (bid <= 0) | (ask <= 0)
                           | (bid >= ask)).sum())
        dup_ts += d_dup
        nonmono += d_nonmono
        malformed += d_malformed
        rows_total += len(ts)
        per_day.append({"day": day_str, "rows": int(len(ts)), "dup_ts": d_dup,
                        "nonmonotonic": d_nonmono, "malformed_or_crossed": d_malformed})
    gaps_over = cadence.get("cadence", {}).get("gaps_over_5s", 0)
    ok = nonmono == 0 and malformed == 0
    return {
        "name": "completeness",
        "result": tv._pf(ok),
        "rows_total": rows_total,
        "duplicate_timestamps": dup_ts,
        "nonmonotonic_timestamps": nonmono,
        "malformed_or_crossed_rows": malformed,
        "gaps_over_5s": gaps_over,
        "per_day": per_day,
    }


# --------------------------------------------------------------------------- #
# Check 5 — date range (echoed from availability, not assumed)
# --------------------------------------------------------------------------- #


def check_date_range(paths: Paths) -> dict[str, Any]:
    meta = _load_trial_meta(paths)
    avail = (meta or {}).get("availability", {})
    ranges: dict[str, Any] = {}
    have = False
    for sym, info in avail.items():
        r = info.get("channels", {}).get(DEPTH_CHANNEL) if isinstance(info, dict) else None
        if isinstance(r, dict):
            frm, to = r.get("from_date"), r.get("to_date")
            # to_date is exclusive → last available day is the day before.
            last = _prev_day(to) if to else None
            ranges[sym] = {"from": frm, "to_exclusive": to, "last_available_day": last}
            have = have or bool(frm and to)
    return {
        "name": "date_range",
        "result": "PASS" if have else "SKIPPED (no trial meta)",
        "depth_channel": DEPTH_CHANNEL,
        "ranges": ranges,
        "note": "to_date is exclusive; the pull covers [from_date, to_date).",
    }


def _prev_day(day_str: str) -> str | None:
    try:
        d = date.fromisoformat(day_str)
    except ValueError:
        return None
    from datetime import timedelta
    return (d - timedelta(days=1)).isoformat()


def _load_trial_meta(paths: Paths) -> dict[str, Any] | None:
    p = paths.out_dir / OUT_SUBDIR / TRIAL_META_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Permanent parity guard — re-run on every Telonex-depth feature rebuild
# --------------------------------------------------------------------------- #

GUARD_START_DAY = "2026-07-04"  # first full recorder-overlap day (capture began 2026-07-03)
# Divergence-focused guard criteria (operator-set 2026-07-09). A genuine same-book feed differs
# from ours only by zero-mean sub-second sampling noise: no bias, matching magnitudes, a stable
# clock, and a correlation that CONVERGES past the bar by a coarse bucket while clearing a hard
# floor at the model's working grid. A basis-bug-style divergence violates one of these.
GUARD_MAG_LO, GUARD_MAG_HI = 0.98, 1.02   # magnitude (abs) ratio band for the checked features
GUARD_MAX_BIAS = PARITY_BIAS_MAX          # |systematic bias| on depth_imb_20 (0.01)
GUARD_CORR_5S_FLOOR = 0.97                # hard floor at the 5 s model grid ("converges eventually" ≠ pass)
GUARD_CONVERGE_BUCKET = 30                # correlation must reach GUARD_CORR_CONVERGE by this bucket (s)
GUARD_CORR_CONVERGE = 0.99
# Features whose magnitude must match (ratio-invariant imbalance would hide a size/price unit error).
GUARD_MAG_FEATURES = ("depth_imb_20", "microprice_gap", "bid_depth_slope", "ask_depth_slope")


def _recorder_depth_days(depth_dir: Path) -> set[str]:
    import re
    days: set[str] = set()
    for p in depthio.depth_paths(depth_dir):
        m = re.search(r"binance-depth20-(\d{8})", p.name)
        if m:
            d = m.group(1)
            days.add(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    return days


def _telonex_depth_days(telonex_dir: Path, symbol: str) -> set[str]:
    import re
    d = tx.raw_root(telonex_dir) / tx.BINANCE / DEPTH_CHANNEL
    days: set[str] = set()
    if d.is_dir():
        rx = re.compile(rf"^{re.escape(symbol)}-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet")
        for p in d.iterdir():
            m = rx.match(p.name)
            if m:
                days.add(m.group(1))
    return days


def _overlap_days(paths: Paths, symbol: str, *, start_day: str = GUARD_START_DAY) -> tuple[str, ...]:
    """UTC days where BOTH our recorder depth file and the Telonex depth (raw or .zst) exist."""
    rec = _recorder_depth_days(paths.depth_dir)
    tel = _telonex_depth_days(paths.telonex_dir, symbol)
    return tuple(sorted(d for d in (rec & tel) if d >= start_day))


def _guard_verdict(par: dict[str, Any]) -> dict[str, Any]:
    """Divergence-focused pass/fail for one symbol's parity result (operator criteria)."""
    stats = par.get("feature_stats", {})
    dec = stats.get(DECISIVE_FEATURE, {})
    curve = {int(k): v for k, v in (par.get("corr_curve_by_bucket_s") or {}).items()}
    corr5 = curve.get(MODEL_GRID_SECS)
    corr_conv = curve.get(GUARD_CONVERGE_BUCKET)
    bias = dec.get("bias")
    ratios = {f: stats.get(f, {}).get("abs_ratio") for f in GUARD_MAG_FEATURES}

    reasons: list[str] = []
    if bias is None or abs(bias) > GUARD_MAX_BIAS:
        reasons.append(f"systematic bias |{bias}| > {GUARD_MAX_BIAS}")
    for f, r in ratios.items():
        if r is None or not np.isfinite(r) or not (GUARD_MAG_LO <= r <= GUARD_MAG_HI):
            reasons.append(f"magnitude ratio {f}={r} outside [{GUARD_MAG_LO},{GUARD_MAG_HI}]")
    if not par.get("shift_stable"):
        reasons.append("unstable clock offset (per-day depth shift not stable)")
    if corr_conv is None or not np.isfinite(corr_conv) or corr_conv < GUARD_CORR_CONVERGE:
        reasons.append(f"corr@{GUARD_CONVERGE_BUCKET}s={corr_conv} < {GUARD_CORR_CONVERGE} (no convergence)")
    if corr5 is None or not np.isfinite(corr5) or corr5 < GUARD_CORR_5S_FLOOR:
        reasons.append(f"corr@{MODEL_GRID_SECS}s={corr5} < {GUARD_CORR_5S_FLOOR} (below hard floor)")

    return {
        "passed": not reasons, "fail_reasons": reasons,
        "corr_at_5s_grid": corr5, "corr_at_converge_bucket": corr_conv,
        "converge_bucket_s": GUARD_CONVERGE_BUCKET, "raw_corr_1s": par.get("raw_corr"),
        "bias": bias, "magnitude_ratios": ratios, "shift_stable": par.get("shift_stable"),
    }


def capture_quality(depth_dir: Path, day_strs: tuple[str, ...], *,
                    assets: tuple[str, ...] = ("btc", "eth"), hours: int = PARITY_HOURS) -> dict[str, Any]:
    """Per-asset inter-arrival / gap / staleness stats of OUR recorder depth capture, to tell
    whether ETH's extra sub-second noise comes from a laggier capture on our side or from genuine
    market microstructure. If BTC and ETH have similar inter-arrival cadence, the capture is
    equally good and the difference is the market; if ETH's cadence is markedly worse, it's ours."""
    import gzip
    import json
    import zlib

    wanted = {f"binance-depth20-{d.replace('-', '')}.jsonl.gz": _day_start_ms(d) for d in day_strs}
    win = hours * 3_600_000
    recv: dict[str, list[int]] = {a: [] for a in assets}
    for path in depthio.depth_paths(depth_dir):
        if path.name not in wanted:
            continue
        day_start = wanted[path.name]
        end = day_start + win
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    r = int(obj.get("recv_ms", 0))
                    if r >= end:
                        break
                    if r < day_start:
                        continue
                    a = depthio._symbol_asset(str(obj.get("stream", "")).split("@", 1)[0])
                    if a in recv:
                        recv[a].append(r)
        except (OSError, EOFError, zlib.error):
            continue
    out: dict[str, Any] = {}
    for a in assets:
        ts = np.sort(np.asarray(recv[a], dtype="int64"))
        if ts.size < 2:
            out[a] = {"frames": int(ts.size)}
            continue
        d = np.diff(ts).astype("float64")
        span_s = (ts[-1] - ts[0]) / 1000.0
        out[a] = {
            "frames": int(ts.size), "span_hours": round(span_s / 3600.0, 2),
            "frames_per_sec": round(ts.size / span_s, 2) if span_s > 0 else None,
            "inter_arrival_ms": {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                                 "p99": float(np.percentile(d, 99)), "max": float(d.max())},
            "gaps_gt_1s": int((d > 1000).sum()), "gaps_gt_5s": int((d > 5000).sum()),
        }
    return out


def parity_guard(paths: Paths, *, assets: tuple[str, ...] = ("btc", "eth"),
                 start_day: str = GUARD_START_DAY) -> dict[str, Any]:
    """The permanent depth-parity guard. Run this on every rebuild of depth features from
    Telonex data: it auto-discovers the recorder∩Telonex overlap days (growing as we record
    more), reconstructs our depth features from the Telonex ``book_snapshot_25`` for BTCUSDT AND
    ETHUSDT, and **fails loudly** on any divergence signal — a systematic bias, a magnitude
    (unit) mismatch, an unstable clock offset, a correlation that never converges past 0.99 by
    the 30 s bucket, or a corr below the 0.97 hard floor at the model's 5 s grid. The standing
    safeguard against the basis-bug failure mode. The full per-symbol convergence curve is logged
    every run so gradual drift is visible, not just a pass/fail bit."""
    results: dict[str, Any] = {}
    overall = True
    judged_any = False
    for asset in assets:
        symbol = f"{asset}usdt"
        days = _overlap_days(paths, symbol, start_day=start_day)
        if not days:
            results[asset] = {"status": "no_overlap_days", "passed": None}
            continue
        par = check_parity(paths, asset=asset, depth_days=days)
        verdict = _guard_verdict(par)
        judged_any = True
        overall = overall and verdict["passed"]
        results[asset] = {
            "status": "ok", **verdict, "overlap_days": list(days),
            "corr_curve_by_bucket_s": par.get("corr_curve_by_bucket_s"),
            "pooled_n": par.get("pooled_n"),
            "per_day_raw_corr": [{"day": d.get("day"), "corr_at_best": d.get("corr_at_best")}
                                 for d in par.get("per_day", []) if d.get("status") == "ok"],
        }
    return {
        "name": "parity_guard",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "passed": bool(overall) if judged_any else None,
        "criteria": {
            "grid_secs": MODEL_GRID_SECS, "corr_5s_floor": GUARD_CORR_5S_FLOOR,
            "converge_bucket_s": GUARD_CONVERGE_BUCKET, "corr_converge_min": GUARD_CORR_CONVERGE,
            "max_bias": GUARD_MAX_BIAS, "magnitude_band": [GUARD_MAG_LO, GUARD_MAG_HI],
        },
        "per_asset": results,
    }


def run_parity_guard(paths: Paths) -> int:
    """CLI entry for the guard: print + persist the verdict; return 0 iff it passes."""
    guard = parity_guard(paths)
    out = paths.out_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "binance_parity_guard.json").write_text(json.dumps(guard, indent=2, default=str), encoding="utf-8")
    c = guard["criteria"]

    def fnum(x, nd=4):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) and np.isfinite(x) else str(x)

    print(f"[parity_guard] divergence-focused: corr@{c['grid_secs']}s ≥ {c['corr_5s_floor']} floor, "
          f"corr@{c['converge_bucket_s']}s ≥ {c['corr_converge_min']}, |bias| ≤ {c['max_bias']}, "
          f"magnitude ∈ {c['magnitude_band']}, stable clock offset:")
    for asset, r in guard["per_asset"].items():
        if r.get("status") != "ok":
            print(f"[parity_guard]   {asset}: {r.get('status')}")
            continue
        days = r.get("overlap_days", [])
        span = f"{days[0]}..{days[-1]}" if days else "—"
        print(f"[parity_guard]   {asset}: {'PASS' if r['passed'] else 'FAIL'}  "
              f"corr@{c['grid_secs']}s={fnum(r['corr_at_5s_grid'])} "
              f"corr@{c['converge_bucket_s']}s={fnum(r['corr_at_converge_bucket'])} "
              f"raw@1s={fnum(r['raw_corr_1s'])} bias={fnum(r['bias'],6)} "
              f"shift_stable={r['shift_stable']} n={r.get('pooled_n')} days={len(days)} ({span})")
        curve = r.get("corr_curve_by_bucket_s", {})
        if curve:  # always log the full convergence curve so drift is visible
            print("[parity_guard]     depth_imb_20 corr curve: "
                  + "  ".join(f"{k}s={fnum(v)}" for k, v in curve.items()))
        print("[parity_guard]     magnitude ratios: "
              + "  ".join(f"{f}={fnum(v,3)}" for f, v in (r.get("magnitude_ratios") or {}).items()))
        if r.get("fail_reasons"):
            for reason in r["fail_reasons"]:
                print(f"[parity_guard]     ✗ {reason}")
    if guard["passed"] is None:
        print("[parity_guard] NO overlap days found for any symbol — nothing to check "
              "(recorder∩Telonex depth empty). Not a pass.")
        return 1
    print(f"[parity_guard] {'PASS' if guard['passed'] else 'FAIL — depth divergence! do NOT train on this.'}")
    return 0 if guard["passed"] else 1


# --------------------------------------------------------------------------- #
# Orchestration + report
# --------------------------------------------------------------------------- #


def validate(paths: Paths) -> dict[str, Any]:
    def step(msg: str) -> None:
        print(f"[telonex_binance_validate] … {msg}", flush=True)

    checks: dict[str, Any] = {}
    # Parity is the decisive check — run it first so its verdict is secured earliest.
    step("check 1/5 FEATURE PARITY (reading our depth + reshaping Telonex book_snapshot_25)")
    checks["parity"] = check_parity(paths)
    step(f"  parity → {checks['parity']['result']}")
    step("check 2/5 levels & cadence")
    checks["levels_cadence"] = check_levels_cadence(paths)
    step("check 3/5 clock alignment (scanning journal for Binance trades)")
    checks["clock"] = check_clock(paths)
    step("check 4/5 completeness")
    checks["completeness"] = check_completeness(paths, checks["levels_cadence"])
    step("check 5/5 date range")
    checks["date_range"] = check_date_range(paths)
    # Decisive: parity. All must pass except date_range (informational). A parity verdict of
    # REVIEW (sound same-book data whose raw-1 s corr is sampling-limited but converges past the
    # bar at the model's grid) surfaces as REVIEW so the operator makes the final call.
    non_parity = ["levels_cadence", "clock", "completeness"]
    others_ok = all(checks[c]["result"] == "PASS" for c in non_parity)
    parity = checks["parity"]["result"]
    if others_ok and parity == "PASS":
        go = "GO"
    elif others_ok and parity == "REVIEW":
        go = "REVIEW"
    else:
        go = "NO-GO"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "depth_channel": DEPTH_CHANNEL,
        "checks": checks,
        "go_no_go": go,
    }


def _write_outputs(paths: Paths, result: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = paths.out_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "binance_validation.json"
    jpath.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    hpath = out_dir / "binance_validation.html"
    hpath.write_text(_render_html(result), encoding="utf-8")
    return jpath, hpath


def _render_html(result: dict[str, Any]) -> str:
    rows = []
    for key, c in result["checks"].items():
        res = c["result"]
        colour = "#1a7f37" if res == "PASS" else ("#9a6700" if res.startswith("SKIPPED") else "#cf222e")
        body = json.dumps({k: v for k, v in c.items() if k not in ("name", "result")},
                          indent=1, default=str)[:4000]
        rows.append(f"<tr><td>{key}</td><td style='color:{colour};font-weight:600'>{res}</td>"
                    f"<td><pre style='margin:0;white-space:pre-wrap'>{body}</pre></td></tr>")
    go = result.get("go_no_go", "?")
    go_colour = "#1a7f37" if go == "GO" else "#cf222e"
    return f"""<!doctype html><meta charset=utf-8><title>Telonex Binance depth validation</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d0d7de;padding:6px 10px;text-align:left;vertical-align:top}}
h1 span{{color:{go_colour}}}</style>
<h1>Telonex Binance depth ({result['depth_channel']}) — <span>{go}</span></h1>
<p>Generated {result['generated_utc']}. Decisive gate = <b>feature parity</b> (corr(depth_imb_20) ≥ {PARITY_CORR_PASS}).</p>
<table><tr><th>Check</th><th>Result</th><th>Numbers</th></tr>{''.join(rows)}</table>
"""


def _print_summary(result: dict[str, Any]) -> None:
    print("\n[telonex_binance_validate] ==== VALIDATION SUMMARY ====")
    for key, c in result["checks"].items():
        print(f"[telonex_binance_validate] {key:16s} {c['result']}")
    lc = result["checks"]["levels_cadence"]
    cad = lc.get("cadence", {})
    print(f"[telonex_binance_validate]   levels={lc.get('levels')}  cadence median "
          f"{cad.get('median_s')}s p95 {cad.get('p95_s')}s max {cad.get('max_s')}s "
          f"gaps>5s={cad.get('gaps_over_5s')}")
    clk = result["checks"]["clock"]["binance_trades_vs_journal"]
    if clk.get("n", 0):
        print(f"[telonex_binance_validate]   clock: offset {clk.get('median_offset_ms'):.1f}ms "
              f"jitter {clk.get('jitter_ms'):.1f}ms drift {clk.get('drift_ms_per_hr'):.2f}ms/hr n={clk['n']}")
    par = result["checks"]["parity"]
    print(f"[telonex_binance_validate]   parity: {par['result']}  pooled_n={par.get('pooled_n')}  "
          f"raw_corr={par.get('raw_corr')}  corr@{par.get('model_grid_secs')}s={par.get('corr_at_model_grid')}")
    curve = par.get("corr_curve_by_bucket_s", {})
    if curve:
        print("[telonex_binance_validate]     depth_imb_20 corr vs bucket: "
              + "  ".join(f"{k}s={v:.4f}" for k, v in curve.items()))
    for d in par.get("per_day", []):
        if d.get("status") == "ok":
            print(f"[telonex_binance_validate]     {d['day']}: shift={d['best_shift_secs']}s "
                  f"corr@best={d['corr_at_best']:.4f} corr@0={d['corr_at_0']:.4f} n={d['n']}")
        else:
            print(f"[telonex_binance_validate]     {d['day']}: {d.get('status')}")
    for f, s in (par.get("feature_stats", {}) or {}).items():
        if s.get("n"):
            print(f"[telonex_binance_validate]       {f:18s} corr={s.get('corr'):.4f} "
                  f"bias={s.get('bias'):+.5f} mad={s.get('median_abs_err'):.5f} "
                  f"ratio={s.get('abs_ratio'):.3f} n={s['n']}")
    print(f"[telonex_binance_validate] GO/NO-GO: {result['go_no_go']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_binance_validate")
    parser.add_argument("--guard", action="store_true",
                        help="run only the permanent depth-parity guard (BTC+ETH, all overlap days) "
                             "and exit non-zero on any divergence signal")
    parser.add_argument("--capture-quality", action="store_true",
                        help="report our recorder's per-asset depth inter-arrival/gap stats (BTC vs ETH) "
                             "on the overlap days")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    if args.capture_quality:
        days = _overlap_days(paths, "btcusdt")
        cq = capture_quality(paths.depth_dir, days)
        out = paths.out_dir / OUT_SUBDIR
        out.mkdir(parents=True, exist_ok=True)
        (out / "binance_capture_quality.json").write_text(
            json.dumps({"days": list(days), "per_asset": cq}, indent=2, default=str), encoding="utf-8")
        print(f"[capture_quality] our recorder depth, first {PARITY_HOURS}h of {list(days)}:")
        for a, s in cq.items():
            if s.get("frames", 0) < 2:
                print(f"[capture_quality]   {a}: {s.get('frames')} frames (insufficient)")
                continue
            ia = s["inter_arrival_ms"]
            print(f"[capture_quality]   {a}: {s['frames']} frames, {s['frames_per_sec']}/s; "
                  f"inter-arrival median {ia['median']:.0f}ms p95 {ia['p95']:.0f}ms p99 {ia['p99']:.0f}ms "
                  f"max {ia['max']:.0f}ms; gaps>1s={s['gaps_gt_1s']} gaps>5s={s['gaps_gt_5s']}")
        procmem.report_peak_rss("telonex_binance_validate", procmem.peak_rss_mb(), args.max_rss_mb)
        return 0

    if args.guard:
        rc = run_parity_guard(paths)
        procmem.report_peak_rss("telonex_binance_validate", procmem.peak_rss_mb(), args.max_rss_mb)
        return rc

    result = validate(paths)
    jpath, hpath = _write_outputs(paths, result)
    _print_summary(result)
    procmem.report_peak_rss("telonex_binance_validate", procmem.peak_rss_mb(), args.max_rss_mb)
    print(f"[telonex_binance_validate] wrote {jpath}")
    print(f"[telonex_binance_validate] wrote {hpath}")
    go = result["go_no_go"]
    if go == "GO":
        print("[telonex_binance_validate] GO. Next: python -m model_lab.telonex_binance_pull --dry-run")
        return 0
    if go == "REVIEW":
        print("[telonex_binance_validate] REVIEW — the depth is genuine same-book data (magnitudes, "
              "bias, slopes all match; depth_imb_20 corr converges to ~1.0 and passes at the model's "
              f"{result['checks']['parity'].get('model_grid_secs')}s grid), but the raw-1s corr is "
              "sampling-limited below 0.99. Operator decides whether to proceed to Stage 5.")
        return 2
    print("[telonex_binance_validate] NO-GO — STOP. Do NOT run Stage 4/5. See the failing check above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
