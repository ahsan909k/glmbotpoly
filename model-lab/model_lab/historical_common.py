"""Shared reconstruction primitives for the **historical** ingest path.

The lab's live path (``dataset`` / ``short_horizon`` / ``money_judge``) reads only the
recorder journal (2026-06-22 →). This module reconstructs the same quantities from the
pre-journal historical sources on disk, so a builder / backtest can extend across months:

- **Binance aggTrades** (``data/aggtrades``, 2025-01 →) — the mid / vol / strike backbone
  (long history), reduced to 1-second bars (reuses ``dataset._day_bars``).
- **Telonex Binance ``book_snapshot_25``** (``data/telonex``, 2026-02-06 → 07-07) — real L2
  depth microstructure, computed with the **same** ``short_horizon._frame_depth_features``
  math the live model uses (so the overlap-day parity guard stays authoritative). Read
  **chunked**, reduced to the last book per second, so a full day never resides in RAM.
- **Telonex Polymarket quotes** (``data/telonex``) — the PM book (fills + microstructure)
  and the window resolution via **quote convergence** (winner token → ~$1).

**Strike & basis (no Chainlink on historical days).** The strike ``K`` is
Binance-anchored (last trade ≤ open, the same rule the ``binance_proxy`` path already
uses). The Chainlink-definition features (``chainlink``, ``basis_bps``, ``basis_ewma`` and
the basis-*corrected* ``z``) are unavailable pre-journal and are left NaN / journal-era-only;
the emitted ``z`` is the price-only (basis = 0) definition, tagged ``label_source =
binance_proxy``. Overlap days (``>= JOURNAL_OWNED_FROM``) verify the two definitions against
the journal's Chainlink strike/z (they differ by ~13 bps by design — see the builder).

No look-ahead: every feature helper here feeds the *same* scanned kernels
(``features.build_asset_features`` / ``short_horizon.build_depth_feature_grid`` /
``build_pm_grid``); the future-reading resolvers are kept apart (labels only).
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import dataset as ds
from . import short_horizon as sh
from . import telonex_binance_validate as tbv
from .config import Paths
from .io import journal as jio
from .io import telonex as tx
from .io import telonex_pm as tpm  # noqa: F401  (re-exported for callers / tests)
from .lib import math as lm

_SEGMENT_DAY_RE = re.compile(r"journal-(\d{4})(\d{2})(\d{2})-")

# --- universe / day-ownership -----------------------------------------------
# The four series the historical path covers (Telonex Polymarket on disk).
DUR_SECS = {"5m": 300, "15m": 900, "1h": 3600}
SYMBOL_BY_ASSET = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
ASSET_BY_SYMBOL = {v: k for k, v in SYMBOL_BY_ASSET.items()}
# Telonex slug prefix ↔ (series key, asset, duration seconds).
SLUG_PREFIXES: dict[str, tuple[str, str, int]] = {
    "btc-updown-5m": ("BTC-5m", "btc", 300),
    "eth-updown-5m": ("ETH-5m", "eth", 300),
    "btc-updown-15m": ("BTC-15m", "btc", 900),
    "eth-updown-15m": ("ETH-15m", "eth", 900),
}
SERIES_TO_PREFIX = {v[0]: k for k, v in SLUG_PREFIXES.items()}
DEFAULT_SERIES: tuple[str, ...] = ("BTC-5m", "ETH-5m", "BTC-15m", "ETH-15m")

# The recorder journal has full feature coverage (Chainlink + depth20 + PM book) from this
# UTC day on; earlier days are Telonex-owned. The single source of truth for day ownership.
JOURNAL_OWNED_FROM = date(2026, 7, 4)

_SLUG_RE = re.compile(r"^([a-z]+-updown-(?:5m|15m|1h))-(\d+)$")


def day_owner(d: date) -> str:
    """``"journal"`` for days the recorder fully covers (``>= JOURNAL_OWNED_FROM``),
    else ``"telonex"``. The one place day ownership is decided (Part 1 dispatch + the
    overlap-parity check)."""
    return "journal" if d >= JOURNAL_OWNED_FROM else "telonex"


def parse_slug(slug: str) -> tuple[str, str, int, int] | None:
    """``slug`` → ``(series_key, asset, dur_secs, open_epoch)`` or ``None`` if it is not
    an in-universe ``{asset}-updown-{dur}-{epoch}`` slug."""
    m = _SLUG_RE.match(slug.strip().lower())
    if not m:
        return None
    prefix, epoch = m.group(1), int(m.group(2))
    meta = SLUG_PREFIXES.get(prefix)
    if meta is None:
        return None
    series, asset, dur = meta
    return series, asset, dur, epoch


def series_prefix(series_key: str) -> str | None:
    return SERIES_TO_PREFIX.get(series_key)


def utc_day(ms: int) -> date:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).date()


def day_bounds_ms(d: date) -> tuple[int, int]:
    start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) * 1000
    return start, start + 86_400_000


# --- window enumeration + missing-window exclusion --------------------------
def _coverage_path(paths: Paths) -> Path:
    return paths.out_dir / "telonex" / "coverage.json"


def load_excluded_slugs(paths: Paths) -> tuple[set[str], dict]:
    """Read ``out/telonex/coverage.json`` and return the set of window slugs to
    **exclude** because a needed channel is missing, plus a small report.

    A ``missing_windows`` entry is ``"{slug}/{outcome}/{channel}"`` (a file the markets
    catalog said had data but Telonex 404'd). A window is unusable for training/backtest
    iff its ``quotes`` are missing for **either** outcome (we need both Up and Down quotes
    for resolution + fills); a missing ``trades`` file alone does not exclude it. Never
    zero-filled — excluded windows are dropped and counted. Empty set if coverage.json is
    absent (enumeration then rests on on-disk file presence alone)."""
    path = _coverage_path(paths)
    excluded: set[str] = set()
    report = {"coverage_json": str(path), "present": path.exists(),
              "missing_window_entries": 0, "excluded_slugs": 0}
    if not path.exists():
        return excluded, report
    try:
        cov = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return excluded, report
    entries = 0
    for series in (cov.get("series") or {}).values():
        for day in (series.get("days") or {}).values():
            for entry in day.get("missing_windows") or []:
                entries += 1
                parts = str(entry).split("/")
                if len(parts) != 3:
                    continue
                slug, _outcome, channel = parts
                if channel == "quotes":
                    excluded.add(slug)
    report["missing_window_entries"] = entries
    report["excluded_slugs"] = len(excluded)
    return excluded, report


def _quotes_dir(paths: Paths) -> Path:
    return tx.raw_root(paths.telonex_dir) / tx.POLYMARKET / "quotes"


def _outcome_file_present(paths: Paths, slug: str, outcome: str, day_str: str) -> bool:
    return tx.raw_or_zst_exists(
        tx.polymarket_raw_path(paths.telonex_dir, "quotes", slug, outcome, day_str)
    )


def telonex_windows_present(
    paths: Paths, series_key: str, d: date, *, excluded: set[str] | None = None
) -> tuple[list[dict], dict]:
    """Enumerate the Telonex-owned windows for ``series_key`` opening on UTC day ``d``.

    A window is *present* iff both its Up and Down ``quotes`` files are on disk and its
    slug is not in ``excluded`` (from :func:`load_excluded_slugs`). Returns
    ``([{series, asset, dur_secs, slug, open_epoch, window_open_ms, window_close_ms}], report)``
    sorted by open time. Token ids are read later (from the quotes), not here."""
    prefix = series_prefix(series_key)
    report = {"candidates": 0, "excluded_missing": 0, "missing_down": 0, "kept": 0}
    if prefix is None:
        return [], report
    excluded = excluded or set()
    day_str = d.isoformat()
    qdir = _quotes_dir(paths)
    pat = re.compile(rf"^{re.escape(prefix)}-(\d+)-Up-{re.escape(day_str)}\.parquet(?:\.zst)?$")
    seen: set[int] = set()
    rows: list[dict] = []
    if qdir.is_dir():
        for p in qdir.glob(f"{prefix}-*-Up-{day_str}.parquet*"):
            m = pat.match(p.name)
            if not m:
                continue
            epoch = int(m.group(1))
            if epoch in seen:
                continue
            seen.add(epoch)
            report["candidates"] += 1
            slug = f"{prefix}-{epoch}"
            if slug in excluded:
                report["excluded_missing"] += 1
                continue
            if not _outcome_file_present(paths, slug, "Down", day_str):
                report["missing_down"] += 1
                continue
            _series, asset, dur, _ep = parse_slug(slug)
            open_ms = epoch * 1000
            rows.append({
                "series": series_key, "asset": asset, "dur_secs": dur, "slug": slug,
                "open_epoch": epoch, "window_open_ms": open_ms,
                "window_close_ms": open_ms + dur * 1000,
            })
    rows.sort(key=lambda r: r["window_open_ms"])
    report["kept"] = len(rows)
    return rows, report


# --- Binance aggTrades bars (mid / vol / strike backbone) -------------------
def binance_bars_for_day(paths: Paths, symbol: str, d: date) -> pd.DataFrame:
    """1-second Binance bars (``sec, price``) covering UTC day ``d`` plus one buffer day
    on each side — the buffer warms the σ EWMA at day start and gives the last window its
    close / forward bar. Reuses ``dataset._day_bars`` (reads only ``transact_time, price``)
    so only ~3 days of 1-second bars are ever resident. Empty if no aggTrades for those
    days."""
    frames: list[pd.DataFrame] = []
    for dd in (d - timedelta(days=1), d, d + timedelta(days=1)):
        p = paths.hist_dir / f"aggtrades-{symbol}-{dd.isoformat()}.parquet"
        if p.exists():
            b = ds._day_bars(p, symbol)
            if not b.empty:
                frames.append(b)
    if not frames:
        return pd.DataFrame({"sec": pd.Series([], dtype="int64"), "price": pd.Series([], dtype="float64")})
    allb = pd.concat(frames, ignore_index=True)
    return allb.groupby("sec", as_index=False)["price"].last().sort_values("sec").reset_index(drop=True)


def binance_anchored_strike(bars: pd.DataFrame, open_ms: int, tol_ms: float) -> dict:
    """Strike = the last Binance 1-second bar at-or-before ``open`` (``LastAtOrBefore``,
    §6). Returns ``{strike, strike_ts_ms, strike_quality}`` (``boundary`` within ``tol_ms``
    of open, else ``distant``; ``missing`` when no bar qualifies)."""
    if bars.empty:
        return {"strike": float("nan"), "strike_ts_ms": None, "strike_quality": "missing"}
    secs = bars["sec"].to_numpy(dtype="int64")
    prices = bars["price"].to_numpy(dtype=float)
    open_sec = open_ms // 1000
    i = int(np.searchsorted(secs, open_sec, side="right")) - 1
    if i < 0:
        return {"strike": float("nan"), "strike_ts_ms": None, "strike_quality": "missing"}
    strike = float(prices[i])
    strike_ts = int(secs[i]) * 1000
    quality = "boundary" if (open_ms - strike_ts) <= tol_ms else "distant"
    return {"strike": strike, "strike_ts_ms": strike_ts, "strike_quality": quality}


def binance_window_close(bars: pd.DataFrame, open_ms: int, close_ms: int) -> tuple[float, float]:
    """``(close_price, realized_1s_vol)`` for a window over the bar series — the
    resolution-feed close (last bar ≤ close − 1s) and the window's realized 1-second
    vol. Reuses ``dataset._window_close_and_vol``. Reads the future — labels/diagnostics
    only, never a feature."""
    if bars.empty:
        return float("nan"), float("nan")
    return ds._window_close_and_vol(
        bars["sec"].to_numpy(dtype="int64"), bars["price"].to_numpy(dtype=float),
        open_ms // 1000, close_ms // 1000,
    )


def binance_outcome(bars: pd.DataFrame, open_ms: int, close_ms: int, tol_ms: float) -> dict:
    """Binance-anchored resolution (the cross-check): ``end ≥ strike ⇒ Up`` (§6 tie → Up).
    Returns ``{outcome, strike, end_price, strike_ts_ms, strike_quality,
    strike_distance_at_close, strike_distance_at_close_vol}`` — ``outcome`` is ``None`` when
    the strike or close is missing."""
    st = binance_anchored_strike(bars, open_ms, tol_ms)
    strike = st["strike"]
    close_price, realized = binance_window_close(bars, open_ms, close_ms)
    abs_d, vol_d = ds._strike_distance(strike, close_price, realized)
    outcome = None
    if strike > 0.0 and close_price > 0.0:
        outcome = "Up" if close_price >= strike else "Down"
    return {
        "outcome": outcome, "strike": strike, "end_price": close_price,
        "strike_ts_ms": st["strike_ts_ms"], "strike_quality": st["strike_quality"],
        "strike_distance_at_close": abs_d, "strike_distance_at_close_vol": vol_d,
    }


# --- quote-convergence resolution (Part 2 primary) --------------------------
CONVERGE_WIN = 0.98      # winner token's final mid must reach ≥ this
CONVERGE_LOSE = 0.02     # loser token's final mid must fall to ≤ this
CONVERGE_TAIL_SECS = 30  # the "last observed stretch" — final quotes must come from within this
CONVERGE_MIN_TAIL = 3    # need at least this many quotes in the stretch (a stretch, not one tick)
CONVERGE_GRACE_SECS = 120  # accept resolution quotes up to this many seconds after close


def resolve_by_quote_convergence(
    up_quotes: pd.DataFrame, down_quotes: pd.DataFrame, open_ms: int, close_ms: int,
    *, win: float = CONVERGE_WIN, lose: float = CONVERGE_LOSE,
    tail_secs: int = CONVERGE_TAIL_SECS, min_tail: int = CONVERGE_MIN_TAIL,
    grace_secs: int = CONVERGE_GRACE_SECS,
) -> dict:
    """Resolve a window from the Polymarket quote convergence (independent of any price
    feed). The winner is the token whose **final** mid (the last quote at-or-before
    ``close + grace``) has decisively converged toward $1: it resolves iff the winner's
    final mid ``≥ win`` **and** the loser's final mid ``≤ lose``, with each side backed by
    ``≥ min_tail`` quotes in the last ``tail_secs`` (a genuine stretch, not one lonely
    tick). Anything short — the market never decisively separated in the recorded data —
    is **ambiguous** and excluded (never guessed). Telonex quotes stop ~at close (before
    settlement fully prints $1), so the final mid caps around 0.97–0.99; the ``win/lose``
    defaults (operator's 0.98/0.02) resolve the decisively-converged majority and exclude
    the rest, cross-validated against Binance.

    Returns ``{outcome, resolved, reason, up_final, down_final, n_tail_up, n_tail_down}``.
    ``outcome`` is ``"Up"``/``"Down"`` when resolved, else ``None``."""
    hi = close_ms + grace_secs * 1000

    def final_and_count(q: pd.DataFrame) -> tuple[float, int]:
        if q.empty:
            return float("nan"), 0
        qq = q[(q["ts_ms"] >= open_ms) & (q["ts_ms"] <= hi)].sort_values("ts_ms")
        if qq.empty:
            return float("nan"), 0
        last_ts = int(qq["ts_ms"].iloc[-1])
        tl = qq[qq["ts_ms"] >= last_ts - tail_secs * 1000]
        mid = 0.5 * (tl["bid_px"].to_numpy(dtype=float) + tl["ask_px"].to_numpy(dtype=float))
        return float(mid[-1]), int(len(tl))

    up_f, n_up = final_and_count(up_quotes)
    dn_f, n_dn = final_and_count(down_quotes)
    detail = {
        "outcome": None, "resolved": False, "reason": "no_tail_quotes",
        "up_final": up_f, "down_final": dn_f, "n_tail_up": n_up, "n_tail_down": n_dn,
    }
    if not (np.isfinite(up_f) and np.isfinite(dn_f) and n_up >= min_tail and n_dn >= min_tail):
        detail["reason"] = "insufficient_tail"
        return detail
    up_wins = bool(up_f >= win and dn_f <= lose)
    down_wins = bool(dn_f >= win and up_f <= lose)
    if up_wins and not down_wins:
        detail.update(outcome="Up", resolved=True, reason="converged")
    elif down_wins and not up_wins:
        detail.update(outcome="Down", resolved=True, reason="converged")
    else:
        detail["reason"] = "not_converged"
    return detail


# --- journal overlap outcomes (segment-filtered, bounded) -------------------
def _segment_day(path: Path) -> date | None:
    m = _SEGMENT_DAY_RE.search(path.name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def journal_resolved_outcomes(
    paths: Paths, series_filter: set[str] | None, day_lo: date, day_hi: date,
    *, max_segments: int | None = None,
) -> dict[tuple[str, int], str]:
    """The journal's authoritative ``Resolved`` outcomes for windows opening in
    ``[day_lo, day_hi]`` — reading **only** the journal segments whose filename date is in
    that range (so the overlap check never decompresses the whole multi-GB journal).
    Returns ``{(series, open_ms): "Up"|"Down"}``. Empty if no journal or no in-range
    segments (e.g. a Telonex-only range). Settlement is the fallback for a window with no
    ``Resolved`` event. ``max_segments`` caps how many in-range segments are read (each is
    ~1 h of a 24/7 day = dozens of windows) — bounds the multi-GB scan for a spot-check."""
    resolved: dict[tuple[str, int], str] = {}
    settle: dict[tuple[str, int], str] = {}
    # start at day_lo (windows can't be journaled before they open) with a trailing-day
    # buffer (a window opening late on day_hi resolves the next day). No leading buffer, so
    # a capped read spends its budget on the overlap days, not the day before.
    lo, hi = day_lo, day_hi + timedelta(days=1)
    read = 0
    for seg in jio.segment_paths(paths.journal_dir):
        sd = _segment_day(seg)
        if sd is None or sd < lo or sd > hi:
            continue
        if max_segments is not None and read >= max_segments:
            break
        read += 1
        for rec in _read_one_segment(seg):
            kind = rec.get("type")
            if kind == "window":
                market = rec.get("market") or {}
                key = jio.window_key(market.get("window"))
                if key == ("", 0) or (series_filter and key[0] not in series_filter):
                    continue
                lifecycle = rec.get("lifecycle")
                if isinstance(lifecycle, dict) and "Resolved" in lifecycle:
                    out = (lifecycle["Resolved"] or {}).get("outcome")
                    if out in ("Up", "Down"):
                        resolved[key] = out
            elif kind == "settlement":
                key = jio.window_key(rec.get("window"))
                if key == ("", 0) or (series_filter and key[0] not in series_filter):
                    continue
                out = rec.get("outcome")
                if out in ("Up", "Down"):
                    settle[key] = out
    for key, out in settle.items():
        resolved.setdefault(key, out)
    return resolved


def _read_one_segment(seg: Path):
    """Yield the ``rec`` objects of a single journal segment (mirrors ``jio.read_records``
    for one file — tolerates a crash-truncated / still-being-written tail)."""
    import gzip
    import json as _json
    import zlib
    try:
        with gzip.open(seg, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    env = _json.loads(line)
                except _json.JSONDecodeError:
                    break
                rec = env.get("rec")
                if isinstance(rec, dict):
                    yield rec
    except (OSError, EOFError, zlib.error):
        return


# --- Telonex Binance book_snapshot_25 depth features (memory-safe) ----------
def _book_path(paths: Paths, symbol: str, d: date) -> Path:
    return tx.binance_raw_path(paths.telonex_dir, "book_snapshot_25", symbol, d.isoformat())


def _level_col_names() -> list[str]:
    cols: list[str] = []
    for side in ("bid", "ask"):
        for i in range(tbv.LEVELS):
            cols += [f"{side}_price_{i}", f"{side}_size_{i}"]
    return cols


def _last_per_second(book_path: Path, since_ms: int, until_ms: int, *, batch_size: int = 65536) -> pd.DataFrame:
    """Stream a ``book_snapshot_25`` parquet in row-group batches, keeping the **last**
    row per second in ``[since_ms, until_ms)`` (exchange ms). Reads the whole 25-level
    ladder but never holds more than one batch + the per-second survivors — a full day
    (~850k rows × 100 string cols) would otherwise be multi-GB. Returns the survivors
    (string level cells + ``timestamp_us``), ≤ one row per second."""
    level_cols = _level_col_names()
    read_cols = ["timestamp_us", *level_cols]
    pf = pq.ParquetFile(book_path)
    survivors: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        df = batch.to_pandas()
        ts_us = df["timestamp_us"].to_numpy(dtype="int64")
        ms = ts_us // 1000
        keep = (ms >= since_ms) & (ms < until_ms)
        if not keep.any():
            continue
        df = df[keep]
        df = df.assign(_sec=(ts_us[keep] // 1_000_000))
        # last-per-second within the batch (batches are timestamp-ordered)
        df = df.sort_values("timestamp_us").drop_duplicates("_sec", keep="last")
        survivors.append(df)
    if not survivors:
        return pd.DataFrame(columns=["_sec", "timestamp_us", *level_cols])
    allsurv = pd.concat(survivors, ignore_index=True)
    return allsurv.sort_values("timestamp_us").drop_duplicates("_sec", keep="last").reset_index(drop=True)


def telonex_depth_feature_grid(paths: Paths, symbol: str, asset: str, d: date) -> pd.DataFrame:
    """Per-second (as-of) depth-feature grid for ``asset`` on UTC day ``d`` from the
    Telonex ``book_snapshot_25`` — computed with the **same** ``short_horizon`` math the
    live model + parity guard use (byte-identical feature definition). Read chunked
    (last book per second). Empty if the file is absent."""
    empty = sh._empty_depth_grid()
    raw = _book_path(paths, symbol, d)
    if not tx.raw_or_zst_exists(raw):
        return empty
    since_ms, until_ms = day_bounds_ms(d)
    scratch = paths.out_dir / "_hist_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        if raw.exists():
            src = raw
        else:
            tmp = scratch / (raw.name + ".depth.tmp.parquet")
            tx.decompress_zstd(tx.zst_sibling(raw), tmp)
            src = tmp
        surv = _last_per_second(src, since_ms, until_ms)
    finally:
        if tmp is not None:
            tx._safe_unlink(tmp)
    if surv.empty:
        return empty

    def block(prefix: str) -> np.ndarray:
        cols = [f"{prefix}_{i}" for i in range(tbv.LEVELS)]
        return surv[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")

    bid_px, bid_sz = block("bid_price"), block("bid_size")
    ask_px, ask_sz = block("ask_price"), block("ask_size")
    recv = surv["timestamp_us"].to_numpy(dtype="int64") // 1000
    frames: list[dict] = []
    for k in range(len(surv)):
        nb = tbv._valid_prefix(bid_px[k], bid_sz[k])
        na = tbv._valid_prefix(ask_px[k], ask_sz[k])
        if nb == 0 or na == 0:
            continue
        frames.append({
            "recv_ms": int(recv[k]), "asset": asset,
            "bid_px": bid_px[k, :nb], "bid_sz": bid_sz[k, :nb],
            "ask_px": ask_px[k, :na], "ask_sz": ask_sz[k, :na],
        })
    return sh.build_depth_feature_grid(frames, asset)


def telonex_top_mid(paths: Paths, symbol: str, d: date) -> pd.DataFrame:
    """Fine-grained top-of-book mid series (``ts_ms, mid``) from the Telonex
    ``book_snapshot_25`` for UTC day ``d`` — the closest historical analogue of the live
    ``BinanceDirect``/``Mid`` bookTicker feed (~100 ms cadence), for the momentum ring.
    Reads only ``timestamp_us``/``bid_price_0``/``ask_price_0`` (light). Empty if absent."""
    raw = _book_path(paths, symbol, d)
    if not tx.raw_or_zst_exists(raw):
        return pd.DataFrame(columns=["ts_ms", "mid"])
    since_ms, until_ms = day_bounds_ms(d)
    scratch = paths.out_dir / "_hist_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        if raw.exists():
            src = raw
        else:
            # Process-unique so parallel stages reading the same depth .zst don't collide.
            tmp = scratch / (raw.name + f".{os.getpid()}.mid.tmp.parquet")
            tx.decompress_zstd(tx.zst_sibling(raw), tmp)
            src = tmp
        pf = pq.ParquetFile(src)
        parts: list[pd.DataFrame] = []
        for batch in pf.iter_batches(batch_size=262144,
                                     columns=["timestamp_us", "bid_price_0", "ask_price_0"]):
            df = batch.to_pandas()
            ms = df["timestamp_us"].to_numpy(dtype="int64") // 1000
            keep = (ms >= since_ms) & (ms < until_ms)
            if not keep.any():
                continue
            bid = pd.to_numeric(df["bid_price_0"], errors="coerce").to_numpy(dtype=float)[keep]
            ask = pd.to_numeric(df["ask_price_0"], errors="coerce").to_numpy(dtype=float)[keep]
            mid = 0.5 * (bid + ask)
            good = np.isfinite(mid) & (bid > 0.0) & (ask > 0.0)
            if good.any():
                parts.append(pd.DataFrame({"ts_ms": ms[keep][good], "mid": mid[good]}))
    finally:
        if tmp is not None:
            tx._safe_unlink(tmp)
    if not parts:
        return pd.DataFrame(columns=["ts_ms", "mid"])
    return pd.concat(parts, ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


# --- model reconstruction (1-second snapshots, for the backtest) -------------
def reconstruct_window_snapshots(grid: pd.DataFrame, window: dict) -> pd.DataFrame:
    """1-second reconstructed model snapshots inside ``[open, close)`` for one window,
    from the asset's as-of feature grid (``dataset.build_feature_grid`` output) and the
    window's Binance-anchored strike. Columns ``ts, p_up, sigma_1s, mid, health`` where a
    warmed σ + fresh mid exists (the historical analogue of the engine's Ready gate —
    there is no Chainlink health pre-journal). ``z`` uses the price-only (basis = 0)
    definition. No look-ahead: each row uses only its own second's bar + the open strike."""
    cols = ["ts", "p_up", "sigma_1s", "mid", "health"]
    strike = float(window.get("strike", float("nan")))
    if grid.empty or not (strike > 0.0):
        return pd.DataFrame(columns=cols)
    open_ms, close_ms = int(window["window_open_ms"]), int(window["window_close_ms"])
    open_sec, close_sec = open_ms // 1000, close_ms // 1000
    g = grid[(grid["sec"] >= open_sec) & (grid["sec"] < close_sec)]
    g = g.dropna(subset=["mid", "sigma_1s"])
    g = g[(g["mid"] > 0.0) & (g["sigma_1s"] > 0.0)]
    if g.empty:
        return pd.DataFrame(columns=cols)
    ts = g["ts_ms"].to_numpy(dtype="int64")
    mid = g["mid"].to_numpy(dtype=float)
    sigma = g["sigma_1s"].to_numpy(dtype=float)
    tau = (close_ms - ts) / 1000.0
    ok = tau > 0.0
    ts, mid, sigma, tau = ts[ok], mid[ok], sigma[ok], tau[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.clip(np.log(mid / strike) / (sigma * np.sqrt(tau)), -lm.Z_CLAMP, lm.Z_CLAMP)
    p_up = lm.norm_cdf(z)
    return pd.DataFrame({
        "ts": ts, "p_up": p_up, "sigma_1s": sigma, "mid": mid,
        "health": np.array(["Ready"] * len(ts), dtype=object),
    })
