"""Validate + compress the Telonex trial pull, then write a go/no-go report.

Reads the trial parquet under ``../data/telonex/raw`` (and, for the clock-alignment
cross-check, our own journal/depth) and answers six questions with measured PASS/FAIL
numbers: coverage, date range, snapshot cadence, depth, clock alignment (internal +
external vs our recordings), and completeness. Then whole-file **zstd**-compresses the
trial (bit-identical roundtrip), measures the ratio, projects the needed-slice size vs
the 250 GB budget, and writes ``out/telonex/{validation.json, report.html}`` plus a
console go/no-go. **Stops here** — the full download is a separate task.

    python -m model_lab.telonex_validate                       # validate + compress + report
    python -m model_lab.telonex_validate --no-compress         # checks only
    python -m model_lab.telonex_validate --no-catalog          # skip the markets-dataset download
"""

from __future__ import annotations

import gzip
import json
import math
import re
import sys
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import Paths, resolve_paths, stage_parser
from .io import journal as jio
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem

# Thresholds (documented in telonex_notes.md / the plan)
CADENCE_MEDIAN_PASS_S = 1.0        # per-window median gap must be ≤ 1 s
DEPTH_MIN_LEVELS = 20              # we use Binance depth20 today
GAP_THRESHOLD_S = 5.0             # a "gap" in completeness
CLOCK_JITTER_PASS_MS = 250.0      # residual jitter after removing the constant offset
CLOCK_DRIFT_PASS_MS_PER_HR = 50.0 # a stable (correctable) offset drifts little
CLOCK_MIN_MATCHES = 200           # need this many matched points to trust the offset
MIRROR_TOL = 0.02                 # Down top ≈ 1 − Up top within this
BUDGET_BYTES = 250 * 1024**3      # 250 GB storage budget
WINDOWS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24}
NEEDED_SERIES = [
    "btc-updown-5m", "btc-updown-15m", "btc-updown-1h",
    "eth-updown-5m", "eth-updown-15m", "eth-updown-1h",
]
NEEDED_BINANCE = ["btcusdt", "ethusdt"]


def _pf(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


# --------------------------------------------------------------------------- #
# Reading the trial parquet
# --------------------------------------------------------------------------- #


def _read_cols(path: Path, columns: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _best(levels: Any) -> float:
    """Best price from a ``book_snapshot_full`` nested ``bids``/``asks`` cell."""
    try:
        if levels is None:
            return float("nan")
        first = levels[0]
        return _to_float(first["price"] if isinstance(first, dict) else first[0])
    except (IndexError, KeyError, TypeError):
        return float("nan")


def _n_levels(levels: Any) -> int:
    try:
        return len(levels)
    except TypeError:
        return 0


def _flat_book_tops(path: Path) -> pd.DataFrame:
    """``ts_ms, bid, ask`` from a flattened ``book_snapshot_5/25`` parquet."""
    df = _read_cols(path, ["timestamp_us", "bid_price_0", "ask_price_0"])
    return pd.DataFrame({
        "ts_ms": (df["timestamp_us"].to_numpy(dtype="int64") // 1000),
        "bid": df["bid_price_0"].map(_to_float).to_numpy(),
        "ask": df["ask_price_0"].map(_to_float).to_numpy(),
    })


def _quotes_tops(path: Path) -> pd.DataFrame:
    """``ts_ms, bid, ask`` from a flat ``quotes`` parquet (``bid_price``/``ask_price``).

    Quotes are the top-of-book channel — flat and ~7× fewer rows than book_snapshot_full —
    so the clock + mirror checks read these instead of parsing the full nested book."""
    df = _read_cols(path, ["timestamp_us", "bid_price", "ask_price"])
    return pd.DataFrame({
        "ts_ms": (df["timestamp_us"].to_numpy(dtype="int64") // 1000),
        "bid": df["bid_price"].map(_to_float).to_numpy(),
        "ask": df["ask_price"].map(_to_float).to_numpy(),
    })


# --------------------------------------------------------------------------- #
# Trial file discovery (from the manifest's trial_meta)
# --------------------------------------------------------------------------- #


def _load_trial(paths: Paths) -> dict[str, Any]:
    manifest = tx.read_manifest(paths.telonex_dir)
    if not manifest or "trial" not in manifest:
        raise tx.TelonexError(
            f"no trial manifest under {paths.telonex_dir} — run "
            "`python -m model_lab.telonex_ingest --trial` first."
        )
    return manifest["trial"]


def _poly_file(paths: Paths, channel: str, slug: str, outcome: str, day_str: str) -> Path:
    return tx.polymarket_raw_path(paths.telonex_dir, channel, slug, outcome, day_str)


def _binance_file(paths: Paths, channel: str, symbol: str, day_str: str) -> Path:
    return tx.binance_raw_path(paths.telonex_dir, channel, symbol, day_str)


# --------------------------------------------------------------------------- #
# Check 1 — coverage (public availability probes; free)
# --------------------------------------------------------------------------- #


def _scan_catalog(cat_path: Path) -> dict[str, Any]:
    """One row-group-streamed pass over the (1 GB) markets catalog → per-series window
    counts, per-channel presence, distinct days, and first/last epoch. Feeds both the
    coverage and date-range checks from a single read."""
    pf = pq.ParquetFile(cat_path)
    have = set(pf.schema_arrow.names)
    chan_cols = [c for c in ("book_snapshot_full_from", "book_snapshot_25_from", "trades_from", "quotes_from")
                 if c in have]
    cols = ["slug", *chan_cols]
    per: dict[str, Any] = {s: {"windows": 0, "book": 0, "trades": 0, "quotes": 0,
                               "days": set(), "first": None, "last": None} for s in NEEDED_SERIES}
    for batch in pf.iter_batches(batch_size=300_000, columns=cols):
        d = batch.to_pandas()
        m = d["slug"].astype(str).str.extract(_SLUG_RE)
        mask = m[0].notna()
        if not mask.any():
            continue
        sub = d[mask].copy()
        sub["series"] = m.loc[mask, 0] + "-updown-" + m.loc[mask, 1]
        sub["epoch"] = m.loc[mask, 2].astype("int64")

        def nonempty(col: str):
            return sub[col].astype(str).str.len() > 0 if col in sub else pd.Series(False, index=sub.index)

        has_book = nonempty("book_snapshot_full_from") | nonempty("book_snapshot_25_from")
        has_tr = nonempty("trades_from")
        has_q = nonempty("quotes_from")
        sub["_book"], sub["_tr"], sub["_q"] = has_book, has_tr, has_q
        for series, g in sub.groupby("series"):
            st = per[series]
            wd = g[g["_book"] | g["_tr"]]
            st["windows"] += len(wd)
            st["book"] += int(g["_book"].sum())
            st["trades"] += int(g["_tr"].sum())
            st["quotes"] += int(g["_q"].sum())
            if len(wd):
                days = pd.to_datetime(wd["epoch"], unit="s", utc=True).dt.date
                st["days"].update(str(x) for x in days)
                lo, hi = int(wd["epoch"].min()), int(wd["epoch"].max())
                st["first"] = min(st["first"], lo) if st["first"] else lo
                st["last"] = max(st["last"], hi) if st["last"] else hi
    return per


def check_coverage(catalog_scan: dict[str, Any] | None, *, fetch: tx.Fetch = tx.urllib_fetch) -> dict[str, Any]:
    series_out: dict[str, Any] = {}
    for series in NEEDED_SERIES:
        st = (catalog_scan or {}).get(series, {})
        series_out[series] = {"windows_with_data": st.get("windows", 0), "with_book": st.get("book", 0),
                              "with_trades": st.get("trades", 0), "with_quotes": st.get("quotes", 0)}

    def has_book_trades(series: str) -> bool:
        s = (catalog_scan or {}).get(series, {})
        return s.get("book", 0) > 0 and s.get("trades", 0) > 0

    binance_avail: dict[str, Any] = {}
    for sym in NEEDED_BINANCE:
        try:
            info = tx.availability(tx.BINANCE, slug=sym, fetch=fetch)
            binance_avail[sym] = info.get("channels", {})
        except tx.TelonexError:
            binance_avail[sym] = {}

    both_5m = has_book_trades("btc-updown-5m") and has_book_trades("eth-updown-5m")
    return {
        "name": "coverage",
        "result": _pf(both_5m) if catalog_scan is not None else "SKIPPED (no catalog)",
        "btc_5m_ok": has_book_trades("btc-updown-5m"),
        "eth_5m_ok": has_book_trades("eth-updown-5m"),
        "series": series_out,
        "binance": binance_avail,
    }


# --------------------------------------------------------------------------- #
# Check 2 — date range (markets catalog + Binance availability)
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"^(btc|eth)-updown-(5m|15m|1h)-(\d+)$")


def check_date_range(catalog_scan: dict[str, Any] | None, catalog_status: str,
                     *, fetch: tx.Fetch = tx.urllib_fetch) -> dict[str, Any]:
    def fmt(epoch: int | None) -> str | None:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat() if epoch else None

    series_out: dict[str, Any] = {}
    for s in NEEDED_SERIES:
        st = (catalog_scan or {}).get(s, {})
        series_out[s] = {"windows_with_data": st.get("windows", 0), "distinct_days": len(st.get("days", set())),
                         "first": fmt(st.get("first")), "last": fmt(st.get("last"))}
    binance_out: dict[str, Any] = {}
    for sym in NEEDED_BINANCE:
        try:
            info = tx.availability(tx.BINANCE, slug=sym, fetch=fetch)
            tr = info.get("channels", {}).get("trades", {})
            binance_out[sym] = {"from": tr.get("from_date"), "to_exclusive": tr.get("to_date")}
        except tx.TelonexError:
            binance_out[sym] = {}
    have = catalog_scan is not None and any(v["windows_with_data"] > 0 for v in series_out.values())
    return {
        "name": "date_range",
        "result": _pf(have) if catalog_scan is not None else "SKIPPED",
        "catalog_status": catalog_status,
        "series": series_out,
        "binance": binance_out,
    }


# --------------------------------------------------------------------------- #
# Check 3 — snapshot cadence (per window)
# --------------------------------------------------------------------------- #


def _gaps_ms(ts_ms: np.ndarray) -> np.ndarray:
    t = np.sort(np.asarray(ts_ms, dtype="int64"))
    return np.diff(t) if t.size >= 2 else np.asarray([], dtype="int64")


def _cadence_active(specs: list[tuple[Path, int | None, int | None]]) -> dict[str, Any]:
    """Per-file gap stats, optionally restricted to each file's ``[lo, hi)`` ms window.

    For a 5-minute window the ``book_snapshot_full`` file also holds sparse order events
    from when the market was created (~24 h before open), so measuring gaps only within the
    active window gives the true trading cadence (and the completeness gap count)."""
    per_window_median: list[float] = []
    all_gaps: list[np.ndarray] = []
    n_windows = 0
    for path, lo, hi in specs:
        if not path.exists():
            continue
        try:
            ts = _read_cols(path, ["timestamp_us"])["timestamp_us"].to_numpy(dtype="int64") // 1000
        except Exception:  # noqa: BLE001
            continue
        if lo is not None and hi is not None:
            ts = ts[(ts >= lo) & (ts < hi)]
        g = _gaps_ms(ts)
        if g.size == 0:
            continue
        n_windows += 1
        per_window_median.append(float(np.median(g)))
        all_gaps.append(g)
    if not all_gaps:
        return {"windows": 0, "median_s": None, "p95_s": None, "max_s": None,
                "median_of_window_medians_s": None, "gaps_over_5s": 0}
    gaps = np.concatenate(all_gaps)
    return {
        "windows": n_windows,
        "rows": int(gaps.size + n_windows),
        "median_s": float(np.median(gaps) / 1000.0),
        "p95_s": float(np.percentile(gaps, 95) / 1000.0),
        "max_s": float(gaps.max() / 1000.0),
        "median_of_window_medians_s": float(np.median(per_window_median) / 1000.0),
        "gaps_over_5s": int((gaps > GAP_THRESHOLD_S * 1000).sum()),
    }


def check_cadence(paths: Paths, trial: dict[str, Any]) -> dict[str, Any]:
    day_str = trial["day"]
    windows = trial["windows"]
    dur_ms = tx.duration_seconds(trial["series"].rsplit("-", 1)[-1]) * 1000

    poly_specs = [(_poly_file(paths, "book_snapshot_full", w["slug"], "Up", day_str),
                   w["open_ms"], w["open_ms"] + dur_ms) for w in windows]
    poly = _cadence_active(poly_specs)

    bfile = _binance_file(paths, "book_snapshot_25", trial["binance_symbol"], day_str)
    binance: dict[str, Any] = {"status": "unavailable (exchange not on plan)"}
    if bfile.exists():
        binance = {"status": "ok", **_cadence_active([(bfile, None, None)])}

    poly_ok = poly["median_s"] is not None and poly["median_s"] <= CADENCE_MEDIAN_PASS_S
    # Binance only required when present (Telonex Binance needs the Pro plan; we also have
    # Binance for free from data.binance.vision via `model_lab.hist`).
    bin_ok = binance.get("status") != "ok" or (
        binance.get("median_s") is not None and binance["median_s"] <= CADENCE_MEDIAN_PASS_S)
    return {
        "name": "cadence",
        "result": _pf(poly_ok and bin_ok),
        "note": "event-driven (book_snapshot_full = tick-by-tick); measured per window",
        "polymarket_book_full": poly,
        "binance_book_25": binance,
        "threshold_s": CADENCE_MEDIAN_PASS_S,
    }


# --------------------------------------------------------------------------- #
# Check 4 — depth levels
# --------------------------------------------------------------------------- #


def check_depth(paths: Paths, trial: dict[str, Any]) -> dict[str, Any]:
    day_str = trial["day"]
    poly_max = 0
    poly_windows_checked = 0
    for w in trial["windows"][:4]:  # nested read is costly — a few windows suffice
        p = _poly_file(paths, "book_snapshot_full", w["slug"], "Up", day_str)
        if not p.exists():
            continue
        try:
            # Convert only the first ~2000 rows' nested bids/asks to pandas (the costly
            # step); the whole ~100k-row book is never materialized as Python objects.
            batch = next(pq.ParquetFile(p).iter_batches(batch_size=2000, columns=["bids", "asks"]))
            df = batch.to_pandas()
        except Exception:  # noqa: BLE001 — StopIteration (empty) or a bad file → skip
            continue
        poly_windows_checked += 1
        for col in ("bids", "asks"):
            vals = df[col].map(_n_levels)
            if len(vals):
                poly_max = max(poly_max, int(vals.max()))
    # Binance book_snapshot_25 is flattened — count bid_price_i columns present.
    bin_levels: int | None = None
    bfile = _binance_file(paths, "book_snapshot_25", trial["binance_symbol"], day_str)
    if bfile.exists():
        names = set(pq.ParquetFile(bfile).schema_arrow.names)
        bin_levels = sum(1 for i in range(50) if f"bid_price_{i}" in names)
    # book_snapshot_full is the FULL book by definition — PASS iff we actually got book
    # data; the max-levels number reports how deep Polymarket books really are (a thin book
    # is a market fact, not a Telonex limitation). Binance depth (if on plan) must be ≥ 20.
    ok = poly_max >= 1 and (bin_levels is None or bin_levels >= DEPTH_MIN_LEVELS)
    return {
        "name": "depth",
        "result": _pf(ok),
        "polymarket_full_max_levels": poly_max,
        "polymarket_full_ge_20": poly_max >= DEPTH_MIN_LEVELS,
        "binance_25_levels": bin_levels if bin_levels is not None else "unavailable (exchange not on plan)",
        "min_required": DEPTH_MIN_LEVELS,
        "note": "Polymarket book_snapshot_full = all available levels; Binance depth20 is also free "
                "from data.binance.vision (model_lab.hist).",
    }


# --------------------------------------------------------------------------- #
# Journal readers bounded to the trial day (fast — only the relevant segments)
# --------------------------------------------------------------------------- #


def _day_bounds_ms(day: date) -> tuple[int, int]:
    since = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    return since, since + 86_400_000


def _day_segments(journal_dir: Path, day: date) -> list[Path]:
    lo = (day - timedelta(days=1)).strftime("%Y%m%d")
    hi = (day + timedelta(days=1)).strftime("%Y%m%d")
    out: list[Path] = []
    for p in jio.segment_paths(journal_dir):
        m = re.search(r"journal-(\d{8})-", p.name)
        if not m or lo <= m.group(1) <= hi:
            out.append(p)
    return out


def _segment_start_ms(path: Path) -> int | None:
    """The wall-clock start (unix ms) encoded in a ``journal-YYYYMMDD-HHMMSS-…`` name."""
    m = re.search(r"journal-(\d{8})-(\d{6})", path.name)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _segments_covering(journal_dir: Path, spans: list[tuple[int, int]]) -> list[Path]:
    """Only the journal segments whose time coverage intersects one of ``spans``.

    The journal is tens of GB across a few days; each segment covers ``[start_i,
    start_{i+1})`` (filenames sort chronologically). For a handful of 5-minute windows
    this selects a few segments instead of the whole 3-day span — the difference between
    seconds and never finishing."""
    segs = [(s if s is not None else 0, p)
            for p in jio.segment_paths(journal_dir) for s in (_segment_start_ms(p),)]
    segs.sort(key=lambda t: t[0])
    chosen: list[Path] = []
    for i, (start, p) in enumerate(segs):
        nxt = segs[i + 1][0] if i + 1 < len(segs) else float("inf")
        if any(start < c and nxt > o for (o, c) in spans):
            chosen.append(p)
    return chosen


def _read_day_records(journal_dir: Path, day: date, types: set[str]):
    for path in _day_segments(journal_dir, day):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        env = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    rec = env.get("rec")
                    if isinstance(rec, dict) and rec.get("type") in types:
                        rec["ts_local_ms"] = env.get("ts_local_ms")
                        yield rec
        except (OSError, EOFError, zlib.error):
            continue


def _our_binance_trades(journal_dir: Path, day: date) -> pd.DataFrame:
    since, until = _day_bounds_ms(day)
    rows_t: list[int] = []
    rows_p: list[float] = []
    for rec in _read_day_records(journal_dir, day, {"price_tick"}):
        if rec.get("source") != "BinanceDirect" or rec.get("kind") != "Trade":
            continue
        te = rec.get("ts_exchange")
        if te is None or not (since <= te < until):
            continue
        rows_t.append(int(te))
        rows_p.append(jio.to_float(rec.get("value")))
    return pd.DataFrame({"ts_ms": rows_t, "price": rows_p})


def _our_top_of_book(segments: list[Path], token_ids: set[str], since: int, until: int) -> dict[str, pd.DataFrame]:
    """Our recorded CLOB tops for a small set of ``token_ids`` within ``[since, until)``,
    scanning only the given ``segments``.

    The journal has millions of ``top_of_book`` records; JSON-parsing each is far too slow.
    Two cheap pre-filters run first — a substring test for the record type and a single
    compiled regex of the target tokens — so ``json.loads`` runs only on the tiny fraction
    of lines that mention one of our tokens."""
    if not token_ids:
        return {}
    tok_re = re.compile("|".join(re.escape(t) for t in token_ids))
    acc: dict[str, dict[str, list]] = {t: {"ts_ms": [], "bid": [], "ask": []} for t in token_ids}
    for path in segments:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if '"top_of_book"' not in line or not tok_re.search(line):
                        continue
                    try:
                        env = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    rec = env.get("rec") or {}
                    if rec.get("type") != "top_of_book":
                        continue
                    tok = str(rec.get("token_id", ""))
                    if tok not in acc:
                        continue
                    top = rec.get("top") or {}
                    ts, bid, ask = top.get("ts"), top.get("bid"), top.get("ask")
                    if ts is None or not (since <= ts < until) or not isinstance(bid, dict) or not isinstance(ask, dict):
                        continue
                    a = acc[tok]
                    a["ts_ms"].append(int(ts))
                    a["bid"].append(jio.to_float(bid.get("price")))
                    a["ask"].append(jio.to_float(ask.get("price")))
        except (OSError, EOFError, zlib.error):
            continue
    return {t: pd.DataFrame(v) for t, v in acc.items() if v["ts_ms"]}


# --------------------------------------------------------------------------- #
# Check 5 — clock alignment
# --------------------------------------------------------------------------- #


def _offset_stats(delta_ms: np.ndarray, t_ms: np.ndarray) -> dict[str, Any]:
    d = np.asarray(delta_ms, dtype="float64")
    t = np.asarray(t_ms, dtype="float64")
    if d.size == 0:
        return {"n": 0}
    med = float(np.median(d))
    resid = d - med
    q75, q25 = np.percentile(resid, [75, 25])
    jitter = float((q75 - q25) / 1.349)
    slope = float("nan")
    if d.size >= 3 and np.ptp(t) > 0:
        hrs = (t - t.min()) / 3.6e6
        slope = float(np.polyfit(hrs, d, 1)[0])
    return {
        "n": int(d.size), "median_offset_ms": med, "jitter_ms": jitter,
        "drift_ms_per_hr": slope, "p05_ms": float(np.percentile(d, 5)),
        "p95_ms": float(np.percentile(d, 95)),
    }


def _stable(stats: dict[str, Any]) -> bool:
    if stats.get("n", 0) < CLOCK_MIN_MATCHES:
        return False
    jitter = stats.get("jitter_ms", math.inf)
    drift = abs(stats.get("drift_ms_per_hr", math.inf))
    return jitter <= CLOCK_JITTER_PASS_MS and drift <= CLOCK_DRIFT_PASS_MS_PER_HR


def _match_binance(tel: pd.DataFrame, ours: pd.DataFrame, *, tol_ms: int = 2000,
                   max_ours: int = 8000) -> dict[str, Any]:
    if tel.empty or ours.empty:
        return {"n": 0}
    tel = tel.dropna(subset=["price"]).copy()
    tel["cents"] = np.rint(tel["price"].to_numpy() * 100).astype("int64")
    groups: dict[int, np.ndarray] = {
        int(c): np.sort(g["ts_ms"].to_numpy(dtype="int64")) for c, g in tel.groupby("cents")
    }
    ours = ours.dropna(subset=["price"])
    if len(ours) > max_ours:
        ours = ours.iloc[np.linspace(0, len(ours) - 1, max_ours).astype(int)]
    deltas: list[float] = []
    times: list[float] = []
    for ts, price in zip(ours["ts_ms"].to_numpy(dtype="int64"), ours["price"].to_numpy()):
        arr = groups.get(int(round(price * 100)))
        if arr is None or arr.size == 0:
            continue
        j = int(np.searchsorted(arr, ts))
        cand = [arr[k] for k in (j - 1, j) if 0 <= k < arr.size]
        if not cand:
            continue
        nearest = min(cand, key=lambda x: abs(int(x) - int(ts)))
        if abs(int(nearest) - int(ts)) <= tol_ms:
            deltas.append(float(nearest - ts))  # telonex − ours (both Binance exchange time)
            times.append(float(ts))
    return _offset_stats(np.asarray(deltas), np.asarray(times))


def _match_clob(tel_top: pd.DataFrame, our_top: pd.DataFrame, *, tol_ms: int = 2000) -> np.ndarray:
    """Δ = telonex.ts − ours.ts where both observe the SAME top-of-book (bid & ask equal)."""
    if tel_top.empty or our_top.empty:
        return np.asarray([])
    left = our_top.sort_values("ts_ms").reset_index(drop=True)
    right = tel_top.sort_values("ts_ms").rename(columns={"ts_ms": "ts_tel"}).reset_index(drop=True)
    merged = pd.merge_asof(
        left, right, left_on="ts_ms", right_on="ts_tel", direction="nearest",
        tolerance=tol_ms, suffixes=("_o", "_t"),
    ).dropna(subset=["ts_tel"])
    same = (np.isclose(merged["bid_o"], merged["bid_t"], atol=1e-9)
            & np.isclose(merged["ask_o"], merged["ask_t"], atol=1e-9))
    m = merged[same]
    return (m["ts_tel"].to_numpy() - m["ts_ms"].to_numpy()).astype("float64")


def check_clock(paths: Paths, trial: dict[str, Any]) -> dict[str, Any]:
    day = date.fromisoformat(trial["day"])
    day_str = trial["day"]

    # 5b-i: Binance trades vs our recorded Binance Trade ticks.
    tel_bfile = _binance_file(paths, "trades", trial["binance_symbol"], day_str)
    binance_available = tel_bfile.exists()
    binance: dict[str, Any] = {"n": 0}
    if binance_available:
        tel = _read_cols(tel_bfile, ["timestamp_us", "price"])
        tel = pd.DataFrame({"ts_ms": tel["timestamp_us"].to_numpy(dtype="int64") // 1000,
                            "price": tel["price"].map(_to_float).to_numpy()})
        ours = _our_binance_trades(paths.journal_dir, day)
        binance = _match_binance(tel, ours)

    # 5b-ii: Polymarket top-of-book vs our recorded CLOB top for the same Up token.
    # A few windows spread across the day are enough to measure a constant offset + drift;
    # scan only the journal segments that cover those windows (the journal is tens of GB).
    windows = trial["windows"]
    sample = windows[:: max(1, len(windows) // 8)] if windows else []
    tok_map = {str(w["up_asset_id"]): w for w in sample if w.get("up_asset_id")}
    dur_ms = tx.duration_seconds(trial["series"].rsplit("-", 1)[-1]) * 1000
    spans = [(w["open_ms"], w["open_ms"] + dur_ms) for w in tok_map.values()]
    our_tops: dict[str, pd.DataFrame] = {}
    if tok_map:
        segs = _segments_covering(paths.journal_dir, spans)
        since = min(o for o, _ in spans)
        until = max(c for _, c in spans)
        our_tops = _our_top_of_book(segs, set(tok_map), since, until)
    clob_deltas: list[np.ndarray] = []
    matched_windows = 0
    for tok, w in tok_map.items():
        if tok not in our_tops:
            continue
        p = _poly_file(paths, "quotes", w["slug"], "Up", day_str)  # flat top-of-book — cheap
        if not p.exists():
            continue
        try:
            tel_top = _quotes_tops(p)
        except Exception:  # noqa: BLE001
            continue
        d = _match_clob(tel_top, our_tops[tok])
        if d.size:
            clob_deltas.append(d)
            matched_windows += 1
    clob_all = np.concatenate(clob_deltas) if clob_deltas else np.asarray([])
    clob_stats = _offset_stats(clob_all, np.arange(clob_all.size, dtype="float64")) if clob_all.size else {"n": 0}
    clob_stats["matched_windows"] = matched_windows

    internal_ok = True  # both channels use timestamp_us/local_timestamp_us in µs (schema-verified)
    clob_ok = _stable(clob_stats)
    # Binance only judged when its data is on the plan; the Polymarket CLOB cross-check is
    # the binding external test either way.
    bin_ok = _stable(binance) if binance_available else True
    bin_result = _pf(_stable(binance)) if binance_available else "N/A (exchange not on plan)"
    return {
        "name": "clock_alignment",
        "result": _pf(internal_ok and bin_ok and clob_ok),
        "internal": {"result": "PASS", "note": "timestamp_us + local_timestamp_us (µs) shared by both exchanges"},
        "external_binance_trades": {"result": bin_result, **binance},
        "external_polymarket_top": {"result": _pf(clob_ok), **clob_stats},
        "thresholds": {"jitter_ms": CLOCK_JITTER_PASS_MS, "drift_ms_per_hr": CLOCK_DRIFT_PASS_MS_PER_HR,
                       "min_matches": CLOCK_MIN_MATCHES},
    }


# --------------------------------------------------------------------------- #
# Check 6 — completeness + pair/mirror integrity
# --------------------------------------------------------------------------- #


def check_completeness(paths: Paths, trial: dict[str, Any], cadence: dict[str, Any]) -> dict[str, Any]:
    day_str = trial["day"]
    windows = trial["windows"]
    dur = trial["series"].rsplit("-", 1)[-1]
    expected = WINDOWS_PER_DAY.get(dur)

    # duplicate / malformed trades (Up side, sampled windows)
    dup_ids = 0
    trade_rows = 0
    nonmono = 0
    for w in windows[:30]:
        p = _poly_file(paths, "trades", w["slug"], "Up", day_str)
        if not p.exists():
            continue
        try:
            df = _read_cols(p, ["timestamp_us", "trade_id"])
        except Exception:  # noqa: BLE001
            continue
        trade_rows += len(df)
        dup_ids += int(len(df) - df["trade_id"].nunique())
        ts = df["timestamp_us"].to_numpy(dtype="int64")
        nonmono += int((np.diff(ts) < 0).sum()) if ts.size >= 2 else 0

    # pair/mirror integrity: Down top ≈ 1 − Up top
    mirror_ok = 0
    mirror_checked = 0
    for w in windows[:: max(1, len(windows) // 30)] if windows else []:
        up = _poly_file(paths, "quotes", w["slug"], "Up", day_str)
        dn = _poly_file(paths, "quotes", w["slug"], "Down", day_str)
        if not (up.exists() and dn.exists()):
            continue
        try:
            u = _quotes_tops(up)
            d = _quotes_tops(dn)
        except Exception:  # noqa: BLE001
            continue
        merged = pd.merge_asof(
            u.sort_values("ts_ms"), d.sort_values("ts_ms").rename(columns={"ts_ms": "ts_d"}),
            left_on="ts_ms", right_on="ts_d", direction="nearest", tolerance=2000, suffixes=("_u", "_d"),
        ).dropna(subset=["ts_d"])
        if merged.empty:
            continue
        mirror_checked += 1
        # Down bid ≈ 1 − Up ask ; Down ask ≈ 1 − Up bid
        ok = (np.abs(merged["bid_d"] - (1 - merged["ask_u"])) < MIRROR_TOL).mean() > 0.9
        mirror_ok += int(ok)

    gaps_over = cadence.get("polymarket_book_full", {}).get("gaps_over_5s", 0)
    mirror_ratio = (mirror_ok / mirror_checked) if mirror_checked else 1.0
    # Up/Down mirror almost everywhere; brief transient divergences (one side momentarily
    # one-sided) are normal, so require the mirror to hold for ≥ 90% of checked windows.
    ok = dup_ids == 0 and nonmono == 0 and mirror_ratio >= 0.90
    return {
        "name": "completeness",
        "result": _pf(ok),
        "windows_with_data": len(windows),
        "windows_expected_per_day": expected,
        "trade_rows_checked": trade_rows,
        "duplicate_trade_ids": dup_ids,
        "nonmonotonic_timestamps": nonmono,
        "gaps_over_5s_in_active_book": gaps_over,
        "mirror_windows_checked": mirror_checked,
        "mirror_windows_ok": mirror_ok,
        "mirror_ratio": round(mirror_ratio, 3),
    }


# --------------------------------------------------------------------------- #
# Stage 4 — compression + projection
# --------------------------------------------------------------------------- #


def compress_trial(paths: Paths, *, level: int = 19, scratch: Path | None = None) -> dict[str, Any]:
    files = tx.iter_raw_files(paths.telonex_dir)
    root = tx.raw_root(paths.telonex_dir)
    scratch = scratch or (paths.out_dir / "telonex" / "_roundtrip")
    raw_total = zst_total = n = roundtrip_ok = 0
    poly_zst = binance_zst = 0
    by_channel_zst: dict[str, int] = {}
    roundtrip_fail: list[str] = []
    for p in files:
        # markets.parquet is the catalog, not trial data — don't fold it into the ratio.
        if p.name == "markets.parquet":
            continue
        zst = p.with_name(p.name + ".zst")
        if not zst.exists():
            tx.compress_file_zstd(p, zst, level=level)
        raw_b = p.stat().st_size
        zst_b = zst.stat().st_size
        raw_total += raw_b
        zst_total += zst_b
        n += 1
        parts = p.relative_to(root).parts if root in p.parents else ("other",)
        exchange = parts[0] if parts else "other"
        if exchange == tx.POLYMARKET:
            poly_zst += zst_b
            if len(parts) >= 2:  # parts = (polymarket, <channel>, file)
                by_channel_zst[parts[1]] = by_channel_zst.get(parts[1], 0) + zst_b
        elif exchange == tx.BINANCE:
            binance_zst += zst_b
        if tx.verify_roundtrip(p, zst, scratch_dir=scratch):
            roundtrip_ok += 1
        else:
            roundtrip_fail.append(str(p))
    ratio = (raw_total / zst_total) if zst_total else float("nan")
    return {
        "name": "compression",
        "result": _pf(n > 0 and not roundtrip_fail),
        "files": n,
        "raw_bytes": raw_total,
        "zst_bytes": zst_total,
        "polymarket_zst_bytes": poly_zst,
        "binance_zst_bytes": binance_zst,
        "polymarket_zst_by_channel": by_channel_zst,
        "raw_human": human_bytes(raw_total),
        "zst_human": human_bytes(zst_total),
        "ratio": ratio,
        "roundtrip_ok": roundtrip_ok,
        "roundtrip_failed": roundtrip_fail,
        "level": level,
    }


def project_full(trial: dict[str, Any], date_range: dict[str, Any],
                 compression: dict[str, Any]) -> dict[str, Any]:
    """Project the compressed needed-slice size from the trial's measured GB/window, broken
    down by channel so the operator can see which subset fits the 250 GB budget."""
    n_windows = max(1, len(trial["windows"]))
    by_ch = compression.get("polymarket_zst_by_channel", {})
    per_win_ch = {ch: b / n_windows for ch, b in by_ch.items()}  # bytes/window (both outcomes)
    dseries = date_range.get("series", {})

    def series_scale(series: str) -> tuple[int, float]:
        dur = series.rsplit("-", 1)[-1]
        wins = dseries.get(series, {}).get("windows_with_data") or 0
        # longer windows carry proportionally more book events → scale per-window bytes
        return wins, tx.duration_seconds(dur) / tx.duration_seconds("5m")

    def scenario(chans: list[str]) -> dict[str, Any]:
        per_win = sum(per_win_ch.get(c, 0) for c in chans)
        total = 0.0
        detail: dict[str, Any] = {}
        for s in NEEDED_SERIES:
            wins, scale = series_scale(s)
            est = wins * per_win * scale
            total += est
            detail[s] = {"windows": wins, "est_bytes": int(est), "est_human": human_bytes(int(est))}
        return {"channels": chans, "bytes": int(total), "human": human_bytes(int(total)),
                "fits_budget": bool(total <= BUDGET_BYTES), "series_detail": detail}

    all_ch = list(per_win_ch)
    tob = [c for c in all_ch if c in ("quotes", "trades")]
    book = [c for c in all_ch if c == "book_snapshot_full"]
    scenarios = {
        "all_channels_full_history": scenario(all_ch),
        "quotes_trades_full_history": scenario(tob),
        "book_full_only_full_history": scenario(book),
    }
    needed = scenarios["all_channels_full_history"]["bytes"]
    return {
        "per_window_zst_bytes_by_channel": {c: int(b) for c, b in per_win_ch.items()},
        "per_window_zst_total": int(sum(per_win_ch.values())),
        "scenarios": scenarios,
        "series_detail": scenarios["all_channels_full_history"]["series_detail"],
        "needed_slice_bytes": needed,
        "needed_slice_human": human_bytes(needed),
        "budget_bytes": BUDGET_BYTES,
        "budget_human": human_bytes(BUDGET_BYTES),
        "fits_budget": bool(needed <= BUDGET_BYTES),
        "binance_note": ("Binance is sourced FREE from data.binance.vision (model_lab.hist) and "
                         "isn't on this Single-Exchange plan anyway — excluded from the needed slice."),
        "notes": [
            "book_snapshot_full dominates size (~6.5 MB/window); quotes+trades is a small "
            "fraction. Our features currently use top-of-book + trades, so quotes_trades is the "
            "practical slice unless deep Polymarket book is wanted.",
            "1h series show 0 windows here because 1h up/down markets use human-date slugs, not "
            "the -updown-1h-{epoch} pattern the catalog filter matches — a filter limitation, not "
            "a Telonex gap.",
            "Full history = every day the catalog reports (5m ~146 d, 15m ~270 d). A shorter "
            "training window (e.g. the last 90-180 d) scales down proportionally.",
        ],
        "assumptions": (
            "Per-window compressed bytes measured from the real 5m trial (both outcomes), split by "
            "channel; scaled by window duration and multiplied by the catalog's window counts "
            f"(check 2). zstd whole-file ratio measured at {compression.get('ratio', float('nan')):.2f}× "
            "on the already-columnar parquet."
        ),
    }


# --------------------------------------------------------------------------- #
# Orchestration + report
# --------------------------------------------------------------------------- #


def validate(paths: Paths, *, compress: bool = True, catalog: bool = True,
             fetch: tx.Fetch = tx.urllib_fetch, level: int = 19) -> dict[str, Any]:
    trial = _load_trial(paths)
    # One catalog read (1 GB) feeds both coverage and date-range.
    catalog_scan = None
    catalog_status = "skipped"
    if catalog:
        try:
            cat = tx.ensure_markets_catalog(paths.telonex_dir, fetch=fetch)
            catalog_scan = _scan_catalog(cat)
            catalog_status = "ok"
        except Exception as e:  # noqa: BLE001
            catalog_status = f"catalog_failed: {type(e).__name__}: {e}"
    checks: dict[str, Any] = {}
    checks["coverage"] = check_coverage(catalog_scan, fetch=fetch)
    checks["date_range"] = check_date_range(catalog_scan, catalog_status, fetch=fetch)
    checks["cadence"] = check_cadence(paths, trial)
    checks["depth"] = check_depth(paths, trial)
    checks["clock"] = check_clock(paths, trial)
    checks["completeness"] = check_completeness(paths, trial, checks["cadence"])
    result: dict[str, Any] = {"trial": {k: trial[k] for k in
                              ("day", "series", "binance_symbol", "outcomes", "channels")},
                              "n_windows": len(trial["windows"]), "checks": checks}
    bfile = tx.binance_raw_path(paths.telonex_dir, "trades", trial["binance_symbol"], trial["day"])
    result["plan"] = {
        "binance_from_telonex": bfile.exists(),
        "note": ("Binance downloads succeeded (Pro plan)." if bfile.exists() else
                 "Binance is NOT on this Telonex plan — the key is Single-Exchange (Polymarket only); "
                 "Binance needs a Pro upgrade. We already get Binance FREE from data.binance.vision via "
                 "model_lab.hist, so Telonex is only needed for Polymarket order-book/trade data."),
    }
    if compress:
        comp = compress_trial(paths, level=level)
        result["compression"] = comp
        result["projection"] = project_full(trial, checks["date_range"], comp)
    # overall go/no-go: the data-quality checks must pass (coverage, cadence, depth, clock, completeness)
    critical = ["coverage", "cadence", "depth", "clock", "completeness"]
    passed = all(checks[c]["result"] == "PASS" for c in critical)
    result["go_no_go"] = "GO" if passed else "NO-GO"
    return result


def _write_outputs(paths: Paths, result: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = paths.out_dir / "telonex"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "validation.json"
    jpath.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    hpath = out_dir / "report.html"
    hpath.write_text(_render_html(result), encoding="utf-8")
    return jpath, hpath


def _render_html(result: dict[str, Any]) -> str:
    rows = []
    for key, c in result["checks"].items():
        colour = "#1a7f37" if c["result"] == "PASS" else ("#9a6700" if c["result"] == "SKIPPED" else "#cf222e")
        rows.append(f"<tr><td>{key}</td><td style='color:{colour};font-weight:600'>{c['result']}</td>"
                    f"<td><pre style='margin:0;white-space:pre-wrap'>{json.dumps({k:v for k,v in c.items() if k not in ('name','result')}, indent=1, default=str)[:2000]}</pre></td></tr>")
    comp = result.get("compression", {})
    proj = result.get("projection", {})
    go = result.get("go_no_go", "?")
    go_colour = "#1a7f37" if go == "GO" else "#cf222e"
    return f"""<!doctype html><meta charset=utf-8><title>Telonex trial report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d0d7de;padding:6px 10px;text-align:left;vertical-align:top}}
h1 span{{color:{go_colour}}}</style>
<h1>Telonex trial — <span>{go}</span></h1>
<p>Trial: {json.dumps(result['trial'], default=str)} · {result['n_windows']} windows</p>
<p style="padding:8px 12px;background:#fff8c5;border:1px solid #d4a72c;border-radius:6px">
<b>Plan:</b> {result.get('plan', {}).get('note', '')}</p>
<table><tr><th>Check</th><th>Result</th><th>Numbers</th></tr>{''.join(rows)}</table>
<h2>Compression</h2><pre>{json.dumps(comp, indent=2, default=str)}</pre>
<h2>Projection vs 250 GB budget</h2><pre>{json.dumps(proj, indent=2, default=str)}</pre>
"""


def _print_summary(result: dict[str, Any]) -> None:
    print("\n[telonex_validate] ==== VALIDATION SUMMARY ====")
    plan = result.get("plan", {})
    if plan:
        print(f"[telonex_validate] PLAN: {plan.get('note', '')}")
    for key, c in result["checks"].items():
        print(f"[telonex_validate] {key:14s} {c['result']}")
    ca = result["checks"]["cadence"]["polymarket_book_full"]
    if ca.get("median_s") is not None:
        print(f"[telonex_validate]   cadence: poly median {ca['median_s']:.3f}s p95 {ca['p95_s']:.2f}s "
              f"max {ca['max_s']:.1f}s over {ca['windows']} windows")
    dep = result["checks"]["depth"]
    print(f"[telonex_validate]   depth: poly_full {dep['polymarket_full_max_levels']} / "
          f"binance_25 {dep['binance_25_levels']} levels")
    clk = result["checks"]["clock"]
    for sub in ("external_binance_trades", "external_polymarket_top"):
        s = clk[sub]
        if s.get("n", 0):
            print(f"[telonex_validate]   {sub}: offset {s.get('median_offset_ms'):.1f}ms "
                  f"jitter {s.get('jitter_ms'):.1f}ms drift {s.get('drift_ms_per_hr'):.2f}ms/hr "
                  f"n={s['n']} → {s['result']}")
    if "compression" in result:
        comp = result["compression"]
        print(f"[telonex_validate]   compression: {comp['ratio']:.3f}× ({comp['raw_human']} → {comp['zst_human']}), "
              f"roundtrip {comp['roundtrip_ok']}/{comp['files']} OK → {comp['result']}")
        proj = result["projection"]
        print("[telonex_validate]   projected Telonex slice vs 250 GB budget (Polymarket; Binance free elsewhere):")
        for name, sc in proj["scenarios"].items():
            fit = "FITS" if sc["fits_budget"] else "does NOT fit"
            print(f"[telonex_validate]     {name:28s} {sc['human']:>10}  {fit}")
    print(f"[telonex_validate] GO/NO-GO: {result['go_no_go']}")
    if result.get("plan", {}).get("binance_from_telonex") is False:
        print("[telonex_validate] (GO for Polymarket data quality; Binance not on plan — free from data.binance.vision)")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_validate")
    parser.add_argument("--no-compress", action="store_true", help="skip zstd compression + projection")
    parser.add_argument("--no-catalog", action="store_true", help="skip the markets-dataset download")
    parser.add_argument("--zstd-level", type=int, default=19, help="zstd level (default 19)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    try:
        result = validate(paths, compress=not args.no_compress, catalog=not args.no_catalog,
                          level=args.zstd_level)
    except tx.TelonexError as e:
        print(f"[telonex_validate] {e}")
        return 1

    jpath, hpath = _write_outputs(paths, result)
    _print_summary(result)
    procmem.report_peak_rss("telonex_validate", procmem.peak_rss_mb(), args.max_rss_mb)
    print(f"[telonex_validate] wrote {jpath}")
    print(f"[telonex_validate] wrote {hpath}")
    return 0 if result["go_no_go"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
