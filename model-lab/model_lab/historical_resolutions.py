"""Official window resolutions (Part A.2) — the authoritative Up/Down label.

Polymarket resolves every window on-chain; the Telonex **markets catalog**
(``data/telonex/raw/polymarket/markets.parquet``) mirrors that resolution as
``result_id`` (``"0"`` ⇒ ``outcome_0``=Up won, ``"1"`` ⇒ ``outcome_1``=Down won) with
``status=resolved`` + ``settled_at_us``. This is the **official** label (validated 0-mismatch
against the Polymarket CLOB API, below), available for ~99.5% of our windows offline — far
higher coverage than a 156k-call API fetch (the CLOB API 404s on ~12% of old markets).

This module:

- builds a compact **resolution cache** ``data/resolutions/catalog_resolutions.parquet``
  (`(series, open_ms) → {outcome, condition_id, status}`) that ``historical_labels`` consumes;
- **validates the catalog against the CLOB API** over a *stratified* sample (≥ 2000 windows
  covering every series, every month, and knife-edge windows specifically) — **any
  catalog-vs-API mismatch is a hard failure**, not a warning; the API results are cached to
  ``data/resolutions/clob_cache.jsonl`` (resumable, rate-limited);
- **validates the catalog against the journal** on overlap days (must match the recorded
  ``Resolved`` outcome exactly — hard fail otherwise).

Run::

    python -m model_lab.historical_resolutions              # build cache + full validation
    python -m model_lab.historical_resolutions --min-sample 3000
    python -m model_lab.historical_resolutions --no-api     # catalog cache + journal check only
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import historical_common as hc
from .config import Paths, resolve_paths, stage_parser
from .io import polymarket as pm
from .lib import procmem

KNIFE_VOL_THRESH = 0.25       # |end − strike| below this (vol units) = knife-edge
STRIKE_TOL_MS = 2500.0
DEFAULT_MIN_SAMPLE = 2000
UNIFORM_PER_STRATUM = 80      # uniform windows kept per (series, month); ALL knife-edge kept too
API_MIN_INTERVAL_S = 0.08     # ~12 req/s — polite
_SLUG_RE = re.compile(r"^((?:btc|eth)-updown-(?:5m|15m))-(\d+)$")


# --- paths ------------------------------------------------------------------
def resolutions_dir(paths: Paths) -> Path:
    return paths.telonex_dir.parent / "resolutions"


def catalog_cache_path(paths: Paths) -> Path:
    return resolutions_dir(paths) / "catalog_resolutions.parquet"


def api_cache_path(paths: Paths) -> Path:
    return resolutions_dir(paths) / "clob_cache.jsonl"


def _catalog_path(paths: Paths) -> Path:
    return hc.tx.raw_root(paths.telonex_dir) / hc.tx.POLYMARKET / "markets.parquet"


# --- catalog (official resolution) ------------------------------------------
_CATALOG_OUT_COLS = ["series", "window_open_ms", "condition_id", "outcome", "status", "settled_at_us"]


def _read_catalog(paths: Paths, series: tuple[str, ...], *, resolved_only: bool = True) -> pd.DataFrame:
    """Read the Telonex markets catalog, filter to our series, decode the winner from
    ``result_id``. Returns ``series, window_open_ms, condition_id, outcome, status,
    settled_at_us`` (``outcome`` is ``None`` for non-resolved windows). With
    ``resolved_only`` (default) only resolved windows with a decoded winner are returned;
    ``resolved_only=False`` returns the full window universe (incl. ``active``). Read in
    row-group batches (the catalog is ~1 GB)."""
    prefixes = {hc.series_prefix(s) for s in series if hc.series_prefix(s)}
    cat = _catalog_path(paths)
    if not cat.exists():
        return pd.DataFrame(columns=_CATALOG_OUT_COLS)
    cols = ["slug", "market_id", "status", "result_id", "outcome_0", "outcome_1", "settled_at_us"]
    keep: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(cat).iter_batches(batch_size=250_000, columns=cols):
        df = batch.to_pandas()
        m = df["slug"].astype(str).str.match(_SLUG_RE)
        df = df[m]
        if df.empty:
            continue
        parts = df["slug"].astype(str).str.extract(_SLUG_RE)
        df = df.assign(_prefix=parts[0], _epoch=parts[1])
        df = df[df["_prefix"].isin(prefixes)]
        keep.append(df)
    if not keep:
        return pd.DataFrame(columns=_CATALOG_OUT_COLS)
    d = pd.concat(keep, ignore_index=True).drop_duplicates("slug")

    def winner(row):
        rid = str(row["result_id"])
        if row["status"] != "resolved":
            return None
        if rid == "0":
            return row["outcome_0"]
        if rid == "1":
            return row["outcome_1"]
        return None

    d["outcome"] = d.apply(winner, axis=1)
    if resolved_only:
        d = d[d["outcome"].isin(("Up", "Down"))]
    d["series"] = d["_prefix"].map({v: k for k, v in {s: hc.series_prefix(s) for s in series}.items()})
    d["window_open_ms"] = d["_epoch"].astype("int64") * 1000
    d = d.rename(columns={"market_id": "condition_id"})
    return d[_CATALOG_OUT_COLS].reset_index(drop=True)


def catalog_windows(paths: Paths, series: tuple[str, ...] | None = None,
                    since_ms: int | None = None, until_ms: int | None = None) -> pd.DataFrame:
    """The full window universe from the catalog (resolved + active), the authoritative
    per-series window list — far faster than globbing the 261k quote files. Bounded by
    ``[since_ms, until_ms)`` on ``window_open_ms``. Cached to
    ``data/resolutions/catalog_windows.parquet``."""
    series = tuple(series) if series else hc.DEFAULT_SERIES
    dest = resolutions_dir(paths) / "catalog_windows.parquet"
    if dest.exists():
        df = pd.read_parquet(dest)
    else:
        df = _read_catalog(paths, series, resolved_only=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, engine="pyarrow", index=False)
    if series:
        df = df[df["series"].isin(series)]
    if since_ms is not None:
        df = df[df["window_open_ms"] >= since_ms]
    if until_ms is not None:
        df = df[df["window_open_ms"] < until_ms]
    return df.reset_index(drop=True)


def build_catalog_cache(paths: Paths, series: tuple[str, ...] | None = None) -> dict:
    """Build ``data/resolutions/catalog_resolutions.parquet`` from the markets catalog
    (once; ``historical_labels`` reads it, not the 1 GB catalog). Returns counts."""
    series = tuple(series) if series else hc.DEFAULT_SERIES
    df = _read_catalog(paths, series)
    dest = catalog_cache_path(paths)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, engine="pyarrow", index=False)
    per_series = {s: int((df["series"] == s).sum()) for s in series}
    return {"resolved_windows": int(len(df)), "per_series": per_series, "path": str(dest)}


def load_resolution_map(paths: Paths, series: tuple[str, ...] | None = None) -> dict[tuple[str, int], dict]:
    """``{(series, window_open_ms): {"outcome", "condition_id"}}`` from the catalog cache
    (built on demand). The official resolution source ``historical_labels`` consumes."""
    dest = catalog_cache_path(paths)
    if not dest.exists():
        build_catalog_cache(paths, series)
    if not dest.exists():
        return {}
    df = pd.read_parquet(dest)
    if series:
        df = df[df["series"].isin(series)]
    return {(r.series, int(r.window_open_ms)): {"outcome": r.outcome, "condition_id": r.condition_id}
            for r in df.itertuples(index=False)}


# --- API cache (resumable) --------------------------------------------------
def _load_api_cache(paths: Paths) -> dict[str, dict]:
    p = api_cache_path(paths)
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("condition_id")
            if cid:
                out[cid] = rec
    return out


def _append_api_cache(paths: Paths, records: list[dict]) -> None:
    p = api_cache_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last = 0.0

    def wait(self, now: float) -> None:
        dt = now - self.last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self.last = time.monotonic()


# --- stratified validation sample -------------------------------------------
def _month(open_ms: int) -> str:
    dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def stratified_sample(
    catalog: pd.DataFrame, paths: Paths, *, min_sample: int = DEFAULT_MIN_SAMPLE,
    uniform_per_stratum: int = UNIFORM_PER_STRATUM, knife_vol: float = KNIFE_VOL_THRESH,
    rng_seed: int = 20260709,
) -> pd.DataFrame:
    """A validation sample covering **every (series, month)** and **knife-edge windows
    specifically**: for each stratum pick representative day(s), compute the Binance-anchored
    ``|end−strike|`` vol distance for their windows (loading each day's aggTrades once), keep
    up to ``uniform_per_stratum`` uniform windows + **all** knife-edge (vol < ``knife_vol``),
    growing until ``>= min_sample``. Returns ``series, window_open_ms, condition_id, outcome,
    month, day, vol_dist, knife_edge``."""
    rng = np.random.default_rng(rng_seed)
    cat = catalog.copy()
    cat["month"] = cat["window_open_ms"].map(_month)
    cat["day"] = cat["window_open_ms"].map(lambda ms: hc.utc_day(int(ms)).isoformat())
    strata = sorted(cat.groupby(["series", "month"]).groups.keys())
    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def day_bars(series: str, day_iso: str) -> pd.DataFrame:
        key = (series, day_iso)
        if key not in bars_cache:
            asset = hc.SLUG_PREFIXES[hc.series_prefix(series)][1]
            bars_cache[key] = hc.binance_bars_for_day(paths, hc.SYMBOL_BY_ASSET[asset], date.fromisoformat(day_iso))
        return bars_cache[key]

    rows: list[dict] = []
    # rotate the day chosen per stratum so multiple passes pick different days
    day_round = 0
    while True:
        added = 0
        for (series, month) in strata:
            g = cat[(cat["series"] == series) & (cat["month"] == month)]
            days = sorted(g["day"].unique())
            if not days:
                continue
            day_iso = days[(len(days) // 2 + day_round) % len(days)]
            wins = g[g["day"] == day_iso]
            if wins.empty:
                continue
            bars = day_bars(series, day_iso)
            recs: list[dict] = []
            for w in wins.itertuples(index=False):
                open_ms = int(w.window_open_ms)
                br = hc.binance_outcome(bars, open_ms, open_ms + hc.SLUG_PREFIXES[hc.series_prefix(series)][2] * 1000, STRIKE_TOL_MS)
                vd = br["strike_distance_at_close_vol"]
                recs.append({"series": series, "window_open_ms": open_ms, "condition_id": w.condition_id,
                             "outcome": w.outcome, "month": month, "day": day_iso,
                             "vol_dist": vd, "knife_edge": bool(vd is not None and np.isfinite(vd) and vd < knife_vol)})
            knife = [r for r in recs if r["knife_edge"]]
            nonknife = [r for r in recs if not r["knife_edge"]]
            take = knife[:]  # all knife-edge
            if len(nonknife) > uniform_per_stratum:
                idx = rng.choice(len(nonknife), size=uniform_per_stratum, replace=False)
                take += [nonknife[i] for i in idx]
            else:
                take += nonknife
            rows += take
            added += len(take)
        day_round += 1
        # stop once we cover ≥ min_sample and have swept ≥1 full round; cap passes
        seen = {(r["series"], r["window_open_ms"]) for r in rows}
        if len(seen) >= min_sample or added == 0 or day_round >= 4:
            break
    df = pd.DataFrame(rows).drop_duplicates(["series", "window_open_ms"]).reset_index(drop=True)
    return df


def validate_vs_api(
    paths: Paths, *, series: tuple[str, ...] | None = None, min_sample: int = DEFAULT_MIN_SAMPLE,
    fetch: pm.FetchFn = pm.http_fetch_json, min_interval: float = API_MIN_INTERVAL_S,
) -> dict:
    """Stratified catalog-vs-CLOB-API validation. **Any mismatch (API resolved but ≠ the
    catalog) is a hard failure.** Fetches are cached (resumable) + rate-limited. Returns a
    report with ``ok`` (False on any mismatch)."""
    series = tuple(series) if series else hc.DEFAULT_SERIES
    catalog = _read_catalog(paths, series)
    sample = stratified_sample(catalog, paths, min_sample=min_sample)
    cache = _load_api_cache(paths)
    limiter = _RateLimiter(min_interval)

    match = mismatch = api_absent = 0
    mismatches: list[dict] = []
    pending: list[dict] = []
    strata_seen: set[tuple[str, str]] = set()
    for r in sample.itertuples(index=False):
        strata_seen.add((r.series, r.month))
        cid = r.condition_id
        rec = cache.get(cid)
        if rec is None:
            limiter.wait(time.monotonic())
            winner = pm.resolution_winner(pm.clob_market(cid, fetch=fetch))
            rec = {"condition_id": cid, "winner": winner, "ok": winner is not None}
            cache[cid] = rec
            pending.append(rec)
            if len(pending) >= 100:
                _append_api_cache(paths, pending)
                pending = []
        api_winner = rec.get("winner")
        if api_winner is None:
            api_absent += 1
            continue
        if api_winner == r.outcome:
            match += 1
        else:
            mismatch += 1
            if len(mismatches) < 50:
                mismatches.append({"series": r.series, "window_open_ms": int(r.window_open_ms),
                                   "condition_id": cid, "catalog": r.outcome, "api": api_winner,
                                   "knife_edge": bool(r.knife_edge)})
    if pending:
        _append_api_cache(paths, pending)

    n_knife = int(sample["knife_edge"].sum())
    return {
        "sample_size": int(len(sample)), "strata_covered": len(strata_seen),
        "months_covered": sorted({m for _, m in strata_seen}),
        "series_covered": sorted({s for s, _ in strata_seen}),
        "knife_edge_in_sample": n_knife,
        "api_cross_checked": match + mismatch, "match": match, "mismatch": mismatch,
        "api_absent": api_absent, "mismatch_examples": mismatches,
        "ok": bool(mismatch == 0), "min_sample": min_sample,
    }


# --- journal overlap validation ---------------------------------------------
def validate_vs_journal(paths: Paths, res_map: dict, series: tuple[str, ...], *,
                        max_days: int = 2, max_segments: int | None = 30) -> dict:
    """On overlap days (``>= JOURNAL_OWNED_FROM``), the catalog outcome must EXACTLY equal
    the journal's ``Resolved`` outcome. Any mismatch is a hard failure. Bounded to the first
    ``max_days`` overlap days and (for the multi-GB 24/7 journal) the first ``max_segments``
    in-range segments — ~1 h each, so dozens of resolved windows — ample to catch a
    systematic catalog-vs-journal mismatch without decompressing days of journal."""
    overlap_days = sorted({hc.utc_day(k[1]) for k in res_map if hc.day_owner(hc.utc_day(k[1])) == "journal"})
    if not overlap_days:
        return {"checked": 0, "mismatches": 0, "examples": [], "ok": True, "days_scanned": 0}
    scan_days = overlap_days[:max_days]
    lo, hi = scan_days[0], scan_days[-1]
    scan_set = set(scan_days)
    overlap = [k for k in res_map if hc.utc_day(k[1]) in scan_set]
    jr = hc.journal_resolved_outcomes(paths, set(series), lo, hi, max_segments=max_segments)
    checked = mism = 0
    examples: list[dict] = []
    for key in overlap:
        jo = jr.get(key)
        if jo is None:
            continue
        checked += 1
        co = res_map[key]["outcome"]
        if co != jo:
            mism += 1
            if len(examples) < 20:
                examples.append({"series": key[0], "window_open_ms": key[1], "catalog": co, "journal": jo})
    return {"checked": checked, "mismatches": mism, "examples": examples, "ok": bool(mism == 0),
            "days_scanned": len(scan_days), "overlap_days_total": len(overlap_days)}


# --- entry ------------------------------------------------------------------
def historical_resolutions(paths: Paths, *, series: tuple[str, ...] | None = None,
                           min_sample: int = DEFAULT_MIN_SAMPLE, run_api: bool = True,
                           fetch: pm.FetchFn = pm.http_fetch_json) -> dict:
    """Build the catalog resolution cache, validate it (API + journal), write the report."""
    series = tuple(series) if series else hc.DEFAULT_SERIES
    out_dir = paths.out_dir / "historical_resolutions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_stats = build_catalog_cache(paths, series)
    res_map = load_resolution_map(paths, series)
    api = validate_vs_api(paths, series=series, min_sample=min_sample, fetch=fetch) if run_api else \
        {"ok": True, "skipped": True}
    journal = validate_vs_journal(paths, res_map, series)
    hard_fail = (run_api and not api["ok"]) or (not journal["ok"])
    result = {
        "params": {"series": list(series), "min_sample": min_sample, "run_api": run_api},
        "catalog": cache_stats, "api_validation": api, "journal_validation": journal,
        "hard_fail": bool(hard_fail), "peak_rss_mb": procmem.peak_rss_mb(),
    }
    (out_dir / "report.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "historical_resolutions")
    parser.add_argument("--series", default=None, help="comma-separated series (default 4-series)")
    parser.add_argument("--min-sample", type=int, default=DEFAULT_MIN_SAMPLE)
    parser.add_argument("--no-api", action="store_true", help="catalog cache + journal check only (skip API)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    print(f"[historical_resolutions] catalog={_catalog_path(paths)}")
    print(f"[historical_resolutions] cache dir={resolutions_dir(paths)}")
    r = historical_resolutions(paths, series=series, min_sample=args.min_sample, run_api=not args.no_api)
    c = r["catalog"]
    print(f"[historical_resolutions] catalog resolved windows: {c['resolved_windows']:,} {c['per_series']}")
    a = r["api_validation"]
    if not a.get("skipped"):
        print(f"[historical_resolutions] API validation: sample={a['sample_size']:,} "
              f"(knife-edge {a['knife_edge_in_sample']:,}, {a['strata_covered']} strata, "
              f"months {len(a['months_covered'])}) → cross-checked {a['api_cross_checked']:,}, "
              f"match {a['match']:,}, MISMATCH {a['mismatch']:,}, api-absent {a['api_absent']:,}")
        for e in a["mismatch_examples"][:5]:
            print(f"[historical_resolutions]   MISMATCH {e}")
    j = r["journal_validation"]
    print(f"[historical_resolutions] journal overlap: checked {j['checked']:,}, mismatches {j['mismatches']:,}")
    procmem.report_peak_rss("historical_resolutions", r.get("peak_rss_mb"), args.max_rss_mb)
    if r["hard_fail"]:
        print("[historical_resolutions] HARD FAIL — catalog disagrees with the API or the journal.")
        return 1
    print("[historical_resolutions] wrote out/historical_resolutions/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
