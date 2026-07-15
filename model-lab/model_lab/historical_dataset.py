"""Stage — historical_dataset (Part 1).

Build **short_horizon-format** training rows across the full historical span, **day by
day, resumable**, tagging every row's depth feed (``depth_source``). Each output day comes
from exactly ONE primary source (no double-counting), per the day-ownership rule:

- **Telonex-owned days** (``< JOURNAL_OWNED_FROM``): reconstructed here — Binance-anchored
  strike + price/vol from aggTrades, **Telonex ``book_snapshot_25`` depth** microstructure
  (``depth_source = telonex``), **Telonex Polymarket quote** microstructure, and the
  **official** Polymarket resolution (Part 2; windows with no official resolution excluded,
  counted, never guessed). ``label_source = telonex``.
- **Journal-owned days** (``>= JOURNAL_OWNED_FROM``): reuse the **existing** journal path
  (``short_horizon`` with ``include_history=False``, bounded to the journal era) verbatim —
  recorder depth20 + PM book, real Chainlink strike + ``Resolved`` outcome. ``depth_source =
  recorder``, ``label_source = chainlink``.

**Strike & basis (historical days have no Chainlink).** ``K`` is Binance-anchored (last
trade ≤ open). The Chainlink-definition features (``chainlink``, ``basis_bps``,
``basis_ewma`` and the basis-*corrected* ``z``) are journal-era-only — NaN on Telonex rows,
where the emitted ``z`` is the price-only (basis = 0) definition. Overlap days verify the
two definitions against the journal (they differ ~13 bps by design — see ``--overlap-parity``).

Resumable: one partition parquet per ``(series, day)`` under
``out/historical_dataset/parts/`` + a ``manifest.json``; a re-run skips days already ``ok``.
``--combine`` streams the partitions into a single ``out/historical_dataset.parquet``.
Memory is bounded to one day at a time (the Telonex depth read is chunked).

Run::

    python -m model_lab.historical_dataset --since 2026-05-01 --until 2026-05-08
    python -m model_lab.historical_dataset --combine        # concat partitions → one parquet
    python -m model_lab.historical_dataset --overlap-parity  # telonex-vs-recorder feature gap
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import dataset as ds
from . import historical_common as hc
from . import historical_labels as hlbl
from . import historical_resolutions as hr
from . import short_horizon as sh
from .config import Paths, resolve_bounds, resolve_paths, stage_parser
from .io import journal as jio
from .io import telonex_pm as tpm
from .lib import procmem

MS_PER_DAY = 86_400_000

# --- schema = short_horizon + depth_source ----------------------------------
# One new meta column; the rest is short_horizon.COLUMN_SPEC verbatim (so the row format
# is the one learn_short_gbt / money_judge already consume). The historical builder is the
# first code to emit depth_source (standing contract, CLAUDE.md 2026-07-09).
_DEPTH_SOURCE_SPEC = (
    "depth_source", "string", "meta",
    "Which L2 book feed the depth features came from — `telonex` (Telonex book_snapshot_25, "
    "historical), `recorder` (our depth20 capture, journal era), or NA (no depth). Standing "
    "contract: a dataset mixing feeds must tag each row's source.",
)
HIST_COLUMN_SPEC = list(sh.COLUMN_SPEC) + [_DEPTH_SOURCE_SPEC]
HIST_COLUMNS = [c[0] for c in HIST_COLUMN_SPEC]


def _hist_arrow_schema() -> pa.Schema:
    base = sh._arrow_schema()
    return base.append(pa.field("depth_source", pa.string()))


def _add_depth_source(batch: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Tag ``depth_source`` = ``feed`` where the depth features are present (a Telonex/
    recorder depth bar matched), NA otherwise; then order to ``HIST_COLUMNS``."""
    out = batch.copy()
    present = out["depth_feat_asof_ts_ms"].notna() if "depth_feat_asof_ts_ms" in out.columns else pd.Series(False, index=out.index)
    out["depth_source"] = np.where(present.to_numpy(), feed, None)
    for col in HIST_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[HIST_COLUMNS].reset_index(drop=True)


# --- Telonex-owned partition (one series, one day) --------------------------
def _telonex_partition(
    paths: Paths, series_key: str, d: date, excluded: set[str], res_map: dict, *,
    grid_secs: int, horizons: tuple[int, ...], max_staleness_ms: int, cov_tol_ms: int,
    strike_tol_ms: float,
) -> tuple[pd.DataFrame, dict]:
    """Build the historical rows for ``series_key`` opening on ``d`` from Telonex +
    aggTrades, labelled by the **official** resolution (``res_map``). Returns
    ``(batch_df, stats)`` (``batch_df`` empty if nothing resolvable)."""
    stats = {"windows": 0, "resolved": 0, "no_official": 0, "no_strike": 0, "samples": 0}
    prefix = hc.series_prefix(series_key)
    asset = hc.SLUG_PREFIXES[prefix][1]
    symbol = hc.SYMBOL_BY_ASSET[asset]
    day_str = d.isoformat()

    wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
    if not wins:
        return pd.DataFrame(columns=HIST_COLUMNS), stats
    bars = hc.binance_bars_for_day(paths, symbol, d)
    grid = ds.build_feature_grid(ds._binance_ticks_from_bars(bars, asset),
                                 pd.DataFrame(columns=ds.DEPTH_COLS), asset)

    win_rows: list[dict] = []
    pm_frames: list[pd.DataFrame] = []
    token_asset: dict[str, str] = {}
    for w in wins:
        stats["windows"] += 1
        open_ms, close_ms = w["window_open_ms"], w["window_close_ms"]
        official = res_map.get((series_key, int(open_ms)))
        if official is None:  # no official resolution → excluded, counted, never guessed
            stats["no_official"] += 1
            continue
        st = hc.binance_anchored_strike(bars, open_ms, strike_tol_ms)
        if not (st["strike"] > 0.0):
            stats["no_strike"] += 1
            continue
        upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)  # for PM features + up_token
        up_token = str(upq["token_id"].iloc[0]) if not upq.empty else None
        outcome = official["outcome"]
        stats["resolved"] += 1
        win_rows.append({
            "series": series_key, "asset": asset,
            "window_open_ms": int(open_ms), "window_close_ms": int(close_ms),
            "up_token": up_token, "down_token": None,
            "strike": st["strike"], "strike_ts_ms": st["strike_ts_ms"],
            "strike_quality": st["strike_quality"], "label_source": "telonex",
            "outcome": outcome, "outcome_up": 1 if outcome == "Up" else 0,
            "split": "train",  # historical era = train; the journal era carries val
        })
        if up_token and not upq.empty:
            token_asset[up_token] = asset
            q = upq.rename(columns={"ts_ms": "ts"})
            q["token_id"] = up_token
            pm_frames.append(q[["token_id", "ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]])

    if not win_rows:
        return pd.DataFrame(columns=HIST_COLUMNS), stats
    windows_df = pd.DataFrame(win_rows)
    depth_grid = hc.telonex_depth_feature_grid(paths, symbol, asset, d)
    pm_df = (pd.concat(pm_frames, ignore_index=True) if pm_frames
             else pd.DataFrame(columns=["token_id", "ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]))
    mid_by_asset = {asset: grid[["sec", "mid"]]} if not grid.empty else {}
    pm_grid = sh.build_pm_grid(pm_df, mid_by_asset, token_asset)

    batch = sh._build_samples(
        windows_df, grid, grid_secs, horizons, max_staleness_ms,
        depth_grid, pm_grid, {}, {}, cov_tol_ms,  # empty recorder-coverage maps → depth/book_covered False
    )
    sh._assert_no_lookahead(batch)  # the hard rule runs over every Telonex batch (sh._build_samples skips it)
    stats["samples"] = int(len(batch))
    return _add_depth_source(batch, "telonex"), stats


# --- journal-owned partitions (one bounded short_horizon run) ---------------
def _journal_partitions(
    paths: Paths, series_keys: tuple[str, ...], journal_days: set[date], *,
    grid_secs: int, horizons: tuple[int, ...], until_ms: int | None,
    max_staleness_ms: int, strike_tol_ms: float,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Reuse the existing journal path: run ``short_horizon`` (include_history=False) ONCE
    over the journal era, tag ``depth_source=recorder``, and split the result into per-
    ``(series, day_iso)`` batches (only the requested ``journal_days``). Returns the map."""
    if not journal_days:
        return {}
    jlo = min(journal_days)
    since_ms = int(datetime(jlo.year, jlo.month, jlo.day, tzinfo=timezone.utc).timestamp()) * 1000
    tmp_out = paths.out_dir / "historical_dataset" / "_journal_tmp"
    tmp_out.mkdir(parents=True, exist_ok=True)
    tmp_paths = replace(paths, out_dir=tmp_out)
    sh.short_horizon(
        tmp_paths, grid_secs=grid_secs, horizons=horizons, series=series_keys,
        strike_tolerance_secs=strike_tol_ms / 1000.0,
        max_feature_staleness_secs=max_staleness_ms // 1000,
        include_history=False, since_ms=since_ms, until_ms=until_ms,
    )
    sh_path = tmp_paths.table("short_horizon")
    if not sh_path.exists():
        return {}
    full = pd.read_parquet(sh_path, dtype_backend="pyarrow")
    if full.empty:
        return {}
    full = full.copy()
    full["_day"] = (full["window_open_ms"].astype("int64") // MS_PER_DAY)
    wanted_days = {d.toordinal(): d.isoformat() for d in journal_days}
    day_to_ord = {int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) // 86400: d.isoformat()
                  for d in journal_days}
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for (series_key, day_idx), g in full.groupby(["series", "_day"]):
        day_iso = day_to_ord.get(int(day_idx))
        if day_iso is None:
            continue
        batch = g.drop(columns=["_day"]).reset_index(drop=True)
        # dtype_backend=pyarrow read gives pandas-nullable dtypes; coerce to the plain
        # numpy/object shape _add_depth_source + the parquet schema expect.
        batch = batch.astype({c: sh.DTYPES[c] for c in sh.COLUMNS if c in batch.columns}, errors="ignore")
        out[(series_key, day_iso)] = _add_depth_source(batch, "recorder")
    _ = wanted_days
    return out


# --- manifest + partitions --------------------------------------------------
def _journal_days_in_range(paths: Paths, since_ms: int | None, until_ms: int | None) -> set[date]:
    """Journal-owned days (``>= JOURNAL_OWNED_FROM``) present in the journal segment
    filenames within ``[since, until]`` — the authoritative source of journal days (so
    days with a journal but no Telonex data, e.g. after the Telonex range, are still
    built)."""
    lo = hc.utc_day(since_ms) if since_ms is not None else None
    hi = hc.utc_day(until_ms - 1) if until_ms is not None else None
    days: set[date] = set()
    for seg in jio.segment_paths(paths.journal_dir):
        sd = hc._segment_day(seg)
        if sd is None or sd < hc.JOURNAL_OWNED_FROM:
            continue
        if (lo is not None and sd < lo) or (hi is not None and sd > hi):
            continue
        days.add(sd)
    return days


def _parts_dir(paths: Paths) -> Path:
    return paths.out_dir / "historical_dataset" / "parts"


def _partition_path(paths: Paths, series_key: str, day_iso: str) -> Path:
    return _parts_dir(paths) / f"{series_key}__{day_iso}.parquet"


def _manifest_path(paths: Paths) -> Path:
    return paths.out_dir / "historical_dataset" / "manifest.json"


def _load_manifest(paths: Paths) -> dict:
    p = _manifest_path(paths)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(paths: Paths, manifest: dict) -> None:
    p = _manifest_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(p)


def _write_partition(paths: Paths, series_key: str, day_iso: str, batch: pd.DataFrame) -> Path:
    """Atomically write one (series, day) partition parquet with the historical schema."""
    dest = _partition_path(paths, series_key, day_iso)
    dest.parent.mkdir(parents=True, exist_ok=True)
    schema = _hist_arrow_schema()
    tmp = dest.with_suffix(".parquet.tmp")
    table = (pa.Table.from_pandas(batch, schema=schema, preserve_index=False)
             if not batch.empty else schema.empty_table())
    pq.write_table(table, tmp)
    tmp.replace(dest)
    return dest


# --- worker -----------------------------------------------------------------
def historical_dataset(
    paths: Paths, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None,
    grid_secs: int = sh.DEFAULT_GRID_SECS, horizons: tuple[int, ...] = sh.DEFAULT_HORIZONS,
    max_feature_staleness_secs: int = sh.DEFAULT_MAX_FEATURE_STALENESS_SECS,
    coverage_tolerance_secs: float = sh.DEFAULT_COVERAGE_TOLERANCE_SECS,
    strike_tolerance_secs: float = ds.DEFAULT_STRIKE_TOLERANCE_SECS,
    res_map: dict | None = None,
    force: bool = False,
) -> dict:
    """Build (or resume) the day-partitioned historical dataset. Returns a metrics dict
    and updates ``out/historical_dataset/{parts/*, manifest.json, metadata.json}``."""
    paths.ensure_out()
    out_dir = paths.out_dir / "historical_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    _parts_dir(paths).mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, excl_report = hc.load_excluded_slugs(paths)
    if res_map is None:
        res_map = hr.load_resolution_map(paths, series_keys)  # official resolution (primary label)
    manifest = _load_manifest(paths)
    max_staleness_ms = max_feature_staleness_secs * 1000
    cov_tol_ms = int(round(coverage_tolerance_secs * 1000.0))
    strike_tol_ms = strike_tolerance_secs * 1000.0

    tot = defaultdict(int)
    per_source = defaultdict(int)
    journal_todo: set[date] = set()
    days_done = days_skipped = 0

    # --- Telonex-owned days: per (series, day), resumable -------------------
    for series_key in series_keys:
        for d in hlbl._days_for_series(paths, series_key, since_ms, until_ms):
            key = f"{series_key}__{d.isoformat()}"
            owner = hc.day_owner(d)
            if owner == "journal":
                journal_todo.add(d)
                continue
            if not force and manifest.get(key, {}).get("status") == "ok" \
                    and _partition_path(paths, series_key, d.isoformat()).exists():
                days_skipped += 1
                continue
            batch, stats = _telonex_partition(
                paths, series_key, d, excluded, res_map, grid_secs=grid_secs, horizons=horizons,
                max_staleness_ms=max_staleness_ms, cov_tol_ms=cov_tol_ms,
                strike_tol_ms=strike_tol_ms)
            _write_partition(paths, series_key, d.isoformat(), batch)
            manifest[key] = {"series": series_key, "day": d.isoformat(), "source": "telonex",
                             "rows": int(len(batch)), **stats, "status": "ok"}
            _save_manifest(paths, manifest)
            days_done += 1
            tot["samples"] += int(len(batch))
            tot["windows"] += stats["windows"]
            tot["resolved"] += stats["resolved"]
            tot["no_official"] += stats["no_official"]
            per_source["telonex"] += int(len(batch))

    # Journal-owned days also come from the journal itself (not just Telonex coverage),
    # so days with a journal but no Telonex data are still built.
    journal_todo |= _journal_days_in_range(paths, since_ms, until_ms)

    # --- journal-owned days: one bounded short_horizon run, then split ------
    needed_journal = {d for d in journal_todo
                      if force or any(manifest.get(f"{s}__{d.isoformat()}", {}).get("status") != "ok"
                                      or not _partition_path(paths, s, d.isoformat()).exists()
                                      for s in series_keys)}
    if needed_journal:
        jparts = _journal_partitions(
            paths, series_keys, needed_journal, grid_secs=grid_secs, horizons=horizons,
            until_ms=until_ms, max_staleness_ms=max_staleness_ms, strike_tol_ms=strike_tol_ms)
        for (series_key, day_iso), batch in jparts.items():
            _write_partition(paths, series_key, day_iso, batch)
            manifest[f"{series_key}__{day_iso}"] = {
                "series": series_key, "day": day_iso, "source": "recorder",
                "rows": int(len(batch)), "windows": 0, "resolved": 0, "no_official": 0,
                "no_strike": 0, "samples": int(len(batch)), "status": "ok"}
            days_done += 1
            tot["samples"] += int(len(batch))
            per_source["recorder"] += int(len(batch))
        _save_manifest(paths, manifest)
        # mark journal days with no rows for a series as done (so they don't re-run)
        for d in needed_journal:
            for s in series_keys:
                key = f"{s}__{d.isoformat()}"
                if manifest.get(key, {}).get("status") != "ok":
                    _write_partition(paths, s, d.isoformat(), pd.DataFrame(columns=HIST_COLUMNS))
                    manifest[key] = {"series": s, "day": d.isoformat(), "source": "recorder",
                                     "rows": 0, "windows": 0, "resolved": 0, "no_official": 0,
                                     "no_strike": 0, "samples": 0, "status": "ok"}
        _save_manifest(paths, manifest)

    result = {
        "params": {"series": list(series_keys), "grid_secs": grid_secs, "horizons": list(horizons),
                   "since_ms": since_ms, "until_ms": until_ms,
                   "journal_owned_from": hc.JOURNAL_OWNED_FROM.isoformat()},
        "counts": {"days_built": days_done, "days_skipped": days_skipped,
                   "samples": int(tot["samples"]), "windows": int(tot["windows"]),
                   "resolved": int(tot["resolved"]), "no_official_excluded": int(tot["no_official"]),
                   "samples_by_depth_source": dict(per_source)},
        "coverage_exclusion": excl_report,
        "partitions": len([k for k, v in manifest.items() if v.get("status") == "ok"]),
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_metadata(out_dir, result)
    _write_schema(out_dir)
    return result


# --- combine ----------------------------------------------------------------
def combine(paths: Paths) -> dict:
    """Stream all partition parquets into one ``out/historical_dataset.parquet`` (via a
    single ParquetWriter — bounded memory). Returns row/partition counts."""
    parts = sorted(_parts_dir(paths).glob("*.parquet"))
    schema = _hist_arrow_schema()
    dest = paths.table("historical_dataset")
    writer = pq.ParquetWriter(dest, schema)
    rows = 0
    used = 0
    try:
        for p in parts:
            table = pq.read_table(p)
            if table.num_rows == 0:
                continue
            writer.write_table(table.cast(schema))
            rows += table.num_rows
            used += 1
        if rows == 0:
            writer.write_table(schema.empty_table())
    finally:
        writer.close()
    return {"partitions": len(parts), "partitions_with_rows": used, "rows": rows,
            "out": str(dest)}


# --- overlap-day parity (telonex-vs-recorder feature gap) -------------------
def overlap_parity(paths: Paths, series_keys: tuple[str, ...], days: list[date]) -> dict:
    """On overlap days (journal-owned), rebuild the **Telonex** version for assertion only
    and compare the source-agnostic features (``mid``, ``sigma_1s``, and the depth
    microstructure) against the journal-owner rows — the telonex-vs-recorder feature gap
    the standing contract wants. Uses the ``telonex_binance_validate`` guard tolerances.
    Chainlink-definition columns (basis-corrected z, chainlink, basis_*) are excluded by
    design. Returns per-feature magnitude/bias/correlation + a PASS/FAIL verdict."""
    from . import telonex_binance_validate as tbv
    guard_features = ["mid", "sigma_1s", "depth_imb_20", "microprice_gap",
                      "bid_depth_slope", "ask_depth_slope"]
    rows: list[dict] = []
    excluded, _ = hc.load_excluded_slugs(paths)
    res_map = hr.load_resolution_map(paths, series_keys)
    max_staleness_ms = sh.DEFAULT_MAX_FEATURE_STALENESS_SECS * 1000
    cov_tol_ms = int(round(sh.DEFAULT_COVERAGE_TOLERANCE_SECS * 1000.0))
    # journal-owner rows come from the built partitions
    for d in days:
        if hc.day_owner(d) != "journal":
            continue
        for series_key in series_keys:
            jpart = _partition_path(paths, series_key, d.isoformat())
            if not jpart.exists():
                continue
            jrows = pd.read_parquet(jpart)
            jrows = jrows[jrows["depth_source"] == "recorder"]
            if jrows.empty:
                continue
            tbatch, _ = _telonex_partition(
                paths, series_key, d, excluded, res_map, grid_secs=sh.DEFAULT_GRID_SECS,
                horizons=sh.DEFAULT_HORIZONS, max_staleness_ms=max_staleness_ms,
                cov_tol_ms=cov_tol_ms, strike_tol_ms=ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0)
            if tbatch.empty:
                continue
            key = ["series", "window_open_ms", "sample_ts_ms"]
            merged = jrows.merge(tbatch, on=key, suffixes=("_j", "_t"))
            for feat in guard_features:
                a = pd.to_numeric(merged.get(f"{feat}_j"), errors="coerce")
                b = pd.to_numeric(merged.get(f"{feat}_t"), errors="coerce")
                m = a.notna() & b.notna()
                if int(m.sum()) < 20:
                    continue
                av, bv = a[m].to_numpy(), b[m].to_numpy()
                bias = float(np.mean(bv - av))
                ratio = float(np.mean(np.abs(bv)) / np.mean(np.abs(av))) if np.mean(np.abs(av)) > 0 else float("nan")
                corr = float(np.corrcoef(av, bv)[0, 1]) if (av.std() > 0 and bv.std() > 0) else 1.0
                rows.append({"series": series_key, "day": d.isoformat(), "feature": feat,
                             "n": int(m.sum()), "bias": bias, "magnitude_ratio": ratio, "corr": corr})
    # verdict per feature (aggregate across days)
    verdicts: list[dict] = []
    ok = True
    for feat in guard_features:
        fr = [r for r in rows if r["feature"] == feat]
        if not fr:
            continue
        n = sum(r["n"] for r in fr)
        bias = float(np.average([r["bias"] for r in fr], weights=[r["n"] for r in fr]))
        ratio = float(np.average([r["magnitude_ratio"] for r in fr], weights=[r["n"] for r in fr]))
        corr = float(np.average([r["corr"] for r in fr], weights=[r["n"] for r in fr]))
        passed = (abs(bias) <= tbv.GUARD_MAX_BIAS or feat in ("mid", "sigma_1s")) \
            and (tbv.GUARD_MAG_LO <= ratio <= tbv.GUARD_MAG_HI) and (corr >= tbv.GUARD_CORR_CONVERGE)
        ok = ok and passed
        verdicts.append({"feature": feat, "n": n, "bias": bias, "magnitude_ratio": ratio,
                         "corr": corr, "pass": bool(passed)})
    return {"verdicts": verdicts, "n_features": len(verdicts), "pass": bool(ok and verdicts),
            "detail": rows}


# --- outputs ----------------------------------------------------------------
def _write_metadata(out_dir: Path, result: dict) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _write_schema(out_dir: Path) -> None:
    cat_label = {"id": "identifier", "meta": "metadata (known at sample time)",
                 "feature": "**feature** (as-of ≤ sample_ts)", "label": "**label** (reads the future)",
                 "coverage": "**coverage** (provenance metadata; not a feature)"}
    lines = [
        "# `historical_dataset.parquet` — schema",
        "",
        "Short_horizon-format rows across the full historical span, day by day. Built by "
        "`python -m model_lab.historical_dataset` (+ `--combine`). Same columns as "
        "`short_horizon` plus `depth_source`.",
        "",
        "| column | dtype | category | description |",
        "|---|---|---|---|",
    ]
    for name, dtype, category, doc in HIST_COLUMN_SPEC:
        lines.append(f"| `{name}` | `{dtype}` | {cat_label[category]} | {doc} |")
    lines += [
        "",
        "## Day ownership & sources",
        "",
        f"- Each day is owned by exactly one source. Journal owns days `>= "
        f"{hc.JOURNAL_OWNED_FROM.isoformat()}` (`label_source=chainlink`, "
        "`depth_source=recorder`, real Chainlink strike + `Resolved` outcome, reusing the "
        "existing `short_horizon` path); Telonex owns earlier days (`label_source=telonex`, "
        "`depth_source=telonex`|NA).",
        "- **Strike & basis (Telonex days):** `K` = last Binance aggTrades trade ≤ open "
        "(Binance-anchored). There is no Chainlink feed pre-journal, so `chainlink`, "
        "`basis_bps`, `basis_ewma` are `NaN` and `z` uses the **price-only** (basis = 0) "
        "definition — journal-era-only for the Chainlink-corrected form. Verified on overlap "
        "days against the journal (they differ ~13 bps by design; see `--overlap-parity`).",
        "- **Outcome (all days):** the **official** Polymarket resolution (`historical_resolutions`, "
        "validated vs the CLOB API + the journal); windows with no official resolution are excluded "
        "(counted, never guessed). Quote-convergence and Binance-anchored are graded cross-checks.",
        "- **`depth_source`:** `telonex` / `recorder` / NA — the standing contract; the "
        "recorder-provenance flags `depth_covered`/`book_covered` are always `False` on "
        "Telonex rows (they predate our capture), while `depth_feat_covered`/`pm_feat_covered` "
        "reflect real Telonex-feature presence.",
        "",
    ]
    (out_dir / "SCHEMA.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "historical_dataset")
    parser.add_argument("--series", default=None,
                        help="comma-separated series filter (default the 4-series universe)")
    parser.add_argument("--grid-secs", type=int, default=sh.DEFAULT_GRID_SECS)
    parser.add_argument("--force", action="store_true", help="rebuild days already in the manifest")
    parser.add_argument("--combine", action="store_true",
                        help="concat existing partitions into out/historical_dataset.parquet (no build)")
    parser.add_argument("--overlap-parity", action="store_true",
                        help="run the telonex-vs-recorder feature-gap parity over overlap days in scope")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    series_keys = series or hc.DEFAULT_SERIES

    if args.combine:
        r = combine(paths)
        print(f"[historical_dataset] combined {r['partitions_with_rows']}/{r['partitions']} "
              f"partitions → {r['rows']:,} rows at {r['out']}")
        return 0

    print(f"[historical_dataset] telonex={paths.telonex_dir} hist={paths.hist_dir}")
    print(f"[historical_dataset] out={paths.out_dir / 'historical_dataset'}")
    result = historical_dataset(
        paths, series=series, since_ms=since_ms, until_ms=until_ms,
        grid_secs=args.grid_secs, force=args.force)
    c = result["counts"]
    print(f"[historical_dataset] built {c['days_built']} day-partitions ({c['days_skipped']} skipped), "
          f"{c['samples']:,} samples by source {c['samples_by_depth_source']}")
    print(f"[historical_dataset] telonex windows {c['windows']:,}: resolved {c['resolved']:,} (official), "
          f"no-official excluded {c['no_official_excluded']:,}")

    if args.overlap_parity:
        days = sorted({d for s in series_keys for d in hlbl._days_for_series(paths, s, since_ms, until_ms)})
        par = overlap_parity(paths, series_keys, days)
        result["overlap_parity"] = par
        _write_metadata(paths.out_dir / "historical_dataset", result)
        if par["n_features"] == 0:
            print("[historical_dataset] overlap-parity: no overlap days with both sources in scope")
        else:
            print(f"[historical_dataset] overlap-parity: {'PASS' if par['pass'] else 'FAIL'} "
                  f"over {par['n_features']} features")
            for v in par["verdicts"]:
                print(f"[historical_dataset]   {v['feature']}: corr={v['corr']:.4f} "
                      f"ratio={v['magnitude_ratio']:.3f} bias={v['bias']:+.4g} "
                      f"{'ok' if v['pass'] else 'FAIL'}")
            if not par["pass"]:
                print("[historical_dataset] overlap-parity FAILED — telonex vs recorder depth gap.")
                return 1
    procmem.report_peak_rss("historical_dataset", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[historical_dataset] manifest: {result['partitions']} ok partitions. "
          f"Run with --combine to build one parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
