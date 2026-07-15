"""Telonex full pull — download the full Polymarket quotes+trades history + verify.

The production download after the ``telonex_ingest`` / ``telonex_validate`` trial went
GO. Pulls the **full available history** of Polymarket order-book **quotes** and
**trades** for the in-scope crypto up/down series (btc/eth × 5m/15m; **1h and Binance
are out of scope**), streaming every daily file to ``../data/telonex/raw`` and
compressing it to ``.zst`` as it lands. Download + verify ONLY — no feature building.

Design (see the plan / telonex_notes.md):

- **Enumerate** every in-scope window from the public markets catalog in one scan, and
  **project** raw+compressed size vs the ``--cap-gb`` budget FIRST (``--dry-run`` stops
  here). If the projection exceeds the cap, drop the OLDEST 15m history first.
- **Quota gate:** read ``X-Downloads-Remaining`` before the bulk loop; STOP and report
  if it can't cover the pull rather than burn a partial download (the Single-Exchange
  plan is unlimited for Polymarket, so this normally passes trivially).
- **Per (series, day), crash-safe:** download (skip if the ``.zst`` already exists) →
  inline integrity on the raw parquet → compress → roundtrip-verify a random sample →
  checkpoint ``coverage.json`` → delete the raw (keeping only the ``.zst``). Every step
  is atomic (``tmp → os.replace``); a resume skips finalized days and re-does the rest.
- Writes an incremental ``out/telonex/{coverage.json, full_pull_report.html}`` with a
  loud per-day PASS/WARN/FAIL table; missing days/windows are flagged.

    python -m model_lab.telonex_pull --dry-run          # project + budget + quota, no download
    python -m model_lab.telonex_pull                    # the full pull (hours; resumable)
    python -m model_lab.telonex_pull --start 2026-04-01 # clamp the date range

The API key is read from ``$telonexdata`` / ``.env`` and never printed.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq

from .config import Paths, resolve_paths, stage_parser
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem
from .telonex_validate import GAP_THRESHOLD_S, MIRROR_TOL

DEFAULT_SERIES = "btc-updown-5m,eth-updown-5m,btc-updown-15m,eth-updown-15m"
DEFAULT_CHANNELS = "quotes,trades"
DEFAULT_OUTCOMES = "Up,Down"
DEFAULT_CAP_GB = 450.0
DEFAULT_JOBS = 3
DEFAULT_ZSTD_LEVEL = 12          # the trial's validated level — fast, ~1.5× ratio (19 is far slower)
DEFAULT_ROUNDTRIP_SAMPLE = 20    # random .zst bit-verified per day before deleting raw
DEFAULT_MIRROR_SAMPLE = 0        # Up/Down mirror cross-check windows/day (0 = off; bonus, not required)
DEFAULT_DISK_FACTOR = 1.5
DEFAULT_DISK_FLOOR_GB = 10.0
DEFAULT_REPORT_EVERY = 5         # re-render the HTML every N finalized days
RECENT_DAY_LAG = 2               # a 404 on one of the last N catalog days = WARN (publish lag), not FAIL
FAIL_MISSING_FRAC = 0.20         # a day missing > this fraction of its expected files = FAIL
GAP_THRESHOLD_US = int(GAP_THRESHOLD_S * 1_000_000)
GAP_STALE_WARN_SECS = 30.0    # a single in-window quiet stretch longer than this flags the day WARN
GAP_OUTLIER_IQR_K = 3.0       # a day's >5s-gap rate above Q3 + K·IQR of the series' own history = outlier
GAP_OUTLIER_MIN_DAYS = 12     # need at least this many days to judge a series' gap-rate distribution
SCHEMA_VERSION = 1
ALLOWED_CHANNELS = ("quotes", "trades")
COVERAGE_NAME = "coverage.json"
REPORT_NAME = "full_pull_report.html"

# Per-window compressed/raw bytes (both outcomes) fallback, if the trial 5m day isn't on
# disk to measure from. Measured from the 2026-07-05 btc-5m trial.
FALLBACK_PER_WINDOW_ZST = {"quotes": 0.58e6, "trades": 0.30e6}
FALLBACK_PER_WINDOW_RAW = {"quotes": 0.73e6, "trades": 0.40e6}


def _safe_unlink(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Enumeration → expected files → projection / budget
# --------------------------------------------------------------------------- #


def _expected_specs(telonex_dir: Path, window: dict[str, Any], outcomes: list[str]) -> list[dict[str, Any]]:
    """Every expected (channel, outcome) file for one window (only channels with data)."""
    day_str = window["open_day"]
    specs: list[dict[str, Any]] = []
    for channel in window["channels"]:  # channels present per the catalog
        for outcome in outcomes:
            raw = tx.polymarket_raw_path(telonex_dir, channel, window["slug"], outcome, day_str)
            specs.append({
                "channel": channel, "slug": window["slug"], "outcome": outcome, "day": day_str,
                "raw": raw, "zst": tx.zst_sibling(raw),
                "open_ms": window.get("open_ms"), "dur": window.get("dur"),
            })
    return specs


def _group_by_day(windows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """``[(series, day, [windows])]`` ordered by (series, day). Enumerate order is preserved."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for w in windows:
        groups.setdefault((w["series"], w["open_day"]), []).append(w)
    return [(s, d, groups[(s, d)]) for (s, d) in sorted(groups)]


def _ref_series(series: list[str]) -> str:
    for s in series:
        if s.endswith("-5m"):
            return s
    return series[0]


def _slug_of(name: str, suffix_len: int, day_len: int = 10) -> str:
    """Recover the window slug from ``{slug}-{outcome}-{day}{suffix}`` (day/outcome have no
    leading slug hyphen ambiguity because we strip fixed-width tails)."""
    stem = name[:-suffix_len] if suffix_len else name
    core = stem[:-(day_len + 1)]          # drop "-YYYY-MM-DD"
    return core.rsplit("-", 1)[0]         # drop "-Up" / "-Down"


def _measure_per_window(
    telonex_dir: Path, channels: list[str], ref_series: str,
) -> tuple[dict[str, float], dict[str, float], str]:
    """Measure per-window (both outcomes) zst+raw bytes from the most complete on-disk day
    of ``ref_series`` (a 5m series), falling back to the trial constants per channel."""
    asset, ref_dur = tx.parse_series(ref_series)
    prefix = f"{ref_series}-"
    root = tx.raw_root(telonex_dir) / tx.POLYMARKET
    per_zst: dict[str, float] = {}
    per_raw: dict[str, float] = {}
    for ch in channels:
        d = root / ch
        if d.is_dir():
            by_day: dict[str, list[Path]] = {}
            for p in d.glob(f"{prefix}*.parquet.zst"):
                by_day.setdefault(p.name[:-len(".parquet.zst")][-10:], []).append(p)
            if by_day:
                day = max(by_day, key=lambda k: len(by_day[k]))
                zst_files = by_day[day]
                slugs = {_slug_of(p.name, len(".parquet.zst")) for p in zst_files}
                n = max(1, len(slugs))
                per_zst[ch] = sum(p.stat().st_size for p in zst_files) / n
                raw_files = list(d.glob(f"{prefix}*-{day}.parquet"))
                if raw_files:
                    rslugs = {_slug_of(p.name, len(".parquet")) for p in raw_files}
                    per_raw[ch] = sum(p.stat().st_size for p in raw_files) / max(1, len(rslugs))
        per_zst.setdefault(ch, FALLBACK_PER_WINDOW_ZST.get(ch, 0.5e6))
        per_raw.setdefault(ch, FALLBACK_PER_WINDOW_RAW.get(ch, per_zst[ch] * 1.5))
    return per_zst, per_raw, ref_dur


def project_pull(
    windows: list[dict[str, Any]], per_zst: dict[str, float], per_raw: dict[str, float],
    ref_dur: str, *, cap_bytes: int,
) -> dict[str, Any]:
    """Project raw + compressed size from the per-window bytes, scaling 15m ≈ 3× 5m."""
    ref_secs = tx.duration_seconds(ref_dur)
    by_series: dict[str, dict[str, Any]] = {}
    for w in windows:
        bs = by_series.setdefault(w["series"], {"dur": w["dur"], "windows": 0,
                                                "chan_windows": {c: 0 for c in per_zst}})
        bs["windows"] += 1
        for ch in w["channels"]:
            if ch in bs["chan_windows"]:
                bs["chan_windows"][ch] += 1
    zst_total = raw_total = 0.0
    detail: dict[str, Any] = {}
    for s, bs in sorted(by_series.items()):
        scale = tx.duration_seconds(bs["dur"]) / ref_secs
        s_zst = sum(bs["chan_windows"][ch] * per_zst[ch] * scale for ch in per_zst)
        s_raw = sum(bs["chan_windows"][ch] * per_raw[ch] * scale for ch in per_raw)
        zst_total += s_zst
        raw_total += s_raw
        detail[s] = {"windows": bs["windows"], "est_zst_bytes": int(s_zst),
                     "est_zst_human": human_bytes(int(s_zst)), "est_raw_bytes": int(s_raw),
                     "est_raw_human": human_bytes(int(s_raw))}
    return {
        "per_window_zst_by_channel": {c: int(v) for c, v in per_zst.items()},
        "per_window_raw_by_channel": {c: int(v) for c, v in per_raw.items()},
        "ref_dur": ref_dur,
        "raw_bytes": int(raw_total), "raw_human": human_bytes(int(raw_total)),
        "zst_bytes": int(zst_total), "zst_human": human_bytes(int(zst_total)),
        "cap_bytes": int(cap_bytes), "cap_human": human_bytes(int(cap_bytes)),
        "fits_cap": bool(zst_total <= cap_bytes),
        "shrink_applied": False, "dropped": [], "series_detail": detail,
    }


def _project_placeholder(
    windows: list[dict[str, Any]], channels: list[str], outcomes: list[str],
) -> dict[str, Any]:
    """A projection from the fallback per-window bytes (no on-disk measurement) — used by
    the coverage stage when no prior ``coverage.json`` projection exists."""
    per_zst = {c: FALLBACK_PER_WINDOW_ZST.get(c, 0.5e6) for c in channels}
    per_raw = {c: FALLBACK_PER_WINDOW_RAW.get(c, per_zst[c] * 1.5) for c in channels}
    ref = _ref_series(sorted({w["series"] for w in windows}) or ["btc-updown-5m"])
    _asset, ref_dur = tx.parse_series(ref)
    return project_pull(windows, per_zst, per_raw, ref_dur, cap_bytes=int(DEFAULT_CAP_GB * 1024 ** 3))


def apply_budget_shrink(
    projection: dict[str, Any], windows: list[dict[str, Any]], per_zst: dict[str, float],
    ref_dur: str, n_outcomes: int, *, cap_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop the OLDEST 15m days first until the projected zst fits ``cap_bytes``. 5m is
    never dropped. Returns ``(kept_windows, shrink_log)``."""
    if projection["zst_bytes"] <= cap_bytes:
        return windows, {"shrink_applied": False, "dropped": [], "freed_bytes": 0, "still_over": False}
    ref_secs = tx.duration_seconds(ref_dur)
    scale = tx.duration_seconds("15m") / ref_secs
    day_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for w in windows:
        if w["dur"] == "15m":
            day_groups.setdefault((w["open_day"], w["series"]), []).append(w)
    dropped: list[dict[str, Any]] = []
    drop_slugs: set[str] = set()
    freed = 0.0
    for (day, s) in sorted(day_groups):  # (day, series) ascending = oldest day first
        gw = day_groups[(day, s)]
        gbytes = sum(sum(per_zst[ch] for ch in w["channels"] if ch in per_zst) * scale for w in gw)
        for w in gw:
            drop_slugs.add(w["slug"])
        dropped.append({"series": s, "day": day,
                        "files": sum(len(w["channels"]) * n_outcomes for w in gw), "bytes": int(gbytes)})
        freed += gbytes
        if projection["zst_bytes"] - freed <= cap_bytes:
            break
    kept = [w for w in windows if w["slug"] not in drop_slugs]
    return kept, {"shrink_applied": True, "dropped": dropped, "freed_bytes": int(freed),
                  "still_over": bool(projection["zst_bytes"] - freed > cap_bytes)}


# --------------------------------------------------------------------------- #
# Download + compress workers (mirrors telonex_ingest._execute — abort on 403)
# --------------------------------------------------------------------------- #


def _run_file(spec: dict[str, Any], *, api_key: str, fetch: tx.Fetch, level: int, jitter_ms: int) -> dict[str, Any]:
    raw, zst = spec["raw"], spec["zst"]
    if zst.exists() and not raw.exists():                      # state D — already final
        return {"spec": spec, "status": "finalized", "zst_bytes": zst.stat().st_size, "remaining": None}
    downloaded = False
    remaining = None
    if not raw.exists():
        res = tx.download_asset_day(tx.POLYMARKET, spec["channel"], spec["day"], raw,
                                    api_key=api_key, fetch=fetch, slug=spec["slug"], outcome=spec["outcome"])
        remaining = res.get("remaining")
        if res.get("status") == "no_data":                     # 404 — legit-empty vs missing decided later
            return {"spec": spec, "status": "no_data", "remaining": remaining}
        downloaded = res.get("status") == "downloaded"
    if not zst.exists():
        tx.compress_file_zstd(raw, zst, level=level)
        if jitter_ms:
            time.sleep(jitter_ms / 1000.0)
    return {"spec": spec, "status": "downloaded" if downloaded else "present",
            "raw_bytes": raw.stat().st_size, "zst_bytes": zst.stat().st_size, "remaining": remaining}


def _run_file_safe(spec, *, api_key, fetch, level, jitter_ms) -> dict[str, Any]:
    try:
        return _run_file(spec, api_key=api_key, fetch=fetch, level=level, jitter_ms=jitter_ms)
    except tx.TelonexDownloadLimit:
        raise
    except tx.TelonexPlanRestricted as e:
        return {"spec": spec, "status": "error", "error": f"plan_restricted: {e}", "remaining": None}
    except Exception as e:  # noqa: BLE001 — one file's failure isn't fatal (unlike a 403)
        return {"spec": spec, "status": "error", "error": str(e), "remaining": None}


def _execute_files(
    todo: list[dict[str, Any]], *, jobs: int, api_key: str, fetch: tx.Fetch, level: int,
    jitter_ms: int, on_result: Callable[[dict[str, Any]], None],
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Download+compress ``todo`` in parallel; return ``(results, aborted, limit_msg)``.
    A ``TelonexDownloadLimit`` (403) cancels the rest and aborts (resume-safe)."""
    results: list[dict[str, Any]] = []
    if not todo:
        return results, False, None
    if jobs <= 1 or len(todo) <= 1:
        for s in todo:
            try:
                r = _run_file_safe(s, api_key=api_key, fetch=fetch, level=level, jitter_ms=jitter_ms)
            except tx.TelonexDownloadLimit as e:
                return results, True, str(e)
            results.append(r)
            on_result(r)
        return results, False, None
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run_file_safe, s, api_key=api_key, fetch=fetch, level=level,
                          jitter_ms=jitter_ms): s for s in todo}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except tx.TelonexDownloadLimit as e:
                for f in futs:
                    f.cancel()
                return results, True, str(e)
            results.append(r)
            on_result(r)
    return results, False, None


# --------------------------------------------------------------------------- #
# Inline integrity (on the raw parquet, before it is deleted)
# --------------------------------------------------------------------------- #


def _integrity_read(path: Path, channel: str, window_us: tuple[int, int] | None = None) -> dict[str, Any]:
    if channel == "trades":
        df = pq.read_table(path, columns=["timestamp_us", "trade_id"]).to_pandas()
        ts = df["timestamp_us"].to_numpy(dtype="int64")
        rows = len(df)
        dup = int(rows - df["trade_id"].nunique()) if rows else 0
        nonmono = int((np.diff(ts) < 0).sum()) if ts.size >= 2 else 0
        return {"rows": rows, "dup": dup, "nonmono": nonmono, "gaps": 0, "max_gap_s": 0.0}
    df = pq.read_table(path, columns=["timestamp_us"]).to_pandas()   # quotes (top-of-book)
    ts = df["timestamp_us"].to_numpy(dtype="int64")
    rows = len(df)
    nonmono = int((np.diff(ts) < 0).sum()) if ts.size >= 2 else 0
    # Gaps only within the active trading interval — the whole file also holds pre-open
    # resting-order events (~hours before open) whose sparsity would swamp the count.
    tw = ts
    if window_us is not None:
        lo, hi = window_us
        tw = ts[(ts >= lo) & (ts < hi)]
    gaps = 0
    max_gap_s = 0.0
    if tw.size >= 2:
        d = np.diff(np.sort(tw))
        gaps = int((d > GAP_THRESHOLD_US).sum())        # the raw >5s count (kept — a liquidity signal)
        max_gap_s = float(d.max()) / 1_000_000.0        # longest single in-window stale stretch
    return {"rows": rows, "dup": 0, "nonmono": nonmono, "gaps": gaps, "max_gap_s": max_gap_s}


def _integrity_file(spec: dict[str, Any], scratch: Path) -> dict[str, Any] | None:
    """Read one file's integrity stats from the raw parquet (or, if only the ``.zst``
    exists, a decompressed temp). ``None`` = no data on disk."""
    raw, zst = spec["raw"], spec["zst"]
    tmp = None
    try:
        if raw.exists():
            path = raw
        elif zst.exists():
            # channel-prefixed: quotes/ and trades/ share filenames, and the deep pass
            # decompresses in parallel — an un-prefixed temp would race across channels.
            tmp = scratch / f"{spec['channel']}__{raw.name}.read.tmp"
            tx.decompress_zstd(zst, tmp)
            path = tmp
        else:
            return None
        window_us = None
        if spec.get("open_ms") is not None and spec.get("dur"):
            lo = int(spec["open_ms"]) * 1000
            window_us = (lo, lo + tx.duration_seconds(spec["dur"]) * 1_000_000)
        return _integrity_read(path, spec["channel"], window_us=window_us)
    except Exception as e:  # noqa: BLE001 — an unreadable file is a per-file error, not a crash
        return {"error": str(e)}
    finally:
        if tmp is not None:
            _safe_unlink(tmp)


def _mirror_check(specs: list[dict[str, Any]], scratch: Path, sample: int, rng: random.Random) -> dict[str, Any]:
    """Optional Up/Down mirror sanity: Down.bid ≈ 1 − Up.ask over a few sampled windows."""
    if sample <= 0:
        return {"frac": None, "checked": 0}
    import pandas as pd

    by_slug: dict[str, dict[str, dict[str, Any]]] = {}
    for s in specs:
        if s["channel"] == "quotes":
            by_slug.setdefault(s["slug"], {})[s["outcome"]] = s
    pairs = [(sl, d) for sl, d in by_slug.items() if "Up" in d and "Down" in d]
    if not pairs:
        return {"frac": None, "checked": 0}
    chosen = pairs if len(pairs) <= sample else rng.sample(pairs, sample)

    def tops(spec: dict[str, Any]):
        raw, zst = spec["raw"], spec["zst"]
        tmp = None
        try:
            if raw.exists():
                path = raw
            elif zst.exists():
                tmp = scratch / (raw.name + ".mirror.tmp")
                tx.decompress_zstd(zst, tmp)
                path = tmp
            else:
                return None
            df = pq.read_table(path, columns=["timestamp_us", "bid_price", "ask_price"]).to_pandas()
            return pd.DataFrame({"ts_ms": df["timestamp_us"].to_numpy("int64") // 1000,
                                 "bid": pd.to_numeric(df["bid_price"], errors="coerce"),
                                 "ask": pd.to_numeric(df["ask_price"], errors="coerce")})
        except Exception:  # noqa: BLE001
            return None
        finally:
            if tmp is not None:
                _safe_unlink(tmp)

    ok = checked = 0
    for _sl, d in chosen:
        u, dn = tops(d["Up"]), tops(d["Down"])
        if u is None or dn is None or u.empty or dn.empty:
            continue
        merged = pd.merge_asof(u.sort_values("ts_ms"),
                               dn.sort_values("ts_ms").rename(columns={"ts_ms": "ts_d"}),
                               left_on="ts_ms", right_on="ts_d", direction="nearest",
                               tolerance=2000, suffixes=("_u", "_d")).dropna(subset=["ts_d"])
        if merged.empty:
            continue
        checked += 1
        frac = float((np.abs(merged["bid_d"] - (1 - merged["ask_u"])) < MIRROR_TOL).mean())
        if frac > 0.9:
            ok += 1
    return {"frac": (ok / checked) if checked else None, "checked": checked}


def _day_record(
    specs: list[dict[str, Any]], status_by_key: dict[tuple, str], *, expected_windows: int,
    scratch: Path, roundtrip_sample: int, mirror_sample: int, rng: random.Random,
) -> dict[str, Any]:
    downloaded_files = zst_files = 0
    missing: list[str] = []
    errors: list[str] = []
    empty: list[str] = []
    rows_total = dup = nonmono = gaps = 0
    max_stale = 0.0
    rt_pairs: list[tuple[Path, Path]] = []
    zst_bytes = 0
    for s in specs:
        key = (s["channel"], s["slug"], s["outcome"])
        has_zst, has_raw = s["zst"].exists(), s["raw"].exists()
        label = f"{s['slug']}/{s['outcome']}/{s['channel']}"
        if not (has_zst or has_raw):
            st = status_by_key.get(key)
            (missing if st == "no_data" else errors).append(
                label if st == "no_data" else f"{label}: {st}")
            continue
        if has_zst:
            zst_files += 1
            zst_bytes += s["zst"].stat().st_size
        downloaded_files += 1
        if has_raw and has_zst:
            rt_pairs.append((s["raw"], s["zst"]))
        integ = _integrity_file(s, scratch)
        if integ is None:
            continue
        if "error" in integ:
            errors.append(f"{label}: read {integ['error']}")
            continue
        rows_total += integ["rows"]
        dup += integ["dup"]
        nonmono += integ["nonmono"]
        gaps += integ["gaps"]
        max_stale = max(max_stale, integ.get("max_gap_s", 0.0))
        if integ["rows"] == 0:
            empty.append(label)
    rt = tx.sample_roundtrip(rt_pairs, k=roundtrip_sample, rng=rng, scratch=scratch)
    mirror = _mirror_check(specs, scratch, mirror_sample, rng)
    return {
        "expected_windows": expected_windows, "expected_files": len(specs),
        "downloaded_files": downloaded_files, "zst_files": zst_files,
        "missing_files": len(missing), "missing_windows": missing[:200],
        "error_files": errors[:50], "empty_files": empty[:50],
        "rows_total": rows_total, "dup_trade_ids": dup, "nonmonotonic_ts": nonmono,
        "gaps_gt_5s": gaps, "max_stale_secs": round(max_stale, 1),
        "mirror_frac": mirror["frac"], "mirror_checked": mirror["checked"],
        "roundtrip_checked": rt["checked"], "roundtrip_ok": rt["ok"], "roundtrip_failed": rt["failed"],
        "zst_bytes": zst_bytes, "raw_deleted": False,
    }


def _classify_day(rec: dict[str, Any], *, recent_day: bool) -> tuple[str, list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    if rec["dup_trade_ids"] > 0:
        problems.append(f"{rec['dup_trade_ids']} duplicate trade_ids")
    if rec["nonmonotonic_ts"] > 0:
        problems.append(f"{rec['nonmonotonic_ts']} non-monotonic timestamps")
    if rec["empty_files"]:
        problems.append(f"{len(rec['empty_files'])} empty expected file(s)")
    if rec["error_files"]:
        problems.append(f"{len(rec['error_files'])} unreadable file(s)")
    if rec["roundtrip_failed"]:
        problems.append(f"{len(rec['roundtrip_failed'])} roundtrip failure(s)")
    exp, miss, got = rec["expected_files"], rec["missing_files"], rec["downloaded_files"]
    if got == 0 and exp > 0:
        problems.append("zero files downloaded for a day the catalog says has data")
    if miss > 0:
        if recent_day:
            warnings.append(f"{miss} missing file(s) (recent-day publish lag)")
        elif exp and miss > FAIL_MISSING_FRAC * exp:
            problems.append(f"{miss}/{exp} expected files missing (catalog says available)")
        else:
            warnings.append(f"{miss} missing file(s) (catalog-available but 404)")
    if rec["mirror_frac"] is not None and rec["mirror_checked"] >= 2 and rec["mirror_frac"] < 0.9:
        warnings.append(f"mirror {rec['mirror_frac']:.2f} below 0.9")
    # Gap-based warnings are NOT decided here — `_apply_gap_labels` sets them with series context
    # (a >5s gap alone is normal book sparsity, not a defect). Status may be upgraded there.
    status = "FAIL" if problems else ("WARN" if warnings else "PASS")
    return status, problems, warnings


def _gap_rate(rec: dict[str, Any]) -> float:
    """A day's >5s-gap rate = in-window >5s gaps per window (consistent across a series' days)."""
    return rec.get("gaps_gt_5s", 0) / max(1, rec.get("expected_windows", 1) or 1)


def _apply_gap_labels(cov: dict[str, Any]) -> None:
    """Set the gap-based WARN label with meaning, over the WHOLE run. A day is flagged on gap
    grounds only if a single in-window stale stretch exceeds ``GAP_STALE_WARN_SECS`` (30s), or
    its >5s-gap rate is a clear outlier (> Q3 + K·IQR) vs the SERIES' own daily history. The raw
    ``gaps_gt_5s`` / ``max_stale_secs`` counts are left untouched — only the label changes; quote
    sparsity stays as a liquidity signal. Idempotent (strips any prior gap warning first), gaps
    only ever upgrade PASS→WARN (never touch FAIL), and status is recomputed."""
    for sd in cov.get("series", {}).values():
        days = sd.get("days", {})
        rates = {d: _gap_rate(r) for d, r in days.items()}
        fence = None
        vals = sorted(rates.values())
        if len(vals) >= GAP_OUTLIER_MIN_DAYS:
            q1, q3 = (float(x) for x in np.percentile(vals, [25, 75]))
            iqr = q3 - q1
            if iqr > 0:
                fence = q3 + GAP_OUTLIER_IQR_K * iqr
        for d, rec in days.items():
            rec["warnings"] = [w for w in rec.get("warnings", [])
                               if not (w.startswith("stale stretch") or w.startswith("gap rate")
                                       or "gap(s)>5s" in w)]
            stale = rec.get("max_stale_secs")
            if stale is not None and stale > GAP_STALE_WARN_SECS:
                rec["warnings"].append(f"stale stretch {stale:.0f}s > {GAP_STALE_WARN_SECS:.0f}s")
            if fence is not None and rates[d] > fence:
                rec["warnings"].append(f"gap rate {rates[d]:.2f}/window outlier for series (> {fence:.2f})")
            rec["status"] = ("FAIL" if rec.get("problems")
                             else ("WARN" if rec.get("warnings") else "PASS"))


def _sweep_tmp(specs: list[dict[str, Any]]) -> None:
    """Remove any stale ``.tmp`` from a killed download/compress for these specs. Sweeps the
    exact per-spec paths (never globs the directory — it holds ~257k files at steady state)."""
    for s in specs:
        _safe_unlink(Path(str(s["raw"]) + ".tmp"))
        _safe_unlink(Path(str(s["zst"]) + ".tmp"))


# --------------------------------------------------------------------------- #
# coverage.json (single source of truth, written atomically after each day)
# --------------------------------------------------------------------------- #


def _load_coverage(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / COVERAGE_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _init_coverage(
    existing: dict[str, Any] | None, *, series: list[str], channels: list[str], outcomes: list[str],
    cap_bytes: int, windows: list[dict[str, Any]], projection: dict[str, Any],
) -> dict[str, Any]:
    existing_days: dict[str, dict[str, Any]] = {}
    if existing:
        for sname, sd in existing.get("series", {}).items():
            existing_days[sname] = sd.get("days", {}) or {}
    cov: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": (existing or {}).get("generated_utc") or _now_utc(),
        "updated_utc": _now_utc(),
        "cap_bytes": int(cap_bytes), "channels": channels, "outcomes": outcomes,
        "projection": projection, "series": {}, "totals": {}, "alerts": [],
        "run": {"started_utc": _now_utc(), "aborted": None, "downloads_remaining": None, "peak_rss_mb": None},
    }
    for sname in series:
        sw = [w for w in windows if w["series"] == sname]
        days = sorted({w["open_day"] for w in sw})
        prior = existing_days.get(sname, {})
        cov["series"][sname] = {
            "asset": sw[0]["asset"] if sw else None, "dur": sw[0]["dur"] if sw else None,
            "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
            "expected_days": len(days),
            "expected_files": sum(len(w["channels"]) * len(outcomes) for w in sw),
            "days": {d: prior[d] for d in days if d in prior},
            "status_rollup": {},
        }
    return cov


def _recompute_rollups(cov: dict[str, Any]) -> None:
    alerts: list[dict[str, Any]] = []
    tot_done = tot_zst = 0
    for sname, sd in cov["series"].items():
        roll = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for day, rec in sd["days"].items():
            st = rec.get("status", "PENDING")
            roll[st] = roll.get(st, 0) + 1
            tot_done += rec.get("downloaded_files", 0)
            tot_zst += rec.get("zst_bytes", 0)
            if st == "FAIL":
                alerts.append({"level": "FAIL", "series": sname, "day": day,
                               "detail": "; ".join(rec.get("problems", [])) or "FAIL"})
            elif st == "WARN":
                alerts.append({"level": "WARN", "series": sname, "day": day,
                               "detail": "; ".join(rec.get("warnings", [])) or "WARN"})
            if rec.get("expected_files", 0) > 0 and rec.get("downloaded_files", 0) == 0:
                alerts.append({"level": "MISSING", "series": sname, "day": day,
                               "detail": "MISSING DAY — zero files for a day the catalog says has data"})
        roll["PENDING"] = max(0, sd["expected_days"] - len(sd["days"]))
        sd["status_rollup"] = roll
    cov["totals"] = {
        "expected_files": sum(sd["expected_files"] for sd in cov["series"].values()),
        "done_files": tot_done, "zst_bytes": tot_zst, "zst_human": human_bytes(tot_zst),
    }
    cov["alerts"] = alerts


def _write_coverage(out_dir: Path, cov: dict[str, Any]) -> None:
    cov["updated_utc"] = _now_utc()
    _recompute_rollups(cov)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / COVERAGE_NAME
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cov, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, p)


# --------------------------------------------------------------------------- #
# HTML report (rendered from coverage.json; colored PASS/WARN/FAIL)
# --------------------------------------------------------------------------- #


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_report(cov: dict[str, Any]) -> str:
    colour = {"PASS": "#1a7f37", "WARN": "#9a6700", "FAIL": "#cf222e", "PENDING": "#57606a"}
    proj = cov.get("projection", {})
    run = cov.get("run", {})
    alerts = cov.get("alerts", [])
    n_fail = sum(1 for a in alerts if a["level"] in ("FAIL", "MISSING"))
    overall = "ALERT" if n_fail else ("PARTIAL" if run.get("aborted") else "GO")
    overall_colour = colour["FAIL"] if overall == "ALERT" else (colour["WARN"] if overall == "PARTIAL" else colour["PASS"])
    tot = cov.get("totals", {})

    banner = ""
    if alerts:
        items = "".join(
            f"<li style='color:{colour.get(a['level'], '#cf222e')}'><b>{_esc(a['level'])}</b> "
            f"{_esc(a['series'])} {_esc(a['day'])} — {_esc(a['detail'])}</li>" for a in alerts[:400])
        more = f"<li>… and {len(alerts) - 400} more</li>" if len(alerts) > 400 else ""
        banner = (f"<div style='background:#fff0f0;border:1px solid #cf222e;border-radius:6px;padding:10px 14px;margin:1rem 0'>"
                  f"<b>{len(alerts)} alert(s)</b><ul style='margin:6px 0'>{items}{more}</ul></div>")
    if run.get("aborted"):
        banner += (f"<div style='background:#fff8c5;border:1px solid #d4a72c;border-radius:6px;padding:8px 12px;margin:1rem 0'>"
                   f"<b>Run aborted:</b> {_esc(run['aborted'])} — resume by re-running the same command.</div>")

    dropped = proj.get("dropped", [])
    shrink = ""
    if dropped:
        shrink = (f"<p style='color:{colour['WARN']}'><b>Budget shrink:</b> dropped {len(dropped)} oldest 15m day(s) "
                  f"to fit {_esc(proj.get('cap_human'))}: "
                  + ", ".join(f"{_esc(d['series'])} {_esc(d['day'])}" for d in dropped[:10])
                  + (" …" if len(dropped) > 10 else "") + "</p>")

    series_tables = []
    cols = [("day", "day"), ("expected_files", "exp"), ("downloaded_files", "got"),
            ("missing_files", "missing"), ("rows_total", "rows"), ("dup_trade_ids", "dup"),
            ("nonmonotonic_ts", "non-mono"), ("gaps_gt_5s", "gaps>5s"), ("max_stale_secs", "max-stale-s"),
            ("zst_bytes", "zst"), ("status", "status")]
    for sname, sd in cov.get("series", {}).items():
        roll = sd.get("status_rollup", {})
        head = "".join(f"<th>{_esc(lbl)}</th>" for _k, lbl in cols)
        body = []
        for day in sorted(sd.get("days", {})):
            rec = sd["days"][day]
            tds = []
            for k, _lbl in cols:
                if k == "day":
                    v = day
                elif k == "status":
                    st = rec.get("status", "PENDING")
                    tds.append(f"<td style='color:{colour.get(st, '#000')};font-weight:600'>{_esc(st)}</td>")
                    continue
                elif k == "zst_bytes":
                    v = human_bytes(rec.get("zst_bytes", 0))
                else:
                    v = rec.get(k, "")
                tds.append(f"<td>{_esc(v)}</td>")
            body.append(f"<tr>{''.join(tds)}</tr>")
        series_tables.append(
            f"<h3>{_esc(sname)} <small style='color:#57606a'>"
            f"({_esc(sd.get('first_day'))}→{_esc(sd.get('last_day'))}, {sd.get('expected_days', 0)} days · "
            f"PASS {roll.get('PASS', 0)} / WARN {roll.get('WARN', 0)} / FAIL {roll.get('FAIL', 0)} / "
            f"PENDING {roll.get('PENDING', 0)})</small></h3>"
            f"<table><tr>{head}</tr>{''.join(body)}</table>")

    return f"""<!doctype html><meta charset=utf-8><title>Telonex full pull</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;max-width:1100px}}
table{{border-collapse:collapse;width:100%;margin:.4rem 0 1.2rem;font-size:13px}}
td,th{{border:1px solid #d0d7de;padding:4px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
small{{font-weight:400}}h1 span{{color:{overall_colour}}}</style>
<h1>Telonex full pull — <span>{overall}</span></h1>
<p>{_esc(cov.get('channels'))} × {_esc(cov.get('outcomes'))} · updated {_esc(cov.get('updated_utc'))}
 · X-Downloads-Remaining(min) {_esc(run.get('downloads_remaining'))} · peak RSS {_esc(run.get('peak_rss_mb'))} MB</p>
<p><b>Projection:</b> raw {_esc(proj.get('raw_human'))} · zst {_esc(proj.get('zst_human'))} vs cap {_esc(proj.get('cap_human'))}
 — {'FITS' if proj.get('fits_cap') else 'OVER CAP'}. Done: {tot.get('done_files', 0)}/{tot.get('expected_files', 0)} files, {_esc(tot.get('zst_human'))} on disk.</p>
{shrink}
{banner}
{''.join(series_tables)}
"""


def _render_and_write_report(out_dir: Path, cov: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / REPORT_NAME
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(render_report(cov), encoding="utf-8")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# Progress + quota gate
# --------------------------------------------------------------------------- #


class _Progress:
    def __init__(self, total: int, projected_zst: int, progress: Callable[[str], None] | None):
        self.total = total
        self.done = 0
        self.zst_bytes = 0
        self.fetched = 0
        self.start = time.monotonic()
        self.min_remaining: int | None = None
        self.progress = progress
        self.projected_zst = projected_zst

    def note_result(self, r: dict[str, Any]) -> None:
        st = r.get("status")
        if st in ("downloaded", "present", "finalized"):
            self.done += 1
            self.zst_bytes += int(r.get("zst_bytes", 0))
        if st in ("downloaded", "present"):
            self.fetched += 1
        rem = r.get("remaining")
        if rem is not None:
            self.min_remaining = rem if self.min_remaining is None else min(self.min_remaining, rem)
        if self.progress and self.done and self.done % 500 == 0:
            self.emit()

    def add_done(self, files: int, zst_bytes: int) -> None:
        self.done += files
        self.zst_bytes += zst_bytes

    def line(self) -> str:
        el = max(1e-6, time.monotonic() - self.start)
        pct = 100 * self.done / self.total if self.total else 0.0
        rate = self.fetched / el if self.fetched else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        eta_s = "?" if eta == float("inf") else f"{int(eta // 3600)}h{int((eta % 3600) // 60):02d}m"
        return (f"{self.done}/{self.total} ({pct:.1f}%) · {human_bytes(self.zst_bytes)}/"
                f"{human_bytes(self.projected_zst)} · {int(el)}s · ETA {eta_s} · remaining "
                f"{self.min_remaining if self.min_remaining is not None else 'unlimited'}")

    def emit(self) -> None:
        if self.progress:
            self.progress(self.line())


def _quota_gate(
    groups: list[tuple[str, str, list[dict[str, Any]]]], *, api_key: str, fetch: tx.Fetch,
    remaining_expected: int, ignore: bool,
) -> dict[str, Any]:
    sample_url = None
    for _s, day, ws in groups:
        for w in ws:
            for ch in w["channels"]:
                sample_url = tx.build_download_url(tx.POLYMARKET, ch, day,
                                                   {"slug": w["slug"], "outcome": "Up"})
                break
            if sample_url:
                break
        if sample_url:
            break
    if sample_url is None:
        return {"stop": False, "remaining": None, "reason": None}
    try:
        remaining = tx.probe_remaining(api_key, sample_url, fetch=fetch)
    except tx.TelonexDownloadLimit as e:
        return {"stop": not ignore, "remaining": e.remaining,
                "reason": f"download limit already hit (X-Downloads-Remaining={e.remaining})"}
    if remaining is None or remaining >= remaining_expected:
        return {"stop": False, "remaining": remaining, "reason": None}
    return {"stop": not ignore, "remaining": remaining,
            "reason": (f"X-Downloads-Remaining={remaining} < {remaining_expected} files still needed. "
                       "Telonex has no bulk endpoint (one file per asset per day), so the only lever is a "
                       "reduced date range (--start). Not starting a partial pull.")}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def pull(
    paths: Paths, *, series: list[str], channels: list[str], outcomes: list[str],
    start: str | None = None, end: str | None = None, cap_bytes: int, jobs: int = DEFAULT_JOBS,
    level: int = DEFAULT_ZSTD_LEVEL, roundtrip_sample: int = DEFAULT_ROUNDTRIP_SAMPLE,
    mirror_sample: int = DEFAULT_MIRROR_SAMPLE, jitter_ms: int = 0,
    disk_factor: float = DEFAULT_DISK_FACTOR, disk_floor_bytes: int = int(DEFAULT_DISK_FLOOR_GB * 1024 ** 3),
    report_every: int = DEFAULT_REPORT_EVERY, dry_run: bool = False, ignore_quota_gate: bool = False,
    max_days: int | None = None, delete_raw: bool = True, refresh_catalog: bool = False,
    api_key: str | None = None, fetch: tx.Fetch = tx.urllib_fetch,
    progress: Callable[[str], None] | None = None, rng: random.Random | None = None,
) -> dict[str, Any]:
    """Run the full pull (or, with ``dry_run``, only enumerate + project + gate)."""
    telonex_dir = paths.telonex_dir
    out_dir = paths.out_dir / "telonex"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_pull_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rng = rng or random.Random(20260708)

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    # 1. Catalog (public, free) + enumerate every in-scope window in one scan.
    if refresh_catalog:
        _safe_unlink(tx.catalog_path(telonex_dir))
        note("refreshing markets catalog (public, no credit)")
    cat = tx.ensure_markets_catalog(telonex_dir, fetch=fetch)
    windows = tx.enumerate_pull_windows(cat, series, channels=channels)
    if start:
        windows = [w for w in windows if w["open_day"] >= start]
    if end:
        windows = [w for w in windows if w["open_day"] <= end]
    if not windows:
        note("no in-scope windows found in the catalog for the requested series/date range")
        return {"status": "no_windows", "windows": 0}
    last_day = max(w["open_day"] for w in windows)
    recent_days = set(sorted({w["open_day"] for w in windows})[-RECENT_DAY_LAG:])

    # 2. Projection + budget (reported BEFORE any download).
    ref = _ref_series(series)
    per_zst, per_raw, ref_dur = _measure_per_window(telonex_dir, channels, ref)
    projection = project_pull(windows, per_zst, per_raw, ref_dur, cap_bytes=cap_bytes)
    kept, shrink = apply_budget_shrink(projection, windows, per_zst, ref_dur, len(outcomes), cap_bytes=cap_bytes)
    if shrink["shrink_applied"]:
        windows = kept
        projection = project_pull(windows, per_zst, per_raw, ref_dur, cap_bytes=cap_bytes)
        projection["shrink_applied"] = True
        projection["dropped"] = shrink["dropped"]
        note(f"budget shrink: dropped {len(shrink['dropped'])} oldest 15m day(s), "
             f"freed {human_bytes(shrink['freed_bytes'])}; still over cap: {shrink['still_over']}")

    groups = _group_by_day(windows)
    if max_days is not None:
        groups = groups[:max_days]
        keep = {(s, d) for s, d, _ in groups}
        windows = [w for w in windows if (w["series"], w["open_day"]) in keep]

    total_expected = sum(len(w["channels"]) * len(outcomes) for w in windows)
    note(f"enumerated {len(windows)} windows across {len(series)} series → {total_expected} expected files "
         f"({len(groups)} series-days), catalog last day {last_day}")
    for s, det in projection["series_detail"].items():
        note(f"  {s:18s} {det['windows']:>6} windows  raw~{det['est_raw_human']:>9}  zst~{det['est_zst_human']:>9}")
    note(f"PROJECTION: raw ~{projection['raw_human']} · zst ~{projection['zst_human']} vs cap "
         f"{projection['cap_human']} → {'FITS' if projection['fits_cap'] else 'OVER CAP'}")

    # 3. coverage skeleton (merges any prior run's day records for resume).
    cov = _init_coverage(_load_coverage(out_dir), series=series, channels=channels, outcomes=outcomes,
                         cap_bytes=cap_bytes, windows=windows, projection=projection)
    cov["run"]["jobs"] = jobs

    # remaining = expected minus already-finalized on disk (for the quota gate + progress seed).
    already_done = sum(sd.get("downloaded_files", 0)
                       for s in cov["series"].values() for sd in s["days"].values())
    remaining_expected = max(0, total_expected - already_done)

    prog = _Progress(total_expected, projection["zst_bytes"], progress)
    prog.add_done(0, 0)

    # 4. Disk preflight (up-front). Steady-state peak = accumulated zst + one day's raw.
    one_day_raw = _max_group_raw(groups, per_raw, ref_dur)
    if not dry_run:
        need = projection["zst_bytes"] + one_day_raw + 5 * 1024 ** 3
        free = tx.free_disk_bytes(telonex_dir)
        note(f"disk preflight: free {human_bytes(free)}, need ~{human_bytes(int(need))} "
             f"(zst {projection['zst_human']} + one-day raw {human_bytes(one_day_raw)} + margin)")
        if free < need:
            cov["run"]["aborted"] = "disk"
            _write_coverage(out_dir, cov)
            _render_and_write_report(out_dir, cov)
            note("ABORTED — insufficient free disk for the projected store.")
            return {"status": "aborted_disk", "free": free, "need": int(need)}

    # 5. Quota gate (STOP before the bulk loop if the plan can't cover it).
    if api_key and remaining_expected > 0:
        gate = _quota_gate(groups, api_key=api_key, fetch=fetch,
                           remaining_expected=remaining_expected, ignore=ignore_quota_gate)
        cov["run"]["downloads_remaining"] = gate["remaining"]
        prog.min_remaining = gate["remaining"]
        note(f"quota: X-Downloads-Remaining={gate['remaining']} vs {remaining_expected} files still needed "
             f"→ {'STOP' if gate['stop'] else 'OK'}")
        if gate["stop"] and not dry_run:
            cov["run"]["aborted"] = "quota"
            _write_coverage(out_dir, cov)
            _render_and_write_report(out_dir, cov)
            note(f"STOP — {gate['reason']}")
            return {"status": "aborted_quota", "remaining": gate["remaining"], "reason": gate["reason"]}
    elif not api_key and not dry_run:
        return {"status": "no_key"}

    # 6. dry-run stops here (projection + gate written to the report).
    if dry_run:
        _write_coverage(out_dir, cov)
        rp = _render_and_write_report(out_dir, cov)
        note(f"dry-run: wrote {out_dir / COVERAGE_NAME} and {rp} (downloaded nothing)")
        return {"status": "dry_run", "windows": len(windows), "expected_files": total_expected,
                "projection": projection, "downloads_remaining": cov["run"]["downloads_remaining"]}

    # 7. Per-(series, day) loop.
    aborted: str | None = None
    finalized_since_report = 0
    for s, day_str, day_windows in groups:
        sd = cov["series"][s]
        prior = sd["days"].get(day_str)
        specs: list[dict[str, Any]] = []
        for w in day_windows:
            specs.extend(_expected_specs(telonex_dir, w, outcomes))

        # 7a. Reconcile-on-entry: a PASS/WARN day is trusted; finish any stray raw delete + skip.
        if prior and prior.get("status") in ("PASS", "WARN"):
            for spec in specs:
                if spec["raw"].exists() and spec["zst"].exists():
                    _safe_unlink(spec["raw"])
            prior["raw_deleted"] = True
            prog.add_done(prior.get("downloaded_files", 0), prior.get("zst_bytes", 0))
            continue

        # 7b. Periodic disk check.
        free = tx.free_disk_bytes(telonex_dir)
        if free < max(disk_floor_bytes, int(disk_factor * one_day_raw)):
            aborted = "disk"
            note(f"ABORTED (disk) at {s} {day_str}: free {human_bytes(free)}")
            break

        _sweep_tmp(specs)
        todo = [sp for sp in specs if not tx.zst_finalized(sp["raw"])]
        results, day_aborted, limit_msg = _execute_files(
            todo, jobs=jobs, api_key=api_key, fetch=fetch, level=level, jitter_ms=jitter_ms,
            on_result=prog.note_result)
        if day_aborted:
            aborted = "limit"
            cov["run"]["limit_message"] = limit_msg
            note(f"ABORTED (download limit) at {s} {day_str}: {limit_msg}")
            break

        status_by_key = {(r["spec"]["channel"], r["spec"]["slug"], r["spec"]["outcome"]): r["status"]
                         for r in results}
        rec = _day_record(specs, status_by_key, expected_windows=len(day_windows), scratch=scratch,
                          roundtrip_sample=roundtrip_sample, mirror_sample=mirror_sample, rng=rng)
        st, problems, warnings = _classify_day(rec, recent_day=(day_str in recent_days))
        rec["status"], rec["problems"], rec["warnings"] = st, problems, warnings

        # 7c. Checkpoint coverage BEFORE deleting raw (invariant: PASS ⇒ zst on disk).
        sd["days"][day_str] = rec
        cov["run"]["downloads_remaining"] = prog.min_remaining
        _write_coverage(out_dir, cov)

        if st in ("PASS", "WARN") and delete_raw:
            for spec in specs:
                if spec["raw"].exists() and spec["zst"].exists():
                    _safe_unlink(spec["raw"])
            rec["raw_deleted"] = True

        finalized_since_report += 1
        if st != "PASS":
            note(f"{st}: {s} {day_str} — {'; '.join(problems + warnings)}")
        if finalized_since_report >= report_every:
            _render_and_write_report(out_dir, cov)
            finalized_since_report = 0
            prog.emit()

    # 8. Finalize.
    cov["run"]["aborted"] = aborted
    cov["run"]["downloads_remaining"] = prog.min_remaining
    cov["run"]["peak_rss_mb"] = procmem.peak_rss_mb()
    _apply_gap_labels(cov)         # series-relative gap WARN rule over the full run
    _write_coverage(out_dir, cov)  # recomputes rollups + alerts (coverage.json is the authoritative record)
    rp = _render_and_write_report(out_dir, cov)
    n_fail = sum(1 for a in cov["alerts"] if a["level"] in ("FAIL", "MISSING"))
    status = "aborted_" + aborted if aborted else ("completed_with_alerts" if n_fail else "ok")
    note(f"{status}: {cov['totals']['done_files']}/{cov['totals']['expected_files']} files, "
         f"{cov['totals']['zst_human']} on disk; {n_fail} FAIL/MISSING alert(s). report → {rp}")
    return {"status": status, "totals": cov["totals"], "alerts": len(cov["alerts"]),
            "fail_alerts": n_fail, "report": str(rp)}


def _max_group_raw(groups, per_raw: dict[str, float], ref_dur: str) -> int:
    ref_secs = tx.duration_seconds(ref_dur)
    best = 0.0
    for _s, _d, ws in groups:
        scale = tx.duration_seconds(ws[0]["dur"]) / ref_secs if ws else 1.0
        gb = sum(sum(per_raw.get(ch, 0) for ch in w["channels"]) * scale * 2 for w in ws)
        best = max(best, gb)
    return int(best)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print(msg: str) -> None:
    print(f"[telonex_pull] {msg}")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_pull")
    parser.add_argument("--series", default=DEFAULT_SERIES, help=f"comma list (default {DEFAULT_SERIES})")
    parser.add_argument("--channels", default=DEFAULT_CHANNELS, help="comma list of quotes,trades")
    parser.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    parser.add_argument("--start", default=None, help="clamp to windows opening on/after this UTC day (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="clamp to windows opening on/before this UTC day")
    parser.add_argument("--cap-gb", type=float, default=DEFAULT_CAP_GB, help=f"storage cap in GiB (default {DEFAULT_CAP_GB})")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help=f"concurrent downloads (default {DEFAULT_JOBS}, polite)")
    parser.add_argument("--jitter-ms", type=int, default=0, help="optional per-request sleep (politeness)")
    parser.add_argument("--zstd-level", type=int, default=DEFAULT_ZSTD_LEVEL, help=f"default {DEFAULT_ZSTD_LEVEL}")
    parser.add_argument("--roundtrip-sample", type=int, default=DEFAULT_ROUNDTRIP_SAMPLE,
                        help="random .zst roundtripped per day before deleting raw")
    parser.add_argument("--mirror-sample", type=int, default=DEFAULT_MIRROR_SAMPLE,
                        help="Up/Down mirror windows checked per day (0 = off)")
    parser.add_argument("--disk-factor", type=float, default=DEFAULT_DISK_FACTOR)
    parser.add_argument("--disk-floor-gb", type=float, default=DEFAULT_DISK_FLOOR_GB)
    parser.add_argument("--report-every", type=int, default=DEFAULT_REPORT_EVERY)
    parser.add_argument("--dry-run", action="store_true", help="enumerate + project + quota only; download nothing")
    parser.add_argument("--ignore-quota-gate", action="store_true", help="proceed even if the quota gate would STOP")
    parser.add_argument("--max-days", type=int, default=None, help="process at most N series-days (debug)")
    parser.add_argument("--no-delete-raw", action="store_true", help="keep raw .parquet alongside .zst (debug)")
    parser.add_argument("--refresh-catalog", action="store_true", help="re-download the markets catalog first (public)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    series = [s.strip().lower() for s in args.series.split(",") if s.strip()]
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]

    bad_series = [s for s in series if "-updown-1h" in s or s.endswith("-1h")]
    if bad_series:
        _print(f"1h is out of scope (slug-format issue, separate task): {bad_series}. Remove them from --series.")
        return 1
    bad_channels = [c for c in channels if c not in ALLOWED_CHANNELS]
    if bad_channels:
        _print(f"only {ALLOWED_CHANNELS} are in scope (book_snapshot_full blows the budget): refused {bad_channels}.")
        return 1
    for d in (args.start, args.end):
        if d:
            try:
                date.fromisoformat(d)
            except ValueError:
                _print(f"bad --start/--end date {d!r}: expected YYYY-MM-DD")
                return 1

    api_key = None
    try:
        api_key = tx.load_api_key()
    except tx.TelonexError as e:
        if not args.dry_run:
            _print(str(e))
            return 1
        _print(f"{e} — dry-run continues with projection only (no quota probe).")

    _print(f"series={','.join(series)}  channels={','.join(channels)}  outcomes={','.join(outcomes)}")
    _print(f"store={paths.telonex_dir}  out={paths.out_dir / 'telonex'}  jobs={args.jobs}  "
           f"level={args.zstd_level}  cap={args.cap_gb} GiB  dry_run={args.dry_run}")

    summary = pull(
        paths, series=series, channels=channels, outcomes=outcomes, start=args.start, end=args.end,
        cap_bytes=int(args.cap_gb * 1024 ** 3), jobs=args.jobs, level=args.zstd_level,
        roundtrip_sample=args.roundtrip_sample, mirror_sample=args.mirror_sample, jitter_ms=args.jitter_ms,
        disk_factor=args.disk_factor, disk_floor_bytes=int(args.disk_floor_gb * 1024 ** 3),
        report_every=args.report_every, dry_run=args.dry_run, ignore_quota_gate=args.ignore_quota_gate,
        max_days=args.max_days, delete_raw=not args.no_delete_raw, refresh_catalog=args.refresh_catalog,
        api_key=api_key, progress=_print,
    )

    procmem.report_peak_rss("telonex_pull", procmem.peak_rss_mb(), args.max_rss_mb)
    status = summary.get("status")
    ok = status in ("ok", "dry_run")
    _print(f"status: {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
