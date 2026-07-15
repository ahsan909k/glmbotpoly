"""Telonex full-pull coverage cross-check — verify the committed store vs the catalog.

A re-runnable, **network-free** (except the public catalog refresh) integrity pass over
the ``.zst`` store the full pull produced. It re-derives the expected file set from the
markets catalog and confirms every expected ``.zst`` is on disk — flagging missing
days/windows **loudly** — then regenerates ``out/telonex/{coverage.json,
full_pull_report.html}``. Downloads nothing, deletes nothing.

By default it is a fast structural check (present vs expected + zst size), carrying the
per-day integrity numbers (rows / dup trade-ids / non-monotonic / gaps) from the pull's
``coverage.json``. With ``--deep`` it re-reads the actual ``.zst`` contents (decompressing
a sample, or all, per day) and recomputes those numbers independently.

    python -m model_lab.telonex_coverage                 # structural cross-check + report
    python -m model_lab.telonex_coverage --deep          # also re-read the .zst contents
    python -m model_lab.telonex_coverage --deep --sample 5   # ...sampling 5 windows/day
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Callable

from .config import Paths, resolve_paths, stage_parser
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem
from .telonex_pull import (
    ALLOWED_CHANNELS,
    COVERAGE_NAME,
    DEFAULT_CHANNELS,
    DEFAULT_OUTCOMES,
    DEFAULT_SERIES,
    RECENT_DAY_LAG,
    _apply_gap_labels,
    _classify_day,
    _expected_specs,
    _group_by_day,
    _init_coverage,
    _integrity_file,
    _load_coverage,
    _project_placeholder,
    _render_and_write_report,
    _write_coverage,
)

_INTEGRITY_CARRY = ("rows_total", "dup_trade_ids", "nonmonotonic_ts", "gaps_gt_5s", "max_stale_secs",
                    "roundtrip_checked", "roundtrip_ok", "roundtrip_failed",
                    "mirror_frac", "mirror_checked")


def _run_integrity(specs, scratch: Path, jobs: int):
    """Read each spec's integrity (decompress + parse), parallelized — zstd/pyarrow release
    the GIL so threads scale. Returns ``[(spec, integ_or_None), …]`` in input order."""
    if jobs <= 1 or len(specs) <= 1:
        return [(s, _integrity_file(s, scratch)) for s in specs]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        integs = list(ex.map(lambda s: _integrity_file(s, scratch), specs))
    return list(zip(specs, integs))


def _coverage_day_record(
    specs: list[dict[str, Any]], *, expected_windows: int, prior: dict[str, Any] | None,
    deep: bool, sample: int, scratch: Path, rng: random.Random, jobs: int = 1,
) -> dict[str, Any]:
    """Structural present/missing from disk; integrity carried from ``prior`` (or, with
    ``deep``, recomputed by decompressing the ``.zst`` contents)."""
    present = 0
    missing: list[str] = []
    zst_bytes = 0
    present_specs: list[dict[str, Any]] = []
    for s in specs:
        label = f"{s['slug']}/{s['outcome']}/{s['channel']}"
        if tx.zst_sibling(s["raw"]).exists() or s["raw"].exists():
            present += 1
            present_specs.append(s)
            if tx.zst_sibling(s["raw"]).exists():
                zst_bytes += tx.zst_sibling(s["raw"]).stat().st_size
        else:
            missing.append(label)

    rec: dict[str, Any] = {
        "expected_windows": expected_windows, "expected_files": len(specs),
        "downloaded_files": present, "zst_files": present, "missing_files": len(missing),
        "missing_windows": missing[:200], "error_files": [], "empty_files": [],
        "rows_total": 0, "dup_trade_ids": 0, "nonmonotonic_ts": 0, "gaps_gt_5s": 0, "max_stale_secs": 0.0,
        "mirror_frac": None, "mirror_checked": 0, "roundtrip_checked": 0, "roundtrip_ok": 0,
        "roundtrip_failed": [], "zst_bytes": zst_bytes, "raw_deleted": True,
    }
    if prior:
        for k in _INTEGRITY_CARRY:
            if k in prior:
                rec[k] = prior[k]
    if deep and present_specs:
        chosen = present_specs if sample <= 0 or len(present_specs) <= sample else rng.sample(present_specs, sample)
        rows = dup = nonmono = gaps = 0
        max_stale = 0.0
        errors: list[str] = []
        empty: list[str] = []
        for s, integ in _run_integrity(chosen, scratch, jobs):
            if integ is None:
                continue
            if "error" in integ:
                errors.append(f"{s['slug']}/{s['outcome']}/{s['channel']}: read {integ['error']}")
                continue
            rows += integ["rows"]
            dup += integ["dup"]
            nonmono += integ["nonmono"]
            gaps += integ["gaps"]
            max_stale = max(max_stale, integ.get("max_gap_s", 0.0))
            if integ["rows"] == 0:
                empty.append(f"{s['slug']}/{s['outcome']}/{s['channel']}")
        rec.update({"rows_total": rows, "dup_trade_ids": dup, "nonmonotonic_ts": nonmono,
                    "gaps_gt_5s": gaps, "max_stale_secs": round(max_stale, 1),
                    "error_files": errors[:50], "empty_files": empty[:50], "deep_checked": len(chosen)})
    return rec


def coverage(
    paths: Paths, *, series: list[str], channels: list[str], outcomes: list[str],
    deep: bool = False, sample: int = 0, jobs: int = 12, fetch: tx.Fetch = tx.urllib_fetch,
    progress: Callable[[str], None] | None = None, rng: random.Random | None = None,
) -> dict[str, Any]:
    telonex_dir = paths.telonex_dir
    out_dir = paths.out_dir / "telonex"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_coverage_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rng = rng or random.Random(20260708)

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    cat = tx.ensure_markets_catalog(telonex_dir, fetch=fetch)
    windows = tx.enumerate_pull_windows(cat, series, channels=channels)
    if not windows:
        note("no in-scope windows in the catalog")
        return {"status": "no_windows"}

    existing = _load_coverage(out_dir)
    projection = (existing or {}).get("projection") or _project_placeholder(windows, channels, outcomes)
    cov = _init_coverage(existing, series=series, channels=channels, outcomes=outcomes,
                         cap_bytes=int(projection.get("cap_bytes", 0)), windows=windows, projection=projection)
    cov["run"]["aborted"] = None
    cov["run"]["mode"] = "coverage-deep" if deep else "coverage"

    recent_days = set(sorted({w["open_day"] for w in windows})[-RECENT_DAY_LAG:])
    prior_days = {(s, d): rec for s, sd in (existing or {}).get("series", {}).items()
                  for d, rec in (sd.get("days") or {}).items()}

    groups = _group_by_day(windows)
    for gi, (s, day_str, day_windows) in enumerate(groups):
        specs: list[dict[str, Any]] = []
        for w in day_windows:
            specs.extend(_expected_specs(telonex_dir, w, outcomes))
        rec = _coverage_day_record(specs, expected_windows=len(day_windows),
                                   prior=prior_days.get((s, day_str)), deep=deep, sample=sample,
                                   scratch=scratch, rng=rng, jobs=jobs)
        st, problems, warnings = _classify_day(rec, recent_day=(day_str in recent_days))
        rec["status"], rec["problems"], rec["warnings"] = st, problems, warnings
        cov["series"][s]["days"][day_str] = rec
        if deep and (gi + 1) % 50 == 0:
            note(f"deep re-read {gi + 1}/{len(groups)} series-days…")
            _write_coverage(out_dir, cov)   # periodic checkpoint so a long deep pass is resumable-safe

    _apply_gap_labels(cov)   # series-relative gap WARN rule over the full store
    cov["run"]["peak_rss_mb"] = procmem.peak_rss_mb()
    _write_coverage(out_dir, cov)
    rp = _render_and_write_report(out_dir, cov)
    n_fail = sum(1 for a in cov["alerts"] if a["level"] in ("FAIL", "MISSING"))
    tot = cov["totals"]
    note(f"coverage: {tot['done_files']}/{tot['expected_files']} files present ({tot['zst_human']} zst); "
         f"{n_fail} FAIL/MISSING alert(s). report → {rp}")
    for a in cov["alerts"][:40]:
        if a["level"] in ("FAIL", "MISSING"):
            note(f"  {a['level']} {a['series']} {a['day']} — {a['detail']}")
    return {"status": "alerts" if n_fail else "ok", "totals": tot, "fail_alerts": n_fail, "report": str(rp)}


def _print(msg: str) -> None:
    print(f"[telonex_coverage] {msg}")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_coverage")
    parser.add_argument("--series", default=DEFAULT_SERIES)
    parser.add_argument("--channels", default=DEFAULT_CHANNELS)
    parser.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    parser.add_argument("--deep", action="store_true", help="decompress + re-read .zst contents (integrity)")
    parser.add_argument("--sample", type=int, default=0, help="with --deep, files/day to re-read (0 = all)")
    parser.add_argument("--jobs", type=int, default=12, help="parallel decompress+read workers for --deep")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    series = [s.strip().lower() for s in args.series.split(",") if s.strip()]
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    bad = [c for c in channels if c not in ALLOWED_CHANNELS]
    if bad:
        _print(f"only {ALLOWED_CHANNELS} are in scope: refused {bad}.")
        return 1

    _print(f"store={paths.telonex_dir}  deep={args.deep}  series={','.join(series)}")
    result = coverage(paths, series=series, channels=channels, outcomes=outcomes,
                      deep=args.deep, sample=args.sample, jobs=args.jobs, progress=_print)
    procmem.report_peak_rss("telonex_coverage", procmem.peak_rss_mb(), args.max_rss_mb)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
