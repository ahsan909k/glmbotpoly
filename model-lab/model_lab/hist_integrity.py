"""Historical data integrity — validate the downloaded aggTrades store.

For every ``aggtrades-{SYMBOL}-{DAY}.parquet`` under the aggTrades store
(``../data/aggtrades``, written by ``python -m model_lab.hist``) this checks, per
day-file:

- **row count** — non-empty, and matching the manifest's recorded count (catches
  truncation/corruption since download);
- **monotonic timestamps** — ``transact_time`` never decreases (equal
  microseconds are allowed — multiple trades can share one timestamp);
- **no duplicate trade ids** — ``agg_trade_id`` strictly increasing and unique.

Cross-day continuity (a day's first id vs the previous day's last id) is reported
as an informational note — a gap at a missing-day boundary is expected, not a
failure.

Verify:  ``python -m model_lab.hist_integrity``  prints per-day PASS/FAIL and
writes ``out/hist_integrity/report.json``; exits nonzero if any day fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import Paths, resolve_paths, stage_parser
from .io.binance_archive import (
    aggtrades_day_paths,
    day_of_path,
    read_aggtrades_columns,
    read_manifest,
    symbols_present,
)


def _check_day(path: Path, symbol: str, manifest_rows: dict[str, Any]) -> dict[str, Any]:
    """Validate one day-file; returns a report dict with ``ok`` + ``problems``."""
    problems: list[str] = []
    day = day_of_path(path)
    day_str = day.isoformat() if day else path.stem

    df = read_aggtrades_columns(path, ["agg_trade_id", "transact_time"])
    n = len(df)
    first_id = int(df["agg_trade_id"].iloc[0]) if n else None
    last_id = int(df["agg_trade_id"].iloc[-1]) if n else None

    if n == 0:
        problems.append("empty (0 rows)")

    expected = manifest_rows.get(day_str)
    if expected is not None and expected != n:
        problems.append(f"row count {n:,} != manifest {expected:,}")

    if n:
        ids = df["agg_trade_id"].to_numpy()
        ts = df["transact_time"].to_numpy()
        # Monotonic timestamps: non-decreasing (equal microseconds allowed).
        bad_ts = int(np.count_nonzero(ts[1:] < ts[:-1]))
        if bad_ts:
            problems.append(f"{bad_ts:,} non-monotonic timestamp step(s)")
        # agg_trade_id strictly increasing (also rules out adjacent duplicates).
        bad_ids = int(np.count_nonzero(ids[1:] <= ids[:-1]))
        if bad_ids:
            problems.append(f"{bad_ids:,} non-increasing agg_trade_id step(s)")
        # Explicit duplicate check (catches non-adjacent duplicates too).
        dups = n - int(np.unique(ids).size)
        if dups:
            problems.append(f"{dups:,} duplicate agg_trade_id(s)")

    return {
        "symbol": symbol,
        "day": day_str,
        "rows": n,
        "first_id": first_id,
        "last_id": last_id,
        "ok": not problems,
        "problems": problems,
        "continuity_warnings": [],
    }


def integrity(paths: Paths, *, symbols: list[str] | None = None) -> dict[str, Any]:
    """Run integrity checks over the aggTrades store; returns a report dict."""
    hist_dir = paths.hist_dir
    manifest = read_manifest(hist_dir)
    manifest_rows: dict[str, dict[str, int]] = {}
    if manifest:
        for sym, meta in manifest.get("symbols", {}).items():
            manifest_rows[sym] = {d: v.get("rows") for d, v in meta.get("per_day", {}).items()}

    syms = symbols if symbols else symbols_present(hist_dir)

    day_reports: list[dict[str, Any]] = []
    for sym in syms:
        prev_last_id: int | None = None
        for p in aggtrades_day_paths(hist_dir, sym):
            rep = _check_day(p, sym, manifest_rows.get(sym, {}))
            if prev_last_id is not None and rep["first_id"] is not None and rep["first_id"] <= prev_last_id:
                rep["continuity_warnings"].append(
                    f"first id {rep['first_id']:,} <= previous day's last id {prev_last_id:,}"
                )
            if rep["last_id"] is not None:
                prev_last_id = rep["last_id"]
            day_reports.append(rep)

    days_failed = sum(1 for r in day_reports if not r["ok"])
    return {
        "hist_dir": str(hist_dir),
        "symbols": list(syms),
        "days_checked": len(day_reports),
        "days_ok": len(day_reports) - days_failed,
        "days_failed": days_failed,
        "days": day_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "hist_integrity")
    parser.add_argument(
        "--symbols", default=None, help="comma-separated symbols (default: all present)"
    )
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    )

    print(f"[hist_integrity] hist={paths.hist_dir}")
    result = integrity(paths, symbols=symbols)

    for rep in result["days"]:
        status = "PASS" if rep["ok"] else "FAIL"
        line = f"[hist_integrity] {status} {rep['symbol']} {rep['day']} {rep['rows']:>12,} rows"
        if not rep["ok"]:
            line += " :: " + "; ".join(rep["problems"])
        print(line)
        for w in rep["continuity_warnings"]:
            print(f"[hist_integrity]   note {rep['symbol']} {rep['day']}: {w}")

    out_dir = paths.out_dir / "hist_integrity"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"[hist_integrity] {result['days_ok']}/{result['days_checked']} day-files PASS; "
        f"{result['days_failed']} FAIL; report -> {report_path}"
    )
    if result["days_checked"] == 0:
        print("[hist_integrity] no aggTrades day-files found — run `python -m model_lab.hist` first.")
        return 1
    return 0 if result["days_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
