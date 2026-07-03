"""Historical data — download Binance spot aggTrades daily archives.

Bulk-downloads Binance spot ``aggTrades`` daily archives from data.binance.vision
for BTCUSDT and ETHUSDT into ``../data/aggtrades`` (one zstd-parquet per
symbol-day), verifies SHA-256 checksums, and maintains a ``manifest.json`` of
exact date coverage per symbol. **Resumable & incremental**: re-running only
fetches days not already on disk, so the same command extends the range from the
default 90 days to 18 months.

Verify:  ``python -m model_lab.hist``  (default last 90 days) prints per-day
progress, a missing-day report (recent days lag ~1 day), and total disk usage.
Then ``python -m model_lab.hist_integrity`` validates the store.

    python -m model_lab.hist                       # last 90 days, BTC+ETH
    python -m model_lab.hist --days 540            # extend to ~18 months (same store)
    python -m model_lab.hist --symbols BTCUSDT --start 2025-01-01 --end 2025-03-31
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import date, timedelta

from .config import Paths, resolve_paths, stage_parser
from .io.binance_archive import (
    SYMBOLS,
    DayResult,
    FetchFn,
    build_manifest,
    daterange,
    day_parquet_path,
    default_end,
    download_day,
    has_aggtrades,
    http_fetch,
    human_bytes,
    manifest_path,
    symbols_present,
    write_manifest,
)


def download(
    paths: Paths,
    *,
    symbols: list[str],
    start: date,
    end: date,
    jobs: int = 4,
    verify_checksum: bool = True,
    fetch: FetchFn = http_fetch,
    progress: Callable[[DayResult], None] | None = None,
) -> dict:
    """Download the requested symbol-days (skipping present ones), rebuild the
    manifest, and return a summary dict. Concurrency-safe: ``progress`` is invoked
    from the calling thread only."""
    hist_dir = paths.hist_dir
    days = daterange(start, end)

    # Incremental: only days without a final parquet are actual work.
    work: list[tuple[str, date]] = []
    skipped_existing = 0
    for sym in symbols:
        for d in days:
            if day_parquet_path(hist_dir, sym, d).exists():
                skipped_existing += 1
            else:
                work.append((sym, d))

    results: list[DayResult] = []

    def run_one(sym: str, d: date) -> DayResult:
        return download_day(sym, d, hist_dir, fetch=fetch, verify_checksum=verify_checksum)

    if jobs <= 1 or len(work) <= 1:
        for sym, d in work:
            r = run_one(sym, d)
            results.append(r)
            if progress:
                progress(r)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_one, sym, d): (sym, d) for sym, d in work}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                if progress:
                    progress(r)

    missing: dict[str, list[date]] = {}
    for r in results:
        if r.status == "missing":
            missing.setdefault(r.symbol, []).append(r.day)

    # Manifest covers requested symbols + anything already on disk (no drift).
    all_syms = sorted(set(symbols) | set(symbols_present(hist_dir)))
    manifest = build_manifest(hist_dir, all_syms, missing=missing)
    write_manifest(hist_dir, manifest)

    def count(status: str) -> int:
        return sum(1 for r in results if r.status == status)

    return {
        "downloaded": count("downloaded"),
        "skipped": skipped_existing + count("skipped"),
        "missing": count("missing"),
        "checksum_failed": count("checksum_failed"),
        "error": count("error"),
        "rows_downloaded": sum(r.rows for r in results if r.status == "downloaded"),
        "bytes_downloaded": sum(r.bytes for r in results if r.status == "downloaded"),
        "disk_usage_bytes": manifest["disk_usage_bytes"],
        "disk_usage_human": manifest["disk_usage_human"],
        "results": results,
        "manifest": manifest,
    }


def _print_progress(r: DayResult) -> None:
    if r.status == "downloaded":
        print(
            f"[hist] {r.symbol} {r.day} downloaded {r.rows:>12,} rows "
            f"({human_bytes(r.bytes)}, checksum {r.checksum})"
        )
    elif r.status == "missing":
        print(f"[hist] {r.symbol} {r.day} MISSING (not published yet / 404)")
    elif r.status == "checksum_failed":
        print(f"[hist] {r.symbol} {r.day} CHECKSUM MISMATCH — not stored")
    elif r.status == "error":
        print(f"[hist] {r.symbol} {r.day} ERROR: {r.error}")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "hist")
    parser.add_argument(
        "--symbols", default=None, help="comma-separated symbols (default BTCUSDT,ETHUSDT)"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="how many days back from --end (default 90)"
    )
    parser.add_argument("--start", default=None, help="first day YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", default=None, help="last day YYYY-MM-DD (default: yesterday UTC)")
    parser.add_argument("--jobs", type=int, default=4, help="concurrent downloads (default 4)")
    parser.add_argument(
        "--no-verify-checksum", action="store_true", help="skip SHA-256 checksum verification"
    )
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else list(SYMBOLS)
    )
    if not symbols:
        print("[hist] no symbols requested.")
        return 1

    try:
        end = date.fromisoformat(args.end) if args.end else default_end()
        start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days - 1)
    except ValueError as e:
        print(f"[hist] bad date: {e}")
        return 1
    if start > end:
        print(f"[hist] empty range: start {start} > end {end}")
        return 1

    n_days = (end - start).days + 1
    print(f"[hist] hist   ={paths.hist_dir}")
    print(
        f"[hist] symbols={','.join(symbols)}  range={start}..{end} ({n_days} days)  "
        f"jobs={args.jobs}  verify_checksum={not args.no_verify_checksum}"
    )

    summary = download(
        paths,
        symbols=symbols,
        start=start,
        end=end,
        jobs=args.jobs,
        verify_checksum=not args.no_verify_checksum,
        progress=_print_progress,
    )

    print(
        f"[hist] done: {summary['downloaded']} downloaded, {summary['skipped']} skipped, "
        f"{summary['missing']} missing, {summary['checksum_failed']} checksum-failed, "
        f"{summary['error']} error"
    )
    print(
        f"[hist] store disk usage: {summary['disk_usage_human']} "
        f"({summary['disk_usage_bytes']:,} bytes) in {paths.hist_dir}"
    )
    print(f"[hist] manifest -> {manifest_path(paths.hist_dir)}")

    if summary["checksum_failed"] or summary["error"]:
        return 1
    if not has_aggtrades(paths.hist_dir):
        print("[hist] WARNING: no aggTrades archives on disk after this run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
