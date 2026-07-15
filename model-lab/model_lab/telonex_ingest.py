"""Telonex trial pull — download ONE day of a crypto up/down series + Binance.

Pulls a single UTC day of a Polymarket ``*-updown-*`` series (all its windows that
actually have data) plus the same day of a Binance symbol, streaming every file to
``../data/telonex/raw`` (16 GB box — nothing is buffered whole). This is the trial
that de-risks the full ~500 GB purchase; the full download is a separate task.

Flow: probe the **public** availability endpoint per window slug (free, no key) to
prune the day's 288 candidate windows to the ones with data → download Binance first
(validates the key + plan) → download one probe window → **disk preflight (hard-abort
if free space < 3× the estimated trial size)** → download the rest. Aborts loudly on
the first HTTP 403 (a free-tier key allows only 5 downloads total).

The API key is read from ``$telonexdata`` / ``.env`` and never printed.

    python -m model_lab.telonex_ingest --trial                    # 2026-07-05 btc-updown-5m + btcusdt
    python -m model_lab.telonex_ingest --trial --day 2026-07-05 --outcomes Up,Down
    python -m model_lab.telonex_ingest --trial --max-windows 24   # a cheap smoke slice
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import Paths, resolve_paths, stage_parser
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem

DEFAULT_DAY = "2026-07-05"
DEFAULT_SERIES = "btc-updown-5m"
DEFAULT_OUTCOMES = "Up,Down"
DEFAULT_CHANNELS = "trades,quotes,book_snapshot_full"
DEFAULT_BINANCE_SYMBOL = "btcusdt"
DEFAULT_BINANCE_CHANNELS = "trades,book_snapshot_25"
DEFAULT_MAX_WINDOWS = 300
DEFAULT_DISK_FACTOR = 3.0


@dataclass
class _Task:
    exchange: str
    channel: str
    date_str: str
    dest: Path
    ident: dict[str, Any]
    label: str


@dataclass
class _Agg:
    downloaded: int = 0
    skipped: int = 0
    no_data: int = 0
    error: int = 0
    plan_restricted: int = 0
    bytes: int = 0
    remaining: int | None = None
    limit_msg: str | None = None
    plan_note: str | None = None
    errors: list[str] = field(default_factory=list)


def _note_remaining(agg: _Agg, remaining: int | None) -> None:
    if remaining is None:
        return
    agg.remaining = remaining if agg.remaining is None else min(agg.remaining, remaining)


def _tally(agg: _Agg, task: _Task, res: dict[str, Any], progress) -> None:
    status = res.get("status", "error")
    if status == "downloaded":
        agg.downloaded += 1
        agg.bytes += int(res.get("bytes", 0))
    elif status == "skipped":
        agg.skipped += 1
        agg.bytes += int(res.get("bytes", 0))
    elif status == "no_data":
        agg.no_data += 1
    elif status == "plan_restricted":
        agg.plan_restricted += 1
        if agg.plan_note is None:
            agg.plan_note = res.get("detail")
    else:
        agg.error += 1
        if res.get("error"):
            agg.errors.append(f"{task.label}: {res['error']}")
    _note_remaining(agg, res.get("remaining"))
    if progress:
        progress(task, res)


def _run_one(task: _Task, api_key: str, fetch: tx.Fetch) -> tuple[_Task, dict[str, Any]]:
    try:
        res = tx.download_asset_day(
            task.exchange, task.channel, task.date_str, task.dest,
            api_key=api_key, fetch=fetch, **task.ident,
        )
        return task, res
    except tx.TelonexPlanRestricted as e:  # exchange not on plan — skip it, don't abort
        return task, {"status": "plan_restricted", "bytes": 0, "remaining": None, "detail": str(e)}
    except tx.TelonexDownloadLimit:
        raise
    except Exception as e:  # noqa: BLE001 — one file's failure isn't fatal (unlike a 403)
        return task, {"status": "error", "bytes": 0, "remaining": None, "error": str(e)}


def _execute(tasks: list[_Task], *, jobs: int, api_key: str, fetch: tx.Fetch, agg: _Agg, progress) -> bool:
    """Run download tasks; return True if aborted by a download-limit (403)."""
    if not tasks:
        return False
    if jobs <= 1 or len(tasks) <= 1:
        for t in tasks:
            try:
                tt, res = _run_one(t, api_key, fetch)
            except tx.TelonexDownloadLimit as e:
                agg.limit_msg = str(e)
                return True
            _tally(agg, tt, res, progress)
        return False

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run_one, t, api_key, fetch): t for t in tasks}
        for fut in as_completed(futs):
            try:
                tt, res = fut.result()
            except tx.TelonexDownloadLimit as e:
                agg.limit_msg = str(e)
                for f in futs:
                    f.cancel()
                return True
            _tally(agg, tt, res, progress)
    return False


def _probe_windows(
    slugs: list[str], *, day: date, channels: list[str], jobs: int, fetch: tx.Fetch, progress,
) -> tuple[list[dict[str, Any]], int, int]:
    """Availability-probe (public) each window slug; keep those with data covering ``day``.

    Returns ``(windows_with_data, n_no_data, n_error)``. Each kept entry carries the
    join keys ``slug / open_epoch / open_ms / up_asset_id / market_id`` and the set of
    requested channels that cover the day.
    """
    def probe(slug: str) -> dict[str, Any]:
        try:
            info = tx.availability(tx.POLYMARKET, slug=slug, outcome="Up", fetch=fetch)
        except tx.TelonexNotFound:
            return {"slug": slug, "status": "no_data"}
        except tx.TelonexError as e:
            return {"slug": slug, "status": "error", "error": str(e)}
        chans = info.get("channels", {}) if isinstance(info, dict) else {}
        covered = [c for c in channels if tx.channel_covers_day(chans.get(c), day)]
        if not covered:
            return {"slug": slug, "status": "no_data"}
        epoch = int(slug.rsplit("-", 1)[1])
        return {
            "slug": slug,
            "status": "ok",
            "open_epoch": epoch,
            "open_ms": epoch * 1000,
            "up_asset_id": str(info.get("asset_id", "")),
            "market_id": str(info.get("market_id", "")),
            "covered_channels": covered,
        }

    results: list[dict[str, Any]] = []
    if jobs <= 1 or len(slugs) <= 1:
        for s in slugs:
            results.append(probe(s))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(probe, s) for s in slugs]
            for fut in as_completed(futs):
                results.append(fut.result())

    kept = sorted((r for r in results if r["status"] == "ok"), key=lambda r: r["open_epoch"])
    n_no_data = sum(1 for r in results if r["status"] == "no_data")
    n_error = sum(1 for r in results if r["status"] == "error")
    if progress:
        progress(f"availability: {len(kept)} windows with data, {n_no_data} empty, {n_error} errors")
    return kept, n_no_data, n_error


def _window_tasks(
    telonex_dir: Path, window: dict[str, Any], *, day_str: str, outcomes: list[str], channels: list[str],
) -> list[_Task]:
    tasks: list[_Task] = []
    slug = window["slug"]
    for outcome in outcomes:
        for channel in channels:
            dest = tx.polymarket_raw_path(telonex_dir, channel, slug, outcome, day_str)
            tasks.append(_Task(
                exchange=tx.POLYMARKET, channel=channel, date_str=day_str, dest=dest,
                ident={"slug": slug, "outcome": outcome}, label=f"{slug}/{outcome}/{channel}",
            ))
    return tasks


def telonex_ingest(
    paths: Paths, *, day: date, series: str = DEFAULT_SERIES, outcomes: list[str] | None = None,
    channels: list[str] | None = None, binance_symbol: str = DEFAULT_BINANCE_SYMBOL,
    binance_channels: list[str] | None = None, max_windows: int = DEFAULT_MAX_WINDOWS,
    jobs: int = 4, disk_factor: float = DEFAULT_DISK_FACTOR, discover: str = "auto",
    api_key: str | None = None, fetch: tx.Fetch = tx.urllib_fetch,
    progress: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Download one trial day of ``series`` + ``binance_symbol`` into ``paths.telonex_dir``.

    Returns a summary dict (never containing the key). ``status`` is ``ok``,
    ``no_windows``, ``aborted_disk``, or ``aborted_limit``.
    """
    outcomes = outcomes or ["Up", "Down"]
    channels = channels or DEFAULT_CHANNELS.split(",")
    binance_channels = binance_channels or DEFAULT_BINANCE_CHANNELS.split(",")
    if api_key is None:
        api_key = tx.load_api_key()
    asset, dur = tx.parse_series(series)
    day_str = day.isoformat()
    telonex_dir = paths.telonex_dir

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    # 1. Discover the day's windows-with-data. Prefer the markets catalog (one request,
    #    no throttling, yields asset ids); fall back to per-slug availability probing.
    slugs = tx.updown_slugs(asset, dur, day)
    windows: list[dict[str, Any]] = []
    n_empty = 0
    n_avail_err = 0
    discovery = "availability"
    if discover in ("auto", "catalog"):
        try:
            cat = tx.ensure_markets_catalog(telonex_dir, fetch=fetch)
            windows = tx.windows_from_catalog(cat, asset, dur, day)
            discovery = "catalog"
            n_empty = len(slugs) - len(windows)
            note(f"catalog discovery: {len(windows)} windows with data (of {len(slugs)} candidates)")
        except Exception as e:  # noqa: BLE001 — fall back to availability probing
            note(f"catalog discovery failed ({e}); falling back to availability probing")
            windows = []
    if not windows and discover != "catalog":
        windows, n_empty, n_avail_err = _probe_windows(
            slugs, day=day, channels=channels, jobs=jobs, fetch=fetch, progress=note,
        )
        discovery = "availability"
    capped = False
    if len(windows) > max_windows:
        windows = windows[:max_windows]
        capped = True
        note(f"capping to --max-windows {max_windows} (of the windows with data)")

    summary: dict[str, Any] = {
        "status": "ok", "day": day_str, "series": series, "asset": asset, "duration": dur,
        "outcomes": outcomes, "channels": channels, "binance_symbol": binance_symbol,
        "binance_channels": binance_channels, "candidate_windows": len(slugs),
        "windows_with_data": len(windows), "windows_empty": n_empty,
        "availability_errors": n_avail_err, "capped": capped, "discovery": discovery,
    }
    if not windows:
        summary["status"] = "no_windows"
        return summary

    agg = _Agg()

    # 2. Binance first — cheap, and it validates the key + plan (403 → abort).
    bin_tasks = [
        _Task(exchange=tx.BINANCE, channel=c, date_str=day_str,
              dest=tx.binance_raw_path(telonex_dir, c, binance_symbol, day_str),
              ident={"slug": binance_symbol}, label=f"{binance_symbol}/{c}")
        for c in binance_channels
    ]
    if _execute(bin_tasks, jobs=jobs, api_key=api_key, fetch=fetch, agg=agg, progress=progress):
        summary.update(_finalize(paths, summary, windows, agg, day_str, aborted="aborted_limit"))
        return summary
    binance_bytes = agg.bytes
    summary["binance_downloaded"] = agg.downloaded
    summary["binance_plan_restricted"] = agg.plan_restricted > 0
    if agg.plan_restricted > 0:
        note(f"Binance is NOT on this plan ({agg.plan_note}); continuing with Polymarket only.")

    # 3. Probe one window (all outcomes/channels) to size the trial.
    probe_agg = _Agg()
    _execute(_window_tasks(telonex_dir, windows[0], day_str=day_str, outcomes=outcomes, channels=channels),
             jobs=jobs, api_key=api_key, fetch=fetch, agg=probe_agg, progress=progress)
    if probe_agg.limit_msg:
        agg.limit_msg = probe_agg.limit_msg
        _merge(agg, probe_agg)
        summary.update(_finalize(paths, summary, windows, agg, day_str, aborted="aborted_limit"))
        return summary
    _merge(agg, probe_agg)

    per_window_bytes = probe_agg.bytes
    est_total = per_window_bytes * len(windows) + binance_bytes
    free = tx.free_disk_bytes(telonex_dir)
    summary["est_per_window_bytes"] = per_window_bytes
    summary["est_total_bytes"] = est_total
    summary["free_disk_bytes"] = free
    summary["disk_factor"] = disk_factor
    note(f"disk preflight: est trial {human_bytes(est_total)} "
         f"(~{human_bytes(per_window_bytes)}/window × {len(windows)} + Binance {human_bytes(binance_bytes)}); "
         f"free {human_bytes(free)}; need ≥ {disk_factor}× = {human_bytes(int(disk_factor * est_total))}")
    if est_total > 0 and free < disk_factor * est_total:
        summary["status"] = "aborted_disk"
        summary.update(_finalize(paths, summary, windows, agg, day_str, aborted="aborted_disk"))
        return summary

    # 4. Download the remaining windows.
    rest: list[_Task] = []
    for w in windows[1:]:
        rest.extend(_window_tasks(telonex_dir, w, day_str=day_str, outcomes=outcomes, channels=channels))
    aborted = _execute(rest, jobs=jobs, api_key=api_key, fetch=fetch, agg=agg, progress=progress)

    summary.update(_finalize(
        paths, summary, windows, agg, day_str, aborted="aborted_limit" if aborted else None,
    ))
    return summary


def _merge(dst: _Agg, src: _Agg) -> None:
    dst.downloaded += src.downloaded
    dst.skipped += src.skipped
    dst.no_data += src.no_data
    dst.error += src.error
    dst.plan_restricted += src.plan_restricted
    dst.bytes += src.bytes
    dst.errors.extend(src.errors)
    if dst.plan_note is None:
        dst.plan_note = src.plan_note
    _note_remaining(dst, src.remaining)


def _finalize(paths: Paths, summary: dict[str, Any], windows: list[dict[str, Any]], agg: _Agg,
              day_str: str, *, aborted: str | None) -> dict[str, Any]:
    """Write manifest + trial_meta (join keys, no key material) and return count fields."""
    trial_meta = {
        "day": day_str,
        "series": summary["series"],
        "outcomes": summary["outcomes"],
        "channels": summary["channels"],
        "binance_symbol": summary["binance_symbol"],
        "binance_channels": summary["binance_channels"],
        "windows": [
            {"slug": w["slug"], "open_epoch": w["open_epoch"], "open_ms": w["open_ms"],
             "up_asset_id": w["up_asset_id"], "market_id": w["market_id"]}
            for w in windows
        ],
    }
    manifest = tx.build_manifest(paths.telonex_dir, extra=trial_meta)
    tx.write_manifest(paths.telonex_dir, manifest)
    out: dict[str, Any] = {
        "downloaded": agg.downloaded, "skipped": agg.skipped, "no_data": agg.no_data,
        "errors": agg.error, "plan_restricted": agg.plan_restricted, "plan_note": agg.plan_note,
        "bytes": agg.bytes, "disk_usage_bytes": manifest["disk_usage_bytes"],
        "downloads_remaining": agg.remaining, "error_samples": agg.errors[:10],
    }
    if aborted:
        out["status"] = aborted
        if agg.limit_msg:
            out["limit_message"] = agg.limit_msg
    elif summary.get("status") == "ok":
        out["status"] = "ok"
    return out


def _print_progress(task_or_msg, res: dict[str, Any] | None = None) -> None:
    if res is None:
        print(f"[telonex_ingest] {task_or_msg}")
        return
    status = res.get("status")
    if status == "downloaded":
        print(f"[telonex_ingest] {task_or_msg.label} downloaded ({human_bytes(int(res.get('bytes', 0)))})")
    elif status == "error":
        print(f"[telonex_ingest] {task_or_msg.label} ERROR: {res.get('error')}")
    # 'skipped' / 'no_data' are quiet (there can be thousands).


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_ingest")
    parser.add_argument("--trial", action="store_true", help="run the single-day trial pull")
    parser.add_argument("--day", default=DEFAULT_DAY, help=f"UTC day YYYY-MM-DD (default {DEFAULT_DAY})")
    parser.add_argument("--series", default=DEFAULT_SERIES, help=f"e.g. {DEFAULT_SERIES}")
    parser.add_argument("--outcomes", default=DEFAULT_OUTCOMES, help="comma list (default Up,Down)")
    parser.add_argument("--channels", default=DEFAULT_CHANNELS, help=f"comma list (default {DEFAULT_CHANNELS})")
    parser.add_argument("--binance-symbol", default=DEFAULT_BINANCE_SYMBOL)
    parser.add_argument("--binance-channels", default=DEFAULT_BINANCE_CHANNELS)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS,
                        help=f"cap windows-with-data (default {DEFAULT_MAX_WINDOWS})")
    parser.add_argument("--discover", choices=("auto", "catalog", "availability"), default="auto",
                        help="window discovery: catalog (markets dataset, one request) or per-slug "
                        "availability probing (default auto = catalog if reachable)")
    parser.add_argument("--jobs", type=int, default=4, help="concurrent requests (default 4)")
    parser.add_argument("--disk-factor", type=float, default=DEFAULT_DISK_FACTOR,
                        help=f"abort if free disk < factor × est trial size (default {DEFAULT_DISK_FACTOR})")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    if not args.trial:
        print("[telonex_ingest] pass --trial. The full download is a separate task (this stage "
              "validates one day before the ~500 GB pull).")
        return 1

    try:
        day = date.fromisoformat(args.day)
    except ValueError as e:
        print(f"[telonex_ingest] bad --day: {e}")
        return 1
    try:
        api_key = tx.load_api_key()
    except tx.TelonexError as e:
        print(f"[telonex_ingest] {e}")
        return 1

    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    binance_channels = [c.strip() for c in args.binance_channels.split(",") if c.strip()]

    print(f"[telonex_ingest] day={day}  series={args.series}  outcomes={','.join(outcomes)}")
    print(f"[telonex_ingest] channels={','.join(channels)}  binance={args.binance_symbol}:{','.join(binance_channels)}")
    print(f"[telonex_ingest] store ={paths.telonex_dir}  jobs={args.jobs}  disk-factor={args.disk_factor}")

    summary = telonex_ingest(
        paths, day=day, series=args.series, outcomes=outcomes, channels=channels,
        binance_symbol=args.binance_symbol, binance_channels=binance_channels,
        max_windows=args.max_windows, jobs=args.jobs, disk_factor=args.disk_factor,
        discover=args.discover, api_key=api_key, progress=_print_progress,
    )

    status = summary.get("status")
    print(f"[telonex_ingest] windows: {summary['windows_with_data']} with data / "
          f"{summary['candidate_windows']} candidates ({summary['windows_empty']} empty, "
          f"{summary['availability_errors']} probe errors)")
    if status == "no_windows":
        print("[telonex_ingest] NO windows with data for this series/day — coverage FAIL. "
              "Check --day / --series against the availability endpoint.")
        procmem.report_peak_rss("telonex_ingest", procmem.peak_rss_mb(), args.max_rss_mb)
        return 1
    print(f"[telonex_ingest] files: {summary.get('downloaded', 0)} downloaded, "
          f"{summary.get('skipped', 0)} skipped, {summary.get('no_data', 0)} no-data, "
          f"{summary.get('plan_restricted', 0)} plan-restricted, {summary.get('errors', 0)} errors  "
          f"({human_bytes(summary.get('bytes', 0))})")
    if summary.get("binance_plan_restricted"):
        print("[telonex_ingest] NOTE: Binance is NOT on this plan (Single-Exchange = Polymarket only; "
              "Binance needs Pro). Pulled Polymarket only; Binance is available free from data.binance.vision.")
    if summary.get("downloads_remaining") is not None:
        print(f"[telonex_ingest] X-Downloads-Remaining (min observed): {summary['downloads_remaining']}")
    procmem.report_peak_rss("telonex_ingest", procmem.peak_rss_mb(), args.max_rss_mb)

    if status == "aborted_limit":
        print(f"[telonex_ingest] ABORTED — download limit hit. {summary.get('limit_message', '')}")
        return 1
    if status == "aborted_disk":
        print(f"[telonex_ingest] ABORTED — insufficient disk: free {human_bytes(summary['free_disk_bytes'])} "
              f"< {args.disk_factor}× est {human_bytes(summary['est_total_bytes'])}.")
        return 1
    if summary.get("errors", 0) > 0:
        print(f"[telonex_ingest] completed with {summary['errors']} file errors "
              f"(e.g. {summary.get('error_samples', [])[:3]}).")
    print(f"[telonex_ingest] done → {paths.telonex_dir}  (manifest {tx.manifest_path(paths.telonex_dir).name})")
    print("[telonex_ingest] next: python -m model_lab.telonex_validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
