"""Telonex Binance depth — Stage 4 (size/quota projection) + Stage 5 (bulk depth pull).

Run ONLY after ``telonex_binance_validate`` reports GO. Downloads the full history of the
chosen depth channel (``book_snapshot_25``) for BTCUSDT + ETHUSDT with the same production
discipline as the Polymarket full pull: resume-safe, streamed to disk, whole-file zstd with a
bit-identical roundtrip check before the raw parquet is deleted, a per-day integrity ledger,
and a disk floor. **Depth only** — trades we already have free.

    python -m model_lab.telonex_binance_pull --dry-run   # Stage 4: project size + disk/quota gate
    python -m model_lab.telonex_binance_pull             # Stage 5: bulk pull (gated on the projection fitting)
    python -m model_lab.telonex_binance_pull --start 2026-05-01 --end 2026-07-07   # a bounded window

Writes ``out/telonex/binance_coverage.json`` + ``binance_pull_report.html`` (namespaced so the
Polymarket ``coverage.json`` / ``full_pull_report.html`` are untouched). The key is read from
``$telonexdata`` / ``.env`` and never printed. Nothing is buffered whole (16 GB box).
"""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import Paths, resolve_paths, stage_parser
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem
from .telonex_binance_trial import DEPTH_CHANNEL, DEFAULT_SYMBOLS, OUT_SUBDIR, TRIAL_META_NAME, ZSTD_LEVEL

GIB = 1 << 30
SAFETY_MARGIN_GIB = 40.0          # free-disk safety margin (task requirement)
POLY_GROSS_RESERVE_GIB = 188.0    # the task's conservative gross Polymarket reservation
DISK_FLOOR_GIB = 10.0             # abort the loop if free disk drops below this
DEFAULT_JOBS = 3                  # polite concurrency (matches the Polymarket pull)

COVERAGE_NAME = "binance_coverage.json"
REPORT_NAME = "binance_pull_report.html"
FALLBACK_FROM = "2026-02-06"      # only used if the trial-meta availability is missing


# --------------------------------------------------------------------------- #
# Date range + trial meta
# --------------------------------------------------------------------------- #


def _load_trial_meta(paths: Paths) -> dict[str, Any] | None:
    p = paths.out_dir / OUT_SUBDIR / TRIAL_META_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _symbol_range(meta: dict[str, Any] | None, symbol: str) -> tuple[date, date] | None:
    """``(first_day, last_day_inclusive)`` for the depth channel from the availability meta.

    ``to_date`` is exclusive, so the last available day is the day before it."""
    avail = (meta or {}).get("availability", {}).get(symbol, {})
    r = avail.get("channels", {}).get(DEPTH_CHANNEL) if isinstance(avail, dict) else None
    if not isinstance(r, dict) or not r.get("from_date") or not r.get("to_date"):
        return None
    try:
        frm = date.fromisoformat(r["from_date"])
        to_excl = date.fromisoformat(r["to_date"])
    except ValueError:
        return None
    return frm, to_excl - timedelta(days=1)


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _resolve_ranges(meta: dict[str, Any] | None, symbols: tuple[str, ...],
                    start: str | None, end: str | None) -> dict[str, tuple[date, date]]:
    """Per-symbol ``(first, last_inclusive)`` days to pull, clamped to any CLI ``--start/--end``."""
    clamp_lo = date.fromisoformat(start) if start else None
    clamp_hi = date.fromisoformat(end) if end else None
    out: dict[str, tuple[date, date]] = {}
    for sym in symbols:
        rng = _symbol_range(meta, sym)
        if rng is None:
            rng = (date.fromisoformat(FALLBACK_FROM), datetime.now(timezone.utc).date() - timedelta(days=2))
        lo, hi = rng
        if clamp_lo:
            lo = max(lo, clamp_lo)
        if clamp_hi:
            hi = min(hi, clamp_hi)
        if lo <= hi:
            out[sym] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# Stage 4 — projection + disk/quota gate
# --------------------------------------------------------------------------- #


def _measured_zst_per_day(meta: dict[str, Any] | None) -> dict[str, float]:
    """Mean measured ``.zst`` bytes/day per symbol from the trial's depth files."""
    by_sym: dict[str, list[int]] = {}
    for f in (meta or {}).get("files", []):
        if f.get("channel") != DEPTH_CHANNEL or not f.get("zst_bytes"):
            continue
        by_sym.setdefault(f["symbol"], []).append(int(f["zst_bytes"]))
    return {s: (sum(v) / len(v)) for s, v in by_sym.items() if v}


def _poly_remaining_bytes(paths: Paths) -> int:
    """Genuinely-unwritten Polymarket bytes = projection − already-written (avoids the
    188 GiB gross reservation double-counting the ~116 GiB already on disk)."""
    p = paths.out_dir / OUT_SUBDIR / "coverage.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        proj = int(d.get("projection", {}).get("zst_bytes", 0))
        done = int(d.get("totals", {}).get("zst_bytes", 0))
        return max(0, proj - done)
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return 0


def project(paths: Paths, *, symbols: tuple[str, ...], start: str | None, end: str | None) -> dict[str, Any]:
    meta = _load_trial_meta(paths)
    per_day = _measured_zst_per_day(meta)
    if not per_day:
        return {"status": "no_trial", "detail": "no measured trial depth sizes — run --trial first"}
    # ETH falls back to the BTC per-day size if it wasn't in the trial.
    default_size = per_day.get("btcusdt") or next(iter(per_day.values()))
    ranges = _resolve_ranges(meta, symbols, start, end)

    per_symbol: dict[str, Any] = {}
    projected = 0.0
    file_count = 0
    total_days = 0
    for sym, (lo, hi) in ranges.items():
        ndays = (hi - lo).days + 1
        size = per_day.get(sym, default_size)
        est = ndays * size
        per_symbol[sym] = {"first_day": lo.isoformat(), "last_day": hi.isoformat(),
                           "days": ndays, "zst_bytes_per_day": int(size),
                           "est_zst_bytes": int(est), "est_zst_human": human_bytes(int(est))}
        projected += est
        file_count += ndays
        total_days = max(total_days, ndays)
    projected = int(projected)

    free = tx.free_disk_bytes(paths.telonex_dir)
    poly_remaining = _poly_remaining_bytes(paths)
    margin = int(SAFETY_MARGIN_GIB * GIB)
    gross_reserve = int(POLY_GROSS_RESERVE_GIB * GIB)
    principled_avail = free - margin - poly_remaining
    conservative_avail = free - margin - gross_reserve
    gate_avail = min(principled_avail, conservative_avail)  # gate on the stricter
    fits = projected <= gate_avail

    # Quota: Pro is unlimited; a finite remaining would be a red flag.
    remaining = (meta or {}).get("downloads_remaining")
    quota_ok = remaining is None or (isinstance(remaining, int) and remaining >= file_count)

    fallback: dict[str, Any] | None = None
    if not fits and total_days > 0:
        per_day_both = sum(per_day.get(s, default_size) for s in ranges)
        days_fit = int(max(0, gate_avail) // per_day_both) if per_day_both > 0 else 0
        fallback = _recent_window(ranges, days_fit, per_day, default_size)

    return {
        "status": "ok",
        "depth_channel": DEPTH_CHANNEL,
        "symbols": list(ranges),
        "per_symbol": per_symbol,
        "projected_zst_bytes": projected,
        "projected_zst_human": human_bytes(projected),
        "file_count": file_count,
        "disk": {
            "free_bytes": free, "free_human": human_bytes(free),
            "safety_margin_gib": SAFETY_MARGIN_GIB,
            "polymarket_remaining_bytes": poly_remaining,
            "polymarket_remaining_human": human_bytes(poly_remaining),
            "polymarket_gross_reserve_gib": POLY_GROSS_RESERVE_GIB,
            "principled_available_bytes": principled_avail,
            "principled_available_human": human_bytes(max(0, principled_avail)),
            "conservative_available_bytes": conservative_avail,
            "conservative_available_human": human_bytes(max(0, conservative_avail)),
            "gate_available_bytes": gate_avail,
            "gate_available_human": human_bytes(max(0, gate_avail)),
        },
        "fits_disk": bool(fits),
        "quota_remaining": remaining,
        "quota_ok": bool(quota_ok),
        "fits": bool(fits and quota_ok),
        "fallback_recent_window": fallback,
    }


def _recent_window(ranges: dict[str, tuple[date, date]], days_fit: int,
                   per_day: dict[str, float], default_size: float) -> dict[str, Any]:
    """Largest recent contiguous window (keep newest days) that fits, + what gets cut."""
    win: dict[str, Any] = {"days_per_symbol": days_fit, "per_symbol": {}, "est_zst_bytes": 0}
    total = 0.0
    for sym, (lo, hi) in ranges.items():
        new_lo = max(lo, hi - timedelta(days=days_fit - 1)) if days_fit > 0 else hi + timedelta(days=1)
        kept = max(0, (hi - new_lo).days + 1) if days_fit > 0 else 0
        cut_days = ((hi - lo).days + 1) - kept
        est = kept * per_day.get(sym, default_size)
        total += est
        win["per_symbol"][sym] = {
            "keep_from": new_lo.isoformat() if days_fit > 0 else None,
            "keep_to": hi.isoformat(), "keep_days": kept,
            "cut_earliest_days": cut_days,
            "cut_range": f"{lo.isoformat()}..{(new_lo - timedelta(days=1)).isoformat()}" if cut_days > 0 else None,
        }
    win["est_zst_bytes"] = int(total)
    win["est_zst_human"] = human_bytes(int(total))
    return win


# --------------------------------------------------------------------------- #
# Stage 5 — bulk pull
# --------------------------------------------------------------------------- #


def _expected_specs(ranges: dict[str, tuple[date, date]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for sym, (lo, hi) in ranges.items():
        for d in _daterange(lo, hi):
            specs.append({"symbol": sym, "day": d.isoformat()})
    return specs


def _pull_one(paths: Paths, spec: dict[str, Any], api_key: str, fetch: tx.Fetch,
              *, level: int, delete_raw: bool, scratch: Path) -> dict[str, Any]:
    """Download → compress → bit-verify → (delete raw). Returns a per-file record."""
    raw = tx.binance_raw_path(paths.telonex_dir, DEPTH_CHANNEL, spec["symbol"], spec["day"])
    rec: dict[str, Any] = {"symbol": spec["symbol"], "day": spec["day"]}
    if tx.zst_finalized(raw):
        zst = tx.zst_sibling(raw)
        rec.update({"status": "skipped", "zst_bytes": zst.stat().st_size, "roundtrip_ok": True,
                    "raw_deleted": True})
        return rec
    res = tx.download_asset_day(tx.BINANCE, DEPTH_CHANNEL, spec["day"], raw,
                               api_key=api_key, fetch=fetch, slug=spec["symbol"])
    status = res.get("status")
    rec["remaining"] = res.get("remaining")
    if status == "no_data":
        rec["status"] = "no_data"
        return rec
    if status not in ("downloaded", "skipped"):
        rec["status"] = "error"
        rec["error"] = f"unexpected status {status}"
        return rec
    zst = tx.zst_sibling(raw)
    if not zst.exists():
        tx.compress_file_zstd(raw, zst, level=level)
    ok = tx.verify_roundtrip(raw, zst, scratch_dir=scratch)
    rec.update({"status": "done" if ok else "roundtrip_failed",
                "zst_bytes": zst.stat().st_size, "roundtrip_ok": bool(ok), "raw_deleted": False})
    if ok and delete_raw and raw.exists():
        tx._safe_unlink(raw)
        rec["raw_deleted"] = True
    return rec


def bulk_pull(paths: Paths, *, symbols: tuple[str, ...], start: str | None, end: str | None,
              jobs: int = DEFAULT_JOBS, level: int = ZSTD_LEVEL, delete_raw: bool = True,
              api_key: str | None = None, fetch: tx.Fetch = tx.urllib_fetch,
              projection: dict[str, Any] | None = None, progress=None) -> dict[str, Any]:
    if api_key is None:
        api_key = tx.load_api_key()
    meta = _load_trial_meta(paths)
    ranges = _resolve_ranges(meta, symbols, start, end)
    specs = _expected_specs(ranges)
    scratch = paths.out_dir / OUT_SUBDIR / "_binance_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    coverage = _init_coverage(paths, ranges, specs, projection)
    todo = [s for s in specs if not tx.zst_finalized(tx.binance_raw_path(
        paths.telonex_dir, DEPTH_CHANNEL, s["symbol"], s["day"]))]
    note(f"{len(specs)} expected files, {len(specs) - len(todo)} already finalized, {len(todo)} to pull")

    # Up-front disk preflight.
    proj_bytes = int((projection or {}).get("projected_zst_bytes", 0))
    free = tx.free_disk_bytes(paths.telonex_dir)
    need = proj_bytes + 200 * (1 << 20) + 5 * GIB  # projection + one raw file + 5 GiB headroom
    if free < need:
        coverage["run"]["aborted"] = "disk_preflight"
        _write_coverage(paths, coverage)
        return {"status": "aborted_disk", "free": free, "need": need, "coverage": coverage}

    done = 0
    aborted: str | None = None
    remaining_seen: int | None = None
    idx = 0
    while idx < len(todo) and aborted is None:
        # Disk floor before each batch.
        free = tx.free_disk_bytes(paths.telonex_dir)
        if free < DISK_FLOOR_GIB * GIB:
            aborted = "disk_floor"
            break
        batch = todo[idx:idx + jobs]
        idx += jobs
        results: list[dict[str, Any]] = []
        if jobs <= 1 or len(batch) == 1:
            for s in batch:
                results.append(_safe_pull(paths, s, api_key, fetch, level=level,
                                          delete_raw=delete_raw, scratch=scratch))
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_safe_pull, paths, s, api_key, fetch, level=level,
                                  delete_raw=delete_raw, scratch=scratch): s for s in batch}
                for fut in as_completed(futs):
                    results.append(fut.result())
        for rec in results:
            if rec.get("status") == "limit":
                aborted = "download_limit"
            if rec.get("remaining") is not None:
                remaining_seen = rec["remaining"] if remaining_seen is None else min(remaining_seen, rec["remaining"])
            _record_day(coverage, rec)
            if rec.get("status") in ("done", "skipped"):
                done += 1
            note(f"{rec['symbol']}/{rec['day']}: {rec.get('status')}"
                 + (f" ({human_bytes(rec['zst_bytes'])})" if rec.get("zst_bytes") else ""))
        _recompute_totals(coverage)
        coverage["run"]["downloads_remaining"] = remaining_seen
        _write_coverage(paths, coverage)  # checkpoint BEFORE the next batch (PASS ⇒ zst on disk)

    coverage["run"]["aborted"] = aborted
    _recompute_totals(coverage)
    _write_coverage(paths, coverage)
    _write_report(paths, coverage)
    return {"status": "aborted" if aborted else "ok", "aborted": aborted, "done": done,
            "expected": len(specs), "coverage": coverage}


def _safe_pull(paths: Paths, spec, api_key, fetch, *, level, delete_raw, scratch) -> dict[str, Any]:
    try:
        return _pull_one(paths, spec, api_key, fetch, level=level, delete_raw=delete_raw, scratch=scratch)
    except tx.TelonexDownloadLimit as e:
        return {"symbol": spec["symbol"], "day": spec["day"], "status": "limit", "error": str(e)}
    except tx.TelonexPlanRestricted as e:
        return {"symbol": spec["symbol"], "day": spec["day"], "status": "restricted", "error": str(e)}
    except Exception as e:  # noqa: BLE001 — one file's failure isn't fatal
        return {"symbol": spec["symbol"], "day": spec["day"], "status": "error", "error": str(e)}


# --------------------------------------------------------------------------- #
# Coverage ledger + report
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _init_coverage(paths: Paths, ranges, specs, projection) -> dict[str, Any]:
    p = paths.out_dir / OUT_SUBDIR / COVERAGE_NAME
    prior: dict[str, Any] = {}
    if p.exists():
        try:
            prior = json.loads(p.read_text(encoding="utf-8")).get("days", {})
        except (OSError, json.JSONDecodeError):
            prior = {}
    days: dict[str, Any] = {}
    for s in specs:
        key = f"{s['symbol']}/{s['day']}"
        raw = tx.binance_raw_path(paths.telonex_dir, DEPTH_CHANNEL, s["symbol"], s["day"])
        if tx.zst_finalized(raw):
            zst = tx.zst_sibling(raw)
            days[key] = {"status": "done", "zst_bytes": zst.stat().st_size,
                         "roundtrip_ok": True, "raw_deleted": True}
        elif key in prior:
            days[key] = prior[key]
        else:
            days[key] = {"status": "pending"}
    return {
        "schema_version": 1, "generated_utc": _now(), "updated_utc": _now(),
        "channel": DEPTH_CHANNEL,
        "ranges": {s: {"first": lo.isoformat(), "last": hi.isoformat(), "days": (hi - lo).days + 1}
                   for s, (lo, hi) in ranges.items()},
        "expected_files": len(specs), "days": days,
        "projection": {k: projection.get(k) for k in ("projected_zst_bytes", "projected_zst_human",
                       "file_count")} if projection else None,
        "totals": {}, "run": {"started_utc": _now(), "aborted": None, "downloads_remaining": None, "jobs": DEFAULT_JOBS},
    }


def _record_day(coverage: dict[str, Any], rec: dict[str, Any]) -> None:
    key = f"{rec['symbol']}/{rec['day']}"
    entry = {k: rec[k] for k in ("status", "zst_bytes", "roundtrip_ok", "raw_deleted", "error")
             if k in rec}
    coverage["days"][key] = entry


def _recompute_totals(coverage: dict[str, Any]) -> None:
    done = zst = failed = no_data = 0
    for e in coverage["days"].values():
        st = e.get("status")
        if st in ("done", "skipped"):
            done += 1
            zst += int(e.get("zst_bytes", 0) or 0)
        elif st == "roundtrip_failed":
            failed += 1
        elif st == "no_data":
            no_data += 1
    coverage["totals"] = {
        "expected_files": coverage["expected_files"], "done_files": done,
        "roundtrip_failed": failed, "no_data": no_data,
        "zst_bytes": zst, "zst_human": human_bytes(zst),
    }
    coverage["updated_utc"] = _now()


def _write_coverage(paths: Paths, coverage: dict[str, Any]) -> Path:
    out = paths.out_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    p = out / COVERAGE_NAME
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _write_report(paths: Paths, coverage: dict[str, Any]) -> Path:
    out = paths.out_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    p = out / REPORT_NAME
    t = coverage.get("totals", {})
    aborted = coverage.get("run", {}).get("aborted")
    status = "ALERT" if aborted else ("GO" if t.get("done_files") == t.get("expected_files") else "PARTIAL")
    colour = {"GO": "#1a7f37", "PARTIAL": "#9a6700", "ALERT": "#cf222e"}.get(status, "#57606a")
    rows = []
    for key, e in sorted(coverage["days"].items()):
        st = e.get("status", "?")
        c = {"done": "#1a7f37", "skipped": "#1a7f37", "pending": "#57606a",
             "no_data": "#9a6700"}.get(st, "#cf222e")
        rows.append(f"<tr><td>{key}</td><td style='color:{c};font-weight:600'>{st}</td>"
                    f"<td>{human_bytes(int(e.get('zst_bytes', 0) or 0))}</td>"
                    f"<td>{e.get('roundtrip_ok', '')}</td><td>{e.get('error', '')}</td></tr>")
    return _write_html(p, status, colour, coverage, t, rows)


def _write_html(p: Path, status, colour, coverage, t, rows) -> Path:
    html = f"""<!doctype html><meta charset=utf-8><title>Telonex Binance depth pull</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;max-width:1000px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d0d7de;padding:5px 9px;text-align:left}}
h1 span{{color:{colour}}}</style>
<h1>Telonex Binance depth ({coverage.get('channel')}) — <span>{status}</span></h1>
<p>{t.get('done_files', 0)}/{t.get('expected_files', 0)} files · {t.get('zst_human', '0')} zst ·
roundtrip-failed {t.get('roundtrip_failed', 0)} · no-data {t.get('no_data', 0)} ·
aborted={coverage.get('run', {}).get('aborted')}</p>
<table><tr><th>symbol/day</th><th>status</th><th>zst</th><th>roundtrip</th><th>error</th></tr>{''.join(rows)}</table>
"""
    p.write_text(html, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print(msg: str) -> None:
    print(f"[telonex_binance_pull] {msg}")


def _print_projection(proj: dict[str, Any]) -> None:
    if proj.get("status") != "ok":
        _print(f"projection: {proj.get('status')} — {proj.get('detail', '')}")
        return
    _print(f"projected FULL history ({proj['depth_channel']}, {', '.join(proj['symbols'])}): "
           f"{proj['projected_zst_human']} zst in {proj['file_count']} files")
    for sym, s in proj["per_symbol"].items():
        _print(f"  {sym}: {s['days']}d {s['first_day']}→{s['last_day']} "
               f"@ {human_bytes(s['zst_bytes_per_day'])}/day = {s['est_zst_human']}")
    d = proj["disk"]
    _print(f"disk: free {d['free_human']}; margin {d['safety_margin_gib']} GiB; "
           f"Polymarket remaining {d['polymarket_remaining_human']} (gross reserve {d['polymarket_gross_reserve_gib']} GiB)")
    _print(f"  available (principled) {d['principled_available_human']}  |  "
           f"(conservative) {d['conservative_available_human']}  → gate on {d['gate_available_human']}")
    _print(f"FITS: {proj['fits']}  (disk {proj['fits_disk']}, quota {proj['quota_ok']}, "
           f"remaining={proj['quota_remaining']})")
    if proj.get("fallback_recent_window"):
        fb = proj["fallback_recent_window"]
        _print(f"does NOT fit — largest recent window: {fb['days_per_symbol']} newest days/symbol "
               f"= {fb.get('est_zst_human')}")
        for sym, w in fb["per_symbol"].items():
            _print(f"  {sym}: keep {w['keep_from']}→{w['keep_to']} ({w['keep_days']}d); "
                   f"CUT earliest {w['cut_earliest_days']}d ({w['cut_range']})")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_binance_pull")
    parser.add_argument("--dry-run", action="store_true", help="Stage 4: project size + disk/quota gate only")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma list (default btcusdt,ethusdt)")
    parser.add_argument("--start", default=None, help="clamp start day YYYY-MM-DD (default = availability first)")
    parser.add_argument("--end", default=None, help="clamp end day YYYY-MM-DD (default = availability last)")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help=f"concurrency (default {DEFAULT_JOBS})")
    parser.add_argument("--zstd-level", type=int, default=ZSTD_LEVEL, help=f"zstd level (default {ZSTD_LEVEL})")
    parser.add_argument("--no-delete-raw", action="store_true", help="keep raw parquet after the roundtrip check")
    parser.add_argument("--force", action="store_true", help="pull even if the projection doesn't fit disk")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())

    proj = project(paths, symbols=symbols, start=args.start, end=args.end)
    _print_projection(proj)

    if args.dry_run:
        procmem.report_peak_rss("telonex_binance_pull", procmem.peak_rss_mb(), args.max_rss_mb)
        if proj.get("status") != "ok":
            return 1
        _print("dry-run only. If FITS, next: python -m model_lab.telonex_binance_pull")
        return 0 if proj.get("fits") else 1

    if proj.get("status") != "ok":
        _print("cannot pull without a projection (run --trial first).")
        return 1
    if not proj.get("fits") and not args.force:
        _print("STOP — projection does not fit disk/quota. Use --start/--end to pull the recent window "
               "shown above, or --force to override the disk gate.")
        return 1

    try:
        api_key = tx.load_api_key()
    except tx.TelonexError as e:
        _print(str(e))
        return 1

    _print(f"starting bulk pull → {tx.raw_root(paths.telonex_dir) / tx.BINANCE / DEPTH_CHANNEL}")
    res = bulk_pull(paths, symbols=symbols, start=args.start, end=args.end, jobs=args.jobs,
                    level=args.zstd_level, delete_raw=not args.no_delete_raw, api_key=api_key,
                    projection=proj, progress=_print)
    cov = res["coverage"]
    t = cov["totals"]
    _print(f"done: {t['done_files']}/{t['expected_files']} files, {t['zst_human']} zst, "
           f"roundtrip-failed {t['roundtrip_failed']}, no-data {t['no_data']}, aborted={res.get('aborted')}")
    _print(f"wrote {paths.out_dir / OUT_SUBDIR / COVERAGE_NAME}")
    _print(f"wrote {paths.out_dir / OUT_SUBDIR / REPORT_NAME}")
    procmem.report_peak_rss("telonex_binance_pull", procmem.peak_rss_mb(), args.max_rss_mb)
    return 0 if res.get("status") == "ok" and t["done_files"] == t["expected_files"] else 1


if __name__ == "__main__":
    sys.exit(main())
