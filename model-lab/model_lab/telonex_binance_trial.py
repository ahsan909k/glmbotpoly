"""Telonex Binance order-book DEPTH — Stage 1 (discover) + Stage 2 (trial download).

This is the validate-first front end for buying historical Binance **spot depth** from
Telonex to train the short-horizon depth model. Our own ``depth20@100ms`` capture only
began 2026-07-03 (<1 week); free ``data.binance.vision`` gives trades but **no order
book**. Telonex sells Binance depth back to ~2026-02-06 — but our key was previously
Single-Exchange (Polymarket-only), so this stage first **re-verifies the (now upgraded)
key can reach Binance at all**, then pulls a tiny trial for the validator to judge.

    python -m model_lab.telonex_binance_trial --discover   # Stage 1 → out/telonex/binance_catalog.md
    python -m model_lab.telonex_binance_trial --trial      # Stage 2 → data/telonex/binance/raw/…

Depth is the point; **trades we already have free** — trades are pulled only in the trial
for the clock cross-check (Stage 3). The bulk pull (``telonex_binance_pull``) is depth only.

The API key is read from ``$telonexdata`` / ``.env`` and is never printed or written to any
artifact. Nothing is buffered whole (16 GB box) — every download streams to disk.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .config import Paths, resolve_paths, stage_parser
from .io import telonex as tx
from .io.binance_archive import human_bytes
from .lib import procmem

# The only Binance channel that gives ≥20 levels (top-25, event-driven from @depth@100ms).
# book_snapshot_5 = top-5 (✗), quotes = level-0 (✗), and there is no book_snapshot_full for
# Binance. So this is both the only and the leanest qualifying depth channel.
DEPTH_CHANNEL = "book_snapshot_25"
TRADES_CHANNEL = "trades"
DEFAULT_SYMBOLS = ("btcusdt", "ethusdt")

# Trial days sit inside our depth capture's coverage (2026-07-03 → 07-07) so the validator
# has live-journaled features to compare against.
TRIAL_BTC_DEPTH_DAYS = ("2026-07-04", "2026-07-05", "2026-07-06")
TRIAL_BTC_TRADE_DAYS = ("2026-07-04", "2026-07-05", "2026-07-06")
TRIAL_ETH_DEPTH_DAYS = ("2026-07-05",)  # one ETH depth day → BTC:ETH size ratio for Stage-4
PROBE_DAY = "2026-07-05"  # a day the notes confirm has btcusdt data — so a 404 can't confuse the gate

ZSTD_LEVEL = 12  # the Polymarket pull's validated level (fast, ~1.5× on already-columnar parquet)

OUT_SUBDIR = "telonex"
CATALOG_NAME = "binance_catalog.md"
TRIAL_META_NAME = "binance_trial.json"

# Access-probe verdicts.
ACCESS_PRO = "pro"          # Binance reachable on this plan → proceed
ACCESS_RESTRICTED = "restricted"  # exchange_restricted / Upgrade to Pro → key NOT upgraded, STOP
ACCESS_LIMIT = "limit"      # a real 403 download limit
ACCESS_AUTH = "auth"        # 401 bad key
ACCESS_ERROR = "error"


# --------------------------------------------------------------------------- #
# Stage 1 — discover
# --------------------------------------------------------------------------- #


def _read_remaining(headers: dict[str, str]) -> int | None:
    for k, v in headers.items():
        if k.lower() == "x-downloads-remaining":
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def probe_access(
    api_key: str, *, symbol: str = "btcusdt", day: str = PROBE_DAY,
    channel: str = DEPTH_CHANNEL, fetch: tx.Fetch = tx.urllib_fetch,
) -> dict[str, Any]:
    """Confirm the (upgraded) key can reach Binance downloads — **without consuming one**.

    Issues the 302 (no-follow) GET on a known-good Binance download URL so we read the
    status/headers only, never the pre-signed S3 body. Distinguishes ``exchange_restricted``
    (the key is still Polymarket-only → hard STOP) from a genuine quota 403 and from a happy
    unlimited plan — which ``tx.probe_remaining`` cannot, since it maps every 403 to a
    download-limit. Returns ``{"access": …, "remaining": int|None, "detail": str}``.
    """
    url = tx.build_download_url(tx.BINANCE, channel, day, {"slug": symbol})
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = fetch(url, headers, False)  # do NOT follow the 302 — no S3 body, no download consumed
    except tx.TelonexHTTPError as e:
        body = e.body.decode("utf-8", "replace") if e.body else ""
        if e.status == 403 and ("exchange_restricted" in body or "Upgrade to Pro" in body):
            return {"access": ACCESS_RESTRICTED, "remaining": None, "detail": body[:200], "status": 403}
        if e.status == 403:
            return {"access": ACCESS_LIMIT, "remaining": _read_remaining(e.headers),
                    "detail": body[:200], "status": 403}
        if e.status == 401:
            return {"access": ACCESS_AUTH, "remaining": None, "detail": "unauthorized (check the key)",
                    "status": 401}
        return {"access": ACCESS_ERROR, "remaining": None, "detail": f"HTTP {e.status}: {body[:160]}",
                "status": e.status}
    try:
        status = resp.status
        remaining = _read_remaining(resp.headers)
    finally:
        resp.close()
    if status in tx._REDIRECT_CODES or status == 200:
        # A 302 to S3 (or a direct 200) means the request was authorized → Binance is on the plan.
        return {"access": ACCESS_PRO, "remaining": remaining, "detail": "", "status": status}
    if status == 404:
        # Authorized but no data for the probe day — still proves plan access.
        return {"access": ACCESS_PRO, "remaining": remaining, "detail": "probe day had no data", "status": 404}
    return {"access": ACCESS_ERROR, "remaining": remaining, "detail": f"unexpected status {status}",
            "status": status}


def discover(
    paths: Paths, *, api_key: str, symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    fetch: tx.Fetch = tx.urllib_fetch, progress=None,
) -> dict[str, Any]:
    """Stage 1: Pro-access gate + public availability for each symbol + channel choice."""

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    access = probe_access(api_key, fetch=fetch)
    note(f"access probe: {access['access']}"
         + (f" (X-Downloads-Remaining={access['remaining']})" if access.get("remaining") is not None else ""))

    result: dict[str, Any] = {
        "generated_utc": _now_utc(),
        "access": access,
        "depth_channel": DEPTH_CHANNEL,
        "symbols": list(symbols),
        "availability": {},
    }
    if access["access"] != ACCESS_PRO:
        result["status"] = access["access"]
        return result

    # Public availability (no key, no credit) per symbol → per-channel {from_date, to_date}.
    for sym in symbols:
        try:
            info = tx.availability(tx.BINANCE, slug=sym, fetch=fetch)
        except tx.TelonexNotFound:
            result["availability"][sym] = {"error": "not_found"}
            continue
        except tx.TelonexError as e:
            result["availability"][sym] = {"error": str(e)}
            continue
        chans = info.get("channels", {}) if isinstance(info, dict) else {}
        result["availability"][sym] = {
            "channels": {c: {"from_date": r.get("from_date"), "to_date": r.get("to_date")}
                         for c, r in chans.items() if isinstance(r, dict)},
        }
        note(f"availability {sym}: " + ", ".join(
            f"{c}[{r.get('from_date')}→{r.get('to_date')})" for c, r in chans.items() if isinstance(r, dict)))

    # Confirm the chosen channel exists for both symbols.
    depth_ok = all(
        DEPTH_CHANNEL in result["availability"].get(sym, {}).get("channels", {})
        for sym in symbols
    )
    result["depth_channel_available"] = depth_ok
    result["status"] = "ok" if depth_ok else "no_depth_channel"
    return result


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _date_range(discover_result: dict[str, Any], symbol: str, channel: str) -> tuple[str | None, str | None]:
    r = discover_result.get("availability", {}).get(symbol, {}).get("channels", {}).get(channel)
    if not isinstance(r, dict):
        return None, None
    return r.get("from_date"), r.get("to_date")


def write_catalog_md(path: Path, res: dict[str, Any]) -> None:
    """Render ``binance_catalog.md`` — the Stage-1 deliverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    acc = res["access"]
    lines: list[str] = []
    lines.append("# Telonex Binance catalog — depth pull discovery")
    lines.append("")
    lines.append(f"_Generated {res['generated_utc']}._")
    lines.append("")
    lines.append("## Plan access (upgraded key)")
    lines.append("")
    if acc["access"] == ACCESS_PRO:
        rem = acc.get("remaining")
        lines.append(f"- **Binance is reachable on this key** (access probe → `{acc.get('status')}`). "
                     + (f"`X-Downloads-Remaining` = {rem} (a finite quota)."
                        if rem is not None else
                        "No `X-Downloads-Remaining` header → **unlimited** (Pro), matching the docs."))
    elif acc["access"] == ACCESS_RESTRICTED:
        lines.append("- **STOP — Binance is NOT on this key's plan** (`exchange_restricted` / "
                     "`Upgrade to Pro`). The key has not been upgraded to Pro. Detail: "
                     f"`{acc.get('detail', '')}`.")
    else:
        lines.append(f"- **STOP — access probe returned `{acc['access']}`**: {acc.get('detail', '')}")
    lines.append("")

    lines.append("## Availability (public endpoint, per symbol)")
    lines.append("")
    lines.append("| symbol | channel | from_date | to_date (exclusive) |")
    lines.append("|---|---|---|---|")
    for sym in res.get("symbols", []):
        chans = res.get("availability", {}).get(sym, {}).get("channels", {})
        if not chans:
            err = res.get("availability", {}).get(sym, {}).get("error", "—")
            lines.append(f"| {sym} | (none) | {err} | |")
            continue
        for c, r in sorted(chans.items()):
            lines.append(f"| {sym} | {c} | {r.get('from_date')} | {r.get('to_date')} |")
    lines.append("")

    lines.append("## Channels & the chosen depth channel")
    lines.append("")
    lines.append("Telonex Binance channels (spot only — there is no futures endpoint, and **no "
                 "`book_snapshot_full`**; max 25 levels):")
    lines.append("")
    lines.append("| channel | levels | source stream | usable for depth ≥20? |")
    lines.append("|---|---|---|---|")
    lines.append("| `trades` | — | `@trade` | trades only (we have these free; trial clock-check) |")
    lines.append("| `quotes` | 1 (L0) | `@depth@100ms` | ✗ top-of-book only |")
    lines.append("| `book_snapshot_5` | 5 | `@depth@100ms` | ✗ too shallow |")
    lines.append("| **`book_snapshot_25`** | **25** | `@depth@100ms` | ✅ **chosen** |")
    lines.append("")
    lines.append(f"**Chosen depth channel: `{DEPTH_CHANNEL}`.** It is the only Binance channel with "
                 "≥20 levels (`depth_imb_20` needs 20; slopes use top-10), and it is already top-25-only "
                 "(event-driven from `@depth@100ms`) — the leanest qualifying channel. There is no "
                 "fixed-cadence `snapshot_N` channel for Binance; effective cadence is measured in Stage 3.")
    lines.append("")
    for sym in res.get("symbols", []):
        frm, to = _date_range(res, sym, DEPTH_CHANNEL)
        lines.append(f"- `{sym}` `{DEPTH_CHANNEL}` true date range: **{frm} → {to}** "
                     "(`to_date` exclusive ⇒ last available day is the day before).")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Stage 2 — trial download
# --------------------------------------------------------------------------- #


def _trial_specs(telonex_dir: Path) -> list[dict[str, Any]]:
    """The fixed trial download set: BTC depth ×3 days, BTC trades ×3 days, ETH depth ×1 day."""
    specs: list[dict[str, Any]] = []
    for day in TRIAL_BTC_DEPTH_DAYS:
        specs.append({"symbol": "btcusdt", "channel": DEPTH_CHANNEL, "day": day})
    for day in TRIAL_BTC_TRADE_DAYS:
        specs.append({"symbol": "btcusdt", "channel": TRADES_CHANNEL, "day": day})
    for day in TRIAL_ETH_DEPTH_DAYS:
        specs.append({"symbol": "ethusdt", "channel": DEPTH_CHANNEL, "day": day})
    for s in specs:
        s["raw"] = tx.binance_raw_path(telonex_dir, s["channel"], s["symbol"], s["day"])
    return specs


def trial(
    paths: Paths, *, api_key: str, fetch: tx.Fetch = tx.urllib_fetch, progress=None,
    min_free_gb: float = 5.0,
) -> dict[str, Any]:
    """Stage 2: download the trial files, compress each to ``.zst`` (measure the ratio),
    keep the raw ``.parquet`` for the validator. Returns a summary + per-file sizes."""

    def note(msg: str) -> None:
        if progress:
            progress(msg)

    telonex_dir = paths.telonex_dir
    specs = _trial_specs(telonex_dir)

    free = tx.free_disk_bytes(telonex_dir)
    if free < min_free_gb * (1 << 30):
        return {"status": "aborted_disk", "free_disk_bytes": free,
                "detail": f"free {human_bytes(free)} < {min_free_gb} GiB floor"}

    files: list[dict[str, Any]] = []
    downloaded = skipped = no_data = errors = 0
    remaining: int | None = None
    for s in specs:
        raw: Path = s["raw"]
        label = f"{s['symbol']}/{s['channel']}/{s['day']}"
        try:
            res = tx.download_asset_day(
                tx.BINANCE, s["channel"], s["day"], raw, api_key=api_key, fetch=fetch, slug=s["symbol"],
            )
        except tx.TelonexPlanRestricted as e:
            return {"status": "restricted", "detail": str(e)}
        except tx.TelonexDownloadLimit as e:
            return {"status": "aborted_limit", "detail": str(e)}
        except Exception as e:  # noqa: BLE001 — record and continue
            errors += 1
            files.append({**_spec_meta(s), "status": "error", "error": str(e)})
            note(f"{label} ERROR: {e}")
            continue

        status = res.get("status")
        if res.get("remaining") is not None:
            remaining = res["remaining"] if remaining is None else min(remaining, res["remaining"])
        if status in ("downloaded", "skipped"):
            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1
            raw_bytes = int(res.get("bytes", raw.stat().st_size if raw.exists() else 0))
            # Compress to .zst (measure ratio); keep the raw for the validator.
            zst = tx.zst_sibling(raw)
            try:
                if not zst.exists():
                    tx.compress_file_zstd(raw, zst, level=ZSTD_LEVEL)
                zst_bytes = zst.stat().st_size
            except Exception as e:  # noqa: BLE001 — compression failure is non-fatal for the trial
                zst_bytes = 0
                note(f"{label} zst compress failed: {e}")
            files.append({**_spec_meta(s), "status": status, "raw_bytes": raw_bytes, "zst_bytes": zst_bytes})
            note(f"{label} {status} ({human_bytes(raw_bytes)} raw → {human_bytes(zst_bytes)} zst)")
        elif status == "no_data":
            no_data += 1
            files.append({**_spec_meta(s), "status": "no_data"})
            note(f"{label} NO DATA (404)")
        else:
            errors += 1
            files.append({**_spec_meta(s), "status": status or "error"})

    return {
        "status": "ok" if errors == 0 and no_data == 0 else "incomplete",
        "downloaded": downloaded, "skipped": skipped, "no_data": no_data, "errors": errors,
        "downloads_remaining": remaining, "files": files,
    }


def _spec_meta(s: dict[str, Any]) -> dict[str, Any]:
    return {"symbol": s["symbol"], "channel": s["channel"], "day": s["day"], "raw_path": str(s["raw"])}


def write_trial_meta(paths: Paths, discover_result: dict[str, Any], trial_result: dict[str, Any]) -> Path:
    """Write ``out/telonex/binance_trial.json`` — the hand-off the validator + projector read
    (file locations, measured sizes, availability date range). No key material."""
    out = paths.out_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    p = out / TRIAL_META_NAME
    meta = {
        "generated_utc": _now_utc(),
        "depth_channel": DEPTH_CHANNEL,
        "trade_channel": TRADES_CHANNEL,
        "symbols": list(DEFAULT_SYMBOLS),
        "trial_btc_depth_days": list(TRIAL_BTC_DEPTH_DAYS),
        "trial_btc_trade_days": list(TRIAL_BTC_TRADE_DAYS),
        "trial_eth_depth_days": list(TRIAL_ETH_DEPTH_DAYS),
        "zstd_level": ZSTD_LEVEL,
        "telonex_dir": str(paths.telonex_dir),
        "availability": discover_result.get("availability", {}),
        "access": discover_result.get("access", {}),
        "files": trial_result.get("files", []),
        "downloads_remaining": trial_result.get("downloads_remaining"),
    }
    p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print(msg: str) -> None:
    print(f"[telonex_binance_trial] {msg}")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "telonex_binance_trial")
    parser.add_argument("--discover", action="store_true", help="Stage 1: probe access + write binance_catalog.md")
    parser.add_argument("--trial", action="store_true", help="Stage 2: download the trial files")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)

    if not (args.discover or args.trial):
        _print("pass --discover (Stage 1) and/or --trial (Stage 2).")
        return 1

    try:
        api_key = tx.load_api_key()
    except tx.TelonexError as e:
        _print(str(e))
        return 1

    out_dir = paths.out_dir / OUT_SUBDIR
    _print(f"store={paths.telonex_dir}  out={out_dir}")

    # Stage 1 — discover (always run: the trial needs the access verdict + availability).
    disc = discover(paths, api_key=api_key, progress=_print)
    write_catalog_md(out_dir / CATALOG_NAME, disc)
    _print(f"wrote {out_dir / CATALOG_NAME}")

    acc = disc["access"]["access"]
    if acc == ACCESS_RESTRICTED:
        _print("STOP — Binance is NOT on this key's plan (exchange_restricted). The key has not been "
               "upgraded to Pro. Nothing downloaded.")
        procmem.report_peak_rss("telonex_binance_trial", procmem.peak_rss_mb(), args.max_rss_mb)
        return 1
    if acc != ACCESS_PRO:
        _print(f"STOP — access probe returned '{acc}': {disc['access'].get('detail', '')}")
        procmem.report_peak_rss("telonex_binance_trial", procmem.peak_rss_mb(), args.max_rss_mb)
        return 1
    _print("Pro access confirmed.")
    if not disc.get("depth_channel_available"):
        _print(f"STOP — {DEPTH_CHANNEL} not available for both symbols per the availability endpoint.")
        procmem.report_peak_rss("telonex_binance_trial", procmem.peak_rss_mb(), args.max_rss_mb)
        return 1

    if not args.trial:
        procmem.report_peak_rss("telonex_binance_trial", procmem.peak_rss_mb(), args.max_rss_mb)
        _print("discover done. Next: --trial")
        return 0

    # Stage 2 — trial download.
    tr = trial(paths, api_key=api_key, progress=_print)
    meta_path = write_trial_meta(paths, disc, tr)
    _print(f"wrote {meta_path}")

    status = tr.get("status")
    if status in ("restricted", "aborted_limit", "aborted_disk"):
        _print(f"STOP — trial {status}: {tr.get('detail', '')}")
        procmem.report_peak_rss("telonex_binance_trial", procmem.peak_rss_mb(), args.max_rss_mb)
        return 1
    _print(f"trial: {tr.get('downloaded', 0)} downloaded, {tr.get('skipped', 0)} skipped, "
           f"{tr.get('no_data', 0)} no-data, {tr.get('errors', 0)} errors")
    if tr.get("downloads_remaining") is not None:
        _print(f"X-Downloads-Remaining (min observed): {tr['downloads_remaining']}")
    procmem.report_peak_rss("telonex_binance_trial", procmem.peak_rss_mb(), args.max_rss_mb)
    if status != "ok":
        _print("trial incomplete — some trial-day files are missing (07-04..06 must be present). "
               "Do NOT proceed to validation until resolved.")
        return 1
    _print("trial complete. Next: python -m model_lab.telonex_binance_validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
