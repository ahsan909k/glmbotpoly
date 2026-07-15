"""Cached, resumable pull of each account's public history.

For every resolved address we cache three things under ``data/competitors/<addr>/``:

- ``activity.jsonl`` — full ``/activity`` stream (TRADE/SPLIT/MERGE/REDEEM/REWARD/
  CONVERSION/TAKER_REBATE). The comprehensive record — drives cash-flow PnL, pair
  discipline, merge/redeem behaviour, sizes, hours, series focus. (TAKER_REBATE is the
  separate pUSD fee-rebate credit; it is also the event type our OWN live rebates arrive as.)
- ``taker_trades.jsonl`` — ``/trades?takerOnly=true`` (trades where the account was
  the aggressor). Differenced against activity TRADEs -> the maker/taker mix.
- ``positions.json`` — current open positions snapshot (realizedPnl cross-check).
- ``official_pnl.json`` — the official profile P/L series from ``user-pnl-api``
  (``[{"t","p"}]``, cumulative all-time PnL). The AUTHORITATIVE money source; the
  cash-flow reconstruction above is demoted to a cross-check.

Large histories are walked newest->oldest by shifting the ``end`` time cursor
(the API caps ``offset`` at 10k). Each endpoint keeps a ``.state.json`` so an
interrupted run resumes from the last cursor instead of re-fetching; a completed
endpoint is skipped entirely. Polite fixed delay between requests.

Run: ``python -m model_lab.competitors.fetch``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from . import api
from .paths import account_dir, competitors_dir

# TAKER_REBATE added 2026-07-15 (out/audits/takerner_pnl_anomaly.md): the taker-fee rebate is
# credited as a separate pUSD /activity event of this type, which the original 6-type filter
# omitted — so taker accounts' rebate income (and the flow OUR OWN live rebates will arrive as)
# was silently missing from the cash-flow reconstruction.
ACTIVITY_TYPES = "TRADE,SPLIT,MERGE,REDEEM,REWARD,CONVERSION,TAKER_REBATE"
MAX_PAGES = 12000  # runaway guard (≈6M records @ 500/page); truncation recorded honestly


def _key(r: dict) -> tuple:
    """Stable dedup key across event types and resume boundaries."""
    return (
        int(r.get("timestamp") or 0),
        str(r.get("type") or ""),
        str(r.get("conditionId") or ""),
        str(r.get("asset") or ""),
        str(r.get("side") or ""),
        round(float(r.get("size") or 0.0), 6),
        round(float(r.get("usdcSize") or 0.0), 6),
        str(r.get("transactionHash") or ""),
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # crash-truncated tail
    return out


def _append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def _read_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"complete": False, "cursor_end": None, "count": 0}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def _paginate(
    page_fn: Callable[[int | None, int], list[dict]],
    *,
    jsonl_path: Path,
    state_path: Path,
    label: str,
    max_records: int | None = None,
) -> list[dict]:
    """Walk a newest->oldest endpoint with (end-cursor, offset) pagination, caching
    incrementally. ``page_fn(end, offset) -> list[dict]``. Resumable via state file.
    ``max_records`` stops the (newest-first) walk once that many records are cached — a
    recent-window sample for ultra-high-frequency accounts whose full history is huge."""
    state = _read_state(state_path)
    records = _load_jsonl(jsonl_path)
    seen = {_key(r) for r in records}
    if state.get("complete"):
        print(f"  [{label}] cached complete — {len(records)} records")
        return records
    if max_records is not None and len(records) >= max_records:
        state.update(complete=True, truncated=True, count=len(records))
        _write_state(state_path, state)
        print(f"  [{label}] recent-window cap reached — {len(records)} records")
        return records

    end = state.get("cursor_end")  # resume cursor (epoch seconds) or None
    pages = 0
    hit_cap = False
    while pages < MAX_PAGES:
        window_min: int | None = None
        window_added = 0
        offset = 0
        while offset < api.OFFSET_CAP:
            page = page_fn(end, offset)
            pages += 1
            if not page:
                break
            fresh = []
            for r in page:
                k = _key(r)
                if k not in seen:
                    seen.add(k)
                    fresh.append(r)
            if fresh:
                _append_jsonl(jsonl_path, fresh)
                records.extend(fresh)
                window_added += len(fresh)
            tsmin = min(int(r.get("timestamp") or 0) for r in page)
            window_min = tsmin if window_min is None else min(window_min, tsmin)
            if len(page) < api.PAGE_LIMIT:
                break
            offset += api.PAGE_LIMIT
            if pages >= MAX_PAGES:
                break
        if window_min is None:  # no records at all in this window
            break
        # Advance the cursor strictly downward; dedup absorbs the boundary second.
        if end is not None and window_min >= end and window_added == 0:
            break  # no progress + nothing new -> done
        if end is not None and window_min >= end:
            # can't lower the cursor but still adding (deep single-second) -> stop, note it
            print(f"  [{label}] WARNING: cursor stuck at ts={end}; stopped (possible >10k/sec)")
            break
        end = window_min
        state.update(cursor_end=end, count=len(records))
        _write_state(state_path, state)
        print(f"  [{label}] {len(records)} records (cursor -> {end})")
        if max_records is not None and len(records) >= max_records:
            hit_cap = True
            break

    truncated = pages >= MAX_PAGES or hit_cap
    state.update(complete=True, truncated=truncated, count=len(records), cursor_end=end)
    _write_state(state_path, state)
    if truncated:
        print(f"  [{label}] TRUNCATED at MAX_PAGES — {len(records)} records (history incomplete before ts={end})")
    else:
        print(f"  [{label}] done — {len(records)} records")
    return records


def fetch_account(client: api.PolymarketClient, address: str, *,
                  max_records: int | None = None, refresh_pnl: bool = False) -> dict:
    d = account_dir(address)
    d.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {address}")

    activity = _paginate(
        lambda end, off: client.activity_page(address, offset=off, types=ACTIVITY_TYPES, end=end),
        jsonl_path=d / "activity.jsonl", state_path=d / "activity.state.json", label="activity",
        max_records=max_records,
    )
    taker = _paginate(
        lambda end, off: client.trades_page(address, offset=off, taker_only=True, end=end),
        jsonl_path=d / "taker_trades.jsonl", state_path=d / "taker_trades.state.json", label="taker",
        max_records=max_records,
    )

    # Positions snapshot (small; refetched each run unless already saved).
    pos_path = d / "positions.json"
    if not pos_path.exists():
        positions: list[dict] = []
        offset = 0
        while offset < api.OFFSET_CAP:
            page = client.positions(address, offset=offset)
            if not page:
                break
            positions.extend(page)
            if len(page) < 500:
                break
            offset += 500
        pos_path.write_text(json.dumps(positions), encoding="utf-8")
        print(f"  [positions] {len(positions)} open")
    else:
        positions = json.loads(pos_path.read_text(encoding="utf-8"))
        print(f"  [positions] cached — {len(positions)} open")

    # Official profile P/L series (one small request). Money is time-sensitive, so
    # --refresh-pnl re-pulls; otherwise fetch-if-missing for reproducibility.
    pnl_path = d / "official_pnl.json"
    if refresh_pnl or not pnl_path.exists():
        series = client.user_pnl(address, interval="all", fidelity="1d")
        pnl_path.write_text(json.dumps({
            "user_address": address, "interval": "all", "fidelity": "1d",
            "fetched_at_unix": int(time.time()), "series": series,
        }), encoding="utf-8")
        print(f"  [official_pnl] {len(series)} points"
              f"{'' if series else ' (empty — endpoint returned nothing)'}")
    else:
        series = (json.loads(pnl_path.read_text(encoding="utf-8")) or {}).get("series", [])
        print(f"  [official_pnl] cached — {len(series)} points")

    return {"address": address, "activity": len(activity),
            "taker_trades": len(taker), "positions": len(positions),
            "official_pnl": len(series)}


def load_resolved(handles_path: Path) -> list[dict]:
    data = json.loads(handles_path.read_text(encoding="utf-8"))
    return [h for h in data.get("handles", []) if h.get("address")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pull each resolved account's public history (cached).")
    ap.add_argument("--dir", default=None, help="cache dir (default data/competitors)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these addresses/handles")
    ap.add_argument("--delay", type=float, default=api.DEFAULT_DELAY_S,
                    help=f"seconds between requests (default {api.DEFAULT_DELAY_S}; keep polite)")
    ap.add_argument("--max-records", type=int, default=None,
                    help="cap cached records per endpoint (recent-window sample for accounts whose "
                         "full history is impractically large; the account is flagged recent-window)")
    ap.add_argument("--refresh-pnl", action="store_true",
                    help="re-pull the official P/L series even if cached (money is time-sensitive; "
                         "the heavy activity/taker caches are untouched)")
    args = ap.parse_args(argv)

    base = Path(args.dir) if args.dir else competitors_dir()
    resolved = load_resolved(base / "handles.json")
    if args.only:
        # honour the order the caller listed --only in (lets us fetch small accounts
        # first and put a heavy account's long tail last).
        by_handle = {h["handle"].lower(): h for h in resolved}
        by_addr = {h["address"].lower(): h for h in resolved}
        ordered = []
        for x in args.only:
            h = by_handle.get(x.lower()) or by_addr.get(x.lower())
            if h and h not in ordered:
                ordered.append(h)
        resolved = ordered

    client = api.PolymarketClient(delay_s=args.delay)
    summary = []
    for h in resolved:
        summary.append({"handle": h["handle"], **fetch_account(
            client, h["address"], max_records=args.max_records, refresh_pnl=args.refresh_pnl)})
    (base / "fetch_summary.json").write_text(
        json.dumps({"accounts": summary, "requests": client.request_count}, indent=2),
        encoding="utf-8",
    )
    print(f"[fetch] complete — {len(summary)} accounts, {client.request_count} requests")
    for s in summary:
        print(f"  {s['handle']:<44} activity={s['activity']:>7}  taker={s['taker_trades']:>7}  "
              f"positions={s['positions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
