"""Tape cross-check: corroborate the maker/taker split and measure reaction speed
to Binance fast-feed moves, using our recorded Telonex tape.

The Data API gives no maker/taker flag and only 1-second-granular fill times, so two
things are validated against the tape (``data/telonex``, coverage 2025-10 → 2026-07-07):

1. **Maker/taker corroboration.** For an account's *grid-priced* our-series fills
   (single-level fills, whose price lands on the 0.01/0.001 grid — multi-level
   marketable takes report a VWAP that maps to no single print and are skipped), we
   match to the Telonex PM print at the same token+price+~second and read the print's
   **aggressor side**. account.side == aggressor ⇒ the account was the taker. We compare
   that tape label to the ``/trades?takerOnly=true`` label and report the agreement rate.

2. **Reaction speed.** For the account's *taker* fills (from ``takerOnly``), we look at
   the Telonex **Binance top-mid** (~100 ms cadence) in the seconds before the fill and
   report the pre-fill move magnitude and the lag from the largest sub-move to the fill.
   Fills are 1-second-granular, so this is an *indicative* (± ~1 s) reaction profile, not
   a sub-second latency measurement.

Bounded/sampled by design (reads real tape while other jobs run). Writes
``out/competitors/tape_validation.json``. Run: ``python -m model_lab.competitors.analyze_tape``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .. import historical_common as hc
from ..config import resolve_paths
from ..io import telonex_pm as tpm
from .analyze import _match_key, _stats
from .paths import account_dir, competitors_dir, out_dir

# Telonex coverage window (UTC) for PM trades / Binance book.
TELONEX_LO = date(2025, 10, 11)
TELONEX_HI = date(2026, 7, 7)
GRID_TOL = 1e-5           # single-level fills sit within float-noise of a tick; VWAPs are ~1e-4 off
PRICE_MATCH_TOL = 1.5e-3  # fill↔print price match tolerance
REACT_LOOKBACK_MS = 3000  # pre-fill window for the move magnitude
REACT_JUMP_MS = 5000      # look back this far for the largest sub-move (lag)


def _is_grid(price: float) -> bool:
    """Whether a fill price sits on the 0.01 (or 0.001) tick grid — i.e. a single-level
    fill, matchable to one print. VWAP prices of multi-level takes are off-grid."""
    if not (0.0 < price < 1.0):
        return False
    g01 = abs(price * 100 - round(price * 100)) / 100
    g001 = abs(price * 1000 - round(price * 1000)) / 1000
    return min(g01, g001) <= GRID_TOL


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return


def _load_our_fills(addr: str):
    """Our-series TRADE fills within Telonex coverage, with API taker labels."""
    d = account_dir(addr)
    taker_keys = {_match_key(r) for r in _iter_jsonl(d / "taker_trades.jsonl")}
    fills = []
    for r in _iter_jsonl(d / "activity.jsonl"):
        if r.get("type") != "TRADE":
            continue
        parsed = hc.parse_slug(str(r.get("slug") or ""))
        if not parsed:
            continue
        ts = int(r.get("timestamp") or 0)
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        if not (TELONEX_LO <= day <= TELONEX_HI):
            continue
        series, asset, dur, epoch = parsed
        fills.append({
            "ts": ts, "day": day.isoformat(), "series": series, "sym": hc.SYMBOL_BY_ASSET[asset],
            "slug": str(r.get("slug")), "outcome": str(r.get("outcome") or ""),
            "token": str(r.get("asset") or ""), "side": str(r.get("side") or ""),
            "price": float(r.get("price") or 0.0), "size": float(r.get("size") or 0.0),
            "api_taker": _match_key(r) in taker_keys,
        })
    return fills


def _maker_taker_corroboration(paths, fills, days, per_day_windows: int) -> dict:
    """Match grid-priced fills to Telonex prints; compare tape aggressor label vs API."""
    # group grid fills by (day, slug, outcome)
    groups: dict[tuple, list] = defaultdict(list)
    for f in fills:
        if _is_grid(f["price"]):
            groups[(f["day"], f["slug"], f["outcome"])].append(f)
    # pick busiest groups within the sampled days
    sample = [(k, v) for k, v in groups.items() if k[0] in days]
    sample.sort(key=lambda kv: -len(kv[1]))
    sample = sample[: per_day_windows * len(days)]

    matched = agree = ambiguous = 0
    n_fills_seen = 0
    for (day, slug, outcome), gfills in sample:
        try:
            prints = tpm.read_pm_trades(paths.telonex_dir, slug, outcome, day)
        except Exception:
            continue
        if prints.empty:
            continue
        pts = prints["ts_ms"].to_numpy(dtype="int64")
        ppx = prints["price"].to_numpy(dtype=float)
        pside = prints["side"].astype(str).to_numpy()
        for f in gfills:
            n_fills_seen += 1
            lo = f["ts"] * 1000 - 1000
            hi = f["ts"] * 1000 + 2000
            cand = np.where((pts >= lo) & (pts <= hi) & (np.abs(ppx - f["price"]) <= PRICE_MATCH_TOL))[0]
            if cand.size == 0:
                continue
            sides = {s.lower() for s in pside[cand]}
            if len(sides) != 1:
                ambiguous += 1  # both aggressor sides traded here — can't isolate this fill
                continue
            matched += 1
            tape_taker = (f["side"].lower() == next(iter(sides)))
            if tape_taker == f["api_taker"]:
                agree += 1
    return {
        "grid_fills_sampled": n_fills_seen,
        "unambiguously_matched": matched,
        "ambiguous_skipped": ambiguous,
        "match_rate": round(matched / n_fills_seen, 3) if n_fills_seen else None,
        "tape_vs_api_agreement": round(agree / matched, 3) if matched else None,
        "note": "grid-priced fills only (VWAP multi-level takes excluded). A fill is labelled "
                "only when all Telonex prints at its token+price+second share one aggressor side; "
                "mixed-aggressor instants are skipped as ambiguous. Agreement on the clean subset "
                "corroborates the takerOnly split.",
    }


# Cache decompressed Binance top-mid per (sym, day) across accounts in one run — the
# book_snapshot_25 files are large, and busy accounts share the same recent days.
_MID_CACHE: dict[tuple, tuple] = {}


def _get_mid(paths, sym: str, day: str):
    key = (sym, day)
    if key not in _MID_CACHE:
        mid = hc.telonex_top_mid(paths, sym, date.fromisoformat(day))
        if mid.empty:
            _MID_CACHE[key] = (None, None)
        else:
            _MID_CACHE[key] = (mid["ts_ms"].to_numpy(dtype="int64"),
                               mid["mid"].to_numpy(dtype=float))
    return _MID_CACHE[key]


def _reaction_speed(paths, fills, days) -> dict:
    """For taker fills on the sampled days: pre-fill Binance move + lag from largest move."""
    taker_fills = [f for f in fills if f["api_taker"] and f["day"] in days]
    by_day_sym: dict[tuple, list] = defaultdict(list)
    for f in taker_fills:
        by_day_sym[(f["day"], f["sym"])].append(f)

    move_bps: list[float] = []
    jump_lag_s: list[float] = []
    frac_had_move = 0
    n = 0
    for (day, sym), fs in by_day_sym.items():
        mt, mv = _get_mid(paths, sym, day)
        if mt is None:
            continue
        for f in fs:
            t = f["ts"] * 1000
            i_now = int(np.searchsorted(mt, t, side="right")) - 1
            i_prev = int(np.searchsorted(mt, t - REACT_LOOKBACK_MS, side="right")) - 1
            if i_now < 0 or i_prev < 0:
                continue
            n += 1
            if mv[i_prev] > 0 and mv[i_now] > 0:
                bps = abs(math.log(mv[i_now] / mv[i_prev])) * 1e4
                move_bps.append(bps)
                if bps >= 5.0:
                    frac_had_move += 1
            # largest 1-sample jump in [t-REACT_JUMP_MS, t] and its lag to the fill
            lo = int(np.searchsorted(mt, t - REACT_JUMP_MS, side="left"))
            seg = mv[lo:i_now + 1]
            segt = mt[lo:i_now + 1]
            if seg.size >= 2:
                dlog = np.abs(np.diff(np.log(np.clip(seg, 1e-9, None))))
                k = int(np.argmax(dlog))
                jump_lag_s.append((t - int(segt[k + 1])) / 1000.0)
    return {
        "taker_fills_evaluated": n,
        "pre_fill_move_bps_3s": _stats(move_bps),
        "frac_with_gt5bps_move_3s": round(frac_had_move / n, 3) if n else None,
        "lag_from_largest_move_s": _stats(jump_lag_s),
        "note": "1s-granular fill times vs Telonex Binance top-mid; indicative (±~1s), not "
                "sub-second. A momentum taker shows a >5bps move shortly before its takes.",
    }


def analyze_account_tape(entry: dict, *, n_days: int, per_day_windows: int) -> dict:
    addr = entry["address"]
    paths = resolve_paths()
    fills = _load_our_fills(addr)
    if not fills:
        return {"handle": entry["handle"], "address": addr,
                "note": "no our-series fills within Telonex coverage"}
    # pick the busiest covered days
    by_day = defaultdict(int)
    for f in fills:
        by_day[f["day"]] += 1
    days = {d for d, _ in sorted(by_day.items(), key=lambda kv: -kv[1])[:n_days]}
    return {
        "handle": entry["handle"], "address": addr,
        "sampled_days": sorted(days),
        "our_fills_in_coverage": len(fills),
        "maker_taker_corroboration": _maker_taker_corroboration(paths, fills, days, per_day_windows),
        "reaction_speed": _reaction_speed(paths, fills, days),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tape cross-check: maker/taker + reaction speed.")
    ap.add_argument("--only", nargs="*", default=None, help="handles/addresses (default: a representative few)")
    ap.add_argument("--days", type=int, default=4, help="busiest covered days to sample per account")
    ap.add_argument("--per-day-windows", type=int, default=30)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    base = Path(args.dir) if args.dir else competitors_dir()
    handles = json.loads((base / "handles.json").read_text(encoding="utf-8"))["handles"]
    resolved = [h for h in handles if h.get("address")]
    if args.only:
        want = {x.lower() for x in args.only}
        resolved = [h for h in resolved if h["address"].lower() in want or h["handle"].lower() in want]

    results = []
    for h in resolved:
        print(f"[tape] {h['handle']} ({h['address']})")
        try:
            results.append(analyze_account_tape(h, n_days=args.days, per_day_windows=args.per_day_windows))
        except Exception as e:  # keep going; tape data can be patchy
            print(f"  [tape] error: {type(e).__name__}: {e}")
            results.append({"handle": h["handle"], "address": h["address"], "error": f"{type(e).__name__}: {e}"})

    odir = out_dir()
    odir.mkdir(parents=True, exist_ok=True)
    out = odir / "tape_validation.json"
    # merge with any existing results (so incremental --only runs accumulate)
    existing = {}
    if out.exists():
        try:
            existing = {a["address"]: a for a in json.loads(out.read_text(encoding="utf-8"))["accounts"]
                        if a.get("address")}
        except (OSError, json.JSONDecodeError, KeyError):
            existing = {}
    for r in results:
        existing[r["address"]] = r
    out.write_text(json.dumps({"accounts": list(existing.values())}, indent=2, default=str), encoding="utf-8")
    print(f"[tape] wrote {out} — {len(results)} updated, {len(existing)} total")
    for r in results:
        mt = r.get("maker_taker_corroboration", {})
        rs = r.get("reaction_speed", {})
        print(f"  {r['handle']:<40} agree={mt.get('tape_vs_api_agreement')} "
              f"(n={mt.get('matched_to_print')})  react>5bps={rs.get('frac_with_gt5bps_move_3s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
