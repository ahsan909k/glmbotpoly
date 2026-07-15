"""Deep per-window reconstruction → one **operating manual** per competitor account.

For each account (0xb27b, takerner, bonereaper, wolf9478, nagi777) this reconstructs,
from the cached ``data/competitors/<addr>/activity.jsonl`` fills joined to our recorded
Telonex Polymarket book tape, HOW they trade each window — as **distributions, not
averages**:

- entry timing within the window (maker vs taker),
- price level vs the book at fill time (at-touch / inside-spread / crossing) [tape-required],
- sequencing (both sides ~simultaneous vs alternating),
- inventory trajectory (peak/end skew),
- size ramping (per-fill notional, ramp slope),
- merge timing (from MERGE events),
- per-window capital deployed.

For **nagi777** specifically it dissects the merge mechanics: frequency, inventory-state
at merge (matched vs skewed), merge timing, capital recycled per merge, and effective
capital velocity (turns/day) — vs a hold-to-resolution account (takerner).

**Provenance (F).** Every number is FACT (raw records) or CALC (exact arithmetic on
them). The behavioral manual carries FACT+CALC only; anything requiring an assumption
(estimated fees, inferred triggers, queue position) is quarantined to a separate
"inference" section (see ``manuals_report``). Each account's reconstruction is
**anchored** against its official Polymarket P/L (:func:`anchor_reconcile`): if the
reconstructed OUR-series economics contradict the official number the discrepancy is
flagged and OUR reconstruction is treated as suspect — never their profile.

What is **not observable** is flagged explicitly (their signals/triggers, true queue
position, off-grid sweep depth, cancels/replaces — only fills are recorded).

Writes ``out/competitors/manuals/<handle>.json``. Run:
``python -m model_lab.competitors.manuals --only <handle>``.
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
from .. import money_judge as mj
from ..config import resolve_paths
from ..io import telonex_pm as tpm
from ..lib import math as lm
from . import analyze
from .analyze import OUR_SET, _iter_jsonl, _match_key, _official_pnl, _p_at, classify_slug
from .paths import account_dir, competitors_dir, out_dir

# Telonex Polymarket coverage window (UTC): the only days a real book exists to join.
TELONEX_LO = date(2025, 10, 11)
TELONEX_HI = date(2026, 7, 7)

# The current-regime boundary (250 ms taker delay locked) — the post-Jun-5 slice used for
# the clone-vs-owner calibration ladder (Q3).
REGIME_START = date(2026, 6, 5)


def _day_start_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) * 1000


REGIME_START_MS = _day_start_ms(REGIME_START)


# ===========================================================================
# pure helpers (unit-tested directly)
# ===========================================================================
def _tick(price: float) -> float:
    """The market tick at ``price`` — 0.001 past the 0.04/0.96 boundary, else 0.01
    (mirrors the venue's tick-flip; used for the half-tick classification tolerance)."""
    return 0.001 if (price < 0.04 or price > 0.96) else 0.01


def price_class(side: str, price: float, bid_px: float, ask_px: float) -> str:
    """Classify a fill's price vs the book at fill time as ``at_touch`` / ``inside`` /
    ``crossing`` (tolerance = half a tick).

    BUY: at-touch if resting at/below the bid, crossing if at/above the ask (marketable),
    inside otherwise. SELL mirrors (passive at the ask, crossing at the bid). Returns
    ``"unknown"`` on a degenerate book."""
    if not (math.isfinite(price) and math.isfinite(bid_px) and math.isfinite(ask_px)
            and 0.0 < bid_px <= ask_px < 1.0):
        return "unknown"
    tol = 0.5 * _tick(price)
    if side == "BUY":
        if price <= bid_px + tol:
            return "at_touch"
        if price >= ask_px - tol:
            return "crossing"
        return "inside"
    # SELL
    if price >= ask_px - tol:
        return "at_touch"
    if price <= bid_px + tol:
        return "crossing"
    return "inside"


def _pctiles(vals: list[float], qs=(5, 10, 25, 50, 75, 90, 95)) -> dict:
    """Percentiles ``{p5,p10,...}`` (linear interpolation), ``None`` on empty."""
    s = sorted(v for v in vals if v is not None and math.isfinite(v))
    out: dict = {}
    if not s:
        return {f"p{q}": None for q in qs}
    for q in qs:
        idx = (q / 100.0) * (len(s) - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, len(s) - 1)
        out[f"p{q}"] = round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 6)
    return out


def _histogram(vals: list[float], edges) -> list[int]:
    """Integer counts per bin (``len(edges) − 1`` bins); values are clipped into
    ``[edges[0], edges[-1]]`` so nothing is silently dropped."""
    clean = np.array([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    if clean.size == 0:
        return [0] * (len(edges) - 1)
    clipped = np.clip(clean, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=np.asarray(edges, dtype=float))
    return [int(c) for c in counts]


def _distribution(vals, *, edges=None) -> dict:
    """A distribution summary: ``n`` + percentiles (+ a histogram when ``edges`` given).

    Deliberately carries no bare mean as a headline — the manual reports distributions."""
    clean = [float(v) for v in vals if v is not None and math.isfinite(v)]
    d: dict = {"n": len(clean)}
    d.update(_pctiles(clean))
    if edges is not None:
        d["hist_edges"] = [float(e) for e in edges]
        d["hist_counts"] = _histogram(clean, edges)
    return d


def _ols_slope(y: list[float]) -> float | None:
    """Slope of ``y`` vs its index (0..n−1), or ``None`` if < 2 points / degenerate."""
    n = len(y)
    if n < 2:
        return None
    x = np.arange(n, dtype=float)
    yv = np.asarray(y, dtype=float)
    xm, ym = x.mean(), yv.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= 0.0:
        return None
    return float(((x - xm) * (yv - ym)).sum() / denom)


# ===========================================================================
# fill loading + per-window grouping (streaming — the 2 GB guard)
# ===========================================================================
def load_our_activity(addr: str, *, since_ms: int | None = None, until_ms: int | None = None,
                      telonex_only: bool = False) -> dict:
    """Stream ``activity.jsonl`` ONCE and keep only OUR-series records (the 2 GB guard —
    filter before building any per-window structure).

    Returns ``{"fills": [...], "events": [...], "taker_keys": set, "n_scanned": int}``.
    A *fill* is an OUR-series ``TRADE`` dict (classified by the ``outcome`` string + ``side``,
    never ``outcomeIndex`` which is the 999 sentinel on real rows). An *event* is an OUR-series
    MERGE/REDEEM/SPLIT/REWARD/CONVERSION (cash events, ``price=0``, ``size==usdcSize``).

    The **fills-only** behaviors (timing, sequencing, inventory, size, merge mechanics /
    velocity) are computed over ALL OUR-series fills in ``[since_ms, until_ms)``; the
    **tape-required** ``price_vs_book`` / hand-trace gate on actual Telonex book presence
    (they report ``book_missing`` for windows outside coverage). ``telonex_only=True``
    restricts to the Telonex window up front (a memory/time bound for ultra-active accounts
    whose full OUR-series history is huge — set it when only the tape-joined metrics matter)."""
    d = account_dir(addr)
    taker_keys = {_match_key(r) for r in _iter_jsonl(d / "taker_trades.jsonl")}
    fills: list[dict] = []
    events: list[dict] = []
    n_scanned = 0
    for r in _iter_jsonl(d / "activity.jsonl"):
        n_scanned += 1
        ts = int(r.get("timestamp") or 0)
        ts_ms = ts * 1000
        if not ts:
            continue
        if since_ms is not None and ts_ms < since_ms:
            continue
        if until_ms is not None and ts_ms >= until_ms:
            continue
        if telonex_only and not (TELONEX_LO <= hc.utc_day(ts_ms) <= TELONEX_HI):
            continue
        parsed = classify_slug(str(r.get("slug") or ""))
        if parsed is None or parsed[0] not in OUR_SET:
            continue
        series, asset, dur, epoch = parsed
        typ = r.get("type")
        common = {
            "ts": ts, "ts_ms": ts_ms, "series": series, "asset": asset, "dur": dur,
            "open_epoch": epoch, "open_ms": epoch * 1000, "slug": str(r.get("slug") or ""),
            "cond": str(r.get("conditionId") or ""), "type": typ,
            "usd": float(r.get("usdcSize") or 0.0), "size": float(r.get("size") or 0.0),
        }
        if typ == "TRADE":
            common.update({
                "side": str(r.get("side") or ""), "outcome": str(r.get("outcome") or ""),
                "price": float(r.get("price") or 0.0),
                "api_taker": _match_key(r) in taker_keys,
                "token": str(r.get("asset") or ""),
            })
            fills.append(common)
        elif typ in ("MERGE", "REDEEM", "SPLIT", "REWARD", "CONVERSION"):
            events.append(common)
    return {"fills": fills, "events": events, "taker_keys": taker_keys, "n_scanned": n_scanned}


def group_by_window(fills: list[dict]) -> dict:
    """Group fills by ``(series, open_ms)`` (each == one window). Fills sorted by ts."""
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for f in fills:
        by_win[(f["series"], f["open_ms"])].append(f)
    for k in by_win:
        by_win[k].sort(key=lambda x: x["ts"])
    return dict(by_win)


# ===========================================================================
# per-behavior reconstruction (distributions)
# ===========================================================================
_TIMING_EDGES = np.linspace(0.0, 1.0, 11)          # 10 bins over the window life
_PRICE_TICK_EDGES = np.arange(-5.5, 6.5, 1.0)      # signed distance-to-touch, in ticks
_SKEW_EDGES = np.linspace(0.0, 1.0, 11)
_SIZE_EDGES = np.array([0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000], dtype=float)


def entry_timing(by_win: dict) -> dict:
    """Fraction of the window elapsed at each fill (0 = open, 1 = close), split
    maker/taker. Fills-only (needs no book)."""
    taker, maker = [], []
    for (series, open_ms), fills in by_win.items():
        for f in fills:
            dur = f["dur"]
            if dur <= 0:
                continue
            frac = (f["ts_ms"] - open_ms) / (dur * 1000.0)
            if 0.0 <= frac <= 1.2:
                (taker if f["api_taker"] else maker).append(min(1.0, frac))
    return {"taker": _distribution(taker, edges=_TIMING_EDGES),
            "maker": _distribution(maker, edges=_TIMING_EDGES),
            "note": "fraction of window elapsed at fill; 1s-granular API timestamps (FACT ts, CALC frac)"}


def sequencing(by_win: dict) -> dict:
    """Both-sides-simultaneous vs alternating per window: first-leg gap (s), same-side
    run lengths, and the adjacent-alternation ratio. Fills-only."""
    leg_gaps, alt_ratios, run_lens = [], [], []
    simultaneous = alternating = single_sided = 0
    for fills in by_win.values():
        buys = [f for f in fills if f["side"] == "BUY"]
        if not buys:
            continue
        first_up = min((f["ts"] for f in buys if f["outcome"] == "Up"), default=None)
        first_dn = min((f["ts"] for f in buys if f["outcome"] == "Down"), default=None)
        sides = [f["outcome"] for f in buys]
        if first_up is None or first_dn is None:
            single_sided += 1
            continue
        gap = abs(first_dn - first_up)
        leg_gaps.append(float(gap))
        # adjacent alternation ratio + run lengths
        switches = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
        alt = switches / max(1, len(sides) - 1)
        alt_ratios.append(alt)
        run = 1
        for a, b in zip(sides, sides[1:]):
            if a == b:
                run += 1
            else:
                run_lens.append(float(run)); run = 1
        run_lens.append(float(run))
        if gap <= 2 and alt >= 0.4:
            simultaneous += 1
        else:
            alternating += 1
    total = simultaneous + alternating
    return {
        "first_leg_gap_secs": _distribution(leg_gaps),
        "alternation_ratio": _distribution(alt_ratios),
        "same_side_run_len": _distribution(run_lens),
        "simultaneous_windows": simultaneous,
        "alternating_windows": alternating,
        "single_sided_windows": single_sided,
        "simultaneous_frac": round(simultaneous / total, 4) if total else None,
        "note": "simultaneous = leg-gap ≤ 2 s AND alternation ≥ 0.4 (CALC over FACT fills)",
    }


def inventory_trajectory(by_win: dict) -> dict:
    """Per window: peak |signed Up−Down excess|, end-skew ``|up−dn|/(up+dn)``, and the
    area under |excess| over the fill sequence. Fills-only."""
    peaks, end_skews = [], []
    for fills in by_win.values():
        up = dn = 0.0
        peak = 0.0
        for f in fills:
            if f["side"] != "BUY":
                continue
            if f["outcome"] == "Up":
                up += f["size"]
            elif f["outcome"] == "Down":
                dn += f["size"]
            peak = max(peak, abs(up - dn))
        if up + dn <= 0.0:
            continue
        peaks.append(peak)
        end_skews.append(abs(up - dn) / (up + dn))
    return {"peak_excess_shares": _distribution(peaks),
            "end_skew": _distribution(end_skews, edges=_SKEW_EDGES),
            "note": "signed Up−Down BUY inventory over the window (CALC over FACT fills)"}


def size_ramping(by_win: dict) -> dict:
    """Per-fill notional ($) distribution + the per-window size-ramp slope (OLS of fill
    size vs fill index) and last/first ratio. Fills-only."""
    notionals, slopes, ratios = [], [], []
    for fills in by_win.values():
        buys = [f for f in fills if f["side"] == "BUY"]
        for f in buys:
            notionals.append(f["usd"])
        sizes = [f["size"] for f in buys]
        if len(sizes) >= 2:
            sl = _ols_slope(sizes)
            if sl is not None:
                slopes.append(sl)
            if sizes[0] > 0:
                ratios.append(sizes[-1] / sizes[0])
    return {"per_fill_notional_usd": _distribution(notionals, edges=_SIZE_EDGES),
            "ramp_slope_shares_per_fill": _distribution(slopes),
            "last_over_first_size": _distribution(ratios),
            "note": "per-window BUY fill sizes; ramp slope = OLS of size vs fill index (CALC)"}


def merge_timing(events: list[dict], win_open: dict) -> dict:
    """For each MERGE event, the fraction of the window elapsed at merge. ``win_open``
    maps ``cond → (open_ms, dur)`` (from the account's fills). Fills-only (event ts)."""
    fracs = []
    for e in events:
        if e["type"] != "MERGE":
            continue
        wo = win_open.get(e["cond"])
        if wo is None:
            continue
        open_ms, dur = wo
        if dur <= 0:
            continue
        frac = (e["ts_ms"] - open_ms) / (dur * 1000.0)
        if 0.0 <= frac <= 1.5:
            fracs.append(min(1.0, frac))
    return {"merge_frac_of_window": _distribution(fracs, edges=_TIMING_EDGES),
            "note": "fraction of window elapsed at MERGE (FACT event ts, CALC frac)"}


def per_window_capital(by_win: dict) -> dict:
    """Gross BUY notional deployed per window (Σ usd of BUY fills). Fills-only."""
    gross = []
    for fills in by_win.values():
        g = sum(f["usd"] for f in fills if f["side"] == "BUY")
        if g > 0:
            gross.append(g)
    return {"gross_deployed_usd": _distribution(gross, edges=_SIZE_EDGES),
            "note": "Σ BUY notional per window (FACT usdcSize; gross deployed, not peak concurrent)"}


def price_vs_book(paths, by_win: dict, *, max_windows: int | None = None) -> dict:
    """At-touch / inside-spread / crossing classification of each BUY/SELL vs the book
    as-of the fill (**tape-required**). Loads each window's book once (no cross-window
    cache — books discarded after use, memory-bounded). Windows with no fresh book are
    counted as ``book_missing``. Reports the 3-way fraction + a signed-distance-to-touch
    (in ticks) histogram.

    ``max_windows`` bounds the number of windows whose book is loaded (an evenly-strided,
    deterministic sample) — the per-window Telonex ``.zst`` reads dominate the runtime for
    ultra-active accounts (b27b has tens of thousands of windows). The sampling is reported
    (``windows_sampled`` / ``windows_total``); the fills-only behaviors are unaffected."""
    counts = {"at_touch": 0, "inside": 0, "crossing": 0, "unknown": 0}
    dist_ticks = []
    windows_with_book = book_missing = 0
    keys = sorted(by_win.keys())
    windows_total = len(keys)
    if max_windows is not None and windows_total > max_windows:
        step = windows_total / max_windows
        keys = [keys[int(i * step)] for i in range(max_windows)]
    sampled = {k: by_win[k] for k in keys}
    for (series, open_ms), fills in sampled.items():
        slug = fills[0]["slug"]
        day_str = hc.utc_day(open_ms).isoformat()
        books = {}
        for oc in ("Up", "Down"):
            q = _safe_read_quotes(paths, slug, oc, day_str)
            if q is not None and not q.empty:
                books[oc] = tpm.pm_book_arrays(q)
        if not books:
            book_missing += 1
            continue
        windows_with_book += 1
        for f in fills:
            book = books.get(f["outcome"])
            if book is None:
                counts["unknown"] += 1
                continue
            # `book` is a single token's pm_book_arrays; use the direct single-token as-of.
            row = _asof_single(book, f["ts_ms"])
            if row is None:
                counts["unknown"] += 1
                continue
            bid_px, _bid_sz, ask_px, _ask_sz = row
            cls = price_class(f["side"], f["price"], bid_px, ask_px)
            counts[cls] += 1
            if cls != "unknown":
                touch = bid_px if f["side"] == "BUY" else ask_px
                dist_ticks.append((f["price"] - touch) / _tick(f["price"]))
    total = counts["at_touch"] + counts["inside"] + counts["crossing"]

    def frac(a):
        return round(a / total, 4) if total else None
    return {
        "classified_fills": total,
        "at_touch": counts["at_touch"], "inside": counts["inside"], "crossing": counts["crossing"],
        "unknown_book": counts["unknown"],
        "at_touch_frac": frac(counts["at_touch"]), "inside_frac": frac(counts["inside"]),
        "crossing_frac": frac(counts["crossing"]),
        "signed_dist_to_touch_ticks": _distribution(dist_ticks, edges=_PRICE_TICK_EDGES),
        "windows_with_book": windows_with_book, "windows_book_missing": book_missing,
        "windows_total": windows_total, "windows_sampled": len(sampled),
        "note": "BUY/SELL price vs the Telonex book as-of the fill; half-tick tolerance "
                "(tape-required — only windows with a fresh book counted; CALC). "
                + (f"Book sampled over {len(sampled):,}/{windows_total:,} windows (--max-book-windows)."
                   if len(sampled) < windows_total else ""),
    }


def _asof_single(book: dict, at_ms: int, staleness_ms: int = mj.BOOK_STALENESS_MS):
    """Backward as-of on a single token's ``pm_book_arrays`` output (mirrors
    ``money_judge._book_asof`` but for one token's arrays directly)."""
    ts = book["ts"]
    if ts.size == 0:
        return None
    i = int(np.searchsorted(ts, at_ms, side="right")) - 1
    if i < 0 or (at_ms - int(ts[i])) > staleness_ms:
        return None
    return (float(book["bid_px"][i]), float(book["bid_sz"][i]),
            float(book["ask_px"][i]), float(book["ask_sz"][i]))


def _safe_read_quotes(paths, slug, outcome, day_str):
    try:
        return tpm.read_pm_quotes(paths.telonex_dir, slug, outcome, day_str)
    except Exception:
        return None


# ===========================================================================
# nagi merge-velocity (item 3)
# ===========================================================================
def merge_velocity(by_win: dict, events: list[dict], *, active_days: int) -> dict:
    """Merge mechanics + effective capital velocity for a merge-heavy account.

    - frequency: MERGE events / active-day and merges / traded-window.
    - inventory-state-at-merge: reconstruct inventory from fills with ``ts ≤ merge_ts`` in
      the window; ``skew = |up−dn|/(up+dn)``; matched if ``≤ 0.1``. Matched-fraction + histogram.
    - timing: fraction of window elapsed at merge.
    - capital recycled / merge: MERGE usdcSize (FACT).
    - velocity (turns/day): daily gross BUY notional ÷ median within-day peak concurrent
      open capital, where a MERGE closes (frees) its matched cost early."""
    win_open = {}
    fills_by_cond: dict[str, list[dict]] = defaultdict(list)
    for (series, open_ms), fills in by_win.items():
        cond = fills[0]["cond"]
        win_open[cond] = (open_ms, fills[0]["dur"])
        fills_by_cond[cond] = fills

    merges = [e for e in events if e["type"] == "MERGE"]
    recycled = [e["usd"] for e in merges]
    skews, matched = [], 0
    for e in merges:
        fc = fills_by_cond.get(e["cond"])
        if not fc:
            continue
        up = sum(f["size"] for f in fc if f["side"] == "BUY" and f["outcome"] == "Up" and f["ts"] <= e["ts"])
        dn = sum(f["size"] for f in fc if f["side"] == "BUY" and f["outcome"] == "Down" and f["ts"] <= e["ts"])
        if up + dn <= 0:
            continue
        sk = abs(up - dn) / (up + dn)
        skews.append(sk)
        if sk <= 0.1:
            matched += 1

    windows_traded = len(by_win)
    turns = _capital_velocity(by_win, merges, win_open)
    return {
        "merge_events": len(merges),
        "merges_per_active_day": round(len(merges) / active_days, 3) if active_days else None,
        "merges_per_traded_window": round(len(merges) / windows_traded, 4) if windows_traded else None,
        "inventory_state_at_merge": {
            "matched_frac": round(matched / len(skews), 4) if skews else None,
            "skew": _distribution(skews, edges=_SKEW_EDGES),
        },
        "merge_timing": merge_timing(events, win_open),
        "capital_recycled_per_merge_usd": _distribution(recycled, edges=_SIZE_EDGES),
        "total_recycled_usd": round(sum(recycled), 2),
        "capital_velocity": turns,
        "note": "MERGE mechanics joined to fills (FACT events + FACT fills → CALC).",
    }


def _capital_velocity(by_win: dict, merges: list[dict], win_open: dict) -> dict:
    """Turns/day = daily gross BUY notional ÷ median within-day peak concurrent open
    capital. Open capital = Σ BUY cost still open; a window's cost is freed at its close
    (or earlier if a MERGE frees the matched portion). Distribution over active days."""
    # events per day: opens (buy cost), closes (free at window close), merge-frees (early)
    by_day_gross: dict[date, float] = defaultdict(float)
    events: dict[date, list[tuple[int, float]]] = defaultdict(list)  # day -> (ts_ms, delta_open)
    merges_by_cond: dict[str, list[dict]] = defaultdict(list)
    for e in merges:
        merges_by_cond[e["cond"]].append(e)
    for (series, open_ms), fills in by_win.items():
        cond = fills[0]["cond"]
        dur = fills[0]["dur"]
        close_ms = open_ms + dur * 1000
        gross = sum(f["usd"] for f in fills if f["side"] == "BUY")
        up_cost = sum(f["usd"] for f in fills if f["side"] == "BUY" and f["outcome"] == "Up")
        dn_cost = sum(f["usd"] for f in fills if f["side"] == "BUY" and f["outcome"] == "Down")
        matched_cost = min(up_cost, dn_cost)
        d = hc.utc_day(open_ms)
        by_day_gross[d] += gross
        # open the full cost at first fill; free at close
        events[d].append((fills[0]["ts_ms"], +gross))
        # a merge frees the matched cost early (once), the rest frees at close
        mlist = sorted(merges_by_cond.get(cond, []), key=lambda x: x["ts"])
        freed_early = 0.0
        for m in mlist:
            free = min(matched_cost - freed_early, m["usd"])
            if free > 0:
                events[d].append((m["ts_ms"], -free))
                freed_early += free
        events[d].append((close_ms, -(gross - freed_early)))

    turns = []
    for d, gross in by_day_gross.items():
        evs = sorted(events[d], key=lambda x: (x[0], x[1]))  # frees before opens at a tie
        cur = peak = 0.0
        for _ts, delta in evs:
            cur += delta
            peak = max(peak, cur)
        if peak > 0:
            turns.append(gross / peak)
    return {"turns_per_day": _distribution(turns),
            "note": "gross BUY notional / peak concurrent open capital, MERGE frees matched cost early (CALC)"}


def hold_velocity(by_win: dict) -> dict:
    """The takerner-style comparison: turns/day with NO early frees (positions held to
    the window close). Capital turns ~once per window lifetime."""
    return {"turns_per_day": _capital_velocity(by_win, [], {})["turns_per_day"],
            "note": "no early frees (hold-to-resolution) — capital locked open→close (CALC)"}


# ===========================================================================
# official-P/L anchor + clone-vs-owner ladder (F)
# ===========================================================================
#: Symmetric anchor tolerances. A recon-vs-official disagreement beyond ``max(ABS, FRAC ×
#: larger-magnitude)`` — OR an opposite-sign disagreement material on both sides — is a
#: contradiction and is flagged SUSPECT (both directions), never rationalized.
SUSPECT_ABS_TOL = 2_000.0
SUSPECT_FRAC_TOL = 0.5
SUSPECT_SIGN_MATERIAL = 2_000.0


def recon_contradicts_official(recon, official) -> tuple[bool, str]:
    """Does the OUR-series reconstruction *contradict* the account-wide official P/L?

    The anchor rule (F), applied **symmetrically**: a reconstruction materially above OR
    below official, or an opposite-sign disagreement (both sides material), is a
    contradiction → ``(True, factual-reason)``; the reconstruction is treated as
    UNRELIABLE, with no narrative ("profit is elsewhere") rationalization. ``(False, "")``
    when the two are consistent within tolerance, or either is missing."""
    if not isinstance(recon, (int, float)) or not isinstance(official, (int, float)):
        return False, ""
    gap = recon - official
    tol = max(SUSPECT_ABS_TOL, SUSPECT_FRAC_TOL * max(abs(official), abs(recon)))
    sign_flip = ((recon > SUSPECT_SIGN_MATERIAL and official < -SUSPECT_SIGN_MATERIAL)
                 or (recon < -SUSPECT_SIGN_MATERIAL and official > SUSPECT_SIGN_MATERIAL))
    if abs(gap) > tol or sign_flip:
        return True, (f"reconstructed OUR-series net ${recon:,.0f} contradicts the official "
                      f"account-wide slice ${official:,.0f} (gap ${gap:,.0f} > tolerance ${tol:,.0f}"
                      f"{'; opposite signs' if sign_flip else ''}) — reconstruction treated as "
                      f"UNRELIABLE, not explained away.")
    return False, ""


class AnchorInconsistency(RuntimeError):
    """Raised when a recon-vs-official contradiction is presented WITHOUT a SUSPECT flag."""


def assert_anchor_consistent(anchor: dict) -> None:
    """FAIL LOUDLY (F, part 4): a reconstruction that contradicts official beyond tolerance MUST
    carry ``suspect=True`` (full-window) and ``slice_ladder.suspect=True`` (post-Jun-5 slice).
    Raises :class:`AnchorInconsistency` on any contradiction that is not flagged — a permanent
    guard against a future rule change, stale data, or a report silently presenting an
    unreconcilable reconstruction as a clean number. No-op when no official P/L is available."""
    if not anchor.get("official_available"):
        return
    s, reason = recon_contradicts_official(anchor.get("reconstructed_our_net_usd"),
                                           anchor.get("official_window_usd"))
    if s and not anchor.get("suspect"):
        raise AnchorInconsistency(f"full-window: {reason} — but suspect flag is not set")
    lad = anchor.get("slice_ladder", {})
    s2, reason2 = recon_contradicts_official(lad.get("owner_our_series_reconstructed_net_usd"),
                                             lad.get("official_account_wide_slice_usd"))
    if s2 and not lad.get("suspect"):
        raise AnchorInconsistency(f"post-Jun-5 slice: {reason2} — but slice_ladder.suspect is not set")


def _reconstructed_net(fills: list[dict], events: list[dict], *,
                       lo_ms: int | None = None, hi_ms: int | None = None) -> dict:
    """OUR-series reconstructed net cash flow over ``[lo_ms, hi_ms)`` — the same
    methodology as ``analyze.py``'s ledger (BUY = −usd−fee, SELL = +usd−fee, plus the
    actual REDEEM/MERGE/REWARD/CONVERSION usd, minus SPLIT usd), restricted to OUR series
    and the window. Fees are ESTIMATED (documented formula) → this is a reconstruction,
    shown alongside the FACT official number, never asserted as equal."""
    def keep(x):
        t = x["open_ms"]
        return (lo_ms is None or t >= lo_ms) and (hi_ms is None or t < hi_ms)
    net = fees = 0.0
    for f in fills:
        if not keep(f):
            continue
        fee = lm.taker_fee(f["size"], analyze.FEE_RATE, f["price"]) if f["api_taker"] else 0.0
        fees += fee
        net += (f["usd"] - fee) if f["side"] == "SELL" else -(f["usd"] + fee)
    for e in events:
        if not keep(e):
            continue
        if e["type"] == "SPLIT":
            net -= e["usd"]
        else:  # REDEEM / MERGE / REWARD / CONVERSION
            net += e["usd"]
    return {"net_usd": round(net, 2), "est_fees_usd": round(fees, 2)}


def anchor_reconcile(addr: str, fills: list[dict], events: list[dict]) -> dict:
    """Reconcile the OUR-series reconstruction against the account's OFFICIAL Polymarket
    P/L (the anchor rule, F), and compute the post-Jun-5 clone-vs-owner ladder inputs.

    Full-window anchor: reconstructed OUR-series net vs the account-wide official slice
    over the captured window. The official curve is ACCOUNT-WIDE (all markets) while the
    reconstruction is OUR-4-series-only, so we never assert equality — but if the
    reconstruction *contradicts* official (reconstructed net exceeds the account-wide
    slice, or signs disagree, or the residual dominates) the account is flagged
    ``suspect`` and OUR reconstruction is treated as the error.

    Post-Jun-5 slice (Q3): (1) account-wide official slice (FACT) + (2) owner's OUR-series
    reconstructed net over the slice (CALC/EST) — the clone report appends (3) the clone
    backtest net and prints the two gaps."""
    off = _official_pnl(addr)
    if not fills and not events:
        first_ms = None
    else:
        first_ms = min([f["ts_ms"] for f in fills] + [e["ts_ms"] for e in events], default=None)

    recon_full = _reconstructed_net(fills, events)
    result: dict = {
        "official_available": bool(off),
        "reconstructed_our_net_usd": recon_full["net_usd"],
        "reconstructed_est_fees_usd": recon_full["est_fees_usd"],
    }
    if not off:
        result.update({"suspect": False, "note": "no official P/L returned — anchor unavailable",
                       "official_window_usd": None})
        result["slice_ladder"] = _slice_ladder(None, fills, events)
        return result

    series = off["series"]
    first_t = (first_ms // 1000) if first_ms is not None else off["first_t"]
    official_window = round(off["all_usd"] - _p_at(series, first_t), 2)
    recon = recon_full["net_usd"]
    # Symmetric contradiction rule (F): any material recon-vs-official disagreement, either
    # direction, is flagged SUSPECT and the reconstruction is treated as the unreliable side.
    suspect, reason = recon_contradicts_official(recon, official_window)
    result.update({
        "official_all_usd": off["all_usd"],
        "official_window_usd": official_window,
        "official_as_of": off["as_of"],
        "window_from": datetime.fromtimestamp(first_t, tz=timezone.utc).isoformat() if first_t else None,
        "suspect": suspect,
        "suspect_reasons": [reason] if reason else [],
        "note": "official P/L (user-pnl-api) is authoritative; the OUR-series reconstruction is a "
                "cash-flow CROSS-CHECK. Any material contradiction with official — EITHER direction — "
                "is flagged SUSPECT and the reconstruction treated as unreliable, never rationalized.",
    })
    result["slice_ladder"] = _slice_ladder(off, fills, events)
    return result


def _slice_ladder(off: dict | None, fills: list[dict], events: list[dict]) -> dict:
    """The post-Jun-5 ladder inputs: (1) account-wide official slice, (2) owner's OUR-series
    reconstructed net over the slice. The clone report appends (3)."""
    recon_slice = _reconstructed_net(fills, events, lo_ms=REGIME_START_MS)
    ladder = {
        "slice_from": REGIME_START.isoformat(),
        "owner_our_series_reconstructed_net_usd": recon_slice["net_usd"],  # (2) CALC/EST
        "owner_our_series_est_fees_usd": recon_slice["est_fees_usd"],
    }
    if off:
        official_slice = round(off["all_usd"] - _p_at(off["series"], REGIME_START_MS // 1000), 2)
        ladder["official_account_wide_slice_usd"] = official_slice  # (1) FACT
        suspect, reason = recon_contradicts_official(recon_slice["net_usd"], official_slice)
        ladder["suspect"] = suspect
        ladder["suspect_reason"] = reason
    else:
        ladder["official_account_wide_slice_usd"] = None
        ladder["suspect"] = False
        ladder["suspect_reason"] = ""
    return ladder


# ===========================================================================
# 10-window hand-trace appendix (F)
# ===========================================================================
def hand_trace_windows(paths, by_win: dict, *, n: int = 10, seed: int = 4242) -> list[dict]:
    """Pick ``n`` random OUR-series windows and, for each fill, print the account's fill
    (ts, side, outcome, price, size) next to the Telonex book as-of the fill (bid/ask,
    both timestamps) and the classification assigned — so any claim is re-derivable by eye.
    A behavior that cannot be established from records prints ``"unknown"``."""
    keys = sorted(by_win.keys())
    if not keys:
        return []
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in rng.permutation(len(keys))[:n]]
    traces: list[dict] = []
    for (series, open_ms) in sorted(pick):
        fills = by_win[(series, open_ms)]
        slug = fills[0]["slug"]
        day_str = hc.utc_day(open_ms).isoformat()
        books = {}
        for oc in ("Up", "Down"):
            q = _safe_read_quotes(paths, slug, oc, day_str)
            if q is not None and not q.empty:
                books[oc] = tpm.pm_book_arrays(q)
        rows = []
        for f in fills:
            book = books.get(f["outcome"])
            row = _asof_single(book, f["ts_ms"]) if book is not None else None
            if row is None:
                rows.append({"fill_ts": _iso(f["ts_ms"]), "side": f["side"], "outcome": f["outcome"],
                             "price": round(f["price"], 4), "size": round(f["size"], 2),
                             "book": "unknown", "class": "unknown"})
                continue
            bid_px, _bs, ask_px, _as = row
            rows.append({
                "fill_ts": _iso(f["ts_ms"]), "side": f["side"], "outcome": f["outcome"],
                "price": round(f["price"], 4), "size": round(f["size"], 2),
                "book_bid": round(bid_px, 4), "book_ask": round(ask_px, 4),
                "class": price_class(f["side"], f["price"], bid_px, ask_px),
            })
        traces.append({"series": series, "open_ms": open_ms, "slug": slug,
                       "open_utc": _iso(open_ms), "fills": rows})
    return traces


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


# ===========================================================================
# build one manual
# ===========================================================================
def build_manual(paths, entry: dict, *, merge_focus: bool | None = None,
                 hand_trace_n: int = 10, hand_trace_seed: int = 4242,
                 telonex_only: bool = False, max_book_windows: int | None = None) -> dict:
    """Assemble the full operating-manual metrics dict for one account. ``merge_focus``
    forces (or suppresses) the nagi-style merge-velocity block; ``None`` auto-detects
    (an account with MERGE > REDEEM events). ``telonex_only`` restricts loading to the
    Telonex window (a memory bound for ultra-active accounts)."""
    addr = entry["address"]
    act = load_our_activity(addr, telonex_only=telonex_only)
    fills, events = act["fills"], act["events"]
    by_win = group_by_window(fills)
    all_ts = [f["ts_ms"] for f in fills] + [e["ts_ms"] for e in events]
    active_days = len({hc.utc_day(t) for t in all_ts})
    n_merge = sum(1 for e in events if e["type"] == "MERGE")
    n_redeem = sum(1 for e in events if e["type"] == "REDEEM")
    if merge_focus is None:
        merge_focus = n_merge > n_redeem and n_merge > 0

    manual: dict = {
        "handle": entry["handle"], "address": addr,
        "profile": entry.get("profile", {}),
        "coverage": {
            "our_fills_in_telonex_window": len(fills),  # OUR-series fills (name kept for compat)
            "our_windows_traded": len(by_win),
            "active_days": active_days,
            "fills_from": _iso(min(all_ts))[:10] if all_ts else None,
            "fills_to": _iso(max(all_ts))[:10] if all_ts else None,
            "telonex_from": TELONEX_LO.isoformat(), "telonex_to": TELONEX_HI.isoformat(),
            "telonex_only": telonex_only,
            "merge_events": n_merge, "redeem_events": n_redeem,
        },
        "entry_timing": entry_timing(by_win),
        "price_vs_book": price_vs_book(paths, by_win, max_windows=max_book_windows),
        "sequencing": sequencing(by_win),
        "inventory_trajectory": inventory_trajectory(by_win),
        "size_ramping": size_ramping(by_win),
        "per_window_capital": per_window_capital(by_win),
        "merge_focus": bool(merge_focus),
        "anchor": anchor_reconcile(addr, fills, events),
        "hand_trace": hand_trace_windows(paths, by_win, n=hand_trace_n, seed=hand_trace_seed),
        "not_observable": [
            "their signal / trigger (why they entered this second) — only fills are recorded",
            "true queue position at a maker fill",
            "off-grid multi-level sweep depth (a VWAP fill maps to no single print)",
            "order cancels / replaces (the Data API records only fills)",
            "capital deployed outside our 4 series",
        ],
    }
    win_open = {fills[0]["cond"]: (open_ms, fills[0]["dur"])
                for (series, open_ms), fills in by_win.items()}
    manual["merge_timing"] = merge_timing(events, win_open)
    # Hold-baseline capital velocity for EVERY account (turns/day with no early frees) so a
    # merge-heavy account's recycle velocity is directly comparable to a hold-to-resolution
    # one (nagi777 ≫ takerner — the §3 comparison).
    manual["capital_velocity_hold"] = hold_velocity(by_win)
    if merge_focus:
        manual["merge_velocity"] = merge_velocity(by_win, events, active_days=active_days)
    return manual


# ===========================================================================
# CLI
# ===========================================================================
def _load_handles(base: Path) -> list[dict]:
    return json.loads((base / "handles.json").read_text(encoding="utf-8"))["handles"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deep per-window operating manuals for competitor accounts.")
    ap.add_argument("--only", nargs="*", default=None, help="handles/addresses (default: the 5 roster accounts)")
    ap.add_argument("--dir", default=None, help="competitors cache dir (default data/competitors)")
    ap.add_argument("--hand-trace-n", type=int, default=10)
    ap.add_argument("--telonex-only", action="store_true",
                    help="restrict loading to the Telonex window (memory bound for ultra-active accounts)")
    ap.add_argument("--max-book-windows", type=int, default=None,
                    help="cap the # windows whose Telonex book is loaded for price-vs-book "
                         "(the per-window .zst reads dominate runtime for ultra-active accounts)")
    args = ap.parse_args(argv)

    paths = resolve_paths()
    base = Path(args.dir) if args.dir else competitors_dir()
    handles = _load_handles(base)
    resolved = [h for h in handles if h.get("address")]
    # b27b's handle in handles.json is its full address (it was supplied as an address).
    roster = {"0xb27bc932bf8110d8f78e55da7d5f0497a18b5b82", "takerner", "bonereaper",
              "wolf9478", "nagi777", "gabigol",
              "0xf3531b23b504cf0aed4ff21325232b2a2d496685"}
    want = {x.lower() for x in args.only} if args.only else roster
    resolved = [h for h in resolved
                if h["handle"].lower() in want or h["address"].lower() in want]

    odir = out_dir() / "manuals"
    odir.mkdir(parents=True, exist_ok=True)
    for h in resolved:
        print(f"[manuals] {h['handle']} ({h['address']})")
        manual = build_manual(paths, h, hand_trace_n=args.hand_trace_n, telonex_only=args.telonex_only,
                              max_book_windows=args.max_book_windows)
        (odir / f"{h['handle']}.json").write_text(
            json.dumps(manual, indent=2, default=str), encoding="utf-8")
        c = manual["coverage"]
        a = manual["anchor"]
        print(f"[manuals]   {c['our_fills_in_telonex_window']:,} our-fills / {c['our_windows_traded']:,} windows; "
              f"merge_focus={manual['merge_focus']}; anchor suspect={a.get('suspect')}")
    print(f"[manuals] wrote {odir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
