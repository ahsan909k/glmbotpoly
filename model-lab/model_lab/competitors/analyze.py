"""Classify each account by measurable behavioural signatures.

Reads the cached ``data/competitors/<addr>/`` history and computes, per account:

- **series focus + coverage** — share of trades (count & $) in our four series
  (btc/eth up-down 5m & 15m) vs everything else.
- **maker/taker mix** — activity TRADEs differenced against ``takerOnly`` trades
  (the API exposes no maker/taker flag; ``takerOnly`` is the ground truth, and the
  Telonex tape join corroborates it — see analyze_tape.py).
- **pair discipline** — per window: two-sided vs one-sided, matched Up/Down pairs,
  combined pair-cost distribution, leg-completion lag.
- **merge vs hold** — MERGE (recycle) vs REDEEM (hold-to-resolution) event mix.
- **timing** — where in the window fills land (at-open / mid / final-seconds),
  split maker vs taker — a tape-free reaction-speed proxy (Data API timestamps are
  1-second granular; sub-second Binance-lag needs the µs tape join, noted).
- **size / cadence / hours** — $ per window, trades/day, active-hours profile.
- **official PnL** — the authoritative profile P/L (from ``user-pnl-api``, cached by
  fetch.py): all-time / 1M / 1W / 1D. The cash-flow reconstruction below is demoted to
  a cross-check; a "trades-only (excludes redemptions)" figure and a profit-source
  decomposition (matched-pair economics, directional settlement, maker rewards, and an
  independent taker-rebate estimate) explain where each account's official profit comes
  from — with residuals flagged, not force-fit.
- **reconstructed PnL** — cash-flow equity curve (buys/sells/redeems/merges/rewards
  minus estimated taker fees) → PnL/day and worst drawdown, with uncertainty notes.

Writes ``out/competitors/analysis.json``. Run: ``python -m model_lab.competitors.analyze``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..lib.math import TAKER_TIERS
from .paths import account_dir, competitors_dir, out_dir

FEE_RATE = 0.07  # crypto taker feeRate (CLAUDE.md §7); makers pay 0.


def taker_fee(shares: float, price: float, rate: float = FEE_RATE) -> float:
    """Polymarket taker fee = shares·rate·p·(1−p), 5-dp half-away, $0.00001 floor
    (matches core-types money.rs / lib.math.taker_fee)."""
    raw = shares * rate * price * (1.0 - price)
    if raw <= 0.0:
        return 0.0
    return max(math.floor(raw * 1e5 + 0.5) / 1e5, 1e-5)


def _parse_iso(s) -> int | None:
    """ISO-8601 (with optional Z / microseconds) -> unix seconds, or None."""
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


def _match_key(r: dict) -> tuple:
    """Identify a single fill across the /activity and /trades endpoints (which carry
    different field sets) — transaction hash + token + side + size."""
    return (str(r.get("transactionHash") or ""), str(r.get("asset") or ""),
            str(r.get("side") or ""), round(float(r.get("size") or 0.0), 6))


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


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p25": None, "p75": None}
    return {
        "n": len(values),
        "median": round(_pct(values, 0.5), 4),
        "mean": round(sum(values) / len(values), 4),
        "p25": round(_pct(values, 0.25), 4),
        "p75": round(_pct(values, 0.75), 4),
    }


# Our four shared series; everything else (SOL/XRP/1h/4h/…) is now broken out too.
OUR = ("BTC-5m", "ETH-5m", "BTC-15m", "ETH-15m")
OUR_SET = frozenset(OUR)

# tier name -> rebate fraction (from lib.math.TAKER_TIERS, the single source of truth).
_TIER_PCT = {name: pct for _th, name, pct in TAKER_TIERS}
_REBATE_BOTTOM_UP_MAX = 400_000  # cap the per-day bottom-up model to keep mega-accounts tractable

_LOCAL_SLUG_RE = re.compile(r"^([a-z0-9]+)-updown-(\d+)([mh])-(\d+)$")
_DUR_UNIT = {"m": 60, "h": 3600}


def classify_slug(slug: str):
    """'{asset}-updown-{n}{m|h}-{epoch}' -> (series_key, asset, dur_secs, epoch) | None.

    A competitor-local generalization of historical_common.parse_slug to ANY asset
    (btc/eth/sol/xrp/…) and duration (5m/15m/1h/4h). Returns the SAME series_key
    ('BTC-5m', …) as parse_slug for the four shared series, so the OUR-series
    aggregation is unchanged; non-shared series now classify instead of collapsing
    into one 'other' bucket."""
    m = _LOCAL_SLUG_RE.match((slug or "").strip().lower())
    if not m:
        return None
    asset, n, unit, epoch = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    return f"{asset.upper()}-{n}{unit}", asset, n * _DUR_UNIT[unit], epoch


def _p_at(series: list[dict], target_t: int) -> float:
    """Cumulative official PnL as of ``target_t``: the last point with ``t <= target_t``
    (step function). 0.0 when ``target_t`` precedes the first point. ``series`` ascending."""
    val = 0.0
    for pt in series:
        if pt.get("t", 0) <= target_t:
            val = pt.get("p", 0.0)
        else:
            break
    return val


def _official_pnl(addr: str) -> dict | None:
    """Read the cached official P/L series and derive ALL / 1M / 1W / 1D PnL.

    ``p`` is cumulative all-time PnL, so ALL = last p and the interval figures are
    deltas of the cumulative curve (last p minus p at the target lookback). Returns
    None when the file is absent/empty (caller flags ``official_pnl_available``)."""
    p = account_dir(addr) / "official_pnl.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    s = sorted((pt for pt in doc.get("series", []) if isinstance(pt, dict) and "t" in pt),
               key=lambda x: x["t"])
    if not s:
        return None
    t_last, p_last = int(s[-1]["t"]), float(s[-1]["p"])
    return {
        "available": True,
        "as_of": datetime.fromtimestamp(t_last, tz=timezone.utc).isoformat(),
        "all_usd": round(p_last, 2),
        "1w_usd": round(p_last - _p_at(s, t_last - 7 * 86400), 2),
        "1m_usd": round(p_last - _p_at(s, t_last - 30 * 86400), 2),
        "1d_usd": round(p_last - _p_at(s, t_last - 1 * 86400), 2),
        "first_t": int(s[0]["t"]),
        "last_t": t_last,
        "points": len(s),
        "series": s,
    }


def _estimate_rebate(taker_fills: list[tuple], est_taker_fees: float, profile: dict) -> dict:
    """Estimate taker-fee rebate income (tier% × taker fees paid).

    Two estimates: (a) **tier-anchored** — the account's REAL Gamma tier
    (``profile.takerTierName``, authoritative, spans all markets) × captured taker
    fees; (b) **bottom-up** — the existing daily rebate model over the captured taker
    fills (reproduces the $1 floor + daily tier drift, but only sees fetched fills).
    The rebate is credited as separate daily pUSD — whether the official P/L number
    already includes it is UNVERIFIED, so callers present it ALONGSIDE, not summed."""
    tier = profile.get("takerTierName")
    pct = _TIER_PCT.get(tier, 0.0)
    tier_anchored = round(pct * est_taker_fees, 2)
    bottom_up = 0.0
    bu_tier = None
    # The bottom-up model has a per-day Python loop; skip it for pathologically large
    # fill sets (mega-accounts) — the tier-anchored estimate is the headline anyway.
    if taker_fills and len(taker_fills) <= _REBATE_BOTTOM_UP_MAX:
        import pandas as pd

        from ..rebate_sim import compute_rebate_timeline
        df = pd.DataFrame(taker_fills, columns=["shares", "price", "fee", "ts"])
        df["net"] = -df["fee"]
        df["day"] = df["ts"] // 86400
        res = compute_rebate_timeline(df)
        bottom_up = round(res["totals"]["total_rebate"], 2)
        bu_tier = res["current"]["tier"]
    elif taker_fills:
        bu_tier = f"skipped ({len(taker_fills):,} fills)"
    return {
        "tier_name": tier,
        "rebate_pct": pct,
        "rebate_income_tier_anchored_usd": tier_anchored,
        "rebate_income_bottom_up_usd": bottom_up,
        "bottom_up_end_tier": bu_tier,
        "note": ("rebate = tier% × taker fees, credited as separate daily pUSD; shown "
                 "ALONGSIDE official PnL (not summed) — inclusion in official is unverified"),
    }


def _uptime(ts_all: list[int]) -> dict:
    """Coverage/uptime from the event-timestamp stream (all events with a ts).

    - ``continuous_coverage_frac``: distinct active UTC-hour buckets / hour-buckets in
      ``[first, last]`` (a 24/7 bot ≈ 1.0).
    - ``active_hours_per_day_median``: median over active UTC dates of #distinct hours
      that saw activity.
    - ``max_gap_hours``: longest silence between consecutive events (downtime detector)."""
    ts = sorted(t for t in ts_all if t)
    if len(ts) < 2:
        return {"continuous_coverage_frac": None, "active_hours_per_day_median": None,
                "max_gap_hours": None, "span_hours": None}
    hour_buckets = {t // 3600 for t in ts}
    span_buckets = (ts[-1] // 3600) - (ts[0] // 3600) + 1
    per_day: dict[str, set] = defaultdict(set)
    for t in ts:
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        per_day[dt.date().isoformat()].add(dt.hour)
    hours_per_day = sorted(len(hrs) for hrs in per_day.values())
    max_gap = max(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
    return {
        "continuous_coverage_frac": round(len(hour_buckets) / span_buckets, 4) if span_buckets else None,
        "active_hours_per_day_median": hours_per_day[len(hours_per_day) // 2],
        "max_gap_hours": round(max_gap / 3600.0, 2),
        "span_hours": round((ts[-1] - ts[0]) / 3600.0, 1),
    }


class WindowAgg:
    """Per-window (conditionId) fill accumulator for our-series pair analysis."""

    __slots__ = ("series", "open_ms", "dur", "up_buy_sh", "up_buy_usd",
                 "dn_buy_sh", "dn_buy_usd", "up_sell_sh", "dn_sell_sh",
                 "first_up", "first_dn", "n_fills")

    def __init__(self, series, open_ms, dur):
        self.series, self.open_ms, self.dur = series, open_ms, dur
        self.up_buy_sh = self.up_buy_usd = 0.0
        self.dn_buy_sh = self.dn_buy_usd = 0.0
        self.up_sell_sh = self.dn_sell_sh = 0.0
        self.first_up = self.first_dn = None
        self.n_fills = 0

    def add(self, outcome, side, size, usd, ts):
        self.n_fills += 1
        up = outcome == "Up"
        if side == "BUY":
            if up:
                self.up_buy_sh += size; self.up_buy_usd += usd
                self.first_up = ts if self.first_up is None else min(self.first_up, ts)
            else:
                self.dn_buy_sh += size; self.dn_buy_usd += usd
                self.first_dn = ts if self.first_dn is None else min(self.first_dn, ts)
        else:  # SELL
            if up:
                self.up_sell_sh += size
            else:
                self.dn_sell_sh += size


def analyze_account(entry: dict) -> dict:
    addr = entry["address"]
    d = account_dir(addr)
    act_path = d / "activity.jsonl"
    if not act_path.exists():
        return {"handle": entry["handle"], "address": addr, "error": "no activity cache"}

    # Taker fill keys (ground-truth aggressor set) + the window they cover. The
    # maker/taker split is only trustworthy from the oldest taker fill onward: a TRADE
    # before that would be mislabelled maker simply because its taker page wasn't pulled
    # (matters only if a very large account's taker history is incompletely fetched).
    taker_keys: set = set()
    taker_ts_min: int | None = None
    for r in _iter_jsonl(d / "taker_trades.jsonl"):
        taker_keys.add(_match_key(r))
        t = int(r.get("timestamp") or 0)
        if t:
            taker_ts_min = t if taker_ts_min is None else min(taker_ts_min, t)
    split_from = taker_ts_min or 0

    # Aggregators.
    series_ct = defaultdict(int)          # series -> trade count
    series_usd = defaultdict(float)       # series -> trade $ volume
    taker_ct = defaultdict(int)
    taker_usd = defaultdict(float)
    maker_ct = defaultdict(int)
    maker_usd = defaultdict(float)
    hours = [0] * 24                      # UTC hour histogram (our-series trades)
    dates = set()                         # active UTC dates (all trades)
    windows: dict[str, WindowAgg] = {}    # our-series per-window aggregation
    # cash-flow ledger (all markets): list of (ts, delta)
    flows: list[tuple[int, float]] = []
    ev_counts = defaultdict(int)
    ev_usd = defaultdict(float)           # per event type $ (merge/redeem/reward/split)
    merge_windows: set[str] = set()
    redeem_windows: set[str] = set()
    n_trade = 0
    n_our_trade = 0
    est_taker_fees = 0.0
    # within-window timing (our series): fraction of window elapsed at fill
    timing_taker: list[float] = []
    timing_maker: list[float] = []
    # per-window $ deployed (our series)
    win_notional: dict[str, float] = defaultdict(float)
    # trades-only cash flow (all markets): Σ(sells − buys − taker_fees); EXCLUDES
    # redeems/merges/rewards/splits — the "trades-only (excludes redemptions)" cross-check.
    trades_only = 0.0
    taker_fills: list[tuple] = []          # (size, price, fee, ts) for the rebate estimate

    for r in _iter_jsonl(act_path):
        typ = r.get("type")
        ev_counts[typ] += 1
        ts = int(r.get("timestamp") or 0)
        cond = str(r.get("conditionId") or "")
        usd = float(r.get("usdcSize") or 0.0)
        size = float(r.get("size") or 0.0)
        price = float(r.get("price") or 0.0)
        slug = str(r.get("slug") or "")
        parsed = classify_slug(slug)  # (series, asset, dur, epoch) for ANY updown series, or None
        is_our = parsed is not None and parsed[0] in OUR_SET
        if ts:
            dates.add(datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat())

        if typ == "TRADE":
            n_trade += 1
            side = str(r.get("side") or "")
            outcome = str(r.get("outcome") or "")
            is_taker = _match_key(r) in taker_keys
            skey = parsed[0] if parsed else "other"
            series_ct[skey] += 1
            series_usd[skey] += usd
            # only fold into the maker/taker split where taker labelling is reliable
            if ts >= split_from:
                if is_taker:
                    taker_ct[skey] += 1; taker_usd[skey] += usd
                else:
                    maker_ct[skey] += 1; maker_usd[skey] += usd
            if is_taker:
                est_taker_fees += taker_fee(size, price)
            # cash flow
            fee = taker_fee(size, price) if is_taker else 0.0
            flows.append((ts, (usd - fee) if side == "SELL" else -(usd + fee)))
            # trades-only (excludes redemptions) + taker fills for the rebate model
            trades_only += (usd - fee) if side == "SELL" else -(usd + fee)
            if is_taker:
                taker_fills.append((size, price, fee, ts))
            if is_our:
                n_our_trade += 1
                series, asset, dur, epoch = parsed
                open_ms = epoch * 1000
                w = windows.get(cond)
                if w is None:
                    w = windows[cond] = WindowAgg(series, open_ms, dur)
                w.add(outcome, side, size, usd, ts)
                win_notional[cond] += usd if side == "BUY" else 0.0
                hours[datetime.fromtimestamp(ts, tz=timezone.utc).hour] += 1
                # within-window timing fraction
                if dur > 0:
                    frac = (ts - epoch) / dur
                    if 0.0 <= frac <= 1.2:
                        (timing_taker if is_taker else timing_maker).append(min(frac, 1.0))
        else:
            # non-trade cash events
            ev_usd[typ] += usd
            if typ == "REDEEM":
                flows.append((ts, +usd)); redeem_windows.add(cond)
            elif typ == "MERGE":
                flows.append((ts, +usd)); merge_windows.add(cond)
            elif typ == "REWARD":
                flows.append((ts, +usd))
            elif typ == "SPLIT":
                flows.append((ts, -usd))
            elif typ == "CONVERSION":
                flows.append((ts, +usd))

    return _summarize(entry, addr, locals())


def _summarize(entry, addr, L) -> dict:
    series_ct, series_usd = L["series_ct"], L["series_usd"]
    taker_ct, taker_usd, maker_ct, maker_usd = L["taker_ct"], L["taker_usd"], L["maker_ct"], L["maker_usd"]
    windows: dict = L["windows"]
    flows = L["flows"]

    total_ct = sum(series_ct.values())
    total_usd = sum(series_usd.values())
    our_ct = sum(series_ct[s] for s in OUR)
    our_usd = sum(series_usd[s] for s in OUR)

    # maker/taker over our series
    our_taker_ct = sum(taker_ct[s] for s in OUR)
    our_maker_ct = sum(maker_ct[s] for s in OUR)
    our_taker_usd = sum(taker_usd[s] for s in OUR)
    our_maker_usd = sum(maker_usd[s] for s in OUR)
    tot_taker_ct = sum(taker_ct.values())
    tot_maker_ct = sum(maker_ct.values())

    def frac(a, b):
        return round(a / b, 4) if b else None

    # pair discipline over our-series windows
    two_sided = one_sided = 0
    pair_costs = []
    leg_gaps = []
    matched_pairs_total = 0.0
    excess_share_total = 0.0
    # Guaranteed economics of the matched portion: matched shares pay $1 at
    # resolution and cost `pair_cost`, so PnL = matched·(1 − pair_cost). POSITIVE
    # when pair_cost < 1, NEGATIVE when > 1 (the crux for takerner/frog). `windows`
    # only ever holds OUR-series windows (created under `if is_our`).
    matched_pair_pnl = 0.0
    for w in windows.values():
        has_up = w.up_buy_sh > 1e-9
        has_dn = w.dn_buy_sh > 1e-9
        if has_up and has_dn:
            two_sided += 1
            vwap_up = w.up_buy_usd / w.up_buy_sh
            vwap_dn = w.dn_buy_usd / w.dn_buy_sh
            pair_costs.append(vwap_up + vwap_dn)
            matched = min(w.up_buy_sh, w.dn_buy_sh)
            matched_pairs_total += matched
            matched_pair_pnl += matched * (1.0 - (vwap_up + vwap_dn))
            excess_share_total += abs(w.up_buy_sh - w.dn_buy_sh)
            if w.first_up is not None and w.first_dn is not None:
                leg_gaps.append(abs(w.first_dn - w.first_up))
        elif has_up or has_dn:
            one_sided += 1

    # PnL cash-flow curve + daily + drawdown
    flows.sort(key=lambda x: x[0])
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    daily = defaultdict(float)
    for ts, delta in flows:
        cum += delta
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)  # most negative
        if ts:
            daily[datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()] += delta
    realized_pnl = round(cum, 2)
    daily_vals = list(daily.values())
    n_days = len(L["dates"]) or 1

    # current open positions (unrealized)
    positions = json.loads((account_dir(addr) / "positions.json").read_text(encoding="utf-8")) \
        if (account_dir(addr) / "positions.json").exists() else []
    open_value = round(sum(float(p.get("currentValue") or 0.0) for p in positions), 2)
    open_cash_pnl = round(sum(float(p.get("cashPnl") or 0.0) for p in positions), 2)

    win_notional = L["win_notional"]
    hours = L["hours"]
    ev_counts, ev_usd = L["ev_counts"], L["ev_usd"]

    # series focus breakdown (top non-our categories folded to families)
    other_families = defaultdict(float)
    for s, v in series_usd.items():
        if s not in OUR:
            other_families[s] += v

    ts_all = [ts for ts, _ in flows if ts]
    first_ts = min(ts_all) if ts_all else None
    last_ts = min(max(ts_all), 4102444800) if ts_all else None
    # The /activity pull can cap before reaching an ultra-high-frequency account's full
    # history (its time cursor advances slowly, so it hits the page cap). Detect it two ways:
    # (a) taker fills predate the oldest activity record; (b) the oldest activity record is
    # well after the account was created. Either ⇒ analysed over a recent window only.
    created_ts = _parse_iso((L["entry"].get("profile") or {}).get("createdAt"))
    activity_truncated = bool(
        (L["split_from"] and first_ts and L["split_from"] < first_ts - 86400)
        or (created_ts and first_ts and first_ts > created_ts + 14 * 86400)
    )

    # --- official PnL (authoritative) + trades-only cross-check + rebate + uptime ---
    profile = entry.get("profile", {})
    off = _official_pnl(addr)
    reward_income = round(ev_usd.get("REWARD", 0.0), 2)
    trades_only_usd = round(L["trades_only"], 2)
    rebate = _estimate_rebate(L["taker_fills"], L["est_taker_fees"], profile)
    uptime = _uptime(ts_all)
    usd_per_window_stats = _stats([v for v in win_notional.values() if v > 0])

    official_all = off["all_usd"] if off else None
    # gap ≈ redemptions/settlement value (official is redemption-inclusive; trades-only is not)
    settlement_gap = round(official_all - trades_only_usd, 2) if official_all is not None else None
    redeem_w, merge_w = len(L["redeem_windows"]), len(L["merge_windows"])
    hold_to_resolution_frac = frac(redeem_w, redeem_w + merge_w)

    # Decompose over the CAPTURED window: official is lifetime, but the reconstruction/
    # fees/rebate are window-scoped for truncated accounts. official_window = the
    # official PnL accrued during [first captured trade, now].
    official_window = residual = None
    if off:
        official_window = round(off["all_usd"] - _p_at(off["series"], first_ts or off["first_t"]), 2)
        residual = round(official_window - realized_pnl - open_cash_pnl, 2)
    rebate_income = rebate["rebate_income_tier_anchored_usd"]
    directional_settlement = round(realized_pnl - matched_pair_pnl - reward_income, 2)
    rebate_load_bearing = bool(
        rebate["rebate_pct"] > 0 and official_window is not None and abs(official_window) > 1
        and rebate_income >= 0.25 * abs(official_window))
    official_ambiguous = (off is None) or (
        residual is not None and official_window is not None
        and abs(residual) > 0.5 * max(abs(official_window), 1.0))

    # full per-series breakdown (SOL / XRP / 1h / 4h no longer collapse to "other")
    series_breakdown = {}
    for s in sorted(series_usd, key=lambda k: -series_usd[k]):
        tc, mc = taker_ct.get(s, 0), maker_ct.get(s, 0)
        series_breakdown[s] = {"count": series_ct.get(s, 0), "usd": round(series_usd[s], 2),
                               "taker_frac": frac(tc, tc + mc)}
    usd_win_med = usd_per_window_stats["median"]
    usd_per_window_vs_our_cap = (round(usd_win_med / 25.0, 1)
                                 if isinstance(usd_win_med, (int, float)) else None)

    return {
        "handle": entry["handle"],
        "address": addr,
        "profile": entry.get("profile", {}),
        "span": {
            "first": datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat() if first_ts else None,
            "last": datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat() if last_ts else None,
            "active_days": len(L["dates"]),
            "activity_truncated": activity_truncated,
            "note": ("RECENT WINDOW ONLY — /activity paging capped for this ultra-active account; "
                     "older history not captured. Rates (PnL/day, /window) are representative; totals are"
                     " for the window shown." if activity_truncated else "full history since first trade"),
        },
        "totals": {
            "activity_events": sum(ev_counts.values()),
            "trades": L["n_trade"],
            "our_series_trades": L["n_our_trade"],
            "event_type_counts": dict(ev_counts),
        },
        "coverage": {
            "our_series_trade_frac_count": frac(our_ct, total_ct),
            "our_series_trade_frac_volume": frac(our_usd, total_usd),
            "our_series_volume_usd": round(our_usd, 2),
            "total_trade_volume_usd": round(total_usd, 2),
            "by_series_volume": {s: round(series_usd[s], 2) for s in OUR if series_usd[s]},
            "full_series_volume": {s: b["usd"] for s, b in series_breakdown.items() if b["usd"]},
            "series_breakdown": series_breakdown,
            "non_our_series": [s for s in series_breakdown if s not in OUR_SET and s != "other"],
            "top_other_categories": dict(sorted(
                ((k, round(v, 2)) for k, v in other_families.items()),
                key=lambda kv: -kv[1])[:6]),
        },
        "maker_taker": {
            "overall_taker_frac_count": frac(tot_taker_ct, tot_taker_ct + tot_maker_ct),
            "our_taker_frac_count": frac(our_taker_ct, our_taker_ct + our_maker_ct),
            "our_taker_frac_volume": frac(our_taker_usd, our_taker_usd + our_maker_usd),
            "our_taker_trades": our_taker_ct,
            "our_maker_trades": our_maker_ct,
            "split_labelled_from": (datetime.fromtimestamp(L["split_from"], tz=timezone.utc).isoformat()
                                    if L["split_from"] else None),
            "split_covers_full_history": bool(L["split_from"] <= (first_ts or L["split_from"])),
            "note": "taker = fill present in /trades?takerOnly=true; maker = the rest. Split computed "
                    "only from split_labelled_from onward (the oldest fetched taker fill).",
        },
        "pair_discipline": {
            "our_windows_traded": len(windows),
            "two_sided_windows": two_sided,
            "one_sided_windows": one_sided,
            "two_sided_frac": frac(two_sided, two_sided + one_sided),
            "pair_cost": _stats(pair_costs),
            "leg_completion_gap_secs": _stats([float(x) for x in leg_gaps]),
            "matched_pairs_shares": round(matched_pairs_total, 1),
            "excess_shares": round(excess_share_total, 1),
        },
        "merge_hold": {
            "merge_events": ev_counts.get("MERGE", 0),
            "redeem_events": ev_counts.get("REDEEM", 0),
            "reward_events": ev_counts.get("REWARD", 0),
            "split_events": ev_counts.get("SPLIT", 0),
            "merge_windows": len(L["merge_windows"]),
            "redeem_windows": len(L["redeem_windows"]),
            "merge_usd": round(ev_usd.get("MERGE", 0.0), 2),
            "redeem_usd": round(ev_usd.get("REDEEM", 0.0), 2),
            "reward_usd": round(ev_usd.get("REWARD", 0.0), 2),
            "behavior": ("merges (recycles capital)" if ev_counts.get("MERGE", 0) > ev_counts.get("REDEEM", 0)
                         else "holds to resolution + redeems" if ev_counts.get("REDEEM", 0) > 0
                         else "n/a"),
        },
        "timing_within_window": {
            "taker_fill_frac": _stats(L["timing_taker"]),
            "maker_fill_frac": _stats(L["timing_maker"]),
            "note": "fraction of window elapsed at fill (0=open,1=close); 1s-granular API timestamps",
        },
        "size_cadence": {
            "usd_per_window": usd_per_window_stats,
            "usd_per_window_vs_our_cap_x": usd_per_window_vs_our_cap,
            "our_trades_per_active_day": round(L["n_our_trade"] / n_days, 1),
            "total_trades_per_active_day": round(L["n_trade"] / n_days, 1),
            "active_hours_utc": hours,
            "peak_hours_utc": [i for i, _ in sorted(enumerate(hours), key=lambda x: -x[1])[:4]],
            "uptime": uptime,
        },
        "pnl": {
            # OFFICIAL (authoritative) — Polymarket's own profile P/L
            "official_pnl_available": bool(off),
            "official_all_usd": official_all,
            "official_1m_usd": off["1m_usd"] if off else None,
            "official_1w_usd": off["1w_usd"] if off else None,
            "official_1d_usd": off["1d_usd"] if off else None,
            "official_as_of": off["as_of"] if off else None,
            "official_series_points": off["points"] if off else 0,
            "official_ambiguous": official_ambiguous,
            # trades-only cross-check (EXCLUDES redemptions) — NOT the PnL
            "trades_only_usd": trades_only_usd,
            "settlement_gap_usd": settlement_gap,   # ≈ redemptions/settlement value
            "hold_to_resolution_frac": hold_to_resolution_frac,
            # cash-flow reconstruction (redemption-inclusive) — demoted to cross-check
            "realized_cashflow_usd": realized_pnl,
            "est_taker_fees_paid_usd": round(L["est_taker_fees"], 2),
            "reward_income_usd": reward_income,
            "open_position_value_usd": open_value,
            "open_unrealized_pnl_usd": open_cash_pnl,
            "total_equity_estimate_usd": round(realized_pnl + open_value, 2),
            "pnl_per_active_day_usd": round(realized_pnl / n_days, 2),
            "pnl_per_day": _stats(daily_vals),
            "worst_drawdown_usd": round(max_dd, 2),
            # taker-fee rebate — an INDEPENDENT estimate, shown alongside (not summed into) official
            "est_taker_rebate_usd": rebate_income,
            "rebate": rebate,
            "rebate_load_bearing": rebate_load_bearing,
            # profit-source decomposition over the captured window
            "decomposition": {
                "window_from": (datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat()
                                if first_ts else None),
                "window_is_recent": activity_truncated,
                "official_window_usd": official_window,
                "matched_pair_pnl_usd": round(matched_pair_pnl, 2),      # guaranteed; NEG if pair-cost>1
                "directional_settlement_usd": directional_settlement,    # up-drift / one-sided winners / sells
                "reward_income_usd": reward_income,                      # maker/liquidity rewards
                "est_taker_rebate_usd": rebate_income,                   # SEPARATE + ambiguous
                "open_mtm_usd": open_cash_pnl,
                "residual_usd": residual,                               # unexplained (see note)
                "note": ("official_window ≈ matched_pair_pnl + directional_settlement + reward + "
                         "open_mtm + residual; rebate is shown ALONGSIDE (not added). residual = "
                         "unexplained: rebate-in-official / other markets / truncation / redemption timing."),
            },
            "uncertainty": "OFFICIAL PnL (user-pnl-api.polymarket.com) is authoritative; "
                           "realized_cashflow / trades_only are cross-checks; est_taker_fees "
                           "ESTIMATED from the taker-fill set; the rebate is an INDEPENDENT "
                           "estimate (tier% × fees) — whether official already includes it is "
                           "UNVERIFIED, so it is flagged, not summed.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify accounts by measurable signatures.")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="re-analyze even if already in analysis.json")
    args = ap.parse_args(argv)

    base = Path(args.dir) if args.dir else competitors_dir()
    handles = json.loads((base / "handles.json").read_text(encoding="utf-8"))["handles"]
    resolved = [h for h in handles if h.get("address")]
    if args.only:
        want = {x.lower() for x in args.only}
        resolved = [h for h in resolved if h["address"].lower() in want or h["handle"].lower() in want]

    odir = out_dir()
    odir.mkdir(parents=True, exist_ok=True)
    out = odir / "analysis.json"
    # Load existing results so a killed run resumes (each big account streams a huge file).
    existing: dict = {}
    if out.exists():
        try:
            existing = {a["address"]: a for a in json.loads(out.read_text(encoding="utf-8"))["accounts"]
                        if a.get("address")}
        except (OSError, json.JSONDecodeError, KeyError):
            existing = {}

    for h in resolved:
        if not args.force and h["address"] in existing and "error" not in existing[h["address"]]:
            print(f"[analyze] {h['handle']} — cached, skip")
            continue
        print(f"[analyze] {h['handle']} ({h['address']})")
        existing[h["address"]] = analyze_account(h)
        # write incrementally: a kill preserves finished accounts
        out.write_text(json.dumps({"accounts": list(existing.values())}, indent=2, default=str),
                       encoding="utf-8")

    print(f"[analyze] wrote {out} — {len(existing)} accounts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
