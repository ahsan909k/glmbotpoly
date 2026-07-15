"""Stage — backtest_conf_pair: the operator's **confidence-ladder hedged-pair** strategy.

Per window, forward over the walk-forward **dir10** OOS predictions
(``out/learn_walkforward/oos_dir10_full.parquet`` — the model deployed live) in
``sample_ts_ms`` order, maintaining per-window inventory as *lots*:

- **Confidence ladder (shares).** ``confidence = max(p_up, 1−p_up)``; side = Up if
  ``p_up ≥ 0.5`` else Down. Size = **40** sh if ``conf ≥ 0.80``, **20** if
  ``0.70 ≤ conf < 0.80``, **10** if ``0.60 ≤ conf < 0.70``, else **skip** (no entry).
- **Max unhedged imbalance = 40 sh.** Track the signed net UNHEDGED imbalance across
  still-open (not-yet-hedged) lots. Before an entry, if adding the *ladder size* to the
  side would push the net imbalance beyond ±40 toward that side, do not enter (skip).
- **Entry (taker).** Buy ``side`` at the displayed ask as-of ``fire_ts + effective_latency``
  (255 ms = network + venue), sized ``min(ladder, displayed_depth, side-budget-left/price)``,
  paying the exact 5-dp taker fee. Open a lot ``(side, n, price_s, entry_fee, open_ts)``.
- **Hedge via a resting maker (zero fee).** For each open lot, rest a maker on the OPPOSITE
  side at ``limit = 1.00 − price_s``. It fills for ``m = min(n, displayed_opp_size)`` shares
  at ``limit`` the first time the recorded opposite ask reaches ``≤ limit`` within
  ``[open_ts, min(close, open_ts+40s))`` → a LOCKED pair (pair cost ``price_s + limit = 1.00``;
  collects $1 at resolution).
- **Forced hedge (taker) at 40 s.** If a lot still has unhedged shares at
  ``min(open_ts+40s, close)``, complete them at MARKET on the opposite side (paying the taker
  fee) → a FORCED-HEDGE pair (pair cost usually > 1 ⇒ a loss). If it can't execute (no fresh
  book, or the completing side's budget is spent), those shares are left NAKED.
- **Naked legs** ride to resolution ($1 if the lot's side wins, else $0).

**Per-side budget = $100.** Cumulative notional bought on the Up side must stay ≤ $100 AND on
the Down side ≤ $100 (each side its own independent cap; ≈ $200/window). Every buy consumes the
budget of the side it *buys*: entry takers, the resting-maker completions, and the forced-hedge
takers. A leg that would exceed its side's $100 is short-filled to the cap (or skipped).

**Three-way P&L split** (reconciles EXACTLY to total net):
``locked_pairs_pnl + forced_hedge_pnl + naked_leg_pnl == total_net``. Reported per version and
per series, over the **current-regime** windows only (``≥ 2026-06-05``, 255 ms effective, per-
market real fees, the 4 series BTC/ETH × 5m/15m). vs never-trading ($0).

**Three versions** (exactly one change each vs BASE): **base** (force at 40 s); **no_forced**
(don't force — stop leaning on the stuck side via the imbalance cap, ride the legs naked);
**cost_cap** (force at 40 s only if ``price_s + market_opp ≤ 1.02``, else naked).

**No retraining / LightGBM-free** — reuses ``money_judge``'s fill primitives + the
``backtest_pair_lean`` reconstruction. Resumable per-(series, day); run per-series in parallel
sharing one ``--out-name``. **Research only.** Output ``out/backtests/conf_pair/``.

Run::

    python -m model_lab.backtest_conf_pair                          # full current regime, 4 series
    python -m model_lab.backtest_conf_pair --series BTC-5m          # one series (parallel worker)
    python -m model_lab.backtest_conf_pair --since 2026-06-05 --until 2026-06-08 --max-days 3
"""

from __future__ import annotations

import heapq
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import historical_common as hc
from . import historical_labels as hlbl
from . import historical_resolutions as hr
from . import money_judge as mj
from .config import Paths, assert_parquet_ready, resolve_bounds, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .lib import math as lm
from .lib import procmem

MS_PER_DAY = 86_400_000
CURRENT_REGIME_FROM = date(2026, 6, 5)  # 250 ms taker-delay lock (out/audits/venue_regime_timeline.md)
_EPS = 1e-9

# ---- strategy constants (config-ish; easy to change) -----------------------
#: Single per-window budget (shared across both legs), a generous safety ceiling that rarely
#: binds — each entry is already ≤ 40 shares (≈ ≤ $40) via the confidence ladder, so the ladder
#: is the binding size limit; $1,000/window just guards a runaway (book glitch / rapid-fire
#: signals). Charged on every TAKER leg (entries + forced hedges); maker completions are zero-fee
#: and budget-free. Operator-set (2026, supersedes the $50-shared / $100-per-side proposals).
WINDOW_BUDGET = 1000.0
#: Confidence → ladder size in shares (descending; first threshold met wins; below the lowest
#: threshold the snapshot is skipped, not a fire).
SIZE_LADDER: tuple[tuple[float, float], ...] = ((0.80, 40.0), (0.70, 20.0), (0.60, 10.0))
#: Cap on the signed net unhedged imbalance (shares) across still-open lots.
MAX_IMBALANCE = 40.0
#: The forced-hedge / maker-window deadline after a lot opens (ms), clamped to the window close.
HEDGE_DEADLINE_MS = 40_000
#: Variant "cost_cap": force at 40 s only when the completed pair cost ≤ this.
FORCED_COST_CAP = 1.02
#: The three strategy versions.
VERSIONS: tuple[str, ...] = ("base", "no_forced", "cost_cap")
VERSION_LABEL = {"base": "BASE (force at 40s)",
                 "no_forced": "no_forced (stop-lean + ride naked)",
                 "cost_cap": "cost_cap (force only if pair cost ≤ 1.02)"}

CAVEATS = [
    "Single per-window budget $1,000 (a generous safety ceiling that rarely binds — each entry is "
    "≤ 40 shares ≈ ≤ $40 via the confidence ladder, so the ladder is the binding size limit). "
    "Shared across both legs; charged on every TAKER leg (entries + forced hedges). Maker "
    "completions are zero-fee and budget-free.",
    "The resting-maker completion fills for min(n, displayed opposite size) at the limit price "
    "= 1 − price_s the first time the recorded OPPOSITE ask reaches ≤ limit within "
    "[open+ε, min(close, open+40s)) — the natural but mildly optimistic maker-fill proxy (Telonex "
    "has no PM trade tape; the real §9 model queues behind displayed size and fills on prints "
    "through the price). Pair cost = price_s + (1 − price_s) = 1.00 ⇒ a locked pair's PnL is just "
    "the pro-rated entry fee (near break-even by construction).",
    "The maker completion pays ZERO fee and NO rebate is credited (conservative); the entry and "
    "forced-hedge takers pay the exact 5-dp taker fee.",
    "Forced hedge and maker completion are capped at the single displayed opposite level (top of "
    "book) — a deeper unseen book could complete more; conservative on depth.",
    "Effective taker fill latency = network + venue delay (default 5 + 250 = 255 ms, the vps255 "
    f"reference); a {mj.BOOK_STALENESS_MS} ms book-staleness skip biases toward fewer fills. A flat "
    "255 ms venue delay is only correct WITHIN the current regime (≥ 2026-06-05).",
    f"Fee rate {mj.CRYPTO_FEE_RATE} (config default; the per-market fees.rate is a follow-up).",
    "Outcome = the official Polymarket resolution (historical_resolutions); a locked/forced pair "
    "pays $1 regardless of outcome, a naked leg $1 iff its side wins.",
    "Maker fill time ≈ the first recorded book ts where the opposite ask crosses the limit; the "
    "imbalance timeline and the 40 s deadline are driven off that approximation.",
    "Float parsing (research, not accounting); the 5-dp fee rounding is replicated at the money "
    "boundary; net = payoff − cost and the three-way split reconcile to < $0.01.",
]


# ---- current-regime helpers (local; learn_walkforward pulls in lightgbm) ----
def _regime_floor_ms(regime_from: date) -> int:
    return int(datetime(regime_from.year, regime_from.month, regime_from.day,
                        tzinfo=timezone.utc).timestamp()) * 1000


def _current_regime_since(since_ms: int | None, regime_from: date) -> int:
    floor = _regime_floor_ms(regime_from)
    return max(floor, since_ms) if since_ms is not None else floor


def _group_current_regime(oos: pd.DataFrame, since_ms: int) -> dict:
    """``{(series, window_open_ms): slice}`` over current-regime windows only."""
    cr = oos[oos["window_open_ms"].to_numpy(dtype="int64") >= since_ms]
    return {k: v for k, v in cr.groupby(["series", "window_open_ms"], sort=False)}


def _ladder_size(conf: float) -> float:
    """Ladder size (shares) for a confidence, or 0 to skip (conf below the lowest threshold)."""
    for thr, size in SIZE_LADDER:
        if conf >= thr:
            return size
    return 0.0


def _maker_completion_ts(book: dict, side_s: str, limit: float, lo_ms: int, hi_ms: int):
    """The first recorded book snapshot in ``[lo_ms, hi_ms)`` where the OPPOSITE-leg ask reaches
    ``≤ limit`` (the market comes to our resting maker bid at ``limit``). Returns ``(fill_ts,
    displayed_opp_size)`` or ``None``. Sibling of ``backtest_pair_lean._maker_completion`` that
    also returns the fill timestamp (needed for the imbalance timeline + the 40 s deadline)."""
    ts = book["ts"]
    lo = int(np.searchsorted(ts, lo_ms, side="left"))
    hi = int(np.searchsorted(ts, hi_ms, side="left"))  # [lo, hi) strictly before hi_ms
    if hi <= lo:
        return None
    if side_s == "Up":  # complete Down: Down ask = 1 − Up bid, size = Up bid size
        opp_ask = 1.0 - book["bid_px"][lo:hi]
        opp_sz = book["bid_sz"][lo:hi]
    else:  # complete Up: Up ask
        opp_ask = book["ask_px"][lo:hi]
        opp_sz = book["ask_sz"][lo:hi]
    mask = (np.isfinite(opp_ask) & (opp_ask > 0.0) & (opp_ask < 1.0)
            & (opp_ask <= limit) & (opp_sz > 0.0))
    if not mask.any():
        return None
    idx = int(mask.argmax())
    return int(ts[lo + idx]), float(opp_sz[idx])


def _opp(side: str) -> str:
    return "Down" if side == "Up" else "Up"


def _sgn(side: str) -> float:
    return 1.0 if side == "Up" else -1.0


# ---- per-window simulation (one version) -----------------------------------
def _simulate_window(fires: pd.DataFrame, books: dict, up_token: str, outcome_up: int,
                     close_ms: int, *, version: str, latency_ms: int, fee_rate: float,
                     window_budget: float) -> dict:
    """Run the confidence-ladder hedged-pair strategy over one window's OOS fires for one
    ``version``. Event-driven: entries at fire times, maker-lock events at the recorded fill ts,
    forced/naked resolution at each lot's ``min(open+40s, close)`` deadline. A single per-window
    budget (shared across both legs) is charged on every TAKER leg (entries + forced hedges);
    maker completions are zero-fee and budget-free. Returns a per-version aggregate for this
    window (no rows retained across windows)."""
    book = books.get(up_token)
    budget_left = window_budget        # single per-window budget, charged on taker legs
    net_imb = 0.0                      # signed net unhedged imbalance (Up +, Down −)
    lots: dict[int, dict] = {}
    events: list[tuple[int, int, int]] = []  # heap of (time_ms, kind{0 maker, 1 deadline}, lot_id)
    lot_rows: list[dict] = []
    n_fires = n_imb_skips = 0
    next_id = 0

    def _finalize_lot(lot: dict, f: float, opp_price: float, hedge_fee: float, k: float) -> None:
        n, price_s, entry_fee = lot["n"], lot["price_s"], lot["entry_fee"]
        m, limit, side = lot["locked_m"], lot["maker_limit"], lot["side"]
        won_side = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
        naked_val = 1.0 if won_side else 0.0
        frac_m, frac_f, frac_k = m / n, f / n, k / n
        locked_pnl = (m - (m * price_s + m * limit + entry_fee * frac_m)) if m > _EPS else 0.0
        forced_pnl = (f - (f * price_s + f * opp_price + entry_fee * frac_f + hedge_fee)
                      ) if f > _EPS else 0.0
        naked_pnl = (k * naked_val - (k * price_s + entry_fee * frac_k)) if k > _EPS else 0.0
        payoff = m + f + k * naked_val
        cost = (n * price_s + entry_fee + (m * limit if m > _EPS else 0.0)
                + (f * opp_price if f > _EPS else 0.0) + hedge_fee)
        net = payoff - cost
        lot_rows.append({
            "side": side, "n": n, "m": m, "f": f, "k": k, "price_s": price_s,
            "entry_fee": entry_fee, "hedge_fee": hedge_fee, "net": net,
            "locked_pnl": locked_pnl, "forced_pnl": forced_pnl, "naked_pnl": naked_pnl,
            "won": bool(net > 0.0),
        })

    def _apply_maker(lot: dict) -> None:
        # Resting-maker completion: zero fee, budget-free (only takers consume the budget).
        nonlocal net_imb
        m = min(lot["unhedged"], lot["maker_size_o"])
        if m > _EPS:
            net_imb -= _sgn(lot["side"]) * m
            lot["locked_m"] += m
            lot["unhedged"] -= m

    def _apply_deadline(lot: dict) -> None:
        nonlocal net_imb, budget_left
        remaining = lot["unhedged"]
        f, opp_price, hedge_fee = 0.0, float("nan"), 0.0
        if remaining > _EPS and version != "no_forced":
            row = mj._book_asof(books, up_token, lot["hedge_deadline"])
            quote = mj._leaned_quote(row, _opp(lot["side"])) if row is not None else None
            if quote is not None:
                mkt_price, opp_avail = quote
                allow = not (version == "cost_cap" and (lot["price_s"] + mkt_price) > FORCED_COST_CAP)
                if allow:
                    f = min(remaining, opp_avail, budget_left / mkt_price if mkt_price > 0.0 else 0.0)
                    if f > _EPS:
                        opp_price = mkt_price
                        hedge_fee = lm.taker_fee(f, fee_rate, mkt_price)
                        budget_left -= f * mkt_price
                        net_imb -= _sgn(lot["side"]) * f
                    else:
                        f = 0.0
        _finalize_lot(lot, f, opp_price, hedge_fee, remaining - f)

    def _drain(now_ms: int) -> None:
        while events and events[0][0] <= now_ms:
            _t, kind, lid = heapq.heappop(events)
            (_apply_maker if kind == 0 else _apply_deadline)(lots[lid])

    for r in fires.sort_values("sample_ts_ms").itertuples(index=False):
        p_up = float(r.p_up)
        if not np.isfinite(p_up):
            continue
        side = "Up" if p_up >= 0.5 else "Down"
        ladder = _ladder_size(max(p_up, 1.0 - p_up))
        if ladder <= 0.0:
            continue
        fire_ts = int(r.sample_ts_ms)
        _drain(fire_ts)                       # apply prior hedge events up to this decision time
        n_fires += 1
        prospective = net_imb + _sgn(side) * ladder
        if (side == "Up" and prospective > MAX_IMBALANCE) or (side == "Down" and prospective < -MAX_IMBALANCE):
            n_imb_skips += 1
            continue
        if budget_left < mj.MIN_NOTIONAL:
            continue
        open_ts = fire_ts + latency_ms
        row = mj._book_asof(books, up_token, open_ts)
        quote = mj._leaned_quote(row, side) if row is not None else None
        if quote is None:
            continue
        price_s, avail = quote
        n = min(ladder, avail, budget_left / price_s)
        if n <= _EPS or n * price_s < mj.MIN_NOTIONAL:
            continue
        entry_fee = lm.taker_fee(n, fee_rate, price_s)
        budget_left -= n * price_s
        net_imb += _sgn(side) * n
        maker_limit = 1.0 - price_s
        hedge_deadline = min(open_ts + HEDGE_DEADLINE_MS, close_ms)
        lot = {"side": side, "n": n, "price_s": price_s, "entry_fee": entry_fee,
               "open_ts": open_ts, "hedge_deadline": hedge_deadline, "maker_limit": maker_limit,
               "unhedged": n, "locked_m": 0.0, "maker_size_o": 0.0}
        maker = (_maker_completion_ts(book, side, maker_limit, open_ts, hedge_deadline)
                 if (book is not None and hedge_deadline > open_ts) else None)
        lots[next_id] = lot
        if maker is not None:
            lot["maker_size_o"] = maker[1]
            heapq.heappush(events, (maker[0], 0, next_id))
        heapq.heappush(events, (hedge_deadline, 1, next_id))
        next_id += 1

    _drain(np.iinfo(np.int64).max)  # flush remaining maker-locks + deadlines (finalize every lot)

    agg = dict(_AGG0)
    agg["fires"] = n_fires
    agg["imbalance_skips"] = n_imb_skips
    agg["windows_traded"] = 1 if lot_rows else 0
    for lr in lot_rows:
        agg["lots"] += 1
        agg["net"] += lr["net"]
        agg["fees"] += lr["entry_fee"] + lr["hedge_fee"]
        agg["wins"] += 1 if lr["won"] else 0
        agg["locked_pnl"] += lr["locked_pnl"]
        agg["forced_pnl"] += lr["forced_pnl"]
        agg["naked_pnl"] += lr["naked_pnl"]
        agg["shares_n"] += lr["n"]
        agg["shares_locked"] += lr["m"]
        agg["shares_forced"] += lr["f"]
        agg["shares_naked"] += lr["k"]
        agg["notional"] += lr["n"] * lr["price_s"]
    return agg


# ---- aggregation -----------------------------------------------------------
_AGG0 = {"windows_traded": 0, "lots": 0, "fires": 0, "imbalance_skips": 0, "wins": 0,
         "net": 0.0, "fees": 0.0, "locked_pnl": 0.0, "forced_pnl": 0.0, "naked_pnl": 0.0,
         "shares_n": 0.0, "shares_locked": 0.0, "shares_forced": 0.0, "shares_naked": 0.0,
         "notional": 0.0}


def _add(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v


# ---- day worker ------------------------------------------------------------
def _process_day(paths, series_key, d, day_str, groups, res_map, excluded, *,
                 latency_ms, fee_rate, window_budget) -> dict:
    """Reconstruct one (series, day)'s windows once (PM book + official outcome + OOS fires) and
    run all 3 versions over the SHARED books. Returns the day's per-version aggregates."""
    wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
    versions = {v: dict(_AGG0) for v in VERSIONS}
    n_win = 0
    for w in wins:
        open_ms, close_ms = int(w["window_open_ms"]), int(w["window_close_ms"])
        official = res_map.get((series_key, open_ms))
        if official is None:
            continue
        g = groups.get((series_key, open_ms))
        if g is None or not len(g):
            continue
        upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
        if upq.empty:
            continue
        up_token = str(upq["token_id"].iloc[0])
        outcome_up = 1 if official["outcome"] == "Up" else 0
        books = {up_token: tpm.pm_book_arrays(upq)}
        fires = pd.DataFrame({"sample_ts_ms": g["sample_ts_ms"].to_numpy(dtype="int64"),
                              "p_up": g["p_up"].to_numpy(dtype=float)})
        n_win += 1
        for v in VERSIONS:
            _add(versions[v], _simulate_window(fires, books, up_token, outcome_up, close_ms,
                                               version=v, latency_ms=latency_ms, fee_rate=fee_rate,
                                               window_budget=window_budget))
    return {"series": series_key, "day": day_str, "windows": n_win, "versions": versions}


# ---- worker ----------------------------------------------------------------
def backtest_conf_pair(
    paths: Paths, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None, max_days: int | None = None,
    network_ms: int = 5, venue_delay_ms: int = 250, window_budget: float = WINDOW_BUDGET,
    fee_rate: float = mj.CRYPTO_FEE_RATE, target: str = "dir10", variant: str = "full",
    oos: pd.DataFrame | None = None, res_map: dict | None = None,
    regime_from: date = CURRENT_REGIME_FROM, out_name: str = "conf_pair",
) -> dict:
    """Replay the confidence-ladder hedged-pair strategy (all 3 versions) over the current-regime
    windows; write the report. Returns the metrics dict. Effective latency = ``network + venue``
    (default 255 ms). ``oos`` / ``res_map`` load from disk when ``None`` (injectable for tests);
    ``regime_from`` / ``max_days`` bound the range. Resumable per-(series, day)."""
    paths.ensure_out()
    effective = network_ms + venue_delay_ms
    out_dir = paths.out_dir / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    since_floor = _current_regime_since(since_ms, regime_from)
    if oos is None:
        path = paths.out_dir / "learn_walkforward" / f"oos_{target}_{variant}.parquet"
        assert_parquet_ready(path, label=f"oos_{target}_{variant}.parquet", min_rows=1)
        # Predicate-pushdown: only this run's series + the current-regime/bounded window range,
        # so per-series parallel workers each hold a small slice of the 10M-row frame.
        filters = [("series", "in", list(series_keys)), ("window_open_ms", ">=", since_floor)]
        if until_ms is not None:
            filters.append(("window_open_ms", "<", until_ms))
        oos = pd.read_parquet(path, columns=["series", "window_open_ms", "sample_ts_ms", "p_up"],
                              filters=filters)
    if res_map is None:
        res_map = hr.load_resolution_map(paths, series_keys)
    groups = _group_current_regime(oos, since_floor)

    agg: dict = {(v, sk): dict(_AGG0) for v in VERSIONS for sk in series_keys}
    windows: dict = defaultdict(int)
    done: set[tuple[str, str]] = set()

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _accumulate(data: dict) -> None:
        sk = data["series"]
        windows[sk] += int(data["windows"])
        for v, a in data["versions"].items():
            _add(agg[(v, sk)], a)

    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            if data["series"] in series_keys:
                _accumulate(data)
            done.add((data["series"], data["day"]))
        except (KeyError, ValueError):
            continue

    for series_key in series_keys:
        days = hlbl._days_for_series(paths, series_key, since_floor, until_ms)
        if max_days is not None:
            days = days[:max_days]
        for d in days:
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            day = _process_day(paths, series_key, d, day_str, groups, res_map, excluded,
                               latency_ms=effective, fee_rate=fee_rate, window_budget=window_budget)
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(day, default=eh._json_default), encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            _accumulate(day)
            print(f"[backtest_conf_pair] {series_key} {day_str}: {day['windows']} windows "
                  f"({len(done)} days done)", flush=True)

    return _finalize(paths, out_dir, series_keys, agg, windows, network_ms, venue_delay_ms,
                     effective, window_budget, fee_rate, target, variant)


# ---- finalize --------------------------------------------------------------
def _rate(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


def _summary(a: dict) -> dict:
    """Per-version summary of an aggregate: total net, the three-way split (+ its reconciliation),
    counts, win rate, fees, share breakdown."""
    split_sum = a["locked_pnl"] + a["forced_pnl"] + a["naked_pnl"]
    resid = a["net"] - split_sum
    return {
        "windows_traded": int(a["windows_traded"]), "lots": int(a["lots"]),
        "fires": int(a["fires"]), "imbalance_skips": int(a["imbalance_skips"]),
        "net_pnl": float(a["net"]), "fees": float(a["fees"]),
        "win_rate": _rate(a["wins"], a["lots"]), "net_per_lot": _rate(a["net"], a["lots"]),
        "locked_pairs_pnl": float(a["locked_pnl"]), "forced_hedge_pnl": float(a["forced_pnl"]),
        "naked_leg_pnl": float(a["naked_pnl"]), "split_sum": float(split_sum),
        "reconcile_residual": float(resid), "reconciles": bool(abs(resid) <= 0.01),
        "shares_entered": float(a["shares_n"]), "shares_locked": float(a["shares_locked"]),
        "shares_forced": float(a["shares_forced"]), "shares_naked": float(a["shares_naked"]),
    }


def _finalize(paths, out_dir, series_keys, agg, windows, network_ms, venue_delay_ms, effective,
              window_budget, fee_rate, target, variant) -> dict:
    n_windows = int(sum(windows.values()))
    versions_out: dict = {}
    per_series_rows: list[dict] = []
    recon_ok = True
    recon_detail: dict = {}
    for v in VERSIONS:
        overall = dict(_AGG0)
        for sk in series_keys:
            _add(overall, agg[(v, sk)])
        osum = _summary(overall)
        per_series = {}
        for sk in series_keys:
            s = _summary(agg[(v, sk)])
            per_series[sk] = s
            per_series_rows.append({"series": sk, "version": v, **s})
            recon_ok = recon_ok and s["reconciles"]
        recon_ok = recon_ok and osum["reconciles"]
        recon_detail[v] = {"overall_residual": osum["reconcile_residual"],
                           "overall_reconciles": osum["reconciles"],
                           "per_series_max_residual": max(
                               (abs(per_series[sk]["reconcile_residual"]) for sk in series_keys),
                               default=0.0)}
        versions_out[v] = {"label": VERSION_LABEL[v], **osum, "per_series": per_series}

    result = {
        "params": {"title": "Confidence-ladder hedged-pair backtest", "series": list(series_keys),
                   "target": target, "variant": variant, "network_ms": network_ms,
                   "venue_delay_ms": venue_delay_ms, "effective_latency_ms": effective,
                   "window_budget": window_budget,
                   "budget_note": "single per-window $%.0f (shared; charged on taker legs)" % window_budget,
                   "fee_rate": fee_rate,
                   "size_ladder": [[t, s] for t, s in SIZE_LADDER], "max_imbalance": MAX_IMBALANCE,
                   "hedge_deadline_ms": HEDGE_DEADLINE_MS, "forced_cost_cap": FORCED_COST_CAP},
        "caveats": CAVEATS,
        "current_regime_windows": n_windows,
        "budget": {"window_budget": window_budget, "per_side": False},
        "baselines": {"never_trading_net": 0.0},
        "versions": versions_out,
        "reconciliation": {"all_ok": bool(recon_ok), "tolerance": 0.01, "by_version": recon_detail},
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_outputs(out_dir, result, per_series_rows)
    if not recon_ok:
        raise AssertionError(
            "three-way P&L split does NOT reconcile to total net within $0.01: "
            + json.dumps(recon_detail, default=eh._json_default))
    return result


# ---- outputs ---------------------------------------------------------------
def _write_outputs(out_dir: Path, result: dict, per_series_rows: list[dict]) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default),
                                          encoding="utf-8")
    ps_cols = ["series", "version", "windows_traded", "lots", "fires", "imbalance_skips",
               "net_pnl", "fees", "win_rate", "locked_pairs_pnl", "forced_hedge_pnl",
               "naked_leg_pnl", "reconcile_residual", "reconciles"]
    pd.DataFrame(per_series_rows, columns=ps_cols).to_csv(out_dir / "per_series.csv", index=False)

    ver_cols = [("version", "version"), ("net_pnl", "net $"), ("locked_pairs_pnl", "locked $"),
                ("forced_hedge_pnl", "forced $"), ("naked_leg_pnl", "naked $"),
                ("lots", "trades"), ("fires", "fires"), ("windows_traded", "windows"),
                ("win_rate", "win"), ("fees", "fees $"), ("reconciles", "reconciles")]
    ver_rows = [{"version": result["versions"][v]["label"], **result["versions"][v]}
                for v in VERSIONS]
    ps_html_cols = [("series", "series"), ("version", "version"), ("net_pnl", "net $"),
                    ("locked_pairs_pnl", "locked $"), ("forced_hedge_pnl", "forced $"),
                    ("naked_leg_pnl", "naked $"), ("lots", "trades"), ("win_rate", "win"),
                    ("fees", "fees $")]
    ps_html_rows = [{**r, "version": VERSION_LABEL.get(r["version"], r["version"])}
                    for r in per_series_rows]
    p = result["params"]
    recon = result["reconciliation"]
    bug = "" if recon["all_ok"] else " bug"
    caveats = "".join(f"<li>{c}</li>" for c in result["caveats"])
    recon_line = ("Three-way split reconciles to total net for all versions (≤ $0.01)."
                  if recon["all_ok"] else
                  "⚠ Three-way split does NOT reconcile — see reconciliation.by_version.")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{p['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} .muted {{ color: #888; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 .verdict.bug {{ border-left-color: #d62728; background: #fdeeee; }}
</style></head><body>
<h1>{p['title']}</h1>
<p class="muted">Confidence-ladder taker entries (40/20/10 sh at conf ≥ 0.80/0.70/0.60), hedged by
a resting maker at 1 − price_s, force-completed (or ridden naked) at 40 s. <b>Single per-window
budget ${p['window_budget']:,.0f}</b> (shared safety ceiling, charged on taker legs; the ≤ 40-share
ladder is the binding size limit); effective latency {p['effective_latency_ms']} ms; current-regime
windows (≥ 2026-06-05); marked to the official resolution. Reconstructed book; conservative fills.</p>
<div class="verdict{bug}">Over {result['current_regime_windows']:,} current-regime windows.
{recon_line} Never-trading nets $0.00. Net PnL per version (three-way split): """ + "; ".join(
        f"{VERSION_LABEL[v]} ${result['versions'][v]['net_pnl']:,.2f} "
        f"(locked ${result['versions'][v]['locked_pairs_pnl']:,.2f} / forced "
        f"${result['versions'][v]['forced_hedge_pnl']:,.2f} / naked "
        f"${result['versions'][v]['naked_leg_pnl']:,.2f})" for v in VERSIONS) + f"""</div>
<h2>By version (overall)</h2>
{eh._table_html(ver_rows, ver_cols)}
<h2>Per series per version</h2>
{eh._table_html(ps_html_rows, ps_html_cols)}
<h2>Caveats (read these)</h2><ul>{caveats}</ul>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "backtest_conf_pair")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    parser.add_argument("--target", default="dir10", help="OOS target (default dir10)")
    parser.add_argument("--variant", default="full", help="OOS feature variant (default full)")
    parser.add_argument("--max-days", type=int, default=None,
                        help="cap the number of UTC days processed per series (for a small slice)")
    parser.add_argument("--network-ms", type=int, default=5, help="network latency ms (default 5, VPS)")
    parser.add_argument("--venue-delay-ms", type=int, default=250,
                        help="venue taker delay ms (default 250; effective 255 = vps255 reference)")
    parser.add_argument("--window-budget", type=float, default=WINDOW_BUDGET,
                        help=f"single per-window budget $ (default {WINDOW_BUDGET:.0f}; a safety ceiling)")
    parser.add_argument("--fee-rate", type=float, default=mj.CRYPTO_FEE_RATE)
    parser.add_argument("--out-name", default="conf_pair", help="output subdir under out/backtests/")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None

    print(f"[backtest_conf_pair] telonex={paths.telonex_dir} "
          f"out={paths.out_dir / 'backtests' / args.out_name} "
          f"latency=net {args.network_ms}+venue {args.venue_delay_ms}={args.network_ms + args.venue_delay_ms}ms "
          f"window-budget=${args.window_budget:,.0f}")
    result = backtest_conf_pair(paths, series=series, since_ms=since_ms, until_ms=until_ms,
                                max_days=args.max_days, network_ms=args.network_ms,
                                venue_delay_ms=args.venue_delay_ms, window_budget=args.window_budget,
                                fee_rate=args.fee_rate, target=args.target, variant=args.variant,
                                out_name=args.out_name)
    print(f"[backtest_conf_pair] {result['current_regime_windows']:,} current-regime windows; "
          f"never-trade $0.00; reconcile_all_ok={result['reconciliation']['all_ok']}")
    for v in VERSIONS:
        s = result["versions"][v]
        print(f"[backtest_conf_pair]   {VERSION_LABEL[v]}: net ${s['net_pnl']:,.2f} = locked "
              f"${s['locked_pairs_pnl']:,.2f} + forced ${s['forced_hedge_pnl']:,.2f} + naked "
              f"${s['naked_leg_pnl']:,.2f} (resid ${s['reconcile_residual']:+.4f}); "
              f"{s['lots']:,} trades, win {s['win_rate']:.0%}, fees ${s['fees']:,.2f}")
        for sk in result["params"]["series"]:
            ps = s["per_series"][sk]
            print(f"[backtest_conf_pair]       {sk}: net ${ps['net_pnl']:,.2f} = "
                  f"L ${ps['locked_pairs_pnl']:,.2f} + F ${ps['forced_hedge_pnl']:,.2f} + "
                  f"N ${ps['naked_leg_pnl']:,.2f} ({ps['lots']:,} trades)")
    procmem.report_peak_rss("backtest_conf_pair", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[backtest_conf_pair] wrote metrics.json + per_series.csv + report.html to "
          f"{paths.out_dir / 'backtests' / args.out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
