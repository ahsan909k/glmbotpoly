"""Stage — backtest_burst_pair: the operator's **high-confidence burst pair-building** strategy.

Per current-regime window, forward over the per-market 15-second OOS predictions from
``accuracy_push`` (``out/accuracy_push/oos_<market>.parquet``) in ``sample_ts_ms`` order. On every
**qualifying** signal (``|p_up − 0.5| ≥ θ_85``, the per-market gate at which held-out accuracy
reaches 85 %):

1. **Burst entry (taker).** Fire a burst on the predicted side — ``BURST_ORDERS`` (3) sequential
   ``BURST_SIZE`` (20)-share FAK orders against the displayed level as-of ``fire_ts + 255 ms``
   (net 5 + venue 250). Each order fills only what the (single-level Telonex) book shows; the three
   orders consume the one displayed level in turn (conservative — no replenishment assumed). The
   per-burst **depth shortfall** (``60 − filled``) is reported, never imagined.
2. **Completion maker (zero fee).** Immediately rest a completion bid on the OPPOSITE side, sized to
   the filled lean shares, priced ``limit = C − price_s`` so the combined pair cost is ``≤ C``. It
   fills for ``min(filled, displayed_opp_size)`` at ``limit`` the first time the recorded opposite
   ask reaches ``≤ limit`` before close (the conservative queue/print proxy). A locked pair collects
   $1 at resolution for a cost of ``C`` ⇒ **positive locked-pair edge ``1 − C``** (unlike the
   break-even 1.00-cost hedge in ``conf_pair``). Sweep ``C ∈ {0.94, 0.96, 0.98}``.
3. **Inventory accumulation, capped.** Repeat on every new qualifying signal; unhedged shares
   accumulate. When total unhedged ``≥ U`` no new burst fires until completions catch up (completions
   reduce unhedged over the window, event-driven). **No force-hedging.** Sweep ``U ∈ {20, 40, 60}``.
4. **Final-seconds rule.** No new entry when ``τ ≤ 30 s``. Legs still unpaired at close ride to the
   **official** resolution ($1 iff the leaned side wins, else $0).

**P&L split** (reconciles EXACTLY to net): ``locked_pairs_pnl + stranded_legs_pnl == net``; fees are
the entry taker fees (completions are maker, zero-fee) and are embedded in both components. Reported
per series over the **current-regime** windows (``≥ 2026-06-05``, 255 ms effective, per-market real
fees, the 4 series BTC/ETH × 5m/15m), for every ``(C, U)`` cell.

**Benchmarks on identical windows:** never-trade ($0); the champion **model-taker** (dir10 at
θ=0.03, the deployed model, engine $10/window budget); the engine **momentum** trigger (same
$10/window; opt-out with ``--no-momentum``). No maker-core Telonex cache exists — reported as N/A.

**Control:** the same sweep on **shuffled** per-(series, day) outcomes. For a pair strategy a locked
pair pays $1 regardless of outcome, so the shuffle only re-marks stranded legs: if the REAL run
profits and the shuffled run *also* profits, the "edge" is the outcome-independent locked-pair
completion assumption (not the 85 % signal) — flagged as **not signal-driven**.

**No retraining / reuses** ``money_judge`` fills + ``backtest_momentum`` reconstruction. Resumable
per-(series, day); run per-series in parallel sharing one ``--out-name``. **Research only.**
Output ``out/backtests/burst_pair/``.

Run::

    python -m model_lab.backtest_burst_pair                       # full current regime, 4 series
    python -m model_lab.backtest_burst_pair --series BTC-5m       # one series (parallel worker)
    python -m model_lab.backtest_burst_pair --no-momentum --max-days 3   # fast slice
"""

from __future__ import annotations

import heapq
import json
import sys
import zlib
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_momentum as bm
from . import dataset as ds
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
CURRENT_REGIME_FROM = date(2026, 6, 5)
_EPS = 1e-9

# ---- strategy constants ----------------------------------------------------
BURST_ORDERS = 3            #: FAK orders per burst
BURST_SIZE = 20.0          #: shares per FAK order (3 × 20 = 60-share burst target)
C_SWEEP = (0.94, 0.96, 0.98)       #: pair-cost ceilings (completion limit = C − price_s)
U_SWEEP = (20.0, 40.0, 60.0)       #: unhedged-share caps (block new bursts at the cap)
NO_ENTRY_FINAL_SECS = 30           #: no new entry within this of close
WINDOW_BUDGET = 1000.0             #: per-window safety ceiling (rarely binds; U + depth + signal count bind)
MODEL_TAKER_THETA = 0.03           #: champion model-taker gate (dir10)
BENCH_BUDGET = mj.DEFAULT_WINDOW_BUDGET  #: $10/window for momentum + model-taker (engine config)
CONFIGS: tuple[tuple[float, float], ...] = tuple((c, u) for c in C_SWEEP for u in U_SWEEP)

CAVEATS = [
    "Burst = 3 sequential 20-share FAK orders consuming ONE displayed (single-level Telonex) book "
    "level in turn at the as-of price (fire + 255 ms); no replenishment between orders (conservative "
    "on depth). Depth shortfall = 60 − filled, reported per burst.",
    "Completion = a resting maker bid at limit = C − price_s; fills for min(filled, displayed "
    "opposite size) at the limit the first time the recorded opposite ask reaches ≤ limit before "
    "close — the natural but mildly optimistic maker-fill proxy (Telonex has no PM trade tape; the "
    "real §9 model queues behind displayed size and fills on prints through the price). One fill "
    "attempt per lot (a persistent maker could complete more) ⇒ conservative on completion.",
    "A locked pair costs exactly C (price_s + limit) and collects $1 ⇒ locked edge 1 − C, minus the "
    "pro-rated entry taker fee. The completion pays ZERO fee and NO rebate (conservative). Avg pair "
    "cost is C by construction; completion RATE is the informative metric.",
    "Effective taker latency = network + venue (5 + 250 = 255 ms, vps255 reference); a "
    f"{mj.BOOK_STALENESS_MS} ms book-staleness skip biases toward fewer fires. Flat 255 ms is only "
    "correct WITHIN the current regime (≥ 2026-06-05).",
    f"Fee rate {mj.CRYPTO_FEE_RATE} (config default; per-market fees.rate is a follow-up).",
    "Signal = the per-market 15 s accuracy_push OOS p_up at the θ_85 gate (held-out ≥ 85 % 15 s "
    "direction accuracy). NOTE the 15 s signal predicts the 15 s mid move, NOT the window outcome; "
    "stranded legs settle on the WINDOW outcome, so an accurate 15 s signal helps the ENTRY, not "
    "directly the stranded settlement.",
    "Outcome = official Polymarket resolution (historical_resolutions). Benchmarks (momentum, "
    "model-taker) use the engine's $10/window budget on identical windows; never-trade = $0. No "
    "maker-core Telonex backtest cache exists (N/A).",
    "Float parsing (research, not accounting); 5-dp fee rounding replicated; net = locked + stranded "
    "reconciles to < $0.01.",
]


# ---- helpers ---------------------------------------------------------------
def _regime_floor_ms(regime_from: date) -> int:
    return int(datetime(regime_from.year, regime_from.month, regime_from.day,
                        tzinfo=timezone.utc).timestamp()) * 1000


def _current_regime_since(since_ms: int | None, regime_from: date) -> int:
    floor = _regime_floor_ms(regime_from)
    return max(floor, since_ms) if since_ms is not None else floor


def _maker_completion_ts(book: dict, side_s: str, limit: float, lo_ms: int, hi_ms: int):
    """First recorded book ts in ``[lo_ms, hi_ms)`` where the OPPOSITE-leg ask reaches ``≤ limit``
    (the market comes to our resting maker bid at ``limit``). Returns ``(fill_ts, displayed_opp_size)``
    or ``None``. Sibling of ``backtest_conf_pair._maker_completion_ts``."""
    ts = book["ts"]
    lo = int(np.searchsorted(ts, lo_ms, side="left"))
    hi = int(np.searchsorted(ts, hi_ms, side="left"))
    if hi <= lo:
        return None
    if side_s == "Up":  # complete Down: Down ask = 1 − Up bid, size = Up bid size
        opp_ask, opp_sz = 1.0 - book["bid_px"][lo:hi], book["bid_sz"][lo:hi]
    else:  # complete Up: Up ask
        opp_ask, opp_sz = book["ask_px"][lo:hi], book["ask_sz"][lo:hi]
    mask = (np.isfinite(opp_ask) & (opp_ask > 0.0) & (opp_ask < 1.0)
            & (opp_ask <= limit) & (opp_sz > 0.0))
    if not mask.any():
        return None
    idx = int(mask.argmax())
    return int(ts[lo + idx]), float(opp_sz[idx])


# ---- per-window simulation (one (C, U) cell) -------------------------------
def _simulate_window(fires: list[tuple[int, float]], books: dict, up_token: str, outcome_up: int,
                     close_ms: int, *, C: float, U: float, latency_ms: int, fee_rate: float,
                     window_budget: float) -> dict:
    """Run the burst pair-building strategy over one window's QUALIFYING fires for one ``(C, U)``.
    Event-driven: bursts at fire times, maker completions at their recorded fill ts (freeing cap),
    stranded legs settled at close. Returns a per-window aggregate (+ arrays for distributions)."""
    book = books.get(up_token)
    budget_left = window_budget
    unhedged = 0.0
    peak_unhedged = 0.0
    lots: dict[int, dict] = {}
    events: list[tuple[int, int]] = []  # heap (fill_ts, lot_id) maker completions
    next_id = 0
    n_fired = n_skip_cap = n_skip_final = n_skip_nobook = n_skip_budget = n_skip_nofill = 0
    shares_entered = 0.0
    shortfalls: list[float] = []

    def _drain(now_ms: int) -> None:
        nonlocal unhedged
        while events and events[0][0] <= now_ms:
            _t, lid = heapq.heappop(events)
            lot = lots[lid]
            m = min(lot["unhedged_lot"], lot["maker_size_o"])
            if m > _EPS:
                unhedged -= m
                lot["locked_m"] += m
                lot["unhedged_lot"] -= m

    for fire_ts, p_up in fires:
        side = "Up" if p_up >= 0.5 else "Down"
        _drain(fire_ts)
        if close_ms - fire_ts <= NO_ENTRY_FINAL_SECS * 1000:
            n_skip_final += 1
            continue
        if unhedged >= U:
            n_skip_cap += 1
            continue
        if budget_left < mj.MIN_NOTIONAL:
            n_skip_budget += 1
            continue
        open_ts = fire_ts + latency_ms
        row = mj._book_asof(books, up_token, open_ts)
        quote = mj._leaned_quote(row, side) if row is not None else None
        if quote is None:
            n_skip_nobook += 1
            continue
        price_s, avail = quote
        # burst: 3 sequential FAK orders consume ONE displayed level in turn.
        filled, remaining_disp = 0.0, avail
        for _k in range(BURST_ORDERS):
            take = min(BURST_SIZE, remaining_disp, budget_left / price_s)
            if take * price_s < mj.MIN_NOTIONAL:
                break
            filled += take
            remaining_disp -= take
            budget_left -= take * price_s
        if filled <= _EPS:
            n_skip_nofill += 1
            continue
        n_fired += 1
        shares_entered += filled
        shortfalls.append(BURST_ORDERS * BURST_SIZE - filled)
        entry_fee = lm.taker_fee(filled, fee_rate, price_s)
        unhedged += filled
        peak_unhedged = max(peak_unhedged, unhedged)
        limit = C - price_s
        lot = {"side": side, "n": filled, "price_s": price_s, "entry_fee": entry_fee,
               "limit": limit, "unhedged_lot": filled, "locked_m": 0.0, "maker_size_o": 0.0}
        if limit > _EPS and book is not None:
            comp = _maker_completion_ts(book, side, limit, open_ts, close_ms)
            if comp is not None:
                lot["maker_size_o"] = comp[1]
                heapq.heappush(events, (comp[0], next_id))
        lots[next_id] = lot
        next_id += 1

    _drain(np.iinfo(np.int64).max)  # apply all remaining completions, then settle stranded at close

    agg = dict(_AGG0)
    agg["fires"] = len(fires)
    agg["bursts_fired"] = n_fired
    agg["skip_cap"] = n_skip_cap
    agg["skip_final"] = n_skip_final
    agg["skip_nobook"] = n_skip_nobook
    agg["skip_budget"] = n_skip_budget
    agg["skip_nofill"] = n_skip_nofill
    agg["windows_traded"] = 1 if lots else 0
    agg["shares_entered"] = shares_entered
    agg["depth_shortfall_total"] = float(sum(shortfalls))
    for lot in lots.values():
        n, m, limit = lot["n"], lot["locked_m"], lot["limit"]
        k = n - m
        price_s, ef, side = lot["price_s"], lot["entry_fee"], lot["side"]
        won = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
        frac_m, frac_k = (m / n, k / n) if n > 0 else (0.0, 0.0)
        locked_pnl = (m - (m * price_s + m * limit + ef * frac_m)) if m > _EPS else 0.0
        stranded_pnl = (k * (1.0 if won else 0.0) - (k * price_s + ef * frac_k)) if k > _EPS else 0.0
        net = locked_pnl + stranded_pnl
        agg["lots"] += 1
        agg["net"] += net
        agg["fees"] += ef
        agg["locked_pnl"] += locked_pnl
        agg["stranded_pnl"] += stranded_pnl
        agg["shares_locked"] += m
        agg["shares_stranded"] += k
        agg["notional"] += n * price_s
        agg["wins"] += 1 if net > 0.0 else 0
    # per-window arrays for distributions (exact quantiles at finalize).
    agg["_shares_window"] = [shares_entered] if agg["windows_traded"] else []
    agg["_peak_window"] = [peak_unhedged] if agg["windows_traded"] else []
    agg["_shortfalls"] = shortfalls
    return agg


_AGG0 = {"windows_traded": 0, "lots": 0, "fires": 0, "bursts_fired": 0, "skip_cap": 0,
         "skip_final": 0, "skip_nobook": 0, "skip_budget": 0, "skip_nofill": 0, "wins": 0,
         "net": 0.0, "fees": 0.0, "locked_pnl": 0.0, "stranded_pnl": 0.0, "shares_entered": 0.0,
         "shares_locked": 0.0, "shares_stranded": 0.0, "depth_shortfall_total": 0.0, "notional": 0.0}
_LIST_KEYS = ("_shares_window", "_peak_window", "_shortfalls")


def _add(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if k in _LIST_KEYS:
            dst.setdefault(k, []).extend(v)
        else:
            dst[k] = dst.get(k, 0.0) + v


# ---- day worker ------------------------------------------------------------
def _process_day(paths, series_key, d, day_str, burst_by_win, dir10_by_win, res_map, excluded,
                 *, latency_ms, fee_rate, window_budget, with_momentum, with_dir10=True,
                 shuffle_seed) -> dict:
    """Reconstruct one (series, day)'s windows once (PM book + official outcome + burst/dir10 fires,
    and — for momentum — the 1 s grid + mid ring) and run all cells + benchmarks over SHARED books.
    The window universe = every window with an official resolution + a non-empty Up book (0-signal
    windows included, so the signals-per-window distribution is honest). Returns the day's
    aggregates: real + shuffled cell aggregates, per-series signal counts, and benchmark trades."""
    wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
    real = {cu: dict(_AGG0) for cu in CONFIGS}
    shuf = {cu: dict(_AGG0) for cu in CONFIGS}
    signals_per_window: list[int] = []
    dir10_fires_rows: list[dict] = []
    books_day: dict[str, dict] = {}
    model_rows: list[dict] = []
    win_by_key: dict[tuple, dict] = {}
    n_win = 0

    # base-rate-preserving shuffled control: permute outcomes across the day's fire-having windows
    # (deterministic per (series, day) via a crc32 salt). Sibling of backtest_momentum._shuffle_outcomes.
    fire_wins = [(int(w["window_open_ms"]),
                  1 if (res_map.get((series_key, int(w["window_open_ms"]))) or {}).get("outcome") == "Up" else 0)
                 for w in wins
                 if res_map.get((series_key, int(w["window_open_ms"])))
                 and burst_by_win.get((series_key, int(w["window_open_ms"])))]
    perm_outcome: dict[int, int] = {}
    if fire_wins:
        opens = [o for o, _ in fire_wins]
        outs = np.array([o for _, o in fire_wins])
        salt = zlib.crc32(f"burst|{series_key}|{day_str}".encode()) % 1_000_000
        perm = np.random.default_rng(shuffle_seed + salt).permutation(len(outs))
        perm_outcome = {o: int(v) for o, v in zip(opens, outs[perm])}

    # momentum reconstruction (optional, identical windows).
    grid = pd.DataFrame()
    ring_ts = ring_px = None
    if with_momentum and wins:
        asset0 = hc.SLUG_PREFIXES[hc.series_prefix(series_key)][1]
        symbol = hc.SYMBOL_BY_ASSET[asset0]
        grid, ring_ts, ring_px = bm._day_grid_and_ring(paths, symbol, asset0, d)
    strike_tol_ms = ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0

    for w in wins:
        open_ms, close_ms = int(w["window_open_ms"]), int(w["window_close_ms"])
        official = res_map.get((series_key, open_ms))
        if official is None:
            continue
        upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
        if upq.empty:
            continue
        up_token = str(upq["token_id"].iloc[0])
        outcome_up = 1 if official["outcome"] == "Up" else 0
        books = {up_token: tpm.pm_book_arrays(upq)}
        books_day[up_token] = books[up_token]
        n_win += 1

        fires = burst_by_win.get((series_key, open_ms)) or []  # (ts, p_up), gated in the loader
        signals_per_window.append(len(fires))
        if fires:
            shuf_outcome = perm_outcome.get(open_ms, outcome_up)
            for cu in CONFIGS:
                C, U = cu
                _add(real[cu], _simulate_window(fires, books, up_token, outcome_up, close_ms,
                                                C=C, U=U, latency_ms=latency_ms, fee_rate=fee_rate,
                                                window_budget=window_budget))
                _add(shuf[cu], _simulate_window(fires, books, up_token, shuf_outcome, close_ms,
                                                C=C, U=U, latency_ms=latency_ms, fee_rate=fee_rate,
                                                window_budget=window_budget))

        d10 = dir10_by_win.get((series_key, open_ms)) if with_dir10 else None
        if d10:
            for ts_i, p_i in d10:
                dir10_fires_rows.append({"series": series_key, "window_open_ms": open_ms,
                                         "sample_ts_ms": ts_i, "p_up": p_i, "up_token": up_token,
                                         "outcome_up": outcome_up})

        if with_momentum and not grid.empty:
            st = hc.binance_anchored_strike(bm._grid_bars(grid), open_ms, strike_tol_ms)
            if st["strike"] > 0.0:
                win_by_key[(series_key, open_ms)] = {"up_token": up_token, "close_time": close_ms,
                                                     "outcome_up": outcome_up}
                model_rows.extend(bm._window_model_rows(grid, w, st))

    momentum_trades: list[dict] = []
    if with_momentum and model_rows:
        asset0 = hc.SLUG_PREFIXES[hc.series_prefix(series_key)][1]
        momentum_trades = mj._run_momentum(model_rows, {asset0: (ring_ts, ring_px)}, win_by_key,
                                           books_day, latency_ms=latency_ms, fee_rate=fee_rate,
                                           window_budget=BENCH_BUDGET)
    dir10_trades: list[dict] = []
    if with_dir10 and dir10_fires_rows:
        df10 = pd.DataFrame(dir10_fires_rows)
        dir10_trades, _ = mj._run_taker(df10, threshold=MODEL_TAKER_THETA, latency_ms=latency_ms,
                                        fee_rate=fee_rate, window_budget=BENCH_BUDGET,
                                        trade_budget=BENCH_BUDGET, books=books_day)

    return {"series": series_key, "day": day_str, "windows": n_win,
            "real": real, "shuf": shuf, "signals_per_window": signals_per_window,
            "momentum_net": float(sum(t["net"] for t in momentum_trades)),
            "momentum_trades": len(momentum_trades),
            "momentum_wins": sum(1 for t in momentum_trades if t["won"]),
            "dir10_net": float(sum(t["net"] for t in dir10_trades)),
            "dir10_trades": len(dir10_trades),
            "dir10_wins": sum(1 for t in dir10_trades if t["won"])}


# ---- fires loading ---------------------------------------------------------
def _load_theta_gates(paths: Paths, markets, oos_subdir: str = "accuracy_push") -> dict:
    """θ_85 gate per market from ``<oos_subdir>`` metrics.json (falls back to ``theta_gate`` when 85 %
    wasn't reached — flagged there). ``oos_subdir`` lets the challenger bake-off point at a different
    model's OOS + gates instead of the champion ``accuracy_push``."""
    path = paths.out_dir / oos_subdir / "metrics.json"
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    gates = {}
    for market in markets:
        r = m["markets"].get(market)
        if r and "theta_gate" in r:
            gates[market] = {"theta_gate": float(r["theta_gate"]),
                             "reached_85": bool(r.get("theta_85_reached", False))}
    return gates


def _load_burst_fires(paths: Paths, market: str, gate: float, since_ms: int, until_ms: int | None,
                      oos_subdir: str = "accuracy_push"):
    """Qualifying (``|p_up-0.5| >= gate``) current-regime fires for a market, grouped
    ``{(series, window_open_ms): [(sample_ts_ms, p_up), …]}`` in ts order. ``oos_subdir`` selects the
    producing model's OOS directory (default the champion ``accuracy_push``)."""
    path = paths.out_dir / oos_subdir / f"oos_{market}.parquet"
    assert_parquet_ready(path, label=f"oos_{market}.parquet", min_rows=1)
    df = pd.read_parquet(path, columns=["series", "window_open_ms", "sample_ts_ms", "p_up"])
    df = df[df["window_open_ms"] >= since_ms]
    if until_ms is not None:
        df = df[df["window_open_ms"] < until_ms]
    p = df["p_up"].to_numpy(float)
    df = df[np.isfinite(p) & (np.abs(p - 0.5) >= gate)]
    out: dict = {}
    for (s, w), g in df.sort_values("sample_ts_ms").groupby(["series", "window_open_ms"], sort=False):
        out[(s, int(w))] = list(zip(g["sample_ts_ms"].astype("int64"), g["p_up"].astype(float)))
    return out


def _load_dir10_fires(paths: Paths, series_keys, since_ms: int, until_ms: int | None):
    """dir10 champion OOS fires (all, ungated — the taker gates at θ) grouped by window."""
    path = paths.out_dir / "learn_walkforward" / "oos_dir10_full.parquet"
    if not path.exists():
        return {}
    filters = [("series", "in", list(series_keys)), ("window_open_ms", ">=", since_ms)]
    if until_ms is not None:
        filters.append(("window_open_ms", "<", until_ms))
    df = pd.read_parquet(path, columns=["series", "window_open_ms", "sample_ts_ms", "p_up"],
                         filters=filters)
    out: dict = {}
    for (s, w), g in df.sort_values("sample_ts_ms").groupby(["series", "window_open_ms"], sort=False):
        out[(s, int(w))] = list(zip(g["sample_ts_ms"].astype("int64"), g["p_up"].astype(float)))
    return out


# ---- orchestration ---------------------------------------------------------
def backtest_burst_pair(
    paths: Paths, *, series: tuple[str, ...] | None = None, since_ms: int | None = None,
    until_ms: int | None = None, max_days: int | None = None, network_ms: int = 5,
    venue_delay_ms: int = 250, window_budget: float = WINDOW_BUDGET,
    fee_rate: float = mj.CRYPTO_FEE_RATE, with_momentum: bool = True, with_dir10: bool = True,
    oos_subdir: str = "accuracy_push", shuffle_seed: int = 20260713,
    regime_from: date = CURRENT_REGIME_FROM, out_name: str = "burst_pair",
) -> dict:
    """Replay the burst pair-building sweep over the current-regime windows; write the report.

    ``oos_subdir`` picks which model's OOS + θ_85 gates drive the fires (the champion
    ``accuracy_push`` by default, or a challenger bake-off dir). ``with_dir10=False`` +
    ``with_momentum=False`` (``--no-benchmarks``) skips the Binance/dir10 reconstruction so the
    challenger money runs reuse the champion's already-computed benchmark numbers."""
    paths.ensure_out()
    effective = network_ms + venue_delay_ms
    out_dir = paths.out_dir / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    since_floor = _current_regime_since(since_ms, regime_from)
    res_map = hr.load_resolution_map(paths, series_keys)
    gates = _load_theta_gates(paths, series_keys, oos_subdir)

    burst_by_win: dict = {}
    for sk in series_keys:
        if sk not in gates:
            print(f"[burst_pair] WARN no gate for {sk} in out/{oos_subdir}; skipping", flush=True)
            continue
        burst_by_win.update(_load_burst_fires(paths, sk, gates[sk]["theta_gate"], since_floor,
                                              until_ms, oos_subdir))
    dir10_by_win = _load_dir10_fires(paths, series_keys, since_floor, until_ms) if with_dir10 else {}

    agg_real = {(cu, sk): dict(_AGG0) for cu in CONFIGS for sk in series_keys}
    agg_shuf = {(cu, sk): dict(_AGG0) for cu in CONFIGS for sk in series_keys}
    signals: dict = defaultdict(list)
    bench = {sk: {"momentum_net": 0.0, "momentum_trades": 0, "momentum_wins": 0,
                  "dir10_net": 0.0, "dir10_trades": 0, "dir10_wins": 0, "windows": 0}
             for sk in series_keys}
    done: set = set()

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _accumulate(data: dict) -> None:
        sk = data["series"]
        for cu_str, a in data["real"].items():
            _add(agg_real[(_ckey(cu_str), sk)], a)
        for cu_str, a in data["shuf"].items():
            _add(agg_shuf[(_ckey(cu_str), sk)], a)
        signals[sk].extend(data["signals_per_window"])
        for k in ("momentum_net", "momentum_trades", "momentum_wins", "dir10_net", "dir10_trades",
                  "dir10_wins"):
            bench[sk][k] += data.get(k, 0)
        bench[sk]["windows"] += int(data["windows"])

    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("series") in series_keys:
            _accumulate(data)
        done.add((data.get("series"), data.get("day")))

    for series_key in series_keys:
        if series_key not in gates:
            continue
        days = hlbl._days_for_series(paths, series_key, since_floor, until_ms)
        if max_days is not None:
            days = days[:max_days]
        for d in days:
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            day = _process_day(paths, series_key, d, day_str, burst_by_win, dir10_by_win, res_map,
                               excluded, latency_ms=effective, fee_rate=fee_rate,
                               window_budget=window_budget, with_momentum=with_momentum,
                               with_dir10=with_dir10, shuffle_seed=shuffle_seed)
            # serialize (config-tuple keys → strings) + write checkpoint atomically.
            ser = {**day, "real": {_cstr(cu): a for cu, a in day["real"].items()},
                   "shuf": {_cstr(cu): a for cu, a in day["shuf"].items()}}
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(ser, default=eh._json_default), encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            _accumulate(ser)
            print(f"[burst_pair] {series_key} {day_str}: {day['windows']} windows "
                  f"({len(done)} days done)", flush=True)

    return _finalize(paths, out_dir, series_keys, agg_real, agg_shuf, signals, bench, gates,
                     network_ms, venue_delay_ms, effective, window_budget, fee_rate, with_momentum,
                     with_dir10)


def _cstr(cu: tuple[float, float]) -> str:
    return f"C{cu[0]:.2f}_U{int(cu[1])}"


def _ckey(cu_str: str) -> tuple[float, float]:
    c, u = cu_str.split("_")
    return (float(c[1:]), float(u[1:]))


# ---- finalize --------------------------------------------------------------
def _rate(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


def _quantiles(vals: list[float]) -> dict:
    if not vals:
        return {"median": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    a = np.asarray(vals, float)
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()), "mean": float(a.mean())}


def _summary(a: dict) -> dict:
    split = a["locked_pnl"] + a["stranded_pnl"]
    resid = a["net"] - split
    shares_pair = a["shares_entered"] or 1.0
    return {
        "windows_traded": int(a["windows_traded"]), "lots": int(a["lots"]),
        "bursts_fired": int(a["bursts_fired"]), "fires": int(a["fires"]),
        "skip_cap": int(a["skip_cap"]), "skip_final": int(a["skip_final"]),
        "skip_nobook": int(a["skip_nobook"]), "skip_budget": int(a["skip_budget"]),
        "skip_nofill": int(a["skip_nofill"]),
        "net_pnl": float(a["net"]), "fees": float(a["fees"]),
        "locked_pairs_pnl": float(a["locked_pnl"]), "stranded_legs_pnl": float(a["stranded_pnl"]),
        "reconcile_residual": float(resid), "reconciles": bool(abs(resid) <= 0.01),
        "win_rate": _rate(a["wins"], a["lots"]),
        "shares_entered": float(a["shares_entered"]), "shares_locked": float(a["shares_locked"]),
        "shares_stranded": float(a["shares_stranded"]),
        "completion_rate": _rate(a["shares_locked"], shares_pair),
        "depth_shortfall_total": float(a["depth_shortfall_total"]),
        "avg_shortfall_per_burst": _rate(a["depth_shortfall_total"], a["bursts_fired"]),
        "shares_per_window": _quantiles(a.get("_shares_window", [])),
        "peak_unhedged_per_window": _quantiles(a.get("_peak_window", [])),
    }


def _finalize(paths, out_dir, series_keys, agg_real, agg_shuf, signals, bench, gates,
              network_ms, venue_delay_ms, effective, window_budget, fee_rate, with_momentum,
              with_dir10=True) -> dict:
    cells: dict = {}
    recon_ok = True
    for cu in CONFIGS:
        overall = dict(_AGG0)
        overall_shuf = dict(_AGG0)
        per_series = {}
        for sk in series_keys:
            _add(overall, agg_real[(cu, sk)])
            _add(overall_shuf, agg_shuf[(cu, sk)])
            per_series[sk] = _summary(agg_real[(cu, sk)])
        s = _summary(overall)
        ss = _summary(overall_shuf)
        recon_ok = recon_ok and s["reconciles"]
        cells[_cstr(cu)] = {"C": cu[0], "U": cu[1], **s,
                            "shuffled_net": ss["net_pnl"], "shuffled_locked": ss["locked_pairs_pnl"],
                            "shuffled_stranded": ss["stranded_legs_pnl"], "per_series": per_series}

    sig_all = [n for sk in series_keys for n in signals.get(sk, [])]
    momentum_net = sum(bench[sk]["momentum_net"] for sk in series_keys)
    dir10_net = sum(bench[sk]["dir10_net"] for sk in series_keys)
    n_windows = sum(bench[sk]["windows"] for sk in series_keys)

    # best real cell by net.
    best_cu = max(cells, key=lambda c: cells[c]["net_pnl"]) if cells else None
    best = cells.get(best_cu, {})

    result = {
        "params": {"title": "High-confidence burst pair-building backtest",
                   "series": list(series_keys), "network_ms": network_ms,
                   "venue_delay_ms": venue_delay_ms, "effective_latency_ms": effective,
                   "window_budget": window_budget, "fee_rate": fee_rate,
                   "burst_orders": BURST_ORDERS, "burst_size": BURST_SIZE,
                   "c_sweep": list(C_SWEEP), "u_sweep": list(U_SWEEP),
                   "no_entry_final_secs": NO_ENTRY_FINAL_SECS,
                   "theta_gates": {sk: gates.get(sk) for sk in series_keys},
                   "with_momentum": with_momentum},
        "caveats": CAVEATS,
        "current_regime_windows": n_windows,
        "signals_per_window": {**_quantiles([float(x) for x in sig_all]),
                               "total_signals": int(sum(sig_all)),
                               "windows_with_signal": int(sum(1 for x in sig_all if x > 0)),
                               "n_windows_universe": len(sig_all)},
        "benchmarks": {
            "never_trading_net": 0.0,
            "momentum_net": float(momentum_net),
            "momentum_trades": int(sum(bench[sk]["momentum_trades"] for sk in series_keys)),
            "momentum_ran": with_momentum,
            "momentum_reference_vps255_net": 5056.69,
            "model_taker_dir10_net": float(dir10_net),
            "model_taker_dir10_trades": int(sum(bench[sk]["dir10_trades"] for sk in series_keys)),
            "model_taker_dir10_ran": with_dir10,
            "model_taker_dir10_reference_net": 2284.01,
            "model_taker_theta": MODEL_TAKER_THETA,
            "maker_core_cache": "N/A (no Telonex maker-core backtest exists)",
            "per_series": {sk: bench[sk] for sk in series_keys},
        },
        "cells": cells,
        "best_cell": best_cu,
        "reconciliation": {"all_ok": bool(recon_ok), "tolerance": 0.01},
        "verdict": _verdict(cells, best_cu, best, momentum_net, dir10_net, sig_all, n_windows,
                            with_momentum, with_dir10),
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_outputs(out_dir, result)
    if not recon_ok:
        raise AssertionError("P&L split does NOT reconcile to net within $0.01")
    return result


def _verdict(cells, best_cu, best, momentum_net, dir10_net, sig_all, n_windows, with_momentum,
             with_dir10=True) -> str:
    if not cells:
        return "No cells — no qualifying signals (the model's θ_85 gates produced no fires)."
    net = best.get("net_pnl", 0.0)
    shuf = best.get("shuffled_net", 0.0)
    sig_mean = float(np.mean(sig_all)) if sig_all else 0.0
    sig_max = int(np.max(sig_all)) if sig_all else 0
    shares_p90 = best.get("shares_per_window", {}).get("p90", 0.0)
    shares_max = best.get("shares_per_window", {}).get("max", 0.0)
    comp = best.get("completion_rate", float("nan"))
    momo = f"${momentum_net:,.0f}" if with_momentum else "$5,057 (vps255 ref)"
    dir10s = f"${dir10_net:,.0f}" if with_dir10 else "$2,284 (vps255 ref)"
    momo_net = momentum_net if with_momentum else 5056.69  # compare against the real bar when not run
    # what stops accumulation?
    skip_cap = best.get("skip_cap", 0)
    skip_final = best.get("skip_final", 0)
    fired = best.get("bursts_fired", 1) or 1
    shortfall = best.get("avg_shortfall_per_burst", 0.0)
    stopper = ("signal scarcity" if sig_mean < 3 else
               ("the U cap" if skip_cap > fired else "displayed depth"))
    signfit = ("MAKES money" if net > 0 else "LOSES money")
    shuf_note = ("and the shuffled control ALSO profits — the 'edge' is the outcome-independent "
                 "locked-pair completion assumption, NOT the 85% signal (not signal-driven)"
                 if net > 0 and shuf > 0 else
                 ("and the shuffled control collapses — profit is signal-driven" if net > 0 else
                  f"(shuffled net ${shuf:,.0f})"))
    return (
        f"Best cell {best_cu}: net ${net:,.2f} ({signfit}; locked "
        f"${best.get('locked_pairs_pnl', 0):,.0f} + stranded ${best.get('stranded_legs_pnl', 0):,.0f}; "
        f"completion rate {comp:.0%}), {shuf_note}. Over {n_windows:,} current-regime windows the "
        f"θ_85 gate yields {sig_mean:.2f} qualifying signals/window (max {sig_max}); inventory built "
        f"per window p90 {shares_p90:.0f} sh / max {shares_max:.0f} sh — 500–1000 sh/window "
        f"{'IS' if shares_max >= 500 else 'is NOT'} reached; the binding constraint is {stopper} "
        f"(avg depth shortfall {shortfall:.0f} sh/burst). Benchmarks on identical windows: momentum "
        f"{momo}, model-taker (dir10) {dir10s}, never-trade $0. "
        f"{'Burst pair-building does NOT beat momentum.' if net <= momo_net else 'Burst pair-building BEATS momentum.'}"
    )


# ---- outputs ---------------------------------------------------------------
def _write_outputs(out_dir: Path, result: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default),
                                          encoding="utf-8")
    # per-cell CSV.
    cell_rows = []
    for cs, c in result["cells"].items():
        cell_rows.append({"cell": cs, "C": c["C"], "U": c["U"], "net_pnl": c["net_pnl"],
                          "locked": c["locked_pairs_pnl"], "stranded": c["stranded_legs_pnl"],
                          "fees": c["fees"], "shuffled_net": c["shuffled_net"],
                          "completion_rate": c["completion_rate"], "bursts": c["bursts_fired"],
                          "shares_entered": c["shares_entered"],
                          "shares_pw_p90": c["shares_per_window"]["p90"],
                          "shares_pw_max": c["shares_per_window"]["max"],
                          "skip_cap": c["skip_cap"], "skip_final": c["skip_final"]})
    pd.DataFrame(cell_rows).to_csv(out_dir / "cells.csv", index=False)

    # per-series (best cell) CSV.
    best = result.get("best_cell")
    ps_rows = []
    if best and best in result["cells"]:
        for sk, s in result["cells"][best]["per_series"].items():
            ps_rows.append({"series": sk, "cell": best, "net_pnl": s["net_pnl"],
                            "locked": s["locked_pairs_pnl"], "stranded": s["stranded_legs_pnl"],
                            "fees": s["fees"], "completion_rate": s["completion_rate"],
                            "bursts": s["bursts_fired"], "shares_entered": s["shares_entered"]})
    pd.DataFrame(ps_rows).to_csv(out_dir / "per_series_best.csv", index=False)

    _write_html(out_dir, result)


def _write_html(out_dir: Path, result: dict) -> None:
    p = result["params"]
    b = result["benchmarks"]
    sig = result["signals_per_window"]
    cell_cols = [("cell", "C_U"), ("net_pnl", "net $"), ("locked_pairs_pnl", "locked $"),
                 ("stranded_legs_pnl", "stranded $"), ("shuffled_net", "shuffled $"),
                 ("completion_rate", "comp rate"), ("bursts_fired", "bursts"),
                 ("shares_entered", "shares"), ("skip_cap", "skip(cap)")]
    cell_rows = [{**result["cells"][cs]} for cs in result["cells"]]
    for r in cell_rows:
        r["cell"] = f"C{r['C']:.2f}/U{int(r['U'])}"
        r["completion_rate"] = f"{r['completion_rate']:.0%}"
    caveats = "".join(f"<li>{c}</li>" for c in result["caveats"])
    recon = result["reconciliation"]
    bug = "" if recon["all_ok"] else " bug"
    momo = (f"${b['momentum_net']:,.2f} ({b['momentum_trades']:,} trades)" if b["momentum_ran"]
            else f"N/A this run — vps255 reference ${b['momentum_reference_vps255_net']:,.2f}")
    dir10s = (f"${b['model_taker_dir10_net']:,.2f} ({b['model_taker_dir10_trades']:,} trades)"
              if b.get("model_taker_dir10_ran", True)
              else f"N/A this run — vps255 reference ${b.get('model_taker_dir10_reference_net', 2284.01):,.2f}")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{p['title']}</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.5rem}} h2{{margin-top:2rem;border-bottom:1px solid #eee}}
table{{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}} td,th{{padding:4px 12px;text-align:left;border-bottom:1px solid #f0f0f0}}
th{{color:#555;font-weight:600}} .muted{{color:#888}}
.verdict{{background:#f6f8fa;border-left:4px solid #1f77b4;padding:.8rem 1rem;border-radius:4px;line-height:1.5}}
.verdict.bug{{border-left-color:#d62728;background:#fdeeee}}</style></head><body>
<h1>{p['title']}</h1>
<p class="muted">Burst = {p['burst_orders']}×{int(p['burst_size'])}-share FAK on the θ_85-gated 15 s
signal; completion maker at C−price_s (pair cost ≤ C); unhedged capped at U; no force-hedge; no entry
in the final {p['no_entry_final_secs']} s. Effective latency {p['effective_latency_ms']} ms;
current-regime windows (≥ 2026-06-05); marked to the official resolution.</p>
<div class="verdict{bug}">{result['verdict']}</div>
<h2>Signals per window (θ_85 gate)</h2>
<p>mean {sig['mean']:.2f} · median {sig['median']:.0f} · p90 {sig['p90']:.0f} · max {sig['max']:.0f}
· {sig['windows_with_signal']:,}/{sig['n_windows_universe']:,} windows have ≥1 signal
· {sig['total_signals']:,} total signals.</p>
<h2>Benchmarks (identical windows)</h2>
<p>never-trade <b>$0.00</b> · momentum <b>{momo}</b> · model-taker dir10 (θ={b['model_taker_theta']})
<b>{dir10s}</b> · maker-core <b>{b['maker_core_cache']}</b>.</p>
<h2>C × U cells (overall)</h2>
{eh._table_html(cell_rows, cell_cols)}
<h2>Caveats (read these)</h2><ul>{caveats}</ul>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "backtest_burst_pair")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    parser.add_argument("--max-days", type=int, default=None, help="cap UTC days per series (slice)")
    parser.add_argument("--network-ms", type=int, default=5)
    parser.add_argument("--venue-delay-ms", type=int, default=250)
    parser.add_argument("--window-budget", type=float, default=WINDOW_BUDGET)
    parser.add_argument("--fee-rate", type=float, default=mj.CRYPTO_FEE_RATE)
    parser.add_argument("--no-momentum", action="store_true", help="skip the momentum benchmark (faster)")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="skip BOTH momentum and dir10 benchmarks (challenger bake-off — reuse "
                             "the champion's reference numbers; skips the Binance/dir10 reconstruction)")
    parser.add_argument("--oos-dir", default="accuracy_push",
                        help="OOS subdir under out/ providing oos_<market>.parquet + metrics.json θ "
                             "gates (default the champion accuracy_push)")
    parser.add_argument("--shuffle-seed", type=int, default=20260713)
    parser.add_argument("--out-name", default="burst_pair", help="output subdir under out/backtests/")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    with_momentum = not (args.no_momentum or args.no_benchmarks)
    with_dir10 = not args.no_benchmarks

    print(f"[burst_pair] out={paths.out_dir / 'backtests' / args.out_name} oos={args.oos_dir} "
          f"latency=net {args.network_ms}+venue {args.venue_delay_ms}={args.network_ms + args.venue_delay_ms}ms "
          f"momentum={with_momentum} dir10={with_dir10} configs={len(CONFIGS)}", flush=True)
    result = backtest_burst_pair(paths, series=series, since_ms=since_ms, until_ms=until_ms,
                                 max_days=args.max_days, network_ms=args.network_ms,
                                 venue_delay_ms=args.venue_delay_ms, window_budget=args.window_budget,
                                 fee_rate=args.fee_rate, with_momentum=with_momentum,
                                 with_dir10=with_dir10, oos_subdir=args.oos_dir,
                                 shuffle_seed=args.shuffle_seed, out_name=args.out_name)
    print(f"[burst_pair] {result['current_regime_windows']:,} windows; reconcile_ok="
          f"{result['reconciliation']['all_ok']}", flush=True)
    print(f"[burst_pair] VERDICT: {result['verdict']}", flush=True)
    procmem.report_peak_rss("burst_pair", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[burst_pair] wrote metrics.json + cells.csv + per_series_best.csv + report.html", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
