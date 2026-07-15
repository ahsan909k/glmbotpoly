"""Stage — backtest_pair_lean: the operator's **pair-lean** strategy over the full history.

Per window, when the walk-forward **dir10** model predicts direction with confidence
``|p_up − 0.5| ≥ T``, **take the predicted side at the real ask** (255 ms effective delay +
the exact taker fee + displayed depth, sized to a fraction of displayed depth), then **rest a
maker quote on the other side** at limit ``L = C − price_s`` so the combined pair cost ≤ ``C``.
A completed pair collects $1 at resolution (locked profit ``= 1 − C`` per pair, minus the taker
fee on that pair); a leg that never completes before the window closes **rides to resolution**.

This differs from ``money_judge._run_pairleg`` (which completes the opposite leg as a *taker*
cross paying a taker fee whenever ``price_s + opposite_ask < 1``): here the completion is a
**resting maker** at limit ``L`` — **zero fee**, and it fills only when the recorded opposite ask
reaches ≤ ``L`` within the window (the market comes to us). Rarer completions, but cheaper, with
an explicit locked margin ``1 − C``.

**No retraining.** The fires are the already-written walk-forward out-of-sample predictions
``out/learn_walkforward/oos_dir10_full.parquet`` (``p_up`` = P(10 s-forward mid moves Up)) — the
model whose taker replay is the $2,284 "model-taker" reference. This stage is **LightGBM-free**
(it reuses ``money_judge``'s fill primitives + the ``backtest_momentum`` reconstruction; it never
imports the trainer).

Sweeps ``T`` (``|p−0.5| ≥ θ`` over the standard grid), ``C`` ∈ {0.90, 0.94, 0.96, 0.98}, and
lean size (fraction of displayed ask depth ∈ {0.25, 0.5, 1.0}). Fires multi-time per window until
the $10/window budget is spent. Judged on **money after fees over current-regime windows**
(≥ 2026-06-05, 255 ms effective) **per series**, vs never-trade, a matched-frequency random-
trigger control + a shuffled-labels control, and vs the momentum ($5,057) and model-taker
($2,284) baselines replayed on the identical windows. Output ``out/backtests/pair_lean/``.
Resumable per-(series, day). **Research only.**

Run::

    python -m model_lab.backtest_pair_lean                     # full current regime, 4 series
    python -m model_lab.backtest_pair_lean --series BTC-5m --seeds 4
"""

from __future__ import annotations

import base64
import io
import json
import math
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
DEFAULT_SEED = 20260711
DEFAULT_SEEDS = 5  # matched-frequency random-trigger control seeds (the dominant runtime cost)
CURRENT_REGIME_FROM = date(2026, 6, 5)  # 250 ms taker-delay lock (out/audits/venue_regime_timeline.md)

# --- sweep grid (operator-confirmed) ----------------------------------------
DEFAULT_THETAS = list(mj.SWEEP_THRESHOLDS)  # |p−0.5| ≥ θ (same grid as the model-taker sweep)
DEFAULT_PAIR_COSTS = [0.90, 0.94, 0.96, 0.98]  # combined pair-cost ceiling C (locked margin 1−C)
DEFAULT_DEPTH_FRACS = [0.25, 0.5, 1.0]  # lean size = fraction of displayed ask depth

CAVEATS = mj.CAVEATS + [
    "The maker completion fills at our limit L = C − price_s the first time the recorded "
    "OPPOSITE ask reaches ≤ L within the window (Telonex has no PM trade tape); the natural "
    "but mildly optimistic maker-fill proxy — the real §9 model queues behind displayed size "
    "and fills on prints through the price.",
    "Maker completion pays ZERO fee and NO rebate is credited (conservative); only the leaned "
    "taker leg pays the exact taker fee.",
    "The per-window $10 budget is charged on the leaned taker leg only (the maker completion is "
    "a hedge that recycles into a $1 locked pair), so the fire set depends only on (θ, depth_frac).",
    "L = C − price_s is used in float book space (no tick-grid snap); locked profit per pair = 1 − C.",
]


# --- current-regime helpers (local; learn_walkforward pulls in lightgbm) -----
def _regime_floor_ms(regime_from: date) -> int:
    return int(datetime(regime_from.year, regime_from.month, regime_from.day,
                        tzinfo=timezone.utc).timestamp()) * 1000


def _current_regime_since(since_ms: int | None, regime_from: date) -> int:
    """The effective floor = max(the 250 ms-delay regime floor, a caller ``--since``)."""
    floor = _regime_floor_ms(regime_from)
    return max(floor, since_ms) if since_ms is not None else floor


def _group_current_regime(oos: pd.DataFrame, since_ms: int) -> dict:
    """{(series, window_open_ms): slice} over current-regime windows only (bounds the per-window
    lookup; avoids an O(windows × rows) scan)."""
    cr = oos[oos["window_open_ms"].to_numpy(dtype="int64") >= since_ms]
    return {k: v for k, v in cr.groupby(["series", "window_open_ms"], sort=False)}


def _fires_from_group(g: pd.DataFrame | None, series_key: str, open_ms: int, up_token: str,
                      outcome_up: int, close_ms: int) -> pd.DataFrame | None:
    """The fires frame the sim consumes, for one window, from a pre-grouped OOS slice."""
    if g is None or not len(g):
        return None
    return pd.DataFrame({
        "series": series_key, "window_open_ms": int(open_ms),
        "sample_ts_ms": g["sample_ts_ms"].to_numpy(dtype="int64"),
        "p_up": g["p_up"].to_numpy(dtype=float), "up_token": up_token,
        "outcome_up": int(outcome_up), "close_time": int(close_ms),
    })


# --- pair-lean sim ----------------------------------------------------------
def _lean_shares(price: float, avail: float, budget_left: float, depth_frac: float) -> float:
    """Leaned taker size = min(a fraction of displayed ask depth, the per-window budget). Below
    the $1 min notional → 0 (skip)."""
    shares = min(depth_frac * avail, budget_left / price)
    if shares <= 0.0 or shares * price < mj.MIN_NOTIONAL:
        return 0.0
    return shares


def _maker_completion(book: dict, side_s: str, limit: float, lo_ms: int, hi_ms: int):
    """The first recorded book snapshot in ``[lo_ms, hi_ms)`` where the OPPOSITE-leg ask reaches
    ``≤ limit`` — i.e. the market comes to our resting maker bid at ``limit``. Returns the
    displayed opposite size there (the fill happens at our maker ``limit``), or ``None``.

    Leaned Up → complete Down (Down ask = ``1 − Up bid``, size = Up bid size); leaned Down →
    complete Up (Up ask, size = Up ask size). Vectorized first-crossing."""
    ts = book["ts"]
    lo = int(np.searchsorted(ts, lo_ms, side="left"))
    hi = int(np.searchsorted(ts, hi_ms, side="right"))
    if hi <= lo:
        return None
    if side_s == "Up":
        opp_ask = 1.0 - book["bid_px"][lo:hi]
        opp_sz = book["bid_sz"][lo:hi]
    else:  # complete Up
        opp_ask = book["ask_px"][lo:hi]
        opp_sz = book["ask_sz"][lo:hi]
    mask = (np.isfinite(opp_ask) & (opp_ask > 0.0) & (opp_ask < 1.0)
            & (opp_ask <= limit) & (opp_sz > 0.0))
    if not mask.any():
        return None
    return float(opp_sz[int(mask.argmax())])


def _run_pair_lean(fires: pd.DataFrame, *, threshold: float, pair_costs, depth_frac: float,
                   latency_ms: int, fee_rate: float, window_budget: float, books: dict):
    """Multi-fire pair-lean over one window's fires. Per qualifying snapshot: a taker lean, then a
    resting maker completion at ``L = C − price_s`` (zero fee) for each pair-cost ``C``. Budget
    charged on the leaned leg only. Returns ``({pair_cost: [rows]}, n_fires)``."""
    rows_by_c: dict[float, list[dict]] = {c: [] for c in pair_costs}
    n_fires = 0
    for (series, open_ms), g in fires.groupby(["series", "window_open_ms"], sort=False):
        budget_left = window_budget
        for r in g.sort_values("sample_ts_ms").itertuples(index=False):
            side = mj._side_of(float(r.p_up), threshold)
            if side is None:
                continue
            n_fires += 1
            if budget_left < mj.MIN_NOTIONAL:
                continue
            fire_ts = int(r.sample_ts_ms)
            book_row = mj._book_asof(books, r.up_token, fire_ts + latency_ms)
            if book_row is None:
                continue
            quote = mj._leaned_quote(book_row, side)
            if quote is None:
                continue
            price_s, avail = quote
            n = _lean_shares(price_s, avail, budget_left, depth_frac)
            if n <= 0.0:
                continue
            budget_left -= n * price_s
            fee_s = lm.taker_fee(n, fee_rate, price_s)
            outcome_up = int(r.outcome_up)
            close_ms = int(r.close_time)
            s_wins = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
            book = books.get(r.up_token)
            lo_ms = fire_ts + latency_ms
            for c in pair_costs:
                limit = c - price_s
                m, price_maker = 0.0, float("nan")
                if limit > 0.0 and book is not None and close_ms > lo_ms:
                    size_o = _maker_completion(book, side, limit, lo_ms, close_ms)
                    if size_o is not None:
                        m, price_maker = min(n, size_o), limit
                stranded = n - m
                cost = n * price_s + fee_s + (m * price_maker if m > 0.0 else 0.0)
                payoff = m * 1.0 + stranded * (1.0 if s_wins else 0.0)
                net = payoff - cost
                fee_s_locked = fee_s * (m / n) if n > 0 else 0.0
                locked_pnl = (m * 1.0 - (m * price_s + m * price_maker + fee_s_locked)) if m > 0.0 else 0.0
                rows_by_c[c].append({
                    "series": series, "window_open_ms": int(open_ms), "sample_ts_ms": fire_ts,
                    "day": int(open_ms) // MS_PER_DAY, "side": side, "n": n, "locked": m,
                    "stranded": stranded, "price_s": price_s, "price_maker": price_maker,
                    "fee": fee_s, "cost": cost, "payoff": payoff, "net": net,
                    "locked_pnl": locked_pnl, "stranded_pnl": net - locked_pnl,
                    "completed": bool(m > 0.0), "won": bool(net > 0.0), "outcome_up": outcome_up,
                })
    return rows_by_c, n_fires


def _run_pair_lean_random(fires: pd.DataFrame, *, fire_prob: float, pair_costs, depth_frac: float,
                          rng: np.random.Generator, latency_ms: int, fee_rate: float,
                          window_budget: float, books: dict):
    """The matched-frequency random-trigger control: fire at random OOS snapshots at ``fire_prob``
    (matched to the real per-θ fire rate), side = the model's lean, everything else identical to
    the real pair-lean mechanics. Net-only. Returns ``{pair_cost: (net, trades)}``."""
    net = {c: 0.0 for c in pair_costs}
    trades = {c: 0 for c in pair_costs}
    for (_series, _open_ms), g in fires.groupby(["series", "window_open_ms"], sort=False):
        budget_left = window_budget
        for r in g.sort_values("sample_ts_ms").itertuples(index=False):
            p_up = float(r.p_up)
            if not math.isfinite(p_up) or rng.random() >= fire_prob:
                continue
            side = "Up" if p_up >= 0.5 else "Down"
            if budget_left < mj.MIN_NOTIONAL:
                continue
            fire_ts = int(r.sample_ts_ms)
            book_row = mj._book_asof(books, r.up_token, fire_ts + latency_ms)
            if book_row is None:
                continue
            quote = mj._leaned_quote(book_row, side)
            if quote is None:
                continue
            price_s, avail = quote
            n = _lean_shares(price_s, avail, budget_left, depth_frac)
            if n <= 0.0:
                continue
            budget_left -= n * price_s
            fee_s = lm.taker_fee(n, fee_rate, price_s)
            outcome_up = int(r.outcome_up)
            close_ms = int(r.close_time)
            s_wins = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
            book = books.get(r.up_token)
            lo_ms = fire_ts + latency_ms
            for c in pair_costs:
                limit = c - price_s
                m, price_maker = 0.0, float("nan")
                if limit > 0.0 and book is not None and close_ms > lo_ms:
                    size_o = _maker_completion(book, side, limit, lo_ms, close_ms)
                    if size_o is not None:
                        m, price_maker = min(n, size_o), limit
                stranded = n - m
                cost = n * price_s + fee_s + (m * price_maker if m > 0.0 else 0.0)
                payoff = m * 1.0 + stranded * (1.0 if s_wins else 0.0)
                net[c] += payoff - cost
                trades[c] += 1
    return net, trades


# --- aggregation ------------------------------------------------------------
_AGG0 = {"n_trades": 0, "net": 0.0, "fees": 0.0, "wins": 0, "n_completed": 0,
         "shares_n": 0.0, "shares_locked": 0.0, "notional": 0.0, "gross": 0.0,
         "locked_pnl": 0.0, "stranded_pnl": 0.0}


def _agg_rows(rows: list[dict]) -> dict:
    """Additive per-config aggregate of a set of pair-lean rows (no rows retained)."""
    a = dict(_AGG0)
    for r in rows:
        a["n_trades"] += 1
        a["net"] += r["net"]
        a["fees"] += r["fee"]
        a["wins"] += 1 if r["won"] else 0
        a["n_completed"] += 1 if r["completed"] else 0
        a["shares_n"] += r["n"]
        a["shares_locked"] += r["locked"]
        a["notional"] += r["n"] * r["price_s"]
        a["gross"] += r["payoff"] - r["n"] * r["price_s"]
        a["locked_pnl"] += r["locked_pnl"]
        a["stranded_pnl"] += r["stranded_pnl"]
    return a


def _agg_taker(rows: list[dict]) -> dict:
    """Additive aggregate of taker/momentum rows (mj._run_taker / _run_momentum shape)."""
    a = {"n_trades": 0, "net": 0.0, "fees": 0.0, "wins": 0, "shares": 0.0, "notional": 0.0}
    for r in rows:
        a["n_trades"] += 1
        a["net"] += r["net"]
        a["fees"] += r["fee"]
        a["wins"] += 1 if r["won"] else 0
        a["shares"] += r["shares"]
        a["notional"] += r["shares"] * r["price"]
    return a


def _add(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v


def _shuffled_net(rows: list[dict], series_key: str, day_str: str, seed: int) -> float:
    """Net PnL of a config's day rows under a per-(series, day) outcome permutation. A locked
    pair pays $1 regardless of outcome, so shuffling only re-marks the stranded legs; recomputed
    from the rows (no re-simulation). Deterministic via a crc32 salt."""
    if not rows:
        return 0.0
    keys = sorted({r["window_open_ms"] for r in rows})
    true_out = {}
    for r in rows:
        true_out.setdefault(r["window_open_ms"], r["outcome_up"])
    outs = np.array([true_out[k] for k in keys])
    salt = zlib.crc32(f"shuffle|{series_key}|{day_str}".encode()) % 1_000_000
    perm = np.random.default_rng(seed + salt).permutation(len(outs))
    perm_out = {k: int(o) for k, o in zip(keys, outs[perm])}
    net = 0.0
    for r in rows:
        o = perm_out[r["window_open_ms"]]
        s_wins = (r["side"] == "Up" and o == 1) or (r["side"] == "Down" and o == 0)
        payoff = r["locked"] * 1.0 + r["stranded"] * (1.0 if s_wins else 0.0)
        net += payoff - r["cost"]
    return float(net)


def _ckey(theta: float, pc: float, frac: float) -> str:
    return f"{theta:g}|{pc:g}|{frac:g}"


# --- worker -----------------------------------------------------------------
def backtest_pair_lean(
    paths: Paths, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None,
    thetas=None, pair_costs=None, depth_fracs=None,
    seeds: int = DEFAULT_SEEDS, seed: int = DEFAULT_SEED,
    network_ms: int = 5, venue_delay_ms: int = 250,
    window_budget: float = mj.DEFAULT_WINDOW_BUDGET, fee_rate: float = mj.CRYPTO_FEE_RATE,
    target: str = "dir10", variant: str = "full", oos: pd.DataFrame | None = None,
    res_map: dict | None = None, regime_from: date = CURRENT_REGIME_FROM,
    out_name: str = "pair_lean",
) -> dict:
    """Replay the pair-lean sweep + baselines + controls over the current-regime windows; write the
    report. Returns the metrics dict. Effective latency = ``network_ms + venue_delay_ms`` (default
    255 ms, matching the vps255 reference that produced the $5,057 / $2,284 numbers).

    ``oos`` (the dir10 OOS predictions) and ``res_map`` (official resolutions) are loaded from disk
    when ``None``; both are injectable for offline tests. ``regime_from`` overrides the current-
    regime floor (tests point it at the fixture day)."""
    paths.ensure_out()
    thetas = list(thetas if thetas is not None else DEFAULT_THETAS)
    pair_costs = list(pair_costs if pair_costs is not None else DEFAULT_PAIR_COSTS)
    depth_fracs = list(depth_fracs if depth_fracs is not None else DEFAULT_DEPTH_FRACS)
    effective = network_ms + venue_delay_ms
    out_dir = paths.out_dir / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    if oos is None:
        path = paths.out_dir / "learn_walkforward" / f"oos_{target}_{variant}.parquet"
        assert_parquet_ready(path, label=f"oos_{target}_{variant}.parquet", min_rows=1)
        # Load only this run's series (predicate pushdown) so per-series parallel invocations —
        # sharing one --out-name checkpoint dir — each hold a slice, not the whole 10M-row frame.
        oos = pd.read_parquet(path, columns=["series", "window_open_ms", "sample_ts_ms", "p_up"],
                              filters=[("series", "in", list(series_keys))])
    if res_map is None:
        res_map = hr.load_resolution_map(paths, series_keys)
    strike_tol_ms = ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0
    since_floor = _current_regime_since(since_ms, regime_from)
    groups = _group_current_regime(oos, since_floor)
    configs = [(t, c, f) for t in thetas for c in pair_costs for f in depth_fracs]

    # accumulators (per series; overall = sum over series). Memory-bounded: aggregates only.
    real: dict = defaultdict(lambda: dict(_AGG0))           # (series, theta, pc, frac) -> agg
    nfires: dict = defaultdict(int)                          # (series, theta) -> fires
    taker: dict = defaultdict(lambda: {"n_trades": 0, "net": 0.0, "fees": 0.0, "wins": 0,
                                       "shares": 0.0, "notional": 0.0})  # (series, theta)
    momentum: dict = defaultdict(lambda: {"n_trades": 0, "net": 0.0, "fees": 0.0, "wins": 0,
                                          "shares": 0.0, "notional": 0.0})  # series
    control_net: dict = defaultdict(lambda: np.zeros(seeds))   # (series, theta, pc, frac)
    control_tr: dict = defaultdict(lambda: np.zeros(seeds))
    shuf_net: dict = defaultdict(lambda: np.zeros(seeds))
    n_windows = 0
    done: set[tuple[str, str]] = set()

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _accumulate(data: dict) -> None:
        nonlocal n_windows
        sk = data["series"]
        for ck, agg in data["real"].items():
            t, c, f = (float(x) for x in ck.split("|"))
            _add(real[(sk, t, c, f)], agg)
        for tk, nf in data["nfires"].items():
            nfires[(sk, float(tk))] += int(nf)
        for tk, tagg in data["taker"].items():
            _add(taker[(sk, float(tk))], tagg)
        _add(momentum[sk], data["momentum"])
        for ck, vec in data["control_net"].items():
            t, c, f = (float(x) for x in ck.split("|"))
            control_net[(sk, t, c, f)] += np.array(vec, dtype=float)
            control_tr[(sk, t, c, f)] += np.array(data["control_trades"][ck], dtype=float)
        for ck, vec in data["shuf_net"].items():
            t, c, f = (float(x) for x in ck.split("|"))
            shuf_net[(sk, t, c, f)] += np.array(vec, dtype=float)
        n_windows += int(data["windows"])

    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(data.get("seeds", -1)) != seeds:  # a seed change invalidates control/shuffled sums
            continue
        try:
            _accumulate(data)
            done.add((data["series"], data["day"]))
        except (KeyError, ValueError):
            continue

    for series_key in series_keys:
        prefix = hc.series_prefix(series_key)
        asset = hc.SLUG_PREFIXES[prefix][1]
        symbol = hc.SYMBOL_BY_ASSET[asset]
        for d in hlbl._days_for_series(paths, series_key, since_floor, until_ms):
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            day = _process_day(paths, series_key, asset, symbol, d, day_str, groups, res_map,
                                excluded, strike_tol_ms, configs, thetas, pair_costs, depth_fracs,
                                effective, fee_rate, window_budget, seeds, seed)
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"seeds": seeds, **day}, default=eh._json_default),
                           encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            _accumulate({"seeds": seeds, **day})
            print(f"[backtest_pair_lean] {series_key} {day_str}: {day['windows']} windows "
                  f"({len(done)} days done, {n_windows:,} cumulative)", flush=True)

    return _finalize(paths, out_dir, series_keys, thetas, pair_costs, depth_fracs, configs,
                     real, nfires, taker, momentum, control_net, control_tr, shuf_net,
                     n_windows, seeds, seed, network_ms, venue_delay_ms, effective,
                     window_budget, fee_rate, target, variant)


def _process_day(paths, series_key, asset, symbol, d, day_str, groups, res_map, excluded,
                 strike_tol_ms, configs, thetas, pair_costs, depth_fracs, effective, fee_rate,
                 window_budget, seeds, seed) -> dict:
    """Reconstruct one (series, day)'s windows once, run the pair-lean sweep + baselines +
    controls, and return the day's AGGREGATES (no rows retained across days)."""
    wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
    grid, ring_ts, ring_px = (bm._day_grid_and_ring(paths, symbol, asset, d) if wins
                              else (pd.DataFrame(), None, None))
    fires_by_win: list[pd.DataFrame] = []
    books: dict[str, dict] = {}
    win_by_key: dict[tuple, dict] = {}
    model_rows: list[dict] = []
    day_p: list[np.ndarray] = []
    n_win = 0
    if wins and not grid.empty:
        mids_by_asset = {asset: (ring_ts, ring_px)}
        bars = bm._grid_bars(grid)
        for w in wins:
            open_ms, close_ms = int(w["window_open_ms"]), int(w["window_close_ms"])
            official = res_map.get((series_key, open_ms))
            if official is None:
                continue
            st = hc.binance_anchored_strike(bars, open_ms, strike_tol_ms)
            if not (st["strike"] > 0.0):
                continue
            upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
            if upq.empty:
                continue
            up_token = str(upq["token_id"].iloc[0])
            outcome_up = 1 if official["outcome"] == "Up" else 0
            # Every reconstructed window counts and feeds the momentum baseline (matching the
            # reference); the pair-lean sweep + model-taker trade only windows with an OOS
            # prediction (fires may be None). This keeps n_windows / momentum identical to the
            # learn_walkforward vps255 reference over the same 24,697 windows.
            books[up_token] = tpm.pm_book_arrays(upq)
            win_by_key[(series_key, open_ms)] = {"up_token": up_token, "close_time": close_ms,
                                                 "outcome_up": outcome_up}
            model_rows.extend(bm._window_model_rows(grid, w, st))
            n_win += 1
            fires = _fires_from_group(groups.get((series_key, open_ms)), series_key, open_ms,
                                      up_token, outcome_up, close_ms)
            if fires is not None:
                fires_by_win.append(fires)
                day_p.append(fires["p_up"].to_numpy(dtype=float))

    real: dict = {_ckey(t, c, f): dict(_AGG0) for (t, c, f) in configs}
    day_nfires: dict = {f"{t:g}": 0 for t in thetas}
    day_taker: dict = {f"{t:g}": _agg_taker([]) for t in thetas}
    control_net: dict = {_ckey(t, c, f): np.zeros(seeds) for (t, c, f) in configs}
    control_tr: dict = {_ckey(t, c, f): np.zeros(seeds) for (t, c, f) in configs}
    shuf_net: dict = {_ckey(t, c, f): np.zeros(seeds) for (t, c, f) in configs}

    if fires_by_win:
        all_fires = pd.concat(fires_by_win, ignore_index=True)
        p_all = np.concatenate(day_p) if day_p else np.empty(0)
        # real sweep + shuffled-labels (post-hoc from the day's rows)
        for t in thetas:
            for f in depth_fracs:
                rows_by_c, nfr = _run_pair_lean(all_fires, threshold=t, pair_costs=pair_costs,
                                                depth_frac=f, latency_ms=effective, fee_rate=fee_rate,
                                                window_budget=window_budget, books=books)
                day_nfires[f"{t:g}"] = int(nfr)  # fires depend only on θ
                for c in pair_costs:
                    rows = rows_by_c[c]
                    _add(real[_ckey(t, c, f)], _agg_rows(rows))
                    if rows:
                        vec = shuf_net[_ckey(t, c, f)]
                        for si in range(seeds):
                            vec[si] += _shuffled_net(rows, series_key, day_str, seed + si * 101)
        # model-taker baseline per θ (same OOS fires)
        for t in thetas:
            tk, _ = mj._run_taker(all_fires, threshold=t, latency_ms=effective, fee_rate=fee_rate,
                                  window_budget=window_budget, trade_budget=window_budget, books=books)
            day_taker[f"{t:g}"] = _agg_taker(tk)
        # matched-frequency random-trigger control (net-only, per (θ, frac), seeded)
        for t in thetas:
            rate = float(np.mean(np.abs(p_all - 0.5) >= t)) if p_all.size else 0.0
            salt = zlib.crc32(f"{series_key}|{day_str}|{t:g}".encode()) % 1_000_000
            for f in depth_fracs:
                for si in range(seeds):
                    rng = np.random.default_rng(seed + si * 7919 + salt)
                    cnet, ctr = _run_pair_lean_random(all_fires, fire_prob=rate, pair_costs=pair_costs,
                                                      depth_frac=f, rng=rng, latency_ms=effective,
                                                      fee_rate=fee_rate, window_budget=window_budget,
                                                      books=books)
                    for c in pair_costs:
                        control_net[_ckey(t, c, f)][si] += cnet[c]
                        control_tr[_ckey(t, c, f)][si] += ctr[c]

    mom = _agg_taker(mj._run_momentum(model_rows, {asset: (ring_ts, ring_px)}, win_by_key, books,
                                      latency_ms=effective, fee_rate=fee_rate,
                                      window_budget=window_budget)) if model_rows else _agg_taker([])

    return {"series": series_key, "day": day_str, "windows": n_win, "real": real,
            "nfires": day_nfires, "taker": day_taker, "momentum": mom,
            "control_net": {k: v.tolist() for k, v in control_net.items()},
            "control_trades": {k: v.tolist() for k, v in control_tr.items()},
            "shuf_net": {k: v.tolist() for k, v in shuf_net.items()}}


# --- finalize ---------------------------------------------------------------
def _rate(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


def _config_row(agg: dict, nfires: int) -> dict:
    n = agg["n_trades"]
    return {
        "n_fires": int(nfires), "trades": int(n), "net_pnl": float(agg["net"]),
        "fees": float(agg["fees"]), "gross_pnl": float(agg["gross"]),
        "win_rate": _rate(agg["wins"], n), "net_per_trade": _rate(agg["net"], n),
        "completion_rate": _rate(agg["n_completed"], n),
        "shares_locked_frac": _rate(agg["shares_locked"], agg["shares_n"]),
        "locked_pnl": float(agg["locked_pnl"]), "stranded_pnl": float(agg["stranded_pnl"]),
    }


def _best_config(net_by_config: dict) -> tuple | None:
    """argmax net; tie → higher θ, then higher C, then larger lean size."""
    items = [(k, v) for k, v in net_by_config.items() if v["trades"] >= 1]
    if not items:
        return None
    return max(items, key=lambda kv: (kv[1]["net_pnl"], kv[0][0], kv[0][1], kv[0][2]))[0]


def _control_stats(vec: np.ndarray, real_net: float, seeds: int) -> dict:
    if not seeds:
        return {"mean": float("nan"), "std": float("nan"), "p_value": float("nan"), "beats": False}
    mean, std = float(np.mean(vec)), float(np.std(vec))
    p = float(np.mean(vec >= real_net))
    return {"mean": mean, "std": std, "p_value": p,
            "beats": bool(real_net > mean and p <= 0.1), "nets": vec.tolist()}


def _finalize(paths, out_dir, series_keys, thetas, pair_costs, depth_fracs, configs,
              real, nfires, taker, momentum, control_net, control_tr, shuf_net,
              n_windows, seeds, seed, network_ms, venue_delay_ms, effective,
              window_budget, fee_rate, target, variant) -> dict:
    # overall (sum over series) per config
    overall: dict = {cfg: dict(_AGG0) for cfg in configs}
    overall_nf: dict = defaultdict(int)
    overall_ctrl: dict = {cfg: np.zeros(seeds) for cfg in configs}
    overall_shuf: dict = {cfg: np.zeros(seeds) for cfg in configs}
    for sk in series_keys:
        for (t, c, f) in configs:
            _add(overall[(t, c, f)], real[(sk, t, c, f)])
            overall_ctrl[(t, c, f)] += control_net[(sk, t, c, f)]
            overall_shuf[(t, c, f)] += shuf_net[(sk, t, c, f)]
        for t in thetas:
            overall_nf[t] += nfires[(sk, t)]

    over_rows = {cfg: _config_row(overall[cfg], overall_nf[cfg[0]]) for cfg in configs}
    best = _best_config(over_rows)

    # baselines on identical windows
    mom_all = {"n_trades": 0, "net": 0.0, "fees": 0.0, "wins": 0, "shares": 0.0, "notional": 0.0}
    for sk in series_keys:
        _add(mom_all, momentum[sk])
    taker_by_theta = {}
    for t in thetas:
        agg = {"n_trades": 0, "net": 0.0, "fees": 0.0, "wins": 0, "shares": 0.0, "notional": 0.0}
        for sk in series_keys:
            _add(agg, taker[(sk, t)])
        taker_by_theta[t] = agg
    taker_best_t = max(thetas, key=lambda t: (taker_by_theta[t]["net"], t)) if thetas else None
    model_taker = taker_by_theta.get(taker_best_t, {"net": 0.0, "n_trades": 0, "wins": 0})

    # per-series best config
    per_series = {}
    for sk in series_keys:
        rows_sk = {cfg: _config_row(real[(sk, cfg[0], cfg[1], cfg[2])], nfires[(sk, cfg[0])])
                   for cfg in configs}
        b = _best_config(rows_sk)
        per_series[sk] = {"best_config": None if b is None else {"theta": b[0], "pair_cost": b[1],
                                                                 "depth_frac": b[2]},
                          "best": None if b is None else rows_sk[b],
                          "control": (_control_stats(control_net[(sk, b[0], b[1], b[2])],
                                                     rows_sk[b]["net_pnl"], seeds) if b else None),
                          "shuffled": (_control_stats(shuf_net[(sk, b[0], b[1], b[2])],
                                                      rows_sk[b]["net_pnl"], seeds) if b else None),
                          "momentum_net": float(momentum[sk]["net"]),
                          "model_taker_net": float(taker[(sk, taker_best_t)]["net"]
                                                   if taker_best_t is not None else 0.0)}

    best_row = over_rows[best] if best else None
    best_ctrl = _control_stats(overall_ctrl[best], best_row["net_pnl"], seeds) if best else None
    best_shuf = _control_stats(overall_shuf[best], best_row["net_pnl"], seeds) if best else None

    verdict = _verdict(best, best_row, best_ctrl, best_shuf, mom_all, model_taker, n_windows, seeds)
    result = {
        "params": {"title": "Pair-lean strategy backtest", "series": list(series_keys),
                   "target": target, "variant": variant, "thetas": thetas, "pair_costs": pair_costs,
                   "depth_fracs": depth_fracs, "seeds": seeds, "seed": seed,
                   "network_ms": network_ms, "venue_delay_ms": venue_delay_ms,
                   "effective_latency_ms": effective, "window_budget": window_budget,
                   "fee_rate": fee_rate},
        "caveats": CAVEATS,
        "current_regime_windows": int(n_windows),
        "reference": {"momentum_vps255": 5056.69, "model_taker_dir10": 2284.01,
                      "source": "out/learn_walkforward (money_comparison.csv)"},
        "baselines": {
            "never_trading_net": 0.0,
            "momentum": {"net_pnl": float(mom_all["net"]), "trades": int(mom_all["n_trades"]),
                         "win_rate": _rate(mom_all["wins"], mom_all["n_trades"])},
            "model_taker": {"net_pnl": float(model_taker["net"]), "trades": int(model_taker["n_trades"]),
                            "best_theta": taker_best_t,
                            "win_rate": _rate(model_taker.get("wins", 0), model_taker["n_trades"])},
        },
        "best_config": None if best is None else {"theta": best[0], "pair_cost": best[1],
                                                  "depth_frac": best[2], **best_row,
                                                  "control": best_ctrl, "shuffled": best_shuf},
        "sweep": [{"theta": t, "pair_cost": c, "depth_frac": f, **over_rows[(t, c, f)]}
                  for (t, c, f) in configs],
        "per_series": per_series,
        "taker_sweep": [{"theta": t, "net_pnl": float(taker_by_theta[t]["net"]),
                         "trades": int(taker_by_theta[t]["n_trades"])} for t in thetas],
        "verdict": verdict, "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_outputs(out_dir, result)
    return result


def _verdict(best, best_row, ctrl, shuf, mom, model_taker, n_windows, seeds) -> str:
    if best is None or best_row is None:
        return ("The pair-lean strategy produced no trades against the recorded current-regime "
                "book (verdict: inconclusive — widen the range or check the OOS predictions).")
    t, c, f = best
    net = best_row["net_pnl"]
    parts = [
        f"Over {n_windows:,} current-regime windows (≥ 2026-06-05, 255 ms effective), the best "
        f"pair-lean config (T=|p−0.5|≥{t:g}, C={c:g}, lean={f:g}×depth) nets ${net:,.2f} after "
        f"${best_row['fees']:,.2f} fees ({'MAKES' if net > 0 else 'LOSES'} money; win rate "
        f"{best_row['win_rate']:.0%}, {best_row['completion_rate']:.0%} of fires complete a pair, "
        f"locked ${best_row['locked_pnl']:,.2f} / stranded ${best_row['stranded_pnl']:,.2f}).",
        f"On the same windows: momentum nets ${mom['net']:,.2f}, model-taker ${model_taker['net']:,.2f}, "
        f"never-trading $0.00.",
    ]
    if seeds and ctrl is not None:
        parts.append(
            f"The matched-frequency random-trigger control nets ${ctrl['mean']:,.2f} ± "
            f"${ctrl['std']:,.2f} over {seeds} seeds; pair-lean {'BEATS' if ctrl['beats'] else 'does NOT beat'} "
            f"it (p={ctrl['p_value']:.2f}). Shuffled-labels control nets ${shuf['mean']:,.2f} "
            f"(p={shuf['p_value']:.2f}).")
        if not ctrl["beats"]:
            parts.append("Verdict: VOID — pair-lean does not reliably beat random timing of the "
                         "same frequency; the PnL is not attributable to the signal.")
        else:
            beats_mom = net > mom["net"]
            parts.append(
                f"Verdict: the pair-lean edge is real on this sample (beats random). It "
                f"{'BEATS' if beats_mom else 'does NOT beat'} the momentum baseline "
                f"(${mom['net']:,.2f}) — read the caveats and confirm on more history.")
    return " ".join(parts)


# --- outputs ----------------------------------------------------------------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap(result: dict) -> str | None:
    """Net PnL over (θ rows, C cols) at the best config's depth_frac."""
    best = result.get("best_config")
    if not best:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frac = best["depth_frac"]
    thetas = result["params"]["thetas"]
    pcs = result["params"]["pair_costs"]
    grid = np.full((len(thetas), len(pcs)), np.nan)
    by = {(r["theta"], r["pair_cost"], r["depth_frac"]): r["net_pnl"] for r in result["sweep"]}
    for i, t in enumerate(thetas):
        for j, c in enumerate(pcs):
            grid[i, j] = by.get((t, c, frac), np.nan)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pcs))); ax.set_xticklabels([f"{c:g}" for c in pcs])
    ax.set_yticks(range(len(thetas))); ax.set_yticklabels([f"{t:g}" for t in thetas], fontsize=8)
    ax.set_xlabel("pair-cost ceiling C"); ax.set_ylabel("confidence |p−0.5| ≥ θ")
    ax.set_title(f"Net PnL $ (lean = {frac:g}× depth)")
    for i in range(len(thetas)):
        for j in range(len(pcs)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _png(fig)


def _write_outputs(out_dir: Path, result: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default),
                                          encoding="utf-8")
    pd.DataFrame(result["sweep"]).to_csv(out_dir / "sweep.csv", index=False)
    ps_rows = []
    for sk, v in result["per_series"].items():
        bc, b = v["best_config"], v["best"]
        ps_rows.append({"series": sk,
                        "theta": None if bc is None else bc["theta"],
                        "pair_cost": None if bc is None else bc["pair_cost"],
                        "depth_frac": None if bc is None else bc["depth_frac"],
                        "net_pnl": None if b is None else b["net_pnl"],
                        "trades": None if b is None else b["trades"],
                        "win_rate": None if b is None else b["win_rate"],
                        "completion_rate": None if b is None else b["completion_rate"],
                        "locked_pnl": None if b is None else b["locked_pnl"],
                        "stranded_pnl": None if b is None else b["stranded_pnl"],
                        "control_beats": None if v["control"] is None else v["control"]["beats"],
                        "momentum_net": v["momentum_net"], "model_taker_net": v["model_taker_net"]})
    pd.DataFrame(ps_rows).to_csv(out_dir / "per_series_best.csv", index=False)

    heat = _heatmap(result)
    bl = result["baselines"]
    bc = result.get("best_config")
    cmp_rows = [
        {"strategy": "pair-lean (best)", "net_pnl": None if bc is None else bc["net_pnl"],
         "trades": None if bc is None else bc["trades"]},
        {"strategy": "momentum", "net_pnl": bl["momentum"]["net_pnl"], "trades": bl["momentum"]["trades"]},
        {"strategy": "model-taker (dir10)", "net_pnl": bl["model_taker"]["net_pnl"],
         "trades": bl["model_taker"]["trades"]},
        {"strategy": "random control (best cfg)",
         "net_pnl": None if bc is None or bc["control"] is None else bc["control"]["mean"], "trades": None},
        {"strategy": "shuffled-labels (best cfg)",
         "net_pnl": None if bc is None or bc["shuffled"] is None else bc["shuffled"]["mean"], "trades": None},
        {"strategy": "never trading", "net_pnl": 0.0, "trades": 0},
    ]
    sweep_cols = [("theta", "θ"), ("pair_cost", "C"), ("depth_frac", "lean×depth"),
                  ("trades", "trades"), ("net_pnl", "net $"), ("win_rate", "win"),
                  ("completion_rate", "pair-completion"), ("locked_pnl", "locked $"),
                  ("stranded_pnl", "stranded $")]
    ps_cols = [("series", "series"), ("theta", "θ"), ("pair_cost", "C"), ("depth_frac", "lean×depth"),
               ("trades", "trades"), ("net_pnl", "net $"), ("win_rate", "win"),
               ("completion_rate", "pair-completion"), ("locked_pnl", "locked $"),
               ("stranded_pnl", "stranded $"), ("control_beats", "beats random"),
               ("momentum_net", "momentum $"), ("model_taker_net", "model-taker $")]
    top_sweep = sorted(result["sweep"], key=lambda r: r["net_pnl"], reverse=True)[:25]
    caveats = "".join(f"<li>{c}</li>" for c in result["caveats"])
    bug = "" if (bc and bc.get("control", {}) and bc["control"].get("beats")) else " bug"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} tr.low td {{ background: #fff5e6; color: #7a5200; }}
 .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 .verdict.bug {{ border-left-color: #d62728; background: #fdeeee; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<p class="muted">Take the walk-forward dir10 side at the ask when |p−0.5| ≥ θ, then rest a maker
quote on the other side so combined pair cost ≤ C. Reconstructed current-regime book; conservative
fills; marked to the official resolution.</p>
<div class="verdict{bug}">{result['verdict']}</div>
<h2>Strategy comparison (identical windows)</h2>
{eh._table_html(cmp_rows, [("strategy", "strategy"), ("net_pnl", "net PnL $"), ("trades", "trades")])}
{eh._img_tag(heat, 'net PnL heatmap over (θ, C)')}
<h2>Best pair-lean config per series</h2>
{eh._table_html(ps_rows, ps_cols)}
<h2>Sweep — top 25 configs by net PnL</h2>
{eh._table_html(top_sweep, sweep_cols)}
<h2>Caveats (read these)</h2><ul>{caveats}</ul>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "backtest_pair_lean")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    parser.add_argument("--target", default="dir10", help="OOS target (default dir10)")
    parser.add_argument("--variant", default="full", help="OOS feature variant (default full)")
    parser.add_argument("--thetas", default=None, help="comma-separated |p−0.5| confidence grid")
    parser.add_argument("--pair-costs", default=None, help="comma-separated pair-cost ceilings C")
    parser.add_argument("--depth-fracs", default=None, help="comma-separated lean depth fractions")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"random-control seeds (default {DEFAULT_SEEDS}; the dominant runtime cost)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--network-ms", type=int, default=5, help="network latency ms (default 5, VPS)")
    parser.add_argument("--venue-delay-ms", type=int, default=250,
                        help="venue taker delay ms (default 250; effective 255 = vps255 reference)")
    parser.add_argument("--window-budget", type=float, default=mj.DEFAULT_WINDOW_BUDGET)
    parser.add_argument("--fee-rate", type=float, default=mj.CRYPTO_FEE_RATE)
    parser.add_argument("--out-name", default="pair_lean", help="output subdir under out/backtests/")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None

    def _floats(s):
        return [float(x) for x in s.split(",")] if s else None

    print(f"[backtest_pair_lean] telonex={paths.telonex_dir} "
          f"out={paths.out_dir / 'backtests' / args.out_name} "
          f"latency=net {args.network_ms}+venue {args.venue_delay_ms}={args.network_ms + args.venue_delay_ms}ms "
          f"seeds={args.seeds}")
    result = backtest_pair_lean(paths, series=series, since_ms=since_ms, until_ms=until_ms,
                                thetas=_floats(args.thetas), pair_costs=_floats(args.pair_costs),
                                depth_fracs=_floats(args.depth_fracs), seeds=args.seeds, seed=args.seed,
                                network_ms=args.network_ms, venue_delay_ms=args.venue_delay_ms,
                                window_budget=args.window_budget, fee_rate=args.fee_rate,
                                target=args.target, variant=args.variant, out_name=args.out_name)
    bc = result["best_config"]
    if bc:
        print(f"[backtest_pair_lean] best T=|p-0.5|>={bc['theta']:g} C={bc['pair_cost']:g} "
              f"lean={bc['depth_frac']:g}: net ${bc['net_pnl']:.2f} ({bc['trades']:,} trades, "
              f"win {bc['win_rate']:.0%}, completion {bc['completion_rate']:.0%})")
    bl = result["baselines"]
    print(f"[backtest_pair_lean] baselines: momentum ${bl['momentum']['net_pnl']:.2f}, "
          f"model-taker ${bl['model_taker']['net_pnl']:.2f}, never-trade $0.00 "
          f"over {result['current_regime_windows']:,} windows")
    procmem.report_peak_rss("backtest_pair_lean", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[backtest_pair_lean] wrote metrics.json + sweep.csv + per_series_best.csv + report.html "
          f"to {paths.out_dir / 'backtests' / args.out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
