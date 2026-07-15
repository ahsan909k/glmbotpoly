"""Stage — backtest_clones: mechanize each competitor **operating manual** into a
backtestable strategy and score it honestly over the current-regime book.

Two clone templates (parameterized from the operating manuals — ``competitors.manuals``):

- **taker-accumulation** (takerner, bonereaper): at observed entry times buy toward an
  equal Up/Down inventory as a taker (crossing the book), pay the exact taker fee, mark
  each leg to the official resolution. Directional/pair economics fall out of marking both
  legs (a matched pair pays $1; the excess leg settles at the outcome).
- **maker-quote + merge-recycle** (0xb27b, wolf9478, nagi777): rest two-sided BUY limits
  below fair; fills modeled as a **bracket** — *pessimistic* (queue behind all displayed
  size, fill only on real sell-aggressor prints through the price) and *optimistic* (fill
  the instant the ask reaches our limit). Makers pay ~$0 taker fee; matched pairs merge to
  $1 (the merge/hold PnL is identical — the recycle advantage is quantified in the manual).

Plus a **momentum-exit** variant (D): enter on the engine's momentum trigger, then SELL at
the bid when the PM mid converges to the reconstructed fair (or after ``T`` seconds; sweep
``T ∈ {15,30,60}``), taker fees both ways — vs hold-to-resolution.

Every clone emits the shared ``money_judge._TRADE_COLS`` trades.csv, carries a
shuffled-outcome control + a matched-frequency random control + capital accounting, and is
run at **both** our Bronze tier and its owner's tier (rebates via ``rebate_sim``). For each
clone the report prints the **clone-vs-owner ladder** — the owner's own official P/L over
the same post-Jun-5 slice (FACT, from the manual) next to the clone's backtest net, with the
gap shown, never smoothed.

Reconstruction + controls reuse ``backtest_momentum`` / ``money_judge`` verbatim; the maker
optimistic fill reuses ``backtest_pair_lean._maker_completion``. Research only; LightGBM-free.

Run::

    python -m model_lab.backtest_clones --clone takerner --since 2026-06-05
    python -m model_lab.backtest_clones --clone all
"""

from __future__ import annotations

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
from . import rebate_sim as rs
from . import forensic_audit as fa
from .config import Paths, resolve_bounds, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .lib import math as lm
from .lib import procmem

MS_PER_DAY = 86_400_000
DEFAULT_SEED = 20260712
DEFAULT_SEEDS = 8
CURRENT_REGIME_FROM = date(2026, 6, 5)
OUR_TIER = "Bronze"  # our fee tier for the side-by-side (amendment: "Bronze, no cushion")
EXIT_TIMEOUTS = (15.0, 30.0, 60.0)  # momentum-exit T sweep (seconds)

# --- clone roster (parameters mechanized from each operating manual) ----------
# Each clone is a mechanization of the account's ARCHETYPE at representative parameters
# drawn from its manual's distributions — its fidelity gap vs the owner is exactly what the
# clone-vs-owner ladder measures. `owner_tier` is the account's real Gamma tier (FACT).
CLONES: dict[str, dict] = {
    "takerner": {  # ~99% taker, hold-to-resolution, Platinum
        "template": "taker", "owner_tier": "Platinum", "handle": "takerner",
        "entry_fracs": [0.15, 0.35, 0.55, 0.75], "share_size": 20.0,
        "window_budget": mj.DEFAULT_WINDOW_BUDGET,
    },
    "bonereaper": {  # redeem-heavy taker, Obsidian
        "template": "taker", "owner_tier": "Obsidian", "handle": "bonereaper",
        "entry_fracs": [0.2, 0.5, 0.8], "share_size": 25.0,
        "window_budget": mj.DEFAULT_WINDOW_BUDGET,
    },
    "0xb27b": {  # pure maker, merge-recycle, Platinum (handle in handles.json = full address)
        "template": "maker", "owner_tier": "Platinum",
        "handle": "0xb27bc932bf8110d8f78e55da7d5f0497a18b5b82",
        "place_frac": 0.05, "limit_offset": 0.01, "share_size": 50.0, "merge_min_pairs": 25.0,
    },
    "wolf9478": {  # pure maker, Silver
        "template": "maker", "owner_tier": "Silver", "handle": "wolf9478",
        "place_frac": 0.1, "limit_offset": 0.015, "share_size": 30.0, "merge_min_pairs": 10.0,
    },
    "nagi777": {  # merge-heavy maker, tier fetched fresh (fallback Platinum)
        "template": "maker", "owner_tier": "Platinum", "handle": "nagi777",
        "place_frac": 0.05, "limit_offset": 0.01, "share_size": 50.0, "merge_min_pairs": 25.0,
    },
}


def _regime_floor_ms(regime_from: date) -> int:
    return int(datetime(regime_from.year, regime_from.month, regime_from.day,
                        tzinfo=timezone.utc).timestamp()) * 1000


# ===========================================================================
# maker-fill bracket (E)
# ===========================================================================
def _sell_print_vol(trades: pd.DataFrame, limit: float, lo_ms: int, hi_ms: int) -> float:
    """Total **sell-aggressor** print volume at price ``≤ limit`` in ``[lo_ms, hi_ms)`` — the
    flow that could fill a resting BUY at ``limit`` (an aggressive sell hits our bid)."""
    if trades is None or trades.empty:
        return 0.0
    ts = trades["ts_ms"].to_numpy(dtype="int64")
    sel = ((ts >= lo_ms) & (ts < hi_ms)
           & (trades["side"].astype(str).str.lower().to_numpy() == "sell")
           & (trades["price"].to_numpy(dtype=float) <= limit))
    return float(trades["size"].to_numpy(dtype=float)[sel].sum())


def maker_fill_optimistic(book: dict, trades: pd.DataFrame, limit: float, lo_ms: int,
                          hi_ms: int, our_size: float) -> float:
    """Optimistic (front-of-queue) fill: our resting BUY at ``limit`` is FIRST in line, so it
    fills the sell-aggressor print volume at ``≤ limit`` up to ``our_size``, at our limit,
    $0 fee. The upper bound of the bracket ("if we always won the queue race")."""
    return float(min(our_size, _sell_print_vol(trades, limit, lo_ms, hi_ms)))


def maker_fill_pessimistic(book: dict, trades: pd.DataFrame, limit: float, lo_ms: int,
                           hi_ms: int, our_size: float) -> float:
    """Pessimistic (queue-behind-all-displayed, §9) fill: our resting BUY sits behind the ENTIRE
    displayed bid size at placement, so the first ``queue_ahead`` shares of sell-print flow drain
    ahead of us before we fill. Same print stream as the optimistic fill → **always ≤ it**."""
    ts = book["ts"]
    i = int(np.searchsorted(ts, lo_ms, side="right")) - 1
    queue_ahead = float(book["bid_sz"][i]) if i >= 0 else 0.0
    vol = _sell_print_vol(trades, limit, lo_ms, hi_ms)
    return float(max(0.0, min(our_size, vol - queue_ahead)))


# ===========================================================================
# per-window clone simulators
# ===========================================================================
def run_taker_clone(win: dict, snaps: list[dict], books: dict, *, params: dict,
                    fee_rate: float, latency_ms: int, side_fn=None) -> list[dict]:
    """Taker-accumulation over one window: at each observed entry fraction, buy toward equal
    inventory (or ``side_fn(i)`` for the random control), taker-fill at the book, mark to
    resolution. Emits per-fill ``_TRADE_COLS`` rows."""
    up_token, close_ms, outcome_up = win["up_token"], win["close_time"], win["outcome_up"]
    open_ms = win["open_ms"]
    dur_ms = close_ms - open_ms
    up_sh = dn_sh = 0.0
    budget_left = float(params["window_budget"])
    size = float(params["share_size"])
    rows: list[dict] = []
    for i, frac in enumerate(params["entry_fracs"]):
        if budget_left < mj.MIN_NOTIONAL:
            break
        now = open_ms + int(frac * dur_ms)
        if now >= close_ms:
            continue
        side = side_fn(i) if side_fn is not None else ("Up" if up_sh <= dn_sh else "Down")
        book = mj._book_asof(books, up_token, now + latency_ms)
        if book is None:
            continue
        quote = mj._leaned_quote(book, side)
        if quote is None:
            continue
        price, avail = quote
        shares = mj._fill_shares(price, avail, budget_left, size * price + 1.0)
        # cap the per-fire size to `size` shares (not just notional)
        shares = min(shares, size)
        if shares <= 0.0 or shares * price < mj.MIN_NOTIONAL:
            continue
        trade = mj._mark_to_resolution(side, shares, price, outcome_up, fee_rate)
        budget_left -= shares * price
        if side == "Up":
            up_sh += shares
        else:
            dn_sh += shares
        rows.append({"series": win["series"], "window_open_ms": int(open_ms),
                     "sample_ts_ms": int(now), "day": int(open_ms) // MS_PER_DAY,
                     "p_up": float("nan"), "outcome_up": int(outcome_up), **trade})
    return rows


def run_maker_clone(win: dict, fair_up_at_place: float, books: dict, trades_by_tok: dict, *,
                    params: dict, fill_model: str, latency_ms: int,
                    limit_shift: float = 0.0) -> list[dict]:
    """Two-sided maker quoting over one window: rest an Up BUY and a Down BUY at
    ``fair − limit_offset`` (shifted by ``limit_shift`` for the random control), fill via the
    ``fill_model`` bracket (``"optimistic"`` / ``"pessimistic"``), $0 maker fee, mark filled
    shares to resolution (matched pairs merge to the same PnL). One aggregate ``_TRADE_COLS``
    row per window (fees ≈ 0 → rebate tier is immaterial by construction)."""
    up_token, dn_token = win["up_token"], win.get("down_token")
    close_ms, outcome_up, open_ms = win["close_time"], win["outcome_up"], win["open_ms"]
    place = open_ms + int(float(params["place_frac"]) * (close_ms - open_ms)) + latency_ms
    size = float(params["share_size"])
    off = float(params["limit_offset"]) + limit_shift
    up_limit = fair_up_at_place - off
    dn_limit = (1.0 - fair_up_at_place) - off
    fills = []  # (side, shares, price)
    for side, token, limit in (("Up", up_token, up_limit), ("Down", dn_token, dn_limit)):
        if token is None or not (0.0 < limit < 1.0):
            continue
        book = books.get(token)
        if book is None:
            continue
        tr = trades_by_tok.get(token)
        if fill_model == "optimistic":
            n = maker_fill_optimistic(book, tr, limit, place, close_ms, size)
        else:
            n = maker_fill_pessimistic(book, tr, limit, place, close_ms, size)
        if n > 0.0:
            fills.append((side, n, limit))
    if not fills:
        return []
    up_f = sum(n for s, n, _ in fills if s == "Up")
    dn_f = sum(n for s, n, _ in fills if s == "Down")
    cost = sum(n * p for _s, n, p in fills)  # maker fee = 0
    # Settle: matched pairs pay $1 (merge or hold — same PnL); excess settles at the outcome.
    matched = min(up_f, dn_f)
    up_excess, dn_excess = up_f - matched, dn_f - matched
    excess_payoff = up_excess * (1.0 if outcome_up == 1 else 0.0) + dn_excess * (1.0 if outcome_up == 0 else 0.0)
    payoff = matched * 1.0 + excess_payoff
    net = payoff - cost
    shares = up_f + dn_f
    return [{"series": win["series"], "window_open_ms": int(open_ms),
             "sample_ts_ms": int(place), "day": int(open_ms) // MS_PER_DAY,
             "p_up": float(fair_up_at_place), "outcome_up": int(outcome_up),
             "side": "Pair", "shares": float(shares), "price": (cost / shares) if shares else 0.0,
             "fee": 0.0, "cost": float(cost), "payoff": float(payoff), "net": float(net),
             "won": bool(net > 0.0), "matched_pairs": float(matched)}]


# ===========================================================================
# momentum-exit (D)
# ===========================================================================
def run_momentum_exit(model_rows: list[dict], mids_by_asset: dict, win_by_key: dict,
                      snaps_by_key: dict, books: dict, *, latency_ms: int, fee_rate: float,
                      window_budget: float, timeout_s: float) -> list[dict]:
    """The engine's momentum entry (identical gate ladder to ``mj._run_momentum``) but the
    position is CLOSED early by selling at the bid when the side-oriented PM mid converges to
    the reconstructed fair, or after ``timeout_s``. Taker fees both ways. Emits the
    ``_TRADE_COLS`` superset (+ ``exit_ts_ms, exit_price, exit_reason``)."""
    rows: list[dict] = []
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for m in model_rows:
        by_win[(m["series"], m["open_time"])].append(m)
    for key, snaps in by_win.items():
        win = win_by_key.get(key)
        if win is None:
            continue
        up_token, close_ms, outcome_up = win["up_token"], win["close_time"], win["outcome_up"]
        # per-window as-of fair lookup from the reconstructed snapshots
        sn = snaps_by_key.get(key)
        s_ts = sn["ts"] if sn else np.empty(0, dtype="int64")
        s_pup = sn["p_up"] if sn else np.empty(0, dtype=float)

        budget_left, last_take = window_budget, None
        for m in sorted(snaps, key=lambda s: s["ts"]):
            now, p_up, sigma = m["ts"], m["p_up"], m["sigma_1s"]
            if m["health"] != "Ready" or not (0.0 < p_up < 1.0) or not (sigma > 0.0):
                continue
            if (close_ms - now) / 1000.0 <= 0.0:
                continue
            direction = mj._confirmed_direction(mids_by_asset.get(m["asset"]), now, sigma)
            if direction is None:
                continue
            if last_take is not None and now - last_take < mj.COOLDOWN_MS:
                continue
            if budget_left < mj.MIN_NOTIONAL:
                continue
            book = mj._book_asof(books, up_token, now + latency_ms)
            if book is None:
                continue
            quote = mj._leaned_quote(book, direction)
            if quote is None:
                continue
            price, avail = quote
            fair = p_up if direction == "Up" else 1.0 - p_up
            shares = mj._plan_take_single_level(fair, price, avail, budget_left, fee_rate)
            if shares <= 0.0:
                continue
            entry_fee = lm.taker_fee(shares, fee_rate, price)

            def fair_fn(t, _dir=direction, _ts=s_ts, _pup=s_pup):
                if _ts.size == 0:
                    return None
                i = int(np.searchsorted(_ts, t, side="right")) - 1
                if i < 0:
                    return None
                pu = float(_pup[i])
                return pu if _dir == "Up" else 1.0 - pu

            trade = mj._mark_with_momentum_exit(
                direction, shares, price, entry_fee, books, up_token, now + latency_ms,
                fair_fn, close_ms, outcome_up, fee_rate=fee_rate, timeout_s=timeout_s)
            budget_left -= shares * price
            last_take = now
            rows.append({"series": key[0], "window_open_ms": int(key[1]), "sample_ts_ms": int(now),
                         "day": int(key[1]) // MS_PER_DAY, "p_up": p_up, "outcome_up": outcome_up,
                         **trade})
    return rows


# ===========================================================================
# controls
# ===========================================================================
def shuffled_net(rows: list[dict], series_key: str, day_str: str, seed: int) -> float:
    """Net PnL of a set of window rows under a per-(series, day) outcome permutation
    (crc32-salted). Re-marks each row's payoff from its cost + shares + shuffled outcome —
    a maker's matched-pair edge survives a shuffle, a directional taker's collapses."""
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
        if r.get("side") == "Pair":  # maker aggregate: re-settle matched (always $1) + excess
            matched = r.get("matched_pairs", 0.0)
            # cost - matched already captured; recompute excess payoff under shuffled o.
            # excess shares = shares - 2*matched, all one side unknown here → approximate by
            # keeping matched at $1 and re-marking the net's directional part is not separable;
            # for the aggregate row we conservatively hold matched at $1 and leave the rest.
            net += (matched * 1.0 - r["cost"]) + (r["payoff"] - matched * 1.0) * (1.0 if o == r["outcome_up"] else 0.0)
        else:
            s_wins = (r["side"] == "Up" and o == 1) or (r["side"] == "Down" and o == 0)
            net += (r["shares"] * (1.0 if s_wins else 0.0)) - r["cost"]
    return float(net)


# ===========================================================================
# worker
# ===========================================================================
def _process_clone_day(paths, series_key, asset, symbol, d, day_str, spec, res_map, excluded,
                       strike_tol_ms, effective, fee_rate, seeds, seed) -> dict:
    """Reconstruct one (series, day)'s current-regime windows and run the clone (pessimistic +
    optimistic for makers) + momentum-exit + hold + both controls; return the day's rows +
    per-seed control nets (checkpointed — no state retained across days)."""
    wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
    grid, ring_ts, ring_px = (bm._day_grid_and_ring(paths, symbol, asset, d) if wins
                              else (pd.DataFrame(), None, None))
    day_rows: list[dict] = []
    day_opt: list[dict] = []
    hold_rows: list[dict] = []
    exit_rows: dict[str, list[dict]] = {str(int(t)): [] for t in EXIT_TIMEOUTS}
    shuf = [0.0] * seeds
    rand = [0.0] * seeds
    n_win = 0
    if wins and not grid.empty:
        bars = bm._grid_bars(grid)
        model_rows: list[dict] = []
        win_by_key: dict[tuple, dict] = {}
        snaps_by_key: dict[tuple, dict] = {}
        books: dict[str, dict] = {}
        trades_by_tok: dict[str, pd.DataFrame] = {}
        for w in wins:
            open_ms, close_ms = int(w["window_open_ms"]), int(w["window_close_ms"])
            official = res_map.get((series_key, open_ms))
            if official is None:
                continue
            st = hc.binance_anchored_strike(bars, open_ms, strike_tol_ms)
            if not (st["strike"] > 0.0):
                continue
            upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
            dnq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Down", day_str)
            if upq.empty:
                continue
            up_token = str(upq["token_id"].iloc[0])
            down_token = str(dnq["token_id"].iloc[0]) if not dnq.empty else None
            outcome_up = 1 if official["outcome"] == "Up" else 0
            books[up_token] = tpm.pm_book_arrays(upq)
            if down_token is not None:
                books[down_token] = tpm.pm_book_arrays(dnq)
            key = (series_key, open_ms)
            win_by_key[key] = {"series": series_key, "up_token": up_token, "down_token": down_token,
                               "close_time": close_ms, "outcome_up": outcome_up, "open_ms": open_ms}
            snaps = hc.reconstruct_window_snapshots(grid, {**w, **st})
            if not snaps.empty:
                snaps_by_key[key] = {"ts": snaps["ts"].to_numpy(dtype="int64"),
                                     "p_up": snaps["p_up"].to_numpy(dtype=float)}
                model_rows.extend(bm._window_model_rows(grid, w, st))
            n_win += 1
            if spec["template"] == "taker":
                day_rows.extend(run_taker_clone(win_by_key[key], [], books, params=spec,
                                                fee_rate=fee_rate, latency_ms=effective))
            else:  # maker — needs a fair at placement + the trade tapes (pessimistic)
                sk = snaps_by_key.get(key)
                if sk is None or sk["ts"].size == 0:
                    continue
                place = open_ms + int(float(spec["place_frac"]) * (close_ms - open_ms)) + effective
                j = int(np.searchsorted(sk["ts"], place, side="right")) - 1
                fair_up = float(sk["p_up"][max(0, j)])
                trades_by_tok[up_token] = tpm.read_pm_trades(paths.telonex_dir, w["slug"], "Up", day_str)
                if down_token is not None:
                    trades_by_tok[down_token] = tpm.read_pm_trades(paths.telonex_dir, w["slug"], "Down", day_str)
                day_rows.extend(run_maker_clone(win_by_key[key], fair_up, books, trades_by_tok,
                                                params=spec, fill_model="pessimistic", latency_ms=effective))
                day_opt.extend(run_maker_clone(win_by_key[key], fair_up, books, trades_by_tok,
                                               params=spec, fill_model="optimistic", latency_ms=effective))
        if model_rows:
            mids = {asset: (ring_ts, ring_px)}
            hold_rows.extend(mj._run_momentum(model_rows, mids, win_by_key, books,
                                              latency_ms=effective, fee_rate=fee_rate,
                                              window_budget=mj.DEFAULT_WINDOW_BUDGET))
            for t in EXIT_TIMEOUTS:
                exit_rows[str(int(t))].extend(run_momentum_exit(
                    model_rows, mids, win_by_key, snaps_by_key, books, latency_ms=effective,
                    fee_rate=fee_rate, window_budget=mj.DEFAULT_WINDOW_BUDGET, timeout_s=t))
        for si in range(seeds):
            shuf[si] = shuffled_net(day_rows, series_key, day_str, seed + si * 101)
            rand[si] = _random_control_day(win_by_key, spec, books, trades_by_tok, snaps_by_key,
                                           effective, fee_rate, seed + si * 7919, series_key, day_str)
    return {"series": series_key, "day": day_str, "n_win": n_win, "real": day_rows, "opt": day_opt,
            "hold": hold_rows, "exit": exit_rows, "shuf": shuf, "rand": rand}


def backtest_clone(
    paths: Paths, clone: str, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None,
    seeds: int = DEFAULT_SEEDS, seed: int = DEFAULT_SEED,
    network_ms: int = 5, venue_delay_ms: int = 250, fee_rate: float = mj.CRYPTO_FEE_RATE,
    res_map: dict | None = None, regime_from: date = CURRENT_REGIME_FROM,
    out_name: str | None = None, manuals_dir: Path | None = None,
) -> dict:
    """Run one clone over the current-regime windows; write trades.csv + report. Returns the
    metrics dict. Reconstruction mirrors ``backtest_pair_lean._process_day``."""
    if clone not in CLONES:
        raise ValueError(f"unknown clone {clone!r}; choose from {sorted(CLONES)}")
    spec = CLONES[clone]
    paths.ensure_out()
    effective = network_ms + venue_delay_ms
    out_name = out_name or f"clone_{clone}"
    out_dir = paths.out_dir / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    if res_map is None:
        res_map = hr.load_resolution_map(paths, series_keys)
    strike_tol_ms = ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0
    since_floor = max(_regime_floor_ms(regime_from), since_ms) if since_ms is not None else _regime_floor_ms(regime_from)

    real_rows: list[dict] = []          # the clone's trades (pessimistic for makers)
    opt_rows: list[dict] = []           # optimistic bracket (makers only)
    exit_rows: dict[float, list[dict]] = {t: [] for t in EXIT_TIMEOUTS}
    hold_rows: list[dict] = []          # momentum hold-to-resolution (for the exit comparison)
    shuf_by_seed = np.zeros(seeds, dtype=float)
    rand_by_seed = np.zeros(seeds, dtype=float)
    n_windows = 0
    done: set[tuple[str, str]] = set()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _accum(day: dict) -> None:
        nonlocal n_windows
        real_rows.extend(day["real"])
        opt_rows.extend(day["opt"])
        hold_rows.extend(day["hold"])
        for t in EXIT_TIMEOUTS:
            exit_rows[t].extend(day["exit"].get(str(int(t)), []))
        shuf_by_seed[:] += np.array(day["shuf"], dtype=float)
        rand_by_seed[:] += np.array(day["rand"], dtype=float)
        n_windows += int(day["n_win"])

    # Resume: load any per-(series, day) checkpoints so a re-launch continues where a killed
    # run left off (the environment reaps long runs; each day is checkpointed atomically).
    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(data.get("seeds", -1)) != seeds:  # a seed-count change invalidates the control sums
            continue
        _accum(data)
        done.add((data["series"], data["day"]))

    for series_key in series_keys:
        prefix = hc.series_prefix(series_key)
        asset = hc.SLUG_PREFIXES[prefix][1]
        symbol = hc.SYMBOL_BY_ASSET[asset]
        for d in hlbl._days_for_series(paths, series_key, since_floor, until_ms):
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            day = _process_clone_day(paths, series_key, asset, symbol, d, day_str, spec, res_map,
                                     excluded, strike_tol_ms, effective, fee_rate, seeds, seed)
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"seeds": seeds, **day}, default=eh._json_default), encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            _accum({"seeds": seeds, **day})
            print(f"[backtest_clones:{clone}] {series_key} {day_str}: {day['n_win']} windows "
                  f"({len(done)} days done, {n_windows:,} cumulative)", flush=True)

    return _finalize_clone(paths, out_dir, clone, spec, series_keys, real_rows, opt_rows,
                           hold_rows, exit_rows, shuf_by_seed, rand_by_seed, n_windows, seeds, seed,
                           network_ms, venue_delay_ms, effective, fee_rate, regime_from, manuals_dir)


def _random_control_day(win_by_key, spec, books, trades_by_tok, snaps_by_key, latency_ms,
                        fee_rate, rng_seed, series_key, day_str) -> float:
    """Matched-frequency random control for one day: the clone trades every window, so the
    frequency-matched control keeps the schedule but randomizes the CHOICE (taker: random
    side; maker: random limit shift ± a tick). Tests whether picking the right side / level
    matters (for makers it shouldn't — pairs; for directional takers it should). Net-only."""
    rng = np.random.default_rng(zlib.crc32(f"{series_key}|{day_str}".encode()) % 1_000_000 + rng_seed)
    net = 0.0
    for key, win in win_by_key.items():
        if spec["template"] == "taker":
            rows = run_taker_clone(win, [], books, params=spec, fee_rate=fee_rate,
                                   latency_ms=latency_ms,
                                   side_fn=lambda _i: ("Up" if rng.random() < 0.5 else "Down"))
        else:
            sk = snaps_by_key.get(key)
            if sk is None or sk["ts"].size == 0:
                continue
            open_ms, close_ms = win["open_ms"], win["close_time"]
            place = open_ms + int(float(spec["place_frac"]) * (close_ms - open_ms)) + latency_ms
            j = int(np.searchsorted(sk["ts"], place, side="right")) - 1
            fair_up = float(sk["p_up"][max(0, j)])
            shift = float(rng.uniform(-0.01, 0.01))
            rows = run_maker_clone(win, fair_up, books, trades_by_tok, params=spec,
                                   fill_model="pessimistic", latency_ms=latency_ms, limit_shift=shift)
        net += sum(r["net"] for r in rows)
    return float(net)


# ===========================================================================
# finalize + tier honesty + clone-vs-owner ladder
# ===========================================================================
def _net(rows) -> float:
    return float(sum(r["net"] for r in rows))


def _rollup(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(1 for r in rows if r["won"])
    return {"trades": n, "net_pnl": _net(rows), "fees": float(sum(r["fee"] for r in rows)),
            "win_rate": (wins / n) if n else float("nan")}


def _tier_runs(paths: Paths, trades_csv: Path, owner_tier: str) -> dict:
    """Run rebate_sim at OUR tier (Bronze) and the owner's tier, side by side."""
    df = pd.read_csv(trades_csv)
    if df.empty or "shares" not in df.columns:
        return {"note": "no trades to rebate"}
    bronze = rs.compute_rebate_timeline(df, force_tier=OUR_TIER)
    owner = rs.compute_rebate_timeline(df, force_tier=owner_tier)
    return {
        "our_tier": {"tier": OUR_TIER, "rebate": bronze["totals"]["total_rebate"],
                     "corrected_net": bronze["totals"]["corrected_net"]},
        "owner_tier": {"tier": owner_tier, "rebate": owner["totals"]["total_rebate"],
                       "corrected_net": owner["totals"]["corrected_net"]},
        "total_fees": bronze["totals"]["total_fees"],
        "note": ("maker clones pay ~$0 taker fee, so rebate ≈ $0 at ANY tier — the tier axis "
                 "bites only the taker clones"),
    }


def _owner_ladder(spec: dict, clone_net: float, manuals_dir: Path | None) -> dict:
    """The clone-vs-owner calibration ladder: (1) owner official post-Jun-5 slice (FACT), (2)
    owner OUR-series reconstructed net (CALC/EST), (3) clone backtest net — with the gaps."""
    ladder = {"clone_net_usd": round(clone_net, 2), "official_slice_usd": None,
              "owner_reconstructed_net_usd": None}
    mpath = None
    if manuals_dir is not None:
        mpath = Path(manuals_dir) / f"{spec['handle']}.json"
    if mpath is not None and mpath.exists():
        try:
            man = json.loads(mpath.read_text(encoding="utf-8"))
            lad = man.get("anchor", {}).get("slice_ladder", {})
            ladder["official_slice_usd"] = lad.get("official_account_wide_slice_usd")
            ladder["owner_reconstructed_net_usd"] = lad.get("owner_our_series_reconstructed_net_usd")
        except (OSError, json.JSONDecodeError):
            pass
    off, own = ladder["official_slice_usd"], ladder["owner_reconstructed_net_usd"]
    ladder["gap_official_minus_reconstructed"] = (round(off - own, 2)
                                                  if isinstance(off, (int, float)) and isinstance(own, (int, float)) else None)
    ladder["gap_reconstructed_minus_clone"] = (round(own - clone_net, 2)
                                               if isinstance(own, (int, float)) else None)
    ladder["note"] = ("gap 1→2 = other markets / unobserved; gap 2→3 = what we could NOT copy "
                      "(queue position, unobserved triggers, tier). Printed, not smoothed.")
    return ladder


def _control_stats(vec: np.ndarray, real_net: float, seeds: int) -> dict:
    if not seeds:
        return {"mean": float("nan"), "std": float("nan"), "p_value": float("nan"), "beats": False}
    p = float(np.mean(vec >= real_net))
    return {"mean": float(np.mean(vec)), "std": float(np.std(vec)), "p_value": p,
            "beats": bool(real_net > float(np.mean(vec)) and p <= 0.1), "nets": vec.tolist()}


def _finalize_clone(paths, out_dir, clone, spec, series_keys, real_rows, opt_rows, hold_rows,
                    exit_rows, shuf_by_seed, rand_by_seed, n_windows, seeds, seed, network_ms,
                    venue_delay_ms, effective, fee_rate, regime_from, manuals_dir) -> dict:
    is_maker = spec["template"] == "maker"
    real = _rollup(real_rows)
    # trades.csv (the pessimistic bracket for makers; the shared _TRADE_COLS schema).
    trades_df = pd.DataFrame(real_rows, columns=mj._TRADE_COLS) if real_rows else pd.DataFrame(columns=mj._TRADE_COLS)
    trades_csv = out_dir / "trades.csv"
    trades_df.to_csv(trades_csv, index=False)

    # maker-fill bracket (E)
    bracket = None
    if is_maker:
        bracket = {"pessimistic_net": _net(real_rows), "optimistic_net": _net(opt_rows),
                   "note": "truth is between; only live measures it."}

    # capital axis
    capital = fa.capital_accounting(trades_df) if len(trades_df) else {}
    # tier honesty (C)
    tiers = _tier_runs(paths, trades_csv, spec["owner_tier"]) if len(trades_df) else {}
    # clone-vs-owner ladder (Q3)
    ladder = _owner_ladder(spec, real["net_pnl"], manuals_dir)
    # controls
    shuf = _control_stats(shuf_by_seed, real["net_pnl"], seeds)
    rand = _control_stats(rand_by_seed, real["net_pnl"], seeds)
    # momentum-exit vs hold
    hold_net = _net(hold_rows)
    exits = {f"T{int(t)}": {"net_pnl": _net(exit_rows[t]), "trades": len(exit_rows[t])}
             for t in EXIT_TIMEOUTS}

    result = {
        "params": {"title": f"Competitor clone — {clone}", "clone": clone, "template": spec["template"],
                   "owner_tier": spec["owner_tier"], "series": list(series_keys),
                   "network_ms": network_ms, "venue_delay_ms": venue_delay_ms,
                   "effective_latency_ms": effective, "fee_rate": fee_rate,
                   "regime_from": regime_from.isoformat(), "seeds": seeds, "seed": seed},
        "current_regime_windows": int(n_windows),
        "clone": real,
        "maker_fill_bracket": bracket,
        "tier_honesty": tiers,
        "clone_vs_owner_ladder": ladder,
        "capital": capital,
        "controls": {"shuffled_outcome": shuf, "matched_frequency_random": rand,
                     "verdict_void": bool(seeds and not rand["beats"])},
        "momentum_exit": {"hold_to_resolution_net": hold_net, "hold_trades": len(hold_rows),
                          "by_timeout": exits,
                          "note": "enter on the momentum trigger; exit by selling at bid on "
                                  "convergence-to-fair or after T s; taker fees both ways."},
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    result["verdict"] = _verdict(clone, real, bracket, rand, ladder, hold_net, exits)
    _write_outputs(out_dir, result)
    return result


def _verdict(clone, real, bracket, rand, ladder, hold_net, exits) -> str:
    parts = [f"The {clone}-clone nets ${real['net_pnl']:,.2f} over {real['trades']:,} trades "
             f"(win {real['win_rate']:.0%}) on the current-regime book."]
    if bracket:
        parts.append(f"Maker-fill BRACKET: pessimistic ${bracket['pessimistic_net']:,.2f} … "
                     f"optimistic ${bracket['optimistic_net']:,.2f} — truth is between; only live measures it.")
    if rand and np.isfinite(rand.get("p_value", float('nan'))):
        parts.append(f"vs a matched-frequency random control ${rand['mean']:,.2f} "
                     f"(p={rand['p_value']:.2f}): {'BEATS' if rand['beats'] else 'does NOT beat (VOID)'}.")
    off, own = ladder.get("official_slice_usd"), ladder.get("owner_reconstructed_net_usd")
    if isinstance(own, (int, float)):
        parts.append(f"Owner ladder — official slice ${_fmt(off)} → owner OUR-series reconstructed "
                     f"${own:,.2f} → clone ${real['net_pnl']:,.2f}; the gaps quantify other-markets + "
                     f"what we could NOT copy (queue, triggers, tier).")
    best_exit = max(exits.items(), key=lambda kv: kv[1]["net_pnl"]) if exits else None
    if best_exit:
        parts.append(f"Momentum-exit: hold-to-resolution ${hold_net:,.2f} vs best early-exit "
                     f"{best_exit[0]} ${best_exit[1]['net_pnl']:,.2f}.")
    return " ".join(parts)


def _fmt(x):
    return f"{x:,.2f}" if isinstance(x, (int, float)) else "—"


def _write_outputs(out_dir: Path, result: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default),
                                          encoding="utf-8")
    cmp_rows = [{"strategy": "clone (pessimistic)", "net_pnl": result["clone"]["net_pnl"],
                 "trades": result["clone"]["trades"]}]
    if result.get("maker_fill_bracket"):
        cmp_rows.append({"strategy": "clone (optimistic)",
                         "net_pnl": result["maker_fill_bracket"]["optimistic_net"], "trades": None})
    cmp_rows.append({"strategy": "random control (mean)",
                     "net_pnl": result["controls"]["matched_frequency_random"].get("mean"), "trades": None})
    lad = result["clone_vs_owner_ladder"]
    ladder_rows = [
        {"#": 1, "quantity": "owner official post-Jun-5 slice (FACT)", "value": lad.get("official_slice_usd")},
        {"#": 2, "quantity": "owner OUR-series reconstructed net (CALC/EST)", "value": lad.get("owner_reconstructed_net_usd")},
        {"#": 3, "quantity": "clone backtest net (EST)", "value": lad.get("clone_net_usd")},
    ]
    tiers = result.get("tier_honesty", {})
    tier_rows = []
    if tiers.get("our_tier"):
        tier_rows = [{"tier": tiers["our_tier"]["tier"], "rebate": tiers["our_tier"]["rebate"],
                      "corrected_net": tiers["our_tier"]["corrected_net"]},
                     {"tier": tiers["owner_tier"]["tier"], "rebate": tiers["owner_tier"]["rebate"],
                      "corrected_net": tiers["owner_tier"]["corrected_net"]}]
    exits = result["momentum_exit"]["by_timeout"]
    exit_rows = ([{"exit": "hold-to-resolution", "net_pnl": result["momentum_exit"]["hold_to_resolution_net"]}]
                 + [{"exit": k, "net_pnl": v["net_pnl"]} for k, v in exits.items()])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:980px; margin:2rem auto; padding:0 1rem; color:#1a1a1a; }}
 h1 {{ font-size:1.5rem; }} h2 {{ margin-top:1.6rem; border-bottom:1px solid #eee; }}
 table {{ border-collapse:collapse; margin:0.6rem 0; font-size:0.92rem; }}
 td,th {{ padding:4px 12px; text-align:left; border-bottom:1px solid #f0f0f0; }} th {{ color:#555; }}
 .verdict {{ background:#f6f8fa; border-left:4px solid #1f77b4; padding:0.8rem 1rem; border-radius:4px; line-height:1.5; }}
 .muted {{ color:#888; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<p class="muted">Template: {result['params']['template']} · owner tier {result['params']['owner_tier']} ·
{result['current_regime_windows']:,} current-regime windows (≥ {result['params']['regime_from']},
{result['params']['effective_latency_ms']} ms effective).</p>
<div class="verdict">{result['verdict']}</div>
<h2>Strategy comparison</h2>
{eh._table_html(cmp_rows, [("strategy", "strategy"), ("net_pnl", "net PnL $"), ("trades", "trades")])}
<h2>Clone-vs-owner calibration ladder</h2>
<p class="muted">{lad['note']}</p>
{eh._table_html(ladder_rows, [("#", "#"), ("quantity", "quantity"), ("value", "value $")])}
<h2>Tier honesty — Bronze vs owner tier</h2>
<p class="muted">{tiers.get('note', '')}</p>
{eh._table_html(tier_rows, [("tier", "tier"), ("rebate", "rebate $"), ("corrected_net", "corrected net $")])}
<h2>Momentum-exit vs hold-to-resolution</h2>
{eh._table_html(exit_rows, [("exit", "exit rule"), ("net_pnl", "net PnL $")])}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "backtest_clones")
    parser.add_argument("--clone", default="all", help="clone name (default all): " + ", ".join(CLONES))
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--network-ms", type=int, default=5)
    parser.add_argument("--venue-delay-ms", type=int, default=250)
    parser.add_argument("--fee-rate", type=float, default=mj.CRYPTO_FEE_RATE)
    parser.add_argument("--manuals-dir", default=None,
                        help="dir with <handle>.json manuals for the owner ladder (default out/competitors/manuals)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    manuals_dir = Path(args.manuals_dir) if args.manuals_dir else paths.out_dir / "competitors" / "manuals"
    clones = list(CLONES) if args.clone == "all" else [args.clone]
    for clone in clones:
        print(f"[backtest_clones] {clone}")
        result = backtest_clone(paths, clone, series=series, since_ms=since_ms, until_ms=until_ms,
                                seeds=args.seeds, seed=args.seed, network_ms=args.network_ms,
                                venue_delay_ms=args.venue_delay_ms, fee_rate=args.fee_rate,
                                manuals_dir=manuals_dir)
        c = result["clone"]
        print(f"[backtest_clones]   net ${c['net_pnl']:.2f} ({c['trades']:,} trades, win {c['win_rate']:.0%}) "
              f"over {result['current_regime_windows']:,} windows")
        procmem.report_peak_rss(f"backtest_clones:{clone}", result.get("peak_rss_mb"), args.max_rss_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
