"""Short-horizon **money judge** — the Phase-1 verdict on whether the 10-second
signal makes *simulated money* on our own recorded weeks.

This replays the recorded journal and, at each timestamp where the trained 10s GBT
model fires beyond a confidence threshold, simulates two trading actions against the
**recorded Polymarket book**, pays the **exact** documented taker fee, and marks the
position **to the window's actual resolution**:

- **(a) taker** — buy the model's side at the displayed ask, up to the displayed
  size, after a configurable latency, then hold to resolution ($1 on the winning side).
- **(b) pair-leg** — take the leaned-side fill, then try to complete the opposite leg
  from the book 5–15 s later; credit a locked pair when the combined cost < $1, and
  honestly book the stranded leg's loss when it never gets cheap enough before close.

It reports per-day and per-window simulated PnL (after fees), trade counts, and win
rate; **sweeps the confidence threshold**; and compares the ML signal against (1)
never trading and (2) the engine's own hand-coded momentum trigger (plus the
late-window certainty taker) evaluated on the *same* replay. It renders a
plain-language verdict.

**Honesty first.** The fires come from the **walk-forward out-of-sample** predictions
that ``learn_short_gbt`` already wrote (``oos_<scope>_fwd10_<variant>.parquet``) — never
in-sample, so the money verdict can't cheat. Down-side fills use the documented CLOB
mirror (the Up **bid** is the mirrored Down **ask**: price ``1 − bid``, size ``bid_sz``
— a real recorded order). Conservative by construction: when in doubt, fill less, pay
more (pay the ask, skip on a stale/absent book, floor at the $1 min notional). This
stage is **LightGBM-free** — it consumes the OOS parquet + the journal only.

Run:  ``python -m model_lab.money_judge``  (after ``learn_short_gbt`` has produced the
OOS parquet; honors ``--journal-dir/--out/--since/--until`` and the sim knobs).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import short_horizon as sh
from .config import (
    ParquetNotReady,
    Paths,
    assert_parquet_ready,
    resolve_bounds,
    resolve_paths,
    stage_parser,
)
from .io import journal as jio
from .lib import math as lm
from .lib import procmem

# --- engine-pinned constants (CLAUDE.md §8 / config/default.toml [engine.defaults]) --
CRYPTO_FEE_RATE = 0.07  # crypto taker feeRate; per-market override a follow-up
MOMENTUM_BUFFER = 0.005  # taker_momentum_buffer
SIGNAL_LOOKBACK_MS = 2000  # signal_lookback_ms
SIGNAL_SIGMA_MULT = 3.0  # signal_sigma_mult
COOLDOWN_MS = 5000  # taker_cooldown_ms
TAKER_REBATE = 0.0  # taker_rebate_pct (immaterial at this volume)
LATE_TAU_SECS = 30.0  # late_window_tau_secs
LATE_CERTAINTY = 0.97  # late_certainty_threshold
LATE_PRICE_CAP = 0.99  # late_taker_price_cap
MIN_NOTIONAL = 1.0  # $1 minimum resting/marketable notional

# --- sim knobs (defaults; overridable via CLI) --------------------------------
DEFAULT_LATENCY_MS = 200  # NETWORK latency (home REST p50 ≈ 180 ms; VPS-colocated ≈ 5 ms)
DEFAULT_VENUE_DELAY_MS = 250  # venue TAKER delay added on top of network for marketable
#   fills — the ~250 ms hold Polymarket imposes before a taker matches (`itode`, enabled on
#   all four series 2026-07-09). Effective taker fill latency = network + venue delay.
BOOK_STALENESS_MS = 2000  # no fresh book within this → no fill (conservative)
SIGNAL_BUCKET_MS = 100  # Binance-mid bucketing for the momentum ring
DEFAULT_WINDOW_BUDGET = 10.0  # taker_budget_per_window
DEFAULT_TRADE_BUDGET = 10.0  # per-fire notional cap
DEFAULT_COMP_MIN_S = 5.0  # pair-leg completion window start
DEFAULT_COMP_MAX_S = 15.0  # pair-leg completion window end
DEFAULT_LOCK_THRESHOLD = 1.0  # combined pair cost < this → lock
DEFAULT_CONV_TOL = 0.005  # momentum-exit: |PM mid − fair| ≤ this → converged (half a cent)
DEFAULT_EXIT_SCAN_MS = 1000  # momentum-exit forward-scan step (matches the 1 s snapshot grid)
MS_PER_DAY = 86_400_000  # UTC DayKey (mirrors crates/analytics)
SWEEP_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]

CAVEATS = [
    "Single displayed level only (top-of-book): a large fire could in reality sweep "
    "deeper unseen levels — conservative on depth.",
    "Fills win the full displayed size at the recorded price (the engine's own paper "
    "taker model for a single actor); real queue competition would fill less.",
    "Binance mid bucketed to 100 ms and the momentum trigger evaluated only at journaled "
    "model snapshots (~1/s) — fewer / possibly-divergent momentum fires than the live engine.",
    "Effective taker fill latency = network + venue delay (see params.network_ms / "
    "venue_delay_ms / effective_latency_ms). A too-short latency fills a staler, less-repriced "
    f"book and OVERSTATES the edge; a {BOOK_STALENESS_MS} ms book-staleness skip biases toward "
    "fewer trades.",
    "A flat venue delay is only correct WITHIN one regime. The full-history run mixes the "
    "pre/post-delay periods (and should use venue_delay=0); read the current-regime slice "
    "(--since 2026-06-05) for the honest verdict.",
    f"Fee rate {CRYPTO_FEE_RATE} (config default; the per-market fees.rate is a follow-up).",
    "Late-window baseline needs Chainlink-anchored snapshots — absent in the synthetic fixture.",
    "Float parsing (research, not accounting); the 5-dp fee rounding is replicated at the money boundary.",
    "Mark-to-resolution holds the taker to settlement (no early exit); the pair-leg models locking instead.",
]


# ===========================================================================
# data loading
# ===========================================================================
def _load_fires(paths: Paths, *, scope: str, variant: str, source: str) -> pd.DataFrame:
    """The model firings = the walk-forward OOS predictions ``learn_short_gbt`` wrote.

    Reads ``out/learn_short_gbt/oos_{scope}_fwd10_{variant}.parquet`` (guarded), keeps
    the harness grid keys + ``p_up``, and filters to ``label_source == source`` (the
    ``chainlink`` windows are our recorded weeks — the only ones with a real book + real
    resolution). Returns ``series, window_open_ms, sample_ts_ms, p_up``.
    """
    path = paths.out_dir / "learn_short_gbt" / f"oos_{scope}_fwd10_{variant}.parquet"
    assert_parquet_ready(path, label=f"oos_{scope}_fwd10_{variant}.parquet", min_rows=1)
    df = pd.read_parquet(path)
    if "label_source" in df.columns and source:
        df = df[df["label_source"] == source]
    return df[["series", "window_open_ms", "sample_ts_ms", "p_up"]].reset_index(drop=True)


def _read_signal_journal(
    paths: Paths,
    *,
    since_ms: int | None,
    until_ms: int | None,
    need_model: bool,
    need_mids: bool,
) -> tuple[list[dict], dict, dict, dict, dict]:
    """One streaming journal pass for the parts ``eval_harness`` drops.

    Returns ``(model_rows, win_meta, outcome_resolved, outcome_settle, mids_by_asset)``.
    ``model_rows`` carries ``sigma_1s``/``anchor`` (which ``eh.read_journal`` omits) for
    the momentum + late-window triggers; the ``win_meta`` / outcome dicts feed
    ``eh.windows_frame`` verbatim; ``mids_by_asset[asset] = (ts[], px[])`` are the
    ``BinanceDirect`` mids bucketed to ``SIGNAL_BUCKET_MS`` (last per bucket) for the
    momentum signal ring — an approximation of the engine's sub-second ring.
    """
    model_rows: list[dict] = []
    win_meta: dict[tuple[str, int], dict] = {}
    outcome_resolved: dict[tuple[str, int], str] = {}
    outcome_settle: dict[tuple[str, int], str] = {}
    mid_bucket: dict[str, dict[int, float]] = defaultdict(dict)  # asset -> {bucket: last px}

    from .config import _in_bounds

    for rec in jio.read_records(paths.journal_dir):
        kind = rec.get("type")
        if kind == "model" and need_model:
            window = rec.get("window")
            ts = rec.get("ts")
            if not window or not _in_bounds(int(ts) if ts is not None else None, since_ms, until_ms):
                continue
            series, open_time = jio.window_key(window)
            model_rows.append(
                {
                    "series": series,
                    "open_time": open_time,
                    "asset": jio.asset_key(rec.get("asset")),
                    "ts": int(ts),
                    "p_up": jio.to_float(rec.get("p_up")),
                    "sigma_1s": jio.to_float(rec.get("sigma_1s")),
                    "anchor": rec.get("anchor"),
                    "health": rec.get("health"),
                }
            )
        elif kind == "price_tick" and need_mids:
            if rec.get("source") != "BinanceDirect" or rec.get("kind") != "Mid":
                continue
            ts_local = rec.get("ts_local")
            if ts_local is None or not _in_bounds(int(ts_local), since_ms, until_ms):
                continue
            px = jio.to_float(rec.get("value"))
            if not (math.isfinite(px) and px > 0.0):
                continue
            asset = jio.asset_key(rec.get("asset"))
            mid_bucket[asset][int(ts_local) // SIGNAL_BUCKET_MS] = px  # keep last in bucket
        elif kind == "window":
            market = rec.get("market") or {}
            key = jio.window_key(market.get("window"))
            if key == ("", 0) or not _in_bounds(key[1], since_ms, until_ms):
                continue
            if key not in win_meta:
                tokens = market.get("tokens") or {}
                win_meta[key] = {
                    "close_time": market.get("close_time"),
                    "up_token": tokens.get("up"),
                    "down_token": tokens.get("down"),
                }
            lifecycle = rec.get("lifecycle")
            if isinstance(lifecycle, dict) and "Resolved" in lifecycle:
                outcome = (lifecycle["Resolved"] or {}).get("outcome")
                if outcome in ("Up", "Down"):
                    outcome_resolved[key] = outcome
        elif kind == "settlement":
            key = jio.window_key(rec.get("window"))
            outcome = rec.get("outcome")
            if key != ("", 0) and outcome in ("Up", "Down") and _in_bounds(key[1], since_ms, until_ms):
                outcome_settle[key] = outcome

    mids_by_asset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for asset, buckets in mid_bucket.items():
        if not buckets:
            continue
        items = sorted(buckets.items())
        ts = np.array([b * SIGNAL_BUCKET_MS for b, _ in items], dtype=np.int64)
        px = np.array([v for _, v in items], dtype=float)
        mids_by_asset[asset] = (ts, px)
    return model_rows, win_meta, outcome_resolved, outcome_settle, mids_by_asset


def _build_book_arrays(pm: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per-token sorted arrays for O(log n) as-of lookups from ``_read_pm_books``."""
    books: dict[str, dict[str, np.ndarray]] = {}
    if pm.empty:
        return books
    for token, g in pm.sort_values("ts").groupby("token_id"):
        books[token] = {
            "ts": g["ts"].to_numpy(dtype=np.int64),
            "bid_px": g["bid_px"].to_numpy(dtype=float),
            "bid_sz": g["bid_sz"].to_numpy(dtype=float),
            "ask_px": g["ask_px"].to_numpy(dtype=float),
            "ask_sz": g["ask_sz"].to_numpy(dtype=float),
        }
    return books


# ===========================================================================
# sim primitives
# ===========================================================================
def _book_asof(books: dict, token, at_ms: int, staleness_ms: int = BOOK_STALENESS_MS):
    """The most recent recorded book for ``token`` at or before ``at_ms`` (backward
    as-of). Returns ``(bid_px, bid_sz, ask_px, ask_sz)`` or ``None`` if there is no
    fresh book within ``staleness_ms`` (conservative no-fill)."""
    b = books.get(token)
    if b is None:
        return None
    ts = b["ts"]
    i = int(np.searchsorted(ts, at_ms, side="right")) - 1
    if i < 0 or (at_ms - int(ts[i])) > staleness_ms:
        return None
    return float(b["bid_px"][i]), float(b["bid_sz"][i]), float(b["ask_px"][i]), float(b["ask_sz"][i])


def _leaned_quote(book_row, side: str):
    """The price + displayed size to BUY ``side``, from a single-level book row.

    Up → the Up-token ask ``(ask_px, ask_sz)``. Down → the CLOB mirror: the Up **bid**
    is the mirrored Down **ask**, so buying Down costs ``1 − bid_px`` for ``bid_sz``
    shares (a real recorded order). Returns ``(price, avail)`` or ``None`` if the quote
    is unusable (price outside ``(0,1)`` or non-positive size)."""
    bid_px, bid_sz, ask_px, ask_sz = book_row
    if side == "Up":
        price, avail = ask_px, ask_sz
    elif side == "Down":
        price, avail = 1.0 - bid_px, bid_sz
    else:
        return None
    if not (math.isfinite(price) and 0.0 < price < 1.0 and math.isfinite(avail) and avail > 0.0):
        return None
    return price, avail


def _fill_shares(price: float, avail: float, budget_left: float, trade_budget: float) -> float:
    """Conservative fill: the min of displayed size, the per-fire budget, and the
    per-window budget remaining. Below the $1 min notional → 0 (skip)."""
    shares = min(avail, trade_budget / price, budget_left / price)
    if shares <= 0.0 or shares * price < MIN_NOTIONAL:
        return 0.0
    return shares


def _mark_to_resolution(side: str, shares: float, price: float, outcome_up, fee_rate: float) -> dict:
    """Buy ``shares`` of ``side`` at ``price`` (+ fee), settle at $1 on the winner.

    ``outcome_up`` is strictly 0/1 (``windows_frame`` already encoded resolution ties as
    Up). Returns the trade dict with ``net = payoff − cost``."""
    fee = lm.taker_fee(shares, fee_rate, price)
    cost = shares * price + fee
    won = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
    payoff = shares * 1.0 if won else 0.0
    return {
        "side": side, "shares": shares, "price": price, "fee": fee,
        "cost": cost, "payoff": payoff, "net": payoff - cost, "won": bool(won),
    }


def _side_of(p_up: float, threshold: float) -> str | None:
    """The leaned side when the model fires beyond ``threshold`` (else ``None``)."""
    if not math.isfinite(p_up):
        return None
    if p_up >= 0.5 + threshold:
        return "Up"
    if p_up <= 0.5 - threshold:
        return "Down"
    return None


# --- sell / exit primitives (mirror of the BUY primitives above) --------------
# The existing `_leaned_quote`/`_fill_shares` model BUYS only. Closing a taker
# position early (the momentum-EXIT variant, D) needs to SELL back into the book —
# these helpers are the exact mirror and are additive (no buy path changes).
def _leaned_sell_quote(book_row, side: str):
    """The price + displayed size to SELL an existing ``side`` position, from a
    single-level book row ``(bid_px, bid_sz, ask_px, ask_sz)``.

    Up  → hit the Up bid: receive ``bid_px`` for up to ``bid_sz`` shares.
    Down → the CLOB mirror: the Down **bid** is ``1 − Up ask``, so selling Down
    receives ``1 − ask_px`` for up to ``ask_sz`` shares (a real recorded order).
    Returns ``(price, avail)`` or ``None`` if the quote is unusable."""
    bid_px, bid_sz, ask_px, ask_sz = book_row
    if side == "Up":
        price, avail = bid_px, bid_sz
    elif side == "Down":
        price, avail = 1.0 - ask_px, ask_sz
    else:
        return None
    if not (math.isfinite(price) and 0.0 < price < 1.0 and math.isfinite(avail) and avail > 0.0):
        return None
    return price, avail


def _sell_at_bid(books: dict, token, at_ms: int, side: str, shares: float, fee_rate: float,
                 staleness_ms: int = BOOK_STALENESS_MS):
    """Sell up to ``shares`` of ``side`` into the recorded book as-of ``at_ms``.

    Returns ``{exit_price, exit_shares, sell_fee, proceeds, unsold}`` (a taker fee is
    paid on the exit leg too), or ``None`` if there is no fresh/usable book (no exit
    possible — the caller then holds the remainder to resolution). Conservative: sells
    at most the displayed size."""
    book = _book_asof(books, token, at_ms, staleness_ms)
    if book is None:
        return None
    quote = _leaned_sell_quote(book, side)
    if quote is None:
        return None
    price, avail = quote
    sell_shares = min(shares, avail)
    if sell_shares <= 0.0:
        return None
    proceeds = sell_shares * price
    sell_fee = lm.taker_fee(sell_shares, fee_rate, price)
    return {"exit_price": price, "exit_shares": sell_shares, "sell_fee": sell_fee,
            "proceeds": proceeds, "unsold": shares - sell_shares}


def _mark_with_momentum_exit(
    side: str, shares: float, entry_price: float, entry_fee: float, books: dict, token,
    entry_ts: int, fair_fn, close_ms: int, outcome_up, *, fee_rate: float, timeout_s: float,
    conv_tol: float = DEFAULT_CONV_TOL, scan_step_ms: int = DEFAULT_EXIT_SCAN_MS,
    staleness_ms: int = BOOK_STALENESS_MS,
) -> dict:
    """Round-trip a momentum taker BUY with an early SELL exit (variant D).

    Scan the recorded book forward from ``entry_ts`` (step ``scan_step_ms``) up to
    ``min(entry_ts + timeout_s, close_ms)``. Exit at the first second where the
    side-oriented PM mid converges to the reconstructed side-oriented fair
    (``|pm_mid_side − fair| ≤ conv_tol``); else force-exit at the deadline (timeout or
    close). ``fair_fn(t) -> side-oriented fair | None`` is a caller-supplied as-of
    lookup over the window's reconstructed snapshots. At exit, ``_sell_at_bid`` sells
    into the book (taker fee both ways); any part that cannot be sold (no fresh book,
    or displayed size < position) is marked to the window's resolution.

    Returns a **documented superset** of the ``_TRADE_COLS`` fields — the base 14
    (``side, shares, price=entry, fee=both-way, cost, payoff, net, won`` + caller-added
    keys) plus ``exit_ts_ms, exit_price, exit_reason ∈ {converged,timeout,close,
    resolution}``. Note ``price`` = the ENTRY price and the row carries one wV leg, so
    rebate_sim's tier volume is entry-only (slightly conservative); the ``fee`` and
    ``net`` are exact both-way."""
    entry_cost = shares * entry_price + entry_fee
    timeout_ms = int(timeout_s * 1000)
    deadline = min(entry_ts + timeout_ms, close_ms)
    won = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)

    exit_ts, exit_reason = None, None
    t = entry_ts + scan_step_ms
    while t <= deadline:
        book = _book_asof(books, token, t, staleness_ms)
        if book is not None:
            bid_px, _bid_sz, ask_px, _ask_sz = book
            if math.isfinite(bid_px) and math.isfinite(ask_px) and 0.0 < bid_px and ask_px < 1.0:
                pm_mid = 0.5 * (bid_px + ask_px)  # Up-token mid
                pm_mid_side = pm_mid if side == "Up" else 1.0 - pm_mid
                fair = fair_fn(t)
                if fair is not None and math.isfinite(fair) and abs(pm_mid_side - fair) <= conv_tol:
                    exit_ts, exit_reason = t, "converged"
                    break
        t += scan_step_ms
    if exit_ts is None:
        exit_ts = deadline
        exit_reason = "timeout" if (entry_ts + timeout_ms) <= close_ms else "close"

    sell = _sell_at_bid(books, token, exit_ts, side, shares, fee_rate, staleness_ms)
    if sell is None:
        # Could not sell → hold everything to resolution.
        payoff = shares * 1.0 if won else 0.0
        net = payoff - entry_cost
        return {"side": side, "shares": shares, "price": entry_price, "fee": entry_fee,
                "cost": entry_cost, "payoff": payoff, "net": net, "won": bool(net > 0.0),
                "exit_ts_ms": int(exit_ts), "exit_price": float("nan"), "exit_reason": "resolution"}
    unsold = sell["unsold"]
    unsold_payoff = unsold * 1.0 if won else 0.0
    total_payoff = sell["proceeds"] + unsold_payoff
    sell_fee = sell["sell_fee"]
    cost = entry_cost + sell_fee
    net = total_payoff - cost
    return {"side": side, "shares": shares, "price": entry_price, "fee": entry_fee + sell_fee,
            "cost": cost, "payoff": total_payoff, "net": net, "won": bool(net > 0.0),
            "exit_ts_ms": int(exit_ts), "exit_price": sell["exit_price"], "exit_reason": exit_reason}


# ===========================================================================
# ML taker (action a)
# ===========================================================================
def _run_taker(
    fires: pd.DataFrame, *, threshold: float, latency_ms: int, fee_rate: float,
    window_budget: float, trade_budget: float, books: dict,
) -> tuple[list[dict], int]:
    """Simulate the taker action over the fires. Returns ``(trade_rows, n_fires)``.

    Fires are processed chronologically per window with a per-window budget. A fire
    with no fresh book at ``sample_ts + latency`` (or a below-min-notional size) counts
    toward ``n_fires`` but produces no trade (conservative skip)."""
    rows: list[dict] = []
    n_fires = 0
    for (series, open_ms), g in fires.groupby(["series", "window_open_ms"], sort=False):
        budget_left = window_budget
        for r in g.sort_values("sample_ts_ms").itertuples(index=False):
            side = _side_of(float(r.p_up), threshold)
            if side is None:
                continue
            n_fires += 1
            if budget_left < MIN_NOTIONAL:
                continue
            book = _book_asof(books, r.up_token, int(r.sample_ts_ms) + latency_ms)
            if book is None:
                continue
            quote = _leaned_quote(book, side)
            if quote is None:
                continue
            price, avail = quote
            shares = _fill_shares(price, avail, budget_left, trade_budget)
            if shares <= 0.0:
                continue
            trade = _mark_to_resolution(side, shares, price, int(r.outcome_up), fee_rate)
            budget_left -= shares * price
            rows.append({
                "series": series, "window_open_ms": int(open_ms),
                "sample_ts_ms": int(r.sample_ts_ms), "day": int(open_ms) // MS_PER_DAY,
                "p_up": float(r.p_up), "outcome_up": int(r.outcome_up), **trade,
            })
    return rows, n_fires


# ===========================================================================
# pair-leg (action b)
# ===========================================================================
def _completion_leg(book: dict, side_s: str, price_s: float, lo_ms: int, hi_ms: int,
                    lock_threshold: float):
    """Scan the recorded book in ``[lo_ms, hi_ms]`` for the first opposite-leg quote
    that locks a pair (``price_s + price_o < lock_threshold``). Returns
    ``(price_o, size_o)`` or ``None``."""
    ts = book["ts"]
    lo = int(np.searchsorted(ts, lo_ms, side="left"))
    hi = int(np.searchsorted(ts, hi_ms, side="right"))
    for i in range(lo, hi):
        if side_s == "Up":  # complete Down = 1 − Up bid
            price_o, size_o = 1.0 - float(book["bid_px"][i]), float(book["bid_sz"][i])
        else:  # side_s == "Down": complete Up = Up ask
            price_o, size_o = float(book["ask_px"][i]), float(book["ask_sz"][i])
        if not (math.isfinite(price_o) and 0.0 < price_o < 1.0 and size_o > 0.0):
            continue
        if price_s + price_o < lock_threshold:
            return price_o, size_o
    return None


def _run_pairleg(
    fires: pd.DataFrame, *, threshold: float, latency_ms: int, fee_rate: float,
    window_budget: float, trade_budget: float, comp_min_s: float, comp_max_s: float,
    lock_threshold: float, books: dict,
) -> tuple[list[dict], int]:
    """Simulate the pair-leg action: leaned fill, then complete the opposite leg from
    the book 5–15 s later; lock what completes and strand the rest to resolution."""
    rows: list[dict] = []
    n_fires = 0
    for (series, open_ms), g in fires.groupby(["series", "window_open_ms"], sort=False):
        budget_left = window_budget
        for r in g.sort_values("sample_ts_ms").itertuples(index=False):
            side = _side_of(float(r.p_up), threshold)
            if side is None:
                continue
            n_fires += 1
            if budget_left < MIN_NOTIONAL:
                continue
            fire_ts = int(r.sample_ts_ms)
            book_row = _book_asof(books, r.up_token, fire_ts + latency_ms)
            if book_row is None:
                continue
            quote = _leaned_quote(book_row, side)
            if quote is None:
                continue
            price_s, avail = quote
            n = _fill_shares(price_s, avail, budget_left, trade_budget)
            if n <= 0.0:
                continue
            budget_left -= n * price_s
            outcome_up = int(r.outcome_up)
            close_ms = int(r.close_time)
            fee_s = lm.taker_fee(n, fee_rate, price_s)

            # Completion scan over [fire+comp_min, min(fire+comp_max, close)].
            m, price_o, fee_o = 0.0, float("nan"), 0.0
            lo_ms = fire_ts + int(comp_min_s * 1000)
            hi_ms = min(fire_ts + int(comp_max_s * 1000), close_ms)
            book = books.get(r.up_token)
            if book is not None and hi_ms > lo_ms:
                comp = _completion_leg(book, side, price_s, lo_ms, hi_ms, lock_threshold)
                if comp is not None:
                    price_o, size_o = comp
                    m = min(n, size_o)
                    fee_o = lm.taker_fee(m, fee_rate, price_o)

            stranded = n - m
            s_wins = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
            cost = n * price_s + fee_s + m * price_o + fee_o if m > 0.0 else n * price_s + fee_s
            payoff = m * 1.0 + stranded * (1.0 if s_wins else 0.0)
            net = payoff - cost
            # Reporting attribution: locked-pair PnL vs stranded-leg PnL (the leaned fee
            # is charged once on the full n; split it pro-rata for the two components).
            fee_s_locked = fee_s * (m / n) if n > 0 else 0.0
            locked_pnl = (m * 1.0 - (m * price_s + m * price_o + fee_s_locked + fee_o)) if m > 0.0 else 0.0
            stranded_pnl = net - locked_pnl
            rows.append({
                "series": series, "window_open_ms": int(open_ms), "sample_ts_ms": fire_ts,
                "day": int(open_ms) // MS_PER_DAY, "side": side, "n": n, "locked": m,
                "stranded": stranded, "price_s": price_s, "price_o": price_o,
                "fee": fee_s + fee_o, "payoff": payoff, "net": net,
                "locked_pnl": locked_pnl, "stranded_pnl": stranded_pnl, "won": bool(net > 0.0),
            })
    return rows, n_fires


# ===========================================================================
# momentum baseline (the engine's hand-coded trigger, fixed logic)
# ===========================================================================
def _confirmed_direction(mids, now_ms: int, sigma_1s: float) -> str | None:
    """Port of ``crates/engine/src/taker/signal.rs::confirmed_direction``.

    ≥2 usable (positive-price) Binance mids in ``[now − 2000, now]``; over their actual
    span ``T`` fire iff ``|ln(S_new/S_old)| > 3.0·σ_1s·√T``; direction by the sign."""
    if mids is None or not (math.isfinite(sigma_1s) and sigma_1s > 0.0):
        return None
    ts, px = mids
    cutoff = now_ms - SIGNAL_LOOKBACK_MS
    lo = int(np.searchsorted(ts, cutoff, side="left"))
    hi = int(np.searchsorted(ts, now_ms, side="right"))
    if hi - lo < 2:
        return None
    ts_w, px_w = ts[lo:hi], px[lo:hi]
    mask = px_w > 0.0
    if int(np.count_nonzero(mask)) < 2:
        return None
    ts_u, px_u = ts_w[mask], px_w[mask]
    t_secs = (int(ts_u[-1]) - int(ts_u[0])) / 1000.0
    if t_secs <= 0.0:
        return None
    s_old, s_new = float(px_u[0]), float(px_u[-1])
    if s_old <= 0.0 or s_new <= 0.0:
        return None
    r = math.log(s_new / s_old)
    if not math.isfinite(r):
        return None
    if abs(r) > SIGNAL_SIGMA_MULT * sigma_1s * math.sqrt(t_secs):
        return "Up" if r > 0.0 else "Down"
    return None


def _plan_take_single_level(fair: float, price: float, avail: float, remaining: float,
                            fee_rate: float, *, buffer: float = MOMENTUM_BUFFER,
                            rebate: float = TAKER_REBATE) -> float:
    """Port of ``crates/engine/src/taker/edge.rs::plan_take`` on a single displayed
    level. Returns the shares to take (0 if the edge gate rejects)."""
    if not (math.isfinite(fair) and math.isfinite(price) and 0.0 < price < 1.0):
        return 0.0
    if fair <= price:  # NoTakeReason::BookAlreadyRepriced
        return 0.0
    edge = fair - price
    fee_ps = fee_rate * price * (1.0 - price)
    if edge <= fee_ps * (1.0 - rebate) + buffer:  # EdgeBelowFeePlusBuffer (strict >)
        return 0.0
    shares = min(avail, remaining / price)
    if shares <= 0.0 or shares * price < MIN_NOTIONAL:
        return 0.0
    return shares


def _run_momentum(
    model_rows: list[dict], mids_by_asset: dict, win_by_key: dict, books: dict, *,
    latency_ms: int, fee_rate: float, window_budget: float, invert: bool = False,
) -> list[dict]:
    """The engine's momentum taker replayed over journaled model snapshots — the full
    gate ladder (Ready, window match, usable fair/vol, τ>0, confirmed move, cooldown,
    budget, book, edge gate) → a taker buy marked to resolution.

    ``invert=True`` flips the confirmed direction (bet AGAINST the signal) through the same
    edge discipline — a forensic falsification: a real directional edge should collapse."""
    rows: list[dict] = []
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for m in model_rows:
        by_win[(m["series"], m["open_time"])].append(m)
    for key, snaps in by_win.items():
        win = win_by_key.get(key)
        if win is None:
            continue
        up_token, close_ms, outcome_up = win["up_token"], win["close_time"], win["outcome_up"]
        budget_left, last_take = window_budget, None
        for m in sorted(snaps, key=lambda s: s["ts"]):
            now = m["ts"]
            p_up, sigma = m["p_up"], m["sigma_1s"]
            if m["health"] != "Ready":
                continue
            if not (math.isfinite(p_up) and 0.0 < p_up < 1.0):
                continue
            if not (math.isfinite(sigma) and sigma > 0.0):
                continue
            if (close_ms - now) / 1000.0 <= 0.0:  # τ > 0
                continue
            direction = _confirmed_direction(mids_by_asset.get(m["asset"]), now, sigma)
            if direction is None:
                continue
            if invert:  # forensic: bet against the signal
                direction = "Down" if direction == "Up" else "Up"
            if last_take is not None and now - last_take < COOLDOWN_MS:
                continue
            if budget_left < MIN_NOTIONAL:
                continue
            book = _book_asof(books, up_token, now + latency_ms)
            if book is None:
                continue
            quote = _leaned_quote(book, direction)
            if quote is None:
                continue
            price, avail = quote
            fair = p_up if direction == "Up" else 1.0 - p_up
            shares = _plan_take_single_level(fair, price, avail, budget_left, fee_rate)
            if shares <= 0.0:
                continue
            trade = _mark_to_resolution(direction, shares, price, outcome_up, fee_rate)
            budget_left -= shares * price
            last_take = now
            rows.append({
                "series": key[0], "window_open_ms": int(key[1]), "sample_ts_ms": int(now),
                "day": int(key[1]) // MS_PER_DAY, "p_up": p_up, "outcome_up": outcome_up, **trade,
            })
    return rows


# ===========================================================================
# late-window certainty baseline (bonus)
# ===========================================================================
def _run_late_window(
    model_rows: list[dict], win_by_key: dict, books: dict, *,
    latency_ms: int, fee_rate: float, window_budget: float,
) -> list[dict]:
    """The engine's late-window certainty taker replayed: Ready + Chainlink-anchored +
    τ ≤ 30 s + p_up beyond the 0.97 certainty band → buy the certain side at asks ≤ 0.99
    (no edge/fee gate) marked to resolution."""
    rows: list[dict] = []
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for m in model_rows:
        by_win[(m["series"], m["open_time"])].append(m)
    for key, snaps in by_win.items():
        win = win_by_key.get(key)
        if win is None:
            continue
        up_token, close_ms, outcome_up = win["up_token"], win["close_time"], win["outcome_up"]
        budget_left, last_take = window_budget, None
        for m in sorted(snaps, key=lambda s: s["ts"]):
            now, p_up = m["ts"], m["p_up"]
            if m["health"] != "Ready" or m["anchor"] != "Chainlink":
                continue
            if not (math.isfinite(p_up) and 0.0 <= p_up <= 1.0):
                continue
            tau = (close_ms - now) / 1000.0
            if not (0.0 < tau <= LATE_TAU_SECS):
                continue
            if p_up >= LATE_CERTAINTY:
                direction = "Up"
            elif p_up <= 1.0 - LATE_CERTAINTY:
                direction = "Down"
            else:
                continue
            if last_take is not None and now - last_take < COOLDOWN_MS:
                continue
            if budget_left < MIN_NOTIONAL:
                continue
            book = _book_asof(books, up_token, now + latency_ms)
            if book is None:
                continue
            quote = _leaned_quote(book, direction)
            if quote is None:
                continue
            price, avail = quote
            if price > LATE_PRICE_CAP:  # AllAsksAbovePriceCap
                continue
            shares = min(avail, budget_left / price)
            if shares <= 0.0 or shares * price < MIN_NOTIONAL:
                continue
            trade = _mark_to_resolution(direction, shares, price, outcome_up, fee_rate)
            budget_left -= shares * price
            last_take = now
            rows.append({
                "series": key[0], "window_open_ms": int(key[1]), "sample_ts_ms": int(now),
                "day": int(key[1]) // MS_PER_DAY, "p_up": p_up, "outcome_up": outcome_up, **trade,
            })
    return rows


# ===========================================================================
# aggregation, sweep, verdict
# ===========================================================================
def _metrics_for_trades(rows: list[dict]) -> dict:
    """PnL roll-up for a set of trade rows (taker, momentum, or pair-leg)."""
    n = len(rows)
    if n == 0:
        return {"n_trades": 0, "filled_shares": 0.0, "gross_pnl": 0.0, "fees": 0.0,
                "net_pnl": 0.0, "pnl_per_trade": float("nan"), "win_rate": float("nan")}
    fees = float(sum(r["fee"] for r in rows))
    net = float(sum(r["net"] for r in rows))
    shares = float(sum(r.get("shares", r.get("n", 0.0)) for r in rows))
    notional = float(sum(r.get("shares", r.get("n", 0.0)) * r.get("price", r.get("price_s", 0.0))
                         for r in rows))
    wins = int(sum(1 for r in rows if r["won"]))
    return {"n_trades": n, "filled_shares": shares, "gross_pnl": float(sum(r["payoff"] for r in rows)) - notional,
            "fees": fees, "net_pnl": net, "pnl_per_trade": net / n, "win_rate": wins / n}


def _sweep(fires: pd.DataFrame, *, thresholds, latency_ms, fee_rate, window_budget,
           trade_budget, comp_min_s, comp_max_s, lock_threshold, books) -> list[dict]:
    """Run the taker + pair-leg over each confidence threshold."""
    out: list[dict] = []
    for thr in thresholds:
        tk, n_fires = _run_taker(fires, threshold=thr, latency_ms=latency_ms, fee_rate=fee_rate,
                                 window_budget=window_budget, trade_budget=trade_budget, books=books)
        pl, _ = _run_pairleg(fires, threshold=thr, latency_ms=latency_ms, fee_rate=fee_rate,
                             window_budget=window_budget, trade_budget=trade_budget,
                             comp_min_s=comp_min_s, comp_max_s=comp_max_s,
                             lock_threshold=lock_threshold, books=books)
        tk_m, pl_m = _metrics_for_trades(tk), _metrics_for_trades(pl)
        out.append({
            "threshold": float(thr), "n_fires": int(n_fires),
            "taker": tk_m, "taker_rows": tk,
            "pairleg": {**pl_m,
                        "locked_pnl": float(sum(r["locked_pnl"] for r in pl)),
                        "stranded_pnl": float(sum(r["stranded_pnl"] for r in pl))},
            "pairleg_rows": pl,
        })
    return out


def _best_threshold(sweep: list[dict], *, min_trades: int = 1) -> dict | None:
    """Argmax taker net PnL among thresholds with ≥ ``min_trades`` trades; tie-break to
    the **higher** threshold (fewer, more-confident trades — the conservative pick)."""
    cands = [s for s in sweep if s["taker"]["n_trades"] >= min_trades]
    if not cands:
        return None
    best = max(cands, key=lambda s: (s["taker"]["net_pnl"], s["threshold"]))
    return best


def _rollup_per_day(rows: list[dict], source: str) -> list[dict]:
    by_day: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    out: list[dict] = []
    for day in sorted(by_day):
        g = by_day[day]
        date_iso = datetime.fromtimestamp(day * 86400, tz=timezone.utc).date().isoformat()
        wins = sum(1 for r in g if r["won"])
        out.append({
            "date": date_iso, "source": source, "trades": len(g),
            "win_rate": wins / len(g) if g else float("nan"),
            "net_pnl": float(sum(r["net"] for r in g)),
            "fees": float(sum(r["fee"] for r in g)),
        })
    return out


def _rollup_per_window(rows: list[dict]) -> list[dict]:
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_win[(r["series"], r["window_open_ms"])].append(r)
    out: list[dict] = []
    for (series, open_ms) in sorted(by_win):
        g = by_win[(series, open_ms)]
        out.append({"series": series, "window_open_ms": int(open_ms), "trades": len(g),
                    "net_pnl": float(sum(r["net"] for r in g))})
    return out


def _lift_over_majority(rows: list[dict]) -> dict:
    """Model dir-acc/Brier vs the always-predict-the-common-outcome baseline, on the
    fired set — model skill as lift. Scored against the **window outcome** (what we
    settle on): does the leaned side win vs always betting the common outcome?"""
    if not rows:
        return {"n": 0, "model": {"diracc": float("nan"), "brier": float("nan")},
                "majority": lm.majority_baseline(np.empty(0)),
                "diracc_lift": float("nan"), "brier_lift": float("nan")}
    p = np.array([r["p_up"] for r in rows], dtype=float)
    y = np.array([r["outcome_up"] for r in rows], dtype=float)
    maj = lm.majority_baseline(y)
    model_diracc = lm.directional_accuracy(p, y)
    model_brier = lm.brier_score(p, y)
    return {"n": len(rows), "model": {"diracc": model_diracc, "brier": model_brier},
            "majority": maj, "diracc_lift": model_diracc - maj["diracc"],
            "brier_lift": maj["brier"] - model_brier}


def _verdict(best, comparison, lift, n_windows_traded) -> str:
    if best is None:
        return ("No fires produced a trade against the recorded book — the signal makes "
                "no simulated money to judge (verdict: inconclusive; likely too few "
                "recorded windows or no fresh book at fire time).")
    tk = best["taker"]
    thr = best["threshold"]
    makes = tk["net_pnl"] > 0.0
    beats_mom = tk["net_pnl"] > comparison["momentum_net"]
    lead = "MAKES money" if makes else "LOSES money"
    parts = [
        f"On our recorded weeks the 10s signal {lead} after fees: at the best confidence "
        f"threshold |p−0.5| ≥ {thr:g}, the taker nets ${tk['net_pnl']:.2f} over "
        f"{tk['n_trades']} trade(s) across {n_windows_traded} window(s) "
        f"(win rate {tk['win_rate']:.0%}, ${tk['pnl_per_trade']:.4f}/trade, ${tk['fees']:.2f} fees).",
        f"Never-trading nets $0.00; the engine's momentum trigger nets "
        f"${comparison['momentum_net']:.2f}; the late-window taker nets "
        f"${comparison['late_window_net']:.2f}.",
        f"The signal {'beats' if beats_mom else 'does not beat'} the hand-coded momentum baseline.",
    ]
    if math.isfinite(lift.get("diracc_lift", float("nan"))):
        parts.append(
            f"On the fired set the leaned side wins {lift['model']['diracc']:.0%} of windows vs "
            f"{lift['majority']['diracc']:.0%} for always-betting-the-common-outcome "
            f"(dir-acc lift {lift['diracc_lift']:+.1%}).")
    if not makes:
        parts.append("Verdict: not yet — the edge does not clear fees + adverse book pricing "
                     "on this sample.")
    else:
        parts.append("Verdict: promising — but read the caveats and confirm on more recorded weeks.")
    return " ".join(parts)


# ===========================================================================
# report
# ===========================================================================
def _sweep_png(sweep: list[dict], comparison: dict) -> str | None:
    if not sweep:
        return None
    import matplotlib.pyplot as plt

    thr = [s["threshold"] for s in sweep]
    tk = [s["taker"]["net_pnl"] for s in sweep]
    pl = [s["pairleg"]["net_pnl"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(thr, tk, "o-", color="#1f77b4", label="ML taker net")
    ax.plot(thr, pl, "s--", color="#9467bd", label="ML pair-leg net")
    ax.axhline(0.0, color="#333", linewidth=0.8, label="never trading ($0)")
    ax.axhline(comparison["momentum_net"], color="#2ca02c", linestyle=":", label="momentum")
    ax.axhline(comparison["late_window_net"], color="#ff7f0e", linestyle=":", label="late-window")
    ax.set_xlabel("confidence threshold |p − 0.5|")
    ax.set_ylabel("net PnL after fees ($)")
    ax.set_title("Simulated PnL vs confidence threshold")
    ax.legend(fontsize=8)
    return eh._png(fig)


def _build_html(metrics: dict) -> str:
    p = metrics["params"]
    sweep_rows = [{
        "threshold": s["threshold"], "n_fires": s["n_fires"],
        "taker_trades": s["taker"]["n_trades"], "taker_net": s["taker"]["net_pnl"],
        "taker_fees": s["taker"]["fees"], "taker_ppt": s["taker"]["pnl_per_trade"],
        "taker_win": s["taker"]["win_rate"], "pairleg_trades": s["pairleg"]["n_trades"],
        "pairleg_locked": s["pairleg"]["locked_pnl"], "pairleg_stranded": s["pairleg"]["stranded_pnl"],
        "pairleg_net": s["pairleg"]["net_pnl"],
    } for s in metrics["sweep"]]
    sweep_cols = [
        ("threshold", "|p−.5| ≥"), ("n_fires", "fires"), ("taker_trades", "taker trades"),
        ("taker_net", "taker net $"), ("taker_fees", "fees $"), ("taker_ppt", "$/trade"),
        ("taker_win", "win rate"), ("pairleg_trades", "pair trades"),
        ("pairleg_locked", "locked $"), ("pairleg_stranded", "stranded $"), ("pairleg_net", "pair net $"),
    ]
    cmp_rows = [
        {"strategy": "ML taker (best)", "net_pnl": metrics["comparison"]["ml_best_taker_net"]},
        {"strategy": "ML pair-leg (best)", "net_pnl": metrics["comparison"]["ml_best_pairleg_net"]},
        {"strategy": "engine momentum", "net_pnl": metrics["comparison"]["momentum_net"]},
        {"strategy": "engine late-window", "net_pnl": metrics["comparison"]["late_window_net"]},
        {"strategy": "never trading", "net_pnl": metrics["comparison"]["never_trading_net"]},
    ]
    lift = metrics["lift"]
    lift_rows = [
        {"metric": "dir-acc", "model": lift["model"]["diracc"], "majority": lift["majority"]["diracc"],
         "lift": lift["diracc_lift"]},
        {"metric": "Brier", "model": lift["model"]["brier"], "majority": lift["majority"]["brier"],
         "lift": lift["brier_lift"]},
    ]
    chart = eh._img_tag(_sweep_png(metrics["sweep"], metrics["comparison"]), "PnL vs threshold")
    tables = "".join([
        "<h2>Sweep — PnL vs confidence threshold</h2>", eh._table_html(sweep_rows, sweep_cols),
        "<h2>Strategy comparison (net PnL, $)</h2>",
        eh._table_html(cmp_rows, [("strategy", "strategy"), ("net_pnl", "net PnL $")]),
        "<h2>Model skill as lift over always-predict-the-common-outcome (fired set, vs window resolution)</h2>",
        eh._table_html(lift_rows, [("metric", "metric"), ("model", "model"),
                                   ("majority", "majority baseline"), ("lift", "lift")]),
        "<h2>Per-day PnL (best threshold + momentum)</h2>",
        eh._table_html(metrics["per_day"], [("date", "date"), ("source", "source"),
                                            ("trades", "trades"), ("win_rate", "win rate"),
                                            ("net_pnl", "net $"), ("fees", "fees $")]),
        "<h2>Per-window PnL (best threshold taker)</h2>",
        eh._table_html(metrics["per_window"][:200], [("series", "series"),
                                                     ("window_open_ms", "window open ms"),
                                                     ("trades", "trades"), ("net_pnl", "net $")]),
    ])
    caveats = "".join(f"<li>{c}</li>" for c in metrics["caveats"])
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{p['title']}</title>
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
</style></head><body>
<h1>{p['title']}</h1>
<p class="muted">Model: <code>{p['model_label']}</code>. Fires = walk-forward OOS predictions
(never in-sample); both sides tradeable via the CLOB mirror; marked to window resolution.
Conservative by construction (pay the ask, skip on a stale book, $1 min notional).</p>
<div class="verdict">{metrics['verdict']}</div>
{chart}
{tables}
<h2>Caveats (read these)</h2><ul>{caveats}</ul>
</body></html>"""


# ===========================================================================
# orchestrator + CLI
# ===========================================================================
def money_judge(
    paths: Paths, *, scope: str = "all", variant: str = "full", source: str = "chainlink",
    latency_ms: int = DEFAULT_LATENCY_MS, venue_delay_ms: int = DEFAULT_VENUE_DELAY_MS,
    window_budget: float = DEFAULT_WINDOW_BUDGET,
    trade_budget: float = DEFAULT_TRADE_BUDGET, comp_min_s: float = DEFAULT_COMP_MIN_S,
    comp_max_s: float = DEFAULT_COMP_MAX_S, lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    fee_rate: float = CRYPTO_FEE_RATE, thresholds=None, run_momentum: bool = True,
    run_late_window: bool = True, since_ms: int | None = None, until_ms: int | None = None,
    oos_override: pd.DataFrame | None = None, max_rss_mb: int | None = None,
) -> dict:
    """Replay + sim + sweep + verdict. Returns the full metrics dict and writes
    ``out/money_judge/{sweep,per_day,per_window}.csv, metrics.json, report.html``.

    ``oos_override`` (a preassembled fires frame with the harness grid keys + ``p_up``)
    lets verify/tests bypass the OOS parquet without LightGBM.
    """
    thresholds = list(thresholds) if thresholds is not None else list(SWEEP_THRESHOLDS)
    # `latency_ms` is the NETWORK round-trip; marketable takers additionally eat the venue
    # delay, so the book is read as-of network + venue. All fill paths here are takers.
    effective_latency_ms = latency_ms + venue_delay_ms
    out_dir = paths.out_dir / "money_judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    fires = oos_override.copy() if oos_override is not None else _load_fires(
        paths, scope=scope, variant=variant, source=source)

    need_model = run_momentum or run_late_window
    model_rows, win_meta, res, settle, mids_by_asset = _read_signal_journal(
        paths, since_ms=since_ms, until_ms=until_ms, need_model=need_model, need_mids=run_momentum)
    windows = eh.windows_frame(win_meta, res, settle)

    result: dict = {
        "params": {"title": "Short-horizon money judge", "model_label": f"short-gbt-{scope}_fwd10_{variant}",
                   "scope": scope, "variant": variant, "source": source,
                   "latency_ms": effective_latency_ms, "network_ms": latency_ms,
                   "venue_delay_ms": venue_delay_ms, "effective_latency_ms": effective_latency_ms,
                   "window_budget": window_budget, "trade_budget": trade_budget,
                   "comp_min_s": comp_min_s, "comp_max_s": comp_max_s, "lock_threshold": lock_threshold,
                   "fee_rate": fee_rate, "thresholds": thresholds},
        "caveats": CAVEATS,
    }

    if windows.empty or fires.empty:
        result["counts"] = {"fires": 0, "resolved_windows": int(len(windows)), "trades_at_best": 0}
        result["sweep"], result["per_day"], result["per_window"] = [], [], []
        result["comparison"] = {"ml_best_taker_net": 0.0, "ml_best_pairleg_net": 0.0,
                                "momentum_net": 0.0, "late_window_net": 0.0, "never_trading_net": 0.0}
        result["momentum"] = _metrics_for_trades([])
        result["late_window"] = _metrics_for_trades([])
        result["lift"] = _lift_over_majority([])
        result["best"] = None
        result["verdict"] = _verdict(None, result["comparison"], result["lift"], 0)
        _write_outputs(out_dir, result)
        procmem.report_peak_rss("money_judge", procmem.peak_rss_mb(), max_rss_mb)
        return result

    # Join fires → resolved windows (drop fires without a resolved window / book target).
    # Reduce to the harness grid contract first so the OOS parquet's extra columns
    # (e.g. its own `dur`) never collide with the window join.
    fires = fires[["series", "window_open_ms", "sample_ts_ms", "p_up"]].copy()
    win_join = windows[["series", "open_time", "close_time", "up_token", "outcome_up", "dur"]].rename(
        columns={"open_time": "window_open_ms"})
    fires = fires.merge(win_join, on=["series", "window_open_ms"], how="inner")
    fires = fires[fires["up_token"].notna()].reset_index(drop=True)

    up_tokens = set(windows["up_token"].dropna().tolist())
    pm = sh._read_pm_books(paths, up_tokens, since_ms, until_ms)
    books = _build_book_arrays(pm)
    win_by_key = {(w.series, int(w.open_time)): {
        "up_token": w.up_token, "close_time": int(w.close_time), "outcome_up": int(w.outcome_up)}
        for w in windows.itertuples(index=False)}

    sweep = _sweep(fires, thresholds=thresholds, latency_ms=effective_latency_ms, fee_rate=fee_rate,
                   window_budget=window_budget, trade_budget=trade_budget, comp_min_s=comp_min_s,
                   comp_max_s=comp_max_s, lock_threshold=lock_threshold, books=books)
    best = _best_threshold(sweep)

    mom_rows = (_run_momentum(model_rows, mids_by_asset, win_by_key, books, latency_ms=effective_latency_ms,
                              fee_rate=fee_rate, window_budget=window_budget) if run_momentum else [])
    late_rows = (_run_late_window(model_rows, win_by_key, books, latency_ms=effective_latency_ms,
                                  fee_rate=fee_rate, window_budget=window_budget) if run_late_window else [])
    mom_m, late_m = _metrics_for_trades(mom_rows), _metrics_for_trades(late_rows)

    comparison = {
        "ml_best_taker_net": best["taker"]["net_pnl"] if best else 0.0,
        "ml_best_pairleg_net": best["pairleg"]["net_pnl"] if best else 0.0,
        "momentum_net": mom_m["net_pnl"], "late_window_net": late_m["net_pnl"],
        "never_trading_net": 0.0,
    }
    best_taker_rows = best["taker_rows"] if best else []
    lift = _lift_over_majority(best_taker_rows)
    per_window = _rollup_per_window(best_taker_rows)
    per_day = _rollup_per_day(best_taker_rows, "ml") + _rollup_per_day(mom_rows, "momentum")

    n_windows_traded = len({(r["series"], r["window_open_ms"]) for r in best_taker_rows})
    total_fires = max((s["n_fires"] for s in sweep), default=0)  # most-permissive threshold
    result.update({
        "counts": {"fires": int(total_fires),
                   "resolved_windows": int(len(windows)), "fires_joined": int(len(fires)),
                   "trades_at_best": len(best_taker_rows), "windows_traded": n_windows_traded},
        # Strip the heavy per-trade row lists from the serialized sweep.
        "sweep": [{k: v for k, v in s.items() if not k.endswith("_rows")} for s in sweep],
        "best": ({"threshold": best["threshold"], "taker": best["taker"], "pairleg": best["pairleg"]}
                 if best else None),
        "comparison": comparison, "momentum": mom_m, "late_window": late_m, "lift": lift,
        "per_day": per_day, "per_window": per_window,
        "peak_rss_mb": procmem.peak_rss_mb(),
        "date_span": ({"min_day": int(fires["window_open_ms"].min() // MS_PER_DAY),
                       "max_day": int(fires["window_open_ms"].max() // MS_PER_DAY)}
                      if len(fires) else {}),
    })
    result["verdict"] = _verdict(best, comparison, lift, n_windows_traded)
    _write_outputs(out_dir, result, best_taker_rows)
    procmem.report_peak_rss("money_judge", procmem.peak_rss_mb(), max_rss_mb)
    return result


#: Per-trade trades.csv header (shared with backtest_momentum; also read by
#: rebate_sim). Kept explicit so the file has a stable header even with 0 trades.
_TRADE_COLS = ["series", "window_open_ms", "sample_ts_ms", "day", "p_up", "outcome_up",
               "side", "shares", "price", "fee", "cost", "payoff", "net", "won"]


def _write_outputs(out_dir, result: dict, best_taker_rows: list | None = None) -> None:
    # The winning-threshold model-taker fills, so rebate_sim can be run on this run
    # (same header as backtest_momentum's trades.csv).
    pd.DataFrame(best_taker_rows or [], columns=_TRADE_COLS).to_csv(
        out_dir / "trades.csv", index=False)
    pd.DataFrame([{"threshold": s["threshold"], "n_fires": s["n_fires"],
                   "taker_n_trades": s["taker"]["n_trades"], "taker_filled_shares": s["taker"]["filled_shares"],
                   "taker_gross_pnl": s["taker"]["gross_pnl"], "taker_fees": s["taker"]["fees"],
                   "taker_net_pnl": s["taker"]["net_pnl"], "taker_pnl_per_trade": s["taker"]["pnl_per_trade"],
                   "taker_win_rate": s["taker"]["win_rate"], "pairleg_n_trades": s["pairleg"]["n_trades"],
                   "pairleg_filled_shares": s["pairleg"]["filled_shares"],
                   "pairleg_locked_pnl": s["pairleg"]["locked_pnl"],
                   "pairleg_stranded_pnl": s["pairleg"]["stranded_pnl"],
                   "pairleg_fees": s["pairleg"]["fees"], "pairleg_net_pnl": s["pairleg"]["net_pnl"],
                   "pairleg_pnl_per_trade": s["pairleg"]["pnl_per_trade"],
                   "pairleg_win_rate": s["pairleg"]["win_rate"]}
                  for s in result["sweep"]]).to_csv(out_dir / "sweep.csv", index=False)
    pd.DataFrame(result["per_day"], columns=["date", "source", "trades", "win_rate", "net_pnl", "fees"]).to_csv(
        out_dir / "per_day.csv", index=False)
    pd.DataFrame(result["per_window"], columns=["series", "window_open_ms", "trades", "net_pnl"]).to_csv(
        out_dir / "per_window.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    (out_dir / "report.html").write_text(_build_html(result), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "short-horizon money judge")
    parser.add_argument("--scope", default="all", choices=["all", "chainlink", "binance_proxy"],
                        help="which learn_short_gbt OOS scope to judge (default all)")
    parser.add_argument("--variant", default="full", choices=["base", "depth", "full"],
                        help="feature variant of the model (default full)")
    parser.add_argument("--source", default="chainlink",
                        help="label_source to trade — 'chainlink' = real-book/real-resolution weeks")
    parser.add_argument("--latency-ms", type=int, default=DEFAULT_LATENCY_MS,
                        help=f"NETWORK latency ms (default {DEFAULT_LATENCY_MS}; VPS-colocated ≈ 5)")
    parser.add_argument("--venue-delay-ms", type=int, default=DEFAULT_VENUE_DELAY_MS,
                        help=f"venue taker delay ms added on top for marketable fills "
                             f"(default {DEFAULT_VENUE_DELAY_MS}; 0 disables)")
    parser.add_argument("--window-budget", type=float, default=DEFAULT_WINDOW_BUDGET)
    parser.add_argument("--trade-budget", type=float, default=DEFAULT_TRADE_BUDGET)
    parser.add_argument("--comp-min-s", type=float, default=DEFAULT_COMP_MIN_S)
    parser.add_argument("--comp-max-s", type=float, default=DEFAULT_COMP_MAX_S)
    parser.add_argument("--lock-threshold", type=float, default=DEFAULT_LOCK_THRESHOLD)
    parser.add_argument("--fee-rate", type=float, default=CRYPTO_FEE_RATE)
    parser.add_argument("--no-momentum", action="store_true", help="skip the momentum baseline")
    parser.add_argument("--no-late-window", action="store_true", help="skip the late-window baseline")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    try:
        m = money_judge(
            paths, scope=args.scope, variant=args.variant, source=args.source,
            latency_ms=args.latency_ms, venue_delay_ms=args.venue_delay_ms,
            window_budget=args.window_budget, trade_budget=args.trade_budget,
            comp_min_s=args.comp_min_s, comp_max_s=args.comp_max_s, lock_threshold=args.lock_threshold,
            fee_rate=args.fee_rate, run_momentum=not args.no_momentum,
            run_late_window=not args.no_late_window, since_ms=since_ms, until_ms=until_ms,
            max_rss_mb=args.max_rss_mb)
    except ParquetNotReady as exc:
        print(f"[money_judge] {exc}")
        return 1

    c = m["counts"]
    print(f"[money_judge] {c['fires']:,} fires over {c['resolved_windows']} resolved windows; "
          f"traded {c['trades_at_best']} at best threshold across {c['windows_traded']} window(s)")
    if m["best"] is not None:
        b = m["best"]
        print(f"[money_judge] best |p−.5|≥{b['threshold']:g}: taker net ${b['taker']['net_pnl']:.2f} "
              f"(win {b['taker']['win_rate']:.0%}), momentum ${m['comparison']['momentum_net']:.2f}, "
              f"late-window ${m['comparison']['late_window_net']:.2f}, never-trade $0.00")
    print(f"[money_judge] {m['verdict']}")
    print(f"[money_judge] wrote {paths.out_dir / 'money_judge'} (sweep/per_day/per_window.csv, "
          "metrics.json, report.html)")
    return 0 if (m["counts"]["fires"] > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
