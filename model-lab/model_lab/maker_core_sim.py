"""Pure, sans-IO maker-core simulator — a faithful Python replay of the engine's
two-sided quoting core against a reconstructed Polymarket book + trade tape.

This module reproduces, offline and deterministically, exactly three things from the
Rust engine so the ``backtest_maker_core`` stage can ask *"does the model defend
quotes better than always-on quoting?"*:

1. **The quoting calculator** (:func:`desired_quotes`) — a line-for-line port of
   ``crates/engine/src/quoting.rs::calculate_quotes`` with ``QuoteParams::default()``
   (``config/default.toml [engine.defaults]``): inventory-skewed ``center = p_up −
   γ·norm_imbalance``; ``half_spread = max(min_edge, σ·(k1·√hold·spike_mult + √rtt))``;
   two BUY ladders (Up fair = center, Down fair = 1 − center) snapped DOWN onto the tick
   grid; the full gate ladder (health → window → τ → no-passive → usable inputs → caps →
   per-level ATM / pair-cost / dedupe).

2. **The print-driven, queue-behind-displayed fill engine** (inside
   :func:`simulate_window`) — a mirror of ``crates/venue-paper/src/engine.rs::on_trade``:
   a resting BUY at price ``P`` is eligible only for a **sell-aggressor** print on its
   token at price ≤ ``P``; the print's size is a finite shared budget consumed in
   price → time priority; each order drains its ``queue_ahead`` (the displayed size it
   sits behind) before it fills; maker fee is $0, fill price is our own limit.

3. **The reactive cancel-first defense** (in :func:`simulate_window`) — the engine's
   ``quote_manager`` behaviour: a resting level is cancelled + replaced when its desired
   price drifts past ``reprice_theta``; the directional urgent-cancel pulls the endangered
   side when the analytic fair moves (``Up`` if it fell, ``Down`` if it rose), escalating
   to a full pull past ``cancel_market_theta``. **This is part of the always-on baseline.**

On top of the baseline, two model overlays are supported (:func:`desired_quotes`
``mode``): **defend** pulls the threatened side entirely while the dir10 signal fires,
and **lean** instead widens the threatened side and tightens the safe side.

Everything is pure: the caller passes reconstructed numpy arrays (per-second model rows,
the token books, the token trade tapes, the dir10 signal) and gets back an additive
metrics dict (net PnL split into locked-pair vs stranded-leg, fills, pair-completion,
5 s/30 s markout on maker fills, maker-rebate estimate). No clock, no IO, no pandas in
the hot loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .lib import math as lm

# --- engine defaults (crates/config/default.toml [engine.defaults] + QuoteParams::default) ---
MAKER_REBATE_SHARE = 0.20  # crypto maker rebate share (CLAUDE §7); estimate on filled maker volume.
CRYPTO_FEE_RATE = 0.07  # taker feeRate — the base the maker-rebate estimate is scaled from.


@dataclass(frozen=True)
class QuoteParams:
    """The engine's per-series quoting parameters (defaults mirror ``config/default.toml``
    ``[engine.defaults]`` + the four engine-only ``QuoteParams::default()`` knobs)."""

    min_edge: float = 0.01
    k1_vol_multiplier: float = 1.0
    expected_hold_secs: float = 30.0
    gamma_inventory_skew: float = 0.05
    touch_size: float = 10.0
    ladder_levels: int = 3
    ladder_size_per_level: float = 10.0
    ladder_tick_offset: int = 1
    pair_cost_threshold: float = 0.98
    no_atm_final_secs: int = 25
    no_passive_final_secs: int = 5
    soft_cap_excess: float = 50.0
    hard_cap_excess: float = 100.0
    max_worst_case_loss: float = 25.0
    # engine-only (no config key) — QuoteParams::default()
    cancel_rtt_secs: float = 0.2
    atm_band: float = 0.10
    vol_spike_threshold_1s: float = 0.001
    vol_spike_mult: float = 1.5
    # quote_manager (config: reprice_threshold_theta / cancel_market_theta)
    reprice_theta: float = 0.005
    cancel_market_theta: float = 0.02


# The dir10 signal stands down when the latest prediction is older than this at the decision
# second (the live shadow's max_prediction_age_ms; the defense reverts to the baseline).
SIGNAL_MAX_STALENESS_MS = 15_000


def market_tick(p_up: float) -> float:
    """The tick grid for a fair near ``p_up`` — 0.01, flipping to 0.001 past 0.96/0.04
    (our markets, CLAUDE §7). A single market tick, as ``calculate_quotes`` receives."""
    return 0.001 if (p_up < 0.04 or p_up > 0.96) else 0.01


def price_down(target: float, tick: float) -> float | None:
    """Snap ``target`` DOWN onto ``tick``'s grid (passive-safe for a resting BUY), mirroring
    ``quoting.rs::price_down``: round to 9 dp first to absorb f64 noise (so a value *meant*
    to be on-grid is not pushed a tick low), quantize DOWN, clamp into ``[tick, 1 − tick]``.
    Returns ``None`` for a non-finite or non-positive target (the level is skipped)."""
    if not math.isfinite(target) or target <= 0.0:
        return None
    d = round(target, 9)
    ticks = math.floor(d / tick + 1e-9)  # +eps absorbs float division noise for on-grid values
    q = round(ticks * tick, 6)
    lo, hi = tick, round(1.0 - tick, 6)
    if q < lo:
        q = lo
    if q > hi:
        q = hi
    if q <= 0.0 or q >= 1.0:
        return None
    return q


@dataclass
class Inventory:
    """Per-window per-side holdings — shares + total cost, all engine math derived (mirrors
    ``core-types::InventorySnapshot::derive``). Maker fees are zero, so cost = Σ shares·price."""

    up_shares: float = 0.0
    up_cost: float = 0.0
    down_shares: float = 0.0
    down_cost: float = 0.0

    def add(self, outcome: str, shares: float, price: float) -> None:
        if outcome == "Up":
            self.up_shares += shares
            self.up_cost += shares * price
        else:
            self.down_shares += shares
            self.down_cost += shares * price

    def avg(self, outcome: str) -> float | None:
        if outcome == "Up":
            return self.up_cost / self.up_shares if self.up_shares > 0.0 else None
        return self.down_cost / self.down_shares if self.down_shares > 0.0 else None

    def excess(self) -> float:
        """Signed unmatched shares: + = Up excess, − = Down excess."""
        return self.up_shares - self.down_shares

    def worst_case_if_excess_loses(self) -> float:
        """Dollars sunk into the excess shares (the binding §8 risk input)."""
        e = self.excess()
        if e > 0.0:
            return e * (self.avg("Up") or 0.0)
        if e < 0.0:
            return -e * (self.avg("Down") or 0.0)
        return 0.0


# --- calculator (crates/engine/src/quoting.rs::calculate_quotes) ------------------------------
_OTHER = {"Up": "Down", "Down": "Up"}


def leaned_side(p_up_sig: float, theta: float) -> str | None:
    """The side the dir10 signal leans toward when it fires beyond ``theta`` (mirrors
    ``money_judge._side_of``): ``Up`` if ``p_up_sig ≥ 0.5+θ``, ``Down`` if ``≤ 0.5−θ``, else
    ``None``. The *threatened* side (the one the predicted move runs over) is the opposite."""
    if not math.isfinite(p_up_sig):
        return None
    if p_up_sig >= 0.5 + theta:
        return "Up"
    if p_up_sig <= 0.5 - theta:
        return "Down"
    return None


def _fair_o(outcome: str, p_up: float) -> float:
    """Outcome-oriented fair: ``p_up`` for an Up position, ``1 − p_up`` for Down."""
    return p_up if outcome == "Up" else 1.0 - p_up


def desired_quotes(
    p_up: float,
    sigma: float,
    tau_secs: float,
    inv: Inventory,
    params: QuoteParams,
    *,
    mode: str = "baseline",
    threatened_side: str | None = None,
    lean_mult: float = 0.0,
) -> tuple[list[tuple[str, int, float, float]], dict]:
    """The desired resting maker quote set for one instant — a faithful port of
    ``calculate_quotes`` (gates G1..S9), plus the model overlays.

    Returns ``(levels, meta)`` where ``levels`` is a list of ``(outcome, level, price, size)``
    (empty on a no-quote) and ``meta`` carries ``{no_quote, center, half_spread}``.

    ``mode``: ``"baseline"`` (faithful engine), ``"defend"`` (drop ``threatened_side``
    entirely — the model pull), or ``"lean"`` (widen ``threatened_side`` by ``lean_mult``
    half-spreads and tighten the safe side by the same). ``threatened_side`` is ``None`` when
    the signal is not firing, so defend/lean then reduce to the baseline for that instant."""
    meta = {"no_quote": None, "center": float("nan"), "half_spread": float("nan")}

    # G3 expiry / G4 no-passive final seconds (G1 health always Ready in reconstruction).
    if not math.isfinite(tau_secs) or tau_secs <= 0.0:
        meta["no_quote"] = "expired"
        return [], meta
    if tau_secs <= params.no_passive_final_secs:
        meta["no_quote"] = "no_passive_final_secs"
        return [], meta
    # G5 usable inputs.
    if not math.isfinite(p_up) or p_up <= 0.0 or p_up >= 1.0:
        meta["no_quote"] = "unusable_fair"
        return [], meta
    if not math.isfinite(sigma) or sigma <= 0.0:
        meta["no_quote"] = "unusable_vol"
        return [], meta

    # S1 normalized inventory imbalance (±1 at the hard cap).
    hard_cap = params.hard_cap_excess
    norm_imbalance = max(-1.0, min(1.0, inv.excess() / hard_cap)) if hard_cap > 0.0 else 0.0
    # S2 inventory-skewed center.
    tick = market_tick(p_up)
    center = p_up - params.gamma_inventory_skew * norm_imbalance
    center = max(tick, min(round(1.0 - tick, 6), center))
    # S3 half-spread.
    latency_premium = sigma * math.sqrt(max(params.cancel_rtt_secs, 0.0))
    vol_spike = sigma >= params.vol_spike_threshold_1s
    spike_mult = params.vol_spike_mult if vol_spike else 1.0
    vol_term = params.k1_vol_multiplier * sigma * math.sqrt(max(params.expected_hold_secs, 0.0)) * spike_mult
    half_spread = max(params.min_edge, vol_term + latency_premium)
    meta["center"], meta["half_spread"] = center, half_spread

    # S4 excess caps.
    excess = inv.excess()
    abs_excess = abs(excess)
    if abs_excess >= params.hard_cap_excess:
        meta["no_quote"] = "inventory_blocked"
        return [], meta
    soft_capped = abs_excess >= params.soft_cap_excess
    worst_breach = inv.worst_case_if_excess_loses() > params.max_worst_case_loss
    excess_side = "Up" if excess > 0.0 else ("Down" if excess < 0.0 else None)
    atm_active = tau_secs <= params.no_atm_final_secs

    levels: list[tuple[str, int, float, float]] = []
    for outcome in ("Up", "Down"):
        # per-side excess cap suppression (soft cap / worst-case only bite the excess side).
        if excess_side == outcome and (soft_capped or worst_breach):
            continue
        # model overlay — defend pulls the threatened side; lean re-scales its half-spread.
        hs = half_spread
        if threatened_side is not None:
            if mode == "defend" and outcome == threatened_side:
                continue  # the model pull
            if mode == "lean":
                if outcome == threatened_side:
                    hs = half_spread * (1.0 + lean_mult)  # widen (further from fair)
                elif outcome == _OTHER[threatened_side]:
                    hs = max(params.min_edge, half_spread * (1.0 - lean_mult))  # tighten safe side

        side_fair = center if outcome == "Up" else 1.0 - center
        accepted: set[float] = set()
        for level in range(params.ladder_levels):
            offset_ticks = params.ladder_tick_offset * level
            target = side_fair - hs - offset_ticks * tick
            price = price_down(target, tick)
            if price is None:
                continue
            # S6 final-seconds ATM gate.
            if atm_active and abs(price - 0.5) <= params.atm_band:
                continue
            size = params.touch_size if level == 0 else params.ladder_size_per_level
            # S7 pair-cost authorization: avg(add side after) + avg(other side) ≤ threshold.
            other_avg = inv.avg(_OTHER[outcome])
            if other_avg is not None:
                add_shares = (inv.up_shares if outcome == "Up" else inv.down_shares) + size
                add_cost = (inv.up_cost if outcome == "Up" else inv.down_cost) + price * size
                add_avg = add_cost / add_shares
                if add_avg + other_avg > params.pair_cost_threshold + 1e-12:
                    continue
            # S8 within-side grid-collision dedupe.
            if price in accepted:
                continue
            accepted.add(price)
            levels.append((outcome, level, price, size))
    if not levels:
        meta["no_quote"] = "no_surviving_levels"
    return levels, meta


# --- book / signal as-of helpers (top-of-book) ------------------------------------------------
def _asof_idx(ts: np.ndarray, at_ms: int, staleness_ms: int | None = None) -> int:
    """Index of the most recent row at or before ``at_ms`` (−1 if none / stale)."""
    if ts.size == 0:
        return -1
    i = int(np.searchsorted(ts, at_ms, side="right")) - 1
    if i < 0:
        return -1
    if staleness_ms is not None and (at_ms - int(ts[i])) > staleness_ms:
        return -1
    return i


def _bid_asof(book: dict, now: int) -> tuple[float, float] | None:
    """(bid_px, bid_sz) at or before ``now`` from a single-level book, or ``None``."""
    i = _asof_idx(book["ts"], now)
    if i < 0:
        return None
    bpx, bsz = float(book["bid_px"][i]), float(book["bid_sz"][i])
    if not (math.isfinite(bpx) and 0.0 < bpx < 1.0 and bsz > 0.0):
        return None
    return bpx, bsz


def _ask_asof(book: dict, now: int) -> float | None:
    """The best ask at or before ``now`` (for the post-only cross check), or ``None``."""
    i = _asof_idx(book["ts"], now)
    if i < 0:
        return None
    apx = float(book["ask_px"][i])
    return apx if (math.isfinite(apx) and 0.0 < apx < 1.0) else None


def _queue_ahead(price: float, book: dict, now: int) -> float:
    """Displayed size a resting BUY at ``price`` queues behind (mirrors
    ``venue-paper::SimBook::displayed_at`` given only top-of-book): 0 if we strictly improve
    the best bid (alone at the front), else the displayed best-bid size (a conservative proxy —
    a below-touch level's true depth is unobservable without the full L2 ladder). Never more
    optimistic than reality."""
    bid = _bid_asof(book, now)
    if bid is None:
        return 0.0
    bpx, bsz = bid
    return 0.0 if price > bpx + 1e-9 else bsz


def _locf(ts: np.ndarray, vals: np.ndarray, at_ms: int) -> float | None:
    i = _asof_idx(ts, at_ms)
    return float(vals[i]) if i >= 0 else None


# --- the per-window simulation ---------------------------------------------------------------
def _empty_metrics() -> dict:
    return {
        "windows": 0, "windows_traded": 0, "net_pnl": 0.0, "locked_pnl": 0.0,
        "stranded_pnl": 0.0, "rebate": 0.0, "n_fills": 0, "fill_shares": 0.0,
        "matched_shares": 0.0, "markout5_wsum": 0.0, "markout30_wsum": 0.0,
        "markout_shares": 0.0, "signal_fire_secs": 0, "lag_ms_sum": 0.0, "lag_n": 0,
    }


def add_metrics(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v


def signal_fires(signal: dict | None, ts: np.ndarray, theta: float) -> bool:
    """Whether the dir10 signal fires beyond ``theta`` at any decision second of a window
    (used to skip re-simulating windows where defend/lean can only equal the baseline)."""
    if signal is None or signal["ts"].size == 0 or ts.size == 0:
        return False
    for now in ts:
        i = _asof_idx(signal["ts"], int(now), SIGNAL_MAX_STALENESS_MS)
        if i >= 0 and leaned_side(float(signal["p_up"][i]), theta) is not None:
            return True
    return False


def simulate_window(
    *,
    rows_ts: np.ndarray,
    rows_p_up: np.ndarray,
    rows_sigma: np.ndarray,
    up_book: dict,
    up_prints: dict,
    down_book: dict,
    down_prints: dict,
    outcome_up: int,
    close_ms: int,
    params: QuoteParams,
    mode: str = "baseline",
    signal: dict | None = None,
    theta: float | None = None,
    lean_mult: float = 0.0,
) -> dict:
    """Replay the maker core over one window and return an additive metrics dict.

    Drives a per-second quoting cycle over the reconstructed model rows; fills resting quotes
    from the real trade prints (queue-behind-displayed); marks the closing inventory to the
    official outcome (locked-pairs + stranded-legs, exact split); measures 5 s/30 s markout on
    every maker fill. ``mode`` selects the always-on baseline or a model overlay (defend / lean)."""
    m = _empty_metrics()
    m["windows"] = 1
    n = rows_ts.size
    if n == 0:
        return m

    books = {"Up": up_book, "Down": down_book}
    prints = {"Up": up_prints, "Down": down_prints}
    # per-token print cursors (prints are ts-sorted).
    cursor = {"Up": 0, "Down": 0}

    inv = Inventory()
    resting: dict[tuple[str, int], dict] = {}  # (outcome, level) -> {price, remaining, queue_ahead, seq}
    fills: list[tuple[str, int, float, float]] = []  # (outcome, ts, price, shares)
    seq = 0
    p_up_last_quote: float | None = None

    for r in range(n):
        now = int(rows_ts[r])
        p_up = float(rows_p_up[r])
        sigma = float(rows_sigma[r])
        tau = (close_ms - now) / 1000.0
        next_ts = int(rows_ts[r + 1]) if r + 1 < n else close_ms

        # (1) reactive urgent-cancel — the always-on baseline defense (all modes).
        if p_up_last_quote is not None and resting:
            d = p_up - p_up_last_quote
            mag = abs(d)
            if mag > params.cancel_market_theta:
                resting.clear()
            elif mag > params.reprice_theta:
                endangered = "Up" if d < 0.0 else "Down"  # fair fell → Up-buys stale; rose → Down-buys
                for slot in [s for s in resting if s[0] == endangered]:
                    del resting[slot]

        # (2) the dir10 signal → the threatened side (defend/lean only).
        threatened = None
        if mode in ("defend", "lean") and signal is not None and theta is not None:
            si = _asof_idx(signal["ts"], now, SIGNAL_MAX_STALENESS_MS)
            if si >= 0:
                leaned = leaned_side(float(signal["p_up"][si]), theta)
                if leaned is not None:
                    m["signal_fire_secs"] += 1
                    m["lag_ms_sum"] += now - int(signal["ts"][si])  # effective lag at DECISION time
                    m["lag_n"] += 1
                    threatened = _OTHER[leaned]  # the side the predicted move runs over

        # (3) desired quotes (baseline calculator + overlay).
        desired, _meta = desired_quotes(p_up, sigma, tau, inv, params, mode=mode,
                                        threatened_side=threatened, lean_mult=lean_mult)
        desired_by_slot = {(o, lvl): (o, lvl, px, sz) for (o, lvl, px, sz) in desired}

        # (4) converge — strict split: cancel stale/unwanted this cycle; place into empty
        #     slots not just cancelled (a replacement waits for the next cycle).
        just_cancelled: set[tuple[str, int]] = set()
        for slot in list(resting.keys()):
            want = desired_by_slot.get(slot)
            if want is None or abs(want[2] - resting[slot]["price"]) > params.reprice_theta:
                del resting[slot]
                just_cancelled.add(slot)
        placed_any = False
        for slot, (o, lvl, px, sz) in desired_by_slot.items():
            if slot in resting or slot in just_cancelled:
                continue
            ask = _ask_asof(books[o], now)
            if ask is not None and px >= ask - 1e-12:  # post-only: never cross the ask
                continue
            resting[slot] = {"price": px, "remaining": sz,
                             "queue_ahead": _queue_ahead(px, books[o], now), "seq": seq}
            seq += 1
            placed_any = True
        if placed_any:
            p_up_last_quote = p_up

        # (5) process prints in [now, next_ts) against resting orders, per token.
        for token in ("Up", "Down"):
            _fill_from_prints(token, prints[token], cursor, now, next_ts, resting, inv, fills)

    _settle(m, inv, outcome_up)
    _markouts(m, fills, rows_ts, rows_p_up, close_ms, outcome_up)
    return m


def _fill_from_prints(token, tp, cursor, now, next_ts, resting, inv, fills) -> None:
    """Consume every sell-aggressor print with ``now ≤ ts < next_ts`` against ``token``'s
    resting BUYs — mirror of ``MatchEngine::on_trade`` (finite shared budget, queue-ahead
    drain, price → seq priority, maker fee $0, fill at our limit). Prints before ``now`` (e.g.
    pre-warmup, before any quote existed) are skipped, never filling a later order."""
    ts = tp["ts"]
    npr = ts.size
    i = cursor[token]
    while i < npr and int(ts[i]) < next_ts:
        if int(ts[i]) < now or not bool(tp["is_sell"][i]):
            i += 1
            continue
        pp = float(tp["price"][i])
        budget = float(tp["size"][i])
        pt = int(ts[i])
        i += 1
        if budget <= 0.0 or not math.isfinite(pp):
            continue
        # eligible resting BUYs on this token, at or through the print price (bid ≥ sell price).
        elig = [(slot, o) for slot, o in resting.items()
                if slot[0] == token and o["price"] >= pp - 1e-12]
        if not elig:
            continue
        # priority: most aggressive (highest bid) first, then earliest placed (seq).
        elig.sort(key=lambda so: (-so[1]["price"], so[1]["seq"]))
        for slot, o in elig:
            if budget <= 1e-12:
                break
            drain = min(budget, o["queue_ahead"])
            o["queue_ahead"] -= drain
            budget -= drain
            if budget <= 1e-12:
                break
            fill = min(budget, o["remaining"])
            if fill <= 0.0:
                continue
            o["remaining"] -= fill
            budget -= fill
            inv.add(token, fill, o["price"])
            fills.append((token, pt, o["price"], fill))
            if o["remaining"] <= 1e-9:
                del resting[slot]
    cursor[token] = i


def _settle(m: dict, inv: Inventory, outcome_up: int) -> None:
    """Mark the closing inventory to the official outcome; split net PnL into locked-pairs +
    stranded-legs exactly (mirrors ``crates/analytics::attribution``). Maker fills paid $0 fee,
    so the maker-rebate estimate is a separate reported line (never in the traded PnL)."""
    up_s, up_c, dn_s, dn_c = inv.up_shares, inv.up_cost, inv.down_shares, inv.down_cost
    winning = up_s if outcome_up == 1 else dn_s
    realized = winning * 1.0 - (up_c + dn_c)
    # stranded (excess directional) leg marked to resolution.
    excess = up_s - dn_s
    if excess > 0.0:
        stranded_shares, excess_avg = excess, (up_c / up_s if up_s > 0 else 0.0)
        excess_won = outcome_up == 1
    elif excess < 0.0:
        stranded_shares, excess_avg = -excess, (dn_c / dn_s if dn_s > 0 else 0.0)
        excess_won = outcome_up == 0
    else:
        stranded_shares, excess_avg, excess_won = 0.0, 0.0, False
    stranded_pnl = (stranded_shares * (1.0 if excess_won else 0.0)) - stranded_shares * excess_avg
    m["net_pnl"] = realized
    m["stranded_pnl"] = stranded_pnl
    m["locked_pnl"] = realized - stranded_pnl  # locked + stranded == realized, exact
    m["matched_shares"] = 2.0 * min(up_s, dn_s)


def _markouts(m, fills, rows_ts, rows_p_up, close_ms, outcome_up) -> None:
    """5 s / 30 s markout on every maker fill (LOCF over reconstructed p_up, clamp-to-settlement
    past close — mirror of ``crates/analytics::markout`` S5) + fill counts + rebate estimate."""
    n_fills = 0
    fill_shares = 0.0
    rebate = 0.0
    mo5_wsum = mo30_wsum = mo_shares = 0.0
    for outcome, t_fill, price, shares in fills:
        n_fills += 1
        fill_shares += shares
        rebate += MAKER_REBATE_SHARE * lm.taker_fee(shares, CRYPTO_FEE_RATE, price)
        before = _locf(rows_ts, rows_p_up, t_fill)
        if before is None:
            continue
        fair_before = _fair_o(outcome, before)
        won = (outcome == "Up" and outcome_up == 1) or (outcome == "Down" and outcome_up == 0)
        mo_shares += shares
        for h, wsum_key in ((5000, "mo5"), (30000, "mo30")):
            deadline = t_fill + h
            if deadline > close_ms:
                fair_after = 1.0 if won else 0.0  # clamp to settlement
            else:
                p_after = _locf(rows_ts, rows_p_up, deadline)
                fair_after = _fair_o(outcome, p_after) if p_after is not None else fair_before
            mo = lm.markout(fair_before, fair_after, 1)  # all positions are BUYs (+1)
            if wsum_key == "mo5":
                mo5_wsum += shares * mo
            else:
                mo30_wsum += shares * mo
    m["n_fills"] = n_fills
    m["fill_shares"] = fill_shares
    m["windows_traded"] = 1 if n_fills > 0 else 0
    m["rebate"] = rebate
    m["markout5_wsum"] = mo5_wsum
    m["markout30_wsum"] = mo30_wsum
    m["markout_shares"] = mo_shares
