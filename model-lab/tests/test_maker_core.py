"""Tests for the maker-core defense backtest — the pure quoting/fill engine invariants
(``maker_core_sim``) plus the stage's structure + primitive-reuse + fixture end-to-end.

The engine port is validated against ``crates/engine/src/quoting.rs``'s own worked example
and gate semantics; the fill engine against ``venue-paper::MatchEngine::on_trade``'s
queue-behind-displayed / shared-budget / sell-aggressor rules."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_lab import backtest_maker_core as bmc
from model_lab import maker_core_sim as sim
from model_lab import money_judge as mj
from model_lab.config import Paths
from model_lab.fixtures import make_historical_fixture, write_historical_coverage


# --- helpers ----------------------------------------------------------------
def _book(bid_px=None, bid_sz=0.0, ask_px=None, ask_sz=0.0, ts=0):
    nan = float("nan")
    return {"ts": np.array([ts], dtype="int64"),
            "bid_px": np.array([bid_px if bid_px is not None else nan], dtype=float),
            "bid_sz": np.array([bid_sz], dtype=float),
            "ask_px": np.array([ask_px if ask_px is not None else nan], dtype=float),
            "ask_sz": np.array([ask_sz], dtype=float)}


def _empty_book():
    return {k: (np.empty(0, dtype="int64") if k == "ts" else np.empty(0, dtype=float))
            for k in ("ts", "bid_px", "bid_sz", "ask_px", "ask_sz")}


def _prints(items):
    if not items:
        return {"ts": np.empty(0, dtype="int64"), "price": np.empty(0, dtype=float),
                "size": np.empty(0, dtype=float), "is_sell": np.empty(0, dtype=bool)}
    ts, px, sz, se = zip(*items)
    return {"ts": np.array(ts, dtype="int64"), "price": np.array(px, dtype=float),
            "size": np.array(sz, dtype=float), "is_sell": np.array(se, dtype=bool)}


def _sig(items):
    ts, p = zip(*items)
    return {"ts": np.array(ts, dtype="int64"), "p_up": np.array(p, dtype=float)}


def _run(*, rows, up_book, up_prints, down_book=None, down_prints=None, outcome_up=1,
         close_ms=200_000, params=None, **kw):
    ts, pu, sg = zip(*rows)
    return sim.simulate_window(
        rows_ts=np.array(ts, dtype="int64"), rows_p_up=np.array(pu, dtype=float),
        rows_sigma=np.array(sg, dtype=float), up_book=up_book, up_prints=up_prints,
        down_book=down_book if down_book is not None else _empty_book(),
        down_prints=down_prints if down_prints is not None else _prints([]),
        outcome_up=outcome_up, close_ms=close_ms, params=params or sim.QuoteParams(), **kw)


# --- calculator port (quoting.rs) -------------------------------------------
def test_worked_example_matches_engine():
    # quoting.rs test 1: p_up=0.5, σ=0.0002, τ=120 → half_spread=0.01, ladders 0.49/0.48/0.47.
    levels, meta = sim.desired_quotes(0.5, 0.0002, 120.0, sim.Inventory(), sim.QuoteParams())
    assert round(meta["center"], 6) == 0.5 and round(meta["half_spread"], 6) == 0.01
    up = sorted(p for (o, _l, p, _s) in levels if o == "Up")
    dn = sorted(p for (o, _l, p, _s) in levels if o == "Down")
    assert up == [0.47, 0.48, 0.49] and dn == [0.47, 0.48, 0.49]


def test_inventory_skew_lowers_center_when_long_up():
    # long 60 Up (excess 60, hard_cap 100 → norm 0.6): center = 0.5 − 0.05·0.6 = 0.47.
    inv = sim.Inventory(up_shares=60.0, up_cost=30.0)
    _levels, meta = sim.desired_quotes(0.5, 0.0002, 120.0, inv, sim.QuoteParams())
    assert round(meta["center"], 6) == 0.47


def test_no_passive_and_atm_gates():
    p = sim.QuoteParams()
    # τ ≤ no_passive_final_secs (5) → no quotes at all.
    lv, meta = sim.desired_quotes(0.5, 0.0002, 4.0, sim.Inventory(), p)
    assert lv == [] and meta["no_quote"] == "no_passive_final_secs"
    # 5 < τ ≤ 25 and center ≈ 0.5 → every level ATM-suppressed.
    lv, meta = sim.desired_quotes(0.5, 0.0002, 10.0, sim.Inventory(), p)
    assert lv == [] and meta["no_quote"] == "no_surviving_levels"
    # far from 0.5 → ATM does not bite.
    lv, _ = sim.desired_quotes(0.8, 0.0002, 10.0, sim.Inventory(), p)
    assert any(o == "Up" for (o, _l, _p, _s) in lv)


def test_pair_cost_gate_suppresses_the_expensive_side():
    # Up avg 0.6 already; adding Down at ~0.47–0.49 → pair cost ≥ 1.07 > 0.98 → every Down level
    # suppressed. The Up side (other side empty) is a one-sided add → authorized.
    inv = sim.Inventory(up_shares=10.0, up_cost=6.0)
    lv, _ = sim.desired_quotes(0.5, 0.0002, 120.0, inv, sim.QuoteParams())
    sides = {o for (o, _l, _p, _s) in lv}
    assert "Up" in sides and "Down" not in sides


def test_defend_pulls_threatened_side_lean_rescales():
    p = sim.QuoteParams()
    # σ high enough that half_spread > min_edge (so the safe-side tighten has room to move).
    sig = 0.003
    base, mb = sim.desired_quotes(0.5, sig, 120.0, sim.Inventory(), p, mode="baseline")
    assert mb["half_spread"] > p.min_edge
    dfd, _ = sim.desired_quotes(0.5, sig, 120.0, sim.Inventory(), p, mode="defend",
                                threatened_side="Down")
    assert {o for (o, *_r) in base} == {"Up", "Down"}
    assert {o for (o, *_r) in dfd} == {"Up"}  # Down pulled
    # lean: threatened Down wider (lower touch), safe Up tighter (higher touch).
    ln, _ = sim.desired_quotes(0.5, sig, 120.0, sim.Inventory(), p, mode="lean",
                               threatened_side="Down", lean_mult=1.0)
    up_touch = max(pp for (o, lvl, pp, _s) in ln if o == "Up")
    dn_touch = max(pp for (o, lvl, pp, _s) in ln if o == "Down")
    base_touch = max(pp for (o, lvl, pp, _s) in base if o == "Up")
    assert up_touch > base_touch  # safe side tightened toward fair
    assert dn_touch < base_touch  # threatened side widened away from fair


# --- fill engine (venue-paper::on_trade) ------------------------------------
def _touch_params():
    return sim.QuoteParams(ladder_levels=1)  # touch only, to isolate a single resting order


def test_queue_behind_displayed():
    # our Up buy at 0.49 sits at the best bid (size 100) → queue_ahead 100.
    p = _touch_params()
    up_book = _book(bid_px=0.49, bid_sz=100.0, ask_px=0.55, ask_sz=100.0)
    rows = [(0, 0.5, 0.0002)]
    # a print of size 40 < queue 100 → 0 fills.
    m = _run(rows=rows, up_book=up_book, up_prints=_prints([(1000, 0.49, 40.0, True)]), params=p)
    assert m["fill_shares"] == 0.0
    # a print of size 150 > queue 100 → drains 100, fills the 10-share touch.
    m = _run(rows=rows, up_book=up_book, up_prints=_prints([(1000, 0.49, 150.0, True)]), params=p)
    assert m["fill_shares"] == pytest.approx(10.0)


def test_shared_budget_no_double_eat():
    # two resting buys (0.49, 0.48), both improve an empty-ish bid (queue 0). One 15-size sell
    # must fill AT MOST 15 total across both (not 20) — the print size is a shared budget.
    p = sim.QuoteParams(ladder_levels=2)
    up_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.55, ask_sz=100.0)
    m = _run(rows=[(0, 0.5, 0.0002)], up_book=up_book,
             up_prints=_prints([(1000, 0.47, 15.0, True)]), params=p)
    assert m["fill_shares"] == pytest.approx(15.0)  # 10 (0.49) + 5 (0.48), never 20


def test_only_sell_aggressor_at_or_through_fills():
    p = _touch_params()
    up_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.55, ask_sz=100.0)  # buy 0.49 improves, queue 0
    # buy-aggressor print → no fill.
    m = _run(rows=[(0, 0.5, 0.0002)], up_book=up_book,
             up_prints=_prints([(1000, 0.49, 100.0, False)]), params=p)
    assert m["fill_shares"] == 0.0
    # sell above our bid (0.50 > 0.49) → not through → no fill.
    m = _run(rows=[(0, 0.5, 0.0002)], up_book=up_book,
             up_prints=_prints([(1000, 0.50, 100.0, True)]), params=p)
    assert m["fill_shares"] == 0.0
    # sell at/through our bid → fills.
    m = _run(rows=[(0, 0.5, 0.0002)], up_book=up_book,
             up_prints=_prints([(1000, 0.49, 100.0, True)]), params=p)
    assert m["fill_shares"] == pytest.approx(10.0)


def test_post_only_never_crosses_the_ask():
    # ask 0.485 below our 0.49 touch → post-only rejects the placement → no fill despite sells.
    p = _touch_params()
    up_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.485, ask_sz=100.0)
    m = _run(rows=[(0, 0.5, 0.0002)], up_book=up_book,
             up_prints=_prints([(1000, 0.45, 100.0, True)]), params=p)
    assert m["fill_shares"] == 0.0


# --- reactive cancel + defense ----------------------------------------------
def _down_only(signal=None, mode="baseline", theta=None):
    """A window where only the Down side can ever fill (no Up book/prints); returns fill_shares."""
    down_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.55, ask_sz=100.0)
    # Down sell prints late (t≈20s), so a stale (t=0) signal is >15s old at fill time.
    down_prints = _prints([(20_000, 0.47, 200.0, True)])
    m = sim.simulate_window(
        rows_ts=np.array([0, 20_000], dtype="int64"),
        rows_p_up=np.array([0.5, 0.5], dtype=float),
        rows_sigma=np.array([0.0002, 0.0002], dtype=float),
        up_book=_empty_book(), up_prints=_prints([]),
        down_book=down_book, down_prints=down_prints,
        outcome_up=1, close_ms=200_000, params=sim.QuoteParams(ladder_levels=1),
        mode=mode, signal=signal, theta=theta)
    return m["fill_shares"]


def test_defend_pulls_the_threatened_down_side():
    # signal fires UP (p_up_sig 0.9 ≥ 0.5+0.1) fresh at t=20s → threatened = Down → pulled.
    fresh = _sig([(19_500, 0.9)])
    base = _down_only()
    defended = _down_only(signal=fresh, mode="defend", theta=0.10)
    assert base > 0.0                       # baseline fills the Down side
    assert defended == 0.0                  # defend pulls Down → no fill


def test_defend_stands_down_when_signal_is_stale():
    # only prediction is at t=0; the Down fill is at t=20s → 20 s > 15 s stale → no pull.
    stale = _sig([(0, 0.9)])
    base = _down_only()
    defended = _down_only(signal=stale, mode="defend", theta=0.10)
    assert defended == pytest.approx(base)  # stand-down ⇒ behaves as baseline


def test_signal_fires_respects_staleness():
    signal = _sig([(0, 0.9)])
    assert sim.signal_fires(signal, np.array([10_000], dtype="int64"), 0.10) is True
    assert sim.signal_fires(signal, np.array([20_000], dtype="int64"), 0.10) is False  # >15 s


def test_reactive_cancel_reprices_away_from_a_stale_price():
    # fair rises 0.5→0.51 between rows → Down endangered, cancelled + repriced lower; a Down sell
    # at the OLD price (0.49) then misses the new (0.48) Down buy — the adverse-selection avoidance.
    p = sim.QuoteParams(ladder_levels=1)
    down_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.60, ask_sz=100.0)
    down_prints = _prints([(1500, 0.49, 200.0, True)])  # arrives after the reprice at t=1000

    def _fill(p_up_second):
        m = sim.simulate_window(
            rows_ts=np.array([0, 1000], dtype="int64"),
            rows_p_up=np.array([0.5, p_up_second], dtype=float),
            rows_sigma=np.array([0.0002, 0.0002], dtype=float),
            up_book=_empty_book(), up_prints=_prints([]),
            down_book=down_book, down_prints=down_prints,
            outcome_up=0, close_ms=200_000, params=p)
        return m["fill_shares"]

    flat = _fill(0.5)     # no move → Down stays at 0.49 → the 0.49 sell fills it
    moved = _fill(0.51)   # +0.01 move → Down repriced to 0.48 → the 0.49 sell misses
    assert flat > 0.0 and moved == 0.0


# --- settlement + markout ----------------------------------------------------
def test_locked_plus_stranded_equals_net():
    up_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.55, ask_sz=100.0)
    down_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.55, ask_sz=100.0)
    m = _run(rows=[(0, 0.5, 0.0002)], up_book=up_book, down_book=down_book,
             up_prints=_prints([(1000, 0.47, 60.0, True)]),
             down_prints=_prints([(1200, 0.47, 60.0, True)]), outcome_up=1,
             params=sim.QuoteParams(ladder_levels=2))
    assert m["n_fills"] > 0
    assert m["net_pnl"] == pytest.approx(m["locked_pnl"] + m["stranded_pnl"], abs=1e-9)


def test_markout_sign_follows_fair_move():
    # buy Up at 0.49 at t≈0; fair then rises 0.5→0.6 → positive 5 s markout.
    p = sim.QuoteParams(ladder_levels=1)
    up_book = _book(bid_px=0.40, bid_sz=0.0, ask_px=0.60, ask_sz=100.0)
    m = _run(rows=[(0, 0.5, 0.0002), (5000, 0.6, 0.0002), (10_000, 0.6, 0.0002)],
             up_book=up_book, up_prints=_prints([(100, 0.49, 50.0, True)]), params=p)
    assert m["markout_shares"] == pytest.approx(10.0)  # the 10-share touch fills from the 50 print
    assert m["markout5_wsum"] / m["markout_shares"] == pytest.approx(0.1, abs=1e-9)  # 0.6 − 0.5


def test_determinism():
    kw = dict(rows=[(0, 0.5, 0.0002), (5000, 0.52, 0.0003)],
              up_book=_book(bid_px=0.40, bid_sz=0.0, ask_px=0.60, ask_sz=100.0),
              up_prints=_prints([(1000, 0.47, 80.0, True), (6000, 0.46, 80.0, True)]),
              params=sim.QuoteParams(ladder_levels=2))
    a, b = _run(**kw), _run(**kw)
    assert a == b


def test_shuffled_signal_preserves_fire_count():
    signal = _sig([(0, 0.9), (1000, 0.5), (2000, 0.1), (3000, 0.5)])
    shuffled = bmc._shuffled_signal(signal, 123)
    assert np.array_equal(shuffled["ts"], signal["ts"])          # timestamps unchanged
    assert sorted(shuffled["p_up"]) == sorted(signal["p_up"])    # same values → same fire count


# --- stage: primitive reuse + fixture end-to-end -----------------------------
def test_reuses_money_judge_primitives():
    assert bmc.mj._book_asof is mj._book_asof            # the stage reuses the mj book helper
    assert bmc.sim is sim                                # and the shared pure sim engine
    assert bmc.tpm.read_pm_trades.__name__ == "read_pm_trades"


def test_end_to_end_on_fixture():
    base = Path(tempfile.mkdtemp()) / "mc"
    res = make_historical_fixture(base / "telonex", base / "aggtrades")
    write_historical_coverage(base / "out", res["coverage"])
    hp = Paths(journal_dir=base / "journal", depth_dir=base / "depth", out_dir=base / "out",
               hist_dir=base / "aggtrades", telonex_dir=base / "telonex")
    oos = pd.DataFrame([
        {"series": sk, "window_open_ms": o, "sample_ts_ms": o + k * 15_000,
         "p_up": 0.9 if v["outcome"] == "Up" else 0.1}
        for (sk, o), v in res["res_map"].items() for k in range(1, 20) if k * 15_000 < 300_000])
    r = bmc.backtest_maker_core(hp, series=("BTC-5m", "ETH-5m"), oos=oos, res_map=res["res_map"],
                                regime_from=date(2026, 5, 1), seeds=2,
                                defend_thetas=[0.10, 0.20], lean_mults=[0.5, 1.0])
    assert r["current_regime_windows"] > 0
    cfgs = {c["config"] for c in r["configs"]}
    assert {"baseline", "defend|0.1", "defend|0.2", "lean|0.5", "lean|1"} <= cfgs
    for c in r["configs"]:
        assert abs(c["net_pnl"] - (c["locked_pnl"] + c["stranded_pnl"])) < 0.01
    assert r["baseline"]["n_fills"] > 0  # the PM trade tape actually fills resting quotes
    for fname in ("metrics.json", "per_series.csv", "report.html"):
        assert (base / "out" / "backtests" / "maker_core" / fname).exists()
