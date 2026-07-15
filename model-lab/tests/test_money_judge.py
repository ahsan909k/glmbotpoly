"""Tests for the short-horizon money judge (``money_judge`` stage).

Unit-tests the fee replica, the CLOB Down-side mirror, the momentum trigger ports
(confirmed-move + edge gate), and the sim primitives (no-book skip, budget
exhaustion, pair-leg locked vs stranded booking); plus fixture-journal integration
that a **profitable** (oracle-nudged) signal books positive net PnL after fees while a
**random** signal bleeds fees, and that the sim is deterministic. LightGBM-free — the
money judge consumes the OOS parquet + journal only, so this runs in the core suite.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model_lab import money_judge as mj
from model_lab.config import Paths
from model_lab.fixtures import make_fixture, make_short_oos_fixture
from model_lab.lib import math as lm


# --------------------------------------------------------------------------- unit
def test_taker_fee_reference_points():
    # Per 100 shares (CLAUDE.md §7 fee table), symmetric around 0.50.
    assert abs(lm.taker_fee(100, 0.07, 0.50) - 1.75) < 1e-9
    assert abs(lm.taker_fee(100, 0.07, 0.25) - 1.3125) < 1e-9
    assert abs(lm.taker_fee(100, 0.07, 0.75) - 1.3125) < 1e-9
    assert abs(lm.taker_fee(100, 0.07, 0.10) - 0.63) < 1e-9
    assert abs(lm.taker_fee(100, 0.07, 0.01) - 0.0693) < 1e-9
    # Zero inputs → 0 with NO floor; a tiny non-zero raw floors to $0.00001.
    assert lm.taker_fee(0.0, 0.07, 0.5) == 0.0
    assert lm.taker_fee(100, 0.07, 1.0) == 0.0
    assert lm.taker_fee(0.0001, 0.07, 0.5) == 1e-5  # raw 1.75e-6 rounds to 0 → floored to $0.00001
    # Half-away-from-zero at a clean 5th-dp .5 boundary (rate 1, p 0.5 → raw = shares/4).
    assert lm.taker_fee(0.0005, 1.0, 0.5) == 0.00013  # raw 1.25e-4 → 0.00013 (not banker's 0.00012)


def test_majority_baseline():
    b = lm.majority_baseline([1, 1, 1, 0])
    assert b["n"] == 4 and abs(b["p"] - 0.75) < 1e-12
    assert abs(b["diracc"] - 0.75) < 1e-12
    assert abs(b["brier"] - 0.1875) < 1e-12
    empty = lm.majority_baseline([])
    assert empty["n"] == 0 and not np.isfinite(empty["brier"])


def test_mirror_correctness():
    book_row = (0.40, 120.0, 0.60, 50.0)  # bid_px, bid_sz, ask_px, ask_sz
    up = mj._leaned_quote(book_row, "Up")
    down = mj._leaned_quote(book_row, "Down")
    assert up == (0.60, 50.0)  # buy Up ask
    assert down == (1.0 - 0.40, 120.0)  # buy Down = 1 − bid, size = bid_sz (mirror)
    # Fee symmetry: p(1-p) identical whether Up-ask or the mirrored Down price.
    assert abs(lm.taker_fee(10, 0.07, 0.60) - lm.taker_fee(10, 0.07, 1.0 - 0.60)) < 1e-12
    # Unusable quotes rejected.
    assert mj._leaned_quote((0.40, 0.0, 0.60, 0.0), "Up") is None
    assert mj._leaned_quote((0.0, 10.0, 1.0, 10.0), "Down") is None


def test_confirmed_direction():
    # Up ramp over 2s beyond 3σ√T fires Up.
    ts = np.array([0, 1000, 2000], dtype=np.int64)
    up = np.array([100.0, 100.5, 101.0])
    assert mj._confirmed_direction((ts, up), 2000, sigma_1s=1e-4) == "Up"
    down = np.array([101.0, 100.5, 100.0])
    assert mj._confirmed_direction((ts, down), 2000, sigma_1s=1e-4) == "Down"
    # Flat / high-σ / <2 ticks / unusable σ → no signal.
    flat = np.array([100.0, 100.0, 100.0])
    assert mj._confirmed_direction((ts, flat), 2000, sigma_1s=1e-4) is None
    assert mj._confirmed_direction((ts, up), 2000, sigma_1s=1.0) is None
    assert mj._confirmed_direction((ts[:1], up[:1]), 2000, sigma_1s=1e-4) is None
    assert mj._confirmed_direction((ts, up), 2000, sigma_1s=0.0) is None


def test_momentum_edge_gate():
    # fair 0.85 vs ask 0.80: edge 0.05 clears fee(0.80)=0.0112 + buffer 0.005 → takes.
    assert mj._plan_take_single_level(0.85, 0.80, 50.0, 10.0, 0.07) > 0
    # fair 0.522 vs ask 0.50: edge 0.022 < fee(0.5)=0.0175 + 0.005 = 0.0225 → refuse.
    assert mj._plan_take_single_level(0.522, 0.50, 50.0, 10.0, 0.07) == 0.0
    # fair 0.524: edge 0.024 > 0.0225 → take.
    assert mj._plan_take_single_level(0.524, 0.50, 50.0, 10.0, 0.07) > 0
    # Book already repriced (ask ≥ fair) → skip.
    assert mj._plan_take_single_level(0.60, 0.60, 50.0, 10.0, 0.07) == 0.0


def _one_window_fires(sample_ts, p_up, *, outcome_up=1, close_time=300_000):
    return pd.DataFrame([{
        "series": "BTC-5m", "window_open_ms": 0, "sample_ts_ms": int(t), "p_up": float(pu),
        "up_token": "UP", "outcome_up": int(outcome_up), "close_time": int(close_time), "dur": 300.0,
    } for t, pu in zip(sample_ts, p_up)])


def _book(ts, bid_px, bid_sz, ask_px, ask_sz):
    n = len(ts)

    def arr(v):
        return np.array(v if isinstance(v, list) else [v] * n, dtype=float)
    return {"UP": {"ts": np.array(ts, dtype=np.int64), "bid_px": arr(bid_px), "bid_sz": arr(bid_sz),
                   "ask_px": arr(ask_px), "ask_sz": arr(ask_sz)}}


def test_no_book_at_fire_plus_latency_skips():
    books = _book([1000, 2000, 3000], 0.40, 100.0, 0.60, 50.0)
    fires = _one_window_fires([6000], [0.9])  # 6000 − 3000 = 3000 > 2000 staleness
    rows, n_fires = mj._run_taker(fires, threshold=0.1, latency_ms=0, fee_rate=0.07,
                                  window_budget=10.0, trade_budget=10.0, books=books)
    assert n_fires == 1 and len(rows) == 0  # counted as a fire, no trade (conservative)


def test_budget_exhaustion():
    books = _book([1000, 2000, 3000], 0.40, 500.0, 0.60, 500.0)
    fires = _one_window_fires([1000, 2000, 3000], [0.9, 0.9, 0.9])
    rows, n_fires = mj._run_taker(fires, threshold=0.1, latency_ms=0, fee_rate=0.07,
                                  window_budget=2.0, trade_budget=10.0, books=books)
    assert n_fires == 3
    spent = sum(r["shares"] * r["price"] for r in rows)
    assert spent <= 2.0 + 1e-9  # per-window budget never exceeded
    assert len(rows) >= 1


def test_pairleg_stranded_booked():
    # Leaned Up @0.60; Down completion price = 1 − bid stays at 0.60 → combined 1.20, never < 1.
    books = _book([1000, 7000], [0.40, 0.40], [100.0, 100.0], [0.60, 0.60], [20.0, 20.0])
    fires = _one_window_fires([1000], [0.9], outcome_up=0)  # S=Up loses (Down wins)
    rows, _ = mj._run_pairleg(fires, threshold=0.1, latency_ms=0, fee_rate=0.07,
                              window_budget=10.0, trade_budget=10.0, comp_min_s=5.0, comp_max_s=15.0,
                              lock_threshold=1.0, books=books)
    assert len(rows) == 1
    r = rows[0]
    assert r["locked"] == 0 and r["locked_pnl"] == 0.0
    assert abs(r["stranded_pnl"] - r["net"]) < 1e-12 and r["net"] < 0.0  # honest loss booked


def test_pairleg_locked_booked():
    # bid rises to 0.65 at t=7000 → Down price 0.35, combined 0.60+0.35=0.95 < 1 → locks.
    books = _book([1000, 7000], [0.40, 0.65], [100.0, 100.0], [0.60, 0.60], [20.0, 20.0])
    fires = _one_window_fires([1000], [0.9], outcome_up=0)
    rows, _ = mj._run_pairleg(fires, threshold=0.1, latency_ms=0, fee_rate=0.07,
                              window_budget=10.0, trade_budget=10.0, comp_min_s=5.0, comp_max_s=15.0,
                              lock_threshold=1.0, books=books)
    r = rows[0]
    assert r["locked"] > 0 and r["locked_pnl"] > 0.0  # a locked pair nets ~$0.05/share − fees
    assert r["stranded"] < 1e-9  # fully completed


# ------------------------------------------------- sell / momentum-exit primitives (D)
def test_sell_mirror_correctness():
    book_row = (0.40, 120.0, 0.60, 50.0)  # bid_px, bid_sz, ask_px, ask_sz
    assert mj._leaned_sell_quote(book_row, "Up") == (0.40, 120.0)  # hit the Up bid
    assert mj._leaned_sell_quote(book_row, "Down") == (1.0 - 0.60, 50.0)  # Down bid = 1 − ask
    # Unusable quotes rejected (zero size, degenerate prices).
    assert mj._leaned_sell_quote((0.40, 0.0, 0.60, 0.0), "Up") is None
    assert mj._leaned_sell_quote((0.0, 10.0, 1.0, 10.0), "Up") is None  # sell Up at 0.0
    assert mj._leaned_sell_quote((0.0, 10.0, 1.0, 10.0), "Down") is None  # sell Down at 1−1.0=0


def test_sell_at_bid_partial_fill_and_no_book():
    books = _book([1000], 0.40, 30.0, 0.60, 50.0)  # only 30 displayed at the Up bid
    sell = mj._sell_at_bid(books, "UP", 1000, "Up", 100.0, 0.07)
    assert sell is not None
    assert sell["exit_shares"] == 30.0 and sell["unsold"] == 70.0  # sells only displayed size
    assert abs(sell["proceeds"] - 30.0 * 0.40) < 1e-12
    assert abs(sell["sell_fee"] - lm.taker_fee(30.0, 0.07, 0.40)) < 1e-12
    # No fresh book within staleness → no exit possible.
    assert mj._sell_at_bid(books, "UP", 4000, "Up", 10.0, 0.07) is None  # 4000−1000 > 2000


def test_momentum_exit_converged():
    # Enter Up @0.60; PM mid rises to 0.70 at t=3000 and the reconstructed fair is 0.70
    # → convergence, sell at the 0.68 bid. Fee is charged BOTH ways.
    books = _book([1000, 2000, 3000, 4000],
                  [0.55, 0.55, 0.68, 0.68], 100.0, [0.65, 0.65, 0.72, 0.72], 100.0)
    entry_fee = lm.taker_fee(10.0, 0.07, 0.60)
    fair_fn = lambda t: 0.70  # side-oriented (Up) fair, constant
    r = mj._mark_with_momentum_exit("Up", 10.0, 0.60, entry_fee, books, "UP", 1000,
                                    fair_fn, close_ms=300_000, outcome_up=1, fee_rate=0.07,
                                    timeout_s=60.0)
    assert r["exit_reason"] == "converged"
    assert abs(r["exit_price"] - 0.68) < 1e-12
    sell_fee = lm.taker_fee(10.0, 0.07, 0.68)
    assert abs(r["fee"] - (entry_fee + sell_fee)) < 1e-12  # both-way fee
    # net = proceeds(6.80) − entry(6.00+entry_fee) − sell_fee.
    assert abs(r["net"] - (6.80 - 6.00 - entry_fee - sell_fee)) < 1e-9
    assert r["won"] is True and abs(r["cost"] - (10.0 * 0.60 + entry_fee + sell_fee)) < 1e-12


def test_momentum_exit_timeout_sells_at_deadline():
    # Fair stays far from mid → never converges; timeout_s=2 forces an exit at t=3000.
    books = _book([1000, 2000, 3000, 4000], 0.55, 100.0, 0.65, 100.0)  # mid 0.60 throughout
    entry_fee = lm.taker_fee(10.0, 0.07, 0.60)
    r = mj._mark_with_momentum_exit("Up", 10.0, 0.60, entry_fee, books, "UP", 1000,
                                    lambda t: 0.99, close_ms=300_000, outcome_up=1,
                                    fee_rate=0.07, timeout_s=2.0)
    assert r["exit_reason"] == "timeout" and r["exit_ts_ms"] == 3000
    assert abs(r["exit_price"] - 0.55) < 1e-12  # sold at the Up bid


def test_momentum_exit_no_book_marks_to_resolution():
    # Only an early book; by the (large) timeout deadline there is no fresh book to sell
    # into → the position is held to resolution (Up wins here).
    books = _book([1000], 0.55, 100.0, 0.65, 100.0)
    entry_fee = lm.taker_fee(10.0, 0.07, 0.60)
    r = mj._mark_with_momentum_exit("Up", 10.0, 0.60, entry_fee, books, "UP", 1000,
                                    lambda t: 0.99, close_ms=300_000, outcome_up=1,
                                    fee_rate=0.07, timeout_s=60.0)
    assert r["exit_reason"] == "resolution"
    assert abs(r["fee"] - entry_fee) < 1e-12  # no sell leg → entry fee only
    assert abs(r["net"] - (10.0 * 1.0 - (10.0 * 0.60 + entry_fee))) < 1e-9  # $1 payoff on the winner


# --------------------------------------------------------------- fixture integration
def _paths(tmp_path, out="out"):
    make_fixture(tmp_path / "journal", tmp_path / "depth")
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", out_dir=tmp_path / out)


def _threshold0(m):
    return next(s for s in m["sweep"] if s["threshold"] == 0.0)


def test_profitable_signal_makes_money(tmp_path):
    paths = _paths(tmp_path)
    make_short_oos_fixture(paths.out_dir, paths.journal_dir, oracle=True)
    m = mj.money_judge(paths, scope="all", variant="full")
    assert m["counts"]["fires"] > 0
    assert m["best"] is not None
    assert m["best"]["taker"]["net_pnl"] > 0.0  # oracle side = winner → profit after fees
    assert _threshold0(m)["taker"]["net_pnl"] > 0.0
    assert m["best"]["taker"]["win_rate"] > 0.9
    assert m["lift"]["diracc_lift"] > 0.0  # beats always-common-side


def test_venue_delay_is_additive_with_network_latency(tmp_path):
    paths = _paths(tmp_path)
    make_short_oos_fixture(paths.out_dir, paths.journal_dir, oracle=True)
    # network 100 + venue 150  ==  network 250 + venue 0  (both 250 ms effective fill latency).
    a = mj.money_judge(paths, scope="all", variant="full", latency_ms=100, venue_delay_ms=150)
    b = mj.money_judge(paths, scope="all", variant="full", latency_ms=250, venue_delay_ms=0)
    assert a["params"]["network_ms"] == 100 and a["params"]["venue_delay_ms"] == 150
    assert a["params"]["effective_latency_ms"] == 250 == b["params"]["effective_latency_ms"]
    assert a["params"]["latency_ms"] == 250  # `latency_ms` records the effective fill latency

    def sim(m):
        return json.dumps({k: m[k] for k in ("sweep", "comparison", "best")},
                          default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
                          sort_keys=True)
    assert sim(a) == sim(b)  # identical effective latency → byte-identical fills


def test_random_signal_loses_fees(tmp_path):
    paths = _paths(tmp_path)
    make_short_oos_fixture(paths.out_dir, paths.journal_dir, oracle=False, seed=3)
    m = mj.money_judge(paths, scope="all", variant="full")
    assert m["counts"]["fires"] > 0
    # At the most-permissive threshold, an uncorrelated signal bleeds spread + fees.
    assert _threshold0(m)["taker"]["net_pnl"] < 0.0


def test_sim_determinism(tmp_path):
    paths_a = _paths(tmp_path, out="out_a")
    paths_b = Paths(journal_dir=paths_a.journal_dir, depth_dir=paths_a.depth_dir,
                    out_dir=tmp_path / "out_b")
    make_short_oos_fixture(paths_a.out_dir, paths_a.journal_dir, oracle=True)
    make_short_oos_fixture(paths_b.out_dir, paths_b.journal_dir, oracle=True)
    a = mj.money_judge(paths_a, scope="all", variant="full")
    b = mj.money_judge(paths_b, scope="all", variant="full")

    def strip(m):
        m = dict(m)
        m.pop("peak_rss_mb", None)
        return json.dumps({k: m[k] for k in ("sweep", "comparison", "best", "per_day", "per_window")},
                          default=lambda o: float(o) if isinstance(o, np.floating) else str(o), sort_keys=True)
    assert strip(a) == strip(b)


def test_missing_oos_parquet_rejected(tmp_path):
    from model_lab.config import ParquetNotReady
    paths = _paths(tmp_path)  # no learn_short_gbt/oos parquet written
    try:
        mj.money_judge(paths, scope="all", variant="full")
    except ParquetNotReady:
        return
    raise AssertionError("expected ParquetNotReady when the OOS parquet is absent")
