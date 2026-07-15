"""Tests for the pair-lean strategy backtest.

Pure-sim invariants (hand-built book arrays + fires): the maker completion locks at limit
``L = C − price_s`` with zero fee, a leg that never completes rides to resolution, ``price_s ≥ C``
can never complete, the $10 budget bounds multi-firing, and the sim is deterministic. The
matched-frequency random control + the post-hoc shuffled-labels control behave. An end-to-end run
over the historical fixture produces the metrics structure, the baselines on identical windows, and
all output files; it reuses money_judge's fill primitives (no re-derived mechanics).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from model_lab import backtest_pair_lean as bpl
from model_lab import fixtures as fx
from model_lab import money_judge as mj
from model_lab.config import Paths


# --- helpers ----------------------------------------------------------------
def _ramp_book(n: int = 301, up_target: float = 0.99) -> dict:
    """Up-token book, mid ramping 0.50 → up_target over the window (Up winning)."""
    ts = np.array([i * 1000 for i in range(n)], dtype="int64")
    mid = np.array([0.50 + (up_target - 0.50) * min(1.0, (i / (n - 1)) ** 1.5) for i in range(n)], float)
    bid = np.clip(mid - 0.005, 0.001, 0.999)
    ask = np.clip(mid + 0.005, 0.001, 0.999)
    return {"ts": ts, "bid_px": bid, "bid_sz": np.full(n, 100.0),
            "ask_px": ask, "ask_sz": np.full(n, 100.0)}


def _fires(p_up: float, tss, outcome_up: int = 1, close_ms: int = 300_000) -> pd.DataFrame:
    return pd.DataFrame({"series": "BTC-5m", "window_open_ms": 0,
                         "sample_ts_ms": np.array(tss, dtype="int64"),
                         "p_up": np.array([p_up] * len(tss), float), "up_token": "T",
                         "outcome_up": outcome_up, "close_time": close_ms})


# --- pure sim ---------------------------------------------------------------
def test_lean_shares_depth_frac_budget_min_notional():
    # depth fraction caps size; budget caps size; sub-$1 notional → 0
    assert bpl._lean_shares(0.10, 100.0, 10.0, 0.5) == 50.0        # 0.5*100 < 10/0.10=100
    assert bpl._lean_shares(0.10, 1000.0, 10.0, 1.0) == 100.0       # budget binds: 10/0.10
    assert bpl._lean_shares(0.10, 5.0, 10.0, 0.25) == 0.0           # 1.25 sh * 0.10 = $0.125 < $1


def test_maker_completion_crossing_and_none():
    book = _ramp_book()
    # complete Down: fills when 1 - Up_bid <= L. Late in the ramp Up_bid ~0.985 → Down ask ~0.015.
    sz = bpl._maker_completion(book, "Up", 0.30, 0, 300_000)
    assert sz is not None and sz > 0.0
    # an impossible limit (Down ask never drops below 0.0) → no completion
    assert bpl._maker_completion(book, "Up", -0.01, 0, 300_000) is None
    # empty scan window → None
    assert bpl._maker_completion(book, "Up", 0.30, 100_000, 100_000) is None


def test_lock_completes_at_limit_zero_fee():
    book = _ramp_book()
    rows_by_c, nf = bpl._run_pair_lean(_fires(0.90, [30_000]), threshold=0.05, pair_costs=[0.98],
                                       depth_frac=1.0, latency_ms=0, fee_rate=0.07,
                                       window_budget=10.0, books={"T": book})
    assert nf == 1
    r = rows_by_c[0.98][0]
    assert r["completed"] and r["locked"] > 0.0
    # combined pair cost equals the ceiling C exactly, and the locked profit = m*(1-C) - fee
    assert abs(r["price_s"] + r["price_maker"] - 0.98) < 1e-9
    assert r["locked_pnl"] > 0.0
    # the maker leg pays ZERO fee: total fee equals the taker fee on the leaned leg only
    from model_lab.lib import math as lm
    assert abs(r["fee"] - lm.taker_fee(r["n"], 0.07, r["price_s"])) < 1e-12


def test_strand_rides_to_resolution():
    book = _ramp_book()
    # Lean Down while Up wins → the Down leg can't complete cheaply and strands, LOSING at resolution.
    rc, _ = bpl._run_pair_lean(_fires(0.10, [30_000], outcome_up=1), threshold=0.05, pair_costs=[0.90],
                               depth_frac=1.0, latency_ms=0, fee_rate=0.07, window_budget=10.0,
                               books={"T": book})
    r = rc[0.90][0]
    assert r["side"] == "Down" and not r["completed"]
    assert r["stranded"] == r["n"] and r["net"] < 0.0     # Down lost → stranded pays $0
    # A stranding leg on the winning side profits (bought below $1, settles $1).
    rc2, _ = bpl._run_pair_lean(_fires(0.90, [290_000], outcome_up=1), threshold=0.05, pair_costs=[0.90],
                                depth_frac=1.0, latency_ms=0, fee_rate=0.07, window_budget=10.0,
                                books={"T": book})
    r2 = rc2[0.90][0]
    assert not r2["completed"] and r2["stranded"] == r2["n"] and r2["net"] > 0.0


def test_price_s_ge_C_never_completes():
    book = _ramp_book()
    # Deep, late leaned Up: price_s ~0.99 ≥ C=0.90 → L ≤ 0 → no maker bid possible → fully stranded.
    rc, _ = bpl._run_pair_lean(_fires(0.90, [290_000]), threshold=0.05, pair_costs=[0.90],
                               depth_frac=1.0, latency_ms=0, fee_rate=0.07, window_budget=10.0,
                               books={"T": book})
    r = rc[0.90][0]
    assert r["price_s"] >= 0.90 and not r["completed"] and r["stranded"] == r["n"]


def test_multifire_bounded_by_budget():
    book = _ramp_book()
    # Many qualifying snapshots at ~0.5 (each ~$5 lean) → the $10 window budget bounds actual trades.
    tss = [10_000 + i * 5_000 for i in range(20)]
    rc, nf = bpl._run_pair_lean(_fires(0.90, tss), threshold=0.05, pair_costs=[0.98],
                                depth_frac=1.0, latency_ms=0, fee_rate=0.07, window_budget=10.0,
                                books={"T": book})
    assert nf == 20                       # every snapshot is a fire attempt
    spent = sum(row["n"] * row["price_s"] for row in rc[0.98])
    assert spent <= 10.0 + 1e-9           # notional never exceeds the window budget
    assert len(rc[0.98]) < 20             # budget exhausts before all 20 fill


def test_sim_deterministic():
    book = _ramp_book()
    f = _fires(0.90, [30_000, 60_000, 90_000])
    a, _ = bpl._run_pair_lean(f, threshold=0.03, pair_costs=[0.94, 0.98], depth_frac=0.5,
                              latency_ms=0, fee_rate=0.07, window_budget=10.0, books={"T": book})
    b, _ = bpl._run_pair_lean(f, threshold=0.03, pair_costs=[0.94, 0.98], depth_frac=0.5,
                              latency_ms=0, fee_rate=0.07, window_budget=10.0, books={"T": book})
    assert a == b


def test_shuffled_net_locked_is_outcome_independent():
    book = _ramp_book()
    # A fully-locked config: shuffling window outcomes must not change net (a pair pays $1 regardless).
    rc, _ = bpl._run_pair_lean(_fires(0.90, [30_000]), threshold=0.05, pair_costs=[0.98],
                               depth_frac=1.0, latency_ms=0, fee_rate=0.07, window_budget=10.0,
                               books={"T": book})
    rows = rc[0.98]
    assert rows and all(r["completed"] and r["stranded"] == 0.0 for r in rows)
    real = sum(r["net"] for r in rows)
    shuf = bpl._shuffled_net(rows, "BTC-5m", "2026-05-01", 7)
    assert abs(shuf - real) < 1e-9
    # deterministic
    assert bpl._shuffled_net(rows, "BTC-5m", "2026-05-01", 7) == shuf


def test_random_control_matched_frequency():
    book = _ramp_book()
    f = _fires(0.90, [10_000 + i * 5_000 for i in range(20)])
    none = bpl._run_pair_lean_random(f, fire_prob=0.0, pair_costs=[0.98], depth_frac=1.0,
                                     rng=np.random.default_rng(1), latency_ms=0, fee_rate=0.07,
                                     window_budget=10.0, books={"T": book})
    assert none[1][0.98] == 0                      # fire_prob 0 → no trades
    some = bpl._run_pair_lean_random(f, fire_prob=1.0, pair_costs=[0.98], depth_frac=1.0,
                                     rng=np.random.default_rng(1), latency_ms=0, fee_rate=0.07,
                                     window_budget=10.0, books={"T": book})
    assert some[1][0.98] >= 1                       # fire_prob 1 → trades


def test_agg_and_config_row():
    book = _ramp_book()
    rc, nf = bpl._run_pair_lean(_fires(0.90, [30_000, 290_000]), threshold=0.05, pair_costs=[0.98],
                                depth_frac=0.3, latency_ms=0, fee_rate=0.07, window_budget=10.0,
                                books={"T": book})
    rows = rc[0.98]
    agg = bpl._agg_rows(rows)
    row = bpl._config_row(agg, nf)
    assert row["trades"] == len(rows) and row["n_fires"] == nf
    # completion_rate = fraction of leaned fires that completed a pair
    assert row["completion_rate"] == sum(1 for r in rows if r["completed"]) / len(rows)
    assert abs(row["locked_pnl"] + row["stranded_pnl"] - row["net_pnl"]) < 1e-9


def test_best_config_tiebreak_prefers_higher_theta_then_C_then_lean():
    net_by = {
        (0.01, 0.90, 0.5): {"net_pnl": 5.0, "trades": 3},
        (0.05, 0.98, 1.0): {"net_pnl": 5.0, "trades": 3},   # same net → higher θ wins
        (0.03, 0.94, 0.25): {"net_pnl": 4.0, "trades": 3},
    }
    assert bpl._best_config(net_by) == (0.05, 0.98, 1.0)
    assert bpl._best_config({(0.0, 0.9, 0.25): {"net_pnl": 1.0, "trades": 0}}) is None  # no trades


def test_reuses_money_judge_primitives():
    # the pair-lean sim must reuse money_judge's fill primitives, not re-derive them
    assert bpl.mj._book_asof is mj._book_asof
    assert bpl.mj._leaned_quote is mj._leaned_quote
    assert bpl.mj._side_of is mj._side_of
    assert bpl.mj.MIN_NOTIONAL == 1.0


# --- end-to-end over the historical fixture ---------------------------------
def _oos_from_resmap(res_map: dict) -> pd.DataFrame:
    rows = []
    for (series, open_ms), v in res_map.items():
        close_ms = open_ms + 300_000
        p = 0.85 if v["outcome"] == "Up" else 0.15
        for k in range(1, 9):
            ts = open_ms + k * 30_000
            if ts >= close_ms:
                break
            rows.append({"series": series, "window_open_ms": open_ms, "sample_ts_ms": ts, "p_up": p})
    return pd.DataFrame(rows)


def _run_fixture(tmp_path, sub: str = "a"):
    res = fx.make_historical_fixture(tmp_path / sub / "telonex", tmp_path / sub / "aggtrades")
    fx.write_historical_coverage(tmp_path / sub / "out", res["coverage"])
    paths = Paths(journal_dir=tmp_path / sub / "journal", depth_dir=tmp_path / sub / "depth",
                  out_dir=tmp_path / sub / "out", hist_dir=tmp_path / sub / "aggtrades",
                  telonex_dir=tmp_path / sub / "telonex")
    r = bpl.backtest_pair_lean(paths, series=("BTC-5m", "ETH-5m"),
                               oos=_oos_from_resmap(res["res_map"]), res_map=res["res_map"],
                               regime_from=date(2026, 5, 1), seeds=3, thetas=[0.0, 0.1],
                               pair_costs=[0.94, 0.98], depth_fracs=[1.0])
    return paths, r


def test_end_to_end_structure_and_files(tmp_path):
    paths, r = _run_fixture(tmp_path)
    assert r["current_regime_windows"] > 0
    assert r["best_config"] is not None                       # something traded
    assert len(r["sweep"]) == 2 * 2 * 1                        # thetas × pair_costs × depth_fracs
    bl = r["baselines"]
    assert bl["never_trading_net"] == 0.0
    assert "net_pnl" in bl["momentum"] and "net_pnl" in bl["model_taker"]
    # per-series present with best + controls
    for sk in ("BTC-5m", "ETH-5m"):
        assert sk in r["per_series"]
        ps = r["per_series"][sk]
        assert "best" in ps and "control" in ps and "shuffled" in ps
    # the best config carries its matched-frequency control verdict
    bc = r["best_config"]
    assert "control" in bc and "p_value" in bc["control"] and "beats" in bc["control"]
    assert "shuffled" in bc
    out = paths.out_dir / "backtests" / "pair_lean"
    for f in ("metrics.json", "sweep.csv", "per_series_best.csv", "report.html"):
        assert (out / f).exists()
    assert (out / "checkpoints").is_dir()


def test_end_to_end_deterministic(tmp_path):
    _, a = _run_fixture(tmp_path, "a")
    _, b = _run_fixture(tmp_path, "b")
    assert a["best_config"]["net_pnl"] == b["best_config"]["net_pnl"]
    assert [s["net_pnl"] for s in a["sweep"]] == [s["net_pnl"] for s in b["sweep"]]
    assert a["baselines"]["momentum"]["net_pnl"] == b["baselines"]["momentum"]["net_pnl"]
