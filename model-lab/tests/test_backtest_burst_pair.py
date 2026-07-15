"""Unit tests for backtest_burst_pair — the burst pair-building strategy sim + metrics.

All tests drive the pure ``_simulate_window`` with hand-built single-level books (no Telonex
fixture), so the burst entry, completion, inventory cap, final-seconds rule, PnL split, and the
shuffled-control semantics are checked in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model_lab import backtest_burst_pair as bp
from model_lab.config import Paths
from model_lab.lib import math as lm


def _paths(tmp_path):
    return Paths(journal_dir=tmp_path / "j", depth_dir=tmp_path / "d", out_dir=tmp_path / "out")


def _book(ts, bid_px, bid_sz, ask_px, ask_sz):
    n = len(ts)

    def arr(v):
        return np.array(v if isinstance(v, list) else [v] * n, dtype=float)
    return {"UP": {"ts": np.array(ts, dtype=np.int64), "bid_px": arr(bid_px), "bid_sz": arr(bid_sz),
                   "ask_px": arr(ask_px), "ask_sz": arr(ask_sz)}}


def _sim(fires, books, outcome_up, *, C=0.94, U=60.0, close_ms=300_000, latency_ms=0,
         budget=1000.0):
    return bp._simulate_window(fires, books, "UP", outcome_up, close_ms, C=C, U=U,
                               latency_ms=latency_ms, fee_rate=0.07, window_budget=budget)


# --- burst entry / depth shortfall -----------------------------------------
def test_burst_consumes_one_level_and_reports_shortfall():
    # Up ask 0.80 with only 30 displayed; burst target 60 → fills 30 (20+10+0), shortfall 30.
    books = _book([1000], 0.10, 500.0, 0.80, 30.0)
    agg = _sim([(1000, 0.95)], books, outcome_up=1)
    assert abs(agg["shares_entered"] - 30.0) < 1e-9
    assert abs(agg["depth_shortfall_total"] - 30.0) < 1e-9  # 60 target − 30 filled
    assert agg["bursts_fired"] == 1


def test_deep_book_fills_full_60():
    books = _book([1000], 0.10, 500.0, 0.80, 500.0)
    agg = _sim([(1000, 0.95)], books, outcome_up=1)
    assert abs(agg["shares_entered"] - 60.0) < 1e-9  # 3 × 20
    assert abs(agg["depth_shortfall_total"]) < 1e-9


# --- completion economics: positive locked edge 1 − C ----------------------
def test_locked_pair_has_positive_edge_one_minus_C():
    # Up leg @0.80; Down ask (1−bid) drops to 0.12 ≤ limit 0.14 at t=5000 → locks all 60.
    books = _book([1000, 5000], [0.10, 0.88], [500.0, 500.0], [0.80, 0.80], [500.0, 500.0])
    agg = _sim([(1000, 0.95)], books, outcome_up=0, C=0.94)  # outcome irrelevant when locked
    assert abs(agg["shares_locked"] - 60.0) < 1e-9
    assert abs(agg["shares_stranded"]) < 1e-9
    # locked pnl = n·(1 − C) − entry_fee (pair cost = price_s + (C − price_s) = C exactly).
    fee = lm.taker_fee(60.0, 0.07, 0.80)
    assert abs(agg["locked_pnl"] - (60.0 * (1.0 - 0.94) - fee)) < 1e-9
    assert agg["locked_pnl"] > 0.0
    assert abs(agg["net"] - agg["locked_pnl"]) < 1e-9  # no stranded


def test_no_completion_strands_and_settles_on_outcome():
    # Down ask stays 0.90 (bid 0.10) → never ≤ limit → all 60 stranded, ride to resolution.
    books = _book([1000, 5000], [0.10, 0.10], [500.0, 500.0], [0.80, 0.80], [500.0, 500.0])
    won = _sim([(1000, 0.95)], books, outcome_up=1)   # Up wins → stranded pays $1
    lost = _sim([(1000, 0.95)], books, outcome_up=0)  # Up loses → stranded pays $0
    assert abs(won["shares_stranded"] - 60.0) < 1e-9 and won["shares_locked"] == 0.0
    fee = lm.taker_fee(60.0, 0.07, 0.80)
    assert abs(won["net"] - (60.0 - (60.0 * 0.80 + fee))) < 1e-9   # +$1/share, minus cost
    assert abs(lost["net"] - (0.0 - (60.0 * 0.80 + fee))) < 1e-9   # $0, full loss
    assert lost["net"] < 0.0 < won["net"]


# --- reconciliation: net == locked + stranded ------------------------------
def test_reconciliation_locked_plus_stranded():
    # 30 lock (bid rises), 30 strand (partial completion size).
    books = _book([1000, 5000], [0.10, 0.88], [500.0, 30.0], [0.80, 0.80], [500.0, 500.0])
    agg = _sim([(1000, 0.95)], books, outcome_up=1, C=0.96)
    assert abs(agg["net"] - (agg["locked_pnl"] + agg["stranded_pnl"])) < 1e-9
    assert agg["shares_locked"] > 0 and agg["shares_stranded"] > 0  # both components present


# --- inventory cap U blocks new bursts -------------------------------------
def test_cap_U_blocks_new_bursts_until_completion():
    # Two signals; first burst fills 20 (displayed 20), unhedged=20; U=20 blocks the second.
    books = _book([1000, 2000, 3000], 0.10, 500.0, 0.80, 20.0)
    agg = _sim([(1000, 0.95), (2000, 0.95)], books, outcome_up=1, U=20.0)
    assert agg["bursts_fired"] == 1
    assert agg["skip_cap"] == 1


def test_higher_cap_allows_second_burst():
    books = _book([1000, 2000, 3000], 0.10, 500.0, 0.80, 20.0)
    agg = _sim([(1000, 0.95), (2000, 0.95)], books, outcome_up=1, U=60.0)
    assert agg["bursts_fired"] == 2 and agg["skip_cap"] == 0


# --- final-seconds rule -----------------------------------------------------
def test_no_entry_in_final_30s():
    books = _book([1000, 295000], 0.10, 500.0, 0.80, 500.0)
    # fire at close−10s (290000) → within 30 s of close (300000) → no entry.
    agg = _sim([(290000, 0.95)], books, outcome_up=1)
    assert agg["skip_final"] == 1 and agg["bursts_fired"] == 0


# --- shuffled-control semantics: locked pairs outcome-independent -----------
def test_shuffle_only_moves_stranded_not_locked():
    # A mix: 30 lock + 30 strand. Locked pnl identical across outcomes; stranded differs.
    books = _book([1000, 5000], [0.10, 0.88], [500.0, 30.0], [0.80, 0.80], [500.0, 500.0])
    a1 = _sim([(1000, 0.95)], books, outcome_up=1, C=0.96)
    a0 = _sim([(1000, 0.95)], books, outcome_up=0, C=0.96)
    assert abs(a1["locked_pnl"] - a0["locked_pnl"]) < 1e-9   # locked is outcome-independent
    assert abs(a1["stranded_pnl"] - a0["stranded_pnl"]) > 1e-6  # stranded flips with outcome


# --- config key round-trip + quantiles -------------------------------------
def test_config_key_roundtrip():
    for cu in bp.CONFIGS:
        assert bp._ckey(bp._cstr(cu)) == cu


def test_quantiles():
    q = bp._quantiles([10.0, 20.0, 30.0, 40.0, 1000.0])
    assert q["max"] == 1000.0 and q["median"] == 30.0
    assert bp._quantiles([]) == {"median": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}


# --- challenger bake-off: --oos-dir loaders --------------------------------
def test_load_theta_gates_reads_custom_oos_dir(tmp_path):
    paths = _paths(tmp_path)
    sub = paths.out_dir / "chal_xgb"
    sub.mkdir(parents=True)
    metrics = {"markets": {"BTC-5m": {"theta_gate": 0.31, "theta_85_reached": True},
                           "ETH-5m": {"theta_gate": 0.28, "theta_85_reached": False}}}
    (sub / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    gates = bp._load_theta_gates(paths, ["BTC-5m", "ETH-5m"], "chal_xgb")
    assert gates["BTC-5m"] == {"theta_gate": 0.31, "reached_85": True}
    assert gates["ETH-5m"] == {"theta_gate": 0.28, "reached_85": False}


def test_load_burst_fires_reads_custom_oos_dir_and_applies_gate(tmp_path):
    paths = _paths(tmp_path)
    sub = paths.out_dir / "chal_mlp"
    sub.mkdir(parents=True)
    # conf |p-0.5| = 0.40, 0.02, 0.40, 0.05 → only rows 0 & 2 clear a 0.30 gate.
    pd.DataFrame({"series": ["BTC-5m"] * 4, "window_open_ms": [1000, 1000, 2000, 2000],
                  "sample_ts_ms": [10, 20, 30, 40], "p_up": [0.90, 0.52, 0.10, 0.55]}
                 ).to_parquet(sub / "oos_BTC-5m.parquet", index=False)
    fires = bp._load_burst_fires(paths, "BTC-5m", 0.30, 0, None, "chal_mlp")
    assert list(fires) == [("BTC-5m", 1000), ("BTC-5m", 2000)]  # ts-sorted, first-seen group order
    assert [ts for ts, _ in fires[("BTC-5m", 1000)]] == [10]
    assert abs(fires[("BTC-5m", 1000)][0][1] - 0.90) < 1e-9
    assert [ts for ts, _ in fires[("BTC-5m", 2000)]] == [30]


def test_load_burst_fires_default_dir_is_accuracy_push(tmp_path):
    paths = _paths(tmp_path)
    sub = paths.out_dir / "accuracy_push"
    sub.mkdir(parents=True)
    pd.DataFrame({"series": ["BTC-5m"], "window_open_ms": [1000], "sample_ts_ms": [10],
                  "p_up": [0.95]}).to_parquet(sub / "oos_BTC-5m.parquet", index=False)
    fires = bp._load_burst_fires(paths, "BTC-5m", 0.30, 0, None)  # default oos_subdir
    assert list(fires) == [("BTC-5m", 1000)]


def test_verdict_uses_reference_bars_when_benchmarks_skipped():
    cell = {"net_pnl": 100.0, "shuffled_net": -50.0, "locked_pairs_pnl": 1000.0,
            "stranded_legs_pnl": -900.0, "completion_rate": 0.35, "bursts_fired": 10,
            "skip_cap": 0, "skip_final": 0, "avg_shortfall_per_burst": 5.0,
            "shares_per_window": {"p90": 60.0, "max": 120.0}}
    v = bp._verdict({"C0.94_U20": cell}, "C0.94_U20", cell, momentum_net=0.0, dir10_net=0.0,
                    sig_all=[1, 0, 2], n_windows=100, with_momentum=False, with_dir10=False)
    assert "vps255 ref" in v  # both benchmarks display the reference, not a spurious $0
    assert "does NOT beat momentum" in v  # net $100 compared against the $5,057 reference, not $0
