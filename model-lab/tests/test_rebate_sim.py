"""Tests for the taker-rebate simulation (``rebate_sim`` stage).

Covers the tier lookup boundaries + wV formula (``lib/math``), the rolling-30-day
window and per-day tier assignment, the ``$1`` daily-accrual carry, corrected-net,
marginal-to-next-tier, empty/missing-column handling, exec-day source selection, and
the stage's file outputs. All deterministic, no IO except the one output-writing test.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from model_lab import rebate_sim as rs
from model_lab.config import ParquetNotReady, Paths
from model_lab.lib import math as lm


# --------------------------------------------------------------------------- tiers
def test_taker_rebate_tier_boundaries():
    assert lm.taker_rebate_tier(0.0) == ("None", 0.0)
    assert lm.taker_rebate_tier(1_999.99) == ("None", 0.0)
    assert lm.taker_rebate_tier(2_000.0) == ("Bronze", 0.03)
    assert lm.taker_rebate_tier(19_999.99) == ("Bronze", 0.03)
    assert lm.taker_rebate_tier(20_000.0) == ("Silver", 0.08)
    assert lm.taker_rebate_tier(200_000.0) == ("Gold", 0.18)
    assert lm.taker_rebate_tier(1_000_000.0) == ("Platinum", 0.32)
    assert lm.taker_rebate_tier(4_000_000.0) == ("Diamond", 0.44)
    assert lm.taker_rebate_tier(10_000_000.0) == ("Obsidian", 0.50)
    assert lm.taker_rebate_tier(50_000_000.0) == ("Obsidian", 0.50)


def test_weighted_volume_formula():
    # size·(1−entry)·weight; crypto weight defaults to 2.3.
    assert abs(lm.weighted_volume(100.0, 0.5) - 100.0 * 0.5 * 2.3) < 1e-9
    assert abs(lm.weighted_volume(100.0, 0.9) - 100.0 * 0.1 * 2.3) < 1e-9
    assert abs(lm.weighted_volume(100.0, 0.5, weight=1.0) - 50.0) < 1e-9


def test_next_tier():
    assert lm.taker_rebate_next_tier(0.0) == ("Bronze", 2_000.0)
    assert lm.taker_rebate_next_tier(2_000.0) == ("Silver", 18_000.0)
    name, need = lm.taker_rebate_next_tier(4_000_000.0)
    assert name == "Obsidian" and abs(need - 6_000_000.0) < 1e-6
    assert lm.taker_rebate_next_tier(10_000_000.0) == (None, 0.0)


# ---------------------------------------------------------------- rolling timeline
def _daily_frame(n_days: int, *, shares=2000.0, price=0.5, fee=10.0, net=-1.0, day0=20_000):
    """One taker trade per UTC day for ``n_days`` days."""
    return pd.DataFrame([
        {"series": "BTC-5m", "sample_ts_ms": (day0 + k) * rs.MS_PER_DAY,
         "shares": shares, "price": price, "fee": fee, "net": net}
        for k in range(n_days)
    ])


def test_rolling_window_tier_progression():
    # wV = 2000·0.5·2.3 = 2300/day. Trailing-30 [D-30,D-1] crosses Bronze ($2k) on
    # day 1 and Silver ($20k) on day 9; never reaches Gold (30·2300 = $69k).
    res = rs.compute_rebate_timeline(_daily_frame(40))
    assert res["tier_days"] == {"None": 1, "Bronze": 8, "Silver": 31}
    assert rs._peak_tier(res["tier_days"]) == "Silver"
    # Rebate = Σ tier% · $10: 8 Bronze days (3%) + 31 Silver days (8%).
    t = res["totals"]
    assert abs(t["total_rebate"] - (8 * 0.30 + 31 * 0.80)) < 1e-9
    assert abs(t["total_fees"] - 400.0) < 1e-9


def test_corrected_net_is_net_plus_rebate():
    res = rs.compute_rebate_timeline(_daily_frame(40))
    t = res["totals"]
    assert abs(t["original_net"] - (-40.0)) < 1e-9  # 40 trades × net −1
    assert abs(t["corrected_net"] - (t["original_net"] + t["total_rebate"])) < 1e-12


def test_marginal_to_next_tier():
    res = rs.compute_rebate_timeline(_daily_frame(40))
    cur = res["current"]
    # Ending 30-day wV (incl. last day) = 30·2300 = 69,000 → Silver, next Gold ($200k).
    assert cur["tier"] == "Silver"
    assert abs(cur["wv_30d"] - 69_000.0) < 1e-6
    assert cur["next_tier"] == "Gold"
    assert abs(cur["wv_to_next_tier"] - (200_000.0 - cur["wv_30d"])) < 1e-6


def test_dollar_carry_floor():
    # rolling_days=1 → tier(day D) = tier(wv of day D−1). Day 0 seeds Bronze ($2,300),
    # so days 1..9 are Bronze (3% of $10 = $0.30/day). The $1 accrual floor pays out
    # only when the running accrual reaches $1: 0.3,0.6,0.9,1.2→pay, ... twice, with
    # a $0.30 trailing residual left uncredited.
    res = rs.compute_rebate_timeline(_daily_frame(10), rolling_days=1)
    t = res["totals"]
    assert res["tier_days"].get("Bronze") == 9
    assert abs(t["total_rebate"] - 9 * 0.30) < 1e-9      # gross
    assert abs(t["total_paid"] - 2.40) < 1e-9            # two $1.20 payouts
    assert abs(t["residual_unpaid"] - 0.30) < 1e-9       # trailing < $1 carry
    assert t["total_paid"] < t["total_rebate"]
    assert (t["total_rebate"] - t["total_paid"]) < 1.0


def test_no_tier_below_threshold():
    # Small volume never reaches Bronze → zero rebate, correct marginal-to-Bronze.
    res = rs.compute_rebate_timeline(_daily_frame(5, shares=10.0))  # wV = 11.5/day
    assert res["totals"]["total_rebate"] == 0.0
    assert res["current"]["tier"] == "None"
    assert res["current"]["next_tier"] == "Bronze"


# --------------------------------------------------------------- force_tier (C)
def test_force_tier_credits_fixed_pct_regardless_of_volume():
    # Tiny volume (naturally "None") forced to Platinum (32%) — the override bypasses
    # the wV→tier gate: every day credits 0.32 × that day's fees.
    df = _daily_frame(5, shares=10.0, fee=10.0)  # wV 11.5/day → earns "None" naturally
    res = rs.compute_rebate_timeline(df, force_tier="Platinum")
    assert res["force_tier"] == "Platinum"
    # Every timeline day is credited at Platinum regardless of the (real) low wV.
    assert all(row["tier"] == "Platinum" and abs(row["rebate_pct"] - 0.32) < 1e-12
               for row in res["timeline"])
    assert all(abs(row["rebate_day"] - 0.32 * row["fees_day"]) < 1e-12 for row in res["timeline"])
    t = res["totals"]
    assert abs(t["total_rebate"] - 0.32 * t["total_fees"]) < 1e-9      # 0.32 × $50
    assert abs(t["corrected_net"] - (t["original_net"] + t["total_rebate"])) < 1e-12
    # wV column still reflects the real (tiny) earned volume — honest.
    assert res["current"]["wv_30d"] < 2_000.0
    assert res["current"]["tier"] == "Platinum" and res["current"]["next_tier"] is None


def test_force_tier_bronze_vs_owner_side_by_side():
    # The two-run pattern backtest_clones uses: same trades, Bronze vs owner tier.
    df = _daily_frame(10, shares=10.0, fee=10.0)
    bronze = rs.compute_rebate_timeline(df, force_tier="Bronze")
    owner = rs.compute_rebate_timeline(df, force_tier="Obsidian")
    assert abs(bronze["totals"]["total_rebate"] - 0.03 * 100.0) < 1e-9
    assert abs(owner["totals"]["total_rebate"] - 0.50 * 100.0) < 1e-9
    assert owner["totals"]["total_rebate"] > bronze["totals"]["total_rebate"]


def test_force_tier_unknown_raises():
    with pytest.raises(ValueError, match="not a known tier"):
        rs.compute_rebate_timeline(_daily_frame(3), force_tier="Titanium")


def test_force_tier_empty_frame():
    res = rs.compute_rebate_timeline(pd.DataFrame(columns=list(rs._REQUIRED)), force_tier="Gold")
    assert res["force_tier"] == "Gold"
    assert res["current"]["tier"] == "Gold" and res["current"]["rebate_pct"] == 0.18


def test_exec_day_from_sample_ts_over_day_column():
    # Two rows on the same window-open day but different fire (sample) days must land
    # in different exec-days (rebate uses fire time, not window open).
    df = pd.DataFrame([
        {"shares": 1.0, "price": 0.5, "fee": 1.0, "net": 0.0, "day": 20_000,
         "sample_ts_ms": 20_000 * rs.MS_PER_DAY + 10},
        {"shares": 1.0, "price": 0.5, "fee": 1.0, "net": 0.0, "day": 20_000,
         "sample_ts_ms": 20_001 * rs.MS_PER_DAY + 10},
    ])
    res = rs.compute_rebate_timeline(df)
    assert res["counts"]["n_days"] == 2


def test_empty_frame():
    res = rs.compute_rebate_timeline(pd.DataFrame(columns=list(rs._REQUIRED)))
    assert res["counts"]["n_trades"] == 0
    assert res["totals"]["total_rebate"] == 0.0
    assert res["totals"]["corrected_net"] == 0.0


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        rs.compute_rebate_timeline(pd.DataFrame({"shares": [1.0], "price": [0.5]}))


def test_missing_trades_file_raises(tmp_path):
    paths = Paths(journal_dir=tmp_path, depth_dir=tmp_path, out_dir=tmp_path / "out")
    with pytest.raises(ParquetNotReady, match="trades.csv not found"):
        rs.rebate_sim(paths, trades=tmp_path / "nope.csv")


def test_rebate_sim_writes_outputs(tmp_path):
    trades = tmp_path / "trades.csv"
    _daily_frame(40).to_csv(trades, index=False)
    paths = Paths(journal_dir=tmp_path, depth_dir=tmp_path, out_dir=tmp_path / "out")
    res = rs.rebate_sim(paths, trades=trades, out_name="t")
    out_dir = tmp_path / "out" / "backtests" / "rebate_sim" / "t"
    for fname in ("tier_timeline.csv", "metrics.json", "report.html"):
        assert (out_dir / fname).exists(), f"{fname} not written"
    # metrics.json round-trips and matches the returned dict's headline.
    meta = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert abs(meta["totals"]["total_rebate"] - res["totals"]["total_rebate"]) < 1e-12
    tl = pd.read_csv(out_dir / "tier_timeline.csv")
    assert len(tl) == res["counts"]["n_days"]
    assert "Silver" in set(tl["tier"])
