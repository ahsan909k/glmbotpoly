"""Tests for the competitor-clone backtests (``backtest_clones``).

Offline via ``make_historical_fixture`` (books + PM trade tapes + res_map), with the
regime floor pointed at the fixture day so its windows count as "current-regime". Covers:
the shared ``_TRADE_COLS`` schema + fee/cost/net identities, the maker-fill bracket
ordering (pessimistic ≤ optimistic), control determinism, capital accounting, the
Bronze-vs-owner tier honesty, the momentum-exit sweep, the clone-vs-owner ladder, and
determinism. LightGBM-free, no network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from model_lab import backtest_clones as bc
from model_lab import fixtures as fx
from model_lab.config import Paths
from model_lab.lib import math as lm

FIX_DAY = date(2026, 5, 1)  # make_historical_fixture default; used as the regime floor here


def _paths(tmp_path):
    telonex, hist = tmp_path / "telonex", tmp_path / "hist"
    res = fx.make_historical_fixture(telonex, hist, day=FIX_DAY)
    fx.write_historical_coverage(tmp_path / "out", res["coverage"])
    paths = Paths(journal_dir=tmp_path, depth_dir=tmp_path, out_dir=tmp_path / "out",
                  hist_dir=hist, telonex_dir=telonex)
    return paths, res


def _run(paths, res, clone, **kw):
    return bc.backtest_clone(paths, clone, series=("BTC-5m", "ETH-5m"), res_map=res["res_map"],
                             regime_from=FIX_DAY, seeds=4, **kw)


# --------------------------------------------------------------- unit
def test_maker_fill_bracket_ordering():
    import pandas as pd
    book = {"ts": __import__("numpy").array([0], dtype="int64"),
            "bid_px": __import__("numpy").array([0.49]), "bid_sz": __import__("numpy").array([30.0]),
            "ask_px": __import__("numpy").array([0.51]), "ask_sz": __import__("numpy").array([30.0])}
    trades = pd.DataFrame({"ts_ms": [100, 200], "price": [0.48, 0.47], "size": [50.0, 60.0],
                           "side": ["sell", "sell"]})
    opt = bc.maker_fill_optimistic(book, trades, 0.49, 0, 1000, our_size=40.0)
    pess = bc.maker_fill_pessimistic(book, trades, 0.49, 0, 1000, our_size=40.0)
    assert opt == 40.0            # front of queue: fills full size from 110 sh of sell flow
    assert pess == 40.0           # 110 − 30 queue = 80 ≥ 40 → still full
    # a bigger queue starves the pessimistic fill but never the optimistic.
    book2 = {**book, "bid_sz": __import__("numpy").array([100.0])}
    assert bc.maker_fill_pessimistic(book2, trades, 0.49, 0, 1000, 40.0) == 10.0  # 110−100
    assert bc.maker_fill_optimistic(book2, trades, 0.49, 0, 1000, 40.0) == 40.0
    assert bc.maker_fill_pessimistic(book2, trades, 0.49, 0, 1000, 40.0) <= \
        bc.maker_fill_optimistic(book2, trades, 0.49, 0, 1000, 40.0)


# --------------------------------------------------------------- taker clone
def test_taker_clone_trades_and_identities(tmp_path):
    paths, res = _paths(tmp_path)
    m = _run(paths, res, "takerner")
    import pandas as pd
    df = pd.read_csv(paths.out_dir / "backtests" / "clone_takerner" / "trades.csv")
    assert list(df.columns) == bc.mj._TRADE_COLS
    assert len(df) > 0
    for r in df.itertuples(index=False):
        assert abs(r.fee - lm.taker_fee(r.shares, m["params"]["fee_rate"], r.price)) < 1e-9
        assert abs(r.cost - (r.shares * r.price + r.fee)) < 1e-9
        assert abs(r.net - (r.payoff - r.cost)) < 1e-9
    assert m["capital"]["peak_concurrent_exposure"] >= 0.0
    assert "shuffled_outcome" in m["controls"] and "matched_frequency_random" in m["controls"]


def test_taker_clone_tier_honesty(tmp_path):
    paths, res = _paths(tmp_path)
    m = _run(paths, res, "takerner")
    t = m["tier_honesty"]
    assert t["our_tier"]["tier"] == "Bronze"
    assert t["owner_tier"]["tier"] == "Platinum"
    # A taker clone pays real fees → owner (32%) rebate strictly exceeds Bronze (3%).
    assert t["owner_tier"]["rebate"] >= t["our_tier"]["rebate"]
    assert abs(t["our_tier"]["corrected_net"] - (m["clone"]["net_pnl"] + t["our_tier"]["rebate"])) < 1e-6


# --------------------------------------------------------------- maker clone
def test_maker_clone_bracket_and_zero_fee(tmp_path):
    paths, res = _paths(tmp_path)
    m = _run(paths, res, "0xb27b")
    b = m["maker_fill_bracket"]
    assert b is not None
    assert b["pessimistic_net"] <= b["optimistic_net"] + 1e-9  # bracket well-ordered
    import pandas as pd
    df = pd.read_csv(paths.out_dir / "backtests" / "clone_0xb27b" / "trades.csv")
    if len(df):
        assert float(df["fee"].abs().sum()) == 0.0  # makers pay no taker fee
    # tier axis is immaterial for a maker (rebate ≈ 0 at any tier).
    t = m["tier_honesty"]
    if t.get("our_tier"):
        assert abs(t["owner_tier"]["rebate"]) < 1e-6


# --------------------------------------------------------------- momentum-exit
def test_momentum_exit_sweep_present(tmp_path):
    paths, res = _paths(tmp_path)
    m = _run(paths, res, "takerner")
    me = m["momentum_exit"]
    assert "hold_to_resolution_net" in me
    assert set(me["by_timeout"]) == {"T15", "T30", "T60"}


# --------------------------------------------------------------- ladder
def test_clone_vs_owner_ladder_reads_manual(tmp_path):
    paths, res = _paths(tmp_path)
    mdir = tmp_path / "manuals"
    mdir.mkdir()
    # a minimal manual carrying the slice ladder inputs the clone report appends (3) to.
    (mdir / "takerner.json").write_text(json.dumps({"anchor": {"slice_ladder": {
        "official_account_wide_slice_usd": 38896.0,
        "owner_our_series_reconstructed_net_usd": -18434.0}}}), encoding="utf-8")
    m = _run(paths, res, "takerner", manuals_dir=mdir)
    lad = m["clone_vs_owner_ladder"]
    assert lad["official_slice_usd"] == 38896.0
    assert lad["owner_reconstructed_net_usd"] == -18434.0
    assert lad["gap_official_minus_reconstructed"] == 38896.0 - (-18434.0)
    assert lad["gap_reconstructed_minus_clone"] is not None  # gap 2→3 printed, not smoothed


# --------------------------------------------------------------- determinism + guards
def test_determinism(tmp_path):
    paths, res = _paths(tmp_path)
    a = _run(paths, res, "takerner")
    b = _run(paths, res, "takerner")
    assert a["clone"]["net_pnl"] == b["clone"]["net_pnl"]
    assert a["controls"]["shuffled_outcome"]["nets"] == b["controls"]["shuffled_outcome"]["nets"]
    assert a["controls"]["matched_frequency_random"]["nets"] == b["controls"]["matched_frequency_random"]["nets"]


def test_unknown_clone_raises(tmp_path):
    paths, res = _paths(tmp_path)
    with pytest.raises(ValueError, match="unknown clone"):
        bc.backtest_clone(paths, "nobody", res_map=res["res_map"], regime_from=FIX_DAY, seeds=1)
