"""Unit tests for forensic_audit — pure accounting + capital-accounting logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_lab import forensic_audit as fa


def _trade(series, open_ms, fire, side, shares, price, outcome_up):
    fee = fa._fee_formula(shares, price)
    won = (side == "Up" and outcome_up == 1) or (side == "Down" and outcome_up == 0)
    payoff = shares if won else 0.0
    cost = shares * price + fee
    return {"series": series, "window_open_ms": open_ms, "sample_ts_ms": fire, "side": side,
            "shares": shares, "price": price, "outcome_up": outcome_up, "fee": fee,
            "cost": cost, "payoff": payoff, "net": payoff - cost, "won": won}


def test_fee_formula_reference_points():
    # 100 shares @ 0.50 → 100·0.07·0.25 = 1.75
    assert abs(fa._fee_formula(100, 0.50) - 1.75) < 1e-9
    assert abs(fa._fee_formula(100, 0.10) - 0.63) < 1e-9
    assert fa._fee_formula(0, 0.5) == 0.0


def test_accounting_check_clean_and_detects_corruption():
    df = pd.DataFrame([_trade("BTC-5m", 0, 100, "Up", 37.0, 0.10, 0),
                       _trade("ETH-5m", 0, 200, "Down", 20.0, 0.30, 0)])
    metrics = {"counts": {"net_pnl": float(df["net"].sum()), "fees": float(df["fee"].sum())}}
    ok = fa.accounting_check(df, metrics)
    assert ok["clean"] and ok["won_mismatches"] == 0 and ok["net_reconciles"] and ok["fee_reconciles"]

    bad = df.copy()
    bad.loc[0, "net"] = bad.loc[0, "net"] + 5.0  # corrupt one net
    res = fa.accounting_check(bad, metrics)
    assert not res["clean"] and res["err_net_identity"] > 1.0


def test_capital_accounting_peak_and_caps():
    # two overlapping BTC-5m trades (close at +300s) + one later
    df = pd.DataFrame([
        {"series": "BTC-5m", "window_open_ms": 0, "sample_ts_ms": 100_000, "cost": 4.0, "payoff": 0.0, "net": -4.0},
        {"series": "BTC-5m", "window_open_ms": 0, "sample_ts_ms": 150_000, "cost": 5.0, "payoff": 10.0, "net": 5.0},
        {"series": "BTC-5m", "window_open_ms": 600_000, "sample_ts_ms": 610_000, "cost": 3.0, "payoff": 6.0, "net": 3.0},
    ])
    cap = fa.capital_accounting(df, caps=[6.0, 9.0])
    assert abs(cap["peak_concurrent_exposure"] - 9.0) < 1e-9  # 4+5 overlap
    assert abs(cap["total_net_pnl"] - 4.0) < 1e-9
    assert abs(cap["pct_return_on_required"] - 4.0 / 9.0) < 1e-9

    by_cap = {c["cap"]: c for c in cap["capped"]}
    # cap 9 affords all three (trade3 after the first two resolve) → net 4, all taken
    assert by_cap[9.0]["n_taken"] == 3 and abs(by_cap[9.0]["net_pnl"] - 4.0) < 1e-9
    # cap 6 affords only trade1 (trade2 needs 5 with only 2 free; trade3 needs 3 with 2 free) → net -4
    assert by_cap[6.0]["n_taken"] == 1 and abs(by_cap[6.0]["net_pnl"] - (-4.0)) < 1e-9


def test_dur_and_slug():
    assert fa._dur_ms("BTC-5m") == 300_000 and fa._dur_ms("ETH-15m") == 900_000
    slug, _day = fa._slug_and_day("BTC-5m", 1780617600000)
    assert slug == "btc-updown-5m-1780617600"
