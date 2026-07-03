"""Unit tests for the engine-math reproductions."""

from __future__ import annotations

import math

import numpy as np

from model_lab.lib import math as lm


def test_norm_cdf_known_values():
    assert abs(float(lm.norm_cdf(0.0)) - 0.5) < 1e-12
    assert abs(float(lm.norm_cdf(1.959963985)) - 0.975) < 1e-6
    # Symmetry: Φ(z) + Φ(−z) == 1.
    for z in (-2.0, -0.5, 0.3, 3.1):
        assert abs(float(lm.norm_cdf(z)) + float(lm.norm_cdf(-z)) - 1.0) < 1e-12


def test_norm_cdf_vectorized():
    z = np.array([-1.0, 0.0, 1.0])
    out = lm.norm_cdf(z)
    assert out.shape == (3,)
    assert out[0] < out[1] < out[2]


def test_fair_p_up_saturates_at_expiry():
    # τ ≤ 0 → the ≥-ties-Up resolution rule.
    assert lm.fair_p_up(101.0, 100.0, 0.001, 0.0) == 1.0
    assert lm.fair_p_up(99.0, 100.0, 0.001, 0.0) == 0.0
    assert lm.fair_p_up(100.0, 100.0, 0.001, 0.0) == 1.0  # tie → Up
    # At the money mid-window → ~0.5.
    p = lm.fair_p_up(100.0, 100.0, 0.001, 60.0)
    assert abs(p - 0.5) < 1e-9


def test_sigma_1s_recovers_constant_return():
    # A constant per-second log return r → σ_1s converges to |r| after warmup.
    r = 0.001
    n = 200
    secs = np.arange(n, dtype=np.int64)
    prices = np.exp(secs * r)
    sigma = lm.sigma_1s_from_bars(secs, prices, warmup=60)
    assert math.isnan(sigma[10])  # before warmup
    assert abs(sigma[-1] - r) < 1e-6


def test_sigma_1s_reanchors_across_a_gap():
    # A gap larger than max_gap_secs freezes (does not fold a huge return).
    secs = np.array([0, 1, 2, 100, 101], dtype=np.int64)
    prices = np.array([100.0, 100.1, 100.2, 200.0, 200.1], dtype=float)
    sigma = lm.sigma_1s_from_bars(secs, prices, warmup=1, max_gap_secs=30)
    # The 100.2 → 200.0 jump spans a 98 s gap and must not blow up the estimate.
    assert sigma[-1] < 0.01


def test_one_second_bars_keeps_last_per_second():
    ts = np.array([0, 400, 900, 1000, 1500, 2100], dtype=np.int64)
    px = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    secs, prices = lm.one_second_bars(ts, px)
    assert list(secs) == [0, 1, 2]
    assert list(prices) == [3.0, 5.0, 6.0]  # last of second 0, 1, 2


def test_auc_separates():
    score = np.array([0.1, 0.2, 0.8, 0.9])
    label = np.array([0, 0, 1, 1])
    assert lm.auc(score, label) == 1.0
    assert lm.auc(-score, label) == 0.0
    # A single class → nan.
    assert math.isnan(lm.auc(score, np.array([1, 1, 1, 1])))


def test_brier_and_reliability():
    prob = np.array([0.0, 1.0, 0.0, 1.0])
    outcome = np.array([0, 1, 0, 1])
    assert lm.brier_score(prob, outcome) == 0.0
    mids, preds, rates = lm.reliability_curve(prob, outcome, n_bins=10)
    assert len(mids) == len(preds) == len(rates)
    assert np.allclose(preds, rates)  # perfectly calibrated


def test_log_loss_known_values():
    # Perfect predictions (after the ±eps clamp) → ~0.
    prob = np.array([1.0, 0.0, 1.0, 0.0])
    outcome = np.array([1, 0, 1, 0])
    assert lm.log_loss(prob, outcome) < 1e-10
    # A confident-but-wrong prediction is large but FINITE (clamp prevents inf).
    wrong = lm.log_loss(np.array([1.0]), np.array([0.0]))
    assert math.isfinite(wrong) and wrong > 20.0
    # Constant p = 0.5 against a balanced outcome → ln 2.
    half = lm.log_loss(np.array([0.5, 0.5]), np.array([1, 0]))
    assert abs(half - math.log(2.0)) < 1e-12
    # Empty → nan.
    assert math.isnan(lm.log_loss(np.array([]), np.array([])))


def test_reliability_table_counts():
    prob = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    outcome = np.array([0, 1, 0, 1, 1])
    rows = lm.reliability_table(prob, outcome, n_bins=10)
    # Counts sum to the number of predictions; every returned bin is non-empty.
    assert sum(r["n"] for r in rows) == 5
    assert all(r["n"] > 0 for r in rows)
    # [0.0, 0.1) holds the single 0.05 prediction.
    first = next(r for r in rows if r["bin_lo"] == 0.0)
    assert first["n"] == 1
    # [0.1, 0.2) holds two, empirical Up rate 0.5.
    b2 = next(r for r in rows if r["bin_lo"] == 0.1)
    assert b2["n"] == 2 and abs(b2["empirical_rate"] - 0.5) < 1e-12
    # p == 1.0 lands in the last bin (closed on the right) alongside 0.95.
    last = next(r for r in rows if r["bin_hi"] == 1.0)
    assert last["n"] == 2
