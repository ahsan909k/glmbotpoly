"""Unit tests for the deterministic XGBoost wrapper (opt-in extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")  # opt-in extra — skip the whole module without it.

from model_lab.lib import math as lm
from model_lab.lib import xgb as xw


def _clean_signal(n: int, seed: int, *, feats: int = 6):
    """A dataset where feature 0 carries the label and the rest are noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, feats)).astype(np.float32)
    y = (x[:, 0] + 0.25 * rng.normal(0, 1, n) > 0).astype(np.float64)
    return x, y


def test_recovers_a_clean_signal():
    x, y = _clean_signal(9000, 0)
    xtr, ytr, xva, yva = x[:6000], y[:6000], x[6000:7500], y[6000:7500]
    xte, yte = x[7500:], y[7500:]
    params = xw.default_params(seed=0, nthread=1)
    booster, best = xw.fit(xtr, ytr, xva, yva, params=params, max_boost_round=300,
                           early_stopping_rounds=30)
    prob = xw.predict_proba(booster, xte, num_iteration=best, num_threads=1)
    assert prob.shape == (len(yte),) and np.all((prob >= 0) & (prob <= 1))
    assert lm.directional_accuracy(prob, yte) > 0.9  # recovers feature-0 sign


def test_same_seed_byte_identical_predictions():
    x, y = _clean_signal(6000, 1)
    xtr, ytr, xva, yva = x[:4000], y[:4000], x[4000:5000], y[4000:5000]
    xte = x[5000:]
    p = xw.default_params(seed=42, nthread=1)
    b1, k1 = xw.fit(xtr, ytr, xva, yva, params=p, max_boost_round=200, early_stopping_rounds=25)
    b2, k2 = xw.fit(xtr, ytr, xva, yva, params=p, max_boost_round=200, early_stopping_rounds=25)
    assert k1 == k2
    pr1 = xw.predict_proba(b1, xte, num_iteration=k1, num_threads=1)
    pr2 = xw.predict_proba(b2, xte, num_iteration=k2, num_threads=1)
    assert np.array_equal(pr1, pr2)  # byte-identical, same seed, single thread


def test_gain_importance_finds_the_planted_feature():
    x, y = _clean_signal(8000, 2, feats=6)
    params = xw.default_params(seed=2, nthread=1)
    booster, best = xw.fit(x[:6000], y[:6000], x[6000:7000], y[6000:7000],
                           params=params, max_boost_round=200, early_stopping_rounds=25)
    names = [f"c{i}" for i in range(6)]
    gain = xw.feature_importance(booster, names)["gain"]
    assert set(gain) == set(names)  # aligned to names, absent features → 0.0
    assert max(gain, key=gain.get) == "c0"  # the label-carrying feature dominates


def test_nan_rows_tolerated_natively():
    x, y = _clean_signal(5000, 3, feats=5)
    x[::7, 2] = np.nan  # scatter NaN through a feature column
    x[::11, 4] = np.nan
    params = xw.default_params(seed=3, nthread=1)
    booster, best = xw.fit(x[:3500], y[:3500], x[3500:4200], y[3500:4200],
                           params=params, max_boost_round=150, early_stopping_rounds=20)
    prob = xw.predict_proba(booster, x[4200:], num_iteration=best, num_threads=1)
    assert np.all(np.isfinite(prob))  # NaN inputs → finite probabilities (native missing handling)


def test_save_load_round_trips(tmp_path):
    x, y = _clean_signal(4000, 4)
    params = xw.default_params(seed=4, nthread=1)
    booster, best = xw.fit(x[:3000], y[:3000], x[3000:3500], y[3000:3500],
                           params=params, max_boost_round=120, early_stopping_rounds=20)
    xte = x[3500:]
    before = xw.predict_proba(booster, xte, num_iteration=best, num_threads=1)
    path = tmp_path / "xgb.json"
    xw.save(booster, path)
    reloaded = xw.load(path)
    after = xw.predict_proba(reloaded, xte, num_iteration=best, num_threads=1)
    assert np.array_equal(before, after)
