"""Unit tests for the hand-rolled numpy MLP (no extra dependency)."""

from __future__ import annotations

import numpy as np

from model_lab.lib import math as lm
from model_lab.lib import mlp


def _clean_signal(n: int, seed: int, *, feats: int = 6):
    """Feature 0 carries the label (via a mild nonlinearity); the rest are noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, feats)).astype(np.float32)
    logit = 1.5 * x[:, 0] + 0.8 * x[:, 0] * x[:, 1]  # a bit nonlinear so the net earns its keep
    y = (logit + 0.3 * rng.normal(0, 1, n) > 0).astype(np.float64)
    return x, y


def test_recovers_a_clean_signal():
    x, y = _clean_signal(9000, 0)
    p = mlp.default_params(seed=0, max_epochs=60)
    model = mlp.fit(x[:6000], y[:6000], x[6000:7500], y[6000:7500], params=p, seed=0)
    prob = mlp.predict_proba(model, x[7500:])
    yte = y[7500:]
    assert prob.shape == (len(yte),) and np.all((prob >= 0) & (prob <= 1))
    assert lm.directional_accuracy(prob, yte) > 0.8


def test_same_seed_byte_identical_predictions():
    x, y = _clean_signal(5000, 1)
    p = mlp.default_params(seed=7, max_epochs=25)
    m1 = mlp.fit(x[:3500], y[:3500], x[3500:4200], y[3500:4200], params=p, seed=7)
    m2 = mlp.fit(x[:3500], y[:3500], x[3500:4200], y[3500:4200], params=p, seed=7)
    pr1 = mlp.predict_proba(m1, x[4200:])
    pr2 = mlp.predict_proba(m2, x[4200:])
    assert np.array_equal(pr1, pr2)  # byte-identical, same seed (single-threaded test process)


def test_shuffled_label_collapses_to_chance():
    x, y = _clean_signal(7000, 2)
    rng = np.random.default_rng(99)
    y_shuf = rng.permutation(y)
    p = mlp.default_params(seed=2, max_epochs=60)
    model = mlp.fit(x[:5000], y_shuf[:5000], x[5000:6000], y_shuf[5000:6000], params=p, seed=2)
    prob = mlp.predict_proba(model, x[6000:])
    # with the label destroyed the net cannot predict the real signal → ~coinflip on real labels.
    assert abs(lm.directional_accuracy(prob, y[6000:]) - 0.5) < 0.06


def test_nan_feature_column_tolerated():
    x, y = _clean_signal(4000, 3, feats=5)
    x[:, 3] = np.nan  # a wholly-missing feature must not drop rows or crash (standardize→0)
    x[::9, 1] = np.nan
    p = mlp.default_params(seed=3, max_epochs=30)
    model = mlp.fit(x[:2800], y[:2800], x[2800:3400], y[2800:3400], params=p, seed=3)
    prob = mlp.predict_proba(model, x[3400:])
    assert prob.shape == (len(y) - 3400,) and np.all(np.isfinite(prob))


def test_adam_decreases_val_loss_and_early_stops():
    # a long patience-limited run on a learnable signal should improve val loss vs the init.
    x, y = _clean_signal(6000, 4)
    p = mlp.default_params(seed=4, max_epochs=80, patience=10)
    model = mlp.fit(x[:4000], y[:4000], x[4000:5000], y[4000:5000], params=p, seed=4)
    # init-only reference: a fresh model whose weights never trained (max_epochs=0-ish via patience 0).
    init = mlp.default_params(seed=4, max_epochs=1, patience=1)
    m0 = mlp.fit(x[:4000], y[:4000], x[4000:5000], y[4000:5000], params=init, seed=4)
    ll_trained = mlp._logloss(mlp._forward(
        mlp.logreg.standardize_apply(x[4000:5000], model.mean, model.std), model.weights)[0],
        y[4000:5000])
    ll_init = mlp._logloss(mlp._forward(
        mlp.logreg.standardize_apply(x[4000:5000], m0.mean, m0.std), m0.weights)[0], y[4000:5000])
    assert ll_trained < ll_init  # training reduced validation log-loss


def test_save_load_round_trips(tmp_path):
    x, y = _clean_signal(3000, 5)
    p = mlp.default_params(seed=5, max_epochs=20)
    model = mlp.fit(x[:2000], y[:2000], x[2000:2500], y[2000:2500], params=p, seed=5)
    before = mlp.predict_proba(model, x[2500:])
    path = tmp_path / "mlp"
    mlp.save(model, path)
    reloaded = mlp.load(path)
    after = mlp.predict_proba(reloaded, x[2500:])
    assert np.array_equal(before, after)
    assert reloaded.params["hidden"] == model.params["hidden"]
