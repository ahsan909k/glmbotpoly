"""Uniform-interface contract tests for the four bake-off backends."""

from __future__ import annotations

import numpy as np
import pytest

from model_lab.lib import backends as bk
from model_lab.lib import math as lm

_DEP = {"gbt": "lightgbm", "xgb": "xgboost", "mlp": None, "logreg": None}
_NAMES = ["mlp", "logreg", "gbt", "xgb"]


def _skip_if_missing(name: str) -> None:
    dep = _DEP[name]
    if dep is not None:
        pytest.importorskip(dep)


def _clean(n: int, seed: int, feats: int = 6):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, feats)).astype(np.float32)
    y = (x[:, 0] + 0.25 * rng.normal(0, 1, n) > 0).astype(np.float64)
    return x, y


def test_registry_has_four_named_backends():
    assert set(bk.BACKENDS) == {"gbt", "xgb", "mlp", "logreg"}
    for name, be in bk.BACKENDS.items():
        assert be.name == name
        assert isinstance(be.has_importance, bool)
        assert be.model_ext.startswith(".")


@pytest.mark.parametrize("name", _NAMES)
def test_fit_predict_shape_range_and_signal(name):
    _skip_if_missing(name)
    be = bk.BACKENDS[name]
    x, y = _clean(6000, 0)
    fitted = be.fit(x[:4000], y[:4000], x[4000:5000], y[4000:5000], seed=0, threads=1)
    assert isinstance(fitted, bk.Fitted)
    prob = fitted.predict_proba(x[5000:], threads=1)
    assert prob.shape == (1000,)
    assert np.all((prob >= 0) & (prob <= 1))
    assert lm.directional_accuracy(prob, y[5000:]) > 0.75  # every backend recovers a linear signal


@pytest.mark.parametrize("name", _NAMES)
def test_feature_importance_contract(name):
    _skip_if_missing(name)
    be = bk.BACKENDS[name]
    x, y = _clean(3000, 1)
    fitted = be.fit(x[:2000], y[:2000], x[2000:2500], y[2000:2500], seed=1, threads=1)
    imp = be.feature_importance(fitted, [f"c{i}" for i in range(6)])
    if be.has_importance:
        assert isinstance(imp, dict) and "gain" in imp
        assert set(imp["gain"]) == {f"c{i}" for i in range(6)}
    else:
        assert imp is None


@pytest.mark.parametrize("name", _NAMES)
def test_save_load_round_trips_to_identical_predictions(name, tmp_path):
    _skip_if_missing(name)
    be = bk.BACKENDS[name]
    x, y = _clean(3000, 2)
    fitted = be.fit(x[:2000], y[:2000], x[2000:2500], y[2000:2500], seed=2, threads=1)
    xte = x[2500:]
    before = fitted.predict_proba(xte, threads=1)
    path = tmp_path / f"model_{name}{be.model_ext}"
    be.save(fitted, path)
    reloaded = be.load(path)
    after = reloaded.predict_proba(xte, threads=1)
    assert np.array_equal(before, after)
