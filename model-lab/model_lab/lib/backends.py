"""Uniform model-backend interface for the challenger bake-off.

Four model classes plug into ``accuracy_push``'s one methodology (train → screen price/flow → refit →
predict OOS) through a single seam. Each backend wraps an **unchanged** ``lib/`` module:

- ``gbt``    → :mod:`model_lab.lib.gbt`    (LightGBM, opt-in ``[gbt]``)
- ``xgb``    → :mod:`model_lab.lib.xgb`    (XGBoost, opt-in ``[xgb]``)
- ``mlp``    → :mod:`model_lab.lib.mlp`    (hand-rolled numpy net, no extra)
- ``logreg`` → :mod:`model_lab.lib.logreg` (numpy IRLS logistic ridge, no extra)

A ``Fitted`` hides each model's per-backend state (best_iteration for the tree boosters; the
``(coef, intercept, mean, std)`` / ``(weights, mean, std)`` for the linear/net models) behind a
uniform ``predict_proba(x, *, threads)``. The heavy tree imports are **lazy** (inside the methods) so
this module — and the numpy-only MLP/LogReg backends — import fine without the opt-in extras.

The GBT backend deliberately uses the same round budget / early-stopping as ``accuracy_push`` so its
OOS is byte-identical to ``accuracy_push.run_market`` (the bake-off's anti-drift tripwire test).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from . import logreg  # pure numpy, always importable

# GBT/XGB share accuracy_push's champion budget so GbtBackend reproduces run_market byte-for-byte.
TREE_MAX_ROUND = 400
TREE_EARLY_STOP = 40
LOGREG_L2 = 1.0


@runtime_checkable
class Fitted(Protocol):
    def predict_proba(self, x: np.ndarray, *, threads: int = 1) -> np.ndarray: ...


@runtime_checkable
class Backend(Protocol):
    name: str
    has_importance: bool
    model_ext: str

    def fit(self, x_tr, y_tr, x_val, y_val, *, seed: int, threads: int) -> Fitted: ...
    def feature_importance(self, model: Fitted, names: list[str]) -> dict | None: ...
    def save(self, model: Fitted, path) -> None: ...
    def load(self, path) -> Fitted: ...


# ===========================================================================
# LightGBM
# ===========================================================================
class _GbtFitted:
    def __init__(self, booster, best: int | None):
        self.booster, self.best = booster, best

    def predict_proba(self, x, *, threads: int = 1) -> np.ndarray:
        from . import gbt
        return gbt.predict_proba(self.booster, x, num_iteration=self.best, num_threads=threads)


class GbtBackend:
    name = "gbt"
    has_importance = True
    model_ext = ".txt"

    def fit(self, x_tr, y_tr, x_val, y_val, *, seed: int, threads: int) -> _GbtFitted:
        from . import gbt
        params = gbt.default_params(seed, num_threads=threads)
        booster, best = gbt.fit(x_tr, y_tr, x_val, y_val, params=params,
                                max_boost_round=TREE_MAX_ROUND, early_stopping_rounds=TREE_EARLY_STOP)
        return _GbtFitted(booster, int(best))

    def feature_importance(self, model: _GbtFitted, names: list[str]) -> dict:
        from . import gbt
        return gbt.feature_importance(model.booster, names)

    def save(self, model: _GbtFitted, path) -> None:
        from . import gbt
        gbt.save(model.booster, path, num_iteration=model.best)  # saved truncated to best trees

    def load(self, path) -> _GbtFitted:
        from . import gbt
        return _GbtFitted(gbt.load(path), None)  # None → all (already-truncated) trees


# ===========================================================================
# XGBoost
# ===========================================================================
class _XgbFitted:
    def __init__(self, booster, best: int | None):
        self.booster, self.best = booster, best

    def predict_proba(self, x, *, threads: int = 1) -> np.ndarray:
        from . import xgb
        return xgb.predict_proba(self.booster, x, num_iteration=self.best, num_threads=threads)


class XgbBackend:
    name = "xgb"
    has_importance = True
    model_ext = ".json"

    def fit(self, x_tr, y_tr, x_val, y_val, *, seed: int, threads: int) -> _XgbFitted:
        from . import xgb
        params = xgb.default_params(seed, nthread=threads)
        booster, best = xgb.fit(x_tr, y_tr, x_val, y_val, params=params,
                                max_boost_round=TREE_MAX_ROUND, early_stopping_rounds=TREE_EARLY_STOP)
        return _XgbFitted(booster, int(best))

    def feature_importance(self, model: _XgbFitted, names: list[str]) -> dict:
        from . import xgb
        return xgb.feature_importance(model.booster, names)

    def save(self, model: _XgbFitted, path) -> None:
        from . import xgb
        # persist best_iteration inside the model json (single-file round-trip).
        model.booster.set_attr(bakeoff_best=str(int(model.best)))
        xgb.save(model.booster, path)

    def load(self, path) -> _XgbFitted:
        from . import xgb
        b = xgb.load(path)
        a = b.attr("bakeoff_best")
        return _XgbFitted(b, int(a) if a is not None else None)


# ===========================================================================
# numpy MLP
# ===========================================================================
class _MlpFitted:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, x, *, threads: int = 1) -> np.ndarray:
        from . import mlp
        return mlp.predict_proba(self.model, x, num_threads=threads)


class MlpBackend:
    name = "mlp"
    has_importance = False
    model_ext = ".npz"

    def fit(self, x_tr, y_tr, x_val, y_val, *, seed: int, threads: int) -> _MlpFitted:
        from . import mlp
        params = mlp.default_params(seed)
        return _MlpFitted(mlp.fit(x_tr, y_tr, x_val, y_val, params=params, seed=seed))

    def feature_importance(self, model: _MlpFitted, names: list[str]) -> None:
        return None

    def save(self, model: _MlpFitted, path) -> None:
        from . import mlp
        mlp.save(model.model, path)

    def load(self, path) -> _MlpFitted:
        from . import mlp
        return _MlpFitted(mlp.load(path))


# ===========================================================================
# logistic-regression floor
# ===========================================================================
class _LogRegFitted:
    def __init__(self, coef: np.ndarray, intercept: float, mean: np.ndarray, std: np.ndarray):
        self.coef, self.intercept, self.mean, self.std = coef, intercept, mean, std

    def predict_proba(self, x, *, threads: int = 1) -> np.ndarray:
        xs = logreg.standardize_apply(x, self.mean, self.std)
        return logreg.predict_proba(xs, self.coef, self.intercept)


class LogRegBackend:
    name = "logreg"
    has_importance = False
    model_ext = ".npz"

    def fit(self, x_tr, y_tr, x_val, y_val, *, seed: int, threads: int) -> _LogRegFitted:
        # linear floor: no early stopping (x_val/y_val ignored), deterministic zero-start IRLS.
        mean, std = logreg.standardize_fit(x_tr)
        xs = logreg.standardize_apply(x_tr, mean, std)
        y = np.asarray(y_tr, dtype=np.float64).ravel()
        ok = np.isfinite(y)
        coef, intercept = logreg.fit_logistic_ridge(xs[ok], y[ok], l2=LOGREG_L2)
        return _LogRegFitted(coef, float(intercept), mean, std)

    def feature_importance(self, model: _LogRegFitted, names: list[str]) -> None:
        return None

    def save(self, model: _LogRegFitted, path) -> None:
        np.savez(str(path), coef=model.coef, intercept=np.array(model.intercept),
                 mean=model.mean, std=model.std)

    def load(self, path) -> _LogRegFitted:
        p = str(path)
        if not p.endswith(".npz"):
            p = p + ".npz"
        z = np.load(p, allow_pickle=False)
        return _LogRegFitted(z["coef"], float(z["intercept"]), z["mean"], z["std"])


BACKENDS: dict[str, Backend] = {
    "gbt": GbtBackend(),
    "xgb": XgbBackend(),
    "mlp": MlpBackend(),
    "logreg": LogRegBackend(),
}
