"""Deterministic XGBoost wrapper — the only module that imports ``xgboost``.

A challenger tree booster alongside :mod:`model_lab.lib.gbt` (LightGBM). XGBoost is a compiled
library (and pulls ``scipy``), heavier than the lab's pure numpy/pandas/pyarrow core, so it is an
**opt-in** extra (``pip install -r requirements-xgb.txt`` / ``pip install -e .[xgb]``). Isolating the
import here keeps every other module xgboost-free; the cost is paid only when the challenger runs.

Mirrors ``lib/gbt.py`` deliberately — same function shapes, same **modest-depth + strong
regularization + bit-reproducibility-within-a-machine** philosophy. Reproducibility levers
(XGBoost's own FAQ names float-summation order + multi-threading as the only non-determinism
sources): ``tree_method="hist"`` on ``device="cpu"``, ``nthread=1``, a fixed ``seed``, and row/column
sub-sampling left OFF (fractions = 1.0) — regularization is carried by the shallow trees, the large
``min_child_weight``, the small ``max_bin`` and the L2 penalty instead. Cross-OS / cross-version
reproducibility is *not* guaranteed — pin the exact ``xgboost`` version.

Missing values are handled natively (``DMatrix`` treats ``np.nan`` as missing and the tree learns the
default branch direction), so — like LightGBM — no imputation is done here.
"""

from __future__ import annotations

import numpy as np

try:  # localized so the whole lab doesn't need the heavy extra installed.
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - exercised only without the extra.
    raise ImportError(
        "xgboost is not installed. The XGBoost challenger is an opt-in extra — install it with "
        "`pip install -r requirements-xgb.txt` (or `pip install -e .[xgb]`)."
    ) from exc


def default_params(
    seed: int,
    *,
    max_depth: int = 4,
    eta: float = 0.03,
    min_child_weight: float = 100.0,
    reg_lambda: float = 5.0,
    reg_alpha: float = 0.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    max_bin: int = 127,
    nthread: int = 1,
) -> dict:
    """A modest-depth, strongly-regularized, deterministic binary-classification parameter set.

    Row/column sub-sampling is left OFF (fractions = 1.0): sampling order interacts with threading and
    is the most fragile part of reproducibility; regularization is carried by the shallow trees, the
    large ``min_child_weight``, the small ``max_bin`` and L2 instead. A single thread + a fixed
    ``seed`` on ``hist`` gives a byte-reproducible booster for a given seed on a given machine.
    """
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": int(max_depth),
        "eta": float(eta),
        "min_child_weight": float(min_child_weight),
        "lambda": float(reg_lambda),
        "alpha": float(reg_alpha),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "max_bin": int(max_bin),
        "nthread": int(nthread),
        "seed": int(seed),
        "verbosity": 0,
    }


def fit(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    params: dict,
    max_boost_round: int,
    early_stopping_rounds: int,
) -> tuple["xgb.Booster", int]:
    """Train a booster, using an inner validation set for early stopping when one is given. Returns
    ``(booster, best_iteration)`` where ``best_iteration`` is the 0-based best round.

    When the inner-val is empty or ``None`` early stopping is skipped and the full ``max_boost_round``
    rounds are used (``best_iteration`` is then the last round, ``max_boost_round - 1``).
    """
    dtr = xgb.DMatrix(np.ascontiguousarray(x_tr, dtype=np.float32), label=y_tr)  # NaN native
    have_val = x_val is not None and y_val is not None and len(y_val) > 0
    if have_val:
        dval = xgb.DMatrix(np.ascontiguousarray(x_val, dtype=np.float32), label=y_val)
        booster = xgb.train(
            params, dtr, num_boost_round=int(max_boost_round), evals=[(dval, "val")],
            early_stopping_rounds=int(early_stopping_rounds), verbose_eval=False,
        )
        best = int(getattr(booster, "best_iteration", int(max_boost_round) - 1))
        if best < 0:
            best = int(max_boost_round) - 1
        return booster, best
    booster = xgb.train(params, dtr, num_boost_round=int(max_boost_round), verbose_eval=False)
    return booster, int(max_boost_round) - 1


def predict_proba(
    booster: "xgb.Booster",
    x: np.ndarray,
    *,
    num_iteration: int | None = None,
    num_threads: int = 1,
) -> np.ndarray:
    """P(Up) for each row, using trees ``[0, num_iteration]`` (0-based inclusive).

    ``num_iteration`` is the 0-based best round from :func:`fit`; ``iteration_range=(0, n+1)`` selects
    the first ``n+1`` trees. ``None`` (or a negative value) → ``(0, 0)`` = all trees (XGBoost's
    "use every tree" sentinel).
    """
    if num_iteration is None or int(num_iteration) < 0:
        rng = (0, 0)
    else:
        rng = (0, int(num_iteration) + 1)
    booster.set_param({"nthread": int(num_threads)})
    dm = xgb.DMatrix(np.ascontiguousarray(x, dtype=np.float32))
    return np.asarray(booster.predict(dm, iteration_range=rng), dtype=float)


def feature_importance(booster: "xgb.Booster", names: list[str]) -> dict:
    """Gain importance as ``{"gain": {name: float}}``, aligned to ``names``.

    ``get_score`` keys DMatrix-from-array features ``"f0","f1",…`` and omits features never split on;
    absent features map to ``0.0``.
    """
    gain = booster.get_score(importance_type="gain")
    return {"gain": {n: float(gain.get(f"f{i}", 0.0)) for i, n in enumerate(names)}}


def save(booster: "xgb.Booster", path) -> None:
    """Persist the native XGBoost booster to ``path`` (a ``.json`` model file)."""
    booster.save_model(str(path))


def load(path) -> "xgb.Booster":
    """Reload a native XGBoost booster saved by :func:`save`."""
    b = xgb.Booster()
    b.load_model(str(path))
    return b
