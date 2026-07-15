"""Deterministic LightGBM wrapper — the only module that imports ``lightgbm``.

LightGBM is a compiled OpenMP library (and pulls ``scipy``), heavier than the
lab's pure numpy/pandas/pyarrow core, so it is an **opt-in** extra
(``pip install -r requirements-gbt.txt``). Isolating the import here means the GBT
stage's orchestration and every other module stay lightgbm-free; the import cost
is paid only when the second challenger actually runs.

The parameters are tuned for **modest depth + strong regularization** and, above
all, **bit-reproducibility within a machine** (``deterministic=True`` +
``force_row_wise=True`` + a single thread + every seed set), because the lab has a
hard determinism contract (same seed → byte-identical artifacts). Cross-OS /
cross-version reproducibility is *not* guaranteed — pin the exact ``lightgbm``
version.

Two prediction flavours share one wrapper:

- **plain** — a binary GBT on the 16 features; ``predict_proba`` returns the
  booster's probability directly.
- **residual** — the model is *anchored on the formula* via an ``init_score`` base
  margin (``logit(Φ(z))``) so the trees learn only the **correction** over the
  formula. LightGBM's ``predict`` does not carry the *test* base margins, so
  ``predict_proba`` adds them back and applies the sigmoid explicitly.
"""

from __future__ import annotations

import numpy as np

try:  # localized so the whole lab doesn't need the heavy extra installed.
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - exercised only without the extra.
    raise ImportError(
        "lightgbm is not installed. The GBT challenger is an opt-in extra — install it with "
        "`pip install -r requirements-gbt.txt` (or `pip install -e .[gbt]`)."
    ) from exc


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic (matches ``lib/logreg.sigmoid``)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def default_params(
    seed: int,
    *,
    num_leaves: int = 15,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    min_child_samples: int = 100,
    reg_lambda: float = 5.0,
    reg_alpha: float = 0.0,
    max_bin: int = 127,
    num_threads: int = 1,
) -> dict:
    """A modest-depth, strongly-regularized, deterministic binary-GBT parameter set.

    Row/feature sub-sampling is left OFF (fractions = 1.0): it is the most fragile
    part of cross-thread reproducibility and regularization is carried instead by
    the shallow trees, the large ``min_child_samples``, the small ``max_bin`` and
    the L2 penalty. Every seed is pinned and a single thread is used so the booster
    serializes byte-identically for a given seed.
    """
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "min_child_samples": int(min_child_samples),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "max_bin": int(max_bin),
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "num_threads": int(num_threads),
        "deterministic": True,
        "force_row_wise": True,
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "data_random_seed": int(seed),
        "verbosity": -1,
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
    init_score_tr: np.ndarray | None = None,
    init_score_val: np.ndarray | None = None,
) -> tuple["lgb.Booster", int]:
    """Train a booster, using an inner validation set for early stopping when one
    is given. Returns ``(booster, best_iteration)``.

    ``init_score_*`` supplies the per-row formula base margin for the *residual*
    variant (``None`` for the plain variant). When the inner-val is empty or ``None``
    early stopping is skipped and the full ``max_boost_round`` rounds are used
    (``best_iteration`` is then reported as ``max_boost_round``).
    """
    train_set = lgb.Dataset(x_tr, label=y_tr, init_score=init_score_tr, free_raw_data=False)
    have_val = x_val is not None and y_val is not None and len(y_val) > 0
    if have_val:
        valid = lgb.Dataset(
            x_val, label=y_val, init_score=init_score_val, reference=train_set, free_raw_data=False
        )
        booster = lgb.train(
            params, train_set, num_boost_round=int(max_boost_round), valid_sets=[valid],
            callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)],
        )
        best = int(booster.best_iteration or 0)
        if best <= 0:  # never early-stopped → all trees.
            best = int(max_boost_round)
        return booster, best
    booster = lgb.train(params, train_set, num_boost_round=int(max_boost_round))
    return booster, int(max_boost_round)


def refit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: dict,
    num_boost_round: int,
    init_score: np.ndarray | None = None,
) -> "lgb.Booster":
    """Fit a booster on the full training window for a fixed number of rounds (no
    early stopping) — used once the round count has been chosen on the inner val."""
    n = int(num_boost_round) if num_boost_round and int(num_boost_round) > 0 else 1
    train_set = lgb.Dataset(x, label=y, init_score=init_score, free_raw_data=False)
    return lgb.train(params, train_set, num_boost_round=n)


def predict_proba(
    booster: "lgb.Booster",
    x: np.ndarray,
    *,
    num_iteration: int | None = None,
    init_score: np.ndarray | None = None,
    num_threads: int = 1,
) -> np.ndarray:
    """P(Up) for each row.

    Plain variant: the booster's probability directly. Residual variant (pass the
    per-row formula base margin as ``init_score``): predict the raw margin the trees
    add on top of the formula, then ``sigmoid(init_score + raw)`` — LightGBM's
    ``predict`` does not know the *test* base margins, so they are added back here.

    ``num_iteration`` is guarded: in LightGBM ``0`` means "use all trees", so callers
    that hit ``best_iteration == 0`` should pass ``None`` (or a positive count).
    """
    n_iter = None if (num_iteration is None or int(num_iteration) <= 0) else int(num_iteration)
    if init_score is None:
        prob = booster.predict(x, num_iteration=n_iter, num_threads=int(num_threads))
        return np.asarray(prob, dtype=float)
    raw = booster.predict(x, num_iteration=n_iter, raw_score=True, num_threads=int(num_threads))
    return _sigmoid(np.asarray(init_score, dtype=float) + np.asarray(raw, dtype=float))


def feature_importance(booster: "lgb.Booster", names: list[str]) -> dict:
    """Gain + split importances as ``{"gain": {name: float}, "split": {name: int}}``
    (aligned to ``names``, best-iteration weighted)."""
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    return {
        "gain": {n: float(v) for n, v in zip(names, gain)},
        "split": {n: int(v) for n, v in zip(names, split)},
    }


def save(booster: "lgb.Booster", path, num_iteration: int | None = None) -> None:
    """Persist the native LightGBM booster to ``path`` (a ``.txt`` model file)."""
    n_iter = None if (num_iteration is None or int(num_iteration) <= 0) else int(num_iteration)
    booster.save_model(str(path), num_iteration=n_iter)


def load(path) -> "lgb.Booster":
    """Reload a native LightGBM booster saved by :func:`save`."""
    return lgb.Booster(model_file=str(path))
