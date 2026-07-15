"""A tiny, inspectable, numpy-only logistic regression — the learner core.

The lab pins a minimal scientific stack (numpy/pandas/pyarrow/matplotlib) with
**no scikit-learn or scipy**, so the first challenger model is hand-written here.
It is deliberately small and dependency-free, mirroring :mod:`model_lab.lib.math`:
numeric primitives live in ``lib/``; the walk-forward orchestration + I/O live in
the ``learn`` stage.

- :func:`sigmoid` — numerically stable logistic (no overflow at large ``|x|``).
- :func:`standardize_fit` / :func:`standardize_apply` — NaN-aware column
  standardization. ``feature_set`` deliberately leaves NaN in expected places
  (warmup, out-of-aggTrades-coverage flow/intensity); fit ignores them, apply
  imputes them to 0 (the standardized mean) so no row is dropped for a missing
  feature.
- :func:`fit_logistic_ridge` — L2-regularized logistic regression by IRLS /
  Newton (the bias is unpenalized). Deterministic from a zero start; the ridge
  keeps separable data bounded.
- :func:`predict_proba` — the fitted probability of the positive class.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    """Numerically stable logistic ``1 / (1 + e^{-x})``.

    Uses ``e^{-|x|}`` internally so a large positive or negative ``x`` never
    overflows: for ``x ≥ 0`` returns ``1/(1+z)``, for ``x < 0`` returns
    ``z/(1+z)`` with ``z = e^{-|x|}`` — the two algebraically-equal, numerically
    safe branches. Accepts a scalar or array; returns a float array.
    """
    x = np.asarray(x, dtype=float)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std over the finite entries only (NaN-aware).

    Returns ``(mean, std)``. A column with no finite values gets ``mean = 0`` and
    a column whose std underflows (constant/empty) gets ``std = 1`` — so
    :func:`standardize_apply` never divides by zero and an all-missing feature
    contributes nothing rather than corrupting the row. Computed without
    ``np.nanmean`` so an all-NaN column is silent (no warning), which matters for
    ``feature_set``'s legitimately-empty flow columns outside aggTrades coverage.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"standardize_fit expects a 2-D array, got shape {X.shape}")
    finite = np.isfinite(X)
    cnt = finite.sum(axis=0)
    denom = np.maximum(cnt, 1)
    total = np.where(finite, X, 0.0).sum(axis=0)
    mean = np.where(cnt > 0, total / denom, 0.0)
    dev = np.where(finite, X - mean, 0.0)
    var = (dev * dev).sum(axis=0) / denom
    std = np.sqrt(var)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Center/scale ``X`` by ``(mean, std)``, then impute any NaN/inf to 0.

    After centering, ``0`` *is* the (standardized) column mean, so imputing a
    missing feature to 0 is the neutral choice for a linear model — it removes the
    feature's contribution for that row without dropping the row.
    """
    Xs = (np.asarray(X, dtype=float) - np.asarray(mean, dtype=float)) / np.asarray(std, dtype=float)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def fit_logistic_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by IRLS / Newton–Raphson.

    Minimizes ``NLL(w) + ½·l2·‖w‖²`` (the intercept is **not** penalized). Each
    step solves ``(XᵀWX + λI)·Δ = grad`` with ``W = p(1−p)`` (floored) via
    ``np.linalg.solve`` (``lstsq`` fallback if the Hessian is singular). Starts
    from zero weights, so it is fully deterministic; the ridge term keeps weights
    bounded even when the classes are linearly separable. Returns
    ``(coef, intercept)`` with ``coef`` of length ``X.shape[1]``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit_logistic_ridge expects a 2-D X, got shape {X.shape}")
    n, d = X.shape
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)}")
    Xb = np.hstack([np.ones((n, 1)), X])  # bias column first
    w = np.zeros(d + 1)
    pen = np.full(d + 1, float(l2))
    pen[0] = 0.0  # never penalize the intercept
    for _ in range(max_iter):
        p = sigmoid(Xb @ w)
        wgt = np.clip(p * (1.0 - p), 1e-6, None)
        grad = Xb.T @ (p - y) + pen * w
        hess = (Xb * wgt[:, None]).T @ Xb
        hess[np.diag_indices_from(hess)] += pen
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w[1:].astype(float), float(w[0])


def predict_proba(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Positive-class probability ``σ(X·coef + intercept)`` for standardized ``X``."""
    return sigmoid(np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float) + float(intercept))
