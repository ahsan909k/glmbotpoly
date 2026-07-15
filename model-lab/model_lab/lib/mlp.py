"""A tiny, inspectable, numpy-only multilayer perceptron — the neural-net challenger.

The lab pins a minimal scientific stack (numpy/pandas/pyarrow/matplotlib) with **no
scikit-learn/scipy/torch**, so — like :mod:`model_lab.lib.logreg` — the net is hand-written here. It
is deliberately small, dependency-free, and **byte-deterministic for a given seed on a given machine**
(the lab's hard determinism contract), which a framework net (PyTorch) would only satisfy behind
several global flags. Cross-BLAS-thread-count reproducibility is not guaranteed — run single-threaded
(``OMP_NUM_THREADS=1``); ``float64`` is used throughout so the stored probabilities are stable.

Architecture (honest defaults for ~24 tabular features, not a strawman): NaN-aware input
standardization (reused from :mod:`logreg`) → Dense(64) → ReLU → inverted-dropout(0.1) → Dense(32) →
ReLU → inverted-dropout(0.1) → Dense(1) → sigmoid, trained with mini-batch **Adam** + decoupled L2
(AdamW-style) on binary cross-entropy, with **early stopping on validation log-loss** (patience,
restore-best). A single ``np.random.default_rng(seed)`` drives He-normal init, the per-epoch shuffle,
and every dropout mask in a fixed draw order → reproducible.
"""

from __future__ import annotations

import json

import numpy as np

from . import logreg


def default_params(
    seed: int,
    *,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
    lr: float = 1e-3,
    l2: float = 1e-4,
    batch: int = 8192,
    max_epochs: int = 200,
    patience: int = 15,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> dict:
    """Honest, modestly-regularized MLP hyperparameters. ``dropout`` + light decoupled ``l2`` +
    early stopping carry regularization; ``batch`` bounds memory on million-row fits."""
    return {"seed": int(seed), "hidden": tuple(int(h) for h in hidden), "dropout": float(dropout),
            "lr": float(lr), "l2": float(l2), "batch": int(batch), "max_epochs": int(max_epochs),
            "patience": int(patience), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)}


class MlpModel:
    """A fitted MLP: standardization stats + dense weights. ``predict_proba`` is dropout-free."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], mean: np.ndarray,
                 std: np.ndarray, params: dict):
        self.weights = weights          # [(W, b), …] per dense layer, input→…→output
        self.mean = mean
        self.std = std
        self.params = params

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xs = logreg.standardize_apply(x, self.mean, self.std)
        return _forward(xs, self.weights)[0].ravel()


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _forward(x: np.ndarray, weights: list[tuple[np.ndarray, np.ndarray]]):
    """Dropout-free forward pass. Returns ``(prob, pre_activations, activations)`` — the caches are
    only used by the training backward pass (ignored at inference)."""
    a = x
    pre, act = [], [x]
    last = len(weights) - 1
    for i, (w, b) in enumerate(weights):
        z = a @ w + b
        pre.append(z)
        a = logreg.sigmoid(z) if i == last else _relu(z)
        act.append(a)
    return act[-1], pre, act


def _he_init(rng: np.random.Generator, dims: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """He-normal weights ``N(0, sqrt(2/fan_in))``, zero biases — drawn from the single seeded rng."""
    w = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        scale = np.sqrt(2.0 / fan_in)
        w.append((rng.normal(0.0, scale, (fan_in, fan_out)), np.zeros(fan_out)))
    return w


def _logloss(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(prob.ravel(), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit(x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray | None, y_val: np.ndarray | None,
        *, params: dict, seed: int) -> MlpModel:
    """Train the MLP with mini-batch Adam(W) + early stopping on val log-loss (restore best). A
    single ``rng(seed)`` drives init, per-epoch shuffles, and dropout masks — reproducible."""
    rng = np.random.default_rng(seed)
    mean, std = logreg.standardize_fit(x_tr)          # NaN-aware, reused from logreg
    xtr = logreg.standardize_apply(x_tr, mean, std)   # NaN→0 (standardized mean)
    ytr = np.asarray(y_tr, dtype=np.float64).ravel()
    n, d = xtr.shape
    dims = [d, *params["hidden"], 1]
    weights = _he_init(rng, dims)

    have_val = x_val is not None and y_val is not None and len(y_val) > 0
    if have_val:
        xva = logreg.standardize_apply(x_val, mean, std)
        yva = np.asarray(y_val, dtype=np.float64).ravel()

    # Adam moment buffers, one pair per (W, b).
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in weights]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in weights]
    lr, l2 = params["lr"], params["l2"]
    b1, b2, eps = params["beta1"], params["beta2"], params["eps"]
    p_drop, batch = params["dropout"], params["batch"]
    last = len(weights) - 1
    t = 0
    best_loss, best_weights, since_improve = np.inf, None, 0

    for _epoch in range(params["max_epochs"]):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, yb = xtr[idx], ytr[idx]
            nb = len(idx)
            # ---- forward with inverted dropout on hidden layers ----
            a = xb
            act = [xb]
            pre = []
            masks: list[np.ndarray | None] = []
            for i, (w, bvec) in enumerate(weights):
                z = a @ w + bvec
                pre.append(z)
                if i == last:
                    a = logreg.sigmoid(z)
                    masks.append(None)
                else:
                    a = _relu(z)
                    if p_drop > 0.0:
                        mask = (rng.random(a.shape) >= p_drop) / (1.0 - p_drop)
                        a = a * mask
                        masks.append(mask)
                    else:
                        masks.append(None)
                act.append(a)
            # ---- backward (mean BCE) ----
            grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(weights)  # type: ignore
            delta = (act[-1].ravel() - yb).reshape(-1, 1) / nb  # d(mean BCE)/dz at the sigmoid output
            for i in range(last, -1, -1):
                w, _b = weights[i]
                a_prev = act[i]
                gw = a_prev.T @ delta   # decoupled L2 (AdamW) is applied in the update, not the grad
                gb = delta.sum(axis=0)
                grads[i] = (gw, gb)
                if i > 0:
                    da = delta @ w.T
                    if masks[i - 1] is not None:
                        da = da * masks[i - 1]         # dropout backprop (same mask)
                    delta = da * (pre[i - 1] > 0.0)    # ReLU'
            # ---- Adam update ----
            t += 1
            for i in range(len(weights)):
                w, bvec = weights[i]
                gw, gb = grads[i]
                mw, mb = m[i]
                vw, vb = v[i]
                mw = b1 * mw + (1 - b1) * gw
                mb = b1 * mb + (1 - b1) * gb
                vw = b2 * vw + (1 - b2) * (gw * gw)
                vb = b2 * vb + (1 - b2) * (gb * gb)
                m[i], v[i] = (mw, mb), (vw, vb)
                mhat_w = mw / (1 - b1 ** t)
                mhat_b = mb / (1 - b1 ** t)
                vhat_w = vw / (1 - b2 ** t)
                vhat_b = vb / (1 - b2 ** t)
                # AdamW: adaptive step + decoupled weight decay on weights (not biases).
                new_w = w - lr * mhat_w / (np.sqrt(vhat_w) + eps) - lr * l2 * w
                new_b = bvec - lr * mhat_b / (np.sqrt(vhat_b) + eps)
                weights[i] = (new_w, new_b)
        # ---- early stopping on val log-loss (dropout off) ----
        if have_val:
            vloss = _logloss(_forward(xva, weights)[0], yva)
            if vloss < best_loss - 1e-9:
                best_loss = vloss
                best_weights = [(w.copy(), b.copy()) for w, b in weights]
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= params["patience"]:
                    break

    if have_val and best_weights is not None:
        weights = best_weights
    return MlpModel(weights, mean, std, params)


def predict_proba(model: MlpModel, x: np.ndarray, *, num_threads: int = 1) -> np.ndarray:
    """P(Up) for each row (dropout off). ``num_threads`` is accepted for a uniform backend signature;
    numpy/BLAS threading is controlled by the environment, not here."""
    return model.predict_proba(x)


def feature_importance(model: MlpModel, names: list[str]) -> None:
    """MLPs have no native gain/split importance — return ``None`` (the bake-off omits top-features)."""
    return None


def save(model: MlpModel, path) -> None:
    """Persist to a ``.npz``: per-layer weights + standardization stats + a JSON params blob."""
    arrays: dict[str, np.ndarray] = {"mean": model.mean, "std": model.std,
                                     "n_layers": np.array(len(model.weights)),
                                     "params": np.array(json.dumps(model.params))}
    for i, (w, b) in enumerate(model.weights):
        arrays[f"W{i}"] = w
        arrays[f"b{i}"] = b
    np.savez(str(path), **arrays)


def load(path) -> MlpModel:
    """Reload an :class:`MlpModel` saved by :func:`save`."""
    p = str(path)
    if not p.endswith(".npz"):
        p = p + ".npz"
    z = np.load(p, allow_pickle=False)
    n_layers = int(z["n_layers"])
    weights = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n_layers)]
    params = json.loads(str(z["params"]))
    params["hidden"] = tuple(params["hidden"])
    return MlpModel(weights, z["mean"], z["std"], params)
