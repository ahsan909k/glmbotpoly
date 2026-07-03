"""Offline reproductions of the engine's fair-value + adverse-selection math.

These mirror the Rust ``crates/model`` and ``crates/analytics`` so the lab can
validate the journaled model outputs without re-deriving the specification:

- :func:`norm_cdf` — Φ(z), the standard-normal CDF (``crates/model::math``,
  Cody CALERF ``erfc`` — reproduced here via ``math.erfc``).
- :func:`fair_p_up` — ``p_up = Φ(ln(S/K) / (σ_1s·√τ))`` (``crates/model::fair``),
  with the τ→0 ≥-ties-Up saturation.
- :func:`sigma_1s_from_bars` — the gap-aware, bias-corrected EWMA of one-second
  log returns (``crates/model::vol``): ``var ← λ^k·var + (1−λ^k)·(r²/k)``,
  ``λ = 2^(−1/half_life)``, warmup + floor/cap clamp.
- :func:`markout` — signed fair-value move after a fill (``crates/analytics``).
"""

from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_erfc = np.vectorize(math.erfc, otypes=[float])

# Engine defaults (CLAUDE.md §8 / config/default.toml [engine.defaults]).
DEFAULT_HALF_LIFE_SECS = 60.0
DEFAULT_VOL_FLOOR_1S = 2e-5
DEFAULT_VOL_CAP_1S = 5e-3
DEFAULT_WARMUP_RETURNS = 60
DEFAULT_MAX_GAP_SECS = 30
Z_CLAMP = 39.0


def norm_cdf(z):
    """Standard-normal CDF Φ(z) = ½·erfc(−z/√2). Accepts a scalar or array."""
    z = np.asarray(z, dtype=float)
    return 0.5 * _erfc(-z / _SQRT2)


def fair_p_up(s: float, k: float, sigma_1s: float, tau_secs: float) -> float:
    """``p_up = Φ(z)``, ``z = ln(S/K)/(σ_1s·√τ)``, clamped to ±39.

    At ``τ ≤ 0`` (or a non-positive vol) it saturates to the resolution rule:
    Up iff ``S ≥ K`` (ties → Up).
    """
    if tau_secs <= 0.0 or sigma_1s <= 0.0 or s <= 0.0 or k <= 0.0:
        return 1.0 if s >= k else 0.0
    sigma_tau = sigma_1s * math.sqrt(tau_secs)
    if sigma_tau <= 0.0:
        return 1.0 if s >= k else 0.0
    z = math.log(s / k) / sigma_tau
    z = max(-Z_CLAMP, min(Z_CLAMP, z))
    return float(norm_cdf(z))


def sigma_1s_from_bars(
    bar_secs: np.ndarray,
    bar_prices: np.ndarray,
    half_life_secs: float = DEFAULT_HALF_LIFE_SECS,
    floor: float = DEFAULT_VOL_FLOOR_1S,
    cap: float = DEFAULT_VOL_CAP_1S,
    warmup: int = DEFAULT_WARMUP_RETURNS,
    max_gap_secs: int = DEFAULT_MAX_GAP_SECS,
) -> np.ndarray:
    """The engine's σ_1s EWMA over one-second bars.

    ``bar_secs`` / ``bar_prices`` are aligned arrays of integer second-buckets
    (strictly increasing) and the closing price of each bar. Returns a
    same-length array of σ_1s, ``nan`` before ``warmup`` folded returns and while
    a gap re-anchors the estimate.
    """
    n = len(bar_secs)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    lam_per_s = 2.0 ** (-1.0 / half_life_secs)
    acc = 0.0
    wsum = 0.0
    count = 0
    prev_sec = int(bar_secs[0])
    prev_price = float(bar_prices[0])
    for i in range(1, n):
        sec = int(bar_secs[i])
        price = float(bar_prices[i])
        k = sec - prev_sec
        if k <= 0 or prev_price <= 0.0 or price <= 0.0:
            prev_sec, prev_price = sec, price
            continue
        if k > max_gap_secs:
            # Re-anchor without folding (freeze the estimate across the gap).
            prev_sec, prev_price = sec, price
            continue
        r = math.log(price / prev_price)
        decay = lam_per_s**k
        acc = decay * acc + (1.0 - decay) * (r * r / k)
        wsum = decay * wsum + (1.0 - decay)
        count += 1
        if wsum > 0.0 and count >= warmup:
            sigma = math.sqrt(acc / wsum)
            out[i] = min(max(sigma, floor), cap)
        prev_sec, prev_price = sec, price
    return out


def one_second_bars(ts_ms: np.ndarray, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapses raw ticks into one-second bars keyed by ``floor(ts/1000)``.

    The last tick in each second closes the bar (the engine's rule). Inputs must
    be time-ordered; returns ``(bar_secs, bar_prices)``.
    """
    if len(ts_ms) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    secs = (np.asarray(ts_ms, dtype=np.int64) // 1000).astype(np.int64)
    prices = np.asarray(prices, dtype=float)
    # Keep the last sample for each second: find where the second changes next.
    keep = np.ones(len(secs), dtype=bool)
    keep[:-1] = secs[1:] != secs[:-1]
    return secs[keep], prices[keep]


def markout(fair_before: float, fair_after: float, position_sign: int) -> float:
    """Signed fair-value move a horizon after a fill (``crates/analytics``).

    ``position_sign`` is ``+1`` for a BUY, ``−1`` for a SELL; ``fair`` is the
    outcome-oriented fair (``p_up`` for an Up position, ``1 − p_up`` for Down).
    A persistently negative markout on passive fills = adverse selection.
    """
    return position_sign * (fair_after - fair_before)


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of predicted probabilities against 0/1 outcomes."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(prob) == 0:
        return float("nan")
    return float(np.mean((prob - outcome) ** 2))


def reliability_curve(
    prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bins predictions and returns ``(bin_mid, mean_pred, empirical_rate)``.

    A well-calibrated model has ``empirical_rate ≈ mean_pred`` (the diagonal).
    Empty bins are dropped.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids, preds, rates = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Last bin is closed on the right so p == 1.0 lands somewhere.
        sel = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if not np.any(sel):
            continue
        mids.append((lo + hi) / 2.0)
        preds.append(float(np.mean(prob[sel])))
        rates.append(float(np.mean(outcome[sel])))
    return np.array(mids), np.array(preds), np.array(rates)


def log_loss(prob: np.ndarray, outcome: np.ndarray, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy of predicted probabilities vs 0/1 outcomes.

    ``−mean[ y·ln(p) + (1−y)·ln(1−p) ]``. ``p`` is clipped to ``[eps, 1−eps]`` so
    a confident-but-wrong prediction (``p == 0`` or ``1``) is a large-but-finite
    penalty rather than ``inf`` — the engine's ``p_up`` saturates to 0/1 as
    ``τ → 0``. Lower is better; ``nan`` on empty input.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(prob) == 0:
        return float("nan")
    p = np.clip(prob, eps, 1.0 - eps)
    return float(np.mean(-(outcome * np.log(p) + (1.0 - outcome) * np.log(1.0 - p))))


def reliability_table(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability bins with sample counts — like :func:`reliability_curve` but
    each bin also carries ``n`` (how many predictions fell in it).

    Returns one dict per **non-empty** bin, in ascending probability order:
    ``{bin_lo, bin_hi, bin_mid, n, mean_pred, empirical_rate}``. Bins tile
    ``[0, 1]``; the last is closed on the right so ``p == 1.0`` is included.
    ``nan`` predictions fall in no bin and are excluded.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Last bin is closed on the right so p == 1.0 lands somewhere.
        sel = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        n = int(np.count_nonzero(sel))
        if n == 0:
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "bin_mid": float((lo + hi) / 2.0),
                "n": n,
                "mean_pred": float(np.mean(prob[sel])),
                "empirical_rate": float(np.mean(outcome[sel])),
            }
        )
    return rows


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann–Whitney U) — no sklearn dependency.

    ``label`` is 0/1; returns 0.5 for a non-discriminating score, ``nan`` if a
    class is absent.
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    pos = label == 1
    neg = label == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # Average ranks within ties for a correct Mann–Whitney statistic.
    _assign_tied_ranks(score, ranks)
    sum_ranks_pos = float(np.sum(ranks[pos]))
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _assign_tied_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for m in range(i, j + 1):
                ranks[order[m]] = avg
        i = j + 1
