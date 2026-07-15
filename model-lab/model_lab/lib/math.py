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
- :func:`taker_fee` — the exact venue taker fee (``crates/core-types::money``):
  ``shares·rate·p·(1−p)``, 5-dp half-away rounding, ``$0.00001`` floor.
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


def taker_fee(shares: float, fee_rate: float, price: float) -> float:
    """The exact venue taker fee, an offline replica of Rust
    ``crates/core-types/src/money.rs::taker_fee``.

    ``raw = shares · fee_rate · price · (1 − price)``. If ``raw`` is zero (any of
    shares/rate/price is 0, or ``price == 1``) the fee is ``0.0`` with **no** floor
    (the Rust ``raw.is_zero()`` early return). Otherwise ``raw`` is rounded to 5
    decimal places **half-away-from-zero** — fees are non-negative, so
    ``floor(raw·1e5 + 0.5)/1e5`` gives that rounding — then floored to the
    ``$0.00001`` minimum charge. ``price`` is a probability in ``(0, 1)``; callers
    guard the domain. Fractional ``shares`` are allowed (venue sizes are decimals).
    Makers pay zero — apply this only to taker fills.
    """
    raw = float(shares) * float(fee_rate) * float(price) * (1.0 - float(price))
    if raw == 0.0:
        return 0.0
    rounded = math.floor(raw * 1e5 + 0.5) / 1e5
    return max(rounded, 1e-5)


# --- Taker Fee Rebate Program -------------------------------------------------
# docs.polymarket.com/trading/taker-rebates. Tier is set by 30-day *rolling*
# WEIGHTED VOLUME  wV = size·(1−entry)·category_weight·bonus  (crypto weight 2.3,
# no crypto bonus). The rebate paid is  tier% × the TAKER FEES paid over the period
# — a fee *refund*, NOT a percentage of notional (3–50% of notional is impossible;
# the parallel maker program is likewise "% of taker fees"). Paid daily 00:00 UTC
# in pUSD, $1 minimum accrual, tier recalculated daily and applied going forward.
#
# Mirror of the Rust consts in ``crates/venue-paper/src/engine.rs`` — keep in sync.
CRYPTO_WV_WEIGHT = 2.3

#: (30-day wV threshold, tier name, rebate fraction), ascending. Below $2k = None.
TAKER_TIERS: list[tuple[float, str, float]] = [
    (0.0, "None", 0.0),
    (2_000.0, "Bronze", 0.03),
    (20_000.0, "Silver", 0.08),
    (200_000.0, "Gold", 0.18),
    (1_000_000.0, "Platinum", 0.32),
    (4_000_000.0, "Diamond", 0.44),
    (10_000_000.0, "Obsidian", 0.50),
]


def weighted_volume(shares: float, price: float, weight: float = CRYPTO_WV_WEIGHT) -> float:
    """One trade's contribution to 30-day weighted volume: ``size·(1−entry)·weight``.

    ``price`` is the entry (fill) price in ``(0, 1)``; ``weight`` defaults to the
    crypto category weight (2.3). Used only to determine the rebate *tier*, never
    the rebate amount (which is tier% of taker fees).
    """
    return float(shares) * (1.0 - float(price)) * float(weight)


def taker_rebate_tier(wv_30d: float) -> tuple[str, float]:
    """Maps a 30-day weighted volume to ``(tier_name, rebate_fraction)``.

    Returns the highest tier whose threshold ``wv_30d`` meets or exceeds, e.g.
    ``$0 → ("None", 0.0)``, ``$2,000 → ("Bronze", 0.03)``, ``$10M →
    ("Obsidian", 0.50)``.
    """
    name, pct = TAKER_TIERS[0][1], TAKER_TIERS[0][2]
    for threshold, tname, tpct in TAKER_TIERS:
        if wv_30d >= threshold:
            name, pct = tname, tpct
        else:
            break
    return name, pct


def taker_rebate_next_tier(wv_30d: float) -> tuple[str | None, float]:
    """The next tier up and the additional 30-day wV needed to reach it.

    Returns ``(next_tier_name, wv_needed)``; ``(None, 0.0)`` when already at the top
    tier (Obsidian).
    """
    for threshold, tname, _pct in TAKER_TIERS:
        if wv_30d < threshold:
            return tname, threshold - wv_30d
    return None, 0.0


def majority_baseline(y_true: np.ndarray) -> dict:
    """The always-predict-the-common-side (base-rate) no-skill baseline.

    Returns ``{"n", "p", "diracc", "brier"}`` where ``p = mean(y_true)`` is the Up
    rate, ``diracc = max(p, 1 − p)`` (accuracy of always calling the more common
    side), and ``brier = p·(1 − p)`` (the Brier of the constant base-rate forecast
    — the standard no-information reference, matching the ``chance_brier`` /
    ``base_rate_acc`` convention already used in the shuffled-label controls). A
    model's *skill* is its lift over this. NaNs on empty input.
    """
    y = np.asarray(y_true, dtype=float)
    y = y[np.isfinite(y)]
    n = int(len(y))
    if n == 0:
        return {"n": 0, "p": float("nan"), "diracc": float("nan"), "brier": float("nan")}
    p = float(np.mean(y))
    return {"n": n, "p": p, "diracc": max(p, 1.0 - p), "brier": p * (1.0 - p)}


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


def directional_accuracy(prob: np.ndarray, outcome: np.ndarray, *, tie_is_up: bool = True) -> float:
    """Fraction of predictions that call the right side of a binary outcome.

    A probability forecast "calls Up" when ``prob > 0.5`` (and, when ``tie_is_up``,
    also at exactly ``0.5`` — matching the engine's ≥-ties-Up resolution rule). The
    call is correct when it agrees with ``outcome`` (1 = Up). Returns the mean
    correct rate in ``[0, 1]``; ``nan`` on empty input. ``nan`` predictions or
    outcomes are excluded. Unlike Brier/log-loss this scores only the *side*, not
    the confidence — a useful, decision-shaped complement.
    """
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    mask = np.isfinite(prob) & np.isfinite(outcome)
    prob = prob[mask]
    outcome = outcome[mask]
    if len(prob) == 0:
        return float("nan")
    calls_up = prob >= 0.5 if tie_is_up else prob > 0.5
    correct = calls_up == (outcome >= 0.5)
    return float(np.mean(correct))


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
