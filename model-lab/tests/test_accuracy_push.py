"""Unit tests for accuracy_push — the per-market 15 s model producer's metrics + fit path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_lab import accuracy_push as ap
from model_lab.lib import math as lm


def test_abstention_curve_monotone_coverage():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (p >= 0.5).astype(float)  # perfectly calibrated → accuracy 1.0 everywhere
    curve = ap.abstention_curve(p, y, [0.0, 0.1, 0.2, 0.3])
    covs = [c["coverage"] for c in curve]
    assert covs == sorted(covs, reverse=True)  # coverage falls as theta rises
    assert all(abs(c["accuracy"] - 1.0) < 1e-9 for c in curve if c["coverage"])


def test_theta_at_accuracy_finds_gate():
    # confident predictions (|p-0.5| large) are accurate; near-0.5 ones are coin flips.
    n = 20000
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, n)
    conf = np.abs(p - 0.5)
    # accurate iff confident (>0.3) else random.
    y = np.where(conf > 0.3, (p >= 0.5).astype(float),
                 (rng.uniform(0, 1, n) >= 0.5).astype(float))
    out = ap.theta_at_accuracy(p, y, target=0.85)
    assert out["theta_85"] is not None
    assert out["theta_85"]["accuracy"] >= 0.85
    assert out["theta_85"]["theta"] > 0.0  # a positive confidence gate is required for 85%
    # the smallest gate reaching 85% is the reported one (nothing lower qualifies).
    below = [c for c in ap.abstention_curve(p, y, ap.FRONTIER_THETAS)
             if c["theta"] < out["theta_85"]["theta"] and c["coverage"] >= ap.MIN_SIGNALS]
    assert all(c["accuracy"] < 0.85 for c in below)


def test_theta_at_accuracy_none_when_unreachable():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.3, 0.7, 5000)          # never very confident
    y = (rng.uniform(0, 1, 5000) >= 0.5).astype(float)  # pure noise
    out = ap.theta_at_accuracy(p, y, target=0.85)
    assert out["theta_85"] is None
    assert out["best_gate"]["accuracy"] < 0.85


def test_fit_predict_learns_a_clean_signal():
    lgb = pytest.importorskip("lightgbm")  # noqa: F841 - opt-in extra
    n = 8000
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, n)
    label = (z > 0).astype(np.float64)  # the clean signal
    data = {f: rng.normal(0, 1, n).astype(np.float32) for f in ap.PRICE_FEATURES}
    data["z"] = z.astype(np.float32)
    data[ap.LABEL] = label
    data["sample_ts_ms"] = np.arange(n, dtype=np.int64) * 5000
    sub = pd.DataFrame(data)
    train_m = np.zeros(n, bool); train_m[:6000] = True
    pred_m = np.zeros(n, bool); pred_m[6000:] = True
    prob, best, info = ap._fit_predict(sub, ap.PRICE_FEATURES, train_m, pred_m, seed=3, threads=2)
    y_pred = label[6000:]
    assert lm.directional_accuracy(prob, y_pred) > 0.9  # recovers z>0
    assert best > 0 and info["top_features"][0][0] == "z"  # z is the dominant feature


def test_regime_floor_ms():
    from datetime import date
    ms = ap._regime_floor_ms(date(2026, 6, 5))
    assert ms % ap.MS_PER_DAY == 0  # midnight UTC


def test_build_split_masks_matches_inline_computation():
    # regression: the extracted split helper must reproduce run_market's original inline masks.
    rng = np.random.default_rng(7)
    floor = ap._regime_floor_ms(__import__("datetime").date(2026, 6, 5))
    win_open = np.sort(rng.integers(floor - 60 * ap.MS_PER_DAY, floor + 10 * ap.MS_PER_DAY, 4000))
    ts = win_open + rng.integers(0, 300_000, 4000)  # a sample somewhere inside the window
    m = ap.build_split_masks(win_open, ts, floor)

    # the exact arithmetic run_market used before the extraction.
    label_info = ts + ap.HORIZON_SECS * 1000
    val_start = floor - ap.VAL_DAYS * ap.MS_PER_DAY
    pre_m = win_open < floor
    assert np.array_equal(m["regime_m"], win_open >= floor)
    assert np.array_equal(m["pre_m"], pre_m)
    assert np.array_equal(m["val_m"], pre_m & (win_open >= val_start))
    assert np.array_equal(m["sel_train_m"], pre_m & (win_open < val_start) & (label_info < val_start))
    assert np.array_equal(m["dep_train_m"], pre_m & (label_info < floor))
    # masks are disjoint where they must be: a sel_train row is never a val row or a regime row.
    assert not (m["sel_train_m"] & m["val_m"]).any()
    assert not (m["sel_train_m"] & m["regime_m"]).any()
    assert not (m["dep_train_m"] & m["regime_m"]).any()


def test_capped_train_idx_and_inner_val_split_match_inline():
    # regression: extracted train-selection helpers reproduce the original inline logic byte-for-byte.
    n = 5000
    rng = np.random.default_rng(11)
    train_m = rng.uniform(0, 1, n) > 0.3
    y = np.where(rng.uniform(0, 1, n) > 0.05, rng.integers(0, 2, n).astype(float), np.nan)
    ts = np.sort(rng.integers(0, 10_000_000, n)).astype(np.int64)
    seed = 11

    got = ap.capped_train_idx(train_m, y, ts, seed)
    # original inline computation (TRAIN_CAP not hit here, so the rng is unused — as before).
    exp = np.flatnonzero(train_m & np.isfinite(y))
    _rng = np.random.default_rng(seed)
    if len(exp) > ap.TRAIN_CAP:
        exp = np.sort(_rng.choice(exp, size=ap.TRAIN_CAP, replace=False))
    exp = exp[np.argsort(ts[exp], kind="stable")]
    assert np.array_equal(got, exp)

    itr, iva = ap.inner_val_split(got)
    n_val = max(1000, int(0.15 * len(got)))
    assert np.array_equal(itr, got[:-n_val]) and np.array_equal(iva, got[-n_val:])
    assert len(itr) + len(iva) == len(got)  # a clean partition
