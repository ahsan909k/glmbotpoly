"""Tests for the shadow feature-parity guard + smoke report."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from model_lab import report_shadow as rs
from model_lab import shadow_parity as sp


def _pair(n=120, noise=1e-4, seed=1):
    rng = np.random.default_rng(seed)
    key = {"series": ["BTC-5m"] * n, "window_open_ms": [0] * n,
           "sample_ts_ms": list(range(0, n * 5000, 5000))}
    off = dict(key)
    live = {**key, "p_up": rng.random(n)}
    for f in sp.FULL_FEATURES:
        base = rng.normal(size=n)
        off[f] = base
        live[f] = base + rng.normal(scale=noise, size=n)
    return pd.DataFrame(live), pd.DataFrame(off)


def test_clean_features_pass():
    live, off = _pair()
    r = sp.run_parity(live, off, min_n=50)
    assert r["passed"], r["failures"]
    assert r["n_matched"] == 120


def test_scale_bug_fails():
    live, off = _pair()
    live = live.copy()
    live["sigma_1s"] = off["sigma_1s"].to_numpy() * 3.0
    r = sp.run_parity(live, off, min_n=50)
    assert not r["passed"]
    assert any("sigma_1s" in f for f in r["failures"])


def test_low_corr_price_fails():
    live, off = _pair(noise=0.0)
    live = live.copy()
    # Randomize a price feature entirely → corr collapses.
    live["z"] = np.random.default_rng(9).normal(size=len(live))
    r = sp.run_parity(live, off, min_n=50)
    assert not r["passed"]
    assert any("z:" in f for f in r["failures"])


def test_nan_rows_excluded_not_failed():
    live, off = _pair()
    live = live.copy()
    # An expected-missing depth feature (NaN on live) is skipped, not a failure.
    live.loc[: len(live) // 2, "depth_imb_20"] = np.nan
    r = sp.run_parity(live, off, min_n=50)
    assert r["passed"], r["failures"]


def test_read_shadow_predictions_and_report(tmp_path):
    d = tmp_path / "shadow"
    d.mkdir()
    recs = []
    for i in range(60):
        feats = [0.5] * 24
        feats[3] = None  # a NaN feature → JSON null
        recs.append({
            "ts": i * 5000, "series": "BTC-5m", "window_open_ms": 0,
            "p_up": 0.5 + 0.001 * i, "features": feats, "features_hash": "x",
            "model_sha256": "abc", "model_trained_through_ms": 0, "model_seed": 1,
        })
    with gzip.open(d / "shadow-20231114.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    df = sp.read_shadow_predictions(d)
    assert len(df) == 60
    assert np.isnan(df["log_s_k"].iloc[0]), "null feature → NaN"
    assert df["ret"].iloc[0] == 0.5

    summary = rs.summarize(df)
    assert summary["total"] == 60
    assert "BTC-5m" in summary["series"]
    assert summary["series"]["BTC-5m"]["predictions"] == 60
    # 23 of 24 finite (index 3 is null).
    assert summary["coverage"]["median_finite"] == 23
