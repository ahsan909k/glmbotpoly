"""Tests for the walk-forward logistic-regression challenger (``learn`` stage) and
its numpy core (``lib/logreg.py``).

Covers the properties that make the stage trustworthy: the model beats chance on a
genuinely-predictive synthetic set, the shuffled-label control collapses to chance
(the anti-cheat proof), training is deterministic, the walk-forward folds are
chronological with a working purge (no look-ahead), the prediction grid matches the
harness contract and joins 1:1, the saved artifact round-trips, and a truncated
input is rejected.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from model_lab.config import ParquetNotReady, Paths
from model_lab.feature_set import FEATURE_NAMES
from model_lab.learn import (
    MS_PER_DAY, TARGETS, _fold_masks, _fold_schedule, learn,
)
from model_lab.lib import logreg as lr
from model_lab.lib import math as lm

# ---------------------------------------------------------------------------
# lib/logreg.py — numeric core
# ---------------------------------------------------------------------------


def test_sigmoid_stable_and_centered():
    assert float(lr.sigmoid(0.0)) == pytest.approx(0.5)
    # No overflow at extreme magnitudes; saturates to the open unit interval.
    assert float(lr.sigmoid(1000.0)) == 1.0
    assert float(lr.sigmoid(-1000.0)) == 0.0
    x = np.linspace(-10, 10, 51)
    s = lr.sigmoid(x)
    assert np.all(np.diff(s) > 0)  # strictly monotone
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_standardize_is_nan_aware():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    mean, std = lr.standardize_fit(X)
    # Column 0: mean over finite {1,2,3} = 2; column 1 is all-NaN → mean 0, std 1.
    assert mean[0] == pytest.approx(2.0)
    assert std[1] == 1.0
    Xs = lr.standardize_apply(X, mean, std)
    assert np.all(np.isfinite(Xs))          # every NaN imputed
    assert np.allclose(Xs[:, 1], 0.0)       # all-missing feature contributes nothing


def test_fit_recovers_coefficients_and_ridge_shrinks():
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=(n, 2))
    logits = 2.5 * x[:, 0] - 1.0 * x[:, 1] + 0.3
    y = (rng.uniform(size=n) < lr.sigmoid(logits)).astype(float)
    m, s = lr.standardize_fit(x)
    xs = lr.standardize_apply(x, m, s)
    coef_lo, _ = lr.fit_logistic_ridge(xs, y, l2=0.1)
    coef_hi, _ = lr.fit_logistic_ridge(xs, y, l2=1000.0)
    # Right signs / ordering at light regularization.
    assert coef_lo[0] > 0 and coef_lo[1] < 0 and coef_lo[0] > abs(coef_lo[1])
    # Heavier ridge shrinks the coefficients toward zero.
    assert np.linalg.norm(coef_hi) < np.linalg.norm(coef_lo)


# ---------------------------------------------------------------------------
# fold schedule + purge (no look-ahead)
# ---------------------------------------------------------------------------


def test_fold_schedule_walk_forward_is_chronological_and_non_overlapping():
    folds, mode = _fold_schedule(np.arange(0, 21), train_days=7, test_days=2)
    assert mode == "walk_forward"
    prev_end = None
    for train_start, test_start, test_end in folds:
        assert train_start == test_start - 7          # trailing window
        assert train_start < test_start <= test_end    # chronological
        if prev_end is not None:
            assert test_start >= prev_end              # test blocks never overlap
        prev_end = test_end


def test_fold_schedule_fallback_and_insufficient():
    folds, mode = _fold_schedule(np.array([0, 1, 2, 3]), train_days=7, test_days=2)
    assert mode == "single_split" and len(folds) == 1
    folds2, mode2 = _fold_schedule(np.array([5]), train_days=7, test_days=2)
    assert mode2 == "insufficient" and folds2 == []


def test_fold_masks_purge_excludes_boundary_leakage():
    ms = MS_PER_DAY
    day = np.array([0, 0, 1, 1, 2, 2])
    # Row 3 (day 1) has a label that becomes known inside the purge window → must drop.
    label_info = np.array([0, 0, 2 * ms - 40_000, 2 * ms - 10_000, 2 * ms + 5_000, 2 * ms + 6_000])
    train_mask, test_mask = _fold_masks(
        day, label_info, train_start=0, test_start=2, test_end=3, purge_ms=30_000)
    assert list(test_mask) == [False, False, False, False, True, True]
    assert list(train_mask) == [True, True, True, False, False, False]  # row 3 purged


# ---------------------------------------------------------------------------
# synthetic feature_set for the end-to-end learner tests
# ---------------------------------------------------------------------------

_BASE_DAY = 20_000  # epoch-day ≈ 2024-10-03


def _write_synth_feature_set(path, *, n_days=21, windows_per_day=8, samples_per_window=10,
                             seed=0, signal=True):
    """Write a synthetic ``feature_set.parquet`` with the columns ``learn`` reads.

    When ``signal`` is set, the ``z`` feature carries the label (each window's
    outcome follows a per-window latent, and samples' ``z`` sit near it), so a
    logistic regression can learn it; the other 15 features are pure noise. Labels
    are ~balanced so a shuffled control collapses to ~0.5.
    """
    rng = np.random.default_rng(seed)
    z_idx = FEATURE_NAMES.index("z")
    rows = []
    for d in range(n_days):
        day_ms = (_BASE_DAY + d) * MS_PER_DAY
        for w in range(windows_per_day):
            open_ms = day_ms + w * 600_000
            close_ms = open_ms + 300_000
            zc = rng.normal(0.0, 1.0)  # per-window latent
            outcome = int((zc + rng.normal(0.0, 0.1)) > 0.0) if signal else int(rng.uniform() < 0.5)
            for s in range(samples_per_window):
                feats = rng.normal(size=len(FEATURE_NAMES))
                if signal:
                    feats[z_idx] = zc + rng.normal(0.0, 0.3)
                fwd = int((feats[z_idx] + rng.normal(0.0, 0.5)) > 0.0) if signal else int(rng.uniform() < 0.5)
                row = {
                    "series": "BTC-5m",
                    "window_open_ms": open_ms,
                    "window_close_ms": close_ms,
                    "sample_ts_ms": open_ms + s * 30_000,
                    "label_source": "chainlink",
                    "in_live_coverage": False,
                    "fwd_up_30s": fwd,
                    "outcome_up": outcome,
                }
                row.update({name: float(feats[i]) for i, name in enumerate(FEATURE_NAMES)})
                rows.append(row)
    df = pd.DataFrame(rows)
    df["window_open_ms"] = df["window_open_ms"].astype("int64")
    df["window_close_ms"] = df["window_close_ms"].astype("int64")
    df["sample_ts_ms"] = df["sample_ts_ms"].astype("int64")
    df["fwd_up_30s"] = df["fwd_up_30s"].astype("Int8")
    df["outcome_up"] = df["outcome_up"].astype("int8")
    df.to_parquet(path, engine="pyarrow", index=False)
    return df


def _paths(tmp_path, out_name="out"):
    out = tmp_path / out_name
    (out).mkdir(parents=True, exist_ok=True)
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", out_dir=out)


# ---------------------------------------------------------------------------
# learner: real beats chance, shuffled collapses
# ---------------------------------------------------------------------------


def test_real_model_beats_chance_and_shuffle_collapses(tmp_path):
    paths = _paths(tmp_path)
    _write_synth_feature_set(paths.table("feature_set"), seed=1)
    m = learn(paths, days=0, train_weeks=1, test_days=2, l2=1.0, seed=7,
              run_harness=False, run_shuffle=True)

    out = m["targets"]["outcome"]
    assert out["mode"] == "walk_forward"           # enough days → true walk-forward
    assert out["n_folds"] >= 2
    pooled = out["pooled"]
    p = pooled["pos_frac"]
    chance_brier = p * (1.0 - p)
    # The real model genuinely learns the signal.
    assert pooled["diracc"] > 0.65
    assert pooled["brier"] < chance_brier - 0.03

    sc = m["shuffled_control"]["outcome"]
    # Shuffled labels → collapse: no better than the base rate, real model wins.
    assert sc["collapsed"] is True
    assert sc["pooled"]["diracc"] < 0.60
    assert sc["pooled"]["brier"] >= sc["chance_brier"] - 0.03
    assert pooled["brier"] < sc["pooled"]["brier"]

    # The fwd30 target trains + reports its own-label reliability.
    fwd = m["targets"]["fwd30"]
    assert fwd["n_oos"] > 0
    assert "own_label_reliability" in fwd


def test_determinism_same_seed_same_artifacts(tmp_path):
    df = _write_synth_feature_set(tmp_path / "feature_set_src.parquet", seed=2)
    a = _paths(tmp_path, "out_a")
    b = _paths(tmp_path, "out_b")
    for p in (a, b):
        df.to_parquet(p.table("feature_set"), engine="pyarrow", index=False)
        learn(p, days=0, train_weeks=1, test_days=2, seed=7, run_harness=False, run_shuffle=True)

    ma = json.loads((a.out_dir / "learn" / "model_outcome.json").read_text())
    mb = json.loads((b.out_dir / "learn" / "model_outcome.json").read_text())
    assert ma["coef"] == mb["coef"] and ma["intercept"] == mb["intercept"]
    assert ma["standardize"] == mb["standardize"]

    pa = pd.read_parquet(a.out_dir / "learn" / "predictions_outcome.parquet")
    pb = pd.read_parquet(b.out_dir / "learn" / "predictions_outcome.parquet")
    pd.testing.assert_frame_equal(pa, pb)
    # The shuffled control is seeded too → identical.
    sa = pd.read_parquet(a.out_dir / "learn" / "predictions_outcome_shuffled.parquet")
    sb = pd.read_parquet(b.out_dir / "learn" / "predictions_outcome_shuffled.parquet")
    pd.testing.assert_frame_equal(sa, sb)


def test_prediction_grid_matches_contract_and_joins_1to1(tmp_path):
    paths = _paths(tmp_path)
    src = _write_synth_feature_set(paths.table("feature_set"), seed=3)
    learn(paths, days=0, train_weeks=1, test_days=2, run_harness=False, run_shuffle=False)

    grid = pd.read_parquet(paths.out_dir / "learn" / "predictions_outcome.parquet")
    assert list(grid.columns) == ["series", "window_open_ms", "sample_ts_ms", "p_up"]
    keys = ["series", "window_open_ms", "sample_ts_ms"]
    assert not grid.duplicated(subset=keys).any()           # unique per key
    merged = grid.merge(src[keys], on=keys, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()                # every grid key exists in the source
    assert grid["p_up"].between(0.0, 1.0).all()


def test_model_artifact_round_trips(tmp_path):
    paths = _paths(tmp_path)
    _write_synth_feature_set(paths.table("feature_set"), seed=4)
    learn(paths, days=0, train_weeks=1, test_days=2, run_harness=False, run_shuffle=False)

    model = json.loads((paths.out_dir / "learn" / "model_outcome.json").read_text())
    assert model["feature_names"] == list(FEATURE_NAMES)
    assert len(model["coef"]) == len(model["feature_names"])
    mean = np.array(model["standardize"]["mean"])
    std = np.array(model["standardize"]["std"])
    coef = np.array(model["coef"])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, len(FEATURE_NAMES)))
    expected = lr.predict_proba(lr.standardize_apply(X, mean, std), coef, model["intercept"])
    # Reconstructing the artifact's transform reproduces the probability exactly.
    direct = lr.sigmoid(lr.standardize_apply(X, mean, std) @ coef + model["intercept"])
    assert np.allclose(expected, direct)


def test_truncated_feature_set_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    paths.table("feature_set").write_bytes(b"not a parquet footer")
    with pytest.raises(ParquetNotReady):
        learn(paths, days=0, run_harness=False, run_shuffle=False)


# ---------------------------------------------------------------------------
# integration: scoring through the harness over the fixture journal
# ---------------------------------------------------------------------------


def test_learn_scores_through_harness_end_to_end(tmp_path):
    from model_lab.dataset import dataset
    from model_lab.feature_set import feature_set
    from model_lab.features import features
    from model_lab.fixtures import make_aggtrades_fixture, make_fixture
    from model_lab.ingest import ingest
    from model_lab.labels import labels

    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=30)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    paths = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                  out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades")
    ingest(paths); features(paths); labels(paths); dataset(paths); feature_set(paths)

    m = learn(paths, days=0, run_harness=True, run_shuffle=True)
    for target in TARGETS:
        blk = m["targets"][target]
        assert blk["n_oos"] > 0
        assert blk["harness"]["n_scored"] > 0       # journal windows scored vs benchmarks
        hdir = paths.out_dir / "learn" / f"harness_{target}"
        for fname in ("metrics.json", "scores.csv", "verdict_table.csv", "reliability.csv", "report.html"):
            assert (hdir / fname).exists()
        assert (paths.out_dir / "learn" / f"harness_{target}_shuffled").is_dir()

    # Exercise the CLI main() print path (incl. the harness-summary formatting).
    import model_lab.learn as learn_mod
    rc = learn_mod.main([
        "--journal-dir", str(paths.journal_dir), "--depth-dir", str(paths.depth_dir),
        "--hist-dir", str(paths.hist_dir), "--out", str(paths.out_dir),
        "--days", "0", "--targets", "outcome",
    ])
    assert rc == 0
