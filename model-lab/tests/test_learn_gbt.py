"""Tests for the GBT challenger (``learn_gbt`` stage) and its LightGBM wrapper
(``lib/gbt.py``).

Mirrors ``test_learn.py``: the model beats chance on a genuinely-predictive
synthetic set (both the plain and formula-residual variants), the shuffled-label
control collapses, training is deterministic (byte-identical booster +
predictions), the prediction grid matches the harness contract, the saved booster
round-trips, a truncated input is rejected, and the GBT-specific analyses (feature
importances, vol regime, residual) are present and internally consistent.

The whole module is skipped when LightGBM (the opt-in ``gbt`` extra) isn't
installed, so the core test suite stays lightgbm-free.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")  # opt-in extra — skip the whole module without it.

from model_lab.config import ParquetNotReady, Paths  # noqa: E402
from model_lab.feature_set import FEATURE_NAMES  # noqa: E402
from model_lab.learn_gbt import learn_gbt  # noqa: E402
from model_lab.lib import gbt  # noqa: E402

MS_PER_DAY = 86_400_000
_BASE_DAY = 20_000

# Small, fast GBT config for the tests (the fixtures are tiny).
_HP = dict(max_boost_round=150, early_stopping_rounds=15, min_child_samples=20)


def _synth_feature_set(path, *, n_days=21, windows_per_day=8, samples_per_window=10,
                       seed=0, signal=True):
    """A synthetic ``feature_set.parquet`` with the columns ``learn_gbt`` reads; the
    ``z`` feature carries the label so a model can learn it and a shuffle collapses."""
    rng = np.random.default_rng(seed)
    z_idx = FEATURE_NAMES.index("z")
    rows = []
    for d in range(n_days):
        day_ms = (_BASE_DAY + d) * MS_PER_DAY
        for w in range(windows_per_day):
            open_ms = day_ms + w * 600_000
            close_ms = open_ms + 300_000
            zc = rng.normal(0.0, 1.0)
            outcome = int((zc + rng.normal(0.0, 0.1)) > 0.0) if signal else int(rng.uniform() < 0.5)
            for s in range(samples_per_window):
                feats = rng.normal(size=len(FEATURE_NAMES))
                if signal:
                    feats[z_idx] = zc + rng.normal(0.0, 0.3)
                fwd = int((feats[z_idx] + rng.normal(0.0, 0.5)) > 0.0) if signal else int(rng.uniform() < 0.5)
                row = {
                    "series": "BTC-5m", "window_open_ms": open_ms, "window_close_ms": close_ms,
                    "sample_ts_ms": open_ms + s * 30_000, "label_source": "chainlink",
                    "in_live_coverage": False, "fwd_up_30s": fwd, "outcome_up": outcome,
                }
                row.update({name: float(feats[i]) for i, name in enumerate(FEATURE_NAMES)})
                rows.append(row)
    df = pd.DataFrame(rows)
    for k in ("window_open_ms", "window_close_ms", "sample_ts_ms"):
        df[k] = df[k].astype("int64")
    df["fwd_up_30s"] = df["fwd_up_30s"].astype("Int8")
    df["outcome_up"] = df["outcome_up"].astype("int8")
    df.to_parquet(path, engine="pyarrow", index=False)
    return df


def _paths(tmp_path, out_name="out"):
    out = tmp_path / out_name
    out.mkdir(parents=True, exist_ok=True)
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", out_dir=out)


# ---------------------------------------------------------------------------
# lib/gbt.py — the wrapper
# ---------------------------------------------------------------------------
def test_gbt_wrapper_learns_and_residual_adds_back_init_score():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(float)
    Xtr, ytr, Xv, yv, Xte, yte = X[:2000], y[:2000], X[2000:2500], y[2000:2500], X[2500:], y[2500:]
    p = gbt.default_params(7)
    b, best = gbt.fit(Xtr, ytr, Xv, yv, params=p, max_boost_round=200, early_stopping_rounds=20)
    assert best > 0
    prob = gbt.predict_proba(b, Xte, num_iteration=best)
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    assert float(((prob >= 0.5) == (yte >= 0.5)).mean()) > 0.8
    # Residual variant: a zero base margin must match the plain prediction.
    zero = np.zeros(len(Xte))
    plain = gbt.predict_proba(b, Xte, num_iteration=best)
    residual = gbt.predict_proba(b, Xte, num_iteration=best, init_score=zero)
    # b was trained without init_score, so sigmoid(0 + raw) == plain prediction.
    assert np.allclose(plain, residual, atol=1e-9)
    fi = gbt.feature_importance(b, ["a", "b", "c", "d"])
    assert fi["gain"]["a"] > max(fi["gain"]["b"], fi["gain"]["c"], fi["gain"]["d"])


def test_gbt_num_iteration_zero_guard_uses_all_trees():
    # In LightGBM num_iteration=0 means "all trees"; our wrapper treats <=0 as None.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] > 0).astype(float)
    b = gbt.refit(X, y, params=gbt.default_params(1), num_boost_round=10)
    p_all = gbt.predict_proba(b, X, num_iteration=None)
    p_zero = gbt.predict_proba(b, X, num_iteration=0)
    assert np.allclose(p_all, p_zero)


# ---------------------------------------------------------------------------
# learn_gbt: real beats chance, shuffle collapses (both variants)
# ---------------------------------------------------------------------------
def test_gbt_real_beats_chance_and_shuffle_collapses(tmp_path):
    paths = _paths(tmp_path)
    _synth_feature_set(paths.table("feature_set"), seed=1)
    m = learn_gbt(paths, days=0, train_weeks=1, test_days=3, seed=7,
                  run_harness=False, run_shuffle=True, **_HP)

    for variant in ("plain", "residual"):
        out = m["targets"]["outcome"][variant]
        assert out["mode"] == "walk_forward"
        assert out["n_folds"] >= 2
        pooled = out["pooled"]
        p = pooled["pos_frac"]
        chance_brier = p * (1.0 - p)
        assert pooled["diracc"] > 0.65, variant
        assert pooled["brier"] < chance_brier - 0.03, variant

        sc = m["shuffled_control"]["outcome"][variant]
        assert sc["collapsed"] is True, variant
        assert sc["pooled"]["brier"] >= sc["chance_brier"] - 0.03
        assert pooled["brier"] < sc["pooled"]["brier"]

    # fwd30 is plain-only and reports its own-label reliability.
    assert list(m["targets"]["fwd30"].keys()) == ["plain"]
    fwd = m["targets"]["fwd30"]["plain"]
    assert fwd["n_oos"] > 0 and "own_label_reliability" in fwd


def test_gbt_determinism_same_seed_same_artifacts(tmp_path):
    df = _synth_feature_set(tmp_path / "src.parquet", seed=2)
    a, b = _paths(tmp_path, "out_a"), _paths(tmp_path, "out_b")
    for p in (a, b):
        df.to_parquet(p.table("feature_set"), engine="pyarrow", index=False)
        learn_gbt(p, days=0, train_weeks=1, test_days=3, seed=7,
                  run_harness=False, run_shuffle=True, **_HP)

    for variant in ("plain", "residual"):
        ta = (a.out_dir / "learn_gbt" / f"model_outcome_{variant}.txt").read_text()
        tb = (b.out_dir / "learn_gbt" / f"model_outcome_{variant}.txt").read_text()
        assert ta == tb, f"booster not deterministic for {variant}"
        pa = pd.read_parquet(a.out_dir / "learn_gbt" / f"predictions_outcome_{variant}.parquet")
        pb = pd.read_parquet(b.out_dir / "learn_gbt" / f"predictions_outcome_{variant}.parquet")
        pd.testing.assert_frame_equal(pa, pb)
    # The seeded shuffled control is identical too.
    sa = pd.read_parquet(a.out_dir / "learn_gbt" / "predictions_outcome_plain_shuffled.parquet")
    sb = pd.read_parquet(b.out_dir / "learn_gbt" / "predictions_outcome_plain_shuffled.parquet")
    pd.testing.assert_frame_equal(sa, sb)


def test_gbt_prediction_grid_matches_contract_and_joins_1to1(tmp_path):
    paths = _paths(tmp_path)
    src = _synth_feature_set(paths.table("feature_set"), seed=3)
    learn_gbt(paths, days=0, train_weeks=1, test_days=3, run_harness=False, run_shuffle=False, **_HP)

    keys = ["series", "window_open_ms", "sample_ts_ms"]
    for variant in ("plain", "residual"):
        grid = pd.read_parquet(paths.out_dir / "learn_gbt" / f"predictions_outcome_{variant}.parquet")
        assert list(grid.columns) == keys + ["p_up"]
        assert not grid.duplicated(subset=keys).any()
        merged = grid.merge(src[keys], on=keys, how="left", indicator=True)
        assert (merged["_merge"] == "both").all()
        assert grid["p_up"].between(0.0, 1.0).all()


def test_gbt_model_artifact_round_trips(tmp_path):
    paths = _paths(tmp_path)
    _synth_feature_set(paths.table("feature_set"), seed=4)
    learn_gbt(paths, days=0, train_weeks=1, test_days=3, run_harness=False, run_shuffle=False, **_HP)

    meta = json.loads((paths.out_dir / "learn_gbt" / "model_outcome_plain.json").read_text())
    assert meta["feature_names"] == list(FEATURE_NAMES)
    assert meta["num_boost_round"] > 0
    assert len(meta["feature_importance"]["gain"]) == len(FEATURE_NAMES)
    booster = gbt.load(paths.out_dir / "learn_gbt" / "model_outcome_plain.txt")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, len(FEATURE_NAMES)))
    p1 = gbt.predict_proba(booster, X)
    # Reloading the saved booster reproduces the prediction exactly.
    p2 = gbt.predict_proba(gbt.load(paths.out_dir / "learn_gbt" / "model_outcome_plain.txt"), X)
    assert np.allclose(p1, p2)


def test_gbt_truncated_feature_set_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    paths.table("feature_set").write_bytes(b"not a parquet footer")
    with pytest.raises(ParquetNotReady):
        learn_gbt(paths, days=0, run_harness=False, run_shuffle=False, **_HP)


# ---------------------------------------------------------------------------
# GBT-specific analyses: feature importance, vol regime, residual
# ---------------------------------------------------------------------------
def test_gbt_regime_and_residual_are_present_and_consistent(tmp_path):
    paths = _paths(tmp_path)
    _synth_feature_set(paths.table("feature_set"), seed=5)
    m = learn_gbt(paths, days=0, train_weeks=1, test_days=3, run_harness=False,
                  run_shuffle=False, **_HP)

    for variant in ("plain", "residual"):
        blk = m["targets"]["outcome"][variant]
        # z carries the signal → it should top the gain importances.
        gain = blk["feature_importance"]["gain"]
        assert max(gain, key=gain.get) == "z"

        regime = blk["regime"]
        assert regime["n"] > 0
        assert regime["vol"]["low"]["n"] > 0 and regime["vol"]["high"]["n"] > 0

        residual = blk["residual"]
        assert residual["n_total"] > 0
        assert 0.0 <= residual["frac_disagree"] <= 1.0
        # Each cell's gbt_better flag matches its own Brier comparison.
        for c in residual["cells"]:
            assert c["gbt_better"] == (c["gbt"]["brier"] < c["formula"]["brier"])

    out_dir = paths.out_dir / "learn_gbt"
    for name in ("feature_importance_outcome_plain.csv", "regime_outcome_plain.csv",
                 "residual_outcome_plain.csv", "oos_outcome_residual.parquet", "folds.csv"):
        assert (out_dir / name).exists(), name


# ---------------------------------------------------------------------------
# integration: scoring through the harness over the fixture journal + CLI main()
# ---------------------------------------------------------------------------
def test_gbt_scores_through_harness_end_to_end(tmp_path):
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

    m = learn_gbt(paths, days=0, run_harness=True, run_shuffle=True, **_HP)
    for target, variants in (("outcome", ("plain", "residual")), ("fwd30", ("plain",))):
        for variant in variants:
            blk = m["targets"][target][variant]
            assert blk["n_oos"] > 0
            assert blk["harness"]["n_scored"] > 0
            hdir = paths.out_dir / "learn_gbt" / f"harness_{target}_{variant}"
            for fname in ("metrics.json", "scores.csv", "verdict_table.csv", "reliability.csv", "report.html"):
                assert (hdir / fname).exists()

    import model_lab.learn_gbt as gbt_mod
    rc = gbt_mod.main([
        "--journal-dir", str(paths.journal_dir), "--depth-dir", str(paths.depth_dir),
        "--hist-dir", str(paths.hist_dir), "--out", str(paths.out_dir),
        "--days", "0", "--targets", "outcome", "--max-boost-round", "80",
        "--early-stopping-rounds", "15", "--min-child-samples", "10",
    ])
    assert rc == 0
