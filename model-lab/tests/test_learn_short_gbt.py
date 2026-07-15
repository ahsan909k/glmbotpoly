"""Tests for the short-horizon GBT challenger (``learn_short_gbt`` stage).

Mirrors ``test_learn_gbt.py`` on the short-horizon dataset: the model beats chance on
a genuinely-predictive synthetic set (all three feature-set variants, both source
scopes), the shuffled-label control collapses, training is deterministic
(byte-identical booster + predictions), the prediction grid matches the harness
contract, a truncated input is rejected, and the stage-specific analyses (abstention
curve, per-regime breakdown, depth/PM contribution) are present and internally
consistent. Also covers the external-flow as-of join across the 5s/15s grid mismatch
and the nullable-label filter.

Skipped entirely when LightGBM (the opt-in ``gbt`` extra) isn't installed, so the core
suite stays lightgbm-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")  # opt-in extra — skip the whole module without it.

from model_lab.config import ParquetNotReady, Paths  # noqa: E402
from model_lab.learn_short_gbt import (  # noqa: E402
    ABSTENTION_THRESHOLDS, DEPTH_FEATURES, FLOW_FEATURES, NATIVE_BASE, PM_FEATURES,
    learn_short_gbt,
)
from model_lab.lib import math as lm  # noqa: E402

MS_PER_DAY = 86_400_000
_BASE_DAY = 20_000

# Small, fast GBT config for the tests (the fixtures are tiny).
_HP = dict(max_boost_round=120, early_stopping_rounds=15, min_child_samples=15)


def _windows(n_days, windows_per_day):
    """Deterministic window layout shared by both synthetic builders: even windows are
    ``chainlink`` (microstructure covered), odd are ``binance_proxy`` (all-NaN)."""
    for d in range(n_days):
        day_ms = (_BASE_DAY + d) * MS_PER_DAY
        for w in range(windows_per_day):
            open_ms = day_ms + w * 600_000
            close_ms = open_ms + 300_000
            source = "chainlink" if (w % 2 == 0) else "binance_proxy"
            yield open_ms, close_ms, source


def _synth_short_horizon(path, *, n_days=18, windows_per_day=6, samples_per_window=10,
                         seed=0, signal=True, na_fwd15=True):
    """A synthetic ``short_horizon.parquet`` with exactly the columns the stage reads.
    The ``z`` feature carries the label (so the model learns it and a shuffle collapses);
    chainlink windows have finite depth/PM/basis, binance_proxy windows have them NaN."""
    rng = np.random.default_rng(seed)
    rows = []
    for open_ms, close_ms, source in _windows(n_days, windows_per_day):
        zc = rng.normal(0.0, 1.0)
        covered = source == "chainlink"
        for s in range(samples_per_window):
            sample_ts = open_ms + s * 5000
            zv = zc + rng.normal(0.0, 0.3)
            f10 = int((zv + rng.normal(0.0, 0.5)) > 0.0) if signal else int(rng.uniform() < 0.5)
            f15 = int((zv + rng.normal(0.0, 0.5)) > 0.0) if signal else int(rng.uniform() < 0.5)
            row = {
                "series": "BTC-5m", "window_open_ms": open_ms, "window_close_ms": close_ms,
                "sample_ts_ms": sample_ts, "label_source": source,
                "ret": float(rng.normal() * 0.001),
                "realized_vol": float(abs(rng.normal()) * 0.001 + 1e-4),
                "sigma_1s": float(abs(rng.normal()) + 0.1),
                "log_s_k": float(rng.normal() * 0.001),
                "z": float(zv), "p_up_model": float(lm.norm_cdf(zv)),
                "tau_secs": (close_ms - sample_ts) / 1000.0,
                "elapsed_secs": (sample_ts - open_ms) / 1000.0,
                "basis_bps": float(rng.normal()) if covered else np.nan,
                "basis_ewma": float(rng.normal() * 0.001) if covered else np.nan,
                "depth_feat_covered": covered, "pm_feat_covered": covered,
                "fwd_up_10s": f10,
                "fwd_up_15s": pd.NA if (na_fwd15 and s == 0) else f15,
            }
            for c in DEPTH_FEATURES:
                row[c] = float(rng.normal()) if covered else np.nan
            for c in PM_FEATURES:
                row[c] = float(rng.normal()) if covered else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    for k in ("window_open_ms", "window_close_ms", "sample_ts_ms"):
        df[k] = df[k].astype("int64")
    df["fwd_up_10s"] = df["fwd_up_10s"].astype("Int8")
    df["fwd_up_15s"] = df["fwd_up_15s"].astype("Int8")
    df["depth_feat_covered"] = df["depth_feat_covered"].astype(bool)
    df["pm_feat_covered"] = df["pm_feat_covered"].astype(bool)
    df.to_parquet(path, engine="pyarrow", index=False)
    return df


def _synth_feature_set_flow(path, *, n_days=18, windows_per_day=6, fs_grid_secs=15, seed=1):
    """A synthetic ``feature_set.parquet`` carrying only the six signed-flow columns on
    a 15s grid over the same windows — to exercise the as-of join across the grid
    mismatch (short_horizon is on a 5s grid)."""
    rng = np.random.default_rng(seed)
    rows = []
    for open_ms, _close_ms, _source in _windows(n_days, windows_per_day):
        for k in range(0, 300, fs_grid_secs):
            row = {"series": "BTC-5m", "window_open_ms": open_ms, "sample_ts_ms": open_ms + k * 1000}
            for c in FLOW_FEATURES:
                row[c] = float(rng.normal())
            rows.append(row)
    df = pd.DataFrame(rows)
    df["window_open_ms"] = df["window_open_ms"].astype("int64")
    df["sample_ts_ms"] = df["sample_ts_ms"].astype("int64")
    df.to_parquet(path, engine="pyarrow", index=False)
    return df


def _paths(tmp_path, out_name="out"):
    out = tmp_path / out_name
    out.mkdir(parents=True, exist_ok=True)
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", out_dir=out)


def _setup(paths, **kw):
    """Write both input parquets (short_horizon + feature_set flow) under the out dir."""
    sh = _synth_short_horizon(paths.table("short_horizon"), **kw)
    fs = _synth_feature_set_flow(paths.table("feature_set"),
                                 n_days=kw.get("n_days", 18), windows_per_day=kw.get("windows_per_day", 6))
    return sh, fs


# ---------------------------------------------------------------------------
# real beats chance, shuffle collapses (all variants, both scopes)
# ---------------------------------------------------------------------------
def test_short_gbt_real_beats_chance_and_shuffle_collapses(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=1)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, seed=7,
                        targets=("fwd10",), run_harness=False, run_shuffle=True, **_HP)

    for scope in ("all", "chainlink"):
        for variant in ("base", "depth", "full"):
            blk = m["scopes"][scope]["targets"]["fwd10"][variant]
            assert blk["mode"] == "walk_forward", (scope, variant)
            assert blk["n_folds"] >= 2, (scope, variant)
            pooled = blk["pooled"]
            p = pooled["pos_frac"]
            chance_brier = p * (1.0 - p)
            assert pooled["diracc"] > 0.65, (scope, variant)
            assert pooled["brier"] < chance_brier - 0.03, (scope, variant)
            assert "own_label_reliability" in blk

            sc = m["scopes"][scope]["shuffled_control"]["fwd10"][variant]
            assert sc["collapsed"] is True, (scope, variant)
            assert sc["pooled"]["brier"] >= sc["chance_brier"] - 0.03
            assert pooled["brier"] < sc["pooled"]["brier"]


def test_short_gbt_determinism_same_seed_same_artifacts(tmp_path):
    sh = _synth_short_horizon(tmp_path / "sh.parquet", seed=2)
    fs = _synth_feature_set_flow(tmp_path / "fs.parquet")
    a, b = _paths(tmp_path, "out_a"), _paths(tmp_path, "out_b")
    for p in (a, b):
        sh.to_parquet(p.table("short_horizon"), engine="pyarrow", index=False)
        fs.to_parquet(p.table("feature_set"), engine="pyarrow", index=False)
        learn_short_gbt(p, days=0, train_weeks=1, test_days=3, seed=7, targets=("fwd10",),
                        variants=("base",), scopes=("all",), run_harness=False,
                        run_shuffle=True, **_HP)

    ta = (a.out_dir / "learn_short_gbt" / "model_all_fwd10_base.txt").read_text()
    tb = (b.out_dir / "learn_short_gbt" / "model_all_fwd10_base.txt").read_text()
    assert ta == tb, "booster not deterministic"
    pa = pd.read_parquet(a.out_dir / "learn_short_gbt" / "predictions_all_fwd10_base.parquet")
    pb = pd.read_parquet(b.out_dir / "learn_short_gbt" / "predictions_all_fwd10_base.parquet")
    pd.testing.assert_frame_equal(pa, pb)
    sa = pd.read_parquet(a.out_dir / "learn_short_gbt" / "predictions_all_fwd10_base_shuffled.parquet")
    sb = pd.read_parquet(b.out_dir / "learn_short_gbt" / "predictions_all_fwd10_base_shuffled.parquet")
    pd.testing.assert_frame_equal(sa, sb)


def test_short_gbt_prediction_grid_matches_contract_and_joins_1to1(tmp_path):
    paths = _paths(tmp_path)
    src, _ = _setup(paths, seed=3)
    learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                    variants=("base",), scopes=("all",), run_harness=False, run_shuffle=False, **_HP)

    keys = ["series", "window_open_ms", "sample_ts_ms"]
    grid = pd.read_parquet(paths.out_dir / "learn_short_gbt" / "predictions_all_fwd10_base.parquet")
    assert list(grid.columns) == keys + ["p_up"]
    assert not grid.duplicated(subset=keys).any()
    merged = grid.merge(src[keys], on=keys, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert grid["p_up"].between(0.0, 1.0).all()


def test_short_gbt_truncated_short_horizon_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    paths.table("short_horizon").write_bytes(b"not a parquet footer")
    with pytest.raises(ParquetNotReady):
        learn_short_gbt(paths, days=0, run_harness=False, run_shuffle=False, **_HP)


# ---------------------------------------------------------------------------
# stage-specific analyses: abstention, regime, depth contribution
# ---------------------------------------------------------------------------
def test_short_gbt_abstention_curve_wellformed(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=4)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                        variants=("base",), scopes=("all",), run_harness=False, run_shuffle=False, **_HP)

    abst = m["scopes"]["all"]["targets"]["fwd10"]["base"]["abstention"]
    assert [r["threshold"] for r in abst] == ABSTENTION_THRESHOLDS
    cov = [r["coverage"] for r in abst]
    assert cov[0] == pytest.approx(1.0)
    for i in range(len(cov) - 1):
        assert cov[i] >= cov[i + 1] - 1e-12  # monotone non-increasing
    for r in abst:
        assert 0.0 <= r["coverage"] <= 1.0
    assert (paths.out_dir / "learn_short_gbt" / "abstention_all_fwd10_base.csv").exists()


def test_short_gbt_regime_present(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=5)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                        variants=("base",), scopes=("all",), run_harness=False, run_shuffle=False, **_HP)

    blk = m["scopes"]["all"]["targets"]["fwd10"]["base"]
    regime = blk["regime"]
    assert regime["n"] > 0
    assert regime["vol"]["low"]["n"] > 0 and regime["vol"]["high"]["n"] > 0
    assert "chainlink" in regime["source"] and "binance_proxy" in regime["source"]
    assert "reference_pooled" in blk and blk["reference_pooled"]["n"] > 0
    assert (paths.out_dir / "learn_short_gbt" / "regime_all_fwd10_base.csv").exists()


def test_short_gbt_majority_pooled_present(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=5)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                        variants=("base",), scopes=("all",), run_harness=False, run_shuffle=False, **_HP)

    blk = m["scopes"]["all"]["targets"]["fwd10"]["base"]
    mj = blk["majority_pooled"]
    assert mj["n"] > 0
    # Base-rate arithmetic: diracc = max(p,1-p), brier = p*(1-p).
    p = mj["p"]
    assert abs(mj["diracc"] - max(p, 1.0 - p)) < 1e-12
    assert abs(mj["brier"] - p * (1.0 - p)) < 1e-12
    # A genuinely-predictive model beats the always-common-side baseline.
    assert blk["pooled"]["diracc"] > mj["diracc"]


def test_short_gbt_depth_contribution_present(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=6)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                        variants=("base", "depth", "full"), scopes=("all",),
                        run_harness=False, run_shuffle=False, **_HP)

    dc = m["scopes"]["all"]["depth_contribution"]["fwd10"]
    for comp in ("base_vs_depth", "depth_vs_full", "base_vs_full"):
        assert comp in dc
        assert dc[comp]["overall"]["n"] > 0
        for key in ("brier_a", "brier_b", "brier_delta", "diracc_a", "diracc_b", "diracc_delta"):
            assert key in dc[comp]["overall"]
    # Depth features exist only on chainlink rows, which appear in the OOS.
    assert dc["base_vs_depth"]["covered"]["n"] > 0
    assert (paths.out_dir / "learn_short_gbt" / "depth_contribution.csv").exists()


def test_short_gbt_feature_sets_nested_and_distinct(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=7)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                        variants=("base", "depth", "full"), scopes=("all",),
                        run_harness=False, run_shuffle=False, **_HP)

    feats = m["features"]
    assert set(feats["base"]) < set(feats["depth"]) < set(feats["full"])
    assert feats["full"] == feats["base"] + DEPTH_FEATURES + PM_FEATURES
    base_blk = m["scopes"]["all"]["targets"]["fwd10"]["base"]
    full_blk = m["scopes"]["all"]["targets"]["fwd10"]["full"]
    assert base_blk["feature_names"] != full_blk["feature_names"]
    # The full model's importances cover the depth features.
    assert any(d in full_blk["feature_importance"]["gain"] for d in DEPTH_FEATURES)


def test_short_gbt_external_flow_join_and_optout(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=8)
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                        variants=("base",), scopes=("all",), run_harness=False, run_shuffle=False, **_HP)
    ef = m["external_flow"]
    assert ef["joined"] is True
    # Synthetic feature_set covers every window with non-NaN flow → both fractions ~1.0.
    assert ef["n_asof_matched"] > 0 and ef["frac_asof_matched"] is not None and ef["frac_asof_matched"] > 0.9
    assert ef["frac_flow_present"] is not None and ef["frac_flow_present"] > 0.9
    assert m["features"]["base"] == NATIVE_BASE + FLOW_FEATURES
    assert all(f in m["scopes"]["all"]["targets"]["fwd10"]["base"]["feature_names"] for f in FLOW_FEATURES)

    # Opt out: base is native-only and the stage still runs without feature_set.parquet.
    paths2 = _paths(tmp_path, "out2")
    _synth_short_horizon(paths2.table("short_horizon"), seed=8)
    m2 = learn_short_gbt(paths2, days=0, train_weeks=1, test_days=3, targets=("fwd10",),
                         variants=("base",), scopes=("all",), external_flow=False,
                         run_harness=False, run_shuffle=False, **_HP)
    assert m2["external_flow"]["joined"] is False
    assert m2["features"]["base"] == NATIVE_BASE
    assert m2["scopes"]["all"]["targets"]["fwd10"]["base"]["n_oos"] > 0


def test_short_gbt_nullable_fwd15_filter(tmp_path):
    paths = _paths(tmp_path)
    _setup(paths, seed=9)  # na_fwd15=True injects pd.NA on the first sample of every window
    m = learn_short_gbt(paths, days=0, train_weeks=1, test_days=3, targets=("fwd15",),
                        variants=("base",), scopes=("all",), run_harness=False, run_shuffle=False, **_HP)
    blk = m["scopes"]["all"]["targets"]["fwd15"]["base"]
    assert blk["n_oos"] > 0
    assert np.isfinite(blk["pooled"]["brier"])


def test_short_gbt_main_cli_smoke(tmp_path):
    import model_lab.learn_short_gbt as mod
    paths = _paths(tmp_path)
    _setup(paths, seed=10)
    rc = mod.main([
        "--journal-dir", str(paths.journal_dir), "--depth-dir", str(paths.depth_dir),
        "--out", str(paths.out_dir), "--days", "0", "--targets", "fwd10", "--scopes", "all",
        "--train-weeks", "1", "--test-days", "3", "--max-boost-round", "80",
        "--early-stopping-rounds", "15", "--min-child-samples", "10",
    ])
    assert rc == 0
    assert (paths.out_dir / "learn_short_gbt" / "metrics.json").exists()
