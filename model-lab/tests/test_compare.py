"""Tests for the one-command model comparison (``compare`` stage).

Runs the fixture pipeline, trains both challengers (logistic + GBT), and asserts
the comparison lines up logistic / GBT-plain / GBT-residual / formula / market on a
common set of rows, with the expected output schema and a deterministic verdict.
Skipped without the opt-in ``gbt`` extra (it needs ``learn_gbt``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from model_lab.compare import PREDICTORS, _CSV_COLS, compare  # noqa: E402
from model_lab.config import ParquetNotReady, Paths  # noqa: E402


def _pipeline(tmp_path) -> Paths:
    from model_lab.dataset import dataset
    from model_lab.feature_set import feature_set
    from model_lab.features import features
    from model_lab.fixtures import make_aggtrades_fixture, make_fixture
    from model_lab.ingest import ingest
    from model_lab.labels import labels
    from model_lab.learn import learn
    from model_lab.learn_gbt import learn_gbt

    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=30)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    paths = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                  out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades")
    ingest(paths); features(paths); labels(paths); dataset(paths); feature_set(paths)
    learn(paths, days=0, run_harness=False, run_shuffle=False)
    learn_gbt(paths, days=0, run_harness=False, run_shuffle=False,
              max_boost_round=80, early_stopping_rounds=15, min_child_samples=10)
    return paths


def test_compare_lines_up_all_models(tmp_path):
    paths = _pipeline(tmp_path)
    m = compare(paths)

    c = m["counts"]
    assert c["scored_rows"] > 0 and c["n_windows"] > 0
    # The fixture grids are produced with the same config, so they align 1:1.
    assert c["n_common"] == c["n_gbt_plain"] == c["n_logistic"]

    assert m["predictors"] == PREDICTORS
    for pred in ("logistic", "gbt_plain", "gbt_residual", "formula"):
        s = m["overall"][pred]
        assert s["n"] > 0 and 0.0 <= s["brier"] <= 1.0
    assert isinstance(m["verdict"], str) and m["verdict"]

    out_dir = paths.out_dir / "compare"
    for fname in ("comparison.csv", "metrics.json", "report.html"):
        assert (out_dir / fname).exists()
    comp = pd.read_csv(out_dir / "comparison.csv")
    assert list(comp.columns) == _CSV_COLS
    # Every predictor appears, and both period + tau-bucket scopes are present.
    assert set(comp["predictor"].dropna().unique()) >= set(PREDICTORS)
    assert set(comp["scope"].unique()) == {"period", "tau_bucket"}

    # metrics.json round-trips and carries the by-period rollups.
    disk = json.loads((out_dir / "metrics.json").read_text())
    assert "day" in disk["by_period"] and "week" in disk["by_period"]


def test_compare_missing_inputs_are_rejected(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    paths = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", out_dir=out)
    with pytest.raises(ParquetNotReady):
        compare(paths)
