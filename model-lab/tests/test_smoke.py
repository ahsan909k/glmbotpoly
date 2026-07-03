"""End-to-end smoke test: fixture → all stages → assertions.

Mirrors ``python -m model_lab.verify`` as a pytest so ``pytest -q`` exercises the
whole pipeline over the synthetic fixture.
"""

from __future__ import annotations

import math

import pandas as pd

from model_lab.calibration_audit import calibration_audit
from model_lab.config import Paths
from model_lab.fixtures import make_aggtrades_fixture, make_fixture
from model_lab.verify import run_checks


def test_pipeline_end_to_end(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=20)
    make_aggtrades_fixture(tmp_path / "aggtrades")  # run_checks exercises the proxy path
    paths = Paths(
        journal_dir=tmp_path / "journal",
        depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out",
        hist_dir=tmp_path / "aggtrades",
    )
    failures = run_checks(paths)
    assert not failures, f"pipeline checks failed: {failures}"

    # The expected parquet tables all exist.
    for name in ("ticks", "model", "fills", "settlements", "windows", "depth", "features", "labels"):
        assert paths.table(name).exists(), f"missing {name}.parquet"

    # Depth features are well-formed.
    feats = pd.read_parquet(paths.table("features"), engine="pyarrow")
    imb = feats["imbalance"].dropna()
    assert not imb.empty
    assert imb.between(-1.0, 1.0).all()


def test_calibration_audit(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=20)
    paths = Paths(
        journal_dir=tmp_path / "journal",
        depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out",
    )
    # Self-contained: the audit reads the journal directly, no prior stage needed.
    m = calibration_audit(paths)
    c = m["counts"]
    assert c["resolved_windows"] > 0
    assert c["joined_snapshots"] > 0
    assert c["market_tops"] > 0

    for fname in ("metrics.json", "scores.csv", "reliability.csv", "report.html"):
        assert (paths.out_dir / "calibration_audit" / fname).exists(), f"missing {fname}"

    overall = next(r for r in m["scope_rows"] if r["scope"] == "overall")
    assert math.isfinite(overall["model_brier"])
    assert math.isfinite(overall["model_logloss"])
    assert math.isfinite(overall["market_brier"])
    # The fixture's market mid = model p_up + noise, so the model must win.
    assert overall["model_brier"] <= overall["market_brier"]
    assert isinstance(m["verdict"], str) and m["verdict"]

    # Every τ bucket appears in the split, and scores.csv is parseable.
    scores = pd.read_csv(paths.out_dir / "calibration_audit" / "scores.csv")
    assert not scores.empty and "model_brier" in scores.columns
    tau_buckets = set(scores.loc[scores["scope"] == "tau_bucket", "tau_bucket"])
    assert {"early", "mid", "final_minute", "final_20s"} & tau_buckets


def test_ingest_handles_missing_depth(tmp_path):
    # Journal present, depth dir absent → depth table is empty, others populated.
    make_fixture(tmp_path / "journal", tmp_path / "depth_unused", n_windows=5)
    paths = Paths(
        journal_dir=tmp_path / "journal",
        depth_dir=tmp_path / "no_such_depth_dir",
        out_dir=tmp_path / "out",
    )
    from model_lab.ingest import ingest

    counts = ingest(paths)
    assert counts["ticks"] > 0
    assert counts["depth"] == 0
