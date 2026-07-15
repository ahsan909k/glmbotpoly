"""Guard test — the basis-corrected feature ``z`` must reproduce the live engine.

The models consume the offline-reconstructed ``z``; if it drifts from the engine's
live basis-corrected fair value (as it once did, silently, on the real Chainlink
slice), every downstream verdict is polluted. This test makes that divergence a
loud, permanent failure:

- ``test_basis_corrected_z_matches_journaled_p_up`` — on chainlink windows,
  Φ(basis-corrected feature z) (dataset ``p_up_model``) tracks the journaled engine
  ``p_up`` within tolerance, per time-remaining bucket.
- ``test_uncorrected_z_would_diverge_negative_control`` — the pre-fix
  (basis-*un*corrected) z visibly diverges near expiry beyond the guard tolerance,
  proving the guard is non-vacuous (a reverted basis correction is caught).

Runs on the synthetic fixture (no real data, no lightgbm), so it is part of the
core suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_lab import eval_harness as eh
from model_lab import verify
from model_lab.config import Paths
from model_lab.dataset import dataset
from model_lab.fixtures import make_aggtrades_fixture, make_fixture
from model_lab.lib import math as lm


def _build(tmp_path) -> Paths:
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=40)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    p = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
              out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades")
    dataset(p)
    return p


def _brier_gap_by_bucket(paths: Paths, grid: pd.DataFrame) -> dict[str, float]:
    """|Brier(model) − Brier(journaled p_up)| per bucket for an arbitrary grid."""
    frame, _ = eh.build_external_frame_from(eh.load_benchmarks(paths), grid)
    frame = frame.dropna(subset=["formula", "model"])
    tb = eh.tau_bucket(frame["tau"].to_numpy(dtype=float), frame["dur"].to_numpy(dtype=float))
    gaps: dict[str, float] = {}
    for bkt in eh.TAU_ORDER:
        g = frame[tb == bkt]
        if len(g) < 5:
            continue
        m = g["model"].to_numpy(dtype=float)
        f = g["formula"].to_numpy(dtype=float)
        o = g["outcome_up"].to_numpy(dtype=float)
        gaps[bkt] = abs(lm.brier_score(m, o) - lm.brier_score(f, o))
    return gaps


def test_basis_corrected_z_matches_journaled_p_up(tmp_path):
    paths = _build(tmp_path)
    rows = verify.basis_calibration_by_bucket(paths)  # scores corrected p_up_model
    assert rows, "no chainlink rows overlapped the journaled p_up"
    for r in rows:
        assert r["brier_gap"] <= verify.BASIS_GUARD_MAX_BRIER_GAP, (r["bucket"], r["brier_gap"])
        assert r["corr"] >= verify.BASIS_GUARD_MIN_CORR, (r["bucket"], r["corr"])


def test_uncorrected_z_would_diverge_negative_control(tmp_path):
    paths = _build(tmp_path)
    df = pd.read_parquet(
        paths.table("dataset"),
        columns=["series", "window_open_ms", "sample_ts_ms", "label_source",
                 "log_s_k", "sigma_1s", "tau_secs"],
    )
    ch = df[df["label_source"] == "chainlink"].copy()
    # Reconstruct the pre-fix (basis-uncorrected) z = log(mid/strike)/(σ√τ).
    sigma_tau = ch["sigma_1s"] * np.sqrt(ch["tau_secs"])
    with np.errstate(divide="ignore", invalid="ignore"):
        z_unc = (ch["log_s_k"] / sigma_tau).clip(-lm.Z_CLAMP, lm.Z_CLAMP)
    ch["p_up"] = lm.norm_cdf(z_unc.to_numpy(dtype=float))
    grid = ch[["series", "window_open_ms", "sample_ts_ms", "p_up"]].dropna()
    unc_gaps = _brier_gap_by_bucket(paths, grid)

    # The uncorrected reconstruction diverges from the live engine near expiry —
    # beyond the guard tolerance, so the guard would fire on a reverted fix.
    assert "final_20s" in unc_gaps, unc_gaps
    assert unc_gaps["final_20s"] > verify.BASIS_GUARD_MAX_BRIER_GAP, unc_gaps

    # ...while the corrected reconstruction stays inside tolerance there — the guard
    # discriminates between a working and a broken basis correction.
    corrected = {r["bucket"]: r["brier_gap"] for r in verify.basis_calibration_by_bucket(paths)}
    assert corrected.get("final_20s", 1.0) <= verify.BASIS_GUARD_MAX_BRIER_GAP, corrected
