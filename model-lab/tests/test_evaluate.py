"""Tests for the evaluation harness (the single source of truth for scoring)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from model_lab import eval_harness as eh
from model_lab.config import ParquetNotReady, Paths
from model_lab.evaluate import evaluate
from model_lab.fixtures import make_fixture, make_predictions_fixture


def _paths(tmp_path, out_name="out"):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=20)
    return Paths(
        journal_dir=tmp_path / "journal",
        depth_dir=tmp_path / "depth",
        out_dir=tmp_path / out_name,
    )


def test_evaluate_self_baseline(tmp_path):
    paths = _paths(tmp_path)
    m = evaluate(paths, min_windows=5)  # no predictions → formula self-baseline
    c = m["counts"]
    assert c["resolved_windows"] > 0 and c["joined_snapshots"] > 0 and c["market_tops"] > 0

    overall = eh.find_scope(m["scope_rows"], "overall")
    assert overall is not None
    for k in ("model_brier", "formula_brier", "market_brier", "model_diracc", "market_diracc"):
        assert math.isfinite(overall[k]), k
    # Self-baseline identity: the model under test IS the formula model.
    assert abs(overall["model_brier"] - overall["formula_brier"]) < 1e-12
    assert m["params"]["self_baseline"] is True

    # The standardized verdict table: both benchmarks, per-period rows + stability.
    day = m["verdict_table"]["day"]
    assert any(r["period"] != "ALL" for r in day["rows"])
    assert set(day["stability"]) == {"formula", "market"}
    # Model-vs-formula ALL improvement is ~0 (identity).
    f_all = next(r for r in day["rows"] if r["benchmark"] == "formula" and r["period"] == "ALL")
    assert abs(f_all["brier_improve_pct"]) < 1e-9

    for fname in ("metrics.json", "scores.csv", "verdict_table.csv", "reliability.csv", "report.html"):
        assert (paths.out_dir / "evaluate" / fname).exists(), fname


def test_evaluate_external_beats_formula(tmp_path):
    paths = _paths(tmp_path)
    preds = make_predictions_fixture(paths.out_dir / "_preds", paths.journal_dir)
    m = evaluate(paths, predictions=preds, min_windows=5)
    assert m["params"]["self_baseline"] is False
    assert m["counts"]["joined_snapshots"] > 0

    rows = m["verdict_table"]["day"]["rows"]
    f_all = next(r for r in rows if r["benchmark"] == "formula" and r["period"] == "ALL")
    # The oracle-nudge fixture moves each prediction toward the truth → strictly
    # better-calibrated than the formula benchmark.
    assert f_all["model_brier"] < f_all["bench_brier"]
    assert f_all["brier_improve_pct"] > 0.0
    # The market benchmark is scored on the recorded subset and is also computed.
    k_all = next(r for r in rows if r["benchmark"] == "market" and r["period"] == "ALL")
    assert math.isfinite(k_all["brier_improve_pct"])
    assert 0.0 < k_all["coverage"] <= 1.0


def test_evaluate_output_schema(tmp_path):
    paths = _paths(tmp_path)
    evaluate(paths, min_windows=5)
    verdict = pd.read_csv(paths.out_dir / "evaluate" / "verdict_table.csv")
    assert list(verdict.columns) == eh._VERDICT_CSV_COLS
    scores = pd.read_csv(paths.out_dir / "evaluate" / "scores.csv")
    assert list(scores.columns) == eh._SCORES_CSV_COLS
    # Both period rollups are present.
    assert set(verdict["by"]) == {"day", "week"}


def test_evaluate_determinism(tmp_path):
    paths_a = _paths(tmp_path, out_name="out_a")
    paths_b = Paths(journal_dir=paths_a.journal_dir, depth_dir=paths_a.depth_dir, out_dir=tmp_path / "out_b")
    evaluate(paths_a, min_windows=5)
    evaluate(paths_b, min_windows=5)
    a = pd.read_csv(paths_a.out_dir / "evaluate" / "verdict_table.csv")
    b = pd.read_csv(paths_b.out_dir / "evaluate" / "verdict_table.csv")
    pd.testing.assert_frame_equal(a, b)


def test_evaluate_rejects_truncated_predictions(tmp_path):
    paths = _paths(tmp_path)
    bad = paths.out_dir / "bad.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a parquet file")
    with pytest.raises(ParquetNotReady):
        evaluate(paths, predictions=bad, min_windows=5)


# --- direct harness math (multi-period, no journal needed) ------------------
def _synthetic_frame(n_days=3, windows_per_day=6, snaps=8, *, market_gap=False, seed=1):
    """A canonical eval frame spanning several UTC days where the model is nudged
    toward the truth (so it beats both benchmarks every period)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for w in range(windows_per_day):
            open_time = d * eh.MS_PER_DAY + w * 300_000
            outcome = int(rng.integers(0, 2))
            for k in range(snaps):
                ts = open_time + k * 30_000
                tau = 300.0 - k * 30.0
                formula = float(np.clip(0.5 + rng.normal(0.0, 0.15), 0.05, 0.95))
                model = float(np.clip(formula + 0.1 * (2 * outcome - 1), 0.02, 0.98))
                market = float(np.clip(formula + rng.normal(0.0, 0.05), 0.02, 0.98))
                if market_gap and (k % 2 == 0):
                    market = float("nan")
                rows.append(
                    {
                        "series": "BTC-5m", "open_time": open_time, "ts": ts,
                        "tau": tau, "outcome_up": float(outcome),
                        "model": model, "formula": formula, "market": market,
                        "health": "Ready",
                    }
                )
    df = pd.DataFrame(rows)
    df["tau_bucket"] = eh.tau_bucket(df["tau"].to_numpy(float), np.full(len(df), 300.0))
    df["day"] = (df["open_time"] // eh.MS_PER_DAY).astype("int64")
    df["week"] = (df["day"] // 7).astype("int64")
    df["in_coverage"] = df["market"].notna()
    return df


def test_stability_across_periods():
    df = _synthetic_frame(n_days=3)
    m = eh.run(df, counts={"resolved_windows": 18, "joined_snapshots": len(df)},
               title="t", model_label="nudged", self_baseline=False, min_windows=1)
    st = m["verdict_table"]["day"]["stability"]["formula"]
    assert st["periods_evaluated"] == 3
    assert st["periods_model_better"] == 3
    assert st["win_rate"] == 1.0
    assert st["mean_brier_improve_pct"] > 0.0
    assert st["reliably_better"] is True
    # Model beats formula overall too.
    overall = eh.find_scope(m["scope_rows"], "overall")
    assert overall["model_brier"] < overall["formula_brier"]
    # Nudging toward the outcome never worsens the side-call, so directional
    # accuracy is at least the formula's (and strictly better here).
    assert overall["model_diracc"] >= overall["formula_diracc"]


def test_market_benchmark_uses_coverage_subset():
    df = _synthetic_frame(n_days=2, market_gap=True)  # half the rows have no market
    m = eh.run(df, counts={"resolved_windows": 12, "joined_snapshots": len(df)},
               title="t", model_label="nudged", self_baseline=False, min_windows=1)
    overall = eh.find_scope(m["scope_rows"], "overall")
    h2h = eh.find_scope(m["scope_rows"], "overall_headtohead")
    # Head-to-head (model AND market) covers strictly fewer rows than the full set.
    assert h2h["n_market"] < overall["n_model"]
    k_all = next(r for r in m["verdict_table"]["day"]["rows"]
                 if r["benchmark"] == "market" and r["period"] == "ALL")
    assert k_all["coverage"] < 1.0
    # The formula benchmark, defined on every row, keeps full coverage.
    f_all = next(r for r in m["verdict_table"]["day"]["rows"]
                 if r["benchmark"] == "formula" and r["period"] == "ALL")
    assert f_all["coverage"] == 1.0
