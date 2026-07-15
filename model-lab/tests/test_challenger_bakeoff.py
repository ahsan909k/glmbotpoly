"""Tests for challenger_bakeoff — the per-backend producer + the aggregator/verdict.

The GBT-backend-byte-identical-to-accuracy_push test is the anti-drift tripwire: it proves the
bake-off reuses accuracy_push's methodology exactly (only the model seam differs).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from model_lab import accuracy_push as ap
from model_lab import challenger_bakeoff as cb
from model_lab.config import Paths
from model_lab.lib import backends as bk

_DEP = {"gbt": "lightgbm", "xgb": "xgboost", "mlp": None, "logreg": None}


def _skip_if_missing(name: str) -> None:
    dep = _DEP[name]
    if dep is not None:
        pytest.importorskip(dep)


def _paths(tmp_path):
    return Paths(journal_dir=tmp_path / "j", depth_dir=tmp_path / "d", out_dir=tmp_path / "out")


def _write_synth_historical(paths: Paths, market: str = "BTC-5m", seed: int = 0) -> None:
    """A synthetic historical_dataset.parquet where feature ``z`` carries the 15 s label; other
    features are noise. Rows span pre-regime (selection), the pre-regime val tail, and the regime OOS,
    with enough rows to clear accuracy_push's guards (sel ≥ 5000, regime ≥ 1000)."""
    rng = np.random.default_rng(seed)
    floor = ap._regime_floor_ms(ap.REGIME_FROM)
    d = ap.MS_PER_DAY

    def block(n, lo_days, hi_days):
        wo = rng.integers(floor + lo_days * d, floor + hi_days * d, n).astype(np.int64)
        return wo

    wo = np.concatenate([block(7000, -60, -26), block(2000, -20, -2), block(2000, 1, 9)])
    n = len(wo)
    ts = wo + rng.integers(0, 250_000, n).astype(np.int64)
    z = rng.normal(0, 1, n)
    label = (z + 0.3 * rng.normal(0, 1, n) > 0).astype(np.float64)

    data = {f: rng.normal(0, 1, n).astype(np.float32) for f in ap.ALL_FEATURES}
    data["z"] = z.astype(np.float32)
    data["tau_secs"] = rng.uniform(5.0, 300.0, n).astype(np.float32)
    data.update({
        "series": market, "window_open_ms": wo, "sample_ts_ms": ts,
        "outcome_up": rng.integers(0, 2, n), "label_source": "chainlink", ap.LABEL: label,
    })
    df = pd.DataFrame(data).sort_values(["window_open_ms", "sample_ts_ms"]).reset_index(drop=True)
    paths.ensure_out()
    df.to_parquet(paths.out_dir / "historical_dataset.parquet", index=False)


_OOS_COLS = ["series", "window_open_ms", "sample_ts_ms", "p_up", "y_true", "outcome_up",
             "tau_secs", "label_source"]


@pytest.mark.parametrize("name", ["mlp", "logreg", "gbt", "xgb"])
def test_backend_recovers_signal_and_writes_contract(tmp_path, name):
    _skip_if_missing(name)
    paths = _paths(tmp_path)
    _write_synth_historical(paths, seed=1)
    out_dir = paths.out_dir / "bakeoff" / name
    out_dir.mkdir(parents=True)
    r = cb.run_market_backend(paths, "BTC-5m", bk.BACKENDS[name], days=None, seed=7, threads=1,
                              out_dir=out_dir, regime_from=ap.REGIME_FROM, run_shuffle=True)
    assert "error" not in r
    assert r["oos_diracc"] > 0.7           # recovers the z→label signal
    assert r["lift_diracc"] > 0.1          # well above the base rate
    assert "theta_gate" in r and np.isfinite(r["theta_gate"])
    # 8-column OOS contract (burst_pair reads 4 of these).
    oos = pd.read_parquet(out_dir / f"oos_BTC-5m.parquet")
    assert list(oos.columns) == _OOS_COLS
    assert oos["p_up"].between(0, 1).all()
    # model-level shuffled-label control collapses (no leakage).
    assert r["shuffled_control"]["collapsed"] is True


def test_gbt_backend_byte_identical_to_accuracy_push(tmp_path):
    pytest.importorskip("lightgbm")
    paths = _paths(tmp_path)
    _write_synth_historical(paths, seed=2)

    ap_dir = paths.out_dir / "accuracy_push"
    ap_dir.mkdir(parents=True)
    ap.run_market(paths, "BTC-5m", days=None, seed=7, threads=1, out_dir=ap_dir,
                  regime_from=ap.REGIME_FROM)

    bk_dir = paths.out_dir / "bakeoff" / "gbt"
    bk_dir.mkdir(parents=True)
    cb.run_market_backend(paths, "BTC-5m", bk.BACKENDS["gbt"], days=None, seed=7, threads=1,
                          out_dir=bk_dir, regime_from=ap.REGIME_FROM, run_shuffle=False)

    a = pd.read_parquet(ap_dir / "oos_BTC-5m.parquet")
    b = pd.read_parquet(bk_dir / "oos_BTC-5m.parquet")
    assert np.array_equal(a["window_open_ms"].to_numpy(), b["window_open_ms"].to_numpy())
    assert np.array_equal(a["sample_ts_ms"].to_numpy(), b["sample_ts_ms"].to_numpy())
    assert np.array_equal(a["p_up"].to_numpy(), b["p_up"].to_numpy())  # byte-identical predictions


def test_run_backend_writes_layout_and_gate(tmp_path):
    # logreg is dependency-free → exercises the full run_backend orchestration + resume.
    paths = _paths(tmp_path)
    _write_synth_historical(paths, seed=3)
    res = cb.run_backend(paths, "logreg", markets=("BTC-5m",), seed=7, threads=1, run_shuffle=False)
    out_dir = paths.out_dir / "bakeoff" / "logreg"
    assert (out_dir / "metrics.json").exists() and (out_dir / "frontier.csv").exists()
    assert (out_dir / "report.html").exists() and (out_dir / "oos_BTC-5m.parquet").exists()
    # metrics.json carries a per-market theta_gate → a valid burst_pair --oos-dir gate source.
    m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "theta_gate" in m["markets"]["BTC-5m"]
    # resume: a second call reloads the existing market (no crash, same layout).
    res2 = cb.run_backend(paths, "logreg", markets=("BTC-5m",), seed=7, threads=1, run_shuffle=False)
    assert res2["markets"]["BTC-5m"]["market"] == "BTC-5m"
    assert res["markets"]["BTC-5m"]["oos_diracc"] == res2["markets"]["BTC-5m"]["oos_diracc"]


def test_beats_champion_two_part_gate():
    champ = {"model": "gbt", "oos_diracc": 0.60, "oos_brier": 0.24}
    # (a) better model + (b) money clears both shuffled bands → beats.
    strong = {"model": "xgb", "oos_diracc": 0.62, "oos_brier": 0.235,
              "net_C0.94_U20": 500.0, "shuf_C0.94_U20": -100.0,
              "net_C0.98_U20": 300.0, "shuf_C0.98_U20": 50.0}
    a, b, phrase = cb._beats_champion(strong, champ)
    assert a and b and "BEATS GBT" in phrase
    # (a) only → "money-inconclusive".
    weak_money = {**strong, "net_C0.94_U20": 20.0, "shuf_C0.94_U20": -100.0}
    a, b, phrase = cb._beats_champion(weak_money, champ)
    assert a and not b and "money-inconclusive" in phrase
    # neither → does not beat.
    worse = {"model": "mlp", "oos_diracc": 0.59, "oos_brier": 0.245,
             "net_C0.94_U20": 10.0, "shuf_C0.94_U20": -5.0,
             "net_C0.98_U20": 10.0, "shuf_C0.98_U20": -5.0}
    a, b, phrase = cb._beats_champion(worse, champ)
    assert not a and "does not beat GBT" in phrase


def test_aggregate_emits_verdict_from_metrics(tmp_path):
    paths = _paths(tmp_path)
    # hand-write two models' model + money metrics on disk.
    for model, diracc, brier in [("gbt", 0.60, 0.240), ("xgb", 0.605, 0.238)]:
        md = paths.out_dir / "bakeoff" / model
        md.mkdir(parents=True)
        markets = {mk: {"oos_diracc": diracc, "oos_brier": brier, "oos_logloss": 0.66,
                        "lift_diracc": diracc - 0.5, "oos_ece": 0.01, "n_labeled": 1000,
                        "coverage_frac_at_gate": 0.02, "avg_signals_per_window": 0.6,
                        "theta_85_reached": True,
                        "shuffled_control": {"collapsed": True}} for mk in ap.MARKETS}
        (md / "metrics.json").write_text(json.dumps({"markets": markets}), encoding="utf-8")
        moneyd = paths.out_dir / "backtests" / f"bakeoff_{model}"
        moneyd.mkdir(parents=True)
        cells = {"C0.94_U20": {"net_pnl": 140.0, "shuffled_net": -280.0},
                 "C0.98_U20": {"net_pnl": -170.0, "shuffled_net": -1000.0}}
        (moneyd / "metrics.json").write_text(
            json.dumps({"cells": cells, "best_cell": "C0.94_U20"}), encoding="utf-8")

    res = cb.aggregate(paths, models=("gbt", "xgb"))
    out_dir = paths.out_dir / "bakeoff"
    assert (out_dir / "verdict.md").exists() and (out_dir / "verdict.html").exists()
    assert (out_dir / "summary.csv").exists()
    assert "xgb" in res["verdicts"]
    # xgb Δdiracc = +0.5 pp (< 1 pp) → does not clear the model gate.
    assert "does not beat GBT" in res["verdicts"]["xgb"]
    assert res["headline"].startswith("Nothing beats GBT")
