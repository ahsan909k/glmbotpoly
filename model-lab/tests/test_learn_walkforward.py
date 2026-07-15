"""Tests for the walk-forward challenger (learn_walkforward).

- expanding-window fold schedule (train anchored at lo, ~2-week blocks; fallbacks);
- the taker_ev label maths;
- walk-forward: a genuinely predictive matrix beats chance out-of-sample, a shuffled matrix
  collapses to chance, and two runs are byte-identical (determinism);
- the depth_source split gate passes when the two sources agree and FAILS loudly on a material gap;
- an end-to-end smoke on a synthetic multi-day historical_dataset (calibration only, no money);
- the money-scorer smoke on the Telonex fixture pointed at a current-regime day.

LightGBM is opt-in (the [gbt] extra); these importorskip it, like test_learn_short_gbt.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from model_lab import fixtures as fx  # noqa: E402
from model_lab import learn_walkforward as lw  # noqa: E402
from model_lab.config import Paths  # noqa: E402
from model_lab.lib import math as lm  # noqa: E402

MS_DAY = 86_400_000


# --- expanding folds --------------------------------------------------------
def test_expanding_folds_are_anchored_and_block_sized():
    days = np.arange(0, 100)
    folds, mode = lw._expanding_folds(days, init_train_days=56, test_days=14)
    assert mode == "walk_forward"
    assert all(f[0] == 0 for f in folds)  # EXPANDING: train_start always == lo
    assert folds[0][1] == 56 and folds[0][2] == 70  # first test block [56, 70)
    assert all((f[2] - f[1]) <= 14 for f in folds)  # ~2-week blocks (last clamped)
    # test blocks are contiguous and cover to the end
    assert folds[-1][2] == 100


def test_expanding_folds_fallbacks():
    short, mode = lw._expanding_folds(np.arange(0, 10), init_train_days=56, test_days=14)
    assert mode == "single_split" and len(short) == 1 and short[0][0] == 0
    none, mode2 = lw._expanding_folds(np.array([5]), init_train_days=56, test_days=14)
    assert mode2 == "insufficient" and none == []


# --- taker_ev label ---------------------------------------------------------
def test_taker_ev_label():
    # Held-to-resolution, buying Up profits iff Up wins AND the entry ask < ~1 (payoff $1 beats
    # ask+fee for any ask below ~0.99997) — so taker_ev ~= outcome_up except at a near-$1 ask.
    mat = pd.DataFrame({
        "pm_mid": [0.40, 0.9999, 0.40, np.nan],
        "pm_spread": [0.02, 0.0002, 0.02, 0.02],
        "outcome_up": [1.0, 1.0, 0.0, 1.0],
    })
    lab = lw._taker_ev_label(mat, fee_rate=0.07)
    assert lab[0] == 1.0  # Up wins, cheap ask ~0.41 → profitable
    assert lab[1] == 0.0  # Up wins but ask ~1.0 → entry too expensive → not profitable
    assert lab[2] == 0.0  # Up loses → never profitable
    assert np.isnan(lab[3])  # no book → NaN


# --- synthetic dataset builders --------------------------------------------
def _synth_matrix(*, n_days: int, per_day: int, seed: int, signal: bool = True) -> pd.DataFrame:
    """A synthetic historical_dataset frame with a predictive `z`→outcome mapping (or noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    base_open = 0  # day 0 windows open at t=0
    for d in range(n_days):
        for w in range(per_day):
            open_ms = (d * MS_DAY) + w * 300_000 + 43_200_000
            close_ms = open_ms + 300_000
            z = float(rng.normal(0, 1))
            p = 1.0 / (1.0 + np.exp(-z))
            # outcome follows z when signal, else independent
            outcome = int(rng.random() < (p if signal else 0.5))
            fwd10 = int(rng.random() < (p if signal else 0.5))
            src = "telonex" if (w % 2 == 0) else "recorder"
            for k in range(6):  # 6 samples per window
                ts = open_ms + 20_000 + k * 40_000
                rows.append({
                    "series": "BTC-5m", "asset": "btc", "window_open_ms": open_ms,
                    "sample_ts_ms": ts, "window_close_ms": close_ms,
                    "ret": float(rng.normal(0, 1e-3)), "realized_vol": 3e-4, "sigma_1s": 3e-4,
                    "log_s_k": z * 1e-3, "z": z, "p_up_model": p,
                    "tau_secs": (close_ms - ts) / 1000.0, "elapsed_secs": (ts - open_ms) / 1000.0,
                    "basis_bps": 0.0, "basis_ewma": 0.0,
                    "depth_imb_1": float(rng.normal(0, 0.3)), "depth_imb_5": 0.0, "depth_imb_10": 0.0,
                    "depth_imb_20": 0.0, "microprice_gap": 0.0, "bid_depth_slope": 1.0,
                    "ask_depth_slope": 1.0, "depth_spread_bps": 1.0,
                    "pm_mid": p, "pm_spread": 0.02, "pm_book_imb": 0.0,
                    "pm_staleness_1s": 0.0, "pm_staleness_2s": 0.0, "pm_staleness_3s": 0.0,
                    "outcome_up": float(outcome), "fwd_up_10s": np.int8(fwd10),
                    "label_source": "telonex", "depth_source": src,
                })
    return pd.DataFrame(rows)


def test_walk_forward_beats_chance_and_is_deterministic():
    mat = _synth_matrix(n_days=12, per_day=8, seed=1, signal=True)
    folds, mode = lw._expanding_folds(np.unique(mat["window_open_ms"].to_numpy() // MS_DAY), 4, 2)
    assert mode == "walk_forward" and len(folds) >= 2
    sub, y, info = lw._prep_target(mat, "dir10", fee_rate=0.07)
    params = lw._params(seed=7)
    oos1, _ = lw._walk_forward(sub, y, info, feature_names=lw.FULL_FEATURES, folds=folds,
                               purge_ms=10_000, params=params)
    oos2, _ = lw._walk_forward(sub, y, info, feature_names=lw.FULL_FEATURES, folds=folds,
                               purge_ms=10_000, params=params)
    assert len(oos1) > 0
    # determinism
    assert np.array_equal(oos1["p_up"].to_numpy(), oos2["p_up"].to_numpy())
    # real signal → OOS directional accuracy vs the fwd10 label beats chance
    acc = lm.directional_accuracy(oos1["p_up"].to_numpy(), oos1["y_true"].to_numpy())
    assert acc > 0.55


def test_shuffled_control_collapses():
    mat = _synth_matrix(n_days=12, per_day=8, seed=2, signal=True)
    folds, _ = lw._expanding_folds(np.unique(mat["window_open_ms"].to_numpy() // MS_DAY), 4, 2)
    sh = lw._shuffled_control(mat, folds, lw._params(seed=3), seed=3, fee_rate=0.07)
    assert sh["collapsed"] is True
    assert sh["pooled_brier"] >= sh["chance_brier"] - 0.03


# --- depth split gate -------------------------------------------------------
def _gate_oos(degrade: float) -> pd.DataFrame:
    """telonex predictions are accurate; recorder predictions are blended toward 0.5 by
    `degrade` (0 = identical to telonex → pass; large = far worse → material gap → fail)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        src = "telonex" if i % 2 == 0 else "recorder"
        y = float(rng.random() < 0.5)
        good = 0.95 if y > 0.5 else 0.05
        p = good if src == "telonex" else (1.0 - degrade) * good + degrade * 0.5
        rows.append({"depth_source": src, "outcome_up": y, "p_up": p,
                     "window_open_ms": i * 300_000})
    return pd.DataFrame(rows)


def test_depth_split_gate_pass_and_fail():
    ok = lw._depth_split_gate(_gate_oos(degrade=0.0))
    assert ok["pass"] is True
    bad = lw._depth_split_gate(_gate_oos(degrade=0.9))  # recorder blended to ~0.5 → far worse
    assert bad["pass"] is False and "gap" in bad["reason"].lower()


# --- end-to-end smoke (no money) -------------------------------------------
def _paths(tmp_path) -> Paths:
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                 out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
                 telonex_dir=tmp_path / "telonex")


def test_learn_walkforward_smoke_no_money(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_out()
    mat = _synth_matrix(n_days=10, per_day=8, seed=4, signal=True)
    mat.to_parquet(paths.table("historical_dataset"), index=False)
    res = lw.learn_walkforward(paths, init_train_days=4, test_days=2, seed=5,
                               run_money=False, run_shuffle=True, out_name="lw")
    assert res["n_rows_loaded"] == len(mat)
    assert res["fold_mode"] == "walk_forward"
    # OOS parquets for both targets x both variants
    for t in lw.TARGET_ORDER:
        for v in lw.VARIANT_ORDER:
            assert (paths.out_dir / "lw" / f"oos_{t}_{v}.parquet").exists()
    assert res["per_block"]["dir10"]  # per-block table
    assert "sources" in res["depth_gate"]
    assert res["shuffled"]["collapsed"] in (True, False)
    assert (paths.out_dir / "lw" / "report.html").exists()
    assert (paths.out_dir / "lw" / "metrics.json").exists()


# --- money-scorer smoke (Telonex fixture, current-regime day) --------------
def test_money_score_smoke(tmp_path):
    day = date(2026, 6, 6)  # current regime (>= 2026-06-05)
    res = fx.make_historical_fixture(tmp_path / "telonex", tmp_path / "aggtrades", day=day)
    fx.write_historical_coverage(tmp_path / "out", res["coverage"])
    paths = _paths(tmp_path)
    paths.ensure_out()
    since = int(datetime(2026, 6, 5, tzinfo=timezone.utc).timestamp()) * 1000
    # Build an OOS that leans strongly toward each window's true outcome.
    rows = []
    for (series, open_ms), meta in res["res_map"].items():
        lean = 0.9 if meta["outcome"] == "Up" else 0.1
        for ts in range(open_ms + 20_000, open_ms + 280_000, 30_000):
            rows.append({"series": series, "window_open_ms": open_ms, "sample_ts_ms": ts,
                         "p_up": lean, "p_up_model": lean, "pm_mid": lean,
                         "outcome_up": 1.0 if meta["outcome"] == "Up" else 0.0,
                         "tau_secs": (open_ms + 300_000 - ts) / 1000.0,
                         "window_close_ms": open_ms + 300_000,
                         "label_source": "telonex", "depth_source": "telonex"})
    oos = pd.DataFrame(rows)
    money = lw._money_score(paths, {"dir10": oos}, since_ms=since, network_ms=5, venue_delay_ms=250,
                            fee_rate=0.07, window_budget=10.0, trade_budget=10.0,
                            res_map=res["res_map"], out_dir=paths.out_dir / "lwm")
    assert money["current_regime_windows"] > 0
    assert money["effective_latency_ms"] == 255
    # the strong-lean model should place at least some trades against the recorded book
    best = money["targets"]["dir10"]["best"]
    assert best is not None and best["taker"]["n_trades"] >= 1
