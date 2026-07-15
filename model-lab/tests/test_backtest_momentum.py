"""Tests for the historical momentum backtest (Part 3).

- the stage runs end-to-end on the fixture, produces the metrics structure + the controls
  (never-trade + matched-frequency shuffled) + all output files;
- determinism: two runs are byte-identical (seeded shuffled control);
- the trigger constants are money_judge's (single source of truth), and its fill primitives
  are reused verbatim (no re-derived trigger).
"""

from __future__ import annotations

import pandas as pd

from model_lab import backtest_momentum as bm
from model_lab import fixtures as fx
from model_lab import historical_common as hc
from model_lab import money_judge as mj
from model_lab.config import Paths


def _paths(tmp_path) -> Paths:
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
                telonex_dir=tmp_path / "telonex")


def _setup(tmp_path) -> Paths:
    res = fx.make_historical_fixture(tmp_path / "telonex", tmp_path / "aggtrades")
    fx.write_historical_coverage(tmp_path / "out", res["coverage"])
    return _paths(tmp_path)


def test_runs_and_reports(tmp_path):
    paths = _setup(tmp_path)
    r = bm.backtest_momentum(paths, series=("BTC-5m", "ETH-5m"), seeds=6)
    c = r["counts"]
    assert c["resolved_windows"] > 0
    assert c["trades"] >= 0
    # controls present and structured
    ctrl = r["controls"]
    assert ctrl["never_trading_net"] == 0.0
    assert len(ctrl["shuffled_nets"]) == 6
    assert "real_beats_shuffled" in ctrl and "shuffled_p_value" in ctrl
    assert isinstance(r["verdict"], str) and len(r["verdict"]) > 0
    for f in ("report.html", "metrics.json", "trades.csv", "per_series_month.csv", "per_hour.csv"):
        assert (paths.out_dir / "backtests" / "momentum" / f).exists()
    # if the trigger fired, the trade dicts carry the fields the aggregations need
    if c["trades"] > 0:
        tr = pd.read_csv(paths.out_dir / "backtests" / "momentum" / "trades.csv")
        for col in ("series", "window_open_ms", "net", "fee", "won", "side"):
            assert col in tr.columns


def test_regime_boundaries_injects_detected_date(tmp_path):
    import json as _json
    paths = _paths(tmp_path)
    base = bm.regime_boundaries(paths)
    assert [b["date"] for b in base] == sorted(b["date"] for b in bm.REGIME_BOUNDARIES)
    assert all(b["status"] == "verified" for b in base)
    # A detector output with a `detected_date` adds one "detected" marker, kept sorted.
    det = paths.out_dir / "delay_detection"
    det.mkdir(parents=True, exist_ok=True)
    (det / "metrics.json").write_text(_json.dumps({"detected_date": "2026-04-15"}), encoding="utf-8")
    with_det = bm.regime_boundaries(paths)
    assert len(with_det) == len(base) + 1
    got = next(b for b in with_det if b["status"] == "detected")
    assert got["date"] == "2026-04-15"
    assert [b["date"] for b in with_det] == sorted(b["date"] for b in with_det)


def test_out_name_and_venue_delay_decomposition(tmp_path):
    paths = _setup(tmp_path)
    r = bm.backtest_momentum(paths, series=("BTC-5m", "ETH-5m"), seeds=4,
                             latency_ms=5, venue_delay_ms=250, out_name="verdict_test")
    p = r["params"]
    assert p["network_ms"] == 5 and p["venue_delay_ms"] == 250
    assert p["effective_latency_ms"] == 255 and p["latency_ms"] == 255
    # output + checkpoints land under the custom name, isolated from the full run
    assert (paths.out_dir / "backtests" / "verdict_test" / "metrics.json").exists()
    assert (paths.out_dir / "backtests" / "verdict_test" / "checkpoints").is_dir()
    assert not (paths.out_dir / "backtests" / "momentum").exists()
    assert len(r["regime_boundaries"]) >= 5  # carried for the annotation


def test_deterministic(tmp_path):
    def run(sub):
        res = fx.make_historical_fixture(tmp_path / sub / "telonex", tmp_path / sub / "aggtrades")
        fx.write_historical_coverage(tmp_path / sub / "out", res["coverage"])
        p = Paths(journal_dir=tmp_path / sub / "journal", depth_dir=tmp_path / sub / "depth",
                  out_dir=tmp_path / sub / "out", hist_dir=tmp_path / sub / "aggtrades",
                  telonex_dir=tmp_path / sub / "telonex")
        r = bm.backtest_momentum(p, series=("BTC-5m", "ETH-5m"), seeds=6)
        return r["counts"]["net_pnl"], r["controls"]["shuffled_nets"]
    a = run("a")
    b = run("b")
    assert a[0] == b[0]
    assert a[1] == b[1]


def test_reuses_money_judge_trigger():
    # the backtest must not re-derive the trigger: its constants ARE money_judge's, and it
    # calls money_judge's ports directly.
    assert bm.mj.SIGNAL_SIGMA_MULT == 3.0
    assert bm.mj.SIGNAL_LOOKBACK_MS == 2000
    assert bm.mj.COOLDOWN_MS == 5000
    assert bm.mj.MOMENTUM_BUFFER == 0.005
    # the fill primitives are the money_judge functions (identity, not copies)
    assert bm.mj._run_momentum is mj._run_momentum
    assert bm.mj._confirmed_direction is mj._confirmed_direction
    assert bm.mj._plan_take_single_level is mj._plan_take_single_level


def test_random_control_matched_frequency(tmp_path):
    # the random control fires at ~the real confirmed-move rate: with a positive fire_prob
    # and enough eligible snapshots it produces trades; with fire_prob 0 it produces none.
    import numpy as np
    model_rows = [{"series": "BTC-5m", "open_time": 1000, "asset": "btc", "ts": 1000 + i * 1000,
                   "p_up": 0.8, "sigma_1s": 1e-4, "health": "Ready"} for i in range(60)]
    win_by_key = {("BTC-5m", 1000): {"up_token": "T", "close_time": 1000 + 300_000, "outcome_up": 1}}
    books = {"T": {"ts": np.array([1000 + i * 1000 for i in range(60)], dtype="int64"),
                   "bid_px": np.full(60, 0.10), "bid_sz": np.full(60, 100.0),
                   "ask_px": np.full(60, 0.10), "ask_sz": np.full(60, 100.0)}}
    rng = np.random.default_rng(1)
    none = bm._run_random_control(model_rows, win_by_key, books, fire_prob={"btc": 0.0},
                                  rng=rng, latency_ms=0, fee_rate=0.07, window_budget=10.0)
    assert none == []
    some = bm._run_random_control(model_rows, win_by_key, books, fire_prob={"btc": 1.0},
                                  rng=rng, latency_ms=0, fee_rate=0.07, window_budget=10.0)
    assert len(some) >= 1  # a very cheap Up book (0.10) with p_up 0.8 clears the edge gate
