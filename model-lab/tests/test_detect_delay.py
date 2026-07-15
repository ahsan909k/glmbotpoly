"""Unit tests for detect_delay — the pure move/lag/step core (no heavy IO)."""

from __future__ import annotations

import numpy as np

from model_lab import detect_delay as dd
from model_lab.config import Paths, parse_time_bound


def test_detect_moves_finds_an_injected_jump():
    ts = np.arange(0, 30000, 100, dtype="int64")  # 300 ticks @ 100 ms
    rng = np.random.default_rng(0)
    px = 100.0 + rng.normal(0, 0.001, ts.size)  # tiny iid noise
    px[ts >= 15000] += 0.6  # a +60 bps step at t = 15 s
    mts, msg = dd.detect_moves(ts, px)
    assert mts.size >= 1
    assert any(14000 <= t <= 16500 and s == 1 for t, s in zip(mts, msg))


def test_detect_moves_flat_series_has_no_moves():
    ts = np.arange(0, 20000, 100, dtype="int64")
    px = np.full(ts.size, 100.0)  # perfectly flat → σ = 0 → no moves
    mts, _ = dd.detect_moves(ts, px)
    assert mts.size == 0


def test_calibrate_side_auto_maps_from_data():
    bts = np.arange(0, 20000, 100, dtype="int64")
    bpx = np.full(bts.size, 100.0)
    bpx[bts >= 10000] = 100.5  # up step at 10 s
    bpx[bts >= 15000] = 100.0  # down step at 15 s
    # BUY prints follow the up step; SELL prints follow the down step.
    p_ts = np.array([10200, 10400, 10600, 10800, 10900,
                     15200, 15400, 15600, 15800, 15900], dtype="int64")
    p_sd = np.array(["BUY"] * 5 + ["SELL"] * 5)
    smap = dd.calibrate_side(p_ts, p_sd, bts, bpx)
    assert smap["BUY"] == 1 and smap["SELL"] == -1


def test_match_lags_takes_first_same_sign_print():
    mts = np.array([1000, 5000], dtype="int64")
    msg = np.array([1, -1])
    # up-move @1000: an opposite-sign print @1200 is skipped, +1 print @1300 → lag 300.
    # down-move @5000: -1 print @5150 → lag 150.
    p_ts = np.array([1200, 1300, 5150], dtype="int64")
    p_sg = np.array([-1, 1, -1])
    assert sorted(dd.match_lags(mts, msg, p_ts, p_sg)) == [150, 300]


def test_match_lags_respects_horizon():
    mts = np.array([0], dtype="int64")
    msg = np.array([1])
    p_ts = np.array([dd.REACT_HORIZON_MS + 500], dtype="int64")  # beyond horizon
    p_sg = np.array([1])
    assert dd.match_lags(mts, msg, p_ts, p_sg) == []


def test_detect_step_finds_injected_step():
    weeks = ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23", "2026-03-30", "2026-04-06"]
    floors = [80, 90, 100, 260, 255, 260]  # clean low→high step at index 3
    recs = [{"series": "BTC-5m", "week_start": wk, "n_lags": 100, "p10": f}
            for wk, f in zip(weeks, floors)]
    step, trusted = dd.detect_step_for_series(recs)
    assert step == "2026-03-23"
    assert len(trusted) == 6


def test_detect_step_none_when_flat():
    recs = [{"series": "X", "week_start": f"2026-03-{d:02d}", "n_lags": 100, "p10": 100}
            for d in (2, 9, 16, 23)]
    step, _ = dd.detect_step_for_series(recs)
    assert step is None


def test_detect_step_ignores_low_sample_weeks():
    # The high weeks are all below the sample floor → not trusted → no step.
    weeks = ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23"]
    recs = [{"series": "X", "week_start": w, "n_lags": (100 if i < 2 else 5), "p10": (80 if i < 2 else 260)}
            for i, w in enumerate(weeks)]
    step, trusted = dd.detect_step_for_series(recs)
    assert step is None and len(trusted) == 2


def test_runs_on_absent_data_writes_null(tmp_path):
    paths = Paths(journal_dir=tmp_path / "j", depth_dir=tmp_path / "d", out_dir=tmp_path / "out",
                  hist_dir=tmp_path / "agg", telonex_dir=tmp_path / "tel")
    r = dd.detect_delay(paths, series=("BTC-5m",),
                        since_ms=parse_time_bound("2026-02-01"), until_ms=parse_time_bound("2026-03-01"))
    assert r["detected_date"] is None
    assert r["per_week_series"] == []
    assert (paths.out_dir / "delay_detection" / "metrics.json").exists()
    assert (paths.out_dir / "delay_detection" / "report.html").exists()
