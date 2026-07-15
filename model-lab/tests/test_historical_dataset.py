"""Tests for the historical dataset builder (Part 1).

- **No-look-ahead differential**: recompute one Telonex sample's features from full,
  truncated-at-T, and future-corrupted feature grids; assert every ``FEATURE_COLS`` value
  is bit-identical (mirrors ``test_dataset.test_no_lookahead_differential`` for the Telonex
  path — depth + PM + price/vol).
- **Exclusion**: coverage.json ``missing_windows`` and ambiguous (unresolved) windows are
  dropped, never guessed.
- **depth_source** tagging (telonex vs recorder) and the schema/columns.
- **Overlap parity**: matching recorder + Telonex depth → the guard PASSes.
- **Resumability**: a re-run skips days already in the manifest.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from model_lab import historical_common as hc
from model_lab import historical_dataset as hd
from model_lab import short_horizon as sh
from model_lab.config import Paths
from model_lab.fixtures import (
    make_fixture,
    make_historical_fixture,
    make_overlap_parity_fixture,
    write_historical_coverage,
)
from model_lab.io import telonex_pm as tpm


def _paths(tmp_path) -> Paths:
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
                telonex_dir=tmp_path / "telonex")


def _setup_hist(tmp_path) -> Paths:
    res = make_historical_fixture(tmp_path / "telonex", tmp_path / "aggtrades")
    write_historical_coverage(tmp_path / "out", res["coverage"])
    return _paths(tmp_path)


# --------------------------------------------------------------------------- #
# no-look-ahead differential (Telonex path)
# --------------------------------------------------------------------------- #
def _hist_grids(paths: Paths, d: date):
    """Build the (window, feature grid, depth grid, pm grid) for the first BTC window on
    ``d`` — the pieces the historical builder attaches from (the outcome is irrelevant to
    the FEATURE_COLS differential test)."""
    excl, _ = hc.load_excluded_slugs(paths)
    wins, _ = hc.telonex_windows_present(paths, "BTC-5m", d, excluded=excl)
    bars = hc.binance_bars_for_day(paths, "BTCUSDT", d)
    grid = hc.ds.build_feature_grid(hc.ds._binance_ticks_from_bars(bars, "btc"),
                                    pd.DataFrame(columns=hc.ds.DEPTH_COLS), "btc")
    depth_grid = hc.telonex_depth_feature_grid(paths, "BTCUSDT", "btc", d)
    day_str = d.isoformat()
    for w in wins:
        upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
        st = hc.binance_anchored_strike(bars, w["window_open_ms"], 2500.0)
        if upq.empty or not (st["strike"] > 0.0):
            continue
        up_token = str(upq["token_id"].iloc[0])
        pm_df = upq.rename(columns={"ts_ms": "ts"})
        pm_df["token_id"] = up_token
        pm_grid = sh.build_pm_grid(pm_df[["token_id", "ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]],
                                   {"btc": grid[["sec", "mid"]]}, {up_token: "btc"})
        wdf = pd.DataFrame([{
            "series": "BTC-5m", "asset": "btc", "window_open_ms": w["window_open_ms"],
            "window_close_ms": w["window_close_ms"], "up_token": up_token, "down_token": None,
            "strike": st["strike"], "strike_ts_ms": st["strike_ts_ms"],
            "strike_quality": st["strike_quality"], "label_source": "telonex",
            "outcome": "Up", "outcome_up": 1, "split": "train",  # outcome irrelevant to features
        }])
        return wdf, grid, depth_grid, pm_grid, "btc"
    pytest.skip("no window in the fixture")


def _sample_row(wdf, grid, depth_grid, pm_grid, T):
    batch = sh._build_samples(wdf, grid, sh.DEFAULT_GRID_SECS, sh.DEFAULT_HORIZONS,
                              sh.DEFAULT_MAX_FEATURE_STALENESS_SECS * 1000, depth_grid, pm_grid,
                              {}, {}, 2000)
    row = batch[batch["sample_ts_ms"] == T]
    return row


def _trunc_grid(g, T):
    return g[g["ts_ms"].astype("int64") <= T].copy() if not g.empty else g


def _corrupt_grid(g, T, cols):
    if g.empty:
        return g
    g = g.copy()
    mask = g["ts_ms"].astype("int64") > T
    for c in cols:
        if c in g.columns:
            g.loc[mask, c] = g.loc[mask, c] * -10.0
    return g


def test_no_lookahead_differential(tmp_path):
    paths = _setup_hist(tmp_path)
    d = date(2026, 5, 1)
    wdf, grid, depth_grid, pm_grid, _ = _hist_grids(paths, d)
    open_ms = int(wdf["window_open_ms"].iloc[0])
    T = open_ms + 150_000  # a mid-window 5s-grid sample

    full = _sample_row(wdf, grid, depth_grid, pm_grid, T)
    assert len(full) == 1, "sample not produced"

    # (i) truncate every feature grid to ≤ T; (ii) corrupt every grid row > T.
    trunc = _sample_row(wdf, _trunc_grid(grid, T), _trunc_grid(depth_grid, T), _trunc_grid(pm_grid, T), T)
    corrupt = _sample_row(
        wdf,
        _corrupt_grid(grid, T, ["mid", "ret", "realized_vol", "sigma_1s", "log_s_k", "z", "p_up_model"]),
        _corrupt_grid(depth_grid, T, [c for c in depth_grid.columns if c not in ("asset", "sec", "ts_ms")]),
        _corrupt_grid(pm_grid, T, [c for c in pm_grid.columns if c not in ("up_token", "sec", "ts_ms", "pm_run_secs")]),
        T,
    )
    for col in sh.FEATURE_COLS:
        a = full[col].to_numpy()
        assert np.array_equal(a, trunc[col].to_numpy(), equal_nan=True), f"{col} changed after truncation"
        assert np.array_equal(a, corrupt[col].to_numpy(), equal_nan=True), f"{col} changed after future corruption"
    # non-vacuity: the sample really has features
    assert np.isfinite(full["mid"].iloc[0]) and np.isfinite(full["p_up_model"].iloc[0])
    assert np.isfinite(full["depth_imb_20"].iloc[0])


# --------------------------------------------------------------------------- #
# exclusion, depth_source, schema
# --------------------------------------------------------------------------- #
def test_exclusion_and_depth_source(tmp_path):
    paths = _setup_hist(tmp_path)
    res = hd.historical_dataset(paths, series=("BTC-5m", "ETH-5m"))
    hd.combine(paths)
    df = pd.read_parquet(paths.table("historical_dataset"))
    assert list(df.columns) == hd.HIST_COLUMNS
    assert (df["label_source"] == "telonex").all()
    # every row's depth is telonex-sourced (the fixture book covers the day)
    assert (df["depth_source"] == "telonex").all()
    # 6 candidates/series → 5 present (1 coverage-excluded via coverage.json), all labelled
    # by the OFFICIAL resolution (the formerly-ambiguous window is now labelled too).
    assert res["counts"]["no_official_excluded"] == 0
    n_windows = df.groupby(["series", "window_open_ms"]).ngroups
    assert n_windows == 10  # 5 present × 2 series
    # chainlink-definition features are journal-era-only (NaN on telonex rows)
    assert df["chainlink"].isna().all()
    # depth_source == "telonex" exactly where the depth features are present
    assert (df.loc[df["depth_imb_1"].notna(), "depth_source"] == "telonex").all()


def test_missing_window_dropped(tmp_path):
    paths = _setup_hist(tmp_path)
    excl, rep = hc.load_excluded_slugs(paths)
    assert rep["excluded_slugs"] >= 1
    wins, wrep = hc.telonex_windows_present(paths, "BTC-5m", date(2026, 5, 1), excluded=excl)
    kept = {w["slug"] for w in wins}
    assert not (kept & excl), "an excluded (missing_windows) slug survived enumeration"


# --------------------------------------------------------------------------- #
# resumability
# --------------------------------------------------------------------------- #
def test_resumable(tmp_path):
    paths = _setup_hist(tmp_path)
    r1 = hd.historical_dataset(paths, series=("BTC-5m",))
    assert r1["counts"]["days_built"] >= 1 and r1["counts"]["days_skipped"] == 0
    r2 = hd.historical_dataset(paths, series=("BTC-5m",))
    assert r2["counts"]["days_built"] == 0 and r2["counts"]["days_skipped"] >= 1


# --------------------------------------------------------------------------- #
# journal-owned path + overlap parity
# --------------------------------------------------------------------------- #
def test_journal_owned_path(tmp_path):
    ts0 = int(datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc).timestamp()) * 1000
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=6, ts0_ms=ts0)
    paths = _paths(tmp_path)
    since = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp()) * 1000
    until = int(datetime(2026, 7, 5, tzinfo=timezone.utc).timestamp()) * 1000
    hd.historical_dataset(paths, series=("BTC-5m",), since_ms=since, until_ms=until)
    hd.combine(paths)
    df = pd.read_parquet(paths.table("historical_dataset"))
    assert len(df) > 0
    assert (df["label_source"] == "chainlink").all()
    assert (df["depth_source"] == "recorder").all()
    assert df["chainlink"].notna().any()  # journal era HAS Chainlink


def test_overlap_parity_passes(tmp_path):
    day = date(2026, 7, 4)
    make_overlap_parity_fixture(tmp_path / "telonex", tmp_path / "aggtrades",
                                tmp_path / "journal", tmp_path / "depth", day=day)
    paths = _paths(tmp_path)
    since = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp()) * 1000
    until = int(datetime(2026, 7, 5, tzinfo=timezone.utc).timestamp()) * 1000
    hd.historical_dataset(paths, series=("BTC-5m",), since_ms=since, until_ms=until)
    par = hd.overlap_parity(paths, ("BTC-5m",), [day])
    assert par["n_features"] == 6, par
    assert par["pass"], par["verdicts"]
    for v in par["verdicts"]:
        assert v["corr"] >= 0.99 and 0.98 <= v["magnitude_ratio"] <= 1.02
