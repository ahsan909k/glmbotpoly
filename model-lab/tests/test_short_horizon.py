"""Tests for the ``short_horizon`` stage — schema, the two 10s/15s forward labels,
per-sample recording coverage, per-day report, and the no-look-ahead guards.

The three guards are exercised as for ``dataset``:

- the static AST scan lives in ``test_dataset.py`` and is extended there with the
  short-horizon label names (the feature functions are shared, so they are already
  scanned) — it is deliberately not re-run here;
- ``test_no_lookahead_differential`` / ``…_historical`` recompute the stage's
  ``FEATURE_COLS`` from full / future-truncated / future-corrupted history and assert
  bit-identity (coverage columns are excluded — they may read a recording just after
  the sample);
- a *positive* test proves the labels genuinely react to ``t+10`` / ``t+15`` (so the
  differential test is not vacuously green);
- the runtime assertion ``short_horizon._assert_no_lookahead`` runs inside every
  ``short_horizon()`` call, so the smoke tests exercise it too.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model_lab import dataset, short_horizon
from model_lab.config import Paths
from model_lab.fixtures import make_aggtrades_fixture, make_fixture
from model_lab.ingest import DEPTH_COLS
from model_lab.io import depth as depth_io


def _paths(tmp_path) -> Paths:
    # An isolated (empty) hist_dir keeps these journal-only tests hermetic.
    return Paths(
        journal_dir=tmp_path / "journal",
        depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out",
        hist_dir=tmp_path / "hist_empty",
    )


def _hist_paths(tmp_path, n_windows: int = 20) -> Paths:
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=n_windows)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    return Paths(
        journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
    )


# --------------------------------------------------------------------------- #
# smoke + schema
# --------------------------------------------------------------------------- #
def test_smoke_and_schema(tmp_path):
    paths = _hist_paths(tmp_path)
    result = short_horizon.short_horizon(paths, grid_secs=5)

    assert paths.table("short_horizon").exists()
    assert (paths.out_dir / "short_horizon" / "SCHEMA.md").exists()
    assert (paths.out_dir / "short_horizon" / "metadata.json").exists()

    df = pd.read_parquet(paths.table("short_horizon"), engine="pyarrow")
    assert len(df) == result["counts"]["samples"] > 0

    # Exact schema: column set + order match the single source of truth.
    assert list(df.columns) == short_horizon.COLUMNS
    assert not (set(short_horizon.FEATURE_COLS) & set(short_horizon.LABEL_COLS))
    assert not (set(short_horizon.FEATURE_COLS) & set(short_horizon.COVERAGE_COLS))

    # Window framing invariants.
    assert (df["window_open_ms"] <= df["sample_ts_ms"]).all()
    assert (df["sample_ts_ms"] < df["window_close_ms"]).all()
    assert (df["tau_secs"] > 0).all()
    assert (df["elapsed_secs"] >= 0).all()

    # Both forward labels well-formed; window outcome secondary labels present.
    assert df["fwd_up_10s"].dropna().isin([0, 1]).all()
    assert df["fwd_up_15s"].dropna().isin([0, 1]).all()
    assert set(df["outcome"].dropna().unique()) <= {"Up", "Down"}
    assert df["outcome_up"].isin([0, 1]).all()

    # Coverage flags are proper bools.
    for col in ("depth_covered", "book_covered", "depth_feat_covered", "pm_feat_covered"):
        assert df[col].dtype == bool, f"{col} not bool"

    # The KEY invariant: the *_feat_covered flags mean the features are ACTUALLY present
    # (what a training-row filter needs), exactly matching the feature columns' presence —
    # unlike book_covered (up-OR-down, ±tol recording provenance).
    assert (df["pm_feat_covered"] == df["pm_mid"].notna()).all()
    assert (df["pm_feat_covered"] == df["pm_book_imb"].notna()).all()
    assert (df["depth_feat_covered"] == df["depth_imb_1"].notna()).all()
    assert (df["depth_feat_covered"] == df["microprice_gap"].notna()).all()

    # Both sources present; dense grid = 300s / 5s = 60 samples per window.
    assert set(df["label_source"].unique()) == {"chainlink", "binance_proxy"}
    per_window = df.groupby(["series", "window_open_ms"]).size()
    assert (per_window == 60).all()

    # New microstructure features: populated on chainlink (our recordings), NaN on proxy.
    ch = df[df["label_source"] == "chainlink"]
    px = df[df["label_source"] == "binance_proxy"]
    for col in ("depth_imb_1", "depth_imb_20", "microprice_gap", "bid_depth_slope",
                "depth_spread_bps", "pm_mid", "pm_spread", "pm_book_imb", "pm_staleness_3s"):
        assert ch[col].notna().any(), f"{col}: no live values on chainlink"
        assert px[col].isna().all(), f"{col}: should be NaN on binance_proxy"
    # Math-bounded features stay in range.
    for col in ("depth_imb_1", "depth_imb_5", "depth_imb_10", "depth_imb_20", "pm_book_imb"):
        assert df[col].dropna().between(-1.0, 1.0).all(), f"{col} out of [-1,1]"
    assert df["pm_mid"].dropna().between(0.0, 1.0).all()

    # The distribution sanity report is written and does not FAIL.
    for fname in ("sanity.json", "sanity.html"):
        assert (paths.out_dir / "short_horizon" / fname).exists(), f"{fname} not written"
    assert result["sanity_status"] in ("PASS", "WARN")


def test_determinism(tmp_path):
    paths = _hist_paths(tmp_path, n_windows=10)
    a = Paths(journal_dir=paths.journal_dir, depth_dir=paths.depth_dir,
              out_dir=tmp_path / "out_a", hist_dir=paths.hist_dir)
    b = Paths(journal_dir=paths.journal_dir, depth_dir=paths.depth_dir,
              out_dir=tmp_path / "out_b", hist_dir=paths.hist_dir)
    short_horizon.short_horizon(a, grid_secs=5)
    short_horizon.short_horizon(b, grid_secs=5)
    da = pd.read_parquet(a.table("short_horizon"), engine="pyarrow")
    db = pd.read_parquet(b.table("short_horizon"), engine="pyarrow")
    pd.testing.assert_frame_equal(da, db)


# --------------------------------------------------------------------------- #
# labels genuinely read the future (positive control)
# --------------------------------------------------------------------------- #
def test_labels_react_to_future():
    secs = np.arange(1000, 1040)

    def grid_with(mid_1010: float, mid_1015: float) -> pd.DataFrame:
        mid = np.full(len(secs), 50_000.0)
        mid[secs == 1010] = mid_1010
        mid[secs == 1015] = mid_1015
        return pd.DataFrame({"asset": "btc", "sec": secs, "mid": mid})

    up = short_horizon.forward_direction_labels_multi(grid_with(50_100.0, 50_100.0), (10, 15))
    r_up = up[up["sec"] == 1000].iloc[0]
    assert r_up["fwd_ret_10s"] > 0 and r_up["fwd_ret_15s"] > 0

    dn = short_horizon.forward_direction_labels_multi(grid_with(49_900.0, 49_900.0), (10, 15))
    r_dn = dn[dn["sec"] == 1000].iloc[0]
    assert r_dn["fwd_ret_10s"] < 0 and r_dn["fwd_ret_15s"] < 0

    # Through _attach_labels: a future-only change (the mid at t+10 / t+15) flips the
    # binary labels. Proves the labels ARE forward-reading (complements the negative
    # differential test, where features must NOT move under future perturbation).
    sample = pd.DataFrame([
        {"asset": "btc", "sample_ts_ms": 1000 * 1000, "window_close_ms": 2000 * 1000}
    ])
    lab_up = short_horizon._attach_labels(sample, up, (10, 15))
    lab_dn = short_horizon._attach_labels(sample, dn, (10, 15))
    assert int(lab_up["fwd_up_10s"].iloc[0]) == 1
    assert int(lab_up["fwd_up_15s"].iloc[0]) == 1
    assert int(lab_dn["fwd_up_10s"].iloc[0]) == 0
    assert int(lab_dn["fwd_up_15s"].iloc[0]) == 0


def test_forward_label_tie_and_na():
    # Ties → Up (>= 0), and an unobserved +horizon second → NA.
    secs = np.arange(0, 20)
    mid = np.full(len(secs), 50_000.0)  # flat ⇒ fwd_ret == 0 ⇒ tie ⇒ Up
    grid = pd.DataFrame({"asset": "btc", "sec": secs, "mid": mid})
    fwd = short_horizon.forward_direction_labels_multi(grid, (10, 15))
    sample = pd.DataFrame([
        {"asset": "btc", "sample_ts_ms": 0, "window_close_ms": 100_000},           # +15 observed
        {"asset": "btc", "sample_ts_ms": 19 * 1000, "window_close_ms": 100_000},   # +10/+15 unobserved
    ])
    out = short_horizon._attach_labels(sample, fwd, (10, 15))
    assert int(out["fwd_up_10s"].iloc[0]) == 1  # flat move is a tie → Up
    assert pd.isna(out["fwd_up_10s"].iloc[1])   # no bar at sec 29 → NA


# --------------------------------------------------------------------------- #
# per-sample coverage marking
# --------------------------------------------------------------------------- #
def test_coverage_marks():
    samples = pd.DataFrame([
        {"label_source": "chainlink", "asset": "btc", "sample_ts_ms": 10_000,
         "up_token": "U", "down_token": "D"},
        {"label_source": "chainlink", "asset": "btc", "sample_ts_ms": 20_000,
         "up_token": "U", "down_token": "D"},
        {"label_source": "binance_proxy", "asset": "btc", "sample_ts_ms": 10_000,
         "up_token": None, "down_token": None},
    ])
    depth_ts = {"btc": np.array([9_500, 30_000], dtype="int64")}  # 9500 near 10000; nothing near 20000
    top_ts = {"D": np.array([21_000], dtype="int64")}             # down-token book near 20000 only
    out = short_horizon._mark_coverage(samples, depth_ts, top_ts, 2_000)

    assert list(out["depth_covered"]) == [True, False, False]  # proxy always False
    assert list(out["book_covered"]) == [False, True, False]   # up OR down token counts


def test_covered_boundaries():
    arr = np.array([100, 200, 300], dtype="int64")
    # exactly at tolerance edge is covered; just past is not.
    got = short_horizon._covered(arr, np.array([98, 103, 205, 250], dtype="int64"), tol_ms=2)
    assert list(got) == [True, False, False, False]
    assert list(short_horizon._covered(np.array([], dtype="int64"),
                                       np.array([1, 2], dtype="int64"), 5)) == [False, False]


# --------------------------------------------------------------------------- #
# no-look-ahead — differential / perturbation (features must not move)
# --------------------------------------------------------------------------- #
def _load_frames(paths: Paths):
    ticks_df, win_meta, res, settle, _top = dataset._read_journal(paths, None)
    windows_df = dataset._windows_frame(win_meta, res, settle)
    strikes = dataset.reconstruct_strikes(ticks_df, windows_df, 180_000, 2500.0)
    windows_df = windows_df.merge(strikes, on=["series", "asset", "window_open_ms"], how="left")
    depth_df = pd.DataFrame(list(depth_io.read_depth_rows(paths.depth_dir)), columns=DEPTH_COLS)
    depth_levels = list(depth_io.read_depth_levels(paths.depth_dir))
    up_tokens = set(windows_df["up_token"].dropna())
    pm_df = short_horizon._read_pm_books(paths, up_tokens)
    token_asset = dict(zip(windows_df["up_token"], windows_df["asset"]))
    return ticks_df, windows_df, depth_df, depth_levels, pm_df, token_asset


def _one_sample(windows_df: pd.DataFrame, idx: int, offset_ms: int) -> tuple[pd.DataFrame, str]:
    w = windows_df.iloc[idx]
    t = int(w.window_open_ms) + offset_ms
    sample = pd.DataFrame([{
        "series": w.series, "asset": w.asset, "up_token": w.up_token,
        "window_open_ms": int(w.window_open_ms), "window_close_ms": int(w.window_close_ms),
        "sample_ts_ms": t, "sample_idx": offset_ms // 1000,
        "strike": float(w.strike), "tau_secs": (int(w.window_close_ms) - t) / 1000.0,
        "elapsed_secs": offset_ms / 1000.0,
    }])
    return sample, w.asset


def _features_for(sample, ticks_df, depth_df, depth_levels, pm_df, token_asset, asset) -> pd.DataFrame:
    # Existing price/vol/depth features use the SAME kernel dataset uses; the new depth
    # + PM microstructure features use short_horizon's own (causal) attach steps.
    grid = dataset.build_feature_grid(ticks_df, depth_df, asset)
    out = dataset._attach_features(sample.copy(), grid)
    out = short_horizon._attach_depth_features(
        out, short_horizon.build_depth_feature_grid(depth_levels, asset), None)
    mid_by_asset = {asset: grid[["sec", "mid"]]} if not grid.empty else {}
    out = short_horizon._attach_pm_features(
        out, short_horizon.build_pm_grid(pm_df, mid_by_asset, token_asset), None)
    return out


def test_no_lookahead_differential(tmp_path):
    # Coverage columns are never in the feature set, so future data can never move a
    # short_horizon feature.
    assert set(short_horizon.FEATURE_COLS).isdisjoint(short_horizon.COVERAGE_COLS)

    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=8)
    paths = _paths(tmp_path)
    ticks_df, windows_df, depth_df, depth_levels, pm_df, token_asset = _load_frames(paths)

    sample, asset = _one_sample(windows_df, idx=4, offset_ms=150_000)
    T = int(sample["sample_ts_ms"].iloc[0])

    full = _features_for(sample, ticks_df, depth_df, depth_levels, pm_df, token_asset, asset)

    # (i) Truncate ALL data strictly after the sample — ticks, depth aggregates, raw
    # depth ladders, and PM books.
    eff = ticks_df["ts_exchange"].fillna(ticks_df["ts_local"]).astype("int64")
    ticks_past = ticks_df[eff <= T].copy()
    depth_past = depth_df[depth_df["recv_ms"].astype("int64") <= T].copy()
    levels_past = [f for f in depth_levels if f["recv_ms"] <= T]
    pm_past = pm_df[pm_df["ts"].astype("int64") <= T].copy()
    trunc = _features_for(sample, ticks_past, depth_past, levels_past, pm_past, token_asset, asset)

    # (ii) Corrupt ALL data strictly after the sample (prices ×10, sizes/imbalance flipped).
    ticks_c = ticks_df.copy()
    ticks_c.loc[eff > T, "value"] *= 10.0
    depth_c = depth_df.copy()
    dfut = depth_c["recv_ms"].astype("int64") > T
    for col in ("mid", "microprice", "spread"):
        depth_c.loc[dfut, col] *= 10.0
    depth_c.loc[dfut, "imbalance"] *= -1.0
    levels_c = []
    for f in depth_levels:
        if f["recv_ms"] > T:
            f = {**f, "bid_px": f["bid_px"] * 10.0, "ask_px": f["ask_px"] * 10.0,
                 "bid_sz": f["ask_sz"].copy(), "ask_sz": f["bid_sz"].copy()}  # flip + scale
        levels_c.append(f)
    pm_c = pm_df.copy()
    pfut = pm_c["ts"].astype("int64") > T
    pm_c.loc[pfut, ["bid_px", "ask_px"]] *= 10.0
    pm_c.loc[pfut, ["bid_sz", "ask_sz"]] = pm_c.loc[pfut, ["ask_sz", "bid_sz"]].to_numpy()
    corrupt = _features_for(sample, ticks_c, depth_c, levels_c, pm_c, token_asset, asset)

    for col in short_horizon.FEATURE_COLS:
        f = full[col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation changed it"
        assert np.array_equal(f, corrupt[col].to_numpy(dtype=float), equal_nan=True), f"{col}: future corruption changed it"

    # Non-vacuous: the sample really does carry live depth + PM microstructure features.
    assert math.isfinite(float(full["mid"].iloc[0]))
    assert math.isfinite(float(full["p_up_model"].iloc[0]))
    assert math.isfinite(float(full["depth_imb_5"].iloc[0]))
    assert math.isfinite(float(full["pm_mid"].iloc[0]))
    assert math.isfinite(float(full["pm_staleness_3s"].iloc[0]))


def _hist_features_for(sample, bars, asset):
    grid = dataset.build_feature_grid(
        dataset._binance_ticks_from_bars(bars, asset), pd.DataFrame(columns=DEPTH_COLS), asset
    )
    out = dataset._attach_features(sample.copy(), grid)
    # Proxy windows have no depth/PM recordings → empty grids → NaN microstructure.
    out = short_horizon._attach_depth_features(out, short_horizon._empty_depth_grid(), None)
    out = short_horizon._attach_pm_features(out, short_horizon._empty_pm_grid(), None)
    return out


def test_no_lookahead_historical(tmp_path):
    make_aggtrades_fixture(tmp_path / "aggtrades")
    sym = "BTCUSDT"
    asset = dataset.SYMBOL_ASSET[sym]
    _yr, _mo, load_paths, mdays = next(dataset._month_batches(tmp_path / "aggtrades", sym))
    bars = dataset._load_month_bars(load_paths, sym)

    day_start = int(datetime(mdays[0].year, mdays[0].month, mdays[0].day, tzinfo=timezone.utc).timestamp())
    open_ms = (day_start + 3600) * 1000
    T = day_start + 3600 + 150
    sample = pd.DataFrame([{
        "series": "BTC-5m", "asset": asset,
        "window_open_ms": open_ms, "window_close_ms": open_ms + 300_000,
        "sample_ts_ms": T * 1000, "sample_idx": 30,
        "strike": 60_000.0, "tau_secs": 150.0, "elapsed_secs": 150.0,
    }])

    full = _hist_features_for(sample, bars, asset)
    trunc = _hist_features_for(sample, bars[bars["sec"] <= T].copy(), asset)
    corrupt = bars.copy()
    corrupt.loc[corrupt["sec"] > T, "price"] *= 10.0
    corrupt = _hist_features_for(sample, corrupt, asset)

    for col in short_horizon.FEATURE_COLS:
        f = full[col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation moved it"
        assert np.array_equal(f, corrupt[col].to_numpy(dtype=float), equal_nan=True), f"{col}: future corruption moved it"
    assert math.isfinite(float(full["mid"].iloc[0]))
    assert math.isfinite(float(full["p_up_model"].iloc[0]))
    # Proxy rows carry no depth/PM recording.
    assert math.isnan(float(full["depth_imb_5"].iloc[0]))
    assert math.isnan(float(full["pm_mid"].iloc[0]))


# --------------------------------------------------------------------------- #
# per-day + coverage report; journal-subset stability; --no-history
# --------------------------------------------------------------------------- #
def test_per_day_and_coverage_report(tmp_path):
    paths = _hist_paths(tmp_path)
    result = short_horizon.short_horizon(paths, grid_secs=5)
    md = json.loads((paths.out_dir / "short_horizon" / "metadata.json").read_text(encoding="utf-8"))

    per_day = md["per_day"]
    assert per_day
    assert sum(r["samples"] for r in per_day) == result["counts"]["samples"]
    assert sum(r["depth_covered"] for r in per_day) == result["counts"]["samples_depth_covered"]
    assert sum(r["book_covered"] for r in per_day) == result["counts"]["samples_book_covered"]
    for r in per_day:
        assert 0.0 <= r["depth_pct"] <= 1.0 and 0.0 <= r["book_pct"] <= 1.0
        # day is a parseable UTC date string.
        datetime.strptime(r["day"], "%Y-%m-%d")

    cov = md["coverage"]
    assert 0.0 <= cov["depth_pct"] <= 1.0 and 0.0 <= cov["book_pct"] <= 1.0
    # Our depth capture covers journal samples; historical-proxy days are 0% coverage.
    assert result["counts"]["samples_depth_covered"] > 0
    assert any(r["depth_covered"] == 0 for r in per_day)  # the proxy days


def test_journal_subset_unchanged(tmp_path):
    # The chainlink subset must be byte-identical with vs without the history extension.
    paths = _hist_paths(tmp_path, n_windows=15)
    pn = Paths(journal_dir=paths.journal_dir, depth_dir=paths.depth_dir,
               out_dir=tmp_path / "out_no", hist_dir=paths.hist_dir)
    rf = short_horizon.short_horizon(paths, grid_secs=5)
    rn = short_horizon.short_horizon(pn, grid_secs=5, include_history=False)
    assert rf["counts"]["windows_binance_proxy"] > 0
    assert rn["counts"]["windows_binance_proxy"] == 0

    full = pd.read_parquet(paths.table("short_horizon"), dtype_backend="pyarrow")
    no = pd.read_parquet(pn.table("short_horizon"), dtype_backend="pyarrow")
    jf = full[full["label_source"] == "chainlink"].reset_index(drop=True)
    pd.testing.assert_frame_equal(jf, no.reset_index(drop=True))
    assert rf["journal_subset"]["windows"] == rn["counts"]["windows"]
    assert rf["journal_subset"]["unchanged"] is True


def test_no_history_smoke(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=10)
    paths = _paths(tmp_path)
    result = short_horizon.short_horizon(paths, grid_secs=5, include_history=False)
    assert result["counts"]["windows_binance_proxy"] == 0
    assert result["counts"]["samples"] > 0
    df = pd.read_parquet(paths.table("short_horizon"), engine="pyarrow")
    assert (df["label_source"] == "chainlink").all()


# --------------------------------------------------------------------------- #
# microstructure features reflect the recordings (positive controls)
# --------------------------------------------------------------------------- #
def test_depth_features_reflect_ladder():
    # A known 3-level book: bids cum sizes 3/5/6, asks cum sizes 1/2/3.
    frame = {
        "recv_ms": 10_000, "asset": "btc",
        "bid_px": np.array([100.0, 99.0, 98.0]), "bid_sz": np.array([3.0, 2.0, 1.0]),
        "ask_px": np.array([101.0, 102.0, 103.0]), "ask_sz": np.array([1.0, 1.0, 1.0]),
    }
    f = short_horizon._frame_depth_features(frame)
    assert abs(f["depth_imb_1"] - 0.5) < 1e-9                 # (3−1)/(3+1)
    assert abs(f["depth_imb_5"] - (3.0 / 9.0)) < 1e-9         # only 3 levels → min(5,3)
    assert abs(f["depth_imb_20"] - (3.0 / 9.0)) < 1e-9
    assert abs(f["microprice_gap"] - 0.25) < 1e-9            # (100·1+101·3)/4 − 100.5
    assert abs(f["bid_depth_slope"] - 3.0) < 1e-9            # cum 6 over (100−98)
    assert abs(f["ask_depth_slope"] - 1.5) < 1e-9            # cum 3 over (103−101)
    assert abs(f["depth_spread_bps"] - 1e4 * 1.0 / 100.5) < 1e-6

    grid = short_horizon.build_depth_feature_grid([frame], "btc")
    assert len(grid) == 1 and int(grid["ts_ms"].iloc[0]) == 10_000
    # A single (top-only) level: slopes need ≥ 2 levels → honest NaN, never imputed.
    thin = {"recv_ms": 1_000, "asset": "btc", "bid_px": np.array([100.0]),
            "bid_sz": np.array([5.0]), "ask_px": np.array([101.0]), "ask_sz": np.array([5.0])}
    tf = short_horizon._frame_depth_features(thin)
    assert math.isnan(tf["bid_depth_slope"]) and math.isnan(tf["ask_depth_slope"])
    assert abs(tf["depth_imb_1"] - 0.0) < 1e-9


def test_pm_features_reflect_book_and_staleness():
    pm_df = pd.DataFrame([
        {"token_id": "U", "ts": 1000, "bid_px": 0.40, "bid_sz": 80.0, "ask_px": 0.44, "ask_sz": 20.0},
        {"token_id": "U", "ts": 2000, "bid_px": 0.50, "bid_sz": 50.0, "ask_px": 0.54, "ask_sz": 50.0},
        {"token_id": "U", "ts": 3000, "bid_px": 0.60, "bid_sz": 10.0, "ask_px": 0.64, "ask_sz": 90.0},
    ])
    mid_by_asset = {"btc": pd.DataFrame({"sec": [1, 2, 3], "mid": [100.0, 110.0, 110.0]})}
    g = short_horizon.build_pm_grid(pm_df, mid_by_asset, {"U": "btc"})

    r2 = g[g["sec"] == 2].iloc[0]
    assert abs(r2["pm_mid"] - 0.52) < 1e-9
    assert abs(r2["pm_spread"] - 0.04) < 1e-9
    assert abs(r2["pm_book_imb"] - 0.0) < 1e-9               # 50/50
    # staleness = Binance log-move − PM mid change over the same 1 s.
    exp = math.log(110.0 / 100.0) - (0.52 - 0.42)
    assert abs(r2["pm_staleness_1s"] - exp) < 1e-9
    r1 = g[g["sec"] == 1].iloc[0]
    assert int(r1["pm_run_secs"]) == 0 and math.isnan(r1["pm_staleness_1s"])  # run-start warmup
    r3 = g[g["sec"] == 3].iloc[0]
    assert abs(r3["pm_book_imb"] - (-0.8)) < 1e-9            # (10−90)/100


def test_microstructure_missing_coverage_is_nan():
    # No bar at all → NaN + NA asof (never imputed).
    sample = pd.DataFrame([{"asset": "btc", "up_token": "U", "sample_ts_ms": 5_000}])
    o = short_horizon._attach_depth_features(sample, short_horizon._empty_depth_grid(), 120_000)
    assert o[short_horizon._DEPTH_FEATURE_NAMES].isna().to_numpy().all()
    assert o["depth_feat_asof_ts_ms"].isna().all()
    p = short_horizon._attach_pm_features(sample, short_horizon._empty_pm_grid(), 120_000)
    assert p[short_horizon._PM_FEATURE_NAMES].isna().to_numpy().all()
    assert p["pm_asof_ts_ms"].isna().all()

    # A bar exists but only AFTER the sample → backward asof finds nothing → NaN.
    frame = {"recv_ms": 9_000, "asset": "btc", "bid_px": np.array([100.0, 99.0]),
             "bid_sz": np.array([2.0, 1.0]), "ask_px": np.array([101.0, 102.0]),
             "ask_sz": np.array([2.0, 1.0])}
    dgrid = short_horizon.build_depth_feature_grid([frame], "btc")
    o2 = short_horizon._attach_depth_features(sample, dgrid, 120_000)
    assert o2["depth_imb_1"].isna().all() and o2["depth_feat_asof_ts_ms"].isna().all()


def test_pm_feat_covered_matches_feature_presence():
    # pm_feat_covered must mean "the PM features are actually present" (backward as-of
    # within staleness), so a training-row filter on it keeps exactly the populated rows.
    pm_df = pd.DataFrame([
        {"token_id": "U", "ts": 1000, "bid_px": 0.40, "bid_sz": 80.0, "ask_px": 0.44, "ask_sz": 20.0},
        {"token_id": "U", "ts": 2000, "bid_px": 0.50, "bid_sz": 50.0, "ask_px": 0.54, "ask_sz": 50.0},
    ])
    pgrid = short_horizon.build_pm_grid(pm_df, {"btc": pd.DataFrame({"sec": [1, 2], "mid": [100.0, 110.0]})},
                                        {"U": "btc"})
    samples = pd.DataFrame([
        {"asset": "btc", "up_token": "U", "sample_ts_ms": 500},    # before any book → absent
        {"asset": "btc", "up_token": "U", "sample_ts_ms": 2500},   # after the 2nd book → present
        {"asset": "btc", "up_token": "U", "sample_ts_ms": 200_000},  # far past → stale → absent
    ])
    out = short_horizon._attach_pm_features(samples, pgrid, 120_000)
    out["pm_feat_covered"] = out["pm_asof_ts_ms"].notna()
    assert list(out["pm_feat_covered"]) == [False, True, False]
    # covered ⟺ the PM book features (mid/spread/imbalance) are present, row by row.
    for col in ("pm_mid", "pm_spread", "pm_book_imb"):
        assert (out["pm_feat_covered"] == out[col].notna()).all(), col


def _write_journal_recs(journal_dir, recs):
    import gzip
    journal_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(journal_dir / "journal-20230101-000000-00000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i, rec in enumerate(recs, 1):
            fh.write(json.dumps({"seq": i, "ts_local_ms": 0, "rec": rec}) + "\n")


def test_read_pm_books(tmp_path):
    recs = [
        {"type": "top_of_book", "token_id": "U",
         "top": {"bid": {"price": "0.40", "size": "80"}, "ask": {"price": "0.44", "size": "20"}, "ts": 1000}},
        {"type": "top_of_book", "token_id": "U",  # one-sided → skipped
         "top": {"bid": {"price": "0.41", "size": "5"}, "ask": None, "ts": 1100}},
        {"type": "top_of_book", "token_id": "X",  # other token → filtered out
         "top": {"bid": {"price": "0.5", "size": "1"}, "ask": {"price": "0.6", "size": "1"}, "ts": 1200}},
    ]
    paths = _paths(tmp_path)
    _write_journal_recs(paths.journal_dir, recs)
    pm = short_horizon._read_pm_books(paths, {"U"})
    assert list(pm["token_id"].unique()) == ["U"]
    assert len(pm) == 1  # one-sided dropped, token X filtered
    row = pm.iloc[0]
    assert row["bid_px"] == 0.40 and row["ask_sz"] == 20.0 and int(row["ts"]) == 1000
    assert short_horizon._read_pm_books(paths, set()).empty  # no tokens → empty


# --------------------------------------------------------------------------- #
# distribution sanity report
# --------------------------------------------------------------------------- #
def _sanity_masks(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_asof_ts_ms": pd.array([1] * n, dtype="Int64"),
        "label_source": ["chainlink"] * n,
        "strike": [100.0] * n,
        "depth_covered": [True] * n,
        "depth_feat_asof_ts_ms": pd.array([1] * n, dtype="Int64"),
        "pm_asof_ts_ms": pd.array([1] * n, dtype="Int64"),
        "pm_run_secs": pd.array([10] * n, dtype="Int64"),
        "sigma_1s": [0.001] * n,
    })


def test_sanity_column_checks():
    n = 1000
    masks = _sanity_masks(n)
    rng = np.random.default_rng(0)
    spec = short_horizon.SanitySpec("depth_imb_5", "float64", "depth_new", None, "partial", True, -1.0, 1.0, "")
    vals = rng.uniform(-0.5, 0.5, n)
    assert short_horizon._column_sanity(spec, vals, masks)["status"] == "PASS"

    bad = vals.copy(); bad[0] = 2.0  # hard-bound violation → FAIL
    assert short_horizon._column_sanity(spec, bad, masks)["status"] == "FAIL"

    nocov = masks.copy()
    nocov["depth_feat_asof_ts_ms"] = pd.array([pd.NA] * n, dtype="Int64")
    allnan = np.full(n, np.nan)  # all-NaN partial-coverage feature → WARN, not FAIL
    assert short_horizon._column_sanity(spec, allnan, nocov)["status"] == "WARN"

    full_spec = short_horizon.SanitySpec("ret", "float64", "price", None, "full", False, -0.05, 0.05, "")
    assert short_horizon._column_sanity(full_spec, np.zeros(n), masks)["status"] == "FAIL"  # constant


def test_sanity_report_over_fixture(tmp_path):
    paths = _hist_paths(tmp_path, n_windows=12)
    short_horizon.short_horizon(paths, grid_secs=5)
    sanity = short_horizon.sanity_report(paths.table("short_horizon"), short_horizon.SANITY_SPEC)
    assert sanity["overall_status"] in ("PASS", "WARN")  # no FAIL on healthy fixture data
    assert sanity["n_features"] == len(short_horizon.FEATURE_COLS)
    assert {c["name"] for c in sanity["columns"]} == set(short_horizon.FEATURE_COLS)
