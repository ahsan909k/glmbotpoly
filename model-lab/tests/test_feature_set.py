"""Tests for the ``feature_set`` stage — schema, config, the sanity report, the
aggressor-flow sign convention, and the hard no-look-ahead rule.

The no-look-ahead guarantee is exercised the same three ways as the dataset stage:
its grid kernels are appended to ``test_dataset._FEATURE_FUNCS`` (the AST scan),
plus differential/corruption tests here for the journal + aggTrades-proxy paths,
plus the runtime ``feature_set._assert_no_lookahead`` inside every batch.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_lab import dataset, feature_set, features
from model_lab.config import ParquetNotReady, Paths, assert_parquet_ready
from model_lab.fixtures import make_aggtrades_fixture, make_fixture


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
def test_feature_set_smoke_and_schema(tmp_path):
    paths = _hist_paths(tmp_path)
    ds_res = dataset.dataset(paths, grid_secs=15, horizon_secs=30)
    result = feature_set.feature_set(paths)

    # Artifacts.
    assert paths.table("feature_set").exists()
    for fname in ("SCHEMA.md", "sanity.json", "sanity.html", "metadata.json"):
        assert (paths.out_dir / "feature_set" / fname).exists(), f"missing {fname}"

    df = pd.read_parquet(paths.table("feature_set"), engine="pyarrow")

    # Exact schema + 1:1 with the dataset skeleton.
    assert list(df.columns) == feature_set.OUTPUT_COLUMNS
    assert len(df) == ds_res["counts"]["samples"] == result["counts"]["samples"]
    ds_df = pd.read_parquet(paths.table("dataset"), engine="pyarrow")
    keys = ["series", "window_open_ms", "sample_ts_ms"]
    assert (df[keys].sort_values(keys).reset_index(drop=True)
            .equals(ds_df[keys].sort_values(keys).reset_index(drop=True)))

    # Feature bounds.
    fi = df["flow_imb_30s"].dropna()
    assert not fi.empty and fi.between(-1.0, 1.0).all()
    assert df["z"].dropna().abs().le(feature_set.Z_CLAMP).all()
    assert df["hour_of_day"].between(0, 23).all()
    assert (df["seconds_remaining"] > 0).all() and (df["seconds_remaining"] <= 300).all()

    # Flow is aggTrades-only: the fixture's agg days differ from the journal day, so
    # chainlink rows have no flow, while proxy rows (from aggTrades) do.
    chain = df[df["label_source"] == "chainlink"]
    proxy = df[df["label_source"] == "binance_proxy"]
    assert chain["flow_imb_30s"].isna().all()
    assert not chain["flow_covered"].any()
    assert proxy["flow_imb_30s"].notna().any()
    assert proxy["flow_covered"].any()
    assert proxy["ret_60s"].notna().sum() > 0

    # Cross-checks vs the dataset skeleton are ~0 (same rebuild kernels).
    cc = result["cross_checks"]
    assert cc["max_abs_mid_diff"] < 1e-6
    assert cc["max_abs_sigma_slow_diff"] < 1e-9
    assert cc["max_abs_z_diff"] < 1e-6

    # The fixture is well-behaved except a constant trade_intensity (1 trade/sec)
    # which is a partial-coverage WARN, never a FAIL.
    assert result["status"] in ("PASS", "WARN")
    assert result["sanity"]["status_tally"]["FAIL"] == 0


def test_feature_set_determinism(tmp_path):
    base = dict(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", hist_dir=tmp_path / "aggtrades")
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=10)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    a = Paths(out_dir=tmp_path / "out_a", **base)
    b = Paths(out_dir=tmp_path / "out_b", **base)
    for p in (a, b):
        dataset.dataset(p, grid_secs=15, horizon_secs=30)
        feature_set.feature_set(p)
    da = pd.read_parquet(a.table("feature_set"), engine="pyarrow")
    db = pd.read_parquet(b.table("feature_set"), engine="pyarrow")
    pd.testing.assert_frame_equal(da, db)


# --------------------------------------------------------------------------- #
# parquet-ready guard — no stage silently consumes a truncated / partial file
# --------------------------------------------------------------------------- #
def test_assert_parquet_ready_rejects_bad_files(tmp_path):
    # Missing file → clear "not found".
    with pytest.raises(ParquetNotReady, match="not found"):
        assert_parquet_ready(tmp_path / "nope.parquet", label="nope.parquet")

    # Truncated / still-being-written file (no valid footer) → clear message.
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"PAR1 not-a-real-parquet-footer-yet")
    with pytest.raises(ParquetNotReady, match="truncated or still being written"):
        assert_parquet_ready(bad, label="bad.parquet")

    # Valid parquet but fewer rows than required (interrupted write).
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"a": pa.array([], type=pa.int64())}), empty)
    with pytest.raises(ParquetNotReady, match="expected"):
        assert_parquet_ready(empty, min_rows=1)
    assert assert_parquet_ready(empty, min_rows=0) == 0  # footer-only check passes

    # A complete file returns its row count.
    good = tmp_path / "good.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), good)
    assert assert_parquet_ready(good, label="good.parquet") == 3


def test_feature_set_refuses_truncated_dataset(tmp_path):
    # A dataset.parquet that is truncated (footer never written) must fail loudly,
    # not silently build features from a partial skeleton.
    paths = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                  out_dir=tmp_path / "out", hist_dir=tmp_path / "hist_empty")
    paths.ensure_out()
    (paths.out_dir / "dataset.parquet").write_bytes(b"PAR1 partial write, no footer")
    with pytest.raises(ParquetNotReady, match="truncated or still being written"):
        feature_set.feature_set(paths)
    # The CLI turns it into a clean message + nonzero exit (not a traceback).
    assert feature_set.main(["--out", str(paths.out_dir)]) == 1


def test_feature_set_config_expansion():
    # The feature set is exactly the knob-list expansion — proving "one config".
    expected = (
        len(feature_set.RETURN_HORIZONS_SECS)         # ret_*
        + 3                                           # sigma_fast, sigma_slow, vol_ratio
        + 2 * len(feature_set.FLOW_LOOKBACKS_SECS)     # flow_imb_*, trade_intensity_*
        + 3                                           # z, seconds_remaining, hour_of_day
    )
    assert len(feature_set.FEATURE_NAMES) == expected
    # Every spec name is unique and present in the output columns.
    assert len(set(feature_set.FEATURE_NAMES)) == len(feature_set.FEATURE_NAMES)
    assert set(feature_set.FEATURE_NAMES) <= set(feature_set.OUTPUT_COLUMNS)


# --------------------------------------------------------------------------- #
# sanity report — non-vacuous control
# --------------------------------------------------------------------------- #
def _spec(name: str) -> feature_set.FeatureSpec:
    return next(s for s in feature_set.FEATURE_SPEC if s.name == name)


def _masks(n: int, *, mid_fresh=True, flow_covered=True, run=120, strike=60_000.0) -> pd.DataFrame:
    return pd.DataFrame({
        "price_run_secs": np.full(n, run, dtype=float),
        "mid_fresh": np.full(n, mid_fresh, dtype=bool),
        "flow_covered": np.full(n, flow_covered, dtype=bool),
        "strike": np.full(n, strike, dtype=float),
    })


def test_sanity_flags_a_healthy_column_pass():
    vals = np.linspace(-0.01, 0.01, 200)  # varied, in-bounds, no NaN, not constant
    st = feature_set._column_sanity(_spec("ret_5s"), vals, _masks(200))
    assert st["status"] == "PASS"


def test_sanity_flags_constant_full_column_fail():
    vals = np.zeros(200)  # a constant full-coverage feature is a bug
    st = feature_set._column_sanity(_spec("ret_1s"), vals, _masks(200))
    assert st["constant_flag"] == "FAIL"
    assert st["status"] == "FAIL"


def test_sanity_flags_unexpected_nan_fail():
    vals = np.linspace(-0.01, 0.01, 200)
    vals[::2] = np.nan  # 50% NaN where warmup/staleness do NOT explain it
    st = feature_set._column_sanity(_spec("ret_5s"), vals, _masks(200))
    assert st["nan_flag"] == "FAIL"
    assert st["status"] == "FAIL"


def test_sanity_flags_out_of_hard_bound_fail():
    vals = np.linspace(-0.5, 0.5, 200)  # varied
    vals[10] = 5.0  # flow imbalance can never exceed 1 — a correctness bug
    st = feature_set._column_sanity(_spec("flow_imb_30s"), vals, _masks(200))
    assert st["outlier_flag"] == "FAIL"
    assert st["status"] == "FAIL"


def test_sanity_partial_column_no_coverage_warns():
    vals = np.full(200, np.nan)  # a partial feature with zero coverage → WARN, not FAIL
    st = feature_set._column_sanity(_spec("flow_imb_120s"), vals, _masks(200, flow_covered=False))
    assert st["nan_flag"] == "WARN"
    assert st["status"] != "FAIL"


def test_sanity_expected_warmup_nan_passes():
    # ret_60s NaN while the run is shorter than 60 s is EXPECTED, not excess.
    vals = np.full(200, np.nan)
    st = feature_set._column_sanity(_spec("ret_60s"), vals, _masks(200, run=10))
    assert st["excess_nan_fraction"] == 0.0
    # (n_finite == 0 on a full column still FAILs, but the NaN here is not counted as excess)
    assert st["nan_flag"] == "FAIL"  # entirely NaN full column


# --------------------------------------------------------------------------- #
# aggressor-flow sign convention
# --------------------------------------------------------------------------- #
def test_is_buyer_maker_sign(tmp_path):
    # is_buyer_maker == True  ⇒ the aggressor was the SELLER ⇒ signed = −quantity.
    # is_buyer_maker == False ⇒ the aggressor was the BUYER  ⇒ signed = +quantity.
    from model_lab.io import binance_archive as ba
    df = pd.DataFrame({
        "transact_time": np.array([100, 101], dtype="int64") * 1_000_000,
        "quantity": [2.0, 3.0],
        "is_buyer_maker": [False, True],  # buy at sec 100, sell at sec 101
    })
    path = tmp_path / "aggtrades-BTCUSDT-2023-11-12.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)

    bars = feature_set._day_flow_bars(path, "BTCUSDT").set_index("sec")
    assert bars.loc[100, "signed_qty"] == 2.0   # buy aggressor → positive
    assert bars.loc[101, "signed_qty"] == -3.0  # sell aggressor → negative
    assert bars.loc[100, "abs_qty"] == 2.0 and bars.loc[101, "abs_qty"] == 3.0
    assert bars.loc[100, "trade_count"] == 1 and bars.loc[101, "trade_count"] == 1


def test_flow_and_intensity_vary_on_nondegenerate_data():
    # A hand-built, deterministic per-second flow series with a VARIABLE number of
    # trades and a drifting buy/sell mix, so both features genuinely move.
    n = 1000
    sec = np.arange(n, dtype="int64")
    count = 1 + (sec % 4)                            # 1..4 trades/sec
    signed = np.sin(sec / 30.0) * count * 0.5        # drifting net flow
    absq = count * 0.5
    flow_bars = pd.DataFrame({"sec": sec, "signed_qty": signed, "abs_qty": absq, "trade_count": count})
    grid = feature_set.build_flow_grid(flow_bars, "btc")

    imb = grid["flow_imb_30s"].dropna().to_numpy()
    inten = grid["trade_intensity_120s"].dropna().to_numpy()
    assert np.unique(imb).size > 5 and (np.abs(imb) <= 1.0 + 1e-9).all()
    assert np.unique(inten).size > 5 and (inten > 0).all()


# --------------------------------------------------------------------------- #
# no-look-ahead — differential / corruption (journal mid path)
# --------------------------------------------------------------------------- #
_EMPTY_FLOW = pd.DataFrame(columns=["sec", "signed_qty", "abs_qty", "trade_count"])
_EMPTY_PRICE = pd.DataFrame(columns=["sec", "mid"])


def _one_sample(open_ms: int, close_ms: int, sample_ms: int, asset: str, strike: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "series": "BTC-5m" if asset == "btc" else "ETH-5m", "asset": asset,
        "window_open_ms": open_ms, "window_close_ms": close_ms,
        "sample_ts_ms": sample_ms, "sample_idx": (sample_ms - open_ms) // 15_000,
        "strike": strike, "tau_secs": (close_ms - sample_ms) / 1000.0,
        "elapsed_secs": (sample_ms - open_ms) / 1000.0,
    }])


def _fs_price_features(sample, bars, asset):
    grid = feature_set.build_price_grid(bars, asset)
    return feature_set._attach_feature_set(sample.copy(), grid, feature_set.build_flow_grid(_EMPTY_FLOW, asset), None)


_PRICE_FEATURES = [*feature_set._RET_NAMES, "sigma_fast", "sigma_slow", "vol_ratio", "z"]


def test_no_lookahead_feature_set_journal(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=8)
    paths = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                  out_dir=tmp_path / "out", hist_dir=tmp_path / "hist_empty")
    ticks_df, win_meta, res, settle, _top = dataset._read_journal(paths, None)
    windows = dataset._windows_frame(win_meta, res, settle)
    strikes = dataset.reconstruct_strikes(ticks_df, windows, 180_000, 2500.0)
    windows = windows.merge(strikes, on=["series", "asset", "window_open_ms"], how="left")

    w = windows.iloc[4]
    open_ms = int(w.window_open_ms)
    sample = _one_sample(open_ms, int(w.window_close_ms), open_ms + 150_000, w.asset, float(w.strike))
    bars = features._binance_mid_bars(ticks_df, w.asset)[["sec", "mid"]]
    T = (open_ms + 150_000) // 1000

    full = _fs_price_features(sample, bars, w.asset)
    trunc = _fs_price_features(sample, bars[bars["sec"] <= T].copy(), w.asset)
    corrupt = bars.copy()
    corrupt.loc[corrupt["sec"] > T, "mid"] *= 10.0
    corrupt = _fs_price_features(sample, corrupt, w.asset)

    for col in _PRICE_FEATURES:
        f = full[col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation moved it"
        assert np.array_equal(f, corrupt[col].to_numpy(dtype=float), equal_nan=True), f"{col}: future corruption moved it"
    assert math.isfinite(float(full["ret_60s"].iloc[0]))
    assert math.isfinite(float(full["sigma_slow"].iloc[0]))
    assert math.isfinite(float(full["z"].iloc[0]))


# --------------------------------------------------------------------------- #
# no-look-ahead — differential / corruption (aggTrades proxy: mid + flow)
# --------------------------------------------------------------------------- #
def _fs_hist_features(sample, bars, flow_bars, asset):
    grid = feature_set.build_price_grid(bars, asset)
    fgrid = feature_set.build_flow_grid(flow_bars, asset)
    return feature_set._attach_feature_set(sample.copy(), grid, fgrid, None)


def test_no_lookahead_feature_set_historical(tmp_path):
    make_aggtrades_fixture(tmp_path / "aggtrades")
    sym = "BTCUSDT"
    asset = dataset.SYMBOL_ASSET[sym]
    _yr, _mo, load_paths, mdays = next(dataset._month_batches(tmp_path / "aggtrades", sym))
    bars = dataset._load_month_bars(load_paths, sym).rename(columns={"price": "mid"})[["sec", "mid"]]
    flow_bars = feature_set._load_month_flow_bars(load_paths, sym)

    day_start = int(datetime(mdays[0].year, mdays[0].month, mdays[0].day, tzinfo=timezone.utc).timestamp())
    open_ms = (day_start + 3600) * 1000               # a 5m boundary 1h into the block
    T = day_start + 3600 + 150                         # sample second, mid-window
    sample = _one_sample(open_ms, open_ms + 300_000, T * 1000, asset, 60_000.0)

    full = _fs_hist_features(sample, bars, flow_bars, asset)

    trunc = _fs_hist_features(sample, bars[bars["sec"] <= T].copy(), flow_bars[flow_bars["sec"] <= T].copy(), asset)

    bars_c = bars.copy()
    bars_c.loc[bars_c["sec"] > T, "mid"] *= 10.0
    flow_c = flow_bars.copy()
    fut = flow_c["sec"] > T
    flow_c.loc[fut, "signed_qty"] *= -10.0
    flow_c.loc[fut, "abs_qty"] *= 10.0
    flow_c.loc[fut, "trade_count"] *= 5
    corrupt = _fs_hist_features(sample, bars_c, flow_c, asset)

    for col in feature_set.FEATURE_NAMES:
        f = full[col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation moved it"
        assert np.array_equal(f, corrupt[col].to_numpy(dtype=float), equal_nan=True), f"{col}: future corruption moved it"
    assert math.isfinite(float(full["flow_imb_30s"].iloc[0]))
    assert math.isfinite(float(full["trade_intensity_30s"].iloc[0]))
    assert math.isfinite(float(full["ret_60s"].iloc[0]))


def test_no_lookahead_flow_synthetic():
    # Cheap control: a flow grid's rows at sec ≤ T are byte-identical after the
    # future is truncated or corrupted.
    n = 600
    sec = np.arange(n, dtype="int64")
    flow_bars = pd.DataFrame({
        "sec": sec, "signed_qty": np.sin(sec / 20.0), "abs_qty": 1.0 + (sec % 3),
        "trade_count": 1 + (sec % 4),
    })
    T = 400
    full = feature_set.build_flow_grid(flow_bars, "btc")
    trunc = feature_set.build_flow_grid(flow_bars[flow_bars["sec"] <= T].copy(), "btc")
    corr = flow_bars.copy()
    fut = corr["sec"] > T
    corr.loc[fut, ["signed_qty", "abs_qty", "trade_count"]] *= -7.0
    corr = feature_set.build_flow_grid(corr, "btc")

    past = full["sec"] <= T
    for col in [*feature_set._FLOW_IMB_NAMES, *feature_set._INTENSITY_NAMES]:
        f = full.loc[past, col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation moved it"
        assert np.array_equal(f, corr.loc[past, col].to_numpy(dtype=float), equal_nan=True), f"{col}: corruption moved it"
