"""Tests for the ``dataset`` stage — schema + the hard no-look-ahead rule.

Three no-look-ahead guards are exercised:

- ``test_no_lookahead_ast_scan`` — a static scan of the *feature* functions for
  forbidden future-reading idioms (negative shift, non-backward ``merge_asof``,
  any reference to a label column). Includes a negative control so the scanner
  itself is proven non-vacuous.
- ``test_no_lookahead_differential`` — recompute one sample's features from full
  history, from history truncated at the sample, and from history with all future
  values corrupted; assert every feature is bit-identical.
- the runtime assertion ``dataset._assert_no_lookahead`` runs inside every
  ``dataset()`` call (so the smoke/determinism tests exercise it too).
"""

from __future__ import annotations

import ast
import inspect
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from model_lab import dataset, feature_set, features, historical_common, short_horizon
from model_lab.config import Paths
from model_lab.fixtures import make_aggtrades_fixture, make_fixture
from model_lab.ingest import DEPTH_COLS
from model_lab.io import depth as depth_io


def _paths(tmp_path) -> Paths:
    # An isolated (empty) hist_dir keeps these journal-only tests hermetic — they
    # never read the real aggTrades store even after it is downloaded.
    return Paths(
        journal_dir=tmp_path / "journal",
        depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out",
        hist_dir=tmp_path / "hist_empty",
    )


# --------------------------------------------------------------------------- #
# smoke + schema
# --------------------------------------------------------------------------- #
def test_dataset_smoke_and_schema(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=20)
    paths = _paths(tmp_path)
    result = dataset.dataset(paths, grid_secs=15, horizon_secs=30)

    # Outputs exist.
    assert paths.table("dataset").exists()
    assert (paths.out_dir / "dataset" / "SCHEMA.md").exists()
    assert (paths.out_dir / "dataset" / "metadata.json").exists()

    df = pd.read_parquet(paths.table("dataset"), engine="pyarrow")
    assert len(df) == result["counts"]["samples"] > 0

    # Exact schema: column set + order match the single source of truth.
    assert list(df.columns) == dataset.COLUMNS
    assert not (set(dataset.FEATURE_COLS) & set(dataset.LABEL_COLS))

    # Window framing invariants.
    assert (df["window_open_ms"] <= df["sample_ts_ms"]).all()
    assert (df["sample_ts_ms"] < df["window_close_ms"]).all()
    assert (df["tau_secs"] > 0).all()
    assert (df["elapsed_secs"] >= 0).all()

    # The as-of feature bar never comes from the future.
    fa = df["feature_asof_ts_ms"].dropna()
    assert (fa.astype("int64") <= df.loc[fa.index, "sample_ts_ms"].astype("int64")).all()

    # Labels are well-formed.
    assert set(df["outcome"].unique()) <= {"Up", "Down"}
    assert df["outcome_up"].isin([0, 1]).all()
    assert df["fwd_up_30s"].dropna().isin([0, 1]).all()

    # Grid cadence: 300s window / 15s → 20 samples per window; indices contiguous.
    per_window = df.groupby(["series", "window_open_ms"]).size()
    assert (per_window == 20).all()

    # The coverage flag is a proper bool and the fixture recorded market mids.
    assert df["in_live_coverage"].dtype == bool
    assert result["counts"]["samples_in_coverage"] > 0
    assert result["counts"]["samples_in_coverage"] <= result["counts"]["samples"]


def test_dataset_determinism(tmp_path):
    # Two runs (journal + historical proxy) must be bit-identical.
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=10)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    base = dict(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth", hist_dir=tmp_path / "aggtrades")
    a = Paths(out_dir=tmp_path / "out_a", **base)
    b = Paths(out_dir=tmp_path / "out_b", **base)
    dataset.dataset(a, grid_secs=15, horizon_secs=30)
    dataset.dataset(b, grid_secs=15, horizon_secs=30)
    da = pd.read_parquet(a.table("dataset"), engine="pyarrow")
    db = pd.read_parquet(b.table("dataset"), engine="pyarrow")
    pd.testing.assert_frame_equal(da, db)


def test_strike_reconstruction(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=8)
    paths = _paths(tmp_path)
    dataset.dataset(paths, grid_secs=15, horizon_secs=30)
    df = pd.read_parquet(paths.table("dataset"), engine="pyarrow")

    # The fixture emits a Chainlink Vendor tick at exactly each window open, so the
    # strike is a clean boundary capture at or before open.
    assert (df["strike_quality"] == "boundary").all()
    st = df["strike_ts_ms"].dropna()
    assert (st.astype("int64") <= df.loc[st.index, "window_open_ms"].astype("int64")).all()
    assert (df["strike"] > 0).all()

    # At k=0 (elapsed 0) the mid ≈ strike (near-the-money at open): |log(S/K)| small.
    at_open = df[df["elapsed_secs"] == 0.0]
    assert not at_open.empty
    assert at_open["log_s_k"].abs().max() < 0.01


def test_coverage_flag_true_and_false():
    # Two windows sharing an asset: one has a top_of_book in-window, one does not.
    windows = pd.DataFrame(
        [
            {"series": "BTC-5m", "asset": "btc", "window_open_ms": 1000, "window_close_ms": 4000,
             "up_token": "U1", "down_token": "D1"},
            {"series": "BTC-5m", "asset": "btc", "window_open_ms": 4000, "window_close_ms": 7000,
             "up_token": "U2", "down_token": "D2"},
        ]
    )
    top_ts_by_token = {"U1": [1500, 2500], "D2": [999999]}  # U1 in window 0; D2 top is far away
    flags = dataset._coverage_flags(windows, top_ts_by_token)
    assert flags == [True, False]


# --------------------------------------------------------------------------- #
# no-look-ahead — (b) AST source scan
# --------------------------------------------------------------------------- #
# The feature functions: every one must read only data at-or-before the sample.
_FEATURE_FUNCS = [
    dataset.build_feature_grid,
    dataset._attach_features,
    dataset._derive_asof_features,
    features.build_asset_features,
    features._binance_mid_bars,
    features._chainlink_bars,
    features._depth_bars,
    features._bars,
    # The feature_set stage's per-second grid kernels — same no-look-ahead rule.
    feature_set.build_price_grid,
    feature_set.mid_return_grid,
    feature_set.ewma_vol_grid,
    feature_set._run_lengths,
    feature_set.build_flow_grid,
    feature_set._flow_rolls,
    feature_set._attach_feature_set,
    feature_set._derive_scalar_features,
    # The short_horizon stage's microstructure (depth + PM book) feature kernels.
    short_horizon._side_slope,
    short_horizon._frame_depth_features,
    short_horizon.build_depth_feature_grid,
    short_horizon.build_pm_grid,
    short_horizon._attach_asof,
    # The historical (Telonex) reconstruction kernels — same no-look-ahead rule.
    historical_common.telonex_depth_feature_grid,
    historical_common.reconstruct_window_snapshots,
]

# Names that would betray a forward read if they appeared in a feature function.
# Includes the short_horizon stage's labels — the shared feature funcs (scanned below)
# must never reference them either.
_FORBIDDEN_LABELS = set(dataset.LABEL_COLS) | set(short_horizon.LABEL_COLS) | {
    "fwd_mid", "fwd_ret_30s", "fwd_up_30s",
    "fwd_ret_10s", "fwd_up_10s", "fwd_ret_15s", "fwd_up_15s",
    "outcome", "outcome_up", "forward",
}


def _scan_for_lookahead(src: str, where: str) -> list[str]:
    """Return a list of look-ahead problems found in a feature-function source."""
    problems: list[str] = []
    tree = ast.parse(src)

    def is_negative(node) -> bool:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return True  # -k or -var: a backward-in-time offset on a future read
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value < 0
        return False

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            # Negative shift/diff reaches into the future.
            if name in ("shift", "tshift", "diff"):
                arg = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg == "periods"), None
                )
                if arg is not None and is_negative(arg):
                    problems.append(f"{where}: negative {name}() reads the future")
            # merge_asof must be explicitly backward.
            if name == "merge_asof":
                direction = next((k.value for k in node.keywords if k.arg == "direction"), None)
                if direction is None:
                    problems.append(f"{where}: merge_asof without an explicit direction")
                elif not (isinstance(direction, ast.Constant) and direction.value == "backward"):
                    val = getattr(direction, "value", "<expr>")
                    problems.append(f"{where}: merge_asof direction={val!r} (must be 'backward')")
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in _FORBIDDEN_LABELS:
                problems.append(f"{where}: references label name {node.id!r}")

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str) and node.value in _FORBIDDEN_LABELS:
                problems.append(f"{where}: uses label column {node.value!r}")

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in _FORBIDDEN_LABELS:
                problems.append(f"{where}: attribute {node.attr!r} is a label name")
            self.generic_visit(node)

    Visitor().visit(tree)
    return problems


def test_no_lookahead_ast_scan():
    problems: list[str] = []
    for fn in _FEATURE_FUNCS:
        src = inspect.getsource(fn)
        problems += _scan_for_lookahead(src, fn.__qualname__)
    assert not problems, "look-ahead found in feature code:\n" + "\n".join(problems)


def test_ast_scanner_catches_leaks():
    # Negative control: the scanner must flag every leak, or it is worthless.
    leaky = (
        "def bad(df):\n"
        "    a = df['mid'].shift(-30)\n"
        "    b = pd.merge_asof(x, y, on='ts')\n"
        "    c = pd.merge_asof(x, y, on='ts', direction='forward')\n"
        "    d = df['outcome_up']\n"
        "    return a, b, c, d\n"
    )
    problems = _scan_for_lookahead(leaky, "bad")
    joined = "\n".join(problems)
    assert "negative shift" in joined
    assert "without an explicit direction" in joined
    assert "direction='forward'" in joined
    assert "outcome_up" in joined


# --------------------------------------------------------------------------- #
# no-look-ahead — (c) differential / perturbation
# --------------------------------------------------------------------------- #
def _load_frames(paths: Paths):
    ticks_df, win_meta, res, settle, _top = dataset._read_journal(paths, None)
    windows_df = dataset._windows_frame(win_meta, res, settle)
    strikes = dataset.reconstruct_strikes(ticks_df, windows_df, 180_000, 2500.0)
    windows_df = windows_df.merge(strikes, on=["series", "asset", "window_open_ms"], how="left")
    depth_df = pd.DataFrame(list(depth_io.read_depth_rows(paths.depth_dir)), columns=DEPTH_COLS)
    return ticks_df, windows_df, depth_df


def _one_sample(windows_df: pd.DataFrame, idx: int, offset_ms: int) -> tuple[pd.DataFrame, str]:
    w = windows_df.iloc[idx]
    t = int(w.window_open_ms) + offset_ms
    sample = pd.DataFrame(
        [{
            "series": w.series, "asset": w.asset,
            "window_open_ms": int(w.window_open_ms), "window_close_ms": int(w.window_close_ms),
            "sample_ts_ms": t, "sample_idx": offset_ms // 1000,
            "strike": float(w.strike), "tau_secs": (int(w.window_close_ms) - t) / 1000.0,
            "elapsed_secs": offset_ms / 1000.0,
        }]
    )
    return sample, w.asset


def _features_for(sample: pd.DataFrame, ticks_df, depth_df, asset) -> pd.DataFrame:
    grid = dataset.build_feature_grid(ticks_df, depth_df, asset)
    return dataset._attach_features(sample.copy(), grid)


def test_no_lookahead_differential(tmp_path):
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=8)
    paths = _paths(tmp_path)
    ticks_df, windows_df, depth_df = _load_frames(paths)

    # A mid-window sample on an emitted-second boundary (bars are stable there).
    sample, asset = _one_sample(windows_df, idx=4, offset_ms=150_000)
    T = int(sample["sample_ts_ms"].iloc[0])

    full = _features_for(sample, ticks_df, depth_df, asset)

    # (i) Truncate all data strictly after the sample.
    eff = ticks_df["ts_exchange"].fillna(ticks_df["ts_local"]).astype("int64")
    ticks_past = ticks_df[eff <= T].copy()
    depth_past = depth_df[depth_df["recv_ms"].astype("int64") <= T].copy()
    trunc = _features_for(sample, ticks_past, depth_past, asset)

    # (ii) Corrupt all data strictly after the sample (prices ×10, imbalance flipped).
    # NOTE: the fixture *generates* depth imbalance from a forward return, but we
    # read it as-of — corrupting the recorded future rows must not move a past bar.
    ticks_c = ticks_df.copy()
    ticks_c.loc[eff > T, "value"] *= 10.0
    depth_c = depth_df.copy()
    dfut = depth_c["recv_ms"].astype("int64") > T
    for col in ("mid", "microprice", "spread"):
        depth_c.loc[dfut, col] *= 10.0
    depth_c.loc[dfut, "imbalance"] *= -1.0
    corrupt = _features_for(sample, ticks_c, depth_c, asset)

    for col in dataset.FEATURE_COLS:
        f = full[col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation changed it"
        assert np.array_equal(f, corrupt[col].to_numpy(dtype=float), equal_nan=True), f"{col}: future corruption changed it"

    # Sanity: at least some features are real numbers (not all-NaN → vacuous test).
    assert math.isfinite(float(full["mid"].iloc[0]))
    assert math.isfinite(float(full["p_up_model"].iloc[0]))


# --------------------------------------------------------------------------- #
# historical (Binance aggTrades proxy) extension
# --------------------------------------------------------------------------- #
def _hist_paths(tmp_path, n_windows: int = 20) -> Paths:
    make_fixture(tmp_path / "journal", tmp_path / "depth", n_windows=n_windows)
    make_aggtrades_fixture(tmp_path / "aggtrades")
    return Paths(
        journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
    )


def test_historical_proxy_extension(tmp_path):
    paths = _hist_paths(tmp_path)
    result = dataset.dataset(paths, grid_secs=15, horizon_secs=30)
    df = pd.read_parquet(paths.table("dataset"), engine="pyarrow")

    # Both sources present; both series (BTC journal + BTC/ETH historical).
    assert set(df["label_source"].unique()) == {"chainlink", "binance_proxy"}
    assert result["counts"]["windows_chainlink"] > 0
    assert result["counts"]["windows_binance_proxy"] > 0
    assert {"BTC-5m", "ETH-5m"} <= set(df["series"].unique())
    assert {"btc", "eth"} <= set(df["asset"].unique())

    proxy = df[df["label_source"] == "binance_proxy"]
    chain = df[df["label_source"] == "chainlink"]

    # Proxy rows: price-only features present; Chainlink/depth features all NaN;
    # never in live coverage; well-formed outcome + primary forward label.
    assert proxy["mid"].notna().any()
    assert proxy["p_up_model"].notna().any()
    assert proxy["chainlink"].isna().all()
    assert proxy["basis_bps"].isna().all()
    assert proxy["imbalance"].isna().all()
    assert (~proxy["in_live_coverage"].astype(bool)).all()
    assert set(proxy["outcome"].dropna().unique()) <= {"Up", "Down"}
    assert proxy["outcome_up"].isin([0, 1]).all()
    assert proxy["fwd_up_30s"].dropna().isin([0, 1]).all()

    # Knife-edge diagnostics: non-negative absolute distance where defined.
    d = proxy["strike_distance_at_close"].dropna()
    assert not d.empty and (d >= 0).all()

    # Journal rows keep a real market-mid benchmark somewhere; proxy never does.
    assert chain["in_live_coverage"].astype(bool).any()

    # Grid cadence still 20 samples/window for every window, both sources.
    assert (df.groupby(["series", "window_open_ms"]).size() == 20).all()


def test_journal_subset_unchanged(tmp_path):
    # The chainlink subset must be byte-identical with vs without the extension.
    paths = _hist_paths(tmp_path, n_windows=15)
    pn = Paths(journal_dir=paths.journal_dir, depth_dir=paths.depth_dir,
               out_dir=tmp_path / "out_no", hist_dir=paths.hist_dir)
    rf = dataset.dataset(paths)
    rn = dataset.dataset(pn, include_history=False)
    assert rf["counts"]["windows_binance_proxy"] > 0
    assert rn["counts"]["windows_binance_proxy"] == 0

    full = pd.read_parquet(paths.table("dataset"), dtype_backend="pyarrow")
    no = pd.read_parquet(pn.table("dataset"), dtype_backend="pyarrow")
    jf = full[full["label_source"] == "chainlink"].reset_index(drop=True)
    pd.testing.assert_frame_equal(jf, no.reset_index(drop=True))
    assert rf["journal_subset"]["windows"] == rn["counts"]["windows"]
    assert rf["journal_subset"]["unchanged"] is True


def test_dedup_journal_wins():
    # A historical window whose (series, open) collides with a journal window is
    # dropped (journal keeps its real labels) and counted as a collision.
    d = date(2023, 11, 12)
    opens = dataset._day_open_ms(d)
    target = int(opens[5])  # a 5m boundary with bar coverage in the block below
    day_start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    secs = np.arange(day_start, day_start + 2 * 3600, dtype="int64")
    prices = 60_000.0 + np.arange(len(secs)) * 0.01
    bars = pd.DataFrame({"sec": secs, "price": prices})

    hw, collisions = dataset._historical_windows(
        bars, [d], "BTCUSDT", {("BTC-5m", target)}, 2500.0
    )
    assert collisions >= 1
    assert target not in set(hw["window_open_ms"].tolist())
    assert int(opens[6]) in set(hw["window_open_ms"].tolist())  # a nearby one survives
    assert (hw["label_source"] == "binance_proxy").all()


def _hist_features_for(sample, bars, asset):
    grid = dataset.build_feature_grid(
        dataset._binance_ticks_from_bars(bars, asset), pd.DataFrame(columns=DEPTH_COLS), asset
    )
    return dataset._attach_features(sample.copy(), grid)


def test_no_lookahead_historical(tmp_path):
    # The Binance-proxy feature path is causal: corrupting/removing future bars must
    # not move a past sample's features (it goes through the same feature kernel).
    make_aggtrades_fixture(tmp_path / "aggtrades")
    sym = "BTCUSDT"
    asset = dataset.SYMBOL_ASSET[sym]
    _yr, _mo, load_paths, mdays = next(dataset._month_batches(tmp_path / "aggtrades", sym))
    bars = dataset._load_month_bars(load_paths, sym)

    day_start = int(datetime(mdays[0].year, mdays[0].month, mdays[0].day, tzinfo=timezone.utc).timestamp())
    open_ms = (day_start + 3600) * 1000            # a 5m boundary 1h into the block
    T = day_start + 3600 + 150                     # sample second, mid-window
    sample = pd.DataFrame([{
        "series": "BTC-5m", "asset": asset,
        "window_open_ms": open_ms, "window_close_ms": open_ms + 300_000,
        "sample_ts_ms": T * 1000, "sample_idx": 10,
        "strike": 60_000.0, "tau_secs": 150.0, "elapsed_secs": 150.0,
    }])

    full = _hist_features_for(sample, bars, asset)
    trunc = _hist_features_for(sample, bars[bars["sec"] <= T].copy(), asset)
    corrupt = bars.copy()
    corrupt.loc[corrupt["sec"] > T, "price"] *= 10.0
    corrupt = _hist_features_for(sample, corrupt, asset)

    for col in dataset.FEATURE_COLS:
        f = full[col].to_numpy(dtype=float)
        assert np.array_equal(f, trunc[col].to_numpy(dtype=float), equal_nan=True), f"{col}: truncation moved it"
        assert np.array_equal(f, corrupt[col].to_numpy(dtype=float), equal_nan=True), f"{col}: future corruption moved it"
    assert math.isfinite(float(full["mid"].iloc[0]))
    assert math.isfinite(float(full["p_up_model"].iloc[0]))


def test_history_report(tmp_path):
    paths = _hist_paths(tmp_path)
    result = dataset.dataset(paths)
    h = result["history"]
    assert h["included"] is True
    assert set(h["symbols_present"]) == {"BTCUSDT", "ETHUSDT"}
    assert len(h["per_symbol_month"]) >= 2  # at least one month for BTC and ETH

    # Per-symbol-month counts reconcile with the proxy totals.
    assert sum(r["windows"] for r in h["per_symbol_month"]) == result["counts"]["windows_binance_proxy"]
    assert sum(r["samples"] for r in h["per_symbol_month"]) == result["counts"]["samples_binance_proxy"]

    # Knife-edge proportions: within [0,1] and non-decreasing in threshold.
    ke = h["knife_edge"]
    assert ke["proxy_windows_scored"] > 0
    vals = [ke["proportion_within"][f"{t:g}"] for t in dataset.KNIFE_EDGE_VOL_THRESHOLDS]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_no_history_flag(tmp_path):
    paths = _hist_paths(tmp_path)
    result = dataset.dataset(paths, include_history=False)
    assert result["counts"]["windows_binance_proxy"] == 0
    assert result["history"]["per_symbol_month"] == []
    df = pd.read_parquet(paths.table("dataset"), engine="pyarrow")
    assert (df["label_source"] == "chainlink").all()
