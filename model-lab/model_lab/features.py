"""Stage 2 — features.

From ``ticks.parquet`` + ``depth.parquet`` build a per-asset, one-second feature
grid → ``features.parquet``:

- price/returns: Binance direct mid, log return, a practical EWMA realized vol;
- basis: ``basis_bps = 1e4·(chainlink/binance − 1)`` (Chainlink LOCF-aligned);
- microstructure (from depth, when present): order-book **imbalance**
  ``(Σbid − Σask)/(Σbid + Σask)``, **microprice**, and **spread**, as-of each
  second.

Verify:  ``python -m model_lab.features``  prints row count and asserts
``imbalance ∈ [−1, 1]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths, resolve_paths, stage_parser

FEATURE_COLS = [
    "asset", "sec", "ts_ms", "mid", "ret", "realized_vol", "chainlink",
    "basis_bps", "depth_mid", "imbalance", "microprice", "spread",
]


def _bars(sel: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Collapses ticks to one-second bars: the last tick (by ms) closes the bar."""
    ts = sel["ts_exchange"].fillna(sel["ts_local"]).astype("int64")
    # Sort by the actual millisecond so groupby-last picks the latest tick in the
    # second (sort_values on `sec` alone is not stable within a second).
    sel = sel.assign(sec=(ts // 1000), _ts=ts).sort_values("_ts")
    bars = sel.groupby("sec", as_index=False)["value"].last().rename(columns={"value": value_name})
    return bars[bars[value_name] > 0].sort_values("sec").reset_index(drop=True)


def _binance_mid_bars(ticks: pd.DataFrame, asset: str) -> pd.DataFrame:
    sel = ticks[
        (ticks["asset"] == asset)
        & (ticks["source"] == "BinanceDirect")
        & (ticks["kind"] == "Mid")
    ].copy()
    if sel.empty:
        return pd.DataFrame(columns=["sec", "mid"])
    return _bars(sel, "mid")


def _chainlink_bars(ticks: pd.DataFrame, asset: str) -> pd.DataFrame:
    sel = ticks[(ticks["asset"] == asset) & (ticks["source"] == "ChainlinkRtds")].copy()
    if sel.empty:
        return pd.DataFrame(columns=["sec", "chainlink"])
    return _bars(sel, "chainlink")


def _depth_bars(depth: pd.DataFrame, asset: str) -> pd.DataFrame:
    if depth.empty:
        return pd.DataFrame(columns=["sec", "depth_mid", "imbalance", "microprice", "spread"])
    sel = depth[depth["asset"] == asset].copy()
    if sel.empty:
        return pd.DataFrame(columns=["sec", "depth_mid", "imbalance", "microprice", "spread"])
    sel = sel.assign(sec=(sel["recv_ms"].astype("int64") // 1000)).sort_values("recv_ms")
    agg = sel.groupby("sec", as_index=False).agg(
        depth_mid=("mid", "last"),
        imbalance=("imbalance", "last"),
        microprice=("microprice", "last"),
        spread=("spread", "last"),
    )
    return agg.sort_values("sec").reset_index(drop=True)


def build_asset_features(
    ticks: pd.DataFrame, depth: pd.DataFrame, asset: str
) -> pd.DataFrame:
    """Assembles the one-second feature grid for one asset."""
    base = _binance_mid_bars(ticks, asset)
    if base.empty:
        return pd.DataFrame(columns=FEATURE_COLS)
    base = base.sort_values("sec").reset_index(drop=True)
    base["ret"] = np.log(base["mid"]).diff()
    # A practical realized vol for research (the EXACT engine EWMA is reproduced
    # in the validate stage and compared to the journaled sigma_1s).
    base["realized_vol"] = base["ret"].ewm(halflife=60, min_periods=10).std()

    cl = _chainlink_bars(ticks, asset)
    if not cl.empty:
        base = pd.merge_asof(base, cl, on="sec", direction="backward")
    else:
        base["chainlink"] = np.nan
    base["basis_bps"] = 1e4 * (base["chainlink"] / base["mid"] - 1.0)

    dep = _depth_bars(depth, asset)
    if not dep.empty:
        base = pd.merge_asof(base, dep, on="sec", direction="backward")
    else:
        for col in ("depth_mid", "imbalance", "microprice", "spread"):
            base[col] = np.nan

    base["asset"] = asset
    base["ts_ms"] = base["sec"] * 1000
    return base[FEATURE_COLS]


def features(paths: Paths) -> int:
    """Runs the features stage; returns the feature-grid row count."""
    paths.ensure_out()
    ticks = pd.read_parquet(paths.table("ticks"), engine="pyarrow")
    depth_path = paths.table("depth")
    depth = pd.read_parquet(depth_path, engine="pyarrow") if depth_path.exists() else pd.DataFrame()

    frames = []
    assets = sorted(ticks["asset"].dropna().unique()) if not ticks.empty else []
    for asset in assets:
        frame = build_asset_features(ticks, depth, asset)
        if not frame.empty:
            frames.append(frame)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FEATURE_COLS)
    out.to_parquet(paths.table("features"), engine="pyarrow", index=False)
    return len(out)


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "features").parse_args(argv)
    paths = resolve_paths(args)
    if not paths.table("ticks").exists():
        print("[features] ticks.parquet missing — run `python -m model_lab.ingest` first.")
        return 1
    n = features(paths)
    print(f"[features] {n:,} rows -> {paths.table('features').name}")
    if n:
        df = pd.read_parquet(paths.table("features"), engine="pyarrow")
        imb = df["imbalance"].dropna()
        if not imb.empty:
            assert imb.between(-1.0, 1.0).all(), "imbalance out of [-1, 1]"
            print(f"[features] depth imbalance ∈ [{imb.min():.3f}, {imb.max():.3f}] over {len(imb):,} secs")
        else:
            print("[features] no depth features (depth capture not present yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
