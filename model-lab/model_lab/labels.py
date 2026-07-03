"""Stage 3 — labels.

Build the forward-looking targets the research + validation stages score against:

- ``labels.parquet`` — per-asset, per-second **forward** returns of the Binance
  mid (``fwd_ret_5s`` and its sign ``fwd_up_5s``), the microstructure target.
- ``window_labels.parquet`` — per-window realized outcome (``outcome_up`` ∈
  {0,1}) from settlements, joined to the window close time, for calibration.

Verify:  ``python -m model_lab.labels``  prints label counts + class balance and
asserts no lookahead (the forward label reads strictly later ticks).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .config import Paths, resolve_paths, stage_parser

HORIZON_SECS = 5
LABEL_COLS = ["asset", "sec", "ts_ms", "fwd_ret_5s", "fwd_up_5s"]
WINDOW_LABEL_COLS = ["series", "open_time", "close_time", "outcome", "outcome_up"]


def _forward_labels(features: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    sel = features[features["asset"] == asset][["sec", "mid"]].dropna().copy()
    if sel.empty:
        return pd.DataFrame(columns=LABEL_COLS)
    sel = sel.sort_values("sec").drop_duplicates("sec")
    lo, hi = int(sel["sec"].min()), int(sel["sec"].max())
    # Reindex to a gap-free second grid so "t + horizon" is a clean shift, then
    # forward-fill the price for the seconds that had no tick.
    grid = sel.set_index("sec").reindex(range(lo, hi + 1))
    grid["mid"] = grid["mid"].ffill()
    grid["fwd_mid"] = grid["mid"].shift(-horizon)
    grid["fwd_ret_5s"] = np.log(grid["fwd_mid"] / grid["mid"])
    grid = grid.reset_index().rename(columns={"index": "sec"})
    # Keep only the seconds that actually had an observation.
    out = grid[grid["sec"].isin(sel["sec"])].dropna(subset=["fwd_ret_5s"]).copy()
    out["fwd_up_5s"] = (out["fwd_ret_5s"] > 0).astype(int)
    out["asset"] = asset
    out["ts_ms"] = out["sec"] * 1000
    return out[LABEL_COLS]


def labels(paths: Paths) -> tuple[int, int]:
    """Runs the labels stage; returns ``(forward_label_rows, window_labels)``."""
    paths.ensure_out()
    features = pd.read_parquet(paths.table("features"), engine="pyarrow")

    frames = []
    for asset in sorted(features["asset"].dropna().unique()) if not features.empty else []:
        frame = _forward_labels(features, asset, HORIZON_SECS)
        if not frame.empty:
            frames.append(frame)
    fwd = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLS)
    fwd.to_parquet(paths.table("labels"), engine="pyarrow", index=False)

    settlements = pd.read_parquet(paths.table("settlements"), engine="pyarrow")
    windows = pd.read_parquet(paths.table("windows"), engine="pyarrow")
    if settlements.empty:
        win = pd.DataFrame(columns=WINDOW_LABEL_COLS)
    else:
        win = settlements[["series", "open_time", "outcome"]].copy()
        win["outcome_up"] = (win["outcome"] == "Up").astype(int)
        if not windows.empty:
            win = win.merge(
                windows[["series", "open_time", "close_time"]],
                on=["series", "open_time"],
                how="left",
            )
        else:
            win["close_time"] = np.nan
        win = win[WINDOW_LABEL_COLS]
    win.to_parquet(paths.table("window_labels"), engine="pyarrow", index=False)

    return len(fwd), len(win)


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "labels").parse_args(argv)
    paths = resolve_paths(args)
    if not paths.table("features").exists():
        print("[labels] features.parquet missing — run `python -m model_lab.features` first.")
        return 1
    n_fwd, n_win = labels(paths)
    print(f"[labels] {n_fwd:,} forward-return labels -> {paths.table('labels').name}")
    print(f"[labels] {n_win:,} window outcomes       -> {paths.table('window_labels').name}")
    if n_win:
        win = pd.read_parquet(paths.table("window_labels"), engine="pyarrow")
        rate = float(win["outcome_up"].mean())
        print(f"[labels] window Up-rate = {rate:.3f} (n={n_win})")
    if n_fwd:
        fwd = pd.read_parquet(paths.table("labels"), engine="pyarrow")
        print(f"[labels] forward 5s Up-rate = {float(fwd['fwd_up_5s'].mean()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
