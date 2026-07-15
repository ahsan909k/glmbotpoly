"""Stage 5 — research (the reason the depth capture exists).

Does Binance L2 order-book microstructure predict short-horizon moves beyond
what the engine already sees? Aligns the depth features to the forward 5-second
Binance-mid return and reports, per asset and pooled:

- **IC** — Pearson correlation of order-book **imbalance** (and the
  **microprice tilt** ``microprice − mid``) with ``fwd_ret_5s``;
- **AUC** — imbalance predicting the forward up-move, against a momentum baseline
  (the current 1-second return) and the 0.5 no-skill line.

A positive IC / AUC > 0.5 is evidence the depth signal is worth feeding into the
fair value. Outputs ``out/research/metrics.json``. If no depth was captured yet,
it says so and exits cleanly.

Verify:  ``python -m model_lab.research``  prints the IC/AUC table.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .config import Paths, require_tables, resolve_paths, stage_parser
from .lib import math as lm


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _asset_metrics(df: pd.DataFrame) -> dict:
    imbalance = df["imbalance"].to_numpy(dtype=float)
    tilt = (df["microprice"] - df["mid"]).to_numpy(dtype=float)
    fwd = df["fwd_ret_5s"].to_numpy(dtype=float)
    up = df["fwd_up_5s"].to_numpy(dtype=float)
    mom = df["ret"].to_numpy(dtype=float)
    n_depth = int(np.isfinite(imbalance).sum())
    return {
        "n": int(len(df)),
        "n_with_depth": n_depth,
        "ic_imbalance": _ic(imbalance, fwd),
        "ic_microprice_tilt": _ic(tilt, fwd),
        "auc_imbalance": lm.auc(imbalance[np.isfinite(imbalance)], up[np.isfinite(imbalance)])
        if n_depth > 0
        else float("nan"),
        "auc_momentum_baseline": lm.auc(mom[np.isfinite(mom)], up[np.isfinite(mom)]),
    }


def research(paths: Paths) -> dict:
    """Runs the research stage; returns the metrics dict (also written to disk)."""
    require_tables(paths, "features", "labels")  # fail loudly on truncated/absent inputs
    out_dir = paths.out_dir / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(paths.table("features"), engine="pyarrow")
    lbls = pd.read_parquet(paths.table("labels"), engine="pyarrow")

    metrics: dict = {"per_asset": {}, "pooled": {}, "has_depth": False}
    if features.empty or lbls.empty:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    df = features.merge(lbls[["asset", "sec", "fwd_ret_5s", "fwd_up_5s"]], on=["asset", "sec"], how="inner")
    metrics["has_depth"] = bool(df["imbalance"].notna().any())

    for asset in sorted(df["asset"].dropna().unique()):
        metrics["per_asset"][asset] = _asset_metrics(df[df["asset"] == asset])
    metrics["pooled"] = _asset_metrics(df)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _fmt(x: float) -> str:
    return "  n/a" if not np.isfinite(x) else f"{x:+.3f}"


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "research").parse_args(argv)
    paths = resolve_paths(args)
    for t in ("features", "labels"):
        if not paths.table(t).exists():
            print(f"[research] {t}.parquet missing — run the earlier stages first.")
            return 1
    m = research(paths)
    if not m["has_depth"]:
        print("[research] no depth20 data present — capture it with `bot record`/`bot run`, "
              "then re-run. (Momentum baseline still reported below.)")
    pooled = m.get("pooled", {})
    print(f"[research] pooled: n={pooled.get('n', 0):,} with_depth={pooled.get('n_with_depth', 0):,}")
    print(f"[research]   IC(imbalance→fwd_ret_5s)      = {_fmt(pooled.get('ic_imbalance', float('nan')))}")
    print(f"[research]   IC(microprice_tilt→fwd_ret_5s)= {_fmt(pooled.get('ic_microprice_tilt', float('nan')))}")
    print(f"[research]   AUC(imbalance→up)             = {_fmt(pooled.get('auc_imbalance', float('nan')))}")
    print(f"[research]   AUC(momentum→up) [baseline]   = {_fmt(pooled.get('auc_momentum_baseline', float('nan')))}")
    print(f"[research] wrote {paths.out_dir / 'research'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
