"""Stage 4 — validate.

Check that the lab reproduces the engine's fair-value math, and score the
model's calibration against realized outcomes:

- **Φ(z) identity** — recompute ``p_up = Φ(z)`` from the journaled ``z`` and
  confirm it matches the journaled ``p_up`` (proves the lab's Φ == the engine's).
- **σ_1s reproduction** — recompute the engine's gap-aware EWMA vol from the raw
  Binance mid ticks and correlate it with the journaled ``sigma_1s``.
- **Calibration** — one mid-window prediction per window vs the realized
  outcome: a reliability curve + Brier score.

Outputs ``out/validation/{metrics.json, reliability.csv}``.
Verify:  ``python -m model_lab.validate``  prints the residuals, correlation, and
Brier score.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .config import Paths, require_tables, resolve_paths, stage_parser
from .lib import math as lm


def _phi_identity(model: pd.DataFrame) -> dict[str, float]:
    z = model["z"].to_numpy(dtype=float)
    z = np.clip(z, -lm.Z_CLAMP, lm.Z_CLAMP)
    recomputed = lm.norm_cdf(z)
    resid = np.abs(recomputed - model["p_up"].to_numpy(dtype=float))
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return {"n": 0, "median_abs": float("nan"), "p95_abs": float("nan"), "max_abs": float("nan")}
    return {
        "n": int(resid.size),
        "median_abs": float(np.median(resid)),
        "p95_abs": float(np.percentile(resid, 95)),
        "max_abs": float(np.max(resid)),
    }


def _sigma_reproduction(model: pd.DataFrame, ticks: pd.DataFrame) -> dict[str, float]:
    pairs_hat: list[float] = []
    pairs_ref: list[float] = []
    for asset in sorted(model["asset"].dropna().unique()):
        mt = ticks[
            (ticks["asset"] == asset)
            & (ticks["source"] == "BinanceDirect")
            & (ticks["kind"] == "Mid")
        ]
        if mt.empty:
            continue
        ts = mt["ts_exchange"].fillna(mt["ts_local"]).astype("int64").to_numpy()
        order = np.argsort(ts, kind="mergesort")
        bar_secs, bar_prices = lm.one_second_bars(ts[order], mt["value"].to_numpy()[order])
        sigma_hat = lm.sigma_1s_from_bars(bar_secs, bar_prices)
        recon = pd.DataFrame({"ts_ms": bar_secs * 1000, "sigma_hat": sigma_hat}).dropna()
        if recon.empty:
            continue
        ms = model[(model["asset"] == asset) & (model["health"] == "Ready")][["ts", "sigma_1s"]].copy()
        ms = ms.dropna().sort_values("ts")
        if ms.empty:
            continue
        merged = pd.merge_asof(
            ms.rename(columns={"ts": "ts_ms"}),
            recon.sort_values("ts_ms"),
            on="ts_ms",
            direction="backward",
        ).dropna(subset=["sigma_hat", "sigma_1s"])
        pairs_ref.extend(merged["sigma_1s"].tolist())
        pairs_hat.extend(merged["sigma_hat"].tolist())

    ref = np.array(pairs_ref, dtype=float)
    hat = np.array(pairs_hat, dtype=float)
    ok = np.isfinite(ref) & np.isfinite(hat) & (ref > 0)
    ref, hat = ref[ok], hat[ok]
    if ref.size < 3:
        return {"n": int(ref.size), "correlation": float("nan"), "median_rel_err": float("nan")}
    corr = float(np.corrcoef(ref, hat)[0, 1]) if np.std(ref) > 0 and np.std(hat) > 0 else float("nan")
    rel_err = float(np.median(np.abs(hat - ref) / ref))
    return {"n": int(ref.size), "correlation": corr, "median_rel_err": rel_err}


def _calibration(model: pd.DataFrame, window_labels: pd.DataFrame) -> dict:
    if model.empty or window_labels.empty:
        return {"n_windows": 0, "brier": float("nan"), "reliability": []}
    joined = model.merge(window_labels, on=["series", "open_time"], how="inner")
    joined = joined.dropna(subset=["p_up", "outcome_up", "close_time"])
    if joined.empty:
        return {"n_windows": 0, "brier": float("nan"), "reliability": []}
    # One mid-window prediction per window: the snapshot nearest (open+close)/2.
    joined["target_ts"] = (joined["open_time"] + joined["close_time"]) / 2.0
    joined["dist"] = (joined["ts"] - joined["target_ts"]).abs()
    picked = joined.sort_values("dist").groupby(["series", "open_time"], as_index=False).first()
    prob = picked["p_up"].to_numpy(dtype=float)
    outcome = picked["outcome_up"].to_numpy(dtype=float)
    brier = lm.brier_score(prob, outcome)
    mids, preds, rates = lm.reliability_curve(prob, outcome, n_bins=10)
    reliability = [
        {"bin_mid": float(m), "mean_pred": float(p), "empirical_rate": float(r)}
        for m, p, r in zip(mids, preds, rates)
    ]
    return {"n_windows": int(len(picked)), "brier": float(brier), "reliability": reliability}


def validate(paths: Paths) -> dict:
    """Runs the validate stage; returns the metrics dict (also written to disk)."""
    require_tables(paths, "model", "ticks")  # must be non-empty
    require_tables(paths, "window_labels", min_rows=0)  # footer-only (may be empty)
    out_dir = paths.out_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = pd.read_parquet(paths.table("model"), engine="pyarrow")
    ticks = pd.read_parquet(paths.table("ticks"), engine="pyarrow")
    window_labels = pd.read_parquet(paths.table("window_labels"), engine="pyarrow")

    metrics = {
        "phi_identity": _phi_identity(model) if not model.empty else {"n": 0},
        "sigma_reproduction": _sigma_reproduction(model, ticks) if not model.empty else {"n": 0},
        "calibration": _calibration(model, window_labels),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rel = pd.DataFrame(metrics["calibration"]["reliability"])
    rel.to_csv(out_dir / "reliability.csv", index=False)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "validate").parse_args(argv)
    paths = resolve_paths(args)
    for t in ("model", "ticks", "window_labels"):
        if not paths.table(t).exists():
            print(f"[validate] {t}.parquet missing — run the earlier stages first.")
            return 1
    m = validate(paths)
    phi = m["phi_identity"]
    sig = m["sigma_reproduction"]
    cal = m["calibration"]
    print(f"[validate] Φ(z) identity: n={phi.get('n', 0)} median|Δ|={phi.get('median_abs', float('nan')):.2e} "
          f"max|Δ|={phi.get('max_abs', float('nan')):.2e}")
    print(f"[validate] σ_1s reproduction: n={sig.get('n', 0)} corr={sig.get('correlation', float('nan')):.3f} "
          f"median_rel_err={sig.get('median_rel_err', float('nan')):.3f}")
    print(f"[validate] calibration: windows={cal.get('n_windows', 0)} Brier={cal.get('brier', float('nan')):.4f}")
    print(f"[validate] wrote {paths.out_dir / 'validation'}")
    # A gross-error gate: the Φ identity must hold (the lab's Φ == the engine's).
    if phi.get("n", 0) > 0 and phi.get("median_abs", 1.0) > 1e-6:
        print("[validate] WARNING: Φ(z) reproduction residual is large — check norm_cdf.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
