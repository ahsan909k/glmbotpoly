"""Live accuracy of the shadow model's predictions.

The champion ``dir10`` predicts ``p_up`` = P(the Binance mid moves **Up** over the
next 10 s) — the ``fwd_up_10s`` label. This scores that live p_up against the
**realized** 10 s move, measured from shadow's own journaled ``log_s_k`` feature
(= ``log(mid/strike)``; the strike is constant within a window, so the 10 s change
in ``log_s_k`` *is* the 10 s mid log-return). Self-contained from
``data/shadow/*.jsonl.gz`` — no journal read. Reports directional accuracy /
Brier / log-loss / a calibration table vs the naive baselines, overall + per
series.

Caveat: shadow samples every 5 s, so the 10 s-forward outcome is the sample two
steps ahead in the *same* window; shadow's mid is the last-completed-bar close
(~1 s streaming lag). A faithful live proxy for the exact 1 s-grid label — the
authoritative offline score is the evaluation harness over the recorded journal.

Run:  python -m model_lab.shadow_accuracy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_paths, stage_parser
from .shadow_parity import _default_shadow_dir, read_shadow_predictions

HORIZON_MS = 10_000
_EPS = 1e-15


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Directional accuracy / Brier / log-loss + baselines for probs ``p`` vs
    binary outcomes ``y``, PLUS the two-sided confusion metrics that reveal
    whether the model calls **down** as well as up (a model that only ever says
    "up" gets 100% up-recall, 0% down-recall, and 50% balanced accuracy — so
    balanced accuracy is the base-rate-proof skill number)."""
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    base = float(np.mean(y))
    pc = np.clip(p, _EPS, 1 - _EPS)
    pred_up = p >= 0.5
    up = y >= 0.5
    n_up = int(up.sum())
    n_down = int((~up).sum())
    up_recall = float(np.mean(pred_up[up])) if n_up else float("nan")
    down_recall = float(np.mean(~pred_up[~up])) if n_down else float("nan")
    # of the times the model actually SAID down (p < 0.5), how often was it right
    said_down = ~pred_up
    down_precision = float(np.mean(~up[said_down])) if said_down.sum() else float("nan")
    return {
        "n": n,
        "dir_acc": float(np.mean(pred_up == up)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "base_rate_up": base,
        "brier_vs_half": 0.25,  # always predict 0.5
        "brier_vs_baserate": float(base * (1 - base)),  # always predict the base rate
        "p_up_mean": float(np.mean(p)),
        # --- two-sided skill (can it call DOWN, not just up?) ---
        "n_up_moves": n_up,
        "n_down_moves": n_down,
        "up_recall": up_recall,          # of actual UP moves, fraction called up
        "down_recall": down_recall,      # of actual DOWN moves, fraction called down
        "down_precision": down_precision,  # when it said down, fraction right
        "pred_down_frac": float(np.mean(said_down)),  # how often it even leans down
        "balanced_acc": (up_recall + down_recall) / 2.0
        if (n_up and n_down)
        else float("nan"),
    }


def _reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Calibration table: mean predicted vs empirical up-rate per probability bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({
            "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
            "n": int(m.sum()),
            "pred_mean": float(np.mean(p[m])),
            "emp_up_rate": float(np.mean(y[m])),
        })
    return out


def realized_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Joins each prediction to its 10 s-forward sample (same window) and derives
    the realized Up outcome from the ``log_s_k`` change (ties → Up, the §6 rule)."""
    d = df.dropna(subset=["p_up", "log_s_k"]).copy()
    d = d.sort_values(["series", "window_open_ms", "sample_ts_ms"])
    fwd = d[["series", "window_open_ms", "sample_ts_ms", "log_s_k"]].copy()
    fwd["sample_ts_ms"] = fwd["sample_ts_ms"] - HORIZON_MS
    fwd = fwd.rename(columns={"log_s_k": "log_s_k_fwd"})
    m = d.merge(fwd, on=["series", "window_open_ms", "sample_ts_ms"], how="inner")
    m = m.dropna(subset=["log_s_k_fwd"])
    m["realized_up"] = (m["log_s_k_fwd"] >= m["log_s_k"]).astype(int)
    return m


def accuracy(df: pd.DataFrame) -> dict:
    m = realized_frame(df)
    if m.empty:
        return {"n": 0, "note": "no matched 10 s-forward pairs yet"}
    p = m["p_up"].to_numpy(float)
    y = m["realized_up"].to_numpy(float)
    result = {"overall": _metrics(p, y), "reliability": _reliability(p, y), "per_series": {}}
    for series, g in m.groupby("series"):
        result["per_series"][str(series)] = _metrics(
            g["p_up"].to_numpy(float), g["realized_up"].to_numpy(float)
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "shadow_accuracy")
    parser.add_argument("--shadow-dir", default=None)
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    shadow_dir = Path(args.shadow_dir) if args.shadow_dir else _default_shadow_dir()

    df = read_shadow_predictions(shadow_dir)
    if df.empty:
        print("[shadow_accuracy] no shadow predictions found")
        return 1
    result = accuracy(df)
    out_dir = paths.out_dir / "shadow_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    o = result.get("overall", {})
    if not o.get("n"):
        print(f"[shadow_accuracy] {result.get('note', 'no scorable pairs yet')}")
        return 0
    print(f"[shadow_accuracy] scored {o['n']:,} predictions on the realized 10s mid move")
    print(f"  directional accuracy = {o['dir_acc']*100:.1f}%   (base rate Up = {o['base_rate_up']*100:.1f}%)")
    print(f"  Brier = {o['brier']:.4f}   vs always-0.5 {o['brier_vs_half']:.4f}   vs base-rate {o['brier_vs_baserate']:.4f}")
    print(f"  log-loss = {o['log_loss']:.4f}   mean p_up = {o['p_up_mean']:.3f}")
    print("  --- two-sided skill (does it call DOWN, not just up?) ---")
    print(f"    UP moves:   {o['n_up_moves']:>6,}   caught {o['up_recall']*100:.1f}% of them")
    print(f"    DOWN moves: {o['n_down_moves']:>6,}   caught {o['down_recall']*100:.1f}% of them "
          f"(when it said 'down' it was right {o['down_precision']*100:.1f}%)")
    print(f"    it leans down on {o['pred_down_frac']*100:.1f}% of predictions")
    print(f"    >>> BALANCED accuracy = {o['balanced_acc']*100:.1f}%   (50% = no real skill / coin flip; base-rate-proof)")
    print("  per series (balanced acc | down-recall):")
    for s, mm in result["per_series"].items():
        ba = mm.get("balanced_acc", float("nan"))
        dr = mm.get("down_recall", float("nan"))
        print(f"    {s:<9} n={mm['n']:>5}  balanced={ba*100:5.1f}%  down_recall={dr*100:5.1f}%  "
              f"(dir_acc={mm['dir_acc']*100:.1f}%, baseUp={mm['base_rate_up']*100:.0f}%)")
    print("  calibration (pred vs empirical Up-rate):")
    for r in result["reliability"]:
        print(f"    {r['bin']:<10} n={r['n']:>5}  pred={r['pred_mean']:.3f}  actual={r['emp_up_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
