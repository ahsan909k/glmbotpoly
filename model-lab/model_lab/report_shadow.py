"""Shadow-mode smoke report — turns a shadow capture into the deliverable numbers.

Reads ``data/shadow/shadow-*.jsonl.gz`` and summarizes the observation run:
prediction count per series (and per UTC day), the deployed-model identity +
staleness, feature-coverage distribution (how many of the 24 features were finite,
a live health signal), and the p_up distribution per series. Pairs with
``shadow_parity`` (the feature-parity verdict) and the Rust
``shadow_order_flow`` test (the order-flow-identical proof) to form the full
24 h smoke report.

The 24 h number is operator-run: enable ``[shadow]`` on the ongoing 24/7 paper
run and re-run this once a day of predictions has accrued (~17 280/series/day at
the 5 s cadence).

Run::

    python -m model_lab.report_shadow
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_paths, stage_parser
from .shadow_parity import _default_shadow_dir, read_shadow_predictions

MS_PER_DAY = 86_400_000


def summarize(shadow: pd.DataFrame) -> dict:
    """Builds the smoke-report summary from the shadow prediction frame."""
    if shadow.empty:
        return {"total": 0, "note": "no shadow predictions found"}
    shadow = shadow.copy()
    shadow["day"] = (shadow["sample_ts_ms"] // MS_PER_DAY).astype("int64")
    # Per-series counts + p_up distribution.
    per_series = {}
    for series, g in shadow.groupby("series"):
        p = g["p_up"].to_numpy(float)
        p = p[np.isfinite(p)]
        per_series[str(series)] = {
            "predictions": int(len(g)),
            "predictions_per_day": {str(d): int(c) for d, c in g.groupby("day").size().items()},
            "p_up_mean": float(np.mean(p)) if len(p) else None,
            "p_up_p05": float(np.percentile(p, 5)) if len(p) else None,
            "p_up_p95": float(np.percentile(p, 95)) if len(p) else None,
        }
    # Feature coverage (finite features / 24) — a live health signal.
    feat_cols = [c for c in shadow.columns if c not in
                 ("series", "window_open_ms", "sample_ts_ms", "p_up", "day")]
    finite = shadow[feat_cols].apply(lambda r: int(np.isfinite(r.to_numpy(float)).sum()), axis=1)
    span_ms = int(shadow["sample_ts_ms"].max() - shadow["sample_ts_ms"].min())
    return {
        "total": int(len(shadow)),
        "span_hours": round(span_ms / 3_600_000, 2),
        "series": per_series,
        "coverage": {
            "mean_finite_of_24": round(float(finite.mean()), 2),
            "min_finite": int(finite.min()),
            "median_finite": int(finite.median()),
        },
        "first_ts_utc": datetime.fromtimestamp(
            int(shadow["sample_ts_ms"].min()) / 1000, tz=timezone.utc).isoformat(),
        "last_ts_utc": datetime.fromtimestamp(
            int(shadow["sample_ts_ms"].max()) / 1000, tz=timezone.utc).isoformat(),
    }


def report_shadow(paths, shadow_dir: Path) -> dict:
    shadow = read_shadow_predictions(shadow_dir)
    summary = summarize(shadow)
    out_dir = paths.out_dir / "shadow_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "report_shadow")
    parser.add_argument("--shadow-dir", default=None)
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    shadow_dir = Path(args.shadow_dir) if args.shadow_dir else _default_shadow_dir()
    summary = report_shadow(paths, shadow_dir)
    print(f"[report_shadow] {summary.get('total', 0):,} predictions over {summary.get('span_hours', 0)} h")
    for series, s in summary.get("series", {}).items():
        print(f"  {series:<10} {s['predictions']:>8,} predictions  "
              f"p_up~{s['p_up_mean']}  ({len(s['predictions_per_day'])} day(s))")
    cov = summary.get("coverage")
    if cov:
        print(f"  coverage: {cov['mean_finite_of_24']}/24 features finite (median {cov['median_finite']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
