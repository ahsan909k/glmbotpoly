"""Shadow feature-parity guard — the mandatory "basis-bug tripwire".

Compares the **live** 24-feature vectors the Rust ``shadow`` crate journaled
(``data/shadow/shadow-*.jsonl.gz``) against the **offline** features the lab
recomputes from the same main journal, joined on ``(series, window_open_ms,
sample_ts_ms)`` — shadow samples on the same window-aligned 5 s grid as
``short_horizon``, so the keys line up exactly. Per feature it reports the lab's
parity statistics (n / corr / bias / median-&-p95 abs err / magnitude ratio) and
gates them tiered by feature class: **tight** for the deterministic price
features, **looser** for the depth/PM features (whose live-vs-offline timing
jitter comes from the two independent captures). A failure means the live Rust
feature computation has drifted from the lab definition the champion was trained
on — exactly the class of bug (an uncorrected basis, a wrong EWMA) that a
backtest cannot catch.

Prerequisite: run ``python -m model_lab.short_horizon`` over the same journal
period first (it produces ``out/short_horizon.parquet``, the offline reference).

Pure-Python; no lightgbm needed for the feature check (an optional ``p_up`` check
uses the ``[gbt]`` extra + the deployed model if both are present).

Run::

    python -m model_lab.short_horizon        # build the offline reference first
    python -m model_lab.shadow_parity
"""

from __future__ import annotations

import gzip
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import learn_walkforward as lw
from .config import REPO_ROOT, assert_parquet_ready, resolve_paths, stage_parser
from .telonex_binance_validate import _feature_stats

FULL_FEATURES = lw.FULL_FEATURES
# Price features are deterministic given the same journal → tight tolerances.
PRICE_FEATURES = lw.BASE_FEATURES
# Depth/PM come from independent captures with sub-second timing jitter → looser.
MICRO_FEATURES = lw.DEPTH_FEATURES + lw.PM_FEATURES

# Gates (corr is scale-invariant; abs_ratio catches unit/scale bugs the imbalance
# corr would hide). Tuned for the live streaming lag (shadow reads the last
# completed bar, ~1 s behind the offline as-of bar); the operator tightens these
# once a real 24 h capture shows the achievable margins.
PRICE_MIN_CORR = 0.99
PRICE_RATIO = (0.95, 1.05)
MICRO_MIN_CORR = 0.95
MICRO_RATIO = (0.8, 1.25)
DEFAULT_MIN_N = 50


def _default_shadow_dir() -> Path:
    return REPO_ROOT / "data" / "shadow"


def read_shadow_predictions(shadow_dir: Path) -> pd.DataFrame:
    """Reads every ``shadow-*.jsonl.gz`` into a tidy frame with the 24 features
    expanded to named columns. NaN features (JSON ``null``) become ``NaN``.
    A truncated trailing gzip member (a live-appending file) is tolerated."""
    rows: list[dict] = []
    for path in sorted(shadow_dir.glob("shadow-*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    feats = rec.get("features") or []
                    row = {
                        "series": rec.get("series"),
                        "window_open_ms": rec.get("window_open_ms"),
                        "sample_ts_ms": rec.get("ts"),
                        "p_up": rec.get("p_up"),
                    }
                    for name, val in zip(FULL_FEATURES, feats):
                        row[name] = np.nan if val is None else float(val)
                    rows.append(row)
        except (OSError, EOFError, zlib.error):
            # Decoded prefix kept; the truncated last member is skipped.
            continue
    return pd.DataFrame(rows)


def run_parity(
    shadow: pd.DataFrame, offline: pd.DataFrame, *, min_n: int = DEFAULT_MIN_N
) -> dict:
    """Joins live vs offline features on the sample key and grades each feature.
    Returns a metrics dict with an overall ``passed`` verdict."""
    key = ["series", "window_open_ms", "sample_ts_ms"]
    off_cols = key + [f for f in FULL_FEATURES if f in offline.columns]
    joined = shadow.merge(
        offline[off_cols], on=key, how="inner", suffixes=("_live", "_off")
    )
    per_feature: dict[str, dict] = {}
    failures: list[str] = []
    n_matched = int(len(joined))
    for feat in FULL_FEATURES:
        live = joined.get(f"{feat}_live")
        off = joined.get(f"{feat}_off")
        if live is None or off is None:
            per_feature[feat] = {"n": 0, "note": "missing column"}
            continue
        # Compare only rows where BOTH are finite (a NaN on either side is an
        # expected-missing feature, e.g. an uncovered depth/PM row).
        mask = np.isfinite(live.to_numpy(float)) & np.isfinite(off.to_numpy(float))
        o = off.to_numpy(float)[mask]
        t = live.to_numpy(float)[mask]
        stats = _feature_stats(o, t)
        klass = "price" if feat in PRICE_FEATURES else "micro"
        min_corr = PRICE_MIN_CORR if klass == "price" else MICRO_MIN_CORR
        lo, hi = PRICE_RATIO if klass == "price" else MICRO_RATIO
        n = stats.get("n", 0)
        corr = stats.get("corr", float("nan"))
        ratio = stats.get("abs_ratio", float("nan"))
        ok = True
        if n >= min_n:
            if not (np.isfinite(corr) and corr >= min_corr):
                ok = False
                failures.append(f"{feat}: corr {corr:.4f} < {min_corr}")
            if np.isfinite(ratio) and not (lo <= ratio <= hi):
                ok = False
                failures.append(f"{feat}: abs_ratio {ratio:.3f} outside [{lo}, {hi}]")
        per_feature[feat] = {**stats, "class": klass, "min_corr": min_corr, "ok": ok}

    # Optional end-to-end p_up correlation (needs no lightgbm: shadow's p_up vs
    # the offline p_up_model formula probability is a coarse sanity signal).
    p_up_corr = None
    if "p_up" in joined.columns and "p_up_model_off" in joined.columns:
        a = joined["p_up"].to_numpy(float)
        b = joined["p_up_model_off"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= min_n and a[m].std() > 0 and b[m].std() > 0:
            p_up_corr = float(np.corrcoef(a[m], b[m])[0, 1])

    passed = n_matched >= min_n and not failures
    return {
        "n_matched": n_matched,
        "min_n": min_n,
        "passed": bool(passed),
        "failures": failures,
        "p_up_vs_formula_corr": p_up_corr,
        "per_feature": per_feature,
    }


def shadow_parity(paths, shadow_dir: Path, *, min_n: int = DEFAULT_MIN_N) -> dict:
    """Runs the guard: reads shadow predictions + the offline reference, grades
    parity, writes the report, returns the metrics dict."""
    short_path = paths.table("short_horizon")
    assert_parquet_ready(short_path, label="short_horizon.parquet", min_rows=1)
    offline = pd.read_parquet(short_path, engine="pyarrow")
    shadow = read_shadow_predictions(shadow_dir)
    if shadow.empty:
        result = {"n_matched": 0, "passed": False, "failures": ["no shadow predictions found"],
                  "per_feature": {}}
    else:
        result = run_parity(shadow, offline, min_n=min_n)

    out_dir = paths.out_dir / "shadow_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "shadow_parity")
    parser.add_argument("--shadow-dir", default=None, help="data/shadow directory (gzip predictions)")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    shadow_dir = Path(args.shadow_dir) if args.shadow_dir else _default_shadow_dir()

    if not paths.table("short_horizon").exists():
        print("[shadow_parity] out/short_horizon.parquet missing — run "
              "`python -m model_lab.short_horizon` over the same journal period first.")
        return 1
    result = shadow_parity(paths, shadow_dir, min_n=args.min_n)
    print(f"[shadow_parity] matched {result['n_matched']} live-vs-offline samples")
    if result.get("p_up_vs_formula_corr") is not None:
        print(f"[shadow_parity] corr(shadow p_up, formula p_up) = {result['p_up_vs_formula_corr']:.4f}")
    for feat, st in result["per_feature"].items():
        if st.get("n", 0) >= result.get("min_n", DEFAULT_MIN_N):
            flag = "ok" if st.get("ok") else "FAIL"
            print(f"  {feat:<18} n={st.get('n',0):>6} corr={st.get('corr', float('nan')):.4f} "
                  f"ratio={st.get('abs_ratio', float('nan')):.3f}  [{flag}]")
    if result["passed"]:
        print("[shadow_parity] PASS")
        return 0
    print("[shadow_parity] FAIL:")
    for f in result.get("failures", []):
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
