"""Stage — learn. The first challenger: walk-forward logistic regression.

Trains an L2-regularized logistic regression (hand-written numpy — the lab has no
scikit-learn) on the 16 engineered features in ``feature_set.parquet``, predicting
two targets **separately**:

- **fwd30** — the 30-second mid direction (``fwd_up_30s``), the short-horizon
  directional signal;
- **outcome** — the window's Up/Down resolution (``outcome_up``), the same
  probability-of-Up the formula model and the market predict.

**Evaluation is strict walk-forward.** For each target we slide a trailing
training window of ``--train-weeks`` weeks, test on the following ``--test-days``
days, and roll forward through the whole span — never shuffling across time. At
every train/test boundary we **purge** training rows whose label information
reaches into the test period (a gap ≥ the label horizon: 30 s for fwd30, the
window close for outcome), so no fold can peek across the boundary. The
out-of-sample (OOS) predictions from all folds are pooled into one grid.

**Everything is reported through the evaluation harness** (:mod:`model_lab.eval_harness`,
the single source of truth): each target's OOS grid is scored vs the formula model
and the market — the harness self-restricts to the real journal (Chainlink)
windows where those benchmarks exist. The 30-second model is *additionally* scored
on its own ``fwd_up_30s`` label (which the harness, hardwired to the window
outcome, can't express), using the same metric primitives.

**Shuffled-label control.** For each target we retrain the *identical* pipeline on
labels permuted once (seeded) — its performance must collapse to chance (it can't
beat the base rate, and it isn't reliably better than the formula). This proves
the pipeline can't cheat.

**Artifacts** (``out/learn/``) — the lab's first persisted models, as inspectable
JSON: ``model_<target>.json`` (coefficients + standardization + the reproducibility
seed), the OOS ``predictions_<target>.parquet`` grids, ``harness_<target>/`` reports,
``metrics.json`` and ``folds.csv``.

Reads ``feature_set.parquet`` (run ``dataset`` then ``feature_set`` first). Needs
the journal for the harness benchmarks (skip it with ``--no-harness``).

Run::

    python -m model_lab.learn                       # both targets, last 90 days
    python -m model_lab.learn --days 90 --train-weeks 4 --test-days 7
    python -m model_lab.learn --targets outcome --no-harness   # quick iteration
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import eval_harness as eh
from .config import ParquetNotReady, Paths, resolve_bounds, resolve_paths, stage_parser
from .feature_set import FEATURE_NAMES, WINDOW_SECS
from .learn_common import (
    MS_PER_DAY, TARGETS, _fold_masks, _fold_metrics, _fold_schedule, _harness_grid,
    _harness_summary, _label_info_ms, _load_matrix, _parse_series, score_through_harness,
)
from .lib import logreg as lr
from .lib import math as lm

DEFAULT_SEED = 20260704
DEFAULT_DAYS = 90
DEFAULT_TRAIN_WEEKS = 4
DEFAULT_TEST_DAYS = 7
DEFAULT_L2 = 1.0
DEFAULT_MIN_WINDOWS = eh.DEFAULT_MIN_WINDOWS

CAVEATS = [
    "Binance-proxy outcomes approximate the true Chainlink resolution (~0.987 agreement); "
    "the harness scores only the real journal windows.",
    "The full-span walk-forward OOS metrics are the primary evaluation; the harness verdict "
    "covers the journal subset and flags thin per-day periods (min_windows).",
    "Samples within a 5-minute window are correlated; folds are grouped by day/window.",
    "Features are as-of by feature_set construction; the purge enforces train/test label "
    "separation; the only shuffle is the deliberate control.",
    "One model per fold is trained across all series (pooled); hour_of_day is a plain numeric "
    "feature — per-series models and cyclic time encoding are noted future improvements.",
]


# ===========================================================================
# walk-forward
# ===========================================================================
def _run_walk_forward(
    sub: pd.DataFrame,
    y: np.ndarray,
    label_info_ms: np.ndarray,
    *,
    train_weeks: int,
    test_days: int,
    purge_ms: int,
    l2: float,
) -> tuple[pd.DataFrame, list[dict], str]:
    """Slide a trailing train window over ``sub`` and collect OOS test predictions.

    ``sub`` carries the key columns + the 16 features (aligned row-for-row with
    ``y`` and ``label_info_ms``). Returns ``(oos_df, fold_records, mode)`` where
    ``oos_df`` has ``[series, window_open_ms, sample_ts_ms, p_up, y_true]``.
    """
    open_ms = sub["window_open_ms"].to_numpy(np.int64)
    day = open_ms // MS_PER_DAY
    X = sub[FEATURE_NAMES].to_numpy(dtype=float)
    train_days = int(train_weeks) * 7
    folds, mode = _fold_schedule(np.unique(day), train_days, test_days)

    oos_parts: list[pd.DataFrame] = []
    records: list[dict] = []
    for i, (train_start, test_start, test_end) in enumerate(folds):
        train_mask, test_mask = _fold_masks(
            day, label_info_ms, train_start, test_start, test_end, purge_ms)
        n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
        if n_tr == 0 or n_te == 0:
            records.append({
                "fold_idx": i, "test_start_day": int(test_start), "test_end_day": int(test_end),
                "n_train": n_tr, "n_test": n_te, "n_test_windows": 0, "skipped": True,
                "pos_frac": float("nan"), "brier": float("nan"),
                "logloss": float("nan"), "diracc": float("nan"),
            })
            continue
        mean, std = lr.standardize_fit(X[train_mask])
        coef, intercept = lr.fit_logistic_ridge(
            lr.standardize_apply(X[train_mask], mean, std), y[train_mask], l2=l2
        )
        prob = lr.predict_proba(lr.standardize_apply(X[test_mask], mean, std), coef, intercept)
        y_te = y[test_mask]
        part = sub.loc[test_mask, ["series", "window_open_ms", "sample_ts_ms"]].copy()
        part["p_up"] = prob
        part["y_true"] = y_te
        oos_parts.append(part)
        m = _fold_metrics(prob, y_te)
        n_win = int(part.groupby(["series", "window_open_ms"]).ngroups)
        records.append({
            "fold_idx": i, "test_start_day": int(test_start), "test_end_day": int(test_end),
            "n_train": n_tr, "n_test": n_te, "n_test_windows": n_win, "skipped": False,
            **m,
        })

    cols = ["series", "window_open_ms", "sample_ts_ms", "p_up", "y_true"]
    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame(columns=cols)
    return oos, records, mode


def _final_model(sub: pd.DataFrame, y: np.ndarray, *, train_weeks: int, l2: float,
                 target: str, label: str, horizon_secs) -> dict:
    """Fit the deployable model on the most recent ``train_weeks`` of data (no test
    after it). Returns a JSON-ready artifact dict."""
    open_ms = sub["window_open_ms"].to_numpy(np.int64)
    day = open_ms // MS_PER_DAY
    hi = int(day.max())
    start_day = hi - int(train_weeks) * 7 + 1
    mask = day >= start_day
    if int(mask.sum()) == 0:
        mask = np.ones(len(day), dtype=bool)
        start_day = int(day.min())
    X = sub[FEATURE_NAMES].to_numpy(dtype=float)[mask]
    yt = y[mask]
    mean, std = lr.standardize_fit(X)
    coef, intercept = lr.fit_logistic_ridge(lr.standardize_apply(X, mean, std), yt, l2=l2)
    return {
        "target": target,
        "label": label,
        "horizon_secs": horizon_secs,
        "model": "logistic_regression_ridge_l2",
        "l2": float(l2),
        "feature_names": list(FEATURE_NAMES),
        "standardize": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
        "coef": [float(v) for v in coef],
        "intercept": float(intercept),
        "n_train": int(mask.sum()),
        "train_span_days": [start_day, hi],
        "train_span_open_ms": [int(open_ms[mask].min()), int(open_ms[mask].max())],
        "class_balance_pos_frac": float(np.mean(yt)),
    }


# ===========================================================================
# per-target driver
# ===========================================================================
def _run_target(
    paths: Paths, mat: pd.DataFrame, target: str, *,
    train_weeks: int, test_days: int, purge_secs: int, l2: float, seed: int,
    min_windows: int, health: str, bench_ctx: dict | None, run_shuffle: bool,
) -> tuple[dict, dict, list[dict]]:
    """Walk-forward + final model + harness + shuffled control for one target.

    Returns ``(target_block, shuffled_block, fold_rows)`` for the top-level metrics.
    """
    spec = TARGETS[target]
    label, horizon = spec["label"], spec["horizon_secs"]
    sub = mat[mat[label].notna()].reset_index(drop=True)
    y = sub[label].to_numpy(dtype=float)
    info = _label_info_ms(sub, horizon)
    purge_ms = int(purge_secs) * 1000

    oos, folds, mode = _run_walk_forward(
        sub, y, info, train_weeks=train_weeks, test_days=test_days, purge_ms=purge_ms, l2=l2,
    )
    pooled = _fold_metrics(oos["p_up"].to_numpy(float), oos["y_true"].to_numpy(float)) \
        if len(oos) else {"n": 0, "pos_frac": float("nan"), "brier": float("nan"),
                          "logloss": float("nan"), "diracc": float("nan")}

    # Persist the OOS grid + the deployable model artifact.
    out_dir = paths.out_dir / "learn"
    out_dir.mkdir(parents=True, exist_ok=True)
    _harness_grid(oos).to_parquet(out_dir / f"predictions_{target}.parquet", engine="pyarrow", index=False)
    model = _final_model(sub, y, train_weeks=train_weeks, l2=l2,
                         target=target, label=label, horizon_secs=horizon)
    model.update({"seed": int(seed), "days_span_days": [int((sub["window_open_ms"].min()) // MS_PER_DAY),
                                                        int((sub["window_open_ms"].max()) // MS_PER_DAY)]})
    (out_dir / f"model_{target}.json").write_text(json.dumps(model, indent=2), encoding="utf-8")

    block: dict = {
        "label": label, "horizon_secs": horizon, "mode": mode,
        "n_folds": int(sum(1 for f in folds if not f["skipped"])),
        "n_oos": int(len(oos)),
        "pooled": pooled,
        "final_model_file": f"model_{target}.json",
        "final_model": {"n_train": model["n_train"], "train_span_days": model["train_span_days"],
                        "class_balance_pos_frac": model["class_balance_pos_frac"]},
    }
    # fwd30's own-label reliability (the native eval the harness can't express).
    if target == "fwd30" and len(oos):
        block["own_label_reliability"] = lm.reliability_table(
            oos["p_up"].to_numpy(float), oos["y_true"].to_numpy(float)
        )

    if bench_ctx is not None and len(oos):
        hm = score_through_harness(
            paths, bench_ctx, oos, subdir="learn", out_name=f"harness_{target}",
            model_label=f"logreg-{target}", title=f"logreg challenger — {target}",
            min_windows=min_windows, health=health,
        )
        block["harness"] = _harness_summary(hm)

    fold_rows = [{"target": target, "control": False, **f} for f in folds]

    # ---- shuffled-label control (proves the pipeline can't cheat) ----
    shuffled: dict = {}
    if run_shuffle and len(oos):
        rng = np.random.default_rng(seed)
        y_shuf = y[rng.permutation(len(y))]
        oos_s, folds_s, _ = _run_walk_forward(
            sub, y_shuf, info, train_weeks=train_weeks, test_days=test_days,
            purge_ms=purge_ms, l2=l2,
        )
        pooled_s = _fold_metrics(oos_s["p_up"].to_numpy(float), oos_s["y_true"].to_numpy(float)) \
            if len(oos_s) else {"n": 0, "brier": float("nan"), "logloss": float("nan"),
                                "diracc": float("nan"), "pos_frac": float("nan")}
        p_oos = float(np.mean(oos_s["y_true"].to_numpy(float))) if len(oos_s) else float("nan")
        chance_brier = p_oos * (1.0 - p_oos)
        base_rate_acc = max(p_oos, 1.0 - p_oos)
        _harness_grid(oos_s).to_parquet(
            out_dir / f"predictions_{target}_shuffled.parquet", engine="pyarrow", index=False)
        reliably_better = None
        if bench_ctx is not None and len(oos_s):
            hm_s = score_through_harness(
                paths, bench_ctx, oos_s, subdir="learn", out_name=f"harness_{target}_shuffled",
                model_label=f"logreg-{target}-shuffled", title=f"logreg shuffled control — {target}",
                min_windows=min_windows, health=health,
            )
            reliably_better = _harness_summary(hm_s)["vs_formula"]["reliably_better"]
        collapsed = bool(
            np.isfinite(pooled_s["brier"]) and np.isfinite(pooled["brier"])
            and pooled_s["brier"] >= chance_brier - 0.03
            and pooled["brier"] < pooled_s["brier"]
            and (reliably_better is not True)
        )
        shuffled = {
            "pooled": pooled_s,
            "chance_brier": chance_brier,
            "base_rate_acc": base_rate_acc,
            "real_pooled_brier": pooled["brier"],
            "real_pooled_diracc": pooled["diracc"],
            "harness_vs_formula_reliably_better": reliably_better,
            "collapsed": collapsed,
        }
        fold_rows += [{"target": target, "control": True, **f} for f in folds_s]

    return block, shuffled, fold_rows


# ===========================================================================
# stage entry
# ===========================================================================
def learn(
    paths: Paths,
    *,
    days: int = DEFAULT_DAYS,
    train_weeks: int = DEFAULT_TRAIN_WEEKS,
    test_days: int = DEFAULT_TEST_DAYS,
    purge_secs: int | None = None,
    l2: float = DEFAULT_L2,
    seed: int = DEFAULT_SEED,
    targets: tuple[str, ...] = ("fwd30", "outcome"),
    source: str = "all",
    series: str | None = None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    health: str = "ready",
    run_harness: bool = True,
    run_shuffle: bool = True,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> dict:
    """Train + walk-forward-evaluate the logistic challenger; write ``out/learn/``.

    Returns the top-level metrics dict (also serialized to
    ``out/learn/metrics.json``). ``since_ms``/``until_ms`` bound the harness journal read."""
    series_filter = _parse_series(series)
    mat = _load_matrix(paths, days=days, source=source, series=series_filter)

    out_dir = paths.out_dir / "learn"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "config": {
            "days": days, "train_weeks": train_weeks, "test_days": test_days,
            "purge_secs_override": purge_secs, "l2": l2, "seed": seed,
            "targets": list(targets), "source": source, "series": series,
            "min_windows": min_windows, "health": health, "run_harness": run_harness,
        },
        "seed": int(seed),
        "features": list(FEATURE_NAMES),
        "n_rows_loaded": int(len(mat)),
        "caveats": CAVEATS,
        "targets": {},
        "shuffled_control": {},
    }
    if mat.empty:
        result["error"] = "feature_set.parquet had no rows in scope (check --days/--source/--series)."
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default),
                                              encoding="utf-8")
        return result

    result["date_span"] = {
        "min_open_ms": int(mat["window_open_ms"].min()),
        "max_open_ms": int(mat["window_open_ms"].max()),
        "min_day": int(mat["window_open_ms"].min() // MS_PER_DAY),
        "max_day": int(mat["window_open_ms"].max() // MS_PER_DAY),
        "distinct_days": int(np.unique(mat["window_open_ms"].to_numpy(np.int64) // MS_PER_DAY).size),
    }

    # Read the journal once for the benchmarks; reused across every grid.
    bench_ctx = eh.load_benchmarks(paths, since_ms=since_ms, until_ms=until_ms) if run_harness else None

    all_folds: list[dict] = []
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r} (expected one of {list(TARGETS)})")
        p_secs = purge_secs if purge_secs is not None else (
            TARGETS[target]["horizon_secs"] or WINDOW_SECS
        )
        block, shuffled, fold_rows = _run_target(
            paths, mat, target, train_weeks=train_weeks, test_days=test_days,
            purge_secs=p_secs, l2=l2, seed=seed, min_windows=min_windows, health=health,
            bench_ctx=bench_ctx, run_shuffle=run_shuffle,
        )
        block["purge_secs"] = p_secs
        result["targets"][target] = block
        if shuffled:
            result["shuffled_control"][target] = shuffled
        all_folds.extend(fold_rows)

    if all_folds:
        fold_cols = ["target", "control", "fold_idx", "test_start_day", "test_end_day",
                     "n_train", "n_test", "n_test_windows", "skipped",
                     "pos_frac", "brier", "logloss", "diracc"]
        pd.DataFrame(all_folds, columns=fold_cols).to_csv(out_dir / "folds.csv", index=False)

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    return result


def _fmt(x, *args) -> str:
    return eh.fmt(x, *args) if x is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "learn")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"trailing days of feature_set to use (0 = all; default {DEFAULT_DAYS})")
    parser.add_argument("--train-weeks", type=int, default=DEFAULT_TRAIN_WEEKS,
                        help=f"trailing training-window length in weeks (default {DEFAULT_TRAIN_WEEKS})")
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS,
                        help=f"walk-forward test-block size in days (default {DEFAULT_TEST_DAYS})")
    parser.add_argument("--purge-secs", type=int, default=None,
                        help="purge gap at each boundary in seconds (default: the label horizon "
                             "— 30 for fwd30, the window length for outcome)")
    parser.add_argument("--l2", type=float, default=DEFAULT_L2,
                        help=f"ridge (L2) strength on standardized features (default {DEFAULT_L2})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"reproducibility seed (shuffle control + saved; default {DEFAULT_SEED})")
    parser.add_argument("--targets", default="fwd30,outcome",
                        help="comma-separated targets: fwd30,outcome (default both)")
    parser.add_argument("--source", choices=("all", "chainlink", "binance_proxy"), default="all",
                        help="train/test on all rows, or only chainlink / binance_proxy (default all)")
    parser.add_argument("--series", default=None, help="comma-separated series filter, e.g. BTC-5m,ETH-5m")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS,
                        help=f"harness low-sample threshold (distinct windows; default {DEFAULT_MIN_WINDOWS})")
    parser.add_argument("--health", choices=("ready", "all"), default="ready",
                        help="harness: score only health=Ready snapshots (default) or all")
    parser.add_argument("--no-harness", action="store_true",
                        help="skip the harness (and its journal read) — faster iteration")
    parser.add_argument("--no-shuffle-control", action="store_true",
                        help="skip the shuffled-label control")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())

    print(f"[learn] feature_set={paths.table('feature_set')}")
    print(f"[learn] out        ={paths.out_dir / 'learn'}")
    try:
        m = learn(
            paths, days=args.days, train_weeks=args.train_weeks, test_days=args.test_days,
            purge_secs=args.purge_secs, l2=args.l2, seed=args.seed, targets=targets,
            source=args.source, series=args.series, min_windows=args.min_windows,
            health=args.health, run_harness=not args.no_harness,
            run_shuffle=not args.no_shuffle_control, since_ms=since_ms, until_ms=until_ms,
        )
    except ParquetNotReady as exc:
        print(f"[learn] {exc}")
        return 1

    if "error" in m:
        print(f"[learn] {m['error']}")
        return 1

    ds = m["date_span"]
    print(f"[learn] rows={m['n_rows_loaded']:,} over {ds['distinct_days']} days "
          f"(day {ds['min_day']}→{ds['max_day']}), seed={m['seed']}")
    for target, blk in m["targets"].items():
        p = blk["pooled"]
        print(f"[learn] {target:<8} [{blk['mode']}, {blk['n_folds']} fold(s), {blk['n_oos']:,} OOS]: "
              f"Brier={_fmt(p['brier'])} log-loss={_fmt(p['logloss'])} dir-acc={_fmt(p['diracc'])}")
        h = blk.get("harness")
        if h:
            vf, vm = h["vs_formula"], h["vs_market"]
            print(f"[learn]   harness (journal, {h['n_scored']:,} scored): "
                  f"vs formula {_fmt(vf['brier_improve_pct'], 1)}% (reliably_better={vf['reliably_better']}), "
                  f"vs market {_fmt(vm['brier_improve_pct'], 1)}%")
        sc = m["shuffled_control"].get(target)
        if sc:
            ps = sc["pooled"]
            print(f"[learn]   shuffled control: Brier={_fmt(ps['brier'])} dir-acc={_fmt(ps['diracc'])} "
                  f"(chance Brier={_fmt(sc['chance_brier'])}) → collapsed={sc['collapsed']}")
    print(f"[learn] wrote {paths.out_dir / 'learn'} "
          "(metrics.json, folds.csv, model_*.json, predictions_*.parquet, harness_*/)")

    ok = all(blk["n_oos"] > 0 for blk in m["targets"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
