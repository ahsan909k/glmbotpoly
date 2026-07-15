"""Stage — evaluate. The standardized way to score any model's probabilities.

Given a model's probability-of-Up outputs on a test period, this scores them with
Brier, log-loss, directional accuracy and a calibration curve, **always side by
side against two benchmarks**: (1) the formula model recomputed at the same
timestamps, and (2) the Polymarket order-book mid on the subset where recordings
exist. It emits one standardized verdict table — model vs each benchmark, per
calendar-day period, with relative-improvement percentages and a stability
summary across periods. All scoring runs through :mod:`model_lab.eval_harness`,
the single source of truth.

**Model input contract** — a grid parquet with columns::

    series, window_open_ms, sample_ts_ms, p_up

(the keys of ``dataset.parquet`` / ``feature_set.parquet``, so a learner's output
joins 1:1). Pass it with ``--predictions``. **With no ``--predictions``**, the
formula model is evaluated as a self-baseline (model-vs-market is meaningful;
model-vs-formula is an identity check).

Outputs ``out/evaluate/{metrics.json, scores.csv, verdict_table.csv,
reliability.csv, report.html}``.

Run::

    python -m model_lab.evaluate
    python -m model_lab.evaluate --predictions out/my_model_preds.parquet
    python -m model_lab.evaluate --predictions preds.parquet --min-windows 50 --series BTC-5m
"""

from __future__ import annotations

import sys

import pandas as pd

from . import eval_harness as eh
from .config import (
    ParquetNotReady, Paths, assert_parquet_ready, resolve_bounds, resolve_paths, stage_parser,
)

DEFAULT_MIN_WINDOWS = eh.DEFAULT_MIN_WINDOWS


def _parse_series(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


def _filter_series(df: pd.DataFrame, series: list[str] | None) -> pd.DataFrame:
    if series is None or df.empty or "series" not in df.columns:
        return df
    return df[df["series"].isin(series)]


def evaluate(
    paths: Paths,
    predictions=None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
    health: str = "ready",
    series: str | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> dict:
    """Evaluate a model through the harness. ``predictions`` is a grid-parquet path
    (or ``None`` for the formula self-baseline). Returns the metrics dict (also
    written to ``out/evaluate/``). ``since_ms``/``until_ms`` bound the journal."""
    series_filter = _parse_series(series)
    if predictions is None:
        df, counts = eh.build_self_baseline_frame(paths, since_ms=since_ms, until_ms=until_ms)
        model_label = "formula model (self-baseline)"
        self_baseline = True
    else:
        assert_parquet_ready(predictions, label="predictions.parquet", min_rows=1)
        pred_df = pd.read_parquet(predictions, engine="pyarrow")
        pred_df = _filter_series(pred_df, series_filter)
        df, counts = eh.build_external_frame(paths, pred_df, since_ms=since_ms, until_ms=until_ms)
        model_label = str(predictions)
        self_baseline = False
    df = _filter_series(df, series_filter)
    metrics = eh.run(
        df, counts=counts, min_windows=min_windows, health=health,
        title="Model evaluation", model_label=model_label, self_baseline=self_baseline,
    )
    eh.write_outputs(paths.out_dir / "evaluate", metrics)
    return metrics


def evaluate_predictions(
    paths: Paths,
    predictions: pd.DataFrame,
    *,
    model_label: str = "model",
    min_windows: int = DEFAULT_MIN_WINDOWS,
    health: str = "ready",
) -> dict:
    """Reusable programmatic entry point for future stages: score an in-memory
    predictions DataFrame (keyed ``series, window_open_ms, sample_ts_ms`` + ``p_up``)
    through the harness. Every later model-scoring stage should report through this."""
    df, counts = eh.build_external_frame(paths, predictions)
    metrics = eh.run(
        df, counts=counts, min_windows=min_windows, health=health,
        title="Model evaluation", model_label=model_label, self_baseline=False,
    )
    eh.write_outputs(paths.out_dir / "evaluate", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "evaluate")
    parser.add_argument("--predictions", default=None,
                        help="grid parquet keyed (series, window_open_ms, sample_ts_ms) + p_up; "
                             "omit to evaluate the formula model itself (self-baseline)")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS,
                        help=f"flag a period/breakdown low-sample below this many distinct windows (default {DEFAULT_MIN_WINDOWS})")
    parser.add_argument("--health", choices=("ready", "all"), default="ready",
                        help="score only health=Ready snapshots (default) or every snapshot")
    parser.add_argument("--series", default=None, help="comma-separated series filter, e.g. BTC-5m,ETH-5m")
    parser.add_argument("--period", choices=("day", "week"), default="day",
                        help="which period rollup to emphasize in the printed summary (both are computed)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    src = args.predictions or "formula model (self-baseline)"
    print(f"[evaluate] journal={paths.journal_dir}")
    print(f"[evaluate] model  ={src}")
    print(f"[evaluate] out    ={paths.out_dir / 'evaluate'}")
    if since_ms is not None or until_ms is not None:
        print(f"[evaluate] bounds ={args.since}..{args.until}")

    try:
        m = evaluate(paths, predictions=args.predictions, min_windows=args.min_windows,
                     health=args.health, series=args.series, since_ms=since_ms, until_ms=until_ms)
    except ParquetNotReady as exc:
        print(f"[evaluate] {exc}")
        return 1

    c = m["counts"]
    print(f"[evaluate] resolved windows={c['resolved_windows']:,} scored snapshots={c['joined_snapshots']:,}")
    overall = eh.find_scope(m["scope_rows"], "overall")
    if overall:
        print(f"[evaluate] model : Brier={eh.fmt(overall['model_brier'])} "
              f"log-loss={eh.fmt(overall['model_logloss'])} dir-acc={eh.fmt(overall['model_diracc'])} "
              f"(windows={overall['n_windows']})")
    period = args.period
    rows = m["verdict_table"][period]["rows"]
    for b in eh.BENCHMARKS:
        allrow = next((r for r in rows if r["benchmark"] == b and r["period"] == "ALL"), None)
        st = m["verdict_table"][period]["stability"][b]
        if allrow:
            print(f"[evaluate] vs {b:<7}: Brier {eh.fmt(allrow['model_brier'])} vs {eh.fmt(allrow['bench_brier'])} "
                  f"({eh.fmt(allrow['brier_improve_pct'], 1)}%), "
                  f"win rate {eh.fmt(st['win_rate'], 2)} over {st['periods_evaluated']} {period}(s)")
    print(f"[evaluate] verdict: {m['verdict']}")
    print(f"[evaluate] wrote {paths.out_dir / 'evaluate'} "
          "(metrics.json, scores.csv, verdict_table.csv, reliability.csv, report.html)")

    if c["resolved_windows"] == 0 or c["joined_snapshots"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
