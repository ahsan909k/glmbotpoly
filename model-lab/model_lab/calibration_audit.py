"""Calibration audit — is the formula model well-calibrated, and does it beat the market?

A thin wrapper over :mod:`model_lab.eval_harness` (the single source of truth for
scoring a model). It evaluates the **formula model** (``crates/model`` ``p_up``)
as the model under test — a *self-baseline*: benchmark (1) is the model itself
(an identity, ~0%), and the meaningful comparison is benchmark (2), the
Polymarket Up-token order-book mid ``(best_bid + best_ask)/2``.

Self-contained: it reads the gzip journal **directly** (not the ``ingest`` /
``labels`` parquet), so a single command works on fresh journal data with no
prerequisite stages. For every resolved window it lines up the two implied
probabilities of *Up* against the realized outcome and scores them with Brier,
log-loss and directional accuracy, split by series, time-remaining bucket, and
calendar-day period, with a reliability table and a plain-language verdict. Every
breakdown carries a **minimum-sample warning** wherever it is backed by too few
distinct *windows* to trust.

Outputs ``out/calibration_audit/{metrics.json, scores.csv, verdict_table.csv,
reliability.csv, report.html}``.

Run (single command, honors ``--journal-dir`` / ``--out``)::

    python -m model_lab.calibration_audit
    python -m model_lab.calibration_audit --min-windows 50 --health ready
"""

from __future__ import annotations

import sys

from . import eval_harness as eh
from .config import Paths, resolve_bounds, resolve_paths, stage_parser

DEFAULT_MIN_WINDOWS = eh.DEFAULT_MIN_WINDOWS


def calibration_audit(
    paths: Paths, min_windows: int = DEFAULT_MIN_WINDOWS, health: str = "ready",
    since_ms: int | None = None, until_ms: int | None = None,
) -> dict:
    """Runs the calibration audit (formula model as a self-baseline through the
    evaluation harness); returns the metrics dict (also written to disk).
    ``since_ms``/``until_ms`` bound the journal by time."""
    df, counts = eh.build_self_baseline_frame(paths, since_ms=since_ms, until_ms=until_ms)
    metrics = eh.run(
        df, counts=counts, min_windows=min_windows, health=health,
        title="Calibration audit", model_label="formula model", self_baseline=True,
    )
    eh.write_outputs(paths.out_dir / "calibration_audit", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "calibration_audit")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS,
                        help=f"flag a breakdown low-sample below this many distinct windows (default {DEFAULT_MIN_WINDOWS})")
    parser.add_argument("--health", choices=("ready", "all"), default="ready",
                        help="score only health=Ready snapshots (default) or every snapshot")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    print(f"[calibration_audit] journal={paths.journal_dir}")
    print(f"[calibration_audit] out    ={paths.out_dir / 'calibration_audit'}")
    if since_ms is not None or until_ms is not None:
        print(f"[calibration_audit] bounds : since={args.since} until={args.until}")

    m = calibration_audit(paths, min_windows=args.min_windows, health=args.health,
                          since_ms=since_ms, until_ms=until_ms)
    c = m["counts"]
    print(f"[calibration_audit] resolved windows={c['resolved_windows']:,} "
          f"model snapshots={c['model_snapshots']:,} market tops={c['market_tops']:,} "
          f"scored={c['joined_snapshots']:,}")
    overall = eh.find_scope(m["scope_rows"], "overall")
    if overall:
        print(f"[calibration_audit] model  : Brier={eh.fmt(overall['model_brier'])} "
              f"log-loss={eh.fmt(overall['model_logloss'])} (windows={overall['n_windows']})")
        print(f"[calibration_audit] market : Brier={eh.fmt(overall['market_brier'])} "
              f"log-loss={eh.fmt(overall['market_logloss'])}")
    n_low = sum(1 for r in m["scope_rows"] if r["scope"] in ("series", "tau_bucket", "series_tau") and r["low_sample"])
    if n_low:
        print(f"[calibration_audit] ⚠ {n_low} breakdown(s) below {args.min_windows} windows — flagged low_sample.")
    print(f"[calibration_audit] verdict: {m['verdict']}")
    print(f"[calibration_audit] wrote {paths.out_dir / 'calibration_audit'} "
          "(metrics.json, scores.csv, verdict_table.csv, reliability.csv, report.html)")

    if c["resolved_windows"] == 0 or c["joined_snapshots"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
