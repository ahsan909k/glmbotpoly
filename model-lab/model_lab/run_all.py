"""Run every stage in order against the configured data (convenience wrapper).

Equivalent to running, in sequence:
``ingest → features → labels → dataset → validate → calibration_audit → research → report``.
Each stage still has its own entry point (``python -m model_lab.<stage>``); this
just chains them for a full refresh over the real journal.

Run:  ``python -m model_lab.run_all``  (honors ``--journal-dir/--depth-dir/--out``).
"""

from __future__ import annotations

import sys

from .calibration_audit import calibration_audit
from .config import resolve_paths, stage_parser
from .dataset import dataset
from .features import features
from .ingest import ingest
from .labels import labels
from .report import report
from .research import research
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "run all stages").parse_args(argv)
    paths = resolve_paths(args)
    print(f"[run_all] journal={paths.journal_dir}  depth={paths.depth_dir}  out={paths.out_dir}")

    counts = ingest(paths)
    print(f"[run_all] ingest: {counts}")
    if counts["ticks"] == 0 and counts["model"] == 0:
        print("[run_all] no journal data found — nothing to do.")
        return 1

    print(f"[run_all] features: {features(paths):,} rows")
    n_fwd, n_win = labels(paths)
    print(f"[run_all] labels: {n_fwd:,} forward, {n_win:,} windows")
    ds = dataset(paths)
    dc = ds["counts"]
    print(f"[run_all] dataset: {dc['samples']:,} samples over {dc['windows']:,} windows "
          f"({dc['samples_in_coverage']:,} in coverage)")
    v = validate(paths)
    print(f"[run_all] validate: Brier={v['calibration'].get('brier')}, "
          f"σ-corr={v['sigma_reproduction'].get('correlation')}")
    ca = calibration_audit(paths)
    ov = next((row for row in ca["scope_rows"] if row["scope"] == "overall"), None)
    print(f"[run_all] calibration_audit: windows={ca['counts']['resolved_windows']}, "
          f"model Brier={ov['model_brier'] if ov else None}, "
          f"market Brier={ov['market_brier'] if ov else None}")
    r = research(paths)
    print(f"[run_all] research: has_depth={r['has_depth']}, "
          f"IC(imbalance)={r.get('pooled', {}).get('ic_imbalance')}")
    print(f"[run_all] report: {report(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
