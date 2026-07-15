"""Run the competitor-analysis pipeline end to end.

``python -m model_lab.competitors [resolve|fetch|analyze|tape|report|all]``

Each stage is idempotent and resumable, so ``all`` can be re-run safely; it simply
skips work already cached under ``data/competitors/`` and re-renders the report.
"""

from __future__ import annotations

import sys

from . import (
    analyze, analyze_tape, fetch, final_report, manuals, manuals_report, report, resolve,
)

STAGES = {
    "resolve": resolve.main,
    "fetch": fetch.main,
    "analyze": analyze.main,
    "tape": analyze_tape.main,
    "report": report.main,
    "manuals": manuals.main,
    "manuals_report": manuals_report.main,
    "final_report": final_report.main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else "all"
    rest = argv[1:]
    if stage == "all":
        for name in ("resolve", "fetch", "analyze", "tape", "report"):
            print(f"\n===== {name} =====")
            rc = STAGES[name](rest if name in ("report",) else [])
            if rc:
                return rc
        return 0
    if stage not in STAGES:
        print(f"unknown stage {stage!r}; choose from {list(STAGES)} or 'all'")
        return 2
    return STAGES[stage](rest)


if __name__ == "__main__":
    raise SystemExit(main())
