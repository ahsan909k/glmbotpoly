"""Shared paths and stage-argument handling.

Every stage reads inputs from the bot's data directories and writes its parquet
outputs under ``model-lab/out/``. Paths resolve in this order (highest first):

1. an explicit CLI flag (``--journal-dir`` / ``--depth-dir`` / ``--out``),
2. an environment variable
   (``MODEL_LAB_JOURNAL_DIR`` / ``MODEL_LAB_DEPTH_DIR`` / ``MODEL_LAB_OUT``),
3. the repo-relative default (``../data/journal``, ``../data/depth``,
   ``./out``).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _enable_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so the stages' Unicode output (Φ, σ, →, and the
    docstrings argparse prints on ``--help``) doesn't crash on a legacy Windows
    console code page (cp1252). Best-effort — a no-op on already-UTF-8 streams or
    ones that can't be reconfigured. Runs on import, so every entry point (each
    stage, verify, run_all) is covered before it prints.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_enable_utf8_console()

# model-lab/ (this file is model-lab/model_lab/config.py)
LAB_ROOT = Path(__file__).resolve().parent.parent
# The repo root is model-lab/.. — the Rust workspace root.
REPO_ROOT = LAB_ROOT.parent


def _default_hist_dir() -> Path:
    # Historical Binance aggTrades archive store — a large, persistent, resumable
    # download, sibling to data/journal and data/depth (git-ignored by `data/`).
    # Defined above Paths so it can back the field's default_factory.
    return REPO_ROOT / "data" / "aggtrades"


@dataclass(frozen=True)
class Paths:
    """Resolved input/output locations for a stage run."""

    journal_dir: Path
    depth_dir: Path
    out_dir: Path
    hist_dir: Path = field(default_factory=_default_hist_dir)

    def table(self, name: str) -> Path:
        """Path to a named parquet table under the output directory."""
        return self.out_dir / f"{name}.parquet"

    def ensure_out(self) -> None:
        """Create the output directory (and its subdirs are made on demand)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)


def _default_journal_dir() -> Path:
    return REPO_ROOT / "data" / "journal"


def _default_depth_dir() -> Path:
    return REPO_ROOT / "data" / "depth"


def _default_out_dir() -> Path:
    return LAB_ROOT / "out"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Adds the shared ``--journal-dir`` / ``--depth-dir`` / ``--out`` flags."""
    parser.add_argument(
        "--journal-dir",
        default=None,
        help="directory of journal-*.jsonl.gz segments (default ../data/journal)",
    )
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="directory of binance-depth20-*.jsonl.gz files (default ../data/depth)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory for parquet tables + reports (default ./out)",
    )
    parser.add_argument(
        "--hist-dir",
        default=None,
        help="Binance aggTrades archive store (default ../data/aggtrades)",
    )


def resolve_paths(args: argparse.Namespace | None = None) -> Paths:
    """Resolves the three paths from CLI args → env → repo-relative defaults."""

    def pick(cli: str | None, env: str, default: Path) -> Path:
        if cli:
            return Path(cli).expanduser().resolve()
        env_val = os.environ.get(env)
        if env_val:
            return Path(env_val).expanduser().resolve()
        return default.resolve()

    ns = args or argparse.Namespace(journal_dir=None, depth_dir=None, out=None, hist_dir=None)
    return Paths(
        journal_dir=pick(getattr(ns, "journal_dir", None), "MODEL_LAB_JOURNAL_DIR", _default_journal_dir()),
        depth_dir=pick(getattr(ns, "depth_dir", None), "MODEL_LAB_DEPTH_DIR", _default_depth_dir()),
        out_dir=pick(getattr(ns, "out", None), "MODEL_LAB_OUT", _default_out_dir()),
        hist_dir=pick(getattr(ns, "hist_dir", None), "MODEL_LAB_HIST_DIR", _default_hist_dir()),
    )


def stage_parser(description: str) -> argparse.ArgumentParser:
    """A parser preloaded with the common path flags, for a stage's ``main``."""
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser
