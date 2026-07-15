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
from datetime import datetime, timezone
from pathlib import Path

# Advisory memory ceiling (MB) for the streaming journal pass — the run reports peak
# RSS and warns if it exceeds this. Use --since/--until to bound the range and stay
# under it on a large journal.
DEFAULT_MAX_RSS_MB = 3072


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


def _default_telonex_dir() -> Path:
    # Purchased Telonex vendor data (Polymarket + Binance parquet), sibling to
    # data/journal and data/aggtrades (git-ignored by `data/`). Same pattern as
    # _default_hist_dir; backs the Paths field's default_factory.
    return REPO_ROOT / "data" / "telonex"


@dataclass(frozen=True)
class Paths:
    """Resolved input/output locations for a stage run."""

    journal_dir: Path
    depth_dir: Path
    out_dir: Path
    hist_dir: Path = field(default_factory=_default_hist_dir)
    telonex_dir: Path = field(default_factory=_default_telonex_dir)

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
    parser.add_argument(
        "--telonex-dir",
        default=None,
        help="Telonex vendor-data store (default ../data/telonex)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="only process journal records at/after this time — `YYYY-MM-DD`, an ISO "
        "datetime `YYYY-MM-DDTHH:MM:SS`, or a raw unix-ms integer (UTC). Bounds memory.",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="only process journal records strictly before this time (same formats as --since; UTC)",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=int,
        default=DEFAULT_MAX_RSS_MB,
        help=f"advisory memory ceiling (MB); the run reports peak RSS and warns if exceeded "
        f"(default {DEFAULT_MAX_RSS_MB}). Not a hard limit — narrow --since/--until to stay under it.",
    )


def parse_time_bound(value: str | int | None) -> int | None:
    """Parse a ``--since``/``--until`` bound to **unix milliseconds (UTC)**.

    Accepts a date ``2026-07-03`` (→ 00:00:00Z that day), an ISO datetime
    ``2026-07-03T04:00:00`` (naive treated as UTC), or a raw integer of unix
    milliseconds. ``None`` / empty → unbounded (``None``)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lstrip("-").isdigit():
        return int(s)  # raw unix ms
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(
            f"bad time bound {value!r}: use YYYY-MM-DD, an ISO datetime, or unix ms"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def resolve_bounds(args: argparse.Namespace | None) -> tuple[int | None, int | None]:
    """``(since_ms, until_ms)`` from parsed args (both ``None`` if absent)."""
    if args is None:
        return None, None
    return parse_time_bound(getattr(args, "since", None)), parse_time_bound(getattr(args, "until", None))


def _in_bounds(t: int | None, since_ms: int | None, until_ms: int | None) -> bool:
    """Whether a record time ``t`` (unix ms) falls in ``[since_ms, until_ms)``. Both
    bounds ``None`` → every record passes (including a record with no timestamp);
    with a bound set, a ``None`` time is excluded. The shared filter used by every
    journal-reading stage."""
    if since_ms is None and until_ms is None:
        return True
    if t is None:
        return False
    if since_ms is not None and t < since_ms:
        return False
    return not (until_ms is not None and t >= until_ms)


def resolve_paths(args: argparse.Namespace | None = None) -> Paths:
    """Resolves the three paths from CLI args → env → repo-relative defaults."""

    def pick(cli: str | None, env: str, default: Path) -> Path:
        if cli:
            return Path(cli).expanduser().resolve()
        env_val = os.environ.get(env)
        if env_val:
            return Path(env_val).expanduser().resolve()
        return default.resolve()

    ns = args or argparse.Namespace(
        journal_dir=None, depth_dir=None, out=None, hist_dir=None, telonex_dir=None
    )
    return Paths(
        journal_dir=pick(getattr(ns, "journal_dir", None), "MODEL_LAB_JOURNAL_DIR", _default_journal_dir()),
        depth_dir=pick(getattr(ns, "depth_dir", None), "MODEL_LAB_DEPTH_DIR", _default_depth_dir()),
        out_dir=pick(getattr(ns, "out", None), "MODEL_LAB_OUT", _default_out_dir()),
        hist_dir=pick(getattr(ns, "hist_dir", None), "MODEL_LAB_HIST_DIR", _default_hist_dir()),
        telonex_dir=pick(getattr(ns, "telonex_dir", None), "MODEL_LAB_TELONEX_DIR", _default_telonex_dir()),
    )


def stage_parser(description: str) -> argparse.ArgumentParser:
    """A parser preloaded with the common path flags, for a stage's ``main``."""
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser


class ParquetNotReady(RuntimeError):
    """A parquet a stage was told to consume is missing, truncated, or still being
    written (no finalized footer). Raised instead of letting a reader blow up with a
    cryptic error — or, worse, silently consume a partial file."""


def assert_parquet_ready(path, *, label: str | None = None, min_rows: int = 1) -> int:
    """Fail loudly if ``path`` is not a complete, readable parquet file.

    A parquet's footer (schema + row-group index) is written only when the writer
    **closes** it, so a truncated or still-being-written file has no valid footer.
    This reads just that footer (cheap — no data pages) and raises a clear,
    actionable :class:`ParquetNotReady` if the file is absent, unreadable, or has
    fewer than ``min_rows`` rows. Returns the row count on success. Call this before
    a stage consumes a parquet produced by an earlier (possibly streamed) stage.
    """
    import pyarrow.parquet as pq  # local import: keep config import light

    p = Path(path)
    name = label or p.name
    if not p.exists():
        raise ParquetNotReady(
            f"{name} not found at {p} — run the stage that produces it first."
        )
    try:
        num_rows = pq.ParquetFile(p).metadata.num_rows
    except Exception as exc:  # ArrowInvalid (bad footer), OSError, …
        raise ParquetNotReady(
            f"{name} at {p} is truncated or still being written "
            f"(no valid parquet footer: {type(exc).__name__}: {exc}). Re-run the "
            f"stage that produces it and let it finish before consuming it."
        ) from exc
    if num_rows < min_rows:
        raise ParquetNotReady(
            f"{name} at {p} has only {num_rows} row(s) (expected ≥ {min_rows}) "
            "— likely an interrupted write; re-run the producing stage."
        )
    return num_rows


def require_tables(paths: "Paths", *names: str, min_rows: int = 1) -> dict[str, int]:
    """Validate that each named ``out/`` parquet table is present and fully written
    before a stage reads it (see :func:`assert_parquet_ready`). Returns
    ``{name: row_count}``; raises :class:`ParquetNotReady` on the first bad one."""
    return {n: assert_parquet_ready(paths.table(n), label=f"{n}.parquet", min_rows=min_rows)
            for n in names}
