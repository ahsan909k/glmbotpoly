"""Read the bot's ``journal-*.jsonl.gz`` segments.

On-disk shape (verified against a live segment) — one JSON object per line, a
**nested** envelope::

    {"seq": 10220322,
     "ts_local_ms": 1782138864710,
     "rec": {"type": "top_of_book", "token_id": "...", "top": {...}}}

The ``rec.type`` tag is ``snake_case`` (the ``JournalRecord`` serde projection);
the *inner* domain enums keep their Rust ``PascalCase`` variant names, so e.g.
``price_tick.source`` is ``"BinanceDirect"``, ``asset`` is ``"Btc"``/``"Eth"``,
``kind`` is ``"Mid"``/``"Trade"``/``"Vendor"``, ``fill.side`` is
``"Buy"``/``"Sell"``, ``liquidity`` is ``"Maker"``/``"Taker"``, and
``model.health`` is ``"Ready"``/``"Degraded"``/``"Unreliable"``.

Money/size fields are exact-decimal **strings** and are parsed to ``float`` at
the boundary (the lab is research, not accounting — the engine keeps the exact
decimals).
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def segment_paths(journal_dir: Path) -> list[Path]:
    """All ``journal-*.jsonl.gz`` segments in a directory, chronological order.

    Filenames sort lexically = chronologically (``journal-YYYYMMDD-HHMMSS-NNNNN``).
    """
    if not journal_dir.is_dir():
        return []
    return sorted(journal_dir.glob("journal-*.jsonl.gz"))


def read_records(journal_dir: Path) -> Iterator[dict[str, Any]]:
    """Yields each ``rec`` object (with ``seq`` and ``ts_local_ms`` merged in).

    A crash-truncated trailing segment yields its decoded prefix and then stops
    (the recorder finalizes gzip trailers on clean shutdown; an abrupt kill can
    leave a partial final member). Unreadable segments are skipped.
    """
    for path in segment_paths(journal_dir):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        env = json.loads(line)
                    except json.JSONDecodeError:
                        # Truncated tail of a killed segment — stop this file.
                        break
                    rec = env.get("rec")
                    if not isinstance(rec, dict):
                        continue
                    rec["seq"] = env.get("seq")
                    rec["ts_local_ms"] = env.get("ts_local_ms")
                    yield rec
        except (OSError, EOFError, zlib.error):
            # An unreadable segment, or a partial trailing deflate member from a
            # still-being-written (live-capture) file: yield the decoded prefix
            # (already emitted above) and move on.
            continue


def records_of_type(journal_dir: Path, *types: str) -> Iterator[dict[str, Any]]:
    """Yields only the records whose ``type`` is one of ``types``."""
    wanted = set(types)
    for rec in read_records(journal_dir):
        if rec.get("type") in wanted:
            yield rec


def to_float(value: Any) -> float:
    """Parses an exact-decimal string (or number) to ``float``; ``nan`` if absent."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def asset_key(asset: Any) -> str:
    """Normalizes a Rust ``Asset`` (``"Btc"``/``"Eth"``) to ``"btc"``/``"eth"``."""
    return str(asset).lower()


def window_key(window: Any) -> tuple[str, int]:
    """Splits a serialized ``WindowId`` ``{"series","open_time"}`` into a key."""
    if not isinstance(window, dict):
        return ("", 0)
    return (str(window.get("series", "")), int(window.get("open_time", 0)))
