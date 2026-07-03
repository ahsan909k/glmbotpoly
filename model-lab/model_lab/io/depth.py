"""Read the Binance ``depth20@100ms`` capture (``binance-depth20-*.jsonl.gz``).

On-disk shape (written by the bot's ``depth_capture`` module) — one JSON object
per line::

    {"recv_ms": 1751500000123,
     "stream": "btcusdt@depth20@100ms",
     "data": {"lastUpdateId": 42,
              "bids": [["63500.01", "1.5"], ...],   # best first, [price, qty]
              "asks": [["63600.00", "3.0"], ...]}}

The partial-book-depth stream is a *snapshot* (20 levels/side), not a diff — so
each frame is a self-contained book. This reader parses each frame into the
microstructure aggregates the lab studies (best bid/ask, mid, microprice,
order-book imbalance, spread, summed depth).
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def depth_paths(depth_dir: Path) -> list[Path]:
    """All ``binance-depth20-*.jsonl.gz`` day-files, chronological order."""
    if not depth_dir.is_dir():
        return []
    return sorted(depth_dir.glob("binance-depth20-*.jsonl.gz"))


def _levels(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for level in raw:
        try:
            out.append((float(level[0]), float(level[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def parse_frame(recv_ms: int, stream: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Parses one wrapped depth frame into an aggregate row, or ``None`` if the
    book is empty/malformed."""
    bids = _levels(data.get("bids"))
    asks = _levels(data.get("asks"))
    if not bids or not asks:
        return None
    best_bid, bid_top_sz = bids[0]
    best_ask, ask_top_sz = asks[0]
    if best_bid <= 0 or best_ask <= 0:
        return None
    sum_bid = sum(q for _, q in bids)
    sum_ask = sum(q for _, q in asks)
    denom = sum_bid + sum_ask
    imbalance = (sum_bid - sum_ask) / denom if denom > 0 else 0.0
    top_denom = bid_top_sz + ask_top_sz
    # Microprice: size-weighted toward the heavier top side.
    microprice = (
        (best_bid * ask_top_sz + best_ask * bid_top_sz) / top_denom
        if top_denom > 0
        else (best_bid + best_ask) / 2.0
    )
    # Binance symbol is the stream prefix, e.g. "btcusdt@depth20@100ms".
    symbol = stream.split("@", 1)[0]
    return {
        "recv_ms": int(recv_ms),
        "symbol": symbol,
        "asset": _symbol_asset(symbol),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": (best_bid + best_ask) / 2.0,
        "microprice": microprice,
        "spread": best_ask - best_bid,
        "imbalance": imbalance,
        "sum_bid_sz": sum_bid,
        "sum_ask_sz": sum_ask,
        "n_levels": min(len(bids), len(asks)),
    }


def _symbol_asset(symbol: str) -> str:
    s = symbol.lower()
    if s.startswith("btc"):
        return "btc"
    if s.startswith("eth"):
        return "eth"
    return s


def read_depth_rows(depth_dir: Path) -> Iterator[dict[str, Any]]:
    """Yields aggregate rows for every parseable depth frame across all files."""
    for path in depth_paths(depth_dir):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    data = obj.get("data")
                    if not isinstance(data, dict):
                        continue
                    row = parse_frame(obj.get("recv_ms", 0), str(obj.get("stream", "")), data)
                    if row is not None:
                        yield row
        except (OSError, EOFError, zlib.error):
            # Unreadable file, or a partial trailing deflate member from a
            # still-being-written (live-capture) file: yield the decoded prefix and
            # move on (mirrors the journal reader).
            continue


def has_depth(depth_dir: Path) -> bool:
    """Whether any depth files are present (capture may not have run yet)."""
    return len(depth_paths(depth_dir)) > 0
