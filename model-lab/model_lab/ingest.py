"""Stage 1 — ingest.

Parse the gzip JSONL journal (and the Binance ``depth20@100ms`` capture, if
present) into tidy parquet tables under ``out/``:

    ticks · model · fills · settlements · windows · depth

Verify:  ``python -m model_lab.ingest``  prints per-table row counts. Depth is
optional — if ``../data/depth`` is empty it writes an empty ``depth.parquet`` and
warns you to run ``bot record`` (or ``bot run``) to capture it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import Paths, _in_bounds, resolve_bounds, resolve_paths, stage_parser
from .io import depth as depth_io
from .io import journal as jio

TICK_COLS = ["ts_local_ms", "source", "asset", "kind", "value", "ts_exchange", "ts_local"]
MODEL_COLS = [
    "ts", "asset", "series", "open_time", "p_up", "z", "sigma_1s", "sigma_tau",
    "basis", "anchor", "health", "reason",
]
FILL_COLS = [
    "ts_venue", "order_id", "series", "open_time", "token_id", "outcome", "side",
    "price", "size", "liquidity", "fee",
]
SETTLE_COLS = [
    "ts", "series", "open_time", "outcome", "up_shares", "down_shares",
    "realized_pnl", "fees_paid",
]
WINDOW_COLS = [
    "series", "open_time", "close_time", "event_slug", "condition_id",
    "up_token", "down_token", "strike",
]
DEPTH_COLS = [
    "recv_ms", "symbol", "asset", "best_bid", "best_ask", "mid", "microprice",
    "spread", "imbalance", "sum_bid_sz", "sum_ask_sz", "n_levels",
]


def _write(rows: list[dict], columns: list[str], path: Path) -> int:
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    df.to_parquet(path, engine="pyarrow", index=False)
    return len(df)


def ingest(paths: Paths, *, since_ms: int | None = None, until_ms: int | None = None) -> dict[str, int]:
    """Runs the ingest stage, writing every table. Returns per-table row counts.
    ``since_ms``/``until_ms`` bound the journal records by time (a bounded raw-tick
    export; unlike ``dataset``/``short_horizon`` this stage keeps every raw tick, so
    narrow the range on a large journal)."""
    paths.ensure_out()

    ticks: list[dict] = []
    models: list[dict] = []
    fills: list[dict] = []
    settlements: list[dict] = []
    windows: dict[tuple[str, int], dict] = {}

    for rec in jio.read_records(paths.journal_dir):
        kind = rec.get("type")
        if kind == "price_tick":
            te, tl = rec.get("ts_exchange"), rec.get("ts_local")
            _ts = te if te is not None else tl
            if not _in_bounds(int(_ts) if _ts is not None else None, since_ms, until_ms):
                continue
            ticks.append(
                {
                    "ts_local_ms": rec.get("ts_local_ms"),
                    "source": rec.get("source"),
                    "asset": jio.asset_key(rec.get("asset")),
                    "kind": rec.get("kind"),
                    "value": jio.to_float(rec.get("value")),
                    "ts_exchange": te,
                    "ts_local": tl,
                }
            )
        elif kind == "model":
            ts = rec.get("ts")
            if not _in_bounds(int(ts) if ts is not None else None, since_ms, until_ms):
                continue
            series, open_time = jio.window_key(rec.get("window"))
            models.append(
                {
                    "ts": rec.get("ts"),
                    "asset": jio.asset_key(rec.get("asset")),
                    "series": series,
                    "open_time": open_time,
                    "p_up": jio.to_float(rec.get("p_up")),
                    "z": jio.to_float(rec.get("z")),
                    "sigma_1s": jio.to_float(rec.get("sigma_1s")),
                    "sigma_tau": jio.to_float(rec.get("sigma_tau")),
                    "basis": jio.to_float(rec.get("basis")),
                    "anchor": rec.get("anchor"),
                    "health": rec.get("health"),
                    "reason": rec.get("reason"),
                }
            )
        elif kind == "fill":
            ts = rec.get("ts_venue")
            if not _in_bounds(int(ts) if ts is not None else None, since_ms, until_ms):
                continue
            series, open_time = jio.window_key(rec.get("window"))
            fills.append(
                {
                    "ts_venue": rec.get("ts_venue"),
                    "order_id": rec.get("order_id"),
                    "series": series,
                    "open_time": open_time,
                    "token_id": rec.get("token_id"),
                    "outcome": rec.get("outcome"),
                    "side": rec.get("side"),
                    "price": jio.to_float(rec.get("price")),
                    "size": jio.to_float(rec.get("size")),
                    "liquidity": rec.get("liquidity"),
                    "fee": jio.to_float(rec.get("fee")),
                }
            )
        elif kind == "settlement":
            ts = rec.get("ts")
            if not _in_bounds(int(ts) if ts is not None else None, since_ms, until_ms):
                continue
            series, open_time = jio.window_key(rec.get("window"))
            up = rec.get("up") or {}
            down = rec.get("down") or {}
            settlements.append(
                {
                    "ts": rec.get("ts"),
                    "series": series,
                    "open_time": open_time,
                    "outcome": rec.get("outcome"),
                    "up_shares": jio.to_float(up.get("shares")),
                    "down_shares": jio.to_float(down.get("shares")),
                    "realized_pnl": jio.to_float(rec.get("realized_pnl")),
                    "fees_paid": jio.to_float(rec.get("fees_paid")),
                }
            )
        elif kind == "window":
            market = rec.get("market") or {}
            key = jio.window_key(market.get("window"))
            if key not in windows and key != ("", 0) and _in_bounds(key[1], since_ms, until_ms):
                tokens = market.get("tokens") or {}
                windows[key] = {
                    "series": key[0],
                    "open_time": key[1],
                    "close_time": market.get("close_time"),
                    "event_slug": market.get("event_slug"),
                    "condition_id": market.get("condition_id"),
                    "up_token": tokens.get("up"),
                    "down_token": tokens.get("down"),
                    "strike": jio.to_float(market.get("strike")),
                }

    depth_rows = list(depth_io.read_depth_rows(paths.depth_dir))

    counts = {
        "ticks": _write(ticks, TICK_COLS, paths.table("ticks")),
        "model": _write(models, MODEL_COLS, paths.table("model")),
        "fills": _write(fills, FILL_COLS, paths.table("fills")),
        "settlements": _write(settlements, SETTLE_COLS, paths.table("settlements")),
        "windows": _write(list(windows.values()), WINDOW_COLS, paths.table("windows")),
        "depth": _write(depth_rows, DEPTH_COLS, paths.table("depth")),
    }
    return counts


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "ingest").parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    print(f"[ingest] journal={paths.journal_dir}")
    print(f"[ingest] depth  ={paths.depth_dir}")
    print(f"[ingest] out    ={paths.out_dir}")
    if since_ms is not None or until_ms is not None:
        print(f"[ingest] bounds ={args.since}..{args.until}")
    counts = ingest(paths, since_ms=since_ms, until_ms=until_ms)
    for name, n in counts.items():
        print(f"[ingest] {name:12s} {n:>10,} rows -> {paths.table(name).name}")
    if counts["depth"] == 0:
        print(
            "[ingest] NOTE: no depth20 data found. Run `cargo run -p bot -- record` "
            "(or `bot run`) with feeds.binance_depth_capture enabled to capture it; "
            "the research stage needs it."
        )
    if counts["ticks"] == 0 and counts["model"] == 0:
        print("[ingest] WARNING: no journal records found — check --journal-dir.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
