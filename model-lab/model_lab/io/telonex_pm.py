"""Read Telonex Polymarket **quote / trade** parquet — the historical PM book + flow.

The historical analogue of the recorder's ``top_of_book`` capture. For each
(window, outcome) Telonex stores a per-event top-of-book ``quotes`` file and a
``trades`` file under::

    data/telonex/raw/polymarket/{channel}/{slug}-{outcome}-{day}.parquet(.zst)

where ``slug = {asset}-updown-{dur}-{open_epoch}`` and ``outcome ∈ {Up, Down}``.
Columns (verified on disk): ``timestamp_us`` (exchange time, µs), ``asset_id`` (the
CLOB token id), ``bid_price``/``bid_size``/``ask_price``/``ask_size`` — all
exact-decimal **strings** (parsed to float at the boundary; the lab is research,
not accounting). The bulk pull deletes the raw ``.parquet`` after finalizing, so a
whole-file ``.zst`` is decompressed to scratch first (mirrors
``telonex_binance_validate._tel_depth_grid_any``).

:func:`pm_book_arrays` yields the per-token sorted arrays ``money_judge._book_asof``
/ ``_leaned_quote`` consume, so historical fills reuse those functions verbatim.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import telonex as tx

# The book columns we read from a quotes parquet (skip the header/id noise except
# asset_id, which carries the CLOB token id).
_QUOTE_COLS = ["timestamp_us", "asset_id", "bid_price", "bid_size", "ask_price", "ask_size"]
_TRADE_COLS = ["timestamp_us", "asset_id", "price", "size", "side"]

# Output frames (float at the boundary; ts_ms = timestamp_us // 1000).
QUOTE_OUT_COLS = ["token_id", "ts_ms", "bid_px", "bid_sz", "ask_px", "ask_sz"]
TRADE_OUT_COLS = ["token_id", "ts_ms", "price", "size", "side"]


def _read_tel_columns(raw_path: Path, columns: list[str], *, scratch: Path | None = None) -> pd.DataFrame | None:
    """Read ``columns`` from a Telonex parquet given its *raw* ``.parquet`` path.

    Reads the raw file if present, else decompresses its finalized ``.parquet.zst``
    sibling to a scratch temp first (the bulk pull deletes the raw). Returns ``None``
    when neither form is on disk. ``pq.read_table`` reads fully into memory (not
    memory-mapped), so the scratch temp is safe to unlink immediately after."""
    raw = Path(raw_path)
    if raw.exists():
        return pq.read_table(raw, columns=columns).to_pandas()
    zst = tx.zst_sibling(raw)
    if not zst.exists():
        return None
    scratch_dir = Path(scratch) if scratch is not None else raw.parent
    scratch_dir.mkdir(parents=True, exist_ok=True)
    # Process-unique temp name so concurrent readers (parallel stages) of the same .zst
    # never clobber each other's scratch (a shared deterministic name races to a 0-byte read).
    tmp = scratch_dir / (raw.name + f".{os.getpid()}.pm.tmp.parquet")
    try:
        tx.decompress_zstd(zst, tmp)
        return pq.read_table(tmp, columns=columns).to_pandas()
    finally:
        tx._safe_unlink(tmp)


def read_pm_quotes(
    telonex_dir: Path, slug: str, outcome: str, day_str: str, *, scratch: Path | None = None
) -> pd.DataFrame:
    """Top-of-book quote series for one (slug, outcome, day) — ``QUOTE_OUT_COLS``,
    sorted by ``ts_ms``. Empty frame if the file is absent. Decimal strings → float;
    rows with a non-finite/zero price on either side are dropped (an unusable book)."""
    raw = tx.polymarket_raw_path(Path(telonex_dir), "quotes", slug, outcome, day_str)
    df = _read_tel_columns(raw, _QUOTE_COLS, scratch=scratch)
    if df is None or df.empty:
        return pd.DataFrame(columns=QUOTE_OUT_COLS)
    out = pd.DataFrame(
        {
            "token_id": df["asset_id"].astype(str).to_numpy(),
            "ts_ms": (df["timestamp_us"].to_numpy(dtype="int64") // 1000),
            "bid_px": pd.to_numeric(df["bid_price"], errors="coerce").to_numpy(dtype=float),
            "bid_sz": pd.to_numeric(df["bid_size"], errors="coerce").to_numpy(dtype=float),
            "ask_px": pd.to_numeric(df["ask_price"], errors="coerce").to_numpy(dtype=float),
            "ask_sz": pd.to_numeric(df["ask_size"], errors="coerce").to_numpy(dtype=float),
        }
    )
    good = (
        np.isfinite(out["bid_px"]) & np.isfinite(out["ask_px"])
        & (out["bid_px"] > 0.0) & (out["ask_px"] > 0.0)
    )
    out = out[good].sort_values("ts_ms").reset_index(drop=True)
    return out


def read_pm_trades(
    telonex_dir: Path, slug: str, outcome: str, day_str: str, *, scratch: Path | None = None
) -> pd.DataFrame:
    """Trade prints for one (slug, outcome, day) — ``TRADE_OUT_COLS``, sorted by
    ``ts_ms`` (optional PM trade-flow features; not consumed by the default build)."""
    raw = tx.polymarket_raw_path(Path(telonex_dir), "trades", slug, outcome, day_str)
    df = _read_tel_columns(raw, _TRADE_COLS, scratch=scratch)
    if df is None or df.empty:
        return pd.DataFrame(columns=TRADE_OUT_COLS)
    out = pd.DataFrame(
        {
            "token_id": df["asset_id"].astype(str).to_numpy(),
            "ts_ms": (df["timestamp_us"].to_numpy(dtype="int64") // 1000),
            "price": pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float),
            "size": pd.to_numeric(df["size"], errors="coerce").to_numpy(dtype=float),
            "side": df["side"].astype(str).to_numpy(),
        }
    )
    return out.sort_values("ts_ms").reset_index(drop=True)


def token_id_of(
    telonex_dir: Path, slug: str, outcome: str, day_str: str, *, scratch: Path | None = None
) -> str | None:
    """The CLOB token id for (slug, outcome, day) — the ``asset_id`` of its quotes
    file (cheap: reads one column). ``None`` if the file is absent/empty."""
    raw = tx.polymarket_raw_path(Path(telonex_dir), "quotes", slug, outcome, day_str)
    df = _read_tel_columns(raw, ["asset_id"], scratch=scratch)
    if df is None or df.empty:
        return None
    return str(df["asset_id"].iloc[0])


def pm_book_arrays(quotes_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-token sorted numpy arrays for O(log n) as-of lookups — the exact shape
    ``money_judge._book_asof`` consumes: ``{ts, bid_px, bid_sz, ask_px, ask_sz}``.
    Assumes a single token's quotes (as produced by :func:`read_pm_quotes`)."""
    if quotes_df.empty:
        return {k: np.empty(0, dtype=float if k != "ts" else "int64")
                for k in ("ts", "bid_px", "bid_sz", "ask_px", "ask_sz")}
    q = quotes_df.sort_values("ts_ms")
    return {
        "ts": q["ts_ms"].to_numpy(dtype="int64"),
        "bid_px": q["bid_px"].to_numpy(dtype=float),
        "bid_sz": q["bid_sz"].to_numpy(dtype=float),
        "ask_px": q["ask_px"].to_numpy(dtype=float),
        "ask_sz": q["ask_sz"].to_numpy(dtype=float),
    }
