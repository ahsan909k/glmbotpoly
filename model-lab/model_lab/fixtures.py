"""Generate a tiny synthetic journal + depth capture for verification.

Produces data in the **exact** on-disk format (nested ``{seq, ts_local_ms, rec}``
envelopes with ``PascalCase`` domain enums; a ``binance-depth20-*.jsonl.gz`` day
file) so the whole pipeline runs end-to-end in seconds without a live capture. By
construction:

- model snapshots satisfy ``p_up == Φ(z)`` exactly (validate's Φ identity),
- ``sigma_1s`` is the engine EWMA of the emitted Binance bars (validate's σ
  reproduction correlates ~1.0),
- order-book **imbalance leads the price** (research finds a positive IC/AUC).

Regenerate the bundled sample:  ``python -m model_lab.fixtures``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LAB_ROOT
from .io import binance_archive as ba
from .lib import math as lm

TS0_MS = 1_700_000_000_000  # 2023-11-14T12:13:20Z — a fixed, deterministic anchor.
WINDOW_SECS = 300
SIGMA_STEP = 2.0e-4  # ~2 bps/s per-second log-return vol.
BASIS = -0.00133  # Chainlink ≈ 13.3 bps below Binance (feed-comparison finding).
START_PRICE = 60_000.0
SERIES = "BTC-5m"
UP_TOKEN = "1000000000000000001"
DOWN_TOKEN = "1000000000000000002"
FIXTURE_DIR = LAB_ROOT / "model_lab" / "fixtures"


def _price_path(n_secs: int, rng: random.Random) -> np.ndarray:
    p = np.empty(n_secs, dtype=float)
    p[0] = START_PRICE
    for t in range(1, n_secs):
        p[t] = p[t - 1] * math.exp(SIGMA_STEP * rng.gauss(0.0, 1.0))
    return p


def _write_journal(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for seq, (ts_local_ms, rec) in enumerate(records, start=1):
            fh.write(json.dumps({"seq": seq, "ts_local_ms": ts_local_ms, "rec": rec}) + "\n")


def _write_depth(path: Path, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame) + "\n")


def _market(open_ms: int, close_ms: int, strike: float) -> dict:
    return {
        "window": {"series": SERIES, "open_time": open_ms},
        "event_slug": f"btc-updown-5m-{open_ms // 1000}",
        "condition_id": "0x" + "ab" * 32,
        "tokens": {"up": UP_TOKEN, "down": DOWN_TOKEN},
        "close_time": close_ms,
        "strike": f"{strike:.2f}",
        "tick_size": "T001",
        "min_order_size": "5",
        "fees": {"rate": "0.07", "exponent": 1, "taker_only": True, "rebate_rate": "0.2", "enabled": True},
        "neg_risk": False,
        "resolution": {"kind": "Chainlink", "raw": "https://data.chain.link/streams/btc-usd"},
    }


def make_fixture(
    journal_dir: Path, depth_dir: Path, n_windows: int = 30, seed: int = 1234
) -> dict[str, int]:
    """Writes a synthetic journal segment + depth day-file. Returns line counts."""
    rng = random.Random(seed)
    n_secs = n_windows * WINDOW_SECS
    price = _price_path(n_secs, rng)

    # The engine σ_1s over one-second bars (one bar per emitted second).
    bar_secs = np.array([TS0_MS // 1000 + t for t in range(n_secs)], dtype=np.int64)
    sigma = lm.sigma_1s_from_bars(bar_secs, price)

    records: list[tuple[int, dict]] = []
    depth: list[dict] = []

    for w in range(n_windows):
        open_t = w * WINDOW_SECS
        close_t = open_t + WINDOW_SECS
        open_ms = TS0_MS + open_t * 1000
        close_ms = TS0_MS + close_t * 1000
        strike = float(price[open_t])
        records.append((open_ms, {"type": "window", "market": _market(open_ms, close_ms, strike), "lifecycle": "Open"}))

        for t in range(open_t, close_t):
            ts = TS0_MS + t * 1000
            s = float(price[t])
            # Binance direct mid + Chainlink ticks (exact-decimal strings).
            records.append((ts, {
                "type": "price_tick", "source": "BinanceDirect", "asset": "Btc", "kind": "Mid",
                "value": f"{s:.8f}", "ts_exchange": ts, "ts_local": ts,
            }))
            records.append((ts, {
                "type": "price_tick", "source": "ChainlinkRtds", "asset": "Btc", "kind": "Vendor",
                "value": f"{s * (1 + BASIS):.8f}", "ts_exchange": ts, "ts_local": ts,
            }))
            # Model snapshot where vol is warmed and time remains.
            tau = close_t - t
            sig = float(sigma[t])
            if math.isfinite(sig) and sig > 0 and tau >= 1:
                sigma_tau = sig * math.sqrt(tau)
                z = max(-lm.Z_CLAMP, min(lm.Z_CLAMP, math.log(s / strike) / sigma_tau))
                p_up = float(lm.norm_cdf(z))
                records.append((ts, {
                    "type": "model", "asset": "Btc",
                    "window": {"series": SERIES, "open_time": open_ms},
                    "p_up": p_up, "z": z, "sigma_1s": sig, "sigma_tau": sigma_tau,
                    "basis": BASIS * 1e4, "anchor": "BinanceCorrected", "health": "Ready",
                    "reason": "Nominal", "input_ages": {"chainlink": 0, "binance": 0}, "ts": ts,
                }))
                # Market Up-token top: the model probability plus noise, so the
                # market is a slightly *worse* predictor than the formula — the
                # calibration audit should find model Brier ≤ market Brier.
                mkt = min(0.98, max(0.02, p_up + rng.gauss(0.0, 0.05)))
                bid, ask = mkt - 0.005, mkt + 0.005
                records.append((ts, {
                    "type": "top_of_book", "token_id": UP_TOKEN,
                    "top": {
                        "bid": {"price": f"{bid:.4f}", "size": "50"},
                        "ask": {"price": f"{ask:.4f}", "size": "50"},
                        "ts": ts,
                    },
                }))
            # Depth frame: imbalance encodes the (realized) forward 5s return + noise.
            fwd = float(math.log(price[min(t + 5, n_secs - 1)] / s))
            imb = max(-0.85, min(0.85, 1500.0 * fwd + 0.05 * rng.gauss(0.0, 1.0)))
            half = s * 5e-6
            base = 10.0
            depth.append({
                "recv_ms": ts, "stream": "btcusdt@depth20@100ms",
                "data": {
                    "lastUpdateId": t,
                    "bids": [[f"{s - half:.2f}", f"{base * (1 + imb):.4f}"]],
                    "asks": [[f"{s + half:.2f}", f"{base * (1 - imb):.4f}"]],
                },
            })

        # A maker fill + the settlement at close.
        outcome = "Up" if float(price[close_t - 1]) >= strike else "Down"
        # The Resolved lifecycle event — fires for every resolved window (the
        # calibration audit's primary outcome source).
        records.append((close_ms, {
            "type": "window", "market": _market(open_ms, close_ms, strike),
            "lifecycle": {"Resolved": {"outcome": outcome}},
        }))
        records.append((open_ms + 100_000, {
            "type": "fill", "order_id": f"paper-{w}", "trade_id": None,
            "window": {"series": SERIES, "open_time": open_ms},
            "token_id": "1000000000000000001", "outcome": "Up", "side": "Buy",
            "price": "0.49", "size": "10", "liquidity": "Maker", "fee": "0",
            "ts_venue": open_ms + 100_000, "ts_local": open_ms + 100_000,
        }))
        records.append((close_ms, {
            "type": "settlement", "window": {"series": SERIES, "open_time": open_ms},
            "outcome": outcome,
            "up": {"shares": "10", "cost": "4.9"}, "down": {"shares": "0", "cost": "0"},
            "matched_pairs": "0", "pair_cost": None, "excess": "10", "merged_pairs": "0",
            "fees_paid": "0", "realized_pnl": ("5.1" if outcome == "Up" else "-4.9"), "ts": close_ms,
        }))

    records.sort(key=lambda r: r[0])
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(TS0_MS / 1000))
    day = time.strftime("%Y%m%d", time.gmtime(TS0_MS / 1000))
    _write_journal(journal_dir / f"journal-{stamp}-00000.jsonl.gz", records)
    _write_depth(depth_dir / f"binance-depth20-{day}.jsonl.gz", depth)
    return {"journal_records": len(records), "depth_frames": len(depth), "windows": n_windows}


# --- synthetic Binance aggTrades store (for the dataset stage's proxy path) --
# Two consecutive UTC days per symbol, distinct from the journal's day (so the
# reconstructed historical windows never collide with the fixture's journal
# windows). One trade per second over a contiguous block, so 1-second bars are
# dense and the engine σ warms; windows outside the block are skipped for want of
# strike/close bars — exactly the real behaviour.
AGG_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
AGG_DAYS: tuple[date, ...] = (date(2023, 11, 12), date(2023, 11, 13))
AGG_START_PRICE = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0}
AGG_BLOCK_SECS = 2 * 3600  # first two hours of each day
_AGG_DTYPES = {
    "agg_trade_id": "int64", "price": "float64", "quantity": "float64",
    "first_trade_id": "int64", "last_trade_id": "int64", "transact_time": "int64",
    "is_buyer_maker": "bool",
}


def make_aggtrades_fixture(
    hist_dir: Path,
    symbols: tuple[str, ...] = AGG_SYMBOLS,
    days: tuple[date, ...] = AGG_DAYS,
    block_secs: int = AGG_BLOCK_SECS,
    seed: int = 99,
) -> dict[str, int]:
    """Write a tiny synthetic Binance aggTrades store (one zstd-parquet per
    symbol-day, the real on-disk shape) so the dataset stage's historical proxy path
    runs end-to-end without a network download. ``transact_time`` is microseconds
    (the store's unit). Returns simple counts."""
    hist_dir = Path(hist_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    agg_id = 1
    rows_total = 0
    for si, sym in enumerate(symbols):
        rng = random.Random(seed + si)
        price = AGG_START_PRICE.get(sym, 100.0)
        for d in days:
            day_start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            rows: list[tuple] = []
            for s in range(block_secs):
                price *= math.exp(SIGMA_STEP * rng.gauss(0.0, 1.0))
                tt_us = (day_start + s) * 1_000_000
                rows.append((agg_id, price, 0.01, agg_id * 2, agg_id * 2 + 1, tt_us, s % 2 == 0))
                agg_id += 1
            df = pd.DataFrame(rows, columns=ba.AGGTRADES_COLS).astype(_AGG_DTYPES)
            df.to_parquet(
                ba.day_parquet_path(hist_dir, sym, d),
                engine="pyarrow", compression="zstd", index=False,
            )
            rows_total += len(df)
    ba.write_manifest(hist_dir, ba.build_manifest(hist_dir, list(symbols)))
    return {"symbols": len(symbols), "days": len(days), "rows": rows_total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="generate the synthetic model-lab fixture")
    parser.add_argument("--journal-dir", default=str(FIXTURE_DIR / "journal"))
    parser.add_argument("--depth-dir", default=str(FIXTURE_DIR / "depth"))
    parser.add_argument("--hist-dir", default=str(FIXTURE_DIR / "aggtrades"))
    parser.add_argument("--windows", type=int, default=30)
    args = parser.parse_args(argv)
    counts = make_fixture(Path(args.journal_dir), Path(args.depth_dir), n_windows=args.windows)
    agg = make_aggtrades_fixture(Path(args.hist_dir))
    print(f"[fixtures] {counts['journal_records']:,} journal records, "
          f"{counts['depth_frames']:,} depth frames, {counts['windows']} windows")
    print(f"[fixtures] aggTrades: {agg['rows']:,} trades over {agg['symbols']} symbols × {agg['days']} days")
    print(f"[fixtures] journal -> {args.journal_dir}")
    print(f"[fixtures] depth   -> {args.depth_dir}")
    print(f"[fixtures] hist    -> {args.hist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
