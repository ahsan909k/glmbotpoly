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
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
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
    journal_dir: Path, depth_dir: Path, n_windows: int = 30, seed: int = 1234,
    ts0_ms: int = TS0_MS,
) -> dict[str, int]:
    """Writes a synthetic journal segment + depth day-file. Returns line counts.

    ``ts0_ms`` anchors the fixture's first window (default the 2023 anchor); pass an
    overlap-era timestamp (``>= 2026-07-04``) to place it on a journal-owned day for the
    historical-dataset journal path. The default keeps existing callers byte-identical."""
    rng = random.Random(seed)
    # A SEPARATE stream for the depth-ladder / PM-book shape draws (half-spread, sizes)
    # so the primary `rng` sequence — the market-mid and imbalance noise the calibration
    # audit is tuned against — stays byte-for-byte unchanged when those draws are added.
    rng2 = random.Random(seed + 4321)
    n_secs = n_windows * WINDOW_SECS
    price = _price_path(n_secs, rng)

    # The engine σ_1s over one-second bars (one bar per emitted second).
    bar_secs = np.array([ts0_ms // 1000 + t for t in range(n_secs)], dtype=np.int64)
    sigma = lm.sigma_1s_from_bars(bar_secs, price)

    records: list[tuple[int, dict]] = []
    depth: list[dict] = []

    for w in range(n_windows):
        open_t = w * WINDOW_SECS
        close_t = open_t + WINDOW_SECS
        open_ms = ts0_ms + open_t * 1000
        close_ms = ts0_ms + close_t * 1000
        strike = float(price[open_t])
        records.append((open_ms, {"type": "window", "market": _market(open_ms, close_ms, strike), "lifecycle": "Open"}))

        for t in range(open_t, close_t):
            ts = ts0_ms + t * 1000
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
                # calibration audit should find model Brier ≤ market Brier. The mid
                # stays `mkt` (symmetric quote); a varying half-spread and sizes that
                # lean with p_up give pm_spread / pm_book_imb non-trivial variation.
                mkt = min(0.98, max(0.02, p_up + rng.gauss(0.0, 0.05)))
                hs = 0.005 * (1.0 + 0.4 * abs(rng2.gauss(0.0, 1.0)))
                bid, ask = mkt - hs, mkt + hs
                bsz = max(1.0, 50.0 + 200.0 * (p_up - 0.5) + 8.0 * rng2.gauss(0.0, 1.0))
                asz = max(1.0, 50.0 - 200.0 * (p_up - 0.5) + 8.0 * rng2.gauss(0.0, 1.0))
                records.append((ts, {
                    "type": "top_of_book", "token_id": UP_TOKEN,
                    "top": {
                        "bid": {"price": f"{bid:.4f}", "size": f"{bsz:.2f}"},
                        "ask": {"price": f"{ask:.4f}", "size": f"{asz:.2f}"},
                        "ts": ts,
                    },
                }))
            # Depth frame: a 20-level ladder whose sizes decay proportionally away from
            # the touch, so the full-ladder imbalance stays exactly `imb` (top-of-book
            # signal preserved) while the multi-level imbalances / depth slopes are
            # well-defined. imbalance encodes the (realized) forward 5s return + noise.
            fwd = float(math.log(price[min(t + 5, n_secs - 1)] / s))
            imb = max(-0.85, min(0.85, 1500.0 * fwd + 0.05 * rng.gauss(0.0, 1.0)))
            half = s * 5e-6
            base = 10.0
            step = s * 1e-6  # ~0.06 USD/level for BTC — levels stay distinct at 2 decimals
            decay = 0.85
            bids = [[f"{s - half - i * step:.2f}", f"{base * (1 + imb) * decay ** i:.4f}"]
                    for i in range(20)]
            asks = [[f"{s + half + i * step:.2f}", f"{base * (1 - imb) * decay ** i:.4f}"]
                    for i in range(20)]
            depth.append({
                "recv_ms": ts, "stream": "btcusdt@depth20@100ms",
                "data": {"lastUpdateId": t, "bids": bids, "asks": asks},
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
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts0_ms / 1000))
    day = time.strftime("%Y%m%d", time.gmtime(ts0_ms / 1000))
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


def make_predictions_fixture(
    out_dir: Path, journal_dir: Path, *, grid_secs: int = 15, edge: float = 0.15, seed: int = 7
) -> Path:
    """Write a synthetic grid predictions parquet for the evaluation harness.

    Reads the fixture journal (via the harness's self-baseline frame), samples the
    formula ``p_up`` on a ``grid_secs`` grid, and nudges each prediction ``edge``
    toward the *realized* outcome. Because the nudge always moves the probability
    toward the truth, the resulting model is provably better-calibrated than the
    formula (``model_brier < formula_brier``) — a clean numeric invariant for
    ``verify`` / tests. This is an oracle-nudge fixture (it peeks at the label): it
    is **only** for exercising the harness, never a real model. Keyed
    ``(series, window_open_ms, sample_ts_ms, p_up)``. Returns the parquet path.
    """
    from .config import Paths
    from . import eval_harness as eh

    _ = seed  # deterministic (no randomness) — kept for signature symmetry.
    jpaths = Paths(journal_dir=Path(journal_dir), depth_dir=Path(journal_dir), out_dir=Path(out_dir))
    df, _counts = eh.build_self_baseline_frame(jpaths)
    step = grid_secs * 1000
    grid = df[((df["ts"] - df["open_time"]) % step == 0)].copy()
    push = edge * (2.0 * grid["outcome_up"].to_numpy(dtype=float) - 1.0)  # +edge if Up, −edge if Down
    p_up = np.clip(grid["formula"].to_numpy(dtype=float) + push, 0.02, 0.98)
    out = pd.DataFrame(
        {
            "series": grid["series"].astype(str).to_numpy(),
            "window_open_ms": grid["open_time"].astype("int64").to_numpy(),
            "sample_ts_ms": grid["ts"].astype("int64").to_numpy(),
            "p_up": p_up.astype(float),
        }
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.parquet"
    out.to_parquet(path, engine="pyarrow", index=False)
    return path


def make_short_oos_fixture(
    out_dir: Path, journal_dir: Path, *, grid_secs: int = 10, edge: float = 0.25,
    oracle: bool = True, seed: int = 7,
) -> Path:
    """Write a synthetic ``learn_short_gbt`` OOS parquet so the **money judge** can run
    fixture-only, with no LightGBM.

    Mirrors the real ``oos_all_fwd10_full.parquet`` (columns of ``_OOS_COLS_SHORT``):
    keyed ``(series, window_open_ms, sample_ts_ms)`` with a model ``p_up``. With
    ``oracle=True`` the probability leans ``edge`` toward each window's realized
    ``outcome_up`` (``p_up = 0.5 ± edge``), so the fired side is the winning side — the
    signal is provably profitable when marked to resolution (a clean money-judge
    invariant). With ``oracle=False`` the lean is a seeded coin flip uncorrelated with
    the outcome (the "random signal bleeds fees" negative case). Writes to
    ``out_dir/learn_short_gbt/oos_all_fwd10_full.parquet``; returns the path.
    """
    from . import eval_harness as eh
    from .config import Paths

    jpaths = Paths(journal_dir=Path(journal_dir), depth_dir=Path(journal_dir), out_dir=Path(out_dir))
    df, _counts = eh.build_self_baseline_frame(jpaths)
    step = grid_secs * 1000
    grid = df[((df["ts"] - df["open_time"]) % step == 0)].copy()
    outcome = grid["outcome_up"].to_numpy(dtype=float)
    if oracle:
        lean = 2.0 * outcome - 1.0  # +1 toward Up windows, −1 toward Down
    else:
        rng = np.random.default_rng(seed)
        lean = rng.choice([-1.0, 1.0], size=len(grid))
    p_up = np.clip(0.5 + edge * lean, 0.02, 0.98)
    tau = grid["tau"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "series": grid["series"].astype(str).to_numpy(),
        "window_open_ms": grid["open_time"].astype("int64").to_numpy(),
        "sample_ts_ms": grid["ts"].astype("int64").to_numpy(),
        "p_up": p_up.astype(float),
        "y_true": outcome.astype("int64"),
        "p_up_model": grid["formula"].to_numpy(dtype=float),
        "sigma_1s": np.full(len(grid), SIGMA_STEP, dtype=float),
        "tau_secs": tau,
        "dur": grid["dur"].to_numpy(dtype=float),
        "label_source": np.array(["chainlink"] * len(grid)),
        "depth_feat_covered": np.ones(len(grid), dtype=bool),
        "pm_feat_covered": np.ones(len(grid), dtype=bool),
    })
    dest = Path(out_dir) / "learn_short_gbt"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "oos_all_fwd10_full.parquet"
    out.to_parquet(path, engine="pyarrow", index=False)
    return path


# --------------------------------------------------------------------------- #
# Telonex trial fixture (offline validation of the telonex_ingest/validate stages)
# --------------------------------------------------------------------------- #

TX_DAY = date(2026, 7, 5)
TX_BINANCE_OFFSET_MS = 40   # telonex.timestamp_us/1000 = our.ts_exchange + this
TX_CLOB_OFFSET_MS = 25      # telonex book top ts = our top.ts + this


def _tx_levels(best: float, step: float, n: int, side: str) -> list[dict]:
    out = []
    for j in range(n):
        px = best - j * step if side == "bid" else best + j * step
        px = min(0.999, max(0.001, px))
        out.append({"price": f"{px:.3f}", "size": f"{100 + j}.0"})
    return out


def make_telonex_fixture(
    telonex_dir: Path, journal_dir: Path, *, day: date = TX_DAY, n_windows: int = 5,
    rows_per_window: int = 60, seed: int = 4242,
) -> dict:
    """Write a synthetic Telonex trial store + a matching journal so the ingest/validate
    stages run end-to-end with **no network**.

    By construction: book snapshots are sub-second (cadence PASS) with 20 levels/side
    (depth PASS); Up/Down tops are exact mirrors (pair check PASS); and our journal's
    Binance trades + CLOB tops are the SAME series shifted by a constant offset
    (:data:`TX_BINANCE_OFFSET_MS` / :data:`TX_CLOB_OFFSET_MS`), so the external clock
    check recovers a stable, correctable offset. Returns the trial metadata.
    """
    telonex_dir = Path(telonex_dir)
    day_str = day.isoformat()
    day_start_s = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    day_start_ms = day_start_s * 1000
    outcomes = ["Up", "Down"]
    channels = ["trades", "quotes", "book_snapshot_full"]
    symbol = "btcusdt"

    def poly_header(slug: str, asset_id: str, market_id: str, outcome: str, ts_us: int) -> dict:
        return {"timestamp_us": ts_us, "local_timestamp_us": ts_us + 1500, "exchange": "polymarket",
                "market_id": market_id, "slug": slug, "asset_id": asset_id, "outcome": outcome}

    windows_meta: list[dict] = []
    journal_records: list[tuple[int, dict]] = []

    for i in range(n_windows):
        open_epoch = day_start_s + 43_200 + i * 300  # around noon UTC
        open_ms = open_epoch * 1000
        slug = f"btc-updown-5m-{open_epoch}"
        up_asset = f"1085336978904702759652678{open_epoch}"
        down_asset = f"6918359367722649995132085{open_epoch}"
        market_id = "0x" + f"{open_epoch:064x}"[:64]
        windows_meta.append({"slug": slug, "open_epoch": open_epoch, "open_ms": open_ms,
                             "up_asset_id": up_asset, "market_id": market_id})

        up_book, down_book, up_trades, down_trades = [], [], [], []
        up_quotes, down_quotes = [], []
        for k in range(rows_per_window):
            t_ms = open_ms + 1000 + k * 200  # 200 ms cadence
            p = 0.5 + 0.02 * math.sin((i + 1) * 0.3 + k * 0.05)
            best_bid, best_ask = round(p, 3), round(p + 0.01, 3)
            # Telonex book: ts shifted +CLOB offset; 20 levels each side.
            tel_ts_us = (t_ms + TX_CLOB_OFFSET_MS) * 1000
            up_book.append({**poly_header(slug, up_asset, market_id, "Up", tel_ts_us),
                            "bids": _tx_levels(best_bid, 0.001, 20, "bid"),
                            "asks": _tx_levels(best_ask, 0.001, 20, "ask")})
            d_bid, d_ask = round(1 - best_ask, 3), round(1 - best_bid, 3)
            down_book.append({**poly_header(slug, down_asset, market_id, "Down", tel_ts_us),
                              "bids": _tx_levels(d_bid, 0.001, 20, "bid"),
                              "asks": _tx_levels(d_ask, 0.001, 20, "ask")})
            # Flat top-of-book quotes (the channel the clock + mirror checks read).
            up_quotes.append({**poly_header(slug, up_asset, market_id, "Up", tel_ts_us),
                              "bid_price": f"{best_bid:.3f}", "bid_size": "500.0",
                              "ask_price": f"{best_ask:.3f}", "ask_size": "400.0"})
            down_quotes.append({**poly_header(slug, down_asset, market_id, "Down", tel_ts_us),
                                "bid_price": f"{d_bid:.3f}", "bid_size": "500.0",
                                "ask_price": f"{d_ask:.3f}", "ask_size": "400.0"})
            # Our journal CLOB top for the Up token, at the un-shifted time.
            journal_records.append((t_ms, {
                "type": "top_of_book", "token_id": up_asset,
                "top": {"bid": {"price": f"{best_bid:.3f}", "size": "500.0"},
                        "ask": {"price": f"{best_ask:.3f}", "size": "400.0"}, "ts": t_ms}}))
            if k % 12 == 0:  # a few trades per window
                tid = f"{slug}-{k}"
                up_trades.append({**poly_header(slug, up_asset, market_id, "Up", tel_ts_us),
                                  "price": f"{best_bid:.3f}", "size": "10.0", "side": "buy",
                                  "trade_id": tid, "origin_asset_id": up_asset})
                down_trades.append({**poly_header(slug, down_asset, market_id, "Down", tel_ts_us),
                                    "price": f"{d_ask:.3f}", "size": "10.0", "side": "sell",
                                    "trade_id": tid + "-d", "origin_asset_id": up_asset})
        _write_tx_parquet(tx_poly_path(telonex_dir, "book_snapshot_full", slug, "Up", day_str), up_book)
        _write_tx_parquet(tx_poly_path(telonex_dir, "book_snapshot_full", slug, "Down", day_str), down_book)
        _write_tx_parquet(tx_poly_path(telonex_dir, "trades", slug, "Up", day_str), up_trades)
        _write_tx_parquet(tx_poly_path(telonex_dir, "trades", slug, "Down", day_str), down_trades)
        _write_tx_parquet(tx_poly_path(telonex_dir, "quotes", slug, "Up", day_str), up_quotes)
        _write_tx_parquet(tx_poly_path(telonex_dir, "quotes", slug, "Down", day_str), down_quotes)

    # Binance: trades (300, unique cents) + book_snapshot_25 (flattened, 25 levels).
    bin_trades, bin_book = [], []
    for i in range(300):
        t_ms = day_start_ms + 43_200_000 + i * 1000
        price = 60_000.0 + i * 0.5
        bin_trades.append({"timestamp_us": (t_ms + TX_BINANCE_OFFSET_MS) * 1000, "local_timestamp_us": (t_ms + TX_BINANCE_OFFSET_MS) * 1000 + 800,
                           "exchange": "binance", "market_id": symbol, "slug": symbol, "asset_id": symbol,
                           "outcome": "", "price": f"{price:.2f}", "size": "0.01", "side": "buy" if i % 2 else "sell",
                           "trade_id": str(1000 + i), "origin_asset_id": symbol})
        journal_records.append((t_ms, {"type": "price_tick", "source": "BinanceDirect", "asset": "Btc",
                                       "kind": "Trade", "value": f"{price:.2f}", "ts_exchange": t_ms, "ts_local": t_ms + 5}))
    for i in range(60):
        t_ms = day_start_ms + 43_200_000 + i * 100
        row = {"timestamp_us": t_ms * 1000, "local_timestamp_us": t_ms * 1000 + 300, "exchange": "binance",
               "market_id": symbol, "slug": symbol, "asset_id": symbol, "outcome": ""}
        mid = 60_000.0 + i * 0.5
        for j in range(25):
            row[f"bid_price_{j}"] = f"{mid - 0.01 * (j + 1):.2f}"
            row[f"bid_size_{j}"] = f"{1.0 + j}"
            row[f"ask_price_{j}"] = f"{mid + 0.01 * (j + 1):.2f}"
            row[f"ask_size_{j}"] = f"{1.0 + j}"
        bin_book.append(row)
    _write_tx_parquet(tx_binance_path(telonex_dir, "trades", symbol, day_str), bin_trades)
    _write_tx_parquet(tx_binance_path(telonex_dir, "book_snapshot_25", symbol, day_str), bin_book)

    # Tiny markets catalog (raw/polymarket/markets.parquet) so the coverage + date-range
    # checks — which read the catalog — run offline. Two windows per needed series.
    cat_rows = []
    for a in ("btc", "eth"):
        for dr, step in (("5m", 300), ("15m", 900), ("1h", 3600)):
            for k in range(2):
                ep = day_start_s + 43_200 + k * step
                cat_rows.append({
                    "slug": f"{a}-updown-{dr}-{ep}", "outcome_0": "Up", "outcome_1": "Down",
                    "asset_id_0": f"{a}-{dr}-{ep}-up", "asset_id_1": f"{a}-{dr}-{ep}-dn",
                    "market_id": f"0x{ep:x}", "book_snapshot_full_from": (day - timedelta(days=1)).isoformat(),
                    "book_snapshot_25_from": (day - timedelta(days=1)).isoformat(),
                    "trades_from": day_str, "quotes_from": (day - timedelta(days=1)).isoformat()})
    cat_path = telonex_dir / "raw" / "polymarket" / "markets.parquet"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cat_rows).to_parquet(cat_path, engine="pyarrow", index=False)

    # Journal segment (filename date == day so the day-bounded reader picks it up).
    jpath = Path(journal_dir) / f"journal-{day.strftime('%Y%m%d')}-120000-00000.jsonl.gz"
    journal_records.sort(key=lambda r: r[0])
    _write_journal(jpath, journal_records)

    trial_meta = {"day": day_str, "series": "btc-updown-5m", "outcomes": outcomes, "channels": channels,
                  "binance_symbol": symbol, "binance_channels": ["trades", "book_snapshot_25"],
                  "windows": windows_meta}
    _write_tx_manifest(telonex_dir, trial_meta)
    return {"windows": len(windows_meta), "day": day_str, "journal_records": len(journal_records),
            "binance_offset_ms": TX_BINANCE_OFFSET_MS, "clob_offset_ms": TX_CLOB_OFFSET_MS,
            "trial": trial_meta}


def _write_tx_parquet(path: Path, rows: list[dict]) -> None:
    from .io import telonex as _tx  # avoid import cycle at module load

    _ = _tx
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def tx_poly_path(telonex_dir: Path, channel: str, slug: str, outcome: str, day_str: str) -> Path:
    return Path(telonex_dir) / "raw" / "polymarket" / channel / f"{slug}-{outcome}-{day_str}.parquet"


def tx_binance_path(telonex_dir: Path, channel: str, symbol: str, day_str: str) -> Path:
    return Path(telonex_dir) / "raw" / "binance" / channel / f"{symbol}-{day_str}.parquet"


def _write_tx_manifest(telonex_dir: Path, trial_meta: dict) -> None:
    from .io import telonex as _tx

    _tx.write_manifest(telonex_dir, _tx.build_manifest(telonex_dir, extra=trial_meta))


def telonex_offline_fetch(*, day: date = TX_DAY, live_slugs=None, remaining: int = 1000,
                          download_body: bytes = b"telonex-parquet-bytes",
                          fail_download_status: int | None = None, restrict_exchange: str | None = None):
    """A ``telonex.Fetch`` seam answering availability + downloads offline (no network).

    ``live_slugs=None`` → any ``*-updown-*`` slug whose epoch falls on ``day`` resolves
    (permissive, for coverage probing); a set restricts which windows have data.
    ``fail_download_status`` (e.g. 403) makes every download raise that status — for the
    download-limit / free-tier abort test.
    """
    import io as _io
    import json as _json
    import urllib.parse as _up

    from .io import telonex as _tx

    dfrom = (day - timedelta(days=1)).isoformat()
    dto = (day + timedelta(days=1)).isoformat()
    poly_channels = {c: {"from_date": dfrom, "to_date": dto}
                     for c in ("trades", "quotes", "book_snapshot_5", "book_snapshot_25", "book_snapshot_full")}
    bin_channels = {c: {"from_date": "2026-02-06", "to_date": dto}
                    for c in ("trades", "quotes", "book_snapshot_5", "book_snapshot_25")}

    def _is_live(slug: str) -> bool:
        if live_slugs is not None:
            return slug in live_slugs
        m = re.match(r"^[a-z]+-updown-(5m|15m|1h)-(\d+)$", slug)
        if not m:
            return False
        epoch = int(m.group(2))
        start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
        return start <= epoch < start + 86_400

    def fetch(url: str, headers=None, follow_redirects: bool = True):
        parsed = _up.urlparse(url)
        q = {k: v[0] for k, v in _up.parse_qs(parsed.query).items()}
        path = parsed.path
        if url.endswith("|s3"):
            return _tx.TelonexResponse(200, {}, _io.BytesIO(download_body))
        if "/availability/polymarket" in path:
            slug = q.get("slug", "")
            if not _is_live(slug):
                raise _tx.TelonexHTTPError(404, {}, url)
            payload = {"exchange": "polymarket", "asset_id": f"asset-{slug}", "market_id": f"0x{abs(hash(slug)):x}",
                       "slug": slug, "outcome": q.get("outcome", "Up"), "outcome_id": 0, "channels": poly_channels}
            return _tx.TelonexResponse(200, {}, _io.BytesIO(_json.dumps(payload).encode()))
        if "/availability/binance" in path:
            sym = q.get("slug", "btcusdt")
            payload = {"exchange": "binance", "asset_id": sym, "market_id": sym, "slug": sym,
                       "outcome": None, "outcome_id": 0, "channels": bin_channels}
            return _tx.TelonexResponse(200, {}, _io.BytesIO(_json.dumps(payload).encode()))
        if "/downloads/" in path or "/datasets/" in path:
            if fail_download_status is not None:
                raise _tx.TelonexHTTPError(fail_download_status, {"X-Downloads-Remaining": "0"}, url)
            if restrict_exchange and f"/downloads/{restrict_exchange}/" in path:
                body = b'{"detail":"Download not allowed: exchange_restricted. Upgrade to Pro for unlimited access."}'
                raise _tx.TelonexHTTPError(403, {"X-Downloads-Remaining": "0"}, url, body)
            if follow_redirects:  # a client that follows would land on S3 directly
                return _tx.TelonexResponse(200, {}, _io.BytesIO(download_body))
            return _tx.TelonexResponse(302, {"Location": url + "|s3", "X-Downloads-Remaining": str(remaining)},
                                       _io.BytesIO(b""))
        raise _tx.TelonexHTTPError(404, {}, url)

    return fetch


# --------------------------------------------------------------------------- #
# Historical fixture (Telonex + aggTrades) for the historical ingest path
# --------------------------------------------------------------------------- #
# A coherent, self-consistent set exercising the whole historical DAG offline:
# aggTrades (mid/vol/strike backbone), Telonex book_snapshot_25 (depth), Telonex PM
# quotes (PM book + quote-convergence resolution) — on a Telonex-owned day (before the
# journal-owned cutoff), so no journal is involved. Windows 0..N-1 are resolvable (the
# Up mid converges to the winner), plus one deliberately-ambiguous window (stuck near
# 0.5, excluded by resolution) and one coverage-excluded window (listed in a synthetic
# coverage.json missing_windows). Binance-anchored and quote-convergence outcomes AGREE
# by construction on the resolvable windows.

HIST_FIX_DAY = date(2026, 5, 1)  # Telonex-owned (< JOURNAL_OWNED_FROM 2026-07-04)
HIST_FIX_SYMBOLS = ("BTCUSDT", "ETHUSDT")
HIST_FIX_START_PRICE = {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0}
_HIST_BOOK_LEVELS = 25


def _hist_price_path(n_secs: int, start: float, rng: random.Random) -> np.ndarray:
    p = np.empty(n_secs, dtype=float)
    p[0] = start
    for t in range(1, n_secs):
        p[t] = p[t - 1] * math.exp(SIGMA_STEP * rng.gauss(0.0, 1.0))
    return p


def _write_aggtrades_day(hist_dir: Path, symbol: str, day: date,
                         secs: np.ndarray, prices: np.ndarray, start_id: int) -> int:
    """Write one aggTrades day-parquet (one trade per given second). Returns next id."""
    rows: list[tuple] = []
    aid = start_id
    for s, px in zip(secs.tolist(), prices.tolist()):
        tt_us = int(s) * 1_000_000
        rows.append((aid, float(px), 0.01, aid * 2, aid * 2 + 1, tt_us, aid % 2 == 0))
        aid += 1
    df = pd.DataFrame(rows, columns=ba.AGGTRADES_COLS).astype(_AGG_DTYPES)
    df.to_parquet(ba.day_parquet_path(hist_dir, symbol, day), engine="pyarrow",
                  compression="zstd", index=False)
    return aid


def _hist_book_rows(symbol_prices: dict[int, float], levels: int = _HIST_BOOK_LEVELS) -> list[dict]:
    """Telonex book_snapshot_25 rows at ~200 ms cadence from a per-second price map — a
    25-level ladder around the mid with sizes decaying away from the touch (well-defined
    depth features)."""
    rows: list[dict] = []
    for sec in sorted(symbol_prices):
        mid = symbol_prices[sec]
        half = mid * 5e-6
        step = mid * 1e-6
        for frac in (0, 200, 400, 600, 800):  # 5 snapshots per second (200 ms)
            t_ms = sec * 1000 + frac
            row = {"timestamp_us": t_ms * 1000, "local_timestamp_us": t_ms * 1000 + 300,
                   "exchange": "binance", "market_id": symbol_prices.get("_sym", ""),
                   "slug": "", "asset_id": "", "outcome": ""}
            for j in range(levels):
                row[f"bid_price_{j}"] = f"{mid - half - j * step:.2f}"
                row[f"bid_size_{j}"] = f"{10.0 * 0.9 ** j:.4f}"
                row[f"ask_price_{j}"] = f"{mid + half + j * step:.2f}"
                row[f"ask_size_{j}"] = f"{9.0 * 0.9 ** j:.4f}"
            rows.append(row)
    return rows


def _pm_quote_rows(slug: str, token: str, outcome: str, open_ms: int, close_ms: int,
                   up_mid_fn) -> list[dict]:
    """PM quote rows (~1 s cadence) for one token over [open, close]. ``up_mid_fn(frac)``
    gives the Up-token mid at fraction ``frac ∈ [0,1]`` of the window; the Down token is
    the mirror (``1 − up_mid``)."""
    rows: list[dict] = []
    n = (close_ms - open_ms) // 1000
    for k in range(n + 1):
        ts_ms = open_ms + k * 1000
        frac = k / max(1, n)
        um = up_mid_fn(frac)
        mid = um if outcome == "Up" else (1.0 - um)
        mid = min(0.995, max(0.005, mid))
        bid, ask = max(0.001, mid - 0.005), min(0.999, mid + 0.005)
        rows.append({
            "timestamp_us": ts_ms * 1000, "local_timestamp_us": ts_ms * 1000 + 500,
            "exchange": "polymarket", "market_id": "0x" + slug.encode().hex()[:16],
            "slug": slug, "asset_id": token, "outcome": outcome,
            "bid_price": f"{bid:.3f}", "bid_size": "100", "ask_price": f"{ask:.3f}", "ask_size": "100",
        })
    return rows


def _pm_trade_rows(slug: str, token: str, outcome: str, open_ms: int, close_ms: int,
                   up_mid_fn) -> list[dict]:
    """PM trade prints (~every 5 s) for one token — a sell-aggressor print just below the mid
    (large size, so it sweeps the displayed bid and fills a resting maker buy) and a buy-aggressor
    print above (which must NOT fill our buys). Exercises ``read_pm_trades`` + the maker fill path.
    ``side`` is the taker/aggressor side (Polymarket convention)."""
    rows: list[dict] = []
    n = (close_ms - open_ms) // 1000
    for k in range(0, n + 1, 5):
        ts_ms = open_ms + k * 1000
        um = up_mid_fn(k / max(1, n))
        mid = um if outcome == "Up" else (1.0 - um)
        mid = min(0.995, max(0.005, mid))
        sell_px = max(0.002, mid - 0.02)
        buy_px = min(0.998, mid + 0.02)
        for px, side in ((sell_px, "sell"), (buy_px, "buy")):
            rows.append({
                "timestamp_us": ts_ms * 1000, "local_timestamp_us": ts_ms * 1000 + 300,
                "exchange": "polymarket", "market_id": "0x" + slug.encode().hex()[:16],
                "slug": slug, "asset_id": token, "outcome": outcome,
                "price": f"{px:.3f}", "size": "500", "side": side, "trade_id": f"{slug}-{outcome}-{k}",
            })
    return rows


def make_historical_fixture(
    telonex_dir: Path, hist_dir: Path, *, day: date = HIST_FIX_DAY, n_resolvable: int = 4,
    seed: int = 5150,
) -> dict:
    """Write a self-consistent Telonex + aggTrades historical store (no journal) exercising
    the historical builder / labels / backtest. Returns a summary with the per-window
    expected outcomes and the excluded-window slugs."""
    telonex_dir, hist_dir = Path(telonex_dir), Path(hist_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    day_str = day.isoformat()
    day_start_s = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    first_open_s = day_start_s + 43_200  # noon UTC
    n_windows = n_resolvable + 2  # + 1 ambiguous + 1 coverage-excluded
    warmup_s = 600
    last_close_s = first_open_s + n_windows * 300
    block_lo, block_hi = first_open_s - warmup_s, last_close_s + 120

    result: dict = {"day": day_str, "windows": [], "excluded_slugs": [], "symbols": list(HIST_FIX_SYMBOLS)}
    agg_id = 1
    all_missing: list[str] = []
    res_map: dict = {}          # (series, open_ms) -> {outcome, condition_id} (official)
    catalog_rows: list[dict] = []

    for symbol in HIST_FIX_SYMBOLS:
        asset = "btc" if symbol.startswith("BTC") else "eth"
        prefix = f"{asset}-updown-5m"
        rng = random.Random(seed + (0 if asset == "btc" else 7))
        secs = np.arange(block_lo, block_hi, dtype=np.int64)
        prices = _hist_price_path(len(secs), HIST_FIX_START_PRICE[symbol], rng)
        price_by_sec = {int(s): float(p) for s, p in zip(secs, prices)}
        agg_id = _write_aggtrades_day(hist_dir, symbol, day, secs, prices, agg_id)
        # Telonex book_snapshot_25 for the block.
        book_prices = {s: price_by_sec[s] for s in range(block_lo, block_hi) if s in price_by_sec}
        book_prices["_sym"] = symbol  # carried into market_id (harmless)
        book_rows = _hist_book_rows({k: v for k, v in book_prices.items() if isinstance(k, int)})
        _write_tx_parquet(tx_binance_path(telonex_dir, "book_snapshot_25", symbol.lower(), day_str), book_rows)

        for w in range(n_windows):
            open_s = first_open_s + w * 300
            close_s = open_s + 300
            open_ms, close_ms = open_s * 1000, close_s * 1000
            slug = f"{prefix}-{open_s}"
            up_token = f"{slug}-up"
            down_token = f"{slug}-dn"
            strike = price_by_sec[open_s]
            close_price = price_by_sec[close_s - 1]
            outcome = "Up" if close_price >= strike else "Down"
            ambiguous = (w == n_resolvable)          # index n_resolvable → stuck near 0.5
            excluded = (w == n_resolvable + 1)        # last → listed missing in coverage.json

            if ambiguous:
                def up_mid_fn(frac, rng=rng):
                    return 0.5 + 0.01 * math.sin(frac * 6.0)
            else:
                # ramp from 0.5 to ~0.99 (Up wins) / ~0.01 (Down wins) by close
                target = 0.99 if outcome == "Up" else 0.01
                def up_mid_fn(frac, target=target):
                    return 0.5 + (target - 0.5) * min(1.0, frac ** 1.5)

            up_rows = _pm_quote_rows(slug, up_token, "Up", open_ms, close_ms, up_mid_fn)
            down_rows = _pm_quote_rows(slug, down_token, "Down", open_ms, close_ms, up_mid_fn)
            _write_tx_parquet(tx_poly_path(telonex_dir, "quotes", slug, "Up", day_str), up_rows)
            _write_tx_parquet(tx_poly_path(telonex_dir, "quotes", slug, "Down", day_str), down_rows)
            # PM trade tapes (Up+Down) — the maker fill driver for backtest_maker_core.
            _write_tx_parquet(tx_poly_path(telonex_dir, "trades", slug, "Up", day_str),
                              _pm_trade_rows(slug, up_token, "Up", open_ms, close_ms, up_mid_fn))
            _write_tx_parquet(tx_poly_path(telonex_dir, "trades", slug, "Down", day_str),
                              _pm_trade_rows(slug, down_token, "Down", open_ms, close_ms, up_mid_fn))
            # Official resolution (from the price path) for EVERY window — the on-chain
            # resolution exists regardless of quote convergence. condition_id unique per
            # (asset, open) — real markets never share a condition id.
            series_key = f"{asset.upper()}-5m"
            tag = "b" if asset == "btc" else "e"
            cid = "0x" + (tag + f"{open_s:x}" + "a" * 64)[:64]
            res_map[(series_key, open_ms)] = {"outcome": outcome, "condition_id": cid}
            catalog_rows.append({"slug": slug, "market_id": cid, "status": "resolved",
                                 "result_id": "0" if outcome == "Up" else "1",
                                 "outcome_0": "Up", "outcome_1": "Down", "settled_at_us": close_ms * 1000})
            if excluded:
                all_missing += [f"{slug}/Up/quotes", f"{slug}/Down/quotes"]
                result["excluded_slugs"].append(slug)
            if asset == "btc":
                result["windows"].append({
                    "slug": slug, "open_ms": open_ms, "outcome": outcome,
                    "ambiguous": ambiguous, "coverage_excluded": excluded,
                })

    # A synthetic coverage.json so the missing-window exclusion path is exercised.
    cov = {"schema_version": 1, "series": {}}
    for asset in ("btc", "eth"):
        cov["series"][f"{asset}-updown-5m"] = {
            "days": {day_str: {"missing_windows": [m for m in all_missing if m.startswith(asset)]}}
        }
    cov_path = telonex_dir / "out_coverage.json"  # placed under out/ by the caller; see below
    _ = cov_path
    # A tiny markets catalog so hr.load_resolution_map works offline (the official source).
    cat_path = telonex_dir / "raw" / "polymarket" / "markets.parquet"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog_rows).to_parquet(cat_path, engine="pyarrow", index=False)
    result["coverage"] = cov
    result["res_map"] = res_map
    result["n_windows_per_symbol"] = n_windows
    return result


def _parity_ladder(mid: float, imb: float, levels: int = _HIST_BOOK_LEVELS):
    """A shared 25-level ladder around ``mid`` with a fixed top-of-book imbalance ``imb``
    and sizes decaying away from the touch — emitted IDENTICALLY as recorder depth20 (top
    20) and Telonex book_snapshot_25 (all 25), so the depth features match by construction."""
    half = mid * 5e-6
    step = mid * 1e-6
    base = 10.0
    decay = 0.85
    bids = [(mid - half - j * step, base * (1 + imb) * decay ** j) for j in range(levels)]
    asks = [(mid + half + j * step, base * (1 - imb) * decay ** j) for j in range(levels)]
    return bids, asks


def make_overlap_parity_fixture(
    telonex_dir: Path, hist_dir: Path, journal_dir: Path, depth_dir: Path, *,
    day: date = date(2026, 7, 4), n_windows: int = 4, seed: int = 909,
) -> dict:
    """Write a coherent recorder + Telonex store on the SAME overlap day, both derived from
    one shared price path and one shared ladder, so the telonex-vs-recorder feature-parity
    guard can compare them (and pass). Returns the per-window outcomes."""
    telonex_dir, hist_dir, journal_dir, depth_dir = (
        Path(telonex_dir), Path(hist_dir), Path(journal_dir), Path(depth_dir))
    hist_dir.mkdir(parents=True, exist_ok=True)
    day_str = day.isoformat()
    day_start_s = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    first_open_s = day_start_s + 43_200
    warmup_s = 120
    block_lo, block_hi = first_open_s - warmup_s, first_open_s + n_windows * 300
    rng = random.Random(seed)
    secs = np.arange(block_lo, block_hi, dtype=np.int64)
    price = _hist_price_path(len(secs), 60_000.0, rng)
    price_by_sec = {int(s): float(p) for s, p in zip(secs, price)}
    imb_by_sec = {int(s): max(-0.8, min(0.8, 0.5 * math.sin(i * 0.03))) for i, s in enumerate(secs)}

    records: list[tuple[int, dict]] = []
    depth: list[dict] = []
    bin_book: list[dict] = []
    windows: list[dict] = []
    symbol = "btcusdt"

    # aggTrades (Telonex-side mid/vol) — full block, one trade/sec.
    _write_aggtrades_day(hist_dir, "BTCUSDT", day, secs, price, 1)

    # continuous Binance/Chainlink ticks + shared ladder (both feeds) over the whole block.
    bar_secs = secs.copy()
    sigma = lm.sigma_1s_from_bars(bar_secs, price)
    for i, s in enumerate(secs.tolist()):
        ts = s * 1000
        mid = price_by_sec[s]
        records.append((ts, {"type": "price_tick", "source": "BinanceDirect", "asset": "Btc",
                             "kind": "Mid", "value": f"{mid:.8f}", "ts_exchange": ts, "ts_local": ts}))
        records.append((ts, {"type": "price_tick", "source": "ChainlinkRtds", "asset": "Btc",
                             "kind": "Vendor", "value": f"{mid * (1 + BASIS):.8f}", "ts_exchange": ts, "ts_local": ts}))
        bids, asks = _parity_ladder(mid, imb_by_sec[s])
        depth.append({"recv_ms": ts, "stream": "btcusdt@depth20@100ms",
                      "data": {"lastUpdateId": s,
                               "bids": [[f"{px:.2f}", f"{sz:.4f}"] for px, sz in bids[:20]],
                               "asks": [[f"{px:.2f}", f"{sz:.4f}"] for px, sz in asks[:20]]}})
        row = {"timestamp_us": ts * 1000, "local_timestamp_us": ts * 1000 + 200, "exchange": "binance",
               "market_id": symbol, "slug": symbol, "asset_id": symbol, "outcome": ""}
        for j in range(_HIST_BOOK_LEVELS):
            row[f"bid_price_{j}"] = f"{bids[j][0]:.2f}"; row[f"bid_size_{j}"] = f"{bids[j][1]:.4f}"
            row[f"ask_price_{j}"] = f"{asks[j][0]:.2f}"; row[f"ask_size_{j}"] = f"{asks[j][1]:.4f}"
        bin_book.append(row)

    for w in range(n_windows):
        open_s = first_open_s + w * 300
        close_s = open_s + 300
        open_ms, close_ms = open_s * 1000, close_s * 1000
        strike = price_by_sec[open_s]
        outcome = "Up" if price_by_sec[close_s - 1] >= strike else "Down"
        slug = f"btc-updown-5m-{open_s}"
        windows.append({"slug": slug, "open_ms": open_ms, "outcome": outcome})
        records.append((open_ms, {"type": "window", "market": _market(open_ms, close_ms, strike), "lifecycle": "Open"}))
        for s in range(open_s, close_s):
            ts = s * 1000
            sig = float(sigma[s - block_lo]) if 0 <= s - block_lo < len(sigma) else float("nan")
            mid = price_by_sec[s]
            if math.isfinite(sig) and sig > 0:
                tau = close_s - s
                z = max(-lm.Z_CLAMP, min(lm.Z_CLAMP, math.log(mid / strike) / (sig * math.sqrt(tau))))
                records.append((ts, {"type": "model", "asset": "Btc",
                                     "window": {"series": SERIES, "open_time": open_ms},
                                     "p_up": float(lm.norm_cdf(z)), "z": z, "sigma_1s": sig,
                                     "sigma_tau": sig * math.sqrt(tau), "basis": BASIS * 1e4,
                                     "anchor": "BinanceCorrected", "health": "Ready", "reason": "Nominal", "ts": ts}))
        records.append((close_ms, {"type": "window", "market": _market(open_ms, close_ms, strike),
                                   "lifecycle": {"Resolved": {"outcome": outcome}}}))
        # Telonex quotes converging to the same outcome.
        up_tok, dn_tok = f"{slug}-up", f"{slug}-dn"
        target = 0.99 if outcome == "Up" else 0.01

        def up_mid_fn(frac, target=target):
            return 0.5 + (target - 0.5) * min(1.0, frac ** 1.5)
        _write_tx_parquet(tx_poly_path(telonex_dir, "quotes", slug, "Up", day_str),
                          _pm_quote_rows(slug, up_tok, "Up", open_ms, close_ms, up_mid_fn))
        _write_tx_parquet(tx_poly_path(telonex_dir, "quotes", slug, "Down", day_str),
                          _pm_quote_rows(slug, dn_tok, "Down", open_ms, close_ms, up_mid_fn))

    _write_tx_parquet(tx_binance_path(telonex_dir, "book_snapshot_25", symbol, day_str), bin_book)
    # markets catalog (official resolution) matching the price-path outcomes + journal.
    catalog_rows = [{"slug": w["slug"], "market_id": "0x" + f"{w['open_ms'] // 1000:x}{'cd' * 32}"[:64],
                     "status": "resolved", "result_id": "0" if w["outcome"] == "Up" else "1",
                     "outcome_0": "Up", "outcome_1": "Down", "settled_at_us": (w["open_ms"] + 300_000) * 1000}
                    for w in windows]
    cat_path = telonex_dir / "raw" / "polymarket" / "markets.parquet"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog_rows).to_parquet(cat_path, engine="pyarrow", index=False)
    records.sort(key=lambda r: r[0])
    _write_journal(journal_dir / f"journal-{day.strftime('%Y%m%d')}-120000-00000.jsonl.gz", records)
    _write_depth(depth_dir / f"binance-depth20-{day.strftime('%Y%m%d')}.jsonl.gz", depth)
    res_map = {("BTC-5m", w["open_ms"]): {"outcome": w["outcome"],
               "condition_id": "0x" + f"{w['open_ms'] // 1000:x}{'cd' * 32}"[:64]} for w in windows}
    return {"day": day_str, "windows": windows, "n_windows": n_windows, "res_map": res_map}


def _comp_fix_book(win_outcome: str, outcome: str, k: int, dur_secs: int) -> tuple[float, float]:
    """The Telonex fixture book (bid, ask) for ``outcome`` at second ``k`` of a window that
    resolves ``win_outcome`` — the EXACT rounded values ``_pm_quote_rows`` writes, so a fill
    placed at ``bid``/``ask`` classifies as at-touch/crossing deterministically."""
    target = 0.99 if win_outcome == "Up" else 0.01
    um = 0.5 + (target - 0.5) * min(1.0, (k / max(1, dur_secs)) ** 1.5)
    mid = um if outcome == "Up" else (1.0 - um)
    mid = min(0.995, max(0.005, mid))
    bid = float(f"{max(0.001, mid - 0.005):.3f}")
    ask = float(f"{min(0.999, mid + 0.005):.3f}")
    return bid, ask


def make_competitor_fills_fixture(
    comp_dir: Path, hist_result: dict, *, addr: str = "0xc10ffee0competitorfixture00000000000000",
    handle: str = "clonetest", pairs: float = 100.0, crossing_per_window: int = 0,
    merge_heavy: bool = True, inconsistent: bool = False,
) -> dict:
    """Write a synthetic competitor cache (``activity.jsonl`` / ``taker_trades.jsonl`` /
    ``official_pnl.json`` / ``positions.json``) under ``comp_dir/<addr>/``, aligned to
    :func:`make_historical_fixture`'s slugs / tokens / books / res_map so the operating
    manual, anchor reconciliation, and hand-trace run fully offline.

    Per resolvable BTC-5m window: an Up BUY of ``pairs`` at the touch (early), a Down BUY
    of ``pairs`` at the touch (late), ``crossing_per_window`` Up taker BUYs at the ask, and
    a MERGE of the matched pairs (``merge_heavy``, → recycled collateral) else a REDEEM. The
    two-sided touch buys have pair-cost ≈ 0.99 and the merge returns matched·$1, so the
    reconstructed OUR-series net is strongly POSITIVE (a maker winner). ``inconsistent``
    writes an official curve whose account-wide slice is ≈ $0 while the reconstruction is
    positive → the anchor's red "OUR RECONSTRUCTION SUSPECT" rule fires; otherwise the
    official curve dwarfs the reconstruction (consistent). Returns the expected counts."""
    acc = Path(comp_dir) / addr.lower()
    acc.mkdir(parents=True, exist_ok=True)
    day_str = hist_result["day"]
    res_map = hist_result["res_map"]
    windows = [w for w in hist_result["windows"] if not w["ambiguous"] and not w["coverage_excluded"]]
    dur_secs = 300
    activity: list[dict] = []
    taker: list[dict] = []
    th = 0
    exp = {"at_touch": 0, "crossing": 0, "n_fills": 0, "n_merge": 0, "n_redeem": 0,
           "maker_timing": [], "taker_timing": [], "n_windows": len(windows)}

    def trade(ts, cid, token, outcome, side, size, price, is_taker):
        nonlocal th
        th += 1
        row = {"type": "TRADE", "timestamp": int(ts), "conditionId": cid, "asset": token,
               "side": side, "size": float(size), "usdcSize": round(float(size) * price, 6),
               "price": float(price), "outcome": outcome, "outcomeIndex": 999, "slug": slug,
               "transactionHash": f"0x{th:064x}", "name": handle, "pseudonym": handle}
        activity.append(row)
        if is_taker:
            taker.append({k: row[k] for k in ("transactionHash", "asset", "side", "size", "price",
                                              "timestamp", "slug", "conditionId", "outcome")})

    for w in windows:
        slug, open_ms, outcome = w["slug"], w["open_ms"], w["outcome"]
        open_s = open_ms // 1000
        cid = res_map[("BTC-5m", open_ms)]["condition_id"]
        up_tok, dn_tok = f"{slug}-up", f"{slug}-dn"
        # Both legs bought early (mids ≈ 0.5) so pair-cost ≈ 0.99 regardless of outcome →
        # merge returns matched·$1 for a deterministic POSITIVE reconstructed net.
        bid, _ = _comp_fix_book(outcome, "Up", 30, dur_secs)
        trade(open_s + 30, cid, up_tok, "Up", "BUY", pairs, bid, False)
        exp["at_touch"] += 1; exp["n_fills"] += 1; exp["maker_timing"].append(30 / dur_secs)
        bid, _ = _comp_fix_book(outcome, "Down", 45, dur_secs)
        trade(open_s + 45, cid, dn_tok, "Down", "BUY", pairs, bid, False)
        exp["at_touch"] += 1; exp["n_fills"] += 1; exp["maker_timing"].append(45 / dur_secs)
        # crossing taker Up buys (mid-window)
        for _i in range(crossing_per_window):
            _, ask = _comp_fix_book(outcome, "Up", 150, dur_secs)
            trade(open_s + 150, cid, up_tok, "Up", "BUY", 3.0, ask, True)
            exp["crossing"] += 1; exp["n_fills"] += 1; exp["taker_timing"].append(150 / dur_secs)
        if merge_heavy:
            activity.append({"type": "MERGE", "timestamp": open_s + 200, "conditionId": cid,
                             "size": float(pairs), "usdcSize": float(pairs), "price": 0,
                             "asset": "", "side": "", "outcome": "", "outcomeIndex": 999, "slug": slug})
            exp["n_merge"] += 1
        else:
            activity.append({"type": "REDEEM", "timestamp": open_s + 305, "conditionId": cid,
                             "size": float(pairs), "usdcSize": float(pairs), "price": 0,
                             "asset": "", "side": "", "outcome": "", "outcomeIndex": 999, "slug": slug})
            exp["n_redeem"] += 1

    (acc / "activity.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in activity), encoding="utf-8")
    (acc / "taker_trades.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in taker), encoding="utf-8")
    (acc / "positions.json").write_text("[]", encoding="utf-8")

    day_start = int(datetime.fromisoformat(day_str + "T00:00:00+00:00").timestamp())
    jun1 = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp())
    end_t = day_start + 86_400
    # Anchor test (symmetric rule): consistent = official ≈ reconstruction (small gap → not
    # suspect); inconsistent = official far from reconstruction (large gap → SUSPECT). The
    # reconstruction is a small positive (~$1/window merge profit), so the consistent official
    # is small and the inconsistent one is a large opposite/divergent figure.
    series_pts = ([{"t": jun1, "p": 0.0}, {"t": end_t, "p": -50_000.0}] if inconsistent
                  else [{"t": jun1, "p": 0.0}, {"t": end_t, "p": 100.0}])
    (acc / "official_pnl.json").write_text(json.dumps(
        {"user_address": addr, "interval": "all", "fidelity": "1d",
         "fetched_at_unix": end_t, "series": series_pts}), encoding="utf-8")
    exp.update({"addr": addr, "handle": handle})
    return exp


def write_historical_coverage(out_dir: Path, coverage: dict) -> Path:
    """Write the synthetic ``coverage.json`` the historical builder reads
    (``out/telonex/coverage.json``)."""
    dest = Path(out_dir) / "telonex"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "coverage.json"
    path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return path


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
