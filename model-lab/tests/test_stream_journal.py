"""Tests for the streaming/bounded journal first pass (``dataset._read_journal``).

The streaming pass reduces raw ticks to what the feature grid + strike actually read
(``BinanceDirect``/``Mid`` per-second bars + all ``ChainlinkRtds`` raw), dropping the
rest — at a fraction of the memory. These tests prove that reduction is **output-
identical** to the full-tick pass on a journal with *many* ticks per second (the
fixture has only one, so it can't exercise the reduction), and that ``--since``/
``--until`` bounds filter identically in both modes.
"""

from __future__ import annotations

import gzip
import json
import random

import pandas as pd

from model_lab import dataset as ds
from model_lab import fixtures as fx
from model_lab import short_horizon as sh
from model_lab.config import Paths, _in_bounds, parse_time_bound
from model_lab.ingest import TICK_COLS


# --------------------------------------------------------------------------- #
# a journal with MANY ticks per second (so the reduction is non-trivial)
# --------------------------------------------------------------------------- #
def _multi_tick_journal(journal_dir, depth_dir, n_windows: int = 4) -> None:
    """Write a synthetic journal that, each second, emits **three** BinanceDirect/Mid
    ticks (distinct ms + one same-ms tie), plus BinanceDirect/Trade and BinanceRtds
    ticks (which nothing downstream reads) and one ChainlinkRtds tick. Also windows,
    settlements, top_of_book, and a small depth ladder — a complete, resolvable set."""
    rng = random.Random(4321)
    records: list[tuple[int, dict]] = []
    depth: list[dict] = []
    price = fx.START_PRICE
    for w in range(n_windows):
        open_t, close_t = w * fx.WINDOW_SECS, (w + 1) * fx.WINDOW_SECS
        open_ms, close_ms = fx.TS0_MS + open_t * 1000, fx.TS0_MS + close_t * 1000
        strike = price
        records.append((open_ms, {"type": "window", "market": fx._market(open_ms, close_ms, strike),
                                   "lifecycle": "Open"}))
        for t in range(open_t, close_t):
            ts = fx.TS0_MS + t * 1000
            price *= 2.718281828 ** (2.0e-4 * rng.gauss(0.0, 1.0))
            s = price
            if t == open_t:
                strike = s * (1 + fx.BASIS)
            # Chainlink (kept raw), the strike/basis feed.
            records.append((ts, {"type": "price_tick", "source": "ChainlinkRtds", "asset": "Btc",
                                  "kind": "Vendor", "value": f"{s * (1 + fx.BASIS):.8f}",
                                  "ts_exchange": ts, "ts_local": ts}))
            # THREE Binance Mid ticks this second (ms 100/500/900) — the last (max ms)
            # closes the second's bar. Only these + Chainlink survive the reduction.
            for j, frac in enumerate((100, 500, 900)):
                mts = ts + frac
                records.append((mts, {"type": "price_tick", "source": "BinanceDirect", "asset": "Btc",
                                      "kind": "Mid", "value": f"{s * (1 + j * 1e-6):.8f}",
                                      "ts_exchange": mts, "ts_local": mts}))
            # A same-ms TIE with the 900-ms tick at one second — appended last, so the
            # stable collapse must keep THIS value in both paths.
            if t == open_t + 50:
                records.append((ts + 900, {"type": "price_tick", "source": "BinanceDirect", "asset": "Btc",
                                           "kind": "Mid", "value": f"{s * (1 + 9e-6):.8f}",
                                           "ts_exchange": ts + 900, "ts_local": ts + 900}))
            # Ticks that NOTHING downstream reads — must be dropped by the reduction.
            records.append((ts, {"type": "price_tick", "source": "BinanceDirect", "asset": "Btc",
                                  "kind": "Trade", "value": f"{s:.8f}", "ts_exchange": ts, "ts_local": ts}))
            records.append((ts, {"type": "price_tick", "source": "BinanceRtds", "asset": "Btc",
                                  "kind": "Vendor", "value": f"{s:.8f}", "ts_exchange": ts, "ts_local": ts}))
            # A model + top_of_book (warmed seconds) + a 2-level depth ladder.
            if t - open_t >= 60:
                mkt = min(0.98, max(0.02, 0.5 + 0.1 * rng.gauss(0.0, 1.0)))
                records.append((ts, {"type": "top_of_book", "token_id": fx.UP_TOKEN,
                                      "top": {"bid": {"price": f"{mkt - 0.006:.4f}", "size": "60"},
                                              "ask": {"price": f"{mkt + 0.006:.4f}", "size": "40"},
                                              "ts": ts}}))
            half = s * 5e-6
            depth.append({"recv_ms": ts, "stream": "btcusdt@depth20@100ms",
                          "data": {"lastUpdateId": t,
                                   "bids": [[f"{s - half:.2f}", "9.0"], [f"{s - 2 * half:.2f}", "7.0"]],
                                   "asks": [[f"{s + half:.2f}", "8.0"], [f"{s + 2 * half:.2f}", "6.0"]]}})
        outcome = "Up" if price >= strike else "Down"
        records.append((close_ms, {"type": "window", "market": fx._market(open_ms, close_ms, strike),
                                    "lifecycle": {"Resolved": {"outcome": outcome}}}))
        records.append((close_ms, {"type": "settlement", "window": {"series": fx.SERIES, "open_time": open_ms},
                                   "outcome": outcome, "up": {"shares": "0", "cost": "0"},
                                   "down": {"shares": "0", "cost": "0"}, "matched_pairs": "0",
                                   "pair_cost": None, "excess": "0", "merged_pairs": "0",
                                   "fees_paid": "0", "realized_pnl": "0", "ts": close_ms}))
    records.sort(key=lambda r: r[0])
    journal_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(journal_dir / "journal-20231114-000000-00000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for seq, (tsl, rec) in enumerate(records, 1):
            fh.write(json.dumps({"seq": seq, "ts_local_ms": tsl, "rec": rec}) + "\n")
    with gzip.open(depth_dir / "binance-depth20-20231114.jsonl.gz", "wt", encoding="utf-8") as fh:
        for frame in depth:
            fh.write(json.dumps(frame) + "\n")


def _paths(tmp_path, out: str) -> Paths:
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                 out_dir=tmp_path / out, hist_dir=tmp_path / "no_hist")


# --------------------------------------------------------------------------- #
# the reduction actually reduces — and only keeps what's read
# --------------------------------------------------------------------------- #
def test_reduction_keeps_only_used_ticks(tmp_path):
    _multi_tick_journal(tmp_path / "journal", tmp_path / "depth")
    paths = _paths(tmp_path, "o")
    full, *_ = ds._read_journal(paths, None, stream=False)
    red, *_ = ds._read_journal(paths, None, stream=True)

    # Full keeps every raw tick; reduced keeps far fewer (Mid bars + Chainlink only).
    assert len(red) < len(full)
    assert set(red["source"].unique()) == {"BinanceDirect", "ChainlinkRtds"}
    assert set(red.loc[red["source"] == "BinanceDirect", "kind"].unique()) == {"Mid"}
    # Reduced BinanceDirect/Mid rows are one bar per (asset, second).
    mid = red[(red["source"] == "BinanceDirect") & (red["kind"] == "Mid")].copy()
    ts = mid["ts_exchange"].fillna(mid["ts_local"]).astype("int64")
    assert not (ts // 1000).duplicated().any()  # exactly one Mid bar per second
    # Chainlink is kept raw (same count as the full pass).
    assert int((red["source"] == "ChainlinkRtds").sum()) == int((full["source"] == "ChainlinkRtds").sum())


# --------------------------------------------------------------------------- #
# stream == full: byte-identical dataset + short_horizon parquets (the proof)
# --------------------------------------------------------------------------- #
def test_dataset_stream_equals_full(tmp_path):
    _multi_tick_journal(tmp_path / "journal", tmp_path / "depth")
    ps = _paths(tmp_path, "stream")
    pf = _paths(tmp_path, "full")
    ds.dataset(ps, grid_secs=5, include_history=False, stream=True)
    ds.dataset(pf, grid_secs=5, include_history=False, stream=False)
    A = pd.read_parquet(ps.table("dataset"), dtype_backend="pyarrow")
    B = pd.read_parquet(pf.table("dataset"), dtype_backend="pyarrow")
    assert len(A) > 0
    pd.testing.assert_frame_equal(A, B)


def test_short_horizon_stream_equals_full(tmp_path):
    _multi_tick_journal(tmp_path / "journal", tmp_path / "depth")
    ps = _paths(tmp_path, "stream")
    pf = _paths(tmp_path, "full")
    sh.short_horizon(ps, grid_secs=5, include_history=False, stream=True)
    sh.short_horizon(pf, grid_secs=5, include_history=False, stream=False)
    A = pd.read_parquet(ps.table("short_horizon"), dtype_backend="pyarrow")
    B = pd.read_parquet(pf.table("short_horizon"), dtype_backend="pyarrow")
    assert len(A) > 0
    pd.testing.assert_frame_equal(A, B)


# --------------------------------------------------------------------------- #
# --since / --until bounds: honored, and stream == full under a bound
# --------------------------------------------------------------------------- #
def test_bounds_filter_records(tmp_path):
    _multi_tick_journal(tmp_path / "journal", tmp_path / "depth", n_windows=4)
    paths = _paths(tmp_path, "o")
    # Keep only the middle two windows [open_1, open_3).
    since = fx.TS0_MS + fx.WINDOW_SECS * 1000
    until = fx.TS0_MS + 3 * fx.WINDOW_SECS * 1000
    _t, win_all, *_ = ds._read_journal(paths, None)
    _t, win_bounded, *_ = ds._read_journal(paths, None, since_ms=since, until_ms=until)
    assert len(win_bounded) == 2 and len(win_all) == 4
    opens = [k[1] for k in win_bounded]
    assert all(since <= o < until for o in opens)


def test_dataset_bounds_stream_equals_full(tmp_path):
    _multi_tick_journal(tmp_path / "journal", tmp_path / "depth", n_windows=4)
    since = fx.TS0_MS + fx.WINDOW_SECS * 1000
    until = fx.TS0_MS + 3 * fx.WINDOW_SECS * 1000
    ps = _paths(tmp_path, "stream")
    pf = _paths(tmp_path, "full")
    rs = ds.dataset(ps, grid_secs=5, include_history=False, stream=True, since_ms=since, until_ms=until)
    rf = ds.dataset(pf, grid_secs=5, include_history=False, stream=False, since_ms=since, until_ms=until)
    assert rs["counts"]["windows"] == 2 and rf["counts"]["windows"] == 2  # bound honored
    A = pd.read_parquet(ps.table("dataset"), dtype_backend="pyarrow")
    B = pd.read_parquet(pf.table("dataset"), dtype_backend="pyarrow")
    pd.testing.assert_frame_equal(A, B)


# --------------------------------------------------------------------------- #
# parse_time_bound / _in_bounds units
# --------------------------------------------------------------------------- #
def test_parse_time_bound():
    assert parse_time_bound(None) is None
    assert parse_time_bound("") is None
    assert parse_time_bound("1783051708299") == 1783051708299          # raw ms
    assert parse_time_bound("2026-07-03") == 1783036800000             # date → 00:00Z
    assert parse_time_bound("2026-07-03T04:00:00") == 1783051200000    # ISO datetime (UTC)


def test_in_bounds():
    assert _in_bounds(5, None, None) is True          # unbounded passes everything
    assert _in_bounds(None, None, None) is True        # incl. a record with no time
    assert _in_bounds(None, 1, 10) is False            # bounded → no-time excluded
    assert _in_bounds(1, 1, 10) is True                # since inclusive
    assert _in_bounds(10, 1, 10) is False              # until exclusive
    assert _in_bounds(0, 1, 10) is False
