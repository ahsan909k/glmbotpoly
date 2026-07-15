"""Unit tests for the Telonex Binance depth pull (reshaping parity, shift search, projection).

The load-bearing bit is the reshaping: a Telonex ``book_snapshot_25`` row must produce
byte-identical depth features to our live ``read_depth_levels`` frame — because both feed the
*same* ``short_horizon`` math, that is what makes the Stage-3 parity comparison meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_lab import short_horizon as sh
from model_lab import telonex_binance_pull as P
from model_lab import telonex_binance_validate as V
from model_lab.config import Paths

LEVELS = 25


def _synthetic_book(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A plausible 25-level book (best-first): bids descending, asks ascending, varied sizes."""
    rng = np.random.default_rng(seed)
    mid = 63000.0 + seed
    bid_px = mid - 0.01 * (1 + np.arange(LEVELS))
    ask_px = mid + 0.01 * (1 + np.arange(LEVELS))
    bid_sz = np.round(rng.uniform(0.1, 5.0, LEVELS), 6)
    ask_sz = np.round(rng.uniform(0.1, 5.0, LEVELS), 6)
    return bid_px, bid_sz, ask_px, ask_sz


def _write_telonex_depth(path: Path, books: list[tuple], base_sec: int = 1_000_000) -> None:
    """Write a Telonex-shaped ``book_snapshot_25`` parquet (flattened string columns, µs ts)."""
    cols: dict[str, list] = {"timestamp_us": []}
    for side in ("bid", "ask"):
        for i in range(LEVELS):
            cols[f"{side}_price_{i}"] = []
            cols[f"{side}_size_{i}"] = []
    for r, (bpx, bsz, apx, asz) in enumerate(books):
        cols["timestamp_us"].append((base_sec + r) * 1_000_000 + 500_000)  # mid-second µs
        for i in range(LEVELS):
            cols[f"bid_price_{i}"].append(f"{bpx[i]:.8f}")
            cols[f"bid_size_{i}"].append(f"{bsz[i]:.8f}")
            cols[f"ask_price_{i}"].append(f"{apx[i]:.8f}")
            cols[f"ask_size_{i}"].append(f"{asz[i]:.8f}")
    fields = {"timestamp_us": pa.array(cols["timestamp_us"], type=pa.int64())}
    for k, v in cols.items():
        if k != "timestamp_us":
            fields[k] = pa.array(v, type=pa.string())
    pq.write_table(pa.table(fields), path)


def test_reshape_matches_read_depth_levels_frame(tmp_path: Path):
    """Telonex row → features == the same book through read_depth_levels' frame shape."""
    books = [_synthetic_book(s) for s in range(8)]
    tel_path = tmp_path / "btcusdt-2026-07-05.parquet"
    _write_telonex_depth(tel_path, books)

    tel_grid = V._tel_depth_grid(tel_path, "btc")

    # Independent "our" path: build the exact frame read_depth_levels yields, same recv_ms.
    frames = []
    for r, (bpx, bsz, apx, asz) in enumerate(books):
        recv_ms = ((1_000_000 + r) * 1_000_000 + 500_000) // 1000
        frames.append({"recv_ms": recv_ms, "asset": "btc",
                       "bid_px": bpx.copy(), "bid_sz": bsz.copy(),
                       "ask_px": apx.copy(), "ask_sz": asz.copy()})
    our_grid = sh.build_depth_feature_grid(frames, "btc")

    assert len(tel_grid) == len(our_grid) == len(books)
    feats = V._feature_cols(tel_grid)
    assert "depth_imb_20" in feats and "microprice_gap" in feats
    for f in feats:
        np.testing.assert_allclose(
            tel_grid[f].to_numpy(dtype="float64"),
            our_grid[f].to_numpy(dtype="float64"),
            rtol=1e-12, atol=1e-12, err_msg=f"feature {f} diverged",
        )


def test_valid_prefix_trims_thin_book():
    px = np.array([100.0, 99.0, np.nan, 98.0])
    sz = np.array([1.0, 2.0, 3.0, 4.0])
    assert V._valid_prefix(px, sz) == 2
    assert V._valid_prefix(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == 0  # non-positive price
    assert V._valid_prefix(np.array([100.0, 99.0]), np.array([1.0, 2.0])) == 2


def test_best_shift_recovers_injected_offset():
    n = 700
    secs = np.arange(n)
    signal = np.sin(secs * 0.07) + 0.3 * np.cos(secs * 0.013)
    our = pd.DataFrame({"sec": secs, "depth_imb_20": signal})
    # tel holds, at sec s, the value our shows at sec s+3 → alignment shift is +3.
    tel = pd.DataFrame({"sec": secs, "depth_imb_20": np.sin((secs + 3) * 0.07)
                        + 0.3 * np.cos((secs + 3) * 0.013)})
    shift, corr_best, corr0, matched = V._best_shift(our, tel, "depth_imb_20", shift_range=10)
    assert shift == 3
    assert corr_best > 0.999
    assert matched >= V.PARITY_MIN_N


def test_projection_arithmetic(tmp_path: Path):
    out = tmp_path / "out"
    (out / "telonex").mkdir(parents=True)
    # Trial meta: measured depth sizes + availability giving a 10-day range (to_date exclusive).
    meta = {
        "availability": {
            "btcusdt": {"channels": {"book_snapshot_25": {"from_date": "2026-02-06", "to_date": "2026-02-16"}}},
            "ethusdt": {"channels": {"book_snapshot_25": {"from_date": "2026-02-06", "to_date": "2026-02-16"}}},
        },
        "files": [
            {"symbol": "btcusdt", "channel": "book_snapshot_25", "day": "2026-07-04", "zst_bytes": 50_000_000},
            {"symbol": "btcusdt", "channel": "book_snapshot_25", "day": "2026-07-05", "zst_bytes": 60_000_000},
            {"symbol": "btcusdt", "channel": "book_snapshot_25", "day": "2026-07-06", "zst_bytes": 70_000_000},
            {"symbol": "ethusdt", "channel": "book_snapshot_25", "day": "2026-07-05", "zst_bytes": 40_000_000},
            {"symbol": "btcusdt", "channel": "trades", "day": "2026-07-05", "zst_bytes": 9_000_000},  # ignored
        ],
        "downloads_remaining": None,
    }
    (out / "telonex" / "binance_trial.json").write_text(json.dumps(meta), encoding="utf-8")
    # Polymarket coverage → ~19.5 GiB genuinely remaining.
    (out / "telonex" / "coverage.json").write_text(json.dumps({
        "totals": {"zst_bytes": 118_321_232_625},
        "projection": {"zst_bytes": 139_307_898_537},
    }), encoding="utf-8")

    paths = Paths(journal_dir=tmp_path, depth_dir=tmp_path, out_dir=out,
                  hist_dir=tmp_path, telonex_dir=tmp_path)
    proj = P.project(paths, symbols=("btcusdt", "ethusdt"), start=None, end=None)

    assert proj["status"] == "ok"
    assert proj["per_symbol"]["btcusdt"]["days"] == 10
    # BTC mean = (50+60+70)/3 = 60 MB/day; ETH = 40 MB/day.
    assert proj["per_symbol"]["btcusdt"]["zst_bytes_per_day"] == 60_000_000
    assert proj["per_symbol"]["ethusdt"]["zst_bytes_per_day"] == 40_000_000
    assert proj["projected_zst_bytes"] == 10 * 60_000_000 + 10 * 40_000_000
    assert proj["file_count"] == 20
    # Polymarket remaining is the small (net) figure, not the 188 GiB gross.
    assert proj["disk"]["polymarket_remaining_bytes"] == 139_307_898_537 - 118_321_232_625
    # Gate is the stricter (conservative) of the two available numbers.
    assert proj["disk"]["gate_available_bytes"] == min(
        proj["disk"]["principled_available_bytes"], proj["disk"]["conservative_available_bytes"])


def test_overlap_days_discovery(tmp_path: Path):
    depth = tmp_path / "depth"
    depth.mkdir()
    for d in ("20260703", "20260704", "20260705"):
        (depth / f"binance-depth20-{d}.jsonl.gz").write_bytes(b"")
    tel_depth = tmp_path / "telonex" / "raw" / "binance" / "book_snapshot_25"
    tel_depth.mkdir(parents=True)
    (tel_depth / "btcusdt-2026-07-04.parquet").write_bytes(b"")          # raw form
    (tel_depth / "btcusdt-2026-07-05.parquet.zst").write_bytes(b"")      # finalized form
    (tel_depth / "btcusdt-2026-02-10.parquet.zst").write_bytes(b"")      # no recorder overlap
    (tel_depth / "ethusdt-2026-07-05.parquet.zst").write_bytes(b"")

    paths = Paths(journal_dir=tmp_path, depth_dir=depth, out_dir=tmp_path / "out",
                  hist_dir=tmp_path, telonex_dir=tmp_path / "telonex")
    # 07-03 excluded by GUARD_START_DAY; 02-10 has no recorder file; both .parquet and .zst count.
    assert V._overlap_days(paths, "btcusdt") == ("2026-07-04", "2026-07-05")
    assert V._overlap_days(paths, "ethusdt") == ("2026-07-05",)


def test_parity_guard_no_overlap(tmp_path: Path):
    depth = tmp_path / "depth"
    depth.mkdir()
    paths = Paths(journal_dir=tmp_path, depth_dir=depth, out_dir=tmp_path / "out",
                  hist_dir=tmp_path, telonex_dir=tmp_path / "telonex")
    g = V.parity_guard(paths)
    assert g["passed"] is None
    assert all(r["status"] == "no_overlap_days" for r in g["per_asset"].values())


def test_project_no_trial_meta(tmp_path: Path):
    out = tmp_path / "out"
    (out / "telonex").mkdir(parents=True)
    paths = Paths(journal_dir=tmp_path, depth_dir=tmp_path, out_dir=out,
                  hist_dir=tmp_path, telonex_dir=tmp_path)
    proj = P.project(paths, symbols=("btcusdt",), start=None, end=None)
    assert proj["status"] == "no_trial"
