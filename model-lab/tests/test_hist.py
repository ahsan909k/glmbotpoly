"""Offline tests for the historical aggTrades download + integrity stage.

No network: every download is driven through the ``fetch`` seam with synthetic
in-memory ZIPs (``zipfile`` + a real SHA-256 for the fake ``.CHECKSUM``).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from model_lab import hist, hist_integrity
from model_lab.config import Paths
from model_lab.io import binance_archive as ba


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_zip(symbol: str, day: date, rows: list[tuple]) -> bytes:
    """Build a synthetic aggTrades zip. Each row: (id, price, qty, first, last,
    ts, maker[, is_best]) — an 8th element adds a trailing is_best_match column."""
    lines = []
    for r in rows:
        line = f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{'true' if r[6] else 'false'}"
        if len(r) > 7:
            line += f",{'true' if r[7] else 'false'}"
        lines.append(line)
    csv = ("\n".join(lines) + "\n").encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{symbol}-aggTrades-{day.isoformat()}.csv", csv)
    return buf.getvalue()


def _publish(store: dict[str, bytes], symbol: str, day: date, rows: list[tuple], *, checksum="good") -> bytes:
    """Register a day's zip (+ optional .CHECKSUM) in a url->bytes store."""
    zb = _make_zip(symbol, day, rows)
    store[ba.day_url(symbol, day)] = zb
    fname = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    if checksum == "good":
        store[ba.checksum_url(symbol, day)] = f"{hashlib.sha256(zb).hexdigest()}  {fname}".encode()
    elif checksum == "bad":
        store[ba.checksum_url(symbol, day)] = f"{'0' * 64}  {fname}".encode()
    # checksum == "absent": leave the .CHECKSUM url unpublished → fetch 404s
    return zb


def _fetcher(store: dict[str, bytes]):
    def fetch(url: str) -> bytes:
        if url in store:
            return store[url]
        raise ba.ArchiveNotFound(url)

    return fetch


def _rows(base_id: int, ts0: int, n: int = 5) -> list[tuple]:
    return [
        (base_id + j, f"60000.{j}", f"0.{j + 1}", (base_id + j) * 2, (base_id + j) * 2 + 1, ts0 + j * 1000, j % 2 == 0)
        for j in range(n)
    ]


def _paths(tmp_path: Path) -> Paths:
    return Paths(
        journal_dir=tmp_path / "j",
        depth_dir=tmp_path / "d",
        out_dir=tmp_path / "o",
        hist_dir=tmp_path / "agg",
    )


def _frame(ids: list[int], ts: list[int]) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "agg_trade_id": np.array(ids, dtype="int64"),
            "price": np.full(n, 60000.0),
            "quantity": np.full(n, 0.5),
            "first_trade_id": np.array(ids, dtype="int64") * 2,
            "last_trade_id": np.array(ids, dtype="int64") * 2 + 1,
            "transact_time": np.array(ts, dtype="int64"),
            "is_buyer_maker": np.zeros(n, dtype=bool),
        }
    )


def _write_day(hist_dir: Path, symbol: str, day: date, df: pd.DataFrame) -> None:
    hist_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ba.day_parquet_path(hist_dir, symbol, day), engine="pyarrow", compression="zstd", index=False)


US = 1_751_000_000_000_000  # a 2025+ microsecond epoch anchor


# --------------------------------------------------------------------------- #
# Checksum
# --------------------------------------------------------------------------- #


def test_verify_sha256_and_tamper():
    zb = _make_zip("BTCUSDT", date(2026, 1, 1), _rows(1, US))
    chk = f"{hashlib.sha256(zb).hexdigest()}  BTCUSDT-aggTrades-2026-01-01.zip"
    assert ba.verify_sha256(zb, chk)
    assert not ba.verify_sha256(zb + b"x", chk)
    assert not ba.verify_sha256(zb, "not-a-checksum-line")
    assert not ba.verify_sha256(zb, "")


# --------------------------------------------------------------------------- #
# CSV parsing (header / column-count / unit variants)
# --------------------------------------------------------------------------- #


def test_parse_headerless_microseconds():
    df = ba.parse_aggtrades_csv(b"100,60000.1,0.5,200,205,1751000000123456,true\n")
    assert list(df.columns) == ba.AGGTRADES_COLS
    assert str(df["agg_trade_id"].dtype) == "int64"
    assert df["is_buyer_maker"].dtype == bool and bool(df["is_buyer_maker"].iloc[0])
    assert int(df["transact_time"].iloc[0]) == 1751000000123456


def test_parse_headered_8col_milliseconds_normalized():
    # Header row + trailing is_best_match column + millisecond timestamps.
    csv = b"a,b,c,d,e,f,g,h\n100,60000.1,0.5,200,205,1699999999123,false,true\n"
    df = ba.parse_aggtrades_csv(csv)
    assert list(df.columns) == ba.AGGTRADES_COLS  # is_best_match dropped
    assert int(df["transact_time"].iloc[0]) == 1699999999123000  # ms → µs
    assert not bool(df["is_buyer_maker"].iloc[0])


def test_parse_empty():
    assert len(ba.parse_aggtrades_csv(b"")) == 0
    assert len(ba.parse_aggtrades_csv(b"   \n")) == 0


# --------------------------------------------------------------------------- #
# download_day — store, skip, missing, checksum
# --------------------------------------------------------------------------- #


def test_download_day_writes_then_skips(tmp_path):
    hd = tmp_path / "agg"
    d = date(2026, 1, 1)
    fetch = _fetcher(_publish_store := {})
    _publish(_publish_store, "BTCUSDT", d, _rows(1000, US))

    r1 = ba.download_day("BTCUSDT", d, hd, fetch=fetch)
    assert r1.status == "downloaded" and r1.rows == 5 and r1.checksum == "verified"
    p = ba.day_parquet_path(hd, "BTCUSDT", d)
    assert p.exists()

    df = pd.read_parquet(p)
    assert list(df.columns) == ba.AGGTRADES_COLS
    assert str(df["agg_trade_id"].dtype) == "int64"
    assert str(df["transact_time"].dtype) == "int64"
    assert df["is_buyer_maker"].dtype == bool

    # Incremental: an existing final parquet is left untouched.
    r2 = ba.download_day("BTCUSDT", d, hd, fetch=fetch)
    assert r2.status == "skipped" and r2.rows == 5


def test_download_day_missing(tmp_path):
    hd = tmp_path / "agg"
    d = date(2026, 1, 1)
    r = ba.download_day("BTCUSDT", d, hd, fetch=_fetcher({}))  # nothing published
    assert r.status == "missing"
    assert not ba.day_parquet_path(hd, "BTCUSDT", d).exists()


def test_download_day_checksum_mismatch_not_stored(tmp_path):
    hd = tmp_path / "agg"
    d = date(2026, 1, 1)
    store: dict[str, bytes] = {}
    _publish(store, "BTCUSDT", d, _rows(1, US), checksum="bad")
    fetch = _fetcher(store)

    r = ba.download_day("BTCUSDT", d, hd, fetch=fetch)
    assert r.status == "checksum_failed" and r.checksum == "mismatch"
    assert not ba.day_parquet_path(hd, "BTCUSDT", d).exists()

    # With verification off, it stores anyway (checksum "skipped").
    r2 = ba.download_day("BTCUSDT", d, hd, fetch=fetch, verify_checksum=False)
    assert r2.status == "downloaded" and r2.checksum == "skipped"


def test_download_day_checksum_absent_still_stores(tmp_path):
    hd = tmp_path / "agg"
    d = date(2026, 1, 1)
    store: dict[str, bytes] = {}
    _publish(store, "BTCUSDT", d, _rows(1, US), checksum="absent")
    r = ba.download_day("BTCUSDT", d, hd, fetch=_fetcher(store))
    assert r.status == "downloaded" and r.checksum == "absent"


# --------------------------------------------------------------------------- #
# hist.download — full run, incremental, manifest, disk usage
# --------------------------------------------------------------------------- #


def _publish_range(store, symbols, days):
    for sym in symbols:
        base = 1000
        for i, dd in enumerate(days):
            _publish(store, sym, dd, _rows(base, US + i * 86_400_000_000))
            base += 100


def test_download_full_manifest_and_incremental(tmp_path):
    paths = _paths(tmp_path)
    symbols = ["BTCUSDT", "ETHUSDT"]
    days = ba.daterange(date(2026, 1, 1), date(2026, 1, 3))
    store: dict[str, bytes] = {}
    _publish_range(store, symbols, days)
    fetch = _fetcher(store)

    # Request one extra day (2026-01-04) served as 404 → missing for both symbols.
    summary = hist.download(
        paths, symbols=symbols, start=date(2026, 1, 1), end=date(2026, 1, 4), jobs=1, fetch=fetch
    )
    assert summary["downloaded"] == 6
    assert summary["missing"] == 2
    assert summary["checksum_failed"] == 0 and summary["error"] == 0
    assert summary["rows_downloaded"] == 30

    man = ba.read_manifest(paths.hist_dir)
    assert man is not None and man["time_unit"] == "microseconds"
    assert man["base_url"] == ba.BASE_URL
    btc = man["symbols"]["BTCUSDT"]
    assert btc["day_count"] == 3
    assert btc["days_present"] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert btc["missing_in_range"] == ["2026-01-04"]
    assert btc["first_day"] == "2026-01-01" and btc["last_day"] == "2026-01-03"
    assert btc["rows_total"] == 15
    assert man["disk_usage_bytes"] > 0
    assert man["disk_usage_bytes"] == ba.dir_disk_usage(paths.hist_dir)

    # Re-run the available range: fully incremental.
    again = hist.download(
        paths, symbols=symbols, start=date(2026, 1, 1), end=date(2026, 1, 3), jobs=1, fetch=fetch
    )
    assert again["downloaded"] == 0 and again["skipped"] == 6

    # Extend the range with the same command: only the new day is fetched.
    _publish(store, "BTCUSDT", date(2026, 1, 4), _rows(1300, US + 3 * 86_400_000_000))
    ext = hist.download(
        paths, symbols=["BTCUSDT"], start=date(2026, 1, 1), end=date(2026, 1, 4), jobs=1, fetch=fetch
    )
    assert ext["downloaded"] == 1 and ext["skipped"] == 3


def test_human_bytes_and_helpers():
    assert ba.human_bytes(0) == "0 B"
    assert ba.human_bytes(900_000_000).endswith("MiB")
    assert ba.human_bytes(54_000_000_000).endswith("GiB")
    assert len(ba.daterange(date(2026, 1, 1), date(2026, 1, 5))) == 5
    assert ba.daterange(date(2026, 1, 5), date(2026, 1, 1)) == []


# --------------------------------------------------------------------------- #
# hist_integrity — pass, corruption detection, manifest drift, continuity
# --------------------------------------------------------------------------- #


def test_integrity_passes_on_downloaded_store(tmp_path):
    paths = _paths(tmp_path)
    symbols = ["BTCUSDT", "ETHUSDT"]
    days = ba.daterange(date(2026, 1, 1), date(2026, 1, 3))
    store: dict[str, bytes] = {}
    _publish_range(store, symbols, days)
    hist.download(paths, symbols=symbols, start=days[0], end=days[-1], jobs=1, fetch=_fetcher(store))

    result = hist_integrity.integrity(paths)
    assert result["days_checked"] == 6
    assert result["days_failed"] == 0
    assert all(d["ok"] for d in result["days"])


def test_integrity_detects_corruption(tmp_path):
    paths = _paths(tmp_path)
    hd = paths.hist_dir
    _write_day(hd, "BTCUSDT", date(2026, 1, 1), _frame([1, 2, 3], [100, 200, 300]))  # good
    _write_day(hd, "BTCUSDT", date(2026, 1, 2), _frame([4, 5, 5], [400, 500, 600]))  # duplicate id
    _write_day(hd, "BTCUSDT", date(2026, 1, 3), _frame([7, 8, 9], [900, 800, 1000]))  # ts out of order
    _write_day(hd, "BTCUSDT", date(2026, 1, 4), _frame([], []))  # empty

    result = hist_integrity.integrity(paths)
    by_day = {d["day"]: d for d in result["days"]}
    assert by_day["2026-01-01"]["ok"]
    assert not by_day["2026-01-02"]["ok"] and any("duplicate" in p for p in by_day["2026-01-02"]["problems"])
    assert not by_day["2026-01-03"]["ok"] and any("non-monotonic" in p for p in by_day["2026-01-03"]["problems"])
    assert not by_day["2026-01-04"]["ok"] and any("empty" in p for p in by_day["2026-01-04"]["problems"])
    assert result["days_failed"] == 3


def test_integrity_row_count_vs_manifest(tmp_path):
    paths = _paths(tmp_path)
    hd = paths.hist_dir
    _write_day(hd, "BTCUSDT", date(2026, 1, 1), _frame([1, 2, 3], [100, 200, 300]))
    # A manifest that overstates the row count → truncation/corruption is flagged.
    man = ba.build_manifest(hd, ["BTCUSDT"])
    man["symbols"]["BTCUSDT"]["per_day"]["2026-01-01"]["rows"] = 5
    ba.write_manifest(hd, man)

    d = hist_integrity.integrity(paths)["days"][0]
    assert not d["ok"] and any("!= manifest" in p for p in d["problems"])


def test_integrity_cross_day_continuity_note(tmp_path):
    paths = _paths(tmp_path)
    hd = paths.hist_dir
    _write_day(hd, "BTCUSDT", date(2026, 1, 1), _frame([10, 11, 12], [100, 200, 300]))
    _write_day(hd, "BTCUSDT", date(2026, 1, 2), _frame([12, 13, 14], [400, 500, 600]))  # 12 <= prev last 12

    result = hist_integrity.integrity(paths)
    d2 = next(d for d in result["days"] if d["day"] == "2026-01-02")
    assert d2["ok"]  # continuity overlap is a note, not a failure
    assert d2["continuity_warnings"]


def test_symbols_present_and_has_aggtrades(tmp_path):
    paths = _paths(tmp_path)
    assert not ba.has_aggtrades(paths.hist_dir)
    _write_day(paths.hist_dir, "BTCUSDT", date(2026, 1, 1), _frame([1], [100]))
    _write_day(paths.hist_dir, "ETHUSDT", date(2026, 1, 1), _frame([1], [100]))
    assert ba.has_aggtrades(paths.hist_dir)
    assert ba.symbols_present(paths.hist_dir) == ["BTCUSDT", "ETHUSDT"]
