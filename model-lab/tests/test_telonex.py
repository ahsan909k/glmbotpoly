"""Offline tests for the Telonex trial pull + validation + compression.

No network: every request is driven through the ``fetch`` seam (``fixtures.
telonex_offline_fetch``) with synthetic responses, and the validation runs over an
on-disk synthetic trial store + matching journal (``fixtures.make_telonex_fixture``).
The zstd-dependent tests ``importorskip`` the optional ``zstandard`` package.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from model_lab.config import Paths
from model_lab.fixtures import make_telonex_fixture, telonex_offline_fetch
from model_lab.io import telonex as tx


def _paths(tmp_path: Path) -> Paths:
    return Paths(
        journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
        out_dir=tmp_path / "out", telonex_dir=tmp_path / "telonex",
    )


DAY = date(2026, 7, 5)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_updown_slugs_5m_is_288_per_day():
    slugs = tx.updown_slugs("btc", "5m", DAY)
    assert len(slugs) == 288
    assert slugs[0].startswith("btc-updown-5m-")
    # strictly 300 s apart, aligned to the UTC day start
    epochs = [int(s.rsplit("-", 1)[1]) for s in slugs]
    assert epochs[0] == tx.day_start_epoch(DAY)
    assert all(b - a == 300 for a, b in zip(epochs, epochs[1:]))


def test_parse_series():
    assert tx.parse_series("btc-updown-5m") == ("btc", "5m")
    assert tx.parse_series("eth-updown-1h") == ("eth", "1h")
    with pytest.raises(ValueError):
        tx.parse_series("BTC-5m")


def test_channel_covers_day_exclusive_to_date():
    assert tx.channel_covers_day({"from_date": "2026-07-04", "to_date": "2026-07-06"}, DAY)
    # to_date is exclusive — a range ending on the day itself does NOT cover it
    assert not tx.channel_covers_day({"from_date": "2026-07-01", "to_date": "2026-07-05"}, DAY)
    assert not tx.channel_covers_day({}, DAY)
    assert not tx.channel_covers_day(None, DAY)


def test_load_api_key_env_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv(tx.ENV_KEY, "  secret-from-env  ")
    assert tx.load_api_key() == "secret-from-env"
    monkeypatch.delenv(tx.ENV_KEY, raising=False)
    envf = tmp_path / ".env"
    envf.write_text('# comment\nOTHER=x\ntelonexdata="secret-from-file"\n', encoding="utf-8")
    assert tx.load_api_key(env_file=envf) == "secret-from-file"
    with pytest.raises(tx.TelonexError):
        tx.load_api_key(env_file=tmp_path / "missing.env")


# --------------------------------------------------------------------------- #
# Network primitive (through the fake fetch)
# --------------------------------------------------------------------------- #


def test_availability_offline():
    fetch = telonex_offline_fetch(day=DAY)
    slug = tx.updown_slugs("btc", "5m", DAY)[100]
    info = tx.availability(tx.POLYMARKET, slug=slug, outcome="Up", fetch=fetch)
    assert info["slug"] == slug
    assert "book_snapshot_full" in info["channels"]
    with pytest.raises(tx.TelonexNotFound):
        tx.availability(tx.POLYMARKET, slug="btc-updown-5m-1", outcome="Up", fetch=fetch)


def test_download_two_step_atomic_and_skip(tmp_path):
    fetch = telonex_offline_fetch(day=DAY, download_body=b"PARQUET-BYTES", remaining=42)
    dest = tmp_path / "sub" / "file.parquet"
    res = tx.download_asset_day(tx.POLYMARKET, "trades", "2026-07-05", dest,
                               api_key="k", slug="btc-updown-5m-1", outcome="Up", fetch=fetch)
    assert res["status"] == "downloaded"
    assert res["remaining"] == 42
    assert dest.read_bytes() == b"PARQUET-BYTES"
    assert not (dest.parent / (dest.name + ".tmp")).exists()  # atomic — no leftover tmp
    # skip-if-exists (resumable)
    res2 = tx.download_asset_day(tx.POLYMARKET, "trades", "2026-07-05", dest,
                                api_key="k", slug="btc-updown-5m-1", outcome="Up", fetch=fetch)
    assert res2["status"] == "skipped"


def test_download_limit_403(tmp_path):
    fetch = telonex_offline_fetch(day=DAY, fail_download_status=403)
    with pytest.raises(tx.TelonexDownloadLimit):
        tx.download_asset_day(tx.BINANCE, "trades", "2026-07-05", tmp_path / "x.parquet",
                             api_key="k", slug="btcusdt", fetch=fetch)


def test_download_404_is_no_data(tmp_path):
    def fetch(url, headers=None, follow_redirects=True):
        raise tx.TelonexHTTPError(404, {}, url)
    res = tx.download_asset_day(tx.POLYMARKET, "trades", "2026-07-05", tmp_path / "x.parquet",
                              api_key="k", slug="btc-updown-5m-1", outcome="Up", fetch=fetch)
    assert res["status"] == "no_data"


# --------------------------------------------------------------------------- #
# Ingest stage
# --------------------------------------------------------------------------- #


def test_ingest_trial_happy_path(tmp_path):
    from model_lab import telonex_ingest as ti

    live = set(tx.updown_slugs("btc", "5m", DAY)[100:103])  # 3 windows with data
    fetch = telonex_offline_fetch(day=DAY, live_slugs=live, download_body=b"x" * 128)
    secret = "TOPSECRET_KEY_XYZ"
    summary = ti.telonex_ingest(
        _paths(tmp_path), day=DAY, series="btc-updown-5m", outcomes=["Up", "Down"],
        channels=["trades", "quotes", "book_snapshot_full"], binance_symbol="btcusdt",
        binance_channels=["trades", "book_snapshot_25"], jobs=1, discover="availability",
        api_key=secret, fetch=fetch,
    )
    assert summary["status"] == "ok"
    assert summary["windows_with_data"] == 3
    # 3 windows × 2 outcomes × 3 channels + 2 Binance files = 20
    assert summary["downloaded"] == 20
    manifest = tx.read_manifest(_paths(tmp_path).telonex_dir)
    assert manifest is not None and len(manifest["trial"]["windows"]) == 3
    assert secret not in str(manifest)  # the key never lands in the manifest


def test_ingest_disk_preflight_aborts(tmp_path):
    from model_lab import telonex_ingest as ti

    live = set(tx.updown_slugs("btc", "5m", DAY)[100:103])
    fetch = telonex_offline_fetch(day=DAY, live_slugs=live, download_body=b"x" * 128)
    summary = ti.telonex_ingest(
        _paths(tmp_path), day=DAY, series="btc-updown-5m", outcomes=["Up"],
        channels=["book_snapshot_full"], binance_channels=["trades"], jobs=1,
        disk_factor=1e15, discover="availability", api_key="k", fetch=fetch,
    )
    assert summary["status"] == "aborted_disk"


def test_ingest_download_limit_aborts(tmp_path):
    from model_lab import telonex_ingest as ti

    live = set(tx.updown_slugs("btc", "5m", DAY)[100:103])
    fetch = telonex_offline_fetch(day=DAY, live_slugs=live, fail_download_status=403)
    summary = ti.telonex_ingest(
        _paths(tmp_path), day=DAY, series="btc-updown-5m", outcomes=["Up"],
        channels=["book_snapshot_full"], binance_channels=["trades"], jobs=1,
        discover="availability", api_key="k", fetch=fetch,
    )
    assert summary["status"] == "aborted_limit"


def test_windows_from_catalog(tmp_path):
    import pandas as pd

    start = tx.day_start_epoch(DAY)
    rows = [
        # two btc-updown-5m windows on the day, one with book data, one without
        {"slug": f"btc-updown-5m-{start}", "outcome_0": "Up", "outcome_1": "Down",
         "asset_id_0": "UP0", "asset_id_1": "DN0", "market_id": "0xaa",
         "book_snapshot_full_from": "2026-07-04", "trades_from": "2026-07-05"},
        {"slug": f"btc-updown-5m-{start + 300}", "outcome_0": "Up", "outcome_1": "Down",
         "asset_id_0": "UP1", "asset_id_1": "DN1", "market_id": "0xbb",
         "book_snapshot_full_from": "", "trades_from": ""},
        # a different day / different series → excluded
        {"slug": f"btc-updown-5m-{start - 86_400}", "outcome_0": "Up", "outcome_1": "Down",
         "asset_id_0": "UPX", "asset_id_1": "DNX", "market_id": "0xcc",
         "book_snapshot_full_from": "2026-07-04", "trades_from": "2026-07-04"},
        {"slug": "will-trump-win-2024", "outcome_0": "Yes", "outcome_1": "No",
         "asset_id_0": "Y", "asset_id_1": "N", "market_id": "0xdd",
         "book_snapshot_full_from": "2024-01-01", "trades_from": "2024-01-01"},
    ]
    cat = tmp_path / "markets.parquet"
    pd.DataFrame(rows).to_parquet(cat, engine="pyarrow", index=False)
    windows = tx.windows_from_catalog(cat, "btc", "5m", DAY)
    assert len(windows) == 1                      # only the in-day window with book data
    assert windows[0]["slug"] == f"btc-updown-5m-{start}"
    assert windows[0]["up_asset_id"] == "UP0"     # outcome_0 == "Up" → asset_id_0
    assert windows[0]["open_ms"] == start * 1000


def test_ingest_binance_plan_restricted_continues(tmp_path):
    from model_lab import telonex_ingest as ti

    live = set(tx.updown_slugs("btc", "5m", DAY)[100:103])
    # Single-Exchange plan: Binance downloads 403 exchange_restricted, Polymarket succeeds.
    fetch = telonex_offline_fetch(day=DAY, live_slugs=live, restrict_exchange="binance",
                                  download_body=b"x" * 64)
    summary = ti.telonex_ingest(
        _paths(tmp_path), day=DAY, series="btc-updown-5m", outcomes=["Up", "Down"],
        channels=["trades", "book_snapshot_full"], binance_channels=["trades", "book_snapshot_25"],
        jobs=1, discover="availability", api_key="k", fetch=fetch,
    )
    assert summary["status"] == "ok"                      # not aborted — Polymarket still pulled
    assert summary["binance_plan_restricted"] is True
    assert summary["plan_restricted"] == 2                # both Binance channels restricted
    # 3 windows × 2 outcomes × 2 channels = 12 Polymarket files downloaded
    assert summary["downloaded"] == 12


def test_ingest_no_windows(tmp_path):
    from model_lab import telonex_ingest as ti

    fetch = telonex_offline_fetch(day=DAY, live_slugs=set())  # nothing has data
    summary = ti.telonex_ingest(_paths(tmp_path), day=DAY, jobs=1, discover="availability",
                               api_key="k", fetch=fetch)
    assert summary["status"] == "no_windows"


# --------------------------------------------------------------------------- #
# Validation (over the on-disk synthetic trial + journal)
# --------------------------------------------------------------------------- #


def test_validate_checks_pass_and_recover_offsets(tmp_path):
    from model_lab import telonex_validate as tv
    from model_lab.fixtures import TX_BINANCE_OFFSET_MS, TX_CLOB_OFFSET_MS

    tdir = tmp_path / "telonex"
    jdir = tmp_path / "journal"
    make_telonex_fixture(tdir, jdir)
    paths = Paths(journal_dir=jdir, depth_dir=tmp_path / "depth", out_dir=tmp_path / "out", telonex_dir=tdir)
    result = tv.validate(paths, compress=False, catalog=True, fetch=telonex_offline_fetch())
    checks = result["checks"]
    for name in ("coverage", "cadence", "depth", "completeness", "clock"):
        assert checks[name]["result"] == "PASS", (name, checks[name])
    assert checks["date_range"]["series"]["btc-updown-5m"]["windows_with_data"] >= 1
    assert checks["cadence"]["polymarket_book_full"]["median_s"] <= 1.0
    assert checks["depth"]["polymarket_full_max_levels"] >= 20
    assert checks["depth"]["binance_25_levels"] == 25
    bin_off = checks["clock"]["external_binance_trades"]["median_offset_ms"]
    clob_off = checks["clock"]["external_polymarket_top"]["median_offset_ms"]
    assert abs(bin_off - TX_BINANCE_OFFSET_MS) < 2.0
    assert abs(clob_off - TX_CLOB_OFFSET_MS) < 2.0
    assert result["go_no_go"] == "GO"


# --------------------------------------------------------------------------- #
# Compression (needs the optional zstandard extra)
# --------------------------------------------------------------------------- #


def test_zstd_roundtrip_bit_identical(tmp_path):
    pytest.importorskip("zstandard")
    src = tmp_path / "blob.bin"
    src.write_bytes(bytes(range(256)) * 500)  # non-trivial content
    zst = tx.compress_file_zstd(src)
    assert zst.exists() and zst.stat().st_size > 0
    assert tx.verify_roundtrip(src, zst)
    # corrupt the archive → roundtrip must fail
    zst.write_bytes(zst.read_bytes()[:-3] + b"\x00\x00\x00")
    assert not tx.verify_roundtrip(src, zst)


def test_compress_trial_ratio_and_roundtrip(tmp_path):
    pytest.importorskip("zstandard")
    from model_lab import telonex_validate as tv

    tdir = tmp_path / "telonex"
    make_telonex_fixture(tdir, tmp_path / "journal")
    paths = Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                  out_dir=tmp_path / "out", telonex_dir=tdir)
    comp = tv.compress_trial(paths)
    assert comp["result"] == "PASS"
    assert comp["files"] > 0
    assert comp["roundtrip_ok"] == comp["files"]
    assert not comp["roundtrip_failed"]
    assert comp["ratio"] > 0
