"""Offline tests for the Telonex full pull (``telonex_pull``) + coverage cross-check.

No network: a synthetic markets catalog is written to disk (so ``ensure_markets_catalog``
skips the download) and every ``/downloads/`` request is served real per-file parquet
bytes through a dict-backed ``fetch`` seam mimicking the two-step 302→S3 flow. The
compression paths ``importorskip("zstandard")``.
"""

from __future__ import annotations

import io as _io
import json
import random
import urllib.parse as _up
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from model_lab import telonex_coverage as tc
from model_lab import telonex_pull as tp
from model_lab.config import Paths
from model_lab.io import telonex as tx

BASE_DAY = date(2026, 7, 5)
DAY_STR = BASE_DAY.isoformat()
START = tx.day_start_epoch(BASE_DAY)


def _paths(base: Path) -> Paths:
    return Paths(journal_dir=base / "journal", depth_dir=base / "depth",
                 out_dir=base / "out", telonex_dir=base / "telonex")


def _day_str(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()


def _win(asset: str, dur: str, epoch: int, channels=("quotes", "trades")) -> dict:
    return {"asset": asset, "dur": dur, "epoch": epoch,
            "slug": f"{asset}-updown-{dur}-{epoch}", "channels": tuple(channels)}


def _cat_rows(wins: list[dict], *, extra_rows=()) -> list[dict]:
    rows = []
    for w in wins:
        d = _day_str(w["epoch"])
        nxt = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        row = {"slug": w["slug"], "outcome_0": "Up", "outcome_1": "Down",
               "asset_id_0": f"UP-{w['slug']}", "asset_id_1": f"DN-{w['slug']}",
               "market_id": f"0x{abs(hash(w['slug'])) % (16 ** 8):08x}",
               "quotes_from": "", "quotes_to": "", "trades_from": "", "trades_to": ""}
        for ch in w["channels"]:
            row[f"{ch}_from"] = d
            row[f"{ch}_to"] = nxt
        rows.append(row)
    rows.extend(extra_rows)
    return rows


def _write_catalog(telonex_dir: Path, rows: list[dict]) -> Path:
    p = tx.catalog_path(telonex_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p, engine="pyarrow", index=False)
    return p


# --- synthetic parquet content -------------------------------------------------

_TS0 = 1_783_252_800_000_000  # µs


def _quotes_df(n=20, *, gap=False, gap_secs=8, nonmono=False, empty=False) -> pd.DataFrame:
    if empty:
        return pd.DataFrame({"timestamp_us": pd.Series([], dtype="int64"),
                             "bid_price": pd.Series([], dtype="object"),
                             "ask_price": pd.Series([], dtype="object")})
    if gap:
        half = max(1, n // 2)
        ts = [_TS0 + i * 100_000 for i in range(half)]
        jump = ts[-1] + int(gap_secs * 1_000_000)
        ts += [jump + i * 100_000 for i in range(n - half)]
    else:
        ts = [_TS0 + i * 100_000 for i in range(n)]
    if nonmono and len(ts) >= 3:
        ts[1], ts[2] = ts[2], ts[1]
    return pd.DataFrame({"timestamp_us": ts, "bid_price": ["0.40"] * len(ts), "ask_price": ["0.42"] * len(ts)})


def _trades_df(n=10, *, dup=False, nonmono=False, empty=False) -> pd.DataFrame:
    if empty:
        return pd.DataFrame({"timestamp_us": pd.Series([], dtype="int64"),
                             "trade_id": pd.Series([], dtype="int64")})
    ts = [_TS0 + i * 1_000_000 for i in range(n)]
    ids = list(range(5000, 5000 + n))
    if dup and n >= 2:
        ids[1] = ids[0]
    if nonmono and n >= 2:
        ts[1] = ts[0] - 1000
    return pd.DataFrame({"timestamp_us": ts, "trade_id": ids})


def _pq_bytes(df: pd.DataFrame) -> bytes:
    buf = _io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    return buf.getvalue()


def _write_raw(telonex_dir, channel, slug, outcome, day_str, df) -> Path:
    p = tx.polymarket_raw_path(telonex_dir, channel, slug, outcome, day_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", index=False)
    return p


def _build_store(wins, *, outcomes=("Up", "Down"), missing=(), content=None) -> dict:
    store: dict = {}
    content = content or {}
    for w in wins:
        d = _day_str(w["epoch"])
        for ch in w["channels"]:
            for oc in outcomes:
                if (ch, w["slug"], oc) in missing:
                    continue
                df = content.get((ch, w["slug"], oc))
                if df is None:
                    df = _quotes_df() if ch == "quotes" else _trades_df()
                store[(ch, w["slug"], oc, d)] = _pq_bytes(df)
    return store


def _make_fetch(store, *, remaining=None, fail_s3_status=None):
    def fetch(url, headers=None, follow_redirects=True):
        is_s3 = url.endswith("|s3")
        base = url[:-3] if is_s3 else url
        parsed = _up.urlparse(base)
        q = {k: v[0] for k, v in _up.parse_qs(parsed.query).items()}
        path = parsed.path
        if "/downloads/" in path:
            if is_s3 and fail_s3_status is not None:
                raise tx.TelonexHTTPError(fail_s3_status, {"X-Downloads-Remaining": "0"}, url)
            parts = path.split("/downloads/")[1].split("/")  # polymarket/{channel}/{date}
            key = (parts[1], q.get("slug", ""), q.get("outcome", ""), parts[2])
            body = store.get(key)
            if body is None:
                hdrs = {"X-Downloads-Remaining": str(remaining)} if remaining is not None else {}
                raise tx.TelonexHTTPError(404, hdrs, url)
            if is_s3 or follow_redirects:
                return tx.TelonexResponse(200, {}, _io.BytesIO(body))
            hdrs = {"Location": url + "|s3"}
            if remaining is not None:
                hdrs["X-Downloads-Remaining"] = str(remaining)
            return tx.TelonexResponse(302, hdrs, _io.BytesIO(b""))
        raise tx.TelonexHTTPError(404, {}, url)
    return fetch


def _load_cov(paths: Paths) -> dict:
    return json.loads((paths.out_dir / "telonex" / COVERAGE_PATH).read_text(encoding="utf-8"))


COVERAGE_PATH = "coverage.json"


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #


def test_enumerate_pull_windows_multi_series(tmp_path):
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START), _win("eth", "5m", START + 300),
            _win("btc", "15m", START), _win("eth", "15m", START + 900)]
    # empty-availability window (never went live) + a 1h window + an unrelated market
    empty = {"slug": f"btc-updown-5m-{START + 600}", "outcome_0": "Up", "outcome_1": "Down",
             "asset_id_0": "U", "asset_id_1": "D", "market_id": "0x00",
             "quotes_from": "", "quotes_to": "", "trades_from": "", "trades_to": ""}
    one_h = _cat_rows([_win("btc", "1h", START)])[0]
    other = {"slug": "will-x-happen", "outcome_0": "Yes", "outcome_1": "No",
             "asset_id_0": "Y", "asset_id_1": "N", "market_id": "0x11",
             "quotes_from": "2024-01-01", "quotes_to": "2024-02-01",
             "trades_from": "2024-01-01", "trades_to": "2024-02-01"}
    cat = _write_catalog(paths.telonex_dir, _cat_rows(wins, extra_rows=[empty, one_h, other]))
    series = ["btc-updown-5m", "eth-updown-5m", "btc-updown-15m", "eth-updown-15m"]
    got = tx.enumerate_pull_windows(cat, series, channels=["quotes", "trades"])
    assert {w["series"] for w in got} == set(series)          # exactly the 4 in-scope series
    assert len(got) == 4                                       # empty/1h/other excluded
    by_slug = {w["slug"]: w for w in got}
    w0 = by_slug[f"btc-updown-5m-{START}"]
    assert set(w0["channels"]) == {"quotes", "trades"}
    assert w0["channels"]["quotes"]["covers_day"] is True
    assert w0["up_asset_id"] == f"UP-btc-updown-5m-{START}"
    assert w0["down_asset_id"] == f"DN-btc-updown-5m-{START}"
    assert w0["open_ms"] == START * 1000


def test_enumerate_quotes_only_when_trades_absent(tmp_path):
    paths = _paths(tmp_path)
    w = _win("btc", "5m", START, channels=("quotes",))  # trades availability empty
    cat = _write_catalog(paths.telonex_dir, _cat_rows([w]))
    got = tx.enumerate_pull_windows(cat, ["btc-updown-5m"], channels=["quotes", "trades"])
    assert len(got) == 1 and set(got[0]["channels"]) == {"quotes"}


# --------------------------------------------------------------------------- #
# Skip predicates
# --------------------------------------------------------------------------- #


def test_skip_predicates(tmp_path):
    raw = tmp_path / "a.parquet"
    zst = tx.zst_sibling(raw)
    assert zst.name == "a.parquet.zst"
    assert not tx.raw_or_zst_exists(raw) and not tx.zst_finalized(raw)          # A: nothing
    raw.write_bytes(b"x")
    assert tx.raw_or_zst_exists(raw) and not tx.zst_finalized(raw)              # B: raw only
    zst.write_bytes(b"z")
    assert tx.raw_or_zst_exists(raw) and not tx.zst_finalized(raw)              # C: raw + zst
    raw.unlink()
    assert tx.raw_or_zst_exists(raw) and tx.zst_finalized(raw)                  # D: zst only


# --------------------------------------------------------------------------- #
# Full per-day pipeline
# --------------------------------------------------------------------------- #


def _run_pull(paths, wins, fetch, **kw):
    kw.setdefault("cap_bytes", 10 ** 12)
    kw.setdefault("jobs", 1)
    kw.setdefault("roundtrip_sample", 100)
    kw.setdefault("api_key", "k")
    return tp.pull(paths, series=sorted({w["asset"] + "-updown-" + w["dur"] for w in wins}),
                   channels=["quotes", "trades"], outcomes=["Up", "Down"], fetch=fetch, **kw)


def test_pull_download_compress_roundtrip_delete(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START), _win("btc", "5m", START + 300)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    summary = _run_pull(paths, wins, _make_fetch(_build_store(wins)))
    assert summary["status"] == "ok"
    for w in wins:
        for ch in ("quotes", "trades"):
            for oc in ("Up", "Down"):
                raw = tx.polymarket_raw_path(paths.telonex_dir, ch, w["slug"], oc, DAY_STR)
                assert tx.zst_sibling(raw).exists(), raw
                assert not raw.exists(), f"raw not deleted: {raw}"
    cov = _load_cov(paths)
    day = cov["series"]["btc-updown-5m"]["days"][DAY_STR]
    assert day["status"] == "PASS"
    assert day["downloaded_files"] == 8 and day["missing_files"] == 0
    assert day["roundtrip_checked"] > 0 and day["roundtrip_ok"] == day["roundtrip_checked"]
    assert day["raw_deleted"] is True
    assert (paths.out_dir / "telonex" / "full_pull_report.html").exists()


def test_pull_resume_skips_zst_no_redownload(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    _run_pull(paths, wins, _make_fetch(_build_store(wins)))

    def boom(url, headers=None, follow_redirects=True):
        if "/downloads/" in url:
            raise AssertionError("resume must not re-download a finalized file")
        raise tx.TelonexHTTPError(404, {}, url)

    summary = _run_pull(paths, wins, boom)   # nothing downloaded → no AssertionError
    assert summary["status"] == "ok"


def test_pull_resume_after_crash_before_delete(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    _run_pull(paths, wins, _make_fetch(_build_store(wins)), delete_raw=False)  # state C: raw + zst
    raw = tx.polymarket_raw_path(paths.telonex_dir, "quotes", wins[0]["slug"], "Up", DAY_STR)
    assert raw.exists() and tx.zst_sibling(raw).exists()

    def boom(url, headers=None, follow_redirects=True):
        if "/downloads/" in url:
            raise AssertionError("reconcile must not re-download")
        raise tx.TelonexHTTPError(404, {}, url)

    _run_pull(paths, wins, boom, delete_raw=True)  # reconcile finishes the delete, no network
    assert not raw.exists() and tx.zst_sibling(raw).exists()


def test_missing_window_flagged_and_fails_non_recent(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    # 3 distinct days so the oldest is NOT within the recent-day lag → a missing file FAILs.
    wins = [_win("btc", "5m", START + k * 86_400) for k in range(3)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    missing = {("trades", wins[0]["slug"], "Up")}
    _run_pull(paths, wins, _make_fetch(_build_store(wins, missing=missing)))
    cov = _load_cov(paths)
    day0 = cov["series"]["btc-updown-5m"]["days"][_day_str(wins[0]["epoch"])]
    assert day0["missing_files"] == 1
    assert any("/Up/trades" in m for m in day0["missing_windows"])
    assert day0["status"] == "FAIL"      # 1/4 = 25% > 20% and the day is not recent
    assert any(a["level"] == "FAIL" and a["day"] == _day_str(wins[0]["epoch"]) for a in cov["alerts"])


# --------------------------------------------------------------------------- #
# Integrity classification (direct, on raw parquet)
# --------------------------------------------------------------------------- #


def _integrity(base: Path, content: dict) -> dict:
    paths = _paths(base)
    scratch = paths.out_dir / "telonex" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    slug = f"btc-updown-5m-{START}"
    w = {"slug": slug, "open_day": DAY_STR, "channels": {"quotes": {}, "trades": {}}}
    specs = tp._expected_specs(paths.telonex_dir, w, ["Up", "Down"])
    for s in specs:
        df = content.get((s["channel"], s["outcome"]))
        if df is not None:
            _write_raw(paths.telonex_dir, s["channel"], s["slug"], s["outcome"], DAY_STR, df)
    rec = tp._day_record(specs, {}, expected_windows=1, scratch=scratch,
                         roundtrip_sample=0, mirror_sample=0, rng=random.Random(0))
    st, probs, warns = tp._classify_day(rec, recent_day=False)
    rec["status"], rec["problems"], rec["warnings"] = st, probs, warns
    return rec


def _clean_content() -> dict:
    return {("quotes", "Up"): _quotes_df(), ("quotes", "Down"): _quotes_df(),
            ("trades", "Up"): _trades_df(), ("trades", "Down"): _trades_df()}


def test_integrity_pass(tmp_path):
    rec = _integrity(tmp_path / "clean", _clean_content())
    assert rec["status"] == "PASS"
    assert rec["rows_total"] > 0 and rec["dup_trade_ids"] == 0 and rec["nonmonotonic_ts"] == 0


def test_gap_below_30s_no_longer_warns(tmp_path):
    # an 8s gap: the >5s count is kept (liquidity signal), but it no longer forces WARN
    c = _clean_content()
    c[("quotes", "Up")] = _quotes_df(gap=True, gap_secs=8)
    rec = _integrity(tmp_path / "gap", c)
    assert rec["gaps_gt_5s"] >= 1 and rec["max_stale_secs"] >= 8
    assert rec["status"] == "PASS"


def test_stale_stretch_over_30s_warns(tmp_path):
    c = _clean_content()
    c[("quotes", "Up")] = _quotes_df(gap=True, gap_secs=45)
    rec = _integrity(tmp_path / "stale", c)
    assert rec["max_stale_secs"] >= 45
    cov = {"series": {"btc-updown-5m": {"days": {DAY_STR: rec}, "expected_days": 1}}}
    tp._apply_gap_labels(cov)
    assert rec["status"] == "WARN" and any("stale stretch" in w for w in rec["warnings"])


def test_gap_rate_outlier_warns():
    days = {}
    for i in range(20):   # a normal series: low, mildly-varying >5s-gap rates
        days[f"2026-01-{i + 1:02d}"] = {"gaps_gt_5s": 3 + (i % 5), "expected_windows": 100,
                                        "max_stale_secs": 6.0, "problems": [], "warnings": [], "status": "PASS"}
    days["2026-02-01"] = {"gaps_gt_5s": 900, "expected_windows": 100, "max_stale_secs": 6.0,
                          "problems": [], "warnings": [], "status": "PASS"}   # a clear outlier day
    cov = {"series": {"s": {"days": days, "expected_days": len(days)}}}
    tp._apply_gap_labels(cov)
    assert days["2026-02-01"]["status"] == "WARN" and any("outlier" in w for w in days["2026-02-01"]["warnings"])
    assert days["2026-01-01"]["status"] == "PASS"   # a normal day is not flagged


def test_integrity_dup_trade_id_fails(tmp_path):
    c = _clean_content()
    c[("trades", "Up")] = _trades_df(dup=True)
    rec = _integrity(tmp_path / "dup", c)
    assert rec["status"] == "FAIL" and rec["dup_trade_ids"] >= 1


def test_integrity_nonmonotonic_fails(tmp_path):
    c = _clean_content()
    c[("quotes", "Down")] = _quotes_df(nonmono=True)
    rec = _integrity(tmp_path / "nm", c)
    assert rec["status"] == "FAIL" and rec["nonmonotonic_ts"] >= 1


def test_integrity_empty_file_fails(tmp_path):
    c = _clean_content()
    c[("trades", "Down")] = _trades_df(empty=True)
    rec = _integrity(tmp_path / "empty", c)
    assert rec["status"] == "FAIL" and rec["empty_files"]


def test_window_restricted_gaps_excludes_preopen(tmp_path):
    lo = START * 1_000_000                         # window open, µs
    hi = lo + 300 * 1_000_000                       # 5-minute window
    pre = [lo - 3600 * 1_000_000 + i * 600 * 1_000_000 for i in range(5)]   # sparse pre-open (600s apart)
    inw = [lo + i * 100_000 for i in range(200)]                            # dense in-window (100ms)
    df = pd.DataFrame({"timestamp_us": pre + inw, "bid_price": ["0.40"] * 205, "ask_price": ["0.42"] * 205})
    p = tmp_path / "q.parquet"
    df.to_parquet(p, engine="pyarrow", index=False)
    assert tp._integrity_read(p, "quotes")["gaps"] >= 5            # whole-file inflated by pre-open
    assert tp._integrity_read(p, "quotes", window_us=(lo, hi))["gaps"] == 0   # in-window is dense


# --------------------------------------------------------------------------- #
# Projection + budget
# --------------------------------------------------------------------------- #


def test_projection_fits(tmp_path):
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START + i * 300) for i in range(4)] + \
           [_win("btc", "15m", START + i * 900) for i in range(2)]
    cat = _write_catalog(paths.telonex_dir, _cat_rows(wins))
    windows = tx.enumerate_pull_windows(cat, ["btc-updown-5m", "btc-updown-15m"], channels=["quotes", "trades"])
    per_zst, per_raw, ref_dur = tp._measure_per_window(paths.telonex_dir, ["quotes", "trades"], "btc-updown-5m")
    proj = tp.project_pull(windows, per_zst, per_raw, ref_dur, cap_bytes=10 ** 12)
    assert proj["fits_cap"] and not proj["shrink_applied"] and proj["zst_bytes"] > 0
    # 15m per-window ≈ 3× the 5m per-window (duration scaling)
    d = proj["series_detail"]
    assert d["btc-updown-15m"]["est_zst_bytes"] > 0 and d["btc-updown-5m"]["est_zst_bytes"] > 0


def test_budget_shrink_drops_oldest_15m_first(tmp_path):
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    for k in range(3):  # three earlier 15m days
        wins.append(_win("btc", "15m", START - (k + 1) * 86_400))
    cat = _write_catalog(paths.telonex_dir, _cat_rows(wins))
    windows = tx.enumerate_pull_windows(cat, ["btc-updown-5m", "btc-updown-15m"], channels=["quotes", "trades"])
    per_zst, per_raw, ref_dur = tp._measure_per_window(paths.telonex_dir, ["quotes", "trades"], "btc-updown-5m")
    proj = tp.project_pull(windows, per_zst, per_raw, ref_dur, cap_bytes=10 ** 12)
    kept, shrink = tp.apply_budget_shrink(proj, windows, per_zst, ref_dur, 2, cap_bytes=proj["zst_bytes"] - 1)
    assert shrink["shrink_applied"] and shrink["dropped"]
    assert all(d["series"] == "btc-updown-15m" for d in shrink["dropped"])       # never 5m
    oldest = min(_day_str(START - (k + 1) * 86_400) for k in range(3))
    assert shrink["dropped"][0]["day"] == oldest                                 # oldest first
    assert any(w["dur"] == "5m" for w in kept)                                    # 5m survives


# --------------------------------------------------------------------------- #
# Safety: disk / quota / dry-run
# --------------------------------------------------------------------------- #


def test_disk_preflight_aborts(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    monkeypatch.setattr(tx, "free_disk_bytes", lambda p: 1)
    summary = _run_pull(paths, wins, _make_fetch(_build_store(wins)))
    assert summary["status"] == "aborted_disk"


def test_download_limit_graceful_stop(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    # 302 (probe) succeeds, but every S3 body fetch 403s → TelonexDownloadLimit → graceful abort.
    summary = _run_pull(paths, wins, _make_fetch(_build_store(wins), fail_s3_status=403))
    assert summary["status"] == "aborted_limit"
    cov = _load_cov(paths)
    assert cov["run"]["aborted"] == "limit"


def test_quota_gate_stops_when_insufficient(tmp_path):
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]          # 1 window × 2 ch × 2 oc = 4 expected files
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    summary = _run_pull(paths, wins, _make_fetch(_build_store(wins), remaining=3))  # 3 < 4
    assert summary["status"] == "aborted_quota"
    for ch in ("quotes", "trades"):
        for oc in ("Up", "Down"):
            raw = tx.polymarket_raw_path(paths.telonex_dir, ch, wins[0]["slug"], oc, DAY_STR)
            assert not raw.exists() and not tx.zst_sibling(raw).exists()   # nothing downloaded


def test_dry_run_projects_without_downloading(tmp_path):
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))

    def boom(url, headers=None, follow_redirects=True):
        if "/downloads/" in url:
            raise AssertionError("dry-run must not touch /downloads/")
        raise tx.TelonexHTTPError(404, {}, url)

    summary = tp.pull(paths, series=["btc-updown-5m"], channels=["quotes", "trades"],
                      outcomes=["Up", "Down"], cap_bytes=10 ** 12, jobs=1, dry_run=True,
                      api_key=None, fetch=boom)
    assert summary["status"] == "dry_run" and summary["expected_files"] == 4
    assert (paths.out_dir / "telonex" / "coverage.json").exists()
    assert (paths.out_dir / "telonex" / "full_pull_report.html").exists()


# --------------------------------------------------------------------------- #
# Coverage cross-check + key hygiene
# --------------------------------------------------------------------------- #


def test_coverage_crosscheck_flags_deleted_zst(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    _run_pull(paths, wins, _make_fetch(_build_store(wins)))
    tx.zst_sibling(tx.polymarket_raw_path(paths.telonex_dir, "trades", wins[0]["slug"], "Up", DAY_STR)).unlink()
    res = tc.coverage(paths, series=["btc-updown-5m"], channels=["quotes", "trades"], outcomes=["Up", "Down"])
    assert res["status"] in ("ok", "alerts")
    cov = _load_cov(paths)
    day = cov["series"]["btc-updown-5m"]["days"][DAY_STR]
    assert day["missing_files"] == 1 and day["status"] in ("WARN", "FAIL")
    assert any("/Up/trades" in m for m in day["missing_windows"])


def test_coverage_deep_reads_zst(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    _run_pull(paths, wins, _make_fetch(_build_store(wins)))
    res = tc.coverage(paths, series=["btc-updown-5m"], channels=["quotes", "trades"],
                      outcomes=["Up", "Down"], deep=True)
    cov = _load_cov(paths)
    day = cov["series"]["btc-updown-5m"]["days"][DAY_STR]
    assert res["status"] == "ok" and day["status"] == "PASS"
    assert day["rows_total"] > 0 and day["deep_checked"] == 4   # re-read the .zst contents


def test_key_never_in_outputs(tmp_path):
    pytest.importorskip("zstandard")
    paths = _paths(tmp_path)
    wins = [_win("btc", "5m", START)]
    _write_catalog(paths.telonex_dir, _cat_rows(wins))
    secret = "TOPSECRET_KEY_ABC123"
    _run_pull(paths, wins, _make_fetch(_build_store(wins)), api_key=secret)
    cov_text = (paths.out_dir / "telonex" / "coverage.json").read_text(encoding="utf-8")
    html = (paths.out_dir / "telonex" / "full_pull_report.html").read_text(encoding="utf-8")
    assert secret not in cov_text and secret not in html
