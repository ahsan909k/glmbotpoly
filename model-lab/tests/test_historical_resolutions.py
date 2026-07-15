"""Tests for the official-resolution module (Part A.2).

- the Telonex catalog decodes to the correct winner + feeds the resolution cache;
- the stratified catalog-vs-CLOB-API validation matches on agreeing data and **hard-fails**
  on any mismatch (offline ``fetch`` seam — no network);
- the journal overlap exact-match validation (clean vs mismatch);
- the API cache is resumable (a re-run does not re-fetch).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from model_lab import fixtures as fx
from model_lab import historical_common as hc
from model_lab import historical_resolutions as hr
from model_lab.config import Paths


def _paths(tmp_path) -> Paths:
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
                telonex_dir=tmp_path / "telonex")


def _setup(tmp_path, day=fx.HIST_FIX_DAY):
    res = fx.make_historical_fixture(tmp_path / "telonex", tmp_path / "aggtrades", day=day)
    fx.write_historical_coverage(tmp_path / "out", res["coverage"])
    return _paths(tmp_path), res


def _api_fetch(res_map: dict, *, flip_cid: str | None = None):
    """An offline CLOB ``fetch`` seam returning each condition_id's official winner (from the
    fixture), optionally flipping one to force a mismatch."""
    cid_out = {v["condition_id"]: v["outcome"] for v in res_map.values()}

    def fetch(url: str):
        cid = url.rstrip("/").rsplit("/", 1)[-1]
        out = cid_out.get(cid)
        if out is None:
            from model_lab.io.polymarket import PolymarketNotFound
            raise PolymarketNotFound(url)
        if flip_cid is not None and cid == flip_cid:
            out = "Up" if out == "Down" else "Down"
        other = "Down" if out == "Up" else "Up"
        return {"tokens": [{"outcome": out, "winner": True}, {"outcome": other, "winner": False}]}

    return fetch


# --- catalog decode ---------------------------------------------------------
def test_catalog_decode(tmp_path):
    paths, res = _setup(tmp_path)
    rm = hr.load_resolution_map(paths)
    assert len(rm) == 12  # 6 windows × 2 series
    for key, official in res["res_map"].items():
        assert rm[key]["outcome"] == official["outcome"]
    assert hr.catalog_cache_path(paths).exists()


# --- API validation (offline) -----------------------------------------------
def test_api_validation_matches(tmp_path):
    paths, res = _setup(tmp_path)
    rep = hr.validate_vs_api(paths, min_sample=5, fetch=_api_fetch(res["res_map"]))
    assert rep["ok"] is True
    assert rep["mismatch"] == 0
    assert rep["api_cross_checked"] >= 1
    assert set(rep["series_covered"]) == {"BTC-5m", "ETH-5m"}


def test_api_validation_hard_fails_on_mismatch(tmp_path):
    paths, res = _setup(tmp_path)
    flip = next(iter(res["res_map"].values()))["condition_id"]
    rep = hr.validate_vs_api(paths, min_sample=5, fetch=_api_fetch(res["res_map"], flip_cid=flip))
    assert rep["ok"] is False
    assert rep["mismatch"] >= 1
    assert rep["mismatch_examples"]


def test_api_cache_resumable(tmp_path):
    paths, res = _setup(tmp_path)
    hr.validate_vs_api(paths, min_sample=5, fetch=_api_fetch(res["res_map"]))
    n1 = sum(1 for _ in hr.api_cache_path(paths).open(encoding="utf-8"))
    calls = {"n": 0}

    def counting(url):
        calls["n"] += 1
        return _api_fetch(res["res_map"])(url)

    hr.validate_vs_api(paths, min_sample=5, fetch=counting)
    assert calls["n"] == 0  # everything served from the cache
    n2 = sum(1 for _ in hr.api_cache_path(paths).open(encoding="utf-8"))
    assert n2 == n1


# --- journal overlap exact-match --------------------------------------------
def _journal(res, tmp_path, day, *, flip_first=False):
    recs = []
    for i, w in enumerate([x for x in res["windows"] if not x["coverage_excluded"]]):
        om, cm = w["open_ms"], w["open_ms"] + 300_000
        o = w["outcome"]
        if flip_first and i == 0:
            o = "Up" if o == "Down" else "Down"
        mkt = {"window": {"series": "BTC-5m", "open_time": om}, "close_time": cm}
        recs.append((cm, {"type": "window", "market": mkt, "lifecycle": {"Resolved": {"outcome": o}}}))
    fx._write_journal(tmp_path / "journal" / f"journal-{day.strftime('%Y%m%d')}-120000-00000.jsonl.gz", recs)


def test_journal_overlap_clean(tmp_path):
    day = date(2026, 7, 4)
    paths, res = _setup(tmp_path, day=day)
    _journal(res, tmp_path, day, flip_first=False)
    rm = hr.load_resolution_map(paths, ("BTC-5m",))
    rep = hr.validate_vs_journal(paths, rm, ("BTC-5m",))
    assert rep["checked"] >= 1 and rep["mismatches"] == 0 and rep["ok"]


def test_journal_overlap_mismatch(tmp_path):
    day = date(2026, 7, 4)
    paths, res = _setup(tmp_path, day=day)
    _journal(res, tmp_path, day, flip_first=True)
    rm = hr.load_resolution_map(paths, ("BTC-5m",))
    rep = hr.validate_vs_journal(paths, rm, ("BTC-5m",))
    assert rep["mismatches"] >= 1 and rep["ok"] is False
