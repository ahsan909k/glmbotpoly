"""Tests for the competitor operating-manual reconstruction (``competitors.manuals``).

Unit-tests the pure price-vs-book classifier + distribution helpers, then drives the
full manual over a synthetic competitor cache aligned to ``make_historical_fixture``'s
Telonex book tape: coverage counts, at-touch/crossing classification, entry-timing
distributions, merge-velocity, the official-P/L anchor (consistent → not suspect; a
deliberately-inconsistent official curve → red "suspect"), the post-Jun-5 slice ladder,
and the 10-window hand-trace. Fully offline (no network, no LightGBM).
"""

from __future__ import annotations

from datetime import date

import pytest

from model_lab import fixtures as fx
from model_lab.competitors import manuals as mn
from model_lab.competitors import manuals_report as mr
from model_lab.config import Paths

POST_JUN5 = date(2026, 6, 15)  # Telonex-owned (< 2026-07-04) AND in the post-Jun-5 slice


# --------------------------------------------------------------------------- pure
def test_price_class_buy():
    assert mn.price_class("BUY", 0.40, 0.40, 0.60) == "at_touch"   # at the bid
    assert mn.price_class("BUY", 0.50, 0.40, 0.60) == "inside"     # mid of a wide spread
    assert mn.price_class("BUY", 0.60, 0.40, 0.60) == "crossing"   # at the ask (marketable)
    assert mn.price_class("BUY", 0.404, 0.40, 0.60) == "at_touch"  # within half a tick of bid
    assert mn.price_class("BUY", 0.10, 0.20, 0.80) == "at_touch"   # below the bid


def test_price_class_sell():
    # SELL mirrors: passive at the ask, crossing at the bid.
    assert mn.price_class("SELL", 0.60, 0.40, 0.60) == "at_touch"
    assert mn.price_class("SELL", 0.50, 0.40, 0.60) == "inside"
    assert mn.price_class("SELL", 0.40, 0.40, 0.60) == "crossing"


def test_price_class_degenerate():
    assert mn.price_class("BUY", 0.5, 0.6, 0.4, ) == "unknown"     # crossed book
    assert mn.price_class("BUY", 0.5, 0.0, 0.6) == "unknown"       # zero bid


def test_distribution_helpers():
    d = mn._distribution([1, 2, 3, 4, 5], edges=[0, 2, 4, 6])
    assert d["n"] == 5 and d["p50"] == 3.0
    assert d["hist_counts"] == [1, 2, 2]  # [0,2):{1}, [2,4):{2,3}, [4,6]:{4,5}
    empty = mn._distribution([])
    assert empty["n"] == 0 and empty["p50"] is None


def test_ols_slope():
    assert abs(mn._ols_slope([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-9  # perfectly ramping
    assert mn._ols_slope([5.0]) is None


# --------------------------------------------------------------------------- setup
def _setup(tmp_path, monkeypatch, *, crossing=0, inconsistent=False, merge_heavy=True):
    telonex, hist, comp = tmp_path / "telonex", tmp_path / "hist", tmp_path / "competitors"
    res = fx.make_historical_fixture(telonex, hist, day=POST_JUN5)
    exp = fx.make_competitor_fills_fixture(
        comp, res, crossing_per_window=crossing, inconsistent=inconsistent, merge_heavy=merge_heavy)
    monkeypatch.setenv("MODEL_LAB_COMPETITORS_DIR", str(comp))
    paths = Paths(journal_dir=tmp_path, depth_dir=tmp_path, out_dir=tmp_path / "out",
                  hist_dir=hist, telonex_dir=telonex)
    entry = {"handle": exp["handle"], "address": exp["addr"],
             "profile": {"takerTierName": "Platinum", "weightedVolume": 1_000_000.0}}
    return paths, entry, exp


# --------------------------------------------------------------- integration
def test_manual_builds_counts_and_classification(tmp_path, monkeypatch):
    paths, entry, exp = _setup(tmp_path, monkeypatch, crossing=2)
    m = mn.build_manual(paths, entry)
    cov = m["coverage"]
    assert cov["our_fills_in_telonex_window"] == exp["n_fills"]
    assert cov["our_windows_traded"] == exp["n_windows"]
    pv = m["price_vs_book"]
    # every fill lands against a real fixture book (no book_missing).
    assert pv["windows_with_book"] == exp["n_windows"] and pv["windows_book_missing"] == 0
    assert pv["at_touch"] == exp["at_touch"]      # 2 touch buys / window
    assert pv["crossing"] == exp["crossing"]      # crossing_per_window taker buys / window
    assert pv["unknown_book"] == 0


def test_entry_timing_splits_maker_taker(tmp_path, monkeypatch):
    paths, entry, exp = _setup(tmp_path, monkeypatch, crossing=1)
    et = mn.build_manual(paths, entry)["entry_timing"]
    assert et["maker"]["n"] == 2 * exp["n_windows"]   # the two touch (maker) buys
    assert et["taker"]["n"] == exp["crossing"]         # the crossing (taker) buys
    assert et["taker"]["p50"] is not None and abs(et["taker"]["p50"] - 0.5) < 0.05  # k=150/300


def test_merge_velocity_present_and_positive(tmp_path, monkeypatch):
    paths, entry, exp = _setup(tmp_path, monkeypatch, merge_heavy=True)
    m = mn.build_manual(paths, entry)
    assert m["merge_focus"] is True
    mv = m["merge_velocity"]
    assert mv["merge_events"] == exp["n_windows"]
    assert mv["inventory_state_at_merge"]["matched_frac"] == 1.0  # both legs equal → matched
    turns = mv["capital_velocity"]["turns_per_day"]
    assert turns["n"] >= 1 and turns["p50"] is not None


def test_recon_contradicts_official_symmetric():
    # The fixed anchor rule flags BOTH directions and sign flips; consistent within tol.
    assert mn.recon_contradicts_official(100.0, 120.0) == (False, "")        # ≈ equal
    assert mn.recon_contradicts_official(100_000.0, 14_000.0)[0] is True     # recon ≫ official (wolf)
    assert mn.recon_contradicts_official(-36_000.0, 38_000.0)[0] is True     # recon ≪ official (takerner)
    assert mn.recon_contradicts_official(-229_000.0, 241_000.0)[0] is True   # opposite signs (bonereaper)
    assert mn.recon_contradicts_official(50_000.0, -50_000.0)[0] is True     # opposite signs, other way
    assert mn.recon_contradicts_official(1_000.0, 1_500.0) == (False, "")    # within the $2k floor
    assert mn.recon_contradicts_official(None, 100.0) == (False, "")         # missing → not flagged


def test_assert_anchor_consistent_fails_loud():
    # A contradiction WITHOUT a suspect flag must raise (the permanent guard).
    bad = {"official_available": True, "reconstructed_our_net_usd": -36_000.0,
           "official_window_usd": 38_000.0, "suspect": False, "slice_ladder": {}}
    with pytest.raises(mn.AnchorInconsistency):
        mn.assert_anchor_consistent(bad)
    # slice-only contradiction also raises.
    bad2 = {"official_available": True, "reconstructed_our_net_usd": 100.0, "official_window_usd": 120.0,
            "suspect": False, "slice_ladder": {"owner_our_series_reconstructed_net_usd": 100.0,
                                               "official_account_wide_slice_usd": 90_000.0, "suspect": False}}
    with pytest.raises(mn.AnchorInconsistency):
        mn.assert_anchor_consistent(bad2)
    # Properly-flagged contradiction passes.
    ok = dict(bad, suspect=True)
    mn.assert_anchor_consistent(ok)  # no raise


def test_anchor_consistent_not_suspect(tmp_path, monkeypatch):
    paths, entry, _ = _setup(tmp_path, monkeypatch, crossing=0, inconsistent=False)
    a = mn.build_manual(paths, entry)["anchor"]
    assert a["official_available"] is True
    assert a["reconstructed_our_net_usd"] > 0.0       # maker winner (pair-cost < 1 + merge)
    # official ≈ reconstruction (small gap) → not suspect, both directions.
    assert a["suspect"] is False
    lad = a["slice_ladder"]
    assert lad["suspect"] is False
    assert lad["owner_our_series_reconstructed_net_usd"] > 0.0
    mn.assert_anchor_consistent(a)  # the guard is happy


def test_anchor_inconsistent_is_flagged_red(tmp_path, monkeypatch):
    paths, entry, _ = _setup(tmp_path, monkeypatch, crossing=0, inconsistent=True)
    a = mn.build_manual(paths, entry)["anchor"]
    # reconstructed OUR-series net is a small positive but official is far away (−$50k)
    # → contradiction → OUR reconstruction flagged suspect (never their profile).
    assert a["reconstructed_our_net_usd"] > 0.0
    assert a["official_window_usd"] < 0.0
    assert a["suspect"] is True and a["suspect_reasons"]
    assert a["slice_ladder"]["suspect"] is True
    mn.assert_anchor_consistent(a)  # consistent flagging → guard passes


def test_hand_trace_has_book_rows(tmp_path, monkeypatch):
    paths, entry, exp = _setup(tmp_path, monkeypatch, crossing=1)
    traces = mn.build_manual(paths, entry)["hand_trace"]
    assert len(traces) == exp["n_windows"]  # only 4 resolvable windows
    for t in traces:
        assert t["fills"]
        for row in t["fills"]:
            assert "book_bid" in row and "book_ask" in row and row["class"] in (
                "at_touch", "inside", "crossing")


# --------------------------------------------------------------- report + provenance
def test_report_renders_and_body_is_fact_calc_only(tmp_path, monkeypatch):
    paths, entry, _ = _setup(tmp_path, monkeypatch, crossing=1)
    m = mn.build_manual(paths, entry)
    # The manual BODY (behavioral distributions) must be FACT+CALC only — no EST/UNV spans.
    body = mr.manual_body(m)
    assert "class='est'" not in body and "class='unv'" not in body
    assert "class='fact'" in body or "class='calc'" in body
    # Full page renders and DOES carry EST/UNV — but only in the anchor/inference sections.
    full = mr.build_manual_html(m)
    assert "<html" in full and "class='est'" in full


def test_report_red_discrepancy_on_suspect(tmp_path, monkeypatch):
    paths, entry, _ = _setup(tmp_path, monkeypatch, crossing=0, inconsistent=True)
    full = mr.build_manual_html(mn.build_manual(paths, entry))
    assert "OUR RECONSTRUCTION SUSPECT" in full and "reddisc" in full
