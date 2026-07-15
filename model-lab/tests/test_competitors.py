"""Offline unit tests for the competitor-analysis tool (no network)."""

from __future__ import annotations

import json

from model_lab.competitors import analyze, api, report, resolve
from model_lab.competitors.analyze_tape import _is_grid


# --- fees / grid -----------------------------------------------------------
def test_taker_fee_reference_points():
    # CLAUDE.md §7: shares·0.07·p·(1−p); per 100 shares $1.75 @ 0.50, symmetric.
    assert analyze.taker_fee(100, 0.5) == 1.75
    assert analyze.taker_fee(100, 0.25) == round(100 * 0.07 * 0.25 * 0.75, 5)
    assert analyze.taker_fee(100, 0.25) == analyze.taker_fee(100, 0.75)
    assert analyze.taker_fee(5, 0.01) >= 1e-5          # min floor
    assert analyze.taker_fee(0, 0.5) == 0.0


def test_is_grid_distinguishes_vwap():
    assert _is_grid(0.87) and _is_grid(0.50) and _is_grid(0.01)
    assert _is_grid(0.961)                               # on the 0.001 sub-grid (>0.96)
    assert _is_grid(0.150000017)                         # real single-level fill (float noise)
    assert not _is_grid(0.9615)                          # between ticks — not a valid price
    assert not _is_grid(0.7896441019)                    # VWAP of a multi-level take
    assert not _is_grid(0.0) and not _is_grid(1.0)


# --- maker/taker match key -------------------------------------------------
def test_match_key_ignores_endpoint_specific_fields():
    activity = {"transactionHash": "0xabc", "asset": "123", "side": "BUY",
                "size": 5.37, "type": "TRADE", "usdcSize": 4.67}
    trade = {"transactionHash": "0xabc", "asset": "123", "side": "BUY", "size": 5.37,
             "price": 0.87, "name": "x"}  # no type / usdcSize
    assert analyze._match_key(activity) == analyze._match_key(trade)


# --- archetype gate --------------------------------------------------------
def _acct(our_trades, taker, two_sided, behavior="holds to resolution + redeems"):
    return {
        "totals": {"our_series_trades": our_trades},
        "maker_taker": {"our_taker_frac_count": taker},
        "pair_discipline": {"two_sided_frac": two_sided},
        "merge_hold": {"behavior": behavior},
    }


def test_archetype_maker_taker_and_lowsample():
    assert report.derive_archetype(_acct(5000, 0.01, 0.99))["archetype"].startswith("Market-maker")
    assert report.derive_archetype(_acct(5000, 0.92, 0.10))["archetype"].startswith("Directional")
    assert report.derive_archetype(_acct(5000, 0.5, 0.6))["archetype"].startswith("Hybrid")
    low = report.derive_archetype(_acct(50, 0.9, 0.1))
    assert low["confidence"] == "low" and "insufficient" in low["archetype"].lower()


def test_stats_and_pct():
    s = analyze._stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["median"] == 2.5 and s["mean"] == 2.5
    assert analyze._stats([])["n"] == 0
    assert report._pct(0.234) == "23.4%" and report._pct(None) == "—"


# --- resolution logic ------------------------------------------------------
def test_address_shapes():
    assert resolve.is_full_address("0x" + "a" * 40)
    assert not resolve.is_full_address("0x" + "a" * 39)
    assert resolve._looks_truncated_address("0x50f7")
    assert not resolve._looks_truncated_address("wolf9478")


def test_exact_profile_match_is_case_insensitive():
    profiles = [
        {"name": "Bonereaper5", "proxyWallet": "0x5"},
        {"name": "Bonereaper", "proxyWallet": "0x0"},
    ]
    exact, others = resolve._exact_profile(profiles, "bonereaper")
    assert exact["proxyWallet"] == "0x0"
    assert any(o["name"] == "Bonereaper5" for o in others)


# --- api helpers -----------------------------------------------------------
def test_as_list_and_build():
    assert api._as_list([{"a": 1}, "x", {"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert api._as_list({"data": [{"a": 1}]}) == [{"a": 1}]
    assert api._as_list({"nope": 1}) == []
    url = api._build(api.DATA_BASE, "/trades", {"user": "0x1", "limit": 500, "skip": None})
    assert url.startswith("https://data-api.polymarket.com/trades?") and "skip" not in url


# --- end-to-end analyze on a synthetic account -----------------------------
def test_analyze_account_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_COMPETITORS_DIR", str(tmp_path))
    addr = "0x" + "1" * 40
    d = tmp_path / addr
    d.mkdir()
    open_epoch = 1783690500  # a btc-5m window open (multiple of 300)
    slug = f"btc-updown-5m-{open_epoch}"
    cond = "0xcond"
    # two-sided pair build at combined 0.98 (maker), then redeem the winner.
    acts = [
        {"timestamp": open_epoch + 10, "type": "TRADE", "conditionId": cond, "slug": slug,
         "asset": "up", "side": "BUY", "outcome": "Up", "size": 100, "usdcSize": 55,
         "price": 0.55, "transactionHash": "0xh1"},
        {"timestamp": open_epoch + 20, "type": "TRADE", "conditionId": cond, "slug": slug,
         "asset": "dn", "side": "BUY", "outcome": "Down", "size": 100, "usdcSize": 43,
         "price": 0.43, "transactionHash": "0xh2"},
        {"timestamp": open_epoch + 320, "type": "REDEEM", "conditionId": cond, "slug": slug,
         "asset": "", "side": "", "outcome": "", "size": 100, "usdcSize": 100,
         "transactionHash": "0xh3"},
    ]
    (d / "activity.jsonl").write_text("\n".join(json.dumps(a) for a in acts), encoding="utf-8")
    (d / "taker_trades.jsonl").write_text("", encoding="utf-8")  # all maker
    (d / "positions.json").write_text("[]", encoding="utf-8")

    out = analyze.analyze_account({"handle": "syn", "address": addr, "profile": {}})
    assert out["coverage"]["our_series_trade_frac_count"] == 1.0
    assert out["maker_taker"]["our_taker_frac_count"] == 0.0     # no taker fills
    assert out["pair_discipline"]["two_sided_windows"] == 1
    assert abs(out["pair_discipline"]["pair_cost"]["median"] - 0.98) < 1e-6
    # cash flow: -55 -43 +100 = +2 realized
    assert abs(out["pnl"]["realized_cashflow_usd"] - 2.0) < 1e-6
    assert out["merge_hold"]["behavior"].startswith("holds")


# --- official PnL endpoint + host/seam -------------------------------------
def test_user_pnl_uses_pnl_host_and_seam():
    captured = {}

    def fake(url):
        captured["url"] = url
        return [{"t": 1, "p": 2.0}, {"t": 2, "p": 3.5}]

    client = api.PolymarketClient(fetch=fake, delay_s=0.0)
    out = client.user_pnl("0xabc", interval="all", fidelity="1d")
    assert captured["url"].startswith(api.USER_PNL_BASE + "/user-pnl?")
    assert "user_address=0xabc" in captured["url"]
    assert "interval=all" in captured["url"] and "fidelity=1d" in captured["url"]
    assert out == [{"t": 1, "p": 2.0}, {"t": 2, "p": 3.5}]


# --- local slug classifier (SOL / 1h / 4h no longer collapse) ---------------
def test_classify_slug():
    assert analyze.classify_slug("sol-updown-1h-1783690800") == ("SOL-1h", "sol", 3600, 1783690800)
    assert analyze.classify_slug("btc-updown-4h-1783690800") == ("BTC-4h", "btc", 14400, 1783690800)
    assert analyze.classify_slug("xrp-updown-15m-1783690800")[0] == "XRP-15m"
    # parity with the OUR-series keys used across the pipeline
    assert analyze.classify_slug("btc-updown-5m-1783690800")[0] == "BTC-5m"
    assert analyze.classify_slug("eth-updown-15m-1783690800")[0] == "ETH-15m"
    assert analyze.classify_slug("not-a-market") is None
    assert analyze.classify_slug("") is None


# --- official PnL derivation (cumulative curve -> ALL/1M/1D deltas) ---------
def test_official_pnl_deltas(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_COMPETITORS_DIR", str(tmp_path))
    addr = "0x" + "2" * 40
    d = tmp_path / addr
    d.mkdir()
    day = 86400
    t0 = 100 * day
    series = [{"t": t0, "p": 10.0}, {"t": t0 + 5 * day, "p": 25.0},
              {"t": t0 + 40 * day, "p": 100.0}, {"t": t0 + 41 * day, "p": 120.0}]
    (d / "official_pnl.json").write_text(json.dumps({"series": series}), encoding="utf-8")
    off = analyze._official_pnl(addr)
    assert off["all_usd"] == 120.0            # last cumulative p
    assert off["1d_usd"] == 20.0              # 120 − p@(last−1d)=100
    assert off["1m_usd"] == 95.0              # 120 − p@(last−30d)=25


def test_official_pnl_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_COMPETITORS_DIR", str(tmp_path))
    addr = "0x" + "3" * 40
    (tmp_path / addr).mkdir()
    assert analyze._official_pnl(addr) is None


# --- uptime metric ---------------------------------------------------------
def test_uptime_metric():
    day = 86400
    ts = [1000, 1000 + 3600, 1000 + 2 * 3600, 1000 + day, 1000 + day + 3600]
    up = analyze._uptime(ts)
    assert up["max_gap_hours"] == round((day - 2 * 3600) / 3600.0, 2)   # the ~22h silence
    assert isinstance(up["continuous_coverage_frac"], float)
    assert up["active_hours_per_day_median"] == 3


# --- official + trades-only + rebate + decomposition + full series ---------
def _stage(tmp_path, addr, profile, acts, taker_lines, official_series):
    d = tmp_path / addr
    d.mkdir()
    (d / "activity.jsonl").write_text("\n".join(json.dumps(a) for a in acts), encoding="utf-8")
    (d / "taker_trades.jsonl").write_text("\n".join(json.dumps(t) for t in taker_lines), encoding="utf-8")
    (d / "positions.json").write_text("[]", encoding="utf-8")
    (d / "official_pnl.json").write_text(json.dumps({"series": official_series}), encoding="utf-8")
    return {"handle": "syn", "address": addr, "profile": profile}


def test_official_trades_only_rebate_decomposition(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_COMPETITORS_DIR", str(tmp_path))
    addr = "0x" + "4" * 40
    oe = 1783690500
    slug = f"btc-updown-5m-{oe}"
    cond = "0xc1"
    up = {"timestamp": oe + 10, "type": "TRADE", "conditionId": cond, "slug": slug, "asset": "up",
          "side": "BUY", "outcome": "Up", "size": 100, "usdcSize": 55, "price": 0.55, "transactionHash": "0xhu"}
    dn = {"timestamp": oe + 20, "type": "TRADE", "conditionId": cond, "slug": slug, "asset": "dn",
          "side": "BUY", "outcome": "Down", "size": 100, "usdcSize": 50, "price": 0.50, "transactionHash": "0xhd"}
    sol = {"timestamp": oe + 30, "type": "TRADE", "conditionId": "0xc2", "slug": "sol-updown-1h-1783688400",
           "asset": "s", "side": "BUY", "outcome": "Up", "size": 10, "usdcSize": 6, "price": 0.60, "transactionHash": "0xhs"}
    redeem = {"timestamp": oe + 400, "type": "REDEEM", "conditionId": cond, "slug": slug, "asset": "",
              "side": "", "outcome": "", "size": 100, "usdcSize": 100, "transactionHash": "0xhr"}
    taker = [
        {"transactionHash": "0xhu", "asset": "up", "side": "BUY", "size": 100, "price": 0.55, "timestamp": oe + 10},
        {"transactionHash": "0xhd", "asset": "dn", "side": "BUY", "size": 100, "price": 0.50, "timestamp": oe + 20},
    ]
    official = [{"t": oe - 5 * 86400, "p": 0.0}, {"t": oe + 500, "p": 40.0}]
    entry = _stage(tmp_path, addr, {"takerTierName": "Platinum"}, [up, dn, sol, redeem], taker, official)
    out = analyze.analyze_account(entry)
    pnl = out["pnl"]
    # OFFICIAL is authoritative
    assert pnl["official_all_usd"] == 40.0 and pnl["official_pnl_available"] is True
    # trades-only EXCLUDES the +100 redeem → negative; distinct from reconstruction
    assert pnl["trades_only_usd"] < 0 and pnl["realized_cashflow_usd"] > pnl["trades_only_usd"]
    assert abs(pnl["settlement_gap_usd"] - (40.0 - pnl["trades_only_usd"])) < 1e-6
    # matched-pair PnL NEGATIVE at pair-cost 1.05 (the takerner/frog crux)
    assert pnl["decomposition"]["matched_pair_pnl_usd"] < 0
    # rebate = Platinum 0.32 × est taker fees > 0 (shown separately, not summed)
    assert pnl["rebate"]["rebate_pct"] == 0.32 and pnl["est_taker_rebate_usd"] > 0
    # full-series breakdown includes SOL (no longer collapsed to "other")
    assert "SOL-1h" in out["coverage"]["full_series_volume"]
    assert "SOL-1h" in out["coverage"]["non_our_series"]


# --- report ranks by OFFICIAL PnL, not the reconstruction ------------------
def test_report_ranks_by_official():
    def acct(handle, addr, official, realized):
        return {
            "handle": handle, "address": addr, "profile": {"takerTierName": "None"},
            "totals": {"our_series_trades": 5000},
            "maker_taker": {"our_taker_frac_count": 0.1},
            "pair_discipline": {"two_sided_frac": 0.9, "pair_cost": {"median": 0.98}},
            "merge_hold": {"behavior": "merges (recycles capital)"},
            "coverage": {}, "size_cadence": {}, "span": {},
            "pnl": {"official_all_usd": official, "official_pnl_available": True,
                    "realized_cashflow_usd": realized, "pnl_per_active_day_usd": 1.0,
                    "decomposition": {}, "rebate": {}},
        }
    analysis = {"accounts": [
        acct("low_official", "0x" + "a" * 40, 100.0, 999999.0),    # big reconstruction, small official
        acct("high_official", "0x" + "b" * 40, 500000.0, -1.0),    # tiny reconstruction, big official
    ]}
    out = report.build_html(analysis, {"handles": []}, None)
    assert out.index("high_official") < out.index("low_official")   # ranked by OFFICIAL
    assert "How to read the labels" in out                          # provenance legend present
    assert "no rebate cushion" in out                               # OUR BOT rebate framing
