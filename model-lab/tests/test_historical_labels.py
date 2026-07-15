"""Tests for the historical window-resolution stage (Part 2), official-resolution primary.

- official resolution labels ~100% of windows present (the coverage fix);
- windows with no official resolution are counted, never guessed;
- the two proxies (quote-convergence, Binance) are graded against official truth;
- the quote-convergence resolver unit (still used as a graded cross-check).
"""

from __future__ import annotations

import pandas as pd

from model_lab import fixtures as fx
from model_lab import historical_common as hc
from model_lab import historical_labels as hl
from model_lab.config import Paths


def _paths(tmp_path) -> Paths:
    return Paths(journal_dir=tmp_path / "journal", depth_dir=tmp_path / "depth",
                out_dir=tmp_path / "out", hist_dir=tmp_path / "aggtrades",
                telonex_dir=tmp_path / "telonex")


def _setup(tmp_path):
    res = fx.make_historical_fixture(tmp_path / "telonex", tmp_path / "aggtrades")
    fx.write_historical_coverage(tmp_path / "out", res["coverage"])
    return _paths(tmp_path), res


# --- resolver unit (the graded proxy) ---------------------------------------
def _quotes(mids, open_ms, close_ms):
    import numpy as np
    n_secs = (close_ms - open_ms) // 1000
    xs = np.linspace(0, n_secs, len(mids))
    rows = []
    for k in range(n_secs + 1):
        m = float(np.interp(k, xs, mids))
        ts = open_ms + k * 1000
        rows.append({"token_id": "T", "ts_ms": ts, "bid_px": max(0.001, m - 0.005),
                     "bid_sz": 100.0, "ask_px": min(0.999, m + 0.005), "ask_sz": 100.0})
    return pd.DataFrame(rows)


def test_resolver_converges_and_excludes():
    o, c = 1_000_000, 1_300_000
    up = _quotes([0.5, 0.7, 0.9, 0.99], o, c)
    dn = _quotes([0.5, 0.3, 0.1, 0.01], o, c)
    assert hc.resolve_by_quote_convergence(up, dn, o, c)["outcome"] == "Up"
    assert hc.resolve_by_quote_convergence(dn, up, o, c)["outcome"] == "Down"
    flat_u = _quotes([0.5, 0.52, 0.49, 0.5], o, c)
    flat_d = _quotes([0.5, 0.48, 0.51, 0.5], o, c)
    assert hc.resolve_by_quote_convergence(flat_u, flat_d, o, c)["outcome"] is None


# --- official-primary coverage ----------------------------------------------
def test_official_primary_coverage(tmp_path):
    paths, _ = _setup(tmp_path)
    r = hl.historical_labels(paths, series=("BTC-5m", "ETH-5m"))
    c = r["counts"]
    # catalog universe = 6 windows × 2 series = 12; all officially resolved.
    assert c["windows_seen"] == 12
    assert c["resolved_official"] == 12
    assert c["coverage_frac"] == 1.0
    assert c["unlabeled_no_official"] == 0
    lab = pd.read_parquet(paths.out_dir / "historical_labels" / "labels.parquet")
    assert len(lab) == 12 and (lab["source"] == "official").all()
    assert lab["outcome"].isin(["Up", "Down"]).all()
    assert lab["condition_id"].notna().all()
    for f in ("labels.parquet", "metrics.json", "report.html"):
        assert (paths.out_dir / "historical_labels" / f).exists()


def test_no_official_is_counted_not_guessed(tmp_path):
    paths, res = _setup(tmp_path)
    # drop one window from the official map → it must be counted unlabeled, never proxy-filled.
    full = res["res_map"]
    drop = next(iter(full))
    partial = {k: v for k, v in full.items() if k != drop}
    r = hl.historical_labels(paths, series=("BTC-5m", "ETH-5m"), res_map=partial, grade=False)
    assert r["counts"]["unlabeled_no_official"] >= 1
    assert r["counts"]["resolved_official"] < r["counts"]["windows_seen"]
    lab = pd.read_parquet(paths.out_dir / "historical_labels" / "labels.parquet")
    assert (drop[0], drop[1]) not in set(zip(lab["series"], lab["window_open_ms"]))


def test_proxy_grading(tmp_path):
    paths, _ = _setup(tmp_path)
    r = hl.historical_labels(paths, series=("BTC-5m", "ETH-5m"))
    g = r["proxy_grading"]
    assert g["sampled"] > 0
    for proxy in ("quote_convergence", "binance_anchored"):
        assert "overall" in g[proxy] and "knife_edge" in g[proxy]
        assert 0.0 <= g[proxy]["overall"]["error_rate"] <= 1.0
    # the fixture's proxies agree with official on the resolvable windows → low error
    assert g["binance_anchored"]["overall"]["error_rate"] == 0.0
