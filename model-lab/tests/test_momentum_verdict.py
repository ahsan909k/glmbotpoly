"""Unit tests for momentum_verdict — side-by-side assembly from metrics.json + trades.csv."""

from __future__ import annotations

import json

import pandas as pd

from model_lab import momentum_verdict as mv
from model_lab.config import Paths

MS_PER_DAY = 86_400_000


def _write_slice(out, subdir, *, net, trades, windows, win_rate, p, beats, params, per_series, days):
    d = out / "backtests" / subdir
    d.mkdir(parents=True, exist_ok=True)
    metrics = {
        "params": params,
        "counts": {"net_pnl": net, "trades": trades, "resolved_windows": windows,
                   "win_rate": win_rate, "fees": 1.0,
                   "pnl_per_trade": (net / trades if trades else 0.0), "max_drawdown": 5.0},
        "controls": {"shuffled_p_value": p, "real_beats_shuffled": beats,
                     "shuffled_mean_net": 0.0, "never_trading_net": 0.0},
        "per_series": per_series,
        "verdict": f"verdict for {subdir}",
    }
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    base = 100 * MS_PER_DAY
    wom = [base + (i % days) * MS_PER_DAY for i in range(trades)]
    pd.DataFrame({"window_open_ms": wom, "series": ["BTC-5m"] * trades}).to_csv(
        d / "trades.csv", index=False)


def test_build_verdict_assembles_and_computes_rates(tmp_path):
    out = tmp_path / "out"
    paths = Paths(journal_dir=tmp_path / "j", depth_dir=tmp_path / "d", out_dir=out)
    _write_slice(out, "momentum", net=1000.0, trades=200, windows=5000, win_rate=0.63, p=0.0, beats=True,
                 params={"latency_ms": 200},  # legacy (no decomposition) — full-history slice
                 per_series={"BTC-5m": {"trades": 200, "win_rate": 0.63, "net_pnl": 1000.0, "drawdown": 5.0}},
                 days=10)
    _write_slice(out, "momentum_current_regime/vps255", net=100.0, trades=40, windows=1000, win_rate=0.58,
                 p=0.2, beats=False,
                 params={"latency_ms": 255, "network_ms": 5, "venue_delay_ms": 250, "effective_latency_ms": 255},
                 per_series={"BTC-5m": {"trades": 40, "win_rate": 0.58, "net_pnl": 100.0, "drawdown": 3.0}},
                 days=5)
    v = mv.build_verdict(paths)

    assert len(v["slices"]) == 2
    full = next(s for s in v["slices"] if s["slice"] == "Full history")
    assert full["effective_latency_ms"] == 200  # legacy latency_ms fallback
    assert abs(full["pnl_per_day"] - 100.0) < 1e-9  # 1000 / 10 days
    assert abs(full["trades_per_day"] - 20.0) < 1e-9

    vps = next(s for s in v["slices"] if "VPS" in s["slice"])
    assert vps["effective_latency_ms"] == 255 and vps["venue_delay_ms"] == 250
    assert abs(vps["pnl_per_day"] - 20.0) < 1e-9  # 100 / 5 days

    assert mv._verdict_word(full).startswith("beats random")
    assert mv._verdict_word(vps).startswith("VOID")

    assert any(r["series"] == "BTC-5m" and r["slice"] == "Full history" for r in v["series_rows"])

    mv._write_outputs(paths, v)
    md = (out / "backtests" / "momentum_verdict.md").read_text(encoding="utf-8")
    assert "Full history" in md and "Current regime · VPS" in md
    assert (out / "backtests" / "momentum_verdict.json").exists()


def test_build_verdict_empty_when_no_runs(tmp_path):
    paths = Paths(journal_dir=tmp_path / "j", depth_dir=tmp_path / "d", out_dir=tmp_path / "out")
    v = mv.build_verdict(paths)
    assert v["slices"] == [] and v["series_rows"] == []
