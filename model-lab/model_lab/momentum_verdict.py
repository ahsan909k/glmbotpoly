"""Stage — momentum_verdict: side-by-side full-history vs current-regime verdict.

Reads each ``backtest_momentum`` run's ``metrics.json`` (+ ``trades.csv`` for the calendar
day-span) and tabulates, per slice and per series: **net PnL, PnL/day, win rate, trades/day,
sample sizes (windows + trades)** — with the **effective fill latency** labelled and the
shuffled-control p-value / VOID verdict carried through. This is the honest read: the
full-history headline is optimistic (flat latency, mixes regimes); the verdict lives on the
current-regime (post-2026-06-05) slices at a realistic effective latency.

Slices (auto-detected — a missing dir is skipped):

===================  ================================================  ==============================
slice                dir                                               effective latency
===================  ================================================  ==============================
full history         backtests/momentum                                flat (mixes regimes — rosy)
current regime VPS   backtests/momentum_current_regime/vps255          net 5 + venue 250 = 255 ms
current regime home  backtests/momentum_current_regime/home450         net 200 + venue 250 = 450 ms
since-delay VPS      backtests/momentum_since_delay/vps255             255 ms (if a date was pinned)
since-delay home     backtests/momentum_since_delay/home450            450 ms (if a date was pinned)
===================  ================================================  ==============================

Run::  python -m model_lab.momentum_verdict
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from .config import Paths, resolve_paths, stage_parser

MS_PER_DAY = 86_400_000

# (key, human label, subdir under out/backtests, effective-latency note)
SLICES = [
    ("full_history", "Full history", "momentum", "flat (mixes regimes — optimistic)"),
    ("current_vps", "Current regime · VPS", "momentum_current_regime/vps255", "255 ms (net 5 + venue 250)"),
    ("current_home", "Current regime · home", "momentum_current_regime/home450", "450 ms (net 200 + venue 250)"),
    ("since_delay_vps", "Since-delay · VPS", "momentum_since_delay/vps255", "255 ms (net 5 + venue 250)"),
    ("since_delay_home", "Since-delay · home", "momentum_since_delay/home450", "450 ms (net 200 + venue 250)"),
]


def _eff_latency(params: dict) -> int | None:
    """Effective fill latency ms — the new decomposed key, else the legacy flat latency_ms."""
    v = params.get("effective_latency_ms", params.get("latency_ms"))
    return int(v) if v is not None else None


def _day_span(run_dir) -> tuple[int | None, int | None, int]:
    """(min_day, max_day, span_days) from the slice's trades.csv (0 span if no trades)."""
    tpath = run_dir / "trades.csv"
    if not tpath.exists():
        return None, None, 0
    try:
        df = pd.read_csv(tpath, usecols=["window_open_ms"])
    except (ValueError, pd.errors.EmptyDataError):
        return None, None, 0
    if df.empty:
        return None, None, 0
    days = (df["window_open_ms"].to_numpy(dtype="int64") // MS_PER_DAY)
    lo, hi = int(days.min()), int(days.max())
    return lo, hi, (hi - lo + 1)


def _slice_metrics(paths: Paths, subdir: str, label: str, latnote: str) -> dict | None:
    run_dir = paths.out_dir / "backtests" / subdir
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    m = json.loads(mpath.read_text(encoding="utf-8"))
    c = m.get("counts", {})
    ctrl = m.get("controls", {})
    params = m.get("params", {})
    lo, hi, span = _day_span(run_dir)
    net = float(c.get("net_pnl", 0.0))
    trades = int(c.get("trades", 0))
    windows = int(c.get("resolved_windows", 0))
    return {
        "key": None, "slice": label, "effective_latency": latnote,
        "effective_latency_ms": _eff_latency(params),
        "network_ms": params.get("network_ms"), "venue_delay_ms": params.get("venue_delay_ms"),
        "windows": windows, "trades": trades,
        "net_pnl": net, "fees": float(c.get("fees", 0.0)),
        "win_rate": float(c.get("win_rate", float("nan"))),
        "pnl_per_window": (net / windows) if windows else float("nan"),
        "pnl_per_trade": float(c.get("pnl_per_trade", float("nan"))),
        "span_days": span,
        "pnl_per_day": (net / span) if span else float("nan"),
        "trades_per_day": (trades / span) if span else float("nan"),
        "max_drawdown": float(c.get("max_drawdown", float("nan"))),
        "shuffled_p_value": ctrl.get("shuffled_p_value"),
        "real_beats_shuffled": ctrl.get("real_beats_shuffled"),
        "per_series": m.get("per_series", {}),
        "verdict": m.get("verdict", ""),
    }


def build_verdict(paths: Paths) -> dict:
    """Assemble the side-by-side comparison from whichever slices are present on disk."""
    slices = [s for s in (_slice_metrics(paths, sub, lab, note) for (_k, lab, sub, note) in SLICES)
              if s is not None]
    # per-series rows: one per (slice, series), PnL/day against the slice's calendar span.
    series_rows = []
    for s in slices:
        span = s["span_days"]
        for sk, ps in sorted(s["per_series"].items()):
            if not ps:
                continue
            snet = float(ps.get("net_pnl", 0.0))
            series_rows.append({
                "slice": s["slice"], "series": sk,
                "trades": int(ps.get("trades", 0)),
                "win_rate": float(ps.get("win_rate", float("nan"))),
                "net_pnl": snet, "pnl_per_day": (snet / span) if span else float("nan"),
                "max_drawdown": float(ps.get("drawdown", float("nan"))),
            })
    return {"slices": slices, "series_rows": series_rows,
            "note": ("Verdict lives on the current-regime slices (post-2026-06-05) at a realistic "
                     "effective latency; the full-history row is optimistic (flat latency, mixes "
                     "pre/post-delay regimes).")}


# --- rendering --------------------------------------------------------------
def _f(x, nd=2):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "—"
        return f"{x:,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x):
    return "—" if (x is None or (isinstance(x, float) and x != x)) else f"{x:.0%}"


def _verdict_word(s: dict) -> str:
    if s["real_beats_shuffled"] is None:
        return "n/a"
    if not s["real_beats_shuffled"]:
        return "VOID (≤ random)"
    return "beats random" + (f" (p={_f(s['shuffled_p_value'], 2)})" if s["shuffled_p_value"] is not None else "")


def _markdown(v: dict) -> str:
    lines = [
        "# Momentum verdict — full history vs current 250 ms-delay regime",
        "",
        "Side-by-side of the engine's momentum trigger (money_judge ports) replayed over the "
        "reconstructed historical book. **Read the current-regime rows for the go/no-go** — the "
        "full-history row blends in the Feb 18 → Jun 5 no-delay window and fills at a flat latency, "
        "so it is optimistic. Effective taker fill latency = network + 250 ms venue delay.",
        "",
        "## Headline (all series combined)",
        "",
        "| slice | eff. latency | windows | trades | net PnL $ | PnL/day $ | PnL/window $ | win rate | trades/day | max DD $ | control |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for s in v["slices"]:
        lines.append(
            f"| {s['slice']} | {s['effective_latency']} | {s['windows']:,} | {s['trades']:,} | "
            f"{_f(s['net_pnl'])} | {_f(s['pnl_per_day'])} | {_f(s['pnl_per_window'], 4)} | "
            f"{_pct(s['win_rate'])} | {_f(s['trades_per_day'], 1)} | {_f(s['max_drawdown'])} | "
            f"{_verdict_word(s)} |")
    lines += ["", "> " + v["note"], "", "## Per series", "",
              "| slice | series | trades | net PnL $ | PnL/day $ | win rate | max DD $ |",
              "|---|---|--:|--:|--:|--:|--:|"]
    for r in v["series_rows"]:
        lines.append(
            f"| {r['slice']} | {r['series']} | {r['trades']:,} | {_f(r['net_pnl'])} | "
            f"{_f(r['pnl_per_day'])} | {_pct(r['win_rate'])} | {_f(r['max_drawdown'])} |")
    lines += ["", "## Per-slice verdict text", ""]
    for s in v["slices"]:
        lines += [f"**{s['slice']}** — {s['verdict']}", ""]
    lines += ["## Sample-size note", "",
              "Sample size is stated as (resolved windows, trades) per slice above. The current-regime "
              "slice is a small window of history (~5 weeks post-2026-06-05) — treat its verdict as "
              "directional and confirm as more current-regime data accrues.", ""]
    return "\n".join(lines)


def _write_outputs(paths: Paths, v: dict) -> None:
    out_dir = paths.out_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "momentum_verdict.md").write_text(_markdown(v), encoding="utf-8")
    (out_dir / "momentum_verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "momentum_verdict")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    v = build_verdict(paths)
    if not v["slices"]:
        print("[momentum_verdict] no backtest metrics.json found under out/backtests/ — run "
              "backtest_momentum first")
        return 1
    _write_outputs(paths, v)
    for s in v["slices"]:
        print(f"[momentum_verdict] {s['slice']:24s} net ${_f(s['net_pnl'])} "
              f"({s['trades']:,} trades / {s['windows']:,} windows, {_pct(s['win_rate'])} win, "
              f"eff {s['effective_latency']}) — {_verdict_word(s)}")
    print(f"[momentum_verdict] wrote {paths.out_dir / 'backtests' / 'momentum_verdict.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
