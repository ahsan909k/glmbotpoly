"""Stage — detect_delay: pin the 250 ms taker-delay introduction date from the tape.

**Physical basis.** The venue's taker delay imposes a *floor* on how fast an aggressive
Polymarket print can clear after a fast-feed (Binance) move: a taker reacting instantly
still cannot print until the delay elapses. So the LOW QUANTILE of the "Binance move →
first aggressive PM print" lag distribution should **step up by ~250 ms** on the day the
delay was introduced (and the pre-2026-02-18 500 ms regime shows a ~500 ms floor, which
drops when that delay is removed — a built-in validation of the method).

Per (ISO-week, series) across the span we measure that lag and report p5/p10/p25/p50 +
count, then scan the weekly **p10 floor** for a clean, sustained step to ~250 ms:

* **Clean step** → report the date + evidence and write ``detected_date`` to
  ``metrics.json``. ``backtest_momentum.regime_boundaries`` folds it into the per-month
  annotation, and a secondary "since-delay" verdict run can use it.
* **No clean step** → ``detected_date = null``; **Jun 5 stays the boundary**.

The print aggressor ``side`` semantics are **auto-calibrated** from the data (the side whose
prints follow positive Binance moves is the up-aggressor) and the agreement is reported, so
we never hardcode a string convention.

Read-only over ``data/aggtrades`` (Binance fast feed) + ``data/telonex`` PM **trades**;
reuses the ``historical_common`` / ``io.telonex_pm`` readers. Streamed week-by-week (bounded
memory). Output ``out/delay_detection/{metrics.json, per_week_series.csv, report.html}``.

Run::

    python -m model_lab.detect_delay                      # defaults 2026-02-01 → 2026-06-15
    python -m model_lab.detect_delay --since 2026-02-01 --until 2026-06-15 --series BTC-15m
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import historical_common as hc
from . import historical_labels as hlbl
from .config import Paths, parse_time_bound, resolve_bounds, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .lib import procmem

# --- move detection + matching knobs ---------------------------------------
MOVE_RET_WINDOW_MS = 1000  # Binance return window that defines a "move"
MOVE_SIGMA_MULT = 3.0      # |ret| must exceed this × the window's return σ
MOVE_REFRACTORY_MS = 2000  # min gap between accepted moves (dedupe a burst)
REACT_HORIZON_MS = 5000    # only count a reacting print within this of the move
SIDE_CALIB_LOOKBACK_MS = 1000  # Binance return before a print, to calibrate its side

# --- step detection knobs ---------------------------------------------------
FLOOR_Q = 10               # the floor percentile (p10) — the delay signature
STEP_LOW_MS = 150.0        # pre-step floor must sit below this
STEP_HIGH_MS = 200.0       # post-step floor must sit above this (and stay)
STEP_SUSTAIN_WEEKS = 2     # ≥ this many consecutive post-step weeks above STEP_HIGH_MS
MIN_LAGS_PER_WEEK = 30     # a week needs this many matched lags to be trusted

DEFAULT_SINCE = "2026-02-01"
DEFAULT_UNTIL = "2026-06-15"


# --- core (pure, unit-tested) ----------------------------------------------
def detect_moves(ts: np.ndarray, px: np.ndarray, *, window_ms: int = MOVE_RET_WINDOW_MS,
                 sigma_mult: float = MOVE_SIGMA_MULT,
                 refractory_ms: int = MOVE_REFRACTORY_MS) -> tuple[np.ndarray, np.ndarray]:
    """Detect fast-feed moves. Returns ``(move_ts, move_sign)``: timestamps where the
    ``window_ms`` log-return exceeds ``sigma_mult`` × its σ, deduped by ``refractory_ms``."""
    ts = np.asarray(ts, dtype="int64")
    px = np.asarray(px, dtype=float)
    n = ts.size
    if n < 10 or not np.all(px > 0):
        return np.empty(0, "int64"), np.empty(0, int)
    lp = np.log(px)
    j = np.searchsorted(ts, ts - window_ms, side="right") - 1  # last tick ≤ ts-window
    valid = j >= 0
    ret = np.zeros(n)
    ret[valid] = lp[valid] - lp[j[valid]]
    if valid.sum() < 10:
        return np.empty(0, "int64"), np.empty(0, int)
    sd = float(np.std(ret[valid]))
    if not (sd > 0.0):
        return np.empty(0, "int64"), np.empty(0, int)
    cand = np.nonzero(valid & (np.abs(ret) > sigma_mult * sd))[0]
    move_ts: list[int] = []
    move_sign: list[int] = []
    last = -(10 ** 18)
    for i in cand:
        if ts[i] - last >= refractory_ms:
            move_ts.append(int(ts[i]))
            move_sign.append(int(np.sign(ret[i])))
            last = int(ts[i])
    return np.array(move_ts, "int64"), np.array(move_sign, int)


def calibrate_side(prints_ts: np.ndarray, prints_side: np.ndarray, bts: np.ndarray,
                   bpx: np.ndarray, *, lookback_ms: int = SIDE_CALIB_LOOKBACK_MS) -> dict:
    """Auto-map each aggressor ``side`` string → ±1 by the sign of the Binance move in the
    ``lookback_ms`` BEFORE its prints (up-aggressor follows up-moves). Sides with too few
    samples are omitted (``.get(s, 0)`` ⇒ matches no move)."""
    bts = np.asarray(bts, dtype="int64")
    bpx = np.asarray(bpx, dtype=float)
    pts = np.asarray(prints_ts, dtype="int64")
    if bts.size < 2 or pts.size == 0 or not np.all(bpx > 0):
        return {}
    lp = np.log(bpx)
    i1 = np.searchsorted(bts, pts, side="right") - 1
    i0 = np.searchsorted(bts, pts - lookback_ms, side="right") - 1
    ok = (i1 >= 0) & (i0 >= 0) & (i1 > i0)
    ret = np.full(pts.size, np.nan)
    ret[ok] = lp[i1[ok]] - lp[i0[ok]]
    mapping: dict = {}
    psd = np.asarray(prints_side)
    for s in np.unique(psd):
        m = (psd == s) & ok
        if int(m.sum()) >= 5:
            mean_r = float(np.nanmean(ret[m]))
            mapping[s] = 1 if mean_r >= 0.0 else -1
    return mapping


def match_lags(move_ts: np.ndarray, move_sign: np.ndarray, print_ts: np.ndarray,
               print_sign: np.ndarray, *, horizon_ms: int = REACT_HORIZON_MS) -> list[int]:
    """For each move, the lag to the FIRST subsequent print of the same sign within
    ``horizon_ms``. ``print_ts`` must be sorted ascending."""
    move_ts = np.asarray(move_ts, dtype="int64")
    move_sign = np.asarray(move_sign, dtype=int)
    print_ts = np.asarray(print_ts, dtype="int64")
    print_sign = np.asarray(print_sign, dtype=int)
    lags: list[int] = []
    for sign in (1, -1):
        st = print_ts[print_sign == sign]
        if st.size == 0:
            continue
        mm = move_ts[move_sign == sign]
        if mm.size == 0:
            continue
        pos = np.searchsorted(st, mm, side="right")  # first print strictly after each move
        for k in range(mm.size):
            p = pos[k]
            if p < st.size:
                lag = int(st[p] - mm[k])
                if 0 < lag <= horizon_ms:
                    lags.append(lag)
    return lags


def week_start(d: date) -> str:
    """ISO date (str) of the Monday of ``d``'s week — the weekly bucket key."""
    return (d - timedelta(days=d.weekday())).isoformat()


def detect_step_for_series(recs: list[dict], *, low: float = STEP_LOW_MS, high: float = STEP_HIGH_MS,
                           sustain: int = STEP_SUSTAIN_WEEKS, min_lags: int = MIN_LAGS_PER_WEEK):
    """Scan a series' weekly floors for a clean low→high step to ~250 ms. Returns
    ``(detected_week_start | None, trusted_recs)``. A step at week *i* needs the trailing
    weeks' median floor < ``low`` and ``sustain`` consecutive weeks from *i* all > ``high``."""
    trusted = [r for r in sorted(recs, key=lambda r: r["week_start"]) if r["n_lags"] >= min_lags]
    floors = [float(r["p10"]) for r in trusted]
    for i in range(1, len(trusted)):
        prev = floors[max(0, i - 3):i]
        post = floors[i:i + sustain]
        if len(post) >= sustain and prev and float(np.median(prev)) < low and all(f > high for f in post):
            return trusted[i]["week_start"], trusted
    return None, trusted


# --- data plumbing ----------------------------------------------------------
def _binance_mid_series(paths: Paths, symbol: str, d: date) -> tuple[np.ndarray, np.ndarray]:
    """A fine Binance mid series (ts_ms, price) for UTC day ``d`` — Telonex ``book_snapshot_25``
    top-mid (~100 ms) where available, else aggTrades 1-second bars."""
    top = hc.telonex_top_mid(paths, symbol, d)
    if not top.empty:
        ts = top["ts_ms"].to_numpy(dtype="int64")
        px = top["mid"].to_numpy(dtype=float)
    else:
        bars = hc.binance_bars_for_day(paths, symbol, d)
        if bars.empty:
            return np.empty(0, "int64"), np.empty(0, float)
        ts = bars["sec"].to_numpy(dtype="int64") * 1000
        px = bars["price"].to_numpy(dtype=float)
    order = np.argsort(ts, kind="stable")
    return ts[order], px[order]


def _week_lag_stats(lags: list[int], n_moves: int, series_key: str, wk: str) -> dict:
    arr = np.asarray(lags, dtype=float)
    def q(p):
        return float(np.percentile(arr, p)) if arr.size else float("nan")
    return {"series": series_key, "week_start": wk, "n_moves": int(n_moves), "n_lags": int(arr.size),
            "p5": q(5), "p10": q(FLOOR_Q), "p25": q(25), "p50": q(50)}


# --- worker -----------------------------------------------------------------
def detect_delay(paths: Paths, *, series: tuple[str, ...] | None = None,
                 since_ms: int | None = None, until_ms: int | None = None) -> dict:
    """Measure the weekly Binance-move → first-aggressive-PM-print lag floor per series and
    detect the 250 ms-delay introduction step. Returns the metrics dict and writes the report."""
    paths.ensure_out()
    out_dir = paths.out_dir / "delay_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)

    rows: list[dict] = []
    side_counts: dict = defaultdict(int)  # observed side → sign votes (for validation)
    for series_key in series_keys:
        prefix = hc.series_prefix(series_key)
        if prefix is None:
            continue
        asset = hc.SLUG_PREFIXES[prefix][1]
        symbol = hc.SYMBOL_BY_ASSET[asset]
        by_week: dict[str, list[date]] = defaultdict(list)
        for d in hlbl._days_for_series(paths, series_key, since_ms, until_ms):
            by_week[week_start(d)].append(d)

        for wk, wk_days in sorted(by_week.items()):
            week_lags: list[int] = []
            week_moves = 0
            for d in wk_days:
                day_str = d.isoformat()
                bts, bpx = _binance_mid_series(paths, symbol, d)
                if bts.size < 10:
                    continue
                mts, msg = detect_moves(bts, bpx)
                if mts.size == 0:
                    continue
                wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
                if not wins:
                    continue
                for w in wins:
                    o, c = int(w["window_open_ms"]), int(w["window_close_ms"])
                    in_win = (mts >= o) & (mts <= c)
                    if not in_win.any():
                        continue
                    pt = tpm.read_pm_trades(paths.telonex_dir, w["slug"], "Up", day_str)
                    if pt.empty:
                        continue
                    p_ts = pt["ts_ms"].to_numpy(dtype="int64")
                    p_sd = pt["side"].to_numpy()
                    smap = calibrate_side(p_ts, p_sd, bts, bpx)
                    for s, sg in smap.items():
                        side_counts[(str(s), int(sg))] += 1
                    p_sign = np.array([smap.get(s, 0) for s in p_sd], dtype=int)
                    lags = match_lags(mts[in_win], msg[in_win], p_ts, p_sign)
                    week_lags.extend(lags)
                    week_moves += int(in_win.sum())
            rows.append(_week_lag_stats(week_lags, week_moves, series_key, wk))

    per_series_step: dict = {}
    detected_candidates: list[str] = []
    for series_key in series_keys:
        recs = [r for r in rows if r["series"] == series_key]
        step, trusted = detect_step_for_series(recs)
        per_series_step[series_key] = {
            "detected_week": step,
            "n_trusted_weeks": len(trusted),
            "pre_step_floor_p10": _pre_post(trusted, step, before=True),
            "post_step_floor_p10": _pre_post(trusted, step, before=False),
        }
        if step:
            detected_candidates.append(step)

    detected_date = min(detected_candidates) if detected_candidates else None
    side_map = _summarize_side_map(side_counts)
    result = {
        "params": {"title": "250 ms taker-delay detection", "series": list(series_keys),
                   "move_ret_window_ms": MOVE_RET_WINDOW_MS, "move_sigma_mult": MOVE_SIGMA_MULT,
                   "react_horizon_ms": REACT_HORIZON_MS, "floor_pctile": FLOOR_Q,
                   "step_low_ms": STEP_LOW_MS, "step_high_ms": STEP_HIGH_MS,
                   "step_sustain_weeks": STEP_SUSTAIN_WEEKS, "min_lags_per_week": MIN_LAGS_PER_WEEK},
        "detected_date": detected_date,
        "per_series_step": per_series_step,
        "side_calibration": side_map,
        "per_week_series": rows,
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    result["verdict"] = _verdict(detected_date, per_series_step, rows)
    _write_outputs(out_dir, result)
    return result


def _pre_post(trusted: list[dict], step: str | None, *, before: bool) -> float:
    if not step:
        return float("nan")
    vals = [float(r["p10"]) for r in trusted if (r["week_start"] < step) == before]
    return float(np.median(vals)) if vals else float("nan")


def _summarize_side_map(side_counts: dict) -> dict:
    """Aggregate the per-window side→sign votes into a stable map + agreement fraction."""
    by_side: dict[str, dict[int, int]] = defaultdict(lambda: {1: 0, -1: 0})
    for (s, sg), n in side_counts.items():
        by_side[s][sg] += n
    out = {}
    for s, votes in by_side.items():
        tot = votes[1] + votes[-1]
        sign = 1 if votes[1] >= votes[-1] else -1
        out[s] = {"sign": sign, "agreement": (max(votes[1], votes[-1]) / tot) if tot else float("nan"),
                  "n": tot}
    return out


def _verdict(detected_date, per_series_step, rows) -> str:
    if not rows:
        return ("No windows/prints matched over this span — inconclusive (check the aggTrades / "
                "Telonex PM-trade coverage and the date range).")
    if detected_date:
        hits = [f"{s} @ {v['detected_week']} "
                f"(floor {v['pre_step_floor_p10']:.0f}→{v['post_step_floor_p10']:.0f} ms)"
                for s, v in per_series_step.items() if v["detected_week"]]
        return (f"A clean ~250 ms floor step was detected. Earliest introduction date ≈ "
                f"{detected_date} (week start). Per series: {'; '.join(hits)}. The pre-step floor "
                f"is fast (arbers print within ~tens–low-hundreds of ms) and the post-step floor "
                f"sits at ~250 ms — the taker-delay signature. Re-run the verdict slice with "
                f"--since {detected_date} as a larger secondary sample.")
    return ("NO clean ~250 ms floor step was found in the weekly p10 lag (either the signature is "
            "too noisy on this coverage, or the introduction was gradual). Jun 5 stays the "
            "boundary — read the current-regime verdict off --since 2026-06-05.")


# --- report -----------------------------------------------------------------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _floor_chart(rows: list[dict], series_keys, detected_date) -> str | None:
    if not rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for sk in series_keys:
        recs = sorted((r for r in rows if r["series"] == sk and r["n_lags"] >= MIN_LAGS_PER_WEEK),
                      key=lambda r: r["week_start"])
        if not recs:
            continue
        xs = [np.datetime64(r["week_start"]) for r in recs]
        ax.plot(xs, [r["p10"] for r in recs], marker="o", markersize=3, linewidth=1.0, label=sk)
    ax.axhline(250, color="#d62728", linestyle=":", linewidth=1.0, label="250 ms")
    for ref, lab, col in [("2026-02-18", "500ms off", "#888"), ("2026-06-05", "250ms lock", "#555")]:
        ax.axvline(np.datetime64(ref), color=col, linestyle="--", linewidth=0.9)
    if detected_date:
        ax.axvline(np.datetime64(detected_date), color="#2ca02c", linewidth=1.4, label="detected")
    ax.set_ylabel(f"p{FLOOR_Q} move→print lag (ms)")
    ax.set_title("Weekly aggressive-print lag FLOOR per series (delay signature)")
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()
    return _png(fig)


def _write_outputs(out_dir, result: dict) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    pd.DataFrame(result["per_week_series"],
                 columns=["series", "week_start", "n_moves", "n_lags", "p5", "p10", "p25", "p50"]
                 ).to_csv(out_dir / "per_week_series.csv", index=False)
    chart = _floor_chart(result["per_week_series"], result["params"]["series"], result["detected_date"])
    step_rows = [{"series": s, "detected_week": v["detected_week"],
                  "pre_floor_ms": v["pre_step_floor_p10"], "post_floor_ms": v["post_step_floor_p10"],
                  "trusted_weeks": v["n_trusted_weeks"]}
                 for s, v in result["per_series_step"].items()]
    side_rows = [{"side": s, "maps_to": ("up-aggressor" if v["sign"] == 1 else "down-aggressor"),
                  "agreement": v["agreement"], "n": v["n"]}
                 for s, v in result["side_calibration"].items()]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 980px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.9rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #2ca02c; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<p class="muted">The 250 ms taker delay floors how fast an aggressive PM print can follow a
Binance move. A clean step in the weekly p{FLOOR_Q} lag floor pins its introduction date.</p>
<div class="verdict">{result['verdict']}</div>
{eh._img_tag(chart, 'weekly lag floor per series')}
<h2>Per-series step</h2>
{eh._table_html(step_rows, [("series", "series"), ("detected_week", "detected week"),
    ("pre_floor_ms", "pre floor ms"), ("post_floor_ms", "post floor ms"), ("trusted_weeks", "weeks")])}
<h2>Aggressor-side calibration (validation)</h2>
<p class="muted">Auto-mapped from data (no hardcoded convention): the side whose prints follow
positive Binance moves is the up-aggressor. High agreement ⇒ the side field is trustworthy.</p>
{eh._table_html(side_rows, [("side", "side"), ("maps_to", "maps to"),
    ("agreement", "agreement"), ("n", "n windows")])}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "detect_delay")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    if since_ms is None:
        since_ms = parse_time_bound(DEFAULT_SINCE)
    if until_ms is None:
        until_ms = parse_time_bound(DEFAULT_UNTIL)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    print(f"[detect_delay] scan {DEFAULT_SINCE if since_ms is None else since_ms} .. "
          f"telonex={paths.telonex_dir}")
    result = detect_delay(paths, series=series, since_ms=since_ms, until_ms=until_ms)
    dd = result["detected_date"]
    print(f"[detect_delay] detected_date = {dd or 'NONE (no clean step; Jun 5 stands)'}")
    for s, v in result["per_series_step"].items():
        print(f"[detect_delay]   {s}: step={v['detected_week']} "
              f"floor {v['pre_step_floor_p10']:.0f}→{v['post_step_floor_p10']:.0f} ms "
              f"({v['n_trusted_weeks']} trusted weeks)")
    procmem.report_peak_rss("detect_delay", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[detect_delay] wrote {paths.out_dir / 'delay_detection'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
