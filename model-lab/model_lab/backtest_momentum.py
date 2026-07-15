"""Stage — backtest_momentum (Part 3).

Replay the live engine's hand-coded **momentum trigger** over the full historical span
(all four series, day by day) and answer "does it make money?", with honest controls.

The trigger is *not* re-derived: it reuses ``money_judge``'s already-verified ports of
``crates/engine/src/taker/{signal,edge,driver}.rs`` — the confirmed-move detector
(``|ln(S_new/S_old)| > 3σ√T`` over a 2 s Binance-mid ring), the per-share fee-aware edge
gate (``(fair − p) > rate·p·(1−p) + buffer``), and the full gate ladder (Ready / usable
fair+vol / τ>0 / cooldown 5 s / per-window $10 budget / book). The same conservative
taker-fill rules as ``money_judge`` (fill only at the displayed best quote up to its
displayed size; the exact 5-dp taker fee; the Down-side CLOB mirror; skip on a stale book;
$1 min notional; mark to the window's resolution).

Historical inputs are reconstructed (there are no journaled model snapshots pre-journal):
p_up / σ from the Binance mid (Telonex ``book_snapshot_25`` top-mid where available, else
aggTrades) against the Binance-anchored strike; the momentum ring from the same fine mid
(100 ms); the PM book from Telonex quotes; the outcome from the **official** Polymarket
resolution (:mod:`model_lab.historical_resolutions`).

**Controls (the real trigger must beat them, else the result is VOID):**
(a) **never-trade** = $0; (b) a **shuffled / random-trigger** control with *matched fire
frequency* — fire at random eligible snapshots at the real confirmed-move rate, everything
else identical, over N seeds; (c) the **majority** (always-common-outcome) baseline.

Reports PnL after fees / win rate / trade count / drawdown **per series per month** and
**by time-of-day**, with sample sizes stated. Output
``out/backtests/momentum/{report.html, metrics.json, trades.csv, per_series_month.csv,
per_hour.csv}``. Stop after the report.

Run::

    python -m model_lab.backtest_momentum --since 2026-05-01 --until 2026-06-01
    python -m model_lab.backtest_momentum --series BTC-5m,ETH-5m --seeds 20
"""

from __future__ import annotations

import base64
import io
import json
import sys
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import historical_common as hc
from . import historical_labels as hlbl
from . import historical_resolutions as hr
from . import money_judge as mj
from . import dataset as ds
from .config import Paths, resolve_bounds, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .lib import math as lm
from .lib import procmem

DEFAULT_SEEDS = 10
DEFAULT_SEED = 20260705
MS_PER_DAY = 86_400_000

# Verified Polymarket venue-regime boundaries (out/audits/venue_regime_timeline.md). The
# 250 ms-delay INTRODUCTION date is unverified in the docs; detect_delay.py pins it from the
# tape and writes `detected_date` to out/delay_detection/metrics.json, which
# regime_boundaries() folds in as an extra "detected" marker.
REGIME_BOUNDARIES = [
    {"date": "2026-01-07", "label": "Dynamic taker fees (15m crypto)", "short": "fees", "status": "verified"},
    {"date": "2026-02-18", "label": "500 ms taker delay removed", "short": "500ms off", "status": "verified"},
    {"date": "2026-03-30", "label": "Fee categories expanded", "short": "fee cats", "status": "verified"},
    {"date": "2026-06-05", "label": "250 ms delay locked (non-cancellable)", "short": "250ms lock", "status": "verified"},
    {"date": "2026-07-07", "label": "GTD min-expiry 1→3 min", "short": "GTD 3m", "status": "verified"},
]


def regime_boundaries(paths: Paths) -> list[dict]:
    """The verified regime boundaries plus the detected 250 ms-delay introduction date, if
    detect_delay.py wrote one to out/delay_detection/metrics.json (``detected_date``).
    Sorted by date; the detected marker is styled distinctly in the chart/table."""
    out = [dict(b) for b in REGIME_BOUNDARIES]
    det = paths.out_dir / "delay_detection" / "metrics.json"
    try:
        date = json.loads(det.read_text(encoding="utf-8")).get("detected_date")
    except (OSError, json.JSONDecodeError):
        date = None
    if date:
        out.append({"date": str(date), "label": "250 ms taker delay introduced (detected)",
                    "short": "250ms in", "status": "detected"})
    return sorted(out, key=lambda b: b["date"])


# --- reconstruction ---------------------------------------------------------
def _day_grid_and_ring(paths: Paths, symbol: str, asset: str, d) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The asset's 1-second feature grid (mid, σ_1s) and the fine momentum-ring mids for
    UTC day ``d``. Mid source = Telonex ``book_snapshot_25`` top-mid (100 ms, the live
    ``BinanceDirect``/``Mid`` analogue) where available (overlap days too), else aggTrades
    1 s bars (pre-Telonex-book 15m tail; a coarser ring)."""
    top = hc.telonex_top_mid(paths, symbol, d)
    if not top.empty:
        bars = (top.assign(sec=top["ts_ms"].to_numpy(dtype="int64") // 1000)
                .groupby("sec", as_index=False)["mid"].last().rename(columns={"mid": "price"}))
        # 100 ms-bucketed ring (last mid per bucket), like money_judge's SIGNAL_BUCKET_MS.
        bkt = top["ts_ms"].to_numpy(dtype="int64") // mj.SIGNAL_BUCKET_MS
        buck = pd.DataFrame({"b": bkt, "mid": top["mid"].to_numpy(dtype=float)}).groupby("b", as_index=False).last()
        ring_ts = (buck["b"].to_numpy(dtype="int64") * mj.SIGNAL_BUCKET_MS)
        ring_px = buck["mid"].to_numpy(dtype=float)
    else:
        bars = hc.binance_bars_for_day(paths, symbol, d)
        if bars.empty:
            return pd.DataFrame(), np.empty(0, dtype="int64"), np.empty(0, dtype=float)
        ring_ts = bars["sec"].to_numpy(dtype="int64") * 1000
        ring_px = bars["price"].to_numpy(dtype=float)
    grid = ds.build_feature_grid(ds._binance_ticks_from_bars(bars, asset),
                                 pd.DataFrame(columns=ds.DEPTH_COLS), asset)
    return grid, ring_ts, ring_px


def _window_model_rows(grid: pd.DataFrame, window: dict, strike: dict) -> list[dict]:
    """1-second reconstructed model snapshots for one window as ``_run_momentum`` dicts."""
    w = {**window, **strike}
    snaps = hc.reconstruct_window_snapshots(grid, w)
    if snaps.empty:
        return []
    series, open_ms, asset = window["series"], window["window_open_ms"], window["asset"]
    return [{"series": series, "open_time": int(open_ms), "asset": asset,
             "ts": int(r.ts), "p_up": float(r.p_up), "sigma_1s": float(r.sigma_1s),
             "health": "Ready"} for r in snaps.itertuples(index=False)]


# --- matched-frequency shuffled control -------------------------------------
def _eligible_and_confirmed(model_rows: list[dict], mids_by_asset: dict, win_by_key: dict) -> tuple[dict, dict]:
    """Per asset: how many snapshots pass the pre-gates (eligible) and how many the
    confirmed-move detector fires on (confirmed) — the real trigger's fire rate, used to
    calibrate the random control's fire probability."""
    elig: dict[str, int] = defaultdict(int)
    conf: dict[str, int] = defaultdict(int)
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for m in model_rows:
        by_win[(m["series"], m["open_time"])].append(m)
    for key, snaps in by_win.items():
        win = win_by_key.get(key)
        if win is None:
            continue
        close_ms = win["close_time"]
        for m in sorted(snaps, key=lambda s: s["ts"]):
            p_up, sigma, now, asset = m["p_up"], m["sigma_1s"], m["ts"], m["asset"]
            if m["health"] != "Ready" or not (0.0 < p_up < 1.0) or not (sigma > 0.0):
                continue
            if (close_ms - now) / 1000.0 <= 0.0:
                continue
            elig[asset] += 1
            if mj._confirmed_direction(mids_by_asset.get(asset), now, sigma) is not None:
                conf[asset] += 1
    return elig, conf


def _run_random_control(model_rows: list[dict], win_by_key: dict, books: dict, *,
                        fire_prob: dict, rng: np.random.Generator, latency_ms: int,
                        fee_rate: float, window_budget: float) -> list[dict]:
    """The random-trigger control: fire at random eligible snapshots at the per-asset
    ``fire_prob`` (matched to the real confirmed-move rate), side = the model's leaned
    direction, everything else identical to ``_run_momentum`` (cooldown, budget, book,
    edge gate, mark-to-resolution). Reuses money_judge's fill primitives verbatim."""
    rows: list[dict] = []
    by_win: dict[tuple, list[dict]] = defaultdict(list)
    for m in model_rows:
        by_win[(m["series"], m["open_time"])].append(m)
    for key, snaps in by_win.items():
        win = win_by_key.get(key)
        if win is None:
            continue
        up_token, close_ms, outcome_up = win["up_token"], win["close_time"], win["outcome_up"]
        budget_left, last_take = window_budget, None
        for m in sorted(snaps, key=lambda s: s["ts"]):
            now, p_up, sigma, asset = m["ts"], m["p_up"], m["sigma_1s"], m["asset"]
            if m["health"] != "Ready" or not (0.0 < p_up < 1.0) or not (sigma > 0.0):
                continue
            if (close_ms - now) / 1000.0 <= 0.0:
                continue
            if rng.random() >= fire_prob.get(asset, 0.0):
                continue
            direction = "Up" if p_up >= 0.5 else "Down"
            if last_take is not None and now - last_take < mj.COOLDOWN_MS:
                continue
            if budget_left < mj.MIN_NOTIONAL:
                continue
            book = mj._book_asof(books, up_token, now + latency_ms)
            if book is None:
                continue
            quote = mj._leaned_quote(book, direction)
            if quote is None:
                continue
            price, avail = quote
            fair = p_up if direction == "Up" else 1.0 - p_up
            shares = mj._plan_take_single_level(fair, price, avail, budget_left, fee_rate)
            if shares <= 0.0:
                continue
            trade = mj._mark_to_resolution(direction, shares, price, outcome_up, fee_rate)
            budget_left -= shares * price
            last_take = now
            rows.append({"series": key[0], "window_open_ms": int(key[1]),
                         "day": int(key[1]) // MS_PER_DAY, **trade})
    return rows


# --- forensic: shuffled-labels falsification -------------------------------
def _shuffle_outcomes(win_by_key: dict, series_key: str, day_str: str, seed: int) -> None:
    """Permute ``outcome_up`` across a day's windows (in place), destroying the
    signal→outcome link. Deterministic per (series, day) via a crc32 salt. If the real
    trigger still 'makes money' on shuffled outcomes, the edge is an artifact."""
    keys = sorted(win_by_key.keys())
    outs = np.array([win_by_key[k]["outcome_up"] for k in keys])
    salt = zlib.crc32(f"shuffle|{series_key}|{day_str}".encode()) % 1_000_000
    perm = np.random.default_rng(seed + salt).permutation(len(outs))
    for k, o in zip(keys, outs[perm]):
        win_by_key[k]["outcome_up"] = int(o)


# --- aggregation ------------------------------------------------------------
def _hour_of(open_ms: int) -> int:
    return datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).hour


def _month_of(open_ms: int) -> str:
    dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def _drawdown(trades: list[dict]) -> float:
    """Max drawdown of the running cumulative net PnL (trades ordered by window open)."""
    if not trades:
        return 0.0
    order = sorted(trades, key=lambda r: r["window_open_ms"])
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for t in order:
        cum += t["net"]
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return float(mdd)


def _agg(trades: list[dict], key_fn) -> list[dict]:
    by: dict = defaultdict(list)
    for t in trades:
        by[key_fn(t)].append(t)
    out = []
    for k in sorted(by):
        g = by[k]
        wins = sum(1 for r in g if r["won"])
        out.append({"key": k, "trades": len(g), "win_rate": wins / len(g) if g else float("nan"),
                    "net_pnl": float(sum(r["net"] for r in g)), "fees": float(sum(r["fee"] for r in g)),
                    "drawdown": _drawdown(g), "low_sample": bool(len(g) < 30)})
    return out


# --- worker -----------------------------------------------------------------
def backtest_momentum(
    paths: Paths, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None,
    seeds: int = DEFAULT_SEEDS, seed: int = DEFAULT_SEED,
    latency_ms: int = mj.DEFAULT_LATENCY_MS, venue_delay_ms: int = mj.DEFAULT_VENUE_DELAY_MS,
    window_budget: float = mj.DEFAULT_WINDOW_BUDGET,
    fee_rate: float = mj.CRYPTO_FEE_RATE, res_map: dict | None = None,
    out_name: str = "momentum", invert: bool = False, shuffle_labels: bool = False,
) -> dict:
    """Replay the momentum trigger + controls over the historical span; write the report.
    Returns the metrics dict.

    ``latency_ms`` is the NETWORK round-trip; the momentum trigger fires FAK (marketable)
    takers, so ``venue_delay_ms`` is added on top and the book is read as-of
    network + venue. ``out_name`` sets the ``out/backtests/<out_name>`` output dir (and its
    own checkpoint dir), so a current-regime verdict run never collides with the full run.
    """
    paths.ensure_out()
    effective_latency_ms = latency_ms + venue_delay_ms
    out_dir = paths.out_dir / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    if res_map is None:
        res_map = hr.load_resolution_map(paths, series_keys)  # official resolution (window outcome)
    strike_tol_ms = ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0

    real_trades: list[dict] = []
    control_net_by_seed = np.zeros(seeds, dtype=float)
    control_trades_by_seed = np.zeros(seeds, dtype=int)
    tot = defaultdict(int)

    # Resume: load any per-(series, day) checkpoints (overnight-safe — a killed run picks up
    # where it left off). Each checkpoint holds a day's trades + the control's per-seed net.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(data.get("seeds", -1)) != seeds:  # a seed-count change invalidates old control sums
            continue
        real_trades.extend(data.get("trades", []))
        cn, ct = np.array(data.get("control_net", [])), np.array(data.get("control_trades", []))
        if cn.shape == control_net_by_seed.shape:
            control_net_by_seed += cn
            control_trades_by_seed += ct
        tot["windows"] += int(data.get("windows", 0))
        done.add((data["series"], data["day"]))

    for series_key in series_keys:
        prefix = hc.series_prefix(series_key)
        asset = hc.SLUG_PREFIXES[prefix][1]
        symbol = hc.SYMBOL_BY_ASSET[asset]
        for d in hlbl._days_for_series(paths, series_key, since_ms, until_ms):
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
            grid, ring_ts, ring_px = (_day_grid_and_ring(paths, symbol, asset, d) if wins
                                      else (pd.DataFrame(), None, None))
            model_rows: list[dict] = []
            win_by_key: dict[tuple, dict] = {}
            books: dict[str, dict] = {}
            if wins and not grid.empty:
                mids_by_asset = {asset: (ring_ts, ring_px)}
                for w in wins:
                    open_ms, close_ms = w["window_open_ms"], w["window_close_ms"]
                    official = res_map.get((series_key, int(open_ms)))
                    if official is None:  # no official resolution → skip (never guessed)
                        continue
                    st = hc.binance_anchored_strike(_grid_bars(grid), open_ms, strike_tol_ms)
                    if not (st["strike"] > 0.0):
                        continue
                    upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
                    up_token = str(upq["token_id"].iloc[0]) if not upq.empty else None
                    if up_token is None:
                        continue
                    key = (series_key, int(open_ms))
                    win_by_key[key] = {"up_token": up_token, "close_time": int(close_ms),
                                       "outcome_up": 1 if official["outcome"] == "Up" else 0}
                    books[up_token] = tpm.pm_book_arrays(upq)
                    model_rows.extend(_window_model_rows(grid, w, st))
                if shuffle_labels and win_by_key:
                    _shuffle_outcomes(win_by_key, series_key, day_str, seed)

            day_trades: list[dict] = []
            day_cnet = np.zeros(seeds, dtype=float)
            day_ctrades = np.zeros(seeds, dtype=int)
            if model_rows:
                day_trades = mj._run_momentum(model_rows, mids_by_asset, win_by_key, books,
                                              latency_ms=effective_latency_ms, fee_rate=fee_rate,
                                              window_budget=window_budget, invert=invert)
                elig, conf = _eligible_and_confirmed(model_rows, mids_by_asset, win_by_key)
                fire_prob = {a: (conf[a] / elig[a]) if elig.get(a) else 0.0 for a in elig}
                # Deterministic per-(series, day) salt (zlib.crc32 is stable across processes,
                # unlike hash() with PYTHONHASHSEED) → reproducible shuffled control.
                salt = zlib.crc32(f"{series_key}|{day_str}".encode()) % 1_000_000
                for si in range(seeds):
                    rng = np.random.default_rng(seed + si * 7919 + salt)
                    ctr = _run_random_control(model_rows, win_by_key, books, fire_prob=fire_prob,
                                              rng=rng, latency_ms=effective_latency_ms, fee_rate=fee_rate,
                                              window_budget=window_budget)
                    day_cnet[si] = float(sum(r["net"] for r in ctr))
                    day_ctrades[si] = len(ctr)

            # checkpoint (atomic) + accumulate
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"series": series_key, "day": day_str, "seeds": seeds,
                                       "windows": len(win_by_key), "trades": day_trades,
                                       "control_net": day_cnet.tolist(),
                                       "control_trades": day_ctrades.tolist()},
                                      default=eh._json_default), encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            real_trades.extend(day_trades)
            control_net_by_seed += day_cnet
            control_trades_by_seed += day_ctrades
            tot["windows"] += len(win_by_key)

    return _finalize(paths, out_dir, real_trades, control_net_by_seed, control_trades_by_seed,
                     series_keys, seeds, seed, latency_ms, venue_delay_ms, effective_latency_ms,
                     window_budget, fee_rate, tot)


def _grid_bars(grid: pd.DataFrame) -> pd.DataFrame:
    """The (sec, price) bars behind a feature grid — for the Binance-anchored strike."""
    if grid.empty:
        return pd.DataFrame({"sec": pd.Series([], dtype="int64"), "price": pd.Series([], dtype="float64")})
    return grid[["sec", "mid"]].rename(columns={"mid": "price"})


def _finalize(paths, out_dir, real_trades, control_net_by_seed, control_trades_by_seed,
              series_keys, seeds, seed, network_ms, venue_delay_ms, effective_latency_ms,
              window_budget, fee_rate, tot) -> dict:
    real_net = float(sum(t["net"] for t in real_trades))
    real_fees = float(sum(t["fee"] for t in real_trades))
    real_wins = sum(1 for t in real_trades if t["won"])
    n = len(real_trades)
    real_dd = _drawdown(real_trades)
    control_mean = float(np.mean(control_net_by_seed)) if seeds else float("nan")
    control_std = float(np.std(control_net_by_seed)) if seeds else float("nan")
    # permutation-style p: fraction of shuffled seeds whose net ≥ the real net.
    p_value = (float(np.mean(control_net_by_seed >= real_net)) if seeds else float("nan"))
    beats_shuffled = bool(seeds and real_net > control_mean and p_value <= 0.1)
    # majority lift: the real momentum trades carry p_up + outcome_up (from _run_momentum).
    lift = mj._lift_over_majority(real_trades)

    per_month = [{**r, "series": None, "month": r["key"]} for r in _agg(real_trades, lambda t: _month_of(t["window_open_ms"]))]
    per_series_month = []
    for sk in series_keys:
        sm = _agg([t for t in real_trades if t["series"] == sk], lambda t: _month_of(t["window_open_ms"]))
        for r in sm:
            per_series_month.append({"series": sk, "month": r["key"], "trades": r["trades"],
                                     "win_rate": r["win_rate"], "net_pnl": r["net_pnl"],
                                     "fees": r["fees"], "drawdown": r["drawdown"], "low_sample": r["low_sample"]})
    per_hour = [{"hour": r["key"], "trades": r["trades"], "win_rate": r["win_rate"],
                 "net_pnl": r["net_pnl"], "low_sample": r["low_sample"]} for r in _agg(real_trades, lambda t: _hour_of(t["window_open_ms"]))]
    per_series = {sk: _agg([t for t in real_trades if t["series"] == sk], lambda t: sk) for sk in series_keys}

    verdict = _verdict(real_net, real_fees, n, real_wins, real_dd, control_mean, control_std,
                       p_value, beats_shuffled, seeds, tot["windows"])
    result = {
        "params": {"title": "Historical momentum backtest", "series": list(series_keys),
                   "seeds": seeds, "seed": seed,
                   "latency_ms": effective_latency_ms, "network_ms": network_ms,
                   "venue_delay_ms": venue_delay_ms, "effective_latency_ms": effective_latency_ms,
                   "window_budget": window_budget, "fee_rate": fee_rate},
        "caveats": mj.CAVEATS,
        "regime_boundaries": regime_boundaries(paths),
        "counts": {"resolved_windows": int(tot["windows"]), "trades": n,
                   "net_pnl": real_net, "fees": real_fees,
                   "win_rate": (real_wins / n) if n else float("nan"),
                   "pnl_per_trade": (real_net / n) if n else float("nan"),
                   "max_drawdown": _drawdown(real_trades)},
        "controls": {"never_trading_net": 0.0, "shuffled_mean_net": control_mean,
                     "shuffled_std_net": control_std,
                     "shuffled_mean_trades": float(np.mean(control_trades_by_seed)) if seeds else 0.0,
                     "shuffled_p_value": p_value, "real_beats_shuffled": beats_shuffled,
                     "shuffled_nets": control_net_by_seed.tolist()},
        "majority_lift": lift,
        "per_month": per_month, "per_series_month": per_series_month, "per_hour": per_hour,
        "per_series": {sk: (rows[0] if rows else {}) for sk, rows in per_series.items()},
        "verdict": verdict, "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_outputs(out_dir, result, real_trades, per_series_month, per_hour)
    return result


def _verdict(real_net, fees, n, wins, dd, cmean, cstd, p, beats, seeds, n_windows) -> str:
    if n == 0:
        return ("The momentum trigger produced no trades against the recorded historical book "
                "over this span (verdict: inconclusive — widen the range or check the data).")
    makes = real_net > 0.0
    parts = [
        f"Over {n_windows:,} resolved windows the engine's momentum trigger fired {n:,} trade(s), "
        f"netting ${real_net:.2f} after ${fees:.2f} fees ({'MAKES' if makes else 'LOSES'} money; "
        f"win rate {wins / n:.0%}, ${real_net / n:.4f}/trade, max drawdown ${dd:.2f}).",
    ]
    if seeds:
        parts.append(
            f"Never-trading nets $0.00. The matched-frequency random-trigger control nets "
            f"${cmean:.2f} ± ${cstd:.2f} over {seeds} seeds; the real trigger "
            f"{'BEATS' if beats else 'does NOT beat'} it (permutation p = {p:.2f}).")
        if not beats:
            parts.append("Verdict: VOID — the momentum trigger does not reliably beat a random "
                         "trigger of the same frequency; any PnL is not attributable to the signal.")
        elif makes:
            parts.append("Verdict: the momentum edge is real on this sample (beats random + makes "
                         "money) — read the caveats and confirm on more history.")
        else:
            parts.append("Verdict: the trigger beats random timing but still loses money after fees "
                         "on this sample — the signal has skill but not enough to clear costs.")
    return " ".join(parts)


# --- charts + report --------------------------------------------------------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _charts(result: dict, real_trades: list[dict]) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out: dict[str, str | None] = {"month": None, "hour": None, "equity": None, "control": None}
    pm = result["per_month"]
    if pm:
        import calendar
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        months = [r["month"] for r in pm]
        ax.bar(range(len(pm)), [r["net_pnl"] for r in pm], color="#1f77b4", width=0.8)
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_xticks(range(len(pm))); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("net PnL $"); ax.set_title("Momentum net PnL per month (venue-regime boundaries)")
        # Regime boundaries as vertical markers at the fractional month position. Bar i occupies
        # the slot [i-0.5, i+0.5]; a boundary on day d of that month sits at i-0.5 + (d-1)/days.
        idx_of = {m: i for i, m in enumerate(months)}
        colors = {"verified": "#555", "detected": "#d62728", "unverified": "#e0a000"}
        _, ymax = ax.get_ylim()
        for b in result.get("regime_boundaries", []):
            try:
                y, mo, dy = (int(x) for x in str(b["date"]).split("-"))
            except (ValueError, KeyError):
                continue
            mkey = f"{y:04d}-{mo:02d}"
            if mkey not in idx_of:
                continue
            x = idx_of[mkey] - 0.5 + (dy - 1) / calendar.monthrange(y, mo)[1]
            col = colors.get(b.get("status"), "#555")
            ax.axvline(x, color=col, linestyle="--", linewidth=1.0, alpha=0.9)
            ax.text(x, ymax, " " + b.get("short", ""), rotation=90, va="top", ha="left",
                    fontsize=6, color=col)
        out["month"] = _png(fig)
    ph = result["per_hour"]
    if ph:
        fig, ax = plt.subplots(figsize=(6.6, 3.0))
        ax.bar([r["hour"] for r in ph], [r["net_pnl"] for r in ph], color="#2ca02c")
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("net PnL $"); ax.set_title("Net PnL by time-of-day")
        out["hour"] = _png(fig)
    if real_trades:
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        for sk in result["params"]["series"]:
            ser = sorted((t for t in real_trades if t["series"] == sk), key=lambda r: r["window_open_ms"])
            if not ser:
                continue
            cum = np.cumsum([t["net"] for t in ser])
            ax.plot(range(len(cum)), cum, label=sk, linewidth=1.1)
        ax.axhline(0, color="#333", linewidth=0.6); ax.set_xlabel("trade #"); ax.set_ylabel("cumulative net $")
        ax.set_title("Equity curve per series"); ax.legend(fontsize=8)
        out["equity"] = _png(fig)
    nets = result["controls"]["shuffled_nets"]
    if nets:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.hist(nets, bins=min(20, max(3, len(nets))), color="#999", alpha=0.8, label="shuffled")
        ax.axvline(result["counts"]["net_pnl"], color="#d62728", linewidth=1.6, label="real")
        ax.set_xlabel("net PnL $"); ax.set_ylabel("seeds"); ax.set_title("Real vs matched-frequency random control")
        ax.legend(fontsize=8)
        out["control"] = _png(fig)
    return out


def _write_outputs(out_dir: Path, result: dict, real_trades, per_series_month, per_hour) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    pd.DataFrame(real_trades).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(per_series_month).to_csv(out_dir / "per_series_month.csv", index=False)
    pd.DataFrame(per_hour).to_csv(out_dir / "per_hour.csv", index=False)
    charts = _charts(result, real_trades)
    def img(b, alt): return eh._img_tag(b, alt)
    sm_cols = [("series", "series"), ("month", "month"), ("trades", "trades"), ("win_rate", "win rate"),
               ("net_pnl", "net $"), ("fees", "fees $"), ("drawdown", "max DD $"), ("low_sample", "low_sample")]
    hr_cols = [("hour", "hour UTC"), ("trades", "trades"), ("win_rate", "win rate"),
               ("net_pnl", "net $"), ("low_sample", "low_sample")]
    ctrl = result["controls"]
    ctrl_rows = [
        {"strategy": "momentum (real)", "net_pnl": result["counts"]["net_pnl"], "trades": result["counts"]["trades"]},
        {"strategy": "random control (mean)", "net_pnl": ctrl["shuffled_mean_net"], "trades": ctrl["shuffled_mean_trades"]},
        {"strategy": "never trading", "net_pnl": 0.0, "trades": 0},
    ]
    rb_rows = [{"date": b["date"], "event": b["label"], "status": b["status"]}
               for b in result.get("regime_boundaries", [])]
    caveats = "".join(f"<li>{c}</li>" for c in result["caveats"])
    bug = " bug" if (result["params"]["seeds"] and not ctrl["real_beats_shuffled"]) else ""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} tr.low td {{ background: #fff5e6; color: #7a5200; }}
 .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 .verdict.bug {{ border-left-color: #d62728; background: #fdeeee; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<p class="muted">The engine's hand-coded momentum trigger (money_judge ports), replayed over
the reconstructed historical book. Conservative fills; marked to the official resolution.</p>
<div class="verdict{bug}">{result['verdict']}</div>
{img(charts['control'], 'real vs random control')}
<h2>Strategy comparison</h2>
{eh._table_html(ctrl_rows, [("strategy", "strategy"), ("net_pnl", "net PnL $"), ("trades", "trades")])}
{img(charts['month'], 'PnL per month')}
<h2>Venue-regime boundaries</h2>
<p class="muted">Vertical markers on the per-month chart. The 250 ms-delay introduction date is
unverified in the docs — a "detected" row (if present) is pinned by <code>detect_delay.py</code>.</p>
{eh._table_html(rb_rows, [("date", "date UTC"), ("event", "event"), ("status", "status")])}
<h2>Per series per month</h2>
{eh._table_html(per_series_month, sm_cols)}
{img(charts['equity'], 'equity per series')}
{img(charts['hour'], 'PnL by hour')}
<h2>By time-of-day (UTC hour)</h2>
{eh._table_html(per_hour, hr_cols)}
<h2>Caveats (read these)</h2><ul>{caveats}</ul>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "backtest_momentum")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help=f"random-control seeds (default {DEFAULT_SEEDS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--latency-ms", type=int, default=mj.DEFAULT_LATENCY_MS,
                        help=f"NETWORK latency ms (default {mj.DEFAULT_LATENCY_MS}; VPS-colocated ≈ 5)")
    parser.add_argument("--venue-delay-ms", type=int, default=mj.DEFAULT_VENUE_DELAY_MS,
                        help=f"venue taker delay ms added on top for marketable (FAK) fills "
                             f"(default {mj.DEFAULT_VENUE_DELAY_MS}; use 0 for a full-history/pre-delay run)")
    parser.add_argument("--out-name", default="momentum",
                        help="output subdir under out/backtests/ (default 'momentum'); use a "
                             "distinct name for a verdict run so its checkpoints stay isolated")
    parser.add_argument("--fee-rate", type=float, default=mj.CRYPTO_FEE_RATE,
                        help=f"taker fee rate (default {mj.CRYPTO_FEE_RATE}; forensic doubled = 0.14)")
    parser.add_argument("--invert-trigger", action="store_true",
                        help="forensic: bet AGAINST the confirmed direction (a real edge collapses)")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="forensic: permute window outcomes per day (destroys signal→outcome)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    print(f"[backtest_momentum] telonex={paths.telonex_dir} out={paths.out_dir / 'backtests' / args.out_name} "
          f"latency=net {args.latency_ms}+venue {args.venue_delay_ms}={args.latency_ms + args.venue_delay_ms}ms")
    result = backtest_momentum(paths, series=series, since_ms=since_ms, until_ms=until_ms,
                               seeds=args.seeds, seed=args.seed, latency_ms=args.latency_ms,
                               venue_delay_ms=args.venue_delay_ms, out_name=args.out_name,
                               fee_rate=args.fee_rate, invert=args.invert_trigger,
                               shuffle_labels=args.shuffle_labels)
    c = result["counts"]
    print(f"[backtest_momentum] {c['trades']:,} trades over {c['resolved_windows']:,} windows: "
          f"net ${c['net_pnl']:.2f} (fees ${c['fees']:.2f}), win {c['win_rate']:.0%}, DD ${c['max_drawdown']:.2f}")
    ct = result["controls"]
    print(f"[backtest_momentum] control: random ${ct['shuffled_mean_net']:.2f}±${ct['shuffled_std_net']:.2f}, "
          f"real beats={ct['real_beats_shuffled']} (p={ct['shuffled_p_value']:.2f})")
    procmem.report_peak_rss("backtest_momentum", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[backtest_momentum] wrote report.html + metrics.json + csvs to {paths.out_dir / 'backtests' / args.out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
