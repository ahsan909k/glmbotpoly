"""Stage — backtest_maker_core: does the champion model DEFEND maker quotes better than
always-on quoting?

Faithfully replays the engine's two-sided **maker core** (``crates/engine/src/quoting.rs`` +
the ``quote_manager`` cancel-first defense, ported in :mod:`maker_core_sim`) over the
current-regime Telonex cache, filling resting post-only quotes from **real Polymarket trade
prints** (queue-behind-displayed, a mirror of ``venue-paper::MatchEngine::on_trade`` — Telonex
*does* carry a per-outcome trade tape, ``io/telonex_pm.read_pm_trades``, correcting the stale
"no PM trade tape" caveat the other backtests inherited).

Runs three variants on **identical** windows:

- **(a) always-on** — the real engine quoting rules, reactive urgent-cancel included (the bar);
- **(b) model-defended** — identical, but when the walk-forward **dir10** model fires
  (``|p_up − 0.5| ≥ θ_defend``, sweep 0.10/0.15/0.20), pull the *threatened* side (the mirror
  side the predicted move runs over) until the signal clears. The signal is taken **as-of** with
  a live-realistic staleness: a prediction older than 15 s ⇒ stand down (the shadow's
  ``max_prediction_age_ms``); the effective decision lag is measured and reported;
- **(c) model-leaned** — don't pull; widen the threatened side and tighten the safe side by a
  fixed half-spread multiple (3-cell sweep 0.5/1.0/1.5, at a fixed ``--lean-theta``).

Reports **per series**: net PnL, fills, pair-completion rate, 5 s/30 s markout on maker fills
(the adverse-selection measure), stranded-leg PnL, plus a maker-rebate estimate. A variant/cell
"wins" only if it beats (a) **pooled AND in a per-series majority** (≥ 3 of 4 series). A
**shuffled-signal** control on the winner must NOT beat (a), else the result is void.

**No retraining.** The dir10 signal = the already-written walk-forward OOS predictions
``out/learn_walkforward/oos_dir10_full.parquet``. This stage is LightGBM-free (it reuses the
``money_judge`` book helpers + the ``backtest_momentum`` reconstruction). Output
``out/backtests/maker_core/``. Resumable per-(series, day). **Research only.**

Run::

    python -m model_lab.backtest_maker_core                        # full current regime, 4 series
    python -m model_lab.backtest_maker_core --series BTC-5m --since 2026-06-05 --until 2026-06-08
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import zlib
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_momentum as bm
from . import dataset as ds
from . import eval_harness as eh
from . import historical_common as hc
from . import historical_labels as hlbl
from . import historical_resolutions as hr
from . import maker_core_sim as sim
from . import money_judge as mj
from .config import Paths, assert_parquet_ready, resolve_bounds, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .lib import procmem

MS_PER_DAY = 86_400_000
DEFAULT_SEED = 20260714
DEFAULT_SEEDS = 3  # shuffled-signal control seeds (each is a full re-simulation of firing windows)
CURRENT_REGIME_FROM = date(2026, 6, 5)  # 250 ms taker-delay lock (out/audits/venue_regime_timeline.md)

# --- sweep grid (operator-confirmed) ----------------------------------------
DEFEND_THETAS = [0.10, 0.15, 0.20]  # |p_up − 0.5| ≥ θ_defend (defend pull trigger)
LEAN_MULTS = [0.5, 1.0, 1.5]  # lean = widen threatened / tighten safe by this × half-spread (3 cells)
DEFAULT_LEAN_THETA = 0.15  # lean fires at a single fixed θ (keeps the lean sweep to 3 cells)

CAVEATS = [
    "The dir10 signal is the 15 s-grid walk-forward OOS (staler than the live shadow's ~5 s); it "
    "is taken as-of with a >15 s stand-down, and the effective decision lag is reported. A staler "
    "signal is conservative against the model.",
    "Sub-touch queue depth is unobservable from the top-of-book Telonex quotes → a resting BUY "
    "below the best bid queues behind the displayed best-bid size (a conservative proxy; a full "
    "L2 ladder exists only for 2026-07-05 BTC-5m).",
    "Reconstructed fair is the price-only Φ(z) (basis = 0, from reconstruct_window_snapshots); a "
    "1 s quote cadence approximates the engine's 250 ms requote budget (slightly staler ⇒ "
    "conservative on both baseline and variants).",
    "Maker fills pay $0 fee; the maker-rebate estimate (20% × the taker fee on filled maker "
    "volume) is a separate reported line, never in the traded PnL.",
    "PM trade `side` is taken as the taker/aggressor side (sell-aggressor prints fill our resting "
    "buys); confirmed with a directional check on the bounded run (--maker-side flips it).",
    "The relative comparison across (a)/(b)/(c) is robust to the fill-model assumptions (all three "
    "run the identical engine + fill model on the identical windows).",
]


# --- current-regime helpers (local; learn_walkforward pulls in lightgbm) -----
def _regime_floor_ms(regime_from: date) -> int:
    return int(datetime(regime_from.year, regime_from.month, regime_from.day,
                        tzinfo=timezone.utc).timestamp()) * 1000


def _current_regime_since(since_ms: int | None, regime_from: date) -> int:
    floor = _regime_floor_ms(regime_from)
    return max(floor, since_ms) if since_ms is not None else floor


def _group_current_regime(oos: pd.DataFrame, since_ms: int) -> dict:
    """{(series, window_open_ms): slice} over current-regime windows only."""
    cr = oos[oos["window_open_ms"].to_numpy(dtype="int64") >= since_ms]
    return {k: v for k, v in cr.groupby(["series", "window_open_ms"], sort=False)}


# --- config keys ------------------------------------------------------------
def _defend_key(theta: float) -> str:
    return f"defend|{theta:g}"


def _lean_key(mult: float) -> str:
    return f"lean|{mult:g}"


def _model_configs(defend_thetas, lean_mults) -> list[str]:
    return [_defend_key(t) for t in defend_thetas] + [_lean_key(m) for m in lean_mults]


# --- per-window driver ------------------------------------------------------
def _print_arrays(trades: pd.DataFrame, maker_side: bool) -> dict:
    """The sell-aggressor print arrays the sim consumes. ``maker_side=False`` (default): the
    Telonex ``side`` is the taker/aggressor side, so a ``sell`` print is a sell-aggressor that
    hits our resting buy. ``maker_side=True`` flips it (``side`` names the resting maker)."""
    if trades is None or trades.empty:
        return {"ts": np.empty(0, dtype="int64"), "price": np.empty(0, dtype=float),
                "size": np.empty(0, dtype=float), "is_sell": np.empty(0, dtype=bool)}
    side = trades["side"].astype(str).str.lower().to_numpy()
    sell_label = "sell"
    is_sell = (side == sell_label)
    if maker_side:  # side names the maker → a `buy`-maker was hit by a sell-aggressor
        is_sell = ~is_sell
    return {"ts": trades["ts_ms"].to_numpy(dtype="int64"),
            "price": trades["price"].to_numpy(dtype=float),
            "size": trades["size"].to_numpy(dtype=float), "is_sell": is_sell}


def _signal_arrays(g: pd.DataFrame | None) -> dict | None:
    if g is None or not len(g):
        return None
    order = np.argsort(g["sample_ts_ms"].to_numpy(dtype="int64"))
    return {"ts": g["sample_ts_ms"].to_numpy(dtype="int64")[order],
            "p_up": g["p_up"].to_numpy(dtype=float)[order]}


def _shuffled_signal(signal: dict, seed_salt: int) -> dict:
    """A within-window signal shuffle: permute the p_up values across the window's prediction
    timestamps. Preserves the per-window fire count exactly, breaks the signal↔price-move timing
    correlation (the essence of the control). Deterministic."""
    p = signal["p_up"].copy()
    np.random.default_rng(seed_salt).shuffle(p)
    return {"ts": signal["ts"], "p_up": p}


def _run_window(win: dict, params, defend_thetas, lean_mults, lean_theta,
                seeds, seed, series_key, day_str) -> tuple[dict, dict]:
    """Run all configs on one window. Returns ``(metrics_by_ckey, shuf_net_by_ckey)`` where
    ``shuf_net_by_ckey[ckey]`` is a length-``seeds`` array of shuffled-signal net PnL."""
    kw = dict(rows_ts=win["rows_ts"], rows_p_up=win["rows_p_up"], rows_sigma=win["rows_sigma"],
              up_book=win["up_book"], up_prints=win["up_prints"],
              down_book=win["down_book"], down_prints=win["down_prints"],
              outcome_up=win["outcome_up"], close_ms=win["close_ms"], params=params)
    signal = win["signal"]

    base = sim.simulate_window(mode="baseline", **kw)
    out = {"baseline": base}
    shuf: dict[str, np.ndarray] = {}

    for theta in defend_thetas:
        ck = _defend_key(theta)
        if signal is not None and sim.signal_fires(signal, win["rows_ts"], theta):
            out[ck] = sim.simulate_window(mode="defend", signal=signal, theta=theta, **kw)
        else:
            out[ck] = dict(base)  # no fire ⇒ identical to baseline
    for mult in lean_mults:
        ck = _lean_key(mult)
        if signal is not None and sim.signal_fires(signal, win["rows_ts"], lean_theta):
            out[ck] = sim.simulate_window(mode="lean", signal=signal, theta=lean_theta,
                                          lean_mult=mult, **kw)
        else:
            out[ck] = dict(base)

    # shuffled-signal control (net-only) for the model configs.
    salt = zlib.crc32(f"{series_key}|{day_str}|{win['open_ms']}".encode()) % 1_000_000
    for ck in _model_configs(defend_thetas, lean_mults):
        shuf[ck] = np.full(seeds, base["net_pnl"], dtype=float)
    if signal is not None and seeds > 0:
        for si in range(seeds):
            ssig = _shuffled_signal(signal, seed + si * 7919 + salt)
            for theta in defend_thetas:
                ck = _defend_key(theta)
                if sim.signal_fires(ssig, win["rows_ts"], theta):
                    shuf[ck][si] = sim.simulate_window(mode="defend", signal=ssig, theta=theta,
                                                       **kw)["net_pnl"]
            for mult in lean_mults:
                ck = _lean_key(mult)
                if sim.signal_fires(ssig, win["rows_ts"], lean_theta):
                    shuf[ck][si] = sim.simulate_window(mode="lean", signal=ssig, theta=lean_theta,
                                                       lean_mult=mult, **kw)["net_pnl"]
    return out, shuf


# --- worker -----------------------------------------------------------------
def backtest_maker_core(
    paths: Paths, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None,
    defend_thetas=None, lean_mults=None, lean_theta: float = DEFAULT_LEAN_THETA,
    seeds: int = DEFAULT_SEEDS, seed: int = DEFAULT_SEED, maker_side: bool = False,
    target: str = "dir10", variant: str = "full", oos: pd.DataFrame | None = None,
    res_map: dict | None = None, regime_from: date = CURRENT_REGIME_FROM,
    out_name: str = "maker_core", finalize_only: bool = False,
) -> dict:
    """Replay the maker-core baseline + defend/lean variants + shuffled control over the
    current-regime windows; write the report. ``oos``/``res_map`` are loaded from disk when
    ``None`` (both injectable for tests); ``regime_from`` overrides the current-regime floor.
    ``finalize_only=True`` skips the day loop and just re-aggregates the existing checkpoints into
    the report (safe to run alongside the per-series workers — reads checkpoints, no day writes)."""
    paths.ensure_out()
    defend_thetas = list(defend_thetas if defend_thetas is not None else DEFEND_THETAS)
    lean_mults = list(lean_mults if lean_mults is not None else LEAN_MULTS)
    params = sim.QuoteParams()
    out_dir = paths.out_dir / "backtests" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    excluded, _ = hc.load_excluded_slugs(paths)
    since_floor = _current_regime_since(since_ms, regime_from)
    strike_tol_ms = ds.DEFAULT_STRIKE_TOLERANCE_SECS * 1000.0
    groups: dict = {}
    if not finalize_only:  # the OOS + res_map + grouping are only needed to PROCESS days
        if oos is None:
            path = paths.out_dir / "learn_walkforward" / f"oos_{target}_{variant}.parquet"
            assert_parquet_ready(path, label=f"oos_{target}_{variant}.parquet", min_rows=1)
            oos = pd.read_parquet(path, columns=["series", "window_open_ms", "sample_ts_ms", "p_up"],
                                  filters=[("series", "in", list(series_keys))])
        if res_map is None:
            res_map = hr.load_resolution_map(paths, series_keys)
        groups = _group_current_regime(oos, since_floor)
    ckeys = ["baseline"] + _model_configs(defend_thetas, lean_mults)

    # accumulators (per series; overall = sum over series). Memory-bounded: aggregates only.
    agg: dict = defaultdict(sim._empty_metrics)              # (series, ckey) -> metrics
    shuf: dict = defaultdict(lambda: np.zeros(seeds))        # (series, model_ckey) -> net per seed
    n_windows = 0
    done: set[tuple[str, str]] = set()

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _accumulate(data: dict) -> None:
        nonlocal n_windows
        sk = data["series"]
        for ck, met in data["agg"].items():
            sim.add_metrics(agg[(sk, ck)], met)
        for ck, vec in data["shuf"].items():
            shuf[(sk, ck)] += np.array(vec, dtype=float)
        n_windows += int(data["windows"])

    for cp in sorted(ckpt_dir.glob("*.json")):
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(data.get("seeds", -1)) != seeds:
            continue
        try:
            _accumulate(data)
            done.add((data["series"], data["day"]))
        except (KeyError, ValueError):
            continue

    for series_key in ([] if finalize_only else series_keys):
        prefix = hc.series_prefix(series_key)
        asset = hc.SLUG_PREFIXES[prefix][1]
        symbol = hc.SYMBOL_BY_ASSET[asset]
        for d in hlbl._days_for_series(paths, series_key, since_floor, until_ms):
            day_str = d.isoformat()
            if (series_key, day_str) in done:
                continue
            day = _process_day(paths, series_key, asset, symbol, d, day_str, groups, res_map,
                               excluded, strike_tol_ms, params, defend_thetas, lean_mults,
                               lean_theta, seeds, seed, maker_side)
            cp = ckpt_dir / f"{series_key}__{day_str}.json"
            tmp = cp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"seeds": seeds, **day}, default=eh._json_default),
                           encoding="utf-8")
            tmp.replace(cp)
            done.add((series_key, day_str))
            _accumulate({"seeds": seeds, **day})
            print(f"[backtest_maker_core] {series_key} {day_str}: {day['windows']} windows "
                  f"({len(done)} days done, {n_windows:,} cumulative), "
                  f"peak RSS {procmem.peak_rss_mb():.0f} MB", flush=True)

    return _finalize(paths, out_dir, series_keys, defend_thetas, lean_mults, lean_theta, ckeys,
                     agg, shuf, n_windows, seeds, seed, since_floor, target, variant)


def _process_day(paths, series_key, asset, symbol, d, day_str, groups, res_map, excluded,
                 strike_tol_ms, params, defend_thetas, lean_mults, lean_theta, seeds, seed,
                 maker_side) -> dict:
    """Reconstruct one (series, day)'s windows once (both Up+Down quotes + trade tapes), run all
    configs + the shuffled control, and return the day's AGGREGATES (no rows retained)."""
    wins, _ = hc.telonex_windows_present(paths, series_key, d, excluded=excluded)
    grid, _ring_ts, _ring_px = (bm._day_grid_and_ring(paths, symbol, asset, d) if wins
                                else (pd.DataFrame(), None, None))
    agg: dict = {ck: sim._empty_metrics() for ck in
                 ["baseline"] + _model_configs(defend_thetas, lean_mults)}
    shuf: dict = {ck: np.zeros(seeds) for ck in _model_configs(defend_thetas, lean_mults)}
    n_win = 0
    if wins and not grid.empty:
        bars = bm._grid_bars(grid)
        for w in wins:
            open_ms, close_ms = int(w["window_open_ms"]), int(w["window_close_ms"])
            official = res_map.get((series_key, open_ms))
            if official is None:
                continue
            st = hc.binance_anchored_strike(bars, open_ms, strike_tol_ms)
            if not (st["strike"] > 0.0):
                continue
            snaps = hc.reconstruct_window_snapshots(grid, {**w, **st})
            if snaps.empty:
                continue
            upq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Up", day_str)
            dnq = tpm.read_pm_quotes(paths.telonex_dir, w["slug"], "Down", day_str)
            if upq.empty or dnq.empty:
                continue
            up_tr = tpm.read_pm_trades(paths.telonex_dir, w["slug"], "Up", day_str)
            dn_tr = tpm.read_pm_trades(paths.telonex_dir, w["slug"], "Down", day_str)
            win = {
                "open_ms": open_ms, "close_ms": close_ms,
                "outcome_up": 1 if official["outcome"] == "Up" else 0,
                "rows_ts": snaps["ts"].to_numpy(dtype="int64"),
                "rows_p_up": snaps["p_up"].to_numpy(dtype=float),
                "rows_sigma": snaps["sigma_1s"].to_numpy(dtype=float),
                "up_book": tpm.pm_book_arrays(upq), "down_book": tpm.pm_book_arrays(dnq),
                "up_prints": _print_arrays(up_tr, maker_side),
                "down_prints": _print_arrays(dn_tr, maker_side),
                "signal": _signal_arrays(groups.get((series_key, open_ms))),
            }
            metrics, shuf_net = _run_window(win, params, defend_thetas, lean_mults, lean_theta,
                                            seeds, seed, series_key, day_str)
            for ck, met in metrics.items():
                sim.add_metrics(agg[ck], met)
            for ck, vec in shuf_net.items():
                shuf[ck] += vec
            n_win += 1

    return {"series": series_key, "day": day_str, "windows": n_win,
            "agg": agg, "shuf": {k: v.tolist() for k, v in shuf.items()}}


# --- finalize ---------------------------------------------------------------
def _rate(a: float, b: float) -> float:
    return (a / b) if b else float("nan")


def _summary_row(a: dict) -> dict:
    fs, ms = a["fill_shares"], a["markout_shares"]
    return {
        "net_pnl": float(a["net_pnl"]), "locked_pnl": float(a["locked_pnl"]),
        "stranded_pnl": float(a["stranded_pnl"]), "rebate": float(a["rebate"]),
        "n_fills": int(a["n_fills"]), "fill_shares": float(fs),
        "windows_traded": int(a["windows_traded"]),
        "pair_completion": _rate(a["matched_shares"], fs),
        "markout5": _rate(a["markout5_wsum"], ms), "markout30": _rate(a["markout30_wsum"], ms),
        "avg_lag_ms": _rate(a["lag_ms_sum"], a["lag_n"]),
    }


def _control_stats(shuf_vec, real_net: float, baseline_net: float, seeds: int) -> dict:
    """The operator's pre-registered shuffled-signal control: the shuffled-signal version of the
    winner (same per-window fire frequency, RANDOM timing) must NOT beat the always-on baseline —
    else the improvement is attributable to pulling *per se* (reduced exposure / quoting less), not
    to the model's signal, and the result is VOID. Also decomposes the winner's improvement over
    baseline into the exposure-reduction effect (baseline → shuffled) and the signal-timing effect
    (shuffled → winner)."""
    if not seeds:
        return {"shuffled_net": float("nan"), "shuffled_beats_baseline": False,
                "winner_beats_shuffled": False, "action_effect": float("nan"),
                "signal_effect": float("nan"), "p_shuffle_beats_base": float("nan"), "nets": []}
    vec = np.asarray(shuf_vec, dtype=float)
    mean = float(np.mean(vec))
    return {
        "shuffled_net": mean,
        "shuffled_beats_baseline": bool(mean > baseline_net),   # operator's VOID trigger
        "winner_beats_shuffled": bool(real_net > mean),
        "action_effect": mean - baseline_net,      # baseline → shuffled: exposure reduction (quoting less)
        "signal_effect": real_net - mean,          # shuffled → winner: the model's signal-timing value
        "p_shuffle_beats_base": float(np.mean(vec > baseline_net)),
        "nets": vec.tolist(),
    }


def _finalize(paths, out_dir, series_keys, defend_thetas, lean_mults, lean_theta, ckeys,
              agg, shuf, n_windows, seeds, seed, since_floor, target, variant) -> dict:
    model_ckeys = _model_configs(defend_thetas, lean_mults)
    # pooled (sum over series) per config.
    pooled: dict = {ck: sim._empty_metrics() for ck in ckeys}
    pooled_shuf: dict = {ck: np.zeros(seeds) for ck in model_ckeys}
    for sk in series_keys:
        for ck in ckeys:
            sim.add_metrics(pooled[ck], agg[(sk, ck)])
        for ck in model_ckeys:
            pooled_shuf[ck] += shuf[(sk, ck)]

    base_pooled = _summary_row(pooled["baseline"])
    rows = {ck: _summary_row(pooled[ck]) for ck in ckeys}

    # per config: pooled delta + per-series majority + shuffled control.
    n_series = len(series_keys)
    majority_needed = (n_series // 2) + 1
    cfg_eval: dict = {}
    for ck in model_ckeys:
        delta = rows[ck]["net_pnl"] - base_pooled["net_pnl"]
        beats_series = sum(1 for sk in series_keys
                           if agg[(sk, ck)]["net_pnl"] > agg[(sk, "baseline")]["net_pnl"])
        ctrl = _control_stats(pooled_shuf[ck], rows[ck]["net_pnl"], base_pooled["net_pnl"], seeds)
        cfg_eval[ck] = {
            "delta_vs_baseline": delta, "beats_series": beats_series, "n_series": n_series,
            "per_series_majority": bool(beats_series >= majority_needed),
            "pooled_beats": bool(delta > 0.0),
            "robust": bool(delta > 0.0 and beats_series >= majority_needed),
            "control": ctrl,
        }

    # winner = the robust model config with the best pooled net; else the best pooled net (flagged).
    robust = [ck for ck in model_ckeys if cfg_eval[ck]["robust"]]
    pool = robust if robust else model_ckeys
    winner = max(pool, key=lambda ck: rows[ck]["net_pnl"]) if pool else None

    per_series = {sk: {ck: _summary_row(agg[(sk, ck)]) for ck in ckeys} for sk in series_keys}

    verdict = _verdict(winner, rows, base_pooled, cfg_eval, n_windows, seeds, robust)
    result = {
        "params": {"title": "Maker-core defense backtest", "series": list(series_keys),
                   "target": target, "variant": variant, "defend_thetas": defend_thetas,
                   "lean_mults": lean_mults, "lean_theta": lean_theta, "seeds": seeds, "seed": seed,
                   "regime_since_ms": since_floor,
                   "quote_params": "engine defaults (config/default.toml [engine.defaults])"},
        "caveats": CAVEATS,
        "current_regime_windows": int(n_windows),
        "baseline": {"net_pnl": base_pooled["net_pnl"], **base_pooled},
        "configs": [{"config": ck, **rows[ck],
                     **({"eval": cfg_eval[ck]} if ck in model_ckeys else {})} for ck in ckeys],
        "winner": None if winner is None else {"config": winner, **rows[winner],
                                               "eval": cfg_eval[winner]},
        "per_series": {sk: {ck: per_series[sk][ck] for ck in ckeys} for sk in series_keys},
        "verdict": verdict, "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_outputs(out_dir, result, ckeys, series_keys)
    return result


def _verdict(winner, rows, base, cfg_eval, n_windows, seeds, robust) -> str:
    base_net = base["net_pnl"]
    parts = [
        f"Over {n_windows:,} current-regime windows (≥ 2026-06-05), the always-on maker baseline "
        f"nets ${base_net:,.2f} ({base['n_fills']:,} fills, {base['pair_completion']:.0%} "
        f"pair-completion, 5 s markout {base['markout5']:+.4f})."
    ]
    if winner is None:
        return " ".join(parts) + " No model variant produced a comparison (verdict: inconclusive)."
    w = rows[winner]
    ev = cfg_eval[winner]
    kind = "defend" if winner.startswith("defend") else "lean"
    d_mo = w["markout5"] - base["markout5"]
    d_comp = w["pair_completion"] - base["pair_completion"]
    parts.append(
        f"The best model variant ({winner.replace('|', '=')}) nets ${w['net_pnl']:,.2f} "
        f"({w['net_pnl'] - base_net:+,.2f} vs baseline), 5 s markout {w['markout5']:+.4f} "
        f"({d_mo:+.4f}), pair-completion {w['pair_completion']:.0%} ({d_comp:+.0%}), "
        f"avg signal lag {w['avg_lag_ms']:.0f} ms. It beats the baseline in "
        f"{ev['beats_series']}/{ev['n_series']} series.")
    if not robust:
        parts.append(
            f"Verdict: NO model variant robustly beats always-on quoting — the best cell fails the "
            f"pooled-AND-per-series-majority bar. Always-on quoting is not improved by the "
            f"model's {kind} on this sample.")
        return " ".join(parts)
    via = ("fewer pick-offs (less-negative markout)" if d_mo > 0 and abs(d_mo) >= abs(d_comp)
           else "more completions" if d_comp > 0 else "a mix")
    ctl = ev["control"]
    total = w["net_pnl"] - base_net
    if seeds and ctl["shuffled_beats_baseline"]:
        # The operator's pre-registered control fires: a same-frequency RANDOM pull already beats
        # baseline, so the improvement is mostly exposure-reduction, not the model.
        sig_frac = (100.0 * ctl["signal_effect"] / total) if total else float("nan")
        parts.append(
            f"Verdict: VOID by the pre-registered shuffled-signal control. A same-frequency RANDOM "
            f"pull already nets ${ctl['shuffled_net']:,.2f} (vs baseline ${base_net:,.2f}) — it "
            f"captures ${ctl['action_effect']:,.2f} of the ${total:,.2f} defend improvement just by "
            f"quoting LESS (reducing exposure). The model's signal timing adds only "
            f"${ctl['signal_effect']:,.2f} on top ({sig_frac:.0f}% of the improvement). The model "
            f"does NOT earn a defensive seat: a signal-agnostic 'quote less in volatile windows' "
            f"rule would capture most of the benefit. (The signal's marginal effect is still "
            f"{'positive' if ctl['signal_effect'] > 0 else 'non-positive'}.)")
    else:
        parts.append(
            f"Verdict: the model's {kind} defense beats always-on robustly (pooled + "
            f"{ev['beats_series']}/{ev['n_series']} series), via {via}, AND its same-frequency "
            f"shuffled control does not beat baseline (shuffled ${ctl['shuffled_net']:,.2f} ≤ "
            f"baseline ${base_net:,.2f}) — so the edge is attributable to the model's signal timing "
            f"(${ctl['signal_effect']:,.2f}), not just to quoting less. Confirm on more history.")
    return " ".join(parts)


# --- outputs ----------------------------------------------------------------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _markout_chart(result: dict, ckeys) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {r["config"]: r for r in result["configs"]}
    labels = [ck.replace("|", "=") for ck in ckeys]
    mo5 = [by[ck]["markout5"] for ck in ckeys]
    net = [by[ck]["net_pnl"] for ck in ckeys]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax1.bar(range(len(ckeys)), mo5, color=["#888"] + ["#1f77b4"] * (len(ckeys) - 1))
    ax1.set_xticks(range(len(ckeys))); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.axhline(0, color="#333", lw=0.8); ax1.set_title("5 s markout on maker fills (↑ = fewer pick-offs)")
    ax2.bar(range(len(ckeys)), net, color=["#888"] + ["#2ca02c"] * (len(ckeys) - 1))
    ax2.set_xticks(range(len(ckeys))); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.axhline(0, color="#333", lw=0.8); ax2.set_title("net PnL $")
    return _png(fig)


def _write_outputs(out_dir: Path, result: dict, ckeys, series_keys) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default),
                                          encoding="utf-8")
    # per_series.csv: one row per (series, config).
    rows = []
    for sk in series_keys:
        for ck in ckeys:
            r = result["per_series"][sk][ck]
            rows.append({"series": sk, "config": ck, **r})
    for r in result["configs"]:  # pooled ("ALL"); drop the nested `eval` dict (metrics.json keeps it)
        rows.append({"series": "ALL", **{k: v for k, v in r.items() if k != "eval"}})
    pd.DataFrame(rows).to_csv(out_dir / "per_series.csv", index=False)

    cols = [("config", "config"), ("net_pnl", "net $"), ("n_fills", "fills"),
            ("pair_completion", "pair-compl"), ("markout5", "5s markout"),
            ("markout30", "30s markout"), ("stranded_pnl", "stranded $"), ("avg_lag_ms", "lag ms")]
    chart = _markout_chart(result, ckeys)
    caveats = "".join(f"<li>{c}</li>" for c in result["caveats"])
    win = result.get("winner")
    seeds = result["params"]["seeds"]
    # green only if the winner is robust AND its shuffled control does NOT beat baseline.
    control_ok = win is not None and (seeds == 0 or not win["eval"]["control"]["shuffled_beats_baseline"])
    robust_win = bool(win and win["eval"]["robust"] and control_ok)
    bug = "" if robust_win else " bug"
    ps_tables = []
    for sk in series_keys:
        srows = [{"config": ck, **result["per_series"][sk][ck]} for ck in ckeys]
        ps_tables.append(f"<h3>{sk}</h3>{eh._table_html(srows, cols)}")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 h3 {{ margin: 1rem 0 0.2rem; font-size: 1rem; }}
 table {{ border-collapse: collapse; margin: 0.4rem 0; font-size: 0.9rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 .verdict.bug {{ border-left-color: #d62728; background: #fdeeee; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<p class="muted">Faithful two-sided maker replay (engine quoting rules + reactive cancel), filled
from real Polymarket trade prints (queue-behind-displayed). Does the dir10 model defend quotes
better than always-on quoting? Reconstructed current-regime cache; marked to official resolution.</p>
<div class="verdict{bug}">{result['verdict']}</div>
<h2>All configs (pooled over series)</h2>
{eh._table_html(result['configs'], cols)}
{eh._img_tag(chart, 'markout + net PnL by config')}
<h2>Per series</h2>
{''.join(ps_tables)}
<h2>Caveats (read these)</h2><ul>{caveats}</ul>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "backtest_maker_core")
    parser.add_argument("--series", default=None, help="comma-separated series filter (default 4-series)")
    parser.add_argument("--target", default="dir10", help="OOS target (default dir10)")
    parser.add_argument("--variant", default="full", help="OOS feature variant (default full)")
    parser.add_argument("--defend-thetas", default=None, help="comma-separated defend |p−0.5| grid")
    parser.add_argument("--lean-mults", default=None, help="comma-separated lean half-spread multiples")
    parser.add_argument("--lean-theta", type=float, default=DEFAULT_LEAN_THETA,
                        help=f"fixed |p−0.5| trigger for the lean variant (default {DEFAULT_LEAN_THETA})")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"shuffled-signal control seeds (default {DEFAULT_SEEDS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maker-side", action="store_true",
                        help="treat the Telonex trade `side` as the MAKER side (flip aggressor)")
    parser.add_argument("--out-name", default="maker_core", help="output subdir under out/backtests/")
    parser.add_argument("--finalize-only", action="store_true",
                        help="skip the day loop; just re-aggregate existing checkpoints into the "
                             "report (safe to run while the per-series workers are still going)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None

    def _floats(s):
        return [float(x) for x in s.split(",")] if s else None

    print(f"[backtest_maker_core] telonex={paths.telonex_dir} "
          f"out={paths.out_dir / 'backtests' / args.out_name} seeds={args.seeds}", flush=True)
    result = backtest_maker_core(paths, series=series, since_ms=since_ms, until_ms=until_ms,
                                 defend_thetas=_floats(args.defend_thetas),
                                 lean_mults=_floats(args.lean_mults), lean_theta=args.lean_theta,
                                 seeds=args.seeds, seed=args.seed, maker_side=args.maker_side,
                                 target=args.target, variant=args.variant, out_name=args.out_name,
                                 finalize_only=args.finalize_only)
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
