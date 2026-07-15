"""Stage — accuracy_push: the **per-market 15-second model** the burst-pair strategy runs on.

The champion (``model_dir10_full``) is a single *pooled* 10-second model. The operator wants,
**per market**, the best **15-second** forward-direction model and — from its confidence frontier
— the gate ``theta_85`` at which held-out accuracy reaches 85 %. This stage produces exactly that,
plus the per-market out-of-sample (OOS) predictions over the current-regime windows that downstream
strategy tests (``backtest_burst_pair``) consume.

For each of the four markets (BTC/ETH x 5m/15m) it trains the same champion LightGBM
(``lib/gbt`` defaults) on ``historical_dataset.parquet`` and screens **two feature sets**:

* **price** — price/vol/formula/basis only (10 features): the "no order-flow" baseline.
* **flow** — price + the order-flow **microstructure** block (depth imbalance/microprice/slopes/
  spread + Polymarket book + staleness, 14 more = the champion's 24): "with order-flow".

  Note: the four-series ``historical_dataset`` carries **no signed-aggressor-flow** columns
  (``flow_imb_*``/``trade_intensity_*`` live only in the 5m-only ``feature_set.parquet``), so
  "flow features" here means the depth/PM **order-flow microstructure** block — the closest
  order-flow signal available for all four markets. The winner is picked per market on a held-out
  pre-regime tail ("flow if it wins; else the best available").

**Leakage-safe, single deployable model per market** (not a walk-forward ensemble — the operator
asked for *the best 15 s model per market*, one model, like the champion):

* ``regime`` = windows with ``window_open_ms >= 2026-06-05`` — the traded set, **held out** for
  prediction. Its 255 ms taker-delay regime is where the strategy trades.
* ``pre`` = everything before. Its last ``VAL_DAYS`` (21 d) = a **validation tail** used only to
  pick the winning variant + as a ``theta_85`` cross-check. The rest = train (early-stopped on its
  own inner-val), purged of any row whose 15 s label peeks across the val boundary.
* The winning variant is **refit on ALL pre-regime** (purged at the regime boundary) for the round
  count chosen on the val tail, then predicts the regime OOS — the freshest possible model that
  still never sees a regime row.

``theta_85`` is the smallest confidence gate ``|p-0.5| >= theta`` at which **held-out** directional
accuracy reaches 85 % on the regime OOS (>= ``MIN_SIGNALS`` samples). It is reported alongside the
val-tail ``theta_85`` (a leakage-clean cross-check), the coverage, and the average qualifying
signals per window per series — the operator's requested screening numbers. If 85 % is never
reached, ``theta_85`` is ``None`` and ``theta_gate`` falls back to the most-accurate gate (flagged),
so the strategy can still run (and honestly report the shortfall).

Outputs ``out/accuracy_push/``: ``oos_<market>.parquet`` (regime OOS predictions the strategy
consumes), ``model_<market>_<variant>.txt`` (both variants, for provenance), ``metrics.json``,
``frontier.csv``, ``report.html``. Opt-in LightGBM (``pip install -e .[gbt]``). Research only.

Run::

    python -m model_lab.accuracy_push                 # all 4 markets, full history
    python -m model_lab.accuracy_push --days 120      # trailing 120 window-days (fast smoke)
    python -m model_lab.accuracy_push --series BTC-5m  # one market
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import eval_harness as eh
from .config import Paths
from .lib import math as lm
from .lib import procmem

MS_PER_DAY = 86_400_000
HORIZON_SECS = 15
LABEL = "fwd_up_15s"
REGIME_FROM = date(2026, 6, 5)  # 250 ms taker-delay lock (out/audits/venue_regime_timeline.md)
MARKETS = ("BTC-5m", "ETH-5m", "BTC-15m", "ETH-15m")
DEFAULT_SEED = 20260713
VAL_DAYS = 21  # held-out pre-regime tail (variant selection + theta_85 cross-check)
TRAIN_CAP = 1_200_000  # max train rows per fit (seeded subsample; bounds memory + time)
MIN_SIGNALS = 50  # a theta gate needs >= this many samples to be trusted
MAX_ROUNDS = 400

#: price/vol/formula/basis features — the "no order-flow" baseline.
PRICE_FEATURES = ["ret", "realized_vol", "sigma_1s", "log_s_k", "z", "p_up_model",
                  "tau_secs", "elapsed_secs", "basis_bps", "basis_ewma"]
#: order-flow microstructure block (depth + Polymarket book) added by the "flow" variant.
FLOW_FEATURES = ["depth_imb_1", "depth_imb_5", "depth_imb_10", "depth_imb_20",
                 "microprice_gap", "bid_depth_slope", "ask_depth_slope", "depth_spread_bps",
                 "pm_mid", "pm_spread", "pm_book_imb",
                 "pm_staleness_1s", "pm_staleness_2s", "pm_staleness_3s"]
VARIANTS = {"price": PRICE_FEATURES, "flow": PRICE_FEATURES + FLOW_FEATURES}
ALL_FEATURES = PRICE_FEATURES + FLOW_FEATURES

_KEY_COLS = ["series", "window_open_ms", "sample_ts_ms", "tau_secs", "outcome_up", "label_source"]
#: theta grid for the confidence frontier (fine so theta_85 is located precisely).
FRONTIER_THETAS = [round(t, 3) for t in np.arange(0.0, 0.50, 0.005)]
#: coarse thetas shown in the report / metrics.
REPORT_THETAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _regime_floor_ms(regime_from: date) -> int:
    return int(datetime(regime_from.year, regime_from.month, regime_from.day,
                        tzinfo=timezone.utc).timestamp()) * 1000


def build_split_masks(win_open: np.ndarray, ts: np.ndarray, floor: int,
                      val_days: int = VAL_DAYS) -> dict:
    """The leakage-safe pre-regime / val-tail / current-regime-OOS split masks.

    Single source of truth, shared verbatim with the challenger bake-off so the purge logic can
    never drift. ``label_info = ts + 15 s`` is when the forward label becomes known; the
    ``label_info < boundary`` terms purge any row whose label would peek across a split boundary.
    Returns ``regime_m`` (current-regime OOS), ``pre_m`` (all pre-regime), ``val_m`` (pre-regime val
    tail for variant selection), ``sel_train_m`` (variant-selection train, purged across the val
    boundary) and ``dep_train_m`` (deployed-model train = all pre-regime, purged across the regime
    boundary).
    """
    label_info = ts + HORIZON_SECS * 1000  # when the 15 s label is known
    regime_m = win_open >= floor
    pre_m = win_open < floor
    val_start = floor - val_days * MS_PER_DAY
    val_m = pre_m & (win_open >= val_start)
    sel_train_m = pre_m & (win_open < val_start) & (label_info < val_start)
    dep_train_m = pre_m & (label_info < floor)
    return {"regime_m": regime_m, "pre_m": pre_m, "val_m": val_m,
            "sel_train_m": sel_train_m, "dep_train_m": dep_train_m,
            "val_start": int(val_start), "label_info": label_info}


# ===========================================================================
# data
# ===========================================================================
def load_market(paths: Paths, market: str, *, days: int | None) -> pd.DataFrame:
    """Load one market's rows (keys + 24 features + label) from ``historical_dataset.parquet``.
    Filtering to a single series keeps peak memory well under the ceiling. ``days`` restricts to
    the trailing window-days (for a fast smoke)."""
    import pyarrow.parquet as pq

    path = paths.out_dir / "historical_dataset.parquet"
    config.assert_parquet_ready(path, label="historical_dataset.parquet", min_rows=1)
    read_cols = list(dict.fromkeys(_KEY_COLS + [LABEL] + ALL_FEATURES))
    filters = [("series", "==", market)]
    if days and days > 0:
        opens = pq.read_table(path, columns=["window_open_ms"]).column(0).to_numpy()
        cutoff = int(opens.max()) - int(days) * MS_PER_DAY
        filters.append(("window_open_ms", ">=", int(cutoff)))
    df = pq.read_table(path, columns=read_cols, filters=filters).to_pandas()
    for c in ALL_FEATURES:
        df[c] = df[c].astype(np.float32)
    df = df.sort_values(["window_open_ms", "sample_ts_ms"]).reset_index(drop=True)
    return df


# ===========================================================================
# metrics
# ===========================================================================
def abstention_curve(prob: np.ndarray, y: np.ndarray, thetas: list[float]) -> list[dict]:
    """Accuracy + coverage on the confident subset ``|p-0.5| >= theta`` (rows with a valid label)."""
    p, y = np.asarray(prob, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    correct = (p >= 0.5).astype(float) == y
    n = len(y)
    out = []
    for t in thetas:
        m = np.abs(p - 0.5) >= t
        cov = int(m.sum())
        out.append({"theta": float(t), "coverage": cov,
                    "coverage_frac": (cov / n) if n else float("nan"),
                    "accuracy": float(correct[m].mean()) if cov else float("nan")})
    return out


def theta_at_accuracy(prob: np.ndarray, y: np.ndarray, target: float = 0.85,
                      min_signals: int = MIN_SIGNALS) -> dict:
    """Smallest ``theta`` (on ``FRONTIER_THETAS``) where held-out accuracy on ``|p-0.5| >= theta``
    reaches ``target`` with ``>= min_signals`` samples. Returns the gate + its coverage/accuracy,
    plus the most-accurate reachable gate as a fallback."""
    p, y = np.asarray(prob, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = len(y)
    correct = (p >= 0.5).astype(float) == y
    theta_85 = None
    best = {"theta": 0.0, "accuracy": float(correct.mean()) if n else float("nan"),
            "coverage": n, "coverage_frac": 1.0 if n else float("nan")}
    for t in FRONTIER_THETAS:
        m = np.abs(p - 0.5) >= t
        cov = int(m.sum())
        if cov < min_signals:
            continue
        acc = float(correct[m].mean())
        if acc > best["accuracy"]:
            best = {"theta": float(t), "accuracy": acc, "coverage": cov,
                    "coverage_frac": cov / n if n else float("nan")}
        if theta_85 is None and acc >= target:
            theta_85 = {"theta": float(t), "accuracy": acc, "coverage": cov,
                        "coverage_frac": cov / n if n else float("nan")}
    return {"theta_85": theta_85, "best_gate": best, "n_labeled": n}


# ===========================================================================
# training
# ===========================================================================
def capped_train_idx(train_m: np.ndarray, y: np.ndarray, ts: np.ndarray, seed: int) -> np.ndarray:
    """Finite-label rows of ``train_m``, capped to ``TRAIN_CAP`` by a seeded subsample and re-sorted
    by ``ts`` (stable) so the ordering — and therefore the fit — is deterministic. Shared with the
    challenger bake-off so the train selection can never drift."""
    tr_idx = np.flatnonzero(train_m & np.isfinite(y))
    rng = np.random.default_rng(seed)
    if len(tr_idx) > TRAIN_CAP:
        tr_idx = np.sort(rng.choice(tr_idx, size=TRAIN_CAP, replace=False))
    order = np.argsort(ts[tr_idx], kind="stable")
    return tr_idx[order]


def inner_val_split(tr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a ts-ordered train index into ``(inner_train, inner_val)`` — the inner-val is the latest
    ``max(1000, 15 %)`` rows (early stopping / variant selection)."""
    n_val = max(1000, int(0.15 * len(tr_idx)))
    return tr_idx[:-n_val], tr_idx[-n_val:]


def _fit_predict(sub: pd.DataFrame, feats: list[str], train_m: np.ndarray, pred_m: np.ndarray,
                 *, seed: int, threads: int) -> tuple[np.ndarray, int, dict]:
    """Fit the champion GBT on ``train_m`` rows of ``sub`` (early-stopped on the latest 15 % as an
    inner-val), predict the ``pred_m`` rows. Returns ``(prob_on_pred, best_iteration, importances)``.
    Train rows are capped by a seeded subsample; the inner-val is the latest-by-ts slice."""
    from .lib import gbt

    x = sub[feats].to_numpy(np.float32)
    y = sub[LABEL].to_numpy(np.float64)
    ts = sub["sample_ts_ms"].to_numpy(np.int64)
    tr_idx = capped_train_idx(train_m, y, ts, seed)
    inner_tr, inner_val = inner_val_split(tr_idx)

    params = gbt.default_params(seed, num_threads=threads)
    booster, best = gbt.fit(x[inner_tr], y[inner_tr], x[inner_val], y[inner_val],
                            params=params, max_boost_round=MAX_ROUNDS, early_stopping_rounds=40)
    pr_idx = np.flatnonzero(pred_m)
    prob = gbt.predict_proba(booster, x[pr_idx], num_iteration=best, num_threads=threads)
    imp = gbt.feature_importance(booster, feats)["gain"]
    tot = sum(imp.values()) or 1.0
    top = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return prob, int(best), {"booster": booster, "best": int(best),
                             "top_features": [[n, round(100.0 * v / tot, 1)] for n, v in top]}


def run_market(paths: Paths, market: str, *, days: int | None, seed: int, threads: int,
               out_dir: Path, regime_from: date) -> dict:
    """Train both variants, pick the winner on the pre-regime val tail, refit the winner on all
    pre-regime, predict the regime OOS, compute ``theta_85``, and persist the OOS + models."""
    from .lib import gbt

    t0 = time.time()
    df = load_market(paths, market, days=days)
    floor = _regime_floor_ms(regime_from)
    win_open = df["window_open_ms"].to_numpy(np.int64)
    ts = df["sample_ts_ms"].to_numpy(np.int64)

    masks = build_split_masks(win_open, ts, floor)
    regime_m = masks["regime_m"]
    val_m = masks["val_m"]
    sel_train_m = masks["sel_train_m"]  # pre-regime before the val tail, purged across the val boundary
    dep_train_m = masks["dep_train_m"]  # ALL pre-regime, purged across the regime boundary

    n_regime = int(regime_m.sum())
    n_regime_win = int(pd.unique(win_open[regime_m]).size)
    if int(sel_train_m.sum()) < 5000 or n_regime < 1000:
        return {"market": market, "error": "insufficient rows",
                "n_train": int(sel_train_m.sum()), "n_regime": n_regime}

    # --- variant screening on the pre-regime val tail --------------------
    variant_metrics: dict = {}
    for name, feats in VARIANTS.items():
        prob_val, _best, _info = _fit_predict(df, feats, sel_train_m, val_m, seed=seed, threads=threads)
        y_val = df[LABEL].to_numpy(np.float64)[np.flatnonzero(val_m)]
        ok = np.isfinite(prob_val) & np.isfinite(y_val)
        variant_metrics[name] = {
            "diracc": lm.directional_accuracy(prob_val[ok], y_val[ok]),
            "brier": lm.brier_score(prob_val[ok], y_val[ok]),
            "n_val": int(ok.sum()),
            "frontier": abstention_curve(prob_val, y_val, REPORT_THETAS),
            "theta_85_val": theta_at_accuracy(prob_val, y_val)["theta_85"],
        }
    # winner: higher val directional accuracy ("flow if it wins; else best available").
    winner = max(VARIANTS, key=lambda v: variant_metrics[v]["diracc"])
    flow_won = winner == "flow"

    # --- deployed model: refit winner on all pre-regime, predict regime OOS ----
    feats = VARIANTS[winner]
    prob_regime, best_it, dep_info = _fit_predict(df, feats, dep_train_m, regime_m,
                                                  seed=seed, threads=threads)
    gbt.save(dep_info["booster"], out_dir / f"model_{market}_{winner}.txt", num_iteration=best_it)

    reg_idx = np.flatnonzero(regime_m)
    oos = pd.DataFrame({
        "series": market,
        "window_open_ms": win_open[reg_idx],
        "sample_ts_ms": ts[reg_idx],
        "p_up": prob_regime.astype(np.float64),
        "y_true": df[LABEL].to_numpy(np.float64)[reg_idx],
        "outcome_up": df["outcome_up"].to_numpy()[reg_idx],
        "tau_secs": df["tau_secs"].to_numpy(np.float64)[reg_idx],
        "label_source": df["label_source"].to_numpy()[reg_idx],
    })
    oos.to_parquet(out_dir / f"oos_{market}.parquet", index=False)

    # --- theta_85 on the regime OOS (the traded set) ---------------------
    y_reg = oos["y_true"].to_numpy(np.float64)
    gate = theta_at_accuracy(prob_regime, y_reg)
    theta_85 = gate["theta_85"]
    reached = theta_85 is not None
    theta_gate = theta_85["theta"] if reached else gate["best_gate"]["theta"]

    # coverage + avg qualifying signals per window at the chosen gate (on the regime OOS).
    conf = np.abs(prob_regime - 0.5)
    qual = conf >= theta_gate
    coverage_frac = float(qual.mean()) if len(qual) else float("nan")
    signals_per_window = (int(qual.sum()) / n_regime_win) if n_regime_win else float("nan")

    return {
        "market": market,
        "elapsed_secs": round(time.time() - t0, 1),
        "winner": winner,
        "flow_won": bool(flow_won),
        "variant_metrics": variant_metrics,
        "n_regime_rows": n_regime,
        "n_regime_windows": n_regime_win,
        "best_iteration": best_it,
        "top_features": dep_info["top_features"],
        "theta_85": theta_85,          # regime OOS (the traded gate) or None
        "theta_85_reached": bool(reached),
        "theta_gate": float(theta_gate),  # what the strategy uses (theta_85 or best-accuracy fallback)
        "best_gate": gate["best_gate"],
        "regime_frontier": abstention_curve(prob_regime, y_reg, REPORT_THETAS),
        "coverage_frac_at_gate": coverage_frac,
        "avg_signals_per_window": signals_per_window,
        "n_qualifying": int(qual.sum()),
    }


# ===========================================================================
# orchestration
# ===========================================================================
def accuracy_push(paths: Paths, *, markets: tuple[str, ...] = MARKETS, days: int | None = None,
                  seed: int = DEFAULT_SEED, threads: int = 8,
                  regime_from: date = REGIME_FROM) -> dict:
    paths.ensure_out()
    out_dir = paths.out_dir / "accuracy_push"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"config": {"horizon_secs": HORIZON_SECS, "seed": seed, "days": days,
                                "val_days": VAL_DAYS, "train_cap": TRAIN_CAP,
                                "regime_from": regime_from.isoformat(),
                                "price_features": PRICE_FEATURES, "flow_features": FLOW_FEATURES},
                     "markets": {}}
    for market in markets:
        print(f"[accuracy_push] === {market} ===", flush=True)
        r = run_market(paths, market, days=days, seed=seed, threads=threads,
                       out_dir=out_dir, regime_from=regime_from)
        results["markets"][market] = r
        if "error" in r:
            print(f"[accuracy_push]   {market}: SKIPPED ({r['error']})", flush=True)
            _write(out_dir, results)
            continue
        t85 = r["theta_85"]
        t85s = (f"theta_85={t85['theta']:.3f} acc={t85['accuracy']:.3f} cov={t85['coverage_frac']:.4f}"
                if t85 else f"85% NOT reached (best {r['best_gate']['accuracy']:.3f} @ "
                            f"theta={r['best_gate']['theta']:.3f})")
        print(f"[accuracy_push]   winner={r['winner']} (flow_won={r['flow_won']}); {t85s}; "
              f"gate={r['theta_gate']:.3f} cov={r['coverage_frac_at_gate']:.4f} "
              f"signals/win={r['avg_signals_per_window']:.3f} "
              f"({r['n_regime_windows']:,} windows, {r['elapsed_secs']:.0f}s)", flush=True)
        _write(out_dir, results)
    _write(out_dir, results)
    _write_report(out_dir, results)
    return results


def _write(out_dir: Path, results: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1, default=eh._json_default),
                                          encoding="utf-8")
    rows = []
    for market, r in results["markets"].items():
        if "error" in r:
            continue
        t85 = r["theta_85"]
        rows.append({"market": market, "winner": r["winner"], "flow_won": r["flow_won"],
                     "theta_85": (t85["theta"] if t85 else None),
                     "theta_85_acc": (t85["accuracy"] if t85 else None),
                     "theta_gate": r["theta_gate"], "reached_85": r["theta_85_reached"],
                     "coverage_frac": r["coverage_frac_at_gate"],
                     "signals_per_window": r["avg_signals_per_window"],
                     "n_windows": r["n_regime_windows"],
                     "price_diracc": r["variant_metrics"]["price"]["diracc"],
                     "flow_diracc": r["variant_metrics"]["flow"]["diracc"]})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "frontier.csv", index=False)


def _write_report(out_dir: Path, results: dict) -> None:
    rows = []
    for market, r in results["markets"].items():
        if "error" in r:
            rows.append({"market": market, "winner": "—", "theta_85": "—", "acc": "—",
                         "coverage": "—", "signals_per_window": "—", "windows": r.get("n_regime", 0)})
            continue
        t85 = r["theta_85"]
        rows.append({
            "market": market, "winner": r["winner"],
            "theta_85": (f"{t85['theta']:.3f}" if t85 else "NOT reached"),
            "acc": (f"{t85['accuracy']:.1%}" if t85 else f"max {r['best_gate']['accuracy']:.1%}"),
            "coverage": f"{r['coverage_frac_at_gate']:.2%}",
            "signals_per_window": f"{r['avg_signals_per_window']:.3f}",
            "windows": r["n_regime_windows"],
        })
    cols = [("market", "market"), ("winner", "variant"), ("theta_85", "θ_85"), ("acc", "acc@θ_85"),
            ("coverage", "coverage"), ("signals_per_window", "signals/window"), ("windows", "windows")]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>accuracy_push — per-market 15s model</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.4rem}} table{{border-collapse:collapse;font-size:.92rem;margin:.6rem 0}}
td,th{{padding:4px 12px;text-align:left;border-bottom:1px solid #f0f0f0}} th{{color:#555}}
.muted{{color:#888}}</style></head><body>
<h1>accuracy_push — per-market 15-second model</h1>
<p class="muted">Per-market champion GBT on the 15 s forward-direction label, best of the price-only
vs order-flow-microstructure variant (picked on a held-out pre-regime tail). θ_85 = smallest
confidence gate |p−0.5|≥θ where held-out accuracy reaches 85 % on the current-regime OOS
(≥ {MIN_SIGNALS} samples). Coverage + signals/window are on the current-regime OOS (≥ 2026-06-05).</p>
{eh._table_html(rows, cols)}
<p class="muted">A tiny coverage / signals-per-window at θ_85 means an 85 %-accurate 15 s gate is
extremely selective — the burst strategy will fire rarely (signal scarcity, not depth or cap).</p>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = config.stage_parser(__doc__ or "accuracy_push")
    ap.add_argument("--series", default=None, help="comma-separated markets (default all 4)")
    ap.add_argument("--days", type=int, default=0, help="trailing window-days (0 = all history)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    paths = config.resolve_paths(args)
    markets = tuple(s.strip() for s in args.series.split(",")) if args.series else MARKETS

    print(f"[accuracy_push] markets={markets} days={args.days or 'all'} out={paths.out_dir/'accuracy_push'}",
          flush=True)
    res = accuracy_push(paths, markets=markets, days=(args.days or None), seed=args.seed,
                        threads=args.threads)
    procmem.report_peak_rss("accuracy_push", procmem.peak_rss_mb(),
                            getattr(args, "max_rss_mb", config.DEFAULT_MAX_RSS_MB))
    print(f"[accuracy_push] wrote metrics.json + frontier.csv + report.html + oos_<market>.parquet "
          f"to {paths.out_dir/'accuracy_push'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
