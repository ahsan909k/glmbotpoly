"""Challenger bake-off — MLP / XGBoost / logistic-regression vs the LightGBM champion.

Answers one question: does a different *model class* produce a better 15-second-direction signal than
the champion (``accuracy_push``, LightGBM), enough to change the burst-pair money verdict? The
comparison is apples-to-apples — **everything is held fixed except the model**:

- Same methodology as ``accuracy_push`` (train on pre-regime, predict the current-regime windows
  out-of-sample, purged at the boundary; screen the price-only vs order-flow-microstructure variant on
  a held-out pre-regime tail; refit the winner on all pre-regime). The leakage-critical split and
  train-selection helpers are imported from :mod:`accuracy_push` verbatim so they can never drift; the
  GBT backend reproduces ``accuracy_push.run_market`` byte-for-byte (the anti-drift tripwire test).
- Same θ_85 gate, same OOS parquet schema — so :mod:`backtest_burst_pair` money-scores every model via
  ``--oos-dir bakeoff/<model>`` with no change.
- Two shuffled-label controls: the **model-level** control here (train on permuted labels → the OOS
  predictions must collapse to chance) proves each challenger isn't leaking; the **money-level**
  control (per-(series, day) outcome shuffle) is applied by ``backtest_burst_pair``.

Two subcommands::

    python -m model_lab.challenger_bakeoff train --backend {gbt,xgb,mlp,logreg} [--days N] [--series ...]
    python -m model_lab.challenger_bakeoff aggregate

``train`` writes ``out/bakeoff/<backend>/{oos_<market>.parquet, metrics.json, frontier.csv,
report.html}``; ``aggregate`` reads every model's model-level metrics + money ``metrics.json`` and
emits the one-page verdict ``out/bakeoff/{verdict.html, verdict.md, summary.csv}``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import accuracy_push as ap
from . import config
from . import eval_harness as eh
from .config import Paths
from .lib import backends as bk
from .lib import math as lm
from .lib import procmem

#: the champion's best-2 burst cells by net PnL (backtest_burst_pair sweep); money headline.
MONEY_CELLS = ("C0.94_U20", "C0.98_U20")
CHAMPION = "gbt"
ALL_MODELS = ("gbt", "xgb", "mlp", "logreg")


# ===========================================================================
# training (per backend, accuracy_push methodology, model seam swapped)
# ===========================================================================
def _fit_predict_backend(sub: pd.DataFrame, feats: list[str], train_m: np.ndarray,
                         pred_m: np.ndarray, backend: bk.Backend, *, seed: int, threads: int):
    """The exact ``accuracy_push._fit_predict`` shape, but the model comes from ``backend``. Reuses
    ``accuracy_push.capped_train_idx`` / ``inner_val_split`` so the train selection is identical for
    every backend (and byte-identical to ``accuracy_push`` for the GBT backend)."""
    x = sub[feats].to_numpy(np.float32)
    y = sub[ap.LABEL].to_numpy(np.float64)
    ts = sub["sample_ts_ms"].to_numpy(np.int64)
    tr_idx = ap.capped_train_idx(train_m, y, ts, seed)
    inner_tr, inner_val = ap.inner_val_split(tr_idx)
    fitted = backend.fit(x[inner_tr], y[inner_tr], x[inner_val], y[inner_val], seed=seed,
                         threads=threads)
    prob = fitted.predict_proba(x[np.flatnonzero(pred_m)], threads=threads)
    imp = backend.feature_importance(fitted, feats)
    return prob, fitted, imp


def _ece(reliability: list[dict]) -> float:
    """Expected calibration error = Σ (n_bin/N)·|mean_pred − empirical_rate| over the reliability bins."""
    n = sum(b["n"] for b in reliability)
    if not n:
        return float("nan")
    return float(sum(b["n"] / n * abs(b["mean_pred"] - b["empirical_rate"]) for b in reliability))


def run_market_backend(paths: Paths, market: str, backend: bk.Backend, *, days: int | None,
                       seed: int, threads: int, out_dir: Path, regime_from, run_shuffle: bool) -> dict:
    """Train ``backend`` with ``accuracy_push``'s methodology on one market, predict the regime OOS,
    persist the OOS + winner model, and compute the model-level metrics + the shuffled-label control."""
    t0 = time.time()
    df = ap.load_market(paths, market, days=days)
    floor = ap._regime_floor_ms(regime_from)
    win_open = df["window_open_ms"].to_numpy(np.int64)
    ts = df["sample_ts_ms"].to_numpy(np.int64)
    masks = ap.build_split_masks(win_open, ts, floor)
    regime_m, val_m = masks["regime_m"], masks["val_m"]
    sel_train_m, dep_train_m = masks["sel_train_m"], masks["dep_train_m"]

    n_regime = int(regime_m.sum())
    n_regime_win = int(pd.unique(win_open[regime_m]).size)
    if int(sel_train_m.sum()) < 5000 or n_regime < 1000:
        return {"market": market, "error": "insufficient rows",
                "n_train": int(sel_train_m.sum()), "n_regime": n_regime}

    # --- variant screening on the pre-regime val tail --------------------
    variant_metrics: dict = {}
    for name, feats in ap.VARIANTS.items():
        prob_val, _f, _i = _fit_predict_backend(df, feats, sel_train_m, val_m, backend,
                                                seed=seed, threads=threads)
        y_val = df[ap.LABEL].to_numpy(np.float64)[np.flatnonzero(val_m)]
        ok = np.isfinite(prob_val) & np.isfinite(y_val)
        variant_metrics[name] = {
            "diracc": lm.directional_accuracy(prob_val[ok], y_val[ok]),
            "brier": lm.brier_score(prob_val[ok], y_val[ok]),
            "n_val": int(ok.sum()),
            "frontier": ap.abstention_curve(prob_val, y_val, ap.REPORT_THETAS),
            "theta_85_val": ap.theta_at_accuracy(prob_val, y_val)["theta_85"],
        }
    winner = max(ap.VARIANTS, key=lambda v: variant_metrics[v]["diracc"])
    feats = ap.VARIANTS[winner]

    # --- deployed model: refit winner on all pre-regime, predict regime OOS ----
    prob_regime, fitted, imp = _fit_predict_backend(df, feats, dep_train_m, regime_m, backend,
                                                    seed=seed, threads=threads)
    backend.save(fitted, out_dir / f"model_{market}_{winner}{backend.model_ext}")

    reg_idx = np.flatnonzero(regime_m)
    oos = pd.DataFrame({
        "series": market,
        "window_open_ms": win_open[reg_idx],
        "sample_ts_ms": ts[reg_idx],
        "p_up": prob_regime.astype(np.float64),
        "y_true": df[ap.LABEL].to_numpy(np.float64)[reg_idx],
        "outcome_up": df["outcome_up"].to_numpy()[reg_idx],
        "tau_secs": df["tau_secs"].to_numpy(np.float64)[reg_idx],
        "label_source": df["label_source"].to_numpy()[reg_idx],
    })
    oos.to_parquet(out_dir / f"oos_{market}.parquet", index=False)

    # --- theta_85 + coverage on the regime OOS (identical to accuracy_push) ----
    y_reg = oos["y_true"].to_numpy(np.float64)
    gate = ap.theta_at_accuracy(prob_regime, y_reg)
    theta_85 = gate["theta_85"]
    reached = theta_85 is not None
    theta_gate = theta_85["theta"] if reached else gate["best_gate"]["theta"]
    qual = np.abs(prob_regime - 0.5) >= theta_gate
    coverage_frac = float(qual.mean()) if len(qual) else float("nan")
    signals_per_window = (int(qual.sum()) / n_regime_win) if n_regime_win else float("nan")

    # --- model-level OOS metrics: accuracy / lift / Brier / log-loss / calibration ----
    ok = np.isfinite(prob_regime) & np.isfinite(y_reg)
    base = lm.majority_baseline(y_reg[ok])
    reliability = lm.reliability_table(prob_regime[ok], y_reg[ok], n_bins=10)
    oos_diracc = lm.directional_accuracy(prob_regime[ok], y_reg[ok])
    oos_brier = lm.brier_score(prob_regime[ok], y_reg[ok])

    # --- model-level shuffled-label control: train on permuted labels → collapse to chance ----
    shuffled_control = None
    if run_shuffle:
        df_shuf = df.copy()
        df_shuf[ap.LABEL] = np.random.default_rng(seed + 1).permutation(
            df[ap.LABEL].to_numpy(np.float64))
        prob_sh, _f, _i = _fit_predict_backend(df_shuf, feats, dep_train_m, regime_m, backend,
                                               seed=seed, threads=threads)
        oks = np.isfinite(prob_sh) & np.isfinite(y_reg)
        sh_brier = lm.brier_score(prob_sh[oks], y_reg[oks])
        sh_diracc = lm.directional_accuracy(prob_sh[oks], y_reg[oks])
        collapsed = bool(sh_brier >= base["brier"] - 0.03 and oos_brier < sh_brier)
        shuffled_control = {"oos_diracc": sh_diracc, "oos_brier": sh_brier,
                            "chance_brier": base["brier"], "collapsed": collapsed}

    top = None
    if imp is not None:
        gain = imp["gain"]
        tot = sum(gain.values()) or 1.0
        top = [[n, round(100.0 * v / tot, 1)]
               for n, v in sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:8]]

    return {
        "market": market,
        "elapsed_secs": round(time.time() - t0, 1),
        "winner": winner,
        "flow_won": winner == "flow",
        "variant_metrics": variant_metrics,
        "n_regime_rows": n_regime,
        "n_regime_windows": n_regime_win,
        "n_labeled": int(ok.sum()),
        "top_features": top,
        "oos_diracc": oos_diracc,
        "oos_brier": oos_brier,
        "oos_logloss": lm.log_loss(prob_regime[ok], y_reg[ok]),
        "base_rate_acc": base["diracc"],
        "lift_diracc": oos_diracc - base["diracc"],
        "oos_ece": _ece(reliability),
        "reliability": reliability,
        "theta_85": theta_85,
        "theta_85_reached": bool(reached),
        "theta_gate": float(theta_gate),
        "best_gate": gate["best_gate"],
        "regime_frontier": ap.abstention_curve(prob_regime, y_reg, ap.REPORT_THETAS),
        "coverage_frac_at_gate": coverage_frac,
        "avg_signals_per_window": signals_per_window,
        "n_qualifying": int(qual.sum()),
        "shuffled_control": shuffled_control,
    }


def run_backend(paths: Paths, backend_name: str, *, markets=ap.MARKETS, days: int | None = None,
                seed: int = ap.DEFAULT_SEED, threads: int = 1, regime_from=ap.REGIME_FROM,
                run_shuffle: bool = True, resume: bool = True) -> dict:
    """Train ``backend_name`` on every market and write the ``accuracy_push``-layout OOS + metrics.
    Resumable: a market whose ``oos_<market>.parquet`` already exists is reloaded, not retrained."""
    backend = bk.BACKENDS[backend_name]
    paths.ensure_out()
    out_dir = paths.out_dir / "bakeoff" / backend_name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_markets(out_dir) if resume else {}
    results: dict = {"config": {"backend": backend_name, "horizon_secs": ap.HORIZON_SECS, "seed": seed,
                                "threads": threads, "days": days, "val_days": ap.VAL_DAYS,
                                "train_cap": ap.TRAIN_CAP, "regime_from": regime_from.isoformat(),
                                "run_shuffle": run_shuffle,
                                "price_features": ap.PRICE_FEATURES, "flow_features": ap.FLOW_FEATURES},
                     "markets": {}}
    for market in markets:
        oos_path = out_dir / f"oos_{market}.parquet"
        if resume and oos_path.exists() and market in existing:
            results["markets"][market] = existing[market]
            print(f"[bakeoff:{backend_name}] {market}: resumed (oos exists)", flush=True)
            ap._write(out_dir, results)
            continue
        print(f"[bakeoff:{backend_name}] === {market} ===", flush=True)
        r = run_market_backend(paths, market, backend, days=days, seed=seed, threads=threads,
                               out_dir=out_dir, regime_from=regime_from, run_shuffle=run_shuffle)
        results["markets"][market] = r
        if "error" in r:
            print(f"[bakeoff:{backend_name}]   {market}: SKIPPED ({r['error']})", flush=True)
        else:
            sc = r["shuffled_control"]
            print(f"[bakeoff:{backend_name}]   winner={r['winner']} OOS diracc={r['oos_diracc']:.4f} "
                  f"(lift {r['lift_diracc']:+.4f}) Brier={r['oos_brier']:.4f} θ_gate={r['theta_gate']:.3f} "
                  f"cov={r['coverage_frac_at_gate']:.4f} sig/win={r['avg_signals_per_window']:.3f} "
                  f"shuffle_collapsed={sc['collapsed'] if sc else 'n/a'} ({r['elapsed_secs']:.0f}s)",
                  flush=True)
        ap._write(out_dir, results)
    ap._write(out_dir, results)
    ap._write_report(out_dir, results)
    return results


def _load_existing_markets(out_dir: Path) -> dict:
    path = out_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("markets", {})
    except (OSError, json.JSONDecodeError):
        return {}


# ===========================================================================
# aggregation + verdict
# ===========================================================================
def _read_model_metrics(paths: Paths, model: str) -> dict | None:
    path = paths.out_dir / "bakeoff" / model / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_money_metrics(paths: Paths, model: str, money_prefix: str) -> dict | None:
    path = paths.out_dir / "backtests" / f"{money_prefix}_{model}" / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _wmean(markets: list[dict], key: str, weight: str = "n_labeled") -> float:
    """n_labeled-weighted mean of a per-market scalar (pooled OOS estimate across the 4 markets)."""
    num = den = 0.0
    for r in markets:
        v, w = r.get(key), r.get(weight, 0)
        if v is not None and np.isfinite(v) and w:
            num += v * w
            den += w
    return (num / den) if den else float("nan")


def _summarize_model(model: str, mm: dict | None, money: dict | None) -> dict:
    row: dict = {"model": model}
    if mm:
        markets = [r for r in mm.get("markets", {}).values() if "error" not in r]
        row["n_markets"] = len(markets)
        row["oos_diracc"] = _wmean(markets, "oos_diracc")
        row["oos_brier"] = _wmean(markets, "oos_brier")
        row["oos_logloss"] = _wmean(markets, "oos_logloss")
        row["lift_diracc"] = _wmean(markets, "lift_diracc")
        row["oos_ece"] = _wmean(markets, "oos_ece")
        row["coverage_at_gate"] = float(np.mean([r["coverage_frac_at_gate"] for r in markets])) \
            if markets else float("nan")
        row["signals_per_window"] = float(np.mean([r["avg_signals_per_window"] for r in markets])) \
            if markets else float("nan")
        row["theta85_reached"] = f"{sum(1 for r in markets if r.get('theta_85_reached'))}/{len(markets)}"
        scs = [r.get("shuffled_control") for r in markets]
        row["shuffle_collapsed"] = all(sc and sc.get("collapsed") for sc in scs) if scs else None
    if money:
        cells = money.get("cells", {})
        for cell in MONEY_CELLS:
            c = cells.get(cell, {})
            row[f"net_{cell}"] = c.get("net_pnl")
            row[f"shuf_{cell}"] = c.get("shuffled_net")
        row["money_best_cell"] = money.get("best_cell")
        row["money_best_net"] = (cells.get(money.get("best_cell", ""), {}) or {}).get("net_pnl")
    return row


def _beats_champion(r: dict, champ: dict) -> tuple[bool, bool, str]:
    """Two-part gate. (a) model: Δ OOS diracc ≥ +1 pp AND Δ Brier < 0. (b) money: net clears the
    ±|shuffled_net| band on BOTH best-2 cells. Returns (part_a, part_b, verdict phrase)."""
    d_dir = (r.get("oos_diracc", float("nan")) - champ.get("oos_diracc", float("nan"))) * 100.0
    d_brier = r.get("oos_brier", float("nan")) - champ.get("oos_brier", float("nan"))
    part_a = bool(np.isfinite(d_dir) and np.isfinite(d_brier) and d_dir >= 1.0 and d_brier < 0.0)
    money_known = all(r.get(f"net_{c}") is not None and r.get(f"shuf_{c}") is not None
                      for c in MONEY_CELLS)
    part_b = money_known and all(abs(r[f"net_{c}"]) > abs(r[f"shuf_{c}"]) and r[f"net_{c}"] > 0
                                 for c in MONEY_CELLS)
    if part_a and part_b:
        phrase = "BEATS GBT (better model AND money clears the noise floor on both cells)"
    elif part_a:
        phrase = "better model, money-inconclusive (Δdiracc≥+1pp but money inside the shuffled band)"
    elif not money_known:
        phrase = "does not beat GBT on the model metrics (money pending)"
    else:
        phrase = "does not beat GBT"
    return part_a, part_b, phrase


def aggregate(paths: Paths, *, models=ALL_MODELS, money_prefix: str = "bakeoff") -> dict:
    """Read every model's model-level + money metrics; emit the one-page verdict."""
    rows = [_summarize_model(m, _read_model_metrics(paths, m), _read_money_metrics(paths, m, money_prefix))
            for m in models]
    champ = next((r for r in rows if r["model"] == CHAMPION), None)
    verdicts: dict = {}
    if champ and "oos_diracc" in champ:
        for r in rows:
            if r["model"] == CHAMPION:
                continue
            if "oos_diracc" not in r:
                verdicts[r["model"]] = "not trained yet"
                continue
            _a, _b, phrase = _beats_champion(r, champ)
            verdicts[r["model"]] = phrase
    any_beats = any("BEATS GBT" in v for v in verdicts.values())
    headline = ("At least one challenger beats GBT by enough to matter." if any_beats else
                "Nothing beats GBT by enough to matter — the LightGBM champion remains the pick.")
    result = {"rows": rows, "champion": CHAMPION, "money_cells": list(MONEY_CELLS),
              "verdicts": verdicts, "headline": headline}
    out_dir = paths.out_dir / "bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_bakeoff_outputs(out_dir, result)
    return result


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _write_bakeoff_outputs(out_dir: Path, result: dict) -> None:
    rows = result["rows"]
    # summary.csv
    csv_cols = ["model", "oos_diracc", "lift_diracc", "oos_brier", "oos_logloss", "oos_ece",
                "coverage_at_gate", "signals_per_window", "theta85_reached", "shuffle_collapsed",
                f"net_{MONEY_CELLS[0]}", f"shuf_{MONEY_CELLS[0]}",
                f"net_{MONEY_CELLS[1]}", f"shuf_{MONEY_CELLS[1]}", "money_best_cell", "money_best_net"]
    pd.DataFrame([{c: r.get(c) for c in csv_cols} for r in rows]).to_csv(out_dir / "summary.csv",
                                                                        index=False)
    # verdict.md
    md = [f"# Challenger bake-off — verdict\n", f"**{result['headline']}**\n",
          f"Champion = `{CHAMPION}` (LightGBM). Money = net PnL on the champion's best-2 burst cells "
          f"{MONEY_CELLS}, vs each cell's shuffled-outcome control (the money noise floor). "
          f"Model metrics are on the 15 s forward-direction label over the current-regime OOS.\n",
          "| model | OOS diracc | lift vs base | Brier | logloss | ECE | cov@θ85 | sig/win | "
          "θ85 reached | shuffle collapsed | "
          f"net {MONEY_CELLS[0]} | shuf | net {MONEY_CELLS[1]} | shuf | own best |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join([
            r["model"], _fmt(r.get("oos_diracc")), _fmt(r.get("lift_diracc")), _fmt(r.get("oos_brier")),
            _fmt(r.get("oos_logloss")), _fmt(r.get("oos_ece"), 4), _fmt(r.get("coverage_at_gate")),
            _fmt(r.get("signals_per_window"), 3), _fmt(r.get("theta85_reached")),
            _fmt(r.get("shuffle_collapsed")), _fmt(r.get(f"net_{MONEY_CELLS[0]}"), 2),
            _fmt(r.get(f"shuf_{MONEY_CELLS[0]}"), 2), _fmt(r.get(f"net_{MONEY_CELLS[1]}"), 2),
            _fmt(r.get(f"shuf_{MONEY_CELLS[1]}"), 2),
            f"{_fmt(r.get('money_best_cell'))} ({_fmt(r.get('money_best_net'), 0)})"]) + " |")
    md.append("\n## Per-challenger verdict (two-part gate)\n")
    md.append("A challenger *beats GBT* only if (a) Δ OOS diracc ≥ +1.0 pp **and** Δ Brier < 0, **and** "
              "(b) its money net clears the ±|shuffled_net| band on **both** best-2 cells.\n")
    for m, v in result["verdicts"].items():
        md.append(f"- **{m}**: {v}")
    (out_dir / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # verdict.html
    tbl_cols = [("model", "model"), ("oos_diracc", "OOS diracc"), ("lift_diracc", "lift"),
                ("oos_brier", "Brier"), ("oos_logloss", "logloss"), ("oos_ece", "ECE"),
                ("coverage_at_gate", "cov@θ85"), ("signals_per_window", "sig/win"),
                ("theta85_reached", "θ85"), ("shuffle_collapsed", "shuffle✗"),
                (f"net_{MONEY_CELLS[0]}", f"net {MONEY_CELLS[0]}"),
                (f"net_{MONEY_CELLS[1]}", f"net {MONEY_CELLS[1]}"),
                ("money_best_cell", "own best")]
    verdict_items = "".join(f"<li><b>{m}</b>: {v}</li>" for m, v in result["verdicts"].items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Challenger bake-off</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1080px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.5rem}} h2{{margin-top:1.6rem;border-bottom:1px solid #eee}}
table{{border-collapse:collapse;font-size:.9rem;margin:.6rem 0}} td,th{{padding:4px 10px;text-align:left;border-bottom:1px solid #f0f0f0}}
th{{color:#555}} .muted{{color:#888}}
.head{{background:#f6f8fa;border-left:4px solid #1f77b4;padding:.8rem 1rem;border-radius:4px;font-weight:600}}</style></head><body>
<h1>Challenger bake-off — MLP / XGBoost / LogReg vs the LightGBM champion</h1>
<div class="head">{result['headline']}</div>
<p class="muted">Champion = <code>{CHAMPION}</code>. Everything held fixed except the model
(accuracy_push's static pre-regime→current-regime split, same θ_85 gate, same burst money scoring).
Money columns = net PnL on the champion's best-2 cells {MONEY_CELLS}; shuffle✗ = the model-level
shuffled-label control collapsed to chance (no leakage).</p>
{eh._table_html(rows, tbl_cols)}
<h2>Per-challenger verdict (two-part gate)</h2>
<p class="muted">Beats GBT iff (a) Δ OOS diracc ≥ +1.0 pp AND Δ Brier &lt; 0, AND (b) money net clears
±|shuffled_net| on both best-2 cells. If only (a): "better model, money-inconclusive".</p>
<ul>{verdict_items}</ul>
</body></html>"""
    (out_dir / "verdict.html").write_text(html, encoding="utf-8")


# ===========================================================================
# CLI
# ===========================================================================
def _main_train(argv: list[str]) -> int:
    ap_ = config.stage_parser("challenger_bakeoff train")
    ap_.add_argument("--backend", required=True, choices=list(ALL_MODELS))
    ap_.add_argument("--series", default=None, help="comma-separated markets (default all 4)")
    ap_.add_argument("--days", type=int, default=0, help="trailing window-days (0 = all history)")
    ap_.add_argument("--seed", type=int, default=ap.DEFAULT_SEED)
    ap_.add_argument("--threads", type=int, default=1,
                     help="model threads (1 = byte-reproducible for every backend)")
    ap_.add_argument("--no-shuffle-control", action="store_true",
                     help="skip the model-level shuffled-label control (one fewer fit per market)")
    ap_.add_argument("--no-resume", action="store_true", help="retrain even if oos_<market>.parquet exists")
    args = ap_.parse_args(argv)
    paths = config.resolve_paths(args)
    markets = tuple(s.strip() for s in args.series.split(",")) if args.series else ap.MARKETS
    print(f"[bakeoff] train backend={args.backend} markets={markets} days={args.days or 'all'} "
          f"threads={args.threads} out={paths.out_dir / 'bakeoff' / args.backend}", flush=True)
    run_backend(paths, args.backend, markets=markets, days=(args.days or None), seed=args.seed,
                threads=args.threads, run_shuffle=not args.no_shuffle_control, resume=not args.no_resume)
    procmem.report_peak_rss(f"bakeoff:{args.backend}", procmem.peak_rss_mb(),
                            getattr(args, "max_rss_mb", config.DEFAULT_MAX_RSS_MB))
    return 0


def _main_aggregate(argv: list[str]) -> int:
    ap_ = config.stage_parser("challenger_bakeoff aggregate")
    ap_.add_argument("--models", default=",".join(ALL_MODELS), help="comma-separated models to compare")
    ap_.add_argument("--money-prefix", default="bakeoff",
                     help="money runs live in out/backtests/<prefix>_<model>/")
    args = ap_.parse_args(argv)
    paths = config.resolve_paths(args)
    models = tuple(s.strip() for s in args.models.split(","))
    res = aggregate(paths, models=models, money_prefix=args.money_prefix)
    print(f"[bakeoff] VERDICT: {res['headline']}", flush=True)
    for m, v in res["verdicts"].items():
        print(f"[bakeoff]   {m}: {v}", flush=True)
    print(f"[bakeoff] wrote verdict.html + verdict.md + summary.csv to {paths.out_dir / 'bakeoff'}",
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        return _main_train(argv[1:])
    if argv and argv[0] == "aggregate":
        return _main_aggregate(argv[1:])
    print("usage: python -m model_lab.challenger_bakeoff {train --backend NAME | aggregate} [...]",
          flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
