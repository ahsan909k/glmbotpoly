"""Stage — compare. One command: logistic vs GBT (×2) vs both benchmarks.

Lines up the two challengers — the logistic regression (:mod:`model_lab.learn`) and
the gradient-boosted trees (:mod:`model_lab.learn_gbt`, plain + formula-anchored
residual) — against the two benchmarks (the formula model and the Polymarket market
mid), on the **outcome** target, across every test period, so "which model is
best, and where" is answered in one place.

The evaluation harness scores **one** model at a time (its canonical frame carries a
single ``model`` column). So this stage builds **one** canonical frame from the GBT
plain grid (which brings in ``outcome_up``, the formula + market benchmarks, and the
day/week/tau buckets), **key-joins** the logistic and residual grids' probabilities
onto it by ``(series, open_time, ts)``, and scores all predictors with the same
metric primitives on the **same rows**. The formula and market benchmarks are
**pairwise-defined** (market only where a book mid exists, formula within its
staleness window), so each is scored on its own defined subset with coverage
reported — there is no single 4-way common row set to claim.

Reads ``out/learn/predictions_outcome.parquet`` and
``out/learn_gbt/predictions_outcome_{plain,residual}.parquet`` — run ``learn`` and
``learn_gbt`` first (or ``run_all --with-gbt``, which produces aligned grids).

Writes ``out/compare/{comparison.csv, metrics.json, report.html}``.

Run::

    python -m model_lab.compare
"""

from __future__ import annotations

import base64
import io
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import eval_harness as eh  # noqa: E402
from .config import ParquetNotReady, Paths, assert_parquet_ready, resolve_paths, stage_parser  # noqa: E402
from .lib import math as lm  # noqa: E402

DEFAULT_MIN_WINDOWS = eh.DEFAULT_MIN_WINDOWS

# Predictor column in the assembled frame → display label + plot colour.
CHALLENGERS = ["logistic", "gbt_plain", "gbt_residual"]
PREDICTORS = CHALLENGERS + ["formula", "market"]
_STYLE = {
    "logistic": ("#9467bd", "logistic"),
    "gbt_plain": ("#1f77b4", "GBT (plain)"),
    "gbt_residual": ("#17becf", "GBT (residual)"),
    "formula": ("#2ca02c", "formula"),
    "market": ("#ff7f0e", "market"),
}


def _load_grid(path, label: str) -> pd.DataFrame:
    assert_parquet_ready(path, label=label, min_rows=1)
    return pd.read_parquet(path)


def _score(df: pd.DataFrame, col: str) -> dict:
    """Brier / log-loss / dir-acc for one predictor column on rows where it (and the
    outcome) are defined."""
    d = df.dropna(subset=[col, "outcome_up"])
    n = int(len(d))
    if n == 0:
        return {"n": 0, "n_windows": 0, "brier": float("nan"),
                "logloss": float("nan"), "diracc": float("nan")}
    p = d[col].to_numpy(dtype=float)
    o = d["outcome_up"].to_numpy(dtype=float)
    nw = int(d.groupby(["series", "open_time"]).ngroups)
    return {"n": n, "n_windows": nw, "brier": lm.brier_score(p, o),
            "logloss": lm.log_loss(p, o), "diracc": lm.directional_accuracy(p, o)}


def _period_rows(df: pd.DataFrame, by: str, total: int, min_windows: int) -> list[dict]:
    rows: list[dict] = []
    for value in sorted(df[by].dropna().unique()):
        sl = df[df[by] == value]
        label = eh._period_label(by, int(value))
        for pred in PREDICTORS:
            s = _score(sl, pred)
            rows.append({"by": by, "period_key": int(value), "period": label, "predictor": pred,
                         "coverage": (s["n"] / len(sl)) if len(sl) else float("nan"),
                         "low_sample": bool(s["n_windows"] < min_windows), **s})
    for pred in PREDICTORS:
        s = _score(df, pred)
        rows.append({"by": by, "period_key": -1, "period": "ALL", "predictor": pred,
                     "coverage": (s["n"] / total) if total else float("nan"),
                     "low_sample": bool(s["n_windows"] < min_windows), **s})
    return rows


def _tau_rows(df: pd.DataFrame, total: int, min_windows: int) -> list[dict]:
    rows: list[dict] = []
    for bucket in eh.TAU_ORDER:
        sl = df[df["tau_bucket"] == bucket]
        if sl.empty:
            continue
        for pred in PREDICTORS:
            s = _score(sl, pred)
            rows.append({"tau_bucket": bucket, "predictor": pred,
                         "coverage": (s["n"] / len(sl)) if len(sl) else float("nan"),
                         "low_sample": bool(s["n_windows"] < min_windows), **s})
    return rows


def _verdict(overall: dict, counts: dict) -> str:
    chall = {k: overall[k]["brier"] for k in CHALLENGERS
             if np.isfinite(overall.get(k, {}).get("brier", float("nan")))}
    if not chall:
        return "No overlapping scored rows across the models — nothing to compare."
    best = min(chall, key=chall.get)
    parts = [
        f"Over {counts['scored_rows']:,} scored snapshot(s) in {counts['n_windows']} window(s), "
        f"Brier: logistic {eh.fmt(overall['logistic']['brier'])}, "
        f"GBT-plain {eh.fmt(overall['gbt_plain']['brier'])}, "
        f"GBT-residual {eh.fmt(overall['gbt_residual']['brier'])}, "
        f"formula {eh.fmt(overall['formula']['brier'])}.",
        f"Best challenger overall: {_STYLE[best][1]} (Brier {eh.fmt(chall[best])}).",
    ]
    fb = overall["formula"]["brier"]
    for m in ("gbt_plain", "gbt_residual"):
        mb, lb = overall[m]["brier"], overall["logistic"]["brier"]
        vs_log = "beats" if mb < lb else ("ties" if abs(mb - lb) < 1e-9 else "trails")
        vs_f = "beats" if mb < fb else ("ties" if abs(mb - fb) < 1e-9 else "trails")
        parts.append(f"{_STYLE[m][1]} {vs_log} logistic and {vs_f} the formula overall.")
    mk = overall["market"]
    if np.isfinite(mk.get("brier", float("nan"))):
        parts.append(f"On the {mk['n']:,}-snapshot market-covered subset, market Brier is "
                     f"{eh.fmt(mk['brier'])} (compare per-predictor coverage in the table).")
    if counts["n_common"] < 0.9 * max(counts["n_gbt_plain"], 1):
        parts.append(f"⚠ Only {counts['n_common']:,} of {counts['n_gbt_plain']:,} GBT rows had all "
                     "models present — the grids may have been produced with different walk-forward "
                     "configs (use run_all --with-gbt for aligned grids).")
    return " ".join(parts)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _period_png(day_rows: list[dict]) -> str | None:
    per = [r for r in day_rows if r["period"] != "ALL"]
    periods = sorted({r["period"] for r in per})
    if not periods:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for pred in PREDICTORS:
        by_p = {r["period"]: r["brier"] for r in per if r["predictor"] == pred}
        vals = [by_p.get(p, float("nan")) for p in periods]
        if not any(np.isfinite(v) for v in vals):
            continue
        color, lbl = _STYLE[pred]
        ax.plot(periods, vals, "o-", color=color, label=lbl, markersize=4)
    ax.set_ylabel("Brier (↓ better)")
    ax.set_title("Brier by day — challengers vs benchmarks")
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    return _png(fig)


def _overall_table(overall: dict) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                   ("model", "n", "windows", "coverage", "Brier", "log-loss", "dir-acc"))
    body = []
    for pred in PREDICTORS:
        s = overall[pred]
        body.append(
            "<tr>" + "".join(f"<td>{c}</td>" for c in (
                _STYLE[pred][1], f"{s['n']:,}", s["n_windows"], eh.fmt(s.get("coverage"), 2),
                eh.fmt(s["brier"]), eh.fmt(s["logloss"]), eh.fmt(s["diracc"]))) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _build_html(metrics: dict) -> str:
    img = metrics.get("_period_png")
    img_tag = (f"<img alt='brier by day' src='data:image/png;base64,{img}'/>"
               if img else "<p class='muted'>per-day chart not available.</p>")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Model comparison — logistic vs GBT</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 980px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }}
 .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
</style></head><body>
<h1>Model comparison — logistic vs GBT vs benchmarks (outcome)</h1>
<p class="muted">All predictors scored on the same GBT-anchored rows; the formula and market
benchmarks are pairwise-defined (coverage shown). Health filter: <code>{metrics['params']['health']}</code>.</p>
<p class="verdict">{metrics['verdict']}</p>
<h2>Overall</h2>
{_overall_table(metrics['overall'])}
<h2>Brier by day</h2>
{img_tag}
<p class="muted">Generated by <code>model_lab.compare</code>.</p>
</body></html>
"""


_CSV_COLS = ["scope", "by", "period", "period_key", "tau_bucket", "predictor",
             "n", "n_windows", "coverage", "brier", "logloss", "diracc", "low_sample"]


def compare(paths: Paths, *, min_windows: int = DEFAULT_MIN_WINDOWS,
            health: str = "ready", series: str | None = None) -> dict:
    """Score logistic + both GBT variants + formula + market on one common set of
    rows; write ``out/compare/``. Returns the metrics dict."""
    log_path = paths.out_dir / "learn" / "predictions_outcome.parquet"
    gbt_plain_path = paths.out_dir / "learn_gbt" / "predictions_outcome_plain.parquet"
    gbt_res_path = paths.out_dir / "learn_gbt" / "predictions_outcome_residual.parquet"
    log_grid = _load_grid(log_path, "learn/predictions_outcome.parquet")
    plain_grid = _load_grid(gbt_plain_path, "learn_gbt/predictions_outcome_plain.parquet")
    res_grid = _load_grid(gbt_res_path, "learn_gbt/predictions_outcome_residual.parquet")

    series_filter = [s.strip() for s in series.split(",")] if series else None
    if series_filter is not None:
        for g in (log_grid, plain_grid, res_grid):
            g.drop(g.index[~g["series"].isin(series_filter)], inplace=True)

    ctx = eh.load_benchmarks(paths)
    frame, counts = eh.build_external_frame_from(ctx, plain_grid)
    out_dir = paths.out_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "params": {"min_windows": int(min_windows), "health": health, "series": series},
        "counts": {"resolved_windows": counts.get("resolved_windows", 0)},
        "predictors": PREDICTORS,
    }
    if frame.empty:
        metrics["counts"].update({"n_logistic": int(len(log_grid)), "n_gbt_plain": int(len(plain_grid)),
                                  "n_gbt_residual": int(len(res_grid)), "n_common": 0,
                                  "scored_rows": 0, "n_windows": 0})
        metrics["overall"] = {p: {"n": 0, "n_windows": 0, "brier": float("nan"),
                                  "logloss": float("nan"), "diracc": float("nan"),
                                  "coverage": float("nan")} for p in PREDICTORS}
        metrics["by_period"] = {"day": [], "week": []}
        metrics["by_tau"] = []
        metrics["verdict"] = "No resolved windows with scored GBT snapshots — nothing to compare."
        _write(out_dir, metrics)
        return metrics

    frame = frame.rename(columns={"model": "gbt_plain"})

    def _attach(grid: pd.DataFrame, name: str) -> None:
        g = grid.rename(columns={"window_open_ms": "open_time", "sample_ts_ms": "ts", "p_up": name})
        frame_merge = g[["series", "open_time", "ts", name]]
        nonlocal frame
        frame = frame.merge(frame_merge, on=["series", "open_time", "ts"], how="left")

    _attach(log_grid, "logistic")
    _attach(res_grid, "gbt_residual")

    scored = frame if health == "all" else frame[frame["health"] == "Ready"]
    scored = scored.reset_index(drop=True)

    common = scored.dropna(subset=["logistic", "gbt_plain", "gbt_residual"])
    total = int(len(scored))
    counts_out = {
        "resolved_windows": counts.get("resolved_windows", 0),
        "n_logistic": int(scored["logistic"].notna().sum()),
        "n_gbt_plain": int(scored["gbt_plain"].notna().sum()),
        "n_gbt_residual": int(scored["gbt_residual"].notna().sum()),
        "n_common": int(len(common)),
        "scored_rows": total,
        "n_windows": int(scored.groupby(["series", "open_time"]).ngroups),
    }
    metrics["counts"] = counts_out

    overall = {}
    for pred in PREDICTORS:
        s = _score(scored, pred)
        s["coverage"] = (s["n"] / total) if total else float("nan")
        overall[pred] = s
    metrics["overall"] = overall
    day_rows = _period_rows(scored, "day", total, min_windows)
    metrics["by_period"] = {"day": day_rows, "week": _period_rows(scored, "week", total, min_windows)}
    metrics["by_tau"] = _tau_rows(scored, total, min_windows)
    metrics["verdict"] = _verdict(overall, counts_out)
    metrics["_period_png"] = _period_png(day_rows)
    _write(out_dir, metrics)
    return metrics


def _write(out_dir, metrics: dict) -> None:
    serializable = {k: v for k, v in metrics.items() if k != "_period_png"}
    (out_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2, default=eh._json_default), encoding="utf-8")
    rows: list[dict] = []
    for r in metrics["by_period"]["day"] + metrics["by_period"]["week"]:
        rows.append({"scope": "period", "tau_bucket": "", **r})
    for r in metrics["by_tau"]:
        rows.append({"scope": "tau_bucket", "by": "", "period": "", "period_key": "", **r})
    pd.DataFrame(rows, columns=_CSV_COLS).to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "report.html").write_text(_build_html(metrics), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "compare")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS)
    parser.add_argument("--health", choices=("ready", "all"), default="ready")
    parser.add_argument("--series", default=None, help="comma-separated series filter")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    print(f"[compare] learn={paths.out_dir / 'learn'}  learn_gbt={paths.out_dir / 'learn_gbt'}")
    try:
        m = compare(paths, min_windows=args.min_windows, health=args.health, series=args.series)
    except ParquetNotReady as exc:
        print(f"[compare] {exc}")
        return 1
    c = m["counts"]
    print(f"[compare] scored {c['scored_rows']:,} rows in {c['n_windows']} windows "
          f"(common across all models: {c['n_common']:,})")
    print(f"[compare] {m['verdict']}")
    print(f"[compare] wrote {paths.out_dir / 'compare'} (comparison.csv, metrics.json, report.html)")
    return 0 if c["scored_rows"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
