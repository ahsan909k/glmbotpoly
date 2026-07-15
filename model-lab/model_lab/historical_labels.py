"""Stage — historical_labels (Part 2).

The **official** Polymarket resolution is the PRIMARY window label (from the Telonex markets
catalog `result_id`, validated 0-mismatch against the CLOB API + the journal by
:mod:`model_lab.historical_resolutions`). Quote-convergence and Binance-anchored `end ≥ strike`
are **cross-checks**, graded against the official truth (which proxy to trust where the API
lacks a market — Part A.3). Nothing is guessed: a window with no official resolution is
**counted and categorized**, never proxy-filled.

Because the label is the official resolution, the per-window pass is a dict lookup (no quote
reads) → near-100% coverage instantly. The proxy **grading** reads quotes + aggTrades for a
*stratified sample* (every series/month + knife-edge) and reports each proxy's error rate
overall and at knife-edge.

Output ``out/historical_labels/{labels.parquet, metrics.json, report.html}``.

Run::

    python -m model_lab.historical_labels                        # full history, official labels
    python -m model_lab.historical_labels --since 2026-05-01 --until 2026-05-08
    python -m model_lab.historical_labels --series BTC-5m,ETH-5m
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import historical_common as hc
from . import historical_resolutions as hr
from .config import Paths, resolve_bounds, resolve_paths, stage_parser
from .io import telonex as tx
from .io import telonex_pm as tpm
from .lib import procmem

STRIKE_TOL_MS = 2500.0
KNIFE_VOL_THRESH = hr.KNIFE_VOL_THRESH
GRADE_PER_STRATUM = 80  # windows sampled per (series, month) for the proxy grading

LABEL_COLS = ["series", "asset", "window_open_ms", "window_close_ms", "condition_id",
              "outcome", "outcome_up", "source", "day", "month"]


def _month_key(open_ms: int) -> str:
    d = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


def _days_for_series(paths: Paths, series_key: str, since_ms: int | None, until_ms: int | None) -> list[date]:
    """UTC days to process for ``series_key`` — the days present in ``coverage.json`` for
    the series (∩ ``[since, until]``), else a disk scan of the quotes dir."""
    prefix = hc.series_prefix(series_key)
    if prefix is None:
        return []
    lo = hc.utc_day(since_ms) if since_ms is not None else None
    hi = hc.utc_day(until_ms - 1) if until_ms is not None else None
    days: set[date] = set()
    cov_path = paths.out_dir / "telonex" / "coverage.json"
    if cov_path.exists():
        try:
            cov = json.loads(cov_path.read_text(encoding="utf-8"))
            for dstr in ((cov.get("series") or {}).get(prefix) or {}).get("days") or {}:
                try:
                    days.add(date.fromisoformat(dstr))
                except ValueError:
                    continue
        except (OSError, json.JSONDecodeError):
            pass
    if not days:
        qdir = tx.raw_root(paths.telonex_dir) / tx.POLYMARKET / "quotes"
        if qdir.is_dir():
            rx = re.compile(rf"^{re.escape(prefix)}-\d+-Up-(\d{{4}}-\d{{2}}-\d{{2}})\.parquet(?:\.zst)?$")
            for p in qdir.glob(f"{prefix}-*-Up-*.parquet*"):
                m = rx.match(p.name)
                if m:
                    try:
                        days.add(date.fromisoformat(m.group(1)))
                    except ValueError:
                        continue
    return sorted(d for d in days if (lo is None or d >= lo) and (hi is None or d <= hi))


def _grade_proxies(paths: Paths, present: pd.DataFrame, res_map: dict, *, per_stratum: int = GRADE_PER_STRATUM,
                   seed: int = 20260709) -> dict:
    """Grade quote-convergence and Binance-anchored against the official label over a
    stratified sample (every series/month + knife-edge): each proxy's error rate overall
    and at knife-edge. Reads quotes + aggTrades only for the sample."""
    rng = np.random.default_rng(seed)
    lab = present[present["outcome"].notna()].copy()
    if lab.empty:
        return {"sampled": 0}
    lab["month"] = lab["window_open_ms"].map(_month_key)
    lab["day"] = lab["window_open_ms"].map(lambda ms: hc.utc_day(int(ms)).isoformat())
    rows: list[dict] = []
    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for (series, month), g in lab.groupby(["series", "month"]):
        days = sorted(g["day"].unique())
        day_iso = days[len(days) // 2]
        wins = g[g["day"] == day_iso]
        if len(wins) > per_stratum:
            wins = wins.iloc[rng.choice(len(wins), size=per_stratum, replace=False)]
        asset = hc.SLUG_PREFIXES[hc.series_prefix(series)][1]
        key = (series, day_iso)
        if key not in bars_cache:
            bars_cache[key] = hc.binance_bars_for_day(paths, hc.SYMBOL_BY_ASSET[asset], date.fromisoformat(day_iso))
        bars = bars_cache[key]
        dstr = day_iso
        for w in wins.itertuples(index=False):
            o, c = int(w.window_open_ms), int(w.window_close_ms)
            slug = _slug_of(series, o)
            upq = tpm.read_pm_quotes(paths.telonex_dir, slug, "Up", dstr)
            dnq = tpm.read_pm_quotes(paths.telonex_dir, slug, "Down", dstr)
            conv = hc.resolve_by_quote_convergence(upq, dnq, o, c)
            binr = hc.binance_outcome(bars, o, c, STRIKE_TOL_MS)
            vd = binr["strike_distance_at_close_vol"]
            rows.append({"official": w.outcome, "quote_conv": conv["outcome"], "binance": binr["outcome"],
                         "knife_edge": bool(vd is not None and np.isfinite(vd) and vd < KNIFE_VOL_THRESH)})
    df = pd.DataFrame(rows)

    def grade(col: str, subset: pd.DataFrame) -> dict:
        d = subset[subset[col].notna()]
        n = int(len(d))
        err = int((d[col] != d["official"]).sum())
        return {"n": n, "errors": err, "error_rate": (err / n) if n else float("nan"),
                "coverage": (n / len(subset)) if len(subset) else float("nan")}

    knife = df[df["knife_edge"]]
    return {
        "sampled": int(len(df)), "knife_edge_sampled": int(len(knife)),
        "quote_convergence": {"overall": grade("quote_conv", df), "knife_edge": grade("quote_conv", knife)},
        "binance_anchored": {"overall": grade("binance", df), "knife_edge": grade("binance", knife)},
    }


def _slug_of(series_key: str, open_ms: int) -> str:
    return f"{hc.series_prefix(series_key)}-{open_ms // 1000}"


def historical_labels(
    paths: Paths, *, series: tuple[str, ...] | None = None,
    since_ms: int | None = None, until_ms: int | None = None,
    res_map: dict | None = None, grade: bool = True,
) -> dict:
    """Assign the official resolution to every window in the catalog universe, report
    coverage + categorize the unlabeled, grade the two proxies, and write the labels. The
    window universe comes from the catalog (fast + authoritative), not a 261k-file glob."""
    paths.ensure_out()
    out_dir = paths.out_dir / "historical_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    series_keys = tuple(series) if series else hc.DEFAULT_SERIES
    if res_map is None:
        res_map = hr.load_resolution_map(paths, series_keys)

    universe = hr.catalog_windows(paths, series_keys, since_ms, until_ms)
    windows_seen = int(len(universe))
    rows: list[dict] = []
    cats: dict[str, int] = defaultdict(int)
    for w in universe.itertuples(index=False):
        series_key = w.series
        open_ms = int(w.window_open_ms)
        official = res_map.get((series_key, open_ms))
        if official is None:  # no official resolution (unsettled/active) → counted, never guessed
            cats[f"unlabeled_{w.status}"] += 1
            continue
        dur = hc.SLUG_PREFIXES[hc.series_prefix(series_key)][2]
        rows.append({
            "series": series_key, "asset": hc.SLUG_PREFIXES[hc.series_prefix(series_key)][1],
            "window_open_ms": open_ms, "window_close_ms": open_ms + dur * 1000,
            "condition_id": official.get("condition_id"),
            "outcome": official["outcome"], "outcome_up": 1 if official["outcome"] == "Up" else 0,
            "source": "official", "day": hc.utc_day(open_ms).isoformat(), "month": _month_key(open_ms),
        })

    df = pd.DataFrame(rows, columns=LABEL_COLS)
    n_resolved = int(len(df))

    per_sm: list[dict] = []
    if not df.empty:
        for (s, m), g in df.groupby(["series", "month"]):
            per_sm.append({"series": s, "month": m, "resolved": int(len(g)),
                           "up_rate": float((g["outcome"] == "Up").mean()), "low_sample": bool(len(g) < 30)})
        per_sm.sort(key=lambda r: (r["series"], r["month"]))

    grading = _grade_proxies(paths, df, res_map) if (grade and not df.empty) else {"sampled": 0}
    df.to_parquet(out_dir / "labels.parquet", engine="pyarrow", index=False)

    counts = {
        "windows_seen": windows_seen, "resolved_official": n_resolved,
        "unlabeled_no_official": int(windows_seen - n_resolved),
        "coverage_frac": (n_resolved / windows_seen) if windows_seen else float("nan"),
        "catalog_resolutions_available": len(res_map),
    }
    result = {
        "params": {"title": "Historical window labels (official resolution primary)",
                   "series": list(series_keys), "since_ms": since_ms, "until_ms": until_ms},
        "counts": counts, "unlabeled_categories": dict(cats),
        "per_series_month": per_sm, "proxy_grading": grading, "peak_rss_mb": procmem.peak_rss_mb(),
    }
    _write_outputs(out_dir, result, per_sm)
    return result


# --- outputs ----------------------------------------------------------------
def _verdict(result: dict) -> str:
    c = result["counts"]
    parts = [
        f"Official Polymarket resolution (Telonex catalog `result_id`, validated 0-mismatch vs "
        f"the CLOB API + the journal) labels {c['resolved_official']:,} of {c['windows_seen']:,} "
        f"windows present on disk ({c['coverage_frac']:.1%}); {c['unlabeled_no_official']:,} have no "
        f"official resolution yet (unsettled/active) and are counted, never guessed.",
    ]
    g = result.get("proxy_grading") or {}
    if g.get("sampled"):
        qc = g["quote_convergence"]; bn = g["binance_anchored"]
        parts.append(
            f"Graded against official truth over {g['sampled']:,} sampled windows "
            f"({g['knife_edge_sampled']:,} knife-edge): quote-convergence errs "
            f"{qc['overall']['error_rate']:.2%} overall / {qc['knife_edge']['error_rate']:.2%} at knife-edge "
            f"(coverage {qc['overall']['coverage']:.0%}); Binance-anchored errs "
            f"{bn['overall']['error_rate']:.2%} / {bn['knife_edge']['error_rate']:.2%} at knife-edge.")
    return " ".join(parts)


def _write_outputs(out_dir: Path, result: dict, per_sm: list[dict]) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")
    verdict = _verdict(result)
    cov_cols = [("series", "series"), ("month", "month"), ("resolved", "resolved (official)"),
                ("up_rate", "Up rate"), ("low_sample", "low_sample")]
    g = result.get("proxy_grading") or {}
    grade_rows = []
    if g.get("sampled"):
        for name, key in (("quote-convergence", "quote_convergence"), ("Binance-anchored", "binance_anchored")):
            grade_rows.append({"proxy": name, "coverage": g[key]["overall"]["coverage"],
                               "err_overall": g[key]["overall"]["error_rate"],
                               "err_knife_edge": g[key]["knife_edge"]["error_rate"],
                               "n_overall": g[key]["overall"]["n"], "n_knife": g[key]["knife_edge"]["n"]})
    grade_html = ("<h2>Proxy grading vs official truth (sampled)</h2>"
                  + eh._table_html(grade_rows, [("proxy", "proxy"), ("coverage", "coverage"),
                                                ("err_overall", "error (overall)"),
                                                ("err_knife_edge", "error (knife-edge)"),
                                                ("n_overall", "n"), ("n_knife", "n knife")])) if grade_rows else ""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} tr.low td {{ background: #fff5e6; color: #7a5200; }}
 .muted {{ color: #888; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<div class="verdict">{verdict}</div>
{grade_html}
<h2>Coverage per series per month</h2>
{eh._table_html(per_sm, cov_cols)}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "historical_labels")
    parser.add_argument("--series", default=None,
                        help="comma-separated series filter (default the 4-series universe)")
    parser.add_argument("--no-grade", action="store_true", help="skip the sampled proxy grading")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    series = tuple(s.strip() for s in args.series.split(",")) if args.series else None
    print(f"[historical_labels] telonex={paths.telonex_dir} out={paths.out_dir / 'historical_labels'}")
    result = historical_labels(paths, series=series, since_ms=since_ms, until_ms=until_ms, grade=not args.no_grade)
    c = result["counts"]
    print(f"[historical_labels] official-labeled {c['resolved_official']:,}/{c['windows_seen']:,} "
          f"({c['coverage_frac']:.1%}); no-official {c['unlabeled_no_official']:,}")
    g = result.get("proxy_grading") or {}
    if g.get("sampled"):
        qc, bn = g["quote_convergence"], g["binance_anchored"]
        print(f"[historical_labels] proxy grading ({g['sampled']:,} sampled, {g['knife_edge_sampled']:,} knife): "
              f"quote-conv err {qc['overall']['error_rate']:.2%} (knife {qc['knife_edge']['error_rate']:.2%}); "
              f"binance err {bn['overall']['error_rate']:.2%} (knife {bn['knife_edge']['error_rate']:.2%})")
    procmem.report_peak_rss("historical_labels", result.get("peak_rss_mb"), args.max_rss_mb)
    if c["resolved_official"] == 0:
        print("[historical_labels] WARNING: no official labels — build the resolution cache "
              "(`python -m model_lab.historical_resolutions`).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
