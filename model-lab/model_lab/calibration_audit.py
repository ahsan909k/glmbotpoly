"""Calibration audit — how well-calibrated is the model, and does it beat the market?

A **self-contained** stage: it reads the gzip journal *directly* (not the
``ingest``/``labels`` parquet), so a single command works on fresh journal data
with no prerequisite stages. For every resolved window it lines up two implied
probabilities of *Up* against the realized outcome:

- the **formula model** ``p_up`` snapshots (``crates/model``), and
- the **market** — the Up-token order-book mid ``(best_bid + best_ask)/2`` from
  the ``top_of_book`` records.

and scores both with the **Brier score** and **log-loss**, split by **series**
and by **time-remaining bucket** (early / mid / final-minute / final-20s), plus a
reliability table (probability bins vs the actual Up frequency, with sample
counts) and a plain-language verdict. Every breakdown carries a **minimum-sample
warning** wherever it is backed by too few distinct *windows* to trust (snapshots
within one window share a single outcome, so windows — not snapshots — are the
unit of trust).

Outputs ``out/calibration_audit/{metrics.json, scores.csv, reliability.csv,
report.html}``.

Run (single command, honors ``--journal-dir`` / ``--out``)::

    python -m model_lab.calibration_audit
    python -m model_lab.calibration_audit --min-windows 50 --health ready
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG, never open a window.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Paths, resolve_paths, stage_parser  # noqa: E402
from .io import journal as jio  # noqa: E402
from .lib import math as lm  # noqa: E402

# --- tuning constants -------------------------------------------------------
N_BINS = 10
DEFAULT_MIN_WINDOWS = 50
# Don't pair a model snapshot with an Up-token mid older than this (guards a
# window whose book went silent). Books on an active pair update ~1/s, so this
# only ever discards genuinely stale mids.
MAX_MID_STALENESS_MS = 120_000

# Time-remaining buckets, in display order (early → about to resolve).
TAU_ORDER = ["early", "mid", "final_minute", "final_20s"]
TAU_LABELS = {
    "early": "early (τ > ½·window)",
    "mid": "mid (60s–½·window)",
    "final_minute": "final minute (20–60s)",
    "final_20s": "final 20s (≤ 20s)",
}

_METRIC_KEYS = [
    "n_snapshots", "n_model", "n_market", "n_windows",
    "model_brier", "model_logloss", "market_brier", "market_logloss",
    "low_sample",
]


# --- journal → tidy frames --------------------------------------------------
def _read_journal(paths: Paths) -> tuple[list[dict], list[tuple], dict, dict, dict]:
    """One pass over the journal. Returns ``(model_rows, top_rows, win_meta,
    outcome_resolved, outcome_settle)``.

    ``win_meta[(series, open_time)] = {close_time, up_token, down_token}`` from
    any ``window`` record; ``outcome_resolved`` from the ``Resolved`` lifecycle
    (fires for every resolved window), ``outcome_settle`` from ``settlement``
    (traded windows only) as a fallback.
    """
    model_rows: list[dict] = []
    top_rows: list[tuple] = []  # (token_id, ts, mid)
    win_meta: dict[tuple[str, int], dict] = {}
    outcome_resolved: dict[tuple[str, int], str] = {}
    outcome_settle: dict[tuple[str, int], str] = {}

    for rec in jio.read_records(paths.journal_dir):
        kind = rec.get("type")
        if kind == "model":
            window = rec.get("window")
            if not window:
                continue
            series, open_time = jio.window_key(window)
            model_rows.append(
                {
                    "series": series,
                    "open_time": open_time,
                    "ts": rec.get("ts"),
                    "p_up": jio.to_float(rec.get("p_up")),
                    "health": rec.get("health"),
                }
            )
        elif kind == "top_of_book":
            top = rec.get("top") or {}
            bid = top.get("bid")
            ask = top.get("ask")
            if not isinstance(bid, dict) or not isinstance(ask, dict):
                continue  # one-sided book → no mid
            bid_p = jio.to_float(bid.get("price"))
            ask_p = jio.to_float(ask.get("price"))
            if not (np.isfinite(bid_p) and np.isfinite(ask_p)):
                continue
            top_rows.append((rec.get("token_id"), top.get("ts"), (bid_p + ask_p) / 2.0))
        elif kind == "window":
            market = rec.get("market") or {}
            key = jio.window_key(market.get("window"))
            if key == ("", 0):
                continue
            if key not in win_meta:
                tokens = market.get("tokens") or {}
                win_meta[key] = {
                    "close_time": market.get("close_time"),
                    "up_token": tokens.get("up"),
                    "down_token": tokens.get("down"),
                }
            lifecycle = rec.get("lifecycle")
            if isinstance(lifecycle, dict) and "Resolved" in lifecycle:
                outcome = (lifecycle["Resolved"] or {}).get("outcome")
                if outcome in ("Up", "Down"):
                    outcome_resolved[key] = outcome
        elif kind == "settlement":
            key = jio.window_key(rec.get("window"))
            outcome = rec.get("outcome")
            if key != ("", 0) and outcome in ("Up", "Down"):
                outcome_settle[key] = outcome

    return model_rows, top_rows, win_meta, outcome_resolved, outcome_settle


def _windows_frame(win_meta: dict, outcome_resolved: dict, outcome_settle: dict) -> pd.DataFrame:
    """Resolved windows only: ``series, open_time, close_time, up_token,
    down_token, outcome_up, dur`` (outcome from Resolved, else settlement)."""
    rows: list[dict] = []
    for key, meta in win_meta.items():
        close_time = meta.get("close_time")
        outcome = outcome_resolved.get(key) or outcome_settle.get(key)
        if close_time is None or outcome not in ("Up", "Down"):
            continue
        series, open_time = key
        rows.append(
            {
                "series": series,
                "open_time": int(open_time),
                "close_time": int(close_time),
                "up_token": meta.get("up_token"),
                "down_token": meta.get("down_token"),
                "outcome_up": 1 if outcome == "Up" else 0,
                "dur": (int(close_time) - int(open_time)) / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _market_frame(top_rows: list[tuple], windows: pd.DataFrame) -> pd.DataFrame:
    """Up-token tops → ``series, open_time, ts_mid, up_mid``.

    A top is assigned to the window whose Up token it carries *and* whose
    ``[open, close)`` contains its timestamp (real Up tokens are unique per
    window; the interval test also correctly partitions the fixture's reused
    ids). Down-token and unmatched tops are dropped.
    """
    if windows.empty or not top_rows:
        return pd.DataFrame(columns=["series", "open_time", "ts_mid", "up_mid"])
    up_windows: dict[Any, list[tuple[int, int, str]]] = defaultdict(list)
    for w in windows.itertuples(index=False):
        up_windows[w.up_token].append((w.open_time, w.close_time, w.series))

    rows: list[dict] = []
    for token_id, ts, mid in top_rows:
        cands = up_windows.get(token_id)
        if not cands or ts is None:
            continue
        for open_time, close_time, series in cands:
            if open_time <= ts < close_time:
                rows.append({"series": series, "open_time": open_time, "ts_mid": int(ts), "up_mid": float(mid)})
                break
    return pd.DataFrame(rows, columns=["series", "open_time", "ts_mid", "up_mid"])


def _tau_bucket(tau: np.ndarray, dur: np.ndarray) -> np.ndarray:
    """Duration-aware time-remaining bucket (first matching condition wins)."""
    return np.select(
        [tau <= 20.0, tau <= 60.0, tau <= 0.5 * dur],
        ["final_20s", "final_minute", "mid"],
        default="early",
    )


def _joined_frame(model_rows: list[dict], windows: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Each Ready-or-not model snapshot in a resolved window, with its nearest
    earlier Up-token mid, ``tau``, ``tau_bucket`` and the realized ``outcome_up``.
    Snapshots with ``tau ≤ 0`` are dropped."""
    model = pd.DataFrame(model_rows)
    if model.empty or windows.empty:
        return pd.DataFrame()
    df = model.merge(
        windows[["series", "open_time", "close_time", "outcome_up", "dur"]],
        on=["series", "open_time"],
        how="inner",
    )
    if df.empty:
        return df
    df = df[df["ts"].notna()].copy()
    if df.empty:
        return df
    df["ts"] = df["ts"].astype("int64")
    df["tau"] = (df["close_time"] - df["ts"]) / 1000.0
    df = df[df["tau"] > 0.0].copy()
    if df.empty:
        return df

    # As-of join to the last Up-token mid at or before each snapshot (per window).
    if market.empty:
        df["up_mid"] = np.nan
    else:
        right = market.sort_values("ts_mid").rename(columns={"ts_mid": "ts"})
        merged = pd.merge_asof(
            df.sort_values("ts"),
            right[["series", "open_time", "ts", "up_mid"]].assign(mid_ts=right["ts"]),
            on="ts",
            by=["series", "open_time"],
            direction="backward",
        )
        stale = ~np.isfinite(merged["mid_ts"]) | ((merged["ts"] - merged["mid_ts"]) > MAX_MID_STALENESS_MS)
        merged.loc[stale, "up_mid"] = np.nan
        df = merged.drop(columns=["mid_ts"])

    df["outcome_up"] = df["outcome_up"].astype(float)
    df["tau_bucket"] = _tau_bucket(df["tau"].to_numpy(dtype=float), df["dur"].to_numpy(dtype=float))
    return df


# --- scoring ----------------------------------------------------------------
def _brier(prob: pd.Series, outcome: pd.Series) -> float:
    return lm.brier_score(prob.to_numpy(dtype=float), outcome.to_numpy(dtype=float)) if len(prob) else float("nan")


def _logloss(prob: pd.Series, outcome: pd.Series) -> float:
    return lm.log_loss(prob.to_numpy(dtype=float), outcome.to_numpy(dtype=float)) if len(prob) else float("nan")


def _score(df: pd.DataFrame, min_windows: int) -> dict:
    """Brier + log-loss for the model and the market over ``df`` (rows already
    filtered to a resolved window with ``tau > 0``)."""
    n_snap = int(len(df))
    n_windows = int(df.groupby(["series", "open_time"]).ngroups) if n_snap else 0
    m = df.dropna(subset=["p_up"])
    k = df.dropna(subset=["up_mid"])
    return {
        "n_snapshots": n_snap,
        "n_model": int(len(m)),
        "n_market": int(len(k)),
        "n_windows": n_windows,
        "model_brier": _brier(m["p_up"], m["outcome_up"]),
        "model_logloss": _logloss(m["p_up"], m["outcome_up"]),
        "market_brier": _brier(k["up_mid"], k["outcome_up"]),
        "market_logloss": _logloss(k["up_mid"], k["outcome_up"]),
        "low_sample": bool(n_windows < min_windows),
    }


def _scope_rows(scored: pd.DataFrame, all_health: pd.DataFrame, min_windows: int) -> list[dict]:
    """The flat list of scored breakdowns, each tagged with ``scope`` + dims."""
    rows: list[dict] = []

    def add(scope: str, df: pd.DataFrame, *, series: str = "", tau_bucket: str = "") -> None:
        rows.append({"scope": scope, "series": series, "tau_bucket": tau_bucket, **_score(df, min_windows)})

    add("overall", scored)
    add("overall_all_health", all_health)
    add("overall_headtohead", scored.dropna(subset=["p_up", "up_mid"]))
    for series in sorted(scored["series"].dropna().unique()):
        add("series", scored[scored["series"] == series], series=series)
    for bucket in TAU_ORDER:
        g = scored[scored["tau_bucket"] == bucket]
        if not g.empty:
            add("tau_bucket", g, tau_bucket=bucket)
    for series in sorted(scored["series"].dropna().unique()):
        for bucket in TAU_ORDER:
            g = scored[(scored["series"] == series) & (scored["tau_bucket"] == bucket)]
            if not g.empty:
                add("series_tau", g, series=series, tau_bucket=bucket)
    return rows


def _reliability(df: pd.DataFrame, col: str, min_windows: int, n_bins: int = N_BINS) -> list[dict]:
    """Reliability bins for ``col`` with snapshot and distinct-window counts."""
    d = df.dropna(subset=[col])
    if d.empty:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(d[col].to_numpy(dtype=float), edges, right=False) - 1, 0, n_bins - 1)
    d = d.assign(_bin=idx)
    out: list[dict] = []
    for b in range(n_bins):
        g = d[d["_bin"] == b]
        if g.empty:
            continue
        nw = int(g.groupby(["series", "open_time"]).ngroups)
        out.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "bin_mid": float((edges[b] + edges[b + 1]) / 2.0),
                "n_snapshots": int(len(g)),
                "n_windows": nw,
                "mean_pred": float(g[col].mean()),
                "empirical_rate": float(g["outcome_up"].mean()),
                "low_sample": bool(nw < min_windows),
            }
        )
    return out


def _ece(reliability: list[dict]) -> float:
    """Expected calibration error: window/snapshot-weighted mean |pred − actual|."""
    if not reliability:
        return float("nan")
    n = sum(r["n_snapshots"] for r in reliability)
    if n == 0:
        return float("nan")
    return sum(r["n_snapshots"] * abs(r["mean_pred"] - r["empirical_rate"]) for r in reliability) / n


# --- verdict ----------------------------------------------------------------
def _f(x: Any, dp: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{dp}f}" if isinstance(x, float) else str(x)


def _find(rows: list[dict], scope: str, **dims) -> dict | None:
    for r in rows:
        if r["scope"] == scope and all(r.get(k) == v for k, v in dims.items()):
            return r
    return None


def _verdict(scope_rows: list[dict], model_rel: list[dict], min_windows: int, health: str) -> str:
    overall = _find(scope_rows, "overall") or {}
    h2h = _find(scope_rows, "overall_headtohead") or {}
    nw = overall.get("n_windows", 0)
    n_model = overall.get("n_model", 0)
    n_market = overall.get("n_market", 0)
    if nw == 0 or n_model == 0:
        return "No resolved windows with model snapshots were found — nothing to audit."

    parts: list[str] = []
    parts.append(
        f"Over {nw} resolved window(s) and {n_model:,} model snapshot(s) "
        f"(health={health}), the formula model scores a Brier of "
        f"{_f(overall.get('model_brier'))} and log-loss {_f(overall.get('model_logloss'))} "
        "(both lower-is-better)."
    )

    ece = _ece(model_rel)
    if np.isfinite(ece):
        # Sign of the mean gap: does the model lean high or low overall?
        gap = sum(r["n_snapshots"] * (r["mean_pred"] - r["empirical_rate"]) for r in model_rel)
        tot = sum(r["n_snapshots"] for r in model_rel)
        signed = gap / tot if tot else 0.0
        if ece < 0.02:
            lean = "is well-calibrated on average"
        elif signed > 0:
            lean = "tends to over-state the chance of Up (over-confident on the high side)"
        else:
            lean = "tends to under-state the chance of Up"
        parts.append(f"Its reliability curve {lean} (expected calibration error {_f(ece)}).")

    if n_market == 0:
        parts.append(
            "No market order-book (top_of_book) data was found in this journal, so the model "
            "could not be compared against the market mid — this is a model-only calibration."
        )
    else:
        cov = (h2h.get("n_snapshots", 0) / n_model) if n_model else 0.0
        mb, kb = h2h.get("model_brier"), h2h.get("market_brier")
        ml, kl = h2h.get("model_logloss"), h2h.get("market_logloss")
        if np.isfinite(mb) and np.isfinite(kb):
            if mb < kb:
                rel = "beats"
            elif mb > kb:
                rel = "trails"
            else:
                rel = "ties"
            parts.append(
                f"Head-to-head on the {h2h.get('n_snapshots', 0):,} snapshot(s) where both a model "
                f"probability and a market Up-mid exist ({cov:.0%} of model snapshots), the model {rel} "
                f"the market: Brier {_f(mb)} vs {_f(kb)}, log-loss {_f(ml)} vs {_f(kl)}."
            )

    # The near-close sharpening story.
    early = _find(scope_rows, "tau_bucket", tau_bucket="early")
    fin = _find(scope_rows, "tau_bucket", tau_bucket="final_20s")
    if fin and np.isfinite(fin.get("model_brier", float("nan"))):
        trust = "" if not fin.get("low_sample") else " (⚠ too few windows to trust)"
        early_b = _f(early.get("model_brier")) if early else "n/a"
        parts.append(
            f"In the final 20 seconds the model's Brier is {_f(fin.get('model_brier'))}{trust}, "
            f"versus {early_b} early in the window — "
            + ("it sharpens as resolution nears." if early and np.isfinite(early.get("model_brier", float("nan"))) and fin["model_brier"] < early["model_brier"] else "compare the time-remaining table below.")
        )

    # Best / worst series among trustworthy ones.
    series_rows = [r for r in scope_rows if r["scope"] == "series" and np.isfinite(r.get("model_brier", float("nan")))]
    trusted = [r for r in series_rows if not r["low_sample"]]
    if trusted:
        best = min(trusted, key=lambda r: r["model_brier"])
        worst = max(trusted, key=lambda r: r["model_brier"])
        if best is worst:
            parts.append(f"Best-calibrated series with enough windows: {best['series']} (Brier {_f(best['model_brier'])}).")
        else:
            parts.append(
                f"Among series with ≥ {min_windows} windows, {best['series']} is best-calibrated "
                f"(Brier {_f(best['model_brier'])}) and {worst['series']} is weakest "
                f"(Brier {_f(worst['model_brier'])})."
            )

    # Minimum-sample caveats.
    low = [r for r in scope_rows if r["scope"] in ("series", "tau_bucket", "series_tau") and r["low_sample"]]
    if low:
        n_low_series = sum(1 for r in scope_rows if r["scope"] == "series" and r["low_sample"])
        parts.append(
            f"⚠ {len(low)} breakdown(s) are backed by fewer than {min_windows} distinct windows and should not "
            f"be trusted{f' (including {n_low_series} whole series)' if n_low_series else ''}; they are flagged "
            "low_sample in the tables."
        )
    else:
        parts.append(f"Every breakdown is backed by at least {min_windows} windows.")

    return " ".join(parts)


# --- HTML report ------------------------------------------------------------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f"<p class='muted'>{alt}: not available.</p>"
    return f"<img alt='{alt}' src='data:image/png;base64,{b64}'/>"


def _reliability_png(model_rel: list[dict], market_rel: list[dict]) -> str | None:
    if not model_rel and not market_rel:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    if model_rel:
        mdf = pd.DataFrame(model_rel)
        ax.plot(mdf["mean_pred"], mdf["empirical_rate"], "o-", color="#1f77b4", label="model")
    if market_rel:
        kdf = pd.DataFrame(market_rel)
        ax.plot(kdf["mean_pred"], kdf["empirical_rate"], "s-", color="#ff7f0e", label="market mid")
    ax.set_xlabel("mean predicted P(Up)")
    ax.set_ylabel("empirical Up rate")
    ax.set_title("Calibration (reliability)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _png(fig)


def _tau_brier_png(scope_rows: list[dict]) -> str | None:
    buckets = [b for b in TAU_ORDER if _find(scope_rows, "tau_bucket", tau_bucket=b)]
    if not buckets:
        return None
    model = [_find(scope_rows, "tau_bucket", tau_bucket=b).get("model_brier", float("nan")) for b in buckets]
    market = [_find(scope_rows, "tau_bucket", tau_bucket=b).get("market_brier", float("nan")) for b in buckets]
    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - 0.2, model, width=0.4, color="#1f77b4", label="model")
    ax.bar(x + 0.2, market, width=0.4, color="#ff7f0e", label="market mid")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=0)
    ax.set_ylabel("Brier (lower = better)")
    ax.set_title("Accuracy by time remaining")
    ax.legend()
    return _png(fig)


def _table_html(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>no rows.</p>"
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = []
    for r in rows:
        cls = " class='low'" if r.get("low_sample") else ""
        cells = []
        for key, _ in columns:
            v = r.get(key)
            if key == "low_sample":
                cells.append("<td>⚠</td>" if v else "<td></td>")
            elif isinstance(v, float):
                cells.append(f"<td>{_f(v)}</td>")
            else:
                cells.append(f"<td>{'' if v is None else v}</td>")
        body.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _build_html(metrics: dict) -> str:
    scope_rows = metrics["scope_rows"]
    model_rel = metrics["reliability"]["model"]
    market_rel = metrics["reliability"]["market"]

    rel_img = _img_tag(_reliability_png(model_rel, market_rel), "reliability curve")
    tau_img = _img_tag(_tau_brier_png(scope_rows), "Brier by time remaining")

    overall_rows = [r for r in scope_rows if r["scope"].startswith("overall")]
    series_rows = [r for r in scope_rows if r["scope"] == "series"]
    tau_rows = [r for r in scope_rows if r["scope"] == "tau_bucket"]
    cross_rows = [r for r in scope_rows if r["scope"] == "series_tau"]

    score_cols = [
        ("n_windows", "windows"), ("n_model", "model n"), ("n_market", "market n"),
        ("model_brier", "model Brier"), ("model_logloss", "model log-loss"),
        ("market_brier", "market Brier"), ("market_logloss", "market log-loss"),
        ("low_sample", "⚠"),
    ]
    overall_cols = [("scope", "scope")] + score_cols
    series_cols = [("series", "series")] + score_cols
    tau_cols = [("tau_bucket", "time left")] + score_cols
    cross_cols = [("series", "series"), ("tau_bucket", "time left")] + score_cols
    rel_cols = [
        ("bin_lo", "bin lo"), ("bin_hi", "bin hi"), ("n_snapshots", "snapshots"),
        ("n_windows", "windows"), ("mean_pred", "mean pred"), ("empirical_rate", "empirical Up"),
        ("low_sample", "⚠"),
    ]

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>calibration audit</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }}
 tr.low td {{ background: #fff5e6; color: #7a5200; }}
 .muted {{ color: #888; }} code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 img {{ max-width: 100%; }} .row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
</style></head><body>
<h1>Calibration audit</h1>
<p class="muted">Is the formula model well-calibrated, and does it beat the Polymarket order-book mid?
Rows shaded <span style="background:#fff5e6;color:#7a5200;padding:0 4px">amber ⚠</span> are backed by
fewer than {metrics['params']['min_windows']} distinct windows — too few to trust.</p>

<p class="verdict">{metrics['verdict']}</p>

<div class="row">{rel_img}{tau_img}</div>

<h2>Overall</h2>
{_table_html(overall_rows, overall_cols)}
<p class="muted"><code>overall</code> = health={metrics['params']['health']} snapshots ·
<code>overall_all_health</code> = every snapshot regardless of model health ·
<code>overall_headtohead</code> = only where both a model and a market probability exist.</p>

<h2>By series</h2>
{_table_html(series_rows, series_cols)}

<h2>By time remaining</h2>
{_table_html(tau_rows, tau_cols)}

<h2>By series × time remaining</h2>
{_table_html(cross_rows, cross_cols)}

<h2>Reliability — model</h2>
{_table_html(model_rel, rel_cols)}

<h2>Reliability — market mid</h2>
{_table_html(market_rel, rel_cols)}

<p class="muted">Generated by <code>python -m model_lab.calibration_audit</code>.</p>
</body></html>
"""


# --- entry ------------------------------------------------------------------
def calibration_audit(paths: Paths, min_windows: int = DEFAULT_MIN_WINDOWS, health: str = "ready") -> dict:
    """Runs the calibration audit; returns the metrics dict (also written to disk)."""
    out_dir = paths.out_dir / "calibration_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows, top_rows, win_meta, res, settle = _read_journal(paths)
    windows = _windows_frame(win_meta, res, settle)
    market = _market_frame(top_rows, windows)
    joined = _joined_frame(model_rows, windows, market)

    params = {"min_windows": int(min_windows), "health": health, "n_bins": N_BINS}
    counts = {
        "resolved_windows": int(len(windows)),
        "model_snapshots": int(len(model_rows)),
        "market_tops": int(len(top_rows)),
        "joined_snapshots": int(len(joined)),
    }

    if joined.empty:
        metrics = {
            "params": params,
            "counts": counts,
            "scope_rows": [],
            "reliability": {"model": [], "market": []},
            "verdict": "No resolved windows with model snapshots were found — nothing to audit. "
            "Check --journal-dir (and that the journal has model + window records).",
        }
        _write_outputs(out_dir, metrics)
        return metrics

    all_health = joined
    scored = joined if health == "all" else joined[joined["health"] == "Ready"]
    scored = scored if not scored.empty else joined.iloc[0:0]

    scope_rows = _scope_rows(scored, all_health, min_windows)
    model_rel = _reliability(scored, "p_up", min_windows)
    market_rel = _reliability(scored, "up_mid", min_windows)
    verdict = _verdict(scope_rows, model_rel, min_windows, health)

    metrics = {
        "params": params,
        "counts": counts,
        "scope_rows": scope_rows,
        "reliability": {"model": model_rel, "market": market_rel},
        "verdict": verdict,
    }
    _write_outputs(out_dir, metrics)
    return metrics


def _write_outputs(out_dir, metrics: dict) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scols = [
        "scope", "series", "tau_bucket", "n_snapshots", "n_model", "n_market", "n_windows",
        "model_brier", "model_logloss", "market_brier", "market_logloss", "low_sample",
    ]
    scores_df = pd.DataFrame(metrics["scope_rows"], columns=scols)
    scores_df.to_csv(out_dir / "scores.csv", index=False)

    rel_rows: list[dict] = []
    for predictor, rel in (("model", metrics["reliability"]["model"]), ("market", metrics["reliability"]["market"])):
        for r in rel:
            rel_rows.append({"predictor": predictor, **r})
    rcols = ["predictor", "bin_lo", "bin_hi", "bin_mid", "n_snapshots", "n_windows", "mean_pred", "empirical_rate", "low_sample"]
    pd.DataFrame(rel_rows, columns=rcols).to_csv(out_dir / "reliability.csv", index=False)

    (out_dir / "report.html").write_text(_build_html(metrics), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "calibration_audit")
    parser.add_argument("--min-windows", type=int, default=DEFAULT_MIN_WINDOWS,
                        help=f"flag a breakdown low-sample below this many distinct windows (default {DEFAULT_MIN_WINDOWS})")
    parser.add_argument("--health", choices=("ready", "all"), default="ready",
                        help="score only health=Ready snapshots (default) or every snapshot")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    print(f"[calibration_audit] journal={paths.journal_dir}")
    print(f"[calibration_audit] out    ={paths.out_dir / 'calibration_audit'}")

    m = calibration_audit(paths, min_windows=args.min_windows, health=args.health)
    c = m["counts"]
    print(f"[calibration_audit] resolved windows={c['resolved_windows']:,} "
          f"model snapshots={c['model_snapshots']:,} market tops={c['market_tops']:,} "
          f"scored={c['joined_snapshots']:,}")
    overall = _find(m["scope_rows"], "overall")
    if overall:
        print(f"[calibration_audit] model  : Brier={_f(overall['model_brier'])} log-loss={_f(overall['model_logloss'])} "
              f"(windows={overall['n_windows']})")
        print(f"[calibration_audit] market : Brier={_f(overall['market_brier'])} log-loss={_f(overall['market_logloss'])}")
    n_low = sum(1 for r in m["scope_rows"] if r["scope"] in ("series", "tau_bucket", "series_tau") and r["low_sample"])
    if n_low:
        print(f"[calibration_audit] ⚠ {n_low} breakdown(s) below {args.min_windows} windows — flagged low_sample.")
    print(f"[calibration_audit] verdict: {m['verdict']}")
    print(f"[calibration_audit] wrote {paths.out_dir / 'calibration_audit'} "
          "(metrics.json, scores.csv, reliability.csv, report.html)")

    if c["resolved_windows"] == 0 or c["joined_snapshots"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
