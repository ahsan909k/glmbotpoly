"""Stage — short_horizon.

A companion to :mod:`model_lab.dataset` focused on **short horizons**: dense-grid,
window-aligned samples whose labels are the direction of the Binance mid **10 and
15 seconds** ahead (``fwd_up_10s`` / ``fwd_up_15s``). Like ``dataset`` it is built
over both sources — the journal Chainlink windows we recorded live and the full
Binance ``aggTrades`` historical proxy — and it reuses the *same* window alignment,
as-of feature kernel (incl. the basis-corrected ``z``), and streaming month-by-month
memory discipline. Only the labels, the per-sample coverage marks, and the reporting
are new; the feature functions are the exact ones ``dataset`` uses, so the existing
no-look-ahead AST scan already covers them.

**Two forward labels.** For each sample at ``sample_ts_ms`` the label is the sign of
``log(mid(t+h) / mid(t))`` for ``h ∈ {10, 15}`` seconds (ties → Up, matching the
venue's ``end ≥ strike ⇒ Up`` rule; NA where the ``+h`` second is unobserved). These
are the only new future-reading functions and are kept out of the feature path.

**Per-sample coverage.** Two kinds. *Recording provenance* (``depth_covered`` /
``book_covered``): whether **our own** Binance ``depth20@100ms`` / Polymarket
``top_of_book`` capture recorded a frame within a tolerance of the sample — symmetric
(±tol) and up-OR-down token, so they may reference a timestamp a hair after the sample;
**not** feature columns, exempt from the no-look-ahead rule (like ``dataset``'s
``in_live_coverage``). *Feature presence* (``depth_feat_covered`` / ``pm_feat_covered``):
whether the microstructure features are **actually present** for this row — a causal
backward as-of bar (≤ ``sample_ts``) within staleness matched. **These are the flags to
filter training rows on** — ``pm_feat_covered`` (Up-token) is exactly ``pm_mid`` etc.
present, whereas ``book_covered`` (up-OR-down, ±tol) is not. Historical ``binance_proxy``
windows predate our captures, so all four flags are ``False`` there.

**Hard rule — no look-ahead.** Identical to ``dataset``: no ``feature`` value is
derived from data after a sample's ``sample_ts_ms``. Enforced by (a) the runtime
assertion :func:`_assert_no_lookahead`, (b) the AST source scan in
``tests/test_dataset.py`` (the shared feature functions) extended with the new label
names, and (c) a differential perturbation test over ``FEATURE_COLS`` in
``tests/test_short_horizon.py`` (both sources), plus a positive test that the labels
genuinely react to the future.

Self-contained: reads the gzip journal directly (no prerequisite ``ingest`` run).
Outputs ``out/short_horizon.parquet`` + ``out/short_horizon/{SCHEMA.md, metadata.json}``.

Run::

    python -m model_lab.short_horizon
    python -m model_lab.short_horizon --grid-secs 5 --series BTC-5m,ETH-5m
    python -m model_lab.short_horizon --grid-secs 1        # maximal density
    python -m model_lab.short_horizon --no-history         # journal-only (skip aggTrades)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import dataset as ds
from . import features as feat
from .config import Paths, resolve_bounds, resolve_paths, stage_parser
from .io import binance_archive as ba
from .io import depth as depth_io
from .io import journal as jio
from .ingest import DEPTH_COLS
from .lib import math as lm
from .lib import procmem

# --- defaults ---------------------------------------------------------------
DEFAULT_GRID_SECS = 5  # dense (60 samples / 5-minute window); operator-chosen default.
DEFAULT_HORIZONS = (10, 15)  # forward-direction label horizons, seconds.
DEFAULT_VAL_FRACTION = ds.DEFAULT_VAL_FRACTION
DEFAULT_STRIKE_LOOKBACK_SECS = ds.DEFAULT_STRIKE_LOOKBACK_SECS
DEFAULT_STRIKE_TOLERANCE_SECS = ds.DEFAULT_STRIKE_TOLERANCE_SECS
DEFAULT_MAX_FEATURE_STALENESS_SECS = ds.DEFAULT_MAX_FEATURE_STALENESS_SECS
# A sample is "covered" by a recording within this gap of its timestamp. Feeds tick
# ~1/s (depth ~10/s), so a small symmetric window is the honest membership test.
DEFAULT_COVERAGE_TOLERANCE_SECS = 2.0

MS_PER_DAY = 86_400_000

# --- microstructure feature knobs (single source of truth) ------------------
# Adding a level-depth or a staleness lookback is a one-line edit here — it flows
# into the column set, parquet schema, SCHEMA.md, and the sanity report.
DEPTH_LEVEL_DEPTHS = (1, 5, 10, 20)   # cumulative-size level counts for multi-level book imbalance
DEPTH_SLOPE_LEVELS = 10                # top levels used for the bid/ask depth slope
STALENESS_LOOKBACKS_SECS = (1, 2, 3)   # Binance-move-vs-PM-mid-change divergence spans (seconds)


def _depth_feature_spec() -> list[tuple[str, str, str, str]]:
    """The new Binance ``depth20`` features (per asset), expanded from the knobs."""
    specs: list[tuple[str, str, str, str]] = []
    for lv in DEPTH_LEVEL_DEPTHS:
        specs.append((
            f"depth_imb_{lv}", "float64", "feature",
            f"Binance book imbalance over the top {lv} level(s): "
            f"`(Σbid_sz − Σask_sz)/(Σbid_sz + Σask_sz)` ∈ [−1, 1], as-of "
            f"(NaN where depth uncovered; always NaN for `binance_proxy`).",
        ))
    specs += [
        ("microprice_gap", "float64", "feature",
         "Binance size-weighted microprice minus mid (USD), as-of — book pressure toward the heavier top side."),
        ("bid_depth_slope", "float64", "feature",
         f"Cumulative bid size per USD below the best bid over the top {DEPTH_SLOPE_LEVELS} levels "
         f"(shares/USD), as-of; NaN with < 2 levels."),
        ("ask_depth_slope", "float64", "feature",
         f"Cumulative ask size per USD above the best ask over the top {DEPTH_SLOPE_LEVELS} levels "
         f"(shares/USD), as-of; NaN with < 2 levels."),
        ("depth_spread_bps", "float64", "feature",
         "Binance spread in basis points: `1e4·(best_ask − best_bid)/mid`, as-of."),
    ]
    return specs


def _pm_feature_spec() -> list[tuple[str, str, str, str]]:
    """The new Polymarket Up-token book features, expanded from the knobs."""
    specs: list[tuple[str, str, str, str]] = [
        ("pm_mid", "float64", "feature",
         "Polymarket Up-token mid `(bid+ask)/2` ∈ [0, 1], as-of (up-token `top_of_book`; "
         "NaN where uncovered; always NaN for `binance_proxy`)."),
        ("pm_spread", "float64", "feature",
         "Polymarket Up-token spread `ask − bid` (probability units), as-of."),
        ("pm_book_imb", "float64", "feature",
         "Polymarket Up-token top-of-book imbalance `(bid_sz − ask_sz)/(bid_sz + ask_sz)` ∈ [−1, 1], as-of."),
    ]
    for lb in STALENESS_LOOKBACKS_SECS:
        specs.append((
            f"pm_staleness_{lb}s", "float64", "feature",
            f"Staleness signal: Binance mid log-move over the last {lb}s minus the PM mid change over "
            f"the same span (fast-feed-vs-market divergence), as-of. Positive ⇒ Binance moved up more "
            f"than the PM book followed.",
        ))
    return specs


_DEPTH_FEATURE_SPEC = _depth_feature_spec()
_PM_FEATURE_SPEC = _pm_feature_spec()
_DEPTH_FEATURE_NAMES = [c[0] for c in _DEPTH_FEATURE_SPEC]
_PM_FEATURE_NAMES = [c[0] for c in _PM_FEATURE_SPEC]


# --- output schema ----------------------------------------------------------
# Single source of truth for the column set, dtypes, category, and the SCHEMA.md
# doc. Categories: "id" / "meta" (both known at sample time) · "feature" (as-of ≤
# sample_ts, the SAME columns dataset builds) · "label" (reads the future) ·
# "coverage" (provenance metadata; may reference a recording just after the sample,
# so NOT a feature and exempt from the no-look-ahead rule).
COLUMN_SPEC: list[tuple[str, str, str, str]] = [
    ("series", "string", "id", "Series key, e.g. `BTC-5m`."),
    ("asset", "string", "id", "Underlying asset, `btc` / `eth` (parsed from the series)."),
    ("window_open_ms", "int64", "id", "Window open time (unix ms), a real 5-minute boundary."),
    ("window_close_ms", "int64", "id", "Window close / resolution time (unix ms)."),
    ("sample_ts_ms", "int64", "id", "The feature timestamp — this grid point (unix ms)."),
    ("sample_idx", "int32", "meta", "Grid index `k` within the window (0-based)."),
    ("elapsed_secs", "float64", "meta", "`(sample_ts - open)/1000` — seconds since open (≥ 0)."),
    ("tau_secs", "float64", "meta", "`(close - sample_ts)/1000` — seconds remaining (> 0); known at sample time."),
    ("strike", "float64", "meta", "Reconstructed price-to-beat at open (as-of open). Chainlink for `chainlink` rows, Binance last-trade for `binance_proxy` rows."),
    ("strike_ts_ms", "Int64", "meta", "`ts` of the tick used for the strike (≤ open); NA if none."),
    ("strike_quality", "string", "meta", "`boundary` (≤ tol before open) / `distant` / `missing`."),
    ("label_source", "string", "meta", "`chainlink` (journal-covered, real resolution) or `binance_proxy` (historical, Binance strike/outcome). Known at build time."),
    ("feature_asof_ts_ms", "Int64", "meta", "`ts_ms` of the feature bar the features came from (≤ sample_ts); NA if none or too stale."),
    ("mid", "float64", "feature", "Binance direct mid at the as-of bar (USD)."),
    ("ret", "float64", "feature", "1-second log return of the mid at the as-of bar."),
    ("realized_vol", "float64", "feature", "Practical EWMA std of 1s returns (halflife 60s), as-of."),
    ("sigma_1s", "float64", "feature", "Engine σ_1s EWMA (`crates/model::vol`), as-of; NaN during warmup."),
    ("chainlink", "float64", "feature", "Chainlink price (LOCF ≤ sample), USD; NaN for `binance_proxy` rows (no Chainlink feed)."),
    ("basis_bps", "float64", "feature", "`1e4·(chainlink/mid − 1)` — instantaneous Chainlink-minus-Binance basis, as-of; NaN for `binance_proxy` rows."),
    ("basis_ewma", "float64", "feature", "Rolling log basis `EWMA(ln(chainlink/mid))` (halflife 120s), as-of. Added to `log_s_k` in `z`; NaN ⇒ zero correction for `binance_proxy` rows (same feed)."),
    ("log_s_k", "float64", "feature", "Raw moneyness `log(mid/strike)` — mid as-of, strike from open (basis correction applied in `z`, not here)."),
    ("z", "float64", "feature", "Basis-corrected `(log_s_k + basis_ewma) / (σ_1s·√τ)`, clipped ±39 (all inputs as-of). One consistent definition on both sources."),
    ("p_up_model", "float64", "feature", "Reconstructed engine `p_up = Φ(z)` (as-of)."),
    ("imbalance", "float64", "feature", "Order-book imbalance from depth, as-of (NaN if no depth; always NaN for `binance_proxy`)."),
    ("microprice", "float64", "feature", "Size-weighted microprice from depth, as-of."),
    ("spread", "float64", "feature", "Best ask − best bid from depth, as-of."),
    ("depth_mid", "float64", "feature", "Depth mid from the Binance L2 book, as-of."),
    # New multi-level Binance depth20 microstructure features (per asset).
    *_DEPTH_FEATURE_SPEC,
    ("depth_feat_asof_ts_ms", "Int64", "meta", "`ts_ms` of the depth-feature bar the multi-level depth features came from (≤ sample_ts); NA if none / too stale."),
    ("fwd_ret_10s", "float64", "label", "LABEL: log return of the mid 10s later; NaN if unobserved."),
    ("fwd_up_10s", "Int8", "label", "LABEL: 1 iff `fwd_ret_10s ≥ 0` (ties → Up); NA if unobserved."),
    ("fwd_ret_15s", "float64", "label", "LABEL: log return of the mid 15s later; NaN if unobserved."),
    ("fwd_up_15s", "Int8", "label", "LABEL: 1 iff `fwd_ret_15s ≥ 0` (ties → Up); NA if unobserved."),
    ("outcome", "string", "label", "LABEL (secondary): window final outcome `Up`/`Down`."),
    ("outcome_up", "int8", "label", "LABEL (secondary): 1 iff the window resolved Up."),
    ("fwd_crosses_close_10s", "bool", "meta", "Whether `sample_ts + 10s` reaches past window close (known at sample time)."),
    ("fwd_crosses_close_15s", "bool", "meta", "Whether `sample_ts + 15s` reaches past window close (known at sample time)."),
    ("depth_covered", "bool", "coverage", "Whether our Binance `depth20@100ms` recording has a frame within the coverage tolerance of `sample_ts` (recording provenance, symmetric ±tol; always False for `binance_proxy`)."),
    ("book_covered", "bool", "coverage", "Whether our Polymarket `top_of_book` recording (up OR down token) has a frame within the coverage tolerance of `sample_ts` (recording provenance, symmetric ±tol; always False for `binance_proxy`). NOTE: not the same as `pm_feat_covered` — use that to filter training rows on PM-feature presence."),
    ("depth_feat_covered", "bool", "coverage", "Whether the multi-level **depth features are actually present** for this row — a Binance `depth20` bar matched at-or-before `sample_ts` within staleness (`depth_feat_asof_ts_ms` not NA) ⟺ `depth_imb_1` etc. present. The flag to filter training rows on (`bid`/`ask_depth_slope` may still be NaN on a <2-level book)."),
    # New Polymarket Up-token book microstructure features (per window Up token).
    *_PM_FEATURE_SPEC,
    ("pm_asof_ts_ms", "Int64", "meta", "`ts_ms` of the Up-token `top_of_book` bar the PM features came from (≤ sample_ts); NA if none / too stale."),
    ("pm_run_secs", "Int64", "meta", "Seconds since the matched Up-token book run started; `pm_staleness_Ns` is (expectedly) NaN while this is < N. NA if no PM bar."),
    ("pm_feat_covered", "bool", "coverage", "Whether the **PM book features are actually present** for this row — an Up-token `top_of_book` bar matched at-or-before `sample_ts` within staleness (`pm_asof_ts_ms` not NA) ⟺ `pm_mid`/`pm_spread`/`pm_book_imb` present. **This is the flag to filter training rows on** (unlike `book_covered`, up-OR-down ±tol provenance); `pm_staleness_*` may still be NaN in the first N seconds of a book run (`pm_run_secs < N`)."),
    ("split", "string", "meta", "Chronological split `train` / `val` (per series, latest windows → val; split independently within each `label_source`)."),
]

COLUMNS = [c[0] for c in COLUMN_SPEC]
DTYPES = {c[0]: c[1] for c in COLUMN_SPEC}
FEATURE_COLS = [c[0] for c in COLUMN_SPEC if c[2] == "feature"]
LABEL_COLS = [c[0] for c in COLUMN_SPEC if c[2] == "label"]
COVERAGE_COLS = [c[0] for c in COLUMN_SPEC if c[2] == "coverage"]

# Invariant: a feature can never also be a label, nor a coverage flag (a bounded
# forward read). Guarded here at import and again at runtime in every short_horizon().
assert not (set(FEATURE_COLS) & set(LABEL_COLS)), "COLUMN_SPEC: feature/label overlap"
assert not (set(FEATURE_COLS) & set(COVERAGE_COLS)), "COLUMN_SPEC: feature/coverage overlap"

_ARROW_TYPES = {
    "string": pa.string(), "int64": pa.int64(), "int32": pa.int32(), "int8": pa.int8(),
    "Int64": pa.int64(), "Int8": pa.int8(), "float64": pa.float64(), "bool": pa.bool_(),
}


def _arrow_schema() -> pa.Schema:
    """The parquet schema derived from ``COLUMN_SPEC`` (for streamed writes)."""
    return pa.schema([(name, _ARROW_TYPES[dtype]) for name, dtype, _cat, _doc in COLUMN_SPEC])


# --- forward labels (read the future — kept apart from the feature path) -----
def forward_direction_labels_multi(
    grid: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """Per-asset forward direction of the mid at each horizon in ``horizons``, on a
    gap-free 1-second grid. Returns ``asset, sec, fwd_ret_{h}s…`` where
    ``fwd_ret_{h}s = log(mid.shift(-h) / mid)``.

    This is a **label** function — it deliberately reads strictly *later* ticks (a
    negative shift) and is excluded from the no-look-ahead AST scan (which only scans
    the feature functions)."""
    ret_cols = [f"fwd_ret_{h}s" for h in horizons]
    if grid.empty or "mid" not in grid.columns:
        return pd.DataFrame(columns=["asset", "sec", *ret_cols])
    frames: list[pd.DataFrame] = []
    for asset, g in grid.groupby("asset"):
        sel = g[["sec", "mid"]].dropna().drop_duplicates("sec").sort_values("sec")
        if sel.empty:
            continue
        lo, hi = int(sel["sec"].min()), int(sel["sec"].max())
        gg = sel.set_index("sec").reindex(range(lo, hi + 1))
        gg["mid"] = gg["mid"].ffill()
        for h in horizons:
            fwd_mid = gg["mid"].shift(-h)
            gg[f"fwd_ret_{h}s"] = np.log(fwd_mid / gg["mid"])
        gg = gg.rename_axis("sec").reset_index()
        # Keep only seconds that actually had an observation (a real grid bar).
        out = gg[gg["sec"].isin(sel["sec"])][["sec", *ret_cols]].copy()
        out["asset"] = asset
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["asset", "sec", *ret_cols])
    return pd.concat(frames, ignore_index=True)


def _attach_labels(
    samples: pd.DataFrame, fwd: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """Join the forward-direction labels by exact ``(asset, sample_sec)`` and set the
    ``fwd_up_{h}s`` (ties → Up, NA where unobserved) + ``fwd_crosses_close_{h}s`` flags."""
    out = samples.copy()
    out["sample_sec"] = out["sample_ts_ms"] // 1000
    out = out.merge(
        fwd.rename(columns={"sec": "sample_sec"}), on=["asset", "sample_sec"], how="left"
    )
    for h in horizons:
        rc = f"fwd_ret_{h}s"
        out[f"fwd_up_{h}s"] = (out[rc] >= 0.0).astype("Int8").mask(out[rc].isna(), pd.NA)
        out[f"fwd_crosses_close_{h}s"] = (out["sample_ts_ms"] + h * 1000) >= out["window_close_ms"]
    return out.drop(columns=["sample_sec"])


# --- per-sample coverage (provenance metadata — NOT a feature) --------------
def _covered(sorted_ts: np.ndarray, targets: np.ndarray, tol_ms: int) -> np.ndarray:
    """Vectorized membership: for each ``t`` in ``targets``, is there a recorded
    timestamp in ``sorted_ts`` within ``±tol_ms`` of ``t``? (The first ts ≥ ``t−tol``,
    if any, must be ≤ ``t+tol``.) Mirrors ``dataset._coverage_flags`` but per-sample."""
    out = np.zeros(len(targets), dtype=bool)
    if len(sorted_ts) == 0 or len(targets) == 0:
        return out
    lo = targets - tol_ms
    hi = targets + tol_ms
    idx = np.searchsorted(sorted_ts, lo, side="left")
    valid = idx < len(sorted_ts)
    out[valid] = sorted_ts[idx[valid]] <= hi[valid]
    return out


def _mark_coverage(
    samples: pd.DataFrame, depth_ts_by_asset: dict, top_ts_by_token: dict, tol_ms: int
) -> pd.DataFrame:
    """Add ``depth_covered`` / ``book_covered`` per sample. ``binance_proxy`` rows →
    both False (our captures predate the historical archive). For live (``chainlink``)
    rows, ``depth_covered`` from the asset's sorted depth ``recv_ms`` and
    ``book_covered`` from the up-OR-down token's sorted ``top_of_book`` ``ts``."""
    n = len(samples)
    depth_cov = np.zeros(n, dtype=bool)
    book_cov = np.zeros(n, dtype=bool)
    if n:
        live = (samples["label_source"] != "binance_proxy").to_numpy()
        ts = samples["sample_ts_ms"].to_numpy(dtype="int64")
        assets = samples["asset"].to_numpy(dtype=object)
        for a in pd.unique(assets[live]):
            arr = depth_ts_by_asset.get(a)
            if arr is None or len(arr) == 0:
                continue
            m = live & (assets == a)
            depth_cov[m] = _covered(arr, ts[m], tol_ms)
        for col in ("up_token", "down_token"):
            if col not in samples.columns:
                continue
            toks = samples[col].to_numpy(dtype=object)
            for tok in pd.unique(toks[live]):
                if tok is None or (isinstance(tok, float) and np.isnan(tok)):
                    continue
                arr = top_ts_by_token.get(tok)
                if arr is None or len(arr) == 0:
                    continue
                m = live & (toks == tok)
                book_cov[m] |= _covered(arr, ts[m], tol_ms)
    samples = samples.copy()
    samples["depth_covered"] = depth_cov
    samples["book_covered"] = book_cov
    return samples


# --- microstructure features (as-of; causal — same no-look-ahead rule) -------
# All computed ONLY where our recordings cover the timestamp; missing-coverage rows
# get honest NaN, never build-time imputation. Depth is per asset (from our Binance
# depth20 capture); PM book is per window Up token (from our Polymarket top_of_book
# capture). Both attach via a backward merge_asof (bar ts_ms ≤ sample_ts).
def _side_slope(px: np.ndarray, cum_sz: np.ndarray, k_levels: int, *, is_bid: bool) -> float:
    """Cumulative size over the top ``k`` levels per USD of price distance from the
    best (shares/USD). NaN with < 2 levels (no distance). Depends on a single book
    snapshot — no time dimension, so trivially causal."""
    k = min(k_levels, px.size)
    if k < 2:
        return float("nan")
    best, edge = px[0], px[k - 1]
    dist = (best - edge) if is_bid else (edge - best)
    if dist <= 0.0:
        return float("nan")
    return float(cum_sz[k - 1] / dist)


def _frame_depth_features(frame: dict) -> dict:
    """Multi-level imbalance / microprice-gap / depth slopes / spread-bps from ONE
    raw ``depth20`` frame (a self-contained book snapshot). Uses ``min(L, n_levels)``
    so a thin book yields honest NaN rather than imputed depth."""
    bid_px, bid_sz = frame["bid_px"], frame["bid_sz"]
    ask_px, ask_sz = frame["ask_px"], frame["ask_sz"]
    best_bid, best_ask = float(bid_px[0]), float(ask_px[0])
    mid = 0.5 * (best_bid + best_ask)
    cum_bid, cum_ask = np.cumsum(bid_sz), np.cumsum(ask_sz)
    out: dict = {"recv_ms": int(frame["recv_ms"]), "asset": frame["asset"]}
    for lv in DEPTH_LEVEL_DEPTHS:
        nb, na = min(lv, bid_sz.size), min(lv, ask_sz.size)
        b = float(cum_bid[nb - 1]) if nb else 0.0
        a = float(cum_ask[na - 1]) if na else 0.0
        denom = b + a
        out[f"depth_imb_{lv}"] = (b - a) / denom if denom > 0.0 else float("nan")
    top_denom = float(bid_sz[0] + ask_sz[0])
    microprice = (
        (best_bid * float(ask_sz[0]) + best_ask * float(bid_sz[0])) / top_denom
        if top_denom > 0.0 else mid
    )
    out["microprice_gap"] = microprice - mid
    out["bid_depth_slope"] = _side_slope(bid_px, cum_bid, DEPTH_SLOPE_LEVELS, is_bid=True)
    out["ask_depth_slope"] = _side_slope(ask_px, cum_ask, DEPTH_SLOPE_LEVELS, is_bid=False)
    out["depth_spread_bps"] = 1e4 * (best_ask - best_bid) / mid if mid > 0.0 else float("nan")
    return out


def build_depth_feature_grid(frames, asset: str) -> pd.DataFrame:
    """Per-second (as-of) grid of the multi-level depth features for one asset: the
    **last** book snapshot in each second (mirrors ``features._depth_bars``; ``ts_ms
    = sec·1000``). Every value at second ``s`` comes from a snapshot within second
    ``s`` — no look-ahead."""
    cols = ["asset", "sec", "ts_ms", *_DEPTH_FEATURE_NAMES]
    rows = [_frame_depth_features(f) for f in frames if f["asset"] == asset]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["sec"] = df["recv_ms"].astype("int64") // 1000
    df = df.sort_values("recv_ms").drop_duplicates("sec", keep="last")
    df["asset"] = asset
    df["ts_ms"] = df["sec"] * 1000
    return df[cols].reset_index(drop=True)


def _read_pm_books(
    paths: Paths, tokens: set, since_ms: int | None = None, until_ms: int | None = None
) -> pd.DataFrame:
    """A second, **filtered streaming** pass over the journal decoding the Polymarket
    ``top_of_book`` bid/ask price+size for the in-scope Up ``tokens`` only (bounded
    memory — avoids holding every book payload for the multi-GB live journal).
    One-sided books are skipped; ``since_ms``/``until_ms`` bound by book ts. Returns
    ``[token_id, ts, bid_px, bid_sz, ask_px, ask_sz]``."""
    cols = ["token_id", "ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]
    if not tokens:
        return pd.DataFrame(columns=cols)
    rows: list[tuple] = []
    for rec in jio.records_of_type(paths.journal_dir, "top_of_book"):
        token = rec.get("token_id")
        if token not in tokens:
            continue
        top = rec.get("top") or {}
        bid, ask, ts = top.get("bid"), top.get("ask"), top.get("ts")
        if not isinstance(bid, dict) or not isinstance(ask, dict) or ts is None:
            continue  # one-sided book: no mid/spread/imbalance
        if not ds._in_bounds(int(ts), since_ms, until_ms):
            continue
        rows.append((token, int(ts), jio.to_float(bid.get("price")), jio.to_float(bid.get("size")),
                     jio.to_float(ask.get("price")), jio.to_float(ask.get("size"))))
    return pd.DataFrame(rows, columns=cols)


def build_pm_grid(pm_df: pd.DataFrame, mid_by_asset: dict, token_asset: dict) -> pd.DataFrame:
    """Per-Up-token, per-second (as-of) grid of the PM book features. ``pm_mid /
    pm_spread / pm_book_imb`` are the last book in each second; the staleness signals
    (``STALENESS_LOOKBACKS_SECS``) compare the Binance mid log-move to the PM mid change
    over the same 1..N-second span, both via **positive** shifts on a per-token
    contiguous 1s reindex (causal — each value at second ``s`` uses only observations at
    seconds ``≤ s``)."""
    lookbacks = STALENESS_LOOKBACKS_SECS
    cols = ["up_token", "sec", "ts_ms", "pm_run_secs", *_PM_FEATURE_NAMES]
    if pm_df.empty:
        return pd.DataFrame(columns=cols)
    frames: list[pd.DataFrame] = []
    for token, g in pm_df.groupby("token_id"):
        g = g.copy()
        g["sec"] = g["ts"].astype("int64") // 1000
        g["pm_mid"] = 0.5 * (g["bid_px"] + g["ask_px"])
        g["pm_spread"] = g["ask_px"] - g["bid_px"]
        denom = g["bid_sz"] + g["ask_sz"]
        g["pm_book_imb"] = np.where(denom > 0.0, (g["bid_sz"] - g["ask_sz"]) / denom, np.nan)
        obs = (g.sort_values("ts").drop_duplicates("sec", keep="last")
               [["sec", "pm_mid", "pm_spread", "pm_book_imb"]].set_index("sec"))
        lo, hi = int(obs.index.min()), int(obs.index.max())
        full = obs.reindex(range(lo, hi + 1))
        bmid = mid_by_asset.get(token_asset.get(token))
        if bmid is not None and not bmid.empty:
            bser = (bmid.drop_duplicates("sec").set_index("sec")["mid"]
                    .reindex(range(lo, hi + 1)).ffill())
        else:
            bser = pd.Series(np.nan, index=full.index)
        pm_ff = full["pm_mid"].ffill()
        with np.errstate(divide="ignore", invalid="ignore"):
            for lb in lookbacks:
                binance_move = np.log(bser / bser.shift(lb))
                pm_change = pm_ff - pm_ff.shift(lb)
                full[f"pm_staleness_{lb}s"] = binance_move - pm_change
        # Seconds since this token's book run started — the staleness lookback needs
        # this many prior seconds, so `pm_run_secs < N` is exactly where pm_staleness_Ns
        # is legitimately NaN (its warmup edge). Classifies expected NaN in the sanity report.
        full["pm_run_secs"] = full.index.to_numpy(dtype="int64") - lo
        full = full.loc[obs.index].reset_index()  # keep only observed PM seconds
        full["up_token"] = token
        full["ts_ms"] = full["sec"] * 1000
        frames.append(full)
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def _attach_asof(
    samples: pd.DataFrame, grid: pd.DataFrame, by: str, asof_col: str,
    value_cols: list[str], max_staleness_ms: int | None, carry_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Bind each sample to its as-of microstructure bar with a **backward**
    ``merge_asof`` (matched bar ``ts_ms ≤ sample_ts_ms``), keyed by ``by`` (asset for
    depth, up_token for PM). Empty grid or a missing key ⇒ all-NaN + NA asof. When
    the matched bar is older than ``max_staleness_ms`` the features are nulled and
    the asof set NA — strictly more conservative, never a forward read. ``carry_cols``
    are extra (integer) diagnostic columns carried from the grid (nulled when stale)."""
    out = samples.copy()
    all_cols = [*value_cols, *carry_cols]
    if grid.empty or by not in out.columns:
        for col in value_cols:
            out[col] = np.nan
        for col in carry_cols:
            out[col] = pd.array([pd.NA] * len(out), dtype="Int64")
        out[asof_col] = pd.array([pd.NA] * len(out), dtype="Int64")
        return out
    right = (grid.rename(columns={"ts_ms": asof_col})[[by, asof_col, *all_cols]]
             .sort_values(asof_col))
    left = out.sort_values("sample_ts_ms")
    merged = pd.merge_asof(
        left, right, left_on="sample_ts_ms", right_on=asof_col, by=by, direction="backward"
    )
    merged[asof_col] = merged[asof_col].astype("Int64")
    if max_staleness_ms is not None:
        stale = ((merged["sample_ts_ms"] - merged[asof_col]) > max_staleness_ms).fillna(False)
        merged.loc[stale, all_cols] = np.nan
        merged.loc[stale, asof_col] = pd.NA
    for col in carry_cols:
        merged[col] = merged[col].astype("Int64")
    return merged


def _attach_depth_features(samples, depth_grid, max_staleness_ms) -> pd.DataFrame:
    return _attach_asof(samples, depth_grid, "asset", "depth_feat_asof_ts_ms",
                        _DEPTH_FEATURE_NAMES, max_staleness_ms)


def _attach_pm_features(samples, pm_grid, max_staleness_ms) -> pd.DataFrame:
    return _attach_asof(samples, pm_grid, "up_token", "pm_asof_ts_ms",
                        _PM_FEATURE_NAMES, max_staleness_ms, carry_cols=("pm_run_secs",))


def _empty_depth_grid() -> pd.DataFrame:
    return pd.DataFrame(columns=["asset", "sec", "ts_ms", *_DEPTH_FEATURE_NAMES])


def _empty_pm_grid() -> pd.DataFrame:
    return pd.DataFrame(columns=["up_token", "sec", "ts_ms", "pm_run_secs", *_PM_FEATURE_NAMES])


# --- sampling ---------------------------------------------------------------
def _sample_points(windows_df: pd.DataFrame, grid_secs: int) -> pd.DataFrame:
    """Emit ``sample_ts_ms = open + k·grid_secs·1000`` for ``k = 0,1,…`` while
    ``< close``, then rejoin the window-level fields (incl. the tokens, which the
    per-sample book coverage needs — dataset's sampler drops them)."""
    step = grid_secs * 1000
    rows: list[tuple] = []
    for w in windows_df.itertuples(index=False):
        open_ms, close_ms = int(w.window_open_ms), int(w.window_close_ms)
        k, ts = 0, open_ms
        while ts < close_ms:
            rows.append((w.series, w.asset, open_ms, ts, k))
            k += 1
            ts = open_ms + k * step
    samples = pd.DataFrame(
        rows, columns=["series", "asset", "window_open_ms", "sample_ts_ms", "sample_idx"]
    )
    if samples.empty:
        return samples
    samples["sample_idx"] = samples["sample_idx"].astype("int32")
    keep = ["series", "asset", "window_open_ms", "window_close_ms", "strike", "strike_ts_ms",
            "strike_quality", "label_source", "outcome", "outcome_up", "split",
            "up_token", "down_token"]
    samples = samples.merge(windows_df[keep], on=["series", "asset", "window_open_ms"], how="left")
    samples["elapsed_secs"] = (samples["sample_ts_ms"] - samples["window_open_ms"]) / 1000.0
    samples["tau_secs"] = (samples["window_close_ms"] - samples["sample_ts_ms"]) / 1000.0
    return samples


# --- the hard rule: runtime assertion --------------------------------------
def _assert_no_lookahead(df: pd.DataFrame) -> None:
    """Fail-closed if any sample's features could have used data after its timestamp.

    Mirrors ``dataset._assert_no_lookahead`` but binds the feature/label disjointness
    to *this* stage's column sets. Coverage columns are deliberately untouched — they
    are provenance metadata, not features."""
    if df.empty:
        return
    fa = df["feature_asof_ts_ms"].notna()
    if fa.any():
        assert (
            df.loc[fa, "feature_asof_ts_ms"].astype("int64")
            <= df.loc[fa, "sample_ts_ms"].astype("int64")
        ).all(), "no-lookahead: a feature bar is AFTER its sample"
    assert (df["window_open_ms"] <= df["sample_ts_ms"]).all(), "no-lookahead: sample precedes window open"
    assert (df["sample_ts_ms"] < df["window_close_ms"]).all(), "no-lookahead: sample at/after window close"
    st = df["strike_ts_ms"].notna()
    if st.any():
        assert (
            df.loc[st, "strike_ts_ms"].astype("int64")
            <= df.loc[st, "window_open_ms"].astype("int64")
        ).all(), "no-lookahead: strike sampled after window open"
    assert (df["tau_secs"] > 0.0).all(), "no-lookahead: non-positive tau_secs"
    assert (df["elapsed_secs"] >= 0.0).all(), "no-lookahead: negative elapsed_secs"
    for asof_col in ("depth_feat_asof_ts_ms", "pm_asof_ts_ms"):
        if asof_col in df.columns:
            m = df[asof_col].notna()
            if m.any():
                assert (
                    df.loc[m, asof_col].astype("int64") <= df.loc[m, "sample_ts_ms"].astype("int64")
                ).all(), f"no-lookahead: {asof_col} is AFTER its sample"
    assert not (set(FEATURE_COLS) & set(LABEL_COLS)), "no-lookahead: feature/label column overlap"
    assert not (set(FEATURE_COLS) & set(COVERAGE_COLS)), "no-lookahead: feature/coverage column overlap"


def _finalize(samples: pd.DataFrame) -> pd.DataFrame:
    """Select the COLUMN_SPEC columns in order and coerce to the declared dtypes."""
    df = samples.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMNS]
    for col, dtype in DTYPES.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Leave nullable/float columns that legitimately carry NaN as-is.
            pass
    return df.reset_index(drop=True)


# --- shared sampling path ---------------------------------------------------
def _build_samples(
    windows_df: pd.DataFrame, grid: pd.DataFrame, grid_secs: int, horizons: tuple[int, ...],
    max_staleness_ms: int, depth_feat_grid: pd.DataFrame, pm_grid: pd.DataFrame,
    depth_ts_by_asset: dict, top_ts_by_token: dict, tol_ms: int,
) -> pd.DataFrame:
    """Windows + feature grid → finalized sample rows (sampling → as-of price/vol
    features → as-of depth + PM microstructure features → forward labels → coverage).
    The single path, shared by the journal and historical batches so both get
    identical no-look-ahead treatment. Historical (``binance_proxy``) batches pass
    empty depth/PM grids, so the microstructure features are NaN by construction."""
    samples = _sample_points(windows_df, grid_secs)
    if samples.empty:
        return _finalize(samples)
    samples = ds._attach_features(samples, grid, max_staleness_ms)
    samples = _attach_depth_features(samples, depth_feat_grid, max_staleness_ms)
    samples = _attach_pm_features(samples, pm_grid, max_staleness_ms)
    fwd = forward_direction_labels_multi(grid, horizons)
    samples = _attach_labels(samples, fwd, horizons)
    samples = _mark_coverage(samples, depth_ts_by_asset, top_ts_by_token, tol_ms)
    # Feature-presence flags: "covered" here means the microstructure features are
    # actually present (a bar matched at-or-before the sample within staleness), the
    # signal to filter training rows on — distinct from the ±tol recording-provenance
    # `depth_covered`/`book_covered`. Derived from the causal as-of columns (≤ sample_ts).
    samples["depth_feat_covered"] = samples["depth_feat_asof_ts_ms"].notna()
    samples["pm_feat_covered"] = samples["pm_asof_ts_ms"].notna()
    return _finalize(samples)


# --- worker -----------------------------------------------------------------
def short_horizon(
    paths: Paths,
    *,
    grid_secs: int = DEFAULT_GRID_SECS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    series: tuple[str, ...] | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    strike_lookback_secs: int = DEFAULT_STRIKE_LOOKBACK_SECS,
    strike_tolerance_secs: float = DEFAULT_STRIKE_TOLERANCE_SECS,
    max_feature_staleness_secs: int = DEFAULT_MAX_FEATURE_STALENESS_SECS,
    coverage_tolerance_secs: float = DEFAULT_COVERAGE_TOLERANCE_SECS,
    include_history: bool = True,
    since_ms: int | None = None,
    until_ms: int | None = None,
    stream: bool = True,
) -> dict:
    """Builds the short-horizon dataset — the journal (Chainlink) windows plus, when
    ``include_history`` and an aggTrades store is present, the historical Binance-proxy
    windows streamed **month by month**. Writes ``out/short_horizon.parquet`` +
    ``out/short_horizon/{SCHEMA.md, metadata.json}``; returns ``{params, counts,
    per_day, coverage, series, journal_subset}``. The journal first pass is
    streaming/bounded (``stream``); ``since_ms``/``until_ms`` bound it by time."""
    paths.ensure_out()
    out_dir = paths.out_dir / "short_horizon"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = tuple(int(h) for h in horizons)
    params = {
        "grid_secs": int(grid_secs),
        "horizons": list(horizons),
        "series_filter": list(series) if series else None,
        "val_fraction": float(val_fraction),
        "strike_lookback_secs": int(strike_lookback_secs),
        "strike_tolerance_secs": float(strike_tolerance_secs),
        "max_feature_staleness_secs": int(max_feature_staleness_secs),
        "coverage_tolerance_secs": float(coverage_tolerance_secs),
        "include_history": bool(include_history),
        "since_ms": since_ms,
        "until_ms": until_ms,
        "stream": bool(stream),
    }
    strike_tol_ms = strike_tolerance_secs * 1000.0
    max_staleness_ms = max_feature_staleness_secs * 1000
    cov_tol_ms = int(round(coverage_tolerance_secs * 1000.0))

    # --- journal (Chainlink strike/outcome) — the unchanged path -------------
    ticks_df, win_meta, res, settle, top_ts = ds._read_journal(
        paths, series, since_ms=since_ms, until_ms=until_ms, stream=stream)
    jwindows = ds._windows_frame(win_meta, res, settle)
    journal_keys: set[tuple[str, int]] = (
        set(zip(jwindows["series"], jwindows["window_open_ms"].astype("int64").tolist()))
        if not jwindows.empty else set()
    )
    top_ts_sorted = {tok: np.sort(np.asarray(v, dtype="int64")) for tok, v in top_ts.items()}
    jgrid = pd.DataFrame(columns=[*feat.FEATURE_COLS, "sigma_1s"])
    depth_ts_by_asset: dict[str, np.ndarray] = {}
    depth_feat_grid = _empty_depth_grid()
    pm_grid = _empty_pm_grid()
    if not jwindows.empty:
        strikes = ds.reconstruct_strikes(ticks_df, jwindows, strike_lookback_secs * 1000, strike_tol_ms)
        jwindows = jwindows.merge(strikes, on=["series", "asset", "window_open_ms"], how="left")
        jwindows["split"] = ds._assign_split(jwindows, val_fraction)  # journal split — unchanged
        jwindows["label_source"] = "chainlink"
        depth_df = pd.DataFrame(list(depth_io.read_depth_rows(paths.depth_dir)), columns=DEPTH_COLS)
        if not depth_df.empty:
            depth_ts_by_asset = {
                a: np.sort(g["recv_ms"].to_numpy(dtype="int64"))
                for a, g in depth_df.groupby("asset")
            }
        assets = sorted(jwindows["asset"].unique())
        grids = [ds.build_feature_grid(ticks_df, depth_df, a) for a in assets]
        grids = [g for g in grids if not g.empty]
        if grids:
            jgrid = pd.concat(grids, ignore_index=True)
        # New microstructure grids — depth (multi-level, per asset) and PM book (per
        # Up token). These recordings exist only for windows we captured live, so the
        # historical binance_proxy batches get empty grids (features NaN by construction).
        depth_levels = list(depth_io.read_depth_levels(paths.depth_dir))
        dgrids = [build_depth_feature_grid(depth_levels, a) for a in assets]
        dgrids = [g for g in dgrids if not g.empty]
        if dgrids:
            depth_feat_grid = pd.concat(dgrids, ignore_index=True)
        up_tokens = set(jwindows["up_token"].dropna().tolist())
        pm_df = _read_pm_books(paths, up_tokens, since_ms=since_ms, until_ms=until_ms)
        mid_by_asset = (
            {a: jgrid.loc[jgrid["asset"] == a, ["sec", "mid"]] for a in assets}
            if not jgrid.empty else {}
        )
        token_asset = dict(zip(jwindows["up_token"], jwindows["asset"]))
        pm_grid = build_pm_grid(pm_df, mid_by_asset, token_asset)

    # --- accumulators + streaming writer -------------------------------------
    schema = _arrow_schema()
    writer = pq.ParquetWriter(paths.table("short_horizon"), schema)
    tot: dict[str, int] = defaultdict(int)
    assets_seen: set[str] = set()
    series_seen: set[str] = set()
    per_day_samples: dict[int, int] = defaultdict(int)
    per_day_depth: dict[int, int] = defaultdict(int)
    per_day_book: dict[int, int] = defaultdict(int)
    dedup_collisions = 0

    def _accumulate_per_day(batch: pd.DataFrame) -> None:
        day = batch["sample_ts_ms"].to_numpy(dtype="int64") // MS_PER_DAY
        tmp = pd.DataFrame({
            "day": day,
            "d": batch["depth_covered"].to_numpy(dtype=bool).astype("int64"),
            "b": batch["book_covered"].to_numpy(dtype=bool).astype("int64"),
        })
        agg = tmp.groupby("day").agg(n=("d", "size"), d=("d", "sum"), b=("b", "sum"))
        for day_i, row in agg.iterrows():
            per_day_samples[int(day_i)] += int(row["n"])
            per_day_depth[int(day_i)] += int(row["d"])
            per_day_book[int(day_i)] += int(row["b"])

    def _write_batch(batch: pd.DataFrame, source_label: str) -> None:
        if batch.empty:
            return
        _assert_no_lookahead(batch)  # the hard rule runs over every batch, both sources
        writer.write_table(pa.Table.from_pandas(batch, schema=schema, preserve_index=False))
        tot["samples"] += len(batch)
        tot["train"] += int((batch["split"] == "train").sum())
        tot["val"] += int((batch["split"] == "val").sum())
        tot[f"samp_{source_label}"] += len(batch)
        tot["depth_cov"] += int(batch["depth_covered"].sum())
        tot["book_cov"] += int(batch["book_covered"].sum())
        tot["depth_feat_cov"] += int(batch["depth_feat_covered"].sum())
        tot["pm_feat_cov"] += int(batch["pm_feat_covered"].sum())
        for h in horizons:
            tot[f"missing_fwd_{h}s"] += int(batch[f"fwd_up_{h}s"].isna().sum())
        _accumulate_per_day(batch)

    def _count_windows(wdf: pd.DataFrame, source_label: str) -> None:
        tot["windows"] += len(wdf)
        tot[f"win_{source_label}"] += len(wdf)
        assets_seen.update(wdf["asset"].unique())
        series_seen.update(wdf["series"].unique())

    # journal batch first — its rows are produced by the unchanged path.
    if not jwindows.empty:
        _write_batch(
            _build_samples(jwindows, jgrid, grid_secs, horizons, max_staleness_ms,
                           depth_feat_grid, pm_grid, depth_ts_by_asset, top_ts_sorted, cov_tol_ms),
            "chainlink",
        )
        _count_windows(jwindows, "chainlink")

    # historical batches — one (symbol, calendar month) at a time; proxy windows have
    # no depth/book recordings, so coverage is constant False (empty coverage maps).
    if include_history and ba.has_aggtrades(paths.hist_dir):
        symbols = [s for s in ba.symbols_present(paths.hist_dir) if ds._symbol_in_scope(s, series)]
        hist_opens: dict[str, list[int]] = {}
        for sym in symbols:
            hist_opens.setdefault(ds.SYMBOL_SERIES[sym], []).extend(ds._hist_candidate_opens(paths.hist_dir, sym))
        hist_cutoffs = ds._split_cutoffs(hist_opens, val_fraction)  # split independent of the journal
        for sym in symbols:
            asset = ds.SYMBOL_ASSET[sym]
            for _yr, _mo, load_paths, mdays in ds._month_batches(paths.hist_dir, sym):
                bars = ds._load_month_bars(load_paths, sym)
                if bars.empty:
                    continue
                hwindows, collisions = ds._historical_windows(bars, mdays, sym, journal_keys, strike_tol_ms)
                dedup_collisions += collisions
                if hwindows.empty:
                    continue
                hwindows["split"] = ds._split_by_cutoff(hwindows, hist_cutoffs)
                grid = ds.build_feature_grid(ds._binance_ticks_from_bars(bars, asset),
                                             pd.DataFrame(columns=DEPTH_COLS), asset)
                hbatch = _build_samples(hwindows, grid, grid_secs, horizons, max_staleness_ms,
                                        _empty_depth_grid(), _empty_pm_grid(), {}, {}, cov_tol_ms)
                _write_batch(hbatch, "binance_proxy")
                _count_windows(hwindows, "binance_proxy")

    if tot["samples"] == 0:
        writer.write_table(schema.empty_table())  # a valid, 0-row parquet with the schema
    writer.close()

    total = int(tot["samples"])
    counts = {
        "windows": int(tot["windows"]),
        "samples": total,
        "samples_train": int(tot["train"]),
        "samples_val": int(tot["val"]),
        "windows_chainlink": int(tot["win_chainlink"]),
        "windows_binance_proxy": int(tot["win_binance_proxy"]),
        "samples_chainlink": int(tot["samp_chainlink"]),
        "samples_binance_proxy": int(tot["samp_binance_proxy"]),
        "samples_depth_covered": int(tot["depth_cov"]),
        "samples_book_covered": int(tot["book_cov"]),
        "samples_depth_feat_covered": int(tot["depth_feat_cov"]),
        "samples_pm_feat_covered": int(tot["pm_feat_cov"]),
    }
    for h in horizons:
        counts[f"samples_missing_fwd_{h}s"] = int(tot[f"missing_fwd_{h}s"])

    per_day = [
        {
            "day": datetime.fromtimestamp(d * 86_400, tz=timezone.utc).strftime("%Y-%m-%d"),
            "samples": per_day_samples[d],
            "depth_covered": per_day_depth[d],
            "book_covered": per_day_book[d],
            "depth_pct": (per_day_depth[d] / per_day_samples[d]) if per_day_samples[d] else 0.0,
            "book_pct": (per_day_book[d] / per_day_samples[d]) if per_day_samples[d] else 0.0,
        }
        for d in sorted(per_day_samples)
    ]
    coverage = {
        "coverage_tolerance_secs": float(coverage_tolerance_secs),
        "depth_pct": (counts["samples_depth_covered"] / total) if total else 0.0,
        "book_pct": (counts["samples_book_covered"] / total) if total else 0.0,
        # Feature-present fractions (the filter signal): the fraction of rows whose
        # depth / PM features are actually populated (as-of bar within staleness).
        "depth_feat_pct": (counts["samples_depth_feat_covered"] / total) if total else 0.0,
        "pm_feat_pct": (counts["samples_pm_feat_covered"] / total) if total else 0.0,
        "days": len(per_day),
    }
    journal_subset = {
        "windows": int(tot["win_chainlink"]),
        "samples": int(tot["samp_chainlink"]),
        "dedup_collisions": int(dedup_collisions),
        # The journal batch is produced by the unchanged path and written before any
        # historical batch; historical windows colliding with a journal (series, open)
        # are dropped, so the chainlink subset is never altered by the extension.
        "unchanged": True,
    }
    result = {
        "params": params,
        "counts": counts,
        "coverage": coverage,
        "per_day": per_day,
        "assets": sorted(assets_seen),
        "series": sorted(series_seen),
        "journal_subset": journal_subset,
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    if total > 0:
        sanity = sanity_report(paths.table("short_horizon"), SANITY_SPEC)
        _write_sanity(out_dir, sanity)
        result["sanity_status"] = sanity["overall_status"]
        result["sanity_tally"] = sanity["status_tally"]
    else:
        result["sanity_status"] = "n/a"
        result["sanity_tally"] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    _write_metadata(out_dir, result)
    _write_schema(out_dir)
    return result


# --- outputs ----------------------------------------------------------------
def _write_metadata(out_dir, result: dict) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _write_schema(out_dir) -> None:
    cat_label = {
        "id": "identifier", "meta": "metadata (known at sample time)",
        "feature": "**feature** (as-of ≤ sample_ts)", "label": "**label** (reads the future)",
        "coverage": "**coverage** (provenance metadata; not a feature)",
    }
    lines = [
        "# `short_horizon.parquet` — schema",
        "",
        "One row per (window, feature-timestamp), dense grid. Built by "
        "`python -m model_lab.short_horizon`. Short-horizon companion to `dataset` "
        "(same alignment + feature kernel), with 10s/15s forward labels and per-sample "
        "recording-coverage marks.",
        "",
        "| column | dtype | category | description |",
        "|---|---|---|---|",
    ]
    for name, dtype, category, doc in COLUMN_SPEC:
        lines.append(f"| `{name}` | `{dtype}` | {cat_label[category]} | {doc} |")
    lines += [
        "",
        "## No-look-ahead contract",
        "",
        "The defining invariant: **no value in a `feature` column is derived from any "
        "data after that row's `sample_ts_ms`.** Features are the *same* as-of columns "
        "`dataset` builds (backward `merge_asof`, causal EWMA vol / `sigma_1s`, `strike` "
        "captured at `window_open_ms`, basis-corrected `z`). The **label** columns "
        "(`fwd_ret_10s`, `fwd_up_10s`, `fwd_ret_15s`, `fwd_up_15s`, `outcome`, "
        "`outcome_up`) deliberately read the future and must never be used as inputs.",
        "",
        "Enforced by (a) a runtime assertion in `short_horizon._assert_no_lookahead`, "
        "(b) the AST source scan (`tests/test_dataset.py::test_no_lookahead_ast_scan`) "
        "over the shared feature functions — extended with the 10s/15s label names — and "
        "(c) a differential perturbation test "
        "(`tests/test_short_horizon.py::test_no_lookahead_differential…`) that corrupts "
        "all future data and asserts every `FEATURE_COLS` value is bit-identical, plus a "
        "positive test that the labels genuinely react to `t+10`/`t+15`.",
        "",
        "## Conventions",
        "",
        "- **Labels:** `fwd_up_{h}s = 1` iff `log(mid(t+h)/mid(t)) ≥ 0` (ties → Up), for "
        "`h ∈ {10, 15}` seconds, on a gap-free 1-second mid grid. `NA` where the `+h` "
        "second is unobserved (end of the journal or an internal gap). Price-only, so "
        "proxy-safe on both sources. `outcome`/`outcome_up` (the window resolution) are "
        "secondary labels carried for free from the window alignment.",
        "- **Recording coverage (`depth_covered` / `book_covered`):** provenance metadata "
        "— whether **our own** Binance `depth20@100ms` capture (`data/depth`) and "
        "Polymarket `top_of_book` capture recorded a frame within `coverage_tolerance_secs` "
        "of the sample. Symmetric (±tol) and up-OR-down token for `book_covered`, so they "
        "may read a recording timestamp just after the sample — **not** features, exempt "
        "from the no-look-ahead rule (like `dataset`'s `in_live_coverage`). Always `False` "
        "for `binance_proxy`.",
        "- **Feature-present coverage (`depth_feat_covered` / `pm_feat_covered`):** the "
        "flags to **filter training rows on**. `pm_feat_covered` is `True` iff the PM book "
        "features (`pm_mid`/`pm_spread`/`pm_book_imb`) are actually present — a **causal** "
        "backward as-of match on the **Up** token within staleness (`pm_asof_ts_ms` ≤ "
        "`sample_ts`, not NA). Unlike `book_covered` (up-OR-down, ±tol, may point just after "
        "the sample), `pm_feat_covered` means the features are really there. Likewise "
        "`depth_feat_covered` ⟺ `depth_imb_*`/`microprice_gap` present. (The derived "
        "`pm_staleness_*` / `*_depth_slope` sub-features can still be NaN at a run-start "
        "warmup / on a <2-level book — see `pm_run_secs`.) `False` for `binance_proxy`.",
        "- **`label_source`:** `chainlink` for windows we recorded live (real Chainlink "
        "strike + `Resolved` outcome) and `binance_proxy` for historical windows "
        "reconstructed from the Binance aggTrades archive. Proxy windows have no "
        "Chainlink/depth feed, so `chainlink`, `basis_bps`, `imbalance`, `microprice`, "
        "`spread`, `depth_mid` are `NaN`; `mid`, `sigma_1s`, `log_s_k`, `z`, `p_up_model` "
        "are still populated. A historical window that collides with a journal window is "
        "dropped, so the `chainlink` subset is never altered.",
        "- **`split`:** per-series chronological — the latest `val_fraction` of each "
        "series' windows are `val`; `chainlink` and `binance_proxy` windows are split "
        "**independently**, so adding history never reshuffles the journal subset.",
        "- **Stale features:** if the nearest at-or-before feature bar is older than "
        "`max_feature_staleness_secs`, the feature columns are nulled and "
        "`feature_asof_ts_ms` is `NA` — the row and its labels are kept.",
        "- **Microstructure features:** the multi-level Binance depth features "
        "(`depth_imb_*`, `microprice_gap`, `bid_depth_slope`, `ask_depth_slope`, "
        "`depth_spread_bps`) and the Polymarket Up-token book features (`pm_mid`, "
        "`pm_spread`, `pm_book_imb`, `pm_staleness_*`) are computed **only where our "
        "recordings cover the timestamp** — attached with a backward `merge_asof` "
        "(`depth_feat_asof_ts_ms` / `pm_asof_ts_ms` ≤ `sample_ts`). Missing-coverage "
        "rows get honest `NaN` (never build-time imputation); all are `NaN` on "
        "`binance_proxy` rows (no live recording). The staleness signal contrasts the "
        "Binance mid move with the PM mid change over the same 1–3 s span (both via "
        "backward-in-time shifts — causal).",
        "- **Distribution sanity:** `sanity.{json,html}` scores every feature column "
        "PASS/WARN/FAIL for unexpected NaN, absurd outliers, and constant columns. A "
        "NaN is *expected* where coverage is absent (or during vol warmup); only "
        "*excess* NaN fails.",
        "",
    ]
    (out_dir / "SCHEMA.md").write_text("\n".join(lines), encoding="utf-8")


# --- distribution sanity report ---------------------------------------------
# Same shape as the feature_set stage's report (PASS/WARN/FAIL per feature column:
# unexpected NaN / absurd outliers / constant column), covering EVERY feature-category
# column. "Expected NaN" is keyed on each column's own coverage mask — a value missing
# because our recording did not cover the sample (or during vol warmup) is expected, not
# a bug; excess (unexpected) NaN is what fails. A lightweight local FeatureSpec keeps this
# module import-cheap; the pure HTML/histogram render helpers are borrowed from feature_set
# lazily (in _write_sanity) so matplotlib is only imported when a report is written.
NAN_WARN_FRAC = 0.005  # excess-NaN fraction: ≤ this WARNs, above FAILs.
OUTLIER_WARN_FRAC = 0.005  # soft-bound outlier fraction: ≤ this WARNs, above FAILs.
_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
_PRICE_HI = 1e12  # a loose ceiling for USD price-scale features (asset-dependent).


def _worse(a: str, b: str) -> str:
    return a if _STATUS_ORDER[a] >= _STATUS_ORDER[b] else b


@dataclass(frozen=True)
class SanitySpec:
    """The sanity expectations for one feature column (name + doc come from
    COLUMN_SPEC; this adds the builder family, coverage, and bounds)."""

    name: str
    dtype: str
    family: str
    param: float | int | None
    coverage: str  # "full" | "partial"
    hard_bounds: bool  # True ⇒ any out-of-[lo,hi] value is a FAIL (a math-bounded feature).
    lo: float
    hi: float
    doc: str


# name -> (family, coverage, hard_bounds, lo, hi). Docs come from COLUMN_SPEC.
_SANITY_META: dict[str, tuple[str, str, bool, float, float]] = {
    "mid": ("price", "full", False, 0.0, _PRICE_HI),
    "ret": ("price", "full", False, -0.05, 0.05),
    "realized_vol": ("price", "full", False, 0.0, lm.DEFAULT_VOL_CAP_1S),
    "sigma_1s": ("vol", "full", False, 0.0, lm.DEFAULT_VOL_CAP_1S),
    "chainlink": ("chainlink", "partial", False, 0.0, _PRICE_HI),
    "basis_bps": ("chainlink", "partial", False, -1000.0, 1000.0),
    "basis_ewma": ("chainlink", "partial", False, -0.1, 0.1),
    "log_s_k": ("moneyness", "full", False, -2.0, 2.0),
    "z": ("zscore", "full", True, -lm.Z_CLAMP, lm.Z_CLAMP),
    "p_up_model": ("prob", "full", True, 0.0, 1.0),
    "imbalance": ("depth_existing", "partial", True, -1.0, 1.0),
    "microprice": ("depth_existing", "partial", False, 0.0, _PRICE_HI),
    "spread": ("depth_existing", "partial", False, 0.0, _PRICE_HI),
    "depth_mid": ("depth_existing", "partial", False, 0.0, _PRICE_HI),
    "microprice_gap": ("depth_new", "partial", False, -1e6, 1e6),
    "bid_depth_slope": ("depth_new", "partial", False, 0.0, _PRICE_HI),
    "ask_depth_slope": ("depth_new", "partial", False, 0.0, _PRICE_HI),
    "depth_spread_bps": ("depth_new", "partial", False, 0.0, 1e5),
    "pm_mid": ("pm", "partial", True, 0.0, 1.0),
    "pm_spread": ("pm", "partial", True, 0.0, 1.0),
    "pm_book_imb": ("pm", "partial", True, -1.0, 1.0),
}
for _lv in DEPTH_LEVEL_DEPTHS:  # multi-level depth imbalances (math-bounded [-1,1])
    _SANITY_META[f"depth_imb_{_lv}"] = ("depth_new", "partial", True, -1.0, 1.0)
for _lb in STALENESS_LOOKBACKS_SECS:  # staleness (soft — log-move minus prob change)
    _SANITY_META[f"pm_staleness_{_lb}s"] = ("pm_staleness", "partial", False, -1.0, 1.0)


def _build_sanity_spec() -> list[SanitySpec]:
    docs = {name: doc for name, _dt, _cat, doc in COLUMN_SPEC}
    stale_param = {f"pm_staleness_{lb}s": lb for lb in STALENESS_LOOKBACKS_SECS}
    specs: list[SanitySpec] = []
    for name in FEATURE_COLS:
        fam, cov, hard, lo, hi = _SANITY_META[name]
        specs.append(SanitySpec(name, "float64", fam, stale_param.get(name), cov, hard, lo, hi, docs[name]))
    return specs


SANITY_SPEC = _build_sanity_spec()
# Every feature column must have sanity metadata (kept in lockstep with COLUMN_SPEC).
assert {s.name for s in SANITY_SPEC} == set(FEATURE_COLS), "SANITY_SPEC/FEATURE_COLS drift"

# Columns read once to classify where a NaN is legitimately expected.
_SANITY_MASK_COLS = [
    "feature_asof_ts_ms", "label_source", "strike", "depth_covered",
    "depth_feat_asof_ts_ms", "pm_asof_ts_ms", "pm_run_secs", "sigma_1s",
]


def _expected_nan_mask(spec: SanitySpec, masks: pd.DataFrame) -> np.ndarray:
    """Rows where this feature is *legitimately* NaN — no price bar / no coverage /
    vol warmup. Excess NaN is anything outside this set."""
    fa = masks["feature_asof_ts_ms"].isna().to_numpy()  # no as-of price bar
    proxy = (masks["label_source"] == "binance_proxy").to_numpy()
    strike = masks["strike"].to_numpy(dtype=float)
    # `sigma_1s` is NaN exactly during the EWMA warmup (the longest of the price
    # warmups — ret/realized_vol/basis_ewma finish sooner), so it is the warmup signal
    # for every value that needs a warmed grid. (short_horizon carries no run-length
    # column like feature_set's `price_run_secs`, so we reuse σ as the proxy.)
    warm = masks["sigma_1s"].isna().to_numpy()
    fam = spec.family
    if fam in ("price", "vol", "chainlink"):  # NaN during warmup / (chainlink) on proxy
        base = fa | warm
        return (proxy | base) if fam == "chainlink" else base
    if fam == "moneyness":  # log(mid/strike): defined from bar 0 — only needs a bar + strike
        return fa | ~(strike > 0.0)
    if fam in ("zscore", "prob"):  # z / p_up need a warmed σ and a positive strike
        return fa | warm | ~(strike > 0.0)
    if fam == "depth_existing":  # depth microstructure from the same recording
        return ~masks["depth_covered"].to_numpy(dtype=bool)
    if fam == "depth_new":
        return masks["depth_feat_asof_ts_ms"].isna().to_numpy()
    if fam == "pm":
        return masks["pm_asof_ts_ms"].isna().to_numpy()
    if fam == "pm_staleness":  # NaN with no book OR within the N-second lookback warmup
        pm_na = masks["pm_asof_ts_ms"].isna().to_numpy()
        run = masks["pm_run_secs"].to_numpy(dtype=float)
        return pm_na | ~(run >= float(spec.param))
    return np.zeros(len(fa), dtype=bool)


def _column_sanity(spec: SanitySpec, values: np.ndarray, masks: pd.DataFrame) -> dict:
    """Distribution + the three checks (no unexpected NaN / no absurd outliers / not
    constant) for one feature column, PASS/WARN/FAIL per check and overall."""
    n = int(len(values))
    finite_mask = np.isfinite(values)
    n_finite = int(finite_mask.sum())
    n_nan = n - n_finite
    finite = values[finite_mask]

    expected = _expected_nan_mask(spec, masks)
    excess = int(((~finite_mask) & ~expected).sum())
    excess_frac = (excess / n) if n else 0.0

    if n_finite:
        pct = np.percentile(finite, [1, 5, 50, 95, 99])
        stats = {
            "min": float(finite.min()), "max": float(finite.max()),
            "mean": float(finite.mean()), "std": float(finite.std()),
            "p1": float(pct[0]), "p5": float(pct[1]), "p50": float(pct[2]),
            "p95": float(pct[3]), "p99": float(pct[4]),
            "unique_count": int(np.unique(finite).size),
        }
        out_count = int(((finite < spec.lo) | (finite > spec.hi)).sum())
        hist_counts, hist_edges = np.histogram(finite, bins=24)
        hist = {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()}
    else:
        stats = {k: float("nan") for k in ("min", "max", "mean", "std", "p1", "p5", "p50", "p95", "p99")}
        stats["unique_count"] = 0
        out_count = 0
        hist = {"counts": [], "edges": []}
    out_frac = (out_count / n_finite) if n_finite else 0.0

    # (1) no unexpected NaN — judged on the *excess* NaN only.
    if spec.coverage == "partial" and n_finite == 0:
        nan_flag, nan_note = "WARN", "no coverage"
    elif spec.coverage == "full" and n_finite == 0:
        nan_flag, nan_note = "FAIL", "full-coverage feature is entirely NaN"
    elif excess_frac == 0.0:
        nan_flag, nan_note = "PASS", ""
    elif excess_frac <= NAN_WARN_FRAC:
        nan_flag, nan_note = "WARN", f"{excess_frac:.2%} unexpected NaN"
    else:
        nan_flag, nan_note = "FAIL", f"{excess_frac:.2%} unexpected NaN"

    # (2) no absurd outliers — hard-bounded features fail on ANY violation.
    if n_finite == 0 or out_count == 0:
        outlier_flag = "PASS"
    elif spec.hard_bounds:
        outlier_flag = "FAIL"
    elif out_frac <= OUTLIER_WARN_FRAC:
        outlier_flag = "WARN"
    else:
        outlier_flag = "FAIL"

    # (3) no constant columns — constant full-coverage = a bug (FAIL); constant
    # partial-coverage may be a limited-coverage artifact (WARN).
    uniq = stats["unique_count"]
    if n_finite == 0:
        constant_flag = "PASS"
    elif uniq <= 1:
        constant_flag = "FAIL" if spec.coverage == "full" else "WARN"
    elif uniq == 2:
        constant_flag = "WARN"
    else:
        constant_flag = "PASS"

    status = _worse(_worse(nan_flag, outlier_flag), constant_flag)
    return {
        "name": spec.name, "family": spec.family, "coverage": spec.coverage, "dtype": spec.dtype,
        "n": n, "n_finite": n_finite, "n_nan": n_nan,
        "nan_fraction": (n_nan / n) if n else 0.0, "excess_nan_fraction": excess_frac,
        **stats, "outlier_count": out_count, "outlier_fraction": out_frac,
        "lo": spec.lo, "hi": spec.hi,
        "nan_flag": nan_flag, "outlier_flag": outlier_flag, "constant_flag": constant_flag,
        "status": status, "note": nan_note, "hist": hist,
    }


def sanity_report(parquet_path, specs: list[SanitySpec]) -> dict:
    """Distribution sanity of every feature column in the finished parquet, read a
    column at a time (bounded memory). Returns per-column stats + an overall status."""
    masks = pq.read_table(parquet_path, columns=_SANITY_MASK_COLS).to_pandas()
    n = int(len(masks))
    columns: list[dict] = []
    overall = "PASS"
    for spec in specs:
        values = pq.read_table(parquet_path, columns=[spec.name]).to_pandas()[spec.name].to_numpy(dtype=float)
        stats = _column_sanity(spec, values, masks)
        columns.append(stats)
        overall = _worse(overall, stats["status"])
    tally = {s: sum(1 for c in columns if c["status"] == s) for s in ("PASS", "WARN", "FAIL")}
    return {"n_rows": n, "n_features": len(specs), "columns": columns,
            "status_tally": tally, "overall_status": overall}


def _write_sanity(out_dir, sanity: dict) -> None:
    """Write ``sanity.json`` (always) and a compact ``sanity.html`` (reusing the
    feature_set render helpers; degrades gracefully if matplotlib is unavailable)."""
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    cols = sanity["columns"]
    tally = sanity["status_tally"]
    try:
        from . import feature_set as fs
        table = fs._table_html(cols, fs._SANITY_TABLE_COLS)
        hist_img = fs._img_tag(fs._hist_png(cols), "feature histograms")
    except Exception:  # pragma: no cover - render is best-effort
        table = "<p class='muted'>table render unavailable.</p>"
        hist_img = "<p class='muted'>histograms unavailable.</p>"
    banner = {"PASS": "#1f7a1f", "WARN": "#a06a00", "FAIL": "#b00020"}[sanity["overall_status"]]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>short_horizon — sanity</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.9rem; }}
 td, th {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }}
 tr.warn td {{ background: #fff5e6; color: #7a5200; }}
 tr.fail td {{ background: #fdecef; color: #8a0018; }}
 .muted {{ color: #888; }} code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
 .verdict {{ background: {banner}12; border-left: 4px solid {banner}; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
 img {{ max-width: 100%; }}
</style></head><body>
<h1>short_horizon — feature distribution sanity</h1>
<p class="verdict"><b>{sanity['overall_status']}</b> — {tally['PASS']} PASS, {tally['WARN']} WARN,
{tally['FAIL']} FAIL over {sanity['n_features']} features and {sanity['n_rows']:,} rows.
Each feature is checked for <b>unexpected NaN</b>, <b>absurd outliers</b>, and being a
<b>constant column</b>. NaN is "expected" where our recording did not cover the sample (or
during vol warmup) — only <i>excess</i> NaN fails.</p>

<h2>Per-feature sanity</h2>
{table}

<h2>Distributions</h2>
{hist_img}

<p class="muted">Generated by <code>python -m model_lab.short_horizon</code>.
Full column docs in <code>SCHEMA.md</code>.</p>
</body></html>
"""
    (out_dir / "sanity.html").write_text(html, encoding="utf-8")


# --- entry ------------------------------------------------------------------
def _parse_horizons(val: str | None) -> tuple[int, ...]:
    if not val:
        return DEFAULT_HORIZONS
    return tuple(int(s.strip()) for s in val.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "short_horizon")
    parser.add_argument("--grid-secs", type=int, default=DEFAULT_GRID_SECS,
                        help=f"dense sampling cadence within each window (default {DEFAULT_GRID_SECS}; use 1 for max density)")
    parser.add_argument("--horizons", default=None,
                        help="comma-separated forward-direction label horizons in seconds (default 10,15)")
    parser.add_argument("--series", default=None,
                        help="comma-separated series filter (default: all 5m series, e.g. BTC-5m,ETH-5m)")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help=f"fraction of each series' latest windows for the val split (default {DEFAULT_VAL_FRACTION})")
    parser.add_argument("--strike-lookback-secs", type=int, default=DEFAULT_STRIKE_LOOKBACK_SECS,
                        help=f"max age of the Chainlink tick used for the strike (default {DEFAULT_STRIKE_LOOKBACK_SECS})")
    parser.add_argument("--strike-tolerance-secs", type=float, default=DEFAULT_STRIKE_TOLERANCE_SECS,
                        help=f"strike counts as 'boundary' within this gap of open (default {DEFAULT_STRIKE_TOLERANCE_SECS})")
    parser.add_argument("--max-feature-staleness-secs", type=int, default=DEFAULT_MAX_FEATURE_STALENESS_SECS,
                        help=f"null as-of features older than this (recording-gap relics) (default {DEFAULT_MAX_FEATURE_STALENESS_SECS})")
    parser.add_argument("--coverage-tolerance-secs", type=float, default=DEFAULT_COVERAGE_TOLERANCE_SECS,
                        help=f"a sample is 'covered' if a recording is within this gap (default {DEFAULT_COVERAGE_TOLERANCE_SECS})")
    parser.add_argument("--no-history", action="store_true",
                        help="journal-only: skip the Binance aggTrades historical proxy windows")
    parser.add_argument("--no-stream", action="store_true",
                        help="read every raw tick into memory (the pre-streaming path; may OOM on a large journal) — for debugging / equivalence checks")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    horizons = _parse_horizons(args.horizons)
    print(f"[short_horizon] journal={paths.journal_dir}")
    print(f"[short_horizon] hist   ={paths.hist_dir} (history {'off' if args.no_history else 'on'})")
    print(f"[short_horizon] out    ={paths.table('short_horizon')}")
    if since_ms is not None or until_ms is not None:
        print(f"[short_horizon] bounds : since={args.since} until={args.until} "
              f"(stream={'off' if args.no_stream else 'on'})")

    result = short_horizon(
        paths,
        grid_secs=args.grid_secs,
        horizons=horizons,
        series=ds._parse_series(args.series),
        val_fraction=args.val_fraction,
        strike_lookback_secs=args.strike_lookback_secs,
        strike_tolerance_secs=args.strike_tolerance_secs,
        max_feature_staleness_secs=args.max_feature_staleness_secs,
        coverage_tolerance_secs=args.coverage_tolerance_secs,
        include_history=not args.no_history,
        since_ms=since_ms,
        until_ms=until_ms,
        stream=not args.no_stream,
    )
    c = result["counts"]
    cov = result["coverage"]
    print(f"[short_horizon] {c['windows']:,} windows -> {c['samples']:,} samples "
          f"({c['samples_train']:,} train / {c['samples_val']:,} val) over {result['series']}, "
          f"horizons={horizons}")
    print(f"[short_horizon] source: chainlink={c['windows_chainlink']:,} win / {c['samples_chainlink']:,} samp"
          f" · binance_proxy={c['windows_binance_proxy']:,} win / {c['samples_binance_proxy']:,} samp")
    print(f"[short_horizon] recording coverage (±{cov['coverage_tolerance_secs']}s): "
          f"depth={cov['depth_pct']:.1%} ({c['samples_depth_covered']:,}) · "
          f"book={cov['book_pct']:.1%} ({c['samples_book_covered']:,}) of {c['samples']:,} samples")
    print(f"[short_horizon] features present (filter on these): "
          f"depth_feat={cov['depth_feat_pct']:.1%} ({c['samples_depth_feat_covered']:,}) · "
          f"pm_feat={cov['pm_feat_pct']:.1%} ({c['samples_pm_feat_covered']:,})")

    # Per-day: print the days with any of our recordings (the journal span) in full;
    # summarize the zero-coverage historical-proxy days as one line (there can be ~550).
    per_day = result["per_day"]
    covered_days = [r for r in per_day if r["depth_covered"] or r["book_covered"]]
    zero_days = [r for r in per_day if not (r["depth_covered"] or r["book_covered"])]
    if covered_days:
        print(f"[short_horizon] per-day (of {len(per_day)} days; {len(covered_days)} with our recordings):")
        for r in covered_days:
            print(f"[short_horizon]   {r['day']}: {r['samples']:,} samples · "
                  f"depth {r['depth_pct']:.1%} · book {r['book_pct']:.1%}")
    if zero_days:
        zsamp = sum(r["samples"] for r in zero_days)
        print(f"[short_horizon]   + {len(zero_days)} historical-proxy days: "
              f"{zsamp:,} samples · 0% coverage")
    print(f"[short_horizon] journal subset: {result['journal_subset']['windows']:,} windows / "
          f"{result['journal_subset']['samples']:,} samples, unchanged "
          f"(dedup dropped {result['journal_subset']['dedup_collisions']:,} colliding proxy windows)")
    procmem.report_peak_rss("short_horizon", result.get("peak_rss_mb"), args.max_rss_mb)
    tally = result.get("sanity_tally", {})
    print(f"[short_horizon] sanity: {result.get('sanity_status')} "
          f"({tally.get('PASS', 0)} PASS / {tally.get('WARN', 0)} WARN / {tally.get('FAIL', 0)} FAIL)")
    print(f"[short_horizon] wrote {paths.table('short_horizon').name} + "
          f"short_horizon/{{SCHEMA.md, metadata.json, sanity.json, sanity.html}}")
    if c["samples"] == 0:
        print("[short_horizon] WARNING: no samples produced — check --journal-dir / --series.")
        return 1
    if result.get("sanity_status") == "FAIL":
        print("[short_horizon] WARNING: distribution sanity FAILED — see short_horizon/sanity.html")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
