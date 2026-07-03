"""Stage — dataset.

Reconstruct real trading windows and emit a **train-ready** supervised table: one
row per (window, feature-timestamp), with point-in-time features and two forward
labels, safe to feed a learner.

For every in-scope window (5-minute BTC/ETH Up/Down by default) this stage:

- reconstructs the **strike** (price-to-beat) = the resolution-grade Chainlink
  Vendor price captured at window open (``LastAtOrBefore``, ``crates/model/strike.rs``
  — every ``window`` record has ``strike: null`` on disk, so we rebuild it);
- takes the authoritative Up/Down **outcome** from the ``Resolved`` lifecycle event
  (present for *every* resolved window, traded or not);
- samples feature timestamps on a **fixed grid** inside ``[open, close)``;
- attaches, per sample, the **as-of** features (mid, returns, EWMA vol, basis,
  moneyness, reconstructed ``p_up``, order-book microstructure) computed strictly
  from data **at or before** the sample instant, reusing the no-look-ahead feature
  kernel ``features.build_asset_features``;
- attaches two **labels** (which read the *future*): (a) the direction of the
  underlying mid ``horizon`` seconds later, and (b) the window's final outcome.

**Hard rule — no look-ahead.** No value derived from data *after* a sample's
timestamp may appear in that sample's features. Enforced three ways: a runtime
assertion here (:func:`_assert_no_lookahead`), an AST source scan
(``tests/test_dataset.py::test_no_lookahead_ast_scan``), and a differential
perturbation test (``…::test_no_lookahead_differential``).

A separate ``in_live_coverage`` column marks samples whose window has a real
Polymarket market-mid benchmark (a ``top_of_book`` recording for its token) — so a
later stage knows where the market can be compared against the model.

**Historical extension (Binance aggTrades proxy).** Beyond the windows we recorded
live, this stage also reconstructs 5-minute windows across the full downloaded
Binance ``aggTrades`` history (``data/aggtrades``, both BTCUSDT + ETHUSDT), so a
learner can train on months of data rather than only our recording period. For
those historical windows there is no Chainlink feed, so the strike and outcome come
from **Binance** (last trade at-or-before open; end ≥ strike ⇒ Up) — a *proxy* for
the true Chainlink resolution. This is made explicit:

- ``label_source`` = ``chainlink`` for journal-covered windows, ``binance_proxy``
  for historical ones;
- ``strike_distance_at_close`` (absolute) and ``strike_distance_at_close_vol`` (in
  units of the window's realized volatility) expose how close the resolution landed
  to the strike, so **knife-edge** windows — where the ~13 bps Binance↔Chainlink
  basis could flip the proxy outcome — can be filtered or down-weighted later.

The 30-second forward-direction label (``fwd_up_30s``) is **price-only** and so is
proxy-safe on both sources — it is the primary label. Journal windows keep their
Chainlink strike/outcome exactly (a historical window is dropped when it collides
with a journal window, so the journal-covered subset is never altered). Every
no-look-ahead guard runs over the extended data unchanged (historical features go
through the same ``features.build_asset_features`` kernel via synthetic Binance-mid
ticks, so the AST scan / differential test remain authoritative).

History is processed **month by month** and the output is streamed to parquet, so
the whole trade history is never held in RAM.

Self-contained: reads the gzip journal directly (no prerequisite ``ingest`` run).
Outputs ``out/dataset.parquet`` + ``out/dataset/{SCHEMA.md, metadata.json}``.

Run::

    python -m model_lab.dataset
    python -m model_lab.dataset --grid-secs 15 --horizon-secs 30 --series BTC-5m,ETH-5m
    python -m model_lab.dataset --no-history          # journal-only (skip aggTrades)
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from math import ceil, log, sqrt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import features as feat
from .config import Paths, resolve_paths, stage_parser
from .io import binance_archive as ba
from .io import depth as depth_io
from .io import journal as jio
from .ingest import DEPTH_COLS, TICK_COLS
from .lib import math as lm

# --- defaults ---------------------------------------------------------------
DEFAULT_GRID_SECS = 15
DEFAULT_HORIZON_SECS = 30
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_STRIKE_LOOKBACK_SECS = 180
DEFAULT_STRIKE_TOLERANCE_SECS = 2.5
# Beyond this, an as-of feature bar is a stale relic across a recording gap (feeds
# tick ~1/s, so this only ever nulls genuinely-down periods — mirrors the market-mid
# guard in calibration_audit). The sample and its outcome label are kept.
DEFAULT_MAX_FEATURE_STALENESS_SECS = 120

# --- historical (Binance aggTrades) extension -------------------------------
WINDOW_SECS = 300  # the 5-minute window length (all series in scope are 5m).
WINDOWS_PER_DAY = 86_400 // WINDOW_SECS
# Which downloaded symbol maps to which series / feature-grid asset.
SYMBOL_SERIES = {"BTCUSDT": "BTC-5m", "ETHUSDT": "ETH-5m"}
SYMBOL_ASSET = {"BTCUSDT": "btc", "ETHUSDT": "eth"}
# "Knife-edge" report thresholds — a proxy window is knife-edge at θ when its close
# landed within θ realized-vol units of the strike (small ⇒ the Binance↔Chainlink
# basis could flip the reconstructed outcome). Reported as proportions.
KNIFE_EDGE_VOL_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)


# --- output schema ----------------------------------------------------------
# Single source of truth for the column set, dtypes, category, and the SCHEMA.md
# doc — so the doc and the FEATURE/LABEL invariants can never drift. Categories:
# "id" / "meta" (both known at sample time) · "feature" (as-of ≤ sample_ts) ·
# "label" (reads the future) · "diag" (label diagnostic, reads the future, NOT a
# training input) · "coverage".
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
    ("basis_bps", "float64", "feature", "`1e4·(chainlink/mid − 1)` — Chainlink-minus-Binance basis, as-of; NaN for `binance_proxy` rows."),
    ("log_s_k", "float64", "feature", "Moneyness `log(mid/strike)` — mid as-of, strike from open."),
    ("z", "float64", "feature", "`log_s_k / (σ_1s·√τ)`, clipped ±39 (all inputs as-of)."),
    ("p_up_model", "float64", "feature", "Reconstructed engine `p_up = Φ(z)` (as-of)."),
    ("imbalance", "float64", "feature", "Order-book imbalance from depth, as-of (NaN if no depth; always NaN for `binance_proxy`)."),
    ("microprice", "float64", "feature", "Size-weighted microprice from depth, as-of."),
    ("spread", "float64", "feature", "Best ask − best bid from depth, as-of."),
    ("depth_mid", "float64", "feature", "Depth mid from the Binance L2 book, as-of."),
    ("fwd_ret_30s", "float64", "label", "LABEL: log return of the mid `horizon`s later; NaN if unobserved."),
    ("fwd_up_30s", "Int8", "label", "LABEL (primary, price-only, proxy-safe): 1 iff `fwd_ret ≥ 0` (ties → Up); NA if unobserved."),
    ("fwd_crosses_close", "bool", "meta", "Whether `sample_ts + horizon` reaches past window close (known at sample time)."),
    ("outcome", "string", "label", "LABEL: window final outcome `Up`/`Down` (Chainlink `Resolved` for `chainlink` rows; Binance end≥strike for `binance_proxy`)."),
    ("outcome_up", "int8", "label", "LABEL: 1 iff the window resolved Up."),
    ("strike_distance_at_close", "float64", "diag", "|close − strike| in USD (resolution-feed close price). Small ⇒ knife-edge outcome."),
    ("strike_distance_at_close_vol", "float64", "diag", "`|log(close/strike)| / (σ_realized_1s·√window)` — the close-to-strike gap in units of the window's realized vol. Small ⇒ a basis shift could flip a `binance_proxy` outcome."),
    ("in_live_coverage", "bool", "coverage", "Whether a Polymarket `top_of_book` benchmark exists for this window (always False for `binance_proxy`)."),
    ("split", "string", "meta", "Chronological split `train` / `val` (per series, latest windows → val; split independently within each `label_source`)."),
]

COLUMNS = [c[0] for c in COLUMN_SPEC]
DTYPES = {c[0]: c[1] for c in COLUMN_SPEC}
FEATURE_COLS = [c[0] for c in COLUMN_SPEC if c[2] == "feature"]
LABEL_COLS = [c[0] for c in COLUMN_SPEC if c[2] == "label"]

# Invariant: a feature can never also be a label. Guarded here at import (so a
# careless edit to COLUMN_SPEC fails fast) and again at runtime in every dataset().
assert not (set(FEATURE_COLS) & set(LABEL_COLS)), "COLUMN_SPEC: feature/label overlap"


# --- series helpers ---------------------------------------------------------
def series_asset(series: str) -> str:
    """Maps a series key (``BTC-5m``) to its feature-grid asset (``btc``)."""
    head = series.split("-", 1)[0].strip().lower()
    if head.startswith("btc"):
        return "btc"
    if head.startswith("eth"):
        return "eth"
    return head


def _in_scope(series: str, series_filter: tuple[str, ...] | None) -> bool:
    """In scope iff it matches an explicit ``--series`` filter, else any 5m series."""
    if series_filter:
        return series in series_filter
    return series.endswith("-5m")


# --- journal → frames -------------------------------------------------------
def _read_journal(
    paths: Paths, series_filter: tuple[str, ...] | None
) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """One pass over the journal. Returns ``(ticks_df, win_meta, outcome_resolved,
    outcome_settle, top_ts_by_token)``.

    ``ticks_df`` has the ``ingest.TICK_COLS`` shape (every ``price_tick``, all
    sources/assets). ``win_meta[(series, open)] = {close_time, up_token,
    down_token}`` for in-scope windows; outcomes from the ``Resolved`` lifecycle
    (preferred) and ``settlement`` (fallback). ``top_ts_by_token[token] = [ts…]``
    are the ``top_of_book`` timestamps used for the live-coverage flag.
    """
    cols: dict[str, list] = {c: [] for c in TICK_COLS}
    win_meta: dict[tuple[str, int], dict] = {}
    outcome_resolved: dict[tuple[str, int], str] = {}
    outcome_settle: dict[tuple[str, int], str] = {}
    top_ts_by_token: dict[str, list[int]] = defaultdict(list)

    for rec in jio.read_records(paths.journal_dir):
        kind = rec.get("type")
        if kind == "price_tick":
            cols["ts_local_ms"].append(rec.get("ts_local_ms"))
            cols["source"].append(rec.get("source"))
            cols["asset"].append(jio.asset_key(rec.get("asset")))
            cols["kind"].append(rec.get("kind"))
            cols["value"].append(jio.to_float(rec.get("value")))
            cols["ts_exchange"].append(rec.get("ts_exchange"))
            cols["ts_local"].append(rec.get("ts_local"))
        elif kind == "top_of_book":
            top = rec.get("top") or {}
            ts = top.get("ts")
            token = rec.get("token_id")
            if token is not None and ts is not None:
                top_ts_by_token[token].append(int(ts))
        elif kind == "window":
            market = rec.get("market") or {}
            key = jio.window_key(market.get("window"))
            if key == ("", 0) or not _in_scope(key[0], series_filter):
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
            if key == ("", 0) or not _in_scope(key[0], series_filter):
                continue
            outcome = rec.get("outcome")
            if outcome in ("Up", "Down"):
                outcome_settle[key] = outcome

    ticks_df = pd.DataFrame(cols, columns=TICK_COLS)
    return ticks_df, win_meta, outcome_resolved, outcome_settle, top_ts_by_token


def _windows_frame(win_meta: dict, outcome_resolved: dict, outcome_settle: dict) -> pd.DataFrame:
    """Resolved in-scope windows: ``series, asset, window_open_ms, window_close_ms,
    up_token, down_token, outcome, outcome_up`` (outcome Resolved-preferred)."""
    rows: list[dict] = []
    for key, meta in win_meta.items():
        close = meta.get("close_time")
        outcome = outcome_resolved.get(key) or outcome_settle.get(key)
        if close is None or outcome not in ("Up", "Down"):
            continue
        series, open_time = key
        rows.append(
            {
                "series": series,
                "asset": series_asset(series),
                "window_open_ms": int(open_time),
                "window_close_ms": int(close),
                "up_token": meta.get("up_token"),
                "down_token": meta.get("down_token"),
                "outcome": outcome,
                "outcome_up": 1 if outcome == "Up" else 0,
            }
        )
    cols = ["series", "asset", "window_open_ms", "window_close_ms",
            "up_token", "down_token", "outcome", "outcome_up"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["series", "window_open_ms"]).reset_index(drop=True)


def reconstruct_strikes(
    ticks_df: pd.DataFrame, windows_df: pd.DataFrame, lookback_ms: int, tolerance_ms: float
) -> pd.DataFrame:
    """Strike per window via ``LastAtOrBefore`` the Chainlink Vendor feed.

    Returns ``series, asset, window_open_ms, strike, strike_ts_ms, strike_quality``.
    The strike is the latest ``ChainlinkRtds``/``Vendor`` price whose
    ``ts_exchange ≤ open`` and within ``lookback_ms`` — the resolution-grade
    price-to-beat captured at window start. ``strike_quality`` is ``boundary`` when
    the chosen tick is within ``tolerance_ms`` of open, else ``distant``; ``missing``
    when no tick qualifies. (5m/15m use ``LastAtOrBefore``; a Binance-candle 1h
    series would want ``FirstAtOrAfter`` — out of scope, see SCHEMA.md.)
    """
    base_cols = ["series", "asset", "window_open_ms"]
    if windows_df.empty:
        return pd.DataFrame(columns=base_cols + ["strike", "strike_ts_ms", "strike_quality"])

    vend = ticks_df[(ticks_df["source"] == "ChainlinkRtds") & (ticks_df["kind"] == "Vendor")].copy()
    vend["ts_exchange"] = vend["ts_exchange"].fillna(vend["ts_local"])
    vend = vend[["asset", "ts_exchange", "value"]].dropna()
    vend = vend[vend["value"] > 0.0]

    left = windows_df[base_cols].copy()
    left["window_open_ms"] = left["window_open_ms"].astype("int64")
    left = left.sort_values("window_open_ms")

    if vend.empty:
        out = left.copy()
        out["strike"] = np.nan
        out["strike_ts_ms"] = pd.array([pd.NA] * len(out), dtype="Int64")
        out["strike_quality"] = "missing"
        return out

    vend["ts_exchange"] = vend["ts_exchange"].astype("int64")
    vend = vend.sort_values("ts_exchange")
    merged = pd.merge_asof(
        left,
        vend.rename(columns={"value": "strike", "ts_exchange": "strike_ts_ms"}),
        left_on="window_open_ms",
        right_on="strike_ts_ms",
        by="asset",
        direction="backward",
        tolerance=int(lookback_ms),
    )
    gap = merged["window_open_ms"] - merged["strike_ts_ms"]
    merged["strike_quality"] = np.where(
        merged["strike_ts_ms"].isna(), "missing",
        np.where(gap <= tolerance_ms, "boundary", "distant"),
    )
    merged["strike_ts_ms"] = merged["strike_ts_ms"].astype("Int64")
    return merged[base_cols + ["strike", "strike_ts_ms", "strike_quality"]]


def _coverage_flags(windows_df: pd.DataFrame, top_ts_by_token: dict) -> list[bool]:
    """Per window: does a ``top_of_book`` for its up/down token fall in ``[open, close)``?

    This is the live-recording / market-mid benchmark flag — true exactly where a
    later stage could attach a real Polymarket mid for this contract.
    """
    sorted_ts = {tok: np.sort(np.asarray(v, dtype="int64")) for tok, v in top_ts_by_token.items()}

    def any_in(token, lo: int, hi: int) -> bool:
        arr = sorted_ts.get(token)
        if arr is None or len(arr) == 0:
            return False
        i = int(np.searchsorted(arr, lo, side="left"))
        return i < len(arr) and int(arr[i]) < hi

    flags: list[bool] = []
    for w in windows_df.itertuples(index=False):
        lo, hi = int(w.window_open_ms), int(w.window_close_ms)
        flags.append(any_in(w.up_token, lo, hi) or any_in(w.down_token, lo, hi))
    return flags


def _assign_split(windows_df: pd.DataFrame, val_fraction: float) -> pd.Series:
    """Chronological per-series split: the latest ``ceil(val_fraction·n)`` windows of
    each series → ``val`` (train on the past, validate on the future); rest ``train``."""
    split = pd.Series("train", index=windows_df.index, dtype="object")
    if val_fraction <= 0.0:
        return split
    for _, idx in windows_df.groupby("series").groups.items():
        order = windows_df.loc[idx].sort_values("window_open_ms").index
        n_val = min(len(order), ceil(val_fraction * len(order)))
        if n_val:
            split.loc[order[-n_val:]] = "val"
    return split


# --- feature grid (as-of; reuses the no-look-ahead kernel) -------------------
def build_feature_grid(ticks_df: pd.DataFrame, depth_df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """The per-second, as-of feature grid for one asset.

    Reuses :func:`features.build_asset_features` (mid, ret, EWMA vol, chainlink LOCF,
    basis, depth microstructure — all built with backward ``merge_asof`` / causal
    EWMA) and appends the engine ``sigma_1s`` (also causal). Every value at second
    ``s`` depends only on data at seconds ``≤ s`` — no look-ahead.
    """
    grid = feat.build_asset_features(ticks_df, depth_df, asset)
    grid = grid.reset_index(drop=True)
    grid["sigma_1s"] = lm.sigma_1s_from_bars(
        grid["sec"].to_numpy(dtype=np.int64), grid["mid"].to_numpy(dtype=float)
    ) if len(grid) else np.array([], dtype=float)
    return grid


def _derive_asof_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived as-of features ``log_s_k, z, p_up_model`` from the matched
    feature bar (``mid``, ``sigma_1s`` — as-of) and per-window scalars (``strike``
    captured at open, ``tau_secs`` known at sample time). No future data is read."""
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_s_k"] = np.log(df["mid"] / df["strike"])
        sigma_tau = df["sigma_1s"] * np.sqrt(df["tau_secs"])
        z = (df["log_s_k"] / sigma_tau).clip(-lm.Z_CLAMP, lm.Z_CLAMP)
    df["z"] = z
    df["p_up_model"] = lm.norm_cdf(z.to_numpy(dtype=float))
    return df


def _attach_features(
    samples: pd.DataFrame, grid: pd.DataFrame, max_staleness_ms: int | None = None
) -> pd.DataFrame:
    """Bind each sample to its as-of feature bar with a **backward** ``merge_asof``
    (the matched bar's ``ts_ms ≤ sample_ts_ms``), then derive ``log_s_k/z/p_up``.

    ``samples`` must carry ``asset, sample_ts_ms, strike, tau_secs``. This is the
    single feature↔sample join, and it is ``direction="backward"`` — the AST scan
    enforces that no feature path ever reads forward. When the matched bar is older
    than ``max_staleness_ms`` (a relic across a recording gap) the features are
    nulled and ``feature_asof_ts_ms`` set to NA — strictly more conservative, never
    a forward read."""
    value_cols = ["mid", "ret", "realized_vol", "sigma_1s", "chainlink",
                  "basis_bps", "imbalance", "microprice", "spread", "depth_mid"]
    if grid.empty:
        out = samples.copy()
        for col in value_cols:
            out[col] = np.nan
        out["feature_asof_ts_ms"] = pd.array([pd.NA] * len(out), dtype="Int64")
        return _derive_asof_features(out)

    right = grid.rename(columns={"ts_ms": "feature_asof_ts_ms"})
    right = right[["asset", "feature_asof_ts_ms", *value_cols]].sort_values("feature_asof_ts_ms")
    left = samples.sort_values("sample_ts_ms")
    merged = pd.merge_asof(
        left,
        right,
        left_on="sample_ts_ms",
        right_on="feature_asof_ts_ms",
        by="asset",
        direction="backward",
    )
    merged["feature_asof_ts_ms"] = merged["feature_asof_ts_ms"].astype("Int64")
    if max_staleness_ms is not None:
        stale = (merged["sample_ts_ms"] - merged["feature_asof_ts_ms"]) > max_staleness_ms
        stale = stale.fillna(False)
        merged.loc[stale, value_cols] = np.nan
        merged.loc[stale, "feature_asof_ts_ms"] = pd.NA
    return _derive_asof_features(merged)


# --- forward labels (read the future — kept apart from the feature path) -----
def forward_direction_labels(grid: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-asset forward ``horizon``-second direction of the mid, on a gap-free
    1-second grid. Returns ``asset, sec, fwd_ret_30s`` (``fwd_mid = mid.shift(-h)``).

    This is a **label** function — it deliberately reads strictly *later* ticks and
    is excluded from the no-look-ahead AST scan."""
    frames: list[pd.DataFrame] = []
    for asset, g in grid.groupby("asset"):
        sel = g[["sec", "mid"]].dropna().drop_duplicates("sec").sort_values("sec")
        if sel.empty:
            continue
        lo, hi = int(sel["sec"].min()), int(sel["sec"].max())
        gg = sel.set_index("sec").reindex(range(lo, hi + 1))
        gg["mid"] = gg["mid"].ffill()
        gg["fwd_mid"] = gg["mid"].shift(-horizon)
        gg["fwd_ret_30s"] = np.log(gg["fwd_mid"] / gg["mid"])
        gg = gg.rename_axis("sec").reset_index()
        # Keep only seconds that actually had an observation (a real grid bar).
        out = gg[gg["sec"].isin(sel["sec"])][["sec", "fwd_ret_30s"]].copy()
        out["asset"] = asset
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["asset", "sec", "fwd_ret_30s"])
    return pd.concat(frames, ignore_index=True)


def _attach_labels(samples: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Join the forward-direction label by exact ``(asset, sample_sec)`` and set the
    window-outcome labels + the ``fwd_crosses_close`` flag."""
    out = samples.copy()
    out["sample_sec"] = out["sample_ts_ms"] // 1000
    out = out.merge(
        fwd.rename(columns={"sec": "sample_sec"}), on=["asset", "sample_sec"], how="left"
    )
    # 1 iff fwd_ret ≥ 0 (ties → Up), NA where the +horizon point was unobserved.
    out["fwd_up_30s"] = (out["fwd_ret_30s"] >= 0.0).astype("Int8").mask(
        out["fwd_ret_30s"].isna(), pd.NA
    )
    out["fwd_crosses_close"] = (out["sample_ts_ms"] + horizon * 1000) >= out["window_close_ms"]
    return out.drop(columns=["sample_sec"])


# --- sampling ---------------------------------------------------------------
def _sample_points(windows_df: pd.DataFrame, grid_secs: int) -> pd.DataFrame:
    """Emit ``sample_ts_ms = open + k·grid_secs·1000`` for ``k = 0,1,…`` while
    ``< close``, then rejoin the window-level fields."""
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
            "strike_quality", "label_source", "outcome", "outcome_up",
            "strike_distance_at_close", "strike_distance_at_close_vol",
            "in_live_coverage", "split"]
    samples = samples.merge(windows_df[keep], on=["series", "asset", "window_open_ms"], how="left")
    samples["elapsed_secs"] = (samples["sample_ts_ms"] - samples["window_open_ms"]) / 1000.0
    samples["tau_secs"] = (samples["window_close_ms"] - samples["sample_ts_ms"]) / 1000.0
    return samples


# --- the hard rule: runtime assertion --------------------------------------
def _assert_no_lookahead(df: pd.DataFrame) -> None:
    """Fail-closed if any sample's features could have used data after its timestamp.

    Checks: the matched feature bar is at-or-before the sample; the sample sits
    inside ``[open, close)``; the strike tick is at-or-before open; τ > 0, elapsed ≥ 0;
    and no column is both a feature and a label. A violation aborts the stage."""
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
    assert not (set(FEATURE_COLS) & set(LABEL_COLS)), "no-lookahead: feature/label column overlap"


# --- integrity diagnostics --------------------------------------------------
def _strike_outcome_agreement(windows_df: pd.DataFrame, grid: pd.DataFrame) -> float:
    """Fraction of windows where the reconstructed rule (end Chainlink ≥ strike ⇒ Up)
    agrees with the authoritative outcome — a sanity check on the reconstructed
    strike, reported in metadata (not a column)."""
    if windows_df.empty or grid.empty or "chainlink" not in grid.columns:
        return float("nan")
    cl = grid[["asset", "ts_ms", "chainlink"]].dropna().sort_values("ts_ms")
    if cl.empty:
        return float("nan")
    left = windows_df[["asset", "window_open_ms", "window_close_ms", "strike", "outcome_up"]].copy()
    left["end_ref"] = left["window_close_ms"].astype("int64") - 1
    left = left.sort_values("end_ref")
    merged = pd.merge_asof(
        left, cl.rename(columns={"ts_ms": "end_ref"}),
        on="end_ref", by="asset", direction="backward",
    )
    ok = merged.dropna(subset=["chainlink", "strike"])
    if ok.empty:
        return float("nan")
    recon_up = (ok["chainlink"] >= ok["strike"]).astype(int)
    return float((recon_up == ok["outcome_up"]).mean())


# --- historical (Binance aggTrades proxy) -----------------------------------
# These reconstruct 5-minute windows from the downloaded aggTrades store using
# Binance as a *proxy* for the Chainlink resolution feed. None of them is a feature
# function (the historical feature grid is built by the SAME scanned kernel via
# `_binance_ticks_from_bars` → `build_feature_grid`); the ones below read a window's
# close (the future) only to build the outcome + knife-edge *diagnostics*.
def _symbol_in_scope(symbol: str, series_filter: tuple[str, ...] | None) -> bool:
    """A downloaded symbol is in scope iff its series is (respecting ``--series``)."""
    series = SYMBOL_SERIES.get(symbol)
    return series is not None and _in_scope(series, series_filter)


def _day_open_ms(d: date) -> list[int]:
    """The 5-minute window opens (unix ms) on UTC date ``d`` — a real boundary grid."""
    start = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    return [(start + k * WINDOW_SECS) * 1000 for k in range(WINDOWS_PER_DAY)]


def _present_opens(days: list[date]) -> list[int]:
    """All window opens (unix ms) on the given present days, sorted."""
    opens: list[int] = []
    for d in days:
        opens.extend(_day_open_ms(d))
    opens.sort()
    return opens


def _hist_candidate_opens(hist_dir, symbol: str) -> list[int]:
    """Every 5m window open (unix ms) on the days present for ``symbol`` — from the
    filenames only (no trade data), used to precompute the chronological split."""
    opens: list[int] = []
    for p in ba.aggtrades_day_paths(hist_dir, symbol):
        d = ba.day_of_path(p)
        if d is not None:
            opens.extend(_day_open_ms(d))
    return opens


def _day_bars(path, symbol: str) -> pd.DataFrame:
    """Reduce one aggTrades day-parquet to 1-second bars (last trade price per
    second). Reads only ``(transact_time, price)`` — one day of raw trades at a time,
    so the whole history is never resident. Returns ``(sec, price)``."""
    df = ba.read_aggtrades_columns(path, ["transact_time", "price"])
    empty = pd.DataFrame({"sec": pd.Series([], dtype="int64"), "price": pd.Series([], dtype="float64")})
    if df.empty:
        return empty
    sec = df["transact_time"].to_numpy(dtype="int64") // 1_000_000  # µs → unix seconds
    price = df["price"].to_numpy(dtype=float)
    good = price > 0.0
    if not good.any():
        return empty
    bars = pd.DataFrame({"sec": sec[good], "price": price[good]})
    # File is trade-id (== time) ordered, so groupby-last is the last trade in the second.
    return bars.groupby("sec", as_index=False)["price"].last()


def _load_month_bars(load_paths, symbol: str) -> pd.DataFrame:
    """1-second bars for one month (+ buffer days), deduped by second and sorted.
    Only ~a month of bars is ever resident (each day reduced then discarded)."""
    frames = [_day_bars(p, symbol) for p in load_paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame({"sec": pd.Series([], dtype="int64"), "price": pd.Series([], dtype="float64")})
    allb = pd.concat(frames, ignore_index=True)
    return allb.groupby("sec", as_index=False)["price"].last().sort_values("sec").reset_index(drop=True)


def _month_batches(hist_dir, symbol: str):
    """Yield ``(year, month, load_paths, month_days)`` for one calendar month of
    present day-parquets at a time, plus one buffer day on each side (for EWMA warmup
    and the close / forward-label of windows that span the month edge)."""
    day_paths: dict[date, object] = {}
    for p in ba.aggtrades_day_paths(hist_dir, symbol):
        d = ba.day_of_path(p)
        if d is not None:
            day_paths[d] = p
    months: "OrderedDict[tuple[int, int], list[date]]" = OrderedDict()
    for d in sorted(day_paths):
        months.setdefault((d.year, d.month), []).append(d)
    for (yr, mo), mdays in months.items():
        prev = mdays[0] - timedelta(days=1)
        nxt = mdays[-1] + timedelta(days=1)
        load = [day_paths[d] for d in [prev, *mdays, nxt] if d in day_paths]
        yield yr, mo, load, mdays


def _window_close_and_vol(
    secs: np.ndarray, prices: np.ndarray, open_sec: int, close_sec: int
) -> tuple[float, float]:
    """Close price (last bar with ``sec ≤ close_sec − 1``) and realized 1-second vol
    (std of the in-``[open, close)`` 1-second log returns) for one window over a
    sorted bar series. ``(nan, nan)`` when the window has no usable bars.

    Reads the window's close (the future) — it feeds the label + knife-edge
    diagnostics only, never a feature — so it is kept out of the feature path."""
    if len(secs) == 0:
        return float("nan"), float("nan")
    j = int(np.searchsorted(secs, close_sec - 1, side="right")) - 1
    close_price = float(prices[j]) if j >= 0 else float("nan")
    lo = int(np.searchsorted(secs, open_sec, side="left"))
    hi = int(np.searchsorted(secs, close_sec, side="left"))
    win = prices[lo:hi]
    win = win[win > 0.0]
    realized = float(np.std(np.diff(np.log(win)))) if len(win) >= 2 else float("nan")
    return close_price, realized


def _strike_distance(strike: float, close_price: float, realized_1s: float) -> tuple[float, float]:
    """The close-to-strike gap, absolute and realized-vol-normalized.

    ``abs = |close − strike|`` (USD). ``vol = |log(close/strike)| / (σ_1s·√WINDOW)``
    — how many window-sigmas the resolution landed from the strike (small ⇒ knife
    edge). NaN when inputs are missing/degenerate."""
    if not (strike > 0.0 and close_price > 0.0):
        return float("nan"), float("nan")
    abs_d = abs(close_price - strike)
    if realized_1s > 0.0:  # nan/0 → the vol distance is undefined
        vol_d = abs(log(close_price / strike)) / (realized_1s * sqrt(WINDOW_SECS))
    else:
        vol_d = float("nan")
    return abs_d, vol_d


def _journal_resolution_distance(ticks_df: pd.DataFrame, jwindows: pd.DataFrame) -> pd.DataFrame:
    """Per journal window: the close-to-strike distance from the **Chainlink** feed
    (its real resolution source). Returns ``series, asset, window_open_ms`` + the two
    ``strike_distance_at_close*`` columns."""
    cols = ["series", "asset", "window_open_ms", "strike_distance_at_close", "strike_distance_at_close_vol"]
    if jwindows.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for asset in sorted(jwindows["asset"].unique()):
        cl = feat._chainlink_bars(ticks_df, asset)
        secs = cl["sec"].to_numpy(dtype="int64") if not cl.empty else np.empty(0, dtype="int64")
        prices = cl["chainlink"].to_numpy(dtype=float) if not cl.empty else np.empty(0, dtype=float)
        for w in jwindows[jwindows["asset"] == asset].itertuples(index=False):
            open_sec = int(w.window_open_ms) // 1000
            close_sec = int(w.window_close_ms) // 1000
            close_price, realized = _window_close_and_vol(secs, prices, open_sec, close_sec)
            strike = float(w.strike) if pd.notna(w.strike) else float("nan")
            abs_d, vol_d = _strike_distance(strike, close_price, realized)
            rows.append({
                "series": w.series, "asset": w.asset, "window_open_ms": int(w.window_open_ms),
                "strike_distance_at_close": abs_d, "strike_distance_at_close_vol": vol_d,
            })
    return pd.DataFrame(rows, columns=cols)


# The unified window-frame columns carried into `_sample_points` (both sources).
_WINDOW_FRAME_COLS = [
    "series", "asset", "window_open_ms", "window_close_ms", "up_token", "down_token",
    "strike", "strike_ts_ms", "strike_quality", "label_source", "outcome", "outcome_up",
    "strike_distance_at_close", "strike_distance_at_close_vol", "in_live_coverage",
]


def _historical_windows(
    bars: pd.DataFrame, mdays: list[date], symbol: str,
    journal_keys: set[tuple[str, int]], strike_tolerance_ms: float,
) -> tuple[pd.DataFrame, int]:
    """Reconstruct 5m windows for one month's present days from the Binance bar
    series: strike = last trade ``≤ open``, outcome = ``end ≥ strike ⇒ Up`` (§6 tie
    rule), plus the knife-edge distance. A window whose ``(series, open)`` collides
    with a journal window is dropped (the journal keeps its real Chainlink labels).
    Returns ``(windows_df, n_dedup_collisions)``."""
    series = SYMBOL_SERIES[symbol]
    asset = SYMBOL_ASSET[symbol]
    secs = bars["sec"].to_numpy(dtype="int64")
    prices = bars["price"].to_numpy(dtype=float)
    rows: list[dict] = []
    collisions = 0
    for open_ms in _present_opens(mdays):
        key = (series, int(open_ms))
        if key in journal_keys:
            collisions += 1
            continue
        if len(secs) == 0:
            continue
        open_sec = open_ms // 1000
        close_ms = open_ms + WINDOW_SECS * 1000
        close_sec = close_ms // 1000
        i = int(np.searchsorted(secs, open_sec, side="right")) - 1  # last bar ≤ open
        if i < 0:
            continue
        strike = float(prices[i])
        strike_ts = int(secs[i]) * 1000
        close_price, realized = _window_close_and_vol(secs, prices, open_sec, close_sec)
        if not (strike > 0.0 and close_price > 0.0):
            continue
        outcome = "Up" if close_price >= strike else "Down"
        quality = "boundary" if (open_ms - strike_ts) <= strike_tolerance_ms else "distant"
        abs_d, vol_d = _strike_distance(strike, close_price, realized)
        rows.append({
            "series": series, "asset": asset,
            "window_open_ms": int(open_ms), "window_close_ms": int(close_ms),
            "up_token": None, "down_token": None,
            "strike": strike, "strike_ts_ms": strike_ts, "strike_quality": quality,
            "label_source": "binance_proxy",
            "outcome": outcome, "outcome_up": 1 if outcome == "Up" else 0,
            "strike_distance_at_close": abs_d, "strike_distance_at_close_vol": vol_d,
            "in_live_coverage": False,
        })
    df = pd.DataFrame(rows, columns=_WINDOW_FRAME_COLS).sort_values("window_open_ms").reset_index(drop=True)
    return df, collisions


def _binance_ticks_from_bars(bars: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Reshape 1-second Binance bars into the ``TICK_COLS`` shape as ``BinanceDirect``
    ``Mid`` ticks, so the historical feature grid is built by the SAME no-look-ahead
    kernel (`features.build_asset_features`) as the journal path — no separate,
    un-scanned feature code. Pure 1:1 reshape; reads no future."""
    ts = bars["sec"].to_numpy(dtype="int64") * 1000
    n = len(ts)
    return pd.DataFrame(
        {
            "ts_local_ms": ts,
            "source": np.repeat("BinanceDirect", n),
            "asset": np.repeat(asset, n),
            "kind": np.repeat("Mid", n),
            "value": bars["price"].to_numpy(dtype=float),
            "ts_exchange": ts,
            "ts_local": ts,
        },
        columns=TICK_COLS,
    )


# --- chronological split (per series, computed independently per source) -----
def _split_cutoffs(opens_by_series: dict[str, list[int]], val_fraction: float) -> dict[str, int]:
    """Per series, the ``open_ms`` at/after which windows fall in the ``val`` split
    (the latest ``ceil(val_fraction·n)`` opens). Empty when no val is assigned."""
    cutoffs: dict[str, int] = {}
    if val_fraction <= 0.0:
        return cutoffs
    for series, opens in opens_by_series.items():
        uniq = sorted({int(o) for o in opens})
        n = len(uniq)
        n_val = min(n, ceil(val_fraction * n))
        if n_val > 0:
            cutoffs[series] = uniq[n - n_val]
    return cutoffs


def _split_by_cutoff(windows_df: pd.DataFrame, cutoffs: dict[str, int]) -> pd.Series:
    """Assign ``train``/``val`` by the precomputed per-series cutoff (``≥ cutoff`` → val)."""
    split = pd.Series("train", index=windows_df.index, dtype="object")
    for series, cut in cutoffs.items():
        split.loc[(windows_df["series"] == series) & (windows_df["window_open_ms"] >= cut)] = "val"
    return split


# --- shared sampling + streaming write --------------------------------------
def _build_samples(
    windows_df: pd.DataFrame, grid: pd.DataFrame, grid_secs: int, horizon_secs: int,
    max_staleness_ms: int,
) -> pd.DataFrame:
    """Windows + feature grid → finalized sample rows (sampling → as-of features →
    forward + outcome labels). The single sampling/feature path, shared by the
    journal and historical batches so both get identical no-look-ahead treatment."""
    samples = _sample_points(windows_df, grid_secs)
    if samples.empty:
        return _finalize(samples)
    samples = _attach_features(samples, grid, max_staleness_ms)
    fwd = forward_direction_labels(grid, horizon_secs)
    samples = _attach_labels(samples, fwd, horizon_secs)
    return _finalize(samples)


_ARROW_TYPES = {
    "string": pa.string(), "int64": pa.int64(), "int32": pa.int32(), "int8": pa.int8(),
    "Int64": pa.int64(), "Int8": pa.int8(), "float64": pa.float64(), "bool": pa.bool_(),
}


def _arrow_schema() -> pa.Schema:
    """The parquet schema derived from ``COLUMN_SPEC`` (for streamed writes)."""
    return pa.schema([(name, _ARROW_TYPES[dtype]) for name, dtype, _cat, _doc in COLUMN_SPEC])


# --- worker -----------------------------------------------------------------
def dataset(
    paths: Paths,
    *,
    grid_secs: int = DEFAULT_GRID_SECS,
    horizon_secs: int = DEFAULT_HORIZON_SECS,
    series: tuple[str, ...] | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    strike_lookback_secs: int = DEFAULT_STRIKE_LOOKBACK_SECS,
    strike_tolerance_secs: float = DEFAULT_STRIKE_TOLERANCE_SECS,
    max_feature_staleness_secs: int = DEFAULT_MAX_FEATURE_STALENESS_SECS,
    include_history: bool = True,
) -> dict:
    """Builds the window-aligned training dataset — the journal (Chainlink) windows
    plus, when ``include_history`` and an aggTrades store is present, the historical
    Binance-proxy windows streamed **month by month**. Writes ``out/dataset.parquet``
    + ``out/dataset/{SCHEMA.md, metadata.json}``; returns ``{params, counts,
    integrity, history, journal_subset}``."""
    paths.ensure_out()
    out_dir = paths.out_dir / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "grid_secs": int(grid_secs),
        "horizon_secs": int(horizon_secs),
        "series_filter": list(series) if series else None,
        "val_fraction": float(val_fraction),
        "strike_lookback_secs": int(strike_lookback_secs),
        "strike_tolerance_secs": float(strike_tolerance_secs),
        "max_feature_staleness_secs": int(max_feature_staleness_secs),
        "include_history": bool(include_history),
    }
    strike_tol_ms = strike_tolerance_secs * 1000.0
    max_staleness_ms = max_feature_staleness_secs * 1000

    # --- journal (Chainlink strike/outcome) — the unchanged path -------------
    ticks_df, win_meta, res, settle, top_ts = _read_journal(paths, series)
    jwindows = _windows_frame(win_meta, res, settle)
    journal_keys: set[tuple[str, int]] = (
        set(zip(jwindows["series"], jwindows["window_open_ms"].astype("int64").tolist()))
        if not jwindows.empty else set()
    )
    jgrid = pd.DataFrame(columns=[*feat.FEATURE_COLS, "sigma_1s"])
    if not jwindows.empty:
        strikes = reconstruct_strikes(ticks_df, jwindows, strike_lookback_secs * 1000, strike_tol_ms)
        jwindows = jwindows.merge(strikes, on=["series", "asset", "window_open_ms"], how="left")
        jwindows["in_live_coverage"] = _coverage_flags(jwindows, top_ts)
        jwindows["split"] = _assign_split(jwindows, val_fraction)  # journal split — unchanged
        jwindows["label_source"] = "chainlink"
        jdist = _journal_resolution_distance(ticks_df, jwindows)
        jwindows = jwindows.merge(jdist, on=["series", "asset", "window_open_ms"], how="left")
        depth_df = pd.DataFrame(list(depth_io.read_depth_rows(paths.depth_dir)), columns=DEPTH_COLS)
        grids = [build_feature_grid(ticks_df, depth_df, a) for a in sorted(jwindows["asset"].unique())]
        grids = [g for g in grids if not g.empty]
        if grids:
            jgrid = pd.concat(grids, ignore_index=True)
    integrity_agreement = _strike_outcome_agreement(jwindows, jgrid) if not jwindows.empty else float("nan")

    # --- accumulators + streaming writer -------------------------------------
    schema = _arrow_schema()
    writer = pq.ParquetWriter(paths.table("dataset"), schema)
    tot = defaultdict(int)
    lag_max_ms = 0
    assets_seen: set[str] = set()
    series_seen: set[str] = set()
    per_symbol_month: list[dict] = []
    knife_counts = {t: 0 for t in KNIFE_EDGE_VOL_THRESHOLDS}
    knife_scored = 0
    proxy_windows_total = 0
    dedup_collisions = 0

    def _write_batch(batch: pd.DataFrame, source_label: str) -> None:
        nonlocal lag_max_ms
        if batch.empty:
            return
        _assert_no_lookahead(batch)  # the hard rule runs over every batch, both sources
        writer.write_table(pa.Table.from_pandas(batch, schema=schema, preserve_index=False))
        tot["samples"] += len(batch)
        tot["train"] += int((batch["split"] == "train").sum())
        tot["val"] += int((batch["split"] == "val").sum())
        tot["in_cov"] += int(batch["in_live_coverage"].sum())
        tot["no_feat"] += int(batch["feature_asof_ts_ms"].isna().sum())
        tot["missing_fwd"] += int(batch["fwd_up_30s"].isna().sum())
        tot["crossing"] += int(batch["fwd_crosses_close"].sum())
        tot[f"samp_{source_label}"] += len(batch)
        lag = (batch["sample_ts_ms"].astype("int64") - batch["feature_asof_ts_ms"]).dropna()
        if len(lag):
            lag_max_ms = max(lag_max_ms, int(lag.max()))

    def _count_windows(wdf: pd.DataFrame, source_label: str) -> None:
        tot["windows"] += len(wdf)
        tot[f"win_{source_label}"] += len(wdf)
        tot["strike_missing"] += int((wdf["strike_quality"] == "missing").sum())
        tot["strike_distant"] += int((wdf["strike_quality"] == "distant").sum())
        assets_seen.update(wdf["asset"].unique())
        series_seen.update(wdf["series"].unique())

    # journal batch first — its rows are produced by the unchanged path.
    if not jwindows.empty:
        _write_batch(_build_samples(jwindows, jgrid, grid_secs, horizon_secs, max_staleness_ms), "chainlink")
        _count_windows(jwindows, "chainlink")

    # historical batches — one (symbol, calendar month) at a time.
    if include_history and ba.has_aggtrades(paths.hist_dir):
        symbols = [s for s in ba.symbols_present(paths.hist_dir) if _symbol_in_scope(s, series)]
        hist_opens: dict[str, list[int]] = {}
        for sym in symbols:
            hist_opens.setdefault(SYMBOL_SERIES[sym], []).extend(_hist_candidate_opens(paths.hist_dir, sym))
        hist_cutoffs = _split_cutoffs(hist_opens, val_fraction)  # split independent of the journal
        for sym in symbols:
            asset = SYMBOL_ASSET[sym]
            for yr, mo, load_paths, mdays in _month_batches(paths.hist_dir, sym):
                bars = _load_month_bars(load_paths, sym)
                if bars.empty:
                    continue
                hwindows, collisions = _historical_windows(bars, mdays, sym, journal_keys, strike_tol_ms)
                dedup_collisions += collisions
                if hwindows.empty:
                    continue
                hwindows["split"] = _split_by_cutoff(hwindows, hist_cutoffs)
                grid = build_feature_grid(_binance_ticks_from_bars(bars, asset),
                                          pd.DataFrame(columns=DEPTH_COLS), asset)
                hbatch = _build_samples(hwindows, grid, grid_secs, horizon_secs, max_staleness_ms)
                _write_batch(hbatch, "binance_proxy")
                _count_windows(hwindows, "binance_proxy")
                vd = hwindows["strike_distance_at_close_vol"].to_numpy(dtype=float)
                finite = vd[np.isfinite(vd)]
                proxy_windows_total += len(hwindows)
                knife_scored += len(finite)
                for t in KNIFE_EDGE_VOL_THRESHOLDS:
                    knife_counts[t] += int((finite <= t).sum())
                per_symbol_month.append({
                    "symbol": sym, "month": f"{yr:04d}-{mo:02d}",
                    "windows": int(len(hwindows)), "samples": int(len(hbatch)),
                })

    if tot["samples"] == 0:
        writer.write_table(schema.empty_table())  # a valid, 0-row parquet with the schema
    writer.close()

    counts = {
        "windows": int(tot["windows"]),
        "windows_resolved": int(tot["windows"]),
        "samples": int(tot["samples"]),
        "samples_train": int(tot["train"]),
        "samples_val": int(tot["val"]),
        "samples_in_coverage": int(tot["in_cov"]),
        "samples_no_features": int(tot["no_feat"]),
        "samples_missing_fwd": int(tot["missing_fwd"]),
        "samples_crossing_close": int(tot["crossing"]),
        "strike_missing": int(tot["strike_missing"]),
        "strike_distant": int(tot["strike_distant"]),
        "windows_chainlink": int(tot["win_chainlink"]),
        "windows_binance_proxy": int(tot["win_binance_proxy"]),
        "samples_chainlink": int(tot["samp_chainlink"]),
        "samples_binance_proxy": int(tot["samp_binance_proxy"]),
    }
    history = {
        "included": bool(include_history),
        "symbols_present": ba.symbols_present(paths.hist_dir),
        "per_symbol_month": per_symbol_month,
        "knife_edge": {
            "vol_thresholds": list(KNIFE_EDGE_VOL_THRESHOLDS),
            "proxy_windows_scored": int(knife_scored),
            "proxy_windows_total": int(proxy_windows_total),
            "proportion_within": {
                f"{t:g}": (knife_counts[t] / knife_scored if knife_scored else float("nan"))
                for t in KNIFE_EDGE_VOL_THRESHOLDS
            },
        },
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
        "integrity": {
            "strike_outcome_agreement": integrity_agreement,
            "feature_asof_max_lag_secs": (lag_max_ms / 1000.0) if lag_max_ms else float("nan"),
        },
        "assets": sorted(assets_seen),
        "series": sorted(series_seen),
        "history": history,
        "journal_subset": journal_subset,
    }
    _write_metadata(out_dir, result)
    _write_schema(out_dir)
    return result


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


# --- outputs ----------------------------------------------------------------
def _write_metadata(out_dir, result: dict) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _write_schema(out_dir) -> None:
    cat_label = {
        "id": "identifier", "meta": "metadata (known at sample time)",
        "feature": "**feature** (as-of ≤ sample_ts)", "label": "**label** (reads the future)",
        "diag": "**diagnostic** (reads the future; for filtering/weighting, not an input)",
        "coverage": "coverage",
    }
    lines = [
        "# `dataset.parquet` — schema",
        "",
        "One row per (window, feature-timestamp). Built by `python -m model_lab.dataset`.",
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
        "data after that row's `sample_ts_ms`.** Concretely —",
        "",
        "- Every feature comes from a per-second bar matched with a **backward** "
        "`merge_asof` (`feature_asof_ts_ms ≤ sample_ts_ms`); the EWMA vol / `sigma_1s` "
        "are causal; `strike` is captured at `window_open_ms ≤ sample_ts_ms`; `tau_secs` "
        "and `fwd_crosses_close` are known at sample time.",
        "- The **label** columns (`fwd_ret_30s`, `fwd_up_30s`, `outcome`, `outcome_up`) "
        "and the **diagnostic** columns (`strike_distance_at_close*`) deliberately read "
        "the future and must never be used as inputs.",
        "- Enforced by (a) a runtime assertion in `dataset._assert_no_lookahead`, "
        "(b) an AST source scan (`tests/test_dataset.py::test_no_lookahead_ast_scan`) "
        "over the feature functions, and (c) a differential perturbation test "
        "(`…::test_no_lookahead_differential`, plus `…::test_no_lookahead_historical` for "
        "the Binance-proxy path) that corrupts all future data and asserts the features "
        "are bit-identical. Historical features go through the **same** feature kernel as "
        "the journal path (synthetic Binance-mid ticks → `features.build_asset_features`), "
        "so the scan covers both sources.",
        "",
        "## Conventions",
        "",
        "- **`label_source`:** `chainlink` for windows we recorded live (real Chainlink "
        "strike + `Resolved` outcome) and `binance_proxy` for historical windows "
        "reconstructed from the Binance aggTrades archive. Proxy windows have **no** "
        "Chainlink/depth feed, so `chainlink`, `basis_bps`, `imbalance`, `microprice`, "
        "`spread`, `depth_mid` are `NaN` there; `mid`, `sigma_1s`, `log_s_k`, `z`, "
        "`p_up_model` are still populated (price-only). A historical window that collides "
        "with a journal window is dropped, so the `chainlink` subset is never altered.",
        "- **Proxy strike/outcome:** for `binance_proxy`, `strike` = last Binance trade "
        "at-or-before open and `outcome` = `end ≥ strike ⇒ Up` on Binance — a proxy for "
        "the true Chainlink resolution (~13 bps basis apart).",
        "- **Knife-edge (`strike_distance_at_close*`):** `strike_distance_at_close` is "
        "`|close − strike|` (USD) and `strike_distance_at_close_vol` normalizes it by the "
        "window's realized vol (`σ_realized_1s·√window`). A small vol distance on a "
        "`binance_proxy` row means the Binance↔Chainlink basis could flip its outcome — "
        "filter or down-weight those. The primary, proxy-safe target is `fwd_up_30s` "
        "(price-only, both sources).",
        "- **Direction tie rule:** `fwd_up_30s = 1` iff `fwd_ret_30s ≥ 0` (ties → Up), "
        "matching the venue's `end ≥ strike ⇒ Up` resolution rule.",
        "- **`in_live_coverage`:** true iff a Polymarket `top_of_book` for the window's "
        "token was recorded inside `[open, close)` — i.e. a real market-mid benchmark "
        "exists for that window. Always `False` for `binance_proxy`.",
        "- **`split`:** per-series chronological — the latest `val_fraction` of each "
        "series' windows are `val` (train on the past, validate on the future); no "
        "sample straddles the boundary (window-level split). The `chainlink` and "
        "`binance_proxy` windows are split **independently**, so adding history never "
        "reshuffles the journal subset's split.",
        "- **`fwd_*` NaN/NA:** a sample whose `+horizon`s point is unobserved (end of the "
        "journal, or an internal gap) has `NA` forward labels but keeps its window outcome.",
        "- **Stale features:** if the nearest at-or-before feature bar is older than "
        "`max_feature_staleness_secs` (a relic across a recording gap), the feature "
        "columns are nulled and `feature_asof_ts_ms` is `NA` — the row and its outcome "
        "label are kept. Downstream can also filter on `sample_ts_ms − feature_asof_ts_ms`.",
        "- **Strike rule:** `LastAtOrBefore` the resolution feed (Chainlink Vendor for "
        "journal windows, Binance last trade for proxy windows). A Binance-candle 1h "
        "series would need `FirstAtOrAfter` and is out of scope of the default (5m) run.",
        "",
    ]
    (out_dir / "SCHEMA.md").write_text("\n".join(lines), encoding="utf-8")


# --- entry ------------------------------------------------------------------
def _parse_series(val: str | None) -> tuple[str, ...] | None:
    if not val:
        return None
    return tuple(s.strip() for s in val.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "dataset")
    parser.add_argument("--grid-secs", type=int, default=DEFAULT_GRID_SECS,
                        help=f"sampling cadence within each window (default {DEFAULT_GRID_SECS})")
    parser.add_argument("--horizon-secs", type=int, default=DEFAULT_HORIZON_SECS,
                        help=f"forward-direction label horizon (default {DEFAULT_HORIZON_SECS})")
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
    parser.add_argument("--no-history", action="store_true",
                        help="journal-only: skip the Binance aggTrades historical proxy windows")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    print(f"[dataset] journal={paths.journal_dir}")
    print(f"[dataset] hist   ={paths.hist_dir} (history {'off' if args.no_history else 'on'})")
    print(f"[dataset] out    ={paths.table('dataset')}")

    result = dataset(
        paths,
        grid_secs=args.grid_secs,
        horizon_secs=args.horizon_secs,
        series=_parse_series(args.series),
        val_fraction=args.val_fraction,
        strike_lookback_secs=args.strike_lookback_secs,
        strike_tolerance_secs=args.strike_tolerance_secs,
        max_feature_staleness_secs=args.max_feature_staleness_secs,
        include_history=not args.no_history,
    )
    c = result["counts"]
    print(f"[dataset] {c['windows']:,} windows -> {c['samples']:,} samples "
          f"({c['samples_train']:,} train / {c['samples_val']:,} val) over {result['series']}")
    print(f"[dataset] source: chainlink={c['windows_chainlink']:,} win / {c['samples_chainlink']:,} samp"
          f" · binance_proxy={c['windows_binance_proxy']:,} win / {c['samples_binance_proxy']:,} samp")
    print(f"[dataset] in-coverage samples={c['samples_in_coverage']:,} "
          f"no-features={c['samples_no_features']:,} missing-fwd={c['samples_missing_fwd']:,} "
          f"crossing-close={c['samples_crossing_close']:,}")
    print(f"[dataset] strike: missing={c['strike_missing']} distant={c['strike_distant']} "
          f"· strike↔outcome agreement={result['integrity']['strike_outcome_agreement']}")

    h = result["history"]
    js = result["journal_subset"]
    if h["per_symbol_month"]:
        print(f"[dataset] history symbols={h['symbols_present']} "
              f"over {len(h['per_symbol_month'])} symbol-months:")
        for row in h["per_symbol_month"]:
            print(f"[dataset]   {row['symbol']} {row['month']}: "
                  f"{row['windows']:,} windows / {row['samples']:,} samples")
        ke = h["knife_edge"]
        prop = ", ".join(f"≤{t:g}σ: {ke['proportion_within'][f'{t:g}']:.1%}"
                         for t in KNIFE_EDGE_VOL_THRESHOLDS)
        print(f"[dataset] knife-edge (of {ke['proxy_windows_scored']:,} scored proxy windows): {prop}")
    elif not args.no_history:
        print(f"[dataset] history: no aggTrades store at {paths.hist_dir} "
              f"(run `python -m model_lab.hist` to download) — journal-only.")
    print(f"[dataset] journal subset: {js['windows']:,} windows / {js['samples']:,} samples, "
          f"unchanged (dedup dropped {js['dedup_collisions']:,} colliding proxy windows)")
    print(f"[dataset] wrote {paths.table('dataset').name} + dataset/SCHEMA.md, metadata.json")
    if c["samples"] == 0:
        print("[dataset] WARNING: no samples produced — check --journal-dir / --series.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
