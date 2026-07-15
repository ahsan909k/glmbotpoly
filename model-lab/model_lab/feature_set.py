"""Stage — feature_set.

Build a **curated, learner-ready feature matrix** on top of the window-aligned
``dataset.parquet``. One row per (window, feature-timestamp) — the *same* rows as
the dataset — augmented with the engineered predictors a directional model wants:

- **multi-horizon returns** ``ret_{1,5,15,60}s`` — the log return of the Binance
  mid over the last h seconds (as-of);
- **signed aggressor flow imbalance** ``flow_imb_{30,120,300}s`` — the net
  taker-buy-minus-sell volume over a trailing lookback, in ``[-1, 1]``, from the
  Binance aggTrades archive (``is_buyer_maker`` + ``quantity``);
- **trade intensity** ``trade_intensity_{30,120,300}s`` — Binance trades per
  second over a trailing lookback;
- **fast/slow EWMA volatility + ratio** ``sigma_fast`` (halflife 10 s),
  ``sigma_slow`` (halflife 60 s = the engine ``sigma_1s``), ``vol_ratio``;
- the **formula z-score** ``z = log(mid/strike)/(sigma_slow·√τ)`` (distance from
  the strike, scaled by vol and time remaining);
- **seconds_remaining** (= τ) and **hour_of_day** (0–23, UTC).

The feature list lives in **one config** (:data:`FEATURE_SPEC`, expanded from the
:data:`RETURN_HORIZONS_SECS` / :data:`FLOW_LOOKBACKS_SECS` / :data:`VOL_HALFLIVES`
knob lists), so adding or removing a feature is a near-one-line change; that single
list drives the output columns, the parquet schema, the ``SCHEMA.md`` doc, and the
sanity report's bounds/coverage expectations.

**Hard rule — no look-ahead.** As in the dataset stage, no value in a feature
column may derive from data after that row's ``sample_ts_ms``. The engineered
features are built from per-second grids with strictly-backward transforms
(positive shift, causal EWMA, trailing rolling, backward ``merge_asof``) and are
covered by the same three guards: a runtime assertion here
(:func:`_assert_no_lookahead`), the AST source scan
(``tests/test_dataset.py::test_no_lookahead_ast_scan`` — this stage's grid kernels
are appended to its ``_FEATURE_FUNCS``), and a differential/corruption test
(``tests/test_feature_set.py``).

**Coverage.** Flow/intensity exist only where the aggTrades archive covers the
day; returns/vols/z warm up over the first seconds of a run. Those NaN are
**expected** (classified exactly from ``mid_fresh`` / ``flow_covered`` /
``price_run_secs``) and the "no NaNs" sanity check fails only on *excess*
(unexpected) NaN. NaN stays in the parquet — a learner must be able to tell real
from missing.

Self-contained on the dataset: reads ``out/dataset.parquet`` for the vetted sample
skeleton and rebuilds the per-second grids from the journal + aggTrades archive
(streamed month-by-month for history). Outputs ``out/feature_set.parquet`` +
``out/feature_set/{SCHEMA.md, sanity.json, sanity.html, metadata.json}``.

Run::

    python -m model_lab.feature_set
    python -m model_lab.feature_set --series BTC-5m,ETH-5m --max-feature-staleness-secs 120
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG, never open a window.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from . import dataset as ds  # noqa: E402
from . import features as feat  # noqa: E402
from .config import (  # noqa: E402
    ParquetNotReady, Paths, assert_parquet_ready, require_tables, resolve_bounds, resolve_paths,
    stage_parser,
)
from .io import binance_archive as ba  # noqa: E402
from .lib import math as lm  # noqa: E402
from .lib import procmem  # noqa: E402

# --- config: the single source of truth for the feature set -----------------
# Adding a horizon / lookback / halflife is a one-line edit to these lists.
RETURN_HORIZONS_SECS = [1, 5, 15, 60]
FLOW_LOOKBACKS_SECS = [30, 120, 300]
INTENSITY_LOOKBACKS_SECS = FLOW_LOOKBACKS_SECS
VOL_HALFLIVES = {"fast": 10.0, "slow": 60.0}  # slow = engine sigma_1s halflife.

# Reused engine/dataset constants.
WINDOW_SECS = ds.WINDOW_SECS  # 300
MAX_GAP_SECS = lm.DEFAULT_MAX_GAP_SECS  # 30 — a run resets across a longer gap.
WARMUP_RETURNS = lm.DEFAULT_WARMUP_RETURNS  # 60 — EWMA warmup, folded returns.
VOL_CAP = lm.DEFAULT_VOL_CAP_1S  # sigma is clamped to this ceiling.
Z_CLAMP = lm.Z_CLAMP  # 39.0
DEFAULT_MAX_STALENESS_SECS = ds.DEFAULT_MAX_FEATURE_STALENESS_SECS  # 120

# Sanity thresholds.
NAN_WARN_FRAC = 0.005  # excess-NaN fraction: ≤ this WARNs, above FAILs.
OUTLIER_WARN_FRAC = 0.005  # soft-bound outlier fraction: ≤ this WARNs, above FAILs.
RET_ABS_BOUND = 0.05  # |log return| beyond this is "absurd" (a 5% move ≤ 60 s).
VOL_RATIO_HI = 50.0  # fast/slow ratio soft ceiling.
INTENSITY_HI = 2000.0  # trades/sec soft ceiling (BTC peaks ~40/s).

# aggTrades symbol ↔ series/asset (reuse dataset's maps; add the reverse).
ASSET_SYMBOL = {"btc": "BTCUSDT", "eth": "ETHUSDT"}


@dataclass(frozen=True)
class FeatureSpec:
    """One engineered feature: its column name, dtype, builder family, and the
    doc + sanity expectations the SCHEMA.md and sanity report derive from it."""

    name: str
    dtype: str
    family: str  # "ret" | "sigma" | "vol_ratio" | "flow_imb" | "trade_intensity" | "zscore" | "time"
    param: float | int | None  # horizon / lookback / halflife; None for scalars.
    coverage: str  # "full" | "partial" (partial = aggTrades-only).
    hard_bounds: bool  # True ⇒ any out-of-[lo,hi] value is a FAIL (a math-bounded feature).
    lo: float
    hi: float
    doc: str


def _build_feature_spec() -> list[FeatureSpec]:
    """Expand the knob lists into the full ordered feature spec."""
    specs: list[FeatureSpec] = []
    for h in RETURN_HORIZONS_SECS:
        specs.append(FeatureSpec(
            f"ret_{h}s", "float64", "ret", h, "full", False, -RET_ABS_BOUND, RET_ABS_BOUND,
            f"Log return of the Binance mid over the last {h} second(s), as-of the sample."))
    specs.append(FeatureSpec(
        "sigma_fast", "float64", "sigma", VOL_HALFLIVES["fast"], "full", False, 0.0, VOL_CAP,
        f"Fast EWMA of 1-second log-return volatility (halflife {VOL_HALFLIVES['fast']:g}s), as-of."))
    specs.append(FeatureSpec(
        "sigma_slow", "float64", "sigma", VOL_HALFLIVES["slow"], "full", False, 0.0, VOL_CAP,
        f"Slow EWMA of 1-second log-return volatility (halflife {VOL_HALFLIVES['slow']:g}s = engine σ_1s), as-of."))
    specs.append(FeatureSpec(
        "vol_ratio", "float64", "vol_ratio", None, "full", False, 0.0, VOL_RATIO_HI,
        "sigma_fast / sigma_slow — > 1 means volatility is rising above its slow baseline."))
    for lb in FLOW_LOOKBACKS_SECS:
        specs.append(FeatureSpec(
            f"flow_imb_{lb}s", "float64", "flow_imb", lb, "partial", True, -1.0, 1.0,
            f"Signed aggressor (taker) volume imbalance over the last {lb}s: "
            f"(buy − sell qty)/(buy + sell qty) ∈ [−1, 1], from Binance aggTrades."))
    for lb in INTENSITY_LOOKBACKS_SECS:
        specs.append(FeatureSpec(
            f"trade_intensity_{lb}s", "float64", "trade_intensity", lb, "partial", False, 0.0, INTENSITY_HI,
            f"Binance trades per second over the last {lb}s (aggTrades count / lookback)."))
    specs.append(FeatureSpec(
        "z", "float64", "zscore", None, "full", True, -Z_CLAMP, Z_CLAMP,
        "Formula z-score log(mid/strike)/(σ_slow·√τ) — distance from the strike in vol-time units (clipped ±39)."))
    specs.append(FeatureSpec(
        "seconds_remaining", "float64", "time", None, "full", True, 0.0, float(WINDOW_SECS),
        "Seconds until the window closes (= τ). Known at sample time."))
    specs.append(FeatureSpec(
        "hour_of_day", "int16", "time", None, "full", True, 0.0, 23.0,
        "UTC hour of the sample timestamp (0–23) — a crude time-of-day feature."))
    return specs


FEATURE_SPEC = _build_feature_spec()
FEATURE_NAMES = [s.name for s in FEATURE_SPEC]
_RET_NAMES = [f"ret_{h}s" for h in RETURN_HORIZONS_SECS]
_FLOW_IMB_NAMES = [f"flow_imb_{lb}s" for lb in FLOW_LOOKBACKS_SECS]
_INTENSITY_NAMES = [f"trade_intensity_{lb}s" for lb in INTENSITY_LOOKBACKS_SECS]

# Carried verbatim from the (vetted) dataset skeleton — ids, framing, labels, split.
PASSTHROUGH_COLS = [
    "series", "asset", "window_open_ms", "window_close_ms", "sample_ts_ms", "sample_idx",
    "elapsed_secs", "tau_secs", "strike", "label_source", "split", "in_live_coverage",
    "fwd_up_30s", "outcome_up",
]
CONTEXT_COLS = ["mid"]  # the rebuilt as-of Binance mid (context for z / returns).
DIAG_COLS = ["price_asof_ts_ms", "flow_asof_ts_ms", "price_run_secs", "mid_fresh", "flow_covered"]
OUTPUT_COLUMNS = PASSTHROUGH_COLS + CONTEXT_COLS + FEATURE_NAMES + DIAG_COLS

DTYPES: dict[str, str] = {
    "series": "string", "asset": "string", "window_open_ms": "int64", "window_close_ms": "int64",
    "sample_ts_ms": "int64", "sample_idx": "int32", "elapsed_secs": "float64", "tau_secs": "float64",
    "strike": "float64", "label_source": "string", "split": "string", "in_live_coverage": "bool",
    "fwd_up_30s": "Int8", "outcome_up": "int8", "mid": "float64",
    "price_asof_ts_ms": "Int64", "flow_asof_ts_ms": "Int64", "price_run_secs": "Int64",
    "mid_fresh": "bool", "flow_covered": "bool",
    **{s.name: s.dtype for s in FEATURE_SPEC},
}

# The as-of feature values the price / flow grids contribute (nulled when stale).
_PRICE_VALUE_COLS = ["mid", *_RET_NAMES, "sigma_fast", "sigma_slow", "vol_ratio"]
_FLOW_VALUE_COLS = [*_FLOW_IMB_NAMES, *_INTENSITY_NAMES]
# Read from the skeleton: the passthrough columns + the three cross-check columns.
# `basis_ewma` is the dataset's rolling log-basis (EWMA(ln(chainlink/mid))) at each
# sample; read it (not renamed / not cross-checked) and fold it into the rebuilt z so
# the feature matrix carries the same basis-corrected z the dataset skeleton does.
_SKELETON_READ_COLS = PASSTHROUGH_COLS + ["mid", "sigma_1s", "z", "basis_ewma"]

_ARROW_TYPES = {
    "string": pa.string(), "int64": pa.int64(), "int32": pa.int32(), "int16": pa.int16(),
    "int8": pa.int8(), "Int64": pa.int64(), "Int32": pa.int32(), "Int8": pa.int8(),
    "float64": pa.float64(), "bool": pa.bool_(),
}


def _arrow_schema() -> pa.Schema:
    """The parquet schema derived from :data:`OUTPUT_COLUMNS` / :data:`DTYPES`."""
    return pa.schema([(name, _ARROW_TYPES[DTYPES[name]]) for name in OUTPUT_COLUMNS])


# --- grid kernels (scanned for no-look-ahead) -------------------------------
def _run_lengths(bar_secs: np.ndarray, max_gap: int = MAX_GAP_SECS) -> np.ndarray:
    """Length of the contiguous observed-second run ending at each bar; the run
    **resets** whenever the gap to the previous bar exceeds ``max_gap`` (mirroring
    the engine EWMA's re-anchor). Depends only on past bars — no look-ahead."""
    n = len(bar_secs)
    run = np.zeros(n, dtype=np.int64)
    if n == 0:
        return run
    run[0] = 1
    for i in range(1, n):
        gap = int(bar_secs[i]) - int(bar_secs[i - 1])
        run[i] = run[i - 1] + 1 if 0 < gap <= max_gap else 1
    return run


def mid_return_grid(
    bar_secs: np.ndarray, bar_prices: np.ndarray, horizons=RETURN_HORIZONS_SECS
) -> dict[str, np.ndarray]:
    """Multi-horizon log returns of the mid at each bar, aligned to ``bar_secs``.

    Reindexes to a gap-free per-second grid, forward-fills the mid (past-only), and
    for each horizon ``h`` takes ``log(mid[s]) − log(mid[s−h])`` — a **positive**
    look-back. A value is nulled when the contiguous run (within ``MAX_GAP_SECS``)
    is not at least ``h`` seconds long, so no return spans a recording gap. Every
    output at second ``s`` reads only bars at ``≤ s`` — no look-ahead."""
    out = {f"ret_{h}s": np.full(len(bar_secs), np.nan, dtype=float) for h in horizons}
    n = len(bar_secs)
    if n == 0:
        return out
    lo, hi = int(bar_secs[0]), int(bar_secs[-1])
    span = hi - lo + 1
    pos = np.asarray(bar_secs, dtype=np.int64) - lo
    dense = np.full(span, np.nan, dtype=float)
    dense[pos] = np.asarray(bar_prices, dtype=float)
    # Causal forward-fill: propagate the last observed price forward only.
    have = ~np.isnan(dense)
    ref = np.where(have, np.arange(span), 0)
    np.maximum.accumulate(ref, out=ref)
    dense = dense[ref]
    run = _run_lengths(np.asarray(bar_secs, dtype=np.int64))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mid = np.log(dense)
    for h in horizons:
        r = np.full(span, np.nan, dtype=float)
        r[h:] = log_mid[h:] - log_mid[:-h]
        vals = r[pos]
        vals[run <= h] = np.nan  # fewer than h+1 contiguous seconds → undefined
        out[f"ret_{h}s"] = vals
    return out


def ewma_vol_grid(
    bar_secs: np.ndarray, bar_prices: np.ndarray, halflives=VOL_HALFLIVES
) -> dict[str, np.ndarray]:
    """Fast + slow causal EWMA volatility and their ratio, on the **raw** bars.

    Two calls to the engine ``sigma_1s_from_bars`` (gap-aware, bias-corrected) — the
    slow halflife reproduces the engine ``sigma_1s``. ``vol_ratio = fast/slow``.
    Each value depends only on bars at or before it — no look-ahead."""
    sec = np.asarray(bar_secs, dtype=np.int64)
    price = np.asarray(bar_prices, dtype=float)
    sigma_fast = lm.sigma_1s_from_bars(sec, price, half_life_secs=halflives["fast"])
    sigma_slow = lm.sigma_1s_from_bars(sec, price, half_life_secs=halflives["slow"])
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(sigma_slow > 0.0, sigma_fast / sigma_slow, np.nan)
    return {"sigma_fast": sigma_fast, "sigma_slow": sigma_slow, "vol_ratio": vol_ratio}


def build_price_grid(
    bars: pd.DataFrame, asset: str, horizons=RETURN_HORIZONS_SECS, halflives=VOL_HALFLIVES
) -> pd.DataFrame:
    """The per-second price feature grid for one asset from ``bars`` (``sec``,
    ``mid``): mid, multi-horizon returns, fast/slow vol, ratio, and the run length.
    Every column is causal (positive look-back / trailing EWMA)."""
    cols = ["asset", "sec", "ts_ms", "mid", *_RET_NAMES, "sigma_fast", "sigma_slow",
            "vol_ratio", "price_run_secs"]
    if bars.empty:
        return pd.DataFrame(columns=cols)
    b = bars.sort_values("sec").reset_index(drop=True)
    sec = b["sec"].to_numpy(dtype=np.int64)
    mid = b["mid"].to_numpy(dtype=float)
    rets = mid_return_grid(sec, mid, horizons)
    vols = ewma_vol_grid(sec, mid, halflives)
    grid = pd.DataFrame({
        "asset": asset, "sec": sec, "ts_ms": sec * 1000, "mid": mid,
        **rets, **vols, "price_run_secs": _run_lengths(sec),
    })
    return grid[cols]


def _flow_rolls(
    sec: np.ndarray, signed: np.ndarray, absq: np.ndarray, count: np.ndarray, lookbacks
) -> dict[str, np.ndarray]:
    """Trailing per-lookback flow imbalance + trade intensity, aligned to ``sec``.

    Reindexes the per-second flow bars to a gap-free 0-filled grid and takes trailing
    ``rolling(L).sum()`` (window ends at the second, so it reads only the past).
    ``flow_imb = signed/|q|`` (NaN when no volume in the window);
    ``trade_intensity = count/L``."""
    out: dict[str, np.ndarray] = {}
    n = len(sec)
    if n == 0:
        for lb in lookbacks:
            out[f"flow_imb_{lb}s"] = np.empty(0, dtype=float)
            out[f"trade_intensity_{lb}s"] = np.empty(0, dtype=float)
        return out
    lo, hi = int(sec[0]), int(sec[-1])
    span = hi - lo + 1
    pos = np.asarray(sec, dtype=np.int64) - lo
    d_signed = np.zeros(span, dtype=float)
    d_abs = np.zeros(span, dtype=float)
    d_count = np.zeros(span, dtype=float)
    d_signed[pos] = np.asarray(signed, dtype=float)
    d_abs[pos] = np.asarray(absq, dtype=float)
    d_count[pos] = np.asarray(count, dtype=float)
    s_signed, s_abs, s_count = pd.Series(d_signed), pd.Series(d_abs), pd.Series(d_count)
    for lb in lookbacks:
        roll_signed = s_signed.rolling(lb, min_periods=1).sum().to_numpy()
        roll_abs = s_abs.rolling(lb, min_periods=1).sum().to_numpy()
        roll_count = s_count.rolling(lb, min_periods=1).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            imb = np.where(roll_abs > 0.0, roll_signed / roll_abs, np.nan)
        # |Σ signed| ≤ Σ|q| exactly, but independent rolling sums can overshoot the
        # [−1, 1] bound by ~1e-12 of float noise — clip to the true mathematical range.
        out[f"flow_imb_{lb}s"] = np.clip(imb[pos], -1.0, 1.0)
        out[f"trade_intensity_{lb}s"] = (roll_count / float(lb))[pos]
    return out


def build_flow_grid(
    flow_bars: pd.DataFrame, asset: str, lookbacks=FLOW_LOOKBACKS_SECS
) -> pd.DataFrame:
    """The per-second flow feature grid for one asset from ``flow_bars`` (``sec``,
    ``signed_qty``, ``abs_qty``, ``trade_count``): imbalance + intensity per
    lookback. Trailing-only — no look-ahead."""
    cols = ["asset", "sec", "ts_ms", *_FLOW_IMB_NAMES, *_INTENSITY_NAMES]
    if flow_bars.empty:
        return pd.DataFrame(columns=cols)
    b = flow_bars.sort_values("sec").reset_index(drop=True)
    sec = b["sec"].to_numpy(dtype=np.int64)
    rolls = _flow_rolls(
        sec, b["signed_qty"].to_numpy(dtype=float), b["abs_qty"].to_numpy(dtype=float),
        b["trade_count"].to_numpy(dtype=float), lookbacks,
    )
    grid = pd.DataFrame({"asset": asset, "sec": sec, "ts_ms": sec * 1000, **rolls})
    return grid[cols]


def _attach_feature_set(
    samples: pd.DataFrame, price_grid: pd.DataFrame, flow_grid: pd.DataFrame,
    max_staleness_ms: int | None,
) -> pd.DataFrame:
    """Bind each sample to its as-of price + flow grid bars with two **backward**
    ``merge_asof`` joins (the matched bar's ``ts_ms ≤ sample_ts_ms``), then derive
    the scalar features. A matched bar older than ``max_staleness_ms`` (a relic
    across a recording gap) is treated as absent: its values are nulled and the
    ``mid_fresh`` / ``flow_covered`` flag set False — strictly more conservative,
    never a forward read."""
    left = samples.sort_values("sample_ts_ms").copy()

    # Price grid → mid, returns, vols, run length + freshness.
    if price_grid.empty:
        for col in _PRICE_VALUE_COLS:
            left[col] = np.nan
        left["price_run_secs"] = np.nan
        left["price_asof_ts_ms"] = np.nan
    else:
        pg = price_grid.rename(columns={"ts_ms": "price_asof_ts_ms"})
        pg = pg[["asset", "price_asof_ts_ms", *_PRICE_VALUE_COLS, "price_run_secs"]]
        pg = pg.sort_values("price_asof_ts_ms")
        left = pd.merge_asof(
            left, pg, left_on="sample_ts_ms", right_on="price_asof_ts_ms",
            by="asset", direction="backward",
        )
    price_asof = left["price_asof_ts_ms"]
    lag = left["sample_ts_ms"] - price_asof
    mid_fresh = price_asof.notna() if max_staleness_ms is None else (
        price_asof.notna() & (lag <= max_staleness_ms)
    )
    mid_fresh = mid_fresh.fillna(False)
    left.loc[~mid_fresh, _PRICE_VALUE_COLS] = np.nan
    left.loc[~mid_fresh, "price_asof_ts_ms"] = np.nan
    left["mid_fresh"] = mid_fresh.to_numpy(dtype=bool)

    # Flow grid → imbalance, intensity + coverage.
    if flow_grid.empty:
        for col in _FLOW_VALUE_COLS:
            left[col] = np.nan
        left["flow_asof_ts_ms"] = np.nan
    else:
        fg = flow_grid.rename(columns={"ts_ms": "flow_asof_ts_ms"})
        fg = fg[["asset", "flow_asof_ts_ms", *_FLOW_VALUE_COLS]].sort_values("flow_asof_ts_ms")
        left = left.sort_values("sample_ts_ms")
        left = pd.merge_asof(
            left, fg, left_on="sample_ts_ms", right_on="flow_asof_ts_ms",
            by="asset", direction="backward",
        )
    flow_asof = left["flow_asof_ts_ms"]
    flag = left["sample_ts_ms"] - flow_asof
    flow_covered = flow_asof.notna() if max_staleness_ms is None else (
        flow_asof.notna() & (flag <= max_staleness_ms)
    )
    flow_covered = flow_covered.fillna(False)
    left.loc[~flow_covered, _FLOW_VALUE_COLS] = np.nan
    left.loc[~flow_covered, "flow_asof_ts_ms"] = np.nan
    left["flow_covered"] = flow_covered.to_numpy(dtype=bool)

    left["price_asof_ts_ms"] = left["price_asof_ts_ms"].astype("Int64")
    left["flow_asof_ts_ms"] = left["flow_asof_ts_ms"].astype("Int64")
    left["price_run_secs"] = left["price_run_secs"].astype("Int64")
    return _derive_scalar_features(left)


def _derive_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``z``, ``seconds_remaining``, ``hour_of_day`` from as-of columns and
    per-sample scalars (``strike`` at open, ``tau_secs`` known at sample time). The
    z-score reproduces the engine formula exactly (``σ_slow`` = engine ``σ_1s``) and is
    **basis-corrected**: the rolling log basis ``basis_ewma`` (from the dataset
    skeleton, NaN ⇒ 0 on same-feed proxy rows) is folded into the moneyness so the
    Binance mid is projected onto the Chainlink strike's feed. No future data is read."""
    df = df.copy()
    basis = df["basis_ewma"].fillna(0.0) if "basis_ewma" in df.columns else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s_k = np.log(df["mid"] / df["strike"])
        sigma_tau = df["sigma_slow"] * np.sqrt(df["tau_secs"])
        z = ((log_s_k + basis) / sigma_tau).clip(-Z_CLAMP, Z_CLAMP)
    df["z"] = z
    df["seconds_remaining"] = df["tau_secs"].astype(float)
    df["hour_of_day"] = ((df["sample_ts_ms"] // 3_600_000) % 24).astype("int16")
    return df


# --- aggTrades flow loaders (mirror dataset._day_bars / _load_month_bars) ----
def _day_flow_bars(path, symbol: str) -> pd.DataFrame:
    """Reduce one aggTrades day-parquet to per-second flow bars. Reads only
    ``(transact_time, quantity, is_buyer_maker)``. ``is_buyer_maker == True`` means
    the aggressor (taker) was the **seller**, so ``signed = −quantity`` there and
    ``+quantity`` for a buy-side aggressor. Returns ``(sec, signed_qty, abs_qty,
    trade_count)``."""
    empty = pd.DataFrame({
        "sec": pd.Series([], dtype="int64"), "signed_qty": pd.Series([], dtype="float64"),
        "abs_qty": pd.Series([], dtype="float64"), "trade_count": pd.Series([], dtype="int64"),
    })
    df = ba.read_aggtrades_columns(path, ["transact_time", "quantity", "is_buyer_maker"])
    if df.empty:
        return empty
    sec = df["transact_time"].to_numpy(dtype="int64") // 1_000_000  # µs → unix seconds
    qty = df["quantity"].to_numpy(dtype=float)
    is_sell = df["is_buyer_maker"].to_numpy(dtype=bool)
    signed = np.where(is_sell, -qty, qty)
    g = pd.DataFrame({"sec": sec, "signed": signed, "absq": np.abs(qty), "one": 1})
    agg = g.groupby("sec", as_index=False).agg(
        signed_qty=("signed", "sum"), abs_qty=("absq", "sum"), trade_count=("one", "sum")
    )
    return agg.sort_values("sec").reset_index(drop=True)


def _load_month_flow_bars(load_paths, symbol: str) -> pd.DataFrame:
    """Per-second flow bars for one month (+ buffer days), deduped by second."""
    frames = [_day_flow_bars(p, symbol) for p in load_paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame({
            "sec": pd.Series([], dtype="int64"), "signed_qty": pd.Series([], dtype="float64"),
            "abs_qty": pd.Series([], dtype="float64"), "trade_count": pd.Series([], dtype="int64"),
        })
    allb = pd.concat(frames, ignore_index=True)
    return allb.groupby("sec", as_index=False).agg(
        signed_qty=("signed_qty", "sum"), abs_qty=("abs_qty", "sum"), trade_count=("trade_count", "sum")
    ).sort_values("sec").reset_index(drop=True)


def _flow_days_for_range(hist_dir, symbol: str, lo_ms: int, hi_ms: int) -> list:
    """aggTrades day-parquets for ``symbol`` whose UTC date overlaps
    ``[lo_ms, hi_ms]`` (± one buffer day), for the journal-window flow grid."""
    lo_day = datetime.fromtimestamp(lo_ms / 1000.0, tz=timezone.utc).date() - timedelta(days=1)
    hi_day = datetime.fromtimestamp(hi_ms / 1000.0, tz=timezone.utc).date() + timedelta(days=1)
    paths = []
    for p in ba.aggtrades_day_paths(hist_dir, symbol):
        d = ba.day_of_path(p)
        if d is not None and lo_day <= d <= hi_day:
            paths.append(p)
    return paths


# --- skeleton + no-look-ahead assertion + finalize --------------------------
def _read_skeleton(ds_path, columns: list[str]) -> pd.DataFrame:
    """Read the light skeleton columns from ``dataset.parquet`` (never all 35),
    renaming the three cross-check columns so they don't collide with the rebuilt
    grid's ``mid`` / ``sigma`` / ``z``."""
    df = pq.read_table(ds_path, columns=columns).to_pandas()
    return df.rename(columns={"mid": "mid_skel", "sigma_1s": "sigma_1s_skel", "z": "z_skel"})


def _assert_no_lookahead(df: pd.DataFrame) -> None:
    """Fail-closed if a batch's features could have read past its sample: the matched
    price/flow bars are at-or-before the sample, and the sample sits inside
    ``[open, close)``."""
    if df.empty:
        return
    for col in ("price_asof_ts_ms", "flow_asof_ts_ms"):
        a = df[col]
        ok = a.notna()
        if ok.any():
            assert (a[ok].astype("int64") <= df.loc[ok.index[ok], "sample_ts_ms"].astype("int64")).all(), \
                f"no-lookahead: {col} is AFTER its sample"
    assert (df["window_open_ms"] <= df["sample_ts_ms"]).all(), "no-lookahead: sample precedes window open"
    assert (df["sample_ts_ms"] < df["window_close_ms"]).all(), "no-lookahead: sample at/after window close"


def _finalize(batch: pd.DataFrame) -> pd.DataFrame:
    """Select the output columns in spec order and coerce to the declared dtypes
    (tolerating NaN-carrying float / nullable columns)."""
    df = batch.copy()
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OUTPUT_COLUMNS]
    for col, dtype in DTYPES.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    return df.reset_index(drop=True)


# --- worker -----------------------------------------------------------------
def feature_set(
    paths: Paths, *, max_feature_staleness_secs: int = DEFAULT_MAX_STALENESS_SECS,
    series: tuple[str, ...] | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    stream: bool = True,
) -> dict:
    """Builds the curated feature matrix on top of ``dataset.parquet``: the journal
    (chainlink) rows in one pass, the historical (binance_proxy) rows streamed
    month-by-month. Writes ``out/feature_set.parquet`` +
    ``out/feature_set/{SCHEMA.md, sanity.json, sanity.html, metadata.json}``; returns
    ``{params, counts, coverage, cross_checks, status, sanity}``. ``since_ms``/
    ``until_ms`` bound the journal mid-grid rebuild (match the range used for
    ``dataset``); ``stream`` uses the bounded journal pass."""
    # Fail loudly on a truncated / still-being-written dataset.parquet rather than
    # silently building features from a partial skeleton (protects every caller).
    require_tables(paths, "dataset")
    paths.ensure_out()
    out_dir = paths.out_dir / "feature_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_staleness_ms = max_feature_staleness_secs * 1000

    params = {
        "max_feature_staleness_secs": int(max_feature_staleness_secs),
        "series_filter": list(series) if series else None,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "stream": bool(stream),
        "return_horizons_secs": list(RETURN_HORIZONS_SECS),
        "flow_lookbacks_secs": list(FLOW_LOOKBACKS_SECS),
        "vol_halflives": dict(VOL_HALFLIVES),
    }

    skel = _read_skeleton(paths.table("dataset"), _SKELETON_READ_COLS)

    schema = _arrow_schema()
    writer = pq.ParquetWriter(paths.table("feature_set"), schema)
    tot: dict[str, int] = defaultdict(int)
    xdiff = {"mid": 0.0, "sigma_slow": 0.0, "z": 0.0}

    def _emit(batch: pd.DataFrame, source: str) -> None:
        if batch.empty:
            return
        _assert_no_lookahead(batch)
        for key, mine, skc in (("mid", "mid", "mid_skel"),
                               ("sigma_slow", "sigma_slow", "sigma_1s_skel"),
                               ("z", "z", "z_skel")):
            if mine in batch and skc in batch:
                a = batch[mine].to_numpy(dtype=float)
                b = batch[skc].to_numpy(dtype=float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.any():
                    xdiff[key] = max(xdiff[key], float(np.max(np.abs(a[m] - b[m]))))
        writer.write_table(pa.Table.from_pandas(_finalize(batch), schema=schema, preserve_index=False))
        tot["samples"] += len(batch)
        tot[f"samples_{source}"] += len(batch)
        tot["mid_fresh"] += int(batch["mid_fresh"].sum())
        tot["flow_covered"] += int(batch["flow_covered"].sum())
        tot[f"flow_covered_{source}"] += int(batch["flow_covered"].sum())

    # --- chainlink batch (journal mid grid + overlapping aggTrades flow) ------
    chain = skel[skel["label_source"] == "chainlink"]
    if not chain.empty:
        ticks_df, *_ = ds._read_journal(paths, series, since_ms=since_ms, until_ms=until_ms, stream=stream)
        has_agg = ba.has_aggtrades(paths.hist_dir)
        for asset in sorted(chain["asset"].dropna().unique()):
            sub = chain[chain["asset"] == asset]
            mid_bars = feat._binance_mid_bars(ticks_df, asset)
            price_grid = build_price_grid(mid_bars[["sec", "mid"]], asset)
            flow_grid = pd.DataFrame(columns=["asset", "sec", "ts_ms", *_FLOW_IMB_NAMES, *_INTENSITY_NAMES])
            sym = ASSET_SYMBOL.get(asset)
            if sym and has_agg:
                fpaths = _flow_days_for_range(
                    paths.hist_dir, sym, int(sub["sample_ts_ms"].min()), int(sub["sample_ts_ms"].max())
                )
                if fpaths:
                    flow_grid = build_flow_grid(_load_month_flow_bars(fpaths, sym), asset)
            _emit(_attach_feature_set(sub.copy(), price_grid, flow_grid, max_staleness_ms), "chainlink")

    # --- historical (binance_proxy) batches, streamed per (symbol, month) -----
    if ba.has_aggtrades(paths.hist_dir):
        symbols = [s for s in ba.symbols_present(paths.hist_dir) if ds._symbol_in_scope(s, series)]
        for sym in symbols:
            asset = ds.SYMBOL_ASSET[sym]
            ser = ds.SYMBOL_SERIES[sym]
            proxy_sym = skel[(skel["label_source"] == "binance_proxy")
                             & (skel["asset"] == asset) & (skel["series"] == ser)]
            if proxy_sym.empty:
                continue
            for _yr, _mo, load_paths, mdays in ds._month_batches(paths.hist_dir, sym):
                opens = set(ds._present_opens(mdays))
                sub = proxy_sym[proxy_sym["window_open_ms"].isin(opens)]
                if sub.empty:
                    continue
                bars = ds._load_month_bars(load_paths, sym)
                if bars.empty:
                    continue
                price_grid = build_price_grid(bars.rename(columns={"price": "mid"})[["sec", "mid"]], asset)
                flow_grid = build_flow_grid(_load_month_flow_bars(load_paths, sym), asset)
                _emit(_attach_feature_set(sub.copy(), price_grid, flow_grid, max_staleness_ms), "binance_proxy")

    if tot["samples"] == 0:
        writer.write_table(schema.empty_table())
    writer.close()

    sanity = sanity_report(paths.table("feature_set"), FEATURE_SPEC)

    n = int(tot["samples"])
    counts = {
        "samples": n,
        "samples_chainlink": int(tot["samples_chainlink"]),
        "samples_binance_proxy": int(tot["samples_binance_proxy"]),
        "features": len(FEATURE_SPEC),
    }
    coverage = {
        "mid_fresh_frac": (tot["mid_fresh"] / n) if n else float("nan"),
        "flow_covered_frac": (tot["flow_covered"] / n) if n else float("nan"),
        "flow_covered_chainlink": int(tot["flow_covered_chainlink"]),
        "flow_covered_proxy": int(tot["flow_covered_binance_proxy"]),
    }
    cross_checks = {
        "max_abs_mid_diff": xdiff["mid"],
        "max_abs_sigma_slow_diff": xdiff["sigma_slow"],
        "max_abs_z_diff": xdiff["z"],
    }
    meta = {
        "params": params, "counts": counts, "coverage": coverage,
        "cross_checks": cross_checks, "sanity_status": sanity["overall_status"],
    }
    _write_outputs(out_dir, sanity, meta)
    return {"params": params, "counts": counts, "coverage": coverage,
            "cross_checks": cross_checks, "status": sanity["overall_status"], "sanity": sanity,
            "peak_rss_mb": procmem.peak_rss_mb()}


# --- sanity report ----------------------------------------------------------
_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_ORDER[a] >= _STATUS_ORDER[b] else b


def _expected_nan_mask(spec: FeatureSpec, masks: pd.DataFrame) -> np.ndarray:
    """The rows where this feature is *legitimately* NaN (warmup / no aggTrades
    coverage / stale as-of bar). Excess NaN is anything outside this."""
    mid_fresh = masks["mid_fresh"].to_numpy(dtype=bool)
    flow_covered = masks["flow_covered"].to_numpy(dtype=bool)
    run = masks["price_run_secs"].to_numpy(dtype=float)
    run0 = np.where(np.isfinite(run), run, 0.0)
    strike = masks["strike"].to_numpy(dtype=float)
    fam = spec.family
    if fam == "ret":
        return (~mid_fresh) | (run0 <= float(spec.param))
    # sigma_1s_from_bars needs `warmup` folds (= run_length − 1: the first bar of a
    # run yields no return), so σ is NaN while run ≤ warmup — not run < warmup.
    if fam in ("sigma", "vol_ratio"):
        return (~mid_fresh) | (run0 <= WARMUP_RETURNS)
    if fam == "zscore":
        return (~mid_fresh) | (run0 <= WARMUP_RETURNS) | ~(strike > 0.0)
    if fam in ("flow_imb", "trade_intensity"):
        return ~flow_covered
    return np.zeros(len(mid_fresh), dtype=bool)  # time features: never expected-NaN


def _column_sanity(spec: FeatureSpec, values: np.ndarray, masks: pd.DataFrame) -> dict:
    """Distribution + the three checks (no NaN / no absurd outliers / no constant)
    for one feature column, with a PASS/WARN/FAIL per check and overall."""
    n = int(len(values))
    finite_mask = np.isfinite(values)
    n_finite = int(finite_mask.sum())
    n_nan = n - n_finite
    finite = values[finite_mask]

    expected = _expected_nan_mask(spec, masks)
    is_nan = ~finite_mask
    excess = int((is_nan & ~expected).sum())
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

    # (1) no NaNs — judged on the *excess* (unexpected) NaN only.
    if spec.coverage == "partial" and n_finite == 0:
        nan_flag, nan_note = "WARN", "no aggTrades coverage"
    elif spec.coverage == "full" and n_finite == 0:
        nan_flag, nan_note = "FAIL", "full-coverage feature is entirely NaN"
    elif excess_frac == 0.0:
        nan_flag, nan_note = "PASS", ""
    elif excess_frac <= NAN_WARN_FRAC:
        nan_flag, nan_note = "WARN", f"{excess_frac:.2%} unexpected NaN"
    else:
        nan_flag, nan_note = "FAIL", f"{excess_frac:.2%} unexpected NaN"

    # (2) no absurd outliers — hard-bounded features fail on any violation.
    if n_finite == 0 or out_count == 0:
        outlier_flag = "PASS"
    elif spec.hard_bounds:
        outlier_flag = "FAIL"
    elif out_frac <= OUTLIER_WARN_FRAC:
        outlier_flag = "WARN"
    else:
        outlier_flag = "FAIL"

    # (3) no constant columns — a constant full-coverage feature is a bug (FAIL);
    # a constant partial-coverage feature may just be a limited-coverage artifact (WARN).
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


def sanity_report(parquet_path, specs: list[FeatureSpec]) -> dict:
    """Distribution sanity of every feature column in the finished parquet, read a
    column at a time (bounded memory). Returns per-column stats + an overall
    PASS/WARN/FAIL status."""
    masks = pq.read_table(
        parquet_path, columns=["price_run_secs", "mid_fresh", "flow_covered", "strike"]
    ).to_pandas()
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


# --- outputs ----------------------------------------------------------------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f"<p class='muted'>{alt}: not available.</p>"
    return f"<img alt='{alt}' src='data:image/png;base64,{b64}'/>"


def _hist_png(columns: list[dict]) -> str | None:
    """Small-multiples histograms of the feature columns that have data."""
    drawable = [c for c in columns if c["hist"]["counts"]]
    if not drawable:
        return None
    ncol = 3
    nrow = (len(drawable) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, drawable):
        edges = np.asarray(c["hist"]["edges"], dtype=float)
        counts = np.asarray(c["hist"]["counts"], dtype=float)
        color = {"PASS": "#1f77b4", "WARN": "#d98b00", "FAIL": "#d62728"}[c["status"]]
        ax.stairs(counts, edges, fill=True, color=color)
        ax.set_title(c["name"], fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(drawable):]:
        ax.axis("off")
    fig.tight_layout()
    return _png(fig)


_SANITY_TABLE_COLS = [
    ("name", "feature"), ("family", "family"), ("coverage", "coverage"),
    ("n_finite", "n finite"), ("nan_fraction", "NaN"), ("excess_nan_fraction", "excess NaN"),
    ("min", "min"), ("p50", "median"), ("max", "max"), ("std", "std"),
    ("unique_count", "unique"), ("outlier_fraction", "outliers"), ("status", "status"),
    ("note", "note"),
]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.4g}"
    return str(v)


def _table_html(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>no rows.</p>"
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = []
    for r in rows:
        status = r.get("status", "PASS")
        cls = f" class='{status.lower()}'" if status in ("WARN", "FAIL") else ""
        cells = "".join(f"<td>{_fmt(r.get(key))}</td>" for key, _ in columns)
        body.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _build_html(sanity: dict, meta: dict) -> str:
    cols = sanity["columns"]
    tally = sanity["status_tally"]
    hist_img = _img_tag(_hist_png(cols), "feature histograms")
    cc = meta["cross_checks"]
    cov = meta["coverage"]
    banner = {"PASS": "#1f7a1f", "WARN": "#a06a00", "FAIL": "#b00020"}[sanity["overall_status"]]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>feature_set — sanity</title>
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
<h1>feature_set — distribution sanity</h1>
<p class="verdict"><b>{sanity['overall_status']}</b> — {tally['PASS']} PASS, {tally['WARN']} WARN,
{tally['FAIL']} FAIL over {sanity['n_features']} features and {sanity['n_rows']:,} rows.
Each feature is checked for <b>unexpected NaN</b>, <b>absurd outliers</b>, and being a
<b>constant column</b>. Rows shaded
<span style="background:#fff5e6;color:#7a5200;padding:0 4px">amber</span> WARN /
<span style="background:#fdecef;color:#8a0018;padding:0 4px">pink</span> FAIL.</p>

<p class="muted">Cross-checks vs the dataset skeleton (should be ≈ 0):
mid Δ = {_fmt(cc['max_abs_mid_diff'])}, σ_slow Δ = {_fmt(cc['max_abs_sigma_slow_diff'])},
z Δ = {_fmt(cc['max_abs_z_diff'])}. Coverage: mid-fresh {_fmt(cov['mid_fresh_frac'])},
flow-covered {_fmt(cov['flow_covered_frac'])}.</p>

<h2>Per-feature sanity</h2>
{_table_html(cols, _SANITY_TABLE_COLS)}

<h2>Distributions</h2>
{hist_img}

<p class="muted">Generated by <code>python -m model_lab.feature_set</code>.
Full column docs in <code>SCHEMA.md</code>.</p>
</body></html>
"""


_PASSTHROUGH_DOC = {
    "series": "Series key, e.g. `BTC-5m` (from the dataset).",
    "asset": "Underlying asset, `btc`/`eth` (from the dataset).",
    "window_open_ms": "Window open time (unix ms).",
    "window_close_ms": "Window close / resolution time (unix ms).",
    "sample_ts_ms": "The feature timestamp — this grid point (unix ms).",
    "sample_idx": "Grid index `k` within the window (0-based).",
    "elapsed_secs": "Seconds since window open (≥ 0).",
    "tau_secs": "Seconds remaining (> 0); known at sample time.",
    "strike": "Price-to-beat captured at open (from the dataset).",
    "label_source": "`chainlink` (journal) or `binance_proxy` (historical).",
    "split": "Chronological split `train`/`val` (from the dataset).",
    "in_live_coverage": "Whether a Polymarket market-mid benchmark exists for this window.",
    "fwd_up_30s": "LABEL: 1 iff the mid rose over the next 30 s (from the dataset).",
    "outcome_up": "LABEL: 1 iff the window resolved Up (from the dataset).",
}
_DIAG_DOC = {
    "mid": "Rebuilt as-of Binance mid (USD) — context for `z` / returns.",
    "price_asof_ts_ms": "`ts_ms` of the price bar the features came from (≤ sample); NA if none/stale.",
    "flow_asof_ts_ms": "`ts_ms` of the flow bar the features came from (≤ sample); NA if none/stale.",
    "price_run_secs": "Length of the contiguous observed-second run at the price bar (for warmup classification).",
    "mid_fresh": "Whether a price bar within the staleness window backs this sample.",
    "flow_covered": "Whether an aggTrades flow bar within the staleness window backs this sample.",
}


def _write_schema(out_dir) -> None:
    lines = [
        "# `feature_set.parquet` — schema",
        "",
        "One row per (window, feature-timestamp) — the same rows as `dataset.parquet`, "
        "augmented with the engineered feature set. Built by `python -m model_lab.feature_set`.",
        "",
        "| column | dtype | category | description |",
        "|---|---|---|---|",
    ]
    for name in PASSTHROUGH_COLS:
        cat = "**label**" if name in ("fwd_up_30s", "outcome_up") else "identifier / metadata"
        lines.append(f"| `{name}` | `{DTYPES[name]}` | {cat} | {_PASSTHROUGH_DOC[name]} |")
    for name in CONTEXT_COLS:
        lines.append(f"| `{name}` | `{DTYPES[name]}` | context | {_DIAG_DOC[name]} |")
    for spec in FEATURE_SPEC:
        lines.append(f"| `{spec.name}` | `{spec.dtype}` | **feature** (as-of ≤ sample_ts) | {spec.doc} |")
    for name in DIAG_COLS:
        lines.append(f"| `{name}` | `{DTYPES[name]}` | diagnostic | {_DIAG_DOC[name]} |")
    lines += [
        "",
        "## Feature families (one config)",
        "",
        "The feature list is generated from the knob lists in `model_lab/feature_set.py` —",
        f"`RETURN_HORIZONS_SECS = {RETURN_HORIZONS_SECS}`, "
        f"`FLOW_LOOKBACKS_SECS = {FLOW_LOOKBACKS_SECS}`, "
        f"`VOL_HALFLIVES = {VOL_HALFLIVES}` — so adding or removing a feature is a one-line change.",
        "",
        "## No-look-ahead contract",
        "",
        "No value in a **feature** column derives from data after that row's `sample_ts_ms`:",
        "",
        "- returns use a positive look-back on a forward-filled per-second mid grid and are "
        "nulled across recording gaps; the EWMA vols are the causal engine σ_1s recurrence; "
        "flow/intensity are trailing rolling sums; all are attached with a **backward** "
        "`merge_asof` (`*_asof_ts_ms ≤ sample_ts_ms`).",
        "- `z` reproduces the engine formula `log(mid/strike)/(σ_slow·√τ)` exactly (σ_slow is the "
        "engine σ_1s), reconciled against the dataset's `z`/`mid`/`sigma_1s` as a drift guard.",
        "- Enforced by a runtime assertion (`feature_set._assert_no_lookahead`), the AST scan over "
        "the grid kernels (`tests/test_dataset.py::test_no_lookahead_ast_scan`), and a "
        "differential/corruption test (`tests/test_feature_set.py`).",
        "",
        "## Coverage & NaN",
        "",
        "- **Flow / intensity** (`flow_imb_*`, `trade_intensity_*`) come only from the Binance "
        "aggTrades archive, so they are NaN where the day is not covered — `flow_covered` marks "
        "where they are real. **Returns / vols / z** are NaN during warmup (the first seconds of a "
        "run, `price_run_secs`) and where the as-of bar is stale (`mid_fresh`).",
        "- Those NaN are **expected** and left in the parquet (a learner must tell real from "
        "missing); the sanity report fails only on *excess* (unexpected) NaN. See "
        "`sanity.json` / `sanity.html` for the per-feature distribution report.",
        "",
    ]
    (out_dir / "SCHEMA.md").write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(out_dir, sanity: dict, meta: dict) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    (out_dir / "sanity.html").write_text(_build_html(sanity, meta), encoding="utf-8")
    _write_schema(out_dir)


# --- entry ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "feature_set")
    parser.add_argument("--max-feature-staleness-secs", type=int, default=DEFAULT_MAX_STALENESS_SECS,
                        help=f"null as-of features older than this (recording-gap relics) (default {DEFAULT_MAX_STALENESS_SECS})")
    parser.add_argument("--series", default=None,
                        help="comma-separated series filter (default: everything in the dataset)")
    parser.add_argument("--no-stream", action="store_true",
                        help="read every raw tick into memory (the pre-streaming path) — for debugging / equivalence checks")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    since_ms, until_ms = resolve_bounds(args)
    try:
        assert_parquet_ready(paths.table("dataset"), label="dataset.parquet")
    except ParquetNotReady as exc:
        print(f"[feature_set] {exc}")
        return 1
    print(f"[feature_set] dataset={paths.table('dataset')}")
    print(f"[feature_set] hist   ={paths.hist_dir}")
    print(f"[feature_set] out    ={paths.table('feature_set')}")

    result = feature_set(
        paths,
        max_feature_staleness_secs=args.max_feature_staleness_secs,
        series=ds._parse_series(args.series),
        since_ms=since_ms,
        until_ms=until_ms,
        stream=not args.no_stream,
    )
    c = result["counts"]
    cov = result["coverage"]
    cc = result["cross_checks"]
    print(f"[feature_set] {c['features']} features over {c['samples']:,} samples "
          f"({c['samples_chainlink']:,} chainlink / {c['samples_binance_proxy']:,} binance_proxy)")
    print(f"[feature_set] coverage: mid-fresh={cov['mid_fresh_frac']:.1%} "
          f"flow-covered={cov['flow_covered_frac']:.1%} "
          f"(proxy flow rows={cov['flow_covered_proxy']:,})")
    print(f"[feature_set] cross-check vs dataset (≈0): mid Δ={cc['max_abs_mid_diff']:.3g} "
          f"σ_slow Δ={cc['max_abs_sigma_slow_diff']:.3g} z Δ={cc['max_abs_z_diff']:.3g}")
    tally = result["sanity"]["status_tally"]
    flagged = [c2["name"] for c2 in result["sanity"]["columns"] if c2["status"] != "PASS"]
    print(f"[feature_set] sanity: {result['status']} "
          f"({tally['PASS']} PASS / {tally['WARN']} WARN / {tally['FAIL']} FAIL)"
          + (f" — flagged: {', '.join(flagged)}" if flagged else ""))
    procmem.report_peak_rss("feature_set", result.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[feature_set] wrote {paths.table('feature_set').name} + "
          "feature_set/SCHEMA.md, sanity.json, sanity.html, metadata.json")
    if c["samples"] == 0:
        print("[feature_set] WARNING: no samples produced — is dataset.parquet empty?")
        return 1
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
