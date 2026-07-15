"""End-to-end verification over the synthetic fixture.

Generates a fresh synthetic journal + depth into a temp directory, runs every
stage against it, and asserts the outputs — the fast green check that the whole
pipeline works without needing a live capture or the real journal.

Run:  ``python -m model_lab.verify``  (prints ``VERIFY: PASS`` / ``FAIL``).
"""

from __future__ import annotations

import hashlib
import io
import math
import sys
import tempfile
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from . import feature_set as fset
from . import hist, hist_integrity
from .calibration_audit import calibration_audit
from .config import Paths
from .dataset import dataset
from .evaluate import evaluate
from .feature_set import feature_set
from .features import features
from .fixtures import make_aggtrades_fixture, make_fixture, make_predictions_fixture, make_short_oos_fixture
from .ingest import ingest
from .io.binance_archive import ArchiveNotFound, checksum_url, daterange, day_url
from .labels import labels
from .learn import learn
from .money_judge import money_judge
from .report import report
from .research import research
from . import short_horizon as shmod
from .validate import validate


def _offline_archive(symbols: list[str], days: list[date]) -> dict[str, bytes]:
    """A ``url -> bytes`` map of synthetic aggTrades zips + ``.CHECKSUM`` sidecars,
    with strictly-increasing ids across days (clean cross-day continuity)."""
    covered: dict[str, bytes] = {}
    for sym in symbols:
        base_id = 1000
        for i, d in enumerate(days):
            ts0 = 1_751_000_000_000_000 + i * 86_400_000_000  # µs, one day apart
            lines = [
                f"{base_id + j},60000.{j},0.{j + 1},{(base_id + j) * 2},{(base_id + j) * 2 + 1},"
                f"{ts0 + j * 1000},{'true' if j % 2 else 'false'}"
                for j in range(5)
            ]
            csv = ("\n".join(lines) + "\n").encode()
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as zf:
                zf.writestr(f"{sym}-aggTrades-{d.isoformat()}.csv", csv)
            zb = zbuf.getvalue()
            covered[day_url(sym, d)] = zb
            covered[checksum_url(sym, d)] = (
                f"{hashlib.sha256(zb).hexdigest()}  {sym}-aggTrades-{d.isoformat()}.zip".encode()
            )
            base_id += 100
    return covered


def _synthetic_fetch(covered: dict[str, bytes]):
    def fetch(url: str) -> bytes:
        if url in covered:
            return covered[url]
        raise ArchiveNotFound(url)

    return fetch


def check_hist_offline(paths: Paths, check) -> None:
    """Exercise the aggTrades download + integrity path with **no network**: feed a
    synthetic in-memory archive through ``hist.download`` + ``hist_integrity`` in a
    throwaway store under ``out/`` (so it never touches the real aggTrades dir)."""
    symbols = ["BTCUSDT", "ETHUSDT"]
    days = daterange(date(2026, 1, 1), date(2026, 1, 3))  # 3 available days
    fetch = _synthetic_fetch(_offline_archive(symbols, days))
    hp = replace(paths, hist_dir=paths.out_dir / "_hist_selfcheck")

    # Request one extra day the fake serves as 404 → recorded missing (self-healing).
    summary = hist.download(
        hp, symbols=symbols, start=date(2026, 1, 1), end=date(2026, 1, 4), jobs=1, fetch=fetch
    )
    check(summary["downloaded"] == 6, f"hist: expected 6 downloads, got {summary['downloaded']}")
    check(summary["missing"] == 2, f"hist: expected 2 missing days, got {summary['missing']}")
    check(
        summary["checksum_failed"] == 0,
        f"hist: unexpected checksum failures ({summary['checksum_failed']})",
    )

    # Re-run the available range: incremental → nothing re-downloaded.
    again = hist.download(
        hp, symbols=symbols, start=date(2026, 1, 1), end=date(2026, 1, 3), jobs=1, fetch=fetch
    )
    check(
        again["downloaded"] == 0 and again["skipped"] == 6,
        f"hist: re-run not incremental ({again['downloaded']} dl / {again['skipped']} skip)",
    )

    integ = hist_integrity.integrity(hp)
    check(integ["days_checked"] == 6, f"hist: integrity checked {integ['days_checked']} days (want 6)")
    check(
        integ["days_failed"] == 0,
        f"hist: integrity failures {[d for d in integ['days'] if not d['ok']]}",
    )


# Tolerances for the offline-vs-live basis guard (see check_basis_offline_vs_live).
BASIS_GUARD_MAX_BRIER_GAP = 0.03
BASIS_GUARD_MIN_CORR = 0.97


def basis_calibration_by_bucket(paths: Paths, p_up_col: str = "p_up_model") -> list[dict]:
    """Per time-remaining bucket, score a reconstructed Φ(feature z) column from
    ``dataset.parquet`` (default the basis-corrected ``p_up_model``) against the
    **journaled engine ``p_up``** on chainlink windows. Returns
    ``[{bucket, n, brier_model, brier_journaled, brier_gap, corr}]`` — the offline
    reconstruction vs live truth, the divergence a forensic session once had to
    find by hand."""
    import numpy as np
    import pandas as pd

    from . import eval_harness as eh
    from .lib import math as lm

    df = pd.read_parquet(
        paths.table("dataset"),
        columns=["series", "window_open_ms", "sample_ts_ms", "label_source", p_up_col],
    )
    ch = df[df["label_source"] == "chainlink"].dropna(subset=[p_up_col])
    if ch.empty:
        return []
    grid = ch[["series", "window_open_ms", "sample_ts_ms", p_up_col]].rename(columns={p_up_col: "p_up"})
    frame, _ = eh.build_external_frame_from(eh.load_benchmarks(paths), grid)
    frame = frame.dropna(subset=["formula", "model"])
    if frame.empty:
        return []
    tb = eh.tau_bucket(frame["tau"].to_numpy(dtype=float), frame["dur"].to_numpy(dtype=float))
    rows: list[dict] = []
    for bkt in eh.TAU_ORDER:
        g = frame[tb == bkt]
        if len(g) < 5:
            continue
        m = g["model"].to_numpy(dtype=float)
        f = g["formula"].to_numpy(dtype=float)
        o = g["outcome_up"].to_numpy(dtype=float)
        corr = float(np.corrcoef(m, f)[0, 1]) if (m.std() > 0 and f.std() > 0) else 1.0
        rows.append({
            "bucket": bkt, "n": int(len(g)),
            "brier_model": lm.brier_score(m, o), "brier_journaled": lm.brier_score(f, o),
            "brier_gap": abs(lm.brier_score(m, o) - lm.brier_score(f, o)), "corr": corr,
        })
    return rows


def check_basis_offline_vs_live(paths: Paths, check) -> None:
    """Guard the feature layer's ``z`` against the live engine: on chainlink windows
    the basis-corrected Φ(feature z) must track the journaled engine ``p_up`` per
    time-remaining bucket. Fails loudly if the Brier gap or correlation drifts beyond
    tolerance — so a reverted/broken basis correction (offline-vs-live divergence)
    can never again pass silently."""
    rows = basis_calibration_by_bucket(paths)
    check(len(rows) > 0, "basis guard: no chainlink rows overlapped the journaled p_up")
    for r in rows:
        check(r["brier_gap"] <= BASIS_GUARD_MAX_BRIER_GAP,
              f"basis guard[{r['bucket']}]: Φ(feature z) Brier diverges from journaled "
              f"p_up by {r['brier_gap']:.4f} (> {BASIS_GUARD_MAX_BRIER_GAP}); basis correction broken?")
        check(r["corr"] >= BASIS_GUARD_MIN_CORR,
              f"basis guard[{r['bucket']}]: Φ(feature z) correlation with journaled "
              f"p_up {r['corr']:.4f} (< {BASIS_GUARD_MIN_CORR})")


def run_checks(paths: Paths) -> list[str]:
    """Runs all stages over ``paths`` and returns a list of failure messages."""
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    counts = ingest(paths)
    check(counts["ticks"] > 0, "ingest: no ticks")
    check(counts["model"] > 0, "ingest: no model snapshots")
    check(counts["settlements"] > 0, "ingest: no settlements")
    check(counts["depth"] > 0, "ingest: no depth frames")

    n_feat = features(paths)
    check(n_feat > 0, "features: empty feature grid")

    n_fwd, n_win = labels(paths)
    check(n_fwd > 0, "labels: no forward labels")
    check(n_win > 0, "labels: no window outcomes")

    ds = dataset(paths)
    dc = ds["counts"]
    check(dc["samples"] > 0, "dataset: no samples produced")
    check(dc["windows_resolved"] > 0, "dataset: no resolved windows")
    check(dc["samples_in_coverage"] > 0, "dataset: no in-coverage samples")
    check(dc["samples_in_coverage"] <= dc["samples"], "dataset: coverage exceeds sample count")
    check(dc["samples_train"] > 0 and dc["samples_val"] > 0, "dataset: split is not a train/val mix")
    check(paths.table("dataset").exists(), "dataset: dataset.parquet not written")
    for fname in ("SCHEMA.md", "metadata.json"):
        check((paths.out_dir / "dataset" / fname).exists(), f"dataset: {fname} not written")
    # Historical (Binance aggTrades) proxy extension: both sources present.
    check(dc["windows_chainlink"] > 0, "dataset: no journal (chainlink) windows")
    check(dc["windows_binance_proxy"] > 0, "dataset: no historical (binance_proxy) windows")
    check(dc["samples_binance_proxy"] > 0, "dataset: no historical proxy samples")
    check(len(ds["history"]["per_symbol_month"]) > 0, "dataset: no per-symbol-month history rows")
    check(ds["history"]["knife_edge"]["proxy_windows_scored"] > 0, "dataset: knife-edge not scored")
    check(len(ds["series"]) >= 2, f"dataset: expected BTC+ETH series, got {ds['series']}")
    # The journal (chainlink) subset must be byte-identical with vs without history.
    # Read with the schema-driven pyarrow backend so nullable-int columns don't flip
    # dtype between the two frames (a proxy row's NA would otherwise make a whole
    # column float in one frame and int in the other, a spurious diff).
    full = pd.read_parquet(paths.table("dataset"), dtype_backend="pyarrow")
    nohist_paths = replace(paths, out_dir=paths.out_dir / "_dataset_nohist")
    ds_nohist = dataset(nohist_paths, include_history=False)
    nohist = pd.read_parquet(nohist_paths.table("dataset"), dtype_backend="pyarrow")
    check(ds_nohist["counts"]["windows_binance_proxy"] == 0, "dataset: --no-history still produced proxy windows")
    j_full = full[full["label_source"] == "chainlink"].reset_index(drop=True)
    check(j_full.equals(nohist.reset_index(drop=True)),
          "dataset: journal (chainlink) subset changed when history was added")

    # --- short-horizon dataset (10s/15s labels + per-sample coverage) ---------
    sh = shmod.short_horizon(paths)
    shc = sh["counts"]
    check(shc["samples"] > 0, "short_horizon: no samples produced")
    check(shc["samples_train"] > 0 and shc["samples_val"] > 0, "short_horizon: split is not a train/val mix")
    check(shc["windows_chainlink"] > 0, "short_horizon: no journal (chainlink) windows")
    check(shc["windows_binance_proxy"] > 0, "short_horizon: no historical (binance_proxy) windows")
    check(shc["samples_binance_proxy"] > 0, "short_horizon: no historical proxy samples")
    check(paths.table("short_horizon").exists(), "short_horizon: short_horizon.parquet not written")
    for fname in ("SCHEMA.md", "metadata.json"):
        check((paths.out_dir / "short_horizon" / fname).exists(), f"short_horizon: {fname} not written")
    check(len(sh["per_day"]) > 0, "short_horizon: per-day report empty")
    check(0.0 <= sh["coverage"]["depth_pct"] <= 1.0 and 0.0 <= sh["coverage"]["book_pct"] <= 1.0,
          "short_horizon: coverage percentage out of [0,1]")
    shdf = pd.read_parquet(paths.table("short_horizon"), engine="pyarrow")
    check(list(shdf.columns) == shmod.COLUMNS, "short_horizon: column set/order != COLUMN_SPEC")
    for lc in ("fwd_up_10s", "fwd_up_15s"):
        check(shdf[lc].dropna().isin([0, 1]).all(), f"short_horizon: {lc} not in {{0,1}}")
    for cc in ("depth_covered", "book_covered", "depth_feat_covered", "pm_feat_covered"):
        check(cc in shdf.columns and shdf[cc].dtype == bool, f"short_horizon: {cc} not a bool column")
    # A real mix: our depth capture covers journal samples; proxy samples never are.
    check(bool(shdf["depth_covered"].any()) and bool((~shdf["depth_covered"]).any()),
          "short_horizon: depth_covered is not a mix of covered/uncovered")
    check(bool(shdf["book_covered"].any()), "short_horizon: no book-covered samples")
    # The *_feat_covered flags must mean "features actually present" — the filter signal.
    check(bool((shdf["pm_feat_covered"] == shdf["pm_mid"].notna()).all()),
          "short_horizon: pm_feat_covered != pm-feature presence")
    check(bool((shdf["depth_feat_covered"] == shdf["depth_imb_1"].notna()).all()),
          "short_horizon: depth_feat_covered != depth-feature presence")
    check(bool(shdf["pm_feat_covered"].any()), "short_horizon: no pm-feature-covered samples")
    # The chainlink subset must be byte-identical with vs without history (pyarrow
    # backend so nullable columns don't flip dtype between the two frames).
    sh_nohist_paths = replace(paths, out_dir=paths.out_dir / "_short_horizon_nohist")
    sh_nohist = shmod.short_horizon(sh_nohist_paths, include_history=False)
    check(sh_nohist["counts"]["windows_binance_proxy"] == 0,
          "short_horizon: --no-history still produced proxy windows")
    sh_full = pd.read_parquet(paths.table("short_horizon"), dtype_backend="pyarrow")
    sh_no = pd.read_parquet(sh_nohist_paths.table("short_horizon"), dtype_backend="pyarrow")
    sh_jf = sh_full[sh_full["label_source"] == "chainlink"].reset_index(drop=True)
    check(sh_jf.equals(sh_no.reset_index(drop=True)),
          "short_horizon: journal (chainlink) subset changed when history was added")
    # New microstructure features: populated on chainlink (our recordings), NaN on proxy.
    for mcol in ("depth_imb_5", "microprice_gap", "pm_mid", "pm_staleness_3s"):
        check(mcol in shdf.columns, f"short_horizon: microstructure column {mcol} missing")
    sh_ch = shdf[shdf["label_source"] == "chainlink"]
    sh_px = shdf[shdf["label_source"] == "binance_proxy"]
    check(bool(sh_ch["depth_imb_5"].notna().any()) and bool(sh_ch["pm_mid"].notna().any()),
          "short_horizon: depth/PM features never populated on chainlink")
    check(bool(sh_px["depth_imb_5"].isna().all()) and bool(sh_px["pm_mid"].isna().all()),
          "short_horizon: depth/PM features not NaN on binance_proxy")
    check(bool(shdf["pm_mid"].dropna().between(0.0, 1.0).all())
          and bool(shdf["depth_imb_5"].dropna().between(-1.0, 1.0).all()),
          "short_horizon: bounded microstructure feature out of range")
    check((paths.out_dir / "short_horizon" / "sanity.json").exists(),
          "short_horizon: sanity.json not written")
    check(sh.get("sanity_status") in ("PASS", "WARN"),
          f"short_horizon: distribution sanity is {sh.get('sanity_status')}")

    # The engineered feature matrix built on top of the dataset.
    fs = feature_set(paths)
    fc = fs["counts"]
    check(fs["status"] != "FAIL", f"feature_set: sanity FAIL ({fs['sanity']['status_tally']})")
    check(fc["samples"] == dc["samples"], f"feature_set: {fc['samples']} samples != dataset {dc['samples']}")
    check(fc["features"] == len(fset.FEATURE_SPEC), "feature_set: feature count mismatch")
    check(fs["coverage"]["flow_covered_proxy"] > 0, "feature_set: no proxy flow coverage")
    check(fs["cross_checks"]["max_abs_z_diff"] < 1e-6, "feature_set: z drifted from the dataset skeleton")
    check(paths.table("feature_set").exists(), "feature_set: feature_set.parquet not written")
    for fname in ("SCHEMA.md", "sanity.json", "sanity.html", "metadata.json"):
        check((paths.out_dir / "feature_set" / fname).exists(), f"feature_set: {fname} not written")

    # Offline-vs-live guard: the basis-corrected feature z must reproduce the live
    # engine p_up on chainlink windows (never let this diverge silently again).
    check_basis_offline_vs_live(paths, check)

    # The first challenger: walk-forward logistic regression, scored through the harness.
    # The fixture spans only a few UTC days, so learn exercises its single-split fallback.
    # (Statistical claims — real beats chance, shuffled collapses — live in test_learn.py;
    # here we assert the pipeline wiring + artifacts + the harness contract.)
    lrn = learn(paths, days=0, run_harness=True, run_shuffle=True)
    for t in ("fwd30", "outcome"):
        blk = lrn["targets"].get(t)
        check(blk is not None and blk["n_oos"] > 0, f"learn: no OOS predictions for {t}")
        check((paths.out_dir / "learn" / f"model_{t}.json").exists(), f"learn: model_{t}.json not written")
        pgrid = paths.out_dir / "learn" / f"predictions_{t}.parquet"
        check(pgrid.exists(), f"learn: predictions_{t}.parquet not written")
        if pgrid.exists():
            cols = list(pd.read_parquet(pgrid).columns)
            check(cols == ["series", "window_open_ms", "sample_ts_ms", "p_up"],
                  f"learn: predictions_{t} grid columns {cols} != harness contract")
        check((blk or {}).get("harness", {}).get("n_scored", 0) > 0, f"learn: {t} not scored through harness")
        sc = lrn["shuffled_control"].get(t, {})
        check("collapsed" in sc, f"learn: shuffled-control collapse verdict missing for {t}")
    for fname in ("metrics.json", "scores.csv", "verdict_table.csv", "reliability.csv", "report.html"):
        check((paths.out_dir / "learn" / "harness_outcome" / fname).exists(),
              f"learn: harness_outcome/{fname} not written")
    check((paths.out_dir / "learn" / "folds.csv").exists(), "learn: folds.csv not written")
    check((paths.out_dir / "learn" / "metrics.json").exists(), "learn: metrics.json not written")

    v = validate(paths)
    phi = v["phi_identity"]
    check(phi.get("n", 0) > 0 and phi["median_abs"] < 1e-6, f"validate: Φ(z) identity off ({phi})")
    sig = v["sigma_reproduction"]
    check(sig.get("n", 0) > 0 and sig["correlation"] > 0.5, f"validate: σ reproduction weak ({sig})")
    check(v["calibration"]["n_windows"] > 0, "validate: no calibratable windows")

    r = research(paths)
    check(r["has_depth"], "research: depth not detected")
    pooled = r.get("pooled", {})
    check(pooled.get("ic_imbalance", 0.0) > 0.0, f"research: imbalance IC not positive ({pooled.get('ic_imbalance')})")
    check(pooled.get("auc_imbalance", 0.0) > 0.5, f"research: imbalance AUC ≤ 0.5 ({pooled.get('auc_imbalance')})")

    ca = calibration_audit(paths)
    counts = ca["counts"]
    check(counts["resolved_windows"] > 0, "calibration_audit: no resolved windows")
    check(counts["joined_snapshots"] > 0, "calibration_audit: no scored snapshots")
    check(counts["market_tops"] > 0, "calibration_audit: no market tops")
    overall = next((row for row in ca["scope_rows"] if row["scope"] == "overall"), None)
    check(overall is not None, "calibration_audit: no overall score")
    if overall is not None:
        check(math.isfinite(overall["model_brier"]) and math.isfinite(overall["model_logloss"]),
              f"calibration_audit: model score not finite ({overall})")
        check(math.isfinite(overall["market_brier"]) and math.isfinite(overall["market_logloss"]),
              f"calibration_audit: market score not finite ({overall})")
        # The fixture's market mid is the model probability plus noise, so the
        # formula model must be at least as accurate as the market.
        check(overall["model_brier"] <= overall["market_brier"],
              f"calibration_audit: model Brier should beat the noisy market "
              f"({overall['model_brier']} vs {overall['market_brier']})")
        # Self-baseline: the model under test IS the formula model → identity.
        check(abs(overall["model_brier"] - overall["formula_brier"]) < 1e-12,
              f"calibration_audit: self-baseline model != formula ({overall})")
    for fname in ("metrics.json", "scores.csv", "verdict_table.csv", "reliability.csv", "report.html"):
        check((paths.out_dir / "calibration_audit" / fname).exists(),
              f"calibration_audit: {fname} not written")

    # --- the standardized evaluation harness (single source of truth) --------
    # Self-baseline: three predictors, model==formula identity, per-period verdict.
    ev = evaluate(paths, min_windows=5)
    ec = ev["counts"]
    check(ec["resolved_windows"] > 0, "evaluate: no resolved windows")
    check(ec["joined_snapshots"] > 0, "evaluate: no scored snapshots")
    ev_overall = next((r for r in ev["scope_rows"] if r["scope"] == "overall"), None)
    check(ev_overall is not None, "evaluate: no overall score")
    if ev_overall is not None:
        for k in ("model_brier", "formula_brier", "market_brier", "model_diracc"):
            check(math.isfinite(ev_overall[k]), f"evaluate: {k} not finite ({ev_overall})")
        check(abs(ev_overall["model_brier"] - ev_overall["formula_brier"]) < 1e-12,
              "evaluate: self-baseline model != formula (identity broken)")
    day = ev["verdict_table"]["day"]
    check(any(r["period"] != "ALL" for r in day["rows"]), "evaluate: no per-period verdict rows")
    check(all(b in day["stability"] for b in ("formula", "market")), "evaluate: stability summary missing")
    for fname in ("metrics.json", "scores.csv", "verdict_table.csv", "reliability.csv", "report.html"):
        check((paths.out_dir / "evaluate" / fname).exists(), f"evaluate: {fname} not written")

    # An external model's predictions score through the same harness; the
    # oracle-nudge fixture must beat the formula benchmark head-to-head.
    preds_path = make_predictions_fixture(paths.out_dir / "_predictions", paths.journal_dir)
    ext = evaluate(paths, predictions=preds_path, min_windows=5)
    check(ext["counts"]["joined_snapshots"] > 0, "evaluate: external predictions not scored")
    ext_formula = next((r for r in ext["verdict_table"]["day"]["rows"]
                        if r["benchmark"] == "formula" and r["period"] == "ALL"), None)
    check(ext_formula is not None and math.isfinite(ext_formula["brier_improve_pct"]),
          "evaluate: external vs-formula improvement not computed")
    if ext_formula is not None:
        check(ext_formula["brier_improve_pct"] > 0.0,
              f"evaluate: nudged model should beat formula (Δ={ext_formula.get('brier_improve_pct')}%)")
    ext_market = next((r for r in ext["verdict_table"]["day"]["rows"]
                      if r["benchmark"] == "market" and r["period"] == "ALL"), None)
    check(ext_market is not None and math.isfinite(ext_market["brier_improve_pct"]),
          "evaluate: external vs-market improvement not computed")

    # --- the short-horizon money judge (Phase-1 verdict) ---------------------
    # Synthesize the walk-forward OOS parquet (oracle-nudged toward each window's
    # outcome) so this runs LightGBM-free, then assert the profitable signal books
    # positive net PnL after fees when traded against the recorded book.
    make_short_oos_fixture(paths.out_dir, paths.journal_dir)
    mj = money_judge(paths, scope="all", variant="full")
    check(mj["counts"]["fires"] > 0, "money_judge: no fires")
    check(len(mj["sweep"]) > 0, "money_judge: empty sweep")
    check(mj["best"] is not None, "money_judge: no tradeable threshold")
    if mj["best"] is not None:
        check(mj["best"]["taker"]["net_pnl"] > 0.0,
              f"money_judge: profitable oracle signal should net > 0 "
              f"(got {mj['best']['taker']['net_pnl']})")
        check(mj["best"]["taker"]["win_rate"] > 0.9,
              f"money_judge: oracle fires should win ~all windows ({mj['best']['taker']['win_rate']})")
    check(mj["momentum"]["n_trades"] >= 0, "money_judge: momentum baseline missing")
    check(mj["late_window"]["n_trades"] == 0, "money_judge: late-window fired on BinanceCorrected fixture")
    check(math.isfinite(mj["lift"]["diracc_lift"]), "money_judge: majority-lift not computed")
    for fname in ("sweep.csv", "per_day.csv", "per_window.csv", "trades.csv", "metrics.json", "report.html"):
        check((paths.out_dir / "money_judge" / fname).exists(), f"money_judge: {fname} not written")

    # --- taker-rebate simulation (Part 2): tier math + stage over a trades.csv --
    check_rebate_sim_offline(paths, check)

    html = report(paths)
    check(Path(html).exists(), "report: report.html not written")

    # Offline self-check of the historical aggTrades download + integrity path.
    check_hist_offline(paths, check)

    # Offline self-check of the Telonex trial validation path (checks only — no zstd, so
    # this stays dependency-free; compression is exercised in tests/test_telonex.py).
    check_telonex_offline(paths, check)

    # Offline self-check of the historical ingest path (labels → dataset → backtest +
    # overlap parity), over a synthetic Telonex + aggTrades + journal store.
    check_historical_offline(paths, check)

    # Offline self-check of the competitor operating manuals (deep reconstruction +
    # provenance + anchor) and the competitor-clone backtests (bracket + tier + ladder).
    check_manuals_offline(paths, check)
    check_clones_offline(paths, check)

    return failures


def check_rebate_sim_offline(paths: Paths, check) -> None:
    """Taker-rebate simulation (Part 2): the tier math on a crafted high-volume frame
    (a clean None→Bronze→Silver crossing) plus the stage wiring over the model-taker
    trades.csv money_judge just wrote."""
    from . import rebate_sim as rs

    # One taker trade/day, wV = 2000·(1−0.5)·2.3 = 2300/day. The trailing-30-day
    # window crosses Bronze ($2k) on day 1 and Silver ($20k) on day 9; it never
    # reaches Gold (30·2300 = $69k). fee = $10/day → rebate = tier% · $10.
    rows = [{"series": "BTC-5m", "sample_ts_ms": (20_000 + k) * 86_400_000,
             "shares": 2000.0, "price": 0.5, "fee": 10.0, "net": -1.0} for k in range(40)]
    res = rs.compute_rebate_timeline(pd.DataFrame(rows))
    check(rs._peak_tier(res["tier_days"]) == "Silver", "rebate_sim: peak tier != Silver on crafted frame")
    check(res["tier_days"].get("None") == 1 and res["tier_days"].get("Bronze") == 8
          and res["tier_days"].get("Silver") == 31, f"rebate_sim: tier-day split wrong ({res['tier_days']})")
    t = res["totals"]
    check(abs(t["total_rebate"] - 27.2) < 1e-9, f"rebate_sim: total rebate {t['total_rebate']} != 27.20")
    check(abs(t["total_fees"] - 400.0) < 1e-9, f"rebate_sim: total fees {t['total_fees']} != 400")
    check(abs(t["corrected_net"] - (t["original_net"] + t["total_rebate"])) < 1e-12,
          "rebate_sim: corrected_net != net + rebate")
    check(t["total_paid"] <= t["total_rebate"] + 1e-9 and (t["total_rebate"] - t["total_paid"]) < 1.0,
          f"rebate_sim: $1-carry residual not < $1 ({t})")
    cur = res["current"]
    check(cur["tier"] == "Silver" and cur["next_tier"] == "Gold"
          and abs(cur["wv_to_next_tier"] - (200_000.0 - cur["wv_30d"])) < 1e-6,
          f"rebate_sim: current/marginal wrong ({cur})")

    # force_tier (C — tier honesty): the same frame forced to Platinum credits 0.32 × fees
    # every day regardless of the (Silver-earning) volume; corrected_net identity holds.
    fres = rs.compute_rebate_timeline(pd.DataFrame(rows), force_tier="Platinum")
    ft = fres["totals"]
    check(fres["force_tier"] == "Platinum" and abs(ft["total_rebate"] - 0.32 * ft["total_fees"]) < 1e-9,
          f"rebate_sim: force_tier rebate {ft['total_rebate']} != 0.32×fees")
    check(abs(ft["corrected_net"] - (ft["original_net"] + ft["total_rebate"])) < 1e-12,
          "rebate_sim: force_tier corrected_net identity broken")

    # Stage wiring over the model-taker trades.csv money_judge produced.
    mj_trades = paths.out_dir / "money_judge" / "trades.csv"
    if mj_trades.exists():
        out = rs.rebate_sim(paths, trades=mj_trades, out_name="verify")
        check(abs(out["totals"]["corrected_net"]
                  - (out["totals"]["original_net"] + out["totals"]["total_rebate"])) < 1e-9,
              "rebate_sim: corrected_net identity broken on money_judge trades")
        for fname in ("tier_timeline.csv", "metrics.json", "report.html"):
            check((paths.out_dir / "backtests" / "rebate_sim" / "verify" / fname).exists(),
                  f"rebate_sim: {fname} not written")


def check_telonex_offline(paths: Paths, check) -> None:
    """Exercise the Telonex validation checks with **no network** and **no zstd**: build a
    synthetic trial store + matching journal, then run ``telonex_validate.validate`` over
    them in a throwaway dir. Asserts each data-quality check passes and that the external
    clock cross-check recovers the constant offset baked into the fixture."""
    from . import telonex_validate as tv
    from .fixtures import (
        TX_BINANCE_OFFSET_MS, TX_CLOB_OFFSET_MS, make_telonex_fixture, telonex_offline_fetch,
    )

    base = paths.out_dir / "_telonex_selfcheck"
    make_telonex_fixture(base / "telonex", base / "journal")
    tp = replace(paths, telonex_dir=base / "telonex", journal_dir=base / "journal", out_dir=base / "out")
    result = tv.validate(tp, compress=False, catalog=True, fetch=telonex_offline_fetch())
    checks = result["checks"]
    for name in ("coverage", "cadence", "depth", "completeness", "clock"):
        check(checks[name]["result"] == "PASS", f"telonex: {name} = {checks[name]['result']}")
    bin_off = checks["clock"]["external_binance_trades"].get("median_offset_ms")
    check(bin_off is not None and abs(bin_off - TX_BINANCE_OFFSET_MS) < 5.0,
          f"telonex: binance clock offset {bin_off} (want ~{TX_BINANCE_OFFSET_MS})")
    clob_off = checks["clock"]["external_polymarket_top"].get("median_offset_ms")
    check(clob_off is not None and abs(clob_off - TX_CLOB_OFFSET_MS) < 5.0,
          f"telonex: clob clock offset {clob_off} (want ~{TX_CLOB_OFFSET_MS})")


def check_historical_offline(paths: Paths, check) -> None:
    """Exercise the historical ingest path with **no network**: a synthetic Telonex +
    aggTrades store (a Telonex-owned day) drives labels → dataset → backtest, and a
    matching recorder+Telonex store on an overlap day drives the telonex-vs-recorder
    parity guard. Asserts resolution/coverage, ``depth_source`` tagging, the parity PASS,
    and the backtest's controls."""
    from datetime import date, datetime, timezone

    from . import backtest_momentum as bmod
    from . import backtest_pair_lean as bplmod
    from . import historical_dataset as hdmod
    from . import historical_labels as hlmod
    from . import historical_resolutions as hrmod
    from .fixtures import (
        make_historical_fixture, make_overlap_parity_fixture, write_historical_coverage,
    )
    from .io.polymarket import PolymarketNotFound

    base = paths.out_dir / "_historical_selfcheck"
    res = make_historical_fixture(base / "telonex", base / "aggtrades")
    write_historical_coverage(base / "out", res["coverage"])
    hp = replace(paths, telonex_dir=base / "telonex", hist_dir=base / "aggtrades",
                 journal_dir=base / "journal", depth_dir=base / "depth", out_dir=base / "out")

    # Part A.2 — official resolution: catalog decodes + validates 0-mismatch vs an offline
    # CLOB API seam (any mismatch is a hard fail).
    cid_out = {v["condition_id"]: v["outcome"] for v in res["res_map"].values()}

    def _api(url: str):
        cid = url.rstrip("/").rsplit("/", 1)[-1]
        out = cid_out.get(cid)
        if out is None:
            raise PolymarketNotFound(url)
        other = "Down" if out == "Up" else "Up"
        return {"tokens": [{"outcome": out, "winner": True}, {"outcome": other, "winner": False}]}

    api = hrmod.validate_vs_api(hp, series=("BTC-5m", "ETH-5m"), min_sample=5, fetch=_api)
    check(api["ok"] and api["mismatch"] == 0, f"historical_resolutions: catalog vs API mismatch ({api['mismatch']})")
    check(api["api_cross_checked"] >= 1, "historical_resolutions: nothing cross-checked vs the API")

    # Part 2 — labels: official resolution primary (~100% coverage), proxy grading.
    lab = hlmod.historical_labels(hp, series=("BTC-5m", "ETH-5m"))
    check(lab["counts"]["resolved_official"] == 12, f"historical_labels: resolved {lab['counts']['resolved_official']} (want 12)")
    check(lab["counts"]["coverage_frac"] == 1.0, f"historical_labels: coverage {lab['counts']['coverage_frac']} (want 1.0)")
    check(lab["counts"]["unlabeled_no_official"] == 0, "historical_labels: unexpected unlabeled")
    check(lab["proxy_grading"]["sampled"] > 0, "historical_labels: proxy grading did not run")

    # Part 1 — dataset: telonex rows, depth_source, combine, schema.
    dsr = hdmod.historical_dataset(hp, series=("BTC-5m", "ETH-5m"))
    check(dsr["counts"]["samples"] > 0, "historical_dataset: no samples")
    check(dsr["counts"]["samples_by_depth_source"].get("telonex", 0) > 0,
          "historical_dataset: no telonex-sourced rows")
    cr = hdmod.combine(hp)
    check(cr["rows"] == dsr["counts"]["samples"], "historical_dataset: combine row mismatch")
    hist_df = pd.read_parquet(hp.table("historical_dataset"))
    check(list(hist_df.columns) == hdmod.HIST_COLUMNS, "historical_dataset: column set != HIST_COLUMNS")
    check((hist_df["label_source"] == "telonex").all(), "historical_dataset: non-telonex label_source")
    check(bool(hist_df["chainlink"].isna().all()), "historical_dataset: chainlink not NaN on telonex era")
    check(bool(hist_df["depth_imb_20"].notna().any()) and bool(hist_df["pm_mid"].notna().any()),
          "historical_dataset: telonex depth/PM features not populated")
    for fname in ("SCHEMA.md", "metadata.json"):
        check((base / "out" / "historical_dataset" / fname).exists(), f"historical_dataset: {fname} not written")

    # Part 3 — backtest: runs, produces controls + outputs.
    bt = bmod.backtest_momentum(hp, series=("BTC-5m", "ETH-5m"), seeds=6)
    check(bt["counts"]["resolved_windows"] > 0, "backtest_momentum: no resolved windows")
    check(len(bt["controls"]["shuffled_nets"]) == 6, "backtest_momentum: shuffled control did not run")
    check("real_beats_shuffled" in bt["controls"], "backtest_momentum: control verdict missing")
    for fname in ("report.html", "metrics.json", "trades.csv", "per_series_month.csv", "per_hour.csv"):
        check((base / "out" / "backtests" / "momentum" / fname).exists(), f"backtest_momentum: {fname} not written")

    # Part 4 — pair-lean backtest: LightGBM-free (reuses the OOS-parquet contract via an injected
    # OOS frame + the official res_map). Synthetic OOS leaning toward each window's outcome so the
    # maker-completion + baselines + controls run; assert the structure + outputs.
    pl_oos = pd.DataFrame([
        {"series": sk, "window_open_ms": open_ms, "sample_ts_ms": open_ms + k * 30_000,
         "p_up": 0.85 if v["outcome"] == "Up" else 0.15}
        for (sk, open_ms), v in res["res_map"].items() for k in range(1, 9)
        if open_ms + k * 30_000 < open_ms + 300_000
    ])
    pl = bplmod.backtest_pair_lean(hp, series=("BTC-5m", "ETH-5m"), oos=pl_oos, res_map=res["res_map"],
                                   regime_from=date(2026, 5, 1), seeds=3, thetas=[0.0, 0.1],
                                   pair_costs=[0.94, 0.98], depth_fracs=[1.0])
    check(pl["current_regime_windows"] > 0, "backtest_pair_lean: no current-regime windows")
    check(pl["best_config"] is not None, "backtest_pair_lean: no tradeable config")
    check(len(pl["sweep"]) == 4, "backtest_pair_lean: sweep grid wrong size")
    check(pl["baselines"]["never_trading_net"] == 0.0, "backtest_pair_lean: never-trade != 0")
    check("net_pnl" in pl["baselines"]["momentum"] and "net_pnl" in pl["baselines"]["model_taker"],
          "backtest_pair_lean: baselines missing")
    if pl["best_config"] is not None:
        check(pl["best_config"]["control"] is not None and "beats" in pl["best_config"]["control"],
              "backtest_pair_lean: control verdict missing")
    for fname in ("metrics.json", "sweep.csv", "per_series_best.csv", "report.html"):
        check((base / "out" / "backtests" / "pair_lean" / fname).exists(),
              f"backtest_pair_lean: {fname} not written")

    # Part 5 — maker-core defense backtest: LightGBM-free (injected dir10 OOS + res_map). The
    # faithful maker replay fills resting quotes from the fixture's PM trade tape; assert the
    # baseline + defend/lean configs run, the locked+stranded split reconciles, the shuffled
    # control ran, and the outputs are written.
    from . import backtest_maker_core as mcmod

    mc_oos = pd.DataFrame([
        {"series": sk, "window_open_ms": open_ms, "sample_ts_ms": open_ms + k * 15_000,
         "p_up": 0.9 if v["outcome"] == "Up" else 0.1}
        for (sk, open_ms), v in res["res_map"].items() for k in range(1, 20)
        if k * 15_000 < 300_000
    ])
    mc = mcmod.backtest_maker_core(hp, series=("BTC-5m", "ETH-5m"), oos=mc_oos,
                                   res_map=res["res_map"], regime_from=date(2026, 5, 1), seeds=2,
                                   defend_thetas=[0.10, 0.20], lean_mults=[0.5, 1.0])
    check(mc["current_regime_windows"] > 0, "backtest_maker_core: no current-regime windows")
    cfgs = {c["config"] for c in mc["configs"]}
    check("baseline" in cfgs and "defend|0.1" in cfgs and "lean|0.5" in cfgs,
          "backtest_maker_core: missing variant configs")
    for c in mc["configs"]:
        recon = abs(c["net_pnl"] - (c["locked_pnl"] + c["stranded_pnl"]))
        check(recon < 0.01, f"backtest_maker_core: {c['config']} locked+stranded != net ({recon:.4f})")
    win = mc.get("winner")
    check(win is None or "control" in win["eval"], "backtest_maker_core: winner control missing")
    for fname in ("metrics.json", "per_series.csv", "report.html"):
        check((base / "out" / "backtests" / "maker_core" / fname).exists(),
              f"backtest_maker_core: {fname} not written")

    # C5 — overlap-day telonex-vs-recorder parity guard (matching book → PASS).
    pday = date(2026, 7, 4)
    pbase = paths.out_dir / "_historical_parity"
    make_overlap_parity_fixture(pbase / "telonex", pbase / "aggtrades", pbase / "journal", pbase / "depth", day=pday)
    pp = replace(paths, telonex_dir=pbase / "telonex", hist_dir=pbase / "aggtrades",
                 journal_dir=pbase / "journal", depth_dir=pbase / "depth", out_dir=pbase / "out")
    since = int(datetime(2026, 7, 4, tzinfo=timezone.utc).timestamp()) * 1000
    until = int(datetime(2026, 7, 5, tzinfo=timezone.utc).timestamp()) * 1000
    hdmod.historical_dataset(pp, series=("BTC-5m",), since_ms=since, until_ms=until)
    par = hdmod.overlap_parity(pp, ("BTC-5m",), [pday])
    check(par["n_features"] == 6, f"overlap_parity: {par['n_features']} features (want 6)")
    check(par["pass"], f"overlap_parity: telonex-vs-recorder gap — {par['verdicts']}")

    check_shadow_parity_offline(check)


def check_manuals_offline(paths: Paths, check) -> None:
    """Competitor operating manual (deep reconstruction + provenance + anchor): builds a
    manual from a synthetic competitor cache aligned to a post-Jun-5 Telonex fixture, and
    asserts the classification counts, the FACT+CALC-only provenance lint, the merge-velocity
    block, and that a consistent official curve is not flagged while an inconsistent one goes
    red (OUR RECONSTRUCTION SUSPECT)."""
    import os
    from datetime import date

    from . import fixtures as fx
    from .competitors import manuals as mn
    from .competitors import manuals_report as mr

    base = paths.out_dir / "_manuals_selfcheck"
    res = fx.make_historical_fixture(base / "telonex", base / "hist", day=date(2026, 6, 15))
    hp = replace(paths, telonex_dir=base / "telonex", hist_dir=base / "hist", out_dir=base / "out")

    def _manual(comp_dir, addr, inconsistent=False, crossing=1):
        exp = fx.make_competitor_fills_fixture(comp_dir, res, addr=addr, crossing_per_window=crossing,
                                               inconsistent=inconsistent)
        prev = os.environ.get("MODEL_LAB_COMPETITORS_DIR")
        os.environ["MODEL_LAB_COMPETITORS_DIR"] = str(comp_dir)
        try:
            m = mn.build_manual(hp, {"handle": exp["handle"], "address": exp["addr"],
                                     "profile": {"takerTierName": "Platinum", "weightedVolume": 1e6}})
        finally:
            if prev is None:
                os.environ.pop("MODEL_LAB_COMPETITORS_DIR", None)
            else:
                os.environ["MODEL_LAB_COMPETITORS_DIR"] = prev
        return m, exp

    m, exp = _manual(base / "comp_ok", "0xc10ffee0aa000000000000000000000000000000")
    check(m["coverage"]["our_fills_in_telonex_window"] == exp["n_fills"], "manuals: fill count mismatch")
    check(m["price_vs_book"]["at_touch"] == exp["at_touch"]
          and m["price_vs_book"]["crossing"] == exp["crossing"], "manuals: classification count mismatch")
    check(m["merge_focus"] and "merge_velocity" in m, "manuals: merge focus/velocity missing")
    check(m["anchor"]["suspect"] is False, "manuals: consistent fixture flagged suspect")
    body = mr.manual_body(m)  # provenance lint: manual body FACT+CALC only
    check("class='est'" not in body and "class='unv'" not in body, "manuals: EST/UNV span in manual body")
    check(len(m["hand_trace"]) == exp["n_windows"], "manuals: hand-trace window count")

    m2, _ = _manual(base / "comp_bad", "0xbadc0ffee000000000000000000000000000000",
                    inconsistent=True, crossing=0)
    check(m2["anchor"]["suspect"] is True, "manuals: inconsistent official curve NOT flagged suspect")
    check(bool(m2["anchor"]["suspect_reasons"]), "manuals: suspect has no reasons")


def check_clones_offline(paths: Paths, check) -> None:
    """Competitor-clone backtests (B/C/D/E): a taker + a maker clone over the historical
    fixture. Asserts the shared trades schema + fee/cost/net identities, the maker-fill
    bracket ordering (pessimistic ≤ optimistic), both controls, tier honesty, the
    momentum-exit sweep, and the clone-vs-owner ladder."""
    from datetime import date

    from . import backtest_clones as bc
    from . import fixtures as fx

    base = paths.out_dir / "_clones_selfcheck"
    res = fx.make_historical_fixture(base / "telonex", base / "hist")
    fx.write_historical_coverage(base / "out", res["coverage"])
    hp = replace(paths, telonex_dir=base / "telonex", hist_dir=base / "hist",
                 journal_dir=base / "journal", depth_dir=base / "depth", out_dir=base / "out")

    for clone in ("takerner", "0xb27b"):
        m = bc.backtest_clone(hp, clone, series=("BTC-5m", "ETH-5m"), res_map=res["res_map"],
                              regime_from=date(2026, 5, 1), seeds=3)
        check(m["current_regime_windows"] > 0, f"clones: {clone} no current-regime windows")
        check("shuffled_outcome" in m["controls"] and "matched_frequency_random" in m["controls"],
              f"clones: {clone} controls missing")
        check(set(m["momentum_exit"]["by_timeout"]) == {"T15", "T30", "T60"},
              f"clones: {clone} momentum-exit sweep missing")
        check("clone_vs_owner_ladder" in m and "gap_reconstructed_minus_clone" in m["clone_vs_owner_ladder"],
              f"clones: {clone} owner ladder missing")
        odir = base / "out" / "backtests" / f"clone_{clone}"
        for fname in ("trades.csv", "metrics.json", "report.html"):
            check((odir / fname).exists(), f"clones: {clone} {fname} not written")
        df = pd.read_csv(odir / "trades.csv")
        if len(df):
            bad = sum(1 for r in df.itertuples(index=False)
                      if abs(r.cost - (r.shares * r.price + r.fee)) > 1e-6
                      or abs(r.net - (r.payoff - r.cost)) > 1e-6)
            check(bad == 0, f"clones: {clone} accounting identity broken ({bad} rows)")
        br = m["maker_fill_bracket"]
        if br is not None:
            check(br["pessimistic_net"] <= br["optimistic_net"] + 1e-9,
                  f"clones: {clone} maker bracket not well-ordered")
            check(m["tier_honesty"].get("our_tier", {}).get("tier") == "Bronze",
                  f"clones: {clone} tier honesty Bronze run missing")


def check_shadow_parity_offline(check) -> None:
    """The shadow feature-parity guard, exercised in-memory (the basis-bug
    tripwire): near-identical live-vs-offline features PASS, and a scale bug on a
    price feature FAILs loudly (abs_ratio out of band)."""
    import numpy as np
    import pandas as pd

    from . import shadow_parity as sp

    rng = np.random.default_rng(7)
    n = 120
    key = {"series": ["BTC-5m"] * n, "window_open_ms": [0] * n,
           "sample_ts_ms": list(range(0, n * 5000, 5000))}
    off = dict(key)
    live = {**key, "p_up": rng.random(n)}
    for f in sp.FULL_FEATURES:
        base = rng.normal(size=n)
        off[f] = base
        live[f] = base + rng.normal(scale=1e-4, size=n)
    good = sp.run_parity(pd.DataFrame(live), pd.DataFrame(off), min_n=50)
    check(good["passed"], f"shadow_parity: clean features should PASS — {good['failures'][:3]}")

    broken = dict(live)
    broken["sigma_1s"] = (np.asarray(off["sigma_1s"]) * 3.0).tolist()
    bad = sp.run_parity(pd.DataFrame(broken), pd.DataFrame(off), min_n=50)
    check(not bad["passed"], "shadow_parity: a 3x scale bug must FAIL the guard")


def main(argv: list[str] | None = None) -> int:
    with tempfile.TemporaryDirectory(prefix="model-lab-verify-") as tmp:
        root = Path(tmp)
        counts = make_fixture(root / "journal", root / "depth")
        agg = make_aggtrades_fixture(root / "aggtrades")
        print(f"[verify] synthetic fixture: {counts['journal_records']:,} records, "
              f"{counts['depth_frames']:,} depth frames, {counts['windows']} windows, "
              f"{agg['rows']:,} aggTrades ({agg['symbols']} symbols)")
        paths = Paths(
            journal_dir=root / "journal", depth_dir=root / "depth",
            out_dir=root / "out", hist_dir=root / "aggtrades",
        )
        failures = run_checks(paths)

    if failures:
        print("VERIFY: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFY: PASS — ingest → features → labels → dataset → short_horizon → feature_set → "
          "basis-guard → learn → validate → calibration_audit → research → evaluate → money_judge "
          "→ rebate_sim → report + hist + historical (labels → dataset → backtest + pair-lean + "
          "parity) + manuals + clones + shadow-parity all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
