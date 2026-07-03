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

from . import hist, hist_integrity
from .calibration_audit import calibration_audit
from .config import Paths
from .dataset import dataset
from .features import features
from .fixtures import make_aggtrades_fixture, make_fixture
from .ingest import ingest
from .io.binance_archive import ArchiveNotFound, checksum_url, daterange, day_url
from .labels import labels
from .report import report
from .research import research
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
    for fname in ("metrics.json", "scores.csv", "reliability.csv", "report.html"):
        check((paths.out_dir / "calibration_audit" / fname).exists(),
              f"calibration_audit: {fname} not written")

    html = report(paths)
    check(Path(html).exists(), "report: report.html not written")

    # Offline self-check of the historical aggTrades download + integrity path.
    check_hist_offline(paths, check)

    return failures


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
    print("VERIFY: PASS — ingest → features → labels → dataset → validate → "
          "calibration_audit → research → report + hist all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
