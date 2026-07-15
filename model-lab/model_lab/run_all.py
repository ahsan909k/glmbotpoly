"""Run every stage in order against the configured data (convenience wrapper).

Equivalent to running, in sequence:
``ingest → features → labels → dataset → short_horizon → feature_set → learn → validate → calibration_audit → research → evaluate → report``.
Each stage still has its own entry point (``python -m model_lab.<stage>``); this
just chains them for a full refresh over the real journal.

Run:  ``python -m model_lab.run_all``  (honors ``--journal-dir/--depth-dir/--out``).
"""

from __future__ import annotations

import sys

from .calibration_audit import calibration_audit
from .config import resolve_paths, stage_parser
from .dataset import dataset
from .evaluate import evaluate
from .feature_set import feature_set
from .features import features
from .ingest import ingest
from .labels import labels
from .learn import learn
from .report import report
from .research import research
from .short_horizon import short_horizon
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "run all stages")
    parser.add_argument(
        "--with-gbt", action="store_true",
        help="also run the opt-in GBT challenger (learn_gbt) + comparison (compare) after learn "
             "— requires the gbt extra (pip install -r requirements-gbt.txt)",
    )
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    print(f"[run_all] journal={paths.journal_dir}  depth={paths.depth_dir}  out={paths.out_dir}")

    counts = ingest(paths)
    print(f"[run_all] ingest: {counts}")
    if counts["ticks"] == 0 and counts["model"] == 0:
        print("[run_all] no journal data found — nothing to do.")
        return 1

    print(f"[run_all] features: {features(paths):,} rows")
    n_fwd, n_win = labels(paths)
    print(f"[run_all] labels: {n_fwd:,} forward, {n_win:,} windows")
    ds = dataset(paths)
    dc = ds["counts"]
    print(f"[run_all] dataset: {dc['samples']:,} samples over {dc['windows']:,} windows "
          f"({dc['samples_in_coverage']:,} in coverage)")
    sh = short_horizon(paths)
    sc = sh["counts"]
    print(f"[run_all] short_horizon: {sc['samples']:,} samples over {sc['windows']:,} windows "
          f"(depth {sh['coverage']['depth_pct']:.1%} / book {sh['coverage']['book_pct']:.1%} covered)")
    fs = feature_set(paths)
    print(f"[run_all] feature_set: {fs['counts']['features']} features over "
          f"{fs['counts']['samples']:,} samples (sanity {fs['status']})")
    lr = learn(paths)
    out_blk = lr["targets"].get("outcome", {})
    out_h = (out_blk.get("harness") or {}).get("vs_formula", {})
    out_sc = lr["shuffled_control"].get("outcome", {})
    print(f"[run_all] learn: outcome OOS Brier={out_blk.get('pooled', {}).get('brier')}, "
          f"vs-formula Δ={out_h.get('brier_improve_pct')}%, "
          f"shuffled-collapsed={out_sc.get('collapsed')}")
    if args.with_gbt:
        # Opt-in second challenger — imported lazily so the core pipeline never needs lightgbm.
        try:
            from .compare import compare
            from .learn_gbt import learn_gbt
            from .learn_short_gbt import learn_short_gbt
            from .money_judge import money_judge
        except ImportError as exc:
            print(f"[run_all] --with-gbt requires the gbt extra: {exc}")
            return 1
        lg = learn_gbt(paths)
        lg_plain = lg["targets"].get("outcome", {}).get("plain", {})
        lg_res = lg["targets"].get("outcome", {}).get("residual", {})
        print(f"[run_all] learn_gbt: outcome/plain Brier={lg_plain.get('pooled', {}).get('brier')}, "
              f"outcome/residual Brier={lg_res.get('pooled', {}).get('brier')}")
        lsg = learn_short_gbt(paths)
        lsg_all = lsg.get("scopes", {}).get("all", {})
        lsg_base = lsg_all.get("targets", {}).get("fwd10", {}).get("base", {})
        lsg_full = lsg_all.get("targets", {}).get("fwd10", {}).get("full", {})
        lsg_bvf = lsg_all.get("depth_contribution", {}).get("fwd10", {}).get("base_vs_full", {}).get("overall", {})
        print(f"[run_all] learn_short_gbt: fwd10/base Brier={lsg_base.get('pooled', {}).get('brier')}, "
              f"fwd10/full Brier={lsg_full.get('pooled', {}).get('brier')}, "
              f"base→full ΔBrier={lsg_bvf.get('brier_delta')}")
        cmp = compare(paths)
        print(f"[run_all] compare: {cmp['verdict']}")
        # The money judge consumes learn_short_gbt's OOS parquet (itself LightGBM-free).
        mj = money_judge(paths)
        mj_best = mj.get("best")
        print(f"[run_all] money_judge: best taker net="
              f"{(mj_best or {}).get('taker', {}).get('net_pnl')}, "
              f"momentum net={mj['comparison']['momentum_net']}")
    v = validate(paths)
    print(f"[run_all] validate: Brier={v['calibration'].get('brier')}, "
          f"σ-corr={v['sigma_reproduction'].get('correlation')}")
    ca = calibration_audit(paths)
    ov = next((row for row in ca["scope_rows"] if row["scope"] == "overall"), None)
    print(f"[run_all] calibration_audit: windows={ca['counts']['resolved_windows']}, "
          f"model Brier={ov['model_brier'] if ov else None}, "
          f"market Brier={ov['market_brier'] if ov else None}")
    r = research(paths)
    print(f"[run_all] research: has_depth={r['has_depth']}, "
          f"IC(imbalance)={r.get('pooled', {}).get('ic_imbalance')}")
    ev = evaluate(paths)
    ev_overall = next((row for row in ev["scope_rows"] if row["scope"] == "overall"), None)
    ev_mkt = next((r for r in ev["verdict_table"]["day"]["rows"]
                   if r["benchmark"] == "market" and r["period"] == "ALL"), None)
    print(f"[run_all] evaluate: windows={ev['counts']['resolved_windows']}, "
          f"model Brier={ev_overall['model_brier'] if ev_overall else None}, "
          f"vs-market Δ={ev_mkt['brier_improve_pct'] if ev_mkt else None}%")
    print(f"[run_all] report: {report(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
