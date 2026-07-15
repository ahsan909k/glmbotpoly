"""Export the champion ``dir10_full`` walk-forward booster for live shadow inference.

The walk-forward stage (:mod:`model_lab.learn_walkforward`) trains a fresh booster per
fold and discards it — only the OOS *prediction* parquets survive. Shadow mode needs a
deployable booster on disk. This module reproduces ``learn_walkforward._final_model``
**deterministically** (full history, 2M-row cap, ``num_boost_round`` + ``seed`` read from
``out/learn_walkforward/metrics.json``) and writes:

  * ``models/model_{target}_{variant}.txt`` — the native LightGBM model the Rust
    ``shadow`` crate loads and walks;
  * ``models/model_{target}_{variant}.meta.json`` — the model identity + training-end
    date the shadow crate journals with every prediction and the dashboard uses for the
    "model stale, refit due" alert;
  * ``crates/shadow/tests/fixtures/{model_{target}_{variant}.txt, parity_cases.json}`` —
    the offline 1e-6 export-parity fixtures (feature vectors + ``booster.predict``
    expectations, **including NaN- and exactly-0-bearing rows** to exercise the walker's
    native-missing / ``missing_type=Zero`` routing).

The full-history ``_final_model`` refit is the deployable model (max data, latest venue
regime, deterministic). Research/deploy artifact producer — nothing here runs inside the
Rust bot; the bot loads the saved ``model.txt`` at runtime via the ``[shadow]`` config.

Opt-in dependency: LightGBM (the ``[gbt]`` extra), isolated in :mod:`model_lab.lib.gbt`.

Run::

    python -m model_lab.export_champion                       # dir10 / full
    python -m model_lab.export_champion --target dir10 --variant full
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import learn_walkforward as lw
from . import money_judge as mj
from .config import REPO_ROOT, resolve_paths, stage_parser
from .lib import gbt

FIXTURE_DIR = REPO_ROOT / "crates" / "shadow" / "tests" / "fixtures"
MODELS_DIR = REPO_ROOT / "models"
N_SAMPLED_CASES = 48  # real feature rows (naturally include NaN patterns)


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git absent / detached
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_edge_cases(feats: list[str]) -> np.ndarray:
    """Rows that stress the tree-walker's native-missing / zero routing: all-NaN,
    all-zero, and a few mixed NaN/0/typical patterns."""
    n = len(feats)
    rows = [
        np.full(n, np.nan, dtype=np.float32),        # everything missing
        np.zeros(n, dtype=np.float32),               # everything exactly 0 (missing_type=Zero)
    ]
    # a mid-market-ish row with the depth/PM block NaN (the recorder-uncovered pattern)
    mixed = np.zeros(n, dtype=np.float32)
    for i, name in enumerate(feats):
        if name == "p_up_model":
            mixed[i] = 0.5
        elif name == "tau_secs":
            mixed[i] = 120.0
        elif name == "elapsed_secs":
            mixed[i] = 180.0
        elif name.startswith(("depth_", "microprice", "bid_depth", "ask_depth", "pm_")):
            mixed[i] = np.float32(np.nan)
    rows.append(mixed)
    return np.asarray(rows, dtype=np.float32)


def export_champion(paths, *, target: str = "dir10", variant: str = "full",
                    seed: int = lw.DEFAULT_SEED, fee_rate: float = mj.CRYPTO_FEE_RATE) -> dict:
    if variant not in lw.VARIANTS:
        raise SystemExit(f"unknown variant {variant!r} (choose from {list(lw.VARIANTS)})")
    feats = lw.VARIANTS[variant]

    # Round count + seed from the canonical walk-forward metrics (median fold best_iteration).
    metrics_path = paths.out_dir / "learn_walkforward" / "metrics.json"
    rounds = None
    used_seed = seed
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = m.get("final_models", {}).get(target, {}).get("num_boost_round")
        used_seed = int(m.get("config", {}).get("seed", seed))
        print(f"[export] metrics.json: seed={used_seed} num_boost_round={rounds}")
    else:
        print(f"[export] no metrics.json at {metrics_path} — using seed={seed}, deriving rounds")
    params = lw._params(used_seed)

    print("[export] loading historical_dataset.parquet (full history)…")
    mat = lw._load_matrix(paths, since_ms=None, until_ms=None, days=0, series=None)
    print(f"[export] loaded {len(mat):,} rows")
    sub, y, info = lw._prep_target(mat, target, fee_rate)
    n = len(y)
    idx = np.arange(n)
    if n > lw.DEFAULT_MAX_TRAIN_ROWS:
        idx = np.sort(np.random.default_rng(int(params["seed"])).choice(
            n, lw.DEFAULT_MAX_TRAIN_ROWS, replace=False))
    x = sub[feats].iloc[idx].to_numpy(dtype=np.float32)
    y_fit = y[idx]
    ts = sub["sample_ts_ms"].to_numpy(dtype="int64")[idx]
    n_rounds = int(rounds) if (rounds and int(rounds) > 0) else lw._inner_best_iteration(
        x, y_fit, ts, info[idx], params=params, purge_ms=0)
    print(f"[export] fitting {n_rounds}-round booster on {len(y_fit):,} rows × {len(feats)} feats…")
    booster = gbt.refit(x, y_fit, params=params, num_boost_round=n_rounds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"model_{target}_{variant}.txt"
    model_path = MODELS_DIR / model_name
    gbt.save(booster, model_path, num_iteration=n_rounds)

    trained_through_ms = int(ts.max())
    meta = {
        "target": target,
        "variant": variant,
        "feature_names": list(feats),
        "seed": int(params["seed"]),
        "num_boost_round": int(n_rounds),
        "n_train": int(len(y_fit)),
        "class_balance_pos_frac": float(np.mean(y_fit)),
        "trained_through_ms": trained_through_ms,
        "trained_through_date": datetime.fromtimestamp(
            trained_through_ms / 1000.0, tz=timezone.utc).date().isoformat(),
        "sha256": _sha256(model_path),
        "git_rev": _git_rev(),
        "exported_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = MODELS_DIR / f"model_{target}_{variant}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- export-parity fixtures (for the Rust 1e-6 walker test) ------------------
    rng = np.random.default_rng(int(params["seed"]) ^ 0xF17)
    sample_idx = rng.choice(len(x), min(N_SAMPLED_CASES, len(x)), replace=False)
    cases_x = np.vstack([x[sample_idx], _synthetic_edge_cases(feats)]).astype(np.float32)
    expected = np.asarray(booster.predict(cases_x, num_iteration=n_rounds), dtype=float)
    parity = {
        "model_file": model_name,
        "feature_names": list(feats),
        "cases": [
            {"features": [None if not np.isfinite(v) else float(v) for v in row],
             "expected_p_up": float(p)}
            for row, p in zip(cases_x.tolist(), expected)
        ],
    }
    (FIXTURE_DIR / model_name).write_bytes(model_path.read_bytes())
    (FIXTURE_DIR / "parity_cases.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")

    print(f"[export] wrote {model_path} ({model_path.stat().st_size:,} bytes, sha {meta['sha256'][:12]})")
    print(f"[export] wrote {meta_path}")
    print(f"[export] wrote {len(parity['cases'])} parity cases -> {FIXTURE_DIR / 'parity_cases.json'}")
    print(f"[export] trained_through={meta['trained_through_date']} class_balance={meta['class_balance_pos_frac']:.4f}")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "export_champion")
    parser.add_argument("--target", default="dir10", help="label target (default dir10)")
    parser.add_argument("--variant", default="full", choices=list(lw.VARIANTS),
                        help="feature set (default full = the 24-feature champion)")
    parser.add_argument("--seed", type=int, default=lw.DEFAULT_SEED)
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    export_champion(paths, target=args.target, variant=args.variant, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
