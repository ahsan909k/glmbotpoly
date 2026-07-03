"""Stage 6 — report.

Assemble a single self-contained ``out/report.html`` from the validate +
research outputs: the model calibration (reliability curve + Brier), the σ_1s
reproduction, and the depth-signal IC/AUC table. Plots are inlined as base64
PNGs so the file is portable.

Verify:  ``python -m model_lab.report``  then open ``out/report.html``.
"""

from __future__ import annotations

import base64
import io
import json
import sys

import matplotlib

matplotlib.use("Agg")  # headless: render to PNG, never open a window.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Paths, resolve_paths, stage_parser  # noqa: E402


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _reliability_png(reliability: list[dict]) -> str | None:
    if not reliability:
        return None
    df = pd.DataFrame(reliability)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    ax.plot(df["mean_pred"], df["empirical_rate"], "o-", color="#1f77b4", label="model")
    ax.set_xlabel("mean predicted p_up")
    ax.set_ylabel("empirical Up rate")
    ax.set_title("Calibration (reliability)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _png(fig)


def _ic_png(per_asset: dict) -> str | None:
    if not per_asset:
        return None
    assets = list(per_asset.keys())
    ic = [per_asset[a].get("ic_imbalance", float("nan")) for a in assets]
    auc = [per_asset[a].get("auc_imbalance", float("nan")) for a in assets]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
    ax1.bar(assets, ic, color="#2ca02c")
    ax1.axhline(0, color="#333", linewidth=0.8)
    ax1.set_title("IC: imbalance → fwd 5s return")
    ax2.bar(assets, auc, color="#ff7f0e")
    ax2.axhline(0.5, color="#333", linewidth=0.8, linestyle="--")
    ax2.set_title("AUC: imbalance → up-move")
    ax2.set_ylim(0, 1)
    return _png(fig)


def _load_json(path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f"<p class='muted'>{alt}: not available.</p>"
    return f"<img alt='{alt}' src='data:image/png;base64,{b64}'/>"


def build_html(validation: dict, research: dict) -> str:
    cal = validation.get("calibration", {})
    phi = validation.get("phi_identity", {})
    sig = validation.get("sigma_reproduction", {})
    pooled = research.get("pooled", {})

    rel_img = _img_tag(_reliability_png(cal.get("reliability", [])), "reliability curve")
    ic_img = _img_tag(_ic_png(research.get("per_asset", {})), "IC/AUC by asset")

    def g(d: dict, k: str) -> str:
        v = d.get(k)
        return "n/a" if v is None or (isinstance(v, float) and v != v) else (f"{v:.4f}" if isinstance(v, float) else str(v))

    depth_note = (
        "Depth20 data present — microstructure signal measured below."
        if research.get("has_depth")
        else "No depth20 data captured yet — run <code>bot record</code> / <code>bot run</code> "
        "with <code>feeds.binance_depth_capture</code> enabled, then re-run the lab."
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>model-lab report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; }} td, th {{ padding: 4px 12px; text-align: left; }}
 .muted {{ color: #888; }} code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
 img {{ max-width: 100%; }}
</style></head><body>
<h1>model-lab report</h1>
<p class="muted">Offline validation + microstructure research over the Polymarket Up/Down bot's journal.</p>

<h2>Model validation</h2>
<table>
 <tr><th>Φ(z) identity — median |Δ|</th><td>{g(phi, 'median_abs')}</td><td>max |Δ| {g(phi, 'max_abs')} (n={g(phi, 'n')})</td></tr>
 <tr><th>σ_1s reproduction — correlation</th><td>{g(sig, 'correlation')}</td><td>median rel. err {g(sig, 'median_rel_err')} (n={g(sig, 'n')})</td></tr>
 <tr><th>Calibration — Brier</th><td>{g(cal, 'brier')}</td><td>windows {g(cal, 'n_windows')}</td></tr>
</table>
{rel_img}

<h2>Depth microstructure research</h2>
<p>{depth_note}</p>
<table>
 <tr><th>IC(imbalance → fwd 5s ret)</th><td>{g(pooled, 'ic_imbalance')}</td></tr>
 <tr><th>IC(microprice tilt → fwd 5s ret)</th><td>{g(pooled, 'ic_microprice_tilt')}</td></tr>
 <tr><th>AUC(imbalance → up)</th><td>{g(pooled, 'auc_imbalance')}</td></tr>
 <tr><th>AUC(momentum → up) [baseline]</th><td>{g(pooled, 'auc_momentum_baseline')}</td></tr>
</table>
{ic_img}
</body></html>
"""


def report(paths: Paths) -> str:
    """Runs the report stage; returns the output HTML path as a string."""
    validation = _load_json(paths.out_dir / "validation" / "metrics.json")
    research_metrics = _load_json(paths.out_dir / "research" / "metrics.json")
    html = build_html(validation, research_metrics)
    out = paths.out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


def main(argv: list[str] | None = None) -> int:
    args = stage_parser(__doc__ or "report").parse_args(argv)
    paths = resolve_paths(args)
    out = report(paths)
    print(f"[report] wrote {out} — open it in a browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
