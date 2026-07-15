"""Render one **operating manual** HTML per competitor account from ``manuals.build_manual``.

Structure enforces the provenance discipline (F): the **manual body** (the behavioral
distributions) carries FACT + CALC only; everything that needs an assumption (estimated
fees, inferred triggers, queue position) is quarantined to a separate *Inference* section,
and the official-P/L *Anchor* ladder shows FACT (official) alongside the CALC/EST
reconstruction — flagged **red** when the reconstruction contradicts official. A 10-window
*Hand-trace* appendix lets any classification be re-derived by eye.

Reads ``out/competitors/manuals/<handle>.json`` (written by ``manuals``). Run:
``python -m model_lab.competitors.manuals_report``.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .paths import out_dir
from .report import CALC, CSS, EST, FACT, UNV

_BLOCKS = "▁▂▃▄▅▆▇█"


def _spark(counts) -> str:
    """A unicode sparkline for a histogram's integer counts."""
    if not counts:
        return ""
    mx = max(counts) or 1
    return "".join(_BLOCKS[min(7, int(7 * c / mx))] for c in counts)


def _f(x, nd=2, dash="—"):
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def _pct(x, dash="—"):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else dash


def _dist_row(label: str, dist: dict, nd: int = 3) -> str:
    """One table row for a distribution: p10 / p25 / p50 / p75 / p90 + n + sparkline."""
    spark = _spark(dist.get("hist_counts")) if dist.get("hist_counts") else ""
    cells = [html.escape(label), str(dist.get("n", 0))]
    for q in ("p10", "p25", "p50", "p75", "p90"):
        cells.append(_f(dist.get(q), nd))
    cells.append(f"<span class='mono'>{spark}</span>")
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _dist_table(rows_html: str) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                   ["metric", "n", "p10", "p25", "p50", "p75", "p90", "shape"])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table>"


def _timing_section(m: dict) -> str:
    et = m["entry_timing"]
    rows = _dist_row("entry frac — maker", et["maker"]) + _dist_row("entry frac — taker", et["taker"])
    mt = m.get("merge_timing", {})
    if mt.get("merge_frac_of_window", {}).get("n"):
        rows += _dist_row("merge frac", mt["merge_frac_of_window"])
    return f"<h3>Entry / merge timing within the window {CALC}</h3>{_dist_table(rows)}"


def _price_section(m: dict) -> str:
    pv = m["price_vs_book"]
    frac_line = (f"at-touch <b>{_pct(pv['at_touch_frac'])}</b> · inside <b>{_pct(pv['inside_frac'])}</b> · "
                 f"crossing <b>{_pct(pv['crossing_frac'])}</b> "
                 f"<span class='small'>({pv['classified_fills']:,} fills; "
                 f"{pv['windows_with_book']:,} windows with book, {pv['windows_book_missing']:,} missing)</span>")
    rows = _dist_row("signed dist to touch (ticks)", pv["signed_dist_to_touch_ticks"], nd=2)
    return (f"<h3>Price level vs the book at fill {CALC} <span class='small'>(tape-required)</span></h3>"
            f"<div>{frac_line}</div>{_dist_table(rows)}")


def _sequencing_section(m: dict) -> str:
    s = m["sequencing"]
    line = (f"simultaneous <b>{_pct(s['simultaneous_frac'])}</b> of two-sided windows "
            f"<span class='small'>({s['simultaneous_windows']} simultaneous / {s['alternating_windows']} "
            f"alternating / {s['single_sided_windows']} single-sided)</span>")
    rows = (_dist_row("first-leg gap (s)", s["first_leg_gap_secs"], nd=1)
            + _dist_row("alternation ratio", s["alternation_ratio"])
            + _dist_row("same-side run len", s["same_side_run_len"], nd=1))
    return f"<h3>Sequencing — simultaneous vs alternating {CALC}</h3><div>{line}</div>{_dist_table(rows)}"


def _inventory_section(m: dict) -> str:
    iv = m["inventory_trajectory"]
    sz = m["size_ramping"]
    cap = m["per_window_capital"]
    rows = (_dist_row("peak |excess| shares", iv["peak_excess_shares"], nd=1)
            + _dist_row("end skew", iv["end_skew"])
            + _dist_row("per-fill notional $", sz["per_fill_notional_usd"], nd=2)
            + _dist_row("ramp slope (sh/fill)", sz["ramp_slope_shares_per_fill"], nd=2)
            + _dist_row("gross deployed / window $", cap["gross_deployed_usd"], nd=2))
    return f"<h3>Inventory, size ramp &amp; per-window capital {CALC}</h3>{_dist_table(rows)}"


def _merge_velocity_section(m: dict) -> str:
    if not m.get("merge_focus") or "merge_velocity" not in m:
        return ""
    mv = m["merge_velocity"]
    isa = mv["inventory_state_at_merge"]
    line = (f"<b>{mv['merge_events']:,}</b> merges · "
            f"<b>{_f(mv['merges_per_active_day'], 2)}</b>/active-day · "
            f"<b>{_f(mv['merges_per_traded_window'], 3)}</b>/traded-window · "
            f"matched-at-merge <b>{_pct(isa['matched_frac'])}</b> · "
            f"total recycled <b>${_f(mv['total_recycled_usd'], 0)}</b>")
    rows = (_dist_row("skew at merge", isa["skew"])
            + _dist_row("capital recycled / merge $", mv["capital_recycled_per_merge_usd"], nd=2)
            + _dist_row("turns / day (merge-recycle)", mv["capital_velocity"]["turns_per_day"], nd=2))
    return (f"<h3>Merge mechanics &amp; capital velocity {CALC} "
            f"<span class='small'>(merge-heavy account)</span></h3><div>{line}</div>{_dist_table(rows)}")


def _anchor_section(m: dict) -> str:
    """The official-P/L anchor ladder — FACT official alongside the CALC/EST reconstruction,
    red when the reconstruction contradicts official (OUR reconstruction treated as suspect)."""
    a = m["anchor"]
    lad = a.get("slice_ladder", {})
    if not a.get("official_available"):
        body = f"<div class='small'>{a.get('note', 'no official P/L returned by the endpoint')}</div>"
    else:
        suspect = a.get("suspect")
        banner = ("<div class='reddisc'><b>⚠ OUR RECONSTRUCTION SUSPECT</b> — the OUR-series "
                  "reconstruction contradicts the official Polymarket P/L; treat OUR reconstruction "
                  f"as the error, not their profile. {html.escape('; '.join(a.get('suspect_reasons', [])))}"
                  "</div>") if suspect else ""
        cls = "reddisc-val" if suspect else "pos"
        body = f"""{banner}
        <div><span class='k'>Official P/L (all-time) {FACT}</span> ${_f(a.get('official_all_usd'), 0)}
             <span class='small'>(as of {html.escape(str(a.get('official_as_of') or '')[:10])})</span></div>
        <div><span class='k'>Official over captured window {FACT}</span>
             <span class='{cls}'>${_f(a.get('official_window_usd'), 0)}</span>
             <span class='small'>(account-wide, all markets; from {html.escape(str(a.get('window_from') or '')[:10])})</span></div>
        <div><span class='k'>OUR-series reconstruction {CALC}{EST}</span>
             ${_f(a.get('reconstructed_our_net_usd'), 0)}
             <span class='small'>(cash-flow ledger; fees {EST} ${_f(a.get('reconstructed_est_fees_usd'), 0)})</span></div>"""
    # The post-Jun-5 clone-vs-owner ladder inputs (the clone report appends (3)).
    ladder = f"""
    <h4>Post-Jun-5 slice — clone-vs-owner calibration inputs</h4>
    <table><thead><tr><th>#</th><th>quantity</th><th>value</th><th>provenance</th></tr></thead><tbody>
      <tr><td>1</td><td>account-wide official slice</td><td>${_f(lad.get('official_account_wide_slice_usd'), 0)}</td><td>{FACT}</td></tr>
      <tr><td>2</td><td>owner's OUR-4-series reconstructed net</td><td>${_f(lad.get('owner_our_series_reconstructed_net_usd'), 0)}</td><td>{CALC}{EST}</td></tr>
      <tr><td>3</td><td>clone backtest net (appended by the clone report)</td><td>—</td><td>{EST}</td></tr>
    </tbody></table>
    <div class='small'>Gap 1→2 = other markets / unobserved; gap 2→3 = what we could NOT copy
    (queue position, unobserved triggers, tier).</div>"""
    return f"<h3>Anchor — reconciliation vs official Polymarket P/L</h3>{body}{ladder}"


def _inference_section(m: dict) -> str:
    items = "".join(f"<li>{html.escape(x)}</li>" for x in m.get("not_observable", []))
    return (f"<h3>Inference &amp; not-observable {EST} {UNV}</h3>"
            "<div class='small'>These are NOT in the manual body above — they need an assumption "
            "or cannot be established from the fill records, and are stated here, never asserted:</div>"
            f"<ul>{items}</ul>")


def _handtrace_section(m: dict) -> str:
    traces = m.get("hand_trace", [])
    if not traces:
        return ""
    blocks = []
    for t in traces:
        rows = []
        for r in t["fills"]:
            if r.get("book") == "unknown":
                book = "unknown"
            else:
                book = f"{_f(r.get('book_bid'), 3)} / {_f(r.get('book_ask'), 3)}"
            rows.append(f"<tr><td>{html.escape(r['fill_ts'][11:19])}</td><td>{r['side']}</td>"
                        f"<td>{r['outcome']}</td><td>{_f(r['price'], 4)}</td><td>{_f(r['size'], 2)}</td>"
                        f"<td>{book}</td><td>{r['class']}</td></tr>")
        blocks.append(
            f"<h4>{html.escape(t['slug'])} <span class='small'>({t['open_utc'][:19]})</span></h4>"
            "<table><thead><tr><th>fill ts</th><th>side</th><th>outcome</th><th>price</th>"
            "<th>size</th><th>book bid/ask</th><th>class</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")
    return (f"<h3>Appendix — 10-window hand-trace {FACT}{CALC}</h3>"
            "<div class='small'>Each fill next to the Telonex book as-of it — any classification "
            "above is re-derivable by eye. 'unknown' = no book established from records.</div>"
            + "".join(blocks))


def manual_body(m: dict) -> str:
    """The behavioral distributions — FACT + CALC ONLY (the provenance-lint target).

    No EST/UNV span may appear here; estimates/assumptions live in the Inference + Anchor
    sections rendered separately by :func:`build_manual_html`."""
    return (_timing_section(m) + _price_section(m) + _sequencing_section(m)
            + _inventory_section(m) + _merge_velocity_section(m))


def build_manual_html(m: dict) -> str:
    prof = m.get("profile", {})
    cov = m["coverage"]
    header = (f"<h1>Operating manual — {html.escape(m['handle'])}</h1>"
              f"<div class='sub'><span class='mono'>{html.escape(m['address'])}</span> · "
              f"{html.escape(str(prof.get('takerTierName') or '?'))} tier {FACT} · "
              f"{cov['our_fills_in_telonex_window']:,} our-series fills / {cov['our_windows_traded']:,} windows "
              f"over Telonex {cov['telonex_from']}→{cov['telonex_to']} {CALC}</div>")
    body = f"<div class='manual-body'>{manual_body(m)}</div>"
    extra_css = """
    .manual-body{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:8px 16px;margin:12px 0}
    .reddisc{background:#ffe0e0;border-left:4px solid #c0341d;color:#7a1005;padding:10px 14px;margin:8px 0;border-radius:4px}
    .reddisc-val{color:#c0341d;font-weight:700}
    h3{margin-top:22px;border-bottom:1px solid #eee;padding-bottom:3px} h4{margin:14px 0 4px}
    """
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Operating manual — {html.escape(m['handle'])}</title><style>{CSS}{extra_css}</style></head><body>
{header}
<div class='legend'><b>Provenance.</b> The manual body below is {FACT} raw records + {CALC} exact
arithmetic on them ONLY. {EST} estimates / {UNV} assumptions (fees, inferred triggers, queue
position) are quarantined to the Inference section and never mixed into the behavioral distributions.</div>
{body}
<div class='card'>{_anchor_section(m)}</div>
<div class='card'>{_inference_section(m)}</div>
<div class='card'>{_handtrace_section(m)}</div>
</body></html>"""


def render_manual(manual_path: Path) -> Path:
    m = json.loads(Path(manual_path).read_text(encoding="utf-8"))
    out = Path(manual_path).with_suffix(".html")
    out.write_text(build_manual_html(m), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render competitor operating-manual HTML.")
    ap.add_argument("--only", nargs="*", default=None, help="handles (default: all manuals on disk)")
    ap.add_argument("--dir", default=None, help="manuals dir (default out/competitors/manuals)")
    args = ap.parse_args(argv)
    mdir = Path(args.dir) if args.dir else out_dir() / "manuals"
    want = {x.lower() for x in args.only} if args.only else None
    n = 0
    for p in sorted(mdir.glob("*.json")):
        if want and p.stem.lower() not in want:
            continue
        out = render_manual(p)
        print(f"[manuals_report] wrote {out}")
        n += 1
    if n == 0:
        print(f"[manuals_report] no manuals found in {mdir} — run `python -m model_lab.competitors.manuals` first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
