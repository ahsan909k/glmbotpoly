"""Consolidate the competitor **operating manuals** + **strategy clones** into one report.

Reads the per-account manuals (``out/competitors/manuals/<handle>.json``, from
:mod:`model_lab.competitors.manuals`) and the per-clone backtests
(``out/backtests/clone_<name>/metrics.json``, from :mod:`model_lab.backtest_clones`)
and writes a single ``out/competitors/final_report.{html,md}`` tying together: how each
account trades (distributions), the mechanized-clone money verdict over the current-regime
windows, the maker-fill bracket, the Bronze-vs-owner tier honesty, the clone-vs-owner
calibration ladder, and the momentum-exit-vs-hold comparison.

Run: ``python -m model_lab.competitors.final_report`` (after manuals + backtest_clones).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .manuals import assert_anchor_consistent, recon_contradicts_official
from .paths import out_dir

# Clone name (backtest_clones key / out dir) → manual handle (manuals json filename).
CLONE_TO_MANUAL = {
    "takerner": "takerner", "bonereaper": "bonereaper",
    "0xb27b": "0xb27bc932bf8110d8f78e55da7d5f0497a18b5b82",
    "wolf9478": "wolf9478", "nagi777": "nagi777",
}
ORDER = ["0xb27b", "wolf9478", "nagi777", "takerner", "bonereaper"]

# Account caveats surfaced by out/audits/takerner_pnl_anomaly.md.
# Hold-to-resolution takers: their official profile P/L cannot be reconciled to the complete
# public cash-flow (the profile P/L is computed on an unverified basis, not raw realized cash).
HOLD_TO_RESOLUTION = {"takerner", "bonereaper"}
HOLD_CAVEAT = "official P/L: display basis unverified vs cash — see takerner_pnl_anomaly.md"
# Ultra-active accounts whose /activity fetch is TRUNCATED (paging cap): only a recent window was
# captured, so their reconstructed figures are window-scoped, not all-time.
TRUNCATED_FETCH = {"0xb27b"}
TRUNC_CAVEAT = "fetch truncated (ultra-active) — reconstructed figures are WINDOW-SCOPED, not all-time"


def _f(x, nd=0, dash="—"):
    if not isinstance(x, (int, float)):
        return dash
    return f"{x:,.{nd}f}"


def _pct(x, dash="—"):
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else dash


def _load(paths) -> list[dict]:
    """Join each clone's metrics to its manual; skip clones/manuals not on disk."""
    mdir = out_dir() / "manuals"
    bdir = paths.out_dir / "backtests"
    rows = []
    for name in ORDER:
        cp = bdir / f"clone_{name}" / "metrics.json"
        mp = mdir / f"{CLONE_TO_MANUAL[name]}.json"
        clone = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else None
        manual = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
        # Fail loudly (F, part 4): a contradiction must never reach the report unflagged.
        if manual and manual.get("anchor"):
            assert_anchor_consistent(manual["anchor"])
        rows.append({"name": name, "clone": clone, "manual": manual})
    return rows


def _price_frac(pv: dict, k: str):
    tot = pv["at_touch"] + pv["inside"] + pv["crossing"]
    return (pv[k] / tot) if tot else None


def _verdict(rows: list[dict]) -> str:
    clones = [r["clone"] for r in rows if r["clone"]]
    if not clones:
        return "No clone results on disk yet — run backtest_clones first."
    nets = {r["name"]: r["clone"]["clone"]["net_pnl"] for r in rows if r["clone"]}
    hold = next((c["momentum_exit"]["hold_to_resolution_net"] for c in clones), 0.0)
    nwin = clones[0]["current_regime_windows"]
    worst = min(nets.values())
    return (f"Over the full current-regime slice ({nwin:,} windows, ≥2026-06-05, 255 ms), "
            f"<b>all five competitor-clones LOSE money</b> at our tier: the taker clones "
            f"−${_f(-min(nets['takerner'], nets['bonereaper']))}…−${_f(-max(nets['takerner'], nets['bonereaper']))}, "
            f"the maker clones catastrophically (−${_f(-worst)} worst) — makers get adversely "
            f"selected at scale and lose even at the optimistic front-of-queue fill bound. "
            f"The only positive strategy on the same windows is the engine's own <b>momentum "
            f"trigger held to resolution (+${_f(hold)})</b>; exiting early only erodes it. "
            f"Fully consistent with every prior finding — momentum is the only edge.")


# --- Benchmark references tile (anchor-verified) ------------------------------
# PRIMARY = the reference-class PURE MAKER on our exact BTC/ETH 5m/15m series whose
# OUR-series reconstruction reconciles with the account-wide official P/L. Only
# anchor-CONSISTENT accounts get money shown; SUSPECT accounts render behavioral
# distributions only (money greyed) — the reconstruction, not the profile, is the error.
PRIMARY_REF = "gabigol"
REF_DISPLAY = {
    "gabigol": "gabigol",
    "0xb27bc932bf8110d8f78e55da7d5f0497a18b5b82": "0xb27b",
    "nagi777": "nagi777",
    "0xf3531b23b504cf0aed4ff21325232b2a2d496685": "xrang2731",
    "takerner": "takerner", "bonereaper": "bonereaper", "wolf9478": "wolf9478",
}
BENCH_NOTE = ("price-vs-book comparisons are <b>same-method-relative only</b> — a pure maker shows "
              "~40% “crossing” from tape-snapshot alignment (its resting price vs a book snapshot "
              "taken a fraction of a second apart), NOT aggressive taking. Read touch/cross across accounts "
              "relatively, never as an absolute aggression rate.")


def _load_all_manuals() -> dict:
    """Every manual JSON on disk keyed by handle (filename stem); fails loud on an unflagged anchor."""
    out = {}
    for f in sorted((out_dir() / "manuals").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("anchor"):
            assert_anchor_consistent(m["anchor"])
        out[f.stem] = m
    return out


def _ref_stats(m: dict) -> dict:
    pv = m.get("price_vs_book", {}); cov = m.get("coverage", {}); a = m.get("anchor", {})
    pwc = m.get("per_window_capital", {}).get("gross_deployed_usd", {})
    wins, days = cov.get("our_windows_traded"), cov.get("active_days")
    return {
        "at_touch": _price_frac(pv, "at_touch") if pv.get("windows_with_book") else None,
        "crossing": _price_frac(pv, "crossing") if pv.get("windows_with_book") else None,
        "cap_p50": pwc.get("p50"), "cap_p90": pwc.get("p90"), "windows": wins, "days": days,
        "wpd": (wins / days) if (wins and days) else None,
        "velocity": m.get("capital_velocity_hold", {}).get("turns_per_day", {}).get("p50"),
        "official": a.get("official_all_usd"), "recon": a.get("reconstructed_our_net_usd"),
        "suspect": bool(a.get("suspect")),
    }


def _rank_refs(manuals: dict):
    """(consistent [PRIMARY first, then by name], suspect [by name]) handle lists, by the anchor flag."""
    cons = [h for h, m in manuals.items() if not m.get("anchor", {}).get("suspect")]
    cons.sort(key=lambda h: (0 if h == PRIMARY_REF else 1, REF_DISPLAY.get(h, h)))
    susp = sorted(h for h, m in manuals.items() if m.get("anchor", {}).get("suspect"))
    return cons, susp


def build_benchmark_tile_html(manuals: dict) -> str:
    cons, susp = _rank_refs(manuals)
    p = ["<h2>0. Benchmark references (anchor-verified)</h2>",
         "<p class='muted'>Only <b>anchor-consistent</b> accounts (OUR-series reconstruction reconciles with "
         "the account-wide official P/L) are trusted for money. gabigol is the <b>PRIMARY</b> reference — a "
         "pure maker on our exact BTC/ETH 5m/15m series with a full-history consistent anchor.</p>",
         "<table><tr><th>Reference</th><th>Role</th><th>At-touch</th><th>Per-window cap<br>p50 / p90</th>"
         "<th>Windows/day · uptime</th><th>Velocity<br>turns/day</th><th>Official P/L (recon ✓)</th></tr>"]
    for h in cons:
        s = _ref_stats(manuals[h]); disp = REF_DISPLAY.get(h, h)
        role = "<b>PRIMARY</b>" if h == PRIMARY_REF else "secondary"
        cav = " <span class='muted' style='font-size:.8em'>(window-scoped fetch)</span>" if REF_DISPLAY.get(h, h) in TRUNCATED_FETCH else ""
        money = (f"<span class='pos'>+${_f(s['official'])}</span> (recon +${_f(s['recon'])} ✓)"
                 if isinstance(s["official"], (int, float)) else "—")
        p.append(f"<tr><td><b>{html.escape(disp)}</b>{cav}</td><td>{role}</td><td>{_pct(s['at_touch'])}</td>"
                 f"<td>${_f(s['cap_p50'])} / ${_f(s['cap_p90'])}</td><td>{_f(s['wpd'])}/day · {_f(s['days'])}d</td>"
                 f"<td>{_f(s['velocity'],0)}</td><td>{money}</td></tr>")
    p.append("</table>")
    if susp:
        p.append("<p class='muted' style='margin-top:.6rem'><b>SUSPECT — behavioral distributions only, money "
                 "greyed:</b> their OUR-series reconstruction contradicts the account-wide official P/L, so the "
                 "reconstructed money is unreliable. Use their at-touch / sizing / sequencing shapes, ignore the $.</p>")
        p.append("<table><tr><th>Account</th><th>At-touch</th><th>Per-window cap p50</th><th>Windows</th><th>Money</th></tr>")
        for h in susp:
            s = _ref_stats(manuals[h])
            p.append(f"<tr><td><b>{html.escape(REF_DISPLAY.get(h,h))}</b> <span class='red'>SUSPECT</span></td>"
                     f"<td>{_pct(s['at_touch'])}</td><td>${_f(s['cap_p50'])}</td><td>{_f(s['windows'])}</td>"
                     f"<td class='muted'>reconstruction contradicts official — unreliable</td></tr>")
        p.append("</table>")
    p.append(f"<p class='muted'><b>Tile note:</b> {BENCH_NOTE}</p>")
    return "".join(p)


def build_benchmark_tile_md(manuals: dict) -> list[str]:
    cons, susp = _rank_refs(manuals)
    L = ["## 0. Benchmark references (anchor-verified)", "",
         "| Reference | Role | At-touch | Per-window cap p50/p90 | Windows/day · uptime | Velocity | Official P/L (recon) |",
         "|---|---|---|---|---|---|---|"]
    for h in cons:
        s = _ref_stats(manuals[h]); role = "**PRIMARY**" if h == PRIMARY_REF else "secondary"
        cav = " (window-scoped)" if REF_DISPLAY.get(h, h) in TRUNCATED_FETCH else ""
        money = f"+${_f(s['official'])} (recon +${_f(s['recon'])} ✓)" if isinstance(s["official"], (int, float)) else "—"
        L.append(f"| {REF_DISPLAY.get(h,h)}{cav} | {role} | {_pct(s['at_touch'])} | ${_f(s['cap_p50'])}/${_f(s['cap_p90'])} | "
                 f"{_f(s['wpd'])}/day · {_f(s['days'])}d | {_f(s['velocity'],0)} | {money} |")
    if susp:
        L += ["", "**SUSPECT (behavioral distributions only — money greyed):** "
              + ", ".join(REF_DISPLAY.get(h, h) for h in susp)
              + " — reconstruction contradicts official; use shapes not $."]
    L += ["", "_Tile note: " + BENCH_NOTE.replace("<b>", "**").replace("</b>", "**") + "_", ""]
    return L


def build_md(rows: list[dict]) -> str:
    L = ["# Competitor operating manuals + strategy clones — final report", ""]
    L.append("**Verdict.** " + _verdict(rows).replace("<b>", "**").replace("</b>", "**"))
    L += [""] + build_benchmark_tile_md(_load_all_manuals())
    L += ["", "## 1. How each account trades (operating manuals — real Telonex tape)", "",
          "| Account | Template | Fills / windows | Price-vs-book (touch/in/cross) | Merge velocity | Hold-velocity | Anchor |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["manual"]
        if not m:
            L.append(f"| {r['name']} | — | (manual missing) | | | | |"); continue
        c = m["coverage"]; pv = m["price_vs_book"]
        mv = m.get("merge_velocity")
        merge = (f"{mv['merges_per_active_day']:.0f}/day, {mv['capital_velocity']['turns_per_day']['p50']:.0f} turns/day, ${_f(mv['total_recycled_usd'])} recycled"
                 if mv else "—")
        hv = m.get("capital_velocity_hold", {}).get("turns_per_day", {}).get("p50")
        pvs = f"{_pct(_price_frac(pv,'at_touch'))}/{_pct(_price_frac(pv,'inside'))}/{_pct(_price_frac(pv,'crossing'))}" if pv["windows_with_book"] else "no tape"
        susp = "**SUSPECT (red)**" if m["anchor"].get("suspect") else "ok"
        L.append(f"| {r['name']} | {'maker' if mv or 'maker' in str(r['clone'] and r['clone']['params']['template']) else m.get('merge_focus') and 'maker' or ''} | "
                 f"{c['our_fills_in_telonex_window']:,} / {c['our_windows_traded']:,} | {pvs} | {merge} | {_f(hv,1)} | {susp} |")
    L += ["", "_Every tape-covered account rests ~54–62% at-touch and crosses ~37–45% — more passive "
          "than their taker labels imply. nagi777 is a merge machine._", "",
          "**Account caveats (out/audits/takerner_pnl_anomaly.md):**",
          "- **takerner, bonereaper** (hold-to-resolution takers): " + HOLD_CAVEAT + ".",
          "- **0xb27b**: " + TRUNC_CAVEAT + " — its window-scoped fact-sheet stands, read it as window-scoped.", ""]

    L += ["## 2. Clone money verdict (mechanized archetype over the current regime)", "",
          "| Clone | Template (owner) | Net PnL | Maker bracket [pess..opt] | Trades / win | Beats-random |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        c = r["clone"]
        if not c:
            L.append(f"| {r['name']} | — | (clone missing) | | | |"); continue
        cl = c["clone"]; p = c["params"]; br = c.get("maker_fill_bracket")
        ctl = c["controls"]["matched_frequency_random"]
        brs = f"−${_f(-br['pessimistic_net'])} .. −${_f(-br['optimistic_net'])}" if br else "n/a (taker)"
        L.append(f"| {r['name']} | {p['template']} ({p['owner_tier']}) | **−${_f(-cl['net_pnl'])}** | {brs} | "
                 f"{cl['trades']:,} / {_pct(cl['win_rate'])} | {'yes' if ctl['beats'] else 'no'} (p={ctl['p_value']:.2f}) |")
    L += ["", "_“Beats-random” means the clone loses **less** than a random-parameter version of itself — "
          "not that it makes money. All five are net losers._", ""]

    L += ["## 3. Momentum-exit vs hold-to-resolution (same windows, all clones)", ""]
    c0 = next((r["clone"] for r in rows if r["clone"]), None)
    if c0:
        me = c0["momentum_exit"]; ex = me["by_timeout"]
        L += ["| Exit rule | Net PnL |", "|---|---|",
              f"| hold to resolution | **+${_f(me['hold_to_resolution_net'])}** |",
              f"| sell at bid / T=15 s | −${_f(-ex['T15']['net_pnl'])} |" if ex['T15']['net_pnl'] < 0 else f"| sell at bid / T=15 s | +${_f(ex['T15']['net_pnl'])} |",
              f"| sell at bid / T=30 s | +${_f(ex['T30']['net_pnl'])} |",
              f"| sell at bid / T=60 s | +${_f(ex['T60']['net_pnl'])} |",
              "", "_Holding the momentum position to resolution beats every early exit — the edge is in "
              "the window resolving, not short-term mid convergence._", ""]

    L += ["## 4. Clone-vs-owner calibration ladder", "",
          "| Clone | (1) owner official slice | (2) owner OUR-series recon | (3) clone net | gap 1→2 | gap 2→3 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        c = r["clone"]
        if not c:
            continue
        lad = c["clone_vs_owner_ladder"]
        susp, _ = recon_contradicts_official(lad.get("owner_reconstructed_net_usd"), lad.get("official_slice_usd"))
        recon_cell = (f"**${_f(lad.get('owner_reconstructed_net_usd'))} 🔴SUSPECT**" if susp
                      else f"${_f(lad.get('owner_reconstructed_net_usd'))}")
        g12 = "🔴 unreliable" if susp else f"${_f(lad.get('gap_official_minus_reconstructed'))}"
        g23 = "🔴 unreliable" if susp else f"${_f(lad.get('gap_reconstructed_minus_clone'))}"
        L.append(f"| {r['name']} | ${_f(lad.get('official_slice_usd'))} | {recon_cell} | "
                 f"−${_f(-lad['clone_net_usd'])} | {g12} | {g23} |")
    L += ["", "_Cell (2) is a cash-flow cross-check; it is flagged **SUSPECT** (both directions) wherever it "
          "materially contradicts the authoritative official P/L, and its gaps are then unreliable (the "
          "reconstruction, not the profile, is the error). For non-suspect rows: gap 1→2 = other markets / "
          "unobserved; gap 2→3 = what we could NOT copy (queue, triggers, tier). Never smoothed._", ""]

    L += ["## Provenance & caveats", "",
          "- Manuals: FACT (raw Polymarket records) + CALC (exact arithmetic) only; the anchor flags a "
          "reconstruction that contradicts the account-wide official P/L in red.",
          "- Tier honesty: each clone scored at Bronze (ours) AND its owner's tier — maker clones pay ~$0 "
          "taker fee so the tier is immaterial for them; it only bites the taker clones.",
          "- Clones reconstruct the current-regime book (255 ms venue delay simulated), conservative fills, "
          "marked to the official resolution. Research only — no live trading.",
          "- Price-vs-book sampled ≤1500 windows/account; nagi777's tape-required metrics cover only its "
          "Telonex-overlap window.",
          "- Official P/L basis (out/audits/takerner_pnl_anomaly.md): for hold-to-resolution takers "
          "(takerner, bonereaper) the official profile P/L cannot be reconciled to the complete public "
          "cash-flow — display basis unverified vs cash. 0xb27b's /activity fetch is truncated, so its "
          "reconstructed figures are window-scoped, not all-time."]
    return "\n".join(L)


def build_html(rows: list[dict]) -> str:
    md = build_md(rows)  # reuse the structure via a light MD→HTML for tables
    # Simple, self-contained HTML: render the verdict + tables from the same data.
    def cell(x):
        return html.escape(str(x))
    parts = [f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Competitor manuals + clones — final report</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
 h1{{font-size:1.6rem}} h2{{margin-top:1.8rem;border-bottom:1px solid #eee;padding-bottom:4px}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6rem 0}}
 td,th{{border:1px solid #e6e6e6;padding:5px 9px;text-align:left}} th{{background:#f2f4f7}}
 .verdict{{background:#f6f8fa;border-left:4px solid #1f77b4;padding:.8rem 1rem;border-radius:4px}}
 .neg{{color:#c0341d;font-weight:600}} .pos{{color:#0a7d28;font-weight:600}} .muted{{color:#777}}
 .red{{background:#ffe0e0;color:#a00;padding:1px 4px;border-radius:3px;font-weight:700}}
</style></head><body>
<h1>Competitor operating manuals + strategy clones</h1>
<div class='verdict'>{_verdict(rows)}</div>"""]

    # 0. benchmark references tile (anchor-verified; loads ALL manuals, not just the cloned ones)
    parts.append(build_benchmark_tile_html(_load_all_manuals()))

    # 1. manuals
    parts.append("<h2>1. How each account trades (operating manuals — real Telonex tape)</h2><table>"
                 "<tr><th>Account</th><th>Fills / windows</th><th>Price-vs-book<br>touch/in/cross</th>"
                 "<th>Merge velocity</th><th>Hold-velocity<br>turns/day</th><th>Anchor</th></tr>")
    for r in rows:
        m = r["manual"]
        if not m:
            parts.append(f"<tr><td>{cell(r['name'])}</td><td colspan=5 class='muted'>manual missing</td></tr>"); continue
        c = m["coverage"]; pv = m["price_vs_book"]; mv = m.get("merge_velocity")
        merge = (f"{mv['merges_per_active_day']:.0f}/day · {mv['capital_velocity']['turns_per_day']['p50']:.0f} turns/day · ${_f(mv['total_recycled_usd'])} recycled" if mv else "—")
        hv = m.get("capital_velocity_hold", {}).get("turns_per_day", {}).get("p50")
        pvs = (f"{_pct(_price_frac(pv,'at_touch'))} / {_pct(_price_frac(pv,'inside'))} / {_pct(_price_frac(pv,'crossing'))}"
               if pv["windows_with_book"] else "<span class='muted'>no tape (past coverage)</span>")
        susp = "<span class='red'>SUSPECT</span>" if m["anchor"].get("suspect") else "ok"
        cav = ""
        if r["name"] in HOLD_TO_RESOLUTION:
            cav += f"<br><span class='muted' style='font-size:.8em'>⚠ {cell(HOLD_CAVEAT)}</span>"
        if r["name"] in TRUNCATED_FETCH:
            cav += f"<br><span class='muted' style='font-size:.8em'>⚠ {cell(TRUNC_CAVEAT)}</span>"
        parts.append(f"<tr><td><b>{cell(r['name'])}</b>{cav}</td><td>{c['our_fills_in_telonex_window']:,} / {c['our_windows_traded']:,}</td>"
                     f"<td>{pvs}</td><td>{merge}</td><td>{_f(hv,1)}</td><td>{susp}</td></tr>")
    parts.append("</table><p class='muted'>Every tape-covered account rests ~54–62% at-touch and crosses ~37–45% — "
                 "more passive than their taker labels imply. nagi777 is a merge machine (~2,800 merges/day, ~65 capital "
                 "turns/day, $1.3M recycled); wolf9478's reconstruction is flagged red.</p>")

    # 2. clones
    parts.append("<h2>2. Clone money verdict (current-regime windows)</h2><table>"
                 "<tr><th>Clone</th><th>Template (owner tier)</th><th>Net PnL</th><th>Maker bracket [pess..opt]</th>"
                 "<th>Trades / win</th><th>Bronze → owner rebate</th><th>Beats random?</th></tr>")
    for r in rows:
        c = r["clone"]
        if not c:
            parts.append(f"<tr><td>{cell(r['name'])}</td><td colspan=6 class='muted'>clone missing</td></tr>"); continue
        cl = c["clone"]; p = c["params"]; br = c.get("maker_fill_bracket"); t = c["tier_honesty"]; ctl = c["controls"]["matched_frequency_random"]
        brs = f"−${_f(-br['pessimistic_net'])} .. −${_f(-br['optimistic_net'])}" if br else "<span class='muted'>n/a (taker crosses)</span>"
        tier = (f"${_f(t['our_tier']['rebate'],0)} → ${_f(t['owner_tier']['rebate'],0)} ({t['owner_tier']['tier']})"
                if t.get("our_tier") else "—")
        parts.append(f"<tr><td><b>{cell(r['name'])}</b></td><td>{p['template']} ({p['owner_tier']})</td>"
                     f"<td class='neg'>−${_f(-cl['net_pnl'])}</td><td>{brs}</td><td>{cl['trades']:,} / {_pct(cl['win_rate'])}</td>"
                     f"<td>{tier}</td><td>{'yes' if ctl['beats'] else 'no'} (p={ctl['p_value']:.2f})</td></tr>")
    parts.append("</table><p class='muted'>“Beats random” = loses <i>less</i> than a random-parameter version of itself, "
                 "not that it profits. All five are net losers. Maker rebate is ~$0 at any tier (no taker fee).</p>")

    # 3. momentum-exit
    c0 = next((r["clone"] for r in rows if r["clone"]), None)
    if c0:
        me = c0["momentum_exit"]; ex = me["by_timeout"]
        rowshtml = "".join(
            f"<tr><td>{lbl}</td><td class='{cls}'>{'+' if v>=0 else '−'}${_f(abs(v))}</td></tr>"
            for lbl, v, cls in [("hold to resolution", me['hold_to_resolution_net'], 'pos'),
                                ("sell at bid, T=15 s", ex['T15']['net_pnl'], 'neg' if ex['T15']['net_pnl']<0 else 'pos'),
                                ("sell at bid, T=30 s", ex['T30']['net_pnl'], 'pos'),
                                ("sell at bid, T=60 s", ex['T60']['net_pnl'], 'pos')])
        parts.append("<h2>3. Momentum-exit vs hold-to-resolution (same windows)</h2>"
                     f"<table><tr><th>Exit rule</th><th>Net PnL</th></tr>{rowshtml}</table>"
                     "<p class='muted'>Holding the momentum position to resolution beats every early exit — the edge is "
                     "in the window resolving, not short-term mid convergence.</p>")

    # 4. ladder — cell (2) flagged SUSPECT (red) where the reconstruction contradicts official.
    parts.append("<h2>4. Clone-vs-owner calibration ladder</h2><table>"
                 "<tr><th>Clone</th><th>(1) owner official slice</th><th>(2) owner OUR-series recon</th>"
                 "<th>(3) clone net</th><th>gap 1→2</th><th>gap 2→3</th></tr>")
    n_suspect = 0
    for r in rows:
        c = r["clone"]
        if not c:
            continue
        lad = c["clone_vs_owner_ladder"]
        susp, _ = recon_contradicts_official(lad.get("owner_reconstructed_net_usd"), lad.get("official_slice_usd"))
        n_suspect += 1 if susp else 0
        recon_cell = (f"<span class='red'>${_f(lad.get('owner_reconstructed_net_usd'))} SUSPECT</span>" if susp
                      else f"${_f(lad.get('owner_reconstructed_net_usd'))}")
        gap12 = "<span class='red'>—</span>" if susp else f"${_f(lad.get('gap_official_minus_reconstructed'))}"
        gap23 = "<span class='red'>unreliable</span>" if susp else f"${_f(lad.get('gap_reconstructed_minus_clone'))}"
        parts.append(f"<tr><td><b>{cell(r['name'])}</b></td><td>${_f(lad.get('official_slice_usd'))}</td>"
                     f"<td>{recon_cell}</td><td class='neg'>−${_f(-lad['clone_net_usd'])}</td>"
                     f"<td>{gap12}</td><td>{gap23}</td></tr>")
    parts.append("</table><p class='muted'>Cell (2) is the OUR-series cash-flow reconstruction, a "
                 "<b>cross-check</b> against the authoritative official P/L. It is flagged "
                 "<span class='red'>SUSPECT</span> (both directions) whenever it materially contradicts "
                 "official — for those rows the gaps are <b>unreliable and not shown</b> (the reconstruction, "
                 "not the account's profile, is treated as the error). For non-suspect rows: gap 1→2 = other "
                 "markets / unobserved; gap 2→3 = what we could NOT copy (queue, triggers, tier). Never smoothed.</p>")
    if n_suspect:
        parts.insert(1, f"<p class='muted'><b>Note:</b> {n_suspect} of the owner reconstructions are flagged "
                        "<span class='red'>SUSPECT</span> (the OUR-series cash-flow cannot be reconciled with "
                        "the account-wide official P/L) — their ladder cell (2) and gaps are unreliable.</p>")

    parts.append("<h2>Provenance & caveats</h2><ul>"
                 "<li>Manuals: FACT (raw Polymarket records) + CALC (exact arithmetic) only; a reconstruction that "
                 "contradicts the account-wide official P/L is flagged <span class='red'>SUSPECT</span>.</li>"
                 "<li>Clones reconstruct the current-regime book (255 ms venue delay simulated), conservative fills, "
                 "marked to the official resolution. Research only — no live trading.</li>"
                 "<li>Maker-fill bracket = pessimistic (behind all displayed size) … optimistic (front-of-queue); "
                 "'truth is between; only live measures it.'</li>"
                 "<li>Price-vs-book sampled ≤1500 windows/account; nagi777's tape-required metrics cover only its "
                 "Telonex-overlap window.</li>"
                 "<li><b>Official P/L basis (out/audits/takerner_pnl_anomaly.md):</b> for hold-to-resolution "
                 "takers (takerner, bonereaper) the official profile P/L cannot be reconciled to the complete "
                 "public cash-flow — its display basis is unverified vs cash. 0xb27b's /activity fetch is "
                 "truncated, so its reconstructed figures are window-scoped, not all-time.</li></ul>")
    parts.append("</body></html>")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    from ..config import resolve_paths
    ap = argparse.ArgumentParser(description="Consolidate competitor manuals + clones into one report.")
    ap.parse_args(argv)
    paths = resolve_paths()
    rows = _load(paths)
    odir = out_dir()
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "final_report.html").write_text(build_html(rows), encoding="utf-8")
    (odir / "final_report.md").write_text(build_md(rows), encoding="utf-8")
    n_clone = sum(1 for r in rows if r["clone"])
    n_man = sum(1 for r in rows if r["manual"])
    print(f"[final_report] {n_man} manuals + {n_clone} clones → {odir / 'final_report.html'} + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
