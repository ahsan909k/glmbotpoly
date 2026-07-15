"""Render the competitor-analysis report (``out/competitors/report.html``).

One row per account — archetype, key inferred parameters, PnL/day — plus a
comparison row for OUR bot's current config, and a data-driven section on the
parameter choices where the profitable accounts consistently differ from ours.

Honesty is enforced structurally: every account carries its sample size, low-sample
accounts are flagged and never given a firm archetype, and PnL is labelled a
reconstruction with its uncertainty. Run: ``python -m model_lab.competitors.report``.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .paths import competitors_dir, out_dir

# --- OUR bot's current configuration (config/default.toml [engine.defaults] etc.) ---
# Kept as a comparison row; values are the committed defaults (see CLAUDE.md §8).
OUR_BOT = {
    "handle": "OUR BOT (current config)",
    "archetype": "Two-sided post-only market-maker + fee-aware taker modules",
    "series_focus": "BTC/ETH × 5m & 15m (edge concentrated in BTC-5m)",
    "taker_frac": "low — taker gated to $10/window budget; core is maker",
    "two_sided": "yes — two BUY ladders (Up + Down), pair-cost ≤ 0.98",
    "pair_cost_rule": 0.98,
    "usd_per_window": "~$25 max worst-case loss/window is the binding cap "
                      "(touch 10sh + 3×10 ladder ≈ tens of $/side)",
    "merge_hold": "MERGES matched pairs (merge_min_pairs=25) to recycle capital",
    "reaction": "cancel-first repricing: reprice θ=0.005, cancel-market θ=0.02; "
                "momentum FAK taker on >3σ moves; late-window taker τ≤30s, p≥0.97",
    "latency": "~250 ms venue taker delay + ~5–200 ms network (VPS vs home)",
    "risk": "max loss $25/window, daily stop $200, max open notional $1000",
    "rebate_tier": "None/Bronze — maker-first, taker gated to $10/window ⇒ 30-day "
                   "taker weighted volume stays below Bronze $2k, so ~0–3% fee rebate",
    "official_pnl": "paper (no on-chain P/L) — profit must come from maker spread "
                    "capture (pair-cost < 1), NOT a rebate cushion we don't have",
}

MIN_SAMPLE = 200  # our-series trades below this -> classification tentative

# Provenance badges — every figure is tagged so fact is never confused with estimate:
#   FACT — straight from Polymarket (the official P/L endpoint, Gamma profile/tier,
#          the raw trade/redeem/merge records, the takerOnly maker/taker labels) or docs.
#   CALC — an EXACT deterministic computation on that raw data (pair-cost, series
#          volumes/coverage, $/window, trades/day, uptime) — not a guess, just arithmetic.
#   EST  — involves an estimate or model, NOT a Polymarket figure (taker fees via the
#          documented formula, rebate income, the cash-flow reconstruction, trades-only,
#          the profit-source decomposition + residual).
#   UNVERIFIED — a genuine assumption we cannot confirm (flagged, never asserted).
FACT = "<span class='fact' title='Straight from Polymarket API/docs'>FACT</span>"
CALC = "<span class='calc' title='Exact arithmetic on Polymarket data'>CALC</span>"
EST = "<span class='est' title='Estimate/model — not a Polymarket figure'>EST</span>"
UNV = "<span class='unv' title='Unverified assumption — flagged, not asserted'>UNVERIFIED</span>"


def _f(x, nd=2, dash="—"):
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def _pct(x, dash="—"):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else dash


def derive_archetype(a: dict) -> dict:
    """Assign an archetype from the measured signatures, with an honesty gate."""
    tot = a.get("totals", {})
    our_trades = tot.get("our_series_trades", 0)
    mt = a.get("maker_taker", {})
    pd_ = a.get("pair_discipline", {})
    mh = a.get("merge_hold", {})
    taker = mt.get("our_taker_frac_count")
    two_sided = pd_.get("two_sided_frac")

    if our_trades < MIN_SAMPLE:
        return {"archetype": "Unclear — insufficient our-series activity",
                "confidence": "low",
                "rationale": f"only {our_trades} our-series trades (< {MIN_SAMPLE})"}

    maker = (1 - taker) if isinstance(taker, (int, float)) else None
    bits = []
    if isinstance(taker, (int, float)):
        bits.append(f"{taker*100:.0f}% taker")
    if isinstance(two_sided, (int, float)):
        bits.append(f"{two_sided*100:.0f}% two-sided windows")
    bits.append(mh.get("behavior", ""))

    if maker is not None and maker >= 0.6 and (two_sided or 0) >= 0.5:
        arch = "Market-maker (pair-builder)"
    elif isinstance(taker, (int, float)) and taker >= 0.7 and (two_sided or 0) < 0.4:
        arch = "Directional / momentum taker"
    elif isinstance(taker, (int, float)) and taker >= 0.7:
        arch = "Taker pair-builder (two-sided taker)"
    elif isinstance(taker, (int, float)) and 0.3 <= taker < 0.7:
        arch = "Hybrid maker/taker"
    else:
        arch = "Market-maker (pair-builder)"
    return {"archetype": arch, "confidence": "measured", "rationale": "; ".join(b for b in bits if b)}


def _account_row(a: dict, tape: dict | None = None) -> dict:
    """Compact per-account values for the summary table."""
    arch = derive_archetype(a)
    cov = a.get("coverage", {})
    mt = a.get("maker_taker", {})
    pd_ = a.get("pair_discipline", {})
    sc = a.get("size_cadence", {})
    pnl = a.get("pnl", {})
    prof = a.get("profile", {})
    tp = (tape or {}).get(a["address"], {}) if tape else {}
    rs = tp.get("reaction_speed", {}) or {}
    mtc = tp.get("maker_taker_corroboration", {}) or {}
    return {
        "react_gt5": rs.get("frac_with_gt5bps_move_3s"),
        "react_move_med": (rs.get("pre_fill_move_bps_3s") or {}).get("median"),
        "react_n": rs.get("taker_fills_evaluated"),
        "tape_agree": mtc.get("tape_vs_api_agreement"),
        "tape_agree_n": mtc.get("unambiguously_matched"),
        "handle": a["handle"],
        "address": a["address"],
        "tier": f"{prof.get('takerTierName','?')} (30d taker ${_f(prof.get('weightedVolume'),0)})",
        "tier_name": prof.get("takerTierName"),
        "archetype": arch["archetype"],
        "confidence": arch["confidence"],
        "rationale": arch["rationale"],
        "our_trades": a.get("totals", {}).get("our_series_trades", 0),
        "coverage_vol": cov.get("our_series_trade_frac_volume"),
        "our_vol": cov.get("our_series_volume_usd"),
        "taker_frac": mt.get("our_taker_frac_count"),
        "two_sided": pd_.get("two_sided_frac"),
        "pair_cost": (pd_.get("pair_cost") or {}).get("median"),
        "usd_win": (sc.get("usd_per_window") or {}).get("median"),
        "trades_day": sc.get("our_trades_per_active_day"),
        "merge_hold": a.get("merge_hold", {}).get("behavior"),
        "pnl_day": pnl.get("pnl_per_active_day_usd"),
        "realized": pnl.get("realized_cashflow_usd"),
        "fees": pnl.get("est_taker_fees_paid_usd"),
        "drawdown": pnl.get("worst_drawdown_usd"),
        # official (authoritative) + cross-checks + rebate + decomposition
        "official_all": pnl.get("official_all_usd"),
        "official_1m": pnl.get("official_1m_usd"),
        "official_1d": pnl.get("official_1d_usd"),
        "official_available": pnl.get("official_pnl_available", False),
        "official_ambiguous": pnl.get("official_ambiguous", False),
        "trades_only": pnl.get("trades_only_usd"),
        "settlement_gap": pnl.get("settlement_gap_usd"),
        "hold_frac": pnl.get("hold_to_resolution_frac"),
        "rebate_est": pnl.get("est_taker_rebate_usd"),
        "rebate_pct": (pnl.get("rebate") or {}).get("rebate_pct"),
        "rebate_load_bearing": pnl.get("rebate_load_bearing", False),
        "decomp": pnl.get("decomposition", {}),
        "usd_vs_cap": sc.get("usd_per_window_vs_our_cap_x"),
        "uptime": sc.get("uptime", {}),
        "full_series_mix": cov.get("full_series_volume", {}),
        "non_our_series": cov.get("non_our_series", []),
        "active_days": a.get("span", {}).get("active_days"),
        "truncated": a.get("span", {}).get("activity_truncated", False),
        "span_first": (a.get("span", {}).get("first") or "")[:10],
        "span_last": (a.get("span", {}).get("last") or "")[:10],
        "series_mix": cov.get("by_series_volume", {}),
    }


def _med(rows, key):
    vals = sorted(r[key] for r in rows if isinstance(r.get(key), (int, float)))
    return vals[len(vals) // 2] if vals else None


def _differences_section(rows: list[dict]) -> str:
    """Ranked by OFFICIAL PnL: where do the winners differ from our bot — and, crucially,
    what is actually copyable at our (None/low) rebate tier?"""
    sized = [r for r in rows if r["our_trades"] >= MIN_SAMPLE
             and isinstance(r.get("official_all"), (int, float))]
    if not sized:
        return "<p>No sufficiently-sampled account with an official PnL to compare against.</p>"
    winners = [r for r in sized if r["official_all"] > 0]
    losers = [r for r in sized if r["official_all"] <= 0]

    pb = lambda rs: [r for r in rs if isinstance(r["two_sided"], (int, float)) and r["two_sided"] >= 0.5]
    over1_win = sorted((r for r in pb(winners) if isinstance(r["pair_cost"], (int, float)) and r["pair_cost"] > 1.0),
                       key=lambda r: -r["official_all"])
    under1_win = sorted((r for r in pb(winners) if isinstance(r["pair_cost"], (int, float)) and r["pair_cost"] < 1.0),
                        key=lambda r: -r["official_all"])
    bearers = sorted((r for r in winners if r.get("rebate_load_bearing")),
                     key=lambda r: -(r["rebate_est"] or 0))
    m_usd = _med(winners, "usd_win")
    m_cap = _med(winners, "usd_vs_cap")
    m_tradesday = _med(winners, "trades_day")
    covs = sorted(r["uptime"]["continuous_coverage_frac"] for r in winners
                  if isinstance(r.get("uptime"), dict)
                  and isinstance(r["uptime"].get("continuous_coverage_frac"), (int, float)))
    med_cov = covs[len(covs) // 2] if covs else None
    bullets = []

    # 1) The reconstruction-era headline breaks under official numbers.
    if over1_win:
        names = "; ".join(
            f"{html.escape(r['handle'])} (pair-cost {_f(r['pair_cost'],4)}, official <b>${_f(r['official_all'],0)}</b>)"
            for r in over1_win[:3])
        bullets.append(
            "<b>The clean “pair-cost &lt; 1 wins / &gt; 1 loses” line does NOT survive official PnL.</b> "
            f"On the cash-flow reconstruction these looked like losers, but Polymarket's own profile P/L shows "
            f"{names} — <b>profitable while paying &gt; $1 per matched pair</b>. The matched portion of a &gt;$1 "
            "pair is a guaranteed loss at resolution, so the profit comes from elsewhere. Decomposing it (below) "
            "shows the source is <b>fee rebates + directional drift</b>, not spread capture — which changes what "
            "is copyable at our tier.")

    # 2) Regime A — trading edge (copyable).
    if under1_win:
        ex = under1_win[0]
        tot_a = sum(r["official_all"] for r in under1_win)
        reb_a = sum((r.get("rebate_est") or 0) for r in under1_win if isinstance(r.get("rebate_est"), (int, float)))
        bullets.append(
            "<b>Regime A — profit from trading edge (copyable at our tier).</b> Only <b>"
            f"{len(under1_win)}</b> of the winners rest as makers <i>inside</i> the spread (pair-cost &lt; 1) — "
            f"e.g. {html.escape(ex['handle'])} (pair-cost {_f(ex['pair_cost'],4)}, official "
            f"<b>${_f(ex['official_all'],0)}</b>) — capturing the matched-pair economics directly (a locked "
            f"$1 − pair-cost per pair). These pure spread-capture makers earn a combined official "
            f"<b>${_f(tot_a,0)}</b> on essentially <b>no rebate</b> (~${_f(reb_a,0)} total). Our "
            "<span class='mono'>pair_cost_threshold = 0.98</span> + post-only puts us squarely here — a real, "
            "standalone profit source we can and do copy, no rebate tier required.")

    # 3) Regime B — directional drift + rebate (NOT copyable at our tier).
    if over1_win:
        r = over1_win[0]
        d = r.get("decomp") or {}
        bullets.append(
            "<b>Regime B — profit from directional drift + rebate (NOT copyable at our tier).</b> The other "
            f"<b>{len(over1_win)}</b> winners pay pair-cost &gt; 1 — a guaranteed loss on the matched portion — "
            f"yet profit anyway. Take the biggest, {html.escape(r['handle'])}: pair-cost {_f(r['pair_cost'],4)}, "
            f"matched-pair PnL <b>${_f(d.get('matched_pair_pnl_usd'),0)}</b> (a loss), official "
            f"<b>${_f(r['official_all'],0)}</b>. The offset is a large directional/settlement swing "
            f"(≈ ${_f(d.get('directional_settlement_usd'),0)} over the captured window) <i>plus</i> an estimated "
            f"<b>${_f(d.get('est_taker_rebate_usd'),0)}</b> taker rebate ({_pct(r.get('rebate_pct'))} at "
            f"{html.escape(str(r.get('tier_name') or '?'))} tier, shown separately not summed) — the residual "
            f"${_f(d.get('residual_usd'),0)} is unexplained and flagged. The conclusion holds regardless of the "
            "exact split: strip out the high-tier rebate cushion and the &gt;$1-pair archetype has no reliable "
            "edge, so it is not a template we can run at None/Bronze tier.")

    # 4) Rebate tier as a moat we lack.
    if bearers:
        total_reb = sum(r["rebate_est"] for r in bearers if isinstance(r.get("rebate_est"), (int, float)))
        top = bearers[0]
        bullets.append(
            "<b>Rebate tier is a competitive moat we don't have.</b> Platinum/Diamond/Obsidian accounts recover "
            f"32–50% of the taker fees they pay; for <b>{len(bearers)}</b> of the winners that rebate is "
            f"load-bearing (≥ 25% of official profit, ≈ <b>${_f(total_reb,0)}</b> combined — e.g. "
            f"{html.escape(top['handle'])} ~${_f(top['rebate_est'],0)} at {_pct(top['rebate_pct'])}). Our maker-first "
            "bot pays almost no taker fees (taker gated to $10/window) and sits at None/Bronze tier, so it earns "
            "essentially no rebate. That is not a gap to close by copying them — it is a structural reason our edge "
            "must come from maker spread capture (Regime A), which the pair-cost&lt;1 winners confirm stands on its own.")

    # 5) Sizing.
    bullets.append(
        f"<b>Sizing.</b> Winning accounts deploy a median ~<b>${_f(m_usd,0)}/window</b>"
        + (f" (~<b>{_f(m_cap,0)}×</b> our $25 max-worst-case-loss cap)" if isinstance(m_cap, (int, float)) else "")
        + ". Same edge per fill, far more size — identical spread capture compounds into orders of magnitude more "
        "PnL. Our binding constraint is the risk cap, not the price.")

    # 6) Cadence / uptime.
    bullets.append(
        f"<b>Cadence / uptime.</b> Winning makers run ~<b>{_f(m_tradesday,0)} our-series trades/day</b>"
        + (f" at ~<b>{_pct(med_cov)}</b> continuous hourly coverage (24/7)" if med_cov is not None else "")
        + ". PnL scales with time-in-market, so downtime — e.g. our feed-staleness cancel-all flapping at home "
        "latency — is pure lost income; a colocated, always-on deployment matters as much as the parameters.")

    # 7) Directional edge at scale.
    top = max(winners, key=lambda r: r["official_all"]) if winners else None
    if top and isinstance(top["two_sided"], (int, float)) and top["two_sided"] < 0.35:
        bullets.append(
            f"<b>A directional edge exists at scale.</b> The most profitable account ({html.escape(top['handle'])}, "
            f"official <b>${_f(top['official_all'],0)}</b>) is <b>mostly one-sided</b> ({_pct(top['two_sided'])} "
            "two-sided) — taking directional positions and redeeming winners, not symmetric market-making. Our "
            "momentum/late-window taker modules gesture at this but are capped at $10/window; a larger directional "
            "sleeve on the best series is worth studying — buying cheap winners, not crossing the spread.")

    # 8) Focus.
    mm = [r for r in winners if r["taker_frac"] is not None and r["taker_frac"] < 0.3 and r["series_mix"]]
    focus = sorted(mm, key=lambda r: -max(r["series_mix"].values()) / max(sum(r["series_mix"].values()), 1))
    if focus:
        f0 = focus[0]
        top_series = max(f0["series_mix"], key=f0["series_mix"].get)
        bullets.append(
            f"<b>Focus.</b> The cleanest profitable maker ({html.escape(f0['handle'])}) concentrates on "
            f"<b>{html.escape(top_series)}</b> with one tight parameter set, rather than spreading thin. Per-series "
            "specialization with full size on the best series (our backtests already point to BTC-5m) beats thin "
            "coverage.")
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"


def _table(rows: list[dict]) -> str:
    head = ["Account", f"Tier (30d taker vol)<br>{FACT}", "Archetype", "Our-series trades",
            f"Coverage (vol)<br>{CALC}", f"Taker%<br>{FACT}", f"Two-sided%<br>{CALC}",
            f"Pair cost<br>{CALC}", f"$/window (× cap)<br>{CALC}",
            "Trades/day", "Merge/Hold",
            f"Official PnL (ALL)<br>{FACT}", f"Official 1M / 1D<br>{FACT}",
            f"Trades-only (✗redeem)<br>{EST}", f"Est. rebate (tier)<br>{EST}"]
    trh = "".join(f"<th>{h}</th>" for h in head)
    trs = []
    for r in rows:
        flag = " <span class='flag'>low-sample</span>" if r["our_trades"] < MIN_SAMPLE else ""
        flag += " <span class='rw'>recent-window</span>" if r.get("truncated") else ""
        oa = r.get("official_all")
        cls = "pos" if isinstance(oa, (int, float)) and oa > 0 else "neg"
        # pair cost > 1 is a matched-portion loss — flag it visually
        pc = r.get("pair_cost")
        pc_cell = (f"<span class='{'neg' if isinstance(pc,(int,float)) and pc > 1 else 'pos'}'>{_f(pc,4)}</span>"
                   if isinstance(pc, (int, float)) else "—")
        off_cell = (f"<span class='{cls}'><b>${_f(oa,0)}</b></span>"
                    + (" <span class='flag'>ambiguous</span>" if r.get("official_ambiguous") else "")
                    ) if isinstance(oa, (int, float)) else "<span class='small'>n/a</span>"
        reb = r.get("rebate_est")
        reb_cell = (f"${_f(reb,0)}<br><span class='small'>{html.escape(str(r.get('tier_name') or '?'))} "
                    f"{_pct(r.get('rebate_pct'))}</span>") if isinstance(reb, (int, float)) else "—"
        cells = [
            f"<b>{html.escape(str(r['handle']))}</b><br><span class='mono'>{r['address'][:10]}…</span>{flag}",
            html.escape(str(r["tier"])),
            html.escape(str(r["archetype"])),
            f"{r['our_trades']:,}",
            _pct(r["coverage_vol"]),
            _pct(r["taker_frac"]),
            _pct(r["two_sided"]),
            pc_cell,
            f"${_f(r['usd_win'],0)}" + (f"<br><span class='small'>{_f(r.get('usd_vs_cap'),1)}× $25</span>"
                                        if isinstance(r.get("usd_vs_cap"), (int, float)) else ""),
            _f(r["trades_day"], 0),
            html.escape(str(r["merge_hold"] or "—")),
            off_cell,
            f"${_f(r.get('official_1m'),0)} / ${_f(r.get('official_1d'),0)}"
            if isinstance(r.get("official_1m"), (int, float)) else "—",
            f"${_f(r.get('trades_only'),0)}",
            reb_cell,
        ]
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    # OUR BOT reference row (paper; no on-chain P/L; None/low rebate tier)
    our = ("<tr class='ourbot'><td><b>OUR BOT</b><br><span class='mono'>current config</span></td>"
           "<td>None/Bronze — ~0–3% rebate</td><td>MM + fee-aware takers</td><td>paper</td><td>—</td>"
           "<td>low ($10/win budget)</td><td>yes</td><td>0.9800</td>"
           "<td>≤$25 max-loss/win (1×)</td><td>—</td><td>MERGES (recycle)</td>"
           "<td>paper</td><td>—</td><td>paper</td><td>~$0 (no rebate cushion)</td></tr>")
    return f"<table><thead><tr>{trh}</tr></thead><tbody>{''.join(trs)}{our}</tbody></table>"


def _card(a: dict, tape: dict | None = None) -> str:
    r = _account_row(a, tape)
    prof = a.get("profile", {})
    sc = a.get("size_cadence", {})
    mh = a.get("merge_hold", {})
    pnl = a.get("pnl", {})
    span = a.get("span", {})
    cov = a.get("coverage", {})
    decomp = pnl.get("decomposition", {})
    reb = pnl.get("rebate", {})
    up = sc.get("uptime", {})
    hours = sc.get("active_hours_utc") or []
    spark = _hours_spark(hours)
    other = cov.get("top_other_categories", {})
    other_txt = ", ".join(f"{html.escape(k)} ${_f(v,0)}" for k, v in other.items()) or "none"
    conf = r["confidence"]
    addr_span = ("" if r["handle"].lower() == r["address"].lower()
                 else f" <span class='mono small'>{r['address']}</span>")
    pseud = prof.get("pseudonym")
    oa = pnl.get("official_all_usd")
    ocls = "pos" if isinstance(oa, (int, float)) and oa > 0 else "neg"
    off_line = ((
        f"ALL <b class='{ocls}'>${_f(oa,0)}</b> · 1M ${_f(pnl.get('official_1m_usd'),0)} · "
        f"1W ${_f(pnl.get('official_1w_usd'),0)} · 1D ${_f(pnl.get('official_1d_usd'),0)} "
        f"<span class='small'>(as of {html.escape(str(pnl.get('official_as_of') or '')[:10])}, "
        f"{pnl.get('official_series_points',0)} pts)</span>"
        + (f" {UNV} decomposition residual dominates" if pnl.get('official_ambiguous') else ""))
        if pnl.get("official_pnl_available") else "<span class='small'>no official P/L returned by the endpoint</span>")
    full_ser = html.escape(json.dumps(cov.get("full_series_volume", {})))
    non_our = ", ".join(html.escape(s) for s in (cov.get("non_our_series") or [])) or "none"
    reb_bearing = " <b class='unv'>rebate load-bearing</b>" if pnl.get("rebate_load_bearing") else ""
    return f"""
    <div class='card'>
      <h3>{html.escape(r['handle'])}{addr_span}{f" <span class='small'>· {html.escape(str(pseud))}</span>" if pseud else ""}</h3>
      <div class='arch'>{html.escape(r['archetype'])} <span class='small'>({conf}: {html.escape(r['rationale'])})</span></div>
      <div class='grid'>
        <div><span class='k'>Official P/L {FACT}</span> {off_line}</div>
        <div><span class='k'>Profit source {EST}</span> matched-pair <b>${_f(decomp.get('matched_pair_pnl_usd'),0)}</b> (guaranteed; neg if pair-cost&gt;1) + directional/settle ${_f(decomp.get('directional_settlement_usd'),0)} + rewards ${_f(decomp.get('reward_income_usd'),0)} + open-MTM ${_f(decomp.get('open_mtm_usd'),0)} + residual ${_f(decomp.get('residual_usd'),0)} ≈ official-window ${_f(decomp.get('official_window_usd'),0)}{reb_bearing}</div>
        <div><span class='k'>Taker rebate {EST} {UNV}</span> ~${_f(reb.get('rebate_income_tier_anchored_usd'),0)} ({_pct(reb.get('rebate_pct'))} of taker fees, {html.escape(str(reb.get('tier_name') or '?'))} tier) · bottom-up ${_f(reb.get('rebate_income_bottom_up_usd'),0)} — shown ALONGSIDE official, not summed (inclusion in official unverified)</div>
        <div><span class='k'>Cross-checks {EST}</span> trades-only (✗redeem) ${_f(pnl.get('trades_only_usd'),0)} · settlement gap ${_f(pnl.get('settlement_gap_usd'),0)} (≈ redemptions; hold-to-resolution {_pct(pnl.get('hold_to_resolution_frac'))}) · reconstruction ${_f(pnl.get('realized_cashflow_usd'),0)} · est. taker fees ${_f(pnl.get('est_taker_fees_paid_usd'),0)}</div>
        <div><span class='k'>Profile {FACT}</span> {html.escape(str(prof.get('pseudonym')))} · {html.escape(str(prof.get('takerTierName')))} tier · 30d taker vol ${_f(prof.get('weightedVolume'),0)}</div>
        <div><span class='k'>Active {FACT}</span> {html.escape(str(span.get('first')))[:10]} → {html.escape(str(span.get('last')))[:10]} ({span.get('active_days')} days){" <b class='rw'>recent-window only — activity paging capped; older history not captured</b>" if span.get('activity_truncated') else ""}</div>
        <div><span class='k'>Full series {CALC}</span> {full_ser} · non-our: {non_our}</div>
        <div><span class='k'>Our-series coverage {CALC}</span> {_pct(r['coverage_vol'])} of $ volume · ${_f(r['our_vol'],0)} · other: {other_txt}</div>
        <div><span class='k'>Maker/Taker {FACT}</span> {_pct(r['taker_frac'])} taker (by count) · {a.get('maker_taker',{}).get('our_taker_trades',0):,} taker / {a.get('maker_taker',{}).get('our_maker_trades',0):,} maker fills</div>
        <div><span class='k'>Pair discipline {CALC}</span> {_pct(r['two_sided'])} two-sided · pair-cost median {_f(r['pair_cost'],4)} (p25 {_f((a.get('pair_discipline',{}).get('pair_cost') or {}).get('p25'),4)}–p75 {_f((a.get('pair_discipline',{}).get('pair_cost') or {}).get('p75'),4)})</div>
        <div><span class='k'>Merge vs hold {FACT}</span> {html.escape(str(mh.get('behavior')))} · {mh.get('merge_events',0):,} merges / {mh.get('redeem_events',0):,} redeems · rewards ${_f(mh.get('reward_usd'),0)}</div>
        {_reaction_line(r)}
        <div><span class='k'>Size / cadence {CALC}</span> ${_f(r['usd_win'],0)}/window ({_f(r.get('usd_vs_cap'),1)}× our $25 cap) · {_f(r['trades_day'],0)} trades/day</div>
        <div><span class='k'>Uptime {CALC}</span> {_pct(up.get('continuous_coverage_frac'))} continuous hourly coverage · {_f(up.get('active_hours_per_day_median'),0)} active hrs/day · max gap {_f(up.get('max_gap_hours'),1)}h</div>
        <div><span class='k'>Active hours (UTC) {CALC}</span> <span class='mono'>{spark}</span></div>
      </div>
    </div>"""


def _reaction_line(r: dict) -> str:
    """Tape-derived reaction + maker/taker corroboration line (empty if no tape data)."""
    if r.get("react_gt5") is None and r.get("tape_agree") is None:
        return ""
    parts = []
    if r.get("react_gt5") is not None:
        parts.append(f"{_pct(r['react_gt5'])} of taker fills follow a &gt;5bps Binance move within 3s "
                     f"(median pre-fill move {_f(r['react_move_med'],2)}bps, n={_f(r['react_n'],0)})")
    if r.get("tape_agree") is not None:
        parts.append(f"tape↔API maker/taker agreement {_pct(r['tape_agree'])} on {_f(r['tape_agree_n'],0)} "
                     "clean-matched fills (weak — coarse timestamps)")
    return ("<div><span class='k'>Reaction (tape)</span> " + " · ".join(parts) + "</div>")


def _hours_spark(hours: list[int]) -> str:
    if not hours:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mx = max(hours) or 1
    return "".join(blocks[min(7, int(7 * h / mx))] for h in hours) + f"  (0–23 UTC, peak {mx:,})"


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa;line-height:1.45}
h1{margin-bottom:2px} .sub{color:#666;margin-bottom:18px}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{border:1px solid #e3e3e3;padding:6px 8px;text-align:right;vertical-align:top}
th{background:#f0f2f5;position:sticky;top:0;text-align:right}
td:first-child,td:nth-child(3),th:first-child,th:nth-child(3){text-align:left}
tr.ourbot td{background:#fff7e6;font-weight:500}
.pos{color:#0a7d28;font-weight:600}.neg{color:#c0341d;font-weight:600}
.mono{font-family:ui-monospace,Consolas,monospace}.small{font-size:11px;color:#777}
.flag{background:#ffe0e0;color:#a00;font-size:10px;padding:1px 4px;border-radius:3px}
.rw{background:#fff0d0;color:#a60;font-size:10px;padding:1px 4px;border-radius:3px}
.fact{background:#e2f5e6;color:#0a7d28;font-size:9.5px;padding:1px 4px;border-radius:3px;font-weight:700;letter-spacing:.3px}
.calc{background:#e0ecfa;color:#1552a0;font-size:9.5px;padding:1px 4px;border-radius:3px;font-weight:700;letter-spacing:.3px}
.est{background:#fff0d0;color:#8a5a00;font-size:9.5px;padding:1px 4px;border-radius:3px;font-weight:700;letter-spacing:.3px}
.unv{background:#ffe0e0;color:#a00;font-size:9.5px;padding:1px 4px;border-radius:3px;font-weight:700;letter-spacing:.3px}
.legend{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:10px 14px;margin:12px 0;font-size:12.5px}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:14px 16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.card h3{margin:0 0 4px} .arch{font-weight:600;color:#2b4c7e;margin-bottom:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;font-size:12.5px}
.grid .k{display:inline-block;min-width:150px;color:#666;font-weight:600}
.note{background:#eef4ff;border-left:3px solid #4472c4;padding:10px 14px;margin:16px 0;font-size:13px}
.warn{background:#fff4e5;border-left:3px solid #e0a030;padding:10px 14px;margin:16px 0;font-size:13px}
ul{line-height:1.6} h2{margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
"""


def build_html(analysis: dict, handles: dict, tape: dict | None = None) -> str:
    accounts = analysis["accounts"]
    rows = [_account_row(a, tape) for a in accounts if "error" not in a]
    # sort by OFFICIAL all-time PnL desc; fall back to reconstruction, unknown last
    def _rank(r):
        for k in ("official_all", "realized"):
            if isinstance(r.get(k), (int, float)):
                return r[k]
        return -1e18
    rows.sort(key=_rank, reverse=True)
    ordered = {r["address"]: r for r in rows}
    acc_by_addr = {a["address"]: a for a in accounts if "error" not in a}
    errored = [a for a in accounts if "error" in a]
    unresolved = [h for h in handles.get("handles", []) if not h.get("address")]

    table = _table(rows)
    cards = "".join(_card(acc_by_addr[addr], tape) for addr in ordered if addr in acc_by_addr)
    diffs = _differences_section(rows)

    resolved_note = ""
    for h in handles.get("handles", []):
        note = h.get("notes", "")
        if h.get("confidence") not in ("high",) or (h.get("alternatives")):
            resolved_note += f"<li><b>{html.escape(h['handle'])}</b>: {h.get('confidence')} — {html.escape(str(note))}</li>"
    resolved_note = f"<ul>{resolved_note}</ul>" if resolved_note else "<p>All handles resolved at high confidence via exact profile-name match.</p>"

    err_html = ""
    if errored or unresolved:
        items = "".join(f"<li>{html.escape(a.get('handle','?'))}: {html.escape(a.get('error','unresolved'))}</li>"
                        for a in errored)
        items += "".join(f"<li>{html.escape(h['handle'])}: unresolved handle</li>" for h in unresolved)
        err_html = f"<div class='warn'><b>Not analyzed:</b><ul>{items}</ul></div>"

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Polymarket competitor analysis — btc/eth up-down</title><style>{CSS}</style></head><body>
<h1>Polymarket competitor analysis</h1>
<div class='sub'>btc/eth/sol Up-Down (all durations) · read-only public-data study · ranked by official Polymarket P/L</div>

<div class='note'><b>What this is.</b> Each account's public trade/activity/position history (from
<span class='mono'>data-api.polymarket.com</span>, cached under <span class='mono'>data/competitors/</span>)
classified by measurable signatures. Maker/taker is the authoritative <span class='mono'>takerOnly</span>
split. <b>PnL is the official Polymarket profile P/L</b> (from <span class='mono'>user-pnl-api.polymarket.com</span>,
the same series the profile chart shows) — authoritative. The old cash-flow reconstruction and a
<b>trades-only (excludes redemptions)</b> figure are kept only as cross-checks; a per-account
<b>profit-source decomposition</b> (matched-pair economics, directional settlement, maker rewards, and an
<i>independent</i> taker-rebate estimate) explains where each official profit comes from, with residuals
flagged rather than force-fit. Timestamps are 1-second-granular (within-window timing is a coarse reaction
proxy — sub-second Binance-lag would need the µs Telonex print join).</div>

<div class='legend'><b>How to read the labels</b> — every figure is tagged by provenance so nothing here is
mistaken for a guess:<br>
{FACT} straight from Polymarket — the official profile P/L (<span class='mono'>user-pnl-api</span>), the
Gamma profile tier + 30-day taker weighted volume, the raw trade/redeem/merge records
(<span class='mono'>data-api</span>), the <span class='mono'>takerOnly</span> maker/taker labels, and the
documented fee/rebate schedule. &nbsp; {CALC} an exact deterministic computation on that raw data (pair-cost,
series volumes/coverage, $/window, trades/day, uptime). &nbsp; {EST} an estimate or model, <b>not</b> a
Polymarket figure (taker fees via the documented formula, rebate income, the cash-flow reconstruction,
trades-only, the profit-source decomposition + residual). &nbsp; {UNV} a genuine assumption we cannot confirm
(above all: whether the official P/L already includes daily rebate credits) — flagged, never asserted.</div>

{err_html}

<h2>Summary — one row per account</h2>
{table}

<h2>Where the profitable accounts differ from our bot</h2>
{diffs}

<h2>Per-account detail</h2>
{cards}

<h2>Handle resolution notes</h2>
{resolved_note}

<div class='warn'><b>Honesty / caveats.</b> (1) The money column is <b>{FACT} official Polymarket P/L</b>
(<span class='mono'>user-pnl-api.polymarket.com</span>, cumulative realized + unrealized mark-to-market). The
cash-flow reconstruction and trades-only figures are {EST} cross-checks only. (2) <b>{UNV} The single biggest
open question:</b> whether the official P/L already includes the daily USDC <b>taker-fee rebate</b>. The rebate
is credited as a separate daily transfer, so it may or may not be inside the position-based P/L number — we
therefore show the estimated rebate <b>alongside</b> official, never summed into it, and flag it. (3) The
<b>rebate income is an {EST}</b>: tier% (from the account's real Gamma <span class='mono'>takerTierName</span>,
{FACT}) × taker fees paid ({EST}, since fees aren't in the API — computed from the documented formula on the
<span class='mono'>takerOnly</span> fill set). (4) The <b>profit-source decomposition is {EST}</b>; its
<i>residual</i> absorbs mark-to-market, other-market PnL, truncation and redemption-timing — it is reported,
never tuned to zero. Accounts whose residual dominates are flagged {UNV}. (5) A few top accounts carry a Gamma
tier label (e.g. Obsidian) that is <i>higher</i> than their shown 30-day weighted volume would imply by the
documented thresholds — we use Polymarket's own <span class='mono'>takerTierName</span> as authoritative for the
rebate % and note the discrepancy rather than recomputing. (6) Accounts with &lt; {MIN_SAMPLE} our-series
trades are <span class='flag'>low-sample</span>; "taker" = a fill present in
<span class='mono'>/trades?takerOnly=true</span>. (7) A wallet can run multiple accounts; similar-name accounts
are listed in resolution notes, not merged. (8) <span class='rw'>recent-window</span> accounts trade so fast the
Data API's activity paging capped before their full history — official P/L is still lifetime, but the
reconstruction/fees/rebate/decomposition are window-scoped (the decomposition uses the official P/L accrued
<i>during</i> that window for an apples-to-apples split).</div>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render competitor analysis HTML report.")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)
    base = Path(args.dir) if args.dir else competitors_dir()
    odir = out_dir()
    analysis = json.loads((odir / "analysis.json").read_text(encoding="utf-8"))
    handles = json.loads((base / "handles.json").read_text(encoding="utf-8"))
    tape_path = odir / "tape_validation.json"
    tape = None
    if tape_path.exists():
        tape = {a["address"]: a for a in json.loads(tape_path.read_text(encoding="utf-8"))["accounts"]
                if a.get("address")}
    html_txt = build_html(analysis, handles, tape)
    out = odir / "report.html"
    out.write_text(html_txt, encoding="utf-8")
    print(f"[report] wrote {out} ({len(html_txt):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
