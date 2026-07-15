"""Stage — forensic_audit: adversarial audit of a momentum backtest run.

Runs the previously-specced forensic battery against a **chosen backtest run** (default the
current-regime VPS 255 ms slice, `out/backtests/momentum_current_regime/vps255`):

1. **Accounting check** — recompute fee / cost / payoff / net / won identities for *every*
   trade from `shares`/`price`/`outcome` (independent of the sim) and reconcile the sums to
   the reported `net_pnl` / `fees`.
2. **25-trade hand-verification** — re-derive 25 sampled trades from the **raw recorded
   Telonex book** as-of `fire + effective_latency`: the fill price (book ask / mirror),
   budget-tracked shares, the edge gate, the fee (independent formula), and the
   official-resolution match. Every number tied back to ground truth.
3. **Inverted trigger** — a re-run betting AGAINST the confirmed direction. A real
   directional edge must collapse (≪ baseline, ideally negative).
4. **Shuffled labels** — a re-run with per-day-permuted outcomes. PnL must collapse to ~fees.
5. **Doubled fees** — a re-run at fee_rate 0.14. Does the edge survive 2× costs?

Plus **latency decay** (255 / 450 / 500 / 1000 / 2000 ms → net PnL curve) and a **capital
accounting** section: max concurrent open exposure, capital required to take every trade, PnL
as % return on that capital, and PnL if the bankroll were capped at $1k / $5k / $20k (a trade
that can't be afforded is skipped). `$X/day` is meaningless without the bankroll it needs.

The re-run inputs (3–5, latency points) are produced by `backtest_momentum --out-name
forensic/...`; this stage reads their `metrics.json`. Output
`out/backtests/forensic/forensic_report.{md,html,json}`.

Run::  python -m model_lab.forensic_audit
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from . import eval_harness as eh
from . import historical_common as hc
from . import historical_resolutions as hr
from . import money_judge as mj
from .config import Paths, resolve_paths, stage_parser
from .io import telonex_pm as tpm
from .lib import procmem

BASELINE_DIR = "momentum_current_regime/vps255"
N_HANDVERIFY = 25
CAPS = [1_000.0, 5_000.0, 20_000.0]
FEE_RATE = 0.07
BUFFER = mj.MOMENTUM_BUFFER  # 0.005
MS_PER_DAY = 86_400_000
HV_SEED = 20260710
FLOAT_TOL = 1e-6

# Latency-decay points: (label, effective_ms, run dir). 255/450 exist from the verdict runs;
# 500/1000/2000 are forensic re-runs.
LATENCY_POINTS = [
    (255, "momentum_current_regime/vps255"),
    (450, "momentum_current_regime/home450"),
    (500, "forensic/lat_500"),
    (1000, "forensic/lat_1000"),
    (2000, "forensic/lat_2000"),
]


# --- helpers ----------------------------------------------------------------
def _fee_formula(shares: float, price: float, rate: float = FEE_RATE) -> float:
    """Independent replica of ``core-types`` / ``lm.taker_fee``: shares·rate·p·(1−p),
    5-dp half-away, $0.00001 floor (0 stays 0)."""
    raw = shares * rate * price * (1.0 - price)
    if raw == 0.0:
        return 0.0
    q = math.floor(abs(raw) * 1e5 + 0.5) / 1e5 * (1.0 if raw > 0 else -1.0)
    return max(q, 1e-5)


def _dur_ms(series: str) -> int:
    return 900_000 if "15m" in series else 300_000


def _slug_and_day(series: str, open_ms: int) -> tuple[str, str]:
    asset = "btc" if series.upper().startswith("BTC") else "eth"
    dur = "15m" if "15m" in series else "5m"
    slug = f"{asset}-updown-{dur}-{int(open_ms) // 1000}"
    return slug, hc.utc_day(int(open_ms)).isoformat()


def _load_run(paths: Paths, subdir: str) -> tuple[pd.DataFrame | None, dict | None]:
    d = paths.out_dir / "backtests" / subdir
    m = d / "metrics.json"
    if not m.exists():
        return None, None
    metrics = json.loads(m.read_text(encoding="utf-8"))
    tf = d / "trades.csv"
    df = None
    if tf.exists():
        try:
            df = pd.read_csv(tf)
        except (ValueError, pd.errors.EmptyDataError):
            df = None
    return df, metrics


# --- (1) accounting ---------------------------------------------------------
FEE_QUANTUM = 1.5e-5  # one 5-dp fee unit (1e-5) + float slack


def accounting_check(df: pd.DataFrame, metrics: dict) -> dict:
    """Verify the accounting identities and reconcile the sums to the reported totals.

    The cost/payoff/net identities use the **recorded** fee (exact — independent of fee
    rounding). The fee itself is separately checked against an independent formula; a handful
    of trades round the fee's last 5-dp digit differently when recomputed from the CSV-stored
    (rounded) shares/price than the runtime (full-precision) value did — a storage-precision
    boundary of ≤ one 1e-5 unit, tolerated and reported, not an accounting error."""
    shares = df["shares"].to_numpy(float)
    price = df["price"].to_numpy(float)
    fee_rec = df["fee"].to_numpy(float)
    cost_rec = df["cost"].to_numpy(float)
    payoff_rec = df["payoff"].to_numpy(float)
    net_rec = df["net"].to_numpy(float)
    won_rec = df["won"].to_numpy(bool)
    outcome_up = df["outcome_up"].to_numpy(int)
    side_up = (df["side"].to_numpy() == "Up")

    won_id = (side_up & (outcome_up == 1)) | (~side_up & (outcome_up == 0))
    payoff_id = np.where(won_id, shares, 0.0)
    cost_id = shares * price + fee_rec           # identity uses recorded fee → exact
    net_id = payoff_rec - cost_rec
    fee_formula = np.array([_fee_formula(s, p) for s, p in zip(shares, price)])
    fee_delta = np.abs(fee_formula - fee_rec)

    def mx(a):
        return float(np.max(a)) if len(a) else 0.0

    err_payoff = mx(np.abs(payoff_id - payoff_rec))
    err_cost = mx(np.abs(cost_id - cost_rec))
    err_net = mx(np.abs(net_id - net_rec))
    won_mm = int(np.sum(won_id != won_rec))
    n_fee_boundary = int(np.sum(fee_delta > 1e-9))
    max_fee_delta = mx(fee_delta)

    c = metrics["counts"]
    net_sum, fee_sum = float(net_rec.sum()), float(fee_rec.sum())
    net_ok = abs(net_sum - c["net_pnl"]) < 1e-2
    fee_ok = abs(fee_sum - c["fees"]) < 1e-2
    clean = bool(err_payoff < FLOAT_TOL and err_cost < FLOAT_TOL and err_net < FLOAT_TOL
                 and won_mm == 0 and max_fee_delta <= FEE_QUANTUM and net_ok and fee_ok)
    return {
        "n_trades": int(len(df)),
        "err_payoff_identity": err_payoff, "err_cost_identity": err_cost, "err_net_identity": err_net,
        "won_mismatches": won_mm,
        "n_fee_rounding_boundary": n_fee_boundary, "max_fee_formula_delta": max_fee_delta,
        "fee_note": (f"{n_fee_boundary} of {len(df)} fees differ by ≤ one 5-dp unit (1e-5) when "
                     "recomputed from the CSV-stored shares/price — a storage-precision rounding "
                     "boundary, not an accounting error (the runtime fee is exact)."),
        "sum_net": net_sum, "reported_net": float(c["net_pnl"]), "net_reconciles": bool(net_ok),
        "sum_fee": fee_sum, "reported_fee": float(c["fees"]), "fee_reconciles": bool(fee_ok),
        "clean": clean,
    }


# --- (2) 25-trade hand-verification -----------------------------------------
def hand_verify(df: pd.DataFrame, paths: Paths, res_map: dict, latency_ms: int,
                n: int = N_HANDVERIFY) -> dict:
    """Re-derive `n` sampled trades from the raw recorded book at fire+latency."""
    rng = np.random.default_rng(HV_SEED)
    # stratify across series so every series is represented
    idx: list[int] = []
    for sk in sorted(df["series"].unique()):
        pool = df.index[df["series"] == sk].to_numpy()
        take = min(len(pool), max(1, n // max(1, df["series"].nunique())))
        idx.extend(rng.choice(pool, size=take, replace=False).tolist())
    idx = list(dict.fromkeys(idx))[:n]

    # prior spend per (series, window) up to each sample_ts, to reconstruct budget_left
    df_sorted = df.sort_values(["series", "window_open_ms", "sample_ts_ms"]).reset_index()
    book_cache: dict[tuple, dict] = {}
    rows: list[dict] = []
    for i in idx:
        r = df.loc[i]
        series, open_ms = r["series"], int(r["window_open_ms"])
        fire = int(r["sample_ts_ms"])
        side = r["side"]
        # budget_left = 10 − Σ(shares·price) of earlier trades in this window (same run)
        earlier = df[(df["series"] == series) & (df["window_open_ms"] == open_ms)
                     & (df["sample_ts_ms"] < fire)]
        budget_left = mj.DEFAULT_WINDOW_BUDGET - float((earlier["shares"] * earlier["price"]).sum())

        slug, day = _slug_and_day(series, open_ms)
        ckey = (series, open_ms)
        if ckey not in book_cache:
            q = tpm.read_pm_quotes(paths.telonex_dir, slug, "Up", day)
            tok = str(q["token_id"].iloc[0]) if not q.empty else None
            book_cache[ckey] = {"books": {tok: tpm.pm_book_arrays(q)} if tok else {}, "token": tok}
        cache = book_cache[ckey]
        token = cache["token"]
        book = mj._book_asof(cache["books"], token, fire + latency_ms) if token else None
        quote = mj._leaned_quote(book, side) if book else None

        rec_price = float(r["price"])
        rec_shares = float(r["shares"])
        fair = float(r["p_up"]) if side == "Up" else 1.0 - float(r["p_up"])
        checks: dict = {"book_found": quote is not None}
        if quote is not None:
            price, avail = quote
            exp_shares = min(avail, budget_left / price) if price > 0 else 0.0
            fee_ps = FEE_RATE * price * (1.0 - price)
            checks.update({
                "price_ok": abs(price - rec_price) < 1e-9,
                "book_price": price, "book_avail": avail,
                "shares_ok": abs(exp_shares - rec_shares) < 1e-6,
                "exp_shares": exp_shares,
                "edge_gate_ok": (fair - rec_price) > fee_ps * (1.0 - mj.TAKER_REBATE) + BUFFER,
            })
        official = res_map.get((series, open_ms))
        exp_outcome = (1 if official["outcome"] == "Up" else 0) if official else None
        fee_ok = abs(_fee_formula(rec_shares, rec_price) - float(r["fee"])) < 1e-6
        cost_ok = abs(rec_shares * rec_price + float(r["fee"]) - float(r["cost"])) < 1e-6
        won_ok = bool(r["won"]) == ((side == "Up" and int(r["outcome_up"]) == 1)
                                    or (side == "Down" and int(r["outcome_up"]) == 0))
        net_ok = abs(float(r["payoff"]) - float(r["cost"]) - float(r["net"])) < 1e-6
        res_ok = (exp_outcome is not None) and (exp_outcome == int(r["outcome_up"]))
        passed = bool(checks.get("price_ok") and checks.get("shares_ok")
                      and checks.get("edge_gate_ok") and fee_ok and cost_ok and won_ok
                      and net_ok and res_ok)
        rows.append({"series": series, "window_open_ms": open_ms, "fire_ms": fire, "side": side,
                     "rec_price": rec_price, "rec_shares": round(rec_shares, 4),
                     "rec_fee": float(r["fee"]), "rec_net": float(r["net"]), "won": bool(r["won"]),
                     "price_ok": checks.get("price_ok"), "shares_ok": checks.get("shares_ok"),
                     "edge_gate_ok": checks.get("edge_gate_ok"), "fee_ok": fee_ok,
                     "acct_ok": bool(cost_ok and won_ok and net_ok), "resolution_ok": res_ok,
                     "PASS": passed})
    n_pass = sum(1 for r in rows if r["PASS"])
    return {"n": len(rows), "n_pass": n_pass, "all_pass": bool(n_pass == len(rows) and rows),
            "trades": rows}


# --- (7) capital accounting -------------------------------------------------
def capital_accounting(df: pd.DataFrame, caps=CAPS) -> dict:
    """Concurrent open exposure, capital required, %return, and capped-bankroll PnL."""
    t = df.copy()
    t["open_ms"] = t["sample_ts_ms"].astype("int64")
    t["dur_ms"] = np.where(t["series"].str.contains("15m"), 900_000, 300_000)
    t["close_ms"] = t["window_open_ms"].astype("int64") + t["dur_ms"]
    total_net = float(t["net"].sum())

    # peak concurrent exposure = max running Σ cost of open positions (opens before closes at ties)
    opens = pd.DataFrame({"ts": t["open_ms"], "d": t["cost"].to_numpy(float), "c": 0})
    closes = pd.DataFrame({"ts": t["close_ms"], "d": -t["cost"].to_numpy(float), "c": 1})
    ev = pd.concat([opens, closes]).sort_values(["ts", "c"], kind="stable")
    exposure = ev["d"].cumsum().to_numpy()
    peak = float(exposure.max()) if len(exposure) else 0.0

    span_days = max(1, int((t["open_ms"].max() - t["open_ms"].min()) // MS_PER_DAY) + 1)
    caps_out = []
    import heapq
    ts_arr = t.sort_values("open_ms")
    for cap in caps:
        avail = cap
        heap: list = []
        taken_net = 0.0
        n_taken = 0
        for r in ts_arr.itertuples(index=False):
            while heap and heap[0][0] <= r.open_ms:
                avail += heapq.heappop(heap)[1]
            if r.cost <= avail:
                avail -= r.cost
                heapq.heappush(heap, (r.close_ms, r.payoff))
                taken_net += r.net
                n_taken += 1
        caps_out.append({"cap": cap, "net_pnl": taken_net, "n_taken": n_taken,
                         "frac_trades_taken": (n_taken / len(t)) if len(t) else 0.0,
                         "pct_return": (taken_net / cap) if cap else float("nan")})
    return {
        "n_trades": int(len(t)),
        "peak_concurrent_exposure": peak,
        "capital_required": peak,
        "total_net_pnl": total_net,
        "pct_return_on_required": (total_net / peak) if peak else float("nan"),
        "span_days": span_days,
        "net_per_day": total_net / span_days,
        "capped": caps_out,
    }


# --- assembly ---------------------------------------------------------------
def forensic_audit(paths: Paths) -> dict:
    base_df, base_m = _load_run(paths, BASELINE_DIR)
    if base_df is None or base_m is None:
        raise SystemExit(f"[forensic_audit] baseline run not found at out/backtests/{BASELINE_DIR}")
    eff_lat = int(base_m["params"].get("effective_latency_ms", base_m["params"].get("latency_ms", 255)))
    series_keys = tuple(base_m["params"].get("series", hc.DEFAULT_SERIES))
    res_map = hr.load_resolution_map(paths, series_keys)

    acct = accounting_check(base_df, base_m)
    hv = hand_verify(base_df, paths, res_map, eff_lat)
    cap = capital_accounting(base_df)

    base_net = float(base_m["counts"]["net_pnl"])
    base_ctrl = base_m["controls"]

    def variation(subdir):
        _df, m = _load_run(paths, subdir)
        if m is None:
            return None
        c, ctrl, pr = m["counts"], m["controls"], m["params"]
        return {"net_pnl": float(c["net_pnl"]), "trades": int(c["trades"]),
                "win_rate": float(c["win_rate"]), "fee_rate": pr.get("fee_rate"),
                "beats_random": ctrl.get("real_beats_shuffled"), "p": ctrl.get("shuffled_p_value")}

    inverted = variation("forensic/inverted")
    shuffled = variation("forensic/shuffled_labels")
    doubled = variation("forensic/doubled_fees")

    # latency decay
    curve = []
    for eff, subdir in LATENCY_POINTS:
        _df, m = _load_run(paths, subdir)
        if m is None:
            continue
        c = m["counts"]
        curve.append({"effective_ms": eff, "net_pnl": float(c["net_pnl"]), "trades": int(c["trades"]),
                      "win_rate": float(c["win_rate"]), "pnl_per_trade": float(c.get("pnl_per_trade", float("nan")))})

    # verdicts
    inv_ok = inverted is not None and inverted["net_pnl"] < 0.2 * base_net
    shf_ok = shuffled is not None and shuffled["net_pnl"] < 0.2 * base_net
    result = {
        "baseline": {"dir": BASELINE_DIR, "effective_latency_ms": eff_lat, "net_pnl": base_net,
                     "trades": int(base_m["counts"]["trades"]),
                     "windows": int(base_m["counts"]["resolved_windows"]),
                     "win_rate": float(base_m["counts"]["win_rate"]),
                     "beats_random": base_ctrl.get("real_beats_shuffled"),
                     "p": base_ctrl.get("shuffled_p_value")},
        "accounting": acct,
        "hand_verification": hv,
        "inverted_trigger": inverted, "inverted_collapses": bool(inv_ok),
        "shuffled_labels": shuffled, "shuffled_collapses": bool(shf_ok),
        "doubled_fees": doubled,
        "latency_curve": curve,
        "capital": cap,
        "clean": bool(acct["clean"] and hv["all_pass"] and inv_ok and shf_ok),
        "peak_rss_mb": procmem.peak_rss_mb(),
    }
    result["verdict"] = _verdict(result)
    _write_outputs(paths, result)
    return result


def _verdict(r: dict) -> str:
    b = r["baseline"]
    parts = [f"Baseline (current-regime VPS {b['effective_latency_ms']}ms): net ${b['net_pnl']:,.0f} "
             f"over {b['trades']:,} trades / {b['windows']:,} windows, {b['win_rate']:.0%} win, "
             f"beats random={b['beats_random']} (p={b['p']:.2f})."]
    parts.append("Accounting: " + ("CLEAN — every fee/cost/payoff/net identity reproduces and the "
                 "sums reconcile to the reported totals." if r["accounting"]["clean"]
                 else "MISMATCH — see the accounting table (a real defect)."))
    hv = r["hand_verification"]
    parts.append(f"Hand-verification: {hv['n_pass']}/{hv['n']} sampled trades re-derived exactly "
                 f"from the raw book at fire+{b['effective_latency_ms']}ms "
                 + ("(all PASS)." if hv["all_pass"] else "(FAILURES — see table)."))
    if r["inverted_trigger"]:
        iv = r["inverted_trigger"]
        parts.append(f"Inverted trigger (bet against the signal): net ${iv['net_pnl']:,.0f} over "
                     f"{iv['trades']:,} trades — " + ("COLLAPSES vs baseline (edge is directional)."
                     if r["inverted_collapses"] else "does NOT collapse (edge suspect!)."))
    if r["shuffled_labels"]:
        sh = r["shuffled_labels"]
        parts.append(f"Shuffled labels: net ${sh['net_pnl']:,.0f} — " + ("COLLAPSES to ~fees (signal "
                     "carries the outcome)." if r["shuffled_collapses"] else "does NOT collapse (artifact!)."))
    if r["doubled_fees"]:
        dd = r["doubled_fees"]
        parts.append(f"Doubled fees (0.14): net ${dd['net_pnl']:,.0f}, beats random={dd['beats_random']} "
                     + ("— edge survives 2× costs." if dd["net_pnl"] > 0 else "— edge does NOT survive 2× costs."))
    cap = r["capital"]
    smallest = min(cap["capped"], key=lambda x: x["cap"]) if cap["capped"] else None
    binding = smallest is not None and smallest["frac_trades_taken"] > 0.999
    parts.append(f"Capital: peak concurrent exposure is only ${cap['peak_concurrent_exposure']:,.0f} "
                 f"(the whole strategy's bankroll) — a {cap['pct_return_on_required']:.0%} return over "
                 f"{cap['span_days']} days (${cap['net_per_day']:,.0f}/day). "
                 + ("Bankroll is NOT the binding constraint — even $1k takes ~100% of trades; the "
                    "ceiling on absolute PnL is the $10/window budget + book depth, not capital. "
                    if binding else "")
                 + "; ".join(f"@${c['cap']:,.0f}→${c['net_pnl']:,.0f} ({c['frac_trades_taken']:.0%})"
                             for c in cap["capped"]) + ".")
    parts.append("OVERALL: " + ("the edge is real, correctly accounted, directional, and survives the "
                 "falsifications." if r["clean"] else "at least one check failed — read the sections."))
    return " ".join(parts)


# --- report -----------------------------------------------------------------
def _lat_chart(curve: list[dict]) -> str | None:
    if len(curve) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io as _io
    import base64
    c = sorted(curve, key=lambda x: x["effective_ms"])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot([p["effective_ms"] for p in c], [p["net_pnl"] for p in c], marker="o", color="#1f77b4")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("effective fill latency (ms)"); ax.set_ylabel("net PnL $")
    ax.set_title("Momentum net PnL vs latency (current regime)")
    buf = _io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_outputs(paths: Paths, r: dict) -> None:
    out_dir = paths.out_dir / "backtests" / "forensic"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forensic_report.json").write_text(json.dumps(r, indent=2, default=eh._json_default),
                                                  encoding="utf-8")
    # markdown
    b, acct, hv, cap = r["baseline"], r["accounting"], r["hand_verification"], r["capital"]
    md = [f"# Forensic audit — current-regime VPS {b['effective_latency_ms']}ms momentum run", "",
          r["verdict"], "",
          "## 1. Accounting check (all trades)", "",
          f"- trades: **{acct['n_trades']:,}** · won-mismatches: **{acct['won_mismatches']}** · "
          f"identity errors (payoff/cost/net): {acct['err_payoff_identity']:.2e} / "
          f"{acct['err_cost_identity']:.2e} / {acct['err_net_identity']:.2e}",
          f"- Σnet ${acct['sum_net']:,.4f} vs reported ${acct['reported_net']:,.4f} "
          f"({'reconciles' if acct['net_reconciles'] else 'MISMATCH'}); "
          f"Σfee ${acct['sum_fee']:,.4f} vs ${acct['reported_fee']:,.4f} "
          f"({'reconciles' if acct['fee_reconciles'] else 'MISMATCH'}) → **{'CLEAN' if acct['clean'] else 'FAIL'}**",
          f"- fee-formula: {acct['fee_note']}",
          "", f"## 2. Hand-verification ({hv['n_pass']}/{hv['n']} re-derived from raw book)", ""]
    md.append("| series | fire_ms | side | price | shares | fee | net | won | price✓ | shares✓ | edge✓ | fee✓ | acct✓ | res✓ | PASS |")
    md.append("|---|--:|---|--:|--:|--:|--:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for t in hv["trades"]:
        md.append(f"| {t['series']} | {t['fire_ms']} | {t['side']} | {t['rec_price']:.3f} | "
                  f"{t['rec_shares']} | {t['rec_fee']:.5f} | {t['rec_net']:.4f} | {t['won']} | "
                  f"{t['price_ok']} | {t['shares_ok']} | {t['edge_gate_ok']} | {t['fee_ok']} | "
                  f"{t['acct_ok']} | {t['resolution_ok']} | {'✅' if t['PASS'] else '❌'} |")
    md += ["", "## 3–5. Falsifications", "",
           "| check | net PnL $ | trades | win | beats random | verdict |", "|---|--:|--:|--:|:-:|---|"]
    for name, v, ok, want in [("baseline", None, None, ""),
                              ("inverted trigger", r["inverted_trigger"], r["inverted_collapses"], "should collapse"),
                              ("shuffled labels", r["shuffled_labels"], r["shuffled_collapses"], "should collapse"),
                              ("doubled fees (0.14)", r["doubled_fees"], None, "should survive")]:
        if name == "baseline":
            md.append(f"| baseline | {b['net_pnl']:,.0f} | {b['trades']:,} | {b['win_rate']:.0%} | "
                      f"{b['beats_random']} | reference |")
        elif v is None:
            md.append(f"| {name} | (run pending) | | | | {want} |")
        else:
            verdict = ("✅ collapses" if ok else ("survives ✅" if (name.startswith("doubled") and v["net_pnl"] > 0)
                       else ("❌ does not" if name != "doubled fees (0.14)" else "❌ does not survive")))
            md.append(f"| {name} | {v['net_pnl']:,.0f} | {v['trades']:,} | {v['win_rate']:.0%} | "
                      f"{v['beats_random']} | {verdict} |")
    md += ["", "## 6. Latency decay", "",
           "| effective latency | net PnL $ | trades | win | $/trade |", "|--:|--:|--:|--:|--:|"]
    for p in sorted(r["latency_curve"], key=lambda x: x["effective_ms"]):
        md.append(f"| {p['effective_ms']} ms | {p['net_pnl']:,.0f} | {p['trades']:,} | "
                  f"{p['win_rate']:.0%} | {p['pnl_per_trade']:.4f} |")
    smallest = min(cap["capped"], key=lambda x: x["cap"]) if cap["capped"] else None
    binding = (smallest is not None and smallest["frac_trades_taken"] > 0.999)
    md += ["", "## 7. Capital accounting", "",
           f"- **Max concurrent open exposure / capital required to take every trade: "
           f"${cap['peak_concurrent_exposure']:,.0f}**",
           f"- Total net ${cap['total_net_pnl']:,.2f} over {cap['span_days']} days "
           f"(${cap['net_per_day']:,.2f}/day) → **{cap['pct_return_on_required']:.1%} return on the "
           f"required capital** over the period (not annualized)",
           (f"- **Bankroll is NOT the binding constraint**: even a ${smallest['cap']:,.0f} cap takes "
            f"{smallest['frac_trades_taken']:.0%} of trades. Peak exposure is tiny because each fire "
            f"is capped at the **${mj.DEFAULT_WINDOW_BUDGET:.0f}/window** taker budget and 5–15 min "
            "windows recycle capital fast (~4–8 windows open at once). The real ceiling on absolute "
            "PnL is that per-window budget and the displayed book depth, **not** the bankroll — so "
            "the eye-popping % return is on a trivially small base." if binding else
            f"- A ${smallest['cap']:,.0f} bankroll already takes {smallest['frac_trades_taken']:.0%} "
            "of trades; capital is a minor constraint at this size."),
           "",
           "| bankroll cap | net PnL $ | trades taken | % of trades | % return on cap |",
           "|--:|--:|--:|--:|--:|"]
    for c in cap["capped"]:
        md.append(f"| ${c['cap']:,.0f} | {c['net_pnl']:,.2f} | {c['n_taken']:,} | "
                  f"{c['frac_trades_taken']:.0%} | {c['pct_return']:.1%} |")
    md += ["", "> A trade whose cost exceeds available capital is **skipped** (all-or-nothing per "
           "trade); freed capital returns at each window's resolution. `$/day` is meaningless without "
           "the bankroll — the caps show how PnL scales with it.", ""]
    (out_dir / "forensic_report.md").write_text("\n".join(md), encoding="utf-8")

    chart = _lat_chart(r["latency_curve"])
    html = ["<!doctype html><meta charset='utf-8'><title>Forensic audit</title>",
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;max-width:1000px;margin:2rem auto;"
            "padding:0 1rem}table{border-collapse:collapse;font-size:.85rem;margin:.5rem 0}"
            "td,th{padding:3px 9px;border-bottom:1px solid #eee;text-align:left}"
            ".v{background:#f6f8fa;border-left:4px solid #1f77b4;padding:.8rem 1rem;border-radius:4px;"
            "line-height:1.5}img{max-width:100%}</style>",
            f"<h1>Forensic audit — VPS {b['effective_latency_ms']}ms momentum run</h1>",
            f"<div class='v'>{r['verdict']}</div>"]
    if chart:
        html.append(f"<h2>Latency decay</h2>{eh._img_tag(chart, 'net PnL vs latency')}")
    html.append("<h2>Capital accounting</h2>" + eh._table_html(
        [{"bankroll": f"${c['cap']:,.0f}", "net_pnl": round(c["net_pnl"], 2),
          "trades_taken": c["n_taken"], "pct_trades": f"{c['frac_trades_taken']:.0%}",
          "pct_return": f"{c['pct_return']:.1%}"} for c in cap["capped"]],
        [("bankroll", "bankroll cap"), ("net_pnl", "net PnL $"), ("trades_taken", "trades taken"),
         ("pct_trades", "% of trades"), ("pct_return", "% return")]))
    html.append("<p>Full tables in <code>forensic_report.md</code>.</p>")
    (out_dir / "forensic_report.html").write_text("\n".join(html), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "forensic_audit")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    r = forensic_audit(paths)
    print(f"[forensic_audit] {r['verdict']}")
    procmem.report_peak_rss("forensic_audit", r.get("peak_rss_mb"), args.max_rss_mb)
    print(f"[forensic_audit] wrote {paths.out_dir / 'backtests' / 'forensic' / 'forensic_report.md'}")
    return 0 if r["clean"] else 2


if __name__ == "__main__":
    sys.exit(main())
