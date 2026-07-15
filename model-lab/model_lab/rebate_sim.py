"""Stage — rebate_sim.

Replay any taker-fill ``trades.csv`` (from :mod:`model_lab.backtest_momentum` or the
model-taker export in :mod:`model_lab.money_judge`) through Polymarket's tiered
**Taker Fee Rebate Program** and report what the rebate is worth.

The program (docs.polymarket.com/trading/taker-rebates, verified 2026-07-13):

- Your **tier** is set by 30-day **rolling** *weighted volume*
  ``wV = size · (1 − entry_price) · category_weight`` (crypto weight ``2.3``; no
  crypto bonus). Tiers: ``None <$2k`` · ``Bronze $2k → 3%`` · ``Silver $20k → 8%`` ·
  ``Gold $200k → 18%`` · ``Platinum $1M → 32%`` · ``Diamond $4M → 44%`` ·
  ``Obsidian $10M → 50%``.
- The rebate **paid** is ``tier% × the TAKER FEES you paid`` over the period — a fee
  *refund*, NOT a percentage of notional. Paid daily 00:00 UTC in pUSD, ``$1``
  minimum accrual, recalculated daily and applied going forward.
- **On-chain / API form:** the credit arrives as a separate ``TAKER_REBATE`` event on the
  Polymarket ``/activity`` feed (NOT inside trade/redeem records), verified 2026-07-15
  (out/audits/takerner_pnl_anomaly.md). This is the event type **our own live rebates will
  arrive as** — reconcile them from ``TAKER_REBATE`` activity, not from the fee math alone.

So the rebate is **path-dependent on trailing volume** and cannot be baked into
per-trade fee math — it is computed here on a daily cycle. For each UTC exec-day D:
``wv_day``, ``fees_day``; ``tier_D`` from the trailing 30 completed days ``[D-30 …
D-1]`` (the docs' non-retroactive "effective next update / going forward" rule);
``rebate_day = tier_D% × fees_day``; a ``$1`` daily-accrual carry (a payout only when
the running accrual reaches $1). Reports the **tier timeline**, **total rebate**,
**corrected net PnL** (``net + rebate``), and the **marginal wV** to the next tier.

The tier table + weight mirror the Rust consts in
``crates/venue-paper/src/engine.rs`` and live in :mod:`model_lab.lib.math` — keep in
sync. Output ``out/backtests/rebate_sim/<out_name>/{tier_timeline.csv, metrics.json,
report.html}``.

Run::

    python -m model_lab.rebate_sim   # defaults to the momentum-verdict baseline
    python -m model_lab.rebate_sim --trades out/money_judge/trades.csv --out-name model_taker
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import eval_harness as eh
from .config import ParquetNotReady, resolve_paths, stage_parser
from .lib import math as lm

MS_PER_DAY = 86_400_000
ROLLING_DAYS = 30
MIN_PAYOUT = 1.0  # $1 minimum daily accrual before a payout (docs).

#: One-time level-up bonus paid the first time a tier is reached (docs). Reported
#: as an informational line only — NOT folded into the running fee rebate.
LEVELUP_BONUS = {
    "Bronze": 10.0,
    "Silver": 50.0,
    "Gold": 250.0,
    "Platinum": 1_500.0,
    "Diamond": 7_500.0,
    "Obsidian": 25_000.0,
}

#: Columns rebate_sim needs from a trades.csv (both producers emit this header).
_REQUIRED = ("shares", "price", "fee", "net")

#: Tier name → rebate fraction, from the single source of truth in lib.math. Used by
#: the ``force_tier`` override (C — tier honesty: credit a clone at its OWNER's tier
#: regardless of the clone's own small 30-day volume).
_TIER_PCT_BY_NAME: dict[str, float] = {name: pct for _th, name, pct in lm.TAKER_TIERS}

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _forced_pct(force_tier: str) -> float:
    """Resolve a tier name to its rebate fraction, or raise with the valid names."""
    if force_tier not in _TIER_PCT_BY_NAME:
        raise ValueError(
            f"force_tier {force_tier!r} is not a known tier; "
            f"expected one of {sorted(_TIER_PCT_BY_NAME)}")
    return _TIER_PCT_BY_NAME[force_tier]


def _date_str(day: int) -> str:
    return (_EPOCH + timedelta(days=int(day))).strftime("%Y-%m-%d")


def _exec_day(df: pd.DataFrame) -> np.ndarray:
    """The UTC exec-day of each trade — from ``sample_ts_ms`` (fire time) when
    present, else the ``day`` column, else ``window_open_ms``."""
    if "sample_ts_ms" in df.columns:
        return (df["sample_ts_ms"].to_numpy(dtype=np.int64) // MS_PER_DAY).astype(np.int64)
    if "day" in df.columns:
        return df["day"].to_numpy(dtype=np.int64)
    if "window_open_ms" in df.columns:
        return (df["window_open_ms"].to_numpy(dtype=np.int64) // MS_PER_DAY).astype(np.int64)
    raise ValueError("trades.csv needs one of: sample_ts_ms, day, window_open_ms")


def compute_rebate_timeline(
    df: pd.DataFrame,
    *,
    weight: float = lm.CRYPTO_WV_WEIGHT,
    rolling_days: int = ROLLING_DAYS,
    min_payout: float = MIN_PAYOUT,
    force_tier: str | None = None,
) -> dict:
    """Pure taker-rebate simulation over a trades frame (no IO — unit-testable).

    Every row is treated as a taker fill (``fee`` already paid). Returns a dict with
    ``timeline`` (per-day rows), ``totals``, ``current`` (end-of-run tier + marginal
    wV), ``tier_days``, ``levelup``, and ``counts``.

    ``force_tier`` (C — tier honesty): when set to a tier name (e.g. ``"Platinum"``),
    every day is credited at that tier's rebate fraction regardless of the trades'
    own 30-day rolling volume — used to score a competitor CLONE at its OWNER's real
    tier. The ``wv_day``/``wv_30d`` timeline still reflects the clone's actual
    (earned) weighted volume, so the report shows the true volume alongside the forced
    rebate %. ``force_tier=None`` (default) keeps the earned-from-volume behavior.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"trades.csv missing required column(s): {missing}")
    forced_pct = _forced_pct(force_tier) if force_tier is not None else None
    if len(df) == 0:
        return {
            "timeline": [], "counts": {"n_trades": 0, "n_days": 0},
            "totals": {"total_wv": 0.0, "total_fees": 0.0, "original_net": 0.0,
                       "total_rebate": 0.0, "total_paid": 0.0, "residual_unpaid": 0.0,
                       "corrected_net": 0.0},
            "current": {"wv_30d": 0.0, "tier": force_tier or "None",
                        "rebate_pct": forced_pct or 0.0,
                        "next_tier": None if force_tier else "Bronze",
                        "wv_to_next_tier": 0.0 if force_tier else 2_000.0},
            "tier_days": {}, "levelup": {"total_bonus": 0.0, "events": []},
            "force_tier": force_tier,
        }

    days = _exec_day(df)
    shares = df["shares"].to_numpy(dtype=float)
    price = df["price"].to_numpy(dtype=float)
    fee = df["fee"].to_numpy(dtype=float)
    net = df["net"].to_numpy(dtype=float)
    wv = shares * (1.0 - price) * float(weight)  # per-trade weighted volume

    # Per-day aggregates over the full contiguous day range (gaps → zero days).
    day_min, day_max = int(days.min()), int(days.max())
    span = range(day_min, day_max + 1)
    wv_by_day = {d: 0.0 for d in span}
    fee_by_day = {d: 0.0 for d in span}
    net_by_day = {d: 0.0 for d in span}
    n_by_day = {d: 0 for d in span}
    for d, w, f, n in zip(days.tolist(), wv.tolist(), fee.tolist(), net.tolist()):
        wv_by_day[d] += w
        fee_by_day[d] += f
        net_by_day[d] += n
        n_by_day[d] += 1

    timeline: list[dict] = []
    accrued = 0.0
    total_paid = 0.0
    levelup_reached: set[str] = set()
    levelup_total = 0.0
    levelup_events: list[dict] = []
    tier_days: dict[str, int] = {}
    for d in span:
        # Tier in effect *during* day d = the trailing `rolling_days` completed days
        # [d-rolling_days, d-1] (docs: recalculated daily, effective next update,
        # applies going forward — today's own volume does not set today's tier).
        wv_30d = sum(wv_by_day.get(x, 0.0) for x in range(d - rolling_days, d))
        if force_tier is not None:
            tier, pct = force_tier, forced_pct
        else:
            tier, pct = lm.taker_rebate_tier(wv_30d)
        rebate_day = pct * fee_by_day[d]
        accrued += rebate_day
        paid_today = 0.0
        if accrued >= min_payout:
            paid_today = accrued
            total_paid += accrued
            accrued = 0.0
        if tier != "None" and tier not in levelup_reached:
            levelup_reached.add(tier)
            bonus = LEVELUP_BONUS.get(tier, 0.0)
            levelup_total += bonus
            levelup_events.append({"date": _date_str(d), "tier": tier, "bonus": bonus})
        tier_days[tier] = tier_days.get(tier, 0) + 1
        timeline.append({
            "date": _date_str(d), "day": d, "trades": n_by_day[d],
            "wv_day": wv_by_day[d], "wv_30d": wv_30d, "tier": tier, "rebate_pct": pct,
            "fees_day": fee_by_day[d], "net_day": net_by_day[d],
            "rebate_day": rebate_day, "paid_today": paid_today,
        })

    # End-of-run "current" state = trailing `rolling_days` INCLUDING the last day
    # (the tier you'd be in going forward), for the marginal-to-next-tier headline.
    # When a tier is forced, the "current" tier reflects the forced tier (the wV is
    # still the real earned volume, so the report shows both honestly).
    wv_30d_current = sum(wv_by_day.get(x, 0.0) for x in range(day_max - rolling_days + 1, day_max + 1))
    if force_tier is not None:
        tier_cur, pct_cur = force_tier, forced_pct
        next_tier, wv_to_next = None, 0.0
    else:
        tier_cur, pct_cur = lm.taker_rebate_tier(wv_30d_current)
        next_tier, wv_to_next = lm.taker_rebate_next_tier(wv_30d_current)

    total_wv = sum(wv_by_day.values())
    total_fees = sum(fee_by_day.values())
    original_net = sum(net_by_day.values())
    total_rebate = sum(t["rebate_day"] for t in timeline)
    corrected_net = original_net + total_rebate
    return {
        "timeline": timeline,
        "counts": {"n_trades": int(len(df)), "n_days": len(timeline),
                   "date_min": _date_str(day_min), "date_max": _date_str(day_max)},
        "totals": {
            "total_wv": total_wv, "total_fees": total_fees, "original_net": original_net,
            "total_rebate": total_rebate, "total_paid": total_paid,
            "residual_unpaid": accrued, "corrected_net": corrected_net,
            "rebate_fraction_of_fees": (total_rebate / total_fees) if total_fees > 0 else 0.0,
        },
        "current": {
            "wv_30d": wv_30d_current, "tier": tier_cur, "rebate_pct": pct_cur,
            "next_tier": next_tier, "wv_to_next_tier": wv_to_next,
        },
        "tier_days": tier_days,
        "levelup": {"total_bonus": levelup_total, "events": levelup_events},
        "force_tier": force_tier,
    }


def _verdict(result: dict) -> str:
    t = result["totals"]
    c = result["current"]
    peak = _peak_tier(result["tier_days"])
    if t["total_rebate"] <= 0.0 and peak == "None":
        need = c["wv_to_next_tier"]
        return (f"The taker rebate is worth $0 on this tape — 30-day weighted volume never "
                f"reached the ${2000:,.0f} Bronze threshold. Ending 30-day wV is "
                f"${c['wv_30d']:,.0f}; ${need:,.0f} more would qualify for {c['next_tier']} (3%).")
    parts = [
        f"The taker rebate adds ${t['total_rebate']:,.2f} "
        f"({t['rebate_fraction_of_fees']:.0%} of the ${t['total_fees']:,.2f} in taker fees), "
        f"lifting net PnL from ${t['original_net']:,.2f} to ${t['corrected_net']:,.2f}.",
        f"Peak tier reached: {peak}. Ending 30-day wV ${c['wv_30d']:,.0f} → tier {c['tier']} "
        f"({c['rebate_pct']:.0%}).",
    ]
    if c["next_tier"] is not None:
        parts.append(f"To reach {c['next_tier']}, ${c['wv_to_next_tier']:,.0f} more 30-day wV is needed.")
    if result["levelup"]["total_bonus"] > 0.0:
        parts.append(f"(Plus ${result['levelup']['total_bonus']:,.0f} in one-time level-up "
                     f"bonuses, not counted in the rebate total.)")
    return " ".join(parts)


def _peak_tier(tier_days: dict) -> str:
    """The highest tier that appears in the timeline (by tier rank)."""
    rank = {t[1]: i for i, t in enumerate(lm.TAKER_TIERS)}
    reached = [t for t in tier_days if t in rank]
    return max(reached, key=lambda t: rank[t]) if reached else "None"


def rebate_sim(
    paths,
    *,
    trades,
    out_name: str = "rebate_sim",
    weight: float = lm.CRYPTO_WV_WEIGHT,
    rolling_days: int = ROLLING_DAYS,
    min_payout: float = MIN_PAYOUT,
    force_tier: str | None = None,
) -> dict:
    """Read a ``trades.csv``, simulate the taker rebate, write the report, return
    the metrics dict.

    ``force_tier`` credits every day at a fixed tier (see
    :func:`compute_rebate_timeline`) — used to run a clone at its owner's tier.
    """
    trades_path = Path(trades)
    if not trades_path.exists():
        raise ParquetNotReady(  # reuse the clear missing-input error type
            f"trades.csv not found at {trades_path} — run backtest_momentum "
            f"(or money_judge for the model-taker) to produce one first.")
    df = pd.read_csv(trades_path)
    result = compute_rebate_timeline(df, weight=weight, rolling_days=rolling_days,
                                     min_payout=min_payout, force_tier=force_tier)
    tier_note = f" (forced {force_tier} tier)" if force_tier else ""
    result["params"] = {"trades": str(trades_path), "weight": weight,
                        "rolling_days": rolling_days, "min_payout": min_payout,
                        "force_tier": force_tier,
                        "title": f"Taker rebate — {trades_path.parent.name or out_name}{tier_note}"}
    result["verdict"] = _verdict(result)

    out_dir = paths.out_dir / "backtests" / "rebate_sim" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(out_dir, result)
    return result


def _timeline_chart(timeline: list[dict]) -> str | None:
    if not timeline:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = list(range(len(timeline)))
    wv = [t["wv_30d"] for t in timeline]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(xs, wv, color="#1f77b4", linewidth=1.2, label="30-day wV")
    ax.fill_between(xs, wv, color="#1f77b4", alpha=0.12)
    top = max(wv) if wv else 0.0
    for threshold, name, _pct in lm.TAKER_TIERS[1:]:
        if threshold <= max(top * 1.15, 1.0):
            ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(0, threshold, f" {name} ${threshold:,.0f}", va="bottom", ha="left",
                    fontsize=6, color="#a01818")
    step = max(1, len(timeline) // 10)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([timeline[i]["date"] for i in xs[::step]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("30-day weighted volume $")
    ax.set_title("Rolling 30-day wV vs tier thresholds")
    ax.legend(fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    import base64
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_outputs(out_dir: Path, result: dict) -> None:
    pd.DataFrame(result["timeline"]).to_csv(out_dir / "tier_timeline.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=eh._json_default), encoding="utf-8")

    t = result["totals"]
    c = result["current"]
    summary_rows = [
        {"metric": "trades", "value": f"{result['counts']['n_trades']:,}"},
        {"metric": "days", "value": f"{result['counts']['n_days']:,}"},
        {"metric": "total taker fees", "value": f"${t['total_fees']:,.2f}"},
        {"metric": "total rebate (gross)", "value": f"${t['total_rebate']:,.2f}"},
        {"metric": "total rebate paid ($1 floor)", "value": f"${t['total_paid']:,.2f}"},
        {"metric": "rebate as % of fees", "value": f"{t['rebate_fraction_of_fees']:.1%}"},
        {"metric": "original net PnL", "value": f"${t['original_net']:,.2f}"},
        {"metric": "corrected net PnL (net + rebate)", "value": f"${t['corrected_net']:,.2f}"},
        {"metric": "ending 30-day wV", "value": f"${c['wv_30d']:,.0f}"},
        {"metric": "ending tier", "value": f"{c['tier']} ({c['rebate_pct']:.0%})"},
        {"metric": "marginal wV to next tier",
         "value": ("—" if c["next_tier"] is None
                   else f"${c['wv_to_next_tier']:,.0f} to {c['next_tier']}")},
        {"metric": "one-time level-up bonuses (informational)",
         "value": f"${result['levelup']['total_bonus']:,.0f}"},
    ]
    tier_rows = [{"tier": name, "days": result["tier_days"].get(name, 0)}
                 for _th, name, _pct in lm.TAKER_TIERS if result["tier_days"].get(name, 0)]
    tl_cols = [("date", "date"), ("trades", "trades"), ("wv_day", "wv day $"),
               ("wv_30d", "30d wV $"), ("tier", "tier"), ("rebate_pct", "rebate %"),
               ("fees_day", "fees $"), ("rebate_day", "rebate $")]
    chart = _timeline_chart(result["timeline"])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{result['params']['title']}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 980px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #eee; }}
 table {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.92rem; }}
 td, th {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
 th {{ color: #555; font-weight: 600; }} .muted {{ color: #888; }} img {{ max-width: 100%; }}
 .verdict {{ background: #f6f8fa; border-left: 4px solid #1f77b4; padding: 0.8rem 1rem;
             border-radius: 4px; line-height: 1.5; }}
</style></head><body>
<h1>{result['params']['title']}</h1>
<p class="muted">Simulated Polymarket Taker Fee Rebate Program (tier% × taker fees; 30-day
rolling weighted volume). Source: <code>{result['params']['trades']}</code>. This is a
projection — the rebate is real income only once these volume tiers are actually reached.</p>
<div class="verdict">{result['verdict']}</div>
<h2>Summary</h2>
{eh._table_html(summary_rows, [("metric", "metric"), ("value", "value")])}
{eh._img_tag(chart, 'rolling 30-day wV vs tiers')}
<h2>Days in each tier</h2>
{eh._table_html(tier_rows, [("tier", "tier"), ("days", "days")])}
<h2>Tier timeline (per day)</h2>
{eh._table_html(result["timeline"], tl_cols)}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = stage_parser(__doc__ or "rebate_sim")
    parser.add_argument("--trades", default=None,
                        help="path to a trades.csv (default: the momentum-verdict baseline "
                             "out/backtests/momentum_current_regime/vps255/trades.csv)")
    parser.add_argument("--out-name", default="momentum",
                        help="output subdir under out/backtests/rebate_sim/ (default 'momentum')")
    parser.add_argument("--weight", type=float, default=lm.CRYPTO_WV_WEIGHT,
                        help=f"category weight for wV (default crypto {lm.CRYPTO_WV_WEIGHT})")
    parser.add_argument("--rolling-days", type=int, default=ROLLING_DAYS,
                        help=f"rolling window for tier determination (default {ROLLING_DAYS})")
    parser.add_argument("--force-tier", default=None, choices=sorted(_TIER_PCT_BY_NAME),
                        help="credit every day at this fixed tier regardless of earned "
                             "volume (tier honesty — score a clone at its owner's tier)")
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    trades = (Path(args.trades) if args.trades
              else paths.out_dir / "backtests" / "momentum_current_regime" / "vps255" / "trades.csv")
    print(f"[rebate_sim] trades={trades} out={paths.out_dir / 'backtests' / 'rebate_sim' / args.out_name}")
    try:
        result = rebate_sim(paths, trades=trades, out_name=args.out_name,
                            weight=args.weight, rolling_days=args.rolling_days,
                            force_tier=args.force_tier)
    except ParquetNotReady as exc:
        print(f"[rebate_sim] {exc}")
        return 1
    t = result["totals"]
    c = result["current"]
    print(f"[rebate_sim] {result['counts']['n_trades']:,} trades over {result['counts']['n_days']} days: "
          f"rebate ${t['total_rebate']:,.2f} ({t['rebate_fraction_of_fees']:.0%} of ${t['total_fees']:,.2f} fees), "
          f"net ${t['original_net']:,.2f} -> ${t['corrected_net']:,.2f}")
    print(f"[rebate_sim] ending tier {c['tier']} ({c['rebate_pct']:.0%}), 30d wV ${c['wv_30d']:,.0f}"
          + (f", ${c['wv_to_next_tier']:,.0f} to {c['next_tier']}" if c["next_tier"] else " (top tier)"))
    print(f"[rebate_sim] wrote report.html + metrics.json + tier_timeline.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
