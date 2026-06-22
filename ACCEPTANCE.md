# ACCEPTANCE.md — Paper-evaluation protocol

How we decide the bot is good enough, and how we pick the final **2–3 series** to
keep trading. This is the paper phase: real Polymarket market data, simulated
money. Everything below is measured from the bot's own journal + analytics — no
external spreadsheet.

> Companion docs: [`LIVE_PREFLIGHT.md`](LIVE_PREFLIGHT.md) (the go-live sequence,
> after this protocol passes) and `CLAUDE.md` §8–§11 (the strategy/risk spec the
> gates trace back to).

---

## 0. What we are evaluating

The six series — **BTC & ETH × {5m, 15m, 1h}** — run concurrently in paper under
`bot run`. The goal of the evaluation is **not** "is the bot profitable in
aggregate"; it is **"which 2–3 series earn a real, risk-clean edge"**, so we can
narrow the live universe to those.

Run it with:

```bash
bot run                         # all six series, paper, real data
# dashboard live at http://127.0.0.1:8080  (series-comparison view is the decision table)
```

Let it accumulate the sample below, then read the gates and the series-comparison
table. Everything is journaled (`data/journal/*.jsonl.gz` + `data/journal.sqlite`),
so the evaluation is reproducible (`bot replay` / `bot sweep`) and survives restarts.

---

## 1. Sample-size requirement

**Minimum 300 settled windows per candidate series, spanning varied volatility
regimes.** A series with fewer than that does not get selected, no matter how good
its numbers look — small samples lie.

`windows_traded` is one increment per **settled** window
(`analytics::rollup::DailyRollup::fold`), aggregated per series. The wall-clock
cost differs sharply by duration:

| Series | Windows/day | Calendar time for 300 windows |
|---|---|---|
| BTC-5m / ETH-5m | ~288 | ~1.1 days |
| BTC-15m / ETH-15m | ~96 | ~3.2 days |
| BTC-1h / ETH-1h | ~24 | ~12.5 days |

**"Varied volatility regimes"** means the 300+ windows must not all come from one
flat afternoon. Span at least a few distinct crypto sessions — include both a calm
stretch and a genuinely volatile one (a CPI print, a large BTC move, a weekend).
Adverse selection (the core failure mode, §8) only shows up when the underlying
moves, so an all-calm sample tells you nothing about the live downside.

The `min_sample_warning` flag on each series-comparison row trips below
`AnalyticsParams.min_sample_windows` (**default 30**) — that is the *floor* below
which a row is untrustworthy, **not** the 300-window selection bar. Treat any row
still showing the warning as "no data".

---

## 2. The four gates

A series is a **selection candidate** only if it passes all four. (G3 and G4 are
whole-bot invariants — if they fail, no series is selectable until the cause is
fixed.)

### G1 — Average 5-second markout above the configured floor

*The adverse-selection gate: are our passive fills systematically picked off?*

- Source: `analytics::health::AdverseSelectionMonitor` →
  `AdverseSelectionState ∈ {InsufficientSample, Ok, Alarm}`.
- The rolling mean of the last `window` (**default 200**) passive-fill 5-second
  markouts must stay **≥ `negative_threshold`** (**default 0.0** — this is "the
  configured floor"), evaluated only after `min_sample` (**default 50**) markouts.
- **Pass for a series:** its series-comparison `health == Ok` (never `Alarm`) and
  `avg_markout_5s ≥ 0.0`. Inspect the full `markout_5s` distribution
  (mean/stddev/p50/p95/histogram) — a positive mean dragged up by a fat right tail
  while p50 is negative is a yellow flag, not a clean pass.
- A persistent `Alarm` means we are being adversely selected on that series: drop
  it (or widen its edge / shorten holding time in config and re-evaluate).

### G2 — Majority of windows flat-or-winner-skewed at close

*Are we holding the wrong side into resolution?*

- Source: `core_types::SettlementSummary { excess, outcome, .. }` (one per settled
  window; queryable from the journal `settlements` table). `excess` is **signed**:
  `> 0` = Up-side unmatched shares, `< 0` = Down-side.
- Classify each window at close:
  - **flat** — `|excess| ≈ 0` (matched book; no directional bet left on),
  - **winner-skewed** — `excess` sign matches `outcome` (the unmatched bet won),
  - **loser-skewed** — `excess` sign opposes `outcome` (the unmatched bet lost).
- **Pass for a series:** **> 50%** of its windows are flat-or-winner-skewed (i.e.
  loser-skewed is a minority). Systematic loser-skew means inventory accumulates on
  the side the underlying is leaving — the inventory-skew / pair-discipline knobs
  (§8) are mis-tuned for that series.

### G3 — Zero risk-invariant violations

*Did the safety system ever actually fail?* (whole-bot)

The §11 invariants: no orphaned open orders; every breaker trip cancels-all and
halts/resumes per the rules; clean rebuild from the journal. Two evidences:

1. **Formal proof — the chaos suite is green:**
   `cargo test -p bot --features chaos --test chaos -- --test-threads=1`, run ~10×,
   100% green (kill each WebSocket, stall each feed, engine restart, clock jump,
   discovery failure at rollover, process restart — each asserts no orphan, the
   right breaker with a journaled cause, correct halt/resume, exact rebuild).
2. **Session review — the journal shows no breach:** read every breaker trip:
   ```sql
   SELECT seq, ts_local_ms, kind, breaker FROM breaker_trips WHERE kind='tripped' ORDER BY seq;
   ```
   (or `JournalIndexReader::breaker_trips()`). **Zero violations ≠ zero trips** — a
   legitimate environmental trip (e.g. an RTDS feed stall) is fine **iff** it
   cancelled-all, halted, and later cleared with **no orphaned order**. A trip with
   no paired clear, or an order left resting through a halt, is a violation and
   fails the gate.

### G4 — Attribution components sum to ledger PnL

*Do the books balance, exactly?* (whole-bot)

- Source: `analytics::attribution`. The identity is exact in `Decimal`:
  - `trading_sum() == realized_pnl` (buckets: `locked_pair_pnl + excess_pnl +
    settlement_remainder + taker_fees`), and
  - `bucket_sum() == realized_pnl + estimated_rebate` (adds the separately-credited
    maker rebate).
- Guaranteed by construction and pinned by tests
  (`random_sequences_reconcile_to_the_ledger`, `balanced_buy_only_remainder_zero_and_sums_to_realized`).
- **Pass:** the per-series locked-pair-vs-inventory split shown in the comparison
  table reconciles to net PnL. A mismatch is a **bug**, not a tuning issue — stop
  and fix before trusting any other number.

---

## 3. Reading the series-comparison table to pick the final 2–3

The decision table is the §10 series-comparison view (dashboard `Series
comparison`, or `bot replay` over the captured journal). Each
`analytics::rollup::SeriesComparisonRow` carries:

| Column | Read it for |
|---|---|
| `windows_traded` | sample size — **must be ≥ 300** (§1); ignore rows with `min_sample_warning` |
| `health` | **must be `Ok`** (G1) |
| `net_pnl`, `pnl_per_window` | the headline — rank by `pnl_per_window` (size-normalized) |
| `fraction_profitable` | consistency — a high mean from one lucky window is fragile |
| `locked_pair_pnl` vs `inventory_pnl` | **where** the PnL comes from — locked-pair (maker edge) is durable; large positive `inventory_pnl` is directional luck and will mean-revert |
| `avg_markout_5s` + `markout_5s` dist | adverse-selection margin (G1); prefer a positive p50, not just a positive mean |
| `fees_paid` vs `rebates_earned` | net fee drag; on crypto, maker rebates offset some taker fees |
| `maker_fill_fraction`, `taker_notional`, `taker_budget_used_fraction` | how much edge came from making vs taking; heavy taker reliance is costlier and budget-bounded |

**Selection procedure**

1. Sort by `SortColumn::PnlPerWindow` (or `NetPnl`) descending.
2. Keep only rows with `windows_traded ≥ 300`, `health == Ok`, and
   `pnl_per_window > 0`.
3. Among those, prefer series whose PnL is **locked-pair-dominated** (durable maker
   edge) over inventory-dominated (directional), with a positive markout p50 and a
   high `fraction_profitable`.
4. Pick the top **2–3**. Note BTC vs ETH and the duration mix — two series that
   win for the same reason are less diversifying than two that win differently.

---

## 4. Sign-off checklist

The paper evaluation is **complete** when, for each of the chosen 2–3 series:

- [ ] `windows_traded ≥ 300`, sample spans ≥ 2 distinct volatility regimes, `min_sample_warning == false`
- [ ] **G1** `health == Ok` and `avg_markout_5s ≥ 0.0`
- [ ] **G2** > 50% of windows flat-or-winner-skewed at close
- [ ] **G3** chaos suite green (~10×) **and** journal review shows zero invariant breaches
- [ ] **G4** attribution reconciles to ledger PnL (exact)
- [ ] `pnl_per_window > 0`, PnL is locked-pair-dominated, `fraction_profitable` acceptable

Only then proceed to [`LIVE_PREFLIGHT.md`](LIVE_PREFLIGHT.md).
