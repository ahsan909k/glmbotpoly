# BTC maker-quoting gap — read-only diagnosis

**Date:** 2026-07-14 · **Scope:** why the maker core places ~0 resting quotes on BTC
series · **Method:** code trace + read-only counts over the live `bot run`'s logs
(`data/logs/bot.2026-07-14.log`) and journal (`data/journal/journal-20260714-05*.jsonl.gz`).
No code or config changes.

---

## Verdict (one line)

BTC gets ~0 maker quotes because the maker core runs **one global `QuoteManager` with a
single `active` window slot**, and every `WindowLifecycle::Open` — for any of the 6
series — unconditionally overwrites it. Window-open events are emitted in `Series::ALL`
order (**all BTC series before all ETH series**), so an ETH window always overwrites the
coincident BTC window at every shared boundary. Only the most-recently-opened window
quotes. **It is not BTC-specific, not config, not model health** — it is the documented
single-instance-engine limitation. The accurate framing is: *5 of 6 series place ~0; only
ETH-5m (and briefly ETH-15m) ever quote.*

---

## Evidence (live run, 2026-07-14)

**1. Placements are ~0 for BTC and concentrated in ETH-5m.**
Journal `order_update` records carry `window.series` directly (the plaintext log's
`placed` lines do not — they only carry `outcome/level/price`, so the log alone cannot
split by series). Counting `state=Open` (a placed, resting order) over a ~12-minute sample
(05:32–05:44 UTC, segments `…00168`, `…00169`):

| Series  | Placements (`state=Open`) | Total order_updates (any state) |
|---------|---------------------------|---------------------------------|
| ETH-5m  | **482**                   | 1002                            |
| ETH-15m | **137**                   | 285                             |
| BTC-5m  | **0**                     | 3                               |
| BTC-15m | **0**                     | 1                               |
| BTC-1h  | **0**                     | 0                               |
| ETH-1h  | **0**                     | 0                               |

Overall states in the sample: Open 619, Canceled 630, Rejected 32, Filled 8,
PartiallyFilled 2. BTC ≈ 0 across the board; even ETH-1h is 0.

**2. Opens are perfectly symmetric — BTC is *announced* ready every window but never
places.** Every series logs `quote-manager: window open: ready to quote` on its cadence.
Over the current 6h log:

| Series  | `window open: ready to quote` |
|---------|-------------------------------|
| BTC-5m / ETH-5m  | 72 / 72 |
| BTC-15m / ETH-15m | 24 / 24 |
| BTC-1h / ETH-1h  | 6 / 6 |

So the calculator never rejects BTC with a reason — BTC simply never becomes the active
window. A starved series is therefore **silent** in the `quote-manager` target (no
`NoQuote` line exists to grep), which is itself the diagnostic signal.

**3. Smoking gun — emission order at a shared boundary.** The scheduler announces opens in
`Series::ALL` order, ~3 ms apart; the last one overwrites the single slot:

```
00:00:00.027  window=BTC-5m     ← overwritten
00:00:00.031  window=BTC-15m    ← overwritten
00:00:00.034  window=BTC-1h     ← overwritten
00:00:00.037  window=ETH-5m     ← overwritten
00:00:00.040  window=ETH-15m    ← overwritten
00:00:00.043  window=ETH-1h     ← holds the slot
```

ETH is always emitted after the coincident BTC window, so ETH always wins. The placement
mix (ETH-5m ≫ ETH-15m ≫ BTC≈0, ETH-1h=0) is exactly what the mechanism predicts: the
frequently-reopening ETH-5m re-grabs the slot at every 5-minute boundary and holds it most
of the time; ETH-15m holds it only in the ~5-minute stretch after a :15/:30/:45 boundary
until the next 5-minute open steals it back; ETH-1h and every BTC window are clobbered
before they can place.

**4. Model health is symmetric across assets — rules out any feed/model asymmetry.**
Same segment, `model` records by `asset`:

| Asset | Ready | Unreliable | Reason (Unreliable) |
|-------|-------|-----------|---------------------|
| Btc   | 11111 | 285       | `ChainlinkStale` |
| Eth   | 11107 | 285       | `ChainlinkStale` |

~97.5% Ready for both, and the degraded slice is identical (`ChainlinkStale`, 285 each).
BTC's model is no less quotable than ETH's.

---

## Root cause (code)

- `crates/engine/src/quote_manager/driver.rs:98` — `active: Option<ActiveWindow>`, a
  **single** slot for the whole bot (the manager doc states it "trades one active window
  at a time").
- `crates/engine/src/quote_manager/driver.rs:264-268` — `WindowLifecycle::Open` ⇒
  `self.active = Some(ActiveWindow::new(...))` + the `window open: ready to quote` log,
  **unconditionally, for every series**. Any prior active window is dropped.
- `crates/engine/src/quote_manager/driver.rs:316-318` — `on_model` drops any snapshot
  whose window ≠ the single active one, so a starved series never even marks `dirty`
  (never triggers a requote).
- `crates/core-types/src/series.rs:97` — `Series::ALL` = `[BTC-5m, BTC-15m, BTC-1h,
  ETH-5m, ETH-15m, ETH-1h]`; the scheduler drives its six machines in this order, so opens
  are emitted BTC-before-ETH.
- `crates/engine/src/quoting.rs` — the pure `calculate_quotes` gate ladder (G1–G5, S1–S8)
  never runs for a starved series and logs nothing.
- **Known deferred limitation** (Decisions Log): the model taker was refactored to
  per-window (`HashMap<WindowId, ModelWindowState>`) while "momentum/late stay
  single-active" and "only the model taker is per-window"; the deferred item reads "fixing
  the single-instance limitation for momentum/late/quoter." This audit is that item's
  concrete cost, measured.

---

## What it is NOT

- **Not config.** BTC is fully enabled — `[engine.series.*]` is empty, so all six series
  run with byte-identical `EngineParams`. There are zero BTC-specific overrides.
- **Not model health / feed.** Health is ~97.5% Ready and identical across assets;
  `ChainlinkStale` hits BTC and ETH equally (see table above).
- **Not price magnitude.** Every gate is scale-free — EWMA of *log* returns, divergence in
  bps (`|ln(cf/cl)|·10⁴`), log-moneyness `z = ln(S/K)/(σ√τ)`, ms-based tolerances. BTC's
  ~$100k vs ETH's ~$3k underlying is mathematically invisible to every gate.
- **Not a calculator gate (G1–G5 / S1–S8).** Those never fire for BTC because BTC is never
  the active window; there is no BTC `NoQuote`/`Suppressed` reason to find.

---

## Secondary factor (asset-agnostic)

Global breakers pull **all** quoting during their trips (current 6h log: FeedStale 269,
FairVsMid 378, plus WsDisconnect) — the documented home-connection jitter. This reduces
even the slot-holder's (ETH-5m) quoting but does not cause the BTC-specific starvation; it
hits every series equally. A colocated VPS would remove most of it, but would not change
the single-slot behavior.

---

## Cost of not quoting BTC

Primarily an **evaluation-validity** cost, not a headline dollar loss:

- The §10.2 per-series maker decision table — the entire point of the paper eval (pick the
  best 2–3 series by maker markout / PnL / % profitable) — is **unachievable** for
  BTC-5m/15m/1h and ETH-1h (≈ 0 maker sample), partial for ETH-15m, and complete only for
  ETH-5m. The maker-side comparison that would justify enabling live making cannot be
  produced from this run.
- The dollar impact is small: maker-core P&L is marginal in this project (the eval found
  momentum is the only clearly-profitable current-regime strategy), and the takers run on
  separate paths — the **model taker is already per-window and covers all four M5/M15
  series**, and momentum/late are single-active like the maker. So the missed profit is
  the maker two-sided edge on ~5 series-windows' worth of coverage per instant, which is
  minor next to the taker P&L.
- Extrapolated over the whole run, **BTC maker fills ≈ 0**.

---

## Fix: one-line vs code

- **One-line / config mitigation (does NOT restore the goal).** Temporarily narrow
  `engine.series` to a single series (or run one bot instance per series) so that series
  quotes cleanly. This only moves *which* single series holds the slot — it cannot produce
  a *concurrent* 6-series maker comparison.
- **Proper fix (code, non-trivial).** Make the maker `QuoteManager` per-window — a
  `HashMap<WindowId, ActiveWindow>` mirroring the model-taker's per-window refactor
  (`crates/engine/src/model_taker/driver.rs`) — plus wiring per-window requote ticks,
  params, and the per-window risk gate. The same refactor would also un-starve the
  single-active momentum/late takers if concurrent per-series coverage of those is wanted.
  This matches the already-logged deferred item ("fixing the single-instance limitation for
  momentum/late/quoter").

---

## Reproduce

```sh
# Per-series placements (BTC ≈ 0, ETH-5m dominant):
gzip -dc data/journal/journal-20260714-05*.jsonl.gz \
  | grep '"type":"order_update"' | grep '"state":"Open"' \
  | grep -oE '"series":"[^"]+"' | sort | uniq -c

# Model health symmetric across assets:
gzip -dc data/journal/journal-20260714-053850-00169.jsonl.gz \
  | grep '"type":"model"' | grep '"asset":"Btc"' \
  | grep -oE '"health":"[^"]+"' | sort | uniq -c   # repeat with "Eth"

# Boundary emission order (ETH always last → wins the single slot):
grep 'window open: ready to quote' data/logs/bot.2026-07-14.log \
  | grep 'T00:00:0' | sed -E 's/.*(window=[^ ]+).*/\1/'
```
