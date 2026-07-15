# Recovery fingerprint audit — model-taker hedge → simple revert

**Question:** is the recovered bot behaviorally identical to the last known-good baseline (original simple/theta model-taker)?
**Date:** 2026-07-13 (~14:50 UTC). **Method:** evidence from journals, decision journal, boot-config logs, binary mtimes, and byte-level file diffs. Read-only; no production code touched.

---

## VERDICT: ✅ Identical baseline

The recovered bot — **source, config, and the currently-running process** — is behaviorally identical to the pre-hedge baseline. The hedge experiment is fully and cleanly undone. The only current-vs-baseline differences are:

1. **Two intended config edits** — `feeds.binance_stale_after_ms` 2500→6000 and `feeds.rtds_stale_after_ms` 5000→10000 (operator-dated 07-13, unrelated to the hedge). `model_taker` + `shadow` config: **0 diff**.
2. **Short-sample / market-drift noise** in absolute fill counts (the "now" trading window is only ~42 min).

Everything that defines the model-taker's behavior matches; nothing that differs is a recovery defect.

| Axis | Result |
|---|---|
| **Source code** (9 files) | Content-identical to the transcript original (7 byte-identical; 2 differ only in CRLF↔LF, on files the hedge never touched) |
| **Config** (model_taker + shadow) | **0 diff** vs pre-hedge baseline; all 9 hedge params cleanly removed |
| **Running binary** | The reverted-simple build (07:58:46 UTC), loaded by the current 14:09 boot — recovered code is **live** |
| **Model-taker decision fingerprint** | Current run matches original/reverted-simple (~2% fire-rate, same series skew, same reason families, **zero hedge markers**); hedge period is clearly distinct |
| **Engine fingerprint** | Cancel ratio ~1.007 identical across every period; same breaker type-mix; same BTC/ETH quoting structure |

---

## Timeline (all UTC; filesystem mtimes are displayed +0800, i.e. local = UTC+8)

| UTC | Event | Evidence |
|---|---|---|
| 03:23:43 | build (hedge-era binary) | `scratchpad/build.log` mtime |
| 03:25:23 | boot → **original-simple** run | boot config: simple schema, theta 0.03, no hedge fields |
| 04:38–07:19 | 6 boots → **hedge** experiment | boot configs carry `conf_high/mid/low`, `size_*`, `max_imbalance`, `pair_cost_cap`, `hedge_timeout_ms` |
| 07:30:59 | hedge code backed up | `scratchpad/mt_backup_current/` mtime |
| 07:49:31 | **revert** to original-simple | `scratchpad/mt_restore/` mtime |
| 07:58:46 | **rebuild** (reverted-simple binary) | `target/debug/bot.exe` mtime |
| 07:59:23 | boot (+37 s) → reverted-simple run #1 | boot config: simple schema |
| 14:09:45 | boot → **current run** (same reverted binary) | boot config: simple schema; no rebuild since 07:58 |
| ~14:51 | now — actively trading | `botrun.log` live tail (quote-manager placing/cancelling) |

The revert (07:49) → rebuild (07:58) → the 07:59 boot loaded the reverted binary, and the current 14:09 boot loaded the same one. **The running process is the recovered code.** (An earlier read mistook the +0800 filesystem display for UTC and suggested the process was pre-rebuild; corrected here — bot.exe's build at **07:58:46 UTC** precedes both simple boots.)

---

## Part 1 — Config diff (from `data/logs/bot.*.log` boot snapshots)

### 1a. Pre-hedge baseline (03:25) → now (14:09) — the recovery-cleanliness test
Full effective config differs on **2 keys only**, both intended feeds edits; **`model_taker` and `shadow`: 0 differing keys.**

| Key | baseline (03:25) | now (14:09) | Nature |
|---|---|---|---|
| `feeds.binance_stale_after_ms` | 2500 | 6000 | intended (07-13 edit) |
| `feeds.rtds_stale_after_ms` | 5000 | 10000 | intended (07-13 edit) |
| `model_taker.*`, `shadow.*` | — | — | **identical** |

### 1b. What the revert removed (hedge boot 05:30 → now) — evidence the hedge is gone
11 keys; every hedge-only field is **absent** now, `theta` restored, `budget` restored:

| Key | hedge value | now | |
|---|---|---|---|
| `conf_high` / `conf_mid` / `conf_low` | 0.8 / 0.7 / 0.6 | absent | removed |
| `size_high` / `size_mid` / `size_low` | 40 / 20 / 10 | absent | removed |
| `max_imbalance` | 40 | absent | removed |
| `pair_cost_cap` | 1 | absent | removed |
| `hedge_timeout_ms` | 40000 | absent | removed |
| `budget_per_window` | 50 | 10 | restored |
| `theta` | absent | 0.03 | restored |

Working-tree `config/default.toml`, `config/bot.local.toml`, and `config/sections/model_taker.rs` grep clean of all hedge keys. `bot.local.toml` sets only `[model_taker] enable = true`.

### 1c. 07-11 soak → now (context) — intended feature growth
14 differing keys, **all intended** and none hedge-related: the whole `model_taker.*` section (feature born 07-13), `paper.taker_rebate_enabled`, and the two feeds-staleness raises. No hedge params.

---

## Part 2 — Behavioral fingerprint

Substrate: `data/journal.sqlite` (validated **exactly complete** vs the gzip source — order_update 5917=5917, fill 18=18 for a sampled complete hour, +0.00%). Model-taker fires/reasons from the multi-member `data/model-taker/` decision journal (member-walk recovered 7129 records; the current-run count reconciles with ≈1 prediction/5 s/4 series).

### 2a. Model-taker decision fingerprint (the core recovery test)

| Window | n | fires (rate) | per-series fires (B5/B15/E5/E15) | hedge marker |
|---|---|---|---|---|
| (b-strict) original-simple 03:25 | 12 | 0 (warming) | 0/0/0/0 | 0 |
| **(b) reverted-simple #1 07:59** | 334 | 8 (2.4%) | 4/1/2/1 | **0** |
| **(c) NOW 14:09 (running)** | 1315 | 26 (2.0%) | 8/2/13/3 | **0** |
| (hedge) 04:38–07:19 | 5468 | 190 (3.5%) | 82/22/73/13 | **33 "below confidence floor"** |

- **(b) vs (c): a match.** Same ~2% fire-rate, same series skew (5m ≫ 15m, ETH-5m the top firer), same suppression-reason families (`taker budget exhausted` dominant, then `below theta`, `insufficient feature coverage`, `standing down`, `no window state`), and **no hedge markers** in either.
- **The hedge is clearly distinct:** higher 3.5% fire-rate, breaker-standdown-dominated, and it emits the hedge-only reason **"below confidence floor"** (33×, the `conf_low` gate) — absent from every simple run. This is the discriminator anchoring the baseline boundary. (Note: the hedge binary actually deployed at the 04:38 boot; the first decision-journal "below confidence floor" lags to ~05:35, so the config-schema boot boundary is the accurate hedge start.)

### 2b. Engine fingerprint (per window; makers/cancels/fills/breakers)

| Window | span | cancel ratio | fills/win | breaker trips (tripped/h) |
|---|---|---|---|---|
| (a) 07-11→12 soak | 24 h | **1.007** | 0.58 | feed_stale 48, fair_vs_mid 22, ws 3 |
| (b) orig pre-hedge | 1.2 h | 1.007 | 1.38 | fair_vs_mid 4, ws 1, feed_stale 1 |
| (hedge) | 3.3 h | 1.007 | 2.10 | feed_stale 14, fair_vs_mid 10, ws 3 |
| (b2) reverted #1 | 1.0 h | 1.008 | 1.87 | feed_stale 15, fair_vs_mid 15, ws 5 |
| **(c) NOW** | ~0.7 h | **1.009** | 2.12 | fair_vs_mid 6, ws 4, feed_stale 2 |

- **Cancel ratio ≈ 1.007–1.009 in every period** — the cancel-first repricing signature is invariant. Strongest engine-identity evidence.
- **Breaker type-mix is consistent** everywhere (feed_stale, fair_vs_mid, ws_disconnect, rare window_loss); no new breaker types, no clock_skew in recent windows. Per-hour rates vary with feed/market conditions (noise), not code.
- `fills/window` varies 0.58→2.12 — driven by taker activity and very small window counts (16–30 windows per short window); within market-drift/short-sample noise, not a behavioral divergence.

**Incidental (pre-existing, not a recovery artifact):** across **every** window including the 24 h soak, **BTC series place ~0 maker quotes** — only ETH series make markets (BTC still *takes*: taker fills present). This is stable across baseline and now, so it does not affect the recovery verdict, but it is worth the operator's separate attention (likely BTC model-health gating maker quoting).

---

## Part 3 — Source code diff

Working-tree model-taker files vs the transcript-reconstructed original (`scratchpad/mt_restore/`, from session `236dfbea`):

| File | vs original |
|---|---|
| `crates/engine/src/model_taker/mod.rs` | byte-identical |
| `crates/engine/src/model_taker/driver.rs` | byte-identical |
| `crates/engine/src/model_taker/edge.rs` | byte-identical |
| `crates/engine/src/arbitration.rs` | identical **modulo CRLF** (file the hedge never changed) |
| `crates/engine/src/self_match.rs` | identical **modulo CRLF** (file the hedge never changed) |
| `crates/bot/src/model_taker_record.rs` | byte-identical |
| `crates/config/src/sections/model_taker.rs` | byte-identical |
| `crates/dashboard/src/model_taker.rs` | byte-identical |
| `crates/bot/tests/model_taker_paper.rs` | byte-identical |

7 byte-identical; the 2 CRLF-only differences are a pre-existing `autocrlf` artifact on files the hedge did not touch (content identical). The subsystem was **never committed** (HEAD `d9c7727` predates it), so there was no git revert — recovery was a manual reconstruction, and it reproduced the original exactly.

**What the revert undid** (hedge backup `mt_backup_current/` → current, lines added/removed):

| File | delta |
|---|---|
| `model_taker/driver.rs` | +37 / −565 |
| `model_taker/mod.rs` | +30 / −171 |
| `model_taker/edge.rs` | +5 / −72 |
| `config/sections/model_taker.rs` | +9 / −97 |
| `config/default.toml` | +1 / −11 |

---

## Caveats / limits

- **Short "now" sample:** the current run is ~42 min old (14:09→14:51 UTC), actively trading. The strict pre-hedge original-simple decision window (03:25) is very thin (12 records, warming); the fuller like-for-like baseline is the 07:59 reverted-simple run — which is the *same binary* as now.
- **Momentum vs late fires** cannot be separated in the journal (no driver tag); reported as combined taker fills.
- **Market drift** between 07-11 and 07-13 — conclusions lean on ratios (cancel ratio, fire-rate, reason mix) and the hedge-marker discriminator, not raw rates.
- **sqlite** validated complete (+0.00% vs gzip); breaker counts filtered to `kind='tripped'`.
- Running binary identity is established by build time + config schema + fingerprint (not a binary hash) — but it is provably the reverted-simple build and no newer binary exists.
