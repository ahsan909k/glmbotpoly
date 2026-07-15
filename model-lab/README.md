# model-lab

Offline research over the Polymarket Up/Down trading bot's journal. A
**standalone Python project**, fully separate from the Rust trading engine — it
never runs in the hot path. It reads the bot's gzip JSONL journal (and the
Binance `depth20@100ms` capture), **validates** that the engine's fair-value math
reproduces offline, and **researches** whether L2 order-book microstructure
signals add predictive value.

> Why it exists: the engine's fair value feeds off Binance bookTicker mids +
> trades and Chainlink — it never looks at L2 depth. The bot now also journals
> Binance `depth20@100ms` (see the engine's `depth_capture` module,
> `feeds.binance_depth_capture`). This lab is where we find out whether that
> depth is worth wiring into the model.

---

## Setup (venv + pinned deps)

Requires **Python ≥ 3.10** (developed/verified on 3.12). From this directory
(`model-lab/`):

```bash
python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# Windows Git Bash:     source .venv/Scripts/activate
# macOS/Linux:          source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are exact-pinned in `requirements.txt` (numpy, pandas, pyarrow,
matplotlib, pytest — a minimal scientific stack; Φ uses stdlib `math.erfc` and
AUC is a rank-based numpy implementation, so no scipy/sklearn in the core). All
ship prebuilt Windows wheels for 3.12, so the core install compiles nothing.

The **second challenger** (`learn_gbt` / `compare`, gradient-boosted trees) needs
**LightGBM** — a heavier, compiled OpenMP library that pulls `scipy` — so it is an
**opt-in extra**, kept out of the core to preserve the lean install:

```bash
pip install -r requirements-gbt.txt      # (or: pip install -e .[gbt])
```

Without it the whole numbered pipeline, `verify`, and `pytest` run unchanged (the
GBT tests skip); only `python -m model_lab.learn_gbt` / `compare` require it.

> **Windows note.** If `python` opens the Microsoft Store (the "App execution
> alias" stub) or `py` isn't found, `python` isn't really on PATH yet. Create the
> venv with the real interpreter — e.g. `py -3.12 -m venv .venv`, or the full
> path
> `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv`.
> After that, activate the venv and plain `python` works (it points at
> `.venv\Scripts\python.exe`). Stage output is UTF-8 (Φ, σ, →); the lab forces a
> UTF-8 console so it renders on a legacy cp1252 code page too.

---

## Large journals: streaming + `--since` / `--until`

The bot's journal grows ~2.5 GB/day. The journal-reading stages (`dataset`,
`short_horizon`, `feature_set`, `ingest`, `calibration_audit`, `evaluate`, `learn`,
`learn_gbt`) read it in a **streaming** pass that never materializes the full raw
tick set — the ~100 ticks/s are reduced on the fly to what the feature grid + strike
actually read (per-second Binance mid bars + Chainlink), so memory grows with the
*time range*, not the tick rate. Every one of those stages accepts:

```bash
--since 2026-07-03          # process only records at/after this time (UTC)
--until 2026-07-04          # ... and strictly before this time
--max-rss-mb 3072           # advisory ceiling; the run reports peak RSS and warns if over
```

`--since` / `--until` take a date (`2026-07-03` → 00:00:00Z), an ISO datetime
(`2026-07-03T04:00:00`, UTC), or a raw unix-ms integer. Use them to bound a run to a
day/week on a very large journal; each journal-reading stage prints its **peak RSS**
against the ceiling. (The old full-tick path is still available for debugging via
`--no-stream`; it may OOM on a multi-day journal and is not the default.)

---

## Verify (do this first)

`verify` generates a tiny **synthetic** journal + depth capture into a temp
directory, runs the whole pipeline against it, and asserts the outputs — so you
can confirm the lab works with **zero** real data:

```bash
python -m model_lab.verify      # prints "VERIFY: PASS"
pytest -q                       # unit + smoke tests
```

The synthetic fixture is built so that, by construction, `p_up == Φ(z)` exactly,
`sigma_1s` is the engine EWMA of the emitted bars, and order-book imbalance
leads the price — so a green `verify` proves each stage's math and I/O.

---

## Data inputs

Paths resolve as **CLI flag → env var → repo-relative default**:

| input        | default             | env var                   | flag             |
|--------------|---------------------|---------------------------|------------------|
| journal      | `../data/journal`   | `MODEL_LAB_JOURNAL_DIR`   | `--journal-dir`  |
| depth        | `../data/depth`     | `MODEL_LAB_DEPTH_DIR`     | `--depth-dir`    |
| aggTrades    | `../data/aggtrades` | `MODEL_LAB_HIST_DIR`      | `--hist-dir`     |
| telonex      | `../data/telonex`   | `MODEL_LAB_TELONEX_DIR`   | `--telonex-dir`  |
| output       | `./out`             | `MODEL_LAB_OUT`           | `--out`          |

- The **journal** (`journal-*.jsonl.gz`) is written by `bot record` / `bot run`.
- The **depth** capture (`binance-depth20-*.jsonl.gz`) appears once you run
  `bot record` (or `bot run`) with `feeds.binance_depth_capture = true`
  (the default). Until then the research stage runs but reports "no depth data".
- For a very large journal, point `--journal-dir` at a subset (segments are
  chronological, so copy the ones for the window you care about).

> A queryable `data/journal.sqlite` index also exists alongside the gzip
> segments; the lab reads the gzip (the source of truth) directly, so sqlite
> isn't required.

---

## Historical market data (Binance aggTrades)

Separate from the journal pipeline: this downloads Binance **spot aggTrades**
daily archives from [data.binance.vision](https://data.binance.vision) for
BTCUSDT and ETHUSDT — a long trade history for backtesting/feature research,
independent of what the bot itself captured. It is the only part of the lab that
touches the network, and is **not** part of `run_all` (which only refreshes the
journal→parquet analysis).

### `python -m model_lab.hist`
Downloads the **last 90 days** (default) into `../data/aggtrades`, one
**zstd-parquet per symbol-day** (`aggtrades-{SYMBOL}-{YYYY-MM-DD}.parquet`), and
writes a `manifest.json` of exact date coverage per symbol.

- **Resumable & incremental** — re-running only fetches days not already on disk,
  so the *same command* extends the range from 90 days to 18 months:
  ```bash
  python -m model_lab.hist                    # last 90 days, BTC+ETH
  python -m model_lab.hist --days 540         # ~18 months (only the gap is fetched)
  python -m model_lab.hist --symbols BTCUSDT --start 2025-01-01 --end 2025-03-31
  ```
- **Checksums** — each archive's SHA-256 is verified against its `.CHECKSUM`
  sidecar (`--no-verify-checksum` to skip; a mismatch is reported and not stored).
- **Missing days** — a day's archive publishes ~next day, so the most recent
  day(s) report `MISSING` (not an error) and fill in on a later run.
- Trades are stored with `transact_time` normalized to **microseconds** (Binance
  emits µs for spot from 2025-01-01, ms before); prices/quantities as float
  (research, not accounting); ids as int64.
- Flags: `--symbols`, `--days`, `--start`/`--end` (YYYY-MM-DD), `--jobs`
  (concurrent downloads, default 4), `--no-verify-checksum`, plus `--hist-dir`.

**Verify:** prints per-day progress, a missing-day report, and the store's total
**disk usage**. zstd keeps it compact (one ETH day ≈ 11 MiB / ~1.2 M trades), so
budget roughly **3–5 GB** for 90 days of BTC+ETH, scaling toward **~20–30 GB** at
18 months — varies with trade volume.

### `python -m model_lab.hist_integrity`
Validates the store per day-file: **row count** (non-empty, matching the
manifest), **monotonic timestamps** (`transact_time` never decreases), and **no
duplicate trade ids** (`agg_trade_id` strictly increasing and unique). Cross-day
id continuity is reported as an informational note (a gap at a missing-day
boundary is expected). Writes `out/hist_integrity/report.json`; exits nonzero if
any day fails.

```bash
python -m model_lab.hist_integrity                 # validate the whole store
python -m model_lab.hist_integrity --symbols BTCUSDT
```

---

## Telonex vendor data — trial pull + validation (opt-in)

A **paid** vendor (Telonex, `telonex.io/docs`) sells tick-level Polymarket + Binance
history as daily parquet. Before committing to the full ~500 GB purchase, this pair of
commands pulls **one day** and validates it against our own recordings. Needs the opt-in
`zstandard` extra (`pip install -e .[telonex]`) and the API key in `.env` under
**`telonexdata`** (read from `$telonexdata` / `.env`, **never printed or committed**).
Not part of `run_all`. See `telonex_notes.md` for the full API reference.

### `python -m model_lab.telonex_ingest --trial`
Pulls one UTC day of a crypto up/down series (all windows with data) + a Binance symbol
into `../data/telonex/raw`, streaming every file to disk. It probes the **public**
availability endpoint (free, no key) to prune the day's 288 candidate windows to the ones
with data, downloads Binance first (validates the key/plan), then one probe window, then
does a **disk preflight (hard-abort if free space < 3× the estimated trial size)** before
the rest. Aborts loudly on the first HTTP 403 (a free-tier key allows only 5 downloads).

```bash
python -m model_lab.telonex_ingest --trial                       # 2026-07-05 btc-updown-5m Up+Down + btcusdt
python -m model_lab.telonex_ingest --trial --day 2026-07-05 --outcomes Up,Down
python -m model_lab.telonex_ingest --trial --max-windows 24      # a cheap smoke slice
```
Flags: `--day`, `--series`, `--outcomes`, `--channels`, `--binance-symbol`,
`--binance-channels`, `--max-windows`, `--jobs`, `--disk-factor`, plus `--telonex-dir`.

### `python -m model_lab.telonex_validate`
Reads the trial parquet (+ our journal for the clock cross-check) and reports six
PASS/FAIL checks with measured numbers — coverage, date range, snapshot **cadence**
(per-window median ≤ 1 s), **depth** (≥ 20 levels), **clock alignment** (internal + a
constant, drift-free offset vs our own Binance/CLOB recordings), and **completeness**
(gaps, dups, Up/Down mirror integrity) — then whole-file **zstd**-compresses the trial
(bit-identical roundtrip), measures the ratio, and projects the needed-slice size against
a **250 GB** budget. Writes `out/telonex/{validation.json, report.html}` and a console
**GO / NO-GO**. **Stops here** — the full download is a separate task.

```bash
python -m model_lab.telonex_validate                 # validate + compress + report
python -m model_lab.telonex_validate --no-compress   # checks only (no zstd needed)
python -m model_lab.telonex_validate --no-catalog    # skip the markets-dataset download
```

---

## Shadow mode — the deployed champion + its guards (BUILD_PLAN 12–13)

The Rust `shadow` crate runs the champion `dir10_full` model **live for
observation only** (it influences nothing). These commands produce/guard it:

```
# Export the deployable booster + Rust parity fixtures (needs the [gbt] extra).
# Reproduces learn_walkforward._final_model deterministically → models/model_dir10_full.{txt,meta.json}.
python -m model_lab.export_champion

# Nightly LIVE feature-parity guard (the mandatory basis-bug tripwire): compares
# shadow's journaled live features (data/shadow/*.jsonl.gz) against the offline
# lab features, per-feature. Build the offline reference over the same period first.
python -m model_lab.short_horizon          # the offline reference
python -m model_lab.shadow_parity          # PASS/FAIL per feature

# The observation smoke report: predictions per series/day, coverage, model identity.
python -m model_lab.report_shadow
```

The **Rust** export-parity gate (`just shadow-parity`) asserts the pure-Rust
tree-walker reproduces `booster.predict` to 1e-6. `python -m model_lab.verify`
exercises the parity guard in-memory (clean → PASS, a scale bug → FAIL).

---

## Stages (one entry command each)

Run them in order — each reads the previous stage's parquet from `out/`. Or run
them all at once with `python -m model_lab.run_all` (which also runs the
**calibration audit** and the **evaluation harness** below).

### 1. `python -m model_lab.ingest`
Parses the journal + depth into tidy parquet tables in `out/`:
`ticks`, `model`, `fills`, `settlements`, `windows`, `depth`.
**Verify:** prints per-table row counts; a nonzero `depth` means the capture is
present. (No depth yet → empty `depth.parquet` + a note to run `bot record`.)

### 2. `python -m model_lab.features`
Builds a per-asset, one-second feature grid → `features.parquet`: Binance mid,
log return, a practical realized vol, Chainlink-minus-Binance `basis_bps`, and
(from depth) order-book **imbalance**, **microprice**, **spread**.
**Verify:** prints the row count and asserts `imbalance ∈ [−1, 1]`.

### 3. `python -m model_lab.labels`
Builds the forward-looking targets → `labels.parquet` (per-second forward 5 s
Binance-mid return + its sign) and `window_labels.parquet` (per-window realized
Up/Down outcome from settlements).
**Verify:** prints label counts + class balance; the forward label reads
strictly later ticks (no lookahead by construction).

### 4. `python -m model_lab.dataset`
Assembles a **train-ready** dataset → `out/dataset.parquet` (+ a generated
`out/dataset/{SCHEMA.md, metadata.json}`). Like the calibration audit, it is
**self-contained** — it reads the journal *directly*, so no prior stage is
required. Per resolved 5-minute window it reconstructs the **strike**
(price-to-beat = the Chainlink `Vendor` price at the window open, `LastAtOrBefore`),
samples feature timestamps on a fixed grid (`--grid-secs`, default 15 s), and
attaches — as-of each sample (backward `merge_asof`, never reading the future) —
the engine features (mid, `sigma_1s`, `basis_bps`, `z`, `p_up`, and depth
imbalance/microprice/spread). Two **labels** are attached: `fwd_up_30s` — the
direction of the mid `--horizon-secs` (default 30 s) later — and `outcome_up` —
the window's final Up/Down resolution. A separate **`in_live_coverage`** flag
marks windows for which a Polymarket market-mid benchmark was recorded, and a
per-series chronological **`split`** (`train`/`val`, latest windows → `val`) is
assigned.

**Historical extension (Binance aggTrades proxy).** Beyond the windows we recorded
live, the stage also reconstructs 5-minute windows across the **full downloaded
Binance aggTrades history** (`../data/aggtrades`, both BTCUSDT + ETHUSDT — see the
`hist` command above), streamed **month by month** so the whole history is never
held in RAM. Those historical windows have no Chainlink feed, so their strike (last
Binance trade at-or-before open) and outcome (`end >= strike => Up`) come from
Binance — a *proxy* for the true resolution, marked explicitly:

- **`label_source`** = `chainlink` (journal-covered) or `binance_proxy` (historical).
  Proxy rows carry the price-only features (`mid`, `sigma_1s`, `log_s_k`, `z`,
  `p_up`); `chainlink` / `basis_bps` / depth features are `NaN` there.
- **`strike_distance_at_close`** (absolute) and **`strike_distance_at_close_vol`**
  (in units of the window's realized vol) expose how close the resolution landed to
  the strike — filter or down-weight **knife-edge** proxy windows where the ~13 bps
  Binance-vs-Chainlink basis could flip the outcome.

The `fwd_up_30s` label is price-only and so proxy-safe on both sources (the primary
target). A historical window colliding with a journal window is dropped, so the
`chainlink` subset is **never** altered. Skip the history with `--no-history`. Until
`python -m model_lab.hist` has populated the store the stage is journal-only (and
says so).

**Verify:** prints per-source window/sample counts, the per-symbol-per-month
history breakdown, the knife-edge proportions, `strike-vs-outcome agreement`
(~1.0), and confirms the journal subset is unchanged. The **hard no-look-ahead
rule** — no feature derived from any data after its sample timestamp — is enforced
three ways over **both** sources: a runtime assertion (`_assert_no_lookahead`, run
on every batch), a static **AST source scan**, and **differential** tests (journal
+ Binance-proxy) that corrupt all future data and assert the features are
bit-identical (`tests/test_dataset.py`). Full column docs in
`out/dataset/SCHEMA.md`.

### 4b. `python -m model_lab.short_horizon`
A **short-horizon companion** to `dataset` → `out/short_horizon.parquet` (+ a generated
`out/short_horizon/{SCHEMA.md, metadata.json, sanity.json, sanity.html}`). Same
self-contained window alignment, as-of feature kernel (incl. basis-corrected `z`), and
month-by-month streaming over **both** sources (journal Chainlink + Binance aggTrades
proxy), but the labels are **short**: `fwd_up_10s` / `fwd_up_15s` — the direction of the
Binance mid 10 and 15 seconds ahead (ties → Up, `NA` if the `+h` second is unobserved) —
sampled on a **dense grid** (`--grid-secs`, default **5 s**; use `--grid-secs 1` for
maximal density). The window `outcome`/`outcome_up` are carried as secondary labels.

**Microstructure features (from our own recordings).** Beyond the price/vol/basis
features, each sample carries — **only where our recordings cover its timestamp** —
depth features from the Binance `depth20@100ms` capture (multi-level book imbalance
`depth_imb_{1,5,10,20}`, `microprice_gap`, `bid_depth_slope`/`ask_depth_slope`,
`depth_spread_bps`) and Polymarket Up-token book features from the `top_of_book` capture
(`pm_mid`, `pm_spread`, `pm_book_imb`, and a **staleness** signal `pm_staleness_{1,2,3}s`
= the Binance mid move minus the PM mid change over the same 1–3 s span). All attach via
a backward `merge_asof` (`depth_feat_asof_ts_ms` / `pm_asof_ts_ms` ≤ `sample_ts`);
missing-coverage rows get honest **`NaN`** (never build-time imputation), and every one
is `NaN` on `binance_proxy` rows. The feature list is config-driven (the
`DEPTH_LEVEL_DEPTHS` / `STALENESS_LOOKBACKS_SECS` knobs), with per-feature docs in
`SCHEMA.md`.

Each sample is also marked with whether **our own** recordings cover its timestamp:
**`depth_covered`** (a Binance `depth20@100ms` frame within `--coverage-tolerance-secs`,
default 2 s) and **`book_covered`** (a Polymarket `top_of_book` for the window's up/down
token within tolerance). These are provenance metadata — they may reference a recording
a hair after the sample, so, like `in_live_coverage`, they are **not** features and are
exempt from the no-look-ahead rule. Historical `binance_proxy` windows predate our
captures, so both flags are `False` there.

**Distribution sanity report** (`sanity.{json,html}`, like `feature_set`'s): every
feature column is scored PASS/WARN/FAIL for unexpected NaN, absurd outliers, and constant
columns. A NaN is *expected* where coverage is absent (or during vol warmup); only
*excess* NaN fails.

**Verify:** prints per-source window/sample counts, **per-day sample counts + depth/book
coverage percentages**, the **sanity status**, and confirms the journal subset is
unchanged. The same **hard no-look-ahead rule** holds: the runtime assertion
(`_assert_no_lookahead`), the AST scan (shared + microstructure feature functions,
extended with the 10s/15s label names), and **differential** tests over `FEATURE_COLS`
for both sources — plus a **positive** test that the labels genuinely react to
`t+10`/`t+15` and that the depth/PM features reflect the recordings
(`tests/test_short_horizon.py`). Full column docs in `out/short_horizon/SCHEMA.md`.

### 5. `python -m model_lab.feature_set`
Builds the **curated, learner-ready feature matrix** on top of `dataset.parquet` →
`out/feature_set.parquet` (+ a generated `out/feature_set/{SCHEMA.md, sanity.json,
sanity.html, metadata.json}`). Same rows as the dataset, augmented with the
engineered predictors, attached as-of each sample (backward `merge_asof`, never
reading the future): **multi-horizon returns** `ret_{1,5,15,60}s`, **signed
aggressor flow imbalance** `flow_imb_{30,120,300}s` and **trade intensity**
`trade_intensity_{30,120,300}s` (from the Binance aggTrades `is_buyer_maker` +
`quantity`), **fast/slow EWMA vol + ratio** (`sigma_fast` hl 10 s, `sigma_slow`
hl 60 s = engine σ_1s, `vol_ratio`), the **formula z-score**
`z = log(mid/strike)/(σ_slow·√τ)`, **seconds_remaining**, and **hour_of_day**.

The whole feature list lives in **one config** (`FEATURE_SPEC`, expanded from the
`RETURN_HORIZONS_SECS` / `FLOW_LOOKBACKS_SECS` / `VOL_HALFLIVES` knob lists at the
top of `feature_set.py`), so adding or removing a feature is a near-one-line
change; that list drives the output columns, the parquet schema, `SCHEMA.md`, and
the sanity report's bounds/coverage expectations. Flow/intensity come only from the
aggTrades archive and vols/returns warm up at each run's start, so those NaN are
**expected** (classified from `mid_fresh` / `flow_covered` / `price_run_secs`) and
left in the parquet — the sanity check fails only on *excess* (unexpected) NaN.

**Verify:** every feature gets a **distribution sanity report** — no unexpected
NaNs, no absurd outliers, no constant columns — with a PASS/WARN/FAIL per column
(`out/feature_set/sanity.html`), plus cross-checks that the rebuilt `mid`/`σ_slow`/
`z` reconcile with the dataset skeleton (≈ 0). The same three-way no-look-ahead
guarantee holds: a runtime assertion, the AST scan (the grid kernels are in
`tests/test_dataset.py::test_no_lookahead_ast_scan`), and differential/corruption
tests for the journal + aggTrades-proxy paths (`tests/test_feature_set.py`). Full
column docs in `out/feature_set/SCHEMA.md`.

### 6. `python -m model_lab.learn`
The first **challenger** model: an L2-regularized logistic regression (hand-written
numpy — the lab has no scikit-learn) on the 16 `feature_set` features, trained to
predict two targets **separately** — the **30-second direction** (`fwd_up_30s`) and
the **window outcome** (`outcome_up`). Evaluation is **strict walk-forward**: a
trailing `--train-weeks`-week training window, tested on the following `--test-days`
days, rolled forward through the whole `--days` span (default 90), **never shuffling
across time**, with a **purge** at every boundary (a gap ≥ the label horizon: 30 s
for the 30-second target, the window close for the outcome target) so no fold peeks
across it. The pooled out-of-sample predictions are scored **through the evaluation
harness** (vs the formula model and the market — the harness self-restricts to the
real journal windows where those benchmarks exist); the 30-second model is
*additionally* scored on its own `fwd_up_30s` label. A **shuffled-label control**
retrains the identical pipeline on permuted labels and must collapse to chance — the
proof the pipeline can't cheat.

Writes `out/learn/{metrics.json, folds.csv, model_<target>.json,
predictions_<target>.parquet, harness_<target>/…}`. `model_<target>.json` is an
inspectable artifact (coefficients + standardization + the reproducibility seed) —
the lab's first persisted model. Reads `feature_set.parquet` (run `dataset` +
`feature_set` first) and the journal (for the benchmarks; skip it with
`--no-harness`).

```bash
python -m model_lab.learn                       # both targets, last 90 days
python -m model_lab.learn --days 90 --train-weeks 4 --test-days 7
python -m model_lab.learn --targets outcome --no-harness   # quick iteration
```

**Verify:** prints each target's pooled Brier/log-loss/dir-acc, the vs-formula and
vs-market improvement, and the shuffled-control collapse; open
`out/learn/harness_outcome/report.html`. On a short span (e.g. the fixture) it
degrades to a single chronological train/test split; with enough days it rolls a
true walk-forward.

### 7. `python -m model_lab.validate`
Checks the lab reproduces the engine and scores model calibration →
`out/validation/{metrics.json, reliability.csv}`:
- **Φ(z) identity** — recompute `p_up = Φ(z)` from the journaled `z`; must match
  the journaled `p_up` (proves the lab's Φ equals the engine's).
- **σ_1s reproduction** — recompute the engine's gap-aware EWMA vol from the raw
  Binance ticks and correlate with the journaled `sigma_1s`.
- **Calibration** — one mid-window prediction per window vs the realized
  outcome: reliability curve + Brier score.

**Verify:** prints the Φ residual (median |Δ| should be ~1e-12), the σ
correlation, and the Brier score. Exits nonzero if the Φ identity is off.

### 8. `python -m model_lab.research`
The reason the depth capture exists — does microstructure predict short-horizon
moves? Aligns depth features to the forward 5 s return and reports, per asset and
pooled → `out/research/metrics.json`:
- **IC** — correlation of imbalance (and microprice tilt) with `fwd_ret_5s`;
- **AUC** — imbalance predicting the up-move, vs a momentum baseline and 0.5.

**Verify:** prints the IC/AUC table. Positive IC / AUC > 0.5 = the depth signal
is worth feeding into fair value. No depth captured yet → says so, cleanly.

### 9. `python -m model_lab.report`
Assembles a single self-contained `out/report.html` (inlined plots) tying the
calibration and depth-signal findings together.
**Verify:** open `out/report.html` in a browser.

### Evaluation harness (the single source of truth) · `python -m model_lab.evaluate`
**Given *any* model's probabilities, is it any good — vs the formula model and vs
the market?** This is the one way every model is scored, so the answer is computed
the same way everywhere (`model_lab.eval_harness` is the shared core; every later
model-scoring stage reports through it). It scores the model with **Brier**,
**log-loss**, **directional accuracy**, and a **calibration curve**, **always side
by side against two benchmarks**: (1) the **formula model** `p_up` recomputed at the
same timestamps, and (2) the **market-implied** probability (the Polymarket Up-token
mid) on the subset where recordings exist. It emits **one standardized verdict
table** — model vs each benchmark, **per calendar-day period** (with a weekly
rollup), with **relative-improvement percentages** and a **stability** summary
(how often, and how consistently, the model beats each benchmark across periods).

**Model input contract** — a grid parquet with columns
`series, window_open_ms, sample_ts_ms, p_up` (the keys of `dataset.parquet` /
`feature_set.parquet`, so a learner's output joins 1:1). Pass it with
`--predictions`. **With no `--predictions`**, the formula model is evaluated as a
*self-baseline* (model-vs-market is meaningful; model-vs-formula is an identity
check). Reads the journal directly for the benchmarks + outcomes — no prior stage
required. Every breakdown/period is flagged when backed by fewer than
`--min-windows` (default 50) distinct **windows** (the unit of trust).

Writes `out/evaluate/{metrics.json, scores.csv, verdict_table.csv, reliability.csv, report.html}`.

```bash
python -m model_lab.evaluate                                    # formula self-baseline vs market
python -m model_lab.evaluate --predictions out/my_model_preds.parquet
python -m model_lab.evaluate --predictions preds.parquet --min-windows 50 --series BTC-5m
```

- `--predictions FILE` the grid parquet of the model under test (omit for the self-baseline).
- `--health ready` (default) scores only snapshots the engine marked healthy; `--health all` scores every one.
- `--min-windows N` low-sample threshold (default 50). `--series` comma-separated filter. `--period {day,week}` which rollup the printed summary emphasizes (both are always computed).

**Verify:** prints the overall Brier/log-loss/dir-acc, the model-vs-formula and
model-vs-market improvement + win rate per period, and the verdict; open
`out/evaluate/report.html`. Programmatic entry point for future stages:
`evaluate.evaluate_predictions(paths, predictions_df, ...)`.

### Calibration audit (standalone) · `python -m model_lab.calibration_audit`
**Is the formula model well-calibrated, and does it actually beat the market?** A
thin wrapper over the evaluation harness above that evaluates the **formula model**
as the model under test (a self-baseline — benchmark 1 is the model itself, and the
market is the real comparison). It stays **self-contained** — it reads the journal
*directly*, so it needs no prior stage. Same scoring as `evaluate`: Brier, log-loss,
directional accuracy and calibration, split by **series**, **time-remaining** bucket
(early / mid / final-minute / final-20s), and **calendar-day period**, with a
reliability table, the standardized verdict table, and a plain-language **verdict**.
Every breakdown is flagged low-sample below `--min-windows` (default 50) distinct
**windows**.

Writes `out/calibration_audit/{metrics.json, scores.csv, verdict_table.csv, reliability.csv, report.html}`.

```bash
python -m model_lab.calibration_audit                        # single command on a fresh journal
python -m model_lab.calibration_audit --min-windows 50 --health ready
```

- `--health ready` (default) scores only the snapshots the engine marked healthy
  (what it would actually act on); `--health all` scores every snapshot.
- `--min-windows N` sets the low-sample threshold (default 50).

**Verify:** prints the overall model-vs-market Brier + log-loss and any
low-sample warnings; open `out/calibration_audit/report.html`.

### Second challenger — GBT (opt-in) · `python -m model_lab.learn_gbt`
The second **challenger**: **LightGBM** gradient-boosted trees on the same 16
features, the same two targets, the same strict walk-forward + purge + shuffled
control as `learn`, reported through the same harness. **Opt-in** — needs the
`gbt` extra (`pip install -r requirements-gbt.txt`); the core pipeline stays
lightgbm-free.

The formula is injected **two ways** and compared on the outcome target (adding
`Φ(z)` as a literal input column is a *no-op* for trees — they are invariant to
monotone transforms of `z`, already a feature):
- **plain** — a GBT on the 16 features, no formula prior;
- **residual** — the formula is the LightGBM `init_score` base margin
  (`logit(Φ(z))`), so the trees learn only the **correction** over the formula.

Boosting rounds are chosen by **forward-chained inner validation** (a purged tail
of each training window), then refit on the full window — never fit on the test.
Beyond the harness scorecard it reports **feature importances** (gain + split), a
**volatility-regime** split, a **time-remaining** breakdown, and a **residual
analysis**: where the GBT disagrees with the formula (`|p_gbt − Φ(z)| > δ`),
bucketed by time-remaining × `|z|`, and who is right there — answering *does the
GBT add anything the formula lacks, and where*. Artifacts are deterministic
(same seed → byte-identical booster + predictions).

Writes `out/learn_gbt/{metrics.json, folds.csv, model_<target>_<variant>.{txt,json},
predictions_<target>_<variant>.parquet, oos_*.parquet, harness_*/…,
feature_importance_*.csv, regime_*.csv, residual_*.csv}`.

```bash
python -m model_lab.learn_gbt                          # both targets, last 90 days
python -m model_lab.learn_gbt --targets outcome --no-harness   # quick iteration
python -m model_lab.learn_gbt --num-leaves 31 --min-child-samples 200 --learning-rate 0.02
```

### Model comparison · `python -m model_lab.compare`
One command: **logistic vs GBT (plain) vs GBT (residual) vs formula vs market**,
on the outcome target, across every period. Builds one harness-anchored frame,
key-joins each model's predictions onto it, and scores all predictors on the same
rows (the formula/market benchmarks are pairwise-defined, coverage reported).
Reads the `learn` + `learn_gbt` grids (run those first, or `run_all --with-gbt`
for aligned grids). Writes `out/compare/{comparison.csv, metrics.json, report.html}`
and prints a one-line verdict (best challenger; whether either GBT beats the
logistic and the formula).

```bash
python -m model_lab.run_all --with-gbt      # full pipeline incl. GBT + comparison
python -m model_lab.compare                 # just the comparison (grids must exist)
```

### Short-horizon challenger — GBT (opt-in) · `python -m model_lab.learn_short_gbt`
A **LightGBM** challenger on the dense `short_horizon.parquet`, predicting the
short-horizon forward-direction labels `fwd_up_10s` (primary) and `fwd_up_15s`
(secondary) — the same strict walk-forward + purge + determinism contract +
shuffled control as `learn_gbt`, but scored on the model's **own** short-horizon
label (dir-acc, Brier, calibration) rather than the window outcome. **Opt-in** —
rides the same `gbt` extra.

- **Trade-flow joined in.** `short_horizon` carries no signed-flow columns, so
  `flow_imb_*` / `trade_intensity_*` are joined from `feature_set.parquet` by a
  **causal backward as-of join** (feature_set is on the 15s grid, short_horizon on
  5s) bounded by a staleness tolerance — look-ahead-safe (matched ts ≤ sample ts).
  `--no-external-flow` skips it (base = native short-horizon features only).
- **Three-way ablation** (`--variants base,depth,full`): **base** (price/vol/model
  + flow) → **depth** (+ Binance `depth20`) → **full** (+ Polymarket book).
  Microstructure is NaN where uncovered / on `binance_proxy`; LightGBM handles the
  NaN natively. The pairwise **contribution** (base→depth, depth→full) reports the
  Brier / dir-acc delta, overall and on the covered subset.
- **Both scopes in one run** (`--scopes all,chainlink`): the primary proxy+chainlink
  scope and the clean chainlink-only microstructure ablation, side by side.
- Beyond the pooled scorecard: an **abstention** curve (coverage-vs-accuracy as the
  confidence threshold `|p−0.5|` rises) and per-regime / per-time-remaining
  breakdowns against the naive `Φ(z)` reference.

Writes `out/learn_short_gbt/{metrics.json, folds.csv, depth_contribution.csv,
model_<scope>_<target>_<variant>.{txt,json}, predictions_*.parquet, oos_*.parquet,
feature_importance_*.csv, regime_*.csv, abstention_*.csv, reliability_*.csv}`.

```bash
python -m model_lab.learn_short_gbt                                # both targets/variants/scopes
python -m model_lab.learn_short_gbt --targets fwd10 --scopes chainlink   # quick iteration
python -m model_lab.learn_short_gbt --with-harness                 # also score vs the window outcome
```

### Short-horizon money judge — the Phase-1 verdict · `python -m model_lab.money_judge`
**Does the 10-second signal actually make simulated money on our own recorded weeks?**
This replays the journal and, at each timestamp where the trained 10s model fires
beyond a confidence threshold, simulates two actions against the **recorded Polymarket
book**, pays the **exact** documented taker fee, and marks the position **to the
window's actual resolution**:

- **taker** — buy the model's side at the displayed ask, up to the displayed size,
  after a configurable latency, then hold to resolution ($1 on the winner);
- **pair-leg** — take the leaned-side fill, then complete the opposite leg from the book
  5–15 s later; credit a locked pair when the combined cost < $1, and honestly book the
  stranded leg's loss when it never gets cheap enough before close.

It **sweeps the confidence threshold** and reports per-day and per-window PnL (after
fees), trade counts, and win rate, comparing the ML signal against **never trading** and
the engine's own hand-coded **momentum** trigger (plus the late-window certainty taker)
evaluated on the *same* replay — then states a plain-language verdict.

Honesty first: the fires are the **walk-forward out-of-sample** predictions
``learn_short_gbt`` already wrote (``oos_<scope>_fwd10_<variant>.parquet`` — never
in-sample). Down-side fills use the documented CLOB mirror (the Up **bid** is the
mirrored Down **ask**: price ``1 − bid``, size ``bid_sz`` — a real recorded order).
Conservative by construction: pay the ask, skip on a stale/absent book, floor at the $1
min notional. **LightGBM-free** — it reads the OOS parquet + the journal only (so run
``learn_short_gbt`` first). A majority-class (always-predict-the-common-side) baseline
is reported so model skill reads as *lift*.

Writes ``out/money_judge/{sweep.csv, per_day.csv, per_window.csv, metrics.json,
report.html}``.

```bash
python -m model_lab.money_judge                              # headline: all-scope, full variant
python -m model_lab.money_judge --scope chainlink --variant base
python -m model_lab.money_judge --since 2026-07-03 --until 2026-07-04   # bound a day on a big journal
```

- `--scope {all,chainlink}` / `--variant {base,depth,full}` pick which OOS model to trade
  (default `all` / `full`); `--source chainlink` trades only the real-book / real-resolution weeks.
- `--latency-ms` (200), `--window-budget` / `--trade-budget` ($10), `--comp-min-s` / `--comp-max-s`
  (5 / 15 s), `--lock-threshold` (1.0), `--fee-rate` (0.07), `--no-momentum`, `--no-late-window`.

---

## Historical ingest — Telonex + aggTrades (full-history dataset + momentum backtest)

Three self-contained stages that extend the lab across the **full historical span** (months
before the recorder journal began), from the data on disk: Binance **aggTrades**
(`data/aggtrades`, the mid/vol/strike backbone), Telonex Binance **`book_snapshot_25`** depth
(`data/telonex`, real L2 microstructure), Telonex Polymarket **quotes** (the PM book +
resolution), and the recorder **journal** (`data/journal`, the overlap era). Universe = the
four series **BTC/ETH × 5m/15m**. Everything is **day-by-day and memory-bounded** — process a
range at a time with `--since` / `--until` on a large store; the Telonex depth read is chunked.

### `python -m model_lab.historical_resolutions` — official resolutions (Part A.2)

The **official** Polymarket resolution is the authoritative Up/Down label. The Telonex markets
catalog (`markets.parquet`) mirrors the on-chain resolution as `result_id` (validated
0-mismatch against the CLOB API, below), giving ~99.5% coverage offline — far more than a
156k-call API fetch (the CLOB API 404s on ~12% of old markets). This stage builds a compact
resolution cache (`data/resolutions/catalog_resolutions.parquet`) and **validates** it:

- **vs the CLOB API** over a *stratified* sample (≥ 2000 windows, every series/month +
  knife-edge windows specifically) — **any catalog-vs-API mismatch is a hard failure**; API
  results cached to `data/resolutions/clob_cache.jsonl` (resumable, rate-limited);
- **vs the journal** on overlap days (must match `Resolved` exactly; bounded segment read).

```bash
python -m model_lab.historical_resolutions               # build cache + full validation
python -m model_lab.historical_resolutions --no-api      # cache + journal check only (offline)
```
Output `out/historical_resolutions/report.json`. Exit non-zero on any mismatch.

### `python -m model_lab.historical_labels` — window labels (Part 2)

Assigns the **official** resolution (primary) to every window present on disk — a dict lookup,
so ~100% coverage instantly, no per-window quote reads. Quote-convergence and Binance-anchored
`end ≥ strike` are **graded cross-checks** (Part A.3): the stage samples them per series/month
+ knife-edge and reports each proxy's error rate vs official truth (which proxy to trust where
the API lacks a market). A window with no official resolution is **counted, never guessed**.

```bash
python -m model_lab.historical_labels                        # full history, official labels
python -m model_lab.historical_labels --since 2026-05-01 --until 2026-05-08
```
Output `out/historical_labels/{labels.parquet, metrics.json, report.html}` (coverage per series
per month + the proxy grading). Run `historical_resolutions` first to build the resolution cache.

### `python -m model_lab.historical_dataset` — the dataset builder (Part 1)

Emits **short_horizon-format** rows (+ a new `depth_source` column) across the span, **day by
day, resumable** (one partition parquet per `(series, day)` + a `manifest.json`; a re-run skips
done days). Day ownership: journal owns `>= 2026-07-04` (`depth_source=recorder`,
`label_source=chainlink`, real Chainlink strike/`Resolved`, reusing the `short_horizon` path
verbatim); Telonex owns earlier days (`depth_source=telonex`, `label_source=telonex`,
Binance-anchored strike + Telonex depth/PM microstructure). The **outcome is the official
Polymarket resolution** everywhere (windows with no official resolution excluded, counted).

**Strike & basis:** historical `K` is Binance-anchored (last trade ≤ open); the Chainlink-
definition features (`chainlink`, `basis_bps`, `basis_ewma`, the basis-*corrected* `z`) are
journal-era-only — NaN on Telonex rows, where `z` is the price-only (basis = 0) definition.
`--overlap-parity` verifies the **source-agnostic** features (mid, σ, depth microstructure) of a
rebuilt-Telonex vs the journal-owner rows match within the `telonex_binance_validate` guard
tolerances (the standing telonex-vs-recorder gap check).

```bash
python -m model_lab.historical_dataset --since 2026-05-01 --until 2026-06-01   # build a month
python -m model_lab.historical_dataset --combine          # concat partitions → out/historical_dataset.parquet
python -m model_lab.historical_dataset --overlap-parity    # telonex-vs-recorder feature gap
```

### `python -m model_lab.backtest_momentum` — the momentum backtest (Part 3)

Replays the engine's hand-coded **momentum trigger** over the reconstructed historical book,
reusing `money_judge`'s already-verified ports of `crates/engine/src/taker/*` (the confirmed-
move detector, the fee-aware edge gate, the gate ladder) and its conservative taker-fill rules
(fill only at the displayed best quote up to its size; exact 5-dp fee; Down = CLOB mirror; skip
on a stale book; mark to quote-convergence resolution). Controls the real trigger must beat,
else **VOID**: (a) never-trade = $0; (b) a **matched-frequency shuffled** control (fire at random
eligible snapshots at the real confirmed-move rate, over `--seeds` seeds — the real net must
exceed the shuffled distribution, permutation `p ≤ 0.1`); (c) the majority baseline. Reports PnL
after fees / win rate / trade count / **drawdown per series per month** and **by time-of-day**.

```bash
python -m model_lab.backtest_momentum --since 2026-05-01 --until 2026-06-01 --seeds 20
python -m model_lab.backtest_momentum --series BTC-5m,ETH-5m
```
Output `out/backtests/momentum/{report.html, metrics.json, trades.csv, per_series_month.csv,
per_hour.csv}`. Stop after the report.

### `python -m model_lab.backtest_pair_lean` — the pair-lean backtest

The operator's **pair-lean** strategy over the current-regime windows (≥ 2026-06-05, 255 ms
effective). Per window, when the walk-forward **dir10** model (the OOS parquet
`out/learn_walkforward/oos_dir10_full.parquet`, **no retraining**) predicts direction with
confidence `|p_up − 0.5| ≥ T`, **take the predicted side at the ask** (taker fee + displayed
depth, sized to a fraction of displayed depth), then **rest a maker quote on the other side** at
limit `L = C − price_s` so combined pair cost ≤ `C`. A completed pair collects $1 (locked profit
`1 − C`, minus the taker fee); an uncompleted leg **rides to resolution**. Unlike
`money_judge`'s taker-cross pair-leg, the completion is a **resting maker** (zero fee, fills only
when the recorded opposite ask reaches ≤ `L`). LightGBM-free (reuses `money_judge`'s fill
primitives + the `backtest_momentum` reconstruction). Sweeps `T` (`|p−0.5| ≥ θ`), `C` ∈
{0.90, 0.94, 0.96, 0.98}, and lean size (fraction of displayed depth ∈ {0.25, 0.5, 1.0}); fires
multi-time per window until the $10 budget is spent. Judged per series vs never-trade, a matched-
frequency random-trigger control + a shuffled-labels control, and the momentum ($5,057) /
model-taker ($2,284) baselines on the identical windows. Resumable per-(series, day); the
`--seeds` random control is the dominant runtime cost.

```bash
python -m model_lab.backtest_pair_lean                       # full current regime, 4 series
python -m model_lab.backtest_pair_lean --series BTC-5m --seeds 4
```
Output `out/backtests/pair_lean/{metrics.json, sweep.csv, per_series_best.csv, report.html}`
(aggregates only — no per-trade dump, so the full 108-config sweep stays memory-bounded).

### `python -m model_lab.rebate_sim` — the taker-rebate simulation

Replays any taker-fill `trades.csv` through Polymarket's tiered **Taker Fee Rebate Program**
(docs.polymarket.com/trading/taker-rebates). Your **tier** is set by 30-day *rolling* weighted
volume `wV = size·(1−entry)·2.3` (crypto weight); the rebate **paid** is `tier% × the taker fees`
you paid (a fee *refund*, NOT a % of notional). Tiers: Bronze $2k→3% · Silver $20k→8% ·
Gold $200k→18% · Platinum $1M→32% · Diamond $4M→44% · Obsidian $10M→50%. Since it is
path-dependent on trailing volume it is computed on a daily cycle (never per-trade): per exec-day
`wv_day`/`fees_day`, `tier_D` from the trailing 30 completed days, `rebate_day = tier_D% ·
fees_day`, `$1` daily-accrual carry. Reports the **tier timeline**, **total rebate**, **corrected
net PnL** (`net + rebate`), and the **marginal wV** to the next tier. The tier table + `2.3`
weight mirror the Rust consts in `crates/venue-paper/src/engine.rs` (and `lib/math.py`).

The model-taker tape is produced by `money_judge` (its winning-threshold fills are written to
`out/money_judge/trades.csv`), so run it after a `money_judge` run for the model-taker.

```bash
python -m model_lab.rebate_sim                                        # momentum-verdict baseline
python -m model_lab.rebate_sim --trades out/money_judge/trades.csv --out-name model_taker
python -m model_lab.rebate_sim --trades <clone>/trades.csv --force-tier Platinum   # score at a fixed tier
```
Output `out/backtests/rebate_sim/<out-name>/{tier_timeline.csv, metrics.json, report.html}`.
`--force-tier` credits every day at a fixed tier regardless of the trades' own 30-day volume —
used by `backtest_clones` to score a competitor clone at BOTH our Bronze tier and its owner's tier.

### `python -m model_lab.competitors manuals` — competitor operating manuals

Deep per-window **reconstruction** of how five real Polymarket accounts (0xb27b, takerner,
bonereaper, wolf9478, nagi777) trade, from their cached `data/competitors/<addr>/activity.jsonl`
fills joined to our Telonex book tape — as **distributions, not averages**: entry timing, price
level vs the book at fill (at-touch / inside / crossing), sequencing, inventory trajectory, size
ramping, merge timing, per-window capital; plus, for merge-heavy accounts (nagi777, 0xb27b), the
**merge mechanics + capital velocity** (turns/day) vs a hold-to-resolution account. Every number is
**FACT** (raw records) or **CALC** (exact arithmetic); the manual body carries FACT+CALC only, with
estimates/assumptions quarantined to an *Inference* section. Each account is **anchored** against its
official Polymarket P/L — a reconstruction that contradicts official is flagged red ("OUR
RECONSTRUCTION SUSPECT"). A 10-window hand-trace appendix makes every classification re-derivable.

**nagi777 is not cached — fetch it fresh first** (read-only Polymarket data-api):
```bash
python -m model_lab.competitors resolve --handles nagi777    # add to handles.json (merges, no overwrite)
python -m model_lab.competitors fetch --only nagi777 --refresh-pnl
python -m model_lab.competitors manuals                       # the 5 roster accounts
python -m model_lab.competitors manuals_report                # → out/competitors/manuals/<handle>.html
```

### `python -m model_lab.backtest_clones` — competitor-clone backtests

Mechanizes each operating manual into a backtestable strategy scored over the current-regime windows
(≥ 2026-06-05, 255 ms). Two templates: **taker-accumulation** (takerner, bonereaper — two-sided
taker toward equal inventory) and **maker-quote + merge-recycle** (0xb27b, wolf9478, nagi777 —
resting two-sided BUY limits). Every clone emits the shared `money_judge._TRADE_COLS` trades.csv,
carries a shuffled-outcome + matched-frequency random control + capital accounting, and is run at
**both** our Bronze tier and its owner's tier (rebates via `rebate_sim`). Maker-style results are a
**bracket** — pessimistic (queue behind all displayed size) *and* optimistic (front-of-queue) — "truth
is between; only live measures it". Also a **momentum-exit** variant (enter on the momentum trigger,
sell at the bid on convergence-to-fair or after `T ∈ {15,30,60}` s, fees both ways) vs hold. For each
clone the report prints the **clone-vs-owner ladder**: the owner's own official P/L over the same
post-Jun-5 slice (FACT, from the manual) → the owner's OUR-series reconstructed net (CALC) → the clone
backtest net, with the two gaps shown (other-markets, and what we could NOT copy — queue, triggers,
tier), never smoothed. LightGBM-free; reuses the `backtest_momentum` reconstruction + `money_judge`
primitives. Run `competitors manuals` first so the owner ladder can read the slice inputs.
```bash
python -m model_lab.backtest_clones --clone all --since 2026-06-05
python -m model_lab.backtest_clones --clone takerner --seeds 8
```
Output `out/backtests/clone_<name>/{trades.csv, metrics.json, report.html}`.

### `python -m model_lab.accuracy_push` — the per-market 15 s model (opt-in GBT)

Produces, **per market** (BTC/ETH × 5m/15m), the best **15-second** forward-direction GBT and its
confidence-frontier gate `θ_85` (where held-out directional accuracy reaches 85 %). Screens two
feature sets — **price** (price/vol/formula/basis) vs **flow** (+ depth/PM order-flow
microstructure) — and picks the winner on a held-out pre-regime tail ("flow if it wins; else the
best available"; the 4-series matrix has no signed-aggressor-flow columns, so "flow" = the
microstructure block). Trains on all pre-regime data, predicts the current-regime windows OOS
(leakage-safe — no regime row is ever trained on), and writes the OOS predictions the burst
strategy consumes. Finding: an 85 %-accurate 15 s gate is **extremely selective** — θ_85 ≈
0.32–0.34 at ~0.15–1.5 % coverage ⇒ **0.18–1.20 qualifying signals per window** (flow wins every
market). Peak RSS can exceed the 3 GB advisory (loads a market's full history) — bound with
`--series` / `--days` if needed.

```bash
python -m model_lab.accuracy_push                 # all 4 markets, full history
python -m model_lab.accuracy_push --series BTC-5m --days 120   # one market, fast slice
```
Output `out/accuracy_push/{oos_<market>.parquet, model_<market>_<variant>.txt, metrics.json,
frontier.csv, report.html}`.

### `python -m model_lab.backtest_burst_pair` — the burst pair-building backtest

The operator's **high-confidence burst pair-building** strategy over the `accuracy_push` θ_85
signal: on each qualifying signal fire a burst (3 × 20-share FAK against the displayed level, depth
shortfall reported), rest a completion maker at `C − price_s` (pair cost ≤ C, locked edge `1 − C`),
accumulate inventory capped at `U` unhedged (no force-hedge; no entry in the final 30 s; unpaired
legs ride to the official resolution). Sweeps `C ∈ {0.94, 0.96, 0.98}` × `U ∈ {20, 40, 60}`. P&L
splits exactly into locked-pairs + stranded-legs. Benchmarks on **identical windows**: never-trade
($0), the champion **model-taker** (dir10 @ θ=0.03) and the engine **momentum** trigger (both at the
$10/window engine budget), + a base-rate-preserving **shuffled-outcome control**. Reuses
`money_judge` fills + the `backtest_momentum` reconstruction; resumable per-(series, day); run
per-series in parallel sharing one `--out-name`, then a final all-series pass aggregates.

```bash
python -m model_lab.backtest_burst_pair                       # full current regime, 4 series
python -m model_lab.backtest_burst_pair --series BTC-5m       # one series (parallel worker)
python -m model_lab.backtest_burst_pair --no-momentum --max-days 3   # fast slice
```
Output `out/backtests/burst_pair/{metrics.json, cells.csv, per_series_best.csv, report.html}`.

### `python -m model_lab.challenger_bakeoff` — MLP / XGBoost / LogReg vs the LightGBM champion

A model **bake-off**: reruns `accuracy_push`'s exact methodology (same static pre-regime→current-regime
split, same θ_85 gate) with the model swapped for a **numpy MLP**, **XGBoost**, or a **logistic-regression
floor**, then money-scores each through `backtest_burst_pair`. Reports accuracy/lift, calibration (ECE),
θ_85 coverage, and burst money PnL per model, plus a one-page verdict. Two shuffled-label controls: a
model-level one (train on permuted labels → OOS collapses to chance) here, and burst_pair's money-level
one. The GBT backend is byte-identical to `accuracy_push` (anti-drift tripwire test). Backends live in
`lib/backends.py` over `lib/{gbt,xgb,mlp,logreg}.py`. XGBoost is an opt-in extra:
`pip install -r requirements-xgb.txt` (or `pip install -e .[xgb]` / `.[bakeoff]`); MLP + LogReg are pure numpy.

```bash
# train each model (resumable per-market; OMP_NUM_THREADS=1 for byte-repro, or more for the MLP's speed)
python -m model_lab.challenger_bakeoff train --backend logreg      # also: xgb, mlp, gbt
# money-score a model's OOS on the burst cells (reuse the champion's benchmark numbers)
python -m model_lab.backtest_burst_pair --oos-dir bakeoff/logreg --no-benchmarks --out-name bakeoff_logreg
# one-page verdict across all models
python -m model_lab.challenger_bakeoff aggregate
```
Output `out/bakeoff/<model>/{oos_<market>.parquet, metrics.json, frontier.csv, report.html}` and
`out/bakeoff/{verdict.html, verdict.md, summary.csv}`. (2026-07-14 run: **nothing beats GBT** — all 4
model classes within 0.24 pp OOS diracc; the burst-pair strategy was the ceiling, not the model.)

### `python -m model_lab.backtest_maker_core` — does the model DEFEND maker quotes?

A **faithful two-sided maker replay** of the engine's quoting core (`crates/engine/src/quoting.rs`
+ the `quote_manager` cancel-first / reactive urgent-cancel defense, ported in `maker_core_sim.py`)
over the current-regime windows, filling resting post-only quotes from **real Polymarket trade
prints** (queue-behind-displayed, a mirror of `venue-paper::MatchEngine::on_trade` — Telonex *does*
carry a per-outcome trade tape, `io/telonex_pm.read_pm_trades`). Runs three variants on identical
windows: **(a) always-on** (the bar); **(b) model-defended** — when the walk-forward **dir10** model
fires (`|p_up−0.5| ≥ θ_defend`, sweep 0.10/0.15/0.20), pull the *threatened* side (the mirror side
the predicted move runs over) until the signal clears, with a live-realistic >15 s staleness
stand-down; **(c) model-leaned** — widen the threatened side / tighten the safe side (3-cell
half-spread sweep). Reports per series net PnL, fills, pair-completion, **5 s/30 s markout on maker
fills** (the adverse-selection measure), stranded-leg PnL. A variant wins only if it beats (a)
pooled **and** in a per-series majority; a shuffled-signal control on the winner must not beat (a).
LightGBM-free (the dir10 signal is the already-written `oos_dir10_full.parquet`).

```bash
python -m model_lab.backtest_maker_core                       # full current regime, 4 series
python -m model_lab.backtest_maker_core --series BTC-5m --since 2026-06-05 --until 2026-06-08
```
Output `out/backtests/maker_core/{metrics.json, per_series.csv, report.html}` (resumable per-day).

---

## Typical session

```bash
# One-time
python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt

# Confirm the lab works (synthetic, no real data needed)
python -m model_lab.verify
pytest -q

# Capture some depth alongside the journal (from the repo root, Rust side):
#   cargo run -p bot -- record --series BTC-5m        # writes data/journal + data/depth
# ...let it run a while, Ctrl-C, then:

# Run the full pipeline over the real journal
python -m model_lab.run_all
# open model-lab/out/report.html
```

---

## Layout

```
model-lab/
  model_lab/
    config.py            paths + stage argument handling
    io/journal.py        read journal-*.jsonl.gz  (nested {seq, ts_local_ms, rec})
    io/depth.py          read binance-depth20-*.jsonl.gz → microstructure aggregates
    io/binance_archive.py download/verify/store Binance aggTrades daily archives
    lib/math.py          Φ(z), engine σ_1s EWMA, markout, AUC/Brier/log-loss/directional-accuracy/reliability
    lib/logreg.py        numpy L2 logistic regression (sigmoid, NaN-aware standardize, IRLS fit) — no sklearn
    lib/gbt.py           deterministic LightGBM wrapper — the ONLY module importing lightgbm (opt-in extra)
    ingest / features / labels / dataset / short_horizon / feature_set / learn / validate / research / report  # the 10 pipeline stages
    dataset.py           window-aligned, leakage-safe training set (standalone; reads the journal directly)
    short_horizon.py     dense-grid short-horizon set: 10s/15s fwd labels + depth/PM-book microstructure features + sanity report
    feature_set.py       curated ML feature matrix on the dataset (one-config features + sanity report)
    learn_common.py      shared walk-forward scaffolding (fold schedule/purge, matrix load, harness scoring)
    learn.py             first challenger: walk-forward logistic regression, scored through the harness
    learn_gbt.py         second challenger (opt-in): walk-forward LightGBM, plain + formula-residual variants
    learn_short_gbt.py   short-horizon challenger (opt-in): LightGBM on fwd_up_10s/15s, base/depth/full ablation + abstention
    money_judge.py       Phase-1 verdict: replay the 10s signal as simulated taker/pair-leg trades vs never-trading + engine momentum
    maker_core_sim.py    pure port of the engine quoting core + print-driven queue-behind-displayed fill engine (for backtest_maker_core)
    backtest_maker_core.py  does the dir10 model defend maker quotes better than always-on? (baseline vs defend vs lean, markout + reconciliation)
    rebate_sim.py        taker fee-rebate over a trades.csv: 30-day rolling wV → tier → daily rebate, corrected net PnL (--force-tier)
    backtest_clones.py   competitor-clone backtests: taker/maker templates + maker-fill bracket + both controls + tier honesty + momentum-exit + clone-vs-owner ladder
    competitors/         read-only competitor study — resolve/fetch/analyze/analyze_tape/report + manuals (per-window operating manuals) + manuals_report
    compare.py           one-command comparison: logistic vs GBT×2 vs formula + market across periods
    eval_harness.py      THE single source of truth for scoring — model vs formula + market, per period (library)
    evaluate.py          score any model's predictions through the harness (standalone CLI)
    calibration_audit.py the formula model as a self-baseline through the harness (standalone)
    hist.py              download historical Binance aggTrades → ../data/aggtrades
    hist_integrity.py    validate the aggTrades store (rows, monotonic ts, dup ids)
    fixtures.py          synthetic journal + depth + predictions generator
    verify.py            fixture → all stages → assertions
    run_all.py           run the pipeline stages + calibration audit + evaluate in order (--with-gbt adds GBT + compare + money_judge)
  tests/                 test_math.py, test_smoke.py, test_dataset.py, test_short_horizon.py, test_feature_set.py, test_hist.py, test_evaluate.py, test_learn.py, test_learn_gbt.py, test_learn_short_gbt.py, test_money_judge.py, test_compare.py, test_backtest_pair_lean.py, test_maker_core.py, test_rebate_sim.py, test_manuals.py, test_backtest_clones.py, test_competitors.py
  requirements.txt       exact pins  (requirements-gbt.txt = the opt-in GBT extra: lightgbm + scipy)
```

## Notes / honesty

- Money/size fields in the journal are exact-decimal strings; the lab parses them
  to `float` at the boundary (it's research, not accounting — the engine keeps
  the exact decimals).
- The `features` realized vol is a practical EWMA (pandas); the **exact** engine
  vol is reproduced only in `validate`, where it's compared to the journaled
  `sigma_1s`.
- Calibration uses one mid-window prediction per window to avoid over-weighting
  long windows and within-window correlation.
