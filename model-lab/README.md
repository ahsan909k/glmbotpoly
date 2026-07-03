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
AUC is a rank-based numpy implementation, so no scipy/sklearn). All ship
prebuilt Windows wheels for 3.12, so nothing compiles.

> **Windows note.** If `python` opens the Microsoft Store (the "App execution
> alias" stub) or `py` isn't found, `python` isn't really on PATH yet. Create the
> venv with the real interpreter — e.g. `py -3.12 -m venv .venv`, or the full
> path
> `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv`.
> After that, activate the venv and plain `python` works (it points at
> `.venv\Scripts\python.exe`). Stage output is UTF-8 (Φ, σ, →); the lab forces a
> UTF-8 console so it renders on a legacy cp1252 code page too.

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

## Stages (one entry command each)

Run them in order — each reads the previous stage's parquet from `out/`. Or run
them all at once with `python -m model_lab.run_all` (which also runs the
standalone **calibration audit** below).

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

### 5. `python -m model_lab.validate`
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

### 6. `python -m model_lab.research`
The reason the depth capture exists — does microstructure predict short-horizon
moves? Aligns depth features to the forward 5 s return and reports, per asset and
pooled → `out/research/metrics.json`:
- **IC** — correlation of imbalance (and microprice tilt) with `fwd_ret_5s`;
- **AUC** — imbalance predicting the up-move, vs a momentum baseline and 0.5.

**Verify:** prints the IC/AUC table. Positive IC / AUC > 0.5 = the depth signal
is worth feeding into fair value. No depth captured yet → says so, cleanly.

### 7. `python -m model_lab.report`
Assembles a single self-contained `out/report.html` (inlined plots) tying the
calibration and depth-signal findings together.
**Verify:** open `out/report.html` in a browser.

### Calibration audit (standalone) · `python -m model_lab.calibration_audit`
**Is the formula model well-calibrated, and does it actually beat the market?**
This stage is **self-contained** — it reads the journal *directly*, so it needs
no prior stage (one command works on fresh journal data). For every resolved
window it lines up two implied probabilities of *Up* against the realized
outcome — the model's `p_up` and the Polymarket **Up-token mid**
`(best_bid+best_ask)/2` — and scores **both** with the **Brier score** *and*
**log-loss**, split by **series** and by **time-remaining** bucket
(early / mid / final-minute / final-20s). It also emits a reliability table
(probability bins vs the actual Up frequency, with sample counts) and a
plain-language **verdict**. Every breakdown is flagged when backed by fewer than
`--min-windows` (default 50) distinct **windows** — too few to trust (snapshots
within one window share a single outcome, so windows are the unit of trust).

Writes `out/calibration_audit/{metrics.json, scores.csv, reliability.csv, report.html}`.

```bash
python -m model_lab.calibration_audit                        # single command on a fresh journal
python -m model_lab.calibration_audit --min-windows 50 --health ready
```

- `--health ready` (default) scores only the snapshots the engine marked healthy
  (what it would actually act on); `--health all` scores every snapshot.
- `--min-windows N` sets the low-sample threshold (default 50).

**Verify:** prints the overall model-vs-market Brier + log-loss and any
low-sample warnings; open `out/calibration_audit/report.html`.

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
    lib/math.py          Φ(z), engine σ_1s EWMA, markout, AUC/Brier/log-loss/reliability
    ingest / features / labels / dataset / validate / research / report  # the 7 pipeline stages
    dataset.py           window-aligned, leakage-safe training set (standalone; reads the journal directly)
    calibration_audit.py model-vs-market calibration (standalone; reads the journal directly)
    hist.py              download historical Binance aggTrades → ../data/aggtrades
    hist_integrity.py    validate the aggTrades store (rows, monotonic ts, dup ids)
    fixtures.py          synthetic journal + depth generator
    verify.py            fixture → all stages → assertions
    run_all.py           run the pipeline stages + calibration audit in order
  tests/                 test_math.py, test_smoke.py, test_dataset.py, test_hist.py
  requirements.txt       exact pins
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
