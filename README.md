# Polymarket Up/Down Trading Bot

High-performance Rust bot for Polymarket's short-duration crypto Up/Down
markets (BTC/ETH × 5m/15m/1h). Paper and live modes share one engine; paper
consumes real live market data with simulated money. Live trading is hard-gated
and disarmed by default. Full spec: [CLAUDE.md](CLAUDE.md).

## 10-minute setup

1. **Install Rust** (stable) via [rustup](https://rustup.rs/) — ~3 min.
2. **Install the task runner and watcher** — ~2 min:
   ```sh
   cargo install just bacon
   ```
3. **Linux only — linker setup** (skip on Windows/macOS, see below) — ~1 min:
   ```sh
   sudo apt install clang mold        # Debian/Ubuntu
   ```
4. **Verify the loop** — first run compiles the (currently tiny) workspace:
   ```sh
   just check
   ```
5. **Iterate** with the watcher while editing:
   ```sh
   just watch                          # bacon: re-checks on save; 'l' = clippy, 't' = tests
   ```

That's it. `just --list` shows all recipes: `check`, `lint`, `fmt`, `test`,
`watch`, `run-paper`, `check-config`, `discover`, `schedule`, `feed`,
`latency`, `build-release`.

### Live discovery probe

`just discover` (= `bot discover`) performs **read-only** requests against
the public Gamma/CLOB REST APIs and prints, for each of the six series, the
currently active window and the upcoming ones: event slug, open/close times
with countdown, condition id, Up/Down token ids, tick size, minimum order
size, fee descriptor, and the per-market resolution source (Chainlink data
stream for 5m/15m, Binance 1H candle for the hourly series). Network
required; exits nonzero if any series fails to discover.

### Scheduler smoke run

`just schedule` (= `bot schedule`) is also **read-only**: it runs the 24/7
per-series window lifecycle state machines against live discovery for every
enabled series, forever. Every lifecycle announcement (Discovered / Open /
Closing / Closed) is logged with the series, slug, and countdown, and a
per-series coverage table prints every 30 seconds (phase, current window,
closes-in, next-window-known, parked count, refresh age, §6 contract state).
No orders are involved; no `Resolved` events appear yet — the market-channel
feed that supplies resolutions is a later task, so closed windows park
silently. Stop with ctrl-c. A 30+ minute run with zero contract violations
across all six series is the §6 acceptance check.

The §11 **clock-skew monitor** runs alongside the smoke run: periodic SNTP
rounds against the `[clock]` servers, with the `ClockSkew` breaker
tripping/clearing on the event bus (printed as `RISK EVENT` lines). NTP
failures never trip the breaker — they warn; treat a persistently stale
clock warning on a VPS as a deployment blocker (likely UDP/123 blocked).

### Feed smoke runs (RTDS + direct Binance)

`just feed` (= `bot feed`, default `--source rtds`) is **read-only**: it
connects to Polymarket's Real-Time Data Socket and streams the Binance-source
(`crypto_prices`) and Chainlink-source (`crypto_prices_chainlink` — the
resolution-grade feed) prices for BTC and ETH side by side, printing a
per-stream table every second (last value, exchange-timestamp age, receive
age, tick count, live/STALE) plus every health transition as a `FEED HEALTH`
line. The driver keeps the socket alive with a 5-second PING and reconnects
automatically with jittered backoff — kill the network mid-run to watch all
four streams go STALE immediately, the reconnect attempts space out, and
each stream recover on its first new tick. RTDS streams **all** symbols per
topic (the only multi-symbol form it supports); untracked ones are dropped
client-side. Each (re)connect first sends per-symbol filtered subscribes to
collect the ~2-minute backfills that pre-seed the model, and a stream that
starves for 6× the staleness threshold (30 s default) triggers a connection
recycle — the self-heal for server-side subscription failures observed live.
Stop with ctrl-c. Per-tick lines log at `RUST_LOG=feed=debug`.

`just feed-binance` (= `bot feed --source binance`) is the same smoke run
over the **direct Binance WebSocket** — BTCUSDT/ETHUSDT top-of-book
(`bookTicker`, published as the midpoint) and trade prints, the bot's
lowest-latency signal. Subscription rides the connect URL (the client sends
zero frames; Binance's 20-second server pings are answered automatically),
trade streams carry a much looser staleness threshold than book streams
(quiet markets legitimately print nothing for seconds), and the venue's
24-hour connection limit lands on the ordinary reconnect path. Note:
`Mid` ticks have no exchange timestamp on the wire, so their `ex-age` column
mirrors `rx-age` by construction.

`bot feed --source <rtds|binance> --raw <file>` additionally taps every
in/out application frame to a JSONL file — this is how the committed wire
fixtures in `crates/feed-rtds/tests/fixtures/` and
`crates/feed-binance/tests/fixtures/` were captured.

### Live order-book ladder (CLOB market channel)

`just ladder` (= `bot ladder`, default `--series BTC-5m`) is **read-only**:
it runs the scheduler for one series with the feed-clob supervisor attached
— one market-channel WebSocket connection per window (token pair), the next
window pre-connected before the current closes (gap-free rollover by
construction) — and redraws the current window's Up/Down L2 ladders in
place every 250 ms. The header prints `polymarket.com/event/<slug>` so the
books can be verified against the website side by side. This run also wires
feed-clob's deduplicated `market_resolved` events into the scheduler, so
parked windows resolve live (`RESOLVED <slug> -> Up|Down` in the events
pane). Flags: `--series <KEY>` (any of the six), `--depth <N>` levels per
side (default 8), `--recycle-after <secs>` forces one connection drop to
demonstrate reconnect-and-resync continuity, and `--raw <file>` taps every
frame of every connection to a JSONL capture — the fixture path for
`crates/feed-clob/tests/fixtures/`. The in-place redraw overwrites console
log lines; everything also lands in the rolling log file under `data/logs/`.

### Feed comparison (lead/lag + basis measurement)

`just compare` (= `bot compare`) is **read-only**: both feed drivers on one
bus, with a per-asset summary printed every minute over a rolling 5-minute
window — per-stream tick rates, pairwise value bases (bps), exact-value
match lags (which direct stream RTDS-Binance actually republishes), and
best-lag cross-correlations (who leads, by how much). Let it run 10+
minutes for stable numbers. This is the measurement that calibrates the
model's Chainlink-minus-Binance basis correction (CLAUDE.md §8).

### Vol-estimator smoke run (σ_1s)

`just vol` (= `bot vol`) is **read-only**: both feed drivers on one bus
feeding four `model::VolEstimator` lanes — (BTC, ETH) × (`chainlink:vendor`,
`binance:mid`) — with a per-lane table every second: σ_1s (and the same in
bps per √second), warmup quality, floor/cap clamp state, and fold/re-anchor/
drop counters. The chainlink lanes go READY within seconds (the RTDS
backfill pre-seeds ~60–120 one-second bars); the binance lanes warm up over
~60 s of live data. Half-life/floor/cap flow from `engine.defaults`
(`ewma_half_life_secs`, `vol_floor_1s`, `vol_cap_1s`); expect BTC σ_1s
around 3e-5..3e-4 (0.3–3 bps/√s) in normal markets with clamp `-` and
reanchors 0 on healthy feeds.

### Fair-value smoke run (p_up vs book mid)

`just fair` (= `bot fair`, default `--series BTC-5m`) is **read-only**: the
scheduler, feed-clob, and both price feeds run on one bus feeding the full
model (§8) for the chosen series' current window. It renders, in place every
250 ms:

- **strike** `K` — the price-to-beat the bot captures itself from the
  Chainlink RTDS stream at the window boundary (the venue exposes no live
  strike). It shows `FROZEN` once settled, plus the `before`/`after` candidate
  offsets from open. The RTDS backfill burst recovers the boundary print even
  when started mid-window, so `K` typically freezes within seconds.
- **inputs** — Chainlink and direct-Binance prices with ages, the
  basis-corrected anchor price `S`, the live basis (bps), and the
  corrected-vs-Chainlink divergence.
- **model** — `σ_1s`, `σ_τ = σ_1s·√τ`, `z = ln(S/K)/σ_τ`, `p_up = Φ(z)`, the
  health gate (`warming`/`healthy`/`unreliable`), and which input was anchored.
- **book** — the Up-token mid and `|p_up − mid|`, flagged `SANITY BREACH` if it
  exceeds `risk.sanity_bound` (0.10) for `risk.sanity_bound_duration_ms` (3 s).

Each computed value is published as `Event::Model` on the bus (the integration
point a future engine/dashboard consumes). Calibration JSONL lands in
`data/calibration/fair-{series}-{stamp}.jsonl`: ~1 Hz model snapshots, strike
freeze/revision records, resolution outcomes, and **post-hoc verification** —
a few minutes after each window resolves the venue publishes
`eventMetadata.priceToBeat`, which the bot fetches and reconciles against its
captured strike (`exact`/`mismatch`/`unavailable`), confirming which
`StrikeRule` the venue actually uses. Mid-window, `p_up` should track the book
mid inside the sanity band; near close it saturates toward the resolving side.

### Journal capture (record real data for replay)

`just record` (= `bot record --out-dir data/journal`, optional `--series`) is
**read-only**: it wires the scheduler + all three feeds onto the bus (like the
smoke runs but with no model) and records **every** `Event` — price ticks,
books, prints, lifecycle, health — to disk for offline strategy work and
replay. Records **all enabled series** by default, or just one with `--series`.

Segments are **gzip-compressed, rotated JSONL** under the output directory
(`journal-{YYYYMMDD-HHMMSS}-{NNNNN}.jsonl.gz`), rotated at 128 MiB or hourly,
so the capture rolls across files indefinitely — built for **days** of data.
Each line is `{"seq","ts_local_ms","rec":{"type":…,…}}`; `gunzip`/`zcat` reads
them directly, and `journal::ReplayReader` decodes and reconstructs the bus
`Event`s (the same path the venue-parity tests replay through `venue-paper`).
The writer runs off the bus on its own thread; if the disk stalls, events are
**dropped and counted** rather than back-pressuring the bus — a nonzero
`dropped` in the ctrl-c summary means the capture is incomplete.

### Replay & parameter sweep (feature-gated backtester)

`just replay` and `just sweep` re-run the **full engine** (the risk-managed quote
manager + momentum/late takers) against a recorded journal, driving a fresh
**seed-stable paper venue**, and emit the §10 series-comparison table. They live
behind a cargo feature so the core bot stays lean (CLAUDE.md §3) — run them with
`--features replay`:

```sh
# Deterministic replay of a recording → the per-series comparison table.
cargo run -p bot --features replay -- replay data/journal            # or: just replay
cargo run -p bot --features replay -- replay data/journal --window 7d --out r.json

# Parameter sweep over the four quoting knobs → a ranked table (best NetPnl first).
cargo run -p bot --features replay -- sweep data/journal \
  --min-edge 0.01,0.02 --gamma 0.05,0.1 --cancel-theta 0.005,0.01 --taker-buffer 0.005,0.01 \
  --rank net-pnl --parallel --out sweep.json                          # or: just sweep
```

How it works: the recorded engine/venue **outputs** (orders, fills, inventory,
settlements) are dropped; every other recorded event is replayed into the engine
+ venue exactly as the live `bot run` loop feeds them. Re-running the engine
produces **fresh** orders → the paper venue produces fresh fills → those drive a
fresh analytics fold. The recorded model snapshots are replayed as-is (model
params aren't swept). Time is driven from each record's `ts_local_ms` on a
`start_paused` clock, so a multi-hour recording replays in seconds.

The **sweep** grids over four `EngineParams` fields — `min_edge`,
`gamma_inventory_skew` (skew), `reprice_threshold_theta` (the cancel threshold;
`cancel_market_theta` scales with it to preserve the §8 ordering), and
`momentum_buffer` (taker buffer). An omitted dimension uses the config base
value; `--rank <col>` picks the ranking column (`net-pnl` default, any §10
column); `--parallel` fans grid points across cores. Each point is a full,
independent replay; the report is sorted deterministically.

**Determinism** is the contract: two replays of the same recording + seed produce
**byte-identical** analytics (proven by `cargo test -p analytics --features
replay`). The replay reproduces live paper *closely* — same recorded inputs, same
engine/venue code, fixed seed — but is **not** bit-for-bit against the original
live session (which used the wall clock + a wall-seeded latency RNG); the exact,
testable guarantee is replay-vs-replay determinism. To produce a recording to
replay, run `just record` against the live venue first.

### Main run mode (`bot run`)

`just run` (= `bot run`, optional `series="BTC-5m"`) is **the** trading command
(CLAUDE.md §5): it starts paper trading on **all enabled series at once** under a
supervision tree. Unlike the smoke subcommands, it wires the *real* engine — the
scheduler + all three feeds + the multi-asset fair-value model + the
single-gateway **risk manager** (the quote manager and the momentum / late-window
takers, behind the §11 breakers) + the paper venue + the journal + analytics +
the dashboard — onto one bus. Real data, paper money; the dashboard binds
`config.dashboard.bind`.

It adds three things over the smoke modes:

- **Supervision.** Every long-running task is restarted with backoff if it dies,
  while the rest keep running. A *critical* dependency's death (a feed or the
  scheduler) triggers an immediate **cancel-all** and gates new order flow until
  it recovers (a restarted clob is re-seeded with the current windows so it
  reconnects without waiting for the next announcement). Watch the `run`-target
  logs for `supervised task …` / `scheduling restart` / `… restarted`.
- **Resilient startup self-check.** The bot **refuses to trade** until the clock
  is sane (the §11 skew monitor's verdict), discovery has a current window for
  every enabled series (the scheduler's `Open` announcements), and the feeds are
  healthy (first ticks on the bus). It stays up retrying and **auto-arms** once
  healthy — logging `ARMED — startup self-check passed` — and never exits on a
  slow/failed self-check (a skewed clock keeps it up but un-armed until you
  resync; gate it with `[run].require_clock_check` / `require_discovery_check`).
- **Graceful shutdown** on Ctrl-C *and* (on Linux) `SIGTERM`: stop strategies →
  cancel all open orders, draining until **zero remain** (logged) → flush the
  journal → exit.

Resource discipline is built in: bounded channels everywhere (bus 256 with
backpressure, journal 16384 drop-and-count, window/market 64), and a periodic
`resource report` log line (RSS on Linux from `/proc/self/statm`, per-window book
counts, settled-inventory prune) so a 24/7 session stays memory stable. The
`[run]` config section tunes the supervision backoff, the self-check gates, and
the cadences. **Acceptance:** a multi-hour session — six series rolling, dashboard
live, the RSS line flat, clean shutdown leaving zero open paper orders.

**Shadow mode (observation only).** Set `[shadow] enable = true` to run the
champion `dir10_full` model live alongside the engine: it predicts every 5 s per
active BTC/ETH × 5m/15m window and journals to `data/shadow/*.jsonl.gz`,
**influencing nothing** (a non-critical, venue-free task — proven by
`crates/bot/tests/shadow_order_flow.rs`). Export the model first
(`python -m model_lab.export_champion` → `models/model_dir10_full.txt`); the
dashboard's Summary view shows a "Shadow" tile (predictions/min, last p_up,
coverage, and a "model stale, refit due" flag). The mandatory feature-parity
guard is `just shadow-parity` (offline 1e-6) + `python -m model_lab.shadow_parity`
(nightly live-vs-offline). See CLAUDE.md §15 (2026-07-12).

### 24-hour soak checklist

Before a build is allowed near real money it must survive a 24-hour paper soak.
Run the deterministic chaos suite first, then leave `bot run` up for a day and
watch the numbers below.

**Pre-flight — the chaos suite (local, deterministic, ~seconds).** `just chaos`
(= `cargo test -p bot --features chaos --test chaos -- --test-threads=1`) injects
each failure class — kill each WebSocket, stall each feed past the staleness
threshold, a matching-engine restart notice, a clock jump, a discovery failure at
rollover, a process restart mid-window — and asserts the §11 invariants: no orphan
open orders, the right breaker trips **with a journaled cause**, trading halts and
resumes exactly per the rules, and state rebuilds exactly from the journal. It is
feature-gated (off in the default `just test`/`just lint` loop) and paper-only.
Acceptance: run it ~10× back-to-back — **100% green, zero flakes** — and lint the
gated code with `cargo clippy -p bot --features chaos --all-targets -- -D warnings`.

**Start the soak.** Resync the VPS clock (`bot schedule` for ~5 min should show no
`ClockSkew` trip), then `just run` (all six series, paper) — or `bot dashboard
--with-model` to watch it in a browser. Wait for `ARMED — startup self-check
passed` in the log; until then the bot is intentionally not trading.

**Watch (dashboard + logs, sampled every 2–4 h).**

- **Markout health (the headline).** Series-comparison → average 5 s markout must
  not be persistently negative (the adverse-selection alarm); a series whose
  markout trends red is being picked off — pull it.
- **Breaker trips.** Risk panel green in steady state; every trip that *does* fire
  must carry a journaled cause — cross-check with the journal
  (`JournalIndexReader::breaker_trips()` / the sqlite index). A trip with no
  cause, or a breaker stuck tripped, disqualifies the build.
- **Memory.** The `resource report` log line (RSS on Linux, per-window book
  counts, settled-inventory prune) must stay **flat** over the full day — any
  steady RSS climb is a leak.
- **Reconnect counts.** Per-feed reconnects should be bounded self-heals, not a
  loop; a feed that never reconnects (permanent stale) is a failure.
- **Feed + WS health.** All feed-health lanes live; windows roll cleanly with the
  late-window gate marks appearing ≤30 s before close; no rejected-order or
  cancel loops in the fills blotter; equity curve and per-series win-rate stable
  comparing hour 0–4 vs 20–24.

**Mid-run drills.** `bot control kill` → TRADING HALTED banner + zero resting
orders → `bot control reset` → quoting resumes. Then Ctrl-C **and** (on Linux)
`kill -TERM <pid>` each → clean shutdown logging **zero open paper orders**.

**Disqualifies the build:** any orphan open order at shutdown; a breaker that
trips without a journaled cause; RSS growth / a memory leak; a feed that never
reconnects; a series that stalls at rollover; any panic or crash; persistently
negative markout (adverse selection); or an unresolved clock-skew trip.

### Dashboard (REST + WebSocket)

`just dashboard` (= `bot dashboard`, optional `--series`, optional
`--with-model`) serves the operator dashboard (CLAUDE.md §10) over a **live paper
pipeline**: the scheduler + feed-clob + the paper venue across **all enabled
series** (or one with `--series`), with a trivial two-sided quoter — real data,
paper money. The axum server binds `config.dashboard.bind` (default
`127.0.0.1:8080`). Open **`http://<bind>/`** in a browser for the single-page UI;
the REST snapshots + WebSocket push are the backend it runs on.

**`--with-model`.** By default the run is lean (no fair-value model). Pass
`--with-model` to additionally wire feed-rtds + feed-binance and the §8
fair-value model across every series, so the **Live** view's *fair vs book mid*
panel and the **Fills** blotter's *5-second markout* coloring populate with real
data. This opens two extra WebSocket connections; the default run leaves the
model off.

**UI.** A static, phone-usable single page is served at `/` (`/app.css`,
`/app.js`) — embedded in the binary, no build step, no external assets. The
landing screen is a calm, explanatory **Summary**; the detailed operational views
sit behind a clearly-secondary **Details** nav (so nothing safety-critical is
lost). It live-updates over the WebSocket (with a visible **"Reconnecting…"**
notice on disconnect and a 1 s countdown ticker) and reconnects on its own.

- **Summary** (default) — per-series tabs (All + each enabled series) + a
  Paper/Live toggle, then: an at-a-glance **health row** (starting capital,
  current equity, today's P/L, win rate) and a one-line plain-English **status
  sentence**; the **two-bucket explainer** (see "How to read it" below); a calm
  **live-activity** strip (orders placed/cancelled/resting/filled per minute, and
  the cancel-first reflex's median time-to-replace / time-to-cancel against their
  config targets); and collapsed-by-default dropdowns for recent fills, recent
  cancels, open positions, and recent resolved windows.
- **Details → Series** — the §9.3 decision table (sortable, min-sample rows muted).
- **Details → Live window** — chips for each active window; the selected window's
  Up/Down book ladder with **our resting quotes highlighted**, fair-vs-mid, a
  countdown with the late-window gate zones marked (late-taker ≤30s, no-ATM ≤25s,
  cancel-all ≤5s), inventory + pair-cost, and recent prints.
- **Details → Fills** — the chronological blotter, filterable by series, each row
  **colored by its 5-second markout** with maker / taker / late attribution tags.
- **Details → Risk** — every breaker + last-trip cause, the risk snapshot, feed
  health, user-WS connectivity, per-asset model health, and **Kill / Reset**.
- **Details → Controls** — paper/live badges, the live equity curve, the
  paper-capital editor (set absolute, or ±$1k), the §11 live-arming flow, the
  per-series enable toggles, the safe-listed parameter editor, and **Kill / Reset**.

**How to read it — the two-bucket explainer.** The Summary answers "why did I earn
for an hour then give the profit back?" A settled window's realized PnL splits
**exactly** into two plain buckets that always sum to total profit/loss:

- **Profit from completed pairs** — money locked from pairs where *both* sides
  filled for under $1. This is the fee-free market-making edge; it should be green.
- **Profit/loss from stuck legs** — money made or lost on legs that filled on one
  side and never got a cheap matching fill before the window closed (plus the taker
  fees paid chasing them). When the market **trends**, legs get stranded and this
  bucket goes sharply **red** — that is exactly the "gave the profit back" cause,
  and the status sentence flips to *Caution* when it does (or when the 5-second
  markout shows quotes are being picked off). Beneath the buckets: how many pairs
  completed vs how many legs were stranded, the average cost to complete a pair,
  and the average loss per stranded leg.

The live-activity **time-to-replace** tile (cancel → fresh quote on the same side)
shows how fast the yank-it-back reflex actually is, green/red against the
`min_requote_interval_ms` target; **time-to-cancel** (the true cancel round-trip)
is shown in live mode and reads "—" in paper, where the simulated venue has no
observable cancel round-trip.

**Auth.** REST is `Authorization: Bearer <token>`; the WebSocket takes the token
as a `?token=` query (browsers can't set a header on a `WebSocket`). The token is
`BOT_SECRET_DASHBOARD_TOKEN` — **required** for a non-loopback bind (the server
refuses to start otherwise), optional on loopback (dev). For a remote bind, open
`http://<host>/#token=<token>` once — the page stores the token and cleans the
URL (or paste it via the ⚙ button). `/health` and the static UI (`/`, `/app.css`,
`/app.js`) are unauthenticated; every `/api/*` data and control route is gated.

Both trading modes are **namespaced and simultaneously available** (`?mode=paper`
default, or `?mode=live`); this pipeline only populates `paper`, so `live` is
present-but-empty until the live orchestrator lands. Endpoints:

| Endpoint | What it returns |
|---|---|
| `GET /health` | machine-readable status (`ok`/`degraded`/`down`) + feed/breaker/model rollup (no auth) |
| `GET /api/summary?mode&series&window=today\|7d\|all&days=N` | the calm landing: headline, status sentence, two-bucket explainer, live-activity strip (scoped to a series or all) |
| `GET /api/settlements?mode&series&limit` | recent resolved windows, each split into the two buckets (completed pairs vs stuck legs) |
| `GET /api/overview` | both modes' badges, equity curve, wallet/ledger, paper capital |
| `GET /api/series-comparison?mode&window=today\|7d\|all&days=N&sort=<col>&dir=asc\|desc` | the §9.3 decision table (sortable) |
| `GET /api/windows?mode` | active windows (shared book/model + this mode's inventory) |
| `GET /api/windows/{Series@open_ms}?mode` | one window: ladders, our resting orders, prints, fair-vs-mid, inventory |
| `GET /api/fills?mode&limit&window&since_ms` | the fills blotter (newest first); each row carries `markout_5s`/`markout_pending` + `attribution` (maker/taker/late) |
| `GET /api/risk?mode` | breakers, feed/book/model health, WS connectivity |
| `GET /api/params` | the current safe-listed parameters + paper capital |
| `POST /api/control/kill` | global kill: cancel everything + halt (latched) |
| `POST /api/control/reset` | clear the operator-latched breakers, resume |
| `POST /api/control/reset-daily-stop` | clear only the daily-stop latch |
| `POST /api/control/paper-capital` | `{"amount":"N"}` (set absolute) or `{"delta":"±N"}` (adjust) |
| `POST /api/control/enable-series` / `disable-series` | `{"series":"BTC-5m"}` — runtime series toggle |
| `POST /api/control/set-param` | `{"series":"BTC-5m"\|null,"key":"min_edge","value":"0.02"}` — safe-listed tunables only |
| `POST /api/control/arm-live/begin` → `arm-live/confirm` `{"phrase":"…"}` → `disarm` | the §11 multi-step live-arming flow |
| `GET /api/control/status` | the control-plane state (kill, enabled series, arming gates, param overrides) |
| `GET /api/ws?token=<t>` | WebSocket: `hello`, then `equity`/`quote`/`fill`/`breaker`/`lifecycle` (+ `top`/`model`) updates |

Every control command is **validated, journaled with origin** (a `ControlAudit`
record → the `command_audit` sqlite table) **and acknowledged with the resulting
state**: each `POST /api/control/*` returns a `ControlOutcome`
(`{kind:"accepted"|"rejected"|"conflict", error?, state}`) and the HTTP status
maps `accepted`→200, `rejected`→400, `conflict`→409 (e.g. arming with a gate
missing). The routes return `503` on a read-only deployment (no request sink) and
require the bearer token like every `/api/*` route.

**CLI control (`bot control …`).** The same control plane is reachable from the
command line — `bot control <kill|reset|reset-daily-stop|set-capital N|adjust-capital ±N|
enable-series KEY|disable-series KEY|set-param KEY VAL [--series KEY]|arm-live|
confirm-arm PHRASE|disarm|status>` is a thin HTTP client to the **running** bot's
dashboard control API (bind from `config.dashboard.bind`, token from
`BOT_SECRET_DASHBOARD_TOKEN`, origin tagged `cli` in the audit trail). It prints
the JSON ack and exits non-zero on a refusal. Examples:

```sh
bot control status
bot control disable-series BTC-5m
bot control set-param min_edge 0.02 --series ETH-5m   # accepted (safe-listed)
bot control set-param touch_size 20                   # rejected (structural — requires restart)
bot control arm-live && bot control confirm-arm arm-live-i-accept-real-money-losses
```

Probe it (loopback dev, no token needed):

```sh
curl -s localhost:8080/health
curl -s localhost:8080/api/overview
curl -s "localhost:8080/api/series-comparison?mode=paper&window=all"
```

Without `--with-model` the "fair vs mid" panel and the markout coloring stay
empty (no model data); the series-comparison table still populates once windows
settle (an in-process `InventoryManager` folds the paper fills into settlements).
The **Control** card on the overview shows the live-arming gates, per-series
enable toggles (a disabled series stops quoting in the running loop), and a
safe-listed parameter editor. The live namespace stays empty until the live
orchestrator lands — the control plane holds the runtime state (enabled series,
parameter overrides, the gate-3 arm flag) that the future orchestrator reads.

**Acceptance — paper run against live data (screenshot checklist).** Start
`bot dashboard --with-model` (loopback, dev — no token needed) and open
`http://localhost:8080/`. Let it warm up a minute (the model needs ~60 ticks +
a window open to produce `p_up`), then verify by eye that each updates smoothly
in real time:

1. **Summary** (landing) — the headline tiles populate (equity, today's P/L, win
   rate); the **two-bucket explainer** fills as windows settle and its two values
   sum to the total; the status sentence reads *Healthy* / *Caution* / *Warming
   up*; the live-activity tiles tick (placed/cancelled/resting/fills per minute,
   time-to-replace green/red, time-to-cancel "—" in paper); the per-series tabs and
   Paper/Live toggle re-scope it; the four dropdowns expand with detail. Kill the
   bot (Details → Controls) and the *TRADING HALTED* notice shows above the nav.
2. **Details → Controls** — the equity curve advances; paper badge shows *running*.
3. **Details → Live window** — active-window chips for the enabled series; pick one and watch the
   Up/Down ladder churn, with at least one bid level highlighted as **ours**
   (the quoter's resting order); the countdown decrements each second and its bar
   crosses the three gate marks (late-taker ≤30 s, no-ATM ≤25 s, cancel-all ≤5 s)
   near close; **fair vs mid** shows a non-blank `p_up`, the Up mid, and their Δ,
   with a model-health pill; inventory + pair-cost and the scrolling prints fill.
4. **Details → Fills** — rows appear as the quoter trades; each is colored by its
   5-second markout (some green/red once matured, some "…" while pending) and
   tagged maker / taker / late; the series filter narrows the list.
5. **Details → Risk** — breaker chips are green (ok); feed-health shows all-live (or
   a red row during an RTDS hiccup); model health shows the per-asset tier; press
   **Kill** → the *TRADING HALTED* banner + a tripped `Manual` breaker, then
   **Reset** clears it.

(The live `--with-model` smoke is the operator's interactive step, like the
`bot fair` / `bot ladder` acceptances — the backend folds and the markout/model
math are covered by unit + integration tests.)

### Latency benchmark (VPS region selection)

`just latency <label>` (= `bot latency --label <label>`, optional
`--out <file>`) is **read-only** and measures, against live endpoints:
warm-pool REST round-trips to the CLOB host (`/time`, `/ok`), the CLOB
server-time delta, an NTP offset reading, and WebSocket
connect/first-message/inter-message-gap stats for the CLOB market channel
(subscribed to all live windows' tokens — gap stats are activity-dependent)
and RTDS crypto prices (steady ticks — the reliable network metric). A JSON
report (`schema_version` 1) is written to `data/latency/` and a p50/p95/p99
summary table prints to the console.

Run it from **every candidate VPS region** (Polymarket's matching engine is
in AWS eu-west-2 per the docs) and from the dev PC as a baseline. Feed the
REST p50/p95 into `paper.placement_latency` / `paper.cancel_latency` and the
engine latency premium **manually** — measured numbers are never auto-wired
into trading config. The full region-selection procedure (choose by CLOB REST
**p99**) and a results-table template are under
[Deployment → VPS region selection](#vps-region-selection).

### Live execution port (`venue-check` + live arming)

`venue-api` is the venue-agnostic execution port (the trait the engine depends
on); `venue-live` implements it over the official Polymarket Rust SDK
(`polymarket_client_sdk_v2`). `just venue-check` (= `bot venue-check`) is an
**offline** demonstration — no credentials, no network — that prints the live
params and the constructed (unsent, unsigned) request for each order class
(GTC/GTD post-only, FAK/FOK marketable), exercising the same translation the
live adapter uses. The GTD 60-second security threshold is applied **inside**
the adapter (the on-wire expiration is floored to ≥ `now + 60s`), so the
`venue-check` GTD row shows a bumped expiration.

**Live arming (CLAUDE.md §11).** A network-capable, order-submitting adapter is
only built by `LiveVenue::connect`, which fails closed unless all three gates
pass: `live.enabled` (config) **and** `BOT_SECRET_LIVE_CONFIRM` = the exact
phrase **and** the dashboard arm (gate 3, a later task). `bot live` routes
through it and, since gate 3 does not exist yet, always refuses with `NotArmed`
**before any SDK client is built or any network call is made**. The
operator-specific, non-secret **funder** (deposit-wallet) address is the only
live network parameter in config (`[live].funder`, a `0x…40-hex` address,
required when `live.enabled`); host / chain id / signature type are code
defaults (deposit-wallet / `POLY_1271`, the path for new API users).

**Operator-only live smoke test.** `crates/venue-live/tests/live_smoke.rs`
places a tiny far-from-touch post-only order on the LIVE venue and cancels it.
It is triple-gated (the `live-smoke` cargo feature **and** `#[ignore]` **and**
`BOT_LIVE_SMOKE=1`) and **never runs by default** — it spends real funds. The
env vars it needs are documented at the top of that file; run it deliberately
with `cargo test -p venue-live --features live-smoke -- --ignored live_smoke`.
One-time on-chain token approvals (the gasless relayer flow) are a prerequisite
for any live order to fill and are a **later task**; until then a missing
allowance surfaces as `Rejected(InsufficientFunds)`.

## Configuration

Layered, lowest to highest precedence (later layers override earlier):

1. **Code defaults** — pinned to the committed file by a test.
2. **`config/default.toml`** — committed defaults, **required** at boot.
3. **`config/bot.local.toml`** — optional operator override, gitignored
   (matches the `*.local.toml` ignore pattern).
4. **Environment variables** — `BOT_<SECTION>__<KEY>` (double underscore
   between path segments), e.g. `BOT_PAPER__STARTING_CAPITAL=25000`,
   `BOT_RISK__DAILY_STOP_LOSS=100`.

The config directory is `./config` by default; override with
`--config-dir <path>` or `BOT_CONFIG_DIR`. Inspect the effective result with
`just check-config` — it loads, validates (all violations reported at once),
and prints the merged configuration. Per-series engine overrides
(`[engine.series.BTC-5m]`) are TOML-only; see the comments in
`config/default.toml`.

### Secrets

Secrets come **only** from environment variables — never from files (the
loader rejects secret keys found in TOML, and secret values are redacted in
all logs and output). Nothing auto-loads `.env`; see
[.env.example](.env.example) for the full list:

| Variable | Purpose |
|---|---|
| `BOT_SECRET_DASHBOARD_TOKEN` | Dashboard auth (required for non-loopback bind) |
| `BOT_SECRET_PM_API_KEY` / `_SECRET` / `_PASSPHRASE` | Polymarket API credentials (live only) |
| `BOT_SECRET_PM_PRIVATE_KEY` | Order-signing key (live only) |
| `BOT_SECRET_LIVE_CONFIRM` | Live-arming confirmation phrase (gate 2 of 3) |

Live trading needs three independent gates (CLAUDE.md §11): `live.enabled` in
config, the exact confirmation phrase in the environment, and the dashboard
arm action. Everything defaults to paper.

### Logging

Console + rolling file (`data/logs/bot.<date>.log` by default; rotation,
retention and directory under `[log]` in config). Filtering uses standard
tracing directives: `RUST_LOG` wins when set, otherwise `log.default_filter`
applies — e.g. `RUST_LOG=info,engine=debug,feed_clob=trace`. Both price-feed
drivers log under the shared target `feed` with a `feed = "rtds" | "binance"`
field (the generic driver in `feed-util` cannot emit per-feed targets —
tracing targets are static).

## Build performance (why iteration stays fast)

These are mandated by CLAUDE.md §3 — do not regress them:

- **Many small crates** (`crates/*`): Rust recompiles per crate, so touching
  one subsystem rebuilds only that crate and its dependents.
- **Dev profile**: workspace code compiles unoptimized with
  `debug = "line-tables-only"` (short link times); **all external dependencies
  compile at `opt-level = 2`** via `[profile.dev.package."*"]` — built once,
  cached, fast at runtime even in dev.
- **Capped parallelism**: `.cargo/config.toml` sets `jobs = 10`
  (operator's 12 logical cores − 2) so builds never freeze the desktop.
- **Iteration = `cargo check`/clippy via bacon**, never full builds. Release
  builds are for deployment only.

### Linker setup (Linux)

`.cargo/config.toml` configures the [mold](https://github.com/rui314/mold)
linker for `x86_64-unknown-linux-gnu` (requires `clang` and `mold` installed,
step 3 above). **Fallback:** if mold is unavailable on your distro, install
`lld` and change the rustflags line in `.cargo/config.toml` to:

```toml
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

Windows and macOS are unaffected (the section is target-scoped); the default
linker is used there.

### VPS notes (building on the VPS)

Build **on the target Linux VPS** (or an identical instance — same distro,
glibc, and CPU family). Cross-compiling from the Windows dev PC is impractical:
mold is Linux-only, and `aws-lc-rs` (the SDK's rustls provider — still rustls,
*not* openssl) and `rusqlite`'s bundled SQLite both compile C.

**Prerequisites** (Debian/Ubuntu):

```sh
sudo apt-get update
sudo apt-get install -y build-essential clang cmake mold pkg-config ca-certificates
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
rustup toolchain install stable      # must satisfy rust-version = 1.94 (edition 2024)
```

**Build** with full parallelism (the committed `jobs = 10` is a dev-PC value)
and record the cold-build cost the first time:

```sh
CARGO_BUILD_JOBS=$(nproc) cargo build --release -p bot --timings
# -> target/release/bot ; archive target/cargo-timings/*.html with the deploy notes
```

- **RAM/CPU:** thin-LTO + `codegen-units = 1` linking is RAM-hungry — provision
  **≥ 4 GB RAM / ≥ 2 vCPU** (a 1–2 GB box gets OOM-killed at the LTO link; a
  temporary swapfile is a stopgap). The cold build is heavy (the aws-lc C build +
  alloy + the LTO link dominate) — **record the actual minutes**.
- **`target-cpu=native`** is opt-in and **only** when the build host CPU == the
  run host CPU (e.g. building on the production VPS itself), else the binary
  `SIGILL`s on a missing instruction:
  ```sh
  RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=$(nproc) cargo build --release -p bot
  ```
- **Portability check:** `ldd target/release/bot` should list only the system
  libc/libm/libgcc/pthread (TLS is rustls, SQLite is bundled, flate2 is pure
  Rust), so the binary moves cleanly to a same-distro host.

If mold is unavailable, fall back to `lld` per [Linker setup](#linker-setup-linux).

## Deployment (VPS)

Operational scaffolding for running the bot 24/7 on a Linux VPS. **Paper phase —
no live keys:** the systemd unit and env file are paper-only, with the live
secrets left as commented placeholders. The artifacts live in
[`deploy/`](deploy/): `bot.service`, `bot.env.example`, `bot.local.toml.example`,
`upgrade.sh`, `backup.sh`. (The release profile is tuned in `Cargo.toml`
`[profile.release]`; build it per [VPS notes](#vps-notes-building-on-the-vps).)
Make the scripts executable after checkout — `chmod +x deploy/*.sh` — or run
them as `bash deploy/<script>.sh`.

### Directory layout

```
/opt/bot/bin/
    bot-<version>-<sha>      # the versioned binary
    bot -> bot-<version>-<sha>   # symlink; systemd ExecStart points here (atomic swaps)
/etc/bot/
    config/default.toml     # copied from repo config/default.toml (REQUIRED)
    config/bot.local.toml   # from deploy/bot.local.toml.example (absolute paths + retention)
    bot.env                 # from deploy/bot.env.example (secrets/env), mode 0600
/var/lib/bot/data/          # the only writable tree: journal/, journal.sqlite(+wal/shm), logs/
```

Data lives under `/var/lib/bot` (FHS variable state), out of `/opt` and `/etc`,
so the hardened unit keeps the binary + config read-only. The absolute paths are
set in `bot.local.toml` (`log.dir`, `journal.dir`, `journal.sqlite_path`).

### systemd service + secrets

Secrets are **environment-only** (never in TOML). Paper phase needs none, unless
you expose a non-loopback dashboard (then `BOT_SECRET_DASHBOARD_TOKEN`). Install:

```sh
# 1. Dedicated unprivileged user.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin bot

# 2. Directories.
sudo mkdir -p /opt/bot/bin /etc/bot/config /var/lib/bot/data

# 3. Binary (versioned) + symlink.
V="0.1.0-$(git rev-parse --short HEAD)"
sudo install -m 0755 target/release/bot "/opt/bot/bin/bot-${V}"
sudo ln -sfn "/opt/bot/bin/bot-${V}" /opt/bot/bin/bot

# 4. Config + secrets (edit the copies in place).
sudo install -m 0644 config/default.toml            /etc/bot/config/default.toml
sudo install -m 0644 deploy/bot.local.toml.example  /etc/bot/config/bot.local.toml
sudo install -m 0600 deploy/bot.env.example         /etc/bot/bot.env
sudo chown root:bot /etc/bot/bot.env                # systemd reads it as root pre-drop

# 5. Ownership.
sudo chown -R bot:bot /var/lib/bot

# 6. Unit.
sudo install -m 0644 deploy/bot.service /etc/systemd/system/bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now bot

# 7. Verify.
systemctl status bot
journalctl -u bot -f          # watch the startup self-check, then "ARMED"
```

The unit drains on `SIGTERM` (`TimeoutStopSec=30` > the 5 s order-drain +
journal flush), `Restart=on-failure` with a crash-loop guard, captures stdout
into journald (the bot *also* writes its own file logs), and applies a
conservative hardening set — **verify `MemoryDenyWriteExecute` /
`SystemCallFilter` / `RestrictAddressFamilies` in a staging run** before trusting
them (see comments in `deploy/bot.service`).
`systemd-analyze verify deploy/bot.service` checks the unit; `systemd-analyze
security bot` scores the hardening.

### Dashboard remote access

**Recommended — loopback + SSH tunnel.** Keep the default
`bind = "127.0.0.1:8080"` (no open port, no token, no TLS):

```sh
ssh -N -L 8080:localhost:8080 bot-vps     # then browse http://localhost:8080
```

A loopback bind needs **no** `BOT_SECRET_DASHBOARD_TOKEN`. *Alternative
(documented only):* a non-loopback `bind = "0.0.0.0:8080"` requires the token
(boot-enforced) **and** a firewall allowlist **and**, because the dashboard is
plain HTTP, a TLS reverse proxy (nginx/caddy) for any public use.

### Log + journal rotation on the server

Already automatic (operator does nothing): the app **file logs** roll daily and
prune to `log.max_files` (14); the **journal segments** rotate at 128 MiB / 1 h.

You **must enable journal retention** for 24/7 — it defaults to *off* (unbounded
growth). Set it in `bot.local.toml`:

```toml
[journal]
retention_max_age_ms = 2592000000     # 30 days
retention_max_total_bytes = 0         # 0 = off; if set, must be >= 134217728 (128 MiB)
```

Size it empirically: let it run a day, then `du -sh /var/lib/bot/data/journal`,
× your retention days. Retention runs on each rotation + at shutdown, always
keeps the newest segment, and prunes the matching sqlite rows.

> **Do NOT point `logrotate` at the journal gzip segments or the sqlite index** —
> it corrupts the WAL database and desyncs the index (there is no `bot reindex`
> tool yet). The built-in retention is the only safe mechanism. The app file
> logs are likewise self-managed — leave them to the bot.

Cap journald (it also holds the captured stdout):

```sh
# /etc/systemd/journald.conf  ->  SystemMaxUse=2G  /  MaxRetentionSec=2week
sudo systemctl restart systemd-journald
# or one-shot: sudo journalctl --vacuum-size=2G
```

(Or set `StandardOutput=null` in the unit to rely solely on the bot's file
logs — you then lose `journalctl -u bot` history.)

### Upgrade procedure (drain before swap)

Use `deploy/upgrade.sh <new-binary>` — it stops the service (SIGTERM →
`armed=false` → `cancel_all` → drain ≤5 s → join → journal flush), verifies
`zero open orders` + `journal flushed` in the journal, atomically repoints the
symlink, restarts, and polls `/health` + `ARMED` (auto-rolling-back on failure):

```sh
sudo deploy/upgrade.sh /path/to/bot-<version>-<sha>
```

Always upgrade via `systemctl stop` (SIGTERM) — **never `bot control kill`**,
which only halts order flow and does not exit the process. **Honest note:** paper
state does *not* survive a restart (no run-loop consumes the journal rebuild
yet) — a restart begins a fresh paper wallet + new journal session. The drain's
value is **zero orphaned open orders** across the swap, not state continuity.
Manual rollback: `sudo ln -sfn /opt/bot/bin/bot-<previous> /opt/bot/bin/bot &&
sudo systemctl restart bot`.

### Backup / restore

Primary backup = the journal **gzip segments** (the source of truth) + the
config; back up the **sqlite index** too (rebuildable in principle, but no
`bot reindex` tool exists yet). Store `bot.env` secrets **separately + encrypted**,
not in the data backup. Use `deploy/backup.sh` (safe while the bot runs):

```sh
REMOTE=backup-host:/backups/bot sudo -E deploy/backup.sh
```

It snapshots sqlite via the online `.backup` API (never raw-`cp` a live WAL DB),
rsyncs the segments + sqlite snapshot + config **off-box**, and prunes local
snapshots. Restore (commands at the bottom of `backup.sh`): stop the bot,
restore `data/` (dropping stale `-wal`/`-shm`) + config, `chown`, start.

### VPS region selection

Pick the region with the lowest **order-path round-trip** to Polymarket's CLOB —
that RTT bounds the cancel-first reprice loop that defends against adverse
selection.

1. **Candidates:** spin up one VPS per shortlisted region near Polymarket infra
   (the matching engine is in AWS eu-west-2/London per the docs — a strong
   candidate), **same instance type** across candidates for a fair comparison.
2. **Measure** (read-only, **no keys**, ~2–3 min each):
   ```sh
   bot latency --label eu-west-2     # -> data/latency/latency-eu-west-2-<ts>.json
   ```
   The harness probes CLOB REST `/time`+`/ok`, the market WS, RTDS WS, and NTP,
   and reports p50/p95/**p99**.
3. **Collect** every region's JSON in one place.
4. **Decide** by the lowest CLOB REST **p99** (`rest[].stats.p99_ms` for the
   `clob/time` + `clob/ok` targets).
5. **Tie-breakers / disqualifiers:** WS `connect_ms` and RTDS `gaps.p50` (lower
   = healthier); **NTP reachability is a hard disqualifier** — a region that
   blocks UDP/123 leaves the clock-skew breaker permanently tripped and the bot
   never arms.
6. **After choosing:** feed the chosen region's measured REST p50/p95 into
   `paper.placement_latency` / `paper.cancel_latency` so paper fills reflect that
   region's real RTT.

| Region | Instance type | CLOB REST p50/p95/**p99** (ms) | WS connect (ms) | RTDS gap p50 (ms) | NTP offset / best-rtt (ms) / reachable? | Decision |
|--------|---------------|--------------------------------|-----------------|-------------------|-----------------------------------------|----------|
| eu-west-2 (London) | 2 vCPU / 4 GB | 6 / 9 / **12** | 4 | 1010 | +1 / 3 / yes | **CHOSEN** |
| us-east-1 (Virginia) | 2 vCPU / 4 GB | 78 / 95 / **110** | 33 | 1015 | +1 / 12 / yes | runner-up |
| eu-central-1 (Frankfurt) | 2 vCPU / 4 GB | 22 / 30 / **—** | 19 | 1011 | n/a / — / **NO (UDP/123 blocked)** | disqualified (NTP) |

> Placeholder numbers — replace each row from that region's `bot latency` JSON
> (`rest[].stats.{p50,p95,p99}_ms`, `ws[].connect_ms`, RTDS `gaps.p50_ms`, the
> `ntp` block). Mark the lowest-p99 **NTP-reachable** region **CHOSEN**, and
> record the date + binary version used.

## Dependency policy

- **TLS is rustls everywhere — never openssl.** When adding `reqwest`, use
  `default-features = false` with the `rustls-tls` feature. (The Polymarket SDK
  brings `aws-lc-rs` as rustls's provider — rustls, not openssl, so the policy
  holds; see the C-toolchain note above and the Decisions Log.)
- Only crates on the CLAUDE.md §3 allowlist may be added; anything else needs
  a written justification in the CLAUDE.md Decisions Log.
- In use so far: `rust_decimal`, `serde`/`serde_json`, `thiserror` (core-types,
  config), `figment` (config layering), `tracing` + `tracing-subscriber` +
  `tracing-appender` (logging), `anyhow` (binary boundary only), `reqwest` +
  `tokio` (discovery, timeutil), `tokio-tungstenite` + `futures-util`
  (timeutil's latency harness — rustls/webpki roots only), and
  `polymarket_client_sdk_v2` + `alloy-signer-local` + `chrono` (venue-live; the
  official SDK and what it transitively requires). Crates pull from the
  allowlist only when their real implementation lands.

## Workspace layout

| Crate | Responsibility |
|---|---|
| `core-types` | Domain types, event enums, ids, decimal newtypes, rounding |
| `config` | Layered config, typed per-subsystem structs, secrets |
| `timeutil` | Clock discipline, τ-remaining helpers, latency harness |
| `discovery` | Gamma/CLOB REST market discovery, metadata cache |
| `scheduler` | Per-series 24/7 window lifecycle state machine |
| `feed-util` | Shared feed plumbing: WS transport seam, generic reconnect/staleness driver, backoff |
| `feed-rtds` | Polymarket RTDS WebSocket (Chainlink + Binance topics) |
| `feed-binance` | Direct Binance WebSocket (lowest-latency signal) + feed comparator |
| `feed-clob` | CLOB market-channel WebSocket, local L2 book |
| `model` | Vol estimator, fair value, basis tracker, health states |
| `venue-api` | Execution port trait (the paper/live seam) |
| `venue-live` | Live adapter over the official Polymarket Rust SDK |
| `venue-paper` | Paper matching engine, ledger, fee/rebate simulation |
| `engine` | Quoting, inventory/pair accounting, takers, risk manager |
| `journal` | Append-only event log + replay reader |
| `analytics` | Markouts, PnL attribution, per-series aggregates |
| `dashboard` | Axum REST + WebSocket dashboard |
| `bot` | The binary: orchestrator, supervision, CLI |

Dependency arrows point inward only: nothing depends on `bot`, and `engine`
depends on `venue-api` but never on a concrete venue adapter.
