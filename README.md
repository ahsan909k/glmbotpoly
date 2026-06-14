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
into trading config.

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

### VPS notes

On the deployment VPS you'll likely want full parallelism. Override the
committed jobs cap per shell:

```sh
CARGO_BUILD_JOBS=$(nproc) cargo build --release -p bot
```

or per invocation: `cargo build --jobs "$(nproc)" --release -p bot`.

**C toolchain for the live SDK.** `venue-live` pulls the Polymarket SDK, whose
rustls stack uses `aws-lc-rs` (still rustls — *not* openssl); its `aws-lc-sys`
build compiles C/assembly, so the VPS needs a C toolchain to build the bot:

```sh
sudo apt install clang cmake        # Debian/Ubuntu (in addition to mold)
```

(The dev PC builds it fine with the bundled MSVC/clang. This only matters for a
clean-room VPS build.)

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
