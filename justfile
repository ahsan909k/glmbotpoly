# Developer workflow (CLAUDE.md §3): iterate with check/clippy via bacon,
# full builds only for deployment.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Default: list recipes
default:
    just --list

# Fast type-check of the whole workspace (the day-to-day loop)
check:
    cargo check --workspace --all-targets

# Format check + clippy with warnings denied (CI gate, CLAUDE.md §12)
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# Apply formatting
fmt:
    cargo fmt --all

# Run all tests in the workspace
test:
    cargo test --workspace

# Deterministic, paper-only §11 chaos suite (feature-gated; off by default so it
# never burdens the day-to-day loop). Single-threaded for paused-clock + journal
# temp-dir determinism. Injects each scripted fault — kill each WebSocket, stall
# each feed, a matching-engine restart, a clock jump, a discovery failure at
# rollover, a process restart mid-window — and asserts no orphan orders, the
# right breaker trips with a journaled cause, halt/resume per the rules, and an
# exact state rebuild from the journal. Acceptance: run it ~10x, 100% green.
# Lint the gated code explicitly (the default `just lint` compiles it empty):
#   cargo clippy -p bot --features chaos --all-targets -- -D warnings
chaos:
    cargo test -p bot --features chaos --test chaos -- --test-threads=1

# Continuous check/clippy watcher (install: cargo install bacon)
watch:
    bacon

# Run the bot in paper mode (boots, logs config, exits — engine not wired yet)
run-paper:
    cargo run -p bot -- paper

# Load + validate the configuration and print the effective (redacted) result
check-config:
    cargo run -p bot -- check-config

# Live read-only probe: discover current + next windows for all six series
# (hits the public Gamma/CLOB APIs; network required)
discover:
    cargo run -p bot -- discover

# Live scheduler smoke run: roll all enabled series continuously, logging
# every lifecycle event + a 30 s coverage table (read-only; ctrl-c to stop;
# 30+ minutes of clean coverage = the §6 acceptance check)
schedule:
    cargo run -p bot -- schedule

# Live RTDS feed smoke run: stream Binance + Chainlink crypto prices side by
# side with per-stream staleness ages (read-only; ctrl-c to stop). Kill the
# network mid-run to watch stale → reconnect → recovered. Add a raw-frame
# capture with: cargo run -p bot -- feed --raw data/captures/rtds.jsonl
feed:
    cargo run -p bot -- feed

# Live direct-Binance feed smoke run: BTCUSDT/ETHUSDT bookTicker midpoints +
# trade prints (read-only; ctrl-c to stop). Raw-frame capture:
# cargo run -p bot -- feed --source binance --raw data/captures/binance.jsonl
feed-binance:
    cargo run -p bot -- feed --source binance

# Live feed comparison: both feeds on one bus, per-asset lead/lag + value
# basis summarized every minute (read-only; ctrl-c to stop). This is the
# measurement that tells us which feed leads and calibrates the model's
# basis correction — let it run 10+ minutes for stable numbers.
compare:
    cargo run -p bot -- compare

# Live vol-estimator smoke run: both feeds on one bus feeding four per-asset
# sigma_1s lanes (read-only; ctrl-c to stop). Watch for: chainlink lanes
# READY within seconds (RTDS backfill pre-seeds ~60-120 bars), binance lanes
# READY after ~60 s of live data, sigma_1s plausibly ~3e-5..3e-4 for BTC with
# clamp "-" in normal markets, reanchors 0 on healthy feeds.
vol:
    cargo run -p bot -- vol

# Live L2 ladder for one series' current window (read-only; ctrl-c to stop;
# default BTC-5m). The scheduler + feed-clob run wired end-to-end, so window
# rollovers and market_resolved settlements happen live. Verify the ladder
# against polymarket.com/event/<slug> (printed in the header). Forced
# reconnect demo: cargo run -p bot -- ladder --recycle-after 60
# Raw-frame capture: cargo run -p bot -- ladder --raw data/captures/clob.jsonl
ladder series="BTC-5m":
    cargo run -p bot -- ladder --series {{series}}

# Live fair-value smoke run for one series' current window (read-only;
# ctrl-c to stop; default BTC-5m). Renders the full model — strike capture,
# sigma_1s, basis, p_up = Phi(z) — next to the Polymarket Up-token book mid.
# Watch for: strike FROZEN within seconds (RTDS backfill recovers the boundary
# print), model healthy after sigma READY (~60 s), p_up tracking the book mid
# inside the |p_up - mid| sanity band mid-window, and VERIFY records landing
# 1-4 min after each window resolves (exact = our captured strike matches the
# venue's priceToBeat). Calibration JSONL lands in data/calibration/.
fair series="BTC-5m":
    cargo run -p bot -- fair --series {{series}}

# Live paper-trading smoke run for one series' current window (real data, paper
# money; ctrl-c to stop; default BTC-5m). Wires the scheduler + feed-clob into
# the paper fill simulator with a trivial two-sided quoter that re-quotes at the
# live touch. Watch for: quotes resting, real prints filling them as Maker
# (fee 0) at our price, the cancel re-quote cycle, and the paper wallet updating
# — never filling more or sooner than the live book/prints shown alongside.
paper-sim series="BTC-5m":
    cargo run -p bot -- paper-sim --series {{series}}

# Serve the dashboard (CLAUDE.md §10) over a live paper pipeline (real data,
# paper money; ctrl-c to stop). Wires the scheduler + feed-clob + the paper venue
# across all enabled series into the axum REST + WebSocket dashboard, bound to
# config.dashboard.bind (default 127.0.0.1:8080). The auth token comes from
# BOT_SECRET_DASHBOARD_TOKEN (required for a non-loopback bind; optional on
# loopback). Scope it to one series with series="BTC-5m". Probe it with:
#   curl -s localhost:8080/health
#   curl -s -H "Authorization: Bearer $BOT_SECRET_DASHBOARD_TOKEN" localhost:8080/api/overview
#   curl -s -H "Authorization: Bearer $BOT_SECRET_DASHBOARD_TOKEN" "localhost:8080/api/series-comparison?mode=paper&window=all"
# WebSocket (token via query): ws://localhost:8080/api/ws?token=$BOT_SECRET_DASHBOARD_TOKEN
dashboard:
    cargo run -p bot -- dashboard

# THE MAIN RUN MODE (§5; real data, paper money; ctrl-c or SIGTERM to stop).
# Starts paper trading on all enabled series at once under a supervision tree:
# feeds + scheduler + the multi-asset model + the risk-managed engine (the real
# quote manager + momentum/late takers behind the §11 breakers) + the paper venue
# + journal + analytics + the dashboard, on config.dashboard.bind. Any crashed
# task restarts with backoff while the rest keep running (a dead feed cancels-all
# and gates orders until it recovers); a resilient startup self-check refuses to
# trade until clocks are sane + discovery has windows + feeds are healthy, then
# auto-arms ("ARMED" in the log). Scope to one series with series="BTC-5m".
# Acceptance: a multi-hour session — six series rolling, dashboard live, the RSS
# report (`resource report` log line) flat, clean shutdown leaving zero open
# paper orders. On Linux, `kill -TERM <pid>` shuts down as cleanly as ctrl-c.
run series="":
    cargo run -p bot -- run {{ if series == "" { "" } else { "--series " + series } }}

# Live journal capture (read-only; ctrl-c to stop). Wires the scheduler + all
# three feeds onto the bus and records every event to gzip-compressed, rotated
# JSONL segments under out-dir (default data/journal/) — built for days of
# capture. Records all enabled series by default; pass series="BTC-5m" for one.
# The captured segments replay through journal::ReplayReader (and feed the
# venue-parity tests). A nonzero "dropped" in the summary = the disk fell behind.
record out-dir="data/journal":
    cargo run -p bot -- record --out-dir {{out-dir}}

# Latency benchmark against live Polymarket endpoints (read-only; network
# required; ~2-3 min). Run from each candidate VPS region with a label;
# the JSON report lands in data/latency/. Note: timeutil's harness-gated
# tests need `cargo test -p timeutil --features harness` when testing
# per-package (the workspace test run covers them via bot's feature request).
latency label="dev-pc":
    cargo run -p bot -- latency --label {{label}}

# Live execution-port demonstration (offline; no credentials, no network):
# prints the live params and the constructed (unsent, unsigned) request for
# each order class — GTC/GTD post-only, FAK/FOK marketable — through the same
# translation the live adapter uses. `bot live` (separately) routes through the
# gated LiveVenue::connect and refuses (NotArmed) until all three §11 gates pass.
venue-check:
    cargo run -p bot -- venue-check

# OPERATOR-ONLY live smoke test — touches the live venue and SPENDS REAL FUNDS.
# Never runs by default (cargo feature + #[ignore] + BOT_LIVE_SMOKE env guard).
# See the env vars documented in crates/venue-live/tests/live_smoke.rs, then:
#   cargo test -p venue-live --features live-smoke -- --ignored live_smoke
# (Not a just recipe on purpose — run it deliberately, by hand.)

# Optimized build for deployment (slow; not for iteration)
build-release:
    cargo build --release -p bot
