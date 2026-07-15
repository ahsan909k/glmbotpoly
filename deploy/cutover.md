# Cutover runbook — land the per-window refactor + eval config on the VPS

Copy-pasteable sequence to move the VPS from the burn-in baseline to the eval
build. Run as `ubuntu` on the box (`ssh -i "C:\Users\U S E R\Downloads\polybot-key.pem" ubuntu@54.154.134.102`).
Each numbered step must succeed before the next. **This does NOT start the
4-day eval clock** — that happens at EVAL START (step 10) only after every gate
and the 24 h rehearsal are green.

Branch to deploy: `per-window-refactor-20260715` (on `origin` + `glm`), which
contains Phase 0 (per-window refactor), Phase 1 (rustls harness fix), Phase 2.0
(shadow stops + clip cap + cumulative deployment budget), and the Phase 2
instrumentation + digest.

## 0. Preconditions
```bash
cd ~ubuntu/new-bot
git status --porcelain        # MUST be empty — no hand-edits on the box (hard rule 7)
git fetch --all --prune
```

## 1. Land the refactor branch
```bash
git checkout per-window-refactor-20260715     # or: git pull --ff-only if already on it
git rev-parse --short HEAD                      # record this SHA for the eval epoch
```

## 2. Build (bounded to 2 vCPU so the running bot keeps its core)
```bash
CARGO_BUILD_JOBS=2 cargo build --release -p bot        # ~6 min
```

## 3. FULL test suite ON THE BOX (the green gate — before swapping)
```bash
CARGO_BUILD_JOBS=2 cargo test --release --workspace 2>&1 \
  | tee /var/lib/bot/data/eval/cutover-tests-$(git rev-parse --short HEAD).log
CARGO_BUILD_JOBS=2 cargo test --release -p bot --features chaos 2>&1 \
  | tee -a /var/lib/bot/data/eval/cutover-tests-$(git rev-parse --short HEAD).log
# ABORT the cutover if any test fails. (This is where model_taker_paper etc.
# that are WDAC-blocked on the Windows dev box run for real — Linux is clean.)
```

## 4. Latency rebench (the rustls fix makes the WS probe work now)
```bash
cd /var/lib/bot
sudo -u bot ~ubuntu/new-bot/target/release/bot latency \
    --label vps-euw1-$(date -u +%Y%m%d) --config-dir /etc/bot/config
# Read data/latency/latency-vps-euw1-*.json : rest[].stats for clob/time -> p50, p95.
# Honest note: ~30 ms warm RTT to Polymarket from Dublin is the network floor;
# single-digit is unattainable trans-region. The target is realism, not a number.
```

## 5. Install the eval config + the measured paper latencies
```bash
sudo install -m 0644 ~ubuntu/new-bot/deploy/eval.bot.local.toml.example \
    /etc/bot/config/bot.local.toml
sudo -e /etc/bot/config/bot.local.toml
#   [paper.placement_latency] mean_ms = <clob p50>   jitter_ms = <p95 - p50>
#   [paper.cancel_latency]    mean_ms = <clob p50>   jitter_ms = <p95 - p50>
sudo -u bot /opt/bot/bin/bot check-config --config-dir /etc/bot/config   # validate the merge
```

## 6. Drain-then-swap (the EXISTING upgrade.sh — do not reimplement)
```bash
sudo ~ubuntu/new-bot/deploy/upgrade.sh ~ubuntu/new-bot/target/release/bot
# stop (SIGTERM drain) -> grep "zero open orders"+"journal flushed" ->
# atomic ln -sfn symlink swap -> start -> poll /health + "ARMED" -> auto-rollback.
```

## 7. Confirm the clock + RTDS/Chainlink are healthy (hard EVAL-START prereqs)
```bash
chronyc tracking | grep 'System time'                      # < 1 ms offset
# RTDS/Chainlink recovered (the #42P01 incident must be over): near-zero cancel-alls
sudo journalctl -u bot --since '5 min ago' | grep -c 'authoritative cancel-all'   # ~0 = recovered
```

**Feed cadence + FeedStale grace (2026-07-16).** The risk fast-feed staleness bound
is 500 ms; on this eu-west-1 box the direct-Binance Mid path stalls p95 ~1.4 s
(bursty trans-region delivery, WS stable), so `risk.feed_staleness_grace_ms = 1500`
(env-correct — see the Decisions Log). Watch the new **feed-cadence tile**
(`GET /api/feed-cadence`: Mid-gap histogram + loop-lag) and the `mid_gap_p95_ms`
resource-report field, NOT breaker flaps, to judge feed health.
**Fallback (only if the rehearsal still shows residual Binance-origin FeedStale at
grace=1500):** try the drop-in endpoint `[feeds] binance_ws_url =
"wss://data-stream.binance.vision"` — a different Binance edge that may have a
better path from Dublin. Re-benchmark the Mid-gap p95 after switching; keep
whichever endpoint's cadence is tighter. ChainlinkRtds `#42P01` RTDS stalls are a
Polymarket vendor incident (annotate, don't count toward the <5/hr gate) and must
be recovered before EVAL START regardless of grace.

## 8. GATE 0 — six-series maker proof (LIVE, after ≥2 window cycles / ~10 min)
```bash
sqlite3 -readonly /var/lib/bot/data/journal.sqlite <<'SQL'
.mode column
.headers on
-- all six series must carry concurrently-resting maker orders, BTC ~ ETH counts
SELECT series, COUNT(DISTINCT order_id) AS resting
FROM orders WHERE state='open'
  AND ts_local_ms > (SELECT MAX(ts_local_ms) - 600000 FROM orders)
GROUP BY series ORDER BY series;
SQL
```
PASS iff **all six** of BTC/ETH × 5m/15m/1h show `resting > 0` with BTC and ETH
placement counts comparable (the pre-refactor bug left 5 of 6 at ~0). Cross-check
maker fills are landing:
```bash
sqlite3 -readonly /var/lib/bot/data/journal.sqlite \
  "SELECT series, COUNT(*) FROM fills WHERE liquidity='maker' GROUP BY series;"
```

## 9. Breaker A/B (VPS strict grace=0 vs home ~73/hr)
```bash
sqlite3 -readonly /var/lib/bot/data/journal.sqlite \
  "SELECT breaker, COUNT(*) FROM breaker_trips WHERE kind='tripped'
     AND ts_local_ms > strftime('%s','now','-1 hour')*1000 GROUP BY breaker;"
```
Steady-state feed_stale/ws_disconnect/fair_vs_mid trips should be < 5/hr on the
colocated VPS (annotate any Polymarket-outage window separately).

## 10. — only after the 24 h rehearsal passes (deploy/rehearsal_check.sh) —
Mark EVAL START and reset paper capital (see the plan Phase 3.1):
```bash
sudo -u bot mkdir -p /var/lib/bot/data/eval
EPOCH_MS=$(($(date -u +%s)*1000))
printf '%s %s %s phase1\n' "$EPOCH_MS" "$(git rev-parse --short HEAD)" "$(date -u +%FT%TZ)" \
  | sudo -u bot tee /var/lib/bot/data/eval/epoch
# capital is $50k already in the eval config; the journal epoch is the session tag.
```

## Rollback
`upgrade.sh` auto-rolls-back on a failed `/health`+ARMED poll. Manual, after the
poll window:
```bash
sudo ln -sfn /opt/bot/bin/bot-<previous-sha> /opt/bot/bin/bot && sudo systemctl restart bot
```
A rollback during the eval is a hard-rule-7 event — journal it in the eval log.
