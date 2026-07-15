# VPS latency benchmark — eu-west-1 vs home (dev-pc)

**Date:** 2026-07-15 · **Box:** AWS `m7i-flex.large` (2 vCPU / 8 GB), Ubuntu 26.04, Elastic IP `54.154.134.102`, eu-west-1 (Ireland).
**Baseline running:** `vps-baseline` = `1f8e5ff` (burn-in; pre per-window-refactor). Full test suite green on the box (1177 tests, 0 failed).

## Headline
**Order-path target ≤ 50 ms: MET.** CLOB REST round-trip from the VPS is **p50 30.7 ms / p99 42.5 ms** — ~6× faster than home. The Windows clock-skew problem is retired (chrony 1.34 µs). The direct-Binance jitter that caused home's breaker flapping is eliminated (0 stale events).

## Latency — CLOB REST `/time` (order path)
| Percentile | Home (dev-pc, 2026-06-11) | **VPS eu-west-1** | Improvement |
|---|---|---|---|
| p50 | 182.1 ms | **30.7 ms** | 5.9× |
| p95 | 190.9 ms | **39.4 ms** | 4.8× |
| p99 | 238.5 ms | **42.5 ms** | 5.6× |
| max | — | 55.0 ms | — |

Method: home = `bot latency` report `data/latency/latency-dev-pc-20260611-222156.json`; VPS = 50-sample curl loop to `https://clob.polymarket.com/time` (the `bot latency` harness panics on its WS probe — see Known issues). Live spot curl agreed: home ~210 ms, VPS ~32 ms.

## Clock (the retired blocker)
- VPS chrony via **Amazon Time Sync** (`169.254.169.123`): **System time 1.34 µs off NTP**, `System clock synchronized: yes`. The §11 ClockSkew breaker never trips; the bot ARMS cleanly.
- Home repeatedly booted −0.5 to −1.9 s off NTP → ClockSkew tripped → refused to arm. **Resolved by the move.**

## Feed connectivity + resources
- All feeds connect from eu-west-1: **6 CLOB market windows** (BTC/ETH × 5m/15m/1h), **direct Binance** (`bookTicker`+`trade`), RTDS. WS connect < ~150 ms each.
- **Direct Binance: 0 stale events** — home's flapping cause (jittery link vs the 500 ms fast-feed bound) is gone.
- **CPU steal 0.0%** (no flex-instance throttling under this load). RSS ~25 MB steady.

## Breaker A/B (FeedStale / FairVsMid)
- **Home:** ~243 breaker trips/hr averaged over 523 h of journal history (config had the feed-staleness grace loosened); up to ~445/hr in bad soaks. Cause: direct-Binance link jitter.
- **VPS (strict `feed_staleness_grace_ms = 0`):** the direct-Binance cause is **eliminated**. However, the true steady-state number is **pending** because of an unrelated external outage (below). The colocated link means the VPS can run strict (no grace) — the whole hypothesis.

## ⚠️ Active external issue at time of test — Polymarket RTDS outage
Independent of the VPS: Polymarket's RTDS (`wss://ws-live-data.polymarket.com`, carrier of **Chainlink ground-truth prices**) returned a backend error on every subscription:
```
leger AddSubscriptions error: rpc error: code = Internal desc = ERROR #42P01
relation "__subscriptions" does not exist
```
`#42P01` = Postgres "undefined table" — a server-side database error. RTDS was reachable (TCP 443 OK) and the VPS subscribed correctly; Polymarket's backend errored. Consequence: Chainlink delivery became intermittent, streams starved, and **FeedStale + FairVsMid flapped (~20 cancel-alls/min)** — the §11 breakers correctly pulling all quotes when ground-truth is unavailable.

**CONFIRMED GLOBAL (not eu-west-1):** a 15 s RTDS probe from the home Windows box returned the **identical** `#42P01 __subscriptions` error. Home also still received Chainlink (btc/usd 64633.7, ~1–3 s gaps) — the outage is intermittent (some subscribes fail → starve, some succeed → gappy data). Both locations affected identically → a Polymarket incident, transient, external. The true steady-state A/B is measurable once Polymarket fixes their RTDS backend; the VPS should then run clean at strict grace=0 (direct Binance is 0-stale and Chainlink's ~1 s cadence is well under the 5 s FeedHealth threshold).

**Engine follow-up (for the refactor session, same code home runs):** during the outage the FeedStale flap looked amplified — a plausible interaction where `run.rs`'s order-path-open gate stops feeding direct-Binance ticks to the risk manager once the path closes, freezing `binance_mid_last` and re-tripping the 500 ms timer. Worth a look; not a migration blocker.

## Known issues / follow-ups
- **`bot latency` bug:** its WS probe panics on rustls `CryptoProvider` (aws-lc-rs vs ring ambiguity after the SDK landed). Harness-only — `bot run` installs `ring` via the feeds and is unaffected. Fix = `install_default` before the harness WS probe.
- **Region:** eu-west-1 (Ireland) delivers p99 42.5 ms; the repo note targeted eu-west-2 (London). eu-west-1 is comfortably under the 50 ms target, so no re-provisioning is warranted on latency grounds.

## Verdict
The migration achieves its purpose: **order-path latency ≤ 50 ms (p99 42.5)**, a sane clock, and elimination of the direct-Binance-jitter breaker flapping. The only remaining flapping is an external Polymarket RTDS outage, not a VPS deficiency.
