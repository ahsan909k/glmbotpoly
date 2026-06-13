//! Sans-IO feed machine: per-stream staleness watchdog and tick
//! normalization, generic over the feed's stream-key type. No clock, no IO,
//! no logging — inputs carry `now`, outputs are values the driver routes
//! (timeutil `SkewMonitor` precedent).
//!
//! Staleness semantics:
//! - Ages are measured on **local receive time** (pipeline liveness); data
//!   recency (`ts_exchange`) is a separate dimension consumers judge per
//!   tick — an old backfill tick counts as liveness here.
//! - Each stream carries its own `stale_after` threshold: a Binance trade
//!   stream legitimately gaps for many seconds in a quiet market while its
//!   top-of-book stream is near-continuous — one global threshold would
//!   flap the quiet-but-healthy stream.
//! - A stream is declared [`FeedHealth::Stale`] when no tick has arrived for
//!   its `stale_after` (measured from its last tick, or from its anchor —
//!   the subscribe/connect time — if it never delivered).
//! - Disconnect is definitive evidence: every non-stale stream is marked
//!   stale immediately, not after the threshold.
//! - Staleness latches: one `Stale` per outage, one [`FeedHealth::Recovered`]
//!   (with the outage length) on the next tick. After a reconnect, streams
//!   recover individually as their data resumes.

use std::collections::BTreeMap;

use core_types::{
    Asset, Decimal, DurationMs, FeedHealth, PriceSource, PriceTick, TickKind, TimestampMs,
};

/// Identity of one tracked stream. The machine derives every bus-facing
/// identity ([`PriceTick`]/[`FeedHealth`] fields) from it, so feed crates
/// only ever hand the driver their own key type.
pub trait StreamKey: Copy + Ord + std::fmt::Display + Send + Sync + 'static {
    /// The bus-level source this stream publishes as.
    fn source(&self) -> PriceSource;
    /// The underlying asset.
    fn asset(&self) -> Asset;
    /// The observation flavor this stream carries.
    fn kind(&self) -> TickKind;
}

/// What the machine wants the driver to do.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Output {
    /// Publish a normalized tick on the bus.
    Publish(PriceTick),
    /// Publish a health transition on the bus.
    Health(FeedHealth),
}

/// Per-stream watchdog state.
#[derive(Debug, Clone)]
struct KeyState {
    /// This stream's staleness threshold.
    stale_after: DurationMs,
    /// Local receive time of the last tick, if any ever arrived.
    last_seen: Option<TimestampMs>,
    /// Fallback age origin: subscribe time, refreshed on every (re)connect.
    anchor: TimestampMs,
    /// Staleness latch (one Stale per outage).
    stale: bool,
    /// Lifetime tick count (status/debug only).
    ticks: u64,
    /// Last published value (status only).
    last_value: Option<Decimal>,
    /// `ts_exchange` of the last tick (status only).
    last_ts_exchange: Option<TimestampMs>,
}

impl KeyState {
    fn new(stale_after: DurationMs, now: TimestampMs) -> Self {
        Self {
            stale_after,
            last_seen: None,
            anchor: now,
            stale: false,
            ticks: 0,
            last_value: None,
            last_ts_exchange: None,
        }
    }

    /// Where this stream's age is measured from.
    fn age_origin(&self) -> TimestampMs {
        self.last_seen.unwrap_or(self.anchor)
    }
}

/// Point-in-time view of one stream, for the status snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyStatus<K> {
    /// The stream.
    pub sub: K,
    /// Last published value.
    pub last_value: Option<Decimal>,
    /// Source timestamp of the last tick.
    pub ts_exchange: Option<TimestampMs>,
    /// Local receive time of the last tick (consumers compute ages against
    /// their own clock so displays never freeze between snapshots).
    pub ts_local: Option<TimestampMs>,
    /// Whether the watchdog currently considers the stream stale.
    pub stale: bool,
    /// Lifetime tick count.
    pub ticks: u64,
}

/// One stream's starvation breach, for the driver's recycle decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StarvationBreach {
    /// How long the stream has been starved within the current episode.
    pub(crate) age: DurationMs,
    /// The threshold it breached (`multiple × stale_after`).
    pub(crate) threshold: DurationMs,
}

/// The sans-IO machine. One instance per feed connection (i.e. per driver).
#[derive(Debug, Clone)]
pub(crate) struct FeedMachine<K> {
    keys: BTreeMap<K, KeyState>,
    connected: bool,
}

impl<K: StreamKey> FeedMachine<K> {
    pub(crate) fn new(keys: impl IntoIterator<Item = (K, DurationMs)>, now: TimestampMs) -> Self {
        Self {
            keys: keys
                .into_iter()
                .map(|(k, stale_after)| (k, KeyState::new(stale_after, now)))
                .collect(),
            connected: false,
        }
    }

    /// Starts tracking a stream (runtime subscribe). Idempotent: an already
    /// tracked stream keeps its state (including its original threshold).
    pub(crate) fn subscribe(&mut self, key: K, stale_after: DurationMs, now: TimestampMs) {
        self.keys
            .entry(key)
            .or_insert_with(|| KeyState::new(stale_after, now));
    }

    /// Stops tracking a stream (runtime unsubscribe): no further outputs for
    /// it, silently — an operator-removed stream is not an outage.
    pub(crate) fn unsubscribe(&mut self, key: K) {
        self.keys.remove(&key);
    }

    /// A connection (re)opened: refresh every anchor so never-delivered
    /// streams get a fresh grace period. Stale latches stay set — streams
    /// recover individually on their first tick.
    pub(crate) fn on_connected(&mut self, now: TimestampMs) {
        self.connected = true;
        for state in self.keys.values_mut() {
            state.anchor = now;
        }
    }

    /// The connection dropped: definitive evidence — every non-stale stream
    /// goes stale immediately (no threshold wait).
    pub(crate) fn on_disconnected(&mut self, now: TimestampMs, out: &mut Vec<Output>) {
        self.connected = false;
        for (key, state) in &mut self.keys {
            if !state.stale {
                state.stale = true;
                out.push(Output::Health(FeedHealth::Stale {
                    source: key.source(),
                    asset: key.asset(),
                    kind: key.kind(),
                    age: now.signed_duration_since(state.age_origin()),
                }));
            }
        }
    }

    /// A parsed price observation arrived. Publishes a tick for tracked
    /// streams (an unsubscribed stream's stragglers are dropped) and clears
    /// the staleness latch.
    pub(crate) fn on_price(
        &mut self,
        key: K,
        value: Decimal,
        ts_exchange: TimestampMs,
        ts_local: TimestampMs,
        out: &mut Vec<Output>,
    ) {
        let Some(state) = self.keys.get_mut(&key) else {
            return;
        };
        if state.stale {
            state.stale = false;
            out.push(Output::Health(FeedHealth::Recovered {
                source: key.source(),
                asset: key.asset(),
                kind: key.kind(),
                gap: ts_local.signed_duration_since(state.age_origin()),
            }));
        }
        state.last_seen = Some(ts_local);
        state.ticks = state.ticks.saturating_add(1);
        state.last_value = Some(value);
        state.last_ts_exchange = Some(ts_exchange);
        out.push(Output::Publish(PriceTick {
            source: key.source(),
            asset: key.asset(),
            kind: key.kind(),
            value,
            ts_exchange,
            ts_local,
        }));
    }

    /// Periodic watchdog scan: declare staleness where a stream's own
    /// threshold passed. While disconnected this is a no-op — disconnect
    /// already marked everything stale.
    pub(crate) fn on_tick(&mut self, now: TimestampMs, out: &mut Vec<Output>) {
        if !self.connected {
            return;
        }
        for (key, state) in &mut self.keys {
            if state.stale {
                continue;
            }
            let age = now.signed_duration_since(state.age_origin());
            if age >= state.stale_after {
                state.stale = true;
                out.push(Output::Health(FeedHealth::Stale {
                    source: key.source(),
                    asset: key.asset(),
                    kind: key.kind(),
                    age,
                }));
            }
        }
    }

    /// The worst starvation breach **within the current episode**, if any
    /// stream is starved beyond `multiple × its own stale_after`: time since
    /// its last tick or since the episode's anchor, whichever is more recent.
    /// The driver recycles the connection on a breach — a stream starved far
    /// beyond its threshold on an otherwise-healthy connection means the
    /// server-side subscription decayed (observed live 2026-06-12: chainlink
    /// btc/usd went silent for 3+ minutes while eth/usd streamed on), and a
    /// reconnect rebuilds it. Measuring from the per-episode anchor (not the
    /// historical `last_seen`) is what spaces recycles a full threshold
    /// apart instead of looping every watchdog tick.
    pub(crate) fn worst_breach(&self, now: TimestampMs, multiple: i64) -> Option<StarvationBreach> {
        self.keys
            .values()
            .filter_map(|state| {
                let origin = state
                    .last_seen
                    .map_or(state.anchor, |seen| seen.max(state.anchor));
                let age = now.signed_duration_since(origin);
                let threshold =
                    DurationMs::from_millis(state.stale_after.as_millis().saturating_mul(multiple));
                (age >= threshold).then_some(StarvationBreach { age, threshold })
            })
            .max_by_key(|b| b.age.as_millis() - b.threshold.as_millis())
    }

    /// Point-in-time snapshot of every tracked stream.
    pub(crate) fn status(&self) -> Vec<KeyStatus<K>> {
        self.keys
            .iter()
            .map(|(key, state)| KeyStatus {
                sub: *key,
                last_value: state.last_value,
                ts_exchange: state.last_ts_exchange,
                ts_local: state.last_seen,
                stale: state.stale,
                ticks: state.ticks,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const STALE_AFTER: DurationMs = DurationMs::from_millis(5_000);

    /// A minimal test key mirroring the RTDS streams (two sources × BTC/ETH,
    /// all `Vendor`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct TestKey {
        source: PriceSource,
        asset: Asset,
    }

    impl std::fmt::Display for TestKey {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}:{}", self.source, self.asset)
        }
    }

    impl StreamKey for TestKey {
        fn source(&self) -> PriceSource {
            self.source
        }
        fn asset(&self) -> Asset {
            self.asset
        }
        fn kind(&self) -> TickKind {
            TickKind::Vendor
        }
    }

    fn key(source: PriceSource, asset: Asset) -> TestKey {
        TestKey { source, asset }
    }

    fn all_keys() -> Vec<(TestKey, DurationMs)> {
        let mut keys = Vec::new();
        for source in [PriceSource::BinanceRtds, PriceSource::ChainlinkRtds] {
            for asset in Asset::ALL {
                keys.push((key(source, asset), STALE_AFTER));
            }
        }
        keys
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn machine_all(now: TimestampMs) -> FeedMachine<TestKey> {
        let mut m = FeedMachine::new(all_keys(), now);
        m.on_connected(now);
        m
    }

    #[test]
    fn tick_publishes_with_both_timestamps_and_key_identity() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Btc),
            dec!(100.5),
            ts(990),
            ts(1_000),
            &mut out,
        );
        assert_eq!(
            out,
            vec![Output::Publish(PriceTick {
                source: PriceSource::BinanceRtds,
                asset: Asset::Btc,
                kind: TickKind::Vendor,
                value: dec!(100.5),
                ts_exchange: ts(990),
                ts_local: ts(1_000),
            })]
        );
    }

    #[test]
    fn never_delivering_stream_goes_stale_from_anchor() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        m.on_tick(ts(4_999), &mut out);
        assert!(out.is_empty(), "below threshold");
        m.on_tick(ts(5_000), &mut out);
        assert_eq!(out.len(), 4, "all four never-delivered streams stale");
        out.clear();
        m.on_tick(ts(6_000), &mut out);
        assert!(out.is_empty(), "stale latches — no re-emission");
    }

    #[test]
    fn per_key_staleness_and_recovery_gap() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        // All four deliver at t=1s.
        for (k, _) in all_keys() {
            m.on_price(k, dec!(100.5), ts(1_000), ts(1_000), &mut out);
        }
        out.clear();
        // Only chainlink:btc keeps delivering.
        m.on_price(
            key(PriceSource::ChainlinkRtds, Asset::Btc),
            dec!(100.5),
            ts(5_500),
            ts(5_500),
            &mut out,
        );
        out.clear();
        m.on_tick(ts(6_000), &mut out);
        let stale: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                Output::Health(FeedHealth::Stale {
                    source, asset, age, ..
                }) => Some((*source, *asset, *age)),
                _ => None,
            })
            .collect();
        assert_eq!(stale.len(), 3, "chainlink:btc stays live: {out:?}");
        assert!(
            stale
                .iter()
                .all(|(_, _, age)| *age == DurationMs::from_millis(5_000))
        );
        assert!(
            !stale
                .iter()
                .any(|(s, a, _)| *s == PriceSource::ChainlinkRtds && *a == Asset::Btc)
        );

        // binance:eth recovers at t=9s → gap from its last tick at t=1s.
        out.clear();
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Eth),
            dec!(100.5),
            ts(9_000),
            ts(9_000),
            &mut out,
        );
        assert_eq!(
            out[0],
            Output::Health(FeedHealth::Recovered {
                source: PriceSource::BinanceRtds,
                asset: Asset::Eth,
                kind: TickKind::Vendor,
                gap: DurationMs::from_millis(8_000),
            })
        );
        assert!(matches!(out[1], Output::Publish(_)));
    }

    #[test]
    fn per_key_thresholds_are_independent() {
        // A slow-tolerant stream (trade-print style, 30 s) next to a fast one
        // (top-of-book style, 2.5 s): only the fast one stales at 2.5 s.
        let fast = key(PriceSource::BinanceDirect, Asset::Btc);
        let slow = key(PriceSource::BinanceDirect, Asset::Eth);
        let mut m = FeedMachine::new(
            [
                (fast, DurationMs::from_millis(2_500)),
                (slow, DurationMs::from_millis(30_000)),
            ],
            ts(0),
        );
        m.on_connected(ts(0));
        let mut out = Vec::new();
        m.on_tick(ts(2_500), &mut out);
        assert_eq!(out.len(), 1, "only the fast stream staled: {out:?}");
        assert!(matches!(
            out[0],
            Output::Health(FeedHealth::Stale {
                asset: Asset::Btc,
                ..
            })
        ));
        out.clear();
        m.on_tick(ts(29_999), &mut out);
        assert!(out.is_empty(), "slow stream still within threshold");
        m.on_tick(ts(30_000), &mut out);
        assert_eq!(out.len(), 1, "slow stream stales at its own threshold");
    }

    #[test]
    fn disconnect_marks_all_non_stale_immediately_exactly_once() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Btc),
            dec!(100.5),
            ts(900),
            ts(1_000),
            &mut out,
        );
        out.clear();

        m.on_disconnected(ts(1_500), &mut out);
        assert_eq!(out.len(), 4, "all streams stale on disconnect");
        let btc_age = out.iter().find_map(|o| match o {
            Output::Health(FeedHealth::Stale {
                source: PriceSource::BinanceRtds,
                asset: Asset::Btc,
                age,
                ..
            }) => Some(*age),
            _ => None,
        });
        assert_eq!(
            btc_age,
            Some(DurationMs::from_millis(500)),
            "age from last tick, not anchor"
        );

        // Watchdog ticks while disconnected emit nothing more.
        out.clear();
        m.on_tick(ts(60_000), &mut out);
        assert!(out.is_empty());
        // A second disconnect-ish call also emits nothing (already stale).
        m.on_disconnected(ts(61_000), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn reconnect_refreshes_anchor_and_streams_recover_individually() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        m.on_disconnected(ts(1_000), &mut out);
        out.clear();

        m.on_connected(ts(10_000));
        // Immediately after reconnect: no spurious events; grace period runs
        // from the new anchor.
        m.on_tick(ts(10_500), &mut out);
        assert!(out.is_empty());

        // One stream recovers; the rest stay stale (latched).
        m.on_price(
            key(PriceSource::ChainlinkRtds, Asset::Eth),
            dec!(100.5),
            ts(11_000),
            ts(11_000),
            &mut out,
        );
        assert!(matches!(
            out[0],
            Output::Health(FeedHealth::Recovered {
                source: PriceSource::ChainlinkRtds,
                asset: Asset::Eth,
                ..
            })
        ));
        out.clear();

        // The others were never marked again; they stay silently stale until
        // data arrives — only the snapshot shows them. (Within the recovered
        // stream's grace window so it doesn't legitimately re-stale.)
        m.on_tick(ts(15_000), &mut out);
        assert!(out.is_empty());
        let stale_count = m.status().iter().filter(|k| k.stale).count();
        assert_eq!(stale_count, 3);

        // And a recovered stream that dries up again re-stales — staleness
        // is per outage, not once per lifetime.
        m.on_tick(ts(16_000), &mut out);
        assert_eq!(out.len(), 1, "chainlink:eth re-stales 5s after its tick");
    }

    #[test]
    fn unsubscribed_stream_stops_tracking_and_drops_stragglers() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        m.unsubscribe(key(PriceSource::BinanceRtds, Asset::Eth));
        // A straggler tick for the removed stream publishes nothing.
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Eth),
            dec!(100.5),
            ts(100),
            ts(100),
            &mut out,
        );
        assert!(out.is_empty());
        // And it never goes stale.
        m.on_tick(ts(60_000), &mut out);
        assert_eq!(out.len(), 3, "only the three tracked streams: {out:?}");
        assert_eq!(m.status().len(), 3);
    }

    #[test]
    fn runtime_subscribe_anchors_grace_at_now() {
        let mut m: FeedMachine<TestKey> = FeedMachine::new([], ts(0));
        m.on_connected(ts(0));
        let k = key(PriceSource::ChainlinkRtds, Asset::Btc);
        m.subscribe(k, STALE_AFTER, ts(100_000));
        let mut out = Vec::new();
        m.on_tick(ts(104_999), &mut out);
        assert!(out.is_empty());
        m.on_tick(ts(105_000), &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn old_backfill_ts_exchange_does_not_affect_health() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        // Backfill: ts_exchange two minutes old, received now — liveness is
        // measured on ts_local, so the stream is live.
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Btc),
            dec!(100.5),
            ts(-120_000),
            ts(1_000),
            &mut out,
        );
        let Output::Publish(tick) = &out[0] else {
            panic!("expected publish");
        };
        assert_eq!(tick.ts_exchange, ts(-120_000), "honest old timestamp");
        out.clear();
        m.on_tick(ts(2_000), &mut out);
        assert!(
            out.is_empty(),
            "not stale — liveness is local-receive based"
        );
    }

    #[test]
    fn worst_breach_is_per_episode_not_historical() {
        let m: FeedMachine<TestKey> = FeedMachine::new([], ts(0));
        assert_eq!(m.worst_breach(ts(99_000), 1), None, "nothing tracked");

        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        for (k, _) in all_keys() {
            m.on_price(k, dec!(100.5), ts(1_000), ts(1_000), &mut out);
        }
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Btc),
            dec!(100.5),
            ts(9_000),
            ts(9_000),
            &mut out,
        );
        // Three streams last seen at t=1s, one at t=9s — at multiple 1 the
        // worst offender is 9s past a 5s threshold.
        assert_eq!(
            m.worst_breach(ts(10_000), 1),
            Some(StarvationBreach {
                age: DurationMs::from_millis(9_000),
                threshold: STALE_AFTER,
            })
        );
        // Below the multiplied threshold: no breach.
        assert_eq!(m.worst_breach(ts(10_000), 6), None);
        // A reconnect refreshes anchors: starvation restarts from the new
        // episode even though last_seen is old — this is what spaces
        // recycle-reconnects a full threshold apart instead of looping.
        m.on_connected(ts(20_000));
        assert_eq!(m.worst_breach(ts(24_999), 1), None);
        assert_eq!(
            m.worst_breach(ts(26_000), 1),
            Some(StarvationBreach {
                age: DurationMs::from_millis(6_000),
                threshold: STALE_AFTER,
            })
        );
    }

    #[test]
    fn worst_breach_respects_per_key_thresholds() {
        // The slow stream is more starved in absolute terms but within its
        // threshold; only the fast stream breaches.
        let fast = key(PriceSource::BinanceDirect, Asset::Btc);
        let slow = key(PriceSource::BinanceDirect, Asset::Eth);
        let mut m = FeedMachine::new(
            [
                (fast, DurationMs::from_millis(2_500)),
                (slow, DurationMs::from_millis(30_000)),
            ],
            ts(0),
        );
        m.on_connected(ts(0));
        let mut out = Vec::new();
        // Fast stream ticks once at t=1s, then dries; slow never delivers.
        m.on_price(fast, dec!(1), ts(1_000), ts(1_000), &mut out);
        // At t=20s: fast starved 19s ≥ 6×2.5s=15s → breach; slow starved 20s
        // < 6×30s=180s → fine.
        assert_eq!(
            m.worst_breach(ts(20_000), 6),
            Some(StarvationBreach {
                age: DurationMs::from_millis(19_000),
                threshold: DurationMs::from_millis(15_000),
            })
        );
    }

    #[test]
    fn status_snapshot_reflects_state() {
        let mut m = machine_all(ts(0));
        let mut out = Vec::new();
        m.on_price(
            key(PriceSource::BinanceRtds, Asset::Btc),
            dec!(100.5),
            ts(900),
            ts(1_000),
            &mut out,
        );
        let status = m.status();
        assert_eq!(status.len(), 4);
        let btc = status
            .iter()
            .find(|k| k.sub == key(PriceSource::BinanceRtds, Asset::Btc))
            .expect("tracked");
        assert_eq!(btc.last_value, Some(dec!(100.5)));
        assert_eq!(btc.ts_local, Some(ts(1_000)));
        assert_eq!(btc.ts_exchange, Some(ts(900)));
        assert_eq!(btc.ticks, 1);
        assert!(!btc.stale);
    }
}
