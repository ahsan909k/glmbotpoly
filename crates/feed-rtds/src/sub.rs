//! The desired-subscription set and its diffing to wire messages.
//!
//! RTDS holds **one filter slot per (connection, topic)** — a later filtered
//! subscribe replaces the earlier one, and no multi-symbol filter form works
//! (live-verified 2026-06-12, `tests/live_probe.rs`). So the wire state is
//! per *topic*: each active topic streams all symbols (untracked ones are
//! dropped client-side by the machine), while tracking stays per
//! (source, asset). Filtered subscribes are still sent first on every
//! (re)connect and on runtime adds — each one triggers the ~2-minute
//! backfill that seeds the vol estimator; the unfiltered subscribe then
//! restores the all-symbols steady state.

use std::collections::BTreeSet;

use core_types::Asset;

use crate::wire::{
    RtdsSource, backfill_subscribe_message, stream_subscribe_message, stream_unsubscribe_message,
};

/// One (source, asset) price stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeedSub {
    /// Which RTDS topic.
    pub source: RtdsSource,
    /// Which underlying asset.
    pub asset: Asset,
}

impl FeedSub {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(source: RtdsSource, asset: Asset) -> Self {
        Self { source, asset }
    }

    /// All four streams the bot trades on (both topics × BTC/ETH).
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut subs = Vec::with_capacity(4);
        for source in [RtdsSource::Binance, RtdsSource::Chainlink] {
            for asset in Asset::ALL {
                subs.push(Self::new(source, asset));
            }
        }
        subs
    }
}

impl std::fmt::Display for FeedSub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.source, self.source.symbol(self.asset))
    }
}

impl feed_util::StreamKey for FeedSub {
    fn source(&self) -> core_types::PriceSource {
        self.source.price_source()
    }

    fn asset(&self) -> Asset {
        self.asset
    }

    /// RTDS payloads carry a bare `value` with no provenance (trade? mid?
    /// index?) — the flavor is the vendor's to know.
    fn kind(&self) -> core_types::TickKind {
        core_types::TickKind::Vendor
    }
}

/// Runtime subscription change, delivered to the driver over its command
/// channel; applied by diffing against the desired set — no reconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedCommand {
    /// Start streaming this (source, asset).
    Subscribe(FeedSub),
    /// Stop streaming this (source, asset).
    Unsubscribe(FeedSub),
}

/// The desired set of streams (stable iteration order for deterministic
/// resubscribes and tests).
#[derive(Debug, Clone, Default)]
pub(crate) struct SubSet {
    desired: BTreeSet<FeedSub>,
}

impl SubSet {
    pub(crate) fn new(initial: impl IntoIterator<Item = FeedSub>) -> Self {
        Self {
            desired: initial.into_iter().collect(),
        }
    }

    /// Adds to the desired set; the messages to send if newly added (a
    /// backfill-triggering filtered subscribe for the new stream, then the
    /// unfiltered subscribe restoring the topic's all-symbols steady state),
    /// empty if it was already desired (idempotent).
    pub(crate) fn add(&mut self, sub: FeedSub) -> Vec<String> {
        if !self.desired.insert(sub) {
            return Vec::new();
        }
        vec![
            backfill_subscribe_message(sub.source, sub.asset),
            stream_subscribe_message(sub.source),
        ]
    }

    /// Removes from the desired set; the messages to send (a topic-level
    /// unsubscribe once its last tracked stream is gone — while any stream
    /// remains, dropping the symbol is purely client-side), empty if it
    /// wasn't desired (idempotent).
    pub(crate) fn remove(&mut self, sub: FeedSub) -> Vec<String> {
        if !self.desired.remove(&sub) {
            return Vec::new();
        }
        if self.topic_active(sub.source) {
            Vec::new()
        } else {
            vec![stream_unsubscribe_message(sub.source)]
        }
    }

    /// The full (re)connect sequence: per active topic, one filtered
    /// subscribe per tracked stream (collecting backfills), then the
    /// unfiltered steady-state subscribe.
    pub(crate) fn connect_messages(&self) -> Vec<String> {
        let mut messages = Vec::new();
        for source in [RtdsSource::Binance, RtdsSource::Chainlink] {
            let assets: Vec<Asset> = self
                .desired
                .iter()
                .filter(|s| s.source == source)
                .map(|s| s.asset)
                .collect();
            if assets.is_empty() {
                continue;
            }
            for asset in assets {
                messages.push(backfill_subscribe_message(source, asset));
            }
            messages.push(stream_subscribe_message(source));
        }
        messages
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = FeedSub> + '_ {
        self.desired.iter().copied()
    }

    fn topic_active(&self, source: RtdsSource) -> bool {
        self.desired.iter().any(|s| s.source == source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_covers_both_topics_and_assets() {
        let all = FeedSub::all();
        assert_eq!(all.len(), 4);
        let set = SubSet::new(all);
        assert_eq!(set.iter().count(), 4);
    }

    #[test]
    fn connect_sequence_is_backfills_then_stream_per_topic() {
        let set = SubSet::new(FeedSub::all());
        let messages = set.connect_messages();
        assert_eq!(messages.len(), 6, "2 backfills + 1 stream per topic");
        // Binance block first (BTreeSet order), then Chainlink.
        assert!(messages[0].contains(r#"{\"symbol\":\"btcusdt\"}"#));
        assert!(messages[1].contains(r#"{\"symbol\":\"ethusdt\"}"#));
        assert_eq!(
            messages[2],
            r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices","type":"update"}]}"#
        );
        assert!(messages[3].contains(r#"{\"symbol\":\"btc/usd\"}"#));
        assert!(messages[4].contains(r#"{\"symbol\":\"eth/usd\"}"#));
        assert_eq!(
            messages[5],
            r#"{"action":"subscribe","subscriptions":[{"filters":"","topic":"crypto_prices_chainlink","type":"*"}]}"#
        );
    }

    #[test]
    fn connect_sequence_skips_inactive_topics() {
        let set = SubSet::new([FeedSub::new(RtdsSource::Chainlink, Asset::Btc)]);
        let messages = set.connect_messages();
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().all(|m| m.contains("chainlink")));
    }

    #[test]
    fn add_emits_backfill_then_stream_and_is_idempotent() {
        let mut set = SubSet::new([]);
        let sub = FeedSub::new(RtdsSource::Chainlink, Asset::Btc);
        let messages = set.add(sub);
        assert_eq!(messages.len(), 2);
        assert!(messages[0].contains(r#"{\"symbol\":\"btc/usd\"}"#));
        assert!(messages[1].contains(r#""filters":"""#));
        assert!(set.add(sub).is_empty(), "duplicate add is a no-op");
    }

    #[test]
    fn remove_unsubscribes_topic_only_when_last_stream_leaves() {
        let mut set = SubSet::new([
            FeedSub::new(RtdsSource::Binance, Asset::Btc),
            FeedSub::new(RtdsSource::Binance, Asset::Eth),
        ]);
        // First removal: topic still tracked — client-side drop only.
        assert!(
            set.remove(FeedSub::new(RtdsSource::Binance, Asset::Btc))
                .is_empty()
        );
        // Last removal: topic-level unsubscribe.
        let messages = set.remove(FeedSub::new(RtdsSource::Binance, Asset::Eth));
        assert_eq!(messages.len(), 1);
        assert!(messages[0].contains(r#""action":"unsubscribe""#));
        assert!(messages[0].contains("crypto_prices"));
        // Double remove is a no-op.
        assert!(
            set.remove(FeedSub::new(RtdsSource::Binance, Asset::Eth))
                .is_empty()
        );
        assert!(set.connect_messages().is_empty());
    }
}
