//! Price/market data feed endpoints and connection behavior (CLAUDE.md §7).

use core_types::DurationMs;
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Feed endpoints and WebSocket connection parameters.
///
/// URL defaults come from CLAUDE.md §7 (verified June 2026). They are
/// config-overridable so a host change never requires a rebuild, but there is
/// normally no reason to touch them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeedsConfig {
    /// Polymarket Real-Time Data Socket (Chainlink + Binance price topics).
    pub rtds_url: String,
    /// Polymarket CLOB market channel WebSocket (public).
    pub clob_market_ws_url: String,
    /// Polymarket CLOB user channel WebSocket (authenticated).
    pub clob_user_ws_url: String,
    /// Polymarket CLOB REST host.
    pub clob_rest_url: String,
    /// Gamma discovery REST host.
    pub gamma_url: String,
    /// Direct Binance WebSocket host (lowest-latency signal feed), as a base
    /// URL — feed-binance appends the combined-stream path
    /// (`/stream?streams=…`). The market-data-only host
    /// `wss://data-stream.binance.vision` is a drop-in override.
    pub binance_ws_url: String,
    /// RTDS keepalive PING interval. The docs require a PING every 5 s, so
    /// values above 5000 ms are rejected.
    pub rtds_ping_interval_ms: DurationMs,
    /// WebSocket connect timeout for all feeds.
    pub ws_connect_timeout_ms: DurationMs,
    /// First reconnect delay after a feed disconnect.
    pub reconnect_initial_backoff_ms: DurationMs,
    /// Reconnect delay ceiling.
    pub reconnect_max_backoff_ms: DurationMs,
    /// Multiplier applied to the delay after each failed reconnect attempt.
    pub reconnect_backoff_multiplier: f64,
    /// Age (on local receive time) past which an RTDS (source, asset) stream
    /// is declared stale by the feed watchdog. RTDS delivers ~1 update/s per
    /// symbol, so this must stay well above 1000 ms to avoid flapping; the
    /// §11 risk-manager staleness bound is a separate, tighter policy over
    /// the fast feeds.
    pub rtds_stale_after_ms: DurationMs,
    /// Staleness threshold for the direct-Binance top-of-book streams
    /// (`bookTicker` — near-continuous on BTC/ETH, so a short threshold is
    /// safe). The watchdog scans at 1 s; values under 500 ms cannot be
    /// honored and would flap.
    pub binance_stale_after_ms: DurationMs,
    /// Staleness threshold for the direct-Binance trade-print streams.
    /// Trades legitimately gap for seconds in a quiet market — this must be
    /// far looser than the top-of-book threshold or a quiet-but-healthy
    /// stream flaps Stale/Recovered on the bus.
    pub binance_trade_stale_after_ms: DurationMs,
    /// CLOB market-channel keepalive PING interval. The docs require a PING
    /// every 10 s; the default halves it (latency-harness precedent), and
    /// values above 10000 ms are rejected.
    pub clob_ping_interval_ms: DurationMs,
    /// Book silence (no book/price_change/best_bid_ask for a token) past
    /// which an *active* window's L2 book is treated as desync evidence and
    /// the connection recycled for a fresh snapshot. Pre-open/post-close
    /// books are exempt (silence there is normal), and feed-clob adapts the
    /// effective threshold upward when a recycle proves the market merely
    /// quiet.
    pub clob_book_stale_after_ms: DurationMs,
    /// How long before a window's open feed-clob connects and subscribes to
    /// its token pair. Large enough that the successor's books are
    /// snapshotted before the predecessor closes — the §6 gap-free rollover.
    pub clob_presubscribe_lead_ms: DurationMs,
    /// Coalescing interval for full `Event::Book` snapshots built from
    /// `price_change` deltas (top-of-book updates always publish
    /// immediately; venue `book` snapshots bypass the coalescer). Bounds bus
    /// and allocation churn at the observed ~800 msg/s per active pair.
    pub clob_book_publish_interval_ms: DurationMs,
    /// How long a closed window's connection lingers waiting for its
    /// `market_resolved` event before being torn down anyway.
    pub clob_resolution_linger_ms: DurationMs,
}

impl Default for FeedsConfig {
    fn default() -> Self {
        Self {
            rtds_url: "wss://ws-live-data.polymarket.com".to_owned(),
            clob_market_ws_url: "wss://ws-subscriptions-clob.polymarket.com/ws/market".to_owned(),
            clob_user_ws_url: "wss://ws-subscriptions-clob.polymarket.com/ws/user".to_owned(),
            clob_rest_url: "https://clob.polymarket.com".to_owned(),
            gamma_url: "https://gamma-api.polymarket.com".to_owned(),
            binance_ws_url: "wss://stream.binance.com:9443".to_owned(),
            rtds_ping_interval_ms: DurationMs::from_millis(5_000),
            ws_connect_timeout_ms: DurationMs::from_millis(5_000),
            reconnect_initial_backoff_ms: DurationMs::from_millis(250),
            reconnect_max_backoff_ms: DurationMs::from_millis(10_000),
            reconnect_backoff_multiplier: 2.0,
            rtds_stale_after_ms: DurationMs::from_millis(5_000),
            binance_stale_after_ms: DurationMs::from_millis(2_500),
            binance_trade_stale_after_ms: DurationMs::from_millis(30_000),
            clob_ping_interval_ms: DurationMs::from_millis(5_000),
            clob_book_stale_after_ms: DurationMs::from_millis(10_000),
            clob_presubscribe_lead_ms: DurationMs::from_millis(90_000),
            clob_book_publish_interval_ms: DurationMs::from_millis(200),
            clob_resolution_linger_ms: DurationMs::from_millis(180_000),
        }
    }
}

impl FeedsConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        let ws_urls = [
            ("feeds.rtds_url", &self.rtds_url),
            ("feeds.clob_market_ws_url", &self.clob_market_ws_url),
            ("feeds.clob_user_ws_url", &self.clob_user_ws_url),
            ("feeds.binance_ws_url", &self.binance_ws_url),
        ];
        for (key, url) in ws_urls {
            v.require(url.starts_with("wss://"), key, "must start with wss://");
        }
        let rest_urls = [
            ("feeds.clob_rest_url", &self.clob_rest_url),
            ("feeds.gamma_url", &self.gamma_url),
        ];
        for (key, url) in rest_urls {
            v.require(url.starts_with("https://"), key, "must start with https://");
        }

        let ping = self.rtds_ping_interval_ms.as_millis();
        v.require(
            ping > 0 && ping <= 5_000,
            "feeds.rtds_ping_interval_ms",
            "must be in (0, 5000] — the RTDS docs require a PING at least every 5 seconds",
        );
        v.require(
            self.ws_connect_timeout_ms.as_millis() > 0,
            "feeds.ws_connect_timeout_ms",
            "must be > 0",
        );
        let initial = self.reconnect_initial_backoff_ms.as_millis();
        let max = self.reconnect_max_backoff_ms.as_millis();
        v.require(
            initial > 0,
            "feeds.reconnect_initial_backoff_ms",
            "must be > 0",
        );
        v.require(
            max >= initial,
            "feeds.reconnect_max_backoff_ms",
            "must be >= reconnect_initial_backoff_ms",
        );
        v.require(
            self.reconnect_backoff_multiplier.is_finite()
                && self.reconnect_backoff_multiplier > 1.0,
            "feeds.reconnect_backoff_multiplier",
            "must be a finite number > 1.0",
        );
        v.require(
            self.rtds_stale_after_ms.as_millis() > 1_000,
            "feeds.rtds_stale_after_ms",
            "must be > 1000 — RTDS updates arrive ~1/s per symbol; lower values flap",
        );
        v.require(
            self.binance_stale_after_ms.as_millis() >= 500,
            "feeds.binance_stale_after_ms",
            "must be >= 500 — the feed watchdog scans at 1 s; lower values flap",
        );
        v.require(
            self.binance_trade_stale_after_ms >= self.binance_stale_after_ms,
            "feeds.binance_trade_stale_after_ms",
            "must be >= binance_stale_after_ms — trade prints legitimately gap for seconds \
             in quiet markets",
        );
        let clob_ping = self.clob_ping_interval_ms.as_millis();
        v.require(
            clob_ping > 0 && clob_ping <= 10_000,
            "feeds.clob_ping_interval_ms",
            "must be in (0, 10000] — the market-channel docs require a PING at least every \
             10 seconds",
        );
        v.require(
            self.clob_book_stale_after_ms.as_millis() >= 2_000,
            "feeds.clob_book_stale_after_ms",
            "must be >= 2000 — the feed watchdog scans at 1 s; lower values flap",
        );
        v.require(
            self.clob_presubscribe_lead_ms.as_millis() > 0,
            "feeds.clob_presubscribe_lead_ms",
            "must be > 0 — the next window must be subscribed before the current one closes",
        );
        let publish = self.clob_book_publish_interval_ms.as_millis();
        v.require(
            publish > 0 && publish <= 2_000,
            "feeds.clob_book_publish_interval_ms",
            "must be in (0, 2000] — consumers need a reasonably fresh full book",
        );
        v.require(
            self.clob_resolution_linger_ms.as_millis() >= 30_000,
            "feeds.clob_resolution_linger_ms",
            "must be >= 30000 — market_resolved can lag close by tens of seconds",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        FeedsConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_bad_urls_and_ranges() {
        let cfg = FeedsConfig {
            rtds_url: "ws://insecure.example.com".to_owned(),
            gamma_url: "http://insecure.example.com".to_owned(),
            rtds_ping_interval_ms: DurationMs::from_millis(6_000),
            reconnect_backoff_multiplier: 1.0,
            rtds_stale_after_ms: DurationMs::from_millis(1_000),
            binance_stale_after_ms: DurationMs::from_millis(499),
            ..FeedsConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"feeds.rtds_url"));
        assert!(keys.contains(&"feeds.gamma_url"));
        assert!(keys.contains(&"feeds.rtds_ping_interval_ms"));
        assert!(keys.contains(&"feeds.reconnect_backoff_multiplier"));
        assert!(keys.contains(&"feeds.rtds_stale_after_ms"));
        assert!(keys.contains(&"feeds.binance_stale_after_ms"));
    }

    #[test]
    fn rejects_bad_clob_ranges() {
        let cfg = FeedsConfig {
            clob_ping_interval_ms: DurationMs::from_millis(10_001),
            clob_book_stale_after_ms: DurationMs::from_millis(1_999),
            clob_presubscribe_lead_ms: DurationMs::from_millis(0),
            clob_book_publish_interval_ms: DurationMs::from_millis(2_001),
            clob_resolution_linger_ms: DurationMs::from_millis(29_999),
            ..FeedsConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"feeds.clob_ping_interval_ms"));
        assert!(keys.contains(&"feeds.clob_book_stale_after_ms"));
        assert!(keys.contains(&"feeds.clob_presubscribe_lead_ms"));
        assert!(keys.contains(&"feeds.clob_book_publish_interval_ms"));
        assert!(keys.contains(&"feeds.clob_resolution_linger_ms"));
    }

    #[test]
    fn trade_staleness_must_be_at_least_book_staleness() {
        let cfg = FeedsConfig {
            binance_stale_after_ms: DurationMs::from_millis(10_000),
            binance_trade_stale_after_ms: DurationMs::from_millis(9_999),
            ..FeedsConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert!(
            violations
                .iter()
                .any(|x| x.key == "feeds.binance_trade_stale_after_ms")
        );
    }
}
