//! Gamma/CLOB market discovery settings (CLAUDE.md §6).

use core_types::{Asset, DurationMs, Series, WindowDuration};
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Gamma series slugs for the six Up/Down products.
///
/// Verified against the live Gamma API on 2026-06-11 (`GET /series?slug=…`).
/// Config-overridable so a Polymarket slug change never requires a rebuild,
/// but there is normally no reason to touch them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SeriesSlugs {
    /// BTC 5-minute series.
    pub btc_5m: String,
    /// BTC 15-minute series.
    pub btc_15m: String,
    /// BTC hourly series.
    pub btc_1h: String,
    /// ETH 5-minute series.
    pub eth_5m: String,
    /// ETH 15-minute series.
    pub eth_15m: String,
    /// ETH hourly series.
    pub eth_1h: String,
}

impl Default for SeriesSlugs {
    fn default() -> Self {
        Self {
            btc_5m: "btc-up-or-down-5m".to_owned(),
            btc_15m: "btc-up-or-down-15m".to_owned(),
            btc_1h: "btc-up-or-down-hourly".to_owned(),
            eth_5m: "eth-up-or-down-5m".to_owned(),
            eth_15m: "eth-up-or-down-15m".to_owned(),
            eth_1h: "eth-up-or-down-hourly".to_owned(),
        }
    }
}

impl SeriesSlugs {
    /// The Gamma series slug for one of the six series.
    #[must_use]
    pub fn get(&self, series: Series) -> &str {
        match (series.asset, series.duration) {
            (Asset::Btc, WindowDuration::M5) => &self.btc_5m,
            (Asset::Btc, WindowDuration::M15) => &self.btc_15m,
            (Asset::Btc, WindowDuration::H1) => &self.btc_1h,
            (Asset::Eth, WindowDuration::M5) => &self.eth_5m,
            (Asset::Eth, WindowDuration::M15) => &self.eth_15m,
            (Asset::Eth, WindowDuration::H1) => &self.eth_1h,
        }
    }
}

/// Discovery subsystem settings: how the bot finds the current and upcoming
/// windows of each series over the Gamma/CLOB REST APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiscoveryConfig {
    /// Gamma series slug per series.
    pub slugs: SeriesSlugs,
    /// HTTP request timeout for Gamma/CLOB REST calls.
    pub http_timeout_ms: DurationMs,
    /// Upcoming windows to hold per series beyond the current one. CLAUDE.md
    /// §6 requires at least the next window, so the minimum is 1.
    pub lookahead: u32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            slugs: SeriesSlugs::default(),
            http_timeout_ms: DurationMs::from_millis(5_000),
            lookahead: 2,
        }
    }
}

impl DiscoveryConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        let slugs = [
            ("discovery.slugs.btc_5m", &self.slugs.btc_5m),
            ("discovery.slugs.btc_15m", &self.slugs.btc_15m),
            ("discovery.slugs.btc_1h", &self.slugs.btc_1h),
            ("discovery.slugs.eth_5m", &self.slugs.eth_5m),
            ("discovery.slugs.eth_15m", &self.slugs.eth_15m),
            ("discovery.slugs.eth_1h", &self.slugs.eth_1h),
        ];
        for (key, slug) in slugs {
            v.require(
                !slug.is_empty() && !slug.chars().any(char::is_whitespace),
                key,
                "must be a non-empty slug without whitespace",
            );
        }
        for (i, (key, slug)) in slugs.iter().enumerate() {
            let duplicated = slugs[..i].iter().any(|(_, other)| other == slug);
            v.require(
                !duplicated,
                *key,
                "duplicates another series slug — every series must map to a distinct Gamma series",
            );
        }
        v.require(
            self.http_timeout_ms.as_millis() > 0,
            "discovery.http_timeout_ms",
            "must be > 0",
        );
        v.require(
            (1..=10).contains(&self.lookahead),
            "discovery.lookahead",
            "must be in [1, 10] — §6 requires holding at least the next window",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        DiscoveryConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn every_series_has_a_distinct_default_slug() {
        let slugs = SeriesSlugs::default();
        let all: Vec<&str> = Series::ALL.iter().map(|&s| slugs.get(s)).collect();
        for (i, slug) in all.iter().enumerate() {
            assert!(!slug.is_empty());
            assert!(!all[..i].contains(slug), "duplicate slug {slug}");
        }
    }

    #[test]
    fn slug_lookup_matches_fields() {
        let slugs = SeriesSlugs::default();
        let btc_1h: Series = "BTC-1h".parse().unwrap();
        let eth_5m: Series = "ETH-5m".parse().unwrap();
        assert_eq!(slugs.get(btc_1h), "btc-up-or-down-hourly");
        assert_eq!(slugs.get(eth_5m), "eth-up-or-down-5m");
    }

    #[test]
    fn rejects_empty_whitespace_and_duplicate_slugs() {
        let cfg = DiscoveryConfig {
            slugs: SeriesSlugs {
                btc_5m: String::new(),
                btc_15m: "has whitespace".to_owned(),
                eth_5m: "eth-up-or-down-hourly".to_owned(), // duplicates eth_1h
                ..SeriesSlugs::default()
            },
            ..DiscoveryConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"discovery.slugs.btc_5m"));
        assert!(keys.contains(&"discovery.slugs.btc_15m"));
        // The later of the duplicate pair is flagged.
        assert!(keys.contains(&"discovery.slugs.eth_1h"));
    }

    #[test]
    fn rejects_zero_timeout_and_out_of_range_lookahead() {
        let cfg = DiscoveryConfig {
            http_timeout_ms: DurationMs::from_millis(0),
            lookahead: 0,
            ..DiscoveryConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"discovery.http_timeout_ms"));
        assert!(keys.contains(&"discovery.lookahead"));

        let cfg = DiscoveryConfig {
            lookahead: 11,
            ..DiscoveryConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_err());
    }
}
