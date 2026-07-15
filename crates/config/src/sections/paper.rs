//! Paper-trading settings (CLAUDE.md §9): simulated money, real market data.

use core_types::{Decimal, Dollars, DurationMs};
use rust_decimal::dec;
use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// A simulated latency: uniform in `[mean_ms, mean_ms + jitter_ms]`.
///
/// Jitter is one-sided **upward** on purpose: paper fills must never be more
/// optimistic than reality (§9), so latency can only be worse than the
/// configured mean, never better.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LatencyConfig {
    /// Minimum simulated latency.
    pub mean_ms: DurationMs,
    /// Additional uniform random latency on top of `mean_ms`.
    pub jitter_ms: DurationMs,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        // Deliberately pessimistic placeholders until the latency harness
        // (timeutil crate) measures real RTTs from the deployment box.
        Self {
            mean_ms: DurationMs::from_millis(150),
            jitter_ms: DurationMs::from_millis(100),
        }
    }
}

impl LatencyConfig {
    fn validate_into(&self, prefix: &str, v: &mut Violations) {
        v.require(
            self.mean_ms.as_millis() >= 1,
            format!("{prefix}.mean_ms"),
            "must be >= 1 — zero-latency paper fills would be more optimistic than reality (§9)",
        );
        v.require(
            self.jitter_ms.as_millis() >= 0,
            format!("{prefix}.jitter_ms"),
            "must be >= 0",
        );
    }
}

/// Paper venue settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PaperConfig {
    /// Starting capital of the paper wallet. Runtime-adjustable from the
    /// dashboard (changes are journaled); this is only the boot value.
    pub starting_capital: Dollars,
    /// Simulated order-placement (network round-trip) latency.
    pub placement_latency: LatencyConfig,
    /// Simulated cancel round-trip latency.
    pub cancel_latency: LatencyConfig,
    /// Venue-imposed **taker** delay, applied on top of `placement_latency` for
    /// **marketable** (FAK / marketable-limit) orders only — the ~250 ms hold
    /// Polymarket imposes before a taker order matches (flagged `itode` on the
    /// market; verified enabled on all four series 2026-07-09). Resting
    /// post-only/GTC makers never match on entry, so they get network latency
    /// and no venue hold. `0` disables it. All four series carry the delay
    /// today; per-market itode-gating is a documented follow-up.
    pub venue_taker_delay_ms: DurationMs,
    /// Fallback taker fee rate used **only** when a market's own fee params
    /// have not been fetched yet — §7: never hardcode 0.07 in logic, always
    /// read per-market params at discovery.
    pub default_fee_rate: Decimal,
    /// Maker rebate share of collected taker fees (crypto category: 20%).
    pub maker_rebate_share: Decimal,
    /// Simulate the tiered **Taker Fee Rebate Program** (30-day rolling weighted
    /// volume → tier → `tier% × taker fees`, reported as a separate line — NEVER
    /// added to paper cash/equity). `false` disables the projection.
    pub taker_rebate_enabled: bool,
}

impl Default for PaperConfig {
    fn default() -> Self {
        Self {
            starting_capital: Dollars::new(Decimal::from(10_000)),
            placement_latency: LatencyConfig::default(),
            cancel_latency: LatencyConfig::default(),
            venue_taker_delay_ms: DurationMs::from_millis(250),
            default_fee_rate: dec!(0.07),
            maker_rebate_share: dec!(0.20),
            taker_rebate_enabled: true,
        }
    }
}

impl PaperConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            self.starting_capital.as_decimal() > Decimal::ZERO,
            "paper.starting_capital",
            "must be > 0",
        );
        self.placement_latency
            .validate_into("paper.placement_latency", v);
        self.cancel_latency.validate_into("paper.cancel_latency", v);
        v.require(
            self.venue_taker_delay_ms.as_millis() >= 0,
            "paper.venue_taker_delay_ms",
            "must be >= 0 — a negative venue delay would fill takers earlier than reality (§9)",
        );
        v.require(
            self.default_fee_rate >= Decimal::ZERO && self.default_fee_rate < Decimal::ONE,
            "paper.default_fee_rate",
            "must be in [0, 1)",
        );
        v.require(
            self.maker_rebate_share >= Decimal::ZERO && self.maker_rebate_share <= Decimal::ONE,
            "paper.maker_rebate_share",
            "must be in [0, 1]",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        PaperConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_nonpositive_capital() {
        for capital in [Decimal::ZERO, Decimal::from(-5)] {
            let cfg = PaperConfig {
                starting_capital: Dollars::new(capital),
                ..PaperConfig::default()
            };
            let mut v = Violations::default();
            cfg.validate_into(&mut v);
            let violations = v.into_result().unwrap_err();
            assert!(violations.iter().any(|x| x.key == "paper.starting_capital"));
        }
    }

    #[test]
    fn rejects_negative_venue_taker_delay() {
        let cfg = PaperConfig {
            venue_taker_delay_ms: DurationMs::from_millis(-1),
            ..PaperConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert!(
            violations
                .iter()
                .any(|x| x.key == "paper.venue_taker_delay_ms")
        );
    }

    #[test]
    fn rejects_zero_latency_paper() {
        let cfg = PaperConfig {
            placement_latency: LatencyConfig {
                mean_ms: DurationMs::ZERO,
                jitter_ms: DurationMs::ZERO,
            },
            ..PaperConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert!(
            violations
                .iter()
                .any(|x| x.key == "paper.placement_latency.mean_ms")
        );
    }
}
