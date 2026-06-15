//! Engine-local risk-manager tunables (CLAUDE.md §11).
//!
//! The bundle the [`RiskManager`](super::RiskManager) destructures to build its
//! [`RiskCore`](super::core::RiskCore) thresholds and to construct the strategy
//! modules it owns. Mapped from `config::RiskConfig` + `config::EngineParams` at
//! the `bot` boundary (the [`InventoryParams`](crate::InventoryParams) /
//! [`NormalizerParams`](crate::NormalizerParams) precedent — the `config` crate
//! is never a dependency of `engine`); [`Default`] mirrors the committed
//! `config/default.toml` `[risk]` section and the strategy defaults.

use core_types::{Decimal, Dollars};

use crate::late_window::LateWindowTakerParams;
use crate::normalize::NormalizerParams;
use crate::quote_manager::QuoteManagerParams;
use crate::quoting::QuoteParams;
use crate::taker::MomentumTakerParams;

/// Risk-manager limits + the strategy bundle the manager owns. Holds `f64`
/// thresholds and embedded strategy params, so **not** `Copy`/`Eq`.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskParams {
    /// Staleness ceiling (ms) for the fast signal feed (BinanceDirect/Mid).
    /// §11 fixes it at 500 ms; the active book's staleness is governed by
    /// `Event::BookHealth` and the Chainlink feed by `Event::FeedHealth` at
    /// their own cadence-appropriate thresholds (a 500 ms bound on a ~1 Hz
    /// Chainlink stream would trip permanently — see the Decisions Log).
    pub feed_staleness_ms: i64,
    /// Daily stop-loss: once cumulative realized PnL for the UTC day reaches
    /// this loss, all trading halts until a manual `ControlEvent::Reset`.
    pub daily_stop_loss: Dollars,
    /// Global cap on open notional across all windows (a pre-trade guard veto).
    pub max_open_notional: Dollars,
    /// Stand-down threshold on `|model fair − book mid|` (probability space).
    pub sanity_bound: f64,
    /// How long the sanity bound must be continuously exceeded before quotes
    /// are pulled (`FairVsMid`).
    pub sanity_bound_duration_ms: i64,
    /// Infra error count within the rolling window that trips the error-rate
    /// breaker.
    pub error_breaker_max_errors: u32,
    /// The error-rate rolling window (ms).
    pub error_breaker_window_ms: i64,
    /// How long after the last matching-engine-restart signal the `EngineRestart`
    /// breaker stays tripped before it clears (no `config` field yet — the
    /// venue's post-restart post-only window is ~2 min, so a few seconds of
    /// cool-down after the last 425/503 is the conservative re-arm point).
    pub engine_restart_cooldown_ms: i64,

    /// Whether the owned quote manager is driven (default `true`).
    pub quoter_enabled: bool,
    /// Whether the owned momentum taker is driven (default `true`).
    pub momentum_enabled: bool,
    /// Whether the owned late-window taker is driven (default `true`).
    pub late_window_enabled: bool,

    /// Tunables for the owned quote manager.
    pub quote_manager: QuoteManagerParams,
    /// Tunables for the owned quoting calculator.
    pub quote: QuoteParams,
    /// Tunables for the order normalizer shared by all strategies.
    pub normalizer: NormalizerParams,
    /// Tunables for the owned momentum taker.
    pub momentum: MomentumTakerParams,
    /// Tunables for the owned late-window taker.
    pub late_window: LateWindowTakerParams,
}

impl Default for RiskParams {
    fn default() -> Self {
        Self {
            feed_staleness_ms: 500,
            daily_stop_loss: Dollars::new(Decimal::from(200)),
            max_open_notional: Dollars::new(Decimal::from(1_000)),
            sanity_bound: 0.10,
            sanity_bound_duration_ms: 3_000,
            error_breaker_max_errors: 10,
            error_breaker_window_ms: 60_000,
            engine_restart_cooldown_ms: 5_000,
            quoter_enabled: true,
            momentum_enabled: true,
            late_window_enabled: true,
            quote_manager: QuoteManagerParams::default(),
            quote: QuoteParams::default(),
            normalizer: NormalizerParams::default(),
            momentum: MomentumTakerParams::default(),
            late_window: LateWindowTakerParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_config_risk_section() {
        // Pinned to config/default.toml [risk] + the §11 per-window cap. The
        // config→RiskParams map at the bot boundary is deferred (the
        // InventoryParams/NormalizerParams precedent), so this pins the values
        // without a `config` dependency.
        let p = RiskParams::default();
        assert_eq!(p.feed_staleness_ms, 500);
        assert_eq!(p.daily_stop_loss, Dollars::new(Decimal::from(200)));
        assert_eq!(p.max_open_notional, Dollars::new(Decimal::from(1_000)));
        assert!((p.sanity_bound - 0.10).abs() < f64::EPSILON);
        assert_eq!(p.sanity_bound_duration_ms, 3_000);
        assert_eq!(p.error_breaker_max_errors, 10);
        assert_eq!(p.error_breaker_window_ms, 60_000);
        assert!(p.quoter_enabled && p.momentum_enabled && p.late_window_enabled);
    }
}
