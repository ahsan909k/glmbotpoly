//! Engine-local risk-manager tunables (CLAUDE.md §11).
//!
//! The bundle the [`RiskManager`](super::RiskManager) destructures to build its
//! [`RiskCore`](super::core::RiskCore) thresholds and to construct the strategy
//! modules it owns. Mapped from `config::RiskConfig` + `config::EngineParams` at
//! the `bot` boundary (the [`InventoryParams`](crate::InventoryParams) /
//! [`NormalizerParams`](crate::NormalizerParams) precedent — the `config` crate
//! is never a dependency of `engine`); [`Default`] mirrors the committed
//! `config/default.toml` `[risk]` section and the strategy defaults.

use std::collections::HashMap;

use core_types::{Asset, Decimal, Dollars};

use crate::arbitration::TakerId;
use crate::late_window::LateWindowTakerParams;
use crate::model_taker::ModelTakerParams;
use crate::normalize::NormalizerParams;
use crate::quote_manager::QuoteManagerParams;
use crate::quoting::QuoteParams;
use crate::taker::MomentumTakerParams;

/// The default per-asset arbitration precedence — the "fortress map": momentum
/// wins BTC conflicts, the model wins ETH conflicts (CLAUDE.md §8). Inert until
/// the model taker is enabled (it only records fires when `model_enabled`).
#[must_use]
pub fn default_series_precedence() -> HashMap<Asset, TakerId> {
    HashMap::from([
        (Asset::Btc, TakerId::Momentum),
        (Asset::Eth, TakerId::Model),
    ])
}

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
    /// **Home-testing concession, default 0 (off).** Extra tolerance (ms) added
    /// to the fast-feed staleness *timer* only: the `FeedStale` breaker trips
    /// when the fast feed has been continuously silent for
    /// `feed_staleness_ms + feed_staleness_grace_ms`. On a colocated VPS leave
    /// this at 0 (strict §11). On a jittery home link the direct-Binance stream
    /// intermittently gaps > 500 ms; without grace every gap fires cancel-all
    /// and the engine never rests quotes long enough to trade. Grace tolerates
    /// transient gaps while a genuine feed death (a gap beyond the sum) still
    /// trips. Latched `FeedHealth`/`BookHealth` stale reports are unaffected.
    pub feed_staleness_grace_ms: i64,
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
    /// Paper-eval only, default `false`. When `true`, the two loss-based stops
    /// (`DailyStop`, per-window `WindowLoss`) are computed exactly but only
    /// RECORDED (a `ShadowStop`) instead of halting/cancelling — trading
    /// continues. The operational breakers stay hard. Keep `false` in any real
    /// deployment (CLAUDE.md §11).
    pub shadow_loss_stops: bool,

    /// Whether the owned quote manager is driven (default `true`).
    pub quoter_enabled: bool,
    /// Whether the owned momentum taker is driven (default `true`).
    pub momentum_enabled: bool,
    /// Whether the owned late-window taker is driven (default `true`).
    pub late_window_enabled: bool,
    /// Whether the owned model taker is driven (**default `false`** — the model
    /// taker is off unless explicitly enabled and shadow is running).
    pub model_enabled: bool,

    /// The conflict-arbitration window (ms): a losing driver is suppressed on a
    /// window only while the winning driver fired that same window within this
    /// span (§8 fortress map).
    pub arbitration_window_ms: i64,
    /// Per-asset arbitration precedence (which taker wins a same-window conflict).
    pub series_precedence: HashMap<Asset, TakerId>,

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
    /// Tunables for the owned model taker.
    pub model: ModelTakerParams,
}

impl Default for RiskParams {
    fn default() -> Self {
        Self {
            feed_staleness_ms: 500,
            feed_staleness_grace_ms: 0,
            daily_stop_loss: Dollars::new(Decimal::from(200)),
            max_open_notional: Dollars::new(Decimal::from(1_000)),
            sanity_bound: 0.10,
            sanity_bound_duration_ms: 3_000,
            error_breaker_max_errors: 10,
            error_breaker_window_ms: 60_000,
            engine_restart_cooldown_ms: 5_000,
            shadow_loss_stops: false,
            quoter_enabled: true,
            momentum_enabled: true,
            late_window_enabled: true,
            model_enabled: false,
            arbitration_window_ms: 3_000,
            series_precedence: default_series_precedence(),
            quote_manager: QuoteManagerParams::default(),
            quote: QuoteParams::default(),
            normalizer: NormalizerParams::default(),
            momentum: MomentumTakerParams::default(),
            late_window: LateWindowTakerParams::default(),
            model: ModelTakerParams::default(),
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
        assert_eq!(p.feed_staleness_grace_ms, 0);
        assert_eq!(p.daily_stop_loss, Dollars::new(Decimal::from(200)));
        assert_eq!(p.max_open_notional, Dollars::new(Decimal::from(1_000)));
        assert!((p.sanity_bound - 0.10).abs() < f64::EPSILON);
        assert_eq!(p.sanity_bound_duration_ms, 3_000);
        assert_eq!(p.error_breaker_max_errors, 10);
        assert_eq!(p.error_breaker_window_ms, 60_000);
        assert!(!p.shadow_loss_stops, "shadow loss stops off by default");
        assert!(p.quoter_enabled && p.momentum_enabled && p.late_window_enabled);
        // The model taker is off by default; the fortress map is the default
        // precedence (momentum wins BTC, model wins ETH), inert until enabled.
        assert!(!p.model_enabled);
        assert_eq!(p.arbitration_window_ms, 3_000);
        assert_eq!(
            p.series_precedence.get(&Asset::Btc),
            Some(&TakerId::Momentum)
        );
        assert_eq!(p.series_precedence.get(&Asset::Eth), Some(&TakerId::Model));
        assert_eq!(p.model, ModelTakerParams::default());
    }
}
