//! The late-window certainty taker (CLAUDE.md §8): the endgame complement to the
//! [momentum taker](crate::taker). When the window is near resolution (τ at or
//! below a threshold) and the **Chainlink-anchored** model probability is beyond a
//! certainty threshold, it fires a single FAK marketable BUY of the near-certain
//! side up to a configured price cap, sized against the displayed ask depth and
//! the per-window taker budget.
//!
//! ## Resolution-grade feed only
//!
//! Unlike the momentum taker, this decision uses ONLY the resolution-grade
//! Chainlink feed — never the fast direct-Binance signal. It consumes no
//! [`PriceTick`](core_types::Event::PriceTick)s at all; the candidate side and its
//! certainty come from the [`ModelSnapshot`](core_types::ModelSnapshot)'s `p_up`,
//! and the taker acts only when that snapshot was anchored on Chainlink
//! ([`AnchorSource::Chainlink`](core_types::AnchorSource::Chainlink)) — so a
//! fast-feed-anchored fair, however certain, is refused. Fees are near zero this
//! deep in the tail, so the accept gate is a simple price cap (not the
//! fee-plus-buffer edge gate the momentum taker uses).
//!
//! ## Tie rule (§6)
//!
//! Resolution is "end price ≥ strike ⇒ Up" — exact-at-strike leans Up. That tie is
//! decided inside the model (its fair value uses `if s >= k { p_up = 1.0 }`, the
//! `>=` making S==K resolve Up), so this taker simply honors the saturated `p_up`:
//! it accepts `p_up ∈ [0, 1]` *inclusive* (a near-certain endgame fair legitimately
//! saturates to 1.0 / 0.0, unlike the momentum taker which needs a strict `(0,1)`
//! interior for its edge math) and selects the side with inclusive comparisons
//! (`p_up >= certainty_threshold` ⇒ Up). So a `p_up` of exactly 1.0 — the model's
//! S≥K representation — buys Up, and 0.0 buys Down.
//!
//! ## Split into a pure calculator + effectful driver
//!
//! Mirrors [`crate::taker`]: [`edge`] ([`plan_certainty_take`] →
//! [`CertaintyTakePlan`]) is pure; [`driver`] ([`LateWindowTaker`]) is the one
//! async, IO-bearing, `tracing`-emitting module (target `late-window-taker`).
//!
//! ## Config exposure deferred / driver-local budget
//!
//! [`LateWindowTakerParams`] is engine-local with a [`Default`] mirroring the
//! relevant `config::EngineParams` (`late_window_tau_secs` /
//! `late_certainty_threshold` / `late_taker_price_cap` / `taker_budget_per_window`
//! / `taker_cooldown_ms`); the `config::EngineParams → LateWindowTakerParams` map
//! at the `bot` boundary is **deferred** to the engine-bus-wiring task (the
//! [`MomentumTakerParams`](crate::MomentumTakerParams) precedent). The `config`
//! crate is untouched. `taker_budget_per_window` is documented "both modules
//! combined", but the late taker tracks its own budget locally for now — the
//! shared momentum+late reconciliation is deferred with the rest of the
//! orchestration wiring, so combined worst-case spend can reach 2× the budget
//! until then.

// The §8 endgame taker: see the module docs above for the anchor gate and tie rule.
mod driver;
mod edge;

use core_types::{AnchorSource, Decimal, Dollars, ModelHealth, ModelHealthReason, Price, WindowId};
use rust_decimal::dec;

pub use driver::LateWindowTaker;
pub use edge::{CertaintyTakePlan, plan_certainty_take};

/// Engine-local late-window-taker tunables (CLAUDE.md §8). `Copy`, but **not**
/// `Eq` (it holds an `f64` field).
///
/// All fields mirror `config::EngineParams` (`late_window_tau_secs` = 30,
/// `late_certainty_threshold` = 0.97, `late_taker_price_cap` = 0.99,
/// `taker_budget_per_window` = $10, `taker_cooldown_ms` = 5000). The
/// config→params map at the `bot` boundary is deferred (the
/// [`MomentumTakerParams`](crate::MomentumTakerParams) precedent), pinned by the
/// `defaults_mirror_config_engine_params` test without a `config` dependency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LateWindowTakerParams {
    /// The taker activates once seconds-remaining is at or below this.
    pub tau_threshold_secs: u32,
    /// `p_up` (Chainlink-anchored) must be beyond this certainty before taking.
    pub certainty_threshold: f64,
    /// Worst price the taker will pay for the near-certain side (the FAK
    /// slippage cap, and the per-level accept gate).
    pub price_cap: Price,
    /// Maximum cumulative taker notional per window (the binding spend cap).
    pub budget_per_window: Dollars,
    /// Minimum wall-time (ms) between takes — a cooldown after each fire.
    pub cooldown_ms: i64,
}

impl Default for LateWindowTakerParams {
    fn default() -> Self {
        Self {
            tau_threshold_secs: 30,
            certainty_threshold: 0.97,
            price_cap: default_price_cap(),
            budget_per_window: Dollars::new(Decimal::from(10)),
            cooldown_ms: 5_000,
        }
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "0.99 lies in (0,1) with scale 2; pinned by the defaults_mirror_config_engine_params test"
)]
fn default_price_cap() -> Price {
    Price::try_from(dec!(0.99)).unwrap()
}

/// Why the late-window taker declined to take this tick.
///
/// Engine-local and data-carrying, for §12 observability (the
/// [`NoTakeReason`](crate::NoTakeReason) / [`NoQuoteReason`](crate::NoQuoteReason)
/// precedent): the driver logs the reason at `debug` on every non-fire.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NoLateTakeReason {
    /// A risk breaker is holding the taker down.
    #[error("standing down on a risk breaker")]
    StandingDown,
    /// No window is open to trade.
    #[error("no active window")]
    NoActiveWindow,
    /// No model snapshot has arrived yet.
    #[error("no model snapshot yet")]
    NoModel,
    /// The model snapshot's window does not match the active window.
    #[error("model/active window mismatch")]
    WindowMismatch {
        /// The model snapshot's window.
        model: Option<WindowId>,
        /// The active window.
        active: WindowId,
    },
    /// The model is not [`ModelHealth::Ready`] — taking is gated to Ready only.
    #[error("model not ready: {health:?} ({reason:?})")]
    ModelNotReady {
        /// The model's health tier.
        health: ModelHealth,
        /// The cause behind that tier.
        reason: ModelHealthReason,
    },
    /// The fair was anchored on the fast feed, not the resolution-grade Chainlink
    /// feed — the late-window decision must rest on Chainlink only (§8).
    #[error("fair anchored on {anchor:?}, not Chainlink")]
    NotChainlinkAnchored {
        /// The snapshot's anchor source.
        anchor: AnchorSource,
    },
    /// The model fair is not a usable probability (not finite / not in `[0, 1]`).
    #[error("model fair {p_up} is not a usable probability")]
    UnusableFair {
        /// The offending fair.
        p_up: f64,
    },
    /// Time-to-expiry is not positive (the window is closing/closed).
    #[error("time to expiry {tau_secs} is not positive")]
    Expired {
        /// The offending seconds-remaining.
        tau_secs: f64,
    },
    /// The window is not yet in its late phase (`τ` still above the threshold).
    #[error("not late yet: {tau_secs}s remaining (> {threshold}s)")]
    NotLateWindow {
        /// Seconds remaining.
        tau_secs: f64,
        /// The configured late-window threshold (seconds).
        threshold: u32,
    },
    /// Neither side's probability is beyond the certainty threshold.
    #[error("not certain: p_up {p_up} (need ≥ {threshold} for Up or ≤ 1−{threshold} for Down)")]
    NotCertain {
        /// The model fair.
        p_up: f64,
        /// The configured certainty threshold.
        threshold: f64,
    },
    /// A take fired within the cooldown window.
    #[error("in cooldown ({remaining_ms} ms remaining)")]
    InCooldown {
        /// Milliseconds left on the cooldown.
        remaining_ms: i64,
    },
    /// The per-window taker budget is exhausted (less than the $1 minimum left).
    #[error("taker budget exhausted: {remaining} left, need {need}")]
    BudgetExhausted {
        /// Budget remaining for the window.
        remaining: Dollars,
        /// Minimum notional needed to take.
        need: Dollars,
    },
    /// No cached book for the candidate outcome's token yet.
    #[error("no book for the candidate token")]
    NoBookForToken,
    /// The cached book has no asks to lift.
    #[error("no asks to lift")]
    NoAsks,
    /// The best (lowest) ask is already above the price cap — nothing buyable
    /// within the cap.
    #[error("all asks above the price cap: best ask {best_ask} > cap {cap}")]
    AllAsksAbovePriceCap {
        /// The best (lowest) ask.
        best_ask: Price,
        /// The configured price cap.
        cap: Price,
    },
    /// The sized take's notional is below the $1 venue minimum.
    #[error("take notional {got} below the $1 minimum")]
    BelowMinNotional {
        /// The sub-minimum notional.
        got: Dollars,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_config_engine_params() {
        let p = LateWindowTakerParams::default();
        // Pinned to config/default.toml [engine.defaults]: late_window_tau_secs =
        // 30, late_certainty_threshold = 0.97, late_taker_price_cap = 0.99,
        // taker_budget_per_window = 10, taker_cooldown_ms = 5000. The
        // config→LateWindowTakerParams map at the bot boundary is deferred (the
        // MomentumTakerParams precedent), so this pins the values without a
        // `config` dependency.
        assert_eq!(p.tau_threshold_secs, 30);
        assert!((p.certainty_threshold - 0.97).abs() < f64::EPSILON);
        assert_eq!(p.price_cap.as_decimal(), dec!(0.99));
        assert_eq!(p.budget_per_window, Dollars::new(dec!(10)));
        assert_eq!(p.cooldown_ms, 5_000);
    }
}
