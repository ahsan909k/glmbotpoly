//! The fee-aware momentum taker (CLAUDE.md §8): the offensive complement to the
//! defensive cancel-first quoter. When a fresh move on the fast signal feed
//! (direct Binance) implies the Polymarket book has not yet repriced, and the
//! model-vs-book edge — after the exact per-market taker fee plus a buffer — is
//! still positive, it fires a single FAK marketable BUY to pick off the lagging
//! book, sized against displayed depth and the per-window taker budget.
//!
//! ## Split into pure calculators + effectful driver
//!
//! Mirrors [`crate::quote_manager`]: everything testable in isolation is pure
//! and sans-IO; only the driver crosses into IO.
//! - [`signal`] — [`SignalWindow`]/[`ConfirmedMove`], the confirmed-move
//!   detector (signed return over a short lookback vs a noise multiple of σ).
//!   Pure.
//! - [`edge`] — [`plan_take`] → [`TakePlan`]/[`NoTakeReason`], the fee-aware edge
//!   walk and depth/budget sizing. Pure.
//! - [`driver`] — [`MomentumTaker`], the async executor that owns the per-asset
//!   signal rings, the cached active book, the per-window budget/cooldown state,
//!   and the risk veto, and fires FAKs through the [`venue_api::VenuePort`] (via
//!   the [`crate::normalize`] chokepoint). The **one** module that `.await`s the
//!   venue and logs via `tracing` (target `momentum-taker`); the pure modules
//!   above stay tracing-free.
//!
//! ## Config exposure deferred
//!
//! [`MomentumTakerParams`] is engine-local with a [`Default`] mirroring the
//! relevant `config::EngineParams` (`taker_momentum_buffer` /
//! `taker_budget_per_window` / `taker_cooldown_ms`) plus the signal-detection
//! knobs that have no `config` field yet; the `config::EngineParams →
//! MomentumTakerParams` map at the `bot` boundary is **deferred** to the
//! engine-bus-wiring task, exactly as
//! [`QuoteParams`](crate::QuoteParams)/[`QuoteManagerParams`](crate::QuoteManagerParams)/[`NormalizerParams`](crate::NormalizerParams)
//! deferred theirs. The `config` crate is untouched.
//!
//! ## No τ floor (operator decision)
//!
//! The momentum taker is purely signal/edge/budget/cooldown/risk driven; takes
//! are allowed right up to resolution (only a defensive τ > 0 expiry guard
//! remains, and the window lifecycle clears the active window on `Closing`). The
//! edge gate (fee + buffer) is the sole profitability guard. The near-certainty
//! endgame is a separate, future late-window taker.

mod driver;
mod edge;
mod signal;

use core_types::{Decimal, Dollars, ModelHealth, ModelHealthReason, Price, WindowId};
use rust_decimal::dec;

pub use driver::MomentumTaker;
pub use edge::{TakePlan, plan_take};
pub use signal::{ConfirmedMove, SignalWindow};

/// Engine-local momentum-taker tunables (CLAUDE.md §8). `Copy`, but **not** `Eq`
/// (it holds `f64` fields).
///
/// The first three fields mirror `config::EngineParams` (`taker_momentum_buffer`
/// = 0.005, `taker_budget_per_window` = $10, `taker_cooldown_ms` = 5000); the
/// signal-detection knobs and the rebate have no `config` field yet (config
/// exposure deferred to the `bot` boundary, like the `cancel_rtt_secs`/`atm_band`
/// knobs in [`QuoteParams`](crate::QuoteParams)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentumTakerParams {
    /// Edge required *beyond* the taker fee before taking, in probability/$
    /// space per share (§8 "fee(p) + configured buffer").
    pub momentum_buffer: Decimal,
    /// Maximum cumulative taker notional per window (the binding spend cap).
    pub budget_per_window: Dollars,
    /// Minimum wall-time (ms) between takes — a cooldown after each fire.
    pub cooldown_ms: i64,
    /// Lookback (ms) over which the signal-feed return is measured.
    pub signal_lookback_ms: i64,
    /// A move is "confirmed" when `|return| > signal_sigma_mult · σ_1s · √T` —
    /// the noise multiple of current volatility.
    pub signal_sigma_mult: f64,
    /// Expected taker rebate as a fraction of the fee, discounting the fee in the
    /// *decision* edge (expected net cost).
    ///
    /// **Defaults to `0.0` — immaterial at our volume.** The Taker Rebate Program
    /// (docs "Taker Rebates") is tiered on 30-day *weighted* volume
    /// `wV = size·(1 − entry)·weight·bonus`: Bronze $2k → 3%, Silver $20k → 8%,
    /// Gold $200k → 18%, Platinum $1M → 32%, Diamond $4M → 44%, Obsidian $10M →
    /// 50%. It is paid **once daily at 00:00 UTC in pUSD**, *not* deducted at
    /// execution. At our scale (tens-to-low-hundreds $/window) we never reach
    /// Bronze, so the rebate is 0% and the edge gate uses the full fee. The math
    /// is wired (`effective_fee = taker_fee · (1 − rebate_pct)` in
    /// [`plan_take`]) so that if the operator ever qualifies, setting this to the
    /// tier fraction correctly lowers the take threshold; `TakePlan::expected_fee`
    /// stays the gross fee the venue charges (the rebate is a separate accrual).
    pub taker_rebate_pct: Decimal,
}

impl Default for MomentumTakerParams {
    fn default() -> Self {
        Self {
            momentum_buffer: dec!(0.005),
            budget_per_window: Dollars::new(Decimal::from(10)),
            cooldown_ms: 5_000,
            signal_lookback_ms: 2_000,
            signal_sigma_mult: 3.0,
            taker_rebate_pct: Decimal::ZERO,
        }
    }
}

/// Why the momentum taker declined to take this tick.
///
/// Engine-local and data-carrying, for §12 observability (the
/// [`NoQuoteReason`](crate::NoQuoteReason) precedent): the driver logs the reason
/// at `debug` on every non-fire, so a quiet taker is always explainable.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NoTakeReason {
    /// A risk breaker is holding the taker down.
    #[error("standing down on a risk breaker")]
    StandingDown,
    /// No window is open to trade.
    #[error("no active window")]
    NoActiveWindow,
    /// No model snapshot has arrived yet for the active window.
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
    /// Time-to-expiry is not positive (the window is closing/closed).
    #[error("time to expiry {tau_secs} is not positive")]
    Expired {
        /// The offending seconds-remaining.
        tau_secs: f64,
    },
    /// The model fair is not a usable probability (not strictly in `(0, 1)`).
    #[error("model fair {p_up} is not a usable probability")]
    UnusableFair {
        /// The offending fair.
        p_up: f64,
    },
    /// The model `σ_1s` is not a usable volatility (not positive/finite) — needed
    /// for the confirmed-move threshold.
    #[error("sigma_1s {sigma_1s} is not a usable volatility")]
    UnusableVol {
        /// The offending `σ_1s`.
        sigma_1s: f64,
    },
    /// No confirmed move on the signal feed this tick.
    #[error("no confirmed signal move")]
    NoConfirmedMove,
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
    /// The best ask is already at or above fair — the book repriced; nothing
    /// stale to pick off (distinct from a fee-eaten edge).
    #[error("book already repriced: best ask {best_ask} ≥ fair {fair}")]
    BookAlreadyRepriced {
        /// The best (lowest) ask.
        best_ask: Price,
        /// The model fair for the candidate outcome.
        fair: Decimal,
    },
    /// There was positive raw edge, but it did not clear the fee plus buffer
    /// (per share, on the best level).
    #[error("edge {edge} below fee {fee} + buffer {buffer}")]
    EdgeBelowFeePlusBuffer {
        /// Best-level per-share raw edge (`fair − best_ask`).
        edge: Decimal,
        /// Best-level per-share effective (rebate-discounted) fee.
        fee: Decimal,
        /// The configured buffer.
        buffer: Decimal,
    },
    /// The sized take's notional is below the $1 venue minimum.
    #[error("take notional {got} below the $1 minimum")]
    BelowMinNotional {
        /// The sub-minimum notional.
        got: Dollars,
    },
    /// A higher-precedence driver owns this window right now (§8 fortress map).
    #[error("suppressed by arbitration: {winner:?} fired this window {remaining_ms} ms ago")]
    ArbitrationSuppressed {
        /// The winning driver for this window's asset.
        winner: crate::arbitration::TakerId,
        /// Milliseconds until the loser may fire.
        remaining_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_config_engine_params() {
        let p = MomentumTakerParams::default();
        // Pinned to config/default.toml [engine.defaults]: taker_momentum_buffer
        // = 0.005, taker_budget_per_window = 10, taker_cooldown_ms = 5000. The
        // config→MomentumTakerParams map at the bot boundary is deferred (the
        // QuoteParams/InventoryParams precedent), so this pins the values without
        // a `config` dependency.
        assert_eq!(p.momentum_buffer, dec!(0.005));
        assert_eq!(p.budget_per_window, Dollars::new(dec!(10)));
        assert_eq!(p.cooldown_ms, 5_000);
        // Signal knobs and rebate are engine-local defaults (no config key yet).
        assert_eq!(p.signal_lookback_ms, 2_000);
        assert!((p.signal_sigma_mult - 3.0).abs() < f64::EPSILON);
        assert_eq!(p.taker_rebate_pct, Decimal::ZERO);
    }
}
