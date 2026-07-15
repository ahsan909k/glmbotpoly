//! The champion-model taker (CLAUDE.md §8): the live PAPER taker driven by the
//! `shadow` crate's per-window `p_up` stream (the dir10 model). It replicates the
//! tested walk-forward recipe (`model-lab/model_lab/money_judge.py::_run_taker`,
//! the dir10 @ θ = 0.03 configuration that scored +$2,284): fire a single FAK
//! marketable BUY of the model-predicted side, sized against displayed depth and a
//! per-window budget, with **no** edge/fee gate, **no** price cap, and **no**
//! cooldown (the ~5 s prediction cadence is the natural throttle). Holds to
//! resolution (the paper venue settles).
//!
//! ## Split into pure calculator + effectful driver
//!
//! Mirrors [`crate::taker`]/[`crate::late_window`]: everything testable is pure
//! and sans-IO; only [`driver`] crosses into IO.
//! - [`edge`] — [`plan_model_take`] → [`ModelTakePlan`]/[`NoModelTakeReason`], the
//!   depth/budget sizing (no gate, optional cap). Pure.
//! - [`driver`] — [`ModelTaker`], the async executor that owns **per-window**
//!   state (this is the one taker that trades all active windows concurrently,
//!   because the `shadow` model produces a prediction per window), the per-window
//!   budget, the risk veto, and fires FAKs through the [`venue_api::VenuePort`]
//!   (via the [`crate::normalize`] chokepoint). Logs via `tracing` (target
//!   `model-taker`).
//!
//! ## `p_up` semantics (important)
//!
//! The shadow model's `p_up` is **P(10 s-forward mid Up)**, not P(window outcome).
//! The paper trade still holds to resolution — this is exactly what the tested
//! recipe simulated. Never conflate it with the analytic `ModelSnapshot.p_up`
//! (the Black-Scholes window fair); this taker never reads that snapshot.

mod driver;
mod edge;

use core_types::{Decimal, Dollars, Outcome, Price, Series, Size, TimestampMs, WindowId};

pub use driver::ModelTaker;
pub use edge::{ModelTakePlan, plan_model_take};

/// One shadow prediction, as the model taker consumes it. The bot builds this
/// from `shadow::ShadowUpdate` (which carries `series`, `window_open_ms`, `ts`,
/// `p_up`, `finite_count`, and the model identity used to derive `model_stale`).
///
/// Engine-local so the engine never depends on the `shadow` crate (§4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrediction {
    /// Which series/window the prediction is for.
    pub series: Series,
    /// The window's open time in unix millis (its identity within the series).
    pub window_open_ms: i64,
    /// When the prediction was sampled (window-grid-aligned, ≈ now).
    pub ts: TimestampMs,
    /// P(10 s-forward mid Up) from the champion model.
    pub p_up: f64,
    /// Count of finite features (out of 24) the prediction was computed from —
    /// a coverage/freshness proxy.
    pub finite_count: u32,
    /// The champion model is past its refit-due age (a stand-down condition).
    pub model_stale: bool,
}

impl ModelPrediction {
    /// The window this prediction is for.
    #[must_use]
    pub fn window(&self) -> WindowId {
        WindowId {
            series: self.series,
            open_time: TimestampMs::from_millis(self.window_open_ms),
        }
    }
}

/// Engine-local model-taker tunables (CLAUDE.md §8). `Copy`, but **not** `Eq`
/// (it holds `f64`/`Option` fields).
///
/// Defaults mirror the tested recipe (θ = 0.03, $10/window, $1 min, no cap); the
/// `config::ModelTakerConfig → ModelTakerParams` map lives at the `bot` boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelTakerParams {
    /// Fire iff `|p_up − 0.5| ≥ theta`; the side is `Up` when `p_up` is high.
    pub theta: f64,
    /// Maximum cumulative taker notional per window (the binding spend cap).
    pub budget_per_window: Dollars,
    /// Minimum finite-feature count (out of 24) required to act on a prediction —
    /// keeps live fires aligned with the fully-featured offline model.
    pub min_finite_count: u32,
    /// Optional worst-price cap. `None` ⇒ no cap (the recipe: take at the
    /// displayed ask regardless of price).
    pub price_cap: Option<Price>,
    /// A book older than this (ms vs `now`) is not fresh → refuse (§9 conservative,
    /// mirrors the offline `BOOK_STALENESS_MS`).
    pub max_book_staleness_ms: i64,
}

impl Default for ModelTakerParams {
    fn default() -> Self {
        Self {
            theta: 0.03,
            budget_per_window: Dollars::new(Decimal::from(10)),
            min_finite_count: 24,
            price_cap: None,
            max_book_staleness_ms: 2_000,
        }
    }
}

/// The result of feeding one prediction to the taker — for §12 observability and
/// decision journaling (the bot writes it to the model-taker side channel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelTakeOutcome {
    /// A FAK was placed.
    Fired {
        /// Window traded.
        window: WindowId,
        /// Side bought.
        outcome: Outcome,
        /// Expected shares.
        shares: Size,
        /// Notional spent.
        notional: Dollars,
        /// Worst-price slippage cap on the FAK.
        worst_price: Price,
    },
    /// No fire, with the reason.
    Suppressed(NoModelTakeReason),
}

/// Why the model taker declined to act on a prediction. Engine-local and
/// data-carrying, for §12 observability (the [`NoTakeReason`](crate::NoTakeReason)
/// precedent).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NoModelTakeReason {
    /// A risk breaker is holding the taker down.
    #[error("standing down on a risk breaker")]
    StandingDown,
    /// No state for the prediction's window (not open, or not yet seen).
    #[error("no window state for the prediction's window")]
    NoWindowState,
    /// The champion model is past its refit-due age.
    #[error("shadow model is stale")]
    ModelStale,
    /// Too few finite features to trust the prediction.
    #[error("insufficient feature coverage: {finite}/{need}")]
    InsufficientCoverage {
        /// Finite features on this prediction.
        finite: u32,
        /// Minimum required.
        need: u32,
    },
    /// The model fair is not a usable probability (not strictly in `(0, 1)`).
    #[error("model fair {p_up} is not a usable probability")]
    UnusableFair {
        /// The offending fair.
        p_up: f64,
    },
    /// `|p_up − 0.5|` did not clear the θ threshold.
    #[error("below theta: |{p_up} − 0.5| < {theta}")]
    BelowTheta {
        /// The prediction.
        p_up: f64,
        /// The threshold.
        theta: f64,
    },
    /// Time-to-expiry is not positive (the window is closing/closed).
    #[error("time to expiry {tau_secs} is not positive")]
    Expired {
        /// The offending seconds-remaining.
        tau_secs: f64,
    },
    /// A higher-precedence driver owns this window right now (§8 fortress map).
    #[error("suppressed by arbitration: {winner:?} fired this window {remaining_ms} ms ago")]
    ArbitrationSuppressed {
        /// The winning driver for this window's asset.
        winner: crate::arbitration::TakerId,
        /// Milliseconds until the loser may fire.
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
    /// The cached book is older than the freshness bound.
    #[error("book is stale ({age_ms} ms old)")]
    BookStale {
        /// The book's age in ms.
        age_ms: i64,
    },
    /// The cached book has no asks to lift.
    #[error("no asks to lift")]
    NoAsks,
    /// The best ask is above the configured price cap.
    #[error("all asks above the price cap: best ask {best_ask} > cap {cap}")]
    AllAsksAbovePriceCap {
        /// The best (lowest) ask.
        best_ask: Price,
        /// The configured cap.
        cap: Price,
    },
    /// The sized take's notional is below the $1 venue minimum.
    #[error("take notional {got} below the $1 minimum")]
    BelowMinNotional {
        /// The sub-minimum notional.
        got: Dollars,
    },
    /// The venue / risk gate rejected the FAK at placement (e.g. a global halt,
    /// a per-window loss halt, or the open-notional cap — the §11 pre-trade
    /// veto). Recorded so budget/cap contention is visible.
    #[error("venue/gate rejected the place")]
    PlaceRejected,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::dec;

    #[test]
    fn defaults_mirror_the_recipe() {
        let p = ModelTakerParams::default();
        // Pinned to config/default.toml [model_taker]: theta 0.03, budget 10,
        // min_finite_count 24, no price cap, book staleness 2000 ms. The
        // config→ModelTakerParams map at the bot boundary is deferred (the
        // MomentumTakerParams precedent), so this pins the values config-free.
        assert!((p.theta - 0.03).abs() < f64::EPSILON);
        assert_eq!(p.budget_per_window, Dollars::new(dec!(10)));
        assert_eq!(p.min_finite_count, 24);
        assert_eq!(p.price_cap, None);
        assert_eq!(p.max_book_staleness_ms, 2_000);
    }

    #[test]
    fn window_maps_from_series_and_open_ms() {
        let pred = ModelPrediction {
            series: Series {
                asset: core_types::Asset::Eth,
                duration: core_types::WindowDuration::M15,
            },
            window_open_ms: 1_781_000_000_000,
            ts: TimestampMs::from_millis(1_781_000_010_000),
            p_up: 0.9,
            finite_count: 24,
            model_stale: false,
        };
        let w = pred.window();
        assert_eq!(w.series.asset, core_types::Asset::Eth);
        assert_eq!(w.open_time.as_millis(), 1_781_000_000_000);
    }
}
