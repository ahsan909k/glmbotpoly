//! The cancel-first quote manager (CLAUDE.md §8): owns convergence from the
//! current open orders to the [quoting calculator](crate::quoting)'s desired set,
//! through the [`venue_api::VenuePort`]. It is the component that turns a
//! *desired* quote set into *actual* resting orders and keeps them converged as
//! the market moves — the primary defence against adverse selection (resting
//! quotes picked off when the underlying moves).
//!
//! ## Split into pure planner + effectful driver
//!
//! Everything testable in isolation is pure and sans-IO; only the driver crosses
//! into IO:
//! - [`client_id`] — the [`ClientId`] attribution scheme (encode/parse). Pure.
//! - [`view`] — [`RestingView`]/[`RestingOrder`], the engine-side open-order view
//!   (the `engine` crate cannot depend on `venue-live`'s `OrderStore`, §4, so the
//!   manager keeps its own view, folded from placements + venue events). Pure.
//! - [`plan`] — [`ConvergencePlanner`] → [`ConvergencePlan`], the cancel-first
//!   diff (which orders to cancel, which levels to place). Pure.
//! - [`driver`] — [`QuoteManager`], the async executor that owns the view, an
//!   [`InventoryManager`](crate::InventoryManager), debounce/rate-budget state,
//!   and drives the port. The **one** module that `.await`s the venue and logs
//!   via `tracing` (the `feed-clob` precedent); the pure modules above stay
//!   tracing-free.
//!
//! ## Split-cycle (strict) cancel-first
//!
//! A stale or adversely-offside quote is cancelled immediately; its replacement
//! is placed only on a *later* converge, once the cancel has terminalized and the
//! slot is empty — so the old and the new never rest on the same side at once
//! (CLAUDE.md §8, operator decision). The planner therefore never emits both a
//! cancel and a place for the *same* slot in one step; placements land only into
//! empty slots.
//!
//! ## Config exposure deferred
//!
//! [`QuoteManagerParams`] is engine-local with a [`Default`] mirroring the
//! relevant `config::EngineParams` (`reprice_threshold_theta` / `cancel_market_theta`)
//! plus the rate-budget knobs that have no `config` field yet; the
//! `config::EngineParams → QuoteManagerParams` map at the `bot` boundary is
//! **deferred** to the engine-bus-wiring task, exactly as
//! [`NormalizerParams`](crate::NormalizerParams)/[`InventoryParams`](crate::InventoryParams)
//! deferred theirs. The `config` crate is untouched.

mod client_id;
mod driver;
mod plan;
mod view;

pub use client_id::ClientId;
pub use driver::QuoteManager;
pub use plan::{
    CancelAction, ConvergeRationale, ConvergencePlan, ConvergencePlanner, DesiredPlace,
};
pub use view::{RestingLookup, RestingOrder, RestingView};

/// Engine-local quote-manager tunables (CLAUDE.md §8). `Copy`, but **not** `Eq`
/// (it holds `f64` thresholds).
///
/// The first two fields mirror `config::EngineParams` (`reprice_threshold_theta`
/// = 0.005, `cancel_market_theta` = 0.02); the rate-budget knobs have no `config`
/// field yet (config exposure deferred to the `bot` boundary, like the
/// `cancel_rtt_secs`/`atm_band` knobs in [`QuoteParams`](crate::QuoteParams)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteManagerParams {
    /// θ — the repricing / adverse-move threshold in probability space. A resting
    /// quote whose desired price has moved by more than this is repriced; a model
    /// fair that moves more than this against a resting quote is cancelled
    /// immediately (the urgent fast path).
    pub reprice_threshold_theta: f64,
    /// A model-fair move larger than this (probability space) escalates to a bulk
    /// `cancel_market` instead of per-order cancels — a big jump moves every level.
    pub cancel_market_theta: f64,
    /// The per-window rate budget: minimum wall-time (ms) between *placement*
    /// cycles. Cancels are never throttled (a cancel is always a safe action);
    /// only re-placements are rate-limited so model/book churn can't blow the
    /// venue's request budget.
    pub min_requote_interval_ms: i64,
    /// Maximum orders handed to one `place_batch`. The port chunks at its
    /// documented 15-order limit internally; this is a defensive cap so a single
    /// converge never exceeds it (a two-sided 3-level ladder is 6).
    pub max_batch: usize,
    /// Per-window cumulative maker deployment budget (dollars of buy notional
    /// filled). Once a window's cumulative maker buy fills reach this, the
    /// manager stops OPENING new buys for that window (cancels still flow).
    /// `ZERO` (default) = unbounded — the pre-budget behavior.
    pub maker_deployment_budget_per_window: core_types::Dollars,
}

impl Default for QuoteManagerParams {
    fn default() -> Self {
        Self {
            reprice_threshold_theta: 0.005,
            cancel_market_theta: 0.02,
            min_requote_interval_ms: 250,
            max_batch: 15,
            maker_deployment_budget_per_window: core_types::Dollars::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuoteManagerParams;

    #[test]
    fn defaults_mirror_config_engine_params() {
        let p = QuoteManagerParams::default();
        // Pinned to config/default.toml [engine.defaults]: reprice_threshold_theta
        // = 0.005, cancel_market_theta = 0.02. The config→QuoteManagerParams map
        // at the bot boundary is deferred (NormalizerParams/InventoryParams
        // precedent), so this pins the values without a `config` dependency.
        assert!((p.reprice_threshold_theta - 0.005).abs() < f64::EPSILON);
        assert!((p.cancel_market_theta - 0.02).abs() < f64::EPSILON);
        assert_eq!(p.max_batch, 15);
        assert!(p.min_requote_interval_ms > 0);
    }
}
