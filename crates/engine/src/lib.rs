//! The per-series trading engine (CLAUDE.md §8): inventory and pair-cost
//! accounting, the pure quoting calculator (inventory-skewed center,
//! vol-and-latency-aware half-spread, post-only ladders, late-window gates),
//! the cancel-first quote manager that defends against adverse selection,
//! the fee-aware momentum and late-window taker modules, the order
//! normalizer enforcing tick/size/notional minimums, and the risk manager
//! through which every order must pass — no order reaches a venue without
//! its approval (CLAUDE.md §5, §11). Depends only on `model` and the
//! `venue-api` port, never on a concrete venue.
//!
//! [`normalize::normalize`] is the single pure, sans-IO chokepoint every
//! outgoing order intent passes through before it can reach a venue: it snaps
//! the price to the market's current tick grid, enforces the tick/size/notional
//! minimums, verifies a post-only order would not cross our book view, and
//! returns either a venue-legal [`NewOrder`](core_types::NewOrder) or a typed
//! [`normalize::NormalizeReject`].
//!
//! [`inventory`] holds the per-window share/cost accounting (CLAUDE.md §8):
//! [`InventoryManager`] folds `Fill` and `Window` bus events into one
//! [`WindowInventory`] per window — deriving matched pairs, pair cost, signed
//! excess, and worst-case-if-excess-loses — and answers the pure §8 policy
//! queries (pair-cost discipline, soft/hard excess caps, the merge request) plus
//! settlement on resolution. It is sans-IO and event-time-driven, so a journal
//! replay rebuilds it exactly.

pub mod inventory;
pub mod normalize;
pub mod quote_manager;
pub mod quoting;

pub use inventory::{
    ExcessConstraint, InventoryEffect, InventoryManager, InventoryParams, MergeIntent,
    WindowInventory, authorizes_passive_add_sides, excess_constraint_sides,
    pair_cost_after_add_sides,
};
pub use normalize::{
    Adjustments, NormalizeReject, Normalized, NormalizerParams, OrderDraft, PriceSnap, normalize,
};
pub use quote_manager::{
    CancelAction, ClientId, ConvergeRationale, ConvergencePlan, ConvergencePlanner, DesiredPlace,
    QuoteManager, QuoteManagerParams, RestingOrder, RestingView,
};
pub use quoting::{
    NoQuoteReason, QuoteDecision, QuoteLevel, QuoteParams, QuoteSet, SuppressReason, Suppressed,
    calculate_quotes,
};
