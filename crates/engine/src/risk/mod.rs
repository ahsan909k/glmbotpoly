//! The risk manager (CLAUDE.md §5/§11): the **single gateway** between every
//! strategy module and the venue port — no order reaches a venue without passing
//! its checks. Enforced by module visibility: the strategy drivers
//! ([`quote_manager`](crate::quote_manager) / [`taker`](crate::taker) /
//! [`late_window`](crate::late_window)) are `pub(crate)` and their driver structs
//! are not re-exported, so the only `pub` path from outside `engine` to the venue
//! is [`RiskManager`], which always interposes the [`GuardedPort`](guard::GuardedPort).
//!
//! ## Split into pure core + effectful driver (the codebase pattern)
//!
//! - [`core`] — [`RiskCore`](core::RiskCore), the sans-IO §11 breaker state
//!   machine. Inputs in with `now`, decisions out as [`RiskOutput`]s. The single
//!   writer of breaker semantics: feeds report, risk reacts.
//! - [`state`] — [`GateState`](state::GateState), the small lock-protected state
//!   the guard reads to admit/refuse a placement and writes back venue-error
//!   observations to; and [`RiskStateSnapshot`], the dashboard projection.
//! - [`guard`] — [`GuardedPort`](guard::GuardedPort), the `VenuePort` the
//!   strategies are handed (brief, `await`-free locks; cancels always pass).
//! - [`driver`] — [`RiskManager`], the async executor that owns the core, the
//!   gate, and the three strategies; publishes breaker events for the dashboard
//!   and journal; logs every trip with cause via `tracing` target `"risk"`.
//!
//! ## Config exposure deferred
//!
//! [`RiskParams`] is engine-local with a [`Default`] mirroring `config`'s
//! `[risk]` section; the `config → RiskParams` map (and the per-series cap map)
//! lives at the `bot` boundary, exactly as the other engine params deferred
//! theirs. The `config` crate is never a dependency of `engine`.

mod core;
mod driver;
mod guard;
mod params;
mod state;

pub use core::RiskOutput;
pub use driver::RiskManager;
pub use params::RiskParams;
pub use state::RiskStateSnapshot;
