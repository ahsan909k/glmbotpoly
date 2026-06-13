//! The execution port (CLAUDE.md §4): venue-agnostic traits and types for
//! placing and cancelling orders (place, cancel, cancel-market, cancel-all),
//! querying balances, and consuming the venue's order/fill event stream.
//! Strategy and engine code depend only on this crate — never on a concrete
//! adapter — which is what guarantees the paper and live engine paths are
//! 100% identical (CLAUDE.md §2.5). Mode differences live exclusively in the
//! adapters implementing this port.
//!
//! Layout:
//! - [`VenuePort`]/[`VenueEvents`] — the trait surface (commands + the
//!   order/fill stream).
//! - [`Accepted`], [`BatchPlaced`], [`CancelReport`], [`Wallet`],
//!   [`VenueEvent`] — engine-facing result/event types, all built from
//!   `core_types`.
//! - [`VenueError`]/[`RejectReason`] — the venue-agnostic error vocabulary:
//!   no adapter or SDK error type ever leaks through this surface, and the
//!   retry/restart variants let the risk manager's backoff and
//!   matching-engine-restart logic (§11) stay venue-independent.

mod error;
mod port;
mod types;

pub use error::{RejectReason, TradingDisabledMode, VenueError};
pub use port::{VenueEvents, VenuePort};
pub use types::{
    Accepted, BatchPlaced, CancelReport, NotCanceled, PlaceRejection, TokenBalance, VenueEvent,
    Wallet,
};
