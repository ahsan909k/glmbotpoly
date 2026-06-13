//! Live execution adapter implementing the `venue-api` port over the official
//! Polymarket Rust SDK (`polymarket-client-sdk-v2`): order building/signing and
//! posting, batch operations, cancels, balances, and an order/fill event
//! stream.
//!
//! # Safety invariant (CLAUDE.md §11)
//!
//! A network-capable adapter can only be built through [`LiveVenue::connect`],
//! which fails closed unless **all three** arming gates pass: `live.enabled` in
//! config (gate 1) AND the env confirmation phrase (gate 2) AND the
//! dashboard-arm flag (gate 3). [`LiveVenue::dry_run`] performs real signing
//! against live market params but never posts an order.
//!
//! # Architecture
//!
//! The SDK is isolated behind the internal [`ClobPort`] seam (the network
//! layer). [`LiveVenue`] is generic over it, so the order-translation,
//! error-mapping, batch-chunking, and reconcile logic are all exercised offline
//! through [`FakeClobPort`] — which is exactly how the request-construction
//! tests prove every port method without touching the network.
//!
//! # Deferred to follow-up tasks
//!
//! - The real-time authenticated `/ws/user` WebSocket order/fill stream: this
//!   task ships an interim REST open-orders-poll reconciler ([`reconcile`]);
//!   the low-latency push replaces it behind the same [`VenueEvent`] channel.
//! - One-time on-chain ERC-20/CTF token approvals (the gasless relayer WALLET
//!   batch): a missing allowance surfaces here as
//!   [`RejectReason::InsufficientFunds`]; the approval flow belongs to the
//!   gasless/CTF task.
//!
//! [`VenueEvent`]: venue_api::VenueEvent
//! [`RejectReason::InsufficientFunds`]: venue_api::RejectReason::InsufficientFunds

mod arming;
mod convert;
mod error;
mod port;
mod reconcile;
mod sdk;
mod store;
mod venue;

#[cfg(any(test, feature = "fake"))]
pub mod fake;

mod params;

pub use convert::{BuiltOrder, OrderClass, build};
pub use error::{Gate, VenueLiveError};
pub use params::{
    DEFAULT_CLOB_HOST, DEFAULT_EVENT_CHANNEL_CAPACITY, LiveParams, POLYGON_CHAIN_ID, SigType,
};
pub use port::{ClobPort, RawAck, RawCancel, RawOpenOrder};
pub use reconcile::reconcile_loop;
pub use sdk::SdkClobPort;
pub use store::{OrderStore, TrackedOrder};
pub use venue::LiveVenue;

#[cfg(any(test, feature = "fake"))]
pub use fake::FakeClobPort;
