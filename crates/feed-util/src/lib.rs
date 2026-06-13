//! Shared plumbing for the price-feed crates (feed-rtds, feed-binance): the
//! WebSocket [`Transport`] seam (real tungstenite impl + the `--raw` tap
//! decorator), the jittered reconnect [`BackoffParams`], the sans-IO
//! per-stream staleness machine, and the generic [`run`] driver loop —
//! connect → subscribe → stream → reconnect, with a dead-socket watchdog
//! and starvation-recycle self-healing — parameterized by a venue
//! [`FeedProtocol`].
//!
//! Extracted verbatim from feed-rtds (Decisions Log 2026-06-12) so
//! feed-binance shares the live-debugged reconnect/staleness machinery
//! instead of duplicating it. This crate knows nothing about any venue's
//! wire format: message building and frame parsing stay in the feed crates.

mod backoff;
mod driver;
mod error;
#[cfg(feature = "fake")]
pub mod fake;
mod machine;
mod transport;

pub use backoff::{Backoff, BackoffParams};
pub use driver::{
    CommandAction, ConnState, DriverArgs, DriverParams, FeedProtocol, FeedStatus, FrameOutcome,
    Keepalive, PriceObs, run,
};
pub use error::FeedError;
pub use machine::{KeyStatus, StreamKey};
pub use transport::{
    Connection, TapDir, TapFrame, TapTransport, Transport, TransportError, WsFrame, WsTransport,
};
