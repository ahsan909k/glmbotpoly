//! Polymarket CLOB market-channel WebSocket client and local L2 order-book
//! maintenance: applies `book` snapshots and `price_change` deltas, tracks
//! `tick_size_change` flips (0.01 <-> 0.001 past 0.96/0.04), and surfaces
//! `last_trade_price`, `best_bid_ask`, `new_market`, and `market_resolved`
//! events (custom feature flag enabled — CLAUDE.md §7). Provides the
//! displayed-depth view consumed by the engine, the paper fill simulator,
//! and the dashboard ladder, with integrity checks (crossed books, venue-top
//! divergence, active-window staleness) and resync-by-reconnect.
//!
//! Topology (§5 + Decisions Log): **one WebSocket connection per window
//! (token pair)**, never one big subscription — 12 tokens on one connection
//! floods ~1500 msg/s and the server RSTs it (live-verified). The supervisor
//! consumes scheduler window-lifecycle announcements, pre-connects the next
//! window before the current closes (gap-free rollover by construction),
//! lets a closed window's connection linger for its `market_resolved`, and
//! forwards deduplicated [`core_types::MarketLifecycleEvent`]s to the
//! scheduler.

pub mod book;
pub mod machine;
pub mod supervisor;
pub mod wire;

mod driver;

pub use book::{BookError, BookState, ChangeOutcome};
pub use driver::WindowFeedStatus;
pub use feed_util::{
    Backoff, BackoffParams, TapDir, TapFrame, TapTransport, Transport, TransportError, WsFrame,
    WsTransport,
};
pub use machine::{ConnMachine, ConnPhase, MachineOutput, MachineParams, MachineStatus};
pub use supervisor::{ClobArgs, ClobCommand, ClobStatus, run};
pub use wire::{ClobEvent, IgnoredReason, ParsedFrame, parse_frame, subscribe_message};

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).

/// Fatal feed errors (re-export of the shared type: only a dead bus or a
/// dead control channel can stop the supervisor; connection trouble never
/// does).
pub use feed_util::FeedError;

/// Connection parameters (mapped from `feeds.*` config by the binary — this
/// crate depends only on core-types and feed-util).
#[derive(Debug, Clone, PartialEq)]
pub struct ClobParams {
    /// Market-channel endpoint, e.g.
    /// `wss://ws-subscriptions-clob.polymarket.com/ws/market`.
    pub url: String,
    /// Keepalive PING interval (config validates (0, 10000]).
    pub ping_interval: std::time::Duration,
    /// TCP+TLS+WS handshake deadline.
    pub connect_timeout: std::time::Duration,
    /// Reconnect backoff curve.
    pub backoff: BackoffParams,
    /// Active-window book silence past this is desync evidence (machine
    /// adapts it upward when a recycle proves the market merely quiet).
    pub book_stale_after: core_types::DurationMs,
    /// How long before a window's open its connection is established.
    pub presubscribe_lead: core_types::DurationMs,
    /// Coalescing interval for delta-driven full-book publication.
    pub publish_interval: core_types::DurationMs,
    /// How long a closed window's connection lingers for `market_resolved`.
    pub resolution_linger: core_types::DurationMs,
}
