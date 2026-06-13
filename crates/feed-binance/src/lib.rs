//! Direct Binance spot WebSocket feed — the lowest-latency underlying signal
//! (CLAUDE.md §8: corrected by a continuously-estimated basis against the
//! resolution-grade Chainlink feed, owned by the `model` crate).
//!
//! Streams BTCUSDT/ETHUSDT top-of-book (`bookTicker`, published as the
//! midpoint, [`core_types::TickKind::Mid`]) and trade prints (`trade`,
//! [`core_types::TickKind::Trade`]), all tagged
//! [`core_types::PriceSource::BinanceDirect`]. Ticks publish as
//! [`core_types::Event::PriceTick`], health transitions as
//! [`core_types::Event::FeedHealth`].
//!
//! Layering (house pattern): [`wire`](crate::parse_frame) is pure protocol
//! (URL builder + lenient parser, never panics); the connection lifecycle —
//! reconnect with jittered backoff, per-stream staleness watchdog,
//! dead-socket teardown, starvation recycle — is feed-util's generic driver,
//! parameterized here with the Binance venue protocol behind the
//! [`Transport`] seam so tests script a fake.
//!
//! Venue constraints, satisfied by construction: subscription rides the
//! connect URL (combined-stream form), so the client sends **zero** frames
//! after the handshake (the 5-inbound-messages/s limit can't be hit); the
//! server's 20-second protocol pings are answered by the transport and
//! surface as liveness; the 24-hour connection lifetime lands on the
//! ordinary reconnect path.
//!
//! The [`compare`] module is the sans-IO feed-comparison diagnostic (lead/lag
//! and value basis between this feed and the two RTDS feeds) behind
//! `bot compare` — measurement tooling, not an engine path; the production
//! basis corrector lives in the `model` crate (§4).

pub mod compare;
mod driver;
mod protocol;
mod sub;
mod wire;

pub use driver::{BinanceArgs, BinanceParams, run};
// The shared plumbing types come from feed-util; re-exported so consumers
// keep a single import path (feed-rtds precedent).
pub use feed_util::{
    BackoffParams, ConnState, Connection, FeedError, TapDir, TapFrame, TapTransport, Transport,
    TransportError, WsFrame, WsTransport,
};
pub use protocol::NoCommand;
pub use sub::{BinanceStream, BinanceSub};
pub use wire::{
    BinanceParsed, BinanceUpdate, IgnoredReason, combined_url, map_stream, map_symbol, parse_frame,
};

/// Per-stream status, keyed by the Binance stream type.
pub type KeyStatus = feed_util::KeyStatus<BinanceSub>;

/// Point-in-time feed snapshot, keyed by the Binance stream type.
pub type FeedStatus = feed_util::FeedStatus<BinanceSub>;
