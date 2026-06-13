//! Client for Polymarket's Real-Time Data Socket (wss://ws-live-data.polymarket.com):
//! subscribes to the `crypto_prices` (Binance-sourced) and
//! `crypto_prices_chainlink` (Chainlink-sourced — the resolution-grade feed
//! for these markets) topics, maintains the 5-second PING keepalive, and
//! publishes normalized price events with staleness tracking. Designed so a
//! direct Chainlink Data Streams source can be slotted in later without
//! touching consumers (CLAUDE.md §7 RTDS notes).
//!
//! Layering (house pattern): [`wire`](crate::parse_frame) is pure protocol
//! (message builders + lenient parser, never panics); the connection
//! lifecycle — reconnect with jittered backoff, staleness watchdog,
//! dead-socket teardown, starvation recycle — is feed-util's generic driver,
//! which this crate parameterizes with its venue protocol (resubscribe
//! sequence, frame classification, runtime subscription commands over a
//! command channel) behind the [`Transport`] seam so tests script a fake.
//! Ticks publish as [`core_types::Event::PriceTick`], health transitions as
//! [`core_types::Event::FeedHealth`].

mod driver;
mod sub;
mod wire;

pub use driver::{RtdsArgs, RtdsParams, run};
// The shared plumbing types come from feed-util; re-exported so consumers
// (and the pre-extraction API) keep a single import path.
pub use feed_util::{
    BackoffParams, ConnState, Connection, FeedError, TapDir, TapFrame, TapTransport, Transport,
    TransportError, WsFrame, WsTransport,
};
pub use sub::{FeedCommand, FeedSub};
pub use wire::{
    IgnoredReason, ParsedFrame, PriceUpdate, RtdsSource, backfill_subscribe_message, map_symbol,
    parse_frame, stream_subscribe_message, stream_unsubscribe_message,
};

/// Per-stream status, keyed by the RTDS stream type.
pub type KeyStatus = feed_util::KeyStatus<FeedSub>;

/// Point-in-time feed snapshot, keyed by the RTDS stream type.
pub type FeedStatus = feed_util::FeedStatus<FeedSub>;
