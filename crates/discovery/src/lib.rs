//! Market discovery over the Gamma and CLOB REST APIs for the six Up/Down
//! series (BTC/ETH x 5m/15m/1h): finds the current and next window per
//! series, and caches per-market metadata the rest of the bot must never
//! hardcode — token ids, condition ids, tick size, minimum order size, fee
//! parameters, neg-risk flag, and the resolution source parsed from the
//! market rules text (CLAUDE.md §6–7). Discovery must always hold the active
//! window plus at least the next one, fetched at least 60 seconds before the
//! current window ends.
//!
//! Layering: [`wire`] mirrors the APIs permissively, [`map`] turns wire data
//! into validated [`core_types::MarketInfo`] values (fail-closed), [`client`]
//! is the REST seam (trait + reqwest implementation), and [`service`] holds
//! the per-series cache the scheduler drives. The service never reads
//! clocks — `now` is always a parameter.

pub mod client;
pub mod error;
pub mod map;
pub mod service;
pub mod wire;

pub use client::{DiscoveryApi, HttpClient};
pub use error::{DiscoveryError, MapError};
pub use service::{DiscoveryService, DiscoverySnapshot, SeriesWindows, classify_windows};

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).
