//! Paper execution adapter implementing the `venue-api` port against real
//! live market data (CLAUDE.md §9): a conservative fill simulator that queues
//! passive orders behind all displayed size at placement, fills only on real
//! trade prints at or through the price after simulated latencies, and walks
//! real displayed depth for marketable orders; plus the paper wallet/ledger
//! with runtime-adjustable starting capital, exact taker-fee math from live
//! per-market fee params, and an estimated maker-rebate accumulator. It must be
//! impossible for paper to be more optimistic than reality in any code path —
//! when in doubt, fill less.
//!
//! # Architecture
//!
//! Mirrors `venue-live`'s split: a sans-IO state machine ([`MatchEngine`]) whose
//! pure, clock-injected mutators return [`PaperEffect`]s, wrapped by an
//! `Arc<Mutex<…>>` async adapter ([`PaperVenue`]) that publishes those effects
//! as [`venue_api::VenueEvent`]s. A spawned timer task drives the
//! simulated-latency deadlines. Market data is pushed in via
//! [`PaperVenue::on_bus_event`].
//!
//! # Scope
//!
//! The **fill simulator** plus a conservation-correct paper ledger: collateral
//! cash-flow and signed positions on every fill, exact taker fees, settlement on
//! the real `market_resolved` outcome ($1 per winning share), instant pair-merge
//! (matched Up+Down → $1 collateral), the daily maker-rebate credit cycle, and a
//! runtime capital set/adjust seam. Still deferred: open-order collateral
//! reservation (so `collateral_available == collateral_total`), promoting
//! `merge` to a venue-agnostic port (with the live CTF on-chain merge), and
//! durable journaling of ledger events (the `journal` crate's job).

mod book;
mod driver;
mod engine;
mod latency;
mod params;
mod wallet;

pub use book::{MarketableWalk, SimBook};
pub use driver::PaperVenue;
pub use engine::{MatchEngine, PaperEffect};
pub use latency::LatencySampler;
pub use params::{LatencySpec, PaperParams, PaperParamsError};
pub use wallet::{PaperLedgerSnapshot, PaperWallet};
