//! The 24/7 autonomy core: one lifecycle state machine per series driving
//! Discovering -> Pending -> Active -> Closing -> Resolved -> roll-to-next,
//! forever, with no human action (CLAUDE.md §6).
//!
//! Three redundant inputs drive transitions: **the wall clock** against
//! discovery-supplied window times (the deterministic roll driver — windows
//! are contiguous back-to-back, so the roll must never wait on anything
//! slower), **discovery refresh** (the metadata authority), and **market-
//! channel events** (`new_market` accelerates refresh, `market_resolved`
//! supplies the outcome for the already-rolled window). A missed event can
//! never stall a series; a series with no discoverable next window logs
//! loudly, retries with backoff, and never crashes the process.
//!
//! Layering: [`SeriesMachine`] is the sans-IO core — clock-free, IO-free,
//! log-free, fully deterministic under test; [`run`] is the thin async driver
//! that owns all six machines, sleeps until the earliest deadline, runs
//! discovery HTTP in a worker task so it can never delay a wall-clock roll,
//! and forwards announcements to the internal bus as
//! [`core_types::Event::Window`].

mod backoff;
mod driver;
mod error;
mod machine;
mod status;
#[cfg(test)]
mod testutil;

pub use driver::{Refresh, SchedulerArgs, run};
pub use error::SchedulerError;
pub use machine::{CoverageWarning, Output, RefreshReason, SeriesMachine, Timing};
pub use status::{SchedulerStatus, SeriesStatus};
