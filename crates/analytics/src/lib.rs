//! Performance measurement over the journal: signed markouts on passive
//! fills (the adverse-selection detector — persistently negative markouts
//! mean our resting quotes are being picked off), PnL attribution split into
//! locked-pair PnL vs inventory PnL, fees paid and rebates earned, and
//! per-series aggregates (windows traded, hit rate, PnL per window, fill
//! counts, minimum-sample warnings) that power the dashboard's series
//! comparison table — the view the operator uses to narrow six series down
//! to the best two or three (CLAUDE.md §10).
//!
//! # Architecture
//!
//! The core is the sans-IO [`Analytics`] fold, modelled on
//! `engine::InventoryManager`: it consumes bus [`Event`](core_types::Event)s
//! through [`Analytics::on_event`] and rebuilds bit-for-bit from a journal
//! replay through [`Analytics::rebuild`] (the §3/§9 restart guarantee — both
//! paths are tested to agree). No clock, no IO, no `tokio`: every decision is
//! driven off event-carried timestamps (`fill.ts_venue`, `snapshot.ts`,
//! `settlement.ts`, `market.close_time`), so the live fold and a replay produce
//! identical state and an identical effect stream. The bot binary owns the
//! async bus → `on_event` glue (deferred, like every engine task's wiring).
//!
//! Money stays [`Decimal`](core_types::Decimal); markouts are model-space
//! probabilities and stay `f64` — the two never cross (the only contact point
//! is [`taker_fee`](core_types::taker_fee), which is pure Decimal).
//!
//! # Modules
//!
//! - [`markout`] — per-passive-fill markouts at 1s/5s/30s/expiry against model fair.
//! - [`attribution`] — per-window PnL decomposition that reconciles to the ledger.
//! - [`activity`] — live order-activity (placed/cancelled/fills, resting, TTC/TTR).
//! - [`health`] — the per-series adverse-selection alarm.
//! - [`rollup`] — daily + per-series aggregates (the §10 comparison columns).
//! - [`params`] — crate-local tuning (`config → params` boundary map deferred).

pub mod activity;
pub mod attribution;
pub mod datekey;
pub mod driver_matrix;
pub mod engine;
pub mod health;
pub mod markout;
pub mod params;
#[cfg(feature = "replay")]
pub mod replay;
pub mod rollup;

pub use activity::OrderActivity;
pub use attribution::{WindowAttrib, WindowAttribution};
pub use datekey::DayKey;
pub use driver_matrix::{DriverCell, DriverMatrix};
pub use engine::{Analytics, AnalyticsEffect};
pub use health::{AdverseSelectionMonitor, AdverseSelectionState, SeriesHealth};
pub use markout::{
    FillMarkout, MarkoutEngine, MarkoutHorizon, MarkoutPoint, MarkoutSource, fair_of, position_sign,
};
pub use params::{AdverseSelectionParams, AnalyticsParams};
#[cfg(feature = "replay")]
pub use replay::{
    ReplayConfig, ReplayError, ReplayOutput, ReplaySummary, SweepAggregate, SweepGrid, SweepPoint,
    SweepReport, SweepRow, run_replay, run_sweep,
};
pub use rollup::{
    ComparisonWindow, DailyRollup, MARKOUT_BIN_COUNT, MARKOUT_BIN_EDGES, MarkoutDistribution,
    RollupStore, SeriesComparison, SeriesComparisonRow, SortColumn,
};
