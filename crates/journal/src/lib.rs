//! Durable record of everything the bot sees and does: an append-only event
//! log written by a dedicated journal task off the hot path, and a replay
//! reader used by analytics and for offline strategy work. Every order intent,
//! placement, cancel, fill, breaker event, config change, and paper-capital
//! adjustment is journaled identically in paper and live modes so downstream
//! consumers treat both uniformly (CLAUDE.md §9, §12).
//!
//! Two storage tiers over one event stream:
//! - **Raw gzip segments (the source of truth).** The full bus
//!   [`core_types::Event`] firehose is projected into a serializable
//!   [`JournalRecord`] and written as gzip-compressed, size/age-rotated JSONL
//!   segments (`journal-*.jsonl.gz`) by [`Recorder`], read back in order by
//!   [`ReplayReader`]. Replay and restart-rebuild read only from here.
//! - **A queryable sqlite index ([`JournalIndex`]).** As each event is written,
//!   the six *structured* low-rate kinds (orders, fills, windows, settlements,
//!   breaker trips, config changes) are also inserted into a sqlite database for
//!   the dashboard and analytics to query via [`JournalIndexReader`]. High-rate
//!   tick/book/model events never touch sqlite. The index is rebuildable from
//!   the segments and safe to delete.
//!
//! Rotation (size/age) and retention (max age + max total bytes) are driven by
//! [`RecorderParams`]; the bot maps them from the `[journal]` config section.

pub mod index;
pub mod record;
pub mod recorder;
pub mod replay;

pub use index::{
    BreakerRow, ControlRow, FillRow, JournalIndex, JournalIndexReader, OrderRow, SettlementRow,
    WindowRow,
};
pub use record::{JournalRecord, RecordEnvelope};
pub use recorder::{JournalError, Recorder, RecorderParams, RecorderStats};
pub use replay::ReplayReader;
