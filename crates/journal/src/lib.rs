//! Durable record of everything the bot sees and does: an append-only event
//! log written by a dedicated journal task off the hot path, and a replay
//! reader used by analytics and for offline strategy work. Every order intent,
//! placement, cancel, fill, breaker event, config change, and paper-capital
//! adjustment is journaled identically in paper and live modes so downstream
//! consumers treat both uniformly (CLAUDE.md §9, §12).
//!
//! This crate currently implements the **file-based recorder + replay reader**:
//! the full bus [`core_types::Event`] firehose is projected into a serializable
//! [`JournalRecord`], written as gzip-compressed, size/age-rotated JSONL
//! segments (`journal-*.jsonl.gz`) by [`Recorder`], and read back in order by
//! [`ReplayReader`]. (A structured sqlite index over the same records is a
//! later task; the raw segments are the source of truth.)

pub mod record;
pub mod recorder;
pub mod replay;

pub use record::{JournalRecord, RecordEnvelope};
pub use recorder::{JournalError, Recorder, RecorderParams, RecorderStats};
pub use replay::ReplayReader;
