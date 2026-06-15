//! Journal storage + rotation + retention (CLAUDE.md §9, §12): the append-only
//! gzip segment log (the source of truth) and the queryable sqlite index over
//! it. These map to `journal::RecorderParams` at the bot boundary.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Where and how the journal persists data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JournalConfig {
    /// Directory holding the `journal-*.jsonl.gz` segment log (the source of
    /// truth for replay and restart-rebuild).
    pub dir: PathBuf,
    /// SQLite database file for the queryable structured index.
    pub sqlite_path: PathBuf,
    /// Rotate to a new segment after this many **uncompressed** input bytes.
    pub max_segment_bytes: u64,
    /// Rotate to a new segment after the current one is this old (ms), even if
    /// it has not reached the size bound — bounds segments in quiet periods.
    pub max_segment_age_ms: i64,
    /// Bounded channel capacity between the bus consumer and the writer thread;
    /// when full, events are dropped + counted (never block the bus).
    pub channel_capacity: usize,
    /// Flush the compressed stream (and commit the index) at least this often
    /// (ms) under continuous load.
    pub flush_interval_ms: i64,
    /// `sync_all` the segment file on every cadence flush (power-loss durability
    /// of the flushed prefix). Off by default; a process crash already replays
    /// the flushed prefix without it.
    pub fsync_on_flush: bool,
    /// Delete segments and index rows older than this (ms). `0` disables
    /// age-based retention.
    pub retention_max_age_ms: i64,
    /// Cap the total on-disk size of the segment directory at this many bytes,
    /// deleting the oldest segments first (the current segment is always kept).
    /// `0` disables size-based retention.
    pub retention_max_total_bytes: u64,
}

impl Default for JournalConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("data/journal"),
            sqlite_path: PathBuf::from("data/journal.sqlite"),
            max_segment_bytes: 134_217_728, // 128 MiB
            max_segment_age_ms: 3_600_000,  // 1 h
            channel_capacity: 16_384,
            flush_interval_ms: 250,
            fsync_on_flush: false,
            retention_max_age_ms: 0,
            retention_max_total_bytes: 0,
        }
    }
}

impl JournalConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            !self.dir.as_os_str().is_empty(),
            "journal.dir",
            "must not be empty",
        );
        v.require(
            !self.sqlite_path.as_os_str().is_empty(),
            "journal.sqlite_path",
            "must not be empty",
        );
        // A segment must hold at least a few KiB of records.
        v.require(
            self.max_segment_bytes >= 4_096,
            "journal.max_segment_bytes",
            "must be at least 4096",
        );
        v.require(
            self.max_segment_age_ms > 0,
            "journal.max_segment_age_ms",
            "must be positive",
        );
        v.require(
            self.channel_capacity >= 1,
            "journal.channel_capacity",
            "must be at least 1",
        );
        v.require(
            self.flush_interval_ms > 0,
            "journal.flush_interval_ms",
            "must be positive",
        );
        v.require(
            self.retention_max_age_ms >= 0,
            "journal.retention_max_age_ms",
            "must not be negative (0 disables)",
        );
        // A size cap below one segment would churn pathologically; 0 disables.
        v.require(
            self.retention_max_total_bytes == 0
                || self.retention_max_total_bytes >= self.max_segment_bytes,
            "journal.retention_max_total_bytes",
            "must be 0 (disabled) or at least journal.max_segment_bytes",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        JournalConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_bad_values() {
        let cfg = JournalConfig {
            dir: PathBuf::new(),
            sqlite_path: PathBuf::new(),
            max_segment_bytes: 10,
            max_segment_age_ms: 0,
            channel_capacity: 0,
            flush_interval_ms: 0,
            fsync_on_flush: false,
            retention_max_age_ms: -1,
            // Below one segment (and segment itself is invalid) — still flagged.
            retention_max_total_bytes: 5,
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        // dir, sqlite_path, max_segment_bytes, max_segment_age_ms,
        // channel_capacity, flush_interval_ms, retention_max_age_ms,
        // retention_max_total_bytes = 8 violations.
        assert_eq!(v.into_result().unwrap_err().len(), 8);
    }

    #[test]
    fn retention_size_cap_must_cover_a_segment() {
        let cfg = JournalConfig {
            retention_max_total_bytes: 1_000, // < max_segment_bytes default
            ..JournalConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert_eq!(v.into_result().unwrap_err().len(), 1);
    }
}
