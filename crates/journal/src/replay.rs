//! Reads recorded journal segments back in order.
//!
//! [`ReplayReader`] globs the `journal-*.jsonl.gz` segments in a directory,
//! sorts them (the segment-name stamp makes lexical order chronological),
//! decompresses each, and yields one [`RecordEnvelope`] per line. [`ReplayReader::events`]
//! maps those back to bus [`Event`]s so a recorded session feeds straight into
//! `PaperVenue::on_bus_event` or analytics.
//!
//! Robust by design: an unreadable segment is skipped, and a crash-truncated
//! trailing segment (a corrupt gzip tail, or a final line written without its
//! newline) yields what decoded and then stops — never a panic.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use core_types::Event;
use flate2::read::MultiGzDecoder;

use crate::record::RecordEnvelope;

/// An in-order reader over a directory of recorded journal segments.
pub struct ReplayReader {
    segments: std::vec::IntoIter<PathBuf>,
    current: Option<BufReader<MultiGzDecoder<File>>>,
}

impl ReplayReader {
    /// Opens every `journal-*.jsonl.gz` segment in `dir`, sorted chronologically.
    ///
    /// # Errors
    /// Returns the IO error if `dir` cannot be listed.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| is_segment(path))
            .collect();
        paths.sort();
        Ok(Self::open_paths(paths))
    }

    /// Builds a reader over an explicit, already-ordered list of segment paths
    /// (tests / selective replay).
    #[must_use]
    pub fn open_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            segments: paths.into_iter(),
            current: None,
        }
    }

    /// Yields the reconstructed bus [`Event`]s (drops each envelope's `seq` /
    /// `ts_local_ms`).
    pub fn events(self) -> impl Iterator<Item = io::Result<Event>> {
        self.map(|res| res.map(|env| env.rec.to_event()))
    }

    /// Opens the next readable segment, skipping any that fail to open. Returns
    /// `false` once the segment list is exhausted.
    fn open_next(&mut self) -> bool {
        for path in self.segments.by_ref() {
            if let Ok(file) = File::open(&path) {
                self.current = Some(BufReader::new(MultiGzDecoder::new(file)));
                return true;
            }
        }
        self.current = None;
        false
    }
}

impl Iterator for ReplayReader {
    type Item = io::Result<RecordEnvelope>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current.is_none() && !self.open_next() {
                return None;
            }
            let reader = self.current.as_mut()?;
            let mut buf = String::new();
            match reader.read_line(&mut buf) {
                // Clean segment EOF → move to the next segment.
                Ok(0) => self.current = None,
                Ok(_) => {
                    // A final line without its newline is a crash-truncated tail:
                    // skip it and treat the segment as done.
                    if !buf.ends_with('\n') {
                        self.current = None;
                        continue;
                    }
                    let trimmed = buf.trim_end();
                    if trimmed.is_empty() {
                        continue;
                    }
                    return Some(
                        serde_json::from_str(trimmed)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
                    );
                }
                // Corrupt gzip tail (truncated mid-write): yield what decoded so
                // far, then stop this segment. Best-effort by design.
                Err(_) => self.current = None,
            }
        }
    }
}

/// A `journal-*.jsonl.gz` segment file.
fn is_segment(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name.starts_with("journal-") && name.ends_with(".jsonl.gz"))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use core_types::{Asset, PriceSource, PriceTick, TickKind, TimestampMs};
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use rust_decimal::dec;

    use super::*;
    use crate::record::JournalRecord;

    fn envelope(seq: u64, i: i64) -> RecordEnvelope {
        RecordEnvelope {
            seq,
            ts_local_ms: 1_781_000_000_000 + i,
            rec: JournalRecord::from_event(&Event::PriceTick(PriceTick {
                source: PriceSource::BinanceDirect,
                asset: Asset::Btc,
                kind: TickKind::Mid,
                value: dec!(60000),
                ts_exchange: TimestampMs::from_millis(1_781_000_000_000 + i),
                ts_local: TimestampMs::from_millis(1_781_000_000_000 + i),
            })),
        }
    }

    /// Writes a gzip segment with the given complete envelopes, optionally
    /// followed by a partial (newline-less) trailing line to simulate a crash.
    fn write_segment(path: &Path, envs: &[RecordEnvelope], trailing_partial: bool) {
        let file = File::create(path).unwrap();
        let mut enc = GzEncoder::new(file, Compression::default());
        for env in envs {
            writeln!(enc, "{}", serde_json::to_string(env).unwrap()).unwrap();
        }
        if trailing_partial {
            // A complete-looking JSON object but with no terminating newline.
            write!(
                enc,
                "{}",
                serde_json::to_string(&envelope(999, 999)).unwrap()
            )
            .unwrap();
        }
        enc.finish().unwrap();
    }

    #[test]
    fn reads_segments_in_chronological_order() {
        let dir = std::env::temp_dir().join(format!("journal-replay-order-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Out-of-order filenames; lexical sort must still interleave correctly.
        write_segment(
            &dir.join("journal-20260611-083000-00001.jsonl.gz"),
            &[envelope(3, 3), envelope(4, 4)],
            false,
        );
        write_segment(
            &dir.join("journal-20260611-083000-00000.jsonl.gz"),
            &[envelope(1, 1), envelope(2, 2)],
            false,
        );

        let seqs: Vec<u64> = ReplayReader::open(&dir)
            .unwrap()
            .map(|r| r.unwrap().seq)
            .collect();
        assert_eq!(seqs, vec![1, 2, 3, 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tolerates_a_truncated_trailing_line() {
        let dir = std::env::temp_dir().join(format!("journal-replay-trunc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_segment(
            &dir.join("journal-20260611-083000-00000.jsonl.gz"),
            &[envelope(1, 1), envelope(2, 2)],
            true, // crash mid-write: a final newline-less record
        );

        let recs: Vec<RecordEnvelope> = ReplayReader::open(&dir)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        // The two complete records survive; the truncated tail is dropped.
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].seq, 1);
        assert_eq!(recs[1].seq, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn events_helper_reconstructs_bus_events() {
        let dir =
            std::env::temp_dir().join(format!("journal-replay-events-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_segment(
            &dir.join("journal-20260611-083000-00000.jsonl.gz"),
            &[envelope(1, 1)],
            false,
        );
        let events: Vec<Event> = ReplayReader::open(&dir)
            .unwrap()
            .events()
            .map(Result::unwrap)
            .collect();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::PriceTick(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
