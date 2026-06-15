//! The off-hot-path journal writer.
//!
//! [`Recorder::record`] is called from the bus-consumer task for **every**
//! event; it only clones the (cheap, `Arc`-backed) [`Event`] and `try_send`s it
//! onto a **bounded** channel — never blocking the bus and never growing without
//! bound over days. If the channel is full (a stalled disk), the event is
//! **dropped** and counted, never back-pressured into the bus.
//!
//! A dedicated **OS thread** (blocking file IO never touches the async runtime,
//! the `bot::feed` writer idiom) projects each `Event` into a
//! [`JournalRecord`](crate::JournalRecord), serializes it to one JSONL line, and
//! writes it into a **gzip-compressed**, **size/age-rotated** segment file. The
//! `Event → JournalRecord` projection and the serialization both run on the
//! writer thread, off the bus.
//!
//! Crash-safety vs. throughput: unlike the ~1 Hz calibration writer (which
//! flushes per record), the journal flushes on a cadence + when it catches up —
//! a per-record `fsync` at ~1500 msg/s would defeat the buffer. The journal is
//! replay/analytics input, not the live order-audit trail; graceful shutdown
//! flushes and finalizes the gzip stream. Optionally (`fsync_on_flush`) each
//! cadence flush also `sync_all`s the segment for power-loss durability.
//!
//! Alongside the gzip log the writer maintains an optional **sqlite index**
//! ([`JournalIndex`]) of the structured low-rate events — strictly secondary
//! and best-effort: if it fails to open or errors mid-run it is dropped and the
//! gzip capture (the source of truth) continues uninterrupted. Old segments and
//! index rows are pruned by the retention policy on each rotation and at finish.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use core_types::{Event, TimestampMs};
use flate2::Compression;
use flate2::write::GzEncoder;
use tokio::sync::mpsc;

use crate::index::JournalIndex;
use crate::record::{JournalRecord, RecordEnvelope};

/// Tuning for a [`Recorder`]. Defaults target the ~1500 msg/s firehose over
/// days (CLAUDE.md §3/§7).
#[derive(Debug, Clone)]
pub struct RecorderParams {
    /// Directory the `journal-*.jsonl.gz` segments are written to.
    pub out_dir: PathBuf,
    /// Rotate to a new segment once this many **uncompressed** input bytes have
    /// been written to the current one (default 128 MiB → ~15–25 MiB on disk at
    /// typical gzip ratios). Tracked on the input side so rotation is exact and
    /// independent of the compressor's internal buffering.
    pub max_segment_bytes: u64,
    /// Rotate to a new segment once the current one is this old, even if it has
    /// not reached the size bound (default 1 h) — bounds segments during quiet
    /// periods.
    pub max_segment_age_ms: i64,
    /// Bounded channel capacity between the bus consumer and the writer thread.
    /// When full, events are dropped + counted (never block the bus).
    pub channel_capacity: usize,
    /// Flush the buffered/compressed stream to disk at least this often while
    /// under continuous load (default 250 ms). The writer also flushes whenever
    /// it drains the backlog, so latency-to-disk is bounded either way.
    pub flush_interval_ms: i64,
    /// Path of the structured sqlite index, or `None` to record gzip-only. When
    /// set, the six structured event kinds (orders, fills, windows, settlements,
    /// breaker trips, config changes) are also inserted for querying.
    pub sqlite_path: Option<PathBuf>,
    /// `sync_all` the segment file on every cadence flush (power-loss durability
    /// of the flushed prefix). Off by default — a process crash already replays
    /// the flushed prefix without it; this only guards against a host power loss.
    pub fsync_on_flush: bool,
    /// Delete segments and index rows older than this many milliseconds. `0`
    /// disables age-based retention.
    pub retention_max_age_ms: i64,
    /// Cap the total on-disk size of the segment directory at this many bytes,
    /// deleting the oldest segments first (the current segment is always kept).
    /// `0` disables size-based retention.
    pub retention_max_total_bytes: u64,
}

impl Default for RecorderParams {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data/journal"),
            max_segment_bytes: 128 * 1024 * 1024,
            max_segment_age_ms: 3_600_000,
            channel_capacity: 16_384,
            flush_interval_ms: 250,
            sqlite_path: None,
            fsync_on_flush: false,
            retention_max_age_ms: 0,
            retention_max_total_bytes: 0,
        }
    }
}

/// What a finished recording produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecorderStats {
    /// Records written to disk.
    pub records: u64,
    /// Records dropped because the bounded channel was full (a slow/stalled
    /// disk) — a nonzero value means the journal is incomplete.
    pub dropped: u64,
    /// Segment files produced.
    pub segments: u64,
    /// Structured rows inserted into the sqlite index.
    pub indexed: u64,
    /// Segments deleted by the retention policy.
    pub pruned_segments: u64,
}

/// Journal-writer failures.
#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    /// A filesystem operation failed.
    #[error("journal io error: {0}")]
    Io(#[from] std::io::Error),
    /// Serializing a record to JSON failed.
    #[error("journal encode error: {0}")]
    Encode(#[from] serde_json::Error),
    /// A sqlite operation on the structured index failed.
    #[error("journal sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A stored index value could not be parsed back into its typed form (a
    /// corrupt row, or one written by an incompatible schema).
    #[error("journal index corrupt: {0}")]
    Corrupt(String),
    /// The writer thread panicked.
    #[error("journal writer thread panicked")]
    WriterPanicked,
}

/// Handle to a running recorder: enqueue events with [`Recorder::record`], stop
/// and finalize with [`Recorder::finish`].
pub struct Recorder {
    tx: mpsc::Sender<Event>,
    dropped: Arc<AtomicU64>,
    writer: JoinHandle<Result<RecorderStats, JournalError>>,
}

impl Recorder {
    /// Creates `params.out_dir`, spawns the writer thread, and returns the
    /// handle. `now_fn` supplies wall-clock millis for segment stamps, rotation
    /// ages, and the per-record `ts_local_ms` (production: `timeutil::wall_now`;
    /// tests: a closure).
    ///
    /// # Errors
    /// Returns the IO error if `out_dir` cannot be created.
    pub fn spawn<F>(params: RecorderParams, now_fn: F) -> std::io::Result<Self>
    where
        F: Fn() -> TimestampMs + Send + 'static,
    {
        std::fs::create_dir_all(&params.out_dir)?;
        let (tx, rx) = mpsc::channel::<Event>(params.channel_capacity.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = {
            let dropped = Arc::clone(&dropped);
            std::thread::Builder::new()
                .name("journal-writer".to_owned())
                .spawn(move || write_loop(params, rx, now_fn, dropped))?
        };
        Ok(Self {
            tx,
            dropped,
            writer,
        })
    }

    /// Enqueues one event for recording. Cheap (clones the `Arc`-backed event,
    /// `try_send`); drops + counts the event if the channel is full. Never
    /// blocks.
    pub fn record(&self, event: &Event) {
        if self.tx.try_send(event.clone()).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records dropped so far (for periodic operator reporting during the run).
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Stops the writer: closes the channel so the thread drains the backlog,
    /// flushes, finalizes the gzip trailer, and returns the totals.
    ///
    /// # Errors
    /// Returns a [`JournalError`] if the final flush/finalize failed or the
    /// writer thread panicked.
    pub fn finish(self) -> Result<RecorderStats, JournalError> {
        let Recorder {
            tx,
            dropped,
            writer,
        } = self;
        drop(tx); // closes the channel → write_loop drains and returns
        let mut stats = writer.join().map_err(|_| JournalError::WriterPanicked)??;
        stats.dropped = dropped.load(Ordering::Relaxed);
        Ok(stats)
    }
}

/// One open gzip segment.
struct Segment {
    encoder: GzEncoder<BufWriter<std::fs::File>>,
    /// A second handle to the same file, for `sync_all` without finalizing the
    /// gzip stream (the encoder owns the primary handle through the `BufWriter`).
    file: std::fs::File,
    /// Uncompressed bytes written so far (the size-rotation counter).
    bytes: u64,
    /// When this segment was opened (the age-rotation anchor).
    opened_at: TimestampMs,
}

/// The writer-thread body: drain the channel, project + serialize + write each
/// event into a rotating gzip segment, flushing on cadence and on catch-up.
fn write_loop<F>(
    params: RecorderParams,
    mut rx: mpsc::Receiver<Event>,
    now_fn: F,
    dropped: Arc<AtomicU64>,
) -> Result<RecorderStats, JournalError>
where
    F: Fn() -> TimestampMs,
{
    let mut current: Option<Segment> = None;
    // The sqlite index is strictly secondary: an open failure logs and drops to
    // gzip-only rather than killing the capture (the segments are the source of
    // truth and the index is rebuildable).
    let mut index: Option<JournalIndex> = match params.sqlite_path.as_deref() {
        Some(path) => match JournalIndex::open(path) {
            Ok(idx) => Some(idx),
            Err(error) => {
                tracing::error!(
                    target: "journal",
                    %error,
                    path = %path.display(),
                    "could not open the sqlite index — continuing gzip-only"
                );
                None
            }
        },
        None => None,
    };
    let mut stats = RecorderStats::default();
    let mut seq: u64 = 0;
    let mut last_flush = now_fn();
    let mut last_warned_dropped: u64 = 0;

    loop {
        // Block for the next event; `None` = every sender dropped → shut down.
        let Some(event) = rx.blocking_recv() else {
            break;
        };
        write_one(
            &params,
            &mut current,
            &mut index,
            &mut stats,
            &mut seq,
            &now_fn,
            &event,
        )?;

        // Drain whatever else is queued without blocking, flushing on cadence
        // so a sustained backlog still reaches disk promptly.
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    write_one(
                        &params,
                        &mut current,
                        &mut index,
                        &mut stats,
                        &mut seq,
                        &now_fn,
                        &event,
                    )?;
                    let now = now_fn();
                    if now.signed_duration_since(last_flush).as_millis() >= params.flush_interval_ms
                        && let Some(seg) = current.as_mut()
                    {
                        flush_segment(seg, params.fsync_on_flush)?;
                        index_commit(&mut index);
                        last_flush = now;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                // Disconnected: the outer `blocking_recv` will return `None`.
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        // Caught up → flush so a quiet period never strands records in the
        // compressor's buffer.
        if let Some(seg) = current.as_mut() {
            flush_segment(seg, params.fsync_on_flush)?;
            index_commit(&mut index);
            last_flush = now_fn();
        }
        warn_on_new_drops(&dropped, &mut last_warned_dropped);
    }

    if let Some(seg) = current.take() {
        // Finalize the gzip trailer + flush the underlying file.
        seg.encoder.finish()?.flush()?;
    }
    // Commit, prune one last time, and close the index cleanly.
    index_commit(&mut index);
    sweep_retention(&params, now_fn(), &mut index, &mut stats);
    if let Some(idx) = index.take()
        && let Err(error) = idx.close()
    {
        tracing::error!(target: "journal", %error, "closing the sqlite index failed");
    }
    Ok(stats)
}

/// Projects, serializes, and writes one event, rotating the segment first if the
/// size or age bound is reached. On rotation the finalized segment's index rows
/// are committed and the retention policy runs.
fn write_one<F>(
    params: &RecorderParams,
    current: &mut Option<Segment>,
    index: &mut Option<JournalIndex>,
    stats: &mut RecorderStats,
    seq: &mut u64,
    now_fn: &F,
    event: &Event,
) -> Result<(), JournalError>
where
    F: Fn() -> TimestampMs,
{
    let now = now_fn();
    let rotate = match current.as_ref() {
        None => true,
        Some(seg) => {
            seg.bytes >= params.max_segment_bytes
                || now.signed_duration_since(seg.opened_at).as_millis() >= params.max_segment_age_ms
        }
    };
    if rotate {
        if let Some(seg) = current.take() {
            seg.encoder.finish()?.flush()?;
            // The just-closed segment is durable; commit its index rows and run
            // retention now that a new current segment is about to be opened.
            index_commit(index);
            sweep_retention(params, now, index, stats);
        }
        *current = Some(open_segment(params, now, stats.segments)?);
        stats.segments += 1;
    }

    *seq += 1;
    let envelope = RecordEnvelope {
        seq: *seq,
        ts_local_ms: now.as_millis(),
        rec: JournalRecord::from_event(event),
    };
    let mut line = serde_json::to_string(&envelope)?;
    line.push('\n');

    let seg = current.as_mut().ok_or_else(|| {
        JournalError::Io(std::io::Error::other(
            "segment unexpectedly absent after rotation",
        ))
    })?;
    seg.encoder.write_all(line.as_bytes())?;
    seg.bytes += line.len() as u64;
    stats.records += 1;
    // Fan out the structured kinds to the sqlite index (best-effort).
    index_record(index, &envelope, stats);
    Ok(())
}

/// Opens a fresh `journal-{YYYYMMDD-HHMMSS}-{index:05}.jsonl.gz` segment. The
/// monotonic `index` suffix keeps the filename's lexical order chronological
/// even when several segments share a wall-clock second.
fn open_segment(
    params: &RecorderParams,
    now: TimestampMs,
    index: u64,
) -> Result<Segment, JournalError> {
    let name = format!("journal-{}-{index:05}.jsonl.gz", segment_stamp(now));
    let path = params.out_dir.join(name);
    let file = std::fs::File::create(&path)?;
    // A second handle to the same file so `sync_all` can run without taking the
    // gzip encoder apart.
    let sync_handle = file.try_clone()?;
    Ok(Segment {
        encoder: GzEncoder::new(BufWriter::new(file), Compression::default()),
        file: sync_handle,
        bytes: 0,
        opened_at: now,
    })
}

/// Flushes the compressed stream to the OS, and (when `fsync`) to disk.
fn flush_segment(seg: &mut Segment, fsync: bool) -> Result<(), JournalError> {
    seg.encoder.flush()?;
    if fsync {
        seg.file.sync_all()?;
    }
    Ok(())
}

/// Indexes one record into the sqlite index, if present. Best-effort: on error
/// the index is dropped and the gzip capture continues uninterrupted.
fn index_record(
    index: &mut Option<JournalIndex>,
    envelope: &RecordEnvelope,
    stats: &mut RecorderStats,
) {
    if let Some(idx) = index.as_mut() {
        match idx.index(envelope.seq, envelope.ts_local_ms, &envelope.rec) {
            Ok(()) => stats.indexed = idx.indexed(),
            Err(error) => {
                tracing::error!(
                    target: "journal",
                    %error,
                    "sqlite index write failed — continuing gzip-only"
                );
                *index = None;
            }
        }
    }
}

/// Commits the index's open transaction, if present. Best-effort (see
/// [`index_record`]).
fn index_commit(index: &mut Option<JournalIndex>) {
    if let Some(idx) = index.as_mut()
        && let Err(error) = idx.commit()
    {
        tracing::error!(
            target: "journal",
            %error,
            "sqlite index commit failed — continuing gzip-only"
        );
        *index = None;
    }
}

/// Applies the retention policy: deletes segments older than the age bound, then
/// the oldest segments until the directory is under the size cap (always keeping
/// the current/newest segment), and prunes index rows older than the age bound.
/// Best-effort and non-fatal — a stuck delete or index error never stops capture.
fn sweep_retention(
    params: &RecorderParams,
    now: TimestampMs,
    index: &mut Option<JournalIndex>,
    stats: &mut RecorderStats,
) {
    if params.retention_max_age_ms <= 0 && params.retention_max_total_bytes == 0 {
        return;
    }
    // Segment files sorted oldest-first (lexical name order == chronological).
    let segments = list_segments(&params.out_dir);
    let infos: Vec<(i64, u64)> = segments.iter().map(|s| (s.open_ms, s.bytes)).collect();
    for idx in segments_to_delete(
        &infos,
        now.as_millis(),
        params.retention_max_age_ms,
        params.retention_max_total_bytes,
    ) {
        let path = &segments[idx].path;
        match std::fs::remove_file(path) {
            Ok(()) => stats.pruned_segments += 1,
            Err(error) => tracing::warn!(
                target: "journal",
                %error,
                path = %path.display(),
                "retention: could not delete an old segment"
            ),
        }
    }

    if params.retention_max_age_ms > 0
        && let Some(idx) = index.as_mut()
    {
        let cutoff = now.as_millis().saturating_sub(params.retention_max_age_ms);
        if let Err(error) = idx.prune(cutoff) {
            tracing::error!(
                target: "journal",
                %error,
                "retention: pruning the sqlite index failed — continuing gzip-only"
            );
            *index = None;
        }
    }
}

/// One on-disk segment for retention accounting.
struct SegmentFile {
    path: PathBuf,
    /// Open time parsed from the filename stamp (the writer's clock domain).
    open_ms: i64,
    /// On-disk (compressed) size in bytes.
    bytes: u64,
}

/// Lists `journal-*.jsonl.gz` segments in `dir`, oldest-first (lexical order).
fn list_segments(dir: &Path) -> Vec<SegmentFile> {
    let mut segments: Vec<SegmentFile> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let name = path.file_name()?.to_str()?;
                if !(name.starts_with("journal-") && name.ends_with(".jsonl.gz")) {
                    return None;
                }
                let open_ms = parse_segment_open_ms(name)?;
                let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                Some(SegmentFile {
                    path,
                    open_ms,
                    bytes,
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    segments.sort_by(|a, b| a.path.cmp(&b.path));
    segments
}

/// Pure retention policy over `(open_ms, bytes)` (oldest-first): returns the
/// indices to delete — those older than `max_age_ms` (when > 0), then the oldest
/// remaining until the total drops to `max_total_bytes` (when > 0). The last
/// (newest) segment is always kept — it is the active one — and its bytes still
/// count toward the size total.
fn segments_to_delete(
    infos: &[(i64, u64)],
    now_ms: i64,
    max_age_ms: i64,
    max_total_bytes: u64,
) -> Vec<usize> {
    let n = infos.len();
    let mut delete = vec![false; n];
    // The newest segment (last, by chronological order) is never deleted.
    let protected = n.saturating_sub(1);
    if max_age_ms > 0 {
        for (i, &(open_ms, _)) in infos.iter().enumerate() {
            if i != protected && now_ms.saturating_sub(open_ms) > max_age_ms {
                delete[i] = true;
            }
        }
    }
    if max_total_bytes > 0 {
        let mut total: u64 = infos
            .iter()
            .enumerate()
            .filter(|(i, _)| !delete[*i])
            .map(|(_, &(_, bytes))| bytes)
            .sum();
        for (i, &(_, bytes)) in infos.iter().enumerate() {
            if total <= max_total_bytes {
                break;
            }
            if delete[i] || i == protected {
                continue;
            }
            delete[i] = true;
            total = total.saturating_sub(bytes);
        }
    }
    (0..n).filter(|&i| delete[i]).collect()
}

/// Parses the open instant (unix ms) out of a `journal-{stamp}-{NNNNN}.jsonl.gz`
/// filename. `stamp` is `YYYYMMDD-HHMMSS` (the normal case) or raw millis (the
/// `segment_stamp` fallback). Returns `None` for an unparseable name.
fn parse_segment_open_ms(name: &str) -> Option<i64> {
    let stem = name.strip_prefix("journal-")?.strip_suffix(".jsonl.gz")?;
    // Split off the trailing `-NNNNN` index, leaving the stamp.
    let (stamp, _index) = stem.rsplit_once('-')?;
    match stamp.split_once('-') {
        Some((date, time)) => parse_stamp(date, time),
        // No inner dash → the raw-millis fallback form.
        None => stamp.parse::<i64>().ok(),
    }
}

/// Parses `YYYYMMDD` + `HHMMSS` (UTC) into unix milliseconds.
fn parse_stamp(date: &str, time: &str) -> Option<i64> {
    if date.len() != 8 || time.len() != 6 {
        return None;
    }
    let year: i32 = date.get(0..4)?.parse().ok()?;
    let month: u8 = date.get(4..6)?.parse().ok()?;
    let day: u8 = date.get(6..8)?.parse().ok()?;
    let hour: u8 = time.get(0..2)?.parse().ok()?;
    let minute: u8 = time.get(2..4)?.parse().ok()?;
    let second: u8 = time.get(4..6)?.parse().ok()?;
    let month = time::Month::try_from(month).ok()?;
    let date = time::Date::from_calendar_date(year, month, day).ok()?;
    let clock = time::Time::from_hms(hour, minute, second).ok()?;
    Some(date.with_time(clock).assume_utc().unix_timestamp() * 1000)
}

/// `YYYYMMDD-HHMMSS` UTC, the `bot::latency::file_stamp` shape. Falls back to
/// the raw millis if the instant is unrepresentable.
fn segment_stamp(ts: TimestampMs) -> String {
    let secs = ts.as_millis().div_euclid(1000);
    time::OffsetDateTime::from_unix_timestamp(secs).map_or_else(
        |_| ts.as_millis().to_string(),
        |odt| {
            format!(
                "{:04}{:02}{:02}-{:02}{:02}{:02}",
                odt.year(),
                u8::from(odt.month()),
                odt.day(),
                odt.hour(),
                odt.minute(),
                odt.second(),
            )
        },
    )
}

/// Warns (rate-limited to once per growth observation) when records have been
/// dropped — a stalled disk that the bus had to shed.
fn warn_on_new_drops(dropped: &AtomicU64, last_warned: &mut u64) {
    let now_dropped = dropped.load(Ordering::Relaxed);
    if now_dropped > *last_warned {
        tracing::warn!(
            target: "journal",
            dropped = now_dropped,
            "journal channel saturated — dropping events (disk too slow?)"
        );
        *last_warned = now_dropped;
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, Ordering};

    use core_types::{Asset, Event, PriceSource, PriceTick, TickKind, TimestampMs};
    use rust_decimal::dec;

    use super::*;
    use crate::index::JournalIndexReader;
    use crate::replay::ReplayReader;

    const BASE_MS: i64 = 1_781_000_000_000;

    fn tick(i: i64) -> Event {
        // Distinct per `i` via the timestamps; value is constant.
        Event::PriceTick(PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Mid,
            value: dec!(60000),
            ts_exchange: TimestampMs::from_millis(BASE_MS + i),
            ts_local: TimestampMs::from_millis(BASE_MS + i),
        })
    }

    /// A clock the test can advance by hand (constant here — these tests isolate
    /// size rotation and round-tripping, not age rotation).
    fn clock() -> (Arc<AtomicI64>, impl Fn() -> TimestampMs + Send + 'static) {
        let at = Arc::new(AtomicI64::new(BASE_MS));
        let reader = Arc::clone(&at);
        (at, move || {
            TimestampMs::from_millis(reader.load(Ordering::Relaxed))
        })
    }

    #[test]
    fn records_round_trip_through_segments() {
        let dir = std::env::temp_dir().join(format!("journal-test-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (_at, now_fn) = clock();
        let params = RecorderParams {
            out_dir: dir.clone(),
            ..RecorderParams::default()
        };
        let recorder = Recorder::spawn(params, now_fn).expect("spawn");
        let events: Vec<Event> = (0..50).map(tick).collect();
        for e in &events {
            recorder.record(e);
        }
        let stats = recorder.finish().expect("finish");
        assert_eq!(stats.records, 50);
        assert_eq!(stats.dropped, 0);
        assert_eq!(stats.segments, 1);

        let replayed: Vec<Event> = ReplayReader::open(&dir)
            .expect("open replay")
            .map(|r| r.expect("record").rec.to_event())
            .collect();
        assert_eq!(replayed, events, "replayed events must equal recorded");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rotates_on_size_and_keeps_monotonic_seq() {
        let dir = std::env::temp_dir().join(format!("journal-test-rot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (_at, now_fn) = clock();
        // A tiny size bound → one segment per couple of records.
        let params = RecorderParams {
            out_dir: dir.clone(),
            max_segment_bytes: 200,
            ..RecorderParams::default()
        };
        let recorder = Recorder::spawn(params, now_fn).expect("spawn");
        for i in 0..20 {
            recorder.record(&tick(i));
        }
        let stats = recorder.finish().expect("finish");
        assert_eq!(stats.records, 20);
        assert!(
            stats.segments >= 2,
            "expected rotation, got {}",
            stats.segments
        );

        // Replay yields every record in order with a strictly increasing seq.
        let envs: Vec<RecordEnvelope> = ReplayReader::open(&dir)
            .expect("open")
            .map(|r| r.expect("rec"))
            .collect();
        assert_eq!(envs.len(), 20);
        for (i, env) in envs.iter().enumerate() {
            assert_eq!(
                env.seq,
                i as u64 + 1,
                "seq must be monotonic across segments"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn full_channel_drops_and_counts() {
        let dir = std::env::temp_dir().join(format!("journal-test-drop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Capacity 1 and a writer we never let drain (we hold the only runtime
        // thread busy) is hard to force deterministically; instead, flood far
        // past capacity synchronously — the writer thread cannot keep up with a
        // capacity-1 channel filled in a tight loop, so some sends drop.
        let (_at, now_fn) = clock();
        let params = RecorderParams {
            out_dir: dir.clone(),
            channel_capacity: 1,
            ..RecorderParams::default()
        };
        let recorder = Recorder::spawn(params, now_fn).expect("spawn");
        for i in 0..10_000 {
            recorder.record(&tick(i));
        }
        let stats = recorder.finish().expect("finish");
        assert_eq!(
            stats.records + stats.dropped,
            10_000,
            "every event is either written or counted as dropped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- index fan-out --------------------------------------------------

    fn fill_event(token: &str, fee: rust_decimal::Decimal, ms: i64) -> Event {
        use core_types::{
            Dollars, Fill, Liquidity, Outcome, Price, RoundDir, Series, Side, Size, TickSize,
            TokenId, WindowDuration, WindowId,
        };
        Event::Fill(Arc::new(Fill {
            order_id: core_types::OrderId::new("paper-1").unwrap(),
            trade_id: Some(format!("t-{ms}")),
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(BASE_MS),
            },
            token_id: TokenId::new(token).unwrap(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: Price::quantize(dec!(0.40), TickSize::T001, RoundDir::Down).unwrap(),
            size: Size::new(dec!(30)).unwrap(),
            liquidity: if fee.is_zero() {
                Liquidity::Maker
            } else {
                Liquidity::Taker
            },
            fee: Dollars::new(fee),
            ts_venue: TimestampMs::from_millis(ms),
            ts_local: TimestampMs::from_millis(ms),
        }))
    }

    #[test]
    fn structured_events_land_in_both_the_segments_and_the_index() {
        let dir = std::env::temp_dir().join(format!("journal-test-idx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let sqlite = dir.join("index.sqlite");
        let (_at, now_fn) = clock();
        let params = RecorderParams {
            out_dir: dir.clone(),
            sqlite_path: Some(sqlite.clone()),
            ..RecorderParams::default()
        };
        let recorder = Recorder::spawn(params, now_fn).expect("spawn");
        // Mix high-rate ticks (gzip only) with structured fills (both sinks).
        recorder.record(&tick(0));
        recorder.record(&fill_event("111", dec!(0), BASE_MS + 1));
        recorder.record(&tick(2));
        recorder.record(&fill_event("111", dec!(0.05), BASE_MS + 3));
        let stats = recorder.finish().expect("finish");
        assert_eq!(stats.records, 4, "all four events in the gzip log");
        assert_eq!(stats.indexed, 2, "only the two fills indexed");

        // The gzip log is the complete record (all four events replay).
        let replayed: Vec<Event> = ReplayReader::open(&dir)
            .expect("open replay")
            .events()
            .map(|r| r.expect("event"))
            .collect();
        assert_eq!(replayed.len(), 4);

        // The sqlite index holds exactly the two fills, queryable.
        let reader = JournalIndexReader::open(&sqlite).expect("open index");
        let fills = reader.recent_fills(10).expect("query fills");
        assert_eq!(fills.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- write-crash-reopen ---------------------------------------------

    #[test]
    fn writer_crash_recovers_flushed_prefix_and_committed_index() {
        // Simulates a process killed mid-write: a gzip segment whose flushed
        // prefix is on disk but never finalized (no trailer), and a sqlite index
        // whose committed rows are durable. Reopening must recover both cleanly.
        let dir = std::env::temp_dir().join(format!("journal-test-crash-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let sqlite = dir.join("index.sqlite");

        // Two complete fills that were "flushed", plus a partial trailing line.
        let e1 = fill_event("111", dec!(0), BASE_MS + 1);
        let e2 = fill_event("111", dec!(0.05), BASE_MS + 2);
        let env1 = RecordEnvelope {
            seq: 1,
            ts_local_ms: BASE_MS + 1,
            rec: JournalRecord::from_event(&e1),
        };
        let env2 = RecordEnvelope {
            seq: 2,
            ts_local_ms: BASE_MS + 2,
            rec: JournalRecord::from_event(&e2),
        };

        // Build a genuinely truncated gzip stream: write two lines + a partial,
        // flush to capture the pre-trailer compressed prefix, then write THOSE
        // bytes to disk (never finalizing → no gzip trailer).
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        writeln!(enc, "{}", serde_json::to_string(&env1).unwrap()).unwrap();
        writeln!(enc, "{}", serde_json::to_string(&env2).unwrap()).unwrap();
        // A partial third line (no terminating newline), as a crash would leave.
        write!(enc, "{}", serde_json::to_string(&env1).unwrap()).unwrap();
        enc.flush().unwrap();
        let truncated = enc.get_ref().clone();
        drop(enc); // finalizes into its own Vec, which we discard.
        std::fs::write(
            dir.join("journal-20260611-083000-00000.jsonl.gz"),
            &truncated,
        )
        .unwrap();

        // The index committed its two fills before the crash.
        {
            let mut idx = JournalIndex::open(&sqlite).expect("open index");
            idx.index(1, BASE_MS + 1, &env1.rec).unwrap();
            idx.index(2, BASE_MS + 2, &env2.rec).unwrap();
            idx.commit().unwrap();
            // Drop without close() — a crash leaves no clean close; WAL recovers.
        }

        // Reopen: the gzip prefix replays the two complete records (partial tail
        // dropped), and the index returns the two committed fills.
        let replayed: Vec<Event> = ReplayReader::open(&dir)
            .expect("open replay")
            .events()
            .map(|r| r.expect("event"))
            .collect();
        assert_eq!(replayed.len(), 2, "the two flushed records survive");
        assert_eq!(replayed[0], e1);
        assert_eq!(replayed[1], e2);

        let reader = JournalIndexReader::open(&sqlite).expect("reopen index");
        let fills = reader.recent_fills(10).expect("query fills");
        assert_eq!(fills.len(), 2, "committed index rows survive the crash");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- retention ------------------------------------------------------

    #[test]
    fn segments_to_delete_age_size_and_combined() {
        // Age only: delete anything older than max_age, keeping the newest.
        let by_age = [(0i64, 10u64), (500, 10), (900, 10)];
        assert_eq!(segments_to_delete(&by_age, 1000, 400, 0), vec![0, 1]);

        // Size only: delete oldest until total ≤ cap (newest always kept).
        let by_size = [(0i64, 30u64), (1, 30), (2, 30)];
        assert_eq!(segments_to_delete(&by_size, 10, 0, 70), vec![0]);

        // Combined: age deletes the oldest, size then trims more (newest kept).
        let both = [(0i64, 30u64), (900, 30), (950, 30)];
        assert_eq!(segments_to_delete(&both, 1000, 500, 40), vec![0, 1]);

        // The newest is never deleted even if it alone exceeds the cap.
        let one = [(0i64, 999u64)];
        assert!(segments_to_delete(&one, 1_000_000, 1, 1).is_empty());
    }

    #[test]
    fn parse_segment_open_ms_round_trips() {
        // Normal stamp form, floored to the second.
        let now = TimestampMs::from_millis(1_781_000_000_123);
        let name = format!("journal-{}-00007.jsonl.gz", segment_stamp(now));
        assert_eq!(parse_segment_open_ms(&name), Some(1_781_000_000_000));
        // Raw-millis fallback form.
        assert_eq!(
            parse_segment_open_ms("journal-1781000000000-00001.jsonl.gz"),
            Some(1_781_000_000_000)
        );
        // Non-segment names.
        assert_eq!(parse_segment_open_ms("not-a-segment.txt"), None);
        assert_eq!(
            parse_segment_open_ms("journal-garbage-00000.jsonl.gz"),
            None
        );
    }

    #[test]
    fn sweep_retention_deletes_old_segments_keeping_the_newest() {
        let dir = std::env::temp_dir().join(format!("journal-test-ret-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        // Three segments (content is opaque to retention — it stats size + parses
        // the name), 1000 bytes each, with ascending stamps.
        for (i, stamp) in ["20260101-000000", "20260101-000001", "20260101-000002"]
            .into_iter()
            .enumerate()
        {
            let name = format!("journal-{stamp}-{i:05}.jsonl.gz");
            std::fs::write(dir.join(name), vec![0u8; 1000]).expect("write segment");
        }
        let params = RecorderParams {
            out_dir: dir.clone(),
            retention_max_total_bytes: 1500,
            ..RecorderParams::default()
        };
        let mut stats = RecorderStats::default();
        let mut index = None;
        sweep_retention(&params, TimestampMs::from_millis(0), &mut index, &mut stats);

        let remaining: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.starts_with("journal-"))
            .collect();
        assert_eq!(stats.pruned_segments, 2);
        assert_eq!(remaining.len(), 1);
        assert!(
            remaining[0].contains("000002"),
            "the newest segment must survive, got {remaining:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
