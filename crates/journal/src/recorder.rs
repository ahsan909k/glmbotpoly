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
//! flushes and finalizes the gzip stream.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use core_types::{Event, TimestampMs};
use flate2::Compression;
use flate2::write::GzEncoder;
use tokio::sync::mpsc;

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
}

impl Default for RecorderParams {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data/journal"),
            max_segment_bytes: 128 * 1024 * 1024,
            max_segment_age_ms: 3_600_000,
            channel_capacity: 16_384,
            flush_interval_ms: 250,
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
    let mut stats = RecorderStats::default();
    let mut seq: u64 = 0;
    let mut last_flush = now_fn();
    let mut last_warned_dropped: u64 = 0;

    loop {
        // Block for the next event; `None` = every sender dropped → shut down.
        let Some(event) = rx.blocking_recv() else {
            break;
        };
        write_one(&params, &mut current, &mut stats, &mut seq, &now_fn, &event)?;

        // Drain whatever else is queued without blocking, flushing on cadence
        // so a sustained backlog still reaches disk promptly.
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    write_one(&params, &mut current, &mut stats, &mut seq, &now_fn, &event)?;
                    let now = now_fn();
                    if now.signed_duration_since(last_flush).as_millis() >= params.flush_interval_ms
                        && let Some(seg) = current.as_mut()
                    {
                        seg.encoder.flush()?;
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
            seg.encoder.flush()?;
            last_flush = now_fn();
        }
        warn_on_new_drops(&dropped, &mut last_warned_dropped);
    }

    if let Some(seg) = current.take() {
        // Finalize the gzip trailer + flush the underlying file.
        seg.encoder.finish()?.flush()?;
    }
    Ok(stats)
}

/// Projects, serializes, and writes one event, rotating the segment first if the
/// size or age bound is reached.
fn write_one<F>(
    params: &RecorderParams,
    current: &mut Option<Segment>,
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
    Ok(Segment {
        encoder: GzEncoder::new(BufWriter::new(file), Compression::default()),
        bytes: 0,
        opened_at: now,
    })
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
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, Ordering};

    use core_types::{Asset, Event, PriceSource, PriceTick, TickKind, TimestampMs};
    use rust_decimal::dec;

    use super::*;
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
}
