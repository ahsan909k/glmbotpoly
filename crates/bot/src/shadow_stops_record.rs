//! The shadow-loss-stop side channel: one `ShadowStopRecord` per loss stop that
//! WOULD have tripped under `risk.shadow_loss_stops` (paper eval), written to its
//! own per-UTC-day gzip series `data/shadow-stops/shadow-stops-{YYYYMMDD}.jsonl.gz`
//! — outside `core_types::Event` and the journal machinery (the
//! `model_taker_record`/`depth_capture` precedent), so a record write can never
//! disturb the engine.
//!
//! It captures the two loss-based stops (`DailyStop`, per-window `WindowLoss`)
//! that the eval runs in shadow mode: the risk manager computes them exactly but
//! records rather than halting, so this file is the durable audit of how often —
//! and by how much — the loss stops would have fired. Off-hot-path:
//! [`ShadowStopsRecorder::record`] `try_send`s onto a bounded channel (drop +
//! count on full) and a dedicated OS thread does the gzip IO.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use core_types::TimestampMs;
use engine::ShadowStop;
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;
use tokio::sync::mpsc;

/// One journaled would-be loss stop (recorded, not enforced).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShadowStopRecord {
    /// The crossing time (local ms) — the day bucket key.
    pub ts: i64,
    /// When the bot recorded it (local ms).
    pub recorded_ms: i64,
    /// `"DailyStop"` or `"WindowLoss"`.
    pub kind: String,
    /// Series key for a `WindowLoss` (e.g. `"BTC-5m"`); `None` for `DailyStop`.
    pub series: Option<String>,
    /// Window open time (ms) for a `WindowLoss`; `None` for `DailyStop`.
    pub window_open_ms: Option<i64>,
    /// The configured limit/cap crossed (a positive dollar amount).
    pub threshold: String,
    /// The observed loss / worst-case at the crossing (signed).
    pub value: String,
}

impl ShadowStopRecord {
    /// Builds a record from a [`ShadowStop`] and the local time it was drained.
    pub(crate) fn from_stop(s: &ShadowStop, recorded: TimestampMs) -> Self {
        Self {
            ts: s.ts.as_millis(),
            recorded_ms: recorded.as_millis(),
            kind: format!("{:?}", s.kind),
            series: s.window.map(|w| w.series.key().to_owned()),
            window_open_ms: s.window.map(|w| w.open_time.as_millis()),
            threshold: s.threshold.as_decimal().to_string(),
            value: s.value.as_decimal().to_string(),
        }
    }
}

/// What a finished recorder produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RecorderStats {
    /// Records written to disk.
    pub written: u64,
    /// Records dropped because the channel was full.
    pub dropped: u64,
    /// Day-files opened.
    pub files: u64,
}

/// The shadow-stops recorder (bounded channel + dedicated writer thread).
pub(crate) struct ShadowStopsRecorder {
    tx: mpsc::Sender<ShadowStopRecord>,
    dropped: Arc<AtomicU64>,
    writer: Option<std::thread::JoinHandle<std::io::Result<(u64, u64)>>>,
}

impl ShadowStopsRecorder {
    /// Spawns the writer thread, creating `out_dir`.
    ///
    /// # Errors
    /// [`std::io::Error`] if the output directory cannot be created.
    pub(crate) fn spawn(out_dir: PathBuf, channel_capacity: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(&out_dir)?;
        let (tx, rx) = mpsc::channel::<ShadowStopRecord>(channel_capacity.max(1));
        let writer = std::thread::Builder::new()
            .name("shadow-stops-writer".to_owned())
            .spawn(move || write_loop(&out_dir, rx))?;
        Ok(Self {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            writer: Some(writer),
        })
    }

    /// Records one shadow stop (off-hot-path; drops + counts on a full channel).
    pub(crate) fn record(&self, record: ShadowStopRecord) {
        if self.tx.try_send(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Stops the writer (draining the backlog + finalizing the gzip) and returns
    /// the run stats.
    #[must_use]
    pub(crate) fn finish(mut self) -> RecorderStats {
        let dropped = self.dropped.load(Ordering::Relaxed);
        let tx = std::mem::replace(&mut self.tx, mpsc::channel(1).0);
        drop(tx);
        let (written, files) = match self.writer.take().map(std::thread::JoinHandle::join) {
            Some(Ok(Ok((w, f)))) => (w, f),
            _ => (0, 0),
        };
        RecorderStats {
            written,
            dropped,
            files,
        }
    }
}

/// One open gzip day-file.
struct DaySegment {
    encoder: GzEncoder<BufWriter<std::fs::File>>,
    day: time::Date,
}

fn write_loop(
    out_dir: &Path,
    mut rx: mpsc::Receiver<ShadowStopRecord>,
) -> std::io::Result<(u64, u64)> {
    let mut current: Option<DaySegment> = None;
    let (mut written, mut files) = (0u64, 0u64);
    while let Some(d) = rx.blocking_recv() {
        write_one(&mut current, out_dir, &d, &mut written, &mut files)?;
        while let Ok(d) = rx.try_recv() {
            write_one(&mut current, out_dir, &d, &mut written, &mut files)?;
        }
        if let Some(seg) = current.as_mut() {
            seg.encoder.flush()?;
        }
    }
    if let Some(seg) = current.take() {
        seg.encoder.finish()?.flush()?;
    }
    Ok((written, files))
}

fn write_one(
    current: &mut Option<DaySegment>,
    out_dir: &Path,
    d: &ShadowStopRecord,
    written: &mut u64,
    files: &mut u64,
) -> std::io::Result<()> {
    let day = utc_date(d.ts);
    if current.as_ref().is_none_or(|seg| seg.day != day) {
        if let Some(seg) = current.take() {
            seg.encoder.finish()?.flush()?;
        }
        *current = Some(open_day(out_dir, day)?);
        *files += 1;
    }
    let Some(seg) = current.as_mut() else {
        return Ok(());
    };
    if let Ok(mut line) = serde_json::to_string(d) {
        line.push('\n');
        seg.encoder.write_all(line.as_bytes())?;
        *written += 1;
    }
    Ok(())
}

fn utc_date(ms: i64) -> time::Date {
    time::OffsetDateTime::from_unix_timestamp(ms.div_euclid(1_000))
        .map(time::OffsetDateTime::date)
        .unwrap_or_else(|_| time::OffsetDateTime::UNIX_EPOCH.date())
}

fn open_day(dir: &Path, day: time::Date) -> std::io::Result<DaySegment> {
    let name = format!(
        "shadow-stops-{:04}{:02}{:02}.jsonl.gz",
        day.year(),
        u8::from(day.month()),
        day.day()
    );
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(name))?;
    Ok(DaySegment {
        encoder: GzEncoder::new(BufWriter::new(file), Compression::default()),
        day,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::io::Read;

    use super::*;
    use core_types::{Asset, BreakerKind, Dollars, Series, TimestampMs, WindowDuration, WindowId};
    use rust_decimal::dec;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_700_000_000_000),
        }
    }

    #[test]
    fn records_a_daily_and_a_window_shadow_stop() {
        let dir = std::env::temp_dir().join(format!("ss-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = ShadowStopsRecorder::spawn(dir.clone(), 128).unwrap();
        // 2023-11-14T22:13:20Z.
        let ts = TimestampMs::from_millis(1_700_000_000_000);
        rec.record(ShadowStopRecord::from_stop(
            &ShadowStop {
                kind: BreakerKind::DailyStop,
                window: None,
                threshold: Dollars::new(dec!(200)),
                value: Dollars::new(dec!(-210)),
                ts,
            },
            ts,
        ));
        rec.record(ShadowStopRecord::from_stop(
            &ShadowStop {
                kind: BreakerKind::WindowLoss,
                window: Some(window()),
                threshold: Dollars::new(dec!(25)),
                value: Dollars::new(dec!(40)),
                ts,
            },
            ts,
        ));
        let stats = rec.finish();
        assert_eq!(stats.written, 2);
        assert_eq!(stats.dropped, 0);

        let path = dir.join("shadow-stops-20231114.jsonl.gz");
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();
        let daily: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(daily["kind"], "DailyStop");
        assert_eq!(daily["series"], serde_json::Value::Null);
        assert_eq!(daily["value"], "-210");
        let win: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(win["kind"], "WindowLoss");
        assert_eq!(win["series"], "BTC-5m");
        assert_eq!(win["threshold"], "25");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
