//! The shadow prediction side-channel: a `ShadowPrediction` per inference,
//! written to its own per-UTC-day gzip series `data/shadow/shadow-{YYYYMMDD}.jsonl.gz`
//! — outside `core_types::Event` and the journal machinery entirely (the
//! `depth_capture` precedent), so a shadow write can never disturb the engine.
//!
//! Off-hot-path by construction: [`ShadowRecorder::record`] `try_send`s onto a
//! bounded channel (drop + count on full, never back-pressures), and a dedicated
//! OS thread does the gzip IO. NaN feature values serialize as JSON `null`
//! (serde_json maps non-finite floats to null); the Python parity guard reads
//! them back as missing.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;
use tokio::sync::mpsc;

/// One journaled shadow prediction — the full feature vector (so the nightly
/// LIVE parity guard can reconstruct-and-compare), the model identity (for the
/// staleness alert), the probability, and the as-of diagnostics.
#[derive(Debug, Clone, Serialize)]
pub struct ShadowPrediction {
    /// Window-aligned sample time (local ms).
    pub ts: i64,
    /// Series key, e.g. `"BTC-5m"`.
    pub series: String,
    /// Window open time (ms) — with `series`, the window key.
    pub window_open_ms: i64,
    /// Model probability of Up.
    pub p_up: f64,
    /// The 24 feature values in `FULL_FEATURES` order (NaN → JSON null).
    pub features: Vec<f32>,
    /// Stable hash of `features` (integrity / dedup).
    pub features_hash: String,
    /// Deployed model file sha256 (identity).
    pub model_sha256: String,
    /// Model training-data end (ms) — drives the staleness alert.
    pub model_trained_through_ms: i64,
    /// Model training seed (identity).
    pub model_seed: i64,
    /// Frozen strike used for `log_s_k` (`None` if not captured yet).
    pub strike: Option<f64>,
    /// As-of second of the price bar.
    pub price_asof_sec: Option<i64>,
    /// As-of ms of the depth features.
    pub depth_asof_ms: Option<i64>,
    /// As-of ms of the PM book.
    pub pm_asof_ms: Option<i64>,
    /// `sigma_1s` folded-return count.
    pub sigma_returns: u64,
    /// `basis_ewma` folded-sample count.
    pub basis_samples: u32,
    /// Seconds since the PM book run started.
    pub pm_run_secs: Option<i64>,
    /// How many of the 24 features are finite.
    pub finite_count: u32,
}

/// What a finished recorder produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecorderStats {
    /// Predictions written to disk.
    pub written: u64,
    /// Predictions dropped because the channel was full.
    pub dropped: u64,
    /// Day-files opened.
    pub files: u64,
}

/// The shadow prediction recorder (bounded channel + dedicated writer thread).
pub struct ShadowRecorder {
    tx: mpsc::Sender<ShadowPrediction>,
    dropped: Arc<AtomicU64>,
    writer: Option<std::thread::JoinHandle<std::io::Result<(u64, u64)>>>,
}

impl ShadowRecorder {
    /// Spawns the writer thread, creating `out_dir`.
    ///
    /// # Errors
    /// [`std::io::Error`] if the output directory cannot be created.
    pub fn spawn(out_dir: PathBuf, channel_capacity: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(&out_dir)?;
        let (tx, rx) = mpsc::channel::<ShadowPrediction>(channel_capacity.max(1));
        let writer = std::thread::Builder::new()
            .name("shadow-writer".to_owned())
            .spawn(move || write_loop(&out_dir, rx))?;
        Ok(Self {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            writer: Some(writer),
        })
    }

    /// Records one prediction (off-hot-path; drops + counts on a full channel).
    pub fn record(&self, pred: ShadowPrediction) {
        if self.tx.try_send(pred).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Predictions dropped so far (a full-channel / slow-disk signal).
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Stops the writer (draining the backlog + finalizing the gzip) and returns
    /// the run stats.
    #[must_use]
    pub fn finish(mut self) -> RecorderStats {
        // Drop the sender so the writer loop ends.
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

/// The writer-thread body: drain the channel, append each prediction as a JSONL
/// line to the current UTC day-file (rotating on a day change).
fn write_loop(
    out_dir: &Path,
    mut rx: mpsc::Receiver<ShadowPrediction>,
) -> std::io::Result<(u64, u64)> {
    let mut current: Option<DaySegment> = None;
    let (mut written, mut files) = (0u64, 0u64);
    while let Some(pred) = rx.blocking_recv() {
        write_one(&mut current, out_dir, &pred, &mut written, &mut files)?;
        while let Ok(pred) = rx.try_recv() {
            write_one(&mut current, out_dir, &pred, &mut written, &mut files)?;
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
    pred: &ShadowPrediction,
    written: &mut u64,
    files: &mut u64,
) -> std::io::Result<()> {
    let day = utc_date(pred.ts);
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
    // A prediction always serializes (all fields are serde + finite-or-null); on
    // the off chance it does not, skip it rather than kill the writer.
    if let Ok(mut line) = serde_json::to_string(pred) {
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
        "shadow-{:04}{:02}{:02}.jsonl.gz",
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

    fn pred(ts: i64) -> ShadowPrediction {
        ShadowPrediction {
            ts,
            series: "BTC-5m".to_owned(),
            window_open_ms: 0,
            p_up: 0.55,
            features: {
                let mut v = vec![0.5_f32; 24];
                v[3] = f32::NAN;
                v
            },
            features_hash: "deadbeef".to_owned(),
            model_sha256: "abc".to_owned(),
            model_trained_through_ms: 1,
            model_seed: 20260710,
            strike: Some(60_000.0),
            price_asof_sec: Some(1),
            depth_asof_ms: Some(1000),
            pm_asof_ms: Some(1000),
            sigma_returns: 60,
            basis_samples: 30,
            pm_run_secs: Some(5),
            finite_count: 23,
        }
    }

    #[test]
    fn writes_and_serializes_nan_as_null() {
        let dir = std::env::temp_dir().join(format!("shadow-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = ShadowRecorder::spawn(dir.clone(), 128).unwrap();
        // 2023-11-14T12:00:00Z.
        rec.record(pred(1_700_000_000_000));
        let stats = rec.finish();
        assert_eq!(stats.written, 1);
        assert_eq!(stats.files, 1);

        let path = dir.join("shadow-20231114.jsonl.gz");
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(v["series"], "BTC-5m");
        assert_eq!(v["features"][3], serde_json::Value::Null, "NaN → null");
        assert_eq!(v["features"][0], 0.5);
        assert_eq!(v["model_seed"], 20260710);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
