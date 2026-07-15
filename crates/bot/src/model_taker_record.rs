//! The model-taker decision side-channel: one `ModelTakerDecision` per shadow
//! prediction the bot feeds the model taker, written to its own per-UTC-day gzip
//! series `data/model-taker/model-taker-{YYYYMMDD}.jsonl.gz` — outside
//! `core_types::Event` and the journal machinery (the `shadow`/`depth_capture`
//! precedent), so a decision write can never disturb the engine.
//!
//! It records **both** fires and suppressions (with the reason), so the 48 h
//! burn-in can report fires/day, arbitration conflicts, budget/cap contention,
//! and breaker interactions per series. Off-hot-path: [`ModelTakerRecorder::record`]
//! `try_send`s onto a bounded channel (drop + count on full) and a dedicated OS
//! thread does the gzip IO.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use engine::{ModelPrediction, ModelTakeOutcome};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;
use tokio::sync::mpsc;

/// One journaled model-taker decision (fired or suppressed, with the reason).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ModelTakerDecision {
    /// Prediction sample time (local ms).
    pub ts: i64,
    /// Series key, e.g. `"BTC-5m"`.
    pub series: String,
    /// Window open time (ms) — with `series`, the window key.
    pub window_open_ms: i64,
    /// The shadow model's `p_up` for this window.
    pub p_up: f64,
    /// How many of the 24 features were finite.
    pub finite_count: u32,
    /// Whether the shadow model was past its refit-due age.
    pub model_stale: bool,
    /// Whether a FAK was placed.
    pub fired: bool,
    /// `"fired"` or the typed suppression reason (§12 observability).
    pub reason: String,
    /// Side bought (`"Up"`/`"Down"`), when fired.
    pub outcome: Option<String>,
    /// Expected shares, when fired.
    pub shares: Option<String>,
    /// Notional spent, when fired.
    pub notional: Option<String>,
    /// FAK worst-price slippage cap, when fired.
    pub worst_price: Option<String>,
}

impl ModelTakerDecision {
    /// Builds a decision record from a prediction and the taker's outcome.
    pub(crate) fn build(pred: &ModelPrediction, outcome: &ModelTakeOutcome) -> Self {
        let (fired, reason, side, shares, notional, worst) = match outcome {
            ModelTakeOutcome::Fired {
                outcome,
                shares,
                notional,
                worst_price,
                ..
            } => (
                true,
                "fired".to_owned(),
                Some(outcome.to_string()),
                Some(shares.as_decimal().to_string()),
                Some(notional.as_decimal().to_string()),
                Some(worst_price.as_decimal().to_string()),
            ),
            ModelTakeOutcome::Suppressed(r) => (false, r.to_string(), None, None, None, None),
        };
        Self {
            ts: pred.ts.as_millis(),
            series: pred.series.key().to_owned(),
            window_open_ms: pred.window_open_ms,
            p_up: pred.p_up,
            finite_count: pred.finite_count,
            model_stale: pred.model_stale,
            fired,
            reason,
            outcome: side,
            shares,
            notional,
            worst_price: worst,
        }
    }
}

/// What a finished recorder produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RecorderStats {
    /// Decisions written to disk.
    pub written: u64,
    /// Decisions dropped because the channel was full.
    pub dropped: u64,
    /// Day-files opened.
    pub files: u64,
}

/// The model-taker decision recorder (bounded channel + dedicated writer thread).
pub(crate) struct ModelTakerRecorder {
    tx: mpsc::Sender<ModelTakerDecision>,
    dropped: Arc<AtomicU64>,
    writer: Option<std::thread::JoinHandle<std::io::Result<(u64, u64)>>>,
}

impl ModelTakerRecorder {
    /// Spawns the writer thread, creating `out_dir`.
    ///
    /// # Errors
    /// [`std::io::Error`] if the output directory cannot be created.
    pub(crate) fn spawn(out_dir: PathBuf, channel_capacity: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(&out_dir)?;
        let (tx, rx) = mpsc::channel::<ModelTakerDecision>(channel_capacity.max(1));
        let writer = std::thread::Builder::new()
            .name("model-taker-writer".to_owned())
            .spawn(move || write_loop(&out_dir, rx))?;
        Ok(Self {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            writer: Some(writer),
        })
    }

    /// Records one decision (off-hot-path; drops + counts on a full channel).
    pub(crate) fn record(&self, decision: ModelTakerDecision) {
        if self.tx.try_send(decision).is_err() {
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
    mut rx: mpsc::Receiver<ModelTakerDecision>,
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
    d: &ModelTakerDecision,
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
        "model-taker-{:04}{:02}{:02}.jsonl.gz",
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
    use core_types::{Asset, Dollars, Outcome, Price, Series, Size, TickSize, TimestampMs, WindowDuration, WindowId};
    use engine::ModelPrediction;
    use rust_decimal::dec;

    fn pred() -> ModelPrediction {
        ModelPrediction {
            series: Series {
                asset: Asset::Eth,
                duration: WindowDuration::M5,
            },
            window_open_ms: 0,
            ts: TimestampMs::from_millis(1_700_000_000_000), // 2023-11-14T22:13:20Z
            p_up: 0.95,
            finite_count: 24,
            model_stale: false,
        }
    }

    #[test]
    fn records_a_fire_and_a_suppression() {
        let dir = std::env::temp_dir().join(format!("mt-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = ModelTakerRecorder::spawn(dir.clone(), 128).unwrap();
        let win = WindowId {
            series: pred().series,
            open_time: TimestampMs::from_millis(0),
        };
        rec.record(ModelTakerDecision::build(
            &pred(),
            &ModelTakeOutcome::Fired {
                window: win,
                outcome: Outcome::Up,
                shares: Size::new(dec!(12.5)).unwrap(),
                notional: Dollars::new(dec!(10)),
                worst_price: Price::on_grid(dec!(0.80), TickSize::T001).unwrap(),
            },
        ));
        rec.record(ModelTakerDecision::build(
            &pred(),
            &ModelTakeOutcome::Suppressed(engine::NoModelTakeReason::NoWindowState),
        ));
        let stats = rec.finish();
        assert_eq!(stats.written, 2);

        let path = dir.join("model-taker-20231114.jsonl.gz");
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();
        let fired: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(fired["fired"], true);
        assert_eq!(fired["series"], "ETH-5m");
        assert_eq!(fired["outcome"], "Up");
        assert_eq!(fired["notional"], "10");
        let supp: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(supp["fired"], false);
        assert_eq!(supp["outcome"], serde_json::Value::Null);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
