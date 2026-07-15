//! The per-driver fill-attribution side channel: one `DriverAttribRecord` per
//! filled order, tagged with the strategy that placed it
//! ([`engine::FillDriver`]), written to its own per-UTC-day gzip series
//! `data/driver-attrib/driver-attrib-{YYYYMMDD}.jsonl.gz` — outside
//! `core_types::Event` and the journal machinery (the
//! `model_taker_record`/`shadow_stops_record`/`depth_capture` precedent), so a
//! record write can never disturb the engine.
//!
//! The driver of a fill is a **live-only** annotation resolved from the risk
//! manager's owned order sets (`RiskManager::driver_of`) — a journal replay
//! cannot reconstruct it (a journaled [`Fill`](core_types::Fill) carries no
//! client id). This side channel is what a per-day digest replays to fold the
//! `(series, driver)` realized-PnL matrix. Off-hot-path:
//! [`DriverAttribRecorder::record`] `try_send`s onto a bounded channel (drop +
//! count on full) and a dedicated OS thread does the gzip IO.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use engine::FillDriver;
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Serialize;
use tokio::sync::mpsc;

/// One journaled driver-tagged fill (its `(series, driver)` realized-PnL row is
/// derived offline by marking it to the window's resolved outcome).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DriverAttribRecord {
    /// Venue-reported fill time (local ms) — the day bucket key.
    pub ts: i64,
    /// Series key, e.g. `"BTC-5m"`.
    pub series: String,
    /// Window open time (ms) — with `series`, the window key.
    pub window_open_ms: i64,
    /// The strategy that placed the order (`"maker-core"`/`"momentum"`/
    /// `"late"`/`"model"`).
    pub driver: String,
    /// Outcome token bought/sold (`"Up"`/`"Down"`).
    pub outcome: String,
    /// Executed shares (exact `Decimal` string).
    pub shares: String,
    /// Execution price (exact `Decimal` string).
    pub price: String,
    /// Fee charged (exact `Decimal` string; zero for maker).
    pub fee: String,
}

impl DriverAttribRecord {
    /// Builds a record from a driver tag and the fill it attributes.
    pub(crate) fn build(driver: FillDriver, fill: &core_types::Fill) -> Self {
        Self {
            ts: fill.ts_venue.as_millis(),
            series: fill.window.series.key().to_owned(),
            window_open_ms: fill.window.open_time.as_millis(),
            driver: driver_label(driver).to_owned(),
            outcome: fill.outcome.to_string(),
            shares: fill.size.as_decimal().to_string(),
            price: fill.price.as_decimal().to_string(),
            fee: fill.fee.as_decimal().to_string(),
        }
    }
}

/// The stable disk label for a [`FillDriver`].
fn driver_label(driver: FillDriver) -> &'static str {
    match driver {
        FillDriver::MakerCore => "maker-core",
        FillDriver::Momentum => "momentum",
        FillDriver::Late => "late",
        FillDriver::Model => "model",
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

/// The driver-attribution recorder (bounded channel + dedicated writer thread).
pub(crate) struct DriverAttribRecorder {
    tx: mpsc::Sender<DriverAttribRecord>,
    dropped: Arc<AtomicU64>,
    writer: Option<std::thread::JoinHandle<std::io::Result<(u64, u64)>>>,
}

impl DriverAttribRecorder {
    /// Spawns the writer thread, creating `out_dir`.
    ///
    /// # Errors
    /// [`std::io::Error`] if the output directory cannot be created.
    pub(crate) fn spawn(out_dir: PathBuf, channel_capacity: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(&out_dir)?;
        let (tx, rx) = mpsc::channel::<DriverAttribRecord>(channel_capacity.max(1));
        let writer = std::thread::Builder::new()
            .name("driver-attrib-writer".to_owned())
            .spawn(move || write_loop(&out_dir, rx))?;
        Ok(Self {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            writer: Some(writer),
        })
    }

    /// Records one driver-tagged fill (off-hot-path; drops + counts on a full
    /// channel).
    pub(crate) fn record(&self, rec: DriverAttribRecord) {
        if self.tx.try_send(rec).is_err() {
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
    mut rx: mpsc::Receiver<DriverAttribRecord>,
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
    d: &DriverAttribRecord,
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
        "driver-attrib-{:04}{:02}{:02}.jsonl.gz",
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Read;

    use super::*;
    use core_types::{
        Asset, Dollars, Fill, Liquidity, OrderId, Outcome, Price, Series, Side, Size, TickSize,
        TimestampMs, TokenId, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    fn fill(outcome: Outcome, price: rust_decimal::Decimal, shares: rust_decimal::Decimal) -> Fill {
        Fill {
            order_id: OrderId::new("paper-1").unwrap(),
            trade_id: None,
            window: WindowId {
                series: Series {
                    asset: Asset::Eth,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(0),
            },
            token_id: TokenId::new("111").unwrap(),
            outcome,
            side: Side::Buy,
            price: Price::on_grid(price, TickSize::T001).unwrap(),
            size: Size::new(shares).unwrap(),
            liquidity: Liquidity::Taker,
            fee: Dollars::new(dec!(0.10)),
            // 2023-11-14T22:13:20Z.
            ts_venue: TimestampMs::from_millis(1_700_000_000_000),
            ts_local: TimestampMs::from_millis(1_700_000_000_000),
        }
    }

    #[test]
    fn records_two_driver_tagged_fills() {
        let dir = std::env::temp_dir().join(format!("da-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = DriverAttribRecorder::spawn(dir.clone(), 128).unwrap();
        rec.record(DriverAttribRecord::build(
            FillDriver::Model,
            &fill(Outcome::Up, dec!(0.80), dec!(10)),
        ));
        rec.record(DriverAttribRecord::build(
            FillDriver::Momentum,
            &fill(Outcome::Down, dec!(0.30), dec!(5)),
        ));
        let stats = rec.finish();
        assert_eq!(stats.written, 2);
        assert_eq!(stats.dropped, 0);

        let path = dir.join("driver-attrib-20231114.jsonl.gz");
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(std::fs::File::open(&path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let lines: Vec<&str> = text.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        // Prices are stored normalized (`0.80` -> `0.8`, `0.30` -> `0.3`).
        let a: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(a["driver"], "model");
        assert_eq!(a["series"], "ETH-5m");
        assert_eq!(a["outcome"], "Up");
        assert_eq!(a["price"], "0.8");
        let b: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(b["driver"], "momentum");
        assert_eq!(b["outcome"], "Down");
        assert_eq!(b["price"], "0.3");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
