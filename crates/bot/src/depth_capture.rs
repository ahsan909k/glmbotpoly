//! Binance `depth20@100ms` partial-book-depth capture (research data for the
//! offline `model-lab`).
//!
//! This is a **self-contained** capture facility, deliberately kept off the
//! event bus: the engine's fair-value model consumes only Binance bookTicker
//! mids + trades, never L2 depth, so routing depth through `core_types::Event`
//! would add a variant that ripples through every exhaustive `match Event` in
//! model/risk/analytics/dashboard for no engine benefit. Instead this task owns
//! its **own** dedicated Binance WebSocket connection (`depth20@100ms` for
//! BTCUSDT + ETHUSDT via the combined-stream URL — zero client frames after
//! connect), reuses the shared [`feed_util`] `Transport`/`Backoff` seams (the
//! `feed-clob` hand-rolled-loop pattern), and writes raw frames to its **own**
//! gzip file series under `data/depth/`, rotated **per UTC day** —
//! `binance-depth20-{YYYYMMDD}.jsonl.gz`, a series entirely separate from the
//! `journal-*.jsonl.gz` segments (so the journal's replay/retention machinery
//! never touches it).
//!
//! Each written line is one JSON object: `{"recv_ms":<local ms>,
//! "stream":"btcusdt@depth20@100ms","data":{…bids/asks verbatim…}}`. The raw
//! depth payload is stored unparsed — the Python lab does the level parsing
//! offline.
//!
//! Reconnect is handled exactly like the other feeds (jittered [`feed_util`]
//! backoff + a dead-socket watchdog); reconnects never rotate the day-file
//! (`depth20@100ms` is a snapshot stream, so a reconnect yields fresh snapshots
//! with no historical backfill burst). Wired into `bot record` and `bot run`
//! (gated on `feeds.binance_depth_capture`); it takes no `bus_tx` and never
//! reaches a venue.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use core_types::TimestampMs;
use feed_util::{Backoff, BackoffParams, Connection, Transport, WsFrame};
use flate2::Compression;
use flate2::write::GzEncoder;
use tokio::sync::{mpsc, watch};

/// The depth streams captured, in a fixed order (the combined-stream URL is
/// built from these). Kept local — deliberately NOT added to
/// `feed_binance::BinanceSub`, so feed-binance's parser/staleness/matches stay
/// untouched.
const DEPTH_STREAMS: [&str; 2] = ["btcusdt@depth20@100ms", "ethusdt@depth20@100ms"];

/// A socket with no inbound frame (data **or** server ping) for this long is
/// assumed half-open and torn down. Comfortably above Binance's ~20 s server
/// ping cadence and the ~10 frames/s/symbol data rate.
const DEAD_SOCKET_SILENCE: Duration = Duration::from_secs(30);

/// Dead-socket watchdog cadence.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(1);

/// Rate-limit for the writer-backlog-full warning (warn once per this many new
/// drops so a stalled disk does not flood the log).
const DROP_WARN_EVERY: u64 = 1_000;

/// Builds the Binance combined-stream URL for the depth streams:
/// `{base}/stream?streams=btcusdt@depth20@100ms/ethusdt@depth20@100ms`. The
/// combined-stream form subscribes by URL alone — no client frames.
#[must_use]
pub fn combined_depth_url(base: &str) -> String {
    format!(
        "{}/stream?streams={}",
        base.trim_end_matches('/'),
        DEPTH_STREAMS.join("/")
    )
}

/// Tuning for a depth-capture run.
#[derive(Debug, Clone)]
pub struct DepthCaptureParams {
    /// The prebuilt combined-stream URL (see [`combined_depth_url`]).
    pub url: String,
    /// WebSocket connect timeout.
    pub connect_timeout: Duration,
    /// Jittered reconnect backoff curve (shared `[feeds] reconnect_*` policy).
    pub backoff: BackoffParams,
    /// Directory the `binance-depth20-*.jsonl.gz` day-files are written to.
    pub out_dir: PathBuf,
    /// Bounded writer-channel capacity; when full, frames are dropped + counted
    /// (never back-pressures the socket).
    pub channel_capacity: usize,
}

/// What a finished capture produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DepthStats {
    /// Frames written to disk.
    pub frames_written: u64,
    /// Frames dropped because the writer channel was full (a slow/stalled disk)
    /// — a nonzero value means the capture is incomplete.
    pub frames_dropped: u64,
    /// Day-files opened.
    pub files: u64,
    /// Reconnect episodes.
    pub reconnects: u64,
}

/// One inbound frame handed to the writer thread: the local receive time and
/// the raw (still-wrapped) frame text.
struct DepthFrame {
    recv_ms: i64,
    text: String,
}

/// Why a streaming episode ended.
enum EpisodeEnd {
    /// Reconnect (with the reason logged).
    Reconnect(String),
    /// Graceful shutdown.
    Shutdown,
}

/// Connects, streams `depth20@100ms` to the per-day gzip writer, and reconnects
/// forever until `shutdown_rx` flips. Generic over [`Transport`] so tests drive
/// a scripted fake.
///
/// # Errors
/// Only a failure to create the output directory is fatal; connection trouble
/// is logged and retried forever, and a failing writer degrades to a warning.
pub async fn run<T, F>(
    params: DepthCaptureParams,
    mut transport: T,
    now_fn: F,
    mut shutdown_rx: watch::Receiver<bool>,
    backoff_seed: Option<u64>,
) -> anyhow::Result<DepthStats>
where
    T: Transport,
    F: Fn() -> TimestampMs + Send,
{
    std::fs::create_dir_all(&params.out_dir)
        .with_context(|| format!("creating depth capture dir {}", params.out_dir.display()))?;
    let (writer_tx, writer_rx) = mpsc::channel::<DepthFrame>(params.channel_capacity.max(1));
    let writer = spawn_depth_writer(params.out_dir.clone(), writer_rx);

    let seed = backoff_seed.unwrap_or_else(|| now_fn().as_millis().unsigned_abs());
    let mut backoff = Backoff::new(params.backoff, seed);
    let mut reconnects: u64 = 0;
    let mut frames_dropped: u64 = 0;
    let mut last_drop_warn: u64 = 0;

    tracing::info!(
        target: "depth",
        url = %params.url,
        out_dir = %params.out_dir.display(),
        "binance depth20 capture starting (per-day gzip; reconnects forever)"
    );

    loop {
        // A dropped shutdown sender is a shutdown too (prevents a closed-watch spin).
        if *shutdown_rx.borrow() || shutdown_rx.has_changed().is_err() {
            break;
        }
        let connect = tokio::select! {
            result = transport.connect(&params.url, params.connect_timeout) => result,
            _ = shutdown_rx.changed() => continue,
        };
        let mut conn = match connect {
            Ok(conn) => conn,
            Err(error) => {
                let delay = backoff.next_delay();
                tracing::warn!(
                    target: "depth", %error, retry_in_ms = delay.as_millis() as u64,
                    "connect failed"
                );
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
                continue;
            }
        };
        tracing::info!(target: "depth", "connected");
        let end = stream_episode(StreamCtx {
            conn: &mut conn,
            now_fn: &now_fn,
            writer_tx: &writer_tx,
            backoff: &mut backoff,
            frames_dropped: &mut frames_dropped,
            last_drop_warn: &mut last_drop_warn,
            shutdown_rx: &mut shutdown_rx,
        })
        .await;
        conn.close().await;
        match end {
            EpisodeEnd::Shutdown => break,
            EpisodeEnd::Reconnect(reason) => {
                reconnects += 1;
                let delay = backoff.next_delay();
                tracing::warn!(
                    target: "depth", reason = %reason,
                    retry_in_ms = delay.as_millis() as u64,
                    "connection ended — reconnecting"
                );
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
            }
        }
    }

    // Stop the writer: dropping the sender drains its backlog and finalizes the
    // gzip trailers on the open day-file(s).
    drop(writer_tx);
    let (frames_written, files) = match writer.join() {
        Ok(Ok(stats)) => (stats.frames_written, stats.files),
        Ok(Err(error)) => {
            tracing::warn!(target: "depth", %error, "depth writer failed");
            (0, 0)
        }
        Err(_) => {
            tracing::warn!(target: "depth", "depth writer panicked");
            (0, 0)
        }
    };
    Ok(DepthStats {
        frames_written,
        frames_dropped,
        files,
        reconnects,
    })
}

/// Borrowed context for one streaming episode (keeps the select loop tidy).
struct StreamCtx<'a, C, F> {
    conn: &'a mut C,
    now_fn: &'a F,
    writer_tx: &'a mpsc::Sender<DepthFrame>,
    backoff: &'a mut Backoff,
    frames_dropped: &'a mut u64,
    last_drop_warn: &'a mut u64,
    shutdown_rx: &'a mut watch::Receiver<bool>,
}

/// Streams one connection episode: forward each inbound frame to the writer,
/// reset the backoff on the first data frame, and tear down a dead socket.
async fn stream_episode<C, F>(ctx: StreamCtx<'_, C, F>) -> EpisodeEnd
where
    C: Connection,
    F: Fn() -> TimestampMs,
{
    let StreamCtx {
        conn,
        now_fn,
        writer_tx,
        backoff,
        frames_dropped,
        last_drop_warn,
        shutdown_rx,
    } = ctx;
    let mut watchdog = tokio::time::interval(WATCHDOG_INTERVAL);
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_inbound = now_fn();
    let mut backoff_reset = false;

    loop {
        tokio::select! {
            frame = conn.recv() => {
                let now = now_fn();
                match frame {
                    None => return EpisodeEnd::Reconnect("peer closed".to_owned()),
                    Some(Err(error)) => {
                        return EpisodeEnd::Reconnect(format!("socket error: {error}"));
                    }
                    Some(Ok(frame)) => {
                        // Any inbound frame (data or a server ping) is liveness.
                        last_inbound = now;
                        let text = match frame {
                            WsFrame::Text(text) => text,
                            // Binary frames parsed defensively as UTF-8.
                            WsFrame::Binary(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                            // Server protocol ping: liveness only (tungstenite pongs it).
                            WsFrame::Ping => continue,
                        };
                        if !backoff_reset {
                            backoff.reset();
                            backoff_reset = true;
                        }
                        match writer_tx.try_send(DepthFrame { recv_ms: now.as_millis(), text }) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                *frames_dropped += 1;
                                if *frames_dropped >= *last_drop_warn + DROP_WARN_EVERY {
                                    *last_drop_warn = *frames_dropped;
                                    tracing::warn!(
                                        target: "depth", dropped = *frames_dropped,
                                        "depth writer backlog full — dropping frames"
                                    );
                                }
                            }
                            // The writer thread is gone — end the episode; `run`
                            // will observe it on the join at shutdown.
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                return EpisodeEnd::Reconnect("writer closed".to_owned());
                            }
                        }
                    }
                }
            }
            _ = watchdog.tick() => {
                let silent = now_fn().signed_duration_since(last_inbound);
                if silent.as_millis() >= DEAD_SOCKET_SILENCE.as_millis() as i64 {
                    return EpisodeEnd::Reconnect(format!("dead socket: silent for {silent}"));
                }
            }
            changed = shutdown_rx.changed() => {
                // A dropped sender (supervisor gone) is a shutdown too.
                if changed.is_err() || *shutdown_rx.borrow() {
                    return EpisodeEnd::Shutdown;
                }
            }
        }
    }
}

/// Sleeps `delay`, cut short by a shutdown signal.
async fn sleep_or_shutdown(delay: Duration, shutdown_rx: &mut watch::Receiver<bool>) {
    tokio::select! {
        () = tokio::time::sleep(delay) => {}
        _ = shutdown_rx.changed() => {}
    }
}

// ============================================================================
// per-day gzip writer (dedicated OS thread — the Recorder / `bot feed` idiom)
// ============================================================================

/// What the writer thread produced.
struct WriterStats {
    frames_written: u64,
    files: u64,
}

/// One open gzip day-file.
struct DaySegment {
    encoder: GzEncoder<BufWriter<std::fs::File>>,
    /// The UTC calendar day this file covers (the rotation key).
    day: time::Date,
}

/// Spawns the per-day gzip writer thread. Blocking file IO never touches the
/// async runtime; the thread exits (finalizing the open gzip trailer) when the
/// sender is dropped.
fn spawn_depth_writer(
    out_dir: PathBuf,
    rx: mpsc::Receiver<DepthFrame>,
) -> std::thread::JoinHandle<anyhow::Result<WriterStats>> {
    std::thread::spawn(move || depth_write_loop(&out_dir, rx))
}

/// The writer-thread body: drain the channel, project + write each frame into
/// the current UTC day-file (rotating on a day change), flushing on catch-up.
fn depth_write_loop(
    out_dir: &Path,
    mut rx: mpsc::Receiver<DepthFrame>,
) -> anyhow::Result<WriterStats> {
    let mut current: Option<DaySegment> = None;
    let mut stats = WriterStats {
        frames_written: 0,
        files: 0,
    };
    while let Some(frame) = rx.blocking_recv() {
        write_frame(&mut current, out_dir, &frame, &mut stats)?;
        // Drain whatever else is queued without blocking, then flush once so a
        // quiet period never strands frames in the compressor's buffer.
        while let Ok(frame) = rx.try_recv() {
            write_frame(&mut current, out_dir, &frame, &mut stats)?;
        }
        if let Some(seg) = current.as_mut() {
            seg.encoder.flush()?;
        }
    }
    if let Some(seg) = current.take() {
        seg.encoder.finish()?.flush()?;
    }
    Ok(stats)
}

/// Projects one frame to a JSONL line and appends it to the current day-file,
/// rotating first if the UTC day changed. Frames that are not a `stream`+`data`
/// object (subscription acks, server errors) are skipped.
fn write_frame(
    current: &mut Option<DaySegment>,
    out_dir: &Path,
    frame: &DepthFrame,
    stats: &mut WriterStats,
) -> anyhow::Result<()> {
    let Some(line) = depth_line(frame.recv_ms, &frame.text) else {
        return Ok(());
    };
    let day = utc_date(frame.recv_ms);
    let rotate = current.as_ref().is_none_or(|seg| seg.day != day);
    if rotate {
        if let Some(seg) = current.take() {
            seg.encoder.finish()?.flush()?;
        }
        *current = Some(open_day(out_dir, day)?);
        stats.files += 1;
    }
    let seg = current
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("depth segment absent after rotation"))?;
    seg.encoder.write_all(line.as_bytes())?;
    stats.frames_written += 1;
    Ok(())
}

/// Builds the `{recv_ms, stream, data}` JSONL line for one wrapped combined-stream
/// frame, or `None` if the frame is not a `stream`+`data` object.
fn depth_line(recv_ms: i64, text: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let stream = value.get("stream")?.as_str()?;
    let data = value.get("data")?;
    if data.is_null() {
        return None;
    }
    let record = serde_json::json!({
        "recv_ms": recv_ms,
        "stream": stream,
        "data": data,
    });
    let mut line = serde_json::to_string(&record).ok()?;
    line.push('\n');
    Some(line)
}

/// The UTC calendar day of a local receive timestamp (ms). Falls back to the
/// Unix epoch day for a timestamp out of `time`'s range (unreachable for real
/// receive times).
fn utc_date(recv_ms: i64) -> time::Date {
    time::OffsetDateTime::from_unix_timestamp(recv_ms.div_euclid(1_000))
        .map(time::OffsetDateTime::date)
        .unwrap_or_else(|_| time::OffsetDateTime::UNIX_EPOCH.date())
}

/// Opens (create-or-append) the `binance-depth20-{YYYYMMDD}.jsonl.gz` file for a
/// UTC day. Append mode makes a same-day restart write a second gzip member
/// (RFC-1952 concatenation), transparently readable by `MultiGzDecoder`/`gzip -d`.
fn open_day(dir: &Path, day: time::Date) -> anyhow::Result<DaySegment> {
    let name = format!(
        "binance-depth20-{:04}{:02}{:02}.jsonl.gz",
        day.year(),
        u8::from(day.month()),
        day.day()
    );
    let path = dir.join(name);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening depth capture file {}", path.display()))?;
    Ok(DaySegment {
        encoder: GzEncoder::new(BufWriter::new(file), Compression::default()),
        day,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::io::Read;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, Ordering};

    use feed_util::WsFrame;
    use feed_util::fake::script;

    use super::*;

    const BASE: &str = "wss://stream.binance.com:9443";
    /// 2023-11-14T12:00:00Z, a clean mid-day anchor.
    const DAY0_MS: i64 = 1_700_000_000_000;
    /// Exactly 24 h later — a different UTC calendar day.
    const DAY1_MS: i64 = DAY0_MS + 86_400_000;

    fn params(dir: PathBuf) -> DepthCaptureParams {
        DepthCaptureParams {
            url: combined_depth_url(BASE),
            connect_timeout: Duration::from_secs(5),
            backoff: BackoffParams {
                initial: Duration::from_millis(250),
                max: Duration::from_millis(10_000),
                multiplier: 2.0,
            },
            out_dir: dir,
            channel_capacity: 4096,
        }
    }

    fn depth_frame(symbol: &str) -> WsFrame {
        WsFrame::Text(format!(
            r#"{{"stream":"{symbol}@depth20@100ms","data":{{"lastUpdateId":42,"bids":[["100.00","1.5"],["99.99","2.0"]],"asks":[["100.01","3.0"]]}}}}"#
        ))
    }

    fn unique_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("depth-cap-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Yields enough for the spawned driver task to drain the fake connection's
    /// queued frames without advancing the (paused) tokio clock.
    async fn settle() {
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    }

    fn day_file(dir: &Path, ms: i64) -> PathBuf {
        let day = utc_date(ms);
        dir.join(format!(
            "binance-depth20-{:04}{:02}{:02}.jsonl.gz",
            day.year(),
            u8::from(day.month()),
            day.day()
        ))
    }

    fn read_lines(path: &Path) -> Vec<serde_json::Value> {
        let file = std::fs::File::open(path).unwrap();
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(file)
            .read_to_string(&mut text)
            .unwrap();
        text.lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn combined_url_lists_both_depth_streams() {
        assert_eq!(
            combined_depth_url(BASE),
            "wss://stream.binance.com:9443/stream?streams=\
             btcusdt@depth20@100ms/ethusdt@depth20@100ms"
        );
        // A trailing slash on the base does not double up.
        assert_eq!(
            combined_depth_url("wss://host/"),
            "wss://host/stream?streams=btcusdt@depth20@100ms/ethusdt@depth20@100ms"
        );
    }

    #[test]
    fn depth_line_keeps_payload_and_skips_acks() {
        let line = depth_line(
            123,
            r#"{"stream":"btcusdt@depth20@100ms","data":{"bids":[["1","2"]],"asks":[["3","4"]]}}"#,
        )
        .expect("a stream+data frame yields a line");
        let v: serde_json::Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(v["recv_ms"], 123);
        assert_eq!(v["stream"], "btcusdt@depth20@100ms");
        assert_eq!(v["data"]["bids"][0][0], "1");
        assert_eq!(v["data"]["asks"][0][1], "4");
        // A subscription ack (no stream+data) is skipped.
        assert!(depth_line(1, r#"{"result":null,"id":1}"#).is_none());
        assert!(depth_line(1, r#"{"stream":"x","data":null}"#).is_none());
        assert!(depth_line(1, "not json").is_none());
    }

    /// The headline test: two frames on UTC day 0, the clock advanced across
    /// midnight, a third frame on day 1 — proving per-UTC-day rotation into two
    /// separate files with the payloads preserved verbatim.
    #[tokio::test(start_paused = true)]
    async fn rotates_into_a_new_file_on_a_utc_day_change() {
        let dir = unique_dir("rotate");
        let clock = Arc::new(AtomicI64::new(DAY0_MS));
        let now_fn = {
            let c = Arc::clone(&clock);
            move || TimestampMs::from_millis(c.load(Ordering::SeqCst))
        };
        let (transport, mut handles, _at) = script(&[true]);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = tokio::spawn(run(
            params(dir.clone()),
            transport,
            now_fn,
            shutdown_rx,
            Some(1),
        ));

        let conn = handles.pop_front().unwrap();
        // Day 0: two frames. Settle so the driver stamps them before the clock moves.
        conn.frame_tx.send(depth_frame("btcusdt")).unwrap();
        conn.frame_tx.send(depth_frame("ethusdt")).unwrap();
        settle().await;
        // Cross UTC midnight, then a third frame on day 1.
        clock.store(DAY1_MS, Ordering::SeqCst);
        conn.frame_tx.send(depth_frame("btcusdt")).unwrap();
        settle().await;

        shutdown_tx.send(true).unwrap();
        let stats = task.await.unwrap().unwrap();

        assert_eq!(stats.frames_written, 3);
        assert_eq!(stats.files, 2, "one file per UTC day");
        assert_eq!(stats.frames_dropped, 0);

        let day0 = read_lines(&day_file(&dir, DAY0_MS));
        let day1 = read_lines(&day_file(&dir, DAY1_MS));
        assert_eq!(day0.len(), 2);
        assert_eq!(day1.len(), 1);
        assert_eq!(day0[0]["recv_ms"], DAY0_MS);
        assert_eq!(day0[0]["data"]["bids"][0][0], "100.00");
        assert_eq!(day1[0]["recv_ms"], DAY1_MS);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A clean peer close reconnects to a fresh connection at the same URL, and
    /// frames from both episodes are captured.
    #[tokio::test(start_paused = true)]
    async fn reconnects_on_peer_close() {
        let dir = unique_dir("reconnect");
        let now_fn = || TimestampMs::from_millis(DAY0_MS);
        let (transport, mut handles, _at) = script(&[true, true]);
        let url_log = transport.url_log();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = tokio::spawn(run(
            params(dir.clone()),
            transport,
            now_fn,
            shutdown_rx,
            Some(1),
        ));

        // Episode 1: one frame, then drop the handle (clean peer close).
        let conn1 = handles.pop_front().unwrap();
        conn1.frame_tx.send(depth_frame("btcusdt")).unwrap();
        settle().await;
        drop(conn1);
        // Let the driver observe the peer close and enter its backoff sleep...
        settle().await;
        // ...then elapse the sleep (equal-jitter of 250 ms ≤ 250) so it reconnects.
        tokio::time::advance(Duration::from_millis(500)).await;
        settle().await;

        // Episode 2: another frame.
        let conn2 = handles.pop_front().unwrap();
        conn2.frame_tx.send(depth_frame("ethusdt")).unwrap();
        settle().await;

        shutdown_tx.send(true).unwrap();
        let stats = task.await.unwrap().unwrap();

        assert_eq!(stats.frames_written, 2);
        assert_eq!(stats.reconnects, 1);
        let urls = url_log.lock().unwrap();
        assert!(urls.len() >= 2, "reconnected at least once: {urls:?}");
        assert!(urls.iter().all(|u| u == &combined_depth_url(BASE)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A connect failure backs off (jittered) before the next attempt.
    #[tokio::test(start_paused = true)]
    async fn backs_off_after_a_connect_failure() {
        let dir = unique_dir("backoff");
        let now_fn = || TimestampMs::from_millis(DAY0_MS);
        let (transport, mut handles, attempt_at) = script(&[false, true]);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = tokio::spawn(run(
            params(dir.clone()),
            transport,
            now_fn,
            shutdown_rx,
            Some(7),
        ));

        // First attempt fails immediately; drive time past the backoff floor so
        // the second attempt fires.
        settle().await;
        tokio::time::advance(Duration::from_millis(300)).await;
        settle().await;
        let conn = handles.pop_front().unwrap();
        conn.frame_tx.send(depth_frame("btcusdt")).unwrap();
        settle().await;

        shutdown_tx.send(true).unwrap();
        let _ = task.await.unwrap().unwrap();

        let times = attempt_at.lock().unwrap();
        assert!(times.len() >= 2, "retried after the failure: {times:?}");
        // The second attempt is delayed by at least the equal-jitter floor
        // (raw/2 = 125 ms for the 250 ms initial).
        assert!(
            times[1] >= Duration::from_millis(125),
            "second attempt at {:?} should be backed off",
            times[1]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
