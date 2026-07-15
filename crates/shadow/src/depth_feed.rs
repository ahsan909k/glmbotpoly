//! Shadow's own Binance `depth20@100ms` subscription — the DEPTH feature input.
//!
//! The engine's `depth_capture` dumps raw frames to gzip and holds nothing in
//! process, so shadow opens its **own** dedicated Binance WebSocket (the same
//! combined-stream form, zero client frames), parses each frame into a
//! [`DepthLadder`], and forwards it over a bounded drop-on-full channel to the
//! [`crate::driver`]. Reuses the shared [`feed_util`] `Transport`/`Backoff` seams
//! and a dead-socket watchdog exactly like `depth_capture`; a failure only
//! reconnects (never reaches a venue, never stalls the engine).

use std::time::Duration;

use core_types::{Asset, TimestampMs};
use feed_util::{Backoff, BackoffParams, Connection, Transport, WsFrame};
use tokio::sync::{mpsc, watch};

use crate::features::depth::DepthLadder;

/// The depth streams captured (fixed order; the combined URL is built from these).
const DEPTH_STREAMS: [&str; 2] = ["btcusdt@depth20@100ms", "ethusdt@depth20@100ms"];
/// No inbound frame (data or server ping) for this long ⇒ half-open socket.
const DEAD_SOCKET_SILENCE: Duration = Duration::from_secs(30);
/// Dead-socket watchdog cadence.
const WATCHDOG_INTERVAL: Duration = Duration::from_secs(1);

/// Builds the Binance combined-stream URL for the depth streams.
#[must_use]
pub fn combined_depth_url(base: &str) -> String {
    format!(
        "{}/stream?streams={}",
        base.trim_end_matches('/'),
        DEPTH_STREAMS.join("/")
    )
}

/// Tuning for the shadow depth feed.
#[derive(Debug, Clone)]
pub struct DepthFeedParams {
    /// The prebuilt combined-stream URL (see [`combined_depth_url`]).
    pub url: String,
    /// WebSocket connect timeout.
    pub connect_timeout: Duration,
    /// Jittered reconnect backoff curve.
    pub backoff: BackoffParams,
}

/// Maps a Binance combined-stream name (`btcusdt@depth20@100ms`) to an asset.
fn asset_of_stream(stream: &str) -> Option<Asset> {
    let symbol = stream.split('@').next()?;
    match symbol {
        "btcusdt" => Some(Asset::Btc),
        "ethusdt" => Some(Asset::Eth),
        _ => None,
    }
}

/// Parses one price/size pair (Binance sends strings; accept numbers too).
fn level(v: &serde_json::Value) -> Option<(f64, f64)> {
    let px = v.get(0)?;
    let sz = v.get(1)?;
    let parse = |x: &serde_json::Value| -> Option<f64> {
        x.as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| x.as_f64())
    };
    Some((parse(px)?, parse(sz)?))
}

/// Parses a combined-stream depth frame into a [`DepthLadder`], or `None` for a
/// non-depth / malformed frame (subscription acks are skipped).
fn parse_ladder(text: &str, recv_ms: i64) -> Option<DepthLadder> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let asset = asset_of_stream(value.get("stream")?.as_str()?)?;
    let data = value.get("data")?;
    let (mut bid_px, mut bid_sz, mut ask_px, mut ask_sz) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for lv in data.get("bids")?.as_array()? {
        if let Some((p, s)) = level(lv) {
            bid_px.push(p);
            bid_sz.push(s);
        }
    }
    for lv in data.get("asks")?.as_array()? {
        if let Some((p, s)) = level(lv) {
            ask_px.push(p);
            ask_sz.push(s);
        }
    }
    Some(DepthLadder {
        asset,
        recv_ms,
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
    })
}

/// Why a streaming episode ended.
enum EpisodeEnd {
    Reconnect(String),
    Shutdown,
}

/// Connects, streams `depth20@100ms` ladders to `ladder_tx`, and reconnects
/// forever until `shutdown_rx` flips. Generic over [`Transport`] for tests.
pub async fn run<T, F>(
    params: DepthFeedParams,
    mut transport: T,
    now_fn: F,
    ladder_tx: mpsc::Sender<DepthLadder>,
    mut shutdown_rx: watch::Receiver<bool>,
    backoff_seed: Option<u64>,
) where
    T: Transport,
    F: Fn() -> TimestampMs + Send,
{
    let seed = backoff_seed.unwrap_or_else(|| now_fn().as_millis().unsigned_abs());
    let mut backoff = Backoff::new(params.backoff, seed);
    tracing::info!(target: "shadow", url = %params.url, "depth20 feed starting");

    loop {
        if *shutdown_rx.borrow() || shutdown_rx.has_changed().is_err() {
            break;
        }
        let connect = tokio::select! {
            r = transport.connect(&params.url, params.connect_timeout) => r,
            _ = shutdown_rx.changed() => continue,
        };
        let mut conn = match connect {
            Ok(conn) => conn,
            Err(error) => {
                let delay = backoff.next_delay();
                tracing::warn!(target: "shadow", %error, retry_in_ms = delay.as_millis() as u64, "depth connect failed");
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
                continue;
            }
        };
        match stream_episode(
            &mut conn,
            &now_fn,
            &ladder_tx,
            &mut backoff,
            &mut shutdown_rx,
        )
        .await
        {
            EpisodeEnd::Shutdown => {
                conn.close().await;
                break;
            }
            EpisodeEnd::Reconnect(reason) => {
                conn.close().await;
                let delay = backoff.next_delay();
                tracing::warn!(target: "shadow", reason = %reason, retry_in_ms = delay.as_millis() as u64, "depth reconnecting");
                sleep_or_shutdown(delay, &mut shutdown_rx).await;
            }
        }
    }
}

async fn stream_episode<C, F>(
    conn: &mut C,
    now_fn: &F,
    ladder_tx: &mpsc::Sender<DepthLadder>,
    backoff: &mut Backoff,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> EpisodeEnd
where
    C: Connection,
    F: Fn() -> TimestampMs,
{
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
                    Some(Err(error)) => return EpisodeEnd::Reconnect(format!("socket error: {error}")),
                    Some(Ok(frame)) => {
                        last_inbound = now;
                        let text = match frame {
                            WsFrame::Text(t) => t,
                            WsFrame::Binary(b) => String::from_utf8_lossy(&b).into_owned(),
                            WsFrame::Ping => continue,
                        };
                        if !backoff_reset {
                            backoff.reset();
                            backoff_reset = true;
                        }
                        if let Some(ladder) = parse_ladder(&text, now.as_millis()) {
                            // Drop-on-full: shadow lag must never back-pressure the socket.
                            let _ = ladder_tx.try_send(ladder);
                        }
                    }
                }
            }
            _ = watchdog.tick() => {
                if now_fn().signed_duration_since(last_inbound).as_millis()
                    >= DEAD_SOCKET_SILENCE.as_millis() as i64
                {
                    return EpisodeEnd::Reconnect("dead socket".to_owned());
                }
            }
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    return EpisodeEnd::Shutdown;
                }
            }
        }
    }
}

async fn sleep_or_shutdown(delay: Duration, shutdown_rx: &mut watch::Receiver<bool>) {
    tokio::select! {
        () = tokio::time::sleep(delay) => {}
        _ = shutdown_rx.changed() => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_lists_both_streams() {
        assert_eq!(
            combined_depth_url("wss://stream.binance.com:9443"),
            "wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@100ms/ethusdt@depth20@100ms"
        );
    }

    #[test]
    fn parses_a_depth_frame() {
        let text = r#"{"stream":"btcusdt@depth20@100ms","data":{"lastUpdateId":1,"bids":[["100.00","1.5"],["99.99","2.0"]],"asks":[["100.01","3.0"]]}}"#;
        let l = parse_ladder(text, 1234).expect("valid depth frame");
        assert_eq!(l.asset, Asset::Btc);
        assert_eq!(l.recv_ms, 1234);
        assert_eq!(l.bid_px, vec![100.00, 99.99]);
        assert_eq!(l.ask_sz, vec![3.0]);
    }

    #[test]
    fn skips_non_depth_frames() {
        assert!(parse_ladder(r#"{"result":null,"id":1}"#, 0).is_none());
        assert!(parse_ladder("not json", 0).is_none());
        // Unknown symbol → skipped.
        assert!(
            parse_ladder(
                r#"{"stream":"solusdt@depth20@100ms","data":{"bids":[],"asks":[]}}"#,
                0
            )
            .is_none()
        );
    }
}
