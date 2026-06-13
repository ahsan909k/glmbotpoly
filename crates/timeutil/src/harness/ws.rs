//! Shared WebSocket probe: connect time, subscribe→first-data time, and
//! inter-message gap distribution over a fixed window. One function serves
//! both the CLOB market channel and RTDS — only the subscribe message, the
//! keepalive text, and the labels differ.
//!
//! Wire facts (docs re-verified 2026-06-12):
//! - Market channel: subscribe `{"assets_ids":[…],"type":"market",
//!   "custom_feature_enabled":true}`; client sends text `"PING"` every 10 s.
//! - RTDS: subscribe `{"action":"subscribe","subscriptions":[{"topic":
//!   "crypto_prices","type":"update","filters":"btcusdt"}]}`; text `"PING"`
//!   every 5 s.

use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::harness::report::WsProbeReport;
use crate::harness::rest::duration_ms;
use crate::stats::stats_from_ms;

/// Everything one WS probe needs.
pub(crate) struct WsProbeSpec {
    /// Report label (e.g. `"clob-market"`).
    pub label: String,
    /// Socket URL.
    pub url: String,
    /// Exact subscribe message text (already JSON-encoded).
    pub subscribe: String,
    /// Keepalive text + interval (both Polymarket sockets want text PINGs).
    pub ping: (String, Duration),
    /// TCP+TLS+WS handshake deadline.
    pub connect_timeout: Duration,
    /// Gap-measurement window (after the first data message).
    pub duration: Duration,
    /// Whether gap stats reflect venue activity rather than the network.
    pub activity_dependent: bool,
}

/// Builds the CLOB market-channel subscribe message (§7 wire shape).
#[must_use]
pub fn market_subscribe_message(asset_ids: &[String]) -> String {
    // Hand-assembled via serde_json values to keep escaping correct without
    // defining wire structs for a one-message protocol.
    serde_json::json!({
        "assets_ids": asset_ids,
        "type": "market",
        "custom_feature_enabled": true,
    })
    .to_string()
}

/// Builds the RTDS subscribe message for Binance-sourced crypto prices.
///
/// The `filters` field is a **JSON-encoded string** (`"{\"symbol\":...}"`),
/// NOT the plain comma-list the docs show for `crypto_prices` — verified
/// live 2026-06-12: the plain form yields an ack and then silence, the
/// JSON-string form yields the 2-minute backfill plus ~1 update/second
/// (Decisions Log).
#[must_use]
pub fn rtds_crypto_subscribe_message(symbol: &str) -> String {
    let filters = serde_json::json!({ "symbol": symbol }).to_string();
    serde_json::json!({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices",
            "type": "update",
            "filters": filters,
        }],
    })
    .to_string()
}

/// Runs one probe. Never fails: errors land in the report.
pub(crate) async fn probe_ws(spec: WsProbeSpec) -> WsProbeReport {
    let mut report = WsProbeReport {
        label: spec.label.clone(),
        url: spec.url.clone(),
        activity_dependent: spec.activity_dependent,
        connect_ms: None,
        subscribe_to_first_msg_ms: None,
        messages: 0,
        duration_ms: 0,
        gaps: None,
        error: None,
    };

    let connect_started = Instant::now();
    let connected = tokio::time::timeout(
        spec.connect_timeout,
        tokio_tungstenite::connect_async(&spec.url),
    )
    .await;
    let mut socket = match connected {
        Ok(Ok((socket, _response))) => socket,
        Ok(Err(error)) => {
            report.error = Some(format!("connect failed: {error}"));
            return report;
        }
        Err(_) => {
            report.error = Some(format!(
                "connect timed out after {:?}",
                spec.connect_timeout
            ));
            return report;
        }
    };
    report.connect_ms = Some(duration_ms(connect_started.elapsed()));

    let subscribe_sent = Instant::now();
    if let Err(error) = socket.send(Message::text(spec.subscribe.clone())).await {
        report.error = Some(format!("subscribe send failed: {error}"));
        return report;
    }

    // Phase 1: wait (up to the connect timeout) for the first data message.
    let first_deadline = tokio::time::Instant::now() + spec.connect_timeout;
    loop {
        let next = tokio::time::timeout_at(first_deadline, socket.next()).await;
        match next {
            Ok(Some(Ok(message))) => {
                tracing::debug!(
                    target: "timeutil::harness",
                    label = %spec.label,
                    preview = preview(&message),
                    "first frame after subscribe"
                );
                if is_data(&message) {
                    report.subscribe_to_first_msg_ms = Some(duration_ms(subscribe_sent.elapsed()));
                    report.messages = 1;
                    break;
                }
                // Control/PONG traffic before the first data message: keep
                // waiting.
            }
            Ok(Some(Err(error))) => {
                report.error = Some(format!("socket error before first message: {error}"));
                return report;
            }
            Ok(None) => {
                report.error = Some("socket closed before first message".to_owned());
                return report;
            }
            Err(_) => {
                report.error = Some(format!(
                    "no data message within {:?} of subscribing",
                    spec.connect_timeout
                ));
                return report;
            }
        }
    }

    // Phase 2: measure inter-message gaps for the configured window,
    // keeping the connection alive with text PINGs.
    let measure_started = Instant::now();
    let window_deadline = tokio::time::Instant::now() + spec.duration;
    let (ping_text, ping_every) = spec.ping;
    let mut ping_interval = tokio::time::interval(ping_every);
    ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ping_interval.reset(); // don't ping immediately
    let mut last_data = Instant::now();
    let mut gaps: Vec<f64> = Vec::new();

    loop {
        tokio::select! {
            () = tokio::time::sleep_until(window_deadline) => break,
            _ = ping_interval.tick() => {
                if let Err(error) = socket.send(Message::text(ping_text.clone())).await {
                    report.error = Some(format!("ping send failed: {error}"));
                    break;
                }
            }
            next = socket.next() => match next {
                Some(Ok(message)) if is_data(&message) => {
                    gaps.push(duration_ms(last_data.elapsed()));
                    last_data = Instant::now();
                    report.messages = report.messages.saturating_add(1);
                }
                Some(Ok(_)) => {} // PONG / control frames: not data
                Some(Err(error)) => {
                    report.error = Some(format!("socket error mid-measurement: {error}"));
                    break;
                }
                None => {
                    report.error = Some("socket closed mid-measurement".to_owned());
                    break;
                }
            }
        }
    }

    report.duration_ms = measure_started
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    report.gaps = stats_from_ms(&mut gaps);
    let _ = socket.close(None).await;
    report
}

/// Short loggable preview of a frame (debug aid for protocol surprises).
fn preview(message: &Message) -> String {
    match message {
        Message::Text(text) => {
            let t = text.trim();
            t.chars().take(160).collect()
        }
        other => format!("{other:?}").chars().take(160).collect(),
    }
}

/// True for messages that count as data (and toward gap stats): non-empty
/// text frames that are not the server's `"PONG"` keepalive reply. RTDS
/// acks a subscription with an empty text frame (observed live 2026-06-12)
/// — not data.
fn is_data(message: &Message) -> bool {
    match message {
        Message::Text(text) => {
            let t = text.trim();
            !t.is_empty() && t != "PONG"
        }
        Message::Binary(_) => true,
        _ => false, // protocol Ping/Pong/Close/Frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn market_subscribe_message_is_pinned_to_docs() {
        let msg = market_subscribe_message(&["111".to_owned(), "222".to_owned()]);
        assert_eq!(
            msg,
            r#"{"assets_ids":["111","222"],"custom_feature_enabled":true,"type":"market"}"#
        );
    }

    #[test]
    fn rtds_subscribe_message_is_pinned_to_live_verified_shape() {
        // JSON-string filter, NOT the docs' plain comma-list — see the
        // builder docs + Decisions Log 2026-06-12.
        let msg = rtds_crypto_subscribe_message("btcusdt");
        assert_eq!(
            msg,
            r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"btcusdt\"}","topic":"crypto_prices","type":"update"}]}"#
        );
    }

    #[test]
    fn pong_empty_and_control_frames_are_not_data() {
        assert!(!is_data(&Message::text("PONG")));
        assert!(!is_data(&Message::text(" PONG ")));
        assert!(!is_data(&Message::text(""))); // RTDS subscription ack
        assert!(!is_data(&Message::text("  ")));
        assert!(!is_data(&Message::Pong(vec![].into())));
        assert!(!is_data(&Message::Ping(vec![].into())));
        assert!(is_data(&Message::text(r#"{"event_type":"book"}"#)));
        assert!(is_data(&Message::Binary(vec![1].into())));
    }
}
