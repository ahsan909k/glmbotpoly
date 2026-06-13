//! Live RTDS probes (ignored by default — network + ~10 s each). These
//! discovered the wire truths recorded in `wire.rs` (2026-06-12):
//! - filters are held per (connection, topic) and REPLACED by later
//!   subscribes — variants A/B/E (multi-symbol filters in every plausible
//!   form) are acked and deliver nothing; C (two entries, one message)
//!   leaves only one symbol live;
//! - the only multi-symbol form is subscribe-to-all: D (Binance, no
//!   `filters` field) and F (Chainlink, `filters:""` + type `"*"`) each
//!   stream every symbol at ~1/s — ~13 msg/s total, filtered client-side.
//!
//! Keep these runnable: they are the regression check if RTDS subscription
//! semantics ever change.
//!
//! Run: `cargo test -p feed-rtds --test live_probe -- --ignored --nocapture`

// Diagnostic tool, not runtime code: panicking on a failed live connect is
// the desired behavior (the helper isn't a #[test] fn, so the clippy.toml
// test exemption doesn't reach it).
#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::time::Duration;

use feed_rtds::{Connection, Transport, WsFrame, WsTransport};

const URL: &str = "wss://ws-live-data.polymarket.com";

/// Connects, sends each message in order, listens for `secs`, and returns
/// counts of type:"update" frames per symbol (other frame types counted
/// under "<type:SYMBOL>").
async fn probe_seq(messages: &[String], secs: u64) -> BTreeMap<String, u64> {
    let mut transport = WsTransport;
    let mut conn = transport
        .connect(URL, Duration::from_secs(5))
        .await
        .expect("connect");
    for message in messages {
        conn.send_text(message).await.expect("subscribe");
    }
    let counts = listen(&mut conn, secs).await;
    conn.close().await;
    counts
}

/// Connects, sends `subscribe_text`, listens for `secs`, and returns
/// counts of type:"update" frames per symbol (backfill "subscribe" frames
/// counted separately under "<backfill:SYMBOL>").
async fn probe(subscribe_text: &str, secs: u64) -> BTreeMap<String, u64> {
    probe_seq(&[subscribe_text.to_owned()], secs).await
}

/// Counts frames per symbol until the deadline.
async fn listen(conn: &mut <WsTransport as Transport>::Conn, secs: u64) -> BTreeMap<String, u64> {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    loop {
        let frame = tokio::select! {
            f = conn.recv() => f,
            () = tokio::time::sleep_until(deadline) => break,
        };
        let Some(Ok(WsFrame::Text(text))) = frame else {
            if frame.is_none() {
                println!("  !! peer closed early");
                break;
            }
            continue;
        };
        if text.trim().is_empty() || text.trim() == "PONG" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let msg_type = v["type"].as_str().unwrap_or("?");
        let symbol = v["payload"]["symbol"].as_str().unwrap_or("?");
        let key = if msg_type == "update" {
            symbol.to_owned()
        } else {
            format!("<{msg_type}:{symbol}>")
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_a_comma_list_inside_json_filter() {
    let msg = r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"btcusdt,ethusdt\"}","topic":"crypto_prices","type":"update"}]}"#;
    println!("A comma-in-json: {:?}", probe(msg, 8).await);
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_b_array_inside_json_filter() {
    let msg = r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":[\"btcusdt\",\"ethusdt\"]}","topic":"crypto_prices","type":"update"}]}"#;
    println!("B array-in-json: {:?}", probe(msg, 8).await);
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_c_two_entries_one_message() {
    let msg = r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"btcusdt\"}","topic":"crypto_prices","type":"update"},{"filters":"{\"symbol\":\"ethusdt\"}","topic":"crypto_prices","type":"update"}]}"#;
    println!("C two-entries: {:?}", probe(msg, 8).await);
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_d_no_filter_all_symbols() {
    let msg =
        r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices","type":"update"}]}"#;
    println!("D no-filter: {:?}", probe(msg, 8).await);
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_e_chainlink_comma_list() {
    let msg = r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"btc/usd,eth/usd\"}","topic":"crypto_prices_chainlink","type":"*"}]}"#;
    println!("E chainlink comma-in-json: {:?}", probe(msg, 8).await);
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_f_chainlink_empty_filter_all() {
    let msg = r#"{"action":"subscribe","subscriptions":[{"filters":"","topic":"crypto_prices_chainlink","type":"*"}]}"#;
    println!("F chainlink all: {:?}", probe(msg, 8).await);
}

// --- Sequence probes: does the all-symbols subscribe REPLACE an existing
// filtered slot? (Observed in `bot feed` 2026-06-12: Binance yes, Chainlink
// filters:"" NO — eth/usd, the last filtered subscribe, kept the slot.)

fn filtered(topic: &str, sub_type: &str, symbol: &str) -> String {
    let filters = serde_json::json!({ "symbol": symbol }).to_string();
    serde_json::json!({
        "action": "subscribe",
        "subscriptions": [{ "topic": topic, "type": sub_type, "filters": filters }],
    })
    .to_string()
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_g_chainlink_backfills_then_no_filter_field() {
    // Like Binance's working form: omit `filters` entirely.
    let messages = vec![
        filtered("crypto_prices_chainlink", "*", "btc/usd"),
        filtered("crypto_prices_chainlink", "*", "eth/usd"),
        r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices_chainlink","type":"*"}]}"#.to_owned(),
    ];
    println!(
        "G chainlink no-field after backfills: {:?}",
        probe_seq(&messages, 8).await
    );
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_h_chainlink_backfills_then_empty_filter() {
    // The shape `bot feed` used — expected to leave only eth/usd live.
    let messages = vec![
        filtered("crypto_prices_chainlink", "*", "btc/usd"),
        filtered("crypto_prices_chainlink", "*", "eth/usd"),
        r#"{"action":"subscribe","subscriptions":[{"filters":"","topic":"crypto_prices_chainlink","type":"*"}]}"#.to_owned(),
    ];
    println!(
        "H chainlink empty-filter after backfills: {:?}",
        probe_seq(&messages, 8).await
    );
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_i_chainlink_backfills_unsubscribe_then_all() {
    // Clear the slot first, then subscribe-all on the clean slot.
    let messages = vec![
        filtered("crypto_prices_chainlink", "*", "btc/usd"),
        filtered("crypto_prices_chainlink", "*", "eth/usd"),
        r#"{"action":"unsubscribe","subscriptions":[{"filters":"","topic":"crypto_prices_chainlink","type":"*"}]}"#.to_owned(),
        r#"{"action":"subscribe","subscriptions":[{"filters":"","topic":"crypto_prices_chainlink","type":"*"}]}"#.to_owned(),
    ];
    println!(
        "I chainlink unsub-then-all: {:?}",
        probe_seq(&messages, 8).await
    );
}

#[tokio::test]
#[ignore = "live network probe"]
async fn variant_j_binance_backfills_then_no_filter_field() {
    // Confirm the Binance sequence `bot feed` uses (worked live).
    let messages = vec![
        filtered("crypto_prices", "update", "btcusdt"),
        filtered("crypto_prices", "update", "ethusdt"),
        r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices","type":"update"}]}"#
            .to_owned(),
    ];
    println!(
        "J binance no-field after backfills: {:?}",
        probe_seq(&messages, 8).await
    );
}
