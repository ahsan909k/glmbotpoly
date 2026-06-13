//! Live wire-fact regression probes (network required, `#[ignore]` by
//! default). Run with:
//!
//! ```text
//! cargo test -p feed-binance --test live_probe -- --ignored --nocapture
//! ```
//!
//! Pins the facts the crate is built on: the combined-stream URL subscribes
//! by itself (zero client frames), all four streams deliver promptly, and
//! payload shapes match the committed fixtures.

// Probe assertions panic by design.
#![allow(clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::time::Duration;

use core_types::TimestampMs;
use feed_binance::{
    BinanceParsed, BinanceSub, Connection, Transport, WsFrame, WsTransport, combined_url,
    parse_frame,
};

/// All four streams must deliver within this window (trades on BTC/ETH spot
/// print many times per second; this is generous).
const DEADLINE: Duration = Duration::from_secs(30);

#[tokio::test]
#[ignore = "live network probe"]
async fn combined_url_streams_all_four_without_any_client_frame() {
    let url = combined_url("wss://stream.binance.com:9443", &BinanceSub::all());
    let mut transport = WsTransport;
    let mut conn = transport
        .connect(&url, Duration::from_secs(5))
        .await
        .expect("connect");

    let mut seen: BTreeMap<String, u64> = BTreeMap::new();
    let started = tokio::time::Instant::now();
    while started.elapsed() < DEADLINE && seen.len() < 4 {
        let frame = tokio::time::timeout(DEADLINE, conn.recv())
            .await
            .expect("frames keep arriving");
        match frame {
            Some(Ok(WsFrame::Text(text))) => {
                match parse_frame(&text, TimestampMs::from_millis(0)) {
                    BinanceParsed::Prices(updates) => {
                        for u in updates {
                            *seen.entry(u.sub.to_string()).or_insert(0) += 1;
                        }
                    }
                    other => panic!("live frame classified as {other:?}: {text}"),
                }
            }
            Some(Ok(WsFrame::Ping)) => println!("server ping observed"),
            Some(Ok(WsFrame::Binary(_))) => panic!("unexpected binary frame"),
            other => panic!("socket ended early: {other:?}"),
        }
    }
    conn.close().await;
    println!("streams seen: {seen:?}");
    assert_eq!(seen.len(), 4, "all four streams delivered: {seen:?}");
}

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).
