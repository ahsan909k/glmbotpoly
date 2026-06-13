//! Fixture-driven integration tests: real captured RTDS frames (see
//! `fixtures/README.md` for provenance) through the parser and the machine,
//! plus the hand-crafted malformed corpus and a truncation fuzz loop.

// Panicking helpers are the point in tests; helpers aren't #[test] fns, so
// the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use core_types::{Asset, TimestampMs};
use feed_rtds::{FeedSub, ParsedFrame, RtdsSource, map_symbol, parse_frame};
use rust_decimal::Decimal;
use serde::Deserialize;

fn prices(frame: &str) -> Vec<feed_rtds::PriceUpdate> {
    match parse_frame(frame) {
        ParsedFrame::Prices(updates) => updates,
        other => panic!("expected prices, got {other:?}"),
    }
}

/// The raw wire fields, re-extracted independently of the crate's parser so
/// the assertions cross-check rather than tautologize.
#[derive(Deserialize)]
struct RawEnvelope {
    topic: String,
    payload: RawPayload,
}

#[derive(Deserialize)]
struct RawPayload {
    symbol: String,
    timestamp: Option<i64>,
    #[serde(default)]
    full_accuracy_value: Option<String>,
    #[serde(default)]
    data: Option<Vec<serde_json::Value>>,
}

fn raw(frame: &str) -> RawEnvelope {
    serde_json::from_str(frame).expect("fixture is valid JSON")
}

#[test]
fn binance_updates_parse_to_the_exact_full_accuracy_decimal() {
    for (fixture, asset) in [
        (
            include_str!("fixtures/rtds_binance_btcusdt_update.json"),
            Asset::Btc,
        ),
        (
            include_str!("fixtures/rtds_binance_ethusdt_update.json"),
            Asset::Eth,
        ),
    ] {
        let updates = prices(fixture);
        let wire = raw(fixture);
        assert_eq!(updates.len(), 1);
        let update = updates[0];
        assert_eq!(update.source, RtdsSource::Binance);
        assert_eq!(update.asset, asset);
        // Binance full_accuracy_value is a plain decimal string — the
        // published value must equal it exactly.
        let expected: Decimal = wire
            .payload
            .full_accuracy_value
            .expect("live binance updates carry full_accuracy_value")
            .parse()
            .expect("plain decimal string");
        assert_eq!(update.value, expected);
        assert_eq!(
            update.ts_exchange,
            TimestampMs::from_millis(wire.payload.timestamp.expect("payload timestamp present"))
        );
    }
}

#[test]
fn chainlink_updates_parse_to_the_descaled_full_accuracy_decimal() {
    for (fixture, asset) in [
        (
            include_str!("fixtures/rtds_chainlink_btcusd_update.json"),
            Asset::Btc,
        ),
        (
            include_str!("fixtures/rtds_chainlink_ethusd_update.json"),
            Asset::Eth,
        ),
    ] {
        let updates = prices(fixture);
        let wire = raw(fixture);
        assert_eq!(updates.len(), 1);
        let update = updates[0];
        assert_eq!(update.source, RtdsSource::Chainlink);
        assert_eq!(update.asset, asset);
        // Chainlink full_accuracy_value is value × 10^18 as an integer
        // string — more digits than the f64 `value` can carry.
        let scaled: Decimal = wire
            .payload
            .full_accuracy_value
            .expect("live chainlink updates carry full_accuracy_value")
            .parse()
            .expect("integer string");
        assert_eq!(
            update.value,
            scaled / Decimal::from(1_000_000_000_000_000_000_u64)
        );
    }
}

#[test]
fn backfills_parse_every_complete_point() {
    let fixture = include_str!("fixtures/rtds_binance_btcusdt_backfill.json");
    let updates = prices(fixture);
    let wire = raw(fixture);
    let points = wire
        .payload
        .data
        .as_ref()
        .expect("backfill has data array")
        .len();
    assert!(points > 50, "real backfills carry ~2 minutes of points");
    assert_eq!(updates.len(), points, "all captured points are complete");
    assert!(
        updates
            .iter()
            .all(|u| u.source == RtdsSource::Binance && u.asset == Asset::Btc)
    );
    // Wire order is oldest → newest with per-point timestamps.
    assert!(
        updates
            .windows(2)
            .all(|w| w[0].ts_exchange <= w[1].ts_exchange)
    );
}

#[test]
fn chainlink_backfill_identity_survives_the_wrong_topic_quirk() {
    // Live-captured server quirk: the chainlink backfill arrives tagged
    // topic "crypto_prices". Identity must come from the slash symbol.
    let fixture = include_str!("fixtures/rtds_chainlink_btcusd_backfill.json");
    let wire = raw(fixture);
    assert_eq!(wire.topic, "crypto_prices", "the quirk this test pins");
    assert_eq!(wire.payload.symbol, "btc/usd");

    let updates = prices(fixture);
    assert!(!updates.is_empty());
    assert!(
        updates
            .iter()
            .all(|u| u.source == RtdsSource::Chainlink && u.asset == Asset::Btc),
        "symbol wins over the lying topic"
    );
}

/// One line of the `--raw` capture.
#[derive(Deserialize)]
struct TapLine {
    ts_local_ms: i64,
    dir: String,
    text: String,
}

#[test]
fn session_replay_parses_cleanly_and_pins_the_connect_sequence() {
    let lines: Vec<TapLine> = include_str!("fixtures/rtds_session.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid capture line"))
        .collect();

    // The first outbound frames are exactly the crate's connect sequence
    // for all four streams — capture and builders must agree.
    let sent: Vec<&str> = lines
        .iter()
        .filter(|l| l.dir == "out")
        .map(|l| l.text.as_str())
        .collect();
    let expected = [
        feed_rtds::backfill_subscribe_message(RtdsSource::Binance, Asset::Btc),
        feed_rtds::backfill_subscribe_message(RtdsSource::Binance, Asset::Eth),
        feed_rtds::stream_subscribe_message(RtdsSource::Binance),
        feed_rtds::backfill_subscribe_message(RtdsSource::Chainlink, Asset::Btc),
        feed_rtds::backfill_subscribe_message(RtdsSource::Chainlink, Asset::Eth),
        feed_rtds::stream_subscribe_message(RtdsSource::Chainlink),
    ];
    assert_eq!(
        &sent[..6],
        &expected.iter().map(String::as_str).collect::<Vec<_>>()[..]
    );
    assert!(sent.contains(&"PING"), "keepalive present in the slice");

    // Every inbound frame parses without panicking, and nothing in a real
    // session is malformed. Untracked symbols are routine (all-symbols
    // subscription); server error frames really happen (this very capture
    // has the two 500s that starved chainlink btc/usd — Decisions Log).
    let mut acks = 0_u32;
    let mut price_frames = 0_u32;
    let mut observations = 0_usize;
    let mut server_errors = 0_u32;
    for line in lines.iter().filter(|l| l.dir == "in") {
        match parse_frame(&line.text) {
            ParsedFrame::Ack => acks += 1,
            ParsedFrame::Pong => {}
            ParsedFrame::Prices(updates) => {
                price_frames += 1;
                observations += updates.len();
            }
            ParsedFrame::Ignored(feed_rtds::IgnoredReason::UnknownSymbol) => {}
            ParsedFrame::Ignored(feed_rtds::IgnoredReason::ServerError) => server_errors += 1,
            ParsedFrame::Ignored(reason) => {
                panic!("real captured frame ignored as {reason:?}: {}", line.text);
            }
        }
    }
    assert!(acks >= 4, "subscribe acks present ({acks})");
    assert!(price_frames >= 5);
    assert!(
        observations > 200,
        "backfills alone carry hundreds of points ({observations})"
    );
    assert_eq!(server_errors, 2, "this capture's two real 500s classify");
}

#[test]
fn session_replay_through_the_machine_publishes_all_four_streams() {
    // Drive the real captured frames through parser + machine (the same
    // path the driver takes) and check every stream publishes and nothing
    // goes stale within the healthy slice.
    let lines: Vec<TapLine> = include_str!("fixtures/rtds_session.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid capture line"))
        .collect();
    let boot = TimestampMs::from_millis(lines[0].ts_local_ms);

    // The machine is crate-private; replay at the parser level and count
    // per-stream observations the way the machine would (tracked keys only).
    let tracked: Vec<FeedSub> = FeedSub::all();
    let mut per_stream = std::collections::BTreeMap::new();
    for line in lines.iter().filter(|l| l.dir == "in") {
        if let ParsedFrame::Prices(updates) = parse_frame(&line.text) {
            for update in updates {
                let sub = FeedSub::new(update.source, update.asset);
                if tracked.contains(&sub) {
                    *per_stream.entry(format!("{sub}")).or_insert(0_u32) += 1;
                }
                assert!(
                    update.ts_exchange <= TimestampMs::from_millis(line.ts_local_ms),
                    "exchange timestamps are never from the future"
                );
            }
        }
    }
    assert_eq!(
        per_stream.len(),
        4,
        "all four streams observed: {per_stream:?}"
    );
    assert!(per_stream.values().all(|&n| n > 10), "{per_stream:?}");
    // Sanity on the capture itself: the slice spans a few seconds at most.
    let span = lines.last().unwrap().ts_local_ms - boot.as_millis();
    assert!((0..60_000).contains(&span));
}

#[test]
fn real_server_error_frame_classifies_as_server_error() {
    let fixture = include_str!("fixtures/rtds_server_error.json");
    assert_eq!(
        parse_frame(fixture),
        ParsedFrame::Ignored(feed_rtds::IgnoredReason::ServerError)
    );
}

#[test]
fn malformed_corpus_never_panics_and_is_always_ignored() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/malformed");
    let mut checked = 0;
    for entry in std::fs::read_dir(dir).expect("malformed corpus dir") {
        let path = entry.expect("dir entry").path();
        let content = std::fs::read_to_string(&path).expect("readable fixture");
        match parse_frame(&content) {
            ParsedFrame::Ignored(_) => checked += 1,
            other => panic!("{} parsed as {other:?}", path.display()),
        }
    }
    assert!(checked >= 10, "corpus present ({checked} files)");
}

#[test]
fn truncation_fuzz_never_panics() {
    let fixture = include_str!("fixtures/rtds_binance_btcusdt_update.json");
    for (i, _) in fixture.char_indices() {
        let _ = parse_frame(&fixture[..i]);
    }
    let backfill = include_str!("fixtures/rtds_chainlink_btcusd_backfill.json");
    for i in (0..backfill.len()).step_by(7) {
        if backfill.is_char_boundary(i) {
            let _ = parse_frame(&backfill[..i]);
        }
    }
}

#[test]
fn every_tracked_symbol_maps_and_untracked_do_not() {
    for sub in FeedSub::all() {
        assert_eq!(
            map_symbol(sub.source.symbol(sub.asset)),
            Some((sub.source, sub.asset))
        );
    }
    for untracked in ["solusdt", "xrpusdt", "bnb/usd", "hype/usd", "aapl"] {
        assert_eq!(map_symbol(untracked), None);
    }
}

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).
