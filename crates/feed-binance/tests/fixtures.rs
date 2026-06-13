//! Fixture-driven integration tests: real captured Binance frames (see
//! `fixtures/README.md` for provenance) through the parser, plus the
//! hand-crafted malformed corpus and a truncation fuzz loop.

// Panicking helpers are the point in tests; helpers aren't #[test] fns, so
// the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use core_types::{Asset, Decimal, TimestampMs};
use feed_binance::{BinanceParsed, BinanceStream, BinanceSub, BinanceUpdate, parse_frame};
use serde::Deserialize;

const NOW: TimestampMs = TimestampMs::from_millis(1_781_247_315_000);

fn prices(frame: &str, now: TimestampMs) -> Vec<BinanceUpdate> {
    match parse_frame(frame, now) {
        BinanceParsed::Prices(updates) => updates,
        other => panic!("expected prices, got {other:?}"),
    }
}

/// The raw wire fields, re-extracted independently of the crate's parser so
/// the assertions cross-check rather than tautologize.
#[derive(Deserialize)]
struct RawWrapper {
    stream: String,
    data: serde_json::Value,
}

fn raw(frame: &str) -> RawWrapper {
    serde_json::from_str(frame).expect("fixture is valid JSON")
}

#[test]
fn book_ticker_samples_publish_the_exact_midpoint() {
    for (fixture, asset) in [
        (
            include_str!("fixtures/binance_bookticker_btcusdt.json"),
            Asset::Btc,
        ),
        (
            include_str!("fixtures/binance_bookticker_ethusdt.json"),
            Asset::Eth,
        ),
    ] {
        let updates = prices(fixture, NOW);
        assert_eq!(updates.len(), 1);
        let update = updates[0];
        assert_eq!(
            update.sub,
            BinanceSub::new(asset, BinanceStream::BookTicker)
        );

        // Cross-check the midpoint against independently re-extracted sides.
        let wire = raw(fixture);
        assert!(wire.stream.ends_with("@bookTicker"));
        let bid: Decimal = wire.data["b"].as_str().unwrap().parse().unwrap();
        let ask: Decimal = wire.data["a"].as_str().unwrap().parse().unwrap();
        assert_eq!(update.value, (bid + ask) / Decimal::TWO);

        // No event time exists on the wire — the tick is stamped at `now`.
        assert_eq!(update.ts_exchange, NOW);
    }
}

#[test]
fn book_ticker_really_lacks_event_fields_on_the_live_wire() {
    // The doc claim this crate's `ts_exchange := ts_local` decision rests on,
    // pinned against real captured frames.
    for fixture in [
        include_str!("fixtures/binance_bookticker_btcusdt.json"),
        include_str!("fixtures/binance_bookticker_ethusdt.json"),
    ] {
        let wire = raw(fixture);
        let data = wire.data.as_object().expect("payload object");
        assert!(!data.contains_key("e"), "no event type");
        assert!(!data.contains_key("E"), "no event time");
        for key in ["u", "s", "b", "B", "a", "A"] {
            assert!(data.contains_key(key), "documented field {key} present");
        }
    }
}

#[test]
fn trade_samples_publish_the_print_at_trade_time() {
    for (fixture, asset) in [
        (
            include_str!("fixtures/binance_trade_btcusdt.json"),
            Asset::Btc,
        ),
        (
            include_str!("fixtures/binance_trade_ethusdt.json"),
            Asset::Eth,
        ),
    ] {
        let updates = prices(fixture, NOW);
        assert_eq!(updates.len(), 1);
        let update = updates[0];
        assert_eq!(update.sub, BinanceSub::new(asset, BinanceStream::Trade));

        let wire = raw(fixture);
        assert!(wire.stream.ends_with("@trade"));
        assert_eq!(wire.data["e"].as_str(), Some("trade"));
        let price: Decimal = wire.data["p"].as_str().unwrap().parse().unwrap();
        assert_eq!(update.value, price);
        let trade_time = wire.data["T"].as_i64().expect("trade time present");
        assert_eq!(update.ts_exchange, TimestampMs::from_millis(trade_time));
    }
}

/// One line of the `--raw` capture.
#[derive(Deserialize)]
struct TapLine {
    ts_local_ms: i64,
    dir: String,
    text: String,
}

#[test]
fn session_replay_parses_cleanly_with_zero_outbound_frames() {
    let lines: Vec<TapLine> = include_str!("fixtures/binance_session.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid capture line"))
        .collect();
    assert!(lines.len() >= 300, "real session slice present");

    // URL-based subscription: the client NEVER sends an application frame.
    assert!(
        lines.iter().all(|l| l.dir == "in"),
        "zero outbound frames in a Binance session"
    );

    // Every frame of a healthy session parses to exactly one price
    // observation — no acks, no errors, nothing ignored.
    let mut per_stream = std::collections::BTreeMap::new();
    for line in &lines {
        let now = TimestampMs::from_millis(line.ts_local_ms);
        let updates = match parse_frame(&line.text, now) {
            BinanceParsed::Prices(updates) => updates,
            other => panic!("real captured frame classified as {other:?}: {}", line.text),
        };
        assert_eq!(updates.len(), 1);
        let update = updates[0];
        *per_stream.entry(update.sub.to_string()).or_insert(0_u32) += 1;
        // Trade times come from Binance's clock; receive times from ours —
        // they must agree to within ordinary skew + transit.
        if update.sub.stream == BinanceStream::Trade {
            let drift = (update.ts_exchange.as_millis() - line.ts_local_ms).abs();
            assert!(
                drift < 10_000,
                "trade time within 10s of receipt ({drift}ms)"
            );
        }
        assert!(update.value > Decimal::ZERO);
    }
    assert_eq!(
        per_stream.len(),
        4,
        "all four streams observed: {per_stream:?}"
    );
}

#[test]
fn malformed_corpus_never_panics_and_is_always_ignored() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/malformed");
    let mut checked = 0;
    for entry in std::fs::read_dir(dir).expect("malformed corpus dir") {
        let path = entry.expect("dir entry").path();
        let content = std::fs::read_to_string(&path).expect("readable fixture");
        match parse_frame(&content, NOW) {
            BinanceParsed::Ignored(_) => checked += 1,
            other => panic!("{} parsed as {other:?}", path.display()),
        }
    }
    assert!(checked >= 15, "corpus present ({checked} files)");
}

#[test]
fn truncation_fuzz_never_panics() {
    for fixture in [
        include_str!("fixtures/binance_bookticker_btcusdt.json"),
        include_str!("fixtures/binance_trade_ethusdt.json"),
    ] {
        for (i, _) in fixture.char_indices() {
            let _ = parse_frame(&fixture[..i], NOW);
        }
    }
}
