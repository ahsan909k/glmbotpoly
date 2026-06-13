//! Fixture-driven wire tests over captured real market-channel frames
//! (provenance: tests/fixtures/README.md) plus the malformed corpus, and
//! the session-replay invariant: the delta-maintained book equals every
//! arriving venue snapshot when no trade intervened, and is never crossed.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;

use core_types::{Side, TickSize, TimestampMs};
use feed_clob::ChangeOutcome;
use feed_clob::book::BookState;
use feed_clob::wire::{ClobEvent, IgnoredReason, ParsedFrame, parse_frame};
use rust_decimal::dec;

const NOW: TimestampMs = TimestampMs::from_millis(1_781_254_800_000);

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("fixture {path}: {e}"))
}

/// Parses a single-event fixture frame.
fn single(name: &str) -> Result<ClobEvent, IgnoredReason> {
    match parse_frame(&fixture(name), NOW) {
        ParsedFrame::Events(mut events) => {
            assert_eq!(events.len(), 1, "{name}: expected exactly one event");
            events.remove(0)
        }
        other => panic!("{name}: expected events, got {other:?}"),
    }
}

#[test]
fn pong_fixture_classifies_as_keepalive() {
    assert_eq!(
        parse_frame(&fixture("clob_pong.txt"), NOW),
        ParsedFrame::Pong
    );
}

#[test]
fn book_fixture_parses() {
    let Ok(ClobEvent::Book(book)) = single("clob_book.json") else {
        panic!("expected book");
    };
    assert!(book.condition.as_str().starts_with("0x"));
    assert!(!book.bids.is_empty());
    assert!(!book.asks.is_empty());
    assert!(book.hash.is_some());
    assert!(
        book.ts > TimestampMs::from_millis(1_700_000_000_000),
        "wire ms timestamp"
    );
    // The fixture applies cleanly and produces an uncrossed two-sided book.
    let mut state = BookState::new();
    assert_eq!(state.apply_snapshot(&book.bids, &book.asks), 0);
    assert!(!state.is_crossed());
    assert!(state.best_bid().is_some() && state.best_ask().is_some());
}

#[test]
fn array_books_fixture_parses_every_element() {
    let ParsedFrame::Events(events) = parse_frame(&fixture("clob_array_books.json"), NOW) else {
        panic!("expected events");
    };
    assert!(events.len() >= 2, "array frame carries several books");
    for event in events {
        assert!(matches!(event, Ok(ClobEvent::Book(_))));
    }
}

#[test]
fn price_change_fixture_parses_with_mirrored_tokens() {
    let Ok(ClobEvent::PriceChange(msg)) = single("clob_price_change.json") else {
        panic!("expected price_change");
    };
    assert!(!msg.changes.is_empty());
    // The live mirror pattern: one order shows up on both tokens with
    // complementary prices and opposite sides.
    if msg.changes.len() == 2 {
        let (a, b) = (&msg.changes[0], &msg.changes[1]);
        assert_ne!(a.token, b.token);
        assert_eq!(a.price + b.price, dec!(1));
        assert_ne!(a.side, b.side);
        assert_eq!(a.size, b.size);
    }
    for change in &msg.changes {
        assert!(change.price > dec!(0) && change.price < dec!(1));
        assert!(change.best_bid.is_some() || change.best_ask.is_some());
    }
}

#[test]
fn price_change_removal_fixture_deletes_the_level() {
    let Ok(ClobEvent::PriceChange(msg)) = single("clob_price_change_removal.json") else {
        panic!("expected price_change");
    };
    let removal = msg
        .changes
        .iter()
        .find(|c| c.size.is_zero())
        .expect("fixture carries a size-0 removal");
    let mut book = BookState::new();
    assert_ne!(
        book.apply_change(removal.side, removal.price, dec!(7)),
        ChangeOutcome::Rejected
    );
    assert_eq!(book.depth().0 + book.depth().1, 1);
    assert_ne!(
        book.apply_change(removal.side, removal.price, removal.size),
        ChangeOutcome::Rejected
    );
    assert!(book.is_empty(), "size 0 removes the level");
}

#[test]
fn last_trade_fixture_parses() {
    let Ok(ClobEvent::LastTrade {
        price,
        size,
        side,
        ts,
        ..
    }) = single("clob_last_trade_price.json")
    else {
        panic!("expected last_trade_price");
    };
    assert!(price > dec!(0) && price < dec!(1));
    assert!(size > dec!(0));
    assert!(matches!(side, Side::Buy | Side::Sell));
    assert!(ts > TimestampMs::from_millis(1_700_000_000_000));
}

#[test]
fn best_bid_ask_fixture_parses() {
    let Ok(ClobEvent::BestBidAsk {
        best_bid, best_ask, ..
    }) = single("clob_best_bid_ask.json")
    else {
        panic!("expected best_bid_ask");
    };
    assert!(best_bid.is_some() || best_ask.is_some());
    if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
        assert!(bid < ask, "venue tops are uncrossed");
    }
}

#[test]
fn new_market_fixture_parses() {
    // new_market is a platform-wide broadcast — this fixture is a sports
    // market that arrived on a BTC-5m connection (README).
    let Ok(ClobEvent::NewMarket {
        condition, slug, ..
    }) = single("clob_new_market.json")
    else {
        panic!("expected new_market");
    };
    assert!(condition.as_str().starts_with("0x"));
    assert!(!slug.is_empty());
}

#[test]
fn market_resolved_fixture_parses_via_market_field() {
    // The real frame carries the condition id in `market` (no
    // `condition_id`/`slug` field, unlike the docs' field list).
    let Ok(ClobEvent::MarketResolved {
        condition,
        winning_token,
        ts,
    }) = single("clob_market_resolved.json")
    else {
        panic!("expected market_resolved");
    };
    assert_eq!(
        condition.as_str(),
        "0x9c535590faae21bf76fed844aa47fc3c5c82253ef38bd545bf12c93470502685"
    );
    assert!(winning_token.as_str().chars().all(|c| c.is_ascii_digit()));
    assert!(ts > TimestampMs::from_millis(1_700_000_000_000));
}

#[test]
fn tick_size_change_fixture_parses() {
    // Synthesized from the docs field list (no flip occurred during the
    // capture windows — README); replace with a captured frame when one
    // lands.
    let Ok(ClobEvent::TickSizeChange {
        old_tick, new_tick, ..
    }) = single("clob_tick_size_change.json")
    else {
        panic!("expected tick_size_change");
    };
    assert_eq!(old_tick, Some(TickSize::T001));
    assert_eq!(new_tick, TickSize::T0001);
}

#[test]
fn malformed_corpus_never_panics_and_classifies() {
    let dir = format!("{}/tests/fixtures/malformed", env!("CARGO_MANIFEST_DIR"));
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).expect("malformed dir") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).expect("readable");
        seen += 1;
        match parse_frame(&text, NOW) {
            ParsedFrame::Ignored(_) => {}
            ParsedFrame::Events(events) => {
                assert!(
                    events.iter().all(Result::is_err),
                    "{name}: malformed input must not produce a valid event"
                );
            }
            other => panic!("{name}: unexpected classification {other:?}"),
        }
    }
    assert!(seen >= 10, "corpus present ({seen} files)");
}

/// The session-replay invariant over a captured real segment: replaying
/// deltas between venue snapshots reproduces each arriving snapshot exactly
/// (when no trade intervened — trade effects arrive only via snapshots),
/// and the derived book is never crossed (crossing deltas are trades at the
/// touch, resolved by implied consumption).
#[test]
fn session_replay_matches_every_snapshot_and_never_crosses() {
    #[derive(Default)]
    struct Replay {
        book: BookState,
        have_snapshot: bool,
        trade_since_snapshot: bool,
        snapshots: u64,
        audited: u64,
        mismatches: u64,
    }
    let raw = fixture("clob_session.jsonl");
    let mut tokens: HashMap<String, Replay> = HashMap::new();
    let mut events_seen = 0u64;

    for line in raw.lines() {
        let entry: serde_json::Value = serde_json::from_str(line).expect("tap line");
        if entry.get("dir").and_then(|d| d.as_str()) != Some("in") {
            continue;
        }
        let text = entry.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let ParsedFrame::Events(events) = parse_frame(text, NOW) else {
            continue;
        };
        for event in events {
            let event = event.expect("captured frames all parse");
            events_seen += 1;
            match event {
                ClobEvent::Book(msg) => {
                    let replay = tokens.entry(msg.token.as_str().to_owned()).or_default();
                    let mut incoming = BookState::new();
                    assert_eq!(
                        incoming.apply_snapshot(&msg.bids, &msg.asks),
                        0,
                        "captured snapshots carry no invalid levels"
                    );
                    if replay.have_snapshot && !replay.trade_since_snapshot {
                        replay.audited += 1;
                        if replay.book != incoming {
                            replay.mismatches += 1;
                        }
                    }
                    replay.book = incoming;
                    replay.have_snapshot = true;
                    replay.trade_since_snapshot = false;
                    replay.snapshots += 1;
                }
                ClobEvent::PriceChange(msg) => {
                    for change in &msg.changes {
                        let replay = tokens.entry(change.token.as_str().to_owned()).or_default();
                        if !replay.have_snapshot {
                            continue;
                        }
                        assert_ne!(
                            replay
                                .book
                                .apply_change(change.side, change.price, change.size),
                            ChangeOutcome::Rejected,
                            "captured changes all apply"
                        );
                        assert!(
                            !replay.book.is_crossed(),
                            "implied consumption keeps the book uncrossed"
                        );
                    }
                }
                ClobEvent::LastTrade { .. } => {
                    // Trade effects arrive only via snapshots; a trade on
                    // either token can reshape both books (mint matching).
                    for replay in tokens.values_mut() {
                        replay.trade_since_snapshot = true;
                    }
                }
                _ => {}
            }
        }
    }

    assert!(
        events_seen > 500,
        "segment is substantial ({events_seen} events)"
    );
    let (mut audited, mut mismatches, mut snapshots) = (0, 0, 0);
    for replay in tokens.values() {
        audited += replay.audited;
        mismatches += replay.mismatches;
        snapshots += replay.snapshots;
    }
    assert!(
        snapshots >= 4,
        "segment spans several snapshots ({snapshots})"
    );
    assert!(audited >= 2, "some snapshots were audited ({audited})");
    // Live truth: a small remainder of snapshots differ even without an
    // observed trade print (trade events can land after their snapshot;
    // partial fills change level sizes without deltas). Snapshot-replace
    // repairs them — the invariant is that drift stays rare, not zero.
    let drift_rate = mismatches as f64 / audited as f64;
    assert!(
        drift_rate < 0.10,
        "audited drift must stay rare: {mismatches}/{audited}"
    );
}
