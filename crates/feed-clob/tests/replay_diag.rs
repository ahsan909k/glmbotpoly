//! Capture replay diagnostic (ignored by default): feeds a raw `--raw`
//! JSONL capture through the wire parser and book state, reporting where
//! the delta-maintained books disagree with arriving venue snapshots.
//! Run explicitly while a fresh capture exists:
//! `cargo test -p feed-clob --test replay_diag -- --ignored --nocapture`

#![allow(clippy::expect_used, clippy::print_stdout, clippy::unwrap_used)]

use std::collections::HashMap;

use core_types::TimestampMs;
use feed_clob::book::BookState;
use feed_clob::wire::{ClobEvent, ParsedFrame, parse_frame};

const CAPTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/captures/clob.jsonl"
);

#[derive(Default)]
struct TokenDiag {
    book: BookState,
    have_snapshot: bool,
    changes_since_snapshot: u64,
    trade_since_snapshot: bool,
    snapshots: u64,
    drift_no_trade: u64,
    drift_with_trade: u64,
    crossed_after_change: u64,
    consumed: u64,
    top_mismatch: u64,
}

#[test]
#[ignore = "diagnostic over a local live capture"]
fn replay_capture_and_report_drift() {
    let raw = std::fs::read_to_string(CAPTURE).expect("capture file present");
    let mut tokens: HashMap<String, TokenDiag> = HashMap::new();
    let mut reported = 0u32;
    let now = TimestampMs::from_millis(0);

    for line in raw.lines() {
        let value: serde_json::Value = serde_json::from_str(line).expect("tap line");
        if value.get("dir").and_then(|d| d.as_str()) != Some("in") {
            continue;
        }
        let text = value.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let ParsedFrame::Events(events) = parse_frame(text, now) else {
            continue;
        };
        for event in events.into_iter().flatten() {
            match event {
                ClobEvent::Book(msg) => {
                    let entry = tokens.entry(msg.token.as_str().to_owned()).or_default();
                    let mut incoming = BookState::new();
                    incoming.apply_snapshot(&msg.bids, &msg.asks);
                    if entry.have_snapshot && entry.book != incoming {
                        if entry.trade_since_snapshot {
                            entry.drift_with_trade += 1;
                        } else {
                            entry.drift_no_trade += 1;
                            if reported < 3 {
                                reported += 1;
                                println!(
                                    "\n=== DRIFT (no trade) token …{} after {} changes ===",
                                    &msg.token.as_str()[msg.token.as_str().len() - 8..],
                                    entry.changes_since_snapshot
                                );
                                println!(
                                    "derived: bid {:?} ask {:?} depth {:?}",
                                    entry.book.best_bid(),
                                    entry.book.best_ask(),
                                    entry.book.depth()
                                );
                                println!(
                                    "venue:   bid {:?} ask {:?} levels {}/{}",
                                    incoming.best_bid(),
                                    incoming.best_ask(),
                                    msg.bids.len(),
                                    msg.asks.len()
                                );
                            }
                        }
                    }
                    entry.book = incoming;
                    entry.have_snapshot = true;
                    entry.trade_since_snapshot = false;
                    entry.changes_since_snapshot = 0;
                    entry.snapshots += 1;
                }
                ClobEvent::PriceChange(msg) => {
                    for change in &msg.changes {
                        let entry = tokens.entry(change.token.as_str().to_owned()).or_default();
                        if !entry.have_snapshot {
                            continue;
                        }
                        if let feed_clob::ChangeOutcome::Applied { consumed } = entry
                            .book
                            .apply_change(change.side, change.price, change.size)
                        {
                            entry.consumed += consumed as u64;
                        }
                        entry.changes_since_snapshot += 1;
                        if entry.book.is_crossed() {
                            entry.crossed_after_change += 1;
                        }
                        // The venue's own carried tops must agree with the
                        // post-consumption derived tops.
                        if change.best_bid.is_some()
                            && entry.book.best_bid().map(|(p, _)| p) != change.best_bid
                        {
                            entry.top_mismatch += 1;
                            if reported < 3 {
                                reported += 1;
                                println!(
                                    "\n=== TOP MISMATCH token …{}: derived bid {:?} venue {:?} ===",
                                    &change.token.as_str()[change.token.as_str().len() - 8..],
                                    entry.book.best_bid(),
                                    change.best_bid,
                                );
                            }
                        }
                    }
                }
                ClobEvent::LastTrade { token, .. } => {
                    // Trade effects arrive only via snapshots; suspend drift
                    // accounting for every token (cross-book mint matching).
                    let _ = token;
                    for entry in tokens.values_mut() {
                        entry.trade_since_snapshot = true;
                    }
                }
                _ => {}
            }
        }
    }

    println!("\n=== summary ===");
    for (token, diag) in &tokens {
        println!(
            "token …{}: snapshots {} drift(no trade) {} drift(trade) {} crossed {} consumed {} top_mismatch {}",
            &token[token.len() - 8..],
            diag.snapshots,
            diag.drift_no_trade,
            diag.drift_with_trade,
            diag.crossed_after_change,
            diag.consumed,
            diag.top_mismatch
        );
    }
}
