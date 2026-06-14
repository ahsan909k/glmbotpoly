//! Inventory rebuild-from-journal integration test (CLAUDE.md §3/§9): the proof
//! that the engine's per-window inventory state "rebuilds exactly from the
//! journal on restart".
//!
//! A scripted session — a window open, a sequence of fills, and the window's
//! resolution — is recorded through the **real** [`journal::Recorder`] (gzip
//! segments on disk), read back with [`journal::ReplayReader::events`], and folded
//! into a fresh [`engine::InventoryManager`] via [`InventoryManager::rebuild`].
//! The rebuilt manager — and the settlement summary it produces — must equal an
//! in-memory manager folded from the same events, proving the
//! `Event → JournalRecord → gzip → JournalRecord → Event` path is lossless for the
//! new `Fill`/`Window`/`Settlement` types and that the fold is replay-exact.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use core_types::{
    Asset, ConditionId, Decimal, Dollars, Event, FeeParams, Fill, Liquidity, MarketInfo, OrderId,
    Outcome, Price, ResolutionSource, RoundDir, Series, Side, Size, TickSize, TimestampMs, TokenId,
    TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{InventoryEffect, InventoryManager};
use journal::{Recorder, RecorderParams, ReplayReader};
use rust_decimal::dec;

const OPEN_MS: i64 = 1_781_000_000_000;
const CLOSE_MS: i64 = 1_781_000_300_000;

fn window() -> WindowId {
    WindowId {
        series: Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        open_time: TimestampMs::from_millis(OPEN_MS),
    }
}

fn px(d: Decimal) -> Price {
    Price::quantize(d, TickSize::T001, RoundDir::Down).expect("price")
}

fn sz(d: Decimal) -> Size {
    Size::new(d).expect("size")
}

fn fill(outcome: Outcome, side: Side, price: Decimal, size: Decimal, fee: Decimal) -> Fill {
    Fill {
        order_id: OrderId::new("paper-1").expect("order id"),
        trade_id: Some("paper-t1".to_owned()),
        window: window(),
        token_id: TokenId::new("1").expect("token"),
        outcome,
        side,
        price: px(price),
        size: sz(size),
        liquidity: if fee.is_zero() {
            Liquidity::Maker
        } else {
            Liquidity::Taker
        },
        fee: Dollars::new(fee),
        ts_venue: TimestampMs::from_millis(OPEN_MS + 100),
        ts_local: TimestampMs::from_millis(OPEN_MS + 100),
    }
}

fn market() -> Arc<MarketInfo> {
    Arc::new(MarketInfo {
        window: window(),
        event_slug: "btc-updown-5m-test".to_owned(),
        condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).expect("cid"),
        tokens: TokenPair {
            up: TokenId::new("1").expect("up"),
            down: TokenId::new("2").expect("down"),
        },
        close_time: TimestampMs::from_millis(CLOSE_MS),
        strike: Some(dec!(60000)),
        tick_size: TickSize::T001,
        min_order_size: sz(dec!(5)),
        fees: FeeParams {
            rate: dec!(0.07),
            exponent: 1,
            taker_only: true,
            rebate_rate: dec!(0.2),
            enabled: true,
        },
        neg_risk: false,
        resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
    })
}

/// A scripted window session: open, a mix of fills, then resolution Up.
fn session() -> Vec<Event> {
    vec![
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Open,
        },
        Event::Fill(Arc::new(fill(
            Outcome::Up,
            Side::Buy,
            dec!(0.40),
            dec!(100),
            dec!(0),
        ))),
        Event::Fill(Arc::new(fill(
            Outcome::Down,
            Side::Buy,
            dec!(0.55),
            dec!(80),
            dec!(0),
        ))),
        // A taker sell with a real fee, to exercise the fee + cash folding.
        Event::Fill(Arc::new(fill(
            Outcome::Up,
            Side::Sell,
            dec!(0.60),
            dec!(30),
            dec!(0.25),
        ))),
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        },
    ]
}

fn settled_summary(effects: &[InventoryEffect]) -> core_types::SettlementSummary {
    effects
        .iter()
        .find_map(|e| match e {
            InventoryEffect::Settled(s) => Some(s.clone()),
            InventoryEffect::Snapshot(_) => None,
        })
        .expect("a settlement effect")
}

#[test]
fn inventory_rebuilds_exactly_from_the_recorded_journal() {
    let dir = std::env::temp_dir().join(format!("inventory-replay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let events = session();

    // In-memory reference: fold the live events, capturing the effect stream.
    let mut live = InventoryManager::new();
    let live_effects: Vec<InventoryEffect> = events.iter().flat_map(|e| live.on_event(e)).collect();

    // Record the session through the real recorder (constant clock — the journal's
    // ts_local_ms is irrelevant to the inventory fold, which is event-time-driven).
    let params = RecorderParams {
        out_dir: dir.clone(),
        ..RecorderParams::default()
    };
    let recorder =
        Recorder::spawn(params, || TimestampMs::from_millis(OPEN_MS)).expect("spawn recorder");
    for event in &events {
        recorder.record(event);
    }
    let stats = recorder.finish().expect("finish recorder");
    assert_eq!(stats.records, events.len() as u64);
    assert_eq!(stats.dropped, 0);

    // Replay the journal back into bus events.
    let replayed: Vec<Event> = ReplayReader::open(&dir)
        .expect("open replay")
        .events()
        .map(|r| r.expect("replayed event"))
        .collect();
    assert_eq!(replayed, events, "the journal round-trip must be lossless");

    // Rebuild the manager from the replayed events.
    let rebuilt = InventoryManager::rebuild(replayed.iter());
    assert_eq!(
        rebuilt, live,
        "rebuilt inventory must equal the live manager"
    );

    // The settlement summary reproduced through the journal matches the live one.
    let mut replayed_mgr = InventoryManager::new();
    let replayed_effects: Vec<InventoryEffect> = replayed
        .iter()
        .flat_map(|e| replayed_mgr.on_event(e))
        .collect();
    let live_summary = settled_summary(&live_effects);
    let replayed_summary = settled_summary(&replayed_effects);
    assert_eq!(
        replayed_summary, live_summary,
        "settlement summary must match"
    );

    // Sanity on the actual numbers: cash_flow = -40 - 44 + 18 - 0.25 = -66.25;
    // +70 winning Up shares → realized +3.75; one matched pair side merged none.
    assert_eq!(live_summary.outcome, Outcome::Up);
    assert_eq!(live_summary.realized_pnl, Dollars::new(dec!(3.75)));
    assert_eq!(live_summary.fees_paid, Dollars::new(dec!(0.25)));
    assert_eq!(live_summary.matched_pairs, sz(dec!(70)));

    let _ = std::fs::remove_dir_all(&dir);
}
