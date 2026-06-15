//! End-to-end journal proof (CLAUDE.md §3/§9): a recorded session — with the
//! sqlite index enabled — replays losslessly, reproduces identical derived
//! inventory state, and is queryable through the structured index.
//!
//! Extends `inventory_replay.rs` (which covers the inventory rebuild from gzip
//! segments) with the two new pieces of this task: order updates in the stream
//! and the sqlite index. The bot's own `boot::rebuild_from_journal` is unit-
//! tested inside `boot.rs`; this integration test exercises the journal + engine
//! crates directly (a `bot/tests` integration test cannot reach the binary's
//! internal modules), the same way `inventory_replay.rs` does.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use core_types::{
    Asset, ConditionId, Decimal, Dollars, Event, FeeParams, Fill, Liquidity, MarketInfo, OrderId,
    OrderState, OrderUpdate, Outcome, Price, ResolutionSource, RoundDir, Series, Side, Size,
    TickSize, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{InventoryEffect, InventoryManager};
use journal::{JournalIndexReader, Recorder, RecorderParams, ReplayReader};
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
fn order(id: &str, state: OrderState) -> Event {
    Event::OrderUpdate(Arc::new(OrderUpdate {
        order_id: OrderId::new(id).expect("oid"),
        window: window(),
        token_id: TokenId::new("1").expect("token"),
        side: Side::Buy,
        state,
        price: px(dec!(0.40)),
        original_size: sz(dec!(10)),
        filled_size: if state == OrderState::Filled {
            sz(dec!(10))
        } else {
            sz(dec!(0))
        },
        reject_reason: None,
        ts_venue: None,
        ts_local: TimestampMs::from_millis(OPEN_MS),
    }))
}
fn fill(outcome: Outcome, side: Side, price: Decimal, size: Decimal, fee: Decimal) -> Event {
    Event::Fill(Arc::new(Fill {
        order_id: OrderId::new("o1").expect("oid"),
        trade_id: Some("t1".to_owned()),
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
    }))
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
fn journal_replays_rebuilds_and_is_queryable_with_the_sqlite_index() {
    let dir = std::env::temp_dir().join(format!("journal-rebuild-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sqlite = dir.join("index.sqlite");

    // A scripted session: window open, two order updates (one driven terminal),
    // a mix of fills, and resolution — plus the settlement summary the engine
    // would emit on the bus (so the sqlite settlements table is exercised too).
    let mut events = vec![
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Open,
        },
        order("o1", OrderState::Open),
        order("o1", OrderState::Filled),
        order("o2", OrderState::Open),
        fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(100), dec!(0)),
        fill(Outcome::Down, Side::Buy, dec!(0.55), dec!(80), dec!(0)),
        fill(Outcome::Up, Side::Sell, dec!(0.60), dec!(30), dec!(0.25)),
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        },
    ];

    // The settlement the engine derives from the fold — append it so it lands in
    // both the gzip log and the sqlite settlements table.
    let mut folder = InventoryManager::new();
    let effects: Vec<InventoryEffect> = events.iter().flat_map(|e| folder.on_event(e)).collect();
    let summary = settled_summary(&effects);
    assert_eq!(summary.realized_pnl, Dollars::new(dec!(3.75)));
    events.push(Event::Settlement(Arc::new(summary)));

    // Record the session with the sqlite index enabled.
    let params = RecorderParams {
        out_dir: dir.clone(),
        sqlite_path: Some(sqlite.clone()),
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
    // Structured rows = 3 orders + 3 fills + 2 windows + 1 settlement = 9.
    assert_eq!(stats.indexed, 9);

    // 1) Replay is lossless.
    let replayed: Vec<Event> = ReplayReader::open(&dir)
        .expect("open replay")
        .events()
        .map(|r| r.expect("event"))
        .collect();
    assert_eq!(replayed, events, "the journal round-trip must be lossless");

    // 2) Replay reproduces identical derived inventory state.
    let rebuilt = InventoryManager::rebuild(replayed.iter());
    assert_eq!(
        rebuilt, folder,
        "rebuilt inventory must equal the live fold"
    );

    // 3) The structured index is queryable and consistent with the stream.
    let reader = JournalIndexReader::open(&sqlite).expect("open index");
    let orders = reader
        .orders_for_window(window().series, OPEN_MS)
        .expect("orders");
    assert_eq!(orders.len(), 3, "three order updates indexed");
    let fills = reader
        .fills_for_window(window().series, OPEN_MS)
        .expect("fills");
    assert_eq!(fills.len(), 3, "three fills indexed");
    let windows = reader.windows().expect("windows");
    assert_eq!(windows.len(), 2, "open + resolved");
    let settlements = reader.settlements().expect("settlements");
    assert_eq!(settlements.len(), 1);
    assert_eq!(settlements[0].realized_pnl, Dollars::new(dec!(3.75)));
    assert_eq!(settlements[0].outcome, Outcome::Up);
    assert_eq!(settlements[0].matched_pairs, sz(dec!(70)));

    let _ = std::fs::remove_dir_all(&dir);
}
