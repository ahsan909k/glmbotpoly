//! Analytics rebuild-from-journal integration test (CLAUDE.md §3/§9 + §8/§10):
//! the single proof that analytics (1) computes incrementally as events arrive
//! AND rebuilds bit-for-bit from the journal, both paths agreeing, and (2) its
//! per-window PnL attribution sums to the ledger's window PnL.
//!
//! A scripted session — a window open, model-fair snapshots, two passive (Maker)
//! fills, more snapshots crossing the 1s/5s/30s markout deadlines, the window's
//! resolution, and the engine-derived [`Settlement`](core_types::Event::Settlement)
//! — is folded live into [`analytics::Analytics`], recorded through the **real**
//! [`journal::Recorder`], replayed with [`journal::ReplayReader::events`], and
//! rebuilt. The replayed events must equal the originals (lossless round-trip
//! incl. `Model`/`Settlement`), the rebuilt engine must equal the live one, the
//! two effect streams must match, and the attribution's five buckets must sum to
//! the comprehensive (rebate-inclusive) window PnL whose four trading buckets are
//! the engine's authoritative `realized_pnl`.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;

use analytics::{
    Analytics, AnalyticsEffect, AnalyticsParams, ComparisonWindow, DayKey, FillMarkout, SortColumn,
    WindowAttribution,
};
use core_types::{
    AnchorSource, Asset, ConditionId, Decimal, Dollars, DurationMs, Event, FeeParams, Fill,
    InputAges, Liquidity, MarketInfo, Mode, ModelHealth, ModelHealthReason, ModelSnapshot, OrderId,
    Outcome, Price, ResolutionSource, RoundDir, Series, Side, Size, TickSize, TimestampMs, TokenId,
    TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{InventoryEffect, InventoryManager};
use journal::{Recorder, RecorderParams, ReplayReader};
use rust_decimal::dec;

const OPEN_MS: i64 = 1_781_481_600_000; // 2026-06-15T00:00:00Z
const CLOSE_MS: i64 = OPEN_MS + 300_000;

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

fn maker_fill(outcome: Outcome, price: Decimal, size: Decimal, ts: i64) -> Event {
    Event::Fill(Arc::new(Fill {
        order_id: OrderId::new("paper-1").expect("order id"),
        trade_id: Some("paper-t1".to_owned()),
        window: window(),
        token_id: TokenId::new(if outcome == Outcome::Up { "1" } else { "2" }).expect("token"),
        outcome,
        side: Side::Buy,
        price: px(price),
        size: sz(size),
        liquidity: Liquidity::Maker,
        fee: Dollars::ZERO,
        ts_venue: TimestampMs::from_millis(ts),
        ts_local: TimestampMs::from_millis(ts),
    }))
}

fn model(p_up: f64, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window()),
        p_up,
        z: 0.0,
        sigma_1s: 0.0005,
        sigma_tau: 0.01,
        basis: -4.0,
        anchor: AnchorSource::Chainlink,
        health: ModelHealth::Ready,
        reason: ModelHealthReason::Nominal,
        input_ages: InputAges {
            chainlink: DurationMs::from_millis(100),
            binance: DurationMs::from_millis(100),
        },
        ts: TimestampMs::from_millis(ts),
    })
}

/// Builds the scripted session, deriving the settlement from the engine's
/// inventory book so `realized_pnl` is the authoritative ledger number.
fn session() -> Vec<Event> {
    let mut events = vec![
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Open,
        },
        model(0.50, OPEN_MS),
        // Two passive fills forming a matched pair (cash -97).
        maker_fill(Outcome::Up, dec!(0.48), dec!(100), OPEN_MS + 10),
        maker_fill(Outcome::Down, dec!(0.49), dec!(100), OPEN_MS + 20),
        // Snapshots crossing the 1s/5s/30s deadlines (p_up rising = Up ages well).
        model(0.52, OPEN_MS + 1_000),
        model(0.55, OPEN_MS + 5_000),
        model(0.60, OPEN_MS + 30_000),
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        },
    ];

    // Fold the fills + resolution through the engine to get the real settlement.
    let mut inv = InventoryManager::new();
    let mut summary = None;
    for event in &events {
        for effect in inv.on_event(event) {
            if let InventoryEffect::Settled(s) = effect {
                summary = Some(s);
            }
        }
    }
    events.push(Event::Settlement(Arc::new(
        summary.expect("engine produced a settlement"),
    )));
    events
}

fn fold(events: &[Event]) -> (Analytics, Vec<AnalyticsEffect>) {
    let mut a = Analytics::new(Mode::Paper, AnalyticsParams::default());
    let effects = events.iter().flat_map(|e| a.on_event(e)).collect();
    (a, effects)
}

fn markouts(effects: &[AnalyticsEffect]) -> Vec<&FillMarkout> {
    effects
        .iter()
        .filter_map(|e| match e {
            AnalyticsEffect::FillMarkout(f) => Some(f),
            _ => None,
        })
        .collect()
}

fn settled(effects: &[AnalyticsEffect]) -> &WindowAttribution {
    effects
        .iter()
        .find_map(|e| match e {
            AnalyticsEffect::WindowSettled(w) => Some(w),
            _ => None,
        })
        .expect("a WindowSettled effect")
}

#[test]
fn analytics_rebuilds_exactly_and_attribution_sums_to_the_ledger() {
    let dir = std::env::temp_dir().join(format!("analytics-replay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let events = session();

    // --- live incremental fold ------------------------------------------------
    let (live, live_effects) = fold(&events);

    // --- record through the real journal, then replay -------------------------
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

    let replayed: Vec<Event> = ReplayReader::open(&dir)
        .expect("open replay")
        .events()
        .map(|r| r.expect("replayed event"))
        .collect();
    assert_eq!(
        replayed, events,
        "journal round-trip must be lossless (incl. Model + Settlement)"
    );

    // --- rebuild from the journal; state + effects must match the live fold ----
    let rebuilt = Analytics::rebuild(Mode::Paper, AnalyticsParams::default(), replayed.iter());
    assert_eq!(
        rebuilt, live,
        "rebuilt analytics must equal the live engine"
    );

    let (_replayed_engine, replayed_effects) = fold(&replayed);
    assert_eq!(
        replayed_effects, live_effects,
        "incremental and replay effect streams must match"
    );

    // --- attribution sums to the ledger window PnL ----------------------------
    let attribution = settled(&live_effects);
    // Engine ledger PnL: cash -97 + 100 winning Up shares = +3.
    assert_eq!(attribution.realized_pnl, Dollars::new(dec!(3)));
    // 4 trading buckets sum EXACTLY to realized_pnl.
    assert_eq!(attribution.trading_sum(), attribution.realized_pnl);
    // 5 buckets sum EXACTLY to the comprehensive (rebate-inclusive) PnL.
    assert_eq!(attribution.bucket_sum(), attribution.comprehensive_pnl());
    // Balanced book: locked-pair holds the edge, no excess/remainder.
    assert_eq!(attribution.locked_pair_pnl, Dollars::new(dec!(3)));
    assert_eq!(attribution.excess_pnl, Dollars::ZERO);
    assert_eq!(attribution.settlement_remainder, Dollars::ZERO);
    // Maker rebate = 0.20 · (taker_fee(100,0.07,0.48) + taker_fee(100,0.07,0.49))
    //             = 0.20 · (1.7472 + 1.7493) = 0.69930.
    assert_eq!(attribution.estimated_rebate, Dollars::new(dec!(0.69930)));
    assert_eq!(attribution.comprehensive_pnl(), Dollars::new(dec!(3.69930)));
    assert_eq!(attribution.maker_fills, 2);

    // --- markouts: Up ages well (fair rose), Down poorly (the mirror) ----------
    let ms = markouts(&live_effects);
    assert_eq!(ms.len(), 2);
    let up = ms.iter().find(|f| f.outcome == Outcome::Up).expect("up");
    let down = ms
        .iter()
        .find(|f| f.outcome == Outcome::Down)
        .expect("down");
    // Up bought at fair 0.50, p_up 0.55 at +5s → +0.05; won → expiry +0.50.
    assert!((up.s5.unwrap().value - 0.05).abs() < 1e-12);
    assert!((up.s30.unwrap().value - 0.10).abs() < 1e-12);
    assert!((up.expiry.unwrap().value - 0.50).abs() < 1e-12);
    // Down is the mirror: fair_Down fell → -0.05 at +5s; lost → expiry -0.50.
    assert!((down.s5.unwrap().value - (-0.05)).abs() < 1e-12);
    assert!((down.expiry.unwrap().value - (-0.50)).abs() < 1e-12);

    // --- the derived series view reproduces on both paths ---------------------
    let live_rows = live.series_comparison();
    let rebuilt_rows = rebuilt.series_comparison();
    assert_eq!(live_rows, rebuilt_rows, "series comparison must reproduce");
    assert_eq!(live_rows.len(), 1);
    assert_eq!(live_rows[0].windows_traded, 1);
    assert_eq!(live_rows[0].net_pnl, Dollars::new(dec!(3)));
    // Only two passive 5s markouts so far → below the adverse-selection sample floor.
    assert_eq!(
        live.series_health(window().series),
        analytics::AdverseSelectionState::InsufficientSample
    );

    // --- windowed query: today selection, distribution, taker-budget, sorting -
    let today = DayKey::from_ts(TimestampMs::from_millis(CLOSE_MS));
    let cmp = live.series_comparison_over(ComparisonWindow::Today, today);
    assert_eq!(cmp.window, ComparisonWindow::Today);
    assert_eq!(cmp.default_sort, SortColumn::NetPnl);
    assert_eq!(cmp.rows.len(), 1);
    let row = &cmp.rows[0];
    assert_eq!(row.windows_traded, 1);
    // Distribution of the two 5s markouts: Up +0.05, Down -0.05 → mean 0.
    assert_eq!(row.markout_5s.n, 2);
    assert!(row.markout_5s.mean.expect("mean").abs() < 1e-12);
    assert!((row.markout_5s.min.expect("min") - (-0.05)).abs() < 1e-12);
    assert!((row.markout_5s.max.expect("max") - 0.05).abs() < 1e-12);
    // The histogram holds exactly the two samples (their computed ±0.05 values
    // land in the bins straddling 0.05, so assert the total rather than ULP-exact
    // bin indices).
    assert_eq!(
        row.markout_5s.histogram.iter().sum::<u64>(),
        2,
        "all samples are binned"
    );
    // Maker-only balanced book: no taker spend, full maker mix.
    assert_eq!(row.taker_notional, Dollars::ZERO);
    assert_eq!(row.taker_budget_used_fraction, Some(0.0)); // 0 / 1 window / $10
    assert!((row.maker_fill_fraction - 1.0).abs() < 1e-12);

    // The windowed query reproduces on the rebuilt engine, bit for bit.
    assert_eq!(
        cmp,
        rebuilt.series_comparison_over(ComparisonWindow::Today, today)
    );

    let _ = std::fs::remove_dir_all(&dir);
}
