//! Per-window maker-core integration tests, driven through the
//! [`engine::RiskManager`] gateway against `venue-paper` (CLAUDE.md §5/§8).
//!
//! These prove the per-window refactor: the maker core ([`engine::QuoteManager`])
//! quotes **every** concurrently-active window at once, so at a shared window
//! boundary no series clobbers another. The pre-refactor bug ran a single global
//! `active` slot, so the last-opened series (ETH-1h in `Series::ALL` order) won it
//! and the other five placed ~0 orders.
//!
//! - `all_coincident_series_quote_concurrently`: open all six series in
//!   `Series::ALL` order (BTC before ETH), converge, and assert every series has
//!   resting orders and BTC/ETH place symmetric ladders;
//! - `closing_one_window_leaves_siblings_resting`: closing one window issues a
//!   scoped `cancel_market` (never `cancel_all`), so siblings keep resting.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot, OrderState,
    Price, ResolutionSource, RoundDir, Series, Size, TickSize, TimestampMs, TokenId, TokenPair,
    WindowDuration, WindowId, WindowLifecycle,
};
use engine::{QuoteManagerParams, RiskManager, RiskParams};
use rust_decimal::dec;
use tokio::sync::mpsc::Receiver;
use venue_api::{VenueEvent, VenueEvents};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;

// ---- per-series builders --------------------------------------------------

fn px(d: Decimal) -> Price {
    Price::quantize(d, TickSize::T001, RoundDir::Down).expect("price")
}
fn sz(d: Decimal) -> Size {
    Size::new(d).expect("size")
}

/// A stable 0..6 index for a series (its position in `Series::ALL`), used to mint
/// distinct condition ids and tokens per series.
fn series_idx(s: Series) -> u32 {
    u32::try_from(Series::ALL.iter().position(|x| *x == s).expect("known series")).expect("idx")
}
fn window_of(s: Series) -> WindowId {
    WindowId {
        series: s,
        open_time: TimestampMs::from_millis(BASE_MS),
    }
}
fn up_token_of(s: Series) -> TokenId {
    TokenId::new(format!("{}11", series_idx(s) + 1)).expect("up token")
}
fn down_token_of(s: Series) -> TokenId {
    TokenId::new(format!("{}22", series_idx(s) + 1)).expect("down token")
}
fn condition_of(s: Series) -> ConditionId {
    ConditionId::new(format!("0x{:064x}", series_idx(s) + 1)).expect("cid")
}
fn duration_ms(s: Series) -> i64 {
    match s.duration {
        WindowDuration::M5 => 300_000,
        WindowDuration::M15 => 900_000,
        WindowDuration::H1 => 3_600_000,
    }
}
fn market_of(s: Series) -> MarketInfo {
    MarketInfo {
        window: window_of(s),
        event_slug: format!("{}-updown-perwindow", s.key()),
        condition_id: condition_of(s),
        tokens: TokenPair {
            up: up_token_of(s),
            down: down_token_of(s),
        },
        close_time: TimestampMs::from_millis(BASE_MS + duration_ms(s)),
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
    }
}
fn window_event_of(s: Series, lifecycle: WindowLifecycle) -> Event {
    Event::Window {
        market: Arc::new(market_of(s)),
        lifecycle,
    }
}
fn book(
    token: TokenId,
    condition: ConditionId,
    bids: &[(Decimal, Decimal)],
    asks: &[(Decimal, Decimal)],
    ts: i64,
) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: token,
        condition_id: condition,
        bids: bids
            .iter()
            .map(|(p, s)| BookLevel {
                price: px(*p),
                size: sz(*s),
            })
            .collect(),
        asks: asks
            .iter()
            .map(|(p, s)| BookLevel {
                price: px(*p),
                size: sz(*s),
            })
            .collect(),
        ts: TimestampMs::from_millis(ts),
        seq_hash: None,
    }))
}
fn model_of(s: Series, p_up: f64, sigma: f64, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: s.asset,
        window: Some(window_of(s)),
        p_up,
        z: 0.0,
        sigma_1s: sigma,
        sigma_tau: sigma * 15.0,
        basis: 0.0,
        anchor: AnchorSource::BinanceCorrected,
        health: ModelHealth::Ready,
        reason: ModelHealthReason::Nominal,
        input_ages: InputAges {
            chainlink: DurationMs::from_millis(100),
            binance: DurationMs::from_millis(50),
        },
        ts: TimestampMs::from_millis(ts),
    })
}

/// A quoter-only risk manager (takers disabled) with roomy loss caps registered
/// for every series, so no §11 breaker trips on this benign data.
fn quoter_only_risk() -> RiskManager {
    let params = RiskParams {
        quote_manager: QuoteManagerParams {
            min_requote_interval_ms: 10,
            ..QuoteManagerParams::default()
        },
        momentum_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    for s in Series::ALL {
        caps.insert(s, Dollars::new(Decimal::from(10_000)));
    }
    RiskManager::new(params, caps)
}

fn params() -> PaperParams {
    PaperParams {
        placement: LatencySpec {
            mean_ms: DurationMs::from_millis(200),
            jitter_ms: DurationMs::ZERO,
        },
        cancel: LatencySpec {
            mean_ms: DurationMs::from_millis(100),
            jitter_ms: DurationMs::ZERO,
        },
        rng_seed: Some(1),
        // Maker convergence is post-only, unaffected by the venue taker delay.
        venue_taker_delay_ms: DurationMs::ZERO,
        ..PaperParams::default()
    }
}

fn make() -> (
    PaperVenue,
    Receiver<VenueEvent>,
    RiskManager,
    tokio::time::Instant,
) {
    let start = tokio::time::Instant::now();
    let mut venue = PaperVenue::spawn(params(), move || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    });
    let rx = venue.take_event_rx().expect("event rx");
    (venue, rx, quoter_only_risk(), start)
}

async fn feed<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    risk: &mut RiskManager,
    ev: &Event,
    now: &F,
) {
    venue.on_bus_event(ev).await;
    risk.on_event(ev, venue, now()).await;
}

async fn drain<F: Fn() -> TimestampMs>(
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    venue: &PaperVenue,
    now: &F,
    recorded: &mut Vec<VenueEvent>,
) {
    while let Ok(ev) = rx.try_recv() {
        risk.on_venue_event(&ev, venue, now()).await;
        recorded.push(ev);
    }
}

/// Feeds Window Open + both books + a Ready model for one series.
async fn arm_series<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    risk: &mut RiskManager,
    s: Series,
    now: &F,
) {
    let cond = condition_of(s);
    feed(venue, risk, &window_event_of(s, WindowLifecycle::Open), now).await;
    feed(
        venue,
        risk,
        &book(
            up_token_of(s),
            cond.clone(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        now,
    )
    .await;
    feed(
        venue,
        risk,
        &book(
            down_token_of(s),
            cond,
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        now,
    )
    .await;
    feed(venue, risk, &model_of(s, 0.50, 0.0005, BASE_MS), now).await;
}

fn resting_len(risk: &RiskManager, s: Series) -> usize {
    risk.resting_view_for(window_of(s)).map_or(0, |v| v.len())
}

// ---- (a) all six series quote concurrently --------------------------------

#[tokio::test(start_paused = true)]
async fn all_coincident_series_quote_concurrently() {
    let (venue, mut rx, mut risk, start) = make();
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    // Arm all six series in Series::ALL order (BTC-5m, BTC-15m, BTC-1h, ETH-5m,
    // ETH-15m, ETH-1h) — the exact BTC-before-ETH order that made the old
    // single-slot maker clobber every window but the last one opened.
    for s in Series::ALL {
        arm_series(&venue, &mut risk, s, &now).await;
    }

    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;

    // EVERY series has resting orders — the bug left five of six empty (the BTC
    // windows and ETH-1h that were clobbered by the last-opened slot).
    for s in Series::ALL {
        let n = resting_len(&risk, s);
        assert!(n > 0, "{} must have resting quotes (got {n})", s.key());
    }

    // Venue Opens grouped by asset: BTC (3 series) and ETH (3 series) place the
    // same number of orders — no asset starved.
    let mut btc_opens = 0usize;
    let mut eth_opens = 0usize;
    for ev in &recorded {
        if let VenueEvent::Order(u) = ev
            && u.state == OrderState::Open
        {
            match u.window.series.asset {
                Asset::Btc => btc_opens += 1,
                Asset::Eth => eth_opens += 1,
            }
        }
    }
    assert!(
        btc_opens > 0 && eth_opens > 0,
        "both assets placed orders (btc {btc_opens}, eth {eth_opens})"
    );
    assert_eq!(
        btc_opens, eth_opens,
        "BTC and ETH place symmetric ladders (btc {btc_opens}, eth {eth_opens})"
    );
    assert!(
        !risk.state_snapshot().any_tripped(),
        "no breaker tripped on benign data"
    );
    venue.shutdown();
}

// ---- (b) closing one window leaves siblings resting -----------------------

#[tokio::test(start_paused = true)]
async fn closing_one_window_leaves_siblings_resting() {
    let (venue, mut rx, mut risk, start) = make();
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    let btc5m = Series::ALL[0]; // BTC-5m
    let eth5m = Series::ALL[3]; // ETH-5m

    for s in [btc5m, eth5m] {
        arm_series(&venue, &mut risk, s, &now).await;
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    assert!(resting_len(&risk, btc5m) > 0, "BTC-5m converged");
    assert!(resting_len(&risk, eth5m) > 0, "ETH-5m converged");

    // Close ONLY BTC-5m — a scoped cancel-market by its condition id, never
    // cancel-all, so ETH-5m's resting orders survive.
    feed(
        &venue,
        &mut risk,
        &window_event_of(btc5m, WindowLifecycle::Closing),
        &now,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;

    assert_eq!(
        resting_len(&risk, btc5m),
        0,
        "BTC-5m withdrawn on Closing"
    );
    assert!(
        resting_len(&risk, eth5m) > 0,
        "ETH-5m siblings still resting after BTC-5m Closing"
    );
    assert!(!risk.state_snapshot().any_tripped(), "guard transparent");
    venue.shutdown();
}
