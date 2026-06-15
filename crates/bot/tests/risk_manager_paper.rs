//! Chaos-style integration tests for the [`engine::RiskManager`] gateway against
//! `venue-paper` (CLAUDE.md §11).
//!
//! Each scenario converges the quote manager (so there are real resting orders),
//! injects a §11 fault, and asserts the non-negotiable invariant: **after the
//! trip, the open orders are cancelled AND no further order reaches the venue
//! until the recovery / reset condition is met.** The risk manager is the single
//! gateway — the strategy drivers are `pub(crate)` in `engine`, so the only path
//! to the venue is through the guard.
//!
//! Covered faults: market-WS disconnect, user-WS disconnect, fast-feed staleness
//! (>500 ms), daily stop (latched until manual reset), per-window loss (per-window
//! halt, not global), and manual kill. The error-rate / engine-restart / clock-skew
//! / sanity breakers are exhaustively unit-tested in `engine::risk::core`.
//!
//! `cargo test -p bot` needs the `venue-live` SDK to build; these tests depend
//! only on the SDK-free crates (engine, venue-paper, venue-api, journal,
//! core-types, tokio), so they can also be driven via the throwaway-harness path
//! used by prior bot-integration tasks when the SDK build state is broken.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookHealth, BookLevel, BookSnapshot, BookUnreliableReason, BreakerKind,
    ConditionId, ControlEvent, Decimal, Dollars, DurationMs, Event, FeeParams, InputAges,
    MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot, OrderState, Outcome, Price,
    PriceSource, PriceTick, ResolutionSource, RiskEvent, RoundDir, Series, Side, Size, TickKind,
    TickSize, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{QuoteManagerParams, RiskManager, RiskParams};
use rust_decimal::dec;
use tokio::sync::mpsc::Receiver;
use venue_api::{VenueEvent, VenueEvents};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;
const CLOSE_MS: i64 = BASE_MS + 300_000;

// ---- builders -------------------------------------------------------------

fn px(d: Decimal) -> Price {
    Price::quantize(d, TickSize::T001, RoundDir::Down).expect("price")
}
fn sz(d: Decimal) -> Size {
    Size::new(d).expect("size")
}
fn up_token() -> TokenId {
    TokenId::new("111").expect("token")
}
fn down_token() -> TokenId {
    TokenId::new("222").expect("token")
}
fn condition() -> ConditionId {
    ConditionId::new(format!("0x{}", "11".repeat(32))).expect("cid")
}
fn window() -> WindowId {
    WindowId {
        series: Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        open_time: TimestampMs::from_millis(BASE_MS),
    }
}
fn market() -> MarketInfo {
    MarketInfo {
        window: window(),
        event_slug: "btc-updown-5m-risk".to_owned(),
        condition_id: condition(),
        tokens: TokenPair {
            up: up_token(),
            down: down_token(),
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
    }
}
fn window_event(lifecycle: WindowLifecycle) -> Event {
    Event::Window {
        market: Arc::new(market()),
        lifecycle,
    }
}
fn book(token: TokenId, bid: Decimal, ask: Decimal, ts: i64) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: token,
        condition_id: condition(),
        bids: vec![BookLevel {
            price: px(bid),
            size: sz(dec!(100)),
        }],
        asks: vec![BookLevel {
            price: px(ask),
            size: sz(dec!(100)),
        }],
        ts: TimestampMs::from_millis(ts),
        seq_hash: None,
    }))
}
fn model(p_up: f64, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window()),
        p_up,
        z: 0.0,
        sigma_1s: 0.0005,
        sigma_tau: 0.0075,
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
fn binance_tick(ts: i64) -> Event {
    Event::PriceTick(PriceTick {
        source: PriceSource::BinanceDirect,
        asset: Asset::Btc,
        kind: TickKind::Mid,
        value: dec!(60000),
        ts_exchange: TimestampMs::from_millis(ts),
        ts_local: TimestampMs::from_millis(ts),
    })
}
fn last_trade(token: TokenId, price: Decimal, size: Decimal, side: Side, ts: i64) -> Event {
    Event::LastTrade {
        token_id: Arc::new(token),
        price: px(price),
        size: sz(size),
        side,
        ts: TimestampMs::from_millis(ts),
    }
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
        ..PaperParams::default()
    }
}

/// A quoter-only risk manager (takers disabled) with the given daily-stop loss
/// and per-series worst-case cap.
fn chaos_risk(daily_stop_loss: Dollars, cap: Dollars) -> RiskManager {
    let rp = RiskParams {
        quote_manager: QuoteManagerParams {
            min_requote_interval_ms: 10,
            ..QuoteManagerParams::default()
        },
        daily_stop_loss,
        momentum_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, cap);
    RiskManager::new(rp, caps)
}

// ---- harness --------------------------------------------------------------

fn make(
    risk: RiskManager,
) -> (
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
    (venue, rx, risk, start)
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
) -> Vec<VenueEvent> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        risk.on_venue_event(&ev, venue, now()).await;
        out.push(ev);
    }
    out
}

fn published_breakers(risk: &mut RiskManager) -> Vec<RiskEvent> {
    risk.take_published()
        .into_iter()
        .filter_map(|e| match e {
            Event::Risk(re) => Some(re),
            _ => None,
        })
        .collect()
}
fn tripped(breakers: &[RiskEvent], b: BreakerKind) -> bool {
    breakers
        .iter()
        .any(|re| matches!(re, RiskEvent::BreakerTripped { breaker } if *breaker == b))
}
fn count_state(recorded: &[VenueEvent], state: OrderState) -> usize {
    recorded
        .iter()
        .filter(|ev| matches!(ev, VenueEvent::Order(u) if u.state == state))
        .count()
}
fn resting_len(risk: &RiskManager) -> usize {
    risk.resting_view().map_or(0, |v| v.len())
}

/// Feeds window-open + both books + a Ready model, converges the quoter, and
/// drains the resulting Opens — leaving six resting orders.
async fn converge<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    now: &F,
) {
    for ev in [
        window_event(WindowLifecycle::Open),
        book(up_token(), dec!(0.45), dec!(0.55), BASE_MS),
        book(down_token(), dec!(0.45), dec!(0.55), BASE_MS),
        model(0.50, BASE_MS),
    ] {
        feed(venue, risk, &ev, now).await;
    }
    risk.on_tick(venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(rx, risk, venue, now).await;
    assert_eq!(resting_len(risk), 6, "converged to a six-level ladder");
    // No breaker tripped during a clean converge.
    let _ = published_breakers(risk);
}

// ---- (1) market-WS disconnect ---------------------------------------------

#[tokio::test(start_paused = true)]
async fn market_ws_disconnect_cancels_all_and_halts_until_recovery() {
    let (venue, mut rx, mut risk, start) = make(chaos_risk(
        Dollars::new(dec!(200)),
        Dollars::new(dec!(10_000)),
    ));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &mut rx, &mut risk, &now).await;

    // Market-WS drops → BookHealth::Unreliable{Disconnected}.
    feed(
        &venue,
        &mut risk,
        &Event::BookHealth(BookHealth::Unreliable {
            window: window(),
            reason: BookUnreliableReason::Disconnected,
            ts: now(),
        }),
        &now,
    )
    .await;
    assert!(
        tripped(&published_breakers(&mut risk), BreakerKind::WsDisconnect),
        "WsDisconnect published"
    );
    assert!(risk.is_halted(), "globally halted on the disconnect");

    // The resting orders are cancelled.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(
        count_state(&recorded, OrderState::Canceled),
        6,
        "all six cancelled"
    );
    assert_eq!(resting_len(&risk), 0, "no resting orders left");

    // Quoting stays halted: a fresh model + tick places nothing.
    feed(&venue, &mut risk, &model(0.50, BASE_MS + 1000), &now).await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(resting_len(&risk), 0, "no quoting while halted");

    // Recovery clears the breaker and quoting resumes.
    feed(
        &venue,
        &mut risk,
        &Event::BookHealth(BookHealth::Recovered {
            window: window(),
            outage: DurationMs::from_millis(1500),
            ts: now(),
        }),
        &now,
    )
    .await;
    assert!(!risk.is_halted(), "cleared on recovery");
    feed(&venue, &mut risk, &model(0.50, BASE_MS + 2000), &now).await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(resting_len(&risk), 6, "quoting resumed after recovery");
    venue.shutdown();
}

// ---- (2) user-WS disconnect (via the venue stream) ------------------------

#[tokio::test(start_paused = true)]
async fn user_ws_disconnect_cancels_all_and_halts_until_reconnect() {
    let (venue, mut rx, mut risk, start) = make(chaos_risk(
        Dollars::new(dec!(200)),
        Dollars::new(dec!(10_000)),
    ));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &mut rx, &mut risk, &now).await;

    // The user-channel WebSocket drops (a live-adapter signal on the venue stream).
    risk.on_venue_event(
        &VenueEvent::Connectivity { connected: false },
        &venue,
        now(),
    )
    .await;
    assert!(
        tripped(&published_breakers(&mut risk), BreakerKind::WsDisconnect),
        "WsDisconnect published"
    );
    assert!(risk.is_halted());

    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(
        count_state(&recorded, OrderState::Canceled),
        6,
        "all six cancelled"
    );

    // Reconnect clears it.
    risk.on_venue_event(&VenueEvent::Connectivity { connected: true }, &venue, now())
        .await;
    assert!(!risk.is_halted(), "cleared on reconnect");
    venue.shutdown();
}

// ---- (3) fast-feed staleness (>500 ms) ------------------------------------

#[tokio::test(start_paused = true)]
async fn fast_feed_staleness_trips_feed_stale_and_recovers() {
    let (venue, mut rx, mut risk, start) = make(chaos_risk(
        Dollars::new(dec!(200)),
        Dollars::new(dec!(10_000)),
    ));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &mut rx, &mut risk, &now).await;
    // Establish a live BinanceDirect/Mid signal stream.
    feed(&venue, &mut risk, &binance_tick(now().as_millis()), &now).await;
    let _ = published_breakers(&mut risk);

    // No fast-feed tick for >500 ms, then a risk tick: FeedStale trips.
    tokio::time::sleep(Duration::from_millis(700)).await;
    risk.on_tick(&venue, now()).await;
    assert!(
        tripped(&published_breakers(&mut risk), BreakerKind::FeedStale),
        "FeedStale trips on >500 ms silence"
    );
    assert!(risk.is_halted());

    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(
        count_state(&recorded, OrderState::Canceled),
        6,
        "all six cancelled"
    );

    // A fresh fast-feed tick clears it.
    feed(&venue, &mut risk, &binance_tick(now().as_millis()), &now).await;
    assert!(!risk.is_halted(), "cleared by a fresh fast-feed tick");
    venue.shutdown();
}

// ---- (4) daily stop, latched until manual reset ---------------------------

#[tokio::test(start_paused = true)]
async fn daily_stop_trips_on_realized_loss_and_blocks_until_reset() {
    // A tiny daily-stop so one small losing window trips it.
    let (venue, mut rx, mut risk, start) = make(chaos_risk(
        Dollars::new(dec!(1)),
        Dollars::new(dec!(10_000)),
    ));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &mut rx, &mut risk, &now).await;

    // A SELL print on the Up token at 0.47 lifts our Up bids (0.49/0.48/0.47) ⇒ we
    // buy ~30 Up shares (~$14.40) — an unmatched long.
    feed(
        &venue,
        &mut risk,
        &last_trade(
            up_token(),
            dec!(0.47),
            dec!(100),
            Side::Sell,
            now().as_millis(),
        ),
        &now,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(&mut rx, &mut risk, &venue, &now).await;
    assert!(
        recorded.iter().any(|e| matches!(e, VenueEvent::Fill(_))),
        "the print filled our Up bids"
    );
    let _ = published_breakers(&mut risk);

    // The window resolves Down → our Up shares lose → realized loss ≪ −$1.
    feed(
        &venue,
        &mut risk,
        &window_event(WindowLifecycle::Resolved {
            outcome: Outcome::Down,
        }),
        &now,
    )
    .await;
    assert!(
        tripped(&published_breakers(&mut risk), BreakerKind::DailyStop),
        "DailyStop trips on the realized loss"
    );
    assert!(risk.is_halted());

    // Nothing trades after: a fresh window + model + tick places nothing.
    for ev in [
        window_event(WindowLifecycle::Open),
        book(up_token(), dec!(0.45), dec!(0.55), BASE_MS + 5000),
        book(down_token(), dec!(0.45), dec!(0.55), BASE_MS + 5000),
        model(0.50, BASE_MS + 5000),
    ] {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(resting_len(&risk), 0, "daily stop blocks all trading");
    assert!(risk.is_halted(), "stays latched across a new window");

    // Only a manual reset clears it.
    feed(
        &venue,
        &mut risk,
        &Event::Control(ControlEvent::Reset),
        &now,
    )
    .await;
    assert!(!risk.is_halted(), "manual reset clears the daily stop");
    venue.shutdown();
}

// ---- (5) per-window loss (per-window halt, not global) --------------------

#[tokio::test(start_paused = true)]
async fn window_loss_halts_that_window_only() {
    // A tiny per-window cap so a small unmatched long breaches it.
    let (venue, mut rx, mut risk, start) =
        make(chaos_risk(Dollars::new(dec!(200)), Dollars::new(dec!(5))));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &mut rx, &mut risk, &now).await;

    // Fill our Up bids ⇒ unmatched Up cost ~$14.40 > the $5 cap.
    feed(
        &venue,
        &mut risk,
        &last_trade(
            up_token(),
            dec!(0.47),
            dec!(100),
            Side::Sell,
            now().as_millis(),
        ),
        &now,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(&mut rx, &mut risk, &venue, &now).await;
    assert!(
        tripped(&published_breakers(&mut risk), BreakerKind::WindowLoss),
        "WindowLoss trips on the worst-case breach"
    );
    // Per-window, NOT a global halt.
    assert!(
        !risk.is_halted(),
        "WindowLoss is per-window, not a global halt"
    );
    assert!(risk.state_snapshot().window_loss_active);

    // The window was cancel-market'd: the resting Down bids are cancelled.
    let _ = recorded;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded2 = drain(&mut rx, &mut risk, &venue, &now).await;
    assert!(
        count_state(&recorded2, OrderState::Canceled) >= 1,
        "the loss-halted window's resting orders are cancel-market'd"
    );

    // The halted window cannot place again (the guard refuses it).
    let cycles = risk.place_cycle_count();
    feed(&venue, &mut risk, &model(0.50, BASE_MS + 1000), &now).await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(
        risk.place_cycle_count(),
        cycles,
        "no placement into the loss-halted window"
    );

    // Resolving the window clears the per-window halt.
    feed(
        &venue,
        &mut risk,
        &window_event(WindowLifecycle::Resolved {
            outcome: Outcome::Up,
        }),
        &now,
    )
    .await;
    assert!(
        !risk.state_snapshot().window_loss_active,
        "resolution clears the per-window halt"
    );
    venue.shutdown();
}

// ---- (6) manual kill ------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn manual_kill_cancels_all_and_blocks_until_reset() {
    let (venue, mut rx, mut risk, start) = make(chaos_risk(
        Dollars::new(dec!(200)),
        Dollars::new(dec!(10_000)),
    ));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &mut rx, &mut risk, &now).await;

    feed(&venue, &mut risk, &Event::Control(ControlEvent::Kill), &now).await;
    assert!(
        tripped(&published_breakers(&mut risk), BreakerKind::Manual),
        "Manual published"
    );
    assert!(risk.is_halted());

    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(
        count_state(&recorded, OrderState::Canceled),
        6,
        "all six cancelled"
    );

    // Blocks until reset.
    feed(&venue, &mut risk, &model(0.50, BASE_MS + 1000), &now).await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(resting_len(&risk), 0, "kill blocks all trading");

    feed(
        &venue,
        &mut risk,
        &Event::Control(ControlEvent::Reset),
        &now,
    )
    .await;
    assert!(!risk.is_halted(), "manual reset clears the kill");
    feed(&venue, &mut risk, &model(0.50, BASE_MS + 2000), &now).await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert_eq!(resting_len(&risk), 6, "trading resumes after reset");
    venue.shutdown();
}
