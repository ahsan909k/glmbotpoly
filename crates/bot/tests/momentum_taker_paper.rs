//! Momentum-taker integration tests, now driven through the [`engine::RiskManager`]
//! gateway against `venue-paper` (CLAUDE.md §8/§9/§11).
//!
//! The taker driver is `pub(crate)` in `engine`; the only path to the venue is
//! `RiskManager`. Only the momentum taker is enabled here, and with no breaker
//! tripped the risk guard is transparent — the take, fill, fee and budget
//! assertions hold exactly as they did when the taker was driven directly, and
//! each test additionally asserts the guard stayed a no-op.
//!
//! Asserts the required end-to-end properties:
//! - a confirmed fast-feed move + a stale-cheap book ⇒ exactly one FAK that
//!   fills against the displayed asks, with the realized taker fee equal to
//!   `taker_fee(filled, rate, price)` and the realized spend bounded by the
//!   per-window budget — driven off journal-recorded → replayed events;
//! - the cooldown blocks a second take within the window and releases after it.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, Liquidity, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot,
    Price, PriceSource, PriceTick, ResolutionSource, RoundDir, Series, Size, TickKind, TickSize,
    TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle, taker_fee,
};
use engine::{MomentumTakerParams, NormalizerParams, RiskManager, RiskParams};
use journal::{Recorder, RecorderParams, ReplayReader};
use rust_decimal::dec;
use tokio::sync::mpsc::Receiver;
use venue_api::{VenueEvent, VenueEvents};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;
const CLOSE_MS: i64 = BASE_MS + 300_000;
const TICK: TickSize = TickSize::T001;

// ---- builders -------------------------------------------------------------

fn px(d: Decimal) -> Price {
    Price::quantize(d, TICK, RoundDir::Down).expect("price")
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
        event_slug: "btc-updown-5m-mt".to_owned(),
        condition_id: condition(),
        tokens: TokenPair {
            up: up_token(),
            down: down_token(),
        },
        close_time: TimestampMs::from_millis(CLOSE_MS),
        strike: Some(dec!(60000)),
        tick_size: TICK,
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

/// A book for one token. `asks` are `(price, size)`, best-first.
fn book(token: TokenId, asks: &[(Decimal, Decimal)], ts: i64) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: token,
        condition_id: condition(),
        bids: Vec::new(),
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

fn model(p_up: f64, sigma: f64, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window()),
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

fn tick(value: Decimal, ts: i64) -> Event {
    Event::PriceTick(PriceTick {
        source: PriceSource::BinanceDirect,
        asset: Asset::Btc,
        kind: TickKind::Mid,
        value,
        ts_exchange: TimestampMs::from_millis(ts),
        ts_local: TimestampMs::from_millis(ts),
    })
}

/// A confirmed up-move: 10 ticks ending at `end_ms`, 100 ms apart, +$30 each.
fn up_ramp(end_ms: i64) -> Vec<Event> {
    (0..10)
        .map(|i| {
            tick(
                dec!(60000) + dec!(30) * Decimal::from(i),
                end_ms - (9 - i) * 100,
            )
        })
        .collect()
}

/// Latencies pinned: place 200 ms, cancel 100 ms, no jitter, fixed seed.
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
        // This test verifies the momentum taker's engine logic, not the venue delay — zero it
        // so FAK fills land at the network-latency deadline the clock advances to.
        venue_taker_delay_ms: DurationMs::ZERO,
        ..PaperParams::default()
    }
}

/// A momentum-only risk manager (quoter + late-window disabled) with a roomy loss
/// cap so no §11 breaker trips on this data — the guard is transparent.
fn momentum_only_risk(mt: MomentumTakerParams) -> RiskManager {
    let params = RiskParams {
        momentum: mt,
        quoter_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(Decimal::from(10_000)));
    RiskManager::new(params, caps)
}

/// Like [`momentum_only_risk`] but with a share clip on the normalizer, so each
/// taker FAK is split into ≤ `clip` share clips (CLAUDE.md §8 clip-splitting).
fn momentum_only_risk_with_clip(mt: MomentumTakerParams, clip: Decimal) -> RiskManager {
    let params = RiskParams {
        momentum: mt,
        normalizer: NormalizerParams {
            clip_size_shares: Some(clip),
            ..NormalizerParams::default()
        },
        quoter_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(Decimal::from(10_000)));
    RiskManager::new(params, caps)
}

// ---- harness --------------------------------------------------------------

fn make(
    taker_params: MomentumTakerParams,
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
    (venue, rx, momentum_only_risk(taker_params), start)
}

fn make_with_risk(
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

/// Feeds one bus event to both the paper venue (fill sim) and the risk manager.
async fn feed<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    risk: &mut RiskManager,
    ev: &Event,
    now: &F,
) {
    venue.on_bus_event(ev).await;
    risk.on_event(ev, venue, now()).await;
}

/// Drains every currently-ready venue event into the risk manager, collecting them.
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

/// Records `session` through the real journal recorder and replays it back.
fn record_then_replay(session: &[Event], dir_tag: &str) -> Vec<Event> {
    let dir = std::env::temp_dir().join(format!("mt_paper_{dir_tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create journal dir");
    let recorder = Recorder::spawn(
        RecorderParams {
            out_dir: dir.clone(),
            ..RecorderParams::default()
        },
        || TimestampMs::from_millis(BASE_MS),
    )
    .expect("spawn recorder");
    for ev in session {
        recorder.record(ev);
    }
    let stats = recorder.finish().expect("finish recorder");
    assert_eq!(stats.dropped, 0, "no journal records dropped");
    let replayed: Vec<Event> = ReplayReader::open(&dir)
        .expect("open replay")
        .events()
        .map(|r| r.expect("replayed event"))
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    replayed
}

fn first_taker_fill(events: &[VenueEvent]) -> Arc<core_types::Fill> {
    events
        .iter()
        .find_map(|ev| match ev {
            VenueEvent::Fill(f) => Some(Arc::clone(f)),
            VenueEvent::Order(_) | VenueEvent::Connectivity { .. } => None,
        })
        .expect("a taker fill")
}

// ---- (a) end-to-end take vs a stale book, on recorded data ----------------

#[tokio::test(start_paused = true)]
async fn takes_and_fills_against_the_stale_book_on_recorded_data() {
    let (venue, mut rx, mut risk, start) = make(MomentumTakerParams::default());
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    let mut session = vec![
        window_event(WindowLifecycle::Open),
        book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS),
    ];
    session.extend(up_ramp(BASE_MS));
    session.push(model(0.85, 1e-4, BASE_MS));

    let replayed = record_then_replay(&session, "take");
    assert_eq!(
        replayed, session,
        "the driving session round-trips losslessly"
    );

    for ev in &replayed {
        feed(&venue, &mut risk, ev, &now).await;
    }
    assert_eq!(risk.momentum_take_count(), 1, "exactly one FAK fired");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = drain(&mut rx, &mut risk, &venue, &now).await;

    // Budget caps the take: $10 / 0.80 = 12.5 shares, $10 notional.
    let fill = first_taker_fill(&events);
    assert_eq!(fill.liquidity, Liquidity::Taker);
    assert_eq!(fill.price.as_decimal(), dec!(0.80));
    assert_eq!(fill.size, sz(dec!(12.5)));
    assert_eq!(
        fill.fee,
        taker_fee(sz(dec!(12.5)), dec!(0.07), px(dec!(0.80)))
    );

    assert_eq!(risk.momentum_realized_spent(), Dollars::new(dec!(10)));
    assert!(risk.momentum_effective_spent().as_decimal() <= dec!(10));
    assert!(
        !risk.state_snapshot().any_tripped(),
        "the guard was transparent — no breaker tripped"
    );
    venue.shutdown();
}

// ---- (a2) the clip cap splits one take into several FAKs ------------------

#[tokio::test(start_paused = true)]
async fn clip_splits_the_take_into_multiple_faks() {
    // A 5-share clip splits the $10 / 12.5-share take into three FAK clips
    // (5, 5, 2.5 shares), each ≤ the clip, that together fill the full size.
    let risk = momentum_only_risk_with_clip(MomentumTakerParams::default(), dec!(5));
    let (venue, mut rx, mut risk, start) = make_with_risk(risk);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    feed(
        &venue,
        &mut risk,
        &window_event(WindowLifecycle::Open),
        &now,
    )
    .await;
    feed(
        &venue,
        &mut risk,
        &book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS),
        &now,
    )
    .await;
    for ev in up_ramp(BASE_MS) {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    feed(&venue, &mut risk, &model(0.85, 1e-4, BASE_MS), &now).await;

    assert_eq!(
        risk.momentum_take_count(),
        3,
        "the take split into three FAK clips"
    );

    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = drain(&mut rx, &mut risk, &venue, &now).await;

    // Every clip fills at 0.80; the sizes sum to the full 12.5-share plan.
    let mut total = Decimal::ZERO;
    for ev in &events {
        if let VenueEvent::Fill(f) = ev {
            assert_eq!(f.liquidity, Liquidity::Taker);
            assert_eq!(f.price.as_decimal(), dec!(0.80));
            assert!(
                f.size.as_decimal() <= dec!(5),
                "each clip fill is ≤ the 5-share clip (got {})",
                f.size
            );
            total += f.size.as_decimal();
        }
    }
    assert_eq!(total, dec!(12.5), "clips fill the full planned size");
    assert!(
        risk.momentum_realized_spent().as_decimal() <= dec!(10),
        "spend stays within the per-window budget"
    );
    assert!(!risk.state_snapshot().any_tripped(), "guard transparent");
    venue.shutdown();
}

// ---- (b) cooldown blocks a second take, then releases ---------------------

#[tokio::test(start_paused = true)]
async fn cooldown_blocks_a_second_take_then_releases() {
    let taker_params = MomentumTakerParams {
        budget_per_window: Dollars::new(dec!(100)),
        ..MomentumTakerParams::default()
    };
    let (venue, mut rx, mut risk, start) = make(taker_params);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    feed(
        &venue,
        &mut risk,
        &window_event(WindowLifecycle::Open),
        &now,
    )
    .await;
    feed(
        &venue,
        &mut risk,
        &book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS),
        &now,
    )
    .await;
    for ev in up_ramp(BASE_MS) {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    feed(&venue, &mut risk, &model(0.85, 1e-4, BASE_MS), &now).await;
    assert_eq!(risk.momentum_take_count(), 1, "first take fired");

    feed(
        &venue,
        &mut risk,
        &book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS + 1),
        &now,
    )
    .await;
    assert_eq!(risk.momentum_take_count(), 1, "blocked by cooldown");

    tokio::time::sleep(Duration::from_secs(6)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;

    let base2 = now().as_millis();
    for ev in up_ramp(base2) {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    feed(
        &venue,
        &mut risk,
        &book(up_token(), &[(dec!(0.80), dec!(50))], base2),
        &now,
    )
    .await;
    feed(&venue, &mut risk, &model(0.85, 1e-4, base2), &now).await;
    assert_eq!(
        risk.momentum_take_count(),
        2,
        "cooldown released ⇒ second take fired"
    );

    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    assert!(
        risk.momentum_realized_spent().as_decimal() <= dec!(100),
        "spend {} within budget",
        risk.momentum_realized_spent()
    );
    assert!(risk.momentum_realized_spent().as_decimal() > dec!(0));
    assert!(!risk.state_snapshot().any_tripped(), "guard transparent");
    venue.shutdown();
}
