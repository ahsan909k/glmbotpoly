//! Model-taker integration tests, driven through the [`engine::RiskManager`]
//! gateway against `venue-paper` (CLAUDE.md §8/§9/§11).
//!
//! The model taker is `pub(crate)` in `engine`; the only path to the venue is
//! `RiskManager::on_model_prediction`. Only the model taker is enabled here.
//! Asserts:
//! - a fresh prediction beyond θ fires exactly one FAK that fills against the
//!   displayed asks, with the realized taker fee equal to `taker_fee(...)` and the
//!   spend bounded by the per-window budget — off journal-recorded → replayed
//!   window/book events;
//! - a forwarded risk breaker stands the taker down (no fire);
//! - the open-notional gate vetoes the FAK (a `PlaceRejected` decision), with no
//!   breaker tripped — the §11 pre-trade guard.
//!
//! (Self-match skip/shrink and momentum-vs-model arbitration are proven in the
//! `engine::self_match` / `engine::arbitration` / `engine::model_taker` unit tests.)

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    Asset, BookLevel, BookSnapshot, ConditionId, ControlEvent, Decimal, Dollars, DurationMs, Event,
    FeeParams, Liquidity, MarketInfo, Price, ResolutionSource, RoundDir, Series, Size, TickSize,
    TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle, taker_fee,
};
use engine::{ModelPrediction, ModelTakeOutcome, RiskManager, RiskParams};
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
        event_slug: "btc-updown-5m-md".to_owned(),
        condition_id: condition(),
        tokens: TokenPair {
            up: up_token(),
            down: TokenId::new("222").expect("token"),
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

fn book(asks: &[(Decimal, Decimal)], ts: i64) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: up_token(),
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

fn pred(p_up: f64, ts: i64) -> ModelPrediction {
    ModelPrediction {
        series: window().series,
        window_open_ms: BASE_MS,
        ts: TimestampMs::from_millis(ts),
        p_up,
        finite_count: 24,
        model_stale: false,
    }
}

/// Latencies pinned: place 200 ms, no jitter, fixed seed, zero venue-taker delay
/// (this tests the model taker's engine logic, not the venue delay).
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
        venue_taker_delay_ms: DurationMs::ZERO,
        ..PaperParams::default()
    }
}

/// A model-only risk manager (quoter + momentum + late disabled) with the given
/// open-notional cap and a roomy loss cap.
fn model_only_risk(max_open_notional: Decimal) -> RiskManager {
    let params = RiskParams {
        model_enabled: true,
        quoter_enabled: false,
        momentum_enabled: false,
        late_window_enabled: false,
        max_open_notional: Dollars::new(max_open_notional),
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(Decimal::from(10_000)));
    RiskManager::new(params, caps)
}

fn make(
    max_open_notional: Decimal,
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
    (venue, rx, model_only_risk(max_open_notional), start)
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

fn record_then_replay(session: &[Event], dir_tag: &str) -> Vec<Event> {
    let dir = std::env::temp_dir().join(format!("md_paper_{dir_tag}"));
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

// ---- (a) a prediction fires a FAK that fills, off recorded data ------------

#[tokio::test(start_paused = true)]
async fn prediction_fires_and_fills_against_the_book() {
    let (venue, mut rx, mut risk, start) = make(dec!(1000));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    // The window/book drive off journal-recorded → replayed events.
    let session = vec![
        window_event(WindowLifecycle::Open),
        book(&[(dec!(0.80), dec!(50))], BASE_MS),
    ];
    let replayed = record_then_replay(&session, "fire");
    assert_eq!(replayed, session, "the driving session round-trips losslessly");
    for ev in &replayed {
        feed(&venue, &mut risk, ev, &now).await;
    }

    // p_up 0.85 ≥ 0.5 + θ ⇒ buy Up; a $10 budget caps 12.5 sh @ 0.80.
    let out = risk
        .on_model_prediction(&pred(0.85, BASE_MS), &venue, now())
        .await;
    assert!(
        matches!(out, Some(ModelTakeOutcome::Fired { .. })),
        "prediction fired: {out:?}"
    );
    assert_eq!(risk.model_take_count(), 1, "exactly one FAK fired");

    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = drain(&mut rx, &mut risk, &venue, &now).await;
    let fill = first_taker_fill(&events);
    assert_eq!(fill.liquidity, Liquidity::Taker);
    assert_eq!(fill.price.as_decimal(), dec!(0.80));
    assert_eq!(fill.size, sz(dec!(12.5)));
    assert_eq!(fill.fee, taker_fee(sz(dec!(12.5)), dec!(0.07), px(dec!(0.80))));
    assert_eq!(risk.model_realized_spent(), Dollars::new(dec!(10)));
    assert!(!risk.state_snapshot().any_tripped(), "guard transparent");
    venue.shutdown();
}

// ---- (b) a forwarded breaker stands the model taker down ------------------

#[tokio::test(start_paused = true)]
async fn a_breaker_stands_the_model_taker_down() {
    let (venue, _rx, mut risk, start) = make(dec!(1000));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    feed(&venue, &mut risk, &window_event(WindowLifecycle::Open), &now).await;
    feed(&venue, &mut risk, &book(&[(dec!(0.80), dec!(50))], BASE_MS), &now).await;

    // A manual kill trips the global halt and forwards the breaker to the taker.
    feed(&venue, &mut risk, &Event::Control(ControlEvent::Kill), &now).await;
    assert!(risk.model_standing_down(), "taker stood down on the breaker");

    let out = risk
        .on_model_prediction(&pred(0.85, now().as_millis()), &venue, now())
        .await;
    assert!(
        matches!(out, Some(ModelTakeOutcome::Suppressed(_))),
        "suppressed while standing down: {out:?}"
    );
    assert_eq!(risk.model_take_count(), 0, "no fire while halted");
    venue.shutdown();
}

// ---- (c) the open-notional gate vetoes the FAK (PlaceRejected) -------------

#[tokio::test(start_paused = true)]
async fn open_notional_gate_vetoes_the_take() {
    // A $1 open-notional cap; the model FAK's $10 notional exceeds it, so the
    // guard refuses the place — no breaker, a PlaceRejected decision.
    let (venue, _rx, mut risk, start) = make(dec!(1));
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    feed(&venue, &mut risk, &window_event(WindowLifecycle::Open), &now).await;
    feed(&venue, &mut risk, &book(&[(dec!(0.80), dec!(50))], BASE_MS), &now).await;

    let out = risk
        .on_model_prediction(&pred(0.85, BASE_MS), &venue, now())
        .await;
    assert!(
        matches!(
            out,
            Some(ModelTakeOutcome::Suppressed(
                engine::NoModelTakeReason::PlaceRejected
            ))
        ),
        "the gate rejected the FAK: {out:?}"
    );
    assert_eq!(risk.model_take_count(), 0, "no fire when gate-vetoed");
    assert!(!risk.state_snapshot().any_tripped(), "a cap veto is not a breaker");
    venue.shutdown();
}
