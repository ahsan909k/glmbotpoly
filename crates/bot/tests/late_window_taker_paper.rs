//! Late-window certainty-taker integration tests, now driven through the
//! [`engine::RiskManager`] gateway against `venue-paper` (CLAUDE.md §8/§9/§11).
//!
//! The taker driver is `pub(crate)` in `engine`; the only path to the venue is
//! `RiskManager`. Only the late-window taker is enabled here, and with no breaker
//! tripped the guard is transparent — the certainty-take, fill, fee and budget
//! assertions hold exactly as before. The window is short (`CLOSE_MS = BASE_MS +
//! 20_000`) so τ ≈ 20 s sits inside the 30 s late-window threshold for the feed.
//!
//! Asserts the required end-to-end properties:
//! - a Ready, **Chainlink-anchored** model beyond the certainty threshold + a
//!   within-cap book ⇒ exactly one FAK that fills against the displayed asks, with
//!   the realized taker fee equal to `taker_fee(filled, rate, price)` and the
//!   realized spend bounded by the per-window budget — driven off journal-recorded
//!   → replayed events;
//! - the same inputs anchored on the fast feed (`BinanceCorrected`) ⇒ no take.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, Liquidity, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot,
    Price, ResolutionSource, RoundDir, Series, Size, TickSize, TimestampMs, TokenId, TokenPair,
    WindowDuration, WindowId, WindowLifecycle, taker_fee,
};
use engine::{LateWindowTakerParams, RiskManager, RiskParams};
use journal::{Recorder, RecorderParams, ReplayReader};
use rust_decimal::dec;
use tokio::sync::mpsc::Receiver;
use venue_api::{VenueEvent, VenueEvents};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;
/// A short window so τ ≈ 20 s ≤ the 30 s late-window threshold throughout.
const CLOSE_MS: i64 = BASE_MS + 20_000;
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
        event_slug: "btc-updown-5m-lw".to_owned(),
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

/// A Ready model with the given `p_up` and `anchor` on `window()`.
fn model(p_up: f64, anchor: AnchorSource, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window()),
        p_up,
        z: 0.0,
        sigma_1s: 1e-4,
        sigma_tau: 1e-4,
        basis: 0.0,
        anchor,
        health: ModelHealth::Ready,
        reason: ModelHealthReason::Nominal,
        input_ages: InputAges {
            chainlink: DurationMs::from_millis(100),
            binance: DurationMs::from_millis(50),
        },
        ts: TimestampMs::from_millis(ts),
    })
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
        ..PaperParams::default()
    }
}

/// A late-window-only risk manager (quoter + momentum disabled) with a roomy loss
/// cap so no §11 breaker trips on this data — the guard is transparent.
fn late_only_risk(lw: LateWindowTakerParams) -> RiskManager {
    let params = RiskParams {
        late_window: lw,
        quoter_enabled: false,
        momentum_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(Decimal::from(10_000)));
    RiskManager::new(params, caps)
}

// ---- harness --------------------------------------------------------------

fn make(
    taker_params: LateWindowTakerParams,
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
    (venue, rx, late_only_risk(taker_params), start)
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
    let dir = std::env::temp_dir().join(format!("lw_paper_{dir_tag}"));
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

// ---- (a) end-to-end certainty take vs a within-cap book, on recorded data --

#[tokio::test(start_paused = true)]
async fn takes_the_near_certain_side_within_cap_on_recorded_data() {
    let (venue, mut rx, mut risk, start) = make(LateWindowTakerParams::default());
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    let session = vec![
        window_event(WindowLifecycle::Open),
        book(up_token(), &[(dec!(0.98), dec!(10))], BASE_MS),
        model(0.99, AnchorSource::Chainlink, BASE_MS),
    ];

    let replayed = record_then_replay(&session, "take");
    assert_eq!(
        replayed, session,
        "the driving session round-trips losslessly"
    );

    for ev in &replayed {
        feed(&venue, &mut risk, ev, &now).await;
    }
    assert_eq!(risk.late_take_count(), 1, "exactly one FAK fired");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = drain(&mut rx, &mut risk, &venue, &now).await;

    let fill = first_taker_fill(&events);
    assert_eq!(fill.liquidity, Liquidity::Taker);
    assert_eq!(fill.price.as_decimal(), dec!(0.98), "filled within the cap");
    assert_eq!(fill.size, sz(dec!(10)), "displayed depth bound the size");
    assert_eq!(
        fill.fee,
        taker_fee(sz(dec!(10)), dec!(0.07), px(dec!(0.98))),
        "realized fee is the exact per-level taker fee"
    );

    assert_eq!(risk.late_realized_spent(), Dollars::new(dec!(9.80)));
    assert!(risk.late_realized_spent().as_decimal() <= dec!(10));
    assert!(
        !risk.state_snapshot().any_tripped(),
        "the guard was transparent — no breaker tripped"
    );
    venue.shutdown();
}

// ---- (b) the fast feed alone never triggers a take ------------------------

#[tokio::test(start_paused = true)]
async fn fast_feed_anchored_fair_does_not_take() {
    let (venue, mut rx, mut risk, start) = make(LateWindowTakerParams::default());
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    let session = vec![
        window_event(WindowLifecycle::Open),
        book(up_token(), &[(dec!(0.98), dec!(10))], BASE_MS),
        model(0.99, AnchorSource::BinanceCorrected, BASE_MS),
    ];
    for ev in &session {
        feed(&venue, &mut risk, ev, &now).await;
    }
    assert_eq!(
        risk.late_take_count(),
        0,
        "no take on a fast-feed-anchored fair"
    );

    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = drain(&mut rx, &mut risk, &venue, &now).await;
    assert!(
        !events.iter().any(|e| matches!(e, VenueEvent::Fill(_))),
        "no fills"
    );
    assert_eq!(risk.late_realized_spent(), Dollars::ZERO);
    let _ = down_token(); // silence unused in this scenario
    venue.shutdown();
}
