//! Quote-manager integration tests, now driven through the [`engine::RiskManager`]
//! gateway against `venue-paper` (CLAUDE.md §8/§11).
//!
//! The strategy drivers are `pub(crate)` in `engine` — the only path to the venue
//! is `RiskManager`, which interposes the risk guard. With **no breakers tripped**
//! the guard is transparent, so every property below holds exactly as it did when
//! the quote manager was driven directly, and each test additionally asserts the
//! gateway stayed a no-op (`!state_snapshot().any_tripped()`). Only the quote
//! manager is enabled (the takers never fire on this benign data anyway, but
//! disabling them keeps the venue stream uncluttered).
//!
//! Asserts the four required properties:
//! - **convergence** — after a stable model/book the resting orders equal the
//!   calculator's desired ladder (driven off journal-recorded → replayed data);
//! - **cancel-before-replace** — under a price jump the old quotes' `Canceled`
//!   events strictly precede the replacements' `Open` events (split-cycle);
//! - **rate-budget compliance** — many rapid triggers in one interval yield at
//!   most one placement cycle; the budget releases after the interval;
//! - **clean final-seconds withdrawal** — Window `Closing` and the τ ≤
//!   `no_passive_final_secs` gate both end at zero resting + no new placements.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, InventorySnapshot, MarketInfo, ModelHealth, ModelHealthReason,
    ModelSnapshot, Outcome, Price, ResolutionSource, RoundDir, Series, SideInventory, Size,
    TickSize, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{
    QuoteDecision, QuoteManagerParams, QuoteParams, RiskManager, RiskParams, calculate_quotes,
};
use journal::{Recorder, RecorderParams, ReplayReader};
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
        event_slug: "btc-updown-5m-qm".to_owned(),
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
        market: std::sync::Arc::new(market()),
        lifecycle,
    }
}

/// A book for one token. `bids`/`asks` are `(price, size)`, best-first.
fn book(
    token: TokenId,
    bids: &[(Decimal, Decimal)],
    asks: &[(Decimal, Decimal)],
    ts: i64,
) -> Event {
    Event::Book(std::sync::Arc::new(BookSnapshot {
        token_id: token,
        condition_id: condition(),
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

fn model_snapshot(p_up: f64, sigma: f64, ts: i64) -> ModelSnapshot {
    ModelSnapshot {
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
    }
}

fn model(p_up: f64, sigma: f64, ts: i64) -> Event {
    Event::Model(model_snapshot(p_up, sigma, ts))
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
        // Quote-manager convergence is maker-only (post-only), unaffected by the venue taker
        // delay; zero it to keep the params pinned to this test's timing assumptions.
        venue_taker_delay_ms: DurationMs::ZERO,
        ..PaperParams::default()
    }
}

/// A quoter-only risk manager (takers disabled) with a roomy loss cap so no §11
/// breaker trips on this benign data — the guard is transparent.
fn quoter_only_risk(qm: QuoteManagerParams) -> RiskManager {
    let params = RiskParams {
        quote_manager: qm,
        momentum_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(Decimal::from(10_000)));
    RiskManager::new(params, caps)
}

/// The desired ladder the calculator produces for `(p_up, sigma)` with an empty
/// inventory — `(outcome, price, size)` sorted. The manager must converge to
/// exactly this.
fn expected_levels(p_up: f64, sigma: f64) -> Vec<(Outcome, Decimal, Decimal)> {
    let inv = InventorySnapshot::derive(
        window(),
        SideInventory::default(),
        SideInventory::default(),
        TimestampMs::from_millis(BASE_MS),
    );
    let snap = model_snapshot(p_up, sigma, BASE_MS);
    let fees = market().fees;
    let qp = QuoteParams::default();
    match calculate_quotes(&snap, &inv, &fees, &qp, 250.0, TickSize::T001) {
        QuoteDecision::Quote(qs) => {
            let mut v: Vec<(Outcome, Decimal, Decimal)> = qs
                .levels
                .iter()
                .map(|l| (l.outcome, l.price.as_decimal(), l.size.as_decimal()))
                .collect();
            v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            v
        }
        QuoteDecision::NoQuote(r) => panic!("expected a quote for p_up={p_up}, got {r:?}"),
    }
}

/// The quoter's current resting orders as `(outcome, price, size)`, sorted —
/// delegated through the risk manager.
fn resting_levels(risk: &RiskManager) -> Vec<(Outcome, Decimal, Decimal)> {
    let mut v: Vec<(Outcome, Decimal, Decimal)> = risk
        .resting_view_for(window())
        .expect("active window")
        .live_orders()
        .map(|o| {
            (
                o.outcome,
                o.price.as_decimal(),
                o.original_size.as_decimal(),
            )
        })
        .collect();
    v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    v
}

// ---- harness --------------------------------------------------------------

/// Spawns a paper venue + risk manager bound to one paused virtual clock.
fn make(
    qm: QuoteManagerParams,
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
    (venue, rx, quoter_only_risk(qm), start)
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

/// Drains every currently-ready venue event into the risk manager (which folds
/// the quoter's view + risk accounting) and a record. Async because the gateway's
/// `on_venue_event` may itself issue cancels through the port.
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

/// Records `session` through the real journal recorder and replays it back —
/// proving the driving data round-trips, and yielding the events to drive off.
fn record_then_replay(session: &[Event], dir_tag: &str) -> Vec<Event> {
    let dir = std::env::temp_dir().join(format!("qm_paper_{dir_tag}"));
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

fn order_states(recorded: &[VenueEvent], state: core_types::OrderState) -> Vec<usize> {
    recorded
        .iter()
        .enumerate()
        .filter_map(|(i, ev)| match ev {
            VenueEvent::Order(u) if u.state == state => Some(i),
            _ => None,
        })
        .collect()
}

// ---- (a) convergence, on journal-recorded data ----------------------------

#[tokio::test(start_paused = true)]
async fn converges_to_the_calculator_ladder_on_recorded_data() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut risk, start) = make(qm);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    let session = vec![
        window_event(WindowLifecycle::Open),
        book(
            up_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        book(
            down_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        model(0.50, 0.0005, BASE_MS),
    ];
    let replayed = record_then_replay(&session, "convergence");
    assert_eq!(
        replayed, session,
        "the driving session round-trips losslessly"
    );

    for ev in &replayed {
        feed(&venue, &mut risk, ev, &now).await;
    }

    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;

    let expected = expected_levels(0.50, 0.0005);
    assert_eq!(expected.len(), 6, "two-sided 3-level ladder");
    assert_eq!(
        resting_levels(&risk),
        expected,
        "resting orders == desired ladder"
    );

    let opened: HashSet<Decimal> = recorded
        .iter()
        .filter_map(|ev| match ev {
            VenueEvent::Order(u) if u.state == core_types::OrderState::Open => {
                Some(u.price.as_decimal())
            }
            _ => None,
        })
        .collect();
    let want: HashSet<Decimal> = expected.iter().map(|(_, p, _)| *p).collect();
    assert_eq!(opened, want, "venue Open prices == desired ladder prices");

    risk.on_tick(&venue, now()).await;
    assert_eq!(risk.place_cycle_count(), 1, "no churn once converged");
    assert!(
        !risk.state_snapshot().any_tripped(),
        "the guard was transparent — no breaker tripped"
    );
    venue.shutdown();
}

// ---- (b) cancel-before-replace under a price jump --------------------------

#[tokio::test(start_paused = true)]
async fn cancels_before_replacing_under_a_price_jump() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut risk, start) = make(qm);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    for ev in [
        window_event(WindowLifecycle::Open),
        book(
            up_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        book(
            down_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        model(0.50, 0.0005, BASE_MS),
    ] {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut first = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut first).await;
    assert_eq!(resting_levels(&risk).len(), 6, "converged before the jump");
    let pre_ids: HashSet<String> = risk
        .resting_view_for(window())
        .unwrap()
        .live_orders()
        .map(|o| o.order_id.as_str().to_owned())
        .collect();

    let mut recorded = Vec::new();
    for ev in [
        book(
            up_token(),
            &[(dec!(0.83), dec!(100))],
            &[(dec!(0.87), dec!(100))],
            BASE_MS + 1000,
        ),
        book(
            down_token(),
            &[(dec!(0.10), dec!(100))],
            &[(dec!(0.16), dec!(100))],
            BASE_MS + 1000,
        ),
        model(0.85, 0.0005, BASE_MS + 1000),
    ] {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    assert!(
        risk.resting_view_for(window()).unwrap().is_empty(),
        "old quotes withdrawn first"
    );

    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;

    let cancel_idx = order_states(&recorded, core_types::OrderState::Canceled);
    let open_idx = order_states(&recorded, core_types::OrderState::Open);
    assert_eq!(cancel_idx.len(), 6, "all six old quotes cancelled");
    assert_eq!(open_idx.len(), 6, "all six replacements opened");
    assert!(
        cancel_idx.iter().max() < open_idx.iter().min(),
        "every Canceled precedes every Open (split-cycle): cancels {cancel_idx:?}, opens {open_idx:?}"
    );

    let new_ids: HashSet<String> = risk
        .resting_view_for(window())
        .unwrap()
        .live_orders()
        .map(|o| o.order_id.as_str().to_owned())
        .collect();
    assert!(
        new_ids.is_disjoint(&pre_ids),
        "replacements use new order ids"
    );
    assert_eq!(
        resting_levels(&risk),
        expected_levels(0.85, 0.0005),
        "fresh ladder at 0.85"
    );
    assert!(!risk.state_snapshot().any_tripped(), "guard transparent");
    venue.shutdown();
}

// ---- (c) rate-budget compliance -------------------------------------------

#[tokio::test(start_paused = true)]
async fn respects_the_per_window_rate_budget() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 1000,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut risk, start) = make(qm);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    for ev in [
        window_event(WindowLifecycle::Open),
        book(
            up_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        book(
            down_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        model(0.50, 0.0005, BASE_MS),
    ] {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    risk.on_tick(&venue, now()).await;
    assert_eq!(
        risk.place_cycle_count(),
        1,
        "initial converge = one placement cycle"
    );

    for i in 0..8 {
        let p = if i % 2 == 0 { 0.49 } else { 0.51 };
        feed(&venue, &mut risk, &model(p, 0.0005, BASE_MS), &now).await;
        risk.on_tick(&venue, now()).await;
    }
    assert_eq!(
        risk.place_cycle_count(),
        1,
        "all re-placements throttled within the rate-budget interval"
    );

    tokio::time::sleep(Duration::from_millis(1100)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    risk.on_tick(&venue, now()).await;
    assert!(
        risk.place_cycle_count() >= 2,
        "the budget releases after the interval (got {})",
        risk.place_cycle_count()
    );
    assert!(!risk.state_snapshot().any_tripped(), "guard transparent");
    venue.shutdown();
}

// ---- (d) clean final-seconds withdrawal -----------------------------------

#[tokio::test(start_paused = true)]
async fn withdraws_cleanly_on_window_closing() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut risk, start) = make(qm);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    for ev in [
        window_event(WindowLifecycle::Open),
        book(
            up_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        book(
            down_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        model(0.50, 0.0005, BASE_MS),
    ] {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    assert_eq!(resting_levels(&risk).len(), 6, "converged before closing");
    let cycles_at_close = risk.place_cycle_count();

    feed(
        &venue,
        &mut risk,
        &window_event(WindowLifecycle::Closing),
        &now,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    assert!(
        risk.resting_view_for(window()).unwrap().is_empty(),
        "all quotes withdrawn on Closing"
    );

    feed(
        &venue,
        &mut risk,
        &model(0.50, 0.0005, BASE_MS + 1000),
        &now,
    )
    .await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    assert!(
        risk.resting_view_for(window()).unwrap().is_empty(),
        "no re-quoting after Closing"
    );
    assert_eq!(
        risk.place_cycle_count(),
        cycles_at_close,
        "no placements after Closing"
    );
    venue.shutdown();
}

#[tokio::test(start_paused = true)]
async fn withdraws_in_the_final_seconds_via_the_tau_gate() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut risk, start) = make(qm);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    for ev in [
        window_event(WindowLifecycle::Open),
        book(
            up_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        book(
            down_token(),
            &[(dec!(0.45), dec!(100))],
            &[(dec!(0.55), dec!(100))],
            BASE_MS,
        ),
        model(0.50, 0.0005, BASE_MS),
    ] {
        feed(&venue, &mut risk, &ev, &now).await;
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;
    assert_eq!(resting_levels(&risk).len(), 6, "converged mid-window");

    // Advance virtual time to within the final-seconds window: τ = 300s − 296s =
    // 4s ≤ no_passive_final_secs (5s) → NoQuote(FinalSecondsNoPassive).
    tokio::time::sleep(Duration::from_secs(296)).await;
    feed(
        &venue,
        &mut risk,
        &model(0.50, 0.0005, BASE_MS + 296_000),
        &now,
    )
    .await;
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut risk, &venue, &now, &mut recorded).await;

    assert!(
        risk.resting_view_for(window()).unwrap().is_empty(),
        "withdrawn in the final seconds"
    );
    venue.shutdown();
}
