//! Quote-manager integration tests against `venue-paper` (CLAUDE.md §8).
//!
//! Drives [`engine::QuoteManager`] through the [`venue_api::VenuePort`] surface,
//! backed by a real [`venue_paper::PaperVenue`] over deterministic market data
//! under tokio's paused virtual clock. The manager and the paper venue share one
//! `now` closure, so the manager's injected `now` and the venue's latency
//! deadlines advance off the same clock.
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

use std::collections::HashSet;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, DurationMs, Event,
    FeeParams, InputAges, InventorySnapshot, MarketInfo, ModelHealth, ModelHealthReason,
    ModelSnapshot, Outcome, Price, ResolutionSource, RoundDir, Series, SideInventory, Size,
    TickSize, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{
    NormalizerParams, QuoteDecision, QuoteManager, QuoteManagerParams, QuoteParams,
    calculate_quotes,
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
        ..PaperParams::default()
    }
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

/// The manager's current resting orders as `(outcome, price, size)`, sorted.
fn resting_levels(manager: &QuoteManager) -> Vec<(Outcome, Decimal, Decimal)> {
    let mut v: Vec<(Outcome, Decimal, Decimal)> = manager
        .resting_view()
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

/// Spawns a paper venue + manager bound to one paused virtual clock; returns the
/// venue, its event rx, the manager, and the clock anchor (build `now` from it).
fn make(
    qm: QuoteManagerParams,
) -> (
    PaperVenue,
    Receiver<VenueEvent>,
    QuoteManager,
    tokio::time::Instant,
) {
    let start = tokio::time::Instant::now();
    let mut venue = PaperVenue::spawn(params(), move || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    });
    let rx = venue.take_event_rx().expect("event rx");
    let manager = QuoteManager::new(qm, QuoteParams::default(), NormalizerParams::default());
    (venue, rx, manager, start)
}

/// Feeds one bus event to both the paper venue (fill sim) and the manager
/// (quoting) — the data flow the real orchestrator wires.
async fn feed<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    manager: &mut QuoteManager,
    ev: &Event,
    now: &F,
) {
    venue.on_bus_event(ev).await;
    manager.on_event(ev, venue, now()).await;
}

/// Drains every currently-ready venue event into the manager's view + a record.
fn drain<F: Fn() -> TimestampMs>(
    rx: &mut Receiver<VenueEvent>,
    manager: &mut QuoteManager,
    now: &F,
    recorded: &mut Vec<VenueEvent>,
) {
    while let Ok(ev) = rx.try_recv() {
        manager.on_venue_event(&ev, now());
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

/// Records the driving session through the journal, replays it, drives the
/// manager+paper venue off the *replayed* events, and asserts the resting orders
/// equal the calculator's desired ladder. Covers "convergence" + "on recorded
/// data" (mirrors `inventory_replay.rs`).
#[tokio::test(start_paused = true)]
async fn converges_to_the_calculator_ladder_on_recorded_data() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut manager, start) = make(qm);
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
        feed(&venue, &mut manager, ev, &now).await;
    }

    // Converge, then let the placement latency fire the Open events.
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut manager, &now, &mut recorded);

    let expected = expected_levels(0.50, 0.0005);
    assert_eq!(expected.len(), 6, "two-sided 3-level ladder");
    assert_eq!(
        resting_levels(&manager),
        expected,
        "resting orders == desired ladder"
    );

    // The venue confirmed all six as Open at the desired prices.
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

    // A second tick is a no-op (already converged).
    manager.on_requote_tick(&venue, now()).await;
    assert_eq!(manager.place_cycle_count(), 1, "no churn once converged");
    venue.shutdown();
}

// ---- (b) cancel-before-replace under a price jump --------------------------

#[tokio::test(start_paused = true)]
async fn cancels_before_replacing_under_a_price_jump() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut manager, start) = make(qm);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    // Converge at p_up = 0.50.
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
        feed(&venue, &mut manager, &ev, &now).await;
    }
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut first = Vec::new();
    drain(&mut rx, &mut manager, &now, &mut first);
    assert_eq!(
        resting_levels(&manager).len(),
        6,
        "converged before the jump"
    );
    let pre_ids: HashSet<String> = manager
        .resting_view()
        .unwrap()
        .live_orders()
        .map(|o| o.order_id.as_str().to_owned())
        .collect();

    // A large jump to 0.85: |Δ| = 0.35 > cancel_market_theta (0.02). Feed new
    // books so the fresh high/low quotes don't cross. Record only post-jump events.
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
        feed(&venue, &mut manager, &ev, &now).await;
    }
    // The urgent path issued a bulk cancel; let it land, then fold the Canceleds.
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut manager, &now, &mut recorded);
    assert!(
        manager.resting_view().unwrap().is_empty(),
        "old quotes withdrawn first"
    );

    // The next converge places the fresh ladder into the now-empty slots.
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    drain(&mut rx, &mut manager, &now, &mut recorded);

    // Cancel-before-replace: every old Canceled strictly precedes every new Open.
    let cancel_idx = order_states(&recorded, core_types::OrderState::Canceled);
    let open_idx = order_states(&recorded, core_types::OrderState::Open);
    assert_eq!(cancel_idx.len(), 6, "all six old quotes cancelled");
    assert_eq!(open_idx.len(), 6, "all six replacements opened");
    assert!(
        cancel_idx.iter().max() < open_idx.iter().min(),
        "every Canceled precedes every Open (split-cycle): cancels {cancel_idx:?}, opens {open_idx:?}"
    );

    // The replacements are brand-new order ids at the 0.85 ladder prices.
    let new_ids: HashSet<String> = manager
        .resting_view()
        .unwrap()
        .live_orders()
        .map(|o| o.order_id.as_str().to_owned())
        .collect();
    assert!(
        new_ids.is_disjoint(&pre_ids),
        "replacements use new order ids"
    );
    assert_eq!(
        resting_levels(&manager),
        expected_levels(0.85, 0.0005),
        "fresh ladder at 0.85"
    );
    venue.shutdown();
}

// ---- (c) rate-budget compliance -------------------------------------------

#[tokio::test(start_paused = true)]
async fn respects_the_per_window_rate_budget() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 1000,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut manager, start) = make(qm);
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
        feed(&venue, &mut manager, &ev, &now).await;
    }
    manager.on_requote_tick(&venue, now()).await;
    assert_eq!(
        manager.place_cycle_count(),
        1,
        "initial converge = one placement cycle"
    );

    // Many rapid supra-θ model ticks within ONE interval (no virtual time passes
    // between them). Each marks dirty; each on_requote_tick is budget-blocked, so
    // no further placement cycle runs.
    for i in 0..8 {
        let p = if i % 2 == 0 { 0.49 } else { 0.51 };
        feed(&venue, &mut manager, &model(p, 0.0005, BASE_MS), &now).await;
        manager.on_requote_tick(&venue, now()).await;
    }
    assert_eq!(
        manager.place_cycle_count(),
        1,
        "all re-placements throttled within the rate-budget interval"
    );

    // After the interval elapses (and the urgent cancels terminalize), the budget
    // releases and exactly one more placement cycle reconciles the freed slots.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut manager, &now, &mut recorded);
    manager.on_requote_tick(&venue, now()).await;
    assert!(
        manager.place_cycle_count() >= 2,
        "the budget releases after the interval (got {})",
        manager.place_cycle_count()
    );
    venue.shutdown();
}

// ---- (d) clean final-seconds withdrawal -----------------------------------

#[tokio::test(start_paused = true)]
async fn withdraws_cleanly_on_window_closing() {
    let qm = QuoteManagerParams {
        min_requote_interval_ms: 10,
        ..QuoteManagerParams::default()
    };
    let (venue, mut rx, mut manager, start) = make(qm);
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
        feed(&venue, &mut manager, &ev, &now).await;
    }
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut manager, &now, &mut recorded);
    assert_eq!(
        resting_levels(&manager).len(),
        6,
        "converged before closing"
    );
    let cycles_at_close = manager.place_cycle_count();

    // Window Closing → cancel-all + stand down for the window.
    feed(
        &venue,
        &mut manager,
        &window_event(WindowLifecycle::Closing),
        &now,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut manager, &now, &mut recorded);
    assert!(
        manager.resting_view().unwrap().is_empty(),
        "all quotes withdrawn on Closing"
    );

    // Subsequent ticks place nothing while closing.
    feed(
        &venue,
        &mut manager,
        &model(0.50, 0.0005, BASE_MS + 1000),
        &now,
    )
    .await;
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    drain(&mut rx, &mut manager, &now, &mut recorded);
    assert!(
        manager.resting_view().unwrap().is_empty(),
        "no re-quoting after Closing"
    );
    assert_eq!(
        manager.place_cycle_count(),
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
    let (venue, mut rx, mut manager, start) = make(qm);
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
        feed(&venue, &mut manager, &ev, &now).await;
    }
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut recorded = Vec::new();
    drain(&mut rx, &mut manager, &now, &mut recorded);
    assert_eq!(resting_levels(&manager).len(), 6, "converged mid-window");

    // Advance virtual time to within the final-seconds window: τ = close − now =
    // 300s − 296s = 4s ≤ no_passive_final_secs (5s). The calculator then returns
    // NoQuote(FinalSecondsNoPassive) → the planner pulls the whole market.
    tokio::time::sleep(Duration::from_secs(296)).await;
    feed(
        &venue,
        &mut manager,
        &model(0.50, 0.0005, BASE_MS + 296_000),
        &now,
    )
    .await;
    manager.on_requote_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    drain(&mut rx, &mut manager, &now, &mut recorded);

    assert!(
        manager.resting_view().unwrap().is_empty(),
        "withdrawn in the final seconds"
    );
    venue.shutdown();
}
