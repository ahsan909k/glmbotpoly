//! Momentum-taker integration tests against `venue-paper` (CLAUDE.md §8/§9).
//!
//! Drives [`engine::MomentumTaker`] through the [`venue_api::VenuePort`] surface,
//! backed by a real [`venue_paper::PaperVenue`] over deterministic market data
//! under tokio's paused virtual clock — the same harness shape as
//! `quote_manager_paper.rs`. The taker and the paper venue share one `now`
//! closure, so the taker's injected `now` and the venue's latency deadlines
//! advance off the same clock.
//!
//! Asserts the required end-to-end properties:
//! - a confirmed fast-feed move + a stale-cheap book ⇒ exactly one FAK that
//!   fills against the displayed asks, with the realized taker fee equal to
//!   `taker_fee(filled, rate, price)` (the paper engine charges per level) and
//!   the realized spend bounded by the per-window budget — driven off
//!   journal-recorded → replayed events;
//! - the cooldown blocks a second take within the window and releases after it.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, Liquidity, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot,
    Price, PriceSource, PriceTick, ResolutionSource, RoundDir, Series, Size, TickKind, TickSize,
    TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle, taker_fee,
};
use engine::{MomentumTaker, MomentumTakerParams, NormalizerParams};
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
        ..PaperParams::default()
    }
}

// ---- harness --------------------------------------------------------------

/// Spawns a paper venue + taker bound to one paused virtual clock.
fn make(
    taker_params: MomentumTakerParams,
) -> (
    PaperVenue,
    Receiver<VenueEvent>,
    MomentumTaker,
    tokio::time::Instant,
) {
    let start = tokio::time::Instant::now();
    let mut venue = PaperVenue::spawn(params(), move || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    });
    let rx = venue.take_event_rx().expect("event rx");
    let taker = MomentumTaker::new(taker_params, NormalizerParams::default());
    (venue, rx, taker, start)
}

/// Feeds one bus event to both the paper venue (fill sim) and the taker.
async fn feed<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    taker: &mut MomentumTaker,
    ev: &Event,
    now: &F,
) {
    venue.on_bus_event(ev).await;
    taker.on_event(ev, venue, now()).await;
}

/// Drains every currently-ready venue event into the taker, collecting them.
fn drain<F: Fn() -> TimestampMs>(
    rx: &mut Receiver<VenueEvent>,
    taker: &mut MomentumTaker,
    now: &F,
) -> Vec<VenueEvent> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        taker.on_venue_event(&ev, now());
        out.push(ev);
    }
    out
}

/// Records `session` through the real journal recorder and replays it back —
/// proving the driving data round-trips, and yielding the events to drive off
/// (mirrors `quote_manager_paper.rs` / `inventory_replay.rs`).
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
            VenueEvent::Order(_) => None,
        })
        .expect("a taker fill")
}

// ---- (a) end-to-end take vs a stale book, on recorded data ----------------

#[tokio::test(start_paused = true)]
async fn takes_and_fills_against_the_stale_book_on_recorded_data() {
    let (venue, mut rx, mut taker, start) = make(MomentumTakerParams::default());
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    // Session: window opens, the Up book is stale-cheap (asks at 0.80 while the
    // model fair is 0.85), a fresh up-move on the signal feed, then a Ready model
    // snapshot that triggers the take.
    let mut session = vec![
        window_event(WindowLifecycle::Open),
        book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS),
    ];
    session.extend(up_ramp(BASE_MS)); // ticks ending at BASE_MS
    session.push(model(0.85, 1e-4, BASE_MS));

    let replayed = record_then_replay(&session, "take");
    assert_eq!(
        replayed, session,
        "the driving session round-trips losslessly"
    );

    for ev in &replayed {
        feed(&venue, &mut taker, ev, &now).await;
    }
    // The model event triggered the take synchronously; let the placement latency
    // (200 ms) fire the fill, then fold the venue stream.
    assert_eq!(taker.take_count(), 1, "exactly one FAK fired");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let events = drain(&mut rx, &mut taker, &now);

    // Budget caps the take: $10 / 0.80 = 12.5 shares, $10 notional.
    let fill = first_taker_fill(&events);
    assert_eq!(fill.liquidity, Liquidity::Taker);
    assert_eq!(fill.price.as_decimal(), dec!(0.80));
    assert_eq!(fill.size, sz(dec!(12.5)));
    // The realized fee equals the exact per-level taker fee the venue charges.
    assert_eq!(
        fill.fee,
        taker_fee(sz(dec!(12.5)), dec!(0.07), px(dec!(0.80)))
    );

    // Budget: realized spend is exactly the $10 budget, never above it.
    assert_eq!(taker.realized_spent(), Dollars::new(dec!(10)));
    assert!(taker.effective_spent().as_decimal() <= dec!(10));
    venue.shutdown();
}

// ---- (b) cooldown blocks a second take, then releases ---------------------

#[tokio::test(start_paused = true)]
async fn cooldown_blocks_a_second_take_then_releases() {
    // A roomy budget so the budget cap doesn't mask the cooldown.
    let taker_params = MomentumTakerParams {
        budget_per_window: Dollars::new(dec!(100)),
        ..MomentumTakerParams::default()
    };
    let (venue, mut rx, mut taker, start) = make(taker_params);
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };

    // First take.
    feed(
        &venue,
        &mut taker,
        &window_event(WindowLifecycle::Open),
        &now,
    )
    .await;
    feed(
        &venue,
        &mut taker,
        &book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS),
        &now,
    )
    .await;
    for ev in up_ramp(BASE_MS) {
        feed(&venue, &mut taker, &ev, &now).await;
    }
    feed(&venue, &mut taker, &model(0.85, 1e-4, BASE_MS), &now).await;
    assert_eq!(taker.take_count(), 1, "first take fired");

    // A second trigger within the cooldown (a fresh book) must NOT fire.
    feed(
        &venue,
        &mut taker,
        &book(up_token(), &[(dec!(0.80), dec!(50))], BASE_MS + 1),
        &now,
    )
    .await;
    assert_eq!(taker.take_count(), 1, "blocked by cooldown");

    // Advance past the 5 s cooldown (and let the first fill settle).
    tokio::time::sleep(Duration::from_secs(6)).await;
    let _ = drain(&mut rx, &mut taker, &now);

    // A fresh move + book + model after the cooldown fires a second take.
    let base2 = now().as_millis();
    for ev in up_ramp(base2) {
        feed(&venue, &mut taker, &ev, &now).await;
    }
    feed(
        &venue,
        &mut taker,
        &book(up_token(), &[(dec!(0.80), dec!(50))], base2),
        &now,
    )
    .await;
    feed(&venue, &mut taker, &model(0.85, 1e-4, base2), &now).await;
    assert_eq!(
        taker.take_count(),
        2,
        "cooldown released ⇒ second take fired"
    );

    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut taker, &now);
    // Two ~$40 takes (50 sh @ 0.80) stay within the $100 budget.
    assert!(
        taker.realized_spent().as_decimal() <= dec!(100),
        "spend {} within budget",
        taker.realized_spent()
    );
    assert!(taker.realized_spent().as_decimal() > dec!(0));
    venue.shutdown();
}
