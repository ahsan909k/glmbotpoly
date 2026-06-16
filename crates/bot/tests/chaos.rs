//! Deterministic, paper-only **chaos suite** (CLAUDE.md §5/§9/§11).
//!
//! Feature-gated (`--features chaos`, run via `just chaos`) so the heavy
//! fault-injection suite never burdens the default check/test loop (§3). Built
//! on the [`risk_manager_paper`](super)-style harness: the single-gateway
//! [`engine::RiskManager`] + `venue-paper` + a real [`journal::Recorder`], all
//! under `#[tokio::test(start_paused = true)]` with a seeded paper venue, so
//! every run is byte-for-byte reproducible.
//!
//! Each scenario injects one scripted fault and asserts the four non-negotiable
//! invariants where they apply:
//! 1. **No orphan open orders** — after the trip every resting order is
//!    `Canceled` and the quoter's resting view is empty.
//! 2. **The right breaker trips with a journaled cause** — the published
//!    `BreakerTripped` is recorded, then queried back via
//!    [`journal::JournalIndexReader::breaker_trips`] and found in the
//!    [`journal::ReplayReader`] stream.
//! 3. **Trading halts and resumes exactly per the rules** — halted while the
//!    fault holds, no placements while halted, resumes only on recovery/reset.
//! 4. **State rebuilds from the journal** — a recorded mid-window session
//!    replays losslessly and rebuilds identical inventory + a known working-order
//!    set (no silent orphans on restart).
//!
//! Faults, mapped to the six classes in the task: kill each WebSocket
//! individually (market-WS, user-WS), stall each feed beyond the staleness
//! threshold (fast binance, chainlink, book), a matching-engine restart notice
//! (via a fault-injecting [`venue_api::VenuePort`] decorator → the guard's
//! observation path), a clock jump (the real [`timeutil::SkewMonitor`]), a
//! discovery failure at rollover (the real [`scheduler::run`] with a starving
//! refresher), and a process restart mid-window (record → replay → rebuild).
//! The per-window-loss / daily-stop / manual-kill breakers live in
//! `risk_manager_paper.rs`; the engine-restart / error-rate / clock-skew breaker
//! *logic* is exhaustively unit-tested in `engine::risk::core`.
//!
//! The transport-death → `FeedHealth`/`BookHealth` event link is proven by the
//! feed crates' own paused-time driver tests (feed-rtds/feed-binance/feed-clob);
//! here we inject the manifested events and prove the **event → system
//! invariant** half.

#![cfg(feature = "chaos")]
// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookHealth, BookLevel, BookSnapshot, BookUnreliableReason, BreakerKind,
    ConditionId, Decimal, Dollars, DurationMs, Event, FeeParams, Fill, InputAges, Liquidity,
    MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot, NewOrder, OrderId, OrderState,
    OrderUpdate, Outcome, Price, PriceSource, ResolutionSource, RiskEvent, RoundDir, Series, Side,
    Size, TickKind, TickSize, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId,
    WindowLifecycle,
};
use engine::{InventoryManager, QuoteManagerParams, RiskManager, RiskParams};
use journal::{JournalIndexReader, Recorder, RecorderParams, ReplayReader};
use rust_decimal::dec;
use scheduler::{Refresh, SchedulerArgs, SchedulerStatus, Timing};
use timeutil::{OffsetMeasurement, SkewMonitor, SkewOutput, SkewParams};
use tokio::sync::mpsc::Receiver;
use tokio::sync::{mpsc, watch};
use venue_api::{
    Accepted, BatchPlaced, CancelReport, VenueError, VenueEvent, VenueEvents, VenuePort, Wallet,
};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;
const CLOSE_MS: i64 = BASE_MS + 300_000;

// ---- builders (mirroring risk_manager_paper.rs) ---------------------------

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
        event_slug: "btc-updown-5m-chaos".to_owned(),
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
    Event::PriceTick(core_types::PriceTick {
        source: PriceSource::BinanceDirect,
        asset: Asset::Btc,
        kind: TickKind::Mid,
        value: dec!(60000),
        ts_exchange: TimestampMs::from_millis(ts),
        ts_local: TimestampMs::from_millis(ts),
    })
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

/// A quoter-only risk manager (takers disabled) with generous daily-stop / cap
/// so only the injected fault trips a breaker.
fn chaos_risk() -> RiskManager {
    let rp = RiskParams {
        quote_manager: QuoteManagerParams {
            min_requote_interval_ms: 10,
            ..QuoteManagerParams::default()
        },
        daily_stop_loss: Dollars::new(dec!(100_000)),
        momentum_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(dec!(1_000_000)));
    RiskManager::new(rp, caps)
}

// ---- a fault-injecting venue port -----------------------------------------

/// Wraps the paper venue and, while armed, fails **placements** with the venue
/// error of choice (a matching-engine restart). Cancels and reads always
/// delegate so the authoritative cancel-all still evacuates the book — failing a
/// cancel would manufacture the orphan we are testing against.
struct FaultyPort<'a> {
    inner: &'a PaperVenue,
    error: VenueError,
    fail_places: AtomicBool,
}
impl<'a> FaultyPort<'a> {
    fn new(inner: &'a PaperVenue, error: VenueError) -> Self {
        Self {
            inner,
            error,
            fail_places: AtomicBool::new(false),
        }
    }
    fn arm(&self) {
        self.fail_places.store(true, Ordering::SeqCst);
    }
    fn disarm(&self) {
        self.fail_places.store(false, Ordering::SeqCst);
    }
}
impl VenuePort for FaultyPort<'_> {
    async fn place(&self, order: &NewOrder) -> Result<Accepted, VenueError> {
        if self.fail_places.load(Ordering::SeqCst) {
            return Err(self.error.clone());
        }
        self.inner.place(order).await
    }
    async fn place_batch(&self, orders: &[NewOrder]) -> Result<BatchPlaced, VenueError> {
        if self.fail_places.load(Ordering::SeqCst) {
            return Err(self.error.clone());
        }
        self.inner.place_batch(orders).await
    }
    async fn cancel(&self, id: &OrderId) -> Result<CancelReport, VenueError> {
        self.inner.cancel(id).await
    }
    async fn cancel_market(&self, market: &ConditionId) -> Result<CancelReport, VenueError> {
        self.inner.cancel_market(market).await
    }
    async fn cancel_all(&self) -> Result<CancelReport, VenueError> {
        self.inner.cancel_all().await
    }
    async fn balances(&self) -> Result<Wallet, VenueError> {
        self.inner.balances().await
    }
}

// ---- harness --------------------------------------------------------------

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
    (venue, rx, chaos_risk(), start)
}

/// Opens a unique temp journal (gzip segments + sqlite index) for one scenario.
fn journal(name: &str) -> (Recorder, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("chaos-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sqlite = dir.join("index.sqlite");
    let rp = RecorderParams {
        out_dir: dir.clone(),
        sqlite_path: Some(sqlite.clone()),
        ..RecorderParams::default()
    };
    let rec = Recorder::spawn(rp, || TimestampMs::from_millis(BASE_MS)).expect("spawn recorder");
    (rec, dir, sqlite)
}

/// Feeds one bus event: records it, applies it to the venue and the risk gate
/// through `port` (the venue itself, or a [`FaultyPort`] over it).
async fn feed<P: VenuePort, F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    port: &P,
    risk: &mut RiskManager,
    rec: &Recorder,
    ev: &Event,
    now: &F,
) {
    rec.record(ev);
    venue.on_bus_event(ev).await;
    risk.on_event(ev, port, now()).await;
}

/// `feed` with the port being the venue itself (the common case).
async fn feed_v<F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    risk: &mut RiskManager,
    rec: &Recorder,
    ev: &Event,
    now: &F,
) {
    feed(venue, venue, risk, rec, ev, now).await;
}

async fn tick<P: VenuePort, F: Fn() -> TimestampMs>(port: &P, risk: &mut RiskManager, now: &F) {
    risk.on_tick(port, now()).await;
}

async fn drain<P: VenuePort, F: Fn() -> TimestampMs>(
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    port: &P,
    now: &F,
) -> Vec<VenueEvent> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        risk.on_venue_event(&ev, port, now()).await;
        out.push(ev);
    }
    out
}

/// Drains the breaker events the risk manager wants published, **records each**
/// (the journaled cause), and returns the `RiskEvent`s for assertions.
fn take_and_record(risk: &mut RiskManager, rec: &Recorder) -> Vec<RiskEvent> {
    let evs = risk.take_published();
    for e in &evs {
        rec.record(e);
    }
    evs.into_iter()
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
async fn converge<P: VenuePort, F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    port: &P,
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    rec: &Recorder,
    now: &F,
) {
    for ev in [
        window_event(WindowLifecycle::Open),
        book(up_token(), dec!(0.45), dec!(0.55), BASE_MS),
        book(down_token(), dec!(0.45), dec!(0.55), BASE_MS),
        model(0.50, BASE_MS),
    ] {
        feed(venue, port, risk, rec, &ev, now).await;
    }
    tick(port, risk, now).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(rx, risk, port, now).await;
    let _ = take_and_record(risk, rec);
    assert_eq!(resting_len(risk), 6, "converged to a six-level ladder");
}

/// After a global trip: let the cancel round-trip elapse, drain, and assert no
/// orphan resting orders remain (the §11 evacuation).
async fn evacuate<P: VenuePort, F: Fn() -> TimestampMs>(
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    port: &P,
    now: &F,
    expect_cancels: usize,
) {
    tokio::time::sleep(Duration::from_millis(200)).await;
    let recorded = drain(rx, risk, port, now).await;
    assert_eq!(
        count_state(&recorded, OrderState::Canceled),
        expect_cancels,
        "the trip cancels every resting order"
    );
    assert_eq!(
        resting_len(risk),
        0,
        "no orphan resting orders after the trip"
    );
}

/// After a breaker clears: re-quote and assert the ladder comes back (trading
/// resumes exactly per the rules).
async fn resume<P: VenuePort, F: Fn() -> TimestampMs>(
    venue: &PaperVenue,
    port: &P,
    rx: &mut Receiver<VenueEvent>,
    risk: &mut RiskManager,
    rec: &Recorder,
    now: &F,
    model_ts: i64,
) {
    feed(venue, port, risk, rec, &model(0.50, model_ts), now).await;
    tick(port, risk, now).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(rx, risk, port, now).await;
    let _ = take_and_record(risk, rec);
    assert_eq!(resting_len(risk), 6, "quoting resumes after recovery");
}

/// Finishes the recorder and asserts the breaker cause is journaled — both in
/// the queryable sqlite index and in the gzip replay stream.
fn assert_journaled_trip(dir: &Path, sqlite: &Path, breaker: BreakerKind) {
    let reader = JournalIndexReader::open(sqlite).expect("open index");
    let trips = reader.breaker_trips().expect("breaker_trips");
    assert!(
        trips
            .iter()
            .any(|r| r.kind == "tripped" && r.breaker == breaker),
        "breaker {breaker:?} trip indexed in sqlite; got {trips:?}"
    );
    let events: Vec<Event> = ReplayReader::open(dir)
        .expect("open replay")
        .events()
        .map(|r| r.expect("event"))
        .collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            Event::Risk(RiskEvent::BreakerTripped { breaker: b }) if *b == breaker
        )),
        "breaker {breaker:?} trip present in the replay stream"
    );
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

// ===========================================================================
// (1) Kill the market-channel WebSocket  →  WsDisconnect
// ===========================================================================

#[tokio::test(start_paused = true)]
async fn kill_market_ws_evacuates_halts_and_recovers() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("market-ws");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &venue, &mut rx, &mut risk, &rec, &now).await;

    // The market WebSocket drops: feed-clob reports BookHealth Disconnected.
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::BookHealth(BookHealth::Unreliable {
            window: window(),
            reason: BookUnreliableReason::Disconnected,
            ts: now(),
        }),
        &now,
    )
    .await;
    let breakers = take_and_record(&mut risk, &rec);
    assert!(tripped(&breakers, BreakerKind::WsDisconnect));
    assert!(risk.is_halted(), "globally halted on the market-WS drop");
    evacuate(&mut rx, &mut risk, &venue, &now, 6).await;

    // Halted: a fresh model + tick places nothing.
    feed_v(&venue, &mut risk, &rec, &model(0.50, BASE_MS + 1000), &now).await;
    tick(&venue, &mut risk, &now).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _ = drain(&mut rx, &mut risk, &venue, &now).await;
    let _ = take_and_record(&mut risk, &rec);
    assert_eq!(resting_len(&risk), 0, "no quoting while halted");

    // The WebSocket reconnects: recovery clears the breaker and quoting resumes.
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::BookHealth(BookHealth::Recovered {
            window: window(),
            outage: DurationMs::from_millis(1500),
            ts: now(),
        }),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "cleared on reconnect");
    resume(
        &venue,
        &venue,
        &mut rx,
        &mut risk,
        &rec,
        &now,
        BASE_MS + 2000,
    )
    .await;

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0, "no journal records dropped");
    assert_journaled_trip(&dir, &sqlite, BreakerKind::WsDisconnect);
    cleanup(&dir);
}

// ===========================================================================
// (2) Kill the user-channel WebSocket  →  WsDisconnect
// ===========================================================================

#[tokio::test(start_paused = true)]
async fn kill_user_ws_evacuates_halts_and_recovers() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("user-ws");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &venue, &mut rx, &mut risk, &rec, &now).await;

    // The authenticated user WebSocket drops (a live-adapter signal on the venue
    // event stream; paper never emits it naturally — we script it).
    risk.on_venue_event(
        &VenueEvent::Connectivity { connected: false },
        &venue,
        now(),
    )
    .await;
    let breakers = take_and_record(&mut risk, &rec);
    assert!(tripped(&breakers, BreakerKind::WsDisconnect));
    assert!(risk.is_halted());
    evacuate(&mut rx, &mut risk, &venue, &now, 6).await;

    // Reconnect clears it and quoting resumes.
    risk.on_venue_event(&VenueEvent::Connectivity { connected: true }, &venue, now())
        .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "cleared on reconnect");
    resume(
        &venue,
        &venue,
        &mut rx,
        &mut risk,
        &rec,
        &now,
        BASE_MS + 2000,
    )
    .await;

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0);
    assert_journaled_trip(&dir, &sqlite, BreakerKind::WsDisconnect);
    cleanup(&dir);
}

// ===========================================================================
// (3) Stall the fast signal feed (>500 ms)  →  FeedStale
// ===========================================================================

#[tokio::test(start_paused = true)]
async fn stall_fast_feed_evacuates_halts_and_recovers() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("fast-feed");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &venue, &mut rx, &mut risk, &rec, &now).await;
    // Establish a live BinanceDirect/Mid stream.
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &binance_tick(now().as_millis()),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);

    // No fast-feed tick for >500 ms, then a risk tick: FeedStale trips.
    tokio::time::sleep(Duration::from_millis(700)).await;
    tick(&venue, &mut risk, &now).await;
    let breakers = take_and_record(&mut risk, &rec);
    assert!(tripped(&breakers, BreakerKind::FeedStale));
    assert!(risk.is_halted());
    evacuate(&mut rx, &mut risk, &venue, &now, 6).await;

    // A fresh fast-feed tick clears it; quoting resumes.
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &binance_tick(now().as_millis()),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "cleared by a fresh fast-feed tick");
    resume(
        &venue,
        &venue,
        &mut rx,
        &mut risk,
        &rec,
        &now,
        BASE_MS + 2000,
    )
    .await;

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0);
    assert_journaled_trip(&dir, &sqlite, BreakerKind::FeedStale);
    cleanup(&dir);
}

// ===========================================================================
// (4) Stall the Chainlink (resolution) feed  →  FeedStale
// ===========================================================================

#[tokio::test(start_paused = true)]
async fn stall_chainlink_feed_evacuates_halts_and_recovers() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("chainlink-feed");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &venue, &mut rx, &mut risk, &rec, &now).await;

    // feed-rtds reports the chainlink stream stale (its own ~1 Hz watchdog).
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::FeedHealth(core_types::FeedHealth::Stale {
            source: PriceSource::ChainlinkRtds,
            asset: Asset::Btc,
            kind: TickKind::Vendor,
            age: DurationMs::from_millis(6000),
        }),
        &now,
    )
    .await;
    let breakers = take_and_record(&mut risk, &rec);
    assert!(tripped(&breakers, BreakerKind::FeedStale));
    assert!(risk.is_halted());
    evacuate(&mut rx, &mut risk, &venue, &now, 6).await;

    // Recovery clears it; quoting resumes.
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::FeedHealth(core_types::FeedHealth::Recovered {
            source: PriceSource::ChainlinkRtds,
            asset: Asset::Btc,
            kind: TickKind::Vendor,
            gap: DurationMs::from_millis(6500),
        }),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "cleared on recovery");
    resume(
        &venue,
        &venue,
        &mut rx,
        &mut risk,
        &rec,
        &now,
        BASE_MS + 2000,
    )
    .await;

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0);
    assert_journaled_trip(&dir, &sqlite, BreakerKind::FeedStale);
    cleanup(&dir);
}

// ===========================================================================
// (5) Stall the order book (non-disconnect)  →  FeedStale
// ===========================================================================

#[tokio::test(start_paused = true)]
async fn stall_book_feed_evacuates_halts_and_recovers() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("book-feed");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &venue, &mut rx, &mut risk, &rec, &now).await;

    // feed-clob reports the book stale (silent, not disconnected) → FeedStale,
    // not WsDisconnect.
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::BookHealth(BookHealth::Unreliable {
            window: window(),
            reason: BookUnreliableReason::Stale,
            ts: now(),
        }),
        &now,
    )
    .await;
    let breakers = take_and_record(&mut risk, &rec);
    assert!(tripped(&breakers, BreakerKind::FeedStale));
    assert!(risk.is_halted());
    evacuate(&mut rx, &mut risk, &venue, &now, 6).await;

    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::BookHealth(BookHealth::Recovered {
            window: window(),
            outage: DurationMs::from_millis(11_000),
            ts: now(),
        }),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "cleared on recovery");
    resume(
        &venue,
        &venue,
        &mut rx,
        &mut risk,
        &rec,
        &now,
        BASE_MS + 2000,
    )
    .await;

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0);
    assert_journaled_trip(&dir, &sqlite, BreakerKind::FeedStale);
    cleanup(&dir);
}

// ===========================================================================
// (6) Matching-engine restart notice  →  EngineRestart
// ===========================================================================

#[tokio::test(start_paused = true)]
async fn engine_restart_notice_evacuates_halts_and_clears_after_cooldown() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("engine-restart");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    let faulty = FaultyPort::new(&venue, VenueError::EngineRestarting);
    converge(&venue, &faulty, &mut rx, &mut risk, &rec, &now).await;

    // The matching engine restarts: the next placement attempt 425s. We force a
    // reprice (a model move beyond the cancel/reprice thresholds, but inside the
    // |fair − mid| sanity bound so only EngineRestart trips), then pump requote
    // ticks; the guard observes the EngineRestarting error and the next turn
    // folds it into the breaker (observations are drained at the top of each
    // turn, so the trip lands a turn after the failed place).
    faulty.arm();
    feed(
        &venue,
        &faulty,
        &mut risk,
        &rec,
        &model(0.55, BASE_MS + 1000),
        &now,
    )
    .await;
    let mut cancels: Vec<VenueEvent> = Vec::new();
    let mut tripped_engine = false;
    for _ in 0..12 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        tick(&faulty, &mut risk, &now).await;
        let breakers = take_and_record(&mut risk, &rec);
        if tripped(&breakers, BreakerKind::EngineRestart) {
            tripped_engine = true;
        }
        cancels.extend(drain(&mut rx, &mut risk, &faulty, &now).await);
        if tripped_engine {
            break;
        }
    }
    assert!(
        tripped_engine,
        "EngineRestart trips after a placement fails with the restart error"
    );
    assert!(risk.is_halted());

    // Every resting order is evacuated (by the reprice's cancel and/or the
    // authoritative cancel-all the trip issues — both delegate through the
    // faulty port to the real venue). No orphans either way.
    tokio::time::sleep(Duration::from_millis(200)).await;
    cancels.extend(drain(&mut rx, &mut risk, &faulty, &now).await);
    assert_eq!(
        count_state(&cancels, OrderState::Canceled),
        6,
        "all six resting orders evacuated"
    );
    assert_eq!(
        resting_len(&risk),
        0,
        "no orphan resting orders after the trip"
    );

    // Past the cooldown the breaker clears; with the engine back (port
    // disarmed), the still-dirty quoter converges to a full ladder again.
    faulty.disarm();
    tokio::time::sleep(Duration::from_millis(6000)).await;
    tick(&faulty, &mut risk, &now).await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "EngineRestart clears past the cooldown");
    let mut resumed = false;
    for _ in 0..6 {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let _ = drain(&mut rx, &mut risk, &faulty, &now).await;
        let _ = take_and_record(&mut risk, &rec);
        if resting_len(&risk) == 6 {
            resumed = true;
            break;
        }
        tick(&faulty, &mut risk, &now).await;
    }
    assert!(
        resumed,
        "quoting resumes to a full ladder after the engine is back"
    );

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0);
    assert_journaled_trip(&dir, &sqlite, BreakerKind::EngineRestart);
    cleanup(&dir);
}

// ===========================================================================
// (7) Jump the mock clock  →  ClockSkew (via the real SkewMonitor)
// ===========================================================================

fn skew_params() -> SkewParams {
    SkewParams {
        trip_bound: DurationMs::from_millis(250),
        clear_bound: DurationMs::from_millis(125),
        trip_after: 2,
        clear_after: 3,
        stale_warn: DurationMs::from_millis(300_000),
    }
}
fn offset(ms: i64) -> OffsetMeasurement {
    OffsetMeasurement {
        offset: Some(DurationMs::from_millis(ms)),
        samples_used: 4,
        queries_failed: 0,
        min_round_trip: Some(DurationMs::from_millis(10)),
    }
}

#[tokio::test(start_paused = true)]
async fn clock_jump_evacuates_halts_and_recovers() {
    let (venue, mut rx, mut risk, start) = make();
    let (rec, dir, sqlite) = journal("clock-jump");
    let now = || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    converge(&venue, &venue, &mut rx, &mut risk, &rec, &now).await;

    // Drive the REAL skew monitor with offsets that simulate the wall clock
    // jumping ~300 ms ahead for two consecutive rounds → a confirmed trip.
    let mut mon = SkewMonitor::new(skew_params());
    let mut sout = Vec::new();
    mon.on_measurement(&offset(300), TimestampMs::from_millis(1_000), &mut sout);
    mon.on_measurement(&offset(310), TimestampMs::from_millis(2_000), &mut sout);
    assert!(
        sout.iter().any(|o| matches!(o, SkewOutput::Trip { .. })),
        "the skew monitor trips on a sustained clock jump"
    );
    assert!(mon.is_tripped());

    // The monitor's driver puts this on the bus; the risk core honors it
    // (forwarded + cancel-all). We record the input event as the journaled
    // cause (ClockSkew is forwarded, not re-published).
    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::Risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::ClockSkew,
        }),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(risk.is_halted(), "globally halted on the clock-skew trip");
    evacuate(&mut rx, &mut risk, &venue, &now, 6).await;

    // The operator resyncs the clock: three in-bound rounds → a confirmed clear.
    let mut sout = Vec::new();
    mon.on_measurement(&offset(10), TimestampMs::from_millis(3_000), &mut sout);
    mon.on_measurement(&offset(10), TimestampMs::from_millis(4_000), &mut sout);
    mon.on_measurement(&offset(10), TimestampMs::from_millis(5_000), &mut sout);
    assert!(
        sout.iter().any(|o| matches!(o, SkewOutput::Clear { .. })),
        "the skew monitor clears after the clock resyncs"
    );
    assert!(!mon.is_tripped());

    feed_v(
        &venue,
        &mut risk,
        &rec,
        &Event::Risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::ClockSkew,
        }),
        &now,
    )
    .await;
    let _ = take_and_record(&mut risk, &rec);
    assert!(!risk.is_halted(), "cleared on resync");
    resume(
        &venue,
        &venue,
        &mut rx,
        &mut risk,
        &rec,
        &now,
        BASE_MS + 2000,
    )
    .await;

    venue.shutdown();
    let stats = rec.finish().expect("finish");
    assert_eq!(stats.dropped, 0);
    assert_journaled_trip(&dir, &sqlite, BreakerKind::ClockSkew);
    cleanup(&dir);
}

// ===========================================================================
// (8) Force discovery failures at rollover  →  the real scheduler
// ===========================================================================

const SCHED_BASE_MS: i64 = 1_800_000_060_000; // 1 min into a 5m boundary.

fn sched_timing() -> Timing {
    Timing {
        refresh_lead: DurationMs::from_millis(120_000),
        next_window_lead: DurationMs::from_millis(60_000),
        closing_lead: DurationMs::from_millis(30_000),
        retry_initial: DurationMs::from_millis(1_000),
        retry_max: DurationMs::from_millis(30_000),
        resolution_timeout: DurationMs::from_millis(120_000),
        max_refresh_interval: DurationMs::from_millis(600_000),
        expect_resolutions: false,
    }
}

fn series_market(series: Series, open_ms: i64) -> Arc<MarketInfo> {
    let dur = series.duration.as_duration().as_millis();
    let salt: u64 = if series.asset == Asset::Eth {
        1 << 40
    } else {
        0
    };
    let n = (open_ms as u64).wrapping_add(salt);
    Arc::new(MarketInfo {
        window: WindowId {
            series,
            open_time: TimestampMs::from_millis(open_ms),
        },
        event_slug: format!(
            "chaos-{}-{open_ms}",
            if series.asset == Asset::Eth {
                "eth"
            } else {
                "btc"
            }
        ),
        condition_id: ConditionId::new(format!("0x{n:064x}")).expect("cid"),
        tokens: TokenPair {
            up: TokenId::new(format!("{}", n.wrapping_mul(2).wrapping_add(1))).expect("up"),
            down: TokenId::new(format!("{}", n.wrapping_mul(2).wrapping_add(2))).expect("down"),
        },
        close_time: TimestampMs::from_millis(open_ms + dur),
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

fn aligned_windows(series: Series, now: TimestampMs) -> discovery::SeriesWindows {
    let dur = series.duration.as_duration().as_millis();
    let open = now.as_millis() - now.as_millis().rem_euclid(dur);
    discovery::SeriesWindows {
        series,
        current: Some(series_market(series, open)),
        upcoming: vec![
            series_market(series, open + dur),
            series_market(series, open + 2 * dur),
        ],
        fetched_at: now,
    }
}

/// A refresher that serves perfectly aligned windows, except it fails the
/// `target` series with a discovery error while `starve` is armed.
struct ChaosRefresh {
    target: Series,
    starve: Arc<AtomicBool>,
}
impl Refresh for ChaosRefresh {
    async fn refresh(
        &mut self,
        series: Series,
        now: TimestampMs,
    ) -> Result<discovery::SeriesWindows, discovery::DiscoveryError> {
        if series == self.target && self.starve.load(Ordering::SeqCst) {
            return Err(discovery::DiscoveryError::NoWindows { series });
        }
        Ok(aligned_windows(series, now))
    }
}

fn paused_now_fn(base: i64) -> impl Fn() -> TimestampMs + Send {
    let start = tokio::time::Instant::now();
    move || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
        TimestampMs::from_millis(base + elapsed)
    }
}

/// Pumps the bus (counting per-series window opens) until `pred` holds over the
/// accumulated opens and the latest status, or the deadline passes.
async fn run_until(
    bus_rx: &mut Receiver<Event>,
    status_rx: &watch::Receiver<SchedulerStatus>,
    opens: &mut HashMap<Series, usize>,
    deadline: tokio::time::Instant,
    mut pred: impl FnMut(&HashMap<Series, usize>, &SchedulerStatus) -> bool,
) -> bool {
    loop {
        {
            let st = status_rx.borrow();
            if pred(opens, &st) {
                return true;
            }
        }
        tokio::select! {
            ev = bus_rx.recv() => match ev {
                Some(Event::Window { market, lifecycle: WindowLifecycle::Open }) => {
                    *opens.entry(market.window.series).or_insert(0) += 1;
                }
                Some(_) => {}
                None => return false,
            },
            () = tokio::time::sleep_until(deadline) => {
                let st = status_rx.borrow();
                return pred(opens, &st);
            }
        }
    }
}

#[tokio::test(start_paused = true)]
async fn discovery_failure_at_rollover_isolates_the_series_and_recovers() {
    let target = Series {
        asset: Asset::Btc,
        duration: WindowDuration::M5,
    };
    let healthy = Series {
        asset: Asset::Eth,
        duration: WindowDuration::M5,
    };
    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (status_tx, status_rx) = watch::channel(SchedulerStatus::default());
    let starve = Arc::new(AtomicBool::new(false));
    let refresher = ChaosRefresh {
        target,
        starve: Arc::clone(&starve),
    };
    let handle = tokio::spawn(scheduler::run(SchedulerArgs {
        timing: sched_timing(),
        series: vec![target, healthy],
        refresher,
        now_fn: paused_now_fn(SCHED_BASE_MS),
        bus_tx,
        market_rx: None,
        status_tx: Some(status_tx),
        shutdown_rx,
    }));

    let mut opens: HashMap<Series, usize> = HashMap::new();

    // Phase 1 — both series boot and open their first window.
    let dl = tokio::time::Instant::now() + Duration::from_secs(700);
    let ok = run_until(&mut bus_rx, &status_rx, &mut opens, dl, |opens, _st| {
        opens.get(&target).copied().unwrap_or(0) >= 1
            && opens.get(&healthy).copied().unwrap_or(0) >= 1
    })
    .await;
    assert!(ok, "both series opened their first window");
    let healthy_at_arm = opens[&healthy];

    // Inject the fault: the target's discovery refreshes now fail.
    starve.store(true, Ordering::SeqCst);

    // Phase 2 — the target exhausts its lookahead and gaps (next window
    // unknown), while the healthy series keeps rolling (per-series isolation).
    let dl = tokio::time::Instant::now() + Duration::from_secs(2_400);
    let gapped = run_until(&mut bus_rx, &status_rx, &mut opens, dl, |opens, st| {
        let target_gapped = st
            .series
            .iter()
            .any(|s| s.series == target && !s.next_known);
        let healthy_progressed = opens.get(&healthy).copied().unwrap_or(0) > healthy_at_arm;
        target_gapped && healthy_progressed
    })
    .await;
    assert!(
        gapped,
        "the starved series loses its next window while the healthy series keeps rolling"
    );
    {
        let st = status_rx.borrow();
        let h = st
            .series
            .iter()
            .find(|s| s.series == healthy)
            .expect("healthy status");
        assert!(
            h.next_known && h.contract_ok,
            "the healthy series is unaffected by the target's discovery failure"
        );
    }
    let target_at_recover = opens.get(&target).copied().unwrap_or(0);

    // Recovery: discovery returns; the target rolls again.
    starve.store(false, Ordering::SeqCst);
    let dl = tokio::time::Instant::now() + Duration::from_secs(2_400);
    let recovered = run_until(&mut bus_rx, &status_rx, &mut opens, dl, |opens, st| {
        let target_known = st.series.iter().any(|s| s.series == target && s.next_known);
        let target_opened = opens.get(&target).copied().unwrap_or(0) > target_at_recover;
        target_known && target_opened
    })
    .await;
    assert!(
        recovered,
        "the starved series recovers and rolls again once discovery returns"
    );

    // The driver never stalled or crashed: it shuts down cleanly.
    shutdown_tx.send(true).expect("driver alive");
    let res = tokio::time::timeout(Duration::from_secs(60), handle)
        .await
        .expect("driver exits promptly")
        .expect("driver task not panicked");
    assert!(matches!(res, Ok(())));
}

// ===========================================================================
// (9) Restart the process mid-window  →  state rebuilds from the journal
// ===========================================================================

fn order_update_event(id: &str, state: OrderState) -> Event {
    Event::OrderUpdate(Arc::new(OrderUpdate {
        order_id: OrderId::new(id).expect("oid"),
        window: window(),
        token_id: up_token(),
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
        ts_local: TimestampMs::from_millis(BASE_MS),
    }))
}
fn fill_event(outcome: Outcome, side: Side, price: Decimal, size: Decimal) -> Event {
    Event::Fill(Arc::new(Fill {
        order_id: OrderId::new("filled").expect("oid"),
        trade_id: Some("t1".to_owned()),
        window: window(),
        token_id: up_token(),
        outcome,
        side,
        price: px(price),
        size: sz(size),
        liquidity: Liquidity::Maker,
        fee: Dollars::ZERO,
        ts_venue: TimestampMs::from_millis(BASE_MS + 100),
        ts_local: TimestampMs::from_millis(BASE_MS + 100),
    }))
}

/// A mid-window session: window open, an order driven terminal (Filled), a
/// still-working order, two fills building inventory — and NO resolution (the
/// process dies mid-window).
fn mid_window_session() -> Vec<Event> {
    vec![
        window_event(WindowLifecycle::Open),
        order_update_event("filled", OrderState::Open),
        order_update_event("filled", OrderState::Filled),
        order_update_event("working", OrderState::Open),
        fill_event(Outcome::Up, Side::Buy, dec!(0.40), dec!(100)),
        fill_event(Outcome::Down, Side::Buy, dec!(0.55), dec!(80)),
    ]
}

/// The bot's boot-time order view (mirrors `bot::boot::RebuiltOrders`): last
/// update per id, terminal updates drop the order.
fn order_view(events: &[Event]) -> HashMap<OrderId, OrderUpdate> {
    let mut m: HashMap<OrderId, OrderUpdate> = HashMap::new();
    for e in events {
        if let Event::OrderUpdate(u) = e {
            if u.state.is_terminal() {
                m.remove(&u.order_id);
            } else {
                m.insert(u.order_id.clone(), (**u).clone());
            }
        }
    }
    m
}

#[test]
fn process_restart_mid_window_rebuilds_inventory_and_known_orders() {
    let dir = std::env::temp_dir().join(format!("chaos-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sqlite = dir.join("index.sqlite");
    let events = mid_window_session();

    // The live state the engine held when the process died.
    let mut live = InventoryManager::new();
    for e in &events {
        let _ = live.on_event(e);
    }
    let live_orders = order_view(&events);

    // Record the session, then `finish()` = the process dying.
    let rp = RecorderParams {
        out_dir: dir.clone(),
        sqlite_path: Some(sqlite.clone()),
        ..RecorderParams::default()
    };
    let rec = Recorder::spawn(rp, || TimestampMs::from_millis(BASE_MS)).expect("spawn recorder");
    for e in &events {
        rec.record(e);
    }
    let stats = rec.finish().expect("finish recorder");
    assert_eq!(stats.records, events.len() as u64);
    assert_eq!(stats.dropped, 0);

    // Restart: replay the journal and rebuild derived state.
    let replayed: Vec<Event> = ReplayReader::open(&dir)
        .expect("open replay")
        .events()
        .map(|r| r.expect("replayed event"))
        .collect();
    assert_eq!(replayed, events, "the journal round-trip must be lossless");

    let rebuilt = InventoryManager::rebuild(replayed.iter());
    assert_eq!(rebuilt, live, "inventory rebuilds exactly from the journal");

    // No silent orphans on restart: the still-working order is known (so the
    // orchestrator's cancel-all / reconcile will evacuate it), the filled one is
    // dropped.
    let rebuilt_orders = order_view(&replayed);
    assert_eq!(rebuilt_orders, live_orders);
    assert!(
        rebuilt_orders.contains_key(&OrderId::new("working").expect("oid")),
        "the open order is known on restart"
    );
    assert!(
        !rebuilt_orders.contains_key(&OrderId::new("filled").expect("oid")),
        "a terminal order is dropped from the rebuilt view"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
