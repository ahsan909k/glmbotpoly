//! Venue-parity golden tests (CLAUDE.md §2.5/§9): prove the engine cannot tell
//! the paper and live venues apart at the `VenuePort` / `VenueEvent` surface.
//!
//! The strategy/engine sees a venue ONLY through `venue_api`: the `place` /
//! `cancel*` return values and the `VenueEvent::{Order, Fill}` stream. This
//! suite drives scripted order sequences through [`venue_paper::PaperVenue`] over
//! deterministic market data, **normalizes** each emitted `VenueEvent` into the
//! engine-visible projection (state / side / price / size / liquidity / reject
//! class — venue-specific ids, timestamps, and exact fees dropped), and asserts
//! the sequence equals a **golden** expectation that encodes the same contract
//! `venue-live`'s in-crate `FakeClobPort` tests already pin on the live side.
//! Each scenario names the live-side test(s) that prove the identical behavior.
//!
//! Why paper-only (not a live/paper side-by-side diff): `LiveVenue` is only
//! publicly constructible through the gated `connect` (arming + network), by
//! design (§11) — its contract is already proven offline by its own unit tests.
//! Both adapters emit the same `venue_api` types against the same
//! `core_types::OrderState` transition table, so "indistinguishable" is
//! established by: *paper's normalized stream == golden* and *golden == the
//! behavior the cross-referenced live tests assert*.
//!
//! Contract cross-reference (live test → covered here):
//! - `store::order_lifecycle_placement_partial_full` → A, the resting progression
//! - `store::trade_maker_fill_zero_fee_and_taker_fill_charged` → A (maker fee 0), C1/D (taker charged)
//! - `venue::rejected_ack_maps_to_typed_reject` → B/C2/D2 (rejections surface as engine-visible terminals)
//! - `convert::qty_kind_mismatches_are_rejected` → H (synchronous structural reject)
//! - `store::order_cancellation_is_terminal` → E/F/I (Canceled terminal)
//! - `venue::cancel_all_and_market_route_to_backend` → F (cancel_all report + terminals)
//! - `venue::cancel_reports_partial_failures` → G (AlreadyGone)

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, Event, FeeParams, Liquidity,
    MarketInfo, NewOrder, OrderId, OrderQty, OrderState, Outcome, Price, ResolutionSource,
    RoundDir, Series, Side, Size, TickSize, TimeInForce, TimestampMs, TokenId, TokenPair,
    WindowDuration, WindowId, WindowLifecycle,
};
use journal::{Recorder, RecorderParams, ReplayReader};
use rust_decimal::dec;
use venue_api::{Accepted, VenueError, VenueEvent, VenueEvents, VenuePort};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;
const PLACE_MS: i64 = 100;
const CANCEL_MS: i64 = 200;

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
fn market() -> Arc<MarketInfo> {
    Arc::new(MarketInfo {
        window: window(),
        event_slug: "btc-updown-5m-parity".to_owned(),
        condition_id: condition(),
        tokens: TokenPair {
            up: up_token(),
            down: down_token(),
        },
        close_time: TimestampMs::from_millis(BASE_MS + 300_000),
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

/// A book snapshot for the Up token.
fn book(bids: &[(Decimal, Decimal)], asks: &[(Decimal, Decimal)], ts: i64) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: up_token(),
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

/// A trade print on the Up token (`side` = aggressor).
fn trade(price: Decimal, size: Decimal, side: Side, ts: i64) -> Event {
    Event::LastTrade {
        token_id: Arc::new(up_token()),
        price: px(price),
        size: sz(size),
        side,
        ts: TimestampMs::from_millis(ts),
    }
}

fn window_event(lifecycle: WindowLifecycle) -> Event {
    Event::Window {
        market: market(),
        lifecycle,
    }
}

fn resting_buy(client: &str, price: Decimal, shares: Decimal) -> NewOrder {
    NewOrder {
        client_id: Some(client.to_owned()),
        window: window(),
        token_id: up_token(),
        outcome: Outcome::Up,
        side: Side::Buy,
        price: px(price),
        qty: OrderQty::Shares(sz(shares)),
        tif: TimeInForce::Gtc { post_only: true },
    }
}

fn gtd_buy(client: &str, price: Decimal, shares: Decimal, expires_at: i64) -> NewOrder {
    NewOrder {
        client_id: Some(client.to_owned()),
        window: window(),
        token_id: up_token(),
        outcome: Outcome::Up,
        side: Side::Buy,
        price: px(price),
        qty: OrderQty::Shares(sz(shares)),
        tif: TimeInForce::Gtd {
            expires_at: TimestampMs::from_millis(expires_at),
            post_only: true,
        },
    }
}

fn marketable_sell(client: &str, worst: Decimal, shares: Decimal, fok: bool) -> NewOrder {
    NewOrder {
        client_id: Some(client.to_owned()),
        window: window(),
        token_id: up_token(),
        outcome: Outcome::Up,
        side: Side::Sell,
        price: px(worst),
        qty: OrderQty::Shares(sz(shares)),
        tif: if fok {
            TimeInForce::Fok
        } else {
            TimeInForce::Fak
        },
    }
}

/// A structurally-invalid order: a GTC (resting) with a notional quantity — the
/// one case both venues reject synchronously from `place()`.
fn bad_qty_gtc(client: &str) -> NewOrder {
    NewOrder {
        client_id: Some(client.to_owned()),
        window: window(),
        token_id: up_token(),
        outcome: Outcome::Up,
        side: Side::Buy,
        price: px(dec!(0.49)),
        qty: OrderQty::Notional(Dollars::new(dec!(10))),
        tif: TimeInForce::Gtc { post_only: true },
    }
}

fn params() -> PaperParams {
    PaperParams {
        placement: LatencySpec {
            mean_ms: core_types::DurationMs::from_millis(PLACE_MS),
            jitter_ms: core_types::DurationMs::ZERO,
        },
        cancel: LatencySpec {
            mean_ms: core_types::DurationMs::from_millis(CANCEL_MS),
            jitter_ms: core_types::DurationMs::ZERO,
        },
        rng_seed: Some(1),
        ..PaperParams::default()
    }
}

/// Builds a paper venue bound to the paused virtual clock, plus its event rx.
fn make_venue() -> (PaperVenue, tokio::sync::mpsc::Receiver<VenueEvent>) {
    let start = tokio::time::Instant::now();
    let now_fn = move || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(0);
        TimestampMs::from_millis(BASE_MS + elapsed)
    };
    let mut venue = PaperVenue::spawn(params(), now_fn);
    let rx = venue.take_event_rx().expect("event rx");
    (venue, rx)
}

// ---- engine-visible normalization -----------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Order,
    Fill,
}

/// Reject classes the engine branches on (paper's reason strings classified the
/// same way `venue_live::error::classify_reject` classifies the venue's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RejectClass {
    Cross,
    Fok,
    Fak,
    Other,
}

/// The engine-visible projection of one `VenueEvent` — everything the strategy
/// branches on, with venue-specific ids/timestamps/exact-fee dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Norm {
    kind: Kind,
    /// Stable per-scenario id = the order's `client_id`.
    alias: String,
    state: Option<OrderState>,
    side: Side,
    price: Decimal,
    /// `filled_size` for an Order, `size` for a Fill.
    size: Decimal,
    liquidity: Option<Liquidity>,
    reject: Option<RejectClass>,
    fee_zero: Option<bool>,
}

fn classify(reason: &str) -> RejectClass {
    let r = reason.to_ascii_lowercase();
    if r.contains("cross") {
        RejectClass::Cross
    } else if r.contains("fok") {
        RejectClass::Fok
    } else if r.contains("fak") {
        RejectClass::Fak
    } else {
        RejectClass::Other
    }
}

fn alias_of(id: &OrderId, aliases: &HashMap<String, String>) -> String {
    aliases
        .get(id.as_str())
        .cloned()
        .unwrap_or_else(|| id.as_str().to_owned())
}

fn normalize(ev: &VenueEvent, aliases: &HashMap<String, String>) -> Norm {
    match ev {
        VenueEvent::Order(u) => Norm {
            kind: Kind::Order,
            alias: alias_of(&u.order_id, aliases),
            state: Some(u.state),
            side: u.side,
            price: u.price.as_decimal(),
            size: u.filled_size.as_decimal(),
            liquidity: None,
            reject: u.reject_reason.as_deref().map(classify),
            fee_zero: None,
        },
        VenueEvent::Fill(f) => Norm {
            kind: Kind::Fill,
            alias: alias_of(&f.order_id, aliases),
            state: None,
            side: f.side,
            price: f.price.as_decimal(),
            size: f.size.as_decimal(),
            liquidity: Some(f.liquidity),
            reject: None,
            fee_zero: Some(f.fee == Dollars::ZERO),
        },
        // User-WS connectivity is a live-only risk-manager signal; venue-paper
        // never emits it, so it can't appear in these golden sequences.
        VenueEvent::Connectivity { .. } => {
            panic!("venue-paper never emits VenueEvent::Connectivity")
        }
    }
}

// ---- golden builders ------------------------------------------------------

fn g_order(alias: &str, state: OrderState, side: Side, price: Decimal, filled: Decimal) -> Norm {
    Norm {
        kind: Kind::Order,
        alias: alias.to_owned(),
        state: Some(state),
        side,
        price,
        size: filled,
        liquidity: None,
        reject: None,
        fee_zero: None,
    }
}

fn g_reject(alias: &str, side: Side, price: Decimal, class: RejectClass) -> Norm {
    Norm {
        kind: Kind::Order,
        alias: alias.to_owned(),
        state: Some(OrderState::Rejected),
        side,
        price,
        size: dec!(0),
        liquidity: None,
        reject: Some(class),
        fee_zero: None,
    }
}

fn g_fill(
    alias: &str,
    side: Side,
    price: Decimal,
    size: Decimal,
    liq: Liquidity,
    fee_zero: bool,
) -> Norm {
    Norm {
        kind: Kind::Fill,
        alias: alias.to_owned(),
        state: None,
        side,
        price,
        size,
        liquidity: Some(liq),
        reject: None,
        fee_zero: Some(fee_zero),
    }
}

// ---- collection under paused time -----------------------------------------

/// Receives exactly `n` venue events. Under `start_paused`, the runtime
/// auto-advances the virtual clock to the next pending engine deadline
/// (activation/cancel/expiry), so this drives all timed transitions without
/// hand-rolled `advance` calls; the long timeout only fires on a genuine bug.
async fn expect_events(
    rx: &mut tokio::sync::mpsc::Receiver<VenueEvent>,
    n: usize,
) -> Vec<VenueEvent> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        match tokio::time::timeout(Duration::from_secs(180), rx.recv()).await {
            Ok(Some(ev)) => out.push(ev),
            Ok(None) => panic!("event stream closed after {i}/{n} events"),
            Err(_) => panic!("timed out waiting for event {}/{n}; got {out:?}", i + 1),
        }
    }
    out
}

/// Asserts no further event arrives soon. The short timeout fires before any
/// far-future deadline (e.g. the daily rebate cycle, which emits no event
/// anyway), so a clean drain is unambiguous.
async fn assert_drained(rx: &mut tokio::sync::mpsc::Receiver<VenueEvent>) {
    match tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {
        Err(_) | Ok(None) => {}
        Ok(Some(ev)) => panic!("unexpected extra venue event: {ev:?}"),
    }
}

/// Places an order, recording its `order_id → client_id` alias for normalization.
async fn place(
    venue: &PaperVenue,
    aliases: &mut HashMap<String, String>,
    order: &NewOrder,
) -> Result<Accepted, VenueError> {
    let acc = venue.place(order).await;
    if let Ok(a) = &acc {
        aliases.insert(
            a.order_id.as_str().to_owned(),
            a.client_id.clone().unwrap_or_default(),
        );
    }
    acc
}

fn normalize_all(events: &[VenueEvent], aliases: &HashMap<String, String>) -> Vec<Norm> {
    events.iter().map(|e| normalize(e, aliases)).collect()
}

// ---- scenarios ------------------------------------------------------------

/// A: a post-only order rests, then a real opposite-aggressor print partially
/// fills it as maker (fee 0) at our limit.
/// Live: `store::order_lifecycle_placement_partial_full`,
/// `store::trade_maker_fill_zero_fee_and_taker_fill_charged`.
#[tokio::test(start_paused = true)]
async fn a_resting_then_partial_maker_fill() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            BASE_MS,
        ))
        .await;

    let acc = place(
        &venue,
        &mut aliases,
        &resting_buy("A", dec!(0.49), dec!(20)),
    )
    .await
    .expect("accepted");
    assert_eq!(
        acc.state,
        OrderState::PendingNew,
        "place returns PendingNew"
    );

    let open = expect_events(&mut rx, 1).await;
    venue
        .on_bus_event(&trade(dec!(0.49), dec!(40), Side::Sell, BASE_MS + 250))
        .await;
    let fill = expect_events(&mut rx, 2).await;
    assert_drained(&mut rx).await;

    let got = normalize_all(&[open, fill].concat(), &aliases);
    let golden = vec![
        g_order("A", OrderState::Open, Side::Buy, dec!(0.49), dec!(0)),
        g_fill("A", Side::Buy, dec!(0.49), dec!(10), Liquidity::Maker, true),
        g_order(
            "A",
            OrderState::PartiallyFilled,
            Side::Buy,
            dec!(0.49),
            dec!(10),
        ),
    ];
    assert_eq!(got, golden);
}

/// B: a post-only order that would cross at activation is rejected — and the
/// rejection is **asynchronous** (an `Order{Rejected}` event), NOT a `place()`
/// error. Live: `venue::rejected_ack_maps_to_typed_reject`.
#[tokio::test(start_paused = true)]
async fn b_post_only_cross_is_async_reject() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.48), dec!(10))],
            &[(dec!(0.50), dec!(10))],
            BASE_MS,
        ))
        .await;

    // A post-only BUY at the ask would cross — accepted now, rejected later.
    let acc = place(
        &venue,
        &mut aliases,
        &resting_buy("B", dec!(0.50), dec!(20)),
    )
    .await
    .expect("accepted");
    assert_eq!(acc.state, OrderState::PendingNew);

    let got = normalize_all(&expect_events(&mut rx, 1).await, &aliases);
    assert_drained(&mut rx).await;
    assert_eq!(
        got,
        vec![g_reject("B", Side::Buy, dec!(0.50), RejectClass::Cross)]
    );
}

/// C1: a FOK marketable order that fills in full walks the book, paying taker
/// fees, then settles `Filled`.
/// Live: `store::trade_maker_fill_zero_fee_and_taker_fill_charged` (taker leg).
#[tokio::test(start_paused = true)]
async fn c1_fok_full_fill_walks_two_levels() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    // Two bid levels to walk a SELL across.
    venue
        .on_bus_event(&book(
            &[(dec!(0.50), dec!(60)), (dec!(0.49), dec!(60))],
            &[(dec!(0.55), dec!(10))],
            BASE_MS,
        ))
        .await;

    place(
        &venue,
        &mut aliases,
        &marketable_sell("C1", dec!(0.49), dec!(100), true),
    )
    .await
    .expect("accepted");

    let got = normalize_all(&expect_events(&mut rx, 3).await, &aliases);
    assert_drained(&mut rx).await;
    assert_eq!(
        got,
        vec![
            g_fill(
                "C1",
                Side::Sell,
                dec!(0.50),
                dec!(60),
                Liquidity::Taker,
                false
            ),
            g_fill(
                "C1",
                Side::Sell,
                dec!(0.49),
                dec!(40),
                Liquidity::Taker,
                false
            ),
            g_order("C1", OrderState::Filled, Side::Sell, dec!(0.49), dec!(100)),
        ]
    );
}

/// C2: a FOK that cannot fill in full is rejected with no partial fills.
/// Live: `venue::rejected_ack_maps_to_typed_reject`.
#[tokio::test(start_paused = true)]
async fn c2_fok_unfillable_rejects_with_no_partial() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.50), dec!(60))],
            &[(dec!(0.55), dec!(10))],
            BASE_MS,
        ))
        .await;

    place(
        &venue,
        &mut aliases,
        &marketable_sell("C2", dec!(0.49), dec!(100), true),
    )
    .await
    .expect("accepted");

    let got = normalize_all(&expect_events(&mut rx, 1).await, &aliases);
    assert_drained(&mut rx).await;
    assert_eq!(
        got,
        vec![g_reject("C2", Side::Sell, dec!(0.49), RejectClass::Fok)]
    );
}

/// D: a FAK fills what it can, then cancels the remainder.
/// Live: `store::order_cancellation_is_terminal` (terminal Canceled).
#[tokio::test(start_paused = true)]
async fn d_fak_partial_then_cancel() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.50), dec!(60))],
            &[(dec!(0.55), dec!(10))],
            BASE_MS,
        ))
        .await;

    place(
        &venue,
        &mut aliases,
        &marketable_sell("D", dec!(0.49), dec!(100), false),
    )
    .await
    .expect("accepted");

    let got = normalize_all(&expect_events(&mut rx, 2).await, &aliases);
    assert_drained(&mut rx).await;
    assert_eq!(
        got,
        vec![
            g_fill(
                "D",
                Side::Sell,
                dec!(0.50),
                dec!(60),
                Liquidity::Taker,
                false
            ),
            g_order("D", OrderState::Canceled, Side::Sell, dec!(0.49), dec!(60)),
        ]
    );
}

/// D2: a FAK that finds nothing within its worst-price limit is rejected.
#[tokio::test(start_paused = true)]
async fn d2_fak_no_match_rejects() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    // Best bid 0.50 is below the 0.55 sell floor → nothing to match.
    venue
        .on_bus_event(&book(
            &[(dec!(0.50), dec!(60))],
            &[(dec!(0.60), dec!(10))],
            BASE_MS,
        ))
        .await;

    place(
        &venue,
        &mut aliases,
        &marketable_sell("D2", dec!(0.55), dec!(100), false),
    )
    .await
    .expect("accepted");

    let got = normalize_all(&expect_events(&mut rx, 1).await, &aliases);
    assert_drained(&mut rx).await;
    assert_eq!(
        got,
        vec![g_reject("D2", Side::Sell, dec!(0.55), RejectClass::Fak)]
    );
}

/// E: the §9 adverse case — a print fills the stale quote in the window between
/// a cancel request and the cancel landing; the remainder then cancels.
/// Live: `store::order_cancellation_is_terminal`.
#[tokio::test(start_paused = true)]
async fn e_cancel_race_print_fills_stale_quote() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    // Empty queue at our level so the print fills us immediately.
    venue
        .on_bus_event(&book(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            BASE_MS,
        ))
        .await;

    let acc = place(
        &venue,
        &mut aliases,
        &resting_buy("E", dec!(0.49), dec!(20)),
    )
    .await
    .expect("accepted");
    let open = expect_events(&mut rx, 1).await;

    // Request the cancel (round-trip pending), then a print lands first.
    let report = venue.cancel(&acc.order_id).await.expect("cancel report");
    assert_eq!(report.canceled, vec![acc.order_id.clone()]);
    venue
        .on_bus_event(&trade(dec!(0.49), dec!(12), Side::Sell, BASE_MS + 150))
        .await;
    let fill = expect_events(&mut rx, 2).await; // Fill + PartiallyFilled
    // Cancel lands on the remainder.
    let canceled = expect_events(&mut rx, 1).await;
    assert_drained(&mut rx).await;

    let got = normalize_all(&[open, fill, canceled].concat(), &aliases);
    assert_eq!(
        got,
        vec![
            g_order("E", OrderState::Open, Side::Buy, dec!(0.49), dec!(0)),
            g_fill("E", Side::Buy, dec!(0.49), dec!(12), Liquidity::Maker, true),
            g_order(
                "E",
                OrderState::PartiallyFilled,
                Side::Buy,
                dec!(0.49),
                dec!(12)
            ),
            g_order("E", OrderState::Canceled, Side::Buy, dec!(0.49), dec!(12)),
        ]
    );
}

/// F: `cancel_all` reports both working orders and terminalizes them.
/// Live: `venue::cancel_all_and_market_route_to_backend`.
#[tokio::test(start_paused = true)]
async fn f_cancel_all_terminalizes_both() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.48), dec!(10))],
            &[(dec!(0.51), dec!(50))],
            BASE_MS,
        ))
        .await;

    let a1 = place(
        &venue,
        &mut aliases,
        &resting_buy("F1", dec!(0.49), dec!(20)),
    )
    .await
    .expect("accepted");
    let a2 = place(
        &venue,
        &mut aliases,
        &resting_buy("F2", dec!(0.50), dec!(20)),
    )
    .await
    .expect("accepted");
    let opens = expect_events(&mut rx, 2).await; // both activate together

    let report = venue.cancel_all().await.expect("cancel_all report");
    assert!(report.all_canceled(), "both reported cancelled");
    assert_eq!(report.canceled.len(), 2);
    assert!(report.canceled.contains(&a1.order_id) && report.canceled.contains(&a2.order_id));

    let cancels = expect_events(&mut rx, 2).await;
    assert_drained(&mut rx).await;

    let got = normalize_all(&[opens, cancels].concat(), &aliases);
    assert_eq!(
        got,
        vec![
            g_order("F1", OrderState::Open, Side::Buy, dec!(0.49), dec!(0)),
            g_order("F2", OrderState::Open, Side::Buy, dec!(0.50), dec!(0)),
            g_order("F1", OrderState::Canceled, Side::Buy, dec!(0.49), dec!(0)),
            g_order("F2", OrderState::Canceled, Side::Buy, dec!(0.50), dec!(0)),
        ]
    );
}

/// G: cancelling an unknown order reports it as `AlreadyGone` and emits nothing.
/// Live: `venue::cancel_reports_partial_failures`.
#[tokio::test(start_paused = true)]
async fn g_cancel_unknown_reports_already_gone() {
    let (venue, mut rx) = make_venue();
    let ghost = OrderId::new("paper-999").expect("id");

    let report = venue.cancel(&ghost).await.expect("cancel report");
    assert!(report.canceled.is_empty());
    assert_eq!(report.not_canceled.len(), 1);
    assert_eq!(report.not_canceled[0].order_id, ghost);
    assert_eq!(
        report.not_canceled[0].reason,
        venue_api::RejectReason::AlreadyGone
    );
    assert_drained(&mut rx).await;
}

/// H: the ONE synchronous rejection — a quantity kind that doesn't match the
/// TIF/side fails `place()` directly (the normalizer forbids it upstream).
/// Live: `convert::qty_kind_mismatches_are_rejected`.
#[tokio::test(start_paused = true)]
async fn h_qty_kind_mismatch_is_synchronous_error() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    let err = place(&venue, &mut aliases, &bad_qty_gtc("H"))
        .await
        .expect_err("must reject synchronously");
    assert!(
        matches!(err, VenueError::Rejected(venue_api::RejectReason::Other(_))),
        "got {err:?}"
    );
    assert_drained(&mut rx).await;
}

/// I: window close terminalizes a working order (trading stopped).
/// Live: `store::order_cancellation_is_terminal`.
#[tokio::test(start_paused = true)]
async fn i_window_close_terminalizes_working_order() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            BASE_MS,
        ))
        .await;
    place(
        &venue,
        &mut aliases,
        &resting_buy("I", dec!(0.49), dec!(20)),
    )
    .await
    .expect("accepted");
    let open = expect_events(&mut rx, 1).await;

    venue
        .on_bus_event(&window_event(WindowLifecycle::Closed))
        .await;
    let canceled = expect_events(&mut rx, 1).await;
    assert_drained(&mut rx).await;

    let got = normalize_all(&[open, canceled].concat(), &aliases);
    assert_eq!(
        got,
        vec![
            g_order("I", OrderState::Open, Side::Buy, dec!(0.49), dec!(0)),
            g_order("I", OrderState::Canceled, Side::Buy, dec!(0.49), dec!(0)),
        ]
    );
}

/// J: a GTD order expires at its (60s-floored) deadline.
#[tokio::test(start_paused = true)]
async fn j_gtd_order_expires() {
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();

    venue
        .on_bus_event(&window_event(WindowLifecycle::Open))
        .await;
    venue
        .on_bus_event(&book(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            BASE_MS,
        ))
        .await;
    // Desired expiry 65s out (above the 60s security floor, so it stands).
    place(
        &venue,
        &mut aliases,
        &gtd_buy("J", dec!(0.49), dec!(20), BASE_MS + 65_000),
    )
    .await
    .expect("accepted");
    let open = expect_events(&mut rx, 1).await;
    let expired = expect_events(&mut rx, 1).await; // auto-advances to the expiry
    assert_drained(&mut rx).await;

    let got = normalize_all(&[open, expired].concat(), &aliases);
    assert_eq!(
        got,
        vec![
            g_order("J", OrderState::Open, Side::Buy, dec!(0.49), dec!(0)),
            g_order("J", OrderState::Expired, Side::Buy, dec!(0.49), dec!(0)),
        ]
    );
}

/// Dogfoods Deliverable 1 end to end: a market-data session recorded through
/// `journal::Recorder` (gzip segments) and read back with `journal::ReplayReader`
/// replays bit-exactly into `PaperVenue::on_bus_event` and drives a real fill —
/// the same code path `bot record` captures with.
#[tokio::test(start_paused = true)]
async fn replayed_recorded_data_drives_a_paper_fill() {
    // Record a tiny session to a temp dir (constant clock — the recorder runs on
    // its own OS thread; the replayed events keep their own timestamps).
    let dir = std::env::temp_dir().join(format!("venue-parity-replay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let recorder = Recorder::spawn(
        RecorderParams {
            out_dir: dir.clone(),
            ..RecorderParams::default()
        },
        || TimestampMs::from_millis(BASE_MS),
    )
    .expect("recorder");
    recorder.record(&window_event(WindowLifecycle::Open));
    recorder.record(&book(
        &[(dec!(0.49), dec!(0))],
        &[(dec!(0.51), dec!(50))],
        BASE_MS,
    ));
    recorder.record(&trade(dec!(0.49), dec!(30), Side::Sell, BASE_MS + 250));
    let stats = recorder.finish().expect("finish");
    assert_eq!(stats.records, 3);
    assert_eq!(stats.dropped, 0);

    let replayed: Vec<Event> = ReplayReader::open(&dir)
        .expect("open replay")
        .events()
        .collect::<std::io::Result<_>>()
        .expect("decode");
    assert_eq!(replayed.len(), 3);

    // Drive the venue with the replayed market data; place a resting order after
    // the book is known, then let the replayed print fill it.
    let (venue, mut rx) = make_venue();
    let mut aliases = HashMap::new();
    venue.on_bus_event(&replayed[0]).await; // Window Open
    venue.on_bus_event(&replayed[1]).await; // Book (empty queue)
    place(
        &venue,
        &mut aliases,
        &resting_buy("R", dec!(0.49), dec!(20)),
    )
    .await
    .expect("accepted");
    let open = expect_events(&mut rx, 1).await;
    venue.on_bus_event(&replayed[2]).await; // LastTrade → fills the 20
    let fill = expect_events(&mut rx, 2).await;
    assert_drained(&mut rx).await;

    let got = normalize_all(&[open, fill].concat(), &aliases);
    assert_eq!(
        got,
        vec![
            g_order("R", OrderState::Open, Side::Buy, dec!(0.49), dec!(0)),
            g_fill("R", Side::Buy, dec!(0.49), dec!(20), Liquidity::Maker, true),
            g_order("R", OrderState::Filled, Side::Buy, dec!(0.49), dec!(20)),
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
