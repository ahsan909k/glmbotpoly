//! Shared builders for the dashboard integration tests: a seeded
//! [`DashboardHandle`], event constructors, routers, and a `oneshot` GET helper.

#![allow(dead_code, clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use core_types::{AnchorSource, ConditionId, OrderId, TokenPair};
use core_types::{
    Asset, BookLevel, BookSnapshot, BreakerKind, Decimal, Dollars, DurationMs, Event, FeedHealth,
    Fill, InputAges, InventorySnapshot, MarketInfo, Mode, ModelHealth, ModelHealthEvent,
    ModelHealthReason, ModelSnapshot, Outcome, Price, PriceSource, RiskEvent, RoundDir, Series,
    SettlementSummary, Side, SideInventory, Size, TickKind, TickSize, TimestampMs, TokenId,
    WindowDuration, WindowId, WindowLifecycle,
};
use dashboard::{
    ArmingView, ControlOutcome, ControlRequest, ControlStateSnapshot, DashboardCommand,
    DashboardHandle, OutcomeKind, ParamsView,
};
use engine::RiskStateSnapshot;
use http_body_util::BodyExt;
use rust_decimal::dec;
use tokio::sync::mpsc;
use tower::ServiceExt;
use venue_api::Wallet;
use venue_paper::PaperLedgerSnapshot;

/// The auth token used by the enforced-router tests.
pub const TOKEN: &str = "test-secret-token";
/// The seeded window's open time (2026-06-15T00:00:00Z).
pub const OPEN_MS: i64 = 1_781_481_600_000;

pub fn ts(ms: i64) -> TimestampMs {
    TimestampMs::from_millis(ms)
}

pub fn series() -> Series {
    Series {
        asset: Asset::Btc,
        duration: WindowDuration::M5,
    }
}

pub fn window_id() -> WindowId {
    WindowId {
        series: series(),
        open_time: ts(OPEN_MS),
    }
}

pub fn up_token() -> TokenId {
    TokenId::new("1").unwrap()
}

pub fn down_token() -> TokenId {
    TokenId::new("2").unwrap()
}

pub fn px(d: Decimal) -> Price {
    Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
}

pub fn sz(d: Decimal) -> Size {
    Size::new(d).unwrap()
}

pub fn condition_id() -> ConditionId {
    ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap()
}

pub fn market() -> Arc<MarketInfo> {
    Arc::new(MarketInfo {
        window: window_id(),
        event_slug: "btc-updown-5m-test".to_owned(),
        condition_id: condition_id(),
        tokens: TokenPair {
            up: up_token(),
            down: down_token(),
        },
        close_time: ts(OPEN_MS + 300_000),
        strike: Some(dec!(60000)),
        tick_size: TickSize::T001,
        min_order_size: sz(dec!(5)),
        fees: core_types::FeeParams {
            rate: dec!(0.07),
            exponent: 1,
            taker_only: true,
            rebate_rate: dec!(0.2),
            enabled: true,
        },
        neg_risk: false,
        resolution: core_types::ResolutionSource::classify(
            "https://data.chain.link/streams/btc-usd",
        ),
    })
}

pub fn loopback() -> SocketAddr {
    "127.0.0.1:8080".parse().unwrap()
}

pub fn public_addr() -> SocketAddr {
    "0.0.0.0:8080".parse().unwrap()
}

// ---- event builders --------------------------------------------------------

pub fn window_event(lifecycle: WindowLifecycle) -> Event {
    Event::Window {
        market: market(),
        lifecycle,
    }
}

pub fn model_event(p_up: f64, ts_ms: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window_id()),
        p_up,
        z: 0.1,
        sigma_1s: 0.0005,
        sigma_tau: 0.01,
        basis: 0.0,
        anchor: AnchorSource::Chainlink,
        health: ModelHealth::Ready,
        reason: ModelHealthReason::Nominal,
        input_ages: InputAges {
            chainlink: DurationMs::from_millis(100),
            binance: DurationMs::from_millis(100),
        },
        ts: ts(ts_ms),
    })
}

pub fn fill_event(outcome: Outcome, price: Decimal, size: Decimal, ts_ms: i64) -> Event {
    let token = if outcome == Outcome::Up {
        up_token()
    } else {
        down_token()
    };
    Event::Fill(Arc::new(Fill {
        order_id: OrderId::new("o-1").unwrap(),
        trade_id: None,
        window: window_id(),
        token_id: token,
        outcome,
        side: Side::Buy,
        price: px(price),
        size: sz(size),
        liquidity: core_types::Liquidity::Maker,
        fee: Dollars::ZERO,
        ts_venue: ts(ts_ms),
        ts_local: ts(ts_ms),
    }))
}

pub fn taker_fill_event(outcome: Outcome, ts_ms: i64, order_id: &str) -> Event {
    let token = if outcome == Outcome::Up {
        up_token()
    } else {
        down_token()
    };
    Event::Fill(Arc::new(Fill {
        order_id: OrderId::new(order_id).unwrap(),
        trade_id: Some(format!("t-{order_id}")),
        window: window_id(),
        token_id: token,
        outcome,
        side: Side::Buy,
        price: px(dec!(0.50)),
        size: sz(dec!(10)),
        liquidity: core_types::Liquidity::Taker,
        fee: Dollars::new(dec!(0.1)),
        ts_venue: ts(ts_ms),
        ts_local: ts(ts_ms),
    }))
}

pub fn order_event(order_id: &str, state: core_types::OrderState) -> Event {
    Event::OrderUpdate(Arc::new(core_types::OrderUpdate {
        order_id: OrderId::new(order_id).unwrap(),
        window: window_id(),
        token_id: up_token(),
        side: Side::Buy,
        state,
        price: px(dec!(0.49)),
        original_size: sz(dec!(10)),
        filled_size: Size::ZERO,
        reject_reason: None,
        ts_venue: None,
        ts_local: ts(OPEN_MS + 5),
    }))
}

pub fn top_event(token: &TokenId, bid: Decimal, ask: Decimal, ts_ms: i64) -> Event {
    Event::TopOfBook {
        token_id: Arc::new(token.clone()),
        top: core_types::TopOfBook {
            bid: Some(BookLevel {
                price: px(bid),
                size: sz(dec!(100)),
            }),
            ask: Some(BookLevel {
                price: px(ask),
                size: sz(dec!(100)),
            }),
            ts: ts(ts_ms),
        },
    }
}

pub fn book_event(token: &TokenId, ts_ms: i64) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: token.clone(),
        condition_id: condition_id(),
        bids: vec![
            BookLevel {
                price: px(dec!(0.49)),
                size: sz(dec!(100)),
            },
            BookLevel {
                price: px(dec!(0.48)),
                size: sz(dec!(200)),
            },
        ],
        asks: vec![BookLevel {
            price: px(dec!(0.51)),
            size: sz(dec!(80)),
        }],
        ts: ts(ts_ms),
        seq_hash: None,
    }))
}

pub fn last_trade_event(token: &TokenId, ts_ms: i64) -> Event {
    Event::LastTrade {
        token_id: Arc::new(token.clone()),
        price: px(dec!(0.50)),
        size: sz(dec!(10)),
        side: Side::Buy,
        ts: ts(ts_ms),
    }
}

pub fn inventory_event() -> Event {
    Event::Inventory(Arc::new(InventorySnapshot::derive(
        window_id(),
        SideInventory {
            shares: sz(dec!(100)),
            cost: Dollars::new(dec!(48)),
        },
        SideInventory {
            shares: sz(dec!(100)),
            cost: Dollars::new(dec!(49)),
        },
        ts(OPEN_MS + 20),
    )))
}

pub fn feed_stale_event() -> Event {
    Event::FeedHealth(FeedHealth::Stale {
        source: PriceSource::ChainlinkRtds,
        asset: Asset::Btc,
        kind: TickKind::Vendor,
        age: DurationMs::from_millis(5_250),
    })
}

pub fn model_health_event() -> Event {
    Event::ModelHealth(ModelHealthEvent {
        asset: Asset::Btc,
        health: ModelHealth::Ready,
        reason: ModelHealthReason::Nominal,
        ts: ts(OPEN_MS + 25),
    })
}

pub fn settlement_event() -> Event {
    Event::Settlement(Arc::new(SettlementSummary::close(
        window_id(),
        Outcome::Up,
        SideInventory {
            shares: sz(dec!(100)),
            cost: Dollars::new(dec!(48)),
        },
        SideInventory {
            shares: sz(dec!(100)),
            cost: Dollars::new(dec!(49)),
        },
        Size::ZERO,
        Dollars::ZERO,
        Dollars::new(dec!(3)),
        ts(OPEN_MS + 300_000),
    )))
}

pub fn wallet(amount: Decimal) -> Wallet {
    Wallet {
        collateral_available: Dollars::new(amount),
        collateral_total: Dollars::new(amount),
        positions: vec![],
    }
}

pub fn ledger() -> PaperLedgerSnapshot {
    PaperLedgerSnapshot {
        collateral: Dollars::new(dec!(10000)),
        positions: vec![(down_token(), dec!(-5))],
        fees_paid: Dollars::new(dec!(1.25)),
        rebate_accrued: Dollars::new(dec!(0.30)),
        rebate_credited: Dollars::new(dec!(2)),
    }
}

pub fn risk_snapshot() -> RiskStateSnapshot {
    RiskStateSnapshot {
        tripped: vec![BreakerKind::Manual],
        window_loss_active: false,
        halted_windows: 0,
        open_notional: Dollars::new(dec!(120)),
        open_notional_cap: Dollars::new(dec!(1000)),
        daily_pnl: Dollars::new(dec!(3)),
        error_count: 0,
        sanity_breached: false,
        globally_halted: false,
    }
}

pub fn params_view() -> ParamsView {
    ParamsView {
        paper_capital: Some(Dollars::new(dec!(10000))),
        entries: vec![
            ("min_edge".to_owned(), "0.01".to_owned()),
            ("k1".to_owned(), "1".to_owned()),
        ],
    }
}

/// A handle pre-loaded with one full paper session (open → fills → resolve →
/// settle), market/feed/model state, a tripped breaker, and the setter-driven
/// wallet/ledger/risk/session/params. The live namespace is left empty (proving
/// namespacing).
pub fn seeded_handle() -> DashboardHandle {
    let h = DashboardHandle::new(64, ts(OPEN_MS));
    let m = market();
    let p = Mode::Paper;
    h.project(
        p,
        &Event::Window {
            market: Arc::clone(&m),
            lifecycle: WindowLifecycle::Open,
        },
        ts(OPEN_MS),
    );
    h.project(p, &model_event(0.50, OPEN_MS), ts(OPEN_MS));
    // A live resting order on the Up token (for the ladder's our_orders).
    h.project(
        p,
        &order_event("o-live", core_types::OrderState::Open),
        ts(OPEN_MS + 5),
    );
    h.project(p, &book_event(&up_token(), OPEN_MS + 1), ts(OPEN_MS + 1));
    h.project(
        p,
        &top_event(&down_token(), dec!(0.50), dec!(0.52), OPEN_MS + 1),
        ts(OPEN_MS + 1),
    );
    h.project(
        p,
        &last_trade_event(&up_token(), OPEN_MS + 2),
        ts(OPEN_MS + 2),
    );
    h.project(
        p,
        &fill_event(Outcome::Up, dec!(0.48), dec!(100), OPEN_MS + 10),
        ts(OPEN_MS + 10),
    );
    h.project(
        p,
        &fill_event(Outcome::Down, dec!(0.49), dec!(100), OPEN_MS + 20),
        ts(OPEN_MS + 20),
    );
    h.project(p, &inventory_event(), ts(OPEN_MS + 20));
    h.project(p, &feed_stale_event(), ts(OPEN_MS + 25));
    h.project(p, &model_health_event(), ts(OPEN_MS + 25));
    h.project(
        p,
        &Event::Risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::Manual,
        }),
        ts(OPEN_MS + 30),
    );
    h.project(p, &model_event(0.55, OPEN_MS + 5_000), ts(OPEN_MS + 5_000));
    h.project(
        p,
        &Event::Window {
            market: Arc::clone(&m),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        },
        ts(OPEN_MS + 300_000),
    );
    h.project(p, &settlement_event(), ts(OPEN_MS + 300_000));
    h.set_wallet(p, wallet(dec!(10000)), ts(OPEN_MS + 300_001));
    h.set_paper_ledger(p, ledger(), ts(OPEN_MS + 300_001));
    h.set_risk(p, risk_snapshot(), ts(OPEN_MS + 300_001));
    h.set_session(p, true, ts(OPEN_MS + 300_001));
    h.set_params(params_view(), ts(OPEN_MS + 300_001));
    h
}

/// A handle with one maker, one mid-window taker, and one late-window taker fill,
/// plus the late-window gate threshold — for attribution-tag tests.
pub fn attribution_handle() -> DashboardHandle {
    let h = DashboardHandle::new(64, ts(OPEN_MS));
    let p = Mode::Paper;
    h.project(p, &window_event(WindowLifecycle::Open), ts(OPEN_MS));
    h.set_params(
        ParamsView {
            paper_capital: None,
            entries: vec![("engine.late_window_tau_secs".to_owned(), "30".to_owned())],
        },
        ts(OPEN_MS),
    );
    // Maker; mid-window taker (>30s to close); late-window taker (<30s to close,
    // close = OPEN_MS + 300_000).
    h.project(
        p,
        &fill_event(Outcome::Up, dec!(0.48), dec!(10), OPEN_MS + 10),
        ts(OPEN_MS + 10),
    );
    h.project(
        p,
        &taker_fill_event(Outcome::Up, OPEN_MS + 100_000, "tk-early"),
        ts(OPEN_MS + 100_000),
    );
    h.project(
        p,
        &taker_fill_event(Outcome::Up, OPEN_MS + 280_000, "tk-late"),
        ts(OPEN_MS + 280_000),
    );
    h
}

/// Builds an unauthenticated (loopback, no token) router for endpoint tests.
pub fn open_router(handle: DashboardHandle) -> Router {
    dashboard::router(handle, loopback(), None).expect("open router")
}

/// Builds an enforced-token router for auth tests.
pub fn enforced_router(handle: DashboardHandle) -> Router {
    dashboard::router(handle, loopback(), Some(TOKEN.to_owned())).expect("enforced router")
}

/// A log of the commands a fake responder received, for control-test assertions.
pub type CommandLog = Arc<std::sync::Mutex<Vec<DashboardCommand>>>;

/// A default control-state snapshot for fake-responder outcomes.
pub fn control_snapshot() -> ControlStateSnapshot {
    ControlStateSnapshot {
        halted: false,
        enabled_series: vec!["BTC-5m".to_owned(), "ETH-15m".to_owned()],
        arming: ArmingView {
            config_enabled: false,
            env_confirmed: false,
            session_armed: false,
            can_arm: false,
            pending: false,
        },
        param_overrides: vec![],
        paper_capital: Some(Dollars::new(dec!(10000))),
    }
}

/// Builds a [`ControlOutcome`] with the default snapshot.
pub fn outcome(kind: OutcomeKind, error: Option<&str>) -> ControlOutcome {
    ControlOutcome {
        kind,
        error: error.map(str::to_owned),
        state: control_snapshot(),
    }
}

/// A handle wired with a request sink, plus a background responder that records
/// every command and replies with `responder(&command)`. Returns the command log
/// so control tests can assert what the orchestrator received. Keep nothing extra
/// alive: dropping the handle stops the responder.
pub fn control_handle_with(
    responder: impl Fn(&DashboardCommand) -> ControlOutcome + Send + 'static,
) -> (DashboardHandle, CommandLog) {
    let (tx, mut rx) = mpsc::channel::<ControlRequest>(8);
    let log: CommandLog = Arc::new(std::sync::Mutex::new(Vec::new()));
    let log2 = Arc::clone(&log);
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            log2.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(req.command.clone());
            let out = responder(&req.command);
            let _ = req.reply.send(out);
        }
    });
    (
        DashboardHandle::new(64, ts(OPEN_MS)).with_request_sink(tx),
        log,
    )
}

/// A control handle whose responder accepts every command.
pub fn control_handle() -> (DashboardHandle, CommandLog) {
    control_handle_with(|_| outcome(OutcomeKind::Accepted, None))
}

/// Issues a POST via `oneshot`. `body` is sent as a JSON request body (with the
/// `application/json` content type); `None` sends an empty body.
pub async fn post(
    router: &Router,
    uri: &str,
    token: Option<&str>,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder().method("POST").uri(uri);
    if let Some(t) = token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let req = match body {
        Some(b) => builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(b.to_string()))
            .expect("request"),
        None => builder.body(Body::empty()).expect("request"),
    };
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, value)
}

/// Issues a GET via `oneshot` and returns the status, `Content-Type`, and the
/// raw body as a string (for the static UI routes, which are not JSON).
pub async fn get_raw(router: &Router, uri: &str) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    (
        status,
        content_type,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

/// Issues a GET via `oneshot` and returns the status and the JSON body
/// (or `Null` for an empty body).
pub async fn get(
    router: &Router,
    uri: &str,
    token: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder().method("GET").uri(uri);
    if let Some(t) = token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let req = builder.body(Body::empty()).expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, value)
}
