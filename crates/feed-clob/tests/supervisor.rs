//! Deterministic supervisor and driver tests: paused tokio time over the
//! shared scripted fake transport (feed-rtds/feed-binance precedent). Pins
//! the per-window connection lifecycle: subscribe-on-connect, PING cadence,
//! presubscribe lead, gap-free rollover overlap, resolution teardown and
//! linger cap, market-event dedup across connections, machine-driven and
//! forced recycles, dead-socket teardown, and clean shutdown.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use core_types::{
    Asset, BookHealth, BookUnreliableReason, ConditionId, DurationMs, Event, FeeParams, MarketInfo,
    MarketLifecycleEvent, ResolutionSource, Series, Size, TickSize, TimestampMs, TokenId,
    TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use feed_clob::supervisor::{ClobArgs, ClobCommand, ClobStatus, run};
use feed_clob::{BackoffParams, ClobParams, FeedError};
use feed_util::WsFrame;
use feed_util::fake::{ConnHandle, FakeTransport, script};
use rust_decimal::dec;
use tokio::sync::{mpsc, watch};

/// One minute past an arbitrary epoch, mirroring the other feed tests.
const BASE_MS: i64 = 1_800_000_060_000;

/// Wall clock locked to the (paused) tokio clock.
fn paused_now_fn() -> impl Fn() -> TimestampMs + Send + Sync + Clone {
    let start = tokio::time::Instant::now();
    move || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
        TimestampMs::from_millis(BASE_MS + elapsed)
    }
}

fn params() -> ClobParams {
    ClobParams {
        url: "wss://fake".to_owned(),
        ping_interval: Duration::from_secs(5),
        connect_timeout: Duration::from_secs(1),
        backoff: BackoffParams {
            initial: Duration::from_millis(250),
            max: Duration::from_millis(10_000),
            multiplier: 2.0,
        },
        book_stale_after: DurationMs::from_millis(10_000),
        presubscribe_lead: DurationMs::from_millis(90_000),
        publish_interval: DurationMs::from_millis(200),
        resolution_linger: DurationMs::from_millis(180_000),
    }
}

/// Test market `n` (distinct ids per n): Up token `"1n"`, Down token `"2n"`,
/// opening at `BASE_MS + open_offset_ms`, 5-minute window.
fn market(n: u8, open_offset_ms: i64) -> Arc<MarketInfo> {
    let open = TimestampMs::from_millis(BASE_MS + open_offset_ms);
    Arc::new(MarketInfo {
        window: WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: open,
        },
        event_slug: format!("btc-updown-5m-test-{n}"),
        condition_id: ConditionId::new(format!("0x{}", format!("{n:02x}").repeat(32)))
            .expect("valid condition id"),
        tokens: TokenPair {
            up: TokenId::new(format!("1{n}")).expect("valid token id"),
            down: TokenId::new(format!("2{n}")).expect("valid token id"),
        },
        close_time: open.saturating_add(DurationMs::from_millis(300_000)),
        strike: Some(dec!(104000)),
        tick_size: TickSize::T001,
        min_order_size: Size::new(dec!(5)).expect("valid size"),
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

fn book_frame(market: &MarketInfo, token: &TokenId, bid: &str, ask: &str) -> WsFrame {
    WsFrame::Text(format!(
        r#"{{"event_type":"book","asset_id":"{}","market":"{}","bids":[{{"price":"{bid}","size":"100"}}],"asks":[{{"price":"{ask}","size":"100"}}],"timestamp":"{BASE_MS}","hash":"h"}}"#,
        token.as_str(),
        market.condition_id.as_str(),
    ))
}

fn change_frame(
    market: &MarketInfo,
    token: &TokenId,
    side: &str,
    price: &str,
    size: &str,
) -> WsFrame {
    WsFrame::Text(format!(
        r#"{{"event_type":"price_change","market":"{}","timestamp":"{BASE_MS}","price_changes":[{{"asset_id":"{}","price":"{price}","size":"{size}","side":"{side}","best_bid":"","best_ask":""}}]}}"#,
        market.condition_id.as_str(),
        token.as_str(),
    ))
}

fn resolved_frame(market: &MarketInfo, winning: &TokenId) -> WsFrame {
    WsFrame::Text(format!(
        r#"{{"event_type":"market_resolved","market":"{}","winning_asset_id":"{}","winning_outcome":"Up","timestamp":"{BASE_MS}"}}"#,
        market.condition_id.as_str(),
        winning.as_str(),
    ))
}

/// Sends both tokens' snapshots so the machine reaches trusted books.
fn snapshot_both(conn: &ConnHandle, market: &MarketInfo) {
    conn.frame_tx
        .send(book_frame(market, &market.tokens.up, "0.48", "0.52"))
        .expect("driver alive");
    conn.frame_tx
        .send(book_frame(market, &market.tokens.down, "0.47", "0.53"))
        .expect("driver alive");
}

struct Harness {
    supervisor: tokio::task::JoinHandle<Result<(), FeedError>>,
    bus_rx: mpsc::Receiver<Event>,
    window_tx: mpsc::Sender<(Arc<MarketInfo>, WindowLifecycle)>,
    market_rx: mpsc::Receiver<MarketLifecycleEvent>,
    command_tx: mpsc::Sender<ClobCommand>,
    status_rx: watch::Receiver<ClobStatus>,
    shutdown_tx: watch::Sender<bool>,
    factory_calls: Arc<Mutex<usize>>,
}

/// Spawns the supervisor over a factory that hands out the scripted
/// transports in order (one per spawned connection task).
fn spawn_supervisor(transports: Vec<FakeTransport>) -> Harness {
    let (bus_tx, bus_rx) = mpsc::channel(256);
    let (window_tx, window_rx) = mpsc::channel(64);
    let (market_tx, market_rx) = mpsc::channel(64);
    let (command_tx, command_rx) = mpsc::channel(8);
    let (status_tx, status_rx) = watch::channel(ClobStatus::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let factory_calls = Arc::new(Mutex::new(0usize));
    let calls = Arc::clone(&factory_calls);
    let mut pool = VecDeque::from(transports);
    let supervisor = tokio::spawn(run(ClobArgs {
        params: params(),
        transport_factory: move || {
            *calls.lock().expect("no poison") += 1;
            pool.pop_front().expect("scripted transports exhausted")
        },
        now_fn: paused_now_fn(),
        bus_tx,
        window_rx,
        market_tx: Some(market_tx),
        command_rx: Some(command_rx),
        status_tx: Some(status_tx),
        shutdown_rx,
        backoff_seed: Some(42),
    }));
    Harness {
        supervisor,
        bus_rx,
        window_tx,
        market_rx,
        command_tx,
        status_rx,
        shutdown_tx,
        factory_calls,
    }
}

/// Receives bus events until `pred` returns Some, skipping everything else.
async fn recv_until<T>(
    bus_rx: &mut mpsc::Receiver<Event>,
    mut pred: impl FnMut(&Event) -> Option<T>,
) -> T {
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let event = bus_rx.recv().await.expect("bus open");
            if let Some(value) = pred(&event) {
                return value;
            }
        }
    })
    .await
    .expect("expected event within (paused) 600s")
}

/// Waits (in paused time) until `n` connect attempts were recorded.
async fn wait_attempts(attempt_at: &Arc<Mutex<Vec<Duration>>>, n: usize) -> Vec<Duration> {
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let times = attempt_at.lock().expect("no poison").clone();
            if times.len() >= n {
                return times;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
    .await
    .expect("expected connect attempts within (paused) 600s")
}

async fn shutdown_and_join(h: Harness) {
    h.shutdown_tx.send(true).expect("supervisor alive");
    let result = tokio::time::timeout(Duration::from_secs(600), h.supervisor)
        .await
        .expect("supervisor exits promptly")
        .expect("no panic");
    assert!(matches!(result, Ok(())));
}

#[tokio::test(start_paused = true)]
async fn open_window_connects_subscribes_pings_and_publishes() {
    let (transport, mut handles, _) = script(&[true]);
    let urls = transport.url_log();
    let mut h = spawn_supervisor(vec![transport]);
    let m = market(1, 0);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let mut conn = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connection spawned");

    // The first outbound frame is THE subscribe message with both tokens
    // and the custom feature flag.
    let subscribe = conn.sent_rx.recv().await.expect("subscribe sent");
    assert_eq!(
        subscribe,
        r#"{"assets_ids":["11","21"],"custom_feature_enabled":true,"type":"market"}"#
    );
    assert_eq!(urls.lock().expect("no poison").as_slice(), ["wss://fake"]);

    // Snapshots publish Book + TopOfBook on the bus.
    snapshot_both(&conn, &m);
    let snap = recv_until(&mut h.bus_rx, |e| match e {
        Event::Book(b) if b.token_id.as_str() == "11" => Some(Arc::clone(b)),
        _ => None,
    })
    .await;
    assert_eq!(snap.bids[0].price.as_decimal(), dec!(0.48));
    assert_eq!(snap.seq_hash.as_deref(), Some("h"));
    recv_until(&mut h.bus_rx, |e| match e {
        Event::TopOfBook { token_id, .. } if token_id.as_str() == "21" => Some(()),
        _ => None,
    })
    .await;

    // A delta updates the book and a trade prints.
    conn.frame_tx
        .send(change_frame(&m, &m.tokens.up, "BUY", "0.49", "10"))
        .expect("driver alive");
    recv_until(&mut h.bus_rx, |e| match e {
        Event::TopOfBook { token_id, top } if token_id.as_str() == "11" => Some(*top),
        _ => None,
    })
    .await;
    conn.frame_tx
        .send(WsFrame::Text(format!(
            r#"{{"event_type":"last_trade_price","asset_id":"11","market":"{}","price":"0.49","side":"BUY","size":"7","fee_rate_bps":"0","timestamp":"{BASE_MS}"}}"#,
            m.condition_id.as_str()
        )))
        .expect("driver alive");
    recv_until(&mut h.bus_rx, |e| match e {
        Event::LastTrade { token_id, .. } if token_id.as_str() == "11" => Some(()),
        _ => None,
    })
    .await;

    // PINGs flow every 5 s; the server PONG keeps the socket alive. Book
    // deltas keep flowing on BOTH tokens so the active-window staleness
    // watchdog (which tracks the stalest token) stays away — this loop
    // tests the keepalive, not staleness.
    let mut pings = 0;
    for _ in 0..3 {
        conn.frame_tx
            .send(change_frame(&m, &m.tokens.up, "BUY", "0.45", "1"))
            .expect("driver alive");
        conn.frame_tx
            .send(change_frame(&m, &m.tokens.down, "BUY", "0.44", "1"))
            .expect("driver alive");
        tokio::time::sleep(Duration::from_secs(5)).await;
        while let Ok(sent) = conn.sent_rx.try_recv() {
            if sent == "PING" {
                pings += 1;
                conn.frame_tx
                    .send(WsFrame::Text("PONG".to_owned()))
                    .expect("driver alive");
            }
        }
    }
    assert!(pings >= 2, "expected PINGs at the 5 s cadence, got {pings}");

    // Status reflects the live window.
    let status = h.status_rx.borrow().clone();
    assert_eq!(status.windows.len(), 1);
    assert!(status.windows[0].connected);
    assert!(status.windows[0].machine.trusted);

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn discovered_window_connects_at_presubscribe_lead() {
    let (transport, _handles, attempt_at) = script(&[true]);
    let h = spawn_supervisor(vec![transport]);
    // Opens 4 minutes from now; lead is 90 s → connect at t ≈ 150 s.
    let m = market(2, 240_000);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Discovered))
        .await
        .expect("supervisor alive");

    tokio::time::sleep(Duration::from_secs(140)).await;
    assert_eq!(
        *h.factory_calls.lock().expect("no poison"),
        0,
        "must not connect before open − presubscribe_lead"
    );
    let times = wait_attempts(&attempt_at, 1).await;
    assert!(
        times[0] >= Duration::from_secs(150) && times[0] < Duration::from_secs(153),
        "connect at open − lead (±scan granularity): {times:?}"
    );

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn rollover_overlaps_connections_and_resolution_tears_down() {
    let (t1, mut handles1, _) = script(&[true]);
    let (t2, mut handles2, _) = script(&[true]);
    let mut h = spawn_supervisor(vec![t1, t2]);
    let current = market(3, 0);
    let next = market(4, 60_000); // opens in 60 s → due immediately (lead 90 s)

    h.window_tx
        .send((Arc::clone(&current), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let conn_current = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles1.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("current window connected");
    snapshot_both(&conn_current, &current);

    // The next window is announced while the current one is live: BOTH
    // connections are up before the boundary — the gap-free rollover.
    h.window_tx
        .send((Arc::clone(&next), WindowLifecycle::Discovered))
        .await
        .expect("supervisor alive");
    let conn_next = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles2.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("next window connected while current still live");
    snapshot_both(&conn_next, &next);
    recv_until(&mut h.bus_rx, |e| match e {
        Event::Book(b) if b.token_id == next.tokens.up => Some(()),
        _ => None,
    })
    .await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let status = h.status_rx.borrow().clone();
    assert_eq!(status.windows.len(), 2, "overlap during the lead window");
    assert!(status.windows.iter().all(|w| w.connected));

    // Boundary: current closes (linger armed, connection stays), then its
    // market_resolved arrives — forwarded to the scheduler channel once and
    // the connection torn down.
    h.window_tx
        .send((Arc::clone(&current), WindowLifecycle::Closed))
        .await
        .expect("supervisor alive");
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        h.status_rx.borrow().windows.len(),
        2,
        "closed window lingers awaiting market_resolved"
    );
    conn_current
        .frame_tx
        .send(resolved_frame(&current, &current.tokens.up))
        .expect("driver alive");
    let resolved = tokio::time::timeout(Duration::from_secs(600), h.market_rx.recv())
        .await
        .expect("resolution forwarded")
        .expect("channel open");
    match resolved {
        MarketLifecycleEvent::MarketResolved {
            condition_id,
            winning_token,
            ..
        } => {
            assert_eq!(condition_id, current.condition_id);
            assert_eq!(winning_token, current.tokens.up);
        }
        other => panic!("expected MarketResolved, got {other:?}"),
    }
    tokio::time::sleep(Duration::from_secs(3)).await;
    let status = h.status_rx.borrow().clone();
    assert_eq!(status.windows.len(), 1, "resolved window torn down");
    assert_eq!(status.windows[0].window, next.window);

    // The next window's connection is untouched by the teardown.
    conn_next
        .frame_tx
        .send(change_frame(&next, &next.tokens.up, "BUY", "0.49", "5"))
        .expect("next window driver alive");
    recv_until(&mut h.bus_rx, |e| match e {
        Event::TopOfBook { token_id, .. } if *token_id.as_ref() == next.tokens.up => Some(()),
        _ => None,
    })
    .await;

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn duplicate_market_resolved_across_connections_is_forwarded_once() {
    let (t1, mut handles1, _) = script(&[true]);
    let (t2, mut handles2, _) = script(&[true]);
    let mut h = spawn_supervisor(vec![t1, t2]);
    let a = market(5, 0);
    let b = market(6, 60_000);

    h.window_tx
        .send((Arc::clone(&a), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    h.window_tx
        .send((Arc::clone(&b), WindowLifecycle::Discovered))
        .await
        .expect("supervisor alive");
    let conn_a = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles1.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("a connected");
    let conn_b = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles2.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("b connected");

    // The same resolution lands on BOTH connections (broadcast scope).
    conn_a
        .frame_tx
        .send(resolved_frame(&a, &a.tokens.down))
        .expect("driver alive");
    conn_b
        .frame_tx
        .send(resolved_frame(&a, &a.tokens.down))
        .expect("driver alive");
    let first = tokio::time::timeout(Duration::from_secs(600), h.market_rx.recv())
        .await
        .expect("first forwarded")
        .expect("channel open");
    assert!(matches!(first, MarketLifecycleEvent::MarketResolved { .. }));
    // No second forward within a generous window.
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert!(
        h.market_rx.try_recv().is_err(),
        "duplicate resolution must be deduplicated"
    );

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn forced_recycle_reconnects_and_recovers() {
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let mut h = spawn_supervisor(vec![transport]);
    let m = market(7, 0);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let conn1 = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connected");
    snapshot_both(&conn1, &m);
    recv_until(&mut h.bus_rx, |e| match e {
        Event::Book(b) if b.token_id == m.tokens.down => Some(()),
        _ => None,
    })
    .await;

    // Forced recycle: Unreliable{Disconnected} on the bus, a second connect
    // attempt, fresh snapshots, Recovered.
    h.command_tx
        .send(ClobCommand::RecycleWindow(m.window))
        .await
        .expect("supervisor alive");
    let unreliable = recv_until(&mut h.bus_rx, |e| match e {
        Event::BookHealth(health @ BookHealth::Unreliable { .. }) => Some(*health),
        _ => None,
    })
    .await;
    assert!(matches!(
        unreliable,
        BookHealth::Unreliable {
            reason: BookUnreliableReason::Disconnected,
            ..
        }
    ));
    wait_attempts(&attempt_at, 2).await;
    let mut conn2 = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("reconnected");
    let resubscribe = conn2.sent_rx.recv().await.expect("resubscribed");
    assert!(
        resubscribe.contains("assets_ids"),
        "subscribe on every reconnect"
    );
    snapshot_both(&conn2, &m);
    recv_until(&mut h.bus_rx, |e| match e {
        Event::BookHealth(BookHealth::Recovered { outage, .. }) => Some(*outage),
        _ => None,
    })
    .await;

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn trade_at_touch_uncrosses_without_reconnecting() {
    // Live-verified: a trade at the touch arrives as a crossing delta. The
    // book uncrosses locally by implied consumption — same connection, no
    // health events, top published from the post-trade state.
    let (transport, mut handles, attempt_at) = script(&[true]);
    let mut h = spawn_supervisor(vec![transport]);
    let m = market(8, 0);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let conn = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connected");
    snapshot_both(&conn, &m);
    recv_until(&mut h.bus_rx, |e| match e {
        Event::Book(b) if b.token_id == m.tokens.down => Some(()),
        _ => None,
    })
    .await;
    // A bid arriving AT the Up book's ask (0.52) = that ask was consumed.
    conn.frame_tx
        .send(change_frame(&m, &m.tokens.up, "BUY", "0.52", "5"))
        .expect("driver alive");
    let top = recv_until(&mut h.bus_rx, |e| match e {
        Event::TopOfBook { token_id, top } if *token_id.as_ref() == m.tokens.up => Some(*top),
        _ => None,
    })
    .await;
    assert_eq!(
        top.bid.map(|b| b.price.as_decimal()),
        Some(dec!(0.52)),
        "the aggressor rests as the new best bid"
    );
    assert!(top.ask.is_none(), "the consumed ask is gone");
    // Same connection (no reconnect), no health transitions; trust intact.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(
        attempt_at.lock().expect("no poison").len(),
        1,
        "trade at the touch must not recycle the connection"
    );
    while let Ok(event) = h.bus_rx.try_recv() {
        assert!(
            !matches!(event, Event::BookHealth(_)),
            "no health transition for a trade at the touch"
        );
    }
    let status = h.status_rx.borrow().clone();
    assert!(status.windows[0].machine.trusted);
    assert_eq!(status.windows[0].machine.consumed, 1);

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn active_silent_socket_recycles_at_staleness_threshold() {
    // An ACTIVE window whose connection never delivers a frame: the 10 s
    // book staleness watchdog recycles before the 15 s dead-socket check.
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let h = spawn_supervisor(vec![transport]);
    let m = market(9, 0);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let _conn1 = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connected");

    let times = wait_attempts(&attempt_at, 2).await;
    let gap = times[1] - times[0];
    assert!(
        gap >= Duration::from_secs(10) && gap < Duration::from_secs(12),
        "staleness recycle at ~10 s: {times:?}"
    );

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn preopen_silent_socket_dies_at_dead_socket_threshold() {
    // A PRE-OPEN window is staleness-exempt (quiet books are legitimate),
    // so a totally silent socket is torn down by the dead-socket check at
    // 3 × ping_interval = 15 s instead.
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let h = spawn_supervisor(vec![transport]);
    // Opens in 60 s — inside the 90 s lead, so it connects immediately but
    // stays PreOpen.
    let m = market(12, 60_000);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Discovered))
        .await
        .expect("supervisor alive");
    let _conn1 = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connected");

    let times = wait_attempts(&attempt_at, 2).await;
    let gap = times[1] - times[0];
    assert!(
        gap >= Duration::from_secs(15) && gap < Duration::from_secs(17),
        "dead socket at ~15 s: {times:?}"
    );

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn linger_cap_tears_down_without_resolution() {
    let (transport, mut handles, _) = script(&[true]);
    let h = spawn_supervisor(vec![transport]);
    let m = market(10, 0);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let conn = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connected");
    snapshot_both(&conn, &m);
    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Closed))
        .await
        .expect("supervisor alive");
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(h.status_rx.borrow().windows.len(), 1, "lingering");

    // No market_resolved ever arrives: the linger cap (180 s) tears down.
    tokio::time::sleep(Duration::from_secs(185)).await;
    assert_eq!(h.status_rx.borrow().windows.len(), 0, "linger cap enforced");

    shutdown_and_join(h).await;
}

#[tokio::test(start_paused = true)]
async fn dropped_bus_is_fatal() {
    let (transport, mut handles, _) = script(&[true]);
    let mut h = spawn_supervisor(vec![transport]);
    let m = market(11, 0);

    h.window_tx
        .send((Arc::clone(&m), WindowLifecycle::Open))
        .await
        .expect("supervisor alive");
    let conn = tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if let Some(conn) = handles.pop_front() {
                return conn;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("connected");

    drop(std::mem::replace(&mut h.bus_rx, mpsc::channel(1).1));
    snapshot_both(&conn, &m);

    let result = tokio::time::timeout(Duration::from_secs(600), h.supervisor)
        .await
        .expect("supervisor exits promptly")
        .expect("no panic");
    assert!(matches!(result, Err(FeedError::BusClosed)));
}
