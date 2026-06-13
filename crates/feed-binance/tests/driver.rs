//! Deterministic driver tests: paused tokio time over the shared scripted
//! fake transport (feed-rtds precedent). Pins the Binance-specific driver
//! behavior: URL-based subscription with zero outbound frames, per-stream-
//! kind staleness thresholds, server-ping liveness, starvation recycle, and
//! the reconnect/recovery paths.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use core_types::{Asset, DurationMs, Event, FeedHealth, PriceSource, TickKind, TimestampMs};
use feed_binance::{BackoffParams, BinanceArgs, BinanceParams, BinanceSub, FeedError, run};
use feed_util::WsFrame;
use feed_util::fake::{FakeTransport, script};
use tokio::sync::{mpsc, watch};

/// One minute past an arbitrary epoch, mirroring the feed-rtds tests.
const BASE_MS: i64 = 1_800_000_060_000;

/// Wall clock locked to the (paused) tokio clock.
fn paused_now_fn() -> impl Fn() -> TimestampMs + Send {
    let start = tokio::time::Instant::now();
    move || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
        TimestampMs::from_millis(BASE_MS + elapsed)
    }
}

/// Default test params: book streams stale at 2.5 s (recycle 15 s), trades
/// at 30 s (recycle 180 s).
fn params() -> BinanceParams {
    BinanceParams {
        url: "wss://fake".to_owned(),
        connect_timeout: Duration::from_secs(1),
        backoff: BackoffParams {
            initial: Duration::from_millis(250),
            max: Duration::from_millis(10_000),
            multiplier: 2.0,
        },
        stale_after: DurationMs::from_millis(2_500),
        trade_stale_after: DurationMs::from_millis(30_000),
    }
}

/// Loose staleness (30 s everywhere → recycle 180 s) so the 60-second
/// dead-socket check is the binding constraint.
fn loose_params() -> BinanceParams {
    BinanceParams {
        stale_after: DurationMs::from_millis(30_000),
        ..params()
    }
}

struct Harness {
    driver: tokio::task::JoinHandle<Result<(), FeedError>>,
    bus_rx: mpsc::Receiver<Event>,
    shutdown_tx: watch::Sender<bool>,
}

fn spawn_driver(transport: FakeTransport, p: BinanceParams) -> Harness {
    let (bus_tx, bus_rx) = mpsc::channel(256);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let driver = tokio::spawn(run(BinanceArgs {
        params: p,
        subscriptions: BinanceSub::all(),
        transport,
        now_fn: paused_now_fn(),
        bus_tx,
        status_tx: None,
        shutdown_rx,
        backoff_seed: Some(42),
    }));
    Harness {
        driver,
        bus_rx,
        shutdown_tx,
    }
}

fn book_frame(asset: Asset, bid: &str, ask: &str) -> WsFrame {
    let symbol = BinanceSub::symbol(asset);
    WsFrame::Text(format!(
        r#"{{"stream":"{symbol}@bookTicker","data":{{"u":1,"s":"{}","b":"{bid}","B":"1.0","a":"{ask}","A":"1.0"}}}}"#,
        symbol.to_uppercase(),
    ))
}

fn trade_frame(asset: Asset, price: &str, trade_ms: i64) -> WsFrame {
    let symbol = BinanceSub::symbol(asset);
    WsFrame::Text(format!(
        r#"{{"stream":"{symbol}@trade","data":{{"e":"trade","E":{trade_ms},"s":"{}","t":1,"p":"{price}","q":"1","T":{trade_ms},"m":true,"M":true}}}}"#,
        symbol.to_uppercase(),
    ))
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

#[tokio::test(start_paused = true)]
async fn connects_via_url_sends_nothing_and_publishes_tagged_ticks() {
    let (transport, mut handles, _) = script(&[true]);
    let urls = transport.url_log();
    let mut h = spawn_driver(transport, params());
    let mut conn = handles.pop_front().expect("one connection");

    // A bookTicker frame publishes the midpoint, kind Mid, stamped at the
    // local receive time (bookTicker carries no event time).
    conn.frame_tx
        .send(book_frame(Asset::Btc, "63500.01", "63500.03"))
        .expect("driver alive");
    let mid = recv_until(&mut h.bus_rx, |e| match e {
        Event::PriceTick(t) => Some(*t),
        _ => None,
    })
    .await;
    assert_eq!(mid.source, PriceSource::BinanceDirect);
    assert_eq!(mid.asset, Asset::Btc);
    assert_eq!(mid.kind, TickKind::Mid);
    assert_eq!(mid.value.to_string(), "63500.02");
    assert_eq!(mid.ts_exchange, mid.ts_local, "no event time on the wire");

    // A trade frame publishes the print, kind Trade, at the trade time.
    conn.frame_tx
        .send(trade_frame(Asset::Eth, "1670.39", BASE_MS - 50))
        .expect("driver alive");
    let trade = recv_until(&mut h.bus_rx, |e| match e {
        Event::PriceTick(t) => Some(*t),
        _ => None,
    })
    .await;
    assert_eq!(trade.asset, Asset::Eth);
    assert_eq!(trade.kind, TickKind::Trade);
    assert_eq!(trade.value.to_string(), "1670.39");
    assert_eq!(trade.ts_exchange, TimestampMs::from_millis(BASE_MS - 50));
    assert!(trade.ts_local >= TimestampMs::from_millis(BASE_MS));

    // Subscription rode the URL...
    assert_eq!(
        urls.lock().expect("no poison").as_slice(),
        [
            "wss://fake/stream?streams=btcusdt@bookTicker/btcusdt@trade/ethusdt@bookTicker/ethusdt@trade"
        ]
    );
    // ...and the driver says nothing at all over 20 s of streaming (no
    // subscribes, no keepalive — the venue's 5-messages/s budget is
    // untouchable by construction). Books keep ticking so staleness/recycle
    // stay away.
    for _ in 0..20 {
        conn.frame_tx
            .send(book_frame(Asset::Btc, "63500.01", "63500.03"))
            .expect("driver alive");
        conn.frame_tx
            .send(book_frame(Asset::Eth, "1670.01", "1670.03"))
            .expect("driver alive");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    assert!(
        matches!(
            conn.sent_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ),
        "the driver must never send a frame"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn disconnect_marks_all_four_stale_and_reconnect_recovers_individually() {
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let mut h = spawn_driver(transport, params());

    // Episode 1: all four streams deliver.
    let conn1 = handles.pop_front().expect("first connection");
    for asset in Asset::ALL {
        conn1
            .frame_tx
            .send(book_frame(asset, "100.0", "100.2"))
            .expect("driver alive");
        conn1
            .frame_tx
            .send(trade_frame(asset, "100.1", BASE_MS))
            .expect("driver alive");
    }
    for _ in 0..4 {
        recv_until(&mut h.bus_rx, |e| match e {
            Event::PriceTick(t) => Some(*t),
            _ => None,
        })
        .await;
    }

    // Kill the connection: all four streams go stale immediately, with their
    // kinds attached.
    drop(conn1.frame_tx);
    let mut stale = Vec::new();
    for _ in 0..4 {
        stale.push(
            recv_until(&mut h.bus_rx, |e| match e {
                Event::FeedHealth(FeedHealth::Stale {
                    source,
                    asset,
                    kind,
                    ..
                }) => Some((*source, *asset, *kind)),
                _ => None,
            })
            .await,
        );
    }
    stale.sort_unstable_by_key(|(s, a, k)| format!("{s:?}{a:?}{k:?}"));
    assert_eq!(stale.len(), 4);
    assert!(
        stale
            .iter()
            .all(|(source, _, _)| *source == PriceSource::BinanceDirect)
    );
    assert_eq!(
        stale.iter().filter(|(_, _, k)| *k == TickKind::Mid).count(),
        2
    );
    assert_eq!(
        stale
            .iter()
            .filter(|(_, _, k)| *k == TickKind::Trade)
            .count(),
        2
    );

    // The redial resubscribes via the URL alone; one stream recovers on its
    // first new tick.
    let conn2 = handles.pop_front().expect("second connection");
    let times = wait_attempts(&attempt_at, 2).await;
    assert_eq!(times.len(), 2);
    conn2
        .frame_tx
        .send(book_frame(Asset::Eth, "1670.01", "1670.03"))
        .expect("driver alive");
    let recovered = recv_until(&mut h.bus_rx, |e| match e {
        Event::FeedHealth(FeedHealth::Recovered {
            source,
            asset,
            kind,
            ..
        }) => Some((*source, *asset, *kind)),
        _ => None,
    })
    .await;
    assert_eq!(
        recovered,
        (PriceSource::BinanceDirect, Asset::Eth, TickKind::Mid)
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn totally_silent_socket_recycles_at_the_book_starvation_threshold() {
    // No stream ever delivers, but the socket stays open: the book streams'
    // starvation watchdog (6 × 2.5 s = 15 s) recycles the connection well
    // before the 60 s dead-socket check.
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let h = spawn_driver(transport, params());
    let _conn1 = handles.pop_front().expect("first connection");

    let times = wait_attempts(&attempt_at, 2).await;
    assert!(
        times[1] >= Duration::from_secs(15) && times[1] < Duration::from_secs(20),
        "recycle at the 6×stale_after starvation window, not the dead-socket window: {times:?}"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn server_pings_defer_the_dead_socket_check() {
    // Loose staleness (recycle at 180 s) so the 60 s dead-socket check is
    // binding. Server pings every 20 s — and nothing else — keep the socket
    // alive; once they stop, the driver redials 60 s after the last one.
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let h = spawn_driver(transport, loose_params());
    let conn1 = handles.pop_front().expect("first connection");

    for _ in 0..5 {
        conn1.frame_tx.send(WsFrame::Ping).expect("driver alive");
        tokio::time::sleep(Duration::from_secs(20)).await;
    }
    // t = 100 s: a frameless socket would have died at 60 s; pings held it.
    assert_eq!(
        attempt_at.lock().expect("no poison").len(),
        1,
        "server pings count as socket liveness"
    );

    // Pings stop (last at t = 80 s): dead socket declared at t = 140 s.
    let times = wait_attempts(&attempt_at, 2).await;
    assert!(
        times[1] >= Duration::from_secs(140) && times[1] < Duration::from_secs(150),
        "redial 60 s after the last ping: {times:?}"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn one_starved_book_stream_recycles_while_quiet_trades_never_alarm() {
    // The live RTDS failure mode transplanted: one stream's server-side
    // subscription decays while the rest stream on. Both trade streams stay
    // completely quiet the whole time — their looser threshold (30 s) must
    // produce no Stale events and no recycle; the starved BTC book stream
    // (2.5 s) drives the recycle at 15 s.
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let mut h = spawn_driver(transport, params());
    let conn1 = handles.pop_front().expect("first connection");

    // Trade-kind Stale events are counted only over the pre-recycle window
    // (t < 14 s): the recycle's own disconnect rightly marks every stream —
    // trades included — stale, which is not a threshold alarm.
    let mut trade_stale = 0_u32;
    for i in 0..18 {
        conn1
            .frame_tx
            .send(book_frame(Asset::Eth, "1670.01", "1670.03"))
            .expect("driver alive");
        tokio::time::sleep(Duration::from_secs(1)).await;
        if i < 13 {
            while let Ok(event) = h.bus_rx.try_recv() {
                if let Event::FeedHealth(FeedHealth::Stale {
                    kind: TickKind::Trade,
                    ..
                }) = event
                {
                    trade_stale += 1;
                }
            }
        }
        if attempt_at.lock().expect("no poison").len() > 1 {
            break;
        }
    }

    let times = attempt_at.lock().expect("no poison").clone();
    assert_eq!(times.len(), 2, "the starved book stream forced a recycle");
    assert!(
        times[1] >= Duration::from_secs(15),
        "recycle waits the full 6×stale_after window: {times:?}"
    );
    assert_eq!(
        trade_stale, 0,
        "quiet trade streams are healthy, not stale — per-kind thresholds"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn malformed_frames_never_kill_the_stream() {
    let (transport, mut handles, attempt_at) = script(&[true]);
    let mut h = spawn_driver(transport, params());
    let conn = handles.pop_front().expect("one connection");

    for junk in [
        WsFrame::Text("not json at all".to_owned()),
        WsFrame::Text(r#"{"stream":"solusdt@trade","data":{"e":"trade","s":"SOLUSDT","p":"1","T":1}}"#.to_owned()),
        WsFrame::Text(r#"{"code":2,"msg":"Invalid request"}"#.to_owned()),
        WsFrame::Text(r#"{"result":null,"id":1}"#.to_owned()),
        WsFrame::Text(String::new()),
        WsFrame::Binary(vec![0xFF, 0xFE, 0x00]),
        WsFrame::Text(r#"{"stream":"btcusdt@bookTicker","data":{"u":1,"s":"BTCUSDT","b":"0","B":"0","a":"0","A":"0"}}"#.to_owned()),
    ] {
        conn.frame_tx.send(junk).expect("driver alive");
    }
    conn.frame_tx
        .send(book_frame(Asset::Btc, "63500.01", "63500.03"))
        .expect("driver alive");

    let tick = recv_until(&mut h.bus_rx, |e| match e {
        Event::PriceTick(t) => Some(*t),
        _ => None,
    })
    .await;
    assert_eq!(tick.asset, Asset::Btc);
    assert_eq!(
        attempt_at.lock().expect("no poison").len(),
        1,
        "no reconnect"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn dropped_bus_is_a_fatal_bus_closed_error() {
    let (transport, mut handles, _) = script(&[true]);
    let mut h = spawn_driver(transport, params());
    let conn = handles.pop_front().expect("one connection");

    drop(std::mem::replace(&mut h.bus_rx, mpsc::channel(1).1));
    conn.frame_tx
        .send(book_frame(Asset::Btc, "1.0", "1.2"))
        .expect("driver alive");

    let result = tokio::time::timeout(Duration::from_secs(600), h.driver)
        .await
        .expect("driver exits promptly")
        .expect("no panic");
    assert!(matches!(result, Err(FeedError::BusClosed)));
}

#[tokio::test(start_paused = true)]
async fn shutdown_during_backoff_exits_cleanly() {
    // Endless connect failures: the driver lives in the backoff loop.
    let (transport, _, attempt_at) = script(&[false, false, false, false, false, false]);
    let h = spawn_driver(transport, params());

    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(attempt_at.lock().expect("no poison").len() >= 2);
    h.shutdown_tx.send(true).expect("driver alive");
    let result = tokio::time::timeout(Duration::from_secs(600), h.driver)
        .await
        .expect("driver exits promptly")
        .expect("no panic");
    assert!(matches!(result, Ok(())));
}
