//! Deterministic driver tests: paused tokio time over a scripted fake
//! transport (scheduler-driver precedent). Every reconnect/resubscribe/
//! staleness/backoff behavior is pinned here without touching the network.

// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used)]

use std::time::Duration;

use core_types::{Asset, DurationMs, Event, FeedHealth, PriceSource, TimestampMs};
use feed_rtds::{
    BackoffParams, FeedCommand, FeedError, FeedStatus, FeedSub, RtdsArgs, RtdsParams, RtdsSource,
    WsFrame, backfill_subscribe_message, run, stream_subscribe_message, stream_unsubscribe_message,
};
use feed_util::fake::{ConnHandle, FakeTransport, script};
use tokio::sync::{mpsc, watch};

/// One minute past an arbitrary epoch, mirroring the scheduler tests.
const BASE_MS: i64 = 1_800_000_060_000;

/// Wall clock locked to the (paused) tokio clock.
fn paused_now_fn() -> impl Fn() -> TimestampMs + Send {
    let start = tokio::time::Instant::now();
    move || {
        let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX);
        TimestampMs::from_millis(BASE_MS + elapsed)
    }
}

fn params() -> RtdsParams {
    RtdsParams {
        url: "wss://fake".to_owned(),
        ping_interval: Duration::from_secs(5),
        connect_timeout: Duration::from_secs(1),
        backoff: BackoffParams {
            initial: Duration::from_millis(250),
            max: Duration::from_millis(10_000),
            multiplier: 2.0,
        },
        stale_after: DurationMs::from_millis(5_000),
    }
}

struct Harness {
    driver: tokio::task::JoinHandle<Result<(), FeedError>>,
    bus_rx: mpsc::Receiver<Event>,
    shutdown_tx: watch::Sender<bool>,
    command_tx: Option<mpsc::Sender<FeedCommand>>,
    #[allow(dead_code)]
    status_rx: watch::Receiver<FeedStatus>,
}

fn spawn_driver(transport: FakeTransport, subs: Vec<FeedSub>, with_commands: bool) -> Harness {
    let (bus_tx, bus_rx) = mpsc::channel(256);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (status_tx, status_rx) = watch::channel(FeedStatus::default());
    let (command_tx, command_rx) = if with_commands {
        let (tx, rx) = mpsc::channel(8);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let driver = tokio::spawn(run(RtdsArgs {
        params: params(),
        subscriptions: subs,
        transport,
        now_fn: paused_now_fn(),
        bus_tx,
        command_rx,
        status_tx: Some(status_tx),
        shutdown_rx,
        backoff_seed: Some(42),
    }));
    Harness {
        driver,
        bus_rx,
        shutdown_tx,
        command_tx,
        status_rx,
    }
}

/// A live update frame (single payload object, realistic topics).
fn update_frame(source: RtdsSource, asset: Asset, ts: i64, value: &str) -> WsFrame {
    WsFrame::Text(format!(
        r#"{{"topic":"{}","type":"update","timestamp":{ts},"payload":{{"symbol":"{}","timestamp":{ts},"value":{value}}}}}"#,
        source.topic(),
        source.symbol(asset),
    ))
}

/// Receives bus events until `pred` returns Some, skipping everything else.
async fn recv_until<T>(
    bus_rx: &mut mpsc::Receiver<Event>,
    mut pred: impl FnMut(&Event) -> Option<T>,
) -> T {
    let deadline = tokio::time::Duration::from_secs(120);
    tokio::time::timeout(deadline, async {
        loop {
            let event = bus_rx.recv().await.expect("bus open");
            if let Some(value) = pred(&event) {
                return value;
            }
        }
    })
    .await
    .expect("expected event within (paused) 120s")
}

/// Reads the next `n` sent messages.
async fn sent(handle: &mut ConnHandle, n: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let message = tokio::time::timeout(Duration::from_secs(120), handle.sent_rx.recv())
            .await
            .expect("expected a sent message within (paused) 120s")
            .expect("connection alive");
        out.push(message);
    }
    out
}

fn connect_sequence_all() -> Vec<String> {
    vec![
        backfill_subscribe_message(RtdsSource::Binance, Asset::Btc),
        backfill_subscribe_message(RtdsSource::Binance, Asset::Eth),
        stream_subscribe_message(RtdsSource::Binance),
        backfill_subscribe_message(RtdsSource::Chainlink, Asset::Btc),
        backfill_subscribe_message(RtdsSource::Chainlink, Asset::Eth),
        stream_subscribe_message(RtdsSource::Chainlink),
    ]
}

#[tokio::test(start_paused = true)]
async fn connects_subscribes_streams_and_pings() {
    let (transport, mut handles, _) = script(&[true]);
    let mut h = spawn_driver(transport, FeedSub::all(), false);
    let mut conn = handles.pop_front().expect("one connection");

    // Exact connect sequence: per topic, filtered backfill subscribes then
    // the unfiltered steady-state subscribe.
    assert_eq!(sent(&mut conn, 6).await, connect_sequence_all());

    // Ack then a live tick.
    conn.frame_tx
        .send(WsFrame::Text(String::new()))
        .expect("driver alive");
    conn.frame_tx
        .send(update_frame(
            RtdsSource::Binance,
            Asset::Btc,
            BASE_MS,
            "63500.5",
        ))
        .expect("driver alive");
    let tick = recv_until(&mut h.bus_rx, |e| match e {
        Event::PriceTick(t) => Some(*t),
        _ => None,
    })
    .await;
    assert_eq!(tick.source, PriceSource::BinanceRtds);
    assert_eq!(tick.asset, Asset::Btc);
    assert_eq!(tick.value.to_string(), "63500.5");
    assert_eq!(tick.ts_exchange, TimestampMs::from_millis(BASE_MS));

    // Keepalive PINGs at the 5 s cadence, forever.
    assert_eq!(sent(&mut conn, 1).await, vec!["PING".to_owned()]);
    assert_eq!(sent(&mut conn, 1).await, vec!["PING".to_owned()]);

    h.shutdown_tx.send(true).expect("driver alive");
    let result = h.driver.await.expect("no panic");
    assert!(matches!(result, Ok(())));
}

#[tokio::test(start_paused = true)]
async fn disconnect_goes_stale_then_reconnect_resubscribes_and_recovers() {
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let mut h = spawn_driver(transport, FeedSub::all(), false);

    // Episode 1: all four streams deliver.
    let mut conn1 = handles.pop_front().expect("first connection");
    let _ = sent(&mut conn1, 6).await;
    for sub in FeedSub::all() {
        conn1
            .frame_tx
            .send(update_frame(sub.source, sub.asset, BASE_MS, "100.0"))
            .expect("driver alive");
    }
    for _ in 0..4 {
        recv_until(&mut h.bus_rx, |e| match e {
            Event::PriceTick(t) => Some(*t),
            _ => None,
        })
        .await;
    }

    // Kill the connection: every stream goes stale immediately (no
    // threshold wait), then the driver redials and resubscribes in full.
    drop(conn1.frame_tx);
    let mut stale = Vec::new();
    for _ in 0..4 {
        stale.push(
            recv_until(&mut h.bus_rx, |e| match e {
                Event::FeedHealth(FeedHealth::Stale { source, asset, .. }) => {
                    Some((*source, *asset))
                }
                _ => None,
            })
            .await,
        );
    }
    stale.sort_unstable_by_key(|(s, a)| (format!("{s:?}"), format!("{a:?}")));
    assert_eq!(stale.len(), 4);

    let mut conn2 = handles.pop_front().expect("second connection");
    assert_eq!(sent(&mut conn2, 6).await, connect_sequence_all());
    assert_eq!(attempt_at.lock().expect("no poison").len(), 2);

    // Streams recover individually on their first new tick.
    conn2
        .frame_tx
        .send(update_frame(
            RtdsSource::Chainlink,
            Asset::Btc,
            BASE_MS + 60_000,
            "101.0",
        ))
        .expect("driver alive");
    let recovered = recv_until(&mut h.bus_rx, |e| match e {
        Event::FeedHealth(FeedHealth::Recovered { source, asset, .. }) => Some((*source, *asset)),
        _ => None,
    })
    .await;
    assert_eq!(recovered, (PriceSource::ChainlinkRtds, Asset::Btc));

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn failed_connects_back_off_with_growing_jittered_delays() {
    let (transport, mut handles, attempt_at) = script(&[false, false, false, true]);
    let h = spawn_driver(
        transport,
        vec![FeedSub::new(RtdsSource::Binance, Asset::Btc)],
        false,
    );

    // The success proves we got through all three failures.
    let mut conn = handles.pop_front().expect("final connection");
    let _ = sent(&mut conn, 2).await;

    let times = attempt_at.lock().expect("no poison").clone();
    assert_eq!(times.len(), 4);
    let gaps: Vec<u128> = times
        .windows(2)
        .map(|w| (w[1] - w[0]).as_millis())
        .collect();
    // Equal jitter: gap ∈ [raw/2, raw], raw doubling 250 → 500 → 1000.
    assert!((125..=250).contains(&gaps[0]), "gap1 {gaps:?}");
    assert!((250..=500).contains(&gaps[1]), "gap2 {gaps:?}");
    assert!((500..=1000).contains(&gaps[2]), "gap3 {gaps:?}");

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn silent_socket_is_assumed_dead_and_redialed() {
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let h = spawn_driver(
        transport,
        vec![FeedSub::new(RtdsSource::Binance, Asset::Btc)],
        false,
    );

    // Episode 1: subscribed but the peer never sends a single frame (not
    // even an ack) — half-open socket. The frame sender stays alive so
    // recv() never errors; only the dead-socket watchdog can fire.
    let mut conn1 = handles.pop_front().expect("first connection");
    let _ = sent(&mut conn1, 2).await;

    // At 3 × ping interval (15 s) without frames the driver redials.
    let mut conn2 = handles.pop_front().expect("second connection");
    let _ = sent(&mut conn2, 2).await;
    let times = attempt_at.lock().expect("no poison").clone();
    assert_eq!(times.len(), 2);
    assert!(
        times[1] >= Duration::from_secs(15),
        "redialed only after the dead-socket window: {times:?}"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn one_dry_stream_goes_stale_while_others_stay_live() {
    let (transport, mut handles, _) = script(&[true]);
    let mut h = spawn_driver(transport, FeedSub::all(), false);
    let mut conn = handles.pop_front().expect("one connection");
    let _ = sent(&mut conn, 6).await;

    // Three streams tick every second; chainlink:eth never does.
    for i in 0..7_i64 {
        let ts = BASE_MS + i * 1_000;
        for sub in [
            FeedSub::new(RtdsSource::Binance, Asset::Btc),
            FeedSub::new(RtdsSource::Binance, Asset::Eth),
            FeedSub::new(RtdsSource::Chainlink, Asset::Btc),
        ] {
            conn.frame_tx
                .send(update_frame(sub.source, sub.asset, ts, "100.0"))
                .expect("driver alive");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    let stale = recv_until(&mut h.bus_rx, |e| match e {
        Event::FeedHealth(FeedHealth::Stale {
            source, asset, age, ..
        }) => Some((*source, *asset, *age)),
        _ => None,
    })
    .await;
    assert_eq!(
        (stale.0, stale.1),
        (PriceSource::ChainlinkRtds, Asset::Eth),
        "only the dry stream staled"
    );
    assert!(stale.2.as_millis() >= 5_000);

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn starved_stream_recycles_the_connection_to_resubscribe() {
    // The live-observed failure (2026-06-12): one stream's server-side
    // subscription decays while the rest stream on. Other traffic keeps the
    // dead-socket check happy; only the per-stream starvation watchdog can
    // recover, by recycling the connection at 6 × stale_after = 30 s.
    let (transport, mut handles, attempt_at) = script(&[true, true]);
    let h = spawn_driver(transport, FeedSub::all(), false);
    let mut conn1 = handles.pop_front().expect("first connection");
    let _ = sent(&mut conn1, 6).await;

    // Everything except chainlink:btc ticks every second, for 35 s.
    for i in 0..35_i64 {
        let ts = BASE_MS + i * 1_000;
        for sub in [
            FeedSub::new(RtdsSource::Binance, Asset::Btc),
            FeedSub::new(RtdsSource::Binance, Asset::Eth),
            FeedSub::new(RtdsSource::Chainlink, Asset::Eth),
        ] {
            conn1
                .frame_tx
                .send(update_frame(sub.source, sub.asset, ts, "100.0"))
                .expect("driver alive");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        if attempt_at.lock().expect("no poison").len() > 1 {
            break; // recycled — stop feeding the dead connection
        }
    }

    // The second dial happened, at ≥30 s, with the full resubscribe.
    let mut conn2 = handles.pop_front().expect("recycled connection");
    assert_eq!(sent(&mut conn2, 6).await, connect_sequence_all());
    let times = attempt_at.lock().expect("no poison").clone();
    assert_eq!(times.len(), 2);
    assert!(
        times[1] >= Duration::from_secs(30),
        "recycle waits the full 6×stale_after window: {times:?}"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn runtime_commands_change_subscriptions_without_reconnecting() {
    let (transport, mut handles, attempt_at) = script(&[true]);
    let mut h = spawn_driver(
        transport,
        vec![FeedSub::new(RtdsSource::Binance, Asset::Btc)],
        true,
    );
    let mut conn = handles.pop_front().expect("one connection");
    assert_eq!(
        sent(&mut conn, 2).await,
        vec![
            backfill_subscribe_message(RtdsSource::Binance, Asset::Btc),
            stream_subscribe_message(RtdsSource::Binance),
        ]
    );
    let command_tx = h.command_tx.take().expect("commands wired");

    // Runtime subscribe: backfill for the new stream + its topic's
    // steady-state subscribe.
    command_tx
        .send(FeedCommand::Subscribe(FeedSub::new(
            RtdsSource::Chainlink,
            Asset::Btc,
        )))
        .await
        .expect("driver alive");
    assert_eq!(
        sent(&mut conn, 2).await,
        vec![
            backfill_subscribe_message(RtdsSource::Chainlink, Asset::Btc),
            stream_subscribe_message(RtdsSource::Chainlink),
        ]
    );

    // Runtime unsubscribe of the topic's last stream: topic-level
    // unsubscribe goes out, and its stragglers stop publishing.
    command_tx
        .send(FeedCommand::Unsubscribe(FeedSub::new(
            RtdsSource::Binance,
            Asset::Btc,
        )))
        .await
        .expect("driver alive");
    assert_eq!(
        sent(&mut conn, 1).await,
        vec![stream_unsubscribe_message(RtdsSource::Binance)]
    );

    conn.frame_tx
        .send(update_frame(
            RtdsSource::Binance,
            Asset::Btc,
            BASE_MS,
            "1.0",
        ))
        .expect("driver alive");
    conn.frame_tx
        .send(update_frame(
            RtdsSource::Chainlink,
            Asset::Btc,
            BASE_MS,
            "2.0",
        ))
        .expect("driver alive");
    let tick = recv_until(&mut h.bus_rx, |e| match e {
        Event::PriceTick(t) => Some(*t),
        _ => None,
    })
    .await;
    assert_eq!(
        (tick.source, tick.asset),
        (PriceSource::ChainlinkRtds, Asset::Btc),
        "the unsubscribed stream's tick was dropped, the live one published"
    );
    assert_eq!(
        attempt_at.lock().expect("no poison").len(),
        1,
        "no reconnect"
    );

    h.shutdown_tx.send(true).expect("driver alive");
    assert!(matches!(h.driver.await.expect("no panic"), Ok(())));
}

#[tokio::test(start_paused = true)]
async fn malformed_frames_never_kill_the_stream() {
    let (transport, mut handles, attempt_at) = script(&[true]);
    let mut h = spawn_driver(
        transport,
        vec![FeedSub::new(RtdsSource::Binance, Asset::Btc)],
        false,
    );
    let mut conn = handles.pop_front().expect("one connection");
    let _ = sent(&mut conn, 2).await;

    for junk in [
        WsFrame::Text("not json at all".to_owned()),
        WsFrame::Text(r#"{"topic":"equity_prices","payload":{"symbol":"AAPL"}}"#.to_owned()),
        WsFrame::Text(r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt"}}"#.to_owned()),
        WsFrame::Text(
            r#"{"topic":"crypto_prices","payload":{"symbol":"solusdt","timestamp":1,"value":1.0}}"#
                .to_owned(),
        ),
        WsFrame::Binary(vec![0xFF, 0xFE, 0x00]),
        WsFrame::Text(String::new()),
        WsFrame::Text("PONG".to_owned()),
    ] {
        conn.frame_tx.send(junk).expect("driver alive");
    }
    conn.frame_tx
        .send(update_frame(
            RtdsSource::Binance,
            Asset::Btc,
            BASE_MS,
            "63500.5",
        ))
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
    let mut h = spawn_driver(
        transport,
        vec![FeedSub::new(RtdsSource::Binance, Asset::Btc)],
        false,
    );
    let mut conn = handles.pop_front().expect("one connection");
    let _ = sent(&mut conn, 2).await;

    drop(std::mem::replace(&mut h.bus_rx, mpsc::channel(1).1));
    conn.frame_tx
        .send(update_frame(
            RtdsSource::Binance,
            Asset::Btc,
            BASE_MS,
            "1.0",
        ))
        .expect("driver alive");

    let result = tokio::time::timeout(Duration::from_secs(120), h.driver)
        .await
        .expect("driver exits promptly")
        .expect("no panic");
    assert!(matches!(result, Err(FeedError::BusClosed)));
}

#[tokio::test(start_paused = true)]
async fn shutdown_during_backoff_exits_cleanly() {
    // Endless connect failures: the driver lives in the backoff loop.
    let (transport, _, attempt_at) = script(&[false, false, false, false, false, false]);
    let h = spawn_driver(
        transport,
        vec![FeedSub::new(RtdsSource::Binance, Asset::Btc)],
        false,
    );

    // Let a couple of attempts happen, then pull the plug mid-backoff.
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(attempt_at.lock().expect("no poison").len() >= 2);
    h.shutdown_tx.send(true).expect("driver alive");
    let result = tokio::time::timeout(Duration::from_secs(120), h.driver)
        .await
        .expect("driver exits promptly")
        .expect("no panic");
    assert!(matches!(result, Ok(())));
}
