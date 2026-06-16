//! WebSocket stream test: a real `axum::serve` instance, a real client, and a
//! scripted event sequence that must arrive as the matching ordered update
//! frames. Also covers query-token auth on the upgrade.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::{OPEN_MS, TOKEN, fill_event, order_event, ts, wallet, window_event};
use core_types::{BreakerKind, Event, Mode, Outcome, RiskEvent, WindowLifecycle};
use dashboard::DashboardHandle;
use futures_util::StreamExt;
use rust_decimal::dec;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Reads the next text frame and parses it as JSON (skipping pings/pongs).
async fn next_json(ws: &mut Client) -> serde_json::Value {
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("frame within timeout")
            .expect("stream not closed")
            .expect("no ws error");
        match frame {
            Message::Text(text) => return serde_json::from_str(text.as_str()).expect("valid json"),
            Message::Ping(_) | Message::Pong(_) => {}
            other => panic!("unexpected ws frame: {other:?}"),
        }
    }
}

/// Binds an ephemeral loopback listener and spawns the dashboard server on it,
/// returning the port, the shared handle, a shutdown trigger, and the join.
async fn spawn_server(
    handle: DashboardHandle,
    token: Option<String>,
) -> (
    u16,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = dashboard::serve_with_listener(handle, listener, token, async move {
            let _ = shutdown_rx.await;
        })
        .await;
    });
    (port, shutdown_tx, task)
}

#[tokio::test(flavor = "multi_thread")]
async fn scripted_sequence_arrives_in_order() {
    let handle = DashboardHandle::new(64, ts(OPEN_MS));
    let (port, shutdown, server) = spawn_server(handle.clone(), Some(TOKEN.to_owned())).await;

    let url = format!("ws://127.0.0.1:{port}/api/ws?token={TOKEN}");
    let (mut ws, _resp) = connect_async(url).await.expect("ws connect");

    // 1. Handshake.
    let hello = next_json(&mut ws).await;
    assert_eq!(hello["type"], "hello");
    assert!(hello["modes"].as_array().unwrap().contains(&"paper".into()));

    // 2. lifecycle (Window Open).
    handle.project(
        Mode::Paper,
        &window_event(WindowLifecycle::Open),
        ts(OPEN_MS),
    );
    let f = next_json(&mut ws).await;
    assert_eq!(f["type"], "lifecycle");
    assert_eq!(f["lifecycle"], "Open");

    // 3. quote (one of our resting orders).
    handle.project(
        Mode::Paper,
        &order_event("q-1", core_types::OrderState::Open),
        ts(OPEN_MS + 1),
    );
    let f = next_json(&mut ws).await;
    assert_eq!(f["type"], "quote");
    assert_eq!(f["mode"], "paper");

    // 4. equity (wallet sample).
    handle.set_wallet(Mode::Paper, wallet(dec!(10000)), ts(OPEN_MS + 2));
    let f = next_json(&mut ws).await;
    assert_eq!(f["type"], "equity");
    assert_eq!(f["equity"], "10000");

    // 5. fill.
    handle.project(
        Mode::Paper,
        &fill_event(Outcome::Up, dec!(0.48), dec!(100), OPEN_MS + 3),
        ts(OPEN_MS + 3),
    );
    let f = next_json(&mut ws).await;
    assert_eq!(f["type"], "fill");

    // 6. breaker change.
    handle.project(
        Mode::Paper,
        &Event::Risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::Manual,
        }),
        ts(OPEN_MS + 4),
    );
    let f = next_json(&mut ws).await;
    assert_eq!(f["type"], "breaker");
    assert_eq!(f["event"], "tripped");
    assert_eq!(f["breaker"], "Manual");

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test(flavor = "multi_thread")]
async fn ws_upgrade_enforces_query_token() {
    let handle = DashboardHandle::new(8, ts(OPEN_MS));
    let (port, shutdown, server) = spawn_server(handle, Some(TOKEN.to_owned())).await;

    // Wrong token → the handler returns 401 before upgrading → connect fails.
    let wrong = format!("ws://127.0.0.1:{port}/api/ws?token=wrong");
    assert!(connect_async(wrong).await.is_err());

    // No token → also refused.
    let none = format!("ws://127.0.0.1:{port}/api/ws");
    assert!(connect_async(none).await.is_err());

    // Correct token → upgrade succeeds.
    let ok = format!("ws://127.0.0.1:{port}/api/ws?token={TOKEN}");
    assert!(connect_async(ok).await.is_ok());

    let _ = shutdown.send(());
    let _ = server.await;
}
