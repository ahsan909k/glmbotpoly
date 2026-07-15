//! The WebSocket push protocol: the [`WsUpdate`] wire schema and the
//! per-connection task that forwards broadcast updates to one client.
//!
//! The projector ([`DashboardHandle::project`](crate::DashboardHandle::project))
//! folds each bus event and emits the incremental [`WsUpdate`]s it implies onto a
//! `tokio::sync::broadcast` channel; every connected client runs one
//! [`ws_connection`] task that serializes each update to a JSON text frame. The
//! five required incremental updates are `equity`, `quote` (our resting order),
//! `fill`, `breaker`, and `lifecycle`; `top`/`model` carry shared market data for
//! the live-window view, and `control` carries session-badge changes.
//!
//! `mode`-tagged variants are per-trading-mode (paper/live); the market-data
//! variants (`top`, `lifecycle`, `model`) carry no mode because the feeds are
//! shared across modes — they apply to the active window regardless of namespace.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use core_types::{
    Asset, BreakerKind, Decimal, Dollars, Mode, OrderUpdate, Outcome, TopOfBook, WindowLifecycle,
};
use serde::Serialize;
use tokio::sync::broadcast;

/// Which kind of breaker transition a [`WsUpdate::Breaker`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerEvent {
    /// A breaker fired.
    Tripped,
    /// A breaker reset.
    Cleared,
    /// A cancel-all was issued on a safety path.
    CancelAll,
}

/// One incremental update pushed to WebSocket clients. Internally tagged by a
/// `type` field; every data-carrying variant includes a `ts_ms` receive time.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsUpdate {
    /// The handshake frame sent once on connect.
    Hello {
        /// Server time (last observed) in unix millis.
        server_time_ms: i64,
        /// The namespaces available (`paper`, `live`).
        modes: Vec<Mode>,
        /// Protocol version.
        protocol: u32,
    },
    /// A mode's equity changed (sampled from the venue wallet).
    Equity {
        /// Which trading mode.
        mode: Mode,
        /// Receive time.
        ts_ms: i64,
        /// The new equity (collateral total).
        equity: Dollars,
    },
    /// One of our resting quote orders changed state (placed/filled/cancelled).
    Quote {
        /// Which trading mode.
        mode: Mode,
        /// Receive time.
        ts_ms: i64,
        /// The order update.
        order: OrderUpdate,
    },
    /// Market top-of-book changed for an active window token (shared).
    Top {
        /// Receive time.
        ts_ms: i64,
        /// The window key (`Series@open_ms`).
        window: String,
        /// Which outcome token.
        outcome: Outcome,
        /// The new top of book.
        top: TopOfBook,
    },
    /// One of our orders executed (a fill).
    Fill {
        /// Which trading mode.
        mode: Mode,
        /// Receive time.
        ts_ms: i64,
        /// The fill.
        fill: core_types::Fill,
    },
    /// A breaker state change (shown on the risk panel).
    Breaker {
        /// Which trading mode.
        mode: Mode,
        /// Receive time.
        ts_ms: i64,
        /// What kind of transition.
        event: BreakerEvent,
        /// Which breaker.
        breaker: BreakerKind,
    },
    /// A window's lifecycle phase changed (shared).
    Lifecycle {
        /// Receive time.
        ts_ms: i64,
        /// The window key (`Series@open_ms`).
        window: String,
        /// The new lifecycle phase.
        lifecycle: WindowLifecycle,
        /// The winning outcome, when resolved.
        outcome: Option<Outcome>,
    },
    /// New model output for an active window (shared): fair vs book mid.
    Model {
        /// Receive time.
        ts_ms: i64,
        /// The asset.
        asset: Asset,
        /// The window key, when window-tagged.
        window: Option<String>,
        /// Model probability that the window resolves Up.
        p_up: f64,
        /// Standardized log-moneyness.
        z: f64,
        /// One-second log-return volatility.
        sigma_1s: f64,
        /// Book midpoint of the Up token, when known.
        book_mid: Option<Decimal>,
    },
    /// A mode's session badges changed (running / armed / capital).
    Control {
        /// Which trading mode.
        mode: Mode,
        /// Receive time.
        ts_ms: i64,
        /// Whether the mode's session is running.
        running: bool,
        /// Whether live is armed.
        armed: bool,
    },
    /// A shadow-model prediction landed (observation only — informs the tile).
    Shadow {
        /// Receive time.
        ts_ms: i64,
        /// Series key.
        series: String,
        /// Model probability of Up.
        p_up: f64,
        /// How many of the 24 features were finite.
        finite_count: u32,
    },
    /// The client's broadcast receiver lagged and dropped updates — it should
    /// re-fetch the REST snapshots to resynchronize.
    Resync,
}

impl WsUpdate {
    /// Builds the handshake frame.
    #[must_use]
    pub fn hello(server_time_ms: i64) -> Self {
        Self::Hello {
            server_time_ms,
            modes: vec![Mode::Paper, Mode::Live],
            protocol: 1,
        }
    }
}

/// Serializes `value` and sends it as a text frame. Returns `Err(())` if the
/// socket is gone (the caller ends the connection); a serialization failure is
/// skipped (it cannot happen for our owned-data updates).
async fn send_json<T: Serialize>(socket: &mut WebSocket, value: &T) -> Result<(), ()> {
    match serde_json::to_string(value) {
        Ok(text) => socket
            .send(Message::Text(text.into()))
            .await
            .map_err(|_| ()),
        Err(err) => {
            tracing::warn!(target: "dashboard::ws", %err, "dropping un-serializable update");
            Ok(())
        }
    }
}

/// Drives one WebSocket client: sends `hello`, then forwards every broadcast
/// [`WsUpdate`] as a JSON text frame until the client disconnects or the channel
/// closes. On a `Lagged` overflow it sends a [`WsUpdate::Resync`] hint and keeps
/// going. Inbound client frames (pings, text) are ignored; a close ends the task.
pub(crate) async fn ws_connection(
    mut socket: WebSocket,
    mut rx: broadcast::Receiver<Arc<WsUpdate>>,
    hello: WsUpdate,
) {
    if send_json(&mut socket, &hello).await.is_err() {
        return;
    }
    loop {
        tokio::select! {
            received = rx.recv() => match received {
                Ok(update) => {
                    if send_json(&mut socket, update.as_ref()).await.is_err() {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::debug!(target: "dashboard::ws", dropped = n, "client lagged; sending resync");
                    if send_json(&mut socket, &WsUpdate::Resync).await.is_err() {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => return,
            },
            inbound = socket.recv() => match inbound {
                // Close frame or stream end — the client went away.
                Some(Ok(Message::Close(_))) | None => return,
                // Ignore pings/pongs/text/binary from the client (read-only push).
                Some(Ok(_)) => {}
                Some(Err(_)) => return,
            },
        }
    }
}
