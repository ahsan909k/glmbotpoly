//! The transport seam: async-fn-in-trait (RPITIT, discovery precedent) so
//! driver tests run over a scripted fake with no network. The crate-local
//! [`WsFrame`] keeps tungstenite types out of the seam.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

/// One inbound frame. Pong/Close control frames are handled inside the
/// transport impl and never surface; server Pings DO surface (payload-free)
/// because they are liveness evidence — on Binance they are the only
/// guaranteed traffic on a data-quiet socket, and the driver's dead-socket
/// watchdog must count them. tungstenite answers them automatically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsFrame {
    /// Text frame (everything RTDS and Binance send is text).
    Text(String),
    /// Binary frame (never observed on either venue; parsed defensively as
    /// UTF-8).
    Binary(Vec<u8>),
    /// A server protocol ping. Liveness-only: tungstenite already queued the
    /// pong reply; the payload is irrelevant and dropped.
    Ping,
}

/// A transport-layer failure. Never fatal to the feed — the driver logs it
/// and reconnects forever.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct TransportError(pub String);

/// Dials connections. `&mut self` so fakes can script successive attempts.
pub trait Transport: Send {
    /// The connection type this transport yields.
    type Conn: Connection;

    /// Opens a connection, bounded by `timeout` (covers TCP+TLS+WS
    /// handshake).
    fn connect(
        &mut self,
        url: &str,
        timeout: Duration,
    ) -> impl Future<Output = Result<Self::Conn, TransportError>> + Send;
}

/// One live connection.
pub trait Connection: Send {
    /// Sends a text frame.
    fn send_text(&mut self, text: &str) -> impl Future<Output = Result<(), TransportError>> + Send;

    /// Receives the next application frame. `None` means the peer closed
    /// cleanly; `Some(Err(_))` is a socket error — both end the episode.
    fn recv(&mut self) -> impl Future<Output = Option<Result<WsFrame, TransportError>>> + Send;

    /// Best-effort close (errors ignored — we are leaving either way).
    fn close(&mut self) -> impl Future<Output = ()> + Send;
}

/// The real tokio-tungstenite transport (rustls, webpki roots — workspace
/// TLS policy).
#[derive(Debug, Clone, Copy, Default)]
pub struct WsTransport;

/// A live tungstenite socket.
pub struct WsConnection {
    socket: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
}

/// Pins the process-level rustls `CryptoProvider` to ring (the provider
/// reqwest uses) exactly once. Without it, a binary whose dependency graph
/// doesn't otherwise select a provider panics at the first TLS connect.
/// `install_default` failing means someone else installed one first — fine.
fn ensure_crypto_provider() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

impl Transport for WsTransport {
    type Conn = WsConnection;

    async fn connect(
        &mut self,
        url: &str,
        timeout: Duration,
    ) -> Result<WsConnection, TransportError> {
        ensure_crypto_provider();
        match tokio::time::timeout(timeout, tokio_tungstenite::connect_async(url)).await {
            Ok(Ok((socket, _response))) => Ok(WsConnection { socket }),
            Ok(Err(error)) => Err(TransportError(format!("connect failed: {error}"))),
            Err(_) => Err(TransportError(format!(
                "connect timed out after {timeout:?}"
            ))),
        }
    }
}

impl Connection for WsConnection {
    async fn send_text(&mut self, text: &str) -> Result<(), TransportError> {
        self.socket
            .send(Message::text(text))
            .await
            .map_err(|error| TransportError(format!("send failed: {error}")))
    }

    async fn recv(&mut self) -> Option<Result<WsFrame, TransportError>> {
        loop {
            match self.socket.next().await {
                Some(Ok(Message::Text(text))) => return Some(Ok(WsFrame::Text(text.to_string()))),
                Some(Ok(Message::Binary(bytes))) => {
                    return Some(Ok(WsFrame::Binary(bytes.to_vec())));
                }
                // tungstenite answers protocol pings internally on the next
                // read/write; the ping still surfaces as liveness evidence.
                Some(Ok(Message::Ping(_))) => return Some(Ok(WsFrame::Ping)),
                Some(Ok(Message::Pong(_) | Message::Frame(_))) => {}
                Some(Ok(Message::Close(_))) => return None,
                Some(Err(error)) => {
                    return Some(Err(TransportError(format!("socket error: {error}"))));
                }
                None => return None,
            }
        }
    }

    async fn close(&mut self) {
        let _ = self.socket.close(None).await;
    }
}

/// Direction of a tapped frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapDir {
    /// Server → us.
    In,
    /// Us → server.
    Out,
}

/// One tapped frame (timestamping is the consumer's job — the writer task
/// in the binary stamps on receipt).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TapFrame {
    /// Direction.
    pub dir: TapDir,
    /// Frame text (binary frames are tapped lossy-UTF-8).
    pub text: String,
}

/// Decorator that forwards every in/out frame to a channel — the `--raw`
/// fixture-capture path. Unbounded + best-effort so a slow consumer can
/// never stall the feed.
pub struct TapTransport<T> {
    inner: T,
    tap: mpsc::UnboundedSender<TapFrame>,
}

impl<T> TapTransport<T> {
    /// Wraps `inner`, mirroring all traffic to `tap`.
    pub const fn new(inner: T, tap: mpsc::UnboundedSender<TapFrame>) -> Self {
        Self { inner, tap }
    }
}

impl<T: Transport> Transport for TapTransport<T> {
    type Conn = TapConnection<T::Conn>;

    async fn connect(
        &mut self,
        url: &str,
        timeout: Duration,
    ) -> Result<Self::Conn, TransportError> {
        let conn = self.inner.connect(url, timeout).await?;
        Ok(TapConnection {
            inner: conn,
            tap: self.tap.clone(),
        })
    }
}

/// A tapped connection (see [`TapTransport`]).
pub struct TapConnection<C> {
    inner: C,
    tap: mpsc::UnboundedSender<TapFrame>,
}

impl<C: Connection> Connection for TapConnection<C> {
    async fn send_text(&mut self, text: &str) -> Result<(), TransportError> {
        let _ = self.tap.send(TapFrame {
            dir: TapDir::Out,
            text: text.to_owned(),
        });
        self.inner.send_text(text).await
    }

    async fn recv(&mut self) -> Option<Result<WsFrame, TransportError>> {
        let frame = self.inner.recv().await;
        if let Some(Ok(ref f)) = frame {
            // Protocol pings are transport noise, not application traffic —
            // they are not tapped (fixtures stay replayable text).
            let text = match f {
                WsFrame::Text(t) => Some(t.clone()),
                WsFrame::Binary(b) => Some(String::from_utf8_lossy(b).into_owned()),
                WsFrame::Ping => None,
            };
            if let Some(text) = text {
                let _ = self.tap.send(TapFrame {
                    dir: TapDir::In,
                    text,
                });
            }
        }
        frame
    }

    async fn close(&mut self) {
        self.inner.close().await;
    }
}
