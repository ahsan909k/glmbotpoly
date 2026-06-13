//! Scripted fake transport for paused-time driver tests (feature `fake`,
//! pulled as a dev-dependency by the feed crates). Each scripted connection
//! attempt either yields a connection the test controls — inbound frames
//! pushed over a channel, outbound sends observable — or a connect failure;
//! when the script runs out, `connect` hangs (the driver sits in its connect
//! race until shutdown).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::transport::{Connection, Transport, TransportError, WsFrame};

/// Scripted connection attempts.
pub struct FakeTransport {
    attempts: VecDeque<Option<FakeConn>>,
    attempt_at: Arc<Mutex<Vec<Duration>>>,
    urls: Arc<Mutex<Vec<String>>>,
    started: tokio::time::Instant,
}

impl FakeTransport {
    /// Handle to the URLs passed to `connect`, for pinning URL-based
    /// subscription (clone it before moving the transport into the driver).
    #[must_use]
    pub fn url_log(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.urls)
    }
}

/// The driver's side of one scripted connection.
pub struct FakeConn {
    frame_rx: mpsc::UnboundedReceiver<WsFrame>,
    sent_tx: mpsc::UnboundedSender<String>,
}

/// The test's side of one scripted connection.
pub struct ConnHandle {
    /// Push inbound frames to the driver; dropping it ends the connection
    /// (the driver sees a clean peer close).
    pub frame_tx: mpsc::UnboundedSender<WsFrame>,
    /// Observe everything the driver sent.
    pub sent_rx: mpsc::UnboundedReceiver<String>,
}

/// Builds a transport scripted with `ok_attempts` (true = a connection,
/// false = a connect failure) plus the per-connection test handles and the
/// recorded attempt times (against the paused tokio clock).
#[must_use]
pub fn script(
    ok_attempts: &[bool],
) -> (
    FakeTransport,
    VecDeque<ConnHandle>,
    Arc<Mutex<Vec<Duration>>>,
) {
    let mut attempts = VecDeque::new();
    let mut handles = VecDeque::new();
    for &ok in ok_attempts {
        if ok {
            let (frame_tx, frame_rx) = mpsc::unbounded_channel();
            let (sent_tx, sent_rx) = mpsc::unbounded_channel();
            attempts.push_back(Some(FakeConn { frame_rx, sent_tx }));
            handles.push_back(ConnHandle { frame_tx, sent_rx });
        } else {
            attempts.push_back(None);
        }
    }
    let attempt_at = Arc::new(Mutex::new(Vec::new()));
    let transport = FakeTransport {
        attempts,
        attempt_at: Arc::clone(&attempt_at),
        urls: Arc::new(Mutex::new(Vec::new())),
        started: tokio::time::Instant::now(),
    };
    (transport, handles, attempt_at)
}

impl Transport for FakeTransport {
    type Conn = FakeConn;

    async fn connect(&mut self, url: &str, _timeout: Duration) -> Result<FakeConn, TransportError> {
        if let Ok(mut times) = self.attempt_at.lock() {
            times.push(self.started.elapsed());
        }
        if let Ok(mut urls) = self.urls.lock() {
            urls.push(url.to_owned());
        }
        match self.attempts.pop_front() {
            Some(Some(conn)) => Ok(conn),
            Some(None) => Err(TransportError("scripted connect failure".to_owned())),
            None => std::future::pending().await,
        }
    }
}

impl Connection for FakeConn {
    async fn send_text(&mut self, text: &str) -> Result<(), TransportError> {
        self.sent_tx
            .send(text.to_owned())
            .map_err(|_| TransportError("test dropped sent_rx".to_owned()))
    }

    async fn recv(&mut self) -> Option<Result<WsFrame, TransportError>> {
        self.frame_rx.recv().await.map(Ok)
    }

    async fn close(&mut self) {}
}
