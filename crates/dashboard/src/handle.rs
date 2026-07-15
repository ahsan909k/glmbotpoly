//! [`DashboardHandle`] — the ingestion and broadcast API.
//!
//! The orchestrator holds one handle and feeds it the bus: `project(mode,
//! event, now)` for every (mode, event) the mode should observe, plus the
//! `set_*` setters for state that does not ride the bus (wallet, ledger, risk
//! snapshot, params, session/arm flags). Each write locks the store briefly,
//! folds synchronously, releases the lock, then broadcasts the incremental
//! [`WsUpdate`]s to WebSocket subscribers. The handle is cheaply `Clone` (an
//! `Arc` to the store plus a broadcast sender), so the server task and the
//! ingestion loop share one store.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use core_types::{CommandOrigin, Event, Mode, TimestampMs};
use engine::RiskStateSnapshot;
use tokio::sync::{broadcast, mpsc, oneshot};
use venue_api::Wallet;
use venue_paper::PaperLedgerSnapshot;

use crate::command::{
    ControlError, ControlOutcome, ControlRequest, ControlStateSnapshot, DashboardCommand,
};
use crate::shadow::ShadowTick;
use crate::state::{DashboardData, FeedCadence, ParamsView};
use crate::ws::WsUpdate;

/// How long the endpoint waits for the orchestrator to reply to a control
/// command before reporting the control channel unavailable.
const CONTROL_REPLY_TIMEOUT: Duration = Duration::from_secs(2);

/// The shared, namespaced dashboard state plus the WebSocket broadcast channel
/// and an optional control-command egress to the orchestrator.
#[derive(Clone)]
pub struct DashboardHandle {
    data: Arc<Mutex<DashboardData>>,
    ws_tx: broadcast::Sender<Arc<WsUpdate>>,
    /// Where the control write-endpoints send operator commands (with a reply
    /// channel for the resulting state). `None` makes the dashboard read-only
    /// (the control endpoints answer `503`).
    request_tx: Option<mpsc::Sender<ControlRequest>>,
}

impl std::fmt::Debug for DashboardHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DashboardHandle")
            .field("subscribers", &self.ws_tx.receiver_count())
            .finish()
    }
}

impl DashboardHandle {
    /// A fresh handle. `broadcast_cap` bounds the per-subscriber WebSocket
    /// backlog (a slow client that lags past it gets a resync hint). `now`
    /// seeds the server-start time.
    #[must_use]
    pub fn new(broadcast_cap: usize, now: TimestampMs) -> Self {
        let (ws_tx, _) = broadcast::channel(broadcast_cap.max(1));
        Self {
            data: Arc::new(Mutex::new(DashboardData::new(now))),
            ws_tx,
            request_tx: None,
        }
    }

    /// Attaches a control-request sink, enabling the `POST /api/control/*`
    /// write-endpoints. Without it the handle is read-only and those endpoints
    /// answer `503`. The orchestrator holds the matching receiver and is the
    /// single owner that validates, applies, journals, and acknowledges the
    /// commands.
    #[must_use]
    pub fn with_request_sink(mut self, request_tx: mpsc::Sender<ControlRequest>) -> Self {
        self.request_tx = Some(request_tx);
        self
    }

    /// Routes an operator control command to the orchestrator and awaits the
    /// resulting [`ControlOutcome`] (with the resulting state). Never blocks the
    /// server: a missing sink, a full channel, a dropped reply, or a slow
    /// orchestrator all resolve to a [`ControlError`] within
    /// [`CONTROL_REPLY_TIMEOUT`].
    ///
    /// # Errors
    /// [`ControlError::Unavailable`] if the handle has no request sink (read-only),
    /// [`ControlError::Unsendable`] if the channel is full, the orchestrator is
    /// gone, or it did not reply in time.
    pub(crate) async fn request(
        &self,
        command: DashboardCommand,
        origin: CommandOrigin,
    ) -> Result<ControlOutcome, ControlError> {
        let tx = self.request_tx.as_ref().ok_or(ControlError::Unavailable)?;
        let (reply, rx) = oneshot::channel();
        tx.try_send(ControlRequest {
            command,
            origin,
            reply,
        })
        .map_err(|_| ControlError::Unsendable)?;
        match tokio::time::timeout(CONTROL_REPLY_TIMEOUT, rx).await {
            Ok(Ok(outcome)) => Ok(outcome),
            // The orchestrator dropped the reply, or did not answer in time.
            Ok(Err(_)) | Err(_) => Err(ControlError::Unsendable),
        }
    }

    /// Locks the store, recovering it if a previous holder panicked (the data is
    /// plain values, never left half-written across an `.await`).
    fn lock(&self) -> MutexGuard<'_, DashboardData> {
        self.data.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn broadcast(&self, updates: Vec<WsUpdate>) {
        for update in updates {
            // A full/empty channel is fine: no subscribers, or a lagging one
            // that will get a resync. Never blocks the ingestion loop.
            let _ = self.ws_tx.send(Arc::new(update));
        }
    }

    /// Folds one bus event into `mode`'s namespace and the shared view, then
    /// broadcasts the resulting incremental updates.
    pub fn project(&self, mode: Mode, event: &Event, now: TimestampMs) {
        let updates = self.lock().project(mode, event, now);
        self.broadcast(updates);
    }

    /// Samples a mode's wallet (the equity-curve source).
    pub fn set_wallet(&self, mode: Mode, wallet: Wallet, now: TimestampMs) {
        let updates = self.lock().set_wallet(mode, wallet, now);
        self.broadcast(updates);
    }

    /// Updates a mode's paper-ledger income lines.
    pub fn set_paper_ledger(&self, mode: Mode, ledger: PaperLedgerSnapshot, now: TimestampMs) {
        self.lock().set_paper_ledger(mode, ledger, now);
    }

    /// Pushes the richer risk-manager projection for a mode's risk panel.
    pub fn set_risk(&self, mode: Mode, snapshot: RiskStateSnapshot, now: TimestampMs) {
        self.lock().set_risk(mode, snapshot, now);
    }

    /// Records one shadow-model prediction for the "Shadow" tile (observation
    /// only). Fed by the shadow observer via a dashboard-owned [`ShadowTick`].
    pub fn set_shadow(&self, tick: &ShadowTick, now: TimestampMs) {
        let updates = self.lock().set_shadow(tick, now);
        self.broadcast(updates);
    }

    /// Records one model-taker decision for the "Model taker" tile (fires/day,
    /// last p_up per series). Polled via `GET /api/model-taker`.
    pub fn set_model_taker(&self, tick: &crate::ModelTakerTick, now: TimestampMs) {
        self.lock().set_model_taker(tick, now);
    }

    /// Buffers one driver-tagged fill for `mode`'s PnL-by-driver (marked at
    /// settlement). The orchestrator tags the fill via `RiskManager::driver_of`.
    pub fn record_driver_fill(
        &self,
        mode: Mode,
        driver: Option<engine::FillDriver>,
        fill: &core_types::Fill,
        now: TimestampMs,
    ) {
        self.lock().record_driver_fill(mode, driver, fill, now);
    }

    /// Records the four strategies' standing-down flags for `mode`'s STATUS strip
    /// (pushed from the orchestrator's risk-sample tick).
    pub fn set_driver_status(
        &self,
        mode: Mode,
        status: crate::state::DriverStatus,
        now: TimestampMs,
    ) {
        self.lock().set_driver_status(mode, status, now);
    }

    /// Folds a drained §10 contention snapshot (arbitration blocks + placement
    /// vetoes) into `mode`'s cumulative counters (pushed from the risk tick).
    pub fn set_contention(
        &self,
        mode: Mode,
        snapshot: &engine::ContentionSnapshot,
        now: TimestampMs,
    ) {
        self.lock().set_contention(mode, snapshot, now);
    }

    /// Sets the displayed parameters view.
    pub fn set_params(&self, params: ParamsView, now: TimestampMs) {
        self.lock().set_params(params, now);
    }

    /// Records the latest feed-cadence + loop-health snapshot (Binance-Mid
    /// inter-update gap histogram + decision lag), pushed from the sample tick.
    /// Served by `GET /api/feed-cadence`.
    pub fn set_feed_cadence(&self, cadence: FeedCadence, now: TimestampMs) {
        self.lock().set_feed_cadence(cadence, now);
    }

    /// Pushes the current control-plane state snapshot (read by
    /// `GET /api/control/status`). The orchestrator calls this after each
    /// command and once at startup.
    pub fn set_control_state(&self, snapshot: ControlStateSnapshot, now: TimestampMs) {
        self.lock().set_control_state(snapshot, now);
    }

    /// Marks a mode's session running (or stopped).
    pub fn set_session(&self, mode: Mode, running: bool, now: TimestampMs) {
        let updates = self.lock().set_session(mode, running, now);
        self.broadcast(updates);
    }

    /// Marks a mode's live-arming state.
    pub fn set_armed(&self, mode: Mode, armed: bool, now: TimestampMs) {
        let updates = self.lock().set_armed(mode, armed, now);
        self.broadcast(updates);
    }

    /// Records a mode's user-channel WebSocket connectivity (live only).
    pub fn set_ws_connected(&self, mode: Mode, connected: bool, now: TimestampMs) {
        self.lock().set_ws_connected(mode, connected, now);
    }

    /// Subscribes a new WebSocket client to the broadcast stream.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<Arc<WsUpdate>> {
        self.ws_tx.subscribe()
    }

    /// Reads the store under the lock and returns an owned projection.
    pub(crate) fn with_data<R>(&self, f: impl FnOnce(&DashboardData) -> R) -> R {
        f(&self.lock())
    }

    /// The last observed server time (unix millis), for the WS handshake.
    pub(crate) fn server_time_ms(&self) -> i64 {
        self.lock().last_now.as_millis()
    }
}
