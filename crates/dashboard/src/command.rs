//! Control commands the dashboard backend issues to the orchestrator, and the
//! typed outcome it sends back (CLAUDE.md §10.6/§11).
//!
//! The dashboard crate stays venue-agnostic for control: the write-endpoints
//! (`POST /api/control/*`) translate an operator action into a typed
//! [`DashboardCommand`], wrap it in a [`ControlRequest`] with the issuing
//! [`CommandOrigin`] and a one-shot reply channel, and hand it to the
//! orchestrator over an mpsc channel held by the
//! [`DashboardHandle`](crate::DashboardHandle). The orchestrator (the bot's
//! `dashboard` run loop) is the single owner that validates the command, applies
//! it to the running pipeline, journals it, and replies with a
//! [`ControlOutcome`] carrying the resulting [`ControlStateSnapshot`]. The
//! endpoint awaits that reply and maps the [`OutcomeKind`] to an HTTP status. A
//! handle built without a request sink — the default, and what the read-only
//! endpoint tests use — reports [`ControlError::Unavailable`] so the endpoint
//! answers `503`.

use core_types::{CommandOrigin, Decimal, Dollars};
use serde::Serialize;
use tokio::sync::oneshot;

/// An operator control action routed from a dashboard write-endpoint to the
/// orchestrator. Series keys and parameter values arrive as **strings** at the
/// HTTP boundary; the orchestrator (which owns config + runtime state) parses
/// and validates them, so the dashboard crate carries no trading semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum DashboardCommand {
    /// Global kill (CLAUDE.md §11): cancel everything and halt all trading,
    /// latched until a [`Reset`](Self::Reset).
    Kill,
    /// Clear the operator-latched breakers (manual kill + daily stop) so
    /// trading can resume.
    Reset,
    /// Clear only the latched daily-stop breaker, leaving a manual kill in
    /// place.
    ResetDailyStop,
    /// Set paper starting capital to an absolute amount.
    SetPaperCapital(Dollars),
    /// Adjust paper capital by a signed delta.
    AdjustPaperCapital(Decimal),
    /// Enable a series for trading at runtime (by its key, e.g. `"BTC-5m"`).
    EnableSeries(String),
    /// Disable a series at runtime.
    DisableSeries(String),
    /// Adjust a safe-listed runtime parameter. `series = None` applies it
    /// globally (all series); `Some(key)` targets one series.
    SetParam {
        /// Target series key, or `None` for all series.
        series: Option<String>,
        /// The safe-list parameter name.
        key: String,
        /// The new value as a string (parsed/validated by the orchestrator).
        value: String,
    },
    /// Begin the multi-step live-arming flow (§11). Refused unless the config
    /// and environment gates already pass; on success a confirmation is pending.
    ArmLiveBegin,
    /// Confirm a pending live-arming by re-typing the confirmation phrase.
    ArmLiveConfirm {
        /// The operator-typed confirmation phrase.
        phrase: String,
    },
    /// Disarm live trading for this session.
    Disarm,
}

/// A command plus its origin and a one-shot reply channel for the resulting
/// [`ControlOutcome`]. The orchestrator (the bot's run loop) receives these off
/// the request channel, processes each, and sends the outcome back over `reply`.
pub struct ControlRequest {
    /// The command to apply.
    pub command: DashboardCommand,
    /// Which frontend issued it (the audit-trail origin).
    pub origin: CommandOrigin,
    /// Where the orchestrator sends the resulting outcome.
    pub reply: oneshot::Sender<ControlOutcome>,
}

/// How a control command resolved. Maps to an HTTP status at the endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    /// The command was validated and applied (`200`).
    Accepted,
    /// The command was rejected for a bad/out-of-range argument (`400`).
    Rejected,
    /// The command conflicts with the current state — e.g. arming with a gate
    /// missing, or confirming with nothing pending (`409`).
    Conflict,
}

/// The result of a control command: how it resolved, an optional error message,
/// and the resulting control-plane state for the acknowledgment.
#[derive(Debug, Clone, Serialize)]
pub struct ControlOutcome {
    /// How the command resolved.
    pub kind: OutcomeKind,
    /// A human-readable reason when the command was not accepted.
    pub error: Option<String>,
    /// The resulting control-plane state (always present — the ack carries it
    /// whether the command was accepted or refused).
    pub state: ControlStateSnapshot,
}

impl ControlOutcome {
    /// Whether the command was accepted.
    #[must_use]
    pub fn accepted(&self) -> bool {
        matches!(self.kind, OutcomeKind::Accepted)
    }
}

/// A serializable view of the live-arming gates for the prominent state display
/// (§11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArmingView {
    /// Gate 1: `live.enabled` is set in config.
    pub config_enabled: bool,
    /// Gate 2: the `BOT_SECRET_LIVE_CONFIRM` environment phrase matches.
    pub env_confirmed: bool,
    /// Gate 3: live is armed for this session (the operator completed the flow).
    pub session_armed: bool,
    /// Whether `arm-live` would be accepted right now (gates 1 and 2 pass).
    pub can_arm: bool,
    /// Whether a confirmation is currently pending (begun, not yet confirmed).
    pub pending: bool,
}

/// One runtime parameter override currently in effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParamOverrideView {
    /// Target series key, or `None` for a global override.
    pub series: Option<String>,
    /// The safe-list parameter name.
    pub key: String,
    /// The current override value.
    pub value: String,
}

/// The control-plane state returned with every acknowledgment and by
/// `GET /api/control/status` — the resulting state of the most recent command.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControlStateSnapshot {
    /// Whether trading is globally halted (manual kill).
    pub halted: bool,
    /// The series enabled for trading, by key.
    pub enabled_series: Vec<String>,
    /// The live-arming gate state.
    pub arming: ArmingView,
    /// The runtime parameter overrides currently in effect.
    pub param_overrides: Vec<ParamOverrideView>,
    /// The current paper capital, when known.
    pub paper_capital: Option<Dollars>,
}

/// Why a control command could not be delivered to the orchestrator. Either way
/// the write-endpoint maps it to `503 Service Unavailable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlError {
    /// The handle has no request sink (a read-only deployment).
    Unavailable,
    /// The request channel is full/closed, or the orchestrator dropped the
    /// reply (gone/lagging).
    Unsendable,
}
