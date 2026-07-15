//! The operator's window into the bot (CLAUDE.md §10): a token-protected axum
//! server providing REST snapshots and WebSocket push, served by the bot
//! process. This crate is the read/push **backend**: the REST endpoints
//! (overview, per-series comparison, active-window detail, fills, risk, params)
//! and the WebSocket are pure projections of in-memory state stores and
//! `analytics` queries — handlers hold no business logic. Both trading modes
//! (paper/live) are namespaced and simultaneously available.
//!
//! # Shape
//!
//! - [`DashboardHandle`] is the ingestion + broadcast API the orchestrator
//!   holds: `project(mode, event, now)` folds the bus into the namespaced
//!   stores and broadcasts the incremental [`WsUpdate`]s; the `set_*` setters
//!   carry state that does not ride the bus (wallet, ledger, risk snapshot,
//!   params, session/arm flags). It is cheaply `Clone`.
//! - [`serve`] / [`serve_with_listener`] start the axum server; [`router`]
//!   builds the router alone (for tests).
//! - The static single-page UI (served at `/`, `/app.css`, `/app.js`) and the
//!   §10.6/§11 control write-endpoints (`POST /api/control/*` — kill, reset,
//!   reset-daily-stop, paper-capital, enable/disable-series, set-param, and the
//!   multi-step live arm/confirm/disarm flow, plus `GET /api/control/status`)
//!   are served from here. The orchestrator opts in by attaching a request sink
//!   via [`DashboardHandle::with_request_sink`]; each [`ControlRequest`] carries
//!   a [`DashboardCommand`], its [`core_types::CommandOrigin`], and a one-shot
//!   reply for the [`ControlOutcome`] (the resulting [`ControlStateSnapshot`]),
//!   which the endpoint returns as the ack. The `config → params` boundary map
//!   remains deferred.

mod auth;
mod command;
mod dto;
mod error;
mod handle;
mod health;
mod live_markout;
mod model_taker;
mod server;
mod shadow;
mod state;
mod ui;
mod ws;

pub use command::{
    ArmingView, ControlOutcome, ControlRequest, ControlStateSnapshot, DashboardCommand,
    OutcomeKind, ParamOverrideView,
};
pub use error::DashboardError;
pub use handle::DashboardHandle;
pub use model_taker::ModelTakerTick;
pub use server::{router, serve, serve_with_listener};
pub use shadow::ShadowTick;
pub use state::{DriverStatus, FeedCadence, ParamsView};
pub use ws::{BreakerEvent, WsUpdate};
