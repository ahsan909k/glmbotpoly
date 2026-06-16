//! Dashboard server error type.

use std::net::SocketAddr;

use thiserror::Error;

/// Why the dashboard server could not be started or stopped serving.
#[derive(Debug, Error)]
pub enum DashboardError {
    /// The bind address could not be bound (port in use, permission denied).
    #[error("binding dashboard to {addr}: {source}")]
    Bind {
        /// The address we tried to bind.
        addr: SocketAddr,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// The server loop failed while accepting connections.
    #[error("serving dashboard: {0}")]
    Serve(#[source] std::io::Error),
    /// A non-loopback bind was requested without an auth token configured. The
    /// dashboard refuses to expose state to a network without authentication —
    /// the same invariant `config::validate` enforces, re-checked here so the
    /// crate fails closed on its own (CLAUDE.md §10/§11).
    #[error(
        "refusing to serve dashboard on non-loopback {0} without an auth token \
         (set BOT_SECRET_DASHBOARD_TOKEN)"
    )]
    NonLoopbackWithoutToken(SocketAddr),
}
