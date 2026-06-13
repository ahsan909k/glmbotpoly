//! Read-only scheduler state snapshots, published on a `watch` channel for
//! the smoke subcommand today and the dashboard's risk/coverage panels later.

use core_types::{Series, TimestampMs, WindowId};
use serde::Serialize;

/// One snapshot of every series' coverage state at one instant.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SchedulerStatus {
    /// The wall-clock instant the snapshot was taken.
    pub at: TimestampMs,
    /// Per-series state, in the driver's series order.
    pub series: Vec<SeriesStatus>,
}

/// Coverage state of one series.
#[derive(Debug, Clone, Serialize)]
pub struct SeriesStatus {
    /// Which series.
    pub series: Series,
    /// Machine phase: `discovering` | `pending` | `active` | `closing`.
    pub phase: &'static str,
    /// The window currently open (active/closing phases only).
    pub current: Option<WindowId>,
    /// Gamma event slug of the current window.
    pub current_slug: Option<String>,
    /// Milliseconds until the current window closes (negative = overdue).
    pub closes_in_ms: Option<i64>,
    /// Whether the next window is already discovered and announced.
    pub next_known: bool,
    /// Milliseconds until the next window opens (negative = already open,
    /// adoption pending).
    pub next_opens_in_ms: Option<i64>,
    /// Closed windows still awaiting `market_resolved`.
    pub parked: usize,
    /// Milliseconds since the last successful discovery refresh.
    pub refresh_age_ms: Option<i64>,
    /// `false` once the §6 contract is in violation: a current window inside
    /// its final `next_window_lead` with no known successor.
    pub contract_ok: bool,
}
