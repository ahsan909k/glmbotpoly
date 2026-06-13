//! Time discipline for a latency-sensitive trader: monotonic vs wall-clock
//! separation, an NTP offset check that can raise the clock-skew alarm
//! (CLAUDE.md §11), helpers for computing tau (seconds remaining in a window)
//! used throughout the model and engine, and the latency measurement harness
//! the operator uses to benchmark VPS regions and to calibrate the paper
//! venue's simulated placement/cancel round-trips.
//!
//! Layout (scheduler-style layering throughout):
//! - [`Clock`] / [`SystemClock`] / [`MockClock`] — the single clock seam.
//! - [`tau_secs`] / [`remaining_ms`] — window-time helpers.
//! - [`sntp`] — hand-rolled SNTPv4 (no NTP crate is §3-allowlisted).
//! - [`OffsetSource`] / [`NtpOffsetSource`] — per-round offset aggregation.
//! - [`SkewMonitor`] (sans-IO) + [`run_skew_monitor`] (driver) — the §11
//!   clock-skew breaker.
//! - [`harness`] (cargo feature `harness`) — the `bot latency` measurement
//!   harness, gated so engine-path consumers never build reqwest/tungstenite.

mod clock;
mod monitor;
mod offset;
mod skew;
pub mod sntp;
mod stats;
mod tau;

#[cfg(feature = "harness")]
pub mod harness;

pub use clock::{Clock, MockClock, SystemClock, wall_now};
pub use monitor::{SkewMonitorArgs, SkewMonitorError, run_skew_monitor};
pub use offset::{NtpOffsetSource, NtpParams, OffsetMeasurement, OffsetSource};
pub use skew::{SkewMonitor, SkewOutput, SkewParams, SkewWarning};
pub use stats::{LatencyStats, percentile_nearest_rank, stats_from_ms};
pub use tau::{remaining_ms, tau_secs};
