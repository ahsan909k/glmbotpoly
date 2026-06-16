//! The machine-readable health rollup behind `GET /health`.
//!
//! Unauthenticated by design (an ops liveness probe behind the VPS): it reveals
//! only whether the server is up and whether feeds/breakers/model are nominal —
//! no balances, fills, or positions. The HTTP status is always `200`; the JSON
//! `status` field carries the signal so a monitor can distinguish "reachable but
//! degraded" from "unreachable".

use core_types::{Mode, ModelHealth};
use serde::Serialize;

use crate::state::DashboardData;

/// Overall server health.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Everything nominal.
    Ok,
    /// A breaker is tripped, a feed is stale, or the model/books are unreliable.
    Degraded,
    /// No trading mode is running.
    Down,
}

/// One mode's liveness summary.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModeHealth {
    /// Whether the mode's session is running.
    pub running: bool,
    /// Whether any global breaker holds all trading down.
    pub globally_halted: bool,
    /// Whether any breaker is currently tripped.
    pub any_breaker_tripped: bool,
    /// Whether the user-channel WebSocket is connected (live only).
    pub ws_connected: bool,
}

/// Both modes' summaries.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModesHealth {
    /// Paper namespace.
    pub paper: ModeHealth,
    /// Live namespace.
    pub live: ModeHealth,
}

/// Feed-staleness summary.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct FeedsHealth {
    /// Whether any feed stream is currently stale.
    pub any_stale: bool,
    /// How many feed streams are stale.
    pub stale_count: usize,
}

/// Model-health summary.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModelsHealth {
    /// Whether any asset's model is unreliable.
    pub any_unreliable: bool,
}

/// Book-health summary.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BooksHealth {
    /// Whether any window's books are unreliable.
    pub any_unreliable: bool,
}

/// The `GET /health` body.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct HealthReport {
    /// Overall status.
    pub status: HealthStatus,
    /// Server uptime (ms).
    pub uptime_ms: i64,
    /// Last observed server time (unix millis).
    pub server_time_ms: i64,
    /// Per-mode liveness.
    pub modes: ModesHealth,
    /// Feed staleness.
    pub feeds: FeedsHealth,
    /// Model health.
    pub models: ModelsHealth,
    /// Book health.
    pub books: BooksHealth,
}

fn mode_health(data: &DashboardData, mode: Mode) -> ModeHealth {
    let ms = data.mode(mode);
    let globally_halted = ms.risk_snapshot.as_ref().is_some_and(|r| r.globally_halted);
    let any_breaker_tripped =
        !ms.tripped.is_empty() || ms.risk_snapshot.as_ref().is_some_and(|r| r.any_tripped());
    ModeHealth {
        running: ms.running,
        globally_halted,
        any_breaker_tripped,
        ws_connected: ms.ws_connected,
    }
}

/// Builds the health rollup from the current state.
#[must_use]
pub(crate) fn health_report(data: &DashboardData) -> HealthReport {
    let paper = mode_health(data, Mode::Paper);
    let live = mode_health(data, Mode::Live);
    let stale_count = data.shared.feed_stale.len();
    let any_feed_stale = stale_count > 0;
    let any_model_unreliable = data
        .shared
        .model_health
        .values()
        .any(|ev| ev.health == ModelHealth::Unreliable);
    let any_book_unreliable = !data.shared.book_unreliable.is_empty();
    let any_running = paper.running || live.running;
    let any_breaker = paper.any_breaker_tripped || live.any_breaker_tripped;

    let status = if !any_running {
        HealthStatus::Down
    } else if any_breaker || any_feed_stale || any_model_unreliable || any_book_unreliable {
        HealthStatus::Degraded
    } else {
        HealthStatus::Ok
    };

    HealthReport {
        status,
        uptime_ms: data.last_now.as_millis() - data.server_started.as_millis(),
        server_time_ms: data.last_now.as_millis(),
        modes: ModesHealth { paper, live },
        feeds: FeedsHealth {
            any_stale: any_feed_stale,
            stale_count,
        },
        models: ModelsHealth {
            any_unreliable: any_model_unreliable,
        },
        books: BooksHealth {
            any_unreliable: any_book_unreliable,
        },
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use core_types::TimestampMs;

    #[test]
    fn status_is_down_when_nothing_running() {
        let data = DashboardData::new(TimestampMs::from_millis(0));
        assert_eq!(health_report(&data).status, HealthStatus::Down);
    }

    #[test]
    fn status_ok_when_running_and_clean() {
        let mut data = DashboardData::new(TimestampMs::from_millis(0));
        let _ = data.set_session(Mode::Paper, true, TimestampMs::from_millis(10));
        let report = health_report(&data);
        assert_eq!(report.status, HealthStatus::Ok);
        assert!(report.modes.paper.running);
        assert_eq!(report.uptime_ms, 10);
    }

    #[test]
    fn status_degraded_on_tripped_breaker() {
        use core_types::{BreakerKind, Event, RiskEvent};
        let mut data = DashboardData::new(TimestampMs::from_millis(0));
        let _ = data.set_session(Mode::Paper, true, TimestampMs::from_millis(1));
        let _ = data.project(
            Mode::Paper,
            &Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::Manual,
            }),
            TimestampMs::from_millis(2),
        );
        assert_eq!(health_report(&data).status, HealthStatus::Degraded);
        assert!(health_report(&data).modes.paper.any_breaker_tripped);
    }
}
