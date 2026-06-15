//! UTC calendar-day bucketing for the daily roll-ups, derived purely from an
//! event timestamp.
//!
//! Deterministic and clock-free: a [`DayKey`] is the number of whole UTC days
//! since the unix epoch (`millis.div_euclid(86_400_000)`), which is exactly the
//! UTC midnight day boundary. `div_euclid` keeps it total for the (never-hit)
//! pre-epoch case. No `time` crate dependency — rendering `YYYY-MM-DD` for the
//! dashboard is deferred to the dashboard task.

use core_types::TimestampMs;
use serde::{Deserialize, Serialize};

/// Milliseconds in one UTC day.
const MS_PER_DAY: i64 = 86_400_000;

/// A UTC calendar day, as whole days since the unix epoch. Cheap `Copy` key for
/// the daily roll-up map and a stable, sortable dashboard identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DayKey(i64);

impl DayKey {
    /// The UTC day that contains `ts`.
    #[must_use]
    pub const fn from_ts(ts: TimestampMs) -> Self {
        Self(ts.as_millis().div_euclid(MS_PER_DAY))
    }

    /// Wraps a raw days-since-epoch value (for tests and dashboard queries).
    #[must_use]
    pub const fn from_days(days: i64) -> Self {
        Self(days)
    }

    /// Whole UTC days since the unix epoch.
    #[must_use]
    pub const fn as_days(self) -> i64 {
        self.0
    }

    /// Unix-millis instant of this day's UTC midnight (its start).
    #[must_use]
    pub const fn start_ms(self) -> TimestampMs {
        TimestampMs::from_millis(self.0 * MS_PER_DAY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_is_utc_midnight_bucket() {
        // 2026-06-15T00:00:00Z = 1_781_481_600_000 ms = day 20619.
        let midnight = TimestampMs::from_millis(1_781_481_600_000);
        let key = DayKey::from_ts(midnight);
        assert_eq!(key.as_days(), 20_619);
        assert_eq!(key.start_ms(), midnight);
    }

    #[test]
    fn instants_within_a_day_share_a_key() {
        let start = 1_781_481_600_000;
        let just_before_next = start + MS_PER_DAY - 1;
        assert_eq!(
            DayKey::from_ts(TimestampMs::from_millis(start)),
            DayKey::from_ts(TimestampMs::from_millis(just_before_next)),
        );
        // One ms later rolls to the next day.
        assert_ne!(
            DayKey::from_ts(TimestampMs::from_millis(start)),
            DayKey::from_ts(TimestampMs::from_millis(start + MS_PER_DAY)),
        );
    }

    #[test]
    fn pre_epoch_stays_total() {
        // -1 ms is the last ms of day -1, never panics or wraps oddly.
        assert_eq!(DayKey::from_ts(TimestampMs::from_millis(-1)).as_days(), -1);
        assert_eq!(DayKey::from_ts(TimestampMs::from_millis(0)).as_days(), 0);
    }
}
