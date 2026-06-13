//! Wall-clock instants and durations as plain unix milliseconds.
//!
//! Polymarket wire timestamps (RTDS payloads, book snapshots) are unix millis,
//! so `i64` millis is the crate-wide lingua franca. Clock discipline — reading
//! the current time, NTP offset checks, monotonic clocks — belongs to the
//! `timeutil` crate; nothing in `core-types` can observe "now".

use std::fmt;

use serde::{Deserialize, Serialize};

/// A wall-clock instant in unix milliseconds (UTC).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct TimestampMs(i64);

impl TimestampMs {
    /// Wraps a raw unix-milliseconds value.
    #[must_use]
    pub const fn from_millis(ms: i64) -> Self {
        Self(ms)
    }

    /// Returns the raw unix-milliseconds value.
    #[must_use]
    pub const fn as_millis(self) -> i64 {
        self.0
    }

    /// Adds a (possibly negative) duration, saturating at the `i64` limits.
    #[must_use]
    pub const fn saturating_add(self, d: DurationMs) -> Self {
        Self(self.0.saturating_add(d.as_millis()))
    }

    /// Signed elapsed time from `earlier` to `self` (negative if `earlier`
    /// is actually later — possible under clock skew; callers must handle it).
    #[must_use]
    pub const fn signed_duration_since(self, earlier: Self) -> DurationMs {
        DurationMs::from_millis(self.0.saturating_sub(earlier.0))
    }
}

impl fmt::Display for TimestampMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A signed span of time in milliseconds.
///
/// Signed on purpose: feed ages and clock deltas can be negative under skew,
/// and silently flooring them at zero would hide exactly the condition the
/// risk manager must alarm on.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct DurationMs(i64);

impl DurationMs {
    /// Zero-length duration.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw signed millisecond count.
    #[must_use]
    pub const fn from_millis(ms: i64) -> Self {
        Self(ms)
    }

    /// Builds a duration from whole seconds, saturating at the `i64` limits.
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs.saturating_mul(1000))
    }

    /// Returns the raw signed millisecond count.
    #[must_use]
    pub const fn as_millis(self) -> i64 {
        self.0
    }

    /// True when the span is negative (clock skew indicator).
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Adds two durations, saturating at the `i64` limits.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

impl fmt::Display for DurationMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ms", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_follows_millis() {
        let a = TimestampMs::from_millis(1_000);
        let b = TimestampMs::from_millis(2_000);
        assert!(a < b);
        assert_eq!(b.signed_duration_since(a), DurationMs::from_millis(1_000));
        assert_eq!(a.signed_duration_since(b), DurationMs::from_millis(-1_000));
        assert!(a.signed_duration_since(b).is_negative());
    }

    #[test]
    fn saturating_arithmetic_does_not_wrap() {
        let max = TimestampMs::from_millis(i64::MAX);
        assert_eq!(max.saturating_add(DurationMs::from_millis(1)), max);
        let min = TimestampMs::from_millis(i64::MIN);
        assert_eq!(min.saturating_add(DurationMs::from_millis(-1)), min);
        assert_eq!(
            DurationMs::from_secs(i64::MAX),
            DurationMs::from_millis(i64::MAX)
        );
    }

    #[test]
    fn from_secs_scales() {
        assert_eq!(DurationMs::from_secs(60), DurationMs::from_millis(60_000));
        assert_eq!(DurationMs::from_secs(-5), DurationMs::from_millis(-5_000));
    }

    #[test]
    fn serde_is_bare_integer() {
        let ts = TimestampMs::from_millis(1_718_000_000_000);
        assert_eq!(serde_json::to_string(&ts).unwrap(), "1718000000000");
        let back: TimestampMs = serde_json::from_str("1718000000000").unwrap();
        assert_eq!(back, ts);
        let d: DurationMs = serde_json::from_str("-250").unwrap();
        assert_eq!(d, DurationMs::from_millis(-250));
    }
}
