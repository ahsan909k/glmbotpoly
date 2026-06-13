//! The single clock abstraction: monotonic instants paired with UTC wall time.
//!
//! Everything below the binary that needs "now" takes a [`Clock`] (or, for
//! sans-IO cores, a plain `TimestampMs` parameter fed by one). Monotonic time
//! is for measuring intervals — it never jumps, never relates to wall time;
//! wall time is for venue-facing timestamps and window arithmetic. Mixing the
//! two is exactly the bug class this crate exists to prevent.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use core_types::{DurationMs, TimestampMs};

/// Paired monotonic + UTC wall clock.
///
/// Implementations must be cheap to clone and safe to share across tasks.
pub trait Clock: Send + Sync {
    /// Monotonic time elapsed since this clock was created. Never decreases
    /// and never relates to wall time — use it for interval measurement only
    /// (latency samples, §12 monotonic log stamps).
    fn mono(&self) -> Duration;

    /// Current UTC wall time in unix milliseconds.
    fn wall(&self) -> TimestampMs;
}

/// Infallible wall-clock read at the system boundary.
///
/// Clamps instead of erroring: a system clock set before the unix epoch reads
/// as 0 and one beyond `i64` milliseconds reads as `i64::MAX` — both are loud
/// downstream (the skew monitor alarms on them) without ever being a panic on
/// a runtime path (§12).
#[must_use]
pub fn wall_now() -> TimestampMs {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    TimestampMs::from_millis(ms)
}

/// Production [`Clock`]: `std::time::Instant` for monotonic time,
/// `SystemTime` (via [`wall_now`]) for wall time.
#[derive(Debug, Clone)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    /// Creates a clock whose monotonic origin is "now".
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn mono(&self) -> Duration {
        self.origin.elapsed()
    }

    fn wall(&self) -> TimestampMs {
        wall_now()
    }
}

#[derive(Debug)]
struct MockState {
    mono: Duration,
    wall_ms: i64,
}

/// Manually driven test clock. Clones share one underlying state, so a test
/// can hold a handle while the code under test holds another.
///
/// Public (not `cfg(test)`) on purpose: downstream crates' tests need it too.
#[derive(Debug, Clone)]
pub struct MockClock {
    inner: Arc<Mutex<MockState>>,
}

impl MockClock {
    /// Creates a mock clock at the given wall time with monotonic time zero.
    #[must_use]
    pub fn new(wall: TimestampMs) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockState {
                mono: Duration::ZERO,
                wall_ms: wall.as_millis(),
            })),
        }
    }

    /// Advances both clocks together — the healthy passage of time.
    pub fn advance(&self, d: Duration) {
        let mut state = self.lock();
        state.mono += d;
        let ms = i64::try_from(d.as_millis()).unwrap_or(i64::MAX);
        state.wall_ms = state.wall_ms.saturating_add(ms);
    }

    /// Steps ONLY the wall clock (positive or negative) — the skew injector.
    /// Monotonic time is unaffected, exactly like a real clock step.
    pub fn step_wall(&self, d: DurationMs) {
        let mut state = self.lock();
        state.wall_ms = state.wall_ms.saturating_add(d.as_millis());
    }

    /// Sets the wall clock outright. Monotonic time is unaffected.
    pub fn set_wall(&self, t: TimestampMs) {
        self.lock().wall_ms = t.as_millis();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MockState> {
        // A poisoned mock clock only means another test thread panicked while
        // holding the lock; the state itself (two integers) is always valid.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Clock for MockClock {
    fn mono(&self) -> Duration {
        self.lock().mono
    }

    fn wall(&self) -> TimestampMs {
        TimestampMs::from_millis(self.lock().wall_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_clock_is_sane() {
        let clock = SystemClock::new();
        let m1 = clock.mono();
        let w = clock.wall();
        let m2 = clock.mono();
        assert!(m2 >= m1, "monotonic time went backwards");
        // Wall time is after 2020-01-01 and before 2100-01-01 on any machine
        // that can run this test suite.
        assert!(w.as_millis() > 1_577_836_800_000);
        assert!(w.as_millis() < 4_102_444_800_000);
    }

    #[test]
    fn mock_advance_moves_both() {
        let clock = MockClock::new(TimestampMs::from_millis(1_000_000));
        clock.advance(Duration::from_millis(2_500));
        assert_eq!(clock.mono(), Duration::from_millis(2_500));
        assert_eq!(clock.wall(), TimestampMs::from_millis(1_002_500));
    }

    #[test]
    fn mock_step_wall_moves_wall_only() {
        let clock = MockClock::new(TimestampMs::from_millis(1_000_000));
        clock.advance(Duration::from_secs(1));
        clock.step_wall(DurationMs::from_millis(-300));
        assert_eq!(clock.mono(), Duration::from_secs(1));
        assert_eq!(clock.wall(), TimestampMs::from_millis(1_000_700));
        clock.set_wall(TimestampMs::from_millis(42));
        assert_eq!(clock.wall(), TimestampMs::from_millis(42));
        assert_eq!(clock.mono(), Duration::from_secs(1));
    }

    #[test]
    fn mock_clones_share_state() {
        let a = MockClock::new(TimestampMs::from_millis(0));
        let b = a.clone();
        a.advance(Duration::from_secs(5));
        assert_eq!(b.wall(), TimestampMs::from_millis(5_000));
        assert_eq!(b.mono(), Duration::from_secs(5));
    }
}
