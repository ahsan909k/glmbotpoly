//! Exponential backoff for discovery retries (CLAUDE.md §6): starts at the
//! configured initial delay, doubles per consecutive failure, saturates at
//! the cap, resets on success.

use core_types::DurationMs;

/// Doubling backoff with a cap. Pure state — the machine owns when to ask
/// for a delay and when to reset.
#[derive(Debug, Clone)]
pub(crate) struct Backoff {
    initial: DurationMs,
    max: DurationMs,
    next: DurationMs,
}

impl Backoff {
    pub(crate) fn new(initial: DurationMs, max: DurationMs) -> Self {
        Self {
            initial,
            max,
            next: initial,
        }
    }

    /// Returns the delay to wait before the next attempt, then doubles the
    /// stored delay (capped). With the defaults: 1s, 2s, 4s, 8s, 16s, 30s,
    /// 30s, …
    pub(crate) fn next_delay(&mut self) -> DurationMs {
        let delay = self.next;
        let doubled = DurationMs::from_millis(self.next.as_millis().saturating_mul(2));
        self.next = if doubled > self.max {
            self.max
        } else {
            doubled
        };
        delay
    }

    /// Back to the initial delay (call on success).
    pub(crate) fn reset(&mut self) {
        self.next = self.initial;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_and_saturates_at_cap() {
        let mut b = Backoff::new(
            DurationMs::from_millis(1_000),
            DurationMs::from_millis(30_000),
        );
        let delays: Vec<i64> = (0..8).map(|_| b.next_delay().as_millis()).collect();
        assert_eq!(
            delays,
            vec![1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000, 30_000]
        );
    }

    #[test]
    fn reset_restarts_the_sequence() {
        let mut b = Backoff::new(
            DurationMs::from_millis(1_000),
            DurationMs::from_millis(30_000),
        );
        let _ = b.next_delay();
        let _ = b.next_delay();
        b.reset();
        assert_eq!(b.next_delay().as_millis(), 1_000);
        assert_eq!(b.next_delay().as_millis(), 2_000);
    }
}
