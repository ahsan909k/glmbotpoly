//! Jittered exponential reconnect backoff.
//!
//! Equal-jitter policy: each attempt sleeps `raw/2 + U[0, raw/2]` where
//! `raw = min(max, initial × multiplier^attempt)` — the `raw/2` floor
//! prevents near-zero thundering reconnects, the `raw` ceiling preserves the
//! configured curve. Randomness comes from a tiny xorshift64* PRNG rather
//! than the `rand` crate (not on the CLAUDE.md §3 allowlist; statistical
//! quality requirements here are trivial). The seed is injectable so driver
//! tests are deterministic.

use std::time::Duration;

/// Reconnect backoff curve parameters (mapped from `feeds.*` config by the
/// binary — this crate depends only on core-types).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackoffParams {
    /// First-attempt raw delay.
    pub initial: Duration,
    /// Raw-delay ceiling.
    pub max: Duration,
    /// Growth factor per failed attempt (> 1.0; validated in config).
    pub multiplier: f64,
}

/// xorshift64* — 8 lines, passes BigCrush's basic batteries, more than
/// enough to decorrelate reconnect storms.
#[derive(Debug, Clone)]
pub(crate) struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seeds the generator; `| 1` guards the all-zero fixed point.
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Stateful backoff: call [`Backoff::next_delay`] before each reconnect
/// attempt, [`Backoff::reset`] once the connection proves healthy again.
///
/// Public because feed crates with a connection lifecycle the generic
/// [`crate::run`] driver cannot express (feed-clob's per-window connections)
/// hand-roll their reconnect loop but must share the equal-jitter policy.
#[derive(Debug, Clone)]
pub struct Backoff {
    params: BackoffParams,
    rng: XorShift64,
    attempt: u32,
}

impl Backoff {
    /// New backoff at the start of the curve; `seed` makes tests deterministic.
    #[must_use]
    pub const fn new(params: BackoffParams, seed: u64) -> Self {
        Self {
            params,
            rng: XorShift64::new(seed),
            attempt: 0,
        }
    }

    /// The jittered delay for the next attempt; advances the curve.
    pub fn next_delay(&mut self) -> Duration {
        let initial_ms = duration_to_ms(self.params.initial);
        let max_ms = duration_to_ms(self.params.max);
        let factor = self
            .params
            .multiplier
            .powi(i32::try_from(self.attempt).unwrap_or(i32::MAX));
        let raw_ms = if factor.is_finite() {
            (initial_ms * factor).min(max_ms)
        } else {
            max_ms
        };
        self.attempt = self.attempt.saturating_add(1);

        // Equal jitter on whole milliseconds: raw/2 + U[0, raw/2]. Modulo
        // bias is irrelevant at this precision.
        let raw = raw_ms.max(0.0).min(u64::MAX as f64) as u64;
        let half = raw / 2;
        let jitter = if half == 0 {
            0
        } else {
            self.rng.next_u64() % (half + 1)
        };
        Duration::from_millis(half + jitter)
    }

    /// Back to the start of the curve (call once the link proves healthy —
    /// the driver waits for a data frame, not a bare connect).
    pub const fn reset(&mut self) {
        self.attempt = 0;
    }
}

fn duration_to_ms(d: Duration) -> f64 {
    // f64 holds every realistic config value exactly enough; the curve is
    // clamped to `max` anyway.
    d.as_millis().min(u128::from(u64::MAX)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: BackoffParams = BackoffParams {
        initial: Duration::from_millis(250),
        max: Duration::from_millis(10_000),
        multiplier: 2.0,
    };

    #[test]
    fn delays_stay_within_equal_jitter_bounds_and_cap() {
        let mut backoff = Backoff::new(PARAMS, 42);
        let mut raw = 250.0_f64;
        for _ in 0..12 {
            let delay = backoff.next_delay().as_millis() as f64;
            let capped = raw.min(10_000.0);
            assert!(
                delay >= (capped / 2.0).floor() && delay <= capped,
                "delay {delay} outside [{}, {capped}]",
                capped / 2.0
            );
            raw *= 2.0;
        }
        // Deep into the curve the raw delay is pinned at max.
        for _ in 0..20 {
            let delay = backoff.next_delay().as_millis();
            assert!((5_000..=10_000).contains(&delay));
        }
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut backoff = Backoff::new(PARAMS, 7);
        for _ in 0..6 {
            let _ = backoff.next_delay();
        }
        backoff.reset();
        let delay = backoff.next_delay().as_millis();
        assert!((125..=250).contains(&delay), "got {delay}");
    }

    #[test]
    fn same_seed_is_deterministic_different_seed_diverges() {
        let series = |seed: u64| -> Vec<u64> {
            let mut b = Backoff::new(PARAMS, seed);
            (0..8).map(|_| b.next_delay().as_millis() as u64).collect()
        };
        assert_eq!(series(1234), series(1234));
        assert_ne!(series(1), series(2));
    }

    #[test]
    fn zero_seed_does_not_stick_at_zero() {
        let mut rng = XorShift64::new(0);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, 0);
        assert_ne!(a, b);
    }
}
