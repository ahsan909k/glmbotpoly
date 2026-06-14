//! Deterministic simulated-latency sampling.
//!
//! A tiny xorshift64* drives a uniform draw in `[mean, mean + jitter]` (the
//! one-sided-upward `LatencySpec`, §9). The same generator pattern is used by
//! `feed-util`'s backoff, but its `XorShift64` is `pub(crate)` there, so a
//! fresh ~15-line copy lives here. Seeded explicitly so tests are reproducible
//! and the binary can seed from the clock.

use core_types::DurationMs;

use crate::params::LatencySpec;

/// Fixed seed used when [`PaperParams::rng_seed`](crate::PaperParams::rng_seed)
/// is `None`. A non-zero constant (xorshift's only forbidden state is all-zero).
const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Minimal xorshift64* PRNG. Deterministic given its seed.
#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seeds the generator. The all-zero state is the one fixed point, so the
    /// seed is forced odd.
    const fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Draws simulated latencies uniformly in `[mean, mean + jitter]`.
#[derive(Debug, Clone)]
pub struct LatencySampler {
    rng: XorShift64,
}

impl LatencySampler {
    /// A sampler with an explicit seed (reproducible).
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: XorShift64::new(seed),
        }
    }

    /// A sampler seeded from `seed` if `Some`, else the fixed default seed.
    #[must_use]
    pub fn from_optional_seed(seed: Option<u64>) -> Self {
        Self::with_seed(seed.unwrap_or(DEFAULT_SEED))
    }

    /// Samples a latency uniformly in `[mean_ms, mean_ms + jitter_ms]`.
    ///
    /// Negative configured values are floored at zero defensively;
    /// `jitter_ms == 0` returns exactly `mean_ms`. The result is never negative.
    #[must_use]
    pub fn sample(&mut self, spec: &LatencySpec) -> DurationMs {
        let mean = spec.mean_ms.as_millis().max(0);
        let jitter = spec.jitter_ms.as_millis().max(0);
        let extra = if jitter == 0 {
            0
        } else {
            // jitter >= 1 here, so the modulus is >= 2 and never zero.
            let span = u64::try_from(jitter).unwrap_or(0).saturating_add(1);
            i64::try_from(self.rng.next_u64() % span).unwrap_or(0)
        };
        DurationMs::from_millis(mean.saturating_add(extra))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(mean: i64, jitter: i64) -> LatencySpec {
        LatencySpec {
            mean_ms: DurationMs::from_millis(mean),
            jitter_ms: DurationMs::from_millis(jitter),
        }
    }

    #[test]
    fn zero_jitter_is_exactly_mean() {
        let mut s = LatencySampler::with_seed(1);
        for _ in 0..1000 {
            assert_eq!(s.sample(&spec(150, 0)), DurationMs::from_millis(150));
        }
    }

    #[test]
    fn samples_stay_in_range_and_are_one_sided_up() {
        let mut s = LatencySampler::with_seed(42);
        let mut saw_above_mean = false;
        for _ in 0..10_000 {
            let v = s.sample(&spec(150, 100)).as_millis();
            assert!((150..=250).contains(&v), "{v} out of [150, 250]");
            if v > 150 {
                saw_above_mean = true;
            }
        }
        assert!(saw_above_mean, "jitter never engaged");
    }

    #[test]
    fn reproducible_for_a_seed() {
        let draws = |seed| {
            let mut s = LatencySampler::with_seed(seed);
            (0..20).map(|_| s.sample(&spec(10, 50))).collect::<Vec<_>>()
        };
        assert_eq!(draws(7), draws(7));
        assert_ne!(draws(7), draws(8));
    }

    #[test]
    fn never_negative_even_with_garbage_spec() {
        let mut s = LatencySampler::with_seed(3);
        let v = s.sample(&spec(-100, -50));
        assert_eq!(v, DurationMs::ZERO);
    }
}
