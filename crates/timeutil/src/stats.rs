//! Latency sample summaries: nearest-rank percentiles over `f64` millisecond
//! samples. Hand-rolled on purpose — no stats crate is on the §3 allowlist
//! and nearest-rank is four lines.

use serde::{Deserialize, Serialize};

/// Summary of one latency sample set, in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Number of samples summarized.
    pub count: u32,
    /// Smallest sample.
    pub min_ms: f64,
    /// Arithmetic mean.
    pub mean_ms: f64,
    /// Nearest-rank 50th percentile.
    pub p50_ms: f64,
    /// Nearest-rank 95th percentile.
    pub p95_ms: f64,
    /// Nearest-rank 99th percentile.
    pub p99_ms: f64,
    /// Largest sample.
    pub max_ms: f64,
}

/// Sorts `samples` in place and summarizes them; `None` when empty.
///
/// Callers must only push real measured durations (finite, ≥ 0) — probes
/// never insert sentinel values, so no NaN handling is needed beyond
/// `total_cmp` ordering.
#[must_use]
pub fn stats_from_ms(samples: &mut [f64]) -> Option<LatencyStats> {
    samples.sort_unstable_by(f64::total_cmp);
    let (&min, &max) = (samples.first()?, samples.last()?);
    let count = samples.len();
    let sum: f64 = samples.iter().sum();
    Some(LatencyStats {
        count: u32::try_from(count).unwrap_or(u32::MAX),
        min_ms: min,
        mean_ms: sum / count as f64,
        p50_ms: percentile_nearest_rank(samples, 50.0),
        p95_ms: percentile_nearest_rank(samples, 95.0),
        p99_ms: percentile_nearest_rank(samples, 99.0),
        max_ms: max,
    })
}

/// Nearest-rank percentile: rank = ⌈(pct / 100) × N⌉, 1-based.
///
/// `sorted` must be sorted ascending; `pct` must be in `(0, 100]`. Returns
/// `NaN` for an empty slice (defensive — never panics on a runtime path).
#[must_use]
pub fn percentile_nearest_rank(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let rank = ((pct / 100.0) * n as f64).ceil() as usize;
    sorted
        .get(rank.clamp(1, n) - 1)
        .copied()
        .unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_textbook_small() {
        // Classic worked example: N=4, p50 → rank ⌈0.5×4⌉ = 2 → 2nd element.
        let v = [15.0, 20.0, 35.0, 50.0];
        assert_eq!(percentile_nearest_rank(&v, 50.0), 20.0);
        assert_eq!(percentile_nearest_rank(&v, 100.0), 50.0);
        // p25 → rank ⌈1.0⌉ = 1 → first element.
        assert_eq!(percentile_nearest_rank(&v, 25.0), 15.0);
    }

    #[test]
    fn nearest_rank_n100() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        // rank = ⌈p⌉ exactly, so pXX is the XXth value (index XX−1).
        assert_eq!(percentile_nearest_rank(&v, 50.0), 50.0);
        assert_eq!(percentile_nearest_rank(&v, 95.0), 95.0);
        assert_eq!(percentile_nearest_rank(&v, 99.0), 99.0);
    }

    #[test]
    fn single_sample_and_empty() {
        let mut one = [42.0];
        let s = stats_from_ms(&mut one).unwrap();
        assert_eq!(s.count, 1);
        assert_eq!(s.min_ms, 42.0);
        assert_eq!(s.p50_ms, 42.0);
        assert_eq!(s.p95_ms, 42.0);
        assert_eq!(s.p99_ms, 42.0);
        assert_eq!(s.max_ms, 42.0);
        assert!(stats_from_ms(&mut []).is_none());
        assert!(percentile_nearest_rank(&[], 50.0).is_nan());
    }

    #[test]
    fn unsorted_input_is_sorted_internally() {
        let mut v = [9.0, 1.0, 5.0, 3.0, 7.0];
        let s = stats_from_ms(&mut v).unwrap();
        assert_eq!(s.min_ms, 1.0);
        assert_eq!(s.max_ms, 9.0);
        assert_eq!(s.p50_ms, 5.0); // rank ⌈2.5⌉ = 3 → 3rd of sorted
        assert_eq!(s.mean_ms, 5.0);
        assert_eq!(v, [1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn stats_serde_round_trips() {
        let mut v = [1.0, 2.0, 3.0];
        let s = stats_from_ms(&mut v).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        let back: LatencyStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}
