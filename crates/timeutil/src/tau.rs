//! τ helpers: time remaining in a window.
//!
//! The fair-value model (CLAUDE.md §8) uses `z = ln(S/K) / (σ_1s × √τ)` with
//! τ in seconds; the engine's end-of-window gates (no-ATM, no-passive) work in
//! signed milliseconds. Both readings come from here so no caller invents its
//! own clamping rules.

use core_types::{DurationMs, TimestampMs};

/// Seconds remaining until `close`, clamped at `0.0`.
///
/// The model boundary: τ feeds `√τ`, so anything at-or-after close is exactly
/// `0.0` — never negative, never NaN.
#[must_use]
pub fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.signed_duration_since(now).as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

/// Signed milliseconds remaining until `close` — negative after close.
///
/// Engine cutoffs need the sign: "5 s before close" and "5 s after close"
/// are very different states.
#[must_use]
pub fn remaining_ms(now: TimestampMs, close: TimestampMs) -> DurationMs {
    close.signed_duration_since(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_is_zero_at_and_after_close() {
        let close = TimestampMs::from_millis(1_000_000);
        assert_eq!(tau_secs(close, close), 0.0);
        assert_eq!(tau_secs(TimestampMs::from_millis(1_000_001), close), 0.0);
        assert_eq!(tau_secs(TimestampMs::from_millis(2_000_000), close), 0.0);
    }

    #[test]
    fn tau_has_millisecond_precision() {
        let now = TimestampMs::from_millis(0);
        let close = TimestampMs::from_millis(90_500);
        assert!((tau_secs(now, close) - 90.5).abs() < f64::EPSILON);
        let close_1ms = TimestampMs::from_millis(1);
        assert!((tau_secs(now, close_1ms) - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn remaining_ms_is_signed() {
        let close = TimestampMs::from_millis(1_000);
        assert_eq!(
            remaining_ms(TimestampMs::from_millis(0), close),
            DurationMs::from_millis(1_000)
        );
        let after = remaining_ms(TimestampMs::from_millis(1_500), close);
        assert_eq!(after, DurationMs::from_millis(-500));
        assert!(after.is_negative());
    }
}
