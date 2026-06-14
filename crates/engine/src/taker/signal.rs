//! Confirmed-move detector for the momentum taker (CLAUDE.md §8).
//!
//! The momentum taker fires only when the *fast signal feed* (direct Binance
//! mid — the lowest-latency leading signal, ~2.7 s ahead of Chainlink per the
//! 2026-06-12 feed-comparison findings) has just made a move large enough to
//! stand out of the noise: a signed log-return over a short lookback exceeding a
//! configured multiple of the current one-second volatility. [`SignalWindow`]
//! keeps a short ring of recent prices and answers that question.
//!
//! Pure and sans-clock: `now` and `sigma_1s` are passed in (the driver supplies
//! `sigma_1s` from the model snapshot, so a take is implicitly gated on model
//! health), and no clock is read here. No panic on any runtime path — every
//! fallible step maps to `None`, the [`crate::quoting`]/[`crate::normalize`]
//! discipline.

use std::collections::VecDeque;

use core_types::{Decimal, Outcome, TimestampMs};
use rust_decimal::prelude::ToPrimitive;

/// A confirmed directional move on the signal feed.
///
/// `outcome` is the side the move made *more* likely (and therefore the side
/// whose book asks may now be stale-cheap): an up-move ⇒ [`Outcome::Up`], a
/// down-move ⇒ [`Outcome::Down`]. The diagnostics (`r`, `threshold`,
/// `baseline_age_secs`) are carried for logging and journal calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfirmedMove {
    /// The outcome the move favors (the candidate side to take).
    pub outcome: Outcome,
    /// Signed log-return over the lookback (`ln(S_now / S_baseline)`).
    pub r: f64,
    /// The noise threshold it cleared (`sigma_mult · σ_1s · √T`).
    pub threshold: f64,
    /// Actual span between the baseline and newest tick used, in seconds.
    pub baseline_age_secs: f64,
}

/// A short ring of recent fast-feed prices for one asset, with confirmed-move
/// detection.
///
/// Entries are `(local-receive time, price)`; the buffer is pruned to the
/// lookback on every push so memory stays bounded (~`lookback × tick-rate`
/// entries; a 2 s lookback over ~100 mid-ticks/s ≈ 200 entries). The asset
/// scope lives in the driver (one `SignalWindow` per [`Asset`](core_types::Asset)),
/// so the ring spans window rolls — the price history is continuous.
#[derive(Debug, Clone)]
pub struct SignalWindow {
    buf: VecDeque<(TimestampMs, Decimal)>,
    lookback_ms: i64,
}

impl SignalWindow {
    /// Builds an empty window with the given lookback (milliseconds).
    #[must_use]
    pub fn new(lookback_ms: i64) -> Self {
        Self {
            buf: VecDeque::new(),
            lookback_ms,
        }
    }

    /// Records one fast-feed observation, then prunes entries older than the
    /// lookback relative to this tick.
    pub fn push(&mut self, ts: TimestampMs, value: Decimal) {
        self.buf.push_back((ts, value));
        let cutoff = ts.as_millis().saturating_sub(self.lookback_ms);
        while let Some(&(front_ts, _)) = self.buf.front() {
            if front_ts.as_millis() < cutoff {
                self.buf.pop_front();
            } else {
                break;
            }
        }
    }

    /// Number of buffered observations (after the last prune).
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True when the ring holds no observations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Detects a confirmed move over the lookback ending at `now`.
    ///
    /// Returns `Some(ConfirmedMove)` iff there are ≥ 2 usable in-window ticks,
    /// `sigma_1s` is a usable volatility (finite, positive), and the signed
    /// log-return between the oldest and newest in-window ticks exceeds
    /// `sigma_mult · sigma_1s · √T` in magnitude, where `T` is the *actual*
    /// span between those two ticks in seconds. Using the actual span (rather
    /// than the nominal lookback) keeps the threshold self-consistent under the
    /// bursty arrival of fast-feed ticks. `None` otherwise.
    #[must_use]
    pub fn confirmed_direction(
        &self,
        now: TimestampMs,
        sigma_1s: f64,
        sigma_mult: f64,
    ) -> Option<ConfirmedMove> {
        if !sigma_1s.is_finite() || sigma_1s <= 0.0 {
            return None;
        }
        let cutoff = now.as_millis().saturating_sub(self.lookback_ms);

        // Oldest and newest *usable* (positive-priced) ticks within the window.
        let mut oldest: Option<(i64, f64)> = None;
        let mut newest: Option<(i64, f64)> = None;
        let mut count = 0u32;
        for &(ts, value) in &self.buf {
            let ms = ts.as_millis();
            if ms < cutoff {
                continue;
            }
            let Some(v) = value.to_f64() else { continue };
            if v <= 0.0 {
                continue;
            }
            count += 1;
            if oldest.is_none_or(|(o_ms, _)| ms < o_ms) {
                oldest = Some((ms, v));
            }
            if newest.is_none_or(|(n_ms, _)| ms > n_ms) {
                newest = Some((ms, v));
            }
        }
        if count < 2 {
            return None;
        }
        let (o_ms, s_old) = oldest?;
        let (n_ms, s_new) = newest?;
        let t_secs = (n_ms - o_ms) as f64 / 1000.0;
        if t_secs <= 0.0 {
            return None;
        }
        let r = (s_new / s_old).ln();
        if !r.is_finite() {
            return None;
        }
        let threshold = sigma_mult * sigma_1s * t_secs.sqrt();
        if r.abs() > threshold {
            let outcome = if r > 0.0 { Outcome::Up } else { Outcome::Down };
            Some(ConfirmedMove {
                outcome,
                r,
                threshold,
                baseline_age_secs: t_secs,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const T0: i64 = 1_781_000_000_000;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    /// A 2-second window seeded with a price ramp from `base`, stepping by
    /// `step` every `dt_ms`, `n` ticks, the first at `T0`.
    fn ramp(base: Decimal, step: Decimal, dt_ms: i64, n: i64) -> SignalWindow {
        let mut w = SignalWindow::new(2_000);
        for i in 0..n {
            w.push(ts(T0 + i * dt_ms), base + step * Decimal::from(i));
        }
        w
    }

    #[test]
    fn confirmed_up_move() {
        // 10 ticks 0→900 ms, +$30 each: 60000 → 60270, r = ln(60270/60000) ≈
        // 0.004491 over T = 0.9 s. σ 1e-4, mult 3 ⇒ threshold 3·1e-4·√0.9 ≈
        // 0.0002846. r ≫ threshold ⇒ Up.
        let w = ramp(dec!(60000), dec!(30), 100, 10);
        let m = w
            .confirmed_direction(ts(T0 + 900), 1e-4, 3.0)
            .expect("a confirmed move");
        assert_eq!(m.outcome, Outcome::Up);
        assert!(m.r > 0.0);
        assert!(m.r > m.threshold);
        assert!((m.baseline_age_secs - 0.9).abs() < 1e-9);
    }

    #[test]
    fn confirmed_down_move() {
        let w = ramp(dec!(60000), dec!(-30), 100, 10);
        let m = w
            .confirmed_direction(ts(T0 + 900), 1e-4, 3.0)
            .expect("a confirmed move");
        assert_eq!(m.outcome, Outcome::Down);
        assert!(m.r < 0.0);
    }

    #[test]
    fn below_threshold_is_not_confirmed() {
        // Same small ramp but a high σ ⇒ threshold dwarfs the realized move.
        let w = ramp(dec!(60000), dec!(30), 100, 10);
        assert!(w.confirmed_direction(ts(T0 + 900), 1e-2, 3.0).is_none());
    }

    #[test]
    fn flat_feed_is_not_confirmed() {
        let mut w = SignalWindow::new(2_000);
        for i in 0..10 {
            w.push(ts(T0 + i * 100), dec!(60000));
        }
        // r = ln(1) = 0 ≤ any positive threshold.
        assert!(w.confirmed_direction(ts(T0 + 900), 1e-6, 3.0).is_none());
    }

    #[test]
    fn fewer_than_two_ticks_is_none() {
        let mut w = SignalWindow::new(2_000);
        assert!(w.confirmed_direction(ts(T0), 1e-4, 3.0).is_none());
        w.push(ts(T0), dec!(60000));
        assert!(w.confirmed_direction(ts(T0), 1e-4, 3.0).is_none());
    }

    #[test]
    fn unusable_sigma_is_none() {
        let w = ramp(dec!(60000), dec!(30), 100, 10);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(w.confirmed_direction(ts(T0 + 900), bad, 3.0).is_none());
        }
    }

    #[test]
    fn pruning_drops_stale_ticks_and_baseline_is_oldest_in_window() {
        // Push old ticks well before the window, then a fresh ramp. The oldest
        // in-window tick (not the long-gone ones) anchors the baseline.
        let mut w = SignalWindow::new(2_000);
        w.push(ts(T0), dec!(50000)); // far in the past relative to `now`
        w.push(ts(T0 + 100), dec!(50000));
        for i in 0..5 {
            w.push(
                ts(T0 + 10_000 + i * 100),
                dec!(60000) + dec!(30) * Decimal::from(i),
            );
        }
        // now = last tick; window = [now-2000, now] excludes the T0 ticks.
        let now = ts(T0 + 10_400);
        let m = w
            .confirmed_direction(now, 1e-4, 3.0)
            .expect("confirmed within the fresh window");
        assert_eq!(m.outcome, Outcome::Up);
        // Baseline is the first fresh tick (age 0.4 s), not the 50000 history.
        assert!((m.baseline_age_secs - 0.4).abs() < 1e-9);
        // The push-time prune kept the buffer bounded (history dropped).
        assert!(w.len() <= 5);
    }

    #[test]
    fn threshold_is_dimensionally_pinned() {
        // Two ticks exactly 2 s apart; threshold must equal mult·σ·√2 to 1e-9.
        let mut w = SignalWindow::new(5_000);
        w.push(ts(T0), dec!(60000));
        w.push(ts(T0 + 2_000), dec!(60000) * (dec!(1) + dec!(0.001)));
        let m = w
            .confirmed_direction(ts(T0 + 2_000), 2e-4, 3.0)
            .expect("confirmed");
        let want = 3.0 * 2e-4 * 2.0_f64.sqrt();
        assert!(
            (m.threshold - want).abs() < 1e-9,
            "{} vs {want}",
            m.threshold
        );
        assert!((m.baseline_age_secs - 2.0).abs() < 1e-9);
    }
}
