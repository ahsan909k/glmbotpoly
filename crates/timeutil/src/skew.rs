//! Sans-IO clock-skew alarm logic (CLAUDE.md §11).
//!
//! Pure state machine, scheduler-style: measurements and ticks go in with
//! `now` as a parameter, decisions come out as values. No clock, no IO, no
//! logging — the driver in [`crate::monitor`] does all of that.
//!
//! Policy (Decisions Log 2026-06-12):
//! - **Trip** only after `trip_after` consecutive rounds with
//!   `|offset| ≥ trip_bound` — one jittery sample never costs a cancel-all.
//! - **Clear** only after `clear_after` consecutive rounds with
//!   `|offset| < clear_bound` (a *lower* bound: hysteresis, no flapping).
//! - Offsets in the dead zone between the bounds reset both streaks.
//! - **A failed round never trips and never clears** — a lost UDP packet says
//!   nothing about local clock health; it resets the clear streak (a trip may
//!   not clear on stale information) and preserves the trip streak.
//! - Staleness (no successful round for `stale_warn`) produces escalating
//!   warnings, never a trip: worst-case crystal drift (~50 ppm) needs over
//!   80 minutes unsynced to accumulate 250 ms.

use core_types::{DurationMs, TimestampMs};

use crate::offset::OffsetMeasurement;

/// Skew-alarm policy knobs (mapped from config by the binary).
#[derive(Debug, Clone, Copy)]
pub struct SkewParams {
    /// `|offset| ≥ this` counts toward a trip.
    pub trip_bound: DurationMs,
    /// `|offset| < this` counts toward a clear. Must be < `trip_bound`.
    pub clear_bound: DurationMs,
    /// Consecutive breaching rounds required to trip.
    pub trip_after: u32,
    /// Consecutive in-bound rounds required to clear.
    pub clear_after: u32,
    /// Age of the last successful round past which staleness warnings fire.
    pub stale_warn: DurationMs,
}

/// A decision produced by the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkewOutput {
    /// The clock-skew breaker must trip (emitted exactly once per episode).
    Trip {
        /// The offset that confirmed the trip.
        offset: DurationMs,
    },
    /// The breaker may clear (emitted exactly once per episode).
    Clear {
        /// The offset that confirmed the clear.
        offset: DurationMs,
    },
    /// Non-actionable observation worth logging.
    Warn(SkewWarning),
}

/// Warnings: visible in logs, never order-flow-affecting by themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkewWarning {
    /// A breaching offset observed before the trip debounce is met.
    BreachObserved {
        /// The breaching offset.
        offset: DurationMs,
        /// Consecutive breaching rounds so far.
        consecutive: u32,
    },
    /// A measurement round produced no usable samples.
    RoundFailed {
        /// Consecutive failed rounds so far.
        consecutive_failures: u32,
    },
    /// No successful round for longer than `stale_warn` (re-emitted once per
    /// `stale_warn` period while the condition persists).
    OffsetStale {
        /// Time since the last successful round (or since startup).
        age: DurationMs,
    },
}

/// The sans-IO skew monitor. Feed it measurement rounds and ticks; act on the
/// outputs.
#[derive(Debug)]
pub struct SkewMonitor {
    params: SkewParams,
    tripped: bool,
    breach_streak: u32,
    ok_streak: u32,
    fail_streak: u32,
    /// Last successful round: (offset, when).
    last_success: Option<(DurationMs, TimestampMs)>,
    /// Staleness anchor when nothing has ever succeeded.
    first_seen: Option<TimestampMs>,
    last_stale_warn: Option<TimestampMs>,
}

impl SkewMonitor {
    /// Creates a monitor in the untripped state.
    #[must_use]
    pub fn new(params: SkewParams) -> Self {
        Self {
            params,
            tripped: false,
            breach_streak: 0,
            ok_streak: 0,
            fail_streak: 0,
            last_success: None,
            first_seen: None,
            last_stale_warn: None,
        }
    }

    /// Processes one measurement round.
    pub fn on_measurement(
        &mut self,
        m: &OffsetMeasurement,
        now: TimestampMs,
        out: &mut Vec<SkewOutput>,
    ) {
        self.first_seen.get_or_insert(now);
        let Some(offset) = m.offset else {
            self.fail_streak = self.fail_streak.saturating_add(1);
            // A trip must not clear on stale information; the trip streak is
            // preserved (failures say nothing either way).
            self.ok_streak = 0;
            out.push(SkewOutput::Warn(SkewWarning::RoundFailed {
                consecutive_failures: self.fail_streak,
            }));
            return;
        };

        self.fail_streak = 0;
        self.last_success = Some((offset, now));
        self.last_stale_warn = None;

        let magnitude = offset.as_millis().abs();
        if magnitude >= self.params.trip_bound.as_millis() {
            self.breach_streak = self.breach_streak.saturating_add(1);
            self.ok_streak = 0;
            if self.tripped {
                // Still skewed while tripped: nothing new to announce.
            } else if self.breach_streak >= self.params.trip_after {
                self.tripped = true;
                out.push(SkewOutput::Trip { offset });
            } else {
                out.push(SkewOutput::Warn(SkewWarning::BreachObserved {
                    offset,
                    consecutive: self.breach_streak,
                }));
            }
        } else if magnitude < self.params.clear_bound.as_millis() {
            self.ok_streak = self.ok_streak.saturating_add(1);
            self.breach_streak = 0;
            if self.tripped && self.ok_streak >= self.params.clear_after {
                self.tripped = false;
                self.ok_streak = 0;
                out.push(SkewOutput::Clear { offset });
            }
        } else {
            // Hysteresis dead zone: neither healthy enough to clear nor bad
            // enough to trip — both streaks reset.
            self.breach_streak = 0;
            self.ok_streak = 0;
        }
    }

    /// Time-based staleness check (call at ~1 s cadence between rounds).
    pub fn on_tick(&mut self, now: TimestampMs, out: &mut Vec<SkewOutput>) {
        let anchor = self
            .last_success
            .map(|(_, at)| at)
            .or(self.first_seen)
            .unwrap_or_else(|| *self.first_seen.get_or_insert(now));
        let age = now.signed_duration_since(anchor);
        if age.as_millis() < self.params.stale_warn.as_millis() {
            return;
        }
        let due = self.last_stale_warn.is_none_or(|last| {
            now.signed_duration_since(last).as_millis() >= self.params.stale_warn.as_millis()
        });
        if due {
            self.last_stale_warn = Some(now);
            out.push(SkewOutput::Warn(SkewWarning::OffsetStale { age }));
        }
    }

    /// Whether the clock-skew breaker is currently tripped.
    #[must_use]
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Last successful offset and when it was measured.
    #[must_use]
    pub fn last_offset(&self) -> Option<(DurationMs, TimestampMs)> {
        self.last_success
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SkewParams {
        SkewParams {
            trip_bound: DurationMs::from_millis(250),
            clear_bound: DurationMs::from_millis(125),
            trip_after: 2,
            clear_after: 3,
            stale_warn: DurationMs::from_millis(300_000),
        }
    }

    fn ok(offset_ms: i64) -> OffsetMeasurement {
        OffsetMeasurement {
            offset: Some(DurationMs::from_millis(offset_ms)),
            samples_used: 4,
            queries_failed: 0,
            min_round_trip: Some(DurationMs::from_millis(10)),
        }
    }

    fn failed() -> OffsetMeasurement {
        OffsetMeasurement {
            offset: None,
            samples_used: 0,
            queries_failed: 6,
            min_round_trip: None,
        }
    }

    fn feed(monitor: &mut SkewMonitor, m: &OffsetMeasurement, at_ms: i64) -> Vec<SkewOutput> {
        let mut out = Vec::new();
        monitor.on_measurement(m, TimestampMs::from_millis(at_ms), &mut out);
        out
    }

    #[test]
    fn trips_only_after_debounce() {
        let mut mon = SkewMonitor::new(params());
        let first = feed(&mut mon, &ok(300), 1_000);
        assert_eq!(
            first,
            vec![SkewOutput::Warn(SkewWarning::BreachObserved {
                offset: DurationMs::from_millis(300),
                consecutive: 1,
            })]
        );
        assert!(!mon.is_tripped());

        let second = feed(&mut mon, &ok(310), 2_000);
        assert_eq!(
            second,
            vec![SkewOutput::Trip {
                offset: DurationMs::from_millis(310)
            }]
        );
        assert!(mon.is_tripped());

        // Still breaching while tripped: silent.
        assert!(feed(&mut mon, &ok(320), 3_000).is_empty());
    }

    #[test]
    fn jitter_spike_never_trips() {
        let mut mon = SkewMonitor::new(params());
        for (i, offset) in [300, 10, 280, 5, 400, 0].into_iter().enumerate() {
            feed(&mut mon, &ok(offset), (i as i64 + 1) * 1_000);
            assert!(!mon.is_tripped(), "tripped on non-consecutive breaches");
        }
    }

    #[test]
    fn negative_offsets_count_by_magnitude() {
        let mut mon = SkewMonitor::new(params());
        feed(&mut mon, &ok(-300), 1_000);
        let out = feed(&mut mon, &ok(-260), 2_000);
        assert_eq!(
            out,
            vec![SkewOutput::Trip {
                offset: DurationMs::from_millis(-260)
            }]
        );
    }

    #[test]
    fn dead_zone_resets_both_streaks() {
        let mut mon = SkewMonitor::new(params());
        // Breach, dead zone, breach, breach → the dead zone broke the first
        // streak, so the trip lands on the 2nd consecutive breach (round 4).
        feed(&mut mon, &ok(300), 1_000);
        feed(&mut mon, &ok(200), 2_000); // dead zone: 125 ≤ 200 < 250
        feed(&mut mon, &ok(300), 3_000);
        assert!(!mon.is_tripped());
        feed(&mut mon, &ok(300), 4_000);
        assert!(mon.is_tripped());

        // While tripped: two clears, a dead-zone reading, then three clears
        // → only the uninterrupted run of three clears.
        feed(&mut mon, &ok(50), 5_000);
        feed(&mut mon, &ok(60), 6_000);
        feed(&mut mon, &ok(150), 7_000); // dead zone resets ok streak
        feed(&mut mon, &ok(40), 8_000);
        feed(&mut mon, &ok(30), 9_000);
        assert!(mon.is_tripped());
        let out = feed(&mut mon, &ok(20), 10_000);
        assert_eq!(
            out,
            vec![SkewOutput::Clear {
                offset: DurationMs::from_millis(20)
            }]
        );
        assert!(!mon.is_tripped());
    }

    #[test]
    fn clear_requires_consecutive_and_failures_reset_it() {
        let mut mon = SkewMonitor::new(params());
        feed(&mut mon, &ok(300), 1_000);
        feed(&mut mon, &ok(300), 2_000);
        assert!(mon.is_tripped());

        feed(&mut mon, &ok(10), 3_000);
        feed(&mut mon, &ok(10), 4_000);
        // A failed round must reset the clear streak: a trip may not clear
        // on stale information.
        let out = feed(&mut mon, &failed(), 5_000);
        assert_eq!(
            out,
            vec![SkewOutput::Warn(SkewWarning::RoundFailed {
                consecutive_failures: 1
            })]
        );
        feed(&mut mon, &ok(10), 6_000);
        feed(&mut mon, &ok(10), 7_000);
        assert!(
            mon.is_tripped(),
            "cleared without 3 consecutive good rounds"
        );
        feed(&mut mon, &ok(10), 8_000);
        assert!(!mon.is_tripped());
    }

    #[test]
    fn failures_never_trip_and_preserve_a_trip() {
        let mut mon = SkewMonitor::new(params());
        for i in 1..=5_i64 {
            let out = feed(&mut mon, &failed(), i * 1_000);
            assert_eq!(
                out,
                vec![SkewOutput::Warn(SkewWarning::RoundFailed {
                    consecutive_failures: u32::try_from(i).unwrap()
                })]
            );
        }
        assert!(!mon.is_tripped());

        // Trip, then fail for a long time: stays tripped.
        feed(&mut mon, &ok(300), 10_000);
        feed(&mut mon, &ok(300), 11_000);
        assert!(mon.is_tripped());
        for i in 0..10_i64 {
            feed(&mut mon, &failed(), 12_000 + i * 1_000);
            assert!(mon.is_tripped());
        }
    }

    #[test]
    fn failures_preserve_the_trip_streak() {
        // Two genuine breaches separated by network failures still trip:
        // both measurements said "skewed"; the failures said nothing.
        let mut mon = SkewMonitor::new(params());
        feed(&mut mon, &ok(300), 1_000);
        feed(&mut mon, &failed(), 2_000);
        feed(&mut mon, &failed(), 3_000);
        feed(&mut mon, &ok(300), 4_000);
        assert!(mon.is_tripped());
    }

    #[test]
    fn staleness_warns_once_per_period() {
        let mut mon = SkewMonitor::new(params());
        feed(&mut mon, &ok(10), 0);

        let mut out = Vec::new();
        mon.on_tick(TimestampMs::from_millis(299_000), &mut out);
        assert!(out.is_empty());

        mon.on_tick(TimestampMs::from_millis(300_000), &mut out);
        assert_eq!(
            out,
            vec![SkewOutput::Warn(SkewWarning::OffsetStale {
                age: DurationMs::from_millis(300_000)
            })]
        );

        // Within the same period: silent. After another full period: warns
        // again, with the larger age.
        out.clear();
        mon.on_tick(TimestampMs::from_millis(400_000), &mut out);
        assert!(out.is_empty());
        mon.on_tick(TimestampMs::from_millis(600_000), &mut out);
        assert_eq!(
            out,
            vec![SkewOutput::Warn(SkewWarning::OffsetStale {
                age: DurationMs::from_millis(600_000)
            })]
        );

        // A successful round re-arms the warning from scratch.
        out.clear();
        feed(&mut mon, &ok(10), 700_000);
        mon.on_tick(TimestampMs::from_millis(999_000), &mut out);
        assert!(out.is_empty());
        mon.on_tick(TimestampMs::from_millis(1_000_000), &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn staleness_anchors_to_first_seen_when_nothing_ever_succeeded() {
        let mut mon = SkewMonitor::new(params());
        let mut out = Vec::new();
        mon.on_tick(TimestampMs::from_millis(0), &mut out);
        assert!(out.is_empty());
        mon.on_tick(TimestampMs::from_millis(300_000), &mut out);
        assert_eq!(out.len(), 1, "no stale warning without any success ever");
    }

    #[test]
    fn last_offset_tracks_successes() {
        let mut mon = SkewMonitor::new(params());
        assert_eq!(mon.last_offset(), None);
        feed(&mut mon, &ok(42), 1_000);
        assert_eq!(
            mon.last_offset(),
            Some((DurationMs::from_millis(42), TimestampMs::from_millis(1_000)))
        );
        feed(&mut mon, &failed(), 2_000);
        assert_eq!(
            mon.last_offset(),
            Some((DurationMs::from_millis(42), TimestampMs::from_millis(1_000)))
        );
    }
}
