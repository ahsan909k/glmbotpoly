//! Dashboard state for the "Shadow" tile: per-series prediction rate + last
//! p_up + feature coverage, plus the deployed-model identity and the
//! "model stale, refit due" flag. Fed by the shadow observer via
//! [`crate::DashboardHandle::set_shadow`] (NOT a bus event); shadow influences
//! nothing, so this is display-only.

use std::collections::HashMap;

use core_types::{Series, TimestampMs};

/// Recent prediction timestamps retained per series (for the per-minute rate).
const RECENT_CAP: usize = 64;
/// Window over which "predictions/min" is measured (ms).
pub(crate) const RATE_WINDOW_MS: i64 = 60_000;
/// Milliseconds in a day (staleness math).
pub(crate) const MS_PER_DAY: i64 = 86_400_000;

/// One shadow prediction handed to the dashboard (dashboard-owned, so the crate
/// keeps no dependency on `shadow`).
#[derive(Debug, Clone)]
pub struct ShadowTick {
    /// Series the prediction is for.
    pub series: Series,
    /// Sample time.
    pub ts: TimestampMs,
    /// Model probability of Up.
    pub p_up: f64,
    /// How many of the 24 features were finite.
    pub finite_count: u32,
    /// Model training-data end (ms).
    pub trained_through_ms: i64,
    /// Short model sha for display.
    pub short_sha: String,
    /// Staleness threshold (days).
    pub staleness_alert_days: i64,
}

/// Per-series shadow state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShadowSeries {
    /// Recent prediction times (ms), bounded — the rate source.
    pub(crate) recent_ts: Vec<i64>,
    pub(crate) last_ts: i64,
    pub(crate) last_p_up: f64,
    pub(crate) last_finite: u32,
}

impl ShadowSeries {
    pub(crate) fn record(&mut self, ts: i64, p_up: f64, finite: u32) {
        self.recent_ts.push(ts);
        if self.recent_ts.len() > RECENT_CAP {
            self.recent_ts.remove(0);
        }
        self.last_ts = ts;
        self.last_p_up = p_up;
        self.last_finite = finite;
    }

    /// Predictions in the last [`RATE_WINDOW_MS`] (== predictions/min for a 60 s window).
    pub(crate) fn per_min(&self, now_ms: i64) -> usize {
        self.recent_ts
            .iter()
            .filter(|&&t| now_ms - t <= RATE_WINDOW_MS)
            .count()
    }
}

/// The full shadow tile state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShadowState {
    pub(crate) by_series: HashMap<Series, ShadowSeries>,
    pub(crate) trained_through_ms: i64,
    pub(crate) short_sha: String,
    pub(crate) staleness_alert_days: i64,
    /// True once any prediction has arrived (shadow is running).
    pub(crate) active: bool,
}

impl ShadowState {
    /// Folds one prediction into the tile state.
    pub(crate) fn record(&mut self, tick: &ShadowTick) {
        self.by_series.entry(tick.series).or_default().record(
            tick.ts.as_millis(),
            tick.p_up,
            tick.finite_count,
        );
        self.trained_through_ms = tick.trained_through_ms;
        self.short_sha = tick.short_sha.clone();
        self.staleness_alert_days = tick.staleness_alert_days;
        self.active = true;
    }

    /// Whether the deployed model's training-end date is more than the threshold
    /// behind `now` (the "model stale, refit due" flag).
    pub(crate) fn is_stale(&self, now: TimestampMs) -> bool {
        self.active
            && self.staleness_alert_days > 0
            && now.as_millis() - self.trained_through_ms > self.staleness_alert_days * MS_PER_DAY
    }
}

#[cfg(test)]
mod tests {
    use core_types::{Asset, WindowDuration};

    use super::*;

    fn series() -> Series {
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        }
    }

    fn tick(ts_ms: i64, trained_through_ms: i64) -> ShadowTick {
        ShadowTick {
            series: series(),
            ts: TimestampMs::from_millis(ts_ms),
            p_up: 0.6,
            finite_count: 22,
            trained_through_ms,
            short_sha: "abcdef012345".to_owned(),
            staleness_alert_days: 30,
        }
    }

    #[test]
    fn records_and_counts_per_minute() {
        let mut st = ShadowState::default();
        // 12 predictions across 55 s → all within the 60 s window.
        for i in 0..12 {
            st.record(&tick(i * 5_000, 0));
        }
        assert!(st.active);
        let s = st.by_series.get(&series()).expect("series present");
        assert_eq!(s.per_min(60_000), 12);
        // A later query drops the ones older than 60 s (all samples ≤ 55 s).
        assert_eq!(s.per_min(120_000), 0);
        assert_eq!(s.last_finite, 22);
    }

    #[test]
    fn staleness_flag_fires_past_threshold() {
        let mut st = ShadowState::default();
        let trained = 1_700_000_000_000;
        st.record(&tick(trained, trained));
        // 10 days later: fresh.
        assert!(!st.is_stale(TimestampMs::from_millis(trained + 10 * MS_PER_DAY)));
        // 31 days later: stale (> 30-day threshold).
        assert!(st.is_stale(TimestampMs::from_millis(trained + 31 * MS_PER_DAY)));
    }

    #[test]
    fn inactive_is_never_stale() {
        let st = ShadowState::default();
        assert!(!st.is_stale(TimestampMs::from_millis(i64::MAX / 2)));
    }
}
