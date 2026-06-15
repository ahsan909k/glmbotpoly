//! The per-series adverse-selection alarm (CLAUDE.md §8/§10/§14).
//!
//! Persistently negative 5-second markouts on passive fills mean our resting
//! quotes are being picked off. This sans-IO monitor — modelled on
//! `model::HealthMonitor` (inputs in, latched transitions out, no clock) — folds
//! finalized 5s markouts for one series and derives a three-state health gate.
//! It stays `InsufficientSample` until enough passive fills are seen, then
//! `Alarm` while the rolling-mean 5s markout is below the (typically zero)
//! threshold, else `Ok`. Transitions are latched (one event per change) so the
//! risk panel / dashboard react once per flip, not per fill.

use std::collections::VecDeque;

use core_types::{Mode, Series};
use serde::{Deserialize, Serialize};

use crate::params::AdverseSelectionParams;

/// The per-series adverse-selection gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdverseSelectionState {
    /// Too few passive 5s markouts to judge yet.
    InsufficientSample,
    /// Passive fills are aging acceptably (rolling-mean 5s markout ≥ threshold).
    Ok,
    /// Persistently negative 5s markouts — we are being adversely selected.
    Alarm,
}

/// A latched adverse-selection state change for one series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeriesHealth {
    /// Series whose adverse-selection state changed.
    pub series: Series,
    /// Session mode.
    pub mode: Mode,
    /// Previous state.
    pub from: AdverseSelectionState,
    /// New state.
    pub to: AdverseSelectionState,
}

/// Folds finalized 5s markouts for one series into the adverse-selection gate.
#[derive(Debug, Clone, PartialEq)]
pub struct AdverseSelectionMonitor {
    params: AdverseSelectionParams,
    /// The most recent `params.window` 5s markouts (probability units).
    recent: VecDeque<f64>,
    /// Total 5s markouts observed (the `min_sample` gate).
    total: u64,
    state: AdverseSelectionState,
}

impl AdverseSelectionMonitor {
    /// A fresh monitor in [`AdverseSelectionState::InsufficientSample`].
    #[must_use]
    pub fn new(params: AdverseSelectionParams) -> Self {
        Self {
            params,
            recent: VecDeque::new(),
            total: 0,
            state: AdverseSelectionState::InsufficientSample,
        }
    }

    /// Folds one passive fill's 5s markout. Returns `(from, to)` only when the
    /// gate state changes (latched — one transition per flip).
    pub fn observe_5s(
        &mut self,
        markout: f64,
    ) -> Option<(AdverseSelectionState, AdverseSelectionState)> {
        self.recent.push_back(markout);
        while self.recent.len() > self.params.window as usize {
            self.recent.pop_front();
        }
        self.total += 1;

        let next = self.evaluate();
        let from = self.state;
        if next == from {
            None
        } else {
            self.state = next;
            Some((from, next))
        }
    }

    /// The current gate state.
    #[must_use]
    pub fn state(&self) -> AdverseSelectionState {
        self.state
    }

    /// Total 5s markouts observed.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.total
    }

    /// The rolling-mean 5s markout, `None` until any markout is observed.
    #[must_use]
    pub fn mean_5s(&self) -> Option<f64> {
        if self.recent.is_empty() {
            None
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "rolling-window length is bounded by params.window"
            )]
            let n = self.recent.len() as f64;
            Some(self.recent.iter().sum::<f64>() / n)
        }
    }

    /// Computes the gate state from the current sample.
    fn evaluate(&self) -> AdverseSelectionState {
        if self.total < u64::from(self.params.min_sample) {
            return AdverseSelectionState::InsufficientSample;
        }
        match self.mean_5s() {
            Some(mean) if mean < self.params.negative_threshold => AdverseSelectionState::Alarm,
            _ => AdverseSelectionState::Ok,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(min_sample: u32, window: u32, threshold: f64) -> AdverseSelectionParams {
        AdverseSelectionParams {
            min_sample,
            negative_threshold: threshold,
            window,
        }
    }

    #[test]
    fn insufficient_until_min_sample_then_judges() {
        let mut m = AdverseSelectionMonitor::new(params(3, 10, 0.0));
        // Below the sample floor: no transition, stays InsufficientSample.
        assert_eq!(m.observe_5s(-1.0), None);
        assert_eq!(m.observe_5s(-1.0), None);
        assert_eq!(m.state(), AdverseSelectionState::InsufficientSample);
        // The third (negative) markout crosses the floor and trips Alarm.
        assert_eq!(
            m.observe_5s(-1.0),
            Some((
                AdverseSelectionState::InsufficientSample,
                AdverseSelectionState::Alarm
            ))
        );
        assert_eq!(m.state(), AdverseSelectionState::Alarm);
    }

    #[test]
    fn positive_markouts_reach_ok() {
        let mut m = AdverseSelectionMonitor::new(params(2, 10, 0.0));
        assert_eq!(m.observe_5s(0.01), None); // still InsufficientSample
        // Second positive markout: mean > 0 ⇒ Ok.
        assert_eq!(
            m.observe_5s(0.01),
            Some((
                AdverseSelectionState::InsufficientSample,
                AdverseSelectionState::Ok
            ))
        );
    }

    #[test]
    fn alarm_then_recovers_to_ok_latched_once_each() {
        let mut m = AdverseSelectionMonitor::new(params(2, 4, 0.0));
        m.observe_5s(-0.02);
        let t = m.observe_5s(-0.02); // → Alarm
        assert_eq!(
            t,
            Some((
                AdverseSelectionState::InsufficientSample,
                AdverseSelectionState::Alarm
            ))
        );
        // A positive markout lifts the rolling mean above 0 → recover to Ok, once.
        // window [-0.02, -0.02, 0.05] mean ≈ +0.003 ≥ 0.
        let recover = m.observe_5s(0.05);
        assert_eq!(
            recover,
            Some((AdverseSelectionState::Alarm, AdverseSelectionState::Ok))
        );
        // No further transition while it stays Ok.
        assert_eq!(m.observe_5s(0.05), None);
    }

    #[test]
    fn rolling_window_forgets_old_markouts() {
        // Window of 2: only the last two markouts matter for the mean.
        let mut m = AdverseSelectionMonitor::new(params(1, 2, 0.0));
        m.observe_5s(-1.0); // mean -1 → Alarm
        assert_eq!(m.state(), AdverseSelectionState::Alarm);
        m.observe_5s(1.0); // window [-1, 1] mean 0 → not < 0 → Ok
        m.observe_5s(1.0); // window [1, 1] mean 1 → Ok
        assert_eq!(m.state(), AdverseSelectionState::Ok);
        assert_eq!(m.mean_5s(), Some(1.0));
    }
}
