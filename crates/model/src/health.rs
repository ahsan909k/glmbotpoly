//! Composite model-health state machine (§8, §11) with hysteresis.
//!
//! The fair-value engine produces a probability and a set of input-quality
//! signals each tick; this machine folds those signals into the single gate
//! the engine reads — [`ModelHealth::Ready`] / [`Degraded`](ModelHealth::Degraded)
//! / [`Unreliable`](ModelHealth::Unreliable) — and emits a latched
//! [`HealthTransition`] whenever the tier changes (the bus carries that as
//! `Event::ModelHealth`; the continuous tier still rides every `ModelSnapshot`).
//!
//! Sans-IO, scheduler/`SkewMonitor` style: inputs go in with `now` as a field;
//! decisions come out as values. No clock, no logging, no allocation. The
//! machine is **per-asset** and outlives the per-window engine, so a window
//! rollover never resets its hysteresis.
//!
//! ## Why hysteresis
//!
//! The gate drives cancel-all / quote-pull, so a flapping gate is expensive.
//! Two mechanisms, mirroring [`timeutil`](../../timeutil)'s `SkewMonitor`:
//!
//! - **Dual magnitude bounds (dead zones).** Chainlink staleness enters at
//!   `chainlink_stale_ms` and only clears at the *lower* `chainlink_fresh_ms`;
//!   divergence enters at `*_bps` and clears at the lower `*_clear_bps`. An
//!   input oscillating around one threshold cannot oscillate the gate.
//! - **Dwell timers.** A noisy worsening condition (divergence, fast-lead-lost)
//!   must persist for its dwell before it is believed — this is what filters the
//!   legitimate transient ~80 bps Binance-lead divergence (it collapses within
//!   ~3 s, far under the 10 s divergence dwell). Any *improvement* to a better
//!   tier must hold for `exit_dwell_ms` before the gate loosens.
//!
//! **Worsen fast, ease slow.** An unambiguous failure (Chainlink stale, no
//! anchor, model not warmed) tightens the gate immediately — like
//! `FeedHealth::Stale` on a disconnect. Every recovery is debounced by the exit
//! dwell. (Divergence and fast-lead-lost are debounced even on the way in, but
//! that is a *confidence* filter for an inherently noisy signal, not a
//! defense-loosening delay.)

use core_types::{AnchorSource, DurationMs, ModelHealth, ModelHealthReason, TimestampMs};
use thiserror::Error;

/// Tuning for a [`HealthMonitor`]. All model-crate constants except
/// `chainlink_stale_ms`, which mirrors `feeds.rtds_stale_after_ms` at the binary
/// boundary (the vol/basis precedent — broader config exposure is deferred).
/// Defaults are journal-calibratable starting points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthParams {
    /// Chainlink age (ms) at or beyond which the resolution feed is stale
    /// (→ UNRELIABLE). The 1 Hz RTDS cadence means 5 s is five missed ticks, so
    /// the threshold is itself the enter debounce. Must be > `chainlink_fresh_ms`.
    pub chainlink_stale_ms: i64,
    /// Chainlink age (ms) at or below which staleness clears. The dead zone
    /// `(fresh, stale)` holds the latch — the anti-flap at the boundary.
    pub chainlink_fresh_ms: i64,
    /// |corrected-fast − Chainlink| divergence (bps) that, sustained, declares
    /// DEGRADED. Below the hard bound.
    pub divergence_soft_bps: f64,
    /// Soft divergence clears below this (bps). Must be ≤ `divergence_soft_bps`.
    pub divergence_soft_clear_bps: f64,
    /// Sustained divergence (bps) that declares UNRELIABLE — a wrong basis or a
    /// broken feed. The stable basis is ≈ 13 bps and the ~2.7 s Binance lead
    /// makes *instantaneous* divergence up to ~80 bps legitimate, so this trips
    /// only on a *persistent* breach.
    pub divergence_hard_bps: f64,
    /// Hard divergence clears below this (bps). Must be ≤ `divergence_hard_bps`.
    pub divergence_hard_clear_bps: f64,
    /// How long (ms) a divergence breach must hold before it is believed —
    /// filters the transient lead-induced divergence (collapses within ~3 s).
    pub divergence_dwell_ms: i64,
    /// How long (ms) the price must anchor on Chainlink alone (fast feed stale
    /// or basis cold) before DEGRADED — short, since losing the fast lead is the
    /// adverse-selection exposure (§8), but non-zero to debounce anchor flicker.
    pub fast_lead_dwell_ms: i64,
    /// How long (ms) an improvement to a better tier must hold before the gate
    /// loosens (recovery debounce; "ease slow").
    pub exit_dwell_ms: i64,
}

impl Default for HealthParams {
    fn default() -> Self {
        Self {
            chainlink_stale_ms: 5_000,
            chainlink_fresh_ms: 3_000,
            divergence_soft_bps: 30.0,
            divergence_soft_clear_bps: 20.0,
            divergence_hard_bps: 50.0,
            divergence_hard_clear_bps: 40.0,
            divergence_dwell_ms: 10_000,
            fast_lead_dwell_ms: 2_000,
            exit_dwell_ms: 3_000,
        }
    }
}

/// Why a [`HealthMonitor`] refused to construct.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum HealthError {
    /// `chainlink_stale_ms` was not strictly greater than `chainlink_fresh_ms`,
    /// or a bound was negative (no dead zone).
    #[error("chainlink staleness needs 0 <= fresh_ms < stale_ms, got fresh {fresh}, stale {stale}")]
    BadChainlinkBounds {
        /// The offending `chainlink_fresh_ms`.
        fresh: i64,
        /// The offending `chainlink_stale_ms`.
        stale: i64,
    },
    /// A divergence bound was non-finite/negative, a clear bound exceeded its
    /// enter bound, or the soft bound exceeded the hard bound.
    #[error("divergence bounds invalid: {0}")]
    BadDivergenceBounds(&'static str),
    /// A dwell was negative.
    #[error("dwell durations must be >= 0, got {0} ms")]
    BadDwell(i64),
}

/// The per-asset input signals the machine evaluates each tick. Built from one
/// `FairValue` (see `FairValue::health_inputs`) — no clock of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthInputs {
    /// Evaluation time (the fair value's `computed_at`).
    pub now: TimestampMs,
    /// Whether a probability was produced (strike, σ, and an anchor all present).
    pub has_probability: bool,
    /// Which input the value anchored on; `None` when neither feed is usable.
    pub anchor: Option<AnchorSource>,
    /// Age of the latest live Chainlink price; `None` if one was never seen.
    pub chainlink_age: Option<DurationMs>,
    /// |corrected-fast − Chainlink| divergence (bps); `None` when not computable
    /// (no fresh corrected-fast and Chainlink to compare).
    pub divergence_bps: Option<f64>,
}

impl HealthInputs {
    /// Whether any usable anchor exists.
    #[must_use]
    const fn has_anchor(&self) -> bool {
        self.anchor.is_some()
    }

    /// Whether the value is anchored on Chainlink alone (the fast lead is lost).
    #[must_use]
    const fn chainlink_only(&self) -> bool {
        matches!(self.anchor, Some(AnchorSource::Chainlink))
    }
}

/// A gate-tier change, emitted exactly once per transition (latched).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthTransition {
    /// Tier before the change.
    pub from: ModelHealth,
    /// Tier after the change.
    pub to: ModelHealth,
    /// Cause behind the new tier (`reason.tier() == to`).
    pub reason: ModelHealthReason,
    /// When the change was detected.
    pub at: TimestampMs,
}

/// The sans-IO composite-health machine. Feed it [`HealthInputs`] each tick;
/// act on the returned tier and any [`HealthTransition`].
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    params: HealthParams,
    health: ModelHealth,
    reason: ModelHealthReason,
    /// Chainlink staleness latch (dual-bound hysteresis; no dwell — immediate).
    chainlink_stale: bool,
    /// When the soft-divergence breach began (continuously), or `None`.
    divergence_soft_since: Option<TimestampMs>,
    /// When the hard-divergence breach began (continuously), or `None`.
    divergence_hard_since: Option<TimestampMs>,
    /// When the Chainlink-only anchor began (continuously), or `None`.
    fast_lead_lost_since: Option<TimestampMs>,
    /// When the current improvement-to-a-better-tier began, or `None`.
    recovering_since: Option<TimestampMs>,
}

impl HealthMonitor {
    /// Builds a monitor in the initial `Unreliable`/`Warming` state (nothing has
    /// warmed up yet), validating `params`.
    ///
    /// # Errors
    /// A [`HealthError`] naming the first invalid parameter.
    pub fn new(params: HealthParams) -> Result<Self, HealthError> {
        if params.chainlink_fresh_ms < 0 || params.chainlink_stale_ms <= params.chainlink_fresh_ms {
            return Err(HealthError::BadChainlinkBounds {
                fresh: params.chainlink_fresh_ms,
                stale: params.chainlink_stale_ms,
            });
        }
        for v in [
            params.divergence_soft_bps,
            params.divergence_soft_clear_bps,
            params.divergence_hard_bps,
            params.divergence_hard_clear_bps,
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(HealthError::BadDivergenceBounds(
                    "bounds must be finite and >= 0",
                ));
            }
        }
        if params.divergence_soft_clear_bps > params.divergence_soft_bps {
            return Err(HealthError::BadDivergenceBounds(
                "soft_clear_bps must be <= soft_bps",
            ));
        }
        if params.divergence_hard_clear_bps > params.divergence_hard_bps {
            return Err(HealthError::BadDivergenceBounds(
                "hard_clear_bps must be <= hard_bps",
            ));
        }
        if params.divergence_soft_bps > params.divergence_hard_bps {
            return Err(HealthError::BadDivergenceBounds(
                "soft_bps must be <= hard_bps",
            ));
        }
        for dwell in [
            params.divergence_dwell_ms,
            params.fast_lead_dwell_ms,
            params.exit_dwell_ms,
        ] {
            if dwell < 0 {
                return Err(HealthError::BadDwell(dwell));
            }
        }
        Ok(Self {
            params,
            health: ModelHealth::Unreliable,
            reason: ModelHealthReason::Warming,
            chainlink_stale: false,
            divergence_soft_since: None,
            divergence_hard_since: None,
            fast_lead_lost_since: None,
            recovering_since: None,
        })
    }

    /// The current health tier.
    #[must_use]
    pub const fn health(&self) -> ModelHealth {
        self.health
    }

    /// The current cause behind the tier.
    #[must_use]
    pub const fn reason(&self) -> ModelHealthReason {
        self.reason
    }

    /// Folds one tick of inputs into the gate. Returns the (possibly unchanged)
    /// tier and cause, plus a [`HealthTransition`] **only** when the tier
    /// changed (latched — one per episode).
    pub fn update(
        &mut self,
        inputs: HealthInputs,
    ) -> (ModelHealth, ModelHealthReason, Option<HealthTransition>) {
        let now = inputs.now;
        self.maintain_latches(now, &inputs);

        let candidate = self.candidate_reason(now, &inputs);
        let candidate_tier = candidate.tier();
        let from = self.health;

        if candidate_tier > self.health {
            // Worse than current: tighten immediately (the candidate already
            // cleared its own dwell, if any).
            self.health = candidate_tier;
            self.reason = candidate;
            self.recovering_since = None;
        } else if candidate_tier == self.health {
            // Same tier: the cause may refine, but the gate does not move and
            // no recovery is in progress.
            self.reason = candidate;
            self.recovering_since = None;
        } else {
            // Better than current: ease only after the improvement has held for
            // the exit dwell.
            let since = *self.recovering_since.get_or_insert(now);
            if now.as_millis() - since.as_millis() >= self.params.exit_dwell_ms {
                self.health = candidate_tier;
                self.reason = candidate;
                self.recovering_since = None;
            }
            // Else: hold the current (worse) tier and cause until the dwell elapses.
        }

        let transition = (self.health != from).then_some(HealthTransition {
            from,
            to: self.health,
            reason: self.reason,
            at: now,
        });
        (self.health, self.reason, transition)
    }

    /// Advances each per-condition latch from this tick's inputs.
    fn maintain_latches(&mut self, now: TimestampMs, inputs: &HealthInputs) {
        // Chainlink staleness: dual-bound hysteresis, no dwell. A never-seen
        // feed is *warming*, not stale, so the latch stays clear.
        match inputs.chainlink_age {
            None => self.chainlink_stale = false,
            Some(age) => {
                let ms = age.as_millis();
                if ms >= self.params.chainlink_stale_ms {
                    self.chainlink_stale = true;
                } else if ms <= self.params.chainlink_fresh_ms {
                    self.chainlink_stale = false;
                }
                // Dead zone (fresh, stale): hold.
            }
        }

        latch(
            &mut self.divergence_hard_since,
            now,
            inputs.divergence_bps,
            self.params.divergence_hard_bps,
            self.params.divergence_hard_clear_bps,
        );
        latch(
            &mut self.divergence_soft_since,
            now,
            inputs.divergence_bps,
            self.params.divergence_soft_bps,
            self.params.divergence_soft_clear_bps,
        );

        if inputs.has_anchor() && inputs.chainlink_only() {
            self.fast_lead_lost_since.get_or_insert(now);
        } else {
            self.fast_lead_lost_since = None;
        }
    }

    /// The worst applicable cause given current latches and dwell satisfaction
    /// (worst-first: UNRELIABLE causes, then DEGRADED, then `Nominal`).
    fn candidate_reason(&self, now: TimestampMs, inputs: &HealthInputs) -> ModelHealthReason {
        let dwell_met = |since: Option<TimestampMs>, dwell: i64| {
            since.is_some_and(|s| now.as_millis() - s.as_millis() >= dwell)
        };
        if inputs.chainlink_age.is_none() {
            ModelHealthReason::Warming
        } else if !inputs.has_anchor() {
            ModelHealthReason::NoAnchor
        } else if self.chainlink_stale {
            ModelHealthReason::ChainlinkStale
        } else if dwell_met(self.divergence_hard_since, self.params.divergence_dwell_ms) {
            ModelHealthReason::DivergenceHard
        } else if !inputs.has_probability {
            ModelHealthReason::Warming
        } else if dwell_met(self.divergence_soft_since, self.params.divergence_dwell_ms) {
            ModelHealthReason::DivergenceSoft
        } else if dwell_met(self.fast_lead_lost_since, self.params.fast_lead_dwell_ms) {
            ModelHealthReason::FastLeadLost
        } else {
            ModelHealthReason::Nominal
        }
    }
}

/// Advances a since-latch with magnitude hysteresis: set (preserving the entry
/// time) at or above `enter`, clear below `clear`, hold in the dead zone, clear
/// when the value is uncomputable (absence of evidence is not evidence).
fn latch(
    since: &mut Option<TimestampMs>,
    now: TimestampMs,
    value: Option<f64>,
    enter: f64,
    clear: f64,
) {
    match value {
        Some(v) if v >= enter => {
            since.get_or_insert(now);
        }
        Some(v) if v < clear => *since = None,
        Some(_) => {}
        None => *since = None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> HealthMonitor {
        HealthMonitor::new(HealthParams::default()).unwrap()
    }

    /// Fully-healthy inputs at `now` (ms): probability present, basis-corrected
    /// anchor, fresh Chainlink, near-zero divergence.
    fn ready(now: i64) -> HealthInputs {
        HealthInputs {
            now: TimestampMs::from_millis(now),
            has_probability: true,
            anchor: Some(AnchorSource::BinanceCorrected),
            chainlink_age: Some(DurationMs::from_millis(200)),
            divergence_bps: Some(2.0),
        }
    }

    /// Drives `m` to steady [`ModelHealth::Ready`] from its initial Unreliable,
    /// returning the time just after the warmup transition lands.
    fn warm_to_ready(m: &mut HealthMonitor) -> i64 {
        let (_, _, t0) = m.update(ready(0));
        assert!(t0.is_none(), "warmup should not snap straight to Ready");
        // Ready lands once the improvement holds for the exit dwell.
        let at = HealthParams::default().exit_dwell_ms;
        let (h, r, t) = m.update(ready(at));
        assert_eq!(h, ModelHealth::Ready);
        assert_eq!(r, ModelHealthReason::Nominal);
        assert_eq!(
            t,
            Some(HealthTransition {
                from: ModelHealth::Unreliable,
                to: ModelHealth::Ready,
                reason: ModelHealthReason::Nominal,
                at: TimestampMs::from_millis(at),
            })
        );
        at
    }

    #[test]
    fn starts_unreliable_warming() {
        let m = monitor();
        assert_eq!(m.health(), ModelHealth::Unreliable);
        assert_eq!(m.reason(), ModelHealthReason::Warming);
    }

    #[test]
    fn warming_to_ready_when_fully_anchored() {
        let mut m = monitor();
        // Holds Unreliable through the exit dwell, then exactly one transition.
        let (h, _, t) = m.update(ready(0));
        assert_eq!(h, ModelHealth::Unreliable);
        assert!(t.is_none());
        let (h, _, t) = m.update(ready(2_999));
        assert_eq!(h, ModelHealth::Unreliable);
        assert!(t.is_none());
        let (h, _, t) = m.update(ready(3_000));
        assert_eq!(h, ModelHealth::Ready);
        assert!(t.is_some());
    }

    #[test]
    fn warming_when_chainlink_seen_but_no_probability() {
        let mut m = monitor();
        // Chainlink fresh but σ/strike not ready → no probability, no anchor.
        let inputs = HealthInputs {
            now: TimestampMs::from_millis(0),
            has_probability: false,
            anchor: None,
            chainlink_age: Some(DurationMs::from_millis(200)),
            divergence_bps: None,
        };
        let (h, r, _) = m.update(inputs);
        // No anchor takes priority over warming; both are Unreliable.
        assert_eq!(h, ModelHealth::Unreliable);
        assert_eq!(r, ModelHealthReason::NoAnchor);

        // Anchor present (Chainlink) but still no probability → Warming.
        let inputs = HealthInputs {
            now: TimestampMs::from_millis(100),
            has_probability: false,
            anchor: Some(AnchorSource::Chainlink),
            chainlink_age: Some(DurationMs::from_millis(200)),
            divergence_bps: None,
        };
        let (h, r, _) = m.update(inputs);
        assert_eq!(h, ModelHealth::Unreliable);
        assert_eq!(r, ModelHealthReason::Warming);
    }

    #[test]
    fn no_anchor_is_immediate_unreliable() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        let inputs = HealthInputs {
            anchor: None,
            has_probability: false,
            ..ready(t + 100)
        };
        let (h, r, tr) = m.update(inputs);
        assert_eq!(h, ModelHealth::Unreliable);
        assert_eq!(r, ModelHealthReason::NoAnchor);
        assert_eq!(
            tr.map(|t| (t.from, t.to)),
            Some((ModelHealth::Ready, ModelHealth::Unreliable))
        );
    }

    #[test]
    fn ready_to_unreliable_on_chainlink_stale() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // One stale Chainlink reading trips immediately (no enter dwell).
        let inputs = HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(5_100)),
            ..ready(t + 100)
        };
        let (h, r, tr) = m.update(inputs);
        assert_eq!(h, ModelHealth::Unreliable);
        assert_eq!(r, ModelHealthReason::ChainlinkStale);
        assert!(tr.is_some());
    }

    #[test]
    fn unreliable_to_ready_on_chainlink_recovery() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        let stale = HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(5_100)),
            ..ready(t + 100)
        };
        m.update(stale);
        // Fresh again, but recovery waits for the exit dwell.
        let recover_at = t + 200;
        let (h, ..) = m.update(ready(recover_at));
        assert_eq!(h, ModelHealth::Unreliable, "no instant clear");
        let (h, r, tr) = m.update(ready(recover_at + 3_000));
        assert_eq!(h, ModelHealth::Ready);
        assert_eq!(r, ModelHealthReason::Nominal);
        assert_eq!(
            tr.map(|t| (t.from, t.to)),
            Some((ModelHealth::Unreliable, ModelHealth::Ready))
        );
    }

    #[test]
    fn ready_to_degraded_on_fast_lead_lost() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // Anchor on Chainlink-only: needs the fast-lead dwell (2 s) before DEGRADED.
        let chainlink_only = |now: i64| HealthInputs {
            anchor: Some(AnchorSource::Chainlink),
            divergence_bps: None,
            ..ready(now)
        };
        let (h, ..) = m.update(chainlink_only(t + 100));
        assert_eq!(h, ModelHealth::Ready, "brief Chainlink-only stays Ready");
        let (h, ..) = m.update(chainlink_only(t + 2_099));
        assert_eq!(h, ModelHealth::Ready);
        let (h, r, tr) = m.update(chainlink_only(t + 2_100));
        assert_eq!(h, ModelHealth::Degraded);
        assert_eq!(r, ModelHealthReason::FastLeadLost);
        assert_eq!(
            tr.map(|t| (t.from, t.to)),
            Some((ModelHealth::Ready, ModelHealth::Degraded))
        );
    }

    #[test]
    fn degraded_to_ready_on_fast_lead_regained() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        let chainlink_only = |now: i64| HealthInputs {
            anchor: Some(AnchorSource::Chainlink),
            divergence_bps: None,
            ..ready(now)
        };
        m.update(chainlink_only(t + 100));
        let (h, ..) = m.update(chainlink_only(t + 2_100));
        assert_eq!(h, ModelHealth::Degraded);
        // Fast feed back (corrected anchor): recovery honors the exit dwell.
        let back = t + 3_000;
        let (h, ..) = m.update(ready(back));
        assert_eq!(h, ModelHealth::Degraded, "no instant clear");
        let (h, _, tr) = m.update(ready(back + 3_000));
        assert_eq!(h, ModelHealth::Ready);
        assert_eq!(
            tr.map(|t| (t.from, t.to)),
            Some((ModelHealth::Degraded, ModelHealth::Ready))
        );
    }

    #[test]
    fn ready_to_degraded_on_soft_divergence() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // 35 bps: in the soft band (30–50). Needs the 10 s divergence dwell.
        let soft = |now: i64| HealthInputs {
            divergence_bps: Some(35.0),
            ..ready(now)
        };
        let (h, ..) = m.update(soft(t + 100));
        assert_eq!(h, ModelHealth::Ready, "below the dwell, no degrade");
        let (h, ..) = m.update(soft(t + 10_099));
        assert_eq!(h, ModelHealth::Ready);
        let (h, r, tr) = m.update(soft(t + 10_100));
        assert_eq!(h, ModelHealth::Degraded);
        assert_eq!(r, ModelHealthReason::DivergenceSoft);
        assert!(tr.is_some());
    }

    #[test]
    fn degraded_to_unreliable_on_hard_divergence() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // Sustained soft → Degraded.
        let soft = |now: i64| HealthInputs {
            divergence_bps: Some(35.0),
            ..ready(now)
        };
        m.update(soft(t + 100));
        let (h, ..) = m.update(soft(t + 10_100));
        assert_eq!(h, ModelHealth::Degraded);
        // Divergence climbs into the hard band; needs its own dwell from here.
        let hard_start = t + 11_000;
        let hard = |now: i64| HealthInputs {
            divergence_bps: Some(60.0),
            ..ready(now)
        };
        let (h, ..) = m.update(hard(hard_start));
        assert_eq!(h, ModelHealth::Degraded, "hard dwell not yet met");
        let (h, r, tr) = m.update(hard(hard_start + 10_000));
        assert_eq!(h, ModelHealth::Unreliable);
        assert_eq!(r, ModelHealthReason::DivergenceHard);
        assert_eq!(
            tr.map(|t| (t.from, t.to)),
            Some((ModelHealth::Degraded, ModelHealth::Unreliable))
        );
    }

    #[test]
    fn transient_lead_divergence_up_to_80bps_stays_ready() {
        // The Decisions-Log-pinned property: an 80 bps lead spike that collapses
        // within ~3 s never completes the 10 s divergence dwell.
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        for i in 0..30 {
            // 80 bps for 3 s (30 × 100 ms), then back to baseline.
            let now = t + 100 + i * 100;
            let (h, _, tr) = m.update(HealthInputs {
                divergence_bps: Some(80.0),
                ..ready(now)
            });
            assert_eq!(h, ModelHealth::Ready, "transient must not degrade");
            assert!(tr.is_none());
        }
        // Divergence collapses; still Ready, latches cleared.
        let (h, ..) = m.update(ready(t + 100 + 3_000));
        assert_eq!(h, ModelHealth::Ready);
    }

    #[test]
    fn sustained_soft_divergence_degrades_then_clears_without_oscillation() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        let mut transitions = 0;
        let soft = |now: i64| HealthInputs {
            divergence_bps: Some(35.0),
            ..ready(now)
        };
        // Hold 35 bps for 11 s: exactly one transition (Ready→Degraded).
        for i in 0..=110 {
            let (_, _, tr) = m.update(soft(t + 100 + i * 100));
            if tr.is_some() {
                transitions += 1;
            }
        }
        assert_eq!(m.health(), ModelHealth::Degraded);
        assert_eq!(transitions, 1, "no flapping while held in the soft band");

        // Clear it and hold: exactly one more transition (Degraded→Ready).
        let clear_start = t + 100 + 110 * 100 + 100;
        for i in 0..=40 {
            let (_, _, tr) = m.update(ready(clear_start + i * 100));
            if tr.is_some() {
                transitions += 1;
            }
        }
        assert_eq!(m.health(), ModelHealth::Ready);
        assert_eq!(transitions, 2, "exactly degrade + recover");
    }

    #[test]
    fn chainlink_dead_zone_prevents_recovery_flap() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // Trip stale.
        let (h, _, _) = m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(5_100)),
            ..ready(t + 100)
        });
        assert_eq!(h, ModelHealth::Unreliable);
        // Oscillate age in the dead zone (4 000 ms) and just over the trip
        // (5 100 ms): never clears, never an extra transition.
        let mut transitions = 0;
        for i in 0..40 {
            let age = if i % 2 == 0 { 4_000 } else { 5_100 };
            let (h, _, tr) = m.update(HealthInputs {
                chainlink_age: Some(DurationMs::from_millis(age)),
                ..ready(t + 200 + i * 100)
            });
            assert_eq!(h, ModelHealth::Unreliable, "dead-zone dip must not clear");
            if tr.is_some() {
                transitions += 1;
            }
        }
        assert_eq!(transitions, 0, "no flapping in the dead zone");
        // Only a sustained drop below `fresh` (3 000) for the exit dwell recovers.
        let rec = t + 200 + 40 * 100;
        m.update(ready(rec));
        let (h, _, tr) = m.update(ready(rec + 3_000));
        assert_eq!(h, ModelHealth::Ready);
        assert!(tr.is_some());
    }

    #[test]
    fn recovery_resets_on_relapse() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(5_100)),
            ..ready(t + 100)
        });
        assert_eq!(m.health(), ModelHealth::Unreliable);
        // Fresh for 2 s (under the 3 s exit dwell)…
        m.update(ready(t + 200));
        let (h, ..) = m.update(ready(t + 2_200));
        assert_eq!(h, ModelHealth::Unreliable, "not yet recovered");
        // …then one stale relapse resets the recovery clock.
        m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(5_100)),
            ..ready(t + 2_300)
        });
        // Fresh again: must wait a *full* exit dwell from here.
        m.update(ready(t + 2_400));
        let (h, ..) = m.update(ready(t + 2_400 + 2_999));
        assert_eq!(h, ModelHealth::Unreliable, "relapse restarted the dwell");
        let (h, _, tr) = m.update(ready(t + 2_400 + 3_000));
        assert_eq!(h, ModelHealth::Ready);
        assert!(tr.is_some());
    }

    #[test]
    fn worsens_fast_eases_slow() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // Worsen: chainlink stale trips on the very next tick (no dwell).
        let (h, _, tr) = m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(5_100)),
            ..ready(t + 100)
        });
        assert_eq!(h, ModelHealth::Unreliable);
        assert!(tr.is_some(), "worsening is immediate");
        // Ease: even with perfect inputs, recovery needs the whole exit dwell.
        let (h, _, tr) = m.update(ready(t + 200));
        assert_eq!(h, ModelHealth::Unreliable);
        assert!(tr.is_none(), "easing is debounced");
        let (h, _, tr) = m.update(ready(t + 200 + 3_000));
        assert_eq!(h, ModelHealth::Ready);
        assert!(tr.is_some());
    }

    #[test]
    fn only_one_transition_per_gate_change() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // First stale tick transitions; subsequent stale ticks are silent.
        let (_, _, tr) = m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(6_000)),
            ..ready(t + 100)
        });
        assert!(tr.is_some());
        for i in 1..10 {
            let (h, _, tr) = m.update(HealthInputs {
                chainlink_age: Some(DurationMs::from_millis(6_000)),
                ..ready(t + 100 + i * 100)
            });
            assert_eq!(h, ModelHealth::Unreliable);
            assert!(tr.is_none(), "latched: no repeats within the same tier");
        }
    }

    #[test]
    fn reason_refines_within_gate_without_re_emitting() {
        let mut m = monitor();
        let t = warm_to_ready(&mut m);
        // Enter Unreliable via Chainlink staleness.
        let (_, r, tr) = m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(6_000)),
            ..ready(t + 100)
        });
        assert_eq!(r, ModelHealthReason::ChainlinkStale);
        assert!(tr.is_some());
        // Now the anchor also vanishes: same tier (Unreliable), cause refines to
        // NoAnchor, but no new transition is emitted.
        let (h, r, tr) = m.update(HealthInputs {
            chainlink_age: Some(DurationMs::from_millis(6_000)),
            anchor: None,
            has_probability: false,
            ..ready(t + 200)
        });
        assert_eq!(h, ModelHealth::Unreliable);
        assert_eq!(r, ModelHealthReason::NoAnchor);
        assert!(tr.is_none(), "same-tier refinement does not re-emit");
    }

    #[test]
    fn params_defaults_sane() {
        let p = HealthParams::default();
        assert!(p.chainlink_fresh_ms >= 0 && p.chainlink_fresh_ms < p.chainlink_stale_ms);
        assert!(p.divergence_soft_clear_bps <= p.divergence_soft_bps);
        assert!(p.divergence_hard_clear_bps <= p.divergence_hard_bps);
        assert!(p.divergence_soft_bps <= p.divergence_hard_bps);
        // The soft band sits above the legit ~80 bps transient is NOT relied on;
        // the dwell filters transients. But the hard trip must not sit absurdly
        // low: keep it at/above the historical 50 bps.
        assert!(p.divergence_hard_bps >= 50.0);
        assert!(p.divergence_dwell_ms >= 0 && p.fast_lead_dwell_ms >= 0 && p.exit_dwell_ms >= 0);
        // Construction accepts the defaults.
        assert!(HealthMonitor::new(p).is_ok());
    }

    #[test]
    fn rejects_invalid_params() {
        // Inverted Chainlink bounds.
        assert!(matches!(
            HealthMonitor::new(HealthParams {
                chainlink_fresh_ms: 6_000,
                chainlink_stale_ms: 5_000,
                ..HealthParams::default()
            }),
            Err(HealthError::BadChainlinkBounds { .. })
        ));
        // Clear above enter.
        assert!(matches!(
            HealthMonitor::new(HealthParams {
                divergence_soft_clear_bps: 40.0,
                divergence_soft_bps: 30.0,
                ..HealthParams::default()
            }),
            Err(HealthError::BadDivergenceBounds(_))
        ));
        // Negative dwell.
        assert!(matches!(
            HealthMonitor::new(HealthParams {
                exit_dwell_ms: -1,
                ..HealthParams::default()
            }),
            Err(HealthError::BadDwell(_))
        ));
    }
}
