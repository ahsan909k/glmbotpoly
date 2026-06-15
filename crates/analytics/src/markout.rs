//! Per-passive-fill markouts against model fair (CLAUDE.md §8/§10/§14).
//!
//! For every passive (Maker) fill we measure how the model fair value of the
//! bought outcome moved over 1s / 5s / 30s and to expiry. **Sign = fair-value
//! move**, oriented by the position so positive means the fill *aged well*:
//!
//! ```text
//! markout_h = position_sign · (fair_O(t_fill + h) − fair_O(t_fill))
//! ```
//!
//! where `fair_O = p_up` for an Up fill and `1 − p_up` for a Down fill, and
//! `position_sign = +1` for a BUY, `−1` for a SELL. Persistently negative
//! 5-second markouts are the adverse-selection alarm: our resting quote was
//! picked off just before the underlying moved against us.
//!
//! # How it stays replay-exact
//!
//! Each active window retains the full series of `(ts, p_up)` model snapshots
//! it has seen (hard-capped, then dropped wholesale at resolution). All markouts
//! for a window are computed in one pass **at resolution**, where the terminal
//! outcome is ground truth, using last-observation-carried-forward (LOCF) over
//! the retained ring. Because LOCF over the full ring at resolution equals LOCF
//! over the ring as it stood when any earlier deadline passed (later snapshots
//! are all past that deadline), the result is identical to an incremental
//! finalisation — but with no pending bookkeeping. Everything keys off
//! event-carried timestamps, so a journal replay reproduces it bit-for-bit.
//!
//! ## Decisions baked in (operator-confirmed)
//!
//! - **Near-close clamp-to-settlement**: a finite horizon whose deadline falls
//!   past the window's close uses the resolved outcome (0/1) as its terminal
//!   fair — so a fill seconds before close still yields a 5s markout for the
//!   alarm rather than being silently dropped (flagged [`MarkoutSource::Settled`]).
//! - **NoAnchor drop**: a fill with no model snapshot at or before it (the model
//!   had not produced a fair for the window yet) is dropped from markout to avoid
//!   lookahead bias; it still counts for PnL attribution.

use std::collections::HashMap;
use std::collections::VecDeque;

use core_types::{DurationMs, Outcome, Side, TimestampMs, WindowId};
use serde::{Deserialize, Serialize};

/// The horizons at which a passive fill's markout is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkoutHorizon {
    /// One second after the fill.
    S1,
    /// Five seconds after the fill — the adverse-selection signal.
    S5,
    /// Thirty seconds after the fill.
    S30,
    /// At window expiry (terminal fair from the resolved outcome).
    Expiry,
}

impl MarkoutHorizon {
    /// The three finite horizons, in order.
    pub const FINITE: [Self; 3] = [Self::S1, Self::S5, Self::S30];

    /// Offset from the fill for a finite horizon; `None` for [`Self::Expiry`].
    #[must_use]
    pub const fn offset_ms(self) -> Option<i64> {
        match self {
            Self::S1 => Some(1_000),
            Self::S5 => Some(5_000),
            Self::S30 => Some(30_000),
            Self::Expiry => None,
        }
    }
}

/// Where a markout's terminal fair came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkoutSource {
    /// A model-fair snapshot at or before the horizon deadline (LOCF).
    Snapshot,
    /// The resolved outcome (0/1) — expiry, or a finite horizon past close.
    Settled,
}

/// One finalised markout value plus where its terminal fair was sourced.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarkoutPoint {
    /// Signed markout (probability units); positive = aged well.
    pub value: f64,
    /// Provenance of the terminal fair.
    pub source: MarkoutSource,
}

/// A passive fill's markouts, emitted once its window resolves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FillMarkout {
    /// Window the fill belonged to.
    pub window: WindowId,
    /// Outcome the fill bought/sold.
    pub outcome: Outcome,
    /// Our side of the fill.
    pub side: Side,
    /// Fill execution time.
    pub ts_fill: TimestampMs,
    /// Model fair of the bought outcome at the fill (LOCF).
    pub fair_at_fill: f64,
    /// Markout 1s after the fill, when measurable.
    pub s1: Option<MarkoutPoint>,
    /// Markout 5s after the fill — feeds the adverse-selection alarm.
    pub s5: Option<MarkoutPoint>,
    /// Markout 30s after the fill, when measurable.
    pub s30: Option<MarkoutPoint>,
    /// Markout at expiry (terminal outcome vs fair at fill).
    pub expiry: Option<MarkoutPoint>,
}

impl FillMarkout {
    /// The markout at a given horizon, if measured.
    #[must_use]
    pub fn at(&self, horizon: MarkoutHorizon) -> Option<MarkoutPoint> {
        match horizon {
            MarkoutHorizon::S1 => self.s1,
            MarkoutHorizon::S5 => self.s5,
            MarkoutHorizon::S30 => self.s30,
            MarkoutHorizon::Expiry => self.expiry,
        }
    }
}

/// A passive fill awaiting its window's resolution to be marked out.
#[derive(Debug, Clone, PartialEq)]
struct PendingMarkout {
    outcome: Outcome,
    side: Side,
    ts_fill: TimestampMs,
}

/// Per-window markout state: the retained model-fair snapshot ring plus the
/// passive fills awaiting resolution.
#[derive(Debug, Clone, Default, PartialEq)]
struct WindowMarkout {
    /// `(ts, p_up)` snapshots in arrival order (≈ ascending ts).
    snaps: VecDeque<(TimestampMs, f64)>,
    /// Passive fills registered in fill order.
    pending: Vec<PendingMarkout>,
}

/// Buffers model-fair snapshots and passive fills per active window, then
/// computes every fill's markouts at the window's resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkoutEngine {
    ring_max: usize,
    windows: HashMap<WindowId, WindowMarkout>,
    no_anchor_dropped: u64,
}

impl MarkoutEngine {
    /// A fresh engine with the given snapshot-ring hard cap.
    #[must_use]
    pub fn new(ring_max: usize) -> Self {
        Self {
            ring_max: ring_max.max(1),
            windows: HashMap::new(),
            no_anchor_dropped: 0,
        }
    }

    /// Records a model-fair snapshot (`p_up`) for a window. Drops the oldest
    /// snapshot once the ring exceeds its cap (a backstop; LOCF degrades
    /// gracefully).
    pub fn on_snapshot(&mut self, window: WindowId, ts: TimestampMs, fair_up: f64) {
        let w = self.windows.entry(window).or_default();
        w.snaps.push_back((ts, fair_up));
        while w.snaps.len() > self.ring_max {
            w.snaps.pop_front();
        }
    }

    /// Registers a passive fill to be marked out at its window's resolution.
    /// Only Maker fills should be passed (taker fills are not adverse-selection
    /// subjects, §10).
    pub fn on_fill(
        &mut self,
        window: WindowId,
        outcome: Outcome,
        side: Side,
        ts_fill: TimestampMs,
    ) {
        self.windows
            .entry(window)
            .or_default()
            .pending
            .push(PendingMarkout {
                outcome,
                side,
                ts_fill,
            });
    }

    /// Resolves a window: computes every pending passive fill's markouts and
    /// drops the window's state. Fills with no model anchor at or before them
    /// are dropped (counted in [`Self::no_anchor_dropped`]). Returns the
    /// [`FillMarkout`]s in fill order.
    pub fn on_resolved(
        &mut self,
        window: WindowId,
        winner: Outcome,
        close_time: TimestampMs,
    ) -> Vec<FillMarkout> {
        let Some(state) = self.windows.remove(&window) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for p in &state.pending {
            let Some(fair_up_at_fill) = locf(&state.snaps, p.ts_fill) else {
                self.no_anchor_dropped += 1;
                continue;
            };
            let fair_at_fill = fair_of(p.outcome, fair_up_at_fill);
            let sign = position_sign(p.side);
            let terminal = terminal_fair(p.outcome, winner);

            let finite = MarkoutHorizon::FINITE.map(|h| {
                let offset = h.offset_ms().unwrap_or(0);
                let deadline = p.ts_fill.saturating_add(DurationMs::from_millis(offset));
                let (fair_at_h, source) = if deadline.as_millis() > close_time.as_millis() {
                    // Past close: no live market — clamp to the settled outcome.
                    (terminal, MarkoutSource::Settled)
                } else {
                    // LOCF always returns at least the fill anchor (ts ≤ t_fill ≤ deadline).
                    let fair_up = locf(&state.snaps, deadline).unwrap_or(fair_up_at_fill);
                    (fair_of(p.outcome, fair_up), MarkoutSource::Snapshot)
                };
                MarkoutPoint {
                    value: sign * (fair_at_h - fair_at_fill),
                    source,
                }
            });

            let expiry = MarkoutPoint {
                value: sign * (terminal - fair_at_fill),
                source: MarkoutSource::Settled,
            };

            out.push(FillMarkout {
                window,
                outcome: p.outcome,
                side: p.side,
                ts_fill: p.ts_fill,
                fair_at_fill,
                s1: Some(finite[0]),
                s5: Some(finite[1]),
                s30: Some(finite[2]),
                expiry: Some(expiry),
            });
        }
        out
    }

    /// How many passive fills have been dropped for lacking a model anchor.
    #[must_use]
    pub fn no_anchor_dropped(&self) -> u64 {
        self.no_anchor_dropped
    }

    /// Number of windows with live markout state (active, pre-resolution).
    #[must_use]
    pub fn active_windows(&self) -> usize {
        self.windows.len()
    }
}

/// Model fair of one outcome from `p_up`.
fn fair_of(outcome: Outcome, fair_up: f64) -> f64 {
    match outcome {
        Outcome::Up => fair_up,
        Outcome::Down => 1.0 - fair_up,
    }
}

/// Terminal (settled) fair of `outcome` given the `winner`: 1.0 if it won, else 0.0.
fn terminal_fair(outcome: Outcome, winner: Outcome) -> f64 {
    if outcome == winner { 1.0 } else { 0.0 }
}

/// Position orientation: +1 for a long (BUY), −1 for a short (SELL).
fn position_sign(side: Side) -> f64 {
    match side {
        Side::Buy => 1.0,
        Side::Sell => -1.0,
    }
}

/// Last-observation-carried-forward: `p_up` of the latest snapshot at or before
/// `deadline`. `None` when no snapshot is at or before it.
fn locf(snaps: &VecDeque<(TimestampMs, f64)>, deadline: TimestampMs) -> Option<f64> {
    snaps
        .iter()
        .rev()
        .find(|(ts, _)| ts.as_millis() <= deadline.as_millis())
        .map(|(_, fair)| *fair)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{Asset, Series, WindowDuration};

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = OPEN_MS + 300_000; // 5-minute window

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    /// A single passive fill marked out, given a snapshot script.
    fn mark_one(
        snaps: &[(i64, f64)],
        outcome: Outcome,
        side: Side,
        fill_ms: i64,
        winner: Outcome,
    ) -> Option<FillMarkout> {
        let mut e = MarkoutEngine::new(1024);
        for (ms, fair) in snaps {
            e.on_snapshot(window(), ts(*ms), *fair);
        }
        e.on_fill(window(), outcome, side, ts(fill_ms));
        e.on_resolved(window(), winner, ts(CLOSE_MS))
            .into_iter()
            .next()
    }

    #[test]
    fn buy_up_positive_when_fair_rises() {
        // Snapshot 0.50 at fill, 0.55 at +5s → +0.05 markout.
        let fm = mark_one(
            &[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.55)],
            Outcome::Up,
            Side::Buy,
            OPEN_MS,
            Outcome::Up,
        )
        .expect("a markout");
        assert!((fm.fair_at_fill - 0.50).abs() < 1e-12);
        let s5 = fm.s5.expect("s5");
        assert!((s5.value - 0.05).abs() < 1e-12);
        assert_eq!(s5.source, MarkoutSource::Snapshot);
    }

    #[test]
    fn buy_up_negative_when_picked_off() {
        // Bought Up at 0.50, fair drops to 0.45 over 5s → −0.05 (adverse).
        let fm = mark_one(
            &[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.45)],
            Outcome::Up,
            Side::Buy,
            OPEN_MS,
            Outcome::Up,
        )
        .expect("a markout");
        assert!((fm.s5.expect("s5").value - (-0.05)).abs() < 1e-12);
    }

    #[test]
    fn buy_down_uses_one_minus_p_up() {
        // Bought Down; p_up falls 0.50→0.40 so fair_Down rises 0.50→0.60 → +0.10.
        let fm = mark_one(
            &[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.40)],
            Outcome::Down,
            Side::Buy,
            OPEN_MS,
            Outcome::Down,
        )
        .expect("a markout");
        assert!((fm.fair_at_fill - 0.50).abs() < 1e-12);
        assert!((fm.s5.expect("s5").value - 0.10).abs() < 1e-12);
    }

    #[test]
    fn sell_flips_the_sign() {
        // SELL Up; fair rises 0.50→0.55 → markout −0.05 (a sale ages well when fair falls).
        let fm = mark_one(
            &[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.55)],
            Outcome::Up,
            Side::Sell,
            OPEN_MS,
            Outcome::Up,
        )
        .expect("a markout");
        assert!((fm.s5.expect("s5").value - (-0.05)).abs() < 1e-12);
    }

    #[test]
    fn locf_uses_latest_snapshot_at_or_before_deadline() {
        // Deadline exactly on a snapshot ts uses that snapshot; a later one is ignored.
        let fm = mark_one(
            &[
                (OPEN_MS, 0.50),
                (OPEN_MS + 5_000, 0.60), // exactly at the 5s deadline
                (OPEN_MS + 6_000, 0.99), // after — must not be used
            ],
            Outcome::Up,
            Side::Buy,
            OPEN_MS,
            Outcome::Up,
        )
        .expect("a markout");
        assert!((fm.s5.expect("s5").value - 0.10).abs() < 1e-12);
    }

    #[test]
    fn expiry_uses_terminal_outcome() {
        // Bought Up at 0.50; window resolves Up → expiry fair 1.0 → +0.50.
        let up = mark_one(
            &[(OPEN_MS, 0.50)],
            Outcome::Up,
            Side::Buy,
            OPEN_MS,
            Outcome::Up,
        )
        .expect("a markout");
        let exp = up.expiry.expect("expiry");
        assert!((exp.value - 0.50).abs() < 1e-12);
        assert_eq!(exp.source, MarkoutSource::Settled);
        // Resolves Down instead → expiry fair 0.0 → −0.50.
        let down = mark_one(
            &[(OPEN_MS, 0.50)],
            Outcome::Up,
            Side::Buy,
            OPEN_MS,
            Outcome::Down,
        )
        .expect("a markout");
        assert!((down.expiry.expect("expiry").value - (-0.50)).abs() < 1e-12);
    }

    #[test]
    fn near_close_horizon_clamps_to_settlement() {
        // Fill 2s before close: the 5s/30s deadlines are past close → clamp to outcome.
        let fill_ms = CLOSE_MS - 2_000;
        let fm = mark_one(
            &[(OPEN_MS, 0.50), (fill_ms, 0.80)],
            Outcome::Up,
            Side::Buy,
            fill_ms,
            Outcome::Up, // won → terminal 1.0
        )
        .expect("a markout");
        // 1s deadline (still before close) uses the snapshot fair 0.80 → 0.0 move.
        let s1 = fm.s1.expect("s1");
        assert_eq!(s1.source, MarkoutSource::Snapshot);
        assert!((s1.value - 0.0).abs() < 1e-12);
        // 5s deadline is past close → clamp: terminal 1.0 − fair_at_fill 0.80 = +0.20.
        let s5 = fm.s5.expect("s5");
        assert_eq!(s5.source, MarkoutSource::Settled);
        assert!((s5.value - 0.20).abs() < 1e-12);
    }

    #[test]
    fn fill_before_any_snapshot_is_dropped_no_anchor() {
        let mut e = MarkoutEngine::new(1024);
        // First snapshot is AFTER the fill — no anchor at or before the fill.
        e.on_fill(window(), Outcome::Up, Side::Buy, ts(OPEN_MS));
        e.on_snapshot(window(), ts(OPEN_MS + 1_000), 0.5);
        let out = e.on_resolved(window(), Outcome::Up, ts(CLOSE_MS));
        assert!(out.is_empty(), "no markout without an anchor");
        assert_eq!(e.no_anchor_dropped(), 1);
    }

    #[test]
    fn markouts_returned_in_fill_order() {
        let mut e = MarkoutEngine::new(1024);
        e.on_snapshot(window(), ts(OPEN_MS), 0.5);
        e.on_fill(window(), Outcome::Up, Side::Buy, ts(OPEN_MS + 10));
        e.on_fill(window(), Outcome::Down, Side::Buy, ts(OPEN_MS + 20));
        let out = e.on_resolved(window(), Outcome::Up, ts(CLOSE_MS));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].ts_fill, ts(OPEN_MS + 10));
        assert_eq!(out[1].ts_fill, ts(OPEN_MS + 20));
    }

    #[test]
    fn ring_hard_cap_drops_oldest() {
        let mut e = MarkoutEngine::new(2);
        e.on_snapshot(window(), ts(OPEN_MS), 0.10);
        e.on_snapshot(window(), ts(OPEN_MS + 1_000), 0.20);
        e.on_snapshot(window(), ts(OPEN_MS + 2_000), 0.30);
        // Cap 2: the OPEN_MS snapshot was evicted, so a fill at OPEN_MS has no
        // anchor at or before it any more.
        e.on_fill(window(), Outcome::Up, Side::Buy, ts(OPEN_MS));
        let out = e.on_resolved(window(), Outcome::Up, ts(CLOSE_MS));
        assert!(out.is_empty());
        assert_eq!(e.no_anchor_dropped(), 1);
    }

    #[test]
    fn resolving_unknown_window_is_empty() {
        let mut e = MarkoutEngine::new(8);
        assert!(
            e.on_resolved(window(), Outcome::Up, ts(CLOSE_MS))
                .is_empty()
        );
    }
}
