//! Cross-taker arbitration (CLAUDE.md §8): the per-window fire ledger and
//! per-asset precedence that decide, when the momentum taker and the model taker
//! would both fire the *same* window, which one wins.
//!
//! Both takers are active on all series; precedence only resolves a **conflict**
//! — the losing driver for an asset is suppressed on a window *only while* the
//! winning driver actually fired that same window within a short arbitration
//! window (the "fortress map": momentum wins BTC conflicts, model wins ETH). When
//! the winning driver has not recently fired the window, the loser fires freely.
//!
//! Owned by [`RiskManager`](crate::risk::RiskManager); pure and sans-IO (it takes
//! `now: TimestampMs` as a parameter and never reads a clock or a venue). A
//! [`FireLedger::default`] (empty precedence, zero window) suppresses **nobody**,
//! so the momentum taker's behaviour is unchanged whenever the model taker is off.

use std::collections::HashMap;

use core_types::{Asset, Series, TimestampMs, WindowId};

/// Which taker is speaking to the arbiter. Momentum and Model are the only two
/// takers that arbitrate against each other (§8); the late-window taker is a
/// distinct end-of-window regime and does not participate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TakerId {
    /// The fee-aware momentum taker.
    Momentum,
    /// The champion-model (dir10) taker.
    Model,
}

/// A refused fire: the winning driver for the window's asset already fired this
/// window `remaining_ms` before the arbitration window would clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArbitrationBlock {
    /// The driver that wins arbitration for this window's asset.
    pub winner: TakerId,
    /// Milliseconds left before the loser could fire this window.
    pub remaining_ms: i64,
}

/// One drained `(series, loser, winner) → count` arbitration-block tally, for the
/// dashboard's contention panel (§10): how often the losing driver was suppressed
/// by the winning driver on a series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArbitrationTally {
    /// The series the conflict occurred on.
    pub series: Series,
    /// The suppressed (losing) driver.
    pub loser: TakerId,
    /// The driver that won and suppressed the loser.
    pub winner: TakerId,
    /// How many fires the loser was blocked for on this series.
    pub count: u64,
}

/// Per-window fire ledger + per-asset precedence. [`RiskManager`] owns exactly one
/// and passes `&`/`&mut` into the takers' `decide`/`fire`.
///
/// [`Clone`] + `Default` so it slots into
/// [`RiskParams`](crate::risk::RiskParams)/`RiskParams::default` and the risk
/// manager's constructor.
#[derive(Debug, Clone, Default)]
pub struct FireLedger {
    /// A conflict on window `W` suppresses the loser only while the winner fired
    /// `W` within this many milliseconds. `0` ⇒ never suppress (the default).
    arbitration_window_ms: i64,
    /// asset → the driver that wins arbitration for that asset. Absent ⇒ momentum.
    precedence: HashMap<Asset, TakerId>,
    /// `(window, driver)` → last fire wall-ms.
    last_fire: HashMap<(WindowId, TakerId), i64>,
    /// `(series, loser, winner)` → cumulative block count, drained by
    /// [`drain_blocks`](FireLedger::drain_blocks) for the contention panel.
    block_tally: HashMap<(Series, TakerId, TakerId), u64>,
}

impl FireLedger {
    /// Builds a ledger with the given arbitration window and per-asset precedence.
    #[must_use]
    pub fn new(arbitration_window_ms: i64, precedence: HashMap<Asset, TakerId>) -> Self {
        Self {
            arbitration_window_ms,
            precedence,
            last_fire: HashMap::new(),
            block_tally: HashMap::new(),
        }
    }

    /// The driver that wins arbitration for `asset` (absent ⇒ [`TakerId::Momentum`]).
    #[must_use]
    pub fn winner_for(&self, asset: Asset) -> TakerId {
        self.precedence
            .get(&asset)
            .copied()
            .unwrap_or(TakerId::Momentum)
    }

    /// Admission check (from `decide`). The winner for the window's asset always
    /// passes; the loser is blocked while the winner fired this window within the
    /// arbitration window. Takes `&mut self` because a block is tallied into the
    /// contention counters (drained by [`drain_blocks`](FireLedger::drain_blocks)).
    ///
    /// # Errors
    /// [`ArbitrationBlock`] when `who` is the losing driver and the winning driver
    /// fired `window` less than `arbitration_window_ms` ago.
    pub fn check(
        &mut self,
        who: TakerId,
        window: WindowId,
        now: TimestampMs,
    ) -> Result<(), ArbitrationBlock> {
        let winner = self.winner_for(window.series.asset);
        if who == winner {
            return Ok(());
        }
        if let Some(&last) = self.last_fire.get(&(window, winner)) {
            let elapsed = now.as_millis() - last;
            if elapsed < self.arbitration_window_ms {
                *self
                    .block_tally
                    .entry((window.series, who, winner))
                    .or_insert(0) += 1;
                return Err(ArbitrationBlock {
                    winner,
                    remaining_ms: self.arbitration_window_ms - elapsed,
                });
            }
        }
        Ok(())
    }

    /// Drains the accumulated arbitration-block tallies (drain-and-report), one
    /// entry per non-zero `(series, loser, winner)`.
    pub fn drain_blocks(&mut self) -> Vec<ArbitrationTally> {
        std::mem::take(&mut self.block_tally)
            .into_iter()
            .map(|((series, loser, winner), count)| ArbitrationTally {
                series,
                loser,
                winner,
                count,
            })
            .collect()
    }

    /// Records a successful fire (from `fire`, after the venue accepts the place).
    /// Both takers record on every fire (symmetry + observability), though only the
    /// winner's timestamp is ever consulted by [`check`](FireLedger::check).
    pub fn record(&mut self, who: TakerId, window: WindowId, now: TimestampMs) {
        self.last_fire.insert((window, who), now.as_millis());
    }

    /// Drops the ledger rows for a window on close — bounded memory for a 24/7
    /// session.
    pub fn drop_window(&mut self, window: WindowId) {
        self.last_fire.retain(|(w, _), _| *w != window);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{Series, TimestampMs, WindowDuration};

    fn win(asset: Asset, open_ms: i64) -> WindowId {
        WindowId {
            series: Series {
                asset,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(open_ms),
        }
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn fortress() -> FireLedger {
        let mut p = HashMap::new();
        p.insert(Asset::Btc, TakerId::Momentum);
        p.insert(Asset::Eth, TakerId::Model);
        FireLedger::new(3_000, p)
    }

    #[test]
    fn winner_passes_loser_blocked_within_window() {
        let mut l = fortress();
        let btc = win(Asset::Btc, 0);
        // BTC → momentum wins. Momentum fires; within the window the model (loser) is blocked.
        assert!(l.check(TakerId::Momentum, btc, ts(0)).is_ok());
        l.record(TakerId::Momentum, btc, ts(0));
        let block = l.check(TakerId::Model, btc, ts(1_000)).unwrap_err();
        assert_eq!(block.winner, TakerId::Momentum);
        assert_eq!(block.remaining_ms, 2_000);
        // The winner is never blocked.
        assert!(l.check(TakerId::Momentum, btc, ts(1_000)).is_ok());
    }

    #[test]
    fn eth_precedence_is_the_model() {
        let mut l = fortress();
        let eth = win(Asset::Eth, 0);
        // ETH → model wins. Model fires; momentum (loser) is blocked in-window.
        l.record(TakerId::Model, eth, ts(0));
        assert!(l.check(TakerId::Model, eth, ts(500)).is_ok());
        let block = l.check(TakerId::Momentum, eth, ts(500)).unwrap_err();
        assert_eq!(block.winner, TakerId::Model);
    }

    #[test]
    fn loser_fires_once_window_elapses() {
        let mut l = fortress();
        let btc = win(Asset::Btc, 0);
        l.record(TakerId::Momentum, btc, ts(0));
        // Exactly at the window boundary the block clears (elapsed == window ⇒ not < window).
        assert!(l.check(TakerId::Model, btc, ts(3_000)).is_ok());
        assert!(l.check(TakerId::Model, btc, ts(5_000)).is_ok());
    }

    #[test]
    fn default_ledger_suppresses_nobody() {
        // Empty precedence ⇒ momentum wins everywhere; zero window ⇒ nobody blocked.
        let mut l = FireLedger::default();
        let btc = win(Asset::Btc, 0);
        let eth = win(Asset::Eth, 0);
        l.record(TakerId::Momentum, btc, ts(0));
        l.record(TakerId::Model, eth, ts(0));
        assert!(l.check(TakerId::Momentum, btc, ts(0)).is_ok());
        assert!(l.check(TakerId::Model, btc, ts(0)).is_ok()); // model loser but window 0 ⇒ ok
        assert!(l.check(TakerId::Momentum, eth, ts(0)).is_ok());
    }

    #[test]
    fn blocks_are_tallied_and_drained() {
        let mut l = fortress();
        let btc = win(Asset::Btc, 0);
        l.record(TakerId::Momentum, btc, ts(0));
        // Two blocked model checks within the window accumulate a tally.
        assert!(l.check(TakerId::Model, btc, ts(500)).is_err());
        assert!(l.check(TakerId::Model, btc, ts(600)).is_err());
        // An admitted check (winner) records nothing.
        assert!(l.check(TakerId::Momentum, btc, ts(700)).is_ok());
        let tallies = l.drain_blocks();
        assert_eq!(tallies.len(), 1);
        assert_eq!(tallies[0].loser, TakerId::Model);
        assert_eq!(tallies[0].winner, TakerId::Momentum);
        assert_eq!(tallies[0].series, btc.series);
        assert_eq!(tallies[0].count, 2);
        // Draining clears the tally.
        assert!(l.drain_blocks().is_empty());
    }

    #[test]
    fn drop_window_clears_rows() {
        let mut l = fortress();
        let btc = win(Asset::Btc, 0);
        l.record(TakerId::Momentum, btc, ts(0));
        l.drop_window(btc);
        // With the row gone, the loser is admitted immediately.
        assert!(l.check(TakerId::Model, btc, ts(100)).is_ok());
    }
}
