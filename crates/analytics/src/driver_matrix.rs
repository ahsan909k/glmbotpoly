//! Per-driver × per-series realized-PnL matrix (CLAUDE.md §10 PnL-by-driver).
//!
//! A settled window's fills are attributed to the strategy that placed them
//! (maker-core / momentum / late / model — an [`engine::FillDriver`]) and marked
//! to the window's resolved outcome, so the operator can see which strategy is
//! actually earning per series and per UTC day.
//!
//! This is **deliberately not** part of the replay-derivable [`Analytics`](crate::Analytics)
//! fold: the driver of a fill is a *live-only* annotation resolved from the risk
//! manager's owned order sets (`RiskManager::driver_of`), which a journal replay
//! cannot reconstruct (a journaled [`Fill`](core_types::Fill) carries no client
//! id). So the driver is passed **explicitly** into [`DriverMatrix::record_fill`],
//! keeping the analytics engine's replay-equality (`rebuild == live`) intact — the
//! matrix is fed alongside it, never from it.
//!
//! It buffers each fill per window and marks the buffer to the resolved outcome
//! at settlement, mirroring the exact `payoff − cost − fee` mark the dashboard's
//! two-bucket panel uses. Bounded for 24/7: buffered windows are capped, and
//! `(series, driver, day)` cells older than ~31 days are pruned.

use std::collections::HashMap;

use core_types::{Decimal, Dollars, Outcome, Price, Series, Size, TimestampMs, WindowId};
use engine::FillDriver;

use crate::datekey::DayKey;

/// Days of `(series, driver, day)` cells retained before pruning (24/7 bound).
const CELL_RETENTION_DAYS: i64 = 31;
/// Windows buffered awaiting settlement before the oldest is evicted.
const PENDING_WINDOWS_CAP: usize = 256;

/// One driver-tagged fill, buffered until its window settles.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DriverFill {
    driver: FillDriver,
    outcome: Outcome,
    shares: Size,
    price: Price,
    fee: Dollars,
    /// Fill time — the UTC day this fill's realized PnL is booked under.
    ts: TimestampMs,
}

/// One `(series, driver, day)` cell of realized activity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DriverCell {
    /// Realized PnL, marked to each window's resolved outcome.
    pub realized_pnl: Dollars,
    /// Fills that have resolved (been marked out at settlement).
    pub resolved_fills: u32,
    /// Resolved fills whose bought outcome won.
    pub winning_fills: u32,
}

impl DriverCell {
    /// Fraction of resolved fills that won, `None` with no resolved fill.
    #[must_use]
    pub fn win_pct(&self) -> Option<f64> {
        if self.resolved_fills == 0 {
            None
        } else {
            Some(f64::from(self.winning_fills) / f64::from(self.resolved_fills))
        }
    }
}

/// Per-`(series, driver, day)` realized-PnL matrix. Fed a driver-tagged fill via
/// [`record_fill`](Self::record_fill) and a window `(window, outcome)` at
/// settlement via [`settle`](Self::settle); marks the window's buffered fills to
/// the outcome and folds them into the cells.
#[derive(Debug, Clone, Default)]
pub struct DriverMatrix {
    pending: HashMap<WindowId, Vec<DriverFill>>,
    cells: HashMap<(Series, FillDriver, DayKey), DriverCell>,
}

impl DriverMatrix {
    /// An empty matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffers one driver-tagged fill (marked to its outcome at settlement).
    #[allow(clippy::too_many_arguments)]
    pub fn record_fill(
        &mut self,
        window: WindowId,
        driver: FillDriver,
        outcome: Outcome,
        shares: Size,
        price: Price,
        fee: Dollars,
        ts: TimestampMs,
    ) {
        self.pending.entry(window).or_default().push(DriverFill {
            driver,
            outcome,
            shares,
            price,
            fee,
            ts,
        });
        // Bound memory: if a window never settles, evict the oldest buffered one.
        if self.pending.len() > PENDING_WINDOWS_CAP
            && let Some(oldest) = self.pending.keys().min_by_key(|w| w.open_time).copied()
        {
            self.pending.remove(&oldest);
        }
    }

    /// Marks a settled window's buffered fills to `outcome`, folding each into its
    /// `(series, driver, day)` cell, then drops the window's buffer and prunes any
    /// cells older than the retention horizon (measured against the fills' day).
    pub fn settle(&mut self, window: WindowId, outcome: Outcome) {
        let Some(fills) = self.pending.remove(&window) else {
            return;
        };
        let series = window.series;
        let mut newest_day = DayKey::from_days(i64::MIN);
        for f in fills {
            let day = DayKey::from_ts(f.ts);
            newest_day = newest_day.max(day);
            let won = f.outcome == outcome;
            let payoff = if won {
                f.shares.as_decimal()
            } else {
                Decimal::ZERO
            };
            let cost = f.shares.as_decimal() * f.price.as_decimal() + f.fee.as_decimal();
            let cell = self.cells.entry((series, f.driver, day)).or_default();
            cell.realized_pnl = cell.realized_pnl + Dollars::new(payoff - cost);
            cell.resolved_fills += 1;
            if won {
                cell.winning_fills += 1;
            }
        }
        self.prune(newest_day);
    }

    /// Drops cells older than the retention horizon relative to `newest_day`.
    fn prune(&mut self, newest_day: DayKey) {
        let cutoff = newest_day.as_days() - CELL_RETENTION_DAYS;
        self.cells.retain(|(_, _, day), _| day.as_days() >= cutoff);
    }

    /// The cell for a `(series, driver, day)`, if any.
    #[must_use]
    pub fn cell(&self, series: Series, driver: FillDriver, day: DayKey) -> Option<&DriverCell> {
        self.cells.get(&(series, driver, day))
    }

    /// Aggregates every retained cell into a `(series, driver) → DriverCell` map
    /// across all buffered days (the dashboard/digest matrix view).
    #[must_use]
    pub fn totals(&self) -> HashMap<(Series, FillDriver), DriverCell> {
        let mut out: HashMap<(Series, FillDriver), DriverCell> = HashMap::new();
        for (&(series, driver, _day), cell) in &self.cells {
            let acc = out.entry((series, driver)).or_default();
            acc.realized_pnl = acc.realized_pnl + cell.realized_pnl;
            acc.resolved_fills += cell.resolved_fills;
            acc.winning_fills += cell.winning_fills;
        }
        out
    }

    /// Aggregates the retained cells restricted to `day` into `(series, driver)`
    /// totals (used by the digest for a single UTC day).
    #[must_use]
    pub fn totals_for_day(&self, day: DayKey) -> HashMap<(Series, FillDriver), DriverCell> {
        let mut out: HashMap<(Series, FillDriver), DriverCell> = HashMap::new();
        for (&(series, driver, cell_day), cell) in &self.cells {
            if cell_day == day {
                *out.entry((series, driver)).or_default() = *cell;
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core_types::{Asset, TickSize, WindowDuration};
    use rust_decimal::dec;

    const OPEN_MS: i64 = 1_781_481_600_000; // 2026-06-15T00:00:00Z

    fn series() -> Series {
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::M5,
        }
    }

    fn window(open_ms: i64) -> WindowId {
        WindowId {
            series: series(),
            open_time: TimestampMs::from_millis(open_ms),
        }
    }

    fn px(v: Decimal) -> Price {
        Price::on_grid(v, TickSize::T001).unwrap()
    }

    fn record(
        m: &mut DriverMatrix,
        driver: FillDriver,
        outcome: Outcome,
        shares: Decimal,
        price: Decimal,
        fee: Decimal,
        ts: i64,
    ) {
        m.record_fill(
            window(OPEN_MS),
            driver,
            outcome,
            Size::new(shares).unwrap(),
            px(price),
            Dollars::new(fee),
            TimestampMs::from_millis(ts),
        );
    }

    #[test]
    fn marks_fills_to_outcome_and_tracks_win_pct() {
        let mut m = DriverMatrix::new();
        // Model buys 10 Up @ 0.80, fee 0.10; momentum buys 5 Down @ 0.30, fee 0.02.
        record(
            &mut m,
            FillDriver::Model,
            Outcome::Up,
            dec!(10),
            dec!(0.80),
            dec!(0.10),
            OPEN_MS + 1,
        );
        record(
            &mut m,
            FillDriver::Momentum,
            Outcome::Down,
            dec!(5),
            dec!(0.30),
            dec!(0.02),
            OPEN_MS + 2,
        );
        m.settle(window(OPEN_MS), Outcome::Up);

        let day = DayKey::from_ts(TimestampMs::from_millis(OPEN_MS));
        let model = m.cell(series(), FillDriver::Model, day).unwrap();
        // payoff 10 − cost (8 + 0.10) = +1.90; won.
        assert_eq!(model.realized_pnl, Dollars::new(dec!(1.90)));
        assert_eq!(model.resolved_fills, 1);
        assert_eq!(model.winning_fills, 1);
        assert_eq!(model.win_pct(), Some(1.0));

        let mom = m.cell(series(), FillDriver::Momentum, day).unwrap();
        // Down lost: payoff 0 − cost (1.5 + 0.02) = −1.52.
        assert_eq!(mom.realized_pnl, Dollars::new(dec!(-1.52)));
        assert_eq!(mom.win_pct(), Some(0.0));
        // Buffer dropped on settle.
        assert!(m.pending.is_empty());
    }

    #[test]
    fn totals_aggregate_across_days() {
        let mut m = DriverMatrix::new();
        let next_day = OPEN_MS + 86_400_000;
        // Two windows on two days, both MakerCore Up wins.
        record(
            &mut m,
            FillDriver::MakerCore,
            Outcome::Up,
            dec!(10),
            dec!(0.50),
            dec!(0),
            OPEN_MS + 1,
        );
        m.settle(window(OPEN_MS), Outcome::Up);
        m.record_fill(
            window(next_day),
            FillDriver::MakerCore,
            Outcome::Up,
            Size::new(dec!(10)).unwrap(),
            px(dec!(0.50)),
            Dollars::ZERO,
            TimestampMs::from_millis(next_day + 1),
        );
        m.settle(window(next_day), Outcome::Up);
        let totals = m.totals();
        let cell = totals.get(&(series(), FillDriver::MakerCore)).unwrap();
        // Each: payoff 10 − cost 5 = +5. Two windows → +10, 2 resolved, 2 won.
        assert_eq!(cell.realized_pnl, Dollars::new(dec!(10)));
        assert_eq!(cell.resolved_fills, 2);
        assert_eq!(cell.winning_fills, 2);
    }

    #[test]
    fn two_instances_fed_the_same_sequence_agree() {
        // The matrix is a deterministic fold: a "live" incremental instance and an
        // "offline" replay of the same (fill, settle) sequence produce identical
        // totals — the live==offline parity the dashboard relies on.
        let build = || {
            let mut m = DriverMatrix::new();
            record(
                &mut m,
                FillDriver::Model,
                Outcome::Up,
                dec!(10),
                dec!(0.80),
                dec!(0.10),
                OPEN_MS + 1,
            );
            record(
                &mut m,
                FillDriver::Momentum,
                Outcome::Down,
                dec!(5),
                dec!(0.30),
                dec!(0.02),
                OPEN_MS + 2,
            );
            record(
                &mut m,
                FillDriver::MakerCore,
                Outcome::Up,
                dec!(20),
                dec!(0.45),
                dec!(0),
                OPEN_MS + 3,
            );
            m.settle(window(OPEN_MS), Outcome::Up);
            m
        };
        assert_eq!(build().totals(), build().totals());
    }

    #[test]
    fn settling_unknown_window_is_a_noop() {
        let mut m = DriverMatrix::new();
        m.settle(window(OPEN_MS), Outcome::Up);
        assert!(m.totals().is_empty());
    }

    #[test]
    fn old_cells_are_pruned_by_retention_horizon() {
        let mut m = DriverMatrix::new();
        // A very old window's fill (day 0-ish) then a fresh one 60 days later.
        record(
            &mut m,
            FillDriver::MakerCore,
            Outcome::Up,
            dec!(10),
            dec!(0.50),
            dec!(0),
            1,
        );
        m.settle(window(0), Outcome::Up);
        let recent = OPEN_MS;
        m.record_fill(
            window(recent),
            FillDriver::MakerCore,
            Outcome::Up,
            Size::new(dec!(10)).unwrap(),
            px(dec!(0.50)),
            Dollars::ZERO,
            TimestampMs::from_millis(recent + 1),
        );
        m.settle(window(recent), Outcome::Up);
        // The ancient cell (day 0) is far more than 31 days before the recent day.
        let old_day = DayKey::from_days(0);
        assert!(m.cell(series(), FillDriver::MakerCore, old_day).is_none());
        let new_day = DayKey::from_ts(TimestampMs::from_millis(recent));
        assert!(m.cell(series(), FillDriver::MakerCore, new_day).is_some());
    }
}
