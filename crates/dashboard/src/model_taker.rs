//! Dashboard state for the "Model taker" tile (§10): per-series fires/min +
//! last `p_up`, plus **PnL attribution split by driver** (maker-core vs momentum
//! vs late vs model) per series.
//!
//! The fires tile is fed by the orchestrator via
//! [`crate::DashboardHandle::set_model_taker`] on each model-taker decision. The
//! driver-PnL accumulator is fed two ways: [`crate::DashboardHandle::record_driver_fill`]
//! buffers each fill (tagged with its driver via `RiskManager::driver_of`), and a
//! window `Settlement` marks the buffered fills to the resolved outcome — so
//! `Σ over drivers` reconciles (approximately, modulo merges/rebates) to the
//! window's realized PnL. Display-only.

use std::collections::HashMap;

use analytics::DriverMatrix;
use core_types::{Dollars, Outcome, Price, Series, Size, TimestampMs, WindowId};
use engine::FillDriver;
use rust_decimal::Decimal;

/// Recent fire timestamps retained per series (for the per-minute rate).
const RECENT_CAP: usize = 64;
/// Window over which "fires/min" is measured (ms).
pub(crate) const RATE_WINDOW_MS: i64 = 60_000;

/// One model-taker decision handed to the dashboard (dashboard-owned).
#[derive(Debug, Clone)]
pub struct ModelTakerTick {
    /// Series the decision is for.
    pub series: Series,
    /// Decision time.
    pub ts: TimestampMs,
    /// The model's `p_up` for this window.
    pub p_up: f64,
    /// Whether a FAK was placed.
    pub fired: bool,
    /// `"fired"` or the suppression reason.
    pub reason: String,
}

/// Per-series fires tile state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ModelTakerSeries {
    pub(crate) recent_fires: Vec<i64>,
    pub(crate) fires_total: u64,
    pub(crate) suppressed_total: u64,
    pub(crate) last_p_up: f64,
    pub(crate) last_reason: String,
    pub(crate) last_ts: i64,
}

impl ModelTakerSeries {
    fn record(&mut self, tick: &ModelTakerTick) {
        let ts = tick.ts.as_millis();
        if tick.fired {
            self.recent_fires.push(ts);
            if self.recent_fires.len() > RECENT_CAP {
                self.recent_fires.remove(0);
            }
            self.fires_total += 1;
        } else {
            self.suppressed_total += 1;
        }
        self.last_p_up = tick.p_up;
        self.last_reason = tick.reason.clone();
        self.last_ts = ts;
    }

    /// Fires in the last [`RATE_WINDOW_MS`] (== fires/min for a 60 s window).
    pub(crate) fn fires_per_min(&self, now_ms: i64) -> usize {
        self.recent_fires
            .iter()
            .filter(|&&t| now_ms - t <= RATE_WINDOW_MS)
            .count()
    }
}

/// The fires tile state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ModelTakerState {
    pub(crate) by_series: HashMap<Series, ModelTakerSeries>,
    /// True once any decision has arrived (the model taker is running).
    pub(crate) active: bool,
}

impl ModelTakerState {
    /// Folds one decision into the tile state.
    pub(crate) fn record(&mut self, tick: &ModelTakerTick) {
        self.by_series.entry(tick.series).or_default().record(tick);
        self.active = true;
    }
}

// ---------------------------------------------------------------------------
// PnL attribution by driver (wraps the shared analytics::DriverMatrix)
// ---------------------------------------------------------------------------

/// The immediate per-`(series, driver)` activity kept alongside the realized-PnL
/// matrix — taker notional and fees are not in the matrix (which tracks realized
/// PnL + win rate), so they are accumulated here for the model-taker tile.
#[derive(Debug, Clone, Copy, Default)]
struct DriverActivity {
    fills: u64,
    taker_notional: Decimal,
    fees: Decimal,
}

/// One combined `(series, driver)` total for the model-taker tile: the activity
/// counters plus the matrix's realized PnL and win rate.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DriverTotals {
    pub(crate) fills: u64,
    /// Taker notional (`Σ price·size`) — zero for the maker-core driver.
    pub(crate) taker_notional: Decimal,
    /// Fees paid (taker only; maker fees are $0).
    pub(crate) fees: Decimal,
    /// Realized PnL, marked to each window's resolved outcome.
    pub(crate) realized_pnl: Decimal,
}

/// PnL-by-driver accumulator: the shared [`DriverMatrix`] (realized PnL + win
/// rate, marked at settlement) plus the immediate taker-notional/fee activity.
/// Owned per-mode by the dashboard, so it is populated for either namespace
/// independent of whether the model taker is running.
#[derive(Debug, Clone, Default)]
pub(crate) struct DriverPnlState {
    matrix: DriverMatrix,
    activity: HashMap<(Series, FillDriver), DriverActivity>,
}

impl DriverPnlState {
    /// Buffers a driver-tagged fill into the matrix (marked at settlement) and
    /// updates the immediate activity totals.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_fill(
        &mut self,
        window: WindowId,
        driver: FillDriver,
        outcome: Outcome,
        shares: Size,
        price: Price,
        fee: Dollars,
        ts: TimestampMs,
    ) {
        self.matrix
            .record_fill(window, driver, outcome, shares, price, fee, ts);
        let t = self.activity.entry((window.series, driver)).or_default();
        t.fills += 1;
        if driver != FillDriver::MakerCore {
            t.taker_notional += price.as_decimal() * shares.as_decimal();
        }
        t.fees += fee.as_decimal();
    }

    /// Marks a settled window's buffered fills to `outcome` in the matrix.
    pub(crate) fn settle(&mut self, window: WindowId, outcome: Outcome) {
        self.matrix.settle(window, outcome);
    }

    /// The underlying realized-PnL / win-rate matrix (for the §10 PnL matrix DTO).
    pub(crate) fn matrix(&self) -> &DriverMatrix {
        &self.matrix
    }

    /// Combined per-`(series, driver)` totals for the model-taker tile.
    pub(crate) fn totals(&self) -> HashMap<(Series, FillDriver), DriverTotals> {
        let cells = self.matrix.totals();
        let mut out: HashMap<(Series, FillDriver), DriverTotals> = HashMap::new();
        // Every key seen in either the activity or the settled matrix.
        let keys: std::collections::HashSet<(Series, FillDriver)> =
            self.activity.keys().chain(cells.keys()).copied().collect();
        for key in keys {
            let a = self.activity.get(&key).copied().unwrap_or_default();
            let cell = cells.get(&key).copied().unwrap_or_default();
            out.insert(
                key,
                DriverTotals {
                    fills: a.fills,
                    taker_notional: a.taker_notional,
                    fees: a.fees,
                    realized_pnl: cell.realized_pnl.as_decimal(),
                },
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{Asset, TickSize, WindowDuration};
    use rust_decimal::dec;

    fn series() -> Series {
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::M5,
        }
    }

    fn window() -> WindowId {
        WindowId {
            series: series(),
            open_time: TimestampMs::from_millis(0),
        }
    }

    fn tick(fired: bool) -> ModelTakerTick {
        ModelTakerTick {
            series: series(),
            ts: TimestampMs::from_millis(1_000),
            p_up: 0.9,
            fired,
            reason: if fired {
                "fired".into()
            } else {
                "below_theta".into()
            },
        }
    }

    #[test]
    fn fires_and_suppressions_counted() {
        let mut st = ModelTakerState::default();
        st.record(&tick(true));
        st.record(&tick(true));
        st.record(&tick(false));
        let s = st.by_series.get(&series()).unwrap();
        assert_eq!(s.fires_total, 2);
        assert_eq!(s.suppressed_total, 1);
        assert_eq!(s.fires_per_min(1_000), 2);
        assert!(st.active);
    }

    #[test]
    fn driver_pnl_marks_at_settlement() {
        let mut d = DriverPnlState::default();
        let px = |v| Price::on_grid(v, TickSize::T001).unwrap();
        let t = TimestampMs::from_millis(1_000);
        // Model buys 10 Up @ 0.80, fee 0.10; momentum buys 5 Down @ 0.30, fee 0.02.
        d.record_fill(
            window(),
            FillDriver::Model,
            Outcome::Up,
            Size::new(dec!(10)).unwrap(),
            px(dec!(0.80)),
            Dollars::new(dec!(0.10)),
            t,
        );
        d.record_fill(
            window(),
            FillDriver::Momentum,
            Outcome::Down,
            Size::new(dec!(5)).unwrap(),
            px(dec!(0.30)),
            Dollars::new(dec!(0.02)),
            t,
        );
        // Immediate activity totals.
        let totals = d.totals();
        let model = totals.get(&(series(), FillDriver::Model)).unwrap();
        assert_eq!(model.taker_notional, dec!(8.0)); // 10·0.80
        assert_eq!(model.fills, 1);

        // Window resolves Up: model won, momentum lost.
        d.settle(window(), Outcome::Up);
        let totals = d.totals();
        let model = totals.get(&(series(), FillDriver::Model)).unwrap();
        // payoff 10 − cost (8 + 0.10) = +1.90; won.
        assert_eq!(model.realized_pnl, dec!(1.90));
        let mom = totals.get(&(series(), FillDriver::Momentum)).unwrap();
        // Down lost: payoff 0 − cost (1.5 + 0.02) = −1.52
        assert_eq!(mom.realized_pnl, dec!(-1.52));
        // Win rate is surfaced via the matrix (the PnL-matrix DTO).
        let day = analytics::DayKey::from_ts(TimestampMs::from_millis(1_000));
        let cell = d.matrix().cell(series(), FillDriver::Model, day).unwrap();
        assert_eq!(cell.winning_fills, 1);
        assert_eq!(cell.resolved_fills, 1);
    }
}
