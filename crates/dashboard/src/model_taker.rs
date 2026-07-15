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

use core_types::{Dollars, Outcome, Price, Series, Size, TimestampMs, WindowId};
use engine::FillDriver;
use rust_decimal::Decimal;

/// Recent fire timestamps retained per series (for the per-minute rate).
const RECENT_CAP: usize = 64;
/// Window over which "fires/min" is measured (ms).
pub(crate) const RATE_WINDOW_MS: i64 = 60_000;
/// Max windows buffered awaiting settlement (bounds memory if one never resolves).
const PENDING_WINDOWS_CAP: usize = 64;

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
// PnL attribution by driver
// ---------------------------------------------------------------------------

/// One driver-tagged fill, buffered until its window settles.
#[derive(Debug, Clone, Copy)]
struct DriverFill {
    driver: FillDriver,
    outcome: Outcome,
    shares: Size,
    price: Price,
    fee: Dollars,
}

/// Running per-(series, driver) totals.
#[derive(Debug, Clone, Default)]
pub(crate) struct DriverTotals {
    pub(crate) fills: u64,
    /// Taker notional (`Σ price·size`) — zero for the maker-core driver.
    pub(crate) taker_notional: Decimal,
    /// Fees paid (taker only; maker fees are $0).
    pub(crate) fees: Decimal,
    /// Realized PnL, marked to each window's resolved outcome.
    pub(crate) realized_pnl: Decimal,
}

/// PnL-by-driver accumulator (buffers fills per window, marks them at settlement).
#[derive(Debug, Clone, Default)]
pub(crate) struct DriverPnlState {
    pending: HashMap<WindowId, Vec<DriverFill>>,
    pub(crate) totals: HashMap<(Series, FillDriver), DriverTotals>,
}

impl DriverPnlState {
    /// Buffers a driver-tagged fill and updates the immediate activity totals.
    pub(crate) fn record_fill(
        &mut self,
        window: WindowId,
        driver: FillDriver,
        outcome: Outcome,
        shares: Size,
        price: Price,
        fee: Dollars,
    ) {
        self.pending.entry(window).or_default().push(DriverFill {
            driver,
            outcome,
            shares,
            price,
            fee,
        });
        // Bound memory: if a window never settles, evict the oldest buffered one.
        if self.pending.len() > PENDING_WINDOWS_CAP
            && let Some(oldest) = self.pending.keys().min_by_key(|w| w.open_time).copied()
        {
            self.pending.remove(&oldest);
        }
        let t = self.totals.entry((window.series, driver)).or_default();
        t.fills += 1;
        if driver != FillDriver::MakerCore {
            t.taker_notional += price.as_decimal() * shares.as_decimal();
        }
        t.fees += fee.as_decimal();
    }

    /// Marks a settled window's buffered fills to `outcome`, folding the realized
    /// PnL into each driver's total, then drops the window's buffer.
    pub(crate) fn settle(&mut self, window: WindowId, outcome: Outcome) {
        let Some(fills) = self.pending.remove(&window) else {
            return;
        };
        for f in fills {
            let won = f.outcome == outcome;
            let payoff = if won { f.shares.as_decimal() } else { Decimal::ZERO };
            let cost = f.shares.as_decimal() * f.price.as_decimal() + f.fee.as_decimal();
            let marked = payoff - cost;
            self.totals
                .entry((window.series, f.driver))
                .or_default()
                .realized_pnl += marked;
        }
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
            reason: if fired { "fired".into() } else { "below_theta".into() },
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
        // Model buys 10 Up @ 0.80, fee 0.10; momentum buys 5 Down @ 0.30, fee 0.02.
        d.record_fill(
            window(),
            FillDriver::Model,
            Outcome::Up,
            Size::new(dec!(10)).unwrap(),
            px(dec!(0.80)),
            Dollars::new(dec!(0.10)),
        );
        d.record_fill(
            window(),
            FillDriver::Momentum,
            Outcome::Down,
            Size::new(dec!(5)).unwrap(),
            px(dec!(0.30)),
            Dollars::new(dec!(0.02)),
        );
        // Immediate activity totals.
        let model = d.totals.get(&(series(), FillDriver::Model)).unwrap();
        assert_eq!(model.taker_notional, dec!(8.0)); // 10·0.80
        assert_eq!(model.fills, 1);

        // Window resolves Up: model won, momentum lost.
        d.settle(window(), Outcome::Up);
        let model = d.totals.get(&(series(), FillDriver::Model)).unwrap();
        // payoff 10 − cost (8 + 0.10) = +1.90
        assert_eq!(model.realized_pnl, dec!(1.90));
        let mom = d.totals.get(&(series(), FillDriver::Momentum)).unwrap();
        // Down lost: payoff 0 − cost (1.5 + 0.02) = −1.52
        assert_eq!(mom.realized_pnl, dec!(-1.52));
        // Buffer dropped.
        assert!(d.pending.is_empty());
    }
}
