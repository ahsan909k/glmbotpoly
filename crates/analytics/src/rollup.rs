//! Daily and per-series aggregates — the data behind the dashboard's §10
//! series-comparison table, the view the operator uses to narrow the six series
//! down to the best two or three.
//!
//! Roll-ups are keyed by `(Mode, Series, DayKey)`; within one per-mode
//! [`Analytics`](crate::Analytics) instance the mode is fixed, so the store keys
//! by `(Series, DayKey)` and stamps the mode on every output. A
//! [`SeriesComparisonRow`] is the cross-day aggregate for one series with its
//! adverse-selection health and minimum-sample warning attached.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;

use core_types::{Decimal, Dollars, Mode, Series};
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::attribution::WindowAttribution;
use crate::datekey::DayKey;
use crate::health::AdverseSelectionState;

/// Number of 5-second-markout histogram bins.
pub const MARKOUT_BIN_COUNT: usize = 17;

/// The 16 interior edges of the 5s-markout histogram (probability units),
/// symmetric about `0` and finest near zero (where the adverse-selection signal
/// lives). Bin `0` is the open underflow `(-inf, -0.10)`; interior bin `i`
/// (`1..=15`) is the half-open `[EDGES[i-1], EDGES[i])`; bin `16` is the open
/// overflow `[0.10, +inf)`. `0.0` lands in the center bin `[-0.001, 0.001)`.
/// Counts are exactly additive across days, so a windowed (today/7d/all)
/// aggregate is the element-wise sum of its per-day histograms.
pub const MARKOUT_BIN_EDGES: [f64; MARKOUT_BIN_COUNT - 1] = [
    -0.10, -0.05, -0.03, -0.02, -0.01, -0.005, -0.002, -0.001, 0.001, 0.002, 0.005, 0.01, 0.02,
    0.03, 0.05, 0.10,
];

/// The histogram bin index `0..MARKOUT_BIN_COUNT` a markout falls in: the count
/// of interior edges `<= v`. Deterministic under replay (identical `f64`
/// comparisons over the identical sample order).
fn markout_bin(v: f64) -> usize {
    let mut i = 0usize;
    while i < MARKOUT_BIN_EDGES.len() && v >= MARKOUT_BIN_EDGES[i] {
        i += 1;
    }
    i
}

/// A summary of the 5-second-markout distribution for one series over a window
/// selection — moments (exactly additive) plus a fixed-bin histogram and
/// histogram-interpolated percentiles. The percentiles are **approximate** (no
/// raw samples are retained, so memory stays bounded for 24/7); the open end
/// bins report at their boundary edge. Fixed-size, so the comparison row stays
/// `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarkoutDistribution {
    /// Number of passive-fill 5s markouts in the selection.
    pub n: u64,
    /// Arithmetic mean, `None` when `n == 0`.
    pub mean: Option<f64>,
    /// Population standard deviation, `None` when `n == 0`.
    pub stddev: Option<f64>,
    /// Smallest observed markout, `None` when `n == 0`.
    pub min: Option<f64>,
    /// Largest observed markout, `None` when `n == 0`.
    pub max: Option<f64>,
    /// Histogram-interpolated median, `None` when `n == 0`.
    pub p50: Option<f64>,
    /// Histogram-interpolated 95th percentile, `None` when `n == 0`.
    pub p95: Option<f64>,
    /// Per-bin counts; bin layout per [`MARKOUT_BIN_EDGES`].
    pub histogram: [u64; MARKOUT_BIN_COUNT],
}

impl MarkoutDistribution {
    /// Builds a distribution from the additive accumulators a roll-up keeps.
    fn from_parts(
        sum: f64,
        sumsq: f64,
        n: u64,
        min: f64,
        max: f64,
        hist: [u64; MARKOUT_BIN_COUNT],
    ) -> Self {
        if n == 0 {
            return Self {
                n: 0,
                mean: None,
                stddev: None,
                min: None,
                max: None,
                p50: None,
                p95: None,
                histogram: hist,
            };
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "markout sample count is bounded by the evaluation window"
        )]
        let nf = n as f64;
        let mean = sum / nf;
        // Population variance; `max(0.0)` guards floating-point cancellation.
        let variance = (sumsq / nf - mean * mean).max(0.0);
        Self {
            n,
            mean: Some(mean),
            stddev: Some(variance.sqrt()),
            min: Some(min),
            max: Some(max),
            p50: histogram_percentile(&hist, n, 0.50),
            p95: histogram_percentile(&hist, n, 0.95),
            histogram: hist,
        }
    }
}

/// The value of the `q`-quantile (`q` in `[0, 1]`) approximated from the
/// histogram: the bin whose cumulative count first reaches rank `q·n`, linearly
/// interpolated within the bin's `[lo, hi]`. Open end bins report at their
/// boundary edge. `None` when `n == 0`.
fn histogram_percentile(hist: &[u64; MARKOUT_BIN_COUNT], n: u64, q: f64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "markout sample count is bounded by the evaluation window"
    )]
    let target = q * n as f64;
    let mut cum: u64 = 0;
    for (bin, &c) in hist.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let next = cum + c;
        #[allow(
            clippy::cast_precision_loss,
            reason = "markout sample count is bounded by the evaluation window"
        )]
        let next_f = next as f64;
        if next_f >= target {
            return Some(interp_in_bin(bin, cum, c, target));
        }
        cum = next;
    }
    // Floating-point rounding can leave `target` a hair above the final
    // cumulative count: report the top non-empty bin.
    (0..MARKOUT_BIN_COUNT)
        .rev()
        .find(|&b| hist[b] > 0)
        .map(|b| interp_in_bin(b, n.saturating_sub(hist[b]), hist[b], target))
}

/// Linearly interpolates the rank `target` within bin `bin` (whose cumulative
/// count before it is `cum` and whose own count is `c`).
fn interp_in_bin(bin: usize, cum: u64, c: u64, target: f64) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "markout sample count is bounded by the evaluation window"
    )]
    let frac = ((target - cum as f64) / c as f64).clamp(0.0, 1.0);
    match bin {
        // Open underflow: report its upper edge.
        0 => MARKOUT_BIN_EDGES[0],
        // Open overflow: report its lower edge.
        b if b == MARKOUT_BIN_COUNT - 1 => MARKOUT_BIN_EDGES[MARKOUT_BIN_COUNT - 2],
        b => {
            let lo = MARKOUT_BIN_EDGES[b - 1];
            let hi = MARKOUT_BIN_EDGES[b];
            lo + frac * (hi - lo)
        }
    }
}

/// Which calendar span a series-comparison query aggregates over. "today" is a
/// [`DayKey`] passed in by the caller (the crate is sans-IO — the bot supplies
/// `DayKey::from_ts(now)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonWindow {
    /// Only the current UTC day.
    Today,
    /// The `n`-day span ending today, inclusive: `[today-(n-1) ..= today]`.
    /// `LastDays(1) == Today`; `LastDays(0)` matches nothing.
    LastDays(u16),
    /// Every day with data (`today` is ignored).
    All,
}

impl ComparisonWindow {
    /// Whether `day` falls inside this window relative to `today`.
    #[must_use]
    pub fn contains(self, day: DayKey, today: DayKey) -> bool {
        match self {
            Self::Today => day == today,
            Self::LastDays(n) => {
                n > 0 && day <= today && day.as_days() > today.as_days() - i64::from(n)
            }
            Self::All => true,
        }
    }
}

/// One day's aggregate for one series in one mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DailyRollup {
    /// Session mode.
    pub mode: Mode,
    /// Series.
    pub series: Series,
    /// UTC day.
    pub day: DayKey,
    /// Windows that traded (settled with attribution).
    pub windows_traded: u32,
    /// Windows whose ledger PnL was positive.
    pub windows_profitable: u32,
    /// Net ledger PnL.
    pub net_realized: Dollars,
    /// Locked-pair PnL bucket sum.
    pub locked_pair_pnl: Dollars,
    /// Excess-inventory PnL bucket sum.
    pub excess_pnl: Dollars,
    /// Settlement-remainder bucket sum.
    pub settlement_remainder: Dollars,
    /// Taker-fees bucket sum (≤ 0).
    pub taker_fees: Dollars,
    /// Estimated maker rebate sum (≥ 0).
    pub estimated_rebate: Dollars,
    /// Maker fills.
    pub maker_fills: u64,
    /// Taker fills.
    pub taker_fills: u64,
    /// `Σ price·size` over taker fills — the taker-budget-usage numerator.
    pub taker_notional: Dollars,
    /// Sum of finalized passive-fill 5s markouts.
    pub markout_5s_sum: f64,
    /// Sum of squares of those 5s markouts (population variance via `Σx²`).
    pub markout_5s_sumsq: f64,
    /// Count of finalized passive-fill 5s markouts.
    pub markout_5s_n: u64,
    /// Smallest 5s markout seen (`+INF` sentinel until any sample).
    pub markout_5s_min: f64,
    /// Largest 5s markout seen (`-INF` sentinel until any sample).
    pub markout_5s_max: f64,
    /// 5s-markout histogram (bin layout per [`MARKOUT_BIN_EDGES`]).
    pub markout_5s_hist: [u32; MARKOUT_BIN_COUNT],
}

impl DailyRollup {
    fn new(mode: Mode, series: Series, day: DayKey) -> Self {
        Self {
            mode,
            series,
            day,
            windows_traded: 0,
            windows_profitable: 0,
            net_realized: Dollars::ZERO,
            locked_pair_pnl: Dollars::ZERO,
            excess_pnl: Dollars::ZERO,
            settlement_remainder: Dollars::ZERO,
            taker_fees: Dollars::ZERO,
            estimated_rebate: Dollars::ZERO,
            maker_fills: 0,
            taker_fills: 0,
            taker_notional: Dollars::ZERO,
            markout_5s_sum: 0.0,
            markout_5s_sumsq: 0.0,
            markout_5s_n: 0,
            markout_5s_min: f64::INFINITY,
            markout_5s_max: f64::NEG_INFINITY,
            markout_5s_hist: [0; MARKOUT_BIN_COUNT],
        }
    }

    fn fold(&mut self, a: &WindowAttribution, markout_5s: &[f64]) {
        self.windows_traded += 1;
        if a.is_profitable() {
            self.windows_profitable += 1;
        }
        self.net_realized = self.net_realized + a.realized_pnl;
        self.locked_pair_pnl = self.locked_pair_pnl + a.locked_pair_pnl;
        self.excess_pnl = self.excess_pnl + a.excess_pnl;
        self.settlement_remainder = self.settlement_remainder + a.settlement_remainder;
        self.taker_fees = self.taker_fees + a.taker_fees;
        self.estimated_rebate = self.estimated_rebate + a.estimated_rebate;
        self.maker_fills += a.maker_fills;
        self.taker_fills += a.taker_fills;
        self.taker_notional = self.taker_notional + a.taker_notional;
        for m in markout_5s {
            if !m.is_finite() {
                continue; // defensive; finalized markouts are finite
            }
            self.markout_5s_sum += *m;
            self.markout_5s_sumsq += *m * *m;
            self.markout_5s_min = self.markout_5s_min.min(*m);
            self.markout_5s_max = self.markout_5s_max.max(*m);
            self.markout_5s_hist[markout_bin(*m)] += 1;
            self.markout_5s_n += 1;
        }
    }

    /// Ledger PnL per traded window, `0` when none traded.
    #[must_use]
    pub fn pnl_per_window(&self) -> Dollars {
        per_window(self.net_realized, self.windows_traded)
    }

    /// Fraction of traded windows that were profitable, in `[0, 1]`.
    #[must_use]
    pub fn fraction_profitable(&self) -> f64 {
        fraction(self.windows_profitable, self.windows_traded)
    }

    /// Average passive-fill 5s markout, `None` with no passive 5s sample.
    #[must_use]
    pub fn avg_markout_5s(&self) -> Option<f64> {
        mean(self.markout_5s_sum, self.markout_5s_n)
    }

    /// The full 5s-markout distribution for this day.
    #[must_use]
    pub fn markout_distribution(&self) -> MarkoutDistribution {
        MarkoutDistribution::from_parts(
            self.markout_5s_sum,
            self.markout_5s_sumsq,
            self.markout_5s_n,
            self.markout_5s_min,
            self.markout_5s_max,
            self.markout_5s_hist.map(u64::from),
        )
    }

    /// The §10 "inventory PnL" half (excess + settlement remainder).
    #[must_use]
    pub fn inventory_pnl(&self) -> Dollars {
        self.excess_pnl + self.settlement_remainder
    }
}

/// Cross-day totals for one series in one mode — the basis of a comparison row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesAggregate {
    /// Session mode.
    pub mode: Mode,
    /// Series.
    pub series: Series,
    /// Windows that traded.
    pub windows_traded: u32,
    /// Windows whose ledger PnL was positive.
    pub windows_profitable: u32,
    /// Net ledger PnL.
    pub net_realized: Dollars,
    /// Locked-pair PnL.
    pub locked_pair_pnl: Dollars,
    /// Excess-inventory PnL.
    pub excess_pnl: Dollars,
    /// Settlement remainder.
    pub settlement_remainder: Dollars,
    /// Taker fees (≤ 0).
    pub taker_fees: Dollars,
    /// Estimated rebate (≥ 0).
    pub estimated_rebate: Dollars,
    /// Maker fills.
    pub maker_fills: u64,
    /// Taker fills.
    pub taker_fills: u64,
    /// `Σ price·size` over taker fills — the taker-budget-usage numerator.
    pub taker_notional: Dollars,
    /// Sum of 5s markouts.
    pub markout_5s_sum: f64,
    /// Sum of squares of 5s markouts.
    pub markout_5s_sumsq: f64,
    /// Count of 5s markouts.
    pub markout_5s_n: u64,
    /// Smallest 5s markout (`+INF` until any sample).
    pub markout_5s_min: f64,
    /// Largest 5s markout (`-INF` until any sample).
    pub markout_5s_max: f64,
    /// 5s-markout histogram (bin layout per [`MARKOUT_BIN_EDGES`]).
    pub markout_5s_hist: [u32; MARKOUT_BIN_COUNT],
}

impl SeriesAggregate {
    fn new(mode: Mode, series: Series) -> Self {
        Self {
            mode,
            series,
            windows_traded: 0,
            windows_profitable: 0,
            net_realized: Dollars::ZERO,
            locked_pair_pnl: Dollars::ZERO,
            excess_pnl: Dollars::ZERO,
            settlement_remainder: Dollars::ZERO,
            taker_fees: Dollars::ZERO,
            estimated_rebate: Dollars::ZERO,
            maker_fills: 0,
            taker_fills: 0,
            taker_notional: Dollars::ZERO,
            markout_5s_sum: 0.0,
            markout_5s_sumsq: 0.0,
            markout_5s_n: 0,
            markout_5s_min: f64::INFINITY,
            markout_5s_max: f64::NEG_INFINITY,
            markout_5s_hist: [0; MARKOUT_BIN_COUNT],
        }
    }

    fn add_daily(&mut self, d: &DailyRollup) {
        self.windows_traded += d.windows_traded;
        self.windows_profitable += d.windows_profitable;
        self.net_realized = self.net_realized + d.net_realized;
        self.locked_pair_pnl = self.locked_pair_pnl + d.locked_pair_pnl;
        self.excess_pnl = self.excess_pnl + d.excess_pnl;
        self.settlement_remainder = self.settlement_remainder + d.settlement_remainder;
        self.taker_fees = self.taker_fees + d.taker_fees;
        self.estimated_rebate = self.estimated_rebate + d.estimated_rebate;
        self.maker_fills += d.maker_fills;
        self.taker_fills += d.taker_fills;
        self.taker_notional = self.taker_notional + d.taker_notional;
        self.markout_5s_sum += d.markout_5s_sum;
        self.markout_5s_sumsq += d.markout_5s_sumsq;
        self.markout_5s_n += d.markout_5s_n;
        self.markout_5s_min = self.markout_5s_min.min(d.markout_5s_min);
        self.markout_5s_max = self.markout_5s_max.max(d.markout_5s_max);
        for (slot, c) in self
            .markout_5s_hist
            .iter_mut()
            .zip(d.markout_5s_hist.iter())
        {
            *slot += *c;
        }
    }

    /// The 5s-markout distribution aggregated across this series' days.
    #[must_use]
    pub fn markout_distribution(&self) -> MarkoutDistribution {
        MarkoutDistribution::from_parts(
            self.markout_5s_sum,
            self.markout_5s_sumsq,
            self.markout_5s_n,
            self.markout_5s_min,
            self.markout_5s_max,
            self.markout_5s_hist.map(u64::from),
        )
    }

    /// Builds the dashboard comparison row, attaching the series'
    /// adverse-selection health and minimum-sample warning. `min_sample_windows`
    /// and `taker_budget_per_window` come from [`AnalyticsParams`].
    ///
    /// [`AnalyticsParams`]: crate::AnalyticsParams
    #[must_use]
    pub fn to_row(
        &self,
        min_sample_windows: u32,
        taker_budget_per_window: Dollars,
        health: AdverseSelectionState,
    ) -> SeriesComparisonRow {
        let markout_5s = self.markout_distribution();
        let budget = taker_budget_per_window.as_decimal();
        let taker_budget_used_fraction = if budget <= Decimal::ZERO || self.windows_traded == 0 {
            None
        } else {
            // Mean per-window taker notional, as a fraction of the budget.
            (self.taker_notional.as_decimal() / Decimal::from(self.windows_traded) / budget)
                .to_f64()
        };
        SeriesComparisonRow {
            series: self.series,
            mode: self.mode,
            windows_traded: self.windows_traded,
            fraction_profitable: fraction(self.windows_profitable, self.windows_traded),
            net_pnl: self.net_realized,
            pnl_per_window: per_window(self.net_realized, self.windows_traded),
            locked_pair_pnl: self.locked_pair_pnl,
            inventory_pnl: self.excess_pnl + self.settlement_remainder,
            fees_paid: -self.taker_fees, // report as a positive cost
            rebates_earned: self.estimated_rebate,
            avg_markout_5s: markout_5s.mean,
            markout_5s,
            maker_fills: self.maker_fills,
            taker_fills: self.taker_fills,
            maker_fill_fraction: fraction64(self.maker_fills, self.maker_fills + self.taker_fills),
            taker_notional: self.taker_notional,
            taker_budget_used_fraction,
            min_sample_warning: self.windows_traded < min_sample_windows,
            health,
        }
    }
}

/// One row of the §10 series-comparison table.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeriesComparisonRow {
    /// Series.
    pub series: Series,
    /// Session mode.
    pub mode: Mode,
    /// Windows traded.
    pub windows_traded: u32,
    /// Fraction of windows profitable, in `[0, 1]`.
    pub fraction_profitable: f64,
    /// Net ledger PnL.
    pub net_pnl: Dollars,
    /// Ledger PnL per traded window.
    pub pnl_per_window: Dollars,
    /// Locked-pair PnL (the guaranteed-edge half of the split).
    pub locked_pair_pnl: Dollars,
    /// Inventory PnL (excess + settlement remainder — the risk half).
    pub inventory_pnl: Dollars,
    /// Taker fees paid (a positive cost).
    pub fees_paid: Dollars,
    /// Estimated maker rebates earned.
    pub rebates_earned: Dollars,
    /// Average passive-fill 5s markout, `None` with no sample. Convenience mirror
    /// of `markout_5s.mean`.
    pub avg_markout_5s: Option<f64>,
    /// Full 5s-markout distribution (moments + histogram + approx percentiles).
    pub markout_5s: MarkoutDistribution,
    /// Maker fills.
    pub maker_fills: u64,
    /// Taker fills.
    pub taker_fills: u64,
    /// Maker share of all fills, in `[0, 1]` (`0` when no fills).
    pub maker_fill_fraction: f64,
    /// `Σ price·size` over taker fills in the selection.
    pub taker_notional: Dollars,
    /// Mean per-window taker notional as a fraction of the per-window taker
    /// budget; `None` when the budget is `0` or no windows traded. `1.0` means
    /// the average window spent its whole taker budget.
    pub taker_budget_used_fraction: Option<f64>,
    /// Whether the series has traded too few windows to trust (§10), evaluated
    /// against the **selected** window's traded count.
    pub min_sample_warning: bool,
    /// Adverse-selection health for the series (the current rolling-window gate,
    /// independent of the calendar selection).
    pub health: AdverseSelectionState,
}

impl SeriesComparisonRow {
    /// An orderable key for sorting the comparison table by `col`. Returned as
    /// `f64` because it is only ever compared, never displayed (money precision
    /// is irrelevant here); a missing optional sorts last under a descending
    /// sort (`-INF`). The dashboard pairs this with [`SortColumn::higher_is_better`].
    #[must_use]
    pub fn sort_key(&self, col: SortColumn) -> f64 {
        let dollars = |d: Dollars| d.as_decimal().to_f64().unwrap_or(0.0);
        match col {
            SortColumn::NetPnl => dollars(self.net_pnl),
            SortColumn::PnlPerWindow => dollars(self.pnl_per_window),
            SortColumn::FractionProfitable => self.fraction_profitable,
            SortColumn::WindowsTraded => f64::from(self.windows_traded),
            SortColumn::AvgMarkout5s => self.avg_markout_5s.unwrap_or(f64::NEG_INFINITY),
            SortColumn::FeesPaid => dollars(self.fees_paid),
            SortColumn::RebatesEarned => dollars(self.rebates_earned),
            SortColumn::LockedPairPnl => dollars(self.locked_pair_pnl),
            SortColumn::InventoryPnl => dollars(self.inventory_pnl),
            SortColumn::TakerBudgetUsed => {
                self.taker_budget_used_fraction.unwrap_or(f64::NEG_INFINITY)
            }
            SortColumn::MakerFills => count_to_f64(self.maker_fills),
            SortColumn::TakerFills => count_to_f64(self.taker_fills),
            // Healthiest first under ascending order: Ok < InsufficientSample < Alarm.
            SortColumn::Health => match self.health {
                AdverseSelectionState::Ok => 0.0,
                AdverseSelectionState::InsufficientSample => 1.0,
                AdverseSelectionState::Alarm => 2.0,
            },
        }
    }
}

/// A sortable column of the §10 series-comparison table — the dashboard's
/// sorting hints. Each carries a stable label and a "higher is better" flag so
/// the table can pick a sensible default direction per column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SortColumn {
    /// Net ledger PnL — the default sort (best first).
    #[default]
    NetPnl,
    /// Ledger PnL per traded window.
    PnlPerWindow,
    /// Fraction of windows profitable.
    FractionProfitable,
    /// Windows traded.
    WindowsTraded,
    /// Average passive-fill 5s markout.
    AvgMarkout5s,
    /// Taker fees paid (a cost).
    FeesPaid,
    /// Estimated maker rebates earned.
    RebatesEarned,
    /// Locked-pair PnL.
    LockedPairPnl,
    /// Inventory PnL (excess + remainder).
    InventoryPnl,
    /// Mean per-window taker-budget usage.
    TakerBudgetUsed,
    /// Maker fill count.
    MakerFills,
    /// Taker fill count.
    TakerFills,
    /// Adverse-selection health.
    Health,
}

impl SortColumn {
    /// Every sortable column, in display order.
    pub const ALL: [Self; 13] = [
        Self::NetPnl,
        Self::PnlPerWindow,
        Self::FractionProfitable,
        Self::WindowsTraded,
        Self::AvgMarkout5s,
        Self::FeesPaid,
        Self::RebatesEarned,
        Self::LockedPairPnl,
        Self::InventoryPnl,
        Self::TakerBudgetUsed,
        Self::MakerFills,
        Self::TakerFills,
        Self::Health,
    ];

    /// A stable column label for the dashboard header.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NetPnl => "Net PnL",
            Self::PnlPerWindow => "PnL / window",
            Self::FractionProfitable => "% profitable",
            Self::WindowsTraded => "Windows",
            Self::AvgMarkout5s => "Avg 5s markout",
            Self::FeesPaid => "Fees paid",
            Self::RebatesEarned => "Rebates",
            Self::LockedPairPnl => "Locked-pair PnL",
            Self::InventoryPnl => "Inventory PnL",
            Self::TakerBudgetUsed => "Taker budget used",
            Self::MakerFills => "Maker fills",
            Self::TakerFills => "Taker fills",
            Self::Health => "Health",
        }
    }

    /// Whether a larger [`sort_key`](SeriesComparisonRow::sort_key) is "better"
    /// for this column — the table's default sort direction. `false` for costs
    /// (fees), taker activity (more taker = less edge / more headroom consumed),
    /// and health (lower ordinal = healthier).
    #[must_use]
    pub const fn higher_is_better(self) -> bool {
        match self {
            Self::NetPnl
            | Self::PnlPerWindow
            | Self::FractionProfitable
            | Self::WindowsTraded
            | Self::AvgMarkout5s
            | Self::RebatesEarned
            | Self::LockedPairPnl
            | Self::InventoryPnl
            | Self::MakerFills => true,
            Self::FeesPaid | Self::TakerBudgetUsed | Self::TakerFills | Self::Health => false,
        }
    }
}

/// A complete series-comparison query result for the dashboard: the rows for a
/// time-window selection, pre-sorted by [`default_sort`](Self::default_sort)
/// (best first), with the selection echoed back so the UI can label it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesComparison {
    /// The window selection these rows cover.
    pub window: ComparisonWindow,
    /// The "today" reference used to resolve the selection.
    pub day: DayKey,
    /// One row per series that traded in the selection, pre-sorted best-first by
    /// `default_sort`. The dashboard may re-sort live via
    /// [`SeriesComparisonRow::sort_key`].
    pub rows: Vec<SeriesComparisonRow>,
    /// The column the rows are sorted by.
    pub default_sort: SortColumn,
}

impl SeriesComparison {
    /// Sorts `rows` best-first by `sort` (descending `sort_key`, stable `Series`
    /// tiebreak) and wraps them.
    #[must_use]
    pub fn new(
        window: ComparisonWindow,
        day: DayKey,
        mut rows: Vec<SeriesComparisonRow>,
        sort: SortColumn,
    ) -> Self {
        rows.sort_by(|a, b| {
            b.sort_key(sort)
                .partial_cmp(&a.sort_key(sort))
                .unwrap_or(Ordering::Equal)
                .then(a.series.cmp(&b.series))
        });
        Self {
            window,
            day,
            rows,
            default_sort: sort,
        }
    }
}

/// The daily roll-up store for one (per-mode) analytics instance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RollupStore {
    daily: HashMap<(Series, DayKey), DailyRollup>,
}

impl RollupStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds a settled window's attribution (and its passive 5s markouts) into
    /// the `(series, day)` daily roll-up.
    pub fn fold(&mut self, a: &WindowAttribution, markout_5s: &[f64]) {
        let day = DayKey::from_ts(a.ts);
        self.daily
            .entry((a.window.series, day))
            .or_insert_with(|| DailyRollup::new(a.mode, a.window.series, day))
            .fold(a, markout_5s);
    }

    /// The daily roll-up for a `(series, day)`, if any.
    #[must_use]
    pub fn daily(&self, series: Series, day: DayKey) -> Option<&DailyRollup> {
        self.daily.get(&(series, day))
    }

    /// All daily roll-ups (unordered).
    pub fn daily_rollups(&self) -> impl Iterator<Item = &DailyRollup> {
        self.daily.values()
    }

    /// Cross-day aggregates over every day, one per series that has traded, in
    /// series order.
    #[must_use]
    pub fn series_aggregates(&self) -> Vec<SeriesAggregate> {
        // `All` ignores `today`; any value works.
        self.series_aggregates_over(ComparisonWindow::All, DayKey::from_days(0))
    }

    /// Cross-day aggregates restricted to `window` (resolved against `today`),
    /// one per series that traded in the selection, in series order. The bins
    /// and moments are exactly additive, so this equals re-folding only the
    /// selected days.
    #[must_use]
    pub fn series_aggregates_over(
        &self,
        window: ComparisonWindow,
        today: DayKey,
    ) -> Vec<SeriesAggregate> {
        // Fold days in a deterministic `(series, day)` order rather than HashMap
        // iteration order, so the f64 markout sums are bit-stable across engine
        // instances (live vs a journal rebuild produce identical query output).
        let mut days: Vec<&DailyRollup> = self
            .daily
            .values()
            .filter(|d| window.contains(d.day, today))
            .collect();
        days.sort_by_key(|d| (d.series, d.day));
        let mut by_series: BTreeMap<Series, SeriesAggregate> = BTreeMap::new();
        for d in days {
            by_series
                .entry(d.series)
                .or_insert_with(|| SeriesAggregate::new(d.mode, d.series))
                .add_daily(d);
        }
        by_series.into_values().collect()
    }
}

/// Dollars per window, `0` when no windows traded.
fn per_window(total: Dollars, windows: u32) -> Dollars {
    if windows == 0 {
        Dollars::ZERO
    } else {
        Dollars::new(total.as_decimal() / Decimal::from(windows))
    }
}

/// Fraction `num / den` in `[0, 1]`, `0` when `den == 0`.
fn fraction(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "window counts are small and well within f64's exact range"
        )]
        let f = f64::from(num) / f64::from(den);
        f
    }
}

/// Fraction `num / den` in `[0, 1]` for `u64` counts, `0` when `den == 0`.
fn fraction64(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        count_to_f64(num) / count_to_f64(den)
    }
}

/// A fill/sample count as `f64`, for ratios and sort keys (display/ordering
/// values, never money).
#[allow(
    clippy::cast_precision_loss,
    reason = "fill/sample counts are display/ordering values, bounded well within f64"
)]
fn count_to_f64(n: u64) -> f64 {
    n as f64
}

/// Mean `sum / n`, `None` when `n == 0`.
fn mean(sum: f64, n: u64) -> Option<f64> {
    if n == 0 {
        None
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "markout sample count is bounded by the evaluation window"
        )]
        let denom = n as f64;
        Some(sum / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{Asset, Outcome, TimestampMs, WindowDuration, WindowId};
    use rust_decimal::dec;

    fn series() -> Series {
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        }
    }

    fn attribution(realized: Decimal, ts: TimestampMs) -> WindowAttribution {
        attribution_notional(realized, ts, Dollars::ZERO)
    }

    fn attribution_notional(
        realized: Decimal,
        ts: TimestampMs,
        taker_notional: Dollars,
    ) -> WindowAttribution {
        WindowAttribution {
            window: WindowId {
                series: series(),
                open_time: ts,
            },
            mode: Mode::Paper,
            outcome: Outcome::Up,
            locked_pair_pnl: Dollars::new(realized),
            excess_pnl: Dollars::ZERO,
            settlement_remainder: Dollars::ZERO,
            taker_fees: Dollars::ZERO,
            estimated_rebate: Dollars::new(dec!(0.1)),
            realized_pnl: Dollars::new(realized),
            maker_fills: 4,
            taker_fills: 1,
            taker_notional,
            ts,
        }
    }

    #[test]
    fn folds_into_utc_day_buckets() {
        let mut store = RollupStore::new();
        // Same UTC day, two windows.
        let day0 = TimestampMs::from_millis(1_781_481_600_000); // 2026-06-15T00:00:00Z
        store.fold(&attribution(dec!(3), day0), &[0.01, -0.02]);
        store.fold(
            &attribution(
                dec!(-1),
                TimestampMs::from_millis(day0.as_millis() + 60_000),
            ),
            &[0.03],
        );
        // Next UTC day, one window.
        let day1 = TimestampMs::from_millis(day0.as_millis() + 86_400_000);
        store.fold(&attribution(dec!(5), day1), &[]);

        let key0 = DayKey::from_ts(day0);
        let r0 = store.daily(series(), key0).expect("day 0");
        assert_eq!(r0.windows_traded, 2);
        assert_eq!(r0.windows_profitable, 1); // +3 profitable, -1 not
        assert_eq!(r0.net_realized, Dollars::new(dec!(2)));
        assert_eq!(r0.pnl_per_window(), Dollars::new(dec!(1)));
        assert_eq!(r0.fraction_profitable(), 0.5);
        assert_eq!(r0.markout_5s_n, 3);

        let r1 = store.daily(series(), DayKey::from_ts(day1)).expect("day 1");
        assert_eq!(r1.windows_traded, 1);
        assert_eq!(r1.avg_markout_5s(), None); // no 5s sample that day
    }

    #[test]
    fn series_aggregate_sums_across_days_and_builds_row() {
        let mut store = RollupStore::new();
        let day0 = TimestampMs::from_millis(1_781_481_600_000);
        let day1 = TimestampMs::from_millis(day0.as_millis() + 86_400_000);
        store.fold(&attribution(dec!(3), day0), &[0.02]);
        store.fold(&attribution(dec!(5), day1), &[-0.04]);

        let aggs = store.series_aggregates();
        assert_eq!(aggs.len(), 1);
        let agg = aggs[0];
        assert_eq!(agg.windows_traded, 2);
        assert_eq!(agg.net_realized, Dollars::new(dec!(8)));

        let row = agg.to_row(5, Dollars::new(dec!(10)), AdverseSelectionState::Ok);
        assert_eq!(row.windows_traded, 2);
        assert_eq!(row.pnl_per_window, Dollars::new(dec!(4)));
        assert_eq!(row.fraction_profitable, 1.0);
        assert_eq!(row.rebates_earned, Dollars::new(dec!(0.2)));
        assert_eq!(row.fees_paid, Dollars::ZERO);
        // avg of [0.02, -0.04] = -0.01.
        assert!((row.avg_markout_5s.expect("mean") - (-0.01)).abs() < 1e-12);
        // The distribution mirrors the average and counts both samples.
        assert_eq!(row.markout_5s.n, 2);
        assert!((row.markout_5s.mean.expect("mean") - (-0.01)).abs() < 1e-12);
        // 2 traded < min_sample 5 ⇒ warning.
        assert!(row.min_sample_warning);
        assert_eq!(row.health, AdverseSelectionState::Ok);
        // Inventory split = excess + remainder = 0 here; locked carries the PnL.
        assert_eq!(row.inventory_pnl, Dollars::ZERO);
        assert_eq!(row.locked_pair_pnl, Dollars::new(dec!(8)));
    }

    // ---- new: time-window selection -----------------------------------------

    #[test]
    fn comparison_window_contains_boundaries() {
        let today = DayKey::from_days(20_619);
        // Today.
        assert!(ComparisonWindow::Today.contains(today, today));
        assert!(!ComparisonWindow::Today.contains(DayKey::from_days(20_618), today));
        // LastDays(7): [today-6 ..= today] inclusive.
        let w = ComparisonWindow::LastDays(7);
        assert!(w.contains(today, today));
        assert!(w.contains(DayKey::from_days(20_613), today)); // today-6 included
        assert!(!w.contains(DayKey::from_days(20_612), today)); // today-7 excluded
        assert!(!w.contains(DayKey::from_days(20_620), today)); // future excluded
        // LastDays(1) == Today; LastDays(0) matches nothing.
        assert!(ComparisonWindow::LastDays(1).contains(today, today));
        assert!(!ComparisonWindow::LastDays(1).contains(DayKey::from_days(20_618), today));
        assert!(!ComparisonWindow::LastDays(0).contains(today, today));
        // All ignores today.
        assert!(ComparisonWindow::All.contains(DayKey::from_days(1), today));
    }

    #[test]
    fn windowed_selection_across_utc_days() {
        let mut store = RollupStore::new();
        let day = |d: i64| DayKey::from_days(d).start_ms();
        // Windows on days 100, 99, 97, 93 — net PnL 1, 2, 4, 8.
        store.fold(&attribution(dec!(1), day(100)), &[]);
        store.fold(&attribution(dec!(2), day(99)), &[]);
        store.fold(&attribution(dec!(4), day(97)), &[]);
        store.fold(&attribution(dec!(8), day(93)), &[]);
        let today = DayKey::from_days(100);

        let today_only = store.series_aggregates_over(ComparisonWindow::Today, today);
        assert_eq!(today_only.len(), 1);
        assert_eq!(today_only[0].windows_traded, 1);
        assert_eq!(today_only[0].net_realized, Dollars::new(dec!(1)));

        // LastDays(7): days 100,99,97 included; 93 (today-7) excluded → 1+2+4 = 7.
        let week = store.series_aggregates_over(ComparisonWindow::LastDays(7), today);
        assert_eq!(week[0].windows_traded, 3);
        assert_eq!(week[0].net_realized, Dollars::new(dec!(7)));

        // LastDays(1) == Today.
        let one = store.series_aggregates_over(ComparisonWindow::LastDays(1), today);
        assert_eq!(one[0].net_realized, Dollars::new(dec!(1)));

        // LastDays(0) → nothing.
        assert!(
            store
                .series_aggregates_over(ComparisonWindow::LastDays(0), today)
                .is_empty()
        );

        // All == legacy total = 1+2+4+8 = 15.
        let all = store.series_aggregates_over(ComparisonWindow::All, today);
        assert_eq!(all[0].net_realized, Dollars::new(dec!(15)));
        assert_eq!(
            store.series_aggregates()[0].net_realized,
            Dollars::new(dec!(15))
        );
    }

    // ---- new: distribution ---------------------------------------------------

    #[test]
    fn markout_distribution_bins_and_moments() {
        let mut store = RollupStore::new();
        let ts = TimestampMs::from_millis(1_781_481_600_000);
        let samples = [-0.20, -0.04, -0.001, 0.0, 0.001, 0.04, 0.20];
        store.fold(&attribution(dec!(1), ts), &samples);
        let agg = store.series_aggregates();
        let d = agg[0].markout_distribution();

        assert_eq!(d.n, 7);
        assert!((d.min.expect("min") - (-0.20)).abs() < 1e-12);
        assert!((d.max.expect("max") - 0.20).abs() < 1e-12);
        // Symmetric samples → mean 0.
        assert!(d.mean.expect("mean").abs() < 1e-12);
        // sumsq = 2·(0.04 + 0.0016 + 1e-6) = 0.083202; var = /7; stddev = sqrt.
        let expected_std = (0.083202_f64 / 7.0).sqrt();
        assert!((d.stddev.expect("std") - expected_std).abs() < 1e-9);
        // Bin placement: -0.20→0, -0.04→2, -0.001→8, 0.0→8, 0.001→9, 0.04→14, 0.20→16.
        let mut expected = [0u64; MARKOUT_BIN_COUNT];
        expected[0] = 1;
        expected[2] = 1;
        expected[8] = 2;
        expected[9] = 1;
        expected[14] = 1;
        expected[16] = 1;
        assert_eq!(d.histogram, expected);
    }

    #[test]
    fn percentiles_interpolate_and_handle_open_bins() {
        // All mass in one interior bin [-0.005, -0.002) (bin 6): p50 within it.
        let mut hist = [0u64; MARKOUT_BIN_COUNT];
        hist[6] = 4;
        let d = MarkoutDistribution::from_parts(-0.014, 5.0e-5, 4, -0.005, -0.002, hist);
        let p50 = d.p50.expect("p50");
        assert!((-0.005..=-0.002).contains(&p50), "p50 {p50} in bin");

        // All mass in the overflow bin → p95 reports its lower edge 0.10.
        let mut over = [0u64; MARKOUT_BIN_COUNT];
        over[MARKOUT_BIN_COUNT - 1] = 10;
        let d2 = MarkoutDistribution::from_parts(2.0, 0.4, 10, 0.2, 0.2, over);
        assert!((d2.p95.expect("p95") - 0.10).abs() < 1e-12);

        // Empty → all None.
        let none = MarkoutDistribution::from_parts(
            0.0,
            0.0,
            0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            [0; MARKOUT_BIN_COUNT],
        );
        assert_eq!(none.p50, None);
        assert_eq!(none.mean, None);
        assert_eq!(none.min, None);
    }

    #[test]
    fn distribution_is_additive_across_days() {
        let samples = [-0.20, -0.04, -0.001, 0.0, 0.001, 0.04];
        // One day with all six.
        let mut a = RollupStore::new();
        a.fold(
            &attribution(dec!(1), DayKey::from_days(100).start_ms()),
            &samples,
        );
        // Two days, split 3 + 3.
        let mut b = RollupStore::new();
        b.fold(
            &attribution(dec!(1), DayKey::from_days(100).start_ms()),
            &samples[..3],
        );
        b.fold(
            &attribution(dec!(1), DayKey::from_days(99).start_ms()),
            &samples[3..],
        );

        let da = a.series_aggregates()[0].markout_distribution();
        let db = b.series_aggregates()[0].markout_distribution();
        // Histogram, count and min/max are EXACTLY additive (integer / min-max).
        assert_eq!(da.histogram, db.histogram);
        assert_eq!(da.n, db.n);
        assert_eq!(da.min, db.min);
        assert_eq!(da.max, db.max);
        // Mean/stddev additive up to f64 grouping (single-fold vs split sums).
        assert!((da.mean.expect("mean a") - db.mean.expect("mean b")).abs() < 1e-12);
        assert!((da.stddev.expect("std a") - db.stddev.expect("std b")).abs() < 1e-12);
    }

    // ---- new: taker-budget usage & maker mix --------------------------------

    #[test]
    fn taker_budget_usage_and_maker_mix() {
        let mut store = RollupStore::new();
        let ts = TimestampMs::from_millis(1_781_481_600_000);
        // Two windows, $20 + $10 taker notional = $30 over 2 windows.
        store.fold(
            &attribution_notional(dec!(1), ts, Dollars::new(dec!(20))),
            &[],
        );
        store.fold(
            &attribution_notional(
                dec!(1),
                TimestampMs::from_millis(ts.as_millis() + 60_000),
                Dollars::new(dec!(10)),
            ),
            &[],
        );
        let agg = store.series_aggregates();

        // Mean per-window notional $15 / $10 budget = 1.5.
        let row = agg[0].to_row(30, Dollars::new(dec!(10)), AdverseSelectionState::Ok);
        assert_eq!(row.taker_notional, Dollars::new(dec!(30)));
        assert!((row.taker_budget_used_fraction.expect("frac") - 1.5).abs() < 1e-12);
        // maker mix: helper sets maker 4 / taker 1 per window → 8 maker / 2 taker = 0.8.
        assert!((row.maker_fill_fraction - 0.8).abs() < 1e-12);

        // Zero budget → None.
        let row0 = agg[0].to_row(30, Dollars::ZERO, AdverseSelectionState::Ok);
        assert_eq!(row0.taker_budget_used_fraction, None);

        // No windows traded → None.
        let empty = SeriesAggregate::new(Mode::Paper, series());
        let rowe = empty.to_row(30, Dollars::new(dec!(10)), AdverseSelectionState::Ok);
        assert_eq!(rowe.taker_budget_used_fraction, None);
        assert_eq!(rowe.maker_fill_fraction, 0.0);
    }

    #[test]
    fn min_sample_flag_uses_the_selected_window_count() {
        let mut store = RollupStore::new();
        let today = DayKey::from_days(100);
        // 3 windows today, 40 more on prior days (43 total).
        for i in 0..3 {
            store.fold(
                &attribution(
                    dec!(1),
                    TimestampMs::from_millis(today.start_ms().as_millis() + i * 1000),
                ),
                &[],
            );
        }
        for d in 1..=40 {
            store.fold(
                &attribution(dec!(1), DayKey::from_days(100 - d).start_ms()),
                &[],
            );
        }

        let today_agg = store.series_aggregates_over(ComparisonWindow::Today, today);
        let today_row = today_agg[0].to_row(30, Dollars::new(dec!(10)), AdverseSelectionState::Ok);
        assert_eq!(today_row.windows_traded, 3);
        assert!(today_row.min_sample_warning); // 3 < 30

        let all_agg = store.series_aggregates();
        let all_row = all_agg[0].to_row(30, Dollars::new(dec!(10)), AdverseSelectionState::Ok);
        assert_eq!(all_row.windows_traded, 43);
        assert!(!all_row.min_sample_warning); // 43 >= 30
    }

    // ---- new: sort hints -----------------------------------------------------

    #[test]
    fn sort_column_hints() {
        assert_eq!(SortColumn::default(), SortColumn::NetPnl);
        assert_eq!(SortColumn::ALL.len(), 13);
        assert!(SortColumn::NetPnl.higher_is_better());
        assert!(SortColumn::AvgMarkout5s.higher_is_better());
        assert!(!SortColumn::FeesPaid.higher_is_better());
        assert!(!SortColumn::TakerBudgetUsed.higher_is_better());
        assert!(!SortColumn::Health.higher_is_better());
        // Every column has a non-empty label.
        for c in SortColumn::ALL {
            assert!(!c.label().is_empty());
        }
    }

    #[test]
    fn series_comparison_sorts_best_first() {
        fn row(net: Decimal, health: AdverseSelectionState) -> SeriesComparisonRow {
            let agg = {
                let mut store = RollupStore::new();
                store.fold(
                    &attribution(net, TimestampMs::from_millis(1_781_481_600_000)),
                    &[],
                );
                store.series_aggregates()[0]
            };
            agg.to_row(1, Dollars::new(dec!(10)), health)
        }
        let rows = vec![
            row(dec!(1), AdverseSelectionState::Ok),
            row(dec!(5), AdverseSelectionState::Ok),
            row(dec!(3), AdverseSelectionState::Ok),
        ];
        let cmp = SeriesComparison::new(
            ComparisonWindow::All,
            DayKey::from_days(0),
            rows,
            SortColumn::NetPnl,
        );
        let nets: Vec<Decimal> = cmp.rows.iter().map(|r| r.net_pnl.as_decimal()).collect();
        assert_eq!(nets, vec![dec!(5), dec!(3), dec!(1)]); // descending
        assert_eq!(cmp.default_sort, SortColumn::NetPnl);

        // Health ordinal: Ok = 0.
        assert!((cmp.rows[0].sort_key(SortColumn::Health) - 0.0).abs() < 1e-12);

        // A row whose budget fraction is `None` sorts last under descending
        // (its key is -INF).
        let none_row = {
            let empty = SeriesAggregate::new(Mode::Paper, series());
            empty.to_row(1, Dollars::ZERO, AdverseSelectionState::Ok)
        };
        assert_eq!(none_row.taker_budget_used_fraction, None);
        assert_eq!(
            none_row.sort_key(SortColumn::TakerBudgetUsed),
            f64::NEG_INFINITY
        );

        // Sort key for an Alarm health is the worst (largest) ordinal.
        let alarm = row(dec!(1), AdverseSelectionState::Alarm);
        assert!(alarm.sort_key(SortColumn::Health) > cmp.rows[0].sort_key(SortColumn::Health));
    }
}
