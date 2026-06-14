//! Per-window inventory accounting snapshot (CLAUDE.md §8: pair discipline).

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::market::{Outcome, WindowId};
use crate::money::{Dollars, Size};
use crate::time::TimestampMs;

/// Holdings on one side (Up or Down) of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SideInventory {
    /// Shares held.
    pub shares: Size,
    /// Total dollars paid for those shares.
    pub cost: Dollars,
}

impl SideInventory {
    /// Average cost per share, `None` when no shares are held.
    #[must_use]
    pub fn avg_price(&self) -> Option<Decimal> {
        if self.shares.is_zero() {
            None
        } else {
            Some(self.cost.as_decimal() / self.shares.as_decimal())
        }
    }
}

/// Point-in-time inventory state for one window. Crosses the bus and lands
/// in the journal/dashboard verbatim, so the derived fields are stored —
/// always build through [`InventorySnapshot::derive`] so they cannot drift
/// from the raw sides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InventorySnapshot {
    /// Window this inventory belongs to.
    pub window: WindowId,
    /// Up-side holdings.
    pub up: SideInventory,
    /// Down-side holdings.
    pub down: SideInventory,
    /// Matched pairs = min(up.shares, down.shares).
    pub matched_pairs: Size,
    /// Combined average cost of one Up + one Down share, in (0, 2);
    /// `None` until both sides are held. Pair discipline: only add passively
    /// while post-trade pair cost stays ≤ the configured threshold (§8).
    pub pair_cost: Option<Decimal>,
    /// Signed unmatched shares: positive = Up excess, negative = Down excess.
    pub excess: Decimal,
    /// Dollars lost if every unmatched share expires worthless: the cost
    /// already sunk into the excess shares. The binding §8 risk input.
    pub worst_case_if_excess_loses: Dollars,
    /// Snapshot creation time.
    pub ts: TimestampMs,
}

impl InventorySnapshot {
    /// The single construction path: computes every derived field from the
    /// two raw sides.
    #[must_use]
    pub fn derive(
        window: WindowId,
        up: SideInventory,
        down: SideInventory,
        ts: TimestampMs,
    ) -> Self {
        let matched_pairs = up.shares.min(down.shares);
        let pair_cost = match (up.avg_price(), down.avg_price()) {
            (Some(u), Some(d)) => Some(u + d),
            _ => None,
        };
        let excess = up.shares.as_decimal() - down.shares.as_decimal();
        // Cost sunk into the excess shares = excess × that side's average price.
        let worst_case = if excess.is_zero() {
            Dollars::ZERO
        } else if excess > Decimal::ZERO {
            Dollars::new(excess * up.avg_price().unwrap_or(Decimal::ZERO))
        } else {
            Dollars::new(-excess * down.avg_price().unwrap_or(Decimal::ZERO))
        };
        Self {
            window,
            up,
            down,
            matched_pairs,
            pair_cost,
            excess,
            worst_case_if_excess_loses: worst_case,
            ts,
        }
    }

    /// Which side holds the excess, if any.
    #[must_use]
    pub fn excess_side(&self) -> Option<Outcome> {
        if self.excess.is_zero() {
            None
        } else if self.excess > Decimal::ZERO {
            Some(Outcome::Up)
        } else {
            Some(Outcome::Down)
        }
    }
}

/// The closing summary for one settled window (CLAUDE.md §8/§10): produced by the
/// engine's inventory book on resolution and published as
/// [`Event::Settlement`](crate::Event) for analytics. Carries the *raw* numbers —
/// the final two sides, the matched/merged pair counts, fees, and the bottom-line
/// realized PnL — so the analytics layer derives the §10 locked-pair-vs-inventory
/// split itself (a pure function of these fields) rather than the engine baking it
/// in. The matched/pair-cost/excess fields are filled through
/// [`InventorySnapshot::derive`] so they can never drift from the two sides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettlementSummary {
    /// Window that settled.
    pub window: WindowId,
    /// Winning outcome (the side that pays $1/share).
    pub outcome: Outcome,
    /// Up-side holdings at close.
    pub up: SideInventory,
    /// Down-side holdings at close.
    pub down: SideInventory,
    /// Matched pairs still held at close = `min(up.shares, down.shares)`. Each
    /// nets $1 at settlement; locked-pair PnL = `matched_pairs · (1 − pair_cost)`.
    pub matched_pairs: Size,
    /// Combined average cost of a held pair (`avg_up + avg_down`), `None` unless
    /// both sides were held.
    pub pair_cost: Option<Decimal>,
    /// Signed unmatched shares at close: positive = Up excess, negative = Down.
    pub excess: Decimal,
    /// Pairs merged back to collateral during the window (the §8 capital-recycling
    /// mechanic). Cumulative over the window's life.
    pub merged_pairs: Size,
    /// Cumulative taker fees paid on this window.
    pub fees_paid: Dollars,
    /// Net realized PnL for the window: trading cash flow (buys, sells, fees,
    /// merge credits) plus the winning side's $1/share payout.
    pub realized_pnl: Dollars,
    /// Settlement time (the window's close time).
    pub ts: TimestampMs,
}

impl SettlementSummary {
    /// The single construction path: derives `matched_pairs`/`pair_cost`/`excess`
    /// from the two sides via [`InventorySnapshot::derive`] so they cannot drift.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "a flat closing snapshot of one window"
    )]
    pub fn close(
        window: WindowId,
        outcome: Outcome,
        up: SideInventory,
        down: SideInventory,
        merged_pairs: Size,
        fees_paid: Dollars,
        realized_pnl: Dollars,
        ts: TimestampMs,
    ) -> Self {
        let derived = InventorySnapshot::derive(window, up, down, ts);
        Self {
            window,
            outcome,
            up,
            down,
            matched_pairs: derived.matched_pairs,
            pair_cost: derived.pair_cost,
            excess: derived.excess,
            merged_pairs,
            fees_paid,
            realized_pnl,
            ts,
        }
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::series::{Asset, Series, WindowDuration};

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(0),
        }
    }

    fn side(shares: Decimal, cost: Decimal) -> SideInventory {
        SideInventory {
            shares: Size::new(shares).unwrap(),
            cost: Dollars::new(cost),
        }
    }

    #[test]
    fn avg_price_none_when_empty() {
        assert_eq!(SideInventory::default().avg_price(), None);
        assert_eq!(side(dec!(10), dec!(4.5)).avg_price(), Some(dec!(0.45)));
    }

    #[test]
    fn derive_balanced_book() {
        // 100 Up @ 0.48, 100 Down @ 0.49 — fully matched, pair cost 0.97.
        let snap = InventorySnapshot::derive(
            window(),
            side(dec!(100), dec!(48)),
            side(dec!(100), dec!(49)),
            TimestampMs::from_millis(1),
        );
        assert_eq!(snap.matched_pairs, Size::new(dec!(100)).unwrap());
        assert_eq!(snap.pair_cost, Some(dec!(0.97)));
        assert_eq!(snap.excess, Decimal::ZERO);
        assert_eq!(snap.excess_side(), None);
        assert_eq!(snap.worst_case_if_excess_loses, Dollars::ZERO);
    }

    #[test]
    fn derive_up_excess() {
        // 150 Up @ 0.40, 100 Down @ 0.50 — 50 Up excess costing 0.40 each.
        let snap = InventorySnapshot::derive(
            window(),
            side(dec!(150), dec!(60)),
            side(dec!(100), dec!(50)),
            TimestampMs::from_millis(1),
        );
        assert_eq!(snap.matched_pairs, Size::new(dec!(100)).unwrap());
        assert_eq!(snap.pair_cost, Some(dec!(0.90)));
        assert_eq!(snap.excess, dec!(50));
        assert_eq!(snap.excess_side(), Some(Outcome::Up));
        assert_eq!(snap.worst_case_if_excess_loses, Dollars::new(dec!(20)));
    }

    #[test]
    fn derive_down_excess_and_one_sided() {
        // Only Down held: no pairs, no pair cost, all of it is excess.
        let snap = InventorySnapshot::derive(
            window(),
            SideInventory::default(),
            side(dec!(40), dec!(10)),
            TimestampMs::from_millis(1),
        );
        assert_eq!(snap.matched_pairs, Size::ZERO);
        assert_eq!(snap.pair_cost, None);
        assert_eq!(snap.excess, dec!(-40));
        assert_eq!(snap.excess_side(), Some(Outcome::Down));
        assert_eq!(snap.worst_case_if_excess_loses, Dollars::new(dec!(10)));
    }

    #[test]
    fn settlement_summary_close_derives_pair_fields() {
        // 150 Up @ 0.40, 100 Down @ 0.50: 100 matched pairs @ 0.90, 50 Up excess.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side(dec!(150), dec!(60)),
            side(dec!(100), dec!(50)),
            Size::new(dec!(20)).unwrap(),
            Dollars::new(dec!(1.25)),
            Dollars::new(dec!(7.5)),
            TimestampMs::from_millis(300_000),
        );
        // Derived fields match InventorySnapshot::derive exactly.
        assert_eq!(summary.matched_pairs, Size::new(dec!(100)).unwrap());
        assert_eq!(summary.pair_cost, Some(dec!(0.90)));
        assert_eq!(summary.excess, dec!(50));
        // Pass-through fields preserved.
        assert_eq!(summary.outcome, Outcome::Up);
        assert_eq!(summary.merged_pairs, Size::new(dec!(20)).unwrap());
        assert_eq!(summary.fees_paid, Dollars::new(dec!(1.25)));
        assert_eq!(summary.realized_pnl, Dollars::new(dec!(7.5)));
        assert_eq!(summary.ts, TimestampMs::from_millis(300_000));
    }

    #[test]
    fn settlement_summary_serde_roundtrip() {
        let summary = SettlementSummary::close(
            window(),
            Outcome::Down,
            side(dec!(10), dec!(4)),
            side(dec!(10), dec!(5)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(1)),
            TimestampMs::from_millis(300_000),
        );
        let json = serde_json::to_string(&summary).unwrap();
        let back: SettlementSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, summary);
    }
}
