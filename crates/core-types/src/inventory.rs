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
}
