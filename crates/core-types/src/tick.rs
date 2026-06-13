//! Tick-size grids and rounding directions.
//!
//! Polymarket supports four tick sizes. Our Up/Down markets trade at 0.01 and
//! flip to 0.001 when the price crosses 0.96 / 0.04 (the live
//! `tick_size_change` event) — the flip is venue behavior; this module only
//! represents the grids themselves.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::order::Side;

/// Price above which (resp. `1 -` below which) the venue flips the tick from
/// 0.01 to 0.001. Documented here so the magic number lives in one place;
/// the actual flip is driven by the live `tick_size_change` event.
pub const TICK_FLIP_HI: Decimal = Decimal::from_parts(96, 0, 0, false, 2); // 0.96
/// Lower flip threshold, mirror of [`TICK_FLIP_HI`].
pub const TICK_FLIP_LO: Decimal = Decimal::from_parts(4, 0, 0, false, 2); // 0.04

/// Error converting a decimal to a supported tick size.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported tick size {0} (expected 0.1, 0.01, 0.001 or 0.0001)")]
pub struct TickSizeError(pub Decimal);

/// One of the four tick sizes Polymarket supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TickSize {
    /// 0.1 — one decimal place.
    T01,
    /// 0.01 — two decimal places (our markets' normal regime).
    T001,
    /// 0.001 — three decimal places (our markets beyond 0.96 / 0.04).
    T0001,
    /// 0.0001 — four decimal places.
    T00001,
}

impl TickSize {
    /// Number of decimal places on this grid (1..=4).
    #[must_use]
    pub const fn decimal_places(self) -> u32 {
        match self {
            Self::T01 => 1,
            Self::T001 => 2,
            Self::T0001 => 3,
            Self::T00001 => 4,
        }
    }

    /// The tick as a decimal (0.1 / 0.01 / 0.001 / 0.0001).
    #[must_use]
    pub const fn as_decimal(self) -> Decimal {
        Decimal::from_parts(1, 0, 0, false, self.decimal_places())
    }

    /// Parses a tick size from its decimal value (as found on the market
    /// object or in `tick_size_change` events).
    ///
    /// # Errors
    /// [`TickSizeError`] if `d` is not numerically one of the four ticks.
    pub fn from_decimal(d: Decimal) -> Result<Self, TickSizeError> {
        for tick in [Self::T01, Self::T001, Self::T0001, Self::T00001] {
            if d == tick.as_decimal() {
                return Ok(tick);
            }
        }
        Err(TickSizeError(d))
    }

    /// Lowest valid price on this grid (= one tick).
    #[must_use]
    pub const fn min_price(self) -> Decimal {
        self.as_decimal()
    }

    /// Highest valid price on this grid (= 1 − one tick).
    #[must_use]
    pub const fn max_price(self) -> Decimal {
        let dp = self.decimal_places();
        // 1 - 10^-dp, expressed at scale dp: 9, 99, 999, 9999.
        let lo = match dp {
            1 => 9,
            2 => 99,
            3 => 999,
            _ => 9999,
        };
        Decimal::from_parts(lo, 0, 0, false, dp)
    }
}

/// Direction to round an off-grid price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoundDir {
    /// Toward zero / the bid side.
    Down,
    /// Toward one / the ask side.
    Up,
}

impl RoundDir {
    /// The passive-safety direction for a resting order on `side`:
    /// bids round **down**, asks round **up** — never round toward crossing.
    #[must_use]
    pub const fn passive_for(side: Side) -> Self {
        match side {
            Side::Buy => Self::Down,
            Side::Sell => Self::Up,
        }
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    #[test]
    fn decimal_values_match() {
        assert_eq!(TickSize::T01.as_decimal(), dec!(0.1));
        assert_eq!(TickSize::T001.as_decimal(), dec!(0.01));
        assert_eq!(TickSize::T0001.as_decimal(), dec!(0.001));
        assert_eq!(TickSize::T00001.as_decimal(), dec!(0.0001));
    }

    #[test]
    fn from_decimal_accepts_all_four() {
        assert_eq!(TickSize::from_decimal(dec!(0.1)), Ok(TickSize::T01));
        assert_eq!(TickSize::from_decimal(dec!(0.01)), Ok(TickSize::T001));
        assert_eq!(TickSize::from_decimal(dec!(0.001)), Ok(TickSize::T0001));
        assert_eq!(TickSize::from_decimal(dec!(0.0001)), Ok(TickSize::T00001));
        // Numerically-equal forms with different scales must also parse.
        assert_eq!(TickSize::from_decimal(dec!(0.100)), Ok(TickSize::T01));
    }

    #[test]
    fn from_decimal_rejects_others() {
        assert!(TickSize::from_decimal(dec!(0.05)).is_err());
        assert!(TickSize::from_decimal(dec!(0.2)).is_err());
        assert!(TickSize::from_decimal(dec!(0)).is_err());
        assert!(TickSize::from_decimal(dec!(1)).is_err());
    }

    #[test]
    fn min_max_prices() {
        assert_eq!(TickSize::T01.min_price(), dec!(0.1));
        assert_eq!(TickSize::T01.max_price(), dec!(0.9));
        assert_eq!(TickSize::T001.min_price(), dec!(0.01));
        assert_eq!(TickSize::T001.max_price(), dec!(0.99));
        assert_eq!(TickSize::T0001.min_price(), dec!(0.001));
        assert_eq!(TickSize::T0001.max_price(), dec!(0.999));
        assert_eq!(TickSize::T00001.min_price(), dec!(0.0001));
        assert_eq!(TickSize::T00001.max_price(), dec!(0.9999));
    }

    #[test]
    fn decimal_places_match() {
        assert_eq!(TickSize::T01.decimal_places(), 1);
        assert_eq!(TickSize::T001.decimal_places(), 2);
        assert_eq!(TickSize::T0001.decimal_places(), 3);
        assert_eq!(TickSize::T00001.decimal_places(), 4);
    }

    #[test]
    fn flip_thresholds() {
        assert_eq!(TICK_FLIP_HI, dec!(0.96));
        assert_eq!(TICK_FLIP_LO, dec!(0.04));
    }

    #[test]
    fn passive_direction() {
        assert_eq!(RoundDir::passive_for(Side::Buy), RoundDir::Down);
        assert_eq!(RoundDir::passive_for(Side::Sell), RoundDir::Up);
    }
}
