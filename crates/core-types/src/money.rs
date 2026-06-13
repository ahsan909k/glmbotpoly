//! Decimal money newtypes: non-negative share quantities and signed dollars,
//! plus the single shared taker-fee implementation (CLAUDE.md §7) used by the
//! paper venue, the engine's taker math, and analytics alike.

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Neg, Sub};

use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};

use crate::price::Price;

/// Error constructing a [`Size`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("size must be non-negative, got {0}")]
pub struct SizeError(pub Decimal);

/// A non-negative share quantity.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(try_from = "Decimal", into = "Decimal")]
pub struct Size(Decimal);

impl Size {
    /// Zero shares.
    pub const ZERO: Self = Self(Decimal::ZERO);

    /// Validates a non-negative quantity.
    ///
    /// # Errors
    /// [`SizeError`] if `d` is negative.
    pub fn new(d: Decimal) -> Result<Self, SizeError> {
        if d.is_sign_negative() && !d.is_zero() {
            Err(SizeError(d))
        } else {
            Ok(Self(d.normalize()))
        }
    }

    /// True for exactly zero shares.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0.is_zero()
    }

    /// Subtraction that returns `None` instead of going negative.
    #[must_use]
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        if other.0 > self.0 {
            None
        } else {
            Some(Self(self.0 - other.0))
        }
    }

    /// Subtraction that floors at zero.
    #[must_use]
    pub fn saturating_sub(self, other: Self) -> Self {
        self.checked_sub(other).unwrap_or(Self::ZERO)
    }

    /// The smaller of two sizes (matched-pairs math).
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        if self.0 <= other.0 { self } else { other }
    }

    /// Rounds **down** to `dp` decimal places — venue size precision must
    /// never be exceeded, and rounding a size up could oversize an order.
    #[must_use]
    pub fn round_down_dp(self, dp: u32) -> Self {
        Self(
            self.0
                .round_dp_with_strategy(dp, RoundingStrategy::ToZero)
                .normalize(),
        )
    }

    /// The underlying decimal value.
    #[must_use]
    pub const fn as_decimal(self) -> Decimal {
        self.0
    }
}

impl Add for Size {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sum for Size {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<Decimal> for Size {
    type Error = SizeError;
    fn try_from(d: Decimal) -> Result<Self, Self::Error> {
        Self::new(d)
    }
}

impl From<Size> for Decimal {
    fn from(s: Size) -> Self {
        s.0
    }
}

/// A signed dollar amount (PnL, fees, balances). Full ring arithmetic is
/// safe — negative dollars are meaningful.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Dollars(Decimal);

impl Dollars {
    /// Zero dollars.
    pub const ZERO: Self = Self(Decimal::ZERO);

    /// Wraps a signed decimal dollar amount.
    #[must_use]
    pub fn new(d: Decimal) -> Self {
        Self(d.normalize())
    }

    /// True for amounts strictly below zero.
    #[must_use]
    pub fn is_negative(self) -> bool {
        self.0.is_sign_negative() && !self.0.is_zero()
    }

    /// True for exactly zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0.is_zero()
    }

    /// Rounds to `dp` decimal places, half away from zero (display/ledger).
    #[must_use]
    pub fn round_dp(self, dp: u32) -> Self {
        Self(
            self.0
                .round_dp_with_strategy(dp, RoundingStrategy::MidpointAwayFromZero)
                .normalize(),
        )
    }

    /// The underlying decimal value.
    #[must_use]
    pub const fn as_decimal(self) -> Decimal {
        self.0
    }
}

impl Add for Dollars {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Dollars {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Dollars {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Sum for Dollars {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl fmt::Display for Dollars {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Decimal> for Dollars {
    fn from(d: Decimal) -> Self {
        Self::new(d)
    }
}

impl From<Dollars> for Decimal {
    fn from(d: Dollars) -> Self {
        d.0
    }
}

/// The protocol taker fee (CLAUDE.md §7, docs "Fees"):
/// `fee = shares × feeRate × p × (1 − p)`, in dollars, rounded to 5 decimal
/// places, with a minimum charged fee of $0.00001 on any non-zero charge.
/// Makers pay zero — callers apply this to taker fills only, with the
/// per-market `feeRate` read at discovery (never the hardcoded 0.07).
#[must_use]
pub fn taker_fee(size: Size, fee_rate: Decimal, price: Price) -> Dollars {
    let p = price.as_decimal();
    let raw = size.as_decimal() * fee_rate * p * (Decimal::ONE - p);
    if raw.is_zero() {
        return Dollars::ZERO;
    }
    let rounded = raw.round_dp_with_strategy(5, RoundingStrategy::MidpointAwayFromZero);
    let min_fee = Decimal::from_parts(1, 0, 0, false, 5); // 0.00001
    Dollars::new(rounded.max(min_fee))
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::tick::{RoundDir, TickSize};

    fn price(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T00001, RoundDir::Down).unwrap()
    }

    fn size(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }

    #[test]
    fn size_rejects_negative() {
        assert_eq!(Size::new(dec!(-1)), Err(SizeError(dec!(-1))));
        assert!(Size::new(dec!(0)).is_ok());
        assert!(Size::new(dec!(-0.0)).is_ok()); // negative zero is zero
    }

    #[test]
    fn size_subtraction_paths() {
        let five = size(dec!(5));
        let three = size(dec!(3));
        assert_eq!(five.checked_sub(three), Some(size(dec!(2))));
        assert_eq!(three.checked_sub(five), None);
        assert_eq!(three.saturating_sub(five), Size::ZERO);
        assert_eq!(five.saturating_sub(three), size(dec!(2)));
    }

    #[test]
    fn size_min_and_round_down() {
        assert_eq!(size(dec!(5)).min(size(dec!(3))), size(dec!(3)));
        assert_eq!(size(dec!(10.129)).round_down_dp(2), size(dec!(10.12)));
        assert_eq!(size(dec!(10.999)).round_down_dp(0), size(dec!(10)));
    }

    #[test]
    fn dollars_arithmetic() {
        let a = Dollars::new(dec!(1.50));
        let b = Dollars::new(dec!(2.25));
        assert_eq!(a + b, Dollars::new(dec!(3.75)));
        assert_eq!(a - b, Dollars::new(dec!(-0.75)));
        assert_eq!(-a, Dollars::new(dec!(-1.50)));
        assert!((a - b).is_negative());
        assert!(!a.is_negative());
        let total: Dollars = [a, b, -a].into_iter().sum();
        assert_eq!(total, b);
    }

    #[test]
    fn displays() {
        assert_eq!(size(dec!(0.5)).to_string(), "0.5");
        assert_eq!(Dollars::new(dec!(-1.25)).to_string(), "-1.25");
    }

    #[test]
    fn serde_size_rejects_negative() {
        assert!(serde_json::from_str::<Size>("\"-1\"").is_err());
        let ok: Size = serde_json::from_str("\"2.5\"").unwrap();
        assert_eq!(ok, size(dec!(2.5)));
    }

    // ---- taker_fee: the §7 reference points -------------------------------

    #[test]
    fn fee_reference_points_per_100_shares() {
        let hundred = size(dec!(100));
        let rate = dec!(0.07);
        // $1.75 at p=0.50
        assert_eq!(
            taker_fee(hundred, rate, price(dec!(0.50))),
            Dollars::new(dec!(1.75))
        );
        // $1.3125 at 0.25 / 0.75 (docs round the display to $1.31)
        assert_eq!(
            taker_fee(hundred, rate, price(dec!(0.25))),
            Dollars::new(dec!(1.3125))
        );
        assert_eq!(
            taker_fee(hundred, rate, price(dec!(0.75))),
            Dollars::new(dec!(1.3125))
        );
        // $0.63 at 0.10 / 0.90
        assert_eq!(
            taker_fee(hundred, rate, price(dec!(0.10))),
            Dollars::new(dec!(0.63))
        );
        // $0.0693 at 0.01 / 0.99 (docs display $0.07)
        assert_eq!(
            taker_fee(hundred, rate, price(dec!(0.01))),
            Dollars::new(dec!(0.0693))
        );
    }

    #[test]
    fn fee_symmetric_around_half() {
        for (a, b) in [(dec!(0.3), dec!(0.7)), (dec!(0.05), dec!(0.95))] {
            assert_eq!(
                taker_fee(size(dec!(37)), dec!(0.07), price(a)),
                taker_fee(size(dec!(37)), dec!(0.07), price(b))
            );
        }
    }

    #[test]
    fn fee_rounds_to_5dp() {
        // 7 × 0.07 × 0.333 × 0.667 = 0.10883439 → 0.10883
        let fee = taker_fee(size(dec!(7)), dec!(0.07), price(dec!(0.333)));
        assert_eq!(fee, Dollars::new(dec!(0.10883)));
        // 1 × 0.07 × 0.105 × 0.895 = 0.00657825 → rounds up to 0.00658.
        let fee = taker_fee(size(dec!(1)), dec!(0.07), price(dec!(0.105)));
        assert_eq!(fee, Dollars::new(dec!(0.00658)));
    }

    #[test]
    fn fee_minimum_floor() {
        // Tiny trade: 0.01 × 0.07 × 0.0001 × 0.9999 ≈ 0.00000007 → floors at 0.00001.
        let fee = taker_fee(size(dec!(0.01)), dec!(0.07), price(dec!(0.0001)));
        assert_eq!(fee, Dollars::new(dec!(0.00001)));
    }

    #[test]
    fn fee_zero_inputs_charge_nothing() {
        assert_eq!(
            taker_fee(Size::ZERO, dec!(0.07), price(dec!(0.5))),
            Dollars::ZERO
        );
        assert_eq!(
            taker_fee(size(dec!(100)), Decimal::ZERO, price(dec!(0.5))),
            Dollars::ZERO
        );
    }
}
