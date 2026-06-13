//! The `Price` newtype: a probability-space limit price that cannot exist
//! outside `(0, 1)` or off the finest supported grid.
//!
//! `Price` deliberately does **not** carry its [`TickSize`]: the venue flips
//! the tick live (`tick_size_change`), which would silently invalidate any
//! embedded tick. The type-level invariant is "strictly inside (0, 1), scale
//! ≤ 4"; membership on the *current* grid is enforced by the constructors,
//! and the engine's order normalizer must re-check against
//! `MarketInfo::tick_size` immediately before every placement.

use std::fmt;

use rust_decimal::prelude::FromPrimitive;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};

use crate::tick::{RoundDir, TickSize};

/// Errors from price construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PriceError {
    /// Value not strictly inside (0, 1) — or outside [0, 1] for quantizing
    /// constructors, which accept the closed interval and clamp the ends.
    #[error("price {0} outside the valid domain")]
    OutOfDomain(Decimal),
    /// Value inside (0, 1) but outside [tick, 1 − tick] for the given grid.
    #[error("price {got} outside [{min}, {max}]")]
    OutOfRange {
        /// The offending value.
        got: Decimal,
        /// Lowest valid price on the grid.
        min: Decimal,
        /// Highest valid price on the grid.
        max: Decimal,
    },
    /// Value in range but not on the tick grid.
    #[error("price {got} not on the {tick:?} grid")]
    OffGrid {
        /// The offending value.
        got: Decimal,
        /// The grid it missed.
        tick: TickSize,
    },
    /// Float input was NaN or infinite.
    #[error("non-finite float {0}")]
    NonFinite(f64),
    /// Float input outside [0.0, 1.0].
    #[error("float {0} outside [0.0, 1.0]")]
    FloatOutOfDomain(f64),
}

/// A validated order-book price, strictly inside `(0, 1)`, scale ≤ 4.
///
/// There is intentionally no arithmetic on `Price` — do math in [`Decimal`]
/// space via [`Price::as_decimal`] and re-enter through a fallible
/// constructor, so an invalid price can never be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "Decimal", into = "Decimal")]
pub struct Price(Decimal);

impl Price {
    /// Internal: validate the type-level invariant (0 < p < 1, scale ≤ 4)
    /// and store normalized so `0.50 == 0.500` and hashing agrees.
    fn from_domain(d: Decimal) -> Result<Self, PriceError> {
        if d <= Decimal::ZERO || d >= Decimal::ONE {
            return Err(PriceError::OutOfDomain(d));
        }
        let n = d.normalize();
        if n.scale() > 4 {
            // Finer than the finest supported grid (0.0001).
            return Err(PriceError::OffGrid {
                got: d,
                tick: TickSize::T00001,
            });
        }
        Ok(Self(n))
    }

    /// Exact constructor: `d` must lie in `[tick, 1 − tick]` **and** exactly
    /// on the grid. Use for prices received from the venue, which are already
    /// supposed to be valid — an error here means our view of the tick is
    /// stale or the input is corrupt.
    ///
    /// # Errors
    /// [`PriceError::OutOfDomain`], [`PriceError::OutOfRange`] or
    /// [`PriceError::OffGrid`].
    pub fn on_grid(d: Decimal, tick: TickSize) -> Result<Self, PriceError> {
        let p = Self::from_domain(d)?;
        if !p.in_range(tick) {
            return Err(PriceError::OutOfRange {
                got: d,
                min: tick.min_price(),
                max: tick.max_price(),
            });
        }
        if !p.is_on_grid(tick) {
            return Err(PriceError::OffGrid { got: d, tick });
        }
        Ok(p)
    }

    /// Quantizing constructor: rounds `d` onto the grid in direction `dir`,
    /// then clamps into `[tick, 1 − tick]`.
    ///
    /// Accepts the closed domain `[0, 1]`: the endpoints quantize to the
    /// nearest legal price (0 → tick, 1 → 1 − tick). Note the clamp can
    /// override `dir` at the extremes (e.g. 0.005 rounding *down* on the
    /// 0.01 grid still yields 0.01) — the floor/ceiling are the only legal
    /// prices there. Callers that must never cross should use
    /// [`Price::on_grid`] and treat the error.
    ///
    /// # Errors
    /// [`PriceError::OutOfDomain`] if `d < 0` or `d > 1`.
    pub fn quantize(d: Decimal, tick: TickSize, dir: RoundDir) -> Result<Self, PriceError> {
        if d < Decimal::ZERO || d > Decimal::ONE {
            return Err(PriceError::OutOfDomain(d));
        }
        let strategy = match dir {
            RoundDir::Down => RoundingStrategy::ToNegativeInfinity,
            RoundDir::Up => RoundingStrategy::ToPositiveInfinity,
        };
        let rounded = d.round_dp_with_strategy(tick.decimal_places(), strategy);
        let clamped = rounded.clamp(tick.min_price(), tick.max_price());
        Self::from_domain(clamped)
    }

    /// Float boundary: converts a model probability (e.g. `p_up`) to a grid
    /// price. Rejects NaN/±∞ and values outside `[0.0, 1.0]`; in-domain
    /// values are converted via shortest-round-trip decimal conversion and
    /// then [quantized](Price::quantize).
    ///
    /// Shortest round-trip matters: `0.29_f64` must become `0.29`, not the
    /// bit-exact `0.2899999…`, which would round Down to `0.28` — a silent
    /// one-tick pricing error.
    ///
    /// # Errors
    /// [`PriceError::NonFinite`] or [`PriceError::FloatOutOfDomain`].
    pub fn from_f64(p: f64, tick: TickSize, dir: RoundDir) -> Result<Self, PriceError> {
        if !p.is_finite() {
            return Err(PriceError::NonFinite(p));
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(PriceError::FloatOutOfDomain(p));
        }
        // Shortest round-trip (NOT from_f64_retain — see doc comment).
        let d = Decimal::from_f64(p).ok_or(PriceError::FloatOutOfDomain(p))?;
        Self::quantize(d, tick, dir)
    }

    /// True if the price lies exactly on the given tick grid.
    #[must_use]
    pub fn is_on_grid(self, tick: TickSize) -> bool {
        self.0.round_dp(tick.decimal_places()) == self.0
    }

    /// True if the price lies within `[tick, 1 − tick]` for the given grid.
    #[must_use]
    pub fn in_range(self, tick: TickSize) -> bool {
        self.0 >= tick.min_price() && self.0 <= tick.max_price()
    }

    /// The underlying decimal value (normalized).
    #[must_use]
    pub const fn as_decimal(self) -> Decimal {
        self.0
    }

    /// `1 − p`: the same price seen from the opposite outcome (Up ↔ Down).
    /// Always valid because the domain is symmetric and open.
    #[must_use]
    pub fn complement(self) -> Self {
        // Invariant-preserving: p ∈ (0,1), scale ≤ 4 ⇒ 1−p ∈ (0,1), scale ≤ 4.
        Self((Decimal::ONE - self.0).normalize())
    }

    /// Boundary conversion out to the float-based model. Lossless for all
    /// scale-≤-4 values.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        // Every (0,1) decimal with scale ≤ 4 is exactly representable enough:
        // to_f64 on rust_decimal never fails for in-range values; fall back
        // to 0.5 is unreachable but keeps the API panic-free.
        self.0.to_f64().unwrap_or(0.5)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<Decimal> for Price {
    type Error = PriceError;

    /// Serde/journal entry point: enforces the domain invariant but cannot
    /// know the market's current tick — grid checks happen at use sites.
    fn try_from(d: Decimal) -> Result<Self, Self::Error> {
        Self::from_domain(d)
    }
}

impl From<Price> for Decimal {
    fn from(p: Price) -> Self {
        p.0
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::tick::{RoundDir as Dir, TickSize as T};

    fn p(d: Decimal, tick: T) -> Price {
        Price::on_grid(d, tick).unwrap()
    }

    // ---- 0.01 regime: on_grid -------------------------------------------

    #[test]
    fn on_grid_t001_accepts_grid_values() {
        assert_eq!(p(dec!(0.50), T::T001).as_decimal(), dec!(0.5));
        assert_eq!(p(dec!(0.01), T::T001).as_decimal(), dec!(0.01)); // lower bound
        assert_eq!(p(dec!(0.99), T::T001).as_decimal(), dec!(0.99)); // upper bound
    }

    #[test]
    fn on_grid_t001_rejects_off_grid() {
        assert!(matches!(
            Price::on_grid(dec!(0.505), T::T001),
            Err(PriceError::OffGrid { .. })
        ));
    }

    #[test]
    fn on_grid_t001_rejects_out_of_range() {
        assert!(matches!(
            Price::on_grid(dec!(0.005), T::T001),
            Err(PriceError::OutOfRange { .. })
        ));
        assert!(matches!(
            Price::on_grid(dec!(0.995), T::T001),
            Err(PriceError::OutOfRange { .. })
        ));
    }

    #[test]
    fn on_grid_rejects_out_of_domain() {
        for bad in [dec!(0), dec!(1), dec!(-0.01), dec!(1.01)] {
            assert!(matches!(
                Price::on_grid(bad, T::T001),
                Err(PriceError::OutOfDomain(_))
            ));
        }
    }

    #[test]
    fn trailing_zeros_normalize() {
        // Scale-3 "0.500" equals scale-2 "0.50" after construction.
        assert_eq!(p(dec!(0.500), T::T001), p(dec!(0.50), T::T001));
    }

    // ---- 0.01 regime: quantize ------------------------------------------

    #[test]
    fn quantize_t001_rounds_directionally() {
        let q = |d, dir| Price::quantize(d, T::T001, dir).unwrap().as_decimal();
        assert_eq!(q(dec!(0.5049), Dir::Down), dec!(0.5));
        assert_eq!(q(dec!(0.5001), Dir::Up), dec!(0.51));
        // On-grid input is a fixed point in both directions.
        assert_eq!(q(dec!(0.50), Dir::Down), dec!(0.5));
        assert_eq!(q(dec!(0.50), Dir::Up), dec!(0.5));
        assert_eq!(q(dec!(0.0149), Dir::Down), dec!(0.01));
    }

    #[test]
    fn quantize_t001_clamps_at_bounds() {
        let q = |d, dir| Price::quantize(d, T::T001, dir).unwrap().as_decimal();
        // Clamp overrides direction at the floor/ceiling (documented).
        assert_eq!(q(dec!(0.009), Dir::Down), dec!(0.01));
        assert_eq!(q(dec!(0.995), Dir::Up), dec!(0.99));
        assert_eq!(q(dec!(1.0), Dir::Down), dec!(0.99));
        assert_eq!(q(dec!(0.0), Dir::Up), dec!(0.01));
        assert_eq!(q(dec!(1.0), Dir::Up), dec!(0.99));
        assert_eq!(q(dec!(0.0), Dir::Down), dec!(0.01));
    }

    #[test]
    fn quantize_rejects_outside_closed_domain() {
        assert!(matches!(
            Price::quantize(dec!(1.5), T::T001, Dir::Down),
            Err(PriceError::OutOfDomain(_))
        ));
        assert!(matches!(
            Price::quantize(dec!(-0.1), T::T001, Dir::Up),
            Err(PriceError::OutOfDomain(_))
        ));
    }

    // ---- 0.001 regime ----------------------------------------------------

    #[test]
    fn on_grid_t0001() {
        assert_eq!(p(dec!(0.965), T::T0001).as_decimal(), dec!(0.965));
        assert_eq!(p(dec!(0.001), T::T0001).as_decimal(), dec!(0.001)); // bounds
        assert_eq!(p(dec!(0.999), T::T0001).as_decimal(), dec!(0.999));
        // Scale-4 value on the scale-3 grid.
        assert!(matches!(
            Price::on_grid(dec!(0.9605), T::T0001),
            Err(PriceError::OffGrid { .. })
        ));
    }

    #[test]
    fn quantize_t0001() {
        let q = |d, dir| Price::quantize(d, T::T0001, dir).unwrap().as_decimal();
        assert_eq!(q(dec!(0.9605), Dir::Down), dec!(0.96));
        assert_eq!(q(dec!(0.9605), Dir::Up), dec!(0.961));
        assert_eq!(q(dec!(0.9995), Dir::Up), dec!(0.999)); // clamp
        assert_eq!(q(dec!(0.0004), Dir::Down), dec!(0.001)); // clamp
    }

    // ---- 0.96 / 0.04 flip boundary ---------------------------------------

    #[test]
    fn fine_grid_price_invalid_after_coarse_flip() {
        let fine = p(dec!(0.965), T::T0001);
        assert!(fine.is_on_grid(T::T0001));
        assert!(!fine.is_on_grid(T::T001)); // flip back to 0.01 invalidates it
    }

    #[test]
    fn coarse_grid_prices_stay_valid_on_fine_grid() {
        // Grid nesting: every 0.01 price is also a 0.001 price.
        for v in [dec!(0.04), dec!(0.05), dec!(0.96), dec!(0.97)] {
            let coarse = p(v, T::T001);
            assert!(coarse.is_on_grid(T::T0001), "{v} should nest");
            assert!(coarse.in_range(T::T0001), "{v} should stay in range");
        }
    }

    #[test]
    fn requantize_after_coarse_flip() {
        assert_eq!(
            Price::quantize(dec!(0.965), T::T001, Dir::Down)
                .unwrap()
                .as_decimal(),
            dec!(0.96)
        );
        assert_eq!(
            Price::quantize(dec!(0.965), T::T001, Dir::Up)
                .unwrap()
                .as_decimal(),
            dec!(0.97)
        );
    }

    #[test]
    fn range_shrinks_on_coarse_flip() {
        // 0.995 is legal at T0001 but out of range at T001.
        let fine = p(dec!(0.995), T::T0001);
        assert!(!fine.in_range(T::T001));
        assert_eq!(
            Price::quantize(dec!(0.995), T::T001, Dir::Up)
                .unwrap()
                .as_decimal(),
            dec!(0.99)
        );
    }

    #[test]
    fn around_the_flip_thresholds() {
        // 0.96 / 0.04 themselves and the 0.001-grid neighbors.
        for v in [dec!(0.04), dec!(0.96)] {
            assert!(p(v, T::T001).is_on_grid(T::T0001));
        }
        for v in [dec!(0.039), dec!(0.041), dec!(0.959), dec!(0.961)] {
            let price = p(v, T::T0001);
            assert!(price.is_on_grid(T::T0001));
            assert!(!price.is_on_grid(T::T001));
        }
        // 0.040 / 0.960 written at scale 3 normalize onto the coarse grid.
        assert!(p(dec!(0.040), T::T0001).is_on_grid(T::T001));
        assert!(p(dec!(0.960), T::T0001).is_on_grid(T::T001));
    }

    // ---- from_f64 boundary -----------------------------------------------

    #[test]
    fn from_f64_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                Price::from_f64(bad, T::T001, Dir::Down),
                Err(PriceError::NonFinite(_))
            ));
        }
    }

    #[test]
    fn from_f64_rejects_out_of_domain() {
        assert!(matches!(
            Price::from_f64(-0.1, T::T001, Dir::Down),
            Err(PriceError::FloatOutOfDomain(_))
        ));
        assert!(matches!(
            Price::from_f64(1.000_000_1, T::T001, Dir::Up),
            Err(PriceError::FloatOutOfDomain(_))
        ));
    }

    #[test]
    fn from_f64_clamps_saturated_probabilities() {
        let f = |v: f64, tick, dir| Price::from_f64(v, tick, dir).unwrap().as_decimal();
        assert_eq!(f(0.0, T::T001, Dir::Down), dec!(0.01));
        assert_eq!(f(1.0, T::T001, Dir::Up), dec!(0.99));
        assert_eq!(f(-0.0, T::T001, Dir::Down), dec!(0.01)); // −0.0 is in-domain
        assert_eq!(f(f64::MIN_POSITIVE, T::T001, Dir::Down), dec!(0.01));
        assert_eq!(f(0.999_999_9, T::T0001, Dir::Up), dec!(0.999));
    }

    #[test]
    fn from_f64_uses_shortest_roundtrip() {
        // 0.29_f64 is bit-exactly 0.2899999…; retain-style conversion would
        // round Down to 0.28. Shortest round-trip must give 0.29.
        assert_eq!(
            Price::from_f64(0.29, T::T001, Dir::Down)
                .unwrap()
                .as_decimal(),
            dec!(0.29)
        );
        // Classic float artifact: 0.1 + 0.2 = 0.30000000000000004.
        #[allow(clippy::suboptimal_flops)]
        let sum = 0.1_f64 + 0.2_f64;
        assert_eq!(
            Price::from_f64(sum, T::T001, Dir::Down)
                .unwrap()
                .as_decimal(),
            dec!(0.3)
        );
        assert_eq!(
            Price::from_f64(0.07, T::T001, Dir::Down)
                .unwrap()
                .as_decimal(),
            dec!(0.07)
        );
        assert_eq!(
            Price::from_f64(0.5649, T::T001, Dir::Down)
                .unwrap()
                .as_decimal(),
            dec!(0.56)
        );
    }

    // ---- misc --------------------------------------------------------------

    #[test]
    fn complement_is_involutive() {
        let price = p(dec!(0.97), T::T001);
        assert_eq!(price.complement().as_decimal(), dec!(0.03));
        assert_eq!(price.complement().complement(), price);
        let fine = p(dec!(0.999), T::T0001);
        assert_eq!(fine.complement().as_decimal(), dec!(0.001));
    }

    #[test]
    fn to_f64_roundtrips_through_model_boundary() {
        let price = p(dec!(0.565), T::T0001);
        let back = Price::from_f64(price.to_f64(), T::T0001, Dir::Down).unwrap();
        assert_eq!(back, price);
    }

    #[test]
    fn serde_roundtrip_and_rejection() {
        let price = p(dec!(0.57), T::T001);
        let json = serde_json::to_string(&price).unwrap();
        assert_eq!(json, "\"0.57\""); // Decimal serializes as a string
        let back: Price = serde_json::from_str(&json).unwrap();
        assert_eq!(back, price);
        assert!(serde_json::from_str::<Price>("\"1.5\"").is_err());
        assert!(serde_json::from_str::<Price>("\"0\"").is_err());
        assert!(serde_json::from_str::<Price>("\"0.00001\"").is_err()); // scale 5
    }

    #[test]
    fn display_is_normalized() {
        assert_eq!(p(dec!(0.500), T::T001).to_string(), "0.5");
        assert_eq!(p(dec!(0.965), T::T0001).to_string(), "0.965");
    }
}
