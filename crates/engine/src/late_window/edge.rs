//! Price-cap & depth sizing for the late-window certainty taker (CLAUDE.md §8).
//!
//! [`plan_certainty_take`] walks the displayed ask depth of the near-certain side
//! and decides how much to take with a FAK marketable BUY: it accepts each level
//! whose price is at or below the configured cap, caps the cumulative spend at the
//! remaining per-window budget, and requires the resulting notional to clear the
//! $1 venue minimum.
//!
//! Pure and sans-IO. Unlike the momentum taker's [`plan_take`](crate::plan_take),
//! there is **no edge/fee/buffer gate**: this deep in the tail the outcome is
//! near-decided and fees are near zero, so the accept criterion is simply "price ≤
//! cap". The fee is still computed exactly with the shared
//! [`core_types::taker_fee`] (the same function the paper and live venues charge
//! with) so the plan's reported `expected_fee` matches what the venue will deduct
//! — §9 conservatism, never assuming zero.

use core_types::{BookLevel, Decimal, Dollars, Outcome, Price, Size, taker_fee};

use super::NoLateTakeReason;

/// Venue minimum order notional ($1.00), mirroring `normalize`'s `MIN_NOTIONAL`.
/// A plan below this is rejected here so it never reaches — and is rejected by —
/// the normalizer.
const MIN_NOTIONAL: Decimal = Decimal::ONE;

/// A sized late-window take: a FAK marketable BUY of `notional` dollars on
/// `outcome`, capped at `worst_price`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CertaintyTakePlan {
    /// The near-certain outcome token to buy.
    pub outcome: Outcome,
    /// Worst-price slippage limit for the FAK = the highest accepted ask price
    /// (always at or below the configured cap).
    pub worst_price: Price,
    /// Dollar amount to spend (the FAK BUY `OrderQty::Notional`).
    pub notional: Dollars,
    /// Estimated shares the notional buys across the accepted levels.
    pub expected_shares: Size,
    /// Gross taker fee the venue will charge on the expected fill (matches
    /// [`core_types::taker_fee`] per level — tiny this deep in the tail, but
    /// never assumed zero).
    pub expected_fee: Dollars,
    /// Volume-weighted average executable price across the accepted levels (6 dp).
    pub vwap: Decimal,
}

/// Plan a certainty take on `outcome`, walking `asks` (ascending, best/lowest
/// first — the [`BookSnapshot`](core_types::BookSnapshot) invariant) and accepting
/// each level priced at or below `price_cap` until the displayed depth or the
/// `remaining_budget` runs out.
///
/// `fee_rate` is the per-market taker rate (the caller passes `0` when fees are
/// disabled). `worst_price` is the highest accepted ask.
///
/// # Errors
/// A typed [`NoLateTakeReason`] (for §12 observability): `NoAsks`,
/// `AllAsksAbovePriceCap` (best ask already above the cap), `BudgetExhausted`
/// (no affordable shares at all), or `BelowMinNotional` (sized below $1).
pub fn plan_certainty_take(
    outcome: Outcome,
    asks: &[BookLevel],
    fee_rate: Decimal,
    price_cap: Price,
    remaining_budget: Dollars,
) -> Result<CertaintyTakePlan, NoLateTakeReason> {
    let Some(first) = asks.first() else {
        return Err(NoLateTakeReason::NoAsks);
    };
    let cap = price_cap.as_decimal();
    if first.price.as_decimal() > cap {
        return Err(NoLateTakeReason::AllAsksAbovePriceCap {
            best_ask: first.price,
            cap: price_cap,
        });
    }

    let budget = remaining_budget.as_decimal();
    let mut taken = Decimal::ZERO;
    let mut notional = Decimal::ZERO;
    let mut gross_fee = Dollars::ZERO;
    let mut worst = first.price;

    for lvl in asks {
        let p = lvl.price.as_decimal();
        if p > cap {
            break; // asks ascending ⇒ everything deeper also exceeds the cap
        }
        // Budget truncation: take the lesser of displayed depth and what the
        // remaining budget affords at this price (`p` > 0 by the Price invariant).
        let remaining = budget - notional;
        if remaining <= Decimal::ZERO {
            break;
        }
        let depth = lvl.size.as_decimal();
        let affordable = remaining / p;
        let s = depth.min(affordable);
        if s <= Decimal::ZERO {
            break;
        }
        let Ok(s_size) = Size::new(s) else { break };

        taken += s;
        notional += s * p;
        gross_fee = gross_fee + taker_fee(s_size, fee_rate, lvl.price);
        worst = lvl.price;

        if s < depth {
            break; // budget bit this level — nothing left for deeper ones
        }
    }

    if taken.is_zero() {
        // Best ask was within the cap, but the budget afforded nothing.
        return Err(NoLateTakeReason::BudgetExhausted {
            remaining: remaining_budget,
            need: Dollars::new(MIN_NOTIONAL),
        });
    }
    if notional < MIN_NOTIONAL {
        return Err(NoLateTakeReason::BelowMinNotional {
            got: Dollars::new(notional),
        });
    }
    let Ok(expected_shares) = Size::new(taken) else {
        return Err(NoLateTakeReason::BelowMinNotional {
            got: Dollars::new(notional),
        });
    };
    let vwap = (notional / taken).round_dp(6);

    Ok(CertaintyTakePlan {
        outcome,
        worst_price: worst,
        notional: Dollars::new(notional),
        expected_shares,
        expected_fee: gross_fee,
        vwap,
    })
}

#[cfg(test)]
mod tests {
    use core_types::TickSize;
    use rust_decimal::dec;

    use super::*;

    const RATE: Decimal = Decimal::from_parts(7, 0, 0, false, 2); // 0.07
    const TICK: TickSize = TickSize::T001;

    fn ask(price: Decimal, size: Decimal) -> BookLevel {
        BookLevel {
            price: Price::on_grid(price, TICK).expect("grid price"),
            size: Size::new(size).expect("size"),
        }
    }

    fn cap(d: Decimal) -> Price {
        Price::on_grid(d, TICK).expect("grid cap")
    }

    fn plan(
        asks: &[BookLevel],
        price_cap: Decimal,
        budget: Decimal,
    ) -> Result<CertaintyTakePlan, NoLateTakeReason> {
        plan_certainty_take(
            Outcome::Up,
            asks,
            RATE,
            cap(price_cap),
            Dollars::new(budget),
        )
    }

    /// Takes the full displayed depth when it is at or below the cap and the
    /// budget is roomy; the fee equals the exact per-level `taker_fee`.
    #[test]
    fn takes_within_cap() {
        let asks = [ask(dec!(0.98), dec!(50))];
        let p = plan(&asks, dec!(0.99), dec!(100)).expect("a take");
        assert_eq!(p.outcome, Outcome::Up);
        assert_eq!(p.expected_shares, Size::new(dec!(50)).unwrap());
        assert_eq!(p.worst_price.as_decimal(), dec!(0.98));
        assert_eq!(p.notional, Dollars::new(dec!(49))); // 50·0.98
        // Gross fee = taker_fee(50, 0.07, 0.98) = 50·0.07·0.98·0.02 = 0.0686.
        assert_eq!(p.expected_fee, Dollars::new(dec!(0.0686)));
    }

    /// An ask exactly at the cap is accepted (inclusive); one tick above is not.
    #[test]
    fn exact_at_cap_boundary() {
        let asks = [ask(dec!(0.99), dec!(20))];
        // cap 0.99: ask == cap ⇒ taken.
        let p = plan(&asks, dec!(0.99), dec!(100)).expect("ask at cap is taken");
        assert_eq!(p.worst_price.as_decimal(), dec!(0.99));
        // cap 0.98: ask 0.99 > cap ⇒ nothing buyable.
        let r = plan(&asks, dec!(0.98), dec!(100)).unwrap_err();
        assert!(matches!(r, NoLateTakeReason::AllAsksAbovePriceCap { .. }));
    }

    /// Best ask above the cap ⇒ nothing buyable within the cap.
    #[test]
    fn all_asks_above_cap() {
        let asks = [ask(dec!(0.96), dec!(50))];
        let r = plan(&asks, dec!(0.95), dec!(100)).unwrap_err();
        assert!(matches!(r, NoLateTakeReason::AllAsksAbovePriceCap { .. }));
    }

    /// The remaining budget truncates the take mid-level.
    #[test]
    fn budget_caps_shares_mid_level() {
        // ask 0.98×100, budget $49 ⇒ affordable 50 shares (< 100).
        let asks = [ask(dec!(0.98), dec!(100))];
        let p = plan(&asks, dec!(0.99), dec!(49)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(50)).unwrap());
        assert_eq!(p.notional, Dollars::new(dec!(49)));
        assert!(p.notional.as_decimal() <= dec!(49));
    }

    /// A take sized below the $1 venue minimum is rejected.
    #[test]
    fn below_min_notional() {
        // ask 0.98×1 ⇒ notional $0.98 < $1.
        let asks = [ask(dec!(0.98), dec!(1))];
        let r = plan(&asks, dec!(0.99), dec!(100)).unwrap_err();
        assert!(matches!(r, NoLateTakeReason::BelowMinNotional { .. }));
    }

    /// Sizes across multiple levels up to the cap; worst price is the deepest
    /// accepted level.
    #[test]
    fn multi_level_depth() {
        let asks = [ask(dec!(0.97), dec!(30)), ask(dec!(0.98), dec!(40))];
        let p = plan(&asks, dec!(0.99), dec!(1000)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(70)).unwrap()); // 30 + 40
        assert_eq!(p.worst_price.as_decimal(), dec!(0.98));
        // 30·0.97 + 40·0.98 = 29.1 + 39.2 = 68.3.
        assert_eq!(p.notional, Dollars::new(dec!(68.3)));
        // VWAP = 68.3 / 70 = 0.975714 (6 dp).
        assert_eq!(p.vwap, dec!(0.975714));
    }

    /// A deeper level above the cap is excluded even with budget to spare.
    #[test]
    fn stops_at_first_level_above_cap() {
        let asks = [ask(dec!(0.97), dec!(30)), ask(dec!(0.98), dec!(40))];
        // cap 0.97 ⇒ only the 0.97 level is taken.
        let p = plan(&asks, dec!(0.97), dec!(10_000)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(30)).unwrap());
        assert_eq!(p.worst_price.as_decimal(), dec!(0.97));
    }

    #[test]
    fn empty_asks_is_no_asks() {
        assert!(matches!(
            plan(&[], dec!(0.99), dec!(100)),
            Err(NoLateTakeReason::NoAsks)
        ));
    }

    /// Even near the cap the fee is computed exactly (never assumed zero, §9).
    #[test]
    fn fee_computed_exactly_near_cap() {
        let asks = [ask(dec!(0.99), dec!(50))];
        let p = plan(&asks, dec!(0.99), dec!(100)).expect("a take");
        // taker_fee(50, 0.07, 0.99) = 50·0.07·0.99·0.01 = 0.03465.
        assert_eq!(p.expected_fee, Dollars::new(dec!(0.03465)));
        assert!(p.expected_fee.as_decimal() > Decimal::ZERO);
    }
}
