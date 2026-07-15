//! Depth & budget sizing for the champion-model taker (CLAUDE.md §8).
//!
//! [`plan_model_take`] walks the displayed ask depth and sizes a FAK marketable
//! BUY: it accepts every level (optionally at or below a configured price cap),
//! caps the cumulative spend at the remaining per-window budget, and requires the
//! resulting notional to clear the $1 venue minimum.
//!
//! Pure and sans-IO. Faithful to the tested recipe
//! (`money_judge.py::_run_taker`): **no** edge/fee/buffer gate and (by default)
//! **no** price cap — it takes at the displayed ask up to the budget. The fee is
//! still computed exactly with the shared [`core_types::taker_fee`] so the plan's
//! reported `expected_fee` matches what the venue deducts (§9), but it never
//! gates.

use core_types::{BookLevel, Decimal, Dollars, Outcome, Price, Size, taker_fee};

use super::NoModelTakeReason;

/// Venue minimum order notional ($1.00), mirroring `normalize`'s `MIN_NOTIONAL`.
const MIN_NOTIONAL: Decimal = Decimal::ONE;

/// A sized model take: a FAK marketable BUY of `notional` dollars on `outcome`,
/// capped at `worst_price`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelTakePlan {
    /// The model-predicted outcome token to buy.
    pub outcome: Outcome,
    /// Worst-price slippage limit for the FAK = the highest accepted ask price.
    pub worst_price: Price,
    /// Dollar amount to spend (the FAK BUY `OrderQty::Notional`).
    pub notional: Dollars,
    /// Estimated shares the notional buys across the accepted levels.
    pub expected_shares: Size,
    /// Gross taker fee the venue will charge (exact per level via
    /// [`core_types::taker_fee`]; reported, never gates).
    pub expected_fee: Dollars,
    /// Volume-weighted average executable price across the accepted levels (6 dp).
    pub vwap: Decimal,
}

/// Plan a model take on `outcome`, walking `asks` (ascending, best/lowest first —
/// the [`BookSnapshot`](core_types::BookSnapshot) invariant) and accepting each
/// level (at or below `price_cap` when set) until the displayed depth or the
/// `remaining_budget` runs out. **No edge/fee gate.**
///
/// `fee_rate` is the per-market taker rate (the caller passes `0` when fees are
/// disabled). `worst_price` is the highest accepted ask.
///
/// # Errors
/// A typed [`NoModelTakeReason`]: `NoAsks`, `AllAsksAbovePriceCap`,
/// `BudgetExhausted`, or `BelowMinNotional`.
pub fn plan_model_take(
    outcome: Outcome,
    asks: &[BookLevel],
    fee_rate: Decimal,
    price_cap: Option<Price>,
    remaining_budget: Dollars,
) -> Result<ModelTakePlan, NoModelTakeReason> {
    let Some(first) = asks.first() else {
        return Err(NoModelTakeReason::NoAsks);
    };
    if let Some(cap) = price_cap
        && first.price.as_decimal() > cap.as_decimal()
    {
        return Err(NoModelTakeReason::AllAsksAbovePriceCap {
            best_ask: first.price,
            cap,
        });
    }

    let budget = remaining_budget.as_decimal();
    let mut taken = Decimal::ZERO;
    let mut notional = Decimal::ZERO;
    let mut gross_fee = Dollars::ZERO;
    let mut worst = first.price;

    for lvl in asks {
        let p = lvl.price.as_decimal();
        if price_cap.is_some_and(|c| p > c.as_decimal()) {
            break; // asks ascending ⇒ everything deeper also exceeds the cap
        }
        // Budget truncation: the lesser of displayed depth and what the remaining
        // budget affords at this price (`p` > 0 by the Price invariant).
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
        return Err(NoModelTakeReason::BudgetExhausted {
            remaining: remaining_budget,
            need: Dollars::new(MIN_NOTIONAL),
        });
    }
    if notional < MIN_NOTIONAL {
        return Err(NoModelTakeReason::BelowMinNotional {
            got: Dollars::new(notional),
        });
    }
    let Ok(expected_shares) = Size::new(taken) else {
        return Err(NoModelTakeReason::BelowMinNotional {
            got: Dollars::new(notional),
        });
    };
    let vwap = (notional / taken).round_dp(6);

    Ok(ModelTakePlan {
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
        price_cap: Option<Price>,
        budget: Decimal,
    ) -> Result<ModelTakePlan, NoModelTakeReason> {
        plan_model_take(Outcome::Up, asks, RATE, price_cap, Dollars::new(budget))
    }

    /// No gate: takes at the displayed ask even at an unfavourable price, up to depth.
    #[test]
    fn takes_at_the_ask_no_gate() {
        let asks = [ask(dec!(0.80), dec!(50))];
        let p = plan(&asks, None, dec!(100)).expect("a take");
        assert_eq!(p.outcome, Outcome::Up);
        assert_eq!(p.expected_shares, Size::new(dec!(50)).unwrap());
        assert_eq!(p.worst_price.as_decimal(), dec!(0.80));
        assert_eq!(p.notional, Dollars::new(dec!(40))); // 50·0.80
        // Gross fee = taker_fee(50, 0.07, 0.80) = 50·0.07·0.80·0.20 = 0.56.
        assert_eq!(p.expected_fee, Dollars::new(dec!(0.56)));
    }

    /// The recipe's sizing: min(displayed depth, budget/price), capped at $10.
    #[test]
    fn sizes_min_depth_budget() {
        // ask 0.80×100, budget $10 ⇒ affordable 12.5 shares.
        let asks = [ask(dec!(0.80), dec!(100))];
        let p = plan(&asks, None, dec!(10)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(12.5)).unwrap());
        assert_eq!(p.notional, Dollars::new(dec!(10)));
    }

    /// Multi-level walk up to the budget; worst price is the deepest accepted level.
    #[test]
    fn multi_level_walk() {
        let asks = [ask(dec!(0.50), dec!(10)), ask(dec!(0.52), dec!(40))];
        let p = plan(&asks, None, dec!(1000)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(50)).unwrap()); // 10 + 40
        assert_eq!(p.worst_price.as_decimal(), dec!(0.52));
        // 10·0.50 + 40·0.52 = 5 + 20.8 = 25.8.
        assert_eq!(p.notional, Dollars::new(dec!(25.8)));
    }

    /// An optional price cap excludes levels above it (ascending ⇒ stop).
    #[test]
    fn optional_price_cap_excludes_deep_levels() {
        let asks = [ask(dec!(0.50), dec!(10)), ask(dec!(0.52), dec!(40))];
        let p = plan(&asks, Some(cap(dec!(0.50))), dec!(1000)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(10)).unwrap());
        assert_eq!(p.worst_price.as_decimal(), dec!(0.50));
        // Best ask above the cap ⇒ nothing buyable.
        let r = plan(&asks, Some(cap(dec!(0.49))), dec!(1000)).unwrap_err();
        assert!(matches!(r, NoModelTakeReason::AllAsksAbovePriceCap { .. }));
    }

    /// A take sized below the $1 venue minimum is rejected.
    #[test]
    fn below_min_notional() {
        let asks = [ask(dec!(0.50), dec!(1))]; // notional $0.50 < $1
        let r = plan(&asks, None, dec!(100)).unwrap_err();
        assert!(matches!(r, NoModelTakeReason::BelowMinNotional { .. }));
    }

    #[test]
    fn empty_asks_is_no_asks() {
        assert!(matches!(
            plan(&[], None, dec!(100)),
            Err(NoModelTakeReason::NoAsks)
        ));
    }
}
