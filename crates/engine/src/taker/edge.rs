//! Fee-aware edge & depth sizing for the momentum taker (CLAUDE.md §8).
//!
//! [`plan_take`] walks the displayed ask depth of the side a confirmed move
//! favors and decides whether — and how much — to take with a FAK marketable
//! BUY: it accepts each level while the per-share net edge (model fair minus the
//! ask price, after the **exact per-market taker fee at that price** plus the
//! configured buffer) stays positive, caps the cumulative spend at the remaining
//! per-window budget, and requires the resulting notional to clear the $1 venue
//! minimum.
//!
//! Pure and sans-IO. The fee is computed with the shared
//! [`core_types::taker_fee`] (the same function the paper and live venues charge
//! with), so the plan's reported `expected_fee` matches what the venue will
//! deduct — §9 conservatism: the decision can never claim an edge the fill won't
//! realize.
//!
//! ## Why a per-share marginal gate is correct
//!
//! The per-share fee `rate·p·(1−p)` is *not* monotone in `p` (it peaks at 0.5),
//! but the per-share *net* edge `(fair − p) − rate·p·(1−p)·(1−rebate) − buffer`
//! is strictly **decreasing** in `p`: its derivative is `−1 − rate·(1−2p)·(1−rebate)`,
//! and `|rate·(1−2p)·(1−rebate)| ≤ rate < 1`, so the linear edge term always
//! dominates the bounded fee slope. The asks are ascending, so once a level
//! fails the gate every deeper level fails too — the walk stops at the first
//! failure. (The exact [`core_types::taker_fee`], with its 5-dp rounding and
//! $0.00001 floor, is used for the *reported* `expected_fee`; the per-share
//! marginal is the economically correct *decision* threshold, differing only by
//! that negligible floor/rounding.)
//!
//! ## Taker rebate
//!
//! `rebate_pct` discounts the fee in the *decision* (expected net cost), modeling
//! the tiered Taker Rebate Program — see [`MomentumTakerParams`](crate::MomentumTakerParams)
//! for why it defaults to zero at our volume. `expected_fee` is always the
//! **gross** fee the venue charges (the rebate, when any, is a separate daily
//! pUSD accrual, not an at-execution reduction), so journal expected-vs-realized
//! fee comparisons stay apples-to-apples.

use core_types::{BookLevel, Decimal, Dollars, Outcome, Price, Size, taker_fee};
use rust_decimal::prelude::FromPrimitive;

use super::NoTakeReason;

/// Venue minimum order notional ($1.00), mirroring `normalize`'s `MIN_NOTIONAL`.
/// A plan below this is rejected here so it never reaches — and is rejected by —
/// the normalizer.
const MIN_NOTIONAL: Decimal = Decimal::ONE;

/// A sized, fee-aware momentum take: a FAK marketable BUY of `notional` dollars
/// on `outcome`, capped at `worst_price`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TakePlan {
    /// Outcome token to buy (the side the confirmed move made cheap).
    pub outcome: Outcome,
    /// Worst-price slippage limit for the FAK = the highest accepted ask price.
    pub worst_price: Price,
    /// Dollar amount to spend (the FAK BUY `OrderQty::Notional`).
    pub notional: Dollars,
    /// Estimated shares the notional buys across the accepted levels.
    pub expected_shares: Size,
    /// Gross taker fee the venue will charge on the expected fill (matches
    /// [`core_types::taker_fee`] per level; *not* rebate-discounted).
    pub expected_fee: Dollars,
    /// Volume-weighted average executable price across the accepted levels (6 dp).
    pub vwap: Decimal,
    /// Expected gross profit: `Σ shares·(fair − price) − gross_fee` (the §8 edge,
    /// fee-adjusted; not buffer-adjusted — the buffer is a decision margin, not a
    /// realized cost).
    pub aggregate_edge: Dollars,
}

/// Plan a momentum take on `outcome`, walking `asks` (ascending, best/lowest
/// first — the [`BookSnapshot`](core_types::BookSnapshot) invariant).
///
/// `fair` is the model probability for `outcome` (so `p_up` for Up, `1 − p_up`
/// for Down). `fee_rate` is the per-market taker rate (the caller passes `0` when
/// the market has fees disabled). `rebate_pct` discounts the *decision* fee.
/// `buffer` is the §8 edge margin required beyond the fee. `remaining_budget`
/// caps the take's notional.
///
/// # Errors
/// A typed [`NoTakeReason`] (for §12 observability): `NoAsks`, `UnusableFair`,
/// `BookAlreadyRepriced` (best ask already ≥ fair — the move was absorbed),
/// `EdgeBelowFeePlusBuffer` (positive raw edge, but fee + buffer eats it), or
/// `BelowMinNotional` (sized below $1).
pub fn plan_take(
    outcome: Outcome,
    fair: f64,
    asks: &[BookLevel],
    fee_rate: Decimal,
    rebate_pct: Decimal,
    buffer: Decimal,
    remaining_budget: Dollars,
) -> Result<TakePlan, NoTakeReason> {
    let Some(fair_dec) = Decimal::from_f64(fair) else {
        return Err(NoTakeReason::UnusableFair { p_up: fair });
    };
    let Some(first) = asks.first() else {
        return Err(NoTakeReason::NoAsks);
    };
    if fair_dec <= first.price.as_decimal() {
        return Err(NoTakeReason::BookAlreadyRepriced {
            best_ask: first.price,
            fair: fair_dec,
        });
    }

    let one_minus_reb = Decimal::ONE - rebate_pct;
    let budget = remaining_budget.as_decimal();

    let mut taken = Decimal::ZERO;
    let mut notional = Decimal::ZERO;
    let mut value = Decimal::ZERO; // Σ shares · fair
    let mut gross_fee = Dollars::ZERO;
    let mut worst = first.price;
    // Best-level per-share diagnostics for the EdgeBelowFeePlusBuffer reject.
    let mut best_edge_ps = Decimal::ZERO;
    let mut best_fee_ps = Decimal::ZERO;

    for lvl in asks {
        let p = lvl.price.as_decimal();
        let edge_ps = fair_dec - p;
        if edge_ps <= Decimal::ZERO {
            break; // deeper asks only have less edge
        }
        let fee_ps = fee_rate * p * (Decimal::ONE - p);
        if taken.is_zero() {
            best_edge_ps = edge_ps;
            best_fee_ps = fee_ps * one_minus_reb;
        }
        // §8 gate: per-share net edge clears the (rebate-discounted) fee + buffer.
        if edge_ps <= fee_ps * one_minus_reb + buffer {
            break;
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
        value += s * fair_dec;
        gross_fee = gross_fee + taker_fee(s_size, fee_rate, lvl.price);
        worst = lvl.price;

        if s < depth {
            break; // budget bit this level — nothing left for deeper ones
        }
    }

    if taken.is_zero() {
        // We passed the BookAlreadyRepriced check, so the best ask had positive
        // raw edge; it just did not clear fee + buffer.
        return Err(NoTakeReason::EdgeBelowFeePlusBuffer {
            edge: best_edge_ps,
            fee: best_fee_ps,
            buffer,
        });
    }
    if notional < MIN_NOTIONAL {
        return Err(NoTakeReason::BelowMinNotional {
            got: Dollars::new(notional),
        });
    }
    let Ok(expected_shares) = Size::new(taken) else {
        return Err(NoTakeReason::BelowMinNotional {
            got: Dollars::new(notional),
        });
    };
    let vwap = (notional / taken).round_dp(6);
    let aggregate_edge = Dollars::new(value - notional) - gross_fee;

    Ok(TakePlan {
        outcome,
        worst_price: worst,
        notional: Dollars::new(notional),
        expected_shares,
        expected_fee: gross_fee,
        vwap,
        aggregate_edge,
    })
}

#[cfg(test)]
mod tests {
    use core_types::{Size, TickSize};
    use rust_decimal::dec;

    use super::*;

    const RATE: Decimal = Decimal::from_parts(7, 0, 0, false, 2); // 0.07
    const BUFFER: Decimal = Decimal::from_parts(5, 0, 0, false, 3); // 0.005

    fn ask(price: Decimal, size: Decimal) -> BookLevel {
        BookLevel {
            price: Price::on_grid(price, TickSize::T001).expect("grid price"),
            size: Size::new(size).expect("size"),
        }
    }

    fn plan(fair: f64, asks: &[BookLevel], budget: Decimal) -> Result<TakePlan, NoTakeReason> {
        plan_take(
            Outcome::Up,
            fair,
            asks,
            RATE,
            Decimal::ZERO,
            BUFFER,
            Dollars::new(budget),
        )
    }

    /// Takes when the edge clears fee + buffer (away from 0.5 the fee is small).
    #[test]
    fn takes_when_edge_clears_fee_plus_buffer() {
        // fair 0.85, ask 0.80×50. edge 0.05; fee 0.07·0.80·0.20 = 0.0112; +buffer
        // 0.005 = 0.0162 < 0.05 ⇒ take all 50.
        let asks = [ask(dec!(0.80), dec!(50))];
        let p = plan(0.85, &asks, dec!(100)).expect("a take");
        assert_eq!(p.outcome, Outcome::Up);
        assert_eq!(p.expected_shares, Size::new(dec!(50)).unwrap());
        assert_eq!(p.worst_price.as_decimal(), dec!(0.80));
        assert_eq!(p.notional, Dollars::new(dec!(40))); // 50·0.80
        // Gross fee = taker_fee(50, 0.07, 0.80) = 50·0.07·0.8·0.2 = 0.56.
        assert_eq!(p.expected_fee, Dollars::new(dec!(0.56)));
        // Gross edge = 50·(0.85−0.80) − 0.56 = 2.50 − 0.56 = 1.94.
        assert_eq!(p.aggregate_edge, Dollars::new(dec!(1.94)));
    }

    /// Refuses at the steepest-fee point (p≈0.5): the §8 boundary.
    #[test]
    fn refuses_when_fee_marginal_near_half() {
        // At p=0.50 the per-share fee is 0.07·0.25 = 0.0175; +buffer 0.005 ⇒ need
        // edge > 0.0225. fair 0.522 ⇒ edge 0.022 < 0.0225 ⇒ refuse.
        let asks = [ask(dec!(0.50), dec!(100))];
        let r = plan(0.522, &asks, dec!(100)).unwrap_err();
        assert!(matches!(r, NoTakeReason::EdgeBelowFeePlusBuffer { .. }));

        // fair 0.524 ⇒ edge 0.024 > 0.0225 ⇒ take.
        let p = plan(0.524, &asks, dec!(100)).expect("a take just over the line");
        assert_eq!(p.worst_price.as_decimal(), dec!(0.50));
    }

    /// Sizes against depth across multiple levels, taking each while the
    /// marginal net edge stays positive.
    #[test]
    fn sizes_against_depth_across_levels() {
        // fair 0.85. 0.80×50 (edge 0.05 ✓), 0.83×40 (edge 0.02; fee 0.07·0.83·0.17
        // = 0.009877; +0.005 = 0.014877 < 0.02 ✓). Both levels taken.
        let asks = [ask(dec!(0.80), dec!(50)), ask(dec!(0.83), dec!(40))];
        let p = plan(0.85, &asks, dec!(1000)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(90)).unwrap()); // 50 + 40
        assert_eq!(p.worst_price.as_decimal(), dec!(0.83));
        assert_eq!(p.notional, Dollars::new(dec!(73.2))); // 50·0.80 + 40·0.83
        // VWAP = 73.2 / 90 = 0.813333.
        assert_eq!(p.vwap, dec!(0.813333));
    }

    /// The third level fails the gate and is excluded even with budget to spare.
    #[test]
    fn stops_at_first_failing_level() {
        let asks = [
            ask(dec!(0.80), dec!(50)),
            ask(dec!(0.83), dec!(40)),
            ask(dec!(0.84), dec!(100)), // edge 0.01 < fee+buffer ⇒ excluded
        ];
        let p = plan(0.85, &asks, dec!(10_000)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(90)).unwrap());
        assert_eq!(p.worst_price.as_decimal(), dec!(0.83)); // 0.84 not taken
    }

    /// The remaining budget truncates the take mid-level.
    #[test]
    fn budget_caps_shares_mid_level() {
        // fair 0.85, ask 0.80×100, budget $20 ⇒ affordable 25 shares (< 100).
        let asks = [ask(dec!(0.80), dec!(100))];
        let p = plan(0.85, &asks, dec!(20)).expect("a take");
        assert_eq!(p.expected_shares, Size::new(dec!(25)).unwrap());
        assert_eq!(p.notional, Dollars::new(dec!(20)));
        assert!(p.notional.as_decimal() <= dec!(20));
    }

    /// Best ask already at/above fair: the move was already absorbed.
    #[test]
    fn book_already_repriced() {
        let asks = [ask(dec!(0.86), dec!(50))];
        let r = plan(0.85, &asks, dec!(100)).unwrap_err();
        assert!(matches!(r, NoTakeReason::BookAlreadyRepriced { .. }));
    }

    /// A take sized below the $1 venue minimum is rejected (pre-empting the
    /// normalizer).
    #[test]
    fn below_min_notional_rejected() {
        // ask 0.80×1 share ⇒ notional $0.80 < $1.
        let asks = [ask(dec!(0.80), dec!(1))];
        let r = plan(0.85, &asks, dec!(100)).unwrap_err();
        assert!(matches!(r, NoTakeReason::BelowMinNotional { .. }));
    }

    #[test]
    fn empty_asks_is_no_asks() {
        assert!(matches!(
            plan(0.85, &[], dec!(100)),
            Err(NoTakeReason::NoAsks)
        ));
    }

    /// The per-market fee rate (not a constant) governs the gate.
    #[test]
    fn fee_rate_is_per_market() {
        // fair 0.52, ask 0.50×100. edge 0.02.
        let asks = [ask(dec!(0.50), dec!(100))];
        // At rate 0.02: fee 0.02·0.25 = 0.005; +buffer 0.005 = 0.010 < 0.02 ⇒ take.
        let lo = plan_take(
            Outcome::Up,
            0.52,
            &asks,
            dec!(0.02),
            Decimal::ZERO,
            BUFFER,
            Dollars::new(dec!(100)),
        );
        assert!(lo.is_ok(), "low fee rate admits the take");
        // At rate 0.10: fee 0.10·0.25 = 0.025; +0.005 = 0.030 > 0.02 ⇒ refuse.
        let hi = plan_take(
            Outcome::Up,
            0.52,
            &asks,
            dec!(0.10),
            Decimal::ZERO,
            BUFFER,
            Dollars::new(dec!(100)),
        );
        assert!(matches!(
            hi,
            Err(NoTakeReason::EdgeBelowFeePlusBuffer { .. })
        ));
    }

    /// A taker rebate discounts the decision fee, admitting a marginal take —
    /// while the reported `expected_fee` stays gross.
    #[test]
    fn rebate_discounts_decision_fee_but_reports_gross() {
        // fair 0.522, ask 0.50×100. Without rebate: edge 0.022 < 0.0225 ⇒ refuse
        // (the marginal test above). With a 50% rebate the effective fee halves to
        // 0.00875; +0.005 = 0.01375 < 0.022 ⇒ take.
        let asks = [ask(dec!(0.50), dec!(100))];
        let p = plan_take(
            Outcome::Up,
            0.522,
            &asks,
            RATE,
            dec!(0.5),
            BUFFER,
            Dollars::new(dec!(100)),
        )
        .expect("rebate admits the take");
        // Gross fee unchanged: taker_fee(100, 0.07, 0.50) = 1.75.
        assert_eq!(p.expected_fee, Dollars::new(dec!(1.75)));
    }

    /// Disabled fees (rate 0) make every positive-edge level clear the gate.
    #[test]
    fn zero_fee_rate_takes_on_any_positive_edge_over_buffer() {
        let asks = [ask(dec!(0.50), dec!(100))];
        // edge 0.022 > buffer 0.005, fee 0 ⇒ take.
        let p = plan_take(
            Outcome::Up,
            0.522,
            &asks,
            Decimal::ZERO,
            Decimal::ZERO,
            BUFFER,
            Dollars::new(dec!(100)),
        )
        .expect("zero-fee take");
        assert_eq!(p.expected_fee, Dollars::ZERO);
    }
}
