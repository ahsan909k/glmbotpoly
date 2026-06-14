//! [`ConvergencePlanner`]: the pure, cancel-first diff between the calculator's
//! desired quote set and the manager's current [`RestingView`].
//!
//! Split-cycle (strict): a single [`ConvergencePlan`] never both cancels and
//! places the *same* slot. Cancels are issued first; a replacement for a
//! cancelled slot is placed on a *later* converge, once the cancel has
//! terminalized and the slot is empty. So `places` only ever targets slots that
//! are empty *now*, and `cancels` only ever targets orders that are stale,
//! unmatched, or being withdrawn.

use std::collections::HashMap;

use core_types::{ConditionId, Decimal, MarketInfo, OrderId, Outcome, Price, Size, TimeInForce};
use rust_decimal::prelude::FromPrimitive;

use super::view::RestingView;
use crate::quoting::{QuoteDecision, QuoteLevel, QuoteSet};

/// What to cancel this converge. Cancel-first: the driver awaits all of this
/// before placing anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelAction {
    /// Nothing to cancel.
    None,
    /// Cancel these specific resting orders (incremental reprice / withdrawal).
    Orders(Vec<OrderId>),
    /// Bulk-cancel the whole market (large adverse move, or a no-quote pull).
    Market(ConditionId),
    /// Cancel everything everywhere (Closing / risk veto — driver-level paths).
    All,
}

/// One desired placement the driver will normalize then batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesiredPlace {
    /// Outcome ladder.
    pub outcome: Outcome,
    /// Ladder level.
    pub level: u32,
    /// Outcome token id (resolved here from `MarketInfo::tokens`).
    pub token_id: core_types::TokenId,
    /// The calculator's on-grid price.
    pub price: Price,
    /// Order size in shares.
    pub size: Size,
    /// Time-in-force (post-only GTC).
    pub tif: TimeInForce,
}

/// Why the planner produced this step (diagnostics + drives the driver's
/// dirty-flag handling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergeRationale {
    /// The calculator declined to quote — cancel any resting orders, place none.
    NoQuotePull,
    /// A large adverse fair move — bulk cancel the whole market, replace next cycle.
    LargeMoveReprice,
    /// Per-level cancels for stale/unmatched levels and/or places into empty slots.
    IncrementalReprice,
    /// Desired matches resting — nothing to do.
    AlreadyConverged,
    /// Desired levels are pending replacement but their slots are still draining
    /// a cancel — nothing to do *yet* (keep waiting; do not treat as converged).
    WaitingForCancel,
}

/// The full convergence step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvergencePlan {
    /// Cancels to issue first.
    pub cancels: CancelAction,
    /// Placements to make after the cancels (only ever into empty slots).
    pub places: Vec<DesiredPlace>,
    /// Why.
    pub rationale: ConvergeRationale,
}

/// The pure cancel-first planner.
pub struct ConvergencePlanner;

impl ConvergencePlanner {
    /// Computes the cancel-first convergence step.
    ///
    /// - `decision` — fresh [`crate::calculate_quotes`] output.
    /// - `view` — the engine-side resting view for the active window.
    /// - `market` — active market (Outcome→TokenId, condition id, tick).
    /// - `p_up_now` — the model fair the `decision` was built from.
    /// - `p_up_at_last_quote` — the fair our currently-resting set was built from
    ///   (`None` if nothing resting / first quote).
    /// - `theta` — the per-level repricing hysteresis (probability space).
    /// - `cancel_market_theta` — a fair move beyond this bulk-cancels the market.
    #[must_use]
    pub fn plan(
        decision: &QuoteDecision,
        view: &RestingView,
        market: &MarketInfo,
        p_up_now: f64,
        p_up_at_last_quote: Option<f64>,
        theta: f64,
        cancel_market_theta: f64,
    ) -> ConvergencePlan {
        match decision {
            // P0: no quote → withdraw any resting orders, place nothing.
            QuoteDecision::NoQuote(_) => {
                if view.has_live_cancelable() {
                    ConvergencePlan {
                        cancels: CancelAction::Market(market.condition_id.clone()),
                        places: Vec::new(),
                        rationale: ConvergeRationale::NoQuotePull,
                    }
                } else {
                    ConvergencePlan {
                        cancels: CancelAction::None,
                        places: Vec::new(),
                        rationale: ConvergeRationale::AlreadyConverged,
                    }
                }
            }
            QuoteDecision::Quote(qs) => Self::plan_quote(
                qs,
                view,
                market,
                p_up_now,
                p_up_at_last_quote,
                theta,
                cancel_market_theta,
            ),
        }
    }

    fn plan_quote(
        qs: &QuoteSet,
        view: &RestingView,
        market: &MarketInfo,
        p_up_now: f64,
        p_up_at_last_quote: Option<f64>,
        theta: f64,
        cancel_market_theta: f64,
    ) -> ConvergencePlan {
        // P1: a large adverse move moves every level — bulk-cancel the whole
        // market and replace the full ladder on the next converge.
        if let Some(p0) = p_up_at_last_quote
            && view.has_live_cancelable()
            && (p_up_now - p0).abs() > cancel_market_theta
        {
            return ConvergencePlan {
                cancels: CancelAction::Market(market.condition_id.clone()),
                places: Vec::new(),
                rationale: ConvergeRationale::LargeMoveReprice,
            };
        }

        // P2: incremental diff.
        let theta_dec = Decimal::from_f64(theta).unwrap_or(Decimal::ZERO);

        // Index desired levels by slot.
        let mut desired: HashMap<(Outcome, u32), &QuoteLevel> = HashMap::new();
        for ql in &qs.levels {
            desired.insert((ql.outcome, ql.level), ql);
        }

        // Pass A — places into currently-empty slots, in the calculator's order
        // (Up ladder then Down ladder, ascending level). A slot occupied by a
        // pending-cancel order is "blocked": we owe its replacement but cannot
        // place until it frees.
        let mut places = Vec::new();
        let mut blocked = false;
        for ql in &qs.levels {
            match view.resting_at(ql.outcome, ql.level) {
                None => places.push(desired_place(ql, market)),
                Some(order) if order.is_pending_cancel() => blocked = true,
                Some(order) => {
                    let stale =
                        (ql.price.as_decimal() - order.price.as_decimal()).abs() > theta_dec;
                    if stale {
                        // Cancelled in pass B; its replacement is owed next cycle.
                        blocked = true;
                    }
                    // else: keep (within-θ hysteresis).
                }
            }
        }

        // Pass B — cancels: stale (price moved beyond θ) or unmatched (the desired
        // set no longer wants this slot). Skip orders already cancelling.
        // Deterministic order (sort by outcome, level, seq) since `live_orders`
        // iterates a HashMap.
        let mut to_cancel: Vec<(Outcome, u32, u64, OrderId)> = Vec::new();
        for order in view.live_orders() {
            if order.is_pending_cancel() {
                continue;
            }
            let cancel = match desired.get(&(order.outcome, order.level)) {
                None => true, // unmatched → withdraw
                Some(ql) => (ql.price.as_decimal() - order.price.as_decimal()).abs() > theta_dec,
            };
            if cancel {
                to_cancel.push((
                    order.outcome,
                    order.level,
                    order.client_id.seq,
                    order.order_id.clone(),
                ));
            }
        }
        to_cancel.sort();
        let cancel_ids: Vec<OrderId> = to_cancel.into_iter().map(|(.., id)| id).collect();

        let cancels_empty = cancel_ids.is_empty();
        let cancels = if cancels_empty {
            CancelAction::None
        } else {
            CancelAction::Orders(cancel_ids)
        };
        let rationale = if !places.is_empty() || !cancels_empty {
            ConvergeRationale::IncrementalReprice
        } else if blocked {
            ConvergeRationale::WaitingForCancel
        } else {
            ConvergeRationale::AlreadyConverged
        };
        ConvergencePlan {
            cancels,
            places,
            rationale,
        }
    }
}

/// Resolves a desired ladder level into a [`DesiredPlace`] (mapping
/// `Outcome`→`TokenId` via the market's token pair).
fn desired_place(ql: &QuoteLevel, market: &MarketInfo) -> DesiredPlace {
    DesiredPlace {
        outcome: ql.outcome,
        level: ql.level,
        token_id: market.tokens.get(ql.outcome).clone(),
        price: ql.price,
        size: ql.size,
        tif: ql.tif,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quote_manager::client_id::ClientId;
    use crate::quoting::NoQuoteReason;
    use core_types::{
        Asset, ConditionId, FeeParams, ModelHealth, ModelHealthReason, OrderState, OrderUpdate,
        ResolutionSource, RoundDir, Series, Side, TickSize, TimeInForce, TimestampMs, TokenId,
        TokenPair, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    const OPEN_MS: i64 = 1_781_000_000_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }
    fn condition() -> ConditionId {
        ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap()
    }
    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn market() -> MarketInfo {
        MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-plan".to_owned(),
            condition_id: condition(),
            tokens: TokenPair {
                up: TokenId::new("111").unwrap(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: TimestampMs::from_millis(OPEN_MS + 300_000),
            strike: Some(dec!(60000)),
            tick_size: TickSize::T001,
            min_order_size: sz(dec!(5)),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        }
    }

    /// A desired two-sided ladder: Up @ prices[..], Down @ prices[..] (level 0..).
    fn quote_set(up: &[Decimal], down: &[Decimal]) -> QuoteDecision {
        let mut levels = Vec::new();
        for (i, p) in up.iter().enumerate() {
            levels.push(QuoteLevel::resting_buy(
                Outcome::Up,
                px(*p),
                sz(dec!(10)),
                i as u32,
            ));
        }
        for (i, p) in down.iter().enumerate() {
            levels.push(QuoteLevel::resting_buy(
                Outcome::Down,
                px(*p),
                sz(dec!(10)),
                i as u32,
            ));
        }
        QuoteDecision::Quote(QuoteSet {
            window: window(),
            levels,
            center: dec!(0.5),
            half_spread: dec!(0.01),
            p_up: 0.5,
            normalized_imbalance: 0.0,
            vol_spike: false,
            suppressed: Vec::new(),
        })
    }

    fn no_quote() -> QuoteDecision {
        QuoteDecision::NoQuote(NoQuoteReason::FinalSecondsNoPassive { tau_secs: 3.0 })
    }

    /// A view holding a resting Open order on `outcome`/`level` at `price`.
    fn rest(
        view: &mut RestingView,
        outcome: Outcome,
        level: u32,
        seq: u64,
        price: Decimal,
        id: &str,
    ) {
        view.record_pending(
            OrderId::new(id).unwrap(),
            ClientId {
                open_ms: OPEN_MS,
                outcome,
                level,
                seq,
            },
            window(),
            px(price),
            sz(dec!(10)),
        );
        // advance PendingNew -> Open
        view.apply_order_update(&OrderUpdate {
            order_id: OrderId::new(id).unwrap(),
            window: window(),
            token_id: market().tokens.get(outcome).clone(),
            side: Side::Buy,
            state: OrderState::Open,
            price: px(price),
            original_size: sz(dec!(10)),
            filled_size: Size::ZERO,
            reject_reason: None,
            ts_venue: None,
            ts_local: TimestampMs::from_millis(OPEN_MS),
        });
    }

    fn model() -> (ModelHealth, ModelHealthReason) {
        (ModelHealth::Ready, ModelHealthReason::Nominal)
    }

    #[test]
    fn first_quote_into_empty_view_places_all_no_cancels() {
        let _ = model();
        let view = RestingView::new();
        let decision = quote_set(&[dec!(0.49), dec!(0.48)], &[dec!(0.49), dec!(0.48)]);
        let plan = ConvergencePlanner::plan(&decision, &view, &market(), 0.5, None, 0.005, 0.02);
        assert_eq!(plan.cancels, CancelAction::None);
        assert_eq!(plan.places.len(), 4);
        assert_eq!(plan.rationale, ConvergeRationale::IncrementalReprice);
    }

    #[test]
    fn converged_view_yields_no_action() {
        let mut view = RestingView::new();
        rest(&mut view, Outcome::Up, 0, 0, dec!(0.49), "u0");
        rest(&mut view, Outcome::Down, 0, 1, dec!(0.49), "d0");
        let decision = quote_set(&[dec!(0.49)], &[dec!(0.49)]);
        let plan =
            ConvergencePlanner::plan(&decision, &view, &market(), 0.5, Some(0.5), 0.005, 0.02);
        assert_eq!(plan.cancels, CancelAction::None);
        assert!(plan.places.is_empty());
        assert_eq!(plan.rationale, ConvergeRationale::AlreadyConverged);
    }

    #[test]
    fn within_theta_keeps_but_beyond_theta_cancels_without_same_cycle_place() {
        // theta = 0.005. Resting Up@0.49; desired Up@0.49 (kept) and a beyond-θ
        // Down: resting Down@0.49, desired Down@0.47 (Δ=0.02 > θ).
        let mut view = RestingView::new();
        rest(&mut view, Outcome::Up, 0, 0, dec!(0.49), "u0");
        rest(&mut view, Outcome::Down, 0, 1, dec!(0.49), "d0");
        let decision = quote_set(&[dec!(0.49)], &[dec!(0.47)]);
        let plan =
            ConvergencePlanner::plan(&decision, &view, &market(), 0.5, Some(0.5), 0.005, 0.02);
        // The stale Down is cancelled; NO Down place this cycle (slot still occupied).
        assert_eq!(
            plan.cancels,
            CancelAction::Orders(vec![OrderId::new("d0").unwrap()])
        );
        assert!(plan.places.is_empty(), "split-cycle: no same-slot replace");
        assert_eq!(plan.rationale, ConvergeRationale::IncrementalReprice);
    }

    #[test]
    fn pending_cancel_slot_is_skipped_and_waits() {
        let mut view = RestingView::new();
        rest(&mut view, Outcome::Up, 0, 0, dec!(0.49), "u0");
        view.mark_pending_cancel(&OrderId::new("u0").unwrap());
        let decision = quote_set(&[dec!(0.47)], &[]);
        let plan =
            ConvergencePlanner::plan(&decision, &view, &market(), 0.5, Some(0.5), 0.005, 0.02);
        assert_eq!(
            plan.cancels,
            CancelAction::None,
            "already cancelling — no re-cancel"
        );
        assert!(
            plan.places.is_empty(),
            "slot still occupied by the draining cancel"
        );
        assert_eq!(plan.rationale, ConvergeRationale::WaitingForCancel);
    }

    #[test]
    fn unmatched_resting_is_withdrawn() {
        // Resting Up has 2 levels; desired only wants level 0 → level 1 withdrawn.
        let mut view = RestingView::new();
        rest(&mut view, Outcome::Up, 0, 0, dec!(0.49), "u0");
        rest(&mut view, Outcome::Up, 1, 1, dec!(0.48), "u1");
        let decision = quote_set(&[dec!(0.49)], &[]);
        let plan =
            ConvergencePlanner::plan(&decision, &view, &market(), 0.5, Some(0.5), 0.005, 0.02);
        assert_eq!(
            plan.cancels,
            CancelAction::Orders(vec![OrderId::new("u1").unwrap()])
        );
        assert!(plan.places.is_empty());
    }

    #[test]
    fn no_quote_pulls_the_whole_market() {
        let mut view = RestingView::new();
        rest(&mut view, Outcome::Up, 0, 0, dec!(0.49), "u0");
        let plan =
            ConvergencePlanner::plan(&no_quote(), &view, &market(), 0.5, Some(0.5), 0.005, 0.02);
        assert_eq!(plan.cancels, CancelAction::Market(condition()));
        assert!(plan.places.is_empty());
        assert_eq!(plan.rationale, ConvergeRationale::NoQuotePull);
    }

    #[test]
    fn no_quote_on_empty_view_is_converged() {
        let view = RestingView::new();
        let plan = ConvergencePlanner::plan(&no_quote(), &view, &market(), 0.5, None, 0.005, 0.02);
        assert_eq!(plan.cancels, CancelAction::None);
        assert_eq!(plan.rationale, ConvergeRationale::AlreadyConverged);
    }

    #[test]
    fn large_move_bulk_cancels_the_market() {
        let mut view = RestingView::new();
        rest(&mut view, Outcome::Up, 0, 0, dec!(0.49), "u0");
        // p_up jumped 0.50 -> 0.85: |Δ| = 0.35 > cancel_market_theta 0.02.
        let decision = quote_set(&[dec!(0.83)], &[dec!(0.13)]);
        let plan =
            ConvergencePlanner::plan(&decision, &view, &market(), 0.85, Some(0.5), 0.005, 0.02);
        assert_eq!(plan.cancels, CancelAction::Market(condition()));
        assert!(plan.places.is_empty(), "replace the full ladder next cycle");
        assert_eq!(plan.rationale, ConvergeRationale::LargeMoveReprice);
    }

    #[test]
    fn empty_view_after_large_move_places_full_fresh_ladder() {
        // The cycle after the bulk cancel: view empty, fair at 0.85 → place fresh.
        let view = RestingView::new();
        let decision = quote_set(&[dec!(0.83), dec!(0.82)], &[dec!(0.13), dec!(0.12)]);
        let plan =
            ConvergencePlanner::plan(&decision, &view, &market(), 0.85, Some(0.5), 0.005, 0.02);
        // P1 guard requires a cancelable view; empty view skips it → P2 places all.
        assert_eq!(plan.cancels, CancelAction::None);
        assert_eq!(plan.places.len(), 4);
        assert_eq!(plan.rationale, ConvergeRationale::IncrementalReprice);
    }

    #[test]
    fn desired_place_maps_outcome_to_token() {
        let view = RestingView::new();
        let decision = quote_set(&[dec!(0.49)], &[dec!(0.49)]);
        let plan = ConvergencePlanner::plan(&decision, &view, &market(), 0.5, None, 0.005, 0.02);
        let up = plan
            .places
            .iter()
            .find(|d| d.outcome == Outcome::Up)
            .unwrap();
        let down = plan
            .places
            .iter()
            .find(|d| d.outcome == Outcome::Down)
            .unwrap();
        assert_eq!(up.token_id, TokenId::new("111").unwrap());
        assert_eq!(down.token_id, TokenId::new("222").unwrap());
        assert_eq!(up.tif, TimeInForce::Gtc { post_only: true });
    }
}
