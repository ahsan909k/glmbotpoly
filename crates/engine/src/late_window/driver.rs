//! [`LateWindowTaker`]: the async, IO-bearing executor for the §8 late-window
//! certainty taker.
//!
//! Mirrors [`MomentumTaker`](crate::MomentumTaker): it needs **no** `tokio` (it
//! only `.await`s the [`VenuePort`] futures and takes `now: TimestampMs` as a
//! parameter — sans-clock, like the rest of `engine`), folds its own fills **only**
//! from the venue stream, owns the `tripped`/`standing_down` risk veto, and routes
//! every FAK through the [`normalize`](crate::normalize) chokepoint. It logs via
//! `tracing` (target `late-window-taker`, §12).
//!
//! The decision is purely model-probability driven: there is **no signal ring and
//! no fast feed**. [`ingest`](LateWindowTaker::ingest) deliberately drops
//! [`Event::PriceTick`](core_types::Event::PriceTick) — the candidate side and its
//! certainty come from the Chainlink-anchored [`ModelSnapshot`](core_types::ModelSnapshot)
//! alone (see [`super`] for the anchor gate and the tie rule).
//!
//! ## Budget accounting
//!
//! Identical committed-in-flight scheme to the momentum taker: `effective_spent =
//! realized_spent + Σ pending`; a `place` `Accepted` commits the planned notional
//! to `pending`; each of our `Fill`s moves notional into `realized_spent` and
//! decrements that order's `pending`; the order's terminal `Order` update drops any
//! residual (a FAK's unfilled remainder was killed, never spent). The budget is
//! driver-local — the shared momentum+late reconciliation is deferred (see
//! [`super`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use core_types::{
    AnchorSource, BookSnapshot, BreakerKind, Decimal, Dollars, Event, Liquidity, MarketInfo,
    ModelSnapshot, OrderQty, Outcome, RiskEvent, Side, TimeInForce, TimestampMs, TokenId, WindowId,
    WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::edge::{CertaintyTakePlan, plan_certainty_take};
use super::{LateWindowTakerParams, NoLateTakeReason};
use crate::normalize::{NormalizerParams, OrderDraft, normalize};
use crate::quote_manager::RestingView;

/// Seconds remaining to close, clamped at zero — computed inline so the driver
/// needs no `timeutil` dependency (the `quote_manager`/`taker` precedent).
fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.as_millis() - now.as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

/// The active window the taker is currently trading.
struct Active {
    window: WindowId,
    market: Arc<MarketInfo>,
}

/// A fully-vetted take decision: the FAK to build.
struct Decision {
    outcome: Outcome,
    token_id: TokenId,
    market: Arc<MarketInfo>,
    plan: CertaintyTakePlan,
}

/// The late-window certainty taker (CLAUDE.md §8). Generic placement happens
/// through the [`VenuePort`] passed to each method; the taker stores no port (and
/// so is not itself generic), keeping it trivially shareable across the bot's
/// venue.
pub struct LateWindowTaker {
    params: LateWindowTakerParams,
    normalizer_params: NormalizerParams,
    /// Latest model snapshot.
    last_model: Option<ModelSnapshot>,
    /// Latest full book per active-window token (depth for sizing).
    books: HashMap<TokenId, Arc<BookSnapshot>>,
    /// The active window, if one is open.
    active: Option<Active>,
    /// True while a risk breaker holds the taker down.
    standing_down: bool,
    /// Currently-tripped breakers (stand down until empty).
    tripped: HashSet<BreakerKind>,
    /// Realized taker spend this window (from our own fills).
    realized_spent: Dollars,
    /// Committed-but-not-yet-realized notional per in-flight take order.
    pending: HashMap<core_types::OrderId, Dollars>,
    /// Order ids of takes we placed this window (fill attribution).
    our_orders: HashSet<core_types::OrderId>,
    /// Wall time (ms) of the last fired take — the cooldown anchor.
    last_take_ms: Option<i64>,
    /// Monotonic per-process placement sequence (client-id uniqueness).
    seq: u64,
    /// Count of takes fired so far (test/diagnostic metric).
    take_count: u64,
}

impl LateWindowTaker {
    /// Builds a taker with the given tunables and an empty state.
    #[must_use]
    pub fn new(params: LateWindowTakerParams, normalizer_params: NormalizerParams) -> Self {
        Self {
            params,
            normalizer_params,
            last_model: None,
            books: HashMap::new(),
            active: None,
            standing_down: false,
            tripped: HashSet::new(),
            realized_spent: Dollars::ZERO,
            pending: HashMap::new(),
            our_orders: HashSet::new(),
            last_take_ms: None,
            seq: 0,
            take_count: 0,
        }
    }

    /// Number of takes fired so far (each is one accepted FAK placement).
    #[must_use]
    pub fn take_count(&self) -> u64 {
        self.take_count
    }

    /// Realized taker spend this window (from our own fills).
    #[must_use]
    pub fn realized_spent(&self) -> Dollars {
        self.realized_spent
    }

    /// Budget committed so far this window: realized fills plus in-flight notional.
    #[must_use]
    pub fn effective_spent(&self) -> Dollars {
        self.pending
            .values()
            .copied()
            .fold(self.realized_spent, |acc, p| acc + p)
    }

    /// Whether a risk breaker currently holds the taker down.
    #[must_use]
    pub fn is_standing_down(&self) -> bool {
        self.standing_down
    }

    /// Whether `id` is one of this taker's placed orders (driver attribution).
    #[must_use]
    pub fn owns(&self, id: &core_types::OrderId) -> bool {
        self.our_orders.contains(id)
    }

    /// Handles one bus event; attempts a take after a model or active-token book
    /// update (a fresh fair or a fresh book may have opened the endgame). A price
    /// tick is ignored — this taker uses no fast-feed signal.
    pub async fn on_event<P: VenuePort>(
        &mut self,
        event: &Event,
        port: &P,
        now: TimestampMs,
        resting: Option<&RestingView>,
    ) {
        if self.ingest(event) {
            self.attempt_take(port, now, resting).await;
        }
    }

    /// Folds one item from the venue's order/fill stream: our own fills charge the
    /// budget; a terminal update on one of our orders drops its in-flight residual.
    pub fn on_venue_event(&mut self, ve: &VenueEvent, _now: TimestampMs) {
        match ve {
            VenueEvent::Fill(f) => {
                if self.our_orders.contains(&f.order_id) {
                    debug_assert_eq!(
                        f.liquidity,
                        Liquidity::Taker,
                        "the late-window taker only ever fires FAK taker orders"
                    );
                    let filled = Dollars::new(f.price.as_decimal() * f.size.as_decimal());
                    self.realized_spent = self.realized_spent + filled;
                    if let Some(rem) = self.pending.get_mut(&f.order_id) {
                        let after = *rem - filled;
                        *rem = if after.is_negative() {
                            Dollars::ZERO
                        } else {
                            after
                        };
                    }
                    tracing::info!(target: "late-window-taker", order = %f.order_id, price = %f.price, size = %f.size, "taker fill");
                }
            }
            VenueEvent::Order(u) => {
                if u.state.is_terminal() && self.our_orders.contains(&u.order_id) {
                    // FAK remainder killed — the unfilled part was never spent.
                    self.pending.remove(&u.order_id);
                }
            }
            // User-WS connectivity is a risk-manager concern; the taker ignores it.
            VenueEvent::Connectivity { .. } => {}
        }
    }

    // ---- ingestion (sync) --------------------------------------------------

    /// Applies one bus event to the taker's state. Returns `true` when a take
    /// should be attempted (a model update, or a fresh book on an active token).
    fn ingest(&mut self, event: &Event) -> bool {
        match event {
            // No fast feed: the late-window decision is Chainlink-anchored only,
            // so price ticks never feed it.
            Event::PriceTick(_) => false,
            Event::Model(snap) => {
                self.last_model = Some(*snap);
                true
            }
            Event::Book(snap) => self.cache_active_book(snap),
            Event::Window { market, lifecycle } => {
                self.on_window(market, *lifecycle);
                false
            }
            Event::Risk(risk) => {
                self.on_risk(*risk);
                false
            }
            _ => false,
        }
    }

    /// Caches a book iff it belongs to the active window's tokens; returns whether
    /// it did (and therefore whether a take should be re-evaluated).
    fn cache_active_book(&mut self, snap: &Arc<BookSnapshot>) -> bool {
        let is_active = self
            .active
            .as_ref()
            .is_some_and(|a| a.market.tokens.outcome_of(&snap.token_id).is_some());
        if is_active {
            self.books.insert(snap.token_id.clone(), Arc::clone(snap));
        }
        is_active
    }

    fn on_window(&mut self, market: &Arc<MarketInfo>, lifecycle: WindowLifecycle) {
        match lifecycle {
            WindowLifecycle::Open => {
                self.active = Some(Active {
                    window: market.window,
                    market: Arc::clone(market),
                });
                // Fresh per-window budget/cooldown/attribution; books re-arrive.
                self.realized_spent = Dollars::ZERO;
                self.pending.clear();
                self.our_orders.clear();
                self.books.clear();
                self.last_take_ms = None;
                tracing::info!(target: "late-window-taker", window = %market.window, "window open: late taker armed");
            }
            WindowLifecycle::Closing
            | WindowLifecycle::Closed
            | WindowLifecycle::Resolved { .. } => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| a.window == market.window)
                {
                    self.active = None;
                    tracing::debug!(target: "late-window-taker", window = %market.window, ?lifecycle, "window ended: late taker stood down");
                }
            }
            WindowLifecycle::Discovered => {}
        }
    }

    fn on_risk(&mut self, risk: RiskEvent) {
        match risk {
            RiskEvent::BreakerTripped { breaker }
            | RiskEvent::CancelAllIssued { reason: breaker } => {
                self.tripped.insert(breaker);
                self.standing_down = true;
                // No cancel-all: a FAK is immediately terminal, so the taker
                // never holds resting orders — it just stops firing.
                tracing::warn!(target: "late-window-taker", ?breaker, "risk veto: late taker standing down");
            }
            RiskEvent::BreakerCleared { breaker } => {
                self.tripped.remove(&breaker);
                if self.tripped.is_empty() {
                    self.standing_down = false;
                    tracing::info!(target: "late-window-taker", "all breakers cleared: late taker resuming");
                }
            }
        }
    }

    // ---- decision (sync, pure) ---------------------------------------------

    /// The full take gate ladder. Each gate maps to a typed [`NoLateTakeReason`] so
    /// every non-fire is explainable. Pure and synchronous.
    fn decide(
        &self,
        now: TimestampMs,
        resting: Option<&RestingView>,
    ) -> Result<Decision, NoLateTakeReason> {
        if self.standing_down {
            return Err(NoLateTakeReason::StandingDown);
        }
        let active = self
            .active
            .as_ref()
            .ok_or(NoLateTakeReason::NoActiveWindow)?;
        let model = self.last_model.ok_or(NoLateTakeReason::NoModel)?;
        if model.window != Some(active.window) {
            return Err(NoLateTakeReason::WindowMismatch {
                model: model.window,
                active: active.window,
            });
        }
        if !model.health.allows_quoting() {
            return Err(NoLateTakeReason::ModelNotReady {
                health: model.health,
                reason: model.reason,
            });
        }
        // Resolution-grade feed only: never act on a fast-feed-anchored fair (§8).
        if model.anchor != AnchorSource::Chainlink {
            return Err(NoLateTakeReason::NotChainlinkAnchored {
                anchor: model.anchor,
            });
        }
        // The certainty taker WANTS a saturated fair (p_up at/near 1.0 or 0.0), so
        // it accepts the closed interval [0, 1] — unlike the momentum taker's
        // strict (0, 1) interior. Only a non-finite or out-of-range fair is unusable.
        let p_up = model.p_up;
        if !p_up.is_finite() || !(0.0..=1.0).contains(&p_up) {
            return Err(NoLateTakeReason::UnusableFair { p_up });
        }
        // Defensive expiry guard, then the late-window activation.
        let tau = tau_secs(now, active.market.close_time);
        if tau <= 0.0 {
            return Err(NoLateTakeReason::Expired { tau_secs: tau });
        }
        if tau > f64::from(self.params.tau_threshold_secs) {
            return Err(NoLateTakeReason::NotLateWindow {
                tau_secs: tau,
                threshold: self.params.tau_threshold_secs,
            });
        }
        // Tie rule (§6): the model encodes S>=K (incl. equality) as p_up saturating
        // toward 1.0, so inclusive comparisons make exact-at-strike lean Up — a
        // p_up of exactly 1.0 buys Up, 0.0 buys Down.
        let outcome = if p_up >= self.params.certainty_threshold {
            Outcome::Up
        } else if p_up <= 1.0 - self.params.certainty_threshold {
            Outcome::Down
        } else {
            return Err(NoLateTakeReason::NotCertain {
                p_up,
                threshold: self.params.certainty_threshold,
            });
        };
        // Cooldown.
        if let Some(last) = self.last_take_ms {
            let elapsed = now.as_millis() - last;
            if elapsed < self.params.cooldown_ms {
                return Err(NoLateTakeReason::InCooldown {
                    remaining_ms: self.params.cooldown_ms - elapsed,
                });
            }
        }
        // Budget (committed-in-flight aware).
        let remaining = self.params.budget_per_window - self.effective_spent();
        if remaining.as_decimal() < Decimal::ONE {
            return Err(NoLateTakeReason::BudgetExhausted {
                remaining,
                need: Dollars::new(Decimal::ONE),
            });
        }
        let token_id = active.market.tokens.get(outcome).clone();
        let book = self
            .books
            .get(&token_id)
            .ok_or(NoLateTakeReason::NoBookForToken)?;
        let fees = &active.market.fees;
        let fee_rate = if fees.enabled {
            fees.rate
        } else {
            Decimal::ZERO
        };
        // §7 self-match filter: never lift our own mirrored resting liquidity.
        let asks = crate::self_match::filter_asks(outcome, &book.asks, active.window, resting);
        let plan = plan_certainty_take(outcome, &asks, fee_rate, self.params.price_cap, remaining)?;
        Ok(Decision {
            outcome,
            token_id,
            market: Arc::clone(&active.market),
            plan,
        })
    }

    // ---- firing (async) ----------------------------------------------------

    async fn attempt_take<P: VenuePort>(
        &mut self,
        port: &P,
        now: TimestampMs,
        resting: Option<&RestingView>,
    ) {
        match self.decide(now, resting) {
            Ok(decision) => self.fire(port, decision, now).await,
            Err(reason) => {
                tracing::debug!(target: "late-window-taker", reason = %reason, "no take");
            }
        }
    }

    async fn fire<P: VenuePort>(&mut self, port: &P, decision: Decision, now: TimestampMs) {
        let seq = self.next_seq();
        let open_ms = decision.market.window.open_time.as_millis();
        let draft = OrderDraft {
            client_id: Some(format!("lw:{open_ms}:{seq}")),
            window: decision.market.window,
            token_id: decision.token_id,
            outcome: decision.outcome,
            side: Side::Buy,
            price: decision.plan.worst_price.as_decimal(),
            qty: OrderQty::Notional(decision.plan.notional),
            tif: TimeInForce::Fak,
        };
        // FAK is not post-only, so the normalizer's cross check is skipped — no
        // book view needed. It only snaps the worst-price cap (already on-grid)
        // and re-checks the $1 notional (guaranteed by plan_certainty_take).
        let order = match normalize(&draft, &decision.market, None, &self.normalizer_params) {
            Ok(n) => n.order,
            Err(e) => {
                tracing::warn!(target: "late-window-taker", reason = %e, "normalize rejected the FAK (unexpected — plan_certainty_take guarantees ≥$1 notional + on-grid worst price)");
                return;
            }
        };
        match port.place(&order).await {
            Ok(acc) => {
                self.our_orders.insert(acc.order_id.clone());
                self.pending.insert(acc.order_id, decision.plan.notional);
                self.last_take_ms = Some(now.as_millis());
                self.take_count += 1;
                tracing::info!(
                    target: "late-window-taker",
                    outcome = %decision.outcome,
                    worst = %decision.plan.worst_price,
                    notional = %decision.plan.notional,
                    shares = %decision.plan.expected_shares,
                    "late-window certainty take (FAK)"
                );
            }
            Err(e) => {
                tracing::warn!(target: "late-window-taker", error = %e, "FAK place failed");
            }
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq = self.seq.wrapping_add(1);
        s
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use core_types::{
        Asset, BookLevel, BookSnapshot, ConditionId, DurationMs, FeeParams, Fill, InputAges,
        Liquidity, ModelHealth, ModelHealthReason, ModelSnapshot, OrderId, OrderState, OrderUpdate,
        Outcome, Price, ResolutionSource, Series, Side, Size, TickSize, TokenId, TokenPair,
        WindowDuration, WindowId, WindowLifecycle,
    };
    use rust_decimal::dec;
    use venue_api::VenueEvent;

    use super::*;

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = OPEN_MS + 300_000;
    /// A "now" with τ = 20 s remaining — inside the 30 s late-window threshold.
    const LATE_MS: i64 = CLOSE_MS - 20_000;
    const TICK: TickSize = TickSize::T001;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn other_window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Eth,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn up_token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn down_token() -> TokenId {
        TokenId::new("222").unwrap()
    }

    fn market() -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-lw".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: up_token(),
                down: down_token(),
            },
            close_time: TimestampMs::from_millis(CLOSE_MS),
            strike: Some(dec!(60000)),
            tick_size: TICK,
            min_order_size: Size::new(dec!(5)).unwrap(),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        })
    }

    fn open_event() -> Event {
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Open,
        }
    }

    fn model_snapshot(
        p_up: f64,
        health: ModelHealth,
        anchor: AnchorSource,
        win: Option<WindowId>,
    ) -> Event {
        Event::Model(ModelSnapshot {
            asset: Asset::Btc,
            window: win,
            p_up,
            z: 0.0,
            sigma_1s: 1e-4,
            sigma_tau: 1e-4,
            basis: 0.0,
            anchor,
            health,
            reason: if matches!(health, ModelHealth::Ready) {
                ModelHealthReason::Nominal
            } else {
                ModelHealthReason::NoAnchor
            },
            input_ages: InputAges {
                chainlink: DurationMs::from_millis(0),
                binance: DurationMs::from_millis(0),
            },
            ts: TimestampMs::from_millis(OPEN_MS),
        })
    }

    /// A Ready, Chainlink-anchored BTC model on `window()`.
    fn ready_chainlink(p_up: f64) -> Event {
        model_snapshot(
            p_up,
            ModelHealth::Ready,
            AnchorSource::Chainlink,
            Some(window()),
        )
    }

    fn book(token: TokenId, asks: &[(Decimal, Decimal)]) -> Event {
        Event::Book(Arc::new(BookSnapshot {
            token_id: token,
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            bids: Vec::new(),
            asks: asks
                .iter()
                .map(|(p, s)| BookLevel {
                    price: Price::on_grid(*p, TICK).unwrap(),
                    size: Size::new(*s).unwrap(),
                })
                .collect(),
            ts: TimestampMs::from_millis(OPEN_MS),
            seq_hash: None,
        }))
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn taker() -> LateWindowTaker {
        LateWindowTaker::new(
            LateWindowTakerParams::default(),
            NormalizerParams::default(),
        )
    }

    /// A fully-armed taker on `window()`: open, a Ready+Chainlink model with the
    /// given `p_up` and anchor, and a cheap (within-cap) book on `token`.
    fn armed_with(
        p_up: f64,
        anchor: AnchorSource,
        token: TokenId,
        asks: &[(Decimal, Decimal)],
    ) -> LateWindowTaker {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&model_snapshot(
            p_up,
            ModelHealth::Ready,
            anchor,
            Some(window()),
        ));
        t.ingest(&book(token, asks));
        t
    }

    /// The canonical armed Up taker: p_up 0.99, Chainlink, Up book at 0.98.
    fn armed_up() -> LateWindowTaker {
        armed_with(
            0.99,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.98), dec!(50))],
        )
    }

    // ---- gate ladder (decide is sync) -------------------------------------

    #[test]
    fn happy_path_decides_an_up_take() {
        let t = armed_up();
        let d = t.decide(ts(LATE_MS), None).expect("a take");
        assert_eq!(d.outcome, Outcome::Up);
        assert_eq!(d.token_id, up_token());
        assert_eq!(d.plan.worst_price.as_decimal(), dec!(0.98));
        // $10 budget caps the take (10 / 0.98 ≈ 10.2 shares, ~$10 notional, never
        // above the budget). 0.98 does not divide $10 evenly, so assert the bound.
        assert!(d.plan.notional.as_decimal() <= dec!(10));
        assert!(d.plan.notional.as_decimal() > dec!(9.9));
    }

    #[test]
    fn certain_down_takes_the_down_side() {
        let t = armed_with(
            0.01,
            AnchorSource::Chainlink,
            down_token(),
            &[(dec!(0.98), dec!(50))],
        );
        let d = t.decide(ts(LATE_MS), None).expect("a down take");
        assert_eq!(d.outcome, Outcome::Down);
        assert_eq!(d.token_id, down_token());
    }

    /// Required case 1: threshold edges. p_up exactly at the threshold takes; just
    /// inside the uncertain band refuses; symmetric for Down at 1 − threshold.
    #[test]
    fn threshold_edges() {
        // p_up == 0.97 ⇒ Up (inclusive).
        let up = armed_with(
            0.97,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.95), dec!(50))],
        );
        assert_eq!(
            up.decide(ts(LATE_MS), None).expect("up take").outcome,
            Outcome::Up
        );
        // p_up == 0.969 ⇒ uncertain.
        let near = armed_with(
            0.969,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.95), dec!(50))],
        );
        assert!(matches!(
            near.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NotCertain { .. })
        ));
        // p_up == 0.03 == 1 − 0.97 ⇒ Down (inclusive).
        let down = armed_with(
            0.03,
            AnchorSource::Chainlink,
            down_token(),
            &[(dec!(0.95), dec!(50))],
        );
        assert_eq!(
            down.decide(ts(LATE_MS), None).expect("down take").outcome,
            Outcome::Down
        );
        // p_up == 0.031 ⇒ uncertain.
        let near_down = armed_with(
            0.031,
            AnchorSource::Chainlink,
            down_token(),
            &[(dec!(0.95), dec!(50))],
        );
        assert!(matches!(
            near_down.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NotCertain { .. })
        ));
    }

    /// Required case 2: the tie lean. The model encodes "S ≥ K ⇒ Up" as p_up = 1.0
    /// (its fair value uses `if s >= k { p_up = 1.0 }`), so a saturated p_up of 1.0
    /// buys Up and 0.0 buys Down — exact-at-strike leans Up.
    #[test]
    fn tie_lean_saturated_fair() {
        let up = armed_with(
            1.0,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.98), dec!(50))],
        );
        assert_eq!(
            up.decide(ts(LATE_MS), None)
                .expect("up take at p_up=1.0")
                .outcome,
            Outcome::Up
        );
        let down = armed_with(
            0.0,
            AnchorSource::Chainlink,
            down_token(),
            &[(dec!(0.98), dec!(50))],
        );
        assert_eq!(
            down.decide(ts(LATE_MS), None)
                .expect("down take at p_up=0.0")
                .outcome,
            Outcome::Down
        );
    }

    /// Required case 3: refusal when only the fast feed confirms. A
    /// BinanceCorrected-anchored fair is refused even when certain with a takeable
    /// book in the late window; the same inputs anchored on Chainlink take.
    #[test]
    fn refuses_when_only_the_fast_feed_confirms() {
        let fast = armed_with(
            0.99,
            AnchorSource::BinanceCorrected,
            up_token(),
            &[(dec!(0.98), dec!(50))],
        );
        assert!(matches!(
            fast.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NotChainlinkAnchored {
                anchor: AnchorSource::BinanceCorrected
            })
        ));
        let chainlink = armed_with(
            0.99,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.98), dec!(50))],
        );
        assert!(chainlink.decide(ts(LATE_MS), None).is_ok());
    }

    #[test]
    fn no_active_window() {
        let t = taker();
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NoActiveWindow)
        ));
    }

    #[test]
    fn no_model_yet() {
        let mut t = taker();
        t.ingest(&open_event());
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NoModel)
        ));
    }

    #[test]
    fn window_mismatch() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&model_snapshot(
            0.99,
            ModelHealth::Ready,
            AnchorSource::Chainlink,
            Some(other_window()),
        ));
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::WindowMismatch { .. })
        ));
    }

    #[test]
    fn model_not_ready_blocks() {
        for health in [ModelHealth::Degraded, ModelHealth::Unreliable] {
            let mut t = armed_with(
                0.99,
                AnchorSource::Chainlink,
                up_token(),
                &[(dec!(0.98), dec!(50))],
            );
            // Overwrite the model with a non-Ready one.
            t.ingest(&model_snapshot(
                0.99,
                health,
                AnchorSource::Chainlink,
                Some(window()),
            ));
            assert!(
                matches!(
                    t.decide(ts(LATE_MS), None),
                    Err(NoLateTakeReason::ModelNotReady { .. })
                ),
                "{health:?} should block"
            );
        }
    }

    #[test]
    fn unusable_fair_blocks() {
        for bad in [f64::NAN, -0.1, 1.1] {
            let mut t = armed_with(
                0.99,
                AnchorSource::Chainlink,
                up_token(),
                &[(dec!(0.98), dec!(50))],
            );
            t.ingest(&model_snapshot(
                bad,
                ModelHealth::Ready,
                AnchorSource::Chainlink,
                Some(window()),
            ));
            assert!(
                matches!(
                    t.decide(ts(LATE_MS), None),
                    Err(NoLateTakeReason::UnusableFair { .. })
                ),
                "{bad} should be unusable"
            );
        }
    }

    /// p_up exactly 1.0 / 0.0 are *usable* (the certainty taker's closed interval).
    #[test]
    fn saturated_fair_is_usable() {
        let t = armed_with(
            1.0,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.98), dec!(50))],
        );
        assert!(t.decide(ts(LATE_MS), None).is_ok());
    }

    #[test]
    fn expired_window_blocks() {
        let t = armed_up();
        assert!(matches!(
            t.decide(ts(CLOSE_MS + 1), None),
            Err(NoLateTakeReason::Expired { .. })
        ));
    }

    #[test]
    fn not_late_window_blocks() {
        let t = armed_up();
        // τ = 60 s > 30 s threshold.
        assert!(matches!(
            t.decide(ts(CLOSE_MS - 60_000), None),
            Err(NoLateTakeReason::NotLateWindow { .. })
        ));
    }

    #[test]
    fn not_certain_blocks() {
        let t = armed_with(
            0.80,
            AnchorSource::Chainlink,
            up_token(),
            &[(dec!(0.78), dec!(50))],
        );
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NotCertain { .. })
        ));
    }

    #[test]
    fn no_book_for_token_blocks() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_chainlink(0.99));
        // No book ingested.
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::NoBookForToken)
        ));
    }

    #[test]
    fn all_asks_above_cap_blocks() {
        // Custom params: cap 0.95, book ask 0.96.
        let params = LateWindowTakerParams {
            price_cap: Price::on_grid(dec!(0.95), TICK).unwrap(),
            ..LateWindowTakerParams::default()
        };
        let mut t = LateWindowTaker::new(params, NormalizerParams::default());
        t.ingest(&open_event());
        t.ingest(&ready_chainlink(0.99));
        t.ingest(&book(up_token(), &[(dec!(0.96), dec!(50))]));
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::AllAsksAbovePriceCap { .. })
        ));
    }

    #[test]
    fn cooldown_blocks() {
        let mut t = armed_up();
        t.last_take_ms = Some(LATE_MS - 100); // fired 100 ms ago < 5 s cooldown
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::InCooldown { .. })
        ));
        // After the cooldown elapses, the take is allowed again.
        t.last_take_ms = Some(LATE_MS - 6_000);
        assert!(t.decide(ts(LATE_MS), None).is_ok());
    }

    /// Required case 4: budget exhaustion — directly, and via in-flight commitment.
    #[test]
    fn budget_exhausted_blocks() {
        let mut t = armed_up();
        t.realized_spent = Dollars::new(dec!(10));
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::BudgetExhausted { .. })
        ));

        // An in-flight take leaving < $1 also blocks a second.
        let mut t2 = armed_up();
        let oid = OrderId::new("lw-1").unwrap();
        t2.our_orders.insert(oid.clone());
        t2.pending.insert(oid, Dollars::new(dec!(9.50)));
        assert!(matches!(
            t2.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::BudgetExhausted { .. })
        ));
    }

    /// A roomy remaining budget sizes the take against the displayed depth, never
    /// exceeding the budget.
    #[test]
    fn sizes_against_depth_within_budget() {
        let params = LateWindowTakerParams {
            budget_per_window: Dollars::new(dec!(49)),
            ..LateWindowTakerParams::default()
        };
        let mut t = LateWindowTaker::new(params, NormalizerParams::default());
        t.ingest(&open_event());
        t.ingest(&ready_chainlink(0.99));
        t.ingest(&book(up_token(), &[(dec!(0.98), dec!(100))]));
        let d = t.decide(ts(LATE_MS), None).expect("a take");
        // 49 / 0.98 = 50 shares, $49 notional ≤ budget.
        assert_eq!(d.plan.expected_shares, Size::new(dec!(50)).unwrap());
        assert!(d.plan.notional.as_decimal() <= dec!(49));
    }

    // ---- event handling (sync) ---------------------------------------------

    /// Price ticks never trigger a take — the late-window taker uses no fast feed.
    #[test]
    fn price_ticks_are_ignored() {
        use core_types::{PriceSource, PriceTick, TickKind};
        let mut t = taker();
        let big_ramp = Event::PriceTick(PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Mid,
            value: dec!(60000),
            ts_exchange: ts(OPEN_MS),
            ts_local: ts(OPEN_MS),
        });
        // ingest returns false ⇒ on_event would attempt no take.
        assert!(!t.ingest(&big_ramp));
    }

    #[test]
    fn standing_down_blocks_and_clears() {
        let mut t = armed_up();
        t.on_risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::FeedStale,
        });
        assert!(t.is_standing_down());
        assert!(matches!(
            t.decide(ts(LATE_MS), None),
            Err(NoLateTakeReason::StandingDown)
        ));
        t.on_risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::FeedStale,
        });
        assert!(!t.is_standing_down());
        assert!(t.decide(ts(LATE_MS), None).is_ok());
    }

    #[test]
    fn multiple_breakers_hold_until_all_clear() {
        let mut t = armed_up();
        t.on_risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::FeedStale,
        });
        t.on_risk(RiskEvent::CancelAllIssued {
            reason: BreakerKind::WsDisconnect,
        });
        t.on_risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::FeedStale,
        });
        assert!(t.is_standing_down(), "one breaker still tripped");
        t.on_risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::WsDisconnect,
        });
        assert!(!t.is_standing_down());
    }

    #[test]
    fn window_open_resets_budget_and_cooldown() {
        let mut t = armed_up();
        t.realized_spent = Dollars::new(dec!(8));
        t.last_take_ms = Some(OPEN_MS + 100);
        t.our_orders.insert(OrderId::new("lw-9").unwrap());
        let next = WindowId {
            series: window().series,
            open_time: TimestampMs::from_millis(CLOSE_MS),
        };
        let mut m = (*market()).clone();
        m.window = next;
        m.close_time = TimestampMs::from_millis(CLOSE_MS + 300_000);
        t.ingest(&Event::Window {
            market: Arc::new(m),
            lifecycle: WindowLifecycle::Open,
        });
        assert_eq!(t.realized_spent(), Dollars::ZERO);
        assert_eq!(t.effective_spent(), Dollars::ZERO);
        assert!(t.last_take_ms.is_none());
        assert!(t.our_orders.is_empty());
    }

    // ---- budget reconciliation from the venue stream (sync) ----------------

    #[test]
    fn fills_charge_budget_and_terminal_drops_residual() {
        let mut t = armed_up();
        let oid = OrderId::new("lw-7").unwrap();
        t.our_orders.insert(oid.clone());
        t.pending.insert(oid.clone(), Dollars::new(dec!(10)));
        assert_eq!(t.effective_spent(), Dollars::new(dec!(10)));

        // A partial taker fill: 5 shares @ 0.98 = $4.90.
        t.on_venue_event(
            &VenueEvent::Fill(Arc::new(fill(&oid, dec!(0.98), dec!(5)))),
            ts(OPEN_MS),
        );
        assert_eq!(t.realized_spent(), Dollars::new(dec!(4.90)));
        // effective = realized 4.90 + pending 5.10 = 10 (unchanged total).
        assert_eq!(t.effective_spent(), Dollars::new(dec!(10)));

        // The FAK remainder is killed — residual pending dropped.
        t.on_venue_event(
            &VenueEvent::Order(Arc::new(order_update(&oid, OrderState::Canceled))),
            ts(OPEN_MS),
        );
        assert_eq!(t.effective_spent(), Dollars::new(dec!(4.90)));
        assert_eq!(t.realized_spent(), Dollars::new(dec!(4.90)));
    }

    #[test]
    fn foreign_fills_are_ignored() {
        let mut t = armed_up();
        t.our_orders.insert(OrderId::new("lw-1").unwrap());
        let foreign = OrderId::new("qm-9").unwrap();
        t.on_venue_event(
            &VenueEvent::Fill(Arc::new(fill(&foreign, dec!(0.98), dec!(20)))),
            ts(OPEN_MS),
        );
        assert_eq!(t.realized_spent(), Dollars::ZERO);
    }

    fn fill(oid: &OrderId, price: Decimal, size: Decimal) -> Fill {
        Fill {
            order_id: oid.clone(),
            trade_id: Some("t1".to_owned()),
            window: window(),
            token_id: up_token(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: Price::on_grid(price, TICK).unwrap(),
            size: Size::new(size).unwrap(),
            liquidity: Liquidity::Taker,
            fee: core_types::taker_fee(
                Size::new(size).unwrap(),
                dec!(0.07),
                Price::on_grid(price, TICK).unwrap(),
            ),
            ts_venue: TimestampMs::from_millis(OPEN_MS),
            ts_local: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn order_update(oid: &OrderId, state: OrderState) -> OrderUpdate {
        OrderUpdate {
            order_id: oid.clone(),
            window: window(),
            token_id: up_token(),
            side: Side::Buy,
            state,
            price: Price::on_grid(dec!(0.98), TICK).unwrap(),
            original_size: Size::new(dec!(10.2)).unwrap(),
            filled_size: Size::new(dec!(5)).unwrap(),
            reject_reason: None,
            ts_venue: Some(TimestampMs::from_millis(OPEN_MS)),
            ts_local: TimestampMs::from_millis(OPEN_MS),
        }
    }
}
