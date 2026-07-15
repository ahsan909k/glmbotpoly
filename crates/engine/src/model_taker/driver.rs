//! [`ModelTaker`]: the async, IO-bearing executor for the champion-model taker.
//!
//! Mirrors [`MomentumTaker`](crate::taker::MomentumTaker) and
//! [`LateWindowTaker`](crate::late_window::LateWindowTaker): it needs **no**
//! `tokio` (it only `.await`s the [`VenuePort`] futures and takes `now:
//! TimestampMs` as a parameter — sans-clock), folds its own fills from the venue
//! stream, owns the `tripped`/`standing_down` risk veto, and routes every FAK
//! through the [`normalize`](crate::normalize) chokepoint. Logs via `tracing`
//! (target `model-taker`).
//!
//! ## Per-window state (the one difference from the other takers)
//!
//! The `shadow` model produces a `p_up` for **every** concurrently-active window,
//! so — unlike the single-`active` momentum/late takers — the model taker keeps a
//! [`ModelWindowState`] **per window** and trades all four series at once. Each
//! window carries its own independent budget/orders; a venue fill is routed to the
//! right window via `order_window`.
//!
//! ## Fire trigger
//!
//! Bus events (`Window`/`Book`/`Risk`) only refresh state; the **only** path that
//! fires is [`ModelTaker::on_prediction`], driven by a fresh shadow prediction
//! (the ~5 s cadence is the natural throttle — no cooldown, per the recipe).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use core_types::{
    BookSnapshot, BreakerKind, Decimal, Dollars, Event, Liquidity, MarketInfo, OrderId, OrderQty,
    Outcome, RiskEvent, Side, TimeInForce, TimestampMs, TokenId, WindowId, WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::edge::{ModelTakePlan, plan_model_take};
use super::{ModelPrediction, ModelTakeOutcome, ModelTakerParams, NoModelTakeReason};
use crate::arbitration::{FireLedger, TakerId};
use crate::normalize::{NormalizerParams, OrderDraft, normalize};
use crate::quote_manager::RestingLookup;

/// Per-window model-taker state: the market metadata, the cached books, and the
/// committed-in-flight budget for this window.
struct ModelWindowState {
    market: Arc<MarketInfo>,
    /// Latest full book per this window's token (depth for sizing).
    books: HashMap<TokenId, Arc<BookSnapshot>>,
    /// Realized taker spend this window (from our own fills).
    realized_spent: Dollars,
    /// Committed-but-not-yet-realized notional per in-flight take order.
    pending: HashMap<OrderId, Dollars>,
    /// Order ids of takes we placed on this window.
    our_orders: HashSet<OrderId>,
}

impl ModelWindowState {
    fn new(market: Arc<MarketInfo>) -> Self {
        Self {
            market,
            books: HashMap::new(),
            realized_spent: Dollars::ZERO,
            pending: HashMap::new(),
            our_orders: HashSet::new(),
        }
    }

    /// Committed-in-flight spend: `realized_spent + Σ pending`.
    fn effective_spent(&self) -> Dollars {
        self.pending
            .values()
            .copied()
            .fold(self.realized_spent, |acc, p| acc + p)
    }
}

/// A fully-vetted take decision: the FAK to build.
#[derive(Debug)]
struct Decision {
    window: WindowId,
    outcome: Outcome,
    token_id: TokenId,
    market: Arc<MarketInfo>,
    plan: ModelTakePlan,
}

/// The champion-model taker (CLAUDE.md §8). Generic placement happens through the
/// [`VenuePort`] passed to each method; the taker stores no port (and so is not
/// itself generic).
pub struct ModelTaker {
    params: ModelTakerParams,
    normalizer_params: NormalizerParams,
    /// Per-window state — the model taker trades every active window concurrently.
    windows: HashMap<WindowId, ModelWindowState>,
    /// `order_id → window`, so a venue fill finds its budget in O(1).
    order_window: HashMap<OrderId, WindowId>,
    /// True while a risk breaker holds the taker down.
    standing_down: bool,
    /// Currently-tripped breakers (stand down until empty).
    tripped: HashSet<BreakerKind>,
    /// Monotonic per-process placement sequence (client-id uniqueness).
    seq: u64,
    /// Count of takes fired so far (test/diagnostic metric).
    take_count: u64,
}

impl ModelTaker {
    /// Builds a taker with the given tunables and an empty state.
    #[must_use]
    pub fn new(params: ModelTakerParams, normalizer_params: NormalizerParams) -> Self {
        Self {
            params,
            normalizer_params,
            windows: HashMap::new(),
            order_window: HashMap::new(),
            standing_down: false,
            tripped: HashSet::new(),
            seq: 0,
            take_count: 0,
        }
    }

    /// Number of takes fired so far (each is one accepted FAK placement).
    #[must_use]
    pub fn take_count(&self) -> u64 {
        self.take_count
    }

    /// Realized taker spend across all live windows.
    #[must_use]
    pub fn realized_spent(&self) -> Dollars {
        self.windows
            .values()
            .fold(Dollars::ZERO, |acc, st| acc + st.realized_spent)
    }

    /// Committed-in-flight taker spend across all live windows.
    #[must_use]
    pub fn effective_spent(&self) -> Dollars {
        self.windows
            .values()
            .fold(Dollars::ZERO, |acc, st| acc + st.effective_spent())
    }

    /// True while a risk breaker holds the taker down.
    #[must_use]
    pub fn is_standing_down(&self) -> bool {
        self.standing_down
    }

    /// Whether `id` is one of this taker's placed orders (driver attribution).
    #[must_use]
    pub fn owns(&self, id: &OrderId) -> bool {
        self.order_window.contains_key(id)
    }

    /// Applies one bus event — **sync state refresh only, never fires** (the fire
    /// trigger is a fresh prediction, [`on_prediction`](ModelTaker::on_prediction)).
    /// Handles window lifecycle, book caching, and the risk veto; ignores
    /// `Model`/`PriceTick` (the model taker's signal is the shadow prediction, not
    /// the analytic model snapshot).
    pub fn on_bus_event(&mut self, event: &Event) {
        match event {
            Event::Window { market, lifecycle } => self.on_window(market, *lifecycle),
            Event::Book(snap) => self.cache_book(snap),
            Event::Risk(risk) => self.on_risk(*risk),
            _ => {}
        }
    }

    /// Folds one item from the venue's order/fill stream: our own fills charge the
    /// owning window's budget; a terminal update drops its in-flight residual.
    pub fn on_venue_event(&mut self, ve: &VenueEvent, _now: TimestampMs) {
        match ve {
            VenueEvent::Fill(f) => {
                if let Some(w) = self.order_window.get(&f.order_id).copied()
                    && let Some(st) = self.windows.get_mut(&w)
                {
                    debug_assert_eq!(
                        f.liquidity,
                        Liquidity::Taker,
                        "the model taker only ever fires FAK taker orders"
                    );
                    let filled = Dollars::new(f.price.as_decimal() * f.size.as_decimal());
                    st.realized_spent = st.realized_spent + filled;
                    if let Some(rem) = st.pending.get_mut(&f.order_id) {
                        let after = *rem - filled;
                        *rem = if after.is_negative() {
                            Dollars::ZERO
                        } else {
                            after
                        };
                    }
                    tracing::info!(target: "model-taker", order = %f.order_id, price = %f.price, size = %f.size, "taker fill");
                }
            }
            VenueEvent::Order(u) => {
                if u.state.is_terminal()
                    && let Some(w) = self.order_window.get(&u.order_id).copied()
                    && let Some(st) = self.windows.get_mut(&w)
                {
                    // FAK remainder killed — the unfilled part was never spent.
                    st.pending.remove(&u.order_id);
                }
            }
            VenueEvent::Connectivity { .. } => {}
        }
    }

    /// Delivers one shadow prediction — the **only** path that can fire. Returns
    /// the outcome (fired / suppressed + reason) for decision journaling.
    pub async fn on_prediction<P: VenuePort>(
        &mut self,
        pred: &ModelPrediction,
        port: &P,
        now: TimestampMs,
        arbiter: &mut FireLedger,
        resting: Option<&dyn RestingLookup>,
    ) -> ModelTakeOutcome {
        match self.decide(pred, now, arbiter, resting) {
            Ok(decision) => self.fire(port, decision, now, arbiter).await,
            Err(reason) => {
                tracing::debug!(target: "model-taker", reason = %reason, "no take");
                ModelTakeOutcome::Suppressed(reason)
            }
        }
    }

    // ---- ingestion (sync) --------------------------------------------------

    fn on_window(&mut self, market: &Arc<MarketInfo>, lifecycle: WindowLifecycle) {
        match lifecycle {
            WindowLifecycle::Open => {
                // Fresh per-window state (budget/orders/books) on open.
                self.windows
                    .insert(market.window, ModelWindowState::new(Arc::clone(market)));
                tracing::info!(target: "model-taker", window = %market.window, "window open: model taker armed");
            }
            WindowLifecycle::Closing
            | WindowLifecycle::Closed
            | WindowLifecycle::Resolved { .. } => {
                if self.windows.remove(&market.window).is_some() {
                    let w = market.window;
                    self.order_window.retain(|_, ow| *ow != w);
                    tracing::debug!(target: "model-taker", window = %w, ?lifecycle, "window ended: state dropped");
                }
            }
            WindowLifecycle::Discovered => {}
        }
    }

    /// Caches a book onto whichever active window owns its token.
    fn cache_book(&mut self, snap: &Arc<BookSnapshot>) {
        for st in self.windows.values_mut() {
            if st.market.tokens.outcome_of(&snap.token_id).is_some() {
                st.books.insert(snap.token_id.clone(), Arc::clone(snap));
                return;
            }
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
                tracing::warn!(target: "model-taker", ?breaker, "risk veto: model taker standing down");
            }
            RiskEvent::BreakerCleared { breaker } => {
                self.tripped.remove(&breaker);
                if self.tripped.is_empty() {
                    self.standing_down = false;
                    tracing::info!(target: "model-taker", "all breakers cleared: model taker resuming");
                }
            }
        }
    }

    // ---- decision (sync, pure) ---------------------------------------------

    /// The full take gate ladder. Each gate maps to a typed [`NoModelTakeReason`].
    /// Pure and synchronous.
    fn decide(
        &self,
        pred: &ModelPrediction,
        now: TimestampMs,
        arbiter: &FireLedger,
        resting: Option<&dyn RestingLookup>,
    ) -> Result<Decision, NoModelTakeReason> {
        if self.standing_down {
            return Err(NoModelTakeReason::StandingDown);
        }
        let window = pred.window();
        let st = self
            .windows
            .get(&window)
            .ok_or(NoModelTakeReason::NoWindowState)?;
        if pred.model_stale {
            return Err(NoModelTakeReason::ModelStale);
        }
        if pred.finite_count < self.params.min_finite_count {
            return Err(NoModelTakeReason::InsufficientCoverage {
                finite: pred.finite_count,
                need: self.params.min_finite_count,
            });
        }
        let p = pred.p_up;
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(NoModelTakeReason::UnusableFair { p_up: p });
        }
        // θ rule: |p − 0.5| ≥ theta; side by the sign.
        let outcome = if p >= 0.5 + self.params.theta {
            Outcome::Up
        } else if p <= 0.5 - self.params.theta {
            Outcome::Down
        } else {
            return Err(NoModelTakeReason::BelowTheta {
                p_up: p,
                theta: self.params.theta,
            });
        };
        // Defensive expiry guard (holds to resolution otherwise — no τ/late gate).
        let tau = tau_secs(now, st.market.close_time);
        if tau <= 0.0 {
            return Err(NoModelTakeReason::Expired { tau_secs: tau });
        }
        // Arbitration: the model taker defers on assets where momentum wins.
        if let Err(block) = arbiter.check(TakerId::Model, window, now) {
            return Err(NoModelTakeReason::ArbitrationSuppressed {
                winner: block.winner,
                remaining_ms: block.remaining_ms,
            });
        }
        // Per-window budget (committed-in-flight aware).
        let remaining = self.params.budget_per_window - st.effective_spent();
        if remaining.as_decimal() < Decimal::ONE {
            return Err(NoModelTakeReason::BudgetExhausted {
                remaining,
                need: Dollars::new(Decimal::ONE),
            });
        }
        let token_id = st.market.tokens.get(outcome).clone();
        let book = st
            .books
            .get(&token_id)
            .ok_or(NoModelTakeReason::NoBookForToken)?;
        let age = now.as_millis() - book.ts.as_millis();
        if age > self.params.max_book_staleness_ms {
            return Err(NoModelTakeReason::BookStale { age_ms: age });
        }
        // §7 self-match filter BEFORE the walk (the maker's per-window view).
        let view = resting.and_then(|r| r.resting_view_for(window));
        let asks = crate::self_match::filter_asks(outcome, &book.asks, window, view);
        let fee_rate = if st.market.fees.enabled {
            st.market.fees.rate
        } else {
            Decimal::ZERO
        };
        let plan = plan_model_take(outcome, &asks, fee_rate, self.params.price_cap, remaining)?;
        Ok(Decision {
            window,
            outcome,
            token_id,
            market: Arc::clone(&st.market),
            plan,
        })
    }

    // ---- firing (async) ----------------------------------------------------

    async fn fire<P: VenuePort>(
        &mut self,
        port: &P,
        decision: Decision,
        now: TimestampMs,
        arbiter: &mut FireLedger,
    ) -> ModelTakeOutcome {
        let seq = self.next_seq();
        let open_ms = decision.window.open_time.as_millis();
        let draft = OrderDraft {
            client_id: Some(format!("md:{open_ms}:{seq}")),
            window: decision.window,
            token_id: decision.token_id.clone(),
            outcome: decision.outcome,
            side: Side::Buy,
            price: decision.plan.worst_price.as_decimal(),
            qty: OrderQty::Notional(decision.plan.notional),
            tif: TimeInForce::Fak,
        };
        // FAK is not post-only, so the normalizer's cross check is skipped — no
        // book view needed. It only snaps the worst-price cap and re-checks the $1
        // notional (both already guaranteed by plan_model_take).
        let order = match normalize(&draft, &decision.market, None, &self.normalizer_params) {
            Ok(n) => n.order,
            Err(e) => {
                tracing::warn!(target: "model-taker", reason = %e, "normalize rejected the FAK (unexpected — plan guarantees ≥$1 notional + on-grid worst price)");
                return ModelTakeOutcome::Suppressed(NoModelTakeReason::PlaceRejected);
            }
        };
        match port.place(&order).await {
            Ok(acc) => {
                let window = decision.window;
                if let Some(st) = self.windows.get_mut(&window) {
                    st.our_orders.insert(acc.order_id.clone());
                    st.pending
                        .insert(acc.order_id.clone(), decision.plan.notional);
                }
                self.order_window.insert(acc.order_id, window);
                self.take_count += 1;
                arbiter.record(TakerId::Model, window, now);
                tracing::info!(
                    target: "model-taker",
                    window = %window,
                    outcome = %decision.outcome,
                    worst = %decision.plan.worst_price,
                    notional = %decision.plan.notional,
                    shares = %decision.plan.expected_shares,
                    "model take (FAK)"
                );
                ModelTakeOutcome::Fired {
                    window,
                    outcome: decision.outcome,
                    shares: decision.plan.expected_shares,
                    notional: decision.plan.notional,
                    worst_price: decision.plan.worst_price,
                }
            }
            Err(e) => {
                tracing::warn!(target: "model-taker", error = %e, "FAK place failed");
                ModelTakeOutcome::Suppressed(NoModelTakeReason::PlaceRejected)
            }
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq = self.seq.wrapping_add(1);
        s
    }
}

/// Seconds remaining to close, clamped at zero — computed inline so the driver
/// needs no `timeutil` dependency (the `quote_manager`/`taker` precedent).
fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.as_millis() - now.as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arbitration::FireLedger;
    use core_types::{
        Asset, BookLevel, ConditionId, FeeParams, Price, ResolutionSource, Series, Size, TickSize,
        TokenId, TokenPair, WindowDuration,
    };
    use rust_decimal::dec;

    const TICK: TickSize = TickSize::T001;

    fn series(asset: Asset) -> Series {
        Series {
            asset,
            duration: WindowDuration::M5,
        }
    }

    fn market(asset: Asset, open_ms: i64) -> Arc<MarketInfo> {
        let win = WindowId {
            series: series(asset),
            open_time: TimestampMs::from_millis(open_ms),
        };
        Arc::new(MarketInfo {
            window: win,
            event_slug: format!("slug-{open_ms}"),
            condition_id: ConditionId::new(format!("0x{open_ms:064x}")).expect("cid"),
            tokens: TokenPair {
                up: TokenId::new((open_ms * 10 + 1).to_string()).expect("up"),
                down: TokenId::new((open_ms * 10 + 2).to_string()).expect("down"),
            },
            close_time: TimestampMs::from_millis(open_ms + 300_000),
            strike: None,
            tick_size: TICK,
            min_order_size: Size::new(dec!(5)).expect("min"),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.20),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("data.chain.link/streams/btc-usd"),
        })
    }

    fn book(
        token: &TokenId,
        cid: &ConditionId,
        asks: &[(Decimal, Decimal)],
        ts: i64,
    ) -> Arc<BookSnapshot> {
        Arc::new(BookSnapshot {
            token_id: token.clone(),
            condition_id: cid.clone(),
            bids: vec![],
            asks: asks
                .iter()
                .map(|(p, s)| BookLevel {
                    price: Price::on_grid(*p, TICK).expect("grid"),
                    size: Size::new(*s).expect("size"),
                })
                .collect(),
            ts: TimestampMs::from_millis(ts),
            seq_hash: None,
        })
    }

    fn pred(asset: Asset, open_ms: i64, p_up: f64, ts: i64) -> ModelPrediction {
        ModelPrediction {
            series: series(asset),
            window_open_ms: open_ms,
            ts: TimestampMs::from_millis(ts),
            p_up,
            finite_count: 24,
            model_stale: false,
        }
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    /// A taker armed on a window, with a book cached for the Up token.
    fn armed(asset: Asset, open_ms: i64) -> ModelTaker {
        let mut t = ModelTaker::new(ModelTakerParams::default(), NormalizerParams::default());
        let m = market(asset, open_ms);
        t.on_bus_event(&Event::Window {
            market: Arc::clone(&m),
            lifecycle: WindowLifecycle::Open,
        });
        let b = book(
            &m.tokens.up,
            &m.condition_id,
            &[(dec!(0.80), dec!(50))],
            open_ms + 1_000,
        );
        t.on_bus_event(&Event::Book(b));
        t
    }

    #[test]
    fn theta_rule_and_side() {
        let t = armed(Asset::Eth, 0);
        let arb = FireLedger::default();
        // p = 0.5 + theta ⇒ Up.
        let d = t
            .decide(&pred(Asset::Eth, 0, 0.53, 2_000), ts(2_000), &arb, None)
            .expect("a take");
        assert_eq!(d.outcome, Outcome::Up);
        // p = 0.5 + theta − ε ⇒ BelowTheta.
        let r = t
            .decide(&pred(Asset::Eth, 0, 0.52, 2_000), ts(2_000), &arb, None)
            .unwrap_err();
        assert!(matches!(r, NoModelTakeReason::BelowTheta { .. }));
    }

    #[test]
    fn down_side_walks_the_down_book() {
        let mut t = ModelTaker::new(ModelTakerParams::default(), NormalizerParams::default());
        let m = market(Asset::Eth, 0);
        t.on_bus_event(&Event::Window {
            market: Arc::clone(&m),
            lifecycle: WindowLifecycle::Open,
        });
        // Cache a Down-token book; a low p_up should buy Down.
        let b = book(
            &m.tokens.down,
            &m.condition_id,
            &[(dec!(0.70), dec!(20))],
            1_000,
        );
        t.on_bus_event(&Event::Book(b));
        let arb = FireLedger::default();
        let d = t
            .decide(&pred(Asset::Eth, 0, 0.10, 2_000), ts(2_000), &arb, None)
            .expect("a take");
        assert_eq!(d.outcome, Outcome::Down);
    }

    #[test]
    fn gates_no_window_stale_model_coverage_expiry_book() {
        let t = armed(Asset::Eth, 0);
        let arb = FireLedger::default();
        // Wrong window.
        assert!(matches!(
            t.decide(&pred(Asset::Eth, 999, 0.9, 2_000), ts(2_000), &arb, None),
            Err(NoModelTakeReason::NoWindowState)
        ));
        // Stale model.
        let mut sp = pred(Asset::Eth, 0, 0.9, 2_000);
        sp.model_stale = true;
        assert!(matches!(
            t.decide(&sp, ts(2_000), &arb, None),
            Err(NoModelTakeReason::ModelStale)
        ));
        // Low coverage.
        let mut lp = pred(Asset::Eth, 0, 0.9, 2_000);
        lp.finite_count = 10;
        assert!(matches!(
            t.decide(&lp, ts(2_000), &arb, None),
            Err(NoModelTakeReason::InsufficientCoverage { .. })
        ));
        // Expired (now past close at open+300_000).
        assert!(matches!(
            t.decide(&pred(Asset::Eth, 0, 0.9, 301_000), ts(301_000), &arb, None),
            Err(NoModelTakeReason::Expired { .. })
        ));
        // Stale book (book ts open+1000, now open+10_000 ⇒ age 9s > 2s).
        assert!(matches!(
            t.decide(&pred(Asset::Eth, 0, 0.9, 10_000), ts(10_000), &arb, None),
            Err(NoModelTakeReason::BookStale { .. })
        ));
    }

    #[test]
    fn arbitration_suppresses_the_loser() {
        let t = armed(Asset::Btc, 0); // BTC ⇒ momentum wins
        let mut p = HashMap::new();
        p.insert(Asset::Btc, TakerId::Momentum);
        let mut arb = FireLedger::new(3_000, p);
        // Momentum just fired this window.
        let win = WindowId {
            series: series(Asset::Btc),
            open_time: ts(0),
        };
        arb.record(TakerId::Momentum, win, ts(1_000));
        let r = t
            .decide(&pred(Asset::Btc, 0, 0.9, 2_000), ts(2_000), &arb, None)
            .unwrap_err();
        assert!(matches!(r, NoModelTakeReason::ArbitrationSuppressed { .. }));
    }

    #[test]
    fn per_window_independence_and_close_prunes() {
        let mut t = ModelTaker::new(ModelTakerParams::default(), NormalizerParams::default());
        for open in [0i64, 300_000] {
            let m = market(Asset::Eth, open);
            t.on_bus_event(&Event::Window {
                market: Arc::clone(&m),
                lifecycle: WindowLifecycle::Open,
            });
        }
        assert_eq!(t.windows.len(), 2);
        // Closing one window drops only its state.
        let m0 = market(Asset::Eth, 0);
        t.on_bus_event(&Event::Window {
            market: m0,
            lifecycle: WindowLifecycle::Closed,
        });
        assert_eq!(t.windows.len(), 1);
        assert!(t.windows.contains_key(&WindowId {
            series: series(Asset::Eth),
            open_time: ts(300_000),
        }));
    }
}
