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
    ModelSnapshot, OrderId, OrderQty, Outcome, RiskEvent, Side, TimeInForce, TimestampMs, TokenId,
    WindowId, WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::edge::{CertaintyTakePlan, plan_certainty_take};
use super::{LateWindowTakerParams, NoLateTakeReason};
use crate::normalize::{NormalizerParams, OrderDraft, normalize, split_take_clips};
use crate::quote_manager::RestingLookup;

/// Seconds remaining to close, clamped at zero — computed inline so the driver
/// needs no `timeutil` dependency (the `quote_manager`/`taker` precedent).
fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.as_millis() - now.as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

/// Per-window late-window-taker state: the market, the cached books, the latest
/// model snapshot, and the committed-in-flight budget/cooldown for this window.
struct State {
    market: Arc<MarketInfo>,
    /// Latest full book per this window's token (depth for sizing).
    books: HashMap<TokenId, Arc<BookSnapshot>>,
    /// Latest model snapshot for this window.
    last_model: Option<ModelSnapshot>,
    /// Realized taker spend this window (from our own fills).
    realized_spent: Dollars,
    /// Committed-but-not-yet-realized notional per in-flight take order.
    pending: HashMap<OrderId, Dollars>,
    /// Order ids of takes we placed on this window (fill attribution).
    our_orders: HashSet<OrderId>,
    /// Wall time (ms) of the last fired take — the cooldown anchor.
    last_take_ms: Option<i64>,
}

impl State {
    fn new(market: Arc<MarketInfo>) -> Self {
        Self {
            market,
            books: HashMap::new(),
            last_model: None,
            realized_spent: Dollars::ZERO,
            pending: HashMap::new(),
            our_orders: HashSet::new(),
            last_take_ms: None,
        }
    }

    /// Budget committed so far this window: realized fills plus in-flight notional.
    fn effective_spent(&self) -> Dollars {
        self.pending
            .values()
            .copied()
            .fold(self.realized_spent, |acc, p| acc + p)
    }
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
    /// Per-window state — the taker trades every active window concurrently.
    windows: HashMap<WindowId, State>,
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

impl LateWindowTaker {
    /// Builds a taker with the given tunables and an empty state.
    #[must_use]
    pub fn new(params: LateWindowTakerParams, normalizer_params: NormalizerParams) -> Self {
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

    /// Realized taker spend across all live windows (from our own fills).
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

    /// Whether a risk breaker currently holds the taker down.
    #[must_use]
    pub fn is_standing_down(&self) -> bool {
        self.standing_down
    }

    /// Whether `id` is one of this taker's placed orders (driver attribution).
    #[must_use]
    pub fn owns(&self, id: &OrderId) -> bool {
        self.order_window.contains_key(id)
    }

    /// Handles one bus event; attempts a take on the affected window after a model
    /// or active-token book update (a fresh fair or a fresh book may have opened the
    /// endgame). A price tick is ignored — this taker uses no fast-feed signal.
    pub async fn on_event<P: VenuePort>(
        &mut self,
        event: &Event,
        port: &P,
        now: TimestampMs,
        resting: Option<&dyn RestingLookup>,
    ) {
        if let Some(window) = self.ingest(event) {
            self.attempt_take(window, port, now, resting).await;
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
                        "the late-window taker only ever fires FAK taker orders"
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
                    tracing::info!(target: "late-window-taker", order = %f.order_id, price = %f.price, size = %f.size, "taker fill");
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
            // User-WS connectivity is a risk-manager concern; the taker ignores it.
            VenueEvent::Connectivity { .. } => {}
        }
    }

    // ---- ingestion (sync) --------------------------------------------------

    /// Applies one bus event to the taker's state. Returns the window to attempt a
    /// take on when a take should be re-evaluated (a model update or a fresh book
    /// on an active token); `None` otherwise.
    fn ingest(&mut self, event: &Event) -> Option<WindowId> {
        match event {
            // No fast feed: the late-window decision is Chainlink-anchored only,
            // so price ticks never feed it.
            Event::PriceTick(_) => None,
            Event::Model(snap) => {
                let win = snap.window?;
                if let Some(st) = self.windows.get_mut(&win) {
                    st.last_model = Some(*snap);
                    Some(win)
                } else {
                    None
                }
            }
            Event::Book(snap) => self.cache_book(snap),
            Event::Window { market, lifecycle } => {
                self.on_window(market, *lifecycle);
                None
            }
            Event::Risk(risk) => {
                self.on_risk(*risk);
                None
            }
            _ => None,
        }
    }

    /// Caches a book onto whichever active window owns its token; returns that
    /// window (so a take can be re-evaluated) if it belonged to one.
    fn cache_book(&mut self, snap: &Arc<BookSnapshot>) -> Option<WindowId> {
        for (win, st) in &mut self.windows {
            if st.market.tokens.outcome_of(&snap.token_id).is_some() {
                st.books.insert(snap.token_id.clone(), Arc::clone(snap));
                return Some(*win);
            }
        }
        None
    }

    fn on_window(&mut self, market: &Arc<MarketInfo>, lifecycle: WindowLifecycle) {
        match lifecycle {
            WindowLifecycle::Open => {
                // Fresh per-window state (budget/cooldown/orders/books) on open.
                self.windows
                    .insert(market.window, State::new(Arc::clone(market)));
                tracing::info!(target: "late-window-taker", window = %market.window, "window open: late taker armed");
            }
            WindowLifecycle::Closing
            | WindowLifecycle::Closed
            | WindowLifecycle::Resolved { .. } => {
                let w = market.window;
                if self.windows.remove(&w).is_some() {
                    self.order_window.retain(|_, ow| *ow != w);
                    tracing::debug!(target: "late-window-taker", window = %w, ?lifecycle, "window ended: late taker stood down");
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
        window: WindowId,
        now: TimestampMs,
        resting: Option<&dyn RestingLookup>,
    ) -> Result<Decision, NoLateTakeReason> {
        if self.standing_down {
            return Err(NoLateTakeReason::StandingDown);
        }
        let st = self
            .windows
            .get(&window)
            .ok_or(NoLateTakeReason::NoActiveWindow)?;
        let model = st.last_model.ok_or(NoLateTakeReason::NoModel)?;
        // Defensive: the model is stored in its own window's slot, so this always
        // holds — kept as a belt-and-suspenders (and for the typed reason).
        if model.window != Some(window) {
            return Err(NoLateTakeReason::WindowMismatch {
                model: model.window,
                active: window,
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
        let tau = tau_secs(now, st.market.close_time);
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
        if let Some(last) = st.last_take_ms {
            let elapsed = now.as_millis() - last;
            if elapsed < self.params.cooldown_ms {
                return Err(NoLateTakeReason::InCooldown {
                    remaining_ms: self.params.cooldown_ms - elapsed,
                });
            }
        }
        // Budget (committed-in-flight aware, per window).
        let remaining = self.params.budget_per_window - st.effective_spent();
        if remaining.as_decimal() < Decimal::ONE {
            return Err(NoLateTakeReason::BudgetExhausted {
                remaining,
                need: Dollars::new(Decimal::ONE),
            });
        }
        let token_id = st.market.tokens.get(outcome).clone();
        let book = st
            .books
            .get(&token_id)
            .ok_or(NoLateTakeReason::NoBookForToken)?;
        let fees = &st.market.fees;
        let fee_rate = if fees.enabled {
            fees.rate
        } else {
            Decimal::ZERO
        };
        // §7 self-match filter: never lift our own mirrored resting liquidity on
        // this window (the maker's per-window view).
        let view = resting.and_then(|r| r.resting_view_for(window));
        let asks = crate::self_match::filter_asks(outcome, &book.asks, window, view);
        let plan = plan_certainty_take(outcome, &asks, fee_rate, self.params.price_cap, remaining)?;
        Ok(Decision {
            outcome,
            token_id,
            market: Arc::clone(&st.market),
            plan,
        })
    }

    // ---- firing (async) ----------------------------------------------------

    async fn attempt_take<P: VenuePort>(
        &mut self,
        window: WindowId,
        port: &P,
        now: TimestampMs,
        resting: Option<&dyn RestingLookup>,
    ) {
        match self.decide(window, now, resting) {
            Ok(decision) => self.fire(window, port, decision, now).await,
            Err(reason) => {
                tracing::debug!(target: "late-window-taker", reason = %reason, "no take");
            }
        }
    }

    async fn fire<P: VenuePort>(
        &mut self,
        window: WindowId,
        port: &P,
        decision: Decision,
        now: TimestampMs,
    ) {
        let open_ms = decision.market.window.open_time.as_millis();
        // Split into sequential FAK clips of ≤ clip_size_shares (§8 burst
        // pattern); no clip ⇒ one full-size FAK (byte-identical to before).
        let clips = split_take_clips(
            decision.plan.expected_shares,
            decision.plan.worst_price,
            decision.plan.notional,
            self.normalizer_params.clip_size_shares,
        );
        for clip_notional in clips {
            let seq = self.next_seq();
            let draft = OrderDraft {
                client_id: Some(format!("lw:{open_ms}:{seq}")),
                window: decision.market.window,
                token_id: decision.token_id.clone(),
                outcome: decision.outcome,
                side: Side::Buy,
                price: decision.plan.worst_price.as_decimal(),
                qty: OrderQty::Notional(clip_notional),
                tif: TimeInForce::Fak,
            };
            // FAK is not post-only, so the normalizer's cross check is skipped — no
            // book view needed. It only snaps the worst-price cap (already on-grid)
            // and re-checks the $1 notional (guaranteed by the plan / clip split).
            let order = match normalize(&draft, &decision.market, None, &self.normalizer_params) {
                Ok(n) => n.order,
                Err(e) => {
                    tracing::warn!(target: "late-window-taker", reason = %e, "normalize rejected a FAK clip (skipping this clip)");
                    continue;
                }
            };
            match port.place(&order).await {
                Ok(acc) => {
                    if let Some(st) = self.windows.get_mut(&window) {
                        st.our_orders.insert(acc.order_id.clone());
                        st.pending.insert(acc.order_id.clone(), clip_notional);
                        st.last_take_ms = Some(now.as_millis());
                    }
                    self.order_window.insert(acc.order_id, window);
                    self.take_count += 1;
                    tracing::info!(
                        target: "late-window-taker",
                        outcome = %decision.outcome,
                        worst = %decision.plan.worst_price,
                        notional = %clip_notional,
                        "late-window certainty take (FAK clip)"
                    );
                }
                Err(e) => {
                    tracing::warn!(target: "late-window-taker", error = %e, "FAK place failed");
                }
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
impl LateWindowTaker {
    /// Test-only: mutable access to a window's per-window state.
    fn test_state_mut(&mut self, w: WindowId) -> &mut State {
        self.windows.get_mut(&w).expect("open window state")
    }

    /// Test-only: registers `oid` as an in-flight take on `w` (the post-fire
    /// bookkeeping) with `notional` pending — for budget-reconciliation tests.
    fn test_register_pending(&mut self, w: WindowId, oid: OrderId, notional: Dollars) {
        if let Some(st) = self.windows.get_mut(&w) {
            st.our_orders.insert(oid.clone());
            st.pending.insert(oid.clone(), notional);
        }
        self.order_window.insert(oid, w);
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
        let d = t.decide(window(), ts(LATE_MS), None).expect("a take");
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
        let d = t.decide(window(), ts(LATE_MS), None).expect("a down take");
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
            up.decide(window(), ts(LATE_MS), None)
                .expect("up take")
                .outcome,
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
            near.decide(window(), ts(LATE_MS), None),
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
            down.decide(window(), ts(LATE_MS), None)
                .expect("down take")
                .outcome,
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
            near_down.decide(window(), ts(LATE_MS), None),
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
            up.decide(window(), ts(LATE_MS), None)
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
            down.decide(window(), ts(LATE_MS), None)
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
            fast.decide(window(), ts(LATE_MS), None),
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
        assert!(chainlink.decide(window(), ts(LATE_MS), None).is_ok());
    }

    #[test]
    fn no_active_window() {
        let t = taker();
        assert!(matches!(
            t.decide(window(), ts(LATE_MS), None),
            Err(NoLateTakeReason::NoActiveWindow)
        ));
    }

    #[test]
    fn no_model_yet() {
        let mut t = taker();
        t.ingest(&open_event());
        assert!(matches!(
            t.decide(window(), ts(LATE_MS), None),
            Err(NoLateTakeReason::NoModel)
        ));
    }

    #[test]
    fn window_mismatch() {
        let mut t = taker();
        t.ingest(&open_event());
        // Poke a snapshot whose window differs into window()'s slot — the
        // defensive WindowMismatch guard (per-window ingest would normally drop a
        // foreign-window model, so we poke it directly).
        if let Event::Model(snap) = model_snapshot(
            0.99,
            ModelHealth::Ready,
            AnchorSource::Chainlink,
            Some(other_window()),
        ) {
            t.test_state_mut(window()).last_model = Some(snap);
        }
        assert!(matches!(
            t.decide(window(), ts(LATE_MS), None),
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
                    t.decide(window(), ts(LATE_MS), None),
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
                    t.decide(window(), ts(LATE_MS), None),
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
        assert!(t.decide(window(), ts(LATE_MS), None).is_ok());
    }

    #[test]
    fn expired_window_blocks() {
        let t = armed_up();
        assert!(matches!(
            t.decide(window(), ts(CLOSE_MS + 1), None),
            Err(NoLateTakeReason::Expired { .. })
        ));
    }

    #[test]
    fn not_late_window_blocks() {
        let t = armed_up();
        // τ = 60 s > 30 s threshold.
        assert!(matches!(
            t.decide(window(), ts(CLOSE_MS - 60_000), None),
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
            t.decide(window(), ts(LATE_MS), None),
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
            t.decide(window(), ts(LATE_MS), None),
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
            t.decide(window(), ts(LATE_MS), None),
            Err(NoLateTakeReason::AllAsksAbovePriceCap { .. })
        ));
    }

    #[test]
    fn cooldown_blocks() {
        let mut t = armed_up();
        t.test_state_mut(window()).last_take_ms = Some(LATE_MS - 100); // fired 100 ms ago < 5 s cooldown
        assert!(matches!(
            t.decide(window(), ts(LATE_MS), None),
            Err(NoLateTakeReason::InCooldown { .. })
        ));
        // After the cooldown elapses, the take is allowed again.
        t.test_state_mut(window()).last_take_ms = Some(LATE_MS - 6_000);
        assert!(t.decide(window(), ts(LATE_MS), None).is_ok());
    }

    /// Required case 4: budget exhaustion — directly, and via in-flight commitment.
    #[test]
    fn budget_exhausted_blocks() {
        let mut t = armed_up();
        t.test_state_mut(window()).realized_spent = Dollars::new(dec!(10));
        assert!(matches!(
            t.decide(window(), ts(LATE_MS), None),
            Err(NoLateTakeReason::BudgetExhausted { .. })
        ));

        // An in-flight take leaving < $1 also blocks a second.
        let mut t2 = armed_up();
        let oid = OrderId::new("lw-1").unwrap();
        t2.test_register_pending(window(), oid, Dollars::new(dec!(9.50)));
        assert!(matches!(
            t2.decide(window(), ts(LATE_MS), None),
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
        let d = t.decide(window(), ts(LATE_MS), None).expect("a take");
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
        // ingest returns None ⇒ on_event would attempt no take.
        assert!(t.ingest(&big_ramp).is_none());
    }

    #[test]
    fn standing_down_blocks_and_clears() {
        let mut t = armed_up();
        t.on_risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::FeedStale,
        });
        assert!(t.is_standing_down());
        assert!(matches!(
            t.decide(window(), ts(LATE_MS), None),
            Err(NoLateTakeReason::StandingDown)
        ));
        t.on_risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::FeedStale,
        });
        assert!(!t.is_standing_down());
        assert!(t.decide(window(), ts(LATE_MS), None).is_ok());
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
    fn window_open_starts_fresh_budget_and_cooldown() {
        let mut t = armed_up();
        {
            let st = t.test_state_mut(window());
            st.realized_spent = Dollars::new(dec!(8));
            st.last_take_ms = Some(OPEN_MS + 100);
            st.our_orders.insert(OrderId::new("lw-9").unwrap());
        }
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
        // The new window's per-window state is fresh.
        let st = t.test_state_mut(next);
        assert_eq!(st.realized_spent, Dollars::ZERO);
        assert_eq!(st.effective_spent(), Dollars::ZERO);
        assert!(st.last_take_ms.is_none());
        assert!(st.our_orders.is_empty());
    }

    // ---- per-window state ---------------------------------------------------

    #[test]
    fn per_window_independence_and_close_prunes() {
        let mut t = taker();
        // Arm two windows of the same series.
        t.ingest(&open_event()); // window() @ OPEN_MS
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
        assert_eq!(t.windows.len(), 2);
        // Register an order on window() so we can prove order_window pruning.
        let oid = OrderId::new("lw-a").unwrap();
        t.test_register_pending(window(), oid.clone(), Dollars::new(dec!(5)));
        assert!(t.order_window.contains_key(&oid));
        // Closing window() drops only its state + its order attribution.
        t.ingest(&Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Closed,
        });
        assert_eq!(t.windows.len(), 1);
        assert!(t.windows.contains_key(&next));
        assert!(
            !t.order_window.contains_key(&oid),
            "order attribution pruned"
        );
    }

    /// A resting maker quote recorded for a *different* window must NOT shield a
    /// take on this window (the self-match filter fetches only this window's view).
    #[test]
    fn a_resting_quote_on_another_window_does_not_shield_a_take() {
        use crate::quote_manager::{ClientId, RestingView};
        use std::collections::HashMap;

        struct MockLookup {
            map: HashMap<WindowId, RestingView>,
        }
        impl RestingLookup for MockLookup {
            fn resting_view_for(&self, window: WindowId) -> Option<&RestingView> {
                self.map.get(&window)
            }
        }

        // Up book @ 0.98×50, certain p_up on window().
        let t = armed_up();
        // A mirror resting BUY Down @ 0.02 (= complement of the 0.98 Up ask) that
        // would fully cover the ask — but recorded on `other_window()`.
        let mut other_view = RestingView::new();
        other_view.record_pending(
            OrderId::new("qm-other").unwrap(),
            ClientId {
                open_ms: other_window().open_time.as_millis(),
                outcome: Outcome::Down,
                level: 0,
                seq: 0,
            },
            other_window(),
            Price::on_grid(dec!(0.02), TICK).unwrap(),
            Size::new(dec!(50)).unwrap(),
        );
        let mut map = HashMap::new();
        map.insert(other_window(), other_view);
        let lookup = MockLookup { map };
        // `resting_view_for(window())` is empty here, so the other window's mirror
        // never shields — the take fires normally.
        let d = t
            .decide(window(), ts(LATE_MS), Some(&lookup))
            .expect("not shielded by another window's quote");
        assert_eq!(d.outcome, Outcome::Up);
    }

    // ---- budget reconciliation from the venue stream (sync) ----------------

    #[test]
    fn fills_charge_budget_and_terminal_drops_residual() {
        let mut t = armed_up();
        let oid = OrderId::new("lw-7").unwrap();
        t.test_register_pending(window(), oid.clone(), Dollars::new(dec!(10)));
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
        t.test_register_pending(window(), OrderId::new("lw-1").unwrap(), Dollars::ZERO);
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
