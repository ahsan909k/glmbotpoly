//! [`MomentumTaker`]: the async, IO-bearing executor for the §8 momentum taker.
//!
//! Mirrors [`QuoteManager`](crate::quote_manager::QuoteManager): it needs **no**
//! `tokio` (it only `.await`s the [`VenuePort`] futures and takes `now:
//! TimestampMs` as a parameter — sans-clock, like the rest of `engine`), folds
//! its own fills **only** from the venue stream, owns the `tripped`/
//! `standing_down` risk veto, and routes every FAK through the
//! [`normalize`](crate::normalize) chokepoint. It logs via `tracing` (target
//! `momentum-taker`, §12).
//!
//! The bot's select-loop owns the timers and the single `VenueEvent` receiver
//! and calls:
//! - [`MomentumTaker::on_event`] for each bus event (price tick → signal ring;
//!   model/book → cache + attempt a take; window lifecycle; risk),
//! - [`MomentumTaker::on_venue_event`] for each item on the venue's order/fill
//!   stream (charges the budget from our own taker fills).
//!
//! ## Ingestion vs decision vs firing
//!
//! State ingestion ([`ingest`](MomentumTaker::ingest)) and the take *decision*
//! ([`decide`](MomentumTaker::decide)) are pure and **synchronous**; only firing
//! (`normalize` + `port.place`) is async. So the entire gate ladder is unit-
//! testable without a runtime, while the async fire/fill path is exercised
//! end-to-end against the real `PaperVenue` in `bot/tests` (the `quote_manager`
//! precedent).
//!
//! ## Budget accounting
//!
//! The per-window budget tracks **committed-in-flight** notional explicitly:
//! `effective_spent = realized_spent + Σ pending`. On a `place` `Accepted` the
//! planned notional is committed to `pending`; each of our `Fill`s moves notional
//! into `realized_spent` and decrements that order's `pending`; the order's
//! terminal `Order` update drops any residual `pending` (a FAK's unfilled
//! remainder was killed, never spent). So a take can never exceed the budget even
//! if a fill lags the next decision — the cooldown is a behavioral throttle, not
//! a correctness crutch (§9 conservatism).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use core_types::{
    Asset, BookSnapshot, BreakerKind, Decimal, Dollars, Event, Liquidity, MarketInfo,
    ModelSnapshot, OrderQty, Outcome, PriceSource, RiskEvent, Side, TickKind, TimeInForce,
    TimestampMs, TokenId, WindowId, WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::edge::{TakePlan, plan_take};
use super::signal::SignalWindow;
use super::{MomentumTakerParams, NoTakeReason};
use crate::arbitration::{FireLedger, TakerId};
use crate::normalize::{NormalizerParams, OrderDraft, normalize};
use crate::quote_manager::RestingView;

/// Seconds remaining to close, clamped at zero — computed inline so the driver
/// needs no `timeutil` dependency (the `quote_manager` precedent).
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
    plan: TakePlan,
}

/// The fee-aware momentum taker (CLAUDE.md §8). Generic placement happens through
/// the [`VenuePort`] passed to each method; the taker stores no port (and so is
/// not itself generic), keeping it trivially shareable across the bot's venue.
pub struct MomentumTaker {
    params: MomentumTakerParams,
    normalizer_params: NormalizerParams,
    /// Per-asset signal rings (span window rolls — price history is continuous).
    signals: HashMap<Asset, SignalWindow>,
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

impl MomentumTaker {
    /// Builds a taker with the given tunables and an empty state.
    #[must_use]
    pub fn new(params: MomentumTakerParams, normalizer_params: NormalizerParams) -> Self {
        Self {
            params,
            normalizer_params,
            signals: HashMap::new(),
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
    /// update (a fresh fair or a fresh book may have opened an opportunity). A
    /// price tick only updates the signal ring — the decision needs a fresh fair.
    pub async fn on_event<P: VenuePort>(
        &mut self,
        event: &Event,
        port: &P,
        now: TimestampMs,
        arbiter: &mut FireLedger,
        resting: Option<&RestingView>,
    ) {
        if self.ingest(event) {
            self.attempt_take(port, now, arbiter, resting).await;
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
                        "the momentum taker only ever fires FAK taker orders"
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
                    tracing::info!(target: "momentum-taker", order = %f.order_id, price = %f.price, size = %f.size, "taker fill");
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
            Event::PriceTick(tick) => {
                // Only the fast leading signal: direct-Binance top-of-book mid.
                if tick.source == PriceSource::BinanceDirect && tick.kind == TickKind::Mid {
                    let lookback = self.params.signal_lookback_ms;
                    self.signals
                        .entry(tick.asset)
                        .or_insert_with(|| SignalWindow::new(lookback))
                        .push(tick.ts_local, tick.value);
                }
                false
            }
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
                tracing::info!(target: "momentum-taker", window = %market.window, "window open: taker armed");
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
                    tracing::debug!(target: "momentum-taker", window = %market.window, ?lifecycle, "window ended: taker stood down");
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
                tracing::warn!(target: "momentum-taker", ?breaker, "risk veto: taker standing down");
            }
            RiskEvent::BreakerCleared { breaker } => {
                self.tripped.remove(&breaker);
                if self.tripped.is_empty() {
                    self.standing_down = false;
                    tracing::info!(target: "momentum-taker", "all breakers cleared: taker resuming");
                }
            }
        }
    }

    // ---- decision (sync, pure) ---------------------------------------------

    /// The full take gate ladder. Each gate maps to a typed [`NoTakeReason`] so
    /// every non-fire is explainable. Pure and synchronous.
    fn decide(
        &self,
        now: TimestampMs,
        arbiter: &FireLedger,
        resting: Option<&RestingView>,
    ) -> Result<Decision, NoTakeReason> {
        if self.standing_down {
            return Err(NoTakeReason::StandingDown);
        }
        let active = self.active.as_ref().ok_or(NoTakeReason::NoActiveWindow)?;
        let model = self.last_model.ok_or(NoTakeReason::NoModel)?;
        if model.window != Some(active.window) {
            return Err(NoTakeReason::WindowMismatch {
                model: model.window,
                active: active.window,
            });
        }
        if !model.health.allows_quoting() {
            return Err(NoTakeReason::ModelNotReady {
                health: model.health,
                reason: model.reason,
            });
        }
        let p_up = model.p_up;
        if !p_up.is_finite() || p_up <= 0.0 || p_up >= 1.0 {
            return Err(NoTakeReason::UnusableFair { p_up });
        }
        let sigma = model.sigma_1s;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(NoTakeReason::UnusableVol { sigma_1s: sigma });
        }
        // Defensive expiry guard (no τ floor: takes are allowed up to resolution).
        let tau = tau_secs(now, active.market.close_time);
        if tau <= 0.0 {
            return Err(NoTakeReason::Expired { tau_secs: tau });
        }
        // A confirmed fast-feed move picks the candidate side.
        let confirmed = self
            .signals
            .get(&active.window.series.asset)
            .and_then(|w| w.confirmed_direction(now, sigma, self.params.signal_sigma_mult))
            .ok_or(NoTakeReason::NoConfirmedMove)?;
        // Cooldown.
        if let Some(last) = self.last_take_ms {
            let elapsed = now.as_millis() - last;
            if elapsed < self.params.cooldown_ms {
                return Err(NoTakeReason::InCooldown {
                    remaining_ms: self.params.cooldown_ms - elapsed,
                });
            }
        }
        // Budget (committed-in-flight aware).
        let remaining = self.params.budget_per_window - self.effective_spent();
        if remaining.as_decimal() < Decimal::ONE {
            return Err(NoTakeReason::BudgetExhausted {
                remaining,
                need: Dollars::new(Decimal::ONE),
            });
        }
        // Arbitration: momentum defers on assets where the model wins (§8).
        if let Err(block) = arbiter.check(TakerId::Momentum, active.window, now) {
            return Err(NoTakeReason::ArbitrationSuppressed {
                winner: block.winner,
                remaining_ms: block.remaining_ms,
            });
        }
        let outcome = confirmed.outcome;
        let token_id = active.market.tokens.get(outcome).clone();
        let book = self
            .books
            .get(&token_id)
            .ok_or(NoTakeReason::NoBookForToken)?;
        let fair = if outcome == Outcome::Up {
            p_up
        } else {
            1.0 - p_up
        };
        let fees = &active.market.fees;
        let fee_rate = if fees.enabled {
            fees.rate
        } else {
            Decimal::ZERO
        };
        // §7 self-match filter: never lift our own mirrored resting liquidity.
        let asks = crate::self_match::filter_asks(outcome, &book.asks, active.window, resting);
        let plan = plan_take(
            outcome,
            fair,
            &asks,
            fee_rate,
            self.params.taker_rebate_pct,
            self.params.momentum_buffer,
            remaining,
        )?;
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
        arbiter: &mut FireLedger,
        resting: Option<&RestingView>,
    ) {
        match self.decide(now, arbiter, resting) {
            Ok(decision) => self.fire(port, decision, now, arbiter).await,
            Err(reason) => {
                tracing::debug!(target: "momentum-taker", reason = %reason, "no take");
            }
        }
    }

    async fn fire<P: VenuePort>(
        &mut self,
        port: &P,
        decision: Decision,
        now: TimestampMs,
        arbiter: &mut FireLedger,
    ) {
        let seq = self.next_seq();
        let open_ms = decision.market.window.open_time.as_millis();
        let draft = OrderDraft {
            client_id: Some(format!("mt:{open_ms}:{seq}")),
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
        // and re-checks the $1 notional (guaranteed by plan_take).
        let order = match normalize(&draft, &decision.market, None, &self.normalizer_params) {
            Ok(n) => n.order,
            Err(e) => {
                tracing::warn!(target: "momentum-taker", reason = %e, "normalize rejected the FAK (unexpected — plan_take guarantees ≥$1 notional + on-grid worst price)");
                return;
            }
        };
        match port.place(&order).await {
            Ok(acc) => {
                self.our_orders.insert(acc.order_id.clone());
                self.pending.insert(acc.order_id, decision.plan.notional);
                self.last_take_ms = Some(now.as_millis());
                self.take_count += 1;
                arbiter.record(TakerId::Momentum, decision.market.window, now);
                tracing::info!(
                    target: "momentum-taker",
                    outcome = %decision.outcome,
                    worst = %decision.plan.worst_price,
                    notional = %decision.plan.notional,
                    shares = %decision.plan.expected_shares,
                    edge = %decision.plan.aggregate_edge,
                    "momentum take (FAK)"
                );
            }
            Err(e) => {
                tracing::warn!(target: "momentum-taker", error = %e, "FAK place failed");
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
        AnchorSource, BookLevel, BookSnapshot, ConditionId, DurationMs, FeeParams, Fill, InputAges,
        Liquidity, ModelHealth, ModelHealthReason, ModelSnapshot, OrderId, OrderState, OrderUpdate,
        Outcome, Price, PriceTick, ResolutionSource, Series, Side, Size, TickSize, TokenId,
        TokenPair, WindowDuration, WindowId, WindowLifecycle,
    };
    use rust_decimal::dec;
    use venue_api::VenueEvent;

    use super::*;

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = OPEN_MS + 300_000;
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
            event_slug: "btc-updown-5m-mt".to_owned(),
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

    fn model_snapshot(p_up: f64, sigma: f64, health: ModelHealth, win: Option<WindowId>) -> Event {
        Event::Model(ModelSnapshot {
            asset: Asset::Btc,
            window: win,
            p_up,
            z: 0.0,
            sigma_1s: sigma,
            sigma_tau: sigma,
            basis: 0.0,
            anchor: AnchorSource::BinanceCorrected,
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

    /// A Ready BTC model on `window()`.
    fn ready_model(p_up: f64, sigma: f64) -> Event {
        model_snapshot(p_up, sigma, ModelHealth::Ready, Some(window()))
    }

    fn tick(value: Decimal, ts: i64) -> Event {
        Event::PriceTick(PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Mid,
            value,
            ts_exchange: TimestampMs::from_millis(ts),
            ts_local: TimestampMs::from_millis(ts),
        })
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

    fn taker() -> MomentumTaker {
        MomentumTaker::new(MomentumTakerParams::default(), NormalizerParams::default())
    }

    /// Ingest a confirmed up-move: 10 ticks 0..900 ms, +$30 each (≈+0.45%).
    fn feed_up_move(t: &mut MomentumTaker) {
        for i in 0..10 {
            t.ingest(&tick(
                dec!(60000) + dec!(30) * Decimal::from(i),
                OPEN_MS + i * 100,
            ));
        }
    }

    /// Ingest a confirmed down-move (mirror).
    fn feed_down_move(t: &mut MomentumTaker) {
        for i in 0..10 {
            t.ingest(&tick(
                dec!(60000) - dec!(30) * Decimal::from(i),
                OPEN_MS + i * 100,
            ));
        }
    }

    /// A fully-armed taker on `window()` with a confirmed up-move and a
    /// stale-cheap Up book.
    fn armed_up() -> MomentumTaker {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.85, 1e-4));
        feed_up_move(&mut t);
        t.ingest(&book(up_token(), &[(dec!(0.80), dec!(50))]));
        t
    }

    // ---- gate ladder (decide is sync) -------------------------------------

    #[test]
    fn happy_path_decides_a_take() {
        let t = armed_up();
        let d = t
            .decide(ts(OPEN_MS + 900), &FireLedger::default(), None)
            .expect("a take");
        assert_eq!(d.outcome, Outcome::Up);
        assert_eq!(d.token_id, up_token());
        assert_eq!(d.plan.worst_price.as_decimal(), dec!(0.80));
        // $10 budget caps the take (10 / 0.80 = 12.5 shares, $10 notional).
        assert_eq!(d.plan.notional, Dollars::new(dec!(10)));
    }

    #[test]
    fn down_move_takes_the_down_side() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.15, 1e-4)); // fair_down = 0.85
        feed_down_move(&mut t);
        t.ingest(&book(down_token(), &[(dec!(0.80), dec!(50))]));
        let d = t
            .decide(ts(OPEN_MS + 900), &FireLedger::default(), None)
            .expect("a down take");
        assert_eq!(d.outcome, Outcome::Down);
        assert_eq!(d.token_id, down_token());
    }

    #[test]
    fn no_active_window() {
        let t = taker();
        assert!(matches!(
            t.decide(ts(OPEN_MS), &FireLedger::default(), None),
            Err(NoTakeReason::NoActiveWindow)
        ));
    }

    #[test]
    fn no_model_yet() {
        let mut t = taker();
        t.ingest(&open_event());
        assert!(matches!(
            t.decide(ts(OPEN_MS), &FireLedger::default(), None),
            Err(NoTakeReason::NoModel)
        ));
    }

    #[test]
    fn window_mismatch() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&model_snapshot(
            0.85,
            1e-4,
            ModelHealth::Ready,
            Some(other_window()),
        ));
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::WindowMismatch { .. })
        ));
    }

    #[test]
    fn model_not_ready_blocks() {
        for health in [ModelHealth::Degraded, ModelHealth::Unreliable] {
            let mut t = taker();
            t.ingest(&open_event());
            t.ingest(&model_snapshot(0.85, 1e-4, health, Some(window())));
            feed_up_move(&mut t);
            t.ingest(&book(up_token(), &[(dec!(0.80), dec!(50))]));
            assert!(
                matches!(
                    t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
                    Err(NoTakeReason::ModelNotReady { .. })
                ),
                "{health:?} should block"
            );
        }
    }

    #[test]
    fn unusable_fair_and_vol_block() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(1.0, 1e-4)); // p_up at the domain edge
        feed_up_move(&mut t);
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::UnusableFair { .. })
        ));

        let mut t2 = taker();
        t2.ingest(&open_event());
        t2.ingest(&ready_model(0.85, 0.0)); // σ unusable
        feed_up_move(&mut t2);
        assert!(matches!(
            t2.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::UnusableVol { .. })
        ));
    }

    #[test]
    fn expired_window_blocks() {
        let t = armed_up();
        // now past close.
        assert!(matches!(
            t.decide(ts(CLOSE_MS + 1), &FireLedger::default(), None),
            Err(NoTakeReason::Expired { .. })
        ));
    }

    #[test]
    fn no_confirmed_move_blocks_even_with_a_standing_edge() {
        // Book is stale-cheap and the model is Ready, but the signal ring is flat
        // — no fresh move, so no take (the staleness alone is not enough).
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.85, 1e-4));
        for i in 0..10 {
            t.ingest(&tick(dec!(60000), OPEN_MS + i * 100)); // flat
        }
        t.ingest(&book(up_token(), &[(dec!(0.80), dec!(50))]));
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::NoConfirmedMove)
        ));
    }

    #[test]
    fn chainlink_or_trade_ticks_do_not_form_a_signal() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.85, 1e-4));
        // A big ramp, but on the wrong source/kind — ignored by the ring.
        for i in 0..10 {
            t.ingest(&Event::PriceTick(PriceTick {
                source: PriceSource::ChainlinkRtds,
                asset: Asset::Btc,
                kind: TickKind::Vendor,
                value: dec!(60000) + dec!(30) * Decimal::from(i),
                ts_exchange: ts(OPEN_MS + i * 100),
                ts_local: ts(OPEN_MS + i * 100),
            }));
        }
        t.ingest(&book(up_token(), &[(dec!(0.80), dec!(50))]));
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::NoConfirmedMove)
        ));
    }

    #[test]
    fn cooldown_blocks() {
        let mut t = armed_up();
        t.last_take_ms = Some(OPEN_MS + 800); // fired 100 ms ago < 5 s cooldown
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::InCooldown { .. })
        ));
        // After the cooldown elapses, the take is allowed again.
        t.last_take_ms = Some(OPEN_MS + 900 - 6_000);
        assert!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None)
                .is_ok()
        );
    }

    #[test]
    fn budget_exhausted_blocks() {
        let mut t = armed_up();
        // Spend the whole $10 budget directly.
        t.realized_spent = Dollars::new(dec!(10));
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::BudgetExhausted { .. })
        ));
    }

    #[test]
    fn no_book_for_token_blocks() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.85, 1e-4));
        feed_up_move(&mut t);
        // No book ingested.
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::NoBookForToken)
        ));
    }

    #[test]
    fn book_already_repriced_blocks() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.85, 1e-4));
        feed_up_move(&mut t);
        t.ingest(&book(up_token(), &[(dec!(0.86), dec!(50))])); // ask ≥ fair
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::BookAlreadyRepriced { .. })
        ));
    }

    #[test]
    fn edge_below_fee_plus_buffer_blocks() {
        let mut t = taker();
        t.ingest(&open_event());
        t.ingest(&ready_model(0.522, 1e-4)); // fair_up 0.522
        feed_up_move(&mut t);
        t.ingest(&book(up_token(), &[(dec!(0.50), dec!(100))])); // edge 0.022 < 0.0225
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::EdgeBelowFeePlusBuffer { .. })
        ));
    }

    #[test]
    fn standing_down_blocks_and_clears() {
        let mut t = armed_up();
        t.on_risk(RiskEvent::BreakerTripped {
            breaker: BreakerKind::FeedStale,
        });
        assert!(t.is_standing_down());
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::StandingDown)
        ));
        t.on_risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::FeedStale,
        });
        assert!(!t.is_standing_down());
        assert!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None)
                .is_ok()
        );
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
        t.our_orders.insert(OrderId::new("fake-1").unwrap());
        // A new window opens.
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
        let oid = OrderId::new("fake-7").unwrap();
        // Simulate a fired take: $10 committed in-flight.
        t.our_orders.insert(oid.clone());
        t.pending.insert(oid.clone(), Dollars::new(dec!(10)));
        assert_eq!(t.effective_spent(), Dollars::new(dec!(10)));

        // A partial taker fill: 5 shares @ 0.80 = $4.
        t.on_venue_event(
            &VenueEvent::Fill(Arc::new(fill(&oid, dec!(0.80), dec!(5)))),
            ts(OPEN_MS),
        );
        assert_eq!(t.realized_spent(), Dollars::new(dec!(4)));
        // effective = realized 4 + pending 6 = 10 (unchanged total).
        assert_eq!(t.effective_spent(), Dollars::new(dec!(10)));

        // The FAK remainder is killed (Canceled) — residual pending dropped.
        t.on_venue_event(
            &VenueEvent::Order(Arc::new(order_update(&oid, OrderState::Canceled))),
            ts(OPEN_MS),
        );
        // Now effective == realized $4 (only what actually filled).
        assert_eq!(t.effective_spent(), Dollars::new(dec!(4)));
        assert_eq!(t.realized_spent(), Dollars::new(dec!(4)));
    }

    #[test]
    fn foreign_fills_are_ignored() {
        let mut t = armed_up();
        let ours = OrderId::new("fake-1").unwrap();
        t.our_orders.insert(ours);
        // A fill for someone else's order (e.g. a quoter maker fill).
        let foreign = OrderId::new("qm-9").unwrap();
        t.on_venue_event(
            &VenueEvent::Fill(Arc::new(fill(&foreign, dec!(0.50), dec!(20)))),
            ts(OPEN_MS),
        );
        assert_eq!(t.realized_spent(), Dollars::ZERO);
    }

    #[test]
    fn an_in_flight_take_blocks_a_second_that_would_exceed_budget() {
        let mut t = armed_up();
        // Commit $9.50 in-flight (not yet filled).
        let oid = OrderId::new("fake-1").unwrap();
        t.our_orders.insert(oid.clone());
        t.pending.insert(oid, Dollars::new(dec!(9.50)));
        // Only $0.50 effective budget left (< $1) ⇒ no second take.
        assert!(matches!(
            t.decide(ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::BudgetExhausted { .. })
        ));
    }

    // ---- fixtures for the venue stream -------------------------------------

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
            price: Price::on_grid(dec!(0.80), TICK).unwrap(),
            original_size: Size::new(dec!(12.5)).unwrap(),
            filled_size: Size::new(dec!(5)).unwrap(),
            reject_reason: None,
            ts_venue: Some(TimestampMs::from_millis(OPEN_MS)),
            ts_local: TimestampMs::from_millis(OPEN_MS),
        }
    }
}
