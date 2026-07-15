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
    ModelSnapshot, OrderId, OrderQty, Outcome, PriceSource, RiskEvent, Side, TickKind, TimeInForce,
    TimestampMs, TokenId, WindowId, WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::edge::{TakePlan, plan_take};
use super::signal::SignalWindow;
use super::{MomentumTakerParams, NoTakeReason};
use crate::arbitration::{FireLedger, TakerId};
use crate::normalize::{NormalizerParams, OrderDraft, normalize, split_take_clips};
use crate::quote_manager::RestingLookup;

/// Seconds remaining to close, clamped at zero — computed inline so the driver
/// needs no `timeutil` dependency (the `quote_manager` precedent).
fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.as_millis() - now.as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

/// Per-window momentum-taker state: the market, the cached books, the latest
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
    plan: TakePlan,
}

/// The fee-aware momentum taker (CLAUDE.md §8). Generic placement happens through
/// the [`VenuePort`] passed to each method; the taker stores no port (and so is
/// not itself generic), keeping it trivially shareable across the bot's venue.
pub struct MomentumTaker {
    params: MomentumTakerParams,
    normalizer_params: NormalizerParams,
    /// Per-asset signal rings (span window rolls — price history is continuous).
    /// Shared across all windows of an asset (the fast-feed signal is per-asset).
    signals: HashMap<Asset, SignalWindow>,
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

impl MomentumTaker {
    /// Builds a taker with the given tunables and an empty state.
    #[must_use]
    pub fn new(params: MomentumTakerParams, normalizer_params: NormalizerParams) -> Self {
        Self {
            params,
            normalizer_params,
            signals: HashMap::new(),
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
    /// or active-token book update (a fresh fair or a fresh book may have opened an
    /// opportunity). A price tick only updates the signal ring — the decision needs
    /// a fresh fair.
    pub async fn on_event<P: VenuePort>(
        &mut self,
        event: &Event,
        port: &P,
        now: TimestampMs,
        arbiter: &mut FireLedger,
        resting: Option<&dyn RestingLookup>,
    ) {
        if let Some(window) = self.ingest(event) {
            self.attempt_take(window, port, now, arbiter, resting).await;
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
                        "the momentum taker only ever fires FAK taker orders"
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
                    tracing::info!(target: "momentum-taker", order = %f.order_id, price = %f.price, size = %f.size, "taker fill");
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
            Event::PriceTick(tick) => {
                // Only the fast leading signal: direct-Binance top-of-book mid.
                if tick.source == PriceSource::BinanceDirect && tick.kind == TickKind::Mid {
                    let lookback = self.params.signal_lookback_ms;
                    self.signals
                        .entry(tick.asset)
                        .or_insert_with(|| SignalWindow::new(lookback))
                        .push(tick.ts_local, tick.value);
                }
                None
            }
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
                tracing::info!(target: "momentum-taker", window = %market.window, "window open: taker armed");
            }
            WindowLifecycle::Closing
            | WindowLifecycle::Closed
            | WindowLifecycle::Resolved { .. } => {
                let w = market.window;
                if self.windows.remove(&w).is_some() {
                    self.order_window.retain(|_, ow| *ow != w);
                    tracing::debug!(target: "momentum-taker", window = %w, ?lifecycle, "window ended: taker stood down");
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
        window: WindowId,
        now: TimestampMs,
        arbiter: &FireLedger,
        resting: Option<&dyn RestingLookup>,
    ) -> Result<Decision, NoTakeReason> {
        if self.standing_down {
            return Err(NoTakeReason::StandingDown);
        }
        let st = self
            .windows
            .get(&window)
            .ok_or(NoTakeReason::NoActiveWindow)?;
        let model = st.last_model.ok_or(NoTakeReason::NoModel)?;
        // Defensive: the model is stored in its own window's slot, so this always
        // holds — kept as a belt-and-suspenders (and for the typed reason).
        if model.window != Some(window) {
            return Err(NoTakeReason::WindowMismatch {
                model: model.window,
                active: window,
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
        let tau = tau_secs(now, st.market.close_time);
        if tau <= 0.0 {
            return Err(NoTakeReason::Expired { tau_secs: tau });
        }
        // A confirmed fast-feed move (shared per-asset signal) picks the side.
        let confirmed = self
            .signals
            .get(&window.series.asset)
            .and_then(|w| w.confirmed_direction(now, sigma, self.params.signal_sigma_mult))
            .ok_or(NoTakeReason::NoConfirmedMove)?;
        // Cooldown.
        if let Some(last) = st.last_take_ms {
            let elapsed = now.as_millis() - last;
            if elapsed < self.params.cooldown_ms {
                return Err(NoTakeReason::InCooldown {
                    remaining_ms: self.params.cooldown_ms - elapsed,
                });
            }
        }
        // Budget (committed-in-flight aware, per window).
        let remaining = self.params.budget_per_window - st.effective_spent();
        if remaining.as_decimal() < Decimal::ONE {
            return Err(NoTakeReason::BudgetExhausted {
                remaining,
                need: Dollars::new(Decimal::ONE),
            });
        }
        // Arbitration: momentum defers on assets where the model wins (§8).
        if let Err(block) = arbiter.check(TakerId::Momentum, window, now) {
            return Err(NoTakeReason::ArbitrationSuppressed {
                winner: block.winner,
                remaining_ms: block.remaining_ms,
            });
        }
        let outcome = confirmed.outcome;
        let token_id = st.market.tokens.get(outcome).clone();
        let book = st
            .books
            .get(&token_id)
            .ok_or(NoTakeReason::NoBookForToken)?;
        let fair = if outcome == Outcome::Up {
            p_up
        } else {
            1.0 - p_up
        };
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
        arbiter: &mut FireLedger,
        resting: Option<&dyn RestingLookup>,
    ) {
        match self.decide(window, now, arbiter, resting) {
            Ok(decision) => self.fire(window, port, decision, now, arbiter).await,
            Err(reason) => {
                tracing::debug!(target: "momentum-taker", reason = %reason, "no take");
            }
        }
    }

    async fn fire<P: VenuePort>(
        &mut self,
        window: WindowId,
        port: &P,
        decision: Decision,
        now: TimestampMs,
        arbiter: &mut FireLedger,
    ) {
        let open_ms = decision.market.window.open_time.as_millis();
        // Split the planned take into sequential FAK clips of ≤ clip_size_shares
        // (§8 burst pattern). With no clip configured this is one full-size FAK
        // (byte-identical to the pre-clip behavior).
        let clips = split_take_clips(
            decision.plan.expected_shares,
            decision.plan.worst_price,
            decision.plan.notional,
            self.normalizer_params.clip_size_shares,
        );
        let mut any_fired = false;
        for clip_notional in clips {
            let seq = self.next_seq();
            let draft = OrderDraft {
                client_id: Some(format!("mt:{open_ms}:{seq}")),
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
            // and re-checks the $1 notional (guaranteed by plan_take / clip split).
            let order = match normalize(&draft, &decision.market, None, &self.normalizer_params) {
                Ok(n) => n.order,
                Err(e) => {
                    tracing::warn!(target: "momentum-taker", reason = %e, "normalize rejected a FAK clip (skipping this clip)");
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
                    any_fired = true;
                    tracing::info!(
                        target: "momentum-taker",
                        outcome = %decision.outcome,
                        worst = %decision.plan.worst_price,
                        notional = %clip_notional,
                        "momentum take (FAK clip)"
                    );
                }
                Err(e) => {
                    tracing::warn!(target: "momentum-taker", error = %e, "FAK place failed");
                }
            }
        }
        // One arbitration record per fire (not per clip): the momentum taker owns
        // this window for the arbitration span once it has fired at all.
        if any_fired {
            arbiter.record(TakerId::Momentum, window, now);
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq = self.seq.wrapping_add(1);
        s
    }
}

#[cfg(test)]
impl MomentumTaker {
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
            .decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None)
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
            .decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None)
            .expect("a down take");
        assert_eq!(d.outcome, Outcome::Down);
        assert_eq!(d.token_id, down_token());
    }

    #[test]
    fn no_active_window() {
        let t = taker();
        assert!(matches!(
            t.decide(window(), ts(OPEN_MS), &FireLedger::default(), None),
            Err(NoTakeReason::NoActiveWindow)
        ));
    }

    #[test]
    fn no_model_yet() {
        let mut t = taker();
        t.ingest(&open_event());
        assert!(matches!(
            t.decide(window(), ts(OPEN_MS), &FireLedger::default(), None),
            Err(NoTakeReason::NoModel)
        ));
    }

    #[test]
    fn window_mismatch() {
        let mut t = taker();
        t.ingest(&open_event());
        // Poke a snapshot whose window differs into window()'s slot — the
        // defensive WindowMismatch guard (per-window ingest would normally drop a
        // foreign-window model, so we poke it directly).
        if let Event::Model(snap) =
            model_snapshot(0.85, 1e-4, ModelHealth::Ready, Some(other_window()))
        {
            t.test_state_mut(window()).last_model = Some(snap);
        }
        assert!(matches!(
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
                    t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::UnusableFair { .. })
        ));

        let mut t2 = taker();
        t2.ingest(&open_event());
        t2.ingest(&ready_model(0.85, 0.0)); // σ unusable
        feed_up_move(&mut t2);
        assert!(matches!(
            t2.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::UnusableVol { .. })
        ));
    }

    #[test]
    fn expired_window_blocks() {
        let t = armed_up();
        // now past close.
        assert!(matches!(
            t.decide(window(), ts(CLOSE_MS + 1), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::NoConfirmedMove)
        ));
    }

    #[test]
    fn cooldown_blocks() {
        let mut t = armed_up();
        t.test_state_mut(window()).last_take_ms = Some(OPEN_MS + 800); // fired 100 ms ago < 5 s cooldown
        assert!(matches!(
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::InCooldown { .. })
        ));
        // After the cooldown elapses, the take is allowed again.
        t.test_state_mut(window()).last_take_ms = Some(OPEN_MS + 900 - 6_000);
        assert!(
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None)
                .is_ok()
        );
    }

    #[test]
    fn budget_exhausted_blocks() {
        let mut t = armed_up();
        // Spend the whole $10 budget directly.
        t.test_state_mut(window()).realized_spent = Dollars::new(dec!(10));
        assert!(matches!(
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
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
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::StandingDown)
        ));
        t.on_risk(RiskEvent::BreakerCleared {
            breaker: BreakerKind::FeedStale,
        });
        assert!(!t.is_standing_down());
        assert!(
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None)
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
    fn window_open_starts_fresh_budget_and_cooldown() {
        let mut t = armed_up();
        // Poke spend/cooldown/orders onto window().
        {
            let st = t.test_state_mut(window());
            st.realized_spent = Dollars::new(dec!(8));
            st.last_take_ms = Some(OPEN_MS + 100);
            st.our_orders.insert(OrderId::new("fake-1").unwrap());
        }
        // A new window opens with its own fresh per-window state.
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

    // ---- budget reconciliation from the venue stream (sync) ----------------

    #[test]
    fn fills_charge_budget_and_terminal_drops_residual() {
        let mut t = armed_up();
        let oid = OrderId::new("fake-7").unwrap();
        // Simulate a fired take: $10 committed in-flight.
        t.test_register_pending(window(), oid.clone(), Dollars::new(dec!(10)));
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
        t.test_register_pending(window(), ours, Dollars::ZERO);
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
        t.test_register_pending(window(), oid, Dollars::new(dec!(9.50)));
        // Only $0.50 effective budget left (< $1) ⇒ no second take.
        assert!(matches!(
            t.decide(window(), ts(OPEN_MS + 900), &FireLedger::default(), None),
            Err(NoTakeReason::BudgetExhausted { .. })
        ));
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
        let oid = OrderId::new("mt-a").unwrap();
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

        let t = armed_up(); // Up book @ 0.80×50, confirmed up-move on window()
        // A mirror resting BUY Down @ 0.20 (= complement of the 0.80 Up ask) that
        // would FULLY cover the ask — but recorded on `other_window()`.
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
            Price::on_grid(dec!(0.20), TICK).unwrap(),
            Size::new(dec!(50)).unwrap(),
        );
        let mut map = HashMap::new();
        map.insert(other_window(), other_view);
        let lookup = MockLookup { map };
        // `resting_view_for(window())` is empty here, so the other window's mirror
        // never shields — the take fires normally.
        let d = t
            .decide(
                window(),
                ts(OPEN_MS + 900),
                &FireLedger::default(),
                Some(&lookup),
            )
            .expect("not shielded by another window's quote");
        assert_eq!(d.outcome, Outcome::Up);
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
