//! [`QuoteManager`]: the async, IO-bearing executor that owns convergence from
//! the current open orders to the calculator's desired set, through a
//! [`venue_api::VenuePort`].
//!
//! This is the one quote-manager module that crosses into IO (it `.await`s the
//! port) and logs via `tracing` (target `quote-manager`, §12). It needs **no**
//! `tokio`: it only awaits the port's futures and takes `now: TimestampMs` as a
//! parameter (sans-clock, like the rest of `engine`). The bot's select-loop owns
//! the actual timers and the single `VenueEvent` receiver, and calls:
//! - [`QuoteManager::on_event`] for each bus event (market data, model, window
//!   lifecycle, risk),
//! - [`QuoteManager::on_venue_event`] for each item on the venue's order/fill
//!   stream (folds the resting view + our own fills into inventory), and
//! - [`QuoteManager::on_requote_tick`] periodically (the debounced converge).
//!
//! ## Fill / inventory source
//!
//! The manager's own fills (and therefore its inventory) come **only** from the
//! venue stream ([`QuoteManager::on_venue_event`]) — the authoritative report of
//! what *we* executed. Bus `Event::Fill`/`Event::OrderUpdate` are for *other*
//! consumers (journal, dashboard, a global inventory ticker); [`on_event`]
//! ignores them, so a fill is never double-counted.
//!
//! [`on_event`]: QuoteManager::on_event

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use core_types::{
    BookSnapshot, BreakerKind, Decimal, Event, InventorySnapshot, MarketInfo, ModelSnapshot,
    OrderQty, Outcome, RiskEvent, Side, SideInventory, Size, TimestampMs, TokenId, TopOfBook,
    WindowId, WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::QuoteManagerParams;
use super::client_id::ClientId;
use super::plan::{CancelAction, ConvergeRationale, ConvergencePlan, ConvergencePlanner};
use super::view::RestingView;
use crate::inventory::InventoryManager;
use crate::normalize::{NormalizerParams, OrderDraft, normalize};
use crate::quoting::{QuoteParams, calculate_quotes};

/// Seconds remaining to close, clamped at zero — computed inline so the driver
/// needs no `timeutil` dependency.
fn tau_secs(now: TimestampMs, close: TimestampMs) -> f64 {
    let ms = close.as_millis() - now.as_millis();
    if ms <= 0 { 0.0 } else { ms as f64 / 1000.0 }
}

/// Per-active-window manager state.
struct ActiveWindow {
    window: WindowId,
    market: Arc<MarketInfo>,
    view: RestingView,
    /// The model fair our currently-resting set was last built from (updated only
    /// when a fresh set is actually placed, so it reflects the resting orders).
    p_up_at_last_quote: Option<f64>,
    /// Latest model snapshot for this window.
    last_model: Option<ModelSnapshot>,
    /// Wall time (ms) of the last placement cycle — the rate-budget anchor.
    last_place_ms: Option<i64>,
    /// A recompute is pending (a trigger arrived since the last converge).
    dirty: bool,
    /// The window has entered its Closing phase — no further placements.
    closing: bool,
}

impl ActiveWindow {
    fn new(market: Arc<MarketInfo>) -> Self {
        Self {
            window: market.window,
            market,
            view: RestingView::new(),
            p_up_at_last_quote: None,
            last_model: None,
            last_place_ms: None,
            dirty: false,
            closing: false,
        }
    }
}

/// The cancel-first quote manager (CLAUDE.md §8). Generic placement happens
/// through the [`VenuePort`] passed to each method; the manager stores no port
/// (and so is not itself generic), which keeps it trivially shareable across the
/// bot's single venue.
pub struct QuoteManager {
    qm_params: QuoteManagerParams,
    quote_params: QuoteParams,
    normalizer_params: NormalizerParams,
    /// Owns the per-window inventory book, folded from our own venue fills —
    /// self-contained for quoting decisions (the alternative, consuming a global
    /// `Event::Inventory`, would couple correctness to bus ordering).
    inv_mgr: InventoryManager,
    active: Option<ActiveWindow>,
    /// True while a risk breaker holds us down (no placements until cleared).
    standing_down: bool,
    /// The set of currently-tripped breakers (stand down until empty).
    tripped: HashSet<BreakerKind>,
    /// Latest top-of-book per token, for the normalizer's post-only cross check
    /// and the "book changed at our levels" trigger.
    tops: HashMap<TokenId, TopOfBook>,
    /// Monotonic placement sequence — every order gets a unique `client_id`.
    seq: u64,
    /// Count of placement cycles (a `place_batch` that placed ≥1 order). Exposed
    /// for the rate-budget integration test.
    place_cycles: u64,
}

impl QuoteManager {
    /// Builds a manager with the given tunables and an empty inventory book.
    #[must_use]
    pub fn new(
        qm_params: QuoteManagerParams,
        quote_params: QuoteParams,
        normalizer_params: NormalizerParams,
    ) -> Self {
        Self {
            qm_params,
            quote_params,
            normalizer_params,
            inv_mgr: InventoryManager::new(),
            active: None,
            standing_down: false,
            tripped: HashSet::new(),
            tops: HashMap::new(),
            seq: 0,
            place_cycles: 0,
        }
    }

    /// The resting view for the active window, if one is open (inspection).
    #[must_use]
    pub fn resting_view(&self) -> Option<&RestingView> {
        self.active.as_ref().map(|a| &a.view)
    }

    /// Number of placement cycles run so far (each is one `place_batch` that
    /// placed at least one order). The rate-budget metric.
    #[must_use]
    pub fn place_cycle_count(&self) -> u64 {
        self.place_cycles
    }

    /// Whether a risk breaker currently holds the manager down.
    #[must_use]
    pub fn is_standing_down(&self) -> bool {
        self.standing_down
    }

    /// Handles one bus event: market data, model, window lifecycle, or risk.
    /// `Event::Fill`/`Event::OrderUpdate` are deliberately ignored — the manager
    /// folds its own fills from the venue stream ([`Self::on_venue_event`]).
    pub async fn on_event<P: VenuePort>(&mut self, event: &Event, port: &P, now: TimestampMs) {
        match event {
            Event::Window { market, lifecycle } => self.on_window(market, *lifecycle, port).await,
            Event::Model(snap) => self.on_model(snap, port).await,
            Event::ModelHealth(_) => self.mark_dirty(),
            Event::TopOfBook { token_id, top } => self.on_top(token_id, *top),
            Event::Book(snap) => self.on_book(snap),
            Event::Risk(risk) => self.on_risk(*risk, port).await,
            _ => {}
        }
        let _ = now;
    }

    /// Folds one item from the venue's order/fill stream: order updates into the
    /// resting view (freeing slots on terminals), our own fills into inventory.
    pub fn on_venue_event(&mut self, ve: &VenueEvent, _now: TimestampMs) {
        match ve {
            VenueEvent::Order(u) => {
                if let Some(a) = self.active.as_mut() {
                    a.view.apply_order_update(u);
                }
            }
            VenueEvent::Fill(f) => {
                let _ = self.inv_mgr.on_event(&Event::Fill(Arc::clone(f)));
                self.mark_dirty();
            }
        }
    }

    /// The debounced converge: if a recompute is pending and the rate budget
    /// allows, recompute the desired set and converge the venue to it.
    pub async fn on_requote_tick<P: VenuePort>(&mut self, port: &P, now: TimestampMs) {
        if self.standing_down {
            return;
        }
        let planned = {
            let Some(a) = self.active.as_ref() else {
                return;
            };
            if a.closing || !a.dirty {
                return;
            }
            // Rate budget: throttle placements (cancels already ran on the trigger).
            if let Some(last) = a.last_place_ms
                && now.as_millis() - last < self.qm_params.min_requote_interval_ms
            {
                return; // defer; keep dirty.
            }
            let Some(model) = a.last_model else {
                return;
            };
            let market = Arc::clone(&a.market);
            let inv = self.inv_mgr.book(a.window).map_or_else(
                || {
                    InventorySnapshot::derive(
                        a.window,
                        SideInventory::default(),
                        SideInventory::default(),
                        now,
                    )
                },
                |b| b.snapshot(now),
            );
            let tau = tau_secs(now, market.close_time);
            let decision = calculate_quotes(
                &model,
                &inv,
                &market.fees,
                &self.quote_params,
                tau,
                market.tick_size,
            );
            let plan = ConvergencePlanner::plan(
                &decision,
                &a.view,
                &market,
                model.p_up,
                a.p_up_at_last_quote,
                self.qm_params.reprice_threshold_theta,
                self.qm_params.cancel_market_theta,
            );
            Some((plan, market, a.window, model.p_up))
        };
        if let Some((plan, market, window, p_up)) = planned {
            self.execute_plan(port, plan, &market, window, p_up, now)
                .await;
        }
    }

    // ---- bus handlers ------------------------------------------------------

    async fn on_window<P: VenuePort>(
        &mut self,
        market: &Arc<MarketInfo>,
        lifecycle: WindowLifecycle,
        port: &P,
    ) {
        match lifecycle {
            WindowLifecycle::Open => {
                self.active = Some(ActiveWindow::new(Arc::clone(market)));
                tracing::info!(target: "quote-manager", window = %market.window, "window open: ready to quote");
            }
            WindowLifecycle::Closing => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| a.window == market.window)
                {
                    if let Some(a) = self.active.as_mut() {
                        a.closing = true;
                    }
                    let report = port.cancel_all().await;
                    self.mark_all_pending_cancel();
                    tracing::info!(target: "quote-manager", window = %market.window, ok = report.is_ok(), "window closing: cancel-all");
                }
            }
            WindowLifecycle::Closed => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| a.window == market.window)
                {
                    self.active = None;
                }
            }
            WindowLifecycle::Resolved { .. } => {
                let _ = self.inv_mgr.on_event(&Event::Window {
                    market: Arc::clone(market),
                    lifecycle,
                });
                if self
                    .active
                    .as_ref()
                    .is_some_and(|a| a.window == market.window)
                {
                    self.active = None;
                }
            }
            WindowLifecycle::Discovered => {}
        }
    }

    async fn on_model<P: VenuePort>(&mut self, snap: &ModelSnapshot, port: &P) {
        // Store the snapshot + mark dirty even while standing down, so a recovery
        // requotes off the latest model.
        {
            let Some(a) = self.active.as_mut() else {
                return;
            };
            if snap.window != Some(a.window) {
                return;
            }
            a.last_model = Some(*snap);
            a.dirty = true;
        }
        if self.standing_down {
            return;
        }
        // Rule B: urgent adverse-move cancel — bypasses the rate budget (a cancel
        // is always safe). The replacement is placed by the next eligible tick.
        if let Some(action) = self.urgent_cancel_action(snap.p_up) {
            self.execute_cancels(port, &action).await;
        }
    }

    async fn on_risk<P: VenuePort>(&mut self, risk: RiskEvent, port: &P) {
        match risk {
            RiskEvent::BreakerTripped { breaker }
            | RiskEvent::CancelAllIssued { reason: breaker } => {
                self.tripped.insert(breaker);
                self.standing_down = true;
                let report = port.cancel_all().await;
                self.mark_all_pending_cancel();
                tracing::warn!(target: "quote-manager", ?breaker, ok = report.is_ok(), "risk veto: cancel-all + stand down");
            }
            RiskEvent::BreakerCleared { breaker } => {
                self.tripped.remove(&breaker);
                if self.tripped.is_empty() {
                    self.standing_down = false;
                    if let Some(a) = self.active.as_mut() {
                        a.dirty = true;
                    }
                    tracing::info!(target: "quote-manager", "all breakers cleared: resuming");
                }
            }
        }
    }

    fn on_top(&mut self, token_id: &TokenId, top: TopOfBook) {
        self.tops.insert(token_id.clone(), top);
        if self.book_touches_our_levels(token_id, &top) {
            self.mark_dirty();
        }
    }

    fn on_book(&mut self, snap: &BookSnapshot) {
        let top = snap.top();
        self.tops.insert(snap.token_id.clone(), top);
        if self.book_touches_our_levels(&snap.token_id, &top) {
            self.mark_dirty();
        }
    }

    /// Cheap coalescing filter: a book change only dirties the manager if its top
    /// is at or through one of our resting bid prices on that token (cross risk or
    /// our level now at the touch). Deep-book churn far from our quotes is ignored.
    fn book_touches_our_levels(&self, token_id: &TokenId, top: &TopOfBook) -> bool {
        let Some(a) = self.active.as_ref() else {
            return false;
        };
        let our: Vec<Decimal> = a
            .view
            .live_orders()
            .filter(|o| a.market.tokens.get(o.outcome) == token_id)
            .map(|o| o.price.as_decimal())
            .collect();
        if our.is_empty() {
            return false;
        }
        if let Some(ask) = top.ask {
            let ask = ask.price.as_decimal();
            if our.iter().any(|p| ask <= *p) {
                return true;
            }
        }
        if let Some(bid) = top.bid {
            let bid = bid.price.as_decimal();
            if our.iter().any(|p| bid >= *p) {
                return true;
            }
        }
        false
    }

    // ---- convergence -------------------------------------------------------

    /// Issues cancels, marking each affected order pending-cancel in the view so a
    /// racing trigger does not re-cancel it.
    async fn execute_cancels<P: VenuePort>(&mut self, port: &P, action: &CancelAction) {
        match action {
            CancelAction::None => {}
            CancelAction::Orders(ids) => {
                for id in ids {
                    let report = port.cancel(id).await;
                    if let Some(a) = self.active.as_mut() {
                        a.view.mark_pending_cancel(id);
                    }
                    tracing::info!(target: "quote-manager", order = %id, ok = report.is_ok(), "cancel (cancel-first)");
                }
            }
            CancelAction::Market(cid) => {
                let report = port.cancel_market(cid).await;
                self.mark_all_pending_cancel();
                tracing::info!(target: "quote-manager", market = %cid, ok = report.is_ok(), "cancel-market (large move)");
            }
            CancelAction::All => {
                let report = port.cancel_all().await;
                self.mark_all_pending_cancel();
                tracing::info!(target: "quote-manager", ok = report.is_ok(), "cancel-all");
            }
        }
    }

    /// Executes a plan: cancels first (awaited), then places the desired levels
    /// (into empty slots) as one batch through the normalizer.
    async fn execute_plan<P: VenuePort>(
        &mut self,
        port: &P,
        plan: ConvergencePlan,
        market: &MarketInfo,
        window: WindowId,
        p_up: f64,
        now: TimestampMs,
    ) {
        let converged = matches!(plan.rationale, ConvergeRationale::AlreadyConverged);

        // 1. Cancels first — awaited before any placement (split-cycle).
        self.execute_cancels(port, &plan.cancels).await;

        // 2. Build the batch: every level through the normalizer chokepoint.
        let mut orders = Vec::with_capacity(plan.places.len());
        let mut cids: Vec<ClientId> = Vec::with_capacity(plan.places.len());
        for d in &plan.places {
            if orders.len() >= self.qm_params.max_batch {
                break;
            }
            let cid = ClientId {
                open_ms: window.open_time.as_millis(),
                outcome: d.outcome,
                level: d.level,
                seq: self.next_seq(),
            };
            let draft = OrderDraft {
                client_id: Some(cid.encode()),
                window,
                token_id: d.token_id.clone(),
                outcome: d.outcome,
                side: Side::Buy,
                price: d.price.as_decimal(),
                qty: OrderQty::Shares(d.size),
                tif: d.tif,
            };
            let top = self.tops.get(&d.token_id);
            match normalize(&draft, market, top, &self.normalizer_params) {
                Ok(n) => {
                    orders.push(n.order);
                    cids.push(cid);
                }
                Err(e) => {
                    tracing::warn!(target: "quote-manager", token = %d.token_id, reason = %e, "normalize rejected a desired level (unexpected — calculator guarantees on-grid/min)");
                }
            }
        }

        // 3. Place as one batch (the port chunks at its 15-order limit internally).
        let mut placed_any = false;
        if !orders.is_empty() {
            match port.place_batch(&orders).await {
                Ok(batch) => {
                    for (res, (order, cid)) in batch
                        .results
                        .into_iter()
                        .zip(orders.iter().zip(cids.iter()))
                    {
                        match res {
                            Ok(acc) => {
                                placed_any = true;
                                let size = match order.qty {
                                    OrderQty::Shares(s) => s,
                                    OrderQty::Notional(_) => Size::ZERO,
                                };
                                if let Some(a) = self.active.as_mut()
                                    && a.window == window
                                {
                                    a.view.record_pending(
                                        acc.order_id,
                                        *cid,
                                        window,
                                        order.price,
                                        size,
                                    );
                                }
                                tracing::info!(target: "quote-manager", outcome = %cid.outcome, level = cid.level, price = %order.price, "placed");
                            }
                            Err(rej) => {
                                tracing::warn!(target: "quote-manager", reason = ?rej.reason, raw = %rej.raw, "place rejected");
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "quote-manager", error = %e, "place_batch failed");
                }
            }
        }

        // 4. Budget + dirty bookkeeping. A successful placement charges the rate
        //    budget but keeps `dirty` (a stale slot's replacement may still be
        //    owed once it frees); only a genuine AlreadyConverged clears it.
        if placed_any {
            self.place_cycles += 1;
        }
        if let Some(a) = self.active.as_mut()
            && a.window == window
        {
            if placed_any {
                a.last_place_ms = Some(now.as_millis());
                a.p_up_at_last_quote = Some(p_up);
            }
            if converged {
                a.dirty = false;
            }
        }
    }

    // ---- helpers -----------------------------------------------------------

    fn mark_dirty(&mut self) {
        if let Some(a) = self.active.as_mut() {
            a.dirty = true;
        }
    }

    fn mark_all_pending_cancel(&mut self) {
        if let Some(a) = self.active.as_mut() {
            for id in a.view.live_ids() {
                a.view.mark_pending_cancel(&id);
            }
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq = self.seq.wrapping_add(1);
        s
    }

    /// The urgent adverse-move cancel (Rule B): on a model fair that moved more
    /// than θ against a resting quote, cancel the *endangered* side immediately
    /// (Up-buys when fair fell, Down-buys when fair rose); escalate to a bulk
    /// market cancel on a move larger than `cancel_market_theta`.
    fn urgent_cancel_action(&self, p_up_now: f64) -> Option<CancelAction> {
        let a = self.active.as_ref()?;
        if a.closing || !a.view.has_live_cancelable() {
            return None;
        }
        let p0 = a.p_up_at_last_quote?;
        let delta = p_up_now - p0;
        let mag = delta.abs();
        if mag <= self.qm_params.reprice_threshold_theta {
            return None;
        }
        if mag > self.qm_params.cancel_market_theta {
            return Some(CancelAction::Market(a.market.condition_id.clone()));
        }
        let endangered = if delta < 0.0 {
            Outcome::Up
        } else {
            Outcome::Down
        };
        let mut ids: Vec<(Outcome, u32, u64, core_types::OrderId)> = a
            .view
            .live_orders()
            .filter(|o| o.outcome == endangered && !o.is_pending_cancel())
            .map(|o| (o.outcome, o.level, o.client_id.seq, o.order_id.clone()))
            .collect();
        if ids.is_empty() {
            return None;
        }
        ids.sort();
        Some(CancelAction::Orders(
            ids.into_iter().map(|(.., id)| id).collect(),
        ))
    }
}
