//! [`RiskManager`]: the single public order-path driver (CLAUDE.md §5/§11).
//!
//! It owns the [`RiskCore`] breaker machine, the shared [`GateState`], and the
//! three strategy modules — which it drives **only** through a freshly-built
//! [`GuardedPort`], so no strategy can reach the venue except through a risk
//! check. The orchestrator calls [`on_event`](RiskManager::on_event),
//! [`on_venue_event`](RiskManager::on_venue_event) and
//! [`on_tick`](RiskManager::on_tick); after each it drains
//! [`take_published`](RiskManager::take_published) and forwards those breaker
//! events to the bus (for the dashboard and journal).
//!
//! Each turn: drain the guard's venue-error observations → fold them and the new
//! input into the core → apply the core's [`RiskOutput`]s (publish breaker events,
//! issue the authoritative cancel-all/cancel-market through the **raw** port,
//! project the gate) → forward the breaker events and the original event to the
//! owned strategies through the guard. The authoritative cancel-all guarantees
//! the §11 evacuation independently of any strategy being enabled or healthy; the
//! strategy-forwarded breaker event lets each strategy stand down and reconcile
//! its own view — an idempotent, intentional belt-and-suspenders.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use core_types::{
    Dollars, Event, OrderId, RiskEvent, Series, TimestampMs, WindowId, WindowLifecycle,
};
use venue_api::{VenueEvent, VenuePort};

use super::core::{RiskCore, RiskOutput};
use super::guard::GuardedPort;
use super::params::RiskParams;
use super::state::{GateState, GuardObservation, RiskStateSnapshot};
use crate::arbitration::FireLedger;
use crate::late_window::LateWindowTaker;
use crate::model_taker::{ModelPrediction, ModelTakeOutcome, ModelTaker};
use crate::quote_manager::{QuoteManager, RestingLookup, RestingView};
use crate::taker::MomentumTaker;

/// Which owned strategy placed an order — the driver a fill is attributed to
/// (CLAUDE.md §10 PnL-by-driver). Resolved from the strategies' own order sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillDriver {
    /// The maker-core quote manager (`qm:` orders).
    MakerCore,
    /// The momentum taker (`mt:` orders).
    Momentum,
    /// The late-window taker (`lw:` orders).
    Late,
    /// The champion-model taker (`md:` orders).
    Model,
}

/// The single gateway between every strategy and the venue port.
pub struct RiskManager {
    params: RiskParams,
    core: RiskCore,
    shared: Arc<Mutex<GateState>>,
    quoter: QuoteManager,
    momentum: MomentumTaker,
    late: LateWindowTaker,
    model: ModelTaker,
    /// The cross-taker fire ledger (momentum vs model arbitration, §8).
    arbiter: FireLedger,
    /// Breaker events to publish on the bus, drained by the orchestrator.
    published: Vec<Event>,
}

fn log_risk(re: &RiskEvent) {
    match re {
        RiskEvent::BreakerTripped { breaker } => {
            tracing::warn!(target: "risk", ?breaker, "breaker tripped");
        }
        RiskEvent::BreakerCleared { breaker } => {
            tracing::info!(target: "risk", ?breaker, "breaker cleared");
        }
        RiskEvent::CancelAllIssued { reason } => {
            tracing::warn!(target: "risk", ?reason, "cancel-all issued");
        }
    }
}

impl RiskManager {
    /// Builds the gateway from the risk tunables and the per-series worst-case
    /// loss caps (`window.series → max_worst_case_loss_per_window`, mapped at the
    /// `bot` boundary). All three strategies are owned and enabled per the
    /// [`RiskParams`] toggles.
    #[must_use]
    pub fn new(params: RiskParams, series_caps: HashMap<Series, Dollars>) -> Self {
        let core = RiskCore::new(&params, series_caps);
        let shared = Arc::new(Mutex::new(GateState {
            open_notional_cap: params.max_open_notional,
            ..GateState::default()
        }));
        let quoter = QuoteManager::new(params.quote_manager, params.quote, params.normalizer);
        let momentum = MomentumTaker::new(params.momentum, params.normalizer);
        let late = LateWindowTaker::new(params.late_window, params.normalizer);
        let model = ModelTaker::new(params.model, params.normalizer);
        let arbiter = FireLedger::new(
            params.arbitration_window_ms,
            params.series_precedence.clone(),
        );
        Self {
            params,
            core,
            shared,
            quoter,
            momentum,
            late,
            model,
            arbiter,
            published: Vec::new(),
        }
    }

    // ---- inspection ---------------------------------------------------------

    /// Drains the breaker events to publish on the bus (call after each input).
    pub fn take_published(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.published)
    }

    /// The current breaker state for the dashboard risk panel (§10.5).
    #[must_use]
    pub fn state_snapshot(&self) -> RiskStateSnapshot {
        self.core.snapshot()
    }

    /// Prunes settled per-window loss accounting older than `cutoff`, bounding
    /// memory for a 24/7 session (the orchestrator calls it periodically; active
    /// windows are always retained). Returns the number of windows dropped.
    pub fn prune_settled_before(&mut self, cutoff: TimestampMs) -> usize {
        self.core.prune_settled_before(cutoff)
    }

    /// Whether a global breaker currently halts all trading.
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.core.is_globally_halted()
    }

    /// Whether the owned quote manager is currently standing down (delegated).
    #[must_use]
    pub fn quoter_standing_down(&self) -> bool {
        self.quoter.is_standing_down()
    }

    /// Whether the owned momentum taker is currently standing down (delegated).
    #[must_use]
    pub fn momentum_standing_down(&self) -> bool {
        self.momentum.is_standing_down()
    }

    /// Whether the owned late-window taker is currently standing down (delegated).
    #[must_use]
    pub fn late_standing_down(&self) -> bool {
        self.late.is_standing_down()
    }

    /// Whether the owned model taker is currently standing down (delegated).
    #[must_use]
    pub fn model_standing_down(&self) -> bool {
        self.model.is_standing_down()
    }

    /// The owned model taker's take count (delegated inspection).
    #[must_use]
    pub fn model_take_count(&self) -> u64 {
        self.model.take_count()
    }

    /// The owned model taker's realized spend across all windows (delegated).
    #[must_use]
    pub fn model_realized_spent(&self) -> Dollars {
        self.model.realized_spent()
    }

    /// The owned model taker's committed-in-flight spend across all windows.
    #[must_use]
    pub fn model_effective_spent(&self) -> Dollars {
        self.model.effective_spent()
    }

    /// Which owned strategy placed `order_id`, for per-driver fill attribution
    /// (§10). Queried at fill time, while the order is still owned by its
    /// strategy (before its terminal update evicts it). `None` if unrecognized.
    #[must_use]
    pub fn driver_of(&self, order_id: &OrderId) -> Option<FillDriver> {
        if self.model.owns(order_id) {
            Some(FillDriver::Model)
        } else if self.momentum.owns(order_id) {
            Some(FillDriver::Momentum)
        } else if self.late.owns(order_id) {
            Some(FillDriver::Late)
        } else if self.quoter.owns(order_id) {
            Some(FillDriver::MakerCore)
        } else {
            None
        }
    }

    /// The owned quote manager's resting-order view for a specific window
    /// (delegated inspection).
    #[must_use]
    pub fn resting_view_for(&self, window: WindowId) -> Option<&RestingView> {
        self.quoter.resting_view_for(window)
    }

    /// The total number of live resting maker orders across every active window
    /// (delegated inspection).
    #[must_use]
    pub fn total_resting_len(&self) -> usize {
        self.quoter.total_resting()
    }

    /// The owned quote manager's placement-cycle count (delegated inspection).
    #[must_use]
    pub fn place_cycle_count(&self) -> u64 {
        self.quoter.place_cycle_count()
    }

    /// The owned momentum taker's take count (delegated inspection).
    #[must_use]
    pub fn momentum_take_count(&self) -> u64 {
        self.momentum.take_count()
    }

    /// The owned momentum taker's realized spend (delegated inspection).
    #[must_use]
    pub fn momentum_realized_spent(&self) -> Dollars {
        self.momentum.realized_spent()
    }

    /// The owned momentum taker's committed-in-flight spend (delegated).
    #[must_use]
    pub fn momentum_effective_spent(&self) -> Dollars {
        self.momentum.effective_spent()
    }

    /// The owned late-window taker's take count (delegated inspection).
    #[must_use]
    pub fn late_take_count(&self) -> u64 {
        self.late.take_count()
    }

    /// The owned late-window taker's realized spend (delegated inspection).
    #[must_use]
    pub fn late_realized_spent(&self) -> Dollars {
        self.late.realized_spent()
    }

    /// The owned late-window taker's committed-in-flight spend (delegated).
    #[must_use]
    pub fn late_effective_spent(&self) -> Dollars {
        self.late.effective_spent()
    }

    // ---- entry points -------------------------------------------------------

    /// Handles one bus event: folds it into the breaker machine, applies the
    /// resulting risk actions, and forwards it (and any breaker event) to the
    /// owned strategies through the guard.
    pub async fn on_event<P: VenuePort>(&mut self, event: &Event, port: &P, now: TimestampMs) {
        let obs = self.take_observations();
        let mut outputs = self.core.fold_observations(&obs, now);
        outputs.extend(self.core.on_event(event, now));
        let forwards = self.apply_outputs(outputs, port).await;
        for ev in &forwards {
            self.forward_bus_event(ev, port, now).await;
        }
        // Risk events are handled via the breaker outputs above (and the
        // ClockSkew `Forward`); forwarding the raw `Event::Risk` again would
        // double-react.
        if !matches!(event, Event::Risk(_)) {
            self.forward_bus_event(event, port, now).await;
        }
    }

    /// Handles one venue-stream item: folds it (open notional, per-window loss,
    /// user-WS connectivity), applies risk actions, then forwards the item to the
    /// owned strategies.
    pub async fn on_venue_event<P: VenuePort>(
        &mut self,
        ve: &VenueEvent,
        port: &P,
        now: TimestampMs,
    ) {
        let obs = self.take_observations();
        let mut outputs = self.core.fold_observations(&obs, now);
        outputs.extend(self.core.on_venue_event(ve, now));
        let forwards = self.apply_outputs(outputs, port).await;
        for ev in &forwards {
            self.forward_bus_event(ev, port, now).await;
        }
        if self.params.quoter_enabled {
            self.quoter.on_venue_event(ve, now);
        }
        if self.params.momentum_enabled {
            self.momentum.on_venue_event(ve, now);
        }
        if self.params.late_window_enabled {
            self.late.on_venue_event(ve, now);
        }
        if self.params.model_enabled {
            self.model.on_venue_event(ve, now);
        }
    }

    /// Delivers one shadow prediction to the owned model taker — the single path
    /// through which the model can fire (a FAK, gated by the interposed
    /// [`GuardedPort`]). The orchestrator calls this when it drains the shadow
    /// prediction stream; it returns the decision outcome for journaling.
    ///
    /// This deliberately does **not** run the observation/fold cycle the other
    /// entry points run: the interposed guard enforces §5/§11 (global halt /
    /// per-window halt / open-notional) on the FAK, and the model taker's own
    /// `standing_down` (set from the forwarded `Event::Risk`) blocks it under a
    /// breaker — so no separate fold is needed here.
    pub async fn on_model_prediction<P: VenuePort>(
        &mut self,
        pred: &ModelPrediction,
        port: &P,
        now: TimestampMs,
    ) -> Option<ModelTakeOutcome> {
        if !self.params.model_enabled {
            return None;
        }
        let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
        // The maker's per-window resting views (immutable, disjoint from the
        // mutable `self.model`/`self.arbiter` borrows) resolve the §7 self-match
        // filter for whichever window the prediction fires.
        let quoter: &dyn RestingLookup = &self.quoter;
        let out = self
            .model
            .on_prediction(pred, &guard, now, &mut self.arbiter, Some(quoter))
            .await;
        Some(out)
    }

    /// The periodic tick: runs the time-based breaker recomputes (the 500 ms
    /// staleness timer, the sanity dwell, the error-window eviction, the
    /// engine-restart cooldown), applies risk actions, then drives the quoter's
    /// debounced converge (guarded).
    pub async fn on_tick<P: VenuePort>(&mut self, port: &P, now: TimestampMs) {
        let obs = self.take_observations();
        let mut outputs = self.core.fold_observations(&obs, now);
        outputs.extend(self.core.on_tick(now));
        let forwards = self.apply_outputs(outputs, port).await;
        for ev in &forwards {
            self.forward_bus_event(ev, port, now).await;
        }
        if self.params.quoter_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            self.quoter.on_requote_tick(&guard, now).await;
        }
    }

    // ---- internals ----------------------------------------------------------

    /// Applies the core's outputs (publish / issue authoritative cancels /
    /// project the gate) and returns the breaker events to forward to strategies.
    async fn apply_outputs<P: VenuePort>(
        &mut self,
        outputs: Vec<RiskOutput>,
        port: &P,
    ) -> Vec<Event> {
        let mut forwards = Vec::new();
        for out in outputs {
            match out {
                RiskOutput::Announce(re) => {
                    log_risk(&re);
                    self.published.push(Event::Risk(re));
                    forwards.push(Event::Risk(re));
                }
                RiskOutput::Publish(re) => {
                    self.published.push(Event::Risk(re));
                }
                RiskOutput::Forward(re) => {
                    log_risk(&re);
                    forwards.push(Event::Risk(re));
                }
                RiskOutput::CancelAll => {
                    let r = port.cancel_all().await;
                    tracing::warn!(target: "risk", ok = r.is_ok(), "authoritative cancel-all");
                }
                RiskOutput::CancelMarket(cid) => {
                    let r = port.cancel_market(&cid).await;
                    tracing::warn!(target: "risk", market = %cid, ok = r.is_ok(), "authoritative cancel-market (window loss)");
                }
            }
        }
        self.project_gate();
        forwards
    }

    /// Forwards one event to each enabled strategy through a fresh guard.
    ///
    /// The resting view (`self.quoter`, immutable) and the fire ledger
    /// (`self.arbiter`, mutable) are threaded into the takers for §7 self-match
    /// prevention and §8 momentum-vs-model arbitration. These are **disjoint
    /// fields** of `self` (the guard only clones `self.shared`), so the borrow
    /// checker admits the immutable `resting` borrow alongside the mutable taker
    /// and arbiter borrows in one call.
    async fn forward_bus_event<P: VenuePort>(&mut self, ev: &Event, port: &P, now: TimestampMs) {
        if self.params.quoter_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            self.quoter.on_event(ev, &guard, now).await;
        }
        if self.params.momentum_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            let quoter: &dyn RestingLookup = &self.quoter;
            self.momentum
                .on_event(ev, &guard, now, &mut self.arbiter, Some(quoter))
                .await;
        }
        if self.params.late_window_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            let quoter: &dyn RestingLookup = &self.quoter;
            self.late.on_event(ev, &guard, now, Some(quoter)).await;
        }
        if self.params.model_enabled {
            // Sync state refresh only — the model fires on a prediction, never a
            // bus event. It holds no venue port here.
            self.model.on_bus_event(ev);
        }
        // Bound the arbiter's per-window ledger for a 24/7 session.
        if let Event::Window { market, lifecycle } = ev
            && matches!(
                lifecycle,
                WindowLifecycle::Closed | WindowLifecycle::Resolved { .. }
            )
        {
            self.arbiter.drop_window(market.window);
        }
    }

    /// Projects the core's gating state into the shared [`GateState`] the guard
    /// reads. A brief, `await`-free lock.
    fn project_gate(&self) {
        let mut g = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        g.halted = self.core.is_globally_halted();
        g.halted_windows = self.core.halted_windows().clone();
        g.open_notional = self.core.open_notional();
        g.open_notional_cap = self.params.max_open_notional;
    }

    /// Drains the guard's venue-error observations for the core to fold.
    fn take_observations(&self) -> Vec<GuardObservation> {
        let mut g = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut g.observations)
    }
}
