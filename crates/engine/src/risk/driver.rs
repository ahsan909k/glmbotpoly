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

use core_types::{Dollars, Event, RiskEvent, Series, TimestampMs};
use venue_api::{VenueEvent, VenuePort};

use super::core::{RiskCore, RiskOutput};
use super::guard::GuardedPort;
use super::params::RiskParams;
use super::state::{GateState, GuardObservation, RiskStateSnapshot};
use crate::late_window::LateWindowTaker;
use crate::quote_manager::{QuoteManager, RestingView};
use crate::taker::MomentumTaker;

/// The single gateway between every strategy and the venue port.
pub struct RiskManager {
    params: RiskParams,
    core: RiskCore,
    shared: Arc<Mutex<GateState>>,
    quoter: QuoteManager,
    momentum: MomentumTaker,
    late: LateWindowTaker,
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
        Self {
            params,
            core,
            shared,
            quoter,
            momentum,
            late,
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

    /// The owned quote manager's resting-order view (delegated inspection).
    #[must_use]
    pub fn resting_view(&self) -> Option<&RestingView> {
        self.quoter.resting_view()
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
    async fn forward_bus_event<P: VenuePort>(&mut self, ev: &Event, port: &P, now: TimestampMs) {
        if self.params.quoter_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            self.quoter.on_event(ev, &guard, now).await;
        }
        if self.params.momentum_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            self.momentum.on_event(ev, &guard, now).await;
        }
        if self.params.late_window_enabled {
            let guard = GuardedPort::new(port, Arc::clone(&self.shared), now);
            self.late.on_event(ev, &guard, now).await;
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
