//! [`RiskCore`]: the sans-IO §11 breaker state machine.
//!
//! Inputs arrive as `&mut self` methods taking the current `now`; decisions
//! leave as a `Vec<RiskOutput>` the [`RiskManager`](super::RiskManager) applies
//! (publish to the bus, forward to the owned strategies, issue authoritative
//! cancels, project the gate). It is the **single writer of breaker semantics**
//! (the contract every feed/model crate documents): feeds report, risk reacts.
//!
//! No clock, no IO, no allocation beyond the returned vec — so it is exhaustively
//! unit-testable, and the same fold reproduces on a journal replay.
//!
//! ## Which input drives which breaker (§11)
//!
//! - `FeedStale` — a 500 ms staleness timer on the **fast** signal feed
//!   (`BinanceDirect`/`Mid`, the only stream whose silence means we are blind to
//!   a move), plus any latched `FeedHealth`/`BookHealth` stale report. Chainlink
//!   (~1 Hz) and the book have their own watchdogs and report through those
//!   events — a literal 500 ms bound on them would trip permanently. The timer
//!   bound is `feed_staleness_ms + feed_staleness_grace_ms`; grace defaults to 0
//!   (strict §11) and is a home-testing concession for jittery links only.
//! - `WsDisconnect` — market-WS via `BookHealth::Unreliable{Disconnected}`,
//!   user-WS via `VenueEvent::Connectivity{connected:false}`.
//! - `ClockSkew` — honored as authoritative input from the `timeutil` skew
//!   monitor's `Event::Risk(BreakerTripped/Cleared{ClockSkew})`; never re-minted
//!   here (forwarded to strategies, not re-published).
//! - `WindowLoss` — per-window: this core's own [`InventoryManager`] worst-case
//!   vs the per-series cap → halt that window + `cancel_market`.
//! - `DailyStop` — cumulative settled realized PnL for the UTC day; latched until
//!   `ControlEvent::Reset`.
//! - `FairVsMid` — `|p_up − book mid|` over the sanity bound for the dwell.
//! - `ErrorRate` / `EngineRestart` — folded from guard-observed venue errors.
//! - `Manual` — `ControlEvent::Kill`; latched until `ControlEvent::Reset`.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use core_types::{
    Asset, BookHealth, BookUnreliableReason, BreakerKind, ConditionId, Dollars, Event, FeedHealth,
    PriceSource, RiskEvent, Series, TickKind, TimestampMs, TokenId, TopOfBook, WindowId,
    WindowLifecycle,
};
use rust_decimal::prelude::ToPrimitive;
use venue_api::VenueEvent;

use super::params::RiskParams;
use super::state::{GuardObservation, RiskStateSnapshot, ShadowStop};
use crate::inventory::{InventoryEffect, InventoryManager};

/// One decision the core hands back to the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskOutput {
    /// Publish to the bus **and** forward to the owned strategies (a breaker the
    /// risk manager minted — the strategies stand down / reconcile on it).
    Announce(RiskEvent),
    /// Publish to the bus only (e.g. the `CancelAllIssued` that accompanies a
    /// trip — informative for the journal/dashboard; the strategies already get
    /// the `BreakerTripped` via [`Announce`](Self::Announce)).
    Publish(RiskEvent),
    /// Forward to the owned strategies only, **without** re-publishing — used for
    /// `ClockSkew`, which the skew monitor already put on the bus.
    Forward(RiskEvent),
    /// Issue the authoritative cancel-all through the raw port.
    CancelAll,
    /// Issue the authoritative cancel for one market (a per-window loss halt).
    CancelMarket(ConditionId),
    /// A loss stop that WOULD have tripped but is only recorded (shadow-loss-stops
    /// mode, §11 paper eval): trading continues. The driver logs it and buffers it
    /// for [`take_shadow_stops`](super::RiskManager::take_shadow_stops) — never a
    /// cancel or halt.
    ShadowStop(ShadowStop),
}

/// Per-window fair-vs-mid sanity accounting (the `FairVsMid` breaker input).
///
/// Two-tier: a moderate divergence must persist past the SLOW dwell (the sanity
/// net — healthy lead-lag self-resolves before it), while a catastrophic
/// divergence trips on the SHORT fast dwell (a basis-bug-class broken fair is
/// caught in seconds, not the ~20 s a lead-lag gap can legitimately take).
#[derive(Default)]
struct WindowSanity {
    /// Latest model `p_up` for this window.
    last_p_up: Option<f64>,
    /// Latest Up-token book mid for this window.
    last_mid: Option<f64>,
    /// When this window first breached the SLOW (sanity-net) bound; `None` while
    /// within it.
    breach_since_slow: Option<TimestampMs>,
    /// When this window first breached the FAST (catastrophic) bound; `None`
    /// while within it.
    breach_since_fast: Option<TimestampMs>,
}

/// The §11 breaker state machine (sans-IO).
pub(crate) struct RiskCore {
    // thresholds (copied from RiskParams at construction)
    feed_staleness_ms: i64,
    feed_staleness_grace_ms: i64,
    book_staleness_dwell_ms: i64,
    daily_stop_loss: Dollars,
    max_open_notional: Dollars,
    // FairVsMid two-tier: SLOW = sanity net (moderate gap, long dwell so healthy
    // lead-lag self-resolves first); FAST = catastrophic gap, short dwell.
    sanity_bound: f64,
    sanity_bound_duration_ms: i64,
    sanity_bound_fast: f64,
    sanity_dwell_fast_ms: i64,
    error_max: u32,
    error_window_ms: i64,
    engine_restart_cooldown_ms: i64,
    series_caps: HashMap<Series, Dollars>,

    // global breaker set (drives the global halt). WindowLoss is NOT here — it is
    // per-window via `halted_windows`.
    tripped: BTreeSet<BreakerKind>,
    halted_windows: HashSet<WindowId>,

    // bounded trailing ring of global-breaker trip times (the "trips in the last
    // hour" rate). Pruned to the trailing hour on every trip and every tick.
    trip_times: VecDeque<TimestampMs>,

    // feed staleness
    binance_mid_last: HashMap<Asset, TimestampMs>,
    stale_feeds: HashSet<(PriceSource, Asset, TickKind)>,
    // Windows whose CLOB book is currently Stale/Crossed/TopDivergence, mapped to
    // the timestamp the (still-unrecovered) staleness began. A window counts
    // toward the global `FeedStale` breaker only once it has been continuously
    // stale for `book_staleness_dwell_ms` — filtering the transient rollover
    // churn while still tripping on a sustained outage.
    stale_books: HashMap<WindowId, TimestampMs>,

    // ws disconnect
    ws_down_windows: HashSet<WindowId>,
    user_ws_down: bool,
    /// Windows that have closed/resolved. Their book-health is ignored: a closed
    /// window's connection teardown emits a `Disconnected` (and, during the
    /// post-close linger, `Stale`) that never gets a paired `Recovered`, so
    /// without this it would latch `WsDisconnect`/`FeedStale` forever — wedging a
    /// permanent global halt after the first window rollover.
    ended_windows: HashSet<WindowId>,

    // sanity (fair vs mid) — per window; any window over the dwell trips the
    // global FairVsMid halt (semantics unchanged — still a global breaker).
    sanity: HashMap<WindowId, WindowSanity>,
    /// Up token → the window it belongs to, for routing an incoming mid.
    sanity_up_token: HashMap<TokenId, WindowId>,

    // error rate + engine restart
    infra_errors: VecDeque<TimestampMs>,
    engine_restart_last: Option<TimestampMs>,

    // daily stop
    daily_realized: Dollars,
    daily_epoch_day: Option<i64>,
    daily_stop_latched: bool,

    // shadow-loss-stops (§11 paper eval): compute the DailyStop/WindowLoss
    // thresholds exactly, but record instead of halting/cancelling.
    shadow_loss_stops: bool,
    shadow_tripped: Vec<ShadowStop>,
    shadow_window_latched: HashSet<WindowId>,

    // manual
    manual_latched: bool,

    // open notional (authoritative, from the venue stream)
    open_orders: HashMap<core_types::OrderId, Dollars>,
    open_notional: Dollars,

    // per-window loss accounting + cancel-market targets
    inv: InventoryManager,
    window_cids: HashMap<WindowId, ConditionId>,
}

const DAY_MS: i64 = 86_400_000;

/// The trailing window for the `trips_last_hour` rate (1 hour in ms).
const TRIPS_WINDOW_MS: i64 = 3_600_000;

fn mid_of(top: &TopOfBook) -> Option<f64> {
    match (top.bid, top.ask) {
        (Some(b), Some(a)) => ((b.price.as_decimal() + a.price.as_decimal())
            / rust_decimal::Decimal::from(2))
        .to_f64(),
        _ => None,
    }
}

impl RiskCore {
    pub(crate) fn new(params: &RiskParams, series_caps: HashMap<Series, Dollars>) -> Self {
        Self {
            feed_staleness_ms: params.feed_staleness_ms,
            feed_staleness_grace_ms: params.feed_staleness_grace_ms,
            book_staleness_dwell_ms: params.book_staleness_dwell_ms,
            daily_stop_loss: params.daily_stop_loss,
            max_open_notional: params.max_open_notional,
            sanity_bound: params.sanity_bound,
            sanity_bound_duration_ms: params.sanity_bound_duration_ms,
            sanity_bound_fast: params.sanity_bound_fast,
            sanity_dwell_fast_ms: params.sanity_bound_duration_fast_ms,
            error_max: params.error_breaker_max_errors,
            error_window_ms: params.error_breaker_window_ms,
            engine_restart_cooldown_ms: params.engine_restart_cooldown_ms,
            series_caps,
            tripped: BTreeSet::new(),
            halted_windows: HashSet::new(),
            trip_times: VecDeque::new(),
            binance_mid_last: HashMap::new(),
            stale_feeds: HashSet::new(),
            stale_books: HashMap::new(),
            ws_down_windows: HashSet::new(),
            user_ws_down: false,
            ended_windows: HashSet::new(),
            sanity: HashMap::new(),
            sanity_up_token: HashMap::new(),
            infra_errors: VecDeque::new(),
            engine_restart_last: None,
            daily_realized: Dollars::ZERO,
            daily_epoch_day: None,
            daily_stop_latched: false,
            shadow_loss_stops: params.shadow_loss_stops,
            shadow_tripped: Vec::new(),
            shadow_window_latched: HashSet::new(),
            manual_latched: false,
            open_orders: HashMap::new(),
            open_notional: Dollars::ZERO,
            inv: InventoryManager::new(),
            window_cids: HashMap::new(),
        }
    }

    // ---- projections read by the driver ------------------------------------

    pub(crate) fn is_globally_halted(&self) -> bool {
        !self.tripped.is_empty()
    }

    pub(crate) fn halted_windows(&self) -> &HashSet<WindowId> {
        &self.halted_windows
    }

    pub(crate) fn open_notional(&self) -> Dollars {
        self.open_notional
    }

    /// Prunes settled per-window loss books (and their cached condition ids)
    /// older than `cutoff`, bounding memory for 24/7 operation. Active windows
    /// are retained regardless of cutoff (their cid is kept so a window-loss can
    /// still `cancel_market`). Returns the number of windows dropped.
    pub(crate) fn prune_settled_before(&mut self, cutoff: TimestampMs) -> usize {
        // Drop cached cids only for windows that are BOTH old and settled —
        // never an active long window whose open is already past the cutoff.
        let old_settled: Vec<WindowId> = self
            .window_cids
            .keys()
            .copied()
            .filter(|w| w.open_time.as_millis() < cutoff.as_millis() && self.inv.is_settled(*w))
            .collect();
        for w in &old_settled {
            self.window_cids.remove(w);
        }
        // Forget ended windows older than the cutoff — far past any window's life
        // plus linger, so no late book event can arrive for them.
        self.ended_windows
            .retain(|w| w.open_time.as_millis() >= cutoff.as_millis());
        // Drop any lingering shadow-loss records for old windows (normally already
        // pruned on Closed/Resolved — belt-and-suspenders for 24/7).
        self.shadow_window_latched
            .retain(|w| w.open_time.as_millis() >= cutoff.as_millis());
        self.shadow_tripped.retain(|s| {
            s.window
                .is_none_or(|w| w.open_time.as_millis() >= cutoff.as_millis())
        });
        self.inv.prune_settled_before(cutoff)
    }

    pub(crate) fn snapshot(&self) -> RiskStateSnapshot {
        let mut halted_windows_list: Vec<WindowId> = self.halted_windows.iter().copied().collect();
        halted_windows_list.sort_unstable();
        RiskStateSnapshot {
            tripped: self.tripped.iter().copied().collect(),
            window_loss_active: !self.halted_windows.is_empty(),
            halted_windows: self.halted_windows.len(),
            halted_windows_list,
            trips_last_hour: u32::try_from(self.trip_times.len()).unwrap_or(u32::MAX),
            open_notional: self.open_notional,
            open_notional_cap: self.max_open_notional,
            daily_pnl: self.daily_realized,
            error_count: u32::try_from(self.infra_errors.len()).unwrap_or(u32::MAX),
            sanity_breached: self
                .sanity
                .values()
                .any(|s| s.breach_since_slow.is_some() || s.breach_since_fast.is_some()),
            globally_halted: !self.tripped.is_empty(),
            shadow_tripped: self.shadow_tripped.clone(),
        }
    }

    /// Records one global-breaker trip at `now` and prunes the ring to the
    /// trailing hour (so [`snapshot`](Self::snapshot) reports `trips_last_hour`
    /// simply as the ring length).
    fn record_trip(&mut self, now: TimestampMs) {
        self.trip_times.push_back(now);
        self.prune_trip_times(now);
    }

    /// Drops trip timestamps older than the trailing-hour window relative to
    /// `now`.
    fn prune_trip_times(&mut self, now: TimestampMs) {
        while let Some(&front) = self.trip_times.front() {
            if now.as_millis().saturating_sub(front.as_millis()) > TRIPS_WINDOW_MS {
                self.trip_times.pop_front();
            } else {
                break;
            }
        }
    }

    /// Global-breaker trips observed within the trailing hour ending at `now`
    /// (exact count; the ring is kept pruned so `snapshot` can read its length).
    #[cfg(test)]
    pub(crate) fn trips_last_hour(&self, now: TimestampMs) -> u32 {
        let n = self
            .trip_times
            .iter()
            .filter(|t| now.as_millis().saturating_sub(t.as_millis()) <= TRIPS_WINDOW_MS)
            .count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }

    // ---- breaker helpers ----------------------------------------------------

    /// Trip a global breaker (evacuate): announce + cancel-all, once. Records the
    /// trip in the trailing-hour ring.
    fn trip_global(&mut self, b: BreakerKind, now: TimestampMs, out: &mut Vec<RiskOutput>) {
        if self.tripped.insert(b) {
            self.record_trip(now);
            out.push(RiskOutput::Announce(RiskEvent::BreakerTripped {
                breaker: b,
            }));
            out.push(RiskOutput::Publish(RiskEvent::CancelAllIssued {
                reason: b,
            }));
            out.push(RiskOutput::CancelAll);
        }
    }

    /// Clear a global breaker, once.
    fn clear_global(&mut self, b: BreakerKind, out: &mut Vec<RiskOutput>) {
        if self.tripped.remove(&b) {
            out.push(RiskOutput::Announce(RiskEvent::BreakerCleared {
                breaker: b,
            }));
        }
    }

    // ---- entry points -------------------------------------------------------

    /// Folds guard-observed venue errors (drained by the driver) into the
    /// error-rate and engine-restart breakers.
    pub(crate) fn fold_observations(
        &mut self,
        obs: &[GuardObservation],
        now: TimestampMs,
    ) -> Vec<RiskOutput> {
        for o in obs {
            match *o {
                GuardObservation::InfraError(ts) => self.infra_errors.push_back(ts),
                GuardObservation::EngineRestart(ts) => self.engine_restart_last = Some(ts),
            }
        }
        let mut out = Vec::new();
        self.recompute_error_rate(now, &mut out);
        self.recompute_engine_restart(now, &mut out);
        out
    }

    /// Folds one bus event.
    pub(crate) fn on_event(&mut self, event: &Event, now: TimestampMs) -> Vec<RiskOutput> {
        let mut out = Vec::new();
        match event {
            Event::PriceTick(t) => {
                if t.source == PriceSource::BinanceDirect && t.kind == TickKind::Mid {
                    // Staleness is local-receive-time liveness (the `FeedHealth`
                    // contract): stamp with `now`, not the tick's self-reported
                    // exchange time, which may legitimately be backdated.
                    self.binance_mid_last.insert(t.asset, now);
                    self.recompute_feed_stale(now, &mut out);
                }
            }
            Event::FeedHealth(FeedHealth::Stale {
                source,
                asset,
                kind,
                ..
            }) => {
                self.stale_feeds.insert((*source, *asset, *kind));
                self.recompute_feed_stale(now, &mut out);
            }
            Event::FeedHealth(FeedHealth::Recovered {
                source,
                asset,
                kind,
                ..
            }) => {
                self.stale_feeds.remove(&(*source, *asset, *kind));
                self.recompute_feed_stale(now, &mut out);
            }
            Event::BookHealth(BookHealth::Unreliable { window, reason, .. }) => {
                // Ignore book-health for windows that have ended: a closed/
                // resolved window's connection teardown emits a Disconnected (and,
                // during linger, Stale) that never gets a paired Recovered, and
                // would otherwise wedge a global breaker (the rollover-halt bug).
                if !self.ended_windows.contains(window) {
                    if *reason == BookUnreliableReason::Disconnected {
                        self.ws_down_windows.insert(*window);
                        self.recompute_ws(now, &mut out);
                    } else {
                        // Keep the FIRST stale-since so a window that stays stale
                        // accrues dwell across repeated Unreliable reports.
                        self.stale_books.entry(*window).or_insert(now);
                        self.recompute_feed_stale(now, &mut out);
                    }
                }
            }
            Event::BookHealth(BookHealth::Recovered { window, .. }) => {
                let was_ws = self.ws_down_windows.remove(window);
                let was_book = self.stale_books.remove(window).is_some();
                if was_ws {
                    self.recompute_ws(now, &mut out);
                }
                if was_book {
                    self.recompute_feed_stale(now, &mut out);
                }
            }
            Event::TopOfBook { token_id, top } => {
                if let Some(w) = self.sanity_up_token.get(token_id.as_ref()).copied()
                    && let Some(m) = mid_of(top)
                {
                    self.sanity.entry(w).or_default().last_mid = Some(m);
                    self.recompute_sanity(now, &mut out);
                }
            }
            Event::Book(snap) => {
                if let Some(w) = self.sanity_up_token.get(&snap.token_id).copied()
                    && let Some(m) = mid_of(&snap.top())
                {
                    self.sanity.entry(w).or_default().last_mid = Some(m);
                    self.recompute_sanity(now, &mut out);
                }
            }
            Event::Model(snap) => {
                // Fold p_up for a window we currently track (open) — mirrors the
                // original `active_window == snap.window` gate, now per window.
                if let Some(w) = snap.window {
                    let folded = if let Some(s) = self.sanity.get_mut(&w) {
                        s.last_p_up = Some(snap.p_up);
                        true
                    } else {
                        false
                    };
                    if folded {
                        self.recompute_sanity(now, &mut out);
                    }
                }
            }
            Event::Window { market, lifecycle } => {
                self.window_cids
                    .insert(market.window, market.condition_id.clone());
                self.on_window(
                    market.window,
                    market.tokens.up.clone(),
                    *lifecycle,
                    event,
                    now,
                    &mut out,
                );
            }
            Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew,
            }) => {
                if self.tripped.insert(BreakerKind::ClockSkew) {
                    out.push(RiskOutput::Forward(RiskEvent::BreakerTripped {
                        breaker: BreakerKind::ClockSkew,
                    }));
                    out.push(RiskOutput::CancelAll);
                }
            }
            Event::Risk(RiskEvent::BreakerCleared {
                breaker: BreakerKind::ClockSkew,
            }) => {
                if self.tripped.remove(&BreakerKind::ClockSkew) {
                    out.push(RiskOutput::Forward(RiskEvent::BreakerCleared {
                        breaker: BreakerKind::ClockSkew,
                    }));
                }
            }
            Event::Control(core_types::ControlEvent::Kill) => {
                self.manual_latched = true;
                self.trip_global(BreakerKind::Manual, now, &mut out);
            }
            Event::Control(core_types::ControlEvent::Reset) => {
                self.manual_latched = false;
                self.daily_stop_latched = false;
                self.clear_global(BreakerKind::Manual, &mut out);
                self.clear_global(BreakerKind::DailyStop, &mut out);
            }
            Event::Control(core_types::ControlEvent::DailyStopReset) => {
                // Targeted reset: clears the daily stop only, leaving an
                // unrelated manual halt latched (its counterpart to `Reset`).
                self.daily_stop_latched = false;
                self.clear_global(BreakerKind::DailyStop, &mut out);
            }
            // Bus `Fill`/`OrderUpdate` are for other consumers; our inventory and
            // open notional come from the venue stream (no double count). Other
            // events and `Risk` echoes of our own breakers are ignored.
            _ => {}
        }
        out
    }

    fn on_window(
        &mut self,
        window: WindowId,
        up_token: TokenId,
        lifecycle: WindowLifecycle,
        event: &Event,
        now: TimestampMs,
        out: &mut Vec<RiskOutput>,
    ) {
        match lifecycle {
            WindowLifecycle::Open => {
                // Fresh per-window sanity slot + its up-token routing on open.
                self.sanity.insert(window, WindowSanity::default());
                self.sanity_up_token.insert(up_token, window);
            }
            WindowLifecycle::Closing | WindowLifecycle::Discovered => {}
            WindowLifecycle::Closed => {
                self.prune_window_sanity(window, now, out);
                self.end_window_book_health(window, now, out);
                self.prune_window_shadow(window);
            }
            WindowLifecycle::Resolved { .. } => {
                for eff in self.inv.on_event(event) {
                    if let InventoryEffect::Settled(summary) = eff {
                        self.apply_settlement(summary.realized_pnl, summary.ts, out);
                    }
                }
                self.resume_window(window, out);
                self.prune_window_sanity(window, now, out);
                self.end_window_book_health(window, now, out);
                self.prune_window_shadow(window);
            }
        }
    }

    /// Drops a window's sanity accounting (and its up-token routing) on
    /// Closed/Resolved, then re-evaluates the global `FairVsMid` breaker so a halt
    /// that was held only by the departing window clears once no live window
    /// breaches. Bounds memory for 24/7 operation.
    fn prune_window_sanity(
        &mut self,
        window: WindowId,
        now: TimestampMs,
        out: &mut Vec<RiskOutput>,
    ) {
        let removed = self.sanity.remove(&window).is_some();
        self.sanity_up_token.retain(|_, w| *w != window);
        if removed {
            self.recompute_sanity(now, out);
        }
    }

    /// Drops a window's shadow-loss-stop record + latch on Closed/Resolved, so a
    /// recorded would-be `WindowLoss` does not linger (bounds memory for 24/7).
    fn prune_window_shadow(&mut self, window: WindowId) {
        self.shadow_window_latched.remove(&window);
        self.shadow_tripped.retain(|s| s.window != Some(window));
    }

    /// Marks a window ended and purges its book-health from the global breakers.
    /// A closed/resolved window's connection teardown emits a `Disconnected`
    /// (and, during the post-close linger, `Stale`) that will never get a paired
    /// `Recovered` — without this purge it latches `WsDisconnect`/`FeedStale`
    /// forever, wedging a permanent global halt after the first window rollover.
    fn end_window_book_health(
        &mut self,
        window: WindowId,
        now: TimestampMs,
        out: &mut Vec<RiskOutput>,
    ) {
        self.ended_windows.insert(window);
        if self.ws_down_windows.remove(&window) {
            self.recompute_ws(now, out);
        }
        if self.stale_books.remove(&window).is_some() {
            self.recompute_feed_stale(now, out);
        }
    }

    /// Folds one venue-stream item.
    pub(crate) fn on_venue_event(&mut self, ve: &VenueEvent, now: TimestampMs) -> Vec<RiskOutput> {
        let mut out = Vec::new();
        match ve {
            VenueEvent::Order(u) => {
                if u.state.is_working() {
                    if !self.open_orders.contains_key(&u.order_id) {
                        let n = Dollars::new(u.price.as_decimal() * u.original_size.as_decimal());
                        self.open_orders.insert(u.order_id.clone(), n);
                        self.open_notional = self.open_notional + n;
                    }
                } else if u.state.is_terminal()
                    && let Some(n) = self.open_orders.remove(&u.order_id)
                {
                    self.open_notional = self.open_notional - n;
                }
            }
            VenueEvent::Fill(f) => {
                for eff in self.inv.on_event(&Event::Fill(Arc::clone(f))) {
                    if let InventoryEffect::Snapshot(snap) = eff {
                        self.check_window_loss(
                            snap.window,
                            snap.worst_case_if_excess_loses,
                            now,
                            &mut out,
                        );
                    }
                }
            }
            VenueEvent::Connectivity { connected } => {
                self.user_ws_down = !connected;
                self.recompute_ws(now, &mut out);
            }
        }
        out
    }

    /// Time-based recomputes (the staleness timer, the sanity dwell, the
    /// error-window eviction, the engine-restart cooldown).
    pub(crate) fn on_tick(&mut self, now: TimestampMs) -> Vec<RiskOutput> {
        let mut out = Vec::new();
        self.prune_trip_times(now);
        self.recompute_feed_stale(now, &mut out);
        self.recompute_sanity(now, &mut out);
        self.recompute_error_rate(now, &mut out);
        self.recompute_engine_restart(now, &mut out);
        out
    }

    // ---- recompute helpers --------------------------------------------------

    fn feed_stale_now(&self, now: TimestampMs) -> bool {
        // The trip threshold is the §11 bound plus the (default-0) home-testing
        // grace: a fast feed continuously silent longer than the sum is stale.
        // `binance_mid_last` resets on every fresh tick, so `now - last` measures
        // one continuous gap — grace tolerates transient gaps but a real feed
        // death (a gap past the sum) still trips.
        let threshold = self
            .feed_staleness_ms
            .saturating_add(self.feed_staleness_grace_ms);
        let timer = self
            .binance_mid_last
            .values()
            .any(|&t| now.as_millis().saturating_sub(t.as_millis()) > threshold);
        // A stale book counts only once it has been continuously stale for the
        // dwell (default 0 => immediate, strict §11). Transient rollover churn
        // (Stale then Recover within the dwell) is dropped from stale_books
        // before the next on_tick recompute catches it, so it never trips; a
        // SUSTAINED outage crosses the dwell at the next 500 ms on_tick and does.
        let dwell = self.book_staleness_dwell_ms;
        let book_stale = self
            .stale_books
            .values()
            .any(|&since| now.as_millis().saturating_sub(since.as_millis()) >= dwell);
        timer || !self.stale_feeds.is_empty() || book_stale
    }

    fn recompute_feed_stale(&mut self, now: TimestampMs, out: &mut Vec<RiskOutput>) {
        if self.feed_stale_now(now) {
            self.trip_global(BreakerKind::FeedStale, now, out);
        } else {
            self.clear_global(BreakerKind::FeedStale, out);
        }
    }

    fn recompute_ws(&mut self, now: TimestampMs, out: &mut Vec<RiskOutput>) {
        if !self.ws_down_windows.is_empty() || self.user_ws_down {
            self.trip_global(BreakerKind::WsDisconnect, now, out);
        } else {
            self.clear_global(BreakerKind::WsDisconnect, out);
        }
    }

    fn recompute_sanity(&mut self, now: TimestampMs, out: &mut Vec<RiskOutput>) {
        // Per window, two independent dwell timers (SLOW sanity-net + FAST
        // catastrophic). The GLOBAL FairVsMid trips iff ANY window is past
        // EITHER tier's dwell, and clears only when NONE is (a healthy sibling
        // never clears a diverged one — semantics unchanged, still global).
        let bound_slow = self.sanity_bound;
        let dwell_slow = self.sanity_bound_duration_ms;
        let bound_fast = self.sanity_bound_fast;
        let dwell_fast = self.sanity_dwell_fast_ms;
        let elapsed = |since: TimestampMs| now.as_millis().saturating_sub(since.as_millis());
        let mut any_over_dwell = false;
        for s in self.sanity.values_mut() {
            let div = match (s.last_p_up, s.last_mid) {
                (Some(p), Some(m)) => Some((p - m).abs()),
                _ => None,
            };
            // SLOW tier (moderate gap, long dwell).
            match div {
                Some(d) if d > bound_slow => {
                    let since = *s.breach_since_slow.get_or_insert(now);
                    if elapsed(since) >= dwell_slow {
                        any_over_dwell = true;
                    }
                }
                _ => s.breach_since_slow = None,
            }
            // FAST tier (catastrophic gap, short dwell) — independent timer.
            match div {
                Some(d) if d > bound_fast => {
                    let since = *s.breach_since_fast.get_or_insert(now);
                    if elapsed(since) >= dwell_fast {
                        any_over_dwell = true;
                    }
                }
                _ => s.breach_since_fast = None,
            }
        }
        if any_over_dwell {
            self.trip_global(BreakerKind::FairVsMid, now, out);
        } else {
            self.clear_global(BreakerKind::FairVsMid, out);
        }
    }

    fn recompute_error_rate(&mut self, now: TimestampMs, out: &mut Vec<RiskOutput>) {
        while let Some(&front) = self.infra_errors.front() {
            if now.as_millis().saturating_sub(front.as_millis()) > self.error_window_ms {
                self.infra_errors.pop_front();
            } else {
                break;
            }
        }
        let count = u32::try_from(self.infra_errors.len()).unwrap_or(u32::MAX);
        if count >= self.error_max {
            self.trip_global(BreakerKind::ErrorRate, now, out);
        } else if count == 0 {
            self.clear_global(BreakerKind::ErrorRate, out);
        }
    }

    fn recompute_engine_restart(&mut self, now: TimestampMs, out: &mut Vec<RiskOutput>) {
        match self.engine_restart_last {
            Some(last)
                if now.as_millis().saturating_sub(last.as_millis())
                    <= self.engine_restart_cooldown_ms =>
            {
                self.trip_global(BreakerKind::EngineRestart, now, out);
            }
            Some(_) => {
                self.engine_restart_last = None;
                self.clear_global(BreakerKind::EngineRestart, out);
            }
            None => {}
        }
    }

    // ---- window loss + daily stop ------------------------------------------

    fn check_window_loss(
        &mut self,
        window: WindowId,
        worst_case: Dollars,
        now: TimestampMs,
        out: &mut Vec<RiskOutput>,
    ) {
        let Some(cap) = self.series_caps.get(&window.series).copied() else {
            return; // unconfigured series — nothing to enforce against
        };
        if worst_case <= cap {
            return;
        }
        // Shadow-loss-stops (paper eval): record the would-be WindowLoss trip once
        // per window crossing and keep trading — no halt, no cancel-market.
        if self.shadow_loss_stops {
            if self.shadow_window_latched.insert(window) {
                let s = ShadowStop {
                    kind: BreakerKind::WindowLoss,
                    window: Some(window),
                    threshold: cap,
                    value: worst_case,
                    ts: now,
                };
                self.shadow_tripped.push(s);
                out.push(RiskOutput::ShadowStop(s));
            }
            return;
        }
        if self.halted_windows.contains(&window) {
            return;
        }
        let first = self.halted_windows.is_empty();
        self.halted_windows.insert(window);
        if first {
            out.push(RiskOutput::Announce(RiskEvent::BreakerTripped {
                breaker: BreakerKind::WindowLoss,
            }));
        }
        if let Some(cid) = self.window_cids.get(&window).cloned() {
            out.push(RiskOutput::CancelMarket(cid));
        }
    }

    fn resume_window(&mut self, window: WindowId, out: &mut Vec<RiskOutput>) {
        if self.halted_windows.remove(&window) && self.halted_windows.is_empty() {
            out.push(RiskOutput::Announce(RiskEvent::BreakerCleared {
                breaker: BreakerKind::WindowLoss,
            }));
        }
    }

    fn apply_settlement(&mut self, realized: Dollars, ts: TimestampMs, out: &mut Vec<RiskOutput>) {
        let day = ts.as_millis().div_euclid(DAY_MS);
        if self.daily_epoch_day != Some(day) {
            self.daily_realized = Dollars::ZERO;
            self.daily_epoch_day = Some(day);
            // A new UTC day: in shadow mode, re-arm the DailyStop shadow so a new
            // day can shadow again (in hard mode the latch stays until `Reset`).
            if self.shadow_loss_stops {
                self.daily_stop_latched = false;
                self.shadow_tripped
                    .retain(|s| s.kind != BreakerKind::DailyStop);
            }
        }
        self.daily_realized = self.daily_realized + realized;
        let neg_limit = -self.daily_stop_loss.as_decimal();
        if self.daily_realized.as_decimal() <= neg_limit && !self.daily_stop_latched {
            self.daily_stop_latched = true;
            if self.shadow_loss_stops {
                // Record the would-be DailyStop and keep trading (no evacuation).
                let s = ShadowStop {
                    kind: BreakerKind::DailyStop,
                    window: None,
                    threshold: self.daily_stop_loss,
                    value: self.daily_realized,
                    ts,
                };
                self.shadow_tripped.push(s);
                out.push(RiskOutput::ShadowStop(s));
            } else {
                self.trip_global(BreakerKind::DailyStop, ts, out);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use core_types::{
        AnchorSource, BookLevel, BookSnapshot, ControlEvent, Decimal, FeeParams, Fill, InputAges,
        Liquidity, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot, OrderId, OrderState,
        OrderUpdate, Outcome, Price, ResolutionSource, RoundDir, Side, Size, TickSize, TokenId,
        TokenPair, WindowDuration,
    };
    use rust_decimal::dec;

    use super::*;

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = OPEN_MS + 300_000;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }
    fn series() -> Series {
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        }
    }
    fn window() -> WindowId {
        WindowId {
            series: series(),
            open_time: ts(OPEN_MS),
        }
    }
    fn up_token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn down_token() -> TokenId {
        TokenId::new("222").unwrap()
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
    fn market() -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-risk".to_owned(),
            condition_id: condition(),
            tokens: TokenPair {
                up: up_token(),
                down: down_token(),
            },
            close_time: ts(CLOSE_MS),
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
        })
    }
    fn win_event(lifecycle: WindowLifecycle) -> Event {
        Event::Window {
            market: market(),
            lifecycle,
        }
    }
    fn core() -> RiskCore {
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        RiskCore::new(&RiskParams::default(), caps)
    }
    fn binance_mid(value: Decimal, ms: i64) -> Event {
        Event::PriceTick(core_types::PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Mid,
            value,
            ts_exchange: ts(ms),
            ts_local: ts(ms),
        })
    }
    fn model(p_up: f64, ms: i64) -> Event {
        model_for(window(), p_up, ms)
    }
    fn model_for(win: WindowId, p_up: f64, ms: i64) -> Event {
        Event::Model(ModelSnapshot {
            asset: Asset::Btc,
            window: Some(win),
            p_up,
            z: 0.0,
            sigma_1s: 5e-4,
            sigma_tau: 5e-3,
            basis: 0.0,
            anchor: AnchorSource::Chainlink,
            health: ModelHealth::Ready,
            reason: ModelHealthReason::Nominal,
            input_ages: InputAges {
                chainlink: core_types::DurationMs::from_millis(100),
                binance: core_types::DurationMs::from_millis(50),
            },
            ts: ts(ms),
        })
    }
    fn top_event(token: TokenId, bid: Decimal, ask: Decimal, ms: i64) -> Event {
        Event::TopOfBook {
            token_id: Arc::new(token),
            top: TopOfBook {
                bid: Some(BookLevel {
                    price: px(bid),
                    size: sz(dec!(100)),
                }),
                ask: Some(BookLevel {
                    price: px(ask),
                    size: sz(dec!(100)),
                }),
                ts: ts(ms),
            },
        }
    }
    fn book_event(token: TokenId, bid: Decimal, ask: Decimal, ms: i64) -> Event {
        Event::Book(Arc::new(BookSnapshot {
            token_id: token,
            condition_id: condition(),
            bids: vec![BookLevel {
                price: px(bid),
                size: sz(dec!(100)),
            }],
            asks: vec![BookLevel {
                price: px(ask),
                size: sz(dec!(100)),
            }],
            ts: ts(ms),
            seq_hash: None,
        }))
    }
    fn order_update(id: &str, state: OrderState, price: Decimal, size: Decimal) -> VenueEvent {
        VenueEvent::Order(Arc::new(OrderUpdate {
            order_id: OrderId::new(id).unwrap(),
            window: window(),
            token_id: up_token(),
            side: Side::Buy,
            state,
            price: px(price),
            original_size: sz(size),
            filled_size: Size::ZERO,
            reject_reason: None,
            ts_venue: None,
            ts_local: ts(OPEN_MS),
        }))
    }
    fn fill(outcome: Outcome, side: Side, price: Decimal, size: Decimal) -> VenueEvent {
        VenueEvent::Fill(Arc::new(Fill {
            order_id: OrderId::new("o").unwrap(),
            trade_id: None,
            window: window(),
            token_id: up_token(),
            outcome,
            side,
            price: px(price),
            size: sz(size),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: ts(OPEN_MS + 100),
            ts_local: ts(OPEN_MS + 100),
        }))
    }

    /// Pulls the breaker kinds announced (`Announce(BreakerTripped)`) from outputs.
    fn tripped_in(out: &[RiskOutput]) -> Vec<BreakerKind> {
        out.iter()
            .filter_map(|o| match o {
                RiskOutput::Announce(RiskEvent::BreakerTripped { breaker }) => Some(*breaker),
                _ => None,
            })
            .collect()
    }
    fn cleared_in(out: &[RiskOutput]) -> Vec<BreakerKind> {
        out.iter()
            .filter_map(|o| match o {
                RiskOutput::Announce(RiskEvent::BreakerCleared { breaker }) => Some(*breaker),
                _ => None,
            })
            .collect()
    }
    fn has_cancel_all(out: &[RiskOutput]) -> bool {
        out.iter().any(|o| matches!(o, RiskOutput::CancelAll))
    }
    /// The breaker kinds carried by any `RiskOutput::ShadowStop` in the outputs.
    fn shadow_kinds(out: &[RiskOutput]) -> Vec<BreakerKind> {
        out.iter()
            .filter_map(|o| match o {
                RiskOutput::ShadowStop(s) => Some(s.kind),
                _ => None,
            })
            .collect()
    }
    fn has_cancel_market(out: &[RiskOutput]) -> bool {
        out.iter().any(|o| matches!(o, RiskOutput::CancelMarket(_)))
    }
    fn shadow_core(daily_stop_loss: Decimal) -> RiskCore {
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        let params = RiskParams {
            daily_stop_loss: Dollars::new(daily_stop_loss),
            shadow_loss_stops: true,
            ..RiskParams::default()
        };
        RiskCore::new(&params, caps)
    }

    fn open_window(c: &mut RiskCore) {
        let _ = c.on_event(&win_event(WindowLifecycle::Open), ts(OPEN_MS));
    }

    // ---- feed staleness (fast feed only) -----------------------------------

    #[test]
    fn binance_mid_staleness_trips_after_500ms_and_clears_on_fresh_tick() {
        let mut c = core();
        open_window(&mut c);
        let _ = c.on_event(&binance_mid(dec!(60000), OPEN_MS), ts(OPEN_MS));
        // 400 ms later — still fresh.
        assert!(c.on_tick(ts(OPEN_MS + 400)).is_empty());
        assert!(!c.is_globally_halted());
        // 600 ms later — stale → trip FeedStale + cancel-all.
        let out = c.on_tick(ts(OPEN_MS + 600));
        assert_eq!(tripped_in(&out), vec![BreakerKind::FeedStale]);
        assert!(has_cancel_all(&out));
        assert!(c.is_globally_halted());
        // A fresh tick clears it.
        let out = c.on_event(&binance_mid(dec!(60010), OPEN_MS + 650), ts(OPEN_MS + 650));
        assert_eq!(cleared_in(&out), vec![BreakerKind::FeedStale]);
        assert!(!c.is_globally_halted());
    }

    #[test]
    fn grace_tolerates_a_transient_gap_within_the_sum_but_a_real_death_still_trips() {
        // Home-testing concession: with a 1500 ms grace, the fast-feed timer
        // only trips past 500 + 1500 = 2000 ms of continuous silence.
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        let params = RiskParams {
            feed_staleness_grace_ms: 1500,
            ..RiskParams::default()
        };
        let mut c = RiskCore::new(&params, caps);
        open_window(&mut c);
        let _ = c.on_event(&binance_mid(dec!(60000), OPEN_MS), ts(OPEN_MS));
        // A 1200 ms gap — would trip the strict 500 ms bound, but is within the
        // 2000 ms grace-extended threshold, so no trip and no cancel-all.
        let out = c.on_tick(ts(OPEN_MS + 1200));
        assert!(
            tripped_in(&out).is_empty(),
            "gap within grace must not trip"
        );
        assert!(!has_cancel_all(&out));
        assert!(!c.is_globally_halted());
        // A genuine feed death — 2100 ms of continuous silence — still trips.
        let out = c.on_tick(ts(OPEN_MS + 2100));
        assert_eq!(tripped_in(&out), vec![BreakerKind::FeedStale]);
        assert!(has_cancel_all(&out));
        assert!(c.is_globally_halted());
    }

    #[test]
    fn chainlink_rate_ticks_never_trip_the_fast_timer() {
        // A stream we never tracked as binance-mid (e.g. only chainlink-rate
        // updates ~1/s) must not trip the 500 ms timer: with no binance-mid seen,
        // the timer has nothing to measure.
        let mut c = core();
        open_window(&mut c);
        // 5 seconds pass with no binance-mid ticks at all.
        for k in 1..=5 {
            assert!(
                c.on_tick(ts(OPEN_MS + k * 1000)).is_empty(),
                "no fast stream tracked ⇒ no staleness trip"
            );
        }
        assert!(!c.is_globally_halted());
    }

    #[test]
    fn feed_health_stale_report_trips_and_recovered_clears() {
        let mut c = core();
        let out = c.on_event(
            &Event::FeedHealth(FeedHealth::Stale {
                source: PriceSource::ChainlinkRtds,
                asset: Asset::Btc,
                kind: TickKind::Vendor,
                age: core_types::DurationMs::from_millis(6000),
            }),
            ts(OPEN_MS),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::FeedStale]);
        let out = c.on_event(
            &Event::FeedHealth(FeedHealth::Recovered {
                source: PriceSource::ChainlinkRtds,
                asset: Asset::Btc,
                kind: TickKind::Vendor,
                gap: core_types::DurationMs::from_millis(6500),
            }),
            ts(OPEN_MS + 6500),
        );
        assert_eq!(cleared_in(&out), vec![BreakerKind::FeedStale]);
    }

    // ---- ws disconnect ------------------------------------------------------

    #[test]
    fn market_ws_disconnect_trips_wsdisconnect_and_recovery_clears() {
        let mut c = core();
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Disconnected,
                ts: ts(OPEN_MS),
            }),
            ts(OPEN_MS),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::WsDisconnect]);
        assert!(has_cancel_all(&out));
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Recovered {
                window: window(),
                outage: core_types::DurationMs::from_millis(1000),
                ts: ts(OPEN_MS + 1000),
            }),
            ts(OPEN_MS + 1000),
        );
        assert_eq!(cleared_in(&out), vec![BreakerKind::WsDisconnect]);
    }

    #[test]
    fn user_ws_disconnect_trips_and_reconnect_clears() {
        let mut c = core();
        let out = c.on_venue_event(&VenueEvent::Connectivity { connected: false }, ts(OPEN_MS));
        assert_eq!(tripped_in(&out), vec![BreakerKind::WsDisconnect]);
        let out = c.on_venue_event(
            &VenueEvent::Connectivity { connected: true },
            ts(OPEN_MS + 1),
        );
        assert_eq!(cleared_in(&out), vec![BreakerKind::WsDisconnect]);
    }

    #[test]
    fn book_stale_non_disconnect_trips_feed_stale_not_ws() {
        let mut c = core();
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Stale,
                ts: ts(OPEN_MS),
            }),
            ts(OPEN_MS),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::FeedStale]);
    }

    fn dwell_core(book_staleness_dwell_ms: i64) -> RiskCore {
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        let params = RiskParams {
            book_staleness_dwell_ms,
            ..RiskParams::default()
        };
        RiskCore::new(&params, caps)
    }

    #[test]
    fn book_staleness_dwell_filters_transient_churn_but_sustained_outage_still_trips() {
        // With a 1500 ms dwell, a book that flaps Stale then Recovers within the
        // dwell must NOT trip FeedStale (the six-series rollover churn — snapshot
        // gaps / per-pair reconnects momentarily flag Stale then Recover in well
        // under a second). A book that STAYS stale past the dwell MUST still trip
        // FeedStale at the next on_tick — the alarm is preserved.
        let mut c = dwell_core(1500);
        open_window(&mut c);

        // --- transient: Stale @ OPEN, still stale at +1000 (< dwell), Recovered
        //     at +1200 (< dwell). Never trips. ---
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Stale,
                ts: ts(OPEN_MS),
            }),
            ts(OPEN_MS),
        );
        assert!(
            tripped_in(&out).is_empty(),
            "a fresh book-stale is within the dwell — must not trip"
        );
        assert!(
            c.on_tick(ts(OPEN_MS + 1000)).is_empty(),
            "still within dwell"
        );
        assert!(!c.is_globally_halted());
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Recovered {
                window: window(),
                outage: core_types::DurationMs::from_millis(1200),
                ts: ts(OPEN_MS + 1200),
            }),
            ts(OPEN_MS + 1200),
        );
        assert!(tripped_in(&out).is_empty() && !c.is_globally_halted());
        // A tick past what WOULD have been the dwell: the book already recovered,
        // so nothing lingers to trip.
        assert!(c.on_tick(ts(OPEN_MS + 2000)).is_empty());
        assert!(!c.is_globally_halted(), "transient churn was filtered");

        // --- sustained: Stale @ +3000 that never recovers. Must trip once the
        //     dwell elapses (crossed at the +4500 on_tick). ---
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Stale,
                ts: ts(OPEN_MS + 3000),
            }),
            ts(OPEN_MS + 3000),
        );
        assert!(
            tripped_in(&out).is_empty(),
            "fresh stale-since, dwell not met"
        );
        assert!(
            c.on_tick(ts(OPEN_MS + 4000)).is_empty(),
            "1000 ms stale < 1500 ms dwell"
        );
        assert!(!c.is_globally_halted());
        let out = c.on_tick(ts(OPEN_MS + 4500));
        assert_eq!(
            tripped_in(&out),
            vec![BreakerKind::FeedStale],
            "a sustained book outage past the dwell STILL trips FeedStale"
        );
        assert!(
            has_cancel_all(&out),
            "FeedStale is operational — cancel-all fires"
        );
        assert!(c.is_globally_halted());
    }

    #[test]
    fn resolved_window_teardown_disconnect_does_not_wedge_wsdisconnect() {
        // The rollover-halt regression: a window's CLOB connection tears down at
        // close/resolution, emitting a BookHealth::Unreliable{Disconnected} that
        // never gets a paired Recovered. WsDisconnect must clear when the window
        // ends, and a late disconnect for the dead window must be ignored.
        let mut c = core();
        open_window(&mut c);
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Disconnected,
                ts: ts(CLOSE_MS),
            }),
            ts(CLOSE_MS),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::WsDisconnect]);
        assert!(c.is_globally_halted());
        // The window resolves — its teardown disconnect must no longer hold the
        // global breaker (no Recovered will ever come for a resolved window).
        let out = c.on_event(
            &win_event(WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            }),
            ts(CLOSE_MS + 1),
        );
        assert!(cleared_in(&out).contains(&BreakerKind::WsDisconnect));
        assert!(
            !c.is_globally_halted(),
            "WsDisconnect must clear once the window ends — else a permanent halt"
        );
        // A late disconnect for the now-ended window is ignored (no re-trip).
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Disconnected,
                ts: ts(CLOSE_MS + 2),
            }),
            ts(CLOSE_MS + 2),
        );
        assert!(tripped_in(&out).is_empty());
        assert!(!c.is_globally_halted());
    }

    #[test]
    fn closed_window_stale_book_clears_feed_stale_on_close() {
        // During the post-close linger a closed window's book gets no updates and
        // reports Stale. That must not hold FeedStale once the window is Closed.
        let mut c = core();
        open_window(&mut c);
        let out = c.on_event(
            &Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Stale,
                ts: ts(CLOSE_MS),
            }),
            ts(CLOSE_MS),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::FeedStale]);
        let out = c.on_event(&win_event(WindowLifecycle::Closed), ts(CLOSE_MS));
        assert!(cleared_in(&out).contains(&BreakerKind::FeedStale));
        assert!(!c.is_globally_halted());
    }

    // ---- clock skew (honored, not re-published) ----------------------------

    #[test]
    fn clock_skew_is_forwarded_not_republished() {
        let mut c = core();
        let out = c.on_event(
            &Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew,
            }),
            ts(OPEN_MS),
        );
        // Forwarded to strategies, cancel-all issued, but NOT announced/published.
        assert!(out.iter().any(|o| matches!(
            o,
            RiskOutput::Forward(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew
            })
        )));
        assert!(has_cancel_all(&out));
        assert!(
            tripped_in(&out).is_empty(),
            "must not re-announce ClockSkew"
        );
        assert!(
            !out.iter()
                .any(|o| matches!(o, RiskOutput::Publish(_) | RiskOutput::Announce(_))),
            "no bus publication for the externally-minted ClockSkew"
        );
        assert!(c.is_globally_halted());
        let out = c.on_event(
            &Event::Risk(RiskEvent::BreakerCleared {
                breaker: BreakerKind::ClockSkew,
            }),
            ts(OPEN_MS + 1),
        );
        assert!(
            out.iter()
                .any(|o| matches!(o, RiskOutput::Forward(RiskEvent::BreakerCleared { .. })))
        );
        assert!(!c.is_globally_halted());
    }

    // ---- per-window loss ----------------------------------------------------

    #[test]
    fn window_loss_halts_that_window_and_resolves_to_clear() {
        let mut c = core();
        open_window(&mut c);
        // 100 Up shares @ 0.40 = $40 worst-case vs the $25 series cap.
        let out = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(100)),
            ts(OPEN_MS),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::WindowLoss]);
        assert!(
            out.iter().any(|o| matches!(o, RiskOutput::CancelMarket(_))),
            "the loss-halted window is cancel-market'd"
        );
        assert!(c.halted_windows().contains(&window()));
        assert!(
            !c.is_globally_halted(),
            "WindowLoss is per-window, not global"
        );
        // Resolution clears the per-window halt.
        let out = c.on_event(
            &win_event(WindowLifecycle::Resolved {
                outcome: Outcome::Down,
            }),
            ts(CLOSE_MS),
        );
        assert!(cleared_in(&out).contains(&BreakerKind::WindowLoss));
        assert!(c.halted_windows().is_empty());
    }

    // ---- daily stop (latched until reset) ----------------------------------

    #[test]
    fn daily_stop_trips_on_realized_loss_and_only_reset_clears() {
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        let params = RiskParams {
            daily_stop_loss: Dollars::new(Decimal::from(5)),
            ..RiskParams::default()
        };
        let mut c = RiskCore::new(&params, caps);
        open_window(&mut c);
        // Buy 50 Up @ 0.40 = -$20 cash; resolve Down ⇒ realized -$20 ≤ -$5.
        let _ = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(50)),
            ts(OPEN_MS),
        );
        let out = c.on_event(
            &win_event(WindowLifecycle::Resolved {
                outcome: Outcome::Down,
            }),
            ts(CLOSE_MS),
        );
        assert!(tripped_in(&out).contains(&BreakerKind::DailyStop));
        assert!(c.is_globally_halted());
        // A new window/settlement does NOT clear it (latched).
        assert!(c.is_globally_halted());
        // Only Reset clears it.
        let out = c.on_event(&Event::Control(ControlEvent::Reset), ts(CLOSE_MS + 1000));
        assert!(cleared_in(&out).contains(&BreakerKind::DailyStop));
        assert!(!c.is_globally_halted());
    }

    #[test]
    fn daily_stop_reset_clears_only_daily_stop_leaving_manual() {
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        let params = RiskParams {
            daily_stop_loss: Dollars::new(Decimal::from(5)),
            ..RiskParams::default()
        };
        let mut c = RiskCore::new(&params, caps);
        open_window(&mut c);
        // Trip both a manual kill and the daily stop.
        let _ = c.on_event(&Event::Control(ControlEvent::Kill), ts(OPEN_MS));
        let _ = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(50)),
            ts(OPEN_MS),
        );
        let _ = c.on_event(
            &win_event(WindowLifecycle::Resolved {
                outcome: Outcome::Down,
            }),
            ts(CLOSE_MS),
        );
        assert!(c.snapshot().is_tripped(BreakerKind::DailyStop));
        assert!(c.snapshot().is_tripped(BreakerKind::Manual));
        // The targeted reset clears DailyStop only; Manual stays latched.
        let out = c.on_event(
            &Event::Control(ControlEvent::DailyStopReset),
            ts(CLOSE_MS + 1),
        );
        assert_eq!(cleared_in(&out), vec![BreakerKind::DailyStop]);
        assert!(!c.snapshot().is_tripped(BreakerKind::DailyStop));
        assert!(c.snapshot().is_tripped(BreakerKind::Manual));
        assert!(c.is_globally_halted(), "manual kill still halts");
    }

    // ---- shadow loss stops (paper eval) ------------------------------------

    #[test]
    fn shadow_daily_stop_records_but_does_not_halt() {
        let mut c = shadow_core(Decimal::from(5));
        open_window(&mut c);
        // Buy 50 Up @ 0.40 (−$20 cash, worst-case $20 < $25 cap ⇒ no window shadow);
        // resolve Down ⇒ realized −$20 ≤ −$5.
        let _ = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(50)),
            ts(OPEN_MS),
        );
        let out = c.on_event(
            &win_event(WindowLifecycle::Resolved {
                outcome: Outcome::Down,
            }),
            ts(CLOSE_MS),
        );
        // A DailyStop shadow is recorded — but nothing tripped or cancelled.
        assert_eq!(shadow_kinds(&out), vec![BreakerKind::DailyStop]);
        assert!(!c.snapshot().is_tripped(BreakerKind::DailyStop));
        assert!(!c.is_globally_halted());
        assert!(!has_cancel_all(&out));
        assert!(
            tripped_in(&out).is_empty(),
            "no hard breaker announced under shadow"
        );
        let snap = c.snapshot();
        assert_eq!(snap.shadow_tripped.len(), 1);
        assert_eq!(snap.shadow_tripped[0].kind, BreakerKind::DailyStop);
        assert_eq!(snap.shadow_tripped[0].window, None);
    }

    #[test]
    fn shadow_window_loss_records_but_does_not_halt() {
        let mut c = shadow_core(Decimal::from(200));
        open_window(&mut c);
        // 100 Up @ 0.40 = $40 worst-case vs the $25 series cap.
        let out = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(100)),
            ts(OPEN_MS),
        );
        assert_eq!(shadow_kinds(&out), vec![BreakerKind::WindowLoss]);
        assert!(
            c.halted_windows().is_empty(),
            "the window is NOT halted under shadow"
        );
        assert!(!has_cancel_market(&out), "no cancel-market under shadow");
        assert!(!c.is_globally_halted());
        // Latched: a further over-cap fill on the same window does not re-record.
        let out2 = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(10)),
            ts(OPEN_MS + 1),
        );
        assert!(
            shadow_kinds(&out2).is_empty(),
            "the window-loss shadow fires once per crossing"
        );
        let snap = c.snapshot();
        assert_eq!(snap.shadow_tripped.len(), 1);
        assert_eq!(snap.shadow_tripped[0].kind, BreakerKind::WindowLoss);
        assert_eq!(snap.shadow_tripped[0].window, Some(window()));
        // Resolution prunes the recorded shadow (bounded memory).
        let _ = c.on_event(
            &win_event(WindowLifecycle::Resolved {
                outcome: Outcome::Down,
            }),
            ts(CLOSE_MS),
        );
        assert!(c.snapshot().shadow_tripped.is_empty());
    }

    #[test]
    fn operational_breakers_still_hard_under_shadow() {
        let mut c = shadow_core(Decimal::from(200));
        open_window(&mut c);
        // Fast-feed staleness still hard-halts + cancels under shadow.
        let _ = c.on_event(&binance_mid(dec!(60000), OPEN_MS), ts(OPEN_MS));
        let out = c.on_tick(ts(OPEN_MS + 600));
        assert_eq!(tripped_in(&out), vec![BreakerKind::FeedStale]);
        assert!(has_cancel_all(&out));
        assert!(c.is_globally_halted());
        assert!(
            shadow_kinds(&out).is_empty(),
            "operational breakers are never shadowed"
        );
        // A manual kill also hard-halts under shadow.
        let out = c.on_event(&Event::Control(ControlEvent::Kill), ts(OPEN_MS + 700));
        assert!(tripped_in(&out).contains(&BreakerKind::Manual));
        assert!(has_cancel_all(&out));
    }

    // ---- sanity (fair vs mid) ----------------------------------------------

    #[test]
    fn fair_vs_mid_trips_after_sustained_divergence_and_clears_when_back() {
        let mut c = core();
        open_window(&mut c);
        // mid ~0.50, p_up 0.85 ⇒ |diff| 0.35 > 0.10.
        let _ = c.on_event(
            &top_event(up_token(), dec!(0.49), dec!(0.51), OPEN_MS),
            ts(OPEN_MS),
        );
        let out = c.on_event(&model(0.85, OPEN_MS), ts(OPEN_MS));
        assert!(tripped_in(&out).is_empty(), "not yet — dwell not elapsed");
        // Still diverged 3 s later → trips.
        let out = c.on_tick(ts(OPEN_MS + 3000));
        assert_eq!(tripped_in(&out), vec![BreakerKind::FairVsMid]);
        // Back within bound (p_up ~0.50) clears.
        let out = c.on_event(&model(0.50, OPEN_MS + 3500), ts(OPEN_MS + 3500));
        assert_eq!(cleared_in(&out), vec![BreakerKind::FairVsMid]);
    }

    #[test]
    fn transient_divergence_under_the_dwell_does_not_trip() {
        let mut c = core();
        open_window(&mut c);
        let _ = c.on_event(
            &book_event(up_token(), dec!(0.49), dec!(0.51), OPEN_MS),
            ts(OPEN_MS),
        );
        let _ = c.on_event(&model(0.85, OPEN_MS), ts(OPEN_MS));
        // 2 s later still diverged (under the 3 s dwell), then recovers.
        assert!(c.on_tick(ts(OPEN_MS + 2000)).is_empty());
        let out = c.on_event(&model(0.50, OPEN_MS + 2500), ts(OPEN_MS + 2500));
        assert!(tripped_in(&out).is_empty());
        assert!(!c.is_globally_halted());
    }

    /// Per-window FairVsMid: window A diverging past the dwell trips the GLOBAL
    /// halt, and a perfectly healthy sibling window B must NOT clear it while A
    /// stays diverged. Only A recovering clears the global breaker.
    #[test]
    fn second_window_divergence_trips_and_healthy_sibling_does_not_clear() {
        let mut c = core();
        open_window(&mut c); // window A (window(), up_token())

        // Open a second window B: same series, later open, its own up token.
        let win_b = WindowId {
            series: series(),
            open_time: ts(OPEN_MS + 300_000),
        };
        let up_b = TokenId::new("333").unwrap();
        let mut m_b = (*market()).clone();
        m_b.window = win_b;
        m_b.tokens = TokenPair {
            up: up_b.clone(),
            down: TokenId::new("444").unwrap(),
        };
        m_b.close_time = ts(OPEN_MS + 600_000);
        let _ = c.on_event(
            &Event::Window {
                market: Arc::new(m_b),
                lifecycle: WindowLifecycle::Open,
            },
            ts(OPEN_MS + 300_000),
        );

        // Window A diverges (mid ~0.50, p_up 0.85) and stays diverged past the dwell.
        let _ = c.on_event(
            &top_event(up_token(), dec!(0.49), dec!(0.51), OPEN_MS),
            ts(OPEN_MS),
        );
        let _ = c.on_event(&model(0.85, OPEN_MS), ts(OPEN_MS));
        let out = c.on_tick(ts(OPEN_MS + 3000));
        assert_eq!(tripped_in(&out), vec![BreakerKind::FairVsMid]);
        assert!(c.is_globally_halted());

        // Window B is perfectly healthy (mid ≈ p_up) — it must NOT clear the halt.
        let _ = c.on_event(
            &top_event(up_b.clone(), dec!(0.49), dec!(0.51), OPEN_MS + 300_100),
            ts(OPEN_MS + 300_100),
        );
        let out = c.on_event(
            &model_for(win_b, 0.50, OPEN_MS + 300_100),
            ts(OPEN_MS + 300_100),
        );
        assert!(
            cleared_in(&out).is_empty(),
            "a healthy sibling must not clear window A's halt"
        );
        assert!(c.is_globally_halted(), "A still diverged ⇒ still halted");

        // A recovers → the global FairVsMid clears.
        let out = c.on_event(&model(0.50, OPEN_MS + 3500), ts(OPEN_MS + 3500));
        assert_eq!(cleared_in(&out), vec![BreakerKind::FairVsMid]);
        assert!(!c.is_globally_halted());
    }

    fn two_tier_core(
        bound_slow: f64,
        dwell_slow: i64,
        bound_fast: f64,
        dwell_fast: i64,
    ) -> RiskCore {
        let mut caps = HashMap::new();
        caps.insert(series(), Dollars::new(Decimal::from(25)));
        let params = RiskParams {
            sanity_bound: bound_slow,
            sanity_bound_duration_ms: dwell_slow,
            sanity_bound_fast: bound_fast,
            sanity_bound_duration_fast_ms: dwell_fast,
            ..RiskParams::default()
        };
        RiskCore::new(&params, caps)
    }

    #[test]
    fn two_tier_fast_catches_catastrophic_fast_slow_waits_for_moderate() {
        // Slow tier 0.10 / 23s (sanity net for lead-lag), fast 0.30 / 3s
        // (catastrophic). A MODERATE gap (0.15, above slow, below fast) must wait
        // the long slow dwell; a CATASTROPHIC gap (>0.30) trips on the short fast
        // dwell — caught in seconds, not 23s.
        let mut c = two_tier_core(0.10, 23_000, 0.30, 3_000);
        open_window(&mut c);
        // Moderate: mid ~0.50, p_up 0.65 ⇒ |div| 0.15 (in (0.10, 0.30]).
        let _ = c.on_event(
            &top_event(up_token(), dec!(0.49), dec!(0.51), OPEN_MS),
            ts(OPEN_MS),
        );
        let _ = c.on_event(&model(0.65, OPEN_MS), ts(OPEN_MS));
        // Past the 3s FAST dwell but under the 23s SLOW dwell → NOT tripped
        // (moderate gaps do not use the fast tier).
        assert!(c.on_tick(ts(OPEN_MS + 5_000)).is_empty());
        assert!(!c.is_globally_halted(), "0.15 gap must wait the slow dwell");
        // Crosses the slow dwell → trips.
        let out = c.on_tick(ts(OPEN_MS + 23_000));
        assert_eq!(tripped_in(&out), vec![BreakerKind::FairVsMid]);
        // Recover.
        let _ = c.on_event(&model(0.50, OPEN_MS + 24_000), ts(OPEN_MS + 24_000));

        // Catastrophic: p_up 0.95 vs mid 0.50 ⇒ |div| 0.45 > 0.30 fast bound.
        let mut c = two_tier_core(0.10, 23_000, 0.30, 3_000);
        open_window(&mut c);
        let _ = c.on_event(
            &top_event(up_token(), dec!(0.49), dec!(0.51), OPEN_MS),
            ts(OPEN_MS),
        );
        let _ = c.on_event(&model(0.95, OPEN_MS), ts(OPEN_MS));
        // Under the 3s fast dwell → not yet.
        assert!(c.on_tick(ts(OPEN_MS + 2_000)).is_empty());
        // Past the 3s FAST dwell (well under the 23s slow dwell) → trips.
        let out = c.on_tick(ts(OPEN_MS + 3_000));
        assert_eq!(
            tripped_in(&out),
            vec![BreakerKind::FairVsMid],
            "a catastrophic gap must trip on the short fast dwell, not wait 23s"
        );
    }

    // ---- error rate ---------------------------------------------------------

    #[test]
    fn error_rate_trips_at_threshold_and_clears_when_window_drains() {
        let mut c = core();
        // 9 infra errors — under the 10 threshold.
        let obs: Vec<GuardObservation> = (0..9)
            .map(|i| GuardObservation::InfraError(ts(OPEN_MS + i)))
            .collect();
        let out = c.fold_observations(&obs, ts(OPEN_MS + 10));
        assert!(tripped_in(&out).is_empty());
        // The 10th trips ErrorRate.
        let out = c.fold_observations(
            &[GuardObservation::InfraError(ts(OPEN_MS + 11))],
            ts(OPEN_MS + 11),
        );
        assert_eq!(tripped_in(&out), vec![BreakerKind::ErrorRate]);
        // After the 60 s window passes with no new errors, it clears.
        let out = c.on_tick(ts(OPEN_MS + 70_000));
        assert_eq!(cleared_in(&out), vec![BreakerKind::ErrorRate]);
    }

    // ---- engine restart -----------------------------------------------------

    #[test]
    fn engine_restart_trips_on_observation_and_clears_after_cooldown() {
        let mut c = core();
        let out = c.fold_observations(&[GuardObservation::EngineRestart(ts(OPEN_MS))], ts(OPEN_MS));
        assert_eq!(tripped_in(&out), vec![BreakerKind::EngineRestart]);
        assert!(has_cancel_all(&out));
        // Within the 5 s cooldown — stays tripped.
        assert!(c.on_tick(ts(OPEN_MS + 3000)).is_empty());
        assert!(c.is_globally_halted());
        // Past the cooldown — clears.
        let out = c.on_tick(ts(OPEN_MS + 6000));
        assert_eq!(cleared_in(&out), vec![BreakerKind::EngineRestart]);
        assert!(!c.is_globally_halted());
    }

    // ---- manual kill --------------------------------------------------------

    #[test]
    fn manual_kill_trips_and_reset_clears() {
        let mut c = core();
        let out = c.on_event(&Event::Control(ControlEvent::Kill), ts(OPEN_MS));
        assert_eq!(tripped_in(&out), vec![BreakerKind::Manual]);
        assert!(has_cancel_all(&out));
        assert!(c.is_globally_halted());
        let out = c.on_event(&Event::Control(ControlEvent::Reset), ts(OPEN_MS + 1));
        assert_eq!(cleared_in(&out), vec![BreakerKind::Manual]);
        assert!(!c.is_globally_halted());
    }

    // ---- open notional ------------------------------------------------------

    #[test]
    fn open_notional_tracks_the_venue_stream() {
        let mut c = core();
        // Open at 0.50 × 100 = $50.
        let _ = c.on_venue_event(
            &order_update("o1", OrderState::Open, dec!(0.50), dec!(100)),
            ts(OPEN_MS),
        );
        assert_eq!(c.open_notional(), Dollars::new(dec!(50)));
        // A second resting order adds.
        let _ = c.on_venue_event(
            &order_update("o2", OrderState::Open, dec!(0.40), dec!(50)),
            ts(OPEN_MS),
        );
        assert_eq!(c.open_notional(), Dollars::new(dec!(70)));
        // o1 cancels → removed.
        let _ = c.on_venue_event(
            &order_update("o1", OrderState::Canceled, dec!(0.50), dec!(100)),
            ts(OPEN_MS),
        );
        assert_eq!(c.open_notional(), Dollars::new(dec!(20)));
    }

    // ---- snapshot -----------------------------------------------------------

    #[test]
    fn snapshot_reflects_tripped_breakers() {
        let mut c = core();
        assert!(!c.snapshot().any_tripped());
        let _ = c.on_event(&Event::Control(ControlEvent::Kill), ts(OPEN_MS));
        let snap = c.snapshot();
        assert!(snap.any_tripped());
        assert!(snap.is_tripped(BreakerKind::Manual));
        assert!(snap.globally_halted);
    }

    #[test]
    fn trips_last_hour_counts_recent_global_trips() {
        let mut c = core();
        open_window(&mut c);
        assert_eq!(c.snapshot().trips_last_hour, 0);
        // Two distinct global trips: WsDisconnect (user-WS down) then Manual.
        let _ = c.on_venue_event(&VenueEvent::Connectivity { connected: false }, ts(OPEN_MS));
        let _ = c.on_event(&Event::Control(ControlEvent::Kill), ts(OPEN_MS + 1000));
        assert_eq!(c.snapshot().trips_last_hour, 2);
        assert_eq!(c.trips_last_hour(ts(OPEN_MS + 1000)), 2);
        // A tick more than an hour past both trips ages the trip *events* out of the
        // ring (the latched breakers stay tripped; only the rate resets).
        let _ = c.on_tick(ts(OPEN_MS + 3_600_002 + 1000));
        assert_eq!(c.snapshot().trips_last_hour, 0);
    }

    #[test]
    fn snapshot_lists_the_halted_windows() {
        let mut c = core();
        open_window(&mut c);
        // 100 Up @ 0.40 = $40 worst-case vs the $25 series cap ⇒ window loss halt.
        let _ = c.on_venue_event(
            &fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(100)),
            ts(OPEN_MS),
        );
        let snap = c.snapshot();
        assert_eq!(snap.halted_windows, 1);
        assert_eq!(snap.halted_windows_list, vec![window()]);
        // Window loss is per-window, not a global trip, so it does not add a rate.
        assert_eq!(snap.trips_last_hour, 0);
    }

    #[test]
    fn multiple_breakers_compose_and_clear_independently() {
        let mut c = core();
        let _ = c.on_event(&Event::Control(ControlEvent::Kill), ts(OPEN_MS));
        let _ = c.on_venue_event(&VenueEvent::Connectivity { connected: false }, ts(OPEN_MS));
        assert!(c.snapshot().tripped.contains(&BreakerKind::Manual));
        assert!(c.snapshot().tripped.contains(&BreakerKind::WsDisconnect));
        // Reconnect clears only WsDisconnect; Manual remains.
        let _ = c.on_venue_event(
            &VenueEvent::Connectivity { connected: true },
            ts(OPEN_MS + 1),
        );
        assert!(!c.snapshot().is_tripped(BreakerKind::WsDisconnect));
        assert!(c.snapshot().is_tripped(BreakerKind::Manual));
        assert!(c.is_globally_halted());
    }
}
