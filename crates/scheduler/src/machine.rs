//! The sans-IO per-series state machine (CLAUDE.md §6).
//!
//! Pure by construction: no clocks, no I/O, no logging, no panics — every
//! observable behavior is an [`Output`] value the driver acts on, so the
//! deterministic tests assert on values, not log lines.
//!
//! The wall clock (fed in as `now`) rolls windows against discovery-supplied
//! times; discovery results and market-channel events are redundant inputs
//! that can only accelerate or annotate the cycle, never stall it. Windows
//! are contiguous back-to-back, so at close the machine rolls immediately and
//! parks the closed window in an awaiting-resolution set — `market_resolved`
//! (or a loud timeout) settles it later without ever blocking coverage.

use std::cmp::max;
use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use core_types::{
    DurationMs, MarketInfo, MarketLifecycleEvent, Outcome, Series, TimestampMs, WindowId,
    WindowLifecycle,
};
use discovery::SeriesWindows;

use crate::backoff::Backoff;
use crate::status::SeriesStatus;

/// An in-flight refresh with no reply after this long is presumed lost (a
/// wedged worker, a dropped request) and retried. Comfortably above the
/// worst-case refresh HTTP time (~4 sequential calls at a 5 s timeout each).
const REFRESH_INFLIGHT_TIMEOUT: DurationMs = DurationMs::from_millis(45_000);

/// Minimum spacing between `new_market`-triggered refresh requests, so a
/// burst of venue announcements cannot turn into a request storm.
const MIN_REFRESH_SPACING: DurationMs = DurationMs::from_millis(1_000);

/// Cap on closed windows parked awaiting resolution. Oldest is evicted past
/// this — resolution PnL attribution is the venue ledger's job; the machine
/// only tracks enough to map late `market_resolved` events.
const MAX_PARKED: usize = 8;

/// Timing knobs, copied out of [`config::SchedulerConfig`] at construction so
/// machine tests need no config plumbing. Field meanings match the config
/// section one-to-one.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// Verify/refresh the next window this long before the current close.
    pub refresh_lead: DurationMs,
    /// §6 contract: next window known at least this long before close.
    pub next_window_lead: DurationMs,
    /// Announce `Closing` this long before close.
    pub closing_lead: DurationMs,
    /// Retry backoff initial delay.
    pub retry_initial: DurationMs,
    /// Retry backoff cap.
    pub retry_max: DurationMs,
    /// Parked window without `market_resolved` warns after this.
    pub resolution_timeout: DurationMs,
    /// Heartbeat: refresh even when everything is known past this age.
    pub max_refresh_interval: DurationMs,
    /// Whether a market-event source is attached. `false` (pre-feed-clob)
    /// suppresses overdue-resolution and eviction warnings — windows still
    /// park, but nobody is expected to resolve them yet.
    pub expect_resolutions: bool,
}

impl Timing {
    /// Builds from the validated config section. `expect_resolutions` starts
    /// `false`; the driver flips it on when a market-event channel is
    /// actually attached.
    #[must_use]
    pub fn from_config(cfg: &config::SchedulerConfig) -> Self {
        Self {
            refresh_lead: cfg.refresh_lead_ms,
            next_window_lead: cfg.next_window_lead_ms,
            closing_lead: cfg.closing_lead_ms,
            retry_initial: cfg.retry_initial_backoff_ms,
            retry_max: cfg.retry_max_backoff_ms,
            resolution_timeout: cfg.resolution_timeout_ms,
            max_refresh_interval: cfg.max_refresh_interval_ms,
            expect_resolutions: false,
        }
    }
}

/// Why the machine asked the driver for a discovery refresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshReason {
    /// First refresh after construction.
    Boot,
    /// A window just opened — top up the lookahead.
    Rolled,
    /// Scheduled verification ahead of the current close (§6).
    PreClose,
    /// Backoff-paced retry after a failure or unsatisfying result.
    Retry,
    /// Nothing was needed, but the last success is older than the heartbeat.
    Heartbeat,
    /// A window closed with no known successor — coverage gap, hunt hard.
    GapRecovery,
    /// An unknown `new_market` event hinted that the venue created windows
    /// we have not seen.
    NewMarketHint,
}

/// Loud coverage anomalies. The driver logs these (WARN/ERROR); tests assert
/// on the values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageWarning {
    /// `close − next_window_lead` passed with no known next window: the §6
    /// contract is violated (emitted once per window).
    ContractViolated {
        /// The current window whose successor is missing.
        window: WindowId,
        /// Its close time.
        close: TimestampMs,
    },
    /// The window closed and no successor was known — a coverage gap began.
    NoNextAtClose {
        /// The window that closed.
        window: WindowId,
    },
    /// Parked longer than `resolution_timeout` without `market_resolved`.
    ResolutionOverdue {
        /// The unresolved window.
        window: WindowId,
        /// When it closed.
        closed_at: TimestampMs,
    },
    /// `market_resolved` arrived for a window the clock says is still open
    /// (or not yet open) — the venue and our clock disagree.
    ResolutionBeforeClose {
        /// The window in question.
        window: WindowId,
    },
    /// `market_resolved` matched a window but the winning token id is
    /// neither of its pair — refusing to fabricate an outcome.
    UnknownWinningToken {
        /// The window in question.
        window: WindowId,
    },
    /// Discovery reported a different current window than the one we were
    /// trading — trusted discovery, rolled ours.
    CurrentSuperseded {
        /// The window we were on.
        old: WindowId,
        /// The window discovery says is current.
        new: WindowId,
    },
    /// The parked set hit its cap; the oldest unresolved window was evicted.
    ParkedEvicted {
        /// The evicted window.
        window: WindowId,
    },
}

/// Everything the machine wants the outside world to do, in emission order.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// Forward to the bus as [`core_types::Event::Window`].
    Announce {
        /// The window's market.
        market: Arc<MarketInfo>,
        /// The lifecycle phase being announced.
        lifecycle: WindowLifecycle,
    },
    /// Run a discovery refresh for this machine's series.
    Refresh {
        /// Why.
        reason: RefreshReason,
    },
    /// Log loudly; never fatal.
    Warn(CoverageWarning),
}

/// Lifecycle phase of the series (not of one window — windows flow through).
#[derive(Debug, Clone)]
enum Phase {
    /// Nothing tradable known; a refresh is in flight or backoff-pending.
    Discovering,
    /// Next window known and announced; waiting for its open time.
    Pending {
        /// The next window to open.
        market: Arc<MarketInfo>,
    },
    /// A window is open and outside its closing zone.
    Active {
        /// The open window.
        market: Arc<MarketInfo>,
    },
    /// Inside the final `closing_lead` before close.
    Closing {
        /// The open window.
        market: Arc<MarketInfo>,
    },
}

/// A closed window awaiting `market_resolved`.
#[derive(Debug, Clone)]
struct Parked {
    market: Arc<MarketInfo>,
    closed_at: TimestampMs,
    warned: bool,
}

/// The per-series state machine. See the module docs for the contract; see
/// [`Output`] for everything it can ask of the driver.
#[derive(Debug)]
pub struct SeriesMachine {
    series: Series,
    timing: Timing,
    phase: Phase,
    /// Future windows strictly after the phase's window, soonest first.
    /// Replaced wholesale on every successful refresh.
    upcoming: VecDeque<Arc<MarketInfo>>,
    /// Closed windows awaiting resolution, oldest first (capped).
    parked: Vec<Parked>,
    /// Windows already announced as `Discovered` (pruned once past close —
    /// discovery can never return those again).
    announced: BTreeSet<WindowId>,
    refresh_inflight: bool,
    refresh_requested_at: Option<TimestampMs>,
    last_refresh_request: Option<TimestampMs>,
    /// The earliest scheduled future refresh, if any.
    next_refresh: Option<(TimestampMs, RefreshReason)>,
    last_refresh_ok: Option<TimestampMs>,
    backoff: Backoff,
    /// `ContractViolated` fires once per window.
    contract_warned_for: Option<WindowId>,
}

impl SeriesMachine {
    /// A machine in `Discovering`, due for a [`RefreshReason::Boot`] refresh
    /// on its first tick.
    #[must_use]
    pub fn new(series: Series, timing: Timing) -> Self {
        Self {
            series,
            timing,
            phase: Phase::Discovering,
            upcoming: VecDeque::new(),
            parked: Vec::new(),
            announced: BTreeSet::new(),
            refresh_inflight: false,
            refresh_requested_at: None,
            last_refresh_request: None,
            // Sentinel "due immediately": any real now is later than this.
            next_refresh: Some((TimestampMs::from_millis(i64::MIN), RefreshReason::Boot)),
            last_refresh_ok: None,
            backoff: Backoff::new(timing.retry_initial, timing.retry_max),
            contract_warned_for: None,
        }
    }

    /// Which series this machine schedules.
    #[must_use]
    pub fn series(&self) -> Series {
        self.series
    }

    /// Wall-clock input. A catch-up loop: processes **every** due boundary in
    /// time order, so an oversleep across open/closing/close still emits the
    /// full `Open`/`Closing`/`Closed`(+roll) sequence in one call. Idempotent
    /// when nothing is due.
    pub fn on_tick(&mut self, now: TimestampMs, out: &mut Vec<Output>) {
        self.catch_up(now, out);

        // Watchdog: a refresh that never came back is retried, so a lost
        // request or wedged worker cannot freeze the series forever.
        if self.refresh_inflight
            && let Some(at) = self.refresh_requested_at
            && now.signed_duration_since(at) >= REFRESH_INFLIGHT_TIMEOUT
        {
            self.refresh_inflight = false;
            self.refresh_requested_at = None;
            self.schedule_refresh(now, RefreshReason::Retry);
        }

        // Due scheduled refresh.
        if !self.refresh_inflight
            && let Some((at, reason)) = self.next_refresh
            && at <= now
        {
            self.next_refresh = None;
            self.emit_refresh(now, reason, out);
        }

        // Overdue resolutions (only meaningful once a resolution source is
        // attached).
        if self.timing.expect_resolutions {
            for p in &mut self.parked {
                if !p.warned
                    && now.signed_duration_since(p.closed_at) >= self.timing.resolution_timeout
                {
                    p.warned = true;
                    out.push(Output::Warn(CoverageWarning::ResolutionOverdue {
                        window: p.market.window,
                        closed_at: p.closed_at,
                    }));
                }
            }
        }

        self.check_contract(now, out);
    }

    /// Discovery result for this series. `None` = the refresh failed (the
    /// driver already logged the error detail); the machine only paces the
    /// retry.
    pub fn on_discovery(
        &mut self,
        result: Option<SeriesWindows>,
        now: TimestampMs,
        out: &mut Vec<Output>,
    ) {
        self.refresh_inflight = false;
        self.refresh_requested_at = None;

        let Some(windows) = result else {
            let delay = self.backoff.next_delay();
            self.schedule_refresh(now.saturating_add(delay), RefreshReason::Retry);
            return;
        };
        self.last_refresh_ok = Some(now);

        // Re-classify against *our* now: HTTP latency can age a snapshot, so
        // a window can have closed (or opened) since discovery classified it.
        let mut all: Vec<Arc<MarketInfo>> = windows
            .current
            .into_iter()
            .chain(windows.upcoming)
            .filter(|m| m.close_time > now)
            .collect();
        all.sort_by_key(|m| m.window.open_time);

        for m in &all {
            if self.announced.insert(m.window) {
                out.push(Output::Announce {
                    market: Arc::clone(m),
                    lifecycle: WindowLifecycle::Discovered,
                });
            }
        }
        self.prune_announced(now);

        // Mirrors discovery::classify_windows: earliest close wins should
        // windows ever overlap.
        let mut current: Option<Arc<MarketInfo>> = None;
        let mut rest: VecDeque<Arc<MarketInfo>> = VecDeque::new();
        for m in all {
            if m.window.open_time <= now {
                if current.as_ref().is_none_or(|c| m.close_time < c.close_time) {
                    current = Some(m);
                }
            } else {
                rest.push_back(m);
            }
        }

        match (&self.phase, current) {
            // Same window we already trade: adopt the fresher metadata (tick
            // size may have flipped), keep the phase kind.
            (Phase::Active { market }, Some(c)) if market.window == c.window => {
                self.phase = Phase::Active { market: c };
            }
            (Phase::Closing { market }, Some(c)) if market.window == c.window => {
                self.phase = Phase::Closing { market: c };
            }
            // Discovery + clock disagree with our current: trust them, roll
            // ours out.
            (Phase::Active { market } | Phase::Closing { market }, Some(c)) => {
                let old = Arc::clone(market);
                out.push(Output::Warn(CoverageWarning::CurrentSuperseded {
                    old: old.window,
                    new: c.window,
                }));
                out.push(Output::Announce {
                    market: Arc::clone(&old),
                    lifecycle: WindowLifecycle::Closed,
                });
                self.park(old, now, out);
                self.enter_active(c, now, out, false);
            }
            // We trade a window discovery no longer returns, but the clock
            // says it is still open — keep it; it rolls on time regardless.
            (Phase::Active { .. } | Phase::Closing { .. }, None) => {}
            (Phase::Discovering | Phase::Pending { .. }, Some(c)) => {
                self.enter_active(c, now, out, false);
            }
            (Phase::Discovering | Phase::Pending { .. }, None) => {
                self.phase = Phase::Discovering;
            }
        }
        self.upcoming = rest;
        if matches!(self.phase, Phase::Discovering)
            && let Some(next) = self.upcoming.pop_front()
        {
            self.phase = Phase::Pending { market: next };
        }

        // A freshly adopted current may already be inside its closing zone.
        self.catch_up(now, out);

        // This result supersedes whatever was scheduled before it (a queued
        // roll top-up, a stale pre-close) — recompute the schedule from
        // scratch so a satisfied need cannot fire a redundant refresh.
        self.next_refresh = None;
        if self.needs_window_data() {
            // Discovery succeeded but did not give us what §6 requires (the
            // venue has not created the next window yet) — keep hunting at
            // backoff pace, do NOT reset, or a long gap would hammer Gamma
            // at the initial delay forever.
            let delay = self.backoff.next_delay();
            self.schedule_refresh(now.saturating_add(delay), RefreshReason::Retry);
        } else {
            self.backoff.reset();
            self.schedule_steady_state_refresh(now);
        }
    }

    /// Market-channel input. Returns `true` when this machine recognized the
    /// condition id (the driver offers each event to every machine; an
    /// unclaimed `market_resolved` is logged at debug and dropped).
    pub fn on_market_event(
        &mut self,
        ev: &MarketLifecycleEvent,
        now: TimestampMs,
        out: &mut Vec<Output>,
    ) -> bool {
        match ev {
            MarketLifecycleEvent::MarketResolved {
                condition_id,
                winning_token,
                ..
            } => {
                // Normal path: resolution for an already-rolled window.
                if let Some(idx) = self
                    .parked
                    .iter()
                    .position(|p| &p.market.condition_id == condition_id)
                {
                    match self.parked[idx].market.tokens.outcome_of(winning_token) {
                        Some(outcome) => {
                            let p = self.parked.remove(idx);
                            out.push(Output::Announce {
                                market: p.market,
                                lifecycle: WindowLifecycle::Resolved { outcome },
                            });
                        }
                        None => {
                            out.push(Output::Warn(CoverageWarning::UnknownWinningToken {
                                window: self.parked[idx].market.window,
                            }));
                        }
                    }
                    return true;
                }
                // Our current window: the venue resolved it while our clock
                // says it is open — warn and force the close path now.
                if let Phase::Active { market } | Phase::Closing { market } = &self.phase
                    && &market.condition_id == condition_id
                {
                    let market = Arc::clone(market);
                    out.push(Output::Warn(CoverageWarning::ResolutionBeforeClose {
                        window: market.window,
                    }));
                    let resolved = market.tokens.outcome_of(winning_token);
                    if resolved.is_none() {
                        out.push(Output::Warn(CoverageWarning::UnknownWinningToken {
                            window: market.window,
                        }));
                    }
                    self.roll_closed(&market, now, out, resolved);
                    return true;
                }
                // A window that never opened resolving is venue weirdness:
                // drop it and let refresh sort out reality.
                if let Phase::Pending { market } = &self.phase
                    && &market.condition_id == condition_id
                {
                    let window = market.window;
                    out.push(Output::Warn(CoverageWarning::ResolutionBeforeClose {
                        window,
                    }));
                    match self.upcoming.pop_front() {
                        Some(next) => self.phase = Phase::Pending { market: next },
                        None => {
                            self.phase = Phase::Discovering;
                            self.request_or_queue_refresh(now, RefreshReason::GapRecovery, out);
                        }
                    }
                    return true;
                }
                if let Some(idx) = self
                    .upcoming
                    .iter()
                    .position(|m| &m.condition_id == condition_id)
                {
                    let window = self.upcoming[idx].window;
                    out.push(Output::Warn(CoverageWarning::ResolutionBeforeClose {
                        window,
                    }));
                    self.upcoming.remove(idx);
                    return true;
                }
                false
            }
            MarketLifecycleEvent::NewMarket { condition_id, .. } => {
                if self.knows_condition(condition_id) {
                    return true;
                }
                // Unknown market: only worth a refresh if we are actually
                // missing window data, nothing is in flight, and we have not
                // just asked.
                if self.needs_window_data()
                    && !self.refresh_inflight
                    && self
                        .last_refresh_request
                        .is_none_or(|t| now.signed_duration_since(t) >= MIN_REFRESH_SPACING)
                {
                    self.emit_refresh(now, RefreshReason::NewMarketHint, out);
                }
                false
            }
        }
    }

    /// Earliest instant at which [`Self::on_tick`] would do something. The
    /// driver sleeps until this (capped). `None` only when truly idle —
    /// in practice never, since `Discovering` always has a refresh pending.
    #[must_use]
    pub fn next_deadline(&self) -> Option<TimestampMs> {
        let mut min: Option<TimestampMs> = None;
        let mut consider = |t: TimestampMs| {
            if min.is_none_or(|m| t < m) {
                min = Some(t);
            }
        };
        match &self.phase {
            Phase::Discovering => {}
            Phase::Pending { market } => consider(market.window.open_time),
            Phase::Active { market } => {
                consider(before(market.close_time, self.timing.closing_lead));
            }
            Phase::Closing { market } => consider(market.close_time),
        }
        if self.refresh_inflight {
            if let Some(at) = self.refresh_requested_at {
                consider(at.saturating_add(REFRESH_INFLIGHT_TIMEOUT));
            }
        } else if let Some((at, _)) = self.next_refresh {
            consider(at);
        }
        if self.timing.expect_resolutions {
            for p in &self.parked {
                if !p.warned {
                    consider(p.closed_at.saturating_add(self.timing.resolution_timeout));
                }
            }
        }
        if let Phase::Active { market } | Phase::Closing { market } = &self.phase
            && self.upcoming.is_empty()
            && self.contract_warned_for != Some(market.window)
        {
            consider(before(market.close_time, self.timing.next_window_lead));
        }
        min
    }

    /// Snapshot for the status channel.
    #[must_use]
    pub fn status(&self, now: TimestampMs) -> SeriesStatus {
        let phase = match &self.phase {
            Phase::Discovering => "discovering",
            Phase::Pending { .. } => "pending",
            Phase::Active { .. } => "active",
            Phase::Closing { .. } => "closing",
        };
        let current = match &self.phase {
            Phase::Active { market } | Phase::Closing { market } => Some(market),
            _ => None,
        };
        let next = match &self.phase {
            Phase::Pending { market } => Some(market),
            _ => self.upcoming.front(),
        };
        SeriesStatus {
            series: self.series,
            phase,
            current: current.map(|m| m.window),
            current_slug: current.map(|m| m.event_slug.clone()),
            closes_in_ms: current.map(|m| m.close_time.signed_duration_since(now).as_millis()),
            next_known: next.is_some(),
            next_opens_in_ms: next
                .map(|m| m.window.open_time.signed_duration_since(now).as_millis()),
            parked: self.parked.len(),
            refresh_age_ms: self
                .last_refresh_ok
                .map(|t| now.signed_duration_since(t).as_millis()),
            contract_ok: current.is_none_or(|m| {
                !self.upcoming.is_empty()
                    || now < before(m.close_time, self.timing.next_window_lead)
            }),
        }
    }

    /// Replays every due phase boundary in time order.
    fn catch_up(&mut self, now: TimestampMs, out: &mut Vec<Output>) {
        loop {
            match &self.phase {
                Phase::Pending { market } if market.window.open_time <= now => {
                    let market = Arc::clone(market);
                    self.enter_active(market, now, out, true);
                }
                Phase::Active { market }
                    if before(market.close_time, self.timing.closing_lead) <= now =>
                {
                    let market = Arc::clone(market);
                    out.push(Output::Announce {
                        market: Arc::clone(&market),
                        lifecycle: WindowLifecycle::Closing,
                    });
                    self.phase = Phase::Closing { market };
                }
                Phase::Closing { market } if market.close_time <= now => {
                    let market = Arc::clone(market);
                    self.roll_closed(&market, now, out, None);
                }
                _ => break,
            }
        }
    }

    /// Enters `Active`, announcing `Open`. `from_tick` distinguishes the
    /// wall-clock path (request a top-up refresh, schedule the pre-close one)
    /// from the discovery path (which just refreshed and schedules its own).
    fn enter_active(
        &mut self,
        market: Arc<MarketInfo>,
        now: TimestampMs,
        out: &mut Vec<Output>,
        from_tick: bool,
    ) {
        out.push(Output::Announce {
            market: Arc::clone(&market),
            lifecycle: WindowLifecycle::Open,
        });
        if from_tick {
            self.request_or_queue_refresh(now, RefreshReason::Rolled, out);
            let pre_close = max(
                market.window.open_time,
                before(market.close_time, self.timing.refresh_lead),
            );
            if pre_close > now {
                self.schedule_refresh(pre_close, RefreshReason::PreClose);
            }
        }
        self.phase = Phase::Active { market };
    }

    /// The close path: announce `Closed`, settle or park, roll to the next
    /// window (or into a loud gap hunt). `resolved` short-circuits parking
    /// when the outcome arrived with the close (forced-close path).
    fn roll_closed(
        &mut self,
        market: &Arc<MarketInfo>,
        now: TimestampMs,
        out: &mut Vec<Output>,
        resolved: Option<Outcome>,
    ) {
        out.push(Output::Announce {
            market: Arc::clone(market),
            lifecycle: WindowLifecycle::Closed,
        });
        match resolved {
            Some(outcome) => out.push(Output::Announce {
                market: Arc::clone(market),
                lifecycle: WindowLifecycle::Resolved { outcome },
            }),
            None => self.park(Arc::clone(market), now, out),
        }
        self.prune_announced(now);
        match self.upcoming.pop_front() {
            Some(next) if next.window.open_time <= now => {
                self.enter_active(next, now, out, true);
            }
            Some(next) => {
                self.phase = Phase::Pending { market: next };
                self.request_or_queue_refresh(now, RefreshReason::Rolled, out);
            }
            None => {
                out.push(Output::Warn(CoverageWarning::NoNextAtClose {
                    window: market.window,
                }));
                self.phase = Phase::Discovering;
                self.request_or_queue_refresh(now, RefreshReason::GapRecovery, out);
            }
        }
    }

    fn park(&mut self, market: Arc<MarketInfo>, closed_at: TimestampMs, out: &mut Vec<Output>) {
        self.parked.push(Parked {
            market,
            closed_at,
            warned: false,
        });
        if self.parked.len() > MAX_PARKED {
            let evicted = self.parked.remove(0);
            if self.timing.expect_resolutions {
                out.push(Output::Warn(CoverageWarning::ParkedEvicted {
                    window: evicted.market.window,
                }));
            }
        }
    }

    fn check_contract(&mut self, now: TimestampMs, out: &mut Vec<Output>) {
        let (window, close) = match &self.phase {
            Phase::Active { market } | Phase::Closing { market } => {
                (market.window, market.close_time)
            }
            _ => return,
        };
        if self.upcoming.is_empty()
            && now >= before(close, self.timing.next_window_lead)
            && self.contract_warned_for != Some(window)
        {
            self.contract_warned_for = Some(window);
            out.push(Output::Warn(CoverageWarning::ContractViolated {
                window,
                close,
            }));
        }
    }

    /// True when discovery has not yet given us what §6 requires: any window
    /// at all while `Discovering`, or a successor beyond the one we hold.
    fn needs_window_data(&self) -> bool {
        match &self.phase {
            Phase::Discovering => true,
            Phase::Pending { .. } | Phase::Active { .. } | Phase::Closing { .. } => {
                self.upcoming.is_empty()
            }
        }
    }

    fn knows_condition(&self, condition_id: &core_types::ConditionId) -> bool {
        let in_phase = match &self.phase {
            Phase::Discovering => false,
            Phase::Pending { market } | Phase::Active { market } | Phase::Closing { market } => {
                &market.condition_id == condition_id
            }
        };
        in_phase
            || self
                .upcoming
                .iter()
                .any(|m| &m.condition_id == condition_id)
            || self
                .parked
                .iter()
                .any(|p| &p.market.condition_id == condition_id)
    }

    /// Emits a refresh now, or — if one is already in flight — queues one to
    /// fire as soon as it completes.
    fn request_or_queue_refresh(
        &mut self,
        now: TimestampMs,
        reason: RefreshReason,
        out: &mut Vec<Output>,
    ) {
        if self.refresh_inflight {
            self.schedule_refresh(now, reason);
        } else {
            self.emit_refresh(now, reason, out);
        }
    }

    fn emit_refresh(&mut self, now: TimestampMs, reason: RefreshReason, out: &mut Vec<Output>) {
        out.push(Output::Refresh { reason });
        self.refresh_inflight = true;
        self.refresh_requested_at = Some(now);
        self.last_refresh_request = Some(now);
    }

    /// Schedules a future refresh, keeping whichever of (existing, new) is
    /// earlier.
    fn schedule_refresh(&mut self, at: TimestampMs, reason: RefreshReason) {
        match self.next_refresh {
            Some((existing, _)) if existing <= at => {}
            _ => self.next_refresh = Some((at, reason)),
        }
    }

    /// Steady-state schedule: the pre-close verification for the current
    /// window (if still ahead) and the heartbeat, whichever comes first.
    fn schedule_steady_state_refresh(&mut self, now: TimestampMs) {
        let pre_close = match &self.phase {
            Phase::Active { market } | Phase::Closing { market } => {
                let t = max(
                    market.window.open_time,
                    before(market.close_time, self.timing.refresh_lead),
                );
                (t > now).then_some(t)
            }
            _ => None,
        };
        if let Some(t) = pre_close {
            self.schedule_refresh(t, RefreshReason::PreClose);
        }
        self.schedule_refresh(
            now.saturating_add(self.timing.max_refresh_interval),
            RefreshReason::Heartbeat,
        );
    }

    /// Drops announce-dedup entries for windows already past close —
    /// discovery can never return those again, so the set stays bounded.
    fn prune_announced(&mut self, now: TimestampMs) {
        let duration = self.series.duration.as_duration();
        self.announced
            .retain(|id| id.open_time.saturating_add(duration) > now);
    }
}

/// `ts − lead`, saturating.
fn before(ts: TimestampMs, lead: DurationMs) -> TimestampMs {
    TimestampMs::from_millis(ts.as_millis().saturating_sub(lead.as_millis()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{BTC_5M, cid, down_token, market, timing, up_token, windows};

    // BTC-5m windows on the real 300 s grid. W(0) opens at 600_000_000 ms.
    const W0_OPEN: i64 = 600_000_000;
    const STEP: i64 = 300_000;

    fn w(i: i64) -> Arc<MarketInfo> {
        market(BTC_5M, W0_OPEN + i * STEP)
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn tick(m: &mut SeriesMachine, now_ms: i64) -> Vec<Output> {
        let mut out = Vec::new();
        m.on_tick(ts(now_ms), &mut out);
        out
    }

    fn discover_ok(
        m: &mut SeriesMachine,
        now_ms: i64,
        current: Option<Arc<MarketInfo>>,
        upcoming: Vec<Arc<MarketInfo>>,
    ) -> Vec<Output> {
        let mut out = Vec::new();
        let snapshot = windows(BTC_5M, current, upcoming, ts(now_ms));
        m.on_discovery(Some(snapshot), ts(now_ms), &mut out);
        out
    }

    fn discover_fail(m: &mut SeriesMachine, now_ms: i64) -> Vec<Output> {
        let mut out = Vec::new();
        m.on_discovery(None, ts(now_ms), &mut out);
        out
    }

    fn market_event(
        m: &mut SeriesMachine,
        now_ms: i64,
        ev: &MarketLifecycleEvent,
    ) -> (bool, Vec<Output>) {
        let mut out = Vec::new();
        let claimed = m.on_market_event(ev, ts(now_ms), &mut out);
        (claimed, out)
    }

    fn announce(market: &Arc<MarketInfo>, lifecycle: WindowLifecycle) -> Output {
        Output::Announce {
            market: Arc::clone(market),
            lifecycle,
        }
    }

    fn refresh(reason: RefreshReason) -> Output {
        Output::Refresh { reason }
    }

    fn resolved_event(open_ms: i64, winning_open_ms: i64, up: bool) -> MarketLifecycleEvent {
        MarketLifecycleEvent::MarketResolved {
            condition_id: cid(open_ms),
            winning_token: if up {
                up_token(winning_open_ms)
            } else {
                down_token(winning_open_ms)
            },
            ts: ts(0),
        }
    }

    /// Boot a machine mid-W0 with `upcoming` lookahead, swallowing the boot
    /// refresh + initial announcements.
    fn booted(
        now_ms: i64,
        upcoming: Vec<Arc<MarketInfo>>,
        expect_resolutions: bool,
    ) -> SeriesMachine {
        let mut m = SeriesMachine::new(BTC_5M, timing(expect_resolutions));
        let boot = tick(&mut m, now_ms);
        assert_eq!(boot, vec![refresh(RefreshReason::Boot)]);
        let _ = discover_ok(&mut m, now_ms, Some(w(0)), upcoming);
        m
    }

    #[test]
    fn boot_requests_refresh_then_opens_mid_window() {
        let mut m = SeriesMachine::new(BTC_5M, timing(false));
        let t0 = W0_OPEN + 60_000;
        assert_eq!(tick(&mut m, t0), vec![refresh(RefreshReason::Boot)]);
        // No duplicate request while one is in flight.
        assert_eq!(tick(&mut m, t0 + 1_000), vec![]);

        let out = discover_ok(&mut m, t0 + 1_500, Some(w(0)), vec![w(1), w(2)]);
        assert_eq!(
            out,
            vec![
                announce(&w(0), WindowLifecycle::Discovered),
                announce(&w(1), WindowLifecycle::Discovered),
                announce(&w(2), WindowLifecycle::Discovered),
                announce(&w(0), WindowLifecycle::Open),
            ]
        );
        // Next deadline = the pre-close refresh (close − 120 s), which comes
        // before the closing transition (close − 30 s).
        assert_eq!(m.next_deadline(), Some(ts(W0_OPEN + STEP - 120_000)));
    }

    #[test]
    fn rolls_across_three_consecutive_windows_with_zero_gap() {
        // The §6 acceptance test: three back-to-back rolls, each Open in the
        // same instant as the predecessor's Closed.
        let t0 = W0_OPEN + 60_000;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        let close = |i: i64| W0_OPEN + (i + 1) * STEP;

        for i in 0..3 {
            // Pre-close verification fires at close − 120 s.
            assert_eq!(
                tick(&mut m, close(i) - 120_000),
                vec![refresh(RefreshReason::PreClose)],
                "window {i}: refresh at close − 120s"
            );
            let topped_up = discover_ok(
                &mut m,
                close(i) - 119_000,
                Some(w(i)),
                vec![w(i + 1), w(i + 2)],
            );
            // Everything here was already announced — no duplicates.
            assert_eq!(topped_up, vec![], "window {i}: top-up announcements");

            // Closing at close − 30 s.
            assert_eq!(
                tick(&mut m, close(i) - 30_000),
                vec![announce(&w(i), WindowLifecycle::Closing)],
                "window {i}: closing signal"
            );
            // The roll: Closed and the successor's Open in one tick, zero gap.
            assert_eq!(
                tick(&mut m, close(i)),
                vec![
                    announce(&w(i), WindowLifecycle::Closed),
                    announce(&w(i + 1), WindowLifecycle::Open),
                    refresh(RefreshReason::Rolled),
                ],
                "window {i}: roll at close"
            );
            // The top-up answer for the new window arrives a second later.
            let out = discover_ok(
                &mut m,
                close(i) + 1_000,
                Some(w(i + 1)),
                vec![w(i + 2), w(i + 3)],
            );
            assert_eq!(
                out,
                vec![announce(&w(i + 3), WindowLifecycle::Discovered)],
                "window {i}: post-roll top-up"
            );
        }
    }

    #[test]
    fn missing_window_gap_recovers_with_backoff_and_warns_once() {
        // The gap acceptance test: discovery succeeds but the venue has no
        // next window. Backoff-paced retries, exactly one §6 warning, a loud
        // gap at close, recovery when the window finally appears.
        let mut m = SeriesMachine::new(BTC_5M, timing(false));
        let t0 = W0_OPEN + 60_000;
        let close = W0_OPEN + STEP;
        assert_eq!(tick(&mut m, t0), vec![refresh(RefreshReason::Boot)]);

        let out = discover_ok(&mut m, t0, Some(w(0)), vec![]);
        assert_eq!(
            out,
            vec![
                announce(&w(0), WindowLifecycle::Discovered),
                announce(&w(0), WindowLifecycle::Open),
            ]
        );

        // Unsatisfying success → retry paced by backoff: +1 s.
        assert_eq!(m.next_deadline(), Some(ts(t0 + 1_000)));
        assert_eq!(
            tick(&mut m, t0 + 1_000),
            vec![refresh(RefreshReason::Retry)]
        );
        // Failure → +2 s.
        assert_eq!(discover_fail(&mut m, t0 + 1_100), vec![]);
        assert_eq!(m.next_deadline(), Some(ts(t0 + 3_100)));
        assert_eq!(
            tick(&mut m, t0 + 3_100),
            vec![refresh(RefreshReason::Retry)]
        );
        // Failure → +4 s.
        assert_eq!(discover_fail(&mut m, t0 + 3_200), vec![]);
        assert_eq!(m.next_deadline(), Some(ts(t0 + 7_200)));

        // Jump to the §6 deadline (close − 60 s): the due retry fires and the
        // contract violation is announced — exactly once.
        let out = tick(&mut m, close - 60_000);
        assert_eq!(
            out,
            vec![
                refresh(RefreshReason::Retry),
                Output::Warn(CoverageWarning::ContractViolated {
                    window: w(0).window,
                    close: ts(close),
                }),
            ]
        );
        let _ = discover_fail(&mut m, close - 59_000);
        assert_eq!(
            tick(&mut m, close - 58_000),
            vec![],
            "no duplicate §6 warning"
        );

        // At close: loud gap, series stays alive and hunting.
        let out = tick(&mut m, close);
        assert_eq!(
            out,
            vec![
                announce(&w(0), WindowLifecycle::Closing),
                announce(&w(0), WindowLifecycle::Closed),
                Output::Warn(CoverageWarning::NoNextAtClose {
                    window: w(0).window
                }),
                refresh(RefreshReason::GapRecovery),
            ]
        );

        // Recovery: the venue finally created windows; W1 is already open.
        let out = discover_ok(&mut m, close + 10_000, Some(w(1)), vec![w(2)]);
        assert_eq!(
            out,
            vec![
                announce(&w(1), WindowLifecycle::Discovered),
                announce(&w(2), WindowLifecycle::Discovered),
                announce(&w(1), WindowLifecycle::Open),
            ]
        );

        // The satisfied need wiped the stale retry schedule…
        assert_eq!(tick(&mut m, close + 11_000), vec![]);
        // …and reset the backoff: after the pre-close refresh fails, the next
        // retry is paced at the initial 1 s again.
        let pre_close = close + STEP - 120_000;
        assert_eq!(
            tick(&mut m, pre_close),
            vec![refresh(RefreshReason::PreClose)]
        );
        assert_eq!(discover_fail(&mut m, pre_close + 100), vec![]);
        assert_eq!(m.next_deadline(), Some(ts(pre_close + 1_100)));
    }

    #[test]
    fn pre_close_refresh_failure_with_known_next_is_quiet() {
        let t0 = W0_OPEN + 60_000;
        let close = W0_OPEN + STEP;
        let mut m = booted(t0, vec![w(1), w(2)], false);

        assert_eq!(
            tick(&mut m, close - 120_000),
            vec![refresh(RefreshReason::PreClose)]
        );
        // Failure with the next window already known: quiet backoff retry, no
        // warning of any kind.
        assert_eq!(discover_fail(&mut m, close - 119_000), vec![]);
        assert_eq!(m.next_deadline(), Some(ts(close - 118_000)));
        assert_eq!(
            tick(&mut m, close - 118_000),
            vec![refresh(RefreshReason::Retry)]
        );
        let out = discover_ok(&mut m, close - 117_000, Some(w(0)), vec![w(1), w(2)]);
        assert_eq!(out, vec![]);
        // The §6 deadline passes silently — the contract is satisfied.
        assert_eq!(tick(&mut m, close - 60_000), vec![]);
    }

    #[test]
    fn resolved_for_parked_window_emits_resolved() {
        let t0 = W0_OPEN + 60_000;
        let close = W0_OPEN + STEP;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        let _ = tick(&mut m, close - 30_000);
        let _ = tick(&mut m, close); // W0 closed and parked, W1 open

        // Up winner.
        let ev = resolved_event(W0_OPEN, W0_OPEN, true);
        let (claimed, out) = market_event(&mut m, close + 2_000, &ev);
        assert!(claimed);
        assert_eq!(
            out,
            vec![announce(
                &w(0),
                WindowLifecycle::Resolved {
                    outcome: Outcome::Up
                }
            )]
        );

        // Roll W1 too, then resolve it Down.
        let _ = discover_ok(&mut m, close + 3_000, Some(w(1)), vec![w(2), w(3)]);
        let close1 = close + STEP;
        let _ = tick(&mut m, close1 - 30_000);
        let _ = tick(&mut m, close1);
        let ev = resolved_event(W0_OPEN + STEP, W0_OPEN + STEP, false);
        let (claimed, out) = market_event(&mut m, close1 + 2_000, &ev);
        assert!(claimed);
        assert_eq!(
            out,
            vec![announce(
                &w(1),
                WindowLifecycle::Resolved {
                    outcome: Outcome::Down
                }
            )]
        );
    }

    #[test]
    fn resolved_unknown_condition_id_is_unclaimed_with_no_outputs() {
        let t0 = W0_OPEN + 60_000;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        // A condition id belonging to no window this machine has ever seen.
        let ev = resolved_event(123_456_789, W0_OPEN, true);
        let (claimed, out) = market_event(&mut m, t0 + 1_000, &ev);
        assert!(!claimed);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn resolved_with_foreign_winning_token_warns_and_keeps_parked() {
        let t0 = W0_OPEN + 60_000;
        let close = W0_OPEN + STEP;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        let _ = tick(&mut m, close - 30_000);
        let _ = tick(&mut m, close);

        // Right condition id, but a winning token from a different window.
        let ev = resolved_event(W0_OPEN, W0_OPEN + STEP, true);
        let (claimed, out) = market_event(&mut m, close + 2_000, &ev);
        assert!(claimed);
        assert_eq!(
            out,
            vec![Output::Warn(CoverageWarning::UnknownWinningToken {
                window: w(0).window
            })]
        );

        // Still parked: a later correct resolution works.
        let ev = resolved_event(W0_OPEN, W0_OPEN, true);
        let (claimed, out) = market_event(&mut m, close + 3_000, &ev);
        assert!(claimed);
        assert_eq!(
            out,
            vec![announce(
                &w(0),
                WindowLifecycle::Resolved {
                    outcome: Outcome::Up
                }
            )]
        );
    }

    #[test]
    fn resolution_overdue_warns_once_then_late_resolution_still_works() {
        let t0 = W0_OPEN + 60_000;
        let close = W0_OPEN + STEP;
        let mut m = booted(t0, vec![w(1), w(2)], true);
        let _ = tick(&mut m, close - 30_000);
        let _ = tick(&mut m, close);
        // Answer the roll's top-up so no refresh timers interfere.
        let _ = discover_ok(&mut m, close + 1_000, Some(w(1)), vec![w(2), w(3)]);

        // 120 s after close with no resolution: one loud warning.
        let out = tick(&mut m, close + 120_000);
        assert_eq!(
            out,
            vec![Output::Warn(CoverageWarning::ResolutionOverdue {
                window: w(0).window,
                closed_at: ts(close),
            })]
        );
        assert_eq!(tick(&mut m, close + 121_000), vec![], "warned once");

        // The late resolution still settles it.
        let ev = resolved_event(W0_OPEN, W0_OPEN, true);
        let (claimed, out) = market_event(&mut m, close + 130_000, &ev);
        assert!(claimed);
        assert_eq!(
            out,
            vec![announce(
                &w(0),
                WindowLifecycle::Resolved {
                    outcome: Outcome::Up
                }
            )]
        );
    }

    #[test]
    fn no_overdue_warning_without_a_resolution_source() {
        // expect_resolutions = false (pre-feed-clob): windows park silently.
        let t0 = W0_OPEN + 60_000;
        let close = W0_OPEN + STEP;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        let _ = tick(&mut m, close - 30_000);
        let _ = tick(&mut m, close);
        let _ = discover_ok(&mut m, close + 1_000, Some(w(1)), vec![w(2), w(3)]);
        assert_eq!(tick(&mut m, close + 120_000), vec![]);
    }

    #[test]
    fn resolution_before_close_force_rolls() {
        let t0 = W0_OPEN + 60_000;
        let mut m = booted(t0, vec![w(1), w(2)], false);

        // W0 resolves while our clock says 100 s remain.
        let ev = resolved_event(W0_OPEN, W0_OPEN, false);
        let (claimed, out) = market_event(&mut m, W0_OPEN + 200_000, &ev);
        assert!(claimed);
        assert_eq!(
            out,
            vec![
                Output::Warn(CoverageWarning::ResolutionBeforeClose {
                    window: w(0).window
                }),
                announce(&w(0), WindowLifecycle::Closed),
                announce(
                    &w(0),
                    WindowLifecycle::Resolved {
                        outcome: Outcome::Down
                    }
                ),
                refresh(RefreshReason::Rolled),
            ]
        );
        // Answer the top-up so nothing is in flight at the boundary.
        let out = discover_ok(&mut m, W0_OPEN + 210_000, None, vec![w(1), w(2)]);
        assert_eq!(out, vec![]);
        // W1 had not opened yet → Pending, then Open exactly at its open time.
        let out = tick(&mut m, W0_OPEN + STEP);
        assert_eq!(
            out,
            vec![
                announce(&w(1), WindowLifecycle::Open),
                refresh(RefreshReason::Rolled),
            ]
        );
    }

    #[test]
    fn new_market_triggers_refresh_only_when_data_is_missing() {
        let t0 = W0_OPEN + 60_000;
        // Fully stocked machine: unknown new_market is ignored.
        let mut m = booted(t0, vec![w(1), w(2)], false);
        let unknown = MarketLifecycleEvent::NewMarket {
            condition_id: cid(987_654_321),
            slug: "something-else".to_owned(),
            ts: ts(t0),
        };
        let (claimed, out) = market_event(&mut m, t0 + 5_000, &unknown);
        assert!(!claimed);
        assert_eq!(out, vec![]);
        // A known condition id is claimed without action.
        let known = MarketLifecycleEvent::NewMarket {
            condition_id: cid(W0_OPEN + STEP),
            slug: "test".to_owned(),
            ts: ts(t0),
        };
        let (claimed, out) = market_event(&mut m, t0 + 6_000, &known);
        assert!(claimed);
        assert_eq!(out, vec![]);

        // A machine missing its next window: the hint fires a refresh (the
        // boot request was ≥ 1 s ago, so spacing allows it).
        let mut m = SeriesMachine::new(BTC_5M, timing(false));
        let _ = tick(&mut m, t0);
        let _ = discover_ok(&mut m, t0, Some(w(0)), vec![]);
        let (claimed, out) = market_event(&mut m, t0 + 1_500, &unknown);
        assert!(!claimed);
        assert_eq!(out, vec![refresh(RefreshReason::NewMarketHint)]);
        // While that hint is in flight: no second request.
        let (_, out) = market_event(&mut m, t0 + 1_700, &unknown);
        assert_eq!(out, vec![]);
        // In-flight cleared but only 400 ms since the last request: spaced.
        let _ = discover_fail(&mut m, t0 + 1_800);
        let (_, out) = market_event(&mut m, t0 + 1_900, &unknown);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn oversleep_catch_up_replays_all_boundaries_in_order() {
        let t0 = W0_OPEN + 60_000;
        let mut m = booted(t0, vec![w(1), w(2)], false);

        // Jump straight to one minute into W2: every missed boundary replays
        // in time order within one tick.
        let out = tick(&mut m, W0_OPEN + 2 * STEP + 60_000);
        assert_eq!(
            out,
            vec![
                announce(&w(0), WindowLifecycle::Closing),
                announce(&w(0), WindowLifecycle::Closed),
                announce(&w(1), WindowLifecycle::Open),
                refresh(RefreshReason::Rolled),
                announce(&w(1), WindowLifecycle::Closing),
                announce(&w(1), WindowLifecycle::Closed),
                announce(&w(2), WindowLifecycle::Open),
            ]
        );
    }

    #[test]
    fn gap_window_goes_pending_then_opens_at_open_time() {
        let mut m = SeriesMachine::new(BTC_5M, timing(false));
        let t0 = W0_OPEN + STEP - 10_000; // 10 s before W1 opens, W0 unknown
        assert_eq!(tick(&mut m, t0), vec![refresh(RefreshReason::Boot)]);
        // Discovery: nothing currently open (a scheduled gap), W1+W2 ahead.
        let out = discover_ok(&mut m, t0, None, vec![w(1), w(2)]);
        assert_eq!(
            out,
            vec![
                announce(&w(1), WindowLifecycle::Discovered),
                announce(&w(2), WindowLifecycle::Discovered),
            ]
        );
        assert_eq!(tick(&mut m, t0 + 5_000), vec![]);
        // Opens exactly on time.
        let out = tick(&mut m, W0_OPEN + STEP);
        assert_eq!(
            out,
            vec![
                announce(&w(1), WindowLifecycle::Open),
                refresh(RefreshReason::Rolled),
            ]
        );
    }

    #[test]
    fn discovered_announced_once_across_refreshes() {
        let t0 = W0_OPEN + 60_000;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        // The same snapshot again: no announcements at all.
        let out = discover_ok(&mut m, t0 + 10_000, Some(w(0)), vec![w(1), w(2)]);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn heartbeat_refresh_fires_when_everything_is_known_but_stale() {
        let mut custom = timing(false);
        custom.max_refresh_interval = DurationMs::from_millis(50_000);
        let mut m = SeriesMachine::new(BTC_5M, custom);
        let t0 = W0_OPEN + 60_000;
        let _ = tick(&mut m, t0);
        let _ = discover_ok(&mut m, t0, Some(w(0)), vec![w(1), w(2)]);
        // Heartbeat (t0+50s) beats the pre-close refresh (close−120s = t0+120s).
        assert_eq!(m.next_deadline(), Some(ts(t0 + 50_000)));
        assert_eq!(
            tick(&mut m, t0 + 50_000),
            vec![refresh(RefreshReason::Heartbeat)]
        );
    }

    #[test]
    fn parked_cap_evicts_oldest_with_warning() {
        let t0 = W0_OPEN + 60_000;
        let mut m = SeriesMachine::new(BTC_5M, timing(true));
        let _ = tick(&mut m, t0);
        // Ten windows known up front; roll through nine of them unresolved.
        let upcoming: Vec<_> = (1..10).map(w).collect();
        let _ = discover_ok(&mut m, t0, Some(w(0)), upcoming);
        let out = tick(&mut m, W0_OPEN + 9 * STEP + 60_000);
        assert!(
            out.contains(&Output::Warn(CoverageWarning::ParkedEvicted {
                window: w(0).window
            })),
            "oldest parked window evicted: {out:?}"
        );
        assert_eq!(m.status(ts(W0_OPEN + 9 * STEP + 60_000)).parked, MAX_PARKED);
    }

    #[test]
    fn inflight_watchdog_retries_a_lost_refresh() {
        let mut m = SeriesMachine::new(BTC_5M, timing(false));
        let t0 = W0_OPEN + 60_000;
        assert_eq!(tick(&mut m, t0), vec![refresh(RefreshReason::Boot)]);
        // The reply never comes.
        assert_eq!(tick(&mut m, t0 + 44_999), vec![]);
        assert_eq!(
            tick(&mut m, t0 + 45_000),
            vec![refresh(RefreshReason::Retry)]
        );
    }

    #[test]
    fn discovery_current_mismatch_supersedes_our_window() {
        let t0 = W0_OPEN + 60_000;
        let mut m = booted(t0, vec![w(1), w(2)], false);
        // Our clock lags: discovery (and reality) are already on W1.
        let out = discover_ok(&mut m, W0_OPEN + STEP + 5_000, Some(w(1)), vec![w(2), w(3)]);
        assert_eq!(
            out,
            vec![
                announce(&w(3), WindowLifecycle::Discovered),
                Output::Warn(CoverageWarning::CurrentSuperseded {
                    old: w(0).window,
                    new: w(1).window,
                }),
                announce(&w(0), WindowLifecycle::Closed),
                announce(&w(1), WindowLifecycle::Open),
            ]
        );
    }

    #[test]
    fn status_reflects_coverage_state() {
        let t0 = W0_OPEN + 60_000;
        let m = booted(t0, vec![w(1), w(2)], false);
        let s = m.status(ts(t0 + 1_000));
        assert_eq!(s.phase, "active");
        assert_eq!(s.current, Some(w(0).window));
        assert_eq!(s.closes_in_ms, Some(STEP - 61_000));
        assert!(s.next_known);
        assert!(s.contract_ok);
        assert_eq!(s.parked, 0);

        // Strip the lookahead: contract turns red inside the final 60 s.
        let mut gap = SeriesMachine::new(BTC_5M, timing(false));
        let _ = tick(&mut gap, t0);
        let _ = discover_ok(&mut gap, t0, Some(w(0)), vec![]);
        assert!(
            gap.status(ts(t0)).contract_ok,
            "still time to find the next"
        );
        assert!(!gap.status(ts(W0_OPEN + STEP - 59_000)).contract_ok);
        assert!(!gap.status(ts(t0)).next_known);
    }
}
