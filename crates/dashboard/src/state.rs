//! The in-memory dashboard state stores and the bus-fold projector.
//!
//! One coarse-grained store ([`DashboardData`], held behind a `std::sync::Mutex`
//! by the [`DashboardHandle`](crate::DashboardHandle)) folds the bus into a
//! shared market view plus per-mode trading state. The fold is brief and
//! synchronous (the "small lock-protected snapshot for the dashboard" idiom,
//! CLAUDE.md §4/§5): the projector locks, folds, computes the incremental
//! [`WsUpdate`]s, and unlocks; handlers lock, read the slice they need, and
//! unlock. No lock is ever held across an `.await`.
//!
//! `SharedView` is single (not per-mode): the live Polymarket feeds are
//! identical for paper and live, so books/model/feed-health are stored once.
//! Only *trading* state (fills, orders, inventory, risk, equity, analytics) is
//! namespaced per [`Mode`].

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

use analytics::{Analytics, AnalyticsParams};
use core_types::{
    Asset, BookHealth, BookSnapshot, BookUnreliableReason, BreakerKind, ConditionId, ControlEvent,
    Dollars, Event, FeedHealth, Fill, InventorySnapshot, MarketInfo, Mode, ModelHealthEvent,
    ModelSnapshot, OrderId, OrderUpdate, Outcome, Price, PriceSource, RiskEvent, SettlementSummary,
    Side, Size, TickKind, TickSize, TimestampMs, TokenId, TopOfBook, WindowId, WindowLifecycle,
};
use engine::RiskStateSnapshot;
use venue_api::Wallet;
use venue_paper::PaperLedgerSnapshot;

use crate::live_markout::{FairRing, LiveMarkout};
use crate::ws::{BreakerEvent, WsUpdate};

/// Fills retained per mode for the blotter.
pub(crate) const FILLS_RING_CAP: usize = 1_000;
/// Settled-window summaries retained per mode.
pub(crate) const SETTLEMENTS_RING_CAP: usize = 256;
/// Equity-curve points retained per mode.
pub(crate) const EQUITY_RING_CAP: usize = 4_096;
/// Recent trade prints retained per active window.
pub(crate) const RECENT_PRINTS_CAP: usize = 64;
/// Live windows kept before the cap backstop drops the oldest.
pub(crate) const WINDOWS_CAP: usize = 64;
/// A resolved window lingers this long (ms) before it is pruned.
pub(crate) const WINDOW_GRACE_MS: i64 = 120_000;

/// A fixed-capacity FIFO ring: pushing past `cap` drops the oldest item. Keeps
/// the 24/7 state bounded.
#[derive(Debug, Clone)]
pub(crate) struct Ring<T> {
    buf: VecDeque<T>,
    cap: usize,
}

impl<T> Ring<T> {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    pub(crate) fn push(&mut self, item: T) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(item);
    }

    /// Oldest-first iterator. (Length is `iter().count()`; the dashboard always
    /// iterates the ring rather than indexing it.)
    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.buf.iter()
    }
}

/// A recent trade print on a window's token.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrintRow {
    pub(crate) price: Price,
    pub(crate) size: Size,
    pub(crate) side: Side,
    pub(crate) ts: TimestampMs,
}

/// A sampled equity point.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EquityPoint {
    pub(crate) ts: TimestampMs,
    pub(crate) equity: Dollars,
}

/// A currently-stale feed stream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeedStaleEntry {
    pub(crate) age_ms: i64,
    #[allow(dead_code, reason = "kept for a future 'stale since' display")]
    pub(crate) since: TimestampMs,
}

/// The market view of one live window (shared across modes).
#[derive(Debug, Clone)]
pub(crate) struct WindowView {
    pub(crate) market: Arc<MarketInfo>,
    pub(crate) lifecycle: WindowLifecycle,
    pub(crate) up_top: Option<TopOfBook>,
    pub(crate) down_top: Option<TopOfBook>,
    pub(crate) up_book: Option<Arc<BookSnapshot>>,
    pub(crate) down_book: Option<Arc<BookSnapshot>>,
    pub(crate) tick: TickSize,
    pub(crate) recent_prints: Ring<PrintRow>,
    pub(crate) resolved_at: Option<TimestampMs>,
    pub(crate) outcome: Option<Outcome>,
}

impl WindowView {
    fn new(market: Arc<MarketInfo>) -> Self {
        let tick = market.tick_size;
        Self {
            market,
            lifecycle: WindowLifecycle::Discovered,
            up_top: None,
            down_top: None,
            up_book: None,
            down_book: None,
            tick,
            recent_prints: Ring::new(RECENT_PRINTS_CAP),
            resolved_at: None,
            outcome: None,
        }
    }
}

/// The shared market view: identical for both trading modes.
#[derive(Debug)]
pub(crate) struct SharedView {
    pub(crate) windows: BTreeMap<WindowId, WindowView>,
    pub(crate) model_by_asset: HashMap<Asset, ModelSnapshot>,
    /// Latest model snapshot per window (the per-window fair the live view shows;
    /// `model_by_asset` only keeps the asset's most-recent, which conflates its
    /// 5m/15m/1h windows). Dropped with its window on prune.
    pub(crate) model_by_window: HashMap<WindowId, ModelSnapshot>,
    pub(crate) model_health: HashMap<Asset, ModelHealthEvent>,
    pub(crate) feed_stale: BTreeMap<(PriceSource, Asset, TickKind), FeedStaleEntry>,
    pub(crate) book_unreliable: BTreeMap<WindowId, BookUnreliableReason>,
    /// Per-window model-fair history feeding the live 5s markout (shared — the
    /// feeds are identical for both modes). Dropped with its window on prune.
    pub(crate) fair_rings: BTreeMap<WindowId, FairRing>,
}

impl SharedView {
    fn new() -> Self {
        Self {
            windows: BTreeMap::new(),
            model_by_asset: HashMap::new(),
            model_by_window: HashMap::new(),
            model_health: HashMap::new(),
            feed_stale: BTreeMap::new(),
            book_unreliable: BTreeMap::new(),
            fair_rings: BTreeMap::new(),
        }
    }

    /// Finds the window and outcome a token belongs to (linear over the few
    /// live windows).
    fn locate_token(&self, token: &TokenId) -> Option<(WindowId, Outcome)> {
        self.windows
            .iter()
            .find_map(|(wid, wv)| wv.market.tokens.outcome_of(token).map(|o| (*wid, o)))
    }

    /// Finds the window a condition id belongs to.
    fn locate_condition(&self, cid: &ConditionId) -> Option<WindowId> {
        self.windows
            .iter()
            .find_map(|(wid, wv)| (wv.market.condition_id == *cid).then_some(*wid))
    }

    /// Drops windows resolved longer than the grace period, then enforces the
    /// hard cap (oldest-resolved first, then oldest-open) as a backstop.
    fn prune(&mut self, now: TimestampMs) {
        let cutoff = now.as_millis() - WINDOW_GRACE_MS;
        self.windows
            .retain(|_, wv| wv.resolved_at.is_none_or(|t| t.as_millis() >= cutoff));
        while self.windows.len() > WINDOWS_CAP {
            let victim = self
                .windows
                .iter()
                .min_by_key(|(wid, wv)| {
                    (
                        wv.resolved_at.is_none(),
                        wv.resolved_at.map_or(i64::MAX, |t| t.as_millis()),
                        wid.open_time.as_millis(),
                    )
                })
                .map(|(wid, _)| *wid);
            match victim {
                Some(v) => {
                    self.windows.remove(&v);
                }
                None => break,
            }
        }
        // Drop per-window model state for windows that no longer exist (24/7 bound).
        let windows = &self.windows;
        self.fair_rings.retain(|wid, _| windows.contains_key(wid));
        self.model_by_window
            .retain(|wid, _| windows.contains_key(wid));
    }
}

/// Per-mode trading state (paper or live).
#[derive(Debug)]
pub(crate) struct ModeState {
    pub(crate) analytics: Analytics,
    pub(crate) fills: Ring<Arc<Fill>>,
    pub(crate) orders: HashMap<OrderId, Arc<OrderUpdate>>,
    pub(crate) inventory: HashMap<WindowId, Arc<InventorySnapshot>>,
    pub(crate) settlements: Ring<Arc<SettlementSummary>>,
    pub(crate) tripped: BTreeSet<BreakerKind>,
    pub(crate) last_cancel_all: Option<BreakerKind>,
    pub(crate) risk_snapshot: Option<RiskStateSnapshot>,
    pub(crate) wallet: Option<Wallet>,
    pub(crate) ledger: Option<PaperLedgerSnapshot>,
    pub(crate) equity: Ring<EquityPoint>,
    pub(crate) last_equity: Option<Dollars>,
    pub(crate) running: bool,
    pub(crate) armed: bool,
    pub(crate) ws_connected: bool,
    pub(crate) last_control: Option<ControlEvent>,
    /// Live 5s markouts for this mode's maker fills (capped to the fills ring so
    /// any displayed fill keeps its value).
    pub(crate) live_markout: LiveMarkout,
}

impl ModeState {
    fn new(mode: Mode) -> Self {
        Self {
            analytics: Analytics::new(mode, AnalyticsParams::default()),
            fills: Ring::new(FILLS_RING_CAP),
            orders: HashMap::new(),
            live_markout: LiveMarkout::new(FILLS_RING_CAP),
            inventory: HashMap::new(),
            settlements: Ring::new(SETTLEMENTS_RING_CAP),
            tripped: BTreeSet::new(),
            last_cancel_all: None,
            risk_snapshot: None,
            wallet: None,
            ledger: None,
            equity: Ring::new(EQUITY_RING_CAP),
            last_equity: None,
            running: false,
            armed: false,
            ws_connected: false,
            last_control: None,
        }
    }
}

/// The current safe-listed parameter view shown by `/api/params` (populated by
/// the orchestrator at boot via [`DashboardData::set_params`]; the
/// `config → params` boundary map stays deferred, so this is a flat
/// key/value projection the bot fills).
#[derive(Debug, Clone, Default)]
pub struct ParamsView {
    /// The paper starting capital, when known.
    pub paper_capital: Option<Dollars>,
    /// Flat `(key, value)` parameter entries for display.
    pub entries: Vec<(String, String)>,
}

/// The single mutable store the dashboard server reads and the projector writes.
#[derive(Debug)]
pub(crate) struct DashboardData {
    pub(crate) shared: SharedView,
    pub(crate) paper: ModeState,
    pub(crate) live: ModeState,
    pub(crate) params: ParamsView,
    /// The latest control-plane state snapshot, pushed by the orchestrator after
    /// each command and read by `GET /api/control/status` (§10.6).
    pub(crate) control_state: Option<crate::command::ControlStateSnapshot>,
    pub(crate) server_started: TimestampMs,
    pub(crate) last_now: TimestampMs,
}

impl DashboardData {
    pub(crate) fn new(now: TimestampMs) -> Self {
        Self {
            shared: SharedView::new(),
            paper: ModeState::new(Mode::Paper),
            live: ModeState::new(Mode::Live),
            params: ParamsView::default(),
            control_state: None,
            server_started: now,
            last_now: now,
        }
    }

    pub(crate) fn mode(&self, mode: Mode) -> &ModeState {
        match mode {
            Mode::Paper => &self.paper,
            Mode::Live => &self.live,
        }
    }

    fn mode_mut(&mut self, mode: Mode) -> &mut ModeState {
        match mode {
            Mode::Paper => &mut self.paper,
            Mode::Live => &mut self.live,
        }
    }

    /// Folds one bus event into the store and returns the incremental
    /// [`WsUpdate`]s to broadcast. Market-data variants update the shared view;
    /// per-mode variants update `mode`'s trading state. `mode`'s analytics is
    /// fed every event (it ignores the ones it does not consume).
    pub(crate) fn project(&mut self, mode: Mode, event: &Event, now: TimestampMs) -> Vec<WsUpdate> {
        self.last_now = now;
        let _ = self.mode_mut(mode).analytics.on_event(event);
        let mut updates = Vec::new();
        let ts_ms = now.as_millis();
        match event {
            Event::Window { market, lifecycle } => {
                let wid = market.window;
                let view = self
                    .shared
                    .windows
                    .entry(wid)
                    .or_insert_with(|| WindowView::new(Arc::clone(market)));
                view.market = Arc::clone(market);
                view.lifecycle = *lifecycle;
                view.tick = market.tick_size;
                let outcome = if let WindowLifecycle::Resolved { outcome } = lifecycle {
                    view.resolved_at = Some(now);
                    view.outcome = Some(*outcome);
                    Some(*outcome)
                } else {
                    None
                };
                updates.push(WsUpdate::Lifecycle {
                    ts_ms,
                    window: wid.to_string(),
                    lifecycle: *lifecycle,
                    outcome,
                });
                self.shared.prune(now);
            }
            Event::Book(snap) => {
                if let Some((wid, outcome)) = self.shared.locate_token(&snap.token_id)
                    && let Some(view) = self.shared.windows.get_mut(&wid)
                {
                    let top = snap.top();
                    match outcome {
                        Outcome::Up => {
                            view.up_book = Some(Arc::clone(snap));
                            view.up_top = Some(top);
                        }
                        Outcome::Down => {
                            view.down_book = Some(Arc::clone(snap));
                            view.down_top = Some(top);
                        }
                    }
                    updates.push(WsUpdate::Top {
                        ts_ms,
                        window: wid.to_string(),
                        outcome,
                        top,
                    });
                }
            }
            Event::TopOfBook { token_id, top } => {
                if let Some((wid, outcome)) = self.shared.locate_token(token_id)
                    && let Some(view) = self.shared.windows.get_mut(&wid)
                {
                    match outcome {
                        Outcome::Up => view.up_top = Some(*top),
                        Outcome::Down => view.down_top = Some(*top),
                    }
                    updates.push(WsUpdate::Top {
                        ts_ms,
                        window: wid.to_string(),
                        outcome,
                        top: *top,
                    });
                }
            }
            Event::LastTrade {
                token_id,
                price,
                size,
                side,
                ts,
            } => {
                if let Some((wid, _)) = self.shared.locate_token(token_id)
                    && let Some(view) = self.shared.windows.get_mut(&wid)
                {
                    view.recent_prints.push(PrintRow {
                        price: *price,
                        size: *size,
                        side: *side,
                        ts: *ts,
                    });
                }
            }
            Event::TickSizeChange {
                condition_id,
                new_tick,
                ..
            } => {
                if let Some(wid) = self.shared.locate_condition(condition_id)
                    && let Some(view) = self.shared.windows.get_mut(&wid)
                {
                    view.tick = *new_tick;
                }
            }
            Event::Model(snap) => {
                self.shared.model_by_asset.insert(snap.asset, *snap);
                if let Some(wid) = snap.window {
                    self.shared.model_by_window.insert(wid, *snap);
                    self.shared
                        .fair_rings
                        .entry(wid)
                        .or_default()
                        .push(snap.ts, snap.p_up);
                }
                let (window_key, book_mid) = match snap.window {
                    Some(wid) => {
                        let mid = self
                            .shared
                            .windows
                            .get(&wid)
                            .and_then(|wv| wv.up_top)
                            .and_then(|t| t.mid());
                        (Some(wid.to_string()), mid)
                    }
                    None => (None, None),
                };
                updates.push(WsUpdate::Model {
                    ts_ms,
                    asset: snap.asset,
                    window: window_key,
                    p_up: snap.p_up,
                    z: snap.z,
                    sigma_1s: snap.sigma_1s,
                    book_mid,
                });
            }
            Event::ModelHealth(ev) => {
                self.shared.model_health.insert(ev.asset, *ev);
            }
            Event::FeedHealth(fh) => match fh {
                FeedHealth::Stale {
                    source,
                    asset,
                    kind,
                    age,
                } => {
                    self.shared.feed_stale.insert(
                        (*source, *asset, *kind),
                        FeedStaleEntry {
                            age_ms: age.as_millis(),
                            since: now,
                        },
                    );
                }
                FeedHealth::Recovered {
                    source,
                    asset,
                    kind,
                    ..
                } => {
                    self.shared.feed_stale.remove(&(*source, *asset, *kind));
                }
            },
            Event::BookHealth(bh) => match bh {
                BookHealth::Unreliable { window, reason, .. } => {
                    self.shared.book_unreliable.insert(*window, *reason);
                }
                BookHealth::Recovered { window, .. } => {
                    self.shared.book_unreliable.remove(window);
                }
            },
            Event::OrderUpdate(u) => {
                let ms = self.mode_mut(mode);
                if u.state.is_terminal() {
                    ms.orders.remove(&u.order_id);
                } else {
                    ms.orders.insert(u.order_id.clone(), Arc::clone(u));
                }
                updates.push(WsUpdate::Quote {
                    mode,
                    ts_ms,
                    order: (**u).clone(),
                });
            }
            Event::Fill(f) => {
                let ms = self.mode_mut(mode);
                ms.fills.push(Arc::clone(f));
                ms.live_markout.on_fill(f);
                updates.push(WsUpdate::Fill {
                    mode,
                    ts_ms,
                    fill: (**f).clone(),
                });
            }
            Event::Inventory(inv) => {
                self.mode_mut(mode)
                    .inventory
                    .insert(inv.window, Arc::clone(inv));
            }
            Event::Settlement(s) => {
                self.mode_mut(mode).settlements.push(Arc::clone(s));
            }
            Event::Risk(re) => {
                let ms = self.mode_mut(mode);
                let (event, breaker) = match re {
                    RiskEvent::BreakerTripped { breaker } => {
                        ms.tripped.insert(*breaker);
                        (BreakerEvent::Tripped, *breaker)
                    }
                    RiskEvent::BreakerCleared { breaker } => {
                        ms.tripped.remove(breaker);
                        (BreakerEvent::Cleared, *breaker)
                    }
                    RiskEvent::CancelAllIssued { reason } => {
                        ms.last_cancel_all = Some(*reason);
                        (BreakerEvent::CancelAll, *reason)
                    }
                };
                updates.push(WsUpdate::Breaker {
                    mode,
                    ts_ms,
                    event,
                    breaker,
                });
            }
            Event::Control(ce) => {
                if let ControlEvent::PaperCapitalSet { amount } = ce
                    && mode == Mode::Paper
                {
                    self.params.paper_capital = Some(*amount);
                }
                let ms = self.mode_mut(mode);
                ms.last_control = Some(ce.clone());
                updates.push(WsUpdate::Control {
                    mode,
                    ts_ms,
                    running: ms.running,
                    armed: ms.armed,
                });
            }
            // Price ticks update no view (the model snapshot is what the
            // dashboard shows); analytics already saw them above.
            Event::PriceTick(_) => {}
            // Command audit records are journaled, not part of the live view
            // (the control-state snapshot the orchestrator pushes carries the
            // current state; `GET /api/control/status` reads it).
            Event::ControlAudit(_) => {}
        }
        // Mature any live 5s markouts now past their deadline (the read path then
        // stays a pure lookup). Disjoint field borrows: shared rings (read) +
        // this mode's tracker (write).
        let rings = &self.shared.fair_rings;
        match mode {
            Mode::Paper => self.paper.live_markout.mature(rings, now),
            Mode::Live => self.live.live_markout.mature(rings, now),
        }
        updates
    }

    /// Samples a mode's wallet (the equity-curve magnitude source). Pushes an
    /// equity point and an `equity` update only when the value changed.
    pub(crate) fn set_wallet(
        &mut self,
        mode: Mode,
        wallet: Wallet,
        now: TimestampMs,
    ) -> Vec<WsUpdate> {
        self.last_now = now;
        let equity = wallet.collateral_total;
        let ms = self.mode_mut(mode);
        ms.wallet = Some(wallet);
        if ms.last_equity == Some(equity) {
            return Vec::new();
        }
        ms.last_equity = Some(equity);
        ms.equity.push(EquityPoint { ts: now, equity });
        vec![WsUpdate::Equity {
            mode,
            ts_ms: now.as_millis(),
            equity,
        }]
    }

    pub(crate) fn set_paper_ledger(
        &mut self,
        mode: Mode,
        ledger: PaperLedgerSnapshot,
        now: TimestampMs,
    ) {
        self.last_now = now;
        self.mode_mut(mode).ledger = Some(ledger);
    }

    pub(crate) fn set_risk(&mut self, mode: Mode, snapshot: RiskStateSnapshot, now: TimestampMs) {
        self.last_now = now;
        self.mode_mut(mode).risk_snapshot = Some(snapshot);
    }

    pub(crate) fn set_params(&mut self, params: ParamsView, now: TimestampMs) {
        self.last_now = now;
        if let Some(cap) = params.paper_capital {
            // Keep the overview's paper-capital field consistent.
            self.params.paper_capital = Some(cap);
        }
        self.params = params;
    }

    pub(crate) fn set_control_state(
        &mut self,
        snapshot: crate::command::ControlStateSnapshot,
        now: TimestampMs,
    ) {
        self.last_now = now;
        self.control_state = Some(snapshot);
    }

    pub(crate) fn set_session(
        &mut self,
        mode: Mode,
        running: bool,
        now: TimestampMs,
    ) -> Vec<WsUpdate> {
        self.last_now = now;
        let ms = self.mode_mut(mode);
        ms.running = running;
        vec![WsUpdate::Control {
            mode,
            ts_ms: now.as_millis(),
            running: ms.running,
            armed: ms.armed,
        }]
    }

    pub(crate) fn set_armed(&mut self, mode: Mode, armed: bool, now: TimestampMs) -> Vec<WsUpdate> {
        self.last_now = now;
        let ms = self.mode_mut(mode);
        ms.armed = armed;
        vec![WsUpdate::Control {
            mode,
            ts_ms: now.as_millis(),
            running: ms.running,
            armed: ms.armed,
        }]
    }

    pub(crate) fn set_ws_connected(&mut self, mode: Mode, connected: bool, now: TimestampMs) {
        self.last_now = now;
        self.mode_mut(mode).ws_connected = connected;
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use core_types::{Series, WindowDuration};

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn window_id(open_ms: i64) -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: ts(open_ms),
        }
    }

    #[test]
    fn ring_evicts_oldest_at_cap() {
        let mut ring = Ring::new(3);
        for i in 0..5 {
            ring.push(i);
        }
        let items: Vec<i32> = ring.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 4]); // oldest (0,1) dropped
    }

    #[test]
    fn ring_minimum_capacity_is_one() {
        let mut ring = Ring::new(0);
        ring.push(1);
        ring.push(2);
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn resolved_window_pruned_after_grace() {
        let mut shared = SharedView::new();
        let wid = window_id(1_000_000);
        let mut view = WindowView::new(super::tests_support::market(wid));
        view.resolved_at = Some(ts(1_000_000));
        shared.windows.insert(wid, view);

        // Within grace: kept.
        shared.prune(ts(1_000_000 + WINDOW_GRACE_MS - 1));
        assert_eq!(shared.windows.len(), 1);
        // Past grace: dropped.
        shared.prune(ts(1_000_000 + WINDOW_GRACE_MS + 1));
        assert!(shared.windows.is_empty());
    }

    #[test]
    fn equity_dedups_unchanged_value() {
        let mut data = DashboardData::new(ts(0));
        let wallet = |c: i64| Wallet {
            collateral_available: Dollars::new(rust_decimal::Decimal::from(c)),
            collateral_total: Dollars::new(rust_decimal::Decimal::from(c)),
            positions: vec![],
        };
        let u1 = data.set_wallet(Mode::Paper, wallet(100), ts(1));
        assert_eq!(u1.len(), 1); // first sample → point
        let u2 = data.set_wallet(Mode::Paper, wallet(100), ts(2));
        assert!(u2.is_empty()); // unchanged → no point
        let u3 = data.set_wallet(Mode::Paper, wallet(150), ts(3));
        assert_eq!(u3.len(), 1); // changed → point
        assert_eq!(data.paper.equity.iter().count(), 2);
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    //! Shared builders for the in-crate unit tests.
    use std::sync::Arc;

    use core_types::{
        ConditionId, FeeParams, MarketInfo, ResolutionSource, Size, TickSize, TimestampMs, TokenId,
        TokenPair, WindowId,
    };
    use rust_decimal::dec;

    pub(crate) fn market(window: WindowId) -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window,
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: TimestampMs::from_millis(window.open_time.as_millis() + 300_000),
            strike: Some(dec!(60000)),
            tick_size: TickSize::T001,
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
}
