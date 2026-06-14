//! The canonical order store: the engine's single source of truth about our
//! live orders.
//!
//! It applies authenticated `/ws/user` events ([`OrderStore::apply_order`],
//! [`OrderStore::apply_trade`]) and REST open-orders reconciliations
//! ([`OrderStore::reconcile`]) **idempotently** — duplicate delivery and
//! out-of-order delivery are harmless — and returns the [`StoreEffect`]s the
//! venue event stream should publish. Pure and clock-injected (the async
//! `user_ws` driver is the only IO), so the whole state machine is unit-testable.
//!
//! ## Idempotency model
//!
//! * **State + cumulative fill** are driven by `order` events (`size_matched`)
//!   and the REST reconcile, combined **monotonically**: `filled_size` is the
//!   max of every cumulative observation, the lifecycle advances only through
//!   [`OrderState::can_transition_to`], and a terminal state latches. A stale or
//!   duplicate cumulative is therefore a no-op; a late event after a terminal is
//!   dropped.
//! * **Discrete fills** are driven by `trade` events, deduplicated by trade id.
//!   `Retrying`/`Failed` trades emit nothing and do not consume the id (a retry
//!   may still settle). Our role (taker iff the taker order is ours; maker for
//!   each of our `maker_orders` legs) sets [`Liquidity`] and the fee.
//! * **Missed fills** (a reconnect recovered them via REST): the reconcile emits
//!   a synthetic maker [`Fill`] for the gap (exact for our post-only resting
//!   orders) plus an [`OrderUpdate`] correction, and remembers the covered
//!   shares so a later real trade for them is not double-counted.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use core_types::{
    Decimal, Dollars, Fill, Liquidity, OrderId, OrderState, OrderUpdate, Outcome, Price, Side,
    Size, TickSize, TimestampMs, TokenId, WindowId, taker_fee,
};

use crate::port::RawOpenOrder;
use crate::user_wire::{OrderEventKind, WireOrder, WireTrade};

/// Upper bound on remembered trade ids (FIFO eviction). A 24/7 session keeps
/// only its most recent fills' ids for dedup; an evicted id can only re-fire if
/// the venue redelivers a fill from > this many fills ago, which it does not.
const MAX_SEEN_TRADES: usize = 50_000;

/// Resolves an outcome token to its window context — the seam the orchestrator
/// populates from scheduler/discovery so the store can attribute (and adopt)
/// orders it did not place itself.
///
/// Until it is wired (a follow-up), the store still functions on orders it
/// placed (tracked via [`OrderStore::track`]); unattributed wire/REST orders
/// are counted ([`OrderStore::unattributed`]) and skipped, never fabricated.
pub trait WindowIndex: Send + Sync {
    /// The window context for an outcome token, or `None` if unknown.
    fn resolve(&self, token: &TokenId) -> Option<WindowCtx>;
}

/// The per-token context the [`WindowIndex`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowCtx {
    /// Window the token trades in.
    pub window: WindowId,
    /// Which outcome the token is.
    pub outcome: Outcome,
    /// The market's taker fee rate (`FeeParams.rate`).
    pub fee_rate: Decimal,
    /// The market's current tick (to quantize an adopted order's price).
    pub tick: TickSize,
}

/// One thing the venue event stream should publish, returned from the store's
/// pure mutators. The driver wraps each into a
/// [`VenueEvent`](venue_api::VenueEvent).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StoreEffect {
    /// An order's lifecycle state or cumulative fill changed.
    Order(OrderUpdate),
    /// One execution (real or — on a reconnect recovery — synthetic).
    Fill(Fill),
}

/// The context [`LiveVenue`](crate::LiveVenue) seeds for one order it placed.
/// `last_state`/`last_filled` are the order's state and cumulative fill at the
/// moment it is first tracked (normally `Open`/`PendingNew` and zero).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedOrder {
    /// Window the order trades.
    pub window: WindowId,
    /// Outcome token.
    pub token_id: TokenId,
    /// Which outcome that token is.
    pub outcome: Outcome,
    /// Order side.
    pub side: Side,
    /// Limit price.
    pub price: Price,
    /// Original size in shares.
    pub original_size: Size,
    /// State at the moment of tracking.
    pub last_state: OrderState,
    /// Cumulative filled size at the moment of tracking (normally zero).
    pub last_filled: Size,
}

/// One order's canonical state inside the store.
#[derive(Debug, Clone)]
struct OrderEntry {
    window: WindowId,
    token_id: TokenId,
    outcome: Outcome,
    side: Side,
    price: Price,
    fee_rate: Decimal,
    original_size: Size,
    /// Lifecycle state (advances only via [`OrderState::can_transition_to`]).
    state: OrderState,
    /// Cumulative filled size — the monotone max of every cumulative
    /// observation and the discrete fills we emitted.
    filled_size: Size,
    /// Cumulative shares we have turned into [`Fill`] effects (real + synthetic).
    emitted_fill_size: Size,
    /// Synthetic-emitted shares not yet matched by a real trade leg; a later
    /// real leg for these is suppressed to avoid double-counting.
    synthetic_covered: Size,
    /// True once the state is terminal (latched — no further state change).
    terminal: bool,
}

/// The canonical store of our live orders, keyed by venue order id.
#[derive(Default)]
pub struct OrderStore {
    orders: HashMap<OrderId, OrderEntry>,
    seen_trades: HashSet<String>,
    seen_order: VecDeque<String>,
    emit_synthetic_fills: bool,
    default_fee_rate: Decimal,
    window_index: Option<Arc<dyn WindowIndex>>,
    /// Count of wire/REST events for orders we could neither track nor adopt.
    unattributed: u64,
}

impl OrderStore {
    /// An empty store (synthetic fills on; fee-rate fallback zero — set it from
    /// [`LiveParams`](crate::LiveParams) via [`Self::set_default_fee_rate`]).
    #[must_use]
    pub fn new() -> Self {
        Self {
            emit_synthetic_fills: true,
            ..Self::default()
        }
    }

    /// Sets the per-token resolver used to attribute (and adopt) orders the
    /// store did not place itself.
    pub fn set_window_index(&mut self, index: Arc<dyn WindowIndex>) {
        self.window_index = Some(index);
    }

    /// Toggles synthetic maker fills on reconnect-recovered fills (default on).
    pub fn set_emit_synthetic_fills(&mut self, on: bool) {
        self.emit_synthetic_fills = on;
    }

    /// Sets the fallback taker fee rate (used only when no [`WindowIndex`] entry
    /// supplies a per-market rate).
    pub fn set_default_fee_rate(&mut self, rate: Decimal) {
        self.default_fee_rate = rate;
    }

    /// Begins (or replaces) tracking an order this adapter placed.
    pub fn track(&mut self, order_id: OrderId, tracked: TrackedOrder) {
        let fee_rate = self.fee_rate_for(&tracked.token_id);
        let terminal = tracked.last_state.is_terminal();
        self.orders.insert(
            order_id,
            OrderEntry {
                window: tracked.window,
                token_id: tracked.token_id,
                outcome: tracked.outcome,
                side: tracked.side,
                price: tracked.price,
                fee_rate,
                original_size: tracked.original_size,
                state: tracked.last_state,
                filled_size: tracked.last_filled,
                emitted_fill_size: Size::ZERO,
                synthetic_covered: Size::ZERO,
                terminal,
            },
        );
    }

    /// Marks an order terminal (the cancel paths call this so a later reconcile
    /// does not re-report a known cancellation). No-op for unknown ids.
    pub fn mark_terminal(&mut self, order_id: &OrderId, state: OrderState) {
        if let Some(e) = self.orders.get_mut(order_id)
            && !e.terminal
        {
            e.state = state;
            e.terminal = state.is_terminal();
        }
    }

    /// Number of tracked orders (diagnostics/tests).
    #[must_use]
    pub fn len(&self) -> usize {
        self.orders.len()
    }

    /// True when nothing is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Count of wire/REST events the store could not attribute to a tracked or
    /// adoptable order (e.g. an untracked marketable taker's trade prints).
    #[must_use]
    pub fn unattributed(&self) -> u64 {
        self.unattributed
    }

    /// Applies one `order` event. Idempotent: `size_matched` folds in as a
    /// monotone max, the state advances only through legal transitions, and a
    /// terminal state latches.
    #[must_use]
    pub(crate) fn apply_order(&mut self, ev: &WireOrder, now: TimestampMs) -> Vec<StoreEffect> {
        let Some(entry) = self.orders.get_mut(&ev.order_id) else {
            // Adoption from wire events is deferred; count and skip (never
            // fabricate a window).
            self.unattributed += 1;
            return Vec::new();
        };
        // A terminal order drops late/reordered order events; a fill that races
        // a cancel still flows through `apply_trade` (concrete money).
        if entry.terminal {
            return Vec::new();
        }
        let prev_filled = entry.filled_size;
        let cumulative = to_size(ev.size_matched);
        entry.filled_size = size_max(entry.filled_size, cumulative);
        let desired = match ev.kind {
            OrderEventKind::Cancellation => OrderState::Canceled,
            OrderEventKind::Placement | OrderEventKind::Update => {
                state_from_fill(entry.filled_size, entry.original_size)
            }
        };
        let state_changed = advance_state(entry, desired, &ev.order_id);
        let filled_changed = entry.filled_size.as_decimal() != prev_filled.as_decimal();
        if state_changed || filled_changed {
            vec![StoreEffect::Order(order_update(
                &ev.order_id,
                entry,
                now,
                Some(ev.ts),
            ))]
        } else {
            Vec::new()
        }
    }

    /// Applies one `trade` event. Idempotent: deduplicated by trade id;
    /// `Retrying`/`Failed` emit nothing and keep the id unseen.
    #[must_use]
    pub(crate) fn apply_trade(&mut self, ev: &WireTrade, now: TimestampMs) -> Vec<StoreEffect> {
        if ev.status.is_inflight_or_failed() {
            return Vec::new();
        }
        let taker_is_ours = ev
            .taker_order_id
            .as_ref()
            .is_some_and(|id| self.orders.contains_key(id));
        let maker_is_ours = ev
            .maker_orders
            .iter()
            .any(|m| self.orders.contains_key(&m.order_id));
        if !taker_is_ours && !maker_is_ours {
            self.unattributed += 1;
            return Vec::new();
        }
        if !self.remember_trade(&ev.trade_id) {
            return Vec::new();
        }

        let mut effects = Vec::new();
        if taker_is_ours && let Some(taker) = ev.taker_order_id.clone() {
            self.apply_trade_leg(
                &taker,
                to_size(ev.size),
                Liquidity::Taker,
                ev,
                now,
                &mut effects,
            );
        }
        for maker in &ev.maker_orders {
            if self.orders.contains_key(&maker.order_id) {
                self.apply_trade_leg(
                    &maker.order_id,
                    to_size(maker.matched_amount),
                    Liquidity::Maker,
                    ev,
                    now,
                    &mut effects,
                );
            }
        }
        effects
    }

    /// Diffs a fresh open-orders poll against the store. Called on every
    /// (re)connect (and a low-frequency safety tick): recovers fills missed
    /// while disconnected (a synthetic maker [`Fill`] for the gap plus an
    /// [`OrderUpdate`] correction), settles vanished orders, and — when a
    /// [`WindowIndex`] is wired — adopts open orders it never tracked.
    #[must_use]
    pub(crate) fn reconcile(
        &mut self,
        polled: &[RawOpenOrder],
        now: TimestampMs,
    ) -> Vec<StoreEffect> {
        // Pass 1: adopt unknown open orders we can attribute (inert until a
        // WindowIndex is wired — see the type docs).
        for po in polled {
            if !self.orders.contains_key(&po.order_id) {
                self.try_adopt(po);
            }
        }
        // Pass 2: diff every tracked order against the poll.
        let polled_by_id: HashMap<&OrderId, &RawOpenOrder> =
            polled.iter().map(|p| (&p.order_id, p)).collect();
        let ids: Vec<OrderId> = self.orders.keys().cloned().collect();
        let mut effects = Vec::new();
        for id in ids {
            let polled = polled_by_id.get(&id).copied();
            self.reconcile_one(&id, polled, now, &mut effects);
        }
        effects
    }

    /// Resolves the fee rate for a token: per-market via the index, else the
    /// configured fallback.
    fn fee_rate_for(&self, token: &TokenId) -> Decimal {
        self.window_index
            .as_ref()
            .and_then(|idx| idx.resolve(token))
            .map_or(self.default_fee_rate, |c| c.fee_rate)
    }

    /// Records a trade id for dedup (FIFO-bounded). Returns `false` if already
    /// seen.
    fn remember_trade(&mut self, id: &str) -> bool {
        if self.seen_trades.contains(id) {
            return false;
        }
        self.seen_trades.insert(id.to_owned());
        self.seen_order.push_back(id.to_owned());
        if self.seen_order.len() > MAX_SEEN_TRADES
            && let Some(old) = self.seen_order.pop_front()
        {
            self.seen_trades.remove(&old);
        }
        true
    }

    /// Applies one fill leg (the caller has verified `order_id` is tracked).
    fn apply_trade_leg(
        &mut self,
        order_id: &OrderId,
        leg_size: Size,
        liquidity: Liquidity,
        ev: &WireTrade,
        now: TimestampMs,
        effects: &mut Vec<StoreEffect>,
    ) {
        let Some(entry) = self.orders.get_mut(order_id) else {
            return;
        };
        if leg_size.as_decimal().is_zero() {
            return;
        }
        // Suppress shares already emitted synthetically by a prior reconcile.
        let cover = leg_size.min(entry.synthetic_covered);
        entry.synthetic_covered = entry.synthetic_covered.saturating_sub(cover);
        // Never emit beyond the order's original size.
        let room = entry.original_size.saturating_sub(entry.emitted_fill_size);
        let chargeable = leg_size.saturating_sub(cover).min(room);
        let prev_filled = entry.filled_size;
        if !chargeable.as_decimal().is_zero() {
            entry.emitted_fill_size = entry.emitted_fill_size + chargeable;
            let fee = match liquidity {
                Liquidity::Maker => Dollars::ZERO,
                Liquidity::Taker => taker_fee(chargeable, entry.fee_rate, entry.price),
            };
            effects.push(StoreEffect::Fill(Fill {
                order_id: order_id.clone(),
                trade_id: Some(ev.trade_id.clone()),
                window: entry.window,
                token_id: entry.token_id.clone(),
                outcome: entry.outcome,
                side: entry.side,
                price: entry.price,
                size: chargeable,
                liquidity,
                fee,
                ts_venue: ev.ts,
                ts_local: now,
            }));
        }
        entry.filled_size = size_max(entry.filled_size, entry.emitted_fill_size);
        let desired = state_from_fill(entry.filled_size, entry.original_size);
        let state_changed = advance_state(entry, desired, order_id);
        let filled_changed = entry.filled_size.as_decimal() != prev_filled.as_decimal();
        if state_changed || filled_changed {
            effects.push(StoreEffect::Order(order_update(
                order_id,
                entry,
                now,
                Some(ev.ts),
            )));
        }
    }

    /// Reconciles one tracked order against its poll entry (or its absence).
    fn reconcile_one(
        &mut self,
        id: &OrderId,
        polled: Option<&RawOpenOrder>,
        now: TimestampMs,
        effects: &mut Vec<StoreEffect>,
    ) {
        let emit_synthetic = self.emit_synthetic_fills;
        let Some(entry) = self.orders.get_mut(id) else {
            return;
        };
        if entry.terminal {
            return;
        }
        let prev_filled = entry.filled_size;
        let (desired, cumulative) = match polled {
            Some(po) => {
                entry.filled_size = size_max(entry.filled_size, po.size_matched);
                (
                    open_status_to_state(&po.status, entry.filled_size, entry.original_size),
                    entry.filled_size,
                )
            }
            None => {
                // Left the book → terminal. Filled if our fill covers the
                // original, else cancelled.
                let filled = entry.filled_size;
                let state = if !entry.original_size.as_decimal().is_zero()
                    && filled.as_decimal() >= entry.original_size.as_decimal()
                {
                    OrderState::Filled
                } else {
                    OrderState::Canceled
                };
                (state, filled)
            }
        };
        // Emit fills for shares filled-but-not-yet-emitted (missed during a gap).
        let gap = cumulative.saturating_sub(entry.emitted_fill_size);
        if !gap.as_decimal().is_zero() {
            if emit_synthetic {
                entry.emitted_fill_size = entry.emitted_fill_size + gap;
                entry.synthetic_covered = entry.synthetic_covered + gap;
                effects.push(StoreEffect::Fill(Fill {
                    order_id: id.clone(),
                    trade_id: None,
                    window: entry.window,
                    token_id: entry.token_id.clone(),
                    outcome: entry.outcome,
                    side: entry.side,
                    price: entry.price,
                    size: gap,
                    liquidity: Liquidity::Maker,
                    fee: Dollars::ZERO,
                    ts_venue: now,
                    ts_local: now,
                }));
            } else {
                // Corrections-only: advance emitted so the gap is not
                // re-detected, but publish no discrete fill.
                entry.emitted_fill_size = size_max(entry.emitted_fill_size, cumulative);
            }
        }
        let state_changed = advance_state(entry, desired, id);
        let filled_changed = entry.filled_size.as_decimal() != prev_filled.as_decimal();
        if state_changed || filled_changed {
            effects.push(StoreEffect::Order(order_update(id, entry, now, None)));
        }
    }

    /// Attempts to adopt an open order the store never tracked, resolving its
    /// window via the [`WindowIndex`]. Inert when no index is wired.
    fn try_adopt(&mut self, po: &RawOpenOrder) {
        let Some(ctx) = self
            .window_index
            .as_ref()
            .and_then(|idx| idx.resolve(&po.token_id))
        else {
            self.unattributed += 1;
            return;
        };
        let Ok(price) = Price::on_grid(po.price, ctx.tick) else {
            // A wire price off the resolved tick grid — refuse to fabricate.
            self.unattributed += 1;
            tracing::warn!(
                target: "venue::live",
                order_id = %po.order_id,
                price = %po.price,
                "skipping adoption of an open order whose price is off the tick grid"
            );
            return;
        };
        self.orders.insert(
            po.order_id.clone(),
            OrderEntry {
                window: ctx.window,
                token_id: po.token_id.clone(),
                outcome: ctx.outcome,
                side: po.side,
                price,
                fee_rate: ctx.fee_rate,
                original_size: po.original_size,
                state: OrderState::PendingNew,
                filled_size: Size::ZERO,
                emitted_fill_size: Size::ZERO,
                synthetic_covered: Size::ZERO,
                terminal: false,
            },
        );
    }
}

/// The larger of two sizes (by value).
fn size_max(a: Size, b: Size) -> Size {
    if a.as_decimal() >= b.as_decimal() {
        a
    } else {
        b
    }
}

/// A non-negative size, flooring any garbage at zero.
fn to_size(d: Decimal) -> Size {
    Size::new(d).unwrap_or(Size::ZERO)
}

/// The lifecycle state implied purely by fill progress.
fn state_from_fill(filled: Size, original: Size) -> OrderState {
    if !original.as_decimal().is_zero() && filled.as_decimal() >= original.as_decimal() {
        OrderState::Filled
    } else if !filled.as_decimal().is_zero() {
        OrderState::PartiallyFilled
    } else {
        OrderState::Open
    }
}

/// Advances an entry's state toward `desired`, honoring the terminal latch and
/// the transition table. Returns whether the state changed. An illegal
/// transition is dropped + warned (a missed message; the next reconcile
/// resyncs).
fn advance_state(entry: &mut OrderEntry, desired: OrderState, order_id: &OrderId) -> bool {
    if entry.terminal || entry.state == desired {
        return false;
    }
    if entry.state.can_transition_to(desired) {
        entry.state = desired;
        entry.terminal = desired.is_terminal();
        true
    } else {
        tracing::warn!(
            target: "venue::live",
            order_id = %order_id,
            from = ?entry.state,
            to = ?desired,
            "skipping illegal order-state transition"
        );
        false
    }
}

/// Builds an [`OrderUpdate`] snapshot from an entry.
fn order_update(
    order_id: &OrderId,
    entry: &OrderEntry,
    now: TimestampMs,
    ts_venue: Option<TimestampMs>,
) -> OrderUpdate {
    OrderUpdate {
        order_id: order_id.clone(),
        window: entry.window,
        token_id: entry.token_id.clone(),
        side: entry.side,
        state: entry.state,
        price: entry.price,
        original_size: entry.original_size,
        filled_size: entry.filled_size,
        reject_reason: None,
        ts_venue,
        ts_local: now,
    }
}

/// Maps a venue open-order status string + fill progress to an [`OrderState`].
fn open_status_to_state(status: &str, filled: Size, original: Size) -> OrderState {
    match status.to_ascii_lowercase().as_str() {
        "canceled" | "cancelled" => OrderState::Canceled,
        "matched" => {
            if !original.as_decimal().is_zero() && filled.as_decimal() >= original.as_decimal() {
                OrderState::Filled
            } else {
                OrderState::PartiallyFilled
            }
        }
        // live / unmatched / delayed / anything else: still working.
        _ => {
            if filled.as_decimal().is_zero() {
                OrderState::Open
            } else {
                OrderState::PartiallyFilled
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::{Asset, ConditionId, RoundDir, Series, WindowDuration};
    use rust_decimal::dec;

    use super::*;
    use crate::user_wire::{
        ParsedUserFrame, TradeStatus, WireMakerOrder, WireUserEvent, parse_user_frame,
    };

    const NOW: TimestampMs = TimestampMs::from_millis(2_000_000);

    fn oid(s: &str) -> OrderId {
        OrderId::new(s).unwrap()
    }
    fn tok() -> TokenId {
        TokenId::new("123").unwrap()
    }
    fn cid() -> ConditionId {
        ConditionId::new(format!("0x{}", "ab".repeat(32))).unwrap()
    }
    fn size(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn price() -> Price {
        Price::quantize(dec!(0.40), TickSize::T001, RoundDir::Down).unwrap()
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_000_000),
        }
    }

    fn store_tracking(id: &str) -> OrderStore {
        let mut store = OrderStore::new();
        store.set_default_fee_rate(dec!(0.07));
        store.track(
            oid(id),
            TrackedOrder {
                window: window(),
                token_id: tok(),
                outcome: Outcome::Up,
                side: Side::Buy,
                price: price(),
                original_size: size(dec!(10)),
                last_state: OrderState::Open,
                last_filled: Size::ZERO,
            },
        );
        store
    }

    fn order_ev(id: &str, kind: OrderEventKind, matched: Decimal) -> WireOrder {
        WireOrder {
            kind,
            order_id: oid(id),
            condition_id: cid(),
            token_id: tok(),
            side: Side::Buy,
            price: dec!(0.40),
            original_size: dec!(10),
            size_matched: matched,
            ts: TimestampMs::from_millis(1_500_000),
        }
    }

    fn maker_trade(id: &str, maker_order: &str, amount: Decimal) -> WireTrade {
        WireTrade {
            trade_id: id.to_owned(),
            taker_order_id: Some(oid("counterparty")),
            condition_id: cid(),
            token_id: tok(),
            side: Side::Sell,
            size: amount,
            price: dec!(0.40),
            status: TradeStatus::Matched,
            maker_orders: vec![WireMakerOrder {
                order_id: oid(maker_order),
                token_id: tok(),
                matched_amount: amount,
                price: dec!(0.40),
            }],
            ts: TimestampMs::from_millis(1_600_000),
        }
    }

    fn raw_open(id: &str, status: &str, matched: Decimal) -> RawOpenOrder {
        RawOpenOrder {
            order_id: oid(id),
            status: status.to_owned(),
            original_size: size(dec!(10)),
            size_matched: size(matched),
            token_id: tok(),
            side: Side::Buy,
            price: dec!(0.40),
            condition_id: cid(),
        }
    }

    fn one_order(effects: &[StoreEffect]) -> &OrderUpdate {
        effects
            .iter()
            .find_map(|e| match e {
                StoreEffect::Order(u) => Some(u),
                StoreEffect::Fill(_) => None,
            })
            .expect("an order update")
    }
    fn one_fill(effects: &[StoreEffect]) -> &Fill {
        effects
            .iter()
            .find_map(|e| match e {
                StoreEffect::Fill(f) => Some(f),
                StoreEffect::Order(_) => None,
            })
            .expect("a fill")
    }

    // ---- lifecycle [A] ----

    #[test]
    fn order_lifecycle_placement_partial_full() {
        let mut store = store_tracking("o1");
        // PLACEMENT (already Open) with no fill → no change.
        assert!(
            store
                .apply_order(&order_ev("o1", OrderEventKind::Placement, dec!(0)), NOW)
                .is_empty()
        );
        // UPDATE 3/10 → PartiallyFilled.
        let e = store.apply_order(&order_ev("o1", OrderEventKind::Update, dec!(3)), NOW);
        let u = one_order(&e);
        assert_eq!(u.state, OrderState::PartiallyFilled);
        assert_eq!(u.filled_size, size(dec!(3)));
        // UPDATE 10/10 → Filled.
        let e = store.apply_order(&order_ev("o1", OrderEventKind::Update, dec!(10)), NOW);
        assert_eq!(one_order(&e).state, OrderState::Filled);
    }

    #[test]
    fn order_cancellation_is_terminal() {
        let mut store = store_tracking("o1");
        let e = store.apply_order(&order_ev("o1", OrderEventKind::Cancellation, dec!(0)), NOW);
        assert_eq!(one_order(&e).state, OrderState::Canceled);
        // A later update cannot revive it.
        assert!(
            store
                .apply_order(&order_ev("o1", OrderEventKind::Update, dec!(5)), NOW)
                .is_empty()
        );
    }

    #[test]
    fn trade_maker_fill_zero_fee_and_taker_fill_charged() {
        // Maker leg: fee zero, liquidity Maker, price = our limit.
        let mut store = store_tracking("o1");
        let e = store.apply_trade(&maker_trade("t1", "o1", dec!(4)), NOW);
        let f = one_fill(&e);
        assert_eq!(f.liquidity, Liquidity::Maker);
        assert_eq!(f.size, size(dec!(4)));
        assert_eq!(f.fee, Dollars::ZERO);
        assert_eq!(f.trade_id.as_deref(), Some("t1"));
        assert_eq!(one_order(&e).state, OrderState::PartiallyFilled);

        // Taker leg: fee = taker_fee(size, rate, price).
        let mut store = store_tracking("o2");
        let trade = WireTrade {
            trade_id: "t2".to_owned(),
            taker_order_id: Some(oid("o2")),
            condition_id: cid(),
            token_id: tok(),
            side: Side::Buy,
            size: dec!(5),
            price: dec!(0.40),
            status: TradeStatus::Matched,
            maker_orders: vec![],
            ts: NOW,
        };
        let e = store.apply_trade(&trade, NOW);
        let f = one_fill(&e);
        assert_eq!(f.liquidity, Liquidity::Taker);
        assert_eq!(f.fee, taker_fee(size(dec!(5)), dec!(0.07), price()));
    }

    #[test]
    fn failed_trade_emits_nothing_and_retry_then_match_still_fills() {
        let mut store = store_tracking("o1");
        let mut failed = maker_trade("t1", "o1", dec!(4));
        failed.status = TradeStatus::Failed;
        assert!(store.apply_trade(&failed, NOW).is_empty());
        // Same trade id later MATCHED → still fills (id was not consumed).
        let e = store.apply_trade(&maker_trade("t1", "o1", dec!(4)), NOW);
        assert_eq!(one_fill(&e).size, size(dec!(4)));
    }

    // ---- duplicate delivery [C] ----

    #[test]
    fn duplicate_order_event_is_a_no_op() {
        let mut store = store_tracking("o1");
        let ev = order_ev("o1", OrderEventKind::Update, dec!(3));
        assert!(!store.apply_order(&ev, NOW).is_empty());
        assert!(
            store.apply_order(&ev, NOW).is_empty(),
            "duplicate is a no-op"
        );
    }

    #[test]
    fn duplicate_trade_event_is_a_no_op() {
        let mut store = store_tracking("o1");
        let ev = maker_trade("t1", "o1", dec!(4));
        assert_eq!(store.apply_trade(&ev, NOW).len(), 2); // fill + order update
        assert!(
            store.apply_trade(&ev, NOW).is_empty(),
            "duplicate trade id deduped"
        );
    }

    // ---- out-of-order delivery [C] ----

    #[test]
    fn stale_smaller_cumulative_is_ignored() {
        let mut store = store_tracking("o1");
        let _ = store.apply_order(&order_ev("o1", OrderEventKind::Update, dec!(7)), NOW);
        // A reordered, smaller cumulative arrives late → ignored.
        assert!(
            store
                .apply_order(&order_ev("o1", OrderEventKind::Update, dec!(3)), NOW)
                .is_empty()
        );
    }

    #[test]
    fn late_event_after_terminal_is_dropped() {
        let mut store = store_tracking("o1");
        let _ = store.apply_order(&order_ev("o1", OrderEventKind::Cancellation, dec!(0)), NOW);
        // Placement arriving after the cancellation must not revive the order.
        assert!(
            store
                .apply_order(&order_ev("o1", OrderEventKind::Placement, dec!(0)), NOW)
                .is_empty()
        );
    }

    // ---- reconnect / missed-fill [B] ----

    #[test]
    fn reconcile_missed_fill_emits_synthetic_fill_and_correction_then_dedups_real() {
        let mut store = store_tracking("o1");
        // A fill happened while disconnected; REST shows 6/10 matched.
        let effects = store.reconcile(&[raw_open("o1", "LIVE", dec!(6))], NOW);
        let f = one_fill(&effects);
        assert_eq!(f.liquidity, Liquidity::Maker);
        assert_eq!(f.size, size(dec!(6)));
        assert_eq!(f.trade_id, None, "synthetic fill has no trade id");
        assert_eq!(f.fee, Dollars::ZERO);
        let u = one_order(&effects);
        assert_eq!(u.state, OrderState::PartiallyFilled);
        assert_eq!(u.filled_size, size(dec!(6)));

        // The real trade for those same 6 shares arrives after the reconnect →
        // suppressed (no second fill), no double-count.
        let real = store.apply_trade(&maker_trade("t-late", "o1", dec!(6)), NOW);
        assert!(
            real.iter().all(|e| matches!(e, StoreEffect::Order(_))),
            "the real trade for already-covered shares emits no Fill, got {real:?}"
        );
    }

    #[test]
    fn reconcile_corrections_only_mode_emits_no_fill() {
        let mut store = store_tracking("o1");
        store.set_emit_synthetic_fills(false);
        let effects = store.reconcile(&[raw_open("o1", "LIVE", dec!(6))], NOW);
        assert!(
            effects.iter().all(|e| matches!(e, StoreEffect::Order(_))),
            "corrections-only emits no synthetic fill"
        );
        assert_eq!(one_order(&effects).filled_size, size(dec!(6)));
    }

    #[test]
    fn vanished_order_is_filled_or_canceled() {
        // Full fill seen, then it left the book → Filled.
        let mut store = store_tracking("o1");
        let _ = store.apply_order(&order_ev("o1", OrderEventKind::Update, dec!(10)), NOW);
        let e = store.reconcile(&[], NOW);
        assert!(e.is_empty(), "already Filled (terminal) → nothing new");

        // Partial fill, then vanished → Canceled.
        let mut store = store_tracking("o2");
        let _ = store.apply_order(&order_ev("o2", OrderEventKind::Update, dec!(4)), NOW);
        let e = store.reconcile(&[], NOW);
        assert_eq!(one_order(&e).state, OrderState::Canceled);
        assert_eq!(one_order(&e).filled_size, size(dec!(4)));
    }

    #[test]
    fn trade_then_reconcile_does_not_double_count() {
        // Normal live ordering: trade first (emits real fill), reconcile after
        // sees the same cumulative → no synthetic fill.
        let mut store = store_tracking("o1");
        let _ = store.apply_trade(&maker_trade("t1", "o1", dec!(4)), NOW);
        let e = store.reconcile(&[raw_open("o1", "LIVE", dec!(4))], NOW);
        assert!(
            e.iter().all(|x| matches!(x, StoreEffect::Order(_))),
            "no synthetic fill when the real trade already covered it, got {e:?}"
        );
    }

    // ---- attribution / adoption ----

    #[test]
    fn unknown_order_events_are_counted_not_fabricated() {
        let mut store = OrderStore::new();
        assert!(
            store
                .apply_order(&order_ev("ghost", OrderEventKind::Update, dec!(1)), NOW)
                .is_empty()
        );
        let trade = WireTrade {
            taker_order_id: Some(oid("ghost-taker")),
            maker_orders: vec![],
            ..maker_trade("t1", "ghost", dec!(1))
        };
        // maker "ghost" not tracked, taker not tracked → unattributed.
        assert!(store.apply_trade(&trade, NOW).is_empty());
        assert_eq!(store.unattributed(), 2);
    }

    struct FakeIndex;
    impl WindowIndex for FakeIndex {
        fn resolve(&self, _token: &TokenId) -> Option<WindowCtx> {
            Some(WindowCtx {
                window: window(),
                outcome: Outcome::Up,
                fee_rate: dec!(0.07),
                tick: TickSize::T001,
            })
        }
    }

    #[test]
    fn reconcile_adopts_unknown_order_when_index_is_wired() {
        let mut store = OrderStore::new();
        store.set_window_index(Arc::new(FakeIndex));
        // An order we never tracked (e.g. a prior-process survivor), 3/10 filled.
        let effects = store.reconcile(&[raw_open("orphan", "LIVE", dec!(3))], NOW);
        assert_eq!(store.len(), 1, "the orphan was adopted");
        let f = one_fill(&effects);
        assert_eq!(f.size, size(dec!(3)));
        assert_eq!(f.liquidity, Liquidity::Maker);
        assert_eq!(one_order(&effects).state, OrderState::PartiallyFilled);
        assert_eq!(store.unattributed(), 0);
    }

    // ---- fixture-driven full lifecycle (committed event shapes) ----

    fn fx_order(json: &str) -> WireOrder {
        match parse_user_frame(json, NOW) {
            ParsedUserFrame::Events(mut e) => match e.remove(0) {
                Ok(WireUserEvent::Order(o)) => o,
                other => panic!("expected an order, got {other:?}"),
            },
            other => panic!("expected events, got {other:?}"),
        }
    }
    fn fx_trade(json: &str) -> WireTrade {
        match parse_user_frame(json, NOW) {
            ParsedUserFrame::Events(mut e) => match e.remove(0) {
                Ok(WireUserEvent::Trade(t)) => t,
                other => panic!("expected a trade, got {other:?}"),
            },
            other => panic!("expected events, got {other:?}"),
        }
    }
    fn track(
        store: &mut OrderStore,
        id: &str,
        token: &str,
        outcome: Outcome,
        side: Side,
        original: Decimal,
    ) {
        store.track(
            oid(id),
            TrackedOrder {
                window: window(),
                token_id: TokenId::new(token).unwrap(),
                outcome,
                side,
                price: price(),
                original_size: size(original),
                last_state: OrderState::Open,
                last_filled: Size::ZERO,
            },
        );
    }

    #[test]
    fn fixture_driven_lifecycle_across_maker_and_taker() {
        let mut store = OrderStore::new();
        store.set_default_fee_rate(dec!(0.07));
        track(
            &mut store,
            "0xorder-up-1",
            "111111111",
            Outcome::Up,
            Side::Buy,
            dec!(20),
        );
        track(
            &mut store,
            "0xorder-down-1",
            "222222222",
            Outcome::Down,
            Side::Sell,
            dec!(10),
        );

        // Placement (already Open, nothing filled) → no effect.
        let e = store.apply_order(
            &fx_order(include_str!("../tests/fixtures/user_order_placement.json")),
            NOW,
        );
        assert!(e.is_empty());

        // Order UPDATE carrying cumulative 8 → PartiallyFilled.
        let e = store.apply_order(
            &fx_order(include_str!(
                "../tests/fixtures/user_order_update_partial.json"
            )),
            NOW,
        );
        assert_eq!(one_order(&e).state, OrderState::PartiallyFilled);
        assert_eq!(one_order(&e).filled_size, size(dec!(8)));

        // Taker trade (our Up order is the taker) → a Taker fill.
        let e = store.apply_trade(
            &fx_trade(include_str!(
                "../tests/fixtures/user_trade_matched_taker.json"
            )),
            NOW,
        );
        let f = one_fill(&e);
        assert_eq!(f.liquidity, Liquidity::Taker);
        assert_eq!(f.size, size(dec!(5)));

        // Maker trade (our Down order is a maker) → a zero-fee Maker fill.
        let e = store.apply_trade(
            &fx_trade(include_str!(
                "../tests/fixtures/user_trade_matched_maker.json"
            )),
            NOW,
        );
        let f = one_fill(&e);
        assert_eq!(f.liquidity, Liquidity::Maker);
        assert_eq!(f.size, size(dec!(7)));
        assert_eq!(f.fee, Dollars::ZERO);

        // FAILED trade for the same Up-order trade id → emits nothing.
        assert!(
            store
                .apply_trade(
                    &fx_trade(include_str!("../tests/fixtures/user_trade_failed.json")),
                    NOW
                )
                .is_empty()
        );

        // Cancellation → terminal Canceled.
        let e = store.apply_order(
            &fx_order(include_str!(
                "../tests/fixtures/user_order_cancellation.json"
            )),
            NOW,
        );
        assert_eq!(one_order(&e).state, OrderState::Canceled);
    }

    #[test]
    fn illegal_transition_from_reconcile_is_dropped() {
        let mut store = store_tracking("o1");
        // Drive to Filled.
        let _ = store.apply_order(&order_ev("o1", OrderEventKind::Update, dec!(10)), NOW);
        // A poll claiming it is back to LIVE/unfilled cannot un-terminal it.
        assert!(
            store
                .reconcile(&[raw_open("o1", "LIVE", dec!(0))], NOW)
                .is_empty()
        );
    }
}
