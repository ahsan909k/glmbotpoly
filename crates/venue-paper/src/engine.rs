//! The sans-IO paper matching engine: the fill simulator (CLAUDE.md §9).
//!
//! Pure and clock-injected — every mutator takes a `now: TimestampMs` and
//! returns the [`PaperEffect`]s the venue event stream should publish, exactly
//! like `venue-live`'s `OrderStore`. The async driver ([`crate::driver`]) is the
//! only IO. This split lets the conservative fill rules be exercised by plain,
//! deterministic scripted tests with no tokio.
//!
//! ## The cardinal rule (§9): never fill MORE or SOONER than reality
//!
//! - Simulated latencies are one-sided-upward, so a paper action never happens
//!   before its configured mean.
//! - A passive order activates only after its placement latency, then queues
//!   **behind all displayed size** at its level — a position that only ever
//!   drains, never grows.
//! - A resting order fills ONLY from real prints whose aggressor is the opposite
//!   side, at-or-through its price, sized by the print as a finite shared budget
//!   (no double-counting). Matching is pure price-and-queue — never gated on
//!   whether the book snapshot already reflected the trade (prints can arrive
//!   before their snapshot).
//! - A cancel only takes effect after its round-trip; prints in the meantime
//!   still fill the stale quote (we lose the race).
//! - Marketable orders walk real displayed depth, respect the worst-price limit,
//!   pay real taker fees, and deplete the local book so simultaneous paper
//!   takers can't double-eat depth.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use core_types::{
    BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Fill, Liquidity, MarketInfo, NewOrder,
    OrderId, OrderQty, OrderState, OrderUpdate, Price, Side, Size, TickSize, TimeInForce,
    TimestampMs, TokenId, TopOfBook, WindowId, WindowLifecycle, taker_fee,
};
use venue_api::{Accepted, CancelReport, NotCanceled, RejectReason, VenueError, Wallet};

use crate::book::SimBook;
use crate::latency::LatencySampler;
use crate::params::PaperParams;
use crate::wallet::{PaperLedgerSnapshot, PaperWallet};

/// The venue's minimum GTD lifetime (the §7 security threshold): a paper GTD
/// order cannot expire before `now + 60s`.
const GTD_MIN_LEAD: DurationMs = DurationMs::from_millis(60_000);

/// The simulated maker-rebate credit cycle period: accrued rebate is credited
/// to cash once per 24h (§9), subject to the $1 minimum accrual.
const REBATE_PERIOD: DurationMs = DurationMs::from_secs(86_400);

/// One thing the venue event stream should publish, returned from the engine's
/// pure mutators (mirrors `venue_live::StoreEffect`). The driver wraps each into
/// a [`VenueEvent`](venue_api::VenueEvent).
#[derive(Debug, Clone, PartialEq)]
pub enum PaperEffect {
    /// An order's lifecycle state or cumulative fill changed.
    Order(OrderUpdate),
    /// One execution (maker on a resting fill, taker on a marketable walk).
    Fill(Fill),
}

/// A timed transition a non-terminal order is waiting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeadlineKind {
    /// A `PendingNew` order reaches the end of its placement latency.
    Activation,
    /// A requested cancel reaches the end of its round-trip.
    Cancel,
    /// A GTD order reaches its expiration.
    Expiry,
}

impl DeadlineKind {
    /// Tie-break priority when two deadlines fall on the same instant
    /// (activation before cancel before expiry).
    const fn priority(self) -> u8 {
        match self {
            Self::Activation => 0,
            Self::Cancel => 1,
            Self::Expiry => 2,
        }
    }
}

/// One simulated order's state inside the engine.
#[derive(Debug, Clone)]
struct SimOrder {
    /// Venue-assigned id (`paper-{n}`).
    order_id: OrderId,
    /// The original request (carries window/token/outcome/side/price/qty/tif).
    order: NewOrder,
    /// Lifecycle state.
    state: OrderState,
    /// Original size in shares (resting & marketable SELL); for a marketable
    /// BUY (notional) it is set to the executed shares at activation.
    original_size: Size,
    /// Cumulative filled shares.
    filled: Size,
    /// Displayed size still ahead of us in queue at our level (only drains).
    queue_ahead: Size,
    /// When a `PendingNew` order becomes active (place time + placement latency).
    activate_at: TimestampMs,
    /// When a requested cancel lands (cancel time + cancel latency), if any.
    cancel_effect_at: Option<TimestampMs>,
    /// GTD expiration, if any (floored to `now + 60s` at placement).
    expires_at: Option<TimestampMs>,
    /// Monotonic placement sequence — the time-priority tiebreak.
    seq: u64,
}

impl SimOrder {
    fn remaining(&self) -> Size {
        self.original_size.saturating_sub(self.filled)
    }
}

/// The sans-IO paper matching engine.
pub struct MatchEngine {
    params: PaperParams,
    sampler: LatencySampler,
    /// Monotonic counter for order ids and the time-priority tiebreak.
    next_seq: u64,
    /// Monotonic counter for synthetic paper trade ids.
    next_trade: u64,
    /// Latest market metadata per window (fee params, condition id, tokens).
    markets: HashMap<core_types::WindowId, Arc<MarketInfo>>,
    /// Per-market taker fee rate keyed by outcome token.
    fee_rate_by_token: HashMap<TokenId, Decimal>,
    /// Latest local book mirror per outcome token.
    books: HashMap<TokenId, SimBook>,
    /// All orders, keyed by venue id (terminal ones are retained for idempotent
    /// cancel reporting; a long-session prune hook is a documented follow-up).
    orders: HashMap<OrderId, SimOrder>,
    /// Windows already settled (idempotency for a repeated `Resolved`).
    settled: HashSet<WindowId>,
    /// When the next daily maker-rebate credit is due; armed on the first
    /// maker-rebate accrual (`new` has no clock), rolled forward thereafter.
    rebate_next_credit_at: Option<TimestampMs>,
    /// The paper wallet/ledger.
    wallet: PaperWallet,
}

impl MatchEngine {
    /// Builds an engine from its parameters.
    #[must_use]
    pub fn new(params: PaperParams) -> Self {
        let sampler = LatencySampler::from_optional_seed(params.rng_seed);
        let wallet = PaperWallet::new(params.starting_capital);
        Self {
            params,
            sampler,
            next_seq: 0,
            next_trade: 0,
            markets: HashMap::new(),
            fee_rate_by_token: HashMap::new(),
            books: HashMap::new(),
            orders: HashMap::new(),
            settled: HashSet::new(),
            rebate_next_credit_at: None,
            wallet,
        }
    }

    // ---- read-side market data --------------------------------------------

    /// Applies a full L2 snapshot (depth ground truth). No fills here — trades
    /// drive fills (`on_trade`), so a snapshot only refreshes state for future
    /// activations and marketable walks.
    pub fn on_book(&mut self, snap: &BookSnapshot) -> Vec<PaperEffect> {
        self.books
            .entry(snap.token_id.clone())
            .or_default()
            .apply_snapshot(snap);
        Vec::new()
    }

    /// Refreshes a token's top of book (keeps the post-only cross check fresh).
    pub fn on_top(&mut self, token: &TokenId, top: &TopOfBook) -> Vec<PaperEffect> {
        self.books
            .entry(token.clone())
            .or_default()
            .apply_top(top.bid, top.ask, top.ts);
        Vec::new()
    }

    /// Tick-size flips don't affect paper matching (orders arrive
    /// pre-normalized; paper never re-validates the grid). No-op by design.
    pub fn on_tick_size(
        &mut self,
        _condition: &ConditionId,
        _new_tick: TickSize,
        _ts: TimestampMs,
    ) -> Vec<PaperEffect> {
        Vec::new()
    }

    /// Registers/refreshes a window's metadata; on `Closed`/`Resolved`
    /// terminalizes any still-working orders for that window (trading stopped),
    /// and on `Resolved` settles the window once — paying $1 per winning share
    /// (§9). The settlement mutates the wallet (reflected in `balances()`) and
    /// emits no order/fill effect.
    pub fn on_window(
        &mut self,
        market: &Arc<MarketInfo>,
        lifecycle: WindowLifecycle,
        now: TimestampMs,
    ) -> Vec<PaperEffect> {
        self.markets.insert(market.window, Arc::clone(market));
        self.fee_rate_by_token
            .insert(market.tokens.up.clone(), market.fees.rate);
        self.fee_rate_by_token
            .insert(market.tokens.down.clone(), market.fees.rate);

        if !matches!(
            lifecycle,
            WindowLifecycle::Closed | WindowLifecycle::Resolved { .. }
        ) {
            return Vec::new();
        }
        let ids: Vec<OrderId> = self
            .orders
            .values()
            .filter(|o| !o.state.is_terminal() && o.order.window == market.window)
            .map(|o| o.order_id.clone())
            .collect();
        let mut effects = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(o) = self.orders.get_mut(&id) {
                o.state = OrderState::Canceled;
                o.cancel_effect_at = None;
                o.expires_at = None;
                effects.push(PaperEffect::Order(order_update(o, now, None)));
            }
        }
        // Settlement once the outcome is known. `Closed` never settles (the
        // outcome is not yet known); `insert` returning false ⇒ already settled.
        if let WindowLifecycle::Resolved { outcome } = lifecycle
            && self.settled.insert(market.window)
        {
            let payout = self
                .wallet
                .settle_window(&market.tokens.up, &market.tokens.down, outcome);
            tracing::info!(
                target: "paper-ledger",
                window = %market.event_slug,
                ?outcome,
                payout = %payout,
                "settled window"
            );
        }
        effects
    }

    // ---- command side ------------------------------------------------------

    /// Admits a new order. Returns `Ok(Accepted{PendingNew})` for any
    /// well-formed order — its real fate (Open / Rejected / marketable fills)
    /// is emitted asynchronously once the placement latency elapses, faithfully
    /// modeling the wire round-trip. Only a structurally-invalid order (a qty
    /// kind that doesn't match its TIF/side — which the normalizer forbids
    /// upstream) is rejected synchronously, mirroring live's build-error path.
    pub fn place(
        &mut self,
        order: &NewOrder,
        now: TimestampMs,
    ) -> (Result<Accepted, VenueError>, Vec<PaperEffect>) {
        if let Err(e) = check_qty_kind(order) {
            return (Err(e), Vec::new());
        }
        self.next_seq += 1;
        let seq = self.next_seq;
        let order_id = match OrderId::new(format!("paper-{seq}")) {
            Ok(id) => id,
            // `paper-{seq}` is never empty; map the impossible error defensively.
            Err(e) => return (Err(VenueError::VenueInternal(e.to_string())), Vec::new()),
        };
        let original_size = match order.qty {
            OrderQty::Shares(s) => s,
            // Marketable BUY (notional): no share original; set at activation.
            OrderQty::Notional(_) => Size::ZERO,
        };
        let activate_at = now.saturating_add(self.sampler.sample(&self.params.placement));
        let expires_at = match order.tif {
            TimeInForce::Gtd { expires_at, .. } => Some(floor_gtd(expires_at, now)),
            _ => None,
        };
        self.orders.insert(
            order_id.clone(),
            SimOrder {
                order_id: order_id.clone(),
                order: order.clone(),
                state: OrderState::PendingNew,
                original_size,
                filled: Size::ZERO,
                queue_ahead: Size::ZERO,
                activate_at,
                cancel_effect_at: None,
                expires_at,
                seq,
            },
        );
        (
            Ok(Accepted {
                client_id: order.client_id.clone(),
                order_id,
                state: OrderState::PendingNew,
            }),
            Vec::new(),
        )
    }

    /// Schedules a cancel to take effect after the cancel round-trip. The order
    /// stays working and fillable until then — prints in the race window still
    /// fill it (we lose the race), exactly the §9 adverse case. Returns the ack
    /// shape live returns (`canceled:[id]`); the terminal `Canceled` event
    /// arrives later from [`MatchEngine::on_clock`].
    pub fn cancel(&mut self, id: &OrderId, now: TimestampMs) -> (CancelReport, Vec<PaperEffect>) {
        let cancelable = self.orders.get(id).is_some_and(|o| !o.state.is_terminal());
        if !cancelable {
            return (already_gone(id), Vec::new());
        }
        let effect_at = now.saturating_add(self.sampler.sample(&self.params.cancel));
        if let Some(o) = self.orders.get_mut(id) {
            o.cancel_effect_at = Some(earliest(o.cancel_effect_at, effect_at));
        }
        (
            CancelReport {
                canceled: vec![id.clone()],
                not_canceled: Vec::new(),
            },
            Vec::new(),
        )
    }

    /// Schedules a cancel round-trip for every working order in `condition_id`'s
    /// market (large-move repricing, §8).
    pub fn cancel_market(
        &mut self,
        condition_id: &ConditionId,
        now: TimestampMs,
    ) -> (CancelReport, Vec<PaperEffect>) {
        let ids: Vec<OrderId> = self
            .orders
            .values()
            .filter(|o| !o.state.is_terminal())
            .filter(|o| {
                self.markets
                    .get(&o.order.window)
                    .is_some_and(|m| m.condition_id == *condition_id)
            })
            .map(|o| o.order_id.clone())
            .collect();
        (self.schedule_cancels(&ids, now), Vec::new())
    }

    /// Schedules a cancel round-trip for every working order (every §11 safety
    /// path wires this).
    pub fn cancel_all(&mut self, now: TimestampMs) -> (CancelReport, Vec<PaperEffect>) {
        let ids: Vec<OrderId> = self
            .orders
            .values()
            .filter(|o| !o.state.is_terminal())
            .map(|o| o.order_id.clone())
            .collect();
        (self.schedule_cancels(&ids, now), Vec::new())
    }

    fn schedule_cancels(&mut self, ids: &[OrderId], now: TimestampMs) -> CancelReport {
        let mut report = CancelReport::default();
        for id in ids {
            let effect_at = now.saturating_add(self.sampler.sample(&self.params.cancel));
            if let Some(o) = self.orders.get_mut(id) {
                o.cancel_effect_at = Some(earliest(o.cancel_effect_at, effect_at));
                report.canceled.push(id.clone());
            }
        }
        report
    }

    // ---- trade-driven fills ------------------------------------------------

    /// Drives resting fills from a real print. A resting order fills only from a
    /// print whose aggressor is the OPPOSITE side, at-or-through its price; the
    /// print's size is a finite budget shared across our eligible orders in
    /// realistic priority (best price first, then time), each draining its own
    /// queue from that budget before filling. Maker fills, fee zero, at our
    /// limit price.
    pub fn on_trade(
        &mut self,
        token: &TokenId,
        price: Price,
        size: Size,
        aggressor: Side,
        ts: TimestampMs,
        now: TimestampMs,
    ) -> Vec<PaperEffect> {
        let our_side = aggressor.opposite();
        // Eligible resting orders: same token, working, on our side, at/through.
        let mut candidates: Vec<(OrderId, Decimal, TimestampMs, u64)> = self
            .orders
            .values()
            .filter(|o| o.order.token_id == *token)
            .filter(|o| matches!(o.state, OrderState::Open | OrderState::PartiallyFilled))
            .filter(|o| o.order.side == our_side && at_or_through(our_side, o.order.price, price))
            .map(|o| {
                (
                    o.order_id.clone(),
                    o.order.price.as_decimal(),
                    o.activate_at,
                    o.seq,
                )
            })
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        // Priority: the most aggressive (closest to touch) price first, then
        // time. A sell aggressor hits the highest bids first; a buy aggressor
        // the lowest asks first.
        candidates.sort_by(|a, b| {
            let by_price = match our_side {
                Side::Buy => b.1.cmp(&a.1),  // higher bid first
                Side::Sell => a.1.cmp(&b.1), // lower ask first
            };
            by_price.then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3))
        });

        let fee_rate = self.fee_rate_for(token);
        let rebate_share = self.params.maker_rebate_share;
        let mut budget = size;
        let mut effects = Vec::new();
        for (id, _, _, _) in candidates {
            if budget.is_zero() {
                break;
            }
            let trade_id = self.mint_trade_id();
            let made = {
                let Some(o) = self.orders.get_mut(&id) else {
                    continue;
                };
                // Drain real queue ahead of us from the shared print budget.
                let drain = budget.min(o.queue_ahead);
                o.queue_ahead = o.queue_ahead.saturating_sub(drain);
                budget = budget.saturating_sub(drain);
                if budget.is_zero() {
                    None
                } else {
                    let fill = budget.min(o.remaining());
                    if fill.is_zero() {
                        None
                    } else {
                        o.filled = o.filled + fill;
                        budget = budget.saturating_sub(fill);
                        o.state = fill_state(o.filled, o.original_size);
                        let f = Fill {
                            order_id: o.order_id.clone(),
                            trade_id: Some(trade_id),
                            window: o.order.window,
                            token_id: o.order.token_id.clone(),
                            outcome: o.order.outcome,
                            side: o.order.side,
                            price: o.order.price,
                            size: fill,
                            liquidity: Liquidity::Maker,
                            fee: Dollars::ZERO,
                            ts_venue: ts,
                            ts_local: now,
                        };
                        let update = order_update(o, now, None);
                        Some((f, update))
                    }
                }
            };
            if let Some((f, update)) = made {
                // Maker rebate estimate: our share of the taker fee on our
                // maker volume (credited later on the daily cycle).
                let rebate =
                    Dollars::new(rebate_share * taker_fee(f.size, fee_rate, f.price).as_decimal());
                self.wallet.apply_fill(&f, rebate);
                // Arm the daily rebate-credit cycle on the first accrual (`new`
                // has no clock); rolled forward by `on_clock` thereafter.
                if self.rebate_next_credit_at.is_none() {
                    self.rebate_next_credit_at = Some(now.saturating_add(REBATE_PERIOD));
                }
                effects.push(PaperEffect::Fill(f));
                effects.push(PaperEffect::Order(update));
            }
            if budget.is_zero() {
                break;
            }
        }
        effects
    }

    // ---- timers ------------------------------------------------------------

    /// Processes every deadline due at or before `now`, in chronological order
    /// (re-scanning after each, since processing one can change another's
    /// applicability — e.g. an activation that the next cancel then terminates).
    pub fn on_clock(&mut self, now: TimestampMs) -> Vec<PaperEffect> {
        let mut effects = Vec::new();
        while let Some((id, kind)) = self.next_due_deadline(now) {
            match kind {
                DeadlineKind::Activation => effects.extend(self.activate(&id, now)),
                DeadlineKind::Cancel => effects.extend(self.apply_cancel_effect(&id, now)),
                DeadlineKind::Expiry => effects.extend(self.apply_expiry(&id, now)),
            }
        }
        self.run_rebate_cycle(now);
        effects
    }

    /// Credits the daily maker rebate for every cycle period elapsed at or
    /// before `now` ($1 minimum accrual, §9). The `while` catches up all periods
    /// spanned by a long sleep; sub-$1 periods carry over. Mutates the wallet
    /// only (no order/fill effect).
    fn run_rebate_cycle(&mut self, now: TimestampMs) {
        while let Some(due) = self.rebate_next_credit_at {
            if due.as_millis() > now.as_millis() {
                break;
            }
            let credited = self.wallet.credit_rebate(Dollars::new(Decimal::ONE));
            if !credited.is_zero() {
                tracing::info!(
                    target: "paper-ledger",
                    credited = %credited,
                    "credited daily maker rebate"
                );
            }
            self.rebate_next_credit_at = Some(due.saturating_add(REBATE_PERIOD));
        }
    }

    /// The earliest deadline the driver should sleep until (across all
    /// non-terminal orders plus the daily rebate cycle), or `None` if nothing is
    /// pending.
    #[must_use]
    pub fn next_deadline(&self) -> Option<TimestampMs> {
        let order_min = self
            .orders
            .values()
            .filter(|o| !o.state.is_terminal())
            .flat_map(|o| applicable_deadlines(o).into_iter().map(|(t, _)| t))
            .min();
        [order_min, self.rebate_next_credit_at]
            .into_iter()
            .flatten()
            .min()
    }

    /// The single earliest deadline due at or before `now`, with deterministic
    /// tie-breaking.
    fn next_due_deadline(&self, now: TimestampMs) -> Option<(OrderId, DeadlineKind)> {
        let mut best: Option<(TimestampMs, u8, u64, OrderId, DeadlineKind)> = None;
        for o in self.orders.values() {
            if o.state.is_terminal() {
                continue;
            }
            for (t, kind) in applicable_deadlines(o) {
                if t.as_millis() > now.as_millis() {
                    continue;
                }
                let cand = (t, kind.priority(), o.seq, o.order_id.clone(), kind);
                best = Some(match best {
                    Some(b) if (b.0, b.1, b.2) <= (cand.0, cand.1, cand.2) => b,
                    _ => cand,
                });
            }
        }
        best.map(|(_, _, _, id, kind)| (id, kind))
    }

    fn activate(&mut self, id: &OrderId, now: TimestampMs) -> Vec<PaperEffect> {
        let Some(o) = self.orders.get(id) else {
            return Vec::new();
        };
        if o.state != OrderState::PendingNew {
            return Vec::new();
        }
        if o.order.tif.is_marketable() {
            self.activate_marketable(id, now)
        } else {
            self.activate_passive(id, now)
        }
    }

    fn activate_passive(&mut self, id: &OrderId, now: TimestampMs) -> Vec<PaperEffect> {
        let Some(o) = self.orders.get(id) else {
            return Vec::new();
        };
        let side = o.order.side;
        let price = o.order.price;
        let token = o.order.token_id.clone();
        let post_only = o.order.tif.is_post_only();

        let book = self.books.get(&token);
        let crosses = post_only && would_cross(side, price, book);
        let queue = book.map_or(Size::ZERO, |b| b.displayed_at(side, price));

        let Some(o) = self.orders.get_mut(id) else {
            return Vec::new();
        };
        if crosses {
            o.state = OrderState::Rejected;
            return vec![PaperEffect::Order(order_update(
                o,
                now,
                Some("post-only order would cross the book".to_owned()),
            ))];
        }
        o.queue_ahead = queue;
        o.state = OrderState::Open;
        vec![PaperEffect::Order(order_update(o, now, None))]
    }

    fn activate_marketable(&mut self, id: &OrderId, now: TimestampMs) -> Vec<PaperEffect> {
        let Some(o) = self.orders.get(id) else {
            return Vec::new();
        };
        let side = o.order.side;
        let price = o.order.price;
        let qty = o.order.qty;
        let token = o.order.token_id.clone();
        let window = o.order.window;
        let outcome = o.order.outcome;
        let fok = matches!(o.order.tif, TimeInForce::Fok);

        let walk = self
            .books
            .get(&token)
            .map(|b| b.walk_marketable(side, price, qty))
            .unwrap_or(crate::book::MarketableWalk {
                fills: Vec::new(),
                fully_filled: false,
            });

        // FOK that can't fill in full: reject entirely, no fills, no depletion.
        if fok && !walk.fully_filled {
            if let Some(o) = self.orders.get_mut(id) {
                o.state = OrderState::Rejected;
                return vec![PaperEffect::Order(order_update(
                    o,
                    now,
                    Some("FOK order could not be filled in full".to_owned()),
                ))];
            }
            return Vec::new();
        }
        // Nothing to match at all: reject (FAK no match).
        if walk.fills.is_empty() {
            if let Some(o) = self.orders.get_mut(id) {
                o.state = OrderState::Rejected;
                return vec![PaperEffect::Order(order_update(
                    o,
                    now,
                    Some("FAK order found nothing to match".to_owned()),
                ))];
            }
            return Vec::new();
        }

        let fee_rate = self.fee_rate_for(&token);
        let mut effects = Vec::new();
        let mut total = Size::ZERO;
        for (lvl_price, lvl_size) in &walk.fills {
            // Deplete the local book so a simultaneous paper taker can't
            // re-take the same shares (the next snapshot overwrites this). A
            // marketable order consumes the OPPOSITE book side (a BUY eats asks,
            // a SELL eats bids), so deplete `side.opposite()`.
            if let Some(b) = self.books.get_mut(&token) {
                b.consume(side.opposite(), *lvl_price, *lvl_size);
            }
            let fee = taker_fee(*lvl_size, fee_rate, *lvl_price);
            let f = Fill {
                order_id: id.clone(),
                trade_id: Some(self.mint_trade_id()),
                window,
                token_id: token.clone(),
                outcome,
                side,
                price: *lvl_price,
                size: *lvl_size,
                liquidity: Liquidity::Taker,
                fee,
                ts_venue: now,
                ts_local: now,
            };
            self.wallet.apply_fill(&f, Dollars::ZERO);
            total = total + *lvl_size;
            effects.push(PaperEffect::Fill(f));
        }

        if let Some(o) = self.orders.get_mut(id) {
            o.filled = total;
            if o.original_size.is_zero() {
                // Marketable BUY (notional): report executed shares as original.
                o.original_size = total;
            }
            o.state = if walk.fully_filled {
                OrderState::Filled
            } else {
                OrderState::Canceled
            };
            effects.push(PaperEffect::Order(order_update(o, now, None)));
        }
        effects
    }

    fn apply_cancel_effect(&mut self, id: &OrderId, now: TimestampMs) -> Vec<PaperEffect> {
        let Some(o) = self.orders.get_mut(id) else {
            return Vec::new();
        };
        o.cancel_effect_at = None;
        if o.state.is_terminal() {
            return Vec::new();
        }
        o.state = OrderState::Canceled;
        vec![PaperEffect::Order(order_update(o, now, None))]
    }

    fn apply_expiry(&mut self, id: &OrderId, now: TimestampMs) -> Vec<PaperEffect> {
        let Some(o) = self.orders.get_mut(id) else {
            return Vec::new();
        };
        o.expires_at = None;
        if o.state.is_terminal() {
            return Vec::new();
        }
        // PendingNew → Expired is illegal in the table; a never-activated GTD
        // becomes Canceled instead (cannot happen under the 60s floor, but the
        // guard keeps the transition table honored).
        o.state = if o.state == OrderState::PendingNew {
            OrderState::Canceled
        } else {
            OrderState::Expired
        };
        vec![PaperEffect::Order(order_update(o, now, None))]
    }

    // ---- balances ----------------------------------------------------------

    /// Current wallet snapshot.
    #[must_use]
    pub fn balances(&self) -> Wallet {
        self.wallet.to_wallet()
    }

    /// Read access to the wallet (for the smoke subcommand's PnL line).
    #[must_use]
    pub fn wallet(&self) -> &PaperWallet {
        &self.wallet
    }

    /// The richer ledger snapshot (income lines + signed positions) for the
    /// dashboard / `bot paper-sim`.
    #[must_use]
    pub fn ledger_snapshot(&self) -> PaperLedgerSnapshot {
        self.wallet.snapshot()
    }

    /// Sets paper capital to `amount` at runtime (dashboard seam, §9). Logged as
    /// the audit trail; the binary additionally journals a
    /// `ControlEvent::PaperCapitalSet` on the bus.
    pub fn set_capital(&mut self, amount: Dollars) {
        self.wallet.set_capital(amount);
        tracing::info!(
            target: "paper-ledger",
            op = "set_capital",
            new_collateral = %amount,
            "paper capital set"
        );
    }

    /// Adjusts paper capital by a signed `delta` at runtime (dashboard seam, §9).
    pub fn adjust_capital(&mut self, delta: Dollars) {
        self.wallet.adjust_capital(delta);
        tracing::info!(
            target: "paper-ledger",
            op = "adjust_capital",
            delta = %delta,
            new_collateral = %self.wallet.collateral(),
            "paper capital adjusted"
        );
    }

    /// Merges up to `requested` matched Up/Down pairs for `window` into
    /// collateral (instant simulated CTF merge, §8): each pair removes one Up and
    /// one Down share and credits $1. Returns the number of pairs merged, or an
    /// error if the window's metadata is not yet known. Produces no order/fill
    /// effect — the change is reflected in `balances()`. (`now` is unused but
    /// kept for the driver's `apply` closure symmetry.)
    pub fn merge(
        &mut self,
        window: WindowId,
        requested: Size,
        _now: TimestampMs,
    ) -> (Result<Size, VenueError>, Vec<PaperEffect>) {
        let Some(market) = self.markets.get(&window) else {
            return (
                Err(VenueError::VenueInternal(format!(
                    "merge: unknown window {window:?}"
                ))),
                Vec::new(),
            );
        };
        let up = market.tokens.up.clone();
        let down = market.tokens.down.clone();
        let slug = market.event_slug.clone();
        let merged = self.wallet.merge(&up, &down, requested);
        if !merged.is_zero() {
            tracing::info!(
                target: "paper-ledger",
                window = %slug,
                pairs = %merged,
                "merged matched pairs to collateral"
            );
        }
        (Ok(merged), Vec::new())
    }

    // ---- helpers -----------------------------------------------------------

    fn fee_rate_for(&self, token: &TokenId) -> Decimal {
        self.fee_rate_by_token
            .get(token)
            .copied()
            .unwrap_or(self.params.default_fee_rate)
    }

    fn mint_trade_id(&mut self) -> String {
        self.next_trade += 1;
        format!("paper-t{}", self.next_trade)
    }
}

/// Applicable deadlines for a non-terminal order (caller filters terminality).
fn applicable_deadlines(o: &SimOrder) -> Vec<(TimestampMs, DeadlineKind)> {
    let mut v = Vec::with_capacity(3);
    if o.state == OrderState::PendingNew {
        v.push((o.activate_at, DeadlineKind::Activation));
    }
    if let Some(t) = o.cancel_effect_at {
        v.push((t, DeadlineKind::Cancel));
    }
    if let Some(t) = o.expires_at {
        v.push((t, DeadlineKind::Expiry));
    }
    v
}

/// True if a post-only resting order at `price` would cross `book`'s opposite
/// top (BUY ≥ best ask, SELL ≤ best bid). A missing book/side never crosses.
fn would_cross(side: Side, price: Price, book: Option<&SimBook>) -> bool {
    let Some(book) = book else {
        return false;
    };
    match side {
        Side::Buy => book
            .best_ask()
            .is_some_and(|a| price.as_decimal() >= a.price.as_decimal()),
        Side::Sell => book
            .best_bid()
            .is_some_and(|b| price.as_decimal() <= b.price.as_decimal()),
    }
}

/// Whether a print at `print_price` is at-or-through a resting order's `price`.
fn at_or_through(our_side: Side, our_price: Price, print_price: Price) -> bool {
    match our_side {
        Side::Buy => print_price.as_decimal() <= our_price.as_decimal(),
        Side::Sell => print_price.as_decimal() >= our_price.as_decimal(),
    }
}

/// The lifecycle state implied purely by fill progress.
fn fill_state(filled: Size, original: Size) -> OrderState {
    if !original.is_zero() && filled.as_decimal() >= original.as_decimal() {
        OrderState::Filled
    } else if filled.is_zero() {
        OrderState::Open
    } else {
        OrderState::PartiallyFilled
    }
}

/// Floors a GTD expiration at the venue's 60-second minimum lead (§7).
fn floor_gtd(expires_at: TimestampMs, now: TimestampMs) -> TimestampMs {
    let min = now.saturating_add(GTD_MIN_LEAD);
    if expires_at.as_millis() < min.as_millis() {
        min
    } else {
        expires_at
    }
}

/// Earliest of an existing deadline and a new one.
fn earliest(existing: Option<TimestampMs>, new: TimestampMs) -> TimestampMs {
    match existing {
        Some(t) if t.as_millis() <= new.as_millis() => t,
        _ => new,
    }
}

/// Validates the qty kind against the TIF/side (mirrors live `convert::require_*`).
fn check_qty_kind(order: &NewOrder) -> Result<(), VenueError> {
    let ok = match order.tif {
        TimeInForce::Gtc { .. } | TimeInForce::Gtd { .. } => {
            matches!(order.qty, OrderQty::Shares(_))
        }
        TimeInForce::Fok | TimeInForce::Fak => match order.side {
            Side::Buy => matches!(order.qty, OrderQty::Notional(_)),
            Side::Sell => matches!(order.qty, OrderQty::Shares(_)),
        },
    };
    if ok {
        Ok(())
    } else {
        Err(VenueError::Rejected(RejectReason::Other(
            "order quantity kind does not match its time-in-force/side".to_owned(),
        )))
    }
}

/// A cancel report rejecting an unknown/terminal order as already gone.
fn already_gone(id: &OrderId) -> CancelReport {
    CancelReport {
        canceled: Vec::new(),
        not_canceled: vec![NotCanceled {
            order_id: id.clone(),
            reason: RejectReason::AlreadyGone,
            raw: "no matching working order".to_owned(),
        }],
    }
}

/// Builds an [`OrderUpdate`] snapshot from a sim order.
fn order_update(o: &SimOrder, now: TimestampMs, reject: Option<String>) -> OrderUpdate {
    OrderUpdate {
        order_id: o.order_id.clone(),
        window: o.order.window,
        token_id: o.order.token_id.clone(),
        side: o.order.side,
        state: o.state,
        price: o.order.price,
        original_size: o.original_size,
        filled_size: o.filled,
        reject_reason: reject,
        ts_venue: Some(now),
        ts_local: now,
    }
}

#[cfg(test)]
impl MatchEngine {
    /// Test-only: current lifecycle state of an order.
    fn dbg_state(&self, id: &OrderId) -> Option<OrderState> {
        self.orders.get(id).map(|o| o.state)
    }
    /// Test-only: cumulative filled size of an order.
    fn dbg_filled(&self, id: &OrderId) -> Option<Size> {
        self.orders.get(id).map(|o| o.filled)
    }
    /// Test-only: remaining queue ahead of an order.
    fn dbg_queue(&self, id: &OrderId) -> Option<Size> {
        self.orders.get(id).map(|o| o.queue_ahead)
    }
}

#[cfg(test)]
mod tests {
    use core_types::{
        Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, FeeParams,
        Liquidity, MarketInfo, NewOrder, OrderQty, OrderState, Outcome, Price, ResolutionSource,
        RoundDir, Series, Side, Size, TickSize, TimeInForce, TimestampMs, TokenId, TokenPair,
        WindowDuration, WindowId, WindowLifecycle, taker_fee,
    };
    use rust_decimal::dec;
    use venue_api::RejectReason;

    use super::*;
    use crate::params::{LatencySpec, PaperParams};

    const T0: i64 = 1_000_000;
    const PLACE_MS: i64 = 100;
    const CANCEL_MS: i64 = 200;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }
    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn up_token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn down_token() -> TokenId {
        TokenId::new("222").unwrap()
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: ts(T0),
        }
    }
    fn condition() -> ConditionId {
        ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap()
    }

    fn engine() -> MatchEngine {
        MatchEngine::new(PaperParams {
            placement: LatencySpec {
                mean_ms: DurationMs::from_millis(PLACE_MS),
                jitter_ms: DurationMs::ZERO,
            },
            cancel: LatencySpec {
                mean_ms: DurationMs::from_millis(CANCEL_MS),
                jitter_ms: DurationMs::ZERO,
            },
            rng_seed: Some(1),
            ..PaperParams::default()
        })
    }

    fn market_with_fee(rate: Decimal) -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: condition(),
            tokens: TokenPair {
                up: up_token(),
                down: down_token(),
            },
            close_time: ts(T0 + 300_000),
            strike: Some(dec!(60000)),
            tick_size: TickSize::T001,
            min_order_size: sz(dec!(5)),
            fees: FeeParams {
                rate,
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        })
    }
    fn market() -> Arc<MarketInfo> {
        market_with_fee(dec!(0.07))
    }

    fn snapshot_for(
        token: TokenId,
        bids: &[(Decimal, Decimal)],
        asks: &[(Decimal, Decimal)],
        at: i64,
    ) -> BookSnapshot {
        BookSnapshot {
            token_id: token,
            condition_id: condition(),
            bids: bids
                .iter()
                .map(|(p, s)| BookLevel {
                    price: px(*p),
                    size: sz(*s),
                })
                .collect(),
            asks: asks
                .iter()
                .map(|(p, s)| BookLevel {
                    price: px(*p),
                    size: sz(*s),
                })
                .collect(),
            ts: ts(at),
            seq_hash: None,
        }
    }
    fn snapshot(bids: &[(Decimal, Decimal)], asks: &[(Decimal, Decimal)], at: i64) -> BookSnapshot {
        snapshot_for(up_token(), bids, asks, at)
    }
    fn down_snapshot(
        bids: &[(Decimal, Decimal)],
        asks: &[(Decimal, Decimal)],
        at: i64,
    ) -> BookSnapshot {
        snapshot_for(down_token(), bids, asks, at)
    }

    fn resting(side: Side, price: Decimal, shares: Decimal, post_only: bool) -> NewOrder {
        NewOrder {
            client_id: Some("c".to_owned()),
            window: window(),
            token_id: up_token(),
            outcome: Outcome::Up,
            side,
            price: px(price),
            qty: OrderQty::Shares(sz(shares)),
            tif: TimeInForce::Gtc { post_only },
        }
    }
    fn buy(price: Decimal, shares: Decimal) -> NewOrder {
        resting(Side::Buy, price, shares, true)
    }
    fn marketable_buy(worst: Decimal, dollars: Decimal, fok: bool) -> NewOrder {
        NewOrder {
            client_id: Some("m".to_owned()),
            window: window(),
            token_id: up_token(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: px(worst),
            qty: OrderQty::Notional(Dollars::new(dollars)),
            tif: if fok {
                TimeInForce::Fok
            } else {
                TimeInForce::Fak
            },
        }
    }
    fn marketable_sell(worst: Decimal, shares: Decimal, fok: bool) -> NewOrder {
        NewOrder {
            client_id: Some("m".to_owned()),
            window: window(),
            token_id: up_token(),
            outcome: Outcome::Up,
            side: Side::Sell,
            price: px(worst),
            qty: OrderQty::Shares(sz(shares)),
            tif: if fok {
                TimeInForce::Fok
            } else {
                TimeInForce::Fak
            },
        }
    }

    fn orders_of(fx: &[PaperEffect]) -> Vec<&OrderUpdate> {
        fx.iter()
            .filter_map(|e| match e {
                PaperEffect::Order(u) => Some(u),
                PaperEffect::Fill(_) => None,
            })
            .collect()
    }
    fn fills_of(fx: &[PaperEffect]) -> Vec<&Fill> {
        fx.iter()
            .filter_map(|e| match e {
                PaperEffect::Fill(f) => Some(f),
                PaperEffect::Order(_) => None,
            })
            .collect()
    }
    fn only_order(fx: &[PaperEffect]) -> &OrderUpdate {
        let v = orders_of(fx);
        assert_eq!(v.len(), 1, "expected exactly one order update, got {fx:?}");
        v[0]
    }
    fn only_fill(fx: &[PaperEffect]) -> &Fill {
        let v = fills_of(fx);
        assert_eq!(v.len(), 1, "expected exactly one fill, got {fx:?}");
        v[0]
    }

    /// Place + activate a passive order against a current book; returns its id.
    fn open_resting(e: &mut MatchEngine, order: &NewOrder) -> OrderId {
        let id = e.place(order, ts(T0)).0.expect("accepted").order_id;
        let fx = e.on_clock(ts(T0 + PLACE_MS));
        assert_eq!(only_order(&fx).state, OrderState::Open);
        id
    }

    // ---- Rule 1: placement latency ----------------------------------------

    #[test]
    fn place_returns_pending_new_and_activates_only_after_latency() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));

        let (acc, fx) = e.place(&buy(dec!(0.49), dec!(20)), ts(T0));
        let acc = acc.expect("accepted");
        assert_eq!(acc.state, OrderState::PendingNew);
        assert_eq!(acc.client_id.as_deref(), Some("c"));
        assert!(fx.is_empty(), "place publishes nothing");

        // Before the placement latency elapses: still nothing.
        assert!(e.on_clock(ts(T0 + PLACE_MS - 1)).is_empty());
        // At the deadline: it activates.
        let fx = e.on_clock(ts(T0 + PLACE_MS));
        let u = only_order(&fx);
        assert_eq!(u.state, OrderState::Open);
        assert_eq!(u.order_id, acc.order_id);
    }

    // ---- Rule 2 / 6: activation post-only cross + would NOT cross ----------

    #[test]
    fn post_only_that_would_cross_at_activation_is_rejected() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        // Best ask 0.50; a post-only BUY at 0.50 would cross.
        e.on_book(&snapshot(
            &[(dec!(0.48), dec!(10))],
            &[(dec!(0.50), dec!(10))],
            T0,
        ));
        let id = e
            .place(&buy(dec!(0.50), dec!(20)), ts(T0))
            .0
            .unwrap()
            .order_id;
        let fx = e.on_clock(ts(T0 + PLACE_MS));
        let u = only_order(&fx);
        assert_eq!(u.state, OrderState::Rejected);
        assert!(u.reject_reason.as_deref().unwrap().contains("cross"));
        assert_eq!(e.dbg_state(&id), Some(OrderState::Rejected));
    }

    #[test]
    fn post_only_just_below_the_touch_rests() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.48), dec!(10))],
            &[(dec!(0.50), dec!(10))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        assert_eq!(e.dbg_state(&id), Some(OrderState::Open));
    }

    // ---- Rule 3: queued behind ALL displayed size at activation -----------

    #[test]
    fn activation_queues_behind_all_displayed_size() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(123))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        assert_eq!(e.dbg_queue(&id), Some(sz(dec!(123))));
    }

    // ---- Rule 4: resting fill from an opposite-aggressor print ------------

    #[test]
    fn resting_buy_fills_from_a_sell_print_at_our_price_as_maker() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));

        // Sell-aggressor print of 40 at 0.49: drains the 30 queue, fills 10.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(40)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        let f = only_fill(&fx);
        assert_eq!(f.liquidity, Liquidity::Maker);
        assert_eq!(f.fee, Dollars::ZERO);
        assert_eq!(f.size, sz(dec!(10)));
        assert_eq!(f.price, px(dec!(0.49))); // our limit, not the print price
        assert_eq!(only_order(&fx).state, OrderState::PartiallyFilled);
        assert_eq!(e.dbg_filled(&id), Some(sz(dec!(10))));
    }

    #[test]
    fn print_through_our_price_fills_us_at_our_limit() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        // A print THROUGH our price (0.47 < 0.49) with no queue → fills at 0.49.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.47)),
            sz(dec!(5)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        let f = only_fill(&fx);
        assert_eq!(f.price, px(dec!(0.49)));
        assert_eq!(f.size, sz(dec!(5)));
        assert_eq!(e.dbg_filled(&id), Some(sz(dec!(5))));
    }

    // ---- Rule 5: aggressor-side gating ------------------------------------

    #[test]
    fn same_side_print_never_fills_our_resting_order() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let _ = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        // A BUY-aggressor print (lifts asks) must not touch our resting bid.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(50)),
            Side::Buy,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        assert!(fills_of(&fx).is_empty());
    }

    #[test]
    fn print_not_through_our_price_does_not_fill() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let _ = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        // Sell print ABOVE our bid (0.50 > 0.49): not at/through → no fill.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.50)),
            sz(dec!(50)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        assert!(fills_of(&fx).is_empty());
    }

    // ---- Rule 6: queue must drain before any fill -------------------------

    #[test]
    fn small_print_only_drains_queue_then_next_print_fills() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));

        // Print of 20 < queue 30 → only drains, no fill.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(20)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        assert!(fills_of(&fx).is_empty());
        assert_eq!(e.dbg_queue(&id), Some(sz(dec!(10))));

        // Next print of 25: drains last 10, fills 15.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(25)),
            Side::Sell,
            ts(T0 + 300),
            ts(T0 + 300),
        );
        let f = only_fill(&fx);
        assert_eq!(f.size, sz(dec!(15)));
        assert_eq!(e.dbg_queue(&id), Some(Size::ZERO));
    }

    // ---- Rule 7: shared print budget across co-located orders -------------

    #[test]
    fn one_print_is_a_shared_budget_no_double_count() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(30))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        // Two orders at the same price; each captures queue_ahead = 30.
        let id1 = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        let id2 = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        assert_eq!(e.dbg_queue(&id1), Some(sz(dec!(30))));
        assert_eq!(e.dbg_queue(&id2), Some(sz(dec!(30))));

        // Print of 80: order1 drains 30 (budget 50) then fills 20 (budget 30);
        // order2 drains its 30 from the remaining budget → 0 left, no fill.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(80)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        let fills = fills_of(&fx);
        assert_eq!(fills.len(), 1, "only order1 fills");
        assert_eq!(fills[0].order_id, id1);
        assert_eq!(fills[0].size, sz(dec!(20)));
        assert_eq!(e.dbg_state(&id1), Some(OrderState::Filled));
        assert_eq!(e.dbg_filled(&id2), Some(Size::ZERO));
        assert_eq!(e.dbg_queue(&id2), Some(Size::ZERO)); // its queue WAS drained

        // A later small print now fills order2 (its queue is gone).
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(5)),
            Side::Sell,
            ts(T0 + 300),
            ts(T0 + 300),
        );
        assert_eq!(only_fill(&fx).order_id, id2);
    }

    // ---- Rule 4: aggressor priority (higher bid first) --------------------

    #[test]
    fn higher_priced_bid_has_priority_for_a_sell_print() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0)), (dec!(0.48), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let high = open_resting(&mut e, &buy(dec!(0.49), dec!(10)));
        let low = open_resting(&mut e, &buy(dec!(0.48), dec!(10)));
        // Sell print at 0.47 (through both), size 12: fills the 0.49 bid first.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.47)),
            sz(dec!(12)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        let fills = fills_of(&fx);
        assert_eq!(fills[0].order_id, high);
        assert_eq!(fills[0].size, sz(dec!(10)));
        assert_eq!(fills[1].order_id, low);
        assert_eq!(fills[1].size, sz(dec!(2)));
    }

    // ---- Rule 8 / 9: cancel round-trip and the adverse race --------------

    #[test]
    fn cancel_takes_effect_after_the_round_trip_no_pending_cancel_event() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));

        // Request cancel at T0+200 → ack canceled, NO event, still working.
        let (report, fx) = e.cancel(&id, ts(T0 + 200));
        assert_eq!(report.canceled, vec![id.clone()]);
        assert!(fx.is_empty(), "no PendingCancel event");
        assert_eq!(e.dbg_state(&id), Some(OrderState::Open));

        // Before the cancel lands: still open.
        assert!(e.on_clock(ts(T0 + 200 + CANCEL_MS - 1)).is_empty());
        // It lands → Canceled.
        let fx = e.on_clock(ts(T0 + 200 + CANCEL_MS));
        assert_eq!(only_order(&fx).state, OrderState::Canceled);
    }

    #[test]
    fn a_print_fills_the_stale_quote_before_the_cancel_lands() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));

        // Operator cancels at T0+200 (lands at +400).
        let _ = e.cancel(&id, ts(T0 + 200));
        // But a sell print at T0+250 (before the cancel lands) fills us — we
        // lose the race exactly as reality would.
        let fx = e.on_trade(
            &up_token(),
            px(dec!(0.49)),
            sz(dec!(12)),
            Side::Sell,
            ts(T0 + 250),
            ts(T0 + 250),
        );
        let f = only_fill(&fx);
        assert_eq!(f.size, sz(dec!(12)));
        assert_eq!(f.liquidity, Liquidity::Maker);

        // The cancel then lands and cancels the unfilled remainder.
        let fx = e.on_clock(ts(T0 + 200 + CANCEL_MS));
        let u = only_order(&fx);
        assert_eq!(u.state, OrderState::Canceled);
        assert_eq!(u.filled_size, sz(dec!(12))); // the raced fill is kept
    }

    #[test]
    fn cancel_unknown_or_terminal_is_already_gone() {
        let mut e = engine();
        let ghost = OrderId::new("paper-999").unwrap();
        let (report, fx) = e.cancel(&ghost, ts(T0));
        assert!(fx.is_empty());
        assert_eq!(report.not_canceled.len(), 1);
        assert_eq!(report.not_canceled[0].reason, RejectReason::AlreadyGone);
    }

    // ---- Rule 5 (marketable): FOK / FAK -----------------------------------

    #[test]
    fn fok_fills_fully_across_levels_as_taker() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(80)), (dec!(0.52), dec!(80))],
            T0,
        ));
        // Sell 120 shares? No — BUY by dollars. $50 fully fills within 0.52.
        let order = marketable_buy(dec!(0.52), dec!(50), true);
        let id = e.place(&order, ts(T0)).0.unwrap().order_id;
        let fx = e.on_clock(ts(T0 + PLACE_MS));
        let fills = fills_of(&fx);
        assert!(!fills.is_empty());
        assert!(fills.iter().all(|f| f.liquidity == Liquidity::Taker));
        // First level 0.51, then 0.52.
        assert_eq!(fills[0].price, px(dec!(0.51)));
        assert_eq!(fills[1].price, px(dec!(0.52)));
        assert_eq!(only_order(&fx).state, OrderState::Filled);
        // Taker fee charged on each leg.
        assert!(fills.iter().all(|f| !f.fee.is_zero()));
        assert_eq!(e.dbg_state(&id), Some(OrderState::Filled));
    }

    #[test]
    fn fok_that_cannot_fully_fill_is_rejected_with_no_partial() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        // Only 80 shares at 0.51 within the limit → 80×0.51 = 40.80 < $50.
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(80)), (dec!(0.60), dec!(999))],
            T0,
        ));
        let order = marketable_buy(dec!(0.51), dec!(50), true);
        let fx = e.on_clock_after_place(&order);
        let u = only_order(&fx);
        assert_eq!(u.state, OrderState::Rejected);
        assert!(u.reject_reason.as_deref().unwrap().contains("FOK"));
        assert!(fills_of(&fx).is_empty(), "no partial fills on FOK reject");
    }

    #[test]
    fn fak_fills_what_it_can_then_cancels_the_remainder() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        // SELL 120 shares, floor 0.48; only 100 available at/above the floor.
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(100)), (dec!(0.47), dec!(999))],
            &[(dec!(0.51), dec!(0))],
            T0,
        ));
        let order = marketable_sell(dec!(0.48), dec!(120), false);
        let fx = e.on_clock_after_place(&order);
        let fills = fills_of(&fx);
        assert_eq!(fills.iter().map(|f| f.size).sum::<Size>(), sz(dec!(100)));
        assert!(fills.iter().all(|f| f.liquidity == Liquidity::Taker));
        assert_eq!(only_order(&fx).state, OrderState::Canceled);
    }

    #[test]
    fn fak_with_no_match_is_rejected() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        // No bids at/above the floor → nothing to sell into.
        e.on_book(&snapshot(
            &[(dec!(0.40), dec!(100))],
            &[(dec!(0.51), dec!(0))],
            T0,
        ));
        let order = marketable_sell(dec!(0.48), dec!(50), false);
        let fx = e.on_clock_after_place(&order);
        let u = only_order(&fx);
        assert_eq!(u.state, OrderState::Rejected);
        assert!(u.reject_reason.as_deref().unwrap().contains("FAK"));
    }

    // ---- Rule 5 (marketable): local depletion -----------------------------

    #[test]
    fn simultaneous_takers_cannot_double_eat_displayed_depth() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        // 60 shares displayed at 0.51 only (nothing deeper within limit).
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(60))],
            T0,
        ));
        // Two FAK buys, each wanting ~$40 (≈78 sh) at limit 0.51, placed
        // together so both activate in the same on_clock.
        let id1 = e
            .place(&marketable_buy(dec!(0.51), dec!(40), false), ts(T0))
            .0
            .unwrap()
            .order_id;
        let id2 = e
            .place(&marketable_buy(dec!(0.51), dec!(40), false), ts(T0))
            .0
            .unwrap()
            .order_id;
        let fx = e.on_clock(ts(T0 + PLACE_MS));
        let total: Size = fills_of(&fx).iter().map(|f| f.size).sum();
        // Combined fills cannot exceed the 60 displayed shares.
        assert!(total.as_decimal() <= dec!(60), "double-ate depth: {total}");
        // First taker eats the whole 60 (partial vs its $40 want → Canceled);
        // the second finds nothing left → rejected. Never double-eaten.
        assert_eq!(e.dbg_state(&id1), Some(OrderState::Canceled));
        assert_eq!(e.dbg_state(&id2), Some(OrderState::Rejected));
    }

    // ---- Rule 7 (window close) -------------------------------------------

    #[test]
    fn window_close_terminalizes_working_orders() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let id = open_resting(&mut e, &buy(dec!(0.49), dec!(20)));
        let fx = e.on_window(&market(), WindowLifecycle::Closed, ts(T0 + 300_000));
        assert_eq!(only_order(&fx).state, OrderState::Canceled);
        assert_eq!(e.dbg_state(&id), Some(OrderState::Canceled));
    }

    // ---- GTD expiry -------------------------------------------------------

    #[test]
    fn gtd_order_expires_at_its_deadline() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        // Desired expiry 90s out (above the 60s floor).
        let order = NewOrder {
            tif: TimeInForce::Gtd {
                expires_at: ts(T0 + 90_000),
                post_only: true,
            },
            ..buy(dec!(0.49), dec!(20))
        };
        let id = open_resting(&mut e, &order);
        // Nothing at +89s; expires at +90s.
        assert!(e.on_clock(ts(T0 + 89_000)).is_empty());
        let fx = e.on_clock(ts(T0 + 90_000));
        assert_eq!(only_order(&fx).state, OrderState::Expired);
        assert_eq!(e.dbg_state(&id), Some(OrderState::Expired));
    }

    #[test]
    fn gtd_expiry_is_floored_to_sixty_seconds() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        // Desired expiry only 5s out → floored to T0+60s.
        let order = NewOrder {
            tif: TimeInForce::Gtd {
                expires_at: ts(T0 + 5_000),
                post_only: true,
            },
            ..buy(dec!(0.49), dec!(20))
        };
        let _ = open_resting(&mut e, &order);
        assert!(
            e.on_clock(ts(T0 + 59_000)).is_empty(),
            "not expired before the floor"
        );
        let fx = e.on_clock(ts(T0 + 60_000));
        assert_eq!(only_order(&fx).state, OrderState::Expired);
    }

    // ---- Rule 16: BUY notional vs SELL shares -----------------------------

    #[test]
    fn marketable_buy_spends_a_dollar_budget() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.50), dec!(1000))],
            T0,
        ));
        // $50 at 0.50 → 100 shares.
        let order = marketable_buy(dec!(0.50), dec!(50), false);
        let fx = e.on_clock_after_place(&order);
        let f = only_fill(&fx);
        assert_eq!(f.size, sz(dec!(100)));
        assert_eq!(only_order(&fx).state, OrderState::Filled);
    }

    #[test]
    fn structural_qty_kind_mismatch_is_rejected_synchronously() {
        let mut e = engine();
        // A resting GTC carrying a Notional qty is structurally invalid.
        let bad = NewOrder {
            qty: OrderQty::Notional(Dollars::new(dec!(10))),
            ..buy(dec!(0.49), dec!(20))
        };
        let (res, fx) = e.place(&bad, ts(T0));
        assert!(matches!(
            res,
            Err(VenueError::Rejected(RejectReason::Other(_)))
        ));
        assert!(fx.is_empty());
    }

    // ---- Rule 19 / 21: fees, wallet, per-market fee rate ------------------

    #[test]
    fn taker_fee_uses_the_per_market_rate_and_wallet_tracks_cash() {
        let mut e = engine();
        // Register a market with a non-default 0.05 fee rate.
        e.on_window(&market_with_fee(dec!(0.05)), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.50), dec!(1000))],
            T0,
        ));
        let order = marketable_buy(dec!(0.50), dec!(50), false); // 100 sh at 0.50
        let fx = e.on_clock_after_place(&order);
        let f = only_fill(&fx);
        assert_eq!(f.fee, taker_fee(sz(dec!(100)), dec!(0.05), px(dec!(0.50))));
        // Collateral: 10000 − 50 (cost) − fee.
        let w = e.balances();
        let expected = Dollars::new(dec!(10000)) - Dollars::new(dec!(50)) - f.fee;
        assert_eq!(w.collateral_total, expected);
        assert_eq!(w.positions.len(), 1);
        assert_eq!(w.positions[0].size, sz(dec!(100)));
    }

    #[test]
    fn fee_falls_back_to_default_when_market_unknown() {
        let mut e = engine();
        // No on_window → token fee rate unknown → default 0.07.
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.50), dec!(1000))],
            T0,
        ));
        let order = marketable_buy(dec!(0.50), dec!(50), false);
        let fx = e.on_clock_after_place(&order);
        let f = only_fill(&fx);
        assert_eq!(f.fee, taker_fee(sz(dec!(100)), dec!(0.07), px(dec!(0.50))));
    }

    #[test]
    fn maker_fill_accrues_a_rebate_estimate() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.50), dec!(0))],
            &[(dec!(0.60), dec!(50))],
            T0,
        ));
        let _ = open_resting(&mut e, &buy(dec!(0.50), dec!(100)));
        let _ = e.on_trade(
            &up_token(),
            px(dec!(0.50)),
            sz(dec!(100)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        // rebate ≈ 0.20 × taker_fee(100, 0.07, 0.50).
        let expected = Dollars::new(
            dec!(0.20) * taker_fee(sz(dec!(100)), dec!(0.07), px(dec!(0.50))).as_decimal(),
        );
        assert_eq!(e.wallet().rebate_accrued(), expected);
    }

    // ---- Rule 22: next_deadline ------------------------------------------

    #[test]
    fn next_deadline_is_the_min_over_pending() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.49), dec!(0))],
            &[(dec!(0.51), dec!(50))],
            T0,
        ));
        let _a = e.place(&buy(dec!(0.49), dec!(20)), ts(T0)); // activate at T0+100
        let _b = e.place(&buy(dec!(0.48), dec!(20)), ts(T0 + 50)); // activate at T0+150
        assert_eq!(e.next_deadline(), Some(ts(T0 + 100)));
    }

    // ---- Ledger: settlement, merge, daily rebate cycle --------------------

    #[test]
    fn resolved_settles_winning_shares_and_is_idempotent() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.40), dec!(0))],
            &[(dec!(0.60), dec!(50))],
            T0,
        ));
        // 100 Up via a maker fill at 0.40 (collateral 10000 − 40 = 9960).
        let _ = open_resting(&mut e, &buy(dec!(0.40), dec!(100)));
        let _ = e.on_trade(
            &up_token(),
            px(dec!(0.40)),
            sz(dec!(100)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        let before = e.balances().collateral_total;
        // Resolve Up → 100 winning shares pay $1 each, positions zeroed.
        let _ = e.on_window(
            &market(),
            WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
            ts(T0 + 300_000),
        );
        let after = e.balances();
        assert_eq!(after.collateral_total, before + Dollars::new(dec!(100)));
        assert!(after.positions.is_empty());
        // Idempotent: a repeated Resolved must not pay again.
        let _ = e.on_window(
            &market(),
            WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
            ts(T0 + 300_001),
        );
        assert_eq!(e.balances().collateral_total, after.collateral_total);
    }

    #[test]
    fn merge_converts_matched_pairs_and_unknown_window_errors() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        // 100 Up via a maker fill at 0.40.
        e.on_book(&snapshot(
            &[(dec!(0.40), dec!(0))],
            &[(dec!(0.60), dec!(50))],
            T0,
        ));
        let _ = open_resting(&mut e, &buy(dec!(0.40), dec!(100)));
        let _ = e.on_trade(
            &up_token(),
            px(dec!(0.40)),
            sz(dec!(100)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        // 60 Down via a marketable taker buy ($30 at 0.50 → 60 shares).
        e.on_book(&down_snapshot(
            &[(dec!(0.45), dec!(0))],
            &[(dec!(0.50), dec!(60))],
            T0,
        ));
        let dbuy = NewOrder {
            token_id: down_token(),
            outcome: Outcome::Down,
            ..marketable_buy(dec!(0.50), dec!(30), false)
        };
        let _ = e.place(&dbuy, ts(T0));
        let _ = e.on_clock(ts(T0 + PLACE_MS));

        let before = e.balances().collateral_total;
        // matched = min(100, 60) = 60, capped at the requested 40.
        let (res, fx) = e.merge(window(), sz(dec!(40)), ts(T0 + 1000));
        assert!(fx.is_empty());
        assert_eq!(res.expect("merged"), sz(dec!(40)));
        assert_eq!(
            e.balances().collateral_total,
            before + Dollars::new(dec!(40))
        );

        // Unknown window → error.
        let other = WindowId {
            open_time: ts(T0 + 999_000),
            ..window()
        };
        let (res, _) = e.merge(other, sz(dec!(5)), ts(T0 + 1000));
        assert!(matches!(res, Err(VenueError::VenueInternal(_))));
    }

    #[test]
    fn rebate_cycle_credits_after_a_day_via_the_deadline() {
        let mut e = engine();
        e.on_window(&market(), WindowLifecycle::Open, ts(T0));
        e.on_book(&snapshot(
            &[(dec!(0.50), dec!(0))],
            &[(dec!(0.60), dec!(50))],
            T0,
        ));
        // 1000-share maker fill at 0.50 accrues ≈ $3.50 rebate (≥ the $1 floor).
        let _ = open_resting(&mut e, &buy(dec!(0.50), dec!(1000)));
        let _ = e.on_trade(
            &up_token(),
            px(dec!(0.50)),
            sz(dec!(1000)),
            Side::Sell,
            ts(T0 + 200),
            ts(T0 + 200),
        );
        let accrued = e.wallet().rebate_accrued();
        assert!(accrued >= Dollars::new(dec!(1)), "accrued {accrued}");
        // The cycle is armed at the fill time + 24h; that is the next deadline.
        let due = e.next_deadline().expect("a rebate deadline");
        assert_eq!(due, ts(T0 + 200).saturating_add(REBATE_PERIOD));
        let cash_before = e.balances().collateral_total;
        // Before the day elapses: nothing credited.
        let _ = e.on_clock(ts(T0 + 201));
        assert_eq!(e.wallet().rebate_credited(), Dollars::ZERO);
        // At the deadline: credited to cash, accrual reset, line tracked.
        let _ = e.on_clock(due);
        assert_eq!(e.wallet().rebate_credited(), accrued);
        assert_eq!(e.balances().collateral_total, cash_before + accrued);
        assert!(e.wallet().rebate_accrued().is_zero());
    }

    // ---- helper: place then activate a marketable order -------------------

    impl MatchEngine {
        fn on_clock_after_place(&mut self, order: &NewOrder) -> Vec<PaperEffect> {
            let _ = self.place(order, ts(T0));
            self.on_clock(ts(T0 + PLACE_MS))
        }
    }
}
