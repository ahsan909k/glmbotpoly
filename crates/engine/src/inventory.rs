//! Per-window inventory and pair accounting (CLAUDE.md §8): the strategy's
//! bookkeeping brain, downstream of every fill and the input to quoting, the
//! merge policy, and the risk manager.
//!
//! [`WindowInventory`] folds [`Fill`](core_types::Fill)s into one window's two
//! sides — shares and cost per side — and derives, through the single
//! [`InventorySnapshot::derive`] path, the matched pairs, pair cost, signed
//! excess, and worst-case-if-excess-loses dollars. On top of the raw book it
//! answers the §8 policy questions as **pure queries**: the pair-cost discipline
//! that authorizes a passive add ([`WindowInventory::authorizes_passive_add`]),
//! the soft/hard excess caps that constrain quoting
//! ([`WindowInventory::excess_constraint`]), and the merge policy that requests
//! recycling matched pairs back to collateral ([`WindowInventory::merge_intent`]).
//! On resolution [`WindowInventory::settle`] closes the books and produces a
//! [`SettlementSummary`] for analytics.
//!
//! [`InventoryManager`] holds one book per window and folds the bus
//! [`Event`](core_types::Event)s — `Fill` and `Window` — that mutate it, emitting
//! [`InventoryEffect`]s the orchestrator publishes (`Event::Inventory` /
//! `Event::Settlement`). It is **pure, sans-IO, and event-time-driven** (no wall
//! clock; snapshot times come from the fill, settlement time from the window's
//! close): a journal replay reproduces the state and the effect stream
//! bit-for-bit, satisfying the §3/§9 restart-rebuild requirement
//! ([`InventoryManager::rebuild`]).
//!
//! Two deliberate scopes, documented so they read as choices, not gaps:
//! - **Merge is advisory here.** [`WindowInventory::merge_intent`] *requests* a
//!   merge and [`WindowInventory::apply_merge`] reflects a confirmed one, but
//!   `on_event` never drives a merge: merge *execution* (`PaperVenue::merge` / the
//!   live CTF merge → the engine) is deferred per CLAUDE.md, so no merge mutates
//!   state at runtime yet and rebuild from `Fill`+`Window` stays exact. When merge
//!   execution is wired, that task must add a journaled merge-confirmation event
//!   (consumed here) to keep rebuild exact.
//! - **Config-boundary mapping is deferred.** [`InventoryParams`] is the
//!   engine-local tunable bundle (mirroring [`NormalizerParams`](crate::NormalizerParams));
//!   the `config::EngineParams` → `InventoryParams` map at the `bot` boundary
//!   lands with the engine-bus-wiring task, exactly as the normalizer deferred its
//!   own mapping.

use std::collections::{HashMap, HashSet};

use core_types::{
    Decimal, Dollars, Event, Fill, InventorySnapshot, Outcome, Price, SettlementSummary, Side,
    SideInventory, Size, TimestampMs, WindowId, WindowLifecycle,
};
use rust_decimal::dec;

/// Builds a [`Size`] from a non-negative integer literal (defaults only).
#[expect(
    clippy::unwrap_used,
    reason = "u32 literals are non-negative; Size::new rejects only negatives"
)]
fn shares(n: u32) -> Size {
    Size::new(Decimal::from(n)).unwrap()
}

/// Engine-local inventory tunables (CLAUDE.md §8).
///
/// The bundle the quoting/risk layer destructures into the per-window pure
/// queries. Mapped from `config::EngineParams` at the `bot` boundary like
/// [`NormalizerParams`](crate::NormalizerParams); [`Default`] mirrors the
/// committed `config/default.toml` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InventoryParams {
    /// Pair-cost discipline (§8): only add passively while the post-trade
    /// `avg_up + avg_down` stays at or below this.
    pub pair_cost_threshold: Decimal,
    /// Excess (unmatched) shares at which quoting narrows to the
    /// deficit-reducing side only.
    pub soft_cap_excess: Size,
    /// Excess shares at which passive quoting stops entirely.
    pub hard_cap_excess: Size,
    /// Matched pairs at or above which a merge is requested (capital recycling).
    pub merge_min_pairs: Size,
    /// Maximum worst-case loss per window — the binding constraint on sizing.
    pub max_worst_case_loss: Dollars,
}

impl Default for InventoryParams {
    fn default() -> Self {
        Self {
            pair_cost_threshold: dec!(0.98),
            soft_cap_excess: shares(50),
            hard_cap_excess: shares(100),
            merge_min_pairs: shares(25),
            max_worst_case_loss: Dollars::new(Decimal::from(25)),
        }
    }
}

/// How the excess (unmatched) inventory constrains quoting (§8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcessConstraint {
    /// `|excess|` is below the soft cap — quote both sides normally.
    Normal,
    /// `|excess|` is at or above the soft cap on `excess_side` — quote only the
    /// deficit-reducing side (the opposite of `excess_side`).
    SoftCapped {
        /// The side currently holding the excess.
        excess_side: Outcome,
    },
    /// `|excess|` is at or above the hard cap on `excess_side` — stop passive
    /// making entirely (optionally work the excess off at/above fair).
    HardCapped {
        /// The side currently holding the excess.
        excess_side: Outcome,
    },
}

/// A request to merge matched Up/Down pairs back to collateral (§8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeIntent {
    /// Window the pairs belong to.
    pub window: WindowId,
    /// Matched pairs to merge.
    pub pairs: Size,
}

/// A side effect of folding an event into the [`InventoryManager`], for the
/// orchestrator to publish on the bus.
#[derive(Debug, Clone, PartialEq)]
pub enum InventoryEffect {
    /// A fresh inventory snapshot for a window (publish as `Event::Inventory`).
    Snapshot(InventorySnapshot),
    /// A window settled (publish as `Event::Settlement`).
    Settled(SettlementSummary),
}

/// Picks one side from a raw `(up, down)` pair (`SideInventory` is `Copy`).
fn side_of(up: &SideInventory, down: &SideInventory, outcome: Outcome) -> SideInventory {
    match outcome {
        Outcome::Up => *up,
        Outcome::Down => *down,
    }
}

/// How the excess (unmatched) inventory constrains quoting given the soft/hard
/// caps (§8), computed straight from the two raw sides.
///
/// The single definition of the §8 cap logic: [`WindowInventory::excess_constraint`]
/// delegates here, and the quoting calculator calls it on an
/// [`InventorySnapshot`](core_types::InventorySnapshot)'s sides — so the rule can
/// never drift between the inventory book and the quoter.
#[must_use]
pub fn excess_constraint_sides(
    up: &SideInventory,
    down: &SideInventory,
    soft: Size,
    hard: Size,
) -> ExcessConstraint {
    let excess = up.shares.as_decimal() - down.shares.as_decimal();
    if excess.is_zero() {
        return ExcessConstraint::Normal;
    }
    let (excess_side, abs) = if excess > Decimal::ZERO {
        (Outcome::Up, excess)
    } else {
        (Outcome::Down, -excess)
    };
    let abs = Size::new(abs).unwrap_or(Size::ZERO);
    if abs >= hard {
        ExcessConstraint::HardCapped { excess_side }
    } else if abs >= soft {
        ExcessConstraint::SoftCapped { excess_side }
    } else {
        ExcessConstraint::Normal
    }
}

/// The pair cost (`avg_up + avg_down`) that would result from passively adding
/// `size` shares of `add` at `price` to the two raw sides. `None` when the
/// *other* side is empty — there is no pair to discipline.
///
/// The single definition behind [`WindowInventory::pair_cost_after_add`] and the
/// quoting calculator.
#[must_use]
pub fn pair_cost_after_add_sides(
    up: &SideInventory,
    down: &SideInventory,
    add: Outcome,
    price: Price,
    size: Size,
) -> Option<Decimal> {
    let mut added = side_of(up, down, add);
    added.shares = added.shares + size;
    added.cost = added.cost + Dollars::new(price.as_decimal() * size.as_decimal());
    let other = side_of(up, down, add.opposite());
    match (added.avg_price(), other.avg_price()) {
        (Some(a), Some(o)) => Some(a + o),
        _ => None,
    }
}

/// Whether a passive add of `size` shares of `add` at `price` is authorized by
/// pair-cost discipline (§8): the post-trade pair cost must stay at or below
/// `threshold`. A one-sided add (the other side empty) is **authorized** —
/// pair-cost discipline governs pairs; a lone side is governed by the excess
/// caps.
///
/// The single definition behind [`WindowInventory::authorizes_passive_add`] and
/// the quoting calculator.
#[must_use]
pub fn authorizes_passive_add_sides(
    up: &SideInventory,
    down: &SideInventory,
    add: Outcome,
    price: Price,
    size: Size,
    threshold: Decimal,
) -> bool {
    pair_cost_after_add_sides(up, down, add, price, size).is_none_or(|pc| pc <= threshold)
}

/// One window's share/cost book, folded from fills (CLAUDE.md §8).
///
/// Holds no market metadata: a [`Fill`] already carries its decoded `outcome`
/// and charged `fee`, and settlement reads the winner + close time off the
/// resolved-window event — so a fill that arrives before its `Window` event is
/// folded identically on live and on replay. `cash_flow` is the single signed
/// money accumulator (buys, sells, fees, merge credits); the bottom-line realized
/// PnL falls out at settlement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowInventory {
    window: WindowId,
    up: SideInventory,
    down: SideInventory,
    cash_flow: Dollars,
    fees_paid: Dollars,
    merged_pairs: Size,
    settled: Option<Outcome>,
}

impl WindowInventory {
    /// A fresh, empty book for `window`.
    #[must_use]
    pub fn new(window: WindowId) -> Self {
        Self {
            window,
            up: SideInventory::default(),
            down: SideInventory::default(),
            cash_flow: Dollars::ZERO,
            fees_paid: Dollars::ZERO,
            merged_pairs: Size::ZERO,
            settled: None,
        }
    }

    fn side(&self, outcome: Outcome) -> &SideInventory {
        match outcome {
            Outcome::Up => &self.up,
            Outcome::Down => &self.down,
        }
    }

    fn side_mut(&mut self, outcome: Outcome) -> &mut SideInventory {
        match outcome {
            Outcome::Up => &mut self.up,
            Outcome::Down => &mut self.down,
        }
    }

    /// Folds one fill into the book.
    ///
    /// BUY adds shares and cost. SELL reduces at the **moving-average** cost basis
    /// (the average is captured before mutating, so it is preserved across a
    /// partial sell) and is **long-only**: an over-sell is clamped to the held
    /// shares (`SideInventory.shares` is a non-negative [`Size`]) — defensive, as
    /// the engine only ever sells what it holds. The trade cash and the fee both
    /// flow into `cash_flow`, and the fee also accumulates into `fees_paid`.
    pub fn apply_fill(&mut self, fill: &Fill) {
        let price = fill.price.as_decimal();
        let size = fill.size.as_decimal();
        let notional = Dollars::new(price * size);
        self.fees_paid = self.fees_paid + fill.fee;
        match fill.side {
            Side::Buy => {
                {
                    let s = self.side_mut(fill.outcome);
                    s.shares = s.shares + fill.size;
                    s.cost = s.cost + notional;
                }
                self.cash_flow = self.cash_flow - notional - fill.fee;
            }
            Side::Sell => {
                let proceeds = {
                    let s = self.side_mut(fill.outcome);
                    // Capture the average BEFORE mutating (moving-average basis).
                    let avg = s.avg_price().unwrap_or(Decimal::ZERO);
                    let sold = fill.size.min(s.shares); // long-only clamp
                    let cost_removed = Dollars::new(avg * sold.as_decimal());
                    s.shares = s.shares.saturating_sub(sold);
                    s.cost = s.cost - cost_removed;
                    Dollars::new(price * sold.as_decimal())
                };
                self.cash_flow = self.cash_flow + proceeds - fill.fee;
            }
        }
    }

    /// Matched pairs currently held = `min(up.shares, down.shares)`.
    #[must_use]
    pub fn matched_pairs(&self) -> Size {
        self.up.shares.min(self.down.shares)
    }

    /// A point-in-time snapshot at `ts` (the derived fields come from
    /// [`InventorySnapshot::derive`]).
    #[must_use]
    pub fn snapshot(&self, ts: TimestampMs) -> InventorySnapshot {
        InventorySnapshot::derive(self.window, self.up, self.down, ts)
    }

    /// Dollars lost if every unmatched share expires worthless — the cost sunk
    /// into the excess (§8's binding risk input). Time-independent.
    #[must_use]
    pub fn worst_case_if_excess_loses(&self) -> Dollars {
        self.snapshot(TimestampMs::from_millis(0))
            .worst_case_if_excess_loses
    }

    /// True when the worst-case excess loss exceeds `limit` (the per-window
    /// `max_worst_case_loss`).
    #[must_use]
    pub fn worst_case_exceeds(&self, limit: Dollars) -> bool {
        self.worst_case_if_excess_loses() > limit
    }

    /// The pair cost (`avg_up + avg_down`) that would result from passively
    /// adding `size` shares of `add` at `price`. `None` when the *other* side is
    /// empty — there is no pair to discipline.
    #[must_use]
    pub fn pair_cost_after_add(&self, add: Outcome, price: Price, size: Size) -> Option<Decimal> {
        pair_cost_after_add_sides(&self.up, &self.down, add, price, size)
    }

    /// Whether a passive add of `size` shares of `add` at `price` is authorized by
    /// pair-cost discipline (§8): the post-trade pair cost must stay at or below
    /// `threshold`. A one-sided add (the other side empty) is **authorized** —
    /// pair-cost discipline governs pairs; a lone side is governed by the excess
    /// caps.
    #[must_use]
    pub fn authorizes_passive_add(
        &self,
        add: Outcome,
        price: Price,
        size: Size,
        threshold: Decimal,
    ) -> bool {
        authorizes_passive_add_sides(&self.up, &self.down, add, price, size, threshold)
    }

    /// How the current excess constrains quoting given the soft/hard caps (§8).
    #[must_use]
    pub fn excess_constraint(&self, soft: Size, hard: Size) -> ExcessConstraint {
        excess_constraint_sides(&self.up, &self.down, soft, hard)
    }

    /// Requests a merge when at least `min_pairs` matched pairs have accumulated
    /// (§8 capital recycling). `None` once the window is settled, when no pairs
    /// are held, or below the threshold.
    #[must_use]
    pub fn merge_intent(&self, min_pairs: Size) -> Option<MergeIntent> {
        if self.settled.is_some() {
            return None;
        }
        let matched = self.matched_pairs();
        if matched.is_zero() || matched < min_pairs {
            return None;
        }
        Some(MergeIntent {
            window: self.window,
            pairs: matched,
        })
    }

    /// Reflects a confirmed merge of `pairs` matched pairs back to collateral:
    /// reduces each side by the merged count at its average cost (preserving the
    /// average), credits `$1` per pair to `cash_flow`, and accumulates
    /// `merged_pairs`. Caps at the matched count; returns the pairs actually
    /// merged.
    ///
    /// Not driven by [`InventoryManager::on_event`] — merge *execution* is
    /// deferred (see the module docs); this keeps the data structure complete and
    /// testable.
    pub fn apply_merge(&mut self, pairs: Size) -> Size {
        let n = pairs.min(self.matched_pairs());
        if n.is_zero() {
            return Size::ZERO;
        }
        let avg_up = self.up.avg_price().unwrap_or(Decimal::ZERO);
        let avg_down = self.down.avg_price().unwrap_or(Decimal::ZERO);
        let nd = n.as_decimal();
        self.up.shares = self.up.shares.saturating_sub(n);
        self.up.cost = self.up.cost - Dollars::new(avg_up * nd);
        self.down.shares = self.down.shares.saturating_sub(n);
        self.down.cost = self.down.cost - Dollars::new(avg_down * nd);
        self.cash_flow = self.cash_flow + Dollars::new(nd); // $1 per pair
        self.merged_pairs = self.merged_pairs + n;
        n
    }

    /// Closes the window's books on resolution and returns the summary. Realized
    /// PnL = trading `cash_flow` + the winning side's `$1`/share payout. Marks the
    /// book settled (idempotency is the manager's responsibility).
    pub fn settle(&mut self, outcome: Outcome, ts: TimestampMs) -> SettlementSummary {
        let payout = Dollars::new(self.side(outcome).shares.as_decimal());
        let realized = self.cash_flow + payout;
        self.settled = Some(outcome);
        SettlementSummary::close(
            self.window,
            outcome,
            self.up,
            self.down,
            self.merged_pairs,
            self.fees_paid,
            realized,
            ts,
        )
    }

    /// The window this book belongs to.
    #[must_use]
    pub fn window(&self) -> WindowId {
        self.window
    }

    /// Up-side holdings.
    #[must_use]
    pub fn up(&self) -> SideInventory {
        self.up
    }

    /// Down-side holdings.
    #[must_use]
    pub fn down(&self) -> SideInventory {
        self.down
    }

    /// Net trading cash flow so far (buys, sells, fees, merge credits).
    #[must_use]
    pub fn cash_flow(&self) -> Dollars {
        self.cash_flow
    }

    /// Cumulative taker fees paid on this window.
    #[must_use]
    pub fn fees_paid(&self) -> Dollars {
        self.fees_paid
    }

    /// Pairs merged back to collateral so far.
    #[must_use]
    pub fn merged_pairs(&self) -> Size {
        self.merged_pairs
    }

    /// The settled outcome, if the window has settled.
    #[must_use]
    pub fn settled(&self) -> Option<Outcome> {
        self.settled
    }
}

/// All live and parked window books, folded from the bus (CLAUDE.md §8).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InventoryManager {
    books: HashMap<WindowId, WindowInventory>,
    settled: HashSet<WindowId>,
}

impl InventoryManager {
    /// An empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one bus event, returning the effects to publish.
    ///
    /// Mutates only on `Event::Fill` (→ one [`InventoryEffect::Snapshot`] stamped
    /// with the fill's venue time) and `Event::Window { Resolved }` (→ at most one
    /// [`InventoryEffect::Settled`], idempotent via the settled set, stamped with
    /// the window's close time). Everything else is ignored. Never iterates
    /// `books`, so per-event effect order is stable and a replay reproduces the
    /// effect stream exactly. A fill for an already-settled window still updates
    /// the book but never re-settles; an untraded window that resolves emits no
    /// summary.
    pub fn on_event(&mut self, event: &Event) -> Vec<InventoryEffect> {
        match event {
            Event::Fill(fill) => {
                let book = self
                    .books
                    .entry(fill.window)
                    .or_insert_with(|| WindowInventory::new(fill.window));
                book.apply_fill(fill);
                vec![InventoryEffect::Snapshot(book.snapshot(fill.ts_venue))]
            }
            Event::Window {
                market,
                lifecycle: WindowLifecycle::Resolved { outcome },
            } => {
                if !self.settled.insert(market.window) {
                    return Vec::new(); // already settled — idempotent
                }
                match self.books.get_mut(&market.window) {
                    Some(book) => {
                        vec![InventoryEffect::Settled(
                            book.settle(*outcome, market.close_time),
                        )]
                    }
                    // A window we never traded: nothing to settle, no summary.
                    None => Vec::new(),
                }
            }
            _ => Vec::new(),
        }
    }

    /// Rebuilds a manager by folding journaled events (discarding effects). The
    /// fold is deterministic and event-time-driven, so this reproduces the live
    /// manager exactly — the §3/§9 restart-rebuild guarantee.
    #[must_use]
    pub fn rebuild<'a>(events: impl IntoIterator<Item = &'a Event>) -> Self {
        let mut manager = Self::default();
        for event in events {
            let _ = manager.on_event(event);
        }
        manager
    }

    /// The book for `window`, if any.
    #[must_use]
    pub fn book(&self, window: WindowId) -> Option<&WindowInventory> {
        self.books.get(&window)
    }

    /// Whether `window` has settled.
    #[must_use]
    pub fn is_settled(&self, window: WindowId) -> bool {
        self.settled.contains(&window)
    }

    /// Number of window books held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.books.len()
    }

    /// True when no window books are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.books.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use core_types::{
        Asset, ConditionId, FeeParams, Liquidity, MarketInfo, OrderId, ResolutionSource, RoundDir,
        Series, TickSize, TokenId, TokenPair, WindowDuration,
    };
    use rust_decimal::dec;

    use super::*;

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = 1_781_000_300_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }

    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }

    /// A fill on `window()`. `fee == 0` ⇒ maker, else taker (for the demo only;
    /// the book reads the fee verbatim either way).
    fn fill(outcome: Outcome, side: Side, price: Decimal, size: Decimal, fee: Decimal) -> Fill {
        Fill {
            order_id: OrderId::new("o").unwrap(),
            trade_id: None,
            window: window(),
            token_id: TokenId::new("1").unwrap(),
            outcome,
            side,
            price: px(price),
            size: sz(size),
            liquidity: if fee.is_zero() {
                Liquidity::Maker
            } else {
                Liquidity::Taker
            },
            fee: Dollars::new(fee),
            ts_venue: TimestampMs::from_millis(OPEN_MS + 100),
            ts_local: TimestampMs::from_millis(OPEN_MS + 100),
        }
    }

    fn buy(outcome: Outcome, price: Decimal, size: Decimal) -> Fill {
        fill(outcome, Side::Buy, price, size, dec!(0))
    }

    fn sell(outcome: Outcome, price: Decimal, size: Decimal) -> Fill {
        fill(outcome, Side::Sell, price, size, dec!(0))
    }

    fn market() -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: TimestampMs::from_millis(CLOSE_MS),
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

    fn resolved(outcome: Outcome) -> Event {
        Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Resolved { outcome },
        }
    }

    // ---- 1. BUY accumulation ----------------------------------------------

    #[test]
    fn buy_accumulates_shares_and_cost() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(100)));
        book.apply_fill(&buy(Outcome::Up, dec!(0.50), dec!(100)));
        assert_eq!(book.up().shares, sz(dec!(200)));
        assert_eq!(book.up().cost, Dollars::new(dec!(90))); // 40 + 50
        assert_eq!(book.up().avg_price(), Some(dec!(0.45)));
        assert_eq!(book.cash_flow(), Dollars::new(dec!(-90)));
        assert_eq!(book.fees_paid(), Dollars::ZERO);
    }

    // ---- 2. SELL preserves the frozen average -----------------------------

    #[test]
    fn sell_preserves_frozen_average() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(100))); // avg 0.40
        book.apply_fill(&sell(Outcome::Up, dec!(0.60), dec!(40))); // sell 40 @ 0.60
        // Average stays 0.40: cost removed at avg, not at the sale price.
        assert_eq!(book.up().shares, sz(dec!(60)));
        assert_eq!(book.up().avg_price(), Some(dec!(0.40)));
        assert_eq!(book.up().cost, Dollars::new(dec!(24))); // 60 × 0.40
        // cash_flow = -40 (buy) + 24 (sell proceeds 0.60×40)
        assert_eq!(book.cash_flow(), Dollars::new(dec!(-16)));
    }

    // ---- 3. Mixed sequence → snapshot + realized --------------------------

    #[test]
    fn mixed_sequence_snapshot_and_realized() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(100)));
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(100)));
        book.apply_fill(&sell(Outcome::Up, dec!(0.60), dec!(40)));
        // cash_flow = -40 - 55 + 24 = -71
        assert_eq!(book.cash_flow(), Dollars::new(dec!(-71)));

        let snap = book.snapshot(TimestampMs::from_millis(OPEN_MS + 100));
        assert_eq!(snap.up.shares, sz(dec!(60)));
        assert_eq!(snap.down.shares, sz(dec!(100)));
        assert_eq!(snap.matched_pairs, sz(dec!(60)));
        assert_eq!(snap.pair_cost, Some(dec!(0.95))); // 0.40 + 0.55
        assert_eq!(snap.excess, dec!(-40)); // 40 Down excess

        // Resolve Up: 60 winning Up shares pay $1 → realized = -71 + 60 = -11.
        let summary = book.settle(Outcome::Up, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(summary.realized_pnl, Dollars::new(dec!(-11)));
        // Resolve Down instead (fresh book) → realized = -71 + 100 = +29.
        let mut book2 = WindowInventory::new(window());
        book2.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(100)));
        book2.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(100)));
        book2.apply_fill(&sell(Outcome::Up, dec!(0.60), dec!(40)));
        let s2 = book2.settle(Outcome::Down, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(s2.realized_pnl, Dollars::new(dec!(29)));
    }

    // ---- 4. Over-sell clamp -----------------------------------------------

    #[test]
    fn over_sell_clamps_to_zero_no_short() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(30)));
        book.apply_fill(&sell(Outcome::Up, dec!(0.50), dec!(50))); // sell 50, hold 30
        assert_eq!(book.up().shares, Size::ZERO);
        assert_eq!(book.up().cost, Dollars::ZERO);
        // Proceeds only on the 30 actually sold: -12 + 15 = +3.
        assert_eq!(book.cash_flow(), Dollars::new(dec!(3)));
        let s = book.settle(Outcome::Up, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(s.realized_pnl, Dollars::new(dec!(3)));
    }

    // ---- 5. pair_cost_after_add -------------------------------------------

    #[test]
    fn pair_cost_after_add_both_and_one_sided() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(100))); // avg down 0.55
        // Other side empty: adding only Up has no pair until Down exists — but
        // Down IS held here, so adding Up produces a pair cost.
        let pc = book.pair_cost_after_add(Outcome::Up, px(dec!(0.40)), sz(dec!(50)));
        assert_eq!(pc, Some(dec!(0.95))); // 0.40 + 0.55

        // Empty book: adding only Up, the other (Down) side is empty → None.
        let empty = WindowInventory::new(window());
        assert_eq!(
            empty.pair_cost_after_add(Outcome::Up, px(dec!(0.40)), sz(dec!(50))),
            None
        );
    }

    // ---- 6. authorizes_passive_add ----------------------------------------

    #[test]
    fn authorizes_passive_add_threshold_and_one_sided() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(100)));
        // Adding Up @ 0.40 → pair cost 0.95.
        assert!(book.authorizes_passive_add(Outcome::Up, px(dec!(0.40)), sz(dec!(50)), dec!(0.98)));
        // Threshold exactly at the post-trade pair cost: authorized (≤).
        assert!(book.authorizes_passive_add(Outcome::Up, px(dec!(0.40)), sz(dec!(50)), dec!(0.95)));
        // Adding Up @ 0.45 → pair cost 1.00 > 0.98: refused.
        assert!(!book.authorizes_passive_add(
            Outcome::Up,
            px(dec!(0.45)),
            sz(dec!(50)),
            dec!(0.98)
        ));
        // One-sided add on an empty book is authorized (no pair to discipline).
        let empty = WindowInventory::new(window());
        assert!(empty.authorizes_passive_add(
            Outcome::Up,
            px(dec!(0.99)),
            sz(dec!(50)),
            dec!(0.50)
        ));
    }

    // ---- 7. excess_constraint transitions ---------------------------------

    #[test]
    fn excess_constraint_normal_soft_hard() {
        let soft = sz(dec!(50));
        let hard = sz(dec!(100));

        // Balanced → Normal.
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(40)));
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(40)));
        assert_eq!(book.excess_constraint(soft, hard), ExcessConstraint::Normal);

        // 60 Up excess (≥ soft, < hard) → SoftCapped on Up.
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(60)));
        assert_eq!(
            book.excess_constraint(soft, hard),
            ExcessConstraint::SoftCapped {
                excess_side: Outcome::Up
            }
        );

        // 120 Up excess (≥ hard) → HardCapped on Up.
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(60)));
        assert_eq!(
            book.excess_constraint(soft, hard),
            ExcessConstraint::HardCapped {
                excess_side: Outcome::Up
            }
        );

        // Down-heavy book → side flips to Down.
        let mut down_book = WindowInventory::new(window());
        down_book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(80)));
        assert_eq!(
            down_book.excess_constraint(soft, hard),
            ExcessConstraint::SoftCapped {
                excess_side: Outcome::Down
            }
        );
    }

    // ---- 8. merge_intent thresholds ---------------------------------------

    #[test]
    fn merge_intent_thresholds() {
        let min = sz(dec!(25));
        let mut book = WindowInventory::new(window());
        // No pairs → None.
        assert_eq!(book.merge_intent(min), None);

        // 20 matched pairs, below threshold → None.
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(20)));
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(20)));
        assert_eq!(book.merge_intent(min), None);

        // 30 matched pairs (≥ 25) → Some(all matched).
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(10)));
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(10)));
        assert_eq!(
            book.merge_intent(min),
            Some(MergeIntent {
                window: window(),
                pairs: sz(dec!(30))
            })
        );

        // After settlement → None.
        book.settle(Outcome::Up, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(book.merge_intent(min), None);
    }

    // ---- 9. apply_merge ---------------------------------------------------

    #[test]
    fn apply_merge_reduces_at_avg_and_credits_dollar_per_pair() {
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(30))); // avg up 0.40
        book.apply_fill(&buy(Outcome::Down, dec!(0.55), dec!(20))); // avg down 0.55
        // Merge 20 (matched = min(30,20) = 20).
        let merged = book.apply_merge(sz(dec!(20)));
        assert_eq!(merged, sz(dec!(20)));
        assert_eq!(book.up().shares, sz(dec!(10)));
        assert_eq!(book.up().avg_price(), Some(dec!(0.40))); // avg preserved
        assert_eq!(book.down().shares, Size::ZERO);
        assert_eq!(book.merged_pairs(), sz(dec!(20)));
        // cash_flow = -12 - 11 + 20 = -3.
        assert_eq!(book.cash_flow(), Dollars::new(dec!(-3)));

        // Requesting more than matched caps at matched; a one-sided book → 0.
        assert_eq!(book.apply_merge(sz(dec!(100))), Size::ZERO); // down is empty now
        // Resolve Up: 10 Up shares pay $1 → realized = -3 + 10 = +7.
        let s = book.settle(Outcome::Up, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(s.realized_pnl, Dollars::new(dec!(7)));
    }

    // ---- 10. worst_case_exceeds -------------------------------------------

    #[test]
    fn worst_case_exceeds_limit() {
        let mut book = WindowInventory::new(window());
        // 50 Up excess @ 0.40 → worst case $20.
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(50)));
        assert_eq!(book.worst_case_if_excess_loses(), Dollars::new(dec!(20)));
        assert!(book.worst_case_exceeds(Dollars::new(dec!(15))));
        assert!(!book.worst_case_exceeds(Dollars::new(dec!(25))));
    }

    // ---- 11. settle realized + ts -----------------------------------------

    #[test]
    fn settle_zero_and_one_sided() {
        // Zero positions: realized 0.
        let mut empty = WindowInventory::new(window());
        let s = empty.settle(Outcome::Up, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(s.realized_pnl, Dollars::ZERO);
        assert_eq!(s.ts, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(s.matched_pairs, Size::ZERO);

        // One-sided Up that loses (Down wins): realized = -cost.
        let mut book = WindowInventory::new(window());
        book.apply_fill(&buy(Outcome::Up, dec!(0.40), dec!(50)));
        let lose = book.settle(Outcome::Down, TimestampMs::from_millis(CLOSE_MS));
        assert_eq!(lose.realized_pnl, Dollars::new(dec!(-20)));
    }

    // ---- 12. on_event -----------------------------------------------------

    #[test]
    fn on_event_fill_emits_one_snapshot() {
        let mut mgr = InventoryManager::new();
        let effects = mgr.on_event(&Event::Fill(Arc::new(buy(
            Outcome::Up,
            dec!(0.40),
            dec!(10),
        ))));
        assert_eq!(effects.len(), 1);
        match &effects[0] {
            InventoryEffect::Snapshot(s) => {
                assert_eq!(s.up.shares, sz(dec!(10)));
                assert_eq!(s.ts, TimestampMs::from_millis(OPEN_MS + 100)); // fill ts_venue
            }
            InventoryEffect::Settled(_) => panic!("expected a snapshot"),
        }
    }

    #[test]
    fn on_event_settlement_is_idempotent() {
        let mut mgr = InventoryManager::new();
        mgr.on_event(&Event::Fill(Arc::new(buy(
            Outcome::Up,
            dec!(0.40),
            dec!(10),
        ))));
        let first = mgr.on_event(&resolved(Outcome::Up));
        assert_eq!(first.len(), 1);
        match &first[0] {
            InventoryEffect::Settled(s) => {
                assert_eq!(s.outcome, Outcome::Up);
                assert_eq!(s.ts, TimestampMs::from_millis(CLOSE_MS)); // close_time
                assert_eq!(s.realized_pnl, Dollars::new(dec!(6))); // -4 + 10
            }
            InventoryEffect::Snapshot(_) => panic!("expected a settlement"),
        }
        // Second Resolved → no effect.
        assert!(mgr.on_event(&resolved(Outcome::Up)).is_empty());
        assert!(mgr.is_settled(window()));
    }

    #[test]
    fn on_event_untraded_window_emits_no_summary() {
        let mut mgr = InventoryManager::new();
        assert!(mgr.on_event(&resolved(Outcome::Up)).is_empty());
        assert!(mgr.is_settled(window()));
        assert!(mgr.book(window()).is_none());
    }

    #[test]
    fn on_event_fill_after_settle_snapshots_without_resettling() {
        let mut mgr = InventoryManager::new();
        mgr.on_event(&Event::Fill(Arc::new(buy(
            Outcome::Up,
            dec!(0.40),
            dec!(10),
        ))));
        mgr.on_event(&resolved(Outcome::Up));
        // A late fill still updates the book + snapshots, but never re-settles.
        let effects = mgr.on_event(&Event::Fill(Arc::new(buy(
            Outcome::Up,
            dec!(0.40),
            dec!(5),
        ))));
        assert_eq!(effects.len(), 1);
        assert!(matches!(effects[0], InventoryEffect::Snapshot(_)));
        assert_eq!(
            mgr.book(window())
                .map(WindowInventory::up)
                .map(|s| s.shares),
            Some(sz(dec!(15)))
        );
    }

    #[test]
    fn on_event_ignores_unrelated_events() {
        let mut mgr = InventoryManager::new();
        // A non-Resolved Window lifecycle is ignored (no settle, no book).
        let open = Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Open,
        };
        assert!(mgr.on_event(&open).is_empty());
        assert!(mgr.book(window()).is_none());
        assert!(!mgr.is_settled(window()));
    }

    // ---- 13. restart-rebuild equivalence ----------------------------------

    #[test]
    fn rebuild_reproduces_state_and_effects() {
        let events = [
            Event::Window {
                market: market(),
                lifecycle: WindowLifecycle::Open,
            },
            Event::Fill(Arc::new(buy(Outcome::Up, dec!(0.40), dec!(100)))),
            Event::Fill(Arc::new(buy(Outcome::Down, dec!(0.55), dec!(80)))),
            Event::Fill(Arc::new(sell(Outcome::Up, dec!(0.60), dec!(30)))),
            resolved(Outcome::Up),
        ];

        // Live fold, capturing the effect stream.
        let mut live = InventoryManager::new();
        let live_effects: Vec<InventoryEffect> =
            events.iter().flat_map(|e| live.on_event(e)).collect();

        // Rebuild from the same (journaled) events.
        let rebuilt = InventoryManager::rebuild(events.iter());
        assert_eq!(rebuilt, live, "rebuilt state must equal the live manager");

        // Two independent folds yield identical effect streams.
        let mut again = InventoryManager::new();
        let again_effects: Vec<InventoryEffect> =
            events.iter().flat_map(|e| again.on_event(e)).collect();
        assert_eq!(again_effects, live_effects, "effect streams must match");

        // The window settled with the expected realized PnL.
        let settled = live_effects
            .iter()
            .find_map(|e| match e {
                InventoryEffect::Settled(s) => Some(s.clone()),
                InventoryEffect::Snapshot(_) => None,
            })
            .expect("a settlement effect");
        // cash_flow = -40 - 44 + 18 = -66; +70 winning Up shares → +4.
        assert_eq!(settled.realized_pnl, Dollars::new(dec!(4)));
    }

    // ---- 14. conservation shadow-oracle -----------------------------------

    /// xorshift64* — the codebase's deterministic test RNG idiom.
    struct Rng {
        state: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// An independent ledger built only from the operation inputs + the §8
    /// formulas — never reads the book. Tracks only what the cash identity needs
    /// (share counts for clamping, cash, fees, merged pairs).
    #[derive(Default)]
    struct Oracle {
        up: Decimal,
        down: Decimal,
        cash: Decimal,
        fees: Decimal,
        merged: Decimal,
    }
    impl Oracle {
        fn buy(&mut self, outcome: Outcome, price: Decimal, size: Decimal, fee: Decimal) {
            match outcome {
                Outcome::Up => self.up += size,
                Outcome::Down => self.down += size,
            }
            self.cash -= price * size;
            self.cash -= fee;
            self.fees += fee;
        }
        fn sell(&mut self, outcome: Outcome, price: Decimal, size: Decimal, fee: Decimal) {
            let held = match outcome {
                Outcome::Up => self.up,
                Outcome::Down => self.down,
            };
            let sold = size.min(held);
            match outcome {
                Outcome::Up => self.up -= sold,
                Outcome::Down => self.down -= sold,
            }
            self.cash += price * sold;
            self.cash -= fee;
            self.fees += fee;
        }
        fn merge(&mut self, req: Decimal) {
            let n = req.min(self.up.min(self.down)).max(Decimal::ZERO);
            self.up -= n;
            self.down -= n;
            self.cash += n;
            self.merged += n;
        }
        fn realized(&self, winner: Outcome) -> Decimal {
            let win = match winner {
                Outcome::Up => self.up,
                Outcome::Down => self.down,
            };
            self.cash + win
        }
    }

    #[test]
    fn random_sequences_conserve_cash_and_realized() {
        for seed in 0..8u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut book = WindowInventory::new(window());
            let mut oracle = Oracle::default();

            for _ in 0..300 {
                let op = rng.below(4);
                let outcome = if rng.below(2) == 0 {
                    Outcome::Up
                } else {
                    Outcome::Down
                };
                // Price on the 0.01 grid, in [0.01, 0.99].
                let p = px(Decimal::from(rng.below(99) + 1) / Decimal::from(100));
                let pd = p.as_decimal();
                let taker = rng.below(2) == 0;

                match op {
                    // BUY.
                    0 | 1 => {
                        let size = Decimal::from(rng.below(20) + 1);
                        let fee = if taker {
                            core_types::taker_fee(sz(size), dec!(0.07), p).as_decimal()
                        } else {
                            Decimal::ZERO
                        };
                        book.apply_fill(&fill(outcome, Side::Buy, pd, size, fee));
                        oracle.buy(outcome, pd, size, fee);
                    }
                    // SELL — bounded by current holdings so the clamp is exercised
                    // only by its dedicated test, not silently here.
                    2 => {
                        let held = match outcome {
                            Outcome::Up => book.up().shares.as_decimal(),
                            Outcome::Down => book.down().shares.as_decimal(),
                        };
                        if held > Decimal::ZERO {
                            let max = u64::try_from(held.trunc()).unwrap_or(1).max(1);
                            let size = Decimal::from(rng.below(max) + 1).min(held);
                            let fee = if taker {
                                core_types::taker_fee(sz(size), dec!(0.07), p).as_decimal()
                            } else {
                                Decimal::ZERO
                            };
                            book.apply_fill(&fill(outcome, Side::Sell, pd, size, fee));
                            oracle.sell(outcome, pd, size, fee);
                        }
                    }
                    // MERGE.
                    _ => {
                        let req = Decimal::from(rng.below(20) + 1);
                        book.apply_merge(sz(req));
                        oracle.merge(req);
                    }
                }

                // The book must match the oracle on every tracked accumulator.
                assert_eq!(book.up().shares.as_decimal(), oracle.up, "seed {seed}: up");
                assert_eq!(
                    book.down().shares.as_decimal(),
                    oracle.down,
                    "seed {seed}: down"
                );
                assert_eq!(
                    book.cash_flow().as_decimal(),
                    oracle.cash,
                    "seed {seed}: cash"
                );
                assert_eq!(
                    book.fees_paid().as_decimal(),
                    oracle.fees,
                    "seed {seed}: fees"
                );
                assert_eq!(
                    book.merged_pairs().as_decimal(),
                    oracle.merged,
                    "seed {seed}: merged"
                );
            }

            // Settlement: realized PnL = cash_flow + winning-share payout.
            let winner = if rng.below(2) == 0 {
                Outcome::Up
            } else {
                Outcome::Down
            };
            let summary = book.settle(winner, TimestampMs::from_millis(CLOSE_MS));
            assert_eq!(
                summary.realized_pnl.as_decimal(),
                oracle.realized(winner),
                "seed {seed}: realized PnL"
            );
        }
    }

    // ---- InventoryParams default mirrors config ---------------------------

    #[test]
    fn inventory_params_default_mirrors_config_defaults() {
        let p = InventoryParams::default();
        assert_eq!(p.pair_cost_threshold, dec!(0.98));
        assert_eq!(p.soft_cap_excess, sz(dec!(50)));
        assert_eq!(p.hard_cap_excess, sz(dec!(100)));
        assert_eq!(p.merge_min_pairs, sz(dec!(25)));
        assert_eq!(p.max_worst_case_loss, Dollars::new(dec!(25)));
    }
}
