//! Sans-IO state machine for one window connection (one token pair). No
//! clock, no IO, no logging — inputs are events and `now`, outputs are
//! values (skew-monitor / feed-machine precedent), so every behavior is
//! deterministically testable.
//!
//! Integrity design (Decisions Log 2026-06-12): venue `book` snapshots are
//! ground truth and **replace** local state wholesale — trades only ever
//! reach the book through them (`price_change` covers order place/cancel
//! only). Connection recycling is reserved for what a snapshot cannot fix:
//! a crossed book after a delta, derived-top vs venue-top divergence beyond
//! a grace period, and active-window book silence. Health is latched per
//! window: one `Unreliable` per episode, one `Recovered` once both tokens
//! re-establish trusted snapshots.

use std::sync::Arc;

use core_types::{
    BookHealth, BookUnreliableReason, ConditionId, Decimal, DurationMs, Event, MarketInfo,
    MarketLifecycleEvent, Outcome, Price, Size, TickSize, TimestampMs, TokenId, WindowLifecycle,
};

use crate::book::BookState;
use crate::wire::{BookMsg, ClobEvent, PriceChangeMsg};

/// How long the derived top may disagree with the venue-reported top before
/// the connection is recycled. A venue-behavior heuristic (same class as
/// feed-util's starvation multiple), so a code constant, not config.
const TOP_DIVERGENCE_GRACE: DurationMs = DurationMs::from_millis(2_000);

/// Cap on the adaptive staleness threshold: `multiple × configured`.
/// Quiet-market episodes double the effective threshold up to this; the
/// first genuine delta resets it.
const QUIET_STALE_CAP_MULTIPLE: i64 = 6;

/// Machine policy knobs (from [`crate::ClobParams`] at the driver boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineParams {
    /// Active-window book silence past this is desync evidence.
    pub book_stale_after: DurationMs,
    /// Coalescing interval for delta-driven full-book publication
    /// (top-of-book always publishes immediately on change).
    pub publish_interval: DurationMs,
}

/// Window phase as relevant to this machine: staleness only ever fires for
/// an active window (quiet pre-open/post-close books are legitimate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnPhase {
    /// Before the window opens (subscribed early for gap-free rollover).
    PreOpen,
    /// Open or closing — book flow is expected.
    Active,
    /// Past close — lingering for `market_resolved` only.
    Closed,
}

impl ConnPhase {
    /// Maps a scheduler lifecycle announcement onto the machine's phase.
    #[must_use]
    pub const fn from_lifecycle(lifecycle: WindowLifecycle) -> Self {
        match lifecycle {
            WindowLifecycle::Discovered => Self::PreOpen,
            WindowLifecycle::Open | WindowLifecycle::Closing => Self::Active,
            WindowLifecycle::Closed | WindowLifecycle::Resolved { .. } => Self::Closed,
        }
    }
}

/// One machine output, executed by the driver in order.
#[derive(Debug, Clone, PartialEq)]
pub enum MachineOutput {
    /// Publish a bus event.
    Publish(Event),
    /// Forward a market lifecycle notice (supervisor dedups, then the
    /// scheduler consumes).
    Market(MarketLifecycleEvent),
    /// Tear the connection down and reconnect (fresh subscribe snapshots
    /// rebuild the books). Emitted once per episode.
    Recycle(BookUnreliableReason),
    /// Telemetry: the derived book disagreed with an arriving venue
    /// snapshot with no intervening trade (logged by the driver, counted in
    /// status; the snapshot itself already repaired the state).
    Drift {
        /// Token whose derived book drifted.
        token: Arc<TokenId>,
    },
}

/// Health latch: `NeverTrusted` exists so the very first trust transition
/// (boot) does not emit a `Recovered` with nothing to recover from, and
/// `announced` pairs every `Recovered` with exactly one prior `Unreliable`
/// (an episode that started before trust was ever announced stays silent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Health {
    NeverTrusted,
    Trusted,
    Unreliable {
        since: TimestampMs,
        reason: BookUnreliableReason,
        announced: bool,
    },
}

/// Per-token book state and publication bookkeeping.
#[derive(Debug, Clone)]
struct TokenState {
    token: Arc<TokenId>,
    outcome: Outcome,
    book: BookState,
    /// A venue snapshot arrived on the current connection (publication and
    /// integrity checks are gated on it — delta-only state is not a book).
    have_snapshot: bool,
    /// Last book/price_change/best_bid_ask receive time (staleness anchor;
    /// reset on connect).
    last_data_at: TimestampMs,
    /// Deltas applied since the last full-book publication.
    dirty: bool,
    last_book_publish: TimestampMs,
    /// Venue time of the newest applied delta (stamps coalesced snapshots).
    last_delta_ts: TimestampMs,
    /// Last published top (price+size), for on-change-only publication.
    published_top: Option<(Option<core_types::BookLevel>, Option<core_types::BookLevel>)>,
    /// Newest venue-reported top since the last snapshot (integrity input;
    /// cleared by snapshots, which supersede it).
    venue_top: Option<(Option<Decimal>, Option<Decimal>)>,
    /// When the derived top started disagreeing with `venue_top`.
    divergence_since: Option<TimestampMs>,
    /// A trade printed since the last snapshot — drift comparison is
    /// meaningless until the next snapshot (trades reach books only through
    /// snapshots).
    trade_since_snapshot: bool,
    last_trade: Option<Decimal>,
    drift: u64,
    anomalies: u64,
    /// Opposite-side levels removed by implied consumption (trades at the
    /// touch arriving as crossing deltas).
    consumed: u64,
}

impl TokenState {
    fn new(token: TokenId, outcome: Outcome, now: TimestampMs) -> Self {
        Self {
            token: Arc::new(token),
            outcome,
            book: BookState::new(),
            have_snapshot: false,
            last_data_at: now,
            dirty: false,
            last_book_publish: TimestampMs::from_millis(i64::MIN),
            last_delta_ts: now,
            published_top: None,
            venue_top: None,
            divergence_since: None,
            trade_since_snapshot: false,
            last_trade: None,
            drift: 0,
            anomalies: 0,
            consumed: 0,
        }
    }

    /// Publishes the validated top-of-book if it changed since the last
    /// publication. A top that fails validation counts as an anomaly (an
    /// invalid level slipped past the apply filter) — never a panic.
    fn publish_top_if_changed(&mut self, ts: TimestampMs, out: &mut Vec<MachineOutput>) {
        if !self.have_snapshot {
            return;
        }
        let Ok(top) = self.book.top_of_book(ts) else {
            self.anomalies += 1;
            return;
        };
        let key = (top.bid, top.ask);
        if self.published_top.as_ref() == Some(&key) {
            return;
        }
        self.published_top = Some(key);
        out.push(MachineOutput::Publish(Event::TopOfBook {
            token_id: Arc::clone(&self.token),
            top,
        }));
    }

    /// Builds and publishes the full book snapshot. `seq_hash` only for
    /// venue-snapshot-driven publications; coalesced delta publications
    /// carry `None`.
    fn publish_book(
        &mut self,
        condition: &ConditionId,
        ts: TimestampMs,
        now: TimestampMs,
        seq_hash: Option<String>,
        out: &mut Vec<MachineOutput>,
    ) {
        match self.book.to_snapshot(&self.token, condition, ts, seq_hash) {
            Ok(snapshot) => {
                self.last_book_publish = now;
                self.dirty = false;
                out.push(MachineOutput::Publish(Event::Book(Arc::new(snapshot))));
            }
            Err(_) => self.anomalies += 1,
        }
    }

    /// Derived-top vs venue-top comparison (prices only — the venue reports
    /// no sizes). `None` when no venue top is on file.
    fn tops_agree(&self) -> Option<bool> {
        let (venue_bid, venue_ask) = self.venue_top.as_ref()?;
        let derived_bid = self.book.best_bid().map(|(p, _)| p);
        let derived_ask = self.book.best_ask().map(|(p, _)| p);
        Some(derived_bid == *venue_bid && derived_ask == *venue_ask)
    }
}

/// Point-in-time status of one token's book (drives the ladder CLI and the
/// supervisor status watch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenStatus {
    /// Which outcome this token is.
    pub outcome: Outcome,
    /// A venue snapshot arrived on the current connection.
    pub have_snapshot: bool,
    /// (bid levels, ask levels).
    pub depth: (usize, usize),
    /// Best bid (price, size).
    pub best_bid: Option<(Decimal, Decimal)>,
    /// Best ask (price, size).
    pub best_ask: Option<(Decimal, Decimal)>,
    /// Last trade print price.
    pub last_trade: Option<Decimal>,
    /// Age of the newest book data (local receive time).
    pub data_age: DurationMs,
}

/// Point-in-time machine status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineStatus {
    /// Books currently trusted (latched health).
    pub trusted: bool,
    /// Window phase.
    pub phase: ConnPhase,
    /// Tick size currently in force (initial from discovery, live-updated
    /// by `tick_size_change`).
    pub tick_size: TickSize,
    /// Per-token book status, `[Up, Down]`.
    pub tokens: [TokenStatus; 2],
    /// Snapshot-vs-derived mismatches with no intervening trade.
    pub drift: u64,
    /// Dropped/unusable inputs (bad levels, unvalidatable tops).
    pub anomalies: u64,
    /// Connection recycles this machine has requested.
    pub recycles: u64,
    /// Levels removed by implied consumption (trades at the touch).
    pub consumed: u64,
}

/// The per-window-connection state machine. See the module docs.
#[derive(Debug, Clone)]
pub struct ConnMachine {
    market: Arc<MarketInfo>,
    condition: Arc<ConditionId>,
    params: MachineParams,
    phase: ConnPhase,
    connected: bool,
    health: Health,
    tokens: [TokenState; 2],
    tick_size: TickSize,
    /// Adaptive staleness threshold (quiet-market protection).
    effective_stale_after: DurationMs,
    /// Books captured when a staleness recycle was requested, for the
    /// quiet-vs-broken comparison after reconnect.
    pre_recycle_books: Option<[BookState; 2]>,
    recycles: u64,
}

impl ConnMachine {
    /// New machine for one window. `lifecycle` is the window's current
    /// phase as known by the supervisor at spawn time.
    #[must_use]
    pub fn new(
        market: Arc<MarketInfo>,
        params: MachineParams,
        lifecycle: WindowLifecycle,
        now: TimestampMs,
    ) -> Self {
        let condition = Arc::new(market.condition_id.clone());
        let up = TokenState::new(market.tokens.up.clone(), Outcome::Up, now);
        let down = TokenState::new(market.tokens.down.clone(), Outcome::Down, now);
        let tick_size = market.tick_size;
        Self {
            market,
            condition,
            params,
            phase: ConnPhase::from_lifecycle(lifecycle),
            connected: false,
            health: Health::NeverTrusted,
            tokens: [up, down],
            tick_size,
            effective_stale_after: params.book_stale_after,
            pre_recycle_books: None,
            recycles: 0,
        }
    }

    /// The window's market.
    #[must_use]
    pub fn market(&self) -> &Arc<MarketInfo> {
        &self.market
    }

    /// A connection (or reconnection) is up and the subscribe was sent.
    /// Books keep their last contents for display continuity but are
    /// untrusted until fresh venue snapshots arrive.
    pub fn on_connected(&mut self, now: TimestampMs) {
        self.connected = true;
        for token in &mut self.tokens {
            token.have_snapshot = false;
            token.last_data_at = now;
            token.dirty = false;
            token.venue_top = None;
            token.divergence_since = None;
        }
    }

    /// The connection dropped (peer close, socket error, or a recycle this
    /// machine requested). Latches `Unreliable{Disconnected}` unless an
    /// earlier unreliable episode is already open (its `since`/reason are
    /// kept so the eventual `Recovered` reports the full outage).
    pub fn on_disconnected(&mut self, now: TimestampMs, out: &mut Vec<MachineOutput>) {
        self.connected = false;
        self.latch_unreliable(BookUnreliableReason::Disconnected, now, out);
    }

    /// Window lifecycle progressed (from the supervisor).
    pub fn on_lifecycle(&mut self, lifecycle: WindowLifecycle) {
        self.phase = ConnPhase::from_lifecycle(lifecycle);
    }

    /// One parsed wire event.
    pub fn on_event(&mut self, event: ClobEvent, now: TimestampMs, out: &mut Vec<MachineOutput>) {
        match event {
            ClobEvent::Book(msg) => self.on_book(msg, now, out),
            ClobEvent::PriceChange(msg) => self.on_price_change(&msg, now, out),
            ClobEvent::BestBidAsk {
                token,
                best_bid,
                best_ask,
                ..
            } => {
                let Some(index) = self.token_index(&token) else {
                    return;
                };
                self.tokens[index].last_data_at = now;
                // Same None-ambiguity rule as the price_change tops.
                if best_bid.is_some() || best_ask.is_some() {
                    self.tokens[index].venue_top = Some((best_bid, best_ask));
                }
                self.check_divergence(index, now, out);
            }
            ClobEvent::TickSizeChange { new_tick, ts, .. } => {
                self.tick_size = new_tick;
                out.push(MachineOutput::Publish(Event::TickSizeChange {
                    condition_id: Arc::clone(&self.condition),
                    new_tick,
                    ts,
                }));
            }
            ClobEvent::LastTrade {
                token,
                price,
                size,
                side,
                ts,
            } => {
                let Some(index) = self.token_index(&token) else {
                    return;
                };
                // A trade can reshape BOTH books (cross-book mint matching),
                // and its effects arrive only via the next snapshots — drift
                // comparison is suspended for both until then.
                for token_state in &mut self.tokens {
                    token_state.trade_since_snapshot = true;
                }
                let (Ok(price), Ok(size)) =
                    (Price::on_grid(price, TickSize::T00001), Size::new(size))
                else {
                    self.tokens[index].anomalies += 1;
                    return;
                };
                self.tokens[index].last_trade = Some(price.as_decimal());
                out.push(MachineOutput::Publish(Event::LastTrade {
                    token_id: Arc::clone(&self.tokens[index].token),
                    price,
                    size,
                    side,
                    ts,
                }));
            }
            ClobEvent::NewMarket {
                condition,
                slug,
                ts,
            } => {
                out.push(MachineOutput::Market(MarketLifecycleEvent::NewMarket {
                    condition_id: condition,
                    slug,
                    ts,
                }));
            }
            ClobEvent::MarketResolved {
                condition,
                winning_token,
                ts,
            } => {
                out.push(MachineOutput::Market(
                    MarketLifecycleEvent::MarketResolved {
                        condition_id: condition,
                        winning_token,
                        ts,
                    },
                ));
            }
        }
    }

    /// Periodic scan (driver calls at ~1 s): staleness watchdog, lingering
    /// divergence, and the coalesced full-book flush.
    pub fn on_tick(&mut self, now: TimestampMs, out: &mut Vec<MachineOutput>) {
        if !self.connected {
            return;
        }
        // Staleness: active windows only; one recycle per episode.
        if self.phase == ConnPhase::Active && !matches!(self.health, Health::Unreliable { .. }) {
            let stalest = self
                .tokens
                .iter()
                .map(|t| now.signed_duration_since(t.last_data_at))
                .max()
                .unwrap_or(DurationMs::ZERO);
            if stalest >= self.effective_stale_after {
                self.pre_recycle_books =
                    Some([self.tokens[0].book.clone(), self.tokens[1].book.clone()]);
                self.latch_unreliable(BookUnreliableReason::Stale, now, out);
                self.request_recycle(BookUnreliableReason::Stale, out);
            }
        }
        // Divergence that latched and then went quiet still times out here.
        for index in 0..self.tokens.len() {
            self.check_divergence(index, now, out);
        }
        // Coalesced flush of delta-dirtied books.
        for index in 0..self.tokens.len() {
            let due = now.signed_duration_since(self.tokens[index].last_book_publish)
                >= self.params.publish_interval;
            if self.tokens[index].dirty && self.tokens[index].have_snapshot && due {
                let ts = self.tokens[index].last_delta_ts;
                self.tokens[index].publish_book(&self.condition, ts, now, None, out);
            }
        }
    }

    /// Point-in-time status.
    #[must_use]
    pub fn status(&self, now: TimestampMs) -> MachineStatus {
        let token_status = |t: &TokenState| TokenStatus {
            outcome: t.outcome,
            have_snapshot: t.have_snapshot,
            depth: t.book.depth(),
            best_bid: t.book.best_bid(),
            best_ask: t.book.best_ask(),
            last_trade: t.last_trade,
            data_age: now.signed_duration_since(t.last_data_at),
        };
        MachineStatus {
            trusted: self.health == Health::Trusted,
            phase: self.phase,
            tick_size: self.tick_size,
            tokens: [token_status(&self.tokens[0]), token_status(&self.tokens[1])],
            drift: self.tokens.iter().map(|t| t.drift).sum(),
            anomalies: self.tokens.iter().map(|t| t.anomalies).sum(),
            recycles: self.recycles,
            consumed: self.tokens.iter().map(|t| t.consumed).sum(),
        }
    }

    fn token_index(&self, token: &TokenId) -> Option<usize> {
        self.tokens.iter().position(|t| t.token.as_ref() == token)
    }

    fn on_book(&mut self, msg: BookMsg, now: TimestampMs, out: &mut Vec<MachineOutput>) {
        let Some(index) = self.token_index(&msg.token) else {
            return;
        };
        // Drift audit: only meaningful when the previous snapshot was
        // delta-maintained with no intervening trade.
        let had_snapshot = self.tokens[index].have_snapshot;
        let audit = had_snapshot && !self.tokens[index].trade_since_snapshot;
        let mut incoming = BookState::new();
        let rejected = incoming.apply_snapshot(&msg.bids, &msg.asks);
        if rejected > 0 {
            self.tokens[index].anomalies += rejected as u64;
        }
        if audit && self.tokens[index].book != incoming {
            self.tokens[index].drift += 1;
            out.push(MachineOutput::Drift {
                token: Arc::clone(&self.tokens[index].token),
            });
        }
        // A crossed venue snapshot is venue-side garbage (deltas can no
        // longer cross our books): withhold trust until a clean one arrives.
        // No recycle — snapshots keep coming on this connection.
        if incoming.is_crossed() {
            self.latch_unreliable(BookUnreliableReason::Crossed, now, out);
        }
        // Replace wholesale: the snapshot is ground truth and supersedes
        // every prior delta and venue-top report.
        self.tokens[index].book = incoming;
        self.tokens[index].have_snapshot = true;
        self.tokens[index].last_data_at = now;
        self.tokens[index].trade_since_snapshot = false;
        self.tokens[index].venue_top = None;
        self.tokens[index].divergence_since = None;
        self.tokens[index].publish_book(&self.condition, msg.ts, now, msg.hash.clone(), out);
        self.tokens[index].publish_top_if_changed(msg.ts, out);
        self.maybe_recover(now, out);
    }

    fn on_price_change(
        &mut self,
        msg: &PriceChangeMsg,
        now: TimestampMs,
        out: &mut Vec<MachineOutput>,
    ) {
        let mut touched = [false; 2];
        for change in &msg.changes {
            let Some(index) = self.token_index(&change.token) else {
                continue;
            };
            touched[index] = true;
            let token = &mut self.tokens[index];
            token.last_data_at = now;
            match token
                .book
                .apply_change(change.side, change.price, change.size)
            {
                crate::book::ChangeOutcome::Rejected => {
                    token.anomalies += 1;
                    continue;
                }
                crate::book::ChangeOutcome::Applied { consumed } => {
                    // Implied consumption = a trade at the touch (the
                    // snapshot that follows confirms) — telemetry only.
                    token.consumed += consumed as u64;
                }
            }
            token.dirty = true;
            token.last_delta_ts = msg.ts;
            // The lenient parse maps "0"/""/absent all to None, so a change
            // with NO top report is indistinguishable from "both sides
            // empty" — only a report with at least one side counts as
            // divergence input (a genuinely empty book is harmless to miss).
            if change.best_bid.is_some() || change.best_ask.is_some() {
                token.venue_top = Some((change.best_bid, change.best_ask));
            }
        }
        // A genuine delta proves the market is not quiet — restore the
        // configured staleness threshold.
        if touched.iter().any(|&t| t) {
            self.effective_stale_after = self.params.book_stale_after;
        }
        for (index, was_touched) in touched.into_iter().enumerate() {
            if !was_touched {
                continue;
            }
            self.check_divergence(index, now, out);
            self.tokens[index].publish_top_if_changed(msg.ts, out);
            // Coalesced full-book publication on the event path (the 1 s
            // tick alone would make `publish_interval` < 1000 meaningless).
            let due = now.signed_duration_since(self.tokens[index].last_book_publish)
                >= self.params.publish_interval;
            if self.tokens[index].dirty && self.tokens[index].have_snapshot && due {
                let ts = self.tokens[index].last_delta_ts;
                self.tokens[index].publish_book(&self.condition, ts, now, None, out);
            }
        }
    }

    /// Divergence latch: starts timing when derived and venue tops disagree,
    /// recycles once the grace expires, clears as soon as they agree again.
    fn check_divergence(&mut self, index: usize, now: TimestampMs, out: &mut Vec<MachineOutput>) {
        if !self.tokens[index].have_snapshot {
            return;
        }
        match self.tokens[index].tops_agree() {
            Some(true) | None => {
                self.tokens[index].divergence_since = None;
            }
            Some(false) => {
                let since = *self.tokens[index].divergence_since.get_or_insert(now);
                if now.signed_duration_since(since) >= TOP_DIVERGENCE_GRACE
                    && !matches!(self.health, Health::Unreliable { .. })
                {
                    self.latch_unreliable(BookUnreliableReason::TopDivergence, now, out);
                    self.request_recycle(BookUnreliableReason::TopDivergence, out);
                }
            }
        }
    }

    fn latch_unreliable(
        &mut self,
        reason: BookUnreliableReason,
        now: TimestampMs,
        out: &mut Vec<MachineOutput>,
    ) {
        match self.health {
            // Already in an episode: keep the original since/reason.
            Health::Unreliable { .. } => {}
            // Trust was never announced — the episode stays silent too.
            Health::NeverTrusted => {
                self.health = Health::Unreliable {
                    since: now,
                    reason,
                    announced: false,
                };
            }
            Health::Trusted => {
                self.health = Health::Unreliable {
                    since: now,
                    reason,
                    announced: true,
                };
                out.push(MachineOutput::Publish(Event::BookHealth(
                    BookHealth::Unreliable {
                        window: self.market.window,
                        reason,
                        ts: now,
                    },
                )));
            }
        }
    }

    fn request_recycle(&mut self, reason: BookUnreliableReason, out: &mut Vec<MachineOutput>) {
        self.recycles += 1;
        out.push(MachineOutput::Recycle(reason));
    }

    /// Trust is re-established once the connection is up, both tokens hold
    /// fresh venue snapshots, and neither book is crossed.
    fn maybe_recover(&mut self, now: TimestampMs, out: &mut Vec<MachineOutput>) {
        let ready = self.connected
            && self
                .tokens
                .iter()
                .all(|t| t.have_snapshot && !t.book.is_crossed());
        if !ready {
            return;
        }
        match self.health {
            Health::Trusted => {}
            // First trust (boot): silent — there is nothing to recover from.
            Health::NeverTrusted => {
                self.health = Health::Trusted;
            }
            Health::Unreliable {
                since,
                reason,
                announced,
            } => {
                // Quiet-vs-broken: a staleness recycle whose fresh snapshots
                // are identical to the pre-recycle books was a quiet market,
                // not an outage — back off the threshold (capped); anything
                // else resets it.
                if let Some(prior) = self.pre_recycle_books.take() {
                    let unchanged = reason == BookUnreliableReason::Stale
                        && prior[0] == self.tokens[0].book
                        && prior[1] == self.tokens[1].book;
                    if unchanged {
                        let cap = self
                            .params
                            .book_stale_after
                            .as_millis()
                            .saturating_mul(QUIET_STALE_CAP_MULTIPLE);
                        let doubled = self.effective_stale_after.as_millis().saturating_mul(2);
                        self.effective_stale_after = DurationMs::from_millis(doubled.min(cap));
                    } else {
                        self.effective_stale_after = self.params.book_stale_after;
                    }
                }
                self.health = Health::Trusted;
                if announced {
                    out.push(MachineOutput::Publish(Event::BookHealth(
                        BookHealth::Recovered {
                            window: self.market.window,
                            outage: now.signed_duration_since(since),
                            ts: now,
                        },
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::{
        Asset, FeeParams, ResolutionSource, Series, Side, TokenPair, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    use super::*;
    use crate::wire::{BookMsg, LevelChange, PriceChangeMsg};

    const T0: TimestampMs = TimestampMs::from_millis(1_700_000_000_000);

    fn at(offset_ms: i64) -> TimestampMs {
        T0.saturating_add(DurationMs::from_millis(offset_ms))
    }

    fn market() -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: T0,
            },
            event_slug: "btc-updown-5m-1700000000".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "ab".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("111").unwrap(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: at(300_000),
            strike: Some(dec!(104000)),
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

    fn params() -> MachineParams {
        MachineParams {
            book_stale_after: DurationMs::from_millis(10_000),
            publish_interval: DurationMs::from_millis(200),
        }
    }

    fn snapshot_event(token: &str, bid: Decimal, ask: Decimal) -> ClobEvent {
        ClobEvent::Book(BookMsg {
            token: TokenId::new(token).unwrap(),
            condition: market().condition_id.clone(),
            bids: vec![(bid, dec!(100))],
            asks: vec![(ask, dec!(100))],
            ts: T0,
            hash: Some("h".to_owned()),
        })
    }

    fn change_event(
        token: &str,
        side: Side,
        price: Decimal,
        size: Decimal,
        tops: Option<(Decimal, Decimal)>,
        ts: TimestampMs,
    ) -> ClobEvent {
        ClobEvent::PriceChange(PriceChangeMsg {
            ts,
            changes: vec![LevelChange {
                token: TokenId::new(token).unwrap(),
                price,
                size,
                side,
                best_bid: tops.map(|(b, _)| b),
                best_ask: tops.map(|(_, a)| a),
            }],
        })
    }

    /// Machine connected with both tokens snapshotted (silently trusted).
    fn trusted_machine(now: TimestampMs) -> ConnMachine {
        let mut machine = ConnMachine::new(market(), params(), WindowLifecycle::Open, now);
        machine.on_connected(now);
        let mut out = Vec::new();
        machine.on_event(snapshot_event("111", dec!(0.48), dec!(0.52)), now, &mut out);
        machine.on_event(snapshot_event("222", dec!(0.47), dec!(0.53)), now, &mut out);
        assert!(machine.status(now).trusted);
        assert!(
            !out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::BookHealth(_)))),
            "boot trust must be silent"
        );
        machine
    }

    fn health_events(out: &[MachineOutput]) -> Vec<BookHealth> {
        out.iter()
            .filter_map(|o| match o {
                MachineOutput::Publish(Event::BookHealth(h)) => Some(*h),
                _ => None,
            })
            .collect()
    }

    fn recycles(out: &[MachineOutput]) -> Vec<BookUnreliableReason> {
        out.iter()
            .filter_map(|o| match o {
                MachineOutput::Recycle(r) => Some(*r),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn snapshot_publishes_book_and_top_immediately() {
        let mut machine = ConnMachine::new(market(), params(), WindowLifecycle::Open, T0);
        machine.on_connected(T0);
        let mut out = Vec::new();
        machine.on_event(snapshot_event("111", dec!(0.48), dec!(0.52)), T0, &mut out);
        let books: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                MachineOutput::Publish(Event::Book(b)) => Some(Arc::clone(b)),
                _ => None,
            })
            .collect();
        assert_eq!(books.len(), 1);
        assert_eq!(books[0].token_id.as_str(), "111");
        assert_eq!(books[0].seq_hash.as_deref(), Some("h"));
        assert_eq!(books[0].bids[0].price.as_decimal(), dec!(0.48));
        assert!(
            out.iter().any(|o| matches!(
                o,
                MachineOutput::Publish(Event::TopOfBook { token_id, .. }) if token_id.as_str() == "111"
            )),
            "top must publish with the first snapshot"
        );
    }

    #[test]
    fn top_publishes_only_on_real_change() {
        let mut machine = trusted_machine(T0);
        // Deep level change: book dirty, but the top is untouched.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.40), dec!(30), None, at(10)),
            at(10),
            &mut out,
        );
        assert!(
            !out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::TopOfBook { .. }))),
            "deep change must not publish a top"
        );
        // Top-level change publishes.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.49), dec!(10), None, at(20)),
            at(20),
            &mut out,
        );
        assert!(
            out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::TopOfBook { .. })))
        );
    }

    #[test]
    fn delta_books_coalesce_but_snapshots_bypass() {
        let mut machine = trusted_machine(T0);
        // First delta right after the snapshot publish: inside the 200 ms
        // window — no full-book publication.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.40), dec!(30), None, at(50)),
            at(50),
            &mut out,
        );
        assert!(
            !out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::Book(_)))),
            "coalescer must hold the first delta publication"
        );
        // A second delta past the interval flushes.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.41), dec!(30), None, at(300)),
            at(300),
            &mut out,
        );
        let books: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                MachineOutput::Publish(Event::Book(b)) => Some(Arc::clone(b)),
                _ => None,
            })
            .collect();
        assert_eq!(books.len(), 1, "past the interval the dirty book flushes");
        assert!(
            books[0].seq_hash.is_none(),
            "delta-built snapshots carry no hash"
        );
        // A venue snapshot publishes immediately regardless of the interval.
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(310),
            &mut out,
        );
        assert!(
            out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::Book(_)))),
            "venue snapshots bypass the coalescer"
        );
        // The tick flushes a dirty book once the interval passes.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.42), dec!(30), None, at(320)),
            at(320),
            &mut out,
        );
        assert!(
            !out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::Book(_))))
        );
        let mut out = Vec::new();
        machine.on_tick(at(600), &mut out);
        assert!(
            out.iter()
                .any(|o| matches!(o, MachineOutput::Publish(Event::Book(_)))),
            "the periodic tick flushes dirty books"
        );
    }

    #[test]
    fn drift_counts_only_without_intervening_trade() {
        let mut machine = trusted_machine(T0);
        // Delta moves the book; the next snapshot agrees with the venue but
        // not with our derived state (simulated divergence) — drift.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.49), dec!(10), None, at(10)),
            at(10),
            &mut out,
        );
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(20),
            &mut out,
        );
        assert!(
            out.iter().any(|o| matches!(o, MachineOutput::Drift { .. })),
            "delta-vs-snapshot mismatch with no trade is drift"
        );
        assert_eq!(machine.status(at(20)).drift, 1);
        // Now a trade intervenes: the following mismatching snapshot is the
        // trade's expected effect, not drift.
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::LastTrade {
                token: TokenId::new("111").unwrap(),
                price: dec!(0.52),
                size: dec!(10),
                side: Side::Buy,
                ts: at(30),
            },
            at(30),
            &mut out,
        );
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.46), dec!(0.54)),
            at(40),
            &mut out,
        );
        assert!(
            !out.iter().any(|o| matches!(o, MachineOutput::Drift { .. })),
            "post-trade snapshots are expected to differ"
        );
        assert_eq!(machine.status(at(40)).drift, 1);
    }

    #[test]
    fn matching_snapshot_after_deltas_is_not_drift() {
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.40), dec!(30), None, at(10)),
            at(10),
            &mut out,
        );
        // Snapshot exactly matching the derived state: no drift.
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::Book(BookMsg {
                token: TokenId::new("111").unwrap(),
                condition: market().condition_id.clone(),
                bids: vec![(dec!(0.48), dec!(100)), (dec!(0.40), dec!(30))],
                asks: vec![(dec!(0.52), dec!(100))],
                ts: at(20),
                hash: None,
            }),
            at(20),
            &mut out,
        );
        assert!(!out.iter().any(|o| matches!(o, MachineOutput::Drift { .. })));
        assert_eq!(machine.status(at(20)).drift, 0);
    }

    #[test]
    fn trade_at_touch_uncrosses_without_recycle_or_health_event() {
        // Live-verified venue behavior: a trade at the touch arrives as a
        // crossing delta (the consumed level's removal only comes in the
        // next snapshot). The machine resolves it locally — no recycle, no
        // health transition (recycling here looped every trade in the first
        // live run).
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        // A bid arriving AT the Up book's ask consumed that ask.
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.52), dec!(5), None, at(10)),
            at(10),
            &mut out,
        );
        assert!(recycles(&out).is_empty(), "no recycle on a trade at touch");
        assert!(health_events(&out).is_empty(), "no health transition");
        let status = machine.status(at(10));
        assert!(status.trusted);
        assert_eq!(status.consumed, 1, "the consumed ask is telemetry");
        assert_eq!(status.tokens[0].best_bid, Some((dec!(0.52), dec!(5))));
        assert_eq!(status.tokens[0].best_ask, None, "single ask was consumed");
        // The published top reflects the post-trade book.
        assert!(out.iter().any(|o| matches!(
            o,
            MachineOutput::Publish(Event::TopOfBook { token_id, top })
                if token_id.as_str() == "111"
                    && top.ask.is_none()
                    && top.bid.is_some_and(|b| b.price.as_decimal() == dec!(0.52))
        )));
    }

    #[test]
    fn crossed_venue_snapshot_latches_until_a_clean_one() {
        let mut machine = trusted_machine(T0);
        // A crossed snapshot from the venue: trust withheld, no recycle
        // (snapshots keep coming on this connection).
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::Book(BookMsg {
                token: TokenId::new("111").unwrap(),
                condition: market().condition_id.clone(),
                bids: vec![(dec!(0.60), dec!(5))],
                asks: vec![(dec!(0.50), dec!(5))],
                ts: at(10),
                hash: None,
            }),
            at(10),
            &mut out,
        );
        assert!(recycles(&out).is_empty(), "no recycle for venue garbage");
        assert_eq!(
            health_events(&out),
            vec![BookHealth::Unreliable {
                window: market().window,
                reason: BookUnreliableReason::Crossed,
                ts: at(10),
            }]
        );
        assert!(!machine.status(at(10)).trusted);
        // The next clean snapshot recovers.
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(500),
            &mut out,
        );
        assert_eq!(
            health_events(&out),
            vec![BookHealth::Recovered {
                window: market().window,
                outage: DurationMs::from_millis(490),
                ts: at(500),
            }]
        );
        assert!(machine.status(at(500)).trusted);
    }

    #[test]
    fn disconnect_latches_once_and_recovers_with_full_outage() {
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        machine.on_disconnected(at(10), &mut out);
        assert_eq!(
            health_events(&out),
            vec![BookHealth::Unreliable {
                window: market().window,
                reason: BookUnreliableReason::Disconnected,
                ts: at(10),
            }]
        );
        // A second disconnect in the same episode stays silent.
        let mut out = Vec::new();
        machine.on_disconnected(at(500), &mut out);
        assert!(
            health_events(&out).is_empty(),
            "already latched — no repeat"
        );
        machine.on_connected(at(1_000));
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(1_100),
            &mut out,
        );
        machine.on_event(
            snapshot_event("222", dec!(0.47), dec!(0.53)),
            at(1_200),
            &mut out,
        );
        assert_eq!(
            health_events(&out),
            vec![BookHealth::Recovered {
                window: market().window,
                outage: DurationMs::from_millis(1_190),
                ts: at(1_200),
            }]
        );
    }

    #[test]
    fn divergence_recycles_only_after_grace() {
        let mut machine = trusted_machine(T0);
        // Venue says the top is elsewhere — divergence starts timing.
        let mut out = Vec::new();
        machine.on_event(
            change_event(
                "111",
                Side::Buy,
                dec!(0.40),
                dec!(5),
                Some((dec!(0.45), dec!(0.55))),
                at(10),
            ),
            at(10),
            &mut out,
        );
        assert!(recycles(&out).is_empty(), "inside the grace: no recycle");
        // Still diverged within the grace.
        let mut out = Vec::new();
        machine.on_tick(at(1_500), &mut out);
        assert!(recycles(&out).is_empty());
        // Grace expires.
        let mut out = Vec::new();
        machine.on_tick(at(2_100), &mut out);
        assert_eq!(recycles(&out), vec![BookUnreliableReason::TopDivergence]);
        assert_eq!(
            health_events(&out),
            vec![BookHealth::Unreliable {
                window: market().window,
                reason: BookUnreliableReason::TopDivergence,
                ts: at(2_100),
            }]
        );
    }

    #[test]
    fn divergence_clears_when_tops_agree_again() {
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        machine.on_event(
            change_event(
                "111",
                Side::Buy,
                dec!(0.40),
                dec!(5),
                Some((dec!(0.45), dec!(0.55))),
                at(10),
            ),
            at(10),
            &mut out,
        );
        // The venue top comes back into agreement before the grace expires.
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::BestBidAsk {
                token: TokenId::new("111").unwrap(),
                best_bid: Some(dec!(0.48)),
                best_ask: Some(dec!(0.52)),
                ts: at(500),
            },
            at(500),
            &mut out,
        );
        let mut out = Vec::new();
        machine.on_tick(at(5_000), &mut out);
        assert!(recycles(&out).is_empty(), "agreement cleared the latch");
        assert!(machine.status(at(5_000)).trusted);
    }

    #[test]
    fn staleness_fires_only_for_active_windows() {
        // Pre-open: quiet books are legitimate forever.
        let mut machine = ConnMachine::new(market(), params(), WindowLifecycle::Discovered, T0);
        machine.on_connected(T0);
        let mut out = Vec::new();
        machine.on_event(snapshot_event("111", dec!(0.48), dec!(0.52)), T0, &mut out);
        machine.on_event(snapshot_event("222", dec!(0.47), dec!(0.53)), T0, &mut out);
        let mut out = Vec::new();
        machine.on_tick(at(60_000), &mut out);
        assert!(
            recycles(&out).is_empty(),
            "pre-open silence is not staleness"
        );
        // The window opens: silence now counts, anchored at the last data.
        machine.on_lifecycle(WindowLifecycle::Open);
        let mut out = Vec::new();
        machine.on_tick(at(61_000), &mut out);
        assert_eq!(recycles(&out), vec![BookUnreliableReason::Stale]);
        // Closed windows are exempt again.
        let mut closed = trusted_machine(T0);
        closed.on_lifecycle(WindowLifecycle::Closed);
        let mut out = Vec::new();
        closed.on_tick(at(600_000), &mut out);
        assert!(
            recycles(&out).is_empty(),
            "post-close silence is not staleness"
        );
    }

    #[test]
    fn quiet_market_doubles_threshold_and_delta_resets_it() {
        let mut machine = trusted_machine(T0);
        // Stale at 10 s.
        let mut out = Vec::new();
        machine.on_tick(at(10_000), &mut out);
        assert_eq!(recycles(&out), vec![BookUnreliableReason::Stale]);
        // Reconnect; identical snapshots prove the market was merely quiet.
        let mut out = Vec::new();
        machine.on_disconnected(at(10_100), &mut out);
        machine.on_connected(at(10_200));
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(10_300),
            &mut out,
        );
        machine.on_event(
            snapshot_event("222", dec!(0.47), dec!(0.53)),
            at(10_300),
            &mut out,
        );
        assert_eq!(health_events(&out).len(), 1, "recovered");
        // The effective threshold doubled to 20 s: quiet at +10 s is fine…
        let mut out = Vec::new();
        machine.on_tick(at(20_300), &mut out);
        assert!(recycles(&out).is_empty(), "doubled threshold holds");
        // …and only trips at +20 s.
        let mut out = Vec::new();
        machine.on_tick(at(30_300), &mut out);
        assert_eq!(recycles(&out), vec![BookUnreliableReason::Stale]);
        // Recover again; this time a genuine delta resets the threshold.
        let mut out = Vec::new();
        machine.on_disconnected(at(30_400), &mut out);
        machine.on_connected(at(30_500));
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(30_600),
            &mut out,
        );
        machine.on_event(
            snapshot_event("222", dec!(0.47), dec!(0.53)),
            at(30_600),
            &mut out,
        );
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Buy, dec!(0.40), dec!(5), None, at(31_000)),
            at(31_000),
            &mut out,
        );
        // Back to the configured 10 s, anchored at the delta.
        let mut out = Vec::new();
        machine.on_tick(at(40_000), &mut out);
        assert!(recycles(&out).is_empty());
        let mut out = Vec::new();
        machine.on_tick(at(41_100), &mut out);
        assert_eq!(recycles(&out), vec![BookUnreliableReason::Stale]);
    }

    #[test]
    fn tick_flip_publishes_and_books_continue() {
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::TickSizeChange {
                token: TokenId::new("111").unwrap(),
                condition: market().condition_id.clone(),
                old_tick: Some(TickSize::T001),
                new_tick: TickSize::T0001,
                ts: at(10),
            },
            at(10),
            &mut out,
        );
        assert!(out.iter().any(|o| matches!(
            o,
            MachineOutput::Publish(Event::TickSizeChange {
                new_tick: TickSize::T0001,
                ..
            })
        )));
        assert_eq!(machine.status(at(10)).tick_size, TickSize::T0001);
        // 3-dp levels apply cleanly after the flip.
        let mut out = Vec::new();
        machine.on_event(
            change_event("111", Side::Sell, dec!(0.965), dec!(5), None, at(20)),
            at(20),
            &mut out,
        );
        assert_eq!(machine.status(at(20)).anomalies, 0);
    }

    #[test]
    fn market_lifecycle_events_surface_for_dedup() {
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::NewMarket {
                condition: ConditionId::new(format!("0x{}", "cd".repeat(32))).unwrap(),
                slug: "btc-updown-5m-next".to_owned(),
                ts: at(10),
            },
            at(10),
            &mut out,
        );
        machine.on_event(
            ClobEvent::MarketResolved {
                condition: market().condition_id.clone(),
                winning_token: TokenId::new("111").unwrap(),
                ts: at(20),
            },
            at(20),
            &mut out,
        );
        let market_events: Vec<_> = out
            .iter()
            .filter_map(|o| match o {
                MachineOutput::Market(ev) => Some(ev.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(market_events.len(), 2);
        assert!(matches!(
            &market_events[0],
            MarketLifecycleEvent::NewMarket { slug, .. } if slug == "btc-updown-5m-next"
        ));
        assert!(matches!(
            &market_events[1],
            MarketLifecycleEvent::MarketResolved { winning_token, .. }
                if winning_token.as_str() == "111"
        ));
    }

    #[test]
    fn last_trade_publishes_immediately_and_unknown_tokens_are_skipped() {
        let mut machine = trusted_machine(T0);
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::LastTrade {
                token: TokenId::new("999").unwrap(),
                price: dec!(0.5),
                size: dec!(1),
                side: Side::Buy,
                ts: at(10),
            },
            at(10),
            &mut out,
        );
        assert!(out.is_empty(), "unknown token: silently skipped");
        let mut out = Vec::new();
        machine.on_event(
            ClobEvent::LastTrade {
                token: TokenId::new("111").unwrap(),
                price: dec!(0.55),
                size: dec!(30),
                side: Side::Sell,
                ts: at(20),
            },
            at(20),
            &mut out,
        );
        assert!(out.iter().any(|o| matches!(
            o,
            MachineOutput::Publish(Event::LastTrade {
                side: Side::Sell,
                ..
            })
        )));
        assert_eq!(
            machine.status(at(20)).tokens[0].last_trade,
            Some(dec!(0.55))
        );
    }

    #[test]
    fn disconnect_before_any_trust_stays_silent_and_recovery_too() {
        let mut machine = ConnMachine::new(market(), params(), WindowLifecycle::Open, T0);
        machine.on_connected(T0);
        let mut out = Vec::new();
        machine.on_disconnected(at(100), &mut out);
        assert!(health_events(&out).is_empty(), "never trusted — no event");
        machine.on_connected(at(200));
        let mut out = Vec::new();
        machine.on_event(
            snapshot_event("111", dec!(0.48), dec!(0.52)),
            at(300),
            &mut out,
        );
        machine.on_event(
            snapshot_event("222", dec!(0.47), dec!(0.53)),
            at(300),
            &mut out,
        );
        assert!(
            health_events(&out).is_empty(),
            "an episode that started before trust was announced ends silently"
        );
        assert!(machine.status(at(300)).trusted);
    }
}
