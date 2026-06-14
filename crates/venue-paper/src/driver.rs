//! [`PaperVenue`]: the [`VenuePort`]/[`VenueEvents`] implementation over the
//! sans-IO [`MatchEngine`].
//!
//! Mirrors `venue-live`'s `LiveVenue`: the engine lives behind an
//! `Arc<Mutex<…>>` (std `Mutex`, never held across an `await`), command methods
//! lock → mutate → collect effects → unlock → `event_tx.send().await`. The one
//! addition over live is a spawned **timer task** that drives the engine's
//! simulated-latency deadlines (`on_clock`), woken either by its computed sleep
//! or by a [`Notify`] whenever a command/market-data event may have changed the
//! schedule.
//!
//! Market data is pushed in via [`PaperVenue::on_bus_event`] (the orchestrator
//! forwards the relevant `core_types::Event`s — book, top, trade, tick, window).
//! There is **no arming gate** — paper is always constructible (§9).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use core_types::{ConditionId, Dollars, Event, NewOrder, OrderId, Size, TimestampMs, WindowId};
use tokio::sync::{Notify, mpsc, watch};
use venue_api::{
    Accepted, BatchPlaced, CancelReport, PlaceRejection, VenueError, VenueEvent, VenueEvents,
    VenuePort, Wallet,
};

use crate::engine::{MatchEngine, PaperEffect};
use crate::params::PaperParams;
use crate::wallet::PaperLedgerSnapshot;

/// Idle sleep when nothing is scheduled — the timer is re-armed by the
/// [`Notify`] the moment a deadline appears, so this is only a safety heartbeat.
const IDLE_SLEEP: Duration = Duration::from_secs(3600);

/// The paper execution adapter.
pub struct PaperVenue {
    engine: Arc<Mutex<MatchEngine>>,
    event_tx: mpsc::Sender<VenueEvent>,
    event_rx: Option<mpsc::Receiver<VenueEvent>>,
    wake: Arc<Notify>,
    now_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync>,
    shutdown_tx: watch::Sender<bool>,
}

impl PaperVenue {
    /// Builds a paper venue and spawns its latency timer task. `now_fn` supplies
    /// wall-clock millis (production: `timeutil::wall_now`; tests: a paused-time
    /// closure).
    #[must_use]
    pub fn spawn<F>(params: PaperParams, now_fn: F) -> Self
    where
        F: Fn() -> TimestampMs + Send + Sync + 'static,
    {
        let now_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync> = Arc::new(now_fn);
        let (event_tx, event_rx) = mpsc::channel(params.event_channel_capacity.max(1));
        let engine = Arc::new(Mutex::new(MatchEngine::new(params)));
        let wake = Arc::new(Notify::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(timer_loop(
            Arc::clone(&engine),
            event_tx.clone(),
            Arc::clone(&wake),
            Arc::clone(&now_fn),
            shutdown_rx,
        ));

        Self {
            engine,
            event_tx,
            event_rx: Some(event_rx),
            wake,
            now_fn,
            shutdown_tx,
        }
    }

    /// Feeds one market-data event into the simulator (book/top/trade/tick/
    /// window). The orchestrator forwards the relevant bus events here.
    pub async fn on_bus_event(&self, event: &Event) {
        match event {
            Event::Book(snap) => self.apply(|e, _| ((), e.on_book(snap))).await,
            Event::TopOfBook { token_id, top } => {
                self.apply(|e, _| ((), e.on_top(token_id, top))).await;
            }
            Event::LastTrade {
                token_id,
                price,
                size,
                side,
                ts,
            } => {
                self.apply(|e, now| ((), e.on_trade(token_id, *price, *size, *side, *ts, now)))
                    .await;
            }
            Event::TickSizeChange {
                condition_id,
                new_tick,
                ts,
            } => {
                self.apply(|e, _| ((), e.on_tick_size(condition_id, *new_tick, *ts)))
                    .await;
            }
            Event::Window { market, lifecycle } => {
                self.apply(|e, now| ((), e.on_window(market, *lifecycle, now)))
                    .await;
            }
            _ => {}
        }
    }

    /// Signals the timer task to stop (also happens on drop).
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    // ---- runtime ledger commands (sync, `set_markets` idiom) ---------------

    /// Sets paper capital to `amount` at runtime (the dashboard capital-adjust
    /// seam, §9). Locks → mutates; emits no `VenueEvent` (the change shows in
    /// `balances()`/`ledger_snapshot()`) and changes no deadline, so the timer
    /// is not nudged. Journaling (a `ControlEvent::PaperCapitalSet` on the bus)
    /// is the caller's job — `venue-paper` holds no bus sender.
    pub fn set_capital(&self, amount: Dollars) {
        self.lock().set_capital(amount);
    }

    /// Adjusts paper capital by a signed `delta` at runtime (dashboard seam).
    pub fn adjust_capital(&self, delta: Dollars) {
        self.lock().adjust_capital(delta);
    }

    /// The richer ledger snapshot (income lines + signed positions) for the
    /// dashboard / `bot paper-sim`.
    #[must_use]
    pub fn ledger_snapshot(&self) -> PaperLedgerSnapshot {
        self.lock().ledger_snapshot()
    }

    /// Merges up to `pairs` matched Up/Down pairs for `window` into collateral
    /// (instant simulated CTF merge, §8), returning the number merged. Errors if
    /// the window's metadata is not yet known. Async for `apply`-closure
    /// symmetry though it schedules nothing.
    pub async fn merge(&self, window: WindowId, pairs: Size) -> Result<Size, VenueError> {
        self.apply(|engine, now| engine.merge(window, pairs, now))
            .await
    }

    /// Runs a sans-IO engine mutation under the lock, then publishes its effects
    /// and nudges the timer. The lock is never held across the publish `await`.
    async fn apply<T, M>(&self, mutate: M) -> T
    where
        M: FnOnce(&mut MatchEngine, TimestampMs) -> (T, Vec<PaperEffect>),
    {
        let now = (self.now_fn)();
        let (out, effects) = {
            let mut engine = self.lock();
            mutate(&mut engine, now)
        };
        self.publish(effects).await;
        // Deadlines may have changed (a placement/cancel was scheduled).
        self.wake.notify_one();
        out
    }

    async fn publish(&self, effects: Vec<PaperEffect>) {
        for effect in effects {
            let event = match effect {
                PaperEffect::Order(update) => VenueEvent::Order(Arc::new(update)),
                PaperEffect::Fill(fill) => VenueEvent::Fill(Arc::new(fill)),
            };
            if self.event_tx.send(event).await.is_err() {
                break; // receiver dropped — stop publishing.
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MatchEngine> {
        self.engine.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for PaperVenue {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
    }
}

impl VenuePort for PaperVenue {
    async fn place(&self, order: &NewOrder) -> Result<Accepted, VenueError> {
        self.apply(|engine, now| engine.place(order, now)).await
    }

    async fn place_batch(&self, orders: &[NewOrder]) -> Result<BatchPlaced, VenueError> {
        let mut results = Vec::with_capacity(orders.len());
        for order in orders {
            match self.place(order).await {
                Ok(accepted) => results.push(Ok(accepted)),
                // Paper only rejects structurally (per-order) — map to a
                // per-order rejection, never sink the whole batch.
                Err(VenueError::Rejected(reason)) => {
                    let raw = format!("{reason:?}");
                    results.push(Err(PlaceRejection {
                        client_id: order.client_id.clone(),
                        reason,
                        raw,
                    }));
                }
                Err(other) => return Err(other),
            }
        }
        Ok(BatchPlaced { results })
    }

    async fn cancel(&self, id: &OrderId) -> Result<CancelReport, VenueError> {
        Ok(self.apply(|engine, now| engine.cancel(id, now)).await)
    }

    async fn cancel_market(&self, market: &ConditionId) -> Result<CancelReport, VenueError> {
        Ok(self
            .apply(|engine, now| engine.cancel_market(market, now))
            .await)
    }

    async fn cancel_all(&self) -> Result<CancelReport, VenueError> {
        Ok(self.apply(|engine, now| engine.cancel_all(now)).await)
    }

    async fn balances(&self) -> Result<Wallet, VenueError> {
        Ok(self.lock().balances())
    }
}

impl VenueEvents for PaperVenue {
    fn take_event_rx(&mut self) -> Option<mpsc::Receiver<VenueEvent>> {
        self.event_rx.take()
    }
}

/// The latency timer task: sleeps until the next deadline (or the idle
/// heartbeat), re-armed early by [`Notify`], and runs `on_clock` to fire due
/// activations/cancels/expiries.
async fn timer_loop(
    engine: Arc<Mutex<MatchEngine>>,
    event_tx: mpsc::Sender<VenueEvent>,
    wake: Arc<Notify>,
    now_fn: Arc<dyn Fn() -> TimestampMs + Send + Sync>,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        let sleep = {
            let engine = engine.lock().unwrap_or_else(|e| e.into_inner());
            match engine.next_deadline() {
                Some(deadline) => {
                    let now = now_fn();
                    let delta = deadline.as_millis().saturating_sub(now.as_millis()).max(0);
                    Duration::from_millis(u64::try_from(delta).unwrap_or(0))
                }
                None => IDLE_SLEEP,
            }
        };

        tokio::select! {
            () = tokio::time::sleep(sleep) => {}
            () = wake.notified() => {}
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
        }
        if *shutdown.borrow() {
            break;
        }

        let now = now_fn();
        let effects = {
            let mut engine = engine.lock().unwrap_or_else(|e| e.into_inner());
            engine.on_clock(now)
        };
        for effect in effects {
            let event = match effect {
                PaperEffect::Order(update) => VenueEvent::Order(Arc::new(update)),
                PaperEffect::Fill(fill) => VenueEvent::Fill(Arc::new(fill)),
            };
            if event_tx.send(event).await.is_err() {
                return; // receiver gone — nothing left to drive.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use core_types::{
        Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, Event, FeeParams, Liquidity,
        MarketInfo, NewOrder, Outcome, Price, ResolutionSource, RoundDir, Series, Side, Size,
        TickSize, TimeInForce, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId,
        WindowLifecycle,
    };
    use rust_decimal::dec;
    use venue_api::{VenueEvent, VenueEvents, VenuePort};

    use super::*;
    use crate::params::{LatencySpec, PaperParams};

    const BASE_MS: i64 = 1_781_000_000_000;

    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(BASE_MS),
        }
    }
    fn market() -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: token(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: TimestampMs::from_millis(BASE_MS + 300_000),
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
    fn book_event(bids: &[(Decimal, Decimal)], asks: &[(Decimal, Decimal)], ts: i64) -> Event {
        Event::Book(Arc::new(BookSnapshot {
            token_id: token(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
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
            ts: TimestampMs::from_millis(ts),
            seq_hash: None,
        }))
    }
    fn buy_gtc() -> NewOrder {
        NewOrder {
            client_id: Some("c0".to_owned()),
            window: window(),
            token_id: token(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: px(dec!(0.49)),
            qty: core_types::OrderQty::Shares(sz(dec!(20))),
            tif: TimeInForce::Gtc { post_only: true },
        }
    }
    fn params() -> PaperParams {
        PaperParams {
            placement: LatencySpec {
                mean_ms: core_types::DurationMs::from_millis(150),
                jitter_ms: core_types::DurationMs::ZERO,
            },
            cancel: LatencySpec {
                mean_ms: core_types::DurationMs::from_millis(150),
                jitter_ms: core_types::DurationMs::ZERO,
            },
            rng_seed: Some(1),
            ..PaperParams::default()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn place_activates_then_fills_over_the_driver() {
        let start = tokio::time::Instant::now();
        let now_fn = move || {
            let elapsed = i64::try_from(start.elapsed().as_millis()).unwrap_or(0);
            TimestampMs::from_millis(BASE_MS + elapsed)
        };
        let mut venue = PaperVenue::spawn(params(), now_fn);
        let mut rx = venue.take_event_rx().expect("event rx");

        // Register the market + a book with 30 displayed at our price (0.49).
        venue
            .on_bus_event(&Event::Window {
                market: market(),
                lifecycle: WindowLifecycle::Open,
            })
            .await;
        venue
            .on_bus_event(&book_event(
                &[(dec!(0.49), dec!(30))],
                &[(dec!(0.51), dec!(50))],
                BASE_MS,
            ))
            .await;

        // Place a post-only BUY → accepted PendingNew, no event yet.
        let accepted = venue.place(&buy_gtc()).await.expect("accepted");
        assert_eq!(accepted.state, core_types::OrderState::PendingNew);
        assert_eq!(accepted.client_id.as_deref(), Some("c0"));

        // Advance past the 150ms placement latency → the timer activates it.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let open = rx.recv().await.expect("an order event");
        match open {
            VenueEvent::Order(u) => {
                assert_eq!(u.state, core_types::OrderState::Open);
                assert_eq!(u.order_id, accepted.order_id);
            }
            other => panic!("expected Open order update, got {other:?}"),
        }

        // A sell-aggressor print at our price: drains the 30 queue then fills 10.
        venue
            .on_bus_event(&Event::LastTrade {
                token_id: Arc::new(token()),
                price: px(dec!(0.49)),
                size: sz(dec!(40)),
                side: Side::Sell,
                ts: TimestampMs::from_millis(BASE_MS + 250),
            })
            .await;

        // Expect a Maker fill (fee 0) then an order update.
        let fill = rx.recv().await.expect("a fill");
        match fill {
            VenueEvent::Fill(f) => {
                assert_eq!(f.liquidity, Liquidity::Maker);
                assert_eq!(f.size, sz(dec!(10)));
                assert_eq!(f.fee, Dollars::ZERO);
                assert_eq!(f.price, px(dec!(0.49)));
            }
            other => panic!("expected a fill, got {other:?}"),
        }
        let upd = rx.recv().await.expect("an order update");
        match upd {
            VenueEvent::Order(u) => {
                assert_eq!(u.state, core_types::OrderState::PartiallyFilled);
                assert_eq!(u.filled_size, sz(dec!(10)));
            }
            other => panic!("expected partial fill update, got {other:?}"),
        }

        // Balances reflect the buy: 10000 − 0.49×10 (maker, no fee) = 9995.10.
        let wallet = venue.balances().await.expect("balances");
        assert_eq!(wallet.collateral_total, Dollars::new(dec!(9995.10)));
        venue.shutdown();
    }
}
