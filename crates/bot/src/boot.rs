//! Boot-time journal wiring (CLAUDE.md §3/§9 restart correctness).
//!
//! Two responsibilities, both at the §4 binary boundary (the `journal` and
//! `engine` crates never depend on `config`):
//! - [`journal_params`] maps the `[journal]` config section into the
//!   [`journal::RecorderParams`] the recorder consumes (mirrors
//!   `paper::paper_params` / `live::live_params`).
//! - [`rebuild_from_journal`] replays a recorded session and reconstructs the
//!   derived state the bot must restore on restart: per-window inventory (via
//!   the proven [`engine::InventoryManager::rebuild`] fold) and a last-known
//!   view of our working orders.
//!
//! The module is named `boot` rather than `journal` on purpose: a `mod journal`
//! in the `bot` binary would shadow the `journal` *crate* in path resolution.
//!
//! **Order-state rebuild — why a bespoke view.** `engine::quote_manager`'s
//! `RestingView` cannot be rebuilt from journaled events: its `apply_order_update`
//! ignores any order id it did not first see via `record_pending`, and
//! `record_pending` needs the `ClientId` (outcome/level) that exists only at
//! placement time and never travels the bus. So we keep a simple bot-local
//! [`RebuiltOrders`] folded purely from `Event::OrderUpdate`. On a **live**
//! restart this is the last-known view, then corrected against the venue's actual
//! open orders by `venue_live::OrderStore::reconcile`; for **paper** the journal
//! is the only source.
//!
//! No production run loop consumes [`RebuiltState`] yet (the orchestrator is
//! deferred); boot currently rebuilds and logs the summary, ready for the
//! orchestrator to seed the live `InventoryManager` / order store. The wiring
//! lives here so the binary owns the cross-crate (journal × engine) glue.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use config::AppConfig;
use core_types::{Event, OrderId, OrderUpdate};
use engine::InventoryManager;
use journal::{RecorderParams, ReplayReader};

/// Maps the `[journal]` config section into [`RecorderParams`] (§4 boundary
/// mapping; the `journal` crate never depends on `config`).
#[must_use]
pub fn journal_params(config: &AppConfig) -> RecorderParams {
    let j = &config.journal;
    RecorderParams {
        out_dir: j.dir.clone(),
        max_segment_bytes: j.max_segment_bytes,
        max_segment_age_ms: j.max_segment_age_ms,
        channel_capacity: j.channel_capacity,
        flush_interval_ms: j.flush_interval_ms,
        sqlite_path: Some(j.sqlite_path.clone()),
        fsync_on_flush: j.fsync_on_flush,
        retention_max_age_ms: j.retention_max_age_ms,
        retention_max_total_bytes: j.retention_max_total_bytes,
    }
}

/// A last-known view of our working (non-terminal) orders, folded from the
/// journaled `Event::OrderUpdate` stream. Last-write-wins by order id; a terminal
/// update drops the order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RebuiltOrders(pub HashMap<OrderId, OrderUpdate>);

impl RebuiltOrders {
    fn fold(&mut self, update: &OrderUpdate) {
        if update.state.is_terminal() {
            self.0.remove(&update.order_id);
        } else {
            self.0.insert(update.order_id.clone(), update.clone());
        }
    }

    /// Number of working orders in the rebuilt view.
    #[must_use]
    pub fn working(&self) -> usize {
        self.0.len()
    }
}

/// Derived state reconstructed from a recorded journal on boot.
#[derive(Debug, Default)]
pub struct RebuiltState {
    /// Per-window inventory, folded exactly as the live engine would (the
    /// §3/§9 restart-rebuild guarantee — see [`InventoryManager::rebuild`]).
    pub inventory: InventoryManager,
    /// Last-known working orders.
    pub orders: RebuiltOrders,
    /// Total events replayed.
    pub events: u64,
}

/// Replays the journal under `dir` and reconstructs [`RebuiltState`]. A missing
/// directory is treated as a first run (empty state). The fold is the same
/// event-time-driven one the live engine uses, so the rebuilt inventory equals
/// `InventoryManager::rebuild` over the same events.
///
/// # Errors
/// Returns an error if the directory exists but cannot be read, or a recorded
/// record cannot be decoded.
pub fn rebuild_from_journal(dir: &Path) -> anyhow::Result<RebuiltState> {
    if !dir.exists() {
        return Ok(RebuiltState::default());
    }
    let reader = ReplayReader::open(dir)
        .with_context(|| format!("opening journal segments in {}", dir.display()))?;
    let mut state = RebuiltState::default();
    for result in reader.events() {
        let event = result.context("replaying a journal event")?;
        state.events += 1;
        let _ = state.inventory.on_event(&event);
        if let Event::OrderUpdate(update) = &event {
            state.orders.fold(update);
        }
    }
    Ok(state)
}

/// Rebuilds from the configured journal directory and logs a one-line summary.
/// A rebuild failure (a corrupt or unreadable journal) is logged and degraded to
/// empty state rather than blocking boot — the bot starts fresh, and live mode
/// reconciles against the venue's actual open orders regardless.
#[must_use]
pub fn rebuild_and_log(config: &AppConfig) -> RebuiltState {
    let dir = &config.journal.dir;
    match rebuild_from_journal(dir) {
        Ok(state) => {
            tracing::info!(
                target: "journal",
                events = state.events,
                windows = state.inventory.len(),
                working_orders = state.orders.working(),
                dir = %dir.display(),
                "restored derived state from the journal on boot"
            );
            state
        }
        Err(error) => {
            tracing::warn!(
                target: "journal",
                %error,
                dir = %dir.display(),
                "could not rebuild state from the journal — starting fresh"
            );
            RebuiltState::default()
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use core_types::{
        Asset, ConditionId, Decimal, Dollars, FeeParams, Fill, Liquidity, MarketInfo, OrderState,
        Outcome, Price, ResolutionSource, RoundDir, Series, Side, Size, TickSize, TimestampMs,
        TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle,
    };
    use engine::InventoryManager;
    use journal::{Recorder, RecorderParams};
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
        Price::quantize(d, TickSize::T001, RoundDir::Down).expect("price")
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).expect("size")
    }
    fn market() -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).expect("cid"),
            tokens: TokenPair {
                up: TokenId::new("1").expect("up"),
                down: TokenId::new("2").expect("down"),
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
    fn order(id: &str, state: OrderState) -> Event {
        Event::OrderUpdate(Arc::new(OrderUpdate {
            order_id: OrderId::new(id).expect("oid"),
            window: window(),
            token_id: TokenId::new("1").expect("token"),
            side: Side::Buy,
            state,
            price: px(dec!(0.40)),
            original_size: sz(dec!(10)),
            filled_size: if state == OrderState::Filled {
                sz(dec!(10))
            } else {
                sz(dec!(0))
            },
            reject_reason: None,
            ts_venue: None,
            ts_local: TimestampMs::from_millis(OPEN_MS),
        }))
    }
    fn fill(outcome: Outcome, side: Side, price: Decimal, size: Decimal) -> Event {
        Event::Fill(Arc::new(Fill {
            order_id: OrderId::new("o1").expect("oid"),
            trade_id: Some("t1".to_owned()),
            window: window(),
            token_id: TokenId::new("1").expect("token"),
            outcome,
            side,
            price: px(price),
            size: sz(size),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: TimestampMs::from_millis(OPEN_MS + 100),
            ts_local: TimestampMs::from_millis(OPEN_MS + 100),
        }))
    }

    /// open, an order driven to Filled (terminal), a still-working order, some
    /// fills, then resolution.
    fn session() -> Vec<Event> {
        vec![
            Event::Window {
                market: market(),
                lifecycle: WindowLifecycle::Open,
            },
            order("o1", OrderState::Open),
            order("o1", OrderState::Filled),
            order("o2", OrderState::Open),
            fill(Outcome::Up, Side::Buy, dec!(0.40), dec!(100)),
            fill(Outcome::Down, Side::Buy, dec!(0.55), dec!(80)),
            Event::Window {
                market: market(),
                lifecycle: WindowLifecycle::Resolved {
                    outcome: Outcome::Up,
                },
            },
        ]
    }

    fn record_session(dir: &std::path::Path, events: &[Event]) {
        let _ = std::fs::remove_dir_all(dir);
        let params = RecorderParams {
            out_dir: dir.to_path_buf(),
            ..RecorderParams::default()
        };
        let recorder =
            Recorder::spawn(params, || TimestampMs::from_millis(OPEN_MS)).expect("spawn recorder");
        for event in events {
            recorder.record(event);
        }
        let stats = recorder.finish().expect("finish recorder");
        assert_eq!(stats.records, events.len() as u64);
    }

    #[test]
    fn rebuild_matches_the_live_fold_and_keeps_only_working_orders() {
        let dir = std::env::temp_dir().join(format!("boot-rebuild-{}", std::process::id()));
        let events = session();
        record_session(&dir, &events);

        let state = rebuild_from_journal(&dir).expect("rebuild");

        // Inventory equals the canonical fold over the same events.
        assert_eq!(state.inventory, InventoryManager::rebuild(events.iter()));
        assert_eq!(state.events, events.len() as u64);
        assert_eq!(state.inventory.len(), 1);

        // o1 went terminal (Filled) → dropped; o2 is still working.
        assert_eq!(state.orders.working(), 1);
        assert!(
            state
                .orders
                .0
                .contains_key(&OrderId::new("o2").expect("oid"))
        );
        assert!(
            !state
                .orders
                .0
                .contains_key(&OrderId::new("o1").expect("oid"))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_journal_dir_is_a_clean_first_run() {
        let dir = std::env::temp_dir().join(format!("boot-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = rebuild_from_journal(&dir).expect("rebuild");
        assert_eq!(state.events, 0);
        assert!(state.inventory.is_empty());
        assert_eq!(state.orders.working(), 0);
    }

    #[test]
    fn journal_params_maps_the_config_section() {
        let mut config = AppConfig::default();
        config.journal.dir = "data/jjj".into();
        config.journal.sqlite_path = "data/jjj.sqlite".into();
        config.journal.fsync_on_flush = true;
        config.journal.retention_max_age_ms = 86_400_000;
        let params = journal_params(&config);
        assert_eq!(params.out_dir, std::path::PathBuf::from("data/jjj"));
        assert_eq!(
            params.sqlite_path,
            Some(std::path::PathBuf::from("data/jjj.sqlite"))
        );
        assert!(params.fsync_on_flush);
        assert_eq!(params.retention_max_age_ms, 86_400_000);
        assert_eq!(params.max_segment_bytes, config.journal.max_segment_bytes);
    }
}
