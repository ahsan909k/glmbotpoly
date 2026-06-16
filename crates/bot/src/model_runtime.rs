//! A multi-asset fair-value runtime for `bot dashboard --with-model`.
//!
//! `bot fair`'s [`crate::fair`] state prices **one** series' current window and is
//! coupled to calibration journaling + post-hoc strike verification. The dashboard
//! runs up to six series across two assets and only needs the live `p_up` next to
//! the book mid — so this is a focused generalisation: per-**asset** vol / basis /
//! health (kept per asset, not per window, so a window rollover never resets the
//! health hysteresis), plus a [`FairValueEngine`] per window. It publishes
//! `Event::Model` and `Event::ModelHealth` onto the same bus the dashboard drains
//! (via `try_send` — the loop both produces and consumes, so an awaited send could
//! deadlock). No calibration, no verification, no rendering.
//!
//! The §8 model itself (strike capture, vol, basis, `p_up = Φ(z)`, the health
//! gate) lives in the `model` crate and is unit-tested there; this module only
//! routes ticks/windows to the right per-asset state and runs the per-asset health
//! monitor **once per heartbeat** (multiple open windows of one asset share its
//! health, so running it per window would double-advance the dwell timers).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Context;
use config::AppConfig;
use core_types::{
    Asset, Event, MarketInfo, ModelHealthEvent, PriceSource, Series, TickKind, TimestampMs,
    WindowId, WindowLifecycle,
};
use model::{
    BasisTracker, FairInputs, FairParams, FairValue, FairValueEngine, HealthMonitor,
    HealthTransition, VolEstimator,
};
use tokio::sync::mpsc;

use crate::fair::{basis_params, fair_params, health_params};
use crate::vol::vol_params;

/// Drop a window's engine/market state this long (ms) after its close.
const PRUNE_AFTER_CLOSE_MS: i64 = 200_000;

/// Per-asset model state: shared by all of that asset's open windows.
struct AssetModel {
    vol: VolEstimator,
    basis: BasisTracker,
    health: HealthMonitor,
    /// Health-tier transitions awaiting publication on the next heartbeat.
    pending_health: Vec<HealthTransition>,
}

/// The dashboard's multi-asset fair-value runtime.
pub(crate) struct ModelRuntime {
    assets: HashMap<Asset, AssetModel>,
    engines: HashMap<WindowId, FairValueEngine>,
    markets: HashMap<WindowId, Arc<MarketInfo>>,
    /// The current open window per series (set on `Open`).
    open: HashMap<Series, WindowId>,
    lifecycles: HashMap<WindowId, WindowLifecycle>,
    last_fair: HashMap<WindowId, FairValue>,
    fair_params: FairParams,
}

impl ModelRuntime {
    /// Builds one per-asset model for each distinct asset across `series`.
    pub(crate) fn new(config: &AppConfig, series: &[Series]) -> anyhow::Result<Self> {
        let engine_defaults = &config.engine.defaults;
        let mut assets = HashMap::new();
        for s in series {
            let asset = s.asset;
            if assets.contains_key(&asset) {
                continue;
            }
            let vol = VolEstimator::new(
                asset,
                vol_params(engine_defaults, PriceSource::BinanceDirect, TickKind::Mid),
            )
            .with_context(|| format!("building {asset} vol estimator"))?;
            let basis = BasisTracker::new(asset, basis_params(&config.feeds))
                .with_context(|| format!("building {asset} basis tracker"))?;
            let health = HealthMonitor::new(health_params(&config.feeds))
                .with_context(|| format!("building {asset} health monitor"))?;
            assets.insert(
                asset,
                AssetModel {
                    vol,
                    basis,
                    health,
                    pending_health: Vec::new(),
                },
            );
        }
        Ok(Self {
            assets,
            engines: HashMap::new(),
            markets: HashMap::new(),
            open: HashMap::new(),
            lifecycles: HashMap::new(),
            last_fair: HashMap::new(),
            fair_params: fair_params(&config.feeds),
        })
    }

    /// Routes a bus event: price ticks feed the matching asset's vol/basis and all
    /// of that asset's window engines; window announcements create engines (at
    /// Discovered, so the strike captures around open) and track the open window.
    pub(crate) fn on_event(&mut self, event: &Event) {
        match event {
            Event::PriceTick(tick) => {
                if let Some(am) = self.assets.get_mut(&tick.asset) {
                    am.vol.on_tick(tick);
                    am.basis.on_tick(tick);
                }
                for (win, engine) in &mut self.engines {
                    if win.series.asset == tick.asset {
                        let _ = engine.on_tick(tick);
                    }
                }
            }
            Event::Window { market, lifecycle } => {
                let win = market.window;
                self.markets.insert(win, Arc::clone(market));
                self.lifecycles.insert(win, *lifecycle);
                match lifecycle {
                    WindowLifecycle::Discovered => self.ensure_engine(market),
                    WindowLifecycle::Open => {
                        self.ensure_engine(market);
                        self.open.insert(win.series, win);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Recomputes every active window's fair value, runs each asset's health gate
    /// once, publishes `Event::Model` + `Event::ModelHealth`, and prunes.
    pub(crate) fn on_heartbeat(&mut self, bus_tx: &mpsc::Sender<Event>, now: TimestampMs) {
        self.recompute(now);
        self.publish(bus_tx);
        self.prune(now);
    }

    fn ensure_engine(&mut self, market: &Arc<MarketInfo>) {
        let win = market.window;
        if self.engines.contains_key(&win) {
            return;
        }
        match FairValueEngine::for_market(market, self.fair_params) {
            Ok(engine) => {
                self.engines.insert(win, engine);
            }
            Err(error) => tracing::warn!(
                target: "dashboard", window = %win, %error,
                "window cannot be priced by the model"
            ),
        }
    }

    fn recompute(&mut self, now: TimestampMs) {
        // Active = the current open window per series, while Open or Closing.
        let active: Vec<WindowId> = self
            .open
            .values()
            .copied()
            .filter(|w| {
                matches!(
                    self.lifecycles.get(w),
                    Some(WindowLifecycle::Open | WindowLifecycle::Closing)
                )
            })
            .collect();
        let mut fairs: HashMap<WindowId, FairValue> = HashMap::new();
        for win in &active {
            let Some(am) = self.assets.get(&win.series.asset) else {
                continue;
            };
            let sigma = am.vol.sigma();
            let basis = am.basis.basis_log();
            if let Some(engine) = self.engines.get_mut(win) {
                let fair = engine.compute(FairInputs {
                    now,
                    sigma_1s: sigma,
                    basis_log: basis,
                });
                fairs.insert(*win, fair);
            }
        }
        // Per-asset health: one update per asset, stamped onto its window fairs.
        let mut by_asset: HashMap<Asset, Vec<WindowId>> = HashMap::new();
        for win in fairs.keys() {
            by_asset.entry(win.series.asset).or_default().push(*win);
        }
        for (asset, wins) in by_asset {
            let Some(am) = self.assets.get_mut(&asset) else {
                continue;
            };
            // Prefer a window that produced a probability so a just-opened
            // (strike-warming) sibling can't drag the whole asset to Warming.
            let rep = wins
                .iter()
                .copied()
                .find(|w| fairs[w].snapshot().is_some())
                .unwrap_or(wins[0]);
            let (health, reason, transition) = am.health.update(fairs[&rep].health_inputs());
            if let Some(t) = transition {
                am.pending_health.push(t);
            }
            for win in &wins {
                if let Some(f) = fairs.get_mut(win) {
                    f.health = health;
                    f.reason = reason;
                }
            }
        }
        self.last_fair = fairs;
    }

    fn publish(&mut self, bus_tx: &mpsc::Sender<Event>) {
        for fair in self.last_fair.values() {
            if let Some(snapshot) = fair.snapshot() {
                let _ = bus_tx.try_send(Event::Model(snapshot));
            }
        }
        // Latched health transitions (rare); re-queue on a full channel.
        for (asset, am) in &mut self.assets {
            while let Some(&t) = am.pending_health.first() {
                let event = Event::ModelHealth(ModelHealthEvent {
                    asset: *asset,
                    health: t.to,
                    reason: t.reason,
                    ts: t.at,
                });
                if bus_tx.try_send(event).is_ok() {
                    am.pending_health.remove(0);
                } else {
                    break;
                }
            }
        }
    }

    fn prune(&mut self, now: TimestampMs) {
        let keep: HashSet<WindowId> = self.open.values().copied().collect();
        let stale: Vec<WindowId> = self
            .markets
            .values()
            .filter(|m| {
                !keep.contains(&m.window)
                    && now.as_millis() - m.close_time.as_millis() >= PRUNE_AFTER_CLOSE_MS
            })
            .map(|m| m.window)
            .collect();
        for w in stale {
            self.engines.remove(&w);
            self.markets.remove(&w);
            self.lifecycles.remove(&w);
            self.last_fair.remove(&w);
        }
    }
}

#[cfg(test)]
impl ModelRuntime {
    pub(crate) fn asset_count(&self) -> usize {
        self.assets.len()
    }

    pub(crate) fn engine_count(&self) -> usize {
        self.engines.len()
    }

    pub(crate) fn current_window(&self, series: Series) -> Option<WindowId> {
        self.open.get(&series).copied()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use core_types::{
        ConditionId, FeeParams, ResolutionSource, Size, TickSize, TokenId, TokenPair,
        WindowDuration,
    };
    use rust_decimal::dec;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn series(asset: Asset, duration: WindowDuration) -> Series {
        Series { asset, duration }
    }

    fn window(s: Series, open_ms: i64) -> WindowId {
        WindowId {
            series: s,
            open_time: ts(open_ms),
        }
    }

    fn market(s: Series, open_ms: i64) -> Arc<MarketInfo> {
        let win = window(s, open_ms);
        Arc::new(MarketInfo {
            window: win,
            event_slug: "test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: ts(open_ms + 300_000),
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

    fn config() -> AppConfig {
        AppConfig::default()
    }

    #[test]
    fn new_builds_one_model_per_asset() {
        let series_list = vec![
            series(Asset::Btc, WindowDuration::M5),
            series(Asset::Btc, WindowDuration::M15),
            series(Asset::Eth, WindowDuration::M5),
        ];
        let rt = ModelRuntime::new(&config(), &series_list).unwrap();
        assert_eq!(rt.asset_count(), 2); // BTC + ETH, not one per series
    }

    #[test]
    fn window_open_creates_engine_and_tracks_current() {
        let s = series(Asset::Btc, WindowDuration::M5);
        let mut rt = ModelRuntime::new(&config(), &[s]).unwrap();
        let m = market(s, 0);
        rt.on_event(&Event::Window {
            market: Arc::clone(&m),
            lifecycle: WindowLifecycle::Open,
        });
        assert_eq!(rt.engine_count(), 1);
        assert_eq!(rt.current_window(s), Some(window(s, 0)));
    }

    #[test]
    fn heartbeat_on_unwarmed_window_publishes_no_model_snapshot() {
        let s = series(Asset::Btc, WindowDuration::M5);
        let mut rt = ModelRuntime::new(&config(), &[s]).unwrap();
        rt.on_event(&Event::Window {
            market: market(s, 0),
            lifecycle: WindowLifecycle::Open,
        });
        let (tx, mut rx) = mpsc::channel(16);
        rt.on_heartbeat(&tx, ts(1_000));
        // No vol/strike yet ⇒ no probability ⇒ no fair snapshot published (a
        // health-tier transition may still be emitted — that's fine).
        let mut got_model = false;
        while let Ok(ev) = rx.try_recv() {
            if matches!(ev, Event::Model(_)) {
                got_model = true;
            }
        }
        assert!(!got_model, "no fair snapshot before warmup");
    }

    #[test]
    fn prune_drops_superseded_windows() {
        let s = series(Asset::Btc, WindowDuration::M5);
        let mut rt = ModelRuntime::new(&config(), &[s]).unwrap();
        // Window A opens, then its successor B opens (A is no longer current).
        rt.on_event(&Event::Window {
            market: market(s, 0),
            lifecycle: WindowLifecycle::Open,
        });
        rt.on_event(&Event::Window {
            market: market(s, 300_000),
            lifecycle: WindowLifecycle::Open,
        });
        assert_eq!(rt.engine_count(), 2);
        // Past A's close + prune horizon: A drops, B (current) stays.
        rt.on_heartbeat(&mpsc::channel(16).0, ts(300_000 + PRUNE_AFTER_CLOSE_MS + 1));
        assert_eq!(rt.engine_count(), 1);
        assert_eq!(rt.current_window(s), Some(window(s, 300_000)));
    }
}
