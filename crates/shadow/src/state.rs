//! The sans-IO shadow core: routes bus events + depth ladders into per-asset
//! price/vol/basis + per-window strike + per-Up-token PM state, and samples the
//! 24-feature vector for an active window at a window-aligned instant.
//!
//! No clock reads, no IO — every input is a method argument, so replay and live
//! folding are identical (the model/engine-crate discipline). The async
//! [`crate::driver`] owns timing and the model inference.

use std::collections::HashMap;
use std::sync::Arc;

use core_types::{
    Asset, Event, MarketInfo, PriceSource, TickKind, TimestampMs, TokenId, WindowDuration,
    WindowId, WindowLifecycle,
};
use model::StrikeCapture;
use rust_decimal::prelude::ToPrimitive;

use crate::features::FeatureVector;
use crate::features::depth::{DepthLadder, N_DEPTH_FEATURES, frame_depth_features};
use crate::features::pm::{PmSample, PmState};
use crate::features::price::{PriceSample, PriceState};

/// Drop a window's state this long (ms) after close (the `model_runtime` grace).
const PRUNE_AFTER_CLOSE_MS: i64 = 200_000;

/// Per-asset latest depth features (last ladder, for the as-of read).
#[derive(Debug, Clone, Default)]
struct DepthState {
    last: Option<(i64, [f64; N_DEPTH_FEATURES])>,
}

/// Per-window state: metadata, strike capture, and its Up token.
struct WindowState {
    market: Arc<MarketInfo>,
    strike: StrikeCapture,
    up_token: TokenId,
    lifecycle: WindowLifecycle,
}

/// One sampled feature vector plus the journal diagnostics.
#[derive(Debug, Clone)]
pub struct SampleOutput {
    /// The assembled 24-feature vector (model input).
    pub features: FeatureVector,
    /// Frozen strike used for `log_s_k` (`None` if not yet captured).
    pub strike: Option<f64>,
    /// As-of second of the price bar (`None` if none).
    pub price_asof_sec: Option<i64>,
    /// `sigma_1s` folded-return count.
    pub sigma_returns: u64,
    /// `basis_ewma` folded-sample count.
    pub basis_samples: u32,
    /// As-of ms of the depth features (`None` if none / stale).
    pub depth_asof_ms: Option<i64>,
    /// As-of ms of the PM book (`None` if none).
    pub pm_asof_ms: Option<i64>,
    /// Seconds since the PM book run started (`None` if none).
    pub pm_run_secs: Option<i64>,
}

/// The sans-IO shadow feature engine.
pub struct ShadowCore {
    assets: HashMap<Asset, PriceState>,
    depth: HashMap<Asset, DepthState>,
    windows: HashMap<WindowId, WindowState>,
    pm: HashMap<TokenId, PmState>,
    max_staleness_ms: i64,
}

impl ShadowCore {
    /// Builds a core for the given `assets` (BTC/ETH for the four champion series).
    ///
    /// # Errors
    /// [`model::VolError`] if a per-asset vol estimator fails to construct
    /// (unreachable with the constant defaults).
    pub fn new(assets: &[Asset], max_staleness_ms: i64) -> Result<Self, model::VolError> {
        let mut price = HashMap::new();
        let mut depth = HashMap::new();
        for &a in assets {
            price.insert(a, PriceState::new(a)?);
            depth.insert(a, DepthState::default());
        }
        Ok(Self {
            assets: price,
            depth,
            windows: HashMap::new(),
            pm: HashMap::new(),
            max_staleness_ms,
        })
    }

    /// Routes one bus event into the feature state.
    pub fn on_event(&mut self, event: &Event) {
        match event {
            Event::PriceTick(tick) => {
                if let Some(price) = self.assets.get_mut(&tick.asset) {
                    match (tick.source, tick.kind) {
                        (PriceSource::BinanceDirect, TickKind::Mid) => price.on_mid_tick(tick),
                        (PriceSource::ChainlinkRtds, _) => price.on_chainlink_tick(tick),
                        _ => {}
                    }
                }
                // Feed Chainlink prints to the strike captures (they self-filter).
                if tick.source == PriceSource::ChainlinkRtds {
                    for ws in self.windows.values_mut() {
                        if ws.market.window.series.asset == tick.asset {
                            let _ = ws.strike.on_tick(tick);
                        }
                    }
                }
            }
            Event::TopOfBook { token_id, top } => {
                if let Some(pm) = self.pm.get_mut(token_id.as_ref())
                    && let (Some(bid), Some(ask)) = (top.bid, top.ask)
                {
                    pm.on_top(
                        top.ts,
                        bid.price.to_f64(),
                        bid.size.as_decimal().to_f64().unwrap_or(f64::NAN),
                        ask.price.to_f64(),
                        ask.size.as_decimal().to_f64().unwrap_or(f64::NAN),
                    );
                }
            }
            Event::Window { market, lifecycle } => self.on_window(market, *lifecycle),
            _ => {}
        }
    }

    /// Ingests one depth ladder from shadow's own Binance depth20 feed.
    pub fn on_depth(&mut self, ladder: &DepthLadder) {
        if let Some(ds) = self.depth.get_mut(&ladder.asset) {
            ds.last = Some((
                ladder.recv_ms.div_euclid(1_000),
                frame_depth_features(ladder),
            ));
        }
    }

    fn on_window(&mut self, market: &Arc<MarketInfo>, lifecycle: WindowLifecycle) {
        let win = market.window;
        match lifecycle {
            WindowLifecycle::Discovered | WindowLifecycle::Open | WindowLifecycle::Closing => {
                let asset = win.series.asset;
                // The champion covers the four BTC/ETH × 5m/15m series (the model's
                // tau range is [5, 900] s); 1h windows are out of distribution.
                if !matches!(
                    win.series.duration,
                    WindowDuration::M5 | WindowDuration::M15
                ) {
                    return;
                }
                if !self.assets.contains_key(&asset) {
                    return; // not a shadow asset — ignore
                }
                if let std::collections::hash_map::Entry::Vacant(e) = self.windows.entry(win) {
                    match StrikeCapture::for_market(market) {
                        Ok(strike) => {
                            let up = market.tokens.up.clone();
                            self.pm.entry(up.clone()).or_default();
                            e.insert(WindowState {
                                market: Arc::clone(market),
                                strike,
                                up_token: up,
                                lifecycle,
                            });
                        }
                        Err(error) => tracing::warn!(
                            target: "shadow", window = %win, %error,
                            "window cannot have its strike captured — not shadowed"
                        ),
                    }
                } else if let Some(ws) = self.windows.get_mut(&win) {
                    ws.lifecycle = lifecycle;
                }
            }
            WindowLifecycle::Closed | WindowLifecycle::Resolved { .. } => {
                if let Some(ws) = self.windows.get_mut(&win) {
                    ws.lifecycle = lifecycle;
                }
            }
        }
    }

    /// Drops windows (and their PM state) more than the grace past close.
    pub fn prune(&mut self, now: TimestampMs) {
        let drop: Vec<WindowId> = self
            .windows
            .values()
            .filter(|ws| now.as_millis() - ws.market.close_time.as_millis() >= PRUNE_AFTER_CLOSE_MS)
            .map(|ws| ws.market.window)
            .collect();
        for win in drop {
            if let Some(ws) = self.windows.remove(&win) {
                self.pm.remove(&ws.up_token);
            }
        }
    }

    /// The windows currently tradable (Open/Closing) — the sample set.
    #[must_use]
    pub fn active_windows(&self) -> Vec<WindowId> {
        self.windows
            .iter()
            .filter(|(_, ws)| {
                matches!(
                    ws.lifecycle,
                    WindowLifecycle::Open | WindowLifecycle::Closing
                )
            })
            .map(|(w, _)| *w)
            .collect()
    }

    /// Samples the 24-feature vector for `window` at `sample_ts`. `None` if the
    /// window is unknown, its asset has no price state, or `tau ≤ 0`.
    #[must_use]
    pub fn sample(&self, window: WindowId, sample_ts: TimestampMs) -> Option<SampleOutput> {
        let ws = self.windows.get(&window)?;
        let asset = ws.market.window.series.asset;
        let price = self.assets.get(&asset)?;
        let tau_secs = (ws.market.close_time.as_millis() - sample_ts.as_millis()) as f64 / 1000.0;
        if tau_secs <= 0.0 {
            return None;
        }
        let elapsed_secs =
            (sample_ts.as_millis() - ws.market.window.open_time.as_millis()) as f64 / 1000.0;
        let strike = ws.strike.strike();
        let base: PriceSample = price.sample(
            strike,
            tau_secs,
            elapsed_secs,
            sample_ts,
            self.max_staleness_ms,
        );

        // Depth: latest ladder, nulled when stale.
        let (depth_feats, depth_asof_ms) = match self.depth.get(&asset).and_then(|d| d.last) {
            Some((sec, feats)) if sample_ts.as_millis() - sec * 1_000 <= self.max_staleness_ms => {
                (feats, Some(sec * 1_000))
            }
            Some((sec, _)) => ([f64::NAN; N_DEPTH_FEATURES], Some(sec * 1_000)),
            None => ([f64::NAN; N_DEPTH_FEATURES], None),
        };

        // PM: by Up token.
        let sample_sec = sample_ts.as_millis().div_euclid(1_000);
        let pm: PmSample = self
            .pm
            .get(&ws.up_token)
            .map(|pm| pm.sample(sample_sec, sample_ts, price, self.max_staleness_ms))
            .unwrap_or(PmSample {
                features: [f64::NAN; crate::features::pm::N_PM_FEATURES],
                asof_ms: None,
                run_secs: None,
            });

        let features = FeatureVector::assemble(&base, &depth_feats, &pm);
        Some(SampleOutput {
            features,
            strike,
            price_asof_sec: base.asof_sec,
            sigma_returns: base.sigma_returns,
            basis_samples: base.basis_samples,
            depth_asof_ms,
            pm_asof_ms: pm.asof_ms,
            pm_run_secs: pm.run_secs,
        })
    }

    /// Number of windows currently tracked (for diagnostics / tests).
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }
}

#[cfg(test)]
mod tests {
    use core_types::{
        ConditionId, FeeParams, ResolutionSource, Series, Size, TickSize, TokenPair, WindowDuration,
    };
    use rust_decimal::dec;

    use super::*;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn market(asset: Asset, open_ms: i64) -> Arc<MarketInfo> {
        let series = Series {
            asset,
            duration: WindowDuration::M5,
        };
        let win = WindowId {
            series,
            open_time: ts(open_ms),
        };
        Arc::new(MarketInfo {
            window: win,
            event_slug: "test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("100").unwrap(),
                down: TokenId::new("200").unwrap(),
            },
            close_time: ts(open_ms + 300_000),
            strike: None,
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

    #[test]
    fn window_open_registers_and_close_prunes() {
        let mut core = ShadowCore::new(&[Asset::Btc], 120_000).unwrap();
        let m = market(Asset::Btc, 0);
        core.on_event(&Event::Window {
            market: Arc::clone(&m),
            lifecycle: WindowLifecycle::Open,
        });
        assert_eq!(core.window_count(), 1);
        assert_eq!(core.active_windows().len(), 1);
        // Past close + grace → pruned.
        core.prune(ts(300_000 + PRUNE_AFTER_CLOSE_MS + 1));
        assert_eq!(core.window_count(), 0);
    }

    #[test]
    fn sample_returns_full_vector_with_tau_present() {
        let mut core = ShadowCore::new(&[Asset::Btc], 120_000).unwrap();
        core.on_event(&Event::Window {
            market: market(Asset::Btc, 0),
            lifecycle: WindowLifecycle::Open,
        });
        let win = WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: ts(0),
        };
        let out = core.sample(win, ts(60_000)).expect("active window samples");
        assert_eq!(out.features.values.len(), 24);
        // tau = (300-60) = 240 s at feature index 6.
        assert!((out.features.values[6] - 240.0).abs() < 1e-3);
        // No feed yet ⇒ price/depth/pm features NaN; strike not captured.
        assert!(out.strike.is_none());
        assert!(out.features.values[2].is_nan(), "sigma NaN before warmup");
    }

    #[test]
    fn tau_zero_after_close_yields_no_sample() {
        let mut core = ShadowCore::new(&[Asset::Btc], 120_000).unwrap();
        core.on_event(&Event::Window {
            market: market(Asset::Btc, 0),
            lifecycle: WindowLifecycle::Open,
        });
        let win = WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: ts(0),
        };
        assert!(core.sample(win, ts(300_000)).is_none(), "tau=0 at close");
    }
}
