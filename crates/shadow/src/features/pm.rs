//! Per-Up-token Polymarket book features — the 6 PM features.
//!
//! Reproduces `short_horizon.build_pm_grid`: `pm_mid`/`pm_spread`/`pm_book_imb`
//! from the last top-of-book in each second, and `pm_staleness_{1,2,3}s` — the
//! Binance-mid log-move minus the PM-mid change over the same 1..3-second span,
//! both via LOCF lookups on the per-token contiguous reindex (causal). Fed from
//! `Event::TopOfBook` (the same `top_of_book` records the lab reads), keyed by
//! the window's **Up** token.

use core_types::TimestampMs;

use super::price::PriceState;

/// Staleness lookback spans (seconds).
const LOOKBACKS: [i64; 3] = [1, 2, 3];
/// Number of PM features (in `PM_FEATURES` order).
pub const N_PM_FEATURES: usize = 6;
/// Recent observed PM seconds retained (>= max lookback + slack).
const OBS_RING_CAP: usize = 8;

/// One observed second's PM top-of-book.
#[derive(Debug, Clone, Copy)]
struct PmObs {
    sec: i64,
    mid: f64,
    spread: f64,
    /// `(bid_sz - ask_sz)/(bid_sz + ask_sz)`; NaN when the denominator is 0.
    imbalance: f64,
}

/// Per-Up-token PM book feature state.
#[derive(Debug, Clone)]
pub struct PmState {
    /// Last observation per second, newest last (cap [`OBS_RING_CAP`]).
    obs: std::collections::VecDeque<PmObs>,
    /// First observed second for this token (drives `pm_run_secs`).
    run_start: Option<i64>,
}

/// The 6 PM features plus diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct PmSample {
    /// `[pm_mid, pm_spread, pm_book_imb, pm_staleness_1s, pm_staleness_2s, pm_staleness_3s]`.
    pub features: [f64; N_PM_FEATURES],
    /// The matched observed-second ts (ms), `None` if no book yet.
    pub asof_ms: Option<i64>,
    /// Seconds since this token's book run started (`None` if no book yet).
    pub run_secs: Option<i64>,
}

impl PmState {
    /// A fresh state for one Up token.
    #[must_use]
    pub fn new() -> Self {
        Self {
            obs: std::collections::VecDeque::with_capacity(OBS_RING_CAP),
            run_start: None,
        }
    }

    /// Folds one top-of-book (bid/ask price+size) at `ts` (ms). Keeps the last
    /// book in each second.
    pub fn on_top(&mut self, ts: TimestampMs, bid_px: f64, bid_sz: f64, ask_px: f64, ask_sz: f64) {
        let sec = ts.as_millis().div_euclid(1_000);
        let denom = bid_sz + ask_sz;
        let obs = PmObs {
            sec,
            mid: 0.5 * (bid_px + ask_px),
            spread: ask_px - bid_px,
            imbalance: if denom > 0.0 {
                (bid_sz - ask_sz) / denom
            } else {
                f64::NAN
            },
        };
        self.run_start.get_or_insert(sec);
        match self.obs.back() {
            Some(last) if last.sec == sec => {
                // Last book of the second wins.
                if let Some(back) = self.obs.back_mut() {
                    *back = obs;
                }
            }
            _ => {
                if self.obs.len() == OBS_RING_CAP {
                    self.obs.pop_front();
                }
                self.obs.push_back(obs);
            }
        }
    }

    /// The last observation at or before `sec`.
    fn obs_at(&self, sec: i64) -> Option<&PmObs> {
        self.obs.iter().rev().find(|o| o.sec <= sec)
    }

    /// `pm_mid` LOCF at or before `sec`.
    fn mid_at(&self, sec: i64) -> Option<f64> {
        self.obs_at(sec).map(|o| o.mid)
    }

    /// Samples the 6 PM features. `price` supplies the Binance-mid LOCF for the
    /// staleness signals. PM features are nulled when the matched book is older
    /// than `max_staleness_ms`.
    #[must_use]
    pub fn sample(
        &self,
        sample_sec: i64,
        sample_ts: TimestampMs,
        price: &PriceState,
        max_staleness_ms: i64,
    ) -> PmSample {
        let mut features = [f64::NAN; N_PM_FEATURES];
        let Some(m) = self.obs_at(sample_sec) else {
            return PmSample {
                features,
                asof_ms: None,
                run_secs: None,
            };
        };
        let run_secs = self.run_start.map(|start| m.sec - start);
        if sample_ts.as_millis() - m.sec * 1_000 > max_staleness_ms {
            return PmSample {
                features,
                asof_ms: Some(m.sec * 1_000),
                run_secs,
            };
        }
        features[0] = m.mid;
        features[1] = m.spread;
        features[2] = m.imbalance;
        // Staleness is computed at the matched observed second (grid semantics).
        let s = m.sec;
        let run = self.run_start.map_or(0, |start| s - start);
        for (i, &lb) in LOOKBACKS.iter().enumerate() {
            features[3 + i] = if run < lb {
                f64::NAN
            } else {
                match (
                    price.mid_asof(s),
                    price.mid_asof(s - lb),
                    self.mid_at(s),
                    self.mid_at(s - lb),
                ) {
                    (Some(bser_s), Some(bser_slb), Some(pm_s), Some(pm_slb))
                        if bser_s > 0.0 && bser_slb > 0.0 =>
                    {
                        (bser_s / bser_slb).ln() - (pm_s - pm_slb)
                    }
                    _ => f64::NAN,
                }
            };
        }
        PmSample {
            features,
            asof_ms: Some(m.sec * 1_000),
            run_secs,
        }
    }
}

impl Default for PmState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use core_types::Asset;

    use super::*;

    const T0: i64 = 1_700_000_000_000;

    fn ts(sec: i64) -> TimestampMs {
        TimestampMs::from_millis(T0 + sec * 1_000)
    }

    #[test]
    fn mid_spread_imbalance_from_last_book_in_second() {
        let mut pm = PmState::new();
        pm.on_top(ts(0), 0.40, 100.0, 0.42, 60.0);
        // A later book in the same second overwrites.
        pm.on_top(TimestampMs::from_millis(T0 + 500), 0.41, 100.0, 0.43, 100.0);
        let price = PriceState::new(Asset::Btc).unwrap();
        let s = pm.sample(ts(0).as_millis().div_euclid(1_000), ts(0), &price, 120_000);
        assert!(
            (s.features[0] - 0.42).abs() < 1e-12,
            "pm_mid = (0.41+0.43)/2"
        );
        assert!(
            (s.features[1] - 0.02).abs() < 1e-12,
            "pm_spread = 0.43-0.41"
        );
        assert!(
            (s.features[2] - 0.0).abs() < 1e-12,
            "imbalance = 0 (100 vs 100)"
        );
    }

    #[test]
    fn staleness_nan_before_run_reaches_lookback() {
        let mut pm = PmState::new();
        pm.on_top(ts(10), 0.40, 50.0, 0.42, 50.0);
        let price = PriceState::new(Asset::Btc).unwrap();
        // run_secs = 0 at the first observed second → all staleness NaN.
        let s = pm.sample(
            ts(10).as_millis().div_euclid(1_000),
            ts(10),
            &price,
            120_000,
        );
        assert!(s.features[3].is_nan() && s.features[4].is_nan() && s.features[5].is_nan());
        assert_eq!(s.run_secs, Some(0));
    }

    #[test]
    fn stale_book_nulls_pm_features() {
        let mut pm = PmState::new();
        pm.on_top(ts(0), 0.40, 50.0, 0.42, 50.0);
        let price = PriceState::new(Asset::Btc).unwrap();
        // Sample 200 s later → matched book stale.
        let s = pm.sample(
            ts(200).as_millis().div_euclid(1_000),
            ts(200),
            &price,
            120_000,
        );
        assert!(s.features.iter().all(|v| v.is_nan()));
        assert_eq!(s.asof_ms, Some(T0));
    }
}
