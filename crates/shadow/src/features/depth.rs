//! Binance `depth20@100ms` microstructure features — the 8 DEPTH features.
//!
//! Reproduces `short_horizon._frame_depth_features` / `_side_slope` exactly over
//! one raw depth ladder: multi-level cumulative imbalance at levels {1,5,10,20},
//! the cross-weighted microprice gap, top-10 bid/ask depth slopes, and the
//! spread in bps. A thin book yields honest `NaN` (never imputed depth), matching
//! the lab's `min(L, n_levels)` rule. Stateless per frame; the per-second grid
//! (last ladder in each second) is maintained by the caller.

use core_types::Asset;

/// Cumulative-size level counts for the multi-level imbalances.
const LEVEL_DEPTHS: [usize; 4] = [1, 5, 10, 20];
/// Top levels used for the bid/ask depth slope.
const SLOPE_LEVELS: usize = 10;
/// Number of DEPTH features (in `DEPTH_FEATURES` order).
pub const N_DEPTH_FEATURES: usize = 8;

/// One parsed depth ladder (best level first: bids descending, asks ascending),
/// as delivered by Binance `depth20@100ms`.
#[derive(Debug, Clone)]
pub struct DepthLadder {
    /// Which asset this ladder is for.
    pub asset: Asset,
    /// Local receive time (ms) — the per-second grid key.
    pub recv_ms: i64,
    /// Bid prices, best (highest) first.
    pub bid_px: Vec<f64>,
    /// Bid sizes, aligned to `bid_px`.
    pub bid_sz: Vec<f64>,
    /// Ask prices, best (lowest) first.
    pub ask_px: Vec<f64>,
    /// Ask sizes, aligned to `ask_px`.
    pub ask_sz: Vec<f64>,
}

/// Cumulative sums of a size vector (`cumsum`).
fn cumsum(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    let mut acc = 0.0;
    for &x in v {
        acc += x;
        out.push(acc);
    }
    out
}

/// `_side_slope`: cumulative size over the top `k` levels per USD of price
/// distance from the best. NaN with < 2 levels or a non-positive distance.
fn side_slope(px: &[f64], cum_sz: &[f64], k_levels: usize, is_bid: bool) -> f64 {
    let k = k_levels.min(px.len());
    if k < 2 {
        return f64::NAN;
    }
    let (best, edge) = (px[0], px[k - 1]);
    let dist = if is_bid { best - edge } else { edge - best };
    if dist <= 0.0 {
        return f64::NAN;
    }
    cum_sz[k - 1] / dist
}

/// Computes the 8 DEPTH features from one ladder, in `DEPTH_FEATURES` order:
/// `[depth_imb_1, depth_imb_5, depth_imb_10, depth_imb_20, microprice_gap,
/// bid_depth_slope, ask_depth_slope, depth_spread_bps]`. An empty side yields
/// all-NaN (no book).
#[must_use]
pub fn frame_depth_features(ladder: &DepthLadder) -> [f64; N_DEPTH_FEATURES] {
    let mut out = [f64::NAN; N_DEPTH_FEATURES];
    if ladder.bid_px.is_empty() || ladder.ask_px.is_empty() {
        return out;
    }
    let (best_bid, best_ask) = (ladder.bid_px[0], ladder.ask_px[0]);
    let mid = 0.5 * (best_bid + best_ask);
    let cum_bid = cumsum(&ladder.bid_sz);
    let cum_ask = cumsum(&ladder.ask_sz);

    for (i, &lv) in LEVEL_DEPTHS.iter().enumerate() {
        let nb = lv.min(ladder.bid_sz.len());
        let na = lv.min(ladder.ask_sz.len());
        let b = if nb > 0 { cum_bid[nb - 1] } else { 0.0 };
        let a = if na > 0 { cum_ask[na - 1] } else { 0.0 };
        let denom = b + a;
        out[i] = if denom > 0.0 {
            (b - a) / denom
        } else {
            f64::NAN
        };
    }

    let top_denom = ladder.bid_sz[0] + ladder.ask_sz[0];
    let microprice = if top_denom > 0.0 {
        (best_bid * ladder.ask_sz[0] + best_ask * ladder.bid_sz[0]) / top_denom
    } else {
        mid
    };
    out[4] = microprice - mid;
    out[5] = side_slope(&ladder.bid_px, &cum_bid, SLOPE_LEVELS, true);
    out[6] = side_slope(&ladder.ask_px, &cum_ask, SLOPE_LEVELS, false);
    out[7] = if mid > 0.0 {
        1e4 * (best_ask - best_bid) / mid
    } else {
        f64::NAN
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> DepthLadder {
        // bids 100.00(1.5), 99.99(2.0), 99.98(0.5); asks 100.01(3.0), 100.02(1.0).
        DepthLadder {
            asset: Asset::Btc,
            recv_ms: 1_700_000_000_000,
            bid_px: vec![100.00, 99.99, 99.98],
            bid_sz: vec![1.5, 2.0, 0.5],
            ask_px: vec![100.01, 100.02],
            ask_sz: vec![3.0, 1.0],
        }
    }

    #[test]
    fn imbalance_and_microprice_match_hand_computation() {
        let f = frame_depth_features(&ladder());
        // depth_imb_1: (1.5 - 3.0)/(1.5 + 3.0) = -1.5/4.5.
        assert!((f[0] - (-1.5 / 4.5)).abs() < 1e-12);
        // depth_imb_5 (min(5,3)=3 bids, min(5,2)=2 asks): cumB=4.0, cumA=4.0 → 0.
        assert!((f[1] - 0.0).abs() < 1e-12);
        // microprice = (100.00*3.0 + 100.01*1.5)/(1.5+3.0); gap = microprice - 100.005.
        let mp = (100.00 * 3.0 + 100.01 * 1.5) / 4.5;
        assert!((f[4] - (mp - 100.005)).abs() < 1e-9);
        // spread_bps = 1e4*(100.01-100.00)/100.005.
        assert!((f[7] - (1e4 * 0.01 / 100.005)).abs() < 1e-9);
        // slopes finite (>=2 levels each side).
        assert!(f[5].is_finite() && f[6].is_finite());
    }

    #[test]
    fn empty_side_is_all_nan() {
        let mut l = ladder();
        l.ask_px.clear();
        l.ask_sz.clear();
        assert!(frame_depth_features(&l).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn single_level_slope_is_nan() {
        let l = DepthLadder {
            asset: Asset::Eth,
            recv_ms: 0,
            bid_px: vec![10.0],
            bid_sz: vec![1.0],
            ask_px: vec![10.1],
            ask_sz: vec![1.0],
        };
        let f = frame_depth_features(&l);
        assert!(f[5].is_nan() && f[6].is_nan(), "slope needs >= 2 levels");
        assert!((f[0] - 0.0).abs() < 1e-12, "imb_1 defined");
    }
}
