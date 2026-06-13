//! Sans-IO feed-comparison diagnostics: who leads, by how much, and what the
//! value basis is between direct-Binance (mid + trades), RTDS-Binance, and
//! RTDS-Chainlink, per asset. This is the measurement behind `bot compare` —
//! it tells the operator which feed actually leads and calibrates the
//! `model` crate's basis correction (§8); it is NOT an engine path, and the
//! production basis estimator does not live here.
//!
//! Method notes:
//! - **All timing uses `ts_local`** — one local clock. Cross-feed timing on
//!   `ts_exchange` would be garbage: vendor clocks differ, the operator's
//!   wall clock has known skew (Decisions Log 2026-06-12), and direct
//!   bookTicker carries no exchange time at all.
//! - **Value basis** is sampled on a 100 ms last-observation-carried-forward
//!   grid: `(a/b − 1) × 10⁴` bps per cell where both streams have a value;
//!   mean and median over the window. Positive = `a` above `b`.
//! - **Cross-correlation lag**: 1-second log-returns on the same grid,
//!   Pearson-correlated at every lag in ±[`MAX_LAG_MS`]; the
//!   correlation-maximizing lag is reported. Positive = first-named stream
//!   leads. Robust to value offsets between venues; resolution is one grid
//!   cell.
//! - **Exact-value match lag** (direct streams vs RTDS-Binance only): RTDS's
//!   `crypto_prices` topic republishes Binance data, so its value
//!   *transitions* should appear verbatim in the direct stream slightly
//!   earlier. For each RTDS value transition, find the latest direct
//!   transition to the same exact [`Decimal`] at-or-before it (lookback
//!   [`MATCH_LOOKBACK_MS`]; small forward window for pathological cases,
//!   reported as negative lag); the per-transition delays give p50/p95 and a
//!   match rate. A high match rate against `direct:trade` but not
//!   `direct:mid` (or vice versa) reveals what the vendor actually publishes.
//!
//! `f64` internally: this is measurement math, not money (§2.6); values stay
//! [`Decimal`] only where exact equality matters (match lag).

use std::collections::BTreeMap;
use std::collections::VecDeque;

use core_types::{Asset, Decimal, DurationMs, PriceSource, PriceTick, TickKind, TimestampMs};
use rust_decimal::prelude::ToPrimitive;

/// Rolling comparison window.
pub const WINDOW: DurationMs = DurationMs::from_millis(5 * 60 * 1_000);

/// LOCF sampling grid step.
pub const GRID_MS: i64 = 100;

/// Cross-correlation lag search bound (each side).
pub const MAX_LAG_MS: i64 = 5_000;

/// Log-return horizon for the cross-correlation series.
pub const RETURN_HORIZON_MS: i64 = 1_000;

/// Minimum overlapping return samples for a correlation to be reported.
pub const MIN_XCORR_SAMPLES: usize = 30;

/// How far back an exact-value match may look.
pub const MATCH_LOOKBACK_MS: i64 = 10_000;

/// Small forward window for exact-value matches (a vendor beating the direct
/// feed is pathological but worth seeing, as a negative lag).
pub const MATCH_FORWARD_MS: i64 = 2_000;

/// One comparison stream: which feed and which flavor of tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreamId {
    /// Producing feed.
    pub source: PriceSource,
    /// Observation flavor.
    pub kind: TickKind,
}

/// Direct-Binance bookTicker midpoint.
pub const DIRECT_MID: StreamId = StreamId {
    source: PriceSource::BinanceDirect,
    kind: TickKind::Mid,
};
/// Direct-Binance trade prints.
pub const DIRECT_TRADE: StreamId = StreamId {
    source: PriceSource::BinanceDirect,
    kind: TickKind::Trade,
};
/// RTDS `crypto_prices` (Binance-sourced, flavor unknown).
pub const RTDS_BINANCE: StreamId = StreamId {
    source: PriceSource::BinanceRtds,
    kind: TickKind::Vendor,
};
/// RTDS `crypto_prices_chainlink` (the resolution-grade feed).
pub const CHAINLINK: StreamId = StreamId {
    source: PriceSource::ChainlinkRtds,
    kind: TickKind::Vendor,
};

/// The basis/cross-correlation pairs, in report order.
const VALUE_PAIRS: [(StreamId, StreamId); 3] = [
    (DIRECT_MID, RTDS_BINANCE),
    (DIRECT_MID, CHAINLINK),
    (RTDS_BINANCE, CHAINLINK),
];

/// The exact-value match pairs (direct flavor under test vs RTDS-Binance).
const MATCH_PAIRS: [(StreamId, StreamId); 2] =
    [(DIRECT_TRADE, RTDS_BINANCE), (DIRECT_MID, RTDS_BINANCE)];

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.source, self.kind) {
            (PriceSource::BinanceDirect, TickKind::Mid) => f.write_str("direct:mid"),
            (PriceSource::BinanceDirect, TickKind::Trade) => f.write_str("direct:trade"),
            (PriceSource::BinanceRtds, _) => f.write_str("rtds-binance"),
            (PriceSource::ChainlinkRtds, _) => f.write_str("chainlink"),
            (source, kind) => write!(f, "{source:?}:{kind:?}"),
        }
    }
}

/// One buffered observation.
#[derive(Debug, Clone, Copy)]
struct Point {
    ts_ms: i64,
    value: Decimal,
    value_f: f64,
}

/// Rolling per-stream buffer, pruned to [`WINDOW`] behind the latest point.
#[derive(Debug, Default)]
struct Ring {
    points: VecDeque<Point>,
}

impl Ring {
    fn push(&mut self, point: Point) {
        // Out-of-order ts_local can't happen from one bus, but never let it
        // corrupt the monotonic buffer.
        if let Some(last) = self.points.back()
            && point.ts_ms < last.ts_ms
        {
            return;
        }
        self.points.push_back(point);
        let cutoff = point.ts_ms - WINDOW.as_millis();
        while self.points.front().is_some_and(|p| p.ts_ms < cutoff) {
            self.points.pop_front();
        }
    }

    /// Window slice (points within [`WINDOW`] of `now`), materialized for
    /// binary searching.
    fn window(&self, now: TimestampMs) -> Vec<Point> {
        let cutoff = now.as_millis() - WINDOW.as_millis();
        self.points
            .iter()
            .filter(|p| p.ts_ms >= cutoff && p.ts_ms <= now.as_millis())
            .copied()
            .collect()
    }
}

/// Per-stream activity stats.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamStat {
    /// The stream.
    pub id: StreamId,
    /// Ticks inside the window.
    pub ticks: usize,
    /// Average tick rate over the observed span.
    pub rate_hz: f64,
    /// Most recent value.
    pub latest: Option<Decimal>,
    /// Age of the most recent value at summary time.
    pub age: Option<DurationMs>,
}

/// Value basis between two streams over the window.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisStat {
    /// First-named stream.
    pub a: StreamId,
    /// Second-named stream.
    pub b: StreamId,
    /// Mean of `(a/b − 1) × 10⁴` over the grid (positive = `a` above `b`).
    pub mean_bps: f64,
    /// Median of the same.
    pub p50_bps: f64,
    /// Grid cells where both streams had a value.
    pub samples: usize,
}

/// Exact-value match lag between a direct stream and RTDS-Binance.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchLagStat {
    /// The direct stream under test.
    pub a: StreamId,
    /// The reference (RTDS-Binance).
    pub b: StreamId,
    /// Median republish delay in ms (positive = `a` led).
    pub p50_ms: i64,
    /// 95th-percentile delay.
    pub p95_ms: i64,
    /// Matched fraction of the reference's value transitions.
    pub match_rate: f64,
    /// Matched transition count.
    pub matches: usize,
}

/// Best cross-correlation lag between two streams.
#[derive(Debug, Clone, PartialEq)]
pub struct XcorrStat {
    /// First-named stream.
    pub a: StreamId,
    /// Second-named stream.
    pub b: StreamId,
    /// Correlation-maximizing lag in ms (positive = `a` leads).
    pub lag_ms: i64,
    /// The correlation at that lag.
    pub correlation: f64,
    /// Overlapping return samples at that lag.
    pub samples: usize,
}

/// One asset's full comparison summary.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetSummary {
    /// The asset.
    pub asset: Asset,
    /// Per-stream activity (only streams that have delivered).
    pub streams: Vec<StreamStat>,
    /// Pairwise value bases (only computable pairs).
    pub bases: Vec<BasisStat>,
    /// Exact-value match lags (only computable pairs).
    pub match_lags: Vec<MatchLagStat>,
    /// Best cross-correlation lags (only computable pairs).
    pub xcorr: Vec<XcorrStat>,
}

/// The comparator: feed it every [`PriceTick`] from the bus, ask for a
/// [`Comparator::summary`] periodically.
#[derive(Debug, Default)]
pub struct Comparator {
    assets: BTreeMap<Asset, BTreeMap<StreamId, Ring>>,
}

impl Comparator {
    /// Empty comparator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffers one observation (any source/kind — streams appear as they
    /// deliver).
    pub fn on_tick(&mut self, tick: &PriceTick) {
        let Some(value_f) = tick.value.to_f64() else {
            return;
        };
        let id = StreamId {
            source: tick.source,
            kind: tick.kind,
        };
        self.assets
            .entry(tick.asset)
            .or_default()
            .entry(id)
            .or_default()
            .push(Point {
                ts_ms: tick.ts_local.as_millis(),
                value: tick.value,
                value_f,
            });
    }

    /// The full per-asset comparison at `now`. Deterministic given the
    /// buffered ticks; streams/pairs without enough data are simply absent.
    #[must_use]
    pub fn summary(&self, now: TimestampMs) -> Vec<AssetSummary> {
        self.assets
            .iter()
            .map(|(&asset, rings)| {
                let windows: BTreeMap<StreamId, Vec<Point>> = rings
                    .iter()
                    .map(|(&id, ring)| (id, ring.window(now)))
                    .collect();
                AssetSummary {
                    asset,
                    streams: stream_stats(&windows, now),
                    bases: basis_stats(&windows, now),
                    match_lags: match_lag_stats(&windows),
                    xcorr: xcorr_stats(&windows, now),
                }
            })
            .collect()
    }
}

fn stream_stats(windows: &BTreeMap<StreamId, Vec<Point>>, now: TimestampMs) -> Vec<StreamStat> {
    windows
        .iter()
        .map(|(&id, points)| {
            let latest = points.last();
            let span_ms = points
                .first()
                .map_or(0, |first| (now.as_millis() - first.ts_ms).max(1));
            StreamStat {
                id,
                ticks: points.len(),
                rate_hz: if span_ms == 0 {
                    0.0
                } else {
                    points.len() as f64 * 1_000.0 / span_ms as f64
                },
                latest: latest.map(|p| p.value),
                age: latest.map(|p| DurationMs::from_millis(now.as_millis() - p.ts_ms)),
            }
        })
        .collect()
}

/// The LOCF grid over the window: cell `c` holds the latest value at or
/// before `start + c × GRID_MS` (`None` before the stream's first point).
fn grid(points: &[Point], start_ms: i64, cells: usize) -> Vec<Option<f64>> {
    let mut out = vec![None; cells];
    let mut idx = 0;
    let mut last = None;
    for (c, cell) in out.iter_mut().enumerate() {
        let t = start_ms + c as i64 * GRID_MS;
        while idx < points.len() && points[idx].ts_ms <= t {
            last = Some(points[idx].value_f);
            idx += 1;
        }
        *cell = last;
    }
    out
}

fn grid_span(now: TimestampMs) -> (i64, usize) {
    let cells = usize::try_from(WINDOW.as_millis() / GRID_MS).unwrap_or(0) + 1;
    (now.as_millis() - WINDOW.as_millis(), cells)
}

fn basis_stats(windows: &BTreeMap<StreamId, Vec<Point>>, now: TimestampMs) -> Vec<BasisStat> {
    let (start, cells) = grid_span(now);
    let mut out = Vec::new();
    for (a, b) in VALUE_PAIRS {
        let (Some(pa), Some(pb)) = (windows.get(&a), windows.get(&b)) else {
            continue;
        };
        let (ga, gb) = (grid(pa, start, cells), grid(pb, start, cells));
        let mut diffs: Vec<f64> = ga
            .iter()
            .zip(&gb)
            .filter_map(|(va, vb)| match (va, vb) {
                (Some(x), Some(y)) if *y != 0.0 => Some((x / y - 1.0) * 10_000.0),
                _ => None,
            })
            .collect();
        if diffs.is_empty() {
            continue;
        }
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        diffs.sort_by(f64::total_cmp);
        out.push(BasisStat {
            a,
            b,
            mean_bps: mean,
            p50_bps: percentile(&diffs, 0.5),
            samples: diffs.len(),
        });
    }
    out
}

fn xcorr_stats(windows: &BTreeMap<StreamId, Vec<Point>>, now: TimestampMs) -> Vec<XcorrStat> {
    let (start, cells) = grid_span(now);
    let horizon = usize::try_from(RETURN_HORIZON_MS / GRID_MS)
        .unwrap_or(1)
        .max(1);
    let max_lag_cells = MAX_LAG_MS / GRID_MS;
    let mut out = Vec::new();
    for (a, b) in VALUE_PAIRS {
        let (Some(pa), Some(pb)) = (windows.get(&a), windows.get(&b)) else {
            continue;
        };
        let ra = log_returns(&grid(pa, start, cells), horizon);
        let rb = log_returns(&grid(pb, start, cells), horizon);
        let mut best: Option<XcorrStat> = None;
        for lag_cells in -max_lag_cells..=max_lag_cells {
            let Some((corr, samples)) = lagged_pearson(&ra, &rb, lag_cells) else {
                continue;
            };
            if best.as_ref().is_none_or(|s| corr > s.correlation) {
                best = Some(XcorrStat {
                    a,
                    b,
                    lag_ms: lag_cells * GRID_MS,
                    correlation: corr,
                    samples,
                });
            }
        }
        if let Some(stat) = best {
            out.push(stat);
        }
    }
    out
}

/// Log-returns over `horizon` cells; `None` where either endpoint is.
fn log_returns(grid: &[Option<f64>], horizon: usize) -> Vec<Option<f64>> {
    (0..grid.len())
        .map(|i| {
            if i < horizon {
                return None;
            }
            match (grid[i - horizon], grid[i]) {
                (Some(past), Some(cur)) if past > 0.0 && cur > 0.0 => Some((cur / past).ln()),
                _ => None,
            }
        })
        .collect()
}

/// Pearson correlation of `a[i]` against `b[i + lag]` (positive lag = `a`
/// leads `b`), over cells where both are present. `None` without enough
/// samples or variance.
fn lagged_pearson(a: &[Option<f64>], b: &[Option<f64>], lag: i64) -> Option<(f64, usize)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, va) in a.iter().enumerate() {
        let j = i as i64 + lag;
        if j < 0 || j as usize >= b.len() {
            continue;
        }
        if let (Some(x), Some(y)) = (va, b[j as usize]) {
            xs.push(*x);
            ys.push(y);
        }
    }
    if xs.len() < MIN_XCORR_SAMPLES {
        return None;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in xs.iter().zip(&ys) {
        cov += (x - mean_x) * (y - mean_y);
        var_x += (x - mean_x).powi(2);
        var_y += (y - mean_y).powi(2);
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    Some((cov / (var_x.sqrt() * var_y.sqrt()), xs.len()))
}

/// Points where the value changed from its predecessor (a vendor repeating
/// an unchanged value each second is not a new observation).
fn transitions(points: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::new();
    for p in points {
        if out.last().is_none_or(|prev| prev.value != p.value) {
            out.push(*p);
        }
    }
    out
}

fn match_lag_stats(windows: &BTreeMap<StreamId, Vec<Point>>) -> Vec<MatchLagStat> {
    let mut out = Vec::new();
    for (a, b) in MATCH_PAIRS {
        let (Some(pa), Some(pb)) = (windows.get(&a), windows.get(&b)) else {
            continue;
        };
        let a_trans = transitions(pa);
        let b_trans = transitions(pb);
        if a_trans.is_empty() || b_trans.is_empty() {
            continue;
        }
        let mut lags: Vec<f64> = Vec::new();
        for bp in &b_trans {
            if let Some(lag) = find_match(&a_trans, bp) {
                lags.push(lag as f64);
            }
        }
        if lags.is_empty() {
            continue;
        }
        lags.sort_by(f64::total_cmp);
        out.push(MatchLagStat {
            a,
            b,
            p50_ms: percentile(&lags, 0.5) as i64,
            p95_ms: percentile(&lags, 0.95) as i64,
            match_rate: lags.len() as f64 / b_trans.len() as f64,
            matches: lags.len(),
        });
    }
    out
}

/// The republish delay for one reference transition: latest direct
/// transition to the same exact value at-or-before it (positive lag), else
/// the earliest one shortly after (negative lag), else `None`.
fn find_match(a_trans: &[Point], bp: &Point) -> Option<i64> {
    let split = a_trans.partition_point(|p| p.ts_ms <= bp.ts_ms);
    for ap in a_trans[..split].iter().rev() {
        if bp.ts_ms - ap.ts_ms > MATCH_LOOKBACK_MS {
            break;
        }
        if ap.value == bp.value {
            return Some(bp.ts_ms - ap.ts_ms);
        }
    }
    for ap in &a_trans[split..] {
        if ap.ts_ms - bp.ts_ms > MATCH_FORWARD_MS {
            break;
        }
        if ap.value == bp.value {
            return Some(bp.ts_ms - ap.ts_ms);
        }
    }
    None
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const T0: i64 = 1_700_000_000_000;

    fn tick(id: StreamId, asset: Asset, ts_ms: i64, value: Decimal) -> PriceTick {
        PriceTick {
            source: id.source,
            asset,
            kind: id.kind,
            value,
            ts_exchange: TimestampMs::from_millis(ts_ms),
            ts_local: TimestampMs::from_millis(ts_ms),
        }
    }

    /// Deterministic walk values (no `rand` on the tree): a few cents of
    /// pseudo-random movement per step around 63 500.
    fn walk(steps: usize, step_ms: i64) -> Vec<(i64, Decimal)> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut value = dec!(63500.00);
        let mut out = Vec::with_capacity(steps);
        for i in 0..steps {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let cents = i64::try_from(state.wrapping_mul(0x2545_F491_4F6C_DD1D) % 41).unwrap_or(0);
            value += Decimal::new(cents - 20, 2); // −0.20 ..= +0.20
            out.push((T0 + i as i64 * step_ms, value));
        }
        out
    }

    #[test]
    fn xcorr_recovers_an_injected_lag() {
        let mut comparator = Comparator::new();
        let series = walk(1_500, 100); // 2.5 minutes at 10 Hz
        for &(ts, value) in &series {
            comparator.on_tick(&tick(DIRECT_MID, Asset::Btc, ts, value));
            // The vendor republishes the same values 700 ms later.
            comparator.on_tick(&tick(RTDS_BINANCE, Asset::Btc, ts + 700, value));
        }
        let now = TimestampMs::from_millis(T0 + 1_500 * 100 + 700);
        let summary = comparator.summary(now);
        assert_eq!(summary.len(), 1);
        let xc = summary[0]
            .xcorr
            .iter()
            .find(|s| (s.a, s.b) == (DIRECT_MID, RTDS_BINANCE))
            .expect("xcorr computed");
        assert_eq!(xc.lag_ms, 700, "injected lag recovered: {xc:?}");
        assert!(xc.correlation > 0.95, "near-perfect copy: {xc:?}");
        assert!(xc.samples >= MIN_XCORR_SAMPLES);
    }

    #[test]
    fn match_lag_recovers_the_republish_delay() {
        let mut comparator = Comparator::new();
        // Direct trades transition every 500 ms; the vendor snapshots each
        // transition 700 ms later (and repeats values in between — repeats
        // must not count as new transitions).
        let series = walk(300, 500);
        for &(ts, value) in &series {
            comparator.on_tick(&tick(DIRECT_TRADE, Asset::Eth, ts, value));
            comparator.on_tick(&tick(RTDS_BINANCE, Asset::Eth, ts + 700, value));
            comparator.on_tick(&tick(RTDS_BINANCE, Asset::Eth, ts + 950, value)); // repeat
        }
        let now = TimestampMs::from_millis(T0 + 300 * 500 + 1_000);
        let summary = comparator.summary(now);
        let ml = summary[0]
            .match_lags
            .iter()
            .find(|s| (s.a, s.b) == (DIRECT_TRADE, RTDS_BINANCE))
            .expect("match lag computed");
        assert_eq!(ml.p50_ms, 700, "{ml:?}");
        assert_eq!(ml.p95_ms, 700, "{ml:?}");
        assert!(ml.match_rate > 0.9, "{ml:?}");
    }

    #[test]
    fn constant_multiplicative_offset_shows_as_basis() {
        let mut comparator = Comparator::new();
        let series = walk(600, 200);
        for &(ts, value) in &series {
            comparator.on_tick(&tick(DIRECT_MID, Asset::Btc, ts, value));
            // Chainlink trades 3 bps above Binance.
            comparator.on_tick(&tick(CHAINLINK, Asset::Btc, ts, value * dec!(1.0003)));
        }
        let now = TimestampMs::from_millis(T0 + 600 * 200);
        let summary = comparator.summary(now);
        let basis = summary[0]
            .bases
            .iter()
            .find(|s| (s.a, s.b) == (DIRECT_MID, CHAINLINK))
            .expect("basis computed");
        // a = direct, b = chainlink = direct × 1.0003 → a/b − 1 ≈ −3 bps.
        assert!(
            (basis.mean_bps + 3.0).abs() < 0.05,
            "expected ≈ −3 bps: {basis:?}"
        );
        assert!((basis.p50_bps + 3.0).abs() < 0.05);
        assert!(basis.samples > 1_000);
    }

    #[test]
    fn rings_prune_to_the_window() {
        let mut comparator = Comparator::new();
        // 20 minutes of 10 Hz ticks — 4× the window.
        for i in 0..12_000_i64 {
            comparator.on_tick(&tick(DIRECT_MID, Asset::Btc, T0 + i * 100, dec!(63500)));
        }
        let ring = &comparator.assets[&Asset::Btc][&DIRECT_MID];
        let span =
            ring.points.back().expect("points").ts_ms - ring.points.front().expect("points").ts_ms;
        assert!(
            span <= WINDOW.as_millis(),
            "ring spans {span}ms > window {}ms",
            WINDOW.as_millis()
        );
        assert!(ring.points.len() <= 3_001);
    }

    #[test]
    fn sparse_or_lone_streams_yield_stats_but_no_pairs() {
        let mut comparator = Comparator::new();
        comparator.on_tick(&tick(DIRECT_MID, Asset::Btc, T0, dec!(63500)));
        comparator.on_tick(&tick(DIRECT_MID, Asset::Btc, T0 + 100, dec!(63501)));
        let summary = comparator.summary(TimestampMs::from_millis(T0 + 200));
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].streams.len(), 1);
        assert_eq!(summary[0].streams[0].ticks, 2);
        assert!(summary[0].bases.is_empty());
        assert!(summary[0].match_lags.is_empty());
        assert!(summary[0].xcorr.is_empty());
        // And a completely empty comparator summarizes to nothing.
        assert!(
            Comparator::new()
                .summary(TimestampMs::from_millis(T0))
                .is_empty()
        );
    }

    #[test]
    fn assets_are_compared_independently() {
        let mut comparator = Comparator::new();
        let series = walk(400, 200);
        for &(ts, value) in &series {
            comparator.on_tick(&tick(DIRECT_MID, Asset::Btc, ts, value));
            comparator.on_tick(&tick(RTDS_BINANCE, Asset::Btc, ts + 500, value));
            // ETH only has one stream — no pairs for it.
            comparator.on_tick(&tick(DIRECT_MID, Asset::Eth, ts, value / dec!(38)));
        }
        let now = TimestampMs::from_millis(T0 + 400 * 200 + 500);
        let summary = comparator.summary(now);
        assert_eq!(summary.len(), 2);
        let btc = summary.iter().find(|s| s.asset == Asset::Btc).expect("btc");
        let eth = summary.iter().find(|s| s.asset == Asset::Eth).expect("eth");
        assert!(!btc.xcorr.is_empty());
        assert!(eth.xcorr.is_empty());
        let xc = &btc.xcorr[0];
        assert_eq!(xc.lag_ms, 500);
    }

    #[test]
    fn display_names_are_operator_friendly() {
        assert_eq!(DIRECT_MID.to_string(), "direct:mid");
        assert_eq!(DIRECT_TRADE.to_string(), "direct:trade");
        assert_eq!(RTDS_BINANCE.to_string(), "rtds-binance");
        assert_eq!(CHAINLINK.to_string(), "chainlink");
    }
}
