//! Live order-activity metrics for the dashboard's calm "live activity" strip:
//! orders placed / cancelled / filled in a trailing window, the count of
//! currently-resting orders, and the median time-to-cancel and time-to-replace
//! that show how fast the cancel-first "yank-it-back" reflex actually is
//! (CLAUDE.md §8/§10).
//!
//! This is a **pure, sans-IO** fold over the bus [`OrderUpdate`] and [`Fill`]
//! events — every query is parameterized by a wall `now` (the dashboard threads
//! the same `now` it folds events with). Unlike [`Analytics`](crate::Analytics)
//! it is **not** part of the journal-replayed state machine: it is best-effort
//! live state (like the dashboard's `LiveMarkout`), owned per-mode by the
//! dashboard and fed from its `project()` choke point, so it needs no
//! replay-determinism guarantee. It holds only timestamps, counts, series and
//! ids — no money.
//!
//! # What is observable
//!
//! Order *state transitions* are visible on the bus; the internal
//! cancel-decision time is not. So:
//!
//! - **time-to-replace** (a slot's `Canceled` → the next placement on the same
//!   `(window, side)` within a short cap) is observable in **both** paper and
//!   live, and is the primary reflex metric.
//! - **time-to-cancel** (the true cancel round-trip `PendingCancel → Canceled`)
//!   is observable only in **live**: the paper venue cancels `Open → Canceled`
//!   with no `PendingCancel` state, so its median is `None` (the dashboard shows
//!   "—" there).

use std::collections::{HashMap, VecDeque};

use core_types::{Fill, OrderState, OrderUpdate, Series, Side, TimestampMs, WindowId};

/// How long a trailing-window count entry is retained (entries older than this
/// are pruned on the next push; a hard cap backstops pathological bursts).
const RETENTION_MS: i64 = 600_000; // 10 minutes
/// Hard cap on each trailing-count ring.
const COUNT_CAP: usize = 16_384;
/// How many recent latency samples to keep per duration ring (the median window).
const DURATION_CAP: usize = 1_024;
/// A `Canceled` only pairs with a *replacement* placement on the same slot if
/// the new quote arrives within this long — beyond it, the placement is fresh
/// quoting, not a yank-and-replace.
const REPLACE_PAIR_CAP_MS: i64 = 5_000;

/// A `(window, side)` quoting slot — the granularity at which a cancel is paired
/// with its replacement for the time-to-replace metric.
type Slot = (WindowId, Side);

/// A trailing-window fold over order/fill events. See the module docs.
#[derive(Debug, Clone, Default)]
pub struct OrderActivity {
    /// `(ts, series)` of each newly-placed order, trailing window.
    placed: VecDeque<(TimestampMs, Series)>,
    /// `(ts, series)` of each cancellation, trailing window.
    cancelled: VecDeque<(TimestampMs, Series)>,
    /// `(ts, series)` of each fill, trailing window.
    fills: VecDeque<(TimestampMs, Series)>,
    /// Currently-working orders → their series, for the resting count.
    working: HashMap<core_types::OrderId, Series>,
    /// Pending cancel requests (`PendingCancel` ts in ms) awaiting their
    /// `Canceled` — the time-to-cancel start times. Empty in paper.
    cancel_requested: HashMap<core_types::OrderId, i64>,
    /// Most-recent `Canceled` ts (ms) per slot, awaiting a replacement.
    last_cancel: HashMap<Slot, i64>,
    /// `(series, ms)` cancel round-trip samples (live only).
    cancel_durations: VecDeque<(Series, i64)>,
    /// `(series, ms)` cancel→replace samples (both modes).
    replace_durations: VecDeque<(Series, i64)>,
}

impl OrderActivity {
    /// A fresh, empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one order-status update at wall time `now`.
    pub fn on_order(&mut self, u: &OrderUpdate, now: TimestampMs) {
        let series = u.window.series;
        let now_ms = now.as_millis();
        let was_working = self.working.contains_key(&u.order_id);

        match u.state {
            OrderState::PendingCancel => {
                // Start of a cancel round-trip (live only).
                self.cancel_requested.insert(u.order_id.clone(), now_ms);
            }
            OrderState::Canceled => {
                push_count(&mut self.cancelled, now, series);
                self.last_cancel.insert((u.window, u.side), now_ms);
                // Close a cancel round-trip if we saw the request (live).
                if let Some(req_ms) = self.cancel_requested.remove(&u.order_id) {
                    let dt = now_ms - req_ms;
                    if dt >= 0 {
                        push_dur(&mut self.cancel_durations, series, dt);
                    }
                }
            }
            _ => {}
        }

        // A placement is an order becoming working for the first time.
        if !was_working && !u.state.is_terminal() {
            push_count(&mut self.placed, now, series);
            // Pair with a recent cancel on the same slot → time-to-replace.
            if let Some(cancel_ms) = self.last_cancel.remove(&(u.window, u.side)) {
                let dt = now_ms - cancel_ms;
                if (0..=REPLACE_PAIR_CAP_MS).contains(&dt) {
                    push_dur(&mut self.replace_durations, series, dt);
                }
            }
        }

        // Maintain the working set (also drops a stale cancel-request on exit).
        if u.state.is_terminal() {
            self.working.remove(&u.order_id);
            self.cancel_requested.remove(&u.order_id);
        } else {
            self.working.insert(u.order_id.clone(), series);
        }
    }

    /// Folds one fill at wall time `now`.
    pub fn on_fill(&mut self, fill: &Fill, now: TimestampMs) {
        push_count(&mut self.fills, now, fill.window.series);
    }

    /// Orders placed within the last `window_ms`, optionally for one series.
    #[must_use]
    pub fn placed_since(&self, now: TimestampMs, window_ms: i64, series: Option<Series>) -> u64 {
        count_since(&self.placed, now, window_ms, series)
    }

    /// Orders cancelled within the last `window_ms`, optionally for one series.
    #[must_use]
    pub fn cancelled_since(&self, now: TimestampMs, window_ms: i64, series: Option<Series>) -> u64 {
        count_since(&self.cancelled, now, window_ms, series)
    }

    /// Fills within the last `window_ms`, optionally for one series.
    #[must_use]
    pub fn fills_since(&self, now: TimestampMs, window_ms: i64, series: Option<Series>) -> u64 {
        count_since(&self.fills, now, window_ms, series)
    }

    /// Orders resting on the book right now, optionally for one series.
    #[must_use]
    pub fn resting(&self, series: Option<Series>) -> u64 {
        let n = self
            .working
            .values()
            .filter(|s| series.is_none_or(|f| **s == f))
            .count();
        u64::try_from(n).unwrap_or(u64::MAX)
    }

    /// Median cancel round-trip in ms (live only; `None` in paper or with no
    /// sample), optionally for one series.
    #[must_use]
    pub fn median_time_to_cancel_ms(&self, series: Option<Series>) -> Option<i64> {
        median_dur(&self.cancel_durations, series)
    }

    /// Median cancel→replace gap in ms (both modes; `None` with no sample),
    /// optionally for one series.
    #[must_use]
    pub fn median_time_to_replace_ms(&self, series: Option<Series>) -> Option<i64> {
        median_dur(&self.replace_durations, series)
    }

    /// 95th-percentile cancel round-trip in ms (live only; `None` in paper or
    /// with no sample), optionally for one series — the benchmark tile's
    /// "cancel-p95" (renders n/a in paper, where the round-trip is unobservable).
    #[must_use]
    pub fn p95_time_to_cancel_ms(&self, series: Option<Series>) -> Option<i64> {
        percentile_dur(&self.cancel_durations, series, 0.95)
    }
}

/// Pushes a trailing-window count entry and prunes by age then hard cap.
fn push_count(ring: &mut VecDeque<(TimestampMs, Series)>, now: TimestampMs, series: Series) {
    ring.push_back((now, series));
    let cutoff = now.as_millis() - RETENTION_MS;
    while ring.front().is_some_and(|(ts, _)| ts.as_millis() < cutoff) {
        ring.pop_front();
    }
    while ring.len() > COUNT_CAP {
        ring.pop_front();
    }
}

/// Pushes a latency sample and caps the ring to the median window.
fn push_dur(ring: &mut VecDeque<(Series, i64)>, series: Series, dt_ms: i64) {
    ring.push_back((series, dt_ms));
    while ring.len() > DURATION_CAP {
        ring.pop_front();
    }
}

/// Counts entries newer than `now - window_ms`, optionally filtered by series.
fn count_since(
    ring: &VecDeque<(TimestampMs, Series)>,
    now: TimestampMs,
    window_ms: i64,
    series: Option<Series>,
) -> u64 {
    let cutoff = now.as_millis() - window_ms;
    let n = ring
        .iter()
        .filter(|(ts, s)| ts.as_millis() >= cutoff && series.is_none_or(|f| *s == f))
        .count();
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// The median of a duration ring, optionally filtered by series; `None` if no
/// matching sample. Even counts average the two central samples.
fn median_dur(ring: &VecDeque<(Series, i64)>, series: Option<Series>) -> Option<i64> {
    let mut v: Vec<i64> = ring
        .iter()
        .filter(|(s, _)| series.is_none_or(|f| *s == f))
        .map(|(_, d)| *d)
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    })
}

/// The `q`-quantile (`q` in `[0, 1]`) of a duration ring, optionally filtered by
/// series; `None` if no matching sample. Nearest-rank on the sorted samples.
fn percentile_dur(ring: &VecDeque<(Series, i64)>, series: Option<Series>, q: f64) -> Option<i64> {
    let mut v: Vec<i64> = ring
        .iter()
        .filter(|(s, _)| series.is_none_or(|f| *s == f))
        .map(|(_, d)| *d)
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let n = v.len();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "sample count is small; nearest-rank index is well within range"
    )]
    let idx = ((q * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    Some(v[idx])
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;
    use core_types::{
        Asset, OrderId, Price, RoundDir, Size, TickSize, TokenId, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    const T0: i64 = 1_781_000_000_000;

    fn series(asset: Asset) -> Series {
        Series {
            asset,
            duration: WindowDuration::M5,
        }
    }

    fn window(asset: Asset) -> WindowId {
        WindowId {
            series: series(asset),
            open_time: TimestampMs::from_millis(T0),
        }
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn update(id: &str, asset: Asset, side: Side, state: OrderState) -> OrderUpdate {
        OrderUpdate {
            order_id: OrderId::new(id).expect("id"),
            window: window(asset),
            token_id: TokenId::new("1").expect("tok"),
            side,
            state,
            price: Price::quantize(dec!(0.5), TickSize::T001, RoundDir::Down).expect("px"),
            original_size: Size::new(dec!(10)).expect("sz"),
            filled_size: Size::ZERO,
            reject_reason: None,
            ts_venue: None,
            ts_local: ts(T0),
        }
    }

    fn fill(asset: Asset, ms: i64) -> Fill {
        Fill {
            order_id: OrderId::new("o").expect("id"),
            trade_id: None,
            window: window(asset),
            token_id: TokenId::new("1").expect("tok"),
            outcome: core_types::Outcome::Up,
            side: Side::Buy,
            price: Price::quantize(dec!(0.5), TickSize::T001, RoundDir::Down).expect("px"),
            size: Size::new(dec!(10)).expect("sz"),
            liquidity: core_types::Liquidity::Maker,
            fee: core_types::Dollars::ZERO,
            ts_venue: ts(ms),
            ts_local: ts(ms),
        }
    }

    #[test]
    fn last_minute_counts_respect_the_window() {
        let mut a = OrderActivity::new();
        let now = ts(T0 + 1_000_000);
        // Three placements inside the last minute, one 90s ago.
        a.on_order(&update("a", Asset::Btc, Side::Buy, OrderState::Open), now);
        a.on_order(&update("b", Asset::Btc, Side::Buy, OrderState::Open), now);
        a.on_order(&update("c", Asset::Btc, Side::Buy, OrderState::Open), now);
        a.on_order(
            &update("d", Asset::Btc, Side::Buy, OrderState::Open),
            ts(now.as_millis() - 90_000),
        );
        assert_eq!(a.placed_since(now, 60_000, None), 3);
        assert_eq!(a.placed_since(now, 120_000, None), 4);
    }

    #[test]
    fn placement_counted_once_across_pendingnew_then_open() {
        let mut a = OrderActivity::new();
        let now = ts(T0);
        a.on_order(
            &update("a", Asset::Btc, Side::Buy, OrderState::PendingNew),
            now,
        );
        a.on_order(&update("a", Asset::Btc, Side::Buy, OrderState::Open), now);
        assert_eq!(a.placed_since(now, 60_000, None), 1);
        assert_eq!(a.resting(None), 1);
    }

    #[test]
    fn resting_count_tracks_working_orders() {
        let mut a = OrderActivity::new();
        let now = ts(T0);
        a.on_order(&update("a", Asset::Btc, Side::Buy, OrderState::Open), now);
        a.on_order(&update("b", Asset::Btc, Side::Buy, OrderState::Open), now);
        assert_eq!(a.resting(None), 2);
        a.on_order(
            &update("a", Asset::Btc, Side::Buy, OrderState::Canceled),
            now,
        );
        assert_eq!(a.resting(None), 1);
        assert_eq!(a.cancelled_since(now, 60_000, None), 1);
    }

    #[test]
    fn time_to_cancel_is_observed_in_live_only() {
        let mut a = OrderActivity::new();
        // Live: Open → PendingCancel@t0 → Canceled@t0+50.
        for (i, delay) in [50_i64, 30, 70].into_iter().enumerate() {
            let id = format!("o{i}");
            a.on_order(
                &update(&id, Asset::Btc, Side::Buy, OrderState::Open),
                ts(T0),
            );
            a.on_order(
                &update(&id, Asset::Btc, Side::Buy, OrderState::PendingCancel),
                ts(T0 + 1000),
            );
            a.on_order(
                &update(&id, Asset::Btc, Side::Buy, OrderState::Canceled),
                ts(T0 + 1000 + delay),
            );
        }
        // Median of [50, 30, 70] = 50.
        assert_eq!(a.median_time_to_cancel_ms(None), Some(50));
    }

    #[test]
    fn p95_time_to_cancel_uses_nearest_rank_and_is_none_in_paper() {
        let mut a = OrderActivity::new();
        // Live round-trips: 20 samples 10..=200ms.
        for i in 1..=20 {
            let id = format!("o{i}");
            a.on_order(
                &update(&id, Asset::Btc, Side::Buy, OrderState::Open),
                ts(T0),
            );
            a.on_order(
                &update(&id, Asset::Btc, Side::Buy, OrderState::PendingCancel),
                ts(T0 + 1000),
            );
            a.on_order(
                &update(&id, Asset::Btc, Side::Buy, OrderState::Canceled),
                ts(T0 + 1000 + i * 10),
            );
        }
        // ceil(0.95*20)=19 → index 18 → 19th value = 190ms.
        assert_eq!(a.p95_time_to_cancel_ms(None), Some(190));
        // Paper (no PendingCancel) → None.
        let mut p = OrderActivity::new();
        p.on_order(
            &update("o", Asset::Btc, Side::Buy, OrderState::Open),
            ts(T0),
        );
        p.on_order(
            &update("o", Asset::Btc, Side::Buy, OrderState::Canceled),
            ts(T0 + 500),
        );
        assert_eq!(p.p95_time_to_cancel_ms(None), None);
    }

    #[test]
    fn paper_has_no_pending_cancel_so_time_to_cancel_is_none() {
        let mut a = OrderActivity::new();
        // Paper: Open → Canceled directly, no PendingCancel.
        a.on_order(
            &update("o", Asset::Btc, Side::Buy, OrderState::Open),
            ts(T0),
        );
        a.on_order(
            &update("o", Asset::Btc, Side::Buy, OrderState::Canceled),
            ts(T0 + 500),
        );
        assert_eq!(a.median_time_to_cancel_ms(None), None);
    }

    #[test]
    fn time_to_replace_pairs_a_cancel_with_the_next_placement_on_the_slot() {
        let mut a = OrderActivity::new();
        // Slot (BTC window, Buy): cancel then replace 100ms later.
        a.on_order(
            &update("a", Asset::Btc, Side::Buy, OrderState::Open),
            ts(T0),
        );
        a.on_order(
            &update("a", Asset::Btc, Side::Buy, OrderState::Canceled),
            ts(T0 + 1000),
        );
        a.on_order(
            &update("b", Asset::Btc, Side::Buy, OrderState::Open),
            ts(T0 + 1100),
        );
        // A second slot: cancel then replace 300ms later.
        a.on_order(
            &update("c", Asset::Btc, Side::Sell, OrderState::Open),
            ts(T0),
        );
        a.on_order(
            &update("c", Asset::Btc, Side::Sell, OrderState::Canceled),
            ts(T0 + 2000),
        );
        a.on_order(
            &update("d", Asset::Btc, Side::Sell, OrderState::Open),
            ts(T0 + 2300),
        );
        // Median of [100, 300] = 200.
        assert_eq!(a.median_time_to_replace_ms(None), Some(200));
    }

    #[test]
    fn replace_pairing_respects_the_cap() {
        let mut a = OrderActivity::new();
        a.on_order(
            &update("a", Asset::Btc, Side::Buy, OrderState::Open),
            ts(T0),
        );
        a.on_order(
            &update("a", Asset::Btc, Side::Buy, OrderState::Canceled),
            ts(T0 + 1000),
        );
        // Replacement 10s later — beyond the 5s cap → not a replace.
        a.on_order(
            &update("b", Asset::Btc, Side::Buy, OrderState::Open),
            ts(T0 + 11_000),
        );
        assert_eq!(a.median_time_to_replace_ms(None), None);
    }

    #[test]
    fn queries_filter_by_series() {
        let mut a = OrderActivity::new();
        let now = ts(T0);
        a.on_order(&update("a", Asset::Btc, Side::Buy, OrderState::Open), now);
        a.on_order(&update("b", Asset::Eth, Side::Buy, OrderState::Open), now);
        a.on_fill(&fill(Asset::Btc, now.as_millis()), now);
        assert_eq!(a.placed_since(now, 60_000, Some(series(Asset::Btc))), 1);
        assert_eq!(a.placed_since(now, 60_000, Some(series(Asset::Eth))), 1);
        assert_eq!(a.placed_since(now, 60_000, None), 2);
        assert_eq!(a.resting(Some(series(Asset::Btc))), 1);
        assert_eq!(a.fills_since(now, 60_000, Some(series(Asset::Btc))), 1);
        assert_eq!(a.fills_since(now, 60_000, Some(series(Asset::Eth))), 0);
    }
}
