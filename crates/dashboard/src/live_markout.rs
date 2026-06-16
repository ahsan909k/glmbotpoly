//! Dashboard-side live 5-second markout for the fills blotter.
//!
//! The analytics [`MarkoutEngine`](analytics::MarkoutEngine) only finalises
//! markouts when a window resolves (so a journal replay reproduces them
//! bit-for-bit) — too late to colour a *live* blotter. This tracker matures each
//! **maker** fill's 5-second markout as soon as 5 s of wall time have passed, so
//! the operator sees adverse selection in near-real time.
//!
//! It reuses the analytics markout convention via the shared
//! [`fair_of`](analytics::fair_of) / [`position_sign`](analytics::position_sign)
//! helpers; only the *policy* differs — a single 5 s horizon and **no clamp to a
//! settled outcome** (a fill within 5 s of close marks out against the last live
//! fair, best-effort). The analytics value computed at resolution stays canonical
//! for the series-comparison table. A maker fill with no model anchor at or
//! before it is dropped (NoAnchor, matching analytics), shown as `—` not pending.
//!
//! Bounded for 24/7: the per-window fair rings live on the shared view and are
//! dropped when their window is pruned; the per-mode pending/resolved sets are
//! capped (oldest-evicted), sized to the fills ring so any displayed fill keeps
//! its markout.

use std::collections::{BTreeMap, HashMap, VecDeque};

use analytics::{fair_of, position_sign};
use core_types::{Fill, Liquidity, OrderId, Outcome, Side, TimestampMs, WindowId};

/// Model-fair snapshots retained per window so a fill can be marked out.
pub(crate) const LIVE_MARKOUT_RING_CAP: usize = 4_096;
/// The markout horizon (ms).
const HORIZON_MS: i64 = 5_000;

/// A bounded `(ts, fair_up)` ring for one window's model-fair history.
#[derive(Debug, Clone, Default)]
pub(crate) struct FairRing {
    buf: VecDeque<(TimestampMs, f64)>,
}

impl FairRing {
    /// Records a model-fair snapshot, dropping the oldest past the cap.
    pub(crate) fn push(&mut self, ts: TimestampMs, fair_up: f64) {
        if self.buf.len() == LIVE_MARKOUT_RING_CAP {
            self.buf.pop_front();
        }
        self.buf.push_back((ts, fair_up));
    }

    /// Latest `fair_up` at or before `deadline` (last-observation-carried-forward).
    fn locf(&self, deadline: TimestampMs) -> Option<f64> {
        self.buf
            .iter()
            .rev()
            .find(|(ts, _)| ts.as_millis() <= deadline.as_millis())
            .map(|(_, fair)| *fair)
    }
}

/// Identifies a fill for markout lookup (robust to a missing trade id).
type FillKey = (OrderId, Option<String>, i64);

fn key(f: &Fill) -> FillKey {
    (
        f.order_id.clone(),
        f.trade_id.clone(),
        f.ts_venue.as_millis(),
    )
}

/// A maker fill awaiting its 5 s deadline.
#[derive(Debug, Clone)]
struct PendingFill {
    key: FillKey,
    window: WindowId,
    outcome: Outcome,
    side: Side,
    ts_fill: TimestampMs,
}

/// Per-mode live-markout state: maker fills awaiting their 5 s deadline, plus the
/// resolved markouts (`Some(v)` matured, `None` NoAnchor-dropped), capped and
/// evicted oldest-first.
#[derive(Debug)]
pub(crate) struct LiveMarkout {
    pending: VecDeque<PendingFill>,
    matured: HashMap<FillKey, Option<f64>>,
    order: VecDeque<FillKey>,
    cap: usize,
}

impl LiveMarkout {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            matured: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    /// Registers a fill. Only Maker fills are markout subjects (§10); takers and
    /// late-window takes are not adverse-selection candidates.
    pub(crate) fn on_fill(&mut self, f: &Fill) {
        if f.liquidity != Liquidity::Maker {
            return;
        }
        if self.pending.len() >= self.cap {
            self.pending.pop_front();
        }
        self.pending.push_back(PendingFill {
            key: key(f),
            window: f.window,
            outcome: f.outcome,
            side: f.side,
            ts_fill: f.ts_venue,
        });
    }

    /// Matures every pending fill whose 5 s deadline has passed, using the shared
    /// per-window fair rings (LOCF). Order-independent: a not-yet-due fill is
    /// retained.
    pub(crate) fn mature(&mut self, rings: &BTreeMap<WindowId, FairRing>, now: TimestampMs) {
        let now_ms = now.as_millis();
        let mut resolved: Vec<(FillKey, Option<f64>)> = Vec::new();
        self.pending.retain(|p| {
            if now_ms < p.ts_fill.as_millis() + HORIZON_MS {
                return true;
            }
            let value = rings.get(&p.window).and_then(|ring| {
                let anchor = ring.locf(p.ts_fill)?; // None => NoAnchor (drop)
                let deadline = TimestampMs::from_millis(p.ts_fill.as_millis() + HORIZON_MS);
                let at_h = ring.locf(deadline).unwrap_or(anchor);
                Some(
                    position_sign(p.side) * (fair_of(p.outcome, at_h) - fair_of(p.outcome, anchor)),
                )
            });
            resolved.push((p.key.clone(), value));
            false
        });
        for (k, v) in resolved {
            self.record(k, v);
        }
    }

    fn record(&mut self, k: FillKey, v: Option<f64>) {
        if self.matured.insert(k.clone(), v).is_none() {
            self.order.push_back(k);
            while self.order.len() > self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.matured.remove(&old);
                }
            }
        }
    }

    /// `(markout_5s, pending)` for a fill: `(Some, false)` matured,
    /// `(None, false)` taker or NoAnchor-dropped, `(None, true)` maker awaiting 5 s.
    pub(crate) fn markout_for(&self, f: &Fill) -> (Option<f64>, bool) {
        if f.liquidity != Liquidity::Maker {
            return (None, false);
        }
        match self.matured.get(&key(f)) {
            Some(v) => (*v, false),
            None => (None, true),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use core_types::{Asset, Dollars, Price, Series, Size, TokenId, WindowDuration};
    use rust_decimal::dec;

    const OPEN_MS: i64 = 1_781_000_000_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn fill(outcome: Outcome, side: Side, liquidity: Liquidity, ts_fill: i64, id: &str) -> Fill {
        Fill {
            order_id: OrderId::new(id).unwrap(),
            trade_id: Some(format!("t-{id}")),
            window: window(),
            token_id: TokenId::new(if outcome == Outcome::Up { "1" } else { "2" }).unwrap(),
            outcome,
            side,
            price: Price::try_from(dec!(0.5)).unwrap(),
            size: Size::new(dec!(10)).unwrap(),
            liquidity,
            fee: Dollars::ZERO,
            ts_venue: ts(ts_fill),
            ts_local: ts(ts_fill),
        }
    }

    fn rings(snaps: &[(i64, f64)]) -> BTreeMap<WindowId, FairRing> {
        let mut r = FairRing::default();
        for (ms, fair) in snaps {
            r.push(ts(*ms), *fair);
        }
        let mut m = BTreeMap::new();
        m.insert(window(), r);
        m
    }

    #[test]
    fn s5_matures_after_5s() {
        let mut lm = LiveMarkout::new(16);
        let f = fill(Outcome::Up, Side::Buy, Liquidity::Maker, OPEN_MS, "a");
        lm.on_fill(&f);
        let r = rings(&[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.55)]);
        lm.mature(&r, ts(OPEN_MS + 5_000));
        let (m, pending) = lm.markout_for(&f);
        assert!(!pending);
        assert!((m.expect("matured") - 0.05).abs() < 1e-12);
    }

    #[test]
    fn pending_before_5s() {
        let mut lm = LiveMarkout::new(16);
        let f = fill(Outcome::Up, Side::Buy, Liquidity::Maker, OPEN_MS, "a");
        lm.on_fill(&f);
        let r = rings(&[(OPEN_MS, 0.50)]);
        lm.mature(&r, ts(OPEN_MS + 4_999));
        let (m, pending) = lm.markout_for(&f);
        assert!(pending);
        assert!(m.is_none());
    }

    #[test]
    fn no_anchor_dropped() {
        let mut lm = LiveMarkout::new(16);
        let f = fill(Outcome::Up, Side::Buy, Liquidity::Maker, OPEN_MS, "a");
        lm.on_fill(&f);
        // First snapshot is AFTER the fill — no anchor at or before it.
        let r = rings(&[(OPEN_MS + 1_000, 0.5)]);
        lm.mature(&r, ts(OPEN_MS + 5_000));
        let (m, pending) = lm.markout_for(&f);
        assert!(!pending); // resolved...
        assert!(m.is_none()); // ...as dropped (shown as —)
    }

    #[test]
    fn taker_not_subject() {
        let mut lm = LiveMarkout::new(16);
        let f = fill(Outcome::Up, Side::Buy, Liquidity::Taker, OPEN_MS, "a");
        lm.on_fill(&f);
        let r = rings(&[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.55)]);
        lm.mature(&r, ts(OPEN_MS + 5_000));
        let (m, pending) = lm.markout_for(&f);
        assert!(!pending);
        assert!(m.is_none());
    }

    #[test]
    fn sign_flip_on_sell() {
        let mut lm = LiveMarkout::new(16);
        // SELL Up; fair rises 0.50→0.55 → markout −0.05.
        let f = fill(Outcome::Up, Side::Sell, Liquidity::Maker, OPEN_MS, "a");
        lm.on_fill(&f);
        let r = rings(&[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.55)]);
        lm.mature(&r, ts(OPEN_MS + 5_000));
        assert!((lm.markout_for(&f).0.expect("matured") - (-0.05)).abs() < 1e-12);
    }

    #[test]
    fn down_uses_one_minus_p_up() {
        let mut lm = LiveMarkout::new(16);
        // Bought Down; p_up falls 0.50→0.40 so fair_Down rises 0.50→0.60 → +0.10.
        let f = fill(Outcome::Down, Side::Buy, Liquidity::Maker, OPEN_MS, "a");
        lm.on_fill(&f);
        let r = rings(&[(OPEN_MS, 0.50), (OPEN_MS + 5_000, 0.40)]);
        lm.mature(&r, ts(OPEN_MS + 5_000));
        assert!((lm.markout_for(&f).0.expect("matured") - 0.10).abs() < 1e-12);
    }

    #[test]
    fn ring_cap_bounds_memory() {
        let mut ring = FairRing::default();
        for i in 0..(LIVE_MARKOUT_RING_CAP as i64 + 100) {
            ring.push(ts(OPEN_MS + i), 0.5);
        }
        assert_eq!(ring.buf.len(), LIVE_MARKOUT_RING_CAP);
        // Resolved set is capped too (oldest evicted).
        let mut lm = LiveMarkout::new(2);
        let r = rings(&[(OPEN_MS, 0.5), (OPEN_MS + 5_000, 0.5)]);
        for i in 0..5 {
            let f = fill(
                Outcome::Up,
                Side::Buy,
                Liquidity::Maker,
                OPEN_MS,
                &format!("o{i}"),
            );
            lm.on_fill(&f);
            lm.mature(&r, ts(OPEN_MS + 5_000));
        }
        assert!(lm.matured.len() <= 2);
        assert!(lm.order.len() <= 2);
    }
}
