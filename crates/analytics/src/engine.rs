//! The top-level analytics fold: consumes bus events, produces effects, and
//! rebuilds bit-for-bit from a journal replay (CLAUDE.md §3/§9).
//!
//! Modelled on `engine::InventoryManager`: pure, sans-IO, event-time-driven, and
//! never iterating a `HashMap` to produce effects — so [`Analytics::on_event`]
//! over the live bus and [`Analytics::rebuild`] over a journal replay yield
//! identical state and an identical effect stream.
//!
//! ## Lifecycle, per window
//!
//! - `Event::Window` caches the market's fee/rebate rate and close time.
//! - `Event::Fill` folds attribution (the maker-rebate base + counts) and, for a
//!   Maker fill, registers a passive markout.
//! - `Event::Model` (window-tagged) feeds the window's model-fair ring.
//! - `Event::Window { Resolved }` marks out every passive fill (terminal outcome
//!   known) and drops the markout ring; for an untraded window it frees the
//!   cached metadata immediately (no settlement will follow).
//! - `Event::Settlement` decomposes the window's PnL, folds it into the daily
//!   roll-ups, feeds the 5s markouts to the series' adverse-selection monitor,
//!   and frees the window's per-window state (bounded memory for 24/7).

use std::collections::HashMap;

use core_types::{
    Decimal, Dollars, Event, Liquidity, Mode, Series, SettlementSummary, TimestampMs, WindowId,
    WindowLifecycle,
};

use crate::attribution::{WindowAttrib, WindowAttribution};
use crate::datekey::DayKey;
use crate::health::{AdverseSelectionMonitor, AdverseSelectionState, SeriesHealth};
use crate::markout::{FillMarkout, MarkoutEngine};
use crate::params::AnalyticsParams;
use crate::rollup::{
    ComparisonWindow, DailyRollup, RollupStore, SeriesComparison, SeriesComparisonRow, SortColumn,
};

/// Cached per-window market metadata needed after the fills arrive.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MarketMeta {
    fee_rate: Decimal,
    rebate_rate: Decimal,
    close_time: TimestampMs,
}

/// What the analytics fold emits per event, for the dashboard and risk panel.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalyticsEffect {
    /// A passive fill's markouts finalised at its window's resolution.
    FillMarkout(FillMarkout),
    /// A window's PnL attribution closed at settlement.
    WindowSettled(WindowAttribution),
    /// A series' adverse-selection state changed (latched).
    SeriesHealth(SeriesHealth),
}

/// The per-mode analytics engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Analytics {
    mode: Mode,
    params: AnalyticsParams,
    markout: MarkoutEngine,
    windows: HashMap<WindowId, WindowAttrib>,
    market_meta: HashMap<WindowId, MarketMeta>,
    resolved_markouts: HashMap<WindowId, Vec<FillMarkout>>,
    rollups: RollupStore,
    adverse: HashMap<Series, AdverseSelectionMonitor>,
}

impl Analytics {
    /// A fresh engine for one session mode.
    #[must_use]
    pub fn new(mode: Mode, params: AnalyticsParams) -> Self {
        Self {
            mode,
            params,
            markout: MarkoutEngine::new(params.snapshot_ring_max),
            windows: HashMap::new(),
            market_meta: HashMap::new(),
            resolved_markouts: HashMap::new(),
            rollups: RollupStore::new(),
            adverse: HashMap::new(),
        }
    }

    /// Folds one bus event, returning the effects to publish.
    pub fn on_event(&mut self, event: &Event) -> Vec<AnalyticsEffect> {
        match event {
            Event::Window { market, lifecycle } => {
                self.market_meta.insert(
                    market.window,
                    MarketMeta {
                        fee_rate: market.fees.rate,
                        rebate_rate: market.fees.rebate_rate,
                        close_time: market.close_time,
                    },
                );
                if let WindowLifecycle::Resolved { outcome } = lifecycle {
                    let fms = self
                        .markout
                        .on_resolved(market.window, *outcome, market.close_time);
                    let effects = fms
                        .iter()
                        .cloned()
                        .map(AnalyticsEffect::FillMarkout)
                        .collect();
                    if self.windows.contains_key(&market.window) {
                        // Traded → a settlement will follow; keep the markouts and
                        // the metadata (for the rebate rate) until then.
                        self.resolved_markouts.insert(market.window, fms);
                    } else {
                        // Untraded → no settlement; free the metadata now.
                        self.market_meta.remove(&market.window);
                    }
                    return effects;
                }
                Vec::new()
            }
            Event::Fill(fill) => {
                let fee_rate = self
                    .market_meta
                    .get(&fill.window)
                    .map_or(self.params.default_fee_rate, |m| m.fee_rate);
                self.windows
                    .entry(fill.window)
                    .or_insert_with(|| WindowAttrib::new(fill.window))
                    .apply_fill(fill, fee_rate);
                if fill.liquidity == Liquidity::Maker {
                    self.markout
                        .on_fill(fill.window, fill.outcome, fill.side, fill.ts_venue);
                }
                Vec::new()
            }
            Event::Model(snap) => {
                if let Some(window) = snap.window {
                    self.markout.on_snapshot(window, snap.ts, snap.p_up);
                }
                Vec::new()
            }
            Event::Settlement(summary) => self.settle(summary),
            _ => Vec::new(),
        }
    }

    /// Closes a settled window: attribution → roll-up → adverse-selection feed,
    /// then frees the window's per-window state.
    fn settle(&mut self, summary: &SettlementSummary) -> Vec<AnalyticsEffect> {
        let attrib = self.windows.remove(&summary.window);
        let rebate_rate = self
            .market_meta
            .get(&summary.window)
            .map_or(self.params.default_rebate_rate, |m| m.rebate_rate);
        let (rebate_base, maker_fills, taker_fills, taker_notional) =
            attrib
                .as_ref()
                .map_or((Dollars::ZERO, 0, 0, Dollars::ZERO), |a| {
                    (
                        a.maker_rebate_base(),
                        a.maker_fills(),
                        a.taker_fills(),
                        a.taker_notional(),
                    )
                });
        let attribution = WindowAttribution::from_summary(
            summary,
            self.mode,
            rebate_base,
            rebate_rate,
            maker_fills,
            taker_fills,
            taker_notional,
        );

        let fms = self
            .resolved_markouts
            .remove(&summary.window)
            .unwrap_or_default();
        let s5s: Vec<f64> = fms.iter().filter_map(|f| f.s5.map(|p| p.value)).collect();
        let s30s: Vec<f64> = fms.iter().filter_map(|f| f.s30.map(|p| p.value)).collect();

        self.rollups.fold_markouts(&attribution, &s5s, &s30s);

        let series = summary.window.series;
        let monitor = self
            .adverse
            .entry(series)
            .or_insert_with(|| AdverseSelectionMonitor::new(self.params.adverse));

        let mut effects = vec![AnalyticsEffect::WindowSettled(attribution)];
        for s5 in &s5s {
            if let Some((from, to)) = monitor.observe_5s(*s5) {
                effects.push(AnalyticsEffect::SeriesHealth(SeriesHealth {
                    series,
                    mode: self.mode,
                    from,
                    to,
                }));
            }
        }

        // The window is fully accounted for: free its metadata.
        self.market_meta.remove(&summary.window);
        effects
    }

    /// Rebuilds the engine by folding journaled events (discarding effects). The
    /// fold is deterministic and event-time-driven, so this reproduces the live
    /// engine exactly — the §3/§9 restart-rebuild guarantee.
    #[must_use]
    pub fn rebuild<'a>(
        mode: Mode,
        params: AnalyticsParams,
        events: impl IntoIterator<Item = &'a Event>,
    ) -> Self {
        let mut analytics = Self::new(mode, params);
        for event in events {
            let _ = analytics.on_event(event);
        }
        analytics
    }

    /// The session mode this engine reports for.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The daily roll-up for a `(series, day)`, if any.
    #[must_use]
    pub fn daily(&self, series: Series, day: DayKey) -> Option<&DailyRollup> {
        self.rollups.daily(series, day)
    }

    /// The §10 series-comparison rows over **all** days, one per series that has
    /// traded, in series order, with adverse-selection health and minimum-sample
    /// warnings attached. For a time-windowed, best-first-sorted view (the
    /// dashboard's query), use [`series_comparison_over`](Self::series_comparison_over).
    #[must_use]
    pub fn series_comparison(&self) -> Vec<SeriesComparisonRow> {
        self.rollups
            .series_aggregates()
            .into_iter()
            .map(|agg| {
                let health = self.series_health(agg.series);
                agg.to_row(
                    self.params.min_sample_windows,
                    self.params.taker_budget_per_window,
                    health,
                )
            })
            .collect()
    }

    /// The §10 series-comparison table for a time-window selection (today / last
    /// N days / all), pre-sorted best-first by net PnL — the dashboard's query.
    /// `today` is supplied by the caller (the crate is sans-IO; the bot passes
    /// `DayKey::from_ts(now)`). The `min_sample_warning` reflects the **selected**
    /// window's traded count; `health` is the current rolling adverse-selection
    /// gate (independent of the calendar selection).
    #[must_use]
    pub fn series_comparison_over(
        &self,
        window: ComparisonWindow,
        today: DayKey,
    ) -> SeriesComparison {
        let rows = self
            .rollups
            .series_aggregates_over(window, today)
            .into_iter()
            .map(|agg| {
                let health = self.series_health(agg.series);
                agg.to_row(
                    self.params.min_sample_windows,
                    self.params.taker_budget_per_window,
                    health,
                )
            })
            .collect();
        SeriesComparison::new(window, today, rows, SortColumn::default())
    }

    /// The adverse-selection state for a series (`InsufficientSample` until the
    /// series has produced any 5s markouts).
    #[must_use]
    pub fn series_health(&self, series: Series) -> AdverseSelectionState {
        self.adverse
            .get(&series)
            .map_or(AdverseSelectionState::InsufficientSample, |m| m.state())
    }

    /// The underlying roll-up store (dashboard read access).
    #[must_use]
    pub fn rollups(&self) -> &RollupStore {
        &self.rollups
    }

    /// Passive fills dropped for lacking a model anchor (diagnostic).
    #[must_use]
    pub fn no_anchor_dropped(&self) -> u64 {
        self.markout.no_anchor_dropped()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use core_types::{
        AnchorSource, Asset, ConditionId, DurationMs, FeeParams, Fill, InputAges, MarketInfo,
        ModelHealth, ModelHealthReason, ModelSnapshot, OrderId, Outcome, Price, ResolutionSource,
        RoundDir, Series as Ser, Side, Size, TickSize, TokenId, TokenPair, WindowDuration,
    };
    use rust_decimal::dec;

    const OPEN_MS: i64 = 1_781_481_600_000; // 2026-06-15T00:00:00Z

    fn series() -> Ser {
        Ser {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        }
    }

    fn window() -> WindowId {
        window_at(OPEN_MS)
    }

    fn window_at(open_ms: i64) -> WindowId {
        WindowId {
            series: series(),
            open_time: TimestampMs::from_millis(open_ms),
        }
    }

    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }

    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }

    fn market() -> Arc<MarketInfo> {
        market_at(OPEN_MS)
    }

    fn market_at(open_ms: i64) -> Arc<MarketInfo> {
        Arc::new(MarketInfo {
            window: window_at(open_ms),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: TimestampMs::from_millis(open_ms + 300_000),
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

    fn window_event(lifecycle: WindowLifecycle) -> Event {
        Event::Window {
            market: market(),
            lifecycle,
        }
    }

    fn maker_fill(outcome: Outcome, price: Decimal, size: Decimal, ts: i64) -> Event {
        fill_at(window(), outcome, price, size, Liquidity::Maker, ts)
    }

    fn fill_at(
        window: WindowId,
        outcome: Outcome,
        price: Decimal,
        size: Decimal,
        liq: Liquidity,
        ts: i64,
    ) -> Event {
        let fee = if liq == Liquidity::Taker {
            core_types::taker_fee(sz(size), dec!(0.07), px(price))
        } else {
            Dollars::ZERO
        };
        Event::Fill(Arc::new(Fill {
            order_id: OrderId::new("o").unwrap(),
            trade_id: None,
            window,
            token_id: TokenId::new(if outcome == Outcome::Up { "1" } else { "2" }).unwrap(),
            outcome,
            side: Side::Buy,
            price: px(price),
            size: sz(size),
            liquidity: liq,
            fee,
            ts_venue: TimestampMs::from_millis(ts),
            ts_local: TimestampMs::from_millis(ts),
        }))
    }

    fn model(p_up: f64, ts: i64) -> Event {
        model_at(window(), p_up, ts)
    }

    fn model_at(window: WindowId, p_up: f64, ts: i64) -> Event {
        Event::Model(ModelSnapshot {
            asset: Asset::Btc,
            window: Some(window),
            p_up,
            z: 0.0,
            sigma_1s: 0.0005,
            sigma_tau: 0.01,
            basis: 0.0,
            anchor: AnchorSource::Chainlink,
            health: ModelHealth::Ready,
            reason: ModelHealthReason::Nominal,
            input_ages: InputAges {
                chainlink: DurationMs::from_millis(100),
                binance: DurationMs::from_millis(100),
            },
            ts: TimestampMs::from_millis(ts),
        })
    }

    fn settlement(realized: Decimal, up: (Decimal, Decimal), down: (Decimal, Decimal)) -> Event {
        settlement_at(OPEN_MS, realized, up, down, Dollars::ZERO)
    }

    fn settlement_at(
        open_ms: i64,
        realized: Decimal,
        up: (Decimal, Decimal),
        down: (Decimal, Decimal),
        fees: Dollars,
    ) -> Event {
        let summary = SettlementSummary::close(
            window_at(open_ms),
            Outcome::Up,
            core_types::SideInventory {
                shares: sz(up.0),
                cost: Dollars::new(up.1),
            },
            core_types::SideInventory {
                shares: sz(down.0),
                cost: Dollars::new(down.1),
            },
            Size::ZERO,
            fees,
            Dollars::new(realized),
            TimestampMs::from_millis(open_ms + 300_000),
        );
        Event::Settlement(Arc::new(summary))
    }

    /// A balanced two-maker-fill session at a given UTC open time, resolving Up.
    fn balanced_session_at(open_ms: i64) -> Vec<Event> {
        let w = window_at(open_ms);
        vec![
            Event::Window {
                market: market_at(open_ms),
                lifecycle: WindowLifecycle::Open,
            },
            model_at(w, 0.50, open_ms),
            fill_at(
                w,
                Outcome::Up,
                dec!(0.48),
                dec!(100),
                Liquidity::Maker,
                open_ms + 10,
            ),
            fill_at(
                w,
                Outcome::Down,
                dec!(0.49),
                dec!(100),
                Liquidity::Maker,
                open_ms + 20,
            ),
            model_at(w, 0.55, open_ms + 5_000),
            Event::Window {
                market: market_at(open_ms),
                lifecycle: WindowLifecycle::Resolved {
                    outcome: Outcome::Up,
                },
            },
            settlement_at(
                open_ms,
                dec!(3),
                (dec!(100), dec!(48)),
                (dec!(100), dec!(49)),
                Dollars::ZERO,
            ),
        ]
    }

    /// A full balanced-book session: open, snapshots, two maker fills, snapshots
    /// crossing the markout deadlines, resolve Up, settle.
    fn session() -> Vec<Event> {
        vec![
            window_event(WindowLifecycle::Open),
            model(0.50, OPEN_MS),
            maker_fill(Outcome::Up, dec!(0.48), dec!(100), OPEN_MS + 10),
            maker_fill(Outcome::Down, dec!(0.49), dec!(100), OPEN_MS + 20),
            model(0.52, OPEN_MS + 1_000),  // +1s after the fills
            model(0.55, OPEN_MS + 5_000),  // +5s
            model(0.60, OPEN_MS + 30_000), // +30s
            window_event(WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            }),
            // realized = cash(-97) + payout(100 Up) = 3.
            settlement(dec!(3), (dec!(100), dec!(48)), (dec!(100), dec!(49))),
        ]
    }

    #[test]
    fn folds_a_session_into_markouts_and_attribution() {
        let mut a = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let effects: Vec<AnalyticsEffect> = session().iter().flat_map(|e| a.on_event(e)).collect();

        // Two FillMarkouts (at resolution) + one WindowSettled.
        let markouts: Vec<&FillMarkout> = effects
            .iter()
            .filter_map(|e| match e {
                AnalyticsEffect::FillMarkout(f) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(markouts.len(), 2);
        // Up fill bought at fair 0.50, fair rose to 0.55 at +5s → +0.05.
        let up = markouts.iter().find(|f| f.outcome == Outcome::Up).unwrap();
        assert!((up.s5.unwrap().value - 0.05).abs() < 1e-12);
        // Up fill won → expiry +0.50.
        assert!((up.expiry.unwrap().value - 0.50).abs() < 1e-12);

        let settled: Vec<&WindowAttribution> = effects
            .iter()
            .filter_map(|e| match e {
                AnalyticsEffect::WindowSettled(w) => Some(w),
                _ => None,
            })
            .collect();
        assert_eq!(settled.len(), 1);
        assert_eq!(settled[0].realized_pnl, Dollars::new(dec!(3)));
        assert_eq!(settled[0].trading_sum(), Dollars::new(dec!(3)));
        // Maker rebate base = taker_fee(100,0.07,0.48) + taker_fee(100,0.07,0.49).
        assert!(settled[0].estimated_rebate > Dollars::ZERO);

        // Series comparison row reflects the settled window.
        let rows = a.series_comparison();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].windows_traded, 1);
        assert_eq!(rows[0].net_pnl, Dollars::new(dec!(3)));
        assert_eq!(rows[0].maker_fills, 2);
        assert!(rows[0].min_sample_warning); // one window < default 30
    }

    #[test]
    fn rebuild_equals_live_and_effect_streams_match() {
        let events = session();
        let mut live = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let live_effects: Vec<AnalyticsEffect> =
            events.iter().flat_map(|e| live.on_event(e)).collect();

        let rebuilt = Analytics::rebuild(Mode::Paper, AnalyticsParams::default(), events.iter());
        assert_eq!(rebuilt, live, "rebuilt state must equal the live engine");

        let mut again = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let again_effects: Vec<AnalyticsEffect> =
            events.iter().flat_map(|e| again.on_event(e)).collect();
        assert_eq!(again_effects, live_effects, "effect streams must match");
    }

    #[test]
    fn untraded_window_frees_metadata_and_produces_nothing() {
        let mut a = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let _ = a.on_event(&window_event(WindowLifecycle::Open));
        let _ = a.on_event(&model(0.5, OPEN_MS));
        let effects = a.on_event(&window_event(WindowLifecycle::Resolved {
            outcome: Outcome::Up,
        }));
        assert!(effects.is_empty());
        // No window traded → no comparison rows, and the metadata was freed.
        assert!(a.series_comparison().is_empty());
        assert!(a.market_meta.is_empty());
    }

    #[test]
    fn maker_fill_without_a_snapshot_is_dropped_no_anchor() {
        let mut a = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let _ = a.on_event(&window_event(WindowLifecycle::Open));
        // Fill BEFORE any model snapshot — no anchor.
        let _ = a.on_event(&maker_fill(Outcome::Up, dec!(0.50), dec!(100), OPEN_MS));
        let _ = a.on_event(&model(0.5, OPEN_MS + 1_000));
        let resolve = a.on_event(&window_event(WindowLifecycle::Resolved {
            outcome: Outcome::Up,
        }));
        // The fill traded, so the window settles, but its markout was dropped.
        assert!(resolve.is_empty());
        assert_eq!(a.no_anchor_dropped(), 1);
        // Settlement still folds attribution.
        let settle = a.on_event(&settlement(
            dec!(50),
            (dec!(100), dec!(50)),
            (dec!(0), dec!(0)),
        ));
        assert!(matches!(settle[0], AnalyticsEffect::WindowSettled(_)));
    }

    /// The balanced session plus a taker BUY (40 Up @ 0.55) — exercises the
    /// taker-notional accumulator and the histogram on rebuild.
    fn session_with_taker() -> Vec<Event> {
        let w = window();
        vec![
            window_event(WindowLifecycle::Open),
            model(0.50, OPEN_MS),
            maker_fill(Outcome::Up, dec!(0.48), dec!(100), OPEN_MS + 10),
            maker_fill(Outcome::Down, dec!(0.49), dec!(100), OPEN_MS + 20),
            // Taker BUY: notional 40·0.55 = $22, fee 40·0.07·0.55·0.45 = 0.693.
            fill_at(
                w,
                Outcome::Up,
                dec!(0.55),
                dec!(40),
                Liquidity::Taker,
                OPEN_MS + 30,
            ),
            model(0.55, OPEN_MS + 5_000),
            window_event(WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            }),
            // Up 140 sh (cost 70), Down 100 sh (cost 49), fees 0.693.
            // cash = -(48+49+22) - 0.693 = -119.693; Up payout 140 → realized 20.307.
            settlement_at(
                OPEN_MS,
                dec!(20.307),
                (dec!(140), dec!(70)),
                (dec!(100), dec!(49)),
                Dollars::new(dec!(0.693)),
            ),
        ]
    }

    #[test]
    fn rebuild_equals_live_with_taker_and_distribution() {
        let events = session_with_taker();
        let mut live = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let live_effects: Vec<AnalyticsEffect> =
            events.iter().flat_map(|e| live.on_event(e)).collect();

        // State equality covers the new f64/array/Dollars DailyRollup fields.
        let rebuilt = Analytics::rebuild(Mode::Paper, AnalyticsParams::default(), events.iter());
        assert_eq!(rebuilt, live, "rebuilt state must equal the live engine");

        let mut again = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let again_effects: Vec<AnalyticsEffect> =
            events.iter().flat_map(|e| again.on_event(e)).collect();
        assert_eq!(again_effects, live_effects, "effect streams must match");

        // The row carries the taker notional and the maker/taker mix.
        let rows = live.series_comparison();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].maker_fills, 2);
        assert_eq!(rows[0].taker_fills, 1);
        assert_eq!(rows[0].taker_notional, Dollars::new(dec!(22)));
        assert!((rows[0].maker_fill_fraction - 2.0 / 3.0).abs() < 1e-12);
        // Fees paid is reported as a positive cost.
        assert_eq!(rows[0].fees_paid, Dollars::new(dec!(0.693)));

        // The dashboard query is bit-stable across instances (deterministic fold).
        let today = DayKey::from_ts(TimestampMs::from_millis(OPEN_MS));
        assert_eq!(
            live.series_comparison_over(ComparisonWindow::All, today),
            rebuilt.series_comparison_over(ComparisonWindow::All, today),
            "live and rebuilt query output must be identical"
        );
    }

    #[test]
    fn series_comparison_over_selects_by_day() {
        let mut a = Analytics::new(Mode::Paper, AnalyticsParams::default());
        let day0 = OPEN_MS;
        let day1 = OPEN_MS + 86_400_000; // next UTC day
        for e in balanced_session_at(day0) {
            let _ = a.on_event(&e);
        }
        for e in balanced_session_at(day1) {
            let _ = a.on_event(&e);
        }
        let today = DayKey::from_ts(TimestampMs::from_millis(day1));

        // Today (day1) → just the one window.
        let today_cmp = a.series_comparison_over(ComparisonWindow::Today, today);
        assert_eq!(today_cmp.window, ComparisonWindow::Today);
        assert_eq!(today_cmp.day, today);
        assert_eq!(today_cmp.default_sort, SortColumn::NetPnl);
        assert_eq!(today_cmp.rows.len(), 1);
        assert_eq!(today_cmp.rows[0].windows_traded, 1);
        assert_eq!(today_cmp.rows[0].net_pnl, Dollars::new(dec!(3)));

        // Last 7 days → both windows.
        let week = a.series_comparison_over(ComparisonWindow::LastDays(7), today);
        assert_eq!(week.rows[0].windows_traded, 2);
        assert_eq!(week.rows[0].net_pnl, Dollars::new(dec!(6)));

        // Legacy all-days, series-ordered view is unchanged in shape.
        let all = a.series_comparison();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].windows_traded, 2);

        // Health is the current monitor state regardless of the calendar window.
        let h = a.series_health(window().series);
        assert_eq!(today_cmp.rows[0].health, h);
        assert_eq!(week.rows[0].health, h);
    }
}
