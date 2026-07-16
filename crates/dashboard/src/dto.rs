//! Serializable response DTOs and the builders that project the state store
//! into them.
//!
//! The handlers contain no business logic: they lock the store, call a builder
//! here, unlock, and return the owned DTO as JSON. Most domain types are already
//! `Serialize` and embed directly; the three venue snapshot types
//! ([`Wallet`](venue_api::Wallet), [`PaperLedgerSnapshot`](venue_paper::PaperLedgerSnapshot),
//! [`RiskStateSnapshot`](engine::RiskStateSnapshot)) are not, so they get thin
//! `#[derive(Serialize)]` DTOs with `From` conversions.

use core_types::{
    Asset, BookLevel, BreakerKind, Decimal, Dollars, Fill, InventorySnapshot, Liquidity, Mode,
    ModelHealth, ModelHealthReason, ModelSnapshot, OrderId, OrderState, Outcome, Price,
    PriceSource, Series, Side, Size, TickKind, TickSize, TokenId, TopOfBook, WindowId,
    WindowLifecycle,
};
use engine::{FillDriver, RiskStateSnapshot};
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use venue_api::Wallet;
use venue_paper::PaperLedgerSnapshot;

use crate::state::DashboardData;
use analytics::{AdverseSelectionState, AnalyticsParams, ComparisonWindow, DayKey, SortColumn};

// ---- venue snapshot DTOs (the three non-Serialize types) -------------------

/// One outcome-token share balance.
#[derive(Debug, Clone, Serialize)]
pub struct TokenBalanceDto {
    /// The outcome token.
    pub token_id: TokenId,
    /// Shares held.
    pub size: Size,
}

/// Collateral and positions read from a venue.
#[derive(Debug, Clone, Serialize)]
pub struct WalletDto {
    /// Collateral available to trade.
    pub collateral_available: Dollars,
    /// Total collateral.
    pub collateral_total: Dollars,
    /// Per-token share balances.
    pub positions: Vec<TokenBalanceDto>,
}

impl From<&Wallet> for WalletDto {
    fn from(w: &Wallet) -> Self {
        Self {
            collateral_available: w.collateral_available,
            collateral_total: w.collateral_total,
            positions: w
                .positions
                .iter()
                .map(|p| TokenBalanceDto {
                    token_id: p.token_id.clone(),
                    size: p.size,
                })
                .collect(),
        }
    }
}

/// A signed per-token net position from the paper ledger.
#[derive(Debug, Clone, Serialize)]
pub struct PositionDto {
    /// The outcome token.
    pub token_id: TokenId,
    /// Signed net shares.
    pub net: Decimal,
}

/// The simulated **Taker Fee Rebate Program** projection (CLAUDE.md §7). This is
/// **report-only** — never part of paper cash/equity. The tier is set by 30-day
/// rolling weighted volume; the rebate is `tier% × taker fees`.
#[derive(Debug, Clone, Serialize)]
pub struct TakerRebateDto {
    /// Rolling 30-day taker weighted volume that sets the tier.
    pub wv_30d: Dollars,
    /// Current tier name (`"None"`, `"Bronze"`, … `"Obsidian"`).
    pub tier: String,
    /// Current rebate fraction (the tier's %).
    pub rebate_pct: Decimal,
    /// Simulated rebate accrued below the $1 floor.
    pub accrued: Dollars,
    /// Simulated rebate past the $1 floor (cumulative).
    pub credited: Dollars,
    /// Projected rebate/day = `tier% × last completed day's taker fees`.
    pub projected_per_day: Dollars,
    /// The next tier up (`null` at the top tier).
    pub next_tier: Option<String>,
    /// Additional 30-day wV needed to reach the next tier (`0` at the top).
    pub wv_to_next_tier: Dollars,
}

impl From<&PaperLedgerSnapshot> for TakerRebateDto {
    fn from(l: &PaperLedgerSnapshot) -> Self {
        Self {
            wv_30d: l.taker_wv_30d,
            tier: l.taker_tier.to_owned(),
            rebate_pct: l.taker_rebate_pct,
            accrued: l.taker_rebate_accrued,
            credited: l.taker_rebate_credited,
            // Estimate the daily run-rate from the last completed day's taker fees.
            projected_per_day: Dollars::new(
                l.taker_rebate_pct * l.last_day_taker_fees.as_decimal(),
            ),
            next_tier: l.taker_next_tier.map(str::to_owned),
            wv_to_next_tier: l.taker_wv_to_next,
        }
    }
}

/// The richer paper-ledger view with its income lines.
#[derive(Debug, Clone, Serialize)]
pub struct LedgerDto {
    /// Signed collateral cash.
    pub collateral: Dollars,
    /// Signed non-zero net positions.
    pub positions: Vec<PositionDto>,
    /// Cumulative taker fees paid.
    pub fees_paid: Dollars,
    /// Maker rebate accrued, not yet credited.
    pub rebate_accrued: Dollars,
    /// Maker rebate credited to cash.
    pub rebate_credited: Dollars,
    /// Simulated taker-rebate projection (report-only).
    pub taker_rebate: TakerRebateDto,
}

impl From<&PaperLedgerSnapshot> for LedgerDto {
    fn from(l: &PaperLedgerSnapshot) -> Self {
        Self {
            collateral: l.collateral,
            positions: l
                .positions
                .iter()
                .map(|(token_id, net)| PositionDto {
                    token_id: token_id.clone(),
                    net: *net,
                })
                .collect(),
            fees_paid: l.fees_paid,
            rebate_accrued: l.rebate_accrued,
            rebate_credited: l.rebate_credited,
            taker_rebate: TakerRebateDto::from(l),
        }
    }
}

/// The risk manager's breaker-state projection.
#[derive(Debug, Clone, Serialize)]
pub struct RiskSnapshotDto {
    /// Global breakers currently tripped.
    pub tripped: Vec<BreakerKind>,
    /// Whether any window is under a per-window loss halt.
    pub window_loss_active: bool,
    /// Number of windows loss-halted.
    pub halted_windows: usize,
    /// Authoritative open notional.
    pub open_notional: Dollars,
    /// The open-notional ceiling.
    pub open_notional_cap: Dollars,
    /// Cumulative realized PnL for the current UTC day.
    pub daily_pnl: Dollars,
    /// Infra errors in the current rolling window.
    pub error_count: u32,
    /// Whether |fair − mid| is outside the sanity bound.
    pub sanity_breached: bool,
    /// Whether any global breaker holds all trading down.
    pub globally_halted: bool,
}

impl From<&RiskStateSnapshot> for RiskSnapshotDto {
    fn from(r: &RiskStateSnapshot) -> Self {
        Self {
            tripped: r.tripped.clone(),
            window_loss_active: r.window_loss_active,
            halted_windows: r.halted_windows,
            open_notional: r.open_notional,
            open_notional_cap: r.open_notional_cap,
            daily_pnl: r.daily_pnl,
            error_count: r.error_count,
            sanity_breached: r.sanity_breached,
            globally_halted: r.globally_halted,
        }
    }
}

// ---- overview --------------------------------------------------------------

/// One equity-curve point.
#[derive(Debug, Clone, Serialize)]
pub struct EquityPointDto {
    /// Sample time (unix millis).
    pub ts_ms: i64,
    /// Equity (collateral total) at that time.
    pub equity: Dollars,
}

/// One mode's overview block.
#[derive(Debug, Clone, Serialize)]
pub struct ModeOverviewDto {
    /// Whether the mode's session is running.
    pub running: bool,
    /// Whether live is armed.
    pub armed: bool,
    /// Latest wallet, if known.
    pub wallet: Option<WalletDto>,
    /// Latest paper ledger, if known.
    pub ledger: Option<LedgerDto>,
    /// Equity curve, oldest→newest.
    pub equity: Vec<EquityPointDto>,
    /// Paper starting capital, if known (paper only).
    pub paper_capital: Option<Dollars>,
}

/// Both mode namespaces.
#[derive(Debug, Clone, Serialize)]
pub struct ModesDto {
    /// The paper namespace.
    pub paper: ModeOverviewDto,
    /// The live namespace.
    pub live: ModeOverviewDto,
}

/// The `/api/overview` response.
#[derive(Debug, Clone, Serialize)]
pub struct OverviewDto {
    /// Last observed server time (unix millis).
    pub server_time_ms: i64,
    /// Both namespaces.
    pub modes: ModesDto,
}

fn mode_overview(data: &DashboardData, mode: Mode) -> ModeOverviewDto {
    let ms = data.mode(mode);
    ModeOverviewDto {
        running: ms.running,
        armed: ms.armed,
        wallet: ms.wallet.as_ref().map(WalletDto::from),
        ledger: ms.ledger.as_ref().map(LedgerDto::from),
        equity: ms
            .equity
            .iter()
            .map(|p| EquityPointDto {
                ts_ms: p.ts.as_millis(),
                equity: p.equity,
            })
            .collect(),
        paper_capital: match mode {
            Mode::Paper => data.params.paper_capital,
            Mode::Live => None,
        },
    }
}

/// Builds the `/api/overview` response (both modes).
#[must_use]
pub(crate) fn overview(data: &DashboardData) -> OverviewDto {
    OverviewDto {
        server_time_ms: data.last_now.as_millis(),
        modes: ModesDto {
            paper: mode_overview(data, Mode::Paper),
            live: mode_overview(data, Mode::Live),
        },
    }
}

// ---- active windows --------------------------------------------------------

/// A summary of one active window for a mode.
#[derive(Debug, Clone, Serialize)]
pub struct WindowSummaryDto {
    /// Window key (`Series@open_ms`).
    pub window: String,
    /// The series.
    pub series: Series,
    /// Lifecycle phase.
    pub lifecycle: WindowLifecycle,
    /// Close time (unix millis).
    pub close_time_ms: i64,
    /// Price-to-beat, when captured.
    pub strike: Option<Decimal>,
    /// Gamma event slug.
    pub event_slug: String,
    /// Current tick size.
    pub tick: TickSize,
    /// Up-token top of book.
    pub up_top: Option<TopOfBook>,
    /// Down-token top of book.
    pub down_top: Option<TopOfBook>,
    /// Latest model snapshot for the window's asset.
    pub model: Option<ModelSnapshot>,
    /// This mode's inventory for the window.
    pub inventory: Option<InventorySnapshot>,
    /// Book-unreliable reason, when latched.
    pub book_unreliable: Option<core_types::BookUnreliableReason>,
    /// Winning outcome, when resolved.
    pub outcome: Option<Outcome>,
}

fn window_summary(data: &DashboardData, mode: Mode, wid: WindowId) -> Option<WindowSummaryDto> {
    let view = data.shared.windows.get(&wid)?;
    let market = &view.market;
    Some(WindowSummaryDto {
        window: wid.to_string(),
        series: wid.series,
        lifecycle: view.lifecycle,
        close_time_ms: market.close_time.as_millis(),
        strike: market.strike,
        event_slug: market.event_slug.clone(),
        tick: view.tick,
        up_top: view.up_top,
        down_top: view.down_top,
        // Prefer this window's own model snapshot; fall back to the asset's latest.
        model: data
            .shared
            .model_by_window
            .get(&wid)
            .copied()
            .or_else(|| data.shared.model_by_asset.get(&wid.series.asset).copied()),
        inventory: data.mode(mode).inventory.get(&wid).map(|i| (**i).clone()),
        book_unreliable: data.shared.book_unreliable.get(&wid).copied(),
        outcome: view.outcome,
    })
}

/// Builds the active-window list for a mode.
#[must_use]
pub(crate) fn windows(data: &DashboardData, mode: Mode) -> Vec<WindowSummaryDto> {
    data.shared
        .windows
        .keys()
        .filter_map(|wid| window_summary(data, mode, *wid))
        .collect()
}

/// One book side's levels.
#[derive(Debug, Clone, Serialize)]
pub struct BookSideDto {
    /// Bid levels, best first.
    pub bids: Vec<BookLevel>,
    /// Ask levels, best first.
    pub asks: Vec<BookLevel>,
    /// Snapshot time (unix millis).
    pub ts_ms: i64,
}

/// A recent trade print.
#[derive(Debug, Clone, Serialize)]
pub struct PrintDto {
    /// Print price.
    pub price: Price,
    /// Print size.
    pub size: Size,
    /// Aggressor side.
    pub side: Side,
    /// Print time (unix millis).
    pub ts_ms: i64,
}

/// One of our resting orders on the window — the ladder highlights these.
#[derive(Debug, Clone, Serialize)]
pub struct OurOrderDto {
    /// Venue order id.
    pub order_id: OrderId,
    /// Order side.
    pub side: Side,
    /// Outcome token (mapped from the order's token id).
    pub outcome: Outcome,
    /// Order limit price.
    pub price: Price,
    /// Original size in shares.
    pub original_size: Size,
    /// Cumulative filled size in shares.
    pub filled_size: Size,
    /// Remaining (unfilled) size, floored at zero.
    pub remaining: Size,
    /// Current lifecycle state.
    pub state: OrderState,
}

/// The `/api/windows/{id}` detail response.
#[derive(Debug, Clone, Serialize)]
pub struct WindowDetailDto {
    /// The window summary fields.
    #[serde(flatten)]
    pub summary: WindowSummaryDto,
    /// Full Up-token ladder, when known.
    pub up_book: Option<BookSideDto>,
    /// Full Down-token ladder, when known.
    pub down_book: Option<BookSideDto>,
    /// Up-token book midpoint.
    pub up_mid: Option<Decimal>,
    /// Down-token book midpoint.
    pub down_mid: Option<Decimal>,
    /// Recent prints, oldest→newest.
    pub recent_prints: Vec<PrintDto>,
    /// Our live resting orders on this window (for ladder highlighting).
    pub our_orders: Vec<OurOrderDto>,
}

/// Builds the active-window detail for a mode, `None` if the window is unknown.
#[must_use]
pub(crate) fn window_detail(
    data: &DashboardData,
    mode: Mode,
    wid: WindowId,
) -> Option<WindowDetailDto> {
    let summary = window_summary(data, mode, wid)?;
    let view = data.shared.windows.get(&wid)?;
    let side = |book: &Option<std::sync::Arc<core_types::BookSnapshot>>| {
        book.as_ref().map(|b| BookSideDto {
            bids: b.bids.clone(),
            asks: b.asks.clone(),
            ts_ms: b.ts.as_millis(),
        })
    };
    // Our live resting orders on this window, with token → outcome resolved off
    // the window's token pair (an order whose token we cannot map is skipped).
    let mut our_orders: Vec<OurOrderDto> = data
        .mode(mode)
        .orders
        .values()
        .filter(|u| u.window == wid)
        .filter_map(|u| {
            let outcome = view.market.tokens.outcome_of(&u.token_id)?;
            Some(OurOrderDto {
                order_id: u.order_id.clone(),
                side: u.side,
                outcome,
                price: u.price,
                original_size: u.original_size,
                filled_size: u.filled_size,
                remaining: u.original_size.saturating_sub(u.filled_size),
                state: u.state,
            })
        })
        .collect();
    // Deterministic order (Side has no Ord, so rank Buy before Sell).
    our_orders.sort_by(|a, b| {
        (
            a.outcome,
            matches!(a.side, Side::Sell),
            a.price,
            &a.order_id,
        )
            .cmp(&(
                b.outcome,
                matches!(b.side, Side::Sell),
                b.price,
                &b.order_id,
            ))
    });
    Some(WindowDetailDto {
        up_mid: view.up_top.and_then(|t| t.mid()),
        down_mid: view.down_top.and_then(|t| t.mid()),
        up_book: side(&view.up_book),
        down_book: side(&view.down_book),
        recent_prints: view
            .recent_prints
            .iter()
            .map(|p| PrintDto {
                price: p.price,
                size: p.size,
                side: p.side,
                ts_ms: p.ts.as_millis(),
            })
            .collect(),
        our_orders,
        summary,
    })
}

// ---- fills -----------------------------------------------------------------

/// Strategy attribution of a fill. Maker/Taker come from the venue's liquidity
/// flag; **Late** is *inferred* — a taker fill struck within
/// `late_window_tau_secs` of the window close (the late-window taker's zone).
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Attribution {
    /// A passive (resting-quote) fill.
    Maker,
    /// An aggressive fill outside the late-window zone.
    Taker,
    /// An aggressive fill inside the late-window zone (inferred from time-to-close).
    Late,
}

/// One blotter row: the fill plus its live 5s markout and attribution tag.
#[derive(Debug, Clone, Serialize)]
pub struct FillRowDto {
    /// The underlying fill (flattened — `order_id`, `price`, `liquidity`, …).
    #[serde(flatten)]
    pub fill: Fill,
    /// Live 5-second markout (probability units), once matured; `null` while
    /// pending, for a taker, or when no model anchor existed (NoAnchor-dropped).
    pub markout_5s: Option<f64>,
    /// True for a maker fill still within its 5s window (markout not yet known).
    pub markout_pending: bool,
    /// Maker / taker / late-window attribution.
    pub attribution: Attribution,
}

/// The `/api/fills` response.
#[derive(Debug, Clone, Serialize)]
pub struct FillsDto {
    /// Which trading mode.
    pub mode: Mode,
    /// Fills, newest first.
    pub fills: Vec<FillRowDto>,
}

/// The late-window taker zone (seconds before close) read from the params view,
/// defaulting to the §8 config default when the orchestrator has not set it.
fn late_window_tau_secs(data: &DashboardData) -> i64 {
    data.params
        .entries
        .iter()
        .find(|(k, _)| k == "engine.late_window_tau_secs")
        .and_then(|(_, v)| v.parse::<i64>().ok())
        .unwrap_or(30)
}

/// Maker → Maker; Taker → Late if struck within `late_tau` of close, else Taker.
/// Falls back to liquidity-only when the window (and its close time) is gone.
fn attribution_of(data: &DashboardData, f: &Fill, late_tau: i64) -> Attribution {
    match f.liquidity {
        Liquidity::Maker => Attribution::Maker,
        Liquidity::Taker => {
            let late = data.shared.windows.get(&f.window).is_some_and(|v| {
                let tau = (v.market.close_time.as_millis() - f.ts_venue.as_millis()) / 1_000;
                tau <= late_tau
            });
            if late {
                Attribution::Late
            } else {
                Attribution::Taker
            }
        }
    }
}

/// Builds the fills blotter for a mode (newest first), filtered and limited, each
/// row carrying its live 5s markout and attribution tag.
#[must_use]
pub(crate) fn fills(
    data: &DashboardData,
    mode: Mode,
    limit: usize,
    window: Option<WindowId>,
    since_ms: Option<i64>,
) -> FillsDto {
    let ms = data.mode(mode);
    let late_tau = late_window_tau_secs(data);
    let fills = ms
        .fills
        .iter()
        .rev()
        .filter(|f| window.is_none_or(|w| f.window == w))
        .filter(|f| since_ms.is_none_or(|s| f.ts_venue.as_millis() >= s))
        .take(limit)
        .map(|f| {
            let (markout_5s, markout_pending) = ms.live_markout.markout_for(f);
            FillRowDto {
                fill: (**f).clone(),
                markout_5s,
                markout_pending,
                attribution: attribution_of(data, f, late_tau),
            }
        })
        .collect();
    FillsDto { mode, fills }
}

// ---- risk ------------------------------------------------------------------

/// One feed stream's health.
#[derive(Debug, Clone, Serialize)]
pub struct FeedHealthDto {
    /// Producing feed.
    pub source: PriceSource,
    /// Underlying asset.
    pub asset: Asset,
    /// Observation flavor.
    pub kind: TickKind,
    /// Whether currently stale.
    pub stale: bool,
    /// Age (ms) when staleness was declared.
    pub age_ms: i64,
}

/// One window's book health.
#[derive(Debug, Clone, Serialize)]
pub struct BookHealthDto {
    /// Window key.
    pub window: String,
    /// Why the books are unreliable.
    pub reason: core_types::BookUnreliableReason,
}

/// One asset's model health.
#[derive(Debug, Clone, Serialize)]
pub struct ModelHealthDto {
    /// The asset.
    pub asset: Asset,
    /// Health tier.
    pub health: ModelHealth,
    /// Cause behind the tier.
    pub reason: ModelHealthReason,
    /// Transition time (unix millis).
    pub ts_ms: i64,
}

/// The `/api/risk` response.
#[derive(Debug, Clone, Serialize)]
pub struct RiskDto {
    /// Which trading mode.
    pub mode: Mode,
    /// The richer risk snapshot, when the orchestrator has pushed one.
    pub snapshot: Option<RiskSnapshotDto>,
    /// Breakers currently tripped (folded from the bus).
    pub tripped: Vec<BreakerKind>,
    /// The last cancel-all reason, if any.
    pub last_cancel_all: Option<BreakerKind>,
    /// Whether the user-channel WebSocket is connected (live only).
    pub ws_connected: bool,
    /// Currently-stale feed streams.
    pub feeds: Vec<FeedHealthDto>,
    /// Currently-unreliable window books.
    pub books: Vec<BookHealthDto>,
    /// Per-series book-unreliable event rate over the trailing hour (events/hr) —
    /// the rollover book-disconnect churn the risk `book_staleness_dwell_ms`
    /// filters. A high number here that is NOT tripping FeedStale is the dwell
    /// working as intended; a genuine sustained outage still trips.
    pub book_stale_per_hour: Vec<BookStaleRateDto>,
    /// WARN-only fair-vs-mid divergence watch: per active window, the % of
    /// observed time spent with |p_up − mid| above the slow FairVsMid bound.
    /// Chronic fair-sickness (a window persistently high but self-resolving
    /// before the dwell) shows here long before any halt fires.
    pub divergence_watch: Vec<DivergenceWatchDto>,
    /// Per-asset model health.
    pub model_health: Vec<ModelHealthDto>,
}

/// One window's fair-vs-mid divergence-watch WARN metric.
#[derive(Debug, Clone, Serialize)]
pub struct DivergenceWatchDto {
    /// Window key.
    pub window: String,
    /// Series key.
    pub series: String,
    /// % of observed time with |p_up − mid| above the slow bound.
    pub above_slow_pct: f64,
    /// Observed seconds backing the percentage (small = not yet meaningful).
    pub observed_secs: i64,
}

/// One series' book-unreliable event rate over the trailing hour.
#[derive(Debug, Clone, Serialize)]
pub struct BookStaleRateDto {
    /// Series key (e.g. "BTC-5m").
    pub series: String,
    /// Book-unreliable events in the trailing hour.
    pub events_per_hour: u32,
}

/// Builds the risk panel for a mode.
#[must_use]
pub(crate) fn risk(data: &DashboardData, mode: Mode) -> RiskDto {
    let ms = data.mode(mode);
    RiskDto {
        mode,
        snapshot: ms.risk_snapshot.as_ref().map(RiskSnapshotDto::from),
        tripped: ms.tripped.iter().copied().collect(),
        last_cancel_all: ms.last_cancel_all,
        ws_connected: ms.ws_connected,
        feeds: data
            .shared
            .feed_stale
            .iter()
            .map(|((source, asset, kind), entry)| FeedHealthDto {
                source: *source,
                asset: *asset,
                kind: *kind,
                stale: true,
                age_ms: entry.age_ms,
            })
            .collect(),
        books: data
            .shared
            .book_unreliable
            .iter()
            .map(|(wid, reason)| BookHealthDto {
                window: wid.to_string(),
                reason: *reason,
            })
            .collect(),
        book_stale_per_hour: data
            .shared
            .book_stale_events_per_hour(data.last_now)
            .into_iter()
            .map(|(series, events_per_hour)| BookStaleRateDto {
                series: series.to_string(),
                events_per_hour,
            })
            .collect(),
        divergence_watch: data
            .shared
            .divergence_watch
            .iter()
            .filter(|(_, w)| w.total_ms > 0)
            .map(|(wid, w)| DivergenceWatchDto {
                window: wid.to_string(),
                series: wid.series.to_string(),
                above_slow_pct: 100.0 * (w.above_ms as f64) / (w.total_ms as f64),
                observed_secs: w.total_ms / 1000,
            })
            .collect(),
        model_health: data
            .shared
            .model_health
            .values()
            .map(|ev| ModelHealthDto {
                asset: ev.asset,
                health: ev.health,
                reason: ev.reason,
                ts_ms: ev.ts.as_millis(),
            })
            .collect(),
    }
}

// ---- params ----------------------------------------------------------------

/// One displayed parameter.
#[derive(Debug, Clone, Serialize)]
pub struct ParamEntryDto {
    /// Parameter key.
    pub key: String,
    /// Parameter value (string-rendered).
    pub value: String,
}

/// The `/api/params` response.
#[derive(Debug, Clone, Serialize)]
pub struct ParamsDto {
    /// Paper starting capital, when known.
    pub paper_capital: Option<Dollars>,
    /// Flat parameter entries.
    pub entries: Vec<ParamEntryDto>,
}

// ---- shadow tile (observation-only) ----------------------------------------

/// Per-series shadow prediction summary.
#[derive(Debug, Clone, Serialize)]
pub struct ShadowSeriesDto {
    /// Series key, e.g. `"BTC-5m"`.
    pub series: String,
    /// Predictions in the last 60 s (≈ predictions/min).
    pub per_min: usize,
    /// Most recent model probability of Up.
    pub last_p_up: Option<f64>,
    /// Finite features on the last prediction (of 24).
    pub last_finite: u32,
    /// Last prediction time (ms).
    pub last_ts_ms: i64,
}

/// The "Shadow" tile: per-series rates + model identity + the staleness flag.
#[derive(Debug, Clone, Serialize)]
pub struct ShadowDto {
    /// Whether the shadow observer has produced any prediction.
    pub active: bool,
    /// Short model sha.
    pub model_short_sha: String,
    /// Model training-data end (ms).
    pub trained_through_ms: i64,
    /// Staleness threshold (days).
    pub staleness_alert_days: i64,
    /// Whether the deployed model is stale ("refit due").
    pub model_stale: bool,
    /// Per-series rows, sorted by series key.
    pub series: Vec<ShadowSeriesDto>,
}

/// Builds the shadow tile from the observation state.
#[must_use]
pub(crate) fn shadow(data: &DashboardData) -> ShadowDto {
    let now = data.last_now;
    let now_ms = now.as_millis();
    let mut series: Vec<ShadowSeriesDto> = data
        .shadow
        .by_series
        .iter()
        .map(|(s, st)| ShadowSeriesDto {
            series: s.to_string(),
            per_min: st.per_min(now_ms),
            last_p_up: (st.last_ts != 0).then_some(st.last_p_up),
            last_finite: st.last_finite,
            last_ts_ms: st.last_ts,
        })
        .collect();
    series.sort_by(|a, b| a.series.cmp(&b.series));
    ShadowDto {
        active: data.shadow.active,
        model_short_sha: data.shadow.short_sha.clone(),
        trained_through_ms: data.shadow.trained_through_ms,
        staleness_alert_days: data.shadow.staleness_alert_days,
        model_stale: data.shadow.is_stale(now),
        series,
    }
}

// ---- model-taker tile (§10) ------------------------------------------------

/// Per-series model-taker fires summary.
#[derive(Debug, Clone, Serialize)]
pub struct ModelTakerSeriesDto {
    /// Series key, e.g. `"BTC-5m"`.
    pub series: String,
    /// Fires in the last 60 s (≈ fires/min).
    pub fires_per_min: usize,
    /// Cumulative fires since start.
    pub fires_total: u64,
    /// Cumulative suppressions since start.
    pub suppressed_total: u64,
    /// Most recent model `p_up`.
    pub last_p_up: Option<f64>,
    /// Last decision reason (`"fired"` or the suppression cause).
    pub last_reason: String,
    /// Last decision time (ms).
    pub last_ts_ms: i64,
}

/// One `(series, driver)` PnL attribution row.
#[derive(Debug, Clone, Serialize)]
pub struct DriverPnlDto {
    /// Series key.
    pub series: String,
    /// Driver: `"maker-core"` | `"momentum"` | `"late"` | `"model"`.
    pub driver: String,
    /// Fill count attributed to this driver.
    pub fills: u64,
    /// Taker notional (`Σ price·size`), zero for maker-core (Decimal string).
    pub taker_notional: String,
    /// Fees paid (Decimal string).
    pub fees: String,
    /// Realized PnL, marked to each window's outcome (Decimal string).
    pub realized_pnl: String,
}

/// The "Model taker" tile: per-series fires + PnL split by driver.
#[derive(Debug, Clone, Serialize)]
pub struct ModelTakerDto {
    /// Whether any model-taker decision has arrived.
    pub active: bool,
    /// Per-series fire rows, sorted by series key.
    pub series: Vec<ModelTakerSeriesDto>,
    /// PnL-by-driver rows, sorted by (series, driver).
    pub pnl_by_driver: Vec<DriverPnlDto>,
}

fn driver_label(d: engine::FillDriver) -> &'static str {
    match d {
        engine::FillDriver::MakerCore => "maker-core",
        engine::FillDriver::Momentum => "momentum",
        engine::FillDriver::Late => "late",
        engine::FillDriver::Model => "model",
    }
}

/// Builds the model-taker tile from the fires + driver-PnL state.
#[must_use]
pub(crate) fn model_taker(data: &DashboardData) -> ModelTakerDto {
    let now_ms = data.last_now.as_millis();
    let mut series: Vec<ModelTakerSeriesDto> = data
        .model_taker
        .by_series
        .iter()
        .map(|(s, st)| ModelTakerSeriesDto {
            series: s.to_string(),
            fires_per_min: st.fires_per_min(now_ms),
            fires_total: st.fires_total,
            suppressed_total: st.suppressed_total,
            last_p_up: (st.last_ts != 0).then_some(st.last_p_up),
            last_reason: st.last_reason.clone(),
            last_ts_ms: st.last_ts,
        })
        .collect();
    series.sort_by(|a, b| a.series.cmp(&b.series));

    // The model taker is paper-only in this bot; read the paper namespace's matrix.
    let mut pnl_by_driver: Vec<DriverPnlDto> = data
        .mode(Mode::Paper)
        .driver_pnl
        .totals()
        .iter()
        .map(|((s, d), t)| DriverPnlDto {
            series: s.to_string(),
            driver: driver_label(*d).to_owned(),
            fills: t.fills,
            taker_notional: t.taker_notional.to_string(),
            fees: t.fees.to_string(),
            realized_pnl: t.realized_pnl.to_string(),
        })
        .collect();
    pnl_by_driver.sort_by(|a, b| {
        a.series
            .cmp(&b.series)
            .then_with(|| a.driver.cmp(&b.driver))
    });

    ModelTakerDto {
        active: data.model_taker.active,
        series,
        pnl_by_driver,
    }
}

/// Builds the parameters view.
#[must_use]
pub(crate) fn params(data: &DashboardData) -> ParamsDto {
    ParamsDto {
        paper_capital: data.params.paper_capital,
        entries: data
            .params
            .entries
            .iter()
            .map(|(key, value)| ParamEntryDto {
                key: key.clone(),
                value: value.clone(),
            })
            .collect(),
    }
}

// ---- the 4 drivers × 6 series grid -----------------------------------------

/// The four owned strategies, in display order.
const DRIVERS: [FillDriver; 4] = [
    FillDriver::MakerCore,
    FillDriver::Momentum,
    FillDriver::Late,
    FillDriver::Model,
];

/// The six trading series (BTC/ETH × 5m/15m/1h), in display order — the fixed
/// grid the PnL matrix and STATUS strip enumerate so every cell is defined.
fn all_series() -> [Series; 6] {
    use core_types::{Asset, WindowDuration};
    [
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M15,
        },
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::H1,
        },
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::M5,
        },
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::M15,
        },
        Series {
            asset: Asset::Eth,
            duration: WindowDuration::H1,
        },
    ]
}

// ---- PnL matrix (per-driver × per-series) ----------------------------------

/// One `(series, driver)` cell of the realized-PnL matrix.
#[derive(Debug, Clone, Serialize)]
pub struct PnlCellDto {
    /// Series key.
    pub series: String,
    /// Driver label (`maker-core` | `momentum` | `late` | `model`).
    pub driver: String,
    /// Realized PnL, marked to each window's outcome (Decimal string).
    pub realized_pnl: String,
    /// Resolved (settled) fills.
    pub trades: u32,
    /// Resolved fills that won.
    pub wins: u32,
    /// Win fraction over resolved fills; `null` with none.
    pub win_pct: Option<f64>,
}

/// A per-driver or per-series subtotal.
#[derive(Debug, Clone, Serialize)]
pub struct PnlTotalDto {
    /// The row (driver) or column (series) key this total is for.
    pub key: String,
    /// Realized PnL summed across the other axis (Decimal string).
    pub realized_pnl: String,
    /// Resolved fills summed.
    pub trades: u32,
    /// Winning fills summed.
    pub wins: u32,
    /// Win fraction over resolved fills; `null` with none.
    pub win_pct: Option<f64>,
}

/// The `/api/pnl-matrix` response.
#[derive(Debug, Clone, Serialize)]
pub struct PnlMatrixDto {
    /// Which trading mode.
    pub mode: Mode,
    /// The time window (Today folds a single UTC day; the others fold all
    /// retained days — the matrix keeps ~31 days).
    pub window: ComparisonWindow,
    /// One cell per `(series, driver)` present, sorted by (series, driver).
    pub cells: Vec<PnlCellDto>,
    /// Per-driver totals across series (rows).
    pub row_totals: Vec<PnlTotalDto>,
    /// Per-series totals across drivers (columns).
    pub col_totals: Vec<PnlTotalDto>,
}

fn win_pct(wins: u32, trades: u32) -> Option<f64> {
    if trades == 0 {
        None
    } else {
        Some(f64::from(wins) / f64::from(trades))
    }
}

/// Builds the per-driver × per-series realized-PnL matrix for a mode + window.
#[must_use]
pub(crate) fn pnl_matrix(
    data: &DashboardData,
    mode: Mode,
    window: ComparisonWindow,
) -> PnlMatrixDto {
    let matrix = data.mode(mode).driver_pnl.matrix();
    let cells_map = match window {
        ComparisonWindow::Today => matrix.totals_for_day(DayKey::from_ts(data.last_now)),
        // The matrix retains ~31 days; treat 7d/all alike as the retained totals.
        _ => matrix.totals(),
    };

    let mut cells = Vec::new();
    let mut row: std::collections::HashMap<FillDriver, (Decimal, u32, u32)> =
        std::collections::HashMap::new();
    let mut col: std::collections::BTreeMap<Series, (Decimal, u32, u32)> =
        std::collections::BTreeMap::new();
    for series in all_series() {
        for driver in DRIVERS {
            let cell = cells_map
                .get(&(series, driver))
                .copied()
                .unwrap_or_default();
            if cell.resolved_fills == 0 && cell.realized_pnl == Dollars::ZERO {
                continue; // skip empty cells (the grid is filled client-side)
            }
            cells.push(PnlCellDto {
                series: series.to_string(),
                driver: driver_label(driver).to_owned(),
                realized_pnl: cell.realized_pnl.as_decimal().to_string(),
                trades: cell.resolved_fills,
                wins: cell.winning_fills,
                win_pct: win_pct(cell.winning_fills, cell.resolved_fills),
            });
            let r = row.entry(driver).or_default();
            r.0 += cell.realized_pnl.as_decimal();
            r.1 += cell.resolved_fills;
            r.2 += cell.winning_fills;
            let c = col.entry(series).or_default();
            c.0 += cell.realized_pnl.as_decimal();
            c.1 += cell.resolved_fills;
            c.2 += cell.winning_fills;
        }
    }
    let row_totals = DRIVERS
        .into_iter()
        .filter_map(|d| row.get(&d).map(|(p, t, w)| (d, *p, *t, *w)))
        .map(|(d, p, t, w)| PnlTotalDto {
            key: driver_label(d).to_owned(),
            realized_pnl: p.to_string(),
            trades: t,
            wins: w,
            win_pct: win_pct(w, t),
        })
        .collect();
    let col_totals = col
        .into_iter()
        .map(|(s, (p, t, w))| PnlTotalDto {
            key: s.to_string(),
            realized_pnl: p.to_string(),
            trades: t,
            wins: w,
            win_pct: win_pct(w, t),
        })
        .collect();
    PnlMatrixDto {
        mode,
        window,
        cells,
        row_totals,
        col_totals,
    }
}

// ---- STATUS strip (no silent states) ---------------------------------------

/// The one state a `(driver, series)` cell resolves to — exactly one, never null.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CellState {
    /// Trading normally.
    Active,
    /// Standing down (driver standing down, series disabled, or model reason).
    StandingDown,
    /// A shadow loss-stop WOULD have tripped (recorded, not enforced) — distinct
    /// from a real halt.
    ShadowTripped,
    /// Halted by an operational global breaker, or a per-window loss halt.
    Halted,
}

/// One STATUS-strip cell — always a definite [`CellState`] plus a human detail.
#[derive(Debug, Clone, Serialize)]
pub struct StatusCellDto {
    /// Series key.
    pub series: String,
    /// Driver label.
    pub driver: String,
    /// The resolved state (never null).
    pub state: CellState,
    /// Plain-English reason / which stop / which breaker.
    pub detail: String,
    /// Crossing time for a shadow stop (unix ms), when applicable.
    pub ts_ms: Option<i64>,
    /// The observed loss / worst-case for a shadow stop (Decimal string), when applicable.
    pub value: Option<String>,
}

/// The `/api/status-strip` response — every `(driver, series)` cell.
#[derive(Debug, Clone, Serialize)]
pub struct StatusStripDto {
    /// Which trading mode.
    pub mode: Mode,
    /// One cell per (driver, series), row-major (series outer, driver inner).
    pub cells: Vec<StatusCellDto>,
}

fn driver_standing_down(status: Option<crate::state::DriverStatus>, driver: FillDriver) -> bool {
    status.is_some_and(|s| match driver {
        FillDriver::MakerCore => s.maker_core_standing_down,
        FillDriver::Momentum => s.momentum_standing_down,
        FillDriver::Late => s.late_standing_down,
        FillDriver::Model => s.model_standing_down,
    })
}

/// Resolves one `(driver, series)` cell to exactly one state, by precedence:
/// HALTED > SHADOW_TRIPPED > STANDING_DOWN > ACTIVE.
fn status_cell(
    data: &DashboardData,
    mode: Mode,
    driver: FillDriver,
    series: Series,
) -> StatusCellDto {
    let ms = data.mode(mode);
    let base = |state: CellState, detail: String, ts_ms: Option<i64>, value: Option<String>| {
        StatusCellDto {
            series: series.to_string(),
            driver: driver_label(driver).to_owned(),
            state,
            detail,
            ts_ms,
            value,
        }
    };
    // 1) HALTED — a global operational breaker halts everything; a per-window
    //    loss halt halts that series.
    if let Some(risk) = ms.risk_snapshot.as_ref() {
        if !risk.tripped.is_empty() {
            let names: Vec<String> = risk.tripped.iter().map(|b| format!("{b:?}")).collect();
            return base(
                CellState::Halted,
                format!("breakers: {}", names.join(", ")),
                None,
                None,
            );
        }
        if risk.halted_windows_list.iter().any(|w| w.series == series) {
            return base(
                CellState::Halted,
                "per-window loss halt".to_owned(),
                None,
                None,
            );
        }
        // 2) SHADOW_TRIPPED — a would-be loss stop for this series (WindowLoss) or
        //    a global DailyStop shadow.
        if let Some(s) = risk
            .shadow_tripped
            .iter()
            .find(|s| s.window.is_none_or(|w| w.series == series))
        {
            return base(
                CellState::ShadowTripped,
                format!("shadow {:?}", s.kind),
                Some(s.ts.as_millis()),
                Some(s.value.as_decimal().to_string()),
            );
        }
    }
    // 3) STANDING_DOWN — series disabled, driver standing down, or (model) reason.
    let series_disabled = data
        .control_state
        .as_ref()
        .is_some_and(|c| !c.enabled_series.iter().any(|k| k == series.key()));
    if series_disabled {
        return base(
            CellState::StandingDown,
            "series disabled".to_owned(),
            None,
            None,
        );
    }
    if driver_standing_down(ms.driver_status, driver) {
        return base(
            CellState::StandingDown,
            "standing down".to_owned(),
            None,
            None,
        );
    }
    if driver == FillDriver::Model
        && let Some(st) = data.model_taker.by_series.get(&series)
        && !st.last_reason.is_empty()
        && st.last_reason != "fired"
    {
        return base(CellState::StandingDown, st.last_reason.clone(), None, None);
    }
    // 4) ACTIVE.
    base(CellState::Active, "active".to_owned(), None, None)
}

/// Builds the STATUS strip for a mode — every `(driver, series)` cell, each with
/// exactly one non-null state.
#[must_use]
pub(crate) fn status_strip(data: &DashboardData, mode: Mode) -> StatusStripDto {
    let mut cells = Vec::with_capacity(24);
    for series in all_series() {
        for driver in DRIVERS {
            cells.push(status_cell(data, mode, driver, series));
        }
    }
    StatusStripDto { mode, cells }
}

// ---- benchmark (ours vs competitor fact-sheet) -----------------------------

const COMPETITORS_JSON: &str = include_str!("../competitors.json");

/// One competitor's documented benchmark fact-sheet (from `competitors.json`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompetitorDto {
    /// Account handle.
    pub handle: String,
    /// `primary` | `secondary`.
    pub role: String,
    /// `maker` | `taker`.
    pub kind: String,
    /// Whether our reconstruction of the account reconciled to its official PnL.
    pub anchor_consistent: bool,
    /// Fraction of orders resting at the touch.
    pub at_touch_frac: Option<f64>,
    /// Median per-window capital ($).
    pub per_window_capital_median_usd: Option<f64>,
    /// p90 per-window capital ($).
    pub per_window_capital_p90_usd: Option<f64>,
    /// Windows traded per day.
    pub windows_per_day: Option<f64>,
    /// Official account PnL ($).
    pub official_pnl_usd: Option<f64>,
    /// Merges per active day.
    pub merges_per_active_day: Option<f64>,
    /// Capital turns per day.
    pub turns_per_day: Option<f64>,
    /// Capital recycled ($).
    pub recycled_usd: Option<f64>,
    /// Median fill size ($ notional).
    pub fill_size_median_usd: Option<f64>,
    /// p90 fill size ($ notional).
    pub fill_size_p90_usd: Option<f64>,
    /// p95 cancel round-trip (ms) — `null` (their Data API records only fills).
    pub cancel_p95_ms: Option<f64>,
    /// Pair-completion fraction — `null` (unobservable for their accounts).
    pub pair_completion_frac: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompetitorFile {
    competitors: Vec<CompetitorDto>,
}

fn competitors() -> Vec<CompetitorDto> {
    serde_json::from_str::<CompetitorFile>(COMPETITORS_JSON)
        .map(|f| f.competitors)
        .unwrap_or_default()
}

/// Our own like-for-like benchmark metrics.
#[derive(Debug, Clone, Serialize)]
pub struct OursBenchmarkDto {
    /// The series scope (`null` = all series combined).
    pub series: Option<String>,
    /// Windows traded in scope.
    pub windows_traded: u32,
    /// Average passive-fill 5s markout (probability units).
    pub avg_markout_5s: Option<f64>,
    /// Average passive-fill 30s markout (probability units).
    pub avg_markout_30s: Option<f64>,
    /// Pairs completed (matched held + merged), in pairs.
    pub pairs_completed: String,
    /// Shares stranded (unmatched at close).
    pub stranded_shares: String,
    /// Pair-completion fraction: pairs / (pairs + stranded/2), `null` with none.
    pub pair_completion_frac: Option<f64>,
    /// Pairs merged back to collateral (the merge-velocity numerator).
    pub merged_pairs: String,
    /// Fraction of placements resting at or better than the touch.
    pub at_touch_frac: Option<f64>,
    /// p95 cancel round-trip (ms) — `null` in paper (unobservable).
    pub cancel_p95_ms: Option<i64>,
    /// Median of our fill-size notional ($) — for the like-for-like vs nagi $4/$21.
    pub fill_size_median_usd: Option<String>,
    /// p90 of our fill-size notional ($).
    pub fill_size_p90_usd: Option<String>,
}

/// The `/api/benchmark` response.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkDto {
    /// Which trading mode.
    pub mode: Mode,
    /// Our metrics.
    pub ours: OursBenchmarkDto,
    /// The competitor fact-sheets.
    pub competitors: Vec<CompetitorDto>,
}

/// The `q`-quantile ($) of the maker fill-size notionals, nearest-rank.
fn fill_size_quantile(mut notionals: Vec<Decimal>, q: f64) -> Option<Decimal> {
    if notionals.is_empty() {
        return None;
    }
    notionals.sort_unstable();
    let n = notionals.len();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "fill count is bounded by the fills ring; nearest-rank index is in range"
    )]
    let idx = ((q * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    Some(notionals[idx])
}

/// Builds the benchmark tile (ours vs the competitor fact-sheet) for a mode and
/// optional series scope.
#[must_use]
pub(crate) fn benchmark(data: &DashboardData, mode: Mode, series: Option<Series>) -> BenchmarkDto {
    let ms = data.mode(mode);
    // Aggregate the matching series aggregates (additive accumulators).
    let mut windows = 0u32;
    let (mut m5s, mut m5n, mut m30s, mut m30n) = (0.0f64, 0u64, 0.0f64, 0u64);
    let (mut matched, mut merged, mut stranded) = (Size::ZERO, Size::ZERO, Decimal::ZERO);
    for agg in ms.analytics.rollups().series_aggregates() {
        if series.is_some_and(|f| f != agg.series) {
            continue;
        }
        windows += agg.windows_traded;
        m5s += agg.markout_5s_sum;
        m5n += agg.markout_5s_n;
        m30s += agg.markout_30s_sum;
        m30n += agg.markout_30s_n;
        matched = matched + agg.matched_pairs;
        merged = merged + agg.merged_pairs;
        stranded += agg.stranded_shares;
    }
    let pairs = matched + merged;
    #[allow(clippy::cast_precision_loss, reason = "markout sample count bounded")]
    let avg_markout_5s = (m5n > 0).then(|| m5s / m5n as f64);
    #[allow(clippy::cast_precision_loss, reason = "markout sample count bounded")]
    let avg_markout_30s = (m30n > 0).then(|| m30s / m30n as f64);
    // pair-completion fraction: completed / (completed + half the stranded legs).
    let pair_completion_frac = {
        let p = pairs.as_decimal();
        let denom = p + stranded / Decimal::from(2);
        (denom > Decimal::ZERO)
            .then(|| (p / denom).to_f64())
            .flatten()
    };
    // at-touch fraction over placements with a book reference.
    let at_touch_frac = {
        let (mut hit, mut total) = (0u64, 0u64);
        for (s, (h, t)) in &ms.at_touch {
            if series.is_none_or(|f| f == *s) {
                hit += *h;
                total += *t;
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "placement count bounded")]
        (total > 0).then(|| hit as f64 / total as f64)
    };
    // Our maker fill-size notionals ($) for the like-for-like vs nagi.
    let notionals: Vec<Decimal> = ms
        .fills
        .iter()
        .filter(|f| f.liquidity == Liquidity::Maker)
        .filter(|f| series.is_none_or(|s| f.window.series == s))
        .map(|f| f.price.as_decimal() * f.size.as_decimal())
        .collect();
    let fill_size_median_usd = fill_size_quantile(notionals.clone(), 0.50).map(|d| d.to_string());
    let fill_size_p90_usd = fill_size_quantile(notionals, 0.90).map(|d| d.to_string());

    BenchmarkDto {
        mode,
        ours: OursBenchmarkDto {
            series: series.map(|s| s.to_string()),
            windows_traded: windows,
            avg_markout_5s,
            avg_markout_30s,
            pairs_completed: pairs.as_decimal().to_string(),
            stranded_shares: stranded.to_string(),
            pair_completion_frac,
            merged_pairs: merged.as_decimal().to_string(),
            at_touch_frac,
            cancel_p95_ms: ms.activity.p95_time_to_cancel_ms(series),
            fill_size_median_usd,
            fill_size_p90_usd,
        },
        competitors: competitors(),
    }
}

// ---- contention counters (§10/§11) -----------------------------------------

/// One arbitration-block row: which loser was suppressed by which winner.
#[derive(Debug, Clone, Serialize)]
pub struct ArbitrationRowDto {
    /// Series key.
    pub series: String,
    /// The suppressed (losing) driver.
    pub loser: String,
    /// The winning driver.
    pub winner: String,
    /// Cumulative block count.
    pub count: u64,
}

/// One placement-veto row: how many placements the risk gate refused.
#[derive(Debug, Clone, Serialize)]
pub struct VetoRowDto {
    /// The reject-detail label.
    pub detail: String,
    /// Series key.
    pub series: String,
    /// Cumulative veto count.
    pub count: u64,
}

/// The `/api/contention` response — cross-taker arbitration blocks + risk vetoes.
#[derive(Debug, Clone, Serialize)]
pub struct ContentionDto {
    /// Which trading mode.
    pub mode: Mode,
    /// Arbitration blocks, ranked by count desc.
    pub arbitration: Vec<ArbitrationRowDto>,
    /// Placement vetoes, ranked by count desc.
    pub vetoes: Vec<VetoRowDto>,
}

/// Builds the contention panel for a mode (both tallies ranked by count).
#[must_use]
pub(crate) fn contention(data: &DashboardData, mode: Mode) -> ContentionDto {
    let c = &data.mode(mode).contention;
    let mut arbitration: Vec<ArbitrationRowDto> = c
        .arbitration
        .iter()
        .map(|((series, loser, winner), count)| ArbitrationRowDto {
            series: series.to_string(),
            loser: (*loser).to_owned(),
            winner: (*winner).to_owned(),
            count: *count,
        })
        .collect();
    arbitration.sort_by(|a, b| b.count.cmp(&a.count).then(a.series.cmp(&b.series)));
    let mut vetoes: Vec<VetoRowDto> = c
        .vetoes
        .iter()
        .map(|((detail, series), count)| VetoRowDto {
            detail: (*detail).to_owned(),
            series: series.to_string(),
            count: *count,
        })
        .collect();
    vetoes.sort_by(|a, b| b.count.cmp(&a.count).then(a.detail.cmp(&b.detail)));
    ContentionDto {
        mode,
        arbitration,
        vetoes,
    }
}

/// One `(label, count)` row of the Binance-Mid gap histogram.
#[derive(Debug, Clone, Serialize)]
pub struct MidGapBucketDto {
    /// Bucket label, e.g. "500-1000ms".
    pub label: String,
    /// Count of Mid gaps in this bucket (cumulative since boot).
    pub count: u64,
}

/// The `/api/feed-cadence` response — direct-Binance Mid inter-update gap
/// distribution + event-loop decision lag. Makes feed cadence (network health)
/// and loop health directly visible so a `FeedStale` rate is never inferred from
/// breaker flaps.
#[derive(Debug, Clone, Serialize)]
pub struct FeedCadenceDto {
    /// Mid inter-update gap median (ms).
    pub mid_gap_p50_ms: i64,
    /// Mid inter-update gap p95 (ms).
    pub mid_gap_p95_ms: i64,
    /// Mid inter-update gap max (ms).
    pub mid_gap_max_ms: i64,
    /// Gap histogram (cumulative).
    pub mid_gap_hist: Vec<MidGapBucketDto>,
    /// Event-loop decision lag (receive→process) p95 (ms).
    pub loop_lag_p95_ms: i64,
    /// Event-loop decision lag max (ms).
    pub loop_lag_max_ms: i64,
    /// The fast-feed staleness trip threshold in effect (500 ms bound + grace):
    /// a Mid gap at or above this trips `FeedStale`.
    pub feed_stale_trip_ms: i64,
}

/// Builds the feed-cadence tile from the shared snapshot.
#[must_use]
pub(crate) fn feed_cadence(data: &DashboardData) -> FeedCadenceDto {
    let c = &data.feed_cadence;
    FeedCadenceDto {
        mid_gap_p50_ms: c.mid_gap_p50_ms,
        mid_gap_p95_ms: c.mid_gap_p95_ms,
        mid_gap_max_ms: c.mid_gap_max_ms,
        mid_gap_hist: c
            .mid_gap_hist
            .iter()
            .map(|(label, count)| MidGapBucketDto {
                label: label.clone(),
                count: *count,
            })
            .collect(),
        loop_lag_p95_ms: c.loop_lag_p95_ms,
        loop_lag_max_ms: c.loop_lag_max_ms,
        feed_stale_trip_ms: c.feed_stale_trip_ms,
    }
}

// ---- series comparison -----------------------------------------------------

/// A sortable column hint for the comparison table.
#[derive(Debug, Clone, Serialize)]
pub struct SortColumnDto {
    /// Stable column key (the `SortColumn` variant name).
    pub key: SortColumn,
    /// Display label.
    pub label: &'static str,
    /// Whether a larger value is "better" for the default direction.
    pub higher_is_better: bool,
}

/// The full set of sortable-column hints.
#[must_use]
pub(crate) fn sort_columns() -> Vec<SortColumnDto> {
    SortColumn::ALL
        .into_iter()
        .map(|c| SortColumnDto {
            key: c,
            label: c.label(),
            higher_is_better: c.higher_is_better(),
        })
        .collect()
}

// ---- resolved windows (the "recent resolved windows" dropdown) -------------

/// One settled window split into the two dashboard buckets, plus the per-position
/// settlement (shares held per side and the proceeds the winning side paid out).
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedWindowDto {
    /// Window key (`Series@open_ms`).
    pub window: String,
    /// The series.
    pub series: Series,
    /// Winning outcome.
    pub outcome: Outcome,
    /// Bucket 1 — profit locked from completed pairs.
    pub completed_pair_pnl: Dollars,
    /// Bucket 2 — profit/loss on stuck legs (incl. taker fees).
    pub stuck_leg_pnl: Dollars,
    /// The realized ledger PnL (the two buckets sum to this).
    pub realized_pnl: Dollars,
    /// Pairs completed (held matched + merged), in pairs.
    pub pairs_completed: Size,
    /// Stranded (unmatched) shares at close, in shares.
    pub stranded_shares: Decimal,
    /// Up-side shares held at settlement.
    pub up_shares: Size,
    /// Down-side shares held at settlement.
    pub down_shares: Size,
    /// Settlement proceeds: `$1` per winning share, `$0` per losing share.
    pub proceeds: Dollars,
    /// Settlement time (unix millis) — the resolution *event* time, not window close.
    pub ts_ms: i64,
}

/// Builds the recent-resolved-windows feed (newest first), each split into the
/// two buckets via the same exact decomposition the analytics rollup uses. The
/// rebate/counts are not needed for the two buckets, so they pass as zero.
#[must_use]
pub(crate) fn resolved_windows(
    data: &DashboardData,
    mode: Mode,
    series: Option<Series>,
    limit: usize,
) -> Vec<ResolvedWindowDto> {
    use analytics::WindowAttribution;
    data.mode(mode)
        .settlements
        .iter()
        .rev()
        .filter(|sw| series.is_none_or(|f| sw.summary.window.series == f))
        .take(limit)
        .map(|sw| {
            let s = &sw.summary;
            let a = WindowAttribution::from_summary(
                s,
                mode,
                Dollars::ZERO,
                Decimal::ZERO,
                0,
                0,
                Dollars::ZERO,
            );
            // Proceeds: the winning side's shares each pay $1, the losing side $0.
            let winning_shares = match s.outcome {
                Outcome::Up => s.up.shares,
                Outcome::Down => s.down.shares,
            };
            ResolvedWindowDto {
                window: s.window.to_string(),
                series: s.window.series,
                outcome: s.outcome,
                completed_pair_pnl: pz(a.completed_pair_pnl()),
                stuck_leg_pnl: pz(a.stuck_leg_pnl()),
                realized_pnl: pz(a.realized_pnl),
                pairs_completed: a.pairs_completed(),
                stranded_shares: a.stranded_shares,
                up_shares: s.up.shares,
                down_shares: s.down.shares,
                proceeds: Dollars::new(winning_shares.as_decimal()),
                ts_ms: sw.resolved_at.as_millis(),
            }
        })
        .collect()
}

// ---- summary (the calm landing screen) -------------------------------------

/// The one-line plain-English health verdict the summary computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    /// Pairs are completing and the stuck-leg bucket is not bleeding.
    Healthy,
    /// Legs are getting stranded / quotes are being picked off.
    Caution,
    /// Not enough resolved windows yet to judge.
    WarmingUp,
}

/// The at-a-glance health row.
#[derive(Debug, Clone, Serialize)]
pub struct HeadlineDto {
    /// Paper starting capital (account-global; paper only).
    pub starting_capital: Option<Dollars>,
    /// Current equity (collateral total; account-global).
    pub equity: Option<Dollars>,
    /// Realized P/L for the current UTC day in the selected scope.
    pub today_pl: Dollars,
    /// Fraction of resolved windows that ended in profit, `None` with none.
    pub win_rate: Option<f64>,
    /// Resolved windows in the selected scope (drives the warming-up flag).
    pub windows_traded: u32,
}

/// The three-card account identity: **Cash + Open positions = Equity**. Open
/// positions are valued at cost basis and exclude settled windows by
/// construction (settled windows drop out of `inventory` at resolution), so this
/// is an exact identity, and for a fully-settled history Open positions is `$0`
/// and Equity collapses to Cash.
#[derive(Debug, Clone, Serialize)]
pub struct AccountDto {
    /// Cash (wallet collateral total). `None` until the first wallet sample.
    pub cash: Option<Dollars>,
    /// Cost basis of open positions in unresolved windows (`$0` when flat).
    pub open_positions: Dollars,
    /// Equity = Cash + Open positions. `None` until the first wallet sample.
    pub equity: Option<Dollars>,
    /// Windows closed but not yet resolved (settlement still pending).
    pub awaiting_resolution: usize,
}

/// Builds the three-card account identity for a mode.
fn account(data: &DashboardData, mode: Mode) -> AccountDto {
    AccountDto {
        cash: data.mode(mode).wallet.as_ref().map(|w| w.collateral_total),
        open_positions: data.open_positions_cost(mode),
        equity: data.equity(mode),
        awaiting_resolution: data.awaiting_resolution(),
    }
}

/// The computed status sentence + its reasoning (the tooltip text).
#[derive(Debug, Clone, Serialize)]
pub struct StatusDto {
    /// The verdict.
    pub state: HealthState,
    /// One-line plain-English status.
    pub sentence: String,
    /// How the verdict was decided (shown in a tooltip).
    pub reason: String,
}

/// The two-bucket "why are we making/losing money" explainer.
#[derive(Debug, Clone, Serialize)]
pub struct ExplainerDto {
    /// Bucket 1 — profit locked from completed pairs (the fee-free maker edge).
    pub completed_pair_pnl: Dollars,
    /// Bucket 2 — profit/loss on legs that filled one side only (incl. taker fees).
    pub stuck_leg_pnl: Dollars,
    /// The realized total the two buckets sum to **exactly** (the ledger figure);
    /// `completed_pair_pnl + stuck_leg_pnl == total_pnl`.
    pub total_pnl: Dollars,
    /// The full economic total: `total_pnl + rebate_estimate`. This is the figure
    /// the account's equity moves by, so `completed_pair_pnl + stuck_leg_pnl +
    /// rebate_estimate == comprehensive_pnl`.
    pub comprehensive_pnl: Dollars,
    /// Pairs the strategy completed (both sides filled), counted in **pairs**.
    pub pairs_completed: Size,
    /// Unmatched **shares** left stranded at window close (one side of a
    /// would-be pair — a "leg" — that never got its cheap match).
    pub legs_stranded: Decimal,
    /// Average cost to complete a pair (`avg_up + avg_down`), `None` with none.
    pub avg_pair_cost: Option<Decimal>,
    /// Average directional P/L per stranded share (before taker fees), `None`
    /// when nothing was stranded.
    pub avg_loss_per_stranded: Option<Dollars>,
    /// Estimated maker rebate (a separate bonus, not in the ledger total).
    pub rebate_estimate: Dollars,
}

/// The calm live-activity strip.
#[derive(Debug, Clone, Serialize)]
pub struct ActivityDto {
    /// Orders placed in the last minute.
    pub placed_1m: u64,
    /// Orders cancelled in the last minute.
    pub cancelled_1m: u64,
    /// Fills in the last minute.
    pub fills_1m: u64,
    /// Orders resting on the book right now.
    pub resting_now: u64,
    /// Median cancel round-trip (ms); `null` in paper (unobservable there).
    pub median_ttc_ms: Option<i64>,
    /// Median cancel→replace gap (ms); `null` with no sample.
    pub median_ttr_ms: Option<i64>,
    /// Time-to-cancel target (ms) from config (`cancel_rtt_secs`).
    pub ttc_target_ms: i64,
    /// Time-to-replace target (ms) from config (`min_requote_interval_ms`).
    pub ttr_target_ms: i64,
    /// Whether the median TTC is within target (true when unobserved).
    pub ttc_ok: bool,
    /// Whether the median TTR is within target (true when unobserved).
    pub ttr_ok: bool,
}

/// The `/api/summary` response — everything the calm landing screen needs.
#[derive(Debug, Clone, Serialize)]
pub struct SummaryDto {
    /// Which trading mode.
    pub mode: Mode,
    /// The series scope (`None` = all enabled series combined).
    pub series: Option<Series>,
    /// The time-window selection the explainer covers.
    pub window: ComparisonWindow,
    /// At-a-glance health row.
    pub headline: HeadlineDto,
    /// The three-card account identity (Cash + Open positions = Equity).
    pub account: AccountDto,
    /// The computed status sentence.
    pub status: StatusDto,
    /// The two-bucket explainer.
    pub explainer: ExplainerDto,
    /// The live-activity strip.
    pub activity: ActivityDto,
    /// The simulated taker-rebate projection (report-only), `None` until the first
    /// paper-ledger sample.
    pub taker_rebate: Option<TakerRebateDto>,
}

/// The raw additive sums across the selected series aggregates.
#[derive(Default)]
struct Combined {
    windows_traded: u32,
    windows_profitable: u32,
    net_pnl: Dollars,
    locked_pair_pnl: Dollars,
    inventory_pnl: Dollars,
    taker_fees: Dollars,
    estimated_rebate: Dollars,
    pairs_completed: Size,
    stranded_shares: Decimal,
    pair_cost_sum: Decimal,
    pair_cost_n: u32,
    any_alarm: bool,
}

/// Folds the selected series' aggregates (over `window`) into the raw sums.
fn combine(
    data: &DashboardData,
    mode: Mode,
    series: Option<Series>,
    window: ComparisonWindow,
) -> Combined {
    let ms = data.mode(mode);
    let today = DayKey::from_ts(data.last_now);
    let mut c = Combined::default();
    for agg in ms.analytics.rollups().series_aggregates_over(window, today) {
        if series.is_some_and(|s| s != agg.series) {
            continue;
        }
        c.windows_traded += agg.windows_traded;
        c.windows_profitable += agg.windows_profitable;
        c.net_pnl = c.net_pnl + agg.net_realized;
        c.locked_pair_pnl = c.locked_pair_pnl + agg.locked_pair_pnl;
        c.inventory_pnl = c.inventory_pnl + agg.excess_pnl + agg.settlement_remainder;
        c.taker_fees = c.taker_fees + agg.taker_fees;
        c.estimated_rebate = c.estimated_rebate + agg.estimated_rebate;
        c.pairs_completed = c.pairs_completed + agg.matched_pairs + agg.merged_pairs;
        c.stranded_shares += agg.stranded_shares;
        c.pair_cost_sum += agg.pair_cost_sum;
        c.pair_cost_n += agg.pair_cost_n;
        if ms.analytics.series_health(agg.series) == AdverseSelectionState::Alarm {
            c.any_alarm = true;
        }
    }
    c
}

/// Reads a flat param entry as `i64`, falling back to `default`.
fn param_i64(data: &DashboardData, key: &str, default: i64) -> i64 {
    data.params
        .entries
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(default)
}

/// The time-to-cancel target in ms, from `engine.cancel_rtt_secs` (seconds).
#[allow(
    clippy::cast_possible_truncation,
    reason = "cancel RTT is a small positive seconds value, rounded before the cast"
)]
fn ttc_target_ms(data: &DashboardData) -> i64 {
    let secs: f64 = data
        .params
        .entries
        .iter()
        .find(|(k, _)| k == "engine.cancel_rtt_secs")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0.2);
    (secs * 1000.0).round() as i64
}

/// Formats dollars to cents for a status sentence.
fn usd(d: Dollars) -> String {
    format!("${}", pz(d).as_decimal().round_dp(2))
}

/// Maps a numerically-zero dollar value to positive zero: `rust_decimal` can
/// carry a `-0` sign (e.g. from negating `$0` of fees) that would render as the
/// jarring "-$0" the calm UI must never show.
fn pz(d: Dollars) -> Dollars {
    if d.as_decimal().is_zero() {
        Dollars::ZERO
    } else {
        d
    }
}

/// Computes the plain-English status verdict from the two buckets, the resolved
/// window count, and the adverse-selection alarm. Pure and unit-tested.
#[must_use]
pub(crate) fn compute_status(
    completed: Dollars,
    stuck: Dollars,
    windows_traded: u32,
    min_sample: u32,
    any_alarm: bool,
) -> StatusDto {
    let reason = format!(
        "Decided from the stuck-leg P/L ({}) and the 5-second markout health ({}). \
         Caution when stranded legs give back the pair profit, or when resting quotes \
         are being picked off.",
        usd(stuck),
        if any_alarm { "alarm" } else { "ok" }
    );

    if windows_traded < min_sample {
        return StatusDto {
            state: HealthState::WarmingUp,
            sentence: format!(
                "Warming up — only {windows_traded} resolved window(s) so far (need {min_sample} to judge)."
            ),
            reason,
        };
    }

    // "Giving the profit back": the stuck-leg bucket is negative and erases at
    // least half of the completed-pair profit (or there is no pair cushion).
    let half_completed = Dollars::new(completed.as_decimal() / Decimal::from(2));
    let giving_back = stuck < Dollars::ZERO
        && (completed <= Dollars::ZERO || stuck + half_completed <= Dollars::ZERO);

    if any_alarm {
        StatusDto {
            state: HealthState::Caution,
            sentence:
                "Caution: resting quotes are being picked off — the 5-second markout is negative."
                    .to_owned(),
            reason,
        }
    } else if giving_back {
        StatusDto {
            state: HealthState::Caution,
            sentence: format!(
                "Caution: market is trending, legs are getting stranded — earned {} from completed pairs but gave back {} on stuck legs.",
                usd(completed),
                usd(-stuck)
            ),
            reason,
        }
    } else {
        StatusDto {
            state: HealthState::Healthy,
            sentence: "Healthy: pairs are completing and spreads are positive.".to_owned(),
            reason,
        }
    }
}

/// Builds the `/api/summary` response for a mode, optional series scope, and time
/// window. The two explainer buckets sum to the realized total (the ledger).
#[must_use]
pub(crate) fn summary(
    data: &DashboardData,
    mode: Mode,
    series: Option<Series>,
    window: ComparisonWindow,
) -> SummaryDto {
    let ms = data.mode(mode);
    let c = combine(data, mode, series, window);
    let today = combine(data, mode, series, ComparisonWindow::Today);
    let min_sample = AnalyticsParams::default().min_sample_windows;

    let completed = c.locked_pair_pnl;
    let stuck = c.inventory_pnl + c.taker_fees;

    let avg_pair_cost = if c.pair_cost_n == 0 {
        None
    } else {
        Some(c.pair_cost_sum / Decimal::from(c.pair_cost_n))
    };
    let avg_loss_per_stranded = if c.stranded_shares.is_zero() {
        None
    } else {
        Some(pz(Dollars::new(
            c.inventory_pnl.as_decimal() / c.stranded_shares,
        )))
    };
    let win_rate = if c.windows_traded == 0 {
        None
    } else {
        Some(f64::from(c.windows_profitable) / f64::from(c.windows_traded))
    };

    let now = data.last_now;
    let ttr_target = param_i64(data, "engine.min_requote_interval_ms", 250);
    let ttc_target = ttc_target_ms(data);
    let median_ttc = ms.activity.median_time_to_cancel_ms(series);
    let median_ttr = ms.activity.median_time_to_replace_ms(series);

    SummaryDto {
        mode,
        series,
        window,
        headline: HeadlineDto {
            starting_capital: match mode {
                Mode::Paper => data.params.paper_capital,
                Mode::Live => None,
            },
            // True equity = cash + open-position cost basis (the three-card
            // total), not cash alone.
            equity: data.equity(mode),
            today_pl: pz(today.net_pnl),
            win_rate,
            windows_traded: c.windows_traded,
        },
        account: account(data, mode),
        status: compute_status(completed, stuck, c.windows_traded, min_sample, c.any_alarm),
        explainer: ExplainerDto {
            completed_pair_pnl: pz(completed),
            stuck_leg_pnl: pz(stuck),
            total_pnl: pz(c.net_pnl),
            comprehensive_pnl: pz(c.net_pnl + c.estimated_rebate),
            pairs_completed: c.pairs_completed,
            legs_stranded: c.stranded_shares,
            avg_pair_cost,
            avg_loss_per_stranded,
            rebate_estimate: pz(c.estimated_rebate),
        },
        activity: ActivityDto {
            placed_1m: ms.activity.placed_since(now, 60_000, series),
            cancelled_1m: ms.activity.cancelled_since(now, 60_000, series),
            fills_1m: ms.activity.fills_since(now, 60_000, series),
            resting_now: ms.activity.resting(series),
            median_ttc_ms: median_ttc,
            median_ttr_ms: median_ttr,
            ttc_target_ms: ttc_target,
            ttr_target_ms: ttr_target,
            ttc_ok: median_ttc.is_none_or(|m| m <= ttc_target),
            ttr_ok: median_ttr.is_none_or(|m| m <= ttr_target),
        },
        taker_rebate: ms.ledger.as_ref().map(TakerRebateDto::from),
    }
}

// ---- helpers ---------------------------------------------------------------

/// Parses a `Series@open_ms` window key back into a [`WindowId`].
#[must_use]
pub(crate) fn parse_window_id(s: &str) -> Option<WindowId> {
    let (series_key, open) = s.split_once('@')?;
    let series: Series = series_key.parse().ok()?;
    let open_ms: i64 = open.parse().ok()?;
    Some(WindowId {
        series,
        open_time: core_types::TimestampMs::from_millis(open_ms),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::state::{DashboardData, ParamsView, tests_support};
    use core_types::{
        AnchorSource, DurationMs, Event, InputAges, Liquidity, OrderUpdate, TimestampMs,
        WindowDuration,
    };
    use rust_decimal::dec;
    use std::sync::Arc;
    use venue_api::TokenBalance;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    fn btc_5m_window(open_ms: i64) -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: ts(open_ms),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn order(
        wid: WindowId,
        id: &str,
        token: &str,
        side: Side,
        state: OrderState,
        price: Decimal,
        orig: Decimal,
        filled: Decimal,
    ) -> OrderUpdate {
        OrderUpdate {
            order_id: OrderId::new(id).unwrap(),
            window: wid,
            token_id: TokenId::new(token).unwrap(),
            side,
            state,
            price: Price::try_from(price).unwrap(),
            original_size: Size::new(orig).unwrap(),
            filled_size: Size::new(filled).unwrap(),
            reject_reason: None,
            ts_venue: None,
            ts_local: ts(0),
        }
    }

    #[test]
    fn our_orders_built_from_live_orders() {
        let wid = btc_5m_window(1_000_000);
        let market = tests_support::market(wid);
        let mut data = DashboardData::new(ts(0));
        data.project(
            Mode::Paper,
            &Event::Window {
                market: Arc::clone(&market),
                lifecycle: WindowLifecycle::Open,
            },
            ts(1),
        );
        // token "1" = Up, "2" = Down (per tests_support::market).
        data.project(
            Mode::Paper,
            &Event::OrderUpdate(Arc::new(order(
                wid,
                "o-up",
                "1",
                Side::Buy,
                OrderState::Open,
                dec!(0.55),
                dec!(10),
                dec!(0),
            ))),
            ts(2),
        );
        data.project(
            Mode::Paper,
            &Event::OrderUpdate(Arc::new(order(
                wid,
                "o-dn",
                "2",
                Side::Buy,
                OrderState::PartiallyFilled,
                dec!(0.40),
                dec!(10),
                dec!(4),
            ))),
            ts(3),
        );
        let detail = window_detail(&data, Mode::Paper, wid).expect("detail");
        assert_eq!(detail.our_orders.len(), 2);
        // Sorted Up before Down.
        assert_eq!(detail.our_orders[0].outcome, Outcome::Up);
        assert_eq!(detail.our_orders[1].outcome, Outcome::Down);
        // Remaining = original − filled, floored at 0.
        assert_eq!(detail.our_orders[1].remaining, Size::new(dec!(6)).unwrap());
        assert_eq!(detail.our_orders[0].remaining, Size::new(dec!(10)).unwrap());
    }

    #[test]
    fn our_orders_excludes_other_windows_and_terminal() {
        let wid = btc_5m_window(1_000_000);
        let other = btc_5m_window(2_000_000);
        let market = tests_support::market(wid);
        let mut data = DashboardData::new(ts(0));
        data.project(
            Mode::Paper,
            &Event::Window {
                market,
                lifecycle: WindowLifecycle::Open,
            },
            ts(1),
        );
        // A live order on a different window (filtered out by window match).
        data.project(
            Mode::Paper,
            &Event::OrderUpdate(Arc::new(order(
                other,
                "o-other",
                "1",
                Side::Buy,
                OrderState::Open,
                dec!(0.55),
                dec!(10),
                dec!(0),
            ))),
            ts(2),
        );
        // A terminal order on this window (the fold drops terminal orders, so it
        // is never in `orders` to begin with).
        data.project(
            Mode::Paper,
            &Event::OrderUpdate(Arc::new(order(
                wid,
                "o-term",
                "1",
                Side::Buy,
                OrderState::Filled,
                dec!(0.55),
                dec!(10),
                dec!(10),
            ))),
            ts(3),
        );
        let detail = window_detail(&data, Mode::Paper, wid).expect("detail");
        assert!(detail.our_orders.is_empty());
    }

    fn model_ev(p_up: f64, ts_ms: i64, wid: WindowId) -> Event {
        Event::Model(ModelSnapshot {
            asset: Asset::Btc,
            window: Some(wid),
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
            ts: ts(ts_ms),
        })
    }

    fn fill_ev(wid: WindowId, outcome: Outcome, liq: Liquidity, ts_ms: i64, id: &str) -> Event {
        Event::Fill(Arc::new(Fill {
            order_id: OrderId::new(id).unwrap(),
            trade_id: Some(format!("t-{id}")),
            window: wid,
            token_id: TokenId::new(if outcome == Outcome::Up { "1" } else { "2" }).unwrap(),
            outcome,
            side: Side::Buy,
            price: Price::try_from(dec!(0.48)).unwrap(),
            size: Size::new(dec!(10)).unwrap(),
            liquidity: liq,
            fee: Dollars::ZERO,
            ts_venue: ts(ts_ms),
            ts_local: ts(ts_ms),
        }))
    }

    #[test]
    fn fills_carry_live_markout_and_attribution() {
        let wid = btc_5m_window(0); // close = +300_000
        let market = tests_support::market(wid);
        let mut data = DashboardData::new(ts(0));
        data.project(
            Mode::Paper,
            &Event::Window {
                market,
                lifecycle: WindowLifecycle::Open,
            },
            ts(0),
        );
        data.project(Mode::Paper, &model_ev(0.50, 0, wid), ts(0));
        data.project(
            Mode::Paper,
            &fill_ev(wid, Outcome::Up, Liquidity::Maker, 10, "a"),
            ts(10),
        );
        data.project(Mode::Paper, &model_ev(0.55, 5_000, wid), ts(5_000));
        // A later event advances `now` past the fill's 5s deadline → matures.
        data.project(Mode::Paper, &model_ev(0.55, 6_000, wid), ts(6_000));
        let dto = fills(&data, Mode::Paper, 10, None, None);
        assert_eq!(dto.fills.len(), 1);
        let row = &dto.fills[0];
        assert!(!row.markout_pending);
        assert!((row.markout_5s.expect("matured") - 0.05).abs() < 1e-9);
        assert!(matches!(row.attribution, Attribution::Maker));
    }

    #[test]
    fn attribution_distinguishes_late_window_takes() {
        let wid = btc_5m_window(0); // close = +300_000
        let market = tests_support::market(wid);
        let mut data = DashboardData::new(ts(0));
        data.project(
            Mode::Paper,
            &Event::Window {
                market,
                lifecycle: WindowLifecycle::Open,
            },
            ts(0),
        );
        data.set_params(
            ParamsView {
                paper_capital: None,
                entries: vec![("engine.late_window_tau_secs".to_owned(), "30".to_owned())],
            },
            ts(0),
        );
        // 200s to close → taker; 20s to close → late.
        data.project(
            Mode::Paper,
            &fill_ev(wid, Outcome::Up, Liquidity::Taker, 100_000, "early"),
            ts(100_000),
        );
        data.project(
            Mode::Paper,
            &fill_ev(wid, Outcome::Up, Liquidity::Taker, 280_000, "late"),
            ts(280_000),
        );
        let dto = fills(&data, Mode::Paper, 10, None, None);
        assert_eq!(dto.fills.len(), 2);
        // Newest first: the late-window take, then the mid-window take.
        assert!(matches!(dto.fills[0].attribution, Attribution::Late));
        assert!(matches!(dto.fills[1].attribution, Attribution::Taker));
    }

    #[test]
    fn wallet_dto_roundtrips_to_json() {
        let wallet = Wallet {
            collateral_available: Dollars::new(dec!(123.45)),
            collateral_total: Dollars::new(dec!(123.45)),
            positions: vec![TokenBalance {
                token_id: TokenId::new("7").unwrap(),
                size: Size::new(dec!(10)).unwrap(),
            }],
        };
        let dto = WalletDto::from(&wallet);
        let json = serde_json::to_value(&dto).unwrap();
        // Decimal serializes as a lossless string.
        assert_eq!(json["collateral_total"], "123.45");
        assert_eq!(json["positions"][0]["token_id"], "7");
        assert_eq!(json["positions"][0]["size"], "10");
    }

    #[test]
    fn ledger_dto_preserves_signed_positions() {
        let ledger = PaperLedgerSnapshot {
            collateral: Dollars::new(dec!(900)),
            positions: vec![(TokenId::new("2").unwrap(), dec!(-5))],
            fees_paid: Dollars::new(dec!(1.25)),
            rebate_accrued: Dollars::new(dec!(0.30)),
            rebate_credited: Dollars::new(dec!(2)),
            taker_rebate_accrued: Dollars::new(dec!(0.10)),
            taker_rebate_credited: Dollars::new(dec!(0.50)),
            taker_wv_30d: Dollars::new(dec!(250000)),
            taker_tier: "Gold",
            taker_rebate_pct: dec!(0.18),
            taker_next_tier: Some("Platinum"),
            taker_wv_to_next: Dollars::new(dec!(750000)),
            last_day_taker_fees: Dollars::new(dec!(3)),
        };
        let json = serde_json::to_value(LedgerDto::from(&ledger)).unwrap();
        assert_eq!(json["positions"][0]["net"], "-5");
        assert_eq!(json["fees_paid"], "1.25");
        // The report-only taker-rebate projection: Gold (18%), projected/day =
        // 0.18 × last day's $3 taker fees = $0.54.
        assert_eq!(json["taker_rebate"]["tier"], "Gold");
        assert_eq!(json["taker_rebate"]["rebate_pct"], "0.18");
        assert_eq!(json["taker_rebate"]["projected_per_day"], "0.54");
        assert_eq!(json["taker_rebate"]["next_tier"], "Platinum");
        assert_eq!(json["taker_rebate"]["wv_to_next_tier"], "750000");
    }

    #[test]
    fn parse_window_id_roundtrip() {
        let wid = WindowId {
            series: Series {
                asset: Asset::Eth,
                duration: WindowDuration::H1,
            },
            open_time: core_types::TimestampMs::from_millis(1_781_000_000_000),
        };
        let key = wid.to_string();
        assert_eq!(parse_window_id(&key), Some(wid));
        assert_eq!(parse_window_id("nonsense"), None);
        assert_eq!(parse_window_id("BTC-5m@notanumber"), None);
    }

    #[test]
    fn sort_columns_cover_all() {
        assert_eq!(sort_columns().len(), 13);
    }

    // ---- summary: status sentence -------------------------------------------

    #[test]
    fn status_is_warming_up_below_min_sample() {
        let s = compute_status(Dollars::new(dec!(5)), Dollars::ZERO, 3, 30, false);
        assert_eq!(s.state, HealthState::WarmingUp);
        assert!(s.sentence.contains("Warming up"));
    }

    #[test]
    fn status_is_healthy_when_legs_are_not_bleeding() {
        // Enough windows, stuck-leg bucket positive, no alarm.
        let s = compute_status(Dollars::new(dec!(20)), Dollars::new(dec!(3)), 50, 30, false);
        assert_eq!(s.state, HealthState::Healthy);
        assert!(s.sentence.contains("Healthy"));
    }

    #[test]
    fn status_is_caution_when_stuck_legs_give_back_the_profit() {
        // Earned 10 on pairs, gave back 8 on stuck legs (> half) → Caution.
        let s = compute_status(
            Dollars::new(dec!(10)),
            Dollars::new(dec!(-8)),
            50,
            30,
            false,
        );
        assert_eq!(s.state, HealthState::Caution);
        assert!(s.sentence.contains("gave back"));
        assert!(s.sentence.contains("$8"));
        // A small negative that does not erase half the profit stays Healthy.
        let ok = compute_status(
            Dollars::new(dec!(10)),
            Dollars::new(dec!(-1)),
            50,
            30,
            false,
        );
        assert_eq!(ok.state, HealthState::Healthy);
    }

    #[test]
    fn status_is_caution_on_adverse_selection_alarm() {
        let s = compute_status(Dollars::new(dec!(10)), Dollars::new(dec!(2)), 50, 30, true);
        assert_eq!(s.state, HealthState::Caution);
        assert!(s.sentence.contains("picked off"));
    }

    // ---- summary: builder ----------------------------------------------------

    /// Drives a balanced-book window to settlement and asserts the summary's
    /// two buckets reconcile and the activity tiles populate.
    #[test]
    fn summary_reconciles_buckets_and_reports_activity() {
        use core_types::{SettlementSummary, SideInventory};
        let wid = btc_5m_window(0);
        let market = tests_support::market(wid);
        let mut data = DashboardData::new(ts(0));
        data.project(
            Mode::Paper,
            &Event::Window {
                market: Arc::clone(&market),
                lifecycle: WindowLifecycle::Open,
            },
            ts(0),
        );
        // A placement and a maker fill feed the live-activity strip.
        data.project(
            Mode::Paper,
            &Event::OrderUpdate(Arc::new(order(
                wid,
                "o1",
                "1",
                Side::Buy,
                OrderState::Open,
                dec!(0.48),
                dec!(100),
                dec!(0),
            ))),
            ts(1_000),
        );
        data.project(
            Mode::Paper,
            &fill_ev(wid, Outcome::Up, Liquidity::Maker, 1_000, "f1"),
            ts(1_000),
        );
        // Resolve Up, then settle: 100 Up @ .48 + 100 Down @ .49, realized $3.
        data.project(
            Mode::Paper,
            &Event::Window {
                market,
                lifecycle: WindowLifecycle::Resolved {
                    outcome: Outcome::Up,
                },
            },
            ts(300_000),
        );
        let settle = SettlementSummary::close(
            wid,
            Outcome::Up,
            SideInventory {
                shares: Size::new(dec!(100)).unwrap(),
                cost: Dollars::new(dec!(48)),
            },
            SideInventory {
                shares: Size::new(dec!(100)).unwrap(),
                cost: Dollars::new(dec!(49)),
            },
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(3)),
            ts(300_000),
        );
        data.project(
            Mode::Paper,
            &Event::Settlement(Arc::new(settle)),
            ts(300_000),
        );

        let s = summary(&data, Mode::Paper, None, ComparisonWindow::All);
        // The two buckets sum to the realized total (here all locked, $3).
        assert_eq!(s.explainer.completed_pair_pnl, Dollars::new(dec!(3)));
        assert_eq!(s.explainer.stuck_leg_pnl, Dollars::ZERO);
        assert_eq!(
            s.explainer.completed_pair_pnl + s.explainer.stuck_leg_pnl,
            s.explainer.total_pnl
        );
        assert_eq!(s.explainer.total_pnl, Dollars::new(dec!(3)));
        // ...and the two buckets plus the estimated rebate reconcile to the
        // comprehensive total the account equity actually moves by (§3 of the task).
        assert_eq!(
            s.explainer.completed_pair_pnl
                + s.explainer.stuck_leg_pnl
                + s.explainer.rebate_estimate,
            s.explainer.comprehensive_pnl
        );
        // 100 matched pairs completed.
        assert_eq!(s.explainer.pairs_completed, Size::new(dec!(100)).unwrap());
        assert_eq!(s.headline.win_rate, Some(1.0));
        // Activity (now = 300_000): the placement/fill are >60s old, so the
        // last-minute counts are 0, but a resting order remains.
        assert_eq!(s.activity.resting_now, 1);
        assert_eq!(s.activity.ttr_target_ms, 250); // default when unset
    }

    fn wallet(cash: Decimal) -> Wallet {
        Wallet {
            collateral_available: Dollars::new(cash),
            collateral_total: Dollars::new(cash),
            positions: vec![],
        }
    }

    fn side(shares: Decimal, cost: Decimal) -> core_types::SideInventory {
        core_types::SideInventory {
            shares: Size::new(shares).unwrap(),
            cost: Dollars::new(cost),
        }
    }

    /// The three-card identity holds through a full settlement cycle: while a
    /// position is open, `Cash + Open positions == Equity == starting capital`
    /// (no PnL is recognized at cost basis); after the window resolves the
    /// position drops out of Open by construction and Equity collapses to Cash,
    /// which equals starting capital plus the realized PnL.
    #[test]
    fn account_three_card_identity_through_settlement() {
        let wid = btc_5m_window(0);
        let market = tests_support::market(wid);
        let mut data = DashboardData::new(ts(0));
        data.set_params(
            ParamsView {
                paper_capital: Some(Dollars::new(dec!(10000))),
                entries: vec![],
            },
            ts(0),
        );
        data.project(
            Mode::Paper,
            &Event::Window {
                market: Arc::clone(&market),
                lifecycle: WindowLifecycle::Open,
            },
            ts(0),
        );
        // Bought 100 Up @ 0.48: cash 10000 − 48 = 9952, holding cost $48.
        data.set_wallet(Mode::Paper, wallet(dec!(9952)), ts(1_000));
        data.project(
            Mode::Paper,
            &Event::Inventory(Arc::new(InventorySnapshot::derive(
                wid,
                side(dec!(100), dec!(48)),
                side(dec!(0), dec!(0)),
                ts(1_000),
            ))),
            ts(1_000),
        );

        let a = account(&data, Mode::Paper);
        assert_eq!(a.cash, Some(Dollars::new(dec!(9952))));
        assert_eq!(a.open_positions, Dollars::new(dec!(48)));
        // Cash + Open == Equity == starting (nothing realized yet).
        assert_eq!(a.equity, Some(Dollars::new(dec!(10000))));
        assert_eq!(
            a.cash.unwrap() + a.open_positions,
            a.equity.unwrap(),
            "Cash + Open positions must equal Equity"
        );
        assert_eq!(a.awaiting_resolution, 0);

        // The window closes but has not resolved yet → awaiting resolution.
        data.project(
            Mode::Paper,
            &Event::Window {
                market: Arc::clone(&market),
                lifecycle: WindowLifecycle::Closed,
            },
            ts(299_000),
        );
        assert_eq!(account(&data, Mode::Paper).awaiting_resolution, 1);

        // Resolve Up + settle: winning 100 Up pay $1 each → cash 9952 + 100.
        data.project(
            Mode::Paper,
            &Event::Window {
                market: Arc::clone(&market),
                lifecycle: WindowLifecycle::Resolved {
                    outcome: Outcome::Up,
                },
            },
            ts(300_000),
        );
        let settle = core_types::SettlementSummary::close(
            wid,
            Outcome::Up,
            side(dec!(100), dec!(48)),
            side(dec!(0), dec!(0)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(52)), // realized = 100 payout − 48 cost
            ts(300_000),
        );
        data.project(
            Mode::Paper,
            &Event::Settlement(Arc::new(settle)),
            ts(300_000),
        );
        data.set_wallet(Mode::Paper, wallet(dec!(10052)), ts(300_001));

        let a = account(&data, Mode::Paper);
        // Settled window dropped out of Open positions by construction.
        assert_eq!(a.open_positions, Dollars::ZERO);
        // Fully settled: Equity == Cash == starting + realized PnL.
        assert_eq!(a.equity, Some(Dollars::new(dec!(10052))));
        assert_eq!(a.cash.unwrap() + a.open_positions, a.equity.unwrap());
        assert_eq!(a.awaiting_resolution, 0);
    }

    /// The resolved-windows feed reports the per-position settlement (shares per
    /// side, proceeds paid) and stamps the resolution *event* time, not close.
    #[test]
    fn resolved_windows_reports_per_position_proceeds_at_event_time() {
        let wid = btc_5m_window(0);
        let mut data = DashboardData::new(ts(0));
        let settle = core_types::SettlementSummary::close(
            wid,
            Outcome::Up,
            side(dec!(100), dec!(48)),
            side(dec!(40), dec!(22)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(30)),
            ts(300_000), // summary ts = window CLOSE
        );
        // Observed the resolution 45s after close.
        data.project(
            Mode::Paper,
            &Event::Settlement(Arc::new(settle)),
            ts(345_000),
        );
        let rows = resolved_windows(&data, Mode::Paper, None, 10);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.up_shares, Size::new(dec!(100)).unwrap());
        assert_eq!(r.down_shares, Size::new(dec!(40)).unwrap());
        // Up won → 100 winning shares × $1 = $100 proceeds (Down pays $0).
        assert_eq!(r.proceeds, Dollars::new(dec!(100)));
        // Displayed at the resolution event time, not the window close time.
        assert_eq!(r.ts_ms, 345_000);
    }
}
