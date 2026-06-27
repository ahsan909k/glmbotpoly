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
use engine::RiskStateSnapshot;
use serde::Serialize;
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
    /// Per-asset model health.
    pub model_health: Vec<ModelHealthDto>,
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

/// One settled window split into the two dashboard buckets.
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
    /// Pairs completed (held matched + merged).
    pub pairs_completed: Size,
    /// Stranded (unmatched) shares at close.
    pub stranded_shares: Decimal,
    /// Settlement time (unix millis).
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
        .filter(|s| series.is_none_or(|f| s.window.series == f))
        .take(limit)
        .map(|s| {
            let a = WindowAttribution::from_summary(
                s,
                mode,
                Dollars::ZERO,
                Decimal::ZERO,
                0,
                0,
                Dollars::ZERO,
            );
            ResolvedWindowDto {
                window: s.window.to_string(),
                series: s.window.series,
                outcome: s.outcome,
                completed_pair_pnl: pz(a.completed_pair_pnl()),
                stuck_leg_pnl: pz(a.stuck_leg_pnl()),
                realized_pnl: pz(a.realized_pnl),
                pairs_completed: a.pairs_completed(),
                stranded_shares: a.stranded_shares,
                ts_ms: s.ts.as_millis(),
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
    /// The realized total the two buckets sum to (the ledger figure).
    pub total_pnl: Dollars,
    /// Pairs the strategy completed (both sides filled).
    pub pairs_completed: Size,
    /// Unmatched shares left stranded at window close.
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
    /// The computed status sentence.
    pub status: StatusDto,
    /// The two-bucket explainer.
    pub explainer: ExplainerDto,
    /// The live-activity strip.
    pub activity: ActivityDto,
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
            equity: ms.wallet.as_ref().map(|w| w.collateral_total),
            today_pl: pz(today.net_pnl),
            win_rate,
            windows_traded: c.windows_traded,
        },
        status: compute_status(completed, stuck, c.windows_traded, min_sample, c.any_alarm),
        explainer: ExplainerDto {
            completed_pair_pnl: pz(completed),
            stuck_leg_pnl: pz(stuck),
            total_pnl: pz(c.net_pnl),
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
        };
        let json = serde_json::to_value(LedgerDto::from(&ledger)).unwrap();
        assert_eq!(json["positions"][0]["net"], "-5");
        assert_eq!(json["fees_paid"], "1.25");
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
        // 100 matched pairs completed.
        assert_eq!(s.explainer.pairs_completed, Size::new(dec!(100)).unwrap());
        assert_eq!(s.headline.win_rate, Some(1.0));
        // Activity (now = 300_000): the placement/fill are >60s old, so the
        // last-minute counts are 0, but a resting order remains.
        assert_eq!(s.activity.resting_now, 1);
        assert_eq!(s.activity.ttr_target_ms, 250); // default when unset
    }
}
