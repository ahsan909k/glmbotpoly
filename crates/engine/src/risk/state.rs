//! The risk gateway's shared gating state and the dashboard projection.
//!
//! [`GateState`] is the **only** mutable state shared between the
//! [`RiskManager`](super::RiskManager) (which projects its [`RiskCore`](super::core::RiskCore)
//! decisions into it) and the [`GuardedPort`](super::guard::GuardedPort) (which
//! reads it to admit/refuse a placement and writes back venue-error
//! observations). It is held behind a `std::sync::Mutex` and locked only for
//! brief, `await`-free critical sections (the `PaperVenue` lock idiom), so the
//! guard's returned future stays `Send`.
//!
//! [`RiskStateSnapshot`] is the cheap, allocation-light projection the dashboard
//! reads for its risk panel (§10.5).

use std::collections::{HashMap, HashSet};

use core_types::{BreakerKind, Dollars, NewOrder, OrderQty, Series, TimestampMs, WindowId};
use venue_api::{RiskRejectDetail, VenueError};

/// A venue-error the guard observed on a port call, for the
/// [`RiskCore`](super::core::RiskCore) to fold into the error-rate and
/// matching-engine-restart breakers. Cancels and reads are never gated, but
/// their errors are still observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardObservation {
    /// An infra-level venue error (transport/internal/rate-limit/auth) — counts
    /// toward the error-rate breaker.
    InfraError(TimestampMs),
    /// The venue reported a matching-engine restart (425) or trading-disabled
    /// (503) — drives the `EngineRestart` breaker.
    EngineRestart(TimestampMs),
}

/// The small lock-protected state the guard consults and writes back.
#[derive(Debug, Default)]
pub(crate) struct GateState {
    /// A global breaker is tripped — every placement is refused.
    pub(crate) halted: bool,
    /// Windows under a per-window loss halt — placements for them are refused.
    pub(crate) halted_windows: HashSet<WindowId>,
    /// Authoritative open notional (resting orders), maintained from the venue
    /// stream by the [`RiskCore`](super::core::RiskCore) and projected here.
    pub(crate) open_notional: Dollars,
    /// Global open-notional ceiling.
    pub(crate) open_notional_cap: Dollars,
    /// Venue-error observations the guard appended; drained each turn by the
    /// driver into the core.
    pub(crate) observations: Vec<GuardObservation>,
    /// Per-series tally of pre-trade placement vetoes, indexed by
    /// [`veto_index`]: how many placements the gate refused per
    /// (series, reject-detail). Drained by
    /// [`RiskManager::contention_snapshot`](super::RiskManager::contention_snapshot).
    pub(crate) veto_tally: HashMap<Series, [u64; VETO_KINDS]>,
}

/// The number of distinct [`RiskRejectDetail`] veto kinds tallied per series.
pub(crate) const VETO_KINDS: usize = 3;

/// The reject details in tally-index order (see [`veto_index`]).
const VETO_DETAILS: [RiskRejectDetail; VETO_KINDS] = [
    RiskRejectDetail::Halted,
    RiskRejectDetail::OpenNotionalCap,
    RiskRejectDetail::WindowHalted,
];

/// Maps a [`RiskRejectDetail`] to its slot in a per-series veto tally array.
const fn veto_index(detail: RiskRejectDetail) -> usize {
    match detail {
        RiskRejectDetail::Halted => 0,
        RiskRejectDetail::OpenNotionalCap => 1,
        RiskRejectDetail::WindowHalted => 2,
    }
}

/// One drained `(reject-detail, series) → count` placement-veto tally, for the
/// dashboard's contention panel (§10/§11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VetoTally {
    /// Why the placement was refused.
    pub detail: RiskRejectDetail,
    /// The series the refused order belonged to.
    pub series: Series,
    /// How many placements were refused for this `(detail, series)`.
    pub count: u64,
}

impl GateState {
    /// The dollar notional a single order would put on the book: `price × shares`
    /// for a resting/SELL order, or the dollar budget for a marketable BUY.
    pub(crate) fn order_notional(order: &NewOrder) -> Dollars {
        match order.qty {
            OrderQty::Shares(s) => Dollars::new(order.price.as_decimal() * s.as_decimal()),
            OrderQty::Notional(d) => d,
        }
    }

    /// Admit-or-refuse one order against the current gate (pure read).
    pub(crate) fn admit(&self, order: &NewOrder) -> Result<(), RiskRejectDetail> {
        if self.halted {
            return Err(RiskRejectDetail::Halted);
        }
        if self.halted_windows.contains(&order.window) {
            return Err(RiskRejectDetail::WindowHalted);
        }
        if self.open_notional + Self::order_notional(order) > self.open_notional_cap {
            return Err(RiskRejectDetail::OpenNotionalCap);
        }
        Ok(())
    }

    /// Admit-or-refuse a whole batch as a unit (the quoter's batch is one
    /// window). The notional check sums the batch so a single converge cannot
    /// straddle the cap.
    pub(crate) fn admit_batch(&self, orders: &[NewOrder]) -> Result<(), RiskRejectDetail> {
        if self.halted {
            return Err(RiskRejectDetail::Halted);
        }
        if orders
            .iter()
            .any(|o| self.halted_windows.contains(&o.window))
        {
            return Err(RiskRejectDetail::WindowHalted);
        }
        let mut total = Dollars::ZERO;
        for o in orders {
            total = total + Self::order_notional(o);
        }
        if self.open_notional + total > self.open_notional_cap {
            return Err(RiskRejectDetail::OpenNotionalCap);
        }
        Ok(())
    }

    /// Classify and record a venue error for the error-rate / restart breakers.
    /// Order rejections (`Rejected`) and `NotArmed` are normal/structural and
    /// are not counted.
    pub(crate) fn observe(&mut self, err: &VenueError, now: TimestampMs) {
        match err {
            VenueError::EngineRestarting | VenueError::TradingDisabled { .. } => {
                self.observations.push(GuardObservation::EngineRestart(now));
            }
            VenueError::Transport(_)
            | VenueError::VenueInternal(_)
            | VenueError::RateLimited { .. }
            | VenueError::Unauthorized(_)
            | VenueError::BatchTooLarge { .. } => {
                self.observations.push(GuardObservation::InfraError(now));
            }
            VenueError::Rejected(_) | VenueError::NotArmed => {}
        }
    }

    /// Records one pre-trade placement veto (the guard calls this on the
    /// `Err(detail)` branch), tallied by `(series, detail)`.
    pub(crate) fn record_veto(&mut self, detail: RiskRejectDetail, series: Series) {
        self.veto_tally.entry(series).or_default()[veto_index(detail)] += 1;
    }

    /// Drains the accumulated placement-veto tallies (drain-and-report, like the
    /// shadow-stop drain), one entry per non-zero `(series, detail)`.
    pub(crate) fn drain_vetoes(&mut self) -> Vec<VetoTally> {
        let mut out = Vec::new();
        for (series, counts) in std::mem::take(&mut self.veto_tally) {
            for (i, &count) in counts.iter().enumerate() {
                if count > 0 {
                    out.push(VetoTally {
                        detail: VETO_DETAILS[i],
                        series,
                        count,
                    });
                }
            }
        }
        out
    }
}

/// A loss-based stop that WOULD have tripped, recorded instead of enforced
/// (CLAUDE.md §11 shadow-loss-stops, paper eval only). Emitted when
/// [`RiskParams::shadow_loss_stops`](super::RiskParams) is set: the `DailyStop`
/// and per-window `WindowLoss` breakers are computed exactly as normal, but
/// instead of halting/cancelling they surface as one of these so trading
/// continues. Operational breakers are never shadowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowStop {
    /// Which loss breaker would have tripped — always `DailyStop` or `WindowLoss`.
    pub kind: BreakerKind,
    /// The window, for a `WindowLoss` shadow; `None` for the global `DailyStop`.
    pub window: Option<WindowId>,
    /// The configured limit/cap that was crossed (a positive dollar amount).
    pub threshold: Dollars,
    /// The observed loss / worst-case at the crossing (signed; a loss is negative
    /// realized PnL, a positive worst-case-if-excess-loses).
    pub value: Dollars,
    /// When the crossing was observed.
    pub ts: TimestampMs,
}

/// A point-in-time projection of the risk manager's breaker state for the
/// dashboard risk panel (§10.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskStateSnapshot {
    /// Global breakers currently tripped (excludes the per-window `WindowLoss`).
    pub tripped: Vec<BreakerKind>,
    /// True when at least one window is under a per-window loss halt.
    pub window_loss_active: bool,
    /// Number of windows currently loss-halted.
    pub halted_windows: usize,
    /// The windows currently under a per-window loss halt (sorted, for a stable
    /// projection). Empowers the status strip to name which series is HALTED.
    pub halted_windows_list: Vec<WindowId>,
    /// Global breaker trips observed in the trailing hour (a "how noisy" rate).
    pub trips_last_hour: u32,
    /// Authoritative open notional across all windows.
    pub open_notional: Dollars,
    /// The open-notional ceiling.
    pub open_notional_cap: Dollars,
    /// Cumulative realized PnL for the current UTC day.
    pub daily_pnl: Dollars,
    /// Infra errors counted in the current rolling window.
    pub error_count: u32,
    /// Whether `|fair − mid|` is currently outside the sanity bound (may be
    /// before the duration has elapsed to trip `FairVsMid`).
    pub sanity_breached: bool,
    /// True when any global breaker holds all trading down.
    pub globally_halted: bool,
    /// Loss stops that WOULD have tripped under shadow-loss-stops mode but were
    /// only recorded (empty unless `shadow_loss_stops` is enabled).
    pub shadow_tripped: Vec<ShadowStop>,
}

impl RiskStateSnapshot {
    /// Whether breaker `b` is currently tripped (global set, or `WindowLoss`).
    #[must_use]
    pub fn is_tripped(&self, b: BreakerKind) -> bool {
        if b == BreakerKind::WindowLoss {
            return self.window_loss_active;
        }
        self.tripped.contains(&b)
    }

    /// Whether any breaker (global or per-window) is currently tripped.
    #[must_use]
    pub fn any_tripped(&self) -> bool {
        !self.tripped.is_empty() || self.window_loss_active
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use core_types::{
        Asset, OrderQty, Price, RoundDir, Series, TickSize, TimestampMs, TokenId, WindowDuration,
        WindowId,
    };
    use rust_decimal::{Decimal, dec};

    use super::*;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_781_000_000_000),
        }
    }
    fn order(price: Decimal, shares: Decimal) -> NewOrder {
        NewOrder {
            client_id: Some("c".to_owned()),
            window: window(),
            token_id: TokenId::new("111").unwrap(),
            outcome: core_types::Outcome::Up,
            side: core_types::Side::Buy,
            price: Price::quantize(price, TickSize::T001, RoundDir::Down).unwrap(),
            qty: OrderQty::Shares(core_types::Size::new(shares).unwrap()),
            tif: core_types::TimeInForce::Gtc { post_only: true },
        }
    }
    fn gate(cap: Decimal) -> GateState {
        GateState {
            open_notional_cap: Dollars::new(cap),
            ..GateState::default()
        }
    }

    #[test]
    fn admit_passes_when_open() {
        assert_eq!(gate(dec!(1000)).admit(&order(dec!(0.50), dec!(10))), Ok(()));
    }

    #[test]
    fn admit_refuses_when_halted() {
        let mut g = gate(dec!(1000));
        g.halted = true;
        assert_eq!(
            g.admit(&order(dec!(0.50), dec!(10))),
            Err(RiskRejectDetail::Halted)
        );
    }

    #[test]
    fn admit_refuses_a_halted_window() {
        let mut g = gate(dec!(1000));
        g.halted_windows.insert(window());
        assert_eq!(
            g.admit(&order(dec!(0.50), dec!(10))),
            Err(RiskRejectDetail::WindowHalted)
        );
    }

    #[test]
    fn admit_refuses_over_the_open_notional_cap() {
        let mut g = gate(dec!(100));
        g.open_notional = Dollars::new(dec!(95));
        // 0.50 × 20 = $10 → 95 + 10 = 105 > 100 cap.
        assert_eq!(
            g.admit(&order(dec!(0.50), dec!(20))),
            Err(RiskRejectDetail::OpenNotionalCap)
        );
        // 0.50 × 10 = $5 → 95 + 5 = 100 ≤ 100 cap (boundary admits).
        assert_eq!(g.admit(&order(dec!(0.50), dec!(10))), Ok(()));
    }

    #[test]
    fn admit_batch_sums_notional() {
        let g = gate(dec!(7));
        let orders = [order(dec!(0.50), dec!(10)), order(dec!(0.40), dec!(10))]; // $5 + $4 = $9
        assert_eq!(
            g.admit_batch(&orders),
            Err(RiskRejectDetail::OpenNotionalCap)
        );
    }

    #[test]
    fn veto_tally_records_and_drains_by_series_and_detail() {
        let mut g = gate(dec!(1000));
        let s = window().series;
        g.record_veto(RiskRejectDetail::Halted, s);
        g.record_veto(RiskRejectDetail::Halted, s);
        g.record_veto(RiskRejectDetail::OpenNotionalCap, s);
        let vetoes = g.drain_vetoes();
        assert_eq!(vetoes.len(), 2);
        let halted = vetoes
            .iter()
            .find(|v| v.detail == RiskRejectDetail::Halted)
            .expect("halted tally");
        assert_eq!(halted.count, 2);
        assert_eq!(halted.series, s);
        let cap = vetoes
            .iter()
            .find(|v| v.detail == RiskRejectDetail::OpenNotionalCap)
            .expect("cap tally");
        assert_eq!(cap.count, 1);
        // Draining clears the tally.
        assert!(g.drain_vetoes().is_empty());
    }

    #[test]
    fn observe_classifies_venue_errors() {
        let mut g = gate(dec!(1000));
        let now = TimestampMs::from_millis(1);
        g.observe(&VenueError::EngineRestarting, now);
        g.observe(&VenueError::Transport("x".to_owned()), now);
        g.observe(
            &VenueError::Rejected(venue_api::RejectReason::CrossedBook),
            now,
        );
        assert_eq!(
            g.observations,
            vec![
                GuardObservation::EngineRestart(now),
                GuardObservation::InfraError(now),
            ],
            "restart + infra recorded; a normal order rejection is not"
        );
    }
}
