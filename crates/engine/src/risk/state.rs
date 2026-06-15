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

use std::collections::HashSet;

use core_types::{BreakerKind, Dollars, NewOrder, OrderQty, TimestampMs, WindowId};
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
