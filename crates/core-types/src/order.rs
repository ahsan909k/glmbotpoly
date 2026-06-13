//! Order vocabulary: sides, time-in-force, intents, lifecycle states, fills.
//!
//! These types are the contract between `engine` and the `venue-api` port —
//! identical for paper and live (CLAUDE.md §2.5).

use serde::{Deserialize, Serialize};

use crate::ids::{ConditionId, OrderId, TokenId};
use crate::market::{Outcome, WindowId};
use crate::money::{Dollars, Size};
use crate::price::Price;
use crate::time::TimestampMs;

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Buy the outcome token.
    Buy,
    /// Sell the outcome token.
    Sell,
}

impl Side {
    /// The other side.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
}

/// Time-in-force / order class (CLAUDE.md §7).
///
/// Post-only exists only on the resting classes: the venue rejects the order
/// instead of letting it cross — our maker quotes always set it so we never
/// accidentally pay taker fees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good-til-cancelled limit order, rests on the book.
    Gtc {
        /// Reject instead of matching if the order would cross.
        post_only: bool,
    },
    /// Good-til-date limit order. `expires_at` is the desired expiration the
    /// engine reasons about; the venue's 60-second security threshold (the
    /// venue rejects a GTD whose expiration is below `now + 60s`) is applied
    /// **inside the venue adapter**, which floors the on-wire expiration at
    /// `now + 60s`. Callers never add the buffer themselves (CLAUDE.md §7).
    Gtd {
        /// Desired expiration timestamp; the adapter floors it to satisfy the
        /// venue's 60-second minimum.
        expires_at: TimestampMs,
        /// Reject instead of matching if the order would cross.
        post_only: bool,
    },
    /// Fill-or-kill: execute fully and immediately, or cancel.
    Fok,
    /// Fill-and-kill (IOC): execute what is available immediately, cancel
    /// the rest.
    Fak,
}

impl TimeInForce {
    /// True for the marketable classes (FOK/FAK).
    #[must_use]
    pub const fn is_marketable(self) -> bool {
        matches!(self, Self::Fok | Self::Fak)
    }

    /// True if the post-only flag is set (resting classes only).
    #[must_use]
    pub const fn is_post_only(self) -> bool {
        matches!(
            self,
            Self::Gtc { post_only: true }
                | Self::Gtd {
                    post_only: true,
                    ..
                }
        )
    }
}

/// Quantity of a new order. Per CLAUDE.md §7: marketable BUY is specified as
/// a dollar amount, marketable SELL as a share count; resting orders are
/// always shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderQty {
    /// A share count (resting orders; marketable SELL).
    Shares(Size),
    /// A dollar amount (marketable BUY).
    Notional(Dollars),
}

/// A fully-specified order request, already normalized: price on the current
/// grid, quantity at venue precision, minimums satisfied. For FOK/FAK the
/// `price` is the worst-price slippage limit, not a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewOrder {
    /// Optional client-side idempotency/attribution tag.
    pub client_id: Option<String>,
    /// Window this order trades.
    pub window: WindowId,
    /// Outcome token being traded.
    pub token_id: TokenId,
    /// Which outcome `token_id` is (redundant with the token pair, carried
    /// for cheap attribution).
    pub outcome: Outcome,
    /// Buy or sell.
    pub side: Side,
    /// Limit price (resting) or worst-price slippage limit (marketable).
    pub price: Price,
    /// Order quantity.
    pub qty: OrderQty,
    /// Time-in-force / order class.
    pub tif: TimeInForce,
}

/// What the engine asks a venue adapter to do. The risk manager sits between
/// the engine and the venue — every intent passes through it (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderIntent {
    /// Place a new order.
    Place(NewOrder),
    /// Cancel a single order by id.
    Cancel(OrderId),
    /// Cancel every order in one market (large-move repricing, §8).
    CancelMarket(ConditionId),
    /// Cancel everything everywhere (safety paths, §11).
    CancelAll,
}

/// Order lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderState {
    /// Sent to the venue, not yet acknowledged.
    PendingNew,
    /// Resting on the book, nothing filled.
    Open,
    /// Resting on the book, partially filled.
    PartiallyFilled,
    /// Cancel sent, not yet acknowledged.
    PendingCancel,
    /// Fully filled (terminal).
    Filled,
    /// Cancelled (terminal; any filled portion was reported via fills).
    Canceled,
    /// Rejected by the venue or risk manager (terminal).
    Rejected,
    /// GTD expired (terminal).
    Expired,
}

impl OrderState {
    /// True for states an order can never leave.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Filled | Self::Canceled | Self::Rejected | Self::Expired
        )
    }

    /// True while the order is (or may still be) live on the book.
    #[must_use]
    pub const fn is_working(self) -> bool {
        matches!(
            self,
            Self::Open | Self::PartiallyFilled | Self::PendingCancel
        )
    }

    /// Pure transition table for the lifecycle state machine. Venue adapters
    /// enforce it; a `false` transition arriving from the wire indicates a
    /// missed message and triggers reconciliation.
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        match self {
            Self::PendingNew => matches!(
                next,
                Self::Open | Self::PartiallyFilled | Self::Filled | Self::Rejected | Self::Canceled
            ),
            Self::Open => matches!(
                next,
                Self::PartiallyFilled
                    | Self::PendingCancel
                    | Self::Filled
                    | Self::Canceled
                    | Self::Expired
            ),
            Self::PartiallyFilled => matches!(
                next,
                Self::PendingCancel | Self::Filled | Self::Canceled | Self::Expired
            ),
            Self::PendingCancel => {
                // A fill can still race the cancel.
                matches!(next, Self::Canceled | Self::Filled | Self::Expired)
            }
            // Terminal states go nowhere.
            Self::Filled | Self::Canceled | Self::Rejected | Self::Expired => false,
        }
    }
}

/// A venue order-status update (user channel live, or the paper engine).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderUpdate {
    /// Venue order id.
    pub order_id: OrderId,
    /// Window the order belongs to.
    pub window: WindowId,
    /// Outcome token of the order.
    pub token_id: TokenId,
    /// Order side.
    pub side: Side,
    /// New lifecycle state.
    pub state: OrderState,
    /// Order limit price.
    pub price: Price,
    /// Original order size in shares.
    pub original_size: Size,
    /// Cumulative filled size in shares.
    pub filled_size: Size,
    /// Venue reject reason, when `state` is [`OrderState::Rejected`].
    pub reject_reason: Option<String>,
    /// Venue-reported event time, when available.
    pub ts_venue: Option<TimestampMs>,
    /// Local receive time.
    pub ts_local: TimestampMs,
}

/// Which side of the match we were on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Liquidity {
    /// Our resting order was hit — no fee, counts toward maker rebates.
    Maker,
    /// We crossed the spread — taker fee applies.
    Taker,
}

/// One execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    /// Our order that traded.
    pub order_id: OrderId,
    /// Venue trade id, when available.
    pub trade_id: Option<String>,
    /// Window the fill belongs to.
    pub window: WindowId,
    /// Outcome token that traded.
    pub token_id: TokenId,
    /// Which outcome that token is.
    pub outcome: Outcome,
    /// Our side of the trade.
    pub side: Side,
    /// Execution price.
    pub price: Price,
    /// Executed shares.
    pub size: Size,
    /// Maker or taker.
    pub liquidity: Liquidity,
    /// Fee charged on this fill (zero for maker), computed upstream with
    /// [`crate::money::taker_fee`] and the per-market fee params.
    pub fee: Dollars,
    /// Venue-reported execution time.
    pub ts_venue: TimestampMs,
    /// Local receive time.
    pub ts_local: TimestampMs,
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::series::{Asset, Series, WindowDuration};
    use crate::tick::{RoundDir, TickSize};

    #[test]
    fn side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
    }

    #[test]
    fn tif_flags() {
        assert!(TimeInForce::Fok.is_marketable());
        assert!(TimeInForce::Fak.is_marketable());
        assert!(!TimeInForce::Gtc { post_only: true }.is_marketable());
        assert!(TimeInForce::Gtc { post_only: true }.is_post_only());
        assert!(!TimeInForce::Gtc { post_only: false }.is_post_only());
        let gtd = TimeInForce::Gtd {
            expires_at: TimestampMs::from_millis(0),
            post_only: true,
        };
        assert!(gtd.is_post_only() && !gtd.is_marketable());
        assert!(!TimeInForce::Fok.is_post_only());
        assert!(!TimeInForce::Fak.is_post_only());
    }

    #[test]
    fn terminal_and_working_partition() {
        use OrderState as S;
        let terminal = [S::Filled, S::Canceled, S::Rejected, S::Expired];
        let working = [S::Open, S::PartiallyFilled, S::PendingCancel];
        for s in terminal {
            assert!(s.is_terminal() && !s.is_working());
        }
        for s in working {
            assert!(s.is_working() && !s.is_terminal());
        }
        assert!(!S::PendingNew.is_terminal() && !S::PendingNew.is_working());
    }

    #[test]
    fn transition_table() {
        use OrderState as S;
        // Happy paths.
        assert!(S::PendingNew.can_transition_to(S::Open));
        assert!(S::PendingNew.can_transition_to(S::Rejected));
        assert!(S::PendingNew.can_transition_to(S::Filled)); // immediate full fill
        assert!(S::Open.can_transition_to(S::PartiallyFilled));
        assert!(S::PartiallyFilled.can_transition_to(S::Filled));
        assert!(S::Open.can_transition_to(S::PendingCancel));
        assert!(S::PendingCancel.can_transition_to(S::Canceled));
        assert!(S::PendingCancel.can_transition_to(S::Filled)); // fill races cancel
        assert!(S::Open.can_transition_to(S::Expired));
        // Illegal moves.
        assert!(!S::Open.can_transition_to(S::Open));
        assert!(!S::Open.can_transition_to(S::PendingNew));
        assert!(!S::PartiallyFilled.can_transition_to(S::Open));
        assert!(!S::PartiallyFilled.can_transition_to(S::Rejected));
        // Terminal states go nowhere.
        for terminal in [S::Filled, S::Canceled, S::Rejected, S::Expired] {
            for next in [
                S::PendingNew,
                S::Open,
                S::PartiallyFilled,
                S::PendingCancel,
                S::Filled,
                S::Canceled,
                S::Rejected,
                S::Expired,
            ] {
                assert!(
                    !terminal.can_transition_to(next),
                    "{terminal:?} -> {next:?}"
                );
            }
        }
    }

    #[test]
    fn fill_serde_roundtrip() {
        let fill = Fill {
            order_id: OrderId::new("o-1").unwrap(),
            trade_id: Some("t-9".to_owned()),
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(1_718_000_000_000),
            },
            token_id: TokenId::new("111").unwrap(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: Price::quantize(dec!(0.57), TickSize::T001, RoundDir::Down).unwrap(),
            size: Size::new(dec!(25)).unwrap(),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: TimestampMs::from_millis(1_718_000_010_000),
            ts_local: TimestampMs::from_millis(1_718_000_010_042),
        };
        let json = serde_json::to_string(&fill).unwrap();
        let back: Fill = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fill);
    }
}
