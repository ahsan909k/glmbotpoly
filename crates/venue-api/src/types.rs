//! Engine-facing result and event types for the execution port.
//!
//! All reference `core_types`; nothing here is venue-specific.

use std::sync::Arc;

use core_types::{Dollars, Fill, OrderId, OrderState, OrderUpdate, Size, TokenId};

use crate::error::RejectReason;

/// A placed order the venue accepted. Subsequent state changes arrive on the
/// fill stream ([`VenueEvent`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    /// Echoes [`core_types::NewOrder`]'s `client_id` for correlation.
    pub client_id: Option<String>,
    /// Venue-assigned order id.
    pub order_id: OrderId,
    /// State at acknowledgement: `PendingNew`/`Open`, or `Filled` for an order
    /// that was marketable and executed immediately.
    pub state: OrderState,
}

/// One order's rejection at placement time — distinct from an outer
/// [`VenueError`](crate::VenueError), which fails the whole request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceRejection {
    /// Echoes the order's `client_id`.
    pub client_id: Option<String>,
    /// Classified reason.
    pub reason: RejectReason,
    /// Verbatim venue message, for the journal.
    pub raw: String,
}

/// Per-order outcomes of a batch place, positionally aligned with the input
/// `orders` slice: `results[i]` is the outcome of `orders[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlaced {
    /// One result per submitted order, in input order.
    pub results: Vec<Result<Accepted, PlaceRejection>>,
}

/// One order a cancel request failed to cancel, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotCanceled {
    /// The order that was not cancelled.
    pub order_id: OrderId,
    /// Why it was not cancelled.
    pub reason: RejectReason,
    /// Verbatim venue message.
    pub raw: String,
}

/// The outcome of a cancel / cancel-market / cancel-all request: which orders
/// the venue confirmed cancelled and which it did not (the venue's
/// `{canceled, not_canceled}` shape, made venue-agnostic). An empty report is a
/// success with nothing to do (e.g. `cancel_all` with no open orders).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CancelReport {
    /// Orders the venue confirmed cancelled.
    pub canceled: Vec<OrderId>,
    /// Orders the venue did not cancel, with reasons.
    pub not_canceled: Vec<NotCanceled>,
}

impl CancelReport {
    /// True when every targeted order was cancelled (nothing left working).
    #[must_use]
    pub fn all_canceled(&self) -> bool {
        self.not_canceled.is_empty()
    }
}

/// A share balance in one outcome token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBalance {
    /// The outcome token.
    pub token_id: TokenId,
    /// Shares held.
    pub size: Size,
}

/// Collateral and outcome-token balances read from the venue.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Wallet {
    /// pUSD collateral available to trade (total minus open-order reservations).
    pub collateral_available: Dollars,
    /// Total pUSD collateral.
    pub collateral_total: Dollars,
    /// Per-outcome-token share balances held.
    pub positions: Vec<TokenBalance>,
}

/// An item on the venue's order/fill stream.
///
/// `Arc`-wrapped to match the bus variants
/// ([`core_types::Event::OrderUpdate`]/[`core_types::Event::Fill`]) so the
/// orchestrator re-publishes onto the bus with no payload clone.
#[derive(Debug, Clone, PartialEq)]
pub enum VenueEvent {
    /// An order's lifecycle state changed.
    Order(Arc<OrderUpdate>),
    /// One of our orders executed.
    Fill(Arc<Fill>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::IdError;

    fn oid(s: &str) -> OrderId {
        OrderId::new(s).expect("valid id")
    }

    #[test]
    fn cancel_report_all_canceled() {
        let empty = CancelReport::default();
        assert!(empty.all_canceled());

        let ok = CancelReport {
            canceled: vec![oid("a"), oid("b")],
            not_canceled: vec![],
        };
        assert!(ok.all_canceled());

        let partial = CancelReport {
            canceled: vec![oid("a")],
            not_canceled: vec![NotCanceled {
                order_id: oid("b"),
                reason: RejectReason::AlreadyGone,
                raw: "order canceled in the CTF exchange contract".to_owned(),
            }],
        };
        assert!(!partial.all_canceled());
    }

    #[test]
    fn batch_placed_preserves_positional_alignment() {
        let results = vec![
            Ok(Accepted {
                client_id: Some("c0".to_owned()),
                order_id: oid("o0"),
                state: OrderState::Open,
            }),
            Err(PlaceRejection {
                client_id: Some("c1".to_owned()),
                reason: RejectReason::CrossedBook,
                raw: "invalid post-only order: order crosses book".to_owned(),
            }),
        ];
        let batch = BatchPlaced { results };
        assert!(batch.results[0].is_ok());
        assert_eq!(
            batch.results[1].as_ref().unwrap_err().reason,
            RejectReason::CrossedBook
        );
    }

    #[test]
    fn order_id_construction_is_validated() {
        assert_eq!(OrderId::new(""), Err(IdError::EmptyOrderId));
    }
}
