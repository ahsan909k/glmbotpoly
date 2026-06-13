//! The internal submission seam: the network layer the SDK lives behind.
//!
//! [`LiveVenue`](crate::LiveVenue) is generic over [`ClobPort`], so all of the
//! order-translation, error-mapping, batch-chunking, and reconcile logic is
//! exercised offline through [`FakeClobPort`](crate::FakeClobPort) while the
//! real [`SdkClobPort`] (added with the SDK) is the only impl that touches the
//! network. RPITIT (no `async_trait`), matching the workspace's other seams.

use std::future::Future;

use core_types::{ConditionId, OrderId, Size};
use venue_api::Wallet;

use crate::convert::BuiltOrder;
use crate::error::VenueLiveError;

/// The venue's acknowledgement of one posted order. A `success == false` ack is
/// a *logical* rejection carried in an otherwise-OK response (the SDK's
/// `PostOrderResponse.success`/`error_msg`), distinct from an HTTP error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawAck {
    /// Echoed client correlation id.
    pub client_id: Option<String>,
    /// Whether the venue accepted the order.
    pub success: bool,
    /// Venue order id, present when accepted.
    pub order_id: Option<OrderId>,
    /// Venue order-status string (e.g. `live`, `matched`).
    pub status: String,
    /// Venue error message, present when rejected.
    pub error: Option<String>,
}

/// The venue's response to a cancel request: which orders were cancelled and
/// which were not (with the venue's reason string).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RawCancel {
    /// Cancelled order ids.
    pub canceled: Vec<OrderId>,
    /// `(order id, reason)` for orders not cancelled.
    pub not_canceled: Vec<(OrderId, String)>,
}

/// One open order as the venue reports it (REST `GET /orders`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawOpenOrder {
    /// Venue order id.
    pub order_id: OrderId,
    /// Venue order-status string.
    pub status: String,
    /// Original order size in shares.
    pub original_size: Size,
    /// Cumulative matched size in shares.
    pub size_matched: Size,
}

/// The network layer. Every method returns the venue's raw (but venue-live-
/// normalized) response; [`LiveVenue`](crate::LiveVenue) maps these to the
/// public `venue_api` types.
pub trait ClobPort: Send + Sync {
    /// Build, sign, and submit one order.
    fn submit(
        &self,
        order: &BuiltOrder,
    ) -> impl Future<Output = Result<RawAck, VenueLiveError>> + Send;

    /// Build, sign, and submit a batch (caller pre-chunked to the venue max).
    /// Returns one ack per input order, in order.
    fn submit_batch(
        &self,
        orders: &[BuiltOrder],
    ) -> impl Future<Output = Result<Vec<RawAck>, VenueLiveError>> + Send;

    /// Cancel one order by id.
    fn cancel_one(
        &self,
        id: &OrderId,
    ) -> impl Future<Output = Result<RawCancel, VenueLiveError>> + Send;

    /// Cancel every order in one market.
    fn cancel_market(
        &self,
        market: &ConditionId,
    ) -> impl Future<Output = Result<RawCancel, VenueLiveError>> + Send;

    /// Cancel everything.
    fn cancel_all(&self) -> impl Future<Output = Result<RawCancel, VenueLiveError>> + Send;

    /// Read collateral + token balances.
    fn balances(&self) -> impl Future<Output = Result<Wallet, VenueLiveError>> + Send;

    /// Fetch all open orders (for the reconcile poll).
    fn open_orders(&self)
    -> impl Future<Output = Result<Vec<RawOpenOrder>, VenueLiveError>> + Send;
}
