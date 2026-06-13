//! The execution port traits.
//!
//! Async is expressed with async-fn-in-trait (RPITIT, `impl Future + Send`),
//! matching the workspace's other seams (`feed_util::Transport`,
//! `discovery::DiscoveryApi`) — no `async_trait`, no boxing on the order path.

use std::future::Future;

use core_types::{ConditionId, NewOrder, OrderId};
use tokio::sync::mpsc;

use crate::error::VenueError;
use crate::types::{Accepted, BatchPlaced, CancelReport, VenueEvent, Wallet};

/// The command surface every venue adapter implements. Strategy/engine code
/// depends only on this trait — never on a concrete adapter — so the paper and
/// live engine paths are identical (CLAUDE.md §2.5).
///
/// The GTD 60-second security threshold (§7) is applied *inside* the adapter:
/// callers express the desired expiration in [`core_types::TimeInForce::Gtd`]
/// and never add the buffer themselves. Likewise the worst-price slippage
/// limit for marketable FOK/FAK orders is carried in
/// [`core_types::NewOrder`]'s `price` field per the §7 convention.
pub trait VenuePort: Send + Sync {
    /// Place one already-normalized order (post-only GTC/GTD, or marketable
    /// FOK/FAK).
    fn place(&self, order: &NewOrder) -> impl Future<Output = Result<Accepted, VenueError>> + Send;

    /// Place many orders, chunked into venue-max batches internally; the
    /// returned [`BatchPlaced::results`] are positionally aligned with `orders`.
    fn place_batch(
        &self,
        orders: &[NewOrder],
    ) -> impl Future<Output = Result<BatchPlaced, VenueError>> + Send;

    /// Cancel one order by venue id.
    fn cancel(&self, id: &OrderId)
    -> impl Future<Output = Result<CancelReport, VenueError>> + Send;

    /// Cancel every order in one market (large-move repricing, §8).
    fn cancel_market(
        &self,
        market: &ConditionId,
    ) -> impl Future<Output = Result<CancelReport, VenueError>> + Send;

    /// Cancel everything everywhere — every §11 safety path wires this.
    fn cancel_all(&self) -> impl Future<Output = Result<CancelReport, VenueError>> + Send;

    /// Read current collateral and outcome-token balances.
    fn balances(&self) -> impl Future<Output = Result<Wallet, VenueError>> + Send;
}

/// Source of the venue's order-status + fill stream. Separate from
/// [`VenuePort`] because exactly one task (the reconciler) consumes the stream
/// while many call sites issue commands.
pub trait VenueEvents: Send {
    /// Take the receiver of order/fill events. Returns `None` if it has already
    /// been taken — the stream has a single owner.
    fn take_event_rx(&mut self) -> Option<mpsc::Receiver<VenueEvent>>;
}
