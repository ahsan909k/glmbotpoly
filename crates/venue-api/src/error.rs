//! The venue-agnostic error and rejection vocabulary.
//!
//! [`RejectReason`] classifies a single order/cancel rejection into categories
//! the engine can branch on without matching raw venue strings; [`VenueError`]
//! is the outer failure type returned by every [`VenuePort`](crate::VenuePort)
//! method. Mapping a concrete venue's HTTP status + body into these types is
//! the *adapter's* job — nothing SDK-specific appears here.

use std::time::Duration;

/// Why the risk manager refused an order at the gateway, before it ever reached
/// the venue (CLAUDE.md §11). Carried by [`RejectReason::RiskRejected`] so the
/// engine can branch on the cause without parsing a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRejectDetail {
    /// A global breaker is tripped — all trading is halted.
    Halted,
    /// The order would push open notional past the global cap (§11).
    OpenNotionalCap,
    /// The order's window is halted for a per-window loss breach (§11).
    WindowHalted,
}

/// Why the venue (or risk manager) refused a single order or cancel, classified
/// into venue-agnostic categories. The original venue text is carried alongside
/// in [`PlaceRejection`](crate::PlaceRejection)/[`NotCanceled`](crate::NotCanceled)
/// for the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// The risk manager refused the order at its gateway before it reached the
    /// venue — a breaker is tripped, or a pre-trade limit would be breached
    /// (§11). Never produced by a venue adapter; minted only by the risk guard.
    RiskRejected(RiskRejectDetail),
    /// A post-only order would have crossed the book — we never pay taker fees
    /// by accident (CLAUDE.md §7).
    CrossedBook,
    /// Order size below the venue minimum.
    BelowMinSize,
    /// Price not on the current tick grid.
    TickRule,
    /// Duplicate of an order already on the book.
    Duplicate,
    /// Not enough collateral / token allowance to back the order.
    InsufficientFunds,
    /// Expiration timestamp invalid (e.g. below the GTD security threshold).
    BadExpiration,
    /// The order/cancel referred to something already gone (cancelled on-chain,
    /// or no matching resting order) — benign for a cancel, a miss for a place.
    AlreadyGone,
    /// A FOK order could not be filled in full.
    FokUnfilled,
    /// A FAK order found nothing to match.
    FakNoMatch,
    /// The market is not yet accepting orders.
    MarketNotReady,
    /// Any rejection the adapter did not classify; the raw venue message is
    /// preserved verbatim.
    Other(String),
}

/// Trading-disabled sub-states a venue can report (Polymarket: HTTP 503), so the
/// risk manager can distinguish a full halt from cancel-only / post-only modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingDisabledMode {
    /// All trading paused.
    Full,
    /// New orders refused; cancels still accepted.
    CancelOnly,
    /// Only post-only orders and cancels accepted.
    PostOnlyMode,
}

/// A failure from a venue operation.
///
/// Deliberately venue-agnostic — no adapter or SDK error type leaks through
/// this surface (CLAUDE.md §2.5). The [`RateLimited`](Self::RateLimited),
/// [`EngineRestarting`](Self::EngineRestarting), and
/// [`TradingDisabled`](Self::TradingDisabled) variants exist so the risk
/// manager's backoff and matching-engine-restart logic (§11) stays independent
/// of which venue produced the error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VenueError {
    /// A single order/cancel was rejected for a classified reason.
    #[error("rejected: {0:?}")]
    Rejected(RejectReason),
    /// Authentication/authorization failed (HTTP 401).
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    /// Rate limited (HTTP 429); back off for `retry_after` when the venue
    /// advised one.
    #[error("rate limited (retry after {retry_after:?})")]
    RateLimited {
        /// Server-advised cool-down, when provided.
        retry_after: Option<Duration>,
    },
    /// The matching engine is restarting (HTTP 425) — reconcile open orders and
    /// retry with backoff (§11).
    #[error("matching engine restarting")]
    EngineRestarting,
    /// Trading is disabled in some mode (HTTP 503).
    #[error("trading disabled ({mode:?}, retry after {retry_after:?})")]
    TradingDisabled {
        /// Which disabled mode the venue reported.
        mode: TradingDisabledMode,
        /// Server-advised cool-down, when provided.
        retry_after: Option<Duration>,
    },
    /// The venue returned an internal error (HTTP 500).
    #[error("venue internal error: {0}")]
    VenueInternal(String),
    /// A batch exceeded the venue's per-request order limit.
    #[error("batch too large: {got} > max {max}")]
    BatchTooLarge {
        /// Orders we tried to submit in one request.
        got: usize,
        /// Venue maximum per request.
        max: usize,
    },
    /// Connect/timeout/serialization failure — nothing the venue rejected, the
    /// round trip just could not complete.
    #[error("transport error: {0}")]
    Transport(String),
    /// The adapter refused to act because live trading is not armed (§11).
    #[error("not armed for live trading")]
    NotArmed,
}
