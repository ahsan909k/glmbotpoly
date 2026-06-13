//! The crate-internal error type and the venue → typed-error mapping.
//!
//! [`VenueLiveError`] carries richer internal variants (construction failures,
//! arming refusals, dry-run) than the public [`venue_api::VenueError`]; the
//! `From` impl narrows it at the [`LiveVenue`](crate::LiveVenue) boundary. The
//! mapping from a Polymarket HTTP status + body string into a typed error
//! ([`map_status_error`]) and the per-order reject classifier
//! ([`classify_reject`]) are the heart of CLAUDE.md §7's error-code handling.

use std::time::Duration;

use venue_api::{RejectReason, TradingDisabledMode, VenueError};

/// Which arming gate (CLAUDE.md §11) refused construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// Gate 1+2: config `live.enabled` and the env confirmation phrase.
    Boot,
    /// Gate 3: the dashboard arm action for this session.
    Dashboard,
}

/// Everything that can go wrong constructing or operating the live adapter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VenueLiveError {
    /// An arming gate refused (CLAUDE.md §11).
    #[error("not armed for live trading (gate: {0:?})")]
    NotArmed(Gate),
    /// One or more of the four Polymarket API credentials is absent.
    #[error("missing Polymarket API credentials")]
    MissingCredentials,
    /// The signature type needs a funder address but none was given.
    #[error("a funder address is required for the deposit/proxy signature type")]
    MissingFunder,
    /// The funder address is not a `0x`-prefixed 40-hex-character address.
    #[error("funder {0:?} is not a 0x-prefixed 40-hex-character address")]
    BadFunder(String),
    /// Invalid live params.
    #[error("invalid live params: {0}")]
    BadConfig(String),
    /// The order's quantity kind does not match its side/class (a normalizer
    /// bug — the engine should never produce one).
    #[error("order/quantity-kind mismatch: {0}")]
    QtyKindMismatch(String),
    /// The dry-run adapter built and signed the order but will not submit it.
    #[error("dry-run adapter does not submit to the venue")]
    DryRun,
    /// Authentication (signer / API-key derivation) failed.
    #[error("authentication failed: {0}")]
    Auth(String),
    /// A single order/cancel was rejected for a classified reason.
    #[error("rejected: {0:?}")]
    Rejected(RejectReason),
    /// HTTP 401.
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    /// HTTP 429.
    #[error("rate limited (retry after {retry_after:?})")]
    RateLimited {
        /// Server-advised cool-down.
        retry_after: Option<Duration>,
    },
    /// HTTP 425 — matching engine restarting.
    #[error("matching engine restarting")]
    EngineRestarting,
    /// HTTP 503.
    #[error("trading disabled ({mode:?}, retry after {retry_after:?})")]
    TradingDisabled {
        /// Which disabled mode.
        mode: TradingDisabledMode,
        /// Server-advised cool-down.
        retry_after: Option<Duration>,
    },
    /// HTTP 500.
    #[error("venue internal error: {0}")]
    VenueInternal(String),
    /// A batch exceeded the venue's per-request order limit.
    #[error("batch too large: {got} > max {max}")]
    BatchTooLarge {
        /// Orders submitted.
        got: usize,
        /// Venue maximum.
        max: usize,
    },
    /// Connect/timeout/serialization failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// A malformed id (e.g. token/condition id that does not parse).
    #[error("bad id: {0}")]
    BadId(#[from] core_types::IdError),
}

impl From<VenueLiveError> for VenueError {
    fn from(e: VenueLiveError) -> Self {
        match e {
            VenueLiveError::Rejected(r) => Self::Rejected(r),
            VenueLiveError::QtyKindMismatch(m) => Self::Rejected(RejectReason::Other(m)),
            VenueLiveError::Unauthorized(m) | VenueLiveError::Auth(m) => Self::Unauthorized(m),
            VenueLiveError::RateLimited { retry_after } => Self::RateLimited { retry_after },
            VenueLiveError::EngineRestarting => Self::EngineRestarting,
            VenueLiveError::TradingDisabled { mode, retry_after } => {
                Self::TradingDisabled { mode, retry_after }
            }
            VenueLiveError::VenueInternal(m) => Self::VenueInternal(m),
            VenueLiveError::BatchTooLarge { got, max } => Self::BatchTooLarge { got, max },
            VenueLiveError::Transport(m) => Self::Transport(m),
            VenueLiveError::BadId(e) => Self::Transport(format!("bad id: {e}")),
            // Construction/arming failures should never reach a port method
            // (construction fails first); map defensively.
            VenueLiveError::NotArmed(_)
            | VenueLiveError::MissingCredentials
            | VenueLiveError::MissingFunder
            | VenueLiveError::BadFunder(_)
            | VenueLiveError::BadConfig(_)
            | VenueLiveError::DryRun => Self::NotArmed,
        }
    }
}

/// Maps a Polymarket HTTP status + response body into a typed error. The body
/// is matched by substring against the documented strings (CLAUDE.md §7);
/// `retry_after` is the parsed `Retry-After` header when present.
#[must_use]
pub fn map_status_error(status: u16, body: &str, retry_after: Option<Duration>) -> VenueLiveError {
    match status {
        401 => VenueLiveError::Unauthorized(body.trim().to_owned()),
        425 => VenueLiveError::EngineRestarting,
        429 => VenueLiveError::RateLimited { retry_after },
        503 => VenueLiveError::TradingDisabled {
            mode: classify_disabled(body),
            retry_after: retry_after.or_else(|| parse_retry_after_seconds(body)),
        },
        500 => VenueLiveError::VenueInternal(body.trim().to_owned()),
        400 => {
            if let Some(err) = parse_batch_too_large(body) {
                err
            } else {
                VenueLiveError::Rejected(classify_reject(body))
            }
        }
        other => VenueLiveError::Transport(format!("HTTP {other}: {}", body.trim())),
    }
}

/// Classifies a rejection body string (an HTTP-400 body, or a logical rejection
/// carried in a 200-OK `PostOrderResponse.error_msg`) into a [`RejectReason`].
#[must_use]
pub fn classify_reject(body: &str) -> RejectReason {
    let b = body.to_ascii_lowercase();
    // FAK/FOK checked before the generic "no orders found" so the FAK message
    // ("no orders found to match with FAK order") is not swallowed by AlreadyGone.
    if b.contains("crosses book") || b.contains("crosses the book") {
        RejectReason::CrossedBook
    } else if b.contains("minimum tick size") {
        RejectReason::TickRule
    } else if b.contains("lower than the minimum") {
        RejectReason::BelowMinSize
    } else if b.contains("duplicated") {
        RejectReason::Duplicate
    } else if b.contains("not enough balance") || b.contains("allowance") {
        RejectReason::InsufficientFunds
    } else if b.contains("invalid expiration") {
        RejectReason::BadExpiration
    } else if b.contains("fak order") {
        RejectReason::FakNoMatch
    } else if b.contains("fok order") || b.contains("couldn't be fully filled") {
        RejectReason::FokUnfilled
    } else if b.contains("canceled in the ctf exchange contract") || b.contains("no orders found") {
        RejectReason::AlreadyGone
    } else if b.contains("not yet ready") {
        RejectReason::MarketNotReady
    } else {
        RejectReason::Other(body.trim().to_owned())
    }
}

/// 503 sub-mode from the body.
fn classify_disabled(body: &str) -> TradingDisabledMode {
    let b = body.to_ascii_lowercase();
    if b.contains("cancel-only") || b.contains("cancel only") {
        TradingDisabledMode::CancelOnly
    } else if b.contains("post-only") || b.contains("post_only") {
        TradingDisabledMode::PostOnlyMode
    } else {
        TradingDisabledMode::Full
    }
}

/// Parses `"Too many orders in payload: {N}, max allowed: {M}"`.
fn parse_batch_too_large(body: &str) -> Option<VenueLiveError> {
    let b = body.to_ascii_lowercase();
    if !b.contains("too many orders in payload") {
        return None;
    }
    let nums: Vec<usize> = digit_runs(body);
    match nums.as_slice() {
        [got, max, ..] => Some(VenueLiveError::BatchTooLarge {
            got: *got,
            max: *max,
        }),
        _ => None,
    }
}

/// Parses the integer following `retry_after_seconds` (a 503 post-only body),
/// returning it as a [`Duration`].
fn parse_retry_after_seconds(body: &str) -> Option<Duration> {
    let idx = body.find("retry_after_seconds")?;
    let tail = &body[idx + "retry_after_seconds".len()..];
    digit_runs(tail)
        .first()
        .map(|s| Duration::from_secs(*s as u64))
}

/// Every maximal run of ASCII digits in `s`, parsed as `usize` (overflow runs
/// are skipped).
fn digit_runs(s: &str) -> Vec<usize> {
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|run| !run.is_empty())
        .filter_map(|run| run.parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_reject_covers_documented_strings() {
        let cases = [
            (
                "invalid post-only order: order crosses book",
                RejectReason::CrossedBook,
            ),
            (
                "order 0x1 is invalid. Price (0.5) breaks minimum tick size rule: 0.01",
                RejectReason::TickRule,
            ),
            (
                "order 0x1 is invalid. Size (1) lower than the minimum: 5",
                RejectReason::BelowMinSize,
            ),
            ("order 0x1 is invalid. Duplicated.", RejectReason::Duplicate),
            (
                "not enough balance / allowance",
                RejectReason::InsufficientFunds,
            ),
            ("invalid expiration", RejectReason::BadExpiration),
            (
                "order canceled in the CTF exchange contract",
                RejectReason::AlreadyGone,
            ),
            (
                "FOK orders are fully filled or killed",
                RejectReason::FokUnfilled,
            ),
            (
                "order couldn't be fully filled. FOK orders are fully filled or killed.",
                RejectReason::FokUnfilled,
            ),
            (
                "no orders found to match with FAK order. FAK orders are partially filled or killed if no match is found.",
                RejectReason::FakNoMatch,
            ),
            (
                "the market is not yet ready to process new orders",
                RejectReason::MarketNotReady,
            ),
        ];
        for (body, want) in cases {
            assert_eq!(classify_reject(body), want, "body: {body}");
        }
    }

    #[test]
    fn unmapped_400_preserves_raw() {
        let raw = "some brand new error string";
        assert_eq!(
            classify_reject(raw),
            RejectReason::Other(raw.to_owned()),
            "raw must be preserved verbatim"
        );
    }

    #[test]
    fn status_mapping() {
        assert!(matches!(
            map_status_error(401, "Unauthorized/Invalid api key", None),
            VenueLiveError::Unauthorized(_)
        ));
        assert!(matches!(
            map_status_error(425, "", None),
            VenueLiveError::EngineRestarting
        ));
        assert!(matches!(
            map_status_error(429, "Too Many Requests", Some(Duration::from_secs(2))),
            VenueLiveError::RateLimited {
                retry_after: Some(d)
            } if d == Duration::from_secs(2)
        ));
        assert!(matches!(
            map_status_error(500, "there are no matching orders", None),
            VenueLiveError::VenueInternal(_)
        ));
    }

    #[test]
    fn trading_disabled_modes_and_retry() {
        assert!(matches!(
            map_status_error(
                503,
                "Trading is currently disabled. Check polymarket.com",
                None
            ),
            VenueLiveError::TradingDisabled {
                mode: TradingDisabledMode::Full,
                ..
            }
        ));
        assert!(matches!(
            map_status_error(
                503,
                "Trading is currently cancel-only. New orders are not accepted",
                None
            ),
            VenueLiveError::TradingDisabled {
                mode: TradingDisabledMode::CancelOnly,
                ..
            }
        ));
        // post-only 503 carries retry_after_seconds in the body.
        match map_status_error(
            503,
            "post-only mode: only post-only orders and cancels are allowed {\"code\":\"post_only_mode\",\"retry_after_seconds\":7}",
            None,
        ) {
            VenueLiveError::TradingDisabled {
                mode: TradingDisabledMode::PostOnlyMode,
                retry_after: Some(d),
            } => assert_eq!(d, Duration::from_secs(7)),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn batch_too_large_parsed() {
        match map_status_error(400, "Too many orders in payload: 18, max allowed: 15", None) {
            VenueLiveError::BatchTooLarge { got, max } => {
                assert_eq!((got, max), (18, 15));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn internal_to_public_conversion() {
        let pub_err: VenueError = VenueLiveError::Rejected(RejectReason::CrossedBook).into();
        assert_eq!(pub_err, VenueError::Rejected(RejectReason::CrossedBook));
        let pub_err: VenueError = VenueLiveError::NotArmed(Gate::Dashboard).into();
        assert_eq!(pub_err, VenueError::NotArmed);
    }
}
