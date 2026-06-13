//! Typed errors for discovery operations (CLAUDE.md §12: thiserror in
//! crates, anyhow only at the binary boundary).

use core_types::{IdError, Series, TickSizeError};

/// Errors from one discovery operation (a series refresh or one of its
/// HTTP calls).
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// The HTTP request itself failed (connect, timeout, TLS, …).
    #[error("http request to {url} failed: {source}")]
    Http {
        /// The requested URL.
        url: String,
        /// The underlying client error.
        #[source]
        source: reqwest::Error,
    },
    /// The server answered with a non-success status.
    #[error("{url} returned status {status}: {body_prefix}")]
    Status {
        /// The requested URL.
        url: String,
        /// HTTP status code.
        status: u16,
        /// Up to the first few hundred bytes of the response body.
        body_prefix: String,
    },
    /// The response body was not the JSON shape we expect.
    #[error("decoding {url}: {source}; body prefix: {body_prefix}")]
    Decode {
        /// The requested URL.
        url: String,
        /// The JSON error.
        #[source]
        source: serde_json::Error,
        /// Up to the first few hundred bytes of the response body.
        body_prefix: String,
    },
    /// `GET /series?slug=…` returned no series with the requested slug.
    #[error("series slug {slug:?} not found on Gamma")]
    SeriesNotFound {
        /// The slug that was looked up.
        slug: String,
    },
    /// The window query returned no live (current or upcoming) windows.
    /// CLAUDE.md §6: the caller logs loudly and retries with backoff.
    #[error("series {series}: no live windows returned (end_date_min filter applied)")]
    NoWindows {
        /// The affected series.
        series: Series,
    },
    /// A returned event could not be mapped to a [`core_types::MarketInfo`].
    /// Fails the whole series refresh: a half-understood series must not
    /// trade.
    #[error("mapping event {event_slug:?}: {source}")]
    Map {
        /// Slug of the offending event.
        event_slug: String,
        /// What was wrong with it.
        #[source]
        source: MapError,
    },
    /// The reqwest client could not be constructed.
    #[error("building http client: {0}")]
    ClientBuild(#[source] reqwest::Error),
    /// The caller-supplied `now` cannot be expressed as RFC3339 — only
    /// possible with a wildly broken clock.
    #[error("cannot format now={0}ms as RFC3339 — clock out of range")]
    BadNow(i64),
}

/// Validation failures turning a Gamma event into a [`core_types::MarketInfo`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MapError {
    /// The event embeds an unexpected number of markets (always exactly 1
    /// for these series).
    #[error("event has {0} markets, expected exactly 1")]
    MarketCount(usize),
    /// The outcomes array is not exactly `["Up", "Down"]` in some order.
    #[error("outcomes {0:?} are not exactly Up/Down")]
    BadOutcomes(String),
    /// `clobTokenIds` and `outcomes` disagree in length.
    #[error("clobTokenIds length {got} != outcomes length {want}")]
    TokenCountMismatch {
        /// Number of token ids.
        got: usize,
        /// Number of outcomes.
        want: usize,
    },
    /// A token or condition id failed validation.
    #[error("bad id: {0}")]
    BadId(#[from] IdError),
    /// A timestamp field was missing, empty, or unparseable.
    #[error("bad timestamp {field} = {value:?}")]
    BadTimestamp {
        /// Which wire field.
        field: &'static str,
        /// The offending value.
        value: String,
    },
    /// Window close − open does not equal the series duration.
    #[error("window duration {got_ms}ms != series duration {want_ms}ms")]
    DurationMismatch {
        /// Observed close − open in ms.
        got_ms: i64,
        /// Expected series duration in ms.
        want_ms: i64,
    },
    /// The market's tick size is not one of the four supported grids.
    #[error("unsupported tick size: {0}")]
    BadTick(#[from] TickSizeError),
    /// The market's minimum order size is negative or unparseable.
    #[error("bad minimum order size")]
    BadMinSize,
    /// A required wire field is absent.
    #[error("missing required field {0}")]
    MissingField(&'static str),
    /// The event's `seriesSlug` is not the series we asked for.
    #[error("event seriesSlug {got:?} != expected {want:?}")]
    SeriesSlugMismatch {
        /// Slug on the event.
        got: String,
        /// Slug configured for the series.
        want: String,
    },
    /// A double-encoded field (JSON array inside a JSON string) failed to
    /// parse.
    #[error("double-encoded field {field} failed to parse: {msg}")]
    DoubleEncoded {
        /// Which wire field.
        field: &'static str,
        /// Why it failed.
        msg: String,
    },
}
