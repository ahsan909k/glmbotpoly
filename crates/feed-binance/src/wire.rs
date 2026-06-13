//! Binance spot wire protocol: the combined-stream connect URL builder and
//! the lenient inbound-frame parser.
//!
//! Wire facts (official docs read in full 2026-06-12; live-verified the same
//! day — fixtures in `tests/fixtures/`):
//! - Connecting to `{base}/stream?streams=a/b/c` subscribes in the URL; every
//!   data frame arrives wrapped `{"stream":"<name>","data":<payload>}`. The
//!   raw `/ws/<name>` form delivers bare payloads — parsed defensively here,
//!   never used by our driver.
//! - `<symbol>@bookTicker` payload: `{"u":<updateId>,"s":"BTCUSDT",
//!   "b":"<bid>","B":"<bidQty>","a":"<ask>","A":"<askQty>"}` — real-time,
//!   **no event-type `e` and no event-time `E` field**. Published as the
//!   bid/ask midpoint with `ts_exchange := ts_local` (there is nothing
//!   better on the wire; the push is real-time, so local receive time is the
//!   honest upper bound — the displayed exchange age of Mid ticks is *not* a
//!   latency measurement).
//! - `<symbol>@trade` payload: `{"e":"trade","E":<eventMs>,"s":…,"t":…,
//!   "p":"<price>","q":"<qty>","T":<tradeMs>,"m":…,"M":…}`. Published at the
//!   trade price with `ts_exchange := T` (fallback `E`).
//! - Live SUBSCRIBE/UNSUBSCRIBE responses look like `{"result":null,"id":1}`
//!   (classified as acks; we never send those messages), errors like
//!   `{"code":<n>,"msg":"…","id":…}`.
//! - Server pings every 20 s are protocol-level frames — handled in the
//!   transport, never seen here.
//!
//! Parsing never panics: anything unrecognized maps to
//! [`BinanceParsed::Ignored`] with a reason, for the driver to log and skip.

use core_types::{Asset, Decimal, TimestampMs};
use serde::Deserialize;

use crate::sub::{BinanceStream, BinanceSub};

/// Builds the combined-stream connect URL: `{base}/stream?streams=a/b/c`.
/// `base` is the config `feeds.binance_ws_url` host (trailing `/` tolerated).
#[must_use]
pub fn combined_url(base: &str, subs: &[BinanceSub]) -> String {
    let streams: Vec<String> = subs.iter().map(BinanceSub::stream_name).collect();
    format!(
        "{}/stream?streams={}",
        base.trim_end_matches('/'),
        streams.join("/")
    )
}

/// Resolves a wire stream name (`btcusdt@bookTicker`) to its subscription.
#[must_use]
pub fn map_stream(name: &str) -> Option<BinanceSub> {
    BinanceSub::all()
        .into_iter()
        .find(|sub| sub.stream_name() == name)
}

/// Resolves a payload symbol (`BTCUSDT` — the server sends uppercase in
/// payloads, lowercase in stream names) to its asset.
#[must_use]
pub fn map_symbol(symbol: &str) -> Option<Asset> {
    Asset::ALL
        .into_iter()
        .find(|&asset| BinanceSub::symbol(asset).eq_ignore_ascii_case(symbol))
}

/// One normalized price observation extracted from an inbound frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinanceUpdate {
    /// Producing stream.
    pub sub: BinanceSub,
    /// Price in underlying units: bookTicker → bid/ask midpoint, trade →
    /// trade price. Wire-exact [`Decimal`] (Binance sends prices as strings).
    pub value: Decimal,
    /// Source timestamp: trade time for trades; the local receive time for
    /// bookTicker (no event time exists on that payload — module docs).
    pub ts_exchange: TimestampMs,
}

/// Why an inbound frame was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IgnoredReason {
    /// Text frame that is not valid JSON.
    MalformedJson,
    /// A server error response (`{"code":…,"msg":…}`).
    ServerError,
    /// A stream name, payload symbol, or event type we don't track — under
    /// URL subscription nothing unexpected should ever arrive, so this warns.
    UnknownStream,
    /// Recognized shape with broken content (zero/negative prices, symbol
    /// disagreeing with the wrapper's stream name, non-object payload).
    MalformedPayload,
    /// A required payload field absent.
    MissingField,
}

impl std::fmt::Display for IgnoredReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::MalformedJson => "malformed JSON",
            Self::ServerError => "server error frame",
            Self::UnknownStream => "untracked stream",
            Self::MalformedPayload => "malformed payload",
            Self::MissingField => "missing required field",
        };
        f.write_str(s)
    }
}

/// Classification of one inbound text frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinanceParsed {
    /// A method response (`{"result":…,"id":…}`) — we send no live
    /// SUBSCRIBE messages, but classify these as acks defensively.
    Ack,
    /// One price observation (Binance frames carry exactly one).
    Prices(Vec<BinanceUpdate>),
    /// Skipped frame with the reason (driver logs and moves on).
    Ignored(IgnoredReason),
}

/// Lenient bookTicker payload (every field optional; unknown tolerated).
#[derive(Debug, Deserialize)]
struct WireBookTicker {
    s: Option<String>,
    b: Option<Decimal>,
    a: Option<Decimal>,
}

/// Lenient trade payload.
#[derive(Debug, Deserialize)]
struct WireTrade {
    s: Option<String>,
    p: Option<Decimal>,
    #[serde(rename = "E")]
    event_time: Option<i64>,
    #[serde(rename = "T")]
    trade_time: Option<i64>,
}

/// Parses one inbound text frame. Never panics; never errors — unparseable
/// input degrades to [`BinanceParsed::Ignored`]. `now` is the local receive
/// time, used as `ts_exchange` for bookTicker observations (module docs).
#[must_use]
pub fn parse_frame(text: &str, now: TimestampMs) -> BinanceParsed {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim()) else {
        return BinanceParsed::Ignored(IgnoredReason::MalformedJson);
    };
    let Some(object) = value.as_object() else {
        return BinanceParsed::Ignored(IgnoredReason::MalformedPayload);
    };
    // Method responses: success carries a `result` key (null on success),
    // failure a `code` + `msg` pair.
    if object.contains_key("result") {
        return BinanceParsed::Ack;
    }
    if object.contains_key("code") && object.contains_key("msg") {
        return BinanceParsed::Ignored(IgnoredReason::ServerError);
    }
    // Combined-stream wrapper: identity comes from the stream name, with the
    // payload symbol cross-checked.
    if let (Some(stream), Some(data)) = (
        object.get("stream").and_then(serde_json::Value::as_str),
        object.get("data"),
    ) {
        let Some(sub) = map_stream(stream) else {
            return BinanceParsed::Ignored(IgnoredReason::UnknownStream);
        };
        return match parse_payload(data, Some(sub), now) {
            Ok(update) => BinanceParsed::Prices(vec![update]),
            Err(reason) => BinanceParsed::Ignored(reason),
        };
    }
    // Bare payload (raw /ws form, defensive): identity from shape + symbol.
    match parse_payload(&value, None, now) {
        Ok(update) => BinanceParsed::Prices(vec![update]),
        Err(reason) => BinanceParsed::Ignored(reason),
    }
}

/// Parses one payload object, typed by its shape (`"e":"trade"` marks a
/// trade; the symbol + bid/ask shape marks a bookTicker) and cross-checked
/// against the wrapper-derived subscription when present.
fn parse_payload(
    data: &serde_json::Value,
    hint: Option<BinanceSub>,
    now: TimestampMs,
) -> Result<BinanceUpdate, IgnoredReason> {
    if !data.is_object() {
        return Err(IgnoredReason::MalformedPayload);
    }
    if let Some(event) = data.get("e").and_then(serde_json::Value::as_str) {
        if event != "trade" {
            return Err(IgnoredReason::UnknownStream);
        }
        return parse_trade(data, hint);
    }
    if data.get("b").is_some() && data.get("a").is_some() {
        return parse_book_ticker(data, hint, now);
    }
    Err(IgnoredReason::MalformedPayload)
}

fn parse_book_ticker(
    data: &serde_json::Value,
    hint: Option<BinanceSub>,
    now: TimestampMs,
) -> Result<BinanceUpdate, IgnoredReason> {
    let Ok(wire) = serde_json::from_value::<WireBookTicker>(data.clone()) else {
        return Err(IgnoredReason::MalformedPayload);
    };
    let sub = resolve_identity(wire.s.as_deref(), BinanceStream::BookTicker, hint)?;
    let bid = wire.b.ok_or(IgnoredReason::MissingField)?;
    let ask = wire.a.ok_or(IgnoredReason::MissingField)?;
    // A zero/negative side means an empty or broken book — never publish a
    // nonsense midpoint.
    if bid <= Decimal::ZERO || ask <= Decimal::ZERO {
        return Err(IgnoredReason::MalformedPayload);
    }
    Ok(BinanceUpdate {
        sub,
        value: (bid + ask) / Decimal::TWO,
        ts_exchange: now,
    })
}

fn parse_trade(
    data: &serde_json::Value,
    hint: Option<BinanceSub>,
) -> Result<BinanceUpdate, IgnoredReason> {
    let Ok(wire) = serde_json::from_value::<WireTrade>(data.clone()) else {
        return Err(IgnoredReason::MalformedPayload);
    };
    let sub = resolve_identity(wire.s.as_deref(), BinanceStream::Trade, hint)?;
    let price = wire.p.ok_or(IgnoredReason::MissingField)?;
    if price <= Decimal::ZERO {
        return Err(IgnoredReason::MalformedPayload);
    }
    let ts = wire
        .trade_time
        .or(wire.event_time)
        .ok_or(IgnoredReason::MissingField)?;
    Ok(BinanceUpdate {
        sub,
        value: price,
        ts_exchange: TimestampMs::from_millis(ts),
    })
}

/// Resolves the stream identity from the payload symbol and shape-derived
/// stream type, requiring agreement with the wrapper-derived hint when both
/// exist (a wrapper lying about its payload is a venue bug worth a warning,
/// not a silent mis-tagged publish).
fn resolve_identity(
    symbol: Option<&str>,
    stream: BinanceStream,
    hint: Option<BinanceSub>,
) -> Result<BinanceSub, IgnoredReason> {
    let symbol = symbol.ok_or(IgnoredReason::MissingField)?;
    let asset = map_symbol(symbol).ok_or(IgnoredReason::UnknownStream)?;
    let sub = BinanceSub::new(asset, stream);
    match hint {
        Some(h) if h != sub => Err(IgnoredReason::MalformedPayload),
        _ => Ok(sub),
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const NOW: TimestampMs = TimestampMs::from_millis(1_700_000_000_000);

    /// The exact bookTicker shape from the official docs (no `e`/`E`).
    const BOOK: &str = r#"{"u":400900217,"s":"BTCUSDT","b":"63500.01000000","B":"31.21000000","a":"63500.03000000","A":"40.66000000"}"#;
    /// The exact trade shape from the official docs.
    const TRADE: &str = r#"{"e":"trade","E":1672515782136,"s":"ETHUSDT","t":12345,"p":"1670.39000000","q":"100","T":1672515782134,"m":true,"M":true}"#;

    fn wrapped(stream: &str, payload: &str) -> String {
        format!(r#"{{"stream":"{stream}","data":{payload}}}"#)
    }

    #[test]
    fn combined_url_joins_streams_on_the_base() {
        let subs = BinanceSub::all();
        assert_eq!(
            combined_url("wss://stream.binance.com:9443", &subs),
            "wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/btcusdt@trade/ethusdt@bookTicker/ethusdt@trade"
        );
        // Trailing slash tolerated; subsets build correctly.
        assert_eq!(
            combined_url(
                "wss://data-stream.binance.vision/",
                &[BinanceSub::new(Asset::Eth, BinanceStream::Trade)]
            ),
            "wss://data-stream.binance.vision/stream?streams=ethusdt@trade"
        );
    }

    #[test]
    fn stream_and_symbol_mapping_round_trip() {
        for sub in BinanceSub::all() {
            assert_eq!(map_stream(&sub.stream_name()), Some(sub));
        }
        assert_eq!(map_stream("solusdt@trade"), None);
        assert_eq!(map_stream("btcusdt@aggTrade"), None);
        assert_eq!(map_symbol("BTCUSDT"), Some(Asset::Btc));
        assert_eq!(map_symbol("ethusdt"), Some(Asset::Eth));
        assert_eq!(map_symbol("SOLUSDT"), None);
    }

    #[test]
    fn wrapped_book_ticker_publishes_the_midpoint_stamped_at_now() {
        let frame = wrapped("btcusdt@bookTicker", BOOK);
        let BinanceParsed::Prices(updates) = parse_frame(&frame, NOW) else {
            panic!("expected prices");
        };
        assert_eq!(
            updates,
            vec![BinanceUpdate {
                sub: BinanceSub::new(Asset::Btc, BinanceStream::BookTicker),
                value: dec!(63500.02000000),
                ts_exchange: NOW,
            }]
        );
    }

    #[test]
    fn wrapped_trade_publishes_the_print_at_trade_time() {
        let frame = wrapped("ethusdt@trade", TRADE);
        let BinanceParsed::Prices(updates) = parse_frame(&frame, NOW) else {
            panic!("expected prices");
        };
        assert_eq!(
            updates,
            vec![BinanceUpdate {
                sub: BinanceSub::new(Asset::Eth, BinanceStream::Trade),
                value: dec!(1670.39000000),
                ts_exchange: TimestampMs::from_millis(1_672_515_782_134),
            }]
        );
    }

    #[test]
    fn trade_falls_back_to_event_time_without_trade_time() {
        let payload = r#"{"e":"trade","E":99,"s":"BTCUSDT","p":"63000.5"}"#;
        let BinanceParsed::Prices(updates) = parse_frame(payload, NOW) else {
            panic!("expected prices");
        };
        assert_eq!(updates[0].ts_exchange, TimestampMs::from_millis(99));
    }

    #[test]
    fn bare_payloads_parse_defensively() {
        // The raw /ws form delivers unwrapped payloads — identity comes from
        // the symbol + shape.
        let BinanceParsed::Prices(updates) = parse_frame(BOOK, NOW) else {
            panic!("expected prices");
        };
        assert_eq!(
            updates[0].sub,
            BinanceSub::new(Asset::Btc, BinanceStream::BookTicker)
        );
        let BinanceParsed::Prices(updates) = parse_frame(TRADE, NOW) else {
            panic!("expected prices");
        };
        assert_eq!(
            updates[0].sub,
            BinanceSub::new(Asset::Eth, BinanceStream::Trade)
        );
    }

    #[test]
    fn acks_and_errors_classify() {
        assert_eq!(
            parse_frame(r#"{"result":null,"id":1}"#, NOW),
            BinanceParsed::Ack
        );
        assert_eq!(
            parse_frame(r#"{"result":["btcusdt@trade"],"id":2}"#, NOW),
            BinanceParsed::Ack
        );
        assert_eq!(
            parse_frame(r#"{"code":2,"msg":"Invalid request"}"#, NOW),
            BinanceParsed::Ignored(IgnoredReason::ServerError)
        );
    }

    #[test]
    fn wrapper_and_payload_disagreement_is_malformed() {
        // Wrapper says ETH trade, payload says BTC trade.
        let frame = wrapped(
            "ethusdt@trade",
            r#"{"e":"trade","E":1,"s":"BTCUSDT","p":"63000.5","T":1}"#,
        );
        assert_eq!(
            parse_frame(&frame, NOW),
            BinanceParsed::Ignored(IgnoredReason::MalformedPayload)
        );
        // Wrapper says bookTicker, payload is a trade.
        let frame = wrapped(
            "btcusdt@bookTicker",
            r#"{"e":"trade","E":1,"s":"BTCUSDT","p":"63000.5","T":1}"#,
        );
        assert_eq!(
            parse_frame(&frame, NOW),
            BinanceParsed::Ignored(IgnoredReason::MalformedPayload)
        );
    }

    #[test]
    fn zero_or_negative_prices_never_publish() {
        let empty_book = r#"{"u":1,"s":"BTCUSDT","b":"0.00000000","B":"0","a":"63500.03","A":"1"}"#;
        assert_eq!(
            parse_frame(empty_book, NOW),
            BinanceParsed::Ignored(IgnoredReason::MalformedPayload)
        );
        let bad_trade = r#"{"e":"trade","E":1,"s":"BTCUSDT","p":"0","T":1}"#;
        assert_eq!(
            parse_frame(bad_trade, NOW),
            BinanceParsed::Ignored(IgnoredReason::MalformedPayload)
        );
    }

    #[test]
    fn malformed_frames_degrade_to_ignored_never_panic() {
        let cases: &[(&str, IgnoredReason)] = &[
            ("not json at all", IgnoredReason::MalformedJson),
            (
                r#"{"stream":"btcusdt@bookTicker","data":{"u":1"#,
                IgnoredReason::MalformedJson,
            ),
            (r#"[1,2,3]"#, IgnoredReason::MalformedPayload),
            (r#""a string""#, IgnoredReason::MalformedPayload),
            (r#"{}"#, IgnoredReason::MalformedPayload),
            // Unknown stream in the wrapper.
            (
                r#"{"stream":"solusdt@trade","data":{"e":"trade","s":"SOLUSDT","p":"1","T":1}}"#,
                IgnoredReason::UnknownStream,
            ),
            // Unknown event type (e.g. an aggTrade leaking through).
            (
                r#"{"e":"aggTrade","E":1,"s":"BTCUSDT","p":"1","T":1}"#,
                IgnoredReason::UnknownStream,
            ),
            // Untracked symbol in a known shape.
            (
                r#"{"e":"trade","E":1,"s":"SOLUSDT","p":"1","T":1}"#,
                IgnoredReason::UnknownStream,
            ),
            // bookTicker with a missing symbol / missing sides.
            (r#"{"b":"1.0","a":"2.0"}"#, IgnoredReason::MissingField),
            (
                r#"{"stream":"btcusdt@bookTicker","data":{"s":"BTCUSDT","b":"63500.01"}}"#,
                IgnoredReason::MalformedPayload,
            ),
            // Trade with no usable timestamp or no price.
            (
                r#"{"e":"trade","s":"BTCUSDT","p":"63000.5"}"#,
                IgnoredReason::MissingField,
            ),
            (
                r#"{"e":"trade","E":1,"s":"BTCUSDT","T":1}"#,
                IgnoredReason::MissingField,
            ),
            // Non-numeric price strings.
            (
                r#"{"e":"trade","E":1,"s":"BTCUSDT","p":"sixty-three","T":1}"#,
                IgnoredReason::MalformedPayload,
            ),
            // Wrapper with a non-object payload.
            (
                r#"{"stream":"btcusdt@trade","data":"nope"}"#,
                IgnoredReason::MalformedPayload,
            ),
        ];
        for (frame, expected) in cases {
            assert_eq!(
                parse_frame(frame, NOW),
                BinanceParsed::Ignored(*expected),
                "frame: {frame}"
            );
        }
    }

    #[test]
    fn midpoint_is_exact_decimal_arithmetic() {
        // Odd spread: the midpoint gains one decimal place, exactly.
        let book = r#"{"u":1,"s":"ETHUSDT","b":"1670.01","B":"1","a":"1670.02","A":"1"}"#;
        let BinanceParsed::Prices(updates) = parse_frame(book, NOW) else {
            panic!("expected prices");
        };
        assert_eq!(updates[0].value, dec!(1670.015));
    }
}
