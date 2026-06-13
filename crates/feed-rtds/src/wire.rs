//! RTDS wire protocol: subscribe/unsubscribe message builders and the
//! lenient inbound-frame parser.
//!
//! Wire facts (docs read in full 2026-06-12, then live-verified the same day
//! — the probe suite in `tests/live_probe.rs` reproduces every claim):
//! - Subscribe envelope: `{"action":"subscribe","subscriptions":[{topic,
//!   type, filters}]}`; `"unsubscribe"` removes; subscriptions are modifiable
//!   without reconnecting.
//! - **One filter slot per (connection, topic)**: a later filtered subscribe
//!   on the same topic REPLACES the earlier filter (it does not add). No
//!   multi-symbol filter form works (comma lists and JSON arrays — inside or
//!   outside the JSON-string form — are acked and deliver nothing). The only
//!   way to stream several symbols on one topic over one connection is the
//!   documented subscribe-to-all form (Binance: no `filters` field;
//!   Chainlink: `"filters":""` with type `"*"`), filtering client-side.
//! - A **filtered** subscribe (JSON-string filter, e.g.
//!   `"{\"symbol\":\"btcusdt\"}"`) triggers a ~2-minute backfill message for
//!   that symbol; the unfiltered form does not. Boot therefore sends
//!   filtered subscribes per tracked symbol (collecting backfills to seed
//!   the model) and then the unfiltered subscribe as the steady state.
//! - Live update payload: `{symbol, timestamp (ms), value (number),
//!   full_accuracy_value (string)}`. `full_accuracy_value` is the lossless
//!   value: Binance sends a plain decimal string (`"1670.39000000"`),
//!   Chainlink an 18-dp fixed-point integer (`"1668111205000000000000"`).
//! - Backfill payload: `{symbol, data: [{timestamp, value}, …]}` with type
//!   `"subscribe"` — and, for Chainlink subscriptions, a WRONG topic
//!   (`crypto_prices`). Symbols are disjoint across topics (`btcusdt` vs
//!   `btc/usd`), so stream identity is resolved from the symbol alone.
//! - The subscription ack is an empty text frame; the server answers our
//!   text `"PING"` with text `"PONG"`.
//!
//! Parsing never panics: anything unrecognized maps to
//! [`ParsedFrame::Ignored`] with a reason, for the driver to log and skip.

use core_types::{Asset, PriceSource, TimestampMs};
use rust_decimal::Decimal;
use serde::Deserialize;

/// Which RTDS crypto-price topic a subscription targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RtdsSource {
    /// Topic `crypto_prices` — Binance-sourced, symbols like `btcusdt`.
    Binance,
    /// Topic `crypto_prices_chainlink` — the resolution-grade feed, symbols
    /// like `btc/usd`.
    Chainlink,
}

impl RtdsSource {
    /// The RTDS topic name on the wire.
    #[must_use]
    pub const fn topic(self) -> &'static str {
        match self {
            Self::Binance => "crypto_prices",
            Self::Chainlink => "crypto_prices_chainlink",
        }
    }

    /// The subscription `type` field. Binance uses `"update"`; Chainlink
    /// subscribes with the wildcard `"*"` (both live-verified).
    #[must_use]
    pub const fn sub_type(self) -> &'static str {
        match self {
            Self::Binance => "update",
            Self::Chainlink => "*",
        }
    }

    /// The wire symbol for an asset on this topic.
    #[must_use]
    pub const fn symbol(self, asset: Asset) -> &'static str {
        match (self, asset) {
            (Self::Binance, Asset::Btc) => "btcusdt",
            (Self::Binance, Asset::Eth) => "ethusdt",
            (Self::Chainlink, Asset::Btc) => "btc/usd",
            (Self::Chainlink, Asset::Eth) => "eth/usd",
        }
    }

    /// Maps a wire topic name back to a source.
    #[must_use]
    pub fn from_topic(topic: &str) -> Option<Self> {
        match topic {
            "crypto_prices" => Some(Self::Binance),
            "crypto_prices_chainlink" => Some(Self::Chainlink),
            _ => None,
        }
    }

    /// The bus-level [`PriceSource`] this topic publishes as.
    #[must_use]
    pub const fn price_source(self) -> PriceSource {
        match self {
            Self::Binance => PriceSource::BinanceRtds,
            Self::Chainlink => PriceSource::ChainlinkRtds,
        }
    }
}

impl std::fmt::Display for RtdsSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binance => f.write_str("binance"),
            Self::Chainlink => f.write_str("chainlink"),
        }
    }
}

/// Resolves a wire symbol to its stream, regardless of topic — symbol
/// namespaces are disjoint (`btcusdt` vs `btc/usd`), and backfill frames for
/// Chainlink subscriptions arrive under the WRONG topic (live-verified), so
/// the symbol is the only trustworthy identity. Case-insensitive (the server
/// returns lowercase).
#[must_use]
pub fn map_symbol(symbol: &str) -> Option<(RtdsSource, Asset)> {
    let lower = symbol.to_ascii_lowercase();
    for source in [RtdsSource::Binance, RtdsSource::Chainlink] {
        for asset in Asset::ALL {
            if source.symbol(asset) == lower {
                return Some((source, asset));
            }
        }
    }
    None
}

/// Builds the filtered (single-symbol) subscribe for one stream. Its only
/// lasting effect is the ~2-minute backfill message it triggers — the filter
/// it installs is superseded by [`stream_subscribe_message`] right after.
#[must_use]
pub fn backfill_subscribe_message(source: RtdsSource, asset: Asset) -> String {
    // The JSON-string `filters` form is the only one the server honors
    // (Decisions Log 2026-06-12) — encode the inner object first to keep
    // escaping correct.
    let filters = serde_json::json!({ "symbol": source.symbol(asset) }).to_string();
    serde_json::json!({
        "action": "subscribe",
        "subscriptions": [{
            "topic": source.topic(),
            "type": source.sub_type(),
            "filters": filters,
        }],
    })
    .to_string()
}

/// Builds the steady-state (all-symbols) subscribe for a topic — the only
/// live-verified way to stream more than one symbol per topic; untracked
/// symbols are dropped client-side.
#[must_use]
pub fn stream_subscribe_message(source: RtdsSource) -> String {
    stream_action_message("subscribe", source)
}

/// Builds the unsubscribe for a topic (mirror of
/// [`stream_subscribe_message`]).
#[must_use]
pub fn stream_unsubscribe_message(source: RtdsSource) -> String {
    stream_action_message("unsubscribe", source)
}

fn stream_action_message(action: &str, source: RtdsSource) -> String {
    // Exactly the live-verified all-symbols shapes: Binance omits `filters`
    // entirely; Chainlink wants an explicit empty string with type "*".
    match source {
        RtdsSource::Binance => serde_json::json!({
            "action": action,
            "subscriptions": [{
                "topic": source.topic(),
                "type": source.sub_type(),
            }],
        })
        .to_string(),
        RtdsSource::Chainlink => serde_json::json!({
            "action": action,
            "subscriptions": [{
                "topic": source.topic(),
                "type": source.sub_type(),
                "filters": "",
            }],
        })
        .to_string(),
    }
}

/// One normalized price observation extracted from an inbound frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceUpdate {
    /// Producing topic.
    pub source: RtdsSource,
    /// Underlying asset.
    pub asset: Asset,
    /// Price in underlying units. Wire-exact: prefers the lossless
    /// `full_accuracy_value` (validated against the float `value`) over the
    /// float.
    pub value: Decimal,
    /// Source timestamp from the payload (envelope timestamp as fallback).
    pub ts_exchange: TimestampMs,
}

/// Why an inbound frame (or payload entry) was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IgnoredReason {
    /// Text frame that is not valid JSON.
    MalformedJson,
    /// A server error frame (`{"body":{"message":…},"statusCode":500}`) —
    /// observed live answering a subscribe (2026-06-12). The affected
    /// subscription is undefined server-side; if a tracked stream starves
    /// because of it, the starvation watchdog recycles the connection.
    ServerError,
    /// Missing or unrecognized `topic`.
    UnknownTopic,
    /// `payload` absent, of the wrong shape, or an empty/valueless backfill.
    MalformedPayload,
    /// Symbol present but not one we track (expected under the all-symbols
    /// subscription — the driver logs this at trace, not warn).
    UnknownSymbol,
    /// A required payload field (`symbol`, `value`, or any timestamp) absent.
    MissingField,
}

impl std::fmt::Display for IgnoredReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::MalformedJson => "malformed JSON",
            Self::ServerError => "server error frame",
            Self::UnknownTopic => "unknown topic",
            Self::MalformedPayload => "malformed payload",
            Self::UnknownSymbol => "untracked symbol",
            Self::MissingField => "missing required field",
        };
        f.write_str(s)
    }
}

/// Classification of one inbound text frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedFrame {
    /// Empty text frame — the RTDS subscription ack.
    Ack,
    /// The server's `"PONG"` keepalive reply.
    Pong,
    /// One or more price observations (a backfill carries up to ~120).
    Prices(Vec<PriceUpdate>),
    /// Skipped frame with the reason (driver logs and moves on).
    Ignored(IgnoredReason),
}

/// Lenient envelope: every field optional, unknown fields tolerated (e.g.
/// `connection_id`) — the venue adds fields freely and an odd frame must
/// never panic or kill the stream.
#[derive(Debug, Deserialize)]
struct WireEnvelope {
    topic: Option<String>,
    timestamp: Option<i64>,
    payload: Option<serde_json::Value>,
    /// Present only on server error frames.
    #[serde(rename = "statusCode")]
    status_code: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct WirePricePayload {
    symbol: Option<String>,
    timestamp: Option<i64>,
    value: Option<Decimal>,
    full_accuracy_value: Option<String>,
    /// Present only on backfill frames.
    data: Option<Vec<WireBackfillPoint>>,
}

#[derive(Debug, Deserialize)]
struct WireBackfillPoint {
    timestamp: Option<i64>,
    value: Option<Decimal>,
}

/// Parses one inbound text frame. Never panics; never errors — unparseable
/// input degrades to [`ParsedFrame::Ignored`].
#[must_use]
pub fn parse_frame(text: &str) -> ParsedFrame {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return ParsedFrame::Ack;
    }
    if trimmed == "PONG" {
        return ParsedFrame::Pong;
    }
    let Ok(envelope) = serde_json::from_str::<WireEnvelope>(trimmed) else {
        return ParsedFrame::Ignored(IgnoredReason::MalformedJson);
    };
    // Server error frames carry a statusCode instead of a topic — surface
    // them distinctly (the WarnGate preview shows the message).
    if envelope.status_code.is_some() {
        return ParsedFrame::Ignored(IgnoredReason::ServerError);
    }
    // The topic gates "is this a crypto-price frame at all"; stream identity
    // comes from the symbol (backfill topics lie — module docs).
    if envelope
        .topic
        .as_deref()
        .and_then(RtdsSource::from_topic)
        .is_none()
    {
        return ParsedFrame::Ignored(IgnoredReason::UnknownTopic);
    }
    let Some(payload) = envelope.payload else {
        return ParsedFrame::Ignored(IgnoredReason::MalformedPayload);
    };
    let entries: Vec<serde_json::Value> = match payload {
        serde_json::Value::Array(items) => items,
        object @ serde_json::Value::Object(_) => vec![object],
        _ => return ParsedFrame::Ignored(IgnoredReason::MalformedPayload),
    };
    if entries.is_empty() {
        return ParsedFrame::Ignored(IgnoredReason::MalformedPayload);
    }

    let mut updates = Vec::with_capacity(entries.len());
    let mut first_reason = None;
    for entry in entries {
        match parse_entry(entry, envelope.timestamp, &mut updates) {
            Ok(()) => {}
            Err(reason) => first_reason = first_reason.or(Some(reason)),
        }
    }
    if updates.is_empty() {
        ParsedFrame::Ignored(first_reason.unwrap_or(IgnoredReason::MalformedPayload))
    } else {
        ParsedFrame::Prices(updates)
    }
}

/// Parses one payload object — a live update or a backfill — appending its
/// observation(s) to `updates`.
fn parse_entry(
    entry: serde_json::Value,
    envelope_ts: Option<i64>,
    updates: &mut Vec<PriceUpdate>,
) -> Result<(), IgnoredReason> {
    let Ok(payload) = serde_json::from_value::<WirePricePayload>(entry) else {
        return Err(IgnoredReason::MalformedPayload);
    };
    let symbol = payload.symbol.ok_or(IgnoredReason::MissingField)?;
    let (source, asset) = map_symbol(&symbol).ok_or(IgnoredReason::UnknownSymbol)?;

    if let Some(points) = payload.data {
        // Backfill: one observation per complete point, in wire (oldest →
        // newest) order — seeds the vol estimator at boot/reconnect.
        let before = updates.len();
        for point in points {
            if let (Some(ts), Some(value)) = (point.timestamp, point.value) {
                updates.push(PriceUpdate {
                    source,
                    asset,
                    value,
                    ts_exchange: TimestampMs::from_millis(ts),
                });
            }
        }
        if updates.len() == before {
            return Err(IgnoredReason::MalformedPayload);
        }
        return Ok(());
    }

    let value = entry_value(payload.value, payload.full_accuracy_value.as_deref())
        .ok_or(IgnoredReason::MissingField)?;
    // Payload timestamp is the source time; the envelope timestamp (RTDS
    // server time) is an acceptable fallback. A tick with neither is useless
    // to the model.
    let ts = payload
        .timestamp
        .or(envelope_ts)
        .ok_or(IgnoredReason::MissingField)?;
    updates.push(PriceUpdate {
        source,
        asset,
        value,
        ts_exchange: TimestampMs::from_millis(ts),
    });
    Ok(())
}

/// 10^18 — Chainlink's `full_accuracy_value` fixed-point scale.
const WEI_SCALE: Decimal = rust_decimal::dec!(1_000_000_000_000_000_000);

/// Picks the most precise trustworthy value. `full_accuracy_value` is
/// preferred (lossless), but only in an interpretation consistent with the
/// float `value` (within 0.1%) — Binance sends it as a plain decimal string,
/// Chainlink as an 18-dp fixed-point integer, and self-validating against
/// the float means a future format change degrades gracefully to the float
/// instead of publishing a wildly scaled price.
fn entry_value(value: Option<Decimal>, full_accuracy: Option<&str>) -> Option<Decimal> {
    if let Some(s) = full_accuracy
        && let Ok(direct) = Decimal::from_str_exact(s.trim())
    {
        match value {
            Some(float_value) => {
                for candidate in [direct, direct / WEI_SCALE] {
                    if close_to(candidate, float_value) {
                        return Some(candidate);
                    }
                }
            }
            // No float to validate against: accept only the unambiguous
            // plain-decimal form.
            None if s.contains('.') => return Some(direct),
            None => {}
        }
    }
    value
}

/// Within 0.1% relative (the float is a rounded rendering of the true
/// value, so genuine matches differ by ~1e-12 relative).
fn close_to(candidate: Decimal, reference: Decimal) -> bool {
    (candidate - reference).abs() <= reference.abs() * rust_decimal::dec!(0.001)
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    #[test]
    fn backfill_subscribe_messages_are_pinned_to_live_verified_shapes() {
        // Byte-for-byte pins; the JSON-string filter is the only form that
        // triggers the backfill (Decisions Log 2026-06-12). serde_json
        // orders keys alphabetically (BTreeMap), matching the harness pins.
        assert_eq!(
            backfill_subscribe_message(RtdsSource::Binance, Asset::Btc),
            r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"btcusdt\"}","topic":"crypto_prices","type":"update"}]}"#
        );
        assert_eq!(
            backfill_subscribe_message(RtdsSource::Binance, Asset::Eth),
            r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"ethusdt\"}","topic":"crypto_prices","type":"update"}]}"#
        );
        assert_eq!(
            backfill_subscribe_message(RtdsSource::Chainlink, Asset::Btc),
            r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"btc/usd\"}","topic":"crypto_prices_chainlink","type":"*"}]}"#
        );
        assert_eq!(
            backfill_subscribe_message(RtdsSource::Chainlink, Asset::Eth),
            r#"{"action":"subscribe","subscriptions":[{"filters":"{\"symbol\":\"eth/usd\"}","topic":"crypto_prices_chainlink","type":"*"}]}"#
        );
    }

    #[test]
    fn stream_subscribe_messages_are_pinned_to_live_verified_shapes() {
        // The all-symbols forms (probe variants D and F): Binance has no
        // filters field at all; Chainlink wants filters:"" with type "*".
        assert_eq!(
            stream_subscribe_message(RtdsSource::Binance),
            r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices","type":"update"}]}"#
        );
        assert_eq!(
            stream_subscribe_message(RtdsSource::Chainlink),
            r#"{"action":"subscribe","subscriptions":[{"filters":"","topic":"crypto_prices_chainlink","type":"*"}]}"#
        );
        for source in [RtdsSource::Binance, RtdsSource::Chainlink] {
            assert_eq!(
                stream_unsubscribe_message(source),
                stream_subscribe_message(source)
                    .replace(r#""action":"subscribe""#, r#""action":"unsubscribe""#)
            );
        }
    }

    #[test]
    fn symbol_mapping_round_trips_and_is_global() {
        for source in [RtdsSource::Binance, RtdsSource::Chainlink] {
            for asset in Asset::ALL {
                assert_eq!(map_symbol(source.symbol(asset)), Some((source, asset)));
            }
        }
        // Case-insensitive (server returns lowercase, but be tolerant).
        assert_eq!(
            map_symbol("BTCUSDT"),
            Some((RtdsSource::Binance, Asset::Btc))
        );
        assert_eq!(
            map_symbol("ETH/USD"),
            Some((RtdsSource::Chainlink, Asset::Eth))
        );
        // Untracked symbols (normal under the all-symbols subscription).
        assert_eq!(map_symbol("solusdt"), None);
        assert_eq!(map_symbol("doge/usd"), None);
    }

    #[test]
    fn ack_and_pong_frames_classify() {
        assert_eq!(parse_frame(""), ParsedFrame::Ack);
        assert_eq!(parse_frame("   "), ParsedFrame::Ack);
        assert_eq!(parse_frame("PONG"), ParsedFrame::Pong);
        assert_eq!(parse_frame(" PONG "), ParsedFrame::Pong);
    }

    #[test]
    fn binance_update_prefers_full_accuracy_string() {
        // Verbatim live frame shape (captured 2026-06-12): full_accuracy is
        // a plain decimal string for Binance.
        let frame = r#"{"connection_id":"gQsKZ-7HgWeIKAIj4A==","payload":{"full_accuracy_value":"1670.39000000","symbol":"ethusdt","timestamp":1781226222000,"value":1670.39},"timestamp":1781226222137,"topic":"crypto_prices","type":"update"}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(
            updates,
            vec![PriceUpdate {
                source: RtdsSource::Binance,
                asset: Asset::Eth,
                value: dec!(1670.39000000),
                ts_exchange: TimestampMs::from_millis(1_781_226_222_000),
            }]
        );
    }

    #[test]
    fn chainlink_update_descales_full_accuracy_18dp() {
        // Verbatim live frame shape: Chainlink full_accuracy is value×10^18;
        // the float loses digits past f64 — the descaled string is exact.
        let frame = r#"{"connection_id":"x","payload":{"full_accuracy_value":"1668291840491238500000","symbol":"eth/usd","timestamp":1781226222000,"value":1668.2918404912384},"timestamp":1781226223571,"topic":"crypto_prices_chainlink","type":"update"}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates[0].source, RtdsSource::Chainlink);
        assert_eq!(updates[0].asset, Asset::Eth);
        assert_eq!(updates[0].value, dec!(1668.2918404912385));
    }

    #[test]
    fn implausible_full_accuracy_falls_back_to_float() {
        // Neither interpretation (direct / 18-dp descale) matches the float:
        // trust the float.
        let frame = r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt","timestamp":1,"value":63500.5,"full_accuracy_value":"99999"}}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates[0].value, dec!(63500.5));
        // Unparseable string: same fallback.
        let frame = r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt","timestamp":1,"value":63500.5,"full_accuracy_value":"n/a"}}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates[0].value, dec!(63500.5));
    }

    #[test]
    fn update_without_full_accuracy_uses_float_exactly() {
        let frame = r#"{"topic":"crypto_prices","type":"update","timestamp":1753314064237,"payload":{"symbol":"btcusdt","timestamp":1753314064213,"value":104233.55}}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        // rust_decimal decodes JSON numbers via shortest round-trip
        // (discovery precedent) — 104233.55 stays exactly 104233.55.
        assert_eq!(updates[0].value, dec!(104233.55));
        assert_eq!(
            updates[0].ts_exchange,
            TimestampMs::from_millis(1_753_314_064_213)
        );
    }

    #[test]
    fn backfill_parses_every_point_with_symbol_resolved_identity() {
        // Shape from the live capture — note the WRONG topic
        // (crypto_prices) on a Chainlink backfill: identity comes from the
        // slash symbol, not the topic.
        let frame = r#"{"payload":{"data":[
            {"timestamp":1781226161000,"value":63433.56803386909},
            {"timestamp":1781226162000,"value":63428.666733545186},
            {"timestamp":1781226163000,"value":63416.99062163242}
        ],"symbol":"btc/usd"},"timestamp":1781226220973,"topic":"crypto_prices","type":"subscribe"}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates.len(), 3);
        assert!(
            updates
                .iter()
                .all(|u| u.source == RtdsSource::Chainlink && u.asset == Asset::Btc)
        );
        // Oldest → newest, per-point timestamps.
        assert_eq!(
            updates[0].ts_exchange,
            TimestampMs::from_millis(1_781_226_161_000)
        );
        assert_eq!(updates[0].value, dec!(63433.56803386909));
        assert_eq!(
            updates[2].ts_exchange,
            TimestampMs::from_millis(1_781_226_163_000)
        );
    }

    #[test]
    fn backfill_skips_incomplete_points_but_keeps_the_rest() {
        let frame = r#"{"topic":"crypto_prices","type":"subscribe","payload":{"symbol":"btcusdt","data":[
            {"timestamp":1,"value":100.0},
            {"timestamp":2},
            {"value":300.0},
            {"timestamp":4,"value":400.0}
        ]}}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[1].value, dec!(400.0));

        // All points incomplete → the frame is malformed, not empty-OK.
        let frame = r#"{"topic":"crypto_prices","type":"subscribe","payload":{"symbol":"btcusdt","data":[{"timestamp":2}]}}"#;
        assert_eq!(
            parse_frame(frame),
            ParsedFrame::Ignored(IgnoredReason::MalformedPayload)
        );
    }

    #[test]
    fn server_error_frames_classify_distinctly() {
        // Live-captured (2026-06-12): RTDS answered a subscribe with a 500.
        let frame = r#"{"body":{"message":"leger AddSubscriptions error: rpc error: code = Internal desc = ERROR #42P01"},"statusCode":500}"#;
        assert_eq!(
            parse_frame(frame),
            ParsedFrame::Ignored(IgnoredReason::ServerError)
        );
    }

    #[test]
    fn untracked_symbol_is_its_own_reason() {
        // Normal traffic under the all-symbols subscription — the driver
        // logs this at trace level, never warn.
        let frame = r#"{"topic":"crypto_prices","type":"update","payload":{"symbol":"solusdt","timestamp":1,"value":189.55}}"#;
        assert_eq!(
            parse_frame(frame),
            ParsedFrame::Ignored(IgnoredReason::UnknownSymbol)
        );
        let frame = r#"{"topic":"crypto_prices_chainlink","type":"update","payload":{"symbol":"hype/usd","timestamp":1,"value":1.0}}"#;
        assert_eq!(
            parse_frame(frame),
            ParsedFrame::Ignored(IgnoredReason::UnknownSymbol)
        );
    }

    #[test]
    fn envelope_timestamp_is_the_fallback() {
        let frame = r#"{"topic":"crypto_prices","timestamp":99,"payload":{"symbol":"btcusdt","value":1.0}}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates[0].ts_exchange, TimestampMs::from_millis(99));
    }

    #[test]
    fn malformed_frames_degrade_to_ignored_never_panic() {
        let cases: &[(&str, IgnoredReason)] = &[
            ("not json at all", IgnoredReason::MalformedJson),
            (
                r#"{"topic":"crypto_prices","payload":{"sym"#,
                IgnoredReason::MalformedJson,
            ),
            (r#"{"no_topic":true}"#, IgnoredReason::UnknownTopic),
            (
                r#"{"topic":"comments","type":"comment_created","payload":{}}"#,
                IgnoredReason::UnknownTopic,
            ),
            (
                r#"{"topic":"crypto_prices"}"#,
                IgnoredReason::MalformedPayload,
            ),
            (
                r#"{"topic":"crypto_prices","payload":"a string"}"#,
                IgnoredReason::MalformedPayload,
            ),
            (
                r#"{"topic":"crypto_prices","payload":[]}"#,
                IgnoredReason::MalformedPayload,
            ),
            (
                r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt","timestamp":1,"value":"not-a-number"}}"#,
                IgnoredReason::MalformedPayload,
            ),
            (
                r#"{"topic":"crypto_prices","payload":{"timestamp":1,"value":1.0}}"#,
                IgnoredReason::MissingField,
            ),
            (
                r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt","timestamp":1}}"#,
                IgnoredReason::MissingField,
            ),
            (
                r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt","value":1.0}}"#,
                IgnoredReason::MissingField,
            ),
        ];
        for (frame, expected) in cases {
            assert_eq!(
                parse_frame(frame),
                ParsedFrame::Ignored(*expected),
                "frame: {frame}"
            );
        }
    }

    #[test]
    fn mixed_array_keeps_good_entries() {
        // Defensive: array payloads have not been observed live, but if one
        // arrives, good entries survive bad ones.
        let frame = r#"{"topic":"crypto_prices","payload":[
            {"symbol":"solusdt","timestamp":1,"value":1.0},
            {"symbol":"btcusdt","timestamp":2,"value":100.0}
        ]}"#;
        let ParsedFrame::Prices(updates) = parse_frame(frame) else {
            panic!("expected prices");
        };
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].asset, Asset::Btc);
    }
}
