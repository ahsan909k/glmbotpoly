//! Pure authenticated user-channel wire protocol: the subscribe-message builder
//! and the frame parser. No IO, no clocks (the caller passes `now` for the
//! timestamp fallback), no logging — fixture tests drive this module directly.
//!
//! Wire facts (docs `market-data/websocket/user-channel`, verified 2026-06-13;
//! independently cross-checked against the SDK 0.5.1 `clob::ws` response types):
//! the subscribe message nests the API credentials under `auth` and lists
//! **condition ids** under `markets`; every event object carries an
//! `event_type` tag (`order` or `trade`); a text frame may carry a single event
//! object or a JSON array of them; numerics arrive as strings (parsed
//! defensively as string-or-number); the client sends the text `PING` and the
//! server replies `PONG`.

use core_types::{ConditionId, Decimal, OrderId, Side, TimestampMs, TokenId};
use serde::Deserialize;

/// The client keepalive text. The docs require one at least every 10 s; the
/// driver sends it every 5 s (project convention).
pub const PING_TEXT: &str = "PING";

/// Builds the user-channel subscribe message, sent once immediately after every
/// (re)connect. Credentials nest under `auth`; `markets` lists condition ids
/// (an empty list is valid — see [`LiveParams::subscribe_all_when_empty`]).
///
/// [`LiveParams::subscribe_all_when_empty`]: crate::LiveParams::subscribe_all_when_empty
#[must_use]
pub fn subscribe_message<'a>(
    api_key: &str,
    secret: &str,
    passphrase: &str,
    markets: impl IntoIterator<Item = &'a ConditionId>,
) -> String {
    let markets: Vec<&str> = markets.into_iter().map(ConditionId::as_str).collect();
    serde_json::json!({
        "auth": { "apiKey": api_key, "secret": secret, "passphrase": passphrase },
        "markets": markets,
        "type": "user",
    })
    .to_string()
}

/// Why a frame (or one event inside an array frame) was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IgnoredReason {
    /// Not JSON, or JSON of a shape that cannot hold events.
    MalformedJson,
    /// An `event_type` this client does not know.
    UnknownEventType,
    /// A required field was absent.
    MissingField,
    /// A field was present but unparseable (bad id, bad decimal, bad side, or
    /// an unrecognized order type / trade status).
    BadValue,
}

impl std::fmt::Display for IgnoredReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MalformedJson => "malformed JSON",
            Self::UnknownEventType => "unknown event_type",
            Self::MissingField => "missing required field",
            Self::BadValue => "unparseable field value",
        })
    }
}

/// The `type` descriptor on an `order` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderEventKind {
    /// The order was placed and is resting.
    Placement,
    /// The order's matched size changed (a fill) or it was otherwise updated.
    Update,
    /// The order was cancelled (terminal).
    Cancellation,
}

/// One `order` event: an order placement / update / cancellation. `size_matched`
/// is cumulative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireOrder {
    /// Which kind of order event this is.
    pub kind: OrderEventKind,
    /// Venue order id.
    pub order_id: OrderId,
    /// Market (condition id) the order trades.
    pub condition_id: ConditionId,
    /// Outcome token the order trades.
    pub token_id: TokenId,
    /// Order side.
    pub side: Side,
    /// Limit price (wire value; informational for tracked orders).
    pub price: Decimal,
    /// Original order size in shares.
    pub original_size: Decimal,
    /// Cumulative matched size in shares.
    pub size_matched: Decimal,
    /// Wire timestamp (caller's `now` when absent/unparseable).
    pub ts: TimestampMs,
}

/// One leg of a trade in which a resting (maker) order participated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireMakerOrder {
    /// The maker order's venue id.
    pub order_id: OrderId,
    /// The maker order's outcome token.
    pub token_id: TokenId,
    /// Shares of the maker order matched in this trade.
    pub matched_amount: Decimal,
    /// The maker order's limit price.
    pub price: Decimal,
}

/// The lifecycle status of a `trade` event. `Matched`/`Mined`/`Confirmed` are
/// progressively-final stages of a real fill; `Retrying` is in-flight; `Failed`
/// is terminal-negative (the trade did not happen).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeStatus {
    /// Initial match (the fill happened on the book).
    Matched,
    /// The settlement transaction was mined.
    Mined,
    /// Settlement confirmed (terminal-positive).
    Confirmed,
    /// Settlement is being retried (in-flight).
    Retrying,
    /// The trade failed (terminal-negative — no fill).
    Failed,
}

impl TradeStatus {
    /// True for statuses that are not yet a settled outcome (`Retrying`) or are
    /// terminal-negative (`Failed`) — neither counts as a fill, and neither
    /// should consume the trade id (a retry may still settle). The remaining
    /// statuses (`Matched`/`Mined`/`Confirmed`) are real fills.
    #[must_use]
    pub const fn is_inflight_or_failed(self) -> bool {
        matches!(self, Self::Retrying | Self::Failed)
    }
}

/// One `trade` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireTrade {
    /// Venue trade id (deduplication key).
    pub trade_id: String,
    /// The taker order's id, when present (it is ours iff we tracked it).
    pub taker_order_id: Option<OrderId>,
    /// Market (condition id) the trade is on.
    pub condition_id: ConditionId,
    /// Outcome token traded (the taker side's token).
    pub token_id: TokenId,
    /// The taker's side.
    pub side: Side,
    /// Trade size in shares (the taker fill amount).
    pub size: Decimal,
    /// Trade price.
    pub price: Decimal,
    /// Trade lifecycle status.
    pub status: TradeStatus,
    /// Maker orders that participated (each may or may not be ours).
    pub maker_orders: Vec<WireMakerOrder>,
    /// Wire timestamp — `matchtime` preferred, then `timestamp`, then `now`.
    pub ts: TimestampMs,
}

/// One typed, validated user-channel event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireUserEvent {
    /// An order placement / update / cancellation.
    Order(WireOrder),
    /// A trade (fill) print.
    Trade(WireTrade),
}

/// One classified inbound text frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedUserFrame {
    /// Empty text frame — ack-style chatter, nothing to do.
    Ack,
    /// The server's `PONG` keepalive reply (liveness evidence).
    Pong,
    /// Parsed events — an array frame carries several; a per-event failure
    /// surfaces as `Err` without poisoning its siblings.
    Events(Vec<Result<WireUserEvent, IgnoredReason>>),
    /// The whole frame was unusable.
    Ignored(IgnoredReason),
}

/// Classifies one inbound text frame. `now` substitutes for absent or
/// unparseable wire timestamps so downstream events are always stamped.
#[must_use]
pub fn parse_user_frame(text: &str, now: TimestampMs) -> ParsedUserFrame {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return ParsedUserFrame::Ack;
    }
    if trimmed.eq_ignore_ascii_case("PONG") {
        return ParsedUserFrame::Pong;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return ParsedUserFrame::Ignored(IgnoredReason::MalformedJson);
    };
    let objects = match value {
        serde_json::Value::Array(items) => items,
        object @ serde_json::Value::Object(_) => vec![object],
        _ => return ParsedUserFrame::Ignored(IgnoredReason::MalformedJson),
    };
    if objects.is_empty() {
        return ParsedUserFrame::Ack;
    }
    ParsedUserFrame::Events(
        objects
            .into_iter()
            .map(|object| parse_event(object, now))
            .collect(),
    )
}

/// String-or-number wire field (Polymarket sends numerics as strings, but a
/// format drift to bare numbers must not break parsing).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StrNum {
    Text(String),
    Int(i64),
    Float(f64),
}

impl StrNum {
    fn to_decimal(&self) -> Option<Decimal> {
        match self {
            Self::Text(s) => {
                let t = s.trim();
                if t.is_empty() { None } else { t.parse().ok() }
            }
            Self::Int(i) => Some(Decimal::from(*i)),
            Self::Float(f) => {
                use rust_decimal::prelude::FromPrimitive as _;
                Decimal::from_f64(*f)
            }
        }
    }

    fn to_ts(&self) -> Option<TimestampMs> {
        match self {
            Self::Text(s) => s.trim().parse::<i64>().ok().map(TimestampMs::from_millis),
            Self::Int(i) => Some(TimestampMs::from_millis(*i)),
            Self::Float(f) if f.is_finite() => Some(TimestampMs::from_millis(*f as i64)),
            Self::Float(_) => None,
        }
    }
}

/// A decimal that defaults to zero when the wire field is absent/unparseable
/// (used for non-load-bearing sizes/prices; the store treats a missing
/// cumulative fill as "no progress", never as a regression).
fn decimal_or_zero(value: Option<&StrNum>) -> Decimal {
    value.and_then(StrNum::to_decimal).unwrap_or(Decimal::ZERO)
}

fn ts_or(now: TimestampMs, wire: Option<&StrNum>) -> TimestampMs {
    wire.and_then(StrNum::to_ts).unwrap_or(now)
}

fn parse_side(s: &str) -> Option<Side> {
    if s.eq_ignore_ascii_case("BUY") {
        Some(Side::Buy)
    } else if s.eq_ignore_ascii_case("SELL") {
        Some(Side::Sell)
    } else {
        None
    }
}

fn parse_order_kind(s: Option<&str>) -> OrderEventKind {
    match s
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_uppercase()
        .as_str()
    {
        "PLACEMENT" => OrderEventKind::Placement,
        "CANCELLATION" => OrderEventKind::Cancellation,
        // UPDATE, or any drift: a state refresh carrying the cumulative fill.
        _ => OrderEventKind::Update,
    }
}

fn parse_trade_status(s: &str) -> Option<TradeStatus> {
    match s.trim().to_ascii_uppercase().as_str() {
        "MATCHED" => Some(TradeStatus::Matched),
        "MINED" => Some(TradeStatus::Mined),
        "CONFIRMED" => Some(TradeStatus::Confirmed),
        "RETRYING" => Some(TradeStatus::Retrying),
        "FAILED" => Some(TradeStatus::Failed),
        _ => None,
    }
}

// --- permissive wire structs (no deny_unknown_fields: the venue adds fields
// --- freely; market-channel-parser precedent) ---

#[derive(Deserialize)]
struct RawOrder {
    #[serde(rename = "type")]
    order_type: Option<String>,
    id: String,
    market: String,
    asset_id: String,
    side: String,
    price: Option<StrNum>,
    original_size: Option<StrNum>,
    size_matched: Option<StrNum>,
    timestamp: Option<StrNum>,
}

#[derive(Deserialize)]
struct RawMakerOrder {
    order_id: String,
    asset_id: String,
    matched_amount: Option<StrNum>,
    price: Option<StrNum>,
}

#[derive(Deserialize)]
struct RawTrade {
    id: String,
    taker_order_id: Option<String>,
    market: String,
    asset_id: String,
    side: String,
    size: Option<StrNum>,
    price: Option<StrNum>,
    status: String,
    #[serde(default)]
    maker_orders: Vec<RawMakerOrder>,
    matchtime: Option<StrNum>,
    timestamp: Option<StrNum>,
}

fn parse_event(
    object: serde_json::Value,
    now: TimestampMs,
) -> Result<WireUserEvent, IgnoredReason> {
    let Some(event_type) = object.get("event_type").and_then(|v| v.as_str()) else {
        return Err(IgnoredReason::MissingField);
    };
    match event_type {
        "order" => {
            let wire: RawOrder =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            Ok(WireUserEvent::Order(WireOrder {
                kind: parse_order_kind(wire.order_type.as_deref()),
                order_id: OrderId::new(wire.id).map_err(|_| IgnoredReason::BadValue)?,
                condition_id: ConditionId::new(wire.market).map_err(|_| IgnoredReason::BadValue)?,
                token_id: TokenId::new(wire.asset_id).map_err(|_| IgnoredReason::BadValue)?,
                side: parse_side(&wire.side).ok_or(IgnoredReason::BadValue)?,
                price: decimal_or_zero(wire.price.as_ref()),
                original_size: decimal_or_zero(wire.original_size.as_ref()),
                size_matched: decimal_or_zero(wire.size_matched.as_ref()),
                ts: ts_or(now, wire.timestamp.as_ref()),
            }))
        }
        "trade" => {
            let wire: RawTrade =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            let status = parse_trade_status(&wire.status).ok_or(IgnoredReason::BadValue)?;
            // A trade id is the dedup key — it must be present and non-empty.
            if wire.id.trim().is_empty() {
                return Err(IgnoredReason::BadValue);
            }
            let taker_order_id = wire
                .taker_order_id
                .filter(|s| !s.trim().is_empty())
                .map(OrderId::new)
                .transpose()
                .map_err(|_| IgnoredReason::BadValue)?;
            // Maker legs are parsed leniently: a malformed leg (often another
            // participant's order) is dropped, never a reason to skip the trade.
            let maker_orders = wire
                .maker_orders
                .into_iter()
                .filter_map(parse_maker_order)
                .collect();
            Ok(WireUserEvent::Trade(WireTrade {
                trade_id: wire.id,
                taker_order_id,
                condition_id: ConditionId::new(wire.market).map_err(|_| IgnoredReason::BadValue)?,
                token_id: TokenId::new(wire.asset_id).map_err(|_| IgnoredReason::BadValue)?,
                side: parse_side(&wire.side).ok_or(IgnoredReason::BadValue)?,
                size: decimal_or_zero(wire.size.as_ref()),
                price: decimal_or_zero(wire.price.as_ref()),
                status,
                maker_orders,
                ts: ts_or(now, wire.matchtime.as_ref().or(wire.timestamp.as_ref())),
            }))
        }
        _ => Err(IgnoredReason::UnknownEventType),
    }
}

fn parse_maker_order(raw: RawMakerOrder) -> Option<WireMakerOrder> {
    Some(WireMakerOrder {
        order_id: OrderId::new(raw.order_id).ok()?,
        token_id: TokenId::new(raw.asset_id).ok()?,
        matched_amount: decimal_or_zero(raw.matched_amount.as_ref()),
        price: decimal_or_zero(raw.price.as_ref()),
    })
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const NOW: TimestampMs = TimestampMs::from_millis(1_700_000_000_000);

    fn cid(byte: &str) -> String {
        format!("0x{}", byte.repeat(32))
    }

    fn single(frame: &str) -> Result<WireUserEvent, IgnoredReason> {
        match parse_user_frame(frame, NOW) {
            ParsedUserFrame::Events(mut events) => {
                assert_eq!(events.len(), 1, "expected exactly one event");
                events.remove(0)
            }
            other => panic!("expected events, got {other:?}"),
        }
    }

    #[test]
    fn subscribe_message_is_pinned() {
        let up = ConditionId::new(cid("ab")).unwrap();
        let down = ConditionId::new(cid("cd")).unwrap();
        let msg = subscribe_message("key-1", "sec-1", "pass-1", [&up, &down]);
        assert_eq!(
            msg,
            format!(
                r#"{{"auth":{{"apiKey":"key-1","passphrase":"pass-1","secret":"sec-1"}},"markets":["{}","{}"],"type":"user"}}"#,
                cid("ab"),
                cid("cd")
            )
        );
    }

    #[test]
    fn subscribe_message_empty_markets() {
        let msg = subscribe_message("k", "s", "p", std::iter::empty());
        assert_eq!(
            msg,
            r#"{"auth":{"apiKey":"k","passphrase":"p","secret":"s"},"markets":[],"type":"user"}"#
        );
    }

    #[test]
    fn empty_and_pong_frames_classify() {
        assert_eq!(parse_user_frame("", NOW), ParsedUserFrame::Ack);
        assert_eq!(parse_user_frame("   ", NOW), ParsedUserFrame::Ack);
        assert_eq!(parse_user_frame("PONG", NOW), ParsedUserFrame::Pong);
        assert_eq!(parse_user_frame("pong", NOW), ParsedUserFrame::Pong);
        assert_eq!(parse_user_frame("[]", NOW), ParsedUserFrame::Ack);
    }

    #[test]
    fn garbage_frames_are_ignored() {
        assert_eq!(
            parse_user_frame("not json", NOW),
            ParsedUserFrame::Ignored(IgnoredReason::MalformedJson)
        );
        assert_eq!(
            parse_user_frame("42", NOW),
            ParsedUserFrame::Ignored(IgnoredReason::MalformedJson)
        );
    }

    #[test]
    fn order_placement_parses() {
        let frame = format!(
            r#"{{"event_type":"order","type":"PLACEMENT","id":"o-1","market":"{}",
                "asset_id":"123","outcome":"Up","side":"BUY","original_size":"10","price":"0.40",
                "size_matched":"0","timestamp":"1700000000123","associate_trades":null}}"#,
            cid("ab")
        );
        let Ok(WireUserEvent::Order(o)) = single(&frame) else {
            panic!("expected order");
        };
        assert_eq!(o.kind, OrderEventKind::Placement);
        assert_eq!(o.order_id.as_str(), "o-1");
        assert_eq!(o.condition_id.as_str(), cid("ab"));
        assert_eq!(o.token_id.as_str(), "123");
        assert_eq!(o.side, Side::Buy);
        assert_eq!(o.price, dec!(0.40));
        assert_eq!(o.original_size, dec!(10));
        assert_eq!(o.size_matched, dec!(0));
        assert_eq!(o.ts, TimestampMs::from_millis(1_700_000_000_123));
    }

    #[test]
    fn order_update_and_cancellation_kinds() {
        let frame = format!(
            r#"{{"event_type":"order","type":"UPDATE","id":"o-1","market":"{}",
                "asset_id":"123","side":"BUY","size_matched":"3"}}"#,
            cid("ab")
        );
        let Ok(WireUserEvent::Order(o)) = single(&frame) else {
            panic!("expected order");
        };
        assert_eq!(o.kind, OrderEventKind::Update);
        assert_eq!(o.size_matched, dec!(3));
        assert_eq!(o.ts, NOW, "absent timestamp falls back to now");

        let frame = format!(
            r#"{{"event_type":"order","type":"CANCELLATION","id":"o-1","market":"{}",
                "asset_id":"123","side":"BUY","size_matched":"3"}}"#,
            cid("ab")
        );
        let Ok(WireUserEvent::Order(o)) = single(&frame) else {
            panic!("expected order");
        };
        assert_eq!(o.kind, OrderEventKind::Cancellation);
    }

    #[test]
    fn trade_taker_parses() {
        let frame = format!(
            r#"{{"event_type":"trade","type":"TRADE","id":"t-1","taker_order_id":"o-9",
                "market":"{}","asset_id":"123","outcome":"Up","side":"BUY","size":"5","price":"0.41",
                "status":"MATCHED","maker_orders":[],"timestamp":"100","matchtime":"105"}}"#,
            cid("ab")
        );
        let Ok(WireUserEvent::Trade(t)) = single(&frame) else {
            panic!("expected trade");
        };
        assert_eq!(t.trade_id, "t-1");
        assert_eq!(t.taker_order_id.as_ref().map(OrderId::as_str), Some("o-9"));
        assert_eq!(t.side, Side::Buy);
        assert_eq!(t.size, dec!(5));
        assert_eq!(t.price, dec!(0.41));
        assert_eq!(t.status, TradeStatus::Matched);
        assert!(t.maker_orders.is_empty());
        assert_eq!(t.ts, TimestampMs::from_millis(105), "matchtime preferred");
    }

    #[test]
    fn trade_maker_legs_parse_leniently() {
        let frame = format!(
            r#"{{"event_type":"trade","id":"t-2","market":"{}","asset_id":"123","side":"SELL",
                "size":"5","price":"0.59","status":"MATCHED","maker_orders":[
                  {{"order_id":"m-1","asset_id":"123","matched_amount":"4","price":"0.59","outcome":"Up"}},
                  {{"order_id":"","asset_id":"123","matched_amount":"1","price":"0.59"}}
                ]}}"#,
            cid("ab")
        );
        let Ok(WireUserEvent::Trade(t)) = single(&frame) else {
            panic!("expected trade");
        };
        assert_eq!(t.taker_order_id, None);
        assert_eq!(t.maker_orders.len(), 1, "the empty-id maker leg is dropped");
        assert_eq!(t.maker_orders[0].order_id.as_str(), "m-1");
        assert_eq!(t.maker_orders[0].matched_amount, dec!(4));
        assert_eq!(t.maker_orders[0].price, dec!(0.59));
    }

    #[test]
    fn array_frame_parses_each_event_independently() {
        let frame = format!(
            r#"[{{"event_type":"order","type":"PLACEMENT","id":"o-1","market":"{c}","asset_id":"1","side":"BUY"}},
                {{"event_type":"mystery"}},
                {{"event_type":"trade","id":"t-1","market":"{c}","asset_id":"1","side":"BUY","status":"MATCHED"}}]"#,
            c = cid("cd")
        );
        let ParsedUserFrame::Events(events) = parse_user_frame(&frame, NOW) else {
            panic!("expected events");
        };
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Ok(WireUserEvent::Order(_))));
        assert_eq!(events[1], Err(IgnoredReason::UnknownEventType));
        assert!(matches!(events[2], Ok(WireUserEvent::Trade(_))));
    }

    #[test]
    fn bad_and_missing_values_are_rejected() {
        // Missing event_type.
        assert_eq!(single(r#"{"id":"o-1"}"#), Err(IgnoredReason::MissingField));
        // Order with a bad token id.
        let frame = format!(
            r#"{{"event_type":"order","type":"UPDATE","id":"o-1","market":"{}","asset_id":"0xnope","side":"BUY"}}"#,
            cid("ab")
        );
        assert_eq!(single(&frame), Err(IgnoredReason::BadValue));
        // Order with a missing required field (no side).
        let frame = format!(
            r#"{{"event_type":"order","type":"UPDATE","id":"o-1","market":"{}","asset_id":"1"}}"#,
            cid("ab")
        );
        assert_eq!(single(&frame), Err(IgnoredReason::MissingField));
        // Trade with an unknown status.
        let frame = format!(
            r#"{{"event_type":"trade","id":"t-1","market":"{}","asset_id":"1","side":"BUY","status":"WAT"}}"#,
            cid("ab")
        );
        assert_eq!(single(&frame), Err(IgnoredReason::BadValue));
        // Trade with an empty id.
        let frame = format!(
            r#"{{"event_type":"trade","id":"","market":"{}","asset_id":"1","side":"BUY","status":"MATCHED"}}"#,
            cid("ab")
        );
        assert_eq!(single(&frame), Err(IgnoredReason::BadValue));
    }

    fn first(json: &str) -> WireUserEvent {
        match parse_user_frame(json, NOW) {
            ParsedUserFrame::Events(mut e) => e.remove(0).expect("the fixture parses"),
            other => panic!("expected events, got {other:?}"),
        }
    }

    #[test]
    fn committed_fixtures_parse_to_expected_shapes() {
        let WireUserEvent::Order(o) =
            first(include_str!("../tests/fixtures/user_order_placement.json"))
        else {
            panic!("placement is an order");
        };
        assert_eq!(o.kind, OrderEventKind::Placement);
        assert_eq!(o.order_id.as_str(), "0xorder-up-1");
        assert_eq!(o.size_matched, dec!(0));

        let WireUserEvent::Order(o) = first(include_str!(
            "../tests/fixtures/user_order_update_partial.json"
        )) else {
            panic!("update is an order");
        };
        assert_eq!(o.kind, OrderEventKind::Update);
        assert_eq!(o.size_matched, dec!(8));

        let WireUserEvent::Order(o) = first(include_str!(
            "../tests/fixtures/user_order_cancellation.json"
        )) else {
            panic!("cancellation is an order");
        };
        assert_eq!(o.kind, OrderEventKind::Cancellation);

        let WireUserEvent::Trade(t) = first(include_str!(
            "../tests/fixtures/user_trade_matched_taker.json"
        )) else {
            panic!("taker fixture is a trade");
        };
        assert_eq!(t.status, TradeStatus::Matched);
        assert_eq!(
            t.taker_order_id.as_ref().map(OrderId::as_str),
            Some("0xorder-up-1")
        );
        assert_eq!(t.size, dec!(5));

        let WireUserEvent::Trade(t) = first(include_str!(
            "../tests/fixtures/user_trade_matched_maker.json"
        )) else {
            panic!("maker fixture is a trade");
        };
        assert_eq!(t.maker_orders.len(), 1);
        assert_eq!(t.maker_orders[0].order_id.as_str(), "0xorder-down-1");
        assert_eq!(t.maker_orders[0].matched_amount, dec!(7));

        let WireUserEvent::Trade(t) =
            first(include_str!("../tests/fixtures/user_trade_failed.json"))
        else {
            panic!("failed fixture is a trade");
        };
        assert_eq!(t.status, TradeStatus::Failed);

        let ParsedUserFrame::Events(events) =
            parse_user_frame(include_str!("../tests/fixtures/user_trade_array.json"), NOW)
        else {
            panic!("array fixture is a multi-event frame");
        };
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Ok(WireUserEvent::Order(_))));
        assert!(matches!(events[1], Ok(WireUserEvent::Trade(_))));
    }

    #[test]
    fn numeric_wire_fields_also_parse() {
        let frame = format!(
            r#"{{"event_type":"order","type":"UPDATE","id":"o-1","market":"{}","asset_id":"1",
                "side":"BUY","size_matched":4,"price":0.4,"timestamp":1700000000123}}"#,
            cid("ab")
        );
        let Ok(WireUserEvent::Order(o)) = single(&frame) else {
            panic!("expected order");
        };
        assert_eq!(o.size_matched, dec!(4));
        assert_eq!(o.price, dec!(0.4));
        assert_eq!(o.ts, TimestampMs::from_millis(1_700_000_000_123));
    }
}
