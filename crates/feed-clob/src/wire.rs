//! Pure market-channel wire protocol: the subscribe-message builder and the
//! frame parser. No IO, no clocks (the caller passes `now` for timestamp
//! fallback), no logging — fixture tests drive this module directly.
//!
//! Wire facts (docs fetched 2026-06-12; re-verified against captured frames):
//! every event object carries an `event_type` tag; prices/sizes/timestamps
//! arrive as strings (parsed defensively as string-or-number); a text frame
//! may carry a single event object or a JSON array of them; the server
//! answers the client's text `PING` with `PONG`.

use core_types::{ConditionId, Decimal, Side, TickSize, TimestampMs, TokenId};
use serde::Deserialize;

/// The client keepalive text; the market-channel docs require one at least
/// every 10 seconds.
pub const PING_TEXT: &str = "PING";

/// Builds the one-and-only subscribe message: sent once per connection,
/// immediately after connect (`custom_feature_enabled` unlocks
/// `best_bid_ask`, `new_market`, and `market_resolved` — CLAUDE.md §7).
#[must_use]
pub fn subscribe_message<'a>(assets: impl IntoIterator<Item = &'a TokenId>) -> String {
    let ids: Vec<&str> = assets.into_iter().map(TokenId::as_str).collect();
    serde_json::json!({
        "assets_ids": ids,
        "type": "market",
        "custom_feature_enabled": true,
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
    /// A field was present but unparseable (bad id, bad decimal, bad side).
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

/// Full L2 snapshot for one token (`book` event). Fires on subscribe and
/// whenever a trade affects the book — the venue continuously hands us
/// ground truth, which is why snapshot-replace is the primary integrity
/// mechanism (machine module).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookMsg {
    /// Token the book belongs to.
    pub token: TokenId,
    /// Market (condition id) the token belongs to.
    pub condition: ConditionId,
    /// Bid levels as (price, size), wire order (not trusted to be sorted).
    pub bids: Vec<(Decimal, Decimal)>,
    /// Ask levels as (price, size), wire order.
    pub asks: Vec<(Decimal, Decimal)>,
    /// Wire timestamp (caller's `now` when absent/unparseable).
    pub ts: TimestampMs,
    /// Opaque venue book hash (algorithm undocumented — stored for the
    /// journal, never validated).
    pub hash: Option<String>,
}

/// One level update inside a `price_change` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelChange {
    /// Token whose book changed.
    pub token: TokenId,
    /// Price level.
    pub price: Decimal,
    /// New TOTAL displayed size at the level (not a delta); zero means the
    /// level was removed.
    pub size: Decimal,
    /// Book side the level lives on (BUY = bid, SELL = ask).
    pub side: Side,
    /// Venue-reported best bid after this change (integrity input only;
    /// `None` when absent or out of the open (0, 1) interval).
    pub best_bid: Option<Decimal>,
    /// Venue-reported best ask after this change.
    pub best_ask: Option<Decimal>,
}

/// Order place/cancel deltas (`price_change` event).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriceChangeMsg {
    /// Wire timestamp (caller's `now` when absent/unparseable).
    pub ts: TimestampMs,
    /// The per-level changes, wire order.
    pub changes: Vec<LevelChange>,
}

/// One typed, validated market-channel event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClobEvent {
    /// Full L2 snapshot for one token.
    Book(BookMsg),
    /// Order place/cancel level deltas.
    PriceChange(PriceChangeMsg),
    /// Tick regime flip (0.01 ↔ 0.001 past 0.96/0.04).
    TickSizeChange {
        /// Token whose price crossed the flip threshold.
        token: TokenId,
        /// Market the flip applies to.
        condition: ConditionId,
        /// Previous tick, when parseable (informational).
        old_tick: Option<TickSize>,
        /// The new tick in force.
        new_tick: TickSize,
        /// Wire timestamp.
        ts: TimestampMs,
    },
    /// A trade printed (`last_trade_price`).
    LastTrade {
        /// Token that traded.
        token: TokenId,
        /// Print price.
        price: Decimal,
        /// Print size in shares.
        size: Decimal,
        /// Aggressor side as reported.
        side: Side,
        /// Wire timestamp.
        ts: TimestampMs,
    },
    /// Venue-reported best bid/ask (`best_bid_ask`, custom feature flag).
    /// Carries no sizes, so it can never populate the bus top-of-book — it
    /// is an integrity cross-check input only.
    BestBidAsk {
        /// Token the tops belong to.
        token: TokenId,
        /// Best bid (`None` when absent/empty-side).
        best_bid: Option<Decimal>,
        /// Best ask.
        best_ask: Option<Decimal>,
        /// Wire timestamp.
        ts: TimestampMs,
    },
    /// A market was created (`new_market`, custom feature flag) — scheduler
    /// refresh hint only; Gamma stays the metadata authority.
    NewMarket {
        /// Condition id of the new market.
        condition: ConditionId,
        /// Wire slug, for logging only.
        slug: String,
        /// Wire timestamp.
        ts: TimestampMs,
    },
    /// A market resolved (`market_resolved`, custom feature flag).
    MarketResolved {
        /// Condition id of the resolved market.
        condition: ConditionId,
        /// Token id of the winning outcome.
        winning_token: TokenId,
        /// Wire timestamp.
        ts: TimestampMs,
    },
}

/// One classified inbound text frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedFrame {
    /// Empty text frame — ack-style chatter, nothing to do.
    Ack,
    /// The server's `PONG` keepalive reply (liveness evidence).
    Pong,
    /// Parsed events — an array frame carries several; a per-event failure
    /// surfaces as `Err` without poisoning its siblings.
    Events(Vec<Result<ClobEvent, IgnoredReason>>),
    /// The whole frame was unusable.
    Ignored(IgnoredReason),
}

/// Classifies one inbound text frame. `now` substitutes for absent or
/// unparseable wire timestamps so downstream events are always stamped.
#[must_use]
pub fn parse_frame(text: &str, now: TimestampMs) -> ParsedFrame {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return ParsedFrame::Ack;
    }
    if trimmed.eq_ignore_ascii_case("PONG") {
        return ParsedFrame::Pong;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return ParsedFrame::Ignored(IgnoredReason::MalformedJson);
    };
    let objects = match value {
        serde_json::Value::Array(items) => items,
        object @ serde_json::Value::Object(_) => vec![object],
        _ => return ParsedFrame::Ignored(IgnoredReason::MalformedJson),
    };
    if objects.is_empty() {
        return ParsedFrame::Ack;
    }
    ParsedFrame::Events(
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
                use rust_decimal::prelude::FromPrimitive;
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

/// Lenient venue-top parse: `"0"`, `""`, absent, or anything outside the
/// open (0, 1) interval reads as "no level on that side" (the empty-side
/// encoding is undocumented — never let it warn-flood or fabricate a top).
fn parse_top(value: Option<&StrNum>) -> Option<Decimal> {
    let d = value?.to_decimal()?;
    if d <= Decimal::ZERO || d >= Decimal::ONE {
        None
    } else {
        Some(d)
    }
}

// --- permissive wire structs (no deny_unknown_fields: the venue adds fields
// --- freely; discovery-crate precedent) ---

#[derive(Deserialize)]
struct WireLevel {
    price: StrNum,
    size: StrNum,
}

#[derive(Deserialize)]
struct WireBook {
    asset_id: String,
    market: String,
    #[serde(default, alias = "buys")]
    bids: Vec<WireLevel>,
    #[serde(default, alias = "sells")]
    asks: Vec<WireLevel>,
    timestamp: Option<StrNum>,
    hash: Option<String>,
}

#[derive(Deserialize)]
struct WireChange {
    asset_id: String,
    price: StrNum,
    size: StrNum,
    side: String,
    best_bid: Option<StrNum>,
    best_ask: Option<StrNum>,
}

#[derive(Deserialize)]
struct WirePriceChange {
    #[serde(default, alias = "changes")]
    price_changes: Vec<WireChange>,
    timestamp: Option<StrNum>,
}

#[derive(Deserialize)]
struct WireTickSizeChange {
    asset_id: String,
    market: String,
    old_tick_size: Option<StrNum>,
    new_tick_size: StrNum,
    timestamp: Option<StrNum>,
}

#[derive(Deserialize)]
struct WireLastTrade {
    asset_id: String,
    price: StrNum,
    size: StrNum,
    side: String,
    timestamp: Option<StrNum>,
}

#[derive(Deserialize)]
struct WireBestBidAsk {
    asset_id: String,
    best_bid: Option<StrNum>,
    best_ask: Option<StrNum>,
    timestamp: Option<StrNum>,
}

#[derive(Deserialize)]
struct WireNewMarket {
    // The docs list both `market` and `condition_id`; prefer the explicit
    // one and fall back (which one is the 0x…64-hex condition id will be
    // pinned by capture — ConditionId validation rejects the wrong one).
    condition_id: Option<String>,
    market: Option<String>,
    #[serde(default)]
    slug: String,
    timestamp: Option<StrNum>,
}

#[derive(Deserialize)]
struct WireMarketResolved {
    condition_id: Option<String>,
    market: Option<String>,
    winning_asset_id: String,
    timestamp: Option<StrNum>,
}

fn parse_levels(levels: Vec<WireLevel>) -> Result<Vec<(Decimal, Decimal)>, IgnoredReason> {
    levels
        .into_iter()
        .map(|level| {
            let price = level.price.to_decimal().ok_or(IgnoredReason::BadValue)?;
            let size = level.size.to_decimal().ok_or(IgnoredReason::BadValue)?;
            Ok((price, size))
        })
        .collect()
}

fn parse_event(object: serde_json::Value, now: TimestampMs) -> Result<ClobEvent, IgnoredReason> {
    let Some(event_type) = object.get("event_type").and_then(|v| v.as_str()) else {
        return Err(IgnoredReason::MissingField);
    };
    match event_type {
        "book" => {
            let wire: WireBook =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            Ok(ClobEvent::Book(BookMsg {
                token: TokenId::new(wire.asset_id).map_err(|_| IgnoredReason::BadValue)?,
                condition: ConditionId::new(wire.market).map_err(|_| IgnoredReason::BadValue)?,
                bids: parse_levels(wire.bids)?,
                asks: parse_levels(wire.asks)?,
                ts: ts_or(now, wire.timestamp.as_ref()),
                hash: wire.hash,
            }))
        }
        "price_change" => {
            let wire: WirePriceChange =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            let ts = ts_or(now, wire.timestamp.as_ref());
            let changes = wire
                .price_changes
                .into_iter()
                .map(|change| {
                    Ok(LevelChange {
                        token: TokenId::new(change.asset_id)
                            .map_err(|_| IgnoredReason::BadValue)?,
                        price: change.price.to_decimal().ok_or(IgnoredReason::BadValue)?,
                        size: change.size.to_decimal().ok_or(IgnoredReason::BadValue)?,
                        side: parse_side(&change.side).ok_or(IgnoredReason::BadValue)?,
                        best_bid: parse_top(change.best_bid.as_ref()),
                        best_ask: parse_top(change.best_ask.as_ref()),
                    })
                })
                .collect::<Result<Vec<_>, IgnoredReason>>()?;
            Ok(ClobEvent::PriceChange(PriceChangeMsg { ts, changes }))
        }
        "tick_size_change" => {
            let wire: WireTickSizeChange =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            let new_tick = wire
                .new_tick_size
                .to_decimal()
                .and_then(|d| TickSize::from_decimal(d).ok())
                .ok_or(IgnoredReason::BadValue)?;
            let old_tick = wire
                .old_tick_size
                .as_ref()
                .and_then(StrNum::to_decimal)
                .and_then(|d| TickSize::from_decimal(d).ok());
            Ok(ClobEvent::TickSizeChange {
                token: TokenId::new(wire.asset_id).map_err(|_| IgnoredReason::BadValue)?,
                condition: ConditionId::new(wire.market).map_err(|_| IgnoredReason::BadValue)?,
                old_tick,
                new_tick,
                ts: ts_or(now, wire.timestamp.as_ref()),
            })
        }
        "last_trade_price" => {
            let wire: WireLastTrade =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            Ok(ClobEvent::LastTrade {
                token: TokenId::new(wire.asset_id).map_err(|_| IgnoredReason::BadValue)?,
                price: wire.price.to_decimal().ok_or(IgnoredReason::BadValue)?,
                size: wire.size.to_decimal().ok_or(IgnoredReason::BadValue)?,
                side: parse_side(&wire.side).ok_or(IgnoredReason::BadValue)?,
                ts: ts_or(now, wire.timestamp.as_ref()),
            })
        }
        "best_bid_ask" => {
            let wire: WireBestBidAsk =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            Ok(ClobEvent::BestBidAsk {
                token: TokenId::new(wire.asset_id).map_err(|_| IgnoredReason::BadValue)?,
                best_bid: parse_top(wire.best_bid.as_ref()),
                best_ask: parse_top(wire.best_ask.as_ref()),
                ts: ts_or(now, wire.timestamp.as_ref()),
            })
        }
        "new_market" => {
            let wire: WireNewMarket =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            let raw = wire
                .condition_id
                .or(wire.market)
                .ok_or(IgnoredReason::MissingField)?;
            Ok(ClobEvent::NewMarket {
                condition: ConditionId::new(raw).map_err(|_| IgnoredReason::BadValue)?,
                slug: wire.slug,
                ts: ts_or(now, wire.timestamp.as_ref()),
            })
        }
        "market_resolved" => {
            let wire: WireMarketResolved =
                serde_json::from_value(object).map_err(|_| IgnoredReason::MissingField)?;
            let raw = wire
                .condition_id
                .or(wire.market)
                .ok_or(IgnoredReason::MissingField)?;
            Ok(ClobEvent::MarketResolved {
                condition: ConditionId::new(raw).map_err(|_| IgnoredReason::BadValue)?,
                winning_token: TokenId::new(wire.winning_asset_id)
                    .map_err(|_| IgnoredReason::BadValue)?,
                ts: ts_or(now, wire.timestamp.as_ref()),
            })
        }
        _ => Err(IgnoredReason::UnknownEventType),
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;

    const NOW: TimestampMs = TimestampMs::from_millis(1_700_000_000_000);

    fn cid(byte: &str) -> String {
        format!("0x{}", byte.repeat(32))
    }

    fn single(frame: &str) -> Result<ClobEvent, IgnoredReason> {
        match parse_frame(frame, NOW) {
            ParsedFrame::Events(mut events) => {
                assert_eq!(events.len(), 1, "expected exactly one event");
                events.remove(0)
            }
            other => panic!("expected events, got {other:?}"),
        }
    }

    #[test]
    fn subscribe_message_is_pinned() {
        let up = TokenId::new("11111").unwrap();
        let down = TokenId::new("22222").unwrap();
        assert_eq!(
            subscribe_message([&up, &down]),
            r#"{"assets_ids":["11111","22222"],"custom_feature_enabled":true,"type":"market"}"#
        );
    }

    #[test]
    fn empty_and_pong_frames_classify() {
        assert_eq!(parse_frame("", NOW), ParsedFrame::Ack);
        assert_eq!(parse_frame("   ", NOW), ParsedFrame::Ack);
        assert_eq!(parse_frame("PONG", NOW), ParsedFrame::Pong);
        assert_eq!(parse_frame("pong", NOW), ParsedFrame::Pong);
        assert_eq!(parse_frame("[]", NOW), ParsedFrame::Ack);
    }

    #[test]
    fn garbage_frames_are_ignored() {
        assert_eq!(
            parse_frame("not json", NOW),
            ParsedFrame::Ignored(IgnoredReason::MalformedJson)
        );
        assert_eq!(
            parse_frame("42", NOW),
            ParsedFrame::Ignored(IgnoredReason::MalformedJson)
        );
        assert_eq!(
            parse_frame(r#""a string""#, NOW),
            ParsedFrame::Ignored(IgnoredReason::MalformedJson)
        );
    }

    #[test]
    fn book_event_parses() {
        let frame = format!(
            r#"{{"event_type":"book","asset_id":"123","market":"{}",
                "bids":[{{"price":"0.48","size":"100"}},{{"price":"0.47","size":"50"}}],
                "asks":[{{"price":"0.52","size":"200"}}],
                "timestamp":"1700000000123","hash":"abc123"}}"#,
            cid("ab")
        );
        let Ok(ClobEvent::Book(book)) = single(&frame) else {
            panic!("expected book");
        };
        assert_eq!(book.token.as_str(), "123");
        assert_eq!(book.condition.as_str(), cid("ab"));
        assert_eq!(
            book.bids,
            vec![(dec!(0.48), dec!(100)), (dec!(0.47), dec!(50))]
        );
        assert_eq!(book.asks, vec![(dec!(0.52), dec!(200))]);
        assert_eq!(book.ts, TimestampMs::from_millis(1_700_000_000_123));
        assert_eq!(book.hash.as_deref(), Some("abc123"));
    }

    #[test]
    fn book_event_with_missing_timestamp_falls_back_to_now() {
        let frame = format!(
            r#"{{"event_type":"book","asset_id":"123","market":"{}","bids":[],"asks":[]}}"#,
            cid("ab")
        );
        let Ok(ClobEvent::Book(book)) = single(&frame) else {
            panic!("expected book");
        };
        assert_eq!(book.ts, NOW);
        assert!(book.bids.is_empty());
        assert!(book.hash.is_none());
    }

    #[test]
    fn array_frame_parses_each_event_independently() {
        let frame = format!(
            r#"[{{"event_type":"book","asset_id":"1","market":"{c}","bids":[],"asks":[]}},
                {{"event_type":"mystery"}},
                {{"event_type":"book","asset_id":"2","market":"{c}","bids":[],"asks":[]}}]"#,
            c = cid("cd")
        );
        let ParsedFrame::Events(events) = parse_frame(&frame, NOW) else {
            panic!("expected events");
        };
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Ok(ClobEvent::Book(_))));
        assert_eq!(events[1], Err(IgnoredReason::UnknownEventType));
        assert!(matches!(events[2], Ok(ClobEvent::Book(_))));
    }

    #[test]
    fn price_change_event_parses_with_lenient_tops() {
        let frame = r#"{"event_type":"price_change","market":"0xignored","timestamp":"5",
            "price_changes":[
              {"asset_id":"7","price":"0.51","size":"0","side":"SELL","best_bid":"0.50","best_ask":"0.52"},
              {"asset_id":"7","price":"0.50","size":"25.5","side":"buy","best_bid":"0","best_ask":""}
            ]}"#;
        let Ok(ClobEvent::PriceChange(msg)) = single(frame) else {
            panic!("expected price_change");
        };
        assert_eq!(msg.ts, TimestampMs::from_millis(5));
        assert_eq!(msg.changes.len(), 2);
        let removal = &msg.changes[0];
        assert_eq!(removal.side, Side::Sell);
        assert!(removal.size.is_zero());
        assert_eq!(removal.best_bid, Some(dec!(0.50)));
        assert_eq!(removal.best_ask, Some(dec!(0.52)));
        let update = &msg.changes[1];
        assert_eq!(update.side, Side::Buy);
        assert_eq!(update.size, dec!(25.5));
        assert_eq!(update.best_bid, None, "\"0\" reads as empty side");
        assert_eq!(update.best_ask, None, "\"\" reads as empty side");
    }

    #[test]
    fn tick_size_change_event_parses() {
        let frame = format!(
            r#"{{"event_type":"tick_size_change","asset_id":"9","market":"{}",
                "old_tick_size":"0.01","new_tick_size":"0.001","timestamp":"77"}}"#,
            cid("ef")
        );
        let Ok(ClobEvent::TickSizeChange {
            token,
            old_tick,
            new_tick,
            ts,
            ..
        }) = single(&frame)
        else {
            panic!("expected tick_size_change");
        };
        assert_eq!(token.as_str(), "9");
        assert_eq!(old_tick, Some(TickSize::T001));
        assert_eq!(new_tick, TickSize::T0001);
        assert_eq!(ts, TimestampMs::from_millis(77));
    }

    #[test]
    fn last_trade_event_parses() {
        let frame = r#"{"event_type":"last_trade_price","asset_id":"5","market":"0xwhatever",
            "price":"0.55","side":"BUY","size":"30","fee_rate_bps":"0","timestamp":"99"}"#;
        let Ok(ClobEvent::LastTrade {
            token,
            price,
            size,
            side,
            ts,
        }) = single(frame)
        else {
            panic!("expected last_trade_price");
        };
        assert_eq!(token.as_str(), "5");
        assert_eq!(price, dec!(0.55));
        assert_eq!(size, dec!(30));
        assert_eq!(side, Side::Buy);
        assert_eq!(ts, TimestampMs::from_millis(99));
    }

    #[test]
    fn best_bid_ask_event_parses() {
        let frame = r#"{"event_type":"best_bid_ask","market":"0x00","asset_id":"4",
            "best_bid":"0.61","best_ask":"0.63","spread":"0.02","timestamp":"3"}"#;
        let Ok(ClobEvent::BestBidAsk {
            token,
            best_bid,
            best_ask,
            ts,
        }) = single(frame)
        else {
            panic!("expected best_bid_ask");
        };
        assert_eq!(token.as_str(), "4");
        assert_eq!(best_bid, Some(dec!(0.61)));
        assert_eq!(best_ask, Some(dec!(0.63)));
        assert_eq!(ts, TimestampMs::from_millis(3));
    }

    #[test]
    fn new_market_event_parses_preferring_condition_id() {
        let frame = format!(
            r#"{{"event_type":"new_market","market":"0xshortaddr","condition_id":"{}",
                "slug":"btc-updown-5m-1781166900","timestamp":"8"}}"#,
            cid("aa")
        );
        let Ok(ClobEvent::NewMarket {
            condition,
            slug,
            ts,
        }) = single(&frame)
        else {
            panic!("expected new_market");
        };
        assert_eq!(condition.as_str(), cid("aa"));
        assert_eq!(slug, "btc-updown-5m-1781166900");
        assert_eq!(ts, TimestampMs::from_millis(8));
    }

    #[test]
    fn market_resolved_event_parses_via_market_field() {
        let frame = format!(
            r#"{{"event_type":"market_resolved","market":"{}","winning_asset_id":"42",
                "winning_outcome":"Up","timestamp":"11"}}"#,
            cid("bb")
        );
        let Ok(ClobEvent::MarketResolved {
            condition,
            winning_token,
            ts,
        }) = single(&frame)
        else {
            panic!("expected market_resolved");
        };
        assert_eq!(condition.as_str(), cid("bb"));
        assert_eq!(winning_token.as_str(), "42");
        assert_eq!(ts, TimestampMs::from_millis(11));
    }

    #[test]
    fn bad_values_are_rejected_per_event() {
        // Bad condition id on a book.
        let frame =
            r#"{"event_type":"book","asset_id":"1","market":"0xnothex","bids":[],"asks":[]}"#;
        assert_eq!(single(frame), Err(IgnoredReason::BadValue));
        // Bad side on a price change.
        let frame = r#"{"event_type":"price_change","price_changes":[
            {"asset_id":"1","price":"0.5","size":"1","side":"HOLD"}]}"#;
        assert_eq!(single(frame), Err(IgnoredReason::BadValue));
        // Bad decimal on a trade.
        let frame = r#"{"event_type":"last_trade_price","asset_id":"1","price":"abc","size":"1","side":"BUY"}"#;
        assert_eq!(single(frame), Err(IgnoredReason::BadValue));
        // Unknown tick size.
        let frame = format!(
            r#"{{"event_type":"tick_size_change","asset_id":"1","market":"{}","new_tick_size":"0.05"}}"#,
            cid("cc")
        );
        assert_eq!(single(&frame), Err(IgnoredReason::BadValue));
        // Missing event_type entirely.
        assert_eq!(
            single(r#"{"price":"0.5"}"#),
            Err(IgnoredReason::MissingField)
        );
        // market_resolved with a non-decimal winning token.
        let frame = format!(
            r#"{{"event_type":"market_resolved","market":"{}","winning_asset_id":"0xdead"}}"#,
            cid("dd")
        );
        assert_eq!(single(&frame), Err(IgnoredReason::BadValue));
    }

    #[test]
    fn numeric_wire_fields_also_parse() {
        // Defensive: a format drift from strings to bare numbers must not
        // break parsing.
        let frame = format!(
            r#"{{"event_type":"book","asset_id":"3","market":"{}",
                "bids":[{{"price":0.48,"size":100}}],"asks":[],"timestamp":1700000000123}}"#,
            cid("ee")
        );
        let Ok(ClobEvent::Book(book)) = single(&frame) else {
            panic!("expected book");
        };
        assert_eq!(book.bids.len(), 1);
        assert_eq!(book.bids[0].1, dec!(100));
        assert_eq!(book.ts, TimestampMs::from_millis(1_700_000_000_123));
    }
}
