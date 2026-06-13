//! Gamma REST wire types (`https://gamma-api.polymarket.com`).
//!
//! Field facts verified live 2026-06-11 (see the fixtures README and the
//! Decisions Log):
//! - `outcomes` / `clobTokenIds` are JSON-encoded strings *inside* the JSON
//!   (double-encoded parallel arrays) — decode via [`parse_double_encoded`].
//! - `market.eventStartTime` is the authoritative window open;
//!   `event.startTime` is empty in list views for hourly events, and
//!   `market.startDate` is the market *creation* time — never the open. It
//!   is deliberately not mapped here.
//! - `feeSchedule` is the authoritative fee descriptor. The flat
//!   `makerBaseFee` / `takerBaseFee` / `makerRebatesFeeShareBps` fields
//!   contradict it on live markets and are deliberately not mapped.
//! - `outcomePrices` belongs to the book feeds, not discovery — not mapped.

use rust_decimal::Decimal;
use serde::Deserialize;

use crate::error::MapError;

/// One element of `GET /series?slug=…`. Gamma serializes `id` as a JSON
/// string (verified in the committed fixture).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaSeries {
    /// Gamma's numeric series id, as a string. This is the `series_id`
    /// value for the events query.
    pub id: String,
    /// The series slug, e.g. `btc-up-or-down-5m`.
    pub slug: String,
    /// Window cadence: `"5m"`, `"15m"`, or `"hourly"` for our series.
    #[serde(default)]
    pub recurrence: Option<String>,
}

/// One element of `GET /events?series_id=…`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaEvent {
    /// Event slug, e.g. `btc-updown-5m-1781166600` (5m/15m embed the unix
    /// open-seconds; hourly slugs are human dates).
    pub slug: String,
    /// Display title.
    #[serde(default)]
    pub title: Option<String>,
    /// Window open time (RFC3339). Empty/absent in list views for hourly
    /// events — never required; the market's `eventStartTime` is preferred.
    #[serde(default)]
    pub start_time: Option<String>,
    /// Window close time (RFC3339).
    #[serde(default)]
    pub end_date: Option<String>,
    /// Slug of the series this event belongs to (sanity-checked against the
    /// configured slug).
    #[serde(default)]
    pub series_slug: Option<String>,
    /// Event-level neg-risk flag (the market-level flag is authoritative).
    #[serde(default)]
    pub neg_risk: Option<bool>,
    /// Event-level resolution source (fallback when the market omits its
    /// own).
    #[serde(default)]
    pub resolution_source: Option<String>,
    /// The markets under this event — exactly one for these series.
    #[serde(default)]
    pub markets: Vec<GammaMarket>,
    /// Resolution metadata. Present only **after** the event resolves
    /// (verified live 2026-06-13: absent ~1 min post-close, present ~4 min
    /// later) — the post-hoc anchor for verifying our self-captured strike.
    #[serde(default)]
    pub event_metadata: Option<GammaEventMetadata>,
}

/// The `eventMetadata` object the venue attaches once a window resolves. The
/// venue samples its own resolution feed at the window boundary, so
/// `priceToBeat` is the authoritative strike and `finalPrice` the close
/// (`finalPrice ≥ priceToBeat ⇒ Up`). Both are full-precision Chainlink
/// values; they decode through `f64` (no `arbitrary_precision`, per the
/// Decisions Log), so consumers compare with a small relative tolerance.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaEventMetadata {
    /// The window's strike — resolution-source price at open.
    #[serde(default)]
    pub price_to_beat: Option<Decimal>,
    /// The window's resolution-source close price.
    #[serde(default)]
    pub final_price: Option<Decimal>,
}

/// The market object embedded in a Gamma event.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaMarket {
    /// CTF condition id (`0x` + 64 hex).
    #[serde(default)]
    pub condition_id: Option<String>,
    /// Double-encoded outcome labels, e.g. `"[\"Up\", \"Down\"]"`.
    #[serde(default)]
    pub outcomes: Option<String>,
    /// Double-encoded token ids, parallel to `outcomes`.
    #[serde(default)]
    pub clob_token_ids: Option<String>,
    /// Current tick size as a JSON number (0.01 normally; 0.001 once the
    /// price drifts past 0.96/0.04).
    #[serde(default)]
    pub order_price_min_tick_size: Option<Decimal>,
    /// Venue minimum resting order size in shares.
    #[serde(default)]
    pub order_min_size: Option<Decimal>,
    /// Market-level neg-risk flag (authoritative; expected `false`).
    #[serde(default)]
    pub neg_risk: Option<bool>,
    /// Whether the protocol charges fees on this market.
    #[serde(default)]
    pub fees_enabled: Option<bool>,
    /// The authoritative fee descriptor.
    #[serde(default)]
    pub fee_schedule: Option<GammaFeeSchedule>,
    /// Whether the CLOB currently accepts orders for this market.
    #[serde(default)]
    pub accepting_orders: Option<bool>,
    /// Window open time (RFC3339) — the authoritative source.
    #[serde(default)]
    pub event_start_time: Option<String>,
    /// Window close time (RFC3339).
    #[serde(default)]
    pub end_date: Option<String>,
    /// Resolution-source URL/text from the market rules.
    #[serde(default)]
    pub resolution_source: Option<String>,
    /// Full rules text.
    #[serde(default)]
    pub description: Option<String>,
}

/// The `feeSchedule` object — the authoritative fee descriptor
/// (`{rate, exponent, takerOnly, rebateRate}`, observed
/// `{0.07, 1, true, 0.2}` on live crypto Up/Down markets).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaFeeSchedule {
    /// Taker fee rate, e.g. `0.07`.
    #[serde(default)]
    pub rate: Option<Decimal>,
    /// Fee curve exponent, observed `1`.
    #[serde(default)]
    pub exponent: Option<u32>,
    /// Whether only takers pay.
    #[serde(default)]
    pub taker_only: Option<bool>,
    /// Maker rebate share of collected taker fees, e.g. `0.2`.
    #[serde(default)]
    pub rebate_rate: Option<Decimal>,
}

/// Decodes Gamma's JSON-encoded-string-inside-JSON arrays
/// (`"[\"Up\", \"Down\"]"` → `["Up", "Down"]`).
///
/// # Errors
/// [`MapError::DoubleEncoded`] when the inner payload is not a JSON array
/// of strings.
pub fn parse_double_encoded(field: &'static str, raw: &str) -> Result<Vec<String>, MapError> {
    serde_json::from_str::<Vec<String>>(raw).map_err(|e| MapError::DoubleEncoded {
        field,
        msg: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;
    use rust_decimal::prelude::ToPrimitive;

    use super::*;

    #[test]
    fn parse_double_encoded_happy_path() {
        let v = parse_double_encoded("outcomes", r#"["Up", "Down"]"#).unwrap();
        assert_eq!(v, vec!["Up".to_owned(), "Down".to_owned()]);
    }

    #[test]
    fn parse_double_encoded_rejects_mangled_json() {
        let err = parse_double_encoded("outcomes", r#"["Up", "Down""#).unwrap_err();
        assert!(matches!(
            err,
            MapError::DoubleEncoded {
                field: "outcomes",
                ..
            }
        ));
    }

    #[test]
    fn parse_double_encoded_rejects_non_string_elements() {
        assert!(parse_double_encoded("clobTokenIds", "[1, 2]").is_err());
        assert!(parse_double_encoded("clobTokenIds", r#"{"a": 1}"#).is_err());
    }

    #[test]
    fn json_number_decimals_decode_exactly() {
        // Pins the shortest-round-trip decision (same precedent as the
        // config crate's TOML floats): JSON numbers must land as the exact
        // decimals the venue means, not float noise.
        let m: GammaMarket = serde_json::from_str(
            r#"{"orderPriceMinTickSize": 0.01, "orderMinSize": 5,
                "feeSchedule": {"rate": 0.07, "exponent": 1,
                                "takerOnly": true, "rebateRate": 0.2}}"#,
        )
        .unwrap();
        assert_eq!(m.order_price_min_tick_size, Some(dec!(0.01)));
        assert_eq!(m.order_min_size, Some(dec!(5)));
        let fs = m.fee_schedule.unwrap();
        assert_eq!(fs.rate, Some(dec!(0.07)));
        assert_eq!(fs.rebate_rate, Some(dec!(0.2)));

        let m: GammaMarket = serde_json::from_str(r#"{"orderPriceMinTickSize": 0.001}"#).unwrap();
        assert_eq!(m.order_price_min_tick_size, Some(dec!(0.001)));
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let e: GammaEvent =
            serde_json::from_str(r#"{"slug": "x", "someBrandNewField": {"deep": [1, 2]}}"#)
                .unwrap();
        assert_eq!(e.slug, "x");
        assert!(e.markets.is_empty());
    }

    #[test]
    fn event_metadata_absent_before_resolution() {
        // A live (unresolved) event has no eventMetadata — the field is None.
        let e: GammaEvent = serde_json::from_str(r#"{"slug": "btc-updown-5m-1"}"#).unwrap();
        assert!(e.event_metadata.is_none());
    }

    #[test]
    fn event_metadata_decodes_from_resolved_fixture() {
        // The committed real resolved-event response carries priceToBeat /
        // finalPrice (decoded through f64 — compare with tolerance, not exact).
        let raw = include_str!("../../tests/fixtures/gamma_event_resolved_5m.json");
        let events: Vec<GammaEvent> = serde_json::from_str(raw).unwrap();
        let meta = events[0]
            .event_metadata
            .as_ref()
            .expect("resolved metadata");
        let ptb = meta.price_to_beat.unwrap().to_f64().unwrap();
        let fp = meta.final_price.unwrap().to_f64().unwrap();
        assert!(
            (ptb - 63_788.969_145_188_41).abs() < 1e-3,
            "price_to_beat {ptb}"
        );
        assert!((fp - 63_757.579_317_401_6).abs() < 1e-3, "final_price {fp}");
        // finalPrice < priceToBeat ⇒ the window resolved Down.
        assert!(fp < ptb);
    }
}
