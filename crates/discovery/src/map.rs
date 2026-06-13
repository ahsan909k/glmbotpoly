//! Pure mapping from wire types to validated [`MarketInfo`] values, plus the
//! Gamma↔CLOB cross-check. No I/O, no clocks — everything here is unit-tested
//! against committed fixtures.

use core_types::{
    ConditionId, DurationMs, FeeParams, MarketInfo, ResolutionSource, Series, Size, TickSize,
    TimestampMs, TokenId, TokenPair, WindowId,
};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::error::MapError;
use crate::wire::clob::ClobMarket;
use crate::wire::gamma::{GammaEvent, parse_double_encoded};

/// Parses an RFC3339 timestamp (`2026-06-11T08:10:00Z`, fractional seconds
/// accepted) into unix milliseconds.
///
/// # Errors
/// [`MapError::BadTimestamp`] when the value does not parse or overflows.
pub fn parse_rfc3339_ms(field: &'static str, value: &str) -> Result<TimestampMs, MapError> {
    let bad = || MapError::BadTimestamp {
        field,
        value: value.to_owned(),
    };
    let odt = OffsetDateTime::parse(value, &Rfc3339).map_err(|_| bad())?;
    let ms = i64::try_from(odt.unix_timestamp_nanos() / 1_000_000).map_err(|_| bad())?;
    Ok(TimestampMs::from_millis(ms))
}

/// Formats a timestamp as whole-second RFC3339 UTC (`2026-06-11T08:31:52Z`),
/// the shape Gamma accepts for `end_date_min`. Milliseconds are truncated
/// toward the past so the formatted instant never lies in the future of `ts`
/// (an event closing between the truncated and the real instant is one we
/// still want to see).
///
/// # Errors
/// [`MapError::BadTimestamp`] when `ts` is outside the RFC3339-representable
/// year range — only possible with a broken clock.
pub fn format_rfc3339_secs(ts: TimestampMs) -> Result<String, MapError> {
    let secs = ts.as_millis().div_euclid(1000);
    let odt = OffsetDateTime::from_unix_timestamp(secs).map_err(|_| MapError::BadTimestamp {
        field: "end_date_min",
        value: ts.as_millis().to_string(),
    })?;
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        odt.year(),
        u8::from(odt.month()),
        odt.day(),
        odt.hour(),
        odt.minute(),
        odt.second()
    ))
}

/// `Some(s)` only when the optional wire string is present and non-empty
/// (Gamma uses `""` where other APIs would use `null`).
fn nonempty(s: Option<&str>) -> Option<&str> {
    s.filter(|v| !v.is_empty())
}

/// Maps one Gamma event (which must embed exactly one market) into a
/// validated [`MarketInfo`] for `series`.
///
/// `expected_slug` is the configured Gamma series slug; an event claiming a
/// different `seriesSlug` is rejected rather than traded under the wrong
/// series.
///
/// # Errors
/// Any [`MapError`] — the caller must fail the whole series refresh, never
/// trade a half-understood window.
pub fn map_event(
    series: Series,
    expected_slug: &str,
    event: &GammaEvent,
) -> Result<MarketInfo, MapError> {
    if event.markets.len() != 1 {
        return Err(MapError::MarketCount(event.markets.len()));
    }
    let market = &event.markets[0];

    if let Some(got) = nonempty(event.series_slug.as_deref())
        && got != expected_slug
    {
        return Err(MapError::SeriesSlugMismatch {
            got: got.to_owned(),
            want: expected_slug.to_owned(),
        });
    }

    // Outcome labels and token ids arrive double-encoded and parallel; the
    // Up/Down assignment is by label, never by position.
    let outcomes_raw = market
        .outcomes
        .as_deref()
        .ok_or(MapError::MissingField("outcomes"))?;
    let outcomes = parse_double_encoded("outcomes", outcomes_raw)?;
    let ids_raw = market
        .clob_token_ids
        .as_deref()
        .ok_or(MapError::MissingField("clobTokenIds"))?;
    let token_ids = parse_double_encoded("clobTokenIds", ids_raw)?;
    if token_ids.len() != outcomes.len() {
        return Err(MapError::TokenCountMismatch {
            got: token_ids.len(),
            want: outcomes.len(),
        });
    }
    let up_idx = outcomes.iter().position(|o| o == "Up");
    let down_idx = outcomes.iter().position(|o| o == "Down");
    let (Some(up_idx), Some(down_idx)) = (up_idx, down_idx) else {
        return Err(MapError::BadOutcomes(format!("{outcomes:?}")));
    };
    if outcomes.len() != 2 || up_idx == down_idx {
        return Err(MapError::BadOutcomes(format!("{outcomes:?}")));
    }
    let tokens = TokenPair {
        up: TokenId::new(token_ids[up_idx].clone())?,
        down: TokenId::new(token_ids[down_idx].clone())?,
    };

    let condition_id = ConditionId::new(
        market
            .condition_id
            .clone()
            .ok_or(MapError::MissingField("conditionId"))?,
    )?;

    // Timing: Gamma is the authority. The market's eventStartTime is the
    // window open; event.startTime is a fallback (empty in hourly list
    // views); last resort is close − duration, which the invariant check
    // below then re-validates trivially but keeps WindowId well-defined.
    let close_raw = nonempty(market.end_date.as_deref())
        .or(nonempty(event.end_date.as_deref()))
        .ok_or(MapError::MissingField("endDate"))?;
    let close_time = parse_rfc3339_ms("endDate", close_raw)?;
    let duration = series.duration.as_duration();
    let open_time = match nonempty(market.event_start_time.as_deref())
        .or(nonempty(event.start_time.as_deref()))
    {
        Some(raw) => parse_rfc3339_ms("eventStartTime", raw)?,
        None => close_time.saturating_add(DurationMs::from_millis(-duration.as_millis())),
    };
    let got = close_time.signed_duration_since(open_time);
    if got != duration {
        return Err(MapError::DurationMismatch {
            got_ms: got.as_millis(),
            want_ms: duration.as_millis(),
        });
    }

    let tick_size = TickSize::from_decimal(
        market
            .order_price_min_tick_size
            .ok_or(MapError::MissingField("orderPriceMinTickSize"))?,
    )?;
    let min_order_size = Size::new(
        market
            .order_min_size
            .ok_or(MapError::MissingField("orderMinSize"))?,
    )
    .map_err(|_| MapError::BadMinSize)?;

    // Fees fail closed: a market whose fee descriptor we cannot read is a
    // market whose edge math we cannot trust.
    let schedule = market
        .fee_schedule
        .as_ref()
        .ok_or(MapError::MissingField("feeSchedule"))?;
    let fees = FeeParams {
        rate: schedule
            .rate
            .ok_or(MapError::MissingField("feeSchedule.rate"))?,
        exponent: schedule
            .exponent
            .ok_or(MapError::MissingField("feeSchedule.exponent"))?,
        taker_only: schedule
            .taker_only
            .ok_or(MapError::MissingField("feeSchedule.takerOnly"))?,
        rebate_rate: schedule
            .rebate_rate
            .ok_or(MapError::MissingField("feeSchedule.rebateRate"))?,
        enabled: market
            .fees_enabled
            .ok_or(MapError::MissingField("feesEnabled"))?,
    };

    let neg_risk = market.neg_risk.ok_or(MapError::MissingField("negRisk"))?;

    let resolution = ResolutionSource::classify(
        nonempty(market.resolution_source.as_deref())
            .or(nonempty(event.resolution_source.as_deref()))
            .unwrap_or(""),
    );

    Ok(MarketInfo {
        window: WindowId { series, open_time },
        event_slug: event.slug.clone(),
        condition_id,
        tokens,
        close_time,
        strike: None,
        tick_size,
        min_order_size,
        fees,
        neg_risk,
        resolution,
    })
}

/// One Gamma↔CLOB disagreement found by [`merge_clob`]. Logged at warn by
/// the caller; the strictest value has already been applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClobMismatch {
    /// Minimum order sizes differ (the larger one was kept).
    MinSize {
        /// Gamma's value.
        gamma: core_types::Decimal,
        /// CLOB's value.
        clob: core_types::Decimal,
    },
    /// Tick sizes differ (the coarser grid was kept: every price on the
    /// 0.01 grid is also on the 0.001 grid, so the coarser grid is the safe
    /// subset; the live `tick_size_change` event corrects it later).
    Tick {
        /// Gamma's value.
        gamma: TickSize,
        /// CLOB's value.
        clob: TickSize,
    },
    /// Neg-risk flags differ (Gamma's value was kept).
    NegRisk {
        /// Gamma's value.
        gamma: bool,
        /// CLOB's value.
        clob: bool,
    },
    /// CLOB token ids/labels could not be matched to Gamma's Up/Down pair
    /// (Gamma's pair was kept).
    TokenIds,
    /// CLOB reports the market as not currently accepting orders.
    NotAcceptingOrders,
}

/// Cross-checks a mapped [`MarketInfo`] against the CLOB market object and
/// applies the strictest of each constraint (CLAUDE.md §7): minimum order
/// size takes the max, tick mismatches take the coarser grid. Identity
/// fields (token ids, neg-risk) keep Gamma's values; disagreements are
/// returned for the caller to log.
pub fn merge_clob(info: &mut MarketInfo, clob: &ClobMarket) -> Vec<ClobMismatch> {
    let mut mismatches = Vec::new();

    if let Some(clob_min) = clob.minimum_order_size
        && clob_min != info.min_order_size.as_decimal()
    {
        mismatches.push(ClobMismatch::MinSize {
            gamma: info.min_order_size.as_decimal(),
            clob: clob_min,
        });
        if let Ok(stricter) = Size::new(clob_min)
            && clob_min > info.min_order_size.as_decimal()
        {
            info.min_order_size = stricter;
        }
    }

    if let Some(clob_tick_raw) = clob.minimum_tick_size
        && let Ok(clob_tick) = TickSize::from_decimal(clob_tick_raw)
        && clob_tick != info.tick_size
    {
        mismatches.push(ClobMismatch::Tick {
            gamma: info.tick_size,
            clob: clob_tick,
        });
        // Coarser grid = fewer decimal places = the smaller enum variant.
        info.tick_size = info.tick_size.min(clob_tick);
    }

    if let Some(clob_neg) = clob.neg_risk
        && clob_neg != info.neg_risk
    {
        mismatches.push(ClobMismatch::NegRisk {
            gamma: info.neg_risk,
            clob: clob_neg,
        });
    }

    let clob_token = |label: &str| {
        clob.tokens
            .iter()
            .find(|t| t.outcome.as_deref() == Some(label))
            .and_then(|t| t.token_id.as_deref())
    };
    let tokens_match = clob_token("Up") == Some(info.tokens.up.as_str())
        && clob_token("Down") == Some(info.tokens.down.as_str());
    if !tokens_match {
        mismatches.push(ClobMismatch::TokenIds);
    }

    if clob.accepting_orders == Some(false) {
        mismatches.push(ClobMismatch::NotAcceptingOrders);
    }

    mismatches
}

#[cfg(test)]
mod tests {
    use core_types::{Asset, Decimal, ResolutionKind, WindowDuration};
    use rust_decimal::dec;

    use super::*;
    use crate::wire::clob::ClobToken;

    const BTC_5M_EVENTS: &str = include_str!("../tests/fixtures/gamma_events_btc_5m.json");
    const BTC_1H_EVENTS: &str = include_str!("../tests/fixtures/gamma_events_btc_1h.json");
    const STALE_EVENT: &str = include_str!("../tests/fixtures/gamma_event_stale.json");

    const BTC_5M: Series = Series {
        asset: Asset::Btc,
        duration: WindowDuration::M5,
    };
    const BTC_1H: Series = Series {
        asset: Asset::Btc,
        duration: WindowDuration::H1,
    };

    fn events(json: &str) -> Vec<GammaEvent> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn maps_live_btc_5m_fixture_exactly() {
        let evs = events(BTC_5M_EVENTS);
        let info = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap();

        // Values pinned from the capture (fixtures README).
        assert_eq!(info.window.series, BTC_5M);
        assert_eq!(info.event_slug, "btc-updown-5m-1781166600");
        assert_eq!(
            info.window.open_time,
            TimestampMs::from_millis(1_781_166_600_000) // 2026-06-11T08:30:00Z
        );
        assert_eq!(
            info.close_time,
            TimestampMs::from_millis(1_781_166_900_000) // 2026-06-11T08:35:00Z
        );
        assert_eq!(
            info.condition_id.as_str(),
            "0xed902f990ca86222fae9df756181931f55c7cc6d2109b5cd030a43e0e3e13e0f"
        );
        assert_eq!(
            info.tokens.up.as_str(),
            "20803498828318266100737171439882396417264539614264015538918718996398906453768"
        );
        assert_eq!(
            info.tokens.down.as_str(),
            "75116187100981907890565535959927609318941259071974991159920661505109811570808"
        );
        assert_eq!(info.tick_size, TickSize::T001);
        assert_eq!(info.min_order_size.as_decimal(), dec!(5));
        assert_eq!(info.fees.rate, dec!(0.07));
        assert_eq!(info.fees.exponent, 1);
        assert!(info.fees.taker_only);
        assert_eq!(info.fees.rebate_rate, dec!(0.2));
        assert!(info.fees.enabled);
        assert!(!info.neg_risk);
        assert_eq!(info.strike, None);
        assert_eq!(info.resolution.kind, ResolutionKind::ChainlinkDataStream);
        assert_eq!(
            info.resolution.raw,
            "https://data.chain.link/streams/btc-usd"
        );
    }

    #[test]
    fn maps_hourly_fixture_via_market_event_start_time() {
        // Hourly list views omit the event-level startTime (absent in the
        // captured fixture, empty string in other views); the open must
        // come from market.eventStartTime (verified live).
        let evs = events(BTC_1H_EVENTS);
        assert!(evs[0].start_time.as_deref().is_none_or(str::is_empty));
        let info = map_event(BTC_1H, "btc-up-or-down-hourly", &evs[0]).unwrap();
        assert_eq!(
            info.window.open_time,
            TimestampMs::from_millis(1_781_164_800_000) // 2026-06-11T08:00:00Z
        );
        assert_eq!(
            info.close_time,
            TimestampMs::from_millis(1_781_168_400_000) // 2026-06-11T09:00:00Z
        );
        // The 1h series resolves on Binance candles, NOT Chainlink (§6).
        assert_eq!(info.resolution.kind, ResolutionKind::BinanceCandle);
    }

    #[test]
    fn maps_drifted_hourly_market_with_0001_tick() {
        // The stale fixture is a real drifted market whose tick flipped to
        // 0.001 — discovery must accept it as-is.
        let evs = events(STALE_EVENT);
        let info = map_event(BTC_1H, "btc-up-or-down-hourly", &evs[0]).unwrap();
        assert_eq!(info.tick_size, TickSize::T0001);
        assert_eq!(info.resolution.kind, ResolutionKind::BinanceCandle);
    }

    #[test]
    fn token_assignment_is_by_label_not_position() {
        let mut evs = events(BTC_5M_EVENTS);
        let market = &mut evs[0].markets[0];
        // Swap the label order but keep ids parallel: Up must still get the
        // id that sits at the "Up" position.
        let original = map_event(BTC_5M, "btc-up-or-down-5m", &events(BTC_5M_EVENTS)[0]).unwrap();
        market.outcomes = Some(r#"["Down", "Up"]"#.to_owned());
        market.clob_token_ids = Some(format!(
            r#"["{}", "{}"]"#,
            original.tokens.down.as_str(),
            original.tokens.up.as_str()
        ));
        let swapped = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap();
        assert_eq!(swapped.tokens, original.tokens);
    }

    #[test]
    fn rejects_non_up_down_outcomes() {
        let mut evs = events(BTC_5M_EVENTS);
        evs[0].markets[0].outcomes = Some(r#"["Yes", "No"]"#.to_owned());
        let err = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap_err();
        assert!(matches!(err, MapError::BadOutcomes(_)), "{err:?}");
    }

    #[test]
    fn rejects_token_count_mismatch() {
        let mut evs = events(BTC_5M_EVENTS);
        evs[0].markets[0].clob_token_ids = Some(r#"["123"]"#.to_owned());
        let err = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap_err();
        assert_eq!(err, MapError::TokenCountMismatch { got: 1, want: 2 });
    }

    #[test]
    fn rejects_missing_fee_schedule() {
        let mut evs = events(BTC_5M_EVENTS);
        evs[0].markets[0].fee_schedule = None;
        let err = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap_err();
        assert_eq!(err, MapError::MissingField("feeSchedule"));
    }

    #[test]
    fn rejects_duration_mismatch() {
        // A 5m event mapped as a 15m series (wrong slug→series wiring) must
        // be rejected by the duration invariant.
        let evs = events(BTC_5M_EVENTS);
        let series_15m = Series {
            asset: Asset::Btc,
            duration: WindowDuration::M15,
        };
        let err = map_event(series_15m, "btc-up-or-down-5m", &evs[0]).unwrap_err();
        assert_eq!(
            err,
            MapError::DurationMismatch {
                got_ms: 300_000,
                want_ms: 900_000,
            }
        );
    }

    #[test]
    fn rejects_wrong_series_slug() {
        let evs = events(BTC_5M_EVENTS);
        let err = map_event(BTC_5M, "eth-up-or-down-5m", &evs[0]).unwrap_err();
        assert_eq!(
            err,
            MapError::SeriesSlugMismatch {
                got: "btc-up-or-down-5m".to_owned(),
                want: "eth-up-or-down-5m".to_owned(),
            }
        );
    }

    #[test]
    fn rejects_wrong_market_count() {
        let mut evs = events(BTC_5M_EVENTS);
        let extra = evs[0].markets[0].clone();
        evs[0].markets.push(extra);
        let err = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap_err();
        assert_eq!(err, MapError::MarketCount(2));

        evs[0].markets.clear();
        let err = map_event(BTC_5M, "btc-up-or-down-5m", &evs[0]).unwrap_err();
        assert_eq!(err, MapError::MarketCount(0));
    }

    #[test]
    fn rfc3339_parsing_and_formatting() {
        assert_eq!(
            parse_rfc3339_ms("t", "2026-06-11T08:30:00Z").unwrap(),
            TimestampMs::from_millis(1_781_166_600_000)
        );
        // Fractional seconds (Gamma's createdAt-style fields carry them).
        assert_eq!(
            parse_rfc3339_ms("t", "2026-06-11T08:30:00.377922Z").unwrap(),
            TimestampMs::from_millis(1_781_166_600_377)
        );
        assert!(parse_rfc3339_ms("t", "").is_err());
        assert!(parse_rfc3339_ms("t", "yesterday").is_err());

        // Round-trip at whole-second resolution; ms truncate toward the past.
        assert_eq!(
            format_rfc3339_secs(TimestampMs::from_millis(1_781_166_712_388)).unwrap(),
            "2026-06-11T08:31:52Z"
        );
        let formatted = format_rfc3339_secs(TimestampMs::from_millis(1_781_166_600_000)).unwrap();
        assert_eq!(
            parse_rfc3339_ms("t", &formatted).unwrap(),
            TimestampMs::from_millis(1_781_166_600_000)
        );
    }

    // ---- merge_clob ----

    /// A CLOB market that agrees with `info` on every cross-checked field.
    fn agreeing_clob(info: &MarketInfo) -> ClobMarket {
        ClobMarket {
            condition_id: Some(info.condition_id.as_str().to_owned()),
            minimum_order_size: Some(info.min_order_size.as_decimal()),
            minimum_tick_size: Some(info.tick_size.as_decimal()),
            neg_risk: Some(info.neg_risk),
            accepting_orders: Some(true),
            tokens: vec![
                ClobToken {
                    token_id: Some(info.tokens.up.as_str().to_owned()),
                    outcome: Some("Up".to_owned()),
                },
                ClobToken {
                    token_id: Some(info.tokens.down.as_str().to_owned()),
                    outcome: Some("Down".to_owned()),
                },
            ],
        }
    }

    fn mapped_5m() -> MarketInfo {
        map_event(BTC_5M, "btc-up-or-down-5m", &events(BTC_5M_EVENTS)[0]).unwrap()
    }

    #[test]
    fn merge_clob_agreement_reports_nothing() {
        let mut info = mapped_5m();
        let clob = agreeing_clob(&info);
        let before = info.clone();
        assert!(merge_clob(&mut info, &clob).is_empty());
        assert_eq!(info, before);
    }

    #[test]
    fn merge_clob_fixture_against_mapped_market() {
        // The committed CLOB fixture is the same market as the first 5m
        // event — real cross-venue agreement.
        let clob: ClobMarket =
            serde_json::from_str(include_str!("../tests/fixtures/clob_market_btc_5m.json"))
                .unwrap();
        let mut info = mapped_5m();
        let mismatches = merge_clob(&mut info, &clob);
        assert!(mismatches.is_empty(), "{mismatches:?}");
        assert_eq!(info.min_order_size.as_decimal(), dec!(5));
        assert_eq!(info.tick_size, TickSize::T001);
    }

    #[test]
    fn merge_clob_takes_strictest_min_size() {
        let mut info = mapped_5m();
        let mut clob = agreeing_clob(&info);
        clob.minimum_order_size = Some(dec!(10));
        let mismatches = merge_clob(&mut info, &clob);
        assert_eq!(info.min_order_size.as_decimal(), dec!(10));
        assert!(mismatches.contains(&ClobMismatch::MinSize {
            gamma: dec!(5),
            clob: dec!(10),
        }));

        // The other direction keeps the (stricter) Gamma value.
        let mut info = mapped_5m();
        let mut clob = agreeing_clob(&info);
        clob.minimum_order_size = Some(dec!(1));
        merge_clob(&mut info, &clob);
        assert_eq!(info.min_order_size.as_decimal(), dec!(5));
    }

    #[test]
    fn merge_clob_takes_coarser_tick_on_mismatch() {
        // Gamma 0.01 vs CLOB 0.001 → coarser 0.01 stays.
        let mut info = mapped_5m();
        let mut clob = agreeing_clob(&info);
        clob.minimum_tick_size = Some(dec!(0.001));
        let mismatches = merge_clob(&mut info, &clob);
        assert_eq!(info.tick_size, TickSize::T001);
        assert!(mismatches.contains(&ClobMismatch::Tick {
            gamma: TickSize::T001,
            clob: TickSize::T0001,
        }));

        // Gamma 0.001 vs CLOB 0.01 → coarser 0.01 wins.
        let mut info = mapped_5m();
        info.tick_size = TickSize::T0001;
        let mut clob = agreeing_clob(&info);
        clob.minimum_tick_size = Some(Decimal::new(1, 2)); // 0.01
        merge_clob(&mut info, &clob);
        assert_eq!(info.tick_size, TickSize::T001);
    }

    #[test]
    fn merge_clob_reports_token_mismatch_and_keeps_gamma() {
        let mut info = mapped_5m();
        let mut clob = agreeing_clob(&info);
        clob.tokens.swap(0, 1); // ids stay attached to their entries…
        clob.tokens[0].outcome = Some("Up".to_owned()); // …but labels now lie
        clob.tokens[1].outcome = Some("Down".to_owned());
        let before = info.tokens.clone();
        let mismatches = merge_clob(&mut info, &clob);
        assert!(mismatches.contains(&ClobMismatch::TokenIds));
        assert_eq!(info.tokens, before);
    }

    #[test]
    fn merge_clob_reports_not_accepting_orders() {
        let mut info = mapped_5m();
        let mut clob = agreeing_clob(&info);
        clob.accepting_orders = Some(false);
        assert!(merge_clob(&mut info, &clob).contains(&ClobMismatch::NotAcceptingOrders));
    }
}
