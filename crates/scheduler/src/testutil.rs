//! Synthetic `MarketInfo` fixtures shared by machine and driver tests.
//! Scheduler tests need no wire-format fixtures — discovery owns the API
//! boundary; these build already-validated domain values.

use std::sync::Arc;

use core_types::{
    Asset, ConditionId, DurationMs, FeeParams, MarketInfo, ResolutionSource, Series, Size,
    TickSize, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId,
};
use discovery::SeriesWindows;
use rust_decimal::dec;

use crate::machine::Timing;

pub(crate) const BTC_5M: Series = Series {
    asset: Asset::Btc,
    duration: WindowDuration::M5,
};

/// A valid condition id derived from the window open time, so every window
/// gets a distinct, reproducible identity.
pub(crate) fn cid(open_ms: i64) -> ConditionId {
    ConditionId::new(format!("0x{open_ms:064x}")).unwrap()
}

/// Up token id for a window ("<open>1").
pub(crate) fn up_token(open_ms: i64) -> TokenId {
    TokenId::new(format!("{open_ms}1")).unwrap()
}

/// Down token id for a window ("<open>2").
pub(crate) fn down_token(open_ms: i64) -> TokenId {
    TokenId::new(format!("{open_ms}2")).unwrap()
}

/// A fully-populated synthetic window market: close = open + series duration.
pub(crate) fn market(series: Series, open_ms: i64) -> Arc<MarketInfo> {
    let close = open_ms + series.duration.as_duration().as_millis();
    Arc::new(MarketInfo {
        window: WindowId {
            series,
            open_time: TimestampMs::from_millis(open_ms),
        },
        event_slug: format!("test-{}-{open_ms}", series.key()),
        condition_id: cid(open_ms),
        tokens: TokenPair {
            up: up_token(open_ms),
            down: down_token(open_ms),
        },
        close_time: TimestampMs::from_millis(close),
        strike: None,
        tick_size: TickSize::T001,
        min_order_size: Size::new(dec!(5)).unwrap(),
        fees: FeeParams {
            rate: dec!(0.07),
            exponent: 1,
            taker_only: true,
            rebate_rate: dec!(0.2),
            enabled: true,
        },
        neg_risk: false,
        resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
    })
}

/// A discovery snapshot for tests.
pub(crate) fn windows(
    series: Series,
    current: Option<Arc<MarketInfo>>,
    upcoming: Vec<Arc<MarketInfo>>,
    fetched_at: TimestampMs,
) -> SeriesWindows {
    SeriesWindows {
        series,
        current,
        upcoming,
        fetched_at,
    }
}

/// The config-default timing.
pub(crate) fn timing(expect_resolutions: bool) -> Timing {
    Timing {
        refresh_lead: DurationMs::from_millis(120_000),
        next_window_lead: DurationMs::from_millis(60_000),
        closing_lead: DurationMs::from_millis(30_000),
        retry_initial: DurationMs::from_millis(1_000),
        retry_max: DurationMs::from_millis(30_000),
        resolution_timeout: DurationMs::from_millis(120_000),
        max_refresh_interval: DurationMs::from_millis(600_000),
        expect_resolutions,
    }
}
