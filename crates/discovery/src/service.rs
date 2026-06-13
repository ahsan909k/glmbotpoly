//! The discovery service: per-series window discovery and metadata cache.
//!
//! Clock-free by design — `now` is always a parameter, so the scheduler
//! (which will own this service) controls all timing policy and tests stay
//! deterministic. Retry/backoff on failure is also the scheduler's job;
//! this module reports, it does not loop.

use std::collections::BTreeMap;
use std::sync::Arc;

use config::{DiscoveryConfig, FeedsConfig, SeriesSlugs};
use core_types::{MarketInfo, Series, TimestampMs};

use crate::client::{DiscoveryApi, HttpClient};
use crate::error::DiscoveryError;
use crate::map::{map_event, merge_clob};

/// Discovered windows for one series at one refresh instant.
#[derive(Debug, Clone)]
pub struct SeriesWindows {
    /// Which series this snapshot describes.
    pub series: Series,
    /// The window with `open ≤ now < close`, if one exists right now (a
    /// between-windows gap is legal and yields `None`).
    pub current: Option<Arc<MarketInfo>>,
    /// Future windows, soonest first. CLAUDE.md §6 requires at least one.
    pub upcoming: Vec<Arc<MarketInfo>>,
    /// The `now` the caller passed to the refresh that produced this.
    pub fetched_at: TimestampMs,
}

/// Result of refreshing all six series. Per-series isolation: one dead
/// series must never poison the others (CLAUDE.md §6).
#[derive(Debug)]
pub struct DiscoverySnapshot {
    /// One entry per series, in [`Series::ALL`] order.
    pub per_series: Vec<(Series, Result<SeriesWindows, DiscoveryError>)>,
}

impl DiscoverySnapshot {
    /// True when every series refreshed successfully.
    #[must_use]
    pub fn all_ok(&self) -> bool {
        self.per_series.iter().all(|(_, r)| r.is_ok())
    }
}

/// Splits mapped windows into (current, upcoming) relative to `now`:
/// windows with `close ≤ now` are dropped as stale, `open ≤ now < close`
/// is current (earliest close wins if the venue ever overlaps windows),
/// `open > now` are upcoming, sorted soonest first.
#[must_use]
pub fn classify_windows(
    markets: Vec<MarketInfo>,
    now: TimestampMs,
) -> (Option<MarketInfo>, Vec<MarketInfo>) {
    let mut current: Option<MarketInfo> = None;
    let mut upcoming: Vec<MarketInfo> = Vec::new();
    for market in markets {
        if market.close_time <= now {
            continue; // stale: already closed (ties: a window closing exactly now is over)
        }
        if market.window.open_time <= now {
            let earlier = current
                .as_ref()
                .is_none_or(|cur| market.close_time < cur.close_time);
            if earlier {
                current = Some(market);
            }
        } else {
            upcoming.push(market);
        }
    }
    upcoming.sort_by_key(|m| m.window.open_time);
    (current, upcoming)
}

/// Per-series discovery + metadata cache, generic over the REST seam.
#[derive(Debug)]
pub struct DiscoveryService<A> {
    api: A,
    slugs: SeriesSlugs,
    lookahead: u32,
    /// Gamma series ids, resolved from slugs on first use and cached for
    /// the process lifetime.
    series_ids: BTreeMap<Series, String>,
    cache: BTreeMap<Series, SeriesWindows>,
}

impl DiscoveryService<HttpClient> {
    /// Production constructor: reqwest client + slugs/lookahead from config.
    ///
    /// # Errors
    /// [`DiscoveryError::ClientBuild`] if the HTTP client cannot be built.
    pub fn from_config(
        feeds: &FeedsConfig,
        discovery: &DiscoveryConfig,
    ) -> Result<Self, DiscoveryError> {
        Ok(Self::new(
            HttpClient::new(feeds, discovery)?,
            discovery.slugs.clone(),
            discovery.lookahead,
        ))
    }
}

impl<A: DiscoveryApi> DiscoveryService<A> {
    /// Test/DI constructor over any [`DiscoveryApi`] implementation.
    pub fn new(api: A, slugs: SeriesSlugs, lookahead: u32) -> Self {
        Self {
            api,
            slugs,
            lookahead,
            series_ids: BTreeMap::new(),
            cache: BTreeMap::new(),
        }
    }

    /// The last successful refresh for `series`, if any.
    #[must_use]
    pub fn cached(&self, series: Series) -> Option<&SeriesWindows> {
        self.cache.get(&series)
    }

    /// Refreshes one series: resolve its Gamma series id (cached after the
    /// first call) → fetch the next `lookahead + 1` live windows → map each
    /// to [`MarketInfo`] → cross-check each against the CLOB market object
    /// (mismatches logged at warn, strictest constraint applied) → classify
    /// into current/upcoming → cache and return.
    ///
    /// # Errors
    /// Any [`DiscoveryError`]. A mapping failure on any event fails the
    /// whole series — a half-understood series must not trade. The caller
    /// (scheduler) owns retry/backoff per CLAUDE.md §6.
    pub async fn refresh_series(
        &mut self,
        series: Series,
        now: TimestampMs,
    ) -> Result<SeriesWindows, DiscoveryError> {
        let series_id = self.resolve_series_id(series).await?;

        let limit = self.lookahead.saturating_add(1);
        let events = self.api.events_for_series(&series_id, now, limit).await?;
        if events.is_empty() {
            return Err(DiscoveryError::NoWindows { series });
        }

        let expected_slug = self.slugs.get(series);
        let mut markets = Vec::with_capacity(events.len());
        for event in &events {
            let mut info =
                map_event(series, expected_slug, event).map_err(|source| DiscoveryError::Map {
                    event_slug: event.slug.clone(),
                    source,
                })?;
            let clob = self.api.clob_market(&info.condition_id).await?;
            for mismatch in merge_clob(&mut info, &clob) {
                tracing::warn!(
                    target: "discovery",
                    series = %series,
                    event = %event.slug,
                    condition_id = %info.condition_id,
                    ?mismatch,
                    "gamma/clob cross-check mismatch (strictest value applied)"
                );
            }
            markets.push(info);
        }

        let (current, upcoming) = classify_windows(markets, now);
        if current.is_none() && upcoming.is_empty() {
            return Err(DiscoveryError::NoWindows { series });
        }
        let windows = SeriesWindows {
            series,
            current: current.map(Arc::new),
            upcoming: upcoming.into_iter().map(Arc::new).collect(),
            fetched_at: now,
        };
        self.cache.insert(series, windows.clone());
        Ok(windows)
    }

    /// Refreshes all six series in [`Series::ALL`] order, collecting
    /// per-series results. Never fails wholesale: each series gets its own
    /// `Result`.
    pub async fn refresh_all(&mut self, now: TimestampMs) -> DiscoverySnapshot {
        let mut per_series = Vec::with_capacity(Series::ALL.len());
        for series in Series::ALL {
            per_series.push((series, self.refresh_series(series, now).await));
        }
        DiscoverySnapshot { per_series }
    }

    /// Slug → Gamma series id, cached forever after the first successful
    /// resolution.
    async fn resolve_series_id(&mut self, series: Series) -> Result<String, DiscoveryError> {
        if let Some(id) = self.series_ids.get(&series) {
            return Ok(id.clone());
        }
        let slug = self.slugs.get(series).to_owned();
        let matches = self.api.series_by_slug(&slug).await?;
        let found = matches
            .into_iter()
            .find(|s| s.slug == slug)
            .ok_or(DiscoveryError::SeriesNotFound { slug })?;
        // Sanity: the Gamma series cadence should match our duration. A
        // mismatch means the configured slug points at the wrong product;
        // the per-event duration check will reject its windows fatally, so
        // warn here rather than fail twice.
        let expected_recurrence = match series.duration {
            core_types::WindowDuration::M5 => "5m",
            core_types::WindowDuration::M15 => "15m",
            core_types::WindowDuration::H1 => "hourly",
        };
        if let Some(recurrence) = &found.recurrence
            && recurrence != expected_recurrence
        {
            tracing::warn!(
                target: "discovery",
                series = %series,
                slug = %found.slug,
                %recurrence,
                expected = expected_recurrence,
                "gamma series recurrence does not match the configured series duration"
            );
        }
        self.series_ids.insert(series, found.id.clone());
        Ok(found.id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    use core_types::{Asset, ConditionId, WindowDuration};

    use super::*;
    use crate::error::MapError;
    use crate::map::map_event;
    use crate::wire::clob::{ClobMarket, ClobToken};
    use crate::wire::gamma::{GammaEvent, GammaSeries};

    /// `now` at fixture capture (2026-06-11T08:31:52.388Z) — see the
    /// fixtures README. Inside the first 5m window (08:30–08:35) and the
    /// 4AM-ET hourly window.
    const CAPTURE_NOW: TimestampMs = TimestampMs::from_millis(1_781_166_712_388);

    const BTC_5M: Series = Series {
        asset: Asset::Btc,
        duration: WindowDuration::M5,
    };

    fn fixture_events() -> Vec<GammaEvent> {
        serde_json::from_str(include_str!("../tests/fixtures/gamma_events_btc_5m.json")).unwrap()
    }

    #[test]
    fn gamma_series_fixture_parses() {
        let series: Vec<GammaSeries> =
            serde_json::from_str(include_str!("../tests/fixtures/gamma_series_btc_5m.json"))
                .unwrap();
        let found = series
            .iter()
            .find(|s| s.slug == "btc-up-or-down-5m")
            .unwrap();
        assert_eq!(found.id, "10684");
        assert_eq!(found.recurrence.as_deref(), Some("5m"));
    }

    /// A CLOB market that agrees with `info` on every cross-checked field.
    fn agreeing_clob(info: &core_types::MarketInfo) -> ClobMarket {
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

    struct FakeApi {
        series: Vec<GammaSeries>,
        events: HashMap<String, Vec<GammaEvent>>,
        clob: HashMap<String, ClobMarket>,
        series_calls: AtomicU32,
    }

    impl FakeApi {
        /// All six series resolvable; events/clob only for what the test
        /// registers.
        fn new() -> Self {
            let slugs = SeriesSlugs::default();
            let series = Series::ALL
                .iter()
                .enumerate()
                .map(|(i, &s)| GammaSeries {
                    id: format!("{}", 10_000 + i),
                    slug: slugs.get(s).to_owned(),
                    recurrence: Some(
                        match s.duration {
                            WindowDuration::M5 => "5m",
                            WindowDuration::M15 => "15m",
                            WindowDuration::H1 => "hourly",
                        }
                        .to_owned(),
                    ),
                })
                .collect();
            Self {
                series,
                events: HashMap::new(),
                clob: HashMap::new(),
                series_calls: AtomicU32::new(0),
            }
        }

        fn fake_id(series: Series) -> String {
            let idx = Series::ALL.iter().position(|&s| s == series).unwrap();
            format!("{}", 10_000 + idx)
        }

        /// Registers the BTC-5m fixture events plus agreeing CLOB markets
        /// for each.
        fn with_btc_5m_fixture(mut self) -> Self {
            let events = fixture_events();
            for event in &events {
                let info = map_event(BTC_5M, "btc-up-or-down-5m", event).unwrap();
                self.clob
                    .insert(info.condition_id.as_str().to_owned(), agreeing_clob(&info));
            }
            self.events.insert(Self::fake_id(BTC_5M), events);
            self
        }
    }

    impl DiscoveryApi for FakeApi {
        async fn series_by_slug(&self, slug: &str) -> Result<Vec<GammaSeries>, DiscoveryError> {
            self.series_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .series
                .iter()
                .filter(|s| s.slug == slug)
                .cloned()
                .collect())
        }

        async fn events_for_series(
            &self,
            series_id: &str,
            _end_date_min: TimestampMs,
            limit: u32,
        ) -> Result<Vec<GammaEvent>, DiscoveryError> {
            let events = self.events.get(series_id).ok_or(DiscoveryError::Status {
                url: format!("fake://events/{series_id}"),
                status: 500,
                body_prefix: "no events registered".to_owned(),
            })?;
            Ok(events.iter().take(limit as usize).cloned().collect())
        }

        async fn clob_market(
            &self,
            condition_id: &ConditionId,
        ) -> Result<ClobMarket, DiscoveryError> {
            self.clob
                .get(condition_id.as_str())
                .cloned()
                .ok_or(DiscoveryError::Status {
                    url: format!("fake://clob/{condition_id}"),
                    status: 404,
                    body_prefix: "no clob market registered".to_owned(),
                })
        }

        async fn event_by_slug(&self, slug: &str) -> Result<Vec<GammaEvent>, DiscoveryError> {
            // Scan every registered series for an event with this slug.
            Ok(self
                .events
                .values()
                .flatten()
                .filter(|e| e.slug == slug)
                .cloned()
                .collect())
        }
    }

    fn service(api: FakeApi) -> DiscoveryService<FakeApi> {
        DiscoveryService::new(api, SeriesSlugs::default(), 2)
    }

    #[tokio::test]
    async fn refresh_series_end_to_end_against_fixtures() {
        let mut svc = service(FakeApi::new().with_btc_5m_fixture());
        let windows = svc.refresh_series(BTC_5M, CAPTURE_NOW).await.unwrap();

        let current = windows.current.expect("capture instant is mid-window");
        assert_eq!(
            current.window.open_time,
            TimestampMs::from_millis(1_781_166_600_000)
        );
        assert_eq!(windows.upcoming.len(), 2);
        assert!(windows.upcoming[0].window.open_time < windows.upcoming[1].window.open_time);
        assert_eq!(windows.upcoming[0].window.open_time, current.close_time);
        assert_eq!(windows.fetched_at, CAPTURE_NOW);

        // The result is cached.
        let cached = svc.cached(BTC_5M).expect("cached after refresh");
        assert_eq!(cached.fetched_at, CAPTURE_NOW);
    }

    #[tokio::test]
    async fn series_id_resolution_is_cached_across_refreshes() {
        let mut svc = service(FakeApi::new().with_btc_5m_fixture());
        svc.refresh_series(BTC_5M, CAPTURE_NOW).await.unwrap();
        svc.refresh_series(BTC_5M, CAPTURE_NOW).await.unwrap();
        assert_eq!(svc.api.series_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn refresh_all_isolates_per_series_failures() {
        // Only BTC-5m has events registered; the other five series fail.
        let mut svc = service(FakeApi::new().with_btc_5m_fixture());
        let snapshot = svc.refresh_all(CAPTURE_NOW).await;

        assert!(!snapshot.all_ok());
        assert_eq!(snapshot.per_series.len(), 6);
        let ok: Vec<Series> = snapshot
            .per_series
            .iter()
            .filter_map(|(s, r)| r.is_ok().then_some(*s))
            .collect();
        assert_eq!(ok, vec![BTC_5M]);
    }

    #[tokio::test]
    async fn empty_event_list_is_no_windows() {
        let mut api = FakeApi::new();
        api.events.insert(FakeApi::fake_id(BTC_5M), Vec::new());
        let mut svc = service(api);
        let err = svc.refresh_series(BTC_5M, CAPTURE_NOW).await.unwrap_err();
        assert!(matches!(err, DiscoveryError::NoWindows { series } if series == BTC_5M));
    }

    #[tokio::test]
    async fn unknown_slug_is_series_not_found() {
        let mut svc = DiscoveryService::new(
            FakeApi::new(),
            SeriesSlugs {
                btc_5m: "btc-renamed".to_owned(),
                ..SeriesSlugs::default()
            },
            2,
        );
        let err = svc.refresh_series(BTC_5M, CAPTURE_NOW).await.unwrap_err();
        assert!(matches!(err, DiscoveryError::SeriesNotFound { slug } if slug == "btc-renamed"));
    }

    #[tokio::test]
    async fn mapping_failure_fails_the_series() {
        let mut api = FakeApi::new().with_btc_5m_fixture();
        let events = api.events.get_mut(&FakeApi::fake_id(BTC_5M)).unwrap();
        events[1].markets[0].fee_schedule = None; // poison the SECOND event
        let mut svc = service(api);
        let err = svc.refresh_series(BTC_5M, CAPTURE_NOW).await.unwrap_err();
        match err {
            DiscoveryError::Map { event_slug, source } => {
                assert_eq!(event_slug, "btc-updown-5m-1781166900");
                assert_eq!(source, MapError::MissingField("feeSchedule"));
            }
            other => panic!("expected Map error, got {other:?}"),
        }
    }

    // ---- classify_windows (pure) ----

    fn mk_info(open_ms: i64, close_ms: i64) -> core_types::MarketInfo {
        let mut info = map_event(BTC_5M, "btc-up-or-down-5m", &fixture_events()[0]).unwrap();
        info.window.open_time = TimestampMs::from_millis(open_ms);
        info.close_time = TimestampMs::from_millis(close_ms);
        info
    }

    #[test]
    fn classify_mid_window_is_current() {
        let (current, upcoming) = classify_windows(
            vec![mk_info(0, 300), mk_info(300, 600)],
            TimestampMs::from_millis(150),
        );
        assert_eq!(current.unwrap().close_time, TimestampMs::from_millis(300));
        assert_eq!(upcoming.len(), 1);
    }

    #[test]
    fn classify_now_equal_to_close_drops_the_window() {
        let (current, upcoming) =
            classify_windows(vec![mk_info(0, 300)], TimestampMs::from_millis(300));
        assert!(current.is_none());
        assert!(upcoming.is_empty());
    }

    #[test]
    fn classify_now_equal_to_open_is_current() {
        let (current, _) = classify_windows(vec![mk_info(300, 600)], TimestampMs::from_millis(300));
        assert!(current.is_some());
    }

    #[test]
    fn classify_all_future_gives_no_current_and_sorts_upcoming() {
        let (current, upcoming) = classify_windows(
            vec![mk_info(900, 1200), mk_info(300, 600), mk_info(600, 900)],
            TimestampMs::from_millis(100),
        );
        assert!(current.is_none());
        let opens: Vec<i64> = upcoming
            .iter()
            .map(|m| m.window.open_time.as_millis())
            .collect();
        assert_eq!(opens, vec![300, 600, 900]);
    }

    #[test]
    fn classify_drops_stale_windows() {
        // The real stale fixture: closed weeks before the capture instant.
        let stale_events: Vec<GammaEvent> =
            serde_json::from_str(include_str!("../tests/fixtures/gamma_event_stale.json")).unwrap();
        let series_1h = Series {
            asset: Asset::Btc,
            duration: WindowDuration::H1,
        };
        let info = map_event(series_1h, "btc-up-or-down-hourly", &stale_events[0]).unwrap();
        let (current, upcoming) = classify_windows(vec![info], CAPTURE_NOW);
        assert!(current.is_none());
        assert!(upcoming.is_empty());
    }
}
