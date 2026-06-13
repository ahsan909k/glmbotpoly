//! The REST seam: a trait describing the three requests discovery needs,
//! plus the production reqwest implementation. Tests implement the trait
//! over committed fixtures — no live network, no mock-server dependency.

use std::future::Future;
use std::time::Duration;

use config::{DiscoveryConfig, FeedsConfig};
use core_types::{ConditionId, TimestampMs};
use serde::de::DeserializeOwned;

use crate::error::DiscoveryError;
use crate::map::format_rfc3339_secs;
use crate::wire::clob::ClobMarket;
use crate::wire::gamma::{GammaEvent, GammaSeries};

/// Maximum response-body bytes echoed into error messages.
const BODY_PREFIX_LIMIT: usize = 256;

/// REST operations discovery needs. The test seam: service-level tests
/// implement this over fixture bytes.
pub trait DiscoveryApi {
    /// `GET {gamma}/series?slug={slug}`.
    fn series_by_slug(
        &self,
        slug: &str,
    ) -> impl Future<Output = Result<Vec<GammaSeries>, DiscoveryError>> + Send;

    /// `GET {gamma}/events?series_id={id}&closed=false&order=endDate&ascending=true&end_date_min={now}&limit={limit}`.
    ///
    /// `end_date_min` is MANDATORY (verified live 2026-06-11): without it,
    /// stale unresolved events — still `closed=false` weeks after their end
    /// date — pollute the result.
    fn events_for_series(
        &self,
        series_id: &str,
        end_date_min: TimestampMs,
        limit: u32,
    ) -> impl Future<Output = Result<Vec<GammaEvent>, DiscoveryError>> + Send;

    /// `GET {clob_rest}/markets/{condition_id}` (public, no auth).
    fn clob_market(
        &self,
        condition_id: &ConditionId,
    ) -> impl Future<Output = Result<ClobMarket, DiscoveryError>> + Send;

    /// `GET {gamma}/events?slug={slug}` — fetch a single event by its slug.
    ///
    /// Used for post-hoc strike verification: a resolved event carries
    /// `eventMetadata.priceToBeat` (the authoritative strike), which appears a
    /// few minutes after close (verified live 2026-06-13).
    fn event_by_slug(
        &self,
        slug: &str,
    ) -> impl Future<Output = Result<Vec<GammaEvent>, DiscoveryError>> + Send;
}

/// reqwest-backed [`DiscoveryApi`] (rustls; timeout from
/// `discovery.http_timeout_ms`).
#[derive(Debug, Clone)]
pub struct HttpClient {
    http: reqwest::Client,
    gamma_base: String,
    clob_base: String,
}

impl HttpClient {
    /// Builds the production client from config. Base URLs come from
    /// `feeds.gamma_url` / `feeds.clob_rest_url`; trailing slashes are
    /// tolerated.
    ///
    /// # Errors
    /// [`DiscoveryError::ClientBuild`] if the TLS/connection-pool setup
    /// fails.
    pub fn new(feeds: &FeedsConfig, discovery: &DiscoveryConfig) -> Result<Self, DiscoveryError> {
        let timeout_ms = discovery.http_timeout_ms.as_millis().max(1);
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms.unsigned_abs()))
            .build()
            .map_err(DiscoveryError::ClientBuild)?;
        Ok(Self {
            http,
            gamma_base: feeds.gamma_url.trim_end_matches('/').to_owned(),
            clob_base: feeds.clob_rest_url.trim_end_matches('/').to_owned(),
        })
    }

    /// GETs `url` and decodes the JSON body, attaching url/status/body
    /// context to every failure mode.
    async fn get_json<T: DeserializeOwned>(&self, url: String) -> Result<T, DiscoveryError> {
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|source| DiscoveryError::Http {
                url: url.clone(),
                source,
            })?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|source| DiscoveryError::Http {
                url: url.clone(),
                source,
            })?;
        let body_prefix =
            || String::from_utf8_lossy(&body[..body.len().min(BODY_PREFIX_LIMIT)]).into_owned();
        if !status.is_success() {
            return Err(DiscoveryError::Status {
                url,
                status: status.as_u16(),
                body_prefix: body_prefix(),
            });
        }
        serde_json::from_slice(&body).map_err(|source| DiscoveryError::Decode {
            url,
            source,
            body_prefix: body_prefix(),
        })
    }
}

impl DiscoveryApi for HttpClient {
    async fn series_by_slug(&self, slug: &str) -> Result<Vec<GammaSeries>, DiscoveryError> {
        let url = format!("{}/series?slug={slug}", self.gamma_base);
        self.get_json(url).await
    }

    async fn events_for_series(
        &self,
        series_id: &str,
        end_date_min: TimestampMs,
        limit: u32,
    ) -> Result<Vec<GammaEvent>, DiscoveryError> {
        let min = format_rfc3339_secs(end_date_min)
            .map_err(|_| DiscoveryError::BadNow(end_date_min.as_millis()))?;
        let url = format!(
            "{}/events?series_id={series_id}&closed=false&order=endDate&ascending=true&end_date_min={min}&limit={limit}",
            self.gamma_base
        );
        self.get_json(url).await
    }

    async fn clob_market(&self, condition_id: &ConditionId) -> Result<ClobMarket, DiscoveryError> {
        let url = format!("{}/markets/{}", self.clob_base, condition_id.as_str());
        self.get_json(url).await
    }

    async fn event_by_slug(&self, slug: &str) -> Result<Vec<GammaEvent>, DiscoveryError> {
        let url = format!("{}/events?slug={slug}", self.gamma_base);
        self.get_json(url).await
    }
}
