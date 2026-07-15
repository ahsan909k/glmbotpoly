//! Runtime parameters and the deployed-model identity.
//!
//! [`ShadowParams`] is mapped from the `[shadow]` config section at the `bot`
//! boundary (the crate has no `config` dependency). [`ModelIdentity`] is loaded
//! from the exported `model_*.meta.json` sidecar and journaled with every
//! prediction; it also drives the dashboard "model stale, refit due" alert.

use std::path::PathBuf;
use std::time::Duration;

use core_types::Asset;
use feed_util::BackoffParams;

use crate::depth_feed::DepthFeedParams;

/// Tuning for a shadow run (mapped from `[shadow]` + `[feeds]` in `bot`).
#[derive(Debug, Clone)]
pub struct ShadowParams {
    /// Assets to track (BTC + ETH for the four champion series).
    pub assets: Vec<Asset>,
    /// Window-aligned inference cadence (seconds).
    pub sample_secs: u64,
    /// As-of staleness null-out (ms) — the lab's 120 s.
    pub max_staleness_ms: i64,
    /// Shadow's own depth20 feed.
    pub depth: DepthFeedParams,
    /// Directory for the `shadow-*.jsonl.gz` prediction series.
    pub shadow_dir: PathBuf,
    /// Bounded prediction-writer channel capacity.
    pub record_channel_cap: usize,
    /// Bounded depth-ladder channel capacity.
    pub depth_channel_cap: usize,
    /// Dashboard "model stale, refit due" threshold (days).
    pub staleness_alert_days: i64,
}

impl Default for ShadowParams {
    fn default() -> Self {
        Self {
            assets: vec![Asset::Btc, Asset::Eth],
            sample_secs: 5,
            max_staleness_ms: 120_000,
            depth: DepthFeedParams {
                url: crate::depth_feed::combined_depth_url("wss://stream.binance.com:9443"),
                connect_timeout: Duration::from_secs(5),
                backoff: BackoffParams {
                    initial: Duration::from_millis(250),
                    max: Duration::from_millis(10_000),
                    multiplier: 2.0,
                },
            },
            shadow_dir: PathBuf::from("data/shadow"),
            record_channel_cap: 16_384,
            depth_channel_cap: 4_096,
            staleness_alert_days: 30,
        }
    }
}

/// The deployed model's identity (from `model_*.meta.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    /// Model file sha256 (hex).
    pub sha256: String,
    /// Training-data end (ms) — drives the staleness alert.
    pub trained_through_ms: i64,
    /// Training seed.
    pub seed: i64,
}

/// The sidecar meta shape (extra fields ignored).
#[derive(serde::Deserialize)]
struct MetaFile {
    sha256: String,
    trained_through_ms: i64,
    seed: i64,
}

impl ModelIdentity {
    /// Loads the identity from a `model_*.meta.json` sidecar.
    ///
    /// # Errors
    /// [`anyhow::Error`] if the file cannot be read or parsed.
    pub fn load(meta_path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(meta_path)
            .map_err(|e| anyhow::anyhow!("reading model meta {}: {e}", meta_path.display()))?;
        let meta: MetaFile = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing model meta {}: {e}", meta_path.display()))?;
        Ok(Self {
            sha256: meta.sha256,
            trained_through_ms: meta.trained_through_ms,
            seed: meta.seed,
        })
    }

    /// Short sha for display.
    #[must_use]
    pub fn short_sha(&self) -> String {
        self.sha256.chars().take(12).collect()
    }
}
