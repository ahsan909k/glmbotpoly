//! Shadow-mode settings (BUILD_PLAN 12–13): the observation-only live inference
//! of the champion model. Off by default (anti-surprise); when enabled the bot
//! runs a non-critical, venue-free shadow task that journals predictions only.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Configuration for the shadow observer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShadowConfig {
    /// Master gate. Off by default — shadow never runs unless explicitly enabled.
    pub enable: bool,
    /// Path to the deployed LightGBM `model.txt` (from `export_champion`).
    pub model_path: PathBuf,
    /// Path to the model's `.meta.json` sidecar (identity + training-end date).
    pub model_meta_path: PathBuf,
    /// Window-aligned inference cadence (seconds).
    pub sample_secs: u64,
    /// Shadow's own Binance depth20 base URL; empty reuses `feeds.binance_ws_url`.
    pub depth_ws_url: String,
    /// Directory for the `shadow-*.jsonl.gz` prediction series (separate from the
    /// journal and depth-capture series).
    pub shadow_dir: PathBuf,
    /// Dashboard "model stale, refit due" threshold (days).
    pub staleness_alert_days: u32,
    /// Capacity of the drop-on-full bus-event clone feeding shadow.
    pub bus_channel_cap: usize,
    /// Capacity of the depth-ladder channel (shadow feed → core).
    pub depth_channel_cap: usize,
    /// Capacity of the prediction-writer channel.
    pub record_channel_cap: usize,
    /// As-of feature staleness null-out (seconds) — the lab's 120 s.
    pub max_staleness_secs: u64,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enable: false,
            model_path: PathBuf::from("models/model_dir10_full.txt"),
            model_meta_path: PathBuf::from("models/model_dir10_full.meta.json"),
            sample_secs: 5,
            depth_ws_url: String::new(),
            shadow_dir: PathBuf::from("data/shadow"),
            staleness_alert_days: 30,
            bus_channel_cap: 1_024,
            depth_channel_cap: 4_096,
            record_channel_cap: 16_384,
            max_staleness_secs: 120,
        }
    }
}

impl ShadowConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(self.sample_secs > 0, "shadow.sample_secs", "must be > 0");
        v.require(
            self.max_staleness_secs > 0,
            "shadow.max_staleness_secs",
            "must be > 0",
        );
        v.require(
            self.staleness_alert_days > 0,
            "shadow.staleness_alert_days",
            "must be > 0",
        );
        for (key, cap) in [
            ("shadow.bus_channel_cap", self.bus_channel_cap),
            ("shadow.depth_channel_cap", self.depth_channel_cap),
            ("shadow.record_channel_cap", self.record_channel_cap),
        ] {
            v.require(cap > 0, key, "must be > 0");
        }
        if self.enable {
            v.require(
                !self.model_path.as_os_str().is_empty(),
                "shadow.model_path",
                "must be set when shadow is enabled",
            );
        }
        if !self.depth_ws_url.is_empty() {
            v.require(
                self.depth_ws_url.starts_with("wss://"),
                "shadow.depth_ws_url",
                "must start with wss:// (or be empty to reuse feeds.binance_ws_url)",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_off() {
        let cfg = ShadowConfig::default();
        assert!(!cfg.enable, "shadow defaults off");
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_bad_values() {
        let cfg = ShadowConfig {
            enable: true,
            model_path: PathBuf::new(),
            sample_secs: 0,
            max_staleness_secs: 0,
            staleness_alert_days: 0,
            bus_channel_cap: 0,
            depth_ws_url: "ws://insecure".to_owned(),
            ..ShadowConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"shadow.sample_secs"));
        assert!(keys.contains(&"shadow.max_staleness_secs"));
        assert!(keys.contains(&"shadow.staleness_alert_days"));
        assert!(keys.contains(&"shadow.bus_channel_cap"));
        assert!(keys.contains(&"shadow.model_path"));
        assert!(keys.contains(&"shadow.depth_ws_url"));
    }
}
