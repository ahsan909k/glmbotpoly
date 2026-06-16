//! The layered configuration for the bot (CLAUDE.md §4): file defaults overridden
//! by an optional local file, overridden by environment variables, parsed
//! into typed per-subsystem structs so each crate receives only the settings
//! it owns. (Runtime safe-list for the control plane lives in `sections::engine`.)
//!
//! - **Layering & loading:** [`load()`] — see `load.rs` for the precedence
//!   rules (code defaults → `default.toml` → `bot.local.toml` → `BOT_*` env).
//! - **Validation:** [`validate()`] — every range/cross-field rule, collected
//!   into one report.
//! - **Secrets:** [`Secrets`]/[`SecretString`] — environment-only, redacted in
//!   all output, not serializable (so the logged effective config can never
//!   contain one).
//! - **Safety invariant (CLAUDE.md §2.4):** nothing in this crate may default
//!   toward live trading. [`LiveConfig`] defaults disarmed; the confirmation
//!   phrase [`LIVE_CONFIRM_PHRASE`] is a hardcoded constant, not a config
//!   value.

mod error;
mod load;
mod secrets;
mod sections;
mod validate;

use serde::{Deserialize, Serialize};

pub use crate::error::{ConfigError, Violation};
pub use crate::load::{ENV_NESTED_SEPARATOR, ENV_PREFIX, load};
pub use crate::secrets::{
    ENV_DASHBOARD_TOKEN, ENV_LIVE_CONFIRM, ENV_PM_API_KEY, ENV_PM_API_PASSPHRASE,
    ENV_PM_API_SECRET, ENV_PM_PRIVATE_KEY, SecretString, Secrets,
};
pub use crate::sections::clock::ClockConfig;
pub use crate::sections::dashboard::DashboardConfig;
pub use crate::sections::discovery::{DiscoveryConfig, SeriesSlugs};
pub use crate::sections::engine::{
    EngineConfig, EngineParams, EngineParamsPatch, ParsedParamChange, SAFE_PARAM_KEYS,
    SeriesOverride, validate_param_change,
};
pub use crate::sections::feeds::FeedsConfig;
pub use crate::sections::journal::JournalConfig;
pub use crate::sections::latency::LatencyHarnessConfig;
pub use crate::sections::live::{LIVE_CONFIRM_PHRASE, LiveArming, LiveConfig};
pub use crate::sections::log::{LogConfig, LogRotation};
pub use crate::sections::paper::{LatencyConfig, PaperConfig};
pub use crate::sections::risk::RiskConfig;
pub use crate::sections::run::RunConfig;
pub use crate::sections::scheduler::SchedulerConfig;
pub use crate::validate::validate;

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).

/// The complete typed configuration tree, one field per subsystem.
///
/// Contains **no secrets** by construction ([`SecretString`] does not
/// implement `Serialize`, so one cannot even be added to a serializable
/// section by accident) — serializing this struct for the boot log is always
/// redaction-safe.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    /// Price/market data feed endpoints and connection behavior.
    pub feeds: FeedsConfig,
    /// Gamma/CLOB market discovery settings.
    pub discovery: DiscoveryConfig,
    /// 24/7 window-rollover scheduler timing policy.
    pub scheduler: SchedulerConfig,
    /// NTP cadence and clock-skew breaker policy (§11).
    pub clock: ClockConfig,
    /// Latency-benchmark harness settings (`bot latency`).
    pub latency: LatencyHarnessConfig,
    /// Per-series strategy parameters (defaults + sparse overrides).
    pub engine: EngineConfig,
    /// Global risk-manager limits.
    pub risk: RiskConfig,
    /// Main run-mode (`bot run`) supervision + startup self-check policy.
    pub run: RunConfig,
    /// Paper venue settings (starting capital, simulated latencies, fees).
    pub paper: PaperConfig,
    /// Live-arming gate 1 (defaults disarmed).
    pub live: LiveConfig,
    /// Journal storage locations.
    pub journal: JournalConfig,
    /// Dashboard server settings.
    pub dashboard: DashboardConfig,
    /// Tracing output settings.
    pub log: LogConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tree_serializes_for_boot_logging() {
        let json =
            serde_json::to_string_pretty(&AppConfig::default()).expect("AppConfig must serialize");
        // Spot-check structure: section names and a §8 default present.
        assert!(json.contains("\"paper\""));
        assert!(json.contains("\"starting_capital\": \"10000\""));
        assert!(json.contains("\"pair_cost_threshold\": \"0.98\""));
    }

    #[test]
    fn default_tree_round_trips_through_json() {
        let json = serde_json::to_string(&AppConfig::default()).expect("serialize");
        let back: AppConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, AppConfig::default());
    }

    #[test]
    fn live_defaults_disarmed() {
        let cfg = AppConfig::default();
        assert!(!cfg.live.enabled, "live trading must default OFF (§2.4)");
    }
}
