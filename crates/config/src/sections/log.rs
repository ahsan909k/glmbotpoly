//! Logging/tracing output settings.
//!
//! This crate stays free of any `tracing` dependency: the bot binary maps
//! [`LogRotation`] onto `tracing_appender::rolling::Rotation` and builds the
//! subscriber. Directive *syntax* is therefore validated in the binary (where
//! `EnvFilter` lives), not here.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// How often the log file rolls over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogRotation {
    /// New file every hour.
    Hourly,
    /// New file every day.
    Daily,
    /// Single file, never rotated.
    Never,
}

/// Tracing output configuration (console + rolling file).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogConfig {
    /// Directory for rolling log files (created on boot if absent).
    pub dir: PathBuf,
    /// Log file name prefix, e.g. `bot` → `bot.2026-06-11.log`.
    pub file_prefix: String,
    /// Rotation cadence.
    pub rotation: LogRotation,
    /// Maximum number of rotated files kept on disk.
    pub max_files: usize,
    /// Default per-target filter directives used when `RUST_LOG` is unset,
    /// e.g. `"info"` or `"info,engine=debug,feed_clob=trace"`.
    pub default_filter: String,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("data/logs"),
            file_prefix: "bot".to_owned(),
            rotation: LogRotation::Daily,
            max_files: 14,
            default_filter: "info".to_owned(),
        }
    }
}

impl LogConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            !self.dir.as_os_str().is_empty(),
            "log.dir",
            "must not be empty",
        );
        v.require(
            !self.file_prefix.is_empty(),
            "log.file_prefix",
            "must not be empty",
        );
        v.require(self.max_files >= 1, "log.max_files", "must be >= 1");
        v.require(
            !self.default_filter.trim().is_empty(),
            "log.default_filter",
            "must not be empty",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        LogConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rotation_parses_lowercase() {
        let r: LogRotation = serde_json::from_str("\"daily\"").unwrap();
        assert_eq!(r, LogRotation::Daily);
        assert!(serde_json::from_str::<LogRotation>("\"Daily\"").is_err());
        assert!(serde_json::from_str::<LogRotation>("\"weekly\"").is_err());
    }

    #[test]
    fn rejects_empty_and_zero() {
        let cfg = LogConfig {
            dir: PathBuf::new(),
            file_prefix: String::new(),
            max_files: 0,
            default_filter: "  ".to_owned(),
            ..LogConfig::default()
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert_eq!(v.into_result().unwrap_err().len(), 4);
    }
}
