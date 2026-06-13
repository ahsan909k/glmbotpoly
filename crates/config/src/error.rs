//! Typed configuration errors (CLAUDE.md §12: thiserror inside crates).
//!
//! Loading reports *all* figment extraction errors and validation reports
//! *all* semantic violations, so the operator can fix a broken config file in
//! one pass instead of replaying error-by-error.

use std::fmt;
use std::path::PathBuf;

/// One semantic validation failure: which key is wrong and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// Dotted config key, e.g. `"paper.starting_capital"` or
    /// `"engine.series.ETH-1h.min_edge"`.
    pub key: String,
    /// Human-readable explanation of the rule that failed.
    pub message: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.message)
    }
}

/// Any failure to produce a usable configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The committed defaults file is missing — the bot must never run on
    /// baked-in defaults alone (a deploy that forgot its config dir should
    /// fail loudly, not trade on silent fallbacks).
    #[error(
        "missing required config file {}: pass --config-dir, set BOT_CONFIG_DIR, \
         or run from the repo root (defaults live in ./config)",
        .path.display()
    )]
    MissingDefaultFile {
        /// The path that was expected to exist.
        path: PathBuf,
    },

    /// A secret-shaped key was found in a config layer (a TOML file, or a
    /// mis-namespaced `BOT_*` environment override). Secrets come only from
    /// `BOT_SECRET_*` environment variables (CLAUDE.md task rule); files get
    /// committed, copied, and backed up — secrets in them leak.
    #[error(
        "secret-shaped key \"{key}\" found in a config layer (a TOML file or a BOT_* \
         environment override) — secrets come only from the BOT_SECRET_* environment \
         variables (see .env.example); remove the key from the file or unset the variable"
    )]
    SecretInConfigFile {
        /// The forbidden dotted key that was present, e.g. `"dashboard.auth_token"`.
        key: String,
    },

    /// Figment extraction failed (type mismatches, unknown keys, malformed
    /// TOML). Each message already carries the key path and source file.
    #[error("failed to load configuration:\n{}", format_lines(.0))]
    Extract(Vec<String>),

    /// The configuration parsed but failed semantic validation. Every
    /// violation is listed.
    #[error("invalid configuration ({} violation{}):\n{}",
        .0.len(),
        if .0.len() == 1 { "" } else { "s" },
        format_violations(.0)
    )]
    Invalid(Vec<Violation>),
}

fn format_lines(lines: &[String]) -> String {
    lines
        .iter()
        .map(|l| format!("  - {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_violations(violations: &[Violation]) -> String {
    violations
        .iter()
        .map(|v| format!("  - {v}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_lists_every_violation() {
        let err = ConfigError::Invalid(vec![
            Violation {
                key: "paper.starting_capital".to_owned(),
                message: "must be > 0".to_owned(),
            },
            Violation {
                key: "risk.feed_staleness_ms".to_owned(),
                message: "must be <= 500".to_owned(),
            },
        ]);
        let text = err.to_string();
        assert!(text.contains("2 violations"));
        assert!(text.contains("paper.starting_capital: must be > 0"));
        assert!(text.contains("risk.feed_staleness_ms: must be <= 500"));
    }

    #[test]
    fn extract_lists_every_error() {
        let err = ConfigError::Extract(vec!["first".to_owned(), "second".to_owned()]);
        let text = err.to_string();
        assert!(text.contains("first"));
        assert!(text.contains("second"));
    }
}
