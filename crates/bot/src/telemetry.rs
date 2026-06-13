//! Tracing setup: per-target env-filter, compact console output, and a
//! non-blocking rolling file appender.
//!
//! Filter precedence: `RUST_LOG` when set and non-empty, otherwise
//! `log.default_filter` from the configuration. The same directives drive
//! both outputs (built twice — `EnvFilter` is not `Clone`).

use config::{LogConfig, LogRotation};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::{Layer as _, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

/// Failure to bring up the tracing stack.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// A filter directive string did not parse.
    #[error("invalid log filter directive {directive:?}: {source}")]
    Filter {
        /// The offending directive string (from `RUST_LOG` or `log.default_filter`).
        directive: String,
        /// The parse failure reported by tracing-subscriber.
        source: tracing_subscriber::filter::ParseError,
    },
    /// The log directory could not be created.
    #[error("failed to create log directory {dir:?}: {source}")]
    CreateDir {
        /// The configured log directory.
        dir: String,
        /// The underlying I/O failure.
        source: std::io::Error,
    },
    /// The rolling file appender could not be created.
    #[error("failed to initialize rolling log appender in {dir:?}: {source}")]
    Appender {
        /// The configured log directory.
        dir: String,
        /// The failure reported by tracing-appender.
        source: tracing_appender::rolling::InitError,
    },
    /// A global subscriber was already installed.
    #[error("failed to install global tracing subscriber: {0}")]
    Init(#[from] tracing_subscriber::util::TryInitError),
}

/// Parses one directive string into an [`EnvFilter`]. Pure: no environment
/// access, so it is unit-testable without touching global state.
///
/// Cosmetic whitespace around commas (`"info, engine=debug"`) and empty
/// segments are tolerated — the underlying parser is strict and would
/// otherwise abort the boot over a space.
fn build_filter(directive: &str) -> Result<EnvFilter, TelemetryError> {
    let normalized = directive
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .collect::<Vec<_>>()
        .join(",");
    EnvFilter::try_new(&normalized).map_err(|source| TelemetryError::Filter {
        directive: directive.to_owned(),
        source,
    })
}

/// The active directive string: `RUST_LOG` if set and non-empty, else the
/// configured default.
fn active_directive(log: &LogConfig) -> String {
    std::env::var("RUST_LOG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| log.default_filter.clone())
}

/// Installs the global subscriber: console layer (compact, ANSI, with
/// targets) plus rolling-file layer (plain text, no ANSI), both behind the
/// same per-target filter directives.
///
/// The returned [`WorkerGuard`] must stay alive until process exit — bind it
/// as `let _guard = …;` in `main` (never `let _ = …`, which drops it
/// immediately and silently discards buffered log lines).
///
/// # Errors
/// [`TelemetryError`] when the directives do not parse, the log directory is
/// unusable, or a global subscriber is already installed.
pub fn init(log: &LogConfig) -> Result<WorkerGuard, TelemetryError> {
    let directive = active_directive(log);
    let console_filter = build_filter(&directive)?;
    let file_filter = build_filter(&directive)?;

    // The appender's max_log_files pruning reads the directory at build time
    // and complains to stderr if it does not exist yet — create it first.
    std::fs::create_dir_all(&log.dir).map_err(|source| TelemetryError::CreateDir {
        dir: log.dir.display().to_string(),
        source,
    })?;

    let rotation = match log.rotation {
        LogRotation::Hourly => Rotation::HOURLY,
        LogRotation::Daily => Rotation::DAILY,
        LogRotation::Never => Rotation::NEVER,
    };
    let appender = RollingFileAppender::builder()
        .rotation(rotation)
        .filename_prefix(log.file_prefix.clone())
        .filename_suffix("log")
        .max_log_files(log.max_files)
        .build(&log.dir)
        .map_err(|source| TelemetryError::Appender {
            dir: log.dir.display().to_string(),
            source,
        })?;
    let (file_writer, guard) = tracing_appender::non_blocking(appender);

    let console_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(true)
        .with_filter(console_filter);
    let file_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(file_writer)
        .with_filter(file_filter);

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .try_init()?;

    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_directives_parse() {
        assert!(build_filter("info").is_ok());
        assert!(build_filter("info,engine=debug,feed_clob=trace").is_ok());
        assert!(build_filter("warn,bot::config=info").is_ok());
    }

    #[test]
    fn cosmetic_whitespace_and_empty_segments_are_tolerated() {
        assert!(build_filter("info, engine=debug").is_ok());
        assert!(build_filter(" info ,, feed_clob=trace ,").is_ok());
    }

    #[test]
    fn invalid_directives_are_typed_errors() {
        // A bare word is a valid target name, so the invalid part must be the
        // level.
        let err = build_filter("engine=superloud").unwrap_err();
        assert!(matches!(err, TelemetryError::Filter { .. }));
        assert!(err.to_string().contains("engine=superloud"));
    }
}

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).
