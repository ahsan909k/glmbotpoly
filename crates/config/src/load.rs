//! Layered configuration assembly.
//!
//! Precedence, lowest to highest:
//!
//! 1. code defaults ([`AppConfig::default`] — the single source of truth),
//! 2. `<config_dir>/default.toml` — committed, **required** (a deploy that
//!    forgot its config dir must fail loudly, not run on baked-in defaults),
//! 3. `<config_dir>/bot.local.toml` — optional operator override, gitignored,
//! 4. `BOT_`-prefixed environment variables, nested keys separated by `__`
//!    (e.g. `BOT_RISK__FEED_STALENESS_MS=400` → `risk.feed_staleness_ms`).
//!
//! `BOT_SECRET_*` variables are **excluded** from the figment layers — they
//! are secrets, read separately by [`Secrets::from_env`] — and any secret key
//! found in a TOML file is rejected outright. Per-series engine overrides are
//! TOML-only: environment keys are lowercased, which cannot match the
//! case-sensitive series keys like `"BTC-5m"`.

use std::path::Path;

use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};

use crate::AppConfig;
use crate::error::ConfigError;
use crate::secrets::Secrets;

/// Environment variable prefix for config overrides (`BOT_`) — distinct from
/// the `BOT_SECRET_` namespace, which the loader filters out.
pub const ENV_PREFIX: &str = "BOT_";

/// Separator for nested keys in environment overrides (`BOT_RISK__…`).
pub const ENV_NESTED_SEPARATOR: &str = "__";

/// Dotted keys that must never appear in any config layer; a hit fails
/// closed. Usually file-sourced (the `BOT_SECRET_*` env namespace is filtered
/// out before merging), but a hand-built non-secret env key such as
/// `BOT_DASHBOARD__AUTH_TOKEN` maps here too — the error message covers both
/// sources. `deny_unknown_fields` on every section backstops spellings not on
/// this list.
const FORBIDDEN_FILE_KEYS: &[&str] = &[
    "dashboard.auth_token",
    "dashboard.token",
    "live.api_key",
    "live.api_secret",
    "live.api_passphrase",
    "live.private_key",
    "live.confirm",
    "live.confirm_phrase",
];

/// Loads the layered configuration from `config_dir` and the secrets from the
/// process environment. Call [`crate::validate()`] on the result before using
/// it.
///
/// # Errors
/// [`ConfigError::MissingDefaultFile`] when `default.toml` is absent,
/// [`ConfigError::SecretInConfigFile`] when a secret key appears in a file
/// layer, or [`ConfigError::Extract`] listing every type/key error figment
/// found.
pub fn load(config_dir: &Path) -> Result<(AppConfig, Secrets), ConfigError> {
    let default_path = config_dir.join("default.toml");
    if !default_path.is_file() {
        return Err(ConfigError::MissingDefaultFile { path: default_path });
    }

    // Both file layers use file_exact, NEVER Toml::file: for relative paths
    // (the default config dir is the relative "config") Toml::file searches
    // the current directory AND every parent directory, so a stray
    // <ancestor>/config/bot.local.toml could silently merge into the trading
    // config. The optional layer is gated on an explicit existence check.
    let local_path = config_dir.join("bot.local.toml");
    let mut figment = Figment::from(Serialized::defaults(AppConfig::default()))
        .merge(Toml::file_exact(&default_path));
    if local_path.is_file() {
        figment = figment.merge(Toml::file_exact(&local_path));
    }
    let figment = figment.merge(
        Env::prefixed(ENV_PREFIX)
            .split(ENV_NESTED_SEPARATOR)
            .filter(|key| {
                let key = key.as_str().to_ascii_lowercase();
                // Secrets never enter figment; BOT_CONFIG_DIR is consumed
                // by the binary before load. Both would otherwise fail
                // extraction as unknown root keys.
                !key.starts_with("secret_") && key != "config_dir"
            }),
    );

    for key in FORBIDDEN_FILE_KEYS {
        if figment.find_value(key).is_ok() {
            return Err(ConfigError::SecretInConfigFile {
                key: (*key).to_owned(),
            });
        }
    }

    let config: AppConfig = figment.extract().map_err(extract_errors)?;
    Ok((config, Secrets::from_env()))
}

/// Flattens figment's multi-error into displayable lines; each line already
/// carries the failing key path and the source file/provider.
fn extract_errors(err: figment::Error) -> ConfigError {
    ConfigError::Extract(err.into_iter().map(|e| e.to_string()).collect())
}

#[cfg(test)]
mod tests {
    // figment::Jail's closure contract returns figment::Error (208 bytes) by
    // value; the size is the library's choice, not ours, and this is test code.
    #![allow(clippy::result_large_err)]

    use core_types::{Decimal, DurationMs};
    use rust_decimal::dec;

    use super::*;

    /// Jail gives each test an isolated temp cwd and env-var scope, and
    /// serializes env access across the test binary.
    fn in_jail(f: impl FnOnce(&mut figment::Jail) -> Result<(), figment::Error>) {
        figment::Jail::expect_with(f);
    }

    fn write_minimal_default(jail: &mut figment::Jail, body: &str) -> Result<(), figment::Error> {
        jail.create_dir("config")?;
        jail.create_file("config/default.toml", body)?;
        Ok(())
    }

    #[test]
    fn layering_precedence_env_over_local_over_default_over_code() {
        in_jail(|jail| {
            // Stage 0+1: code defaults, overridden by default.toml.
            write_minimal_default(
                jail,
                r#"
                [paper]
                starting_capital = 1111
                "#,
            )?;
            let (cfg, _) = load(Path::new("config")).expect("default.toml only");
            assert_eq!(cfg.paper.starting_capital.as_decimal(), Decimal::from(1111));
            // Untouched key falls through to the code default.
            assert_eq!(cfg.paper.maker_rebate_share, dec!(0.20));

            // Stage 2: bot.local.toml overrides default.toml.
            jail.create_file(
                "config/bot.local.toml",
                r#"
                [paper]
                starting_capital = 2222
                "#,
            )?;
            let (cfg, _) = load(Path::new("config")).expect("with local override");
            assert_eq!(cfg.paper.starting_capital.as_decimal(), Decimal::from(2222));

            // Stage 3: environment overrides both files.
            jail.set_env("BOT_PAPER__STARTING_CAPITAL", "3333");
            let (cfg, _) = load(Path::new("config")).expect("with env override");
            assert_eq!(cfg.paper.starting_capital.as_decimal(), Decimal::from(3333));
            Ok(())
        });
    }

    #[test]
    fn missing_default_toml_is_an_error_missing_local_is_fine() {
        in_jail(|jail| {
            jail.create_dir("config")?;
            let err = load(Path::new("config")).expect_err("no default.toml");
            assert!(matches!(err, ConfigError::MissingDefaultFile { .. }));

            // With default.toml present and no bot.local.toml, loading works.
            jail.create_file("config/default.toml", "")?;
            assert!(load(Path::new("config")).is_ok());
            Ok(())
        });
    }

    #[test]
    fn env_nested_keys_map_to_sections() {
        in_jail(|jail| {
            write_minimal_default(jail, "")?;
            jail.set_env("BOT_RISK__FEED_STALENESS_MS", "400");
            jail.set_env("BOT_LOG__FILE_PREFIX", "custom");
            jail.set_env("BOT_DISCOVERY__LOOKAHEAD", "3");
            jail.set_env("BOT_DISCOVERY__SLUGS__BTC_5M", "btc-renamed-5m");
            let (cfg, _) = load(Path::new("config")).expect("env overrides");
            assert_eq!(cfg.risk.feed_staleness_ms, DurationMs::from_millis(400));
            assert_eq!(cfg.log.file_prefix, "custom");
            assert_eq!(cfg.discovery.lookahead, 3);
            assert_eq!(cfg.discovery.slugs.btc_5m, "btc-renamed-5m");
            // Untouched sibling slug falls through to the code default.
            assert_eq!(cfg.discovery.slugs.eth_5m, "eth-up-or-down-5m");
            Ok(())
        });
    }

    #[test]
    fn secret_and_config_dir_env_vars_are_filtered_not_errors() {
        in_jail(|jail| {
            write_minimal_default(jail, "")?;
            jail.set_env("BOT_SECRET_PM_API_KEY", "key-material");
            jail.set_env("BOT_SECRET_DASHBOARD_TOKEN", "token-material");
            jail.set_env("BOT_CONFIG_DIR", "somewhere/else");
            // deny_unknown_fields would reject these as root keys if the
            // filter ever regressed.
            let (cfg, secrets) = load(Path::new("config")).expect("secrets filtered");
            assert_eq!(cfg, AppConfig::default());
            assert!(secrets.pm_api_key.is_some());
            assert!(secrets.dashboard_token.is_some());
            Ok(())
        });
    }

    #[test]
    fn secret_keys_in_files_are_rejected() {
        let cases = [
            ("[dashboard]\nauth_token = \"oops\"", "dashboard.auth_token"),
            ("[live]\napi_key = \"oops\"", "live.api_key"),
            ("[live]\nprivate_key = \"oops\"", "live.private_key"),
            ("[live]\nconfirm = \"oops\"", "live.confirm"),
        ];
        for (body, expected_key) in cases {
            in_jail(|jail| {
                write_minimal_default(jail, body)?;
                let err = load(Path::new("config")).expect_err("secret in file");
                match err {
                    ConfigError::SecretInConfigFile { key } => assert_eq!(key, expected_key),
                    other => panic!("expected SecretInConfigFile, got {other:?}"),
                }
                Ok(())
            });
        }
    }

    #[test]
    fn secret_in_local_override_file_is_also_rejected() {
        in_jail(|jail| {
            write_minimal_default(jail, "")?;
            jail.create_file("config/bot.local.toml", "[live]\napi_secret = \"oops\"")?;
            let err = load(Path::new("config")).expect_err("secret in local file");
            assert!(
                matches!(err, ConfigError::SecretInConfigFile { key } if key == "live.api_secret")
            );
            Ok(())
        });
    }

    #[test]
    fn unknown_keys_are_extraction_errors_naming_the_key() {
        in_jail(|jail| {
            write_minimal_default(jail, "[engine.defaults]\nmin_edgee = 0.02")?;
            let err = load(Path::new("config")).expect_err("typo'd key");
            let ConfigError::Extract(lines) = err else {
                panic!("expected Extract");
            };
            assert!(
                lines.iter().any(|l| l.contains("min_edgee")),
                "error should name the unknown key: {lines:?}"
            );
            Ok(())
        });
    }

    #[test]
    fn decimals_parse_exactly_from_toml_floats_strings_and_env() {
        in_jail(|jail| {
            write_minimal_default(
                jail,
                r#"
                [paper]
                default_fee_rate = 0.07
                [engine.defaults]
                min_edge = "0.02"
                pair_cost_threshold = 0.98
                "#,
            )?;
            jail.set_env("BOT_PAPER__STARTING_CAPITAL", "2500.50");
            let (cfg, _) = load(Path::new("config")).expect("decimal forms");
            // The classic float trap: 0.07 must arrive as exactly 0.07.
            assert_eq!(cfg.paper.default_fee_rate, dec!(0.07));
            assert_eq!(cfg.engine.defaults.min_edge, dec!(0.02));
            assert_eq!(cfg.engine.defaults.pair_cost_threshold, dec!(0.98));
            assert_eq!(cfg.paper.starting_capital.as_decimal(), dec!(2500.50));
            Ok(())
        });
    }

    #[test]
    fn per_series_override_parses_from_toml() {
        in_jail(|jail| {
            write_minimal_default(
                jail,
                r#"
                [engine.series.BTC-1h]
                enabled = false

                [engine.series.ETH-5m.params]
                min_edge = 0.02
                "#,
            )?;
            let (cfg, _) = load(Path::new("config")).expect("series overrides");
            let btc_1h: core_types::Series = "BTC-1h".parse().expect("valid key");
            let eth_5m: core_types::Series = "ETH-5m".parse().expect("valid key");
            assert!(!cfg.engine.is_enabled(btc_1h));
            assert_eq!(cfg.engine.resolved(eth_5m).min_edge, dec!(0.02));
            // The patched series keeps every other default.
            assert_eq!(
                cfg.engine.resolved(eth_5m).pair_cost_threshold,
                cfg.engine.defaults.pair_cost_threshold
            );
            Ok(())
        });
    }

    #[test]
    fn committed_default_toml_matches_code_defaults() {
        // Anti-drift pin between config/default.toml (committed) and
        // AppConfig::default() (code). Value drift on either side fails here.
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/default.toml");
        assert!(
            path.is_file(),
            "committed defaults file missing at {}",
            path.display()
        );
        let cfg: AppConfig = Figment::from(Serialized::defaults(AppConfig::default()))
            .merge(Toml::file_exact(path))
            .extract()
            .expect("committed default.toml must extract cleanly");
        assert_eq!(cfg, AppConfig::default());
    }
}
