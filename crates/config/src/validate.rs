//! Semantic validation: range and cross-field rules that types alone cannot
//! express. Collects **every** violation before failing so the operator fixes
//! a broken config in one pass.

use crate::AppConfig;
use crate::error::{ConfigError, Violation};
use crate::secrets::{ENV_DASHBOARD_TOKEN, ENV_LIVE_CONFIRM, Secrets};
use crate::sections::live::LIVE_CONFIRM_PHRASE;

/// Accumulates violations across all sections.
#[derive(Debug, Default)]
pub(crate) struct Violations {
    items: Vec<Violation>,
}

impl Violations {
    /// Records a violation for `key` unless `condition` holds.
    pub(crate) fn require(
        &mut self,
        condition: bool,
        key: impl Into<String>,
        message: impl Into<String>,
    ) {
        if !condition {
            self.items.push(Violation {
                key: key.into(),
                message: message.into(),
            });
        }
    }

    /// `Ok` when nothing was recorded, otherwise every violation.
    pub(crate) fn into_result(self) -> Result<(), Vec<Violation>> {
        if self.items.is_empty() {
            Ok(())
        } else {
            Err(self.items)
        }
    }
}

/// Validates the loaded configuration against every range and cross-field
/// rule, including the cross-checks against environment secrets (dashboard
/// exposure, live-arming completeness).
///
/// # Errors
/// [`ConfigError::Invalid`] listing **all** violations found.
pub fn validate(config: &AppConfig, secrets: &Secrets) -> Result<(), ConfigError> {
    let mut v = Violations::default();

    config.feeds.validate_into(&mut v);
    config.discovery.validate_into(&mut v);
    config.scheduler.validate_into(&mut v);
    config.clock.validate_into(&mut v);
    config.latency.validate_into(&mut v);
    config.engine.validate_into(&mut v);
    config.risk.validate_into(&mut v);
    config.run.validate_into(&mut v);
    config.paper.validate_into(&mut v);
    config.journal.validate_into(&mut v);
    config.log.validate_into(&mut v);
    config.shadow.validate_into(&mut v);
    config.model_taker.validate_into(&mut v);

    // The model taker consumes the shadow model's prediction stream, so it
    // cannot run unless shadow is enabled — a config error, not a silent no-op.
    if config.model_taker.enable && !config.model_taker.kill_switch {
        v.require(
            config.shadow.enable,
            "model_taker.enable",
            "requires shadow.enable = true (the model taker's prediction source)",
        );
    }

    // Exposing the dashboard beyond loopback without auth would hand the
    // kill switch and live-arming UI to the local network.
    if !config.dashboard.bind.ip().is_loopback() {
        v.require(
            secrets.dashboard_token.is_some(),
            "dashboard.bind",
            format!("binding a non-loopback address requires {ENV_DASHBOARD_TOKEN} to be set"),
        );
    }

    // Checked even when booting in paper mode: a half-armed live config
    // should fail at boot, not on the day the operator flips to live.
    if config.live.enabled {
        v.require(
            secrets.has_all_pm_credentials(),
            "live.enabled",
            "live mode requires all four BOT_SECRET_PM_* credentials to be set",
        );
        let confirmed = secrets
            .live_confirm
            .as_ref()
            .is_some_and(|phrase| phrase.expose() == LIVE_CONFIRM_PHRASE);
        v.require(
            confirmed,
            "live.enabled",
            format!(
                "live mode requires {ENV_LIVE_CONFIRM} to contain the exact confirmation phrase"
            ),
        );
        // The default signature type is the deposit wallet (POLY_1271), whose
        // maker/signer is the funder address — live trading cannot construct an
        // order without it. A public address, so it lives in config (§live).
        match config.live.funder.as_deref() {
            None => v.require(
                false,
                "live.funder",
                "live mode requires live.funder (the deposit/proxy wallet address) to be set",
            ),
            Some(addr) => v.require(
                is_eth_address(addr),
                "live.funder",
                format!("live.funder {addr:?} must be a 0x-prefixed 40-hex-character address"),
            ),
        }
    }

    v.into_result().map_err(ConfigError::Invalid)
}

/// True for a `0x`-prefixed 40-hex-character Ethereum address (the shape of a
/// funder/deposit-wallet address).
fn is_eth_address(s: &str) -> bool {
    s.len() == 42 && s.starts_with("0x") && s[2..].bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use core_types::{Decimal, Dollars, DurationMs};
    use rust_decimal::dec;

    use super::*;
    use crate::secrets::SecretString;

    #[test]
    fn default_config_with_no_secrets_is_valid() {
        assert!(validate(&AppConfig::default(), &Secrets::default()).is_ok());
    }

    #[test]
    fn collects_all_violations_in_one_pass() {
        let mut config = AppConfig::default();
        config.paper.starting_capital = Dollars::new(Decimal::from(-5));
        config.risk.feed_staleness_ms = DurationMs::from_millis(600);
        config.engine.defaults.min_edge = dec!(0.005);

        let err = validate(&config, &Secrets::default()).unwrap_err();
        let ConfigError::Invalid(violations) = err else {
            panic!("expected Invalid, got {err:?}");
        };
        // min_edge is reported once per series (6) + capital + staleness.
        assert_eq!(violations.len(), 8);
        let keys: Vec<&str> = violations.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"paper.starting_capital"));
        assert!(keys.contains(&"risk.feed_staleness_ms"));
        assert!(keys.contains(&"engine.series.ETH-1h.min_edge"));
    }

    #[test]
    fn non_loopback_dashboard_requires_token() {
        let mut config = AppConfig::default();
        config.dashboard.bind = "0.0.0.0:8080".parse().unwrap();

        let err = validate(&config, &Secrets::default()).unwrap_err();
        assert!(err.to_string().contains(ENV_DASHBOARD_TOKEN));

        let secrets = Secrets {
            dashboard_token: Some(SecretString::new("a-token".to_owned())),
            ..Secrets::default()
        };
        assert!(validate(&config, &secrets).is_ok());
    }

    #[test]
    fn live_enabled_requires_credentials_and_exact_phrase() {
        let mut config = AppConfig::default();
        config.live.enabled = true;

        // Nothing set: both live violations fire.
        let err = validate(&config, &Secrets::default()).unwrap_err();
        let ConfigError::Invalid(violations) = err else {
            panic!("expected Invalid");
        };
        assert_eq!(
            violations
                .iter()
                .filter(|x| x.key == "live.enabled")
                .count(),
            2
        );

        // Credentials but a wrong phrase: still invalid.
        let secret = |s: &str| Some(SecretString::new(s.to_owned()));
        let mut secrets = Secrets {
            pm_api_key: secret("k"),
            pm_api_secret: secret("s"),
            pm_api_passphrase: secret("p"),
            pm_private_key: secret("pk"),
            live_confirm: secret("wrong phrase"),
            ..Secrets::default()
        };
        assert!(validate(&config, &secrets).is_err());

        // Exact phrase + a valid funder: valid.
        secrets.live_confirm = secret(LIVE_CONFIRM_PHRASE);
        config.live.funder = Some(format!("0x{}", "ab".repeat(20)));
        assert!(validate(&config, &secrets).is_ok());
    }

    #[test]
    fn live_enabled_requires_a_well_formed_funder() {
        let secret = |s: &str| Some(SecretString::new(s.to_owned()));
        let armed_secrets = Secrets {
            pm_api_key: secret("k"),
            pm_api_secret: secret("s"),
            pm_api_passphrase: secret("p"),
            pm_private_key: secret("pk"),
            live_confirm: secret(LIVE_CONFIRM_PHRASE),
            ..Secrets::default()
        };

        // Armed but no funder: a single live.funder violation.
        let mut config = AppConfig::default();
        config.live.enabled = true;
        let err = validate(&config, &armed_secrets).unwrap_err();
        let ConfigError::Invalid(violations) = err else {
            panic!("expected Invalid");
        };
        assert_eq!(
            violations.iter().filter(|x| x.key == "live.funder").count(),
            1
        );

        // Malformed funder (too short / non-hex / no 0x): still a violation.
        let bad_funders = [
            "0x1234".to_owned(),              // too short
            "ab".repeat(21),                  // no 0x prefix, wrong length
            format!("0x{}", "zz".repeat(20)), // right length, non-hex
        ];
        for bad in bad_funders {
            config.live.funder = Some(bad);
            assert!(validate(&config, &armed_secrets).is_err());
        }

        // Well-formed funder: valid.
        config.live.funder = Some(format!("0x{}", "Ab".repeat(20)));
        assert!(validate(&config, &armed_secrets).is_ok());
    }
}
