//! Secret handling: values that must never appear in files, logs, or
//! serialized output.
//!
//! Secrets are read **only** from environment variables ([`Secrets::from_env`])
//! — never from config files (the loader actively rejects secret keys found in
//! TOML, see [`crate::load()`]). [`SecretString`] redacts itself in `Debug` and
//! `Display`, and deliberately implements **neither** `Serialize` nor
//! `Deserialize`: a secret can never ride along in the serialized effective
//! config the bot logs at boot, and can never be sourced from a file layer —
//! both enforced at compile time.

use std::fmt;

/// Environment variable holding the dashboard auth token.
pub const ENV_DASHBOARD_TOKEN: &str = "BOT_SECRET_DASHBOARD_TOKEN";
/// Environment variable holding the Polymarket API key.
pub const ENV_PM_API_KEY: &str = "BOT_SECRET_PM_API_KEY";
/// Environment variable holding the Polymarket API secret.
pub const ENV_PM_API_SECRET: &str = "BOT_SECRET_PM_API_SECRET";
/// Environment variable holding the Polymarket API passphrase.
pub const ENV_PM_API_PASSPHRASE: &str = "BOT_SECRET_PM_API_PASSPHRASE";
/// Environment variable holding the wallet private key used for order signing.
pub const ENV_PM_PRIVATE_KEY: &str = "BOT_SECRET_PM_PRIVATE_KEY";
/// Environment variable that must contain the live-arming confirmation phrase
/// ([`crate::LIVE_CONFIRM_PHRASE`]) — gate 2 of the three §11 arming gates.
pub const ENV_LIVE_CONFIRM: &str = "BOT_SECRET_LIVE_CONFIRM";

/// A secret value. Redacted in `Debug`/`Display`; not serializable; the raw
/// value is reachable only through the explicit [`SecretString::expose`] call.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wraps an already-obtained secret value.
    #[must_use]
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the raw secret. Call sites of this method are the complete
    /// audit surface for secret usage — keep them few and obvious, and never
    /// pass the result to anything that logs or serializes.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

/// All secrets the bot knows about, each `None` when its environment variable
/// is unset or empty. Deriving `Debug` is safe: the inner type redacts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Secrets {
    /// Dashboard auth token ([`ENV_DASHBOARD_TOKEN`]). Required when the
    /// dashboard binds a non-loopback address.
    pub dashboard_token: Option<SecretString>,
    /// Polymarket API key ([`ENV_PM_API_KEY`]). Live trading only.
    pub pm_api_key: Option<SecretString>,
    /// Polymarket API secret ([`ENV_PM_API_SECRET`]). Live trading only.
    pub pm_api_secret: Option<SecretString>,
    /// Polymarket API passphrase ([`ENV_PM_API_PASSPHRASE`]). Live trading only.
    pub pm_api_passphrase: Option<SecretString>,
    /// Wallet private key ([`ENV_PM_PRIVATE_KEY`]). Live trading only.
    pub pm_private_key: Option<SecretString>,
    /// Live-arming confirmation phrase ([`ENV_LIVE_CONFIRM`]); must equal
    /// [`crate::LIVE_CONFIRM_PHRASE`] exactly for gate 2 to pass.
    pub live_confirm: Option<SecretString>,
}

impl Secrets {
    /// Reads every secret from the process environment. Unset **or empty**
    /// variables yield `None` — an empty secret is treated as absent so that
    /// `BOT_SECRET_X=""` can never satisfy a presence check.
    #[must_use]
    pub fn from_env() -> Self {
        fn read(name: &str) -> Option<SecretString> {
            std::env::var(name)
                .ok()
                .filter(|v| !v.is_empty())
                .map(SecretString::new)
        }
        Self {
            dashboard_token: read(ENV_DASHBOARD_TOKEN),
            pm_api_key: read(ENV_PM_API_KEY),
            pm_api_secret: read(ENV_PM_API_SECRET),
            pm_api_passphrase: read(ENV_PM_API_PASSPHRASE),
            pm_private_key: read(ENV_PM_PRIVATE_KEY),
            live_confirm: read(ENV_LIVE_CONFIRM),
        }
    }

    /// Presence map for boot logging: which secrets are set, never their
    /// values. Safe to log and to serialize.
    #[must_use]
    pub fn presence(&self) -> Vec<(&'static str, bool)> {
        vec![
            (ENV_DASHBOARD_TOKEN, self.dashboard_token.is_some()),
            (ENV_PM_API_KEY, self.pm_api_key.is_some()),
            (ENV_PM_API_SECRET, self.pm_api_secret.is_some()),
            (ENV_PM_API_PASSPHRASE, self.pm_api_passphrase.is_some()),
            (ENV_PM_PRIVATE_KEY, self.pm_private_key.is_some()),
            (ENV_LIVE_CONFIRM, self.live_confirm.is_some()),
        ]
    }

    /// True when all four Polymarket API credentials are present (the set
    /// live trading needs before the venue adapter can even be constructed).
    #[must_use]
    pub fn has_all_pm_credentials(&self) -> bool {
        self.pm_api_key.is_some()
            && self.pm_api_secret.is_some()
            && self.pm_api_passphrase.is_some()
            && self.pm_private_key.is_some()
    }
}

#[cfg(test)]
mod tests {
    // figment::Jail's closure contract returns figment::Error (208 bytes) by
    // value; the size is the library's choice, not ours, and this is test code.
    #![allow(clippy::result_large_err)]

    use super::*;

    #[test]
    fn debug_and_display_redact() {
        let secret = SecretString::new("hunter2-very-secret".to_owned());
        assert_eq!(format!("{secret:?}"), "[REDACTED]");
        assert_eq!(format!("{secret}"), "[REDACTED]");
        assert_eq!(secret.expose(), "hunter2-very-secret");
    }

    #[test]
    fn secrets_struct_debug_redacts() {
        let secrets = Secrets {
            dashboard_token: Some(SecretString::new("token-value-123".to_owned())),
            ..Secrets::default()
        };
        let dump = format!("{secrets:?}");
        assert!(!dump.contains("token-value-123"));
        assert!(dump.contains("[REDACTED]"));
    }

    #[test]
    fn from_env_treats_empty_as_absent() {
        // Jail serializes env access across tests and restores state after.
        figment::Jail::expect_with(|jail| {
            jail.set_env(ENV_DASHBOARD_TOKEN, "");
            jail.set_env(ENV_PM_API_KEY, "k");
            let secrets = Secrets::from_env();
            assert!(secrets.dashboard_token.is_none());
            assert!(secrets.pm_api_key.is_some());
            assert!(!secrets.has_all_pm_credentials());
            Ok(())
        });
    }

    #[test]
    fn presence_reports_flags_only() {
        let secrets = Secrets {
            pm_api_key: Some(SecretString::new("abc".to_owned())),
            ..Secrets::default()
        };
        let presence = secrets.presence();
        assert_eq!(presence.len(), 6);
        assert!(presence.contains(&(ENV_PM_API_KEY, true)));
        assert!(presence.contains(&(ENV_DASHBOARD_TOKEN, false)));
    }
}
