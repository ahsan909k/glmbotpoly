//! Live-trading arming state (CLAUDE.md §11 — safety invariant, not a feature).
//!
//! Live orders require **three** independent gates:
//!
//! 1. `live.enabled = true` in the config file (this section),
//! 2. the environment variable [`crate::ENV_LIVE_CONFIRM`] containing
//!    exactly [`LIVE_CONFIRM_PHRASE`],
//! 3. the operator arming live mode in the dashboard for the session
//!    (runtime state, owned by a later task).
//!
//! Absent any one of them, the live venue adapter must refuse to construct.
//! The confirmation phrase is a hardcoded constant on purpose: making it
//! configurable would let a config file weaken gate 2.

use serde::{Deserialize, Serialize};

use crate::secrets::Secrets;

/// The exact phrase [`crate::ENV_LIVE_CONFIRM`] must contain for gate 2 to
/// pass. Hardcoded; never configurable.
pub const LIVE_CONFIRM_PHRASE: &str = "arm-live-i-accept-real-money-losses";

/// Live-mode configuration. Defaults fully disarmed; nothing in this struct
/// may ever default toward live trading.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LiveConfig {
    /// Gate 1: live trading enabled in config. Default `false`.
    pub enabled: bool,
    /// The operator's funder address — the on-chain deposit/proxy wallet that
    /// holds collateral and is the `maker`/`signer` for POLY_1271 orders. A
    /// public address, **not a secret**, so it lives in config (auditable)
    /// rather than the environment. `None` when unset. Required when
    /// `enabled` is `true` (the default signature type is the deposit wallet);
    /// validated to be a `0x`-prefixed 40-hex-char Ethereum address.
    #[serde(default)]
    pub funder: Option<String>,
}

/// Evaluated state of the two boot-time arming gates (gate 3 — the dashboard
/// arm action — is runtime session state and lives with the control plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveArming {
    /// Gate 1: `live.enabled` was set in config.
    pub config_enabled: bool,
    /// Gate 2: the confirmation env var matched [`LIVE_CONFIRM_PHRASE`].
    pub env_confirmed: bool,
}

impl LiveArming {
    /// Evaluates gates 1 and 2 from the loaded config and environment secrets.
    #[must_use]
    pub fn evaluate(config: &LiveConfig, secrets: &Secrets) -> Self {
        let env_confirmed = secrets
            .live_confirm
            .as_ref()
            .is_some_and(|phrase| phrase.expose() == LIVE_CONFIRM_PHRASE);
        Self {
            config_enabled: config.enabled,
            env_confirmed,
        }
    }

    /// True when both boot-time gates pass. The venue adapter additionally
    /// requires gate 3 (dashboard arm) before any order can exist.
    #[must_use]
    pub fn boot_gates_pass(self) -> bool {
        self.config_enabled && self.env_confirmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::SecretString;

    #[test]
    fn default_is_fully_disarmed() {
        let arming = LiveArming::evaluate(&LiveConfig::default(), &Secrets::default());
        assert!(!arming.config_enabled);
        assert!(!arming.env_confirmed);
        assert!(!arming.boot_gates_pass());
    }

    #[test]
    fn wrong_phrase_fails_gate_two() {
        let secrets = Secrets {
            live_confirm: Some(SecretString::new("yes please".to_owned())),
            ..Secrets::default()
        };
        let arming = LiveArming::evaluate(
            &LiveConfig {
                enabled: true,
                ..LiveConfig::default()
            },
            &secrets,
        );
        assert!(arming.config_enabled);
        assert!(!arming.env_confirmed);
        assert!(!arming.boot_gates_pass());
    }

    #[test]
    fn both_gates_pass_only_with_exact_phrase_and_flag() {
        let secrets = Secrets {
            live_confirm: Some(SecretString::new(LIVE_CONFIRM_PHRASE.to_owned())),
            ..Secrets::default()
        };
        let enabled = LiveConfig {
            enabled: true,
            ..LiveConfig::default()
        };
        assert!(LiveArming::evaluate(&enabled, &secrets).boot_gates_pass());
        // Phrase alone is not enough: gate 1 still closed.
        assert!(!LiveArming::evaluate(&LiveConfig::default(), &secrets).boot_gates_pass());
    }
}
