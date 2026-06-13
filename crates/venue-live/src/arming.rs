//! The fail-closed live-arming gate (CLAUDE.md §11).
//!
//! [`check_arming`] evaluates all three gates plus credential presence in order,
//! and is the first thing `LiveVenue::connect` does — so a refusal returns
//! before any SDK client is built or any network call is made.

use config::{LiveArming, Secrets};

use crate::error::{Gate, VenueLiveError};

/// Evaluates the §11 arming gates. Returns `Ok(())` only when all pass:
/// 1. boot gates (config `live.enabled` + env confirmation phrase),
/// 2. the dashboard arm flag for this session,
/// 3. all four Polymarket API credentials present.
///
/// # Errors
/// [`VenueLiveError::NotArmed`] for a closed gate, or
/// [`VenueLiveError::MissingCredentials`].
pub(crate) fn check_arming(
    arming: LiveArming,
    dashboard_armed: bool,
    secrets: &Secrets,
) -> Result<(), VenueLiveError> {
    if !arming.boot_gates_pass() {
        return Err(VenueLiveError::NotArmed(Gate::Boot));
    }
    if !dashboard_armed {
        return Err(VenueLiveError::NotArmed(Gate::Dashboard));
    }
    if !secrets.has_all_pm_credentials() {
        return Err(VenueLiveError::MissingCredentials);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use config::{LIVE_CONFIRM_PHRASE, LiveConfig, SecretString};

    use super::*;

    fn armed_secrets() -> Secrets {
        let s = |v: &str| Some(SecretString::new(v.to_owned()));
        Secrets {
            pm_api_key: s("k"),
            pm_api_secret: s("s"),
            pm_api_passphrase: s("p"),
            pm_private_key: s("pk"),
            live_confirm: s(LIVE_CONFIRM_PHRASE),
            ..Secrets::default()
        }
    }

    fn boot_open() -> LiveArming {
        LiveArming::evaluate(
            &LiveConfig {
                enabled: true,
                ..LiveConfig::default()
            },
            &armed_secrets(),
        )
    }

    #[test]
    fn closed_boot_gates_refuse() {
        let arming = LiveArming::evaluate(&LiveConfig::default(), &Secrets::default());
        assert_eq!(
            check_arming(arming, true, &armed_secrets()),
            Err(VenueLiveError::NotArmed(Gate::Boot))
        );
    }

    #[test]
    fn closed_dashboard_gate_refuses() {
        assert_eq!(
            check_arming(boot_open(), false, &armed_secrets()),
            Err(VenueLiveError::NotArmed(Gate::Dashboard))
        );
    }

    #[test]
    fn missing_credentials_refuse_even_when_armed() {
        // Boot gates pass on the phrase, dashboard armed, but creds absent.
        let secrets = Secrets {
            live_confirm: Some(SecretString::new(LIVE_CONFIRM_PHRASE.to_owned())),
            ..Secrets::default()
        };
        let arming = LiveArming::evaluate(
            &LiveConfig {
                enabled: true,
                ..LiveConfig::default()
            },
            &secrets,
        );
        assert_eq!(
            check_arming(arming, true, &secrets),
            Err(VenueLiveError::MissingCredentials)
        );
    }

    #[test]
    fn all_gates_open_passes() {
        assert_eq!(check_arming(boot_open(), true, &armed_secrets()), Ok(()));
    }
}
