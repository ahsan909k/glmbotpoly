//! Paper vs Live mode. Everything defaults to paper; live is gated by the
//! §11 arming flow — this enum only names the modes, it grants nothing.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Execution mode of a trading session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Simulated money against real live market data (the default).
    Paper,
    /// Real orders, real money — requires the full §11 arming flow.
    Live,
}

impl Mode {
    /// True only for [`Mode::Live`].
    #[must_use]
    pub const fn is_live(self) -> bool {
        matches!(self, Self::Live)
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Paper => "paper",
            Self::Live => "live",
        })
    }
}

/// Error parsing a [`Mode`] string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown mode {0:?} (expected \"paper\" or \"live\")")]
pub struct ParseModeError(pub String);

impl FromStr for Mode {
    type Err = ParseModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "paper" => Ok(Self::Paper),
            "live" => Ok(Self::Live),
            other => Err(ParseModeError(other.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_parse_roundtrip() {
        for mode in [Mode::Paper, Mode::Live] {
            assert_eq!(mode.to_string().parse::<Mode>().unwrap(), mode);
        }
        assert!("Live".parse::<Mode>().is_err()); // case-sensitive on purpose
        assert!("".parse::<Mode>().is_err());
    }

    #[test]
    fn serde_lowercase() {
        assert_eq!(serde_json::to_string(&Mode::Paper).unwrap(), "\"paper\"");
        assert_eq!(
            serde_json::from_str::<Mode>("\"live\"").unwrap(),
            Mode::Live
        );
    }

    #[test]
    fn is_live() {
        assert!(Mode::Live.is_live());
        assert!(!Mode::Paper.is_live());
    }
}
