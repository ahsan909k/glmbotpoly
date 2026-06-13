//! The trading universe: assets, window durations, and the six series.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::time::DurationMs;

/// An underlying crypto asset we trade Up/Down markets on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Asset {
    /// Bitcoin (BTC/USD underlying).
    Btc,
    /// Ether (ETH/USD underlying).
    Eth,
}

impl Asset {
    /// Both supported assets, in stable display order.
    pub const ALL: [Self; 2] = [Self::Btc, Self::Eth];

    /// Upper-case ticker, e.g. `"BTC"`.
    #[must_use]
    pub const fn ticker(self) -> &'static str {
        match self {
            Self::Btc => "BTC",
            Self::Eth => "ETH",
        }
    }
}

impl fmt::Display for Asset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.ticker())
    }
}

/// The length of one Up/Down window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WindowDuration {
    /// Five-minute windows.
    M5,
    /// Fifteen-minute windows.
    M15,
    /// One-hour windows.
    H1,
}

impl WindowDuration {
    /// All supported durations, shortest first.
    pub const ALL: [Self; 3] = [Self::M5, Self::M15, Self::H1];

    /// The window length as a duration.
    #[must_use]
    pub const fn as_duration(self) -> DurationMs {
        match self {
            Self::M5 => DurationMs::from_millis(5 * 60 * 1000),
            Self::M15 => DurationMs::from_millis(15 * 60 * 1000),
            Self::H1 => DurationMs::from_millis(60 * 60 * 1000),
        }
    }

    /// Short label, e.g. `"5m"`, `"15m"`, `"1h"`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::M5 => "5m",
            Self::M15 => "15m",
            Self::H1 => "1h",
        }
    }
}

impl fmt::Display for WindowDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One repeating Up/Down product: an asset × a window duration.
///
/// Serializes as its stable key (e.g. `"BTC-5m"`), which is also the
/// config/journal/dashboard identifier for the series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Series {
    /// Underlying asset.
    pub asset: Asset,
    /// Window length.
    pub duration: WindowDuration,
}

impl Series {
    /// All six series in stable dashboard order
    /// (BTC-5m, BTC-15m, BTC-1h, ETH-5m, ETH-15m, ETH-1h).
    pub const ALL: [Self; 6] = [
        Self {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        Self {
            asset: Asset::Btc,
            duration: WindowDuration::M15,
        },
        Self {
            asset: Asset::Btc,
            duration: WindowDuration::H1,
        },
        Self {
            asset: Asset::Eth,
            duration: WindowDuration::M5,
        },
        Self {
            asset: Asset::Eth,
            duration: WindowDuration::M15,
        },
        Self {
            asset: Asset::Eth,
            duration: WindowDuration::H1,
        },
    ];

    /// The stable string key, e.g. `"BTC-5m"`. Zero-allocation const lookup.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match (self.asset, self.duration) {
            (Asset::Btc, WindowDuration::M5) => "BTC-5m",
            (Asset::Btc, WindowDuration::M15) => "BTC-15m",
            (Asset::Btc, WindowDuration::H1) => "BTC-1h",
            (Asset::Eth, WindowDuration::M5) => "ETH-5m",
            (Asset::Eth, WindowDuration::M15) => "ETH-15m",
            (Asset::Eth, WindowDuration::H1) => "ETH-1h",
        }
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

/// Error parsing a series key.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown series key {0:?} (expected e.g. \"BTC-5m\")")]
pub struct ParseSeriesError(pub String);

impl FromStr for Series {
    type Err = ParseSeriesError;

    /// Parses a stable key like `"BTC-5m"`. Case-sensitive: keys are written
    /// by [`Series::key`] and must round-trip exactly.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|series| series.key() == s)
            .ok_or_else(|| ParseSeriesError(s.to_owned()))
    }
}

impl TryFrom<String> for Series {
    type Error = ParseSeriesError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

impl From<Series> for String {
    fn from(s: Series) -> Self {
        s.key().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn all_six_unique_and_roundtrip() {
        assert_eq!(Series::ALL.len(), 6);
        let keys: HashSet<&str> = Series::ALL.iter().map(|s| s.key()).collect();
        assert_eq!(keys.len(), 6);
        for series in Series::ALL {
            let parsed: Series = series.key().parse().unwrap();
            assert_eq!(parsed, series);
            assert_eq!(series.to_string(), series.key());
        }
    }

    #[test]
    fn parse_rejects_bad_keys() {
        assert!("btc-5m".parse::<Series>().is_err());
        assert!("BTC-5M".parse::<Series>().is_err());
        assert!("XRP-5m".parse::<Series>().is_err());
        assert!("BTC-2m".parse::<Series>().is_err());
        assert!("".parse::<Series>().is_err());
    }

    #[test]
    fn serde_uses_key_string() {
        let s = Series {
            asset: Asset::Eth,
            duration: WindowDuration::H1,
        };
        assert_eq!(serde_json::to_string(&s).unwrap(), "\"ETH-1h\"");
        let back: Series = serde_json::from_str("\"ETH-1h\"").unwrap();
        assert_eq!(back, s);
        assert!(serde_json::from_str::<Series>("\"eth-1h\"").is_err());
    }

    #[test]
    fn durations_are_correct() {
        assert_eq!(WindowDuration::M5.as_duration(), DurationMs::from_secs(300));
        assert_eq!(
            WindowDuration::M15.as_duration(),
            DurationMs::from_secs(900)
        );
        assert_eq!(
            WindowDuration::H1.as_duration(),
            DurationMs::from_secs(3600)
        );
    }
}
