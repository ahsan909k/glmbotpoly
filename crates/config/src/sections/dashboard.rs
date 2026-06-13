//! Dashboard server settings (CLAUDE.md §10).
//!
//! The auth token is deliberately **not** a field here: it is a secret and
//! comes only from the environment ([`crate::ENV_DASHBOARD_TOKEN`]).
//! Cross-validation in [`crate::validate()`] requires it whenever the bind
//! address is not loopback.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use serde::{Deserialize, Serialize};

/// Dashboard HTTP server settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DashboardConfig {
    /// Address the axum server binds. Parsed from a string like
    /// `"127.0.0.1:8080"`; a malformed address is a load-time error.
    pub bind: SocketAddr,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bind_is_loopback() {
        assert!(DashboardConfig::default().bind.ip().is_loopback());
        assert_eq!(DashboardConfig::default().bind.port(), 8080);
    }

    #[test]
    fn bind_deserializes_from_string() {
        let cfg: DashboardConfig = serde_json::from_str(r#"{ "bind": "0.0.0.0:9000" }"#).unwrap();
        assert!(!cfg.bind.ip().is_loopback());
        assert!(serde_json::from_str::<DashboardConfig>(r#"{ "bind": "not-an-addr" }"#).is_err());
    }
}
