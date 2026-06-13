//! Non-secret live network parameters.
//!
//! Host / chain id / signature type are code defaults (rarely changed); the
//! operator's funder address is sourced from config (`[live].funder`) at the
//! bot boundary and threaded in here. Secrets (the private key, API creds) are
//! never in this struct — they come from [`config::Secrets`] at `connect` time.

use crate::error::VenueLiveError;

/// Default CLOB REST host. CLAUDE.md §7 names `clob.polymarket.com`; the v2 SDK
/// examples use `clob-v2.polymarket.com`. Defaulted to the §7 host; the
/// operator overrides via [`LiveParams`] if the v2 endpoint differs (verify
/// live before funding — see the Decisions Log).
pub const DEFAULT_CLOB_HOST: &str = "https://clob.polymarket.com";

/// Polygon mainnet chain id (the SDK validates the signer carries this).
pub const POLYGON_CHAIN_ID: u64 = 137;

/// Default bound capacity of the order/fill event channel.
pub const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 1024;

/// Order-signing scheme. Our enum; mapped to the SDK's `SignatureType` at the
/// single conversion site in the SDK backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigType {
    /// Externally-owned account — the signer wallet is itself the maker. No
    /// funder required. Simplest; used for testing and EOA accounts.
    Eoa,
    /// Deposit wallet (the SDK's `Poly1271`, signature type 3) — the path for
    /// new API users (CLAUDE.md §7). The funder (deposit-wallet address) is the
    /// maker/signer and is **required**.
    DepositWallet,
}

/// Non-secret network parameters for the live adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveParams {
    /// CLOB REST host.
    pub clob_host: String,
    /// EVM chain id (137 for Polygon mainnet).
    pub chain_id: u64,
    /// Order signature scheme.
    pub sig_type: SigType,
    /// Deposit/proxy wallet address (`0x` + 40 hex). Required for
    /// [`SigType::DepositWallet`]; ignored for [`SigType::Eoa`].
    pub funder: Option<String>,
    /// Bound capacity of the order/fill event channel.
    pub event_channel_capacity: usize,
}

impl Default for LiveParams {
    fn default() -> Self {
        Self {
            clob_host: DEFAULT_CLOB_HOST.to_owned(),
            chain_id: POLYGON_CHAIN_ID,
            sig_type: SigType::DepositWallet,
            funder: None,
            event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
        }
    }
}

impl LiveParams {
    /// Fail-closed validation. A deposit/proxy signature type requires a
    /// well-formed funder address; the event channel must be non-empty.
    ///
    /// # Errors
    /// [`VenueLiveError::MissingFunder`], [`VenueLiveError::BadFunder`], or
    /// [`VenueLiveError::BadConfig`].
    pub fn validate(&self) -> Result<(), VenueLiveError> {
        if self.event_channel_capacity == 0 {
            return Err(VenueLiveError::BadConfig(
                "event_channel_capacity must be > 0".to_owned(),
            ));
        }
        if self.chain_id == 0 {
            return Err(VenueLiveError::BadConfig(
                "chain_id must be non-zero".to_owned(),
            ));
        }
        match self.sig_type {
            SigType::Eoa => Ok(()),
            SigType::DepositWallet => match self.funder.as_deref() {
                None => Err(VenueLiveError::MissingFunder),
                Some(addr) if is_eth_address(addr) => Ok(()),
                Some(addr) => Err(VenueLiveError::BadFunder(addr.to_owned())),
            },
        }
    }
}

/// True for a `0x`-prefixed 40-hex-character Ethereum address.
fn is_eth_address(s: &str) -> bool {
    s.len() == 42 && s.starts_with("0x") && s[2..].bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr() -> String {
        format!("0x{}", "ab".repeat(20))
    }

    #[test]
    fn default_is_deposit_wallet_on_polygon() {
        let p = LiveParams::default();
        assert_eq!(p.chain_id, POLYGON_CHAIN_ID);
        assert_eq!(p.sig_type, SigType::DepositWallet);
        assert_eq!(p.clob_host, DEFAULT_CLOB_HOST);
        // Default has no funder, so the default params are NOT valid for live —
        // the operator must supply one.
        assert!(matches!(p.validate(), Err(VenueLiveError::MissingFunder)));
    }

    #[test]
    fn deposit_wallet_requires_well_formed_funder() {
        let mut p = LiveParams {
            funder: Some(addr()),
            ..LiveParams::default()
        };
        assert!(p.validate().is_ok());

        for bad in [
            "0x1234",
            &"ab".repeat(21),
            &format!("0x{}", "zz".repeat(20)),
        ] {
            p.funder = Some((*bad).to_owned());
            assert!(matches!(p.validate(), Err(VenueLiveError::BadFunder(_))));
        }
    }

    #[test]
    fn eoa_ignores_funder() {
        let p = LiveParams {
            sig_type: SigType::Eoa,
            funder: None,
            ..LiveParams::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn zero_capacity_is_rejected() {
        let p = LiveParams {
            sig_type: SigType::Eoa,
            event_channel_capacity: 0,
            ..LiveParams::default()
        };
        assert!(matches!(p.validate(), Err(VenueLiveError::BadConfig(_))));
    }
}
