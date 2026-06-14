//! Non-secret live network parameters.
//!
//! Host / chain id / signature type are code defaults (rarely changed); the
//! operator's funder address is sourced from config (`[live].funder`) at the
//! bot boundary and threaded in here. Secrets (the private key, API creds) are
//! never in this struct — they come from [`config::Secrets`] at `connect` time.

use std::time::Duration;

use core_types::Decimal;
use feed_util::BackoffParams;

use crate::error::VenueLiveError;

/// Default CLOB REST host. CLAUDE.md §7 names `clob.polymarket.com`; the v2 SDK
/// examples use `clob-v2.polymarket.com`. Defaulted to the §7 host; the
/// operator overrides via [`LiveParams`] if the v2 endpoint differs (verify
/// live before funding — see the Decisions Log).
pub const DEFAULT_CLOB_HOST: &str = "https://clob.polymarket.com";

/// Default authenticated user-channel WebSocket URL (CLAUDE.md §7).
pub const DEFAULT_USER_WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/user";

/// Polygon mainnet chain id (the SDK validates the signer carries this).
pub const POLYGON_CHAIN_ID: u64 = 137;

/// Default bound capacity of the order/fill event channel.
pub const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 1024;

/// The default user-channel keepalive PING interval. The docs require ≤10 s; we
/// halve it to 5 s (project convention, matching feed-clob) to never graze the
/// deadline.
pub const DEFAULT_WS_PING_INTERVAL: Duration = Duration::from_secs(5);

/// Default user-channel reconnect backoff (equal-jitter; feed-crate defaults).
pub const DEFAULT_WS_BACKOFF: BackoffParams = BackoffParams {
    initial: Duration::from_millis(250),
    max: Duration::from_secs(10),
    multiplier: 2.0,
};

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
///
/// Not `Eq` because [`ws_backoff`](Self::ws_backoff)'s multiplier is an `f64`.
#[derive(Debug, Clone, PartialEq)]
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
    /// Authenticated user-channel WebSocket URL.
    pub user_ws_url: String,
    /// Client keepalive PING cadence on the user channel (docs require ≤10 s).
    pub ws_ping_interval: Duration,
    /// A connection with no inbound frames for this long is torn down as
    /// half-open (defaults to `3 ×` the ping interval).
    pub ws_dead_after: Duration,
    /// Timeout for the user-channel connect (TCP+TLS+WS handshake).
    pub ws_connect_timeout: Duration,
    /// Reconnect backoff curve for the user channel.
    pub ws_backoff: BackoffParams,
    /// How often the user-channel task runs a belt-and-suspenders full REST
    /// reconcile while connected (in addition to the one on every (re)connect).
    pub safety_reconcile_interval: Duration,
    /// When the desired subscription set is empty, subscribe with `markets: []`
    /// (the user channel *may* then stream all of the operator's orders — an
    /// undocumented behaviour the REST reconcile backstops either way).
    pub subscribe_all_when_empty: bool,
    /// Emit a synthetic maker [`Fill`](core_types::Fill) for fills first observed
    /// via the REST reconcile (a reconnect recovered them). Exact for our
    /// post-only resting orders; turn off for OrderUpdate-only corrections.
    pub emit_synthetic_fills: bool,
    /// Fallback taker fee rate used when no per-market `FeeParams` is available
    /// through the [`WindowIndex`](crate::WindowIndex) (CLAUDE.md §7 — config
    /// default, never a hardcoded fee in logic).
    pub default_taker_fee_rate: Decimal,
}

impl Default for LiveParams {
    fn default() -> Self {
        Self {
            clob_host: DEFAULT_CLOB_HOST.to_owned(),
            chain_id: POLYGON_CHAIN_ID,
            sig_type: SigType::DepositWallet,
            funder: None,
            event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
            user_ws_url: DEFAULT_USER_WS_URL.to_owned(),
            ws_ping_interval: DEFAULT_WS_PING_INTERVAL,
            ws_dead_after: DEFAULT_WS_PING_INTERVAL.saturating_mul(3),
            ws_connect_timeout: Duration::from_secs(10),
            ws_backoff: DEFAULT_WS_BACKOFF,
            safety_reconcile_interval: Duration::from_secs(30),
            subscribe_all_when_empty: true,
            emit_synthetic_fills: true,
            default_taker_fee_rate: rust_decimal::dec!(0.07),
        }
    }
}

impl LiveParams {
    /// Fail-closed validation. A deposit/proxy signature type requires a
    /// well-formed funder address; the event channel must be non-empty; the
    /// user-channel timers must be sane.
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
        if self.ws_ping_interval.is_zero() {
            return Err(VenueLiveError::BadConfig(
                "ws_ping_interval must be > 0".to_owned(),
            ));
        }
        if self.ws_dead_after < self.ws_ping_interval {
            return Err(VenueLiveError::BadConfig(
                "ws_dead_after must be ≥ ws_ping_interval".to_owned(),
            ));
        }
        if self.ws_connect_timeout.is_zero() {
            return Err(VenueLiveError::BadConfig(
                "ws_connect_timeout must be > 0".to_owned(),
            ));
        }
        if self.safety_reconcile_interval.is_zero() {
            return Err(VenueLiveError::BadConfig(
                "safety_reconcile_interval must be > 0".to_owned(),
            ));
        }
        if self.ws_backoff.multiplier.is_nan() || self.ws_backoff.multiplier <= 1.0 {
            return Err(VenueLiveError::BadConfig(
                "ws_backoff.multiplier must be > 1.0".to_owned(),
            ));
        }
        if self.default_taker_fee_rate.is_sign_negative() {
            return Err(VenueLiveError::BadConfig(
                "default_taker_fee_rate must be ≥ 0".to_owned(),
            ));
        }
        if self.user_ws_url.trim().is_empty() {
            return Err(VenueLiveError::BadConfig(
                "user_ws_url must be non-empty".to_owned(),
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
