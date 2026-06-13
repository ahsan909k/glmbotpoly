//! CLOB REST wire types (`https://clob.polymarket.com`).
//!
//! Used as a cross-check source only. WARNING (verified live 2026-06-11):
//! `end_date_iso` is date-truncated (`…T00:00:00Z` for an intraday close) —
//! Gamma is the timing authority and the field is deliberately not mapped.

use rust_decimal::Decimal;
use serde::Deserialize;

/// `GET {clob_rest}/markets/{condition_id}` (no auth).
#[derive(Debug, Clone, Deserialize)]
pub struct ClobMarket {
    /// CTF condition id, echoed back.
    #[serde(default)]
    pub condition_id: Option<String>,
    /// Venue minimum resting order size in shares.
    #[serde(default)]
    pub minimum_order_size: Option<Decimal>,
    /// Current tick size.
    #[serde(default)]
    pub minimum_tick_size: Option<Decimal>,
    /// Market neg-risk flag.
    #[serde(default)]
    pub neg_risk: Option<bool>,
    /// Whether the CLOB currently accepts orders.
    #[serde(default)]
    pub accepting_orders: Option<bool>,
    /// Outcome tokens with their labels.
    #[serde(default)]
    pub tokens: Vec<ClobToken>,
}

/// One outcome token of a CLOB market.
#[derive(Debug, Clone, Deserialize)]
pub struct ClobToken {
    /// The CLOB token id (decimal string).
    #[serde(default)]
    pub token_id: Option<String>,
    /// Outcome label, `"Up"` / `"Down"` for these markets.
    #[serde(default)]
    pub outcome: Option<String>,
}
