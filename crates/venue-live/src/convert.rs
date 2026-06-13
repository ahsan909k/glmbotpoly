//! Translating a normalized [`core_types::NewOrder`] into a [`BuiltOrder`] —
//! the venue-live-internal, pre-signing representation the SDK backend turns
//! into builder calls.
//!
//! This is where the GTD 60-second security threshold (CLAUDE.md §7) is applied
//! and where the marketable BUY-dollars / SELL-shares convention is enforced.
//! It is pure and clock-injected (`now` is a parameter), so every translation
//! is unit-testable without the SDK.

use core_types::{DurationMs, NewOrder, OrderQty, Price, Side, TimeInForce, TimestampMs, TokenId};

use crate::error::VenueLiveError;
use crate::params::{LiveParams, SigType};

/// The venue's GTD security threshold: a GTD expiration must be at least this
/// far in the future (CLAUDE.md §7).
const GTD_MIN_LEAD: DurationMs = DurationMs::from_secs(60);

/// Extra margin added on top of [`GTD_MIN_LEAD`] so the on-wire expiration never
/// grazes the threshold under request latency or clock skew.
const GTD_SAFETY_MARGIN: DurationMs = DurationMs::from_secs(5);

/// The resolved venue order class for a [`BuiltOrder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderClass {
    /// A resting limit order. `expiration` `None` is GTC; `Some` is GTD with the
    /// on-wire expiration already floored to satisfy the venue's 60s minimum.
    Limit {
        /// Reject instead of crossing.
        post_only: bool,
        /// Floored on-wire expiration, or `None` for GTC.
        expiration: Option<TimestampMs>,
    },
    /// A marketable order. `fok` `true` is FOK, `false` is FAK.
    Marketable {
        /// Fill-or-kill (`true`) vs fill-and-kill (`false`).
        fok: bool,
    },
}

/// A normalized order ready to hand to the SDK backend: everything needed to
/// build, sign, and post, but with no SDK types. Recorded verbatim by
/// [`FakeClobPort`](crate::FakeClobPort) so tests can assert request
/// construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltOrder {
    /// Client correlation id, echoed back in the ack.
    pub client_id: Option<String>,
    /// Outcome token traded.
    pub token_id: TokenId,
    /// Buy or sell.
    pub side: Side,
    /// Limit price (resting) or worst-price slippage limit (marketable).
    pub price: Price,
    /// Quantity — shares for resting/marketable-SELL, dollars for marketable-BUY.
    pub amount: OrderQty,
    /// Resolved venue order class.
    pub class: OrderClass,
    /// Signature scheme.
    pub sig_type: SigType,
    /// Funder address (deposit/proxy wallet); `None` for EOA.
    pub funder: Option<String>,
}

/// Builds a [`BuiltOrder`] from a normalized [`NewOrder`]. `now` is the current
/// wall-clock instant, used only for the GTD expiration floor.
///
/// # Errors
/// [`VenueLiveError::QtyKindMismatch`] when the order's quantity kind does not
/// match its side/class (a normalizer bug — the engine should never produce
/// one).
pub fn build(
    order: &NewOrder,
    now: TimestampMs,
    params: &LiveParams,
) -> Result<BuiltOrder, VenueLiveError> {
    let class = match order.tif {
        TimeInForce::Gtc { post_only } => {
            require_shares(order)?;
            OrderClass::Limit {
                post_only,
                expiration: None,
            }
        }
        TimeInForce::Gtd {
            expires_at,
            post_only,
        } => {
            require_shares(order)?;
            OrderClass::Limit {
                post_only,
                expiration: Some(floor_gtd_expiration(expires_at, now)),
            }
        }
        TimeInForce::Fok => {
            require_marketable_amount(order)?;
            OrderClass::Marketable { fok: true }
        }
        TimeInForce::Fak => {
            require_marketable_amount(order)?;
            OrderClass::Marketable { fok: false }
        }
    };

    Ok(BuiltOrder {
        client_id: order.client_id.clone(),
        token_id: order.token_id.clone(),
        side: order.side,
        price: order.price,
        amount: order.qty,
        class,
        sig_type: params.sig_type,
        funder: params.funder.clone(),
    })
}

/// Floors a GTD expiration to satisfy the venue's 60-second minimum: the on-wire
/// expiration is `max(expires_at, now + 60s + margin)`. When the engine's
/// desired expiration already clears the threshold it passes through unchanged.
fn floor_gtd_expiration(expires_at: TimestampMs, now: TimestampMs) -> TimestampMs {
    let min = now
        .saturating_add(GTD_MIN_LEAD)
        .saturating_add(GTD_SAFETY_MARGIN);
    if expires_at.as_millis() < min.as_millis() {
        min
    } else {
        expires_at
    }
}

/// Resting orders are always specified in shares.
fn require_shares(order: &NewOrder) -> Result<(), VenueLiveError> {
    match order.qty {
        OrderQty::Shares(_) => Ok(()),
        OrderQty::Notional(_) => Err(VenueLiveError::QtyKindMismatch(
            "resting orders must be a share count, got a dollar notional".to_owned(),
        )),
    }
}

/// Marketable BUY is a dollar amount; marketable SELL is a share count (§7).
fn require_marketable_amount(order: &NewOrder) -> Result<(), VenueLiveError> {
    match (order.side, order.qty) {
        (Side::Buy, OrderQty::Notional(_)) | (Side::Sell, OrderQty::Shares(_)) => Ok(()),
        (Side::Buy, OrderQty::Shares(_)) => Err(VenueLiveError::QtyKindMismatch(
            "marketable BUY must be a dollar notional".to_owned(),
        )),
        (Side::Sell, OrderQty::Notional(_)) => Err(VenueLiveError::QtyKindMismatch(
            "marketable SELL must be a share count".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use core_types::{Asset, Dollars, Outcome, Series, Size, TickSize, WindowDuration, WindowId};
    use rust_decimal::dec;

    use super::*;

    fn params() -> LiveParams {
        LiveParams {
            sig_type: SigType::DepositWallet,
            funder: Some(format!("0x{}", "ab".repeat(20))),
            ..LiveParams::default()
        }
    }

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_000_000),
        }
    }

    fn price(d: rust_decimal::Decimal) -> Price {
        Price::quantize(d, TickSize::T001, core_types::RoundDir::Down).unwrap()
    }

    fn order(side: Side, qty: OrderQty, tif: TimeInForce) -> NewOrder {
        NewOrder {
            client_id: Some("c-1".to_owned()),
            window: window(),
            token_id: TokenId::new("123").unwrap(),
            outcome: Outcome::Up,
            side,
            price: price(dec!(0.50)),
            qty,
            tif,
        }
    }

    #[test]
    fn gtc_post_only_builds_limit() {
        let o = order(
            Side::Buy,
            OrderQty::Shares(Size::new(dec!(10)).unwrap()),
            TimeInForce::Gtc { post_only: true },
        );
        let b = build(&o, TimestampMs::from_millis(0), &params()).unwrap();
        assert_eq!(
            b.class,
            OrderClass::Limit {
                post_only: true,
                expiration: None
            }
        );
        assert_eq!(b.funder, params().funder);
        assert_eq!(b.sig_type, SigType::DepositWallet);
    }

    #[test]
    fn gtd_expiration_floored_to_now_plus_65s_when_too_soon() {
        let now = TimestampMs::from_millis(1_000_000);
        // Desired expiry only 10s out — below the 60s threshold, so it is bumped.
        let o = order(
            Side::Sell,
            OrderQty::Shares(Size::new(dec!(7)).unwrap()),
            TimeInForce::Gtd {
                expires_at: now.saturating_add(DurationMs::from_secs(10)),
                post_only: true,
            },
        );
        let b = build(&o, now, &params()).unwrap();
        let OrderClass::Limit {
            expiration: Some(exp),
            ..
        } = b.class
        else {
            panic!("expected GTD limit");
        };
        assert_eq!(exp.as_millis(), now.as_millis() + 65_000);
    }

    #[test]
    fn gtd_expiration_passes_through_when_already_past_threshold() {
        let now = TimestampMs::from_millis(1_000_000);
        let desired = now.saturating_add(DurationMs::from_secs(300));
        let o = order(
            Side::Buy,
            OrderQty::Shares(Size::new(dec!(5)).unwrap()),
            TimeInForce::Gtd {
                expires_at: desired,
                post_only: true,
            },
        );
        let b = build(&o, now, &params()).unwrap();
        let OrderClass::Limit {
            expiration: Some(exp),
            ..
        } = b.class
        else {
            panic!("expected GTD limit");
        };
        assert_eq!(exp, desired);
    }

    #[test]
    fn marketable_buy_is_dollars_sell_is_shares() {
        let buy = order(
            Side::Buy,
            OrderQty::Notional(Dollars::new(dec!(25))),
            TimeInForce::Fak,
        );
        assert_eq!(
            build(&buy, TimestampMs::from_millis(0), &params())
                .unwrap()
                .class,
            OrderClass::Marketable { fok: false }
        );

        let sell = order(
            Side::Sell,
            OrderQty::Shares(Size::new(dec!(10)).unwrap()),
            TimeInForce::Fok,
        );
        assert_eq!(
            build(&sell, TimestampMs::from_millis(0), &params())
                .unwrap()
                .class,
            OrderClass::Marketable { fok: true }
        );
    }

    #[test]
    fn qty_kind_mismatches_are_rejected() {
        // Resting with a dollar notional.
        let bad = order(
            Side::Buy,
            OrderQty::Notional(Dollars::new(dec!(10))),
            TimeInForce::Gtc { post_only: true },
        );
        assert!(matches!(
            build(&bad, TimestampMs::from_millis(0), &params()),
            Err(VenueLiveError::QtyKindMismatch(_))
        ));

        // Marketable BUY with shares.
        let bad = order(
            Side::Buy,
            OrderQty::Shares(Size::new(dec!(10)).unwrap()),
            TimeInForce::Fak,
        );
        assert!(matches!(
            build(&bad, TimestampMs::from_millis(0), &params()),
            Err(VenueLiveError::QtyKindMismatch(_))
        ));

        // Marketable SELL with a notional.
        let bad = order(
            Side::Sell,
            OrderQty::Notional(Dollars::new(dec!(10))),
            TimeInForce::Fok,
        );
        assert!(matches!(
            build(&bad, TimestampMs::from_millis(0), &params()),
            Err(VenueLiveError::QtyKindMismatch(_))
        ));
    }
}
