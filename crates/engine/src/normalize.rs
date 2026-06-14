//! The order normalizer (CLAUDE.md §7, §8): the single pure function every
//! outgoing order intent passes through before it can reach a venue.
//!
//! It snaps the draft price to the market's *current* tick grid with
//! passive-safe rounding (resting bids round down / asks up, never toward
//! crossing; marketable slippage caps round outward so a worst-price limit is
//! never tightened into a no-fill), enforces the price range `[tick, 1 − tick]`,
//! enforces the strictest order-size minimum — `max(5 shares, shares to reach
//! $1 notional at the *rounded* price, the market's `minimum_order_size`)`,
//! recomputing the $1 check *after* price rounding — enforces size precision,
//! and verifies a post-only order would not cross our current view of the book.
//!
//! Output is either a venue-legal [`Normalized`] order or a typed
//! [`NormalizeReject`]. The normalizer never silently shrinks size below the
//! caller's intent; it is allowed to *bump up* to a minimum, and flags the bump
//! in [`Adjustments`] so the journal records it. It is sans-IO and has no clock:
//! the only fallible operations are mapped to rejects — no `unwrap`/`panic` on a
//! runtime path (CLAUDE.md §12).

use core_types::{
    Decimal, MarketInfo, NewOrder, OrderQty, Price, PriceError, RoundDir, Side, Size, TickSize,
    TopOfBook,
};
use rust_decimal::RoundingStrategy;

/// Venue minimum resting / marketable-SELL order size: 5 shares (CLAUDE.md §7).
const MIN_SHARES: Decimal = Decimal::from_parts(5, 0, 0, false, 0);

/// Venue minimum order notional: $1.00 (CLAUDE.md §7). The minimum share count
/// is `ceil_to_size_precision(MIN_NOTIONAL / rounded_price)`, recomputed *after*
/// price rounding.
const MIN_NOTIONAL: Decimal = Decimal::from_parts(1, 0, 0, false, 0);

/// A desired order *before* normalization.
///
/// Distinct from [`NewOrder`], whose `price` is an already-validated on-grid
/// [`Price`]: a draft carries the *target* price as a raw [`Decimal`] so the
/// normalizer can snap it onto the market's current grid, and so a garbage price
/// is a testable rejection rather than an unrepresentable state.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderDraft {
    /// Optional client-side idempotency/attribution tag (passed through).
    pub client_id: Option<String>,
    /// Window this order trades (passed through).
    pub window: core_types::WindowId,
    /// Outcome token being traded (passed through).
    pub token_id: core_types::TokenId,
    /// Which outcome `token_id` is (passed through).
    pub outcome: core_types::Outcome,
    /// Buy or sell.
    pub side: Side,
    /// Desired (pre-grid-snap) limit price for resting orders, or the desired
    /// worst-price slippage limit for marketable orders. Raw decimal: the
    /// normalizer snaps it onto [`MarketInfo::tick_size`].
    pub price: Decimal,
    /// Desired quantity. Must match the side/class convention (CLAUDE.md §7):
    /// resting & marketable-SELL = [`OrderQty::Shares`]; marketable-BUY =
    /// [`OrderQty::Notional`]. A mismatch is a [`NormalizeReject::QtyKindMismatch`].
    pub qty: OrderQty,
    /// Time-in-force / order class (passed through). Drives marketable-vs-resting
    /// price rounding and whether the post-only cross check runs.
    pub tif: core_types::TimeInForce,
}

/// Tunables the normalizer needs that are not on the market object.
///
/// Mapped at the binary boundary from config, the way `model` maps
/// `VolParams`/`FairParams`. [`MIN_SHARES`] and [`MIN_NOTIONAL`] are venue
/// constants, not knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NormalizerParams {
    /// Venue size precision in decimal places. Sizes round *down* to this many
    /// places (never up — that would oversize), and the min-notional ceil rounds
    /// *up* to it. [`MarketInfo`] carries no size-precision field today; the
    /// venue's true precision must be verified live (see the Decisions Log).
    /// Default `0` = whole shares.
    pub size_decimals: u32,
}

/// How the draft price was moved onto the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceSnap {
    /// The draft's raw target price.
    pub from: Decimal,
    /// The on-grid price actually used (`from == to.as_decimal()` when the draft
    /// was already on-grid and in range).
    pub to: Price,
    /// True when the `[tick, 1 − tick]` clamp bound — an extreme target was
    /// pulled into range rather than merely snapped to the grid.
    pub clamped_into_range: bool,
}

/// Non-rejecting, audit-worthy changes the normalizer made.
///
/// An order can be adjusted and still succeed; everything here is logged with
/// the placement. The size flags are `false` for a marketable-BUY (notional)
/// order, which carries no share count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adjustments {
    /// The price snap (always present).
    pub price: PriceSnap,
    /// The desired share size was rounded *down* to the size precision.
    pub size_rounded_down: bool,
    /// The size was *bumped up* to the strictest minimum (5 shares / $1 notional
    /// / market minimum). Explicitly flagged: bumping up is allowed, silently
    /// reducing below intent is not (CLAUDE.md §7).
    pub size_bumped_to_min: bool,
}

/// A normalized order plus the audit trail of what was adjusted.
#[derive(Debug, Clone, PartialEq)]
pub struct Normalized {
    /// The order to hand downstream — price on the current grid, size at venue
    /// precision, all minimums satisfied. Exactly what `venue-live::convert::build`
    /// expects.
    pub order: NewOrder,
    /// What the normalizer changed (price snap, size round-down, min bump).
    pub adjustments: Adjustments,
}

/// Why the normalizer refused a draft.
///
/// Engine-local and data-carrying: unlike the venue's flat, post-hoc
/// [`venue_api::RejectReason`](venue_api::RejectReason), this is a *pre-flight*
/// check that records the offending values for the journal. Project onto the
/// venue vocabulary with [`NormalizeReject::as_reject_reason`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NormalizeReject {
    /// Target price was outside the closed `[0, 1]` domain — there is no legal
    /// grid price to snap to.
    #[error("price {0} is outside the valid [0,1] domain")]
    PriceOutOfDomain(Decimal),

    /// The desired share size was *exactly* zero. We never place a zero order; a
    /// *positive* sub-minimum size is bumped up instead, so zero signals a caller
    /// bug.
    #[error("desired size is zero")]
    ZeroSize,

    /// The quantity kind does not match the side/class convention (CLAUDE.md §7):
    /// resting & marketable-SELL must be `Shares`, marketable-BUY must be
    /// `Notional`.
    #[error("quantity kind mismatch: {0}")]
    QtyKindMismatch(String),

    /// A marketable-BUY's dollar notional is below the $1.00 venue minimum.
    /// Rejected rather than bumped: raising spend changes risk/sizing, which is
    /// the risk manager's call, not the normalizer's.
    #[error("marketable BUY notional {got} below ${min} minimum")]
    BelowMinNotional {
        /// The submitted notional.
        got: Decimal,
        /// The venue minimum ([`MIN_NOTIONAL`]).
        min: Decimal,
    },

    /// A post-only order at the snapped price would cross our current view of the
    /// book — the venue would reject it, so we reject first (CLAUDE.md §7).
    #[error("post-only {side:?} at {price} would cross the book (touch {touch})")]
    WouldCross {
        /// The order side.
        side: Side,
        /// Our snapped resting price.
        price: Price,
        /// The opposing touch it would have crossed.
        touch: Price,
    },
}

impl NormalizeReject {
    /// Best-effort projection onto the venue's flat reject vocabulary, for the
    /// journal and the risk manager's classification.
    #[must_use]
    pub fn as_reject_reason(&self) -> venue_api::RejectReason {
        use venue_api::RejectReason as R;
        match self {
            Self::PriceOutOfDomain(_) => R::TickRule,
            Self::ZeroSize | Self::BelowMinNotional { .. } => R::BelowMinSize,
            Self::WouldCross { .. } => R::CrossedBook,
            Self::QtyKindMismatch(m) => R::Other(m.clone()),
        }
    }
}

/// Normalize one outgoing order draft against the market's current grid and
/// minimums.
///
/// Pure: no IO, no clock, no allocation beyond the returned values. `book` is
/// our current top-of-book view for this token; `None` (or a missing side) means
/// we cannot prove a post-only order would cross, so we allow it — the venue
/// makes the final call (CLAUDE.md §9 conservatism: we never *fabricate* a cross).
///
/// # Errors
/// [`NormalizeReject`] — see its variants.
pub fn normalize(
    draft: &OrderDraft,
    market: &MarketInfo,
    book: Option<&TopOfBook>,
    params: &NormalizerParams,
) -> Result<Normalized, NormalizeReject> {
    let marketable = draft.tif.is_marketable();

    // 1. Quantity-kind gate (mirrors venue-live::convert) — a draft that passes
    //    here can never trip the venue adapter's require_shares/_marketable_amount.
    check_qty_kind(draft.side, marketable, &draft.qty)?;

    // 2. Snap the price onto the current grid. Resting orders round passively
    //    (away from crossing); marketable slippage caps round outward.
    let dir = if marketable {
        RoundDir::passive_for(draft.side.opposite())
    } else {
        RoundDir::passive_for(draft.side)
    };
    let snapped = Price::quantize(draft.price, market.tick_size, dir).map_err(map_price_err)?;
    let price = PriceSnap {
        from: draft.price,
        to: snapped,
        clamped_into_range: price_was_clamped(draft.price, market.tick_size, dir),
    };

    // 3. Quantity: notional minimum (marketable BUY) or share minimum + precision.
    let (qty, size_rounded_down, size_bumped_to_min) = match &draft.qty {
        OrderQty::Notional(dollars) => {
            if dollars.as_decimal() < MIN_NOTIONAL {
                return Err(NormalizeReject::BelowMinNotional {
                    got: dollars.as_decimal(),
                    min: MIN_NOTIONAL,
                });
            }
            (OrderQty::Notional(*dollars), false, false)
        }
        OrderQty::Shares(size) => {
            let (final_size, rounded_down, bumped) =
                normalize_shares(*size, snapped, market, params)?;
            (OrderQty::Shares(final_size), rounded_down, bumped)
        }
    };

    // 4. Post-only orders must not cross our book view (resting classes only;
    //    marketable orders are never post-only). Checked last, so the WouldCross
    //    payload reflects the fully-sized order.
    if draft.tif.is_post_only() {
        check_no_cross(draft.side, snapped, book)?;
    }

    // 5. Assemble the venue-legal order.
    let order = NewOrder {
        client_id: draft.client_id.clone(),
        window: draft.window,
        token_id: draft.token_id.clone(),
        outcome: draft.outcome,
        side: draft.side,
        price: snapped,
        qty,
        tif: draft.tif,
    };
    Ok(Normalized {
        order,
        adjustments: Adjustments {
            price,
            size_rounded_down,
            size_bumped_to_min,
        },
    })
}

/// Enforces the §7 quantity-kind convention; reuses `venue-live::convert`'s
/// wording so the two layers never disagree.
fn check_qty_kind(side: Side, marketable: bool, qty: &OrderQty) -> Result<(), NormalizeReject> {
    // §7 convention: resting & marketable-SELL are share counts; marketable-BUY
    // is a dollar notional.
    let ok = matches!(
        (marketable, side, qty),
        (false, _, OrderQty::Shares(_))
            | (true, Side::Sell, OrderQty::Shares(_))
            | (true, Side::Buy, OrderQty::Notional(_))
    );
    if ok {
        return Ok(());
    }
    let msg = match (marketable, side) {
        (false, _) => "resting orders must be a share count, got a dollar notional",
        (true, Side::Buy) => "marketable BUY must be a dollar notional",
        (true, Side::Sell) => "marketable SELL must be a share count",
    };
    Err(NormalizeReject::QtyKindMismatch(msg.to_owned()))
}

/// Computes the final share size: rounds the desired size down to precision, then
/// raises it to the strictest minimum if needed (recomputing the $1 floor against
/// the *rounded* price). Returns `(final_size, size_rounded_down, size_bumped_to_min)`.
fn normalize_shares(
    desired: Size,
    price: Price,
    market: &MarketInfo,
    params: &NormalizerParams,
) -> Result<(Size, bool, bool), NormalizeReject> {
    // A literal-zero intent is a caller bug; a positive intent is never rejected
    // on size — it is bumped up to the floor below.
    if desired.is_zero() {
        return Err(NormalizeReject::ZeroSize);
    }
    let rp = price.as_decimal(); // strictly in (0,1) — division is safe.
    // Shares needed to reach $1 at the rounded price, ceiled to size precision.
    let notional_min = (MIN_NOTIONAL / rp)
        .round_dp_with_strategy(params.size_decimals, RoundingStrategy::ToPositiveInfinity);
    let floor = MIN_SHARES
        .max(notional_min)
        .max(market.min_order_size.as_decimal());

    let rounded = desired.round_down_dp(params.size_decimals);
    let size_rounded_down = rounded.as_decimal() != desired.as_decimal();

    if rounded.as_decimal() < floor {
        // `floor` is ≥ 5, so this never fails; map defensively rather than unwrap.
        let bumped = Size::new(floor).map_err(|_| NormalizeReject::ZeroSize)?;
        Ok((bumped, size_rounded_down, true))
    } else {
        Ok((rounded, size_rounded_down, false))
    }
}

/// Post-only cross check against our book view. A BUY crosses iff its price is at
/// or above the best ask; a SELL iff at or below the best bid. A missing side or
/// absent book cannot be crossed.
fn check_no_cross(
    side: Side,
    price: Price,
    book: Option<&TopOfBook>,
) -> Result<(), NormalizeReject> {
    let Some(top) = book else {
        return Ok(());
    };
    match side {
        Side::Buy => {
            if let Some(ask) = top.ask
                && price >= ask.price
            {
                return Err(NormalizeReject::WouldCross {
                    side,
                    price,
                    touch: ask.price,
                });
            }
        }
        Side::Sell => {
            if let Some(bid) = top.bid
                && price <= bid.price
            {
                return Err(NormalizeReject::WouldCross {
                    side,
                    price,
                    touch: bid.price,
                });
            }
        }
    }
    Ok(())
}

/// Maps a [`PriceError`] from [`Price::quantize`] onto a [`NormalizeReject`].
/// `quantize` only ever returns [`PriceError::OutOfDomain`]; the other variants
/// are structurally unreachable here and mapped defensively (never panic).
fn map_price_err(err: PriceError) -> NormalizeReject {
    match err {
        PriceError::OutOfDomain(d)
        | PriceError::OutOfRange { got: d, .. }
        | PriceError::OffGrid { got: d, .. } => NormalizeReject::PriceOutOfDomain(d),
        PriceError::NonFinite(_) | PriceError::FloatOutOfDomain(_) => {
            NormalizeReject::PriceOutOfDomain(Decimal::ZERO)
        }
    }
}

/// True when rounding `target` onto the grid lands outside `[tick, 1 − tick]`, so
/// [`Price::quantize`]'s clamp engaged. Only meaningful for `target ∈ [0, 1]`
/// (the only case that reaches here after a successful quantize).
fn price_was_clamped(target: Decimal, tick: TickSize, dir: RoundDir) -> bool {
    let strategy = match dir {
        RoundDir::Down => RoundingStrategy::ToNegativeInfinity,
        RoundDir::Up => RoundingStrategy::ToPositiveInfinity,
    };
    let rounded = target.round_dp_with_strategy(tick.decimal_places(), strategy);
    rounded < tick.min_price() || rounded > tick.max_price()
}

#[cfg(test)]
mod tests {
    use core_types::{
        Asset, BookLevel, ConditionId, Dollars, FeeParams, Outcome, ResolutionSource, Series,
        TimeInForce, TimestampMs, TokenId, TokenPair, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    use super::*;

    // ---- fixtures --------------------------------------------------------

    fn market(tick: TickSize, min_shares: Decimal) -> MarketInfo {
        MarketInfo {
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(1_781_000_000_000),
            },
            event_slug: "btc-updown-5m-1781000000".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "ab".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("111").unwrap(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: TimestampMs::from_millis(1_781_000_300_000),
            strike: Some(dec!(63500)),
            tick_size: tick,
            min_order_size: Size::new(min_shares).unwrap(),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        }
    }

    /// Market with the default 5-share venue minimum.
    fn mkt(tick: TickSize) -> MarketInfo {
        market(tick, dec!(5))
    }

    fn p(d: Decimal, tick: TickSize) -> Price {
        Price::on_grid(d, tick).unwrap()
    }

    fn shares(n: Decimal) -> OrderQty {
        OrderQty::Shares(Size::new(n).unwrap())
    }

    fn notional(n: Decimal) -> OrderQty {
        OrderQty::Notional(Dollars::new(n))
    }

    fn gtc_po() -> TimeInForce {
        TimeInForce::Gtc { post_only: true }
    }

    fn draft(side: Side, price: Decimal, qty: OrderQty, tif: TimeInForce) -> OrderDraft {
        OrderDraft {
            client_id: Some("c0".to_owned()),
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(1_781_000_000_000),
            },
            token_id: TokenId::new("111").unwrap(),
            outcome: Outcome::Up,
            side,
            price,
            qty,
            tif,
        }
    }

    fn top(bid: Option<Decimal>, ask: Option<Decimal>, tick: TickSize) -> TopOfBook {
        let level = |d: Decimal| BookLevel {
            price: p(d, tick),
            size: Size::new(dec!(100)).unwrap(),
        };
        TopOfBook {
            bid: bid.map(level),
            ask: ask.map(level),
            ts: TimestampMs::from_millis(1),
        }
    }

    /// Unwrap a successful normalization to its (qty, adjustments).
    fn ok(
        side: Side,
        price: Decimal,
        qty: OrderQty,
        tif: TimeInForce,
        m: &MarketInfo,
        book: Option<&TopOfBook>,
    ) -> Normalized {
        normalize(
            &draft(side, price, qty, tif),
            m,
            book,
            &NormalizerParams::default(),
        )
        .unwrap()
    }

    fn shares_of(n: &Normalized) -> Decimal {
        match n.order.qty {
            OrderQty::Shares(s) => s.as_decimal(),
            OrderQty::Notional(_) => panic!("expected shares"),
        }
    }

    // ---- the six required cases ------------------------------------------

    #[test]
    fn price_010_requires_10_shares() {
        let m = mkt(TickSize::T001);
        let n = ok(Side::Buy, dec!(0.10), shares(dec!(1)), gtc_po(), &m, None);
        assert_eq!(shares_of(&n), dec!(10)); // ceil(1/0.10) = 10 > 5
        assert!(n.adjustments.size_bumped_to_min);
        assert_eq!(n.order.price, p(dec!(0.10), TickSize::T001));
    }

    #[test]
    fn price_050_requires_5_shares() {
        let m = mkt(TickSize::T001);
        let n = ok(Side::Buy, dec!(0.50), shares(dec!(1)), gtc_po(), &m, None);
        assert_eq!(shares_of(&n), dec!(5)); // 5-share floor binds over ceil(1/0.5)=2
        assert!(n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn fine_grid_near_099_binds_five_shares() {
        let m = mkt(TickSize::T0001);
        let n = ok(Side::Sell, dec!(0.999), shares(dec!(1)), gtc_po(), &m, None);
        assert_eq!(n.order.price, p(dec!(0.999), TickSize::T0001));
        assert_eq!(shares_of(&n), dec!(5)); // ceil(1/0.999)=2, 5 binds
        assert!(n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn fine_grid_near_001_needs_a_thousand_shares() {
        let m = mkt(TickSize::T0001);
        let n = ok(Side::Buy, dec!(0.001), shares(dec!(1)), gtc_po(), &m, None);
        assert_eq!(n.order.price, p(dec!(0.001), TickSize::T0001));
        assert_eq!(shares_of(&n), dec!(1000)); // ceil(1/0.001) = 1000
        assert!(n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn tick_flip_boundary_stays_on_fine_grid() {
        // On the 0.001 grid, 0.961 is a legal price and must NOT be snapped to
        // the 0.01-grid 0.96/0.97.
        let m = mkt(TickSize::T0001);
        let n = ok(
            Side::Sell,
            dec!(0.961),
            shares(dec!(10)),
            gtc_po(),
            &m,
            None,
        );
        assert_eq!(n.order.price, p(dec!(0.961), TickSize::T0001));
        assert!(!n.adjustments.size_bumped_to_min); // 10 > max(5, ceil(1/0.961)=2)
        assert!(!n.adjustments.price.clamped_into_range);
    }

    #[test]
    fn rounding_pushes_notional_below_one_dollar_so_size_bumps() {
        // 14 × 0.07 = $0.98 < $1; ceil(1/0.07) = ceil(14.28) = 15.
        let m = mkt(TickSize::T001);
        let n = ok(Side::Buy, dec!(0.07), shares(dec!(14)), gtc_po(), &m, None);
        assert_eq!(shares_of(&n), dec!(15));
        assert!(n.adjustments.size_bumped_to_min);
        // Pin the boundary the rule defends.
        assert!(dec!(14) * dec!(0.07) < MIN_NOTIONAL);
        assert!(dec!(15) * dec!(0.07) >= MIN_NOTIONAL);
    }

    #[test]
    fn price_rounds_down_so_same_shares_no_longer_reach_one_dollar() {
        // Buy rounds 0.074 DOWN to 0.07; the $1 floor is recomputed at 0.07, so
        // 14 shares is no longer enough → bump to 15.
        let m = mkt(TickSize::T001);
        let n = ok(Side::Buy, dec!(0.074), shares(dec!(14)), gtc_po(), &m, None);
        assert_eq!(n.order.price, p(dec!(0.07), TickSize::T001));
        assert_eq!(n.adjustments.price.from, dec!(0.074));
        assert_eq!(shares_of(&n), dec!(15));
        assert!(n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn post_only_buy_at_or_above_ask_crosses() {
        let m = mkt(TickSize::T001);
        let book = top(Some(dec!(0.53)), Some(dec!(0.55)), TickSize::T001);
        let err = normalize(
            &draft(Side::Buy, dec!(0.55), shares(dec!(10)), gtc_po()),
            &m,
            Some(&book),
            &NormalizerParams::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            NormalizeReject::WouldCross {
                side: Side::Buy,
                price: p(dec!(0.55), TickSize::T001),
                touch: p(dec!(0.55), TickSize::T001),
            }
        );
    }

    #[test]
    fn post_only_sell_at_or_below_bid_crosses() {
        let m = mkt(TickSize::T001);
        let book = top(Some(dec!(0.55)), Some(dec!(0.57)), TickSize::T001);
        let err = normalize(
            &draft(Side::Sell, dec!(0.55), shares(dec!(10)), gtc_po()),
            &m,
            Some(&book),
            &NormalizerParams::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            NormalizeReject::WouldCross {
                side: Side::Sell,
                ..
            }
        ));
    }

    #[test]
    fn post_only_non_crossing_passes() {
        let m = mkt(TickSize::T001);
        let book = top(Some(dec!(0.53)), Some(dec!(0.55)), TickSize::T001);
        let n = ok(
            Side::Buy,
            dec!(0.54),
            shares(dec!(10)),
            gtc_po(),
            &m,
            Some(&book),
        );
        assert_eq!(n.order.price, p(dec!(0.54), TickSize::T001));
    }

    #[test]
    fn post_only_with_empty_or_one_sided_book_passes() {
        let m = mkt(TickSize::T001);
        // No book at all.
        assert!(
            normalize(
                &draft(Side::Buy, dec!(0.99), shares(dec!(10)), gtc_po()),
                &m,
                None,
                &NormalizerParams::default(),
            )
            .is_ok()
        );
        // Book with no ask cannot be crossed by a buy.
        let one_sided = top(Some(dec!(0.90)), None, TickSize::T001);
        assert!(
            normalize(
                &draft(Side::Buy, dec!(0.99), shares(dec!(10)), gtc_po()),
                &m,
                Some(&one_sided),
                &NormalizerParams::default(),
            )
            .is_ok()
        );
    }

    // ---- additional edges -------------------------------------------------

    #[test]
    fn clamp_into_range_is_flagged() {
        // Sell rounds 0.995 UP to 1.00, then the clamp pulls it to 0.99.
        let m = mkt(TickSize::T001);
        let n = ok(
            Side::Sell,
            dec!(0.995),
            shares(dec!(10)),
            gtc_po(),
            &m,
            None,
        );
        assert_eq!(n.order.price, p(dec!(0.99), TickSize::T001));
        assert!(n.adjustments.price.clamped_into_range);
    }

    #[test]
    fn size_rounded_down_to_precision() {
        let m = mkt(TickSize::T001);
        // size_decimals default 0: 12.7 shares → 12.
        let n = ok(
            Side::Buy,
            dec!(0.20),
            shares(dec!(12.7)),
            gtc_po(),
            &m,
            None,
        );
        assert_eq!(shares_of(&n), dec!(12)); // 12 ≥ max(5, ceil(1/0.20)=5)
        assert!(n.adjustments.size_rounded_down);
        assert!(!n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn positive_size_that_rounds_to_zero_bumps_not_rejects() {
        let m = mkt(TickSize::T001);
        let n = ok(Side::Buy, dec!(0.50), shares(dec!(0.4)), gtc_po(), &m, None);
        assert_eq!(shares_of(&n), dec!(5)); // 0.4 → round-down 0 → bump to 5-floor
        assert!(n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn literal_zero_size_is_rejected() {
        let m = mkt(TickSize::T001);
        let err = normalize(
            &draft(Side::Buy, dec!(0.50), shares(dec!(0)), gtc_po()),
            &m,
            None,
            &NormalizerParams::default(),
        )
        .unwrap_err();
        assert_eq!(err, NormalizeReject::ZeroSize);
    }

    #[test]
    fn qty_kind_mismatches_are_rejected() {
        let m = mkt(TickSize::T001);
        let pr = &NormalizerParams::default();
        // Resting order given a dollar notional.
        assert!(matches!(
            normalize(
                &draft(Side::Buy, dec!(0.5), notional(dec!(10)), gtc_po()),
                &m,
                None,
                pr
            ),
            Err(NormalizeReject::QtyKindMismatch(_))
        ));
        // Marketable BUY given shares.
        assert!(matches!(
            normalize(
                &draft(Side::Buy, dec!(0.5), shares(dec!(10)), TimeInForce::Fak),
                &m,
                None,
                pr
            ),
            Err(NormalizeReject::QtyKindMismatch(_))
        ));
        // Marketable SELL given a notional.
        assert!(matches!(
            normalize(
                &draft(Side::Sell, dec!(0.5), notional(dec!(10)), TimeInForce::Fok),
                &m,
                None,
                pr
            ),
            Err(NormalizeReject::QtyKindMismatch(_))
        ));
    }

    #[test]
    fn marketable_buy_below_one_dollar_is_rejected() {
        let m = mkt(TickSize::T001);
        let err = normalize(
            &draft(
                Side::Buy,
                dec!(0.50),
                notional(dec!(0.50)),
                TimeInForce::Fak,
            ),
            &m,
            None,
            &NormalizerParams::default(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            NormalizeReject::BelowMinNotional {
                got: dec!(0.50),
                min: MIN_NOTIONAL,
            }
        );
    }

    #[test]
    fn marketable_buy_rounds_slippage_cap_outward() {
        // A marketable BUY's worst-price limit rounds UP (permissive), so it is
        // never tightened into a no-fill. 0.541 → 0.55.
        let m = mkt(TickSize::T001);
        let n = ok(
            Side::Buy,
            dec!(0.541),
            notional(dec!(25)),
            TimeInForce::Fak,
            &m,
            None,
        );
        assert_eq!(n.order.price, p(dec!(0.55), TickSize::T001));
        assert_eq!(n.order.qty, notional(dec!(25)));
        assert!(!n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn marketable_sell_rounds_slippage_cap_outward() {
        // A marketable SELL's worst-price limit rounds DOWN. 0.549 → 0.54.
        let m = mkt(TickSize::T001);
        let n = ok(
            Side::Sell,
            dec!(0.549),
            shares(dec!(10)),
            TimeInForce::Fak,
            &m,
            None,
        );
        assert_eq!(n.order.price, p(dec!(0.54), TickSize::T001));
    }

    #[test]
    fn price_outside_domain_is_rejected() {
        let m = mkt(TickSize::T001);
        let pr = &NormalizerParams::default();
        for bad in [dec!(1.5), dec!(-0.1)] {
            assert!(matches!(
                normalize(
                    &draft(Side::Buy, bad, shares(dec!(10)), gtc_po()),
                    &m,
                    None,
                    pr
                ),
                Err(NormalizeReject::PriceOutOfDomain(_))
            ));
        }
    }

    #[test]
    fn market_minimum_order_size_can_bind() {
        // Market requires 20 shares; that beats both 5 and ceil(1/0.50)=2.
        let m = market(TickSize::T001, dec!(20));
        let n = ok(Side::Buy, dec!(0.50), shares(dec!(1)), gtc_po(), &m, None);
        assert_eq!(shares_of(&n), dec!(20));
        assert!(n.adjustments.size_bumped_to_min);
    }

    #[test]
    fn already_valid_order_is_unchanged_and_unflagged() {
        let m = mkt(TickSize::T001);
        // 50 shares at 0.50 on-grid: no rounding, no bump, no clamp.
        let n = ok(Side::Buy, dec!(0.50), shares(dec!(50)), gtc_po(), &m, None);
        assert_eq!(shares_of(&n), dec!(50));
        assert!(!n.adjustments.size_rounded_down);
        assert!(!n.adjustments.size_bumped_to_min);
        assert!(!n.adjustments.price.clamped_into_range);
        assert_eq!(n.adjustments.price.from, dec!(0.50));
        assert_eq!(n.adjustments.price.to, p(dec!(0.50), TickSize::T001));
    }

    #[test]
    fn successful_share_orders_satisfy_every_minimum() {
        // Construction-invariant guard across a sweep of prices/sizes.
        let m = mkt(TickSize::T001);
        let pr = NormalizerParams::default();
        for price in [
            dec!(0.01),
            dec!(0.07),
            dec!(0.10),
            dec!(0.5),
            dec!(0.93),
            dec!(0.99),
        ] {
            for want in [dec!(1), dec!(7), dec!(50), dec!(500)] {
                let n = normalize(
                    &draft(Side::Buy, price, shares(want), gtc_po()),
                    &m,
                    None,
                    &pr,
                )
                .unwrap();
                let s = shares_of(&n);
                let rp = n.order.price.as_decimal();
                assert!(s >= MIN_SHARES, "{s} < 5 at price {price}");
                assert!(s >= m.min_order_size.as_decimal());
                assert!(s * rp >= MIN_NOTIONAL, "notional {} < $1", s * rp);
            }
        }
    }

    #[test]
    fn reject_reasons_bridge_to_venue_vocabulary() {
        use venue_api::RejectReason as R;
        assert_eq!(
            NormalizeReject::PriceOutOfDomain(dec!(2)).as_reject_reason(),
            R::TickRule
        );
        assert_eq!(
            NormalizeReject::ZeroSize.as_reject_reason(),
            R::BelowMinSize
        );
        assert_eq!(
            NormalizeReject::BelowMinNotional {
                got: dec!(0.5),
                min: MIN_NOTIONAL
            }
            .as_reject_reason(),
            R::BelowMinSize
        );
        assert_eq!(
            NormalizeReject::WouldCross {
                side: Side::Buy,
                price: p(dec!(0.55), TickSize::T001),
                touch: p(dec!(0.55), TickSize::T001),
            }
            .as_reject_reason(),
            R::CrossedBook
        );
        assert_eq!(
            NormalizeReject::QtyKindMismatch("x".to_owned()).as_reject_reason(),
            R::Other("x".to_owned())
        );
    }
}
