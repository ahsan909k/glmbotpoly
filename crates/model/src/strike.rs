//! Per-window strike capture — the "price to beat" sampled at window open from
//! the resolution-grade feed (§6, §8).
//!
//! Sans-IO and **arrival-order independent** by construction. The venue has no
//! API that exposes a live window's strike (verified 2026-06-13: Gamma only
//! publishes `eventMetadata.priceToBeat` minutes *after* resolution), so the
//! bot must capture it itself from the Chainlink RTDS stream. RTDS delivers a
//! ~2-minute backfill burst on (re)subscribe whose prints arrive in one local
//! instant with honest-but-old exchange timestamps, possibly reordered — so
//! the capture must converge to the same strike whatever order it sees.
//!
//! It does this with two ideas:
//! - **Monotone candidate refinement.** Two slots are tracked: `before`, the
//!   latest print at-or-before open, and `after`, the earliest at-or-after
//!   open. Each only ever moves *toward* the boundary, so the final pair is a
//!   pure function of the set of delivered prints — independent of order.
//! - **Settle-gated freeze.** The strike `K` is frozen only once the capture
//!   has *settled*: a genuinely live print (recent exchange time) at-or-after
//!   open has been seen, proving wall-clock has passed the window open and the
//!   initial backfill is therefore complete. Freezing on the first
//!   "good-enough" print instead would be order-dependent (a burst can deliver
//!   a within-tolerance-but-not-closest print first). Settling defers the
//!   freeze until `before`/`after` have stopped moving.
//!
//! After freezing, `K` never changes (the engine needs determinism); a later
//! print that would have changed the chosen candidate is surfaced as
//! [`StrikeOutcome::RevisedCandidate`] and journaled, but `K` holds.

use core_types::{
    Asset, MarketInfo, PriceSource, PriceTick, ResolutionKind, TickKind, TimestampMs, WindowId,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use thiserror::Error;

/// Longest a tick's exchange timestamp may lead its local receive time before
/// it is treated as a glitch (vol.rs precedent). Backfill prints have
/// `ts_exchange ≪ ts_local` and are unaffected; a real live print is at most a
/// second or two old.
const MAX_EXCHANGE_LEAD_MS: i64 = 10_000;

/// A print counts as **live** (rather than backfill) when its exchange time is
/// within this of local receipt. RTDS live updates run ~1 s behind exchange
/// time; backfill prints are minutes old — 5 s separates them cleanly. A live
/// at-or-after-open print is the settle signal.
const LIVENESS_MS: i64 = 5_000;

/// Which print of the boundary is the strike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrikeRule {
    /// The latest print at or before open — the value in effect at the
    /// boundary instant (LOCF). Matches the verified Chainlink chain
    /// continuity (`priceToBeat(N) = finalPrice(N−1)`) for the 5m/15m series.
    LastAtOrBefore,
    /// The earliest print at or after open — a candle-open proxy for the 1h
    /// Binance-resolved series. Provisional; see [`StrikeCapture::for_market`].
    FirstAtOrAfter,
}

/// Parameters for one [`StrikeCapture`]. Code constants, not config (the
/// vol-estimator precedent: config exposure is the model-driver task's job).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrikeParams {
    /// Feed the strike is sampled from.
    pub source: PriceSource,
    /// Observation flavor from that feed.
    pub kind: TickKind,
    /// Which boundary print is the strike.
    pub rule: StrikeRule,
    /// How close (ms) the chosen print's exchange time must be to open for the
    /// strike to be trusted. Wider than the Chainlink 1 Hz cadence so the
    /// last-before-open print (≤ 1 s out) always qualifies, with margin.
    pub boundary_tolerance_ms: i64,
    /// Prints older than `open − lookback` are ignored as `before` candidates
    /// — an ancient print is never a plausible strike.
    pub lookback_ms: i64,
}

impl Default for StrikeParams {
    /// The 5m/15m Chainlink defaults.
    fn default() -> Self {
        Self {
            source: PriceSource::ChainlinkRtds,
            kind: TickKind::Vendor,
            rule: StrikeRule::LastAtOrBefore,
            boundary_tolerance_ms: 2_500,
            lookback_ms: 180_000,
        }
    }
}

/// Why a market cannot have its strike captured by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StrikeError {
    /// The market's resolution source is not a feed the model can anchor on
    /// (`ResolutionKind::Other`) — the window is untradable (§6).
    #[error("resolution kind {0:?} has no strike feed")]
    UnsupportedResolution(ResolutionKind),
}

/// One observed boundary-candidate print, retained for journaling and
/// post-hoc verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidatePrint {
    /// Wire-exact price.
    pub value: Decimal,
    /// Same value as `f64` for the fair-value math.
    pub value_f64: f64,
    /// Source exchange timestamp.
    pub ts_exchange: TimestampMs,
    /// Local receive time.
    pub ts_local: TimestampMs,
}

impl CandidatePrint {
    /// Signed offset of this print's exchange time from window open
    /// (negative = before open).
    #[must_use]
    pub const fn offset_ms(&self, open: TimestampMs) -> i64 {
        self.ts_exchange.as_millis() - open.as_millis()
    }
}

/// Why [`StrikeCapture::on_tick`] did not consume a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrikeIgnoreReason {
    /// Not this capture's (asset, source, kind).
    WrongStream,
    /// Price non-finite, zero, negative, or unrepresentable as `f64`.
    BadValue,
    /// A `before`-side print older than `open − lookback` — too old to be a
    /// plausible strike.
    TooOld,
    /// `ts_exchange` leads `ts_local` beyond [`MAX_EXCHANGE_LEAD_MS`].
    ExchangeAheadOfLocal,
}

/// What one tick did to the capture — telemetry; the machine stays log-free.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrikeOutcome {
    /// Tick not consumed; see the reason.
    Ignored(StrikeIgnoreReason),
    /// A valid tick that did not move either candidate.
    Observed,
    /// The `before` candidate moved closer to open.
    ImprovedBefore,
    /// The `after` candidate moved closer to open.
    ImprovedAfter,
    /// The strike froze on this tick (capture settled, candidate trusted).
    Frozen {
        /// The frozen strike value.
        value: f64,
        /// The chosen print's offset from open (ms).
        offset_ms: i64,
    },
    /// Post-freeze: a better candidate arrived with a *different* value. `K`
    /// stays frozen; the operator/journal sees the discrepancy.
    RevisedCandidate {
        /// Relative gap between the new candidate and the frozen strike.
        delta_rel: f64,
    },
}

/// How trustworthy the strike estimate is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrikeQuality {
    /// No usable candidate yet — the model must stay `Warming`.
    Missing,
    /// A candidate exists but its exchange time is farther from open than the
    /// tolerance — used for display, never frozen.
    Distant,
    /// A candidate within tolerance of open — trustworthy; frozen once settled.
    Boundary,
}

/// Diagnostic snapshot of the capture (journaling / rendering surface). The
/// engine consumes [`StrikeCapture::strike`] instead, which is `Some` only
/// once frozen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrikeEstimate {
    /// Chosen strike (`f64`); `None` when [`StrikeQuality::Missing`].
    pub k: Option<f64>,
    /// Chosen strike as a [`Decimal`]; `None` when missing.
    pub k_decimal: Option<Decimal>,
    /// The rule in force.
    pub rule: StrikeRule,
    /// Trust level of `k`.
    pub quality: StrikeQuality,
    /// Whether `k` is frozen (the engine's committed strike).
    pub frozen: bool,
    /// Latest at-or-before-open print, if any.
    pub before: Option<CandidatePrint>,
    /// Earliest at-or-after-open print, if any.
    pub after: Option<CandidatePrint>,
}

/// Per-window strike capture (§6, §8). See the module docs for the invariants.
#[derive(Debug, Clone)]
pub struct StrikeCapture {
    window: WindowId,
    asset: Asset,
    open_time: TimestampMs,
    params: StrikeParams,
    before: Option<CandidatePrint>,
    after: Option<CandidatePrint>,
    frozen: Option<CandidatePrint>,
    /// A live at-or-after-open print has been seen — wall-clock passed open, so
    /// the initial backfill is complete and the candidates have settled.
    settled: bool,
}

impl StrikeCapture {
    /// Builds a capture for `window` (open at `open_time`) tracking `asset`.
    #[must_use]
    pub const fn new(
        window: WindowId,
        asset: Asset,
        open_time: TimestampMs,
        params: StrikeParams,
    ) -> Self {
        Self {
            window,
            asset,
            open_time,
            params,
            before: None,
            after: None,
            frozen: None,
            settled: false,
        }
    }

    /// Builds a capture for a market, selecting the feed and rule from its
    /// resolution source (§6: 5m/15m resolve on Chainlink, 1h on Binance).
    ///
    /// The Binance-candle path is **provisional**: it samples the first direct
    /// trade at-or-after open as a candle-open proxy and is unverified (the
    /// acceptance run only covers BTC-5m). The 5m/15m Chainlink path is the
    /// tested one.
    ///
    /// # Errors
    /// [`StrikeError::UnsupportedResolution`] when the market resolves on a
    /// feed the model cannot identify.
    pub fn for_market(market: &MarketInfo) -> Result<Self, StrikeError> {
        let (source, kind, rule) = match market.resolution.kind {
            ResolutionKind::ChainlinkDataStream => (
                PriceSource::ChainlinkRtds,
                TickKind::Vendor,
                StrikeRule::LastAtOrBefore,
            ),
            ResolutionKind::BinanceCandle => (
                PriceSource::BinanceDirect,
                TickKind::Trade,
                StrikeRule::FirstAtOrAfter,
            ),
            ResolutionKind::Other => {
                return Err(StrikeError::UnsupportedResolution(market.resolution.kind));
            }
        };
        Ok(Self::new(
            market.window,
            market.window.series.asset,
            market.window.open_time,
            StrikeParams {
                source,
                kind,
                rule,
                ..StrikeParams::default()
            },
        ))
    }

    /// Consumes one price tick.
    pub fn on_tick(&mut self, tick: &PriceTick) -> StrikeOutcome {
        if tick.asset != self.asset
            || tick.source != self.params.source
            || tick.kind != self.params.kind
        {
            return StrikeOutcome::Ignored(StrikeIgnoreReason::WrongStream);
        }
        let Some(value_f64) = tick.value.to_f64() else {
            return StrikeOutcome::Ignored(StrikeIgnoreReason::BadValue);
        };
        if !value_f64.is_finite() || value_f64 <= 0.0 {
            return StrikeOutcome::Ignored(StrikeIgnoreReason::BadValue);
        }
        let lead = tick.ts_exchange.signed_duration_since(tick.ts_local);
        if lead.as_millis() > MAX_EXCHANGE_LEAD_MS {
            return StrikeOutcome::Ignored(StrikeIgnoreReason::ExchangeAheadOfLocal);
        }

        let te = tick.ts_exchange.as_millis();
        let open = self.open_time.as_millis();
        let cand = CandidatePrint {
            value: tick.value,
            value_f64,
            ts_exchange: tick.ts_exchange,
            ts_local: tick.ts_local,
        };

        // Settle signal: a live print at or after open proves wall-clock has
        // passed open, so the backfill burst is done and the candidates have
        // stopped moving.
        if te >= open && (tick.ts_local.as_millis() - te) <= LIVENESS_MS {
            self.settled = true;
        }

        let mut improved_before = false;
        let mut improved_after = false;
        if te <= open {
            if te < open - self.params.lookback_ms {
                return StrikeOutcome::Ignored(StrikeIgnoreReason::TooOld);
            }
            // Keep the latest at-or-before open (largest ts_exchange).
            if self.before.is_none_or(|b| te > b.ts_exchange.as_millis()) {
                self.before = Some(cand);
                improved_before = true;
            }
        }
        if te >= open {
            // Keep the earliest at-or-after open (smallest ts_exchange).
            if self.after.is_none_or(|a| te < a.ts_exchange.as_millis()) {
                self.after = Some(cand);
                improved_after = true;
            }
        }

        let rule_candidate = self.rule_candidate();

        if self.frozen.is_none() {
            if self.settled
                && let Some(c) = rule_candidate
                && self.quality_of(&c) == StrikeQuality::Boundary
            {
                let offset_ms = c.offset_ms(self.open_time);
                self.frozen = Some(c);
                return StrikeOutcome::Frozen {
                    value: c.value_f64,
                    offset_ms,
                };
            }
            return improvement_outcome(improved_before, improved_after);
        }

        // Already frozen: surface a value-changing improvement to the chosen
        // candidate, but never move K.
        if let Some(frozen) = self.frozen {
            let rule_improved = match self.params.rule {
                StrikeRule::LastAtOrBefore => improved_before,
                StrikeRule::FirstAtOrAfter => improved_after,
            };
            if rule_improved
                && let Some(c) = rule_candidate
                && c.value_f64 != frozen.value_f64
            {
                let delta_rel = ((c.value_f64 - frozen.value_f64) / frozen.value_f64).abs();
                return StrikeOutcome::RevisedCandidate { delta_rel };
            }
        }
        improvement_outcome(improved_before, improved_after)
    }

    /// The rule-selected candidate (frozen value not consulted here).
    fn rule_candidate(&self) -> Option<CandidatePrint> {
        match self.params.rule {
            StrikeRule::LastAtOrBefore => self.before,
            StrikeRule::FirstAtOrAfter => self.after,
        }
    }

    /// Quality of a candidate by its distance from open.
    fn quality_of(&self, c: &CandidatePrint) -> StrikeQuality {
        if c.offset_ms(self.open_time).abs() <= self.params.boundary_tolerance_ms {
            StrikeQuality::Boundary
        } else {
            StrikeQuality::Distant
        }
    }

    /// The engine consumption path: the frozen strike, `Some` only once
    /// captured. A warming/uncertain capture can never leak a strike into
    /// fair value.
    #[must_use]
    pub fn strike(&self) -> Option<f64> {
        self.frozen.map(|c| c.value_f64)
    }

    /// The frozen strike as a [`Decimal`] (journaling / verification).
    #[must_use]
    pub fn strike_decimal(&self) -> Option<Decimal> {
        self.frozen.map(|c| c.value)
    }

    /// Whether the strike has frozen.
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.frozen.is_some()
    }

    /// Full diagnostic snapshot.
    #[must_use]
    pub fn estimate(&self) -> StrikeEstimate {
        // Once frozen, K is the frozen print; before then, the rule candidate
        // (always Distant if present — a Boundary candidate would have frozen
        // once settled).
        let chosen = self.frozen.or_else(|| self.rule_candidate());
        let (k, k_decimal, quality) = match chosen {
            Some(c) => {
                let quality = if self.frozen.is_some() {
                    StrikeQuality::Boundary
                } else {
                    self.quality_of(&c)
                };
                (Some(c.value_f64), Some(c.value), quality)
            }
            None => (None, None, StrikeQuality::Missing),
        };
        StrikeEstimate {
            k,
            k_decimal,
            rule: self.params.rule,
            quality,
            frozen: self.frozen.is_some(),
            before: self.before,
            after: self.after,
        }
    }

    /// The window this capture belongs to.
    #[must_use]
    pub const fn window(&self) -> WindowId {
        self.window
    }

    /// The parameters in force.
    #[must_use]
    pub const fn params(&self) -> &StrikeParams {
        &self.params
    }
}

/// Maps the two improvement flags to a non-freeze outcome.
const fn improvement_outcome(improved_before: bool, improved_after: bool) -> StrikeOutcome {
    if improved_before {
        StrikeOutcome::ImprovedBefore
    } else if improved_after {
        StrikeOutcome::ImprovedAfter
    } else {
        StrikeOutcome::Observed
    }
}

#[cfg(test)]
mod tests {
    use core_types::{
        ConditionId, FeeParams, MarketInfo, ResolutionSource, Series, Size, TickSize, TokenId,
        TokenPair, WindowDuration,
    };
    use rust_decimal::dec;
    use rust_decimal::prelude::FromPrimitive;

    use super::*;

    /// Window open at this unix-ms instant in every test.
    const OPEN: i64 = 1_781_290_500_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN),
        }
    }

    fn capture() -> StrikeCapture {
        StrikeCapture::new(
            window(),
            Asset::Btc,
            TimestampMs::from_millis(OPEN),
            StrikeParams::default(),
        )
    }

    /// A Chainlink BTC vendor tick: exchange time `ex_off` ms from open,
    /// received `rx_off` ms from open.
    fn tick(ex_off: i64, rx_off: i64, price: f64) -> PriceTick {
        PriceTick {
            source: PriceSource::ChainlinkRtds,
            asset: Asset::Btc,
            kind: TickKind::Vendor,
            value: Decimal::from_f64(price).unwrap(),
            ts_exchange: TimestampMs::from_millis(OPEN + ex_off),
            ts_local: TimestampMs::from_millis(OPEN + rx_off),
        }
    }

    /// A live tick (exchange ≈ local).
    fn live(ex_off: i64, price: f64) -> PriceTick {
        tick(ex_off, ex_off, price)
    }

    #[test]
    fn in_order_live_stream_freezes_on_last_before_open() {
        let mut cap = capture();
        // Pre-open prints arrive live, in order; none settles.
        assert_eq!(
            cap.on_tick(&live(-3_000, 100.0)),
            StrikeOutcome::ImprovedBefore
        );
        assert_eq!(
            cap.on_tick(&live(-2_000, 101.0)),
            StrikeOutcome::ImprovedBefore
        );
        assert_eq!(
            cap.on_tick(&live(-1_000, 102.0)),
            StrikeOutcome::ImprovedBefore
        );
        assert!(!cap.is_frozen(), "no after-open print yet → not settled");
        // First post-open print settles and freezes on the −1 000 ms print.
        assert_eq!(
            cap.on_tick(&live(500, 103.0)),
            StrikeOutcome::Frozen {
                value: 102.0,
                offset_ms: -1_000
            }
        );
        assert_eq!(cap.strike(), Some(102.0));
        let est = cap.estimate();
        assert_eq!(est.quality, StrikeQuality::Boundary);
        assert!(est.frozen);
    }

    #[test]
    fn exact_boundary_print_freezes_immediately() {
        let mut cap = capture();
        // A live print exactly at open fills both slots and settles at once.
        assert_eq!(
            cap.on_tick(&live(0, 99.5)),
            StrikeOutcome::Frozen {
                value: 99.5,
                offset_ms: 0
            }
        );
        assert_eq!(cap.strike(), Some(99.5));
    }

    #[test]
    fn backfill_burst_converges_regardless_of_order() {
        // The boundary prints, delivered in one local instant (backfill), with
        // old exchange times. Three orderings must reach the same frozen K.
        let prices = [
            (-2_400, 100.0),
            (-1_400, 101.0),
            (-400, 102.5),
            (600, 103.0),
            (1_600, 104.0),
        ];
        let orders: [&[usize]; 3] = [
            &[0, 1, 2, 3, 4], // chronological
            &[4, 3, 2, 1, 0], // reverse
            &[3, 0, 4, 2, 1], // shuffled
        ];
        for order in orders {
            let mut cap = capture();
            // Entire burst delivered at one local instant 2 minutes after open
            // (honest old exchange times) — nothing settles during the burst.
            let rx = 120_000;
            for &i in order {
                let (ex_off, price) = prices[i];
                cap.on_tick(&tick(ex_off, rx, price));
            }
            assert!(
                !cap.is_frozen(),
                "backfill alone never settles (order {order:?})"
            );
            // before = −400 print (latest ≤ open), after = +600 (earliest ≥).
            let est = cap.estimate();
            assert_eq!(est.before.map(|c| c.value_f64), Some(102.5));
            assert_eq!(est.after.map(|c| c.value_f64), Some(103.0));
            // A live post-open tick then settles and freezes on −400.
            let out = cap.on_tick(&live(2_000, 105.0));
            assert_eq!(
                out,
                StrikeOutcome::Frozen {
                    value: 102.5,
                    offset_ms: -400
                },
                "order {order:?}"
            );
            assert_eq!(cap.strike(), Some(102.5), "order {order:?}");
        }
    }

    #[test]
    fn late_boot_recovers_strike_from_backfill() {
        // Boot ~1 min into the window: the backfill burst (old exchange times,
        // one local instant) includes the pre-open print, then live resumes.
        let mut cap = capture();
        let rx = 60_000;
        for (ex_off, price) in [(-1_800, 100.0), (-800, 101.0), (200, 101.5), (1_200, 102.0)] {
            cap.on_tick(&tick(ex_off, rx, price));
        }
        assert!(!cap.is_frozen());
        // Live tick settles → freeze on the −800 ms print.
        assert!(matches!(
            cap.on_tick(&live(61_000 - 60_000 + 500, 102.5)),
            StrikeOutcome::Frozen { value, offset_ms: -800 } if value == 101.0
        ));
        assert_eq!(cap.strike(), Some(101.0));
    }

    #[test]
    fn freeze_then_late_closer_print_revises_but_holds_k() {
        let mut cap = capture();
        // Settle + freeze on a −1 500 ms before print (within 2 500 tolerance);
        // the closest pre-open print was missed on first delivery.
        cap.on_tick(&live(-1_500, 100.0));
        assert_eq!(
            cap.on_tick(&live(400, 101.0)),
            StrikeOutcome::Frozen {
                value: 100.0,
                offset_ms: -1_500
            }
        );
        // A backfill re-delivery now supplies the missed −300 ms print with a
        // different value: the candidate improves, K stays frozen.
        let out = cap.on_tick(&tick(-300, 70_000, 100.4));
        match out {
            StrikeOutcome::RevisedCandidate { delta_rel } => {
                assert!((delta_rel - 0.004).abs() < 1e-9, "delta_rel {delta_rel}");
            }
            other => panic!("expected RevisedCandidate, got {other:?}"),
        }
        assert_eq!(cap.strike(), Some(100.0), "frozen K must not move");
        // The journal still sees the closer candidate.
        assert_eq!(cap.estimate().before.map(|c| c.value_f64), Some(100.4));
    }

    #[test]
    fn distant_only_candidate_does_not_freeze() {
        let mut cap = capture();
        // Best before print is 4 s before open — beyond the 2 500 ms tolerance.
        cap.on_tick(&live(-4_000, 100.0));
        // A settling after-print arrives, but the before candidate is Distant.
        let out = cap.on_tick(&live(500, 101.0));
        assert!(matches!(
            out,
            StrikeOutcome::ImprovedAfter | StrikeOutcome::Observed
        ));
        assert!(!cap.is_frozen(), "Distant candidate must not freeze");
        assert_eq!(cap.strike(), None);
        assert_eq!(cap.estimate().quality, StrikeQuality::Distant);
    }

    #[test]
    fn missing_until_any_candidate() {
        let cap = capture();
        let est = cap.estimate();
        assert_eq!(est.quality, StrikeQuality::Missing);
        assert_eq!(est.k, None);
        assert_eq!(cap.strike(), None);
    }

    #[test]
    fn rejects_wrong_stream_bad_value_too_old_and_glitch() {
        let mut cap = capture();
        let wrong_asset = PriceTick {
            asset: Asset::Eth,
            ..live(-100, 100.0)
        };
        let wrong_source = PriceTick {
            source: PriceSource::BinanceDirect,
            ..live(-100, 100.0)
        };
        let wrong_kind = PriceTick {
            kind: TickKind::Mid,
            ..live(-100, 100.0)
        };
        for t in [&wrong_asset, &wrong_source, &wrong_kind] {
            assert_eq!(
                cap.on_tick(t),
                StrikeOutcome::Ignored(StrikeIgnoreReason::WrongStream)
            );
        }
        let zero = PriceTick {
            value: Decimal::ZERO,
            ..live(-100, 1.0)
        };
        assert_eq!(
            cap.on_tick(&zero),
            StrikeOutcome::Ignored(StrikeIgnoreReason::BadValue)
        );
        // 3.5 min before open — beyond the 180 s lookback.
        assert_eq!(
            cap.on_tick(&tick(-210_000, 0, 100.0)),
            StrikeOutcome::Ignored(StrikeIgnoreReason::TooOld)
        );
        // Exchange clock 11 s ahead of local.
        let glitch = PriceTick {
            ts_exchange: TimestampMs::from_millis(OPEN + 11_000),
            ts_local: TimestampMs::from_millis(OPEN),
            ..live(0, 100.0)
        };
        assert_eq!(
            cap.on_tick(&glitch),
            StrikeOutcome::Ignored(StrikeIgnoreReason::ExchangeAheadOfLocal)
        );
        assert!(!cap.is_frozen());
    }

    fn market_with(resolution: &str) -> MarketInfo {
        MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-1781290500".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: TimestampMs::from_millis(OPEN + 300_000),
            strike: None,
            tick_size: TickSize::T01,
            min_order_size: Size::new(dec!(5)).unwrap(),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify(resolution),
        }
    }

    #[test]
    fn for_market_selects_feed_and_rule_per_resolution() {
        let chainlink =
            StrikeCapture::for_market(&market_with("https://data.chain.link/streams/btc-usd"))
                .unwrap();
        assert_eq!(chainlink.params().source, PriceSource::ChainlinkRtds);
        assert_eq!(chainlink.params().kind, TickKind::Vendor);
        assert_eq!(chainlink.params().rule, StrikeRule::LastAtOrBefore);

        let binance =
            StrikeCapture::for_market(&market_with("https://www.binance.com/en/trade/BTC_USDT"))
                .unwrap();
        assert_eq!(binance.params().source, PriceSource::BinanceDirect);
        assert_eq!(binance.params().kind, TickKind::Trade);
        assert_eq!(binance.params().rule, StrikeRule::FirstAtOrAfter);

        let other = StrikeCapture::for_market(&market_with("https://coinbase.com"));
        assert_eq!(
            other.unwrap_err(),
            StrikeError::UnsupportedResolution(ResolutionKind::Other)
        );
    }

    #[test]
    fn first_at_or_after_rule_freezes_on_earliest_after_open() {
        let mut cap = StrikeCapture::new(
            window(),
            Asset::Btc,
            TimestampMs::from_millis(OPEN),
            StrikeParams {
                source: PriceSource::BinanceDirect,
                kind: TickKind::Trade,
                rule: StrikeRule::FirstAtOrAfter,
                ..StrikeParams::default()
            },
        );
        let trade = |ex_off: i64, price: f64| PriceTick {
            source: PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: TickKind::Trade,
            value: Decimal::from_f64(price).unwrap(),
            ts_exchange: TimestampMs::from_millis(OPEN + ex_off),
            ts_local: TimestampMs::from_millis(OPEN + ex_off),
        };
        cap.on_tick(&trade(-200, 100.0)); // before bracket
        // Earliest at-or-after open is the +300 ms trade; it settles and freezes.
        assert_eq!(
            cap.on_tick(&trade(300, 100.2)),
            StrikeOutcome::Frozen {
                value: 100.2,
                offset_ms: 300
            }
        );
        assert_eq!(cap.strike(), Some(100.2));
    }
}
