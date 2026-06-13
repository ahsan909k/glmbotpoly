//! Window/market identity and per-market metadata.

use std::fmt;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::ids::{ConditionId, TokenId};
use crate::money::Size;
use crate::series::Series;
use crate::tick::TickSize;
use crate::time::TimestampMs;

/// One of the two outcomes of an Up/Down market.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Outcome {
    /// Resolves Up (end price ≥ price-to-beat; ties go Up).
    Up,
    /// Resolves Down.
    Down,
}

impl Outcome {
    /// Both outcomes.
    pub const ALL: [Self; 2] = [Self::Up, Self::Down];

    /// The other outcome.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Up => "Up",
            Self::Down => "Down",
        })
    }
}

/// The two outcome-token ids of one window market, explicitly labeled so the
/// Up/Down mapping can never be positional guesswork.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenPair {
    /// Token id of the Up outcome.
    pub up: TokenId,
    /// Token id of the Down outcome.
    pub down: TokenId,
}

impl TokenPair {
    /// The token id for an outcome.
    #[must_use]
    pub const fn get(&self, outcome: Outcome) -> &TokenId {
        match outcome {
            Outcome::Up => &self.up,
            Outcome::Down => &self.down,
        }
    }

    /// Which outcome a token id belongs to, if it belongs to this pair.
    #[must_use]
    pub fn outcome_of(&self, token: &TokenId) -> Option<Outcome> {
        if token == &self.up {
            Some(Outcome::Up)
        } else if token == &self.down {
            Some(Outcome::Down)
        } else {
            None
        }
    }
}

/// Identity of one tradable window: the series plus its open instant.
/// Cheap `Copy` key for maps, journal rows, and dashboard grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WindowId {
    /// Which repeating product this window belongs to.
    pub series: Series,
    /// Window open time (unix millis).
    pub open_time: TimestampMs,
}

impl fmt::Display for WindowId {
    /// Stable key form, e.g. `BTC-5m@1718000000000`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.series, self.open_time)
    }
}

/// Per-market fee parameters, read from the market object's `feeSchedule`
/// descriptor at discovery. Never hardcode the crypto-category 0.07 in logic
/// (CLAUDE.md §7) — this struct is the only place fee rates enter the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeeParams {
    /// Taker fee rate (`feeSchedule.rate`), e.g. `0.07` for the crypto
    /// category.
    pub rate: Decimal,
    /// Fee curve exponent (`feeSchedule.exponent`; observed `1`, reserved by
    /// the protocol for future curves).
    pub exponent: u32,
    /// `feeSchedule.takerOnly` — makers pay zero when `true`.
    pub taker_only: bool,
    /// `feeSchedule.rebateRate` — maker rebate share of collected taker fees
    /// (`0.2` for crypto), needed by the paper venue's rebate estimator.
    pub rebate_rate: Decimal,
    /// The market object's `feesEnabled` flag.
    pub enabled: bool,
}

/// How a market's resolution-source text was classified at discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResolutionKind {
    /// Chainlink data stream (`data.chain.link` URLs) — verified live for
    /// the 5m/15m series.
    ChainlinkDataStream,
    /// Binance spot candle (`binance.com` URLs) — verified live for the 1h
    /// series (close ≥ open of the 1H candle).
    BinanceCandle,
    /// Anything else. The model must not anchor on a feed it cannot
    /// identify; treat such windows as untradable.
    Other,
}

/// The per-market resolution source, parsed from the market object at
/// discovery and stored rather than assumed (CLAUDE.md §6): the 1h series
/// resolves on Binance candles while 5m/15m resolve on Chainlink streams.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResolutionSource {
    /// Parsed feed kind the model should anchor on.
    pub kind: ResolutionKind,
    /// Raw `resolutionSource` text from the market object, verbatim (kept
    /// for the journal, dashboard, and dispute forensics).
    pub raw: String,
}

impl ResolutionSource {
    /// Classifies a raw resolution-source string by case-insensitive
    /// substring: `chain.link` → Chainlink data stream, `binance.com` →
    /// Binance candle, anything else (including empty) → [`ResolutionKind::Other`].
    #[must_use]
    pub fn classify(raw: impl Into<String>) -> Self {
        let raw = raw.into();
        let lower = raw.to_ascii_lowercase();
        let kind = if lower.contains("chain.link") {
            ResolutionKind::ChainlinkDataStream
        } else if lower.contains("binance.com") {
            ResolutionKind::BinanceCandle
        } else {
            ResolutionKind::Other
        };
        Self { kind, raw }
    }
}

/// Everything the engine needs to trade one window market.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketInfo {
    /// Window identity (series + open time; open time lives here only).
    pub window: WindowId,
    /// Gamma event slug, e.g. `btc-updown-5m-1781166600` — the human/URL
    /// identity of the window (`polymarket.com/event/{slug}`), for the
    /// dashboard, journal, and operator verification.
    pub event_slug: String,
    /// Polymarket condition id of the market.
    pub condition_id: ConditionId,
    /// Up/Down outcome token ids.
    pub tokens: TokenPair,
    /// Window close (resolution) time.
    pub close_time: TimestampMs,
    /// The price-to-beat: resolution-source price captured at window open.
    /// `None` until captured. An *underlying* price (e.g. ≈104000 for BTC),
    /// so a plain [`Decimal`], never a [`crate::Price`].
    pub strike: Option<Decimal>,
    /// Current tick size. Mutated upstream on `tick_size_change` events —
    /// always re-validate prices against this before placing orders.
    pub tick_size: TickSize,
    /// Venue minimum resting order size in shares (market object
    /// `minimum_order_size`). The engine normalizer combines this with the
    /// $1-notional minimum.
    pub min_order_size: Size,
    /// Per-market fee parameters.
    pub fees: FeeParams,
    /// Market `neg_risk` flag — expected `false` for these binaries; stored
    /// because the order builder needs it and we verify per market.
    pub neg_risk: bool,
    /// Resolution source parsed from the market object at discovery (§6) —
    /// the 1h series resolves on Binance candles, 5m/15m on Chainlink.
    pub resolution: ResolutionSource,
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::series::{Asset, WindowDuration};

    fn pair() -> TokenPair {
        TokenPair {
            up: TokenId::new("111").unwrap(),
            down: TokenId::new("222").unwrap(),
        }
    }

    #[test]
    fn opposite_is_involutive() {
        for o in Outcome::ALL {
            assert_eq!(o.opposite().opposite(), o);
        }
        assert_eq!(Outcome::Up.opposite(), Outcome::Down);
    }

    #[test]
    fn token_pair_lookups() {
        let p = pair();
        assert_eq!(p.get(Outcome::Up).as_str(), "111");
        assert_eq!(p.get(Outcome::Down).as_str(), "222");
        assert_eq!(
            p.outcome_of(&TokenId::new("111").unwrap()),
            Some(Outcome::Up)
        );
        assert_eq!(
            p.outcome_of(&TokenId::new("222").unwrap()),
            Some(Outcome::Down)
        );
        assert_eq!(p.outcome_of(&TokenId::new("333").unwrap()), None);
    }

    #[test]
    fn window_id_display_is_stable() {
        let w = WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_718_000_000_000),
        };
        assert_eq!(w.to_string(), "BTC-5m@1718000000000");
    }

    #[test]
    fn market_info_serde_roundtrip() {
        let info = MarketInfo {
            window: WindowId {
                series: Series {
                    asset: Asset::Eth,
                    duration: WindowDuration::M15,
                },
                open_time: TimestampMs::from_millis(1_718_000_000_000),
            },
            event_slug: "eth-updown-15m-1718000000".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "ab".repeat(32))).unwrap(),
            tokens: pair(),
            close_time: TimestampMs::from_millis(1_718_000_900_000),
            strike: Some(dec!(2650.12345678)),
            tick_size: TickSize::T001,
            min_order_size: Size::new(dec!(5)).unwrap(),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/eth-usd"),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: MarketInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back, info);
    }

    #[test]
    fn classify_recognizes_chainlink() {
        let src = ResolutionSource::classify("https://data.chain.link/streams/btc-usd");
        assert_eq!(src.kind, ResolutionKind::ChainlinkDataStream);
        assert_eq!(src.raw, "https://data.chain.link/streams/btc-usd");
    }

    #[test]
    fn classify_recognizes_binance() {
        let src = ResolutionSource::classify("https://www.binance.com/en/trade/BTC_USDT");
        assert_eq!(src.kind, ResolutionKind::BinanceCandle);
        assert_eq!(src.raw, "https://www.binance.com/en/trade/BTC_USDT");
    }

    #[test]
    fn classify_is_case_insensitive() {
        let src = ResolutionSource::classify("HTTPS://DATA.CHAIN.LINK/STREAMS/BTC-USD");
        assert_eq!(src.kind, ResolutionKind::ChainlinkDataStream);
        // Raw text is preserved verbatim, not lowercased.
        assert_eq!(src.raw, "HTTPS://DATA.CHAIN.LINK/STREAMS/BTC-USD");
    }

    #[test]
    fn classify_unknown_and_empty_are_other() {
        assert_eq!(
            ResolutionSource::classify("https://www.coinbase.com").kind,
            ResolutionKind::Other
        );
        assert_eq!(ResolutionSource::classify("").kind, ResolutionKind::Other);
    }

    #[test]
    fn resolution_source_serde_roundtrip() {
        let src = ResolutionSource::classify("https://www.binance.com/en/trade/ETH_USDT");
        let json = serde_json::to_string(&src).unwrap();
        let back: ResolutionSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back, src);
    }
}
