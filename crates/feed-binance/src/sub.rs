//! The direct-Binance stream universe: BTCUSDT/ETHUSDT × top-of-book ticker
//! (`bookTicker`) + trade prints (`trade`). Subscription rides the connect
//! URL (combined-stream form), so unlike RTDS there is no runtime
//! subscription protocol — the set is fixed per driver instance.

use core_types::{Asset, PriceSource, TickKind};

/// Which Binance market stream of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinanceStream {
    /// `<symbol>@bookTicker` — real-time best bid/ask; published as the
    /// midpoint ([`TickKind::Mid`]). Near-continuous on BTC/ETH.
    BookTicker,
    /// `<symbol>@trade` — every trade print ([`TickKind::Trade`]). Gaps for
    /// seconds in quiet markets, hence its looser staleness threshold.
    Trade,
}

impl BinanceStream {
    /// The stream-name suffix on the wire.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::BookTicker => "bookTicker",
            Self::Trade => "trade",
        }
    }
}

/// One (asset, stream) subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinanceSub {
    /// Underlying asset.
    pub asset: Asset,
    /// Which stream of its symbol.
    pub stream: BinanceStream,
}

impl BinanceSub {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(asset: Asset, stream: BinanceStream) -> Self {
        Self { asset, stream }
    }

    /// All four streams the bot consumes (BTC/ETH × bookTicker/trade), in
    /// deterministic order.
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut subs = Vec::with_capacity(4);
        for asset in Asset::ALL {
            for stream in [BinanceStream::BookTicker, BinanceStream::Trade] {
                subs.push(Self::new(asset, stream));
            }
        }
        subs
    }

    /// The lowercase spot symbol (stream names require lowercase).
    #[must_use]
    pub const fn symbol(asset: Asset) -> &'static str {
        match asset {
            Asset::Btc => "btcusdt",
            Asset::Eth => "ethusdt",
        }
    }

    /// The full wire stream name, e.g. `btcusdt@bookTicker`.
    #[must_use]
    pub fn stream_name(&self) -> String {
        format!("{}@{}", Self::symbol(self.asset), self.stream.suffix())
    }
}

impl std::fmt::Display for BinanceSub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", Self::symbol(self.asset), self.stream.suffix())
    }
}

impl feed_util::StreamKey for BinanceSub {
    fn source(&self) -> PriceSource {
        PriceSource::BinanceDirect
    }

    fn asset(&self) -> Asset {
        self.asset
    }

    fn kind(&self) -> TickKind {
        match self.stream {
            BinanceStream::BookTicker => TickKind::Mid,
            BinanceStream::Trade => TickKind::Trade,
        }
    }
}

#[cfg(test)]
mod tests {
    use feed_util::StreamKey;

    use super::*;

    #[test]
    fn all_covers_both_assets_and_streams_in_order() {
        let all = BinanceSub::all();
        assert_eq!(all.len(), 4);
        let names: Vec<String> = all.iter().map(BinanceSub::stream_name).collect();
        assert_eq!(
            names,
            vec![
                "btcusdt@bookTicker",
                "btcusdt@trade",
                "ethusdt@bookTicker",
                "ethusdt@trade"
            ]
        );
    }

    #[test]
    fn stream_key_identity_maps_kinds() {
        let book = BinanceSub::new(Asset::Btc, BinanceStream::BookTicker);
        assert_eq!(book.source(), PriceSource::BinanceDirect);
        assert_eq!(book.kind(), TickKind::Mid);
        let trade = BinanceSub::new(Asset::Eth, BinanceStream::Trade);
        assert_eq!(trade.asset(), Asset::Eth);
        assert_eq!(trade.kind(), TickKind::Trade);
        assert_eq!(trade.to_string(), "ethusdt@trade");
    }
}
