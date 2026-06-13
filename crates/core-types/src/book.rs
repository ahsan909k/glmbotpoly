//! Order-book types: levels, top-of-book, and full L2 snapshots.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::ids::{ConditionId, TokenId};
use crate::money::Size;
use crate::price::Price;
use crate::time::TimestampMs;

/// One price level of the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookLevel {
    /// Level price.
    pub price: Price,
    /// Displayed size at that price, in shares.
    pub size: Size,
}

/// Best bid / best ask for one outcome token. Either side may be empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopOfBook {
    /// Best bid, if any.
    pub bid: Option<BookLevel>,
    /// Best ask, if any.
    pub ask: Option<BookLevel>,
    /// When this top-of-book was observed.
    pub ts: TimestampMs,
}

impl TopOfBook {
    /// Midpoint of best bid and ask, when both sides exist.
    #[must_use]
    pub fn mid(&self) -> Option<Decimal> {
        let bid = self.bid?.price.as_decimal();
        let ask = self.ask?.price.as_decimal();
        Some((bid + ask) / Decimal::TWO)
    }

    /// Ask minus bid, when both sides exist.
    #[must_use]
    pub fn spread(&self) -> Option<Decimal> {
        let bid = self.bid?.price.as_decimal();
        let ask = self.ask?.price.as_decimal();
        Some(ask - bid)
    }
}

/// Full L2 snapshot for one outcome token. Large — always travels the event
/// bus behind an `Arc` (see [`crate::event::Event::Book`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BookSnapshot {
    /// Token the book belongs to.
    pub token_id: TokenId,
    /// Market (condition id) the token belongs to.
    pub condition_id: ConditionId,
    /// Bid levels, sorted best (highest price) first. Sorting is `feed-clob`'s
    /// contract; consumers may rely on it.
    pub bids: Vec<BookLevel>,
    /// Ask levels, sorted best (lowest price) first.
    pub asks: Vec<BookLevel>,
    /// Snapshot timestamp.
    pub ts: TimestampMs,
    /// The market-channel `hash` field, used for gap detection.
    pub seq_hash: Option<String>,
}

impl BookSnapshot {
    /// Top-of-book view of this snapshot.
    #[must_use]
    pub fn top(&self) -> TopOfBook {
        TopOfBook {
            bid: self.bids.first().copied(),
            ask: self.asks.first().copied(),
            ts: self.ts,
        }
    }
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::tick::{RoundDir, TickSize};

    fn level(p: Decimal, s: Decimal) -> BookLevel {
        BookLevel {
            price: Price::quantize(p, TickSize::T001, RoundDir::Down).unwrap(),
            size: Size::new(s).unwrap(),
        }
    }

    #[test]
    fn mid_and_spread() {
        let top = TopOfBook {
            bid: Some(level(dec!(0.55), dec!(100))),
            ask: Some(level(dec!(0.58), dec!(40))),
            ts: TimestampMs::from_millis(1),
        };
        assert_eq!(top.mid(), Some(dec!(0.565)));
        assert_eq!(top.spread(), Some(dec!(0.03)));
    }

    #[test]
    fn one_sided_book_has_no_mid() {
        let top = TopOfBook {
            bid: Some(level(dec!(0.55), dec!(100))),
            ask: None,
            ts: TimestampMs::from_millis(1),
        };
        assert_eq!(top.mid(), None);
        assert_eq!(top.spread(), None);
        let empty = TopOfBook {
            bid: None,
            ask: None,
            ts: TimestampMs::from_millis(1),
        };
        assert_eq!(empty.mid(), None);
    }

    #[test]
    fn snapshot_top_takes_first_levels() {
        let snap = BookSnapshot {
            token_id: TokenId::new("1").unwrap(),
            condition_id: ConditionId::new(format!("0x{}", "00".repeat(32))).unwrap(),
            bids: vec![level(dec!(0.55), dec!(10)), level(dec!(0.54), dec!(20))],
            asks: vec![level(dec!(0.57), dec!(5))],
            ts: TimestampMs::from_millis(7),
            seq_hash: Some("abc".to_owned()),
        };
        let top = snap.top();
        assert_eq!(top.bid, Some(level(dec!(0.55), dec!(10))));
        assert_eq!(top.ask, Some(level(dec!(0.57), dec!(5))));
        assert_eq!(top.ts, TimestampMs::from_millis(7));
    }
}
