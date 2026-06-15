//! The serializable projection of the bus [`Event`].
//!
//! [`core_types::Event`] is deliberately **not** `Serialize` (its variants wrap
//! `Arc`s; serializing them would buy nothing). Every *inner* payload struct is
//! serde, though, so [`JournalRecord`] mirrors `Event` one-for-one, holding the
//! inner payloads by value (deref the `Arc`s on the way in, re-wrap on the way
//! out). This is the only place besides `bot` that names `Event` and reaches
//! through its `Arc`s — exactly `journal`'s job.
//!
//! Wire shape: an internally-tagged enum (`{"type":"last_trade", …}`), the
//! `CalRecord` precedent in `bot fair`. One [`RecordEnvelope`] per JSONL line.

use std::sync::Arc;

use core_types::{
    BookHealth, BookSnapshot, ConditionId, ControlEvent, Event, FeedHealth, Fill,
    InventorySnapshot, MarketInfo, ModelHealthEvent, ModelSnapshot, OrderUpdate, Price, PriceTick,
    RiskEvent, SettlementSummary, Side, Size, TickSize, TimestampMs, TokenId, TopOfBook,
    WindowLifecycle,
};
use serde::{Deserialize, Serialize};

/// A serializable mirror of one [`Event`].
///
/// The variant set is **exhaustive** over `Event` on purpose — `Event` is not
/// `#[non_exhaustive]`, so a new bus variant forces [`JournalRecord::from_event`]
/// (and this enum) to be updated rather than silently dropping the event. The
/// wide payloads (`Book`, `Window`/`MarketInfo`, `OrderUpdate`, `Fill`,
/// `Inventory`) are boxed so the enum stays small; boxing is transparent to
/// serde, so the on-wire shape is unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JournalRecord {
    /// [`Event::PriceTick`].
    PriceTick(PriceTick),
    /// [`Event::FeedHealth`].
    FeedHealth(FeedHealth),
    /// [`Event::BookHealth`].
    BookHealth(BookHealth),
    /// [`Event::Book`].
    Book(Box<BookSnapshot>),
    /// [`Event::TopOfBook`].
    TopOfBook {
        /// Token the update belongs to.
        token_id: TokenId,
        /// The new top of book.
        top: TopOfBook,
    },
    /// [`Event::TickSizeChange`].
    TickSizeChange {
        /// Market whose tick changed.
        condition_id: ConditionId,
        /// The new tick.
        new_tick: TickSize,
        /// Event time.
        ts: TimestampMs,
    },
    /// [`Event::LastTrade`].
    LastTrade {
        /// Token that traded.
        token_id: TokenId,
        /// Print price.
        price: Price,
        /// Print size.
        size: Size,
        /// Aggressor side as reported.
        side: Side,
        /// Print time.
        ts: TimestampMs,
    },
    /// [`Event::Window`].
    Window {
        /// The window's market.
        market: Box<MarketInfo>,
        /// New lifecycle phase.
        lifecycle: WindowLifecycle,
    },
    /// [`Event::OrderUpdate`].
    OrderUpdate(Box<OrderUpdate>),
    /// [`Event::Fill`].
    Fill(Box<Fill>),
    /// [`Event::Model`].
    Model(ModelSnapshot),
    /// [`Event::ModelHealth`].
    ModelHealth(ModelHealthEvent),
    /// [`Event::Inventory`].
    Inventory(Box<InventorySnapshot>),
    /// [`Event::Settlement`].
    Settlement(Box<SettlementSummary>),
    /// [`Event::Risk`].
    Risk(RiskEvent),
    /// [`Event::Control`].
    Control(ControlEvent),
}

impl JournalRecord {
    /// Projects a bus [`Event`] into its serializable record (clones the inner
    /// payload out of the `Arc`).
    #[must_use]
    pub fn from_event(event: &Event) -> Self {
        match event {
            Event::PriceTick(t) => Self::PriceTick(*t),
            Event::FeedHealth(h) => Self::FeedHealth(*h),
            Event::BookHealth(h) => Self::BookHealth(*h),
            Event::Book(snap) => Self::Book(Box::new((**snap).clone())),
            Event::TopOfBook { token_id, top } => Self::TopOfBook {
                token_id: (**token_id).clone(),
                top: *top,
            },
            Event::TickSizeChange {
                condition_id,
                new_tick,
                ts,
            } => Self::TickSizeChange {
                condition_id: (**condition_id).clone(),
                new_tick: *new_tick,
                ts: *ts,
            },
            Event::LastTrade {
                token_id,
                price,
                size,
                side,
                ts,
            } => Self::LastTrade {
                token_id: (**token_id).clone(),
                price: *price,
                size: *size,
                side: *side,
                ts: *ts,
            },
            Event::Window { market, lifecycle } => Self::Window {
                market: Box::new((**market).clone()),
                lifecycle: *lifecycle,
            },
            Event::OrderUpdate(u) => Self::OrderUpdate(Box::new((**u).clone())),
            Event::Fill(f) => Self::Fill(Box::new((**f).clone())),
            Event::Model(m) => Self::Model(*m),
            Event::ModelHealth(h) => Self::ModelHealth(*h),
            Event::Inventory(i) => Self::Inventory(Box::new((**i).clone())),
            Event::Settlement(s) => Self::Settlement(Box::new((**s).clone())),
            Event::Risk(r) => Self::Risk(*r),
            Event::Control(c) => Self::Control(c.clone()),
        }
    }

    /// Reconstructs a bus [`Event`] from this record (re-wraps the payload in an
    /// `Arc`). Lossless for every variant — in particular for the five the paper
    /// venue consumes (`Book`/`TopOfBook`/`LastTrade`/`TickSizeChange`/`Window`),
    /// so a recorded session replays bit-exactly into `PaperVenue::on_bus_event`.
    #[must_use]
    pub fn to_event(&self) -> Event {
        match self {
            Self::PriceTick(t) => Event::PriceTick(*t),
            Self::FeedHealth(h) => Event::FeedHealth(*h),
            Self::BookHealth(h) => Event::BookHealth(*h),
            Self::Book(snap) => Event::Book(Arc::new((**snap).clone())),
            Self::TopOfBook { token_id, top } => Event::TopOfBook {
                token_id: Arc::new(token_id.clone()),
                top: *top,
            },
            Self::TickSizeChange {
                condition_id,
                new_tick,
                ts,
            } => Event::TickSizeChange {
                condition_id: Arc::new(condition_id.clone()),
                new_tick: *new_tick,
                ts: *ts,
            },
            Self::LastTrade {
                token_id,
                price,
                size,
                side,
                ts,
            } => Event::LastTrade {
                token_id: Arc::new(token_id.clone()),
                price: *price,
                size: *size,
                side: *side,
                ts: *ts,
            },
            Self::Window { market, lifecycle } => Event::Window {
                market: Arc::new((**market).clone()),
                lifecycle: *lifecycle,
            },
            Self::OrderUpdate(u) => Event::OrderUpdate(Arc::new((**u).clone())),
            Self::Fill(f) => Event::Fill(Arc::new((**f).clone())),
            Self::Model(m) => Event::Model(*m),
            Self::ModelHealth(h) => Event::ModelHealth(*h),
            Self::Inventory(i) => Event::Inventory(Arc::new((**i).clone())),
            Self::Settlement(s) => Event::Settlement(Arc::new((**s).clone())),
            Self::Risk(r) => Event::Risk(*r),
            Self::Control(c) => Event::Control(c.clone()),
        }
    }
}

/// One journal line: a global monotonic sequence number, the local write time,
/// and the record. JSONL — one of these per line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordEnvelope {
    /// Global monotonic sequence number (gap detector + stable tiebreak across
    /// segments).
    pub seq: u64,
    /// Local wall-clock time the recorder stamped this line, unix millis.
    pub ts_local_ms: i64,
    /// The projected event.
    pub rec: JournalRecord,
}

#[cfg(test)]
mod tests {
    use core_types::{
        Asset, BookLevel, Decimal, Dollars, FeeParams, Liquidity, OrderState, Outcome,
        ResolutionSource, RoundDir, Series, TokenPair, WindowDuration, WindowId,
    };
    use rust_decimal::dec;

    use super::*;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }
    fn token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn cid() -> ConditionId {
        ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap()
    }
    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: ts(1_781_000_000_000),
        }
    }
    fn market() -> MarketInfo {
        MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: cid(),
            tokens: TokenPair {
                up: token(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: ts(1_781_000_300_000),
            strike: Some(dec!(60000)),
            tick_size: TickSize::T001,
            min_order_size: sz(dec!(5)),
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

    /// A representative event of every bus variant, for the exhaustive
    /// round-trip sweep.
    fn sample_events() -> Vec<Event> {
        use core_types::{AnchorSource, InputAges};
        use core_types::{
            BookSnapshot, BookUnreliableReason, BreakerKind, ControlEvent, FeedHealth, Fill,
            InventorySnapshot, ModelHealth, ModelHealthEvent, ModelHealthReason, ModelSnapshot,
            OrderUpdate, PriceSource, PriceTick, RiskEvent, SettlementSummary, SideInventory,
            TickKind,
        };

        let order = OrderUpdate {
            order_id: core_types::OrderId::new("paper-1").unwrap(),
            window: window(),
            token_id: token(),
            side: Side::Buy,
            state: OrderState::PartiallyFilled,
            price: px(dec!(0.49)),
            original_size: sz(dec!(20)),
            filled_size: sz(dec!(10)),
            reject_reason: None,
            ts_venue: Some(ts(1_781_000_000_100)),
            ts_local: ts(1_781_000_000_100),
        };
        let fill = Fill {
            order_id: core_types::OrderId::new("paper-1").unwrap(),
            trade_id: Some("paper-t1".to_owned()),
            window: window(),
            token_id: token(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: px(dec!(0.49)),
            size: sz(dec!(10)),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: ts(1_781_000_000_100),
            ts_local: ts(1_781_000_000_100),
        };
        let inv = InventorySnapshot::derive(
            window(),
            SideInventory {
                shares: sz(dec!(10)),
                cost: Dollars::new(dec!(4.9)),
            },
            SideInventory {
                shares: sz(dec!(0)),
                cost: Dollars::ZERO,
            },
            ts(1_781_000_000_100),
        );
        let settlement = SettlementSummary::close(
            window(),
            Outcome::Up,
            SideInventory {
                shares: sz(dec!(10)),
                cost: Dollars::new(dec!(4.9)),
            },
            SideInventory {
                shares: sz(dec!(8)),
                cost: Dollars::new(dec!(4.0)),
            },
            sz(dec!(2)),
            Dollars::new(dec!(0.13)),
            Dollars::new(dec!(1.10)),
            ts(1_781_000_300_000),
        );
        let model = ModelSnapshot {
            asset: Asset::Btc,
            window: Some(window()),
            p_up: 0.51,
            z: 0.04,
            sigma_1s: 0.0004,
            sigma_tau: 0.02,
            basis: -4.7,
            anchor: AnchorSource::BinanceCorrected,
            health: ModelHealth::Ready,
            reason: ModelHealthReason::Nominal,
            input_ages: InputAges {
                chainlink: core_types::DurationMs::from_millis(120),
                binance: core_types::DurationMs::from_millis(20),
            },
            ts: ts(1_781_000_000_100),
        };
        vec![
            Event::PriceTick(PriceTick {
                source: PriceSource::BinanceDirect,
                asset: Asset::Btc,
                kind: TickKind::Mid,
                value: dec!(60123.45),
                ts_exchange: ts(1_781_000_000_000),
                ts_local: ts(1_781_000_000_010),
            }),
            Event::FeedHealth(FeedHealth::Stale {
                source: PriceSource::ChainlinkRtds,
                asset: Asset::Eth,
                kind: TickKind::Vendor,
                age: core_types::DurationMs::from_millis(5_250),
            }),
            Event::BookHealth(BookHealth::Unreliable {
                window: window(),
                reason: BookUnreliableReason::Stale,
                ts: ts(1_781_000_000_100),
            }),
            Event::Book(Arc::new(BookSnapshot {
                token_id: token(),
                condition_id: cid(),
                bids: vec![BookLevel {
                    price: px(dec!(0.49)),
                    size: sz(dec!(30)),
                }],
                asks: vec![BookLevel {
                    price: px(dec!(0.51)),
                    size: sz(dec!(50)),
                }],
                ts: ts(1_781_000_000_000),
                seq_hash: Some("abc".to_owned()),
            })),
            Event::TopOfBook {
                token_id: Arc::new(token()),
                top: TopOfBook {
                    bid: Some(BookLevel {
                        price: px(dec!(0.49)),
                        size: sz(dec!(30)),
                    }),
                    ask: Some(BookLevel {
                        price: px(dec!(0.51)),
                        size: sz(dec!(50)),
                    }),
                    ts: ts(1_781_000_000_000),
                },
            },
            Event::TickSizeChange {
                condition_id: Arc::new(cid()),
                new_tick: TickSize::T0001,
                ts: ts(1_781_000_000_050),
            },
            Event::LastTrade {
                token_id: Arc::new(token()),
                price: px(dec!(0.49)),
                size: sz(dec!(40)),
                side: Side::Sell,
                ts: ts(1_781_000_000_080),
            },
            Event::Window {
                market: Arc::new(market()),
                lifecycle: WindowLifecycle::Open,
            },
            Event::Window {
                market: Arc::new(market()),
                lifecycle: WindowLifecycle::Resolved {
                    outcome: Outcome::Up,
                },
            },
            Event::OrderUpdate(Arc::new(order)),
            Event::Fill(Arc::new(fill)),
            Event::Model(model),
            Event::ModelHealth(ModelHealthEvent {
                asset: Asset::Btc,
                health: ModelHealth::Degraded,
                reason: ModelHealthReason::FastLeadLost,
                ts: ts(1_781_000_000_100),
            }),
            Event::Inventory(Arc::new(inv)),
            Event::Settlement(Arc::new(settlement)),
            Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew,
            }),
            Event::Control(ControlEvent::PaperCapitalSet {
                amount: Dollars::new(dec!(10000)),
            }),
            Event::Control(ControlEvent::Kill),
            Event::Control(ControlEvent::Reset),
        ]
    }

    #[test]
    fn every_variant_round_trips_through_json_and_back_to_event() {
        for event in sample_events() {
            let rec = JournalRecord::from_event(&event);
            let line = serde_json::to_string(&rec).expect("serialize record");
            // Internally tagged: every line carries a "type" discriminator.
            assert!(line.contains("\"type\":"), "no type tag in {line}");
            let back: JournalRecord = serde_json::from_str(&line).expect("deserialize record");
            assert_eq!(back, rec, "record round-trip mismatch");
            // The full Event reconstruction is lossless (Event derives PartialEq;
            // Arc compares by value).
            assert_eq!(back.to_event(), event, "event reconstruction mismatch");
        }
    }

    #[test]
    fn envelope_is_one_flat_line_with_nested_record() {
        let event = Event::LastTrade {
            token_id: Arc::new(token()),
            price: px(dec!(0.49)),
            size: sz(dec!(40)),
            side: Side::Sell,
            ts: ts(1_781_000_000_080),
        };
        let env = RecordEnvelope {
            seq: 7,
            ts_local_ms: 1_781_000_000_090,
            rec: JournalRecord::from_event(&event),
        };
        let line = serde_json::to_string(&env).expect("serialize envelope");
        assert!(!line.contains('\n'), "must be one line");
        assert!(line.contains("\"seq\":7"));
        assert!(line.contains("\"type\":\"last_trade\""));
        let back: RecordEnvelope = serde_json::from_str(&line).expect("deserialize envelope");
        assert_eq!(back, env);
    }

    #[test]
    fn paper_consumed_variants_reconstruct_exactly() {
        // The five variants PaperVenue::on_bus_event acts on must replay exactly.
        for event in sample_events() {
            match &event {
                Event::Book(_)
                | Event::TopOfBook { .. }
                | Event::LastTrade { .. }
                | Event::TickSizeChange { .. }
                | Event::Window { .. } => {
                    let rec = JournalRecord::from_event(&event);
                    assert_eq!(rec.to_event(), event);
                }
                _ => {}
            }
        }
    }
}
