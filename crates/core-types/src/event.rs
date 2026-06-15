//! The internal event bus vocabulary (CLAUDE.md §5): one enum every
//! long-running task publishes and consumes over typed channels.
//!
//! Design rules:
//! - **Cloning is O(1)-ish.** Every variant payload is either `Copy`-small or
//!   `Arc`-wrapped; a guard test pins `size_of::<Event>()` so a `Vec` payload
//!   can never sneak in.
//! - **Not `Serialize`.** The journal persists the inner payload structs
//!   (which all are serde) — serializing `Arc`s on the bus type would buy
//!   nothing and complicate sharing.
//! - **Exhaustive on purpose** (no `#[non_exhaustive]`): adding a variant
//!   must force every consumer `match` to decide how to handle it.

use std::sync::Arc;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::book::{BookSnapshot, TopOfBook};
use crate::ids::{ConditionId, TokenId};
use crate::inventory::{InventorySnapshot, SettlementSummary};
use crate::market::{MarketInfo, Outcome, WindowId};
use crate::mode::Mode;
use crate::model_state::{ModelHealth, ModelHealthReason, ModelSnapshot};
use crate::money::{Dollars, Size};
use crate::order::{Fill, OrderUpdate, Side};
use crate::price::Price;
use crate::series::{Asset, Series};
use crate::tick::TickSize;
use crate::time::{DurationMs, TimestampMs};

/// Which upstream produced a price observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PriceSource {
    /// RTDS `crypto_prices_chainlink` topic — the resolution-grade feed.
    ChainlinkRtds,
    /// RTDS `crypto_prices` topic (Binance-sourced).
    BinanceRtds,
    /// Direct Binance WebSocket — the lowest-latency signal.
    BinanceDirect,
}

/// What flavor of observation a price stream carries. Streams of different
/// kinds from one source are independent for staleness purposes (a quiet
/// trade stream is a quiet market; a quiet top-of-book stream is an outage),
/// and the feed-comparison diagnostics need to know which flavor a vendor
/// feed matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TickKind {
    /// Whatever the upstream vendor publishes — flavor unspecified by the
    /// wire (RTDS topics carry a bare `value` with no provenance).
    Vendor,
    /// Top-of-book midpoint: (best bid + best ask) / 2.
    Mid,
    /// A trade print.
    Trade,
}

/// One underlying price observation. Value kept wire-exact as [`Decimal`];
/// the model converts to `f64` internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriceTick {
    /// Producing feed.
    pub source: PriceSource,
    /// Underlying asset.
    pub asset: Asset,
    /// Observation flavor.
    pub kind: TickKind,
    /// Price in underlying units (e.g. USD per BTC).
    pub value: Decimal,
    /// Exchange/source timestamp from the wire.
    pub ts_exchange: TimestampMs,
    /// Local receive time.
    pub ts_local: TimestampMs,
}

/// Per-(source, asset, kind) price-feed health transitions.
///
/// Emitted by the feed crates' staleness watchdogs; consumed by the model
/// (input ages feed its UNRELIABLE state), the dashboard feed panel (§10.5),
/// and the future risk manager. Deliberately **not** a [`RiskEvent`]: breaker
/// semantics (`BreakerKind::FeedStale` + cancel-all per §11) have a single
/// writer — the risk manager — which decides its own policy over these
/// observations. Feeds report; risk reacts.
///
/// Ages are measured on local receive time (pipeline liveness). Data recency
/// (`ts_exchange`) is a separate dimension judged by consumers per tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedHealth {
    /// No tick from this (source, asset, kind) stream for at least its
    /// staleness threshold — or its connection dropped (definitive evidence,
    /// reported immediately).
    Stale {
        /// Producing feed.
        source: PriceSource,
        /// Underlying asset.
        asset: Asset,
        /// Observation flavor (a stale `Trade` stream is far weaker evidence
        /// of trouble than a stale `Mid` or `Vendor` stream).
        kind: TickKind,
        /// Time since the last tick (or since the subscription was anchored,
        /// if it never delivered) when staleness was declared.
        age: DurationMs,
    },
    /// A previously stale stream delivered a tick again.
    Recovered {
        /// Producing feed.
        source: PriceSource,
        /// Underlying asset.
        asset: Asset,
        /// Observation flavor.
        kind: TickKind,
        /// Length of the outage: time between the last tick before the gap
        /// (or the subscription anchor) and the recovering tick.
        gap: DurationMs,
    },
}

/// Why a window's L2 books are currently untrustworthy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BookUnreliableReason {
    /// The market-channel connection for this window is down (definitive
    /// evidence, reported immediately).
    Disconnected,
    /// No book/price_change/best_bid_ask traffic for an *active* window's
    /// token within the staleness threshold (quiet pre-open/post-close books
    /// are exempt — silence there is normal).
    Stale,
    /// A venue `book` snapshot itself arrived crossed (best bid ≥ best ask)
    /// — venue-side garbage; trust is withheld until a clean snapshot.
    /// (Crossing *deltas* are normal — they are trades at the touch and are
    /// resolved by implied consumption, never flagged.)
    Crossed,
    /// The derived top-of-book disagreed with the venue-reported best
    /// bid/ask beyond the divergence grace period.
    TopDivergence,
}

/// Per-window L2 book health transitions from feed-clob.
///
/// Same contract as [`FeedHealth`]: feeds report, the (future) risk manager
/// reacts (§11 cancel-all on market-WS trouble has a single writer). Latched:
/// one `Unreliable` per outage episode, one `Recovered` once a trusted
/// snapshot is re-established — never a stream of repeats.
///
/// Granularity is the *window* (both outcome tokens ride one connection and
/// the engine trades them as a pair), not the individual token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BookHealth {
    /// The window's books stopped being trustworthy.
    Unreliable {
        /// Affected window.
        window: WindowId,
        /// What broke.
        reason: BookUnreliableReason,
        /// When it was detected (local time).
        ts: TimestampMs,
    },
    /// A fresh venue snapshot re-established trusted books.
    Recovered {
        /// Affected window.
        window: WindowId,
        /// Length of the unreliable episode.
        outage: DurationMs,
        /// When trust was re-established (local time).
        ts: TimestampMs,
    },
}

/// A composite model-health transition for one asset (§8).
///
/// Same report/react contract as [`FeedHealth`] / [`BookHealth`]: the model
/// **reports** its gate changes here; the engine and the future risk manager
/// **react** (refuse to quote on anything but [`ModelHealth::Ready`], pull
/// quotes on [`ModelHealth::Unreliable`]). Latched — emitted exactly once per
/// tier change, never a stream of repeats — so a consumer matches one event
/// rather than inspecting every [`Event::Model`] snapshot. The continuous
/// `health`/`reason` still ride every [`ModelSnapshot`] for the dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHealthEvent {
    /// Asset whose model changed tier.
    pub asset: Asset,
    /// The new health tier.
    pub health: ModelHealth,
    /// The cause behind the new tier (`reason.tier() == health`).
    pub reason: ModelHealthReason,
    /// When the transition was detected (local time).
    pub ts: TimestampMs,
}

/// Window lifecycle phases announced by the scheduler (§6).
///
/// Per-window emission contract: `Discovered → Open → Closing → Closed →
/// Resolved`. Windows are contiguous back-to-back, so the *next* window's
/// `Open` is emitted at the same instant as the current one's `Closed`;
/// `Resolved` arrives separately — possibly well after the successor opened —
/// once the venue reports the outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowLifecycle {
    /// Market discovered and metadata cached; not yet open.
    Discovered,
    /// Window is open and tradable.
    Open,
    /// Final-seconds phase: §8 end-of-window quoting rules apply.
    Closing,
    /// Past close time: trading must have stopped; outcome not yet known.
    Closed,
    /// Window resolved with the given outcome.
    Resolved {
        /// Winning outcome.
        outcome: Outcome,
    },
}

/// The §11 circuit breakers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BreakerKind {
    /// A feed exceeded the staleness bound (500 ms default).
    FeedStale,
    /// Market- or user-WebSocket disconnected.
    WsDisconnect,
    /// Clock-skew alarm from timeutil.
    ClockSkew,
    /// Daily stop-loss reached — halt until manual reset.
    DailyStop,
    /// Per-window maximum loss reached.
    WindowLoss,
    /// API error-rate circuit breaker.
    ErrorRate,
    /// |model fair − book mid| sanity bound exceeded too long.
    FairVsMid,
    /// Operator-initiated kill (dashboard or CLI).
    Manual,
    /// Matching-engine restart detected — orders may be gone, reconcile.
    EngineRestart,
}

/// Risk-manager notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskEvent {
    /// A breaker fired.
    BreakerTripped {
        /// Which breaker.
        breaker: BreakerKind,
    },
    /// A breaker reset.
    BreakerCleared {
        /// Which breaker.
        breaker: BreakerKind,
    },
    /// A cancel-all was issued on a safety path.
    CancelAllIssued {
        /// The breaker that caused it.
        reason: BreakerKind,
    },
}

/// Operator/control-plane notifications.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlEvent {
    /// Session mode announced/changed.
    ModeChanged {
        /// New mode.
        mode: Mode,
    },
    /// A series was enabled at runtime.
    SeriesEnabled {
        /// Which series.
        series: Series,
    },
    /// A series was disabled at runtime.
    SeriesDisabled {
        /// Which series.
        series: Series,
    },
    /// Paper starting capital adjusted from the dashboard (journaled).
    PaperCapitalSet {
        /// New paper capital.
        amount: Dollars,
    },
    /// Graceful shutdown requested: stop strategies → cancel all → flush.
    Shutdown,
    /// Operator manual kill (dashboard or CLI): halt all trading and cancel
    /// everything immediately, latched until a [`Reset`](Self::Reset). The risk
    /// manager trips [`BreakerKind::Manual`] on it (§11).
    Kill,
    /// Operator reset: clears the operator-latched breakers
    /// ([`BreakerKind::Manual`] and [`BreakerKind::DailyStop`]) so trading can
    /// resume after a manual kill or a daily stop (§11 "until manual reset").
    Reset,
}

/// Market lifecycle notices from the market-channel WebSocket (`new_market` /
/// `market_resolved`, unlocked by `custom_feature_enabled` — CLAUDE.md §7).
/// Constructed by feed-clob, consumed by the scheduler on a dedicated typed
/// channel; deliberately **not** a bus [`Event`] variant because it is
/// scheduler *input*, not an announcement.
///
/// Deliberately minimal: the wire `new_market` payload carries no window
/// open/close times, so it can only ever be a refresh *trigger* — Gamma
/// (discovery) stays the metadata authority. `market_resolved`'s
/// `winning_outcome` label is omitted on purpose: the scheduler trusts only
/// the winning token id (mapped via [`crate::TokenPair::outcome_of`]);
/// feed-clob may cross-check the label against the id and warn on mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketLifecycleEvent {
    /// A new market appeared on the venue. Refresh trigger only.
    NewMarket {
        /// Wire `market` field — the market's condition id.
        condition_id: ConditionId,
        /// Wire `slug`, carried for logging only (never parsed for identity).
        slug: String,
        /// Wire `timestamp`.
        ts: TimestampMs,
    },
    /// A market resolved with the given winning outcome token.
    MarketResolved {
        /// Wire `market` field — the market's condition id.
        condition_id: ConditionId,
        /// Wire `winning_asset_id` — the token id of the winning outcome.
        winning_token: TokenId,
        /// Wire `timestamp`.
        ts: TimestampMs,
    },
}

/// THE internal bus event. See the module docs for the design rules.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Underlying price observation from one of the feeds.
    PriceTick(PriceTick),
    /// A price-feed stream went stale or recovered.
    FeedHealth(FeedHealth),
    /// A window's L2 books became unreliable or recovered.
    BookHealth(BookHealth),
    /// Full L2 book snapshot (subscribe or trade-driven rebuild).
    Book(Arc<BookSnapshot>),
    /// Best bid/ask update for one token.
    TopOfBook {
        /// Token the update belongs to.
        token_id: Arc<TokenId>,
        /// The new top of book.
        top: TopOfBook,
    },
    /// The venue flipped the tick size (0.01 ↔ 0.001 at 0.96/0.04).
    TickSizeChange {
        /// Market whose tick changed.
        condition_id: Arc<ConditionId>,
        /// The new tick.
        new_tick: TickSize,
        /// Event time.
        ts: TimestampMs,
    },
    /// A trade printed on the venue (paper fill simulation + markouts).
    LastTrade {
        /// Token that traded.
        token_id: Arc<TokenId>,
        /// Print price.
        price: Price,
        /// Print size.
        size: Size,
        /// Aggressor side as reported.
        side: Side,
        /// Print time.
        ts: TimestampMs,
    },
    /// Window lifecycle transition with the full market metadata attached.
    Window {
        /// The window's market.
        market: Arc<MarketInfo>,
        /// New lifecycle phase.
        lifecycle: WindowLifecycle,
    },
    /// Order status changed.
    OrderUpdate(Arc<OrderUpdate>),
    /// One of our orders executed.
    Fill(Arc<Fill>),
    /// New model output.
    Model(ModelSnapshot),
    /// The composite model health for an asset changed tier (latched).
    ModelHealth(ModelHealthEvent),
    /// New inventory snapshot for a window.
    Inventory(Arc<InventorySnapshot>),
    /// A window settled: its closing inventory summary, for analytics (§8/§10).
    Settlement(Arc<SettlementSummary>),
    /// Risk-manager notification.
    Risk(RiskEvent),
    /// Control-plane notification.
    Control(ControlEvent),
}

#[cfg(test)]
mod tests {
    use rust_decimal::dec;

    use super::*;
    use crate::market::{FeeParams, ResolutionSource, TokenPair, WindowId};
    use crate::series::WindowDuration;

    const fn assert_bus_friendly<T: Send + Sync + Clone + 'static>() {}

    #[test]
    fn event_is_send_sync_clone() {
        assert_bus_friendly::<Event>();
    }

    #[test]
    fn event_stays_small() {
        // Cheap-clone guard: a Vec or String inlined into a variant would
        // blow this budget. Tune deliberately if a legitimate variant grows.
        // Bumped 96 → 128 (2026-06-13, fair-value task): the widest variant,
        // the now-`Copy` ModelSnapshot, gained z / σ_τ / anchor so the engine
        // and the |fair − mid| breaker read them off the bus (it later also
        // gained a 1-byte `reason`; the new `ModelHealth` event variant is far
        // smaller) — still ≤ 128 and Copy-cheap.
        assert!(
            size_of::<Event>() <= 128,
            "Event grew to {} bytes — keep payloads Copy-small or Arc-wrapped",
            size_of::<Event>()
        );
    }

    #[test]
    fn cloning_book_event_bumps_arc_not_data() {
        let snap = Arc::new(BookSnapshot {
            token_id: TokenId::new("1").unwrap(),
            condition_id: ConditionId::new(format!("0x{}", "00".repeat(32))).unwrap(),
            bids: vec![],
            asks: vec![],
            ts: TimestampMs::from_millis(0),
            seq_hash: None,
        });
        let event = Event::Book(Arc::clone(&snap));
        let clone = event.clone();
        assert_eq!(Arc::strong_count(&snap), 3); // snap + event + clone
        drop(clone);
        assert_eq!(Arc::strong_count(&snap), 2);
    }

    #[test]
    fn window_event_carries_market() {
        let market = Arc::new(MarketInfo {
            window: WindowId {
                series: Series {
                    asset: Asset::Btc,
                    duration: WindowDuration::M5,
                },
                open_time: TimestampMs::from_millis(0),
            },
            event_slug: "btc-updown-5m-0".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("1").unwrap(),
                down: TokenId::new("2").unwrap(),
            },
            close_time: TimestampMs::from_millis(300_000),
            strike: Some(dec!(104000)),
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
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        });
        let event = Event::Window {
            market: Arc::clone(&market),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        };
        match event {
            Event::Window {
                market: m,
                lifecycle: WindowLifecycle::Resolved { outcome },
            } => {
                assert_eq!(outcome, Outcome::Up);
                assert_eq!(m.tokens.up.as_str(), "1");
            }
            other => panic!("wrong variant: {other:?}"),
        }

        // The Closed phase (past close, outcome pending) travels the bus too.
        let closed = Event::Window {
            market,
            lifecycle: WindowLifecycle::Closed,
        };
        match closed {
            Event::Window { lifecycle, .. } => assert_eq!(lifecycle, WindowLifecycle::Closed),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn feed_health_serde_round_trips() {
        let stale = FeedHealth::Stale {
            source: PriceSource::ChainlinkRtds,
            asset: Asset::Btc,
            kind: TickKind::Vendor,
            age: crate::time::DurationMs::from_millis(5_250),
        };
        let recovered = FeedHealth::Recovered {
            source: PriceSource::BinanceRtds,
            asset: Asset::Eth,
            kind: TickKind::Trade,
            gap: crate::time::DurationMs::from_millis(12_000),
        };
        for ev in [stale, recovered] {
            let json = serde_json::to_string(&ev).unwrap();
            let back: FeedHealth = serde_json::from_str(&json).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn feed_health_event_is_copy_small() {
        // FeedHealth rides the bus inline (no Arc): it must stay Copy-small.
        const fn assert_copy<T: Copy>() {}
        assert_copy::<FeedHealth>();
        assert!(size_of::<FeedHealth>() <= 32);
    }

    #[test]
    fn book_health_serde_round_trips_and_stays_copy_small() {
        // BookHealth rides the bus inline (no Arc): it must stay Copy-small.
        const fn assert_copy<T: Copy>() {}
        assert_copy::<BookHealth>();
        assert!(size_of::<BookHealth>() <= 48);

        let window = WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_781_166_600_000),
        };
        let unreliable = BookHealth::Unreliable {
            window,
            reason: BookUnreliableReason::Crossed,
            ts: TimestampMs::from_millis(1_781_166_700_000),
        };
        let recovered = BookHealth::Recovered {
            window,
            outage: DurationMs::from_millis(1_350),
            ts: TimestampMs::from_millis(1_781_166_701_350),
        };
        for ev in [unreliable, recovered] {
            let json = serde_json::to_string(&ev).unwrap();
            let back: BookHealth = serde_json::from_str(&json).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn model_health_event_serde_round_trips_and_stays_copy_small() {
        use crate::model_state::{ModelHealth, ModelHealthReason};

        // Rides the bus inline (no Arc): it must stay Copy-small.
        const fn assert_copy<T: Copy>() {}
        assert_copy::<ModelHealthEvent>();
        assert!(size_of::<ModelHealthEvent>() <= 32);

        let degraded = ModelHealthEvent {
            asset: Asset::Btc,
            health: ModelHealth::Degraded,
            reason: ModelHealthReason::FastLeadLost,
            ts: TimestampMs::from_millis(1_781_166_700_000),
        };
        let unreliable = ModelHealthEvent {
            asset: Asset::Eth,
            health: ModelHealth::Unreliable,
            reason: ModelHealthReason::ChainlinkStale,
            ts: TimestampMs::from_millis(1_781_166_705_000),
        };
        for ev in [degraded, unreliable] {
            // The event's reason and tier always agree.
            assert_eq!(ev.reason.tier(), ev.health);
            let json = serde_json::to_string(&ev).unwrap();
            let back: ModelHealthEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn market_lifecycle_event_serde_round_trips() {
        let cid = ConditionId::new(format!("0x{}", "ab".repeat(32))).unwrap();
        let new_market = MarketLifecycleEvent::NewMarket {
            condition_id: cid.clone(),
            slug: "btc-updown-5m-1781166900".to_owned(),
            ts: TimestampMs::from_millis(1_781_166_900_123),
        };
        let resolved = MarketLifecycleEvent::MarketResolved {
            condition_id: cid,
            winning_token: TokenId::new("123456789").unwrap(),
            ts: TimestampMs::from_millis(1_781_167_200_456),
        };
        for ev in [new_market, resolved] {
            let json = serde_json::to_string(&ev).unwrap();
            let back: MarketLifecycleEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn market_lifecycle_event_rejects_bad_ids_on_deserialize() {
        // Id validation runs inside serde (try_from), so a malformed wire id
        // can never construct the event.
        let bad = r#"{"MarketResolved":{"condition_id":"0xnothex","winning_token":"1","ts":0}}"#;
        assert!(serde_json::from_str::<MarketLifecycleEvent>(bad).is_err());
        let bad_token = r#"{"MarketResolved":{"condition_id":"0x"#.to_owned()
            + &"ab".repeat(32)
            + r#"","winning_token":"0xdead","ts":0}}"#;
        assert!(serde_json::from_str::<MarketLifecycleEvent>(&bad_token).is_err());
    }
}
