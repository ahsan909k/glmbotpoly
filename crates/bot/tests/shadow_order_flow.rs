//! Order-flow-identical proof (BUILD_PLAN 12–13): the observation-only shadow
//! observer must influence **nothing**. This drives the real
//! [`engine::RiskManager`] → [`venue_paper::PaperVenue`] gateway through one
//! scripted event sequence **twice** — once with the shadow task spawned (fed a
//! clone of every event over its own channel) and once without — and asserts the
//! two venue-event streams (normalized to strip venue timestamps) are
//! byte-identical. Shadow holds no venue port and no `bus_tx`, so it is
//! structurally incapable of reaching the venue; this proves it empirically.
//!
//! Built on the `risk_manager_paper` harness (`#[tokio::test(start_paused)]`,
//! seeded `PaperParams`), with a fake depth transport so shadow never touches the
//! network.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot, Price,
    PriceSource, PriceTick, ResolutionSource, RoundDir, Series, Size, TickKind, TickSize,
    TimestampMs, TokenId, TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::{QuoteManagerParams, RiskManager, RiskParams};
use rust_decimal::dec;
use shadow::{LgbmModel, ModelIdentity, ShadowParams};
use tokio::sync::{mpsc, watch};
use venue_api::{VenueEvent, VenueEvents};
use venue_paper::{LatencySpec, PaperParams, PaperVenue};

const BASE_MS: i64 = 1_781_000_000_000;
const CLOSE_MS: i64 = BASE_MS + 300_000;

/// A minimal valid 24-feature LightGBM model (one tree). Shadow only needs to
/// *run* for the isolation proof; the prediction value is irrelevant.
const MINI_MODEL: &str = "tree
version=v4
num_class=1
max_feature_idx=23
objective=binary sigmoid:1
feature_names=Column_0 Column_1 Column_2 Column_3 Column_4 Column_5 Column_6 Column_7 Column_8 Column_9 Column_10 Column_11 Column_12 Column_13 Column_14 Column_15 Column_16 Column_17 Column_18 Column_19 Column_20 Column_21 Column_22 Column_23

Tree=0
num_leaves=2
num_cat=0
split_feature=5
threshold=0.5
decision_type=2
left_child=-1
right_child=-2
leaf_value=0.1 0.2
end of trees
";

fn px(d: Decimal) -> Price {
    Price::quantize(d, TickSize::T001, RoundDir::Down).expect("price")
}
fn sz(d: Decimal) -> Size {
    Size::new(d).expect("size")
}
fn up_token() -> TokenId {
    TokenId::new("111").expect("token")
}
fn down_token() -> TokenId {
    TokenId::new("222").expect("token")
}
fn condition() -> ConditionId {
    ConditionId::new(format!("0x{}", "11".repeat(32))).expect("cid")
}
fn window() -> WindowId {
    WindowId {
        series: Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        open_time: TimestampMs::from_millis(BASE_MS),
    }
}
fn market() -> MarketInfo {
    MarketInfo {
        window: window(),
        event_slug: "btc-updown-5m-shadow".to_owned(),
        condition_id: condition(),
        tokens: TokenPair {
            up: up_token(),
            down: down_token(),
        },
        close_time: TimestampMs::from_millis(CLOSE_MS),
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
fn window_event(lifecycle: WindowLifecycle) -> Event {
    Event::Window {
        market: Arc::new(market()),
        lifecycle,
    }
}
fn book(token: TokenId, bid: Decimal, ask: Decimal, ts: i64) -> Event {
    Event::Book(Arc::new(BookSnapshot {
        token_id: token,
        condition_id: condition(),
        bids: vec![BookLevel {
            price: px(bid),
            size: sz(dec!(100)),
        }],
        asks: vec![BookLevel {
            price: px(ask),
            size: sz(dec!(100)),
        }],
        ts: TimestampMs::from_millis(ts),
        seq_hash: None,
    }))
}
fn model(p_up: f64, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window()),
        p_up,
        z: 0.0,
        sigma_1s: 0.0005,
        sigma_tau: 0.0075,
        basis: 0.0,
        anchor: AnchorSource::BinanceCorrected,
        health: ModelHealth::Ready,
        reason: ModelHealthReason::Nominal,
        input_ages: InputAges {
            chainlink: DurationMs::from_millis(100),
            binance: DurationMs::from_millis(50),
        },
        ts: TimestampMs::from_millis(ts),
    })
}
fn binance_tick(price: Decimal, ts: i64) -> Event {
    Event::PriceTick(PriceTick {
        source: PriceSource::BinanceDirect,
        asset: Asset::Btc,
        kind: TickKind::Mid,
        value: price,
        ts_exchange: TimestampMs::from_millis(ts),
        ts_local: TimestampMs::from_millis(ts),
    })
}
fn params() -> PaperParams {
    PaperParams {
        placement: LatencySpec {
            mean_ms: DurationMs::from_millis(200),
            jitter_ms: DurationMs::ZERO,
        },
        cancel: LatencySpec {
            mean_ms: DurationMs::from_millis(100),
            jitter_ms: DurationMs::ZERO,
        },
        rng_seed: Some(1),
        venue_taker_delay_ms: DurationMs::ZERO,
        ..PaperParams::default()
    }
}
fn risk_manager() -> RiskManager {
    let rp = RiskParams {
        quote_manager: QuoteManagerParams {
            min_requote_interval_ms: 10,
            ..QuoteManagerParams::default()
        },
        momentum_enabled: false,
        late_window_enabled: false,
        ..RiskParams::default()
    };
    let mut caps = HashMap::new();
    caps.insert(window().series, Dollars::new(dec!(25)));
    RiskManager::new(rp, caps)
}

/// A venue event projected to its engine-meaningful fields (venue timestamps
/// stripped, since virtual-time advancement need not be bit-identical across the
/// two runs; the paper order ids ARE deterministic and kept).
fn norm(ve: &VenueEvent) -> String {
    match ve {
        VenueEvent::Order(u) => format!(
            "O {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
            u.order_id, u.state, u.side, u.price, u.original_size, u.filled_size, u.reject_reason
        ),
        VenueEvent::Fill(f) => format!(
            "F {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
            f.order_id, f.outcome, f.side, f.price, f.size, f.liquidity, f.fee
        ),
        other => format!("{other:?}"),
    }
}

/// Drives the scripted flow once; returns the normalized venue-event stream.
async fn drive_flow(with_shadow: bool) -> Vec<String> {
    let start = tokio::time::Instant::now();
    let now = move || {
        TimestampMs::from_millis(BASE_MS + i64::try_from(start.elapsed().as_millis()).unwrap_or(0))
    };
    let mut venue = PaperVenue::spawn(params(), now);
    let mut rx = venue.take_event_rx().expect("event rx");
    let mut risk = risk_manager();

    // Spawn the shadow observer fed a clone of every event (fake depth transport).
    let (shadow_tx, shutdown_tx, shadow_task, _handles) = if with_shadow {
        let (tx, srx) = mpsc::channel::<Event>(1024);
        let (utx, _urx) = mpsc::channel(256);
        let (sdtx, sdrx) = watch::channel(false);
        let (fake, handles, _at) = feed_util::fake::script(&[true]);
        let model = Arc::new(LgbmModel::from_text(MINI_MODEL).expect("mini model"));
        let identity = ModelIdentity {
            sha256: "test".to_owned(),
            trained_through_ms: 0,
            seed: 1,
        };
        let dir = std::env::temp_dir().join(format!("shadow-of-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = ShadowParams {
            shadow_dir: dir,
            assets: vec![Asset::Btc],
            ..ShadowParams::default()
        };
        let task = tokio::spawn(shadow::run(p, model, identity, srx, utx, fake, now, sdrx));
        (Some(tx), Some(sdtx), Some(task), Some(handles))
    } else {
        (None, None, None, None)
    };

    let mut out = Vec::new();
    let mut push = |ve: &VenueEvent| out.push(norm(ve));

    // Converge a two-sided ladder, then jump the model (cancel-first repricing),
    // feeding shadow a clone of each event.
    let script = [
        window_event(WindowLifecycle::Open),
        binance_tick(dec!(60000), BASE_MS),
        book(up_token(), dec!(0.45), dec!(0.55), BASE_MS),
        book(down_token(), dec!(0.45), dec!(0.55), BASE_MS),
        model(0.50, BASE_MS),
    ];
    for ev in &script {
        venue.on_bus_event(ev).await;
        risk.on_event(ev, &venue, now()).await;
        if let Some(tx) = &shadow_tx {
            let _ = tx.try_send(ev.clone());
        }
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    while let Ok(ve) = rx.try_recv() {
        risk.on_venue_event(&ve, &venue, now()).await;
        push(&ve);
    }

    let jump = model(0.85, BASE_MS + 1_000);
    venue.on_bus_event(&jump).await;
    risk.on_event(&jump, &venue, now()).await;
    if let Some(tx) = &shadow_tx {
        let _ = tx.try_send(jump.clone());
    }
    risk.on_tick(&venue, now()).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    while let Ok(ve) = rx.try_recv() {
        risk.on_venue_event(&ve, &venue, now()).await;
        push(&ve);
    }

    if let Some(sd) = shutdown_tx {
        let _ = sd.send(true);
    }
    if let Some(task) = shadow_task {
        let _ = task.await;
    }
    venue.shutdown();
    out
}

#[tokio::test(start_paused = true)]
async fn shadow_on_off_yields_identical_order_flow() {
    let without = drive_flow(false).await;
    let with = drive_flow(true).await;

    assert!(
        !without.is_empty(),
        "the scripted flow must produce venue events (else the test proves nothing)"
    );
    // Sanity: real order placements happened.
    assert!(
        without.iter().any(|s| s.starts_with("O ")),
        "expected order placements in the flow"
    );
    assert_eq!(
        without, with,
        "shadow ON must produce byte-identical order flow to shadow OFF"
    );
}
