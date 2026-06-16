//! Determinism acceptance tests for the replay harness (feature `replay`,
//! CLAUDE.md §3/§9). Built with `cargo test -p analytics --features replay`;
//! compiles to nothing without the feature.
//!
//! The headline guarantee: two replays of the same recording + seed produce
//! **byte-identical** analytics. Plus: a replay does non-trivial work (fills +
//! settlement), a sweep completes / ranks / is itself deterministic (sequential
//! == parallel), and the disk path (`ReplayReader`-read envelopes) reproduces the
//! in-memory result.

#![cfg(feature = "replay")]
// Panicking helpers are the point in tests; the helpers aren't #[test] fns
// themselves, so the clippy.toml test exemption doesn't reach them.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::io::Write;

use analytics::{
    AnalyticsParams, ComparisonWindow, ReplayConfig, SortColumn, SweepGrid, run_replay, run_sweep,
};
use core_types::{
    AnchorSource, Asset, BookLevel, BookSnapshot, ConditionId, Decimal, Dollars, DurationMs, Event,
    FeeParams, InputAges, MarketInfo, ModelHealth, ModelHealthReason, ModelSnapshot, Outcome,
    Price, ResolutionSource, RoundDir, Series, Side, Size, TickSize, TimestampMs, TokenId,
    TokenPair, WindowDuration, WindowId, WindowLifecycle,
};
use engine::RiskParams;
use flate2::Compression;
use flate2::write::GzEncoder;
use journal::{JournalRecord, RecordEnvelope, ReplayReader};
use rust_decimal::dec;
use venue_paper::{LatencySpec, PaperParams};

const BASE: i64 = 1_781_000_000_000;
const CLOSE: i64 = BASE + 300_000;

// ---- builders -------------------------------------------------------------

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
fn series() -> Series {
    Series {
        asset: Asset::Btc,
        duration: WindowDuration::M5,
    }
}
fn window() -> WindowId {
    WindowId {
        series: series(),
        open_time: TimestampMs::from_millis(BASE),
    }
}
fn market() -> MarketInfo {
    MarketInfo {
        window: window(),
        event_slug: "btc-updown-5m-replay".to_owned(),
        condition_id: condition(),
        tokens: TokenPair {
            up: up_token(),
            down: down_token(),
        },
        close_time: TimestampMs::from_millis(CLOSE),
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
        market: std::sync::Arc::new(market()),
        lifecycle,
    }
}

fn book(
    token: TokenId,
    bids: &[(Decimal, Decimal)],
    asks: &[(Decimal, Decimal)],
    ts: i64,
) -> Event {
    Event::Book(std::sync::Arc::new(BookSnapshot {
        token_id: token,
        condition_id: condition(),
        bids: bids
            .iter()
            .map(|(p, s)| BookLevel {
                price: px(*p),
                size: sz(*s),
            })
            .collect(),
        asks: asks
            .iter()
            .map(|(p, s)| BookLevel {
                price: px(*p),
                size: sz(*s),
            })
            .collect(),
        ts: TimestampMs::from_millis(ts),
        seq_hash: None,
    }))
}

fn model(p_up: f64, sigma: f64, ts: i64) -> Event {
    Event::Model(ModelSnapshot {
        asset: Asset::Btc,
        window: Some(window()),
        p_up,
        z: 0.0,
        sigma_1s: sigma,
        sigma_tau: sigma * 15.0,
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

/// A sell-aggressor print on the Up token at `price` (fills a resting Up buy).
fn last_trade(price: Decimal, size: Decimal, ts: i64) -> Event {
    Event::LastTrade {
        token_id: std::sync::Arc::new(up_token()),
        price: px(price),
        size: sz(size),
        side: Side::Sell,
        ts: TimestampMs::from_millis(ts),
    }
}

fn env(seq: u64, ts: i64, ev: &Event) -> RecordEnvelope {
    RecordEnvelope {
        seq,
        ts_local_ms: ts,
        rec: JournalRecord::from_event(ev),
    }
}

/// A synthetic tape that drives the real engine to a maker fill and a settlement:
/// open → books (Up/Down at 0.45/0.55) → a Ready model at 0.50 (the quoter
/// converges to a 0.49/0.48/0.47 ladder over the ticks) → a sell print at 0.49
/// that fills the resting 0.49 Up buy → resolve Up. Timestamps are spaced so the
/// 100 ms risk ticks place + the 150 ms placement latency activates the quotes
/// well before the fill print.
fn synthetic_tape() -> Vec<RecordEnvelope> {
    vec![
        env(0, BASE, &window_event(WindowLifecycle::Open)),
        env(
            1,
            BASE + 10,
            &book(
                up_token(),
                &[(dec!(0.45), dec!(100))],
                &[(dec!(0.55), dec!(100))],
                BASE + 10,
            ),
        ),
        env(
            2,
            BASE + 20,
            &book(
                down_token(),
                &[(dec!(0.45), dec!(100))],
                &[(dec!(0.55), dec!(100))],
                BASE + 20,
            ),
        ),
        env(3, BASE + 30, &model(0.50, 0.0005, BASE + 30)),
        env(
            4,
            BASE + 5_000,
            &last_trade(dec!(0.49), dec!(10), BASE + 5_000),
        ),
        env(
            5,
            BASE + 10_000,
            &window_event(WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            }),
        ),
    ]
}

/// The base replay config: a fixed seed + zero jitter (deterministic latencies),
/// the full default engine, a roomy per-series cap so no breaker trips, and a
/// dense 100 ms tick cadence.
fn base_config() -> ReplayConfig {
    let mut caps = HashMap::new();
    caps.insert(series(), Dollars::new(Decimal::from(10_000)));
    ReplayConfig {
        risk: RiskParams::default(),
        series_caps: caps,
        paper: PaperParams {
            placement: LatencySpec {
                mean_ms: DurationMs::from_millis(150),
                jitter_ms: DurationMs::ZERO,
            },
            cancel: LatencySpec {
                mean_ms: DurationMs::from_millis(100),
                jitter_ms: DurationMs::ZERO,
            },
            rng_seed: Some(1),
            ..PaperParams::default()
        },
        analytics: AnalyticsParams::default(),
        mode: core_types::Mode::Paper,
        risk_tick_ms: 100,
        comparison_window: ComparisonWindow::All,
    }
}

// ---- tests ----------------------------------------------------------------

#[test]
fn two_replays_are_byte_identical() {
    let tape = synthetic_tape();
    let cfg = base_config();
    let a = run_replay(&tape, &cfg).expect("replay a");
    let b = run_replay(&tape, &cfg).expect("replay b");
    let ja = serde_json::to_string(&a).expect("json a");
    let jb = serde_json::to_string(&b).expect("json b");
    assert_eq!(
        ja, jb,
        "two replays of the same tape + seed must be byte-identical"
    );
}

#[test]
fn replay_does_nontrivial_work() {
    let tape = synthetic_tape();
    let out = run_replay(&tape, &base_config()).expect("replay");
    assert!(out.summary.fills > 0, "the engine should have been filled");
    assert!(
        out.summary.windows_settled >= 1,
        "the window should have settled"
    );
    assert!(
        out.comparison.rows.iter().any(|r| r.windows_traded >= 1),
        "the comparison should carry a traded series"
    );
    // The Up buy filled at 0.49 and Up won: realized ≈ 10·(1 − 0.49) = 5.10.
    let row = out
        .comparison
        .rows
        .iter()
        .find(|r| r.series == series())
        .expect("a BTC-5m row");
    assert!(
        row.net_pnl > Dollars::ZERO,
        "winning fill yields positive PnL"
    );
    assert!(row.maker_fills >= 1, "the fill was a maker fill");
    assert_eq!(out.summary.events_total, tape.len() as u64);
    // The recorded outputs are all inputs here (synthetic tape has none to drop).
    assert_eq!(out.summary.events_dropped, 0);
}

#[test]
fn empty_and_unseeded_are_rejected() {
    let cfg = base_config();
    assert!(matches!(
        run_replay(&[], &cfg),
        Err(analytics::ReplayError::EmptyTape)
    ));
    let mut unseeded = base_config();
    unseeded.paper.rng_seed = None;
    assert!(matches!(
        run_replay(&synthetic_tape(), &unseeded),
        Err(analytics::ReplayError::UnseededPaper)
    ));
}

#[test]
fn sweep_completes_ranks_and_is_deterministic() {
    let tape = synthetic_tape();
    let base = base_config();
    let grid = SweepGrid {
        min_edge: vec![dec!(0.01), dec!(0.02)],
        gamma: vec![0.05, 0.10],
        cancel_theta: vec![], // → base value (single)
        taker_buffer: vec![], // → base value (single)
    };

    let r1 = run_sweep(&tape, &base, &grid, SortColumn::NetPnl, false).expect("sweep");
    assert_eq!(r1.points_total, 4, "2×2 grid → 4 points");
    assert_eq!(r1.rows.len(), 4);
    assert_eq!(r1.rank_metric, SortColumn::NetPnl);
    // Ranked best-first: NetPnl is higher-is-better → non-increasing rank_key.
    for pair in r1.rows.windows(2) {
        assert!(
            pair[0].aggregate.rank_key >= pair[1].aggregate.rank_key,
            "rows are ranked best-first"
        );
    }

    // Deterministic run-to-run.
    let r2 = run_sweep(&tape, &base, &grid, SortColumn::NetPnl, false).expect("sweep again");
    assert_eq!(
        serde_json::to_string(&r1).expect("json r1"),
        serde_json::to_string(&r2).expect("json r2"),
        "two sweeps of the same tape must be byte-identical"
    );

    // Parallelism speeds wall time but never changes the report.
    let r3 = run_sweep(&tape, &base, &grid, SortColumn::NetPnl, true).expect("sweep parallel");
    assert_eq!(
        serde_json::to_string(&r1).expect("json r1"),
        serde_json::to_string(&r3).expect("json r3"),
        "parallel and sequential sweeps must be identical"
    );
}

#[test]
fn record_gzip_roundtrip_replays() {
    let tape = synthetic_tape();
    let cfg = base_config();

    // Write the envelopes to a gzip segment (preserving each `ts_local_ms`
    // exactly) and read them back through the production `ReplayReader` — the
    // disk path the `bot replay` subcommand uses.
    let dir = std::env::temp_dir().join(format!("replay_roundtrip_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("journal-20260615-000000-00000.jsonl.gz");
    {
        let file = std::fs::File::create(&path).expect("create segment");
        let mut enc = GzEncoder::new(file, Compression::default());
        for record in &tape {
            writeln!(enc, "{}", serde_json::to_string(record).expect("json")).expect("write");
        }
        enc.finish().expect("finish gzip");
    }
    let read_back: Vec<RecordEnvelope> = ReplayReader::open(&dir)
        .expect("open replay")
        .map(|r| r.expect("envelope"))
        .collect();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        read_back, tape,
        "envelopes round-trip through gzip losslessly"
    );
    let from_disk = run_replay(&read_back, &cfg).expect("replay from disk");
    let in_memory = run_replay(&tape, &cfg).expect("replay in memory");
    assert_eq!(
        serde_json::to_string(&from_disk).expect("json disk"),
        serde_json::to_string(&in_memory).expect("json mem"),
        "disk replay reproduces the in-memory replay"
    );
}
