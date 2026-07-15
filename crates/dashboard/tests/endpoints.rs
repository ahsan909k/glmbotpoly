//! Endpoint contract tests: every REST endpoint against a seeded state store,
//! for both the paper (populated) and live (empty-but-present) namespaces.
//
// (remint marker — Windows App Control 4551 workaround)

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use axum::http::StatusCode;
use common::{
    OPEN_MS, get, market, open_router, raw_fill, seeded_handle, series, settlement_event, ts,
    window_event,
};
use core_types::{BreakerKind, Dollars, Event, Mode, Outcome, WindowLifecycle};
use dashboard::{DashboardHandle, DriverStatus};
use engine::{
    ArbitrationTally, ContentionSnapshot, FillDriver, RiskStateSnapshot, ShadowStop, TakerId,
    VetoTally,
};
use rust_decimal::dec;
use venue_api::RiskRejectDetail;

#[tokio::test]
async fn overview_carries_both_namespaces() {
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/api/overview", None).await;
    assert_eq!(status, StatusCode::OK);

    // Paper is populated.
    assert_eq!(body["modes"]["paper"]["running"], true);
    assert_eq!(
        body["modes"]["paper"]["wallet"]["collateral_total"],
        "10000"
    );
    assert_eq!(body["modes"]["paper"]["paper_capital"], "10000");
    assert!(
        !body["modes"]["paper"]["equity"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(!body["modes"]["paper"]["ledger"].is_null());

    // Live is present but empty (namespacing).
    assert_eq!(body["modes"]["live"]["running"], false);
    assert!(body["modes"]["live"]["wallet"].is_null());
    assert!(
        body["modes"]["live"]["equity"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn series_comparison_paper_then_live_and_resort() {
    let router = open_router(seeded_handle());

    let (status, body) = get(
        &router,
        "/api/series-comparison?mode=paper&window=all",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["mode"], "paper");
    assert_eq!(body["comparison"]["default_sort"], "NetPnl");
    assert_eq!(body["sort_columns"].as_array().unwrap().len(), 13);
    let rows = body["comparison"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["windows_traded"], 1);
    assert_eq!(rows[0]["net_pnl"], "3");
    assert_eq!(rows[0]["maker_fills"], 2);

    // A re-sort echoes the requested column.
    let (status, body) = get(
        &router,
        "/api/series-comparison?mode=paper&sort=FractionProfitable&dir=asc",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["comparison"]["default_sort"], "FractionProfitable");

    // Live namespace has no rows.
    let (status, body) = get(&router, "/api/series-comparison?mode=live", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["comparison"]["rows"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn summary_reconciles_buckets_and_scopes_by_series() {
    let router = open_router(seeded_handle());

    // All-series paper summary: one settled window, realized $3 all locked.
    let (status, body) = get(&router, "/api/summary?mode=paper&window=all", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["mode"], "paper");
    assert_eq!(body["explainer"]["completed_pair_pnl"], "3");
    assert_eq!(body["explainer"]["stuck_leg_pnl"], "0");
    assert_eq!(body["explainer"]["total_pnl"], "3");
    assert_eq!(body["explainer"]["pairs_completed"], "100");
    // Headline: account-global equity + paper capital, win rate, today's P/L.
    assert_eq!(body["headline"]["equity"], "10000");
    assert_eq!(body["headline"]["starting_capital"], "10000");
    assert_eq!(body["headline"]["win_rate"], 1.0);
    assert_eq!(body["headline"]["windows_traded"], 1);
    assert_eq!(body["headline"]["today_pl"], "3");
    // Only one resolved window (< min sample) ⇒ warming up.
    assert_eq!(body["status"]["state"], "warming_up");
    // Activity: one resting order; targets fall back to engine defaults.
    assert_eq!(body["activity"]["resting_now"], 1);
    assert_eq!(body["activity"]["ttr_target_ms"], 250);
    assert_eq!(body["activity"]["ttc_target_ms"], 200);
    // Paper has no PendingCancel ⇒ no time-to-cancel sample.
    assert!(body["activity"]["median_ttc_ms"].is_null());

    // Scoping to the (only) traded series gives the same numbers.
    let (status, scoped) = get(
        &router,
        "/api/summary?mode=paper&series=BTC-5m&window=all",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(scoped["explainer"]["total_pnl"], "3");
    assert_eq!(scoped["series"], "BTC-5m");

    // Live namespace is empty-but-present.
    let (status, live) = get(&router, "/api/summary?mode=live", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(live["explainer"]["total_pnl"], "0");
    assert_eq!(live["headline"]["windows_traded"], 0);
    assert!(live["headline"]["win_rate"].is_null());

    // An unknown series is a 400.
    let (status, _) = get(&router, "/api/summary?mode=paper&series=nonsense", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn windows_list_is_shared_with_per_mode_inventory() {
    let router = open_router(seeded_handle());

    let (status, body) = get(&router, "/api/windows?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    let windows = body.as_array().unwrap();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0]["outcome"], "Up");
    assert!(!windows[0]["model"].is_null());
    assert!(!windows[0]["inventory"].is_null());

    // The window is shared, but live has no inventory for it (namespacing).
    let (_status, body) = get(&router, "/api/windows?mode=live", None).await;
    let windows = body.as_array().unwrap();
    assert_eq!(windows.len(), 1);
    assert!(windows[0]["inventory"].is_null());
}

#[tokio::test]
async fn window_detail_and_unknown_404() {
    let router = open_router(seeded_handle());

    let (status, body) = get(
        &router,
        "/api/windows/BTC-5m@1781481600000?mode=paper",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body["up_book"]["bids"].as_array().unwrap().is_empty());
    assert_eq!(body["recent_prints"].as_array().unwrap().len(), 1);
    assert!(!body["up_mid"].is_null());
    // Our resting order is exposed for ladder highlighting.
    let our = body["our_orders"].as_array().unwrap();
    assert_eq!(our.len(), 1);
    assert_eq!(our[0]["outcome"], "Up");
    assert_eq!(our[0]["side"], "Buy");
    assert_eq!(our[0]["remaining"], "10");

    // A well-formed key for a window we don't hold → 404.
    let (status, _) = get(&router, "/api/windows/BTC-5m@999999?mode=paper", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn fills_blotter_namespaced_and_filtered() {
    let router = open_router(seeded_handle());

    let (status, body) = get(&router, "/api/fills?mode=paper&limit=10", None).await;
    assert_eq!(status, StatusCode::OK);
    let fills = body["fills"].as_array().unwrap();
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0]["liquidity"], "Maker");
    // Newest first: Down @ +20 (p_up 0.50→0.55 ⇒ fair_Down falls ⇒ −0.05),
    // then Up @ +10 (fair_Up rises ⇒ +0.05); both matured (not pending).
    assert_eq!(fills[0]["outcome"], "Down");
    assert_eq!(fills[0]["attribution"], "maker");
    assert_eq!(fills[0]["markout_pending"], false);
    assert!((fills[0]["markout_5s"].as_f64().unwrap() - (-0.05)).abs() < 1e-9);
    assert!((fills[1]["markout_5s"].as_f64().unwrap() - 0.05).abs() < 1e-9);

    // Window filter keeps both (same window).
    let (_status, body) = get(
        &router,
        "/api/fills?mode=paper&window=BTC-5m@1781481600000",
        None,
    )
    .await;
    assert_eq!(body["fills"].as_array().unwrap().len(), 2);

    // Live namespace has no fills.
    let (_status, body) = get(&router, "/api/fills?mode=live", None).await;
    assert!(body["fills"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn fills_carry_maker_taker_late_attribution() {
    let router = open_router(common::attribution_handle());
    let (status, body) = get(&router, "/api/fills?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    let fills = body["fills"].as_array().unwrap();
    assert_eq!(fills.len(), 3);
    // Newest first: late-window taker, mid-window taker, maker.
    assert_eq!(fills[0]["attribution"], "late");
    assert_eq!(fills[1]["attribution"], "taker");
    assert_eq!(fills[2]["attribution"], "maker");
}

#[tokio::test]
async fn risk_panel_carries_breaker_and_health() {
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/api/risk?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["tripped"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "Manual")
    );
    assert!(!body["snapshot"].is_null());
    assert_eq!(body["snapshot"]["daily_pnl"], "3");
    assert_eq!(body["ws_connected"], false);
    assert!(!body["feeds"].as_array().unwrap().is_empty());
    assert!(!body["model_health"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn params_endpoint() {
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/api/params", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["paper_capital"], "10000");
    assert_eq!(body["entries"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn health_is_unauthenticated_and_machine_readable() {
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    // Running + a tripped breaker ⇒ degraded.
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["modes"]["paper"]["running"], true);
    assert_eq!(body["feeds"]["any_stale"], true);
}

#[tokio::test]
async fn invalid_mode_is_400() {
    let router = open_router(seeded_handle());
    let (status, _) = get(&router, "/api/risk?mode=nonsense", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ---- D2: per-driver × per-series PnL matrix --------------------------------

#[tokio::test]
async fn pnl_matrix_populated_per_driver_with_win_pct() {
    let h = DashboardHandle::new(64, ts(OPEN_MS));
    let p = Mode::Paper;
    h.project(p, &window_event(WindowLifecycle::Open), ts(OPEN_MS));
    // Model buys Up @ 0.80 (will win); Momentum buys Down @ 0.30 (will lose).
    h.record_driver_fill(
        p,
        Some(FillDriver::Model),
        &raw_fill(Outcome::Up, dec!(0.80), dec!(10), OPEN_MS + 10),
        ts(OPEN_MS + 10),
    );
    h.record_driver_fill(
        p,
        Some(FillDriver::Momentum),
        &raw_fill(Outcome::Down, dec!(0.30), dec!(5), OPEN_MS + 20),
        ts(OPEN_MS + 20),
    );
    // Resolve Up + settle → the matrix marks the buffered fills.
    h.project(
        p,
        &Event::Window {
            market: market(),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        },
        ts(OPEN_MS + 300_000),
    );
    h.project(p, &settlement_event(), ts(OPEN_MS + 300_000));

    let router = open_router(h);
    let (status, body) = get(&router, "/api/pnl-matrix?mode=paper&window=all", None).await;
    assert_eq!(status, StatusCode::OK);
    let cells = body["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 2);
    let model = cells.iter().find(|c| c["driver"] == "model").unwrap();
    assert_eq!(model["trades"], 1);
    assert_eq!(model["wins"], 1);
    assert_eq!(model["win_pct"], 1.0);
    let mom = cells.iter().find(|c| c["driver"] == "momentum").unwrap();
    assert_eq!(mom["wins"], 0);
    assert_eq!(mom["win_pct"], 0.0);
    // Row (driver) and column (series) totals are present.
    assert!(!body["row_totals"].as_array().unwrap().is_empty());
    assert!(!body["col_totals"].as_array().unwrap().is_empty());
    // Live namespace is empty.
    let (_s, live) = get(&router, "/api/pnl-matrix?mode=live", None).await;
    assert!(live["cells"].as_array().unwrap().is_empty());
}

// ---- D3: STATUS strip (no silent states) -----------------------------------

#[tokio::test]
async fn status_strip_every_cell_defined_halted_under_a_breaker() {
    // seeded_handle sets a risk snapshot with Manual tripped.
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/api/status-strip?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    let cells = body["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 24, "4 drivers × 6 series");
    // EVERY cell resolves to exactly one non-null state.
    for c in cells {
        assert!(c["state"].is_string(), "no silent (null) cell state");
    }
    // A tripped global breaker halts every cell.
    assert!(cells.iter().all(|c| c["state"] == "halted"));
}

#[tokio::test]
async fn status_strip_shadow_tripped_is_distinct_from_halted() {
    let h = DashboardHandle::new(64, ts(OPEN_MS));
    // No tripped breaker, but a global DailyStop shadow was recorded.
    let snap = RiskStateSnapshot {
        tripped: vec![],
        window_loss_active: false,
        halted_windows: 0,
        halted_windows_list: vec![],
        trips_last_hour: 0,
        open_notional: Dollars::ZERO,
        open_notional_cap: Dollars::new(dec!(1000)),
        daily_pnl: Dollars::new(dec!(-50)),
        error_count: 0,
        sanity_breached: false,
        globally_halted: false,
        shadow_tripped: vec![ShadowStop {
            kind: BreakerKind::DailyStop,
            window: None,
            threshold: Dollars::new(dec!(40)),
            value: Dollars::new(dec!(-50)),
            ts: ts(OPEN_MS),
        }],
    };
    h.set_risk(Mode::Paper, snap, ts(OPEN_MS));
    let router = open_router(h);
    let (status, body) = get(&router, "/api/status-strip?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    let cells = body["cells"].as_array().unwrap();
    assert!(cells.iter().all(|c| c["state"] == "shadow_tripped"));
    assert!(cells.iter().all(|c| c["state"] != "halted"));
    // The shadow cell carries which stop + its value.
    assert!(cells[0]["detail"].as_str().unwrap().contains("DailyStop"));
    assert_eq!(cells[0]["value"], "-50");
}

#[tokio::test]
async fn status_strip_driver_standing_down() {
    let h = DashboardHandle::new(64, ts(OPEN_MS));
    // Clean risk state, but the momentum taker is standing down.
    h.set_driver_status(
        Mode::Paper,
        DriverStatus {
            maker_core_standing_down: false,
            momentum_standing_down: true,
            late_standing_down: false,
            model_standing_down: false,
        },
        ts(OPEN_MS),
    );
    let router = open_router(h);
    let (status, body) = get(&router, "/api/status-strip?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    let cells = body["cells"].as_array().unwrap();
    // Momentum cells stand down; the others are active (no breaker, no disable).
    for c in cells {
        let expected = if c["driver"] == "momentum" {
            "standing_down"
        } else {
            "active"
        };
        assert_eq!(c["state"], expected);
    }
}

// ---- D4: benchmark tile ----------------------------------------------------

#[tokio::test]
async fn benchmark_has_competitors_and_our_metrics() {
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/api/benchmark?mode=paper&series=BTC-5m", None).await;
    assert_eq!(status, StatusCode::OK);
    let comps = body["competitors"].as_array().unwrap();
    // gabigol is the PRIMARY anchor-consistent maker.
    let gabigol = comps.iter().find(|c| c["handle"] == "gabigol").unwrap();
    assert_eq!(gabigol["role"], "primary");
    assert_eq!(gabigol["at_touch_frac"], 0.58);
    // nagi's fill-size fact-sheet is present for the like-for-like comparison.
    let nagi = comps.iter().find(|c| c["handle"] == "nagi777").unwrap();
    assert_eq!(nagi["fill_size_median_usd"], 4.0);
    assert_eq!(nagi["fill_size_p90_usd"], 21.0);
    // Our metrics: two maker fills give a fill-size median; paper cancel-p95 n/a.
    assert_eq!(body["ours"]["series"], "BTC-5m");
    assert!(!body["ours"]["fill_size_median_usd"].is_null());
    assert!(body["ours"]["cancel_p95_ms"].is_null());
}

// ---- D5: contention counters -----------------------------------------------

#[tokio::test]
async fn contention_surfaces_arbitration_and_veto_tallies() {
    let h = DashboardHandle::new(64, ts(OPEN_MS));
    let snap = ContentionSnapshot {
        arbitration_blocks: vec![ArbitrationTally {
            series: series(),
            loser: TakerId::Model,
            winner: TakerId::Momentum,
            count: 3,
        }],
        risk_vetoes: vec![VetoTally {
            detail: RiskRejectDetail::Halted,
            series: series(),
            count: 2,
        }],
    };
    h.set_contention(Mode::Paper, &snap, ts(OPEN_MS));
    // A second drain accumulates.
    h.set_contention(Mode::Paper, &snap, ts(OPEN_MS + 1));

    let router = open_router(h);
    let (status, body) = get(&router, "/api/contention?mode=paper", None).await;
    assert_eq!(status, StatusCode::OK);
    let arb = body["arbitration"].as_array().unwrap();
    assert_eq!(arb[0]["loser"], "model");
    assert_eq!(arb[0]["winner"], "momentum");
    assert_eq!(arb[0]["count"], 6); // 3 + 3 accumulated
    let vetoes = body["vetoes"].as_array().unwrap();
    assert_eq!(vetoes[0]["detail"], "halted");
    assert_eq!(vetoes[0]["count"], 4);
}

// ---- D6c: digest tile ------------------------------------------------------

#[tokio::test]
async fn digest_returns_placeholder_when_absent() {
    let router = open_router(seeded_handle());
    let (status, body) = get(&router, "/api/digest?date=2000-01-01", None).await;
    assert_eq!(status, StatusCode::OK);
    // No such digest file → a present:false placeholder (never an error).
    assert_eq!(body["present"], false);
    assert_eq!(body["markdown"], "");
}

#[tokio::test]
async fn static_ui_is_served_unauthenticated() {
    let router = open_router(seeded_handle());

    let (status, ct, body) = common::get_raw(&router, "/").await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.contains("text/html"));
    assert!(body.contains("Trading Bot"));
    // The calm summary landing is present.
    assert!(body.contains("view-summary"));

    let (status, ct, _) = common::get_raw(&router, "/app.js").await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.contains("javascript"));

    let (status, ct, _) = common::get_raw(&router, "/app.css").await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.contains("text/css"));
}
