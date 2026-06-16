//! Control write-endpoint contract tests: auth gating, command emission onto the
//! request sink, the outcome→status mapping (200 / 400 / 409 / 503), the
//! resulting-state ack, and bad-body (`400`) handling.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use axum::http::StatusCode;
use common::{
    TOKEN, control_handle, control_handle_with, enforced_router, open_router, outcome, post,
    seeded_handle,
};
use core_types::Dollars;
use dashboard::{DashboardCommand, OutcomeKind};
use rust_decimal::dec;
use serde_json::json;

#[tokio::test]
async fn control_endpoints_emit_typed_commands() {
    let (handle, log) = control_handle();
    let router = open_router(handle);

    // Kill.
    let (status, body) = post(&router, "/api/control/kill", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["kind"], "accepted");
    assert_eq!(log.lock().unwrap()[0], DashboardCommand::Kill);

    // Reset.
    let (status, _) = post(&router, "/api/control/reset", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(log.lock().unwrap()[1], DashboardCommand::Reset);

    // Reset daily stop.
    let (status, _) = post(&router, "/api/control/reset-daily-stop", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(log.lock().unwrap()[2], DashboardCommand::ResetDailyStop);

    // Set absolute capital.
    let (status, _) = post(
        &router,
        "/api/control/paper-capital",
        None,
        Some(json!({ "amount": "12000" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        log.lock().unwrap()[3],
        DashboardCommand::SetPaperCapital(Dollars::new(dec!(12000)))
    );

    // Adjust by a signed delta.
    let (status, _) = post(
        &router,
        "/api/control/paper-capital",
        None,
        Some(json!({ "delta": "-1000" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        log.lock().unwrap()[4],
        DashboardCommand::AdjustPaperCapital(dec!(-1000))
    );

    // Enable / disable series.
    let (status, _) = post(
        &router,
        "/api/control/enable-series",
        None,
        Some(json!({ "series": "BTC-5m" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        log.lock().unwrap()[5],
        DashboardCommand::EnableSeries("BTC-5m".to_owned())
    );

    // Set a parameter (the orchestrator validates; this responder accepts).
    let (status, _) = post(
        &router,
        "/api/control/set-param",
        None,
        Some(json!({ "series": "BTC-5m", "key": "min_edge", "value": "0.02" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        log.lock().unwrap()[6],
        DashboardCommand::SetParam {
            series: Some("BTC-5m".to_owned()),
            key: "min_edge".to_owned(),
            value: "0.02".to_owned(),
        }
    );

    // The multi-step arming flow.
    let (status, _) = post(&router, "/api/control/arm-live/begin", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(log.lock().unwrap()[7], DashboardCommand::ArmLiveBegin);
    let (status, _) = post(
        &router,
        "/api/control/arm-live/confirm",
        None,
        Some(json!({ "phrase": "arm-live-i-accept-real-money-losses" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        log.lock().unwrap()[8],
        DashboardCommand::ArmLiveConfirm {
            phrase: "arm-live-i-accept-real-money-losses".to_owned()
        }
    );
}

#[tokio::test]
async fn control_ack_carries_resulting_state() {
    let (handle, _log) = control_handle();
    let router = open_router(handle);
    let (status, body) = post(&router, "/api/control/kill", None, None).await;
    assert_eq!(status, StatusCode::OK);
    // The ack carries the resulting control state (the fake snapshot).
    assert_eq!(body["state"]["halted"], false);
    assert_eq!(body["state"]["enabled_series"][0], "BTC-5m");
    assert_eq!(body["state"]["paper_capital"], "10000");
}

#[tokio::test]
async fn control_rejected_outcome_is_400() {
    // The orchestrator rejects an out-of-range parameter → 400 with the reason.
    let (handle, _log) = control_handle_with(|_| {
        outcome(
            OutcomeKind::Rejected,
            Some("min_edge must be in [0.01, 0.5)"),
        )
    });
    let router = open_router(handle);
    let (status, body) = post(
        &router,
        "/api/control/set-param",
        None,
        Some(json!({ "key": "min_edge", "value": "0.005" })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["kind"], "rejected");
    assert!(body["error"].as_str().unwrap().contains("min_edge"));
}

#[tokio::test]
async fn control_conflict_outcome_is_409() {
    // Arming with a gate missing comes back as a conflict → 409.
    let (handle, _log) =
        control_handle_with(|_| outcome(OutcomeKind::Conflict, Some("cannot arm: gate 1")));
    let router = open_router(handle);
    let (status, body) = post(&router, "/api/control/arm-live/begin", None, None).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["kind"], "conflict");
}

#[tokio::test]
async fn control_bad_capital_body_is_400() {
    let (handle, _log) = control_handle();
    let router = open_router(handle);
    for body in [
        json!({ "amount": "-5" }),              // negative
        json!({ "amount": "abc" }),             // unparseable
        json!({ "amount": "1", "delta": "1" }), // both
        json!({}),                              // neither
    ] {
        let (status, _) = post(&router, "/api/control/paper-capital", None, Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn control_on_read_only_handle_is_503() {
    // The seeded handle has no request sink → read-only.
    let router = open_router(seeded_handle());
    let (status, _) = post(&router, "/api/control/kill", None, None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn control_status_reads_pushed_snapshot() {
    // No snapshot pushed yet → 404; after a push, the projection is readable.
    use common::{get, loopback};

    let (handle, _log) = control_handle();
    handle.set_control_state(common::control_snapshot(), common::ts(common::OPEN_MS));
    let router = dashboard::router(handle, loopback(), None).expect("router");
    let (status, body) = get(&router, "/api/control/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["enabled_series"][0], "BTC-5m");
}

#[tokio::test]
async fn control_requires_token_when_enforced() {
    let (handle, _log) = control_handle();
    let router = enforced_router(handle);

    // No token → 401 (the middleware gates /api/control/* like other /api routes).
    let (status, _) = post(&router, "/api/control/kill", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // With the token → accepted.
    let (status, body) = post(&router, "/api/control/kill", Some(TOKEN), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["kind"], "accepted");
}
