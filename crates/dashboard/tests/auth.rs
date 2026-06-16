//! Authentication tests: a protected endpoint rejects an unauthenticated or
//! wrong-token request and accepts a correct one; `/health` is open; an
//! open-loopback router needs no token.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use axum::http::StatusCode;
use common::{TOKEN, enforced_router, get, loopback, open_router, public_addr, seeded_handle};

#[tokio::test]
async fn protected_endpoint_requires_a_valid_token() {
    let router = enforced_router(seeded_handle());

    // No token → 401.
    let (status, _) = get(&router, "/api/overview", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Wrong token → 401.
    let (status, _) = get(&router, "/api/overview", Some("wrong-token")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Correct token → 200.
    let (status, _) = get(&router, "/api/overview", Some(TOKEN)).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn health_is_open_even_when_enforced() {
    let router = enforced_router(seeded_handle());
    let (status, _) = get(&router, "/health", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn open_loopback_needs_no_token() {
    let router = open_router(seeded_handle());
    let (status, _) = get(&router, "/api/overview", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn non_loopback_without_token_refuses_to_build() {
    // The crate enforces its own invariant: no token + non-loopback bind fails
    // closed (mirrors config::validate).
    let result = dashboard::router(seeded_handle(), public_addr(), None);
    assert!(matches!(
        result,
        Err(dashboard::DashboardError::NonLoopbackWithoutToken(_))
    ));

    // A token unlocks it.
    assert!(dashboard::router(seeded_handle(), public_addr(), Some(TOKEN.to_owned())).is_ok());
    // Loopback is fine without a token.
    assert!(dashboard::router(seeded_handle(), loopback(), None).is_ok());
}
