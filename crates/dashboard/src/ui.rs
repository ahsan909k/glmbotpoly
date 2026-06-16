//! The static single-page UI, embedded into the binary and served from
//! unauthenticated GET routes.
//!
//! The three assets ([`index.html`](INDEX_HTML), [`app.css`](APP_CSS),
//! [`app.js`](APP_JS)) are `include_str!`'d at build time — no `tower-http`
//! `ServeDir` (not on the CLAUDE.md §3 allowlist), no filesystem reads at
//! runtime, no build pipeline. The shell carries no secrets (the operator
//! supplies the bearer token into the page), so these routes are open like
//! `/health`; the data endpoints behind them remain token-gated.

use axum::Router;
use axum::http::header;
use axum::response::{Html, IntoResponse};
use axum::routing::get;

const INDEX_HTML: &str = include_str!("../ui/index.html");
const APP_CSS: &str = include_str!("../ui/app.css");
const APP_JS: &str = include_str!("../ui/app.js");

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn app_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS)
}

async fn app_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        APP_JS,
    )
}

/// The static-UI routes (`GET /`, `/app.css`, `/app.js`), generic over the
/// router state so they merge into the (stateful) open router unchanged.
pub(crate) fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/", get(index))
        .route("/app.css", get(app_css))
        .route("/app.js", get(app_js))
}
