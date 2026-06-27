//! The axum router, request handlers, and the `serve` entry points.
//!
//! Handlers are thin: each parses its query, locks the store via the handle,
//! calls a [`dto`](crate::dto) builder or an `analytics` query, unlocks, and
//! returns JSON. All `/api/*` routes are behind the bearer-token middleware;
//! `/health` and `/api/ws` are not (`/health` is an unauthenticated liveness
//! probe; `/api/ws` authenticates with a `?token=` query because browsers can't
//! set a WebSocket header).

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use analytics::{ComparisonWindow, DayKey, SeriesComparison, SortColumn};
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use core_types::{CommandOrigin, Decimal, Dollars, Mode, Series};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::auth::{Auth, WsAuthQuery, require_auth, unauthorized};
use crate::command::{ControlError, ControlOutcome, DashboardCommand, OutcomeKind};
use crate::dto::{
    self, FillsDto, OverviewDto, ParamsDto, RiskDto, WindowDetailDto, WindowSummaryDto,
};
use crate::error::DashboardError;
use crate::handle::DashboardHandle;
use crate::health::{self, HealthReport};
use crate::ws::{WsUpdate, ws_connection};

/// The router state shared by every handler.
#[derive(Clone)]
struct AppState {
    handle: DashboardHandle,
    auth: Arc<Auth>,
}

/// A request-level error rendered as a JSON body with a status code.
enum ApiError {
    /// A malformed or unknown query parameter / body.
    BadRequest(String),
    /// The requested resource does not exist.
    NotFound,
    /// The action is currently unavailable (e.g. control on a read-only handle).
    Unavailable(String),
}

#[derive(Serialize)]
struct ApiErrorBody {
    error: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not found".to_owned()),
            ApiError::Unavailable(m) => (StatusCode::SERVICE_UNAVAILABLE, m),
        };
        (status, Json(ApiErrorBody { error })).into_response()
    }
}

impl From<ControlError> for ApiError {
    fn from(e: ControlError) -> Self {
        match e {
            ControlError::Unavailable => ApiError::Unavailable(
                "control plane not available (read-only dashboard)".to_owned(),
            ),
            ControlError::Unsendable => ApiError::Unavailable(
                "control channel unavailable (orchestrator not accepting commands)".to_owned(),
            ),
        }
    }
}

// ---- query parsing ---------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ModeQuery {
    mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CompareQuery {
    mode: Option<String>,
    window: Option<String>,
    days: Option<u16>,
    sort: Option<String>,
    dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FillsQuery {
    mode: Option<String>,
    limit: Option<usize>,
    window: Option<String>,
    since_ms: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct SummaryQuery {
    mode: Option<String>,
    series: Option<String>,
    window: Option<String>,
    days: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct SettledQuery {
    mode: Option<String>,
    series: Option<String>,
    limit: Option<usize>,
}

fn parse_mode(s: Option<&str>) -> Result<Mode, ApiError> {
    match s {
        None | Some("paper") => Ok(Mode::Paper),
        Some("live") => Ok(Mode::Live),
        Some(other) => Err(ApiError::BadRequest(format!("unknown mode {other:?}"))),
    }
}

fn parse_window(window: Option<&str>, days: Option<u16>) -> Result<ComparisonWindow, ApiError> {
    if let Some(n) = days {
        return Ok(ComparisonWindow::LastDays(n));
    }
    match window {
        None | Some("all") => Ok(ComparisonWindow::All),
        Some("today") => Ok(ComparisonWindow::Today),
        Some("7d") => Ok(ComparisonWindow::LastDays(7)),
        Some(other) => Err(ApiError::BadRequest(format!("unknown window {other:?}"))),
    }
}

/// `None`/`all`/empty → all series combined; otherwise the named series.
fn parse_series(s: Option<&str>) -> Result<Option<Series>, ApiError> {
    match s {
        None | Some("all" | "") => Ok(None),
        Some(other) => other
            .parse::<Series>()
            .map(Some)
            .map_err(|_| ApiError::BadRequest(format!("unknown series {other:?}"))),
    }
}

fn parse_sort(s: &str) -> Result<SortColumn, ApiError> {
    SortColumn::ALL
        .into_iter()
        .find(|c| format!("{c:?}").eq_ignore_ascii_case(s))
        .ok_or_else(|| ApiError::BadRequest(format!("unknown sort column {s:?}")))
}

// ---- series comparison -----------------------------------------------------

/// The `/api/series-comparison` response.
#[derive(Debug, Serialize)]
struct SeriesComparisonDto {
    mode: Mode,
    comparison: SeriesComparison,
    sort_columns: Vec<dto::SortColumnDto>,
}

/// Re-sorts the comparison rows by `col` (a mechanical re-order, not business
/// logic — the keys come from `SeriesComparisonRow::sort_key`).
fn resort(comparison: &mut SeriesComparison, col: SortColumn, dir: Option<&str>) {
    let desc = match dir {
        Some("asc") => false,
        Some("desc") => true,
        _ => col.higher_is_better(),
    };
    comparison.rows.sort_by(|a, b| {
        let ordered = a
            .sort_key(col)
            .partial_cmp(&b.sort_key(col))
            .unwrap_or(std::cmp::Ordering::Equal);
        let ordered = if desc { ordered.reverse() } else { ordered };
        ordered.then(a.series.cmp(&b.series))
    });
    comparison.default_sort = col;
}

// ---- handlers --------------------------------------------------------------

async fn overview(State(app): State<AppState>) -> Json<OverviewDto> {
    Json(app.handle.with_data(dto::overview))
}

async fn series_comparison(
    State(app): State<AppState>,
    Query(q): Query<CompareQuery>,
) -> Result<Json<SeriesComparisonDto>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    let window = parse_window(q.window.as_deref(), q.days)?;
    let sort = q.sort.as_deref().map(parse_sort).transpose()?;
    let dir = q.dir.clone();
    let dto = app.handle.with_data(|d| {
        let today = DayKey::from_ts(d.last_now);
        let mut comparison = d.mode(mode).analytics.series_comparison_over(window, today);
        if let Some(col) = sort {
            resort(&mut comparison, col, dir.as_deref());
        }
        SeriesComparisonDto {
            mode,
            comparison,
            sort_columns: dto::sort_columns(),
        }
    });
    Ok(Json(dto))
}

async fn summary(
    State(app): State<AppState>,
    Query(q): Query<SummaryQuery>,
) -> Result<Json<dto::SummaryDto>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    let window = parse_window(q.window.as_deref(), q.days)?;
    let series = parse_series(q.series.as_deref())?;
    Ok(Json(
        app.handle
            .with_data(|d| dto::summary(d, mode, series, window)),
    ))
}

async fn settlements(
    State(app): State<AppState>,
    Query(q): Query<SettledQuery>,
) -> Result<Json<Vec<dto::ResolvedWindowDto>>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    let series = parse_series(q.series.as_deref())?;
    let limit = q.limit.unwrap_or(50).min(256);
    Ok(Json(app.handle.with_data(|d| {
        dto::resolved_windows(d, mode, series, limit)
    })))
}

async fn windows_list(
    State(app): State<AppState>,
    Query(q): Query<ModeQuery>,
) -> Result<Json<Vec<WindowSummaryDto>>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    Ok(Json(app.handle.with_data(|d| dto::windows(d, mode))))
}

async fn window_detail(
    State(app): State<AppState>,
    Path(window): Path<String>,
    Query(q): Query<ModeQuery>,
) -> Result<Json<WindowDetailDto>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    let wid = dto::parse_window_id(&window).ok_or(ApiError::NotFound)?;
    app.handle
        .with_data(|d| dto::window_detail(d, mode, wid))
        .map(Json)
        .ok_or(ApiError::NotFound)
}

async fn fills(
    State(app): State<AppState>,
    Query(q): Query<FillsQuery>,
) -> Result<Json<FillsDto>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    let limit = q.limit.unwrap_or(200).min(1_000);
    let window = match q.window.as_deref() {
        Some(w) => Some(
            dto::parse_window_id(w)
                .ok_or_else(|| ApiError::BadRequest(format!("malformed window {w:?}")))?,
        ),
        None => None,
    };
    let since = q.since_ms;
    Ok(Json(
        app.handle
            .with_data(|d| dto::fills(d, mode, limit, window, since)),
    ))
}

async fn risk(
    State(app): State<AppState>,
    Query(q): Query<ModeQuery>,
) -> Result<Json<RiskDto>, ApiError> {
    let mode = parse_mode(q.mode.as_deref())?;
    Ok(Json(app.handle.with_data(|d| dto::risk(d, mode))))
}

async fn params(State(app): State<AppState>) -> Json<ParamsDto> {
    Json(app.handle.with_data(dto::params))
}

async fn health(State(app): State<AppState>) -> Json<HealthReport> {
    Json(app.handle.with_data(health::health_report))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(app): State<AppState>,
    Query(q): Query<WsAuthQuery>,
) -> Response {
    if !app.auth.check(q.token.as_deref().map(str::as_bytes)) {
        return unauthorized();
    }
    let rx = app.handle.subscribe();
    let hello = WsUpdate::hello(app.handle.server_time_ms());
    ws.on_upgrade(move |socket| ws_connection(socket, rx, hello))
}

// ---- control (write) endpoints ---------------------------------------------

/// The `POST /api/control/paper-capital` body: exactly one of `amount` (set the
/// absolute paper capital) or `delta` (signed adjustment), as decimal strings.
#[derive(Debug, Deserialize)]
struct CapitalBody {
    amount: Option<String>,
    delta: Option<String>,
}

/// The `POST /api/control/{enable,disable}-series` body.
#[derive(Debug, Deserialize)]
struct SeriesBody {
    series: String,
}

/// The `POST /api/control/set-param` body.
#[derive(Debug, Deserialize)]
struct SetParamBody {
    series: Option<String>,
    key: String,
    value: String,
}

/// The `POST /api/control/arm-live/confirm` body.
#[derive(Debug, Deserialize)]
struct ConfirmBody {
    phrase: String,
}

/// Resolves the command origin from the `X-Bot-Control-Origin` header (set by
/// the `bot control` CLI); any other request is treated as the dashboard UI.
fn origin_of(headers: &HeaderMap) -> CommandOrigin {
    match headers
        .get("x-bot-control-origin")
        .and_then(|v| v.to_str().ok())
    {
        Some("cli") => CommandOrigin::Cli,
        _ => CommandOrigin::Dashboard,
    }
}

/// Hands a validated command to the orchestrator, awaits the resulting outcome,
/// and maps the [`OutcomeKind`] to an HTTP status (the outcome JSON — with the
/// resulting state — is the body in every case). A missing/closed control
/// channel maps to `503`.
async fn submit(app: &AppState, command: DashboardCommand, origin: CommandOrigin) -> Response {
    match app.handle.request(command, origin).await {
        Ok(outcome) => control_response(outcome),
        Err(e) => ApiError::from(e).into_response(),
    }
}

fn control_response(outcome: ControlOutcome) -> Response {
    let status = match outcome.kind {
        OutcomeKind::Accepted => StatusCode::OK,
        OutcomeKind::Rejected => StatusCode::BAD_REQUEST,
        OutcomeKind::Conflict => StatusCode::CONFLICT,
    };
    (status, Json(outcome)).into_response()
}

async fn control_kill(State(app): State<AppState>, headers: HeaderMap) -> Response {
    submit(&app, DashboardCommand::Kill, origin_of(&headers)).await
}

async fn control_reset(State(app): State<AppState>, headers: HeaderMap) -> Response {
    submit(&app, DashboardCommand::Reset, origin_of(&headers)).await
}

async fn control_reset_daily_stop(State(app): State<AppState>, headers: HeaderMap) -> Response {
    submit(&app, DashboardCommand::ResetDailyStop, origin_of(&headers)).await
}

async fn control_enable_series(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SeriesBody>,
) -> Response {
    submit(
        &app,
        DashboardCommand::EnableSeries(body.series),
        origin_of(&headers),
    )
    .await
}

async fn control_disable_series(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SeriesBody>,
) -> Response {
    submit(
        &app,
        DashboardCommand::DisableSeries(body.series),
        origin_of(&headers),
    )
    .await
}

async fn control_set_param(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SetParamBody>,
) -> Response {
    submit(
        &app,
        DashboardCommand::SetParam {
            series: body.series,
            key: body.key,
            value: body.value,
        },
        origin_of(&headers),
    )
    .await
}

async fn control_arm_begin(State(app): State<AppState>, headers: HeaderMap) -> Response {
    submit(&app, DashboardCommand::ArmLiveBegin, origin_of(&headers)).await
}

async fn control_arm_confirm(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ConfirmBody>,
) -> Response {
    submit(
        &app,
        DashboardCommand::ArmLiveConfirm {
            phrase: body.phrase,
        },
        origin_of(&headers),
    )
    .await
}

async fn control_disarm(State(app): State<AppState>, headers: HeaderMap) -> Response {
    submit(&app, DashboardCommand::Disarm, origin_of(&headers)).await
}

/// `GET /api/control/status`: a projection read of the latest control-plane
/// state snapshot the orchestrator pushed. `404` until the first push.
async fn control_status(State(app): State<AppState>) -> Result<Response, ApiError> {
    match app.handle.with_data(|d| d.control_state.clone()) {
        Some(state) => Ok(Json(state).into_response()),
        None => Err(ApiError::NotFound),
    }
}

async fn control_paper_capital(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CapitalBody>,
) -> Response {
    let cmd = match (body.amount.as_deref(), body.delta.as_deref()) {
        (Some(a), None) => {
            let d = match Decimal::from_str(a.trim()) {
                Ok(d) => d,
                Err(_) => {
                    return ApiError::BadRequest(format!("malformed amount {a:?}")).into_response();
                }
            };
            if d.is_sign_negative() {
                return ApiError::BadRequest("paper capital must be non-negative".to_owned())
                    .into_response();
            }
            DashboardCommand::SetPaperCapital(Dollars::new(d))
        }
        (None, Some(x)) => match Decimal::from_str(x.trim()) {
            Ok(d) => DashboardCommand::AdjustPaperCapital(d),
            Err(_) => {
                return ApiError::BadRequest(format!("malformed delta {x:?}")).into_response();
            }
        },
        (Some(_), Some(_)) => {
            return ApiError::BadRequest("provide exactly one of `amount` or `delta`".to_owned())
                .into_response();
        }
        (None, None) => {
            return ApiError::BadRequest(
                "provide `amount` (absolute) or `delta` (signed)".to_owned(),
            )
            .into_response();
        }
    };
    submit(&app, cmd, origin_of(&headers)).await
}

// ---- assembly + serving ----------------------------------------------------

/// Builds the dashboard router. `bind` is used only to resolve the auth policy
/// (a non-loopback bind requires a token); the listener is bound separately by
/// [`serve`]. Exposed so tests can drive it with `tower::ServiceExt::oneshot`.
///
/// # Errors
/// [`DashboardError::NonLoopbackWithoutToken`] if `bind` is not loopback and no
/// token is provided.
pub fn router(
    handle: DashboardHandle,
    bind: SocketAddr,
    token: Option<String>,
) -> Result<Router, DashboardError> {
    let auth = Arc::new(Auth::resolve(bind, token.as_deref())?);
    let state = AppState {
        handle,
        auth: Arc::clone(&auth),
    };
    let protected = Router::new()
        .route("/api/overview", get(overview))
        .route("/api/summary", get(summary))
        .route("/api/settlements", get(settlements))
        .route("/api/series-comparison", get(series_comparison))
        .route("/api/windows", get(windows_list))
        .route("/api/windows/{window}", get(window_detail))
        .route("/api/fills", get(fills))
        .route("/api/risk", get(risk))
        .route("/api/params", get(params))
        .route("/api/control/kill", post(control_kill))
        .route("/api/control/reset", post(control_reset))
        .route(
            "/api/control/reset-daily-stop",
            post(control_reset_daily_stop),
        )
        .route("/api/control/paper-capital", post(control_paper_capital))
        .route("/api/control/enable-series", post(control_enable_series))
        .route("/api/control/disable-series", post(control_disable_series))
        .route("/api/control/set-param", post(control_set_param))
        .route("/api/control/arm-live/begin", post(control_arm_begin))
        .route("/api/control/arm-live/confirm", post(control_arm_confirm))
        .route("/api/control/disarm", post(control_disarm))
        .route("/api/control/status", get(control_status))
        .route_layer(axum::middleware::from_fn_with_state(auth, require_auth));
    let open = Router::new()
        .route("/health", get(health))
        .route("/api/ws", any(ws_handler))
        .merge(crate::ui::routes());
    Ok(protected.merge(open).with_state(state))
}

/// Binds `bind` and serves the dashboard until `shutdown` resolves.
///
/// # Errors
/// [`DashboardError`] if the address cannot be bound, the auth policy refuses to
/// serve, or the accept loop fails.
pub async fn serve<F>(
    handle: DashboardHandle,
    bind: SocketAddr,
    token: Option<String>,
    shutdown: F,
) -> Result<(), DashboardError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind(bind)
        .await
        .map_err(|source| DashboardError::Bind { addr: bind, source })?;
    serve_with_listener(handle, listener, token, shutdown).await
}

/// Serves the dashboard on an already-bound listener (so callers that need the
/// assigned port — e.g. an ephemeral `127.0.0.1:0` bind in tests — can read it
/// from the listener first).
///
/// # Errors
/// [`DashboardError`] if the local address cannot be read, the auth policy
/// refuses to serve, or the accept loop fails.
pub async fn serve_with_listener<F>(
    handle: DashboardHandle,
    listener: TcpListener,
    token: Option<String>,
    shutdown: F,
) -> Result<(), DashboardError>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let addr = listener.local_addr().map_err(DashboardError::Serve)?;
    let router = router(handle, addr, token.clone())?;
    let auth_mode = if token.as_deref().is_some_and(|t| !t.is_empty()) {
        "enforced"
    } else {
        "open-loopback"
    };
    tracing::info!(target: "dashboard", %addr, auth = auth_mode, "dashboard server listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(DashboardError::Serve)?;
    tracing::info!(target: "dashboard", %addr, "dashboard server stopped");
    Ok(())
}
