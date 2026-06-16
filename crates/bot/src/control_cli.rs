//! `bot control …` — the CLI frontend to the control plane (§10.6/§11).
//!
//! The CLI is a thin HTTP client to the **running** bot's dashboard control API:
//! each subcommand POSTs (or GETs, for `status`) the matching `/api/control/*`
//! endpoint, tagging itself `cli` via the `X-Bot-Control-Origin` header so the
//! command's audit record records the right origin. The dashboard bind address
//! comes from `config.dashboard.bind` and the bearer token from
//! `BOT_SECRET_DASHBOARD_TOKEN`; the JSON acknowledgment (the resulting state,
//! or the rejection reason) is printed, and a non-2xx status exits non-zero.

use anyhow::Context;
use config::{AppConfig, Secrets};
use serde_json::{Value, json};

/// A parsed `bot control` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlSub {
    /// `status` — show the current control-plane state.
    Status,
    /// `kill` — global kill.
    Kill,
    /// `reset` — clear operator-latched breakers.
    Reset,
    /// `reset-daily-stop` — clear the daily stop only.
    ResetDailyStop,
    /// `set-capital <amount>` — absolute paper capital.
    SetCapital(String),
    /// `adjust-capital <delta>` — signed paper-capital adjustment.
    AdjustCapital(String),
    /// `enable-series <KEY>`.
    EnableSeries(String),
    /// `disable-series <KEY>`.
    DisableSeries(String),
    /// `set-param <key> <value> [--series KEY]`.
    SetParam {
        /// Target series key, or `None` for all series.
        series: Option<String>,
        /// The safe-list parameter name.
        key: String,
        /// The new value.
        value: String,
    },
    /// `arm-live` — begin the live-arming flow.
    ArmLive,
    /// `confirm-arm <phrase>` — confirm a pending arming.
    ConfirmArm(String),
    /// `disarm` — disarm live trading.
    Disarm,
}

/// The HTTP route + body for a subcommand: `(is_post, path, optional JSON body)`.
fn route(sub: &ControlSub) -> (bool, &'static str, Option<Value>) {
    match sub {
        ControlSub::Status => (false, "/api/control/status", None),
        ControlSub::Kill => (true, "/api/control/kill", None),
        ControlSub::Reset => (true, "/api/control/reset", None),
        ControlSub::ResetDailyStop => (true, "/api/control/reset-daily-stop", None),
        ControlSub::SetCapital(a) => (
            true,
            "/api/control/paper-capital",
            Some(json!({ "amount": a })),
        ),
        ControlSub::AdjustCapital(d) => (
            true,
            "/api/control/paper-capital",
            Some(json!({ "delta": d })),
        ),
        ControlSub::EnableSeries(s) => (
            true,
            "/api/control/enable-series",
            Some(json!({ "series": s })),
        ),
        ControlSub::DisableSeries(s) => (
            true,
            "/api/control/disable-series",
            Some(json!({ "series": s })),
        ),
        ControlSub::SetParam { series, key, value } => (
            true,
            "/api/control/set-param",
            Some(json!({ "series": series, "key": key, "value": value })),
        ),
        ControlSub::ArmLive => (true, "/api/control/arm-live/begin", None),
        ControlSub::ConfirmArm(p) => (
            true,
            "/api/control/arm-live/confirm",
            Some(json!({ "phrase": p })),
        ),
        ControlSub::Disarm => (true, "/api/control/disarm", None),
    }
}

/// Runs a `bot control` subcommand against the running dashboard.
///
/// # Errors
/// If the runtime cannot be built, the dashboard is unreachable, or the command
/// is refused (non-2xx).
pub fn execute(config: &AppConfig, secrets: &Secrets, sub: ControlSub) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run(config, secrets, sub))
}

async fn run(config: &AppConfig, secrets: &Secrets, sub: ControlSub) -> anyhow::Result<()> {
    let base = format!("http://{}", config.dashboard.bind);
    let token = secrets
        .dashboard_token
        .as_ref()
        .map(|s| s.expose().to_owned());
    let client = reqwest::Client::builder()
        .build()
        .context("building the http client")?;

    let (post, path, body) = route(&sub);
    let url = format!("{base}{path}");
    let mut req = if post {
        client.post(&url)
    } else {
        client.get(&url)
    };
    req = req.header("x-bot-control-origin", "cli");
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }
    if let Some(b) = &body {
        req = req
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serde_json::to_string(b).context("serializing the request body")?);
    }

    let resp = req.send().await.with_context(|| {
        format!("contacting the dashboard control API at {base} — is the bot running?")
    })?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    println!("{} {}", status.as_u16(), pretty(&text));
    if !status.is_success() {
        anyhow::bail!("control command refused (HTTP {})", status.as_u16());
    }
    Ok(())
}

/// Pretty-prints a JSON body; falls back to the raw text if it does not parse.
fn pretty(text: &str) -> String {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| text.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_map_to_the_right_endpoints() {
        assert_eq!(route(&ControlSub::Kill).1, "/api/control/kill");
        assert_eq!(
            route(&ControlSub::Status),
            (false, "/api/control/status", None)
        );
        let (post, path, body) = route(&ControlSub::SetCapital("12000".to_owned()));
        assert!(post);
        assert_eq!(path, "/api/control/paper-capital");
        assert_eq!(body, Some(json!({ "amount": "12000" })));
        let (_, path, body) = route(&ControlSub::SetParam {
            series: Some("BTC-5m".to_owned()),
            key: "min_edge".to_owned(),
            value: "0.02".to_owned(),
        });
        assert_eq!(path, "/api/control/set-param");
        assert_eq!(
            body,
            Some(json!({ "series": "BTC-5m", "key": "min_edge", "value": "0.02" }))
        );
        assert_eq!(
            route(&ControlSub::ConfirmArm("phrase".to_owned())).1,
            "/api/control/arm-live/confirm"
        );
    }

    #[test]
    fn pretty_falls_back_to_raw_on_non_json() {
        assert_eq!(pretty("not json"), "not json");
        assert!(pretty(r#"{"a":1}"#).contains("\"a\""));
    }
}
