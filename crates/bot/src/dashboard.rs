//! `bot dashboard` — serves the dashboard over a live paper pipeline (§10, §12.6).
//!
//! Wires the scheduler + feed-clob (real Polymarket book/trade data on the bus)
//! into a [`PaperVenue`] with a trivial two-sided quoter — the same topology as
//! `bot paper-sim`, but across all enabled series — and feeds every bus and venue
//! event into a [`DashboardHandle`], serving the axum REST + WebSocket dashboard
//! concurrently on `config.dashboard.bind`. A per-mode [`InventoryManager`] folds
//! the venue's fills + the window lifecycle into `Inventory`/`Settlement` events
//! so the live window view and the series-comparison table populate from real
//! paper trading. Real data, paper money; the live namespace stays empty until
//! the live orchestrator lands (deferred).
//!
//! With `--with-model`, feed-rtds + feed-binance + a multi-asset
//! [`ModelRuntime`](crate::model_runtime::ModelRuntime) are wired onto the same
//! bus, publishing `Event::Model`/`Event::ModelHealth` so the dashboard's
//! fair-vs-mid panel and the live 5s markout populate. The default run leaves the
//! model off (lean — clob feed only).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use config::{AppConfig, Secrets};
use core_types::{
    ControlEvent, Dollars, Event, MarketInfo, Mode, NewOrder, OrderQty, Outcome, Series, Side,
    Size, TimeInForce, TokenId, TopOfBook, WindowLifecycle,
};
use dashboard::{ControlRequest, DashboardHandle, ParamsView};
use engine::{InventoryEffect, InventoryManager, QuoteManagerParams, QuoteParams};
use feed_binance::{BinanceArgs, BinanceSub};
use feed_clob::{ClobArgs, ClobCommand, ClobStatus};
use feed_rtds::{FeedSub, RtdsArgs};
use feed_util::{FeedError, WsTransport};
use journal::Recorder;
use rust_decimal::Decimal;
use scheduler::{SchedulerArgs, Timing};
use timeutil::wall_now;
use tokio::sync::{mpsc, oneshot, watch};
use venue_api::{VenueEvent, VenueEvents, VenuePort};
use venue_paper::PaperVenue;

use crate::boot::journal_params;
use crate::control::{ControlPlane, Decision, RuntimeControlState, VenueAction};
use crate::feed::{binance_params, clob_params, rtds_params};
use crate::model_runtime::ModelRuntime;
use crate::paper::{drain_join, paper_params};

/// WebSocket broadcast backlog per subscriber.
const WS_BROADCAST_CAP: usize = 1_024;
/// Wallet/ledger sampling cadence (drives the equity curve + overview).
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
/// Re-quote cadence (cancel-all + fresh two-sided quotes at the touch).
const QUOTE_PERIOD: Duration = Duration::from_secs(5);
/// Shares per quote (well above the 5-share / $1 minimums at these prices).
const QUOTE_SHARES: i64 = 10;
/// Model recompute + publish cadence with `--with-model` (covers τ-decay between
/// ticks; `bot fair` precedent).
const MODEL_HEARTBEAT: Duration = Duration::from_millis(100);

/// A spawned feed driver's join handle (rtds / binance under `--with-model`).
type FeedTask = tokio::task::JoinHandle<Result<(), FeedError>>;

/// Awaits an optional task, pending forever when absent — so a disabled feed
/// branch in the supervision `select!` simply never fires.
async fn join_opt<T>(
    task: &mut Option<tokio::task::JoinHandle<T>>,
) -> Result<T, tokio::task::JoinError> {
    match task.as_mut() {
        Some(t) => t.await,
        None => std::future::pending().await,
    }
}

/// Builds the runtime and serves the dashboard until ctrl-c. With `with_model`,
/// the fair-value model + its price feeds are wired so fair-vs-mid and the live
/// 5s markout populate.
pub fn execute(
    config: &AppConfig,
    secrets: &Secrets,
    series: Option<Series>,
    with_model: bool,
) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run(config, secrets, series, with_model))
}

/// Maps a few safe-listed config values into the dashboard's parameters view
/// (the full `config → params` boundary map is deferred). Includes the §8
/// late-window gate thresholds the live-window countdown marks (no-ATM,
/// no-passive, late-taker) and the fills blotter's late-window attribution.
/// Shared with `bot run` (the main run mode reuses the same view).
pub(crate) fn params_view(config: &AppConfig) -> ParamsView {
    let r = &config.risk;
    let p = &config.paper;
    let e = &config.engine.defaults;
    ParamsView {
        paper_capital: Some(p.starting_capital),
        entries: vec![
            (
                "risk.daily_stop_loss".to_owned(),
                r.daily_stop_loss.to_string(),
            ),
            (
                "risk.max_open_notional".to_owned(),
                r.max_open_notional.to_string(),
            ),
            (
                "risk.feed_staleness_ms".to_owned(),
                r.feed_staleness_ms.as_millis().to_string(),
            ),
            ("risk.sanity_bound".to_owned(), r.sanity_bound.to_string()),
            (
                "paper.default_fee_rate".to_owned(),
                p.default_fee_rate.to_string(),
            ),
            (
                "paper.maker_rebate_share".to_owned(),
                p.maker_rebate_share.to_string(),
            ),
            (
                "paper.taker_rebate_enabled".to_owned(),
                p.taker_rebate_enabled.to_string(),
            ),
            (
                "engine.no_atm_final_secs".to_owned(),
                e.no_atm_final_secs.to_string(),
            ),
            (
                "engine.no_passive_final_secs".to_owned(),
                e.no_passive_final_secs.to_string(),
            ),
            (
                "engine.late_window_tau_secs".to_owned(),
                e.late_window_tau_secs.to_string(),
            ),
            (
                "engine.atm_band".to_owned(),
                QuoteParams::default().atm_band.to_string(),
            ),
            // The "yank-it-back reflex" targets the live-activity strip compares
            // the measured time-to-replace / time-to-cancel against (no config
            // key; sourced from the engine defaults).
            (
                "engine.min_requote_interval_ms".to_owned(),
                QuoteManagerParams::default()
                    .min_requote_interval_ms
                    .to_string(),
            ),
            (
                "engine.cancel_rtt_secs".to_owned(),
                QuoteParams::default().cancel_rtt_secs.to_string(),
            ),
        ],
    }
}

async fn run(
    config: &AppConfig,
    secrets: &Secrets,
    series_filter: Option<Series>,
    with_model: bool,
) -> anyhow::Result<()> {
    let series_list: Vec<Series> = match series_filter {
        Some(s) => vec![s],
        None => config.engine.enabled_series(),
    };
    if series_list.is_empty() {
        anyhow::bail!("no enabled series to trade (enable some in config or pass --series)");
    }

    let service = discovery::DiscoveryService::from_config(&config.feeds, &config.discovery)
        .context("building discovery service")?;
    let bind = config.dashboard.bind;
    let token = secrets
        .dashboard_token
        .as_ref()
        .map(|s| s.expose().to_owned());

    tracing::info!(
        target: "dashboard",
        %bind,
        series = ?series_list.iter().map(|s| s.key()).collect::<Vec<_>>(),
        auth = token.is_some(),
        "dashboard starting (real data, paper money; ctrl-c to stop)"
    );

    // Restore prior inventory + order state from the journal (§3/§9); the live
    // engine wiring that consumes it is deferred.
    let _restored = crate::boot::rebuild_and_log(config);

    // The control plane: every command (from the dashboard UI or the `bot
    // control` CLI) is validated, journaled with origin, and acknowledged with
    // the resulting state. This loop is the single owner of the venue.
    let mut control = ControlPlane::new(config, secrets);
    let control_state: RuntimeControlState = control.state.clone();
    let (req_tx, mut req_rx) = mpsc::channel::<ControlRequest>(16);
    let handle = DashboardHandle::new(WS_BROADCAST_CAP, wall_now()).with_request_sink(req_tx);
    handle.set_session(Mode::Paper, true, wall_now());
    handle.set_params(params_view(config), wall_now());
    handle.set_control_state(control.snapshot(), wall_now());

    // Journal every bus event + every command audit (§3/§9/§11) so the running
    // process produces the durable, queryable audit trail.
    let recorder = Recorder::spawn(journal_params(config), wall_now)
        .context("spawning the journal recorder")?;

    // Bind the server listener up front so a bind failure surfaces immediately.
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("binding dashboard to {bind}"))?;
    let local = listener.local_addr().unwrap_or(bind);
    let (server_shutdown_tx, server_shutdown_rx) = oneshot::channel::<()>();
    let server_handle = handle.clone();
    let server_token = token.clone();
    let server_task = tokio::spawn(async move {
        dashboard::serve_with_listener(server_handle, listener, server_token, async move {
            let _ = server_shutdown_rx.await;
        })
        .await
    });
    tracing::info!(
        target: "dashboard",
        %local,
        "dashboard listening — probe GET /health and (with the token) GET /api/overview"
    );

    // The bus + paper pipeline (mirrors `bot paper-sim`, across all series). With
    // `--with-model`, feed-rtds + feed-binance + the model are wired onto the same
    // bus so the fair-vs-mid panel and the live 5s markout populate with real data.
    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    // Retained senders so `bus_rx.recv()` never returns `None` while tasks run.
    let bus_keepalive = bus_tx.clone();
    let model_tx = bus_tx.clone();
    let (window_tx, window_rx) = mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(64);
    let (market_tx, market_rx) = mpsc::channel(64);
    let (_command_tx, command_rx) = mpsc::channel::<ClobCommand>(8);
    let (clob_status_tx, _clob_status_rx) = watch::channel(ClobStatus::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Build the model runtime first (fail closed before spawning any feed) so a
    // bad config never leaves orphaned connections.
    let mut model = if with_model {
        Some(ModelRuntime::new(config, &series_list).context("building model runtime")?)
    } else {
        None
    };

    let mut sched_task = tokio::spawn(scheduler::run(SchedulerArgs {
        timing: Timing::from_config(&config.scheduler),
        series: series_list.clone(),
        refresher: service,
        now_fn: wall_now,
        bus_tx: bus_tx.clone(),
        market_rx: Some(market_rx),
        status_tx: None,
        shutdown_rx: shutdown_tx.subscribe(),
    }));

    // The model's price feeds onto the same bus (only with `--with-model`).
    let mut rtds_task: Option<FeedTask> = None;
    let mut binance_task: Option<FeedTask> = None;
    if with_model {
        rtds_task = Some(tokio::spawn(feed_rtds::run(RtdsArgs {
            params: rtds_params(&config.feeds),
            subscriptions: FeedSub::all(),
            transport: WsTransport,
            now_fn: wall_now,
            bus_tx: bus_tx.clone(),
            command_rx: None,
            status_tx: None,
            shutdown_rx: shutdown_tx.subscribe(),
            backoff_seed: None,
        })));
        binance_task = Some(tokio::spawn(feed_binance::run(BinanceArgs {
            params: binance_params(&config.feeds),
            subscriptions: BinanceSub::all(),
            transport: WsTransport,
            now_fn: wall_now,
            bus_tx: bus_tx.clone(),
            status_tx: None,
            shutdown_rx: shutdown_tx.subscribe(),
            backoff_seed: None,
        })));
    }

    let mut clob_task = tokio::spawn(feed_clob::run(ClobArgs {
        params: clob_params(&config.feeds),
        transport_factory: || WsTransport,
        now_fn: wall_now,
        bus_tx,
        window_rx,
        market_tx: Some(market_tx),
        command_rx: Some(command_rx),
        status_tx: Some(clob_status_tx),
        shutdown_rx,
        backoff_seed: None,
    }));

    let mut venue = PaperVenue::spawn(paper_params(config), wall_now);
    let mut venue_rx = venue
        .take_event_rx()
        .context("paper venue event stream already taken")?;

    let mut inventory = InventoryManager::new();
    let mut quoter = Quoter::default();

    let mut sample = tokio::time::interval(SAMPLE_PERIOD);
    sample.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut quote = tokio::time::interval(QUOTE_PERIOD);
    quote.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut heartbeat = tokio::time::interval(MODEL_HEARTBEAT);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Per-task join results, filled as a task exits or is drained at shutdown.
    let mut sched_result = None;
    let mut clob_result = None;
    let mut rtds_result = None;
    let mut binance_result = None;

    loop {
        tokio::select! {
            joined = &mut sched_task, if sched_result.is_none() => {
                tracing::warn!(target: "dashboard", "scheduler exited — shutting down");
                sched_result = Some(joined);
                break;
            }
            joined = &mut clob_task, if clob_result.is_none() => {
                tracing::warn!(target: "dashboard", "clob feed exited — shutting down");
                clob_result = Some(joined);
                break;
            }
            joined = join_opt(&mut rtds_task) => {
                tracing::warn!(target: "dashboard", "rtds feed exited — shutting down");
                rtds_result = Some(joined);
                rtds_task = None;
                break;
            }
            joined = join_opt(&mut binance_task) => {
                tracing::warn!(target: "dashboard", "binance feed exited — shutting down");
                binance_result = Some(joined);
                binance_task = None;
                break;
            }
            maybe = bus_rx.recv() => match maybe {
                Some(event) => {
                    let now = wall_now();
                    recorder.record(&event);
                    if let Event::Window { market, lifecycle } = &event {
                        let _ = window_tx.try_send((Arc::clone(market), *lifecycle));
                    }
                    venue.on_bus_event(&event).await;
                    handle.project(Mode::Paper, &event, now);
                    quoter.on_bus_event(&event);
                    if let Some(m) = &mut model {
                        m.on_event(&event);
                    }
                    for effect in inventory.on_event(&event) {
                        publish_inventory_effect(&handle, effect, now);
                    }
                }
                None => unreachable!("dashboard retains a bus sender (bus_keepalive)"),
            },
            maybe = venue_rx.recv() => {
                if let Some(ve) = maybe {
                    on_venue_event(&handle, &mut inventory, &ve, wall_now());
                }
            }
            maybe = req_rx.recv() => {
                if let Some(req) = maybe {
                    let now = wall_now();
                    let decision = control.decide(req.command, req.origin, now);
                    apply_decision(decision, req.reply, &venue, &handle, &recorder, now).await;
                }
            }
            _ = quote.tick() => {
                // The global kill flag (set by an operator KILL, cleared by
                // RESET) stands the quoter down; disabled series are skipped.
                if !control_state.is_halted() {
                    quoter.requote(&venue, &control_state).await;
                }
            }
            _ = sample.tick() => {
                let wallet = venue.balances().await.unwrap_or_default();
                handle.set_wallet(Mode::Paper, wallet, wall_now());
                handle.set_paper_ledger(Mode::Paper, venue.ledger_snapshot(), wall_now());
            }
            _ = heartbeat.tick(), if model.is_some() => {
                if let Some(m) = &mut model {
                    m.on_heartbeat(&model_tx, wall_now());
                }
            }
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "dashboard", "ctrl-c — cancelling paper orders and shutting down");
                let _ = venue.cancel_all().await;
                venue.shutdown();
                break;
            }
        }
    }

    // Graceful shutdown: signal every task, release our bus senders, then drain +
    // join whatever is still running (draining the bus so no task deadlocks).
    let _ = shutdown_tx.send(true);
    drop(bus_keepalive);
    drop(model_tx);
    if sched_result.is_none() {
        sched_result = Some(drain_join(&shutdown_tx, &mut sched_task, &mut bus_rx).await);
    }
    if clob_result.is_none() {
        clob_result = Some(drain_join(&shutdown_tx, &mut clob_task, &mut bus_rx).await);
    }
    if let Some(task) = &mut rtds_task {
        rtds_result = Some(drain_join(&shutdown_tx, task, &mut bus_rx).await);
    }
    if let Some(task) = &mut binance_task {
        binance_result = Some(drain_join(&shutdown_tx, task, &mut bus_rx).await);
    }

    // Stop the dashboard server.
    let _ = server_shutdown_tx.send(());
    match server_task.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(target: "dashboard", error = %e, "dashboard server error"),
        Err(e) => tracing::warn!(target: "dashboard", error = %e, "dashboard server task panicked"),
    }

    // Flush + finalize the journal (gzip trailers + sqlite commit).
    match recorder.finish() {
        Ok(stats) => tracing::info!(
            target: "dashboard",
            records = stats.records,
            indexed = stats.indexed,
            dropped = stats.dropped,
            "journal flushed"
        ),
        Err(e) => tracing::warn!(target: "dashboard", error = %e, "journal flush error"),
    }

    if let Some(r) = sched_result {
        r.context("scheduler driver panicked")?
            .context("scheduler driver failed")?;
    }
    if let Some(r) = clob_result {
        r.context("clob feed panicked")?
            .context("clob feed failed")?;
    }
    if let Some(r) = rtds_result {
        r.context("rtds feed panicked")?
            .context("rtds feed failed")?;
    }
    if let Some(r) = binance_result {
        r.context("binance feed panicked")?
            .context("binance feed failed")?;
    }
    Ok(())
}

/// Converts a venue order/fill event into bus events for the dashboard, folding
/// fills through the inventory manager to produce inventory + settlement.
fn on_venue_event(
    handle: &DashboardHandle,
    inventory: &mut InventoryManager,
    event: &VenueEvent,
    now: core_types::TimestampMs,
) {
    match event {
        VenueEvent::Order(u) => {
            handle.project(Mode::Paper, &Event::OrderUpdate(Arc::clone(u)), now);
        }
        VenueEvent::Fill(f) => {
            let fill_event = Event::Fill(Arc::clone(f));
            handle.project(Mode::Paper, &fill_event, now);
            for effect in inventory.on_event(&fill_event) {
                publish_inventory_effect(handle, effect, now);
            }
        }
        VenueEvent::Connectivity { connected } => {
            handle.set_ws_connected(Mode::Paper, *connected, now);
        }
    }
}

/// Publishes an inventory-manager effect to the dashboard as the matching bus
/// event.
fn publish_inventory_effect(
    handle: &DashboardHandle,
    effect: InventoryEffect,
    now: core_types::TimestampMs,
) {
    let event = match effect {
        InventoryEffect::Snapshot(snapshot) => Event::Inventory(Arc::new(snapshot)),
        InventoryEffect::Settled(summary) => Event::Settlement(Arc::new(summary)),
    };
    handle.project(Mode::Paper, &event, now);
}

/// Applies a control-plane [`Decision`] to the running paper pipeline and
/// replies the resulting outcome. This loop is the single owner of the venue:
/// it performs the venue action, journals every event (audit + state-change),
/// projects the displayable ones onto the dashboard (which folds `Event::Risk`
/// into the breaker panel and `ControlEvent::PaperCapitalSet` into the displayed
/// capital), fills the authoritative paper capital into the ack, pushes the
/// control-state snapshot for `GET /api/control/status`, and replies.
async fn apply_decision(
    decision: Decision,
    reply: tokio::sync::oneshot::Sender<dashboard::ControlOutcome>,
    venue: &PaperVenue,
    handle: &DashboardHandle,
    recorder: &Recorder,
    now: core_types::TimestampMs,
) {
    let Decision {
        mut outcome,
        events,
        venue_action,
    } = decision;

    // Perform the venue side-effect.
    let capital_change = matches!(
        venue_action,
        Some(VenueAction::SetCapital(_) | VenueAction::AdjustCapital(_))
    );
    match venue_action {
        Some(VenueAction::CancelAll) => {
            let _ = venue.cancel_all().await;
        }
        Some(VenueAction::SetCapital(amount)) => venue.set_capital(amount),
        Some(VenueAction::AdjustCapital(delta)) => venue.adjust_capital(Dollars::new(delta)),
        None => {}
    }

    // Journal every event; project the displayable (Risk / Control) ones.
    for event in &events {
        recorder.record(event);
        if matches!(event, Event::Risk(_) | Event::Control(_)) {
            handle.project(Mode::Paper, event, now);
        }
    }

    // The venue is authoritative for capital: read it back, project the
    // resulting PaperCapitalSet, and resample the wallet so the change shows.
    let capital = venue.balances().await.ok().map(|w| w.collateral_total);
    if capital_change && let Some(amount) = capital {
        let ev = Event::Control(ControlEvent::PaperCapitalSet { amount });
        recorder.record(&ev);
        handle.project(Mode::Paper, &ev, now);
        resample_wallet(venue, handle, now).await;
    }

    outcome.state.paper_capital = capital;
    handle.set_control_state(outcome.state.clone(), now);

    if !outcome.accepted() {
        tracing::warn!(
            target: "control",
            kind = ?outcome.kind,
            error = outcome.error.as_deref().unwrap_or(""),
            "control command refused"
        );
    }
    let _ = reply.send(outcome);
}

/// Re-samples the venue wallet/ledger immediately so a capital change shows on
/// the dashboard without waiting for the next periodic sample tick.
async fn resample_wallet(
    venue: &PaperVenue,
    handle: &DashboardHandle,
    now: core_types::TimestampMs,
) {
    let wallet = venue.balances().await.unwrap_or_default();
    handle.set_wallet(Mode::Paper, wallet, now);
    handle.set_paper_ledger(Mode::Paper, venue.ledger_snapshot(), now);
}

/// The trivial multi-series quoter: tracks each series' current window and the
/// latest top of book, and re-quotes a small two-sided ladder at the touch every
/// [`QUOTE_PERIOD`]. Post-only-safe by quoting *at* the touch (joining, never
/// crossing).
#[derive(Default)]
struct Quoter {
    markets: HashMap<Series, Arc<MarketInfo>>,
    tops: HashMap<TokenId, TopOfBook>,
}

impl Quoter {
    fn on_bus_event(&mut self, event: &Event) {
        match event {
            Event::Window {
                market,
                lifecycle: WindowLifecycle::Open,
            } => {
                self.markets
                    .insert(market.window.series, Arc::clone(market));
            }
            Event::Book(snap) => {
                self.tops.insert(snap.token_id.clone(), snap.top());
            }
            Event::TopOfBook { token_id, top } => {
                self.tops.insert((**token_id).clone(), *top);
            }
            _ => {}
        }
    }

    async fn requote(&self, venue: &PaperVenue, control: &RuntimeControlState) {
        if self.markets.is_empty() {
            return;
        }
        let _ = venue.cancel_all().await;
        let qty = match Size::new(Decimal::from(QUOTE_SHARES)) {
            Ok(s) => OrderQty::Shares(s),
            Err(_) => return,
        };
        for market in self.markets.values() {
            // Honor the runtime series-enable toggle (the control-plane seam).
            if !control.is_series_enabled(market.window.series) {
                continue;
            }
            if let Some(top) = self.tops.get(&market.tokens.up)
                && let (Some(bid), Some(ask)) = (top.bid, top.ask)
            {
                let buy = NewOrder {
                    client_id: Some("dash-up-bid".to_owned()),
                    window: market.window,
                    token_id: market.tokens.up.clone(),
                    outcome: Outcome::Up,
                    side: Side::Buy,
                    price: bid.price,
                    qty,
                    tif: TimeInForce::Gtc { post_only: true },
                };
                let sell = NewOrder {
                    side: Side::Sell,
                    price: ask.price,
                    client_id: Some("dash-up-ask".to_owned()),
                    ..buy.clone()
                };
                let _ = venue.place(&buy).await;
                let _ = venue.place(&sell).await;
            }
            if let Some(dtop) = self.tops.get(&market.tokens.down)
                && let Some(dbid) = dtop.bid
            {
                let down_buy = NewOrder {
                    client_id: Some("dash-down-bid".to_owned()),
                    window: market.window,
                    token_id: market.tokens.down.clone(),
                    outcome: Outcome::Down,
                    side: Side::Buy,
                    price: dbid.price,
                    qty,
                    tif: TimeInForce::Gtc { post_only: true },
                };
                let _ = venue.place(&down_buy).await;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use core_types::{CommandOrigin, TimestampMs};
    use dashboard::{ControlOutcome, DashboardCommand};
    use journal::RecorderParams;
    use venue_paper::PaperParams;

    use super::*;

    fn dollars(v: i64) -> Dollars {
        Dollars::new(Decimal::from(v))
    }

    /// Drives one command through the control plane + `apply_decision` and
    /// returns the resulting outcome.
    async fn run_cmd(
        control: &mut ControlPlane,
        cmd: DashboardCommand,
        venue: &PaperVenue,
        handle: &DashboardHandle,
        recorder: &Recorder,
        now: TimestampMs,
    ) -> ControlOutcome {
        let decision = control.decide(cmd, CommandOrigin::Dashboard, now);
        let (tx, rx) = tokio::sync::oneshot::channel();
        apply_decision(decision, tx, venue, handle, recorder, now).await;
        rx.await.expect("apply_decision replies")
    }

    #[tokio::test]
    async fn apply_decision_drives_capital_and_halt() {
        let venue = PaperVenue::spawn(PaperParams::default(), || {
            TimestampMs::from_millis(1_000_000)
        });
        let handle = DashboardHandle::new(8, TimestampMs::from_millis(0));
        let dir = std::env::temp_dir().join(format!("bot-dash-ctl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let recorder = Recorder::spawn(
            RecorderParams {
                out_dir: dir.clone(),
                ..RecorderParams::default()
            },
            || TimestampMs::from_millis(1_000_000),
        )
        .expect("recorder");
        let mut control = ControlPlane::new(&AppConfig::default(), &Secrets::default());
        let now = TimestampMs::from_millis(1_000_000);

        // Set absolute capital — the ack carries the resulting capital.
        let out = run_cmd(
            &mut control,
            DashboardCommand::SetPaperCapital(dollars(12_000)),
            &venue,
            &handle,
            &recorder,
            now,
        )
        .await;
        assert!(out.accepted());
        assert_eq!(out.state.paper_capital, Some(dollars(12_000)));
        assert_eq!(venue.ledger_snapshot().collateral, dollars(12_000));

        // Adjust by a signed delta.
        let out = run_cmd(
            &mut control,
            DashboardCommand::AdjustPaperCapital(Decimal::from(-2_000)),
            &venue,
            &handle,
            &recorder,
            now,
        )
        .await;
        assert_eq!(out.state.paper_capital, Some(dollars(10_000)));
        assert_eq!(venue.ledger_snapshot().collateral, dollars(10_000));

        // Kill halts; reset clears (cancel-all is a no-op with no resting orders).
        assert!(!control.state.is_halted());
        let out = run_cmd(
            &mut control,
            DashboardCommand::Kill,
            &venue,
            &handle,
            &recorder,
            now,
        )
        .await;
        assert!(out.state.halted);
        assert!(control.state.is_halted());
        run_cmd(
            &mut control,
            DashboardCommand::Reset,
            &venue,
            &handle,
            &recorder,
            now,
        )
        .await;
        assert!(!control.state.is_halted());

        let _ = recorder.finish();
        let _ = std::fs::remove_dir_all(&dir);
        venue.shutdown();
    }
}
