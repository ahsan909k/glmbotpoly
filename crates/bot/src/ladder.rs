//! `bot ladder` — the live L2 ladder smoke run (read-only): runs the
//! scheduler for ONE series with the feed-clob supervisor attached, renders
//! an in-place ladder of the current window's Up/Down books every 250 ms,
//! and — for the first time anywhere — wires feed-clob's deduplicated
//! `market_resolved` events into the scheduler's `market_rx`, so parked
//! windows resolve live (`RESOLVED <slug> -> Up|Down` in the events pane
//! comes from the scheduler's bus announcement, proving the end-to-end
//! path).
//!
//! Verification aids: the header prints `polymarket.com/event/<slug>` for
//! side-by-side comparison with the website; `--recycle-after <secs>` drops
//! the current window's connection once to demonstrate reconnect-and-resync
//! continuity; `--raw <file>` taps every frame of every connection to a
//! JSONL capture (the fixture path for `crates/feed-clob/tests/fixtures/`).
//! The screen is redrawn in place — console log lines get overwritten;
//! everything also lands in the rolling log file.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{
    BookHealth, BookSnapshot, BookUnreliableReason, Decimal, Event, MarketInfo, Outcome, Series,
    TickSize, TimestampMs, TokenId, WindowLifecycle,
};
use feed_clob::{ClobArgs, ClobCommand, ClobStatus, TapTransport, WsTransport};
use scheduler::{SchedulerArgs, Timing};
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};

use crate::discover::{fmt_countdown, fmt_ts};
use crate::feed::{clob_params, spawn_raw_writer};

/// Redraw cadence.
const RENDER_PERIOD: Duration = Duration::from_millis(250);

/// Recent notable events kept in the footer pane.
const EVENT_PANE_LINES: usize = 8;

/// Builds the runtime and runs the ladder until ctrl-c.
pub fn execute(
    config: &AppConfig,
    series: Series,
    depth: usize,
    raw: Option<&Path>,
    recycle_after: Option<u64>,
) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run_ladder(config, series, depth, raw, recycle_after))
}

async fn run_ladder(
    config: &AppConfig,
    series: Series,
    depth: usize,
    raw: Option<&Path>,
    recycle_after: Option<u64>,
) -> anyhow::Result<()> {
    let service = discovery::DiscoveryService::from_config(&config.feeds, &config.discovery)
        .context("building discovery service")?;
    tracing::info!(
        target: "ladder",
        series = series.key(),
        depth,
        raw = raw.map(|p| p.display().to_string()),
        recycle_after,
        "ladder run starting (read-only; ctrl-c to stop)"
    );

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    let (window_tx, window_rx) = mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(64);
    let (market_tx, market_rx) = mpsc::channel(64);
    let (command_tx, command_rx) = mpsc::channel::<ClobCommand>(8);
    let (clob_status_tx, clob_status_rx) = watch::channel(ClobStatus::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // The scheduler drives ONLY the chosen series — and, for the first time,
    // consumes market lifecycle events (expect_resolutions flips on).
    let mut sched_task = tokio::spawn(scheduler::run(SchedulerArgs {
        timing: Timing::from_config(&config.scheduler),
        series: vec![series],
        refresher: service,
        now_fn: wall_now,
        bus_tx: bus_tx.clone(),
        market_rx: Some(market_rx),
        status_tx: None,
        shutdown_rx: shutdown_tx.subscribe(),
    }));

    // The clob supervisor; with --raw, every connection's frames are tapped
    // into one JSONL file (interleaved across connections — fixtures filter
    // by content).
    let params = clob_params(&config.feeds);
    let (mut clob_task, writer) = match raw {
        Some(path) => {
            let (tap_tx, tap_rx) = mpsc::unbounded_channel();
            let writer = spawn_raw_writer(path.to_path_buf(), tap_rx)?;
            let task = tokio::spawn(feed_clob::run(ClobArgs {
                params,
                transport_factory: move || TapTransport::new(WsTransport, tap_tx.clone()),
                now_fn: wall_now,
                bus_tx,
                window_rx,
                market_tx: Some(market_tx),
                command_rx: Some(command_rx),
                status_tx: Some(clob_status_tx),
                shutdown_rx,
                backoff_seed: None,
            }));
            (task, Some(writer))
        }
        None => (
            tokio::spawn(feed_clob::run(ClobArgs {
                params,
                transport_factory: || WsTransport,
                now_fn: wall_now,
                bus_tx,
                window_rx,
                market_tx: Some(market_tx),
                command_rx: Some(command_rx),
                status_tx: Some(clob_status_tx),
                shutdown_rx,
                backoff_seed: None,
            })),
            None,
        ),
    };

    let mut render = tokio::time::interval(RENDER_PERIOD);
    render.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut state = RenderState::new(series, depth);
    let recycle_at = recycle_after.map(|secs| {
        wall_now().saturating_add(core_types::DurationMs::from_secs(
            i64::try_from(secs).unwrap_or(i64::MAX),
        ))
    });
    let mut recycle_pending = recycle_at;
    let started = wall_now();

    let results = loop {
        tokio::select! {
            joined = &mut sched_task => {
                let clob = shutdown_and_join(&shutdown_tx, &mut clob_task, &mut bus_rx).await;
                break (joined, clob);
            }
            joined = &mut clob_task => {
                let sched = shutdown_and_join(&shutdown_tx, &mut sched_task, &mut bus_rx).await;
                break (sched, joined);
            }
            maybe = bus_rx.recv() => match maybe {
                Some(event) => state.on_event(&event, &window_tx),
                None => unreachable!("ladder holds a bus sender only inside the drivers"),
            },
            _ = render.tick() => {
                let now = wall_now();
                if let Some(at) = recycle_pending
                    && now >= at
                    && let Some(market) = &state.current
                {
                    tracing::info!(
                        target: "ladder", window = %market.window,
                        "sending forced recycle (--recycle-after)"
                    );
                    let _ = command_tx.try_send(ClobCommand::RecycleWindow(market.window));
                    state.push_event(now, "FORCED RECYCLE sent (--recycle-after)".to_owned());
                    recycle_pending = None;
                }
                state.draw(&clob_status_rx.borrow(), now)?;
            }
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "ladder", "ctrl-c — shutting down");
                let _ = shutdown_tx.send(true);
                let sched = shutdown_and_join(&shutdown_tx, &mut sched_task, &mut bus_rx).await;
                let clob = shutdown_and_join(&shutdown_tx, &mut clob_task, &mut bus_rx).await;
                break (sched, clob);
            }
        }
    };

    if let Some(writer) = writer {
        match writer.join() {
            Ok(Ok(lines)) => println!("raw capture: {lines} frames written"),
            Ok(Err(error)) => tracing::warn!(target: "ladder", %error, "raw capture writer failed"),
            Err(_) => tracing::warn!(target: "ladder", "raw capture writer panicked"),
        }
    }
    let ran_for = wall_now().signed_duration_since(started);
    println!(
        "\nrun summary: {} books, {} trades, {} health transitions, {} resolutions over {}",
        state.book_count,
        state.trade_count,
        state.health_count,
        state.resolved_count,
        fmt_countdown(ran_for)
    );
    let (sched_result, clob_result) = results;
    sched_result
        .context("scheduler driver panicked")?
        .context("scheduler driver failed")?;
    clob_result
        .context("clob feed panicked")?
        .context("clob feed failed")?;
    Ok(())
}

/// Signals shutdown and awaits one task while draining the bus (a full
/// channel must never deadlock an exiting driver).
async fn shutdown_and_join<T>(
    shutdown_tx: &watch::Sender<bool>,
    task: &mut tokio::task::JoinHandle<T>,
    bus_rx: &mut mpsc::Receiver<Event>,
) -> Result<T, tokio::task::JoinError> {
    let _ = shutdown_tx.send(true);
    loop {
        tokio::select! {
            joined = &mut *task => break joined,
            _ = bus_rx.recv() => {}
        }
    }
}

/// Everything the renderer knows.
struct RenderState {
    series: Series,
    depth: usize,
    /// The window currently rendered (Open/Closing/Closed).
    current: Option<Arc<MarketInfo>>,
    lifecycle: Option<WindowLifecycle>,
    /// The next window, if announced.
    next: Option<Arc<MarketInfo>>,
    books: HashMap<TokenId, Arc<BookSnapshot>>,
    trades: HashMap<TokenId, (Decimal, Decimal)>,
    tick: Option<TickSize>,
    trusted: Option<bool>,
    events: VecDeque<String>,
    book_count: u64,
    trade_count: u64,
    health_count: u64,
    resolved_count: u64,
}

impl RenderState {
    fn new(series: Series, depth: usize) -> Self {
        Self {
            series,
            depth: depth.max(1),
            current: None,
            lifecycle: None,
            next: None,
            books: HashMap::new(),
            trades: HashMap::new(),
            tick: None,
            trusted: None,
            events: VecDeque::new(),
            book_count: 0,
            trade_count: 0,
            health_count: 0,
            resolved_count: 0,
        }
    }

    fn push_event(&mut self, now: TimestampMs, line: String) {
        if self.events.len() >= EVENT_PANE_LINES {
            self.events.pop_front();
        }
        self.events.push_back(format!("{} {line}", fmt_ts(now)));
    }

    fn on_event(
        &mut self,
        event: &Event,
        window_tx: &mpsc::Sender<(Arc<MarketInfo>, WindowLifecycle)>,
    ) {
        let now = wall_now();
        match event {
            Event::Window { market, lifecycle } => {
                // Forward EVERY announcement into the supervisor; try_send on
                // purpose (never block the bus drain — connects are
                // idempotent, a dropped event is recoverable).
                if let Err(error) = window_tx.try_send((Arc::clone(market), *lifecycle)) {
                    tracing::warn!(
                        target: "ladder", %error,
                        "window forward dropped (supervisor busy?)"
                    );
                }
                if market.window.series != self.series {
                    return;
                }
                match lifecycle {
                    WindowLifecycle::Discovered => {
                        if self.current.as_ref().map(|m| m.window) != Some(market.window) {
                            self.next = Some(Arc::clone(market));
                        }
                    }
                    WindowLifecycle::Open => {
                        self.push_event(now, format!("OPEN {}", market.event_slug));
                        self.current = Some(Arc::clone(market));
                        self.lifecycle = Some(*lifecycle);
                        self.tick = Some(market.tick_size);
                        if self.next.as_ref().map(|m| m.window) == Some(market.window) {
                            self.next = None;
                        }
                        self.prune(now);
                    }
                    WindowLifecycle::Closing | WindowLifecycle::Closed => {
                        if self.current.as_ref().map(|m| m.window) == Some(market.window) {
                            self.lifecycle = Some(*lifecycle);
                        }
                    }
                    WindowLifecycle::Resolved { outcome } => {
                        self.resolved_count += 1;
                        self.push_event(
                            now,
                            format!("RESOLVED {} -> {outcome}", market.event_slug),
                        );
                    }
                }
            }
            Event::Book(snapshot) => {
                self.book_count += 1;
                self.books
                    .insert(snapshot.token_id.clone(), Arc::clone(snapshot));
            }
            Event::TopOfBook { .. } => {}
            Event::LastTrade {
                token_id,
                price,
                size,
                ..
            } => {
                self.trade_count += 1;
                self.trades.insert(
                    token_id.as_ref().clone(),
                    (price.as_decimal(), size.as_decimal()),
                );
            }
            Event::TickSizeChange {
                condition_id,
                new_tick,
                ..
            } => {
                if self
                    .current
                    .as_ref()
                    .is_some_and(|m| m.condition_id == **condition_id)
                {
                    self.tick = Some(*new_tick);
                    self.push_event(now, format!("TICK SIZE -> {}", new_tick.as_decimal()));
                }
            }
            Event::BookHealth(health) => {
                self.health_count += 1;
                match health {
                    BookHealth::Unreliable { window, reason, .. } => {
                        if *window == self.current_window_or(*window) {
                            self.trusted = Some(false);
                        }
                        self.push_event(
                            now,
                            format!("BOOKS UNRELIABLE {window}: {}", reason_label(*reason)),
                        );
                    }
                    BookHealth::Recovered { window, outage, .. } => {
                        if *window == self.current_window_or(*window) {
                            self.trusted = Some(true);
                        }
                        self.push_event(
                            now,
                            format!(
                                "BOOKS RECOVERED {window} (outage {})",
                                fmt_countdown(*outage)
                            ),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    fn current_window_or(&self, fallback: core_types::WindowId) -> core_types::WindowId {
        self.current.as_ref().map_or(fallback, |m| m.window)
    }

    /// Drops book/trade state for tokens no longer in the current or next
    /// window (memory hygiene across rollovers).
    fn prune(&mut self, _now: TimestampMs) {
        let keep: Vec<TokenId> = self
            .current
            .iter()
            .chain(self.next.iter())
            .flat_map(|m| [m.tokens.up.clone(), m.tokens.down.clone()])
            .collect();
        self.books.retain(|token, _| keep.contains(token));
        self.trades.retain(|token, _| keep.contains(token));
    }

    /// Full-frame in-place redraw.
    fn draw(&self, clob: &ClobStatus, now: TimestampMs) -> anyhow::Result<()> {
        let mut frame = String::with_capacity(4_096);
        frame.push_str("\x1b[H\x1b[J");
        let Some(market) = &self.current else {
            frame.push_str(&format!(
                "ladder [{}] waiting for the current window (discovery + scheduler warming up)…\n",
                self.series.key()
            ));
            print_frame(&frame)?;
            return Ok(());
        };
        let closes_in = market.close_time.signed_duration_since(now);
        frame.push_str(&format!(
            "{} {}  polymarket.com/event/{}\n",
            self.series.key(),
            market.window,
            market.event_slug
        ));
        // Trust comes from the live status watch (the latched bus events are
        // silent on the boot path by design); bus transitions override
        // between status pushes.
        let trusted = clob
            .windows
            .iter()
            .find(|w| w.window == market.window)
            .map(|w| w.machine.trusted)
            .or(self.trusted);
        frame.push_str(&format!(
            "phase {:<10} closes in {:>8}  tick {}  books {}\n",
            self.lifecycle
                .map_or_else(|| "-".to_owned(), |l| format!("{l:?}")),
            fmt_countdown(closes_in),
            self.tick
                .map_or_else(|| "-".to_owned(), |t| t.as_decimal().to_string()),
            match trusted {
                Some(true) => "TRUSTED",
                Some(false) => "UNRELIABLE",
                None => "warming up",
            },
        ));
        for status in &clob.windows {
            frame.push_str(&format!(
                "conn {:<26} {}  episodes {}  recycles {}  drift {}  anomalies {}\n",
                status.window.to_string(),
                if status.connected { "UP" } else { "DOWN" },
                status.episodes,
                status.machine.recycles,
                status.machine.drift,
                status.machine.anomalies,
            ));
        }
        frame.push('\n');
        for outcome in Outcome::ALL {
            let token = market.tokens.get(outcome);
            self.draw_side(&mut frame, outcome, token);
        }
        frame.push_str("recent events:\n");
        if self.events.is_empty() {
            frame.push_str("  (none yet)\n");
        }
        for line in &self.events {
            frame.push_str("  ");
            frame.push_str(line);
            frame.push('\n');
        }
        print_frame(&frame)
    }

    fn draw_side(&self, frame: &mut String, outcome: Outcome, token: &TokenId) {
        let book = self.books.get(token);
        let last_trade = self.trades.get(token);
        frame.push_str(&format!(
            "[{outcome}]  token …{}\n",
            tail(token.as_str(), 8)
        ));
        let Some(book) = book else {
            frame.push_str("  (no book yet)\n\n");
            return;
        };
        // Asks: worst of the displayed depth first, best last — the
        // conventional ladder reading order down to the spread.
        for level in book.asks.iter().take(self.depth).rev() {
            frame.push_str(&format!(
                "  {:>8}  {:>12}\n",
                level.price.as_decimal(),
                level.size.as_decimal()
            ));
        }
        let top = book.top();
        frame.push_str(&format!(
            "  ── mid {} spread {} last {} ──\n",
            top.mid().map_or_else(|| "-".to_owned(), |m| m.to_string()),
            top.spread()
                .map_or_else(|| "-".to_owned(), |s| s.to_string()),
            last_trade.map_or_else(|| "-".to_owned(), |(price, size)| format!("{price}×{size}")),
        ));
        for level in book.bids.iter().take(self.depth) {
            frame.push_str(&format!(
                "  {:>8}  {:>12}\n",
                level.price.as_decimal(),
                level.size.as_decimal()
            ));
        }
        frame.push('\n');
    }
}

fn reason_label(reason: BookUnreliableReason) -> &'static str {
    match reason {
        BookUnreliableReason::Disconnected => "disconnected",
        BookUnreliableReason::Stale => "stale book",
        BookUnreliableReason::Crossed => "crossed book",
        BookUnreliableReason::TopDivergence => "top divergence",
    }
}

fn tail(s: &str, n: usize) -> &str {
    &s[s.len().saturating_sub(n)..]
}

fn print_frame(frame: &str) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(frame.as_bytes())
        .context("writing ladder frame")?;
    stdout.flush().context("flushing ladder frame")
}
