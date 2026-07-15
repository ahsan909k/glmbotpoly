//! `bot paper-sim` — a live paper-trading smoke run (CLAUDE.md §9, §12.6).
//!
//! Wires the scheduler + feed-clob (real Polymarket book/trade data on the bus)
//! into a [`PaperVenue`] and runs a deliberately trivial two-sided post-only
//! quoter on the current window's Up token: every few seconds it cancels all and
//! re-quotes at the live touch, so the full place → rest → fill / cancel path is
//! exercised against real prints. It renders our quotes, the fills (with
//! maker/taker + fee), and the paper wallet in place, until ctrl-c.
//!
//! This is the integration demonstration; the conservative fill *rules*
//! (queue-behind-displayed, the cancel race, marketable depth-walking) are
//! pinned by `venue-paper`'s scripted unit tests. Only paper orders are placed —
//! nothing live is sent. [`paper_params`] is the §4 config→params boundary
//! mapping, mirroring `bot::live::live_params`.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use config::AppConfig;
use core_types::{
    BookSnapshot, ControlEvent, Dollars, Event, MarketInfo, NewOrder, OrderQty, OrderState,
    Outcome, Series, Side, Size, TokenId, TopOfBook, WindowLifecycle,
};
use feed_clob::{ClobArgs, ClobCommand, ClobStatus, WsTransport};
use scheduler::{SchedulerArgs, Timing};
use timeutil::wall_now;
use tokio::sync::{mpsc, watch};
use venue_api::{VenueEvent, VenueEvents, VenuePort};
use venue_paper::{LatencySpec, PaperLedgerSnapshot, PaperParams, PaperVenue};

use crate::discover::{fmt_countdown, fmt_ts};
use crate::feed::clob_params;

// Cadences for the live smoke loop.
/// Redraw cadence.
const RENDER_PERIOD: Duration = Duration::from_millis(500);
/// Re-quote cadence (cancel-all + fresh two-sided quote at the touch).
const QUOTE_PERIOD: Duration = Duration::from_secs(5);
/// Shares per quote (well above the 5-share / $1 minimums at these prices).
const QUOTE_SHARES: i64 = 10;
/// Recent venue events kept in the footer pane.
const EVENT_PANE_LINES: usize = 10;
/// One-shot demonstration of the runtime capital-adjust seam, fired this long
/// into the run.
const CAPITAL_DEMO_AFTER: Duration = Duration::from_secs(30);
/// The demonstration capital injection (dollars).
const CAPITAL_DEMO_DELTA: i64 = 1000;

/// Maps the `[paper]` config section into the plain [`PaperParams`] the venue
/// consumes (§4 boundary mapping; the venue crate never depends on `config`).
#[must_use]
pub fn paper_params(config: &AppConfig) -> PaperParams {
    let p = &config.paper;
    PaperParams {
        starting_capital: p.starting_capital,
        placement: LatencySpec {
            mean_ms: p.placement_latency.mean_ms,
            jitter_ms: p.placement_latency.jitter_ms,
        },
        cancel: LatencySpec {
            mean_ms: p.cancel_latency.mean_ms,
            jitter_ms: p.cancel_latency.jitter_ms,
        },
        venue_taker_delay_ms: p.venue_taker_delay_ms,
        default_fee_rate: p.default_fee_rate,
        maker_rebate_share: p.maker_rebate_share,
        taker_rebate_enabled: p.taker_rebate_enabled,
        event_channel_capacity: 1024,
        // Seed the latency jitter from the clock so runs differ; determinism
        // is unimportant for a live smoke.
        rng_seed: Some(u64::try_from(wall_now().as_millis()).unwrap_or(0) | 1),
    }
}

/// Builds the runtime and runs the paper-sim until ctrl-c.
pub fn execute(config: &AppConfig, series: Series) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building tokio runtime")?;
    runtime.block_on(run(config, series))
}

async fn run(config: &AppConfig, series: Series) -> anyhow::Result<()> {
    let service = discovery::DiscoveryService::from_config(&config.feeds, &config.discovery)
        .context("building discovery service")?;
    let starting_capital = config.paper.starting_capital;
    tracing::info!(
        target: "paper-sim",
        series = series.key(),
        starting_capital = %starting_capital,
        "paper-sim starting (real data, paper money; ctrl-c to stop)"
    );
    // Restore prior inventory + order state from the journal (§3/§9). Demonstrates
    // the boot-rebuild path; the live engine wiring that consumes it is deferred.
    let _restored = crate::boot::rebuild_and_log(config);

    let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(256);
    // A bus sender retained by the orchestrator so the capital-adjust command
    // can journal a `ControlEvent::PaperCapitalSet` (a future journal subscriber
    // persists it). venue-paper itself holds no bus sender.
    let control_tx = bus_tx.clone();
    let (window_tx, window_rx) = mpsc::channel::<(Arc<MarketInfo>, WindowLifecycle)>(64);
    let (market_tx, market_rx) = mpsc::channel(64);
    // The command plane is unused here, but ClobArgs expects the receiver.
    let (_command_tx, command_rx) = mpsc::channel::<ClobCommand>(8);
    let (clob_status_tx, clob_status_rx) = watch::channel(ClobStatus::default());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

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

    // The paper venue + its order/fill event stream.
    let mut venue = PaperVenue::spawn(paper_params(config), wall_now);
    let mut venue_rx = venue
        .take_event_rx()
        .context("paper venue event stream already taken")?;

    let mut render = tokio::time::interval(RENDER_PERIOD);
    render.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut quote = tokio::time::interval(QUOTE_PERIOD);
    quote.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut state = State::new(series, starting_capital);
    let start = tokio::time::Instant::now();
    let mut capital_demo_done = false;

    let results = loop {
        tokio::select! {
            joined = &mut sched_task => {
                let clob = drain_join(&shutdown_tx, &mut clob_task, &mut bus_rx).await;
                break (joined, clob);
            }
            joined = &mut clob_task => {
                let sched = drain_join(&shutdown_tx, &mut sched_task, &mut bus_rx).await;
                break (sched, joined);
            }
            maybe = bus_rx.recv() => match maybe {
                Some(event) => {
                    // feed-clob's supervisor consumes window announcements.
                    if let Event::Window { market, lifecycle } = &event {
                        let _ = window_tx.try_send((Arc::clone(market), *lifecycle));
                    }
                    // The paper venue consumes market data + window lifecycle
                    // (settlement happens here, before we read the balance below).
                    venue.on_bus_event(&event).await;
                    state.on_bus_event(&event);
                    if let Event::Window {
                        market,
                        lifecycle: WindowLifecycle::Resolved { outcome },
                    } = &event
                        && market.window.series == series
                    {
                        let bal = venue.balances().await.unwrap_or_default();
                        state.push_event(format!(
                            "SETTLED {} -> {outcome}  collateral {}",
                            market.event_slug, bal.collateral_total
                        ));
                    }
                }
                // run() retains `control_tx`, so the bus cannot close here.
                None => unreachable!("paper-sim retains a bus sender (control_tx)"),
            },
            maybe = venue_rx.recv() => {
                // `None` = venue task gone; keep rendering until shutdown.
                if let Some(event) = maybe {
                    state.on_venue_event(&event);
                }
            },
            _ = quote.tick() => {
                requote(&venue, &mut state).await;
            }
            _ = render.tick() => {
                // One-shot demonstration of the runtime capital-adjust seam.
                if !capital_demo_done && start.elapsed() >= CAPITAL_DEMO_AFTER {
                    capital_demo_done = true;
                    let delta = Dollars::new(rust_decimal::Decimal::from(CAPITAL_DEMO_DELTA));
                    venue.adjust_capital(delta);
                    let coll = venue.balances().await.unwrap_or_default().collateral_total;
                    // Journal the change on the bus for a future journal subscriber.
                    let _ = control_tx
                        .try_send(Event::Control(ControlEvent::PaperCapitalSet { amount: coll }));
                    state.push_event(format!("CAPITAL +{delta} -> collateral {coll}"));
                }
                let wallet = venue.balances().await.unwrap_or_default();
                let ledger = venue.ledger_snapshot();
                state.draw(&clob_status_rx.borrow(), &wallet, &ledger, wall_now())?;
            }
            signal = tokio::signal::ctrl_c() => {
                signal.context("listening for ctrl-c")?;
                tracing::info!(target: "paper-sim", "ctrl-c — cancelling paper orders and shutting down");
                let _ = venue.cancel_all().await;
                venue.shutdown();
                let _ = shutdown_tx.send(true);
                let sched = drain_join(&shutdown_tx, &mut sched_task, &mut bus_rx).await;
                let clob = drain_join(&shutdown_tx, &mut clob_task, &mut bus_rx).await;
                break (sched, clob);
            }
        }
    };

    let (sched_result, clob_result) = results;
    sched_result
        .context("scheduler driver panicked")?
        .context("scheduler driver failed")?;
    clob_result
        .context("clob feed panicked")?
        .context("clob feed failed")?;
    println!(
        "\npaper-sim summary: {} fills ({} maker / {} taker), fees {}, rebate credited {}, \
         {} merges, final collateral {}",
        state.fill_count,
        state.maker_fills,
        state.taker_fills,
        state.fees_paid,
        state.rebate_credited,
        state.merges,
        state.collateral
    );
    Ok(())
}

/// Cancels all resting paper quotes and re-quotes at the live touch. It bids and
/// offers the Up token and *also* bids the Down token, so matched Up/Down pairs
/// can form — which it then recycles to cash via an instant merge (the §8
/// capital-recycling mechanic). A no-op until we have a window and a book;
/// post-only-safe by quoting *at* the touch (joining, never crossing).
async fn requote(venue: &PaperVenue, state: &mut State) {
    let Some(market) = state.current.clone() else {
        return;
    };
    // Recycle any matched pairs accumulated since the last round.
    try_merge(venue, state, &market).await;

    let Some(top) = state.up_top else {
        return;
    };
    let (Some(bid), Some(ask)) = (top.bid, top.ask) else {
        return;
    };
    let _ = venue.cancel_all().await;
    let qty = match Size::new(rust_decimal::Decimal::from(QUOTE_SHARES)) {
        Ok(s) => OrderQty::Shares(s),
        Err(_) => return,
    };
    let buy = NewOrder {
        client_id: Some("paper-sim-bid".to_owned()),
        window: market.window,
        token_id: market.tokens.up.clone(),
        outcome: Outcome::Up,
        side: Side::Buy,
        price: bid.price,
        qty,
        tif: core_types::TimeInForce::Gtc { post_only: true },
    };
    let sell = NewOrder {
        side: Side::Sell,
        price: ask.price,
        client_id: Some("paper-sim-ask".to_owned()),
        ..buy.clone()
    };
    let _ = venue.place(&buy).await;
    let _ = venue.place(&sell).await;
    state.quotes_placed += 2;

    // Bid the Down token at its touch so matched pairs (and merges) can form.
    if let Some(dbid) = state.down_top.and_then(|t| t.bid) {
        let down_buy = NewOrder {
            client_id: Some("paper-sim-down-bid".to_owned()),
            token_id: market.tokens.down.clone(),
            outcome: Outcome::Down,
            price: dbid.price,
            ..buy.clone()
        };
        let _ = venue.place(&down_buy).await;
        state.quotes_placed += 1;
    }
}

/// Merges any matched Up/Down pairs currently held in `market`'s window back into
/// collateral (instant simulated CTF merge, §8).
async fn try_merge(venue: &PaperVenue, state: &mut State, market: &Arc<MarketInfo>) {
    let snap = venue.ledger_snapshot();
    let net = |tok: &TokenId| {
        snap.positions
            .iter()
            .find(|(t, _)| t == tok)
            .map_or(rust_decimal::Decimal::ZERO, |(_, n)| *n)
    };
    let matched = net(&market.tokens.up).min(net(&market.tokens.down));
    if matched <= rust_decimal::Decimal::ZERO {
        return;
    }
    let Ok(pairs) = Size::new(matched) else {
        return;
    };
    if let Ok(merged) = venue.merge(market.window, pairs).await
        && !merged.is_zero()
    {
        state.merges += 1;
        state.push_event(format!("MERGE {merged} pairs -> collateral"));
    }
}

/// Signals shutdown and awaits one task while draining the bus (a full channel
/// must never deadlock an exiting driver). Shared with `bot dashboard`.
pub(crate) async fn drain_join<T>(
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

/// Everything the renderer tracks.
struct State {
    series: Series,
    starting_capital: Dollars,
    current: Option<Arc<MarketInfo>>,
    lifecycle: Option<WindowLifecycle>,
    up_top: Option<TopOfBook>,
    down_top: Option<TopOfBook>,
    /// Working orders we placed: id → (side, price, filled).
    working: HashMap<core_types::OrderId, (Side, core_types::Price, Size)>,
    quotes_placed: u64,
    fill_count: u64,
    maker_fills: u64,
    taker_fills: u64,
    merges: u64,
    fees_paid: Dollars,
    rebate_accrued: Dollars,
    rebate_credited: Dollars,
    collateral: Dollars,
    events: std::collections::VecDeque<String>,
}

impl State {
    fn new(series: Series, starting_capital: Dollars) -> Self {
        Self {
            series,
            starting_capital,
            current: None,
            lifecycle: None,
            up_top: None,
            down_top: None,
            working: HashMap::new(),
            quotes_placed: 0,
            fill_count: 0,
            maker_fills: 0,
            taker_fills: 0,
            merges: 0,
            fees_paid: Dollars::ZERO,
            rebate_accrued: Dollars::ZERO,
            rebate_credited: Dollars::ZERO,
            collateral: starting_capital,
            events: std::collections::VecDeque::new(),
        }
    }

    fn push_event(&mut self, line: String) {
        if self.events.len() >= EVENT_PANE_LINES {
            self.events.pop_front();
        }
        self.events
            .push_back(format!("{} {line}", fmt_ts(wall_now())));
    }

    fn on_bus_event(&mut self, event: &Event) {
        match event {
            Event::Window { market, lifecycle } => {
                if market.window.series != self.series {
                    return;
                }
                match lifecycle {
                    WindowLifecycle::Open => {
                        self.current = Some(Arc::clone(market));
                        self.lifecycle = Some(*lifecycle);
                        self.up_top = None;
                        self.down_top = None;
                        self.working.clear();
                        self.push_event(format!("OPEN {}", market.event_slug));
                    }
                    WindowLifecycle::Closing | WindowLifecycle::Closed => {
                        if self.current.as_ref().map(|m| m.window) == Some(market.window) {
                            self.lifecycle = Some(*lifecycle);
                        }
                    }
                    // Settlement (and its collateral jump) is surfaced by
                    // `run()`, which reads the post-settle balance.
                    WindowLifecycle::Resolved { .. } => {}
                    WindowLifecycle::Discovered => {}
                }
            }
            Event::Book(snap) => self.note_top(snap),
            Event::TopOfBook { token_id, top } => self.note_token_top(token_id, *top),
            _ => {}
        }
    }

    fn note_top(&mut self, snap: &BookSnapshot) {
        self.note_token_top(&snap.token_id, snap.top());
    }

    fn note_token_top(&mut self, token: &TokenId, top: TopOfBook) {
        let Some(market) = &self.current else {
            return;
        };
        if *token == market.tokens.up {
            self.up_top = Some(top);
        } else if *token == market.tokens.down {
            self.down_top = Some(top);
        }
    }

    fn on_venue_event(&mut self, event: &VenueEvent) {
        match event {
            VenueEvent::Order(u) => {
                if u.state.is_terminal() {
                    self.working.remove(&u.order_id);
                    if u.state == OrderState::Rejected {
                        self.push_event(format!(
                            "REJECT {} {:?} {}",
                            short(u.order_id.as_str()),
                            u.side,
                            u.reject_reason.as_deref().unwrap_or("")
                        ));
                    }
                } else {
                    self.working
                        .insert(u.order_id.clone(), (u.side, u.price, u.filled_size));
                }
            }
            VenueEvent::Fill(f) => {
                self.fill_count += 1;
                match f.liquidity {
                    core_types::Liquidity::Maker => self.maker_fills += 1,
                    core_types::Liquidity::Taker => self.taker_fills += 1,
                }
                // `fees_paid`/rebate lines are sourced authoritatively from the
                // ledger snapshot in `draw`.
                self.push_event(format!(
                    "FILL {:?} {} @ {} ({:?}{})",
                    f.side,
                    f.size,
                    f.price.as_decimal(),
                    f.liquidity,
                    if f.fee.is_zero() {
                        String::new()
                    } else {
                        format!(", fee {}", f.fee)
                    }
                ));
            }
            // Paper has no user channel; this is a live-only risk-manager signal.
            VenueEvent::Connectivity { .. } => {}
        }
    }

    fn draw(
        &mut self,
        clob: &ClobStatus,
        wallet: &venue_api::Wallet,
        ledger: &PaperLedgerSnapshot,
        now: core_types::TimestampMs,
    ) -> anyhow::Result<()> {
        self.collateral = wallet.collateral_total;
        self.fees_paid = ledger.fees_paid;
        self.rebate_accrued = ledger.rebate_accrued;
        self.rebate_credited = ledger.rebate_credited;
        let mut frame = String::with_capacity(2_048);
        frame.push_str("\x1b[H\x1b[J");
        frame.push_str(&format!(
            "paper-sim [{}]  (real data, paper money)\n",
            self.series.key()
        ));
        match &self.current {
            Some(market) => {
                let closes_in = market.close_time.signed_duration_since(now);
                frame.push_str(&format!(
                    "{}  phase {:<9} closes in {:>8}  polymarket.com/event/{}\n",
                    market.window,
                    self.lifecycle
                        .map_or_else(|| "-".to_owned(), |l| format!("{l:?}")),
                    fmt_countdown(closes_in),
                    market.event_slug
                ));
                let conns: usize = clob.windows.iter().filter(|w| w.connected).count();
                frame.push_str(&format!(
                    "feed: {conns} connection(s) up   {}\n",
                    fmt_top("Up", self.up_top)
                ));
                frame.push_str(&format!(
                    "                          {}\n",
                    fmt_top("Down", self.down_top)
                ));
            }
            None => frame.push_str("waiting for the current window…\n"),
        }
        frame.push('\n');

        frame.push_str("our quotes:\n");
        if self.working.is_empty() {
            frame.push_str("  (none resting)\n");
        }
        let mut working: Vec<_> = self.working.iter().collect();
        working.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        for (id, (side, price, filled)) in working {
            frame.push_str(&format!(
                "  {:<14} {:?} @ {} filled {}\n",
                short(id.as_str()),
                side,
                price.as_decimal(),
                filled
            ));
        }
        frame.push('\n');

        let pnl = self.collateral - self.starting_capital;
        frame.push_str(&format!(
            "wallet: collateral {}  (start {}, Δ {})  positions {}\n",
            self.collateral,
            self.starting_capital,
            pnl,
            wallet.positions.len()
        ));
        for pos in &wallet.positions {
            frame.push_str(&format!(
                "  {} : {}\n",
                short(pos.token_id.as_str()),
                pos.size
            ));
        }
        frame.push_str(&format!(
            "stats: quotes placed {}  fills {} ({} maker / {} taker)  merges {}\n",
            self.quotes_placed, self.fill_count, self.maker_fills, self.taker_fills, self.merges
        ));
        frame.push_str(&format!(
            "income: fees {}  rebate accrued {}  credited {}\n",
            self.fees_paid, self.rebate_accrued, self.rebate_credited
        ));
        frame.push('\n');

        frame.push_str("recent events:\n");
        if self.events.is_empty() {
            frame.push_str("  (none yet)\n");
        }
        for line in &self.events {
            frame.push_str("  ");
            frame.push_str(line);
            frame.push('\n');
        }

        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(frame.as_bytes())
            .context("writing paper-sim frame")?;
        stdout.flush().context("flushing paper-sim frame")
    }
}

fn fmt_top(label: &str, top: Option<TopOfBook>) -> String {
    match top {
        Some(t) => format!(
            "{label}: bid {} ask {} mid {}",
            t.bid
                .map_or_else(|| "-".to_owned(), |l| l.price.as_decimal().to_string()),
            t.ask
                .map_or_else(|| "-".to_owned(), |l| l.price.as_decimal().to_string()),
            t.mid().map_or_else(|| "-".to_owned(), |m| m.to_string()),
        ),
        None => format!("{label}: (no book yet)"),
    }
}

fn short(s: &str) -> &str {
    &s[s.len().saturating_sub(12)..]
}
