//! Orchestrator binary (CLAUDE.md Â§4â€“5): wires every subsystem together â€”
//! feeds, scheduler, model, engines, venue adapter, journal, analytics,
//! dashboard, control plane â€” supervises the long-running tasks, and owns the
//! graceful-shutdown order (stop strategies -> cancel all open orders ->
//! flush journal -> exit).
//!
//! Current state: configuration and telemetry are wired; the full engine is
//! not. `paper` (the default) boots, logs the redacted effective configuration,
//! and exits cleanly. `paper-sim` runs the paper fill simulator (venue-paper)
//! against live market data with a trivial quoter. `check-config` prints the
//! effective configuration to stdout for inspection. `live` refuses: the §11
//! arming gates fail closed (gate 3, the dashboard arm, does not exist yet).
//!
//! Mode is a CLI subcommand on purpose: it is **not** a config value, so no
//! config file can ever push the bot toward live trading.

mod boot;
mod compare;
mod control;
mod control_cli;
mod dashboard;
mod depth_capture;
mod digest;
mod discover;
mod driver_attrib_record;
mod fair;
mod feed;
mod ladder;
mod latency;
mod live;
mod model_runtime;
mod model_taker_record;
mod paper;
mod record;
#[cfg(feature = "replay")]
mod replay_cmd;
mod run;
mod schedule;
mod shadow_stops_record;
mod telemetry;
mod timecfg;
mod vol;

// Test-exe hash remint marker (Windows App Control 4551 workaround; harmless).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, bail};
use config::{LiveArming, Secrets};
use core_types::Series;

use crate::control_cli::ControlSub;
use crate::feed::FeedSource;

const USAGE: &str = "usage: bot [run|paper|paper-sim|live|venue-check|check-config|discover|schedule|feed|\
                     compare|vol|ladder|fair|record|latency|dashboard|control|replay|sweep|digest] \
                     [--config-dir <path>] \
                     [--label <name>] [--out <file>] [--raw <file>] [--source rtds|binance] \
                     [--series <KEY>] [--depth <N>] [--recycle-after <secs>] [--out-dir <dir>] \
                     [--with-model] [--date YYYY-MM-DD]\n\
                     \n  digest [--date YYYY-MM-DD] [--out <file>]  (per-day operator digest from the \
                     sqlite index + side channels; no live process)\
                     \n  control <status|kill|reset|reset-daily-stop|set-capital AMT|adjust-capital DELTA|\
                     enable-series KEY|disable-series KEY|set-param KEY VAL [--series KEY]|\
                     arm-live|confirm-arm PHRASE|disarm>  (HTTP client to the running dashboard)\
                     \n  replay [<journal-dir>] [--window today|<N>d|all] [--out <file>]  (needs --features replay)\
                     \n  sweep  [<journal-dir>] [--min-edge a,b] [--gamma a,b] [--cancel-theta a,b] \
                     [--taker-buffer a,b] [--rank <col>] [--parallel] [--window ...] [--out <file>]  (needs --features replay)";

/// What the operator asked the binary to do.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    /// The main run mode (§5): start paper trading on all enabled series (or
    /// `--series`) concurrently under a supervision tree — feeds, scheduler,
    /// model, the risk-managed engine, the paper venue, journal, analytics, and
    /// the dashboard — with a resilient startup self-check and graceful
    /// shutdown. Real data, paper money. Runs until ctrl-c / SIGTERM.
    Run,
    /// Paper trading (default): real market data, simulated money.
    Paper,
    /// Live read-only paper-trading smoke run for one series' current window
    /// (`--series BTC-5m` default): scheduler + feed-clob wired into the paper
    /// venue with a trivial two-sided quoter, rendering fills and the paper
    /// wallet. Real data, paper money. Runs until ctrl-c.
    PaperSim,
    /// Live trading. Hard-gated (Â§11); refuses unless all three arming gates
    /// pass (gate 3, the dashboard arm, is a later task — so it fails closed).
    Live,
    /// Read-only demonstration of the live execution port: prints the live
    /// params and the constructed (unsent, unsigned) request for each order
    /// class. No credentials, no network.
    VenueCheck,
    /// Load + validate the configuration, print it, exit.
    CheckConfig,
    /// Live read-only probe: discover current + upcoming windows for all six
    /// series and print the table.
    Discover,
    /// Live read-only smoke run: roll all enabled series 24/7, logging every
    /// lifecycle event + a periodic coverage table, until ctrl-c (Â§6).
    Schedule,
    /// Live read-only feed run (`--source rtds` default, or `binance`):
    /// stream prices side by side with per-stream staleness ages, until
    /// ctrl-c (`--raw` taps every frame to a JSONL capture file).
    Feed,
    /// Live read-only feed comparison: both feeds on one bus, per-asset
    /// lead/lag + value-basis summary printed every minute, until ctrl-c.
    Compare,
    /// Live read-only vol-estimator smoke run: both feeds on one bus feeding
    /// four per-asset σ_1s estimator lanes, status table every second, until
    /// ctrl-c.
    Vol,
    /// Live read-only L2 ladder for one series' current window
    /// (`--series BTC-5m` default), with the scheduler + feed-clob wired
    /// end-to-end (`market_resolved` resolves parked windows). `--depth N`
    /// levels per side, `--raw` taps frames, `--recycle-after SECS` forces
    /// one reconnect to demonstrate continuity. Runs until ctrl-c.
    Ladder,
    /// Live read-only fair-value smoke run for one series' current window
    /// (`--series BTC-5m` default): the full model (strike capture, vol, basis,
    /// `p_up = Φ(z)`) next to the Polymarket book mid, with calibration records
    /// journaled to `data/calibration/` and post-hoc strike verification
    /// against the venue's resolved `priceToBeat`. Runs until ctrl-c.
    Fair,
    /// Live read-only journal capture: wire the scheduler + all feeds onto the
    /// bus and record every event to gzip-compressed, rotated JSONL segments
    /// under `--out-dir` (default `data/journal/`). Records all enabled series,
    /// or just `--series`. Runs until ctrl-c (days of capture roll across
    /// segments).
    Record,
    /// Latency benchmark against live endpoints; writes a JSON report
    /// (`--label` names the host/region, `--out` overrides the report path).
    Latency,
    /// Serve the dashboard (§10) over a live paper pipeline: scheduler +
    /// feed-clob + the paper venue across all enabled series (or `--series`),
    /// with REST + WebSocket on `config.dashboard.bind`. Real data, paper money.
    /// Runs until ctrl-c.
    Dashboard,
    /// Send a control-plane command to the running bot's dashboard control API
    /// (§10.6/§11): an HTTP client that POSTs/GETs `/api/control/*` and prints
    /// the JSON acknowledgment (the resulting state).
    Control(ControlSub),
    /// Deterministically re-run the full engine over a recorded journal
    /// (`<journal-dir>` positional, default `config.journal.dir`) and print the
    /// §10 series-comparison table. Requires `--features replay`.
    Replay,
    /// Run a parameter sweep over (`--min-edge`, `--gamma`, `--cancel-theta`,
    /// `--taker-buffer`) on a recorded journal and print a ranked comparison.
    /// Requires `--features replay`.
    Sweep,
    /// Write a per-day operator digest (per-driver PnL, maker fill quality,
    /// breakers, would-be shadow loss-stops, settlements) folded from the sqlite
    /// journal index + the driver-attribution and shadow-loss-stop side
    /// channels. Read-only over the WAL sqlite + gzip — needs no live process.
    /// `--date YYYY-MM-DD` (default today, UTC); `--out <file>` overrides the
    /// output path (default `data/digests/{YYYY-MM-DD}.md`).
    Digest,
}

#[derive(Debug, PartialEq, Eq)]
struct Cli {
    command: Command,
    config_dir: Option<PathBuf>,
    /// Host/region label for `latency` reports.
    label: Option<String>,
    /// Report path override for `latency`.
    out: Option<PathBuf>,
    /// Raw-frame JSONL capture path for `feed`/`ladder`.
    raw: Option<PathBuf>,
    /// Which feed `feed` runs (`rtds` default).
    source: Option<FeedSource>,
    /// Which series `ladder` renders (`BTC-5m` default).
    series: Option<Series>,
    /// Ladder depth (levels per side, default 8).
    depth: Option<usize>,
    /// Force one connection recycle after this many seconds (`ladder`).
    recycle_after: Option<u64>,
    /// Output directory for `record` journal segments (default `data/journal`).
    out_dir: Option<PathBuf>,
    /// Wire the fair-value model into `dashboard` (fair-vs-mid + live markout).
    with_model: bool,
    /// Positional journal directory for `replay`/`sweep` (default
    /// `config.journal.dir`). Only read under `--features replay`.
    #[cfg_attr(not(feature = "replay"), allow(dead_code))]
    journal_dir: Option<PathBuf>,
    /// Comparison window for `replay`/`sweep` (`today|<N>d|all`).
    window: Option<String>,
    /// `replay`/`sweep` grid: comma-separated `min_edge` values.
    min_edge: Option<String>,
    /// `sweep` grid: comma-separated skew-strength (`gamma`) values.
    gamma: Option<String>,
    /// `sweep` grid: comma-separated cancel-threshold (`reprice_theta`) values.
    cancel_theta: Option<String>,
    /// `sweep` grid: comma-separated taker-buffer values.
    taker_buffer: Option<String>,
    /// `sweep` ranking column key (default `net-pnl`).
    rank: Option<String>,
    /// `sweep`: fan grid points across worker threads.
    parallel: bool,
    /// `digest` day to report (`YYYY-MM-DD`, default today UTC).
    date: Option<String>,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut command = None;
    let mut config_dir = None;
    let mut label = None;
    let mut out = None;
    let mut raw = None;
    let mut source = None;
    let mut series = None;
    let mut depth = None;
    let mut recycle_after = None;
    let mut out_dir = None;
    let mut with_model = false;
    let mut journal_dir = None;
    let mut window = None;
    let mut min_edge = None;
    let mut gamma = None;
    let mut cancel_theta = None;
    let mut taker_buffer = None;
    let mut rank = None;
    let mut parallel = false;
    let mut date = None;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "run" | "paper" | "paper-sim" | "live" | "venue-check" | "check-config"
            | "discover" | "schedule" | "feed" | "compare" | "vol" | "ladder" | "fair"
            | "record" | "latency" | "dashboard" | "digest" => {
                if command.is_some() {
                    return Err(format!("more than one subcommand given (second: {arg:?})"));
                }
                command = Some(match arg.as_str() {
                    "run" => Command::Run,
                    "paper" => Command::Paper,
                    "paper-sim" => Command::PaperSim,
                    "live" => Command::Live,
                    "venue-check" => Command::VenueCheck,
                    "discover" => Command::Discover,
                    "schedule" => Command::Schedule,
                    "feed" => Command::Feed,
                    "compare" => Command::Compare,
                    "vol" => Command::Vol,
                    "ladder" => Command::Ladder,
                    "fair" => Command::Fair,
                    "record" => Command::Record,
                    "latency" => Command::Latency,
                    "dashboard" => Command::Dashboard,
                    "digest" => Command::Digest,
                    _ => Command::CheckConfig,
                });
            }
            "control" => {
                if command.is_some() {
                    return Err(format!("more than one subcommand given (second: {arg:?})"));
                }
                command = Some(Command::Control(parse_control_sub(&mut args)?));
            }
            "replay" | "sweep" => {
                if command.is_some() {
                    return Err(format!("more than one subcommand given (second: {arg:?})"));
                }
                command = Some(if arg == "replay" {
                    Command::Replay
                } else {
                    Command::Sweep
                });
                // Optional positional <journal-dir> (a non-flag token right after
                // the subcommand), like `control`'s positionals.
                if matches!(args.peek(), Some(v) if !v.starts_with("--"))
                    && let Some(dir) = args.next()
                {
                    journal_dir = Some(PathBuf::from(dir));
                }
            }
            "--config-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--config-dir requires a value".to_owned())?;
                config_dir = Some(PathBuf::from(value));
            }
            "--label" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--label requires a value".to_owned())?;
                label = Some(value);
            }
            "--out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--out requires a value".to_owned())?;
                out = Some(PathBuf::from(value));
            }
            "--raw" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--raw requires a value".to_owned())?;
                raw = Some(PathBuf::from(value));
            }
            "--source" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--source requires a value (rtds|binance)".to_owned())?;
                source = Some(value.parse::<FeedSource>()?);
            }
            "--series" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--series requires a value (e.g. BTC-5m)".to_owned())?;
                series = Some(parse_series(&value)?);
            }
            "--depth" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--depth requires a value".to_owned())?;
                depth = Some(parse_depth(&value)?);
            }
            "--recycle-after" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--recycle-after requires a value (seconds)".to_owned())?;
                recycle_after = Some(parse_secs(&value)?);
            }
            "--out-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--out-dir requires a value".to_owned())?;
                out_dir = Some(PathBuf::from(value));
            }
            "--with-model" => {
                with_model = true;
            }
            "--window" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--window requires a value (today|<N>d|all)".to_owned())?;
                window = Some(value);
            }
            "--date" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--date requires a value (YYYY-MM-DD)".to_owned())?;
                date = Some(value);
            }
            "--min-edge" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--min-edge requires a comma-separated list".to_owned())?;
                min_edge = Some(value);
            }
            "--gamma" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--gamma requires a comma-separated list".to_owned())?;
                gamma = Some(value);
            }
            "--cancel-theta" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--cancel-theta requires a comma-separated list".to_owned())?;
                cancel_theta = Some(value);
            }
            "--taker-buffer" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--taker-buffer requires a comma-separated list".to_owned())?;
                taker_buffer = Some(value);
            }
            "--rank" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--rank requires a value".to_owned())?;
                rank = Some(value);
            }
            "--parallel" => {
                parallel = true;
            }
            other => {
                if let Some(value) = other.strip_prefix("--config-dir=") {
                    config_dir = Some(PathBuf::from(value));
                } else if let Some(value) = other.strip_prefix("--label=") {
                    label = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--out=") {
                    out = Some(PathBuf::from(value));
                } else if let Some(value) = other.strip_prefix("--raw=") {
                    raw = Some(PathBuf::from(value));
                } else if let Some(value) = other.strip_prefix("--source=") {
                    source = Some(value.parse::<FeedSource>()?);
                } else if let Some(value) = other.strip_prefix("--series=") {
                    series = Some(parse_series(value)?);
                } else if let Some(value) = other.strip_prefix("--depth=") {
                    depth = Some(parse_depth(value)?);
                } else if let Some(value) = other.strip_prefix("--recycle-after=") {
                    recycle_after = Some(parse_secs(value)?);
                } else if let Some(value) = other.strip_prefix("--out-dir=") {
                    out_dir = Some(PathBuf::from(value));
                } else if let Some(value) = other.strip_prefix("--window=") {
                    window = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--date=") {
                    date = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--min-edge=") {
                    min_edge = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--gamma=") {
                    gamma = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--cancel-theta=") {
                    cancel_theta = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--taker-buffer=") {
                    taker_buffer = Some(value.to_owned());
                } else if let Some(value) = other.strip_prefix("--rank=") {
                    rank = Some(value.to_owned());
                } else {
                    return Err(format!("unrecognized argument: {other:?}"));
                }
            }
        }
    }
    let command = command.unwrap_or(Command::Paper);
    if label.is_some() && command != Command::Latency {
        return Err("--label only applies to the latency subcommand".to_owned());
    }
    if out.is_some()
        && !matches!(
            command,
            Command::Latency | Command::Replay | Command::Sweep | Command::Digest
        )
    {
        return Err(
            "--out only applies to the latency, replay, sweep, and digest subcommands".to_owned(),
        );
    }
    if date.is_some() && command != Command::Digest {
        return Err("--date only applies to the digest subcommand".to_owned());
    }
    if window.is_some() && !matches!(command, Command::Replay | Command::Sweep) {
        return Err("--window only applies to the replay and sweep subcommands".to_owned());
    }
    let sweep_only = min_edge.is_some()
        || gamma.is_some()
        || cancel_theta.is_some()
        || taker_buffer.is_some()
        || rank.is_some()
        || parallel;
    if sweep_only && command != Command::Sweep {
        return Err(
            "--min-edge/--gamma/--cancel-theta/--taker-buffer/--rank/--parallel only apply to the \
             sweep subcommand"
                .to_owned(),
        );
    }
    if raw.is_some() && command != Command::Feed && command != Command::Ladder {
        return Err("--raw only applies to the feed and ladder subcommands".to_owned());
    }
    if source.is_some() && command != Command::Feed {
        return Err("--source only applies to the feed subcommand".to_owned());
    }
    if series.is_some()
        && command != Command::Ladder
        && command != Command::Fair
        && command != Command::PaperSim
        && command != Command::Record
        && command != Command::Dashboard
        && command != Command::Run
    {
        return Err(
            "--series only applies to the run, ladder, fair, paper-sim, record, and dashboard \
             subcommands"
                .to_owned(),
        );
    }
    if (depth.is_some() || recycle_after.is_some()) && command != Command::Ladder {
        return Err("--depth/--recycle-after only apply to the ladder subcommand".to_owned());
    }
    if out_dir.is_some() && command != Command::Record {
        return Err("--out-dir only applies to the record subcommand".to_owned());
    }
    if with_model && command != Command::Dashboard {
        return Err("--with-model only applies to the dashboard subcommand".to_owned());
    }
    Ok(Cli {
        command,
        config_dir,
        label,
        out,
        raw,
        source,
        series,
        depth,
        recycle_after,
        out_dir,
        with_model,
        journal_dir,
        window,
        min_edge,
        gamma,
        cancel_theta,
        taker_buffer,
        rank,
        parallel,
        date,
    })
}

fn parse_series(value: &str) -> Result<Series, String> {
    value
        .parse::<Series>()
        .map_err(|_| format!("unknown series {value:?} (expected a key like BTC-5m, ETH-1h)"))
}

fn parse_depth(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!("--depth must be a positive integer, got {value:?}")),
    }
}

fn parse_secs(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("--recycle-after must be a number of seconds, got {value:?}"))
}

/// Parses a `bot control <sub> [args]` into a [`ControlSub`], consuming only the
/// tokens it needs (a fixed number of positionals, plus an optional `--series`
/// for `set-param`) so any remaining global flags flow back to the main parser.
fn parse_control_sub<I: Iterator<Item = String>>(
    args: &mut std::iter::Peekable<I>,
) -> Result<ControlSub, String> {
    let sub = args.next().ok_or_else(|| {
        "control requires a subcommand (status|kill|reset|reset-daily-stop|set-capital|\
         adjust-capital|enable-series|disable-series|set-param|arm-live|confirm-arm|disarm)"
            .to_owned()
    })?;
    let positional = |args: &mut std::iter::Peekable<I>, what: &str| -> Result<String, String> {
        match args.next() {
            Some(v) if !v.starts_with("--") => Ok(v),
            Some(v) => Err(format!("control {sub} expected {what}, got flag {v:?}")),
            None => Err(format!("control {sub} requires {what}")),
        }
    };
    Ok(match sub.as_str() {
        "status" => ControlSub::Status,
        "kill" => ControlSub::Kill,
        "reset" => ControlSub::Reset,
        "reset-daily-stop" => ControlSub::ResetDailyStop,
        "set-capital" => ControlSub::SetCapital(positional(args, "<amount>")?),
        "adjust-capital" => ControlSub::AdjustCapital(positional(args, "<delta>")?),
        "enable-series" => ControlSub::EnableSeries(positional(args, "<series-key>")?),
        "disable-series" => ControlSub::DisableSeries(positional(args, "<series-key>")?),
        "set-param" => {
            let key = positional(args, "<key>")?;
            let value = positional(args, "<value>")?;
            let series = match args.peek().map(String::as_str) {
                Some("--series") => {
                    args.next();
                    Some(
                        args.next()
                            .ok_or_else(|| "--series requires a value".to_owned())?,
                    )
                }
                Some(s) if s.starts_with("--series=") => {
                    let v = s["--series=".len()..].to_owned();
                    args.next();
                    Some(v)
                }
                _ => None,
            };
            ControlSub::SetParam { series, key, value }
        }
        "arm-live" => ControlSub::ArmLive,
        "confirm-arm" => ControlSub::ConfirmArm(positional(args, "<phrase>")?),
        "disarm" => ControlSub::Disarm,
        other => return Err(format!("unknown control subcommand {other:?}")),
    })
}

/// Config directory precedence: `--config-dir` flag > `BOT_CONFIG_DIR` env >
/// `./config`. An empty env value counts as unset (same convention as the
/// `BOT_SECRET_*` variables).
fn resolve_config_dir(cli_dir: Option<PathBuf>) -> PathBuf {
    cli_dir
        .or_else(|| {
            std::env::var_os("BOT_CONFIG_DIR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from("config"))
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let cli =
        parse_args(std::env::args().skip(1)).map_err(|msg| anyhow::anyhow!("{msg}\n{USAGE}"))?;
    let config_dir = resolve_config_dir(cli.config_dir);

    let (config, secrets) = config::load(&config_dir)?;
    config::validate(&config, &secrets)?;
    let arming = LiveArming::evaluate(&config.live, &secrets);

    // AppConfig contains no secrets by construction (SecretString is not
    // serializable), so rendering the whole tree is redaction-safe.
    let rendered =
        serde_json::to_string_pretty(&config).context("rendering effective configuration")?;

    match cli.command {
        Command::CheckConfig => {
            println!("configuration: OK (loaded from {})", config_dir.display());
            println!("{rendered}");
            print_presence(&secrets, arming);
            Ok(ExitCode::SUCCESS)
        }
        Command::Paper => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            log_boot("paper", &config_dir, &config, &secrets, arming, &rendered);
            // Restore inventory + order state from the journal (§3/§9). The
            // orchestrator that consumes this to seed the engine is a later task.
            let _restored = boot::rebuild_and_log(&config);
            tracing::info!("scaffold boot complete â€” engine not wired yet; exiting cleanly");
            Ok(ExitCode::SUCCESS)
        }
        Command::Discover => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                "live discovery probe (read-only)"
            );
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("building tokio runtime")?;
            let mut service =
                discovery::DiscoveryService::from_config(&config.feeds, &config.discovery)
                    .context("building discovery service")?;
            let now = timeutil::wall_now();
            let snapshot = runtime.block_on(service.refresh_all(now));
            discover::print_snapshot(&snapshot, now);
            Ok(if snapshot.all_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        Command::Schedule => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                "scheduler smoke run (read-only)"
            );
            schedule::execute(&config)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Feed => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            let source = cli.source.unwrap_or_default();
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                %source,
                "feed smoke run (read-only)"
            );
            feed::execute(&config, source, cli.raw.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Compare => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                "feed comparison run (read-only)"
            );
            compare::execute(&config)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Vol => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                "vol estimator smoke run (read-only)"
            );
            vol::execute(&config)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Ladder => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            let series = cli.series.unwrap_or_else(default_btc_5m_series);
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                series = series.key(),
                "ladder run (read-only)"
            );
            ladder::execute(
                &config,
                series,
                cli.depth.unwrap_or(8),
                cli.raw.as_deref(),
                cli.recycle_after,
            )?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Fair => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            let series = cli.series.unwrap_or_else(default_btc_5m_series);
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                series = series.key(),
                "fair-value smoke run (read-only)"
            );
            fair::execute(&config, series)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Record => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            let out_dir = cli
                .out_dir
                .clone()
                .unwrap_or_else(|| config.journal.dir.clone());
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                series = cli.series.map(|s| s.key()),
                out_dir = %out_dir.display(),
                "journal capture (read-only)"
            );
            record::execute(&config, cli.series, &out_dir)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::PaperSim => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            let series = cli.series.unwrap_or_else(default_btc_5m_series);
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                series = series.key(),
                "paper-sim run (real data, paper money)"
            );
            paper::execute(&config, series)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Run => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            log_boot("run", &config_dir, &config, &secrets, arming, &rendered);
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                bind = %config.dashboard.bind,
                series = cli.series.map(|s| s.key()),
                "main run mode (real data, paper money)"
            );
            run::execute(&config, &secrets, cli.series)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Dashboard => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                bind = %config.dashboard.bind,
                series = cli.series.map(|s| s.key()),
                with_model = cli.with_model,
                "dashboard run (real data, paper money)"
            );
            dashboard::execute(&config, &secrets, cli.series, cli.with_model)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Latency => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                label = cli.label.as_deref().unwrap_or("unlabeled"),
                "latency benchmark (read-only, live endpoints)"
            );
            latency::execute(&config, cli.label.as_deref(), cli.out.as_deref())
        }
        Command::VenueCheck => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            tracing::info!(
                version = env!("CARGO_PKG_VERSION"),
                config_dir = %config_dir.display(),
                "venue-check (read-only, no network)"
            );
            live::venue_check(&config)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Live => {
            let _guard = telemetry::init(&config.log).context("initializing telemetry")?;
            log_boot("live", &config_dir, &config, &secrets, arming, &rendered);
            // Restore inventory + order state from the journal before arming (§3/§9);
            // on live the venue's open-orders reconcile then corrects this view.
            let _restored = boot::rebuild_and_log(&config);
            // The live adapter's constructor is the gate: `connect` runs the
            // §11 arming check first and fails closed before any SDK client is
            // built or any network call is made. Gate 3 (dashboard arm) does not
            // exist yet, so this build always refuses â€” proving the wiring.
            let params = live::live_params(&config);
            let dashboard_armed = false;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("building tokio runtime")?;
            match runtime.block_on(venue_live::LiveVenue::connect(
                arming,
                dashboard_armed,
                &secrets,
                params,
            )) {
                Ok(_venue) => {
                    // Unreachable while gate 3 is hardwired off; guard anyway.
                    tracing::warn!("live venue constructed unexpectedly; refusing to trade");
                    bail!("live mode refused: dashboard arm (gate 3) is not implemented yet");
                }
                Err(e) => bail!(
                    "live mode refused: {e} (CLAUDE.md Â§11; gate 3 = dashboard arm \
                     is a later task)"
                ),
            }
        }
        Command::Control(sub) => {
            // A CLI HTTP client to the running dashboard's control API. No
            // telemetry — the JSON ack is printed to stdout.
            control_cli::execute(&config, &secrets, sub)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Replay => {
            // No telemetry: replay is a batch analysis tool whose output is the
            // stdout table; the engine's tracing would flood it. Errors surface
            // via the `Result`.
            #[cfg(feature = "replay")]
            {
                let journal_dir = cli
                    .journal_dir
                    .clone()
                    .unwrap_or_else(|| config.journal.dir.clone());
                let window = replay_cmd::parse_window(cli.window.as_deref())
                    .map_err(|m| anyhow::anyhow!(m))?;
                replay_cmd::execute_replay(&config, &journal_dir, window, cli.out.as_deref())?;
                Ok(ExitCode::SUCCESS)
            }
            #[cfg(not(feature = "replay"))]
            {
                bail!("the `replay` subcommand requires building with --features replay");
            }
        }
        Command::Sweep => {
            #[cfg(feature = "replay")]
            {
                let journal_dir = cli
                    .journal_dir
                    .clone()
                    .unwrap_or_else(|| config.journal.dir.clone());
                let args = replay_cmd::SweepArgs {
                    journal_dir: &journal_dir,
                    window: cli.window.as_deref(),
                    min_edge: cli.min_edge.as_deref(),
                    gamma: cli.gamma.as_deref(),
                    cancel_theta: cli.cancel_theta.as_deref(),
                    taker_buffer: cli.taker_buffer.as_deref(),
                    rank: cli.rank.as_deref(),
                    parallel: cli.parallel,
                    out: cli.out.as_deref(),
                };
                replay_cmd::execute_sweep(&config, &args)?;
                Ok(ExitCode::SUCCESS)
            }
            #[cfg(not(feature = "replay"))]
            {
                bail!("the `sweep` subcommand requires building with --features replay");
            }
        }
        Command::Digest => {
            // No telemetry (like replay): the digest writes a markdown file and
            // prints its path; tracing would only clutter that output.
            let date = cli.date.as_deref().map(parse_date).transpose()?;
            digest::execute(&config, date, cli.out.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Parses a `YYYY-MM-DD` string into a [`time::Date`] (no formatting feature
/// needed — split and validate the three components).
fn parse_date(s: &str) -> anyhow::Result<time::Date> {
    let mut parts = s.split('-');
    let (Some(y), Some(m), Some(d), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        bail!("--date must be YYYY-MM-DD, got {s:?}");
    };
    let year: i32 = y
        .parse()
        .map_err(|_| anyhow::anyhow!("--date has an invalid year in {s:?}"))?;
    let month_num: u8 = m
        .parse()
        .map_err(|_| anyhow::anyhow!("--date has an invalid month in {s:?}"))?;
    let day: u8 = d
        .parse()
        .map_err(|_| anyhow::anyhow!("--date has an invalid day in {s:?}"))?;
    let month = time::Month::try_from(month_num)
        .map_err(|_| anyhow::anyhow!("--date month must be 1..=12 in {s:?}"))?;
    time::Date::from_calendar_date(year, month, day)
        .map_err(|e| anyhow::anyhow!("--date {s:?} is not a valid calendar date: {e}"))
}

/// The acceptance-test series for `bot ladder` / `bot fair` when `--series` is
/// not given.
fn default_btc_5m_series() -> Series {
    Series {
        asset: core_types::Asset::Btc,
        duration: core_types::WindowDuration::M5,
    }
}

fn print_presence(secrets: &Secrets, arming: LiveArming) {
    println!("\nsecrets (values are env-only and never shown):");
    for (name, present) in secrets.presence() {
        println!("  {name}: {}", if present { "set" } else { "not set" });
    }
    println!(
        "live arming gates: config_enabled={} env_confirmed={} (gate 3 is the dashboard arm action)",
        arming.config_enabled, arming.env_confirmed
    );
}

fn log_boot(
    mode: &str,
    config_dir: &std::path::Path,
    config: &config::AppConfig,
    secrets: &Secrets,
    arming: LiveArming,
    rendered: &str,
) {
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        mode,
        config_dir = %config_dir.display(),
        "polymarket up/down bot starting"
    );
    tracing::info!(target: "bot::config", "effective configuration:\n{rendered}");
    for (name, present) in secrets.presence() {
        tracing::info!(target: "bot::config", secret = name, present, "secret presence");
    }
    tracing::info!(
        target: "bot::config",
        config_enabled = arming.config_enabled,
        env_confirmed = arming.env_confirmed,
        "live arming gates 1+2 (gate 3 is the dashboard arm action)"
    );
    let enabled: Vec<&'static str> = config
        .engine
        .enabled_series()
        .into_iter()
        .map(|s| s.key())
        .collect();
    tracing::info!(target: "bot::config", series = ?enabled, "enabled series");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, String> {
        parse_args(args.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn defaults_to_paper() {
        let cli = parse(&[]).unwrap();
        assert_eq!(cli.command, Command::Paper);
        assert_eq!(cli.config_dir, None);
    }

    #[test]
    fn parses_subcommands_and_config_dir() {
        let cli = parse(&["check-config", "--config-dir", "somewhere"]).unwrap();
        assert_eq!(cli.command, Command::CheckConfig);
        assert_eq!(cli.config_dir, Some(PathBuf::from("somewhere")));

        let cli = parse(&["live", "--config-dir=elsewhere"]).unwrap();
        assert_eq!(cli.command, Command::Live);
        assert_eq!(cli.config_dir, Some(PathBuf::from("elsewhere")));

        let cli = parse(&["discover"]).unwrap();
        assert_eq!(cli.command, Command::Discover);

        let cli = parse(&["schedule"]).unwrap();
        assert_eq!(cli.command, Command::Schedule);
    }

    #[test]
    fn parses_feed_with_raw_both_forms() {
        let cli = parse(&["feed"]).unwrap();
        assert_eq!(cli.command, Command::Feed);
        assert_eq!(cli.raw, None);
        assert_eq!(cli.source, None);

        let cli = parse(&["feed", "--raw", "data/captures/rtds.jsonl"]).unwrap();
        assert_eq!(cli.command, Command::Feed);
        assert_eq!(cli.raw, Some(PathBuf::from("data/captures/rtds.jsonl")));

        let cli = parse(&["feed", "--raw=cap.jsonl"]).unwrap();
        assert_eq!(cli.raw, Some(PathBuf::from("cap.jsonl")));
    }

    #[test]
    fn parses_feed_source_both_forms() {
        let cli = parse(&["feed", "--source", "binance"]).unwrap();
        assert_eq!(cli.command, Command::Feed);
        assert_eq!(cli.source, Some(FeedSource::Binance));

        let cli = parse(&["feed", "--source=rtds", "--raw=cap.jsonl"]).unwrap();
        assert_eq!(cli.source, Some(FeedSource::Rtds));
        assert_eq!(cli.raw, Some(PathBuf::from("cap.jsonl")));

        // Bad values and missing values are rejected.
        assert!(parse(&["feed", "--source", "coinbase"]).is_err());
        assert!(parse(&["feed", "--source"]).is_err());
    }

    #[test]
    fn parses_compare() {
        let cli = parse(&["compare"]).unwrap();
        assert_eq!(cli.command, Command::Compare);
        // compare always runs both feeds â€” --source doesn't apply.
        assert!(parse(&["compare", "--source", "binance"]).is_err());
        assert!(parse(&["compare", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["compare", "paper"]).is_err());
    }

    #[test]
    fn parses_vol() {
        let cli = parse(&["vol"]).unwrap();
        assert_eq!(cli.command, Command::Vol);
        // vol always runs both feeds with the four fixed lanes â€” no flags
        // from other subcommands apply.
        assert!(parse(&["vol", "--source", "binance"]).is_err());
        assert!(parse(&["vol", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["vol", "--series", "BTC-5m"]).is_err());
        assert!(parse(&["vol", "--label", "x"]).is_err());
        assert!(parse(&["vol", "paper"]).is_err());
    }

    #[test]
    fn rejects_raw_on_other_subcommands() {
        assert!(parse(&["paper", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["latency", "--raw=x.jsonl"]).is_err());
        // No subcommand defaults to paper â€” rejected there too.
        assert!(parse(&["--raw", "x.jsonl"]).is_err());
        assert!(parse(&["feed", "--raw"]).is_err());
        // And latency flags don't leak onto feed.
        assert!(parse(&["feed", "--label", "x"]).is_err());
        // --source is feed-only.
        assert!(parse(&["schedule", "--source", "binance"]).is_err());
        assert!(parse(&["--source=binance"]).is_err());
    }

    #[test]
    fn parses_ladder_flags_both_forms() {
        use core_types::{Asset, WindowDuration};

        let cli = parse(&["ladder"]).unwrap();
        assert_eq!(cli.command, Command::Ladder);
        assert_eq!(cli.series, None);
        assert_eq!(cli.depth, None);
        assert_eq!(cli.recycle_after, None);
        assert_eq!(cli.raw, None);

        let cli = parse(&[
            "ladder",
            "--series",
            "ETH-1h",
            "--depth",
            "12",
            "--recycle-after",
            "60",
            "--raw",
            "cap.jsonl",
        ])
        .unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::H1,
            })
        );
        assert_eq!(cli.depth, Some(12));
        assert_eq!(cli.recycle_after, Some(60));
        assert_eq!(cli.raw, Some(PathBuf::from("cap.jsonl")));

        let cli = parse(&[
            "ladder",
            "--series=BTC-15m",
            "--depth=4",
            "--recycle-after=30",
        ])
        .unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Btc,
                duration: WindowDuration::M15,
            })
        );
        assert_eq!(cli.depth, Some(4));
        assert_eq!(cli.recycle_after, Some(30));
    }

    #[test]
    fn rejects_bad_ladder_flags() {
        // Unknown series, case-sensitive keys, bad numbers.
        assert!(parse(&["ladder", "--series", "DOGE-5m"]).is_err());
        assert!(parse(&["ladder", "--series", "btc-5m"]).is_err());
        assert!(parse(&["ladder", "--depth", "0"]).is_err());
        assert!(parse(&["ladder", "--depth", "many"]).is_err());
        assert!(parse(&["ladder", "--recycle-after", "-5"]).is_err());
        assert!(parse(&["ladder", "--series"]).is_err());
        // Ladder flags don't leak onto other subcommands.
        assert!(parse(&["feed", "--series", "BTC-5m"]).is_err());
        assert!(parse(&["schedule", "--depth", "8"]).is_err());
        assert!(parse(&["--recycle-after", "60"]).is_err());
        // Feed/latency flags don't leak onto ladder.
        assert!(parse(&["ladder", "--source", "rtds"]).is_err());
        assert!(parse(&["ladder", "--label", "x"]).is_err());
    }

    #[test]
    fn parses_fair_flags() {
        use core_types::{Asset, WindowDuration};

        // Bare fair defaults to BTC-5m at dispatch (series None here).
        let cli = parse(&["fair"]).unwrap();
        assert_eq!(cli.command, Command::Fair);
        assert_eq!(cli.series, None);

        // --series applies to fair (shared with ladder), both forms.
        let cli = parse(&["fair", "--series", "ETH-15m"]).unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::M15,
            })
        );
        assert_eq!(
            parse(&["fair", "--series=BTC-5m"]).unwrap().command,
            Command::Fair
        );

        // Ladder-only and feed/latency flags don't apply to fair.
        assert!(parse(&["fair", "--depth", "8"]).is_err());
        assert!(parse(&["fair", "--recycle-after", "30"]).is_err());
        assert!(parse(&["fair", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["fair", "--source", "rtds"]).is_err());
        assert!(parse(&["fair", "--label", "x"]).is_err());
        assert!(parse(&["fair", "--series", "DOGE-5m"]).is_err());
    }

    #[test]
    fn parses_paper_sim() {
        use core_types::{Asset, WindowDuration};

        // Bare paper-sim defaults to BTC-5m at dispatch (series None here).
        let cli = parse(&["paper-sim"]).unwrap();
        assert_eq!(cli.command, Command::PaperSim);
        assert_eq!(cli.series, None);

        // --series applies to paper-sim (shared with ladder/fair), both forms.
        let cli = parse(&["paper-sim", "--series", "ETH-15m"]).unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::M15,
            })
        );
        assert_eq!(
            parse(&["paper-sim", "--series=BTC-1h"]).unwrap().command,
            Command::PaperSim
        );

        // `paper` and `paper-sim` are distinct; ladder-only/feed/latency flags
        // don't apply to paper-sim.
        assert_eq!(parse(&["paper"]).unwrap().command, Command::Paper);
        assert!(parse(&["paper-sim", "--depth", "8"]).is_err());
        assert!(parse(&["paper-sim", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["paper-sim", "--source", "rtds"]).is_err());
        assert!(parse(&["paper-sim", "--label", "x"]).is_err());
        assert!(parse(&["paper-sim", "paper"]).is_err());
        assert!(parse(&["paper-sim", "--series", "DOGE-5m"]).is_err());
    }

    #[test]
    fn parses_record_flags() {
        use core_types::{Asset, WindowDuration};

        // Bare record: all enabled series, default out-dir at dispatch.
        let cli = parse(&["record"]).unwrap();
        assert_eq!(cli.command, Command::Record);
        assert_eq!(cli.series, None);
        assert_eq!(cli.out_dir, None);

        // --series (shared with ladder/fair/paper-sim) + --out-dir, both forms.
        let cli = parse(&["record", "--series", "ETH-15m", "--out-dir", "data/cap"]).unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::M15,
            })
        );
        assert_eq!(cli.out_dir, Some(PathBuf::from("data/cap")));
        let cli = parse(&["record", "--series=BTC-1h", "--out-dir=/tmp/j"]).unwrap();
        assert_eq!(cli.out_dir, Some(PathBuf::from("/tmp/j")));

        // --out-dir is record-only; ladder/feed/latency flags don't apply.
        assert!(parse(&["record", "--depth", "8"]).is_err());
        assert!(parse(&["record", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["record", "--source", "rtds"]).is_err());
        assert!(parse(&["record", "--label", "x"]).is_err());
        assert!(parse(&["record", "--series", "DOGE-5m"]).is_err());
        assert!(parse(&["record", "paper"]).is_err());
        // --out-dir does not leak onto other subcommands.
        assert!(parse(&["ladder", "--out-dir", "x"]).is_err());
        assert!(parse(&["--out-dir", "x"]).is_err());
        assert!(parse(&["record", "--out-dir"]).is_err());
    }

    #[test]
    fn parses_dashboard_flags() {
        use core_types::{Asset, WindowDuration};

        // Bare dashboard: all enabled series at dispatch (series None here).
        let cli = parse(&["dashboard"]).unwrap();
        assert_eq!(cli.command, Command::Dashboard);
        assert_eq!(cli.series, None);
        assert!(!cli.with_model);

        // --with-model opts the model in; it is dashboard-only.
        let cli = parse(&["dashboard", "--with-model"]).unwrap();
        assert_eq!(cli.command, Command::Dashboard);
        assert!(cli.with_model);
        let cli = parse(&["dashboard", "--with-model", "--series", "BTC-5m"]).unwrap();
        assert!(cli.with_model);
        assert!(parse(&["paper-sim", "--with-model"]).is_err());
        assert!(parse(&["--with-model"]).is_err());

        // --series (shared with ladder/fair/paper-sim/record) scopes it, both forms.
        let cli = parse(&["dashboard", "--series", "ETH-15m"]).unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::M15,
            })
        );
        assert_eq!(
            parse(&["dashboard", "--series=BTC-1h"]).unwrap().command,
            Command::Dashboard
        );
        let cli = parse(&["dashboard", "--config-dir", "cfg"]).unwrap();
        assert_eq!(cli.config_dir, Some(PathBuf::from("cfg")));

        // Other subcommands' flags don't apply; can't combine with a mode.
        assert!(parse(&["dashboard", "--depth", "8"]).is_err());
        assert!(parse(&["dashboard", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["dashboard", "--source", "rtds"]).is_err());
        assert!(parse(&["dashboard", "--label", "x"]).is_err());
        assert!(parse(&["dashboard", "--out-dir", "x"]).is_err());
        assert!(parse(&["dashboard", "--series", "DOGE-5m"]).is_err());
        assert!(parse(&["dashboard", "paper"]).is_err());
    }

    #[test]
    fn parses_run_flags() {
        use core_types::{Asset, WindowDuration};

        // Bare run: all enabled series at dispatch (series None here).
        let cli = parse(&["run"]).unwrap();
        assert_eq!(cli.command, Command::Run);
        assert_eq!(cli.series, None);

        // --series (shared with ladder/fair/paper-sim/record/dashboard) scopes it.
        let cli = parse(&["run", "--series", "ETH-15m"]).unwrap();
        assert_eq!(
            cli.series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::M15,
            })
        );
        assert_eq!(
            parse(&["run", "--series=BTC-1h"]).unwrap().command,
            Command::Run
        );
        let cli = parse(&["run", "--config-dir", "cfg"]).unwrap();
        assert_eq!(cli.config_dir, Some(PathBuf::from("cfg")));

        // Other subcommands' flags don't apply; can't combine with a mode.
        assert!(parse(&["run", "--with-model"]).is_err());
        assert!(parse(&["run", "--depth", "8"]).is_err());
        assert!(parse(&["run", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["run", "--source", "rtds"]).is_err());
        assert!(parse(&["run", "--label", "x"]).is_err());
        assert!(parse(&["run", "--out-dir", "x"]).is_err());
        assert!(parse(&["run", "--series", "DOGE-5m"]).is_err());
        assert!(parse(&["run", "paper"]).is_err());
    }

    #[test]
    fn parses_replay_flags() {
        // Bare replay: default journal-dir + window at dispatch (None here).
        let cli = parse(&["replay"]).unwrap();
        assert_eq!(cli.command, Command::Replay);
        assert_eq!(cli.journal_dir, None);
        assert_eq!(cli.window, None);
        assert_eq!(cli.out, None);

        // Positional journal-dir + --window + --out (replay accepts --out).
        let cli = parse(&[
            "replay",
            "data/journal",
            "--window",
            "7d",
            "--out",
            "r.json",
        ])
        .unwrap();
        assert_eq!(cli.command, Command::Replay);
        assert_eq!(cli.journal_dir, Some(PathBuf::from("data/journal")));
        assert_eq!(cli.window.as_deref(), Some("7d"));
        assert_eq!(cli.out, Some(PathBuf::from("r.json")));

        // A leading flag means no positional dir is consumed.
        let cli = parse(&["replay", "--window=all"]).unwrap();
        assert_eq!(cli.journal_dir, None);
        assert_eq!(cli.window.as_deref(), Some("all"));

        // A non-flag token after `replay` is the positional journal-dir.
        let cli = parse(&["replay", "paper"]).unwrap();
        assert_eq!(cli.command, Command::Replay);
        assert_eq!(cli.journal_dir, Some(PathBuf::from("paper")));

        // Sweep-only grid flags don't apply to replay; series/ladder flags don't either.
        assert!(parse(&["replay", "--min-edge", "0.01"]).is_err());
        assert!(parse(&["replay", "--parallel"]).is_err());
        assert!(parse(&["replay", "--series", "BTC-5m"]).is_err());
        assert!(parse(&["replay", "--depth", "8"]).is_err());
        // --window is replay/sweep-only; --out is not record/feed/etc.
        assert!(parse(&["paper", "--window", "all"]).is_err());
        assert!(parse(&["schedule", "--out", "r.json"]).is_err());
        // A second subcommand after a mode is rejected.
        assert!(parse(&["paper", "replay"]).is_err());
    }

    #[test]
    fn parses_sweep_flags() {
        // Bare sweep.
        let cli = parse(&["sweep"]).unwrap();
        assert_eq!(cli.command, Command::Sweep);
        assert_eq!(cli.journal_dir, None);
        assert!(!cli.parallel);

        // Full grid, both flag forms, plus positional dir + rank + parallel.
        let cli = parse(&[
            "sweep",
            "data/j",
            "--min-edge",
            "0.01,0.02",
            "--gamma=0.05,0.1",
            "--cancel-theta",
            "0.005,0.01",
            "--taker-buffer=0.005",
            "--rank",
            "pnl-per-window",
            "--parallel",
            "--window",
            "today",
        ])
        .unwrap();
        assert_eq!(cli.command, Command::Sweep);
        assert_eq!(cli.journal_dir, Some(PathBuf::from("data/j")));
        assert_eq!(cli.min_edge.as_deref(), Some("0.01,0.02"));
        assert_eq!(cli.gamma.as_deref(), Some("0.05,0.1"));
        assert_eq!(cli.cancel_theta.as_deref(), Some("0.005,0.01"));
        assert_eq!(cli.taker_buffer.as_deref(), Some("0.005"));
        assert_eq!(cli.rank.as_deref(), Some("pnl-per-window"));
        assert!(cli.parallel);
        assert_eq!(cli.window.as_deref(), Some("today"));

        // Grid flags are sweep-only; can't combine with another mode.
        assert!(parse(&["replay", "--gamma", "0.1"]).is_err());
        assert!(parse(&["paper", "sweep"]).is_err());
        assert!(parse(&["--min-edge", "0.01"]).is_err());
    }

    #[test]
    fn parses_latency_flags_both_forms() {
        let cli = parse(&["latency", "--label", "eu-west-2", "--out", "r.json"]).unwrap();
        assert_eq!(cli.command, Command::Latency);
        assert_eq!(cli.label.as_deref(), Some("eu-west-2"));
        assert_eq!(cli.out, Some(PathBuf::from("r.json")));

        let cli = parse(&["latency", "--label=vps-1", "--out=data/r.json"]).unwrap();
        assert_eq!(cli.label.as_deref(), Some("vps-1"));
        assert_eq!(cli.out, Some(PathBuf::from("data/r.json")));

        // Bare latency works; flags default to None.
        let cli = parse(&["latency"]).unwrap();
        assert_eq!(cli.command, Command::Latency);
        assert_eq!(cli.label, None);
        assert_eq!(cli.out, None);
    }

    #[test]
    fn rejects_latency_flags_on_other_subcommands() {
        assert!(parse(&["paper", "--label", "x"]).is_err());
        assert!(parse(&["schedule", "--out", "r.json"]).is_err());
        // No subcommand defaults to paper â€” flags are rejected there too.
        assert!(parse(&["--label", "x"]).is_err());
        assert!(parse(&["latency", "--label"]).is_err());
    }

    #[test]
    fn rejects_schedule_combined_with_another_mode() {
        assert!(parse(&["schedule", "paper"]).is_err());
    }

    #[test]
    fn parses_venue_check() {
        let cli = parse(&["venue-check"]).unwrap();
        assert_eq!(cli.command, Command::VenueCheck);

        let cli = parse(&["venue-check", "--config-dir", "cfg"]).unwrap();
        assert_eq!(cli.config_dir, Some(PathBuf::from("cfg")));

        // venue-check takes no other flags, and cannot combine with a mode.
        assert!(parse(&["venue-check", "--source", "rtds"]).is_err());
        assert!(parse(&["venue-check", "--series", "BTC-5m"]).is_err());
        assert!(parse(&["venue-check", "--raw", "x.jsonl"]).is_err());
        assert!(parse(&["venue-check", "paper"]).is_err());
    }

    #[test]
    fn parses_digest_flags() {
        // Bare digest: date + out default to None (resolved at dispatch).
        let cli = parse(&["digest"]).unwrap();
        assert_eq!(cli.command, Command::Digest);
        assert_eq!(cli.date, None);
        assert_eq!(cli.out, None);

        // --date + --out both apply to digest, both flag forms.
        let cli = parse(&["digest", "--date", "2026-06-15", "--out", "d.md"]).unwrap();
        assert_eq!(cli.command, Command::Digest);
        assert_eq!(cli.date.as_deref(), Some("2026-06-15"));
        assert_eq!(cli.out, Some(PathBuf::from("d.md")));
        let cli = parse(&["digest", "--date=2026-06-15", "--out=d.md"]).unwrap();
        assert_eq!(cli.date.as_deref(), Some("2026-06-15"));
        assert_eq!(cli.out, Some(PathBuf::from("d.md")));

        // --date is digest-only; digest can't combine with a mode; other
        // subcommands' flags don't apply; a missing value is rejected.
        assert!(parse(&["paper", "--date", "x"]).is_err());
        assert!(parse(&["digest", "--date"]).is_err());
        assert!(parse(&["digest", "--source", "rtds"]).is_err());
        assert!(parse(&["digest", "--depth", "8"]).is_err());
        assert!(parse(&["paper", "digest"]).is_err());
        assert!(parse(&["digest", "paper"]).is_err());
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(&["fly"]).is_err());
        assert!(parse(&["paper", "live"]).is_err());
        assert!(parse(&["--config-dir"]).is_err());
        assert!(parse(&["--unknown-flag"]).is_err());
    }

    #[test]
    fn parses_control_subcommands() {
        assert_eq!(
            parse(&["control", "kill"]).unwrap().command,
            Command::Control(ControlSub::Kill)
        );
        assert_eq!(
            parse(&["control", "status"]).unwrap().command,
            Command::Control(ControlSub::Status)
        );
        assert_eq!(
            parse(&["control", "reset-daily-stop"]).unwrap().command,
            Command::Control(ControlSub::ResetDailyStop)
        );
        assert_eq!(
            parse(&["control", "set-capital", "12000"]).unwrap().command,
            Command::Control(ControlSub::SetCapital("12000".to_owned()))
        );
        assert_eq!(
            parse(&["control", "enable-series", "BTC-5m"])
                .unwrap()
                .command,
            Command::Control(ControlSub::EnableSeries("BTC-5m".to_owned()))
        );
        // set-param with the optional --series, both forms.
        assert_eq!(
            parse(&[
                "control",
                "set-param",
                "min_edge",
                "0.02",
                "--series",
                "BTC-5m"
            ])
            .unwrap()
            .command,
            Command::Control(ControlSub::SetParam {
                series: Some("BTC-5m".to_owned()),
                key: "min_edge".to_owned(),
                value: "0.02".to_owned(),
            })
        );
        assert_eq!(
            parse(&[
                "control",
                "set-param",
                "min_edge",
                "0.02",
                "--series=ETH-1h"
            ])
            .unwrap()
            .command,
            Command::Control(ControlSub::SetParam {
                series: Some("ETH-1h".to_owned()),
                key: "min_edge".to_owned(),
                value: "0.02".to_owned(),
            })
        );
        // set-param without --series is a global override.
        assert_eq!(
            parse(&["control", "set-param", "min_edge", "0.02"])
                .unwrap()
                .command,
            Command::Control(ControlSub::SetParam {
                series: None,
                key: "min_edge".to_owned(),
                value: "0.02".to_owned(),
            })
        );
        assert_eq!(
            parse(&["control", "confirm-arm", "the-phrase"])
                .unwrap()
                .command,
            Command::Control(ControlSub::ConfirmArm("the-phrase".to_owned()))
        );
        // A global flag after the control subcommand still parses.
        let cli = parse(&["control", "kill", "--config-dir", "cfg"]).unwrap();
        assert_eq!(cli.command, Command::Control(ControlSub::Kill));
        assert_eq!(cli.config_dir, Some(PathBuf::from("cfg")));
    }

    #[test]
    fn rejects_bad_control_args() {
        assert!(parse(&["control"]).is_err(), "missing subcommand");
        assert!(
            parse(&["control", "nonsense"]).is_err(),
            "unknown subcommand"
        );
        assert!(
            parse(&["control", "set-capital"]).is_err(),
            "missing positional"
        );
        assert!(
            parse(&["control", "set-capital", "--series"]).is_err(),
            "flag where a positional is required"
        );
        assert!(
            parse(&["control", "kill", "kill"]).is_err(),
            "stray trailing positional is an unrecognized argument"
        );
    }
}
