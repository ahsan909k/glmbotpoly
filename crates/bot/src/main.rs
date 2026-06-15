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
mod discover;
mod fair;
mod feed;
mod ladder;
mod latency;
mod live;
mod paper;
mod record;
mod schedule;
mod telemetry;
mod timecfg;
mod vol;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, bail};
use config::{LiveArming, Secrets};
use core_types::Series;

use crate::feed::FeedSource;

const USAGE: &str = "usage: bot [paper|paper-sim|live|venue-check|check-config|discover|schedule|feed|\
                     compare|vol|ladder|fair|record|latency] [--config-dir <path>] [--label <name>] \
                     [--out <file>] [--raw <file>] [--source rtds|binance] [--series <KEY>] \
                     [--depth <N>] [--recycle-after <secs>] [--out-dir <dir>]";

/// What the operator asked the binary to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
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
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "paper" | "paper-sim" | "live" | "venue-check" | "check-config" | "discover"
            | "schedule" | "feed" | "compare" | "vol" | "ladder" | "fair" | "record"
            | "latency" => {
                if command.is_some() {
                    return Err(format!("more than one subcommand given (second: {arg:?})"));
                }
                command = Some(match arg.as_str() {
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
                    _ => Command::CheckConfig,
                });
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
                } else {
                    return Err(format!("unrecognized argument: {other:?}"));
                }
            }
        }
    }
    let command = command.unwrap_or(Command::Paper);
    if (label.is_some() || out.is_some()) && command != Command::Latency {
        return Err("--label/--out only apply to the latency subcommand".to_owned());
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
    {
        return Err(
            "--series only applies to the ladder, fair, paper-sim, and record subcommands"
                .to_owned(),
        );
    }
    if (depth.is_some() || recycle_after.is_some()) && command != Command::Ladder {
        return Err("--depth/--recycle-after only apply to the ladder subcommand".to_owned());
    }
    if out_dir.is_some() && command != Command::Record {
        return Err("--out-dir only applies to the record subcommand".to_owned());
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
    }
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
    fn rejects_garbage() {
        assert!(parse(&["fly"]).is_err());
        assert!(parse(&["paper", "live"]).is_err());
        assert!(parse(&["--config-dir"]).is_err());
        assert!(parse(&["--unknown-flag"]).is_err());
    }
}
