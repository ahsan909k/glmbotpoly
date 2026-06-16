//! `bot replay` / `bot sweep` — deterministic backtest + parameter sweep over a
//! recorded journal (feature `replay`, CLAUDE.md §3/§10).
//!
//! `replay` re-runs the full engine + a seed-stable paper venue over a recorded
//! tape and prints the §10 series-comparison table; `sweep` runs a grid over four
//! quoting parameters (minimum edge, skew strength, cancel threshold, taker
//! buffer) and prints a ranked comparison. The harness lives in
//! [`analytics::replay`]; this module maps `config → params` at the bot boundary
//! (the `vol_params`/`paper_params` precedent — the `analytics` crate stays
//! `config`-free) and renders the results.

use std::path::Path;

use analytics::{
    AnalyticsParams, ComparisonWindow, ReplayConfig, ReplayOutput, SeriesComparison, SortColumn,
    SweepGrid, SweepReport, run_replay, run_sweep,
};
use anyhow::Context;
use config::AppConfig;
use core_types::{Decimal, DurationMs, Mode};
use journal::{RecordEnvelope, ReplayReader};
use venue_paper::PaperParams;

/// Fixed paper-venue RNG seed for a deterministic replay (overrides the
/// wall-clock seed `paper::paper_params` uses for live smoke runs).
const REPLAY_SEED: u64 = 1;

/// Flags for `bot sweep`, threaded from the CLI parser.
pub struct SweepArgs<'a> {
    /// Directory of recorded `journal-*.jsonl.gz` segments.
    pub journal_dir: &'a Path,
    /// Comparison window selection (`today` | `<N>d` | `all`; default `all`).
    pub window: Option<&'a str>,
    /// Comma-separated `min_edge` values (omitted → the config base value).
    pub min_edge: Option<&'a str>,
    /// Comma-separated skew-strength (`gamma`) values.
    pub gamma: Option<&'a str>,
    /// Comma-separated cancel-threshold (`reprice_theta`) values.
    pub cancel_theta: Option<&'a str>,
    /// Comma-separated taker-buffer values.
    pub taker_buffer: Option<&'a str>,
    /// Ranking column key (default `net-pnl`).
    pub rank: Option<&'a str>,
    /// Fan grid points across worker threads.
    pub parallel: bool,
    /// Optional JSON output path for the full report.
    pub out: Option<&'a Path>,
}

/// Parses the `--window` selection (`today` | `<N>d` | `all`; default `all`).
///
/// # Errors
/// Returns a message for an unrecognized selection.
pub fn parse_window(arg: Option<&str>) -> Result<ComparisonWindow, String> {
    match arg {
        None | Some("all") => Ok(ComparisonWindow::All),
        Some("today") => Ok(ComparisonWindow::Today),
        Some(s) => s
            .strip_suffix('d')
            .and_then(|n| n.parse::<u16>().ok())
            .map(ComparisonWindow::LastDays)
            .ok_or_else(|| format!("invalid --window {s:?} (expected today|<N>d|all)")),
    }
}

/// Re-runs the engine over the recorded tape and prints the comparison.
///
/// # Errors
/// If the journal cannot be read or the replay fails.
pub fn execute_replay(
    config: &AppConfig,
    journal_dir: &Path,
    window: ComparisonWindow,
    out: Option<&Path>,
) -> anyhow::Result<()> {
    let events = read_tape(journal_dir)?;
    let cfg = replay_config(config, window);
    let output = run_replay(&events, &cfg).map_err(|e| anyhow::anyhow!("replay failed: {e}"))?;
    print_replay(&output);
    if let Some(path) = out {
        let json = serde_json::to_string_pretty(&output).context("serializing replay output")?;
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

/// Runs the parameter sweep over the recorded tape and prints the ranked report.
///
/// # Errors
/// If the journal cannot be read, a flag is malformed, or the sweep fails.
pub fn execute_sweep(config: &AppConfig, args: &SweepArgs) -> anyhow::Result<()> {
    let events = read_tape(args.journal_dir)?;
    let window = parse_window(args.window).map_err(|m| anyhow::anyhow!(m))?;
    let base = replay_config(config, window);
    let grid = build_grid(args).map_err(|m| anyhow::anyhow!(m))?;
    let rank = parse_rank(args.rank).map_err(|m| anyhow::anyhow!(m))?;
    let report = run_sweep(&events, &base, &grid, rank, args.parallel)
        .map_err(|e| anyhow::anyhow!("sweep failed: {e}"))?;
    print_sweep(&report);
    if let Some(path) = args.out {
        let json = serde_json::to_string_pretty(&report).context("serializing sweep report")?;
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

/// Reads every journal segment in `dir` into an ordered tape.
fn read_tape(dir: &Path) -> anyhow::Result<Vec<RecordEnvelope>> {
    let reader =
        ReplayReader::open(dir).with_context(|| format!("opening journal at {}", dir.display()))?;
    let mut tape = Vec::new();
    for item in reader {
        tape.push(
            item.with_context(|| format!("reading a journal record from {}", dir.display()))?,
        );
    }
    if tape.is_empty() {
        anyhow::bail!(
            "no journal records found in {} (expected journal-*.jsonl.gz segments)",
            dir.display()
        );
    }
    Ok(tape)
}

/// The base replay config: the full risk/strategy params + per-series caps mapped
/// from `config` (reusing the live `bot run` boundary maps), a seed-stable paper
/// venue (fixed seed, zero jitter — deterministic latencies), and the default
/// analytics params (the `config → AnalyticsParams` map is deferred, per the
/// crate docs).
fn replay_config(config: &AppConfig, window: ComparisonWindow) -> ReplayConfig {
    let enabled = config.engine.enabled_series();
    ReplayConfig {
        risk: crate::run::risk_params(config),
        series_caps: crate::run::series_caps(config, &enabled),
        paper: replay_paper_params(config),
        analytics: AnalyticsParams::default(),
        mode: Mode::Paper,
        risk_tick_ms: config.run.risk_tick_ms.as_millis(),
        comparison_window: window,
    }
}

/// `paper::paper_params` with the seed/jitter forced to deterministic values.
fn replay_paper_params(config: &AppConfig) -> PaperParams {
    let mut p = crate::paper::paper_params(config);
    p.rng_seed = Some(REPLAY_SEED);
    p.placement.jitter_ms = DurationMs::ZERO;
    p.cancel.jitter_ms = DurationMs::ZERO;
    p
}

/// Builds the sweep grid from the CLI flags (an omitted dimension yields an empty
/// vec, which the harness fills with the config base value).
fn build_grid(args: &SweepArgs) -> Result<SweepGrid, String> {
    Ok(SweepGrid {
        min_edge: args
            .min_edge
            .map(parse_decimal_list)
            .transpose()?
            .unwrap_or_default(),
        gamma: args
            .gamma
            .map(parse_f64_list)
            .transpose()?
            .unwrap_or_default(),
        cancel_theta: args
            .cancel_theta
            .map(parse_f64_list)
            .transpose()?
            .unwrap_or_default(),
        taker_buffer: args
            .taker_buffer
            .map(parse_decimal_list)
            .transpose()?
            .unwrap_or_default(),
    })
}

/// Parses a comma-separated list of decimals (blanks skipped).
fn parse_decimal_list(s: &str) -> Result<Vec<Decimal>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<Decimal>()
                .map_err(|_| format!("invalid decimal {t:?}"))
        })
        .collect()
}

/// Parses a comma-separated list of floats (blanks skipped).
fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .map_err(|_| format!("invalid number {t:?}"))
        })
        .collect()
}

/// Maps a `--rank` key to a [`SortColumn`] (default `net-pnl`).
fn parse_rank(arg: Option<&str>) -> Result<SortColumn, String> {
    Ok(match arg.unwrap_or("net-pnl") {
        "net-pnl" => SortColumn::NetPnl,
        "pnl-per-window" => SortColumn::PnlPerWindow,
        "fraction-profitable" => SortColumn::FractionProfitable,
        "windows" => SortColumn::WindowsTraded,
        "avg-markout" | "markout" => SortColumn::AvgMarkout5s,
        "fees" => SortColumn::FeesPaid,
        "rebates" => SortColumn::RebatesEarned,
        "locked-pair" => SortColumn::LockedPairPnl,
        "inventory" => SortColumn::InventoryPnl,
        "taker-budget" => SortColumn::TakerBudgetUsed,
        "maker-fills" => SortColumn::MakerFills,
        "taker-fills" => SortColumn::TakerFills,
        "health" => SortColumn::Health,
        other => {
            return Err(format!(
                "unknown --rank {other:?} (valid: net-pnl, pnl-per-window, fraction-profitable, \
                 windows, avg-markout, fees, rebates, locked-pair, inventory, taker-budget, \
                 maker-fills, taker-fills, health)"
            ));
        }
    })
}

/// Prints a replay's summary + series-comparison table.
fn print_replay(out: &ReplayOutput) {
    let s = &out.summary;
    println!(
        "replay: {} events ({} dropped) → {} fills, {} order updates, {} windows settled; net PnL {}",
        s.events_total, s.events_dropped, s.fills, s.order_updates, s.windows_settled, s.net_pnl
    );
    print_comparison(&out.comparison);
}

/// Prints a §10 series-comparison table.
fn print_comparison(cmp: &SeriesComparison) {
    println!(
        "\nseries comparison ({:?}, best by {}):",
        cmp.window,
        cmp.default_sort.label()
    );
    println!(
        "  {:<10} {:>5} {:>11} {:>10} {:>7} {:>10} {:>10} {:>5} {:>5} {:>10}",
        "series",
        "wins",
        "net PnL",
        "PnL/win",
        "%prof",
        "fees",
        "rebates",
        "mkr",
        "tkr",
        "5s mkout"
    );
    if cmp.rows.is_empty() {
        println!("  (no series traded in the recording)");
        return;
    }
    for r in &cmp.rows {
        println!(
            "  {:<10} {:>5} {:>11} {:>10} {:>6.1}% {:>10} {:>10} {:>5} {:>5} {:>10}",
            r.series.key(),
            r.windows_traded,
            r.net_pnl,
            r.pnl_per_window,
            r.fraction_profitable * 100.0,
            r.fees_paid,
            r.rebates_earned,
            r.maker_fills,
            r.taker_fills,
            r.avg_markout_5s
                .map_or_else(|| "-".to_owned(), |m| format!("{m:.5}")),
        );
    }
}

/// Prints a ranked sweep report.
fn print_sweep(report: &SweepReport) {
    println!(
        "sweep: {} parameter sets, ranked best-first by {}:",
        report.points_total,
        report.rank_metric.label()
    );
    println!(
        "  {:>3} {:>9} {:>7} {:>8} {:>9} {:>11} {:>10} {:>5} {:>7} {:>10}",
        "#",
        "min_edge",
        "gamma",
        "cncl_th",
        "tkr_buf",
        "net PnL",
        "PnL/win",
        "wins",
        "%prof",
        "5s mkout"
    );
    for (i, row) in report.rows.iter().enumerate() {
        let p = &row.point;
        let a = &row.aggregate;
        println!(
            "  {:>3} {:>9} {:>7.3} {:>8.4} {:>9} {:>11} {:>10} {:>5} {:>6.1}% {:>10}",
            i + 1,
            p.min_edge,
            p.gamma,
            p.cancel_theta,
            p.taker_buffer,
            a.net_pnl,
            a.pnl_per_window,
            a.windows_traded,
            a.fraction_profitable * 100.0,
            a.avg_markout_5s
                .map_or_else(|| "-".to_owned(), |m| format!("{m:.5}")),
        );
    }
}
