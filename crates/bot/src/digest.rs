//! The per-day operator digest: a human-readable markdown report folded from
//! the sqlite journal index plus the driver-attribution and shadow-loss-stop
//! side channels.
//!
//! It reads **only** durable artifacts (the WAL sqlite index + the per-UTC-day
//! gzip side channels), so it needs **no live process** — run it any time to
//! summarize a day. The report has five sections:
//!
//! 1. Per-driver realized PnL (maker-core / momentum / late / model × series),
//!    folded by replaying the day's `data/driver-attrib/*.gz` records into the
//!    shared [`analytics::DriverMatrix`] and settling with the day's settlements.
//! 2. Our maker fill-size quality (median / p90 notional) next to a competitor
//!    reference (markout / at-touch are dashboard-only, marked `n/a`).
//! 3. Breakers: trips by kind, cancel-alls, and a trips/hour histogram.
//! 4. Shadow loss-stops that would have tripped (paper eval).
//! 5. The day's settlements.

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::str::FromStr as _;

use analytics::DriverMatrix;
use anyhow::{Context, bail};
use core_types::{Dollars, Liquidity, Outcome, Price, Series, Size, TimestampMs, WindowId};
use engine::FillDriver;
use flate2::read::MultiGzDecoder;
use journal::JournalIndexReader;
use rust_decimal::Decimal;
use serde::Deserialize;

/// The competitor reference maker fill-size (from the `nagi777` manual).
const NAGI_REFERENCE: &str = "median $4 / p90 $21";

/// One journaled driver-attribution line (mirror of
/// [`crate::driver_attrib_record::DriverAttribRecord`]).
#[derive(Debug, Clone, Deserialize)]
struct DriverAttribLine {
    ts: i64,
    series: String,
    window_open_ms: i64,
    driver: String,
    outcome: String,
    shares: String,
    price: String,
    fee: String,
}

/// One journaled shadow-loss-stop line (mirror of
/// `crate::shadow_stops_record::ShadowStopRecord`).
#[derive(Debug, Clone, Deserialize)]
struct ShadowStopLine {
    ts: i64,
    kind: String,
    #[serde(default)]
    series: Option<String>,
    #[serde(default)]
    window_open_ms: Option<i64>,
    threshold: String,
    value: String,
}

/// Renders the markdown digest for `day` from the sqlite index + the two side
/// channels. Pure over its inputs (the fixture test drives it directly).
///
/// # Errors
/// [`anyhow::Error`] if the sqlite index cannot be opened or queried.
pub(crate) fn build_markdown(
    sqlite_path: &Path,
    driver_attrib_dir: &Path,
    shadow_stops_dir: &Path,
    day: time::Date,
) -> anyhow::Result<String> {
    let reader = JournalIndexReader::open(sqlite_path)
        .with_context(|| format!("opening the journal index at {}", sqlite_path.display()))?;
    let settlements = reader
        .settlements()
        .context("reading settlements from the journal index")?;
    let breaker_trips = reader
        .breaker_trips()
        .context("reading breaker trips from the journal index")?;
    let recent_fills = reader
        .recent_fills(100_000)
        .context("reading fills from the journal index")?;

    // Side channels (missing dirs yield no records, never an error).
    let driver_attrib: Vec<DriverAttribLine> =
        read_gz_lines::<DriverAttribLine>(driver_attrib_dir, "driver-attrib-")
            .into_iter()
            .filter(|r| day_of_ms(r.ts) == Some(day))
            .collect();
    let mut shadow_stops: Vec<ShadowStopLine> =
        read_gz_lines::<ShadowStopLine>(shadow_stops_dir, "shadow-stops-")
            .into_iter()
            .filter(|r| day_of_ms(r.ts) == Some(day))
            .collect();
    shadow_stops.sort_by_key(|s| s.ts);

    // Fold the driver matrix: record every fill first, then settle each window
    // once (so a window's buffer is complete before it is marked out).
    let mut matrix = DriverMatrix::new();
    for r in &driver_attrib {
        record_driver_fill(&mut matrix, r);
    }
    for s in &settlements {
        if day_of_ms(s.settle_ms) == Some(day) {
            matrix.settle(s.window, s.outcome);
        }
    }

    let mut md = String::new();
    writeln!(md, "# Daily digest — {}", fmt_date(day))?;
    writeln!(md)?;

    render_per_driver_pnl(&mut md, &matrix)?;
    render_maker_quality(&mut md, &recent_fills, day)?;
    render_breakers(&mut md, &breaker_trips, day)?;
    render_shadow_stops(&mut md, &shadow_stops)?;
    render_settlements(&mut md, &settlements, day)?;

    Ok(md)
}

/// Derives the paths + day from config, builds the digest, and writes it.
///
/// The sqlite index is `config.journal.sqlite_path`; the side channels live
/// beside the journal directory (`<journal.dir parent>/driver-attrib` and
/// `.../shadow-stops`, falling back to `data/`). `date` defaults to today (UTC)
/// and `out` to `data/digests/{YYYY-MM-DD}.md`.
///
/// # Errors
/// [`anyhow::Error`] if the index is missing/unreadable or the file cannot be
/// written.
pub fn execute(
    cfg: &config::AppConfig,
    date: Option<time::Date>,
    out: Option<&Path>,
) -> anyhow::Result<()> {
    let sqlite_path = &cfg.journal.sqlite_path;
    if !sqlite_path.exists() {
        bail!(
            "no journal index at {} — run `bot run` or `bot record` first (nothing to digest)",
            sqlite_path.display()
        );
    }
    let base = match cfg.journal.dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("data"),
    };
    let driver_attrib_dir = base.join("driver-attrib");
    let shadow_stops_dir = base.join("shadow-stops");
    let day = date.unwrap_or_else(|| time::OffsetDateTime::now_utc().date());
    let out_path = match out {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from("data/digests").join(format!("{}.md", fmt_date(day))),
    };

    let markdown = build_markdown(sqlite_path, &driver_attrib_dir, &shadow_stops_dir, day)?;

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating digest directory {}", parent.display()))?;
    }
    std::fs::write(&out_path, markdown)
        .with_context(|| format!("writing digest {}", out_path.display()))?;
    println!("{}", out_path.display());
    Ok(())
}

// ----------------------------------------------------------------------------
// Section renderers
// ----------------------------------------------------------------------------

/// Section 1: per-`(series, driver)` realized PnL, sorted by series then driver.
fn render_per_driver_pnl(md: &mut String, matrix: &DriverMatrix) -> anyhow::Result<()> {
    writeln!(md, "## Per-driver PnL")?;
    writeln!(md)?;
    let totals = matrix.totals();
    if totals.is_empty() {
        writeln!(md, "No resolved driver-tagged fills today.")?;
        writeln!(md)?;
        return Ok(());
    }
    let mut rows: Vec<((Series, FillDriver), _)> = totals.into_iter().collect();
    rows.sort_by_key(|((s, d), _)| (s.key(), driver_label(*d)));
    writeln!(
        md,
        "| series | driver | realized PnL | resolved fills | win% |"
    )?;
    writeln!(md, "|---|---|---:|---:|---:|")?;
    for ((series, driver), cell) in rows {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            series.key(),
            driver_label(driver),
            fmt_money(cell.realized_pnl),
            cell.resolved_fills,
            fmt_pct(cell.win_pct()),
        )?;
    }
    writeln!(md)?;
    Ok(())
}

/// Section 2: our maker fill-size quality next to the competitor reference.
fn render_maker_quality(
    md: &mut String,
    fills: &[journal::FillRow],
    day: time::Date,
) -> anyhow::Result<()> {
    writeln!(md, "## Maker fill quality vs competitors")?;
    writeln!(md)?;
    let mut notionals: Vec<Decimal> = fills
        .iter()
        .filter(|f| f.liquidity == Liquidity::Maker && day_of_ms(f.ts_venue_ms) == Some(day))
        .map(|f| f.price.as_decimal() * f.size.as_decimal())
        .collect();
    notionals.sort();
    if notionals.is_empty() {
        writeln!(md, "- Our maker fills today: none.")?;
    } else {
        writeln!(
            md,
            "- Our maker fill notional (n={}): median {} / p90 {}",
            notionals.len(),
            fmt_dec_money(percentile(&notionals, 50)),
            fmt_dec_money(percentile(&notionals, 90)),
        )?;
    }
    writeln!(md, "- Competitor reference (nagi777): {NAGI_REFERENCE}")?;
    writeln!(
        md,
        "- At-touch %: n/a (requires live dashboard — not in the sqlite index)"
    )?;
    writeln!(
        md,
        "- 5s / 30s markout: n/a (requires live dashboard — not in the sqlite index)"
    )?;
    writeln!(md)?;
    Ok(())
}

/// Section 3: breaker trips by kind, cancel-alls, and a trips/hour histogram.
fn render_breakers(
    md: &mut String,
    trips: &[journal::BreakerRow],
    day: time::Date,
) -> anyhow::Result<()> {
    writeln!(md, "## Breakers")?;
    writeln!(md)?;
    let today: Vec<&journal::BreakerRow> = trips
        .iter()
        .filter(|r| day_of_ms(r.ts_local_ms) == Some(day))
        .collect();

    // Trips grouped by breaker (kind == "tripped").
    let mut by_breaker: Vec<(String, u32)> = Vec::new();
    for r in today.iter().filter(|r| r.kind == "tripped") {
        let label = format!("{:?}", r.breaker);
        match by_breaker.iter_mut().find(|(l, _)| *l == label) {
            Some((_, n)) => *n += 1,
            None => by_breaker.push((label, 1)),
        }
    }
    by_breaker.sort();
    if by_breaker.is_empty() {
        writeln!(md, "Trips by breaker: none")?;
    } else {
        writeln!(md, "Trips by breaker:")?;
        for (label, n) in &by_breaker {
            writeln!(md, "- {label}: {n}")?;
        }
    }

    let cancel_all = today.iter().filter(|r| r.kind == "cancel_all").count();
    writeln!(md, "Cancel-all: {cancel_all}")?;

    // Trips/hour histogram (UTC), tripped only.
    let mut buckets = [0u32; 24];
    let mut total = 0u32;
    for r in today.iter().filter(|r| r.kind == "tripped") {
        if let Some(h) = hour_of_ms(r.ts_local_ms) {
            buckets[usize::from(h)] += 1;
            total += 1;
        }
    }
    let hours: Vec<String> = buckets
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(h, &n)| format!("{h:02}: {n}"))
        .collect();
    if hours.is_empty() {
        writeln!(md, "Trips/hour (UTC): none")?;
    } else {
        writeln!(
            md,
            "Trips/hour (UTC): {}  (total {total})",
            hours.join(", ")
        )?;
    }
    writeln!(md)?;
    Ok(())
}

/// Section 4: would-be shadow loss-stops for the day, one line each.
fn render_shadow_stops(md: &mut String, stops: &[ShadowStopLine]) -> anyhow::Result<()> {
    writeln!(md, "## Shadow loss-stops (would-have-tripped)")?;
    writeln!(md)?;
    if stops.is_empty() {
        writeln!(md, "- none")?;
    } else {
        for s in stops {
            let scope = match (&s.series, s.window_open_ms) {
                (Some(series), Some(open_ms)) => format!("{series}@{}", iso_time(open_ms)),
                (Some(series), None) => series.clone(),
                _ => "global".to_owned(),
            };
            writeln!(
                md,
                "- {} {} {} threshold={} value={}",
                iso_time(s.ts),
                s.kind,
                scope,
                s.threshold,
                s.value,
            )?;
        }
    }
    writeln!(md)?;
    Ok(())
}

/// Section 5: the day's settlements (oldest first, newest last).
fn render_settlements(
    md: &mut String,
    settlements: &[journal::SettlementRow],
    day: time::Date,
) -> anyhow::Result<()> {
    writeln!(md, "## Settlements")?;
    writeln!(md)?;
    let mut today: Vec<&journal::SettlementRow> = settlements
        .iter()
        .filter(|s| day_of_ms(s.settle_ms) == Some(day))
        .collect();
    today.sort_by_key(|s| s.settle_ms);
    if today.is_empty() {
        writeln!(md, "- none")?;
    } else {
        for s in today {
            writeln!(
                md,
                "- {} @ {} → {} {}",
                s.window.series.key(),
                iso_time(s.window.open_time.as_millis()),
                s.outcome,
                fmt_money(s.realized_pnl),
            )?;
        }
    }
    writeln!(md)?;
    Ok(())
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Folds one journaled driver-attribution line into the matrix, skipping any
/// record whose typed fields do not parse (defensive; never panics).
fn record_driver_fill(matrix: &mut DriverMatrix, r: &DriverAttribLine) {
    let (Ok(series), Some(driver), Some(outcome)) = (
        Series::from_str(&r.series),
        parse_driver(&r.driver),
        parse_outcome(&r.outcome),
    ) else {
        return;
    };
    let (Ok(shares_dec), Ok(price_dec), Ok(fee_dec)) = (
        Decimal::from_str(&r.shares),
        Decimal::from_str(&r.price),
        Decimal::from_str(&r.fee),
    ) else {
        return;
    };
    let (Ok(size), Ok(price)) = (Size::new(shares_dec), Price::try_from(price_dec)) else {
        return;
    };
    matrix.record_fill(
        WindowId {
            series,
            open_time: TimestampMs::from_millis(r.window_open_ms),
        },
        driver,
        outcome,
        size,
        price,
        Dollars::new(fee_dec),
        TimestampMs::from_millis(r.ts),
    );
}

/// Reads every `{prefix}*.jsonl.gz` in `dir`, deserializing each line into `T`.
/// A missing directory yields no records; unparseable lines and a
/// crash-truncated trailing gzip member are skipped (decoded prefix kept).
fn read_gz_lines<T: serde::de::DeserializeOwned>(dir: &Path, prefix: &str) -> Vec<T> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".jsonl.gz"))
        })
        .collect();
    paths.sort();
    for path in paths {
        let Ok(file) = std::fs::File::open(&path) else {
            continue;
        };
        let mut text = String::new();
        // A partial trailing member returns an error but leaves the decoded
        // prefix in `text`; parse whatever complete lines survived.
        let _ = MultiGzDecoder::new(file).read_to_string(&mut text);
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<T>(line) {
                out.push(v);
            }
        }
    }
    out
}

/// The stable disk label for a [`FillDriver`] (mirrors the writer).
fn driver_label(driver: FillDriver) -> &'static str {
    match driver {
        FillDriver::MakerCore => "maker-core",
        FillDriver::Momentum => "momentum",
        FillDriver::Late => "late",
        FillDriver::Model => "model",
    }
}

/// Parses a driver disk label back into a [`FillDriver`].
fn parse_driver(label: &str) -> Option<FillDriver> {
    Some(match label {
        "maker-core" => FillDriver::MakerCore,
        "momentum" => FillDriver::Momentum,
        "late" => FillDriver::Late,
        "model" => FillDriver::Model,
        _ => return None,
    })
}

/// Parses an outcome label (`"Up"`/`"Down"`, as [`Outcome`]'s `Display` writes).
fn parse_outcome(label: &str) -> Option<Outcome> {
    match label {
        "Up" => Some(Outcome::Up),
        "Down" => Some(Outcome::Down),
        _ => None,
    }
}

/// Nearest-rank percentile of a sorted slice (`p` in `0..=100`).
fn percentile(sorted: &[Decimal], p: usize) -> Decimal {
    if sorted.is_empty() {
        return Decimal::ZERO;
    }
    let n = sorted.len();
    // ceil(p/100 * n), 1-based rank.
    let rank = (p * n).div_ceil(100).max(1);
    let idx = (rank - 1).min(n - 1);
    sorted[idx]
}

/// Formats a signed dollar amount, e.g. `$1.90` or `$-1.52`.
fn fmt_money(d: Dollars) -> String {
    format!("${}", d.as_decimal())
}

/// Formats a raw decimal as a dollar amount.
fn fmt_dec_money(d: Decimal) -> String {
    format!("${d}")
}

/// Formats an optional win fraction as a percentage, e.g. `100.0%` or `—`.
fn fmt_pct(p: Option<f64>) -> String {
    match p {
        Some(f) => format!("{:.1}%", f * 100.0),
        None => "—".to_owned(),
    }
}

/// The UTC calendar date of a unix-ms timestamp.
fn day_of_ms(ms: i64) -> Option<time::Date> {
    time::OffsetDateTime::from_unix_timestamp(ms.div_euclid(1_000))
        .ok()
        .map(time::OffsetDateTime::date)
}

/// The UTC hour (0..=23) of a unix-ms timestamp.
fn hour_of_ms(ms: i64) -> Option<u8> {
    time::OffsetDateTime::from_unix_timestamp(ms.div_euclid(1_000))
        .ok()
        .map(|d| d.hour())
}

/// `YYYY-MM-DD` for a date (dependency-free, no formatting feature needed).
fn fmt_date(day: time::Date) -> String {
    format!(
        "{:04}-{:02}-{:02}",
        day.year(),
        u8::from(day.month()),
        day.day()
    )
}

/// `YYYY-MM-DDThh:mm:ssZ` for a unix-ms timestamp (dependency-free).
fn iso_time(ms: i64) -> String {
    match time::OffsetDateTime::from_unix_timestamp(ms.div_euclid(1_000)) {
        Ok(dt) => format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            dt.year(),
            u8::from(dt.month()),
            dt.day(),
            dt.hour(),
            dt.minute(),
            dt.second(),
        ),
        Err(_) => format!("ms:{ms}"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write as _;
    use std::sync::Arc;

    use super::*;
    // `Dollars`, `Liquidity`, `Outcome`, `Price`, `Series`, `Size`, `TimestampMs`
    // and `WindowId` come in via `super::*`.
    use core_types::{
        Asset, BreakerKind, ConditionId, Event, FeeParams, Fill, MarketInfo, OrderId,
        ResolutionSource, RiskEvent, SettlementSummary, Side, SideInventory, TickSize, TokenId,
        TokenPair, WindowDuration, WindowLifecycle,
    };
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use journal::{JournalIndex, JournalRecord};
    use rust_decimal::dec;

    // 2023-11-14T22:13:20Z.
    const DAY_MS: i64 = 1_700_000_000_000;

    fn series() -> Series {
        Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        }
    }

    fn window() -> WindowId {
        WindowId {
            series: series(),
            open_time: TimestampMs::from_millis(DAY_MS),
        }
    }

    fn px(d: rust_decimal::Decimal) -> Price {
        Price::on_grid(d, TickSize::T001).unwrap()
    }

    fn sz(d: rust_decimal::Decimal) -> Size {
        Size::new(d).unwrap()
    }

    fn market() -> MarketInfo {
        MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: TokenId::new("111").unwrap(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: TimestampMs::from_millis(DAY_MS + 300_000),
            strike: Some(dec!(60000.5)),
            tick_size: TickSize::T001,
            min_order_size: sz(dec!(5)),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        }
    }

    fn maker_fill() -> Event {
        Event::Fill(Arc::new(Fill {
            order_id: OrderId::new("qm-1").unwrap(),
            trade_id: Some("t-1".to_owned()),
            window: window(),
            token_id: TokenId::new("111").unwrap(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: px(dec!(0.40)),
            size: sz(dec!(30)),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: TimestampMs::from_millis(DAY_MS + 50),
            ts_local: TimestampMs::from_millis(DAY_MS + 50),
        }))
    }

    fn rec(event: &Event) -> JournalRecord {
        JournalRecord::from_event(event)
    }

    fn write_gz(path: &Path, lines: &[String]) {
        let file = std::fs::File::create(path).unwrap();
        let mut enc = GzEncoder::new(file, Compression::default());
        for line in lines {
            enc.write_all(line.as_bytes()).unwrap();
            enc.write_all(b"\n").unwrap();
        }
        enc.finish().unwrap();
    }

    #[test]
    fn digest_folds_index_and_side_channels() {
        let root = std::env::temp_dir().join(format!("digest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let sqlite = root.join("index.sqlite");
        let da_dir = root.join("driver-attrib");
        let ss_dir = root.join("shadow-stops");
        std::fs::create_dir_all(&da_dir).unwrap();
        std::fs::create_dir_all(&ss_dir).unwrap();

        // Build a tiny sqlite index: window open, a maker fill, resolve, settle,
        // and two breaker trips at 22:13 and 23:13 UTC.
        let settlement = SettlementSummary::close(
            window(),
            Outcome::Up,
            SideInventory {
                shares: sz(dec!(30)),
                cost: Dollars::new(dec!(12)),
            },
            SideInventory::default(),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(3.75)),
            TimestampMs::from_millis(DAY_MS + 200_000),
        );
        let events: Vec<(i64, Event)> = vec![
            (
                DAY_MS,
                Event::Window {
                    market: Arc::new(market()),
                    lifecycle: WindowLifecycle::Open,
                },
            ),
            (DAY_MS + 50, maker_fill()),
            (
                DAY_MS + 100,
                Event::Risk(RiskEvent::BreakerTripped {
                    breaker: BreakerKind::FeedStale,
                }),
            ),
            (
                DAY_MS + 3_600_000,
                Event::Risk(RiskEvent::BreakerTripped {
                    breaker: BreakerKind::FairVsMid,
                }),
            ),
            (
                DAY_MS + 200_000,
                Event::Window {
                    market: Arc::new(market()),
                    lifecycle: WindowLifecycle::Resolved {
                        outcome: Outcome::Up,
                    },
                },
            ),
            (DAY_MS + 200_000, Event::Settlement(Arc::new(settlement))),
        ];
        let mut idx = JournalIndex::open(&sqlite).unwrap();
        for (i, (ts, ev)) in events.iter().enumerate() {
            idx.index(i as u64 + 1, *ts, &rec(ev)).unwrap();
        }
        idx.commit().unwrap();
        idx.close().unwrap();

        // Driver-attrib side channel: model buys Up (wins), momentum buys Down
        // (loses), on the window that resolves Up.
        write_gz(
            &da_dir.join("driver-attrib-20231114.jsonl.gz"),
            &[
                format!(
                    r#"{{"ts":{DAY_MS},"series":"BTC-5m","window_open_ms":{DAY_MS},"driver":"model","outcome":"Up","shares":"10","price":"0.8","fee":"0.10"}}"#
                ),
                format!(
                    r#"{{"ts":{DAY_MS},"series":"BTC-5m","window_open_ms":{DAY_MS},"driver":"momentum","outcome":"Down","shares":"5","price":"0.3","fee":"0.02"}}"#
                ),
            ],
        );

        // Shadow-stops side channel: one DailyStop that would have tripped.
        write_gz(
            &ss_dir.join("shadow-stops-20231114.jsonl.gz"),
            &[format!(
                r#"{{"ts":{DAY_MS},"recorded_ms":{DAY_MS},"kind":"DailyStop","series":null,"window_open_ms":null,"threshold":"200","value":"-210"}}"#
            )],
        );

        let day = time::Date::from_calendar_date(2023, time::Month::November, 14).unwrap();
        let md = build_markdown(&sqlite, &da_dir, &ss_dir, day).unwrap();

        // (a) Per-driver PnL: model +1.90 (100% win), momentum -1.52 (0% win).
        assert!(md.contains("model"), "model driver row:\n{md}");
        assert!(md.contains("momentum"), "momentum driver row:\n{md}");
        // Dollars are stored normalized, so +1.90 renders as `$1.9`.
        assert!(md.contains("$1.9"), "model realized +1.90:\n{md}");
        assert!(md.contains("$-1.52"), "momentum realized -1.52:\n{md}");
        assert!(md.contains("100.0%"), "model win%:\n{md}");
        assert!(md.contains("0.0%"), "momentum win%:\n{md}");

        // (a2) Maker quality section reports our notional + the nagi reference.
        assert!(md.contains("nagi777"), "competitor reference:\n{md}");
        assert!(md.contains("$12"), "maker notional 30 * 0.40 = $12:\n{md}");

        // (b) Trips/hour reflects the two inserted trips (22:xx and 23:xx UTC).
        assert!(md.contains("22: 1"), "hour 22 trip:\n{md}");
        assert!(md.contains("23: 1"), "hour 23 trip:\n{md}");
        assert!(md.contains("FeedStale"), "trip by breaker:\n{md}");

        // (c) The shadow DailyStop line appears.
        assert!(md.contains("DailyStop"), "shadow loss-stop line:\n{md}");

        // Section 5 shows the settlement's realized PnL.
        assert!(md.contains("$3.75"), "settlement realized pnl:\n{md}");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_side_channel_dirs_are_not_an_error() {
        let root = std::env::temp_dir().join(format!("digest-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let sqlite = root.join("index.sqlite");
        // A valid but empty index.
        let idx = JournalIndex::open(&sqlite).unwrap();
        idx.close().unwrap();

        let day = time::Date::from_calendar_date(2023, time::Month::November, 14).unwrap();
        let md = build_markdown(
            &sqlite,
            &root.join("driver-attrib"),
            &root.join("shadow-stops"),
            day,
        )
        .unwrap();
        assert!(md.contains("No resolved driver-tagged fills today."));
        assert!(md.contains("- none"), "empty shadow/settlements:\n{md}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
