//! Renderer for the `bot discover` subcommand: a per-series block table of
//! the live discovery snapshot, with full ids so the operator can verify
//! every value against the website / raw API.

use core_types::{DurationMs, MarketInfo, ResolutionKind, TimestampMs};
use discovery::{DiscoverySnapshot, SeriesWindows};

/// Prints the whole snapshot, one block per series in [`core_types::Series::ALL`]
/// order. Failed series render an ERROR line instead of a block.
pub fn print_snapshot(snapshot: &DiscoverySnapshot, now: TimestampMs) {
    println!(
        "discovery snapshot at {} (now = {} ms)",
        fmt_ts(now),
        now.as_millis()
    );
    for (series, result) in &snapshot.per_series {
        println!();
        match result {
            Ok(windows) => print_series(windows, now),
            Err(err) => println!("{:<8} ERROR: {err}", series.key()),
        }
    }
    println!();
    if snapshot.all_ok() {
        println!("all 6 series discovered OK");
    } else {
        let failed = snapshot
            .per_series
            .iter()
            .filter(|(_, r)| r.is_err())
            .count();
        println!("{failed} of 6 series FAILED — see ERROR lines above");
    }
}

fn print_series(windows: &SeriesWindows, now: TimestampMs) {
    println!(
        "{:<8} ({} upcoming)",
        windows.series.key(),
        windows.upcoming.len()
    );
    match &windows.current {
        Some(market) => {
            println!(
                "  current {}   {} -> {}   closes in {}",
                market.event_slug,
                fmt_ts(market.window.open_time),
                fmt_ts(market.close_time),
                fmt_countdown(market.close_time.signed_duration_since(now)),
            );
            print_market_detail(market);
        }
        None => println!("  current (none — between windows right now)"),
    }
    for (i, market) in windows.upcoming.iter().enumerate() {
        println!(
            "  next{:<4}{}   opens {} (in {})",
            if i == 0 {
                String::new()
            } else {
                format!("+{i}")
            },
            market.event_slug,
            fmt_ts(market.window.open_time),
            fmt_countdown(market.window.open_time.signed_duration_since(now)),
        );
    }
}

fn print_market_detail(market: &MarketInfo) {
    println!("    cid   {}", market.condition_id);
    println!("    up    {}", market.tokens.up);
    println!("    down  {}", market.tokens.down);
    println!(
        "    tick {}   min {} sh   fee {}{} (rebate {}, {})   neg_risk {}   resolution {}",
        market.tick_size.as_decimal(),
        market.min_order_size.as_decimal(),
        market.fees.rate,
        if market.fees.taker_only {
            " taker-only"
        } else {
            ""
        },
        market.fees.rebate_rate,
        if market.fees.enabled {
            "enabled"
        } else {
            "DISABLED"
        },
        market.neg_risk,
        fmt_resolution(market),
    );
}

fn fmt_resolution(market: &MarketInfo) -> String {
    let kind = match market.resolution.kind {
        ResolutionKind::ChainlinkDataStream => "chainlink-data-stream",
        ResolutionKind::BinanceCandle => "binance-candle",
        ResolutionKind::Other => "OTHER/UNKNOWN",
    };
    format!("{kind} [{}]", market.resolution.raw)
}

/// RFC3339 UTC at second resolution; falls back to raw millis only if the
/// timestamp is outside the formattable range.
pub(crate) fn fmt_ts(ts: TimestampMs) -> String {
    discovery::map::format_rfc3339_secs(ts).unwrap_or_else(|_| format!("{}ms", ts.as_millis()))
}

/// `mm:ss` (or `h:mm:ss` from one hour up); negative spans render with a
/// leading `-`.
pub(crate) fn fmt_countdown(d: DurationMs) -> String {
    let total_secs = d.as_millis() / 1000;
    let sign = if total_secs < 0 { "-" } else { "" };
    let secs = total_secs.abs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{sign}{h}:{m:02}:{s:02}")
    } else {
        format!("{sign}{m:02}:{s:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn countdown_formats() {
        assert_eq!(fmt_countdown(DurationMs::from_millis(0)), "00:00");
        assert_eq!(fmt_countdown(DurationMs::from_secs(133)), "02:13");
        assert_eq!(fmt_countdown(DurationMs::from_secs(3_725)), "1:02:05");
        assert_eq!(fmt_countdown(DurationMs::from_secs(-95)), "-01:35");
    }

    #[test]
    fn timestamp_formats_as_rfc3339() {
        assert_eq!(
            fmt_ts(TimestampMs::from_millis(1_781_166_600_000)),
            "2026-06-11T08:30:00Z"
        );
    }
}
