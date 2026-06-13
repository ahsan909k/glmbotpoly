//! Aggregated clock-offset measurement: several SNTP queries across several
//! servers collapsed into one median offset per round.
//!
//! The trait seam ([`OffsetSource`]) is what the skew-monitor driver consumes
//! — tests implement it over scripted measurements, exactly like discovery's
//! `DiscoveryApi` fixture pattern.

use std::future::Future;
use std::time::Duration;

use core_types::DurationMs;

use crate::clock::Clock;
use crate::sntp;

/// One measurement round, aggregated across all configured queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetMeasurement {
    /// Median offset (`true_time − local_wall`; positive = local clock
    /// behind) over all successful samples. `None` means every query failed
    /// — which says nothing about local clock health, only about the network.
    pub offset: Option<DurationMs>,
    /// Successful samples in the median.
    pub samples_used: u32,
    /// Failed queries this round.
    pub queries_failed: u32,
    /// Smallest observed network round-trip — a quality indicator (a median
    /// built from low-RTT samples is trustworthy at a tighter bound).
    pub min_round_trip: Option<DurationMs>,
}

/// Source of clock-offset measurements — the skew monitor's seam.
pub trait OffsetSource: Send {
    /// Runs one full measurement round. Never fails: failures are folded
    /// into the returned [`OffsetMeasurement`].
    fn measure(&mut self) -> impl Future<Output = OffsetMeasurement> + Send;
}

/// Parameters for [`NtpOffsetSource`].
#[derive(Debug, Clone)]
pub struct NtpParams {
    /// SNTP servers (`"host"` or `"host:port"`).
    pub servers: Vec<String>,
    /// Queries per server per round.
    pub samples_per_server: u32,
    /// Per-query response deadline.
    pub query_timeout: Duration,
    /// Pause between consecutive queries (rate hygiene toward public pools).
    pub query_spacing: Duration,
}

/// Production [`OffsetSource`]: sequential SNTP queries, median aggregation.
#[derive(Debug, Clone)]
pub struct NtpOffsetSource<C> {
    params: NtpParams,
    clock: C,
}

impl<C: Clock> NtpOffsetSource<C> {
    /// Builds a source over the given clock.
    pub fn new(params: NtpParams, clock: C) -> Self {
        Self { params, clock }
    }
}

impl<C: Clock + 'static> OffsetSource for NtpOffsetSource<C> {
    async fn measure(&mut self) -> OffsetMeasurement {
        let mut offsets: Vec<i64> = Vec::new();
        let mut failed: u32 = 0;
        let mut min_rtt: Option<i64> = None;
        let mut first = true;
        for server in &self.params.servers {
            for _ in 0..self.params.samples_per_server {
                if !first {
                    tokio::time::sleep(self.params.query_spacing).await;
                }
                first = false;
                match sntp::query(server, self.params.query_timeout, &self.clock).await {
                    Ok(sample) => {
                        offsets.push(sample.offset.as_millis());
                        let rtt = sample.round_trip.as_millis();
                        min_rtt = Some(min_rtt.map_or(rtt, |m| m.min(rtt)));
                        tracing::debug!(
                            target: "timeutil::offset",
                            server,
                            offset_ms = sample.offset.as_millis(),
                            rtt_ms = rtt,
                            "sntp sample"
                        );
                    }
                    Err(error) => {
                        failed += 1;
                        tracing::debug!(
                            target: "timeutil::offset",
                            server,
                            error = %error,
                            "sntp query failed"
                        );
                    }
                }
            }
        }
        offsets.sort_unstable();
        OffsetMeasurement {
            offset: median(&offsets).map(DurationMs::from_millis),
            samples_used: u32::try_from(offsets.len()).unwrap_or(u32::MAX),
            queries_failed: failed,
            min_round_trip: min_rtt.map(DurationMs::from_millis),
        }
    }
}

/// Median of a sorted slice; midpoint average for even lengths.
fn median(sorted: &[i64]) -> Option<i64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let mid = n / 2;
    if n % 2 == 1 {
        sorted.get(mid).copied()
    } else {
        let a = sorted.get(mid - 1)?;
        let b = sorted.get(mid)?;
        Some(i64::midpoint(*a, *b))
    }
}

#[cfg(test)]
mod tests {
    use core_types::TimestampMs;

    use super::*;
    use crate::clock::MockClock;

    #[test]
    fn median_odd_even_empty() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[5]), Some(5));
        assert_eq!(median(&[-10, 3, 7]), Some(3));
        assert_eq!(median(&[1, 3]), Some(2));
        assert_eq!(median(&[-5, -1]), Some(-3));
    }

    #[tokio::test]
    async fn all_failures_yield_none_offset() {
        // Unroutable per RFC 5737 (TEST-NET-1) with a tiny timeout: every
        // query fails fast, and the measurement must say so without erroring.
        let source_params = NtpParams {
            servers: vec!["192.0.2.1:123".to_owned()],
            samples_per_server: 2,
            query_timeout: Duration::from_millis(30),
            query_spacing: Duration::from_millis(1),
        };
        let clock = MockClock::new(TimestampMs::from_millis(1_750_000_000_000));
        let mut source = NtpOffsetSource::new(source_params, clock);
        let m = source.measure().await;
        assert_eq!(m.offset, None);
        assert_eq!(m.samples_used, 0);
        assert_eq!(m.queries_failed, 2);
        assert_eq!(m.min_round_trip, None);
    }
}
