//! REST round-trip probe: warmup requests (TLS + connection-pool setup),
//! then timed full round-trips. The warm-pool RTT is what the SDK's order
//! path sees in production, which is exactly the number the paper simulator
//! and the §8 latency premium need.

use std::time::{Duration, Instant};

use core_types::TimestampMs;

use crate::clock::Clock;
use crate::harness::report::{ClobTimeReport, RestTargetReport};
use crate::harness::{HarnessParams, RestTarget};
use crate::stats::stats_from_ms;

/// Probes one REST target: `warmup` untimed requests, then `samples` timed
/// ones spaced `spacing` apart. Non-2xx or transport errors count as error
/// samples (excluded from stats); the probe itself never fails.
pub(crate) async fn probe_rest_target(
    http: &reqwest::Client,
    target: &RestTarget,
    params: &HarnessParams,
) -> RestTargetReport {
    let mut report = RestTargetReport {
        label: target.label.clone(),
        url: target.url.clone(),
        samples: params.rest_samples,
        errors: 0,
        stats: None,
        error: None,
    };

    for _ in 0..params.rest_warmup {
        // Warmup failures are ignored: the timed loop will surface anything
        // persistent as error samples.
        if let Ok(response) = http.get(&target.url).send().await {
            let _ = response.bytes().await;
        }
    }

    let mut rtts: Vec<f64> = Vec::with_capacity(params.rest_samples as usize);
    for i in 0..params.rest_samples {
        if i > 0 {
            tokio::time::sleep(params.rest_spacing).await;
        }
        let start = Instant::now();
        let outcome = match http.get(&target.url).send().await {
            // The body read is part of the round-trip the order path pays.
            Ok(response) => match response.error_for_status() {
                Ok(ok) => ok.bytes().await.map(|_| ()),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match outcome {
            Ok(()) => rtts.push(duration_ms(start.elapsed())),
            Err(error) => {
                report.errors += 1;
                tracing::debug!(
                    target: "timeutil::harness",
                    url = %target.url,
                    error = %error,
                    "rest sample failed"
                );
            }
        }
    }

    report.stats = stats_from_ms(&mut rtts);
    report
}

/// CLOB `GET /time` venue-clock cross-check: parses the raw unix-seconds
/// body and compares against the local wall clock at receive time.
pub(crate) async fn probe_clob_time(
    http: &reqwest::Client,
    url: &str,
    clock: &impl Clock,
) -> ClobTimeReport {
    let start = Instant::now();
    let outcome = async {
        let response = http.get(url).send().await?.error_for_status()?;
        response.bytes().await
    }
    .await;
    let rtt = duration_ms(start.elapsed());
    let local = clock.wall();

    match outcome {
        Ok(body) => {
            let text = String::from_utf8_lossy(&body);
            match text.trim().parse::<i64>() {
                Ok(server_secs) => ClobTimeReport {
                    server_unix_secs: Some(server_secs),
                    local_unix_ms: local.as_millis(),
                    delta_ms: Some(delta_ms(server_secs, local)),
                    rtt_ms: Some(rtt),
                    error: None,
                },
                Err(_) => ClobTimeReport {
                    server_unix_secs: None,
                    local_unix_ms: local.as_millis(),
                    delta_ms: None,
                    rtt_ms: Some(rtt),
                    error: Some(format!(
                        "unparseable /time body: {:?}",
                        &text[..text.len().min(64)]
                    )),
                },
            }
        }
        Err(error) => ClobTimeReport {
            server_unix_secs: None,
            local_unix_ms: local.as_millis(),
            delta_ms: None,
            rtt_ms: None,
            error: Some(error.to_string()),
        },
    }
}

fn delta_ms(server_secs: i64, local: TimestampMs) -> i64 {
    server_secs
        .saturating_mul(1000)
        .saturating_sub(local.as_millis())
}

pub(crate) fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_is_server_minus_local() {
        let local = TimestampMs::from_millis(1_750_000_000_123);
        assert_eq!(delta_ms(1_750_000_000, local), -123);
        assert_eq!(delta_ms(1_750_000_001, local), 877);
    }

    #[test]
    fn duration_ms_converts() {
        assert_eq!(duration_ms(Duration::from_millis(250)), 250.0);
        assert!((duration_ms(Duration::from_micros(1500)) - 1.5).abs() < 1e-9);
    }
}
