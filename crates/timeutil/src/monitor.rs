//! Async driver for the clock-skew monitor: runs measurement rounds, feeds
//! the sans-IO [`SkewMonitor`], and announces breaker state on the event bus.
//!
//! The driver only *announces* (`Event::Risk(BreakerTripped/Cleared)` with
//! [`BreakerKind::ClockSkew`]); reacting — the §11 cancel-all — is the risk
//! manager's job.

use std::time::Duration;

use core_types::{BreakerKind, Event, RiskEvent};
use tokio::sync::{mpsc, watch};

use crate::clock::Clock;
use crate::offset::OffsetSource;
use crate::skew::{SkewMonitor, SkewOutput, SkewParams, SkewWarning};

/// Spacing between the startup-burst rounds: a badly skewed clock must alarm
/// within seconds of boot, not after `trip_after × check_interval`.
const STARTUP_SPACING: Duration = Duration::from_secs(2);

/// Cap on any single sleep so staleness ticks keep firing between rounds and
/// clock steps/suspends are noticed within a second (scheduler convention).
const MAX_TICK: Duration = Duration::from_secs(1);

/// Everything [`run_skew_monitor`] needs.
pub struct SkewMonitorArgs<S, C> {
    /// Trip/clear policy.
    pub params: SkewParams,
    /// Steady-state cadence between measurement rounds.
    pub check_interval: Duration,
    /// Offset measurements (production: [`crate::NtpOffsetSource`]).
    pub source: S,
    /// Wall clock for sample ages.
    pub clock: C,
    /// The event bus.
    pub bus_tx: mpsc::Sender<Event>,
    /// Cooperative shutdown signal (`true` = stop).
    pub shutdown_rx: watch::Receiver<bool>,
}

/// Fatal driver failure.
#[derive(Debug, thiserror::Error)]
pub enum SkewMonitorError {
    /// The event bus receiver was dropped — nobody is listening to risk
    /// events, which is never survivable for a §11 component.
    #[error("event bus closed")]
    BusClosed,
}

/// Runs the skew monitor until shutdown.
///
/// The first `params.trip_after` rounds run [`STARTUP_SPACING`] apart (the
/// boot burst), then one round per `check_interval`. Trips and clears go to
/// the bus; warnings go to the log.
///
/// # Errors
/// [`SkewMonitorError::BusClosed`] if the bus receiver is gone.
pub async fn run_skew_monitor<S: OffsetSource, C: Clock>(
    args: SkewMonitorArgs<S, C>,
) -> Result<(), SkewMonitorError> {
    let SkewMonitorArgs {
        params,
        check_interval,
        mut source,
        clock,
        bus_tx,
        mut shutdown_rx,
    } = args;

    let startup_rounds = params.trip_after.max(1);
    let mut monitor = SkewMonitor::new(params);
    let mut rounds_done: u32 = 0;
    let mut out: Vec<SkewOutput> = Vec::new();
    let mut next_round = tokio::time::Instant::now();

    loop {
        if *shutdown_rx.borrow() {
            tracing::info!(target: "timeutil::monitor", "skew monitor shutting down");
            return Ok(());
        }

        if tokio::time::Instant::now() >= next_round {
            // Abandon an in-flight round on shutdown — a measurement can
            // take several seconds (servers × samples × timeouts).
            let measurement = tokio::select! {
                m = source.measure() => m,
                changed = shutdown_rx.changed() => {
                    if changed.is_err() {
                        // Sender dropped: nobody can ever stop us politely.
                        return Ok(());
                    }
                    continue;
                }
            };
            let now = clock.wall();
            tracing::debug!(
                target: "timeutil::monitor",
                offset_ms = measurement.offset.map(|o| o.as_millis()),
                samples_used = measurement.samples_used,
                queries_failed = measurement.queries_failed,
                min_rtt_ms = measurement.min_round_trip.map(|r| r.as_millis()),
                "offset measurement round"
            );
            monitor.on_measurement(&measurement, now, &mut out);
            emit(&mut out, &bus_tx).await?;

            rounds_done = rounds_done.saturating_add(1);
            let spacing = if rounds_done < startup_rounds {
                STARTUP_SPACING
            } else {
                check_interval
            };
            next_round = tokio::time::Instant::now() + spacing;
        }

        monitor.on_tick(clock.wall(), &mut out);
        emit(&mut out, &bus_tx).await?;

        let deadline = next_round.min(tokio::time::Instant::now() + MAX_TICK);
        tokio::select! {
            () = tokio::time::sleep_until(deadline) => {}
            changed = shutdown_rx.changed() => {
                // A dropped sender counts as a shutdown request (and must
                // not busy-loop: a closed watch returns Err immediately).
                if changed.is_err() {
                    tracing::info!(
                        target: "timeutil::monitor",
                        "shutdown channel closed — skew monitor stopping"
                    );
                    return Ok(());
                }
            }
        }
    }
}

/// Drains outputs: trips/clears to the bus, warnings to the log.
async fn emit(
    out: &mut Vec<SkewOutput>,
    bus_tx: &mpsc::Sender<Event>,
) -> Result<(), SkewMonitorError> {
    for output in out.drain(..) {
        match output {
            SkewOutput::Trip { offset } => {
                tracing::error!(
                    target: "timeutil::monitor",
                    offset_ms = offset.as_millis(),
                    "CLOCK SKEW breaker tripped (§11): local clock differs from NTP"
                );
                bus_tx
                    .send(Event::Risk(RiskEvent::BreakerTripped {
                        breaker: BreakerKind::ClockSkew,
                    }))
                    .await
                    .map_err(|_| SkewMonitorError::BusClosed)?;
            }
            SkewOutput::Clear { offset } => {
                tracing::info!(
                    target: "timeutil::monitor",
                    offset_ms = offset.as_millis(),
                    "clock skew breaker cleared"
                );
                bus_tx
                    .send(Event::Risk(RiskEvent::BreakerCleared {
                        breaker: BreakerKind::ClockSkew,
                    }))
                    .await
                    .map_err(|_| SkewMonitorError::BusClosed)?;
            }
            SkewOutput::Warn(warning) => match warning {
                SkewWarning::BreachObserved {
                    offset,
                    consecutive,
                } => {
                    tracing::warn!(
                        target: "timeutil::monitor",
                        offset_ms = offset.as_millis(),
                        consecutive,
                        "clock offset breach observed (below trip debounce)"
                    );
                }
                SkewWarning::RoundFailed {
                    consecutive_failures,
                } => {
                    tracing::warn!(
                        target: "timeutil::monitor",
                        consecutive_failures,
                        "ntp measurement round failed (no usable samples)"
                    );
                }
                SkewWarning::OffsetStale { age } => {
                    tracing::warn!(
                        target: "timeutil::monitor",
                        age_ms = age.as_millis(),
                        "clock offset is STALE — no successful NTP round; \
                         treat persistent staleness as a deployment blocker"
                    );
                }
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use core_types::{DurationMs, TimestampMs};

    use super::*;
    use crate::clock::MockClock;
    use crate::offset::OffsetMeasurement;

    /// Scripted offset source: pops the front measurement, then repeats the
    /// last one forever.
    struct FakeSource {
        script: VecDeque<OffsetMeasurement>,
        last: OffsetMeasurement,
    }

    impl FakeSource {
        fn new(script: Vec<OffsetMeasurement>) -> Self {
            let last = script.last().copied().unwrap_or(OffsetMeasurement {
                offset: Some(DurationMs::ZERO),
                samples_used: 1,
                queries_failed: 0,
                min_round_trip: Some(DurationMs::ZERO),
            });
            Self {
                script: script.into(),
                last,
            }
        }
    }

    impl OffsetSource for FakeSource {
        async fn measure(&mut self) -> OffsetMeasurement {
            self.script.pop_front().unwrap_or(self.last)
        }
    }

    fn breaching(ms: i64) -> OffsetMeasurement {
        OffsetMeasurement {
            offset: Some(DurationMs::from_millis(ms)),
            samples_used: 4,
            queries_failed: 0,
            min_round_trip: Some(DurationMs::from_millis(8)),
        }
    }

    fn params() -> SkewParams {
        SkewParams {
            trip_bound: DurationMs::from_millis(250),
            clear_bound: DurationMs::from_millis(125),
            trip_after: 2,
            clear_after: 3,
            stale_warn: DurationMs::from_millis(300_000),
        }
    }

    /// THE acceptance test: with a mocked clock and a skewed offset source,
    /// the clock-skew alarm fires on the event bus, and clears once the
    /// offsets come back in bound.
    #[tokio::test(start_paused = true)]
    async fn alarm_fires_on_bus_with_mocked_clock() {
        let clock = MockClock::new(TimestampMs::from_millis(1_750_000_000_000));
        // Two breaching rounds (the startup burst), then in-bound forever.
        let source = FakeSource::new(vec![
            breaching(400),
            breaching(380),
            breaching(10),
            breaching(5),
            breaching(0),
        ]);
        let (bus_tx, mut bus_rx) = mpsc::channel::<Event>(16);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let driver = tokio::spawn(run_skew_monitor(SkewMonitorArgs {
            params: params(),
            check_interval: Duration::from_secs(60),
            source,
            clock,
            bus_tx,
            shutdown_rx,
        }));

        // The startup burst (2 rounds, 2 s apart) must produce the trip.
        let tripped = tokio::time::timeout(Duration::from_secs(30), bus_rx.recv())
            .await
            .expect("no trip within the startup burst")
            .expect("bus closed early");
        assert_eq!(
            tripped,
            Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew,
            })
        );

        // Three in-bound rounds at the steady 60 s cadence must clear it.
        let cleared = tokio::time::timeout(Duration::from_secs(600), bus_rx.recv())
            .await
            .expect("no clear after in-bound rounds")
            .expect("bus closed early");
        assert_eq!(
            cleared,
            Event::Risk(RiskEvent::BreakerCleared {
                breaker: BreakerKind::ClockSkew,
            })
        );

        shutdown_tx.send(true).expect("driver gone");
        let result = tokio::time::timeout(Duration::from_secs(5), driver)
            .await
            .expect("driver did not shut down")
            .expect("driver panicked");
        assert!(result.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn bus_closed_is_fatal() {
        let clock = MockClock::new(TimestampMs::from_millis(1_750_000_000_000));
        let source = FakeSource::new(vec![breaching(400), breaching(400)]);
        let (bus_tx, bus_rx) = mpsc::channel::<Event>(16);
        drop(bus_rx);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let result = tokio::time::timeout(
            Duration::from_secs(30),
            run_skew_monitor(SkewMonitorArgs {
                params: params(),
                check_interval: Duration::from_secs(60),
                source,
                clock,
                bus_tx,
                shutdown_rx,
            }),
        )
        .await
        .expect("driver hung on a closed bus");
        assert!(matches!(result, Err(SkewMonitorError::BusClosed)));
    }

    #[tokio::test(start_paused = true)]
    async fn dropped_shutdown_sender_stops_the_driver() {
        let clock = MockClock::new(TimestampMs::from_millis(1_750_000_000_000));
        let source = FakeSource::new(vec![]);
        let (bus_tx, _bus_rx) = mpsc::channel::<Event>(16);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let driver = tokio::spawn(run_skew_monitor(SkewMonitorArgs {
            params: params(),
            check_interval: Duration::from_secs(60),
            source,
            clock,
            bus_tx,
            shutdown_rx,
        }));
        // Give the driver a beat to start, then drop the sender WITHOUT
        // signalling: the driver must stop (and not busy-loop) regardless.
        tokio::time::sleep(Duration::from_secs(1)).await;
        drop(shutdown_tx);
        let result = tokio::time::timeout(Duration::from_secs(5), driver)
            .await
            .expect("driver did not stop on a dropped shutdown sender")
            .expect("driver panicked");
        assert!(result.is_ok());
    }
}
