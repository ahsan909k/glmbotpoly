//! Config → `timeutil` parameter mapping.
//!
//! `timeutil` depends only on `core-types` (§4), so the translation from
//! [`config::ClockConfig`] to its parameter structs lives here at the binary
//! boundary (the inverse of scheduler's `Timing::from_config`).

use std::time::Duration;

use core_types::DurationMs;

/// Converts a validated config duration to `std::time::Duration` (negative
/// values clamp to zero — validation already rejects them at boot).
pub(crate) fn std_duration(d: DurationMs) -> Duration {
    Duration::from_millis(d.as_millis().max(0).unsigned_abs())
}

/// Maps the `[clock]` section to NTP measurement parameters.
pub(crate) fn ntp_params(cfg: &config::ClockConfig) -> timeutil::NtpParams {
    timeutil::NtpParams {
        servers: cfg.ntp_servers.clone(),
        samples_per_server: cfg.samples_per_server,
        query_timeout: std_duration(cfg.query_timeout_ms),
        query_spacing: std_duration(cfg.query_spacing_ms),
    }
}

/// Maps the `[clock]` section to the skew trip/clear policy.
pub(crate) fn skew_params(cfg: &config::ClockConfig) -> timeutil::SkewParams {
    timeutil::SkewParams {
        trip_bound: cfg.trip_bound_ms,
        clear_bound: cfg.clear_bound_ms,
        trip_after: cfg.trip_after,
        clear_after: cfg.clear_after,
        stale_warn: cfg.stale_warn_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_convert_and_clamp() {
        assert_eq!(
            std_duration(DurationMs::from_millis(1_500)),
            Duration::from_millis(1_500)
        );
        assert_eq!(std_duration(DurationMs::from_millis(-5)), Duration::ZERO);
    }

    #[test]
    fn clock_section_maps_one_to_one() {
        let cfg = config::ClockConfig::default();
        let ntp = ntp_params(&cfg);
        assert_eq!(ntp.servers, cfg.ntp_servers);
        assert_eq!(ntp.samples_per_server, cfg.samples_per_server);
        let skew = skew_params(&cfg);
        assert_eq!(skew.trip_bound, cfg.trip_bound_ms);
        assert_eq!(skew.clear_bound, cfg.clear_bound_ms);
        assert_eq!(skew.trip_after, cfg.trip_after);
        assert_eq!(skew.clear_after, cfg.clear_after);
        assert_eq!(skew.stale_warn, cfg.stale_warn_ms);
    }
}
