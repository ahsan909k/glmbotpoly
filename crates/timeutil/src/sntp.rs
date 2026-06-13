//! Minimal SNTPv4 client (RFC 4330), hand-rolled over `tokio::net::UdpSocket`.
//!
//! No NTP crate is on the §3 allowlist, and the whole protocol need is one
//! 48-byte packet: send a client request stamped with T1, receive the server's
//! (T2 receive, T3 transmit) stamps, note T4 locally, and compute
//! `offset = ((T2−T1) + (T3−T4)) / 2`, `delay = (T4−T1) − (T3−T2)`.
//!
//! All timestamp arithmetic happens in NTP 64-bit fixed point (32 bits of
//! seconds since 1900, 32 bits of binary fraction) using **wrapping**
//! subtraction, which makes the math immune to the 2036 era rollover for any
//! real offset under ±68 years.
//!
//! T1/T4 are deliberately **wall**-clock readings — the wall clock is the
//! thing being measured. A wall step in the middle of a query corrupts that
//! one sample; the caller (`offset::NtpOffsetSource`) takes a median across
//! several, which absorbs it.

use std::time::Duration;

use core_types::{DurationMs, TimestampMs};

use crate::clock::Clock;

/// Seconds between the NTP epoch (1900-01-01) and the unix epoch (1970-01-01).
const NTP_UNIX_OFFSET_SECS: i64 = 2_208_988_800;

/// Wire size of an SNTP packet without authentication.
const PACKET_LEN: usize = 48;

/// LI=0, VN=4, Mode=3 (client).
const REQUEST_HEADER: u8 = 0x23;

/// One successful SNTP exchange.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SntpSample {
    /// `true_time − local_wall`: positive means the local clock is behind.
    pub offset: DurationMs,
    /// Network round-trip (excluding server processing time).
    pub round_trip: DurationMs,
    /// The server this sample came from, as configured.
    pub server: String,
}

/// SNTP query failure.
#[derive(Debug, thiserror::Error)]
pub enum SntpError {
    /// No response within the deadline.
    #[error("sntp query timed out after {0:?}")]
    Timeout(Duration),
    /// Socket-level failure (bind, send, recv, DNS).
    #[error("sntp udp io: {0}")]
    Io(#[from] std::io::Error),
    /// DNS resolved to zero addresses.
    #[error("sntp dns: no address for {0:?}")]
    NoAddress(String),
    /// The response failed protocol validation.
    #[error("sntp bad response: {0}")]
    BadResponse(&'static str),
    /// Stratum-0 kiss-of-death (e.g. RATE, DENY) — stop using this server.
    #[error("sntp kiss-of-death from server (code {0:?})")]
    KissOfDeath(String),
}

/// Converts unix milliseconds to NTP 64-bit fixed point (era-wrapped).
fn ms_to_ntp(ms: i64) -> u64 {
    let secs = ms.div_euclid(1000).wrapping_add(NTP_UNIX_OFFSET_SECS);
    let frac_ms = ms.rem_euclid(1000) as u64; // 0..=999
    let frac = (frac_ms << 32) / 1000;
    ((secs as u64) << 32) | frac
}

/// Signed delta `a − b` between two NTP fixed-point stamps, in milliseconds.
///
/// Wrapping subtraction first (era-safe), then a single fixed-point→ms
/// conversion in `i128` so nothing overflows.
#[cfg(test)]
fn ntp_delta_ms(a: u64, b: u64) -> i64 {
    let delta = a.wrapping_sub(b) as i64; // signed 32.32 fixed point
    fixed_to_ms(i128::from(delta))
}

/// Converts a signed 32.32 fixed-point second count to milliseconds,
/// rounding to nearest.
///
/// Rounding (not truncation) matters: `ms_to_ntp` already truncated the
/// fraction once, and truncating again here would systematically lose 1 ms
/// (e.g. a 350 ms delta reading back as 349).
fn fixed_to_ms(fixed: i128) -> i64 {
    // (x*1000 + 2^31) >> 32 — arithmetic shift is floor, so adding half a
    // unit first gives round-to-nearest for both signs.
    i64::try_from((fixed * 1000 + (1_i128 << 31)) >> 32).unwrap_or(if fixed < 0 {
        i64::MIN
    } else {
        i64::MAX
    })
}

/// Builds the 48-byte client request with `t1_wall` in the Transmit field.
fn encode_request(t1_wall: TimestampMs) -> [u8; PACKET_LEN] {
    let mut buf = [0u8; PACKET_LEN];
    buf[0] = REQUEST_HEADER;
    buf[40..48].copy_from_slice(&ms_to_ntp(t1_wall.as_millis()).to_be_bytes());
    buf
}

fn read_u64(buf: &[u8], at: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[at..at + 8]);
    u64::from_be_bytes(bytes)
}

/// Validates a server response and computes `(offset, delay)` in ms.
///
/// `t1`/`t4` are the local wall stamps taken immediately before send and
/// after receive.
///
/// # Errors
/// [`SntpError::BadResponse`] on length/mode/leap/origin violations,
/// [`SntpError::KissOfDeath`] on a stratum-0 reply.
fn decode_response(
    buf: &[u8],
    t1: TimestampMs,
    t4: TimestampMs,
) -> Result<(DurationMs, DurationMs), SntpError> {
    if buf.len() < PACKET_LEN {
        return Err(SntpError::BadResponse("packet shorter than 48 bytes"));
    }
    let mode = buf[0] & 0x07;
    if mode != 4 {
        return Err(SntpError::BadResponse("mode is not 4 (server)"));
    }
    if buf[0] >> 6 == 3 {
        return Err(SntpError::BadResponse(
            "leap indicator 3 (clock unsynchronized)",
        ));
    }
    if buf[1] == 0 {
        // Kiss-of-death: the reference-id field carries a 4-char ASCII code.
        let code = String::from_utf8_lossy(&buf[12..16]).into_owned();
        return Err(SntpError::KissOfDeath(code));
    }
    let t1_ntp = ms_to_ntp(t1.as_millis());
    if read_u64(buf, 24) != t1_ntp {
        // The Originate field must echo our Transmit stamp — pairs the
        // response to OUR request (stale/foreign datagram defense).
        return Err(SntpError::BadResponse(
            "originate stamp does not echo our request",
        ));
    }
    let t2 = read_u64(buf, 32);
    let t3 = read_u64(buf, 40);
    if t3 == 0 {
        return Err(SntpError::BadResponse("zero transmit timestamp"));
    }
    let t4_ntp = ms_to_ntp(t4.as_millis());

    let d21 = i128::from(t2.wrapping_sub(t1_ntp) as i64);
    let d34 = i128::from(t3.wrapping_sub(t4_ntp) as i64);
    let offset = fixed_to_ms((d21 + d34) / 2);

    let d41 = i128::from(t4_ntp.wrapping_sub(t1_ntp) as i64);
    let d32 = i128::from(t3.wrapping_sub(t2) as i64);
    let delay = fixed_to_ms(d41 - d32);

    Ok((
        DurationMs::from_millis(offset),
        DurationMs::from_millis(delay),
    ))
}

/// Runs one SNTP exchange against `server` (`"host"` or `"host:port"`; the
/// port defaults to 123 — explicit ports make loopback fake-server tests
/// possible). IPv6 literals must be bracketed (`"[::1]:123"`).
///
/// # Errors
/// Any [`SntpError`]; network failures here say nothing about local clock
/// health (see the skew policy in [`crate::SkewMonitor`]).
pub async fn query(
    server: &str,
    timeout: Duration,
    clock: &impl Clock,
) -> Result<SntpSample, SntpError> {
    let target = if server.contains(':') {
        server.to_owned()
    } else {
        format!("{server}:123")
    };
    let addr = tokio::net::lookup_host(&target)
        .await?
        .next()
        .ok_or_else(|| SntpError::NoAddress(server.to_owned()))?;
    let bind_addr = if addr.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    };
    let socket = tokio::net::UdpSocket::bind(bind_addr).await?;
    socket.connect(addr).await?;

    let t1 = clock.wall();
    socket.send(&encode_request(t1)).await?;
    let mut buf = [0u8; 68];
    let n = tokio::time::timeout(timeout, socket.recv(&mut buf))
        .await
        .map_err(|_| SntpError::Timeout(timeout))??;
    let t4 = clock.wall();

    let (offset, round_trip) = decode_response(&buf[..n], t1, t4)?;
    Ok(SntpSample {
        offset,
        round_trip,
        server: server.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MockClock;

    /// Builds a server response echoing `t1` with the given T2/T3 stamps.
    fn response(t1: TimestampMs, t2_ms: i64, t3_ms: i64) -> [u8; PACKET_LEN] {
        let mut buf = [0u8; PACKET_LEN];
        buf[0] = 0x24; // LI=0, VN=4, Mode=4 (server)
        buf[1] = 2; // stratum 2
        buf[24..32].copy_from_slice(&ms_to_ntp(t1.as_millis()).to_be_bytes());
        buf[32..40].copy_from_slice(&ms_to_ntp(t2_ms).to_be_bytes());
        buf[40..48].copy_from_slice(&ms_to_ntp(t3_ms).to_be_bytes());
        buf
    }

    #[test]
    fn request_header_and_t1_round_trip() {
        let t1 = TimestampMs::from_millis(1_718_000_000_123);
        let req = encode_request(t1);
        assert_eq!(req[0], 0x23);
        assert!(req[1..40].iter().all(|&b| b == 0));
        // The transmit stamp must decode back to T1 (fixed-point fraction
        // rounds down: ms-exact both ways for any input).
        let t1_back = ntp_delta_ms(read_u64(&req, 40), ms_to_ntp(0));
        assert_eq!(t1_back, t1.as_millis());
    }

    #[test]
    fn decode_computes_known_offsets_both_signs() {
        // Local clock 300 ms BEHIND: server stamps run 300 ms ahead of ours,
        // symmetric 50 ms each-way network.
        let t1 = TimestampMs::from_millis(1_000_000);
        let buf = response(t1, 1_000_350, 1_000_360); // T2 = t1+50+300
        let t4 = TimestampMs::from_millis(1_000_110); // 10 ms server processing
        let (offset, delay) = decode_response(&buf, t1, t4).unwrap();
        assert_eq!(offset.as_millis(), 300);
        assert_eq!(delay.as_millis(), 100);

        // Local clock 300 ms AHEAD.
        let buf = response(t1, 999_750, 999_760);
        let (offset, delay) = decode_response(&buf, t1, t4).unwrap();
        assert_eq!(offset.as_millis(), -300);
        assert_eq!(delay.as_millis(), 100);
    }

    #[test]
    fn decode_rejects_bad_packets() {
        let t1 = TimestampMs::from_millis(1_000_000);
        let t4 = TimestampMs::from_millis(1_000_100);
        let good = response(t1, 1_000_050, 1_000_060);

        assert!(matches!(
            decode_response(&good[..40], t1, t4),
            Err(SntpError::BadResponse(_))
        ));

        let mut bad_mode = good;
        bad_mode[0] = 0x23; // client mode, not server
        assert!(matches!(
            decode_response(&bad_mode, t1, t4),
            Err(SntpError::BadResponse(_))
        ));

        let mut alarm = good;
        alarm[0] = 0xE4; // LI=3
        assert!(matches!(
            decode_response(&alarm, t1, t4),
            Err(SntpError::BadResponse(_))
        ));

        let mut kod = good;
        kod[1] = 0;
        kod[12..16].copy_from_slice(b"RATE");
        match decode_response(&kod, t1, t4) {
            Err(SntpError::KissOfDeath(code)) => assert_eq!(code, "RATE"),
            other => panic!("expected KissOfDeath, got {other:?}"),
        }

        // Originate mismatch: response paired to some other request.
        let foreign = response(TimestampMs::from_millis(999_999), 1_000_050, 1_000_060);
        assert!(matches!(
            decode_response(&foreign, t1, t4),
            Err(SntpError::BadResponse(_))
        ));

        let mut zero_t3 = good;
        zero_t3[40..48].copy_from_slice(&[0u8; 8]);
        assert!(matches!(
            decode_response(&zero_t3, t1, t4),
            Err(SntpError::BadResponse(_))
        ));
    }

    #[test]
    fn era_wrap_safe_offset() {
        // NTP era 0 ends 2036-02-07. Place the exchange right across the
        // boundary: local wall just before the wrap, server stamps just
        // after. Wrapping fixed-point math must still yield the small true
        // offset instead of ±136 years.
        let era_end_unix_ms = (0x1_0000_0000_i64 - NTP_UNIX_OFFSET_SECS) * 1000;
        let t1 = TimestampMs::from_millis(era_end_unix_ms - 400);
        let t4 = TimestampMs::from_millis(era_end_unix_ms - 300);
        let buf = response(t1, era_end_unix_ms + 150, era_end_unix_ms + 160);
        let (offset, delay) = decode_response(&buf, t1, t4).unwrap();
        assert_eq!(offset.as_millis(), 505);
        assert_eq!(delay.as_millis(), 90);
    }

    #[tokio::test]
    async fn loopback_query_end_to_end() {
        // Fake SNTP server: reads one request, echoes it back as a stratum-2
        // server response with T2/T3 = client T1 + 250 ms (so the measured
        // offset is ~+250 with ~zero recorded network delay).
        let server = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let serve = tokio::spawn(async move {
            let mut buf = [0u8; PACKET_LEN];
            let (n, peer) = server.recv_from(&mut buf).await.unwrap();
            assert_eq!(n, PACKET_LEN);
            assert_eq!(buf[0], 0x23);
            let t1 = read_u64(&buf, 40);
            let mut resp = [0u8; PACKET_LEN];
            resp[0] = 0x24;
            resp[1] = 2;
            resp[24..32].copy_from_slice(&t1.to_be_bytes());
            let skewed = t1.wrapping_add(((250_u64) << 32) / 1000);
            resp[32..40].copy_from_slice(&skewed.to_be_bytes());
            resp[40..48].copy_from_slice(&skewed.to_be_bytes());
            server.send_to(&resp, peer).await.unwrap();
        });

        // MockClock: wall frozen during the query, so T1 == T4 and the
        // computed offset is exactly the injected +250 ms.
        let clock = MockClock::new(TimestampMs::from_millis(1_750_000_000_000));
        let target = format!("127.0.0.1:{}", server_addr.port());
        let sample = query(&target, Duration::from_secs(2), &clock)
            .await
            .unwrap();
        serve.await.unwrap();
        assert_eq!(sample.offset.as_millis(), 250);
        assert_eq!(sample.round_trip.as_millis(), 0);
        assert_eq!(sample.server, target);
    }

    #[tokio::test]
    async fn query_times_out_against_silent_server() {
        // A bound socket that never answers.
        let silent = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let target = format!("127.0.0.1:{}", silent.local_addr().unwrap().port());
        let clock = MockClock::new(TimestampMs::from_millis(1_750_000_000_000));
        let err = query(&target, Duration::from_millis(50), &clock)
            .await
            .unwrap_err();
        assert!(matches!(err, SntpError::Timeout(_)));
    }
}
