//! The quote manager's `client_id` attribution scheme.
//!
//! Every placement carries a `client_id` that encodes `(window, outcome, level,
//! seq)`. Two reasons:
//! 1. **Correlation** — a `place()` returns only `Accepted { client_id, order_id,
//!    state }`; parsing the `client_id` recovers which `(outcome, level)` intent
//!    the venue-assigned `order_id` belongs to (and a `PlaceRejection` echoes only
//!    the `client_id`).
//! 2. **Disambiguation** — `seq` is a single process-global counter the driver
//!    bumps on *every* placement, so a cancel+replace on the same `(outcome,
//!    level)` slot is always a **distinct** id; the old and the replacement are
//!    never confused, even while both briefly exist in the view.
//!
//! Pure and sans-IO.

use core_types::Outcome;

/// Wire prefix marking one of our quote-manager order ids.
const PREFIX: &str = "qm";

/// Parsed placement attribution carried in an order's `client_id`.
///
/// Wire form `qm:{open_ms}:{U|D}:{level}:{seq}`, e.g. `qm:1781000000000:U:0:7`.
/// The window's `series` is intentionally not encoded — the manager trades one
/// active window at a time and always knows its [`WindowId`](core_types::WindowId);
/// `open_ms` is enough to reject a stale-window order folding into a new window's
/// view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientId {
    /// The window's open time in unix millis (its identity within the series).
    pub open_ms: i64,
    /// Which outcome ladder this order belongs to.
    pub outcome: Outcome,
    /// Ladder level index (`0` = touch).
    pub level: u32,
    /// Monotonic per-process placement sequence — makes every id unique.
    pub seq: u64,
}

impl ClientId {
    /// Encodes the attribution into the stable, parseable wire form.
    #[must_use]
    pub fn encode(&self) -> String {
        let o = match self.outcome {
            Outcome::Up => 'U',
            Outcome::Down => 'D',
        };
        format!("{PREFIX}:{}:{o}:{}:{}", self.open_ms, self.level, self.seq)
    }

    /// Parses a wire `client_id` back into attribution. Returns `None` for a
    /// foreign or malformed id — the manager places everything, so a non-`qm`
    /// id is simply ignored, never a panic.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let mut parts = s.split(':');
        if parts.next()? != PREFIX {
            return None;
        }
        let open_ms = parts.next()?.parse::<i64>().ok()?;
        let outcome = match parts.next()? {
            "U" => Outcome::Up,
            "D" => Outcome::Down,
            _ => return None,
        };
        let level = parts.next()?.parse::<u32>().ok()?;
        let seq = parts.next()?.parse::<u64>().ok()?;
        // Reject trailing junk so the form stays exact.
        if parts.next().is_some() {
            return None;
        }
        Some(Self {
            open_ms,
            outcome,
            level,
            seq,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_both_outcomes() {
        for (outcome, tag) in [(Outcome::Up, "U"), (Outcome::Down, "D")] {
            let id = ClientId {
                open_ms: 1_781_000_000_000,
                outcome,
                level: 2,
                seq: 7,
            };
            let wire = id.encode();
            assert_eq!(wire, format!("qm:1781000000000:{tag}:2:7"));
            assert_eq!(ClientId::parse(&wire), Some(id));
        }
    }

    #[test]
    fn rejects_foreign_and_malformed() {
        for bad in [
            "",
            "foo:1:U:0:0",      // wrong prefix
            "qm:1:X:0:0",       // bad outcome
            "qm:notnum:U:0:0",  // bad open_ms
            "qm:1:U:notnum:0",  // bad level
            "qm:1:U:0:notnum",  // bad seq
            "qm:1:U:0",         // too few fields
            "qm:1:U:0:0:extra", // trailing junk
        ] {
            assert_eq!(ClientId::parse(bad), None, "{bad:?} should not parse");
        }
    }

    #[test]
    fn distinct_seq_makes_distinct_ids() {
        let base = ClientId {
            open_ms: 1,
            outcome: Outcome::Up,
            level: 0,
            seq: 0,
        };
        let replacement = ClientId { seq: 1, ..base };
        assert_ne!(base.encode(), replacement.encode());
    }
}
