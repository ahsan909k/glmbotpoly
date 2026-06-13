//! RTDS glue over the generic feed-util driver: the venue protocol
//! ([`FeedProtocol`] impl — connect-message sequence, frame classification,
//! command handling) plus the public [`run`] entry point that maps
//! [`RtdsParams`] onto generic driver parameters. The connection lifecycle,
//! keepalive, watchdog, dead-socket teardown, and starvation recycle all
//! live in feed-util (extracted verbatim 2026-06-12 — see Decisions Log).
//!
//! Backfill note: the first tick per subscription may be ~2 minutes old (the
//! RTDS backfill message). It is published with its honest old
//! `ts_exchange` — consumers judge data recency themselves (CLAUDE.md §9
//! conservatism: the feed never fabricates freshness).

use std::time::Duration;

use core_types::{DurationMs, Event, TimestampMs};
use feed_util::{
    BackoffParams, CommandAction, DriverArgs, DriverParams, FeedProtocol, FrameOutcome, Keepalive,
    PriceObs, Transport,
};
use tokio::sync::{mpsc, watch};

use crate::sub::{FeedCommand, FeedSub, SubSet};
use crate::wire::{IgnoredReason, ParsedFrame, parse_frame};
use crate::{FeedError, FeedStatus};

/// The client keepalive text; the RTDS docs require one at least every 5 s.
const PING_TEXT: &str = "PING";

/// A connection with no frames at all (not even PONGs) for this many ping
/// intervals is assumed half-open and torn down.
const DEAD_SOCKET_PINGS: u32 = 3;

/// Connection parameters (mapped from `feeds.*` config by the binary — this
/// crate depends only on core-types and feed-util).
#[derive(Debug, Clone, PartialEq)]
pub struct RtdsParams {
    /// RTDS endpoint, e.g. `wss://ws-live-data.polymarket.com`.
    pub url: String,
    /// Keepalive PING interval (config validates (0, 5000]).
    pub ping_interval: Duration,
    /// TCP+TLS+WS handshake deadline.
    pub connect_timeout: Duration,
    /// Reconnect backoff curve.
    pub backoff: BackoffParams,
    /// Per-stream staleness threshold for the watchdog.
    pub stale_after: DurationMs,
}

/// Everything [`run`] needs. Construct one per feed instance.
pub struct RtdsArgs<T, F> {
    /// Connection parameters.
    pub params: RtdsParams,
    /// Initial subscriptions (the bot passes all four streams).
    pub subscriptions: Vec<FeedSub>,
    /// The transport (real: [`feed_util::WsTransport`]; tests script a fake).
    pub transport: T,
    /// Wall-clock source; the only clock surface in the crate.
    pub now_fn: F,
    /// The internal bus. Sends are awaited — backpressure is explicit.
    pub bus_tx: mpsc::Sender<Event>,
    /// Runtime subscription changes; `None` until a control plane exists.
    pub command_rx: Option<mpsc::Receiver<FeedCommand>>,
    /// Optional status snapshots (CLI table, dashboard later).
    pub status_tx: Option<watch::Sender<FeedStatus>>,
    /// Flip to `true` for graceful shutdown.
    pub shutdown_rx: watch::Receiver<bool>,
    /// Backoff jitter seed; `None` seeds from the clock. Tests inject one
    /// for determinism.
    pub backoff_seed: Option<u64>,
}

/// Runs the feed until shutdown.
///
/// # Errors
/// [`FeedError::BusClosed`] when every bus receiver is gone. Connection
/// trouble is logged and retried forever, never returned.
pub async fn run<T, F>(args: RtdsArgs<T, F>) -> Result<(), FeedError>
where
    T: Transport,
    F: Fn() -> TimestampMs + Send,
{
    let RtdsArgs {
        params,
        subscriptions,
        transport,
        now_fn,
        bus_tx,
        command_rx,
        status_tx,
        shutdown_rx,
        backoff_seed,
    } = args;
    let protocol = RtdsProtocol {
        subs: SubSet::new(subscriptions),
        stale_after: params.stale_after,
    };
    feed_util::run(DriverArgs {
        params: DriverParams {
            url: params.url,
            connect_timeout: params.connect_timeout,
            backoff: params.backoff,
            dead_after: params.ping_interval.saturating_mul(DEAD_SOCKET_PINGS),
            keepalive: Some(Keepalive {
                interval: params.ping_interval,
                text: PING_TEXT.to_owned(),
            }),
        },
        protocol,
        transport,
        now_fn,
        bus_tx,
        command_rx,
        status_tx,
        shutdown_rx,
        backoff_seed,
    })
    .await
}

/// The RTDS venue protocol. One instance lives for the driver's whole life;
/// the desired-subscription set persists across reconnects.
struct RtdsProtocol {
    subs: SubSet,
    stale_after: DurationMs,
}

impl FeedProtocol for RtdsProtocol {
    type Key = FeedSub;
    type Reason = IgnoredReason;
    type Command = FeedCommand;

    fn name(&self) -> &'static str {
        "rtds"
    }

    fn keys(&self) -> Vec<(FeedSub, DurationMs)> {
        self.subs
            .iter()
            .map(|sub| (sub, self.stale_after))
            .collect()
    }

    /// Resubscribe sequence per topic: filtered subscribes (each triggers a
    /// ~2-minute backfill seeding the model) then the unfiltered all-symbols
    /// steady state — RTDS holds one filter slot per topic (sub.rs module
    /// docs).
    fn connect_messages(&self) -> Vec<String> {
        self.subs.connect_messages()
    }

    fn handle_frame(
        &mut self,
        text: &str,
        _now: TimestampMs,
    ) -> FrameOutcome<FeedSub, IgnoredReason> {
        match parse_frame(text) {
            ParsedFrame::Ack => FrameOutcome::Ack,
            ParsedFrame::Pong => FrameOutcome::Keepalive,
            ParsedFrame::Prices(updates) => FrameOutcome::Prices(
                updates
                    .into_iter()
                    .map(|update| PriceObs {
                        key: FeedSub::new(update.source, update.asset),
                        value: update.value,
                        ts_exchange: update.ts_exchange,
                    })
                    .collect(),
            ),
            // Untracked symbols are normal traffic under the all-symbols
            // subscription — never a warning.
            ParsedFrame::Ignored(IgnoredReason::UnknownSymbol) => {
                FrameOutcome::IgnoredQuiet(IgnoredReason::UnknownSymbol)
            }
            ParsedFrame::Ignored(reason) => FrameOutcome::Ignored(reason),
        }
    }

    fn on_command(&mut self, command: FeedCommand) -> Vec<CommandAction<FeedSub>> {
        match command {
            FeedCommand::Subscribe(sub) => {
                // Idempotence gate: track + send only when newly desired.
                let messages = self.subs.add(sub);
                if messages.is_empty() {
                    return Vec::new();
                }
                std::iter::once(CommandAction::Track {
                    key: sub,
                    stale_after: self.stale_after,
                })
                .chain(messages.into_iter().map(CommandAction::Send))
                .collect()
            }
            FeedCommand::Unsubscribe(sub) => {
                if !self.subs.iter().any(|s| s == sub) {
                    return Vec::new();
                }
                // Removal may need no wire action (the topic keeps streaming
                // for its other tracked symbol) — the machine drop is what
                // stops publication either way.
                let messages = self.subs.remove(sub);
                std::iter::once(CommandAction::Untrack(sub))
                    .chain(messages.into_iter().map(CommandAction::Send))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::Asset;

    use super::*;
    use crate::wire::{
        RtdsSource, backfill_subscribe_message, stream_subscribe_message,
        stream_unsubscribe_message,
    };

    fn protocol() -> RtdsProtocol {
        RtdsProtocol {
            subs: SubSet::new([FeedSub::new(RtdsSource::Binance, Asset::Btc)]),
            stale_after: DurationMs::from_millis(5_000),
        }
    }

    #[test]
    fn frames_classify_through_the_protocol() {
        let mut p = protocol();
        let now = TimestampMs::from_millis(0);
        assert_eq!(p.handle_frame("", now), FrameOutcome::Ack);
        assert_eq!(p.handle_frame("PONG", now), FrameOutcome::Keepalive);
        // Untracked symbol → quiet skip; other junk → warn-gated skip.
        assert_eq!(
            p.handle_frame(
                r#"{"topic":"crypto_prices","payload":{"symbol":"solusdt","timestamp":1,"value":1.0}}"#,
                now
            ),
            FrameOutcome::IgnoredQuiet(IgnoredReason::UnknownSymbol)
        );
        assert_eq!(
            p.handle_frame("not json", now),
            FrameOutcome::Ignored(IgnoredReason::MalformedJson)
        );
        let FrameOutcome::Prices(obs) = p.handle_frame(
            r#"{"topic":"crypto_prices","payload":{"symbol":"btcusdt","timestamp":7,"value":63500.5}}"#,
            now,
        ) else {
            panic!("expected prices");
        };
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].key, FeedSub::new(RtdsSource::Binance, Asset::Btc));
        assert_eq!(obs[0].ts_exchange, TimestampMs::from_millis(7));
    }

    #[test]
    fn subscribe_command_tracks_before_sending_and_is_idempotent() {
        let mut p = protocol();
        let sub = FeedSub::new(RtdsSource::Chainlink, Asset::Btc);
        let actions = p.on_command(FeedCommand::Subscribe(sub));
        assert_eq!(
            actions,
            vec![
                CommandAction::Track {
                    key: sub,
                    stale_after: DurationMs::from_millis(5_000)
                },
                CommandAction::Send(backfill_subscribe_message(
                    RtdsSource::Chainlink,
                    Asset::Btc
                )),
                CommandAction::Send(stream_subscribe_message(RtdsSource::Chainlink)),
            ]
        );
        assert!(p.on_command(FeedCommand::Subscribe(sub)).is_empty());
    }

    #[test]
    fn unsubscribe_command_untracks_even_without_wire_messages() {
        let mut p = RtdsProtocol {
            subs: SubSet::new([
                FeedSub::new(RtdsSource::Binance, Asset::Btc),
                FeedSub::new(RtdsSource::Binance, Asset::Eth),
            ]),
            stale_after: DurationMs::from_millis(5_000),
        };
        // Topic still active for ETH: machine drop only, no wire message.
        let sub = FeedSub::new(RtdsSource::Binance, Asset::Btc);
        assert_eq!(
            p.on_command(FeedCommand::Unsubscribe(sub)),
            vec![CommandAction::Untrack(sub)]
        );
        // Last stream on the topic: untrack + topic-level unsubscribe.
        let last = FeedSub::new(RtdsSource::Binance, Asset::Eth);
        assert_eq!(
            p.on_command(FeedCommand::Unsubscribe(last)),
            vec![
                CommandAction::Untrack(last),
                CommandAction::Send(stream_unsubscribe_message(RtdsSource::Binance)),
            ]
        );
        // Unknown stream: no-op.
        assert!(p.on_command(FeedCommand::Unsubscribe(sub)).is_empty());
    }
}
