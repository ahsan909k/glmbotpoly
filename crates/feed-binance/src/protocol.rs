//! The Binance venue protocol over feed-util's generic driver: fixed stream
//! universe with per-stream-kind staleness thresholds, zero connect messages
//! (subscription rides the URL — also how the 5-inbound-messages/s venue
//! limit is satisfied by construction), and no runtime commands.

use core_types::{DurationMs, TimestampMs};
use feed_util::{CommandAction, FeedProtocol, FrameOutcome, PriceObs};

use crate::sub::{BinanceStream, BinanceSub};
use crate::wire::{BinanceParsed, IgnoredReason, parse_frame};

/// Uninhabited: the Binance feed has no runtime commands (the stream set is
/// fixed in the connect URL).
#[derive(Debug, Clone, Copy)]
pub enum NoCommand {}

/// The Binance venue protocol. One instance lives for the driver's whole
/// life.
pub(crate) struct BinanceProtocol {
    pub(crate) subs: Vec<BinanceSub>,
    /// Staleness threshold for bookTicker streams.
    pub(crate) stale_after: DurationMs,
    /// Staleness threshold for trade streams (much looser — quiet markets
    /// legitimately print nothing for seconds).
    pub(crate) trade_stale_after: DurationMs,
}

impl BinanceProtocol {
    fn stale_after_for(&self, sub: BinanceSub) -> DurationMs {
        match sub.stream {
            BinanceStream::BookTicker => self.stale_after,
            BinanceStream::Trade => self.trade_stale_after,
        }
    }
}

impl FeedProtocol for BinanceProtocol {
    type Key = BinanceSub;
    type Reason = IgnoredReason;
    type Command = NoCommand;

    fn name(&self) -> &'static str {
        "binance"
    }

    fn keys(&self) -> Vec<(BinanceSub, DurationMs)> {
        self.subs
            .iter()
            .map(|&sub| (sub, self.stale_after_for(sub)))
            .collect()
    }

    /// Empty: the combined-stream URL is the subscription, so a reconnect
    /// resubscribes implicitly and the driver never sends a frame.
    fn connect_messages(&self) -> Vec<String> {
        Vec::new()
    }

    fn handle_frame(
        &mut self,
        text: &str,
        now: TimestampMs,
    ) -> FrameOutcome<BinanceSub, IgnoredReason> {
        match parse_frame(text, now) {
            BinanceParsed::Ack => FrameOutcome::Ack,
            BinanceParsed::Prices(updates) => FrameOutcome::Prices(
                updates
                    .into_iter()
                    .map(|update| PriceObs {
                        key: update.sub,
                        value: update.value,
                        ts_exchange: update.ts_exchange,
                    })
                    .collect(),
            ),
            // Under URL subscription nothing unexpected should arrive — every
            // skip is warn-gated (no quiet class, unlike RTDS's all-symbols
            // traffic).
            BinanceParsed::Ignored(reason) => FrameOutcome::Ignored(reason),
        }
    }

    fn on_command(&mut self, command: NoCommand) -> Vec<CommandAction<BinanceSub>> {
        match command {}
    }
}

#[cfg(test)]
mod tests {
    use core_types::Asset;

    use super::*;

    fn protocol() -> BinanceProtocol {
        BinanceProtocol {
            subs: BinanceSub::all(),
            stale_after: DurationMs::from_millis(2_500),
            trade_stale_after: DurationMs::from_millis(30_000),
        }
    }

    #[test]
    fn keys_carry_per_stream_kind_thresholds() {
        let keys = protocol().keys();
        assert_eq!(keys.len(), 4);
        for (sub, stale_after) in keys {
            let expected = match sub.stream {
                BinanceStream::BookTicker => 2_500,
                BinanceStream::Trade => 30_000,
            };
            assert_eq!(stale_after.as_millis(), expected, "{sub}");
        }
    }

    #[test]
    fn no_connect_messages_and_frames_classify() {
        let mut p = protocol();
        assert!(p.connect_messages().is_empty());
        let now = TimestampMs::from_millis(5);
        assert_eq!(
            p.handle_frame(r#"{"result":null,"id":1}"#, now),
            FrameOutcome::Ack
        );
        assert_eq!(
            p.handle_frame("junk", now),
            FrameOutcome::Ignored(IgnoredReason::MalformedJson)
        );
        let frame = r#"{"stream":"btcusdt@bookTicker","data":{"u":1,"s":"BTCUSDT","b":"100.0","B":"1","a":"101.0","A":"1"}}"#;
        let FrameOutcome::Prices(obs) = p.handle_frame(frame, now) else {
            panic!("expected prices");
        };
        assert_eq!(obs.len(), 1);
        assert_eq!(
            obs[0].key,
            BinanceSub::new(Asset::Btc, BinanceStream::BookTicker)
        );
        assert_eq!(obs[0].ts_exchange, now, "bookTicker stamps at now");
    }
}
