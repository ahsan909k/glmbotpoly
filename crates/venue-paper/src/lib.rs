//! Paper execution adapter implementing the `venue-api` port against real
//! live market data (CLAUDE.md §9): a conservative fill simulator that queues
//! passive orders behind all displayed size at placement, fills only on real
//! trade prints at or through the price after simulated latencies, and walks
//! real displayed depth for marketable orders; plus the paper wallet/ledger
//! with runtime-adjustable starting capital, exact taker-fee math from live
//! per-market fee params, estimated maker rebates on a simulated daily cycle,
//! and instant simulated pair-merge. It must be impossible for paper to be
//! more optimistic than reality in any code path — when in doubt, fill less.
