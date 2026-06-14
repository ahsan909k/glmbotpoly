# `venue-live` user-channel fixtures

Recorded/representative authenticated `/ws/user` frames, consumed by the
fixture-driven tests in `src/user_wire.rs` (parsing) and `src/store.rs`
(full-lifecycle application through the canonical `OrderStore`).

## Provenance

**Synthesized from the docs field list (`market-data/websocket/user-channel`,
verified 2026-06-13) and independently cross-checked against the
`polymarket_client_sdk_v2` 0.5.1 `clob::ws` response types** (`OrderMessage` /
`TradeMessage` / `MakerOrder`), whose field names match the docs exactly.

Capturing real frames requires armed live credentials and a live order
round-trip (operator-gated, like the `live-smoke` test), so these are documented
shapes — **replace with a real capture when one is taken**, the same precedent as
`feed-clob`'s synthesized `tick_size_change` fixture.

All frames use one condition id
`0x7f3a1c0b9e8d6f5a4c3b2a1908172635445362718091a2b3c4d5e6f708192a3b`, an Up token
`111111111` (order `0xorder-up-1`), and a Down token `222222222`
(order `0xorder-down-1`).

| File | Shape |
|---|---|
| `user_order_placement.json` | `order` / `PLACEMENT`, Up BUY, 0 filled |
| `user_order_update_partial.json` | `order` / `UPDATE`, Up BUY, `size_matched` 8 |
| `user_order_cancellation.json` | `order` / `CANCELLATION`, Up BUY |
| `user_trade_matched_taker.json` | `trade` / `MATCHED`, our order is the taker (5) |
| `user_trade_matched_maker.json` | `trade` / `MATCHED`, our order is a maker (7) |
| `user_trade_failed.json` | `trade` / `FAILED` (must not count as a fill) |
| `user_trade_array.json` | one frame carrying an array: an `order` then a `trade` |
