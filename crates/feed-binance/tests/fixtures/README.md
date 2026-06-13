# feed-binance test fixtures

## Provenance

Captured live from `wss://stream.binance.com:9443` on **2026-06-12** (06:55 UTC,
operator dev PC) with:

```
cargo run -p bot -- feed --source binance --raw data/captures/binance-session.jsonl
```

`TapTransport` records every in/out application frame verbatim; the JSONL
wrapper is `{"ts_local_ms":…,"dir":"in"|"out","text":…}`, stamped at write
time. Protocol ping/pong frames are deliberately not tapped (transport-level
noise; the server's 20-second pings were answered automatically by
tungstenite throughout the session).

The full session ran ~3.5 minutes and captured 9 992 frames — **all inbound**
(5 147 btcusdt@bookTicker, 1 886 btcusdt@trade, 2 151 ethusdt@bookTicker,
808 ethusdt@trade). Zero outbound frames: subscription rides the combined
connect URL (`/stream?streams=btcusdt@bookTicker/btcusdt@trade/
ethusdt@bookTicker/ethusdt@trade`), confirmed live — the client never sends
an application frame.

## Files

| File | What it pins |
|---|---|
| `binance_session.jsonl` | First 400 frames of the capture verbatim (all four streams represented). Session-replay tests parse every frame and pin the zero-outbound invariant. |
| `binance_bookticker_btcusdt.json` | One verbatim wrapped bookTicker frame. Confirms live what the docs say: payload is `{u,s,b,B,a,A}` — **no event-type `e`, no event-time `E`**. |
| `binance_bookticker_ethusdt.json` | Same, ETH. |
| `binance_trade_btcusdt.json` | One verbatim wrapped trade frame: `{e:"trade",E,s,t,p,q,T,m,M}` with both `E` (event) and `T` (trade) millisecond timestamps. |
| `binance_trade_ethusdt.json` | Same, ETH. |
| `malformed/` | **Hand-crafted** (not captured — a healthy session contains no malformed frames): truncations, unknown streams/symbols/event types, missing fields, zero-priced books, wrapper/payload disagreements, the documented error-response shape, junk. The corpus test asserts every file degrades to `Ignored`, never a panic or publish. |

## Live quirks confirmed by this capture

- The combined-stream wrapper is exactly `{"stream":"<name>","data":{…}}`.
- bookTicker really lacks any timestamp field on the live wire — the crate
  stamps those ticks `ts_exchange := ts_local` (see `wire.rs` module docs).
- Payload symbols arrive uppercase (`BTCUSDT`), stream names lowercase.
- No ack of any kind arrives after a URL-based subscribe — data just flows
  (the first frame arrived ~300 ms after the WS handshake).
