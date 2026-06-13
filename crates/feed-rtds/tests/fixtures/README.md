# RTDS wire fixtures

Real frames captured from `wss://ws-live-data.polymarket.com` on
**2026-06-12** via `bot feed --raw data/captures/rtds-session.jsonl` (the
`TapTransport` records every in/out frame verbatim; the JSONL wrapper is
`{"ts_local_ms":…,"dir":"in"|"out","text":…}` with `text` holding the raw
frame). Consumed by `tests/fixtures.rs`.

| File | What it is |
|---|---|
| `rtds_binance_btcusdt_update.json` | Live `crypto_prices` update (note `full_accuracy_value` as plain decimal string, plus the undocumented `connection_id`) |
| `rtds_binance_ethusdt_update.json` | Same, ETH |
| `rtds_chainlink_ethusd_update.json` | Live `crypto_prices_chainlink` update — `full_accuracy_value` is an 18-dp fixed-point integer string (value × 10¹⁸) |
| `rtds_chainlink_btcusd_update.json` | Same, BTC (captured in the acceptance run) |
| `rtds_binance_btcusdt_backfill.json` | The ~2-minute backfill triggered by a filtered subscribe: `payload.data` = array of `{timestamp, value}` points, type `"subscribe"` |
| `rtds_chainlink_btcusd_backfill.json` | Chainlink backfill — **note the WRONG topic** (`crypto_prices` instead of `crypto_prices_chainlink`): stream identity must come from the symbol |
| `rtds_session.jsonl` | The first 50 frames of a session verbatim: 6 out-subscribes (per topic: filtered backfill subscribes, then the unfiltered steady-state subscribe), empty-text acks, all 4 backfills, early live updates, the first out-PING |
| `malformed/*` | Hand-crafted (NOT captured) parser-robustness corpus |

Observed-but-undocumented facts these fixtures pin (see `src/wire.rs` module
docs and the Decisions Log): one filter slot per (connection, topic) with
replace-not-add semantics; backfill only from filtered subscribes; the
wrong-topic chainlink backfill; `full_accuracy_value` formats; no text
`"PONG"` reply observed from RTDS (1556-frame session) — the parser's Pong
arm is defensive.
