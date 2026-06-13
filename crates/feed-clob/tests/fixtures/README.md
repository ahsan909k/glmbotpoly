# feed-clob wire fixtures

Captured from the live Polymarket CLOB market channel
(`wss://ws-subscriptions-clob.polymarket.com/ws/market`) on **2026-06-12**
via `bot ladder --series BTC-5m --raw data/captures/clob.jsonl` (the
TapTransport JSONL capture path), during live BTC-5m windows around
09:00–09:30 UTC. Each `clob_*.json` file is the verbatim `text` of one
inbound frame; `clob_pong.txt` is the server's keepalive reply to our text
`PING`.

- `clob_book.json` — one `book` snapshot (fires on subscribe and after
  every trade that affects the book).
- `clob_array_books.json` — one text frame carrying a JSON **array** of
  `book` events (the on-subscribe delivery for a multi-token subscription);
  array frames are real, the parser must accept both forms.
- `clob_price_change.json` — order place/cancel deltas. Note the mirror
  pattern: one order appears as a change on **both** tokens (BUY Up at p ↔
  SELL Down at 1−p), each entry carrying its own token's post-change
  `best_bid`/`best_ask`.
- `clob_price_change_removal.json` — includes a `size:"0"` level removal.
- `clob_last_trade_price.json` — a trade print.
- `clob_best_bid_ask.json` — venue-reported tops (custom feature flag).
- `clob_new_market.json` — `new_market` is a **platform-wide broadcast**
  (this one is an Italian basketball market that arrived on a BTC-5m
  connection); carries both `market` and `condition_id` (equal).
- `clob_market_resolved.json` — `market_resolved` for our subscribed
  window; the condition id is in `market` (no `condition_id`/`slug` field,
  unlike the docs' field list), winner in `winning_asset_id`.
- `clob_tick_size_change.json` — **synthesized from the docs field list**
  (no tick flip occurred during the capture windows; BTC-5m stayed
  mid-range). Replace with a captured frame when one lands — the event
  needs a window drifting past 0.96/0.04 before close.
- `clob_session.jsonl` — a contiguous ~900-frame slice of the raw capture
  (tap JSONL: `{"ts_local_ms":…,"dir":…,"text":…}`), starting at subscribe
  and spanning deltas, trades, and the trade-driven snapshot replacements.
  Drives the replay test's invariant: the delta-maintained book must equal
  every arriving venue snapshot when no trade intervened, and may never be
  crossed (crossing deltas = trades at the touch, resolved by implied
  consumption).
- `malformed/` — hand-written negative cases for the parser (not captured).

Wire truths these fixtures pin (Decisions Log 2026-06-12): trades reach
books only through fresh `book` snapshots (`price_change` covers order
place/cancel only); a trade at the touch arrives as a crossing delta whose
consumed opposite level is never removed by a delta; prices/sizes/
timestamps are strings; `PONG` answers our `PING`.
