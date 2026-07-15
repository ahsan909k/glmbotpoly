# Telonex API — notes for the trial pull / validation stage

Reference for `model_lab/io/telonex.py`, `model_lab/telonex_ingest.py`, and
`model_lab/telonex_validate.py`. Everything here was read from the live docs
(`telonex.io/docs/*`, rendered in a browser — the pages are a client-side SPA, so
`WebFetch`/`.md` only returned the nav shell) and confirmed with three read-only,
no-key, zero-credit probes of the public availability endpoint (see bottom).

Vendor: **Telonex** — historical prediction-market + Binance spot data, delivered as
daily **Apache Parquet** files via a REST API. Docs index: `https://telonex.io/llms.txt`.

---

## Auth & base URL

- Base URL: **`https://api.telonex.io/v1`**.
- Auth: **`Authorization: Bearer <API_KEY>`** on the download endpoints. The
  availability and dataset (catalog) endpoints are **public — no key**.
- Our key lives in repo-root **`.env`** under the variable **`telonexdata`**
  (`telonexdata=<value>`). Read it via `os.environ["telonexdata"]`, falling back to
  parsing `.env`. **Never print it, never commit it, never put it in a manifest/log or
  the pre-signed URL trace.** (`.env`, `.env.*`, and `data/` are already git-ignored.)

## Endpoints

### Download (auth) — the only credit-consuming call
```
GET /v1/downloads/{exchange}/{channel}/{date}?<identifier>
Authorization: Bearer <key>
```
- `date` = `YYYY-MM-DD` (UTC). One file **per asset per day** ("each file contains all
  data for that asset on that date").
- **Returns HTTP 302** → a pre-signed **S3** URL (`Location` header, expires 15 min).
  urllib/requests follow it automatically **but** — to avoid S3 rejecting a doubled auth
  mechanism — we do it in two steps: GET the endpoint **without following** (read
  `Location` + `X-Downloads-Remaining`), then stream the S3 URL **with no Authorization
  header**. Stream to disk (never buffer whole files — 16 GB box, prior OOMs).
- Errors: **400** bad params · **403** download limit exceeded (body carries
  `X-Downloads-Remaining`) · **404** no data for those params · **422** invalid
  channel/date.
- Identifier (one of): `asset_id` · `market_id`+`outcome` · `market_id`+`outcome_id` ·
  `slug`+`outcome` · `slug`+`outcome_id`. **We download by `slug`+`outcome`** (no asset_id
  needed to download). `outcome` is `Up`/`Down` for crypto up/down; `outcome_id` 0/1.

### Availability (PUBLIC, no key, no credit)
```
GET /v1/availability/{exchange}?<identifier>
→ {exchange, asset_id, market_id, slug, outcome, outcome_id,
   channels: {<channel>: {from_date, to_date}}}
```
`to_date` is **exclusive** (the day AFTER the last available date). Only channels with
data appear. Resolves any identifier → the full asset info (so it also gives us the
Up-token `asset_id`, which we cross-check against our journal). This is how we prune a
day's 288 candidate slugs to the ones that actually have data — for free.

### Markets dataset / catalog (PUBLIC)
```
GET /v1/datasets/polymarket/markets   → one parquet (all markets' metadata)
```
Per market: `market_id`(on-chain condition id), `slug`, `event_slug`, `question`,
`category`, `tags`, `outcome_0/1`, `asset_id_0/1`, `status`
(`unopened|active|closed|resolved`), `result_id`, `start_date_us`/`end_date_us`/
`created_at_us` (**microseconds** UTC), and **per-channel availability**:
`trades_from/to`, `quotes_from/to`, `book_snapshot_{5,25,full}_from/to`,
`onchain_fills_from/to` (dates `YYYY-MM-DD`, `_to` **exclusive**). Updated daily. Read it
column-projected / row-group-streamed (it is the whole Polymarket catalog — large).
> **CAUTION (from the docs):** *some markets have no data at all — created on-chain but
> never went live on the Polymarket frontend (**e.g. 5M crypto markets**). Filter on
> non-empty availability.* Our **live-traded** windows DO have data (verified below); this
> caution is about the many unopened windows.

## Channels

- **Polymarket:** `trades, quotes, book_snapshot_5, book_snapshot_25, book_snapshot_full,
  onchain_fills, crypto_prices`. (`crypto_prices` = Polymarket-hosted Chainlink feeds,
  single-asset `asset_id=btcusd` — a bonus for future strike reconstruction, out of trial
  scope.)
- **Binance:** `trades, quotes, book_snapshot_5, book_snapshot_25` — **NO `_full`; max 25
  levels.**
- **Densest book:** Polymarket = `book_snapshot_full`; Binance = `book_snapshot_25`.

## Identifiers / naming

- **Crypto up/down windows** are slugged **`{asset}-updown-{dur}-{open_epoch_secs}`** with
  outcome `Up`/`Down` — e.g. `btc-updown-5m-1783252800`, `eth-updown-5m-…`,
  `xrp-updown-15m-1764246600`. The trailing integer is the window **open time in seconds**
  (our journal `event_slug` is identical; our `window.open_time` is the same value in ms).
  A 5m day ⇒ 288 windows, each a distinct asset.
- **Binance:** `slug = market_id = asset_id = <lowercase symbol>` (e.g. `btcusdt`); no
  outcome. Instruments: `btcusdt, ethusdt, solusdt, xrpusdt`. Binance data starts
  **2026-02-06**; collected with `timeUnit=MICROSECOND` (`@trade` → trades;
  `@depth@100ms` → quotes/book_snapshot_5/25; `@bookTicker` deliberately NOT used — no
  timestamp).

## Schemas (all prices/sizes are decimal **strings**; convert to float)

**Timestamps everywhere:** `timestamp_us` (int64, **microseconds**, exchange time) and
`local_timestamp_us` (int64 µs, collector receipt). **Same field names + µs epoch across
Polymarket AND Binance** → internal clock convention is shared (Stage-3 check 5a).

- **trades:** `timestamp_us, local_timestamp_us, exchange, market_id, slug, asset_id,
  outcome, price, size, side`(buy/sell = aggressor), `trade_id, origin_asset_id`.
  `origin_asset_id` differs from `asset_id` when the row was derived via **sibling
  mirroring** (Polymarket Up↔Down) — matters for dedup + the pair/mirror check.
- **book_snapshot_5 / _25 (flattened):** `timestamp_us, local_timestamp_us, exchange,
  market_id, slug, asset_id, outcome, bid_price_0, bid_size_0, …, ask_price_0, ask_size_0,
  …` (level 0 = best). `quotes` = level 0 of a book.
- **book_snapshot_full (nested):** same header + `bids`/`asks` = lists of
  `{price, size}` objects (unlimited depth).
- **Cadence is EVENT-DRIVEN, not interval-sampled.** `book_snapshot_full` = a row on every
  book event (tick-by-tick, all levels). `_25`/`_5` = a row only when the top-N changes.
  ⇒ cadence must be measured **per window** (each 5m slug lives ~5 min; concatenating the
  day fakes huge inter-window gaps), and quiet windows gap legitimately.

## Pricing / quotas

- Free = **5 downloads total**; **Single-Exchange $99/mo = unlimited (one exchange)**;
  **Pro $199/mo = unlimited, all exchanges incl. Binance**; Enterprise = custom.
- We need Polymarket **and** Binance ⇒ **Pro ⇒ unlimited downloads**. No documented rate/
  concurrency limit. Verify our plan at runtime via `X-Downloads-Remaining` / a 403 (a
  free-tier key would 403 after 5 files — the ingest stage aborts loudly on the first 403).
- "Daily updates within hours of midnight UTC" — the most recent day may 404 briefly.

---

## Our own data & the cross-check join (Stage-3 check 5b)

- Journal `data/journal/journal-*.jsonl.gz`: **2026-06-22 → 2026-07-07**
  (`price_tick` Binance ticks, CLOB `top_of_book`/`book`). Depth
  `data/depth/binance-depth20-*.jsonl.gz`: **2026-07-03 → 2026-07-07**.
- **Trial day = 2026-07-05** — a complete UTC day inside journal∩depth overlap and present
  on Telonex (probes below).
- **Exact join keys** (Telonex ↔ our journal): `asset_id` ↔ `token_id`;
  `market_id` ↔ `condition_id`; slug `btc-updown-5m-{secs}` ↔ `event_slug`;
  `(series, open_time_ms)` with `open_time_ms = secs * 1000`.
  Binance `trades.timestamp_us / 1000` ↔ our `price_tick` (`source=BinanceDirect`,
  `kind=Trade`) `ts_exchange` (both real Binance exchange time). Polymarket book/quotes top
  `timestamp_us / 1000` ↔ our `top_of_book` `top.ts` for the same Up token.

## Live read-only probes (public availability — 2026-07-08, zero credits)

1. `btc-updown-5m-1782131400` (2026-06-22 window) Up →
   `asset_id 108533…628`, `market_id 0x9204bd50…`, channels `book_snapshot_full/25/5`
   `2026-06-21→06-23`, `quotes`, `trades`+`onchain_fills` `06-22→06-23`.
   **`asset_id` and `market_id` match our journal's `token_id` and `condition_id`
   byte-for-byte** — the cross-check join is exact.
2. `btc-updown-5m-1783252800` (2026-07-05 12:00 UTC) Up → `book_snapshot_full` +
   `trades` present for 07-05 (`asset_id 105163…880`, `market_id 0xcd9233…`).
3. `btcusdt` (Binance) → all channels `2026-02-06 → 2026-07-07` (`_to` exclusive ⇒ last
   day 2026-07-06).

**Conclusion:** live-traded `btc-updown-5m` windows ARE captured with full book depth;
the trial day is viable; the join to our own recordings is exact.
