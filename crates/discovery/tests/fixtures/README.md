# Recorded live API responses (CLAUDE.md §12 fixtures)

Captured read-only from the public APIs on **2026-06-11 at 08:31:52 UTC**
(`nowMs = 1781166712388` — tests pin this as `CAPTURE_NOW_MS` so window
classification is deterministic). Files are verbatim response bodies; none
are hand-edited.

| File | Request |
|---|---|
| `gamma_series_btc_5m.json` | `GET https://gamma-api.polymarket.com/series?slug=btc-up-or-down-5m` |
| `gamma_events_btc_5m.json` | `GET https://gamma-api.polymarket.com/events?series_id=10684&closed=false&order=endDate&ascending=true&end_date_min=2026-06-11T08:31:52Z&limit=3` |
| `gamma_events_btc_1h.json` | same, `series_id=10114` (BTC hourly) |
| `gamma_event_stale.json` | `GET https://gamma-api.polymarket.com/events?slug=bitcoin-up-or-down-may-20-2026-6am-et` — an event still `closed=false` weeks after its `endDate` (why `end_date_min` is mandatory); its market also carries a real drifted `orderPriceMinTickSize` of `0.001` and a Binance resolution source |
| `clob_market_btc_5m.json` | `GET https://clob.polymarket.com/markets/0xed902f990ca86222fae9df756181931f55c7cc6d2109b5cd030a43e0e3e13e0f` (the then-current 5m window; note the date-truncated `end_date_iso`) |
| `gamma_event_resolved_5m.json` | `GET https://gamma-api.polymarket.com/events?slug=btc-updown-5m-1781290200` captured **2026-06-12 ~18:59 UTC** (a few minutes after the 18:50–18:55 window resolved). Carries `eventMetadata` `{priceToBeat: 63788.96914518841, finalPrice: 63757.5793174016}` — the post-hoc strike-verification anchor (`finalPrice < priceToBeat ⇒ resolved Down`). Confirms `eventMetadata` is absent during the window and appears only after resolution. |

At capture time the 5m response held the active window
`btc-updown-5m-1781166600` (08:30–08:35 UTC) plus the next two; the hourly
response held `bitcoin-up-or-down-june-11-2026-4am-et` (08:00–09:00 UTC,
event-level `startTime` empty, `market.eventStartTime` populated) plus the
next two.
