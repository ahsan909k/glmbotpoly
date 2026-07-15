# Competitor analysis (read-only)

Classifies a set of Polymarket accounts by measurable trading signatures, restricted
to our four series (btc/eth Up-Down 5m & 15m). **Read-only**: it only GETs public
endpoints and never touches the trading bot or its state.

## Run

```bash
# full pipeline (each stage cached + resumable)
python -m model_lab.competitors all

# or stage by stage
python -m model_lab.competitors.resolve      # handles -> data/competitors/handles.json
python -m model_lab.competitors.fetch        # per-account activity/trades/positions (cached)
python -m model_lab.competitors.analyze      # -> out/competitors/analysis.json
python -m model_lab.competitors.analyze_tape # tape cross-check -> out/competitors/tape_validation.json
python -m model_lab.competitors.report       # -> out/competitors/report.html
```

Handles default to the operator's list; override with `--handles ...` / `--only ...`.

## Data sources

- **Gamma** `public-search` / `public-profile` — resolve a handle to a proxy wallet
  (there is no other public username→address path).
- **Data API** `data-api.polymarket.com` — `/activity` (TRADE/SPLIT/MERGE/REDEEM/REWARD),
  `/trades?takerOnly=` (the maker/taker ground truth), `/positions`.
- **Telonex tape** (`data/telonex`) — PM prints (aggressor side) + Binance top-mid, used
  only to cross-check maker/taker and measure reaction speed to Binance moves.

Cache lives under `data/competitors/<addr>/` (git-ignored). Full history for an active
5m market-maker is ~300k events/month, so the cache can reach hundreds of MB per account;
fetches are resumable (`.state.json` per endpoint) and politely rate-limited.

## What is measured

Per account: series coverage %, maker/taker mix, pair discipline (two-sided %, pair-cost
distribution, leg-completion lag), merge-vs-redeem behaviour, within-window fill timing,
$/window, trades/day, active hours, reaction speed to Binance moves, and a **cash-flow
reconstructed** PnL/day + worst drawdown. See the report's honesty/caveats box for the
uncertainty on each. `python -m pytest tests/test_competitors.py` covers the pure logic.
