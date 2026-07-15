# LIVE_PREFLIGHT.md — Going live (the exact sequence)

> ⚠️ **Real money.** Do not start this until [`ACCEPTANCE.md`](ACCEPTANCE.md) has
> passed and you have chosen the final 2–3 series. The bot is paper-only today;
> this is the documented procedure for the future live phase. Every step is
> docs-driven — follow the linked Polymarket pages, do not improvise endpoints or
> auth.

The bot is **incapable of sending a live order** unless three independent
conditions are all satisfied (§11). This document walks the full setup, the
three-condition arming flow, a conservative first-live configuration, and the
rollback. Polymarket docs base: `https://docs.polymarket.com`.

---

## Step 0 — Prerequisites

- The paper evaluation passed; final series chosen.
- A funded **deposit wallet** on Polygon with USDC collateral.
- The signer's private key for that wallet.
- You have read: `/polymarket-101`, `/trading/overview`, `/trading/quickstart`,
  `/concepts/order-lifecycle`, `/concepts/pusd`.

---

## Step 1 — Deposit wallet + signature type

New API users use a **deposit wallet** (Polymarket signature type **3 =
`Poly1271`**; the bot's `SigType::DepositWallet`, the default). The deposit-wallet
address is the **funder** — the maker/signer of every order.

- Set the public funder address in config (it is **not** a secret, so it lives in
  TOML and is auditable):
  ```toml
  [live]
  enabled = false               # leave false until Step 4
  funder  = "0x<your-40-hex-deposit-wallet-address>"
  ```
  Validated as a `0x`-prefixed 40-hex address (`config/sections/live.rs`); wired to
  the SDK as `funder(address).signature_type(Poly1271)` in `venue-live/src/sdk.rs`.
- Docs: **`/trading/deposit-wallets`** (signature types, funder), `/concepts/pusd`.

---

## Step 2 — On-chain approvals  ⚠️ NOT WIRED IN THE BOT

The exchange contracts need one-time **ERC-20 (USDC) and CTF token approvals**
before any order can fill. **The bot does NOT perform these approvals** — it is a
documented deferred gap (`venue-live/src/lib.rs` "Deferred to follow-up tasks").
A missing allowance surfaces at order time as **`Rejected(InsufficientFunds)`**.

**You must approve collateral on-chain manually before going live**, following:

- **`/trading/deposit-wallets`** (the required approvals for the exchange + CTF
  contracts),
- **`/trading/gasless`** (the relayer path for those approvals — gas-sponsored),
- **`/trading/ctf/overview`**, `/trading/ctf/merge`, `/trading/ctf/redeem` (the
  pair-merge / redemption mechanics the strategy relies on to recycle collateral).

Verify the approvals landed on-chain (a block explorer / the relayer receipt)
**before** arming. This is the single most likely cause of a first-live failure.

### Step 2b — Settlement / redemption parity  ⚠️ NOT WIRED IN THE BOT

**Paper settles instantly at resolution.** When a window resolves, the paper
venue credits **$1.00 per winning share and $0 per losing share**, closes the
positions, and returns the cash to the wallet — synchronously, at the resolution
event (`venue-paper::MatchEngine::on_window` → `PaperWallet::settle_window`). The
dashboard's equity, the two-bucket PnL, and the "open positions" identity all
assume that money is back in the account the moment a window resolves.

**Live must match, or the §9 paper/live parity is violated.** On Polymarket a
resolved position does not become collateral by itself — the winning CTF
outcome tokens must be **redeemed on-chain** (a real transaction: gas, a relayer
round-trip, and confirmation latency; matched Up+Down pairs can also be
**merged**). Until the live adapter performs this automatically:

- resolved positions sit as un-redeemed tokens, so live cash/equity **lags** the
  paper model (which shows the money instantly), and capital does not recycle
  into the next window as the strategy's sizing assumes;
- a failed/dropped redeem transaction must be **retried** — settlement is not a
  fire-and-forget.

**Required before real scale:** the live adapter must **auto-redeem resolved
positions** (and merge matched pairs) with gas handling and retry logic, on the
same trigger paper uses (`market_resolved`), so that live settlement mirrors
paper's instant settlement. This is currently a **deferred gap** — track it with
the on-chain approvals gap (Step 2). Docs: **`/trading/ctf/redeem`**,
`/trading/ctf/merge`, `/trading/gasless`, `/concepts/resolution`.

---

## Step 3 — API credentials (L2)

The authenticated CLOB + user-WebSocket paths need the operator's L2 credentials.
These are **secrets — environment only**, never TOML (`config/secrets.rs`,
consumed in `venue-live/src/sdk.rs`):

```bash
# /etc/bot/bot.env  (mode 0600) — see deploy/bot.env.example
BOT_SECRET_PM_API_KEY=...           # L2 API key (UUID)
BOT_SECRET_PM_API_SECRET=...        # L2 secret
BOT_SECRET_PM_API_PASSPHRASE=...    # L2 passphrase
BOT_SECRET_PM_PRIVATE_KEY=0x...     # signer private key
```

All four must be present or the venue refuses to construct
(`MissingCredentials`). Docs: **`/trading/clients/l2`** (creating/deriving L2
creds), `/trading/clients/public`.

---

## Step 4 — The three-condition arming flow (§11)

Live orders are possible **only** when all three gates pass *and* the four creds
are present. The live adapter (`venue-live::check_arming`) returns `NotArmed`
otherwise and never even constructs a network client.

**Gate 1 — config flag (persistent):**
```toml
[live]
enabled = true
funder  = "0x..."     # required when enabled
```

**Gate 2 — environment confirmation phrase (persistent):**
```bash
BOT_SECRET_LIVE_CONFIRM=arm-live-i-accept-real-money-losses
```
Must **exactly** equal the hardcoded `LIVE_CONFIRM_PHRASE`
(`config/sections/live.rs`). A half-armed config (gate 1 on, phrase wrong/absent)
fails closed at boot.

**Gate 3 — runtime arm (per session, expires):**
```bash
bot control arm-live                                   # begins; 60 s window opens
bot control confirm-arm "arm-live-i-accept-real-money-losses"   # within 60 s
# bot control status                                   # verify session_armed=true
```
`PENDING_TTL_MS = 60_000` (`bot/src/control.rs`). If the phrase is wrong, the
window expired, or a boot gate regressed, `confirm-arm` is rejected.
Dashboard equivalent: `POST /api/control/arm-live/begin` → `…/confirm` (body
`{"phrase":"…"}`); `GET /api/control/status` shows `config_enabled` /
`env_confirmed` / `pending` / `session_armed`.

Only with **gate1 ∧ gate2 ∧ gate3 ∧ all-4-creds** does `LiveVenue::connect`
build a network-capable, order-posting client.

---

## Step 5 — First-live configuration: tiny caps, takers OFF

Start as small and as passive as the venue allows. Override in
`/etc/bot/config/bot.local.toml`:

```toml
[engine.defaults]
# THE binding per-window sizing constraint (§8). Start at the floor.
max_worst_case_loss_per_window = 1          # default 25 — start tiny
# Disable BOTH taker modules. They are hardcoded enabled in run.rs::risk_params
# (momentum_enabled / late_window_enabled = true, no config key), so a zero
# shared budget is the off switch — it makes every take fail the budget gate.
taker_budget_per_window = 0                  # default 10 -> 0 = takers off

[risk]
max_open_notional = 25                        # default 1000 — small global cap
daily_stop_loss   = 10                         # default 200 — halt fast on a bad day
```

Restrict to the chosen series only (config `[engine.series.<name>].enabled` or
`bot run --series <one>` to start with a single series). Watch the dashboard's
risk panel and fills blotter for the first windows before scaling any cap.

---

## Step 5b — Self-match / self-trade prevention  ✅ IMPLEMENTED — needs live-mode verification

**Client-side self-trade prevention is now wired for ALL taker paths** (momentum,
late-window, model — 2026-07-13). Before any taker walks the book for a FAK BUY on
outcome O, `engine::self_match::filter_asks` removes or shrinks ask levels that are
our *own* §7-mirrored resting liquidity — i.e. a live resting BUY on `O.opposite()`
at the complement price `1 − p` (which surfaces as an ask on O at price p). A level
fully covered by our own size is skipped; a partially-covered one is shrunk to the
external remainder. The `RestingView` is threaded from the quote manager into every
taker's `decide` by the `RiskManager` fan-out.

**Still needs live verification:** paper venues cannot reproduce the collision (a
paper FAK never matches our own disjoint resting paper orders), so this guard is
proven only by unit tests (`engine::self_match` + `bot/tests/model_taker_paper.rs`:
a synthetic book containing our mirrored quote → the level is skipped/shrunk) until
it runs live. On the first live session, confirm no self-cross fills occur (watch
for taker fills against our own order ids and the venue's server-side STP behavior).

**Original verdict (from the code audit that motivated the fix, 2026-07-13):**

- **Paper venue — NO, structurally impossible.** A marketable FAK walks only the
  feed-fed real book (`venue-paper::SimBook.asks`, `engine.rs::activate_marketable`
  → `book.rs::walk_marketable`); our own resting paper orders live in a disjoint
  `orders` map and are never part of that depth. Resting orders fill only from an
  external `Event::LastTrade` with an opposite aggressor. So **paper backtest/paper
  PnL is free of self-trade artifacts** — but paper also gives you **no** warning of
  this live risk.
- **Live engine — NO safeguard exists; a self-crossing taker CAN be sent.** The
  takers hold no reference to our resting quotes (no `RestingView`), the risk
  gateway checks only halt / per-window loss / open-notional (`risk/guard.rs`), and
  there is no `self_trade` / `self_match` / `wash` logic anywhere. Two paths differ:
  - The **momentum taker is incidentally protected**: its edge gate refuses any ask
    at/above fair (`taker/edge.rs`, `BookAlreadyRepriced`), and our maker quotes rest
    on the far side of fair — so it will not lift them *as long as pricing stays
    conservative*.
  - The **late-window certainty taker has NO edge gate** (`late_window/driver.rs`):
    it buys any ask ≤ `late_taker_price_cap` (0.99) once the outcome is deemed
    certain. Our deep resting BUY (e.g. BUY Up @ 0.05 ⇄ Down ask @ 0.95 ≤ cap) is
    exactly such an ask. **This is the exposed path.**

  Whether a self-cross then actually *trades* is left entirely to Polymarket's
  server-side matching / self-trade-prevention, which our code neither invokes nor
  documents (`venue-live` is a passive mirror). A self-trade wastes taker fees on
  our own liquidity and distorts inventory/markout attribution.

**Prevention as implemented (2026-07-13):**

1. ✅ The taker modules now receive the quote manager's `RestingView` (threaded from
   the `RiskManager` fan-out into each taker's `decide`).
2. ✅ Before a FAK BUY on outcome O, `engine::self_match::filter_asks` excludes/shrinks
   ask levels matching our own §7-mirrored resting liquidity (skip-or-shrink, applied
   to momentum, late-window, and model takers). This is a **skip**, not a
   cancel-own-quote-first — a conflicting resting quote is left in place and simply
   not lifted.
3. ⏳ A self-cross veto in the risk gateway (defense-in-depth) is **not** added — the
   per-taker filter is the single guard. Consider a gateway veto if a future taker is
   added that does not route through `filter_asks`.

The guard is best-effort (it excludes our *known* resting liquidity at the exact
mirror price; it does not cancel-first) and is proven only by unit tests until live —
so at the first live session with `taker_budget_per_window > 0`, verify no self-cross
fills occur before scaling.

---

## Step 6 — Rollback (fastest → most thorough)

1. **Disarm the session** (instant, keeps the process up):
   ```bash
   bot control disarm
   ```
2. **Emergency kill** (cancel-all + global halt, journaled):
   ```bash
   bot control kill          # then `bot control reset` to resume after investigating
   ```
3. **Disable in config + restart** (fails gate 1 closed):
   ```toml
   [live]
   enabled = false
   ```
   then `systemctl restart bot`.
4. **Full stop** (graceful drain: stop strategies → cancel-all → flush journal):
   ```bash
   systemctl stop bot        # or Ctrl-C / kill -TERM
   ```

---

## Step 7 — Week 1 live: reconcile OUR profile P/L against OUR cash ledger

Within the first week live, **reconcile our own Polymarket profile P/L (`user-pnl-api`)
against our own cash ledger** — the journal's fills + merges/redeems + `TAKER_REBATE`
credits. This settles, with **ground truth we control**, the P/L-basis question the
competitor study could only flag from the outside.

- Our journal records every fill, merge, and redeem, and — now that `TAKER_REBATE` is
  captured (`competitors/fetch` `ACTIVITY_TYPES`) — the daily fee-rebate credits too, so we
  can compute our **exact realized cash**.
- Compare that to our profile's `user-pnl-api` figure. The competitor audit
  (`model-lab/out/audits/takerner_pnl_anomaly.md`) found the profile P/L for hold-to-resolution
  takers does **not** equal raw realized cash (near equal-and-opposite for takerner) — its
  display basis is unverified. Reconciling OUR OWN account resolves whether that is a
  marking / cost-basis convention or a genuine discrepancy.
- If they diverge, the dashboard equity (cash-basis, §9) and the profile P/L are measuring
  **different things** — decide and document which is authoritative for the go/no-go P/L
  **before scaling caps**.

---

## Reference — Polymarket docs map for going live

| Area | Pages |
|---|---|
| Orientation | `/polymarket-101`, `/quickstart`, `/concepts/order-lifecycle`, `/concepts/positions-tokens`, `/concepts/pusd`, `/concepts/resolution` |
| Wallets & approvals | **`/trading/deposit-wallets`**, **`/trading/gasless`** |
| Auth & clients | **`/trading/clients/l2`**, `/trading/clients/public`, `/trading/clients/l1` |
| Orders | `/trading/orders/overview`, `/trading/orders/create`, `/trading/orders/cancel`, `/trading/matching-engine` |
| On-chain ops | **`/trading/ctf/overview`**, `/trading/ctf/merge`, `/trading/ctf/redeem`, `/resources/contracts` |
| Fees & rebates | `/trading/fees`, `/trading/taker-rebates`, `/market-makers/maker-rebates` |
| Errors | `/resources/error-codes` |

> Open gaps to close before real scale:
> 1. Automate Step 2 (ERC-20/CTF approvals via the gasless relayer) inside the bot
>    so a fresh deposit wallet can self-provision. Until then, Step 2 is a manual,
>    verify-on-chain prerequisite.
> 2. Auto-redeem resolved positions and merge matched pairs on-chain (Step 2b), so
>    live settlement matches paper's instant settlement at resolution and the §9
>    parity holds. Until then, live cash/equity lags the paper model after every
>    resolution.
> 3. **Self-match / self-trade prevention (Step 5b).** ✅ IMPLEMENTED for all taker
>    paths (momentum, late-window, model) via `engine::self_match::filter_asks` —
>    each excludes/shrinks our own §7-mirrored resting liquidity before a FAK.
>    **Needs live-mode verification:** paper can never reproduce the collision, so
>    the guard is proven only by unit tests until it runs live.
