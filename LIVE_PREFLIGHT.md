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

> Open gap to close before real scale: automate Step 2 (ERC-20/CTF approvals via
> the gasless relayer) inside the bot so a fresh deposit wallet can self-provision.
> Until then, Step 2 is a manual, verify-on-chain prerequisite.
