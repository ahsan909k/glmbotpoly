# GTD rule-change exposure audit

**Regime change:** Polymarket raised the GTD minimum expiration from **1 min → 3 min** on **2026-07-07 14:00 UTC**. A GTD order whose expiration is less than `now + 180 s` is now rejected at the API level.

**Scope:** read-only audit of the Rust workspace order-placement paths (`venue-api`, `venue-live`, `venue-paper`, `engine`, `config`). No code changed.

**Working date:** 2026-07-09.

---

## Verdict (headline)

**Zero live exposure today. No production order path emits a GTD order.** The three order-producing engine paths use GTC (maker) and FAK (both takers), none of which are affected by the GTD rule.

GTD is fully *plumbed* (type, normalizer, live converter, SDK mapping, paper simulator) but **unused** — it is constructed only in unit tests and in the `bot live` dry-run demonstration, which builds+signs but never posts. So nothing reaches the venue to be rejected.

The **latent** risk is that the plumbing is now stale: the live converter and paper simulator both floor GTD to ~60 s, which the new 180 s venue rule would reject. If a GTD path is ever wired in, it would (a) be rejected live and (b) diverge from paper. This must be fixed *before* any GTD adoption, not now.

---

## 1. Every order-placement path, with order type

| Path | Order class emitted | Expiration math | Affected by 3-min GTD rule? |
|---|---|---|---|
| **Maker quoting** (passive two-sided ladders) | `TimeInForce::Gtc { post_only: true }` | none (GTC rests until cancelled) | **No** |
| **Momentum taker** | `TimeInForce::Fak` | none (marketable, immediate-or-cancel) | **No** |
| **Late-window certainty taker** | `TimeInForce::Fak` | none (marketable, immediate-or-cancel) | **No** |
| GTD (any) | *not emitted in production* | — | n/a |

### Exact lines

**Maker quotes — the single point that pins the passive class to GTC post-only:**

`crates/engine/src/quoting.rs:206-218`
```rust
    /// A resting post-only BUY quote level — the only kind the passive
    /// calculator emits. Pins the `side == Buy` and `tif == Gtc { post_only:
    /// true }` invariant in one place.
    pub fn resting_buy(outcome: Outcome, price: Price, size: Size, level: u32) -> Self {
        Self {
            outcome,
            side: Side::Buy,
            ...
            tif: TimeInForce::Gtc { post_only: true },
        }
    }
```
The quote manager and normalizer pass this TIF through unchanged (`quote_manager/plan.rs:239` `tif: ql.tif`, `quote_manager/driver.rs:502` `tif: d.tif`, `normalize.rs:255` `tif: draft.tif`). Quote lifetime is managed by the **cancel-first repricing reflex** and **final-seconds cancel-all** (§8/§11) — there is **no GTD auto-expiry** in the maker path.

**Momentum taker — marketable FAK BUY:**

`crates/engine/src/taker/driver.rs:403-412`
```rust
        let draft = OrderDraft {
            client_id: Some(format!("mt:{open_ms}:{seq}")),
            ...
            side: Side::Buy,
            price: decision.plan.worst_price.as_decimal(),
            qty: OrderQty::Notional(decision.plan.notional),
            tif: TimeInForce::Fak,
        };
```

**Late-window taker — marketable FAK BUY:**

`crates/engine/src/late_window/driver.rs:394-403`
```rust
        let draft = OrderDraft {
            client_id: Some(format!("lw:{open_ms}:{seq}")),
            ...
            side: Side::Buy,
            price: decision.plan.worst_price.as_decimal(),
            qty: OrderQty::Notional(decision.plan.notional),
            tif: TimeInForce::Fak,
        };
```

**GTD type definition (plumbed, with the venue-threshold contract documented on it):**

`crates/core-types/src/order.rs:46-57`
```rust
    /// Good-til-date limit order. `expires_at` is the desired expiration the
    /// engine reasons about; the venue's 60-second security threshold ...
    /// is applied **inside the venue adapter**, which floors the on-wire
    /// expiration at `now + 60s`. ...
    Gtd {
        expires_at: TimestampMs,
        post_only: bool,
    },
```
> Note: this doc comment (and CLAUDE.md §7) still say **60 s** — now stale vs the 180 s venue rule.

**Every other `Gtd` reference in the tree is a test or the dry-run demo, never a posted production order:**
- `crates/venue-live/src/venue.rs:421-443` — `#[cfg(test)]` dry-run test.
- `crates/bot/src/live.rs:129-135` — `bot live` **dry-run** table entry ("GTD post-only SELL … floored to ≥ now+60s"); `LiveVenue::dry_run` builds+signs but **never POSTs** (CLAUDE.md §11 gate 3 is hardwired false).
- `crates/engine/src/risk/state.rs:188` — test helper (uses GTC anyway).
- `crates/bot/tests/venue_parity.rs`, `crates/venue-live/src/convert.rs` tests, `crates/venue-paper/src/engine.rs` tests — unit tests.

---

## 2. Would each path be rejected under the 3-min rule?

- **Maker (GTC):** No. GTC carries no expiration; it rests until we cancel it. Unaffected.
- **Momentum (FAK) / Late-window (FAK):** No. FOK/FAK are marketable immediate-or-cancel with no expiration timestamp; the GTD minimum does not apply.
- **GTD:** Would be rejected *if it were ever emitted*, because the adapter floors to `now + 65 s` (see §3), which is below the new `now + 180 s` minimum. But no production path emits GTD, so this rejection cannot occur today.

**Late-window / final-seconds check (explicitly requested):** the late-window taker uses FAK, and the final-seconds invariant is *cancel-all*, not a resting expiry — so there is no near-window-end GTD that the 3-min rule could newly reject. In fact the rule change **strengthens** the case for the current GTC-only design: on BTC/ETH 5 m and 15 m windows, once you are within 3 minutes of window close you could **not create any GTD at all** (its natural expiry at window-end would be < 3 min away → rejected). GTD is now effectively unusable for the tail of short windows; GTC + cancel-first sidesteps this entirely.

**Crash-safety auto-expiry:** there is **none** based on GTD. Resting quotes are GTC and rely on `cancel-all` safety paths (on WS disconnect, feed staleness > 500 ms, engine restart, clock skew — §11) plus final-seconds cancel-all. The 3-min rule does not touch this. (Corollary: we have no GTD dead-man's-switch, so a hard crash between cancel-all triggers leaves GTC orders resting — an *existing* design property, unchanged by this event.)

---

## 3. Where the stale 60-s floor lives (the latent fix sites)

**Live adapter — floors GTD to `now + 65 s`:**

`crates/venue-live/src/convert.rs:15-21, 117-129`
```rust
const GTD_MIN_LEAD: DurationMs = DurationMs::from_secs(60);
const GTD_SAFETY_MARGIN: DurationMs = DurationMs::from_secs(5);
...
fn floor_gtd_expiration(expires_at: TimestampMs, now: TimestampMs) -> TimestampMs {
    let min = now.saturating_add(GTD_MIN_LEAD).saturating_add(GTD_SAFETY_MARGIN);
    if expires_at.as_millis() < min.as_millis() { min } else { expires_at }
}
```
Result: a too-soon GTD is floored to `now + 65 s`. Under the old rule the venue accepted this (65 > 60). **Under the new rule the venue rejects it (65 < 180).**

SDK mapping is otherwise correct — `crates/venue-live/src/sdk.rs:146-150` maps `Some(expiration) → OrderType::GTD`, `None → OrderType::GTC`.

**Paper simulator — floors GTD to `now + 60 s` and *accepts* (never rejects for being too-soon):**

`crates/venue-paper/src/engine.rs:43-44, 282-283, 885-893`
```rust
const GTD_MIN_LEAD: DurationMs = DurationMs::from_millis(60_000);
...
TimeInForce::Gtd { expires_at, .. } => Some(floor_gtd(expires_at, now)),
...
fn floor_gtd(expires_at: TimestampMs, now: TimestampMs) -> TimestampMs {
    let min = now.saturating_add(GTD_MIN_LEAD);
    if expires_at.as_millis() < min.as_millis() { min } else { expires_at }
}
```

**Two problems here (both latent, both violate a stated invariant if GTD is used):**
1. **Paper ≠ live divergence (CLAUDE.md §2.5 / §9).** Paper floors to 60 s and would *fill or expire* the order; live now *rejects* it. Paper is thus **more optimistic than reality** — a direct §9 violation ("paper must never be more optimistic than reality in any code path"). Paper does not model the GTD too-soon rejection at all today.
2. **Floor value mismatch.** Live floors to 60 s **+ 5 s margin**; paper floors to 60 s with no margin. Even pre-change these disagreed by the safety margin.

---

## 4. Recommended minimal fix (recommendation only — nothing changed)

Because there is **no live exposure today**, this is a *hygiene / future-proofing* fix, not an incident response. Do it before any GTD path is wired.

**Preferred: keep GTC + cancel-first; do not adopt GTD.** The engine already achieves resting-quote lifetime management through cancel-first repricing and the §11 cancel-all safety mesh, and the 3-min minimum makes GTD unusable for the tail of 5 m/15 m windows anyway. The cheapest correct action is to leave the maker path on GTC and treat GTD as unsupported.

**If GTD support is kept in the plumbing (recommended, since the mapping/tests exist), make it correct under the new rule — a single-constant change on each side plus doc updates:**
1. `crates/venue-live/src/convert.rs` — raise `GTD_MIN_LEAD` from `60 s` to **`180 s`** (keep `GTD_SAFETY_MARGIN` 5 s ⇒ on-wire floor `now + 185 s`).
2. `crates/venue-paper/src/engine.rs` — raise `GTD_MIN_LEAD` from `60_000 ms` to **`180_000 ms`**, and add the **+5 s margin** so paper matches live exactly. Better still, make paper **reject** (not accept) a GTD whose *desired* expiry is below the minimum, mirroring the live venue's new behavior, so paper stops being more optimistic than live (§9). Minimum bar: the floors must match live.
3. Update the stale doc comment in `crates/core-types/src/order.rs:46-57` (says "60-second security threshold" / "now + 60s") and CLAUDE.md §7 ("GTD has a 60-second security threshold … now + 60 + N") to the 3-min figure.
4. Update the affected unit tests (`convert.rs` `gtd_expiration_floored_to_now_plus_65s_when_too_soon`, `venue-paper` `gtd_expiry_is_floored_to_sixty_seconds`, `bot/src/live.rs` dry-run label, `venue.rs` dry-run assertion) to the new floor.

Optionally, factor the GTD minimum into a single config/const shared by both adapters so live and paper can never drift again.

**Do NOT** switch the taker paths (FAK) or the maker path (GTC) — they are unaffected and correct as-is.

---

## 5. Files inspected

`crates/core-types/src/order.rs` · `crates/venue-live/src/convert.rs` · `crates/venue-live/src/sdk.rs` · `crates/venue-live/src/venue.rs` · `crates/bot/src/live.rs` · `crates/venue-paper/src/engine.rs` · `crates/engine/src/quoting.rs` · `crates/engine/src/quote_manager/{plan,driver}.rs` · `crates/engine/src/taker/driver.rs` · `crates/engine/src/late_window/driver.rs` · `crates/engine/src/normalize.rs` · `crates/engine/src/risk/state.rs`.
