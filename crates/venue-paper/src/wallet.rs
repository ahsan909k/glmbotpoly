//! The paper wallet: a fill-driven ledger so `balances()` reflects reality
//! (§9 — real fee math, paper money).
//!
//! State: signed collateral cash, signed net per-token positions, cumulative
//! taker fees paid, an accruing maker-rebate estimate, the rebate already
//! credited (a separate income line), and cumulative injected capital (the
//! conservation anchor). Operations: per-fill cash-flow/position update, window
//! settlement on the real `market_resolved` outcome ($1 per winning share),
//! instant pair-merge (matched Up+Down → $1 collateral), a daily rebate credit,
//! and runtime capital set/adjust. Net positions are tracked *signed* (a
//! negative net = a short / the merged complement); the reported [`Wallet`]
//! omits non-positive nets — use [`PaperWallet::positions_view`] for the signed
//! view. Open-order collateral reservation is still deferred (so
//! `collateral_available == collateral_total`).
//!
//! Cost-basis is intentionally **not** tracked here: the task is cash + signed
//! shares, and settlement marks every share at $1/$0, so realized edge falls out
//! of cash with no average-cost bookkeeping. Per-side cost for the §8
//! pair-discipline analytics lives in [`core_types::SideInventory`].
//!
//! ## Conservation (the property the tests pin)
//!
//! With every share marked at its eventual settlement value, no operation
//! creates or destroys value — cash + marked inventory moves only by fees,
//! rebates, and realized edge. Equivalently the identity
//! `cash + inventory_at_settlement == starting_capital + capital_injected −
//! fees_paid + rebate_credited + realized_edge` holds exactly (Decimal) after
//! any sequence of fills, merges, and settlements.

use std::collections::HashMap;

use core_types::{Decimal, Dollars, Fill, Outcome, Side, Size, TokenId};
use venue_api::{TokenBalance, Wallet};

/// A richer ledger view than the venue-agnostic [`Wallet`]: keeps signed
/// positions and exposes the fee/rebate income lines for the dashboard and the
/// `bot paper-sim` smoke. Read via [`PaperWallet::snapshot`].
#[derive(Debug, Clone, PartialEq)]
pub struct PaperLedgerSnapshot {
    /// Signed collateral cash.
    pub collateral: Dollars,
    /// Signed non-zero net positions, sorted by token id.
    pub positions: Vec<(TokenId, Decimal)>,
    /// Cumulative taker fees paid.
    pub fees_paid: Dollars,
    /// Maker rebate accrued, not yet credited.
    pub rebate_accrued: Dollars,
    /// Maker rebate credited to cash (separate income line).
    pub rebate_credited: Dollars,
}

/// A simulated wallet backing the paper venue.
#[derive(Debug, Clone)]
pub struct PaperWallet {
    /// pUSD collateral (cash). Signed: a strategy can overspend in paper.
    collateral: Dollars,
    /// Signed net shares per outcome token (negative = short / merged side).
    positions: HashMap<TokenId, Decimal>,
    /// Running maker-rebate estimate not yet credited (carry bucket).
    rebate_accrued: Dollars,
    /// Cumulative taker fees folded into cash (separate income line).
    fees_paid: Dollars,
    /// Cumulative maker rebate moved accrued→cash by the daily cycle.
    rebate_credited: Dollars,
    /// Cumulative signed capital injected (start + runtime adjustments) — the
    /// conservation anchor, never spent.
    capital_injected: Dollars,
}

impl PaperWallet {
    /// A wallet with `starting_capital` collateral and no positions.
    #[must_use]
    pub fn new(starting_capital: Dollars) -> Self {
        Self {
            collateral: starting_capital,
            positions: HashMap::new(),
            rebate_accrued: Dollars::ZERO,
            fees_paid: Dollars::ZERO,
            rebate_credited: Dollars::ZERO,
            capital_injected: starting_capital,
        }
    }

    /// Sets collateral to `amount` — the dashboard runtime capital-adjust seam
    /// (§9). Records the implied signed delta as injected capital so the
    /// conservation identity still closes.
    pub fn set_capital(&mut self, amount: Dollars) {
        let delta = amount - self.collateral;
        self.collateral = amount;
        self.capital_injected = self.capital_injected + delta;
    }

    /// Adjusts collateral by a signed `delta` (the dashboard +/- capital seam).
    pub fn adjust_capital(&mut self, delta: Dollars) {
        self.collateral = self.collateral + delta;
        self.capital_injected = self.capital_injected + delta;
    }

    /// Applies one fill's cash-flow and position change, accumulates the taker
    /// fee, and accrues the maker-rebate estimate (`rebate_estimate` is zero for
    /// taker fills).
    ///
    /// BUY debits `price × size + fee` and credits shares; SELL credits
    /// `price × size − fee` and debits shares.
    pub fn apply_fill(&mut self, fill: &Fill, rebate_estimate: Dollars) {
        let notional = Dollars::new(fill.price.as_decimal() * fill.size.as_decimal());
        let entry = self.positions.entry(fill.token_id.clone()).or_default();
        match fill.side {
            Side::Buy => {
                self.collateral = self.collateral - notional - fill.fee;
                *entry += fill.size.as_decimal();
            }
            Side::Sell => {
                self.collateral = self.collateral + notional - fill.fee;
                *entry -= fill.size.as_decimal();
            }
        }
        // One line covers both sides — `fill.fee` is zero for maker fills.
        self.fees_paid = self.fees_paid + fill.fee;
        self.rebate_accrued = self.rebate_accrued + rebate_estimate;
    }

    /// Settles a resolved window: the winning token's shares each pay $1, the
    /// losing token's pay $0, then both positions are zeroed. Returns the cash
    /// paid out. Signed arithmetic handles a short: a net-short winning position
    /// (e.g. −10) correctly *debits* $10.
    pub fn settle_window(&mut self, up: &TokenId, down: &TokenId, winner: Outcome) -> Dollars {
        let winning = match winner {
            Outcome::Up => up,
            Outcome::Down => down,
        };
        let payout = Dollars::new(self.net_position(winning));
        self.collateral = self.collateral + payout;
        self.positions.remove(up);
        self.positions.remove(down);
        payout
    }

    /// Merges up to `requested` matched Up/Down pairs into collateral: each
    /// merged pair removes one Up and one Down share and credits $1 (a matched
    /// pair is worth exactly $1 at settlement). `matched = min(net_up, net_down)`
    /// and only when both sides are net long. Returns the number of pairs merged.
    pub fn merge(&mut self, up: &TokenId, down: &TokenId, requested: Size) -> Size {
        let net_up = self.net_position(up);
        let net_down = self.net_position(down);
        if net_up <= Decimal::ZERO || net_down <= Decimal::ZERO {
            return Size::ZERO;
        }
        // Both strictly positive ⇒ the min is a valid non-negative `Size`.
        let merged = match Size::new(net_up.min(net_down)) {
            Ok(matched) => matched.min(requested),
            Err(_) => return Size::ZERO,
        };
        if merged.is_zero() {
            return Size::ZERO;
        }
        let m = merged.as_decimal();
        *self.positions.entry(up.clone()).or_default() -= m;
        *self.positions.entry(down.clone()).or_default() -= m;
        self.collateral = self.collateral + Dollars::new(m);
        merged
    }

    /// The daily maker-rebate credit (§9): if the accrued estimate has reached
    /// `min_credit` ($1 minimum accrual), credit the whole balance to cash, add
    /// it to the `rebate_credited` income line, and reset the carry bucket.
    /// Below the threshold nothing moves (carry to the next cycle). Returns the
    /// amount credited.
    pub fn credit_rebate(&mut self, min_credit: Dollars) -> Dollars {
        if self.rebate_accrued.is_zero() || self.rebate_accrued < min_credit {
            return Dollars::ZERO;
        }
        let credited = self.rebate_accrued;
        self.collateral = self.collateral + credited;
        self.rebate_credited = self.rebate_credited + credited;
        self.rebate_accrued = Dollars::ZERO;
        credited
    }

    /// Current collateral (signed).
    #[must_use]
    pub fn collateral(&self) -> Dollars {
        self.collateral
    }

    /// Signed net position in a token (zero if none).
    #[must_use]
    pub fn net_position(&self, token: &TokenId) -> Decimal {
        self.positions.get(token).copied().unwrap_or(Decimal::ZERO)
    }

    /// Running maker-rebate estimate not yet credited.
    #[must_use]
    pub fn rebate_accrued(&self) -> Dollars {
        self.rebate_accrued
    }

    /// Cumulative taker fees paid (separate income line).
    #[must_use]
    pub fn fees_paid(&self) -> Dollars {
        self.fees_paid
    }

    /// Cumulative maker rebate credited to cash by the daily cycle.
    #[must_use]
    pub fn rebate_credited(&self) -> Dollars {
        self.rebate_credited
    }

    /// Cumulative signed capital injected (start + runtime adjustments).
    #[must_use]
    pub fn capital_injected(&self) -> Dollars {
        self.capital_injected
    }

    /// The signed non-zero net positions, sorted by token id (keeps shorts
    /// visible, unlike [`PaperWallet::to_wallet`] which drops non-positive nets).
    #[must_use]
    pub fn positions_view(&self) -> Vec<(TokenId, Decimal)> {
        let mut v: Vec<(TokenId, Decimal)> = self
            .positions
            .iter()
            .filter(|(_, net)| !net.is_zero())
            .map(|(t, net)| (t.clone(), *net))
            .collect();
        v.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        v
    }

    /// The richer ledger snapshot (signed positions + income lines).
    #[must_use]
    pub fn snapshot(&self) -> PaperLedgerSnapshot {
        PaperLedgerSnapshot {
            collateral: self.collateral,
            positions: self.positions_view(),
            fees_paid: self.fees_paid,
            rebate_accrued: self.rebate_accrued,
            rebate_credited: self.rebate_credited,
        }
    }

    /// Snapshot as the venue-agnostic [`Wallet`]. `available == total` (no
    /// open-order reservation yet); non-positive net positions are omitted.
    #[must_use]
    pub fn to_wallet(&self) -> Wallet {
        let mut positions: Vec<TokenBalance> = self
            .positions
            .iter()
            .filter_map(|(token, net)| {
                Size::new(*net)
                    .ok()
                    .filter(|s| !s.is_zero())
                    .map(|size| TokenBalance {
                        token_id: token.clone(),
                        size,
                    })
            })
            .collect();
        // Stable ordering for deterministic snapshots/tests.
        positions.sort_by(|a, b| a.token_id.as_str().cmp(b.token_id.as_str()));
        Wallet {
            collateral_available: self.collateral,
            collateral_total: self.collateral,
            positions,
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::{Asset, Series, WindowDuration};
    use core_types::{Liquidity, Outcome, RoundDir, TickSize, TimestampMs, WindowId, taker_fee};
    use rust_decimal::dec;

    use super::*;

    fn token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn down_token() -> TokenId {
        TokenId::new("222").unwrap()
    }
    fn px(d: Decimal) -> core_types::Price {
        core_types::Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    /// A maker/taker fill on an explicit token+outcome (settlement/merge tests).
    fn fill_on(
        tok: &TokenId,
        outcome: Outcome,
        side: Side,
        price: Decimal,
        size: Decimal,
        fee: Dollars,
    ) -> Fill {
        Fill {
            order_id: core_types::OrderId::new("paper-1").unwrap(),
            trade_id: None,
            window: window(),
            token_id: tok.clone(),
            outcome,
            side,
            price: px(price),
            size: Size::new(size).unwrap(),
            liquidity: if fee.is_zero() {
                Liquidity::Maker
            } else {
                Liquidity::Taker
            },
            fee,
            ts_venue: TimestampMs::from_millis(1),
            ts_local: TimestampMs::from_millis(1),
        }
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1_000_000),
        }
    }
    fn fill(side: Side, price: Decimal, size: Decimal, liquidity: Liquidity, fee: Dollars) -> Fill {
        Fill {
            order_id: core_types::OrderId::new("paper-1").unwrap(),
            trade_id: None,
            window: window(),
            token_id: token(),
            outcome: Outcome::Up,
            side,
            price: px(price),
            size: Size::new(size).unwrap(),
            liquidity,
            fee,
            ts_venue: TimestampMs::from_millis(1),
            ts_local: TimestampMs::from_millis(1),
        }
    }

    #[test]
    fn buy_debits_and_credits_shares() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        let fee = taker_fee(Size::new(dec!(100)).unwrap(), dec!(0.07), px(dec!(0.40)));
        w.apply_fill(
            &fill(Side::Buy, dec!(0.40), dec!(100), Liquidity::Taker, fee),
            Dollars::ZERO,
        );
        // 1000 − (0.40 × 100) − fee.
        assert_eq!(
            w.collateral(),
            Dollars::new(dec!(1000)) - Dollars::new(dec!(40)) - fee
        );
        assert_eq!(w.net_position(&token()), dec!(100));
        let wallet = w.to_wallet();
        assert_eq!(wallet.positions.len(), 1);
        assert_eq!(wallet.positions[0].size, Size::new(dec!(100)).unwrap());
    }

    #[test]
    fn sell_credits_and_debits_shares() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        // Buy 100 first, then sell 60 as maker (fee 0).
        w.apply_fill(
            &fill(
                Side::Buy,
                dec!(0.40),
                dec!(100),
                Liquidity::Maker,
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        w.apply_fill(
            &fill(
                Side::Sell,
                dec!(0.60),
                dec!(60),
                Liquidity::Maker,
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        // 1000 − 40 + 36 = 996.
        assert_eq!(w.collateral(), Dollars::new(dec!(996)));
        assert_eq!(w.net_position(&token()), dec!(40));
    }

    #[test]
    fn negative_net_is_omitted_from_reported_wallet() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        w.apply_fill(
            &fill(
                Side::Sell,
                dec!(0.60),
                dec!(10),
                Liquidity::Maker,
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        assert_eq!(w.net_position(&token()), dec!(-10));
        assert!(w.to_wallet().positions.is_empty());
    }

    #[test]
    fn rebate_estimate_accrues() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        let rebate = Dollars::new(dec!(0.35));
        w.apply_fill(
            &fill(
                Side::Sell,
                dec!(0.50),
                dec!(100),
                Liquidity::Maker,
                Dollars::ZERO,
            ),
            rebate,
        );
        assert_eq!(w.rebate_accrued(), rebate);
    }

    #[test]
    fn set_capital_resets_collateral() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        w.set_capital(Dollars::new(dec!(25000)));
        assert_eq!(w.collateral(), Dollars::new(dec!(25000)));
    }

    #[test]
    fn settles_one_dollar_per_winning_share() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Buy,
                dec!(0.40),
                dec!(100),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        w.apply_fill(
            &fill_on(
                &down_token(),
                Outcome::Down,
                Side::Buy,
                dec!(0.55),
                dec!(40),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        let before = w.collateral();
        let payout = w.settle_window(&token(), &down_token(), Outcome::Up);
        assert_eq!(payout, Dollars::new(dec!(100))); // 100 winning shares × $1
        assert_eq!(w.collateral(), before + Dollars::new(dec!(100)));
        assert!(w.net_position(&token()).is_zero());
        assert!(w.net_position(&down_token()).is_zero());
    }

    #[test]
    fn settles_signed_short() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        // Sell 10 Up we don't hold → net −10 Up, cash +5.
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Sell,
                dec!(0.50),
                dec!(10),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        assert_eq!(w.net_position(&token()), dec!(-10));
        let before = w.collateral();
        let payout = w.settle_window(&token(), &down_token(), Outcome::Up);
        assert_eq!(payout, Dollars::new(dec!(-10))); // short winner owes $10
        assert_eq!(w.collateral(), before - Dollars::new(dec!(10)));
        assert!(w.net_position(&token()).is_zero());
    }

    #[test]
    fn merge_pays_one_dollar_per_pair() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Buy,
                dec!(0.40),
                dec!(30),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        w.apply_fill(
            &fill_on(
                &down_token(),
                Outcome::Down,
                Side::Buy,
                dec!(0.55),
                dec!(20),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        let before = w.collateral();
        let merged = w.merge(&token(), &down_token(), Size::new(dec!(25)).unwrap());
        assert_eq!(merged, Size::new(dec!(20)).unwrap()); // capped at matched=20
        assert_eq!(w.collateral(), before + Dollars::new(dec!(20)));
        assert_eq!(w.net_position(&token()), dec!(10));
        assert!(w.net_position(&down_token()).is_zero());
    }

    #[test]
    fn merge_caps_at_requested_and_skips_one_sided() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Buy,
                dec!(0.40),
                dec!(30),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        // Only one side held → nothing to merge.
        assert_eq!(
            w.merge(&token(), &down_token(), Size::new(dec!(10)).unwrap()),
            Size::ZERO
        );
        w.apply_fill(
            &fill_on(
                &down_token(),
                Outcome::Down,
                Side::Buy,
                dec!(0.55),
                dec!(30),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        // Request fewer than matched → capped at the request.
        assert_eq!(
            w.merge(&token(), &down_token(), Size::new(dec!(10)).unwrap()),
            Size::new(dec!(10)).unwrap()
        );
        assert_eq!(w.net_position(&token()), dec!(20));
        assert_eq!(w.net_position(&down_token()), dec!(20));
    }

    #[test]
    fn rebate_min_carryover() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        // Accrue $0.80 (carried by a zero-fee maker fill).
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Buy,
                dec!(0.50),
                dec!(1),
                Dollars::ZERO,
            ),
            Dollars::new(dec!(0.80)),
        );
        assert_eq!(w.credit_rebate(Dollars::new(dec!(1))), Dollars::ZERO); // below $1
        assert_eq!(w.rebate_accrued(), Dollars::new(dec!(0.80))); // carried over
        // Accrue +$0.40 → $1.20, now over the threshold.
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Buy,
                dec!(0.50),
                dec!(1),
                Dollars::ZERO,
            ),
            Dollars::new(dec!(0.40)),
        );
        let before = w.collateral();
        assert_eq!(
            w.credit_rebate(Dollars::new(dec!(1))),
            Dollars::new(dec!(1.20))
        );
        assert_eq!(w.collateral(), before + Dollars::new(dec!(1.20)));
        assert_eq!(w.rebate_credited(), Dollars::new(dec!(1.20)));
        assert!(w.rebate_accrued().is_zero());
    }

    #[test]
    fn fees_paid_accumulates_taker_only() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        let fee = taker_fee(Size::new(dec!(100)).unwrap(), dec!(0.07), px(dec!(0.50)));
        // One taker fill (fee) + one maker fill (zero).
        w.apply_fill(
            &fill_on(&token(), Outcome::Up, Side::Buy, dec!(0.50), dec!(100), fee),
            Dollars::ZERO,
        );
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Sell,
                dec!(0.50),
                dec!(50),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        assert_eq!(w.fees_paid(), fee);
    }

    #[test]
    fn adjust_capital_tracks_injection() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        w.adjust_capital(Dollars::new(dec!(500)));
        assert_eq!(w.collateral(), Dollars::new(dec!(1500)));
        assert_eq!(w.capital_injected(), Dollars::new(dec!(1500)));
        w.adjust_capital(Dollars::new(dec!(-200)));
        assert_eq!(w.collateral(), Dollars::new(dec!(1300)));
        assert_eq!(w.capital_injected(), Dollars::new(dec!(1300)));
    }

    #[test]
    fn set_capital_records_implied_injection() {
        let mut w = PaperWallet::new(Dollars::new(dec!(1000)));
        // Spend some cash so the implied delta is measured from current, not start.
        w.apply_fill(
            &fill_on(
                &token(),
                Outcome::Up,
                Side::Buy,
                dec!(0.40),
                dec!(100),
                Dollars::ZERO,
            ),
            Dollars::ZERO,
        );
        let before = w.collateral(); // 960
        w.set_capital(Dollars::new(dec!(2500)));
        assert_eq!(w.collateral(), Dollars::new(dec!(2500)));
        // injected = start 1000 + (2500 − 960).
        assert_eq!(
            w.capital_injected(),
            Dollars::new(dec!(1000)) + (Dollars::new(dec!(2500)) - before)
        );
    }
}

#[cfg(test)]
mod conservation {
    //! Property-style conservation. Across random sequences of fills, merges,
    //! settlements, capital adjustments, and rebate credits, an **independent
    //! shadow oracle** — built only from the operation inputs and the spec
    //! formulas, never from the wallet's own accumulators — must equal the
    //! wallet exactly (Decimal) after every step. The closing identity collapses
    //! to `wallet.collateral == oracle.cash` once everything is settled, so the
    //! wallet's number is only ever checked against the oracle, never recomputed
    //! via the wallet itself: a value leak (forgetting to zero a settled
    //! position, double-crediting a merge, mischarging a fee) shows as a
    //! mismatch. With every share marked at its $1/$0 settlement value, no
    //! operation creates or destroys value — cash + marked inventory moves only
    //! by fees, rebates, and realized edge.

    use std::collections::HashMap;

    use core_types::{
        Asset, Decimal, Dollars, Fill, Liquidity, Outcome, Price, RoundDir, Series, Side, Size,
        TickSize, TimestampMs, TokenId, WindowDuration, WindowId, taker_fee,
    };
    use rust_decimal::dec;

    use super::PaperWallet;

    /// Minimal xorshift64* (the `latency.rs` idiom), deterministic per seed.
    struct Rng {
        state: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
        fn coin(&mut self) -> bool {
            self.below(2) == 0
        }
    }

    fn up() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn down() -> TokenId {
        TokenId::new("222").unwrap()
    }
    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(1),
        }
    }
    /// `n` in 1..=99 → the `0.0n` price on the 0.01 grid.
    fn price(n: u64) -> Price {
        Price::quantize(Decimal::new(n as i64, 2), TickSize::T001, RoundDir::Down).unwrap()
    }

    /// The independent shadow ledger.
    #[derive(Default)]
    struct Oracle {
        cash: Dollars,
        shares: HashMap<TokenId, Decimal>,
        fees: Dollars,
        rebate_accrued: Dollars,
        rebate_credited: Dollars,
        capital_injected: Dollars,
    }
    impl Oracle {
        fn new(start: Dollars) -> Self {
            Self {
                cash: start,
                capital_injected: start,
                ..Default::default()
            }
        }
        fn net(&self, t: &TokenId) -> Decimal {
            self.shares.get(t).copied().unwrap_or(Decimal::ZERO)
        }
    }

    /// Asserts the wallet equals the oracle on every tracked accumulator.
    fn assert_consistent(w: &PaperWallet, o: &Oracle, ctx: &str) {
        assert_eq!(w.collateral(), o.cash, "cash diverged ({ctx})");
        assert_eq!(
            w.net_position(&up()),
            o.net(&up()),
            "up shares diverged ({ctx})"
        );
        assert_eq!(
            w.net_position(&down()),
            o.net(&down()),
            "down shares diverged ({ctx})"
        );
        assert_eq!(w.fees_paid(), o.fees, "fees diverged ({ctx})");
        assert_eq!(
            w.rebate_accrued(),
            o.rebate_accrued,
            "rebate accrued diverged ({ctx})"
        );
        assert_eq!(
            w.rebate_credited(),
            o.rebate_credited,
            "rebate credited diverged ({ctx})"
        );
        assert_eq!(
            w.capital_injected(),
            o.capital_injected,
            "capital injected diverged ({ctx})"
        );
    }

    #[test]
    fn random_sequences_conserve_value() {
        let fee_rate = dec!(0.07);
        let rebate_share = dec!(0.20);
        let min_credit = Dollars::new(dec!(1));
        let start = Dollars::new(dec!(10000));

        for seed in 1..=8u64 {
            let mut rng = Rng::new(seed);
            let mut w = PaperWallet::new(start);
            let mut o = Oracle::new(start);
            assert_consistent(&w, &o, "init");

            for _ in 0..400 {
                match rng.below(10) {
                    // Fill (≈60% of steps): random token/side/maker-taker/price/size.
                    0..=5 => {
                        let tok = if rng.coin() { up() } else { down() };
                        let outcome = if tok == up() {
                            Outcome::Up
                        } else {
                            Outcome::Down
                        };
                        let side = if rng.coin() { Side::Buy } else { Side::Sell };
                        let taker = rng.coin();
                        let p = price(1 + rng.below(99));
                        let qty = Size::new(Decimal::from(1 + rng.below(50))).unwrap();
                        let fee = if taker {
                            taker_fee(qty, fee_rate, p)
                        } else {
                            Dollars::ZERO
                        };
                        let rebate = if taker {
                            Dollars::ZERO
                        } else {
                            Dollars::new(rebate_share * taker_fee(qty, fee_rate, p).as_decimal())
                        };
                        let fill = Fill {
                            order_id: core_types::OrderId::new("paper-1").unwrap(),
                            trade_id: None,
                            window: window(),
                            token_id: tok.clone(),
                            outcome,
                            side,
                            price: p,
                            size: qty,
                            liquidity: if taker {
                                Liquidity::Taker
                            } else {
                                Liquidity::Maker
                            },
                            fee,
                            ts_venue: TimestampMs::from_millis(1),
                            ts_local: TimestampMs::from_millis(1),
                        };
                        w.apply_fill(&fill, rebate);
                        // Oracle mirror (spec formulas, not wallet internals).
                        let notional = Dollars::new(p.as_decimal() * qty.as_decimal());
                        match side {
                            Side::Buy => {
                                o.cash = o.cash - notional - fee;
                                *o.shares.entry(tok).or_default() += qty.as_decimal();
                            }
                            Side::Sell => {
                                o.cash = o.cash + notional - fee;
                                *o.shares.entry(tok).or_default() -= qty.as_decimal();
                            }
                        }
                        o.fees = o.fees + fee;
                        o.rebate_accrued = o.rebate_accrued + rebate;
                    }
                    // Merge a random pair count.
                    6 => {
                        let requested = Size::new(Decimal::from(rng.below(40))).unwrap();
                        let merged = w.merge(&up(), &down(), requested);
                        let nu = o.net(&up());
                        let nd = o.net(&down());
                        let o_merged = if nu <= Decimal::ZERO || nd <= Decimal::ZERO {
                            Decimal::ZERO
                        } else {
                            nu.min(nd).min(requested.as_decimal())
                        };
                        assert_eq!(merged.as_decimal(), o_merged, "merge count diverged");
                        if o_merged > Decimal::ZERO {
                            *o.shares.entry(up()).or_default() -= o_merged;
                            *o.shares.entry(down()).or_default() -= o_merged;
                            o.cash = o.cash + Dollars::new(o_merged);
                        }
                    }
                    // Settle the window on a random outcome.
                    7 => {
                        let winner = if rng.coin() {
                            Outcome::Up
                        } else {
                            Outcome::Down
                        };
                        let payout = w.settle_window(&up(), &down(), winner);
                        let winning = if winner == Outcome::Up { up() } else { down() };
                        let o_payout = Dollars::new(o.net(&winning));
                        assert_eq!(payout, o_payout, "settle payout diverged");
                        o.cash = o.cash + o_payout;
                        o.shares.remove(&up());
                        o.shares.remove(&down());
                    }
                    // Adjust capital ±.
                    8 => {
                        let mag = Decimal::from(rng.below(2000));
                        let delta = if rng.coin() {
                            Dollars::new(mag)
                        } else {
                            Dollars::new(-mag)
                        };
                        w.adjust_capital(delta);
                        o.cash = o.cash + delta;
                        o.capital_injected = o.capital_injected + delta;
                    }
                    // Daily rebate credit ($1 minimum accrual).
                    _ => {
                        let credited = w.credit_rebate(min_credit);
                        let o_credited =
                            if o.rebate_accrued.is_zero() || o.rebate_accrued < min_credit {
                                Dollars::ZERO
                            } else {
                                o.rebate_accrued
                            };
                        assert_eq!(credited, o_credited, "rebate credit diverged");
                        if !o_credited.is_zero() {
                            o.cash = o.cash + o_credited;
                            o.rebate_credited = o.rebate_credited + o_credited;
                            o.rebate_accrued = Dollars::ZERO;
                        }
                    }
                }
                assert_consistent(&w, &o, "step");
            }

            // Closing identity: settle everything → all positions zero → the
            // marked NAV is just cash, and it matches the independent oracle.
            let payout = w.settle_window(&up(), &down(), Outcome::Up);
            assert_eq!(payout, Dollars::new(o.net(&up())));
            o.cash = o.cash + payout;
            o.shares.clear();
            assert!(w.net_position(&up()).is_zero());
            assert!(w.net_position(&down()).is_zero());
            assert_eq!(
                w.collateral(),
                o.cash,
                "closing identity broke (seed {seed})"
            );
        }
    }
}
