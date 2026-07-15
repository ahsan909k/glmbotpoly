//! Per-window PnL attribution that reconciles to the ledger (CLAUDE.md §8/§10).
//!
//! A settled window's realized PnL is decomposed into five buckets:
//!
//! - **locked-pair PnL** — the guaranteed `$1`-per-matched-pair edge.
//! - **excess-inventory PnL** — the unmatched directional bet, marked to settlement.
//! - **taker fees** — the taker fees paid (a negative bucket).
//! - **estimated maker rebates** — `rebate_share · Σ taker_fee(maker volume)`.
//! - **settlement remainder** — realized intra-window trading (sells/merges),
//!   exactly `0` in today's buy-only, merge-deferred system.
//!
//! # The reconciliation (exact in Decimal)
//!
//! The engine's ledger PnL is `realized_pnl = cash_flow + winning·$1`. Define the
//! **held-inventory mark** as the division-free, exact quantity
//!
//! ```text
//! held_mark = (winning side shares)·$1 − (up.cost + down.cost)
//! ```
//!
//! Then `realized_pnl = held_mark + traded_out_pnl − fees_paid` is an exact
//! identity (the per-share average-cost rounding in any sell/merge cancels
//! between `held_cost` and `traded_out_pnl`). So we set
//!
//! ```text
//! settlement_remainder = realized_pnl + fees_paid − held_mark   (== traded_out_pnl, exact)
//! excess_pnl           = (excess wins ? |excess| : 0) − |excess|·avg_excess
//! locked_pair_pnl      = held_mark − excess_pnl                 (absorbs avg rounding)
//! taker_fees           = −fees_paid
//! ```
//!
//! anchoring on `held_mark` rather than the rounded `pair_cost` so the
//! identities below hold to the last Decimal place:
//!
//! - **4-bucket (ledger):** `locked + excess + remainder + taker_fees == realized_pnl`.
//! - **5-bucket (comprehensive):** add `estimated_rebate` to reach the window's
//!   full economic PnL `realized_pnl + estimated_rebate` (rebates are credited on
//!   a separate daily cycle, so they extend the ledger figure rather than sit in
//!   it — CLAUDE.md §9).
//!
//! `excess_pnl` is the clean directional formula (exactly `0` for a balanced
//! book), so `locked_pair_pnl` absorbs the tiny average-cost rounding — invisible
//! at any display precision and confined to the matched bucket.

use core_types::{
    Decimal, Dollars, Fill, Liquidity, Mode, Outcome, SettlementSummary, Size, TimestampMs,
    WindowId, taker_fee,
};
use serde::{Deserialize, Serialize};

/// Per-window accumulator: the only thing folded from the fill stream that the
/// [`SettlementSummary`] cannot supply — the maker-rebate base and the fill
/// counts. The locked/excess/remainder buckets are a pure function of the
/// settlement summary (see [`WindowAttribution::from_summary`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowAttrib {
    window: WindowId,
    /// `Σ taker_fee(size, fee_rate, price)` over our MAKER fills — the §9 rebate
    /// base (maker fills carry `fee = $0`, so the fee is recomputed here).
    maker_rebate_base: Dollars,
    /// `Σ price·size` over our TAKER fills — the taker-budget-usage numerator
    /// (taker spend is not on the bus, so it is reconstructed here).
    taker_notional: Dollars,
    maker_fills: u64,
    taker_fills: u64,
}

impl WindowAttrib {
    /// A fresh accumulator for a window.
    #[must_use]
    pub fn new(window: WindowId) -> Self {
        Self {
            window,
            maker_rebate_base: Dollars::ZERO,
            taker_notional: Dollars::ZERO,
            maker_fills: 0,
            taker_fills: 0,
        }
    }

    /// The window this accumulator belongs to.
    #[must_use]
    pub fn window(&self) -> WindowId {
        self.window
    }

    /// Folds one fill: accumulates the maker-rebate base (maker fills only,
    /// recomputing the taker-fee-equivalent at `fee_rate`), the taker notional
    /// (taker fills only), and the fill counts.
    pub fn apply_fill(&mut self, fill: &Fill, fee_rate: Decimal) {
        match fill.liquidity {
            Liquidity::Maker => {
                self.maker_rebate_base =
                    self.maker_rebate_base + taker_fee(fill.size, fee_rate, fill.price);
                self.maker_fills += 1;
            }
            Liquidity::Taker => {
                self.taker_notional = self.taker_notional
                    + Dollars::new(fill.price.as_decimal() * fill.size.as_decimal());
                self.taker_fills += 1;
            }
        }
    }

    /// Maker-rebate base accumulated so far.
    #[must_use]
    pub fn maker_rebate_base(&self) -> Dollars {
        self.maker_rebate_base
    }

    /// Taker notional (`Σ price·size` over taker fills) accumulated so far.
    #[must_use]
    pub fn taker_notional(&self) -> Dollars {
        self.taker_notional
    }

    /// Maker fill count.
    #[must_use]
    pub fn maker_fills(&self) -> u64 {
        self.maker_fills
    }

    /// Taker fill count.
    #[must_use]
    pub fn taker_fills(&self) -> u64 {
        self.taker_fills
    }
}

/// The decomposed PnL of one settled window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowAttribution {
    /// Window that settled.
    pub window: WindowId,
    /// Session mode (stamped by the [`Analytics`](crate::Analytics) instance;
    /// events carry no mode tag).
    pub mode: Mode,
    /// Winning outcome.
    pub outcome: Outcome,
    /// Guaranteed `$1`-per-matched-pair edge (absorbs average-cost rounding).
    pub locked_pair_pnl: Dollars,
    /// Unmatched directional-bet PnL, marked to settlement.
    pub excess_pnl: Dollars,
    /// Realized intra-window trading (sells/merges); `0` in the buy-only system.
    pub settlement_remainder: Dollars,
    /// Taker fees paid (≤ 0).
    pub taker_fees: Dollars,
    /// Estimated maker rebate accrued on this window (≥ 0). Not in `realized_pnl`.
    pub estimated_rebate: Dollars,
    /// The engine's authoritative ledger PnL for the window.
    pub realized_pnl: Dollars,
    /// Maker fill count.
    pub maker_fills: u64,
    /// Taker fill count.
    pub taker_fills: u64,
    /// `Σ price·size` over our taker fills — the taker-budget-usage input. An
    /// **independent accumulator, NOT a PnL bucket**: it is excluded from
    /// [`trading_sum`](Self::trading_sum) / [`bucket_sum`](Self::bucket_sum) (the
    /// taker fee it incurs is already in `taker_fees`).
    pub taker_notional: Dollars,
    /// Matched pairs still held at close (both sides filled): `min(up, down)`.
    /// Dashboard display only — not a PnL bucket.
    pub matched_pairs: Size,
    /// Pairs merged back to collateral during the window (also "completed").
    pub merged_pairs: Size,
    /// Magnitude of the unmatched (stranded) shares at close: `|excess|`.
    pub stranded_shares: Decimal,
    /// Combined average cost of a held pair (`avg_up + avg_down`), `None` unless
    /// both sides were held — the "average cost to complete a pair".
    pub pair_cost: Option<Decimal>,
    /// Settlement time (window close).
    pub ts: TimestampMs,
}

impl WindowAttribution {
    /// Decomposes a settlement summary into the five buckets.
    pub fn from_summary(
        summary: &SettlementSummary,
        mode: Mode,
        maker_rebate_base: Dollars,
        rebate_rate: Decimal,
        maker_fills: u64,
        taker_fills: u64,
        taker_notional: Dollars,
    ) -> Self {
        let held_cost = summary.up.cost + summary.down.cost;
        let winning_shares = match summary.outcome {
            Outcome::Up => summary.up.shares,
            Outcome::Down => summary.down.shares,
        };
        let payout = Dollars::new(winning_shares.as_decimal());
        let held_mark = payout - held_cost;

        let excess_pnl = excess_inventory_pnl(summary);
        let locked_pair_pnl = held_mark - excess_pnl;
        let taker_fees = -summary.fees_paid;
        // == traded_out_pnl, exactly: realized = held_mark + traded_out − fees.
        let settlement_remainder = (summary.realized_pnl + summary.fees_paid) - held_mark;
        let estimated_rebate = Dollars::new(rebate_rate * maker_rebate_base.as_decimal());

        Self {
            window: summary.window,
            mode,
            outcome: summary.outcome,
            locked_pair_pnl,
            excess_pnl,
            settlement_remainder,
            taker_fees,
            estimated_rebate,
            realized_pnl: summary.realized_pnl,
            maker_fills,
            taker_fills,
            taker_notional,
            matched_pairs: summary.matched_pairs,
            merged_pairs: summary.merged_pairs,
            stranded_shares: summary.excess.abs(),
            pair_cost: summary.pair_cost,
            ts: summary.ts,
        }
    }

    /// The four trading buckets summed — equals `realized_pnl` exactly.
    #[must_use]
    pub fn trading_sum(&self) -> Dollars {
        self.locked_pair_pnl + self.excess_pnl + self.settlement_remainder + self.taker_fees
    }

    /// All five buckets summed — equals the comprehensive window PnL exactly.
    #[must_use]
    pub fn bucket_sum(&self) -> Dollars {
        self.trading_sum() + self.estimated_rebate
    }

    /// The window's full economic PnL: ledger PnL plus the estimated rebate.
    #[must_use]
    pub fn comprehensive_pnl(&self) -> Dollars {
        self.realized_pnl + self.estimated_rebate
    }

    /// The §10 "inventory PnL" half of the locked-pair-vs-inventory split:
    /// excess plus settlement remainder.
    #[must_use]
    pub fn inventory_pnl(&self) -> Dollars {
        self.excess_pnl + self.settlement_remainder
    }

    /// Dashboard **bucket 1 — "Profit from completed pairs"**: the guaranteed,
    /// fee-free `$1`-per-matched-pair maker edge.
    #[must_use]
    pub fn completed_pair_pnl(&self) -> Dollars {
        self.locked_pair_pnl
    }

    /// Dashboard **bucket 2 — "Profit/loss from stuck legs"**: the unmatched
    /// directional inventory (excess + intra-window trading) net of the taker
    /// fees paid chasing it. With [`completed_pair_pnl`](Self::completed_pair_pnl)
    /// these two buckets sum to [`realized_pnl`](Self::realized_pnl) **exactly**
    /// (they re-label the four-bucket ledger identity — see
    /// [`trading_sum`](Self::trading_sum)).
    #[must_use]
    pub fn stuck_leg_pnl(&self) -> Dollars {
        self.inventory_pnl() + self.taker_fees
    }

    /// Pairs the strategy completed (both sides filled): held matched pairs plus
    /// pairs merged back to collateral.
    #[must_use]
    pub fn pairs_completed(&self) -> Size {
        self.matched_pairs + self.merged_pairs
    }

    /// Whether the window's ledger PnL was positive.
    #[must_use]
    pub fn is_profitable(&self) -> bool {
        self.realized_pnl > Dollars::ZERO
    }
}

/// The clean directional PnL of the unmatched excess shares, marked to the
/// resolved outcome (exactly `0` for a balanced book).
fn excess_inventory_pnl(summary: &SettlementSummary) -> Dollars {
    let x = summary.excess;
    if x.is_zero() {
        return Dollars::ZERO;
    }
    let (excess_side, shares, avg) = if x > Decimal::ZERO {
        (
            Outcome::Up,
            x,
            summary.up.avg_price().unwrap_or(Decimal::ZERO),
        )
    } else {
        (
            Outcome::Down,
            -x,
            summary.down.avg_price().unwrap_or(Decimal::ZERO),
        )
    };
    let payoff = if excess_side == summary.outcome {
        shares
    } else {
        Decimal::ZERO
    };
    Dollars::new(payoff - shares * avg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{
        Asset, OrderId, Price, RoundDir, Series, Side, Size, TickSize, TokenId, WindowDuration,
    };
    use rust_decimal::dec;

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = OPEN_MS + 300_000;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }

    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }

    fn side_inv(shares: Decimal, cost: Decimal) -> core_types::SideInventory {
        core_types::SideInventory {
            shares: sz(shares),
            cost: Dollars::new(cost),
        }
    }

    fn fill(outcome: Outcome, side: Side, price: Decimal, size: Decimal, liq: Liquidity) -> Fill {
        Fill {
            order_id: OrderId::new("o").unwrap(),
            trade_id: None,
            window: window(),
            token_id: TokenId::new("1").unwrap(),
            outcome,
            side,
            price: px(price),
            size: sz(size),
            liquidity: liq,
            fee: if liq == Liquidity::Taker {
                taker_fee(sz(size), dec!(0.07), px(price))
            } else {
                Dollars::ZERO
            },
            ts_venue: TimestampMs::from_millis(OPEN_MS + 100),
            ts_local: TimestampMs::from_millis(OPEN_MS + 100),
        }
    }

    // ---- buy-only, balanced book: remainder is exactly zero ------------------

    #[test]
    fn balanced_buy_only_remainder_zero_and_sums_to_realized() {
        // BUY 100 Up @ 0.48 + BUY 100 Down @ 0.49 (makers). Resolve Up.
        // cash = -97, payout = 100, realized = 3. pair_cost 0.97, locked = 3.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side_inv(dec!(100), dec!(48)),
            side_inv(dec!(100), dec!(49)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(3)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            Dollars::ZERO,
            dec!(0.2),
            2,
            0,
            Dollars::ZERO,
        );
        assert_eq!(a.locked_pair_pnl, Dollars::new(dec!(3)));
        assert_eq!(a.excess_pnl, Dollars::ZERO);
        assert_eq!(a.settlement_remainder, Dollars::ZERO);
        assert_eq!(a.taker_fees, Dollars::ZERO);
        assert_eq!(a.trading_sum(), Dollars::new(dec!(3)));
        assert_eq!(a.trading_sum(), a.realized_pnl);
    }

    // ---- excess inventory, marked to the winning outcome ---------------------

    #[test]
    fn up_excess_wins_pays_out() {
        // 150 Up @ 0.40 (cost 60), 100 Down @ 0.50 (cost 50). 50 Up excess @ 0.40.
        // Resolve Up: payout 150, held_cost 110, held_mark 40.
        // excess = 50·$1 − 50·0.40 = 50 − 20 = 30. locked = 40 − 30 = 10
        //   (== 100 matched · (1 − 0.90)). remainder = realized + 0 − 40.
        // cash = -110, realized (Up) = -110 + 150 = 40 ⇒ remainder 0.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side_inv(dec!(150), dec!(60)),
            side_inv(dec!(100), dec!(50)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(40)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            Dollars::ZERO,
            dec!(0.2),
            0,
            0,
            Dollars::ZERO,
        );
        assert_eq!(a.excess_pnl, Dollars::new(dec!(30)));
        assert_eq!(a.locked_pair_pnl, Dollars::new(dec!(10)));
        assert_eq!(a.settlement_remainder, Dollars::ZERO);
        assert_eq!(a.trading_sum(), a.realized_pnl);
    }

    #[test]
    fn up_excess_loses_costs_us() {
        // Same book, resolve Down: payout 100, held_cost 110, held_mark -10.
        // excess (Up loses) = 0 − 50·0.40 = -20. locked = -10 − (-20) = 10.
        // realized (Down) = -110 + 100 = -10 ⇒ remainder 0.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Down,
            side_inv(dec!(150), dec!(60)),
            side_inv(dec!(100), dec!(50)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(-10)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            Dollars::ZERO,
            dec!(0.2),
            0,
            0,
            Dollars::ZERO,
        );
        assert_eq!(a.excess_pnl, Dollars::new(dec!(-20)));
        assert_eq!(a.locked_pair_pnl, Dollars::new(dec!(10)));
        assert_eq!(a.settlement_remainder, Dollars::ZERO);
        assert_eq!(a.trading_sum(), a.realized_pnl);
    }

    // ---- two-bucket split (the dashboard explainer) --------------------------

    #[test]
    fn two_bucket_split_reconciles_to_realized() {
        // up_excess_wins fixture: locked 10, excess 30, remainder 0, fees 0.
        // bucket 1 (completed pairs) = locked = 10.
        // bucket 2 (stuck legs)      = inventory (30) + fees (0) = 30.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side_inv(dec!(150), dec!(60)),
            side_inv(dec!(100), dec!(50)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(40)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            Dollars::ZERO,
            dec!(0.2),
            0,
            0,
            Dollars::ZERO,
        );
        assert_eq!(a.completed_pair_pnl(), Dollars::new(dec!(10)));
        assert_eq!(a.stuck_leg_pnl(), Dollars::new(dec!(30)));
        // The two buckets sum to the ledger's realized PnL, exactly.
        assert_eq!(a.completed_pair_pnl() + a.stuck_leg_pnl(), a.realized_pnl);
        // Stranded shares: 50 Up excess. Matched pairs none held (summary has
        // matched_pairs derived from the inventory): min(150,100)=100.
        assert_eq!(a.stranded_shares, dec!(50));
        assert_eq!(a.matched_pairs, sz(dec!(100)));
        assert_eq!(a.pairs_completed(), sz(dec!(100)));
    }

    #[test]
    fn two_bucket_split_folds_taker_fee_into_stuck_legs() {
        // One taker BUY 100 Up @ 0.50, fee 1.75; resolve Up. realized 48.25.
        // All 100 are Up excess (no Down), so they are "stuck" until settlement.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side_inv(dec!(100), dec!(50)),
            side_inv(dec!(0), dec!(0)),
            Size::ZERO,
            Dollars::new(dec!(1.75)),
            Dollars::new(dec!(48.25)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            Dollars::ZERO,
            dec!(0.2),
            0,
            1,
            Dollars::new(dec!(50)),
        );
        // No matched pairs ⇒ no completed-pair edge; everything is the stuck leg
        // net of the fee. The two buckets still sum to realized exactly.
        assert_eq!(a.completed_pair_pnl(), Dollars::ZERO);
        assert_eq!(
            a.stuck_leg_pnl(),
            a.inventory_pnl() + Dollars::new(dec!(-1.75))
        );
        assert_eq!(a.completed_pair_pnl() + a.stuck_leg_pnl(), a.realized_pnl);
    }

    // ---- fees and rebate -----------------------------------------------------

    #[test]
    fn taker_fee_is_a_negative_bucket() {
        // One taker BUY 100 Up @ 0.50: fee = 100·0.07·0.50·0.50 = 1.75.
        let f = fill(
            Outcome::Up,
            Side::Buy,
            dec!(0.50),
            dec!(100),
            Liquidity::Taker,
        );
        assert_eq!(f.fee, Dollars::new(dec!(1.75)));
        // Resolve Up: payout 100, held_cost 50, held_mark 50.
        // cash = -50 - 1.75 = -51.75, realized = 48.25.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side_inv(dec!(100), dec!(50)),
            side_inv(dec!(0), dec!(0)),
            Size::ZERO,
            f.fee,
            Dollars::new(dec!(48.25)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        // Taker notional = 100 · 0.50 = $50 (the budget-usage numerator).
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            Dollars::ZERO,
            dec!(0.2),
            0,
            1,
            Dollars::new(dec!(50)),
        );
        assert_eq!(a.taker_fees, Dollars::new(dec!(-1.75)));
        assert_eq!(a.estimated_rebate, Dollars::ZERO); // taker fill earns no rebate base
        assert_eq!(a.settlement_remainder, Dollars::ZERO);
        assert_eq!(a.taker_notional, Dollars::new(dec!(50)));
        // The taker notional is NOT a PnL bucket — the ledger identity holds.
        assert_eq!(a.trading_sum(), a.realized_pnl);
        assert_eq!(a.bucket_sum(), a.comprehensive_pnl());
    }

    #[test]
    fn maker_rebate_base_uses_fee_formula_over_maker_volume() {
        // Two maker fills accumulate the rebate base at the §7 reference points.
        let mut attrib = WindowAttrib::new(window());
        attrib.apply_fill(
            &fill(
                Outcome::Up,
                Side::Buy,
                dec!(0.50),
                dec!(100),
                Liquidity::Maker,
            ),
            dec!(0.07),
        );
        attrib.apply_fill(
            &fill(
                Outcome::Down,
                Side::Buy,
                dec!(0.25),
                dec!(100),
                Liquidity::Maker,
            ),
            dec!(0.07),
        );
        // §7: $1.75 @ 0.50 + $1.3125 @ 0.25 = $3.0625 base.
        assert_eq!(attrib.maker_rebate_base(), Dollars::new(dec!(3.0625)));
        assert_eq!(attrib.maker_fills(), 2);
        // Rebate estimate = 0.20 · base.
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            side_inv(dec!(100), dec!(50)),
            side_inv(dec!(100), dec!(25)),
            Size::ZERO,
            Dollars::ZERO,
            Dollars::new(dec!(25)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let a = WindowAttribution::from_summary(
            &summary,
            Mode::Paper,
            attrib.maker_rebate_base(),
            dec!(0.2),
            attrib.maker_fills(),
            attrib.taker_fills(),
            attrib.taker_notional(),
        );
        assert_eq!(a.estimated_rebate, Dollars::new(dec!(0.6125)));
        // Both fills were makers, so no taker notional was accumulated.
        assert_eq!(a.taker_notional, Dollars::ZERO);
        // Rebate is NOT in the 4-bucket ledger identity...
        assert_eq!(a.trading_sum(), a.realized_pnl);
        // ...but the 5-bucket sum reaches the comprehensive (rebate-inclusive) PnL.
        assert_eq!(a.bucket_sum(), a.comprehensive_pnl());
        assert_eq!(
            a.comprehensive_pnl(),
            a.realized_pnl + Dollars::new(dec!(0.6125))
        );
        // The dashboard's two buckets plus the estimated rebate reconcile to the
        // comprehensive total exactly (the identity the summary panel asserts).
        assert_eq!(
            a.completed_pair_pnl() + a.stuck_leg_pnl() + a.estimated_rebate,
            a.comprehensive_pnl()
        );
    }

    #[test]
    fn taker_notional_accumulates_on_taker_fills_only() {
        let mut attrib = WindowAttrib::new(window());
        // Two makers (no notional contribution) + two takers.
        attrib.apply_fill(
            &fill(
                Outcome::Up,
                Side::Buy,
                dec!(0.50),
                dec!(100),
                Liquidity::Maker,
            ),
            dec!(0.07),
        );
        attrib.apply_fill(
            &fill(
                Outcome::Up,
                Side::Buy,
                dec!(0.40),
                dec!(50),
                Liquidity::Taker,
            ),
            dec!(0.07),
        );
        attrib.apply_fill(
            &fill(
                Outcome::Down,
                Side::Buy,
                dec!(0.60),
                dec!(20),
                Liquidity::Taker,
            ),
            dec!(0.07),
        );
        // 0.40·50 + 0.60·20 = 20 + 12 = $32; makers contribute $0.
        assert_eq!(attrib.taker_notional(), Dollars::new(dec!(32)));
        assert_eq!(attrib.taker_fills(), 2);
        assert_eq!(attrib.maker_fills(), 1);
    }

    // ---- shadow-oracle conservation -----------------------------------------

    /// xorshift64* — the codebase's deterministic test RNG idiom.
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
    }

    /// An independent ledger built only from operation inputs + the spec
    /// formulas. Tracks each side's shares AND cost (for the summary), cash,
    /// fees, merged pairs, and — crucially — the realized PnL of shares that
    /// LEFT via sell/merge, accumulated as it happens (`traded_out_pnl`), so the
    /// attribution's `settlement_remainder` can be checked against an
    /// independent number rather than its own residual.
    #[derive(Default)]
    struct Oracle {
        up_shares: Decimal,
        up_cost: Decimal,
        down_shares: Decimal,
        down_cost: Decimal,
        cash: Decimal,
        fees: Decimal,
        merged: Decimal,
        traded_out_pnl: Decimal,
    }
    impl Oracle {
        fn buy(&mut self, outcome: Outcome, price: Decimal, size: Decimal, fee: Decimal) {
            match outcome {
                Outcome::Up => {
                    self.up_shares += size;
                    self.up_cost += price * size;
                }
                Outcome::Down => {
                    self.down_shares += size;
                    self.down_cost += price * size;
                }
            }
            self.cash -= price * size + fee;
            self.fees += fee;
        }
        fn sell(&mut self, outcome: Outcome, price: Decimal, size: Decimal, fee: Decimal) {
            let (shares, cost) = match outcome {
                Outcome::Up => (&mut self.up_shares, &mut self.up_cost),
                Outcome::Down => (&mut self.down_shares, &mut self.down_cost),
            };
            if *shares <= Decimal::ZERO {
                return;
            }
            let avg = *cost / *shares;
            let sold = size.min(*shares);
            let cost_removed = avg * sold;
            let proceeds = price * sold;
            *shares -= sold;
            *cost -= cost_removed;
            self.cash += proceeds - fee;
            self.fees += fee;
            self.traded_out_pnl += proceeds - cost_removed;
        }
        fn merge(&mut self, req: Decimal) {
            let n = req
                .min(self.up_shares.min(self.down_shares))
                .max(Decimal::ZERO);
            if n.is_zero() {
                return;
            }
            let avg_up = self.up_cost / self.up_shares;
            let avg_down = self.down_cost / self.down_shares;
            let cost_removed = n * avg_up + n * avg_down;
            self.up_shares -= n;
            self.up_cost -= n * avg_up;
            self.down_shares -= n;
            self.down_cost -= n * avg_down;
            self.cash += n;
            self.merged += n;
            self.traded_out_pnl += n - cost_removed;
        }
        fn realized(&self, winner: Outcome) -> Decimal {
            let win = match winner {
                Outcome::Up => self.up_shares,
                Outcome::Down => self.down_shares,
            };
            self.cash + win
        }
        fn summary(&self, winner: Outcome) -> SettlementSummary {
            SettlementSummary::close(
                window(),
                winner,
                side_inv(self.up_shares, self.up_cost),
                side_inv(self.down_shares, self.down_cost),
                sz(self.merged),
                Dollars::new(self.fees),
                Dollars::new(self.realized(winner)),
                TimestampMs::from_millis(CLOSE_MS),
            )
        }
    }

    #[test]
    fn random_sequences_reconcile_to_the_ledger() {
        for seed in 0..8u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut oracle = Oracle::default();

            for _ in 0..300 {
                let op = rng.below(4);
                let outcome = if rng.below(2) == 0 {
                    Outcome::Up
                } else {
                    Outcome::Down
                };
                let p = px(Decimal::from(rng.below(99) + 1) / Decimal::from(100));
                let pd = p.as_decimal();
                let taker = rng.below(2) == 0;
                let fee = |size: Decimal| {
                    if taker {
                        taker_fee(sz(size), dec!(0.07), p).as_decimal()
                    } else {
                        Decimal::ZERO
                    }
                };
                match op {
                    0 | 1 => {
                        let size = Decimal::from(rng.below(20) + 1);
                        oracle.buy(outcome, pd, size, fee(size));
                    }
                    2 => {
                        let held = match outcome {
                            Outcome::Up => oracle.up_shares,
                            Outcome::Down => oracle.down_shares,
                        };
                        if held > Decimal::ZERO {
                            let max = u64::try_from(held.trunc()).unwrap_or(1).max(1);
                            let size = Decimal::from(rng.below(max) + 1).min(held);
                            oracle.sell(outcome, pd, size, fee(size));
                        }
                    }
                    _ => {
                        oracle.merge(Decimal::from(rng.below(20) + 1));
                    }
                }
            }

            let winner = if rng.below(2) == 0 {
                Outcome::Up
            } else {
                Outcome::Down
            };
            let summary = oracle.summary(winner);
            let a = WindowAttribution::from_summary(
                &summary,
                Mode::Paper,
                Dollars::ZERO,
                dec!(0.2),
                0,
                0,
                Dollars::ZERO,
            );

            // 4-bucket ledger identity — EXACT (telescopes to the ledger number).
            assert_eq!(
                a.trading_sum(),
                summary.realized_pnl,
                "seed {seed}: trading sum"
            );
            // The dashboard's TWO-bucket split is the same identity re-labeled
            // (locked + (excess+remainder) + fees). It sums to the ledger to
            // money precision; the different Decimal *addition grouping* leaves
            // only the same sub-money 28th-digit noise `excess_pnl` carries (the
            // exact, clean-number case is pinned in `two_bucket_split_*`).
            assert_eq!(
                (a.completed_pair_pnl() + a.stuck_leg_pnl()).round_dp(18),
                summary.realized_pnl.round_dp(18),
                "seed {seed}: two-bucket sum"
            );
            // The remainder IS the independently-accumulated departed-share PnL.
            // Both involve the average-cost division but via different operation
            // orders, so they agree only beyond money precision (the 28th
            // significant Decimal digit) — the buy-only path, division-free, is
            // checked exactly in `balanced_buy_only_remainder_zero...`.
            assert_eq!(
                a.settlement_remainder.round_dp(18),
                Dollars::new(oracle.traded_out_pnl).round_dp(18),
                "seed {seed}: remainder == traded_out_pnl"
            );
            // The held-inventory split (locked + excess) sums to the held mark.
            // `excess_pnl` carries the average-cost division, so compare beyond
            // any money precision (the 28th-significant-digit Decimal noise).
            let held_mark = Dollars::new(
                match winner {
                    Outcome::Up => oracle.up_shares,
                    Outcome::Down => oracle.down_shares,
                } - (oracle.up_cost + oracle.down_cost),
            );
            assert_eq!(
                (a.locked_pair_pnl + a.excess_pnl).round_dp(18),
                held_mark.round_dp(18),
                "seed {seed}: held mark"
            );
        }
    }
}
