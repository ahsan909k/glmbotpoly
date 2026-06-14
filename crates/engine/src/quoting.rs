//! The pure quoting calculator (CLAUDE.md §8): the strategy "brain" that turns
//! a model snapshot plus the current inventory into the complete desired set of
//! resting maker quotes for one window, or an explicit no-quote decision with a
//! reason.
//!
//! [`calculate_quotes`] is **pure and sans-IO** — no clock, no logging, no panic
//! on any runtime path (every fallible step maps to a skipped level or a typed
//! [`NoQuoteReason`]) — exactly the discipline of [`crate::normalize`] and
//! [`crate::inventory`]. It is the desired-state input the (later) cancel-first
//! quote manager diffs against the resting book to defend against adverse
//! selection: when the underlying moves, recomputing here reprices the offside
//! quotes away from their stale levels so the manager withdraws them.
//!
//! ## Market model (no short selling)
//!
//! Polymarket has no short selling, so two-sided market making on an Up/Down
//! market is **two BUY ladders** (the §7 mirror: BUY Up @ p ≡ SELL Down @ 1−p,
//! the same pattern the `bot paper-sim` quoter already uses to form mergeable
//! pairs):
//! - an **Up-buy** ladder descending from `center − half_spread` (buy Up below
//!   its fair `p_up`), and
//! - a **Down-buy** ladder descending from `(1 − center) − half_spread` (buy
//!   Down below its fair `1 − p_up`).
//!
//! Both sides are [`Side::Buy`], post-only [`TimeInForce::Gtc`]. The Down fair is
//! `1 − center` (not an independently re-skewed value) so the two fairs sum to
//! exactly 1 and the pair fair is `1 − 2·half_spread` — making pair-cost
//! discipline predictable. Matched Up+Down pairs are merged back to collateral
//! elsewhere (CLAUDE.md §8 capital recycling).
//!
//! ## Formulas (§8)
//!
//! - **Inventory-skewed center:** `center = p_up − γ · norm_imbalance`, where
//!   `norm_imbalance = clamp(excess / hard_cap, −1, 1)` is positive when long Up.
//!   Long Up lowers `center` (bid Up lower) and raises `1 − center` (bid Down
//!   higher), pushing inventory back toward balance.
//! - **Half-spread:** `max(min_edge, σ_1s · (k1·√hold·spike_mult + √rtt))`. The
//!   `σ_1s·√rtt` term is the *latency premium*: the expected adverse underlying
//!   move over the cancel round-trip we cannot pull within. The `spike_mult`
//!   widens the vol term when `σ_1s` is in a spike (§8 "widen on vol spikes").
//!
//! ## Engine-local params
//!
//! [`QuoteParams`] mirrors the relevant `config::EngineParams` defaults plus a
//! few brand-new knobs (latency RTT, ATM band, vol-spike threshold/multiplier)
//! that have no `config` field yet; the `config::EngineParams → QuoteParams` map
//! at the `bot` boundary is **deferred** to the engine-bus-wiring task, exactly
//! as [`NormalizerParams`](crate::NormalizerParams)/[`InventoryParams`](crate::InventoryParams)
//! deferred theirs, and as the model crate kept new `VolParams`/`HealthParams`
//! knobs crate-local. The caller maps each emitted [`QuoteLevel`] to an
//! [`OrderDraft`](crate::normalize::OrderDraft) (filling `token_id` from
//! `MarketInfo::tokens`) and runs it through [`normalize`](crate::normalize)
//! before placement — the calculator emits on-grid prices and sizes at or above
//! the configured minimums, so `normalize` only ever *bumps*, never rejects.

use core_types::{
    Decimal, Dollars, FeeParams, InventorySnapshot, ModelHealth, ModelHealthReason, ModelSnapshot,
    Outcome, Price, RoundDir, Side, Size, TickSize, TimeInForce, WindowId,
};
use rust_decimal::dec;
use rust_decimal::prelude::ToPrimitive;

use crate::inventory::{ExcessConstraint, authorizes_passive_add_sides, excess_constraint_sides};

/// Builds a [`Size`] from a non-negative integer literal (defaults only).
#[expect(
    clippy::unwrap_used,
    reason = "u32 literals are non-negative; Size::new rejects only negatives"
)]
fn shares(n: u32) -> Size {
    Size::new(Decimal::from(n)).unwrap()
}

/// Engine-local quoting tunables (CLAUDE.md §8).
///
/// The first block mirrors `config::EngineParams` (mapping from config to here is
/// deferred to the `bot` boundary, like [`NormalizerParams`](crate::NormalizerParams)).
/// The second block are knobs absent from `EngineParams` today, kept crate-local
/// with config exposure deferred (the `VolParams`/`HealthParams` precedent).
/// [`Default`] mirrors the committed `config/default.toml` values. `Copy`, but
/// **not** `Eq` (it holds `f64` fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteParams {
    /// Minimum half-spread (probability space) — quoting never goes tighter.
    pub min_edge: Decimal,
    /// `k1` on the `σ_1s · √hold` vol term.
    pub k1_vol_multiplier: f64,
    /// Expected passive holding time in seconds (the `√hold` horizon).
    pub expected_hold_secs: f64,
    /// `γ`: quote-center shift per unit of normalized inventory imbalance.
    pub gamma_inventory_skew: f64,
    /// Shares quoted at the touch (level 0).
    pub touch_size: Size,
    /// Total ladder levels including the touch.
    pub ladder_levels: u32,
    /// Shares per ladder level behind the touch.
    pub ladder_size_per_level: Size,
    /// Tick gap between consecutive ladder levels.
    pub ladder_tick_offset: u32,
    /// Pair-cost discipline: only add passively while post-trade `avg_up +
    /// avg_down` stays at or below this.
    pub pair_cost_threshold: Decimal,
    /// No at-the-money quoting once `τ` drops to this many seconds or fewer.
    pub no_atm_final_secs: u32,
    /// No passive orders at all (cancel-all) once `τ` drops to this or fewer.
    pub no_passive_final_secs: u32,
    /// Excess (unmatched) shares at which quoting narrows to the deficit side.
    pub soft_cap_excess: Size,
    /// Excess shares at which all passive making stops.
    pub hard_cap_excess: Size,
    /// Maximum worst-case loss per window — the binding sizing/risk constraint.
    pub max_worst_case_loss: Dollars,
    /// Measured cancel round-trip in seconds; the latency premium is
    /// `σ_1s · √(cancel_rtt_secs)` (config exposure deferred).
    pub cancel_rtt_secs: f64,
    /// A quote price within this distance of `0.50` counts as at-the-money for
    /// the final-seconds ATM gate (config exposure deferred).
    pub atm_band: Decimal,
    /// `σ_1s` at or above which the explicit vol-spike multiplier engages
    /// (config exposure deferred).
    pub vol_spike_threshold_1s: f64,
    /// Multiplier applied to the vol term during a spike (config exposure
    /// deferred).
    pub vol_spike_mult: f64,
}

impl Default for QuoteParams {
    fn default() -> Self {
        Self {
            min_edge: dec!(0.01),
            k1_vol_multiplier: 1.0,
            expected_hold_secs: 30.0,
            gamma_inventory_skew: 0.05,
            touch_size: shares(10),
            ladder_levels: 3,
            ladder_size_per_level: shares(10),
            ladder_tick_offset: 1,
            pair_cost_threshold: dec!(0.98),
            no_atm_final_secs: 25,
            no_passive_final_secs: 5,
            soft_cap_excess: shares(50),
            hard_cap_excess: shares(100),
            max_worst_case_loss: Dollars::new(Decimal::from(25)),
            cancel_rtt_secs: 0.2,
            atm_band: dec!(0.10),
            vol_spike_threshold_1s: 0.001,
            vol_spike_mult: 1.5,
        }
    }
}

/// The calculator's decision for one window: a desired quote set, or no quote.
///
/// A one-sided result (only the deficit side survives the caps/filters) is still
/// a [`QuoteDecision::Quote`]; [`QuoteDecision::NoQuote`] means *nothing* is
/// quoted.
#[derive(Debug, Clone, PartialEq)]
pub enum QuoteDecision {
    /// Quote the contained levels (one or both sides).
    Quote(QuoteSet),
    /// Do not quote at all this tick, with the reason.
    NoQuote(NoQuoteReason),
}

/// A complete desired quote set for one window: the Up-buy and Down-buy ladders
/// plus the diagnostics behind them.
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteSet {
    /// Window these quotes trade.
    pub window: WindowId,
    /// Surviving levels: the Up ladder (ascending `level`) then the Down ladder.
    pub levels: Vec<QuoteLevel>,
    /// The inventory-skewed center used (probability space, 6 dp diagnostic).
    pub center: Decimal,
    /// The half-spread used (probability space, 6 dp diagnostic).
    pub half_spread: Decimal,
    /// The model fair `p_up` this set was built from.
    pub p_up: f64,
    /// The normalized inventory imbalance (signed, clamped to `[−1, 1]`).
    pub normalized_imbalance: f64,
    /// Whether the explicit vol-spike multiplier engaged.
    pub vol_spike: bool,
    /// Sides/levels that were dropped, and why (diagnostics only — never turns a
    /// non-empty quote set into a no-quote).
    pub suppressed: Vec<Suppressed>,
}

/// One resting quote level in a ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuoteLevel {
    /// Outcome token being bid.
    pub outcome: Outcome,
    /// Order side (always [`Side::Buy`] for the passive maker).
    pub side: Side,
    /// On-grid limit price.
    pub price: Price,
    /// Order size in shares.
    pub size: Size,
    /// Ladder level index (`0` = touch).
    pub level: u32,
    /// Time-in-force (always post-only [`TimeInForce::Gtc`]).
    pub tif: TimeInForce,
}

impl QuoteLevel {
    /// A resting post-only BUY quote level — the only kind the passive
    /// calculator emits. Pins the `side == Buy` and `tif == Gtc { post_only:
    /// true }` invariant in one place.
    #[must_use]
    pub fn resting_buy(outcome: Outcome, price: Price, size: Size, level: u32) -> Self {
        Self {
            outcome,
            side: Side::Buy,
            price,
            size,
            level,
            tif: TimeInForce::Gtc { post_only: true },
        }
    }
}

/// A side or level that was dropped from the quote set, with the cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Suppressed {
    /// The outcome whose side/level was dropped.
    pub outcome: Outcome,
    /// The ladder level, or `None` when the whole side was dropped.
    pub level: Option<u32>,
    /// Why it was dropped.
    pub reason: SuppressReason,
}

/// Why a side or level was suppressed (diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressReason {
    /// The whole side was dropped: it holds the excess at or above the soft cap
    /// (quote the deficit side only).
    SoftCapExcessSide,
    /// The whole side was dropped: the per-window worst-case loss is breached, so
    /// we stop adding to the excess side.
    WorstCaseExcessSide,
    /// A level was dropped: adding it would push the pair cost above the
    /// threshold (§8 pair-cost discipline).
    PairCostExceeded,
    /// A level was dropped: its price is at-the-money in the final-seconds window.
    AtmFinalSeconds,
    /// A level was dropped: its snapped price collided with an already-accepted
    /// level on the same side.
    PriceCollision,
}

/// Why the calculator declined to quote at all.
///
/// `Copy` and data-carrying for the journal; not `Eq` (it carries `f64`s).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NoQuoteReason {
    /// The model is not [`ModelHealth::Ready`] — quoting is gated to Ready only
    /// (Degraded holds off adding risk; Unreliable forces a pull).
    #[error("model not ready: {health:?} ({reason:?})")]
    ModelNotReady {
        /// The model's health tier.
        health: ModelHealth,
        /// The cause behind that tier.
        reason: ModelHealthReason,
    },
    /// The model snapshot has no active window.
    #[error("model snapshot has no active window")]
    NoActiveWindow,
    /// The model's window and the inventory's window disagree.
    #[error("model/inventory window mismatch")]
    WindowMismatch {
        /// The model's window.
        model: Option<WindowId>,
        /// The inventory's window.
        inventory: WindowId,
    },
    /// Time-to-expiry is not positive (or not finite).
    #[error("time to expiry {tau_secs} is not positive")]
    Expired {
        /// The offending seconds-remaining.
        tau_secs: f64,
    },
    /// Inside the final-seconds window: no passive orders at all (cancel-all).
    #[error("final {tau_secs}s: no passive orders")]
    FinalSecondsNoPassive {
        /// The seconds-remaining that tripped the gate.
        tau_secs: f64,
    },
    /// The model fair `p_up` is not a usable probability (not strictly in `(0,1)`).
    #[error("model fair {p_up} is not a usable probability")]
    UnusableFair {
        /// The offending `p_up`.
        p_up: f64,
    },
    /// The model `σ_1s` is not a usable volatility (not positive/finite).
    #[error("sigma_1s {sigma_1s} is not a usable volatility")]
    UnusableVol {
        /// The offending `σ_1s`.
        sigma_1s: f64,
    },
    /// Inventory caps block all passive making (hard cap reached).
    #[error("inventory caps block all passive making")]
    InventoryBlocked,
    /// Every candidate level was filtered out (ATM / pair-cost / collision).
    #[error("all candidate levels were filtered out")]
    NoSurvivingLevels,
}

/// The side currently holding the unmatched excess, if any.
fn excess_side_of(inv: &InventorySnapshot) -> Option<Outcome> {
    if inv.excess > Decimal::ZERO {
        Some(Outcome::Up)
    } else if inv.excess < Decimal::ZERO {
        Some(Outcome::Down)
    } else {
        None
    }
}

/// Side-level suppression from the soft cap / worst-case bound (§8). The hard cap
/// is handled separately because it stops *both* sides.
fn side_cap_suppression(
    outcome: Outcome,
    constraint: ExcessConstraint,
    worst_breach: bool,
    excess_side: Option<Outcome>,
) -> Option<SuppressReason> {
    if excess_side != Some(outcome) {
        return None;
    }
    if matches!(constraint, ExcessConstraint::SoftCapped { .. }) {
        return Some(SuppressReason::SoftCapExcessSide);
    }
    if worst_breach {
        return Some(SuppressReason::WorstCaseExcessSide);
    }
    None
}

/// f64 → [`Decimal`] rounded to 6 dp for diagnostics (panic-free).
fn dec6(x: f64) -> Decimal {
    Decimal::from_f64_retain(x)
        .unwrap_or(Decimal::ZERO)
        .round_dp(6)
}

/// Snaps an f64 target *down* onto `tick`'s grid (passive-safe for a resting BUY).
///
/// The target is rounded to 9 dp first to absorb the noise of the f64 arithmetic
/// that produced it (a few subtractions of ~1-magnitude values accumulate
/// ~1e-16, far below the ≤4-dp grid), so a value *meant* to be on-grid — e.g.
/// `0.49` arriving as `0.489999999…` — is not silently pushed a tick low; it is
/// then quantized DOWN onto the grid by [`Price::quantize`] (which also clamps
/// into `[tick, 1−tick]`). Returns `None` for a non-finite or out-of-`[0,1]`
/// target so the level is skipped, never a panic.
fn price_down(target: f64, tick: TickSize) -> Option<Price> {
    if !target.is_finite() {
        return None;
    }
    let d = Decimal::from_f64_retain(target)?.round_dp(9);
    Price::quantize(d, tick, RoundDir::Down).ok()
}

/// Compute the complete desired resting maker quote set for one window (§8), or
/// an explicit no-quote decision.
///
/// Pure and sans-IO: no clock, no logging, no panic on any runtime path. The
/// inputs are exactly the §8 set — model state, inventory snapshot, fee
/// parameters, per-series config, seconds-remaining `tau_secs` (the caller
/// derives it from `close_time − now`; the calculator never reads a clock), and
/// the market's current `tick`. `fees` is part of the §8 signature; passive
/// maker quotes pay zero fees so it does not tighten the spread here (it is
/// reserved for the fee-aware taker modules).
#[must_use]
pub fn calculate_quotes(
    model: &ModelSnapshot,
    inv: &InventorySnapshot,
    fees: &FeeParams,
    params: &QuoteParams,
    tau_secs: f64,
    tick: TickSize,
) -> QuoteDecision {
    let _ = fees; // makers pay zero; kept for §8 signature completeness.

    // G1 — model-health gate: only a Ready model may drive quotes (§8). This is
    // the core adverse-selection defense: a stale/diverging model pulls quotes.
    if !model.health.allows_quoting() {
        return QuoteDecision::NoQuote(NoQuoteReason::ModelNotReady {
            health: model.health,
            reason: model.reason,
        });
    }

    // G2 — an active window we can attribute to, matching the inventory we skew
    // against.
    let Some(window) = model.window else {
        return QuoteDecision::NoQuote(NoQuoteReason::NoActiveWindow);
    };
    if window != inv.window {
        return QuoteDecision::NoQuote(NoQuoteReason::WindowMismatch {
            model: model.window,
            inventory: inv.window,
        });
    }

    // G3 — expiry.
    if !tau_secs.is_finite() || tau_secs <= 0.0 {
        return QuoteDecision::NoQuote(NoQuoteReason::Expired { tau_secs });
    }

    // G4 — final-seconds cancel-all: no passive orders at all (§8).
    if tau_secs <= f64::from(params.no_passive_final_secs) {
        return QuoteDecision::NoQuote(NoQuoteReason::FinalSecondsNoPassive { tau_secs });
    }

    // G5 — usable model inputs.
    let p_up = model.p_up;
    if !p_up.is_finite() || p_up <= 0.0 || p_up >= 1.0 {
        return QuoteDecision::NoQuote(NoQuoteReason::UnusableFair { p_up });
    }
    let sigma = model.sigma_1s;
    if !sigma.is_finite() || sigma <= 0.0 {
        return QuoteDecision::NoQuote(NoQuoteReason::UnusableVol { sigma_1s: sigma });
    }

    // S1 — normalized inventory imbalance (signed; ±1 at the hard cap).
    let hard_cap = params.hard_cap_excess.as_decimal().to_f64().unwrap_or(0.0);
    let excess = inv.excess.to_f64().unwrap_or(0.0);
    let norm_imbalance = if hard_cap > 0.0 {
        (excess / hard_cap).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // S2 — inventory-skewed center, clamped into the representable grid range.
    let tick_min = tick.min_price().to_f64().unwrap_or(0.0);
    let tick_max = tick.max_price().to_f64().unwrap_or(1.0);
    let center = (p_up - params.gamma_inventory_skew * norm_imbalance).clamp(tick_min, tick_max);

    // S3 — half-spread = max(min_edge, σ·(k1·√hold·spike_mult + √rtt)).
    let min_edge = params.min_edge.to_f64().unwrap_or(0.0);
    let latency_premium = sigma * params.cancel_rtt_secs.max(0.0).sqrt();
    let vol_spike = sigma >= params.vol_spike_threshold_1s;
    let spike_mult = if vol_spike {
        params.vol_spike_mult
    } else {
        1.0
    };
    let vol_term =
        params.k1_vol_multiplier * sigma * params.expected_hold_secs.max(0.0).sqrt() * spike_mult;
    let half_spread = min_edge.max(vol_term + latency_premium);

    // S4 — excess-cap eligibility. The hard cap stops *all* passive making (§8 /
    // ExcessConstraint doc: "stop passive making entirely"); working the excess
    // off at/above fair is a taker concern, not this calculator's.
    let constraint = excess_constraint_sides(
        &inv.up,
        &inv.down,
        params.soft_cap_excess,
        params.hard_cap_excess,
    );
    if let ExcessConstraint::HardCapped { .. } = constraint {
        return QuoteDecision::NoQuote(NoQuoteReason::InventoryBlocked);
    }
    let worst_breach = inv.worst_case_if_excess_loses > params.max_worst_case_loss;
    let excess_side = excess_side_of(inv);

    let tick_step = tick.as_decimal().to_f64().unwrap_or(0.0);
    let atm_active = tau_secs <= f64::from(params.no_atm_final_secs);
    let half = dec!(0.5);

    let mut suppressed: Vec<Suppressed> = Vec::new();
    let mut levels: Vec<QuoteLevel> = Vec::new();

    // S5..S8 — build each eligible side's ladder.
    for outcome in [Outcome::Up, Outcome::Down] {
        if let Some(reason) = side_cap_suppression(outcome, constraint, worst_breach, excess_side) {
            suppressed.push(Suppressed {
                outcome,
                level: None,
                reason,
            });
            continue;
        }

        let side_fair = if outcome == Outcome::Up {
            center
        } else {
            1.0 - center
        };
        let mut accepted: Vec<Decimal> = Vec::new();

        for level in 0..params.ladder_levels {
            let offset_ticks = params.ladder_tick_offset.saturating_mul(level);
            let target = side_fair - half_spread - f64::from(offset_ticks) * tick_step;
            let Some(price) = price_down(target, tick) else {
                continue; // off-domain target → skip this level
            };

            // S6 — final-seconds ATM gate: drop levels resting near 0.50.
            if atm_active && (price.as_decimal() - half).abs() <= params.atm_band {
                suppressed.push(Suppressed {
                    outcome,
                    level: Some(level),
                    reason: SuppressReason::AtmFinalSeconds,
                });
                continue;
            }

            let size = if level == 0 {
                params.touch_size
            } else {
                params.ladder_size_per_level
            };

            // S7 — pair-cost authorization.
            if !authorizes_passive_add_sides(
                &inv.up,
                &inv.down,
                outcome,
                price,
                size,
                params.pair_cost_threshold,
            ) {
                suppressed.push(Suppressed {
                    outcome,
                    level: Some(level),
                    reason: SuppressReason::PairCostExceeded,
                });
                continue;
            }

            // S8 — grid-collision dedupe within the side.
            if accepted.contains(&price.as_decimal()) {
                suppressed.push(Suppressed {
                    outcome,
                    level: Some(level),
                    reason: SuppressReason::PriceCollision,
                });
                continue;
            }

            accepted.push(price.as_decimal());
            levels.push(QuoteLevel::resting_buy(outcome, price, size, level));
        }
    }

    // S9 — assemble.
    if levels.is_empty() {
        return QuoteDecision::NoQuote(NoQuoteReason::NoSurvivingLevels);
    }
    QuoteDecision::Quote(QuoteSet {
        window,
        levels,
        center: dec6(center),
        half_spread: dec6(half_spread),
        p_up,
        normalized_imbalance: norm_imbalance,
        vol_spike,
        suppressed,
    })
}

#[cfg(test)]
mod tests {
    use core_types::{
        AnchorSource, Asset, DurationMs, InputAges, Series, SideInventory, TimestampMs,
        WindowDuration,
    };

    use super::*;

    // ---- fixtures --------------------------------------------------------

    const OPEN_MS: i64 = 1_781_000_000_000;
    /// The common tick grid for these markets (0.01).
    const TICK: TickSize = TickSize::T001;

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    /// A different window (different series) for the mismatch test.
    fn other_window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Eth,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }

    fn model_full(
        p_up: f64,
        sigma_1s: f64,
        health: ModelHealth,
        reason: ModelHealthReason,
        win: Option<WindowId>,
    ) -> ModelSnapshot {
        ModelSnapshot {
            asset: Asset::Btc,
            window: win,
            p_up,
            z: 0.0,
            sigma_1s,
            sigma_tau: sigma_1s, // unused by the calculator
            basis: 0.0,
            anchor: AnchorSource::BinanceCorrected,
            health,
            reason,
            input_ages: InputAges {
                chainlink: DurationMs::from_millis(0),
                binance: DurationMs::from_millis(0),
            },
            ts: TimestampMs::from_millis(OPEN_MS),
        }
    }

    /// A Ready model on `window()`.
    fn model_ready(p_up: f64, sigma_1s: f64) -> ModelSnapshot {
        model_full(
            p_up,
            sigma_1s,
            ModelHealth::Ready,
            ModelHealthReason::Nominal,
            Some(window()),
        )
    }

    fn fees() -> FeeParams {
        FeeParams {
            rate: dec!(0.07),
            exponent: 1,
            taker_only: true,
            rebate_rate: dec!(0.2),
            enabled: true,
        }
    }

    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }

    /// A side holding `shares` at average price `avg`.
    fn side(shares: Decimal, avg: Decimal) -> SideInventory {
        SideInventory {
            shares: sz(shares),
            cost: Dollars::new(shares * avg),
        }
    }

    fn inv(up: SideInventory, down: SideInventory) -> InventorySnapshot {
        InventorySnapshot::derive(window(), up, down, TimestampMs::from_millis(OPEN_MS))
    }

    fn empty_inv() -> InventorySnapshot {
        inv(SideInventory::default(), SideInventory::default())
    }

    /// Run the calculator on `TICK` with the standard fee params.
    fn run(m: &ModelSnapshot, i: &InventorySnapshot, p: &QuoteParams, tau: f64) -> QuoteDecision {
        calculate_quotes(m, i, &fees(), p, tau, TICK)
    }

    fn quotes(d: &QuoteDecision) -> &QuoteSet {
        match d {
            QuoteDecision::Quote(q) => q,
            QuoteDecision::NoQuote(r) => panic!("expected Quote, got NoQuote: {r:?}"),
        }
    }

    fn no_quote(d: &QuoteDecision) -> &NoQuoteReason {
        match d {
            QuoteDecision::NoQuote(r) => r,
            QuoteDecision::Quote(_) => panic!("expected NoQuote, got Quote"),
        }
    }

    /// On-grid prices for one side, in ladder order.
    fn prices(q: &QuoteSet, o: Outcome) -> Vec<Decimal> {
        q.levels
            .iter()
            .filter(|l| l.outcome == o)
            .map(|l| l.price.as_decimal())
            .collect()
    }

    /// Levels for one side, in ladder order.
    fn side_levels(q: &QuoteSet, o: Outcome) -> Vec<QuoteLevel> {
        q.levels
            .iter()
            .filter(|l| l.outcome == o)
            .copied()
            .collect()
    }

    fn level_count(q: &QuoteSet, o: Outcome) -> usize {
        q.levels.iter().filter(|l| l.outcome == o).count()
    }

    fn count_suppressed(q: &QuoteSet, o: Outcome, r: SuppressReason) -> usize {
        q.suppressed
            .iter()
            .filter(|s| s.outcome == o && s.reason == r)
            .count()
    }

    fn has_side_drop(q: &QuoteSet, o: Outcome, r: SuppressReason) -> bool {
        q.suppressed
            .iter()
            .any(|s| s.outcome == o && s.level.is_none() && s.reason == r)
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // ---- tests -----------------------------------------------------------

    /// 1. Balanced book, calm: symmetric two-sided ladders, min_edge binds.
    #[test]
    fn balanced_book_two_sided() {
        let d = run(
            &model_ready(0.50, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        // hs: vol 0.0002·√30≈0.001095 + lat 0.0002·√0.2≈0.0000894 ≈ 0.001185 < 0.01 ⇒ min_edge.
        assert_eq!(q.half_spread, dec!(0.01));
        assert_eq!(q.center, dec!(0.5));
        assert!(!q.vol_spike);
        // center 0.50 − 0.01 = 0.49, then a tick per level.
        assert_eq!(
            prices(q, Outcome::Up),
            vec![dec!(0.49), dec!(0.48), dec!(0.47)]
        );
        // Down fair = 1 − 0.50 = 0.50 ⇒ same ladder.
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.49), dec!(0.48), dec!(0.47)]
        );
        assert_eq!(q.levels.len(), 6);
        let up = side_levels(q, Outcome::Up);
        assert_eq!(up[0].size, sz(dec!(10))); // touch
        assert_eq!(up[1].size, sz(dec!(10))); // ladder
        assert!(q.suppressed.is_empty());
    }

    /// 2. min_edge binds at an off-center fair; the Down fair mirrors to 0.40.
    #[test]
    fn min_edge_binds() {
        let d = run(
            &model_ready(0.60, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        assert_eq!(q.half_spread, dec!(0.01)); // 0.001185 < 0.01
        // 0.60 − 0.01 = 0.59 ; Down fair 0.40 − 0.01 = 0.39.
        assert_eq!(
            prices(q, Outcome::Up),
            vec![dec!(0.59), dec!(0.58), dec!(0.57)]
        );
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.39), dec!(0.38), dec!(0.37)]
        );
    }

    /// 3. The vol term binds and widens the spread well past min_edge.
    #[test]
    fn vol_term_binds() {
        // Disable the explicit spike so the widening is purely the σ·√hold term.
        let params = QuoteParams {
            vol_spike_threshold_1s: 0.01,
            ..QuoteParams::default()
        };
        let d = run(&model_ready(0.50, 0.004), &empty_inv(), &params, 120.0);
        let q = quotes(&d);
        // vol 0.004·√30 ≈ 0.021909 + lat 0.004·√0.2 ≈ 0.001789 ≈ 0.023698 > 0.01.
        assert!(q.half_spread > dec!(0.02));
        assert!(!q.vol_spike);
        // 0.50 − 0.023698 = 0.476302 ⇒ rounds DOWN to 0.47.
        assert_eq!(
            prices(q, Outcome::Up),
            vec![dec!(0.47), dec!(0.46), dec!(0.45)]
        );
    }

    /// 4. The latency premium alone tips the spread over min_edge, moving the
    ///    touch one tick lower vs the zero-latency control.
    #[test]
    fn latency_premium_contribution() {
        let params = QuoteParams {
            vol_spike_threshold_1s: 0.01, // spike off
            ..QuoteParams::default()
        };
        // vol 0.0017·√30 ≈ 0.009311 < 0.01 alone; + lat 0.0017·√0.2 ≈ 0.000760 ⇒ 0.010072 > 0.01.
        let d = run(&model_ready(0.50, 0.0017), &empty_inv(), &params, 120.0);
        let q = quotes(&d);
        assert!(q.half_spread > dec!(0.01));
        // 0.50 − 0.010072 = 0.489928 ⇒ DOWN to 0.48.
        assert_eq!(prices(q, Outcome::Up)[0], dec!(0.48));

        // Control: zero cancel RTT ⇒ no latency premium ⇒ min_edge binds ⇒ touch 0.49.
        let control = QuoteParams {
            cancel_rtt_secs: 0.0,
            ..params
        };
        let dc = run(&model_ready(0.50, 0.0017), &empty_inv(), &control, 120.0);
        assert_eq!(prices(quotes(&dc), Outcome::Up)[0], dec!(0.49));
    }

    /// 5. Inventory skew: long Up bids Up lower and Down higher.
    #[test]
    fn inventory_skew_long_up() {
        // 70 Up @0.40, 30 Down @0.55 ⇒ excess +40 (< soft 50), worst-case 40·0.40=$16 < $25.
        let i = inv(side(dec!(70), dec!(0.40)), side(dec!(30), dec!(0.55)));
        let d = run(
            &model_ready(0.50, 0.0002),
            &i,
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        // norm = 40/100 = 0.4 ⇒ center = 0.50 − 0.05·0.4 = 0.48.
        assert!(approx(q.normalized_imbalance, 0.4));
        assert_eq!(q.center, dec!(0.48));
        // Up touch 0.48 − 0.01 = 0.47 (lower); Down fair 0.52 − 0.01 = 0.51 (higher).
        assert_eq!(
            prices(q, Outcome::Up),
            vec![dec!(0.47), dec!(0.46), dec!(0.45)]
        );
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.51), dec!(0.50), dec!(0.49)]
        );
    }

    /// 6. Soft cap: the excess side is silenced, the deficit side quotes alone.
    #[test]
    fn soft_cap_one_sided() {
        // 90 Up @0.40, 30 Down @0.55 ⇒ excess +60 (50 ≤ 60 < 100) ⇒ SoftCapped{Up}.
        let i = inv(side(dec!(90), dec!(0.40)), side(dec!(30), dec!(0.55)));
        let d = run(
            &model_ready(0.50, 0.0002),
            &i,
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        assert_eq!(level_count(q, Outcome::Up), 0);
        assert!(has_side_drop(
            q,
            Outcome::Up,
            SuppressReason::SoftCapExcessSide
        ));
        // norm = 0.6 ⇒ center 0.47 ⇒ Down fair 0.53 − 0.01 = 0.52.
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.52), dec!(0.51), dec!(0.50)]
        );
    }

    /// 7. Hard cap: all passive making stops entirely.
    #[test]
    fn hard_cap_silence() {
        // 130 Up @0.40, 20 Down @0.55 ⇒ excess +110 ≥ hard 100 ⇒ HardCapped.
        let i = inv(side(dec!(130), dec!(0.40)), side(dec!(20), dec!(0.55)));
        let d = run(
            &model_ready(0.50, 0.0002),
            &i,
            &QuoteParams::default(),
            120.0,
        );
        assert!(matches!(no_quote(&d), NoQuoteReason::InventoryBlocked));
    }

    /// 8. Worst-case breach (small but expensive excess) silences the excess
    ///    side while the deficit side keeps quoting to de-risk.
    #[test]
    fn worst_case_breach_silences_excess() {
        // 45 Up @0.40 (excess +45 < soft 50, Normal) but worst-case 45·0.40 = $18.
        let params = QuoteParams {
            max_worst_case_loss: Dollars::new(dec!(5)), // breached by $18
            ..QuoteParams::default()
        };
        let i = inv(side(dec!(45), dec!(0.40)), SideInventory::default());
        let d = run(&model_ready(0.50, 0.0002), &i, &params, 120.0);
        let q = quotes(&d);
        assert_eq!(level_count(q, Outcome::Up), 0);
        assert!(has_side_drop(
            q,
            Outcome::Up,
            SuppressReason::WorstCaseExcessSide
        ));
        // norm 0.45 ⇒ center 0.4775 ⇒ Down fair 0.5225 − 0.01 = 0.5125 ⇒ DOWN 0.51.
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.51), dec!(0.50), dec!(0.49)]
        );
    }

    /// 9. Pair-cost discipline rejects a whole side whose pairs are too dear.
    #[test]
    fn pair_cost_rejection_drops_side() {
        // 30 Down @0.55, 0 Up ⇒ excess −30 (Normal). Up adds form pairs ≈ price + 0.55 > 0.98.
        let i = inv(SideInventory::default(), side(dec!(30), dec!(0.55)));
        let d = run(
            &model_ready(0.50, 0.0002),
            &i,
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        // norm −0.3 ⇒ center 0.515 ⇒ Up touch 0.505→0.50 ⇒ pair 0.50+0.55=1.05 > 0.98 ⇒ reject all.
        assert_eq!(level_count(q, Outcome::Up), 0);
        assert_eq!(
            count_suppressed(q, Outcome::Up, SuppressReason::PairCostExceeded),
            3
        );
        // Down adds to a side whose opposite is empty ⇒ no pair to discipline ⇒ authorized.
        // Down fair 1 − 0.515 = 0.485 − 0.01 = 0.475 ⇒ DOWN 0.47.
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.47), dec!(0.46), dec!(0.45)]
        );
    }

    /// 10. Pair-cost allows a cheap add: both sides quote.
    #[test]
    fn pair_cost_allows_cheap_add() {
        // 30 Down @0.40, 0 Up ⇒ Up touch 0.50, pair 0.50+0.40 = 0.90 ≤ 0.98 ⇒ authorized.
        let i = inv(SideInventory::default(), side(dec!(30), dec!(0.40)));
        let d = run(
            &model_ready(0.50, 0.0002),
            &i,
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        // center 0.515 ⇒ Up fair 0.515 − 0.01 = 0.505 ⇒ DOWN 0.50.
        assert_eq!(
            prices(q, Outcome::Up),
            vec![dec!(0.50), dec!(0.49), dec!(0.48)]
        );
        assert!(level_count(q, Outcome::Down) > 0);
    }

    /// 11. Vol-spike widening: the explicit multiplier widens vs the no-spike
    ///     control at the same σ.
    #[test]
    fn vol_spike_widening() {
        // σ 0.0012 ≥ threshold 0.001 ⇒ spike, ×1.5 on the vol term.
        let d = run(
            &model_ready(0.50, 0.0012),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        assert!(q.vol_spike);
        assert!(q.half_spread > dec!(0.01)); // widened past min_edge

        // Control: lift the threshold so the same σ is NOT a spike ⇒ min_edge binds.
        let control = QuoteParams {
            vol_spike_threshold_1s: 0.01,
            ..QuoteParams::default()
        };
        let dc = run(&model_ready(0.50, 0.0012), &empty_inv(), &control, 120.0);
        let qc = quotes(&dc);
        assert!(!qc.vol_spike);
        assert_eq!(qc.half_spread, dec!(0.01));
    }

    /// 12. Final-seconds ATM gate: levels resting near 0.50 are dropped per-level.
    #[test]
    fn atm_final_seconds_suppresses() {
        let params = QuoteParams {
            atm_band: dec!(0.04),
            ..QuoteParams::default()
        };
        // p 0.55, τ 20 (≤ no_atm 25, > no_passive 5).
        let d = run(&model_ready(0.55, 0.0002), &empty_inv(), &params, 20.0);
        let q = quotes(&d);
        // Up ladder 0.54/0.53/0.52 all within 0.04 of 0.50 ⇒ all suppressed.
        assert_eq!(level_count(q, Outcome::Up), 0);
        assert_eq!(
            count_suppressed(q, Outcome::Up, SuppressReason::AtmFinalSeconds),
            3
        );
        // Down ladder 0.44/0.43/0.42 all > 0.04 from 0.50 ⇒ survive.
        assert_eq!(
            prices(q, Outcome::Down),
            vec![dec!(0.44), dec!(0.43), dec!(0.42)]
        );
    }

    /// 13. Final ~5s: cancel-all, no passive orders at all.
    #[test]
    fn no_passive_final_seconds() {
        let d = run(
            &model_ready(0.50, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            4.0,
        );
        assert!(matches!(
            no_quote(&d),
            NoQuoteReason::FinalSecondsNoPassive { .. }
        ));
    }

    /// 14. Unreliable model ⇒ pull quotes.
    #[test]
    fn unreliable_gating() {
        let m = model_full(
            0.50,
            0.0002,
            ModelHealth::Unreliable,
            ModelHealthReason::NoAnchor,
            Some(window()),
        );
        let d = run(&m, &empty_inv(), &QuoteParams::default(), 120.0);
        assert!(matches!(
            no_quote(&d),
            NoQuoteReason::ModelNotReady {
                health: ModelHealth::Unreliable,
                reason: ModelHealthReason::NoAnchor,
            }
        ));
    }

    /// 15. Degraded model ⇒ no new quotes (only Ready may quote).
    #[test]
    fn degraded_gating() {
        let m = model_full(
            0.50,
            0.0002,
            ModelHealth::Degraded,
            ModelHealthReason::FastLeadLost,
            Some(window()),
        );
        let d = run(&m, &empty_inv(), &QuoteParams::default(), 120.0);
        assert!(matches!(
            no_quote(&d),
            NoQuoteReason::ModelNotReady {
                health: ModelHealth::Degraded,
                ..
            }
        ));
    }

    /// 16. Canonical adverse-selection scenario: the underlying jumps up, and
    ///     the desired set reprices BOTH ladders away from their stale levels —
    ///     a diff against the resting book withdraws the offside quote rather
    ///     than refreshing it in place.
    #[test]
    fn adverse_selection_jump() {
        let calm = run(
            &model_ready(0.50, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        // Jump: p 0.50 → 0.85, σ spikes to 0.0015.
        let jump = run(
            &model_ready(0.85, 0.0015),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        let calm_q = quotes(&calm);
        let jump_q = quotes(&jump);

        // After the up-jump the Down (former Up-ask mirror) side is withdrawn
        // from its stale ~0.49 to far OTM.
        assert_eq!(
            prices(jump_q, Outcome::Up),
            vec![dec!(0.83), dec!(0.82), dec!(0.81)]
        );
        assert_eq!(
            prices(jump_q, Outcome::Down),
            vec![dec!(0.13), dec!(0.12), dec!(0.11)]
        );
        assert!(jump_q.vol_spike);

        // Not a single price survives the jump in place: every stale quote moves,
        // so the manager cancels, never refreshes-in-place.
        let calm_prices: std::collections::HashSet<Decimal> =
            calm_q.levels.iter().map(|l| l.price.as_decimal()).collect();
        assert!(
            jump_q
                .levels
                .iter()
                .all(|l| !calm_prices.contains(&l.price.as_decimal()))
        );
    }

    /// 17. Grid-collision dedupe: a zero tick offset collapses the ladder to one
    ///     price per side.
    #[test]
    fn grid_collision_dedupe() {
        let params = QuoteParams {
            ladder_tick_offset: 0,
            ..QuoteParams::default()
        };
        let d = run(&model_ready(0.50, 0.0002), &empty_inv(), &params, 120.0);
        let q = quotes(&d);
        assert_eq!(prices(q, Outcome::Up), vec![dec!(0.49)]); // one survives
        assert_eq!(
            count_suppressed(q, Outcome::Up, SuppressReason::PriceCollision),
            2
        );
        assert_eq!(
            count_suppressed(q, Outcome::Down, SuppressReason::PriceCollision),
            2
        );
    }

    /// 18. Tick snapping at the low and high extremes of the 0.01 grid.
    #[test]
    fn tick_snapping_extremes() {
        // Low: p 0.05, 80 Down @0.40 (SoftCapped{Down}) ⇒ Up quotes near the floor.
        let lo = inv(SideInventory::default(), side(dec!(80), dec!(0.40)));
        let dl = run(
            &model_ready(0.05, 0.0002),
            &lo,
            &QuoteParams::default(),
            120.0,
        );
        let ql = quotes(&dl);
        // norm −0.8 ⇒ center 0.09 ⇒ Up touch 0.08.
        assert_eq!(
            prices(ql, Outcome::Up),
            vec![dec!(0.08), dec!(0.07), dec!(0.06)]
        );
        assert_eq!(level_count(ql, Outcome::Down), 0);

        // High mirror: p 0.95, 80 Up @0.60 (SoftCapped{Up}) ⇒ Down quotes near the floor.
        let hi = inv(side(dec!(80), dec!(0.60)), SideInventory::default());
        let dh = run(
            &model_ready(0.95, 0.0002),
            &hi,
            &QuoteParams::default(),
            120.0,
        );
        let qh = quotes(&dh);
        // norm +0.8 ⇒ center 0.91 ⇒ Down fair 0.09 ⇒ Down touch 0.08.
        assert_eq!(
            prices(qh, Outcome::Down),
            vec![dec!(0.08), dec!(0.07), dec!(0.06)]
        );
        assert_eq!(level_count(qh, Outcome::Up), 0);
    }

    /// 19. Ladder geometry: touch size, per-level size, and tick spacing.
    #[test]
    fn ladder_sizes_and_spacing() {
        let params = QuoteParams {
            ladder_levels: 3,
            ladder_tick_offset: 2,
            touch_size: sz(dec!(10)),
            ladder_size_per_level: sz(dec!(20)),
            ..QuoteParams::default()
        };
        let d = run(&model_ready(0.50, 0.0002), &empty_inv(), &params, 120.0);
        let up = side_levels(quotes(&d), Outcome::Up);
        // 0.49 (touch, 10) ; 0.47 (level 1, 20) ; 0.45 (level 2, 20).
        assert_eq!(up.len(), 3);
        assert_eq!(
            (up[0].price.as_decimal(), up[0].size, up[0].level),
            (dec!(0.49), sz(dec!(10)), 0)
        );
        assert_eq!(
            (up[1].price.as_decimal(), up[1].size, up[1].level),
            (dec!(0.47), sz(dec!(20)), 1)
        );
        assert_eq!(
            (up[2].price.as_decimal(), up[2].size, up[2].level),
            (dec!(0.45), sz(dec!(20)), 2)
        );
    }

    /// 20. A saturated fair (p_up at the open domain bounds) is not quotable.
    #[test]
    fn p_up_extreme_unusable() {
        let hi = run(
            &model_ready(1.0, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        assert!(matches!(no_quote(&hi), NoQuoteReason::UnusableFair { .. }));
        let lo = run(
            &model_ready(0.0, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        assert!(matches!(no_quote(&lo), NoQuoteReason::UnusableFair { .. }));
    }

    /// 21. σ_1s gating: zero/non-finite is unusable; the configured floor quotes.
    #[test]
    fn sigma_gating() {
        let zero = run(
            &model_ready(0.50, 0.0),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        assert!(matches!(no_quote(&zero), NoQuoteReason::UnusableVol { .. }));
        let nan = run(
            &model_ready(0.50, f64::NAN),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        assert!(matches!(no_quote(&nan), NoQuoteReason::UnusableVol { .. }));
        // The vol floor (2e-5) is a valid, quotable volatility (min_edge binds).
        let floor = run(
            &model_ready(0.50, 0.00002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        assert!(matches!(floor, QuoteDecision::Quote(_)));
    }

    /// 22. Expiry gating: zero, negative, and NaN τ all decline.
    #[test]
    fn tau_zero_negative_nan() {
        for tau in [0.0, -5.0, f64::NAN] {
            let d = run(
                &model_ready(0.50, 0.0002),
                &empty_inv(),
                &QuoteParams::default(),
                tau,
            );
            assert!(
                matches!(no_quote(&d), NoQuoteReason::Expired { .. }),
                "tau {tau}"
            );
        }
    }

    /// 23. Window gating: no active window, and a model/inventory mismatch.
    #[test]
    fn window_mismatch_and_none() {
        let none = model_full(
            0.50,
            0.0002,
            ModelHealth::Ready,
            ModelHealthReason::Nominal,
            None,
        );
        let dn = run(&none, &empty_inv(), &QuoteParams::default(), 120.0);
        assert!(matches!(no_quote(&dn), NoQuoteReason::NoActiveWindow));

        let mismatched = model_full(
            0.50,
            0.0002,
            ModelHealth::Ready,
            ModelHealthReason::Nominal,
            Some(other_window()),
        );
        let dm = run(&mismatched, &empty_inv(), &QuoteParams::default(), 120.0);
        assert!(matches!(
            no_quote(&dm),
            NoQuoteReason::WindowMismatch { .. }
        ));
    }

    /// 24. Invariant guard: every emitted level is a post-only BUY on the grid,
    ///     at or above the venue share minimum.
    #[test]
    fn every_level_is_post_only_buy() {
        let d = run(
            &model_ready(0.50, 0.0002),
            &empty_inv(),
            &QuoteParams::default(),
            120.0,
        );
        let q = quotes(&d);
        for l in &q.levels {
            assert_eq!(l.side, Side::Buy);
            assert!(l.tif.is_post_only());
            assert!(matches!(l.outcome, Outcome::Up | Outcome::Down));
            assert!(l.price.is_on_grid(TICK));
            assert!(l.size.as_decimal() >= dec!(5));
        }
    }
}
