//! Per-series strategy/engine parameters (CLAUDE.md §8).
//!
//! Shape: one full [`EngineParams`] under `engine.defaults`, plus an optional
//! per-series table of [`SeriesOverride`]s, each carrying an enable flag and a
//! sparse [`EngineParamsPatch`] (every field optional). [`EngineConfig::resolved`]
//! merges a patch onto the defaults, so the operator overrides one knob for one
//! series without copying the other two dozen.
//!
//! Per-series overrides are TOML-only: environment keys are lowercased on the
//! way in, which can never match the case-sensitive series keys (`"BTC-5m"`).
//!
//! Type discipline (CLAUDE.md §2.6): money and probability thresholds are
//! `Decimal`-based newtypes; `f64` appears only for quantities consumed by the
//! float-space model math (vol, CDF inputs, skew coefficients).

use std::collections::BTreeMap;

use core_types::{Decimal, Dollars, DurationMs, Price, Series, Size, TickSize, WindowDuration};
use rust_decimal::dec;
use serde::{Deserialize, Serialize};

use crate::error::Violation;
use crate::validate::Violations;

/// Builds a [`Size`] from a non-negative integer literal (defaults only).
#[expect(
    clippy::unwrap_used,
    reason = "u32 literals are non-negative; Size::new rejects only negatives"
)]
fn shares(n: u32) -> Size {
    Size::new(Decimal::from(n)).unwrap()
}

/// The full parameter set one engine instance runs a series with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineParams {
    /// Minimum half-spread (probability space). Quoting never goes tighter
    /// than this; must be at least one base tick (0.01).
    pub min_edge: Decimal,
    /// `k1` in `half_spread = max(min_edge, k1 · σ_1s · √hold + latency premium)`.
    pub k1_vol_multiplier: f64,
    /// Expected passive holding time in seconds (the `√hold` horizon).
    pub expected_hold_secs: f64,
    /// `γ`: quote-center shift per unit of normalized inventory imbalance.
    pub gamma_inventory_skew: f64,
    /// Shares quoted at the touch (best level).
    pub touch_size: Size,
    /// Total ladder levels including the touch.
    pub ladder_levels: u32,
    /// Shares per ladder level behind the touch.
    pub ladder_size_per_level: Size,
    /// Tick gap between consecutive ladder levels.
    pub ladder_tick_offset: u32,
    /// Pair-cost discipline (§8): only add passively to a side if the
    /// post-trade `avg_up + avg_down` stays at or below this.
    pub pair_cost_threshold: Decimal,
    /// Merge policy (§8): request merging matched Up/Down pairs back to
    /// collateral once at least this many pairs have accumulated (capital
    /// recycling). The engine reads it via `InventoryParams.merge_min_pairs`.
    pub merge_min_pairs: Size,
    /// No at-the-money quoting once τ drops below this many seconds.
    pub no_atm_final_secs: u32,
    /// No passive orders at all (cancel-all) once τ drops below this.
    pub no_passive_final_secs: u32,
    /// `θ`: fair-value move (probability space) against a resting quote that
    /// triggers cancel-first repricing.
    pub reprice_threshold_theta: f64,
    /// Larger move that escalates to cancel-market instead of per-order cancels.
    pub cancel_market_theta: f64,
    /// Half-life of the EWMA over 1-second log returns (σ_1s estimator).
    pub ewma_half_life_secs: f64,
    /// Floor on σ_1s — the model never reports calmer than this.
    pub vol_floor_1s: f64,
    /// Cap on σ_1s — and never wilder than this.
    pub vol_cap_1s: f64,
    /// Extra model-vs-book edge (beyond the taker fee) required before the
    /// momentum taker fires, in probability space.
    pub taker_momentum_buffer: Decimal,
    /// Late-window taker activates once τ is at or below this many seconds.
    pub late_window_tau_secs: u32,
    /// …and `p_up` (Chainlink-anchored) is beyond this certainty.
    pub late_certainty_threshold: f64,
    /// Worst price the late-window taker will pay for the near-certain side.
    pub late_taker_price_cap: Price,
    /// Taker spend budget per window (both modules combined).
    pub taker_budget_per_window: Dollars,
    /// Cooldown between taker executions within one window.
    pub taker_cooldown_ms: DurationMs,
    /// Excess (unmatched) shares at which quoting narrows to the
    /// deficit-reducing side only.
    pub soft_cap_excess_shares: Size,
    /// Excess shares at which passive quoting stops entirely.
    pub hard_cap_excess_shares: Size,
    /// Maximum worst-case loss per window — the binding constraint on all
    /// sizing (§8).
    pub max_worst_case_loss_per_window: Dollars,
    /// Defense-in-depth clip cap (shares) enforced by the order normalizer:
    /// every outgoing share-sized order (and each taker FAK clip) is capped at
    /// this many shares. `0` (default) disables it. When set, `touch_size` and
    /// `ladder_size_per_level` must not exceed it (a maker level may never be
    /// bigger than the clip).
    pub clip_size_shares: Size,
    /// Per-window cumulative maker deployment budget (dollars of buy notional
    /// filled). Once a window's cumulative maker buy fills reach this, the quote
    /// manager stops OPENING new buys for that window (existing orders are still
    /// managed/cancelled). `0` (default) = unbounded (existing behavior).
    pub maker_deployment_budget_per_window: Dollars,
}

impl Default for EngineParams {
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
            merge_min_pairs: shares(25),
            no_atm_final_secs: 25,
            no_passive_final_secs: 5,
            reprice_threshold_theta: 0.005,
            cancel_market_theta: 0.02,
            ewma_half_life_secs: 60.0,
            vol_floor_1s: 0.00002,
            vol_cap_1s: 0.005,
            taker_momentum_buffer: dec!(0.005),
            late_window_tau_secs: 30,
            late_certainty_threshold: 0.97,
            late_taker_price_cap: default_late_price_cap(),
            taker_budget_per_window: Dollars::new(Decimal::from(10)),
            taker_cooldown_ms: DurationMs::from_millis(5_000),
            soft_cap_excess_shares: shares(50),
            hard_cap_excess_shares: shares(100),
            max_worst_case_loss_per_window: Dollars::new(Decimal::from(25)),
            clip_size_shares: Size::ZERO,
            maker_deployment_budget_per_window: Dollars::ZERO,
        }
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "0.99 lies in (0,1) with scale 2; pinned by defaults_resolve_valid test"
)]
fn default_late_price_cap() -> Price {
    Price::try_from(dec!(0.99)).unwrap()
}

impl EngineParams {
    /// Appends every §8 range/cross-field violation for these params, keyed
    /// under `prefix` (e.g. `"engine.series.BTC-5m"`), checked against the
    /// series' own window length.
    pub(crate) fn validate_into(&self, prefix: &str, window: WindowDuration, v: &mut Violations) {
        let key = |field: &str| format!("{prefix}.{field}");
        let window_secs = window.as_duration().as_millis() / 1_000;

        v.require(
            self.min_edge >= TickSize::T001.as_decimal() && self.min_edge < dec!(0.5),
            key("min_edge"),
            "must be in [0.01, 0.5) — a half-spread below one tick (0.01) is unquotable",
        );
        v.require(
            self.k1_vol_multiplier.is_finite() && self.k1_vol_multiplier >= 0.0,
            key("k1_vol_multiplier"),
            "must be finite and >= 0",
        );
        v.require(
            self.expected_hold_secs.is_finite() && self.expected_hold_secs > 0.0,
            key("expected_hold_secs"),
            "must be finite and > 0",
        );
        v.require(
            self.gamma_inventory_skew.is_finite() && self.gamma_inventory_skew >= 0.0,
            key("gamma_inventory_skew"),
            "must be finite and >= 0",
        );
        let min_shares = Decimal::from(5);
        v.require(
            self.touch_size.as_decimal() >= min_shares,
            key("touch_size"),
            "must be >= 5 shares (venue minimum resting order, §7)",
        );
        v.require(
            self.ladder_levels >= 1,
            key("ladder_levels"),
            "must be >= 1",
        );
        v.require(
            self.ladder_size_per_level.as_decimal() >= min_shares,
            key("ladder_size_per_level"),
            "must be >= 5 shares (venue minimum resting order, §7)",
        );
        v.require(
            self.ladder_tick_offset >= 1,
            key("ladder_tick_offset"),
            "must be >= 1",
        );
        v.require(
            self.pair_cost_threshold > Decimal::ZERO && self.pair_cost_threshold < Decimal::ONE,
            key("pair_cost_threshold"),
            "must be in (0, 1) — at 1.00 a matched pair can never lock in profit",
        );
        v.require(
            !self.merge_min_pairs.is_zero(),
            key("merge_min_pairs"),
            "must be > 0 — merging zero pairs is a no-op",
        );
        v.require(
            self.no_passive_final_secs <= self.no_atm_final_secs,
            key("no_passive_final_secs"),
            "must be <= no_atm_final_secs (the cancel-all window nests inside the no-ATM window)",
        );
        v.require(
            i64::from(self.no_atm_final_secs) <= window_secs,
            key("no_atm_final_secs"),
            "must not exceed the window length",
        );
        v.require(
            self.reprice_threshold_theta.is_finite() && self.reprice_threshold_theta > 0.0,
            key("reprice_threshold_theta"),
            "must be finite and > 0",
        );
        v.require(
            self.cancel_market_theta.is_finite()
                && self.cancel_market_theta >= self.reprice_threshold_theta,
            key("cancel_market_theta"),
            "must be finite and >= reprice_threshold_theta",
        );
        v.require(
            self.ewma_half_life_secs.is_finite() && self.ewma_half_life_secs > 0.0,
            key("ewma_half_life_secs"),
            "must be finite and > 0",
        );
        v.require(
            self.vol_floor_1s.is_finite() && self.vol_floor_1s > 0.0,
            key("vol_floor_1s"),
            "must be finite and > 0",
        );
        v.require(
            self.vol_cap_1s.is_finite() && self.vol_cap_1s > self.vol_floor_1s,
            key("vol_cap_1s"),
            "must be finite and > vol_floor_1s",
        );
        v.require(
            self.taker_momentum_buffer >= Decimal::ZERO,
            key("taker_momentum_buffer"),
            "must be >= 0",
        );
        let tau = i64::from(self.late_window_tau_secs);
        v.require(
            tau > 0 && tau < window_secs,
            key("late_window_tau_secs"),
            "must be in (0, window length)",
        );
        v.require(
            self.late_certainty_threshold.is_finite()
                && self.late_certainty_threshold > 0.5
                && self.late_certainty_threshold < 1.0,
            key("late_certainty_threshold"),
            "must be in (0.5, 1)",
        );
        v.require(
            self.late_taker_price_cap.as_decimal() > dec!(0.5),
            key("late_taker_price_cap"),
            "must be > 0.5 — the late taker buys the near-certain side",
        );
        v.require(
            self.taker_budget_per_window.as_decimal() >= Decimal::ZERO,
            key("taker_budget_per_window"),
            "must be >= 0",
        );
        v.require(
            self.taker_cooldown_ms.as_millis() >= 0,
            key("taker_cooldown_ms"),
            "must be >= 0",
        );
        v.require(
            !self.hard_cap_excess_shares.is_zero(),
            key("hard_cap_excess_shares"),
            "must be > 0",
        );
        v.require(
            self.soft_cap_excess_shares.as_decimal() <= self.hard_cap_excess_shares.as_decimal(),
            key("soft_cap_excess_shares"),
            "must be <= hard_cap_excess_shares",
        );
        v.require(
            self.max_worst_case_loss_per_window.as_decimal() > Decimal::ZERO,
            key("max_worst_case_loss_per_window"),
            "must be > 0 — this is the binding constraint on all sizing (§8)",
        );
        v.require(
            self.clip_size_shares.as_decimal() >= Decimal::ZERO,
            key("clip_size_shares"),
            "must be >= 0 (0 disables the clip)",
        );
        if !self.clip_size_shares.is_zero() {
            v.require(
                self.touch_size.as_decimal() <= self.clip_size_shares.as_decimal(),
                key("clip_size_shares"),
                "must be >= touch_size — a maker level may never exceed the clip",
            );
            v.require(
                self.ladder_size_per_level.as_decimal() <= self.clip_size_shares.as_decimal(),
                key("clip_size_shares"),
                "must be >= ladder_size_per_level — a maker level may never exceed the clip",
            );
        }
        v.require(
            self.maker_deployment_budget_per_window.as_decimal() >= Decimal::ZERO,
            key("maker_deployment_budget_per_window"),
            "must be >= 0 (0 disables the budget)",
        );
    }
}

/// A sparse override of [`EngineParams`]: only the set fields replace the
/// defaults. Field meanings and units are identical to [`EngineParams`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[expect(
    missing_docs,
    reason = "each field mirrors the EngineParams field of the same name"
)]
pub struct EngineParamsPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_edge: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k1_vol_multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_hold_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gamma_inventory_skew: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub touch_size: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ladder_levels: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ladder_size_per_level: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ladder_tick_offset: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair_cost_threshold: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_min_pairs: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_atm_final_secs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_passive_final_secs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reprice_threshold_theta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel_market_theta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ewma_half_life_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vol_floor_1s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vol_cap_1s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub taker_momentum_buffer: Option<Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub late_window_tau_secs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub late_certainty_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub late_taker_price_cap: Option<Price>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub taker_budget_per_window: Option<Dollars>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub taker_cooldown_ms: Option<DurationMs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft_cap_excess_shares: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hard_cap_excess_shares: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_worst_case_loss_per_window: Option<Dollars>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_size_shares: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maker_deployment_budget_per_window: Option<Dollars>,
}

impl EngineParamsPatch {
    /// Merges this patch onto `base`: set fields win, unset fields keep the
    /// base value.
    #[must_use]
    pub fn apply(&self, base: &EngineParams) -> EngineParams {
        EngineParams {
            min_edge: self.min_edge.unwrap_or(base.min_edge),
            k1_vol_multiplier: self.k1_vol_multiplier.unwrap_or(base.k1_vol_multiplier),
            expected_hold_secs: self.expected_hold_secs.unwrap_or(base.expected_hold_secs),
            gamma_inventory_skew: self
                .gamma_inventory_skew
                .unwrap_or(base.gamma_inventory_skew),
            touch_size: self.touch_size.unwrap_or(base.touch_size),
            ladder_levels: self.ladder_levels.unwrap_or(base.ladder_levels),
            ladder_size_per_level: self
                .ladder_size_per_level
                .unwrap_or(base.ladder_size_per_level),
            ladder_tick_offset: self.ladder_tick_offset.unwrap_or(base.ladder_tick_offset),
            pair_cost_threshold: self.pair_cost_threshold.unwrap_or(base.pair_cost_threshold),
            merge_min_pairs: self.merge_min_pairs.unwrap_or(base.merge_min_pairs),
            no_atm_final_secs: self.no_atm_final_secs.unwrap_or(base.no_atm_final_secs),
            no_passive_final_secs: self
                .no_passive_final_secs
                .unwrap_or(base.no_passive_final_secs),
            reprice_threshold_theta: self
                .reprice_threshold_theta
                .unwrap_or(base.reprice_threshold_theta),
            cancel_market_theta: self.cancel_market_theta.unwrap_or(base.cancel_market_theta),
            ewma_half_life_secs: self.ewma_half_life_secs.unwrap_or(base.ewma_half_life_secs),
            vol_floor_1s: self.vol_floor_1s.unwrap_or(base.vol_floor_1s),
            vol_cap_1s: self.vol_cap_1s.unwrap_or(base.vol_cap_1s),
            taker_momentum_buffer: self
                .taker_momentum_buffer
                .unwrap_or(base.taker_momentum_buffer),
            late_window_tau_secs: self
                .late_window_tau_secs
                .unwrap_or(base.late_window_tau_secs),
            late_certainty_threshold: self
                .late_certainty_threshold
                .unwrap_or(base.late_certainty_threshold),
            late_taker_price_cap: self
                .late_taker_price_cap
                .unwrap_or(base.late_taker_price_cap),
            taker_budget_per_window: self
                .taker_budget_per_window
                .unwrap_or(base.taker_budget_per_window),
            taker_cooldown_ms: self.taker_cooldown_ms.unwrap_or(base.taker_cooldown_ms),
            soft_cap_excess_shares: self
                .soft_cap_excess_shares
                .unwrap_or(base.soft_cap_excess_shares),
            hard_cap_excess_shares: self
                .hard_cap_excess_shares
                .unwrap_or(base.hard_cap_excess_shares),
            max_worst_case_loss_per_window: self
                .max_worst_case_loss_per_window
                .unwrap_or(base.max_worst_case_loss_per_window),
            clip_size_shares: self.clip_size_shares.unwrap_or(base.clip_size_shares),
            maker_deployment_budget_per_window: self
                .maker_deployment_budget_per_window
                .unwrap_or(base.maker_deployment_budget_per_window),
        }
    }
}

// ----------------------------------------------------------------------------
// Runtime safe-list (control plane, §10.6/§11)
// ----------------------------------------------------------------------------

/// The [`EngineParams`] fields the control plane may adjust at runtime — the
/// spreads, buffers, and budgets (CLAUDE.md "define the safe list in config").
/// Every other field is **structural** ([`STRUCTURAL_PARAM_KEYS`]) and requires
/// a restart: touch/ladder sizing changes the order structure, the final-seconds
/// windows are window-length-coupled, the excess caps gate sizing, and
/// `max_worst_case_loss_per_window` is the §8 risk limit — none safe to mutate
/// under a live book without a clean restart.
pub const SAFE_PARAM_KEYS: &[&str] = &[
    "min_edge",
    "k1_vol_multiplier",
    "expected_hold_secs",
    "gamma_inventory_skew",
    "pair_cost_threshold",
    "merge_min_pairs",
    "reprice_threshold_theta",
    "cancel_market_theta",
    "ewma_half_life_secs",
    "vol_floor_1s",
    "vol_cap_1s",
    "taker_momentum_buffer",
    "late_certainty_threshold",
    "late_taker_price_cap",
    "taker_budget_per_window",
    "taker_cooldown_ms",
];

/// The [`EngineParams`] fields that exist but require a restart to change. A
/// `set-param` on one of these is rejected with a "requires a restart" message
/// rather than silently ignored. Together with [`SAFE_PARAM_KEYS`] this covers
/// every [`EngineParams`] field (pinned by a test).
const STRUCTURAL_PARAM_KEYS: &[&str] = &[
    "touch_size",
    "ladder_levels",
    "ladder_size_per_level",
    "ladder_tick_offset",
    "no_atm_final_secs",
    "no_passive_final_secs",
    "late_window_tau_secs",
    "soft_cap_excess_shares",
    "hard_cap_excess_shares",
    "max_worst_case_loss_per_window",
    "clip_size_shares",
    "maker_deployment_budget_per_window",
];

/// A validated runtime parameter change: the safe-list `key` and the
/// single-field [`EngineParamsPatch`] that applies it.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedParamChange {
    /// The safe-list field name that changed.
    pub key: String,
    /// The sparse patch carrying exactly that one field.
    pub patch: EngineParamsPatch,
}

/// Validates a runtime parameter change against the safe-list and the §8 range
/// rules, returning the typed single-field patch when it is accepted.
///
/// `base` is the series' current effective [`EngineParams`] (the resolved config
/// plus any prior runtime overrides), `key`/`value` are the requested change as
/// strings (the HTTP/CLI boundary), and `window` is the series' window length
/// (some §8 rules are window-relative). The same [`EngineParams::validate_into`]
/// rules the config loader runs are reused, so a runtime change can never push a
/// series outside what a fresh config would accept.
///
/// # Errors
/// A single [`Violation`] when the key is unknown, structural (requires a
/// restart), the value does not parse into the field's type, or the resulting
/// params would break a range or cross-field rule.
pub fn validate_param_change(
    base: &EngineParams,
    key: &str,
    value: &str,
    window: WindowDuration,
) -> Result<ParsedParamChange, Violation> {
    let vkey = format!("param.{key}");
    let reject = |message: String| Violation {
        key: vkey.clone(),
        message,
    };

    if !SAFE_PARAM_KEYS.contains(&key) {
        let message = if STRUCTURAL_PARAM_KEYS.contains(&key) {
            format!(
                "`{key}` is a structural/risk parameter — it requires a restart, \
                 not a runtime change"
            )
        } else {
            format!("unknown engine parameter `{key}` — not in the runtime-tunable safe list")
        };
        return Err(reject(message));
    }

    let patch = parse_param_patch(key, value.trim()).map_err(reject)?;
    let candidate = patch.apply(base);
    if let Some(v) = introduced_violations(base, &candidate, window)
        .into_iter()
        .next()
    {
        return Err(v);
    }
    Ok(ParsedParamChange {
        key: key.to_owned(),
        patch,
    })
}

/// Builds a single-field [`EngineParamsPatch`] from a safe-list `key` and a
/// string `value`, parsing it into the field's type.
fn parse_param_patch(key: &str, value: &str) -> Result<EngineParamsPatch, String> {
    let base = EngineParamsPatch::default();
    let patch = match key {
        "min_edge" => EngineParamsPatch {
            min_edge: Some(parse_decimal(value)?),
            ..base
        },
        "k1_vol_multiplier" => EngineParamsPatch {
            k1_vol_multiplier: Some(parse_f64(value)?),
            ..base
        },
        "expected_hold_secs" => EngineParamsPatch {
            expected_hold_secs: Some(parse_f64(value)?),
            ..base
        },
        "gamma_inventory_skew" => EngineParamsPatch {
            gamma_inventory_skew: Some(parse_f64(value)?),
            ..base
        },
        "pair_cost_threshold" => EngineParamsPatch {
            pair_cost_threshold: Some(parse_decimal(value)?),
            ..base
        },
        "merge_min_pairs" => EngineParamsPatch {
            merge_min_pairs: Some(parse_size(value)?),
            ..base
        },
        "reprice_threshold_theta" => EngineParamsPatch {
            reprice_threshold_theta: Some(parse_f64(value)?),
            ..base
        },
        "cancel_market_theta" => EngineParamsPatch {
            cancel_market_theta: Some(parse_f64(value)?),
            ..base
        },
        "ewma_half_life_secs" => EngineParamsPatch {
            ewma_half_life_secs: Some(parse_f64(value)?),
            ..base
        },
        "vol_floor_1s" => EngineParamsPatch {
            vol_floor_1s: Some(parse_f64(value)?),
            ..base
        },
        "vol_cap_1s" => EngineParamsPatch {
            vol_cap_1s: Some(parse_f64(value)?),
            ..base
        },
        "taker_momentum_buffer" => EngineParamsPatch {
            taker_momentum_buffer: Some(parse_decimal(value)?),
            ..base
        },
        "late_certainty_threshold" => EngineParamsPatch {
            late_certainty_threshold: Some(parse_f64(value)?),
            ..base
        },
        "late_taker_price_cap" => EngineParamsPatch {
            late_taker_price_cap: Some(parse_price(value)?),
            ..base
        },
        "taker_budget_per_window" => EngineParamsPatch {
            taker_budget_per_window: Some(parse_dollars(value)?),
            ..base
        },
        "taker_cooldown_ms" => EngineParamsPatch {
            taker_cooldown_ms: Some(parse_duration_ms(value)?),
            ..base
        },
        // Unreachable: the caller gates on SAFE_PARAM_KEYS first.
        other => return Err(format!("`{other}` is not a runtime-tunable parameter")),
    };
    Ok(patch)
}

fn parse_decimal(value: &str) -> Result<Decimal, String> {
    value
        .parse::<Decimal>()
        .map_err(|e| format!("not a valid decimal number: {e}"))
}

fn parse_f64(value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|e| format!("not a valid number: {e}"))
}

fn parse_size(value: &str) -> Result<Size, String> {
    Size::new(parse_decimal(value)?).map_err(|e| format!("invalid share count: {e}"))
}

fn parse_price(value: &str) -> Result<Price, String> {
    Price::try_from(parse_decimal(value)?).map_err(|e| format!("invalid price: {e}"))
}

fn parse_dollars(value: &str) -> Result<Dollars, String> {
    Ok(Dollars::new(parse_decimal(value)?))
}

fn parse_duration_ms(value: &str) -> Result<DurationMs, String> {
    let ms: i64 = value
        .parse()
        .map_err(|e| format!("not a valid integer number of milliseconds: {e}"))?;
    Ok(DurationMs::from_millis(ms))
}

/// The §8 violations a candidate introduces over `base` — i.e. those present
/// after the change but not before — so a pre-existing (unrelated) violation in
/// `base` is never attributed to this change.
fn introduced_violations(
    base: &EngineParams,
    candidate: &EngineParams,
    window: WindowDuration,
) -> Vec<Violation> {
    let before = collect_violations(base, window);
    let after = collect_violations(candidate, window);
    after.into_iter().filter(|v| !before.contains(v)).collect()
}

fn collect_violations(p: &EngineParams, window: WindowDuration) -> Vec<Violation> {
    let mut v = Violations::default();
    p.validate_into("param", window, &mut v);
    v.into_result().err().unwrap_or_default()
}

/// One series' entry in the override table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SeriesOverride {
    /// Whether this series trades at all. Series absent from the table are
    /// enabled with defaults.
    pub enabled: bool,
    /// Sparse parameter overrides for this series.
    pub params: EngineParamsPatch,
}

impl Default for SeriesOverride {
    fn default() -> Self {
        Self {
            enabled: true,
            params: EngineParamsPatch::default(),
        }
    }
}

/// The full engine section: shared defaults plus per-series overrides.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Parameters every series starts from.
    pub defaults: EngineParams,
    /// Per-series enable flags and sparse overrides, keyed by series key
    /// (e.g. `[engine.series.BTC-5m]`). Empty means: all six series enabled
    /// with defaults.
    pub series: BTreeMap<Series, SeriesOverride>,
}

impl EngineConfig {
    /// The effective parameters for `series`: the per-series patch (if any)
    /// applied onto the defaults.
    #[must_use]
    pub fn resolved(&self, series: Series) -> EngineParams {
        match self.series.get(&series) {
            Some(over) => over.params.apply(&self.defaults),
            None => self.defaults.clone(),
        }
    }

    /// True unless the series is explicitly disabled in the override table.
    #[must_use]
    pub fn is_enabled(&self, series: Series) -> bool {
        self.series.get(&series).is_none_or(|over| over.enabled)
    }

    /// All enabled series in stable dashboard order.
    #[must_use]
    pub fn enabled_series(&self) -> Vec<Series> {
        Series::ALL
            .into_iter()
            .filter(|s| self.is_enabled(*s))
            .collect()
    }

    /// Validates the **resolved** params of all six series — disabled ones
    /// too, so a broken override can never lurk until someone re-enables it.
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        for series in Series::ALL {
            let prefix = format!("engine.series.{}", series.key());
            self.resolved(series)
                .validate_into(&prefix, series.duration, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::Asset;
    use serde_json::json;

    use super::*;

    fn series(key: &str) -> Series {
        key.parse().unwrap()
    }

    #[test]
    fn defaults_resolve_valid_for_all_six_series() {
        let mut v = Violations::default();
        EngineConfig::default().validate_into(&mut v);
        assert_eq!(v.into_result(), Ok(()));
    }

    #[test]
    fn patch_overrides_only_named_fields() {
        let patch = EngineParamsPatch {
            min_edge: Some(dec!(0.02)),
            ..EngineParamsPatch::default()
        };
        let base = EngineParams::default();
        let merged = patch.apply(&base);
        assert_eq!(merged.min_edge, dec!(0.02));
        assert_eq!(
            EngineParams {
                min_edge: dec!(0.02),
                ..base
            },
            merged
        );
    }

    #[test]
    fn resolved_and_enabled_series() {
        let mut cfg = EngineConfig::default();
        cfg.series.insert(
            series("BTC-5m"),
            SeriesOverride {
                enabled: false,
                ..SeriesOverride::default()
            },
        );
        cfg.series.insert(
            series("ETH-1h"),
            SeriesOverride {
                params: EngineParamsPatch {
                    min_edge: Some(dec!(0.03)),
                    ..EngineParamsPatch::default()
                },
                ..SeriesOverride::default()
            },
        );

        assert_eq!(cfg.resolved(series("ETH-1h")).min_edge, dec!(0.03));
        assert_eq!(cfg.resolved(series("BTC-15m")), EngineParams::default());

        let enabled = cfg.enabled_series();
        assert_eq!(enabled.len(), 5);
        assert!(!enabled.contains(&series("BTC-5m")));
        assert!(enabled.contains(&series("ETH-1h")));
    }

    #[test]
    fn gate_ordering_and_window_length_rules() {
        // no_passive > no_atm: nesting violated.
        let mut cfg = EngineConfig::default();
        cfg.defaults.no_atm_final_secs = 25;
        cfg.defaults.no_passive_final_secs = 30;
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert!(
            violations
                .iter()
                .any(|x| x.key.ends_with("no_passive_final_secs"))
        );

        // no_atm 400s exceeds the 5m (300s) windows but fits 15m/1h: exactly
        // the two 5-minute series must flag it.
        let mut cfg = EngineConfig::default();
        cfg.defaults.no_atm_final_secs = 400;
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        let atm: Vec<&str> = violations
            .iter()
            .filter(|x| x.key.ends_with("no_atm_final_secs"))
            .map(|x| x.key.as_str())
            .collect();
        assert_eq!(
            atm,
            vec![
                "engine.series.BTC-5m.no_atm_final_secs",
                "engine.series.ETH-5m.no_atm_final_secs"
            ]
        );
    }

    #[test]
    fn range_rules_reject_nonsense() {
        let mut cfg = EngineConfig::default();
        cfg.defaults.min_edge = dec!(0.005); // below one tick
        cfg.defaults.vol_floor_1s = 0.01;
        cfg.defaults.vol_cap_1s = 0.005; // cap below floor
        cfg.defaults.touch_size = shares(3); // below venue minimum
        cfg.defaults.soft_cap_excess_shares = shares(200); // above hard cap
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        for field in [
            "min_edge",
            "vol_cap_1s",
            "touch_size",
            "soft_cap_excess_shares",
        ] {
            assert!(
                violations
                    .iter()
                    .any(|x| x.key == format!("engine.series.BTC-5m.{field}")),
                "expected a BTC-5m violation for {field}"
            );
        }
    }

    #[test]
    fn merge_min_pairs_must_be_positive() {
        let mut cfg = EngineConfig::default();
        cfg.defaults.merge_min_pairs = Size::ZERO;
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert!(
            violations
                .iter()
                .any(|x| x.key == "engine.series.BTC-5m.merge_min_pairs"),
            "expected a merge_min_pairs violation"
        );
    }

    #[test]
    fn clip_size_smaller_than_a_maker_level_is_rejected() {
        // clip 8, but the ladder places 10-share levels ⇒ a level would exceed
        // the clip. touch_size 10 also exceeds it.
        let mut cfg = EngineConfig::default();
        cfg.defaults.clip_size_shares = shares(8);
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        let violations = v.into_result().unwrap_err();
        assert!(
            violations
                .iter()
                .any(|x| x.key == "engine.series.BTC-5m.clip_size_shares"),
            "a clip below a maker level must be rejected"
        );
    }

    #[test]
    fn clip_zero_disables_the_constraint() {
        // The default clip is 0 (disabled), so the 10-share ladder is fine.
        let mut v = Violations::default();
        EngineConfig::default().validate_into(&mut v);
        assert_eq!(v.into_result(), Ok(()));
    }

    #[test]
    fn patch_covers_every_params_field() {
        // Anti-drift: deserializing a full EngineParams JSON into the patch
        // must succeed (deny_unknown_fields ⇒ Params ⊆ Patch) and populate
        // every key (⇒ Patch ⊇ Params, since None fields are skipped).
        let params_json = serde_json::to_value(EngineParams::default()).unwrap();
        let full_patch: EngineParamsPatch = serde_json::from_value(params_json.clone()).unwrap();
        let patch_json = serde_json::to_value(&full_patch).unwrap();
        let params_keys: Vec<&String> = params_json.as_object().unwrap().keys().collect();
        let patch_keys: Vec<&String> = patch_json.as_object().unwrap().keys().collect();
        assert_eq!(params_keys, patch_keys);
    }

    #[test]
    fn series_table_accepts_known_keys_and_rejects_unknown() {
        let cfg: EngineConfig = serde_json::from_value(json!({
            "series": { "BTC-5m": { "enabled": false } }
        }))
        .unwrap();
        assert!(!cfg.is_enabled(Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        }));

        let err = serde_json::from_value::<EngineConfig>(json!({
            "series": { "XRP-5m": { "enabled": false } }
        }))
        .unwrap_err();
        assert!(err.to_string().contains("XRP-5m"));
    }

    // ---- runtime safe-list (control plane) ---------------------------------

    #[test]
    fn safe_and_structural_keys_cover_every_params_field_exactly() {
        // Anti-drift: every EngineParams field is in exactly one of the two
        // lists, and the lists are disjoint — so a new field forces a decision.
        let params_json = serde_json::to_value(EngineParams::default()).unwrap();
        let mut all: Vec<String> = params_json.as_object().unwrap().keys().cloned().collect();
        all.sort();
        let mut listed: Vec<String> = SAFE_PARAM_KEYS
            .iter()
            .chain(STRUCTURAL_PARAM_KEYS.iter())
            .map(|s| (*s).to_owned())
            .collect();
        listed.sort();
        assert_eq!(all, listed, "safe ∪ structural must cover every field once");
        for key in SAFE_PARAM_KEYS {
            assert!(
                !STRUCTURAL_PARAM_KEYS.contains(key),
                "{key} cannot be both safe and structural"
            );
        }
    }

    #[test]
    fn validate_param_change_accepts_in_range() {
        let base = EngineParams::default();
        let change = validate_param_change(&base, "min_edge", "0.02", WindowDuration::M5).unwrap();
        assert_eq!(change.key, "min_edge");
        assert_eq!(change.patch.apply(&base).min_edge, dec!(0.02));
        // A budget and a buffer too.
        assert!(
            validate_param_change(&base, "taker_budget_per_window", "25", WindowDuration::M5)
                .is_ok()
        );
        assert!(
            validate_param_change(&base, "taker_momentum_buffer", "0.01", WindowDuration::H1)
                .is_ok()
        );
    }

    #[test]
    fn validate_param_change_rejects_out_of_range() {
        let base = EngineParams::default();
        // Below one tick.
        let err =
            validate_param_change(&base, "min_edge", "0.005", WindowDuration::M5).unwrap_err();
        assert_eq!(err.key, "param.min_edge");
        // A negative budget.
        assert!(
            validate_param_change(&base, "taker_budget_per_window", "-5", WindowDuration::M5)
                .is_err()
        );
    }

    #[test]
    fn validate_param_change_enforces_cross_field_rules() {
        // Raising reprice above the existing cancel-market theta trips the
        // cross-field rule, reported on the cancel_market_theta key (not the
        // changed key) — the diff-against-base approach catches it.
        let base = EngineParams::default(); // reprice 0.005, cancel 0.02
        let err =
            validate_param_change(&base, "reprice_threshold_theta", "0.05", WindowDuration::M5)
                .unwrap_err();
        assert_eq!(err.key, "param.cancel_market_theta");
        assert!(err.message.contains("reprice_threshold_theta"));
    }

    #[test]
    fn validate_param_change_rejects_structural_and_risk_keys() {
        let base = EngineParams::default();
        for key in [
            "touch_size",
            "no_passive_final_secs",
            "max_worst_case_loss_per_window",
        ] {
            let err = validate_param_change(&base, key, "20", WindowDuration::M5).unwrap_err();
            assert!(
                err.message.contains("requires a restart"),
                "{key}: {}",
                err.message
            );
        }
    }

    #[test]
    fn validate_param_change_rejects_unknown_key_and_unparseable_value() {
        let base = EngineParams::default();
        let unknown =
            validate_param_change(&base, "nonsense", "1", WindowDuration::M5).unwrap_err();
        assert!(unknown.message.contains("unknown engine parameter"));

        let bad = validate_param_change(&base, "min_edge", "not-a-number", WindowDuration::M5)
            .unwrap_err();
        assert_eq!(bad.key, "param.min_edge");
        assert!(bad.message.contains("decimal"));

        let bad_ms = validate_param_change(&base, "taker_cooldown_ms", "3.5", WindowDuration::M5)
            .unwrap_err();
        assert!(bad_ms.message.contains("milliseconds"));
    }
}
