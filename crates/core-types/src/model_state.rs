//! Model output types. Floats are legal here by design (CLAUDE.md §2.6):
//! the fair-value math runs in `f64` and converts to decimals only at the
//! quoting boundary via [`crate::Price::from_f64`]. The bus-facing
//! [`ModelSnapshot`] stays `Copy`-small; the model crate keeps its richer
//! per-window state to itself.

use serde::{Deserialize, Serialize};

use crate::market::WindowId;
use crate::series::Asset;
use crate::time::{DurationMs, TimestampMs};

/// Composite health gate of the fair-value model for one asset (§8). Three
/// tiers the future engine consumes directly:
///
/// - [`Ready`](ModelHealth::Ready): inputs fresh and coherent — **the only
///   state that may quote** ([`allows_quoting`](ModelHealth::allows_quoting)).
/// - [`Degraded`](ModelHealth::Degraded): still priceable, but a soft concern
///   holds (the fast Binance lead was lost so we anchor Chainlink-only and lose
///   the adverse-selection defense, or the basis/divergence sits in a sustained
///   soft band). The engine should not place **new** quotes, but need not pull
///   resting ones.
/// - [`Unreliable`](ModelHealth::Unreliable): a required input is stale, there
///   is no usable anchor, divergence is sustained-and-large, or the model has
///   not warmed up yet — **the engine must pull quotes**
///   ([`requires_pull`](ModelHealth::requires_pull)).
///
/// The variants are ordered worst-ascending (`Ready < Degraded < Unreliable`)
/// so the hysteresis machine can compare tiers as "is this an improvement".
/// The precise cause is carried separately in [`ModelHealthReason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ModelHealth {
    /// Inputs fresh and coherent — quoting allowed.
    Ready,
    /// Priceable but a soft-quality concern holds — hold off adding risk.
    Degraded,
    /// Stale/diverging/warming — the engine must pull quotes (§8).
    Unreliable,
}

impl ModelHealth {
    /// Only a [`ModelHealth::Ready`] model may drive quotes.
    #[must_use]
    pub const fn allows_quoting(self) -> bool {
        matches!(self, Self::Ready)
    }

    /// A [`ModelHealth::Unreliable`] model forces the engine to cancel resting
    /// quotes (§8 cancel-first / pull-quotes).
    #[must_use]
    pub const fn requires_pull(self) -> bool {
        matches!(self, Self::Unreliable)
    }

    /// Whether the model is in the soft middle tier (priceable, no new quotes).
    #[must_use]
    pub const fn is_degraded(self) -> bool {
        matches!(self, Self::Degraded)
    }
}

/// Why the model is in its current [`ModelHealth`] tier — the cause behind the
/// gate, for the dashboard and the journal. The [`tier`](ModelHealthReason::tier)
/// of a reason is exactly the [`ModelHealth`] it produces; the two always agree
/// on the snapshot (pinned by a test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelHealthReason {
    /// Everything fresh and coherent (→ [`ModelHealth::Ready`]).
    Nominal,
    /// Not enough data yet: strike not frozen, vol not warmed, or no Chainlink
    /// price ever (→ [`ModelHealth::Unreliable`]).
    Warming,
    /// The resolution-grade Chainlink feed is stale beyond threshold
    /// (→ [`ModelHealth::Unreliable`]).
    ChainlinkStale,
    /// Neither feed is fresh enough to anchor a price (→ [`ModelHealth::Unreliable`]).
    NoAnchor,
    /// Corrected-fast vs Chainlink divergence is sustained beyond the hard
    /// bound — a wrong basis or a broken feed (→ [`ModelHealth::Unreliable`]).
    DivergenceHard,
    /// The fast Binance lead was lost (feed stale or basis not warmed) so the
    /// price anchors Chainlink-only — we keep pricing but lose the fast
    /// adverse-selection defense (→ [`ModelHealth::Degraded`]).
    FastLeadLost,
    /// Divergence is sustained in the soft band, below the hard trip
    /// (→ [`ModelHealth::Degraded`]).
    DivergenceSoft,
}

impl ModelHealthReason {
    /// The health tier this reason maps to. The partition is total and the
    /// single source of truth for the reason→tier relationship.
    #[must_use]
    pub const fn tier(self) -> ModelHealth {
        match self {
            Self::Nominal => ModelHealth::Ready,
            Self::FastLeadLost | Self::DivergenceSoft => ModelHealth::Degraded,
            Self::Warming | Self::ChainlinkStale | Self::NoAnchor | Self::DivergenceHard => {
                ModelHealth::Unreliable
            }
        }
    }
}

/// Which input price the fair value was anchored on (§8: Chainlink is ground
/// truth; basis-corrected direct Binance is the fast signal when fresh).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnchorSource {
    /// Anchored directly on the resolution-grade Chainlink price.
    Chainlink,
    /// Anchored on the direct-Binance price corrected by the live
    /// Chainlink-minus-Binance basis (used only when the fast feed is fresh
    /// and the basis estimate has warmed up).
    BinanceCorrected,
}

/// Ages of the model's input feeds at snapshot time. Negative ages indicate
/// clock skew and must trip the risk manager's staleness checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAges {
    /// Age of the latest Chainlink (resolution-grade) price.
    pub chainlink: DurationMs,
    /// Age of the latest Binance (fast-signal) price.
    pub binance: DurationMs,
}

/// Point-in-time model output for one asset, optionally tied to a window.
/// Small and `Copy` — travels the event bus by value, no `Arc` needed.
///
/// Carries the inputs and intermediate `z`/`σ_τ` alongside `p_up` so the
/// consumers that need them — the engine's cancel-first repricing, the
/// |fair − mid| sanity breaker (§11), the dashboard — read one source of
/// truth rather than re-deriving. The richer per-window state (strike, both
/// strike candidates, divergence, prices) stays in the model crate's
/// `FairValue` and never rides the bus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    /// Underlying asset this model tracks.
    pub asset: Asset,
    /// Window the probability refers to, when one is active.
    pub window: Option<WindowId>,
    /// Model probability that the window resolves Up: `Φ(z)`.
    pub p_up: f64,
    /// The standardized log-moneyness `z = ln(S/K) / σ_τ`, clamped to a finite
    /// range so it always serializes (JSON has no infinity).
    pub z: f64,
    /// One-second log-return volatility (EWMA, floored/capped).
    pub sigma_1s: f64,
    /// Volatility over the remaining horizon: `σ_1s · √τ`.
    pub sigma_tau: f64,
    /// Estimated Chainlink-minus-Binance basis, in **basis points**
    /// (`(e^b − 1)·10⁴` of the log basis `b = ln(P_chainlink / P_binance)`).
    pub basis: f64,
    /// Which input the fair value was anchored on.
    pub anchor: AnchorSource,
    /// Model health gate.
    pub health: ModelHealth,
    /// Cause behind the current health tier (`reason.tier() == health`).
    pub reason: ModelHealthReason,
    /// Input feed ages at snapshot time.
    pub input_ages: InputAges,
    /// Snapshot creation time (`computed_at`).
    pub ts: TimestampMs,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_ready_allows_quoting() {
        assert!(ModelHealth::Ready.allows_quoting());
        assert!(!ModelHealth::Degraded.allows_quoting());
        assert!(!ModelHealth::Unreliable.allows_quoting());
    }

    #[test]
    fn only_unreliable_requires_pull() {
        assert!(ModelHealth::Unreliable.requires_pull());
        assert!(!ModelHealth::Ready.requires_pull());
        assert!(!ModelHealth::Degraded.requires_pull());
    }

    #[test]
    fn degraded_is_quotable_but_flagged() {
        // Degraded neither quotes nor pulls — the "hold, don't add risk" tier.
        assert!(ModelHealth::Degraded.is_degraded());
        assert!(!ModelHealth::Degraded.allows_quoting());
        assert!(!ModelHealth::Degraded.requires_pull());
    }

    #[test]
    fn health_orders_worst_ascending() {
        assert!(ModelHealth::Ready < ModelHealth::Degraded);
        assert!(ModelHealth::Degraded < ModelHealth::Unreliable);
    }

    #[test]
    fn reason_tier_is_consistent() {
        use ModelHealthReason::{
            ChainlinkStale, DivergenceHard, DivergenceSoft, FastLeadLost, NoAnchor, Nominal,
            Warming,
        };
        assert_eq!(Nominal.tier(), ModelHealth::Ready);
        assert_eq!(FastLeadLost.tier(), ModelHealth::Degraded);
        assert_eq!(DivergenceSoft.tier(), ModelHealth::Degraded);
        for r in [Warming, ChainlinkStale, NoAnchor, DivergenceHard] {
            assert_eq!(r.tier(), ModelHealth::Unreliable);
        }
    }

    #[test]
    fn snapshot_serde_roundtrip() {
        let snap = ModelSnapshot {
            asset: Asset::Btc,
            window: None,
            p_up: 0.6234,
            z: 0.314_5,
            sigma_1s: 0.00021,
            sigma_tau: 0.003_1,
            basis: -13.3,
            anchor: AnchorSource::BinanceCorrected,
            health: ModelHealth::Ready,
            reason: ModelHealthReason::Nominal,
            input_ages: InputAges {
                chainlink: DurationMs::from_millis(120),
                binance: DurationMs::from_millis(35),
            },
            ts: TimestampMs::from_millis(1_718_000_000_000),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: ModelSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, snap);
        // The snapshot's reason and health always agree.
        assert_eq!(back.reason.tier(), back.health);
    }
}
