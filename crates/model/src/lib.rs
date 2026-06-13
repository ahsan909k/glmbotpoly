//! The pricing brain (CLAUDE.md §8): an EWMA estimator of 1-second log-return
//! volatility (configurable half-life, floored and capped), the fair-value
//! engine computing p_up = Phi(ln(S/K) / (sigma_1s * sqrt(tau))), and the
//! continuously-estimated Chainlink-minus-Binance basis that corrects the
//! fast Binance signal toward the resolution-grade Chainlink anchor. Floating
//! point is permitted inside this crate's math only; values convert to
//! decimals at the boundary with explicit rounding. A composite, hysteretic
//! health gate ([`health::HealthMonitor`]) folds per-input staleness and the
//! basis/divergence signals into a single READY / DEGRADED / UNRELIABLE state
//! so the engine quotes only on READY and pulls quotes on UNRELIABLE.

pub mod basis;
pub mod fair;
pub mod health;
pub mod math;
pub mod strike;
pub mod vol;

pub use basis::{
    BasisError, BasisEstimate, BasisIgnoreReason, BasisOutcome, BasisParams, BasisTracker,
};
pub use fair::{FairInputs, FairParams, FairValue, FairValueEngine};
pub use health::{HealthError, HealthInputs, HealthMonitor, HealthParams, HealthTransition};
pub use math::{erf, erfc, norm_cdf};
pub use strike::{
    CandidatePrint, StrikeCapture, StrikeError, StrikeEstimate, StrikeIgnoreReason, StrikeOutcome,
    StrikeParams, StrikeQuality, StrikeRule,
};
pub use vol::{
    Clamp, IgnoreReason, TickOutcome, VolError, VolEstimate, VolEstimator, VolParams, VolQuality,
};
