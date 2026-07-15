//! Observation-only live inference of the champion `dir10_full` model
//! (CLAUDE.md BUILD_PLAN 12–13, the shadow phase).
//!
//! Shadow subscribes to the same live feeds the engine uses (via a drop-on-full
//! `Event` clone) plus its **own** Binance `depth20@100ms` feed, computes the 24
//! short-horizon features **exactly per the lab's definitions** (reusing
//! `model::VolEstimator`/`norm_cdf` where parity is by construction), runs a
//! pure-Rust LightGBM tree-walk every 5 s per active window, and journals each
//! prediction to a side-channel gzip series — **influencing nothing**. It holds
//! no `bus_tx` and no venue port; a shadow crash or lag can never reach a venue
//! or stall the engine (it is wired as a non-critical supervised task fed by a
//! `try_send`-drop channel).
//!
//! Floating point is permitted inside this crate's feature math (the `model`
//! crate precedent); values that leave the crate are the prediction + the
//! feature vector for the parity guard.

pub mod features;

mod depth_feed;
mod driver;
mod lgbm;
mod params;
mod record;
mod state;

pub use depth_feed::{DepthFeedParams, combined_depth_url};
pub use driver::run;
pub use features::{FULL_FEATURES, FeatureVector, N_FEATURES};
pub use lgbm::{LgbmError, LgbmModel};
pub use params::{ModelIdentity, ShadowParams};
pub use record::{RecorderStats, ShadowPrediction, ShadowRecorder};
pub use state::{SampleOutput, ShadowCore};

use core_types::{Series, TimestampMs};

/// A lightweight shadow prediction update pushed to the dashboard (NOT a bus
/// event). The "Shadow" tile aggregates predictions/min per series, the last
/// `p_up`, feature coverage, and the model-staleness flag.
#[derive(Debug, Clone)]
pub struct ShadowUpdate {
    /// Which series this prediction is for.
    pub series: Series,
    /// The window's open time in unix millis — its identity within the series.
    /// Carried so the model taker can map the prediction to its window id;
    /// `ShadowUpdate` itself remains display-only.
    pub window_open_ms: i64,
    /// Sample time.
    pub ts: TimestampMs,
    /// Model probability of Up.
    pub p_up: f64,
    /// How many of the 24 features were finite (a live coverage signal).
    pub finite_count: u32,
    /// Model training-data end (ms) — the dashboard flags staleness past the
    /// threshold.
    pub model_trained_through_ms: i64,
    /// Short model sha for display.
    pub model_short_sha: String,
    /// Staleness threshold (days) for the "refit due" flag.
    pub staleness_alert_days: i64,
}

/// What a finished shadow run produced (supervision logging).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShadowStats {
    /// Predictions produced.
    pub predictions: u64,
    /// Predictions dropped by the writer channel.
    pub predictions_dropped: u64,
    /// Depth ladders received.
    pub depth_frames: u64,
}
