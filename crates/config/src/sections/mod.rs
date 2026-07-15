//! One module per subsystem configuration section of [`crate::AppConfig`].
//!
//! Every section struct follows the same contract: `serde(default,
//! deny_unknown_fields)` (missing keys fall back to code defaults, typos are
//! hard errors), a hand-written `Default` that is the single source of truth
//! for default values (the committed `config/default.toml` mirrors it and a
//! test pins them together), and a `validate_into` hook that appends semantic
//! violations to the shared collector.

pub mod clock;
pub mod dashboard;
pub mod discovery;
pub mod engine;
pub mod feeds;
pub mod journal;
pub mod latency;
pub mod live;
pub mod log;
pub mod model_taker;
pub mod paper;
pub mod risk;
pub mod run;
pub mod scheduler;
pub mod shadow;
