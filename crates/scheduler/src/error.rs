//! Scheduler driver errors. Per-series failures are never errors — they are
//! warnings plus retries (CLAUDE.md §6); the driver only fails when the
//! process around it is structurally gone.

/// Fatal driver conditions. Either one means the orchestrator is shutting
/// down (or broken) — there is nobody left to schedule for.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// Every bus receiver was dropped — announcements have no audience.
    #[error("event bus closed: every consumer dropped its receiver")]
    BusClosed,
    /// The discovery worker task died unexpectedly.
    #[error("discovery worker channel closed unexpectedly")]
    WorkerGone,
}
