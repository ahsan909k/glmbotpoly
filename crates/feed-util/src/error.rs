//! Feed errors. Deliberately tiny: transport trouble is never fatal (the
//! driver reconnects forever); only a torn-down process can stop the feed.

/// Fatal feed-driver errors.
#[derive(Debug, thiserror::Error)]
pub enum FeedError {
    /// The event bus receiver was dropped — the process is shutting down (or
    /// mis-wired); the feed cannot do useful work.
    #[error("event bus closed — receiver dropped")]
    BusClosed,
}
