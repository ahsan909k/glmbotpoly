//! Durable record of everything the bot sees and does: an append-only event
//! log (sqlite for structured events plus raw tick capture files) written by
//! a dedicated journal task off the hot path, and a replay reader used by
//! analytics and for offline strategy work. Every order intent, placement,
//! cancel, fill, breaker event, config change, and paper-capital adjustment
//! is journaled identically in paper and live modes so downstream consumers
//! treat both uniformly (CLAUDE.md §9, §12).
