//! The operator's window into the bot (CLAUDE.md §10): a token-protected
//! axum server providing REST snapshots and WebSocket push to a single-page
//! static UI usable on a phone. Views: overview with equity curves and the
//! global kill button, the per-series comparison/decision table, the live
//! window view (book ladder with our quotes, fair vs mid, countdown,
//! inventory), the fills blotter with markout coloring, the risk panel, and
//! controls — series enable/disable, safe-listed parameter tuning, paper
//! capital adjustment, and the multi-step live arming flow (CLAUDE.md §11).
