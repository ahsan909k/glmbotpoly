//! The per-series trading engine (CLAUDE.md §8): inventory and pair-cost
//! accounting, the pure quoting calculator (inventory-skewed center,
//! vol-and-latency-aware half-spread, post-only ladders, late-window gates),
//! the cancel-first quote manager that defends against adverse selection,
//! the fee-aware momentum and late-window taker modules, the order
//! normalizer enforcing tick/size/notional minimums, and the risk manager
//! through which every order must pass — no order reaches a venue without
//! its approval (CLAUDE.md §5, §11). Depends only on `model` and the
//! `venue-api` port, never on a concrete venue.
