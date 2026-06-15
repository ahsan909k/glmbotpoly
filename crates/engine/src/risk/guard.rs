//! [`GuardedPort`]: the literal gateway every strategy order passes through.
//!
//! It wraps the real [`VenuePort`] and is the only object the owned strategies
//! ever receive, so no order can reach the venue without a risk check (§5/§11).
//! Placements are gated against the shared [`GateState`] (global halt /
//! per-window loss halt / open-notional cap); cancels, cancel-market, cancel-all
//! and balance reads **always** pass through (a cancel is always a safe action,
//! and a §11 safety path must never be blocked). Every port result is observed so
//! the venue's matching-engine-restart (425) and infra errors feed the
//! `EngineRestart` / `ErrorRate` breakers.
//!
//! ## Lock discipline (why the returned future stays `Send`)
//!
//! Each method takes the `GateState` lock in a brief, `await`-free critical
//! section — once before the await to decide admission, once after to record an
//! error — and **never holds the guard across `.await`** (the `PaperVenue::apply`
//! idiom). A `std::sync::MutexGuard` is `!Send`, so the compiler enforces this
//! invariant for us: holding one across the await would fail the `impl Future +
//! Send` bound.

use std::sync::{Arc, Mutex};

use core_types::{ConditionId, NewOrder, OrderId, TimestampMs};
use venue_api::{
    Accepted, BatchPlaced, CancelReport, PlaceRejection, RejectReason, VenueError, VenuePort,
    Wallet,
};

use super::state::GateState;

/// A risk-gated [`VenuePort`] wrapping the real port for the duration of one
/// strategy fan-out. Cheap to build (borrows the port, clones an `Arc`).
pub(crate) struct GuardedPort<'a, P: VenuePort> {
    inner: &'a P,
    shared: Arc<Mutex<GateState>>,
    now: TimestampMs,
}

impl<'a, P: VenuePort> GuardedPort<'a, P> {
    pub(crate) fn new(inner: &'a P, shared: Arc<Mutex<GateState>>, now: TimestampMs) -> Self {
        Self { inner, shared, now }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<P: VenuePort> VenuePort for GuardedPort<'_, P> {
    async fn place(&self, order: &NewOrder) -> Result<Accepted, VenueError> {
        // Brief, await-free admission decision; the guard is dropped before the
        // await below.
        let decision = self.lock().admit(order);
        match decision {
            Err(detail) => Err(VenueError::Rejected(RejectReason::RiskRejected(detail))),
            Ok(()) => {
                let r = self.inner.place(order).await;
                if let Err(e) = &r {
                    self.lock().observe(e, self.now);
                }
                r
            }
        }
    }

    async fn place_batch(&self, orders: &[NewOrder]) -> Result<BatchPlaced, VenueError> {
        let decision = self.lock().admit_batch(orders);
        match decision {
            // Refuse the whole batch as positionally-aligned per-order rejections
            // (never sink it as one outer error).
            Err(detail) => Ok(BatchPlaced {
                results: orders
                    .iter()
                    .map(|o| {
                        Err(PlaceRejection {
                            client_id: o.client_id.clone(),
                            reason: RejectReason::RiskRejected(detail),
                            raw: format!("risk: {detail:?}"),
                        })
                    })
                    .collect(),
            }),
            Ok(()) => {
                let r = self.inner.place_batch(orders).await;
                if let Err(e) = &r {
                    self.lock().observe(e, self.now);
                }
                r
            }
        }
    }

    async fn cancel(&self, id: &OrderId) -> Result<CancelReport, VenueError> {
        let r = self.inner.cancel(id).await;
        if let Err(e) = &r {
            self.lock().observe(e, self.now);
        }
        r
    }

    async fn cancel_market(&self, market: &ConditionId) -> Result<CancelReport, VenueError> {
        let r = self.inner.cancel_market(market).await;
        if let Err(e) = &r {
            self.lock().observe(e, self.now);
        }
        r
    }

    async fn cancel_all(&self) -> Result<CancelReport, VenueError> {
        let r = self.inner.cancel_all().await;
        if let Err(e) = &r {
            self.lock().observe(e, self.now);
        }
        r
    }

    async fn balances(&self) -> Result<Wallet, VenueError> {
        let r = self.inner.balances().await;
        if let Err(e) = &r {
            self.lock().observe(e, self.now);
        }
        r
    }
}
