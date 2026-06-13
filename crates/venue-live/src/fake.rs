//! A recording, scriptable [`ClobPort`] for tests (and reusable by other crates
//! via the `fake` feature).
//!
//! It records every [`BuiltOrder`] submitted and every cancel issued — which is
//! how the request-construction tests assert that each port method builds the
//! right request — and lets a test script the acks, cancel reports, balances,
//! and successive open-orders poll responses it returns. It performs no signing
//! and touches no network: it stands in for the entire SDK.

use std::collections::VecDeque;
use std::sync::Mutex;

use core_types::{ConditionId, OrderId};
use venue_api::Wallet;

use crate::convert::BuiltOrder;
use crate::error::VenueLiveError;
use crate::port::{ClobPort, RawAck, RawCancel, RawOpenOrder};

/// One cancel invocation, recorded for assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelCall {
    /// `cancel_one(id)`.
    One(OrderId),
    /// `cancel_market(condition_id)`.
    Market(ConditionId),
    /// `cancel_all()`.
    All,
}

#[derive(Default)]
struct FakeState {
    submitted: Vec<BuiltOrder>,
    batch_calls: Vec<usize>,
    cancel_calls: Vec<CancelCall>,
    ack_script: VecDeque<RawAck>,
    cancel_script: VecDeque<RawCancel>,
    open_orders_script: VecDeque<Vec<RawOpenOrder>>,
    balances: Wallet,
    counter: u64,
}

/// A recording, scriptable fake venue backend.
#[derive(Default)]
pub struct FakeClobPort {
    state: Mutex<FakeState>,
}

impl FakeClobPort {
    /// A fresh fake with default (synthesized-success) behavior.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FakeState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Scripts the next ack returned by `submit`/`submit_batch` (FIFO). When the
    /// script is empty, a success ack with a synthetic order id is synthesized.
    pub fn push_ack(&self, ack: RawAck) {
        self.lock().ack_script.push_back(ack);
    }

    /// Scripts the next cancel report (FIFO across `cancel_one`/`_market`/`_all`).
    pub fn push_cancel(&self, report: RawCancel) {
        self.lock().cancel_script.push_back(report);
    }

    /// Scripts the next `open_orders` poll response (FIFO). When exhausted,
    /// `open_orders` returns an empty list.
    pub fn push_open_orders(&self, orders: Vec<RawOpenOrder>) {
        self.lock().open_orders_script.push_back(orders);
    }

    /// Sets the wallet `balances` returns.
    pub fn set_balances(&self, wallet: Wallet) {
        self.lock().balances = wallet;
    }

    /// Every [`BuiltOrder`] submitted so far, in order.
    #[must_use]
    pub fn submitted(&self) -> Vec<BuiltOrder> {
        self.lock().submitted.clone()
    }

    /// Every cancel invocation so far, in order.
    #[must_use]
    pub fn cancel_calls(&self) -> Vec<CancelCall> {
        self.lock().cancel_calls.clone()
    }

    /// The size of each `submit_batch` call so far — lets tests assert the
    /// caller chunked correctly (e.g. `[15, 1]` for 16 orders).
    #[must_use]
    pub fn batch_calls(&self) -> Vec<usize> {
        self.lock().batch_calls.clone()
    }

    fn ack_for(state: &mut FakeState, order: &BuiltOrder) -> RawAck {
        if let Some(ack) = state.ack_script.pop_front() {
            return ack;
        }
        state.counter += 1;
        RawAck {
            client_id: order.client_id.clone(),
            success: true,
            // `format!` is always non-empty, so `.ok()` is always `Some`.
            order_id: OrderId::new(format!("fake-{}", state.counter)).ok(),
            status: "live".to_owned(),
            error: None,
        }
    }
}

impl ClobPort for FakeClobPort {
    async fn submit(&self, order: &BuiltOrder) -> Result<RawAck, VenueLiveError> {
        let mut state = self.lock();
        state.submitted.push(order.clone());
        Ok(Self::ack_for(&mut state, order))
    }

    async fn submit_batch(&self, orders: &[BuiltOrder]) -> Result<Vec<RawAck>, VenueLiveError> {
        let mut state = self.lock();
        state.batch_calls.push(orders.len());
        let mut acks = Vec::with_capacity(orders.len());
        for order in orders {
            state.submitted.push(order.clone());
            let ack = Self::ack_for(&mut state, order);
            acks.push(ack);
        }
        Ok(acks)
    }

    async fn cancel_one(&self, id: &OrderId) -> Result<RawCancel, VenueLiveError> {
        let mut state = self.lock();
        state.cancel_calls.push(CancelCall::One(id.clone()));
        Ok(state
            .cancel_script
            .pop_front()
            .unwrap_or_else(|| RawCancel {
                canceled: vec![id.clone()],
                not_canceled: vec![],
            }))
    }

    async fn cancel_market(&self, market: &ConditionId) -> Result<RawCancel, VenueLiveError> {
        let mut state = self.lock();
        state.cancel_calls.push(CancelCall::Market(market.clone()));
        Ok(state.cancel_script.pop_front().unwrap_or_default())
    }

    async fn cancel_all(&self) -> Result<RawCancel, VenueLiveError> {
        let mut state = self.lock();
        state.cancel_calls.push(CancelCall::All);
        Ok(state.cancel_script.pop_front().unwrap_or_default())
    }

    async fn balances(&self) -> Result<Wallet, VenueLiveError> {
        Ok(self.lock().balances.clone())
    }

    async fn open_orders(&self) -> Result<Vec<RawOpenOrder>, VenueLiveError> {
        Ok(self
            .lock()
            .open_orders_script
            .pop_front()
            .unwrap_or_default())
    }
}
