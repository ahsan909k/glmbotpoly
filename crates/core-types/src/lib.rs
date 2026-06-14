//! Foundation crate for the whole workspace: domain types (series, windows,
//! markets, orders, fills), the internal event enums that flow over channels
//! between long-running tasks, strongly-typed ids, and decimal newtypes for
//! prices, sizes, fees, and balances (money is `rust_decimal`, never float —
//! CLAUDE.md §2.6) together with the tick/size rounding helpers used at every
//! float-to-decimal boundary. This crate sits at the bottom of the dependency
//! graph and must depend on nothing else in the workspace.
//!
//! No I/O, no async, no clocks: nothing in here can observe "now" or touch
//! the network — construction and validation only.

pub mod book;
pub mod event;
pub mod ids;
pub mod inventory;
pub mod market;
pub mod mode;
pub mod model_state;
pub mod money;
pub mod order;
pub mod price;
pub mod series;
pub mod tick;
pub mod time;

pub use book::{BookLevel, BookSnapshot, TopOfBook};
pub use event::{
    BookHealth, BookUnreliableReason, BreakerKind, ControlEvent, Event, FeedHealth,
    MarketLifecycleEvent, ModelHealthEvent, PriceSource, PriceTick, RiskEvent, TickKind,
    WindowLifecycle,
};
pub use ids::{ConditionId, IdError, OrderId, TokenId};
pub use inventory::{InventorySnapshot, SettlementSummary, SideInventory};
pub use market::{
    FeeParams, MarketInfo, Outcome, ResolutionKind, ResolutionSource, TokenPair, WindowId,
};
pub use mode::{Mode, ParseModeError};
pub use model_state::{AnchorSource, InputAges, ModelHealth, ModelHealthReason, ModelSnapshot};
pub use money::{Dollars, Size, SizeError, taker_fee};
pub use order::{
    Fill, Liquidity, NewOrder, OrderIntent, OrderQty, OrderState, OrderUpdate, Side, TimeInForce,
};
pub use price::{Price, PriceError};
pub use series::{Asset, ParseSeriesError, Series, WindowDuration};
pub use tick::{RoundDir, TICK_FLIP_HI, TICK_FLIP_LO, TickSize, TickSizeError};
pub use time::{DurationMs, TimestampMs};

// Re-export the decimal type so every downstream crate names a single source of truth.
pub use rust_decimal::Decimal;
