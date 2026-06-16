//! The queryable sqlite index over the structured low-rate events.
//!
//! The gzip segments (see [`recorder`](crate::recorder)) remain the complete,
//! ordered **source of truth** — replay and restart-rebuild read only from them.
//! This module adds a *secondary, rebuildable* index: as the writer thread
//! projects each [`Event`](core_types::Event) to a [`JournalRecord`], the six
//! **structured** kinds (orders, fills, windows, settlements, breaker trips,
//! config changes) are also inserted into a sqlite database so the dashboard and
//! analytics can query them. High-rate tick/book/model events never touch sqlite
//! (they stay compact in the gzip segments).
//!
//! Two halves:
//! - [`JournalIndex`] — the writer. Owns a [`rusqlite::Connection`] created and
//!   used **entirely on the recorder's writer thread**; inserts batch into one
//!   transaction committed on the recorder's flush cadence.
//! - [`JournalIndexReader`] — a separate **read-only** connection (WAL mode
//!   allows it to read concurrently with the live writer), with focused query
//!   methods returning owned, typed rows.
//!
//! Column representation is lossless by construction: decimals (prices, sizes,
//! dollars, fee rates) are stored as their exact `Decimal` strings as `TEXT`
//! (never `REAL`); ids and enum labels as `TEXT`; timestamps as `INTEGER`
//! milliseconds. [`WindowId`] is decomposed into `series` + `open_ms` columns so
//! windowed queries can filter by series and range on the open time. Readers
//! reconstruct typed newtypes through their fallible constructors, so a corrupt
//! cell is rejected ([`JournalError::Corrupt`]), never silently coerced.

use std::path::Path;
use std::str::FromStr;

use core_types::{
    BreakerKind, CommandOrigin, ConditionId, Decimal, Dollars, Mode, Outcome, Price,
    ResolutionKind, RiskEvent, Series, Side, Size, TickSize, TimestampMs, TokenId, TokenPair,
    WindowId, WindowLifecycle,
};
use core_types::{Liquidity, OrderState};
use rusqlite::types::Type;
use rusqlite::{Connection, OpenFlags, Row, params};

use crate::record::JournalRecord;
use crate::recorder::JournalError;

/// The structured-index schema version, stored in the `meta` table.
///
/// v2 (2026-06-15, control-plane task): added the `command_audit` table. The
/// addition is purely additive (`CREATE TABLE IF NOT EXISTS`); an old v1 DB is
/// upgraded in place by the idempotent DDL and the version is only ever written
/// on first creation, so existing rows are untouched and the index stays
/// rebuildable from the gzip source of truth regardless.
const SCHEMA_VERSION: i64 = 2;

/// Idempotent DDL: six structured tables + a `meta` table + query indexes.
const SCHEMA_DDL: &str = "\
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    seq           INTEGER NOT NULL,
    ts_local_ms   INTEGER NOT NULL,
    order_id      TEXT NOT NULL,
    series        TEXT NOT NULL,
    open_ms       INTEGER NOT NULL,
    token_id      TEXT NOT NULL,
    side          TEXT NOT NULL,
    state         TEXT NOT NULL,
    price         TEXT NOT NULL,
    original_size TEXT NOT NULL,
    filled_size   TEXT NOT NULL,
    reject_reason TEXT,
    ts_venue_ms   INTEGER
);
CREATE INDEX IF NOT EXISTS orders_window ON orders(series, open_ms);
CREATE INDEX IF NOT EXISTS orders_oid    ON orders(order_id);
CREATE INDEX IF NOT EXISTS orders_ts     ON orders(ts_local_ms);
CREATE TABLE IF NOT EXISTS fills (
    seq         INTEGER NOT NULL,
    ts_local_ms INTEGER NOT NULL,
    order_id    TEXT NOT NULL,
    trade_id    TEXT,
    series      TEXT NOT NULL,
    open_ms     INTEGER NOT NULL,
    token_id    TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    side        TEXT NOT NULL,
    price       TEXT NOT NULL,
    size        TEXT NOT NULL,
    liquidity   TEXT NOT NULL,
    fee         TEXT NOT NULL,
    ts_venue_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS fills_window ON fills(series, open_ms);
CREATE INDEX IF NOT EXISTS fills_ts     ON fills(ts_local_ms);
CREATE TABLE IF NOT EXISTS windows (
    seq              INTEGER NOT NULL,
    ts_local_ms      INTEGER NOT NULL,
    series           TEXT NOT NULL,
    open_ms          INTEGER NOT NULL,
    lifecycle        TEXT NOT NULL,
    resolved_outcome TEXT,
    event_slug       TEXT NOT NULL,
    condition_id     TEXT NOT NULL,
    token_up         TEXT NOT NULL,
    token_down       TEXT NOT NULL,
    close_ms         INTEGER NOT NULL,
    strike           TEXT,
    tick_size        TEXT NOT NULL,
    min_order_size   TEXT NOT NULL,
    fee_rate         TEXT NOT NULL,
    fee_exponent     INTEGER NOT NULL,
    fee_taker_only   INTEGER NOT NULL,
    fee_rebate_rate  TEXT NOT NULL,
    fee_enabled      INTEGER NOT NULL,
    neg_risk         INTEGER NOT NULL,
    resolution_kind  TEXT NOT NULL,
    resolution_raw   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS windows_window    ON windows(series, open_ms);
CREATE INDEX IF NOT EXISTS windows_lifecycle ON windows(lifecycle);
CREATE INDEX IF NOT EXISTS windows_ts        ON windows(ts_local_ms);
CREATE TABLE IF NOT EXISTS settlements (
    seq           INTEGER NOT NULL,
    ts_local_ms   INTEGER NOT NULL,
    series        TEXT NOT NULL,
    open_ms       INTEGER NOT NULL,
    outcome       TEXT NOT NULL,
    up_shares     TEXT NOT NULL,
    up_cost       TEXT NOT NULL,
    down_shares   TEXT NOT NULL,
    down_cost     TEXT NOT NULL,
    matched_pairs TEXT NOT NULL,
    pair_cost     TEXT,
    excess        TEXT NOT NULL,
    merged_pairs  TEXT NOT NULL,
    fees_paid     TEXT NOT NULL,
    realized_pnl  TEXT NOT NULL,
    settle_ms     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS settlements_window ON settlements(series, open_ms);
CREATE INDEX IF NOT EXISTS settlements_ts     ON settlements(ts_local_ms);
CREATE TABLE IF NOT EXISTS breaker_trips (
    seq         INTEGER NOT NULL,
    ts_local_ms INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    breaker     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS breaker_trips_ts ON breaker_trips(ts_local_ms);
CREATE TABLE IF NOT EXISTS control_events (
    seq         INTEGER NOT NULL,
    ts_local_ms INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    mode        TEXT,
    series      TEXT,
    amount      TEXT
);
CREATE INDEX IF NOT EXISTS control_events_ts ON control_events(ts_local_ms);
CREATE TABLE IF NOT EXISTS command_audit (
    seq         INTEGER NOT NULL,
    ts_local_ms INTEGER NOT NULL,
    origin      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    detail      TEXT NOT NULL,
    accepted    INTEGER NOT NULL,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS command_audit_ts ON command_audit(ts_local_ms);
";

/// The seven structured tables, in a stable order — used by retention pruning.
const STRUCTURED_TABLES: [&str; 7] = [
    "orders",
    "fills",
    "windows",
    "settlements",
    "breaker_trips",
    "control_events",
    "command_audit",
];

// ----------------------------------------------------------------------------
// Writer half
// ----------------------------------------------------------------------------

/// The sqlite index writer. Lives on the recorder's writer thread; its
/// connection never crosses threads.
pub struct JournalIndex {
    conn: Connection,
    /// Whether an insert transaction is currently open (lazily begun, committed
    /// on the recorder's flush cadence).
    txn_open: bool,
    /// Cumulative structured rows inserted (for [`RecorderStats`](crate::RecorderStats)).
    indexed: u64,
}

impl JournalIndex {
    /// Opens (creating if needed) the index at `path`, enables WAL mode, applies
    /// the schema, and records the schema version.
    ///
    /// # Errors
    /// [`JournalError::Io`] if the parent directory cannot be created, or
    /// [`JournalError::Sqlite`] on any sqlite failure.
    pub fn open(path: &Path) -> Result<Self, JournalError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        // WAL gives crash-safe durability of committed transactions and lets a
        // separate read-only connection query concurrently with the writer.
        // synchronous=NORMAL is the recommended WAL pairing (a power loss may
        // lose the last commit but never corrupts; the gzip log is the source
        // of truth and this index is rebuildable regardless).
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        conn.execute_batch(SCHEMA_DDL)?;
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?1)
             ON CONFLICT(key) DO NOTHING",
            params![SCHEMA_VERSION.to_string()],
        )?;
        Ok(Self {
            conn,
            txn_open: false,
            indexed: 0,
        })
    }

    /// Indexes one record. Only the six structured kinds insert a row; every
    /// other (high-rate) kind is a no-op. Inserts batch into one transaction,
    /// begun lazily here and committed by [`JournalIndex::commit`].
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] on an insert failure.
    pub fn index(
        &mut self,
        seq: u64,
        ts_local_ms: i64,
        rec: &JournalRecord,
    ) -> Result<(), JournalError> {
        let seq = seq as i64;
        match rec {
            JournalRecord::OrderUpdate(u) => {
                self.begin_if_needed()?;
                self.conn.execute(
                    "INSERT INTO orders
                     (seq, ts_local_ms, order_id, series, open_ms, token_id, side, state,
                      price, original_size, filled_size, reject_reason, ts_venue_ms)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
                    params![
                        seq,
                        ts_local_ms,
                        u.order_id.as_str(),
                        u.window.series.key(),
                        u.window.open_time.as_millis(),
                        u.token_id.as_str(),
                        side_str(u.side),
                        order_state_str(u.state),
                        u.price.as_decimal().to_string(),
                        u.original_size.as_decimal().to_string(),
                        u.filled_size.as_decimal().to_string(),
                        u.reject_reason.as_deref(),
                        u.ts_venue.map(TimestampMs::as_millis),
                    ],
                )?;
                self.indexed += 1;
            }
            JournalRecord::Fill(f) => {
                self.begin_if_needed()?;
                self.conn.execute(
                    "INSERT INTO fills
                     (seq, ts_local_ms, order_id, trade_id, series, open_ms, token_id, outcome,
                      side, price, size, liquidity, fee, ts_venue_ms)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                    params![
                        seq,
                        ts_local_ms,
                        f.order_id.as_str(),
                        f.trade_id.as_deref(),
                        f.window.series.key(),
                        f.window.open_time.as_millis(),
                        f.token_id.as_str(),
                        outcome_str(f.outcome),
                        side_str(f.side),
                        f.price.as_decimal().to_string(),
                        f.size.as_decimal().to_string(),
                        liquidity_str(f.liquidity),
                        f.fee.as_decimal().to_string(),
                        f.ts_venue.as_millis(),
                    ],
                )?;
                self.indexed += 1;
            }
            JournalRecord::Window { market, lifecycle } => {
                self.begin_if_needed()?;
                let resolved_outcome = match lifecycle {
                    WindowLifecycle::Resolved { outcome } => Some(outcome_str(*outcome)),
                    _ => None,
                };
                self.conn.execute(
                    "INSERT INTO windows
                     (seq, ts_local_ms, series, open_ms, lifecycle, resolved_outcome, event_slug,
                      condition_id, token_up, token_down, close_ms, strike, tick_size,
                      min_order_size, fee_rate, fee_exponent, fee_taker_only, fee_rebate_rate,
                      fee_enabled, neg_risk, resolution_kind, resolution_raw)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22)",
                    params![
                        seq,
                        ts_local_ms,
                        market.window.series.key(),
                        market.window.open_time.as_millis(),
                        lifecycle_phase_str(*lifecycle),
                        resolved_outcome,
                        market.event_slug.as_str(),
                        market.condition_id.as_str(),
                        market.tokens.up.as_str(),
                        market.tokens.down.as_str(),
                        market.close_time.as_millis(),
                        market.strike.map(|d| d.to_string()),
                        market.tick_size.as_decimal().to_string(),
                        market.min_order_size.as_decimal().to_string(),
                        market.fees.rate.to_string(),
                        i64::from(market.fees.exponent),
                        i64::from(market.fees.taker_only),
                        market.fees.rebate_rate.to_string(),
                        i64::from(market.fees.enabled),
                        i64::from(market.neg_risk),
                        resolution_kind_str(market.resolution.kind),
                        market.resolution.raw.as_str(),
                    ],
                )?;
                self.indexed += 1;
            }
            JournalRecord::Settlement(s) => {
                self.begin_if_needed()?;
                self.conn.execute(
                    "INSERT INTO settlements
                     (seq, ts_local_ms, series, open_ms, outcome, up_shares, up_cost, down_shares,
                      down_cost, matched_pairs, pair_cost, excess, merged_pairs, fees_paid,
                      realized_pnl, settle_ms)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
                    params![
                        seq,
                        ts_local_ms,
                        s.window.series.key(),
                        s.window.open_time.as_millis(),
                        outcome_str(s.outcome),
                        s.up.shares.as_decimal().to_string(),
                        s.up.cost.as_decimal().to_string(),
                        s.down.shares.as_decimal().to_string(),
                        s.down.cost.as_decimal().to_string(),
                        s.matched_pairs.as_decimal().to_string(),
                        s.pair_cost.map(|d| d.to_string()),
                        s.excess.to_string(),
                        s.merged_pairs.as_decimal().to_string(),
                        s.fees_paid.as_decimal().to_string(),
                        s.realized_pnl.as_decimal().to_string(),
                        s.ts.as_millis(),
                    ],
                )?;
                self.indexed += 1;
            }
            JournalRecord::Risk(r) => {
                self.begin_if_needed()?;
                let (kind, breaker) = risk_components(*r);
                self.conn.execute(
                    "INSERT INTO breaker_trips (seq, ts_local_ms, kind, breaker)
                     VALUES (?1,?2,?3,?4)",
                    params![seq, ts_local_ms, kind, breaker_str(breaker)],
                )?;
                self.indexed += 1;
            }
            JournalRecord::Control(c) => {
                self.begin_if_needed()?;
                let (kind, mode, series, amount) = control_components(c);
                self.conn.execute(
                    "INSERT INTO control_events (seq, ts_local_ms, kind, mode, series, amount)
                     VALUES (?1,?2,?3,?4,?5,?6)",
                    params![seq, ts_local_ms, kind, mode, series, amount],
                )?;
                self.indexed += 1;
            }
            JournalRecord::ControlAudit(a) => {
                self.begin_if_needed()?;
                self.conn.execute(
                    "INSERT INTO command_audit
                     (seq, ts_local_ms, origin, kind, detail, accepted, error)
                     VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        seq,
                        ts_local_ms,
                        origin_str(a.origin),
                        a.kind.as_str(),
                        a.detail.as_str(),
                        i64::from(a.accepted),
                        a.error.as_deref(),
                    ],
                )?;
                self.indexed += 1;
            }
            // High-rate / non-structured kinds: stay in the gzip log only.
            _ => {}
        }
        Ok(())
    }

    /// Commits the open insert transaction, if any. A no-op when nothing has
    /// been inserted since the last commit.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] if the commit fails.
    pub fn commit(&mut self) -> Result<(), JournalError> {
        if self.txn_open {
            self.conn.execute_batch("COMMIT")?;
            self.txn_open = false;
        }
        Ok(())
    }

    /// Deletes every indexed row older than `cutoff_ms` (by `ts_local_ms`),
    /// returning the number of rows removed. Commits any pending insert
    /// transaction first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] on a delete failure.
    pub fn prune(&mut self, cutoff_ms: i64) -> Result<u64, JournalError> {
        self.commit()?;
        let tx = self.conn.transaction()?;
        let mut total = 0u64;
        for table in STRUCTURED_TABLES {
            // `table` is one of a fixed set of literals — no injection surface.
            let removed = tx.execute(
                &format!("DELETE FROM {table} WHERE ts_local_ms < ?1"),
                params![cutoff_ms],
            )?;
            total += removed as u64;
        }
        tx.commit()?;
        Ok(total)
    }

    /// Commits any pending transaction and closes the connection.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] if the final commit fails.
    pub fn close(mut self) -> Result<(), JournalError> {
        self.commit()
    }

    /// Cumulative structured rows inserted so far.
    #[must_use]
    pub fn indexed(&self) -> u64 {
        self.indexed
    }

    fn begin_if_needed(&mut self) -> Result<(), JournalError> {
        if !self.txn_open {
            self.conn.execute_batch("BEGIN")?;
            self.txn_open = true;
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Reader half
// ----------------------------------------------------------------------------

/// A read-only query handle over a journal index. Opens its own connection
/// (WAL mode allows it to read concurrently with the live writer).
pub struct JournalIndexReader {
    conn: Connection,
}

impl JournalIndexReader {
    /// Opens the index at `path` read-only.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self, JournalError> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY
                | OpenFlags::SQLITE_OPEN_URI
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        Ok(Self { conn })
    }

    /// The schema version recorded in the index.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`] on a bad/absent value.
    pub fn schema_version(&self) -> Result<i64, JournalError> {
        let value: String = self.conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )?;
        value
            .parse()
            .map_err(|_| JournalError::Corrupt(format!("schema_version {value:?}")))
    }

    /// Every fill for one window, oldest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn fills_for_window(
        &self,
        series: Series,
        open_ms: i64,
    ) -> Result<Vec<FillRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, order_id, trade_id, series, open_ms, token_id, outcome,
                    side, price, size, liquidity, fee, ts_venue_ms
             FROM fills WHERE series = ?1 AND open_ms = ?2 ORDER BY seq ASC",
        )?;
        collect(stmt.query_map(params![series.key(), open_ms], fill_row)?)
    }

    /// The most recent `limit` fills across all windows, newest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn recent_fills(&self, limit: usize) -> Result<Vec<FillRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, order_id, trade_id, series, open_ms, token_id, outcome,
                    side, price, size, liquidity, fee, ts_venue_ms
             FROM fills ORDER BY seq DESC LIMIT ?1",
        )?;
        collect(stmt.query_map(params![limit as i64], fill_row)?)
    }

    /// Every order update for one window, oldest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn orders_for_window(
        &self,
        series: Series,
        open_ms: i64,
    ) -> Result<Vec<OrderRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, order_id, series, open_ms, token_id, side, state,
                    price, original_size, filled_size, reject_reason, ts_venue_ms
             FROM orders WHERE series = ?1 AND open_ms = ?2 ORDER BY seq ASC",
        )?;
        collect(stmt.query_map(params![series.key(), open_ms], order_row)?)
    }

    /// Every window-lifecycle row, oldest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn windows(&self) -> Result<Vec<WindowRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, series, open_ms, lifecycle, resolved_outcome, event_slug,
                    condition_id, token_up, token_down, close_ms, strike, tick_size,
                    min_order_size, fee_rate, neg_risk, resolution_kind, resolution_raw
             FROM windows ORDER BY seq ASC",
        )?;
        collect(stmt.query_map([], window_row)?)
    }

    /// Every settlement summary, oldest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn settlements(&self) -> Result<Vec<SettlementRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, series, open_ms, outcome, up_shares, up_cost, down_shares,
                    down_cost, matched_pairs, pair_cost, excess, merged_pairs, fees_paid,
                    realized_pnl, settle_ms
             FROM settlements ORDER BY seq ASC",
        )?;
        collect(stmt.query_map([], settlement_row)?)
    }

    /// Every breaker-trip event, oldest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn breaker_trips(&self) -> Result<Vec<BreakerRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, kind, breaker FROM breaker_trips ORDER BY seq ASC",
        )?;
        collect(stmt.query_map([], breaker_row)?)
    }

    /// Every control-plane (config-change) event, oldest first.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn control_events(&self) -> Result<Vec<ControlRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, kind, mode, series, amount
             FROM control_events ORDER BY seq ASC",
        )?;
        collect(stmt.query_map([], control_row)?)
    }

    /// Every command-audit record (the §10.6/§11 operator audit trail), oldest
    /// first — accepted and rejected commands alike, with origin and outcome.
    ///
    /// # Errors
    /// [`JournalError::Sqlite`] / [`JournalError::Corrupt`].
    pub fn command_audit(&self) -> Result<Vec<ControlAuditRow>, JournalError> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ts_local_ms, origin, kind, detail, accepted, error
             FROM command_audit ORDER BY seq ASC",
        )?;
        collect(stmt.query_map([], audit_row)?)
    }
}

/// Collapses a `query_map` iterator into a `Vec`, surfacing the first error.
fn collect<T>(rows: impl Iterator<Item = rusqlite::Result<T>>) -> Result<Vec<T>, JournalError> {
    rows.collect::<rusqlite::Result<Vec<T>>>()
        .map_err(JournalError::from)
}

// ----------------------------------------------------------------------------
// Owned, typed query rows
// ----------------------------------------------------------------------------

/// An indexed order update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Venue order id.
    pub order_id: core_types::OrderId,
    /// Window the order trades.
    pub window: WindowId,
    /// Outcome token of the order.
    pub token_id: TokenId,
    /// Order side.
    pub side: Side,
    /// Lifecycle state at this update.
    pub state: OrderState,
    /// Order limit price.
    pub price: Price,
    /// Original order size.
    pub original_size: Size,
    /// Cumulative filled size.
    pub filled_size: Size,
    /// Reject reason, when rejected.
    pub reject_reason: Option<String>,
    /// Venue-reported event time, when available.
    pub ts_venue_ms: Option<i64>,
}

/// An indexed fill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Our order that traded.
    pub order_id: core_types::OrderId,
    /// Venue trade id, when available.
    pub trade_id: Option<String>,
    /// Window the fill belongs to.
    pub window: WindowId,
    /// Outcome token that traded.
    pub token_id: TokenId,
    /// Which outcome that token is.
    pub outcome: Outcome,
    /// Our side of the trade.
    pub side: Side,
    /// Execution price.
    pub price: Price,
    /// Executed shares.
    pub size: Size,
    /// Maker or taker.
    pub liquidity: Liquidity,
    /// Fee charged (zero for maker).
    pub fee: Dollars,
    /// Venue-reported execution time (unix ms).
    pub ts_venue_ms: i64,
}

/// An indexed window-lifecycle row with its market metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Window identity.
    pub window: WindowId,
    /// Lifecycle phase label (`discovered`/`open`/`closing`/`closed`/`resolved`).
    pub lifecycle: String,
    /// Winning outcome, when the phase is `resolved`.
    pub resolved_outcome: Option<Outcome>,
    /// Gamma event slug.
    pub event_slug: String,
    /// Market condition id.
    pub condition_id: ConditionId,
    /// Up/Down outcome tokens.
    pub tokens: TokenPair,
    /// Window close time (unix ms).
    pub close_ms: i64,
    /// Captured price-to-beat, when known.
    pub strike: Option<Decimal>,
    /// Tick size at this announcement.
    pub tick_size: TickSize,
    /// Venue minimum resting order size.
    pub min_order_size: Size,
    /// Taker fee rate.
    pub fee_rate: Decimal,
    /// `neg_risk` flag.
    pub neg_risk: bool,
    /// Parsed resolution-source kind.
    pub resolution_kind: ResolutionKind,
    /// Raw resolution-source text.
    pub resolution_raw: String,
}

/// An indexed settlement summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Window that settled.
    pub window: WindowId,
    /// Winning outcome.
    pub outcome: Outcome,
    /// Up-side shares at close.
    pub up_shares: Size,
    /// Up-side cost at close.
    pub up_cost: Dollars,
    /// Down-side shares at close.
    pub down_shares: Size,
    /// Down-side cost at close.
    pub down_cost: Dollars,
    /// Matched pairs held at close.
    pub matched_pairs: Size,
    /// Combined pair cost, when both sides held.
    pub pair_cost: Option<Decimal>,
    /// Signed unmatched shares (positive = Up excess).
    pub excess: Decimal,
    /// Pairs merged during the window.
    pub merged_pairs: Size,
    /// Cumulative taker fees paid.
    pub fees_paid: Dollars,
    /// Net realized PnL.
    pub realized_pnl: Dollars,
    /// Settlement time (unix ms).
    pub settle_ms: i64,
}

/// An indexed breaker-trip event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakerRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Event kind (`tripped`/`cleared`/`cancel_all`).
    pub kind: String,
    /// Which breaker.
    pub breaker: BreakerKind,
}

/// An indexed control-plane (config-change) event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Event kind (`mode_changed`/`series_enabled`/`series_disabled`/
    /// `paper_capital_set`/`shutdown`/`kill`/`reset`/`daily_stop_reset`).
    pub kind: String,
    /// New mode, for `mode_changed`.
    pub mode: Option<Mode>,
    /// Affected series, for `series_enabled`/`series_disabled`.
    pub series: Option<Series>,
    /// New paper capital, for `paper_capital_set`.
    pub amount: Option<Dollars>,
}

/// An indexed command-audit record — the §10.6/§11 operator audit trail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlAuditRow {
    /// Journal sequence number of the record.
    pub seq: u64,
    /// Local write time (unix ms).
    pub ts_local_ms: i64,
    /// Which frontend issued the command.
    pub origin: CommandOrigin,
    /// Stable command-kind label (e.g. `kill`, `set_param`, `arm_live`).
    pub kind: String,
    /// Arguments / resulting-state summary.
    pub detail: String,
    /// Whether the command was accepted.
    pub accepted: bool,
    /// The rejection reason when not accepted.
    pub error: Option<String>,
}

// ----------------------------------------------------------------------------
// Row constructors (rusqlite `Row` → typed row)
// ----------------------------------------------------------------------------

fn order_row(row: &Row) -> rusqlite::Result<OrderRow> {
    Ok(OrderRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        order_id: get_order_id(row, 2)?,
        window: get_window(row, 3, 4)?,
        token_id: get_token_id(row, 5)?,
        side: get_side(row, 6)?,
        state: get_order_state(row, 7)?,
        price: get_price(row, 8)?,
        original_size: get_size(row, 9)?,
        filled_size: get_size(row, 10)?,
        reject_reason: row.get(11)?,
        ts_venue_ms: row.get(12)?,
    })
}

fn fill_row(row: &Row) -> rusqlite::Result<FillRow> {
    Ok(FillRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        order_id: get_order_id(row, 2)?,
        trade_id: row.get(3)?,
        window: get_window(row, 4, 5)?,
        token_id: get_token_id(row, 6)?,
        outcome: get_outcome(row, 7)?,
        side: get_side(row, 8)?,
        price: get_price(row, 9)?,
        size: get_size(row, 10)?,
        liquidity: get_liquidity(row, 11)?,
        fee: get_dollars(row, 12)?,
        ts_venue_ms: row.get(13)?,
    })
}

fn window_row(row: &Row) -> rusqlite::Result<WindowRow> {
    Ok(WindowRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        window: get_window(row, 2, 3)?,
        lifecycle: row.get(4)?,
        resolved_outcome: get_opt_outcome(row, 5)?,
        event_slug: row.get(6)?,
        condition_id: get_condition_id(row, 7)?,
        tokens: TokenPair {
            up: get_token_id(row, 8)?,
            down: get_token_id(row, 9)?,
        },
        close_ms: row.get(10)?,
        strike: get_opt_decimal(row, 11)?,
        tick_size: get_tick(row, 12)?,
        min_order_size: get_size(row, 13)?,
        fee_rate: get_decimal(row, 14)?,
        neg_risk: row.get::<_, i64>(15)? != 0,
        resolution_kind: get_resolution_kind(row, 16)?,
        resolution_raw: row.get(17)?,
    })
}

fn settlement_row(row: &Row) -> rusqlite::Result<SettlementRow> {
    Ok(SettlementRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        window: get_window(row, 2, 3)?,
        outcome: get_outcome(row, 4)?,
        up_shares: get_size(row, 5)?,
        up_cost: get_dollars(row, 6)?,
        down_shares: get_size(row, 7)?,
        down_cost: get_dollars(row, 8)?,
        matched_pairs: get_size(row, 9)?,
        pair_cost: get_opt_decimal(row, 10)?,
        excess: get_decimal(row, 11)?,
        merged_pairs: get_size(row, 12)?,
        fees_paid: get_dollars(row, 13)?,
        realized_pnl: get_dollars(row, 14)?,
        settle_ms: row.get(15)?,
    })
}

fn breaker_row(row: &Row) -> rusqlite::Result<BreakerRow> {
    Ok(BreakerRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        kind: row.get(2)?,
        breaker: get_breaker(row, 3)?,
    })
}

fn control_row(row: &Row) -> rusqlite::Result<ControlRow> {
    Ok(ControlRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        kind: row.get(2)?,
        mode: get_opt_mode(row, 3)?,
        series: get_opt_series(row, 4)?,
        amount: get_opt_dollars(row, 5)?,
    })
}

fn audit_row(row: &Row) -> rusqlite::Result<ControlAuditRow> {
    Ok(ControlAuditRow {
        seq: get_seq(row, 0)?,
        ts_local_ms: row.get(1)?,
        origin: get_origin(row, 2)?,
        kind: row.get(3)?,
        detail: row.get(4)?,
        accepted: row.get::<_, i64>(5)? != 0,
        error: row.get(6)?,
    })
}

// ----------------------------------------------------------------------------
// Column reading helpers (TEXT → typed, mapping parse failures to a sqlite error)
// ----------------------------------------------------------------------------

fn corrupt<T, E>(col: usize, r: Result<T, E>) -> rusqlite::Result<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    r.map_err(|e| rusqlite::Error::FromSqlConversionFailure(col, Type::Text, Box::new(e)))
}

fn bad_label(col: usize, kind: &'static str, value: String) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(col, Type::Text, Box::new(LabelError { kind, value }))
}

#[derive(Debug, thiserror::Error)]
#[error("unrecognized {kind} label {value:?} in journal index")]
struct LabelError {
    kind: &'static str,
    value: String,
}

fn get_seq(row: &Row, col: usize) -> rusqlite::Result<u64> {
    let v: i64 = row.get(col)?;
    Ok(v as u64)
}

fn get_decimal(row: &Row, col: usize) -> rusqlite::Result<Decimal> {
    let s: String = row.get(col)?;
    corrupt(col, Decimal::from_str(&s))
}

fn get_opt_decimal(row: &Row, col: usize) -> rusqlite::Result<Option<Decimal>> {
    let s: Option<String> = row.get(col)?;
    match s {
        Some(s) => Ok(Some(corrupt(col, Decimal::from_str(&s))?)),
        None => Ok(None),
    }
}

fn get_price(row: &Row, col: usize) -> rusqlite::Result<Price> {
    corrupt(col, Price::try_from(get_decimal(row, col)?))
}

fn get_size(row: &Row, col: usize) -> rusqlite::Result<Size> {
    corrupt(col, Size::new(get_decimal(row, col)?))
}

fn get_dollars(row: &Row, col: usize) -> rusqlite::Result<Dollars> {
    Ok(Dollars::new(get_decimal(row, col)?))
}

fn get_opt_dollars(row: &Row, col: usize) -> rusqlite::Result<Option<Dollars>> {
    Ok(get_opt_decimal(row, col)?.map(Dollars::new))
}

fn get_order_id(row: &Row, col: usize) -> rusqlite::Result<core_types::OrderId> {
    let s: String = row.get(col)?;
    corrupt(col, core_types::OrderId::new(s))
}

fn get_token_id(row: &Row, col: usize) -> rusqlite::Result<TokenId> {
    let s: String = row.get(col)?;
    corrupt(col, TokenId::new(s))
}

fn get_condition_id(row: &Row, col: usize) -> rusqlite::Result<ConditionId> {
    let s: String = row.get(col)?;
    corrupt(col, ConditionId::new(s))
}

fn get_window(row: &Row, series_col: usize, open_col: usize) -> rusqlite::Result<WindowId> {
    let s: String = row.get(series_col)?;
    let series = corrupt(series_col, Series::from_str(&s))?;
    let open_ms: i64 = row.get(open_col)?;
    Ok(WindowId {
        series,
        open_time: TimestampMs::from_millis(open_ms),
    })
}

fn get_side(row: &Row, col: usize) -> rusqlite::Result<Side> {
    let s: String = row.get(col)?;
    side_from(&s).ok_or_else(|| bad_label(col, "side", s))
}

fn get_outcome(row: &Row, col: usize) -> rusqlite::Result<Outcome> {
    let s: String = row.get(col)?;
    outcome_from(&s).ok_or_else(|| bad_label(col, "outcome", s))
}

fn get_opt_outcome(row: &Row, col: usize) -> rusqlite::Result<Option<Outcome>> {
    let s: Option<String> = row.get(col)?;
    match s {
        Some(s) => Ok(Some(
            outcome_from(&s).ok_or_else(|| bad_label(col, "outcome", s))?,
        )),
        None => Ok(None),
    }
}

fn get_liquidity(row: &Row, col: usize) -> rusqlite::Result<Liquidity> {
    let s: String = row.get(col)?;
    liquidity_from(&s).ok_or_else(|| bad_label(col, "liquidity", s))
}

fn get_order_state(row: &Row, col: usize) -> rusqlite::Result<OrderState> {
    let s: String = row.get(col)?;
    order_state_from(&s).ok_or_else(|| bad_label(col, "order_state", s))
}

fn get_tick(row: &Row, col: usize) -> rusqlite::Result<TickSize> {
    corrupt(col, TickSize::from_decimal(get_decimal(row, col)?))
}

fn get_breaker(row: &Row, col: usize) -> rusqlite::Result<BreakerKind> {
    let s: String = row.get(col)?;
    breaker_from(&s).ok_or_else(|| bad_label(col, "breaker", s))
}

fn get_resolution_kind(row: &Row, col: usize) -> rusqlite::Result<ResolutionKind> {
    let s: String = row.get(col)?;
    resolution_kind_from(&s).ok_or_else(|| bad_label(col, "resolution_kind", s))
}

fn get_opt_mode(row: &Row, col: usize) -> rusqlite::Result<Option<Mode>> {
    let s: Option<String> = row.get(col)?;
    match s {
        Some(s) => Ok(Some(
            mode_from(&s).ok_or_else(|| bad_label(col, "mode", s))?,
        )),
        None => Ok(None),
    }
}

fn get_opt_series(row: &Row, col: usize) -> rusqlite::Result<Option<Series>> {
    let s: Option<String> = row.get(col)?;
    match s {
        Some(s) => Ok(Some(corrupt(col, Series::from_str(&s))?)),
        None => Ok(None),
    }
}

fn get_origin(row: &Row, col: usize) -> rusqlite::Result<CommandOrigin> {
    let s: String = row.get(col)?;
    origin_from(&s).ok_or_else(|| bad_label(col, "command_origin", s))
}

// ----------------------------------------------------------------------------
// Enum ↔ stable DB label (deliberately not serde variant names — a new variant
// forces an update here rather than silently writing an unstable label)
// ----------------------------------------------------------------------------

fn side_str(s: Side) -> &'static str {
    match s {
        Side::Buy => "buy",
        Side::Sell => "sell",
    }
}
fn side_from(s: &str) -> Option<Side> {
    match s {
        "buy" => Some(Side::Buy),
        "sell" => Some(Side::Sell),
        _ => None,
    }
}

fn outcome_str(o: Outcome) -> &'static str {
    match o {
        Outcome::Up => "up",
        Outcome::Down => "down",
    }
}
fn outcome_from(s: &str) -> Option<Outcome> {
    match s {
        "up" => Some(Outcome::Up),
        "down" => Some(Outcome::Down),
        _ => None,
    }
}

fn liquidity_str(l: Liquidity) -> &'static str {
    match l {
        Liquidity::Maker => "maker",
        Liquidity::Taker => "taker",
    }
}
fn liquidity_from(s: &str) -> Option<Liquidity> {
    match s {
        "maker" => Some(Liquidity::Maker),
        "taker" => Some(Liquidity::Taker),
        _ => None,
    }
}

fn order_state_str(s: OrderState) -> &'static str {
    match s {
        OrderState::PendingNew => "pending_new",
        OrderState::Open => "open",
        OrderState::PartiallyFilled => "partially_filled",
        OrderState::PendingCancel => "pending_cancel",
        OrderState::Filled => "filled",
        OrderState::Canceled => "canceled",
        OrderState::Rejected => "rejected",
        OrderState::Expired => "expired",
    }
}
fn order_state_from(s: &str) -> Option<OrderState> {
    Some(match s {
        "pending_new" => OrderState::PendingNew,
        "open" => OrderState::Open,
        "partially_filled" => OrderState::PartiallyFilled,
        "pending_cancel" => OrderState::PendingCancel,
        "filled" => OrderState::Filled,
        "canceled" => OrderState::Canceled,
        "rejected" => OrderState::Rejected,
        "expired" => OrderState::Expired,
        _ => return None,
    })
}

fn lifecycle_phase_str(l: WindowLifecycle) -> &'static str {
    match l {
        WindowLifecycle::Discovered => "discovered",
        WindowLifecycle::Open => "open",
        WindowLifecycle::Closing => "closing",
        WindowLifecycle::Closed => "closed",
        WindowLifecycle::Resolved { .. } => "resolved",
    }
}

fn breaker_str(b: BreakerKind) -> &'static str {
    match b {
        BreakerKind::FeedStale => "feed_stale",
        BreakerKind::WsDisconnect => "ws_disconnect",
        BreakerKind::ClockSkew => "clock_skew",
        BreakerKind::DailyStop => "daily_stop",
        BreakerKind::WindowLoss => "window_loss",
        BreakerKind::ErrorRate => "error_rate",
        BreakerKind::FairVsMid => "fair_vs_mid",
        BreakerKind::Manual => "manual",
        BreakerKind::EngineRestart => "engine_restart",
    }
}
fn breaker_from(s: &str) -> Option<BreakerKind> {
    Some(match s {
        "feed_stale" => BreakerKind::FeedStale,
        "ws_disconnect" => BreakerKind::WsDisconnect,
        "clock_skew" => BreakerKind::ClockSkew,
        "daily_stop" => BreakerKind::DailyStop,
        "window_loss" => BreakerKind::WindowLoss,
        "error_rate" => BreakerKind::ErrorRate,
        "fair_vs_mid" => BreakerKind::FairVsMid,
        "manual" => BreakerKind::Manual,
        "engine_restart" => BreakerKind::EngineRestart,
        _ => return None,
    })
}

fn resolution_kind_str(k: ResolutionKind) -> &'static str {
    match k {
        ResolutionKind::ChainlinkDataStream => "chainlink_data_stream",
        ResolutionKind::BinanceCandle => "binance_candle",
        ResolutionKind::Other => "other",
    }
}
fn resolution_kind_from(s: &str) -> Option<ResolutionKind> {
    Some(match s {
        "chainlink_data_stream" => ResolutionKind::ChainlinkDataStream,
        "binance_candle" => ResolutionKind::BinanceCandle,
        "other" => ResolutionKind::Other,
        _ => return None,
    })
}

fn mode_str(m: Mode) -> &'static str {
    match m {
        Mode::Paper => "paper",
        Mode::Live => "live",
    }
}
fn mode_from(s: &str) -> Option<Mode> {
    match s {
        "paper" => Some(Mode::Paper),
        "live" => Some(Mode::Live),
        _ => None,
    }
}

fn origin_str(o: CommandOrigin) -> &'static str {
    match o {
        CommandOrigin::Dashboard => "dashboard",
        CommandOrigin::Cli => "cli",
    }
}
fn origin_from(s: &str) -> Option<CommandOrigin> {
    match s {
        "dashboard" => Some(CommandOrigin::Dashboard),
        "cli" => Some(CommandOrigin::Cli),
        _ => None,
    }
}

/// Splits a [`RiskEvent`] into its `(kind, breaker)` columns.
fn risk_components(r: RiskEvent) -> (&'static str, BreakerKind) {
    match r {
        RiskEvent::BreakerTripped { breaker } => ("tripped", breaker),
        RiskEvent::BreakerCleared { breaker } => ("cleared", breaker),
        RiskEvent::CancelAllIssued { reason } => ("cancel_all", reason),
    }
}

/// Splits a [`ControlEvent`] into its `(kind, mode?, series?, amount?)` columns.
fn control_components(
    c: &core_types::ControlEvent,
) -> (
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
    Option<String>,
) {
    use core_types::ControlEvent as C;
    match c {
        C::ModeChanged { mode } => ("mode_changed", Some(mode_str(*mode)), None, None),
        C::SeriesEnabled { series } => ("series_enabled", None, Some(series.key()), None),
        C::SeriesDisabled { series } => ("series_disabled", None, Some(series.key()), None),
        C::PaperCapitalSet { amount } => (
            "paper_capital_set",
            None,
            None,
            Some(amount.as_decimal().to_string()),
        ),
        C::Shutdown => ("shutdown", None, None, None),
        C::Kill => ("kill", None, None, None),
        C::Reset => ("reset", None, None, None),
        C::DailyStopReset => ("daily_stop_reset", None, None, None),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use core_types::{
        Asset, ConditionId, ControlEvent, Decimal, Dollars, Event, FeeParams, Fill, Liquidity,
        MarketInfo, OrderId, OrderState, Outcome, Price, ResolutionSource, RiskEvent, RoundDir,
        Series, SettlementSummary, Side, SideInventory, Size, TickSize, TimestampMs, TokenId,
        TokenPair, WindowDuration, WindowId, WindowLifecycle,
    };
    use rust_decimal::dec;

    use super::*;
    use crate::record::JournalRecord;

    const OPEN_MS: i64 = 1_781_000_000_000;
    const CLOSE_MS: i64 = 1_781_000_300_000;

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("journal-index-{tag}-{}.sqlite", std::process::id()))
    }

    fn window() -> WindowId {
        WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(OPEN_MS),
        }
    }
    fn px(d: Decimal) -> Price {
        Price::quantize(d, TickSize::T001, RoundDir::Down).unwrap()
    }
    fn sz(d: Decimal) -> Size {
        Size::new(d).unwrap()
    }
    fn token() -> TokenId {
        TokenId::new("111").unwrap()
    }
    fn market() -> MarketInfo {
        MarketInfo {
            window: window(),
            event_slug: "btc-updown-5m-test".to_owned(),
            condition_id: ConditionId::new(format!("0x{}", "11".repeat(32))).unwrap(),
            tokens: TokenPair {
                up: token(),
                down: TokenId::new("222").unwrap(),
            },
            close_time: TimestampMs::from_millis(CLOSE_MS),
            strike: Some(dec!(60000.5)),
            tick_size: TickSize::T001,
            min_order_size: sz(dec!(5)),
            fees: FeeParams {
                rate: dec!(0.07),
                exponent: 1,
                taker_only: true,
                rebate_rate: dec!(0.2),
                enabled: true,
            },
            neg_risk: false,
            resolution: ResolutionSource::classify("https://data.chain.link/streams/btc-usd"),
        }
    }

    /// Opens a fresh index, indexes the given records (one per seq), commits.
    fn build_index(tag: &str, recs: &[JournalRecord]) -> std::path::PathBuf {
        let path = tmp(tag);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("sqlite-wal"));
        let _ = std::fs::remove_file(path.with_extension("sqlite-shm"));
        let mut idx = JournalIndex::open(&path).unwrap();
        for (i, rec) in recs.iter().enumerate() {
            idx.index(i as u64 + 1, OPEN_MS + i as i64, rec).unwrap();
        }
        idx.commit().unwrap();
        idx.close().unwrap();
        path
    }

    fn rec(event: &Event) -> JournalRecord {
        JournalRecord::from_event(event)
    }

    #[test]
    fn orders_round_trip_through_the_index() {
        let open = Event::OrderUpdate(Arc::new(core_types::OrderUpdate {
            order_id: OrderId::new("paper-1").unwrap(),
            window: window(),
            token_id: token(),
            side: Side::Buy,
            state: OrderState::Open,
            price: px(dec!(0.49)),
            original_size: sz(dec!(20)),
            filled_size: sz(dec!(0)),
            reject_reason: None,
            ts_venue: None,
            ts_local: TimestampMs::from_millis(OPEN_MS),
        }));
        let rejected = Event::OrderUpdate(Arc::new(core_types::OrderUpdate {
            order_id: OrderId::new("paper-2").unwrap(),
            window: window(),
            token_id: token(),
            side: Side::Sell,
            state: OrderState::Rejected,
            price: px(dec!(0.51)),
            original_size: sz(dec!(10)),
            filled_size: sz(dec!(0)),
            reject_reason: Some("would cross".to_owned()),
            ts_venue: Some(TimestampMs::from_millis(OPEN_MS + 7)),
            ts_local: TimestampMs::from_millis(OPEN_MS + 7),
        }));
        let path = build_index("orders", &[rec(&open), rec(&rejected)]);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.orders_for_window(window().series, OPEN_MS).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].order_id.as_str(), "paper-1");
        assert_eq!(rows[0].state, OrderState::Open);
        assert_eq!(rows[0].price, px(dec!(0.49)));
        assert_eq!(rows[0].reject_reason, None);
        assert_eq!(rows[0].ts_venue_ms, None);
        assert_eq!(rows[1].state, OrderState::Rejected);
        assert_eq!(rows[1].side, Side::Sell);
        assert_eq!(rows[1].reject_reason.as_deref(), Some("would cross"));
        assert_eq!(rows[1].ts_venue_ms, Some(OPEN_MS + 7));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fills_round_trip_and_recent_orders_by_seq_desc() {
        let maker = Event::Fill(Arc::new(Fill {
            order_id: OrderId::new("paper-1").unwrap(),
            trade_id: Some("t-1".to_owned()),
            window: window(),
            token_id: token(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: px(dec!(0.40)),
            size: sz(dec!(30)),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: TimestampMs::from_millis(OPEN_MS + 50),
            ts_local: TimestampMs::from_millis(OPEN_MS + 50),
        }));
        let taker = Event::Fill(Arc::new(Fill {
            order_id: OrderId::new("paper-2").unwrap(),
            trade_id: None,
            window: window(),
            token_id: token(),
            outcome: Outcome::Up,
            side: Side::Sell,
            price: px(dec!(0.60)),
            size: sz(dec!(10)),
            liquidity: Liquidity::Taker,
            fee: Dollars::new(dec!(0.0168)),
            ts_venue: TimestampMs::from_millis(OPEN_MS + 90),
            ts_local: TimestampMs::from_millis(OPEN_MS + 90),
        }));
        let path = build_index("fills", &[rec(&maker), rec(&taker)]);
        let reader = JournalIndexReader::open(&path).unwrap();

        let window_fills = reader.fills_for_window(window().series, OPEN_MS).unwrap();
        assert_eq!(window_fills.len(), 2);
        assert_eq!(window_fills[0].liquidity, Liquidity::Maker);
        assert_eq!(window_fills[0].fee, Dollars::ZERO);
        assert_eq!(window_fills[0].trade_id.as_deref(), Some("t-1"));
        assert_eq!(window_fills[1].liquidity, Liquidity::Taker);
        assert_eq!(window_fills[1].fee, Dollars::new(dec!(0.0168)));
        assert_eq!(window_fills[1].trade_id, None);
        assert_eq!(window_fills[1].size, sz(dec!(10)));

        // recent_fills is newest (highest seq) first.
        let recent = reader.recent_fills(10).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].seq, 2);
        assert_eq!(recent[1].seq, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn windows_round_trip_open_and_resolved() {
        let open = Event::Window {
            market: Arc::new(market()),
            lifecycle: WindowLifecycle::Open,
        };
        let resolved = Event::Window {
            market: Arc::new(market()),
            lifecycle: WindowLifecycle::Resolved {
                outcome: Outcome::Up,
            },
        };
        let path = build_index("windows", &[rec(&open), rec(&resolved)]);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.windows().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].lifecycle, "open");
        assert_eq!(rows[0].resolved_outcome, None);
        assert_eq!(rows[0].strike, Some(dec!(60000.5)));
        assert_eq!(rows[0].tick_size, TickSize::T001);
        assert_eq!(rows[0].tokens.up.as_str(), "111");
        assert_eq!(rows[0].tokens.down.as_str(), "222");
        assert_eq!(rows[0].fee_rate, dec!(0.07));
        assert!(!rows[0].neg_risk);
        assert_eq!(rows[0].resolution_kind, ResolutionKind::ChainlinkDataStream);
        assert_eq!(rows[1].lifecycle, "resolved");
        assert_eq!(rows[1].resolved_outcome, Some(Outcome::Up));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn settlements_round_trip() {
        let summary = SettlementSummary::close(
            window(),
            Outcome::Up,
            SideInventory {
                shares: sz(dec!(150)),
                cost: Dollars::new(dec!(60)),
            },
            SideInventory {
                shares: sz(dec!(100)),
                cost: Dollars::new(dec!(50)),
            },
            sz(dec!(20)),
            Dollars::new(dec!(0.25)),
            Dollars::new(dec!(3.75)),
            TimestampMs::from_millis(CLOSE_MS),
        );
        let event = Event::Settlement(Arc::new(summary));
        let path = build_index("settlements", &[rec(&event)]);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.settlements().unwrap();
        assert_eq!(rows.len(), 1);
        let s = &rows[0];
        assert_eq!(s.outcome, Outcome::Up);
        assert_eq!(s.matched_pairs, sz(dec!(100)));
        assert_eq!(s.pair_cost, Some(dec!(0.9)));
        assert_eq!(s.excess, dec!(50));
        assert_eq!(s.merged_pairs, sz(dec!(20)));
        assert_eq!(s.fees_paid, Dollars::new(dec!(0.25)));
        assert_eq!(s.realized_pnl, Dollars::new(dec!(3.75)));
        assert_eq!(s.settle_ms, CLOSE_MS);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn breaker_trips_round_trip_all_kinds() {
        let events = [
            Event::Risk(RiskEvent::BreakerTripped {
                breaker: BreakerKind::ClockSkew,
            }),
            Event::Risk(RiskEvent::BreakerCleared {
                breaker: BreakerKind::ClockSkew,
            }),
            Event::Risk(RiskEvent::CancelAllIssued {
                reason: BreakerKind::FeedStale,
            }),
        ];
        let recs: Vec<JournalRecord> = events.iter().map(rec).collect();
        let path = build_index("breakers", &recs);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.breaker_trips().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].kind, "tripped");
        assert_eq!(rows[0].breaker, BreakerKind::ClockSkew);
        assert_eq!(rows[1].kind, "cleared");
        assert_eq!(rows[2].kind, "cancel_all");
        assert_eq!(rows[2].breaker, BreakerKind::FeedStale);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn control_events_round_trip() {
        let events = [
            Event::Control(ControlEvent::PaperCapitalSet {
                amount: Dollars::new(dec!(12345.67)),
            }),
            Event::Control(ControlEvent::ModeChanged {
                mode: core_types::Mode::Paper,
            }),
            Event::Control(ControlEvent::SeriesEnabled {
                series: Series {
                    asset: Asset::Eth,
                    duration: WindowDuration::H1,
                },
            }),
            Event::Control(ControlEvent::Kill),
            Event::Control(ControlEvent::Reset),
        ];
        let recs: Vec<JournalRecord> = events.iter().map(rec).collect();
        let path = build_index("controls", &recs);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.control_events().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].kind, "paper_capital_set");
        assert_eq!(rows[0].amount, Some(Dollars::new(dec!(12345.67))));
        assert_eq!(rows[1].kind, "mode_changed");
        assert_eq!(rows[1].mode, Some(core_types::Mode::Paper));
        assert_eq!(rows[2].kind, "series_enabled");
        assert_eq!(
            rows[2].series,
            Some(Series {
                asset: Asset::Eth,
                duration: WindowDuration::H1,
            })
        );
        assert_eq!(rows[3].kind, "kill");
        assert_eq!(rows[4].kind, "reset");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn high_rate_variants_are_noops() {
        let tick = Event::PriceTick(core_types::PriceTick {
            source: core_types::PriceSource::BinanceDirect,
            asset: Asset::Btc,
            kind: core_types::TickKind::Mid,
            value: dec!(60000),
            ts_exchange: TimestampMs::from_millis(OPEN_MS),
            ts_local: TimestampMs::from_millis(OPEN_MS),
        });
        let path = tmp("highrate");
        let _ = std::fs::remove_file(&path);
        let mut idx = JournalIndex::open(&path).unwrap();
        for i in 0..50 {
            idx.index(i + 1, OPEN_MS + i as i64, &rec(&tick)).unwrap();
        }
        idx.commit().unwrap();
        assert_eq!(idx.indexed(), 0, "high-rate ticks must not be indexed");
        idx.close().unwrap();
        let reader = JournalIndexReader::open(&path).unwrap();
        assert!(reader.recent_fills(10).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn schema_version_meta_written() {
        let path = build_index("schema", &[]);
        let reader = JournalIndexReader::open(&path).unwrap();
        assert_eq!(reader.schema_version().unwrap(), SCHEMA_VERSION);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reopen_read_only_sees_committed_rows_while_writer_open() {
        // WAL lets a separate read-only connection see committed rows while the
        // writer connection is still open.
        let path = tmp("wal");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("sqlite-wal"));
        let _ = std::fs::remove_file(path.with_extension("sqlite-shm"));
        let fill = Event::Fill(Arc::new(Fill {
            order_id: OrderId::new("paper-1").unwrap(),
            trade_id: Some("t-1".to_owned()),
            window: window(),
            token_id: token(),
            outcome: Outcome::Up,
            side: Side::Buy,
            price: px(dec!(0.40)),
            size: sz(dec!(30)),
            liquidity: Liquidity::Maker,
            fee: Dollars::ZERO,
            ts_venue: TimestampMs::from_millis(OPEN_MS + 50),
            ts_local: TimestampMs::from_millis(OPEN_MS + 50),
        }));
        let mut idx = JournalIndex::open(&path).unwrap();
        idx.index(1, OPEN_MS, &rec(&fill)).unwrap();
        idx.commit().unwrap();

        // Reader opens while the writer connection is still alive.
        let reader = JournalIndexReader::open(&path).unwrap();
        assert_eq!(reader.recent_fills(10).unwrap().len(), 1);

        idx.close().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prune_deletes_index_rows_older_than_cutoff() {
        // Three settlements at ts_local 100/200/300; prune(250) leaves only 300.
        let path = tmp("prune");
        let _ = std::fs::remove_file(&path);
        let mut idx = JournalIndex::open(&path).unwrap();
        for (i, ts) in [100i64, 200, 300].into_iter().enumerate() {
            let summary = SettlementSummary::close(
                window(),
                Outcome::Up,
                SideInventory {
                    shares: sz(dec!(10)),
                    cost: Dollars::new(dec!(4)),
                },
                SideInventory::default(),
                Size::ZERO,
                Dollars::ZERO,
                Dollars::new(dec!(6)),
                TimestampMs::from_millis(ts),
            );
            idx.index(
                i as u64 + 1,
                ts,
                &rec(&Event::Settlement(Arc::new(summary))),
            )
            .unwrap();
        }
        idx.commit().unwrap();
        let removed = idx.prune(250).unwrap();
        assert_eq!(removed, 2);
        idx.close().unwrap();

        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.settlements().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].settle_ms, 300);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn corrupt_decimal_cell_is_rejected_not_coerced() {
        // Hand-write a bad price into the fills table; the reader must error.
        let path = tmp("corrupt");
        let _ = std::fs::remove_file(&path);
        let idx = JournalIndex::open(&path).unwrap();
        idx.conn
            .execute(
                "INSERT INTO fills
                 (seq, ts_local_ms, order_id, trade_id, series, open_ms, token_id, outcome,
                  side, price, size, liquidity, fee, ts_venue_ms)
                 VALUES (1, 0, 'o', NULL, 'BTC-5m', 0, '1', 'up', 'buy', 'not-a-number',
                         '10', 'maker', '0', 0)",
                [],
            )
            .unwrap();
        idx.close().unwrap();
        let reader = JournalIndexReader::open(&path).unwrap();
        let err = reader.recent_fills(10).unwrap_err();
        assert!(matches!(err, JournalError::Sqlite(_)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn command_audit_round_trips_accepted_and_rejected() {
        use core_types::{CommandOrigin, ControlAudit};

        let accepted = Event::ControlAudit(Arc::new(ControlAudit {
            ts: TimestampMs::from_millis(OPEN_MS),
            origin: CommandOrigin::Cli,
            kind: "set_param".to_owned(),
            detail: "series=BTC-5m key=min_edge value=0.02".to_owned(),
            accepted: true,
            error: None,
        }));
        let rejected = Event::ControlAudit(Arc::new(ControlAudit {
            ts: TimestampMs::from_millis(OPEN_MS + 1),
            origin: CommandOrigin::Dashboard,
            kind: "set_param".to_owned(),
            detail: "key=touch_size".to_owned(),
            accepted: false,
            error: Some("touch_size requires a restart".to_owned()),
        }));
        let path = build_index("audit", &[rec(&accepted), rec(&rejected)]);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.command_audit().unwrap();
        assert_eq!(rows.len(), 2, "one audit row per command, accepted or not");
        assert_eq!(rows[0].origin, CommandOrigin::Cli);
        assert_eq!(rows[0].kind, "set_param");
        assert!(rows[0].accepted);
        assert_eq!(rows[0].error, None);
        assert_eq!(rows[1].origin, CommandOrigin::Dashboard);
        assert!(!rows[1].accepted);
        assert_eq!(
            rows[1].error.as_deref(),
            Some("touch_size requires a restart")
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn daily_stop_reset_indexes_into_control_events() {
        let ev = Event::Control(ControlEvent::DailyStopReset);
        let path = build_index("dsr", &[rec(&ev)]);
        let reader = JournalIndexReader::open(&path).unwrap();
        let rows = reader.control_events().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "daily_stop_reset");
        let _ = std::fs::remove_file(&path);
    }
}
