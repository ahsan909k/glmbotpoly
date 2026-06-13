//! Wire-format structs mirroring the Gamma and CLOB REST responses.
//!
//! Deliberately permissive: no `deny_unknown_fields`, every field optional
//! unless structurally required — the APIs add fields freely and discovery
//! must not break when they do. Strictness lives in [`crate::map`], which
//! turns these into validated [`core_types::MarketInfo`] values.

pub mod clob;
pub mod gamma;
