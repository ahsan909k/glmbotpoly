//! Strongly-typed Polymarket identifiers.
//!
//! All three ids are validated at construction and deserialize through the
//! same validation (`#[serde(try_from = "String")]`), so a corrupt journal row
//! or API response cannot smuggle an invalid id into the system.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Errors from id validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    /// Condition ids are `0x` followed by exactly 64 hex characters.
    #[error("condition id must be 0x + 64 hex chars, got {0:?}")]
    BadConditionId(String),
    /// Token ids are non-empty ASCII-decimal strings (256-bit unsigned ints).
    #[error("token id must be a non-empty decimal string, got {0:?}")]
    BadTokenId(String),
    /// Order ids are opaque but must be non-empty.
    #[error("order id must be non-empty")]
    EmptyOrderId,
}

/// A Polymarket condition id: `0x` + 64 hex chars (32 bytes), stored
/// lowercase-normalized so equality and hashing are case-insensitive.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ConditionId(String);

impl ConditionId {
    /// Validates and lowercase-normalizes a condition id.
    ///
    /// # Errors
    /// [`IdError::BadConditionId`] unless the input is `0x` + 64 hex chars.
    pub fn new(s: impl Into<String>) -> Result<Self, IdError> {
        let s: String = s.into();
        let ok =
            s.len() == 66 && s.starts_with("0x") && s[2..].bytes().all(|b| b.is_ascii_hexdigit());
        if ok {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(IdError::BadConditionId(s))
        }
    }

    /// The id as a lowercase `0x…` string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConditionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ConditionId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<ConditionId> for String {
    fn from(id: ConditionId) -> Self {
        id.0
    }
}

/// A CLOB outcome-token id: a 256-bit unsigned integer in decimal form.
/// Kept as a string — it does not fit in `u128`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TokenId(String);

impl TokenId {
    /// Validates a token id: non-empty, ASCII digits only (no sign, no `0x`).
    ///
    /// # Errors
    /// [`IdError::BadTokenId`] on any other input.
    pub fn new(s: impl Into<String>) -> Result<Self, IdError> {
        let s: String = s.into();
        if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
            Ok(Self(s))
        } else {
            Err(IdError::BadTokenId(s))
        }
    }

    /// The id as a decimal string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for TokenId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<TokenId> for String {
    fn from(id: TokenId) -> Self {
        id.0
    }
}

/// A venue-assigned order id. Opaque; only required to be non-empty.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct OrderId(String);

impl OrderId {
    /// Validates an order id (non-empty).
    ///
    /// # Errors
    /// [`IdError::EmptyOrderId`] if the input is empty.
    pub fn new(s: impl Into<String>) -> Result<Self, IdError> {
        let s: String = s.into();
        if s.is_empty() {
            Err(IdError::EmptyOrderId)
        } else {
            Ok(Self(s))
        }
    }

    /// The id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for OrderId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<OrderId> for String {
    fn from(id: OrderId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEX64: &str = "AbCdEf0123456789aBcDeF0123456789abcdef0123456789ABCDEF0123456789";

    #[test]
    fn condition_id_valid_and_lowercased() {
        let id = ConditionId::new(format!("0x{HEX64}")).unwrap();
        assert_eq!(id.as_str(), format!("0x{}", HEX64.to_ascii_lowercase()));
    }

    #[test]
    fn condition_id_rejects_bad_forms() {
        // Missing 0x prefix.
        assert!(ConditionId::new(format!("00{HEX64}")).is_err());
        // 65 chars of hex (67 total).
        assert!(ConditionId::new(format!("0x{HEX64}0")).is_err());
        // 63 chars of hex (65 total).
        assert!(ConditionId::new(format!("0x{}", &HEX64[1..])).is_err());
        // Non-hex character.
        assert!(ConditionId::new(format!("0xg{}", &HEX64[1..])).is_err());
        assert!(ConditionId::new("").is_err());
    }

    #[test]
    fn token_id_valid_and_invalid() {
        assert!(TokenId::new("123456789012345678901234567890").is_ok());
        assert!(TokenId::new("7").is_ok());
        assert!(TokenId::new("").is_err());
        assert!(TokenId::new("0x12").is_err());
        assert!(TokenId::new("12a").is_err());
        assert!(TokenId::new("-12").is_err());
        assert!(TokenId::new("+12").is_err());
    }

    #[test]
    fn order_id_nonempty() {
        assert!(OrderId::new("abc-123").is_ok());
        assert_eq!(OrderId::new(""), Err(IdError::EmptyOrderId));
    }

    #[test]
    fn serde_rejects_invalid_input() {
        let ok: Result<TokenId, _> = serde_json::from_str("\"123\"");
        assert!(ok.is_ok());
        let bad: Result<TokenId, _> = serde_json::from_str("\"0x123\"");
        assert!(bad.is_err());
        let bad: Result<ConditionId, _> = serde_json::from_str("\"0xzz\"");
        assert!(bad.is_err());
        let bad: Result<OrderId, _> = serde_json::from_str("\"\"");
        assert!(bad.is_err());
    }

    #[test]
    fn serde_roundtrip_preserves_value() {
        let id = TokenId::new("99887766554433221100").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"99887766554433221100\"");
        let back: TokenId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }
}
