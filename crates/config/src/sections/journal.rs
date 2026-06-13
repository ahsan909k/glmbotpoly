//! Journal storage locations (append-only event log + raw tick files).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::validate::Violations;

/// Where the journal crate persists its data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JournalConfig {
    /// SQLite database file for the append-only event log.
    pub sqlite_path: PathBuf,
    /// Directory for raw tick capture files.
    pub tick_dir: PathBuf,
}

impl Default for JournalConfig {
    fn default() -> Self {
        Self {
            sqlite_path: PathBuf::from("data/journal.sqlite"),
            tick_dir: PathBuf::from("data/ticks"),
        }
    }
}

impl JournalConfig {
    pub(crate) fn validate_into(&self, v: &mut Violations) {
        v.require(
            !self.sqlite_path.as_os_str().is_empty(),
            "journal.sqlite_path",
            "must not be empty",
        );
        v.require(
            !self.tick_dir.as_os_str().is_empty(),
            "journal.tick_dir",
            "must not be empty",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let mut v = Violations::default();
        JournalConfig::default().validate_into(&mut v);
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn rejects_empty_paths() {
        let cfg = JournalConfig {
            sqlite_path: PathBuf::new(),
            tick_dir: PathBuf::new(),
        };
        let mut v = Violations::default();
        cfg.validate_into(&mut v);
        assert_eq!(v.into_result().unwrap_err().len(), 2);
    }
}
