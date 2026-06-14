//! Architecture lint (CLAUDE.md §4): `engine` must depend on the venue *port*
//! (`venue-api`) and **never** on a concrete adapter (`venue-live` /
//! `venue-paper`), so the strategy/engine paths are identical across venues and
//! the dependency arrows point inward only.
//!
//! This guards the Cargo manifest directly: the compiler only complains when a
//! forbidden crate is actually *used*, but the rule is stronger — "not even
//! declared". A section-aware scan of `engine`'s own `Cargo.toml` (no
//! `cargo_metadata` dependency — not on the §3 allowlist) is enough; hyphens and
//! underscores are normalized so either spelling is caught.

use std::collections::HashSet;

/// Collects the dependency keys declared under every `[*dependencies]` table,
/// normalizing `_` → `-` so `venue_live` and `venue-live` are the same key.
fn dependency_keys(manifest: &str) -> HashSet<String> {
    let mut keys = HashSet::new();
    let mut in_deps = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            // A dependency table is any section whose name ends in
            // "dependencies": [dependencies], [dev-dependencies],
            // [build-dependencies], [target.'cfg(...)'.dependencies].
            let name = line.trim_start_matches('[').trim_end_matches(']').trim();
            in_deps = name.ends_with("dependencies");
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        // The dependency key is the first token before `=` (value) or `.`
        // (dotted form like `tokio.workspace = true`).
        let key = line
            .split(['=', '.'])
            .next()
            .unwrap_or("")
            .trim()
            .trim_matches('"')
            .replace('_', "-");
        if !key.is_empty() {
            keys.insert(key);
        }
    }
    keys
}

#[test]
fn engine_depends_only_on_the_venue_port() {
    let manifest = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
    let deps = dependency_keys(manifest);

    assert!(
        deps.contains("venue-api"),
        "engine must depend on the venue port `venue-api` (declared deps: {deps:?})"
    );
    assert!(
        !deps.contains("venue-live"),
        "engine must NOT depend on `venue-live` — only the venue-api port (CLAUDE.md §4)"
    );
    assert!(
        !deps.contains("venue-paper"),
        "engine must NOT depend on `venue-paper` — only the venue-api port (CLAUDE.md §4)"
    );
}

#[test]
fn dependency_key_scan_handles_table_and_dotted_forms() {
    // Guards the parser itself against the manifest shapes it must understand.
    let sample = "\
[package]\n\
name = \"x\"\n\
[dependencies]\n\
venue-api = { path = \"../venue-api\" }\n\
# a comment\n\
tokio.workspace = true\n\
reqwest = { default-features = false }\n\
[dev-dependencies]\n\
rust_decimal = { workspace = true }\n\
[lints]\n\
workspace = true\n";
    let keys = dependency_keys(sample);
    assert!(keys.contains("venue-api"));
    assert!(keys.contains("tokio"));
    assert!(keys.contains("reqwest")); // not "default-features"
    assert!(keys.contains("rust-decimal")); // underscore normalized
    assert!(!keys.contains("workspace")); // [lints].workspace is not a dependency
    assert!(!keys.contains("name")); // [package].name is not a dependency
}
