//! End-to-end test of the wired path: spawn the real binary, point it at a
//! config directory, and assert on exit code and output — including that no
//! secret value can ever appear on stdout.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The binary with every ambient `BOT_*` variable scrubbed, so a developer's
/// shell environment cannot change test outcomes. Case-insensitive: figment
/// matches env keys case-insensitively, so `bot_paper__x` would count too.
fn bot_command() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bot"));
    for (key, _) in std::env::vars_os() {
        if key
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("BOT_")
        {
            cmd.env_remove(&key);
        }
    }
    cmd
}

fn repo_config_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")
}

#[test]
fn check_config_accepts_committed_defaults_and_redacts_secrets() {
    let secret_value = "super-secret-token-value-12345";
    let output = bot_command()
        .arg("check-config")
        .arg("--config-dir")
        .arg(repo_config_dir())
        .env("BOT_SECRET_DASHBOARD_TOKEN", secret_value)
        .output()
        .expect("binary should spawn");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "expected success, stderr: {stderr}"
    );
    // The effective config is printed...
    assert!(stdout.contains("configuration: OK"));
    assert!(stdout.contains("starting_capital"));
    // ...the secret's presence is reported...
    assert!(stdout.contains("BOT_SECRET_DASHBOARD_TOKEN: set"));
    // ...but its value never appears anywhere.
    assert!(!stdout.contains(secret_value));
    assert!(!stderr.contains(secret_value));
}

#[test]
fn check_config_rejects_invalid_fixture_naming_every_violation() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/invalid");
    let output = bot_command()
        .arg("check-config")
        .arg("--config-dir")
        .arg(fixture)
        .output()
        .expect("binary should spawn");

    assert!(!output.status.success(), "invalid config must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Both violations reported in one pass, each naming its key.
    assert!(
        stderr.contains("paper.starting_capital"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("min_edge"), "stderr: {stderr}");
}

#[test]
fn missing_config_dir_is_a_clear_error() {
    let output = bot_command()
        .arg("check-config")
        .arg("--config-dir")
        .arg("does/not/exist")
        .output()
        .expect("binary should spawn");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("default.toml"), "stderr: {stderr}");
}

#[test]
fn paper_boot_logs_redacted_config_flushes_file_and_exits_cleanly() {
    let secret_value = "paper-secret-value-98765";
    let log_dir = std::env::temp_dir().join(format!("bot-paper-boot-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&log_dir);

    let output = bot_command()
        .arg("paper")
        .arg("--config-dir")
        .arg(repo_config_dir())
        .env("BOT_SECRET_DASHBOARD_TOKEN", secret_value)
        .env("BOT_LOG__DIR", &log_dir)
        .output()
        .expect("binary should spawn");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "paper boot must exit cleanly, stderr: {stderr}"
    );
    // The console layer carried the whole boot sequence...
    assert!(
        stdout.contains("effective configuration"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("scaffold boot complete"),
        "stdout: {stdout}"
    );
    // ...with the secret's presence reported but its value redacted everywhere.
    assert!(stdout.contains("BOT_SECRET_DASHBOARD_TOKEN"));
    assert!(!stdout.contains(secret_value));
    assert!(!stderr.contains(secret_value));

    // The rolling file was created in the configured dir and fully flushed on
    // exit (WorkerGuard drop): its last lines include the final boot message.
    let log_file = std::fs::read_dir(&log_dir)
        .expect("log dir must exist")
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("bot."))
        .expect("a rolling log file must exist");
    let contents = std::fs::read_to_string(log_file.path()).expect("log file must be readable");
    assert!(
        contents.contains("scaffold boot complete"),
        "log not flushed"
    );
    assert!(
        !contents.contains(secret_value),
        "secret leaked into log file"
    );

    let _ = std::fs::remove_dir_all(&log_dir);
}

#[test]
fn live_subcommand_refuses_when_disarmed() {
    let output = bot_command()
        .arg("live")
        .arg("--config-dir")
        .arg(repo_config_dir())
        .output()
        .expect("binary should spawn");

    assert!(!output.status.success(), "live must refuse when disarmed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("live mode refused"), "stderr: {stderr}");
}
