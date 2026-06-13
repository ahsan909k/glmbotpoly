//! Temporary probe (review verification): does an env-var key shaped like
//! `BOT_DASHBOARD__AUTH_TOKEN` pass the env filter and trip the
//! forbidden-file-key check?

#![allow(clippy::result_large_err)]

use std::path::Path;

use config::{ConfigError, load};

#[test]
fn env_dotted_secret_key_behavior() {
    figment::Jail::expect_with(|jail| {
        jail.create_dir("config")?;
        jail.create_file("config/default.toml", "")?;
        jail.set_env("BOT_DASHBOARD__AUTH_TOKEN", "sneaky-token");
        match load(Path::new("config")) {
            Ok(_) => println!("PROBE RESULT: load Ok — env key did NOT trip anything"),
            Err(ConfigError::SecretInConfigFile { key }) => {
                println!("PROBE RESULT: SecretInConfigFile fired for key {key}");
                println!("PROBE MESSAGE: {}", ConfigError::SecretInConfigFile { key });
            }
            Err(other) => println!("PROBE RESULT: other error: {other}"),
        }
        Ok(())
    });
}

#[test]
fn env_dotted_secret_key_live_api_key() {
    figment::Jail::expect_with(|jail| {
        jail.create_dir("config")?;
        jail.create_file("config/default.toml", "")?;
        jail.set_env("BOT_LIVE__API_KEY", "sneaky-key");
        match load(Path::new("config")) {
            Ok(_) => println!("PROBE RESULT 2: load Ok"),
            Err(e) => println!("PROBE RESULT 2: {e}"),
        }
        Ok(())
    });
}
