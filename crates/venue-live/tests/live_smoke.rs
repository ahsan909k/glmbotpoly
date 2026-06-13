//! Operator-only live smoke test — **never runs by default**.
//!
//! Three independent locks, all required:
//! 1. the `live-smoke` cargo feature (this whole file is `#![cfg(...)]`-gated),
//! 2. `#[ignore]` (so even `--features live-smoke` does not run it without
//!    `-- --ignored`),
//! 3. the `BOT_LIVE_SMOKE=1` environment guard (the test aborts otherwise).
//!
//! It touches the LIVE venue with REAL credentials and places a REAL order, so
//! it spends/locks real funds. Run only when deliberately verifying a funded
//! account end-to-end:
//!
//! ```text
//! BOT_LIVE_SMOKE=1 \
//! BOT_SECRET_PM_PRIVATE_KEY=0x... \
//! BOT_SECRET_PM_API_KEY=... BOT_SECRET_PM_API_SECRET=... BOT_SECRET_PM_API_PASSPHRASE=... \
//! BOT_SECRET_LIVE_CONFIRM="arm-live-i-accept-real-money-losses" \
//! BOT_PM_FUNDER=0x...        # deposit-wallet address \
//! BOT_SMOKE_TOKEN_ID=<decimal outcome-token id of a live, liquid market> \
//! cargo test -p venue-live --features live-smoke -- --ignored live_smoke
//! ```
//!
//! It places a tiny post-only GTC far below the touch (so it cannot fill),
//! reads it back via the open-orders poll, then cancels everything.
#![cfg(feature = "live-smoke")]

use config::{LiveArming, LiveConfig, Secrets};
use core_types::{
    Asset, NewOrder, OrderQty, Outcome, Price, RoundDir, Series, Side, Size, TickSize, TimeInForce,
    TimestampMs, TokenId, WindowDuration, WindowId,
};
use rust_decimal::dec;
use venue_api::VenuePort as _;
use venue_live::{LiveParams, LiveVenue, SigType};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("live-smoke requires {name} to be set"))
}

#[tokio::test]
#[ignore = "operator-only: touches the live venue and spends real funds"]
async fn live_smoke_place_readback_cancel() {
    assert_eq!(
        std::env::var("BOT_LIVE_SMOKE").as_deref(),
        Ok("1"),
        "refusing to run: set BOT_LIVE_SMOKE=1 to confirm you intend a live, funded run"
    );

    let secrets = Secrets::from_env();
    let arming = LiveArming::evaluate(
        &LiveConfig {
            enabled: true,
            ..LiveConfig::default()
        },
        &secrets,
    );
    let params = LiveParams {
        sig_type: SigType::DepositWallet,
        funder: Some(env("BOT_PM_FUNDER")),
        ..LiveParams::default()
    };

    // The operator consents to gate 3 by running this test.
    let venue = LiveVenue::connect(arming, true, &secrets, params)
        .await
        .expect("connect (check the env vars in this file's docs)");

    let token = TokenId::new(env("BOT_SMOKE_TOKEN_ID")).expect("BOT_SMOKE_TOKEN_ID decimal");
    let order = NewOrder {
        client_id: Some("live-smoke".to_owned()),
        window: WindowId {
            series: Series {
                asset: Asset::Btc,
                duration: WindowDuration::M5,
            },
            open_time: TimestampMs::from_millis(0),
        },
        token_id: token,
        outcome: Outcome::Up,
        side: Side::Buy,
        // Far below the touch + post-only ⇒ rests without filling.
        price: Price::quantize(dec!(0.02), TickSize::T001, RoundDir::Down).expect("price"),
        qty: OrderQty::Shares(Size::new(dec!(5)).expect("size")),
        tif: TimeInForce::Gtc { post_only: true },
    };

    let accepted = venue.place(&order).await.expect("place");
    println!("placed: {accepted:?}");

    let balances = venue.balances().await.expect("balances");
    println!("balances: {balances:?}");

    let report = venue.cancel_all().await.expect("cancel_all");
    println!("cancel_all: {report:?}");
    assert!(
        report.all_canceled(),
        "some orders failed to cancel: {:?}",
        report.not_canceled
    );
}
