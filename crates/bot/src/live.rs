//! Live-venue wiring for the binary: the non-secret [`LiveParams`] mapping and
//! the read-only `venue-check` demonstration.
//!
//! `live_params` mirrors the `vol_params`/`health_params` boundary-mapping
//! convention: host / chain id / signature type are venue-live code defaults;
//! only the operator-specific, non-secret funder address comes from config
//! (`[live].funder`).
//!
//! `venue_check` builds representative orders through the same translation the
//! live adapter uses ([`venue_live::build`]) and prints the constructed (unsent,
//! unsigned) requests — a §12.6 demonstration that needs no credentials and
//! makes no network call. Real signing + submission is the operator-only
//! `live-smoke` test.

use anyhow::Context as _;
use config::AppConfig;
use core_types::{
    Asset, Decimal, Dollars, DurationMs, NewOrder, OrderQty, Outcome, Price, RoundDir, Series,
    Side, Size, TickSize, TimeInForce, TimestampMs, TokenId, WindowDuration, WindowId,
};
use venue_live::{BuiltOrder, LiveParams};

/// Maps config + code defaults into the non-secret [`LiveParams`].
#[must_use]
pub fn live_params(config: &AppConfig) -> LiveParams {
    LiveParams {
        funder: config.live.funder.clone(),
        ..LiveParams::default()
    }
}

/// Read-only demonstration: prints the live params and the constructed
/// [`BuiltOrder`] for one of each order class, exercising
/// [`venue_live::build`] offline.
pub fn venue_check(config: &AppConfig) -> anyhow::Result<()> {
    let params = live_params(config);
    let now = timeutil::wall_now();

    println!("live params (non-secret):");
    println!("  clob_host   = {}", params.clob_host);
    println!("  chain_id    = {}", params.chain_id);
    println!("  sig_type    = {:?}", params.sig_type);
    println!(
        "  funder      = {}",
        params
            .funder
            .as_deref()
            .unwrap_or("<unset — required before arming live>")
    );
    match params.validate() {
        Ok(()) => println!("  validate()  = OK"),
        Err(e) => println!("  validate()  = {e} (expected until [live].funder is set)"),
    }

    println!("\nconstructed (unsent, unsigned) requests for each order class:");
    for (label, order) in representative_orders(now)? {
        match venue_live::build(&order, now, &params) {
            Ok(built) => print_built(label, &built),
            Err(e) => println!("  {label}: BUILD ERROR: {e}"),
        }
    }

    println!(
        "\n`bot live` constructs LiveVenue::connect, which refuses (NotArmed) until \
         all three §11 gates pass; signing + submission is the operator-only \
         `cargo test -p venue-live --features live-smoke -- --ignored` path."
    );
    Ok(())
}

fn print_built(label: &str, built: &BuiltOrder) {
    println!("  {label}:");
    println!("    token_id  = {}", built.token_id);
    println!("    side      = {:?}", built.side);
    println!("    price     = {}", built.price.as_decimal());
    println!("    amount    = {:?}", built.amount);
    println!("    class     = {:?}", built.class);
    println!("    sig_type  = {:?}", built.sig_type);
    println!(
        "    funder    = {}",
        built.funder.as_deref().unwrap_or("<none>")
    );
}

fn mk_price(s: &str) -> anyhow::Result<Price> {
    let d: Decimal = s.parse().context("parsing demo price")?;
    Price::quantize(d, TickSize::T001, RoundDir::Down).context("quantizing demo price")
}

fn mk_shares(n: i64) -> anyhow::Result<OrderQty> {
    Ok(OrderQty::Shares(
        Size::new(Decimal::from(n)).context("building demo share size")?,
    ))
}

/// One representative [`NewOrder`] per class, on a synthetic BTC-5m window.
fn representative_orders(now: TimestampMs) -> anyhow::Result<Vec<(&'static str, NewOrder)>> {
    let window = WindowId {
        series: Series {
            asset: Asset::Btc,
            duration: WindowDuration::M5,
        },
        open_time: TimestampMs::from_millis(1_781_320_800_000),
    };
    let token = TokenId::new("123456789").context("building demo token id")?;

    let base = |side, qty, tif, price| NewOrder {
        client_id: Some("venue-check".to_owned()),
        window,
        token_id: token.clone(),
        outcome: Outcome::Up,
        side,
        price,
        qty,
        tif,
    };

    Ok(vec![
        (
            "GTC post-only BUY",
            base(
                Side::Buy,
                mk_shares(10)?,
                TimeInForce::Gtc { post_only: true },
                mk_price("0.40")?,
            ),
        ),
        (
            "GTD post-only SELL (expiry in 10s → floored to ≥ now+60s)",
            base(
                Side::Sell,
                mk_shares(7)?,
                TimeInForce::Gtd {
                    expires_at: now.saturating_add(DurationMs::from_secs(10)),
                    post_only: true,
                },
                mk_price("0.55")?,
            ),
        ),
        (
            "FAK marketable BUY (dollar notional)",
            base(
                Side::Buy,
                OrderQty::Notional(Dollars::new(Decimal::from(25))),
                TimeInForce::Fak,
                mk_price("0.60")?,
            ),
        ),
        (
            "FOK marketable SELL (share count)",
            base(
                Side::Sell,
                mk_shares(10)?,
                TimeInForce::Fok,
                mk_price("0.45")?,
            ),
        ),
    ])
}
