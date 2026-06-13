//! The real [`ClobPort`] backed by the official Polymarket SDK, and the gated
//! [`LiveVenue::connect`] / [`LiveVenue::dry_run`] constructors.
//!
//! This is the only module that names SDK types. Everything else in the crate
//! is exercised offline through [`FakeClobPort`](crate::FakeClobPort); this
//! module is exercised by the operator-only `live-smoke` test and the
//! `bot venue-check` demonstration.
//!
//! `connect` is the sole network-capable, order-submitting path and runs the
//! §11 arming gate first. `dry_run` builds and signs real orders against live
//! market params but never posts one (its `submit`/`submit_batch` return
//! synthetic acks after signing).

use std::str::FromStr as _;
use std::time::Duration;

use alloy_signer_local::PrivateKeySigner;
use chrono::{DateTime, Utc};
use config::{LiveArming, Secrets};
use core_types::{ConditionId, Dollars, OrderId, OrderQty, Side, Size, TimestampMs};
use polymarket_client_sdk_v2::auth::state::Authenticated;
use polymarket_client_sdk_v2::auth::{Credentials, Normal, Signer as _, Uuid};
use polymarket_client_sdk_v2::clob::types::request::{
    BalanceAllowanceRequest, CancelMarketOrderRequest, OrdersRequest,
};
use polymarket_client_sdk_v2::clob::types::response::{CancelOrdersResponse, PostOrderResponse};
use polymarket_client_sdk_v2::clob::types::{
    Amount, AssetType, OrderType, Side as SdkSide, SignatureType, SignedOrder,
};
use polymarket_client_sdk_v2::clob::{Client, Config};
use polymarket_client_sdk_v2::error::{Error as SdkError, Status};
use polymarket_client_sdk_v2::types::{Address, B256, U256};
use venue_api::Wallet;

use crate::arming::check_arming;
use crate::convert::{BuiltOrder, OrderClass};
use crate::error::{VenueLiveError, map_status_error};
use crate::params::{LiveParams, SigType};
use crate::port::{ClobPort, RawAck, RawCancel, RawOpenOrder};
use crate::reconcile::reconcile_loop;
use crate::venue::LiveVenue;

/// How often the interim reconcile loop polls open orders.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(2);

/// Pagination safety cap for `open_orders`.
const MAX_OPEN_ORDER_PAGES: usize = 50;

/// The live SDK-backed network layer.
pub struct SdkClobPort {
    client: Client<Authenticated<Normal>>,
    signer: PrivateKeySigner,
    /// When true, orders are built and signed but never posted.
    dry_run: bool,
}

impl SdkClobPort {
    /// Builds the signer, authenticates the SDK client (L1 → L2), and returns
    /// the backend. `dry_run` controls whether `submit` posts.
    async fn authenticate(
        secrets: &Secrets,
        params: &LiveParams,
        dry_run: bool,
    ) -> Result<Self, VenueLiveError> {
        let key = secrets
            .pm_private_key
            .as_ref()
            .ok_or(VenueLiveError::MissingCredentials)?;
        let signer = PrivateKeySigner::from_str(key.expose())
            .map_err(|e| VenueLiveError::Auth(format!("invalid private key: {e}")))?
            .with_chain_id(Some(params.chain_id));

        let client =
            Client::new(&params.clob_host, Config::default()).map_err(|e| map_sdk_error(&e))?;
        let mut builder = client.authentication_builder(&signer);
        builder = match params.sig_type {
            SigType::Eoa => builder.signature_type(SignatureType::Eoa),
            SigType::DepositWallet => {
                let funder = params
                    .funder
                    .as_deref()
                    .ok_or(VenueLiveError::MissingFunder)?;
                let address = Address::from_str(funder)
                    .map_err(|_| VenueLiveError::BadFunder(funder.to_owned()))?;
                builder
                    .funder(address)
                    .signature_type(SignatureType::Poly1271)
            }
        };
        // Reuse the operator's existing L2 API credentials when all are present
        // (the §11 gate guarantees them for `connect`). Absent any, `authenticate()`
        // create-or-derives them from the signer.
        if let (Some(key), Some(secret), Some(passphrase)) = (
            secrets.pm_api_key.as_ref(),
            secrets.pm_api_secret.as_ref(),
            secrets.pm_api_passphrase.as_ref(),
        ) {
            let api_key = Uuid::parse_str(key.expose())
                .map_err(|e| VenueLiveError::Auth(format!("API key is not a UUID: {e}")))?;
            builder = builder.credentials(Credentials::new(
                api_key,
                secret.expose().to_owned(),
                passphrase.expose().to_owned(),
            ));
        }
        let client = builder
            .authenticate()
            .await
            .map_err(|e| map_sdk_error(&e))?;

        Ok(Self {
            client,
            signer,
            dry_run,
        })
    }

    /// Builds and signs a [`BuiltOrder`] into an SDK [`SignedOrder`]. `build()`
    /// fetches live market params (tick/neg-risk/fee) over the network; `sign()`
    /// is the local EIP-712 step.
    async fn build_and_sign(&self, order: &BuiltOrder) -> Result<SignedOrder, VenueLiveError> {
        let token = U256::from_str_radix(order.token_id.as_str(), 10)
            .map_err(|e| VenueLiveError::Transport(format!("bad token id: {e}")))?;
        let side = match order.side {
            Side::Buy => SdkSide::Buy,
            Side::Sell => SdkSide::Sell,
        };

        let signable = match order.class {
            OrderClass::Limit {
                post_only,
                expiration,
            } => {
                let OrderQty::Shares(size) = order.amount else {
                    return Err(VenueLiveError::QtyKindMismatch(
                        "limit order amount must be shares".to_owned(),
                    ));
                };
                let mut builder = self
                    .client
                    .limit_order()
                    .token_id(token)
                    .side(side)
                    .price(order.price.as_decimal())
                    .size(size.as_decimal())
                    .post_only(post_only);
                builder = match expiration {
                    Some(exp) => builder
                        .order_type(OrderType::GTD)
                        .expiration(millis_to_datetime(exp)?),
                    None => builder.order_type(OrderType::GTC),
                };
                builder.build().await.map_err(|e| map_sdk_error(&e))?
            }
            OrderClass::Marketable { fok } => {
                let amount = match order.amount {
                    OrderQty::Notional(dollars) => {
                        Amount::usdc(dollars.as_decimal()).map_err(|e| map_sdk_error(&e))?
                    }
                    OrderQty::Shares(shares) => {
                        Amount::shares(shares.as_decimal()).map_err(|e| map_sdk_error(&e))?
                    }
                };
                let order_type = if fok { OrderType::FOK } else { OrderType::FAK };
                self.client
                    .market_order()
                    .token_id(token)
                    .side(side)
                    .order_type(order_type)
                    .price(order.price.as_decimal())
                    .amount(amount)
                    .build()
                    .await
                    .map_err(|e| map_sdk_error(&e))?
            }
        };

        self.client
            .sign(&self.signer, signable)
            .await
            .map_err(|e| map_sdk_error(&e))
    }

    fn dry_run_ack(client_id: Option<String>) -> RawAck {
        RawAck {
            client_id,
            success: true,
            order_id: OrderId::new("dry-run").ok(),
            status: "dry-run".to_owned(),
            error: None,
        }
    }
}

impl ClobPort for SdkClobPort {
    async fn submit(&self, order: &BuiltOrder) -> Result<RawAck, VenueLiveError> {
        let signed = self.build_and_sign(order).await?;
        if self.dry_run {
            return Ok(Self::dry_run_ack(order.client_id.clone()));
        }
        let resp = self
            .client
            .post_order(signed)
            .await
            .map_err(|e| map_sdk_error(&e))?;
        Ok(post_resp_to_ack(order.client_id.clone(), resp))
    }

    async fn submit_batch(&self, orders: &[BuiltOrder]) -> Result<Vec<RawAck>, VenueLiveError> {
        let mut signed = Vec::with_capacity(orders.len());
        for order in orders {
            signed.push(self.build_and_sign(order).await?);
        }
        if self.dry_run {
            return Ok(orders
                .iter()
                .map(|o| Self::dry_run_ack(o.client_id.clone()))
                .collect());
        }
        let resps = self
            .client
            .post_orders(signed)
            .await
            .map_err(|e| map_sdk_error(&e))?;
        if resps.len() != orders.len() {
            return Err(VenueLiveError::Transport(format!(
                "venue returned {} acks for {} orders",
                resps.len(),
                orders.len()
            )));
        }
        Ok(orders
            .iter()
            .zip(resps)
            .map(|(o, r)| post_resp_to_ack(o.client_id.clone(), r))
            .collect())
    }

    async fn cancel_one(&self, id: &OrderId) -> Result<RawCancel, VenueLiveError> {
        let resp = self
            .client
            .cancel_order(id.as_str())
            .await
            .map_err(|e| map_sdk_error(&e))?;
        Ok(cancel_resp_to_raw(resp))
    }

    async fn cancel_market(&self, market: &ConditionId) -> Result<RawCancel, VenueLiveError> {
        let cid = B256::from_str(market.as_str())
            .map_err(|e| VenueLiveError::Transport(format!("bad condition id: {e}")))?;
        let request = CancelMarketOrderRequest::builder().market(cid).build();
        let resp = self
            .client
            .cancel_market_orders(&request)
            .await
            .map_err(|e| map_sdk_error(&e))?;
        Ok(cancel_resp_to_raw(resp))
    }

    async fn cancel_all(&self) -> Result<RawCancel, VenueLiveError> {
        let resp = self
            .client
            .cancel_all_orders()
            .await
            .map_err(|e| map_sdk_error(&e))?;
        Ok(cancel_resp_to_raw(resp))
    }

    async fn balances(&self) -> Result<Wallet, VenueLiveError> {
        let request = BalanceAllowanceRequest::builder()
            .asset_type(AssetType::Collateral)
            .build();
        let resp = self
            .client
            .balance_allowance(request)
            .await
            .map_err(|e| map_sdk_error(&e))?;
        // NOTE: the SDK reports `balance` as a `Decimal`; the pUSD/USDC base-unit
        // vs human-dollar denomination must be verified live before funding
        // (Decisions Log). Treated as dollars here. Per-token positions need
        // per-token queries — left empty in the interim (the engine tracks
        // inventory from fills).
        let collateral = Dollars::new(resp.balance);
        Ok(Wallet {
            collateral_available: collateral,
            collateral_total: collateral,
            positions: vec![],
        })
    }

    async fn open_orders(&self) -> Result<Vec<RawOpenOrder>, VenueLiveError> {
        let request = OrdersRequest::builder().build();
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_OPEN_ORDER_PAGES {
            let page = self
                .client
                .orders(&request, cursor.clone())
                .await
                .map_err(|e| map_sdk_error(&e))?;
            for order in page.data {
                if let Ok(order_id) = OrderId::new(order.id) {
                    out.push(RawOpenOrder {
                        order_id,
                        status: format!("{:?}", order.status),
                        original_size: Size::new(order.original_size).unwrap_or(Size::ZERO),
                        size_matched: Size::new(order.size_matched).unwrap_or(Size::ZERO),
                    });
                }
            }
            if page.next_cursor.is_empty() || Some(&page.next_cursor) == cursor.as_ref() {
                break;
            }
            cursor = Some(page.next_cursor);
        }
        Ok(out)
    }
}

impl LiveVenue<SdkClobPort> {
    /// The ONLY network-capable, order-submitting constructor. Fails closed:
    /// runs the §11 arming gate first (so a refusal never authenticates the SDK
    /// or touches the network), validates params, authenticates, and spawns the
    /// interim reconcile poll.
    ///
    /// # Errors
    /// Any [`VenueLiveError`] from the arming gate, param validation, or auth.
    pub async fn connect(
        arming: LiveArming,
        dashboard_armed: bool,
        secrets: &Secrets,
        params: LiveParams,
    ) -> Result<Self, VenueLiveError> {
        check_arming(arming, dashboard_armed, secrets)?;
        params.validate()?;
        let backend = SdkClobPort::authenticate(secrets, &params, false).await?;
        let venue = Self::with_backend(backend, params);
        let (backend, store, events) = venue.reconcile_handles();
        tokio::spawn(reconcile_loop(backend, store, events, RECONCILE_INTERVAL));
        Ok(venue)
    }

    /// Builds and signs real orders against live market params but never posts
    /// one. No arming gates (it cannot submit), but it still authenticates, so
    /// it needs the private key and reaches the network. Used by the operator's
    /// pre-flight check and the `live-smoke` test.
    ///
    /// # Errors
    /// Any [`VenueLiveError`] from param validation or auth.
    pub async fn dry_run(secrets: &Secrets, params: LiveParams) -> Result<Self, VenueLiveError> {
        params.validate()?;
        let backend = SdkClobPort::authenticate(secrets, &params, true).await?;
        Ok(Self::with_backend(backend, params))
    }
}

/// Converts an SDK error into our typed error: HTTP `Status` errors carry the
/// status code + body, mapped via [`map_status_error`]; everything else is a
/// transport error.
fn map_sdk_error(err: &SdkError) -> VenueLiveError {
    if let Some(status) = err.downcast_ref::<Status>() {
        map_status_error(status.status_code.as_u16(), &status.message, None)
    } else {
        VenueLiveError::Transport(err.to_string())
    }
}

/// Converts a `PostOrderResponse` into a [`RawAck`], correlating our `client_id`
/// positionally (the venue does not echo it).
fn post_resp_to_ack(client_id: Option<String>, resp: PostOrderResponse) -> RawAck {
    RawAck {
        client_id,
        success: resp.success,
        order_id: if resp.order_id.is_empty() {
            None
        } else {
            OrderId::new(resp.order_id).ok()
        },
        status: format!("{:?}", resp.status),
        error: resp.error_msg,
    }
}

/// Converts an SDK `CancelOrdersResponse` into a [`RawCancel`].
fn cancel_resp_to_raw(resp: CancelOrdersResponse) -> RawCancel {
    RawCancel {
        canceled: resp
            .canceled
            .into_iter()
            .filter_map(|s| OrderId::new(s).ok())
            .collect(),
        not_canceled: resp
            .not_canceled
            .into_iter()
            .filter_map(|(id, reason)| OrderId::new(id).ok().map(|o| (o, reason)))
            .collect(),
    }
}

/// Converts unix millis to a `chrono::DateTime<Utc>` for the SDK's GTD field.
fn millis_to_datetime(ts: TimestampMs) -> Result<DateTime<Utc>, VenueLiveError> {
    DateTime::<Utc>::from_timestamp_millis(ts.as_millis()).ok_or_else(|| {
        VenueLiveError::BadConfig(format!("expiration {} ms out of range", ts.as_millis()))
    })
}
