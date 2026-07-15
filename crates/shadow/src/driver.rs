//! The async shadow driver: fans the bus event clone + shadow's own depth feed
//! into [`ShadowCore`], runs a window-aligned 5 s inference over every active
//! window, journals each prediction, and pushes a lightweight update to the
//! dashboard. Holds no `bus_tx` and no venue port — structurally incapable of
//! reaching a venue or stalling the engine.

use std::sync::Arc;
use std::time::Duration;

use core_types::{Event, TimestampMs};
use feed_util::Transport;
use tokio::sync::{mpsc, watch};
use tokio::time::{MissedTickBehavior, interval};

use crate::features::depth::DepthLadder;
use crate::lgbm::LgbmModel;
use crate::params::{ModelIdentity, ShadowParams};
use crate::record::{ShadowPrediction, ShadowRecorder};
use crate::state::ShadowCore;
use crate::{ShadowStats, ShadowUpdate, depth_feed};

/// How often stale windows are pruned.
const PRUNE_INTERVAL: Duration = Duration::from_secs(30);

/// Runs the shadow observer until `shutdown_rx` flips or the bus closes.
///
/// `depth_transport` is the (injectable) WebSocket transport for shadow's own
/// Binance depth20 feed (`WsTransport` in production, a fake in tests).
#[allow(
    clippy::too_many_arguments,
    reason = "the orchestrator wiring entry point"
)]
pub async fn run<T, F>(
    params: ShadowParams,
    model: Arc<LgbmModel>,
    identity: ModelIdentity,
    mut bus_rx: mpsc::Receiver<Event>,
    update_tx: mpsc::Sender<ShadowUpdate>,
    depth_transport: T,
    now_fn: F,
    mut shutdown_rx: watch::Receiver<bool>,
) -> ShadowStats
where
    T: Transport + Send + 'static,
    F: Fn() -> TimestampMs + Clone + Send + Sync + 'static,
{
    let mut core = match ShadowCore::new(&params.assets, params.max_staleness_ms) {
        Ok(core) => core,
        Err(error) => {
            tracing::error!(target: "shadow", %error, "shadow core failed to construct");
            return ShadowStats::default();
        }
    };
    let recorder = match ShadowRecorder::spawn(params.shadow_dir.clone(), params.record_channel_cap)
    {
        Ok(recorder) => recorder,
        Err(error) => {
            tracing::error!(target: "shadow", %error, "shadow recorder failed to start");
            return ShadowStats::default();
        }
    };

    // Shadow's own depth20 feed → bounded ladder channel.
    let (depth_tx, mut depth_rx) = mpsc::channel::<DepthLadder>(params.depth_channel_cap.max(1));
    let depth_task = tokio::spawn(depth_feed::run(
        params.depth.clone(),
        depth_transport,
        now_fn.clone(),
        depth_tx,
        shutdown_rx.clone(),
        None,
    ));

    let grid_ms = i64::try_from(params.sample_secs.max(1)).unwrap_or(5) * 1_000;
    let mut sample_timer = interval(Duration::from_secs(params.sample_secs.max(1)));
    sample_timer.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut prune_timer = interval(PRUNE_INTERVAL);
    prune_timer.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut stats = ShadowStats::default();
    let mut last_sample_ms = i64::MIN;

    tracing::info!(
        target: "shadow",
        assets = params.assets.len(), sample_secs = params.sample_secs,
        trees = model.num_trees(), model_sha = %identity.short_sha(),
        "shadow observer running (observation only — influences nothing)"
    );

    loop {
        tokio::select! {
            event = bus_rx.recv() => match event {
                Some(event) => core.on_event(&event),
                None => break, // the drop-on-full bus clone closed → shutting down
            },
            ladder = depth_rx.recv() => {
                if let Some(ladder) = ladder {
                    stats.depth_frames = stats.depth_frames.saturating_add(1);
                    core.on_depth(&ladder);
                }
            }
            _ = sample_timer.tick() => {
                let now = now_fn();
                // Window-aligned instant (open times are 5 s-grid multiples, so
                // flooring to the global grid matches the lab's open + k·5 s).
                let sample_ms = now.as_millis().div_euclid(grid_ms) * grid_ms;
                if sample_ms <= last_sample_ms {
                    continue;
                }
                last_sample_ms = sample_ms;
                let sample_ts = TimestampMs::from_millis(sample_ms);
                for win in core.active_windows() {
                    let Some(out) = core.sample(win, sample_ts) else {
                        continue;
                    };
                    let p_up = model.predict(&out.features.values);
                    let finite = out.features.finite_count();
                    recorder.record(ShadowPrediction {
                        ts: sample_ms,
                        series: win.series.to_string(),
                        window_open_ms: win.open_time.as_millis(),
                        p_up,
                        features: out.features.values.to_vec(),
                        features_hash: out.features.hash_hex(),
                        model_sha256: identity.sha256.clone(),
                        model_trained_through_ms: identity.trained_through_ms,
                        model_seed: identity.seed,
                        strike: out.strike,
                        price_asof_sec: out.price_asof_sec,
                        depth_asof_ms: out.depth_asof_ms,
                        pm_asof_ms: out.pm_asof_ms,
                        sigma_returns: out.sigma_returns,
                        basis_samples: out.basis_samples,
                        pm_run_secs: out.pm_run_secs,
                        finite_count: finite,
                    });
                    let _ = update_tx.try_send(ShadowUpdate {
                        series: win.series,
                        window_open_ms: win.open_time.as_millis(),
                        ts: sample_ts,
                        p_up,
                        finite_count: finite,
                        model_trained_through_ms: identity.trained_through_ms,
                        model_short_sha: identity.short_sha(),
                        staleness_alert_days: params.staleness_alert_days,
                    });
                    stats.predictions = stats.predictions.saturating_add(1);
                }
            }
            _ = prune_timer.tick() => core.prune(now_fn()),
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    break;
                }
            }
        }
    }

    // The depth task shares `shutdown_rx`; wait for it, then finalize the writer.
    let _ = depth_task.await;
    let rec_stats = recorder.finish();
    stats.predictions_dropped = rec_stats.dropped;
    tracing::info!(
        target: "shadow",
        predictions = stats.predictions, dropped = rec_stats.dropped,
        written = rec_stats.written, depth_frames = stats.depth_frames,
        "shadow run ended"
    );
    stats
}
