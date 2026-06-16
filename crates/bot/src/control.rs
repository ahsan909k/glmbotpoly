//! The control plane (CLAUDE.md §10.6/§11): the single typed command processor
//! the bot binary runs.
//!
//! Both frontends — the dashboard backend and the `bot control` CLI (an HTTP
//! client to the dashboard) — funnel an operator action into a
//! [`DashboardCommand`], which the run loop hands to [`ControlPlane::decide`].
//! `decide` is pure-ish: it validates the command, mutates the owned/shared
//! runtime state, and returns a [`Decision`] — the [`ControlOutcome`] (the ack,
//! carrying the resulting [`ControlStateSnapshot`]), the bus [`Event`]s to
//! journal and project (including a [`core_types::ControlAudit`] for *every*
//! command, accepted or refused), and an optional [`VenueAction`] for the loop
//! to perform. The loop performs the side effects and replies the outcome.
//!
//! [`RuntimeControlState`] is the shared seam the future production orchestrator
//! reads: the global kill flag (the dashboard loop already reads it to gate its
//! quoter), the enabled-series set, the safe-listed parameter overrides, and the
//! live-arm session flag (§11 gate 3, what the live venue's `check_arming` reads
//! at connect time). The full engine orchestrator that consumes the param
//! overrides and series toggles is deferred, consistent with prior tasks.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

use config::{
    AppConfig, EngineConfig, LIVE_CONFIRM_PHRASE, LiveArming, Secrets, validate_param_change,
};
use core_types::{
    BreakerKind, CommandOrigin, ControlAudit, ControlEvent, Decimal, Dollars, Event, RiskEvent,
    Series, TimestampMs,
};
use dashboard::{
    ArmingView, ControlOutcome, ControlStateSnapshot, DashboardCommand, OutcomeKind,
    ParamOverrideView,
};

/// How long a begun live-arming stays open for confirmation before it expires.
const PENDING_TTL_MS: i64 = 60_000;

/// A venue side-effect the run loop performs after a [`Decision`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VenueAction {
    /// Cancel every open order (the kill path).
    CancelAll,
    /// Set paper capital to an absolute amount.
    SetCapital(Dollars),
    /// Adjust paper capital by a signed delta.
    AdjustCapital(Decimal),
}

/// The result of processing one command: the ack outcome, the bus events to
/// journal + project, and an optional venue action.
#[derive(Debug)]
pub struct Decision {
    /// The acknowledgment to send back to the frontend.
    pub outcome: ControlOutcome,
    /// Bus events to record (journal) and, for `Risk`/`Control`, project to the
    /// dashboard. Always includes the [`Event::ControlAudit`] for this command.
    pub events: Vec<Event>,
    /// A venue side-effect for the loop to perform, if any.
    pub venue_action: Option<VenueAction>,
}

/// The multi-step live-arming state machine (§11 gate 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArmingMachine {
    /// Not armed.
    Disarmed,
    /// `arm-live` succeeded; awaiting the confirmation phrase until `deadline`.
    PendingConfirm {
        /// When the pending confirmation expires.
        deadline: TimestampMs,
    },
    /// Armed for this session.
    Armed,
}

/// The shared, mutable runtime control state. The control plane writes it; the
/// run loop (and the future orchestrator) read it through cheap clones.
#[derive(Clone)]
pub struct RuntimeControlState(Arc<RwLock<Inner>>);

#[derive(Debug)]
struct Inner {
    halted: bool,
    enabled_series: BTreeSet<Series>,
    param_overrides: BTreeMap<(Option<Series>, String), String>,
    session_armed: bool,
}

impl RuntimeControlState {
    fn new(enabled_series: BTreeSet<Series>) -> Self {
        Self(Arc::new(RwLock::new(Inner {
            halted: false,
            enabled_series,
            param_overrides: BTreeMap::new(),
            session_armed: false,
        })))
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Inner> {
        self.0.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Inner> {
        self.0.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether trading is globally halted (a manual kill). The dashboard loop
    /// reads this to gate its quoter; the future orchestrator gates all flow.
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.read().halted
    }

    /// Whether `series` is currently enabled for trading.
    #[must_use]
    pub fn is_series_enabled(&self, series: Series) -> bool {
        self.read().enabled_series.contains(&series)
    }

    /// Whether live trading is armed for this session (§11 gate 3).
    ///
    /// The orchestrator seam the live venue's `check_arming` reads at connect
    /// time; that wiring is deferred (exercised by the control-plane tests).
    #[must_use]
    #[allow(
        dead_code,
        reason = "orchestrator seam (live venue gate 3); deferred, tested"
    )]
    pub fn session_armed(&self) -> bool {
        self.read().session_armed
    }

    /// The current override value for `(series, key)`, applying the series-scoped
    /// override if present, else the global one.
    ///
    /// The orchestrator seam the future engine reads to apply runtime parameter
    /// changes; that wiring is deferred (exercised by the control-plane tests).
    #[must_use]
    #[allow(
        dead_code,
        reason = "orchestrator seam (engine param overrides); deferred, tested"
    )]
    pub fn param_override(&self, series: Series, key: &str) -> Option<String> {
        let inner = self.read();
        inner
            .param_overrides
            .get(&(Some(series), key.to_owned()))
            .or_else(|| inner.param_overrides.get(&(None, key.to_owned())))
            .cloned()
    }
}

/// The typed control-command processor. Owned by the run loop; one per process.
pub struct ControlPlane {
    /// The shared runtime state (cloned into the loop for the quote gate).
    pub state: RuntimeControlState,
    arming: ArmingMachine,
    engine_cfg: EngineConfig,
    /// Gate 1: `live.enabled` in config.
    config_enabled: bool,
    /// Gate 2: the env confirmation phrase matched at boot.
    env_confirmed: bool,
}

impl ControlPlane {
    /// Builds the control plane from the loaded configuration and secrets,
    /// seeding the enabled-series set from `config.engine.enabled_series()` and
    /// capturing the live-arming boot gates.
    #[must_use]
    pub fn new(config: &AppConfig, secrets: &Secrets) -> Self {
        let arming = LiveArming::evaluate(&config.live, secrets);
        let enabled: BTreeSet<Series> = config.engine.enabled_series().into_iter().collect();
        Self {
            state: RuntimeControlState::new(enabled),
            arming: ArmingMachine::Disarmed,
            engine_cfg: config.engine.clone(),
            config_enabled: arming.config_enabled,
            env_confirmed: arming.env_confirmed,
        }
    }

    /// Processes one command, mutating runtime state and returning the
    /// [`Decision`]. Pure with respect to IO — the loop performs the venue
    /// action, journals and projects the events, and replies the outcome.
    pub fn decide(
        &mut self,
        command: DashboardCommand,
        origin: CommandOrigin,
        now: TimestampMs,
    ) -> Decision {
        let (kind, result) = self.apply(&command, now);
        let (outcome_kind, error, mut events, venue_action) = match result {
            Applied::Accepted {
                events,
                venue_action,
            } => (OutcomeKind::Accepted, None, events, venue_action),
            Applied::Rejected(msg) => (OutcomeKind::Rejected, Some(msg), Vec::new(), None),
            Applied::Conflict(msg) => (OutcomeKind::Conflict, Some(msg), Vec::new(), None),
        };

        let state = self.snapshot();
        // Every command — accepted or refused — leaves an audit record.
        let detail = audit_detail(&command, &state);
        events.push(Event::ControlAudit(Arc::new(ControlAudit {
            ts: now,
            origin,
            kind: kind.to_owned(),
            detail,
            accepted: matches!(outcome_kind, OutcomeKind::Accepted),
            error: error.clone(),
        })));

        Decision {
            outcome: ControlOutcome {
                kind: outcome_kind,
                error,
                state,
            },
            events,
            venue_action,
        }
    }

    /// The current control-plane state snapshot. `paper_capital` is left `None`;
    /// the loop fills it from the (authoritative) venue.
    #[must_use]
    pub fn snapshot(&self) -> ControlStateSnapshot {
        let inner = self.state.read();
        ControlStateSnapshot {
            halted: inner.halted,
            enabled_series: inner
                .enabled_series
                .iter()
                .map(|s| s.key().to_owned())
                .collect(),
            arming: ArmingView {
                config_enabled: self.config_enabled,
                env_confirmed: self.env_confirmed,
                session_armed: inner.session_armed,
                can_arm: self.config_enabled && self.env_confirmed,
                pending: matches!(self.arming, ArmingMachine::PendingConfirm { .. }),
            },
            param_overrides: inner
                .param_overrides
                .iter()
                .map(|((scope, key), value)| ParamOverrideView {
                    series: scope.map(|s| s.key().to_owned()),
                    key: key.clone(),
                    value: value.clone(),
                })
                .collect(),
            paper_capital: None,
        }
    }

    /// Validates and applies one command, returning its audit kind label and the
    /// applied result. Does not build the audit event (the caller does that).
    fn apply(&mut self, command: &DashboardCommand, now: TimestampMs) -> (&'static str, Applied) {
        match command {
            DashboardCommand::Kill => ("kill", self.kill()),
            DashboardCommand::Reset => ("reset", self.reset()),
            DashboardCommand::ResetDailyStop => ("reset_daily_stop", self.reset_daily_stop()),
            DashboardCommand::SetPaperCapital(amount) => (
                "set_capital",
                Applied::accepted_action(VenueAction::SetCapital(*amount)),
            ),
            DashboardCommand::AdjustPaperCapital(delta) => (
                "adjust_capital",
                Applied::accepted_action(VenueAction::AdjustCapital(*delta)),
            ),
            DashboardCommand::EnableSeries(key) => ("enable_series", self.set_series(key, true)),
            DashboardCommand::DisableSeries(key) => ("disable_series", self.set_series(key, false)),
            DashboardCommand::SetParam { series, key, value } => {
                ("set_param", self.set_param(series.as_deref(), key, value))
            }
            DashboardCommand::ArmLiveBegin => ("arm_live", self.arm_begin(now)),
            DashboardCommand::ArmLiveConfirm { phrase } => {
                ("confirm_arm", self.arm_confirm(phrase, now))
            }
            DashboardCommand::Disarm => ("disarm", self.disarm()),
        }
    }

    fn kill(&mut self) -> Applied {
        self.state.write().halted = true;
        Applied::Accepted {
            events: vec![
                Event::Control(ControlEvent::Kill),
                Event::Risk(RiskEvent::BreakerTripped {
                    breaker: BreakerKind::Manual,
                }),
                Event::Risk(RiskEvent::CancelAllIssued {
                    reason: BreakerKind::Manual,
                }),
            ],
            venue_action: Some(VenueAction::CancelAll),
        }
    }

    fn reset(&mut self) -> Applied {
        self.state.write().halted = false;
        Applied::Accepted {
            events: vec![
                Event::Control(ControlEvent::Reset),
                Event::Risk(RiskEvent::BreakerCleared {
                    breaker: BreakerKind::Manual,
                }),
                Event::Risk(RiskEvent::BreakerCleared {
                    breaker: BreakerKind::DailyStop,
                }),
            ],
            venue_action: None,
        }
    }

    fn reset_daily_stop(&mut self) -> Applied {
        Applied::Accepted {
            events: vec![
                Event::Control(ControlEvent::DailyStopReset),
                Event::Risk(RiskEvent::BreakerCleared {
                    breaker: BreakerKind::DailyStop,
                }),
            ],
            venue_action: None,
        }
    }

    fn set_series(&mut self, key: &str, enable: bool) -> Applied {
        let Ok(series) = key.parse::<Series>() else {
            return Applied::Rejected(format!("unknown series key `{key}`"));
        };
        let mut inner = self.state.write();
        let changed = if enable {
            inner.enabled_series.insert(series)
        } else {
            inner.enabled_series.remove(&series)
        };
        drop(inner);
        let event = if enable {
            ControlEvent::SeriesEnabled { series }
        } else {
            ControlEvent::SeriesDisabled { series }
        };
        // Idempotent: re-enabling an enabled series is still accepted (no-op),
        // but only journal the state-change event when it actually changed.
        let events = if changed {
            vec![Event::Control(event)]
        } else {
            Vec::new()
        };
        Applied::Accepted {
            events,
            venue_action: None,
        }
    }

    fn set_param(&mut self, series: Option<&str>, key: &str, value: &str) -> Applied {
        // Resolve the optional target series.
        let target = match series {
            Some(s) => match s.parse::<Series>() {
                Ok(series) => Some(series),
                Err(_) => return Applied::Rejected(format!("unknown series key `{s}`")),
            },
            None => None,
        };

        // Validate against every affected series (one for a targeted change, all
        // six for a global one), so a global change that breaks any series'
        // §8 rules is refused.
        let affected: Vec<Series> = match target {
            Some(series) => vec![series],
            None => Series::ALL.to_vec(),
        };
        for series in &affected {
            let base = self.effective_params(*series);
            if let Err(v) = validate_param_change(&base, key, value, series.duration) {
                let prefix = if target.is_none() {
                    format!("[{}] ", series.key())
                } else {
                    String::new()
                };
                return Applied::Rejected(format!("{prefix}{}", v.message));
            }
        }

        self.state
            .write()
            .param_overrides
            .insert((target, key.to_owned()), value.to_owned());
        // The override is held in runtime state for the future orchestrator to
        // read; no subsystem consumes it in the current loop, so no bus
        // `ControlEvent` is emitted (only the audit record).
        Applied::Accepted {
            events: Vec::new(),
            venue_action: None,
        }
    }

    /// The current effective [`config::EngineParams`] for a series: the resolved
    /// config with every applicable runtime override (global then series-scoped)
    /// applied in order.
    fn effective_params(&self, series: Series) -> config::EngineParams {
        let inner = self.state.read();
        let mut params = self.engine_cfg.resolved(series);
        for ((scope, key), value) in &inner.param_overrides {
            if (scope.is_none() || *scope == Some(series))
                && let Ok(change) = validate_param_change(&params, key, value, series.duration)
            {
                params = change.patch.apply(&params);
            }
        }
        params
    }

    fn arm_begin(&mut self, now: TimestampMs) -> Applied {
        if !self.config_enabled {
            return Applied::Conflict(
                "cannot arm: live trading is not enabled in config (gate 1)".to_owned(),
            );
        }
        if !self.env_confirmed {
            return Applied::Conflict(
                "cannot arm: the BOT_SECRET_LIVE_CONFIRM environment phrase is absent or wrong \
                 (gate 2)"
                    .to_owned(),
            );
        }
        self.arming = ArmingMachine::PendingConfirm {
            deadline: TimestampMs::from_millis(now.as_millis() + PENDING_TTL_MS),
        };
        Applied::Accepted {
            events: Vec::new(),
            venue_action: None,
        }
    }

    fn arm_confirm(&mut self, phrase: &str, now: TimestampMs) -> Applied {
        let deadline = match self.arming {
            ArmingMachine::PendingConfirm { deadline } => deadline,
            ArmingMachine::Armed => {
                return Applied::Conflict("already armed".to_owned());
            }
            ArmingMachine::Disarmed => {
                return Applied::Conflict("no arming pending — call arm-live first".to_owned());
            }
        };
        if now.as_millis() > deadline.as_millis() {
            self.arming = ArmingMachine::Disarmed;
            return Applied::Conflict(
                "arming confirmation expired — call arm-live again".to_owned(),
            );
        }
        if phrase != LIVE_CONFIRM_PHRASE {
            return Applied::Rejected("confirmation phrase does not match".to_owned());
        }
        // Defensive re-check: the boot gates can never regress at runtime, but a
        // confirm must never arm if a gate is somehow missing.
        if !(self.config_enabled && self.env_confirmed) {
            self.arming = ArmingMachine::Disarmed;
            return Applied::Conflict("cannot arm: a boot gate is missing".to_owned());
        }
        self.arming = ArmingMachine::Armed;
        self.state.write().session_armed = true;
        Applied::Accepted {
            events: Vec::new(),
            venue_action: None,
        }
    }

    fn disarm(&mut self) -> Applied {
        self.arming = ArmingMachine::Disarmed;
        self.state.write().session_armed = false;
        Applied::Accepted {
            events: Vec::new(),
            venue_action: None,
        }
    }
}

/// The internal result of applying a command (before the audit event is built).
enum Applied {
    Accepted {
        events: Vec<Event>,
        venue_action: Option<VenueAction>,
    },
    Rejected(String),
    Conflict(String),
}

impl Applied {
    fn accepted_action(action: VenueAction) -> Self {
        Applied::Accepted {
            events: Vec::new(),
            venue_action: Some(action),
        }
    }
}

/// A one-line human-readable summary of a command + its result, for the audit
/// detail column.
fn audit_detail(command: &DashboardCommand, state: &ControlStateSnapshot) -> String {
    match command {
        DashboardCommand::Kill => format!("halted={}", state.halted),
        DashboardCommand::Reset => format!("halted={}", state.halted),
        DashboardCommand::ResetDailyStop => "cleared daily stop".to_owned(),
        DashboardCommand::SetPaperCapital(amount) => format!("set capital to {amount}"),
        DashboardCommand::AdjustPaperCapital(delta) => format!("adjust capital by {delta}"),
        DashboardCommand::EnableSeries(key) => format!("enable series {key}"),
        DashboardCommand::DisableSeries(key) => format!("disable series {key}"),
        DashboardCommand::SetParam { series, key, value } => {
            let scope = series.as_deref().unwrap_or("all");
            format!("set {key}={value} for {scope}")
        }
        DashboardCommand::ArmLiveBegin => format!("arm begin (pending={})", state.arming.pending),
        DashboardCommand::ArmLiveConfirm { .. } => {
            format!("arm confirm (armed={})", state.arming.session_armed)
        }
        DashboardCommand::Disarm => "disarm".to_owned(),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use config::{LiveConfig, SecretString};

    use super::*;

    fn ts(ms: i64) -> TimestampMs {
        TimestampMs::from_millis(ms)
    }

    /// A control plane with the live boot gates open (config enabled + env phrase).
    fn armed_gates_plane() -> ControlPlane {
        let mut config = AppConfig::default();
        config.live.enabled = true;
        let secrets = Secrets {
            live_confirm: Some(SecretString::new(LIVE_CONFIRM_PHRASE.to_owned())),
            ..Secrets::default()
        };
        ControlPlane::new(&config, &secrets)
    }

    /// A control plane with no live gates (the default, paper-only deployment).
    fn paper_plane() -> ControlPlane {
        ControlPlane::new(&AppConfig::default(), &Secrets::default())
    }

    fn decide(p: &mut ControlPlane, cmd: DashboardCommand) -> Decision {
        p.decide(cmd, CommandOrigin::Cli, ts(1_000))
    }

    fn audit(events: &[Event]) -> &ControlAudit {
        events
            .iter()
            .find_map(|e| match e {
                Event::ControlAudit(a) => Some(a.as_ref()),
                _ => None,
            })
            .expect("an audit event is emitted for every command")
    }

    #[test]
    fn kill_halts_and_resets_resume_and_both_audit() {
        let mut p = paper_plane();
        let d = decide(&mut p, DashboardCommand::Kill);
        assert!(d.outcome.accepted());
        assert!(d.outcome.state.halted);
        assert_eq!(d.venue_action, Some(VenueAction::CancelAll));
        assert!(p.state.is_halted());
        assert!(audit(&d.events).accepted);
        assert_eq!(audit(&d.events).kind, "kill");

        let d = decide(&mut p, DashboardCommand::Reset);
        assert!(d.outcome.accepted());
        assert!(!d.outcome.state.halted);
        assert!(!p.state.is_halted());
        assert_eq!(audit(&d.events).kind, "reset");
    }

    #[test]
    fn reset_daily_stop_emits_targeted_control_event() {
        let mut p = paper_plane();
        let d = decide(&mut p, DashboardCommand::ResetDailyStop);
        assert!(d.outcome.accepted());
        assert!(
            d.events
                .iter()
                .any(|e| matches!(e, Event::Control(ControlEvent::DailyStopReset)))
        );
        assert_eq!(audit(&d.events).kind, "reset_daily_stop");
    }

    #[test]
    fn enable_disable_series_updates_state_and_rejects_unknown() {
        let mut p = paper_plane();
        let btc5 = "BTC-5m".parse::<Series>().unwrap();
        // Disabled, then re-enabled.
        let d = decide(&mut p, DashboardCommand::DisableSeries("BTC-5m".to_owned()));
        assert!(d.outcome.accepted());
        assert!(!p.state.is_series_enabled(btc5));
        let d = decide(&mut p, DashboardCommand::EnableSeries("BTC-5m".to_owned()));
        assert!(d.outcome.accepted());
        assert!(p.state.is_series_enabled(btc5));

        let d = decide(&mut p, DashboardCommand::DisableSeries("XRP-9m".to_owned()));
        assert_eq!(d.outcome.kind, OutcomeKind::Rejected);
        assert!(!audit(&d.events).accepted);
    }

    #[test]
    fn set_param_applies_in_range_and_records_override() {
        let mut p = paper_plane();
        let d = decide(
            &mut p,
            DashboardCommand::SetParam {
                series: Some("BTC-5m".to_owned()),
                key: "min_edge".to_owned(),
                value: "0.02".to_owned(),
            },
        );
        assert!(d.outcome.accepted());
        let btc5 = "BTC-5m".parse::<Series>().unwrap();
        assert_eq!(
            p.state.param_override(btc5, "min_edge").as_deref(),
            Some("0.02")
        );
        assert!(
            d.outcome
                .state
                .param_overrides
                .iter()
                .any(|o| o.key == "min_edge" && o.value == "0.02")
        );
    }

    #[test]
    fn set_param_out_of_range_rejected_and_state_unchanged() {
        let mut p = paper_plane();
        let d = decide(
            &mut p,
            DashboardCommand::SetParam {
                series: None,
                key: "min_edge".to_owned(),
                value: "0.005".to_owned(),
            },
        );
        assert_eq!(d.outcome.kind, OutcomeKind::Rejected);
        assert!(d.outcome.state.param_overrides.is_empty());
        assert!(!audit(&d.events).accepted);
    }

    #[test]
    fn set_param_structural_key_rejected() {
        let mut p = paper_plane();
        let d = decide(
            &mut p,
            DashboardCommand::SetParam {
                series: None,
                key: "touch_size".to_owned(),
                value: "20".to_owned(),
            },
        );
        assert_eq!(d.outcome.kind, OutcomeKind::Rejected);
        assert!(d.outcome.error.unwrap().contains("requires a restart"));
    }

    #[test]
    fn arm_refused_when_config_gate_missing() {
        // config.live.enabled = false (paper plane), env phrase present.
        let mut config = AppConfig::default();
        config.live.enabled = false;
        let secrets = Secrets {
            live_confirm: Some(SecretString::new(LIVE_CONFIRM_PHRASE.to_owned())),
            ..Secrets::default()
        };
        let mut p = ControlPlane::new(&config, &secrets);
        let d = decide(&mut p, DashboardCommand::ArmLiveBegin);
        assert_eq!(d.outcome.kind, OutcomeKind::Conflict);
        assert!(!d.outcome.state.arming.can_arm);
        assert!(!d.outcome.state.arming.pending);
    }

    #[test]
    fn arm_refused_when_env_gate_missing() {
        // config enabled, but the env phrase absent.
        let config = AppConfig {
            live: LiveConfig {
                enabled: true,
                ..LiveConfig::default()
            },
            ..AppConfig::default()
        };
        let mut p = ControlPlane::new(&config, &Secrets::default());
        let d = decide(&mut p, DashboardCommand::ArmLiveBegin);
        assert_eq!(d.outcome.kind, OutcomeKind::Conflict);
        assert!(!d.outcome.state.arming.can_arm);
        assert!(!p.state.session_armed());
    }

    #[test]
    fn full_arm_then_disarm_round_trip() {
        let mut p = armed_gates_plane();
        // Begin → pending.
        let d = p.decide(
            DashboardCommand::ArmLiveBegin,
            CommandOrigin::Dashboard,
            ts(1_000),
        );
        assert!(d.outcome.accepted());
        assert!(d.outcome.state.arming.pending);
        assert!(!d.outcome.state.arming.session_armed);
        // Confirm with the phrase → armed.
        let d = p.decide(
            DashboardCommand::ArmLiveConfirm {
                phrase: LIVE_CONFIRM_PHRASE.to_owned(),
            },
            CommandOrigin::Dashboard,
            ts(2_000),
        );
        assert!(d.outcome.accepted());
        assert!(d.outcome.state.arming.session_armed);
        assert!(p.state.session_armed());
        // Disarm.
        let d = p.decide(
            DashboardCommand::Disarm,
            CommandOrigin::Dashboard,
            ts(3_000),
        );
        assert!(d.outcome.accepted());
        assert!(!p.state.session_armed());
    }

    #[test]
    fn confirm_refused_wrong_phrase() {
        let mut p = armed_gates_plane();
        p.decide(
            DashboardCommand::ArmLiveBegin,
            CommandOrigin::Cli,
            ts(1_000),
        );
        let d = p.decide(
            DashboardCommand::ArmLiveConfirm {
                phrase: "wrong".to_owned(),
            },
            CommandOrigin::Cli,
            ts(2_000),
        );
        assert_eq!(d.outcome.kind, OutcomeKind::Rejected);
        assert!(!p.state.session_armed());
    }

    #[test]
    fn confirm_refused_after_nonce_expiry() {
        let mut p = armed_gates_plane();
        p.decide(
            DashboardCommand::ArmLiveBegin,
            CommandOrigin::Cli,
            ts(1_000),
        );
        // Past the TTL.
        let d = p.decide(
            DashboardCommand::ArmLiveConfirm {
                phrase: LIVE_CONFIRM_PHRASE.to_owned(),
            },
            CommandOrigin::Cli,
            ts(1_000 + PENDING_TTL_MS + 1),
        );
        assert_eq!(d.outcome.kind, OutcomeKind::Conflict);
        assert!(!p.state.session_armed());
        assert!(!d.outcome.state.arming.pending);
    }

    #[test]
    fn confirm_without_begin_is_conflict() {
        let mut p = armed_gates_plane();
        let d = p.decide(
            DashboardCommand::ArmLiveConfirm {
                phrase: LIVE_CONFIRM_PHRASE.to_owned(),
            },
            CommandOrigin::Cli,
            ts(1_000),
        );
        assert_eq!(d.outcome.kind, OutcomeKind::Conflict);
    }

    #[test]
    fn every_command_emits_exactly_one_audit() {
        let mut p = paper_plane();
        for cmd in [
            DashboardCommand::Kill,
            DashboardCommand::Reset,
            DashboardCommand::ResetDailyStop,
            DashboardCommand::EnableSeries("BTC-5m".to_owned()),
            DashboardCommand::SetParam {
                series: None,
                key: "min_edge".to_owned(),
                value: "0.005".to_owned(), // rejected, still audited
            },
        ] {
            let d = decide(&mut p, cmd);
            let n = d
                .events
                .iter()
                .filter(|e| matches!(e, Event::ControlAudit(_)))
                .count();
            assert_eq!(n, 1, "exactly one audit per command");
        }
    }

    /// End-to-end: drive a sequence of commands through a real [`ControlPlane`]
    /// and a real [`journal::Recorder`], then query the sqlite audit trail — one
    /// row per command, accepted or refused, with the right origin and kind
    /// (Required test 3). [`ControlPlane`] lives in the `bot` binary, so this is
    /// an in-crate test rather than a `tests/` integration file (a bin crate has
    /// no lib target a `tests/` file could reach).
    #[test]
    fn audit_trail_records_every_command_end_to_end() {
        use journal::{JournalIndexReader, Recorder, RecorderParams};

        let dir = std::env::temp_dir().join(format!("bot-ctl-audit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let sqlite = dir.join("index.sqlite");
        let recorder = Recorder::spawn(
            RecorderParams {
                out_dir: dir.clone(),
                sqlite_path: Some(sqlite.clone()),
                ..RecorderParams::default()
            },
            || ts(1_000_000),
        )
        .expect("recorder");

        let mut p = armed_gates_plane();
        let commands = [
            (DashboardCommand::Kill, CommandOrigin::Cli, "kill", true),
            (
                DashboardCommand::Reset,
                CommandOrigin::Dashboard,
                "reset",
                true,
            ),
            (
                // Structural param → refused, but still audited.
                DashboardCommand::SetParam {
                    series: None,
                    key: "touch_size".to_owned(),
                    value: "20".to_owned(),
                },
                CommandOrigin::Cli,
                "set_param",
                false,
            ),
            (
                DashboardCommand::ArmLiveBegin,
                CommandOrigin::Cli,
                "arm_live",
                true,
            ),
        ];
        for (i, (cmd, origin, _, _)) in commands.iter().enumerate() {
            let d = p.decide(cmd.clone(), *origin, ts(1_000 + i as i64));
            for ev in &d.events {
                recorder.record(ev);
            }
        }
        recorder.finish().expect("finish");

        let reader = JournalIndexReader::open(&sqlite).expect("open index");
        let rows = reader.command_audit().expect("query audit");
        assert_eq!(rows.len(), commands.len(), "one audit row per command");
        for (row, (_, origin, kind, accepted)) in rows.iter().zip(commands.iter()) {
            assert_eq!(&row.kind, kind);
            assert_eq!(row.origin, *origin);
            assert_eq!(row.accepted, *accepted);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
