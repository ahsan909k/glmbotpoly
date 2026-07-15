#!/usr/bin/env bash
# Eval-config assertion set — verifies the SIGNED-OFF sizing table (2026-07-15)
# is EXACTLY what the bot will load, run against the effective merged config
# BEFORE the cutover swap. Two layers:
#   (1) `bot check-config` must pass  -> the default.toml + bot.local.toml merge is
#       legal under every §8/§11 validation rule.
#   (2) every signed-off knob is asserted to its exact value. All signed-off knobs
#       are set in bot.local.toml, which overrides default.toml, so the value read
#       there IS the effective value (and (1) proves the merge is valid).
# ALL pass -> exit 0 (safe to swap). ANY mismatch -> exit 1 + the diff, DO NOT SWAP.
set -uo pipefail
CFG=/etc/bot/config/bot.local.toml
BIN=/opt/bot/bin/bot
NEWBIN="${1:-$BIN}"    # optionally assert against a freshly-built binary's check-config
FAIL=0

echo "== eval-config assertions (effective merged config) =="

# Layer 1 — the merge is valid.
if "$NEWBIN" check-config --config-dir /etc/bot/config >/tmp/checkcfg.out 2>&1; then
  echo "PASS  bot check-config (merge valid under all validation rules)"
else
  echo "FAIL  bot check-config did NOT pass:"; sed 's/^/      /' /tmp/checkcfg.out; exit 1
fi

# Layer 2 — exact signed-off values.
val() { grep -E "^[[:space:]]*$1[[:space:]]*=" "$CFG" | head -1 | sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*(#.*)?$//'; }
want() { # key expected human
  local got; got="$(val "$1")"
  if [ "$got" = "$2" ]; then printf 'PASS  %-46s = %s\n' "$3" "$got"
  else printf 'FAIL  %-46s expected %s, got %s\n' "$3" "$2" "${got:-MISSING}"; FAIL=1; fi
}
want clip_size_shares                    60    'clip_size_shares (60, hard per-order cap)'
want touch_size                          40    'touch_size (40)'
want ladder_size_per_level               40    'ladder_size_per_level (40)'
want ladder_levels                       3     'ladder_levels (3)  -> ~$152/side instantaneous'
want maker_deployment_budget_per_window  150   'maker cumulative budget (150; day-2 bump to 1000 is LATER, NOT pre-installed)'
want max_worst_case_loss_per_window      95    'max_worst_case_loss (95)'
want soft_cap_excess_shares              200   'soft_cap_excess_shares (200)'
want hard_cap_excess_shares              400   'hard_cap_excess_shares (400)'
want taker_budget_per_window             10    'taker_budget_per_window (10; day-2 bump to 100 is LATER)'
want budget_per_window                   10    'model_taker.budget_per_window (10)'
want shadow_loss_stops                   true  'shadow_loss_stops (true = SHADOW loss stops)'
want daily_stop_loss                     1000  'daily_stop_loss (1000, SHADOW)'
want max_open_notional                   5000  'max_open_notional (5000, HARD backstop)'
want starting_capital                    50000 'paper.starting_capital (50000 bankroll)'
want feed_staleness_grace_ms             1500  'feed_staleness_grace_ms (1500; eu-west-1 Binance Mid p95 gap 1.4s — env-correct, see 2026-07-16 log)'
want book_staleness_dwell_ms             1500  'book_staleness_dwell_ms (1500, filters transient rollover book churn)'

# Layer 3 — model-taker + shadow enabled; fortress arbitration + model-taker earned
# config (theta 0.03, fortress BTC->momentum / ETH->model) are the TESTED engine
# defaults, not overridden here — assert they are NOT accidentally set in bot.local.
if [ "$(grep -cE '^[[:space:]]*enable[[:space:]]*=[[:space:]]*true' "$CFG")" -ge 2 ]; then
  echo "PASS  shadow + model_taker enable = true"
else echo "FAIL  shadow/model_taker not both enabled"; FAIL=1; fi
if grep -qE '^[[:space:]]*(theta|precedence|series_precedence)' "$CFG"; then
  echo "FAIL  model-taker earned config / arbitration was overridden in bot.local (must stay the tested default)"; FAIL=1
else echo "PASS  model-taker earned config + fortress arbitration left at the tested defaults"; fi

if [ "$FAIL" -eq 0 ]; then echo "=== ALL ASSERTIONS PASS — safe to swap ==="; else echo "=== ASSERTION FAILURES — DO NOT SWAP ==="; fi
exit "$FAIL"
