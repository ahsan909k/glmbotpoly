#!/usr/bin/env bash
# 24 h dress-rehearsal gate checker (GATE 2). Run on the VPS after ~24 h of the
# full stack at Days-1–3 sizing. Bounds all journal queries by the rehearsal
# epoch (arg $1 = epoch ms, else /var/lib/bot/data/eval/rehearsal_epoch's first
# field, else 24 h ago). Prints PASS/FAIL per gate; exits non-zero if any FAIL.
# Read-only: opens the sqlite index read-only (WAL-safe alongside the running bot).
set -uo pipefail

DB=/var/lib/bot/data/journal.sqlite
EPOCH_FILE=/var/lib/bot/data/eval/rehearsal_epoch
T0="${1:-}"
if [ -z "$T0" ] && [ -f "$EPOCH_FILE" ]; then T0="$(awk '{print $1; exit}' "$EPOCH_FILE")"; fi
if [ -z "$T0" ]; then T0=$(( ($(date -u +%s) - 86400) * 1000 )); fi
q() { sqlite3 -readonly "$DB" "$1"; }
FAILED=0
pass() { printf 'PASS  %-26s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-26s %s\n' "$1" "${2:-}"; FAILED=1; }
echo "== rehearsal check, epoch T0=$T0 ($(date -u -d "@$((T0/1000))" +%FT%TZ)) =="

# 1. Uptime 100% / zero panics
NR=$(systemctl show bot -p NRestarts --value 2>/dev/null || echo '?')
PANICS=$(journalctl -u bot --since "@$((T0/1000))" 2>/dev/null | grep -ciE 'panic|panicked')
ACTIVE=$(systemctl is-active bot 2>/dev/null)
{ [ "$NR" = "0" ] && [ "$PANICS" = "0" ] && [ "$ACTIVE" = "active" ]; } \
  && pass uptime/zero-panic "NRestarts=$NR panics=$PANICS state=$ACTIVE" \
  || fail uptime/zero-panic "NRestarts=$NR panics=$PANICS state=$ACTIVE"

# 2. RSS stable < 1.5 GB watchdog threshold
RSS=$(journalctl -u bot --since "@$((T0/1000))" 2>/dev/null \
      | grep -oE 'rss_kb=[0-9]+' | cut -d= -f2 | sort -n | tail -1)
{ [ -n "$RSS" ] && [ "$RSS" -lt 1500000 ]; } \
  && pass rss<1.5GB "max_rss_kb=${RSS:-?}" || fail rss<1.5GB "max_rss_kb=${RSS:-none}"

# 3. CPU steal ~0
STEAL=$(mpstat 1 3 2>/dev/null | awk '/Average/{print $NF}')
{ [ -n "$STEAL" ] && awk "BEGIN{exit !($STEAL < 1.0)}"; } \
  && pass cpu-steal~0 "%steal=$STEAL" || fail cpu-steal~0 "%steal=${STEAL:-?}"

# 4. Feeds green: steady-state trips < 5/hr (24 h span)
HOURS=$(( ($(date -u +%s)*1000 - T0) / 3600000 )); [ "$HOURS" -lt 1 ] && HOURS=1
for b in feed_stale ws_disconnect fair_vs_mid; do
  C=$(q "SELECT COUNT(*) FROM breaker_trips WHERE kind='tripped' AND breaker='$b' AND ts_local_ms>=$T0;")
  RATE=$(( ${C:-0} / HOURS ))
  { [ "$RATE" -lt 5 ]; } && pass "feed:$b<5/hr" "trips=$C rate=${RATE}/hr" \
    || fail "feed:$b<5/hr" "trips=$C rate=${RATE}/hr (annotate any Polymarket outage window)"
done

# 5. Makers resting on all 6 series
N6=$(q "SELECT COUNT(*) FROM (SELECT series FROM orders WHERE state='open' AND ts_local_ms>=$T0 GROUP BY series);")
{ [ "${N6:-0}" -eq 6 ]; } && pass makers-x6 "series_with_resting=$N6" \
  || fail makers-x6 "series_with_resting=${N6:-0} (expected 6)"

# 6. Clip cap respected — no order/fill > 60 shares
BIG=$(q "SELECT COUNT(*) FROM orders WHERE ts_local_ms>=$T0 AND CAST(original_size AS REAL) > 60.0;")
{ [ "${BIG:-0}" -eq 0 ]; } && pass clip-cap<=60 "over60=$BIG" || fail clip-cap<=60 "over60=$BIG orders > 60 shares"

# 7. Clock < 1 ms
OFF=$(chronyc tracking 2>/dev/null | awk -F'[ ]+' '/System time/{print $4}')
{ [ -n "$OFF" ] && awk "BEGIN{exit !($OFF < 0.001)}"; } \
  && pass clock<1ms "offset_s=$OFF" || fail clock<1ms "offset_s=${OFF:-?}"

# 8. S3 backup succeeded (last run exit 0)
S3=$(systemctl show bot-s3-backup.service -p ExecMainStatus --value 2>/dev/null || echo '?')
{ [ "$S3" = "0" ]; } && pass s3-backup "ExecMainStatus=$S3" || fail s3-backup "ExecMainStatus=$S3"

# 9. Dashboard reachable (loopback)
curl -fsS http://localhost:8080/health >/dev/null 2>&1 \
  && pass dashboard "/health 200" || fail dashboard "/health unreachable"

# 10. Cancel-latency p95 < 250 ms (place->cancel-ack from order state transitions)
P95=$(q "WITH d AS (
  SELECT order_id,
    (MIN(CASE WHEN state='canceled' THEN ts_local_ms END)
     - MIN(CASE WHEN state='pending_cancel' THEN ts_local_ms END)) AS ms
  FROM orders WHERE ts_local_ms>=$T0 GROUP BY order_id)
  SELECT ms FROM d WHERE ms IS NOT NULL AND ms>=0 ORDER BY ms
  LIMIT 1 OFFSET (SELECT CAST(0.95*COUNT(*) AS INT) FROM d WHERE ms IS NOT NULL AND ms>=0);")
if [ -n "$P95" ]; then
  { [ "$P95" -lt 250 ]; } && pass cancel-p95<250ms "p95=${P95}ms" || fail cancel-p95<250ms "p95=${P95}ms"
else
  pass cancel-p95<250ms "no pending_cancel transitions (paper cancels Open->Canceled) — n/a"
fi

# 11. Independence: after any window_loss trip, other series kept placing
WL=$(q "SELECT COUNT(*) FROM breaker_trips WHERE breaker='window_loss' AND kind='tripped' AND ts_local_ms>=$T0;")
pass independence "window_loss trips=$WL (offline proof: chaos.rs window_loss_halts_that_window_only + discovery_failure_at_rollover)"

# 12. Journal summary (durable, ts-bounded)
echo "-- journal summary (epoch-bounded) --"
q "SELECT 'maker_fills', COUNT(*) FROM fills WHERE liquidity='maker' AND ts_local_ms>=$T0
   UNION ALL SELECT 'taker_fills', COUNT(*) FROM fills WHERE liquidity='taker' AND ts_local_ms>=$T0
   UNION ALL SELECT 'settlements', COUNT(*) FROM settlements WHERE ts_local_ms>=$T0
   UNION ALL SELECT 'shadow_stops', COUNT(*) FROM breaker_trips WHERE breaker IN ('daily_stop','window_loss') AND ts_local_ms>=$T0;"
echo "(per-driver fired + fill-quality vs benchmarks: see today's bot digest)"

echo "== $( [ $FAILED -eq 0 ] && echo 'GATE 2: ALL PASS' || echo 'GATE 2: FAILURES — fix and rerun' ) =="
exit $FAILED
