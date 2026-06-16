#!/usr/bin/env bash
# Drain-then-swap upgrade for the bot. Stops the service (SIGTERM -> graceful
# drain: armed=false -> cancel_all -> drain <=5s -> join -> journal flush),
# verifies a clean shutdown, atomically repoints the binary symlink, restarts,
# and verifies health + ARMED. Auto-rolls-back to the previous binary on failure.
#
# Idempotent: re-running with the same NEW_BINARY just re-installs + re-links it.
#
# Usage:  sudo deploy/upgrade.sh /path/to/new/bot-<version>-<sha>
#
# HONEST NOTE: paper state does NOT survive a restart (no run-loop consumes the
# journal rebuild yet) — a restart begins a fresh paper wallet + new journal
# session. The drain's value is ZERO orphaned open orders across the swap, not
# state continuity. Use this (SIGTERM), NEVER `bot control kill` (that only
# halts order flow; it does not exit the process), to upgrade.
set -euo pipefail

BIN_DIR=/opt/bot/bin
LINK="${BIN_DIR}/bot"
HEALTH_URL="${HEALTH_URL:-http://localhost:8080/health}"

NEW_BINARY="${1:?usage: upgrade.sh /path/to/new/bot-<version>-<sha>}"
[ -f "${NEW_BINARY}" ] || { echo "!! not a file: ${NEW_BINARY}" >&2; exit 1; }
VERSION_NAME="$(basename "${NEW_BINARY}")"
TARGET="${BIN_DIR}/${VERSION_NAME}"

echo "==> Installing new binary as ${TARGET}"
sudo install -m 0755 "${NEW_BINARY}" "${TARGET}"

PREVIOUS="$(readlink -f "${LINK}" 2>/dev/null || true)"
echo "==> Current symlink target: ${PREVIOUS:-<none>}"

echo "==> Stopping bot (graceful drain, up to TimeoutStopSec)"
sudo systemctl stop bot

echo "==> Verifying clean shutdown in the journal"
recent="$(sudo journalctl -u bot -n 200 --no-pager || true)"
echo "${recent}" | grep -q "zero open orders" \
  || echo "!! WARNING: did not see 'zero open orders' in recent logs" >&2
echo "${recent}" | grep -q "journal flushed" \
  || echo "!! WARNING: did not see 'journal flushed' in recent logs" >&2

echo "==> Atomically repointing ${LINK} -> ${TARGET}"
sudo ln -sfn "${TARGET}" "${LINK}"

echo "==> Starting bot"
sudo systemctl start bot

echo "==> Waiting for health + ARMED (up to ~60s)"
ok=0
for _ in $(seq 1 30); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1 \
     && sudo journalctl -u bot --since "2 min ago" --no-pager | grep -q "ARMED"; then
    ok=1; break
  fi
  sleep 2
done

if [ "${ok}" -eq 1 ]; then
  echo "==> Upgrade OK: ${VERSION_NAME} is live and ARMED."
  exit 0
fi

echo "!! Upgrade verification FAILED." >&2
if [ -n "${PREVIOUS}" ] && [ "${PREVIOUS}" != "${TARGET}" ]; then
  echo "!! Rolling back to ${PREVIOUS}" >&2
  sudo systemctl stop bot
  sudo ln -sfn "${PREVIOUS}" "${LINK}"
  sudo systemctl start bot
  echo "!! Rolled back to ${PREVIOUS}. Investigate before retrying." >&2
else
  echo "!! No previous binary to roll back to; leaving ${VERSION_NAME} in place." >&2
fi
exit 1
