#!/usr/bin/env bash
# Off-box backup of the journal (the source of truth) + the sqlite index + the
# config. Safe to run WHILE the bot is live: rotated gzip segments are immutable,
# the active trailing segment is tolerated by ReplayReader, and the sqlite index
# is snapshotted via SQLite's online .backup (never a raw cp of a live WAL DB).
#
# Set REMOTE to an off-box / off-site rsync target before relying on this.
# Run as the `bot` user (or root). Schedule from cron (example at the bottom).
#
# Secrets (/etc/bot/bot.env) are NOT backed up here — store them separately in an
# encrypted secrets manager.
set -euo pipefail

DATA="${DATA:-/var/lib/bot/data}"
CONFIG_DIR="${CONFIG_DIR:-/etc/bot/config}"
REMOTE="${REMOTE:?set REMOTE, e.g. backup-host:/backups/bot}"
LOCAL_SNAP="${LOCAL_SNAP:-/var/lib/bot/backup}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${LOCAL_SNAP}"

echo "==> Consistent sqlite snapshot (online .backup — no stop needed)"
sqlite3 "${DATA}/journal.sqlite" ".backup '${LOCAL_SNAP}/journal-${STAMP}.sqlite'"

echo "==> Pushing journal segments off-box (immutable once rotated)"
rsync -a "${DATA}/journal/" "${REMOTE}/journal/"

echo "==> Pushing the sqlite snapshot off-box"
rsync -a "${LOCAL_SNAP}/journal-${STAMP}.sqlite" "${REMOTE}/sqlite/"

echo "==> Pushing config off-box (secrets are handled separately)"
rsync -a "${CONFIG_DIR}/" "${REMOTE}/config/"

echo "==> Pruning local sqlite snapshots older than 7 days"
find "${LOCAL_SNAP}" -name 'journal-*.sqlite' -mtime +7 -delete

echo "==> Backup ${STAMP} complete."

# --- Restore (manual) -------------------------------------------------------
#   sudo systemctl stop bot
#   sudo rsync -a backup-host:/backups/bot/journal/ /var/lib/bot/data/journal/
#   sudo cp /restore/journal-<stamp>.sqlite /var/lib/bot/data/journal.sqlite
#   sudo rm -f /var/lib/bot/data/journal.sqlite-wal /var/lib/bot/data/journal.sqlite-shm
#   sudo rsync -a backup-host:/backups/bot/config/ /etc/bot/config/
#   sudo chown -R bot:bot /var/lib/bot/data
#   sudo systemctl start bot
#
# --- Schedule (crontab; run nightly off the hour to avoid rotation contention)
#   17 3 * * *  REMOTE=backup-host:/backups/bot /opt/bot/deploy/backup.sh \
#                 >> /var/lib/bot/data/logs/backup.log 2>&1
#   (rotate that backup.log yourself — it is outside the bot's rolling appender.)
