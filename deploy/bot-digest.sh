#!/usr/bin/env bash
# Daily eval digest: render yesterday's per-driver/per-series report from the
# durable journal (restart-proof), archive it to S3, and leave it where the
# dashboard's /api/digest tile can serve it. Run as the `bot` user by
# bot-digest.timer ~03:40 UTC (after the 03:17 S3 backup).
set -euo pipefail

CONFIG_DIR=/etc/bot/config
DIGEST_DIR=/var/lib/bot/data/digests
BIN=/opt/bot/bin/bot
DATE="$(date -u -d 'yesterday' +%F)"
OUT="$DIGEST_DIR/$DATE.md"

mkdir -p "$DIGEST_DIR"

# bot digest reads the sqlite index (WAL — safe concurrently with the running
# bot) + the driver-attrib / shadow-stops side-channels; it does NOT need the
# live process and never touches the paper wallet (PnL of record is journalled).
"$BIN" digest --date "$DATE" --out "$OUT" --config-dir "$CONFIG_DIR"

# Archive to S3 so the operator can read it from a phone without SSH. The bucket
# is set in /etc/bot/s3-backup.env; the EC2 instance role has write+list only.
if [ -f /etc/bot/s3-backup.env ]; then
  # shellcheck disable=SC1091
  . /etc/bot/s3-backup.env
  aws s3 cp "$OUT" "s3://${BUCKET}/vps/digests/$DATE.md" >/dev/null && \
    echo "digest $DATE -> s3://${BUCKET}/vps/digests/$DATE.md"
fi
echo "digest written: $OUT"
