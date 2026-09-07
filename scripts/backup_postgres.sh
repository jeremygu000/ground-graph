#!/usr/bin/env bash
# backup_postgres.sh — PostgreSQL backup with point-in-time recovery support.
# Usage: ./scripts/backup_postgres.sh [BUCKET] [PGHOST] [PGPORT]
# Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

set -euo pipefail

BUCKET="${1:-${S3_BUCKET_BACKUP:-groundgraph-backups}}"
PGHOST="${2:-${POSTGRES_HOST:-localhost}}"
PGPORT="${3:-${POSTGRES_PORT:-5432}}"
PGUSER="${POSTGRES_USER:-groundgraph}"
PGDB="${POSTGRES_DB:-groundgraph}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
BACKUP_DIR="/tmp/pg_backup.$$"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

cleanup() { rm -rf "$BACKUP_DIR"; }
trap cleanup EXIT

mkdir -p "$BACKUP_DIR"

echo "[pg-backup] Starting backup of $PGDB@$PGHOST:$PGPORT"

# Base backup
pg_basebackup \
  -h "$PGHOST" \
  -p "$PGPORT" \
  -U "$PGUSER" \
  -D "$BACKUP_DIR/base" \
  --wal-method=stream \
  --checkpoint=fast \
  --progress

# Archive WAL for PITR
LATEL_WAL=$(find "$BACKUP_DIR/base/pg_wal" -name "*.wal" | head -1 || true)
if [[ -n "$LATEL_WAL" ]]; then
  tar -czf "$BACKUP_DIR/wal.tar.gz" -C "$BACKUP_DIR/base/pg_wal" .
fi

# Compress base backup
tar -czf "$BACKUP_DIR/base.tar.gz" -C "$BACKUP_DIR/base" .

# Upload to S3
S3_PATH="s3://${BUCKET}/postgres/${TIMESTAMP}/"
echo "[pg-backup] Uploading to $S3_PATH"

if command -v aws &>/dev/null; then
  aws s3 cp "$BACKUP_DIR/base.tar.gz" "${S3_PATH}base.tar.gz"
  if [[ -f "$BACKUP_DIR/wal.tar.gz" ]]; then
    aws s3 cp "$BACKUP_DIR/wal.tar.gz" "${S3_PATH}wal.tar.gz"
  fi
  aws s3 cp - "$S3_PATHbackup_count" --metadata "count=$(date +%s)" <<<"${TIMESTAMP}"
else
  echo "[pg-backup] WARNING: aws CLI not found; backup files remain in $BACKUP_DIR"
fi

echo "[pg-backup] Backup complete: $TIMESTAMP"
