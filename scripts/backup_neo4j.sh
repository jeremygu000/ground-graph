#!/usr/bin/env bash
# backup_neo4j.sh — Neo4j backup using neo4j-admin database dump.
# Usage: ./scripts/backup_neo4j.sh [BUCKET] [NEO4J_HOST]
# Env: NEO4J_AUTH (user/password), AWS_*

set -euo pipefail

BUCKET="${1:-${S3_BUCKET_BACKUP:-groundgraph-backups}}"
NEO4J_HOST="${2:-${NEO4J_HOST:-localhost}}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
BACKUP_DIR="/tmp/neo4j_backup.$$"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

NEO4J_AUTH="${NEO4J_AUTH:-neo4j/admin}"
USER="${NEO4J_AUTH%/*}"
PASS="${NEO4J_AUTH#*/}"

cleanup() { rm -rf "$BACKUP_DIR"; }
trap cleanup EXIT

mkdir -p "$BACKUP_DIR/dumps"

echo "[neo4j-backup] Starting backup from $NEO4J_HOST"

DUMP_PATH="$BACKUP_DIR/dumps/groundgraph-${TIMESTAMP}.dump"

neo4j-admin database dump neo4j \
  --to-path="$BACKUP_DIR/dumps" \
  --overwrite-destination=true \
  2>&1 || {
    echo "[neo4j-backup] WARNING: neo4j-admin dump failed; trying cypher dump"
    cypher-shell -u "$USER" -p "$PASS" \
      "CALL apoc.export.cypher.all('$DUMP_PATH', {format:'dump', useOptimizations:{type:'NONE'}})" \
      2>&1 || echo "[neo4j-backup] Both dump methods failed"
  }

if [[ -f "$DUMP_PATH" ]] || find "$BACKUP_DIR/dumps" -name "*.dump" -size +0 2>/dev/null | read -r; then
  DUMP_FILE=$(find "$BACKUP_DIR/dumps" -name "*.dump" | head -1)
  tar -czf "$BACKUP_DIR/neo4j-${TIMESTAMP}.tar.gz" -C "$BACKUP_DIR/dumps" "$(basename "$DUMP_FILE")"

  S3_PATH="s3://${BUCKET}/neo4j/${TIMESTAMP}/"
  echo "[neo4j-backup] Uploading to $S3_PATH"

  if command -v aws &>/dev/null; then
    aws s3 cp "$BACKUP_DIR/neo4j-${TIMESTAMP}.tar.gz" "${S3_PATH}neo4j-${TIMESTAMP}.tar.gz"
  else
    echo "[neo4j-backup] WARNING: aws CLI not found; backup remains in $BACKUP_DIR"
  fi
fi

echo "[neo4j-backup] Backup complete: $TIMESTAMP"
