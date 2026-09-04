#!/bin/bash
# groundgraph: first-boot PostgreSQL bootstrap.
#
# Runs once when the data directory is empty (via the official
# postgres entrypoint). The official entrypoint executes any
# ``*.sh`` file in /docker-entrypoint-initdb.d with the env vars it
# itself received (POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB).
#
# Responsibilities:
#   1. Create the dedicated Phoenix database (PHOENIX_DB) on the same
#      instance, separate from the application database. The two
#      databases MUST NOT share an Alembic-managed schema; see
#      plan.md §5.3.
#   2. Enable the pgvector extension in the application database only.
#
# Identifier quoting:
#   PostgreSQL double-quotes identifiers but a `"` inside the
#   identifier is allowed only if doubled. We do not accept
#   arbitrary user input here, but we do still sanitize via
#   ``psql -tAc`` to refuse anything containing a ``"``.

set -euo pipefail

# Sanitize: the entrypoint passes its env through to us. Refuse
# identifiers containing a double quote, semicolon, or backslash.
sanitize_ident() {
    local v="${1-}"
    if [[ -z "$v" ]]; then
        echo "init: identifier is empty" >&2
        exit 1
    fi
    if [[ "$v" == *'"'* || "$v" == *';'* || "$v" == *'\\'* ]]; then
        echo "init: unsafe identifier: $v" >&2
        exit 1
    fi
    printf '%s' "$v"
}

APP_DB="$(sanitize_ident "${POSTGRES_DB:-groundgraph}")"
PHOENIX_DB_VAL="$(sanitize_ident "${PHOENIX_DB:-phoenix}")"

if [[ "$APP_DB" == "$PHOENIX_DB_VAL" ]]; then
    echo "init: POSTGRES_DB and PHOENIX_DB must differ (got '$APP_DB')" >&2
    exit 1
fi

echo "init: creating phoenix database '$PHOENIX_DB_VAL' (app: '$APP_DB')"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$APP_DB" <<SQL
CREATE DATABASE "$PHOENIX_DB_VAL";
SQL

echo "init: enabling pgvector in '$APP_DB'"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$APP_DB" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;
SQL

echo "init: bootstrap complete"
