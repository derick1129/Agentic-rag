#!/bin/sh
set -eu

max_attempts="${MIGRATION_MAX_ATTEMPTS:-30}"
attempt=1

while ! alembic upgrade head; do
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "Database migrations failed after ${max_attempts} attempts." >&2
        exit 1
    fi
    echo "Database is not ready; retrying migrations (${attempt}/${max_attempts})..." >&2
    attempt=$((attempt + 1))
    sleep "${MIGRATION_RETRY_SECONDS:-2}"
done

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec uvicorn adaptive.api.main:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}"
