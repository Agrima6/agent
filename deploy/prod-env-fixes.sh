#!/usr/bin/env bash
# One-time, idempotent: the settings production needs BEFORE the new WorkmateIQ code is merged to main.
# Run as root, after setup-s3.sh. Backs up each .env, only adds what is missing, prints names only.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
. /root/s3.env
SV=/opt/workmateiq/services

add() {  # add <service> KEY=VALUE ...
  local f="$SV/$1/.env"; shift
  [ -f "$f.bak7.$(date +%F)" ] || cp "$f" "$f.bak7.$(date +%F)"
  for kv in "$@"; do k=${kv%%=*}; grep -q "^$k=." "$f" || echo "$kv" >> "$f"; done
}

# onboarding-service: stores uploads in the same self-hosted S3 (it refuses to start in production without credentials).
add onboarding-service "AWS_ACCESS_KEY_ID=$S3_USER" "AWS_SECRET_ACCESS_KEY=$S3_PASS" "AWS_REGION=us-east-1" \
  "S3_ENDPOINT=http://127.0.0.1:9000" "S3_FORCE_PATH_STYLE=true" "S3_BUCKET_NAME=ai-interview-storage"

# communication-service: no queue here, so keep sending emails inline exactly as production does today.
# The AWS values only let the service start; they are never used while SQS_QUEUE_URL is unset.
add communication-service "AWS_ACCESS_KEY_ID=$S3_USER" "AWS_SECRET_ACCESS_KEY=$S3_PASS" "AWS_REGION=us-east-1" \
  "ALLOW_SYNC_NOTIFY=true" "PUBLIC_BASE_URL=https://api.workmateiq.com"

echo "--- present now (names only):"
for p in "onboarding-service AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_ENDPOINT S3_BUCKET_NAME" \
         "communication-service AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY ALLOW_SYNC_NOTIFY PUBLIC_BASE_URL"; do
  set -- $p; s=$1; shift
  for k in "$@"; do grep -q "^$k=." "$SV/$s/.env" && echo "OK       $s $k" || echo "MISSING  $s $k"; done
done
