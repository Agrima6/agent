#!/usr/bin/env bash
# One-time: self-hosted S3-compatible storage (SeaweedFS) for WorkmateIQ client-service, plus the env it needs.
# Run as root on the server. Idempotent. Credentials are generated here and never printed.
set -euo pipefail
ENV=/opt/workmateiq/services/client-service/.env
BUCKET=ai-interview-storage

# Remove the failed MinIO attempt, if any (MinIO no longer publishes images or binaries).
systemctl disable --now minio 2>/dev/null || true
rm -f /etc/systemd/system/minio.service /etc/minio.env /usr/local/bin/minio /usr/local/bin/mc
systemctl daemon-reload

cp "$ENV" "$ENV.bak6.$(date +%F)"
umask 077
[ -f /root/s3.env ] || printf 'S3_USER=wmiq%s\nS3_PASS=%s\n' "$(openssl rand -hex 6)" "$(openssl rand -hex 24)" > /root/s3.env
. /root/s3.env
cat > /root/s3.json <<JSON
{"identities":[{"name":"app","credentials":[{"accessKey":"$S3_USER","secretKey":"$S3_PASS"}],"actions":["Admin","Read","Write","List","Tagging"]}]}
JSON
install -d -m 700 /opt/seaweed-data

docker rm -f s3store >/dev/null 2>&1 || true
docker run -d --name s3store --restart unless-stopped -p 127.0.0.1:9000:8333 \
  -v /opt/seaweed-data:/data -v /root/s3.json:/etc/s3.json:ro \
  chrislusf/seaweedfs server -dir=/data -s3 -s3.config=/etc/s3.json >/dev/null
echo "waiting for storage..."; sleep 20

aws() { docker run --rm --network host -v /tmp:/tmp -e AWS_ACCESS_KEY_ID="$S3_USER" -e AWS_SECRET_ACCESS_KEY="$S3_PASS" \
  -e AWS_DEFAULT_REGION=us-east-1 amazon/aws-cli --endpoint-url http://127.0.0.1:9000 "$@"; }
aws s3 mb "s3://$BUCKET" || true
echo "round-trip-ok" > /tmp/s3-test.txt
aws s3 cp /tmp/s3-test.txt "s3://$BUCKET/_test.txt" >/dev/null
echo "read back: $(aws s3 cp "s3://$BUCKET/_test.txt" - )"
aws s3 rm "s3://$BUCKET/_test.txt" >/dev/null

for kv in "AWS_ACCESS_KEY_ID=$S3_USER" "AWS_SECRET_ACCESS_KEY=$S3_PASS" "AWS_REGION=us-east-1" \
  "S3_ENDPOINT=http://127.0.0.1:9000" "S3_FORCE_PATH_STYLE=true" "S3_BUCKET_NAME=$BUCKET" \
  "AUTH_SERVICE_URL=http://localhost:4001" "COMMUNICATION_SERVICE_URL=http://localhost:4005" \
  "FRONTEND_BASE_URL=https://workmateiq.com"; do
  k=${kv%%=*}; grep -q "^$k=." "$ENV" || echo "$kv" >> "$ENV"
done
echo "--- client-service env (names only):"
for k in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3_ENDPOINT S3_BUCKET_NAME AUTH_SERVICE_URL COMMUNICATION_SERVICE_URL FRONTEND_BASE_URL AGENT_SERVICE_URL AGENT_SERVICE_KEY; do
  grep -q "^$k=." "$ENV" && echo "OK       $k" || echo "MISSING  $k"
done
