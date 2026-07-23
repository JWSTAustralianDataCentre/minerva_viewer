#!/bin/bash
# Start (or ensure) the MINERVA viewer in PUBLIC mode: Basic-auth-gated server
# + ngrok tunnel on port 8321. Idempotent — safe to re-run; prints the public URL.
# Credentials live in .secrets/public_auth (chmod 600, gitignored).
set -e
cd "$(dirname "$0")"
source .secrets/public_auth
export MINERVA_AUTH_USER MINERVA_AUTH_PASS

# server (restart only if it isn't already running WITH auth: probe for 401)
code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8321/api/fields || true)
if [ "$code" != "401" ]; then
  pkill -f "uvicorn server[.]main" 2>/dev/null || true; sleep 2
  nohup bash run.sh > /tmp/minerva_viewer_public.log 2>&1 &
  for i in $(seq 1 60); do
    [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8321/healthz || true)" = "200" ] && break
    sleep 2
  done
fi

# ngrok (start only if its local API isn't answering)
if ! curl -s -m 2 http://127.0.0.1:4040/api/tunnels >/dev/null 2>&1; then
  nohup ./.tools/ngrok http 8321 --log stdout > /tmp/minerva_ngrok.log 2>&1 &
  sleep 5
fi

URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -oP '"public_url":"\K[^"]+' | head -1)
echo "public URL: ${URL:-NOT UP - check /tmp/minerva_ngrok.log}"
echo "login: $MINERVA_AUTH_USER / (see .secrets/public_auth)"
