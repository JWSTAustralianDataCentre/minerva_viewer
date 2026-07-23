#!/usr/bin/env bash
# Launch the MINERVA Viewer v2 server.
# Binds 127.0.0.1:8321 by default; override via MINERVA_HOST / MINERVA_PORT.
# Run from the app/ directory so `server.main:app` imports correctly.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PYTHON="${MINERVA_PYTHON:-/otdata2/themiya/grizli_rebels/bin/python3}"

exec "$PYTHON" -m uvicorn server.main:app \
    --host "${MINERVA_HOST:-127.0.0.1}" \
    --port "${MINERVA_PORT:-8321}"
