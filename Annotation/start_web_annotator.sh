#!/usr/bin/env bash
# Start the fast web annotator (recommended).
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
WEB_PORT="${WEB_PORT:-6080}"
PYTHON="${PYTHON:-/root/miniconda3/envs/annotation/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

export PYTHONUNBUFFERED=1

# Free the port from old noVNC / previous runs.
pkill -f 'Annotation/web/server.py' 2>/dev/null || true
pkill -f "websockify.*${WEB_PORT}" 2>/dev/null || true
sleep 1

nohup "${PYTHON}" "${REPO}/web/server.py" --host 0.0.0.0 --port "${WEB_PORT}" \
  > /tmp/web_annotator.log 2>&1 &

for _ in $(seq 1 20); do
  if curl -sf "http://127.0.0.1:${WEB_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -sf "http://127.0.0.1:${WEB_PORT}/api/health" >/dev/null 2>&1; then
  echo "ERROR: annotator failed to start. Log:"
  tail -30 /tmp/web_annotator.log 2>/dev/null || true
  exit 1
fi

echo "Web annotator is running."
echo ""
echo "  1. In Cursor: open the Ports panel and forward port ${WEB_PORT}"
echo "  2. Open in browser:"
echo "       http://localhost:${WEB_PORT}/"
echo ""
echo "Logs: /tmp/web_annotator.log"
echo "Stop: pkill -f 'Annotation/web/server.py'"
