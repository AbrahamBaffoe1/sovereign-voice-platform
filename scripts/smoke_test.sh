#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
HEADER=()
if [[ -n "${VOICE_API_KEY:-}" ]]; then HEADER=(-H "X-Voice-API-Key: ${VOICE_API_KEY}"); fi
curl -fsS "${BASE_URL}/healthz" | python -m json.tool
curl -fsS "${HEADER[@]}" "${BASE_URL}/v1/languages" | python -m json.tool
