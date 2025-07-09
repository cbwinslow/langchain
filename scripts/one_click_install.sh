#!/bin/bash
set -euo pipefail

log_file=$(mktemp)

function cleanup {
  rm -f "$log_file"
}
trap cleanup EXIT

echo "[INFO] Starting one-click installer..."

command -v docker >/dev/null 2>&1 || { echo "[ERROR] Docker not found. Please install Docker."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "[ERROR] docker-compose not found. Please install Docker Compose."; exit 1; }
command -v ollama >/dev/null 2>&1 || echo "[WARN] Ollama executable not found. Troubleshooting agent may not work."

if docker-compose up -d >"$log_file" 2>&1; then
  echo "[INFO] Services started successfully."
else
  echo "[ERROR] docker-compose failed. Invoking AI troubleshooting agent..."
  cat "$log_file" | python scripts/troubleshoot_agent.py || true
fi
