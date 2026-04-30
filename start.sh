#!/usr/bin/env bash
# start.sh — activate the venv and launch EVA's UI (Linux / macOS).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
if [ ! -d .venv ]; then
  echo "[eva] No .venv found — run install.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec python run.py "${@:-ui}"
