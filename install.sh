#!/usr/bin/env bash
# install.sh — set up EVA on Linux / macOS.
#
# Usage:
#   bash install.sh            # CPU install, creates .venv, installs with [ui]
#   bash install.sh --gpu      # install the CUDA wheel of torch
#   bash install.sh --no-ui    # skip fastapi/uvicorn
#
# After install:
#   source .venv/bin/activate
#   python run.py ui

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

GPU=0
WITH_UI=1
for arg in "$@"; do
  case "${arg}" in
    --gpu)   GPU=1 ;;
    --no-ui) WITH_UI=0 ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "unknown option: ${arg}" >&2
      exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "${PY_VER}" in
  3.10|3.11|3.12|3.13) : ;;
  *)
    echo "EVA requires Python 3.10+ (found ${PY_VER})." >&2
    exit 1 ;;
esac

echo "[eva] Creating virtual environment in .venv (Python ${PY_VER})"
"${PYTHON_BIN}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools >/dev/null

if [ "${GPU}" -eq 1 ]; then
  echo "[eva] Installing CUDA torch (this downloads ~2 GB)."
  pip install --index-url https://download.pytorch.org/whl/cu121 torch
else
  echo "[eva] Installing CPU torch."
  pip install torch
fi

EXTRA=""
if [ "${WITH_UI}" -eq 1 ]; then
  EXTRA="[ui]"
fi

echo "[eva] Installing EVA (editable, extras=${EXTRA:-none})."
pip install -e ".${EXTRA}"

if [ -f ".pre-commit-config.yaml" ]; then
  echo "[eva] Configuring pre-commit hooks."
  pip install pre-commit >/dev/null
  pre-commit install >/dev/null
fi

echo
echo "[eva] Install complete."
echo "     source .venv/bin/activate"
echo "     python run.py ui     # open http://127.0.0.1:8765"
