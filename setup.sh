#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Missing requirements.txt at ${REQ_FILE}" >&2
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Creating virtual environment at ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
else
  echo "Reusing existing virtual environment at ${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
pip install -r "${REQ_FILE}"

echo "Environment ready. Activate with:"
echo "  source ${VENV_PATH}/bin/activate"

