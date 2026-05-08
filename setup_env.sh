#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

: "${UV_DEFAULT_INDEX:=https://mirrors.aliyun.com/pypi/simple}"
export UV_DEFAULT_INDEX

cd "${PROJECT_DIR}"

uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" .venv
uv sync

echo "Environment ready: ${PROJECT_DIR}/.venv"
echo "Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
