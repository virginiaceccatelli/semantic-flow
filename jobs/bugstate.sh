#!/usr/bin/env bash
# Run `run` inside a GPU allocation. Preparation is CPU/network work.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv-bugstate/bin/python}
case "${1:-}" in
  setup)
    python3 -m venv .venv-bugstate
    "$PYTHON" -m pip install -r requirements-bugstate.txt
    ;;
  prepare|run|report)
    stage=$1
    shift
    "$PYTHON" -m src.bugstate "$stage" "$@"
    ;;
  download)
    "$PYTHON" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B")'
    ;;
  test)
    "$PYTHON" -m pytest tests/bugstate -q
    ;;
  *) echo 'Usage: bash jobs/bugstate.sh {setup|prepare|download|run|report|test} [options]' >&2; exit 2 ;;
esac
