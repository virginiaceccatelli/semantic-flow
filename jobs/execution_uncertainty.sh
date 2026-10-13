#!/usr/bin/env bash
# Run inside an allocated cluster job; this script does not request GPUs.
# CPU: bash jobs/execution_uncertainty.sh prepare
# GPU: bash jobs/execution_uncertainty.sh evaluate
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-deepseek-coder-1.3b}
LAYER=${LAYER:-6}
LOCATION=${LOCATION:-prompt_end}
DATASETS=${DATASETS:-"mbpp cruxeval"}
OUT=${OUT:-results/execution_uncertainty}
LENS_DIR=${LENS_DIR:-results/workspace_lens/$MODEL/j-lens}

case "${1:-}" in
  prepare)
    for dataset in $DATASETS; do
      "$PYTHON" scripts/290_execution_uncertainty.py prepare \
        --dataset "$dataset" --output "$OUT/prepared/$dataset"
    done
    ;;
  evaluate)
    for dataset in $DATASETS; do
      "$PYTHON" scripts/290_execution_uncertainty.py evaluate \
        --prepared "$OUT/prepared/$dataset" \
        --output "$OUT/$MODEL/block${LAYER}-${LOCATION}/$dataset" \
        --model "$MODEL" --lens-dir "$LENS_DIR" --layer "$LAYER" \
        --location "$LOCATION" --resume
    done
    ;;
  *) echo "Usage: bash jobs/execution_uncertainty.sh {prepare|evaluate}" >&2; exit 2 ;;
esac
