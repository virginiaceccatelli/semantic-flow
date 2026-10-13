#!/usr/bin/env bash
# Stage 291. Run inside an allocated cluster job; this script does not request GPUs.
# GPU: MODEL=deepseek-coder-1.3b bash jobs/execution_uncertainty_sweep.sh evaluate
# CPU: bash jobs/execution_uncertainty_sweep.sh analyze
# Reuses the stage-290 prepared datasets (bash jobs/execution_uncertainty.sh prepare).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-deepseek-coder-1.3b}
MODELS=${MODELS:-"deepseek-coder-1.3b deepseek-coder-6.7b"}
DATASETS=${DATASETS:-"mbpp cruxeval"}
PREPARED=${PREPARED:-results/execution_uncertainty/prepared}
OUT=${OUT:-results/execution_uncertainty/stage291}
LENS_DIR=${LENS_DIR:-results/workspace_lens/$MODEL/j-lens}

case "${1:-}" in
  evaluate)
    for dataset in $DATASETS; do
      "$PYTHON" scripts/291_execution_uncertainty_sweep.py evaluate \
        --prepared "$PREPARED/$dataset" --output "$OUT/$MODEL/$dataset" \
        --model "$MODEL" --lens-dir "$LENS_DIR" --resume
    done
    ;;
  analyze)
    runs=()
    for model in $MODELS; do
      for dataset in $DATASETS; do
        [ -f "$OUT/$model/$dataset/gates.json" ] && runs+=("$OUT/$model/$dataset")
      done
    done
    "$PYTHON" scripts/291_execution_uncertainty_sweep.py analyze --runs "${runs[@]}" --output "$OUT/analysis"
    ;;
  *) echo "Usage: bash jobs/execution_uncertainty_sweep.sh {evaluate|analyze}" >&2; exit 2 ;;
esac
