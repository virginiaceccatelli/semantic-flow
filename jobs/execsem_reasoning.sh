#!/usr/bin/env bash
# Stage 293: execution uncertainty + E-Space with reasoning models (upstream U-Space pipeline).
# Run inside an allocated cluster job; this script does not request GPUs.
#
#   bash jobs/uspace_reasoning.sh setup                       # once (stage 292): checkout + .venv-uspace (no vLLM)
#   bash jobs/execsem_reasoning.sh prepare                    # CPU, network: CRUXEval 800, MBPP 200, pairs
#   MODEL=qwen bash jobs/uspace_reasoning.sh download         # weights + J-lens (same entries as 292)
#   MODEL=qwen bash jobs/execsem_reasoning.sh run             # GPU: steps 01-06 + bank + E-Space pass
#   bash jobs/execsem_reasoning.sh freeze                     # CPU, once, after the freeze_on model ran
#   MODELS=qwen bash jobs/execsem_reasoning.sh tables         # upstream Tables 1-2
#   bash jobs/execsem_reasoning.sh analyze                    # CPU, repository venv
set -euo pipefail
cd "$(dirname "$0")/.."

UPY=${UPY:-.venv-uspace/bin/python}   # upstream pipeline environment (torch + transformers, no vLLM)
PYTHON=${PYTHON:-.venv/bin/python}    # this repository's environment (prepare, freeze, analysis)
MODEL=${MODEL:-qwen}
MODELS=${MODELS:-$MODEL}
SEEDS=${SEEDS:-"41,42,43"}
BATCH=${BATCH:-8}                     # generation batch size (transformers; no vLLM)
STAGE=scripts/293_execsem_reasoning.py

case "${1:-}" in
  prepare)
    for ds in cruxeval mbpp cfpairs; do
      "$PYTHON" "$STAGE" prepare --dataset "$ds"
    done
    ;;
  run)
    "$UPY" "$STAGE" step 01_sample_items
    seeds=${SEEDS//,/ }
    "$UPY" "$STAGE" generate --model "$MODEL" --seeds $seeds --batch-size "$BATCH"
    for step in 03_label 04_forward; do
      "$UPY" "$STAGE" step "$step" --model "$MODEL" --seeds $seeds
    done
    "$UPY" "$STAGE" step 05_build_uspace --model "$MODEL"
    "$UPY" "$STAGE" bank --model "$MODEL"
    "$UPY" "$STAGE" forward --model "$MODEL" --seeds $seeds
    "$UPY" "$STAGE" step 06_score --model "$MODEL" --seeds $seeds
    ;;
  freeze)
    "$PYTHON" "$STAGE" freeze
    ;;
  tables)
    "$UPY" "$STAGE" step 07_tables --only-models "$MODELS" --only-seeds "$SEEDS"
    ;;
  analyze)
    "$PYTHON" "$STAGE" analyze
    ;;
  *) echo "Usage: bash jobs/execsem_reasoning.sh {prepare|run|freeze|tables|analyze}" >&2; exit 2 ;;
esac
