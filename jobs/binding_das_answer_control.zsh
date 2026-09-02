#!/bin/zsh
# Final E13 run: binding DAS plus one matched, causally trained answer control.
# J/R lenses are intentionally absent: they remain readout experiments and are
# not required to identify the causal intervention.
set -euo pipefail

: ${MODEL:=deepseek-coder-6.7b}
: ${DTYPE:=float16}
: ${LAYERS:=6,12,18}
: ${RANKS:=1,2,4,8}
: ${PYTHON:=python}

OUT="results/binding/${MODEL}"
VARIANTS="das_binding,das_answer_control,mean_difference,random_rank,random_norm,noop,whole_state"

echo "=== ${MODEL}: whole-state ceiling (H3), dtype ${DTYPE} ==="
"$PYTHON" scripts/105_binding_ceiling.py \
  --model "$MODEL" --layers "$LAYERS" --dtype "$DTYPE" --strict

echo "=== ${MODEL}: binding DAS + matched answer-DAS control (H4/H5) ==="
# No --layers: stage 106 must use the single layer selected by stage 105.
# No --strict: a negative H5 is a valid result and must still reach the report.
"$PYTHON" scripts/106_binding_interchange.py \
  --model "$MODEL" --ranks "$RANKS" --dtype "$DTYPE" \
  --variants "$VARIANTS"

echo "=== ${MODEL}: report and diagnosis ==="
"$PYTHON" scripts/107_binding_report.py --model "$MODEL"
"$PYTHON" scripts/108_binding_diagnose.py --model "$MODEL" --verbose

echo "Read ${OUT}/e13_report.md and ${OUT}/interchange_panel.csv"
