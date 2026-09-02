#!/bin/csh
# Final E13 run, native csh/tcsh: binding DAS plus the matched causal
# answer-token control. J/R lenses are optional diagnostics and are deliberately
# omitted from this claim-bearing run.

if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE float16
if (! $?LAYERS) setenv LAYERS "6,12,18"
if (! $?RANKS) setenv RANKS "1,2,4,8"

set OUT = "results/binding/${MODEL}"
set VARIANTS = "das_binding,das_answer_control,mean_difference,random_rank,random_norm,noop,whole_state"

echo "=== ${MODEL}: whole-state ceiling (H3), dtype ${DTYPE} ==="
$PYTHON scripts/105_binding_ceiling.py --model "$MODEL" \
    --layers "$LAYERS" --dtype "$DTYPE" --strict
if ($status != 0) exit 1

echo "=== ${MODEL}: binding DAS + matched answer-DAS control (H4/H5) ==="
# No --layers: stage 106 uses the single layer selected by stage 105.
# No --strict: a negative H5 is a valid result and must still be reported.
$PYTHON scripts/106_binding_interchange.py --model "$MODEL" \
    --ranks "$RANKS" --dtype "$DTYPE" --variants "$VARIANTS"
if ($status != 0) exit 1

echo "=== ${MODEL}: report and diagnosis ==="
$PYTHON scripts/107_binding_report.py --model "$MODEL"
if ($status != 0) exit 1
$PYTHON scripts/108_binding_diagnose.py --model "$MODEL" --verbose
if ($status != 0) exit 1

echo "Read ${OUT}/e13_report.md and ${OUT}/interchange_panel.csv"
