#!/bin/csh
# Re-run only E13 stages 106-108 with the PUBLISHED stage-201 J/R lenses.
# DAS is fitted independently; the lenses are loaded only for answer-direction
# controls. Existing H0-H3 artifacts are reused, so do not run this until those
# gates and results/workspace_lens/$MODEL/{j-lens,r-lens}/lens.pt exist.
#
# From tcsh/csh, inside a detached screen session:
#   screen -L -Logfile binding-jr-6.7b.log -dmS binding-jr-6.7b \
#       env MODEL=deepseek-coder-6.7b jobs/binding_jr_controls.csh

if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE bfloat16
if (! $?RANKS) setenv RANKS "1,2,4,8"

set OUT = "results/binding/${MODEL}"
set JLENS = "results/workspace_lens/${MODEL}/j-lens"
set RLENS = "results/workspace_lens/${MODEL}/r-lens"
set PM_ARGS = ()
if ("$MODEL" == "starcoder2-3b") set PM_ARGS = ( --rlens-paperminimal auto )

if (! -e "${JLENS}/lens.pt") then
    echo "*** Missing ${JLENS}/lens.pt — run stage 201 first."
    exit 1
endif
if (! -e "${RLENS}/lens.pt") then
    echo "*** Missing ${RLENS}/lens.pt — run stage 201 first."
    exit 1
endif

echo "=== stage 106: DAS plus published J/R-lens answer controls — GPU ==="
$PYTHON scripts/106_binding_interchange.py --model "$MODEL" \
    --ranks "$RANKS" --dtype "$DTYPE" --jlens "$JLENS" --rlens "$RLENS" \
    --require-rlens $PM_ARGS
if ($status != 0) exit 1

echo "=== stage 107: regenerate the gated report — CPU ==="
$PYTHON scripts/107_binding_report.py --model "$MODEL"
if ($status != 0) exit 1

echo "=== stage 108: diagnostics — CPU ==="
$PYTHON scripts/108_binding_diagnose.py --model "$MODEL" --verbose
if ($status != 0) exit 1

echo "Read ${OUT}/e13_report.md and ${OUT}/interchange_panel.csv."
