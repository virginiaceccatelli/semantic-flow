#!/bin/csh
# BranchExec (stages 250-255) — on REAL code: is the branch an `if` will take
# decided at the `if` itself once the input is known, and does that internal
# decision control the model's predicted output? Synthetic programs only supply
# the direction, the layer and the dose. See docs/BRANCHEXEC.md.
#
#   jobs/branchexec.csh                                   # deepseek-coder-6.7b, full run
#   setenv SMOKE 1; jobs/branchexec.csh                   # small end-to-end run first
#   setenv MODEL deepseek-coder-1.3b; jobs/branchexec.csh # replication
#
# Env overrides: MODEL, DTYPE, TAG, SMOKE, RUN.
# Run one model at a time (6.7b co-resident with another model gets offloaded).
# Every stage writes gates.json; `|| exit 1` stops at the first failure.
source jobs/common.csh
if (! $?RUN) setenv RUN "$PYTHON"
if (! $?DTYPE) setenv DTYPE float16
if (! $?TAG) setenv TAG full

if ($?SMOKE) then
    set TAG = smoke
    set BUILD_ARGS = (--synthetic-programs 150 --limit 60)
    set STEER_ARGS = (--max-syn 30 --max-real 30 --examples 3 --doses 0.05,0.2)
else
    set BUILD_ARGS = ()
    set STEER_ARGS = ()
endif

set OUT = "results/branchexec/${MODEL}/${TAG}"
mkdir -p "$OUT"

echo "=== stage 250: execution-verified flip pairs (synthetic + CruxEval/MBPP/HumanEval) — CPU ==="
$RUN scripts/250_branch_build.py --model "$MODEL" --output "$OUT/build" $BUILD_ARGS || exit 1
cat "$OUT/build/summary.json"

echo "=== stage 251: residual states, both prompt orders, every layer — GPU ==="
$RUN scripts/251_branch_extract.py --build "$OUT/build" --output "$OUT/extract" \
    --model "$MODEL" --dtype "$DTYPE" || exit 1

echo "=== stage 252: synthetic directions, frozen, read on real pairs — CPU ==="
$RUN scripts/252_branch_readout.py --build "$OUT/build" --extract "$OUT/extract" \
    --output "$OUT/readout" || exit 1
cat "$OUT/readout/report.md"

echo "=== stage 253: unsteered behaviour and capability gate — GPU ==="
$RUN scripts/253_branch_behaviour.py --build "$OUT/build" --output "$OUT/behaviour" \
    --model "$MODEL" --dtype "$DTYPE" || exit 1
cat "$OUT/behaviour/behaviour_summary.csv"

echo "=== stage 254: steer toward the other branch, scored by execution — GPU ==="
$RUN scripts/254_branch_steer.py --build "$OUT/build" --readout "$OUT/readout" \
    --behaviour "$OUT/behaviour" --output "$OUT/steer" --model "$MODEL" --dtype "$DTYPE" \
    $STEER_ARGS || exit 1

echo "=== stage 255: report — CPU ==="
$RUN scripts/255_branch_report.py --build "$OUT/build" --extract "$OUT/extract" \
    --readout "$OUT/readout" --behaviour "$OUT/behaviour" --steer "$OUT/steer" \
    --output "$OUT/report" || exit 1
echo "Report: $OUT/report/report.md"
