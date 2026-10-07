#!/bin/csh
# BranchExec (stages 250-260) for one model — on REAL code: is the branch an `if`
# will take computed at the `if` once the input is known, does the model's answer
# use it, and can steering repair wrong-branch answers? Synthetic programs only
# supply the direction, the layer and the dose. See docs/BRANCHEXEC.md.
#
#   jobs/branchexec.csh                                   # deepseek-coder-6.7b, full run
#   setenv SMOKE 1; jobs/branchexec.csh                   # small end-to-end run first
#   setenv MODEL deepseek-coder-1.3b; jobs/branchexec.csh # another model
#
# Stages whose gates.json already says "passed" are SKIPPED, so re-running this
# on a finished run only adds what is missing (e.g. stages 256-257). A stage that
# failed must have its output directory removed (or use --resume for 251/254/257).
#
# Env overrides: MODEL, DTYPE, TAG, SMOKE, RUN.
# Run one model at a time (6.7b co-resident with another model gets offloaded).
source jobs/common.csh
if (! $?RUN) setenv RUN "$PYTHON"
if (! $?DTYPE) setenv DTYPE float16
if (! $?TAG) setenv TAG full

if ($?SMOKE) then
    set TAG = smoke
    set BUILD_ARGS = (--synthetic-programs 150 --limit 60)
    set STEER_ARGS = (--max-syn 30 --max-real 30 --examples 3 --doses 0.05,0.2)
    set REPAIR_ARGS = (--max-members 30 --examples 3 --doses 0.1,0.4)
    set LOCATE_ARGS = (--max-members 10 --layer-stride 4 --doses 1.0)
else
    set BUILD_ARGS = ()
    set STEER_ARGS = ()
    set REPAIR_ARGS = ()
    set LOCATE_ARGS = ()
endif

set OUT = "results/branchexec/${MODEL}/${TAG}"
mkdir -p "$OUT"
echo "=== BranchExec: $MODEL ($DTYPE) -> $OUT ==="

set S = `grep -sc '"status": "passed"' "$OUT/build/gates.json"`
if ("$S" != "1") then
    echo "=== stage 250: execution-verified flip pairs (synthetic + CruxEval/MBPP/HumanEval) — CPU ==="
    $RUN scripts/250_branch_build.py --model "$MODEL" --output "$OUT/build" $BUILD_ARGS || exit 1
endif
cat "$OUT/build/summary.json"

set S = `grep -sc '"status": "passed"' "$OUT/extract/gates.json"`
if ("$S" != "1") then
    echo "=== stage 251: residual states, both prompt orders, every layer — GPU ==="
    $RUN scripts/251_branch_extract.py --build "$OUT/build" --output "$OUT/extract" \
        --model "$MODEL" --dtype "$DTYPE" || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/readout/gates.json"`
if ("$S" != "1") then
    echo "=== stage 252: synthetic directions, frozen, read on real pairs — CPU ==="
    $RUN scripts/252_branch_readout.py --build "$OUT/build" --extract "$OUT/extract" \
        --output "$OUT/readout" || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/behaviour/gates.json"`
if ("$S" != "1") then
    echo "=== stage 253: unsteered behaviour — GPU ==="
    $RUN scripts/253_branch_behaviour.py --build "$OUT/build" --output "$OUT/behaviour" \
        --model "$MODEL" --dtype "$DTYPE" || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/link/gates.json"`
if ("$S" != "1") then
    echo "=== stage 256: readout vs the model's own branch choice — CPU ==="
    $RUN scripts/256_branch_link.py --build "$OUT/build" --extract "$OUT/extract" \
        --readout "$OUT/readout" --behaviour "$OUT/behaviour" --output "$OUT/link" || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/repair/gates.json"`
if ("$S" != "1") then
    echo "=== stage 257: repair wrong-branch answers — GPU ==="
    $RUN scripts/257_branch_repair.py --build "$OUT/build" --readout "$OUT/readout" \
        --behaviour "$OUT/behaviour" --output "$OUT/repair" --model "$MODEL" --dtype "$DTYPE" \
        $REPAIR_ARGS || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/natural/gates.json"`
if ("$S" != "1") then
    echo "=== stage 259: branch-following in the CruxEval order — GPU ==="
    $RUN scripts/259_branch_natural.py --build "$OUT/build" --behaviour "$OUT/behaviour" \
        --output "$OUT/natural" --model "$MODEL" --dtype "$DTYPE" || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/locate/gates.json"`
if ("$S" != "1") then
    echo "=== stage 260: single-block repair across depth (if / body / answer) — GPU ==="
    $RUN scripts/260_branch_locate.py --build "$OUT/build" --readout "$OUT/readout" \
        --behaviour "$OUT/behaviour" --output "$OUT/locate" --model "$MODEL" --dtype "$DTYPE" \
        $LOCATE_ARGS || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/steer/gates.json"`
if ("$S" != "1") then
    echo "=== stage 254: flip steering toward the other branch — GPU ==="
    $RUN scripts/254_branch_steer.py --build "$OUT/build" --readout "$OUT/readout" \
        --behaviour "$OUT/behaviour" --output "$OUT/steer" --model "$MODEL" --dtype "$DTYPE" \
        $STEER_ARGS || exit 1
endif

set S = `grep -sc '"status": "passed"' "$OUT/report_v2/gates.json"`
if ("$S" != "1") then
    echo "=== stage 255: per-run report (v2) — CPU ==="
    $RUN scripts/255_branch_report.py --build "$OUT/build" --extract "$OUT/extract" \
        --readout "$OUT/readout" --behaviour "$OUT/behaviour" --steer "$OUT/steer" \
        --output "$OUT/report_v2" || exit 1
endif
echo "Done: $OUT  (link, repair, natural, locate, report_v2: each has report.md)"
