#!/bin/csh
# E11 PILOT — 1.3b, 200 pairs, two operation families, four layers.
#
# The whole pilot in one job: pairs -> frozen lenses -> readout -> swap ->
# go/no-go. Stage 71 gates 72/73 (a failed lens check aborts the run), and
# stage 74 writes results/jspace/$MODEL/go_no_go.{yaml,md} with the verdict.
#
# Only recommend the full run if stage 74 says GO:
#   behavioural balanced accuracy >= 0.75, the J-lens readout beats the
#   Gram-matched random control, and the coordinate swap moves the logits
#   toward the swapped-in value's answer.
#
#   setenv MODEL deepseek-coder-1.3b; jobs/jspace_pilot.csh
#
# Env overrides: MODEL, DTYPE (float32 if fp16 gradients go non-finite),
# LAYERS, FAMILIES, NBASES.
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?MODEL) setenv MODEL deepseek-coder-1.3b
if (! $?DTYPE) setenv DTYPE float16
if (! $?LAYERS) setenv LAYERS "6,12,18,23"
if (! $?FAMILIES) setenv FAMILIES "affine,threshold"
if (! $?NBASES) setenv NBASES 100

set PAIRS = "data/synthetic/jspace_pairs_${MODEL}.jsonl"

echo "=== stage 70: pairs ($NBASES bases x $FAMILIES) ==="
$RUN scripts/70_jspace_pairs.py --model "$MODEL" \
    --n-bases "$NBASES" --families "$FAMILIES" || exit 1

echo "=== stage 71: frozen lenses (GATE) ==="
$RUN scripts/71_jspace_lens.py --model "$MODEL" \
    --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" \
    --n-build 150 --n-seeds 3 || exit 1

echo "=== stage 72: readout ==="
$RUN scripts/72_jspace_readout.py --model "$MODEL" \
    --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" || exit 1

echo "=== stage 73: coordinate swap ==="
$RUN scripts/73_jspace_swap.py --model "$MODEL" \
    --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" \
    --band-width 3 || exit 1

echo "=== stage 74: go/no-go ==="
$RUN scripts/74_jspace_report.py --model "$MODEL"

echo "=== stage 90: figures ==="
$RUN scripts/90_make_paper_assets.py
