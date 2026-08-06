#!/bin/csh
# E11 FULL RUN — 6.7b, all five operation families, all probed layers.
#
# Run ONLY after jobs/jspace_pilot.csh reports GO on 1.3b. The full run is
# roughly 20x the pilot's forward passes (five families instead of two, ten
# layers plus eight bands instead of four layers plus two, and a 4096-d model),
# so it wants its own screen session:
#
#   screen -dmS jspace-full env MODEL=deepseek-coder-6.7b jobs/jspace_full.csh
#
# Env overrides: MODEL, DTYPE, LAYERS, FAMILIES, NBASES, MAXPAIRS.
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
if (! $?DTYPE) setenv DTYPE float16
if (! $?LAYERS) setenv LAYERS "0,4,8,12,16,20,24,28,31"
if (! $?FAMILIES) setenv FAMILIES "affine,mul_sub,threshold,modulus,index"
if (! $?NBASES) setenv NBASES 120

set PAIRS = "data/synthetic/jspace_pairs_${MODEL}.jsonl"

echo "=== stage 70: pairs ($NBASES bases x $FAMILIES) ==="
$RUN scripts/70_jspace_pairs.py --model "$MODEL" \
    --n-bases "$NBASES" --families "$FAMILIES" || exit 1

echo "=== stage 71: frozen lenses (GATE) ==="
$RUN scripts/71_jspace_lens.py --model "$MODEL" \
    --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" \
    --n-corpus 160 --n-build 300 --n-seeds 3 --n-eval 200 || exit 1

echo "=== stage 72: readout ==="
$RUN scripts/72_jspace_readout.py --model "$MODEL" \
    --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" || exit 1

echo "=== stage 73: coordinate swap ==="
if ($?MAXPAIRS) then
    $RUN scripts/73_jspace_swap.py --model "$MODEL" \
        --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" \
        --band-width 3 --max-pairs "$MAXPAIRS" || exit 1
else
    $RUN scripts/73_jspace_swap.py --model "$MODEL" \
        --pairs "$PAIRS" --layers "$LAYERS" --dtype "$DTYPE" \
        --band-width 3 || exit 1
endif

echo "=== stage 74: summary report ==="
$RUN scripts/74_jspace_report.py --model "$MODEL"

echo "=== stage 90: figures ==="
$RUN scripts/90_make_paper_assets.py
