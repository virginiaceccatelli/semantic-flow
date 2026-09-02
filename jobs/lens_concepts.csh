#!/bin/csh
# Run stage 206 against existing stage-201 J/R artifacts, then regenerate the
# stage-205 report. This does NOT refit lenses.
#
#   screen -L -Logfile concepts-6.7b.log -dmS concepts-6.7b \
#       env MODEL=deepseek-coder-6.7b jobs/lens_concepts.csh

if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE bfloat16
if (! $?N_BASES) setenv N_BASES 100
if (! $?UNEMBED_BATCH) setenv UNEMBED_BATCH 32

set OUT = "results/workspace_lens/${MODEL}"
if (! -e "${OUT}/j-lens/lens.pt") then
    echo "*** Missing ${OUT}/j-lens/lens.pt — run stage 201 first."
    exit 1
endif
if (! -e "${OUT}/r-lens/lens.pt") then
    echo "*** Missing ${OUT}/r-lens/lens.pt — run stage 201 first."
    exit 1
endif

echo "=== stage 206: semantic-concept J/R/logit panel — GPU ==="
$PYTHON scripts/206_lens_concepts.py --model "$MODEL" --lens-dir "$OUT" \
    --output "${OUT}/concepts" --dtype "$DTYPE" --n-bases "$N_BASES" \
    --unembed-batch-size "$UNEMBED_BATCH" --checkpoint-every 5 --resume \
    --require-rlens
if ($status != 0) exit 1

echo "=== stage 205: regenerate the complete lens report — CPU ==="
$PYTHON scripts/205_lens_report.py --model "$MODEL"
if ($status != 0) exit 1

echo "Read ${OUT}/concepts/workspace_lens_concepts.md and ${OUT}/workspace_lens_report.md."
