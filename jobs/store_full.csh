#!/bin/csh
# E12 FULL INSTRUMENT VALIDATION — 6.7b, 400 bases.
#
# Run ONLY after jobs/store_pilot.csh reaches INSTRUMENT VALIDATED on 1.3b, or
# after a deliberate decision that a pilot failure was about the small model
# rather than about the apparatus. E11's history is the reason for that rule:
# the 1.3b arm was run to completion before its behavioural gate came back at
# 0.53, and the 6.7b arm's own pre-registered gate failed at 0.706.
#
# 400 bases rather than E11's 120 because that run ended with 42 test base
# programs, which is the number every one of its cluster bootstraps had.
#
#   setenv MODEL deepseek-coder-6.7b; jobs/store_full.csh
#
# Env overrides: MODEL, DTYPE, LAYERS, RANKS, NBASES.
# NOTE: the default MUST be set BEFORE sourcing jobs/common.csh, which
# sets MODEL to deepseek-coder-6.7b when it is unset. Sourcing first made
# this script's own default dead code, so a bare invocation silently ran
# the large model instead of the pilot one.
if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?DTYPE) setenv DTYPE float16
if (! $?LAYERS) setenv LAYERS "8,12,16,20,24"
if (! $?RANKS) setenv RANKS "1,2,4,8,16"
if (! $?NBASES) setenv NBASES 400

set OUT = "results/store/${MODEL}"

echo "=== stage 80: text-absent counterfactuals ($NBASES bases) — CPU ==="
$RUN scripts/80_store_pairs.py --model "$MODEL" --n-bases "$NBASES" || exit 1

echo "=== stage 81: trace + reference interpreter (G0) — CPU ==="
$RUN scripts/81_store_verify.py --model "$MODEL" --strict || exit 1

echo "=== stage 82: can the model solve them? (G1) — GPU ==="
$RUN scripts/82_store_behaviour.py --model "$MODEL" \
    --dtype "$DTYPE" --strict || exit 1

echo "=== stage 83: cache anchor states — GPU ==="
$RUN scripts/83_store_extract.py --model "$MODEL" \
    --layers "$LAYERS" --dtype "$DTYPE" || exit 1

echo "=== stage 84: decodability (G2) — CPU ==="
$RUN scripts/84_store_decode.py --model "$MODEL" --strict || exit 1

echo "=== stage 85: natural transitions (G3) — CPU ==="
$RUN scripts/85_store_transition.py --model "$MODEL" --strict || exit 1

echo "=== stage 86: whole-state interchange ceiling (G4) — GPU ==="
$RUN scripts/86_store_ceiling.py --model "$MODEL" \
    --layers "$LAYERS" --dtype "$DTYPE" --strict || exit 1

echo "=== stage 87: DAS low-rank interchange + six controls (G5) — GPU ==="
$RUN scripts/87_store_interchange.py --model "$MODEL" \
    --layers "$LAYERS" --ranks "$RANKS" --dtype "$DTYPE" || exit 1

echo "=== stage 88: gated report — CPU ==="
$RUN scripts/88_store_report.py --model "$MODEL"

echo ""
echo "Read $OUT/e12_report.md. INSTRUMENT VALIDATED means the apparatus works."
echo "It is not a finding and must not be written up as one — the next step is"
echo "choosing a semantic extension in docs/RESULTS.md (open items)."
