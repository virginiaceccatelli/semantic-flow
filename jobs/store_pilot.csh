#!/bin/csh
# E12 INSTRUMENT PILOT — 1.3b, 120 bases, three layers, ranks 1/2/4.
#
# E12 is instrument validation, not a result. It asks whether a computed,
# TEXT-ABSENT program value can be identified and interchanged such that
# downstream computation transforms it. Causal state interchange is established
# method (DAS, Othello-GPT, variable binding in symbolic programs), so a pass
# licenses the semantic extension in docs/RESULTS.md (open items) — it is not
# itself a finding, and nothing here should be written up as one.
#
# Every stage is hard-gated and exits 2 when a prerequisite gate failed, so the
# `|| exit 1` chaining below stops at the first real problem rather than
# producing uninterpretable numbers downstream. To inspect a failure without
# fixing it first, re-run that stage by hand with
# `--override-gate 'reason'` — recorded permanently in gates.yaml.
#
#   setenv MODEL deepseek-coder-1.3b; jobs/store_pilot.csh
#
# Env overrides: MODEL, DTYPE, LAYERS, RANKS, NBASES.
# NOTE: the default MUST be set BEFORE sourcing jobs/common.csh, which
# sets MODEL to deepseek-coder-6.7b when it is unset. Sourcing first made
# this script's own default dead code, so a bare invocation silently ran
# the large model instead of the pilot one.
if (! $?MODEL) setenv MODEL deepseek-coder-1.3b
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?DTYPE) setenv DTYPE float16
if (! $?LAYERS) setenv LAYERS "6,12,18"
if (! $?RANKS) setenv RANKS "1,2,4"
if (! $?NBASES) setenv NBASES 120

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
echo "Read $OUT/e12_report.md. A verdict of INSTRUMENT VALIDATED means the"
echo "apparatus works — it is NOT a result. Next: docs/RESULTS.md (open items)."
