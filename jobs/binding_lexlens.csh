#!/bin/csh
# E18 — unprompted J-lens vocabulary readout of the E13 binding state.
#
# A pair is clear only when it reverses on >=80% of held-out bases in BOTH value
# arms, exceeds 99% of 500 Gram-matched directions in both arms, and repeats at
# adjacent tested layers. "No consistent pair" is an informative negative when
# the binding probe succeeds.
#
# From the repository root:
#   screen -dmS binding-lexlens env MODEL=deepseek-coder-6.7b jobs/binding_lexlens.csh
if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE float16

set OUT = "results/binding/${MODEL}"

echo "=== stage 160: unprompted J-lens word pairs + direction controls — GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null
$PYTHON scripts/160_binding_lexlens.py --model "$MODEL" --dtype "$DTYPE" \
    --n-seeds 5 --n-random-seeds 500 --n-corpus 200 --n-eval 200

echo "=== stage 161: per-word report — CPU ==="
$PYTHON scripts/161_binding_lexlens_report.py --model "$MODEL"

echo "Read $OUT/e18_report.md. Start with 'Which word contrasts are distinctly readable?'."
