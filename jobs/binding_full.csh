#!/bin/csh
# E13 FULL RUN — binding interchange, 6.7b, 400 bases.
#
# E13 asks whether a low-rank, magnitude-free interchange at the site where a
# variable binding is resolved transports WHICH DEFINITION IS IN SCOPE, rather
# than a token or an answer direction. The identification is a 2x2: the same
# binding flip demands opposite token movements in the two value assignments,
# the alignment is fitted on arm `ab`, and the claim is read on arm `ba`.
#
# Unlike E12 there is no arithmetic anywhere — the model has to return a
# variable. E12 failed because it made two-step arithmetic the load-bearing
# capability; that is the mistake this design removes.
#
# Every stage is hard-gated and exits 2 when a prerequisite failed, so the
# `|| exit 1` chaining stops at the first real problem. To inspect a failure
# without fixing it, re-run that stage with `--override-gate 'reason'` —
# recorded permanently in gates.yaml and in every output row.
#
#   setenv MODEL deepseek-coder-6.7b; jobs/binding_full.csh
#
# Env overrides: MODEL, DTYPE, LAYERS, RANKS, NBASES.
# NOTE: the default MUST be set BEFORE sourcing jobs/common.csh, which sets
# MODEL to deepseek-coder-6.7b when it is unset.
if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?DTYPE) setenv DTYPE float16
if (! $?LAYERS) setenv LAYERS "8,12,16,20,24"
if (! $?RANKS) setenv RANKS "1,2,4,8,16"
if (! $?NBASES) setenv NBASES 400

set OUT = "results/binding/${MODEL}"

echo "=== stage 100: binding x value factorial ($NBASES bases) — CPU ==="
$RUN scripts/100_binding_pairs.py --model "$MODEL" --n-bases "$NBASES" || exit 1

echo "=== stage 101: independent scope-aware reading (H0) — CPU ==="
$RUN scripts/101_binding_verify.py --model "$MODEL" --strict || exit 1

echo "=== stage 102: does the model return the bound variable? (H1) — GPU ==="
$RUN scripts/102_binding_behaviour.py --model "$MODEL" --dtype "$DTYPE" --strict || exit 1

echo "=== stage 103: cache anchor states — GPU ==="
$RUN scripts/103_binding_extract.py --model "$MODEL" \
    --layers "$LAYERS" --dtype "$DTYPE" || exit 1

echo "=== stage 104: is the binding decodable at the use? (H2) — CPU ==="
$RUN scripts/104_binding_decode.py --model "$MODEL" --strict || exit 1

echo "=== stage 105: whole-state interchange ceiling, per arm (H3) — GPU ==="
$RUN scripts/105_binding_ceiling.py --model "$MODEL" \
    --layers "$LAYERS" --dtype "$DTYPE" --strict || exit 1

echo "=== stage 106: DAS interchange + held-out-arm falsification (H4, H5) — GPU ==="
$RUN scripts/106_binding_interchange.py --model "$MODEL" \
    --layers "$LAYERS" --ranks "$RANKS" --dtype "$DTYPE" || exit 1

echo "=== stage 107: gated report — CPU ==="
$RUN scripts/107_binding_report.py --model "$MODEL"

echo ""
echo "Read $OUT/e13_report.md."
echo "H4 without H5 is E11 again: an effect on the training arm alone cannot"
echo "separate a binding subspace from an answer direction. Read the"
echo "answer_direction rows before drawing any conclusion from H5."
