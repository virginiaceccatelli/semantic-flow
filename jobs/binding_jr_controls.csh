#!/bin/csh
# Re-run E13 stages 105-108 with the PUBLISHED stage-201 J/R lenses.
#
# DAS is fitted independently; the lenses are loaded only for the
# answer-direction controls. Existing H0-H2 artifacts are reused, so do not run
# this until those gates and
# results/workspace_lens/$MODEL/{j-lens,r-lens}/lens.pt exist.
#
# From tcsh/csh, inside a detached screen session:
#   screen -L -Logfile binding-jr-6.7b.log -dmS binding-jr-6.7b \
#       env MODEL=deepseek-coder-6.7b jobs/binding_jr_controls.csh
#
# ── Why stage 105 is in here now (2026-09-02) ────────────────────────────────
# The previous version started at stage 106 and reused the ceiling on disk. Two
# things went wrong because of that.
#
#  1. THE STRUCTURAL ZEROS. `run_grid` compared clean log-probs taken one prompt
#     at a time against patched log-probs taken in batches of 32. In reduced
#     precision the LM head's matmul is a different kernel at a different shape,
#     so the provable zeros came out at one or two ulps of the logits instead of
#     at zero: 0.25 in the bfloat16 interchange, 0.0156 in the float16 ceiling.
#     The edits themselves were exactly the zero vector. `run_grid` now computes
#     the clean baseline over the SAME batch, which fixes both stages — so both
#     have to be re-run, not just 106.
#
#  2. THE DTYPE SPLIT. The ceiling ran in float16 and the interchange in
#     bfloat16, so H4 and H5 were normalising against a ceiling measured at a
#     different precision. One DTYPE now drives both stages, and the run refuses
#     if they would disagree.
#
# Env overrides: MODEL, DTYPE, LAYERS, RANKS.

if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
# float16 throughout, matching jobs/binding_full.csh and the stage defaults.
# NOT bfloat16: it carries 3 fewer mantissa bits, which is why the artifact
# above was 4x larger in stage 106 than in stage 105.
if (! $?DTYPE) setenv DTYPE float16
# The layers the ceiling is measured at; stage 106 then intervenes at the single
# layer this stage selects on calibration. Matches the layer set behind the
# run being replaced, so the selection is reproduced rather than moved.
if (! $?LAYERS) setenv LAYERS "6,12,18"
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

echo "=== ${MODEL}: dtype ${DTYPE}, ceiling layers ${LAYERS}, ranks ${RANKS} ==="

echo "=== stage 105: whole-state ceiling, per arm (H3) — GPU ==="
# Re-run because the structural-zero fix is in `run_grid`, which this stage
# also uses, and because H3 now FAILS when the provable zeros do not hold.
$PYTHON scripts/105_binding_ceiling.py --model "$MODEL" \
    --layers "$LAYERS" --dtype "$DTYPE" --strict
if ($status != 0) exit 1

echo "=== stage 106: DAS plus published J/R-lens answer controls — GPU ==="
# No --layers: stage 106 uses the single layer stage 105 just chose on
# calibration, so the two stages cannot end up at different layers.
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

echo ""
echo "Read ${OUT}/e13_report.md and ${OUT}/interchange_panel.csv."
echo "CHECK FIRST: the structural zeros must be 0.00e+00 in both stages."
echo "Anything else and no claim from the run is licensed — read the"
echo "max_abs_edit_norm column to tell an arithmetic fault from a"
echo "forward-pass one."
