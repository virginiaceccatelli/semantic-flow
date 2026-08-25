#!/bin/csh
# E16 — the OBSERVATIONAL R-lens readout of E13's binding counterfactual.
#
# E13 (R10) is the CAUSAL result on this corpus: a rank-1, magnitude-free DAS
# interchange at the use anchor makes the model emit the value the installed
# binding selects on 100% of held-out rows in BOTH arms. E16 asks the
# observational question beside it, with the R-lens E14 validated: when the same
# binding flips and exactly ONE token of the program changes, does the model's
# own attribution of its answer move from the definition that just went out of
# scope to the one that just came into scope?
#
# THESE ARE DIFFERENT QUANTITIES. DAS intervenes and reads the output; the
# R-lens reads a decomposition of the output and intervenes on nothing. Stage
# 141 prints them side by side and computes no ratio between them. A relevance
# shift is not weak causal evidence — it is evidence about a different thing.
#
#   140  relevance by token role      H6   minutes   one backward pass per
#                                                    (cell, layer, target mode)
#   141  verdict + DAS comparison     -    seconds   CPU, recomputes nothing
#
# Requires E13's stage 101 to have recorded H0 for this model. It does NOT
# require H1: that gate fails on deepseek-coder-1.3b (0.809 overall, cell
# ab_target 0.571) and requiring it would delete the smaller model from a
# question it can be asked. Behavioural correctness is joined from
# results/binding/$MODEL/behaviour.csv and reported as a stratifier.
#
# STAGE 140 REFUSES ON starcoder2-3b and exits non-zero on purpose: LayerNorm
# plus a non-gated MLP means both homogenising LRP rules bind to nothing, so
# there is no relevance conservation to read and the numbers would be raw
# autograd wearing the name relevance. It records H6 as not applicable with the
# rule counts before exiting. E13's DAS result on starcoder2 is unaffected. That
# is why this script does not chain with `&&` — 141 must still run.
#
# float32, not float16: this reads a BACKWARD pass, and fp16 gradients underflow
# on short sequences. The sequences are ~21 tokens, so float32 costs nothing here
# in time — but it does cost VRAM, and that is the one operational trap.
#
# ** RUN THE TWO MODELS ONE AT A TIME. ** `ModelLoader` loads with
# device_map="auto", so a 6.7b float32 load (~27 GB) that does not fit in the
# VRAM *currently free* is silently split, and the offloaded tail comes back as
# meta placeholders. The tail is `model.norm` and `lm_head` — exactly what the
# lens cotangent reads — so the symptom is a meta-tensor error from the lens
# rather than an out-of-memory error from the loader. Stage 140 now refuses at
# load with that explanation instead of failing mid-loop.
#
# If the card genuinely cannot hold 6.7b in float32, re-run that model with
# `env DTYPE=bfloat16`: the checkpoint is natively bfloat16, it halves the
# footprint, and unlike float16 it keeps float32's exponent range so the
# backward pass does not underflow. It costs precision, so read
# `relevance/relevance_conservation.csv` — the fraction reading is gated on it.
#
# Run inside a screen session, sequentially:
#   screen -dmS binding-rlens-1.3b env MODEL=deepseek-coder-1.3b jobs/binding_rlens.csh
#   # wait for that to finish (screen -ls), then:
#   screen -dmS binding-rlens-6.7b env MODEL=deepseek-coder-6.7b jobs/binding_rlens.csh
if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE float32

set OUT = "results/binding/${MODEL}"

echo "=== stage 140: relevance by token role (H6) — GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null
# No --layers: the registry's probe layers in [0, last) are the profile, and the
# reported layer is picked on CALIBRATION bases by stage 141. No --split either:
# one run covers calib and test, so the selection is held out without a second
# pass over the GPU.
$PYTHON scripts/140_binding_relevance.py --model "$MODEL" --dtype "$DTYPE"

echo "=== stage 141: verdict + the DAS comparison — CPU ==="
$PYTHON scripts/141_binding_relevance_report.py --model "$MODEL"

echo ""
echo "Read $OUT/e16_report.md."
echo "Read the arm-agreement table (table 3) BEFORE the headline: under `bound`"
echo "the scored token moves in OPPOSITE directions in the two arms, so a shift"
echo "that does not replicate across them is an output-token artifact, not a"
echo "binding effect. Tables 4 and 8 are the other two controls."
echo "The verdict is observational in every branch. E13's DAS result is the"
echo "causal benchmark and the report does not convert between them."
