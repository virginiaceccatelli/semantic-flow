#!/bin/csh
# E17 — is variable binding VERBALISED?
#
# R10 (DAS) shows the models causally USE a binding representation. R11 (R-lens)
# shows the answer's relevance moves from the definition that goes out of scope to
# the one that comes into scope, over token-identical text. Both are read in the
# model's internal coordinates. E17 asks the question neither answers: does
# anything about "which definition this use refers to" surface in the model's own
# WORDS — and if it does, is that word read off the same structure the R-lens
# attributes the answer to?
#
#   150  lexicon + discovery           H7   ~10 min   full-vocabulary sweep on
#                                                     CALIBRATION bases only,
#                                                     frozen to disk
#   151  forced choice + contrast      H8   ~20 min   forward passes only
#   152  R-lens on the WORD            H9   ~35 min   one backward pass per
#                                                     (cell, layer, pole)
#   153  verdict + R10/R11 comparison   -   seconds   CPU, recomputes nothing
#
# Roughly one hour per model. Every stage requires E13's stage 101 to have
# recorded H0 for this model, and NOTHING ELSE — H1 is deliberately not required
# (it fails on deepseek-coder-1.3b at 0.809), because whether a model that
# answers the VALUE question at 0.809 can answer a WORD question is one of the
# things this track exists to measure.
#
# ** RUN THE TWO MODELS ONE AT A TIME. ** Same trap as jobs/binding_rlens.csh:
# `ModelLoader` loads with device_map="auto", so a 6.7b float32 load (~27 GB)
# that does not fit in the VRAM *currently free* is silently split and the
# offloaded tail comes back as meta placeholders. The tail is `model.norm` and
# `lm_head` — exactly what the lens cotangent reads — so the symptom is a
# meta-tensor error from the lens rather than an out-of-memory error from the
# loader. Every stage below refuses at load with that explanation instead of
# failing mid-loop, but the fix is still "do not co-reside".
#
# Stage 150 is the most VRAM-hungry of the four despite being the cheapest in
# time: it materialises the FULL unembedding as float32 (32256 x 4096 is about
# half a gigabyte on 6.7b) to rank every vocabulary token. It frees it before
# returning.
#
# If the card cannot hold 6.7b in float32, re-run with `env DTYPE=bfloat16`: the
# checkpoint is natively bfloat16, it halves the footprint, and unlike float16 it
# keeps float32's exponent range so the backward pass does not underflow. It
# costs precision, so read verbal/verbal_relevance_conservation.csv — the share
# reading is gated on it.
#
# STAGE 152 REFUSES ON starcoder2 and exits non-zero on purpose: LayerNorm plus a
# non-gated MLP means both homogenising LRP rules bind to nothing, so there is no
# relevance conservation to read. Stages 150, 151 and 153 are unaffected — the
# BEHAVIOURAL half of E17 needs no lens at all, and it is the half that answers
# the headline question. That is why this script does not chain with `&&`.
#
# Run inside a screen session, sequentially:
#   screen -dmS binding-verbal-1.3b env MODEL=deepseek-coder-1.3b jobs/binding_verbal.csh
#   # wait for that to finish (screen -ls), then:
#   screen -dmS binding-verbal-6.7b env MODEL=deepseek-coder-6.7b jobs/binding_verbal.csh
if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE float32
# The question style the relevance sweep reads. `scope` is the declared
# PRIMARY_STYLE: it names the construction ("assigned inside f") rather than a
# technical term, so a model with no word for shadowing can still answer it,
# which is what makes a null on it the strongest null available. Override only
# with a reason.
if (! $?STYLE) setenv STYLE scope

set OUT = "results/binding/${MODEL}"

echo "=== stage 150: lexicon + full-vocabulary discovery (H7) — GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null
# No --split: the stage takes the CALIBRATION bases itself and refuses if there
# are none, because discovery must not select tokens on the bases it will be
# evaluated on. No --layers: the registry's probe layers are the profile, and
# unlike the relevance stages the FINAL layer is meaningful here — the logit lens
# at the last layer is the model's actual output distribution.
$PYTHON scripts/150_binding_verbal_discover.py --model "$MODEL" --dtype "$DTYPE" --style "$STYLE"

echo ""
echo "=== stage 151: the forced choice + vocabulary contrast (H8) — GPU ==="
# All nine questions (four word styles x two variants, plus the value positive
# control). One run covers calib and test so stage 153's layer selection is held
# out without a second pass over the GPU.
$PYTHON scripts/151_binding_verbal_behaviour.py --model "$MODEL" --dtype "$DTYPE"

echo ""
echo "=== stage 152: the R-lens on the WORD (H9) — GPU ==="
# float32 by default: this reads a BACKWARD pass and fp16 gradients underflow on
# short sequences. One style per run — each costs a full backward sweep — and the
# pole-MARGIN reading that the headline rests on is derived from the two pole
# passes for free.
$PYTHON scripts/152_binding_verbal_relevance.py --model "$MODEL" --dtype "$DTYPE" --style "$STYLE"

echo ""
echo "=== stage 153: verdict + the R10/R11 comparison — CPU ==="
$PYTHON scripts/153_binding_verbal_report.py --model "$MODEL"

echo ""
echo "Read $OUT/e17_report.md, in the order it is written."
echo ""
echo "Section 2 (the forced choice) comes BEFORE section 4 (the attribution) on"
echo "purpose. A redistribution of a word's relevance means something different"
echo "depending on whether the model can produce that word at all, and reading"
echo "them the other way round is how a relevance shift gets reported as"
echo "verbalisation. Check the value row of table 2 first: it is the POSITIVE"
echo "control, and word styles at chance mean nothing unless it is at ceiling."
echo ""
echo "Then check, in this order:"
echo "  * says_inner_rate — 0.000 or 1.000 beside an accuracy of 0.500 is a model"
echo "    that always gives the same answer, not a model that is half right;"
echo "  * the per-variant rows — a style that only works in one option ORDER is"
echo "    reporting a position bias;"
echo "  * arm consistency — the value-independence control, and NOT the same"
echo "    control the arms provide in R11;"
echo "  * the positivity table in section 4 — it bounds the said/unsaid/fixed_*"
echo "    rows, though not the margin headline."
echo ""
echo "The verdict is observational in every branch. R10's DAS interchange is the"
echo "causal benchmark and the report computes no ratio between them."
