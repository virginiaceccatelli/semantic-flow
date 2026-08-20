#!/bin/csh
# Stages 128-130 — E15-D: the three follow-ups to the E15-C null. All three need
# a GPU; stage 131 is CPU and is run at the end anyway because it costs seconds.
#
#   128  full-vocabulary alignment      J2   ~15 min   one matmul per cell
#   129  the POSITIVE CONTROL           J3   ~1-3 h    3 forwards per prompt +
#                                                      one lens build
#   130  relevance by AST role          J4   ~30-90 m  one backward per
#                                                      (member, layer, target)
#   131  verdicts                       -    seconds
#
# STAGE 130 REFUSES ON starcoder2-3b and exits non-zero on purpose: LayerNorm
# plus a non-gated MLP means both homogenising LRP rules bind to nothing, so
# there is no relevance conservation to read. It records J4 as not applicable
# with the rule counts before exiting. That is why the line below is prefixed
# with `-` semantics in the Makefile and why this script does not chain with
# `&&` — the run must continue to stage 131 regardless.
#
# Run inside a screen session:
#   screen -dmS sinkflow-lens-6.7b env MODEL=deepseek-coder-6.7b jobs/sinkflow_lens.csh
source jobs/common.csh

$PYTHON scripts/128_sinkflow_align.py      --model "$MODEL"
$PYTHON scripts/129_sinkflow_positive.py   --model "$MODEL" --conditions all
$PYTHON scripts/130_sinkflow_relevance.py  --model "$MODEL"
$PYTHON scripts/131_sinkflow_lens_report.py --model "$MODEL"
