#!/bin/csh
# Stage 40 (E6). Requires stage-20 taint_state probes for $MODEL.
#
# LAYER  — set for the layer sweep; each layer writes its own output dir.
# PROBES — override the probe dir (default results/probes/$MODEL/core). Needed
#          when the frozen checkpoints were pickled by a different sklearn than
#          the current env: refit taint_state into a side dir and point here.
#   setenv MODEL deepseek-coder-6.7b; setenv LAYER 15; jobs/leadtime.csh
source jobs/common.csh
if (! $?PROBES) setenv PROBES "results/probes/$MODEL/core"
if (! $?LAYER) then
    $PYTHON scripts/40_behavioral_leadtime.py --model "$MODEL" \
        --dataset data/synthetic/core.jsonl \
        --probes "$PROBES"
else
    $PYTHON scripts/40_behavioral_leadtime.py --model "$MODEL" \
        --dataset data/synthetic/core.jsonl \
        --probes "$PROBES" \
        --layer "$LAYER" \
        --output "results/leadtime/$MODEL/layer_$LAYER" \
        --no-tables
endif
