#!/bin/csh
# Stage 40 (E6). Requires stage-20 probes for $MODEL/core.
# For the layer sweep, set LAYER before invoking (each writes its own dir so
# parallel screen sessions don't clobber each other):
#   setenv MODEL deepseek-coder-6.7b; setenv LAYER 15; screen -dmS e6-6.7b-L15 jobs/leadtime.csh
source jobs/common.csh
if (! $?LAYER) then
    $PYTHON scripts/40_behavioral_leadtime.py --model "$MODEL" \
        --dataset data/synthetic/core.jsonl \
        --probes "results/probes/$MODEL/core"
else
    $PYTHON scripts/40_behavioral_leadtime.py --model "$MODEL" \
        --dataset data/synthetic/core.jsonl \
        --probes "results/probes/$MODEL/core" \
        --layer "$LAYER" \
        --output "results/leadtime/$MODEL/layer_$LAYER" \
        --no-tables
endif
