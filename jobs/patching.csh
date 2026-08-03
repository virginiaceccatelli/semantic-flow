#!/bin/csh
# Stage 50 (E7). Requires stage-20 probes for $MODEL/core.
source jobs/common.csh
$PYTHON scripts/50_causal_patching.py --model "$MODEL" \
    --pairs data/synthetic/minimal_pairs.jsonl \
    --probes "results/probes/$MODEL/core"
