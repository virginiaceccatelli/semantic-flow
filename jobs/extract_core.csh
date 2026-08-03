#!/bin/csh
# Stage 10 over the core dataset (E1-E4 activations).
# Run inside a screen session:
#   screen -dmS extract-core-$MODEL env MODEL=deepseek-coder-6.7b jobs/extract_core.csh
source jobs/common.csh
$PYTHON scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl
