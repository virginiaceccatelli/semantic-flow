#!/bin/csh
# Stage 10 over the context variants (E5 activations). Longer sequences.
source jobs/common.csh
$PYTHON scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/synthetic/context.jsonl --max-length 2048
