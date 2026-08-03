#!/bin/csh
# Stage 10 over the obfuscation-ladder variants (E9 activations).
source jobs/common.csh
$PYTHON scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/synthetic/obfuscation.jsonl
