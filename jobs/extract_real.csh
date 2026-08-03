#!/bin/csh
# Stage 10 over the real-code set (E8). csn_python_200.jsonl is committed to
# the repo already (generated locally, needs network) — just git pull.
source jobs/common.csh
$PYTHON scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/real/csn_python_200.jsonl
