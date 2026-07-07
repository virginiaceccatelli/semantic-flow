#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-patching
#$ -l tmem=32G
#$ -l h_rt=08:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -cwd
# Stage 50 (E7). Requires stage-20 probes for $MODEL/core.
source jobs/common.sh
python scripts/50_causal_patching.py --model "$MODEL" \
    --pairs data/synthetic/minimal_pairs.jsonl \
    --probes "results/probes/$MODEL/core"
