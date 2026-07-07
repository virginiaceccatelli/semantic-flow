#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-leadtime
#$ -l tmem=32G
#$ -l h_rt=08:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -cwd
# Stage 40 (E6). Requires stage-20 probes for $MODEL/core.
source jobs/common.sh
python scripts/40_behavioral_leadtime.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl \
    --probes "results/probes/$MODEL/core"
