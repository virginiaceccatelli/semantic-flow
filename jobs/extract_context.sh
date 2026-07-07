#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-extract-context
#$ -l tmem=32G
#$ -l h_rt=08:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -cwd
# Stage 10 over the context variants (E5 activations). Longer sequences.
source jobs/common.sh
python scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/synthetic/context.jsonl --max-length 2048
