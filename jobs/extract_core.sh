#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-extract-core
#$ -l tmem=32G
#$ -l h_rt=04:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -cwd
# Stage 10 over the core dataset (E1-E4 activations).
# Usage: qsub -v MODEL=deepseek-coder-6.7b jobs/extract_core.sh
source jobs/common.sh
python scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl
