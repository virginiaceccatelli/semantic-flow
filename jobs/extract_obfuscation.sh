#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-extract-obfuscation
#$ -l tmem=32G
#$ -l h_rt=04:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -cwd
# Stage 10 over the obfuscation-ladder variants (E9 activations).
source jobs/common.sh
python scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/synthetic/obfuscation.jsonl
