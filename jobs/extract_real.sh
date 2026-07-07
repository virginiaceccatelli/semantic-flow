#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-extract-real
#$ -l tmem=32G
#$ -l h_rt=04:00:00
#$ -l gpu=true
#$ -pe gpu 1
#$ -j y
#$ -cwd
# Stage 10 over the real-code set (E8). Generate the jsonl locally first
# (stage 00 --real needs network) and rsync it to the cluster.
source jobs/common.sh
python scripts/10_extract_activations.py --model "$MODEL" \
    --dataset data/real/csn_python_200.jsonl
