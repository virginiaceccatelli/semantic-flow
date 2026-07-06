#!/bin/bash
#$ -S /bin/bash
#$ -N semflow-phases12       # job name
#$ -l tmem=32G               # memory
#$ -l h_rt=04:00:00          # wall time
#$ -l gpu=true               # request a GPU
#$ -pe gpu 1                 # 1 GPU
#$ -j y                      # merge stdout/stderr into one log
#$ -cwd                      # run from directory where you ran qsub

source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate semflow

export HF_HOME=$HOME/Scratch/hf_cache
export TRANSFORMERS_CACHE=$HOME/Scratch/hf_cache
export PYTHONPATH=$HOME/semantic-flow

cd $HOME/semantic-flow
python scripts/run_phases12.py

# COMMANDS
# qsub job.sh           submit
# qstat                 check status (qw = queued, r = running)
# qdel <job-ID>         cancel job
