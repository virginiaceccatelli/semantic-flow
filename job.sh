#!/bin/bash
#$ -S /bin/bash
#$ -N my_experiment          # job name  
#$ -l tmem=16G               # memory   
#$ -l h_rt=24:00:00          # wall time 
#$ -l gpu=true               # request a GPU 
#$ -pe gpu 2                 # parallel environment with 2 GPUs (** multiple)
#$ -R y                      # GPU reservation for multiple GPUs (** multiple
#$ -j y                      # merge stdout/stderr
#$ -cwd                      # run from directory where you ran qsub
#$ -N JOBNAME                # FILL: job name


# activate environment 
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate semflow

export HF_HOME=/scratch/youruser/hf_cache
export TRANSFORMERS_CACHE=/scratch/youruser/hf_cache

python3 train.py


# COMMANDS
# qsub job.sh     submit
# qstat                 check status (qw = queued, r = running)
# qdel <job-ID>         kill it if needed

