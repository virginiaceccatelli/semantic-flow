#!/bin/bash
# Shared environment setup for SGE jobs. Sourced by every jobs/*.sh.
source /share/apps/source_files/anaconda/conda-2022-5.source
conda activate semflow

export HF_HOME=$HOME/Scratch/hf_cache
export PYTHONPATH=$HOME/semantic-flow
cd $HOME/semantic-flow

MODEL=${MODEL:-deepseek-coder-6.7b}
