#!/bin/csh
# Shared environment setup for GPU-host jobs (no scheduler here — run each
# jobs/*.csh inside its own `screen` session). Sourced by every jobs/*.csh.

setenv PYTHON /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/envs/uq/bin/python
setenv HF_HOME /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/hf-cache
setenv HF_DATASETS_CACHE $HF_HOME/datasets
setenv MAMBA_ROOT_PREFIX /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/micromamba-root
setenv MAMBA_EXE /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/micromamba/bin/micromamba

# Put the uq env on PATH too (console scripts, not just $PYTHON).
eval `$MAMBA_EXE shell hook --shell csh`
micromamba activate uq

setenv PYTHONPATH /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow
cd /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow

if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
