#!/bin/csh
# Shared environment setup for GPU-host jobs (no scheduler here — run each
# jobs/*.csh inside its own `screen` session). Sourced by every jobs/*.csh.

setenv PYTHON /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow/.venv/bin/python
setenv HF_HOME /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/hf-cache
setenv HF_DATASETS_CACHE $HF_HOME/datasets
setenv UV_PYTHON_INSTALL_DIR /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semflow-python
setenv UV_CACHE_DIR /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/.uv-cache

# Use the project venv for console scripts as well as explicit $PYTHON calls.
setenv PATH /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow/.venv/bin:$PATH

setenv PYTHONPATH /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow
cd /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow

if (! $?MODEL) setenv MODEL deepseek-coder-6.7b
