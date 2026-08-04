#!/bin/csh
# Stage 60 (E10 Phase 0+1). The GATE for the whole J-lens track — stages 61/62
# are not interpretable until this passes. Needs no probes.
#
# DTYPE — set to float32 if check 0.3 reports non-finite gradients under fp16.
#   setenv MODEL deepseek-coder-6.7b; jobs/jlens_validate.csh
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?DTYPE) setenv DTYPE float16
$RUN scripts/60_jlens_validate.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl \
    --dtype "$DTYPE" \
    --lens-out "results/jlens/$MODEL/lenses"
