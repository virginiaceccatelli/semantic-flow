#!/bin/csh
# Stage 110 (E14 gate R). The GATE for the whole R-lens track — stage 111 is
# not interpretable until this passes. Needs no probes and no E13 pairs.
#
# DTYPE — set to float32 if R2 reports non-finite ratios under fp16. The LRP
# rules should HELP here (detaching the RMSNorm denominator and the SiLU
# factor removes the two largest sources of gradient blow-up), so fp16
# trouble that fp32 fixes is worth recording rather than just working around.
#
#   setenv MODEL deepseek-coder-6.7b; jobs/rlens_validate.csh
source jobs/common.csh
if (! $?RUN) setenv RUN "$MAMBA_EXE run -n semflow python"
if (! $?DTYPE) setenv DTYPE float16
$RUN scripts/110_rlens_validate.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl \
    --dtype "$DTYPE" \
    --n-r2 10 \
    --output "results/rlens/$MODEL/validate"
