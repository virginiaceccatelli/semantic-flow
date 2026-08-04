#!/bin/csh
# Stage 62 (E10-3). Is control dependence verbalizable, or only decodable?
# Run jobs/jlens_validate.csh FIRST and confirm it passed.
#
#   setenv MODEL deepseek-coder-6.7b; jobs/jlens_controldep.csh
source jobs/common.csh
if (! $?DTYPE) setenv DTYPE float16
$PYTHON scripts/62_jlens_controldep.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl \
    --dtype "$DTYPE" \
    --lens-out "results/jlens/$MODEL/lenses"
