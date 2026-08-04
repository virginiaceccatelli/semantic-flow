#!/bin/csh
# Stage 61 (E10-2). The priority J-lens experiment: does the taint state live
# in a verbalizable workspace, and does that explain E6's 6.7b-only lead time?
# Run jobs/jlens_validate.csh FIRST and confirm it passed.
#
# PROBES — stage-20 dir for the probe comparison (default results/probes/$MODEL/core).
#          Same sklearn-version caveat as jobs/leadtime.csh applies.
#   setenv MODEL deepseek-coder-6.7b; jobs/jlens_taint.csh
source jobs/common.csh
if (! $?PROBES) setenv PROBES "results/probes/$MODEL/core"
if (! $?DTYPE) setenv DTYPE float16
$PYTHON scripts/61_jlens_taint.py --model "$MODEL" \
    --dataset data/synthetic/core.jsonl \
    --probes "$PROBES" \
    --dtype "$DTYPE" \
    --lens-out "results/jlens/$MODEL/lenses"
