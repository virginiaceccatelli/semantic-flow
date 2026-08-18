#!/bin/csh
# Stage 125 — E15-C: build and freeze the three lenses and the discovered token
# set. The only GPU stage of the vocabulary track; 126 and 127 are CPU, because
# scoring a state against a lens already on disk is a matrix multiply.
#
# Cost is n_candidates x n_build x n_tprime backward passes per (layer, lens),
# so --max-candidates, --n-build and --layers are the knobs. Records J0.
#
# Run inside a screen session:
#   screen -dmS sinkflow-vocab-$MODEL env MODEL=deepseek-coder-6.7b jobs/sinkflow_vocab.csh
source jobs/common.csh
$PYTHON scripts/125_sinkflow_vocab_discover.py --model "$MODEL"
$PYTHON scripts/126_sinkflow_vocab_contrast.py --model "$MODEL"
$PYTHON scripts/127_sinkflow_vocab_report.py   --model "$MODEL"
