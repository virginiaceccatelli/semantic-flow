#!/bin/csh
# Stage 121 over the three E15 shards (train + heldout + obfuscated ladder).
# The only GPU stage of the E15 track; 120 and 122-124 are CPU.
# Run inside a screen session:
#   screen -dmS sinkflow-extract-$MODEL env MODEL=deepseek-coder-6.7b jobs/sinkflow_extract.csh
source jobs/common.csh
$PYTHON scripts/121_sinkflow_extract.py --model "$MODEL"
