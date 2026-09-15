#!/usr/bin/env bash
# Run from bash or invoke with bash from csh. No environment activation required.
# Usage: bash scripts/cruxeval/run_probes_cluster.sh PREFLIGHT RUN_DIR [SYNTHETIC_PAIRS]
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'Usage: bash scripts/cruxeval/run_probes_cluster.sh PREFLIGHT RUN_DIR [SYNTHETIC_PAIRS]' >&2
    exit 2
fi
preflight=$1
run_dir=$2
synthetic_pairs=${3:-1000}
python_bin=${PYTHON:-"$repo_dir/.venv/bin/python"}
export HF_HOME="${HF_HOME:-$(dirname "$repo_dir")/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
model=${MODEL:-deepseek-coder-6.7b}
seed=${SEED:-42}
iterations=${MAX_ITER:-20000}

# Repeated invocation skips only completed preparation artifacts. Every consumer
# rechecks their digests. Other stages certify/resume their exact configuration.
prepare_population() {
    local population=$1
    local destination="$run_dir/prepared_$population"
    if [[ -f "$destination/gates.json" ]]; then
        "$python_bin" -c 'import sys; from src.cruxeval.artifacts import checked_gate,sha256; g=checked_gate(sys.argv[1],"231_cruxeval_prepare"); assert g["args"]["preflight_sha256"] == sha256(sys.argv[2]+"/gates.json"); assert g["args"]["seed"] == int(sys.argv[3]); assert g["args"]["max_iter"] == int(sys.argv[4])' \
            "$destination" "$preflight" "$seed" "$iterations"
    else
        "$python_bin" scripts/231_cruxeval_prepare.py --preflight "$preflight" \
            --population "$population" --seed "$seed" --max-iter "$iterations" --output "$destination"
    fi
}

prepare_population all
prepare_population transfer_compatible
"$python_bin" scripts/232_cruxeval_extract.py --prepared "$run_dir/prepared_all" \
    --model "$model" --seed "$seed" --output "$run_dir/activations_real" --resume
"$python_bin" scripts/234_cruxeval_probes.py --design within --prepared "$run_dir/prepared_all" \
    --store "$run_dir/activations_real" --seed "$seed" --max-iter "$iterations" --output "$run_dir/A_within" --resume
"$python_bin" scripts/232_cruxeval_extract.py --synthetic-pairs "$synthetic_pairs" \
    --model "$model" --seed "$seed" --output "$run_dir/activations_synthetic" --resume
"$python_bin" scripts/233_cruxeval_synthetic.py --store "$run_dir/activations_synthetic" \
    --seed "$seed" --max-iter "$iterations" --output "$run_dir/synthetic_probes" --resume
"$python_bin" scripts/234_cruxeval_probes.py --design transfer --prepared "$run_dir/prepared_transfer_compatible" \
    --store "$run_dir/activations_real" --synthetic-probes "$run_dir/synthetic_probes" \
    --seed "$seed" --max-iter "$iterations" --output "$run_dir/B_transfer" --resume
echo "Completed. Reports: $run_dir/A_within/report.md and $run_dir/B_transfer/report.md"
