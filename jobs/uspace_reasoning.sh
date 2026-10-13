#!/usr/bin/env bash
# Stage 292: upstream U-Space pipeline on CRUXEval/MBPP with a reasoning model.
# Run inside an allocated cluster job; this script does not request GPUs.
#
#   bash jobs/uspace_reasoning.sh setup                      # once, CPU, needs network (no vLLM)
#   TORCH_CUDA=cu128 bash jobs/uspace_reasoning.sh setup     # torch build for a newer driver (default cu126)
#   bash jobs/uspace_reasoning.sh download                   # once per model, CPU, needs network
#   MODEL=qwen4b SEEDS=42 bash jobs/uspace_reasoning.sh run  # GPU: steps 01-06 + controls
#   MODELS=qwen4b SEEDS=42 bash jobs/uspace_reasoning.sh tables
#   bash jobs/uspace_reasoning.sh analyze                    # CPU, repository venv
set -euo pipefail
cd "$(dirname "$0")/.."

UPY=${UPY:-.venv-uspace/bin/python}   # upstream pipeline environment (torch + transformers, no vLLM)
PYTHON=${PYTHON:-.venv/bin/python}    # this repository's environment (analysis)
HFCLI=${HFCLI:-.venv-uspace/bin/hf}
MODEL=${MODEL:-qwen4b}
MODELS=${MODELS:-$MODEL}
SEEDS=${SEEDS:-"41,42,43"}
BATCH=${BATCH:-8}                     # generation batch size (transformers; no vLLM)
CHECKOUT=third_party/u-space-pipeline
STAGE=scripts/292_uspace_reasoning.py

case "${1:-}" in
  setup)
    if [ ! -d "$CHECKOUT/.git" ]; then
      git clone https://github.com/s2labres/U-Space "$CHECKOUT"
    fi
    git -C "$CHECKOUT" checkout --quiet "$(cat third_party/u-space/COMMIT)"
    # torch 2.11 wheels exist for Python 3.10-3.13. Prefer the repository venv's
    # interpreter (3.11); BASE_PYTHON overrides.
    ok='import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'
    if [ -x "$UPY" ] && ! "$UPY" -c "$ok"; then
      echo "$UPY is Python $("$UPY" -V 2>&1); torch 2.11 needs 3.10-3.13." >&2
      echo "Remove it (rm -rf .venv-uspace) and rerun setup." >&2
      exit 1
    fi
    if [ ! -x "$UPY" ]; then
      base=${BASE_PYTHON:-}
      for candidate in .venv/bin/python python3.11 python3.12 python3.13 python3.10; do
        [ -n "$base" ] && break
        command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$ok" && base=$candidate
      done
      [ -n "$base" ] || { echo "No Python 3.10-3.13 found; set BASE_PYTHON" >&2; exit 1; }
      "$base" -c "$ok" || { echo "BASE_PYTHON $base is not 3.10-3.13" >&2; exit 1; }
      echo "Creating .venv-uspace with $base ($("$base" -V 2>&1))"
      "$base" -m venv .venv-uspace
    fi
    # No vLLM: step 02 runs with transformers (src/execution_uncertainty/generate.py).
    # torch 2.11.0 (upstream's pin) built for the node's driver: TORCH_CUDA=cu126 runs on
    # CUDA 12.6+ drivers; PyPI's default build is CUDA 13 (driver >= 580).
    TORCH_CUDA=${TORCH_CUDA:-cu126}
    "$UPY" -m pip install --no-cache-dir --upgrade pip
    "$UPY" -m pip install --no-cache-dir "torch==2.11.0" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    grep -v -E '^(vllm|torch)==' "$CHECKOUT/requirements.txt" > .venv-uspace/requirements-novllm.txt
    "$UPY" -m pip install --no-cache-dir -r .venv-uspace/requirements-novllm.txt accelerate scipy scikit-learn
    # Triton kernels for Qwen3.5's linear-attention layers; without them transformers
    # falls back to a much slower PyTorch implementation (same results).
    "$UPY" -m pip install --no-cache-dir flash-linear-attention ||
      echo "flash-linear-attention not installed: Qwen3.5 will use the slow PyTorch path" >&2
    "$UPY" -c 'import torch, transformers, accelerate
print("torch", torch.__version__, "CUDA build", torch.version.cuda, "| transformers", transformers.__version__)
print("GPU visible:", torch.cuda.is_available(), "(expected False on a CPU-only login node)")'
    "$UPY" -m pip check || true
    ;;
  download)
    # Weights at the pinned revision, plus the published J-lens for that model.
    set -- $("$UPY" -c 'import sys, yaml
m = yaml.safe_load(open("configs/uspace_reasoning.yaml"))["models"][sys.argv[1]]
l = m["lens"]
print(m["hf"], m["revision"], l.get("repo", "-"), l.get("file", "-"), l.get("revision", "-"))' "$MODEL")
    "$HFCLI" download "$1" --revision "$2"
    if [ "$3" != "-" ]; then
      "$HFCLI" download "$3" "$4" --revision "$5"
    else
      echo "Lens shipped in the upstream checkout: $CHECKOUT/data/lens/"
    fi
    ;;
  run)
    "$UPY" "$STAGE" step 01_sample_items
    seeds=${SEEDS//,/ }
    "$UPY" "$STAGE" generate --model "$MODEL" --seeds $seeds --batch-size "$BATCH"
    for step in 03_label 04_forward; do
      "$UPY" "$STAGE" step "$step" --model "$MODEL" --seeds $seeds
    done
    "$UPY" "$STAGE" step 05_build_uspace --model "$MODEL"
    "$UPY" "$STAGE" step 06_score --model "$MODEL" --seeds $seeds
    "$UPY" "$STAGE" controls --model "$MODEL"
    ;;
  tables)
    "$UPY" "$STAGE" step 07_tables --only-models "$MODELS" --only-seeds "$SEEDS"
    ;;
  analyze)
    "$PYTHON" "$STAGE" analyze
    ;;
  *) echo "Usage: bash jobs/uspace_reasoning.sh {setup|download|run|tables|analyze}" >&2; exit 2 ;;
esac
