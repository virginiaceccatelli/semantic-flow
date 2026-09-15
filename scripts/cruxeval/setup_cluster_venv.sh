#!/usr/bin/env bash
# Invoke with bash from any login shell, including csh/tcsh.
# Creates an independent environment; never removes an existing environment.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch_dir=$(dirname -- "$repo_dir")
cd -- "$repo_dir"

for required in requirements-cluster.txt requirements-cruxeval.txt scripts/230_cruxeval_preflight.py; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required: copy the CruxEval additions to this checkout first." >&2
        exit 1
    fi
done

if command -v uv >/dev/null 2>&1; then
    uv_bin=$(command -v uv)
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    echo 'Installing uv with the system Python; no model downloads occur here.'
    /usr/bin/python3 -m pip install --user uv
    uv_bin="$HOME/.local/bin/uv"
fi

# Keep this interpreter outside micromamba and outside the other project's env.
export UV_PYTHON_INSTALL_DIR="$scratch_dir/semflow-python"
export UV_CACHE_DIR="$scratch_dir/.uv-cache"
export HF_HOME="$scratch_dir/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
venv_dir="$repo_dir/.venv"

if [[ ! -e "$venv_dir" ]]; then
    "$uv_bin" python install 3.11
    "$uv_bin" venv --managed-python --python 3.11 --seed "$venv_dir"
fi

# Reuse an existing Python 3.11 venv when its base is already independent of
# micromamba/conda. uv may have installed that base under ~/.local/share/uv
# before UV_PYTHON_INSTALL_DIR was configured; that is still a valid standalone
# interpreter and survives removal of micromamba-root.
"$venv_dir/bin/python" - "$UV_PYTHON_INSTALL_DIR" "$scratch_dir/micromamba-root" <<'PY'
import pathlib
import sys
managed_root = pathlib.Path(sys.argv[1]).resolve()
micromamba_root = pathlib.Path(sys.argv[2]).resolve()
base = pathlib.Path(sys.base_prefix).resolve()
assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, found {sys.version}"
assert sys.prefix != sys.base_prefix, "Expected an isolated virtual environment"
assert not base.is_relative_to(micromamba_root), (
    f"Existing .venv still depends on micromamba: {base}. "
    "Rename .venv and rerun setup to build an independent replacement."
)
conda_marker = base / "conda-meta"
assert not conda_marker.exists(), (
    f"Existing .venv uses a conda base: {base}. "
    "Rename .venv and rerun setup to build an independent replacement."
)
location = "configured scratch uv root" if base.is_relative_to(managed_root) else "existing uv-managed root"
print(f"Interpreter: {sys.executable}\nIndependent base: {base}\nSource: {location}")
PY

"$uv_bin" pip install --python "$venv_dir/bin/python" \
    -r requirements-cluster.txt -r requirements-cruxeval.txt -e . pytest
"$uv_bin" pip check --python "$venv_dir/bin/python"
"$venv_dir/bin/python" -m pytest tests/test_cruxeval.py -q

mkdir -p results/cruxeval_cluster/environment
"$uv_bin" pip freeze --python "$venv_dir/bin/python" \
    > results/cruxeval_cluster/environment/requirements-frozen.txt
"$venv_dir/bin/python" - <<'PY' > results/cruxeval_cluster/environment/python.json
import json
import sys
import torch
print(json.dumps({"executable": sys.executable, "version": sys.version,
                  "base_prefix": sys.base_prefix, "torch": torch.__version__,
                  "cuda_available": torch.cuda.is_available()}, indent=2))
PY
echo 'Environment ready. No CruxEval experiment or micromamba removal was performed.'
echo 'In csh/tcsh: source jobs/common.csh'
