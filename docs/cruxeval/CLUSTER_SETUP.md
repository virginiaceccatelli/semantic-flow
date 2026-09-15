# Independent cluster environment

The system `/usr/bin/python3` on `rad` is Python 3.9; this project requires
Python >=3.10. `scripts/cruxeval/setup_cluster_venv.sh` uses uv to install
Python 3.11 into the repository's sibling `semflow-python/` directory, then
creates `semantic-flow/.venv`. The base interpreter must remain in place.
Neither depends on `micromamba-root` or the `vfield-mi` environment.

Run from the cluster repository (bash works even from a csh login):

```csh
bash scripts/cruxeval/setup_cluster_venv.sh
source jobs/common.csh
$PYTHON scripts/230_cruxeval_preflight.py --model deepseek-coder-6.7b --sample-size 100 --seed 42 --folds 5 --max-iter 20000 --out-root results/cruxeval_cluster
```

The setup script locates uv or installs it using system Python's user pip.
If that Python has no pip, install uv following its official instructions:
https://docs.astral.sh/uv/getting-started/installation/
The managed interpreter policy follows:
https://docs.astral.sh/uv/concepts/python-versions/#requiring-or-disabling-managed-python-versions

Setup installs the project/cluster dependencies and pytest, checks dependency
compatibility, runs the eight CruxEval unit tests, and writes the installed
versions and interpreter provenance under `results/cruxeval_cluster/environment/`.
It can download substantial PyTorch dependencies, but does not download model
weights or run activation extraction. CUDA availability is recorded; a CPU-only
preflight does not verify GPU compatibility for future stages.

The existing `hf-cache` stays in use for tokenizer/model assets and future
dataset downloads. Public CruxEval/tokenizer downloads need no access token.
Do not source the sibling `env.sh`: it uses bash syntax and configures
`vfield-mi` and a different Hugging Face cache. It belongs to that other project.

## Removing the old environment

First complete setup and the 100-program preflight, stop any jobs that use the
old environment, and check other environments' `sys.base_prefix` values. In
particular, inspect `vfield-mi/.venv312/bin/python` before removing a Python
installation that it might depend on. Updating `jobs/common.csh` affects new
jobs only; running jobs do not migrate.

Retire the old root by renaming it before deleting it. Then verify both the new
SemFlow environment and any other retained environments still run. Keep
`hf-cache`, `semflow-python`, `.uv-cache`, and `vfield-mi`. No setup script
automatically removes micromamba. Remove an unused micromamba executable only
after locating its actual path; the old job configuration references a path
whose existence has not been established.
