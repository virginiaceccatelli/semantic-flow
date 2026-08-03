# Setup

Local Mac (development, MPS) and a shared GPU host (main runs, no scheduler —
jobs run in `screen`). Always work inside the `semflow` conda env locally / the
`uq` micromamba env on the GPU host — the base env has a different Python and
packages.

## 1. Environment

```bash
brew install miniforge                 # if conda is missing (Apple Silicon)
conda create -n semflow python=3.11 -y
conda activate semflow
pip install -e ".[dev]"                # or: pip install -r requirements.txt
```

## 2. Verify

```bash
pytest tests/ -v          # all pass, CPU-only, no model download
python -c "import torch; print(torch.backends.mps.is_available())"   # True on M-series
```

## 3. Known pitfall: the tokenizer (IMPORTANT)

With transformers 5.x, `AutoTokenizer.from_pretrained("deepseek-ai/...")`
silently loads a broken slow tokenizer that destroys code
(`def func` → `['de','ff','unc']`, whitespace lost). **Never load tokenizers
directly** — use `src.models.loader.load_tokenizer(hf_id)`, which loads the
fast tokenizer and verifies an exact code round-trip, or `ModelLoader`, which
does so internally. All pipeline scripts already do this.

## 4. First run (smoke, ~5 min on MPS)

```bash
make smoke                 # tiny end-to-end pass: stages 00→10→20→30→40→50→90
```

Downloads deepseek-coder-1.3b (~2.7 GB) into `~/.cache/huggingface/hub/`
on first use. Then the real thing:

```bash
make data                  # stage 00 full synthetic datasets
make extract probes context leadtime patching assets   # full 1.3b pipeline
```

Stage-by-stage details, artifacts, and the cluster workflow: `docs/PIPELINE.md`.

## 5. GPU host (no scheduler — screen)

There is no `qsub`/SGE on this host. Job scripts are **csh** and the env is
**micromamba**, not conda. Every long-running stage goes in its own detached
`screen` session so it survives disconnects:

```csh
# once:
setenv MAMBA_ROOT_PREFIX /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/micromamba-root
setenv MAMBA_EXE /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/micromamba/bin/micromamba
eval `$MAMBA_EXE shell hook --shell csh`
micromamba create -n uq python=3.11 -y && micromamba activate uq
pip install -r requirements-cluster.txt
setenv HF_HOME /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/hf-cache
setenv HF_DATASETS_CACHE $HF_HOME/datasets
python -c "from src.models.loader import load_tokenizer; load_tokenizer('deepseek-ai/deepseek-coder-6.7b-base')"

# per run — one screen session per job:
cd /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow
screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b jobs/extract_core.csh
screen -ls                       # list running sessions
screen -r extract-core-6.7b      # attach; Ctrl-A D to detach again
```

`jobs/common.csh` centralizes `$PYTHON` (the `uq` env's interpreter,
`/scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/envs/uq/bin/python`), `HF_HOME`/
`HF_DATASETS_CACHE`, `MAMBA_ROOT_PREFIX`/`MAMBA_EXE`, and `PYTHONPATH`/`cd` into
the repo at `/scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow` — edit
paths there if the layout changes. Job scripts invoke `$PYTHON` directly rather
than a bare `python`; `env MODEL=... jobs/foo.csh` sets the variable the script
reads without needing `setenv` in the parent shell first.

If the cluster has no internet: run `make data-real` locally and rsync
`data/` (and the HF cache) up.

## 6. Long local jobs

Background shells die on session reset. Use nohup with the full env python:

```bash
nohup /opt/homebrew/Caskroom/miniforge/base/envs/semflow/bin/python \
    scripts/10_extract_activations.py --model deepseek-coder-1.3b \
    --dataset data/synthetic/core.jsonl > results/extract.log 2>&1 &
tail -f results/extract.log
```

## 7. Model sizes

| Model | Download | VRAM (fp16) | Where |
|---|---|---|---|
| deepseek-coder-1.3b | ~2.7 GB | ~3 GB | Mac MPS ok |
| deepseek-coder-6.7b | ~13 GB | ~14 GB | cluster GPU |
| starcoder2-3b | ~6 GB | ~6 GB | cluster GPU |
