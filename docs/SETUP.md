# Setup

Local Mac (development, MPS) and SGE cluster (main runs). Always work inside
the `semflow` conda env — the base env has a different Python and packages.

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

## 5. Cluster (SGE)

```bash
# once, on a network-enabled node:
conda create -n semflow python=3.11 -y && conda activate semflow
pip install -r requirements-cluster.txt
export HF_HOME=$HOME/Scratch/hf_cache
python -c "from src.models.loader import load_tokenizer; load_tokenizer('deepseek-ai/deepseek-coder-6.7b-base')"

# per run:
qsub -v MODEL=deepseek-coder-6.7b jobs/extract_core.sh
qstat                      # qw = queued, r = running
```

`jobs/common.sh` centralizes the conda source line, `HF_HOME`, and
`PYTHONPATH` — edit paths there if your cluster layout differs.

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
