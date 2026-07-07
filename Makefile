# semantic-flow pipeline targets.
#
#   make smoke                end-to-end tiny run on the local machine (MPS/CPU, 1.3b)
#   make data                 stage 00 full synthetic datasets
#   make data-real            stage 00 incl. CodeSearchNet sample (needs network)
#   make extract MODEL=...    stage 10 over core + context (GPU/MPS)
#   make probes MODEL=...     stage 20 (CPU)
#   make context MODEL=...    stage 30 (CPU)
#   make leadtime MODEL=...   stage 40 (GPU/MPS)
#   make patching MODEL=...   stage 50 (GPU/MPS)
#   make assets               stage 90 tables + figures (CPU)
#   make test                 pytest
#
# Cluster: submit jobs/*.sh with qsub instead of the GPU targets.

PY ?= python
MODEL ?= deepseek-coder-1.3b
ACT := results/activations/$(MODEL)
PROBES := results/probes/$(MODEL)/core

.PHONY: smoke data data-real extract probes context leadtime patching assets test

data:
	$(PY) scripts/00_generate_data.py --model $(MODEL)

data-real:
	$(PY) scripts/00_generate_data.py --model $(MODEL) --real

extract:
	$(PY) scripts/10_extract_activations.py --model $(MODEL) --dataset data/synthetic/core.jsonl
	$(PY) scripts/10_extract_activations.py --model $(MODEL) --dataset data/synthetic/context.jsonl --max-length 2048

probes:
	$(PY) scripts/20_run_probes.py --activations $(ACT)/core

context:
	$(PY) scripts/30_context_degradation.py --activations $(ACT)/context --probes $(PROBES)

leadtime:
	$(PY) scripts/40_behavioral_leadtime.py --model $(MODEL) --probes $(PROBES)

patching:
	$(PY) scripts/50_causal_patching.py --model $(MODEL) --probes $(PROBES)

assets:
	$(PY) scripts/90_make_paper_assets.py

test:
	$(PY) -m pytest tests/ -q

# ── smoke: tiny end-to-end run, asserts every stage produces its artifacts ────
SMOKE_DATA := data/smoke

smoke:
	$(PY) scripts/00_generate_data.py --model $(MODEL) --out-dir $(SMOKE_DATA) \
		--n-binding 12 --n-taint 12 --n-shadow 6 --n-context-bases 3 --n-pairs 5
	$(PY) scripts/10_extract_activations.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl --output results/smoke/act/core
	$(PY) scripts/10_extract_activations.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/context.jsonl --output results/smoke/act/context --max-length 2048
	$(PY) scripts/20_run_probes.py --activations results/smoke/act/core \
		--output results/smoke/probes --max-samples 4000 --cv-folds 3 --no-strict --no-tables
	$(PY) scripts/30_context_degradation.py --activations results/smoke/act/context \
		--probes results/smoke/probes --output results/smoke/context --no-tables
	$(PY) scripts/40_behavioral_leadtime.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl --probes results/smoke/probes \
		--output results/smoke/leadtime --n-examples 8 --no-tables
	$(PY) scripts/50_causal_patching.py --model $(MODEL) \
		--pairs $(SMOKE_DATA)/synthetic/minimal_pairs.jsonl --probes results/smoke/probes \
		--output results/smoke/patching --max-pairs 3 --no-tables
	$(PY) scripts/90_make_paper_assets.py
	@echo "--- smoke artifacts ---"
	@test -f results/smoke/probes/static_probes.csv
	@test -f results/smoke/context/context_degradation.csv
	@test -f results/smoke/leadtime/behavioral_leadtime.csv
	@test -f results/smoke/patching/causal_patching.csv
	@echo "SMOKE OK"
