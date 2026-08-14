# semantic-flow pipeline targets.
#
#   make smoke                end-to-end tiny run on the local machine (MPS/CPU, 1.3b)
#   make data                 stage 00 full synthetic datasets
#   make data-real            stage 00 incl. CodeSearchNet sample (needs network)
#   make extract MODEL=...    stage 10 over core + context (GPU/MPS)
#   make probes MODEL=...     stage 20 (CPU)
#   make context MODEL=...    stage 30 (CPU)
#   make obfuscation MODEL=.. stage 31 (CPU)
#   make leadtime MODEL=...   stage 40 (GPU/MPS)
#   make patching MODEL=...   stage 50 (GPU/MPS)
#   make jlens-validate MODEL=...   stage 60 — E10 gate, must pass first (GPU/MPS)
#   make jlens-taint MODEL=...      stage 61 — E10-2, ARCHIVED (GPU/MPS)
#   make jlens-controldep MODEL=... stage 62 — E10-3, ARCHIVED (GPU/MPS)
#   make jlens MODEL=...            stages 60→62 in order (60 gates the rest)
#
#   ── E14, the R-lens representational track ──
#   make rlens-validate MODEL=...   stage 110 — E14 gate R, must pass (GPU/MPS)
#
#   ── E11, the active direction: J-space binding routing ──
#   make jspace-pairs MODEL=...     stage 70 counterfactual pairs (CPU)
#   make jspace-lens MODEL=...      stage 71 frozen lenses — GATE (GPU/MPS)
#   make jspace-readout MODEL=...   stage 72 readout (GPU/MPS)
#   make jspace-swap MODEL=...      stage 73 coordinate swap (GPU/MPS)
#   make jspace-report MODEL=...    stage 74 go/no-go (CPU)
#   make jspace-diagnose MODEL=...  stage 75 read a NO-GO (CPU, no re-run)
#   make jspace MODEL=...           stages 70→74 in order
#   make jspace-pilot               the 1.3b pilot exactly as pre-registered
#
#   ── E12, instrument validation (NOT a result): latent store transitions ──
#   make store-pairs MODEL=...      stage 80 text-absent counterfactuals (CPU)
#   make store-verify MODEL=...     stage 81 trace + interpreter — G0 (CPU)
#   make store-behaviour MODEL=...  stage 82 can the model solve them — G1 (GPU)
#   make store-extract MODEL=...    stage 83 cache anchor states (GPU)
#   make store-decode MODEL=...     stage 84 decodability — G2 (CPU)
#   make store-transition MODEL=... stage 85 natural transitions — G3 (CPU)
#   make store-ceiling MODEL=...    stage 86 whole-state interchange — G4 (GPU)
#   make store-interchange MODEL=.. stage 87 DAS low-rank + controls — G5 (GPU)
#   make store-report MODEL=...     stage 88 gated report (CPU)
#   make store-diagnose MODEL=...   stage 89 read a failed gate (CPU)
#   make store-sweep MODEL=...      stage 89 prompt/family sweep — the G1 fix (GPU)
#   make store MODEL=...            stages 80→88 in order; each refuses on a failed gate
#   make store-pilot                the cheap 1.3b instrument pilot
#
#   ── E13, the active direction: binding interchange (no arithmetic) ──
#   make binding-pairs MODEL=...      stage 100 binding x value factorial (CPU)
#   make binding-verify MODEL=...     stage 101 scope-aware reading — H0 (CPU)
#   make binding-behaviour MODEL=...  stage 102 returns the bound variable — H1 (GPU)
#   make binding-extract MODEL=...    stage 103 cache anchor states (GPU)
#   make binding-decode MODEL=...     stage 104 binding decodable — H2 (CPU)
#   make binding-ceiling MODEL=...    stage 105 whole-state, per arm — H3 (GPU)
#   make binding-interchange MODEL=.. stage 106 DAS + held-out arm — H4, H5 (GPU)
#   make binding-report MODEL=...     stage 107 gated report (CPU)
#   make binding-diagnose MODEL=...   stage 108 DID IT RUN WELL? (CPU, read-only)
#   make binding MODEL=...            stages 100→107; each refuses on a failed gate
#   make binding-pilot                the 1.3b pilot
#
#   make assets               stage 90 tables + figures, archived excluded (CPU)
#   make assets-all           stage 90 including archived experiments (CPU)
#   make test                 pytest
#
# GPU host (no scheduler): run jobs/*.csh in a screen session instead of the
# GPU targets, e.g. `screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b jobs/extract_core.csh`.

PY ?= python
MODEL ?= deepseek-coder-1.3b
ACT := results/activations/$(MODEL)
PROBES := results/probes/$(MODEL)/core

.PHONY: smoke data data-real extract probes context obfuscation leadtime patching \
        jlens jlens-validate jlens-taint jlens-controldep rlens-validate \
        jspace jspace-pairs jspace-lens jspace-readout jspace-swap jspace-report \
        jspace-diagnose jspace-pilot assets assets-all test \
        store store-pairs store-verify store-behaviour store-extract store-decode \
        store-transition store-ceiling store-interchange store-report store-pilot \
        store-diagnose store-sweep \
        binding binding-pairs binding-verify binding-behaviour binding-extract \
        binding-decode binding-ceiling binding-interchange binding-report \
        binding-diagnose binding-pilot

JSPACE_PAIRS := data/synthetic/jspace_pairs_$(MODEL).jsonl
STORE_LAYERS ?= 6,12,18
BINDING_LAYERS ?= 6,12,18
BINDING_RANKS ?= 1,2,4,8
STORE_RANKS ?= 1,2,4,8

data:
	$(PY) scripts/00_generate_data.py --model $(MODEL)

data-real:
	$(PY) scripts/00_generate_data.py --model $(MODEL) --real

extract:
	$(PY) scripts/10_extract_activations.py --model $(MODEL) --dataset data/synthetic/core.jsonl
	$(PY) scripts/10_extract_activations.py --model $(MODEL) --dataset data/synthetic/context.jsonl --max-length 2048
	$(PY) scripts/10_extract_activations.py --model $(MODEL) --dataset data/synthetic/obfuscation.jsonl

probes:
	$(PY) scripts/20_run_probes.py --activations $(ACT)/core

context:
	$(PY) scripts/30_context_degradation.py --activations $(ACT)/context --probes $(PROBES)

obfuscation:
	$(PY) scripts/31_obfuscation.py --activations $(ACT)/obfuscation --probes $(PROBES)

leadtime:
	$(PY) scripts/40_behavioral_leadtime.py --model $(MODEL) --probes $(PROBES)

patching:
	$(PY) scripts/50_causal_patching.py --model $(MODEL) --probes $(PROBES)

# ── E10 J-lens (stage 60 gates 61/62 — it exits non-zero if a check fails) ───
jlens-validate:
	$(PY) scripts/60_jlens_validate.py --model $(MODEL)

jlens-taint:
	$(PY) scripts/61_jlens_taint.py --model $(MODEL) --probes $(PROBES)

jlens-controldep:
	$(PY) scripts/62_jlens_controldep.py --model $(MODEL)

jlens: jlens-validate jlens-taint jlens-controldep

# ── E14 R-lens (stage 110 is the gate; it exits non-zero if a check fails) ───
rlens-validate:
	$(PY) scripts/110_rlens_validate.py --model $(MODEL)

# ── E11 J-space binding routing (stage 71 gates 72/73) ──────────────────────
jspace-pairs:
	$(PY) scripts/70_jspace_pairs.py --model $(MODEL)

jspace-lens:
	$(PY) scripts/71_jspace_lens.py --model $(MODEL) --pairs $(JSPACE_PAIRS)

jspace-readout:
	$(PY) scripts/72_jspace_readout.py --model $(MODEL) --pairs $(JSPACE_PAIRS)

jspace-swap:
	$(PY) scripts/73_jspace_swap.py --model $(MODEL) --pairs $(JSPACE_PAIRS)

jspace-report:
	$(PY) scripts/74_jspace_report.py --model $(MODEL)

# Read a NO-GO: is it the position, a dead readout, or a real dissociation?
jspace-diagnose:
	$(PY) scripts/75_jspace_diagnose.py --model $(MODEL)

jspace: jspace-pairs jspace-lens jspace-readout jspace-swap jspace-report

# The pre-registered pilot: 200 pairs, two operation families, four layers.
jspace-pilot:
	$(PY) scripts/70_jspace_pairs.py --model deepseek-coder-1.3b \
		--n-bases 100 --families affine,threshold
	$(PY) scripts/71_jspace_lens.py --model deepseek-coder-1.3b \
		--pairs data/synthetic/jspace_pairs_deepseek-coder-1.3b.jsonl \
		--layers 6,12,18,23 --n-build 150
	$(PY) scripts/72_jspace_readout.py --model deepseek-coder-1.3b \
		--pairs data/synthetic/jspace_pairs_deepseek-coder-1.3b.jsonl \
		--layers 6,12,18,23
	$(PY) scripts/73_jspace_swap.py --model deepseek-coder-1.3b \
		--pairs data/synthetic/jspace_pairs_deepseek-coder-1.3b.jsonl \
		--layers 6,12,18,23 --band-width 3
	$(PY) scripts/74_jspace_report.py --model deepseek-coder-1.3b

# ── E12 instrument validation (stages 80→88; every stage is gated) ──────────
# Each stage refuses to run (exit 2) unless its prerequisite gates passed. To
# run one anyway, add --override-gate 'reason' — it is recorded permanently.
store-pairs:
	$(PY) scripts/80_store_pairs.py --model $(MODEL)

store-verify:
	$(PY) scripts/81_store_verify.py --model $(MODEL)

store-behaviour:
	$(PY) scripts/82_store_behaviour.py --model $(MODEL)

store-extract:
	$(PY) scripts/83_store_extract.py --model $(MODEL) --layers $(STORE_LAYERS)

store-decode:
	$(PY) scripts/84_store_decode.py --model $(MODEL)

store-transition:
	$(PY) scripts/85_store_transition.py --model $(MODEL)

store-ceiling:
	$(PY) scripts/86_store_ceiling.py --model $(MODEL)

store-interchange:
	$(PY) scripts/87_store_interchange.py --model $(MODEL) --ranks $(STORE_RANKS)

store-report:
	$(PY) scripts/88_store_report.py --model $(MODEL)

# Read a failed gate: constant responder, wrong answer format, the model
# answering the intermediate, or a genuine capability limit. CPU, no re-run.
store-diagnose:
	$(PY) scripts/89_store_diagnose.py --model $(MODEL)

# Search for a prompt format and family set that elicits the task (GPU, ~2 min).
store-sweep:
	$(PY) scripts/89_store_diagnose.py --model $(MODEL) --sweep-prompts

store: store-pairs store-verify store-behaviour store-extract store-decode \
       store-transition store-ceiling store-interchange store-report

# The cheap instrument pilot: 120 bases, three layers, ranks 1/2/4.
store-pilot:
	$(MAKE) store MODEL=deepseek-coder-1.3b STORE_LAYERS=6,12,18 STORE_RANKS=1,2,4

# ── E13 binding interchange (stages 100→107; every stage is gated) ──────────
binding-pairs:
	$(PY) scripts/100_binding_pairs.py --model $(MODEL)

binding-verify:
	$(PY) scripts/101_binding_verify.py --model $(MODEL)

binding-behaviour:
	$(PY) scripts/102_binding_behaviour.py --model $(MODEL)

binding-extract:
	$(PY) scripts/103_binding_extract.py --model $(MODEL) --layers $(BINDING_LAYERS)

binding-decode:
	$(PY) scripts/104_binding_decode.py --model $(MODEL)

binding-ceiling:
	$(PY) scripts/105_binding_ceiling.py --model $(MODEL) --layers $(BINDING_LAYERS)

binding-interchange:
	$(PY) scripts/106_binding_interchange.py --model $(MODEL) \
		--layers $(BINDING_LAYERS) --ranks $(BINDING_RANKS)

binding-report:
	$(PY) scripts/107_binding_report.py --model $(MODEL)

# Separates "did the apparatus work" from "did the claim hold", and refuses to
# give a reading when the machinery is broken.
binding-diagnose:
	$(PY) scripts/108_binding_diagnose.py --model $(MODEL) --verbose

binding: binding-pairs binding-verify binding-behaviour binding-extract \
         binding-decode binding-ceiling binding-interchange binding-report \
         binding-diagnose

binding-pilot:
	$(MAKE) binding MODEL=deepseek-coder-1.3b BINDING_LAYERS=6,12,18 BINDING_RANKS=1,2,4

assets:
	$(PY) scripts/90_make_paper_assets.py

assets-all:
	$(PY) scripts/90_make_paper_assets.py --include-archived

test:
	$(PY) -m pytest tests/ -q

# ── smoke: tiny end-to-end run, asserts every stage produces its artifacts ────
SMOKE_DATA := data/smoke

smoke:
	$(PY) scripts/00_generate_data.py --model $(MODEL) --out-dir $(SMOKE_DATA) \
		--n-binding 12 --n-taint 12 --n-shadow 6 --n-context-bases 3 --n-pairs 5 --n-obf-bases 2
	$(PY) scripts/10_extract_activations.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl --output results/smoke/act/core
	$(PY) scripts/10_extract_activations.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/context.jsonl --output results/smoke/act/context --max-length 2048
	$(PY) scripts/10_extract_activations.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/obfuscation.jsonl --output results/smoke/act/obfuscation
	$(PY) scripts/20_run_probes.py --activations results/smoke/act/core \
		--output results/smoke/probes --max-samples 4000 --cv-folds 3 --no-strict --no-tables
	$(PY) scripts/30_context_degradation.py --activations results/smoke/act/context \
		--probes results/smoke/probes --output results/smoke/context --no-tables
	$(PY) scripts/31_obfuscation.py --activations results/smoke/act/obfuscation \
		--probes results/smoke/probes --output results/smoke/obfuscation --no-tables
	$(PY) scripts/40_behavioral_leadtime.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl --probes results/smoke/probes \
		--output results/smoke/leadtime --n-examples 8 --no-tables
	$(PY) scripts/50_causal_patching.py --model $(MODEL) \
		--pairs $(SMOKE_DATA)/synthetic/minimal_pairs.jsonl --probes results/smoke/probes \
		--output results/smoke/patching --max-pairs 3 --no-tables
	$(PY) scripts/60_jlens_validate.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl --output results/smoke/jlens/validate \
		--layers 0,11 --n-build 4 --n-eval 4 --n-sources 4 --n-taint 3 \
		--no-strict --no-tables
	$(PY) scripts/61_jlens_taint.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl --probes results/smoke/probes \
		--output results/smoke/jlens/taint --layers 0,11 --n-examples 6 --no-tables
	$(PY) scripts/62_jlens_controldep.py --model $(MODEL) \
		--dataset $(SMOKE_DATA)/synthetic/core.jsonl \
		--output results/smoke/jlens/controldep --layers 0,11 \
		--n-examples 8 --n-build 3 --no-tables
	$(PY) scripts/70_jspace_pairs.py --model $(MODEL) \
		--output $(SMOKE_DATA)/synthetic/jspace_pairs.jsonl \
		--n-bases 4 --families affine,threshold
	$(PY) scripts/71_jspace_lens.py --model $(MODEL) \
		--pairs $(SMOKE_DATA)/synthetic/jspace_pairs.jsonl \
		--output results/smoke/jspace/lens --lens-out results/smoke/jspace/lenses \
		--layers 0,11 --n-corpus 8 --n-build 6 --n-seeds 2 --n-eval 8 \
		--no-strict --no-tables
	$(PY) scripts/72_jspace_readout.py --model $(MODEL) \
		--pairs $(SMOKE_DATA)/synthetic/jspace_pairs.jsonl \
		--lenses results/smoke/jspace/lenses --output results/smoke/jspace/readout \
		--layers 0,11 --positions use,answer --n-boot 100 --no-tables
	$(PY) scripts/73_jspace_swap.py --model $(MODEL) \
		--pairs $(SMOKE_DATA)/synthetic/jspace_pairs.jsonl \
		--lenses results/smoke/jspace/lenses --output results/smoke/jspace/swap \
		--behaviour results/smoke/jspace/readout/jspace_behaviour.csv \
		--layers 0,11 --band-width 0 --n-boot 100 --no-tables
	$(PY) scripts/74_jspace_report.py --model $(MODEL) --results results/smoke/jspace
	$(PY) scripts/90_make_paper_assets.py
	@echo "--- smoke artifacts ---"
	@test -f results/smoke/probes/static_probes.csv
	@test -f results/smoke/context/context_degradation.csv
	@test -f results/smoke/obfuscation/obfuscation_robustness.csv
	@test -f results/smoke/leadtime/behavioral_leadtime.csv
	@test -f results/smoke/patching/causal_patching.csv
	@test -f results/smoke/jlens/validate/jlens_validation_checks.csv
	@test -f results/smoke/jlens/taint/jlens_taint_summary.csv
	@test -f results/smoke/jlens/controldep/jlens_controldep_summary.csv
	@test -f results/smoke/jspace/lens/jspace_lens_stability.csv
	@test -f results/smoke/jspace/readout/jspace_readout_summary.csv
	@test -f results/smoke/jspace/swap/jspace_swap_summary.csv
	@test -f results/smoke/jspace/go_no_go.yaml
	@echo "SMOKE OK"
