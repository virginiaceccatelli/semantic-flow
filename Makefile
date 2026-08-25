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
#   ── E15, the security audit track: source → sink under obfuscation ──
#   make sinkflow-generate MODEL=..    stage 120 benchmark + validation — S0 (CPU)
#   make sinkflow-extract MODEL=...    stage 121 hidden states — S1 (GPU/MPS)
#   make sinkflow-probe MODEL=...      stage 122 clean readout + 4 arms — S2 (CPU)
#   make sinkflow-obf MODEL=...        stage 123 atomic+cumulative frozen eval — S3 (CPU)
#   make sinkflow-report MODEL=...     stage 124 gated report + figures (CPU)
#   make sinkflow MODEL=...            stages 120→124; each refuses on a failed gate
#   make sinkflow-smoke                the tiny 1.3b end-to-end check (A + B)
#
#   ── E15-C, the observational vocabulary-space contrast ──
#   make sinkflow-vocab-discover MODEL=..  stage 125 lenses + frozen tokens — J0 (GPU)
#   make sinkflow-vocab MODEL=...          stage 126 held-out contrast — J1 (CPU)
#   make sinkflow-vocab-report MODEL=...   stage 127 tables 6-10 + verdict (CPU)
#   make sinkflow-vocab-all MODEL=...      stages 125→127 in order
#   make sinkflow-vocab-smoke              the tiny 1.3b end-to-end check (C)
#
#   make sinkflow-align MODEL=...          stage 128 full-vocab direction — J2 (GPU)
#   make sinkflow-positive MODEL=...       stage 129 the POSITIVE CONTROL — J3 (GPU)
#   make sinkflow-relevance MODEL=...      stage 130 relevance by AST role — J4 (GPU)
#   make sinkflow-lens-report MODEL=...    stage 131 tables 13-20 + verdicts (CPU)
#   make sinkflow-lens-all MODEL=...       stages 128→131 in order
#   make sinkflow-lens-smoke               the tiny 1.3b end-to-end check (D)
#
#   ── E16, the OBSERVATIONAL R-lens readout of E13's binding counterfactual ──
#   make binding-relevance MODEL=...       stage 140 relevance by role — H6 (GPU)
#   make binding-relevance-report MODEL=.. stage 141 verdict + DAS comparison (CPU)
#   make binding-rlens MODEL=...           stages 140→141 in order
#   make binding-rlens-smoke               the tiny 1.3b end-to-end check
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
        binding-diagnose binding-pilot \
        sinkflow sinkflow-generate sinkflow-extract sinkflow-probe sinkflow-obf \
        sinkflow-report sinkflow-smoke sinkflow-vocab sinkflow-vocab-discover \
        sinkflow-vocab-report sinkflow-vocab-all sinkflow-vocab-smoke \
        sinkflow-align sinkflow-positive sinkflow-relevance \
        sinkflow-lens-report sinkflow-lens-all sinkflow-lens-smoke \
        binding-relevance binding-relevance-report binding-rlens binding-rlens-smoke

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

# ── E15 source→sink under obfuscation (stages 120→124; every stage is gated) ─
# Only stage 121 needs a GPU. Layers MUST include -1: the embedding control is
# one of the controls S2 refuses to run without.
sinkflow-generate:
	$(PY) scripts/120_sinkflow_generate.py --model $(MODEL)

sinkflow-extract:
	$(PY) scripts/121_sinkflow_extract.py --model $(MODEL)

sinkflow-probe:
	$(PY) scripts/122_sinkflow_probe.py --model $(MODEL)

sinkflow-obf:
	$(PY) scripts/123_sinkflow_obfuscation.py --model $(MODEL)

sinkflow-report:
	$(PY) scripts/124_sinkflow_report.py --model $(MODEL)

sinkflow: sinkflow-generate sinkflow-extract sinkflow-probe sinkflow-obf sinkflow-report

# 96 clean programs, 3 layers, separate data/results trees — minutes on a laptop.
sinkflow-smoke:
	$(PY) scripts/120_sinkflow_generate.py --model $(MODEL) \
		--out-dir $(SMOKE_DATA)/synthetic --output results/smoke/sinkflow \
		--n-seeds 4 --n-train-seeds 3
	$(PY) scripts/121_sinkflow_extract.py --model $(MODEL) \
		--data-dir $(SMOKE_DATA)/synthetic --output results/smoke/sinkflow \
		--activations results/smoke/act --layers=-1,0,11 --max-length 512
	$(PY) scripts/122_sinkflow_probe.py --model $(MODEL) \
		--activations results/smoke/act/sinkflow_train \
		--output results/smoke/sinkflow --cv-folds 3 --no-tables
	$(PY) scripts/123_sinkflow_obfuscation.py --model $(MODEL) \
		--activations results/smoke/act --output results/smoke/sinkflow \
		--data-dir $(SMOKE_DATA)/synthetic --no-tables
	$(PY) scripts/124_sinkflow_report.py --model $(MODEL) \
		--results results/smoke/sinkflow --figures results/smoke/figures
	@test -f results/smoke/sinkflow/sinkflow_clean.csv
	@test -f results/smoke/sinkflow/sinkflow_obfuscation.csv
	@test -f results/smoke/sinkflow/e15_report.md
	@echo "SINKFLOW SMOKE OK"

# ── E15-C the observational vocabulary-space contrast (125→127) ──────────────
# Only stage 125 needs a GPU: it builds the lens vectors and measures their
# fidelity. Stage 126 scores states against lens vectors already on disk, which
# is a matrix multiply, so it is CPU-only — and that is also why the freeze of
# the token set is a filesystem boundary rather than a promise.
sinkflow-vocab-discover:
	$(PY) scripts/125_sinkflow_vocab_discover.py --model $(MODEL)

sinkflow-vocab:
	$(PY) scripts/126_sinkflow_vocab_contrast.py --model $(MODEL)

sinkflow-vocab-report:
	$(PY) scripts/127_sinkflow_vocab_report.py --model $(MODEL)

sinkflow-vocab-all: sinkflow-vocab-discover sinkflow-vocab sinkflow-vocab-report

# One layer, 8 candidate tokens, 2 lens triples with one readout position each.
# Assumes `make sinkflow-smoke` has already written results/smoke/act.
#
# float32 and these tiny numbers are for MPS: a J/R-lens vector is one backward
# pass per (candidate, t'), fp16 gradients do not survive that path on MPS (they
# come back non-finite at every scale in the retry ladder), and fp32 backward on
# MPS is slow. The canonical runs are CUDA with the stage's defaults; this target
# exists to prove the pipeline runs end to end, not to produce a number.
sinkflow-vocab-smoke:
	$(PY) scripts/125_sinkflow_vocab_discover.py --model $(MODEL) \
		--activations results/smoke/act --output results/smoke/sinkflow \
		--layers=11 --dtype float32 --n-corpus 6 --n-build 2 --n-tprime 1 \
		--lens-max-length 192 --n-pool 4 --n-random 3 \
		--max-candidates 8 --top-k 2 --n-diagnostic 3 --n-conservation 1 \
		--n-invariance 1 --no-tables
	$(PY) scripts/126_sinkflow_vocab_contrast.py --model $(MODEL) \
		--activations results/smoke/act --output results/smoke/sinkflow \
		--n-permutations 200 --no-tables
	$(PY) scripts/127_sinkflow_vocab_report.py --model $(MODEL) \
		--results results/smoke/sinkflow
	@test -f results/smoke/sinkflow/vocab/vocab_discovery.json
	@test -f results/smoke/sinkflow/vocab/vocab_summary.csv
	@test -f results/smoke/sinkflow/vocab/e15c_report.md
	@echo "SINKFLOW VOCAB SMOKE OK"

# ── E15-D the three follow-ups to the E15-C null (128→131) ───────────────────
# 128 asks whether a shared direction exists over the WHOLE vocabulary, so its
# null cannot be blamed on a candidate pool. 129 is the POSITIVE CONTROL, and it
# is what turns E15-C's null from unfalsifiable into a claim: it runs the
# identical readout on a property the models demonstrably answer. 130 reads the
# R-lens as a conserving attribution rather than as a vocabulary projection,
# which needs no lexicalisation at all.
#
# All three need a GPU. 131 recomputes nothing and is CPU-only.
#
# 130 REFUSES on architectures where the homogenising LRP rules bind to nothing
# (starcoder2: LayerNorm + non-gated MLP). That is a fact about the
# architecture, not a failure; `sinkflow-lens-all` therefore tolerates it.
sinkflow-align:
	$(PY) scripts/128_sinkflow_align.py --model $(MODEL)

sinkflow-positive:
	$(PY) scripts/129_sinkflow_positive.py --model $(MODEL)

sinkflow-relevance:
	$(PY) scripts/130_sinkflow_relevance.py --model $(MODEL)

sinkflow-lens-report:
	$(PY) scripts/131_sinkflow_lens_report.py --model $(MODEL)

sinkflow-lens-all:
	$(PY) scripts/128_sinkflow_align.py --model $(MODEL)
	$(PY) scripts/129_sinkflow_positive.py --model $(MODEL)
	-$(PY) scripts/130_sinkflow_relevance.py --model $(MODEL)
	$(PY) scripts/131_sinkflow_lens_report.py --model $(MODEL)

# Two layers, six bases, tiny lens builds. float32 for the same MPS reason as
# `sinkflow-vocab-smoke`: fp16 gradients come back non-finite on that backend.
# Assumes `make sinkflow-smoke` and `make sinkflow-vocab-smoke` have run.
sinkflow-lens-smoke:
	$(PY) scripts/128_sinkflow_align.py --model $(MODEL) \
		--activations results/smoke/act --output results/smoke/sinkflow \
		--layers=-1,11 --dtype float32 --n-boot 200 --n-loadings 5 --no-tables
	$(PY) scripts/129_sinkflow_positive.py --model $(MODEL) \
		--data-dir $(SMOKE_DATA)/synthetic --output results/smoke/sinkflow \
		--layers=11,23 --dtype float32 --n-random 12 --n-corpus 6 --n-build 2 \
		--n-tprime 1 --lens-max-length 192 --n-permutations 200 --n-bases 6 \
		--no-tables
	$(PY) scripts/130_sinkflow_relevance.py --model $(MODEL) \
		--data-dir $(SMOKE_DATA)/synthetic --output results/smoke/sinkflow \
		--layers=11 --dtype float32 --n-permutations 200 --n-bases 6 --no-tables
	$(PY) scripts/131_sinkflow_lens_report.py --model $(MODEL) \
		--results results/smoke/sinkflow
	@test -f results/smoke/sinkflow/align/align_summary.csv
	@test -f results/smoke/sinkflow/positive/positive_summary.csv
	@test -f results/smoke/sinkflow/relevance/relevance_summary.csv

# ── E16 the observational R-lens readout of E13's binding pairs (140→141) ─────
# E13 (R10) is the CAUSAL result on this corpus: a rank-1 DAS interchange at the
# use anchor transports which definition is in scope. E16 asks the observational
# question beside it — when the same binding flips and exactly ONE token changes,
# does the model's own attribution of its answer move from the definition that
# went out of scope to the one that came in? The two are different quantities and
# stage 141's report never divides one by the other.
#
# Stage 140 needs a GPU and requires H0 only. H1 is deliberately NOT a
# prerequisite: it fails on deepseek-coder-1.3b, and requiring it would delete
# that model from a question it can be asked. Behavioural correctness is carried
# into every row as `correct_both` and reported as a stratifier instead.
#
# 140 REFUSES on architectures where the homogenising LRP rules bind to nothing
# (starcoder2: LayerNorm + non-gated MLP) and exits non-zero on purpose, so
# `binding-rlens` tolerates it with a `-` prefix and still runs the report.
binding-relevance:
	$(PY) scripts/140_binding_relevance.py --model $(MODEL)

binding-relevance-report:
	$(PY) scripts/141_binding_relevance_report.py --model $(MODEL)

binding-rlens:
	-$(PY) scripts/140_binding_relevance.py --model $(MODEL)
	$(PY) scripts/141_binding_relevance_report.py --model $(MODEL)

# One layer, six bases, float32 for the same MPS reason as `sinkflow-lens-smoke`:
# fp16 gradients come back non-finite on that backend. The gate override is what
# makes this runnable without E13's stages 100-101 having been run here.
binding-rlens-smoke:
	$(PY) scripts/140_binding_relevance.py --model $(MODEL) \
		--output results/smoke/binding/$(MODEL) --layers=6 --n-bases 6 \
		--dtype float32 --n-permutations 100 --n-boot 100 --n-determinism 2 \
		--no-tables --override-gate 'smoke run'
	$(PY) scripts/141_binding_relevance_report.py --model $(MODEL) \
		--results results/smoke/binding/$(MODEL)
	@test -f results/smoke/binding/$(MODEL)/relevance/relevance_summary.csv
	@test -f results/smoke/binding/$(MODEL)/e16_report.md
	@test -f results/smoke/sinkflow/e15d_report.md
	@echo "SINKFLOW LENS SMOKE OK"

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
