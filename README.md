# Tracing Semantic State in Code LLMs

**Extracting and Testing Latent Program-Graph Representations Across Context**

---

## Overview

This project asks whether code language models maintain an internal semantic state of a program as they process long contexts, and whether failures in that state can be detected before the model produces an incorrect prediction.

The core claim: modern code LLMs may preserve **lexical** surface structure well over long contexts while their internal representation of **semantic program structure** (def-use chains, control dependencies, taint paths) degrades earlier and more unevenly. By reconstructing latent semantic graphs from hidden activations and tracking their stability across layers, positions, and context lengths, we can identify where semantic state is lost, which relations are most fragile, and whether this loss predicts or causes downstream reasoning failures.

---

## Research Questions

1. **Representation Presence** — Do code LLMs internally represent semantic program relations (variable binding, def-use links, data dependencies, control dependencies, reachability)?

2. **Layer and Position Dynamics** — Where do these representations emerge? Are they strongest in early, middle, or late layers? Do they persist across long contexts?

3. **Lexical vs Semantic Stability** — Are lexical features (identifier names, nearby tokens) more stable than semantic relations ("this value flows into this sink")?

4. **Failure Precedence** — Does degradation of latent semantic structure appear *before* observable prediction failure (incorrect vulnerability classification, wrong code completion)?

5. **Encoding vs Use** — When a semantic relation is decodable from hidden states, is the model actually *using* it? Or is it present but causally disconnected from the final answer?

---

## Project Phases

### Phase 1 — Lexical and Local Semantic Probes
Establish whether models track identifier identity, declaration-use links, and local def-use edges. Critical experimental contrast: examples where lexical cues are misleading (renamed variables, shadowed identifiers, semantically equivalent rewrites).

**Output:** Baseline probe accuracy curves across layers and positions.

### Phase 2 — Graph-Like Semantic Structure Recovery
Train low-capacity probes to recover graph primitives from hidden states: pairwise adjacency (is there a semantic edge between span A and B?), node role (source/sink/sanitizer/guard), reachability. Compare reconstructed latent graph against ground-truth program-analysis graphs (AST, def-use, CFG, control-dependence).

**Output:** Per-layer semantic graph reconstruction quality.

### Phase 3 — Semantic Degradation Across Context
Apply probes across increasing context lengths, with relevant semantic relations placed nearby, hundreds of tokens away, across functions, and after distracting decoys. Measure how recovered structure changes across layer, token position, distance between related spans, and semantic complexity.

**Output:** Degradation maps; identification of which relation types fragment first.

### Phase 4 — Internal Degradation → Behavioral Failure
Pair internal measurements with model outputs on vulnerability detection, taint-flow judgment, and code completion tasks. Test whether probe-derived semantic instability predicts failure *earlier* than output-level uncertainty.

**Output:** Early-warning signal evaluation; correlation between internal state quality and downstream accuracy.

### Phase 5 — Causal Tests (Encoding vs Use)
Identify hidden states associated with semantic edges; patch activations from correct examples into incorrect ones; test whether restoring latent data-flow relations improves answer correctness. Classify relations as: *encoded and used* / *encoded but unused* / *not encoded* / *encoded transiently*.

**Output:** Causal attribution results; mechanistic interpretability contribution.

---

## Repository Structure

```
semantic-flow/
├── README.md
├── SETUP.md               # Detailed setup and GPU/CPU guidance
├── pyproject.toml
├── requirements.txt
│
├── src/
│   ├── models/
│   │   ├── loader.py          # Load code LLMs; model registry and config
│   │   └── hooks.py           # PyTorch forward hooks; activation cache
│   ├── graphs/
│   │   ├── ast_extractor.py   # AST extraction using Python's built-in ast
│   │   ├── dfg_extractor.py   # Def-use chains and data-flow graph
│   │   ├── cfg_extractor.py   # Control-flow graph
│   │   └── pdg_extractor.py   # Program-dependence graph (CFG + CDG)
│   ├── probes/
│   │   ├── base.py            # LinearProbe, MLPProbe, ProbeConfig, cross_validate_probe
│   │   ├── lexical.py         # Identifier identity and binding probes
│   │   ├── defuse.py          # Def-use edge probes (pairwise)
│   │   └── control.py         # Control-dependency and branch probes
│   ├── data/
│   │   ├── dataset.py         # CodeProbeDataset, ProbeExample, save/load utilities
│   │   └── generator.py       # Synthetic code with known semantic structure
│   ├── analysis/
│   │   ├── metrics.py         # Probe metrics; degradation statistics
│   │   └── visualization.py   # Layer curves, heatmaps, graph overlays
│   └── experiments/
│       ├── phase1_lexical.py
│       ├── phase2_graph.py
│       └── phase3_context.py
│
├── scripts/
│   ├── extract_activations.py # CLI: run model, dump hidden states
│   ├── train_probes.py        # CLI: train probes on saved activations
│   ├── evaluate_probes.py     # CLI: evaluate probes, generate reports
│   └── run_experiment.py      # CLI: end-to-end experiment runner
│
├── configs/
│   ├── models.yaml            # Model registry and layer configs
│   └── experiments.yaml       # Experiment hyperparameters
│
├── data/
│   ├── raw/                   # Downloaded datasets (gitignored)
│   ├── processed/             # Tokenized + graph-annotated samples
│   └── synthetic/             # Synthetically generated code
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_probe_development.ipynb
│   └── 03_degradation_analysis.ipynb
│
├── results/                   # Saved activations, probe checkpoints, plots
│
└── tests/
    ├── test_data.py
    ├── test_graphs.py
    └── test_probes.py
```

---

## Setup

### 1. Create environment

```bash
conda create -n semflow python=3.11
conda activate semflow
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

Or directly:

```bash
pip install -r requirements.txt
```

### 3. Add the Jupyter kernel

```bash
python -m ipykernel install --user --name semflow --display-name "semantic-flow"
```

### 4. Verify installation

```bash
pytest tests/ -v
```

Expected: 37 passed, 2 xfailed. The two xfails are CFG intra-function tests deferred to Phase 2.

---

## Quick Start

The steps below are grouped by what requires a GPU and what runs locally on CPU.

### CPU — Generate synthetic data

```bash
python -c "
from src.data.generator import SyntheticCodeGenerator
from src.data.dataset import save_jsonl

gen = SyntheticCodeGenerator(seed=42)
examples = gen.generate_batch() 
save_jsonl(examples, 'data/synthetic/phase1_binding.jsonl')
print(f'Generated {len(examples)} examples')
"
```

where generate_batch(n_binding=100, n_taint=100, n_shadow=50)

Binding: tests whether the model understands which definition a variable or name refers to, such as local variables, function parameters, imports, or names captured from an outer scope.

Taint: tests whether data from an untrusted or sensitive source flows into a dangerous sink without being sanitized. Typical examples involve user input reaching SQL execution, shell commands, file paths, or HTML output.

Shadow: tests name shadowing, where a new variable or parameter reuses an existing name and temporarily hides the earlier definition.

### CPU — Inspect graph extraction on a code snippet

```python
from src.graphs.dfg_extractor import DefUseExtractor
from src.graphs.cfg_extractor import CFGExtractor

code = """
x = get_input()
y = x + 1
return y
"""

dfg = DefUseExtractor().extract(code)
for edge in dfg.edges:
    print(edge)
```

### GPU — Extract activations from a code model

```bash
python scripts/extract_activations.py \
    --model deepseek-coder-1.3b \
    --dataset data/synthetic/phase1_binding.jsonl \
    --output results/activations/deepseek_phase1 \
    --layers all
```

### CPU — Train lexical binding probes on saved activations

```bash
python scripts/train_probes.py \
    --activations results/activations/deepseek_phase1 \
    --task binding \
    --probe linear \
    --output results/probes/deepseek_binding
```

### CPU — Evaluate and plot

```bash
python scripts/evaluate_probes.py \
    --probes results/probes/deepseek_binding \
    --output results/plots/deepseek_binding_layers.png
```

### GPU — Run a full phase end-to-end

```bash
python scripts/run_experiment.py \
    --config configs/experiments.yaml \
    --phase 1 \
    --model deepseek-coder-1.3b
```

---

## GPU Requirements

Hidden state extraction (model forward passes) requires a GPU for any practical dataset size. Probe training and analysis operate on saved activations and are CPU-only — these can run locally.

| Model | VRAM (float16) | Notes |
|---|---|---|
| deepseek-coder-1.3b | ~3 GB | Feasible on MPS (Apple Silicon) for small runs |
| starcoder2-3b | ~6 GB | Needs dedicated GPU |
| deepseek-coder-6.7b / starcoder2-7b / codellama-7b | ~14 GB | Needs A100 or equivalent |

**Recommended workflow:** run extraction on a GPU cluster (e.g. Compute Canada), save activations to disk, pull locally for probe training and analysis.

To use Apple Silicon MPS instead of CUDA for the 1.3b model:

```python
config = ModelConfig.from_registry("deepseek-coder-1.3b", device="mps")
```

---

## Target Models

| Model | Size | HuggingFace ID |
|---|---|---|
| DeepSeek-Coder | 1.3B | `deepseek-ai/deepseek-coder-1.3b-base` |
| DeepSeek-Coder | 6.7B | `deepseek-ai/deepseek-coder-6.7b-base` |
| StarCoder2 | 3B | `bigcode/starcoder2-3b` |
| StarCoder2 | 7B | `bigcode/starcoder2-7b` |
| CodeLlama | 7B | `codellama/CodeLlama-7b-hf` |

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Model loading and inference |
| `torch` | Activation extraction, probe training |
| `networkx` | Graph operations on program graphs |
| `scikit-learn` | Linear and MLP probes |
| `datasets` | HuggingFace dataset loading |
| `rich` / `typer` | CLI tooling |

Graph extraction currently uses Python's built-in `ast` module. `tree-sitter` is listed in `requirements.txt` for future multi-language support.

---

## Operationalizing "Semantic State"

A model's **semantic state** at layer `l`, position `p` is the set of program relations recoverable from `hidden[l, p]`.

| State type | Ground-truth source | Probe task |
|---|---|---|
| Lexical | Token type, identifier occurrence | Classification |
| Binding | Scope resolution (ast.NodeVisitor) | Pairwise match |
| Data-flow | Def-use chains (DFG) | Edge prediction |
| Control-flow | CFG basic blocks | Reachability |
| Control-dep | Control-dependence graph | Dependency prediction |
| Security | Taint paths (source→sink) | Path existence |

---

## Expected Contributions

1. A probing framework for extracting lexical, binding, def-use, and control-dependency relations from hidden states of decoder-only code LLMs.
2. Layer-wise and position-wise maps of where semantic information appears, persists, or degrades.
3. Evidence distinguishing surface lexical retention from true semantic relation tracking.
4. An early-warning signal for code reasoning failure based on internal semantic instability.
5. Initial causal tests showing whether recovered semantic representations are actually used.

---

## Research Trajectory

```
Phase 1: Can we decode lexical and binding relations?
    ↓
Phase 2: Can we decode local semantic edges (def-use)?
    ↓
Phase 3: Do these graphs degrade with long context?
    ↓
Phase 4: Does degradation predict model failure?
    ↓
Phase 5: Are the recovered relations causally used?
    ↓
Extension: Can semantic-state monitoring improve reliability or flag unsafe answers?
```

---

## Notes

- Start with Python; expand to Java once the framework is stable.
- Keep probes intentionally low-capacity (linear preferred over MLP where possible) to avoid probes memorizing surface features.
- Always run a **selectivity control**: train probes on shuffled labels to confirm probes are not exploiting spurious statistical regularities.
- Record exact random seeds and layer indices for reproducibility.
