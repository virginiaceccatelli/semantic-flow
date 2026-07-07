# Tracing Semantic State in Code LLMs

**Do code language models internally represent the *semantics* of a program —
which definition a name refers to, where a value flows, what guards its
execution, whether it is tainted — or only its surface text? And when that
internal state degrades, does the failure show up inside the model before it
shows up in its output?**

---

## 1. The question, and why it matters

A code LLM that merely tracks surface form can look competent on most code:
identifiers usually keep their meaning, and nearby tokens usually predict the
right continuation. The failures that matter — using a stale variable,
trusting a value that was never sanitized, completing code under the wrong
branch — are exactly the cases where surface form and semantics come apart:
shadowed names, long distances between definition and use, distractor code
between related statements.

Three questions structure the project:

1. **Representation** — are semantic relations (binding, def-use edges,
   control dependence, taint state) linearly decodable from hidden states,
   *beyond* what surface form predicts? At which layers?
2. **Stability** — how does that decodability degrade as context grows and
   distractors intervene, and does *semantic* structure degrade before
   *lexical* structure?
3. **Consequence** — is the internal state real in the two senses that
   matter: does its corruption *precede* behavioral failure (an early-warning
   signal), and is it *causally used* by the model (patching it changes the
   answer)?

The gap this fills: classic probing shows information is *present*;
mechanistic interpretability shows circuits are *used*; neither usually
connects internal degradation to *when and how the model actually fails*.
This pipeline measures all three on the same programs with the same ground
truth.

## 2. The approach in one paragraph

Generate small Python programs whose semantic structure is **known exactly**
(because we generate it), plus adversarial variants where surface form and
semantics disagree. Extract ground-truth program graphs by static analysis,
and map every def/use/guard/sink event to its exact token position. Run a
frozen code LLM over the programs once, saving hidden states at selected
layers. Train **low-capacity linear probes** to read semantic relations out
of those states, with controls that rule out every shortcut we could think
of. Then stress the representation (context filler), race it against behavior
(lead time), and intervene on it (activation patching).

## 3. What is constructed, and why that way

### A synthetic corpus with exact ground truth (`src/data/generator.py`)

Programs with variable-binding chains (optionally under branches), taint
flows (source → propagation → optional sanitizer → sink) with **per-line
taint labels**, and shadowing programs where the same name has two bindings.

*Why synthetic:* probing needs per-token labels. On real code, static
analysis is approximate and labels inherit its errors; on generated code the
ground truth is exact by construction, and confounds (name reuse, distances,
branch structure) are controllable rather than accidental. *Why not only
synthetic:* probes could fit generator artifacts — so a fixed-seed sample of
~200 real, `ast`-parseable CodeSearchNet functions (E8) checks that probe
accuracy transfers.

### Program graphs from the standard library `ast` (`src/graphs/`)

Reaching-definition def-use chains (each use resolved to its most recent
in-scope definition), a statement-level CFG that descends into function
bodies, and control-dependence read off guard nesting (join-point exact: a
statement after an `if` is *not* dependent on it).

*Why `ast` and not tree-sitter/a full analyzer:* the corpus is
single-function Python; Python's own parser is exact for it, dependency-free,
and auditable. Precision of ground truth matters more than language coverage
here.

### Verified token alignment (`src/data/alignment.py`)

Every AST event maps to token indices via character offsets that are
**validated against the source** (the tokenizer's offset mapping is used only
if it exactly tiles the string; otherwise offsets are rebuilt by incremental
decoding). The probing position for a span is its last covering token — under
causal attention, the first position that has seen the whole span.

*Why so paranoid:* subword vocabularies mean "the token for variable `x`"
is not well-defined by string matching — and we found that `AutoTokenizer`
on transformers 5.x silently mis-tokenizes deepseek-coder entirely
(`def func` → `['de','ff','unc']`). The loader therefore refuses any
tokenizer that fails an exact code round-trip. Wrong alignment doesn't crash;
it silently relabels every example — the worst failure mode for a paper.

### An activation store (`src/data/activation_store.py`)

One GPU pass per (model, dataset) saves per-example compressed arrays:
hidden states at the probed layers, input ids, and the verified offsets.
Everything downstream — probe training, stratified analyses, degradation
evaluation — is CPU-only and re-runnable without touching the model.

*Why decouple:* GPU time is the scarce resource (cluster queues); probe
iteration is the frequent activity. Extract once, analyze forever, and the
store is the single interface between them.

### Linear probes with adversarial controls (`src/probes/`)

Logistic regression only, on `h` (single-position tasks) or
`[h_i; h_j; h_i−h_j; |h_i−h_j|]` (pairwise tasks).

*Why linear:* the claim is about the *representation*, not about what a
powerful decoder could compute from it. A linear readout is the standard
operationalization of "explicitly represented."

Every probe run carries three defenses, because each kills a known way to
get a high number that means nothing:

| Defense | Shortcut it kills |
|---|---|
| **Grouped CV** (folds split by source program, never within) | pairs from the same program share hidden vectors — random folds leak train data into test |
| **Selectivity control** (identical probe retrained on labels shuffled within each program; across programs when labels are program-level) | accuracy from class priors and per-program regularities; we report `selectivity = accuracy − control` |
| **Negative strata**, reported separately | `same_name_diff_binding` kills the lexical shortcut ("same string ⇒ same variable"); `distance_matched` kills the positional shortcut ("nearby ⇒ related") |

### Context stress variants, evaluated with frozen probes (E5)

Filler blocks inserted between a definition and its use, in five types —
inert prose, dead code, lexically similar decoys, scope-shadowing nested
functions, and *competing updates* that genuinely rebind the variable — at
sizes 0→1000 tokens **counted with the actual tokenizer**.

*Why frozen probes:* retraining per condition would measure each condition's
learnability. Freezing the stage-2 probe and evaluating it under stress
measures the stability of the representation the model actually maintains.
*Why recompute ground truth per variant:* competing updates change the true
reaching definition — the interesting question is whether the model's state
updates with it.

### Two independent failure signals (E6)

For taint programs grown line by line: a **frozen taint-state probe** decodes
"is the live value tainted?" from the hidden state at each prefix (threshold
calibrated on a held-out split), while the **model itself** answers the same
question as a yes/no forced choice via continuation log-probabilities.
`t_latent` and `t_failure` are the first prefixes where each goes wrong;
their difference is the lead time.

*Why two mechanisms:* a linear readout of the residual stream vs the model's
own output head. They can disagree, which is precisely what makes "the
internal state degraded *before* the behavior failed" a falsifiable claim —
deriving both from the same logits (a design this replaced) makes it
tautological.

### Length-matched minimal pairs and positional patching (E7)

Clean/corrupted program pairs that tokenize to **identical sequences except
the sink argument** (clean sinks the sanitized variable, corrupted the raw
one — verified at generation). The corrupted run's residual stream is
overwritten with the clean run's vector at one (layer, position) at a time.

*Why length-matched:* patching requires positions to correspond; otherwise
you are comparing states of different tokens. *Why a position sweep with the
last token quarantined:* patching the readout position at a late layer
trivially forces the answer and proves nothing about semantics — recovery at
the *sink-argument* position at mid layers is the evidence that the encoded
relation is causally used.

## 4. What is computed, and why

| Quantity | Definition | What it answers |
|---|---|---|
| **Selectivity** per task × layer | held-out accuracy − shuffled-label control | is the relation encoded beyond dataset regularities, and where in depth |
| **Hard-stratum accuracy** | held-out accuracy on same-name-different-binding pairs | names vs meaning — the honest headline for binding |
| **Distance-bucket accuracy** | def-use accuracy for token gaps 0–10 / 10–50 / 50–200 / 200+ | how relation decoding decays with span distance |
| **Degradation curves** | frozen-probe accuracy vs filler size, per filler type × layer | which kinds of context break which relations, and how fast |
| **t_latent, t_failure, lead time** (+ bootstrap CI) | first prefix where probe / model goes wrong; their difference | does internal corruption precede behavioral failure — is there an early-warning signal |
| **Logit-diff recovery** per layer × position | `(ld_patched − ld_corr)/(ld_clean − ld_corr)`, `ld = logP(no) − logP(yes)` | is the encoded relation causally load-bearing for the answer |
| **Causal classes** | encoded_and_used / encoded_but_unused / not_encoded | reconciles probing with causation per layer |
| **Synthetic vs real gap** | same probe metrics on CodeSearchNet functions | do the findings survive contact with real code |

Every number lands in a tidy CSV (`results/tables/`, the raw data of record);
all figures and Markdown tables are regenerated from those CSVs alone.

## 5. Experiments at a glance

| ID | Question | Key design element |
|----|----------|--------------------|
| E1 | token-type baseline (pipeline sanity) | must be ~ceiling, or something is broken |
| E2 | binding: names vs meaning | same-name-different-binding hard negatives |
| E3 | def-use edges | distance strata + distance-matched negatives |
| E4 | control dependence | guard-expression anchors, join-point-exact truth |
| E5 | degradation under context | frozen probes, token-counted fillers, 5 filler types |
| E6 | latent vs behavioral failure | independent probe/behavior signals, calibrated threshold |
| E7 | encoding vs use | length-matched pairs, layer × position patching |
| E8 | real-code generalization | fixed-seed CodeSearchNet sample, unchanged pipeline |

Full specifications with hypotheses and decision rules: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 6. Models

| Model | Role | Why |
|---|---|---|
| `deepseek-coder-1.3b-base` | development + smoke | runs on Apple-Silicon MPS; full pipeline in minutes |
| `deepseek-coder-6.7b-base` | main results | strong open code model; fits one cluster GPU in fp16 |
| `starcoder2-3b` | optional replication | different corpus/architecture family |

Base (non-instruct) models on purpose: the object of study is the
representation built during pretraining on code, not chat behavior. All
stages take `--model`; probed layers per model live in `configs/models.yaml`.

## 7. Pipeline

```
00 generate data      CPU   corpus + context variants + minimal pairs + real sample
10 extract activations GPU  one pass per (model, dataset) → activation store
20 static probes      CPU   E1–E4 (+E8), grouped CV, frozen checkpoints
30 context degradation CPU  E5 over the context store
40 behavioral lead time GPU E6
50 causal patching    GPU   E7
90 paper assets       CPU   all tables + figures from CSVs
```

Each stage is one CLI in `scripts/`, writes a run manifest (git sha, args,
wall time) to `results/manifests/`, and is covered by `make` targets and SGE
job scripts (`jobs/`). Commands, artifacts, and the cluster workflow:
[docs/PIPELINE.md](docs/PIPELINE.md).

## 8. Repository map

```
src/
  data/        generator (programs + ground truth), alignment, activation store, loaders
  graphs/      ast / def-use / CFG / PDG extraction (ground truth)
  models/      model+tokenizer loading (with round-trip guard), forward hooks, patching
  probes/      linear probe, grouped CV + controls, dataset builders (single source of truth)
  experiments/ E1–E4 static probes, E5 degradation, E6 lead time, E7 patching
  analysis/    metrics, table rendering, figures
scripts/       numbered stage CLIs (00–90)
jobs/          SGE scripts per GPU stage
configs/       model registry + canonical experiment settings
docs/          PIPELINE / EXPERIMENTS / METHODS / RESULTS
tests/         60 CPU-only tests (alignment exactness, CV leakage, strata, pairs, …)
```

## 9. Quickstart

```bash
conda create -n semflow python=3.11 -y && conda activate semflow
pip install -e ".[dev]"
make test                     # 60 CPU-only tests
make smoke                    # tiny end-to-end run on this machine (~15 min, MPS)

# full run (development model)
python scripts/00_generate_data.py --model deepseek-coder-1.3b --real
make extract probes context leadtime patching assets MODEL=deepseek-coder-1.3b
```

Setup details and known pitfalls (tokenizer!): [SETUP.md](SETUP.md).
Methodological commitments, written for the paper's Methods section:
[docs/METHODS.md](docs/METHODS.md). Results index: [docs/RESULTS.md](docs/RESULTS.md).

## 10. Intended contributions

1. Layer-resolved maps of where binding, data-flow, control-dependence, and
   taint relations are linearly decodable in code LLMs — with controls that
   separate semantic tracking from lexical shortcuts.
2. A quantitative account of how that structure degrades with context, by
   distractor type — distinguishing distance effects from genuine state
   updates.
3. A falsifiable test of latent-before-behavioral failure (lead time), i.e.
   whether internal semantic state yields an early-warning signal.
4. Causal evidence (activation patching on aligned minimal pairs) for which
   of the decodable relations the model actually *uses*.
5. A fully scripted, manifest-tracked pipeline where every figure is
   regenerable from raw CSVs.

Future work (out of scope here): reasoning-trajectory probing on instruct
models, multi-language extension via tree-sitter.
