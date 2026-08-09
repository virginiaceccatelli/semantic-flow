# Tracing Semantic State in Code LLMs

**Do code language models internally represent the *semantics* of a program —
which definition a name refers to, where a value flows, what guards its
execution, whether it is tainted — or only its surface text? And when that
internal state degrades, does the failure show up inside the model before it
shows up in its output?**

---

## 1. The question

A code LLM that only tracks surface form can still look competent: identifiers
usually keep their meaning and nearby tokens usually predict the right
continuation. The failures that matter — using a stale variable, trusting a
value that was never sanitized, completing code under the wrong branch — are
exactly the cases where **surface form and semantics come apart**: shadowed
names, long distances between definition and use, distractor code between
related statements.

We attack this in three parts, and each experiment belongs to one of them:

| | Question | Experiments |
|---|---|---|
| **Representation** | Are binding, data-flow, control-dependence, and taint relations *linearly decodable* from hidden states, beyond what the surface text predicts — and at which layers? | E1–E4, E8 |
| **Stability** | How does that decodability decay as context grows, distractors intervene, and the surface form is rewritten while semantics are held fixed — and does *semantic* structure break before *lexical* structure? | E5, E9 |
| **Consequence** | Is the internal state real: does its corruption *precede* behavioral failure (early warning), and does the model *causally use* it (patching changes the answer)? | E6, E7 |

Classic probing shows information is *present*; mechanistic interpretability
shows circuits are *used*; neither usually ties internal degradation to *when
the model actually fails*. This pipeline measures all three on the same
programs against the same ground truth.

## 2. How it works

```
generate programs        run the model once        read state back out       stress / race / intervene
with exact ground   →    save hidden states    →   train linear probes   →   on the frozen probes
truth (static AST)       at chosen layers          + honest controls         (E5 / E6 / E7)
```

We generate small Python programs whose semantic structure is **known exactly
by construction**, extract the ground-truth program graph by static analysis,
and map every def/use/guard/sink event to its exact token position. A frozen
code LLM reads each program once; we save the hidden states. Then we train
**low-capacity linear probes** to decode the relations, with controls that rule
out the obvious shortcuts — and finally we stress, race, and intervene on those
representations.

*Why synthetic programs, why linear probes, why one frozen pass:* per-token
labels have to be exact (static analysis on real code is approximate and its
errors become label noise); a linear readout is the standard operationalization
of "explicitly represented" — a stronger probe would measure the probe, not the
model; and one GPU pass decouples scarce GPU time from the frequent, CPU-only
probing work. Full rationale in [docs/METHODS.md](docs/METHODS.md).

## 3. The programs we study

Four program families, all single functions, all with ground truth attached
(`src/data/generator.py`). Every experiment draws its examples from these.

**Binding** — def-use chains, ~half under a branch (E1–E4). Ground truth: which
definition each occurrence binds to.

```python
def func():
    a = 42
    b = 17
    if a > 50:
        b = b + a      # which `b`, which `a`?
    return b
```

**Taint** — `source → propagation → (optional sanitizer) → sink`, with a
**per-line taint label** for the live value (E6). Ground truth: is the value
reaching the sink tainted?

```python
def func():
    x = input()        # tainted source
    v0 = x             # propagates taint
    safe = html.escape(v0)   # sanitizer — present only in the "safe" variant
    eval(safe)         # sink
```

**Shadow** — the same name carries two different bindings (E2's hard case).
Ground truth: the two `data` occurrences are *different* variables.

```python
def func(data):
    r = data * 3
    if r > 20:
        data = r - 5   # shadows the parameter
        r = data + 2   # uses the reassigned binding
    return r
```

**Minimal pairs** — clean/corrupted taint programs that tokenize
**identically except the sink argument** (E7), so activation patching can line
up positions one-to-one:

```python
# clean      ... s0 = shlex.quote(v1); os.system(s0)   ← sanitized value sinks
# corrupted  ... s0 = shlex.quote(v1); os.system(v1)   ← raw tainted value sinks
```

A fixed-seed sample of ~200 real, `ast`-parseable **CodeSearchNet** functions
(E8) checks that nothing we find is a generator artifact.

## 4. The pipeline

Numbered stages, each one CLI in `scripts/`, each writing a run manifest
(git sha, args, wall time) to `results/manifests/`. Extract on GPU once;
everything else is CPU and re-runnable.

```
00 generate data       CPU   programs + context variants + minimal pairs + obfuscation ladder + real sample
10 extract activations GPU   one forward pass per (model, dataset) → activation store
20 static probes       CPU   E1–E4 (+E8): grouped CV, controls, frozen probe checkpoints
30 context degradation CPU   E5: frozen probes re-evaluated on filler variants
31 obfuscation         CPU   E9: frozen probes re-evaluated on the semantics-verified obfuscation ladder
40 behavioral leadtime GPU   E6  — ARCHIVED (docs/LEGACY_RESULTS.md)
50 causal patching     GPU   E7: patch clean→corrupted activations, layer × position
60 J-lens validation   GPU   E10-0: the instrument check E11 reuses — a GATE
61/62 J-lens taint/cd  GPU   E10-2 / E10-3 — ARCHIVED
70 jspace pairs        CPU   E11: token-aligned binding counterfactuals, execution-verified
71 jspace lens         GPU   E11: frozen per-layer J-lens from a held-out corpus — a GATE
72 jspace readout      GPU   E11: paired counterfactual margin reversals
73 jspace swap         GPU   E11: the coordinate swap and its six controls
74 jspace report       CPU   E11: pre-registered go/no-go
80–89 store transitions      E12: INSTRUMENT VALIDATION, claims nothing — can a
                             computed, text-absent program value be interchanged
                             so downstream computation transforms it? Six gates
                             (G0–G5); each stage refuses to run on a failed one.
                             Design: docs/design/E12_PLAN.md · Commands:
                             docs/RUNBOOK_E12.md · What a pass licenses:
                             docs/design/E13_DIRECTIONS.md
90 paper assets        CPU   every table + figure, regenerated from CSVs alone
```

Stage status lives in `results/STATUS.yaml`; stage 90 reads it and skips
archived experiments unless run with `--include-archived`. Their raw CSVs,
figures and manifests are all preserved.

The activation store (stage 10) is the single interface between the model and
all analysis: one compressed `.npz` per example holding hidden states at the
probed layers, input ids, and verified token offsets. Stages 20/30/40/50 read
it (or, for E6/E7, run the model directly) and never re-extract. Commands,
outputs, and the GPU-host workflow: [docs/PIPELINE.md](docs/PIPELINE.md).

## 5. The experiments

Each experiment isolates one relation and pairs it with a **control that kills
the cheap way to score high**. Full specs, hypotheses, and decision rules:
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### Representation — is it decodable, and where? (E1–E4)

Probe hidden states for a relation; report **selectivity** (accuracy minus the
same probe retrained on shuffled labels) so a high number can't come from class
priors alone.

- **E1 · token type** — multiclass probe on single positions (keyword /
  identifier / literal / …). A sanity baseline: it must be near-ceiling at every
  layer, or the extraction/alignment machinery is broken.

- **E2 · binding — names vs meaning.** Pairwise probe: do two occurrences bind
  to the same definition? The control is the whole point — a
  `same_name_diff_binding` negative (the two `data`s in the shadow program)
  looks identical to a lexical probe. **Measures:** accuracy on that hard
  stratum vs on same-binding positives, per layer. The gap is the project's
  central "does it track meaning?" figure.

- **E3 · def-use edges — data flow, over distance.** Same pairwise setup for
  (definition, use) edges, but negatives are **distance-matched** (equal token
  gap, no real edge) so the probe can't win by "nearby ⇒ related." **Measures:**
  accuracy bucketed by def-use token distance (0–10 / 10–50 / 50–200 / 200+) —
  how far data-flow tracking reaches.

- **E4 · control dependence.** Does a statement execute under a given guard?
  Probe the (guard-expression state, statement state) pair; negatives are
  statements in the *same* program that are **not** guarded (e.g. after the `if`
  joins). Ground truth is join-point-exact from AST nesting. **Measures:**
  selectivity for the guard→statement relation by layer.

### Stability — does it survive context and surface change? (E5, E9)

- **E5 · context degradation.** Take the **frozen** E2/E3 probes and, without
  retraining, evaluate them on variants where filler is inserted between a
  definition and its use — five filler kinds at 0→1000 tokens, counted with the
  real tokenizer:

  | filler | what it tests |
  |---|---|
  | prose comment / dead code | pure distance (inert) |
  | lexical decoy | similar-looking but irrelevant names |
  | scope shadow | a nested scope reusing the name |
  | competing update | genuinely rebinds the variable — truth *changes* |

  **Measures:** frozen-probe accuracy vs filler size, per filler type × layer.
  Freezing is deliberate: retraining per condition would measure each
  condition's learnability, not the stability of the state the model actually
  keeps. For `competing_update` the ground truth is recomputed per variant, so
  it asks whether the model's state *updates*, not just whether it survives
  distance.

- **E9 · obfuscation robustness — same semantics, harder surface.** The
  transformation-based counterpart to E5: instead of pushing definition and use
  apart, rewrite the *whole program* while provably preserving what it computes.
  A Tigress-inspired ladder ([tigress.wtf](https://tigress.wtf) — C-only, so
  re-implemented natively for Python in `src/data/obfuscation.py`) applies
  cumulative levels of increasing difficulty:

  | level | transformation |
  |---|---|
  | 0 | normalize (formatting baseline) |
  | 1 | consistent renaming of every local — isolates lexical reliance |
  | 2 | + dead branches under opaque predicates (provably false, e.g. `v*v % 4 == 3`) |
  | 3 | + mixed boolean-arithmetic encoding (`a+b → (a^b)+((a&b)<<1)`) |
  | 4 | + control-flow flattening into a shuffled while/state-machine |

  Every variant is **executed and verified** observationally equivalent to its
  base; all levels of a base are kept or dropped together, so level curves
  compare identical program sets. The **frozen** E2/E3 probes are evaluated
  with ground truth rebuilt per variant. **Measures:** accuracy vs level, per
  task × layer. A collapse already at level 1 convicts the probe (and the
  model's accessible state) of lexical shortcuts; graceful decay across levels
  2–4 is evidence the relations are carried semantically.

### Consequence — is the state causally reusable? (E11; E7 preliminary)

- **E11 · J-space binding routing — the active direction.** Token-aligned
  counterfactual pairs where a one-token mutation of an inner definition's
  *name* flips which value a marked use selects, with both values present in
  both programs and the answer computed by five different downstream operations
  (affine, multiply/subtract, threshold, modulus, list indexing). Two questions,
  in order. **Readout:** does a frozen J-lens — built from a held-out generic
  corpus, never from these programs — rank the *bound* value above the
  distractor, and does that ranking *flip* with the counterfactual?
  **Intervention:** exchanging just the two value coordinates,
  `h ← h + V(swap(c) − c)`, leaves everything else in the state untouched; does
  the output then move toward the answer implied by the swapped-in value? The
  falsification test is that one edit must produce a *different* correct answer
  in each operation family — answer steering cannot do that. Six controls,
  including a Gram-matched random subspace and a provably-inert same-value
  no-op. Design and pre-registered go/no-go: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §2.

- **E7 · causal patching — preliminary.** On a minimal pair, run the corrupted
  program but overwrite its residual stream with the *clean* run's vector at one
  (layer, position) at a time. **Measures:** logit-diff recovery
  `(ld_patched − ld_corr)/(ld_clean − ld_corr)` per layer × position. It shows
  the causal locus of the decision migrating from the sink-argument token to the
  last token across the middle of the network. It does **not** isolate semantic
  use — the sink-argument position is the only place the two programs differ, so
  patching there transports the surface difference too. That gap is what E11's
  coordinate swap is built to close.

- **E6 · lead time — archived.** The original positive was measured on a
  constant responder, and the early-warning metric rewards unreliable readouts
  rather than anticipation. Data preserved;
  see [docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md).

### Generalization (E8)

- **E8 · real code.** Run stages 10+20 unchanged on the ~200 CodeSearchNet
  functions. **Measures:** synthetic-vs-real accuracy/selectivity side by side.
  A large gap would mean the probes fit generator artifacts; a small one means
  the findings transfer.

## 6. What every experiment is really defending against

Four rigor commitments run through all of them; they are why the numbers mean
what they claim to (details in [docs/METHODS.md](docs/METHODS.md)):

| Commitment | Shortcut it kills |
|---|---|
| **Grouped CV** — folds split by source program, never within | rows from one program share hidden vectors; random folds leak train into test |
| **Selectivity control** — identical probe on shuffled labels | accuracy from class priors / per-program regularities |
| **Negative strata**, reported separately | `same_name_diff_binding` kills "same string ⇒ same variable"; `distance_matched` kills "nearby ⇒ related" |
| **Verified token alignment** — AST spans → offsets checked against the source | string-matching a variable name silently mislabels shadows — the exact thing E2 measures |
| **Cross-validated ground truth** — def-use edges differentially tested against beniget (independent reaching-defs analysis); obfuscation variants execution-verified | labels that are wrong in the same way for train and test look like signal; this check already caught a real `b = b + a` mislabeling bug |

> One non-obvious hazard worth flagging: `AutoTokenizer` on transformers 5.x
> silently mis-tokenizes deepseek-coder (`def func` → `['de','ff','unc']`),
> which relabels *every* example without crashing. The loader refuses any
> tokenizer that fails an exact code round-trip. See
> [docs/METHODS.md](docs/METHODS.md) § Tokenizer integrity.

Every measurement lands in a tidy CSV in `results/tables/` (one row per
measurement, the data of record); stage 90 regenerates all figures and Markdown
tables from those CSVs alone, so `data → figure` is fully auditable.

## 7. Models

| Model | Role | Why |
|---|---|---|
| `deepseek-coder-1.3b-base` | development + smoke | runs on Apple-Silicon MPS; full pipeline in minutes |
| `deepseek-coder-6.7b-base` | main results | strong open code model; one cluster GPU in fp16 |
| `starcoder2-3b` | optional replication | different corpus/architecture family |

Base (non-instruct) models on purpose: the object of study is the
representation built during code pretraining, not chat behavior. Every stage
takes `--model`; probed layers per model live in `configs/models.yaml`.

## 8. Repository map

```
src/
  data/        generator (programs + ground truth), E11 counterfactual pairs, alignment, activation store
  graphs/      ast / def-use / CFG / control-dependence extraction (ground truth)
  models/      model + tokenizer loading (round-trip guard), hooks, patching, J-lens, J-space swap
  probes/      linear probe, grouped CV + controls, dataset builders (single source of truth)
  experiments/ E1–E4 static probes, E5 degradation, E7 patching, E9 obfuscation, E10 J-lens, E11 jspace_*
  analysis/    metrics, table rendering, figures, cluster bootstrap
scripts/       numbered stage CLIs (00–90)
jobs/          csh scripts per GPU stage (run under screen; no scheduler)
configs/       model registry + canonical experiment settings
results/       STATUS.yaml (what each experiment currently claims) + tables, figures, manifests
docs/          living reference: SETUP · PIPELINE · EXPERIMENTS · METHODS · RESULTS · LEGACY_RESULTS
  design/      design rationale, one file per track: JLENS_PLAN (E10/E11) · E12_PLAN (next direction)
tests/         253 CPU-only tests (alignment exactness, CV leakage, strata, pairs, obfuscation semantics,
               ground-truth cross-check, swap invariants, build/test separation, …)
```

## 9. Quickstart

```bash
conda create -n semflow python=3.11 -y && conda activate semflow
pip install -e ".[dev]"
make test                     # 253 CPU-only tests
make smoke                    # tiny end-to-end run on this machine (~15 min, MPS)

# full run (development model)
python scripts/00_generate_data.py --model deepseek-coder-1.3b --real
make extract probes context obfuscation patching assets MODEL=deepseek-coder-1.3b

# the active direction: E11 pilot (200 pairs, two operations, four layers)
make jspace-pilot             # then read results/jspace/deepseek-coder-1.3b/go_no_go.md
```

Setup and known pitfalls (the tokenizer!): [docs/SETUP.md](docs/SETUP.md) ·
Pipeline commands and cluster workflow: [docs/PIPELINE.md](docs/PIPELINE.md) ·
Experiment specs: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) ·
Methodology for the paper: [docs/METHODS.md](docs/METHODS.md) ·
Supported findings: [docs/RESULTS.md](docs/RESULTS.md) ·
Retired claims and why: [docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md) ·
Next direction, designed but not run: [docs/design/E12_PLAN.md](docs/design/E12_PLAN.md).

## 10. Intended contributions

1. Layer-resolved maps of where binding, data-flow, control-dependence, and
   taint relations are linearly decodable in code LLMs — with controls that
   separate semantic tracking from lexical shortcuts.
2. A quantitative account of how that structure degrades with context, by
   distractor type — distinguishing distance effects from genuine state updates —
   and under semantics-preserving obfuscation of increasing difficulty,
   separating lexical from semantic carriers of the same relations.
3. A falsifiable test of latent-before-behavioral failure (lead time): whether
   internal semantic state gives an early-warning signal.
4. Causal evidence (activation patching on aligned minimal pairs) for which
   decodable relations the model actually *uses*.
5. A fully scripted, manifest-tracked pipeline where every figure is
   regenerable from raw CSVs.

Out of scope here (future work): reasoning-trajectory probing on instruct
models, multi-language extension via tree-sitter.
