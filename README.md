# Semantic Flow

**Do code language models build representations of program *semantics* — which
definition a name refers to, where a value flows, whether a dangerous argument
is attacker-controlled — that are distinct from lexical and syntactic
regularity? And does the model's own computation *causally use* them?**

---

## 1. The problem, and the construction that makes it tractable

A code model that tracks only surface form can look competent. Identifiers
usually keep their meaning, related statements are usually written near each
other, and indentation usually exposes control structure. So a probe that
recovers binding or data flow from hidden states may be recovering the model's
computation — or may be recovering the text, which the hidden states also
contain.

The usual response is to *estimate* how much a shortcut could explain and
subtract it. Code allows something better: **make the shortcut carry no
information at all.** Two programs differing in one character:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7          ← one character
    return x               return x
#   → 3                    → 7
```

The token we ask about is at the same index in both. The words around it are
the same. The distance to everything is unchanged. Only the answer flips. A
predictor with access to nearby tokens and distances is therefore right
**exactly 0.500 of the time — by construction, not by estimate.** Everything
above that floor is something the model computed.

That construction is the project's central asset, and it is what lets a
representational claim be *pinned* rather than argued.

## 2. Two axes: program structure, and evidential strength

Every experiment in this repository is located by two coordinates.

### Axis 1 — what is asked about: relations from the code property graph

Ground truth is never taken from the model, from a heuristic, or from a
labeller. It is computed from the program's own structure, by the extractors in
`src/graphs/`, which build the standard layers of a **code property graph**:

| layer | module | what it provides | relation it grounds |
|---|---|---|---|
| **AST** | `ast_extractor.py` | character-offset-aware syntax nodes | token alignment for every label |
| **CFG** | `cfg_extractor.py` | statement-level control flow, join points | control dependence — *measured and archived*, see below |
| **DFG** | `dfg_extractor.py` | definitions, uses, reaching-definition edges | **binding (R1)**, **def–use (R2)** |
| **PDG** | `pdg_extractor.py` | union of def–use and control-dependence edges | **source→sink taint paths (R5)** |

The graph is what makes the labels *exact*, and exactness is what makes the
floor argument possible: an approximate label becomes label noise, and label
noise becomes the finding.

The criterion also **excludes** a CPG relation, which is what makes it a
criterion rather than a slogan. Control dependence is decodable at ceiling
(AUC 0.999) — but a model-free reader of token windows and indentation already
scores **0.927** on it, because a statement's guard is usually its nearest
enclosing `if`. It is therefore not reported as a result; the numbers and the
reasoning are in [docs/ARCHIVE.md §4.3](docs/ARCHIVE.md#43-control-dependence).

### Axis 2 — how hard the question is asked: four instruments

The instruments differ in what they are *entitled* to conclude, and conflating
them is the source of every claim this project has withdrawn.

| # | instrument | licenses | does **not** license |
|---|---|---|---|
| 1 | **Linear probe** against a construction-pinned floor | the relation is linearly *present* in the state | that the model uses it |
| 2 | **Frozen probe** transferred across meaning-preserving rewrites | what the representation is *made of*; which rewrite destroys it | that the model lost the information (a probe can fail where a model does not) |
| 3 | **Output-basis readouts** — a vocabulary projection, and the **R-lens** as a conserving attribution | whether the distinction lives in the model's *own output coordinates*, whether any *word* carries it, and where relevance is routed | anything causal; a projection is not an edit |
| 4 | **DAS interchange** — a learned, magnitude-free rank-1 edit | that the model's downstream computation *reads* the subspace | generality beyond the site, layer, model and construction tested |

Instruments 1–2 are correlational by nature. Instrument 3 is still
observational but asks about *format* rather than mere decodability.
Instrument 4 is the only causal one.

## 3. What was found

**The relation is there, and it is built with depth.** Binding is *absent* at
the input (exactly 0.500 at the embedding layer and at the model-free surface
baseline), constructed within the first few transformer blocks, peaks near
**0.984** in the middle of the network, and is partly shed before the output.
Replicated at 1.3B and 6.7B at matched relative depth. Def–use follows the same
profile with mild, honest decay by distance.

**It breaks on difficulty, not on distance or spelling.** A thousand tokens of
inert comments costs almost nothing (0.921); the same length of filler that
*reuses the tracked names* drives binding to chance. Renaming every identifier
leaves middle layers at 0.85–0.90 while pushing the embedding layer *below*
chance. Control-flow flattening is the real limit.

**The same boundary holds for a property an auditor would actually ask for.**
E15 reads "is the value at this `os.system` / `cursor.execute` / `eval`
argument source-derived?" at **1.000** on held-out programs, over *two* measured
chance floors. Applying each rewrite on its own, in three models: opaque dead
branches and arithmetic rewriting cost **exactly nothing**, renaming costs
0.01–0.12, and **control-flow flattening alone costs 0.31–0.34** — within 0.03
of what the entire four-transformation composition costs. Composition adds no
interaction distinguishable from measured draw noise. **One transformation
carries the whole failure.**

**Decodable and verbalised come apart.** Differencing each matched pair over the
**whole 32k-token vocabulary** finds a direction that **72 of 72 held-out pairs
project positively onto, in every model**, over a token-identity floor of
*exactly zero* — appearing at a quarter of the way up the stack and collapsing
under flattening alone. Its top-loading tokens are meaningless fragments, so the
distinction is **output-aligned, distributed, and carried by no word for it**.
That is not an instrument failure: at the cell where the same readout fires on a
property these models *do* express (0.85–0.94, tracking the model's own answer
margin), the security lexicon separates the pair at **0.347 / 0.389** —
significantly in the *wrong* direction.

**And for binding, the causal question has an affirmative answer.** A rank-1,
magnitude-free **DAS interchange** at the site where the binding is resolved
transports *which definition is in scope* into both value assignments of a 2×2,
including the arm it was never fitted on — where a token-direction or
answer-direction account demands the opposite movement. It reaches **100% of
held-out rows in both arms** while moving 0.479 of ‖h‖, against 76% at 0.711 for
the closed-form difference-in-means baseline and 1–2% for a dose-matched random
subspace.

## 4. Where to read what

| Document | Scope |
|---|---|
| **[docs/METHODS.md](docs/METHODS.md)** | how everything is measured — the CPG ground truth, the floors and controls, the security benchmark, **DAS**, and the **J-/R-lens** stack, in full |
| **[docs/RESULTS.md](docs/RESULTS.md)** | every completed, successful result, each as *research question → hypothesis → method → result → what it means*; then the synthesis and the boundaries |
| **[docs/PIPELINE.md](docs/PIPELINE.md)** | setup, every stage, its command, its gates and its outputs |
| **[docs/ARCHIVE.md](docs/ARCHIVE.md)** | every retired or abandoned design, with the reason and the lesson it produced |

Machine-readable status per experiment: `results/STATUS.yaml`. Auto-generated
per-run reports live beside their data under `results/{binding,sinkflow,…}/` and
are regenerated by their report stages; they are outputs, not documentation.

## 5. How it works

```
generate programs      run the model once     read the state back    intervene
with exact ground  →   save hidden states  →  linear probes and    → on frozen
truth (CPG or          at chosen layers       output-basis lenses    artifacts
 execution)                                   + honest controls      (DAS)
```

Small Python programs whose semantic structure is **known exactly by
construction**; ground truth from the code property graph or from *executing*
them; every def/use/guard/sink event mapped to its exact token position through
verified AST-span→offset alignment. A frozen model reads each program once and
the hidden states are saved. Then deliberately **low-capacity linear probes**
decode the relations, with controls that kill the cheap ways to score high.

*Why synthetic, why linear, why one pass:* per-token labels must be exact
(static analysis on real code is approximate, and its errors become label
noise); a linear readout is the standard operationalisation of "explicitly
represented" — a stronger probe would measure the probe, not the model; and one
GPU pass decouples scarce GPU time from frequent CPU-only analysis. Full
rationale in [docs/METHODS.md](docs/METHODS.md).

## 6. What every experiment defends against

| Commitment | Shortcut it kills |
|---|---|
| **Grouped CV** — folds split by source program | rows from one program share hidden vectors; random folds leak train into test |
| **Selectivity control** — identical probe on shuffled labels | accuracy from class priors or per-program regularities |
| **Negative strata reported separately** | a headline averaged over easy negatives |
| **Measured surface baselines** — a ±3-token reader and a whole-program lexical reader, neither seeing hidden states | claiming a result beats "the text" without checking |
| **Verified token alignment** — AST spans → offsets, checked against source | string-matching a name silently mislabels shadows, which is what E2 measures |
| **Cross-validated ground truth** — `beniget`, instrumented execution, an independent scope-aware interpreter | labels wrong the same way in train and test look like signal |
| **Cluster bootstrap** over programs; controls paired on the same rows | intervals too narrow, in the direction that makes a null look like a finding |
| **Hard gates** — a stage refuses to run on a failed prerequisite | a control silently skipped, which is how E11 lost `probe_basis` |

> One non-obvious hazard: `AutoTokenizer` on transformers 5.x silently
> mis-tokenizes deepseek-coder (`def func` → `['de','ff','unc']`), relabelling
> *every* example without crashing. `src/models/loader.py::load_tokenizer`
> refuses any tokenizer that fails an exact code round-trip. See
> [docs/METHODS.md](docs/METHODS.md) §2.3.

## 7. Repository map

```
src/
  graphs/      AST / CFG / DFG / PDG extraction — the code property graph, and
               the source of every label
  data/        generators (foundation programs, obfuscation ladder, security
               benchmark, binding factorials), alignment, execution and
               reference-interpreter ground truth
  models/      loading (round-trip guard), hooks, J-lens, LRP rules (R-lens),
               DAS interchange
  probes/      linear probe, grouped CV + controls, dataset builders
  experiments/ one module per experiment family
  analysis/    metrics, tables, figures, cluster bootstrap
scripts/       numbered stage CLIs (00–131)
jobs/          csh scripts per GPU stage (run under screen; no scheduler)
configs/       model registry + canonical experiment settings
results/       STATUS.yaml + tables, figures, manifests, per-run reports
docs/          METHODS · RESULTS · PIPELINE · ARCHIVE
tests/         489 CPU-only tests (alignment exactness, CV leakage, strata,
               interchange algebra, gate refusal, obfuscation semantics,
               source→sink label recovery, …)
```

## 8. Quickstart

```bash
conda create -n semflow python=3.11 -y && conda activate semflow
pip install -e ".[dev]"
make test                     # 489 CPU-only tests
make smoke                    # tiny end-to-end run on this machine (~15 min, MPS)

# the foundation
python scripts/00_generate_data.py --model deepseek-coder-1.3b
make extract probes context obfuscation assets MODEL=deepseek-coder-1.3b

# the security track (E15 / E15-C / E15-D)
make sinkflow MODEL=deepseek-coder-1.3b

# the causal track (E13, DAS)
make binding-pilot            # then read results/binding/*/e13_report.md
```

## 9. Models

| Model | Role | Why |
|---|---|---|
| `deepseek-coder-1.3b-base` | development, smoke, pilots | runs on Apple-Silicon MPS; full pipeline in minutes |
| `deepseek-coder-6.7b-base` | main results | strong open code model; one cluster GPU in fp16 |
| `starcoder2-3b` | architecture replication | different corpus and architecture family. E15, E15-C and E15-D stages 128–129 all complete. The **R-lens does not exist there** — LayerNorm plus a non-gated MLP means the homogenising LRP rules bind to nothing, so stage 130 records that rather than producing numbers |

Base (non-instruct) models on purpose: the object of study is the
representation built during code pretraining, not chat behaviour.

## 10. Contributions

1. **Layer-resolved maps** of where CPG relations — binding, def–use, taint
   flow — are linearly decodable, against a floor pinned to chance *by
   construction* rather than estimated.
2. **A failure surface** for those representations, attributed to the
   individual transformation that causes it: robust to distance and to
   identifier spelling in middle layers, fragile under scope interference and
   **control-flow flattening alone**.
3. **A format result**: the security distinction is present in output-aligned
   coordinates and distributed across the vocabulary, carried by no word for it
   — established with a positive control that rules out instrument blindness,
   and with the security words running *backwards* at the cell where that
   control fires.
4. **A causal result** at the site where a binding is resolved, whose
   falsification refutes an answer-direction account rather than assuming it
   away.
5. **An instrument result about the tools themselves**: on a gated-MLP
   transformer it is the **gate's bilinearity, not the norm**, that carries the
   faithfulness gain of LRP (4.5×, replicated across two models and two dtypes,
   falsifying the pre-registered prediction) — and, separately, that the
   expensive lenses buy nothing over a plain logit lens when used as vocabulary
   projections. Reporting a validated instrument as unnecessary is part of the
   result.
6. **Methodology**, from four failed interventions: floors pinned by
   construction; positive controls matched in kind *and* in scale;
   magnitude-free interventions; and hard gates so a missing control is refused
   rather than skipped.
