# Methods

## How to read this document

This document explains exactly how each experiment was run. It starts by
defining what counts as a semantic representation, then explains how program
structure becomes exact token-level labels, and finally describes the four
steps of the active argument. Each step answers a different question:

- a **linear probe** asks whether information is present in a hidden state;
- **frozen transfer** asks whether the same representation survives a program
  rewrite;
- **DAS interchange** asks the causal question: whether changing only a learned
  binding component changes the downstream answer;
- after DAS establishes causal use, the published **J-lens** asks what that
  representation looks like in vocabulary coordinates—specifically, whether
  it verbalizes the language of binding; **R-lens** repeats the analysis as a
  supporting replication. Concrete-value recovery is a secondary contrast and
  positive control.

The controls are part of the method, not optional checks. Grouped splits prevent
nearly identical rows from leaking across train and test; shuffled labels test
selectivity; model-free readers measure what the text alone can reveal; and
cluster bootstraps keep uncertainty at the level of independent programs.
Whenever a later section says that a gate “passes,” it means these predeclared
requirements were satisfied. A pass makes the corresponding measurement
interpretable; it does not automatically prove the broadest possible claim.

How every number in `results/` is produced, and what each measurement is
entitled to conclude. Each section states **what** is done, **why** it is
necessary, and **how** it works, so the claims in
[RESULTS.md](RESULTS.md) can be judged without reading the code.

The overall logic: extract a frozen model's internal hidden states while it
reads a program, then ask a deliberately weak readout to recover a semantic fact
about that program whose truth is fixed by the program's own structure. If a
weak readout can recover the fact, the model has already made it linearly
available. The hard part is ensuring the readout is reading the *model's
computation* rather than a shortcut in the text — most of this document is about
closing those loopholes. The later sections distinguish causal use from
published lens readout rather than treating decodability as verbalizability.

### Contents

- [§0 What "semantic" means here](#0-what-semantic-means-here)
- [§1 The program-structure layer: the code property graph](#1-the-program-structure-layer-the-code-property-graph)
- [§2 From graph to token: alignment, ground truth, integrity](#2-from-graph-to-token-alignment-ground-truth-integrity)
- [§3 Instrument 1 — linear probes and their floors](#3-instrument-1--linear-probes-and-their-floors)
- [§4 Instrument 2 — frozen transfer and the obfuscation ladder](#4-instrument-2--frozen-transfer-and-the-obfuscation-ladder)
- [Part III — From representation, to causal use, to verbalization](#part-iii--from-representation-to-causal-use-to-verbalization)
- [§5 DAS — causal interchange of a binding component](#5-das--causal-interchange-of-a-binding-component)
- [§6 J-lens after DAS — identifying and verbalizing the used representation](#6-j-lens-after-das--identifying-and-verbalizing-the-used-representation)
- [§7 Statistics, gates and reproducibility](#7-statistics-gates-and-reproducibility)

---

# 0. What "semantic" means here

This word carries the whole project, so it is defined operationally rather than
gestured at.

## 0.1 Two notions, both called semantic

| | What it is | Ground truth from | Used by |
|---|---|---|---|
| **Abstract semantics** | sound approximations of behaviour: reaching definitions and def–use edges | the code property graph (`src/graphs/`) | E2, E3, E4 |
| **Concrete semantics** | what the program actually computes when run | execution — `execute_program`, observational equivalence, `interpret_scoped` | E9, E13 |

These are different levels and the project uses both. A claim about one is not
automatically a claim about the other: E2 says the model tracks *which
definition reaches a use* (abstract), E13 asks whether it causally uses *which
value that use resolves to* (concrete).

## 0.2 The working definition

The operative definition is negative, and it is **enforced rather than
estimated**:

> A property of a program is **semantic** to the extent that a bounded reader of
> the program's surface form cannot recover it.

Operationally: construct program pairs in which every feature such a reader can
see is held identical while the property flips. The model-free baseline then
scores **exactly 0.500 by construction**, not approximately (§3.4). Anything
above that floor is something the model computed rather than read off the page.

Two consequences make the definition usable rather than rhetorical.

**It is falsifiable, and it has already excluded something.** Control
dependence has a *measured* surface floor of **0.927** — a statement's guard is
usually its nearest enclosing `if`, so token windows plus indentation recover
most of the relation with no model at all. By this criterion control dependence
is *mostly syntactic*, so decoding it at ceiling (which the models do:
AUC 0.999) constrains nothing about semantic representation. It is therefore
**not reported as a result at all**; the measurement and the reasoning are in
[ARCHIVE.md §4.3](ARCHIVE.md#43-control-dependence). A definition that never
excludes anything is not doing work, and this is the exclusion.

**"Surface" is relative to a stated reader.** The floor is pinned against the
baseline in §3.4 (a ±3-token-id window plus bucketed distance). It is *not*
pinned against every computable text
feature — a cross-position string-equality feature lies outside the ±3 window
and is an open item. Claims should always say which reader the floor is pinned
against.

## 0.3 Semantics requires sensitivity *and* invariance

A property being semantic imposes two dual requirements on any representation
claiming to encode it:

- **Sensitivity** — hold the form fixed, change the meaning: the representation
  must change. (E2's `context_matched` stratum; E13's two arms.)
- **Invariance** — change the form, hold the meaning fixed: the representation
  must not care. (E9's obfuscation ladder, every variant execution-verified
  observationally equivalent to its base.)

Only one of the two is cheap to satisfy. Invariance alone is a hash of
behaviour; sensitivity alone is a diff of text. Semantics is the conjunction,
and the conjunction is measurable. The four cells make the design space
explicit:

|  | **form held fixed** | **form changed** |
|---|---|---|
| **meaning held fixed** | identity (trivial) | **E9** — the obfuscation ladder |
| **meaning changed** | **E2, E13** — one token, one relation | two unrelated programs (trivial) |

The bottom-left cell contains only *one-token, one-relation* instances. A
general form-preserving, meaning-breaking construction — a program that presents
as one thing and computes another — does not exist in this repository, and it is
the cell closest to the adversarial motivation.

## 0.4 What the definition does not reach

1. **Decidability.** Every relation studied here is exactly decidable by the
   code property graph or by execution. That is deliberate: approximate labels
   become label noise, and label noise becomes the finding. "Semantic" therefore
   means *decidably semantic on a restricted fragment*.
2. **Scale.** Binding and def–use are single-relation facts at token positions.
   Whole-program behaviour ("does this sort, or exfiltrate?") is a different
   object, not a larger one.
3. **Adversariality.** Nothing in the corpus is written by an adversary
   optimising to look benign. E9's transformations preserve meaning by
   design; a genuine attacker breaks meaning while preserving appearance.

---

# 1. The program-structure layer: the code property graph

Every label in this project comes from the program's own structure. The four
extractors in `src/graphs/` build the standard layers of a code property graph
(CPG), and each layer grounds a different relation.

## 1.1 The four layers

**AST — `ast_extractor.py`.** Character-offset-aware syntax nodes. Every node
carries `start_char`/`end_char` alongside line/column, which is what makes the
span→token alignment of §2.1 exact rather than approximate. The AST is not
itself a target relation; it is the coordinate system in which every other label
is expressed.

**CFG — `cfg_extractor.py`.** A statement-level control-flow graph with branch
targets and **join points resolved exactly**, so control does not "leak" past
where branches merge. Control dependence is read off guard nesting against this
graph.

**DFG — `dfg_extractor.py`.** Definitions (assignments, parameters, imports,
`for`/`with`/comprehension targets) and uses (`Name` nodes in `Load` context),
connected by **reaching-definition** edges. Two relations come from here:

- **Binding**: which *definition* a given identifier occurrence resolves to.
  This is scope resolution, not string matching — two occurrences of `data` may
  be the same variable or two different ones.
- **Def–use**: the directed definition→use edge itself, with the token distance
  between endpoints recorded so accuracy can be bucketed by reach.

## 1.2 Why the graph, and not a labeller

Two reasons, and both are load-bearing.

**Exactness.** A probe trained on approximate labels measures the approximation
as much as the model. Static analysis of *real* code is necessarily
approximate — which is why the primary corpora are synthetic programs whose
graph is known by construction, and why real-code transfer is reported as an
upper bound rather than as a representational result (see
[ARCHIVE.md](ARCHIVE.md), E8).

**The floor.** The construction-pinned 0.500 floor of §0.2 only exists because
the generator can emit *two programs with the same surface and different
graphs*. That requires authoring the graph, not inferring it.

## 1.3 Independent cross-checks of the graph

Because every downstream label depends on these extractors, they are validated
against independent implementations.

- **Def–use vs `beniget`** (`tests/test_ground_truth_crosscheck.py`), a mature,
  independently written reaching-definitions analysis. The two answer slightly
  different questions — ours resolves each use to the *single most recent*
  reaching definition, `beniget` returns *all* possibly-reaching definitions
  across branches — so the sound comparison is **set inclusion** (our edges ⊆
  `beniget`'s) with **exact equality on straight-line code**. This caught a real
  bug: uses in self-referential updates like `b = b + a` were linked to the
  *same-line* target definition instead of the prior one. The extractor now
  resolves in execution order (right-hand side before the assignment target).
- **Obfuscation vs execution.** Every transformed variant is executed and
  checked observationally equivalent to its base (§4.2).
- **Binding factorials vs a scope-aware reference interpreter.** E13's programs
  are decided twice — by execution and by `interpret_scoped` — and both must
  agree with the intended label (§5.5).

This is the same "validate the program graph against a second implementation"
discipline that production CPG tools (Joern, llvm2cpg) use.

---

# 2. From graph to token: alignment, ground truth, integrity

## 2.1 The alignment problem, and how it is solved exactly

The CPG speaks in source coordinates ("line 5, column 8"); the model speaks in
subword tokens. The translation must be exact, or every label is attached to the
wrong hidden state.

Each event is located by its **AST span** and mapped to token indices through a
**verified offset table**. Offsets are computed by incremental prefix decoding —
decode the first *n* tokens, measure how many characters they cover, repeat —
and the resulting table is checked to **reproduce the source exactly** before it
is used (`src/data/alignment.py`). A program whose round-trip fails is skipped
and counted, never silently mislabelled.

**Why not just match token strings.** With a subword vocabulary a variable name
may not be any single token, and string matching silently mislabels *shadowed*
names — two different variables spelled the same — which is precisely the
phenomenon binding measures. String matching would build the shortcut into the
ground truth.

## 2.2 Where in the model we read

**Layers.** Hidden states are captured at a fixed set of transformer blocks
spanning input to output, plus one special layer:

- **Layer −1 = the embedding output** — the token representation *before* any
  attention or context mixing. It encodes token identity only, and serves as the
  **context-free reference**: anything decodable here is a property of the token
  string, not of the model's computation.
- **Layer 0 and up = decoder-block outputs.** Note that "layer 0" is the output
  of the *first* transformer block, which has already mixed context once. That
  is exactly why layer −1 is extracted separately.

**Position.** For a task about a source-code event (such as a variable use), the
state is read at the event's **last covering token** — the
first position whose state can see the whole event under causal attention.
Reading earlier would miss part of the event; reading later folds in unrelated
downstream tokens.

**Cross-model reading is by relative depth, never by index.** The three models
have 24, 32 and 30 layers, so index 11 is 48% of depth in one and 35% in
another. Every result row carries a `relative_depth` column, and cross-model
tables are read at matched relative depth.

## 2.3 Tokenizer integrity

`AutoTokenizer` on transformers 5.x silently resolves deepseek-coder to a slow
SentencePiece path that **mis-tokenizes code** — `def func` becomes
`['de','ff','unc']` with whitespace dropped. Any activations or labels built
with it are garbage, and nothing crashes.
`src/models/loader.py::load_tokenizer` therefore loads via
`PreTrainedTokenizerFast` and **rejects any tokenizer failing an exact code
round-trip**. All results predating this guard are invalid and were rerun.

---

# 3. Instrument 1 — linear probes and their floors

## 3.1 The probe

Every decoding result uses a single, deliberately weak classifier: **logistic
regression** (`C=0.1`, class-balanced) on standardised features, `saga` solver,
`max_iter=2000`, `tol=1e-3`. No neural-net probes.

**Why weak on purpose.** A high-capacity probe can *learn* the semantic relation
itself from raw activations. If that happens, "the fact is decodable" tells you
about the probe's power, not the model's representation. A linear probe can only
read features the model has already made linearly available, so "decodable"
stays a statement about the model.

**Inputs.** Single-position tasks probe one token's state `h_i` directly.
Pairwise tasks — binding, def–use, control dependence — ask about a *relation*
between two tokens, so the feature vector captures both endpoints and their
interaction:

```
[ h_i ; h_j ; h_i − h_j ; |h_i − h_j| ]
```

**Convergence bookkeeping.** Whether each fit converged is recorded in every
results row (`converged`), and the probe stage fails its sanity check if any
*reported* fit did not converge. Shuffled-label control fits routinely hit the
iteration cap by design and are tracked separately as `control_converged`.

## 3.2 Cross-validation without leakage

Many probe examples come from the *same program* and therefore share overlapping
hidden-state vectors. Ordinary random k-fold would put some rows of a program in
training and others in test, letting the probe memorise program-specific quirks.

All cross-validation is **`StratifiedGroupKFold` grouped by source-example id**:
every row from one program stays entirely within one fold. When dataset size is
capped for tractability (`max_samples=20000` per task × layer), **whole groups
are dropped, never individual rows**.

## 3.3 Selectivity: the shuffled-label control

For every probe, the **identical** classifier is retrained on shuffled labels
and

> `selectivity = accuracy − control_accuracy`

is reported. **Claims are made on selectivity, not raw accuracy.** The shuffle is
done carefully: for pairwise and per-token tasks labels are shuffled *within*
each program, preserving each program's label mix and destroying only the
token→label pairing; for example-level tasks where the label is constant across
the program, the program→label assignment is permuted across programs instead.

## 3.4 The floors: two model-free readers

**The local surface baseline.** A probe seeing **no hidden states at all** —
only the ±3-token window of token ids around each anchor plus the bucketed
distance between anchors — fit with the same grouped CV and reported per
stratum.

This exists because of a specific failure. The first full run scored ~0.98 on
*every* task and layer, including the supposedly hard strata at the earliest
layer. The no-model baseline reproduced that ~0.98, proving the templated corpus
was **leaking labels through local token context**. The baseline is now a
permanent floor: *a hidden-state result only counts if it beats the surface
baseline on the same stratum.*

## 3.5 Negative strata: the honest headline

For a relation like binding, most negative pairs are trivially separable from
the text alone. Negatives are therefore split into **strata**, reported
separately from easiest to decisive:

| Stratum | What it is | What it controls for |
|---|---|---|
| `diff_name` | different variable names (capped at 3× positives) | trivial baseline |
| `distance_matched` | negatives at the same token distance as positives | positional shortcuts |
| `same_name_diff_binding` | same name, different actual binding | the name-identity shortcut |
| **`context_matched`** | **two token-identical programs differing by one binding-flipping character** | **every fixed-offset surface cue at once** |

`context_matched` is the only clean headline. Its two programs are identical
token-for-token except the single character that flips the label; the anchor
windows and token distance are identical; and both members share one CV group so
neither can be memorised through the other. By construction *no* feature of the
text can separate the labels, and the surface probe scores exactly 0.5 there.

---

# 4. Instrument 2 — frozen transfer and the obfuscation ladder

## 4.1 Freezing is the point

For robustness experiments the probes trained on clean base programs are
evaluated, **unchanged**, on transformed programs. They are never retrained per
condition.

**Why not retrain.** Retraining on each condition would measure how *learnable*
the relation is under that condition — a different and easier question, and one
that lets the probe find a *new* shortcut per condition, concealing exactly the
representational failure being looked for. Freezing measures whether the
representation the probe already found still holds when the input is stressed.

**Ground truth is recomputed for every variant.** Some transformations genuinely
change the program graph (competing updates in the context ladder; inserted
opaque branches and flattened control flow in the obfuscation ladder). The CPG is
re-extracted from each variant's own source, so a frozen probe is always scored
against the transformed program's real labels.

**What a frozen-probe failure does and does not show.** It shows the readout no
longer transfers. It does *not* prove the model lost the information — a
different probe, refitted, might recover it. Every claim in
[RESULTS.md](RESULTS.md) built on this instrument is phrased as a statement about
a frozen linear readout at a named position.

## 4.2 The obfuscation ladder

A cumulative, Tigress-inspired ladder implemented natively for Python
(`src/data/obfuscation.py`):

| level | transformation | what it changes |
|---:|---|---|
| 0 | normalize (`ast.unparse`) | formatting only — the baseline |
| 1 | + consistent renaming of every local | identifier strings |
| 2 | + dead branches under provably false opaque predicates | apparent control structure, not real flow |
| 3 | + mixed boolean-arithmetic rewriting (`a+b → (a^b)+((a&b)<<1)`) | expression form |
| 4 | + control-flow flattening into a dispatch loop | the entire control skeleton |

**Equivalence is verified, never assumed.** Every variant is executed and checked
observationally equivalent to its base — the same I/O-equivalence standard
Tigress uses. All levels of a given base are kept or dropped together, so
per-level curves always compare an identical set of base programs.

## 4.3 Atomic versus cumulative conditions

A ladder applied only cumulatively **cannot attribute a failure**: its last rung
contains every earlier one. The robustness analysis therefore applies the same
four rewrites **both individually and composed**, which gives three differences
per reported cell:

- `delta_clean` — what a condition costs relative to clean held-out;
- `delta_previous` — the marginal cost of the step a cumulative condition adds;
- `delta_atomic` — cumulative minus its atomic counterpart, i.e. **the
  interaction**.

The interaction is read against a **measured** draw-noise floor rather than
assumed to be zero: two conditions apply the *identical* transformation under
independent random draws (`rename_only` vs `rename_cumulative`), and their
difference is that floor.

Each variant's transformations are **read off its own AST** and must equal
exactly what its condition declares, so an arm cannot quietly contain more than
it says.

---

# Part III — From representation, to causal use, to verbalization

Parts I and II establish that binding is represented and measure its stability.
Part III follows the claims in their evidential order. First, DAS asks whether
downstream computation causally uses the representation. Only after that is
established does the published J-lens ask what the used representation is like:
is binding expressed in output-vocabulary language? R-lens supplies a supporting
replication. The lens is not evidence for causal use and does not find or train
the DAS direction.

The former security benchmark, earlier general output-vocabulary experiments,
standalone cotangent-lens studies, including E16 and E18, are preserved in
[ARCHIVE.md](ARCHIVE.md). They remain reproducible but are not needed for the
active binding argument.

# 5. DAS — causal interchange of a binding component

Probes and lenses show a fact is *present*, or *present in output coordinates*.
Neither can show it is **used**. A representation can be a faithful shadow of a
computation happening somewhere else, and no amount of decoding distinguishes
the two. Phase III needs an intervention, and the requirements are strict enough
that three earlier designs failed them ([ARCHIVE.md](ARCHIVE.md)).

## 5.0 Plain-language map of one DAS run

The shortest description is: **learn one internal axis that distinguishes which
definition of a variable is active, then copy only that coordinate from a donor
program into a host program and see whether the answer follows the donor.**

### First, one concrete example

Suppose the two variable names are `x` and `y`, and the two values are `3` and
`8`. The experiment builds these four programs:

| cell | program | normal answer | what changed? |
|---|---|---:|---|
| `ab_source` | `x = 3; def f(): y = 8; return x` | `3` | inner name is `y`, so it does not replace `x` |
| `ab_target` | `x = 3; def f(): x = 8; return x` | `8` | inner name is also `x`, so it shadows the outer `x` |
| `ba_source` | `x = 8; def f(): y = 3; return x` | `8` | same non-shadowing structure, but values are reversed |
| `ba_target` | `x = 8; def f(): x = 3; return x` | `3` | same shadowing structure, but values are reversed |

`source` and `target` do **not** mean training and test. They name the two
binding structures inside each arm:

- `source` means the inner assignment uses a different name, so `return x`
  reads the outer definition.
- `target` means the inner assignment reuses `x`, so `return x` reads the inner
  definition.

`ab` and `ba` name the value assignments:

- in `ab`, outer = `3` and inner = `8`;
- in `ba`, outer = `8` and inner = `3`.

Now take `ab_source` as the **host** and `ab_target` as the **donor**. Both runs
reach the identical visible token `x` in `return x`. Binding DAS replaces one
learned internal coordinate of the host state with the donor's coordinate. If
that coordinate carries which definition of `x` is active, the edited host
should answer `8` instead of `3`.

That result alone is not enough. A direction that merely pushes the token `8`
would also pass. The decisive repeat uses `ba`: the same structural change must
now make the answer move from `8` to `3`. A fixed “push `8`” direction fails,
whereas a binding coordinate can still succeed because it means “use the inner
definition,” not “say 8.”

| term | simple meaning |
|---|---|
| **host** | the program whose answer we are trying to change |
| **donor** | the matched program containing the binding we want to install |
| **site** | the token position where the hidden state is edited; here, the unchanged variable-use token |
| **layer** | the transformer block at which the edit is made |
| **subspace / `R`** | the learned internal direction or directions being exchanged |
| **rank 1** | only one learned direction is exchanged |
| **interchange** | keep the host state except for its coordinate along `R`, which is replaced by the donor's coordinate |
| **installed answer** | the value that would be correct if the donor's binding had been installed successfully |

### The run, one small step at a time

One complete run has these steps:

1. Generate four matched programs for each base example: two binding structures
   crossed with two assignments of the literal values (`ab` and `ba`).
2. Verify independently that every program has the intended answer and that the
   crossed arm reverses which literal is required.
3. Check that the unedited model can solve every cell.
4. At the same visible use token, check that binding is decodable. This is only
   a prerequisite, not the causal result.
5. Split base programs before fitting. The **calibration split** is used to
   choose the layer, site, and rank and to learn directions. The **test split**
   is kept unseen for the final numbers.
6. On calibration examples from `ab` only, learn `R`. The language model itself
   stays frozen; only the direction `R` changes during optimization.
7. For each calibration host/donor pair, run the model with the interchange,
   measure the probability of the donor-selected answer, and update `R` to make
   that answer more likely. Re-orthonormalizing `R` keeps it a valid projection.
8. Freeze the layer, site, rank, and learned direction. No test result is used
   to retune them.
9. Evaluate the frozen intervention once on held-out `ab`, held-out reversed
   `ba`, and every control below.
10. Apply gates H0–H5 in order. H5 supplies the binding-versus-answer
   discrimination; earlier gates establish that H5 is meaningful.

### Published DAS idea versus this project's design

| component | origin |
|---|---|
| Learn a low-dimensional alignment and replace aligned coordinates between causal runs | **Published:** Distributed Alignment Search and interchange interventions (Geiger et al., 2021; 2023). |
| Projection formula in §5.2 and optimization of an orthonormal `R` | **Published DAS mechanism**, implemented here for this model and site. |
| Python variable-shadowing programs, four-program factorial, and unchanged use-token anchor | **This project.** |
| Fit on `ab`, freeze the fit, and test the opposite token requirement on `ba` | **This project.** This separates binding from a fixed answer direction. |
| Separately trained, row-wise dose-matched `das_answer_control` | **This project.** It tests “the optimizer merely learned to push the fitted answer token.” |
| H0–H5, structural zeros, surface floor, whole-state reference, mean/random controls, and cluster bootstrap | **This project.** These controls license the narrow claim around the published DAS core. |

### Normal binding DAS versus answer-only DAS

These are **two different trained interventions**, not two output modes of the
same fit.

| | `das_binding` | `das_answer_control` (“answer-only”) |
|---|---|---|
| training information | paired hidden states with different active definitions | current and requested answer-token identities |
| receives a donor binding state? | **yes** | **no** |
| what it can learn | a donor-copyable coordinate that may encode which definition is active | a direct actuator such as “move from answer `a` toward answer `b`” |
| test-time dose | natural donor-coordinate replacement | forced to equal binding DAS's edit norm on the same row |
| expected on fitted `ab` | succeed | succeed; otherwise it is a dead control |
| expected on reversed `ba` | still succeed if relational | attenuate or point the wrong way because its fitted token orientation is frozen |

The answer-only control is deliberately capable. Its failure on `ba` is useful
only after it proves on `ab` that it can move the model at this site.

## 5.1 What a usable intervention must have

Three properties at once:

1. **Act where the two programs are token-identical.** Otherwise a patched state
   transports the *input difference* along with any semantic content, and the
   design cannot separate them. This is the failure that retired whole-state
   activation patching.
2. **Edit a nameable part of the state, not the state.** A whole-state swap
   replaces everything the position holds; what was installed is unspecifiable.
3. **Act at a magnitude the site can actually register.** An edit below the
   site's causal dose produces a null that means nothing. A rank-2 additive swap
   moving 2–4% of ‖h‖ at a site whose dose–response is **18× convex** registers
   nothing whether or not the coordinates are read — which is what retired the
   coordinate-swap design.

## 5.2 The interchange operator, and why it has no dose knob

Every earlier causal instrument set the size of the edit by hand. An
**interchange** has no such parameter:

> `h' = h_self + R Rᵀ (h_other − h_self)`

`R` (`d × r`, orthonormal columns) *names a subspace*; the intervention installs
whatever **the other run actually has** in that subspace and leaves the
orthogonal complement of `h_self` untouched. There is no `α` to choose, so "was
the dose enough?" is not a question the design has to answer. The size of the
edit is a **measured consequence** (`edit_fraction`), reported for every
condition and used in the decision rule.

Three algebraic properties, each pinned by a unit test:

- the component of `h_self` orthogonal to `span(R)` is untouched;
- `h_other == h_self` gives *exactly* the zero edit, so the no-op control is
  **provably** inert rather than approximately so;
- at `rank == d` the result is exactly `h_other` — the whole-state patch is the
  rank-`d` limit of the same operator, which is what makes it the right ceiling
  to normalise against.

The algebra is defined in float64 on numpy; only the per-call products run on
device, and the fp16 hidden state is upcast before the edit and cast back by the
hook, so the intervention never happens in half precision.

## 5.3 How the subspace is learned, and what a null means

`R` is **learned** in the style of distributed alignment search (Geiger et al.,
[arXiv:2303.02536](https://arxiv.org/abs/2303.02536)) by maximising interchange
accuracy on a **disjoint calibration split**:

- only `R` is trained; every model parameter is frozen;
- the parameter is unconstrained and re-orthonormalised **inside the graph**
  (`torch.linalg.qr`) at every step, so the operator is a true interchange at
  every point of the optimisation, never only at the end;
- the intervention site's *incoming* state is **detached**, which is exact rather
  than an approximation: layers before `L` cannot depend on `R`, so no gradient
  should flow there, and detaching keeps the backward pass to the tail of the
  network;
- the loss is the negative log-probability of the token the *donor's* value
  implies, at the final position.

Two consequences for interpretation:

- **A null is strong.** "No `r`-dimensional subspace here behaves this way" is a
  much stronger statement than "the two directions I picked did not".
- **A positive is weak without controls.** Because it is learned, the method is
  expressive enough to find structure that is not there. Everything in §5.4 and
  §5.5 exists for that reason.

## 5.4 Controls and competing explanations

Here, a **control** is a comparison intervention designed to answer one specific
objection. A **gate** is a rule that decides whether the run is interpretable.
A **metric** is merely a number reported by an intervention. These three things
are different.

### Treatment: `das_binding`

This is the intervention whose claim is being tested. It learns from real
host/donor binding pairs. At test time it copies the donor's coordinate in the
learned subspace into the host. It must install the donor-selected answer in
both value arms. It is not itself a control.

### Main alternative explanation: `das_answer_control`

This control asks: **could an optimizer get the same result by learning how to
push answer tokens, without learning binding?**

It comes from this project, not from the published DAS recipe. It is deliberately
trained with nearly every advantage given to binding DAS:

- the same model, layer, token position, calibration/test split, Adam optimizer,
  number of steps, and random seed;
- a rank-1 vector for each needed token transition;
- exactly the same edit length as binding DAS on each test row.

The one crucial difference is its information. It never receives the donor's
binding state. It sees only “the current answer token is `a`; make the requested
answer token `b`.” On the fitted `ab` arm it learns that orientation. The vector
is then frozen. On `ba`, the correct movement reverses, but the control is not
allowed to relearn the reverse orientation.

Why it is necessary: without this control, successful binding DAS might only be
a sophisticated answer-token actuator. Why it must pass on `ab`: if it cannot
move the model anywhere, its later failure on `ba` proves nothing. Why it should
attenuate on `ba`: that is the behavioral signature of a fixed answer movement,
which binding DAS must differ from. This is the decisive H5 discriminator.

### Capability reference: `whole_state`

This copies the donor's **entire** hidden state at the same site. It is the
rank-`d` limit of the same interchange formula. It asks whether that location
can causally affect the answer at all.

Why it is necessary: if even the whole donor state cannot move the answer in an
arm, then a low-rank failure there is uninterpretable; the site may simply be
causally ineffective. It is called a ceiling or reference, but its numerical
effect need not exceed targeted DAS. Copying everything can also copy irrelevant
or opposing information, so a clean rank-1 edit can outperform it.

### Simple non-learned baseline: `mean_difference`

For every calibration pair, compute donor state minus host state, average those
differences, and use the one-dimensional span of that average. There is no
optimizer and no answer-label training.

Why it is necessary: perhaps the binding difference is so consistent that a
plain average already transports it. DAS should be compared with that cheap
explanation, not only with zero. A positive mean baseline is informative: it
means binding is partly present in a broad average direction, even if DAS is
more selective and reliable.

### Random floor 1: `random_rank`

Draw a random subspace with the same rank as DAS. Since binding DAS is rank 1,
this is one random direction.

Why it is necessary: it checks whether any arbitrary direction of the same
dimensionality changes the output. Its weakness is that a random rank-1 direction
usually makes a much smaller edit than a learned direction, so rank matching
alone is not a fair dose comparison.

### Random floor 2: `random_norm`

Draw a random subspace large enough that its actual hidden-state edit matches
binding DAS's edit magnitude.

Why it is necessary: a large perturbation can damage the model and accidentally
raise some scores. This control asks whether DAS's effect is special or merely
what happens when the state is disturbed by the same amount. This is the more
important random control because it matches **dose**, not just rank.

### Machinery zero 1: `noop`

Apply an edit that is exactly the zero vector.

Why it is necessary: any measured output change must then come from batching,
precision, hook, or comparison machinery—not semantics. This is an engineering
correctness check. A nonzero result invalidates the causal run.

### Machinery zero 2: `def_source`

Intervene at a position before the one-token source/target mutation. At this
point the host and donor states should be identical, so interchange must make no
change.

Why it is necessary: it independently catches wrong token anchors, wrong cached
states, and injection artifacts. Like `noop`, it is a provable zero, not a weak
scientific baseline.

### Optional lens diagnostics

`answer_direction_jlens`, `answer_direction_rlens`, and
`answer_direction_unembedding` are fixed answer directions derived respectively
from the published J-lens, the published R-lens, or the raw output embedding.
They are scaled to the DAS edit norm and use the same test rows.

Why they exist: they diagnose whether a direction useful for **reading** an
answer also works for **causing** that answer at the DAS layer, and whether
Jacobian transport adds anything beyond a raw output direction. They do not
train, initialize, constrain, or validate binding DAS. They are descriptive and
do not enter H5. Earlier versions mistakenly used a cotangent-lens direction as
the main answer control; those versions are retired because the direction was
not a live causal actuator at the selected site.

### All controls at a glance

| item | question it answers | required behavior |
|---|---|---|
| `das_binding` | does the learned donor coordinate transport binding? | succeeds on `ab` and reversed `ba` |
| `das_answer_control` | is ordinary answer pushing a different explanation? | live on `ab`, attenuated on `ba` |
| `whole_state` | can this site affect the answer? | live in both arms |
| `mean_difference` | does a simple average direction already work? | measured baseline; DAS should dominate for H4 |
| `random_rank` | does any rank-1 direction work? | much weaker than DAS |
| `random_norm` | does any equally large disturbance work? | much weaker than DAS |
| `noop` | does the apparatus invent change with a zero edit? | exactly zero within tolerance |
| `def_source` | are anchors, caches, and hooks correct before the mutation? | exactly zero within tolerance |
| J/R/unembedding directions | do fixed read/output directions steer at this site? | descriptive only; no required causal result |

### Advanced implementation history

The remainder of §5.4 records why older controls were retired and why the zero
checks use matched execution paths. It is useful for reproducing or auditing the
code, but it is not required for the first conceptual reading. A first-time
reader can continue at [§5.5](#55-the-crossed-22-that-identifies-binding).

**Retired 2026-09-01: the cotangent `answer_direction`.** Until then the arm
above was built from a *corpus-averaged cotangent readout over the two answer
tokens*, fitted inside stage 106 from the DAS calibration programs. That is a
different estimator from the published J-lens — a fixed-candidate-vocabulary
readout with the final normalizer dropped, tabulated against the published
method in [WORKSPACE_LENS.md §1](WORKSPACE_LENS.md) — and naming its output
"J-lens vectors" made the E13 control unreadable next to E19. A later attempt
used the actual stage-201 read directions, but those were causally dead at the
chosen site. Both attempts are archived, not carried forward. The active H5
discriminator is the trained `das_answer_control`; no lens artifact is required.
See [ARCHIVE.md](ARCHIVE.md).

**DAS itself remains lens-independent.** No lens initializes, constrains,
trains or gates the alignment. Stage 106 can run the complete H4/H5 experiment
without a lens artifact. If requested, J/R artifacts are read only for optional
control-panel diagnostics.

**The structural zeros, and why the clean pass is batched.** `noop` and the
pre-mutation `def_source` site are *provable* zeros: the no-op edit is the zero
vector and at `def_source` the host and donor are the same state, so the model's
output cannot move. Both are checked against a `1e-4` tolerance — a float32
rounding allowance, not a fitted threshold — and **H3, H4 and H5 all carry the
check as a precondition**: a run whose provable zeros do not hold has produced no
result, so no claim gate may pass on it.

Making that check mean anything requires the reference and the treatment to run
through the *same execution path*. They did not, until 2026-09-02, and there
were **two** distinct faults — the second only visible once the first was fixed.

**Fault one: the logits were compared across batch shapes.** The clean baseline
came from `collect_states`, which runs one prompt per forward call, while the
patched logits come from a batch of 32 — and in reduced precision the LM head's
matmul is a different cuBLAS kernel at a different shape. On DeepSeek-Coder 6.7B
that surfaced as no-op deltas of exactly 0, ±0.125 and ±0.25 (bfloat16, one ulp
at |logit| ≈ 64 being 0.25), and ±0.0156 in the float16 ceiling. The edits in
those rows were exactly the zero vector — `edit_norm == 0.0`, computed in numpy
and unable to be affected by anything the GPU does — so the arithmetic was never
in doubt; the comparison was.

**Fault two: the states are cached at batch 1 and injected at batch 32.**
`collect_states` captures the residual stream one prompt at a time, and the grid
writes it back inside a batch, so a cached state differs from the live one by
about an ulp per component *even when it is the right state*. `noop` hides this
— its edit is a low-rank projection of that difference, which rounds away, and
in stage 105 the basis is rank 0 so nothing is written at all — but
`whole_state` at the pre-mutation site installs the cached state wholesale and
does not. That is why fixing fault one left `noop` at exactly `0.0` while
`pre_mutation_whole_state` stayed at `0.03125` with `edit_norm` exactly `0.0`:
the two conditions were never testing the same thing.

**The fix for both is one change.** The clean reference is now the
**self-interchange** — each variant's own operator with the donor replaced by
the host's *own cached state* — run over the same batch as the treatment. In
exact arithmetic `interchange(h, h, R) == h`, so this is still "no edit"; in
finite precision the reference and the treatment now carry the identical
cached-vs-live offset and it cancels out of their difference. At the
pre-mutation site the donor state *is* the host state bit for bit, so the two
passes are the same pass and `delta_ld` is exactly `0.0`. Everywhere else the
measured effect is exactly the donor-minus-host state difference, with the
injection artifact removed from both sides rather than from neither — which
makes the whole-state ceiling itself slightly cleaner than before.

Both discrepancies are retained per row as `clean_path_shift`, so precision
effects once mistaken for failed zeros stay visible rather than being silently
corrected away. The cost is one extra forward pass per batch, which at E13's
uniform 21-token prompts is the cheapest part of the stage. **The tolerance was
not widened to fit the failure.**

**Why `random_norm` and not just `random_rank`.** For an orthogonal projector
only `span(R)` matters, so matching the Gram matrix of the rows says nothing. A
random rank-`r` subspace of a `d`-dimensional stream captures on average `r/d` of
the state's energy, while a learned one is *selected* to capture far more — so
rank-matching alone leaves the two conditions dose-mismatched **in the direction
that manufactures a positive**. The required rank is estimated in closed form
from `E‖R Rᵀ d‖² = (r/d)‖d‖²`, then bracketed exactly; the rank actually reached
is reported, because needing many random dimensions to match one learned
dimension is itself informative.

**A control has to be shown to work, not assumed to.** The discriminator's job
is to fail on the crossed arm — but that is only informative if it *succeeds* on
the arm it was built for. Both halves are therefore read on `says_installed`,
the full-vocabulary argmax, and the success half additionally has to clear the
**dose-matched random floor**: the control must emit the installed answer more
often than a random direction of the same edit norm does. Until 2026-09-02 the
success half read `delta_ld` instead, and the two disagree exactly when it
matters — `delta_ld` is positively biased here (H1 is 1.000, so any large edit
disrupts a confident distribution and lifts the margin with nothing
transported), so a dead control can show a tight positive interval while never
once producing the installed answer. On the published-J-lens run it did: the
control was correctly dose-matched at 0.416 of ‖h‖ and produced the installed
answer on **0.0%** of training rows against a random floor of 1.6%, while its
margin interval was `+0.098 [0.083, 0.113]`. That is the E10-3 lesson applied to
the control itself.

**Why the answer-direction control is decisive.** In the fitted `ab` arm, the
donor binding changes the required answer from value `a` to value `b`. The
control directly learns a causal mid-layer actuator from `a` toward `b`, without
seeing the donor's binding state. Its dose is fixed by binding DAS rather than
tuned. A synthetic donor makes the interchange an exact
push along this direction, and its length is set equal to the DAS edit on that
same example. The comparison therefore does not give DAS a larger dose.

The held-out `ba` arm reverses the assigned values. There the same binding swap
requires the answer to move from `b` to `a`. The fixed `a`-to-`b` control should
now fail or point the wrong way, while a direction carrying the abstract fact
“switch from the outer definition to the inner definition” should still install
the correct value. Thus the control demonstrates both that an output-directed
edit can affect the model at this site and that such an edit has a different
cross-arm signature from binding transport.

**A diagnostic worth naming: `concentration`.** Transformer residual streams have
a handful of massive-activation dimensions whose values dwarf the rest, and an
unconstrained low-rank fit maximising a logit shift will happily align with one
of them — producing a large effect while transporting nothing about the variable
under study. `AlignedSubspace.concentration(top_k)` reports the share of the
basis's mass on its largest dimensions: a basis spread over the stream gives
≈`top_k/d`; one riding a rogue dimension approaches 1.0.

## 5.5 The crossed 2×2 that identifies binding

The design crosses **binding structure** with **value assignment**. Four programs
per base, all token-identical except one character:

```python
# ARM ab: (outer, inner) = (a, b)        # ARM ba: (outer, inner) = (b, a)
x = a                                    x = b
def f():                                 def f():
    y = b   # target program: `x = b`        y = a   # target: `x = a`
    return x                                 return x
#   → a  (outer)   /  → b  (inner)       #   → b  (outer)   /  → a  (inner)
```

Install the *target* run's state into the *source* run at the marked use. In arm
`ab` the answer must move **a → b**; in arm `ba` the same intervention must move
it **b → a**. **Fit the alignment on `ab`; read the claim on `ba`.**

In concrete terms, three outcomes have different meanings:

- If both binding DAS and `das_answer_control` worked equally in both arms, the
  experiment could not distinguish binding transport from a lens-visible answer
  direction, and the causal verdict must not pass.
- If neither worked in the held-out arm, that arm might simply be insensitive to
  intervention, so a DAS null would be inconclusive.
- The identifying result is that DAS follows the binding in both arms while the
  deliberately fixed answer direction works in the fitted arm and attenuates or
  reverses in the crossed arm.

| account of what the subspace carries | arm `ab` | arm `ba` |
|---|---|---|
| *which definition is in scope* | positive | **positive** |
| *the token `b`* | positive | **negative** |
| *the answer* | positive | **negative** |

This is the table earlier designs could not build. With an arithmetic operation
between the value and the answer, a design must forbid `answer == value` to avoid
circularity, and pays for it with a capability requirement the model may not
meet. **Here the answer *is* the bound value — deliberately — and the arm
crossing breaks the circularity instead.** So there is **no arithmetic
anywhere**: the model returns a variable.

## 5.6 The outcome metric, and why it is the argmax

The primary outcome is **`says_installed`**: whether the model's full-vocabulary
argmax is the token the *installed* binding selects. It is not the two-way logit
margin `delta_ld`.

The reason is specific and was written into the module docstring before any
large-model run. When clean behavioural accuracy is at ceiling, the clean
distribution is confident and `logP(own)` sits far above `logP(installed)`. Any
edit that merely **disrupts** the state regresses both toward the middle and
*raises* `delta_ld` with nothing transported. A disruption cannot systematically
produce the *correct installed token* as the argmax. Both are recorded on every
row; the gates read the argmax. The history of that correction — including the
verdicts under both rules — is in [ARCHIVE.md](ARCHIVE.md).

### Metric dictionary for the run reports

| field | meaning | how to read it |
|---|---|---|
| `says_installed` / `installed` | edited model's full-vocabulary top token is the donor-selected answer | primary success measure |
| `flip` | the original answer became the installed answer | intuitive behavior-change rate; near `installed` when clean accuracy is perfect |
| `delta_ld` | change in `log P(installed) - log P(own)` | effect size only; generic disruption can raise it |
| 95% CI | cluster-bootstrap interval over base programs | supports the sign of a mean effect when it excludes zero; does not replace `installed` |
| `|edit|` | Euclidean length of the hidden-state change | absolute intervention dose |
| `|edit|/|h|` / `edit_fraction` | edit length divided by state length | comparable dose across layers and models |
| `% of whole-state ceiling` | low-rank `delta_ld` divided by whole-state `delta_ld` | relative reference, not accuracy; may exceed 100% if the targeted edit avoids competing donor information |
| `vs das` | paired difference between binding DAS and a control | positive means binding DAS made the larger requested movement |
| `selectivity` | real-label decoder score relative to shuffled labels | prerequisite check against classifier flexibility |
| `n` / `bases` | intervention rows / independent base programs | intervals cluster by base because rows from one base are related |

## 5.7 The six gates

Each refuses to run downstream stages until it passes.

| gate | asserts | threshold |
|---|---|---|
| **H0** | execution and a scope-aware reference interpreter agree; every invariant holds, **including the arm crossing** | ≥ 0.999 of bases |
| **H1** | the model returns the correctly bound variable | ≥ 0.85 overall, ≥ 0.75 per cell |
| **H2** | which definition is in scope is decodable at the use anchor | ≥ 0.80, and ≥ 0.10 over the *measured* surface baseline |
| **H3** | whole-state interchange flips the answer **in both arms** | CI > 0, flip rate ≥ 0.25 |
| **H4** | low-rank interchange beats matched controls on the **training** arm | ≥ 50% of that arm's ceiling |
| **H5** | the same subspace transfers to the **held-out** arm | ≥ 50% of that arm's ceiling, **and** `das_answer_control` **works on the training arm and fails on the held-out one**. Both halves are read on `says_installed`; "works" means ≥25% installed answers and more than a random direction at the same edit norm. A control that fails everywhere separates nothing, so H5 fails with it |

H5's discriminator is `das_answer_control` and only that arm. Published J/R
directions may be reported beside it, but they are readout diagnostics and do
not gate the causal experiment. A summary from an older discriminator is
refused rather than silently rescored.

H1 exists because of a lesson recorded in [ARCHIVE.md](ARCHIVE.md): check the
model can do the task *before* building an instrument on top of it.

### Why each gate exists

- **H0 — data are logically correct.** Python execution and a separate
  scope-aware interpreter must agree. Values and tokens must be distinct, the
  mutation must be far enough from the use token, and `ab`/`ba` must demand
  opposite token movements. If H0 fails, the experiment does not test its stated
  question.
- **H1 — the model can do the unedited task.** A causal edit cannot reveal how a
  model performs a task it does not perform. Overall and worst-cell thresholds
  prevent one easy condition from hiding a failed shadowing condition.
- **H2 — binding information is present at the chosen site.** A frozen decoder
  must recover the binding above both an absolute threshold and the measured
  local-text baseline. This does not prove use; it prevents searching for a
  causal binding coordinate where the prerequisite representation was not found.
- **H3 — the site is causally reachable.** Whole-state interchange must move the
  answer in both value arms. Otherwise failure of a smaller edit has no clear
  meaning. The structural-zero checks must also remain zero.
- **H4 — the learned intervention works where it was fitted.** On `ab`, binding
  DAS must reach at least half the whole-state reference and clear its specified
  controls. This checks that fitting produced a real, usable intervention before
  asking whether it generalizes.
- **H5 — the interpretation survives the falsification.** The frozen binding
  subspace must work on reversed `ba`. At the same time, the answer-only control
  must be demonstrably live on `ab` and attenuate on `ba`. H5 is the step that
  supports “binding transport” rather than merely “an intervention changed the
  answer.”

The numerical thresholds are project decision rules fixed before the final
test result, not values supplied by the published DAS paper. A PASS means the
predeclared minimum was met. It does not mean the measurement is perfect, and
it does not broaden the conclusion beyond this task, layer, site, and model.

### What the final pattern does and does not establish

The completed pattern is:

1. unedited models solve the task;
2. a whole-state donor patch shows the site can affect the answer;
3. rank-1 binding DAS installs the donor-selected answer in both token
   orientations;
4. the equally dosed answer-only actuator is strong on its fitted orientation
   and weaker after the required token movement reverses;
5. random and zero controls cannot explain the effect.

This supports the narrow statement that downstream computation at the tested
site uses a compact component whose causal effect follows which definition is
active. It does **not** prove that the direction is unique, that the model has a
human-like concept of scope, that every layer uses the same code, or that the
result automatically transfers to arbitrary real programs.

---

# 6. J-lens after DAS — identifying and verbalizing the used representation

## 6.0 Which J-lens operations are actually applied?

| operation | applied here? | exact determination |
|---|---|---|
| **READ** | **Yes; the main E19 operation.** | Transport `h_l` with `J_l`, apply the model's final norm, unembed over the full vocabulary, and rank. Runtime-value recovery and binding-language verbalization are both READ panels. |
| **WRITE** | **No.** | No J-lens vector is added as `h <- h + alpha v_t` to steer generation. |
| **PATCH** | **No J-lens PATCH.** | No two J-lens coordinates are extracted with a pseudoinverse and swapped. DAS exchanges a separately learned binding coordinate; that does not make it a J-lens PATCH run. |
| **ABLATE** | **Only a project-specific single-direction variant.** | The code removes one target token's read direction, `u = J_l^T(g W_U[w])`, at one layer at a time. It does not use gradient pursuit to find and remove the top 10 active subframes across a layer band. |

In short: **READ was applied; WRITE and J-lens PATCH were not; the causal test
is an ABLATE-style read-direction erasure developed for this project.**

### Published J-lens components versus this project's additions

| component | origin |
|---|---|
| Full Jacobian transport, penultimate target layer, final normalization/unembedding, and full-vocabulary ranking | **Published J-lens idea and released implementation.** |
| LN, activation-identity, and gated-product half backward rules | **Published RelP/R-lens idea.** |
| Independent pretraining-like fitting corpus | **Published recipe**, instantiated here with a reproducible 100-prompt prefix. |
| Code-binding task and four reads: `use`, `post_use`, `call`, `answer` | **This project.** |
| Separate concrete-result READ and binding-language READ | **This project.** |
| Binding lexicon, crossed value arms, and generic/random/positional controls | **This project.** |
| One-direction erasure with distractor and dose-matched random controls | **This project**, using a published lens direction but not the paper's top-10 ABLATE recipe. |

## 6.1 The inferential question and the published lenses

DAS has now established that a compact binding component is causally used at
the tested site. The next question is not whether binding is present or used,
but **what that representation is and whether it is verbalized**. The primary
test therefore targets binding language with the published J-lens; value-token
recovery is retained as a secondary comparison and the answer position as a
positive control. R-lens repeats the same measurements as supporting evidence.

E19's primary instrument is the J-lens. It is a separate instrument, not a relabelling of §6.8. It vendors Anthropic's
released Jacobian-lens implementation at commit `581d398`, fits the full
`d_model × d_model` transport from every source layer to the released
penultimate-layer target, and reads with the model's own final normalization and
unembedding over the complete vocabulary. J and R use the same independent
100-prompt `NeelNanda/pile-10k` prefix; R differs only through the published LN,
activation-identity, and gated-product half rules.

The evaluation fixes target concepts from program execution. Value programs are
read at four positions in one prompt: use, post-use, call, and answer. The answer
position is the positive control; the earlier three test availability for
verbal report before emission. Target-absent concepts distinguish computation
from copying.

The causal arm projects out `J_lᵀ(gW_U[w])` or its R-lens analogue and scores the
model's own target-minus-distractor answer margin. Controls are the plain
unembedding direction, a stable-seeded random projection, an independent random
displacement with exactly the J erase magnitude, and separate J/R distractor
directions. Contrasts are paired within program and layer with 95% cluster-
bootstrap intervals.

Required gates cover corpus independence, matched provenance, identity-anchor
readout, equivalence to the LM head, RelP forward invariance, complete rule
binding, and a nontrivial J/R difference. StarCoder2's LayerNorm rule is a
documented analogue, so a paper-minimal sensitivity fit disables it and retains
only the exact GELU identity-rule. See [WORKSPACE_LENS.md](WORKSPACE_LENS.md).

## 6.2 Binding-language verbalization: exact vocabulary and procedure

A **separate** question from runtime-value recovery, run by stage 206 and kept in
its own tables: at the same four read positions, does the lens surface the
*language of binding* — `local`, `global`, `inner`, `outer`, `scope`, `scoped`,
`shadow`, `shadowed`, `binding`, `bound`, `active`, `inactive`, `definition`,
`variable`, `value` — over the full vocabulary? A null in one panel says nothing
about the other, so they are never pooled.

The exact declared spellings are: `local`/`Local`/`LOCAL`;
`global`/`Global`/`GLOBAL`; `inner`/`Inner`; `outer`/`Outer`; `scope`/`Scope`;
`scoped`/`Scoped`; `shadow`/`Shadow`; `shadowed`/`Shadowed`;
`binding`/`Binding`; `bound`/`Bound`; `active`/`Active`;
`inactive`/`Inactive`; `definition`/`Definition`/`def`;
`variable`/`Variable`/`var`; and `value`/`Value`/`val`. Each spelling is tried
both bare and space-prefixed. The matched generic-code controls are
`return`/`Return`, `import`/`Import`, `class`/`Class`, `print`/`Print`,
`range`/`Range`, `index`/`Index`/`idx`, `result`/`Result`,
`buffer`/`Buffer`/`buf`, `string`/`String`/`str`,
`number`/`Number`/`num`, `file`/`File`, `path`/`Path`,
`error`/`Error`/`err`, `count`/`Count`, and `total`/`Total`. The confound
diagnostics are `earlier`/`Earlier`/`first`/`First`,
`later`/`Later`/`last`/`Last`, `kept`/`Kept`/`keep`, and
`replaced`/`Replaced`/`replace`.

Everything is predeclared in `src/workspace_lens/concepts.py`: the concept sets
and their spellings, the four read positions, the controls, and the four
conditions a positive must meet. A concept is a *set* of single-token spellings
and scores as the best rank over that set; a word the tokenizer splits is
recorded as unavailable and scored on nothing, never reduced to an unrelated
first token, and every accepted token id and decoded spelling is written into
the manifest and the report.

The programs are the same shadowing construction, crossed on the **value
assignment** as well as on the binding, so all four required contrasts exist in
one corpus: binding-flipped arms that are token-identical at the read position,
value-crossed arms with the literals swapped, values changed across bases, and
matched controls — unrelated code vocabulary of comparable frequency and
tokenization, size- and frequency-band-matched random concept sets, and
positional/action wording (`earlier`/`later`, `kept`/`replaced`) carried
explicitly as a **confound diagnostic** rather than as binding semantics.

Reported per (lens, layer, read, concept): full-vocabulary rank, pass@k for
k ∈ {1, 5, 10, 50, 100}, the earliest layer entering each threshold, the paired
inner-minus-outer score difference with a cluster bootstrap over base programs,
its agreement across the crossed value arms, and its invariance to which literal
is in scope.

A **supported positive requires all four** of: predeclared binding concepts
moving consistently with the binding; agreement across the crossed value arms;
stronger movement than the matched generic and positional controls; and
replication across prompts, preferably across models. Nothing is redefined
afterwards around whichever word happened to rank well — one word such as
`local` ranking highly does not show the model represents lexical *scope*, which
is exactly what the positional controls exist to catch. A **null** means only
that the published linear token-indexed J/R lenses do not surface these concepts
at these positions; it does not contradict the probe or DAS evidence, which read
a different object by a different method.


# 7. Statistics, gates and reproducibility

## 7.1 Uncertainty

**Cluster bootstrap over source programs**, never over rows. Rows from one
program share hidden vectors, so a row-level bootstrap gives intervals that are
too narrow — in the direction that makes a null look like a finding. Control
comparisons are **paired on the same rows** so that the difference, not each
arm separately, carries the interval.

Where a quantity is heavy-tailed, the **median and the sign** are reported beside
the mean and its paired uncertainty rather than hidden by one summary statistic.

## 7.2 Calibration/test separation

Layer and site are chosen on a **calibration** split and recorded before any test
number is read. A site picked after seeing the test split is a maximum, not a
site. Frozen artifacts — probes, lenses, subspaces, discovered token sets — are
written to disk on the calibration side and **read back from disk** on the
evaluation side, so the separation is a filesystem boundary rather than a
promise.

## 7.3 Gates

Two strengths.

**Weak gates** exit non-zero on a failed check; later stages are not
interpretable until they pass.

**Hard gates** declare their prerequisites and **refuse to run** (exit 2) unless
those have passed, whoever invokes them and in whatever order.
`--override-gate REASON` is permitted for diagnostics and is recorded permanently
in `gates.yaml`, in the run manifest, and in **every output row**, so a number
produced under an override can never later be mistaken for one produced under a
passing gate.

That mechanism exists because of a specific failure: a swap stage once ran
without its predecessor's frozen probes on disk and **silently skipped a control**
rather than refusing.

**A passing gate is not a scientific claim.** It says the measurement works at
that step. Gates are deliberately mechanical where the scientific result may be
null; a valid instrument must still pass when no semantic effect is found.

## 7.4 Reproducibility

- **Seed 42 everywhere** by default (generator, CV splits, subsampling,
  bootstrap).
- **Every stage writes a manifest** (`results/manifests/`) recording the git SHA,
  the arguments and wall-clock time.
- **All figures and tables regenerate from the tidy CSVs alone** (stage 90), so
  the chain from raw data to published figure is auditable end to end.
- **Do not re-run a generation stage to "refresh" anything.** Regenerating
  redraws every random transformation and changes every downstream number.
