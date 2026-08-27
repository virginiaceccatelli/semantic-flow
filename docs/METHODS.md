# Methods

## How to read this document

This document explains exactly how each experiment was run. It starts by
defining what counts as a semantic representation, then explains how program
structure becomes exact token-level labels, and finally describes the five
steps of the active argument. Each step answers a different question:

- a **linear probe** asks whether information is present in a hidden state;
- **frozen transfer** asks whether the same representation survives a program
  rewrite;
- **DAS interchange** asks the causal question: whether changing only a learned
  binding component changes the downstream answer;
- the **R-lens** asks the separate observational question: whether the answer
  score is attributed to the definition selected by the binding; and
- the **verbalisation study** asks whether the same binding distinction becomes
  expressible in the model's output vocabulary and forced-choice answers.

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
closing those loopholes. The later sections then distinguish causal use,
attribution, and verbal expression rather than treating them as one claim.

### Contents

- [§0 What "semantic" means here](#0-what-semantic-means-here)
- [§1 The program-structure layer: the code property graph](#1-the-program-structure-layer-the-code-property-graph)
- [§2 From graph to token: alignment, ground truth, integrity](#2-from-graph-to-token-alignment-ground-truth-integrity)
- [§3 Instrument 1 — linear probes and their floors](#3-instrument-1--linear-probes-and-their-floors)
- [§4 Instrument 2 — frozen transfer and the obfuscation ladder](#4-instrument-2--frozen-transfer-and-the-obfuscation-ladder)
- [Part III — From representation to causal use, attribution, and verbalisation](#part-iii--from-representation-to-causal-use-attribution-and-verbalisation)
- [§5 DAS — causal interchange of a binding component](#5-das--causal-interchange-of-a-binding-component)
- [§6 R-lens attribution on the binding programs](#6-r-lens-attribution-on-the-binding-programs)
- [§7 Verbalisation of the binding relation](#7-verbalisation-of-the-binding-relation)
- [§8 Statistics, gates and reproducibility](#8-statistics-gates-and-reproducibility)

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

# Part III — From representation to causal use, attribution, and verbalisation

Parts I and II use probes to establish that binding is represented and to measure
the stability of that representation. Part III follows the representation into
three different consequences on the same controlled programs. DAS asks whether
replacing one learned binding component makes the answer follow the installed
binding. The R-lens leaves the forward computation unchanged and asks whether
the answer score is attributed to the definition selected by that binding. The
verbalisation study then asks whether the same distinction becomes expressible
in a forced-choice answer and in output-aligned scope vocabulary.

DAS comes first because only it licenses the causal claim. The R-lens describes
the unedited answer, and verbalisation describes what the model can express.
Neither lens is used to prove causation, and verbalisation is not treated as
introspection.

The former security benchmark, earlier general output-vocabulary experiments,
standalone J-lens studies, and R-lens taint-routing study are preserved in
[ARCHIVE.md](ARCHIVE.md). They remain reproducible but are not needed for the
active binding argument.

# 5. DAS — causal interchange of a binding component

Probes and lenses show a fact is *present*, or *present in output coordinates*.
Neither can show it is **used**. A representation can be a faithful shadow of a
computation happening somewhere else, and no amount of decoding distinguishes
the two. Phase III needs an intervention, and the requirements are strict enough
that three earlier designs failed them ([ARCHIVE.md](ARCHIVE.md)).

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

| control | construction | what it refutes |
|---|---|---|
| **`whole_state`** | the rank-`d` limit — install the entire donor state | it is the *ceiling*, per arm, and its being alive in both arms is what makes a null in either arm interpretable |
| **`mean_difference`** | rank-1 span of the **mean** donor−host difference; no optimiser, no labels, one fixed direction for every example | the cheapest thing that could work. A learned direction must *dominate* it, not merely beat zero |
| **`answer_direction`** | subtract the J-lens direction for the training arm's current answer from the J-lens direction for its installed answer; keep that direction fixed across arms and scale each edit to the DAS edit norm | tests the simpler account “the learned subspace just pushes toward the answer token required in the fitted arm”; it should work on that arm and fail or reverse when the crossed arm requires the opposite token |
| **`random_norm`** | a random subspace whose interchange moves the **same fraction of ‖h‖** | disruption. Rank-matching alone is not dose-matching |
| **`random_rank`** | a random subspace of the same *rank* | the weaker, rank-matched floor, reported alongside |
| **`noop`** | provably the zero edit | machinery: it must be exactly 0.00e+00 |
| **`def_source` site** | a site where the programs are token-identical *before* the mutation | a structural zero: any effect here is a bug |

**Why `random_norm` and not just `random_rank`.** For an orthogonal projector
only `span(R)` matters, so matching the Gram matrix of the rows says nothing. A
random rank-`r` subspace of a `d`-dimensional stream captures on average `r/d` of
the state's energy, while a learned one is *selected* to capture far more — so
rank-matching alone leaves the two conditions dose-mismatched **in the direction
that manufactures a positive**. The required rank is estimated in closed form
from `E‖R Rᵀ d‖² = (r/d)‖d‖²`, then bracketed exactly; the rank actually reached
is reported, because needing many random dimensions to match one learned
dimension is itself informative.

**Why the answer-direction control is decisive.** In the fitted `ab` arm, the
donor binding changes the required answer from value `a` to value `b`. The
control therefore constructs a direction that explicitly pushes the model from
the `a` output direction toward the `b` output direction. It uses J-lens vectors,
not raw final-layer unembedding rows, because the intervention occurs partway
through the network and the J-lens estimates how the remaining layers map a
change there to the output. A synthetic donor makes the interchange an exact
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

- If both DAS and `answer_direction` worked equally in both arms, the experiment
  could not distinguish binding from an output-token push.
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

## 5.7 The six gates

Each refuses to run downstream stages until it passes.

| gate | asserts | threshold |
|---|---|---|
| **H0** | execution and a scope-aware reference interpreter agree; every invariant holds, **including the arm crossing** | ≥ 0.999 of bases |
| **H1** | the model returns the correctly bound variable | ≥ 0.85 overall, ≥ 0.75 per cell |
| **H2** | which definition is in scope is decodable at the use anchor | ≥ 0.80, and ≥ 0.10 over the *measured* surface baseline |
| **H3** | whole-state interchange flips the answer **in both arms** | CI > 0, flip rate ≥ 0.25 |
| **H4** | low-rank interchange beats matched controls on the **training** arm | ≥ 50% of that arm's ceiling |
| **H5** | the same subspace transfers to the **held-out** arm | ≥ 50% of that arm's ceiling, **and** `answer_direction` fails there |

H1 exists because of a lesson recorded in [ARCHIVE.md](ARCHIVE.md): check the
model can do the task *before* building an instrument on top of it.

---

# 6. R-lens attribution on the binding programs

## 6.1 Why this experiment follows DAS

DAS shows that changing a rank-1 component at the use site changes the answer as
the binding predicts. It does not say which source locations the unedited
model's answer depends on. The R-lens asks that second, observational question
on exactly the same four-program factorial.

This order matters. If the R-lens were the only experiment, a relevance shift
could easily be overread as weak causal evidence. Here the causal fact comes
from DAS. The R-lens contributes a decomposition of the output score, not a
second intervention.

## 6.2 What the R-lens computes

Choose one output score `s`, here the model's score for the value selected by the
program's binding. Ordinary gradients measure local sensitivity but do not add
up to `s`. The R-lens instead modifies only the backward calculation with
layer-wise relevance-propagation rules. The forward activations, logits, and
emitted token remain unchanged.

The backward rules freeze normalization and the attention pattern, treat SiLU as
elementwise scaling, and split the relevance of a gated MLP equally between its
two multiplicative branches. For compatible models, the resulting position
relevances approximately satisfy:

> `sum of relevance over input positions = selected output score`.

Each position's relevance can then be divided by a **positive** selected score to
form a share. The shares are summed into syntactic roles derived from the AST:
outer definition, inner name, inner value, use site, signature, `return`, suffix,
and residual text.

Freezing the attention pattern is an important limitation. The method attributes
what the fixed pattern transports; it does not attribute relevance to how queries
and keys formed that pattern. It therefore cannot establish the mechanism
“attention found the correct definition.”

## 6.3 Instrument checks

The binding analysis is interpreted only after the following checks:

| Check | Required behavior |
|---|---|
| **forward invariance** | installing the backward rules changes no forward output |
| **rules bound** | the normalization, attention, and gated-MLP rules attach to the intended modules |
| **conservation** | relevance across positions sums back to the selected output score at every reported layer |
| **role partition** | every encoded input token belongs to exactly one syntactic role |
| **same-program reread** | reading the same program twice produces exactly zero redistribution |

The implemented rules pass on the tested DeepSeek architectures. They do not
match StarCoder2's LayerNorm and non-gated MLP, so the pipeline refuses to report
R-lens semantics for that model. This is an architecture boundary, not a null
result.

Conservation alone is insufficient when shares are reported. If `s` is zero or
negative, `R_t / s` is unstable or reverses its ordinary interpretation. The
score sign must therefore be checked separately. This condition fails often
enough on DeepSeek-Coder 1.3B that its binding shares are not interpreted.

## 6.4 The one-token binding contrast

The R-lens reuses DAS's two crossed value-assignment arms. Within either arm,
the non-shadowing and shadowing programs differ at exactly one token index out of
roughly 21: the inner definition's name.

```python
x = a                      x = a
def f():                   def f():
    y = b                      x = b
    return x                  return x
# outer binding             # inner binding
```

The tokenizer-level invariants are measured again during the R-lens run rather
than trusted from generation. The outer definition, inner value, use token,
signature, and suffix must be identical and aligned. This makes it possible to
ask whether the changed name reorganizes attribution over text that itself did
not change.

The declared statistic is:

> share gained by the token-identical inner value
> minus share retained by the token-identical outer definition.

A positive value means attribution moved toward the definition that came into
scope.

## 6.5 Controls and what each isolates

| Control | Mechanism isolated |
|---|---|
| **token-identical statistic** | excludes direct relevance at the changed name, length differences, and positional drift |
| **crossed `ab` / `ba` arms** | the scored bound-value token moves in opposite directions; agreement rules out a fixed output-token explanation |
| **`fixed_a` / `fixed_b`** | both programs are scored at the same literal token id, removing output-token identity entirely |
| **competing target** | scoring the value not selected by the binding should reverse the attribution shift |
| **same-binding contrasts** | values change in the same way while binding does not; these should remain flat |
| **random orientation** | randomly reversing pair direction should destroy the signed mean and sign consistency |
| **mismatched bases** | tests whether the effect depends on the exact pairing rather than the two template-level conditions |
| **same-program reread** | must be a structural zero |

The mismatched-base control has limited power on this corpus. Every base shares
one program template and differs mainly in names and literals, so mismatching
still compares non-shadowing with shadowing. Reproducing the treatment under
mismatching therefore bounds the finding to a population-level template
contrast; it does not by itself show that the attribution effect is spurious.

## 6.6 Selection and interpretation

Layers are selected using calibration bases and read once on held-out test
bases. Both arms, fixed-token conditions, competing-target conditions, and
same-binding controls are reported. Effect sizes are preferred to p-values
because the single-template construction makes the many generated bases closer
to repeated measurements of one contrast than to diverse programs.

The output is a layer profile of attribution, not a chronology of computation.
A peak at one layer means that the chosen answer's relevance is most strongly
redistributed there under these backward rules. It does not identify the layer
where binding is first computed.

## 6.7 What the experiment can establish

A controlled positive result supports this statement:

> When the binding changes, the unedited model's answer score is reassigned from
> the definition that becomes inactive toward the definition that becomes
> active, including over definition tokens that did not change.

It does not establish causal necessity, a complete attention mechanism, or
verbalisation. Section 7 therefore uses a different readout: a matched prompt and
an output-space test asking whether the binding relation becomes expressible in
meaningful vocabulary.

# 7. Verbalisation of the binding relation

Stages 150–153 (`src/experiments/binding_verbalisation.py`). Run on
deepseek-coder-6.7b (400 bases, 280 held out) and deepseek-coder-1.3b (200 bases,
140 held out); 47 and 16 minutes. Gates H7, H8, H9 pass in both.
Results: [RESULTS R12](RESULTS.md#r12--verbalisation-the-binding-is-expressed-in-the-models-own-scope-words-late).

## 7.1 The question §6 cannot answer

§5 establishes that the models causally use a binding representation and §6 that
the answer's relevance moves with the binding. Both are read in the model's
internal coordinates. Neither says whether the distinction surfaces in anything
the model *emits*, and that is a separate empirical question rather than a
corollary of either.

It splits into two, and this experiment keeps them apart because they can come
apart in both directions:

    behavioural     asked in words, does the model answer correctly?
    attributional   when the answer IS that word, does relevance sit on the
                    competing definitions the way it does for the value?

A model can answer from a shallow cue while all the relevance sits on the
question text. A model can carry the distinction internally and be unable to name
it. Collapsing the two is the main way to get this wrong, so the report orders
them behaviour-first and the verdict space contains a name for each combination.

## 7.2 The corpus is unchanged, and that is the point

E13's factorial is reused whole — the same four programs per base, the same
frozen calibration/test split, the same anchor positions. Only a question is
appended:

```
z = 6                        z = 6
def f():                     def f():
    d = 3                        z = 3
    return z                     return z
# Question: does f return the z assigned inside f or outside f? Answer:
#                                                     → outside / → inside
```

Every template renders from the **outer** name only — the letter both members of
a pair share — plus the literal `f`. So the rendered question is byte-identical in
all four cells of a base, the full prompt still differs at exactly one token, and
the one-token control §6.4 relies on survives intact. It is re-measured on the
encoded prompts rather than inherited, because appending a question changes the
string the forward pass sees.

A question rendered from the inner name would put the answer in the prompt. H8
and H9 both refuse the run if any rendered question contains a standalone
occurrence of it.

## 7.3 Narrowing the words

A code model that represents "which definition is in scope" could express it in
several unrelated vocabularies, and guessing one is how a vocabulary study
manufactures a null. The lexicon is therefore **matched opposing pairs across
four families**, so a frequency imbalance between the poles cancels in the paired
contrast and a family-level pattern is visible if one exists:

| family | pairs (inner / outer) | what it tests |
|---|---|---|
| scope | local/global, inner/outer, inside/outside, nested/module | the language's own vocabulary |
| shadowing | hidden/visible, masked/exposed | the name of the phenomenon |
| ordinal | second/first, later/earlier, new/original | position, needing no scope concept |
| action | replaced/kept, changed/unchanged | what happened to the binding |

The `ordinal` family is there to be separated from `scope`, not pooled with it:
"the nearest assignment wins" is a positional rule that requires no notion of
scope at all, and a result carried entirely by that family means something weaker
than one carried by `scope`.

Three design choices keep this vocabulary test interpretable. First, pairs are
dropped whole rather than one word at a time. If either pole is not a stable
single token, both poles are removed and the reason is recorded; retaining half
a pair would reintroduce the frequency imbalance that pairing was meant to
cancel. Ten of eleven pairs survived both DeepSeek tokenizers. Only
`masked/exposed` was removed because `masked` is multi-token, so all four word
families remained represented.

Second, encodability was checked before the list was fixed. Candidate words such
as `shadowed`, `reassigned`, `overwritten`, and `rebound` are multi-token on
DeepSeek-Coder and were therefore excluded rather than truncated. A first
sub-token is not treated as though it were the complete word.

Third, a separate non-polar set—words such as `scope`, `binding`, `namespace`,
and `lookup`—tests whether binding vocabulary is generally active against a
random floor. These words are never pooled with the opposing-pole contrast. A
word that rises for both bindings cancels in a paired contrast but would
incorrectly inflate a simple vocabulary-mass measure.

Because a hand-written list is a hypothesis about the model rather than a fact
about it, stage 150 also ranks the **full vocabulary** by its mean paired
logit-lens delta between the two bindings, on **calibration bases only**, with
both arms pooled — pooling makes a token that rises in only one arm cancel rather
than rank, since such a token is tracking the returned literal. The result is
frozen to disk and stage 151 loads it, so the freeze is a filesystem boundary
rather than a promise. The pool is logit-lens-selected, so a direction only a
corrected lens would surface cannot be found this way; that limitation is
inherited from the E15-C design and recorded in the frozen file's provenance.

This step earned its cost. On 6.7B the top-rising tokens under the shadowing
binding at layers 23–31 are ` Inside`, ` inside`, ` Within`, ` interior`,
` inner`, `within`, `ins`, ` dentro` — a coherent insideness cluster across
casings and languages, from a ranking given no lexicon. Only 18 of 432 discovered
rows were in the hand-written list, so most of the evidence for the vocabulary
claim is in words the design did not guess.

## 7.4 The forced choice, and its three controls

Four question styles, each asked in two variants, plus the value control:

| style | choices | why this one |
|---|---|---|
| `scope` (primary) | inside / outside | names the construction, not a technical term, so a model with no word for shadowing can still answer |
| `binding` | inner / outer | the vocabulary §6's roles are named in |
| `pyscope` | local / global | Python's keyword pair — the strongest pretraining prior, and therefore the style most likely to be answered without reading this program |
| `shadow` | yes / no | single tokens under any tokenizer, so it cannot be dropped by lexicon validation |
| `value` | the two literals | **the positive control** — E13's own forced choice |

**Chance is 0.500 by construction.** Within a base the correct answer is "outer"
in two cells and "inner" in the other two, so a model that always answers the
same way scores exactly 0.500. `says_inner_rate` is reported beside accuracy
because only it separates "right half the time" from "always says outer" — and in
the run it did exactly that work: `binding` scored 0.500 with `says_inner` 1.000
(always " inner") and `shadow` scored 0.500 with the polarity variants at 1.000
and 0.000 (always " yes"), so both nulls are diagnosable rather than blank.

The first control changes option order, or polarity for the yes/no question. A
model that merely selects the last-mentioned option will score high in one
variant and low in the other. This control exposed a large wording effect: on
6.7B, the primary `scope` style scored 0.502 in one order and 0.980 in the other;
on 1.3B it became a pure last-mentioned rule. By contrast, `pyscope` exceeded
chance in both orders, at 0.923 and 0.878, so its pooled 0.900 represents stable
performance rather than an average of incompatible behaviors.

The second control compares the two value arms. `ab_source` and `ba_source` have
the same binding but different literals, so their correct scope word is the same
while their correct value differs. Agreement across these arms is evidence that
the word follows the binding rather than the returned literal.

The third control asks for the returned value using E13's original question. It
uses the same harness, bases, cells, and readout position, so it tests whether a
failure on scope words could simply reflect an incapable model or broken
measurement. The control reaches 1.000 on 6.7B, which makes chance performance
on two word styles interpretable as phrasing-specific failures. It reaches only
0.811 on 1.3B, reproducing H1's failure; consequently the word-level null on that
model is uninformative and is labeled `not_verbalised_instrument_untested`.

## 7.5 The R-lens on the word, and the one thing that had to change

Stage 152 is §6's readout with a single substitution: a pole word's unembedding
row as the cotangent instead of a value literal's. Same programs, same conserving
rules, same four contrasts, same nine program roles — plus two, `question_var`
for the variable's mentions inside the question text and `question` for the rest,
because "the model routes its answer through the question rather than through the
definitions" is a live alternative hypothesis and needs its own column.

The change forced by R11's first run is the scored quantity. A raw logit has no
meaningful sign — softmax is shift-invariant, so `s > 0` for one word is a fact
about the arbitrary offset of the logit vector — and `R_t / s` is a share of the
answer only when `s > 0`. R11 lost its 1.3B result to exactly that: 7.56% of
readings at `s ≤ 0`, with conservation holding at 1.6e-7 throughout and noticing
nothing (§6.3 measures completeness, not sign). So the headline here is the
**pole margin**:

    s = logit(inner word) − logit(outer word)

which is shift-invariant, is the quantity the forced choice actually reads, and
whose fractions are invariant under `s → −s` because relevance is linear in the
cotangent — both numerator and denominator flip. The sign problem does not merely
pass here, it cannot arise. And because relevance is linear in the cotangent, the
margin decomposition is exactly `R(inner) − R(outer)` over `s_inner − s_outer`
and costs **no extra backward pass**, the same arithmetic that makes §6.5's
`fixed_*` conditions free. The single-pole readings are still reported, with their
positive-score rate beside them, because they are what shows which pole moved.

### What the run showed about this choice, in both directions

The protection worked where it was designed to. On 1.3B the single-pole scores are
negative on 800/800 readings (median −126), so the conditions that need a positive
score came back `usable = 0` with `positive_layers = []` — the failure that
silently voided R11's 1.3B result, flagged this time before interpretation. On
6.7B both poles score positive on 1600/1600 (median logits 303 and 325), so the
single-pole conditions are licensed there.

But **a guard against dividing by zero is not a guard against ill-conditioning**,
and that is where the margin failed. `MIN_MARGIN_RELATIVE = 1e-6` admits any
margin above about 3e-4 of the pole scale, and the observed margins sit at
|s| ≈ 21 against poles of 303 and 325. The fractions are then a difference of two
near-cancelling large quantities over a small denominator: the median ratio
|s_margin| / max(|s_inner|, |s_outer|) is 0.064 on 6.7B and 0.011 on 1.3B, giving
amplification factors of 15.7× and 88×. The symptoms are unambiguous: mean shifts
up to
0.53 of the answer score, a same-binding control interval of [−0.77, +0.35], and
the mismatched-pair control reproducing the treatment at a ratio of 0.81–0.99 at
every layer. Conservation reported 1.0e-6 throughout and noticed nothing, which
is the same shape of blind spot as R11's sign problem one level further in:
completeness constrains the numerator, and neither completeness nor sign
constrains the *conditioning* of the quotient.

The threshold therefore belongs on |s_margin| / max(|s_inner|, |s_outer|). Stage
153 now measures that ratio per layer, prints it beside conservation and
positivity, and marks the `margin` rows unreadable when no layer clears 0.10 —
which is reporting, not gating: the verdict mapping is still exactly the one
declared before the run, and what the caveat says is that its grounding clause was
*not evaluated*. Moving the threshold into stage 152 as a first-class validity
condition, with a fallback to the single-pole conditions, is
[RESULTS open item 3](RESULTS.md#open-items).

### And the primary style has to be answerable, not just well-motivated

`PRIMARY_STYLE = "scope"` was declared before the run because it names the
construction rather than a technical term. On 6.7B that style turned out to be the
one the model answers with a constant in the wording stage 152 defaults to
(`scope/direct`, 0.502, `says_inner` 0.002), while the style that *is* answered —
`pyscope`, 0.900 in both orders — was never read by the relevance sweep, because
each style costs a full backward sweep and only one runs per invocation.

The declaration was not wrong to make in advance; the ordering was wrong. A style
can only be chosen a priori on how well it *names* the relation, and whether the
model answers it is a fact about the model that stage 151 measures. So the
relevance sweep should take its style from stage 151's calibration behaviour
rather than from a module constant — a selection made on held-out calibration
rows, which is the same discipline that already picks the reported layer.

## 7.6 What the pooled variant row is and is not

The design called the two-variant mean "the bias-free number", and that is right
only when both variants measure the same thing. On 6.7B's `scope` style they did
not: 0.502 and 0.980 average to 0.741, which is neither variant and describes no
behaviour. The pooled row is the right summary when the variants agree and a
warning when they diverge, so both are reported and the spread is what to read.
`argmax_is_a_choice` belongs beside it — it is 1.000 for `pyscope` and 0.000 for
`scope`, meaning neither `scope` choice word is ever the model's own top
continuation, which bounds what its 0.980 shows.

## 7.7 Two controls that swap roles relative to §6

Worth stating explicitly, because reusing §6's tables without noticing this would
mean reading two different controls under one name:

| | §6 (value scored) | §7 (word scored) |
|---|---|---|
| the arms | the **output-token** control: the scored value token moves in opposite directions per arm | a **value-independence** control: the scored token does not move between arms while the literals do |
| fixed conditions | `fixed_a`/`fixed_b`, free but base-dependent | `fixed_inner`/`fixed_outer`, free *and* base-independent — the pole tokens are the same ids in every base, so the control is exact for every contrast; the `margin` condition goes further and scores both members by the same linear *functional* |
| same-binding controls | move the scored token the same way as the treatment | move the *value* while the correct word does not move at all — a sharper test of value contamination |

## 7.8 What each outcome licenses

Declared before the run, in `binding_verbalisation.VERBAL_VERDICTS`:

| outcome | reading |
|---|---|
| words at chance, value at ceiling | the distinction is not verbalised; the instrument is exonerated by the control |
| words above chance, relevance redistributes like §6's | the word is read off the same def-use structure the answer is — the strongest outcome available, still observational |
| words above chance, relevance on the question text | verbalised but not grounded in the program; must not be merged with R11 |
| words above chance, controls fire or arms disagree | the word tracks the literal, not the binding |
| words at chance, relevance still redistributes | the structure is there and the model cannot say it; report as attribution of an unsaid word, not as verbalisation |

### What came out

**6.7B: `verbalised_not_grounded`; 1.3B: `not_verbalised_instrument_untested`.**

The first label needs reading with care, and RESULTS R12 says so rather than
quoting it. The behavioural half is unambiguous — `pyscope` at 0.900 in both
orders against a constructed 0.500 floor, with the positive control at 1.000 — and
the vocabulary half is strong and arm-replicated. What is *not* established is the
grounding, because the two conditions that would have settled it both failed for
mechanical reasons above: the declared headline was ill-conditioned, and the
secondary single-pole condition was read on a wording the model answers with a
constant. `verbalised_not_grounded` is a function of the reported cell, and the
reported cell is the one layer where the ill-conditioned statistic happened to
change sign. The honest statement is that R12's attribution half is unresolved
and one re-run resolves it.

One further defect the run exposed in the machinery rather than in the model:
`select_verbal_cell` maximises the mean, so with a negative effect at every layer
it selected the least-negative one — layer 27, where the effect is not significant
(p = 0.654). A two-sided outcome needs selection on |effect| or on a declared
direction.

**No branch licenses a causal claim.** §5's interchange is the causal instrument;
a word is an output, and attribution of a word is still attribution. Two further
limits belong on every reading. First, answering the question is not
introspection: the question is about the program, and a model can answer it by
reading the text at inference time exactly as a reader would — nothing here
separates a report about the model's own computation from a correct answer about
the code. Second, the attn-rule detaches q and k, so "attend to the right
definition" remains precisely the mechanism this instrument cannot see (§6.2).

## 7.9 The three gates

All mechanical, for the reason §8.3 gives and H6 already follows: the informative
outcome here may well be that the models verbalise nothing, and a gate that made
a null hard to report would be a gate that chose the result.

| gate | stage | what it checks |
|---|---|---|
| **H7** | 150 | every declared pair kept whole or dropped whole with a reason; enough pairs from more than one family; discovery ran on calibration bases only and the frozen file records which; the candidate set holds the lexicon, the discovered pool and the random controls without duplicates |
| **H8** | 151 | every declared (base, cell, question) scored; the rendered question identical in all four cells of a base; no question names the inner definition; both choices distinct single tokens; both variants of every style ran; the value positive control ran |
| **H9** | 152 | H6's checks on the verbalisation prompt, re-measured; the margin condition formed; `fixed_*` really token-fixed; the roles partition every token; the redistribution closes; a deterministic re-read |

Each requires **H0 only**. H1 is deliberately not required — it fails on
deepseek-coder-1.3B at 0.809, and whether a model that answers the *value*
question at 0.809 can answer a *word* question is one of the things this track
exists to measure. Stage 152 does not require H8 either, for the mirror-image
reason: the decomposition is well defined whatever the model answers, and
requiring the behavioural gate would delete `shift_without_verbalisation` from
the verdict space before it could be observed.

---

# 8. Statistics, gates and reproducibility

## 8.1 Uncertainty

**Cluster bootstrap over source programs**, never over rows. Rows from one
program share hidden vectors, so a row-level bootstrap gives intervals that are
too narrow — in the direction that makes a null look like a finding. Control
comparisons are **paired on the same rows** so that the difference, not each
arm separately, carries the interval.

Where a quantity is heavy-tailed, the **median and the sign** are reported beside
the mean and its paired uncertainty rather than hidden by one summary statistic.

## 8.2 Calibration/test separation

Layer and site are chosen on a **calibration** split and recorded before any test
number is read. A site picked after seeing the test split is a maximum, not a
site. Frozen artifacts — probes, lenses, subspaces, discovered token sets — are
written to disk on the calibration side and **read back from disk** on the
evaluation side, so the separation is a filesystem boundary rather than a
promise.

## 8.3 Gates

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

## 8.4 Reproducibility

- **Seed 42 everywhere** by default (generator, CV splits, subsampling,
  bootstrap).
- **Every stage writes a manifest** (`results/manifests/`) recording the git SHA,
  the arguments and wall-clock time.
- **All figures and tables regenerate from the tidy CSVs alone** (stage 90), so
  the chain from raw data to published figure is auditable end to end.
- **Do not re-run a generation stage to "refresh" anything.** Regenerating
  redraws every random transformation and changes every downstream number.
