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
- the **R-lens** asks the separate observational question: whether the answer
  score is attributed to the definition selected by the binding.

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
observational attribution rather than treating them as one claim.

### Contents

- [§0 What "semantic" means here](#0-what-semantic-means-here)
- [§1 The program-structure layer: the code property graph](#1-the-program-structure-layer-the-code-property-graph)
- [§2 From graph to token: alignment, ground truth, integrity](#2-from-graph-to-token-alignment-ground-truth-integrity)
- [§3 Instrument 1 — linear probes and their floors](#3-instrument-1--linear-probes-and-their-floors)
- [§4 Instrument 2 — frozen transfer and the obfuscation ladder](#4-instrument-2--frozen-transfer-and-the-obfuscation-ladder)
- [Part III — From representation to causal use and attribution](#part-iii--from-representation-to-causal-use-and-attribution)
- [§5 DAS — causal interchange of a binding component](#5-das--causal-interchange-of-a-binding-component)
- [§6 R-lens attribution on the binding programs](#6-r-lens-attribution-on-the-binding-programs)
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

# Part III — From representation to causal use and attribution

Parts I and II use probes to establish that binding is represented and to measure
the stability of that representation. Part III follows the representation into
two different consequences on the same controlled programs. DAS asks whether
replacing one learned binding component makes the answer follow the installed
binding. The R-lens leaves the forward computation unchanged and asks whether
the answer score is attributed to the definition selected by that binding.

DAS comes first because only it licenses the causal claim. The R-lens describes
the unedited answer and is not used to prove causation.

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
internal verbalisation. The latter is tested separately in E18.

## 6.8 E18: unprompted lexical expression of the binding state

E18 reads the same unchanged variable-use token with no appended question or
answer suffix. Nine predeclared opposing word pairs cover scope, positional, and
action vocabulary. For each pair, layer, and crossed value arm, the statistic is
the share of held-out programs on which the inner-minus-outer J-lens margin moves
in the predicted direction when only the binding changes.

Raw reversal is descriptive because an arbitrary fixed direction can align with
the near-rank-1 counterfactual displacement. The specificity reference is
therefore 500 independent readouts with the same J-lens row Gram matrix. Each
random direction is scored on the same 280 test bases; calibration and test rates
remain separate. A pair is clear at one layer only if reversal is at least 0.80
in both value arms and its rate is at or above the 99th percentile of matched
directions in both. The verdict additionally requires the same scope pair at two
adjacent entries of the declared layer grid. Pairs are never pooled to decide the
claim. The plain logit lens is reported to distinguish vocabulary alignment from
an effect added by the Jacobian correction.

The positive control is a calibration-fitted binding probe evaluated on the same
held-out states. Its success makes a lexical null informative: the conclusion is
limited to this lexicon, position, model, and linear readout, rather than absence
of the represented binding.


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
