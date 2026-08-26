# Methods

## How to read this document

This document explains exactly how each experiment was run. It starts by
defining what counts as a semantic representation, then explains how program
structure becomes exact token-level labels, and finally describes the four
measurement instruments. Each instrument answers a different question:

- a **linear probe** asks whether information is present in a hidden state;
- **frozen transfer** asks whether the same representation survives a program
  rewrite;
- the **lens stack** asks whether the information is aligned with the model's
  own output vocabulary and where relevance is routed; and
- **DAS interchange** asks the causal question: whether changing only the
  learned subspace changes the model's downstream answer.

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
closing those loopholes, and the last two sections are about the two instruments
that go beyond decoding entirely.

### Contents

- [§0 What "semantic" means here](#0-what-semantic-means-here)
- [§1 The program-structure layer: the code property graph](#1-the-program-structure-layer-the-code-property-graph)
- [§2 From graph to token: alignment, ground truth, integrity](#2-from-graph-to-token-alignment-ground-truth-integrity)
- [§3 Instrument 1 — linear probes and their floors](#3-instrument-1--linear-probes-and-their-floors)
- [§4 Instrument 2 — frozen transfer and the obfuscation ladder](#4-instrument-2--frozen-transfer-and-the-obfuscation-ladder)
- [§5 The security benchmark (E15): construction, threat model, metrics](#5-the-security-benchmark-e15-construction-threat-model-metrics)
- [§6 Instrument 3 — the lens stack: logit, J-lens, R-lens](#6-instrument-3--the-lens-stack-logit-j-lens-r-lens)
- [§7 Reading the lens as a contrast, and the three ways a null can be wrong](#7-reading-the-lens-as-a-contrast-and-the-three-ways-a-null-can-be-wrong)
- [§8 Instrument 4 — DAS: magnitude-free interchange on a learned subspace](#8-instrument-4--das-magnitude-free-interchange-on-a-learned-subspace)
- [§9 Statistics, gates and reproducibility](#9-statistics-gates-and-reproducibility)

---

# 0. What "semantic" means here

This word carries the whole project, so it is defined operationally rather than
gestured at.

## 0.1 Two notions, both called semantic

| | What it is | Ground truth from | Used by |
|---|---|---|---|
| **Abstract semantics** | sound approximations of behaviour: reaching definitions, def–use edges, control dependence, taint reachability | the code property graph (`src/graphs/`) | E2, E3, E4, E15 |
| **Concrete semantics** | what the program actually computes when run | execution — `execute_program`, observational equivalence, `interpret_scoped` | E9, E13, E15 label recovery |

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
baselines in §3.4 (a ±3-token-id window plus bucketed distance; and, in E15, a
whole-program lexical reader). It is *not* pinned against every computable text
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
| **meaning held fixed** | identity (trivial) | **E9, E15** — the obfuscation ladder |
| **meaning changed** | **E2, E13, E15** — one token, one relation | two unrelated programs (trivial) |

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
   optimising to look benign. E9's and E15's transformations preserve meaning by
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

**PDG — `pdg_extractor.py`.** The union of def–use and control-dependence edges,
which is the minimal structure over which a taint query is well-posed:
`taint_paths(source_line, sink_line)` enumerates simple paths from an untrusted
source node to a sensitive sink node. This is what makes E15's "is the value at
this argument source-derived?" a graph-reachability question with an exact
answer rather than a judgement call.

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
  agree with the intended label (§8.3).
- **Security labels vs instrumented execution *and* a static taint fixpoint.**
  Two independent readings must agree with each other and with the intended
  label, or the program is refused (§5.3).

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

**Position.** For a task about a source-code event (a variable use, a guard, a
sink argument), the state is read at the event's **last covering token** — the
first position whose state can see the whole event under causal attention.
Reading earlier would miss part of the event; reading later folds in unrelated
downstream tokens.

**Cross-model reading is by relative depth, never by index.** The three models
have 24, 32 and 30 layers, so index 11 is 48% of depth in one and 35% in
another. Every result row carries a `relative_depth` column, and cross-model
tables are read at matched depth (E15 reports at 48%).

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

**The whole-program lexical baseline** (E15). Token n-grams plus character
3–5-grams over the **entire** program file, again with no hidden states. The two
readers bound different things: the local one bounds "the identifier gives it
away", the whole-program one bounds "the generator left a shortcut somewhere in
the text".

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
contains every earlier one. E15 therefore applies the same four rewrites **both
individually and composed**, which gives three differences per reported cell:

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

# 5. The security benchmark (E15): construction, threat model, metrics

This is the only part of the project framed around a *security* property rather
than a graph-theoretic relation, and it is the easiest thing here to overstate.
This section says precisely what it does and does not commit to.

## 5.1 The one bit under audit

> Is the value passed to this security-sensitive, **code-bearing** argument
> derived from untrusted input?

That is the whole property. It is a PDG reachability question (§1.1): does a
path exist from an untrusted source node to the sink argument?

It is **not** vulnerability detection — real exploitability depends on path
feasibility, environment, sink semantics and mitigations that live outside this
benchmark. It is **not** malware classification. What it *is* is the one bit a
taint analysis exists to produce, posed so a model's internal states can be
asked for it directly.

Three sink families, each with a real CWE behind it:

| family | sinks | untrusted source |
|---|---|---|
| command execution | `os.system`, `subprocess.call(..., shell=True)` | request / CLI input |
| SQL execution | `cursor.execute` | request input |
| dynamic execution | `eval`, `exec` | request / stdin |

## 5.2 The corpus, and the matched pair

**3 sink families × 4 flow structures × 20 base seeds × 2 labels = 480 clean
programs.** The four flow structures — `direct`, `assign_chain`, `branch_merge`,
`helper` — vary how the value reaches the sink, which is what makes "the readout
tracks flow, not the API name" a testable claim rather than an assumption.

Every base seed yields a **matched unsafe/safe pair** holding the same source,
the same propagation code, the same trusted alternative and the same sink. The
two members differ **only in the sink-argument span**, and that is *checked
character-exactly* (`pair_diff_is_confined_to_sink_arg`), not asserted. Which of
the two chain names carries the taint alternates with the base index, so the
anchor token identity is uninformative about the label across the corpus.

**Split.** 14 seeds per cell train the readout, 6 are held out, split by base so
no base appears on both sides. **Only held-out programs are transformed**, and
the readout is fitted once on clean training programs and frozen (§4.1).

**Ten conditions.** `clean`, `normalize`, four **atomic** arms (`rename_only`,
`opaque_only`, `encode_only`, `flatten_only`) and four **cumulative** arms
(`rename_cumulative` → `rename_opaque` → `rename_opaque_encode` →
`rename_opaque_encode_flatten`). 1296 held-out variants per model; 1776 programs
extracted in total.

## 5.3 Ground truth is never taken from the generator

Two independent readings must agree with each other *and* with the intended
label, or the program is refused:

- **`observe_program`** — instrumented execution. The module runs with
  `__builtins__ = {}` and every sensitive API (`os.system`, `subprocess.*`,
  `cursor.execute`, `eval`, `exec`) replaced by a **recorder**, with a
  provenance-carrying `str` subclass standing in for the untrusted input. This
  reading is flow-sensitive because it *is* the execution. **No dangerous API is
  ever executed**, at any transformation level, even if a generated program were
  wrong.
- **`static_sink_label`** — a flow-insensitive taint fixpoint over the AST,
  written against *call shapes* (attribute paths, not variable names) so it still
  reads a renamed, flattened variant. Interprocedural to one level via
  per-function parameter→return summaries, which the `helper` structure requires.

Being flow-insensitive, the static reading is an over-approximation: it calls a
program unsafe if *any* assignment could carry the source to the sink variable.
That is exactly why it is paired with execution rather than trusted alone — the
two disagree precisely on the programs whose label nobody could defend, and those
are refused.

Label preservation is re-verified per obfuscated variant by the same two
readings.

## 5.4 Why "safe" is a trusted literal, not a sanitizer

The generic sanitizer list used elsewhere in this repository (`html.escape`,
`shlex.quote`) is deliberately **not** reused. `html.escape` before `exec` and
`shlex.quote` before `eval` are not mitigations, so a benchmark whose "safe"
class was built from them would be labelling genuinely vulnerable programs safe —
and every accuracy number downstream would measure the wrong thing.

The safe member instead passes an **independently trusted literal**: a constant
that never touches the source, through the same propagation, to the same sink.

## 5.5 The threat model, stated as a threat model

| | |
|---|---|
| **Defender** | an auditor using a fixed code model and a frozen linear readout |
| **Adversary** | controls the program's *surface form*; applies semantics-preserving transformations; does **not** change the weights or the security label |
| **Protected property** | whether the value at a code-bearing sensitive argument derives from untrusted input |
| **Attack success** | reducing the readout's ability to separate a matched safe/unsafe pair — **especially by producing false negatives** |
| **Out of scope** | executable malware, reflection, dynamic loading, heap aliasing, concurrency, and any adaptive white-box attack optimised against this model or this probe |

The adversary here is **the surface form, not the model**. That is what makes
the matched-pair construction possible, and what makes a *frozen* readout the
right instrument (§4.1).

## 5.6 The metrics are safety metrics, and pooled accuracy is not one

This is the methodological commitment the track exists to demonstrate. An
accuracy of 0.5 has at least two very different causes and the number alone
cannot tell them apart:

- **the information is gone** — the readout gives both members of a pair the
  *same* label, because the position no longer distinguishes them
  (`pairs_same_label` → 1, `frac_predicted_unsafe` collapses toward one class);
- **the information is there and no longer means taint** — the readout still
  splits the pair, but the direction is now arbitrary (`pairs_same_label` stays
  low while accuracy falls to chance).

So every reported cell carries, beside accuracy and its cluster-bootstrap
interval:

| metric | why an auditor needs it |
|---|---|
| `acc_unsafe` / `acc_safe` | a symmetric 0.07 loss and a one-sided 0.24 loss are different failures with the same headline |
| **`false_negative_rate`** | a vulnerable program called safe — the failure direction that matters, named rather than left as `1 − acc_unsafe` |
| `false_positive_rate` | the cost side: an auditor drowning in false alarms stops reading them |
| `frac_predicted_unsafe` | detects collapse onto a class prior |
| **`pairs_same_label`** | the two members differ *only* at the sink argument, so this rising is the sharpest possible evidence that the position stopped carrying the distinction |

This is not pedantry: under the full transformation composition all three models
land within 0.08 of each other while **biasing in opposite directions**, and one
model's entire renaming loss is false negatives. A pooled number reports all of
that as "mostly fine".

---

# 6. Instrument 3 — the lens stack: logit, J-lens, R-lens

## 6.1 What a lens is meant to tell us

A probe asks whether a new classifier can recover a label from a hidden state.
A lens asks a narrower question: **does the hidden state already point in a
direction used by the model's own output system?** This is closer to the model's
computation, but it is still observational. A lens does not show that the model
needs the signal or uses it causally.

The project tested three versions:

| method | plain-language operation | role in the final analysis |
|---|---|---|
| **logit lens** | apply the model's ordinary output head to an intermediate state | sufficient for every surviving vocabulary-space result |
| **J-lens** | estimate how the remaining layers would transform a small change at that state | instrument validated, but no unique semantic result survived |
| **R-lens** | modify the backward calculation so one output score can be divided among earlier token positions | used only for the routing experiment; applicable to the tested DeepSeek architecture, not StarCoder2 |

This distinction is central: **R7 and R8 do not need the J- or R-lens.** Their
conclusions come from the ordinary logit lens. The only question for which the
R-lens adds a capability is R9: where in the input an answer score is assigned.

## 6.2 The logit lens: the baseline that proved sufficient

At layer `l` and position `t`, take the hidden state `h_l,t` and pass it through
the model's normal output head. The result is one score per vocabulary token.
This lets us ask whether a safe/unsafe pair differs in the model's own output
coordinates.

The method is exact at the final layer. Earlier in the network it is only a
readout: it ignores the transformations still to come. That limitation motivated
the two more elaborate lenses. In the actual experiments, however, those lenses
did not alter the vocabulary-space conclusions, so the simpler reading is the
one reported.

For the main full-vocabulary experiment (R7), the procedure is:

1. At a position whose token is identical in the safe and unsafe program, score
   all roughly 32,000 output tokens.
2. Subtract the safe score vector from the unsafe score vector.
3. Average these difference vectors on the training pairs to define one
   safe-to-unsafe direction.
4. Freeze that direction and test whether held-out pairs point the same way.
5. Compare with same-label pairs and the embedding-layer floor, where identical
   tokens must produce an exactly zero difference.

This tests whether a **repeatable output-aligned direction** exists. It does not
test whether a particular word such as `unsafe` represents the concept, nor
whether the direction causes the model's behaviour.

## 6.3 The J-lens: valid instrument, no useful result here

The J-lens adds a local, first-order estimate of the remaining network. In plain
language, it asks: “if this intermediate state changed slightly in this token
direction, how would the final state change?” The estimate is averaged over a
separate corpus so it can be reused.

Two checks show that the implementation works mechanically:

- At the last layer, where no transformer blocks remain, the J-lens must equal
  the logit lens. Its cosine with the logit lens is **1.0000** on all three
  models.
- Before the final layer it recovers the next token better than the plain logit
  lens, improving top-1 recovery by about **0.15–0.22**.

These are validation results, not evidence of semantic understanding. On the
semantic tasks, the J-lens did not yield a result that both survived the controls
and was absent from the logit lens. Its earlier J-space intervention also failed
to isolate a causal value subspace. The J-lens is therefore not used to support
the final semantic claims.

The likely reason is simple: a single averaged linear approximation is a poor
summary of many nonlinear, context-dependent layers. Better next-token recovery
does not guarantee a better readout of an abstract relation such as data flow.

## 6.4 The R-lens: a conserving attribution method

### The problem it addresses

To say that 20% of an answer score belongs to one input position and 10% to
another, the pieces must add back to the original score. Ordinary gradients do
not have this property in a transformer: normalization and multiplicative gates
can shrink or double-count the quantity propagated backward.

The R-lens changes **only the backward calculation**. It leaves the model's
forward activations and output unchanged, but uses layer-wise relevance rules so
that the relevance values approximately satisfy

> `sum of relevance over positions = selected output score`.

The rules freeze normalization and attention-pattern factors during the backward
pass, treat SiLU as an elementwise scaling, and split the relevance of a gated
MLP equally between its two multiplicative branches. Freezing the attention
pattern means the method attributes what attention moved, not why the model
chose to attend there; this is an important limitation for a data-flow task.

### How it was validated

Before semantic results are read, four checks are run:

| check | question |
|---|---|
| forward invariance | did the rules leave the model's actual output unchanged? |
| final-layer equality | does the method reduce to the ordinary logit lens when no blocks remain? |
| improvement over autograd | is conservation closer to 1 at every tested layer? |
| absolute conservation | is the remaining early/middle-layer error small enough to interpret shares? |

Both DeepSeek models pass: the final-layer cosine is **1.0000**, and median
conservation error is **0.0000** for 1.3B and **0.0001** for 6.7B. Removing one
rule at a time shows that the 50/50 split at the gated MLP is the most important
correction. This is a result about the attribution machinery, not about code
semantics.

The method is **not valid for the tested StarCoder2 model**. Its LayerNorm and
non-gated MLP do not match the implemented rules, so the relevant corrections
never attach. The pipeline detects this and refuses to report R-lens semantics
for that architecture.

## 6.5 How the R-lens is used for the semantic test

R9 selects the model's answer score, propagates it backward with the validated
R-lens, and sums relevance by syntactic role: the tainted chain, trusted chain,
sink argument, and so on. Each relevance value is divided by the selected score,
so the role shares form an approximately complete partition.

The safe and unsafe members of a pair differ only at the sink argument. All other
role tokens are identical. A consistent change in relevance at those unchanged
roles would therefore indicate that the model routes the same text differently
depending on which chain reaches the sink.

The experiment has now been run on DeepSeek-Coder 1.3B and 6.7B. It reports both
the median paired shift and the preregistered permutation test of the mean,
because the two statistics can disagree. On 1.3B most pairs shift in the same
direction but the delta distribution is heavy-tailed enough that the mean-based
control does not fire, giving the verdict
`redistribution_consistent_but_not_in_mean`; the statistic that survives there is
the sign, whose exact null under the same random-orientation scheme is a binomial
test. On 6.7B both statistics agree and all five declared checks hold, giving
`redistribution_found`. The routing pattern is therefore a **replicated
observational result on the DeepSeek family**, with its magnitude — 1–2% of the
answer score — rather than its controls as the main limitation.

The two models do not route at the same depth, and this is reported rather than
smoothed. The pattern being located is a paired one — the tainted chain losing
share while the trusted chain gains — and 1.3B shows it at layers 0 and 3, with
the tainted side gone by layer 11 and the trusted side still elevated at 19. On
6.7B layers 0 and 3 do not show it: the two chains move together or sit at
chance. It appears at layer 7, peaks at 11, holds at 15, and is gone by 19.
Within each model both target tokens give the same profile, so the difference is
not an artifact of which output token the relevance is taken for. The method
remains inapplicable to StarCoder2, so this is one architecture family measured
twice, not a cross-family replication.

## 6.5b The same R-lens applied to the binding counterfactual (E16)

R9's construction leaves one thing on the table. Its pair members are
token-identical at the roles it measures but differ at the sink argument, and its
programs are not token-aligned index for index. E13's binding factorial is
tighter on both counts, and E16 reuses it unchanged.

Within one arm of that factorial, `source` and `target` differ at **exactly one
token** out of about twenty-one — the inner definition's *name* — while sharing a
token length, identical anchor positions, and an identical token at the use site.
Those are generation-time invariants (`binding_pairs._finalize`), and stage 140
re-measures them on the encoded prompts rather than inheriting them, because the
whole reading depends on them. So the outer definition, the inner definition's
*value*, the use site, the signature and the answer suffix are all
token-identical at identical indices, and a redistribution among them cannot be
the differing token, a length effect, a tokenisation artifact, or positional
drift.

The relevance is taken for the model's output score of the **bound value**, which
is the quantity the question is about. That means the scored token changes across
a binding flip — and it changes in *opposite* directions in the two arms, because
the factorial crosses binding structure with value assignment. Arm sign agreement
is therefore the output-token control, and it is the same crossing that
identifies the DAS result in §8.5. Two further conditions cost no extra backward
pass: each program is read at both candidate tokens, so `fixed_a` and `fixed_b`
can score *both* members at literally the same token id, removing the output
token from the contrast entirely.

The headline statistic is declared before the run: the inner definition's
token-identical half gaining relevance share minus the wholly token-identical
outer definition losing it. Positive means relevance moved toward the definition
that just came into scope. Four controls run alongside — the token-identical
restriction, the random-orientation permutation null and its exact sign-test
counterpart, two same-binding contrasts where the bound token moves the same way
while the binding does not, and a mismatched-pair recombination — plus a
re-reading structural zero, which is the R-lens analogue of the DAS `noop` arm:
the lens has no dose to zero out, so the available zero is reading the same
program twice and requiring the same fractions.

Behavioural accuracy is a **stratifier here, not a gate**. H1 fails on
deepseek-coder-1.3b (0.809 overall, cell `ab_target` 0.571) and passes at 1.000
on 6.7b, and requiring it would delete the smaller model from a question it can
be asked. The decomposition is well defined whatever the scored token's rank — it
is the partition of *that token's* score — but what it licenses is not, so every
row carries `correct_both` and the shift is reported on all pairs and on the
subset the model answers.

**E16 does not extend the causal claim and is not designed to.** E13's DAS
interchange is the causal benchmark on this corpus; E16 reads a decomposition of
the model's output and intervenes on nothing. The two are different quantities
measured on the same programs, so the report puts them side by side and computes
no ratio between them. What they can jointly support is a conjunction — the
binding is causally transportable at this site *and* the attribution redistributes
with it — or the more interesting disjunction, where the causal fact holds and
the attribution does not move, which would show attribution and use coming apart
on a corpus where the causal question is already settled.

### Two things the first run taught, both about the method

**The share reading needs a positive score, and nothing checked that.**
Conservation (`Σ R_t = s`) is necessary but not sufficient for reading `R_t / s`
as a share. When `s` is near zero the shares explode, and when `s` is *negative*
they invert: a role that supports the answer takes a negative "share". On
deepseek-coder-1.3b **7.56%** of readings have a bound-value score at or below
zero — all of them in the *shadowing* cell, exactly where H1's behavioural
failure sits — and the resulting role fractions run from −517 to +599 while
conservation stays at 1.6e−7. Conservation was doing its job and answering a
different question. A positive-score condition belongs beside it; see
[RESULTS Open items](RESULTS.md#open-items). On 6.7b no reading has a
non-positive score and every share lies in [−0.03, +0.83].

**The mismatched-pair control loses its power on a single-template corpus.** On
E15-D's benchmark, different bases are different programs — different sink
families, different flow structures — so pairing across bases destroys a great
deal and the control is informative. E13's factorial is one template with
substituted names and values, so a mismatched pair *still* contrasts
non-shadowing against shadowing: the semantic contrast survives the mismatch and
only the identifiers and literals are destroyed. On 6.7b the control therefore
reproduces the treatment to four decimals, and gating on it would be a false
negative. What the control still reports, correctly, is that the effect is a
difference of cell population means rather than a per-program quantity — which is
the fact that bounds how the *p*-values should be read, and is why E16's write-up
quotes effect sizes instead.

## 6.6 What these tools can and cannot establish

- A vocabulary lens can show that a distinction is aligned with the output
  basis. It cannot by itself show that the model understands the distinction or
  uses it.
- A meaningful token loading would support lexicalisation. A direction spread
  over many unrelated tokens supports only distributed output alignment.
- A conserving R-lens can describe where an output score is attributed. It does
  not establish causal necessity, and its answer depends on the chosen backward
  rules. On the binding corpus (E16) this distinction is sharper than usual
  rather than softer, because a causal answer already exists there from DAS: a
  relevance shift agreeing with it is not confirmation of it, and a relevance
  shift absent alongside it is not a refutation.
- The attn-rule detaches q and k, so the lens attributes no relevance to
  *pattern formation*. For a binding task, where "attend to the right
  definition" is the plausible mechanism, that is the one thing the instrument
  cannot see, and it belongs in any reading of E16.
- Agreement between logit, J-, and R-lenses does not make a semantic claim
  stronger when the plain logit lens already gives the same result.

The practical status is therefore modest: the logit lens reveals a reliable but
distributed output-space distinction; the J-lens adds no semantic result; and the
R-lens produces one routing result that clears every declared control on 6.7B and
all but the mean-based one on 1.3B, at a magnitude of 1–2% of the answer score
and on one architecture family.

# 7. Reading the lens as a contrast, and the three ways a null can be wrong

## 7.1 Four problems that appear only when a lens scores a pair

Every semantic comparison uses matched safe/unsafe programs and fixes the
orientation as `unsafe − safe`. Scores are z-scored across candidate tokens
within each program before the pair is compared. This removes irrelevant
differences in score scale between positions while preserving which vocabulary
directions are relatively stronger.

Training data may define a direction or choose a layer; evaluation data may not.
The selected direction is written to disk and then applied unchanged to the
held-out pairs.

## 7.2 What the contrast is controlled against

R7 uses the entire output vocabulary, avoiding the main weakness of the earlier
small security-word experiment. Its important controls are:

- **same-label pairs:** two safe programs or two unsafe programs test whether
  ordinary program variation aligns with the learned direction;
- **identical-token floor:** at the chosen position both members have the same
  token, so their embedding-layer difference must be zero;
- **held-out direction test:** the direction is learned on training pairs and
  must orient unseen pairs without refitting;
- **dominance test:** a separate singular-value statistic asks whether the label
  direction is the largest difference between programs, rather than merely a
  consistent one.

Generalisation and dominance answer different questions. A direction can be
small but consistent enough to orient every held-out pair while still failing to
dominate the many other ways two programs differ. That is exactly the observed
outcome.

## 7.3 Three ways a null could be wrong, and the measurement for each

For R9, the non-sink roles contain identical tokens across each pair. Identifier
and source-order swaps test whether a name or location creates the shift. The
analysis reports a sign test, a median shift, and the preregistered permutation
test of the mean.

The permutation test is decisive for the final status. Although most pairs have
the same sign, the mean does not beat randomly reoriented pairs. The result is
kept as a potentially useful pattern but is not promoted to a semantic finding.

## 7.4 Two statistics that disagreed, and why that is not a contradiction

A lens null can mean that the semantic signal is absent, that it is not aligned
with output tokens, or that the lens is unreliable at that layer. Instrument
validation separates the last possibility from the first two. The positive
control in R8 then shows that the vocabulary readout can detect an explicitly
expressed yes/no answer. Together these checks justify the narrow conclusion
that the unprompted property is not concentrated in meaningful security words;
they do not justify saying that the model lacks the property altogether.


# 8. Instrument 4 — DAS: magnitude-free interchange on a learned subspace

Probes and lenses show a fact is *present*, or *present in output coordinates*.
Neither can show it is **used**. A representation can be a faithful shadow of a
computation happening somewhere else, and no amount of decoding distinguishes
the two. Phase III needs an intervention, and the requirements are strict enough
that three earlier designs failed them ([ARCHIVE.md](ARCHIVE.md)).

## 8.1 What a usable intervention must have

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

## 8.2 The interchange operator, and why it has no dose knob

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

## 8.3 How the subspace is learned (DAS), and what a null then means

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
  expressive enough to find structure that is not there. Everything in §8.4 and
  §8.5 exists for that reason.

## 8.4 The controls, and what each refutes

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

## 8.5 The identification: a 2×2 that refutes rather than fails to support

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

## 8.6 The outcome metric, and why it is the argmax

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

## 8.7 The six gates

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

# 9. Statistics, gates and reproducibility

## 9.1 Uncertainty

**Cluster bootstrap over source programs**, never over rows. Rows from one
program share hidden vectors, so a row-level bootstrap gives intervals that are
too narrow — in the direction that makes a null look like a finding. Control
comparisons are **paired on the same rows** so that the difference, not each
arm separately, carries the interval.

Where a quantity is heavy-tailed, the **median and the sign** are the summary,
with the exact binomial null; a mean-based permutation test is reported beside it
rather than instead of it (§7.4).

## 9.2 Calibration/test separation

Layer and site are chosen on a **calibration** split and recorded before any test
number is read. A site picked after seeing the test split is a maximum, not a
site. Frozen artifacts — probes, lenses, subspaces, discovered token sets — are
written to disk on the calibration side and **read back from disk** on the
evaluation side, so the separation is a filesystem boundary rather than a
promise.

## 9.3 Gates

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
that step. Gates are also deliberately *mechanical* where the science may be
null: the lens gates J0/J1 must pass when the semantic result is a null, and no
gate anywhere requires a positive security-token result.

## 9.4 Reproducibility

- **Seed 42 everywhere** by default (generator, CV splits, subsampling,
  bootstrap).
- **Every stage writes a manifest** (`results/manifests/`) recording the git SHA,
  the arguments and wall-clock time.
- **All figures and tables regenerate from the tidy CSVs alone** (stage 90), so
  the chain from raw data to published figure is auditable end to end.
- **Do not re-run a generation stage to "refresh" anything.** Regenerating
  redraws every random transformation and changes every downstream number.
