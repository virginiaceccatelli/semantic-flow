# Methods

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

**It is falsifiable, and it has already falsified something.** Control
dependence has a *measured* surface floor of **0.927** — a statement's guard is
usually its nearest enclosing `if`, so token windows plus indentation recover
most of the relation with no model at all. By this criterion control dependence
is *mostly syntactic*, and it is reported as a contrast rather than as a
finding. A definition that never excludes anything is not doing work.

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

## 6.1 The question a lens asks that a probe cannot

Sections 3–5 measure whether a *supervised* readout can recover a relation. A
linear probe can read any direction that happens to correlate with the label,
including one the model never uses, and it chooses its own basis. A **lens**
reads the state through the model's **own output head**, so a high score means
"this state pushes the model toward emitting token `w`".

That makes a different question askable: is the relation in a form the model is
*disposed to act on* — in output-aligned coordinates — not merely one a
classifier can extract? A relation can be **decodable but not verbalised**, and
that gap is the finding the lens track exists to test.

Three lenses are used, in increasing order of faithfulness and cost.

## 6.2 The logit lens

The baseline: score a hidden state `h_l,t` against candidate token `w` by
`v_w = g · W_U[w]`, i.e. project the state directly onto the unembedding row.
Cheap, exact at the last layer, and progressively less meaningful going backwards
because it ignores everything the remaining blocks would do to the state.

It is also the **decisive control** for the J-lens: the J-lens's only claimed
value-add is the causal correction, so a J-lens result that the logit lens
reproduces is not a J-lens result.

## 6.3 The J-lens: a first-order causal correction

For a candidate token `w`, the J-lens vector is

> `v_w = J_l^T (g · W_U[w])`,  where  `J_l = E[ ∂h_final,t' / ∂h_l,t ]`

computed as **one vector–Jacobian product per candidate** — backpropagate the
scalar `(g · W_U[w]) · h_final,t'` back to `h_l,t` — and averaged over a corpus.
The `d_model × d_model` Jacobian is never materialised; only the handful of lens
vectors actually scored against.

Three implementation facts that matter:

- **A small candidate vocabulary is legitimate here** (in the binding track): the
  generator draws every identifier from a fixed pool, and a forced-choice taint
  readout needs exactly two tokens. This is a property of the corpus, not an
  approximation of the method. Where it *is* a limitation — E15-C's 196-token
  pool — §7 says so and §7.2 removes it.
- **Identifiers are read space-prefixed.** Under byte-BPE, `    x = 5` tokenizes
  as `['   ', ' x', ' = ', '5']`, so the token the model actually emits for that
  variable is `' x'`. A lens on the bare `'x'` row would describe a token that
  essentially never occurs in Python source.
- **The scale caveat.** `norm(x) = g·x / rms(x)`, and `rms(x) > 0` does not
  depend on `w`, so dropping it leaves rankings, argmax and the *sign* of a score
  difference exact while making raw magnitudes incomparable across positions.
  Every lens statistic is therefore rank- or sign-based, or z-scored (§7.1).

**How we know the implementation is right.** At the last decoder layer `J` is
provably the identity, so the J-lens must equal the logit lens exactly. Stage 60
asserts this and measures **cosine 1.0000** on all three models — a closed-form
check of the entire gradient path, not a plausibility argument. Next-token top-1
recovery is 0.633–0.650 against a chance level of 0.038, and the Jacobian
correction beats the plain logit lens by +0.15 to +0.22 at pre-final layers.

**Numerical care.** The lens stages run the only *backward* passes in the
pipeline, so they are the only ones exposed to fp16 gradient overflow/underflow.
Each sample is retried down a ladder of loss scales, and any sample still
non-finite is dropped and counted rather than averaged in. On Apple silicon the
fp16 backward through this path returns non-finite gradients at every scale, so
`--dtype float32` is required there.

## 6.4 The R-lens: what it is, what it fixes, and how we know

### The problem the R-lens solves

The J-lens is an **averaged first-order** readout, and its backward pass is raw
autograd through modules that are **not degree-1 homogeneous**. The consequence
is not a vague loss of accuracy — it is measurable, and it has a name.

**Relevance conservation** is the completeness property any relevance
decomposition must have:

> `ρ_l = Σ_t ⟨ ∂s/∂h_l,t , h_l,t ⟩ / s`

which equals **exactly 1** when the tail of the network above layer `l` is
degree-1 homogeneous, by Euler's identity for homogeneous functions. Under raw
autograd `ρ` wanders and **inverts sign** with depth — on a reference
architecture it runs 3.15 / −1.99 / 0.67 across depth. That is the mechanism
behind the non-monotonic J-lens curves in `jlens_validation_*.csv`: the backward
pass is not conserving anything, so a mid-layer reading has no fixed
interpretation.

### What the R-lens actually does

The R-lens (`src/models/lrp.py`) installs four **layerwise-relevance-propagation
rules** on module *instances* inside a context manager, and removes them in a
`finally`. Nothing is monkeypatched at class level — a leaked patch would
silently change every later stage in the same process.

| rule | what it changes | why |
|---|---|---|
| **LN-rule** | detach RMSNorm's `1/rms` factor, making the norm a **diagonal** map | RMSNorm's true Jacobian is `(1/rms)(I − h hᵀ/(d·rms²))diag(g)`; the second term subtracts the component *along h itself* — the direction the residual stream actually carries. Applied once it is a mild shrink; composed over 30 blocks it is the "relevance collapse" |
| **identity-rule** | detach the sigmoid factor of SiLU, making it **elementwise** | `silu(g) = g·sigmoid(g)`; detaching the sigmoid makes the activation degree-1 homogeneous in `g` |
| **half-rule** | split a gate's relevance **50/50** instead of double-counting | a gated MLP `up(x) * act(gate(x))` is bilinear, and autograd counts the same relevance through both branches |
| **attn-rule** | detach `q` and `k`, freezing the attention **pattern** | `A(q,k) @ V(x)` is bilinear too — the same failure the half-rule fixes for the MLP, on the one path the original R-lens formulation leaves alone |

**Every rule preserves the forward value.** `silu(g) = g·sigmoid(g)`, so
detaching the sigmoid changes no value; `0.5(ab) + 0.5(ab) = ab`; detaching a
multiplicative scalar changes nothing. **Only the local derivative moves.** That
is what licenses reading an R-lens against hidden states extracted *without* the
rules installed — it is the same model. Preservation is algebraic, not bitwise
(the half-rule replaces one fused multiply with two multiplies and an add), so
the verification uses a tolerance rather than exact equality.

### The attention rule is a deliberate deviation, and it costs something

The published R-lens formulation leaves attention unmodified. On
deepseek-coder-1.3b that does not conserve: measured in fp32, the three-rule
configuration overshoots by ~0.07 of relevance per block traversed, reaching
`ρ = 2.69` across 24 blocks. Detaching `q` and `k` makes the pattern constant,
the block linear in `x` through `V`, and conservation exact.

**What this costs is real and belongs in any write-up.** With `q` and `k`
detached, the lens attributes no relevance to *pattern formation* — only to what
attention moved, not to the decision of where to look. For a binding task, where
"attend to the right definition" is plausibly the mechanism of interest, that is
a genuine limitation. `attn=False` reproduces the published configuration, and
the ablation measures both so the choice stays visible in every run.

### How the R-lens is validated: gate R

Four checks, all of which must pass before any R-lens number is read
(`scripts/110_rlens_validate.py`):

| check | what it asserts | how |
|---|---|---|
| **R0** forward invariance | the rules change no activation | compare ordinary forward logits with and without the rules, within a **relative** tolerance (deepseek logits reach ~80, where an absolute 1e-4 bound would fail on float32 rounding alone) |
| **R1** last layer = logit lens | regression guard | at the final layer the LRP path is not traversed, so cosine must be 1.0000 |
| **R2a** LRP beats raw autograd | the rules help, at *every* testable layer | median \|ρ−1\| lower under LRP than under autograd, layer by layer |
| **R2b** conservation in early layers | the estimator is sound where it is used | median \|ρ−1\| over the early/middle layer set below threshold |
| **R2c** rule ablation | *which* rule does the work | remove one rule at a time and measure the resulting \|ρ−1\| — reported, not gated |

R2c is the one that reports rather than gates, because its purpose is
attribution rather than validation.

### Architecture scope, and a diagnostic worth knowing

The rules bind by **architecture match**: `norm_eps_attr` identifies RMSNorm,
`is_gated_mlp` identifies a gated MLP. On a model with **LayerNorm** (which
subtracts the mean, so the rule's algebra differs) and a **non-gated MLP**, both
homogenising rules bind to *nothing*. Only the attention hooks register — enough
to satisfy a naive "rules installed" check — and a lens labelled `rlens` gets
built that is arithmetically a **J-lens**.

The tell is diagnostic and general: **an R0 forward delta of exactly 0.0.**
Value-preserving rules still perturb float arithmetic; rules that were never
installed do not. A perfectly passing R0 is the signature of an empty install.
Gate `J0` now records how many modules each rule bound to and refuses an R-lens
where neither homogenising rule matched (`rlens_rules_bound`).

### Which lens to use where

Near the last layer all three coincide by construction. Below that, prefer the
R-lens — and **say so before looking at results**: E15-C declares
`PRIMARY_LENS = "rlens"` in code for exactly this reason, because the target
includes early and middle layers, which is where the J-lens backward is least
faithful. Report all three anyway: their *agreement* is itself evidence, and
their *disagreement* localises an instrument problem rather than a finding.

## 6.5 Lens fidelity is a diagnostic, never a gate

Next-token recovery, agreement with the final-layer distribution, relevance
conservation and the random/Gram-matched floors are measured per (layer, lens),
emit warnings, and **never block execution**. A test asserts that no gate
function reads any fidelity variable.

The reason is selection: refusing to run at low-fidelity layers would silently
restrict every lens experiment to the layers where the instrument is
comfortable, and early and middle layers are usually the target. Reports
therefore distinguish four outcomes — *mechanically invalid*, *mechanically valid
with weak lens fidelity*, *valid null*, and *positive above controls* — rather
than collapsing them into pass/fail.

---

# 7. Reading the lens as a contrast, and the three ways a null can be wrong

## 7.1 Four problems that appear only when a lens scores a *pair*

Earlier lens work scores one state against a candidate vocabulary. E15-C scores
a **matched pair** and takes the difference, which introduces four problems.

**1. Orientation must be fixed once.** Every pair is oriented
`delta(pair, token) = score_unsafe(token) − score_safe(token)`, recorded on every
output row, and gate J1 refuses a run whose rows disagree. A per-cell orientation
choice would make every sign statistic meaningless.

**2. The scale caveat now bites.** A paired contrast compares *two different
positions*, and the J/R lenses drop a positive per-position factor (§6.3). Every
statistic is therefore carried in three conventions:

| convention | meaning | exactness |
|---|---|---|
| `score` | raw lens score | exact for the logit lens, scale-carrying for J/R |
| `z` | z-scored across the candidate set at that position | **exactly** invariant to the dropped factor — the scale-safe way to compare positions |
| `prob` | softmax over the candidate set | exact for the logit lens; inherits the factor for J/R |

**And the convention is checked, not assumed.** Z-scoring removes a *shared*
scale factor but not a systematic difference in the distribution's *shape*
between the two members — and such a difference would move the contrast in a
fixed direction with no concept involved. Stage 126 therefore records each
member's candidate-distribution entropy and score-vector norm, and stage 127
correlates them with the contrast per pair.

**3. The candidate vocabulary cannot be the whole vocabulary.** A J/R lens vector
is one vector–Jacobian product *per candidate token*, so a 32k-row lens at every
layer is infeasible, not merely slow. Discovery is two-phase: a full-vocabulary
**logit-lens** ranking on clean *training* pairs selects a candidate pool (196
tokens), then each lens ranks that pool by its own training delta. The limitation
this creates is recorded inside the frozen artifact: *a direction only the J- or
R-lens would surface, on a token outside the pool, cannot be discovered here.*

**4. Discovery must not see the evaluation.** The frozen token set is written to
`vocab/vocab_discovery.json` by stage 125 and **read back from disk** by stage
126 — a filesystem boundary rather than a promise. J1 additionally checks that
the recorded discovery digest is the training split's and differs from the
evaluated split's.

**Concept tokens are validated per model, and nothing is substituted.** A lexicon
word is used only if it encodes to exactly one token in one of the prompt-space
variants (`" word"`, `"word"`, `" word\n"`) *and* decodes back to the variant
that produced it. Every omission is recorded with its reason. Coverage is
genuinely model-specific: `" vulnerable"` survives on both deepseek models but
`unsafe`, `untrusted` and `tainted` all split; on starcoder2-3b `" unsafe"`
survives and `vulnerable` does not. The first token of a split word is a
*prefix*, not the word, and using it would measure the prefix.

## 7.2 What the contrast is controlled against

| control | what it rules out |
|---|---|
| **permutation** — re-orient each base at random | that anything other than the safe→unsafe *direction* carries the effect; keeps every pair and magnitude |
| **same-label pairs** — both members drawn from one pole | that the effect is any difference between two programs of this kind. Expected contrast is exactly zero, so this is the arm a *label* claim must clear |
| **embedding layer (−1)** | token identity: at `sink_arg` the state *is* the anchor token's embedding, so this is the token-identity contrast exactly |
| **`last_token` site** | ditto from the other side — both members carry the *same* token there, so the floor is exactly zero |
| **identifier-role strata** | that the generator's tainted/trusted name assignment drives the sign |
| **random and Gram-matched lenses** | that any direction of that norm — or of that norm *and* those pairwise angles — would separate the pairs |

**A control that was replaced.** The original `mismatched_pairs` arm redrew the
*safe* partner from the same safe pool, so the label difference survived it: the
arm averages over the very set the main arm averages over, and its expected mean
is the main arm's exactly. Measured, the two agree to four decimal places on all
three models. It falsifies "specific to this pairing", not "about the label", and
the same-label arm above replaces it. Stage 127 reports `pairing_gain` — what
base matching actually buys — as a number.

**What licenses a semantic reading.** All of: train-only discovery frozen before
scoring; held-out replication in the *hypothesised direction* (deliberately
one-sided — a two-sided test would report a consistently reversed contrast as a
positive result); one recorded orientation; an effect above both the permutation
and the same-label control; stability across identifier roles; and evidence not
reducible to the differing sink-argument token. **"The token `unsafe` appeared in
a top-k list" is not a result** and is not allowed to become one.

## 7.3 Three ways a null could be wrong, and the measurement for each

A null needs different evidence from a positive result. Every control in §7.2 is
*negative*: they establish that a positive result is not an artifact and are
**silent about a null**. Nothing there separates "the models do not verbalise
this" from "this readout could not detect verbalisation if it were there". Three
measurements address that.

**(a) The candidate pool could have missed it — stage 128.** Remove the pool
entirely: form each matched pair's difference over the **whole vocabulary**,
z-scored per member exactly as in §7.1, estimate the mean direction on the
*training* split, and project held-out pairs onto it. A null here cannot be
blamed on a pool because there is no pool. Two statistics, answering different
questions:

- **projection** — does a label-defined direction *generalise* to unseen
  programs? (sign consistency of held-out projections, cluster-bootstrapped);
- **concentration** — is the label axis the *dominant* axis of variation?

  `sv1_share = λ_max(U Uᵀ) / trace(U Uᵀ)`, `U` the unit-normalised differences,

  which is `1/n` for unrelated differences and 1 for identical ones, and is
  *sign-invariant* — which is what lets it be compared against a null whose
  members have no canonical orientation.

The primary site is **`last_token`**, not `sink_arg`, because it is the only site
where both members carry the same token id in 100% of pairs; layer −1 is the
explicit surface floor, and at `last_token` that floor is **exactly zero**, since
identical tokens give identical embeddings.

**(b) The readout could be blind — stage 129, the positive control.** Run the
*identical* measurement on a property the models demonstrably answer: a
forced-choice taint question whose answer is a single token. One candidate basis
carries both properties, and both contrasts go through the same `pair_contrast`
call in the same convention with the same orientation, so the two readouts differ
**only** in which token positions are named as poles — gate J3 refuses the run
otherwise. The model's own forced-choice margin is recorded per program, so
`lens_tracks_model` separates "the lens sees what the model says" from "the lens
sees something".

The behavioural statistic is **`pair_separation`** — the fraction of bases where
the unsafe member draws a higher yes-margin than its matched safe counterpart —
not accuracy, because a model that always answers "no" scores 0.5 accuracy for
free while pair separation's chance level of 0.5 is immune to answer bias. Two
prompt styles are run, so prompt sensitivity is measured rather than assumed.

Four outcomes are declared in advance, **including the one that would retire the
track**: if the model answers and the lens does not see it, the null is about the
method.

**(c) The distinction might not be lexicalised at all — stage 130.** Both
readouts above require the concept to live in vocabulary space. Under the LRP
rules the tail is degree-1 homogeneous, so `Σ_t R_t = s` and **`R_t/s` is a
partition of the model's own answer across input positions**. Two programs give
different scores, so raw relevances are not comparable — but fractions are, and
because they sum to one in both members, a paired difference is a genuine
**redistribution** rather than a change of scale. That is a property the
vocabulary readout never had.

Relevance is aggregated by **AST role**, recomputed from each variant's own
source. The control comes free: **only `sink_arg` differs in tokens between the
two members** — enforced at generation time in every condition, and verified
across all held-out programs to give identical per-role token counts within every
pair. A shift among the token-identical roles therefore has no surface account.
Conservation is measured per (pair, layer) and the reading is refused where it
does not hold — including outright on architectures where the homogenising rules
bind to nothing (§6.4).

## 7.4 Two statistics that disagreed, and why that is not a contradiction

Both stages that fired reported a pre-declared criterion that *failed* beside one
that passed. In both cases the two answer different questions.

- **Stage 128**: the projection asks whether a label-defined direction
  *generalises*; `sv1_share` asks whether the label axis *dominates* the
  difference vectors. The first passed and the second failed, because two
  programs of the same label already differ along a shared axis of comparable
  size. That they are *different axes* is established separately: the two
  directions' top-100 loadings overlap at a Jaccard index of ≈0.
- **Stage 130**: `sign_consistency` and its exact binomial null fired at
  p ≤ 4e-11 while the **mean's** permutation null did not, because relevance
  deltas are heavy-tailed enough that a handful of outlier pairs flip the mean
  while the median holds.

In both cases both numbers are reported, the verdict label names what actually
happened, and **the median — not the mean — is the summary to read for a
heavy-tailed quantity.**

---

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
| **`answer_direction`** | an explicit output-aligned answer direction, norm-matched to the treatment | the positive control **for the falsification itself**: it *must* pass on the fitted arm and *must* fail on the held-out arm |
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
