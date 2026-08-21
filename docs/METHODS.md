# Methods

This file explains the methodology behind every number in `results/tables/`.
Each section states **what** we do, **why** it is necessary, and **how** it
works, so a reader can judge the claims without reading the code. It is written
to be lifted into the paper's Methods section.

The overall logic: we extract a model's internal hidden states while it reads a
program, and train a simple linear classifier ("probe") to predict a semantic
fact about that program (e.g. "are these two tokens the same variable?"). If a
*simple* probe can recover the fact, the model must already represent it. The
hard part is making sure the probe is reading the *model's computation* and not
some shortcut in the text — most of this document is about closing those
loopholes.

---

## 0. What "semantic" means here

This word carries the whole project, and until now it was only implicit in the
constructions. Stated explicitly.

### 0.1 Two notions, both called semantic

| | What it is | Ground truth | Used by |
|---|---|---|---|
| **Abstract semantics** | sound approximations of behaviour: reaching definitions, def-use edges, control dependence, taint | static analysis (`src/graphs/`) | E2, E3, E4, E6, E7 |
| **Concrete semantics** | what the program actually computes when run | execution — `execute_program`, observational equivalence, `interpret_scoped` | E9, E11, E12, E13 |

These are different levels and the project uses both. A claim about one is not
automatically a claim about the other: E2 says the model tracks *which
definition reaches a use* (abstract), E13 asks whether it causally uses *which
value that use resolves to* (concrete).

### 0.2 The working definition

Neither table entry is what the word does methodologically. The operative
definition is negative, and it is **enforced rather than estimated**:

> A property of a program is **semantic** to the extent that a bounded reader of
> the program's surface form cannot recover it.

Operationally: construct program pairs in which every feature such a reader can
see is held identical while the property flips. The model-free baseline then
scores **exactly 0.500 by construction**, not approximately (§7). Anything above
that floor is something the model computed rather than read off the page.

Two consequences worth stating, because they are what make the definition
usable rather than rhetorical:

**It is falsifiable, and it has already falsified something.** Control
dependence (E4) has a measured surface floor of 0.927 — a statement's guard is
usually its nearest enclosing `if`, so token windows plus distance recover most
of it. By this criterion control dependence is *mostly syntactic*, and E4 was
demoted to "supporting, not central" for exactly that reason. A definition that
never excludes anything is not doing work.

**"Surface" is relative to a stated reader.** The floor is pinned against the
baseline in §7 (±3 token ids plus bucketed distance). It is not pinned against
*every* computable text feature — a cross-position string-equality feature is
outside that window and is not currently in the baseline. Claims should say
which reader the floor is pinned against.

### 0.3 Semantics requires sensitivity *and* invariance

A property being semantic implies two dual requirements on any representation
that claims to encode it:

- **Sensitivity** — hold the form fixed, change the meaning: the representation
  must change. (E2's `context_matched`; E13's two arms.)
- **Invariance** — change the form, hold the meaning fixed: the representation
  must not care. (E9's obfuscation ladder, every variant execution-verified
  observationally equivalent to its base.)

Only one of the two is cheap to satisfy. A representation with invariance alone
is a hash of behaviour; with sensitivity alone it is a diff of text. Semantics
is the conjunction, and the conjunction is measurable: E9 found mid layers hold
invariance under identifier renaming (0.85–0.90) and lose it under control-flow
flattening (0.750), while E2 found sensitivity holds against a 0.500 floor.

The four cells make the design space explicit:

|  | **form held fixed** | **form changed** |
|---|---|---|
| **meaning held fixed** | identity (trivial) | **E9** — obfuscation ladder |
| **meaning changed** | **E2, E13** — one token, one relation | two unrelated programs (trivial) |

Note what the bottom-left cell contains: only *one-token, one-relation*
instances. A general form-preserving, meaning-breaking construction — a program
that presents as one thing and computes another — does not exist in this
repository. It is the dual of `src/data/obfuscation.py` and it is the cell
closest to the adversarial motivation.

### 0.4 What this definition does not yet reach

Three limits, stated so that results are not read past them.

1. **Decidability.** Every relation studied here is exactly decidable by static
   analysis or by execution. That is deliberate — approximate labels become
   label noise, and label noise becomes the finding. "Semantic" therefore means
   *decidably semantic on a restricted fragment*. §0.5 is about escaping this.
2. **Scale.** Binding and def-use are single-relation facts at token positions.
   Whole-program behaviour ("does this sort, or exfiltrate?") is a different
   object, not a larger one.
3. **Adversariality.** Nothing in the corpus is written by an adversary
   optimizing to look benign. E9's transformations preserve meaning by design;
   the security-relevant case breaks meaning while preserving appearance.

### 0.5 Undecidable properties are not out of reach

Undecidability is a property of a problem *class* quantified over all programs,
not of a concrete instance: a given finite program either aliases on a given
input or it does not. So "can we study undecidable semantics" decomposes into
three separable questions, each with a workable method.

**(a) Relations undecidable in general, known by construction.** The generator
authors the program, so it knows the answer without deciding anything. May-alias
is undecidable in general; in

```python
if int(input()) > 0: r = p
else:                r = q
r[0] = 9
```

the points-to set at the write is `{p, q}` by construction and confirmable by
bounded enumeration. This is the same move E2 already makes for binding, and it
inherits the same defence: the floor, not the difficulty of the relation, is
what licenses the claim. The honest limitation is that the *instances* are easy
even though the *class* is hard.

**(b) The uncertainty undecidability forces — the sharper question.** Rather than
asking a model to decide the undecidable, ask whether it represents the
tri-partition a sound analyser is obliged to make:

| class | holds on | a sound analyser must say |
|---|---|---|
| **must** | every feasible path | yes |
| **must-not** | no feasible path | no |
| **may** | some paths, not others | *both are possible* |

All three are exactly constructible, so labels are exact. The question becomes:
does the model's state separate `may` from `must` and `must-not`, or does it
collapse `may` into whichever of the two it lexically resembles? A model doing
something analysis-like represents the third class distinctly; a pattern matcher
cannot. This tests representation *of the undecidability itself* and requires no
oracle. The relevant threat is that transformers linearly represent
distributions over states generically (Shai et al.,
[arXiv:2405.15943](https://arxiv.org/abs/2405.15943)), so a two-element mixture
at a merge point may be the expected result rather than evidence of a join —
which is why exclusion of a *provably infeasible* third value, and collapse
under a strong update, are the auxiliary predictions that separate "a set" from
"uncertainty".

**(c) Instances that are hard, decided by a solver.** For path conditions,
alias queries and equivalence on real fragments, an SMT solver or symbolic
executor decides many instances that no syntactic analysis can. Keep the decided
subset, **report the decided fraction**, and stratify by time-to-solve. The
difficulty gradient is itself a measurement: decodability flat in solver time
says the model is not doing anything search-like; decodability that decays with
it says something more interesting. The threat is selection bias — solver-decided
instances may be systematically easier — and it is handled by reporting rather
than by assumption.

**(d) No labels at all — equivalence classes instead.** Where a property cannot
be labelled, it can still induce an equivalence relation that is *sampleable*.
`obfuscation.semantically_equivalent` already provides execution-verified
equivalent variants at five levels of surface distortion. One can then ask
whether a program and its flattened twin occupy nearer states than two
surface-similar programs that compute different things — a representational-
similarity question that needs no per-token labels, with the ladder itself as
the surface control.

Routes (a) and (d) are available with what is already in this repository; (b)
and (c) need new generation and, for (c), a solver dependency.

---

## 1. The probes

**What.** Every result uses a single, deliberately weak classifier: **logistic
regression** (`C=0.1`, class-balanced) on standardized features. No neural-net
probes.

**Why weak on purpose.** A high-capacity probe (e.g. an MLP) can *learn* the
semantic relation itself from raw activations. If that happens, "the fact is
decodable" tells you about the probe's power, not the model's representation. A
linear probe can only read features the model has already made linearly
available — so "decodable" stays a statement about the model.

**How the inputs are built.**
- **Single-position tasks** (token type, taint state) probe one token's hidden
  state directly, `h_i`.
- **Pairwise tasks** (binding, def-use, control dependence) ask about a *relation
  between two tokens*, so the probe sees a feature vector that captures both
  tokens and their interaction:
  `[h_i ; h_j ; h_i − h_j ; |h_i − h_j|]` (concatenation, difference, and
  absolute difference).

**Convergence bookkeeping.** Fits use the `saga` solver with `max_iter=2000`,
`tol=1e-3`. Whether each fit actually converged is recorded and appears in every
results row (`converged`). Stage 20 fails its sanity check if any *reported*
fit did not converge, so a non-converged probe can never quietly become a
headline number. (The shuffled-label control fits — Section 5 — routinely hit
the iteration cap by design and are tracked separately as `control_converged`.)

---

## 2. Where in the model we read (layers and positions)

**Layers.** Hidden states are captured at a fixed set of transformer blocks
(spanning input to output) plus one special layer:

- **Layer −1 = the embedding output.** This is the token representation *before
  any attention or context mixing*. It encodes token identity only. We use it as
  the **context-free reference**: anything decodable here is a property of the
  token string, not of the model's reasoning.
- **Layer 0 and up = decoder-block outputs.** Note that "layer 0" is the output
  of the first transformer block, which has *already* mixed context once. This is
  why we extract layer −1 separately — without it, there is no truly context-free
  baseline in the layer sweep.

**Position.** For a task about a source-code event (a variable use, a guard, a
sink argument), we read the hidden state at the event's **last covering token**
— the first position whose state can "see" the whole event under causal
(left-to-right) attention. Reading earlier would miss part of the event; reading
later would fold in unrelated downstream tokens.

---

## 3. Ground truth and token alignment

**What.** The labels the probe is trained against come from **static analysis of
the same program**, not from the model:
- **Def-use chains** from reaching-definition analysis.
- **Control dependence** from the AST (guard nesting, with join points resolved
  exactly so control does not "leak" past where branches merge).
- **Taint state per line**, known because the generator produced the program.

**The alignment problem.** Static analysis speaks in source coordinates ("line
5, column 8"); the model speaks in subword tokens. We have to translate one to
the other exactly, or every label is attached to the wrong hidden state.

**How.** Each source event is located by its **AST span** and mapped to token
indices through a **verified offset table**. Offsets are computed by incremental
prefix decoding — decode the first *n* tokens, see how many characters they
cover, repeat — and the result is checked to **reproduce the source exactly**
before it is used (`src/data/alignment.py`).

**Why not just match token strings.** With a subword vocabulary a variable name
may not be any single token, and string matching silently mislabels *shadowed*
names (two different variables spelled the same) — which is precisely the
phenomenon E2 is trying to measure. String matching would build the shortcut
into the ground truth.

### 3a. Independent cross-check of the ground truth

Because every downstream label depends on our extractor, its def-use edges are
**differentially tested against `beniget`**, a mature, independently written
reaching-definitions analysis (`tests/test_ground_truth_crosscheck.py`). This is
the same "validate the program graph against a second implementation" discipline
that code-property-graph tools (Joern, llvm2cpg) use.

The two analyses answer slightly different questions — ours resolves each use to
the *single most-recent* reaching definition, while beniget returns *all*
possibly-reaching definitions across branches — so the sound comparison is
**set inclusion** (our edges ⊆ beniget's), with **exact equality on
straight-line code**. This check caught a real bug: uses in self-referential
updates like `b = b + a` were being linked to the *same-line* target definition
instead of the prior one. The extractor now resolves reaching definitions in
execution order (right-hand side before the assignment target).

### 3b. Tokenizer integrity

`AutoTokenizer` on transformers 5.x silently resolves deepseek-coder to a slow
SentencePiece path that **mis-tokenizes code** — `def func` becomes
`['de','ff','unc']` with whitespace dropped. Any activations or labels built with
it are garbage. `src/models/loader.py::load_tokenizer` therefore loads via
`PreTrainedTokenizerFast` and **rejects any tokenizer that fails an exact
code round-trip**. All results predating this guard are invalid.

---

## 4. Cross-validation without leakage

**The problem.** Many probe examples come from the *same program* and therefore
share overlapping hidden-state vectors. Ordinary random k-fold cross-validation
would put some rows of a program in training and others in test, letting the
probe memorize program-specific quirks and inflate test accuracy — leakage.

**How we prevent it.** All cross-validation is **`StratifiedGroupKFold` grouped
by source-example id**: every row from one program stays entirely within one
fold. When we cap dataset size for tractability (`max_samples=20000` per
task × layer), we drop **whole groups (programs), never individual rows**, so a
program is either fully in or fully out.

---

## 5. Selectivity control (guarding against "easy" accuracy)

**The problem.** A probe can score high accuracy for boring reasons: class
imbalance, or per-program regularities that correlate with the label. High
accuracy alone is not evidence of a *semantic* representation.

**How.** For every probe we retrain the **identical** classifier on **shuffled
labels** and report:

> `selectivity = accuracy − control_accuracy`

If the real structure matters, the true-label probe should beat the shuffled one;
if the "signal" was just priors and regularities, the shuffled probe matches it
and selectivity ≈ 0. **Claims are made on selectivity, not raw accuracy.**

The shuffle is done carefully:
- For **pairwise/per-token tasks**, labels are shuffled *within* each program, so
  each program's label mix is preserved and only the token→label pairing is
  destroyed.
- For **example-level tasks** where the label is constant across the whole
  program (taint_state), a within-program shuffle would do nothing, so instead
  the **program→label assignment is permuted across programs**.

---

## 6. Negative-sampling strata (the honest headline)

**The problem.** For a relation like binding, most "negative" pairs are trivially
separable from the text alone (two differently-named variables). A probe scoring
well on those is not demonstrating semantic understanding. So we break the
negatives into **strata** and report held-out accuracy for each, from easiest to
hardest:

| Stratum | What it is | What it controls for |
|---|---|---|
| `diff_name` | different variable names (capped at 3× positives) | trivial baseline |
| `distance_matched` | negatives at the same token distance as positives | positional shortcuts |
| `same_name_diff_binding` | same name, different actual binding | the name-identity shortcut |
| **`context_matched`** | **two token-identical programs differing by one binding-flipping character** | **every surface cue at once** |

**Why `context_matched` is the one that matters.** Its two programs are identical
token-for-token except the single character that flips the correct label; the
anchor windows and token distance are identical, and the pair shares one CV
group. By construction, *no* feature of the text can separate the labels — only
something the model computed can. **The honest headline number is
`context_matched` accuracy measured against the surface baseline (Section 7),
not the pooled accuracy across strata.**

---

## 7. The surface-shortcut baseline

**What.** A probe that sees **no hidden states at all** — only the ±3-token
window of token ids around each anchor plus the bucketed distance between them —
fit with the same grouped CV and reported per stratum (`features="surface"`).

**Why it exists.** The first full 1.3b run scored ~0.98 on *every* task and layer,
including the supposedly hard `same_name_diff_binding` stratum at the earliest
layer. This no-model baseline reproduced that ~0.98 — proving the templated
corpus was **leaking labels through local token context**, so the "semantic"
result was a mirage. The baseline is now a permanent floor: **a hidden-state
result only counts if it beats the surface baseline on the same stratum.** By
construction the surface probe scores exactly 0.5 on `context_matched`, which is
why that stratum is the clean one.

---

## 8. Frozen-probe evaluation (E5 context, E9 obfuscation, E15 security flow)

**What.** For robustness experiments we take the probes trained in stage 20 and
**evaluate them, unchanged, on transformed programs** — we never retrain per
condition.

**Atomic vs cumulative conditions (E15).** A transformation ladder applied only
cumulatively cannot attribute a failure: its last rung contains every earlier
one. E15 therefore applies the same four rewrites **both** individually and
composed, giving three differences per reported cell — `delta_clean` (what a
condition costs), `delta_previous` (the marginal cost of the step a cumulative
condition adds) and `delta_atomic` (cumulative minus its atomic counterpart: the
interaction). The interaction is read against a **measured** draw-noise floor
rather than assumed to be zero: two conditions apply the *identical*
transformation under independent draws, and their difference is that floor.
Each variant's transformations are read off its own AST and must equal exactly
what its condition declares, so an arm cannot quietly contain more than it says.

**Why not retrain.** Retraining on each condition would measure how *learnable*
the relation is under that condition, which is a different and easier question.
Freezing the probe measures whether the **representation it already found still
holds up** when the input is stressed — which is the actual research question.

**How truth stays correct.** Some transformations genuinely change the program
graph (competing updates in E5; inserted opaque branches and flattened control
flow in E9). Ground truth is therefore **recomputed for every variant**, so the
frozen probe is always scored against the transformed program's real labels.

**E9's equivalence guarantee.** "Same semantics" is never assumed. Every
obfuscated variant is **executed and checked to be observationally equivalent**
to its base program (the same I/O-equivalence standard Tigress uses). All levels
of a given base program are kept or dropped together, so per-level comparisons
always hold the set of base programs fixed.

---

## 9. Causal claims (E13 interchange)

Probes show a fact is *present*; they cannot show it is *used*. The requirement
for an intervention that shows use is strict, and three earlier designs failed it
(`docs/ARCHIVE.md`). What E13 does differently:

- **Intervene where the programs are token-identical.** The 2×2 factorial crosses
  binding structure with value assignment, and the edit is applied at a position
  whose tokens are the same in both members — so a patched state cannot transport
  the input difference along with the semantic one. This is the failure that
  retired E7.
- **Edit a nameable part of the state, not the state.** A rank-1 subspace fitted
  by DAS, rather than a whole-state swap, so what was installed is specifiable.
- **Measure the dose.** The edit norm is reported beside the effect; an
  intervention below the site's causal dose produces a null that means nothing,
  which is what retired E11.
- **Test on the arm it was never fitted on.** H5 evaluates on the value
  assignment requiring the *opposite* answer-token movement, where a token- or
  answer-direction account predicts failure. Passing there is what separates
  "transported the binding" from "pushed the output".
- **Beat a closed-form baseline.** The difference-in-means direction transports
  too (76%); the learned direction must dominate it, and does, at two-thirds the
  intervention norm.

Gates H0–H5 are recorded per model in `results/binding/{model}/gates.yaml`, and a
stage refuses to run on a failed prerequisite.

---

## 10. The lens track: decodable vs verbalizable (E10-0, E14, E15-C)

**What.** Sections 1–10 all measure whether a *supervised probe* can recover
a relation from the hidden state. E10 adds an **unsupervised** readout built
from the model's own output head — the Jacobian lens — and asks a stronger
question: is the relation in a form the model is *disposed to act on*, not
merely one a classifier can extract?

**Why it is a different question.** A linear probe can read any direction
that happens to correlate with the label, including one the model never
uses. The lens direction is a *causal derivative of the model's own output*,
so a high score means "this state pushes the model toward saying `w`". A
relation can therefore be decodable but not verbalizable — and that gap is
the finding E10 looks for.

**How.** For a candidate token `w` the lens vector is

> `v_w = J_l^T (g * W_U[w])`, where `J_l = E[d h_final,t' / d h_l,t]`

computed as one vector-Jacobian product per candidate — backpropagate the
scalar `(g * W_U[w]) . h_final,t'` to `h_l,t` — and averaged over a corpus.
The `d_model x d_model` Jacobian is never materialized; only the handful of
lens vectors we actually score against.

**Why a small candidate vocabulary is legitimate here.** The paper scores
against a ~32k vocabulary. We do not have to: the generator draws every
identifier from a fixed 26-letter pool (`SAFE_NAMES`), and the taint readout
needs exactly two tokens (`" yes"`, `" no"`). This is a property of the
corpus, not an approximation of the method — and it is what makes E10 cost
about as much as one existing GPU stage rather than a cluster-scale run.

**Identifiers are read space-prefixed.** Under byte-BPE, `    x = 5`
tokenizes as `['   ', ' x', ' = ', '5']`, so the token the model actually
emits for that variable is `' x'`. Candidates are therefore built from the
leading-space variant; a lens on the bare `'x'` row would describe a token
that essentially never occurs in Python source.

**The scale caveat.** `norm(x) = g * x / rms(x)`, and `rms(x) > 0` does not
depend on `w`, so dropping it leaves rankings, argmax, and the *sign* of a
score difference exact while making raw magnitudes incomparable across
positions. Every E10 statistic is rank- or sign-based for this reason.

**How we know the implementation is right.** At the last decoder layer `J`
is the identity, so the J-lens must equal the logit lens exactly. Stage 60
asserts this (V1) and measures cosine 1.0000 — a closed-form check of the
entire gradient path, not a plausibility argument.

**Numerical care.** The lens is the only stage in the pipeline that runs a
*backward* pass, so it is the only one exposed to fp16 gradient
overflow/underflow. Each sample is retried down a ladder of loss scales, and
any sample still non-finite is dropped and counted rather than averaged in.
A high drop count is the signal to re-run with `--dtype float32`.

**Controls (same discipline as Sections 5–7).**

| Control | What it rules out |
|---|---|
| **logit lens** (`v_w = g * W_U[w]`, no Jacobian) | that the unembedding matrix alone explains the result — the decisive control, since the J-lens's only claimed value-add is the causal correction |
| **random lens** (norm-matched directions) | that any direction of that magnitude would rank things similarly |
| **frozen build/eval split** | that the lens was fit to the states it is scored on — lenses are built on a calibration split and frozen, exactly as Section 8 freezes probes |
| **paired layer comparison** | that a noisy floor won by taking its own maximum over ~10 layers; controls are read at the J-lens's best layer, not at their own |

---

## 10a. From J-lens to R-lens: why there are two, and which to trust where

The J-lens above is an *averaged first-order* readout. That approximation is
excellent near the output and progressively worse going backwards, because the
backward pass it relies on is raw autograd through modules that are not
degree-1 homogeneous. E14 measured exactly how bad, using **relevance
conservation** — the completeness property a relevance decomposition must have:

> `rho_l = sum_t <ds/dh_l,t , h_l,t> / s`, which is exactly **1** if the tail of
> the network above layer `l` is degree-1 homogeneous (Euler's identity).

Under raw autograd `rho` wanders — on a reference architecture it runs 3.15 /
−1.99 / 0.67 across depth, **inverting sign**, which is the mechanism behind the
non-monotonic J-lens curves in `results/tables/jlens_validation_*.csv`. The
**R-lens** installs the LRP rules of `src/models/lrp.py` (RMSNorm becomes
diagonal, SiLU elementwise, the gate splits evenly) so the traversed tail *is*
homogeneous, and `rho` holds near 1.

The rules are **value-preserving**: they change no activation, only the backward
graph. That is checked, not assumed — gate R0 / J0 compares the ordinary forward
logits with and without the rules and requires agreement within a *relative*
tolerance (deepseek logits reach ~80, where an absolute 1e-4 bound would fail on
float32 rounding alone).

**Which lens to use where.** Near the last layer they coincide with the logit
lens by construction. Below that, prefer the R-lens, and say so **before**
looking at results — E15-C declares `PRIMARY_LENS = "rlens"` in code for exactly
this reason. Report all three anyway: their *agreement* is itself evidence, and
their *disagreement* localises an instrument problem rather than a finding.

**Measured, gate R (stage 110) on both Llama-family models:** `rho` holds within
**1e-4 at every layer**, and LRP beats raw autograd at 7/7 (1.3B) and 9/9 (6.7B)
testable layers. E14's reference-architecture target reproduces on real models.

**Which rule does the work.** §2.1 of the plan predicted the LN-rule; an early
fp16 run recorded that as half wrong, and these two runs settle it across models
and dtypes:

| rule removed | 1.3B | 6.7B |
|---|---:|---:|
| **`no_half`** (gated-MLP split) | **4.4203** | **4.4628** |
| `no_ln` (RMSNorm → diagonal) | 0.9806 | 0.9885 |
| `no_identity` (SiLU → elementwise) | 0.2265 | 0.3941 |
| `no_attn` (attention, unmodified by design) | 0.5128 | 0.3044 |

The gated-MLP split carries the faithfulness gain by ~4.5×. Attention's cost,
the one path left alone by design, is 0.30–0.51 — bounded and measured rather
than unknown.

The two deepseek models reproduce E14's target essentially exactly. **StarCoder2-3b
does not** — and the reason is not that the rules conserve badly there, it is
that **they never install**. StarCoder2 uses LayerNorm (deliberately unmatched:
it subtracts the mean, so the rule's algebra differs) and a non-gated MLP, so
`norm_eps_attr` and `is_gated_mlp` both decline and the two homogenising rules
bind to nothing. Only the attention hooks register, which is enough to satisfy
`lrp_rules`' own `strict` check — so a lens labelled `rlens` gets built that is
arithmetically a J-lens, and 0.154 is just what raw autograd gives.

**J0 now refuses this** (`rlens_rules_bound`, added 2026-08-19): stage 125 records
how many modules each rule bound to and the gate fails an R-lens where neither
homogenising rule matched. The gate did not exist when the canonical runs were
made, which is why the artifact exists at all — the diagnostic surfaced it, and
the gate now prevents it.

**Diagnostics are not gates, deliberately.** Lens fidelity — next-token recovery,
agreement with the final-layer distribution, relevance conservation, and the
random / Gram-matched floors — is measured per (layer, lens), emits warnings, and
**never blocks execution**. A test asserts that no gate function reads any
fidelity variable. The reason is selection: refusing to run at low-fidelity
layers would silently restrict every lens experiment to the layers where the
instrument is comfortable, and early and middle layers are usually the target.
The report therefore distinguishes four outcomes — *mechanically invalid*,
*mechanically valid with weak lens fidelity*, *valid null*, and *positive above
controls* — rather than collapsing them into pass/fail.

## 10b. Using the lens as a *contrast* (E15-C), and what it cost to do honestly

E10/E11 score one state against a candidate vocabulary. E15-C scores a **matched
pair** and takes the difference, which introduces four problems the earlier
lens work never had to face.

**1. Orientation must be fixed once.** Every pair is oriented
`delta(pair, token) = score_unsafe(token) − score_safe(token)`, recorded on every
output row, and gate J1 refuses a run whose rows disagree. A per-cell orientation
choice would make every sign statistic meaningless.

**2. The scale caveat now bites.** `JLens.scores` drops a positive per-position
factor, and a paired contrast compares *two different positions*. So every
statistic is carried in three conventions:

| convention | meaning | exactness |
|---|---|---|
| `score` | raw lens score | exact for the logit lens, scale-carrying for J/R |
| `z` | z-scored across the candidate set at that position | **exactly** invariant to the dropped factor — the scale-safe way to compare positions |
| `prob` | softmax over the candidate set ("probability mass") | exact for the logit lens; inherits the factor for J/R |

Sign consistency is reported in both `z` and `prob`, because for the J/R lenses
they can disagree and a reader is entitled to see when they do.

**And the convention is checked, not assumed.** Z-scoring removes a *shared*
scale factor, but it does not remove a systematic difference in the distribution's
*shape* between the two members — and such a difference would move the contrast in
a fixed direction with no concept involved. Stage 126 therefore records each
member's candidate-distribution entropy and score-vector norm, and stage 127
correlates them with the contrast per pair. In the canonical E15-C runs the
correlations never exceed |r| = 0.39 and sit at ≤ 0.29 in the reported cells, so
the measured contrast is not a distribution artifact. This check exists because
the alternative — asserting that z-scoring handles it — is exactly the kind of
claim this repository does not make without a measurement.

**3. The candidate vocabulary cannot be the whole vocabulary.** A J/R lens vector
is one vector-Jacobian product *per candidate token*, so a 32k-row lens at every
layer is infeasible, not merely slow. Discovery is therefore two-phase:
a full-vocabulary **logit-lens** ranking on clean *training* pairs selects a
candidate pool (196 tokens in the canonical runs), then each lens ranks that pool
by its own training delta. The limitation this creates is recorded inside the
frozen artifact itself: *a direction only the J-lens or R-lens would surface, on a
token outside the pool, cannot be discovered here.*

**4. Discovery must not see the evaluation.** The frozen token set is written to
`vocab/vocab_discovery.json` by stage 125 and **read back from disk** by stage 126
— a filesystem boundary rather than a promise. J1 additionally checks that the
recorded discovery digest is the training split's and differs from the evaluated
split's.

**Concept tokens are validated per model, and nothing is substituted.** A lexicon
word is used only if it encodes to exactly one token in one of the prompt-space
variants (`" word"`, `"word"`, `" word\n"`) *and* that token decodes back to the
variant that produced it. Every omission is recorded with its reason. Coverage is
genuinely model-specific: `" vulnerable"` survives on both deepseek models but
`unsafe`, `untrusted` and `tainted` all split; on starcoder2-3b `" unsafe"`
survives and `vulnerable` does not. The first token of a split word is a *prefix*,
not the word, and using it would measure the prefix.

**What the contrast is controlled against.**

| control | what it rules out |
|---|---|
| **permutation** — re-orient each base at random | that anything other than the safe→unsafe *direction* carries the effect; keeps every pair and magnitude |
| **mismatched pairs** — unsafe and safe from different bases | that the effect is any difference between two programs of this kind, rather than the class difference |
| **embedding layer (−1)** | token identity: at `sink_arg` the state *is* the anchor token's embedding, so this is the token-identity contrast exactly |
| **`last_token` site** | ditto from the other side — both members carry the *same* token there |
| **identifier-role strata** | that the generator's tainted/trusted name assignment drives the sign |
| **random and Gram-matched lenses** | that any direction of that norm — or of that norm *and* those pairwise angles — would separate the pairs |

**What licenses a semantic reading.** All of: train-only discovery frozen before
scoring; held-out replication in the *hypothesised direction* (the test is
deliberately one-sided — a two-sided test would report a consistently reversed
contrast as a positive result); one recorded orientation; an effect above **both**
the permutation and mismatched-pair controls; stability across identifier roles;
and evidence not reducible to the differing sink-argument token. Stage 127 decides
this by explicit checklist and prints the checklist. **"The token `unsafe`
appeared in a top-k list" is not a result** and is not allowed to become one.

**The finding (three models, canonical scale).** A null. See
`docs/RESULTS.md` and `docs/design/E15_SINKFLOW_PLAN.md` §8.6: no
security-vocabulary concept in any model, the direction not even consistent
across models, 1.3B significantly *inverted*, and the three lenses in agreement —
so the null is a property of the models, not of the instrument. What the design
licenses from that is a real and reportable claim: **linear decodability and
expression in a model's own output vocabulary are different properties.**

**One correction to the controls above.** The mismatched-pair arm named in "what
licenses a semantic reading" is weaker than it was described as being, and §10c
replaces it. It redraws the *safe* partner from the same safe pool, so the label
difference survives it and its expected mean is the main arm's exactly — it
falsifies "specific to this pairing", not "about the label".

---

## 10c. Three ways the E15-C null could be wrong — and what testing them found (E15-D)

A null needs different evidence from a positive result. Every control in §10b is
*negative*: permutation, mismatched pairs, random and Gram-matched lenses. Those
establish that a positive result is not an artifact. **They are silent about a
null**, and nothing in §10b separates "the models do not verbalise this" from
"this readout could not detect verbalisation if it were there". §10c is the three
measurements that address that, built as stages 128–131 with gates J2/J3/J4. Full
design and pre-declared thresholds: `docs/design/E15D_LENS_FOLLOWUPS_PLAN.md`.

**Result: the first overturned the reading of the null, and the second removed
its ambiguity.** The positive control fired on all three models — the identical
readout detects a property the models express (sign consistency 0.85–0.94,
tracking the model's own forced-choice margin at 0.71–0.92), so §10b's null
cannot be blamed on the instrument. One qualification: on deepseek-coder-6.7b the
security lexicon *also* separates the pair at the answer position of a prompt
that asks the question, so §10b's null is a null about the **unprompted** state
at the sink argument.

**And the first of the three overturned the reading of the null.** Removing
the candidate pool found a label-defined direction that generalises to held-out
programs in 72/72 pairs on all three models. So the correct summary of the lens
track is not "the distinction is absent from output-aligned coordinates" but
**"it is present in output-aligned coordinates and is not carried by any word for
it"**. §10b's null stands as a null *about the security lexicon*; it was never a
null about the output basis, and the difference is now measured rather than
argued. Numbers in `docs/RESULTS.md`.

**The candidate pool could have missed it (stage 128).** §10b's readout can only
find a concept that some token in a 196-token, logit-lens-selected pool carries,
and that pool is ranked by the *mean* paired delta. A large mean is compatible
with every pair's difference pointing somewhere different. So: form each pair's
difference over the **whole vocabulary**, z-scored per member exactly as in §10b,
and measure **concentration** rather than the mean —

    sv1_share = lambda_max(U Uᵀ) / trace(U Uᵀ),   U the unit-normalised differences

which is 1/n for unrelated differences and 1 for identical ones, and is
*sign-invariant*, which is what lets it be compared against a null whose members
have no canonical orientation. The direction is estimated on the training split
and projected onto held-out pairs. A null here cannot be blamed on a pool,
because there is no pool. The primary site is `last_token`, not `sink_arg`,
because it is the only site where both members carry the same token id in 100% of
pairs; layer −1 is the explicit surface floor, and at `last_token` that floor is
exactly zero, since identical tokens give identical embeddings.

**The readout could be blind (stage 129) — the positive control.** Run the
identical measurement on a property the models demonstrably answer: the E6/E7
forced-choice taint question, whose answer is a single token. One candidate basis
carries both properties, and both contrasts go through the same
`pair_contrast` call in the same convention with the same orientation, so the two
readouts differ *only* in which token positions are named as poles — J3 refuses
the run otherwise. The model's own forced-choice margin is recorded per program,
so `lens_tracks_model` can separate "the lens sees what the model says" from "the
lens sees something". The behavioural statistic is `pair_separation` rather than
accuracy, because a model that always answers "no" scores 0.5 accuracy for free
while pair separation's chance level of 0.5 is immune to answer bias. Four
outcomes are declared in advance, including the one that would retire the track:
if the model answers and the lens does not see it, the null is about the method.

**The distinction might not be lexicalised at all (stage 130).** Both readouts
above require the concept to live in vocabulary space. Under the LRP rules of
§10a the tail is degree-1 homogeneous, so `Σ_t R_t = s` and `R_t/s` is a
*partition* of the model's own answer across input positions. Two programs give
different scores, so raw relevances are not comparable — but fractions are, and
because they sum to one in both members, a difference is a genuine
**redistribution** rather than a change of scale. That is a property §10b's
readout never had: its z-score convention exists because `JLens.scores` drops an
unknown positive factor, and here conservation fixes the total instead.
Relevance is aggregated by AST role recomputed from each variant's own source,
and the control is free: **only `sink_arg` differs in tokens between the two
members**, enforced at generation time in every condition and verified across all
1440 held-out programs to give identical per-role token counts within every pair.
A shift among the token-identical roles has no surface account. Conservation is
measured per (pair, layer) and the reading is refused where it does not hold —
including outright on architectures where the homogenising rules bind to nothing.

**Two statistics, and why they disagreed.** Both stages that fired reported a
pre-declared criterion that *failed* beside one that passed, and in both cases
the two answer different questions rather than contradicting each other. Stage
128: the projection onto a frozen direction asks whether a label-defined
direction **generalises**; the concentration statistic `sv1_share` asks whether
the label axis **dominates** the difference vectors. The first passed and the
second failed, because two programs of the same label already differ along a
shared axis of comparable size — established as a separate axis by the two
directions' top-100 loadings overlapping at a Jaccard index of ≈0. Stage 130:
`sign_consistency` and its exact binomial null fired at p ≤ 4e-11 while the
**mean's** permutation null did not, because relevance deltas are heavy-tailed
enough that a handful of outlier pairs flip the mean while the median holds. In
both cases both numbers are reported, the verdict label names what actually
happened, and the median — not the mean — is the summary to read for a
heavy-tailed quantity.

**A better negative control, retrofitted to §10b.** The mismatched-pair arm
cannot falsify a label claim: its partner is still a *safe* program, so it
averages over the very set the main arm averages over and its expected mean is
the main arm's exactly — measured, the two agree to four decimal places on all
three models, and on deepseek-coder-6.7b the control is *more* sign-consistent
than the main arm. The replacement takes **both** members from one pole:
everything a matched pair differs in is still present, the label difference is
gone, so the expected contrast is zero. Stage 126 now runs both, and stage 127
reports `pairing_gain` — what base matching actually buys — as a number.

---

## 11. Reproducibility

- **Seed 42 everywhere** by default (generator, CV splits, subsampling,
  bootstrap).
- **Every stage writes a manifest** (`results/manifests/`) recording the git
  SHA, the arguments, and wall-clock time.
- **All figures and tables regenerate from the tidy CSVs alone** (stage 90), so
  the entire chain from raw data to published figure is auditable end to end.
