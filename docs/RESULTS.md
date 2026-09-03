# Results

## How each result is explained

Every result below follows the same reading order: the question being tested,
why it matters, the experimental comparison, the controls, the exact result,
and the conclusion that result supports.

The active result is one cumulative binding story. First, controlled probes
show that variable binding and definition-to-use relations become linearly
recoverable in middle layers. Second, frozen probes show that this representation
survives many surface changes but weakens under scope interference and
control-flow flattening. Third, a DAS intervention shows that a rank-1 binding
component is causally used at the variable-use site. Now that presence and
causal use are established, the published J-lens asks what that representation
is like and whether it is verbalized in the language of binding; the R-lens is
retained as a supporting replication. Concrete-value recovery is secondary.

The earlier cotangent-lens and conserving-cotangent-lens studies, including E16
and E18, are preserved in [ARCHIVE.md](ARCHIVE.md). They are method development,
not results from the published J-lens or R-lens. The lens portion of Part III
reports only E19 using the actual J-lens and R-lens.

Every active result below completed at canonical scale and is paired with the
control that could have falsified it. Read “represented” as *recoverable from a
hidden state*, “robust” as *the same frozen readout still transfers*, “used” only
where an intervention changes the downstream answer.

### Contents

- [The argument in one page](#the-argument-in-one-page)
- [Status at a glance](#status-at-a-glance)
- [Part I — The relation is represented](#part-i--the-relation-is-represented)
- [Part II — Robustness and failure boundaries](#part-ii--robustness-and-failure-boundaries)
- [Part III — Causal use, then binding-language verbalization](#part-iii--causal-use-then-binding-language-verbalization)
  - [After DAS: is the binding representation verbalized?](#e19--after-das-is-the-binding-representation-verbalized)
- [Synthesis](#synthesis-the-main-finding)
- [Boundaries](#boundaries-what-this-project-does-not-claim)
- [Open items](#open-items)

---

## The argument in one page

**1. Representation.** In controlled program pairs, the queried identifier is
the same token at the same position while its binding changes. The embedding
layer and a bounded model-free reader are at exactly 0.500, but a linear probe
reaches approximately 0.984 in middle layers. The binding relation is therefore
made linearly available by contextual processing rather than read directly from
the local token.

**2. Robustness.** The clean probe is frozen and evaluated after
meaning-preserving transformations. Long irrelevant context and consistent
renaming cause limited middle-layer damage. Reusing the tracked names in
competing scopes drives binding toward chance, and control-flow flattening causes
the largest reproducible collapse. This identifies a structural failure boundary
for the original readout; it does not prove that every possible representation
has vanished.

**3. Causal use.** DAS learns a rank-1 subspace at the unchanged variable-use
token. It is fitted where installing the other binding requires answer **a → b**
and tested where the same binding change requires **b → a**. The model emits the
value selected by the installed binding on 100% of held-out cases in both arms,
in DeepSeek-Coder 6.7B and StarCoder2 3B. A fixed answer direction attenuates or
reverses, and dose-matched random edits are much weaker.

**4. Binding-language verbalization.** With causal use already established, the
J-lens asks whether the use-site state is expressed as words for binding. It
surfaces a controlled binding-vocabulary-family signal in both completed
DeepSeek panels (`scope` on 1.3B and `global` on 6.7B). It does not expose the
concrete value before emission. R-lens closely reproduces this pattern.

---

## Status at a glance

| Stage | Instrument | Main evidence | Scope |
|---|---|---|---|
| **representation** | controlled linear probe | binding reaches ~0.984 over a 0.500 floor | DeepSeek 1.3B/6.7B; partial StarCoder2 replication |
| **robustness** | frozen-probe transfer | resilient to distance and renaming; fragile to scope interference and flattening | three tested models where reported |
| **causal use (R10/E13)** | rank-1 DAS interchange | 100% installed answer in both crossed arms | DeepSeek 6.7B and StarCoder2 3B |
| **binding-language verbalization (E19)** | published full-Jacobian J-lens + RelP R-lens | controlled binding-vocabulary signal on two DeepSeek models; concrete value absent before emission | concept result on two models; value null on three |

Missing result numbers refer to studies now documented in
[ARCHIVE.md](ARCHIVE.md), not to missing active experiments.

---

# Part I — The relation is represented

*Instrument 1: linear probes against a construction-pinned floor
([METHODS §3](METHODS.md#3-instrument-1--linear-probes-and-their-floors)).*

## R1 — Variable binding

### Research question

Does the model represent **which definition** an identifier occurrence refers
to — as opposed to merely which characters it is spelled with?

### Why this relation

This is the cleanest case in all of code where surface form and meaning come
apart. Two occurrences of `data` may be the same variable or two different ones
depending on scope, and nothing local tells you which. It is also a **DFG
relation** with an exact answer (METHODS §1.1), so the label is not a judgement
call.

### Hypothesis

If the model resolves scope, binding will be decodable from hidden states
*above* the level any surface reader can reach, and it will be **absent at the
embedding layer** — because scope resolution is a computation, not a lookup.
If instead the model pattern-matches on identifier strings, decodability will
appear at the embedding layer and will not improve with depth.

### Method

A pairwise linear probe on `[h_i ; h_j ; h_i−h_j ; |h_i−h_j|]` asking *"do these
two occurrences bind to the same definition?"*. Negatives are split into strata
and reported separately, from trivial to decisive:

| stratum | what it removes |
|---|---|
| `diff_name` | nothing — the trivial baseline (capped at 3× positives) |
| `distance_matched` | the "nearby ⇒ related" shortcut |
| `same_name_diff_binding` | the "same string ⇒ same variable" shortcut |
| **`context_matched`** | **every fixed-offset surface cue at once** |

**The control is the experiment.** On `context_matched` the two programs are
token-identical except the single binding-flipping character; the anchor windows
and the token distance are identical while the label flips; and both programs
share one cross-validation group, so neither can be memorised through the other.
Grouped CV, shuffled-label selectivity control, and a model-free surface baseline
run on the same rows (METHODS §3.2–§3.4).

### Result

`context_matched`, the only clean headline:

| | 1.3B | 6.7B |
|---|---:|---:|
| surface baseline (±3 token ids + distance, no model) | **0.500** | **0.500** |
| embedding layer (−1, token identity only) | **0.500** | **0.500** |
| block 0 (first transformer layer) | 0.570 | 0.531 |
| layer 3 | 0.961 | 0.914 |
| **peak (mid layers)** | **0.984** (L7) | **0.984** (L11–15) |
| last layer | 0.930 (L23) | 0.914 (L31) |

Three phases, each saying something different:

1. **Nothing at the input.** Both floors are *exactly* 0.500 — by construction,
   and confirmed in the data. The binding information is not in the tokens; it
   has to be built.
2. **Built in the first few blocks**, reaching 0.91–0.96 by layer 3 and
   plateauing near 0.98 through the middle. That is early for a relation
   requiring scope resolution.
3. **Partly shed near the output** (0.91–0.93), consistent with the final layers
   reorganising toward next-token prediction.

**Cross-scale.** The two models agree on shape and differ only where a scaling
account predicts: 6.7B does slightly less work in block 0 and holds its peak
longer — the same relative depth, stretched. The surface-baseline and
embedding rows are numerically *identical* across models, which they must be
since neither involves the model; that identity doubles as a corpus-integrity
check.

**Only `context_matched` is a clean headline.** The other strata sit at ~0.99
from block 0 because the token strings already separate them — the surface
baseline scores 0.78–0.94 on them too, and reporting a pooled number across
strata would hide exactly that.

### What it means

The model constructs a scope-resolution fact that is not present in its input
representation and cannot be recovered by any bounded surface reader. This is
the project's foundation, and the reason it can use the word *semantic* at all
(METHODS §0.2).

---

## R2 — Def–use edges

### Research question

Is a directed **definition→use** edge decodable from hidden states, and how far
does it reach?

### Hypothesis

If the model tracks data flow rather than adjacency, decodability will survive
long distances between endpoints, with at most graceful decay. If it tracks
adjacency, accuracy will fall sharply with token distance.

### Method

The same pairwise setup and the same strata as R1, over reaching-definition edges
from the DFG. Negatives are **distance-matched**, so "nearby ⇒ related" cannot
win. Accuracy is bucketed by token distance (0–10, 10–50, 50–200, 200+).

### Result

Same profile as R1 — peak ~0.99 at layers 7–11 over the same 0.500 floor — with
honest decay by distance. The hardest bucket (50–200 tokens apart) holds at
**0.96–0.99** against ~0.99 for nearby pairs.

### What it means

The model tracks def–use links across real distance rather than adjacency. Taken
with R1, the two foundational CPG relations are decodable over floors that no
surface feature can exceed.

> **Why control dependence is not R3.** The natural third CPG relation — does
> this statement execute under that guard? — was measured and is **not reported
> as a result**. Its model-free surface floor is **0.927**, because a statement's
> guard is usually its nearest enclosing `if`, so token windows plus indentation
> recover most of it with no model at all. By this project's criterion
> (METHODS §0.2) that relation is *mostly syntactic*, and no representational
> conclusion follows from decoding it. The measurement, its numbers and the
> reason it was demoted are in [ARCHIVE.md §4.3](ARCHIVE.md#43-control-dependence).
> That the criterion **excludes** something is what makes it a criterion rather
> than a slogan.

---

# Part II — Robustness and failure boundaries

*Instrument 2: frozen probes transferred across meaning-preserving rewrites
([METHODS §4](METHODS.md#4-instrument-2--frozen-transfer-and-the-obfuscation-ladder)).
A change in accuracy here is a change in the model's state, not in the probe.*

## Part II in plain language

Part I showed that hidden states contain information about binding and data
flow. Part II asks what that information depends on. The experiments keep the
program's behavior fixed while changing its presentation. If the same frozen
probe still works after a rewrite, the original representation survived in a
form that the probe can still recognize. If its accuracy falls, the rewrite
changed that representation or changed how it is encoded.

Three distinctions are important:

- **Distance versus interference.** Adding harmless text makes two related
  tokens farther apart. Reusing the same variable names creates genuinely
  competing references. R3 tests these separately.
- **Atomic versus cumulative transformations.** An atomic condition applies one
  rewrite by itself. A cumulative condition adds several rewrites in sequence.
  Atomic conditions identify which rewrite causes a failure; cumulative
  conditions show what happens when the rewrites are combined.
- **Representation loss versus probe-transfer failure.** These experiments use
  a probe trained only on clean programs. If that frozen probe fails on a
  rewritten program, we know its original readout no longer transfers. We do
  **not** automatically know that the model has lost every possible encoding of
  the information.

The result of Part II is:

- Long context by itself causes little damage.
- Context containing confusing uses of the tracked names causes much more
  damage.
- Consistent renaming changes early lexical representations, but much of the
  mid-layer semantic representation survives.
- Control-flow flattening is the tested rewrite that causes the largest,
  reproducible collapse in binding and def–use readout transfer.

## R3 — Distance is cheap; interference is not

### Research question

Does the representation degrade because things get *far apart*, or because the
*problem gets harder*?

### Hypothesis

A positional heuristic degrades with distance. A computed relation degrades with
task difficulty. Inserting filler of **fixed length** but varying *semantic
interference* separates the two.

### Method

Insert filler between the tracked definition and its use, sized by real tokenizer
counts (0 → 1000 tokens), varying only what the filler *does*. Binding and
def–use probes are **frozen** from the clean corpus and never refitted; ground
truth is recomputed for every variant.

The comparison is controlled: two conditions can add the same number of tokens
while differing in whether those tokens create a harder reference-resolution
problem. This lets the experiment attribute a loss to semantic interference
rather than merely to sequence length.

| filler | what it adds |
|---|---|
| `comment_prose` | inert English |
| `dead_code` | unreachable statements |
| `lexical_decoy` | similar-looking but irrelevant fresh names |
| `competing_update` | genuinely rebinds *other* variables |
| `scope_shadow` | reuses the *tracked* names in a nested scope |

### Result

6.7B binding accuracy at 500 inserted filler tokens:

| filler | acc | reading |
|---|---:|---|
| `comment_prose` | **0.921** | length is almost free |
| `dead_code` | 0.794 | mild |
| `lexical_decoy` | 0.795 | mild |
| `competing_update` | 0.859 | moderate |
| `scope_shadow` | **0.570** | **severe** |

At 1000 tokens `scope_shadow` reaches chance (0.498) while every other filler
stays above 0.70.

**The per-layer detail is the sharper finding.** Under `scope_shadow`, block 0 is
the *most stable* part of the network (flat ~0.75) while the middle layers — the
ones doing the binding work — collapse.

### What it means

The interference lands on **the computation, not on a lookup**. The
representation degrades when the semantic task gets harder, not when the context
gets longer — which is what one wants from a computed relation rather than a
positional heuristic.

---

## R4 — Atomic obfuscation: renaming is survivable mid-layer, flattening is not

### Research question

Is the relation carried by the identifiers, or by something that survives
rewriting them? And **which** rewrite is responsible for whatever damage occurs?

### Hypothesis

If mid-layer states encode binding relative to the *control structure* rather
than to identifier strings, then consistent renaming should be survivable in the
middle of the network while dissolving the control skeleton should not.

### Method

The cumulative, execution-verified obfuscation ladder (METHODS §4.2): normalize →
rename → opaque predicates → MBA arithmetic → control-flow flattening. Every
variant is executed and checked observationally equivalent to its base; all
levels of a base are kept or dropped together so level curves compare identical
program sets. Frozen probes, ground truth recomputed per variant.

Here, **normalize** means parsing and printing the program without intending to
change its structure. **Rename** consistently replaces local identifier names.
An **opaque predicate** adds a branch whose outcome is known by construction but
is less obvious from its syntax. **MBA arithmetic** rewrites a simple expression
as an equivalent mixed Boolean-arithmetic expression. **Control-flow
flattening** replaces the original structured flow with a dispatcher-like form.
All of these preserve the observed program result, but flattening most directly
changes the visible control structure used to determine which definitions reach
which uses.

### Result

6.7B binding, best-layer accuracy per cumulative level:

| level | transform | best layer |
|---:|---|---:|
| 0 | normalize | ~1.000 |
| 1 | + rename every local | **0.897** (L11) |
| 2 | + opaque predicates | 0.857 |
| 3 | + MBA arithmetic | 0.846 |
| 4 | + control-flow flatten | **0.750** |

Across all three models, at the best layer per task:

| | rename | opaque | encode | flatten |
|---|---:|---:|---:|---:|
| binding — 1.3B / 6.7B / SC2-3B | 0.783 / 0.883 / 0.708 | 0.801 / 0.862 / 0.743 | 0.834 / 0.857 / 0.790 | **0.555 / 0.615 / 0.527** |
| def–use — 1.3B / 6.7B / SC2-3B | 0.819 / 0.864 / 0.689 | 0.799 / 0.846 / 0.731 | 0.800 / 0.833 / 0.747 | **0.461 / 0.545 / 0.402** |

**The layer breakdown is the finding.** Renaming pushes the *embedding and
block-0* probes **below chance** (0.29–0.33) — those layers keyed on identifier
strings, and renaming actively misleads them — while mid layers 7–15 hold at
0.85–0.90. Opaque predicates and rewritten arithmetic barely register, because
they do not change which definition reaches which use.

### What it means

Together, R3 and R4 describe **one failure surface**: the representation is
robust to *how far apart* things are and to *what things are called*, and it
fails when the scope or control structure it is a representation *of* becomes
harder. That is the signature of a computed relation, and it is a first, coarse
map of when a tool built on these representations should not be trusted.

---

# Part III — Causal use, then binding-language verbalization

Part I established that binding and def–use relations are recoverable from hidden
states. Part II showed where those readouts remain stable and where they fail.
Part III asks what happens after a binding representation has formed: does the
model use it to choose an answer? DAS changes one learned component at the
unchanged variable-use token and tests whether the emitted value follows the
donor program's binding. Once that causal-use question is answered, E19 asks
what the used representation is like and whether J-lens surfaces it as binding
language.

Earlier cotangent-lens attribution and lexical experiments are reproducible but
are not published J-lens/R-lens results. They are preserved in
[ARCHIVE.md](ARCHIVE.md).

## Where the evidence lives

| Claim | Active method | Detailed generated reports |
|---|---|---|
| causal binding transport | [METHODS §5](METHODS.md#5-das--causal-interchange-of-a-binding-component) | [DeepSeek-Coder 6.7B](../results/binding/deepseek-coder-6.7b/e13_report.md), [StarCoder2 3B](../results/binding/starcoder2-3b/e13_report.md) |

## R10 — DAS: the binding representation is causally used

For a slower explanation of every step, term, control, gate, and metric, start
with [METHODS §5.0](METHODS.md#50-plain-language-map-of-one-das-run). This
section focuses on the result.

### Question

A linear probe can recover which definition a variable refers to, but that does
not show that the model uses this information. R10 asks whether changing only a
small binding-related component of the hidden state changes the model's answer
as the binding predicts.

### Controlled programs

Each base produces four programs. In every program the model simply returns a
variable; there is no arithmetic capability mixed into the task. One member has
an outer binding and the other has a shadowing inner binding:

```python
x = a                      x = a
def f():                   def f():
    y = b                      x = b
    return x                  return x
# answer a                 # answer b
```

The intervention is made at the `x` in `return x`. This use token occupies the
same position and has the same identity in both programs. The edit therefore
cannot transport a different input token at the intervention site.

The value assignment is then crossed in a second arm:

| arm | outer value | inner value | installing the inner binding must cause |
|---|---|---|---|
| `ab` — fitted arm | `a` | `b` | answer `a → b` |
| `ba` — held-out arm | `b` | `a` | answer `b → a` |

DAS is fitted only on `ab`. The claim is evaluated on `ba`, which it never sees
during fitting. This crossing is the central isolation mechanism. A subspace
that represents “use the inner definition” should work in both arms. A direction
that merely means “increase answer token `b`” should help in `ab` and fail or
reverse in `ba`.

### What DAS changes

At the use position, DAS learns a rank-1 basis `R`. Given a host hidden state and
a donor hidden state with the other binding, the intervention replaces only the
host's component along `R`:

> `h' = h_host + R Rᵀ (h_donor − h_host)`

Everything orthogonal to `R` remains the host's state. The language model is
frozen; only `R` is optimized on the calibration split. There is no manually
chosen edit multiplier. The donor's actual component is installed, and the
resulting edit norm is measured afterward.

### Controls and what each rules out

| Control | Why it is needed | Expected result if DAS carries binding |
|---|---|---|
| **crossed `ba` arm** | separates binding from a fixed answer token | DAS follows the reversed answer requirement |
| **`das_answer_control`** | learns answer-token actuator vectors at the same site, with the same optimiser, steps, split and per-row edit norm as binding DAS, but never receives the donor binding state | must work on fitted `ab`, then attenuate on crossed `ba`. **This is H5's discriminator** |
| **J-lens and R-lens answer directions** | apply the published lens read directions as interventions | supporting diagnostics only: a direction can be a valid readout yet a poor actuator |
| **dose-matched random subspace** | tests generic disruption at a comparable edit size | much weaker than DAS |
| **rank-matched random subspace** | provides a simple random rank-1 floor | near zero |
| **no-op** | detects hook or measurement artifacts | exactly zero |
| **whole-state donor patch** | verifies that this site can change the answer in each arm | moves the answer in both arms |
| **mean donor−host direction** | tests the cheapest non-learned rank-1 alternative | may transport, but should be less effective or require a larger edit |

The answer-only control is deliberately simple and directly matched. It learns
a vector `u_w` for each answer token on calibration data. To push from answer
`a` toward answer `b`, it adds the normalized direction `u_b − u_a`. The edit
length is not tuned: on every example it is set to the edit length produced by
binding DAS. Thus, if the control fails, it cannot be dismissed as receiving a
smaller intervention. Its direction is defined from the fitted `ab` arm and
held fixed on `ba`, where the correct answer movement reverses.

This control differs from binding DAS in exactly the way the causal question
requires. Binding DAS receives the donor program's hidden state and can copy a
component that says which definition is active. `das_answer_control` receives
no donor binding state; it can only learn how to push toward answer tokens.
Both use the same model, layer, position, Adam optimiser, 200 steps,
calibration/test division and row-wise dose.

The published J- and R-lens answer directions were also tested during
development. They were valid lens read directions but causally dead at the DAS
layers: on the training arm they installed fewer answers than a norm-matched
random edit. They therefore could not serve as H5's positive control. This is
not a failure of J-lens as a readout. Reading information and exerting causal
control are different jobs. Those runs and the reason they were superseded are
recorded in [ARCHIVE §4d](ARCHIVE.md).

> **Changed 2026-09-01.** This control previously used a *corpus-averaged
> cotangent readout over the two answer tokens*, fitted inside stage 106 from
> the DAS calibration programs. That is a different estimator from the published
> J-lens ([WORKSPACE_LENS §1](WORKSPACE_LENS.md)), and the numbers it produced
> are archived in [ARCHIVE §4b](ARCHIVE.md), not carried forward.

### Result

The completed experiment passes H0–H5 on DeepSeek-Coder 6.7B and
StarCoder2-3B, two different architecture families. All provable structural
zeros are exactly zero, all five fits per model converge, and the selected
subspace is rank 1. The headline outcome below is the fraction of held-out
test examples on which the model's full-vocabulary argmax is the value selected
by the installed intervention.

| intervention | DeepSeek `ab` | DeepSeek `ba` | StarCoder2 `ab` | StarCoder2 `ba` |
|---|---:|---:|---:|---:|
| **DAS rank 1** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **answer-only DAS control** | **76.8%** | **21.1%** | **96.6%** | **45.2%** |
| whole state | 73.8% | 73.4% | 62.7% | 64.6% |
| mean difference | 68.2% | 67.5% | 47.1% | 45.9% |
| dose-matched random | 1.6% | 1.8% | 21.2% | 22.3% |
| rank-matched random | 0.0% | 0.0% | 2.3% | 1.4% |

The first important comparison is within binding DAS itself. It installs the
correct donor-selected value on 100% of `ab` examples and 100% of `ba` examples.
The value assignment reversal therefore produces no attenuation at all. The
learned component follows the relation “which definition is active,” not the
literal that happened to be correct during training.

The second comparison establishes that the answer-only control is genuinely
alive. On its fitted `ab` arm it installs 76.8% of answers on DeepSeek and 96.6%
on StarCoder2, far above the corresponding random floors of 1.6% and 21.2%.
This repairs the dead-control problem of the earlier J-lens intervention.

The third comparison is the crossed test. When `a` and `b` exchange semantic
roles, the answer-only control falls from 76.8% to 21.1% on DeepSeek and from
96.6% to 45.2% on StarCoder2. Its crossed/training installed-answer ratio is
0.274 on DeepSeek and 0.468 on StarCoder2. H5 compares this with half the
whole-state transport ratio: the respective cutoffs are 0.498 and 0.516. Both
models pass. The StarCoder2 separation is narrower and should be described as
attenuation, not as complete failure of the answer control.

The closed-form mean direction is a real positive baseline, not a dead control:
it transports 68% of DeepSeek examples and roughly 46% of StarCoder2 examples.
This tells us that a broad average difference between the two program states
already contains some binding signal. DAS nevertheless reaches 100% with a
smaller edit: about 0.416 of the hidden-state norm on DeepSeek and 0.466 on
StarCoder2, compared with roughly 0.58 and 0.70 for the mean direction.

The whole-state patch reaching only 73–74% and 63–65% is not a numerical
contradiction. A whole-state patch copies every difference between donor and
host, including irrelevant or competing components. Rank-1 DAS can preserve the
host computation while changing the one component most useful for the answer.

### Conclusion and limits

At the tested use position and layer, downstream computation reads a compact
component whose effect follows **which definition is in scope** rather than the
answer token associated with the fitted value assignment. This is the project's
causal finding, supported by a live, matched answer-only control.

It remains local to one synthetic binding construction, one site and one layer
per model. The rank-1 edit also outperforms the whole-state patch, so the latter
is not a true numerical ceiling; the plausible explanation that a full patch
adds both helpful and opposing components remains to be tested.

## E19 — After DAS: is the binding representation verbalized?

The probes show that the representation is present, and DAS shows that it is
causally used. J-lens now asks what that representation is like in vocabulary
coordinates—specifically, whether it is verbalized as the language of binding.
Concretely, it estimates the average full
Jacobian from an intermediate layer to the late residual stream, then transports
each vocabulary direction back through that Jacobian. The result is a score and
rank for every token in the model's vocabulary—not merely a hand-chosen list.

The R-lens repeats the same measurement while applying the published RelP
backward rules. Because its findings track J-lens closely throughout this study,
it is treated as a supporting replication rather than a second main instrument.
The ordinary logit lens is the simpler comparison: it reads the vocabulary
directly without Jacobian transport.

All active results use the released 2026 implementation, an independent
100-prompt fitting corpus and the complete model vocabulary. The repository's
older cotangent and conserving-cotangent constructions are different methods
and remain archived.

### Operation audit: what was and was not run

| J-space operation | status in E19 |
|---|---|
| **READ** | **Run.** This produces every vocabulary ranking discussed below. |
| **WRITE** | **Not run.** No concept vector was added to steer the model. |
| **PATCH** | **Not run with J-lens coordinates.** The earlier DAS interchange is separate and lens-independent. |
| **ABLATE** | **Modified variant only.** E19 erases one target token's J-lens read direction at one layer, with controls. It does not remove the top 10 gradient-pursuit subframes across a band as in the published ABLATE operation. |

The main result below is therefore a **READ result**. The erasure section is a
separate causal check on one read direction, not evidence that WRITE or PATCH
was performed.

### Keep the two READ questions separate

There are two targets, and “the lens found the target” means something different
in each panel:

| panel | target being searched for | example | scientific question |
|---|---|---|---|
| **binding language** | an English/code word describing the relation | `scope`, `local`, `variable` | does the internal state verbalize what kind of relation is active? |
| **runtime result** | the concrete value this particular program will emit | a single-digit token such as `7` | is the answer token already vocabulary-readable while the variable is being used? |

A binding-language hit is not recovery of the answer, and an answer hit does not
show that the model names the relation as “scope.”

### Question 1: does J-lens verbalize the binding representation?

The complete predeclared binding lexicon is `local`, `global`, `inner`, `outer`,
`scope`, `scoped`, `shadow`, `shadowed`, `binding`, `bound`, `active`,
`inactive`, `definition`, `variable`, and `value`. Each concept consists of all
declared capitalization variants in both bare and space-prefixed form that the
tokenizer represents as exactly one token. Split spellings are logged as
unavailable and never truncated. The lens reads each concept over the full
vocabulary at the unchanged use token, post-use token, call site, and answer
position, for every selected layer.

The programs cross binding with value assignment: the same binding flip occurs
once with `(outer, inner)=(a,b)` and once with `(b,a)`. A binding-language signal
must move consistently when the answer literal reverses and remain stable as
literals change across bases. It must beat matched generic-code words,
deterministic size/frequency-band-matched random concepts, and the explicit
recency/survival confounds `earlier`/`later` and `kept`/`replaced`. For every
`(lens, layer, read, concept)`, the report gives full-vocabulary rank,
pass@1/5/10/50/100, threshold-entry layer, paired inner-minus-outer score
difference, crossed-arm agreement, literal invariance, and a cluster bootstrap
over base programs.

The two completed DeepSeek panels support a binding-vocabulary-family signal:
`scope` is clearest on DeepSeek-Coder 1.3B at L9 (+7.645/+7.637 across value
arms), and `global` on 6.7B at L20 (-9.199/-9.040). R-lens closely reproduces
the candidates; the logit lens also carries related effects. The result is not
a unique J-lens code or one universal internal word, and StarCoder2's concept
panel remains incomplete.

#### Exactly which words were tested

The binding family contained these 15 predeclared concepts:

`local`, `global`, `inner`, `outer`, `scope`, `scoped`, `shadow`, `shadowed`,
`binding`, `bound`, `active`, `inactive`, `definition`, `variable`, `value`.

For each concept, the test tried its declared lowercase and capitalization
variants, both bare and space-prefixed, but scored only spellings that were one
token for the model. `definition` also included `def`; `variable` included
`var`; and `value` included `val`. This is one semantic family, not 15 separate
post-hoc hypotheses.

Every available spelling was read at four exact places:

| read position | where it is | why it is tested |
|---|---|---|
| `use` | the unchanged variable token in `return x` | primary point: the binding is being resolved here |
| `post_use` | the token immediately after that variable | checks whether verbalization appears one token later |
| `call` | the later function-call position | checks whether the relation survives until the call/result is requested |
| `answer` | the position immediately before the output token | positive-control region near verbal emission |

#### Exactly which words were found, and where

“Found” needs two definitions because the analysis records both absolute rank
and controlled movement:

1. **Top-10 appearance:** did a word enter the ten highest-ranked tokens in the
   full vocabulary?
2. **Binding-tracking contrast:** did its lens score move reliably between inner
   and outer binding while agreeing across reversed value assignments?

Those criteria should not be silently substituted for each other. The clearest
results are:

| model | top-10 J-lens binding words | strongest controlled J-lens word | location |
|---|---|---|---|
| DeepSeek-Coder 1.3B | no concept has a concept-level earliest-top-10 entry; some individual rows nevertheless contribute to use-position pass@10 = 0.415 | `scope` | layer 9, chiefly the `use` read; binding deltas +7.645 and +7.637 in the two value arms |
| DeepSeek-Coder 6.7B | `value` and `variable` first enter at layer 11; `local` at layer 13; `global` at layer 14 — all at `use` | `global` | layer 20 at `use`; binding deltas −9.199 and −9.040 |
| StarCoder2-3B | not determined | concept panel not run | — |

Thus, the strongest 1.3B claim is **not** “`scope` was a top-10 prediction.” It
is that the predeclared `scope` score changed with the binding in both crossed
arms. For 6.7B there is also direct top-10 vocabulary evidence at the use token,
but the strongest controlled contrast occurs later and uses `global`. No J-lens
binding-concept top-10 entries were recorded at `post_use`, `call`, or `answer`
for these two completed panels under the concept-level threshold-entry summary.

The sign of a binding delta is arbitrary with respect to the names “inner” and
“outer”; consistency across the two value arms matters, not whether the number
is positive. This is why 6.7B's two large negative `global` deltas support the
same kind of result as 1.3B's positive `scope` deltas.

### Question 2: is the concrete runtime value verbalized early?

Here the tested “words” are not the 15 binding terms above. They are the
program-specific answer tokens. Because these tokenizers split multi-digit
numbers, the usable numeric vocabulary is the single-token digits `2` through
`9`; each item's execution determines which one is the target and which matched
value is the distractor. The `binding`, `alias`, `call`, and `defuse` families
test values already present in the prompt. The `arith`, `typeof`, and `loopvar`
families include targets absent from the prompt, which checks computation rather
than simple copying.

All required gates pass on DeepSeek-Coder 1.3B/6.7B and StarCoder2-3B. At the
answer position, J-lens reaches pass@10 = 1.000 for every value family. This is
the essential positive control: when the model has prepared the answer for
emission, the lens reads it perfectly.

At the variable-use token, the following token, and the later call site, the
needed values are essentially absent from J-lens's top vocabulary predictions.
This includes computed arithmetic targets that never occur in the prompt, so
the result cannot be dismissed as merely confusing copied and computed values.

The simple interpretation is not that the model lacks the information. The
probe and DAS experiments show that binding information is present and used.
Rather, the operative mid-network state is not yet organized as the vocabulary
token naming the final value. **Representation** and **verbalizability** are
therefore different properties.

### Question 3: is the J-lens value direction causally used mid-network?

The causal test erases the J-lens direction for the correct value and measures
the change in the model's own target-versus-distractor margin. Erasing a matched
distractor direction, a random direction and an exactly magnitude-matched
random edit prevents generic damage from being mistaken for semantic use.

- On StarCoder2, the mid-network J-lens effect is indistinguishable from the
  matched-random floor.
- On DeepSeek 6.7B, layer 20 shows a small target-specific effect: J minus the
  magnitude-matched random control is −0.018, 95% CI [−0.033, −0.002]. J-lens
  nevertheless does not beat the ordinary logit direction.
- On DeepSeek 1.3B, layer 20 also shows an effect, but the logit direction is
  substantially stronger (`J − logit = +0.204`, [0.149, 0.270]).

Near the final layers, erasures become much larger and target/distractor effects
separate cleanly. That result mainly describes the neighbourhood of the output
head, where vocabulary alignment is expected. It does not establish an early
global workspace.

The controls that were missing from the preliminary analysis are present:
stable seeds, separate J/R distractor directions, an exact edit-magnitude random
arm, four read positions, and paired cluster-bootstrap intervals. StarCoder2 is
also replicated with a paper-minimal R-lens that disables the unpublished
LayerNorm analogue; the conclusion is unchanged.

R-lens closely replicates this pattern. It sometimes produces a slightly larger
effect or an earlier rank than J-lens, especially on DeepSeek, but it does not
change any model-level conclusion and does not consistently outperform the
logit lens. It is therefore supporting evidence for the robustness of the
J-lens finding, not the headline result.

### Supporting interpretation of the binding-language result

Concrete values and semantic concepts are different targets. A model might not
encode the token `7` at `return x`, yet its state could still align with words
such as `scope`, `global`, `local`, `binding` or `variable`. Stage 206 tests this
using predeclared concept sets, both crossed value assignments, matched generic
code words, positional/action confounds, and frequency/size-matched random word
sets.

The two completed DeepSeek panels support such an abstract vocabulary signal:

| model | J-lens use-position binding pass@10 | clearest crossed J-lens concept |
|---|---:|---|
| DeepSeek-Coder 1.3B | 0.415 | `scope`, L9: +7.645 / +7.637 across the two value arms |
| DeepSeek-Coder 6.7B | 0.938 | `global`, L20: −9.199 / −9.040 across the two value arms |

In each case the concept changes with the binding in both value assignments,
while the difference between value assignments contains zero. In plain terms:
the word-level signal follows which definition is active, not whether the
literal happens to be `a` or `b`. R-lens reproduces the same candidates closely
and is sometimes numerically stronger. The logit lens also shows related
effects, so this is evidence for vocabulary alignment, not evidence that
Jacobian transport uniquely discovers it.

This positive semantic-concept result does not contradict the negative value
result. Together they say that the model can expose an abstract signal related
to binding before it exposes the concrete answer token. StarCoder2 does not yet
have this semantic-concept panel, so the concept result is scoped to the two
DeepSeek models. Because the strongest named word differs across model sizes,
the replicated claim is at the **binding-concept family** level, not that every
model internally uses one universal word such as `scope`.

### Lens conclusion

The complete lens result is mixed and informative, in the intended inferential
order:

1. The earlier DAS result independently establishes causal use.
2. J-lens reveals binding-related vocabulary signals in the two completed
   DeepSeek panels, addressing what the used representation may verbalize.
3. As a secondary contrast, J-lens works technically and reads the concrete
   value perfectly at emission but not generally mid-network.
4. Its value directions have little or no special causal purchase mid-network
   beyond simpler directions.
5. R-lens closely supports these conclusions without supplying a distinct main
   finding.

# Synthesis: the main finding

The active evidence now forms one cumulative argument.

**First, binding is represented.** At the input layer, the controlled binding
pairs are indistinguishable to the measured local surface reader and binding is
at chance. Through the transformer, a linear readout rises to approximately
0.98. This establishes that a binding distinction becomes recoverable from the
model's contextual state. It does not establish that the model uses it.

**Second, the representation has a specific robustness profile.** It survives
identifier renaming, long irrelevant context, and several equivalent source
rewrites much better than it survives competing scopes or control-flow
flattening. These are frozen-readout results: they show when the original linear
representation transfers, not that every possible encoding has disappeared when
it fails.

**Third, the model causally uses a compact binding component.** DAS learns a
rank-1 subspace at the unchanged variable-use token. Installing the donor's
component makes the recipient emit the value selected by the donor binding on
100% of held-out cases in both crossed value-assignment arms, in two architecture
families. The crossed arm matters because it reverses which answer token is
correct. A fixed answer-token direction therefore weakens or reverses, whereas
DAS follows the binding. Dose-matched random edits, no-op checks, a whole-state
patch, and a closed-form difference-of-means baseline rule out generic
disruption, implementation error, an unresponsive site, and the cheapest
non-learned alternative.

**Fourth, the binding-language panel gives a qualified positive.** Now that DAS
has established causal use, this experiment asks what the used representation
is and whether
the J-lens surfaces the *language of binding*—words such as `scope`, `local`, or
`global`—at the use site. Both completed DeepSeek panels pass their predeclared
controls: `scope` is the strongest J-lens contrast for 1.3B at layer 9
(+7.645 and +7.637 in the crossed value arms), and `global` is strongest for
6.7B at layer 20 (-9.199 and -9.040). The near equality across the two value
assignments is important: the signal follows binding structure, not which
concrete value happens to be returned. The R-lens reproduces the result, often
with a somewhat larger contrast, and the logit lens also carries part of it.
Therefore the positive is a family-level binding-vocabulary signal, not a claim
that J-lens uniquely discovers one canonical word. StarCoder2 has no completed
semantic-concept panel, so cross-architecture replication is still absent for
this particular finding.

**Fifth, the secondary concrete-value test is negative before emission.**
J-lens recovers values perfectly at emission, establishing a working positive
control, but essentially never surfaces them at the three preceding positions.
Mid-network direction erasures are null or small and do not provide a consistent
advantage over the logit lens. The R-lens closely replicates this conclusion.

The strongest conclusion is therefore deliberately narrow:

> In these controlled programs, variable binding becomes linearly represented,
> remains stable under many surface changes but is fragile to structural
> interference, and is causally read from a rank-1 component at the use site,
> while the published J-lens does not surface the needed concrete value as a
> mid-network verbalizable token. It does surface controlled binding-related
> vocabulary in two DeepSeek models; the R-lens supports both conclusions.

These clauses are complementary, not contradictory. The concrete-value J-lens
null is a null for that published linear readout, not evidence that binding is absent;
the probe and DAS independently establish representation and causal use.

---

# Boundaries: what this project does not claim

- **Not that code models understand programs in general.** Every claim concerns
  named models, controlled programs, layers, sites, and readouts.
- **Not that probe accuracy proves use.** The representation claim is
  observational. Only the DAS intervention supports causal use.
- **Not that a failing frozen probe proves all binding information disappeared.**
  Robustness results describe whether one clean-trained linear readout transfers.
- **Not that binding is causally used everywhere.** DAS establishes transport at
  one use site and one selected layer per model, on one synthetic construction.
- **Not that the DAS direction is unique.** The closed-form mean direction also
  transports binding, less reliably and with a larger edit.
- **Not that the J-lens concrete-value null proves the value is absent.** It
  proves that this published linear, token-indexed readout does not surface it
  at the tested positions. The successful probes and DAS intervention
  independently show that binding information is present and used.
- **Not that the controlled isolation transfers to real code.** The exact 0.500
  floor relies on the synthetic paired construction.

---

# Open items

Ordered by how directly they would strengthen the active narrative.

1. **Explain why rank-1 DAS outperforms the whole-state patch.** The likely
   account—helpful and opposing donor components entering together—has not been
   independently tested.
2. **Test DAS at a second site.** Replication across architectures is complete;
   localization within each network remains narrow.
3. **Build context-matched mutations of real code.** This is required before the
   construction-pinned representational claim can be extended beyond synthetic
   programs.
4. **Add a cross-position string-equality baseline.** The current bounded surface
   reader cannot express the global feature “inner definition name equals use
   name.”
5. **Reconcile the configured and generated layer grids.** This would make
   cross-model relative-depth comparisons easier to audit.
