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
component is causally used at the variable-use site. Finally, an R-lens analysis
of the same programs shows that the answer score is reassigned from the inactive
definition toward the active one.

The security, output-vocabulary, standalone J-lens, and taint-routing studies are
preserved in [ARCHIVE.md](ARCHIVE.md). They are not needed to evaluate the active
claim.

Every active result below completed at canonical scale and is paired with the
control that could have falsified it. Read “represented” as *recoverable from a
hidden state*, “robust” as *the same frozen readout still transfers*, “used” only
where an intervention changes the downstream answer, and “attributed” as an
observational decomposition of an unchanged output score.

### Contents

- [The argument in one page](#the-argument-in-one-page)
- [Status at a glance](#status-at-a-glance)
- [Part I — The relation is represented](#part-i--the-relation-is-represented)
- [Part II — Robustness and failure boundaries](#part-ii--robustness-and-failure-boundaries)
- [Part III — Causal use and binding attribution](#part-iii--causal-use-and-binding-attribution)
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

**4. Binding attribution.** On the same contrast in DeepSeek-Coder 6.7B, the
R-lens assigns less answer relevance to the definition that leaves scope and more
to the one that enters scope. The effect is carried almost entirely by unchanged
definition tokens, survives crossed and fixed-output-token controls, reverses
when the competing answer is scored, and is absent when values change without a
binding change. This is observational and does not extend the causal claim.

**5. Verbalisation.** Asked in words whether the returned variable is *local* or
*global*, DeepSeek-Coder 6.7B is right on 0.900 of held-out programs, in both
option orders, while answering the value question at 1.000. Mapping the
answer-position state into vocabulary space, the inner-pole and outer-pole scope
words almost completely swap mass with the binding at layers 23–27 — and a
calibration-only ranking of the full vocabulary recovers an insideness cluster
(`inside`, `inner`, `within`, `interior`, `dentro`) without being given a
lexicon. The scope word family carries three times what the positional family
does. So the distinction is sayable, in some phrasings and not others, and it
becomes sayable much later in the network than it becomes represented or used.

---

## Status at a glance

| Stage | Instrument | Main evidence | Scope |
|---|---|---|---|
| **representation** | controlled linear probe | binding reaches ~0.984 over a 0.500 floor | DeepSeek 1.3B/6.7B; partial StarCoder2 replication |
| **robustness** | frozen-probe transfer | resilient to distance and renaming; fragile to scope interference and flattening | three tested models where reported |
| **causal use (R10/E13)** | rank-1 DAS interchange | 100% installed answer in both crossed arms | DeepSeek 6.7B and StarCoder2 3B |
| **binding attribution (R11/E16)** | conserving R-lens | 280/280 shifts on 6.7B; peak ~22% of answer score | interpretable on DeepSeek 6.7B only |
| **verbalisation (R12/E17)** | forced choice + output-vocabulary contrast | 0.900 on local/global; vocabulary poles swap at layers 23–27 | 6.7B; 1.3B untestable (its positive control fails) |

The R10/R11 labels are retained because generated reports and artifact names use
them. Missing numbers R5–R9 refer to studies now documented in
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

# Part III — Causal use and binding attribution

Part I established that binding and def–use relations are recoverable from hidden
states. Part II showed where those readouts remain stable and where they fail.
Part III asks the stronger question: **does the model use the binding
representation when producing its answer?**

Two experiments use the same controlled binding programs:

1. **DAS changes the model.** It replaces one learned component at the unchanged
   variable-use token and asks whether the emitted value follows the donor
   program's binding. This is the causal result.
2. **The R-lens does not change the model.** It divides the answer score among
   input roles and asks whether relevance moves from the definition that becomes
   inactive to the one that becomes active. This is an attribution result.

Reading them together is useful because they answer different questions on the
same examples. DAS establishes causal use at a tested site and layer. The R-lens
describes how the resulting answer is attributed across the source program. A
positive R-lens result is not extra causal evidence, and a null R-lens result
would not undo the DAS result.

The security benchmark, output-vocabulary study, standalone J-lens experiments,
and R-lens taint-routing experiment remain reproducible but are no longer part of
the main argument. Their methods, results, limitations, and artifact locations
are preserved in [ARCHIVE.md](ARCHIVE.md).

## Where the evidence lives

| Claim | Active method | Detailed generated reports |
|---|---|---|
| causal binding transport | [METHODS §5](METHODS.md#5-das--causal-interchange-of-a-binding-component) | [DeepSeek-Coder 6.7B](../results/binding/deepseek-coder-6.7b/e13_report.md), [StarCoder2 3B](../results/binding/starcoder2-3b/e13_report.md) |
| binding attribution | [METHODS §6](METHODS.md#6-r-lens-attribution-on-the-binding-programs) | [DeepSeek-Coder 6.7B](../results/binding/deepseek-coder-6.7b/e16_report.md), [DeepSeek-Coder 1.3B](../results/binding/deepseek-coder-1.3b/e16_report.md) |
| verbalisation | [METHODS §7](METHODS.md#7-verbalisation-of-the-binding-relation) | [DeepSeek-Coder 6.7B](../results/binding/deepseek-coder-6.7b/e17_report.md), [DeepSeek-Coder 1.3B](../results/binding/deepseek-coder-1.3b/e17_report.md) |

## R10 — DAS: the binding representation is causally used

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
| **answer direction** | tests whether DAS is merely an output-token push | works better on fitted `ab`, then attenuates or reverses on `ba` |
| **dose-matched random subspace** | tests generic disruption at a comparable edit size | much weaker than DAS |
| **rank-matched random subspace** | provides a simple random rank-1 floor | near zero |
| **no-op** | detects hook or measurement artifacts | exactly zero |
| **whole-state donor patch** | verifies that this site can change the answer in each arm | moves the answer in both arms |
| **mean donor−host direction** | tests the cheapest non-learned rank-1 alternative | may transport, but should be less effective or require a larger edit |

The answer-direction control is deliberately strong. It uses the validated
J-lens mapping to construct an effective `a → b` output push at the middle layer
and is scaled on every row to match DAS's edit norm. The J-lens does **not** find
the DAS direction; it is used only to implement this competing explanation.

### Result

The experiment succeeds in DeepSeek-Coder 6.7B and StarCoder2 3B, two different
architecture families:

| intervention | DeepSeek `ab` | DeepSeek `ba` | StarCoder2 `ab` | StarCoder2 `ba` |
|---|---:|---:|---:|---:|
| **DAS rank 1** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| whole state | 85.7% | 87.9% | 68.8% | 67.3% |
| mean difference | 76.1% | 76.8% | 54.6% | 54.5% |
| answer direction | 27.9% | 4.3% | 44.8% | 18.4% |
| dose-matched random | 2.1% | 1.8% | 30.9% | 30.4% |

The headline outcome is the full-vocabulary emitted token: whether the model
actually says the value selected by the installed binding. DAS reaches 100% in
both crossed arms. The explicit answer direction attenuates 6.9-fold on
DeepSeek and reverses in the relevant logit contrast on StarCoder2. DAS therefore
does not behave like a fixed push toward the answer required during fitting.

The closed-form mean direction is a real positive baseline, not a dead control:
it transports much of the binding. DAS nevertheless reaches 100% while changing
about 0.48 of the hidden-state norm, whereas the mean direction changes about
0.71. The learned direction works more reliably with a smaller edit.

### Conclusion and limits

At the tested use position and layer, downstream computation reads a compact
component whose effect follows **which definition is in scope** rather than a
fixed answer token. This is the project's causal finding.

It remains local to one synthetic binding construction, one site and one layer
per model. The rank-1 edit also outperforms the whole-state patch, so the latter
is not a true numerical ceiling; the plausible explanation that a full patch
adds both helpful and opposing components remains to be tested.

## R11 — R-lens: the answer is attributed to the active definition

### Question

R11 reuses the same binding programs but asks a different question. When the
binding flips, does the model's own answer score become attributed less to the
definition that leaves scope and more to the definition that enters scope?

Unlike DAS, the R-lens changes no activation and no output. It is observational.
Its value is that it connects the final answer back to named syntactic roles on
the same corpus where causal use has already been established independently.

### What the R-lens measures

The R-lens propagates the score of the bound value backward through the model
using layer-wise relevance rules. It divides the score among input positions and
then sums positions into roles: outer definition, inner definition name and
value, use site, signature, `return`, suffix, and residual text.

Before interpreting the decomposition, the experiment checks that:

- the special backward rules leave the forward output unchanged;
- the rules actually attach to the model's normalization, attention, and gated
  MLP modules;
- relevance assigned to all positions sums back to the selected output score;
- every input token belongs to exactly one role; and
- reading the same program twice gives exactly zero redistribution.

These checks pass with conservation error near numerical precision on the tested
DeepSeek models. The rules do not match StarCoder2's architecture, so no R-lens
claim is reported for that model.

### Isolation and controls

Within a binding-flip pair, exactly one of roughly 21 tokens changes: the inner
definition's **name**. The outer definition, inner value, use token, signature,
and answer suffix remain identical at the same token indices. The main statistic
therefore ignores the changed name and asks whether relevance moves between the
unchanged definitions.

| Control | What it rules out |
|---|---|
| **token-identical roles** | a local response to the changed name, token length, or positional drift |
| **crossed value arm** | attribution tied to one particular output token |
| **fixed-output-token conditions** | the same artifact under the strict condition that both programs are scored at literally the same token id |
| **competing-value score** | predicts that a binding-sensitive attribution should reverse |
| **same-binding pairs** | movement caused by changing values without changing the binding |
| **same-program reread** | numerical or backward-pass nondeterminism |

### Result on DeepSeek-Coder 6.7B

The attribution moves in the predicted direction on all 280 held-out bases. At
the first reported layer, the unchanged inner value belonging to the newly active
definition gains about 4.9% of the answer score, while the newly inactive outer
definition loses about 7.8%. The combined redistribution is about 12.6% there,
rises smoothly to approximately 21.9% near the middle of the network, and falls
to about 2.5% near the end.

The single token that differs between the programs contributes only about 1.5%
of the movement. Both crossed arms agree. Fixed-output-token conditions retain
the effect, scoring the competing value reverses it, and the two same-binding
controls remain approximately zero. The result is therefore not explained by
the changed token or by which answer token was selected for decomposition.

One control exposes an important limit. Pairing source and target programs from
different generated bases reproduces the average effect because every base uses
the same underlying template. The finding is consequently best described as a
stable property of this **binding contrast**, not as evidence of diverse
program-by-program routing. Effect sizes are more meaningful here than extremely
small p-values.

### Why the 1.3B result is not interpreted

DeepSeek-Coder 1.3B does not reliably solve the shadowing condition. About 7.6%
of the bound-value scores being decomposed are zero or negative. Dividing by a
near-zero or negative score makes the purported relevance shares unstable or
inverted, even though raw relevance still conserves. The observed directional
tendency is therefore suggestive but does not support an attribution claim for
this model.

### Conclusion and limits

On DeepSeek-Coder 6.7B, changing the binding reallocates the final answer score
from the definition that becomes inactive toward the one that becomes active,
mostly through tokens that did not change. This is consistent with the DAS
result but does not strengthen its causal status: DAS intervenes and measures an
answer change; the R-lens decomposes an unchanged forward computation.

The layer profile indicates where attribution under these backward rules is
redistributed, not where binding is computed. The attention rule also freezes
query/key pattern formation, so the experiment cannot establish the intuitive
mechanism “the model attends to the correct definition.” Finally, this is not a
verbalisation result. A future verbalisation study would need to test whether the
binding relation becomes expressible in meaningful output vocabulary or under a
matched prompt, rather than where an existing answer score is attributed.

## R12 — Verbalisation: the binding is expressed in the model's own scope words, late

### Question

R10 shows the models causally use a binding representation; R11 shows the
answer's relevance moves with the binding. Both are read in internal
coordinates. Does the distinction surface in anything the model **emits** — and
if it does, is that word attributed to the same definitions the value is?

This is [open item 6](#open-items) built as specified there: a new matched study
on the same corpus, with an output-vocabulary contrast and a relation-matched
positive control, not a reinterpretation of R11.

### Setup

Stages 150–153 over E13's factorial, unchanged: the same four programs per base,
the same frozen calibration/test split, the same anchor positions. A question is
appended, rendered from the **outer** name only — the letter both members of a
pair share — so the rendered question is byte-identical in all four cells and the
prompt still differs at exactly one token out of 35.

```
z = 6                        z = 6
def f():                     def f():
    d = 3                        z = 3
    return z                     return z
# Question: does f return the z assigned inside f or outside f? Answer:
#                → outside                        → inside
```

Four question styles, each asked in two variants, plus E13's own value question:

| style | choices | why this one |
|---|---|---|
| `scope` (declared primary) | inside / outside | names the construction, not a technical term |
| `binding` | inner / outer | the vocabulary R11's roles are named in |
| `pyscope` | local / global | Python's keyword pair |
| `shadow` | yes / no | single tokens under any tokenizer |
| `value` | the two literals | **the positive control** — E13's own forced choice |

**Chance is 0.500 by construction**: within a base the correct answer is "outer"
in two cells and "inner" in two, so a model that always answers the same way
scores exactly 0.500. Each two-option style is asked in both option *orders* and
the yes/no style in both *polarities*.

Ran on 400 bases for 6.7B (280 held out) and 200 for 1.3B (140 held out); 47 and
16 minutes respectively. Every mechanical control holds in both models: 800/800
and 400/400 binding-flip pairs differ at exactly the recorded mutation index;
the use token is identical in 1600/1600 and 800/800 contrasts; no rendered
question contains the inner definition's name and every one names the tracked
variable; the same-binding controls differ at exactly two indices as designed;
re-reading a prompt twice moves no fraction at all (max |Δ| = 0.0 over 12
re-reads); conservation is 1.3e-7 and 3.0e-7. Gates H7, H8, H9 pass in both.

Of 11 declared lexicon pairs, 10 survive both tokenizers — only `masked/exposed`
drops, because `masked` is multi-token on deepseek-coder — leaving all four
families represented.

### Result 1 — 6.7B can say it, in one vocabulary and not others

Held-out accuracy, both variants pooled, cluster intervals over bases:

| style | accuracy | 95% CI | says_inner | verbalised |
|---|---|---|---|---|
| `pyscope` (local/global) | **0.900** | [0.888, 0.913] | 0.523 | **yes** |
| `scope` (inside/outside) | 0.741 | [0.735, 0.746] | 0.261 | see below |
| `binding` (inner/outer) | 0.500 | [0.500, 0.500] | 1.000 | no |
| `shadow` (yes/no) | 0.500 | [0.500, 0.500] | 0.500 | no |
| `value` — positive control | **1.000** | [1.000, 1.000] | 0.500 | yes |

`pyscope` is the result. It clears chance in **both** option orders — 0.923
direct, 0.878 swapped — and its pooled `says_inner` of 0.523 shows it is not a
constant answer. Asked whether the returned variable is *local* or *global*, the
model is right nine times in ten.

The two nulls are informative rather than empty, because each fails in a
diagnosable way. `binding` answers " inner" for every program in both orders, so
the technical-term wording is answered by prior, not by reading. `shadow` answers
" yes" to both polarities — the acquiescence bias the polarity variant was built
to catch — which is why its accuracy is exactly 0.500 in both.

**And the declared primary style has to be reported as two numbers, not one.**
`scope` pooled to 0.741, but that is a mixture of a degenerate wording and a
near-perfect one:

| variant | accuracy | says_inner |
|---|---|---|
| `scope` direct ("inside f or outside f?") | 0.502 | 0.002 |
| `scope` swapped ("outside f or inside f?") | **0.980** | 0.520 |

One wording collapses to answering " outside" for essentially every program; the
other is near ceiling with a balanced answer rate. The variant control is exactly
what surfaces this, and the pooled row should not be quoted as an accuracy — a
design lesson recorded in [METHODS §7.4](METHODS.md#74-the-forced-choice-and-its-three-controls).
One further caveat on the 0.980: `argmax_is_a_choice` is **0.000** for this
style, so neither choice word is ever the model's own top continuation there,
whereas it is 1.000 for `pyscope`.

Two more controls, both on 6.7B:

- **Value independence.** Where the answer is not saturated, `pyscope` agrees
  across the arms on 0.793 and 0.854 of bases against a 0.500 floor, and
  `scope` swapped on 0.979 and 1.000. So the word largely, but not entirely,
  ignores which literal is returned. Agreement is trivially 1.000 wherever the
  model gives a constant answer, so only the non-saturated cells inform.
- **Dissociation is essentially absent.** The value is answered correctly on
  every cell, so `word_given_value` equals word accuracy: 0.923 / 0.878 for
  `pyscope` and 0.980 for `scope` swapped. Where 6.7B computes the binding it can
  usually also name it.

### Result 2 — the distinction is in output-aligned coordinates, and it arrives late

This is the strongest single piece of evidence here. Mapping the answer-position
state into vocabulary space and taking the paired change in (inner-word mass −
outer-word mass) over the frozen candidate set, oriented `target − source`:

| layer | Δ contrast prob | 95% CI | sign consistency |
|---|---|---|---|
| 0–19 | ≤ 1e-4 in magnitude, sign unstable across layers | | |
| 23 | **+0.780** | [+0.692, +0.869] | 0.846 |
| 27 | **+0.821** | [+0.710, +0.932] | 0.950 |
| 31 | +0.030 | [+0.015, +0.049] | 0.996 |

Both arms agree to three decimals at every layer. Between layers 19 and 23 the
candidate-set distribution goes from carrying essentially nothing to almost
completely swapping poles with the binding.

**The family split favours a scope reading over a positional one.** In the
z-scored convention, which is exact under this readout, at layer 27:

| family | Δz | sign |
|---|---|---|
| **scope** (local/global, inner/outer, inside/outside, nested/module) | **+0.444** | 1.000 |
| ordinal (second/first, later/earlier, new/original) | +0.145 | 0.996 |
| action (replaced/kept, changed/unchanged) | +0.122 | 1.000 |
| shadowing (hidden/visible) | +0.055 | 0.843 |
| random control (selected by no delta) | +0.024 | 0.989 |
| non-polar mechanism set | −0.042 | 0.011 |

`scope` is three times `ordinal` and eighteen times the random floor. Since
"the nearest assignment wins" is a purely positional rule that needs no notion of
scope, and `ordinal` is the family that would carry it, this comparison is
evidence for the scope reading. The non-polar mechanism set moves slightly
*negative*, so "the vocabulary of binding is in play" is **not** supported — only
the polar contrast is. The random floor is small but not zero, so a global shift
affecting all tokens exists and is an order of magnitude below the effect.

**Discovery found the words without being told them.** Ranking the full 32k
vocabulary by paired logit-lens delta on **calibration bases only**, the tokens
that rise most under the shadowing binding at layers 23–31 are:

> ` Inside`, ` inside`, ` Within`, ` interior`, `interior`, ` inner`, `inner`,
> `Inner`, `within`, `ins`, ` dentro`

A semantically coherent insideness cluster across casings and even across
languages, from a ranking that had no lexicon supplied. Only 18 of 432 discovered
rows were in the hand-written lexicon, so this is mostly vocabulary nobody
guessed, converging on the same families that carried the contrast. The falling
side is looser but not meaningless: ` wrong`, `old`, ` old`, `undefined`,
`extern`, ` fail`.

**The depth is the point.** The binding is linearly decodable in middle layers
(R1, ~0.984), causally transportable at layer 8 (R10), attributed at layer 15
(R11) — and becomes expressible in output-aligned words only at layers 23–27, 72
to 84% of the way through the network, falling again at the last layer.
Representation and use come early; sayability comes late.

### Result 3 — the attribution half did not work, for two separate reasons

Both are reportable as they stand, and neither is a result about the model.

**The pre-declared primary style turned out to be the degenerate one.** Stage 152
reads one style per run and defaults to `scope/direct` — which is precisely the
wording where the answer is a constant (0.502, `says_inner` 0.002). So the R-lens
was applied to a question the model is not answering. That is a design miss, and
the fix is a re-run with `--style pyscope` ([open item 3](#open-items)).

**The declared headline condition is ill-conditioned on this corpus.** The
headline scores the pole margin, chosen because a raw logit's sign is arbitrary
and `R_t/s` is a share only when `s > 0` — the failure that voided R11's 1.3B
readings. That protection worked exactly as intended: the margin's fractions are
sign-invariant and conservation holds at median |ρ−1| = 1.0e-6. But the guard
against *dividing by zero* is not a guard against *ill-conditioning*, and
|s_margin| ≈ 21 against single-pole logits of 303 and 325 means the fractions are
a difference of two near-cancelling large quantities over a small denominator. The
median ratio |s_margin| / max(|s_inner|, |s_outer|) is **0.064**, so every share is
inflated about **15.7×**; on 1.3B the ratio is 0.011 and the amplification 88×.
The consequences are visible and disqualifying:
mean shifts up to 0.53 of the answer score, a `same_inner` control interval of
[−0.77, +0.35], and the mismatched-pair control reproducing the treatment at a
ratio of 0.81–0.99 at every layer. Nothing can be read from it, and stage 153 now
measures the ratio, prints it per layer, and refuses to let the verdict's
grounding clause be read as answered when no layer clears 0.10.

**The secondary single-pole condition is readable on 6.7B, and it is clean.**
Both poles score positive on 1600/1600 readings (median logits 303 and 325), so
the share reading is licensed. At layer 0, on 280 held-out bases:

| role | Δ share | 95% CI | sign | token-identical |
|---|---|---|---|---|
| `inner_def_value` | +0.00135 | [+0.00121, +0.00149] | 0.871 | yes |
| `outer_def` | −0.00518 | [−0.00551, −0.00486] | 0.036 | yes |
| **`binding_shift_identical`** | **+0.00653** | [+0.00616, +0.00689] | **0.993** | yes |
| `inner_def_name` — the one differing token | −0.00004 | [−0.00042, +0.00035] | 0.496 | **no** |
| `question_all` | −0.00142 | [−0.00197, −0.00084] | 0.389 | yes |

The one token the counterfactual edits carries nothing (p = 0.824), the roles sum
to −2.5e-8, and the question text moves *less* than the definitions and in the
opposite direction — so under this condition the word's attribution is grounded
in the program, not in the question. The arms agree in sign at all eight layers
(ratio 0.99 → 0.62, both significant), and both same-binding controls are flat
(|mean| ≤ 0.0005 and ≤ 0.00013). The profile decays monotonically with depth,
+0.0065 at layer 0 to +0.0003 at layer 27.

**But it is suggestive, not established, and two things stop it short.** The
output-token control does not retain the effect the way R11's did:
`fixed_inner` is −0.0021 (sign 0.186) and `fixed_outer` +0.0033 (sign 0.896) —
*opposite* signs, so the redistribution is not independent of which pole is
scored. And no mismatched-pair control exists for this condition at all, because
that control is only formed for the pole modes. Direction matches R11 and
magnitude is about 5% of it (+0.0065 against +0.127), with an opposite depth
profile — R11 peaks at layer 15, this decays from layer 0.

Because the effect is negative at every layer under the declared headline, the
reported cell also exposes a defect in the selection rule: `select_verbal_cell`
maximises the mean, so on calibration it picked layer 27 — the one layer where
the margin shift is positive, and where it is not significant (p = 0.654). The
verdict `verbalised_not_grounded` is a function of that cell and should be read
as "the declared headline could not be evaluated", not as a finding about
grounding.

### The 1.3B model: nothing is learned, and the report says so

Every word style sits at exactly chance, and **the positive control fails**:
0.811, which is H1's own failure on this model (0.809). With the control below
ceiling, a null on the word styles cannot be told apart from a harness that could
not detect verbalisation if it were there — so the verdict is
`not_verbalised_instrument_untested` and nothing about 1.3B follows.

Descriptively, its word answers are pure position: under `scope` direct
`says_inner` is 0.038 (always the last-mentioned "outside") and under swapped it
is 1.000 (always the last-mentioned "inside"), for 0.500 accuracy either way.
Its single-pole scores are negative on 800/800 readings (median −126), so the
single-pole conditions are void — the R11 failure mode, this time **caught and
flagged by the gate** (`usable = 0`, `positive_layers = []`) rather than passing
unnoticed. The margin there is worse conditioned still (|s| ≈ 1.5 against poles
of 126) and returns shifts of 135% of the answer score. Discovery on 1.3B finds a
mutation cluster rather than a scope one — `created`, `changed`, `updated`,
`destroyed` — with 3 of 336 rows in the lexicon; with behaviour at chance and the
control failing, this is recorded as description and licenses nothing.

### What this adds to R10 and R11, and what it does not

It **does not strengthen the causal claim**. Nothing here intervenes; a word is
an output, and attribution of a word is still attribution. R10 remains the causal
result.

What it adds is a new observational fact with its own controls: the binding
distinction is expressed in the model's own output-aligned scope vocabulary, that
expression emerges in the last quarter of the network, and 6.7B can report the
distinction in words at 0.88–0.98 depending on phrasing while answering the value
question perfectly. The three results locate the phenomenon at three different
depths — causal use at layer 8, answer attribution peaking at layer 15,
vocabulary expression at layers 23–27 — and their units do not convert.

### Limits

- **Answering the question is not introspection.** The question is about the
  program, and the model can answer it by reading the text at inference time as
  any reader would. Nothing here separates a report about the model's own
  computation from a correct answer about the code.
- **Phrasing dominates.** Two of four styles are at chance, and the primary style
  spans 0.502 to 0.980 across two orderings of the same question. Four question
  forms and eight choice words is a sample of phrasings, not coverage.
- **The vocabulary contrast is mass within a ~160-token candidate set**, not over
  the full vocabulary, so "+0.821" is a share of that set. Both poles are in it
  and the swap is near-complete, but the number is not a probability over all
  tokens.
- **The attribution half is unresolved**, for the two reasons above.
- **The attn-rule blindness carries over from R11 unchanged**: with q and k
  detached, "attend to the right definition" remains the mechanism this
  instrument cannot see.
- Two models, one architecture family, one binding template, synthetic programs.

Method: [METHODS §7](METHODS.md#7-verbalisation-of-the-binding-relation).
Commands: [PIPELINE Part F.3](PIPELINE.md#part-f3--is-the-binding-verbalised-150153).
Reports: [DeepSeek-Coder 6.7B](../results/binding/deepseek-coder-6.7b/e17_report.md),
[DeepSeek-Coder 1.3B](../results/binding/deepseek-coder-1.3b/e17_report.md).

---

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

**Fourth, the answer is observationally attributed to the selected
definition.** On the same binding contrast in DeepSeek-Coder 6.7B, the conserving
R-lens moves answer relevance away from the definition that leaves scope and
toward the definition that enters scope. The movement is carried almost entirely
by token-identical text, not by the single changed name token. It appears in both
crossed arms, survives fixed-output-token conditions, reverses when the competing
value is scored, and is absent in same-binding controls.

**Fifth, the distinction is expressible in the model's own words, and late.**
Asked whether the returned variable is local or global, DeepSeek-Coder 6.7B
answers correctly on 0.900 of held-out programs in both option orders, while
answering the value question at 1.000 — so there is no dissociation to speak of
on that phrasing. In vocabulary space the two word poles almost completely swap
mass with the binding, but only at layers 23–27, and the scope word family
carries three times what the purely positional family does. Two of four phrasings
sit at exactly chance, each for a diagnosable reason, so this is a claim about
some ways of asking and not about the concept.

The strongest conclusion is therefore deliberately narrow:

> In these controlled programs, variable binding becomes linearly represented,
> remains stable under many surface changes but is fragile to structural
> interference, is causally read from a rank-1 component at the use site, is
> reflected in how the final answer is attributed to the active definition, and
> becomes expressible in output-aligned scope vocabulary in the final quarter of
> the network.

These clauses are complementary, not interchangeable. DAS supports causal use.
The R-lens supports attribution. The verbalisation result supports expressibility
in output coordinates. They locate the phenomenon at three different depths —
layer 8, layer 15, layers 23–27 — and their units do not convert. None
establishes a complete mechanism, and none shows that a correct verbal answer is
introspective rather than a correct reading of the program text.

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
- **Not that the R-lens proves causation.** It decomposes an unchanged answer
  score. Its agreement with DAS is a conjunction of two different findings.
- **Not that the R-lens identifies the attention mechanism or computation
  layer.** Its rules freeze attention-pattern formation, and its layer profile is
  a profile of attribution under those rules.
- **Not that a correct verbal answer is introspection.** R12's question is about
  the program, and the model can answer it by reading the text at inference time
  as any reader would. Nothing separates that from a report about its own
  computation.
- **Not that the binding is verbalisable in general.** Two of four question forms
  sit at exactly chance and the primary form spans 0.502 to 0.980 across two
  orderings of the same question. R12 bounds the phrasings tested, not the
  concept.
- **Not that the word's relevance has been located.** R12's attribution half is
  unresolved: it was run on the one wording the model answers with a constant,
  and its declared headline condition proved numerically ill-conditioned.
- **Not that the controlled isolation transfers to real code.** The exact 0.500
  floor relies on the synthetic paired construction.

---

# Open items

Ordered by how directly they would strengthen the active narrative.

1. **Re-run the DeepSeek 6.7B DAS report under the final H5 discriminator.** The
   underlying interchange rows are unchanged, but the generated gate file still
   contains the superseded logit-margin verdict.
2. **Re-run R12's relevance stage on a wording the model actually answers.**
   Stage 152 defaults to the declared primary style `scope/direct`, which turned
   out to be the one wording the model answers with a constant — so the R-lens
   was read on a question it is not answering. `--style pyscope` (0.900 in both
   orders) is the right target, and `--style scope --variant swapped` (0.980) is
   the second. This is the single cheapest thing that would resolve R12's
   attribution half; nothing else about R12 depends on it.
3. **Condition, not just guard, the pole-margin denominator.**
   `MIN_MARGIN_RELATIVE = 1e-6` prevents division by zero but not
   ill-conditioning, and |s_margin| ≈ 21 against pole logits of 303 and 325
   inflates every fraction about fifteenfold. The declared headline condition is
   therefore unreadable on this corpus while conservation reports 1.0e-6 and
   notices nothing — the same shape of failure as R11's sign problem, one level
   further in. The threshold should be on
   |s_margin| / max(|s_inner|, |s_outer|), reported per layer beside conservation
   and positivity, and the run should fall back to the single-pole conditions
   where it fails.
4. **Fix `select_verbal_cell` for two-sided outcomes.** It maximises the mean, so
   when the true effect is negative at every layer it selects the least-negative
   one — on 6.7B that picked layer 27, the only layer where the sign flips and
   where the effect is not significant (p = 0.654). It should select on
   |effect| or on a declared direction.
5. **Add a second binding template for R11 and R12.** The current mismatched-base
   control lacks power because all bases share one template. A structurally
   different shadowing construction would test whether the attribution follows
   binding beyond that template contrast, and would give R12's vocabulary
   contrast a second construction to replicate on.
6. **Backport R12's positive-score gate to R11.** R12 measures the rate per
   (layer, pole) and gates on it, and it worked: on 1.3B the single-pole
   conditions came back `usable = 0`, `positive_layers = []`, so the failure that
   silently voided R11's 1.3B result was flagged before interpretation this time.
   R11 itself still has no such check. Changing a gate after seeing the result it
   would have caught is what the gate discipline exists to prevent, so this waits
   for an R11 re-run rather than a patch.
7. **Explain why rank-1 DAS outperforms the whole-state patch.** The likely
   account—helpful and opposing donor components entering together—has not been
   independently tested.
8. **Test DAS at a second site.** Replication across architectures is complete;
   localization within each network remains narrow.
9. **Extend R12 beyond four phrasings.** Two of four question forms sit at
   exactly chance and the primary form spans 0.502 to 0.980 across two orderings,
   so phrasing currently dominates the result. Candidate next steps: the words
   discovery surfaced but the lexicon never asked with (`within`, `interior`), a
   free-generation rather than forced-choice readout, and a second model family
   to test whether `pyscope` beating `binding` is about Python's keywords or
   about this model.
10. **Build context-matched mutations of real code.** This is required before the
   construction-pinned representational claim can be extended beyond synthetic
   programs.
11. **Add a cross-position string-equality baseline.** The current bounded surface
   reader cannot express the global feature “inner definition name equals use
   name.”
12. **Reconcile the configured and generated layer grids.** This would make
   cross-model relative-depth comparisons and the DAS/R-lens layer comparison
   easier to audit.
