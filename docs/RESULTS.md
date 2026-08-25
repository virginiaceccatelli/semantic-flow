# Results

## How each result is explained

Every result below follows the same reading order: the question being tested,
why it matters, the expected outcome, the experimental comparison, the exact
result, and the conclusion that result supports. Read “represented” as
*recoverable from a hidden state*. Read “used” only where an intervention changes
the downstream computation. Those are different claims, and the experiments
keep them separate.

The overall picture is simple. The models construct binding and data-flow
information as hidden states pass through the network. That information is not
explained by the local words or token distances used in the controls. It usually
survives renaming and irrelevant context, but it degrades sharply when the same
names create interference or when control flow is flattened. The security
property has a reliable distributed direction in output space, although no
single security word carries it. Finally, the binding intervention shows that
the tested model uses the learned subspace at the tested site. Exact model-,
layer-, sample-, accuracy-, interval-, and control-specific results remain in
the sections below.

Every result in this file **completed at canonical scale, passed its own
controls, and clears the floor it set for itself**. Each is stated in the same
four-part form — *research question*, *hypothesis*, *method*, *result* —
followed by what it does and does not mean. Methodology is
[METHODS.md](METHODS.md).

Work that was retired, parked, superseded, or that failed a bar it later set is
in [ARCHIVE.md](ARCHIVE.md), with the reason in each case. The data behind all of
it is preserved and every stage still runs.

Nothing here is a summary of intent. Every row is a measurement that exists in
`results/tables/*.csv` or in a per-run report under `results/`, and every claim
is paired with the control that could have falsified it.

### Contents

- [The argument in one page](#the-argument-in-one-page)
- [Status at a glance](#status-at-a-glance)
- [Part I — The relation is represented](#part-i--the-relation-is-represented) — R1 binding, R2 def–use
- [Part II — What the representation is made of](#part-ii--what-the-representation-is-made-of) — R3 interference, R4 atomic attribution, R5 the security audit
- [Part III — What form it is in](#part-iii--what-form-it-is-in) — which lens does the work; R6 the R-lens, R7 the output-basis direction, R8 the positive control, R9 relevance routing
- [Part IV — Whether the model uses it](#part-iv--whether-the-model-uses-it) — R10 the DAS interchange
- [Synthesis](#synthesis-what-this-says-about-semantic-understanding)
- [Boundaries](#boundaries-what-this-project-does-not-claim)
- [Open items](#open-items)

---

## The argument in one page

The project has four instruments of increasing strength (see
[METHODS §0–§8](METHODS.md)), each asking a harder question about relations
drawn from the program's own **code property graph**.

**1. The relation is there, and it is computed rather than read.** Which
definition an identifier refers to is linearly decodable from mid-network states
at **0.984**, over a floor pinned to **exactly 0.500 by construction** — the
model-free surface reader and the embedding layer both sit at 0.500 because the
two programs are token-identical. It is absent at the input, built within the
first few transformer blocks, and partly shed near the output. Def–use follows
the same profile, with honest decay by distance.

**2. It is made of control structure, not of names or distances.** A thousand
tokens of inert filler cost almost nothing; filler that reuses the tracked names
drives binding to chance. Renaming every identifier leaves middle layers at
0.85–0.90 while pushing the embedding layer *below* chance. Applying the four
obfuscating rewrites **one at a time**, in three models: opaque predicates and
arithmetic rewriting cost **exactly nothing**, renaming costs 0.01–0.12, and
**control-flow flattening alone costs 0.31–0.34** — within 0.03 of what the whole
composition costs, with the interaction inside measured draw noise.

**3. The same holds for a property an auditor would actually ask for.** "Is the
value at this `os.system` / `cursor.execute` / `eval` argument source-derived?"
reads at **1.000** on held-out programs over *two* measured chance floors, and
fails under exactly the same one transformation. What survives flattening is each
model's class prior, not retained flow information.

**4. The distinction is in the models' own output coordinates — and no word
carries it.** Differencing each matched pair over the **whole 32k-token
vocabulary** finds a direction that **72 of 72 held-out pairs project positively
onto, in every model**, over a token-identity floor of *exactly zero*. Its
top-loading tokens are meaningless fragments. At the one cell where a readout
*does* fire on a property these models express (0.85–0.94, tracking the model's
own answer margin), the security lexicon at that same cell is significantly
**inverted** — so the gap between *decodable* and *verbalised* is a fact about
the models, not a limitation of the instrument.

**5. And for binding, the model's own computation reads it.** A rank-1,
magnitude-free **DAS interchange** at the binding-resolution site, fitted on one
arm of a 2×2 and evaluated on the arm it never saw, makes the model emit the
value the *installed binding* selects on **100.0% of held-out rows in both
arms** — where a token-direction or answer-direction account demands the opposite
movement.

---

## Status at a glance

| Result | Instrument | 1.3B | 6.7B | SC2-3B | Finding |
|---|---|:--:|:--:|:--:|---|
| **R1** variable binding | probe | ● | ● | ◑ | decodable from mid layers over a construction-pinned 0.500 floor |
| **R2** def–use edges | probe | ● | ● | ◑ | same floor, same profile, mild distance decay |
| **R3** context interference | frozen probe | ● | ● | ☐ | survives 1000 tokens of filler; collapses under scope interference |
| **R4** atomic obfuscation | frozen probe | ● | ● | ● | flattening alone breaks binding and def–use; the boundary is general |
| **R5** source→sink audit | frozen probe | ● | ● | ● | 1.000 over **two** floors; **flattening alone** causes the whole collapse |
| **R6** R-lens conservation | lens instrument | ● | ● | n/a | conservation within 1e-4 at every layer; the **gated-MLP rule dominates by 4.5×**, falsifying the pre-registered prediction. Does not apply to StarCoder2 |
| **R7** full-vocabulary direction | vocabulary contrast | ● | ● | ● | **72/72** held-out pairs project positively; onset ~25% depth; collapses under flattening alone; loadings are meaningless |
| **R8** positive control | vocabulary contrast | ● | ● | ● | the readout is **not blind** — 0.85–0.94 on a property the models express, while the security lexicon at the same cell is *inverted* |
| **R9** relevance routing | R-lens attribution | ● | ● | n/a | the chain feeding the sink loses relevance share on token-identical text — **65/72** pairs on 1.3B, **64/72** on 6.7B, where the mean-based control fires too |
| **R10** binding interchange | DAS | ☐ | ● | ● | rank-1 transport at **100% in both arms in two architecture families**, beating a closed-form baseline at two-thirds the dose |

Legend: ☐ not run · ◑ partially run · ● run at canonical scale

**Recently moved to [ARCHIVE.md](ARCHIVE.md):** control dependence (its surface
floor is 0.927, so it was never a representational result) and the security-lexicon
vocabulary contrast (at the reported cell it does not beat a random direction on
one of three models, and both halves of its reading were superseded by R7 and R8).

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

# Part II — What the representation is made of

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
- Control-flow flattening is the only tested rewrite that causes a large,
  reproducible collapse across binding, def–use, and source-to-sink flow.

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

R4 is also the **companion control** for R5: it establishes that this boundary is
a general limit of frozen linear readouts of program relations, so R5's failure
cannot be read as a security-specific fragility.

---

## R5 — The source→sink security audit

### Research question

Binding and def–use are the mechanism; *"untrusted data reaches `os.system`"* is
the property an auditor actually wants and an adversary has an interest in
hiding. So: **is the value that reaches a code-bearing, security-sensitive
argument derived from untrusted input; does a readout of that fact, frozen on
clean programs, survive obfuscation; and which transformation breaks it, on its
own?**

### Hypothesis

Three, tested simultaneously:

1. The property is decodable well above any lexical floor — including one that
   reads the **entire program text**, not just a token window.
2. If R4's account is right, the *same* transformation that breaks binding —
   control-flow flattening — breaks this too, and the ones that do not change
   flow cost nothing.
3. Residual accuracy after a damaging transformation is **not** retained flow
   information but class prior, and pooled accuracy cannot tell the difference.

### Method

Full construction in [METHODS §5](METHODS.md#5-the-security-benchmark-e15-construction-threat-model-metrics).
In brief: 3 sink families × 4 flow structures × 20 base seeds × 2 labels = **480
clean programs**; each base a matched unsafe/safe pair differing **only at the
sink argument**, checked character-exactly; 14 seeds per cell train, 6 held out,
**only held-out programs transformed**; the readout fitted once on clean training
programs and frozen. Ten conditions — clean, `normalize`, four **atomic** arms and
four **cumulative** arms — 1296 held-out variants per model.

**Four control arms, all frozen and all transferred**: `local_surface` (±3 token
ids at the anchor), `whole_program_lexical` (token n-grams + char 3–5-grams over
the **entire** program), `embedding` (layer −1), and `hidden_state`.

Ground truth recovered independently for every program and every variant by
instrumented execution *and* a static taint fixpoint that must agree (METHODS
§5.3). No dangerous API is ever executed.

Read at the layer nearest **48% of network depth** (1.3B L11, 6.7B L15,
starcoder2-3b L15 — in all three also the argmax of clean-training CV). Cluster
bootstrap over bases. **All six gates pass in every run, with no overrides.**

### Result 1 — the property is there, and it is not the text

On clean training programs the local surface baseline is 0.488–0.491, the
**whole-program lexical baseline** is 0.464, and the embedding layer is 0.482 —
all chance — while accuracy reaches **1.000** (both deepseek models) and **0.997**
(starcoder2-3b) near half depth. Held out, across all ten conditions, the floors
never leave chance: local surface 0.431–0.521, whole-program lexical 0.465–0.535,
embedding 0.451–0.569.

### Result 2 — flattening alone accounts for the entire collapse

| condition | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| clean held-out / normalize | **1.000** | **1.000** | **1.000** |
| `rename_only` | 0.938 | 0.986 | 0.882 |
| `opaque_only` | **1.000** | **1.000** | **1.000** |
| `encode_only` | **1.000** | **1.000** | **1.000** |
| **`flatten_only`** | **0.688** | **0.667** | **0.660** |
| full ladder (rename+opaque+encode+flatten) | 0.729 | 0.653 | 0.674 |

Three things follow:

1. **Two of the four transformations are free.** Opaque dead branches and
   mixed-boolean-arithmetic rewriting score *exactly* 1.000 on their own, in
   every model.
2. **Flattening alone costs 0.312 / 0.333 / 0.340**, within 0.03 of what the
   entire four-transformation composition costs.
3. **The interaction is null.** Cumulative minus atomic is +0.042 / −0.014 /
   +0.014, against a *measured* draw-noise floor of 0.021 / 0.035 / 0.028
   (`rename_only` vs `rename_cumulative` — identical transformation, independent
   draws). Composition adds nothing distinguishable from noise.

**So the claim is "control-flow flattening breaks the readout", not "the
cumulative ladder does".** The earlier marginal phrasing is retired on evidence.

### Result 3 — what survives flattening is class bias, not flow

At `flatten_only`, unsafe/safe accuracy is 0.625/0.750, **0.833/0.500** and
0.667/0.653, with 51% / 56% / 46% of matched pairs receiving the **same** label.
Under the full ladder the biases run in *opposite* directions — 6.7B toward
"unsafe" (0.861/0.444), starcoder2-3b toward "safe" (0.569/0.778). A constant
predictor of either class scores exactly 0.500 on this balanced set, so residual
accuracy that biases oppositely across models is each model's own prior.

### Result 4 — the dangerous errors arrive before any structural change

Under renaming **alone**, starcoder2-3b is 0.882 pooled but **0.764 on unsafe
against 1.000 on safe** — the entire loss is **false negatives**, with the control
flow untouched. By flow structure, the **assignment chain is the fragile one
under renaming** in all three models (0.778 / 0.972 / 0.639) against
`branch_merge` at 1.000 everywhere. By sink family nothing reproduces, which is
the null the design wanted: the readout tracks *flow*, not which dangerous API
sits at the end of it.

### What it means

An auditor's property behaves exactly like the graph relations underneath it. The
frozen readout is not reading text — two independently measured lexical floors say
so — and it fails under exactly one of four semantics-preserving rewrites. Since
R4 shows binding and def–use break the same way, the supported claim is
*"structural obfuscation breaks frozen linear readouts of program relations,
security ones included"* — **not** "security representations are specifically
fragile".

Result 4 is the reason METHODS §5.6 insists on disaggregated metrics: a pooled
0.882 hides a one-sided false-negative failure with the control flow untouched,
which is the failure direction an auditor cares about most.

### What it does not establish

- The floor is pinned only against *declared* feature families. Both measured
  floors sit at chance, but a reader that ran the taint analysis itself would
  score 1.0. This is an audit of a **readout's transfer**, not an R1-style
  construction-pinned representational claim.
- Eight transformation arms, not the full 15-combination lattice.
- "Flattening breaks the readout" is a statement about a **frozen linear readout
  at one position**. A failing probe does not prove the model lost the
  information — though R7's and R8's parallel failures are consistent with real
  loss.
- Nothing causal is claimed or tested for the security property.

---

# Part III — What form it is in

*These experiments are observational. They describe what can be read from the
model, not what the model causally needs.*

## Current status in plain language

The lens experiments support one strong representational result and one weaker
attribution result:

1. **Strong, but obtained with the simple logit lens:** safe and unsafe programs
   differ along a repeatable direction in the model's output vocabulary space.
   The direction generalises to every held-out pair, but it is spread over many
   unrelated token dimensions rather than concentrated in words such as
   `unsafe`. This is evidence for a distributed, output-aligned representation,
   not for an explicit semantic label.
2. **Replicated, obtained with the R-lens:** the model redistributes a small
   amount of answer relevance between the two data-flow chains even where their
   text is identical, in both DeepSeek models. On 6.7B the effect clears every
   declared control, including the permutation test of the mean that fails on
   1.3B. The magnitude stays small — 1–2% of the answer score — and the method
   does not apply to StarCoder2, so this is one architecture family measured
   twice.

The **J-lens has no surviving semantic result**. It passes its engineering
checks and improves generic next-token recovery, but its semantic findings
either fail the required controls or are reproduced by the cheaper logit lens.

## Which lens matters here

| lens | did it work mechanically? | did it add a semantic result? |
|---|---|---|
| **logit lens** | yes | **yes** — R7 and the R8 positive control |
| **J-lens** | yes | **no** |
| **R-lens** | yes on DeepSeek; not applicable to StarCoder2 | **yes** — R9 relevance routing, replicated on both DeepSeek models |

The extra complexity of the J- and R-lenses did not improve the vocabulary-space
conclusions. At the strongest positive-control cell, all three give essentially
the same sign consistency: **0.889** for 1.3B, approximately **0.8–0.85** for
6.7B, and **0.944** for StarCoder2. The plain logit lens is therefore the right
tool for the main semantic summary.

## R6 — Does the R-lens conserve relevance?

### Method

R6 is an instrument test. It checks that the modified backward rules leave the
forward model unchanged and that the relevance assigned to earlier positions
adds back to the chosen output score. It then removes each rule in turn to see
which correction matters.

### Result

The gate passes on both DeepSeek models:

| check | DeepSeek 1.3B | DeepSeek 6.7B |
|---|---:|---:|
| final-layer agreement with logit lens | **1.0000** | **1.0000** |
| median early/middle conservation error | **0.0000** | **0.0001** |
| layers where R-lens beats raw autograd | **7/7** | **9/9** |

The most important correction is the 50/50 split at the gated MLP. Without it,
the conservation error is about **4.4**, far larger than the quantity being
conserved. This falsifies the original expectation that normalization would be
the dominant issue.

StarCoder2 is out of scope: its normalization and MLP do not match the implemented
rules, so the pipeline correctly refuses to treat its backward pass as an
R-lens.

### What this means

The R-lens is a valid attribution instrument for the tested DeepSeek models. R6
does **not** show semantic understanding; it only makes the later routing test
interpretable.

## R7 — The distinction is in the output basis, and it is not a word

### Method

For each matched pair, the ordinary logit lens produces scores for all roughly
32,000 vocabulary tokens at the final program position. This position contains
the same token in both programs, eliminating token identity as an explanation.

The training pairs define the average unsafe-minus-safe vocabulary direction.
That direction is frozen. Held-out pairs then test whether the unsafe program
projects farther along it than its matched safe program. Same-label pairs test
whether ordinary differences between programs produce the same effect.

### Result

| clean held-out pairs | 1.3B | 6.7B | StarCoder2 3B |
|---|---:|---:|---:|
| pairs in the predicted direction | **72/72** | **72/72** | **72/72** |
| mean cosine with training direction | **0.383** | **0.380** | **0.390** |
| identical-token embedding floor | **0** | **0** | **0** |

Same-label controls remain near chance. The direction first becomes clear about
one quarter of the way through the network and then persists. Flattening the
program's control flow weakens it sharply, paralleling the probe results.

The important limitation is what the direction looks like. Its largest token
loadings are unrelated fragments such as `Lemmon`, `椒`, and `bootstrapcdn`, and
the weight is spread thinly across thousands of tokens. The direction also fails
the preregistered dominance test (`0.76 / 0.97 / 0.76`, below the required
`2.0`): it is repeatable, but it is not the largest difference between programs.

### What this means

The models contain a stable safe/unsafe difference in their own output
coordinates. But the lens does not reveal a human-readable semantic variable.
The most defensible description is **distributed output alignment**: the
distinction is statistically present across the vocabulary without being
represented by a word that names it.

This is also why vocabulary lenses are limited tools for semantics. They are
built to ask “which output-token directions does this state resemble?” Abstract
program relations need not be organised like words. A strong distributed
direction can demonstrate a consistent internal difference, but it cannot tell
us that the model represents the human concept *unsafe* or understands data
flow in the human sense.

## R8 — The positive control: the readout is not blind, and the security words run backwards

### Method

To check that R7's lack of meaningful security words is not merely a blind
instrument, the model is explicitly asked a yes/no taint question. The readout
tests whether the unsafe member receives a larger `yes`-minus-`no` margin than
the matched safe member.

### Result

All models choose the same answer token for every program, so ordinary accuracy
is **0.500**. Their graded margins nevertheless order the pairs correctly in
**0.694 / 0.750 / 0.917** of cases. At the best internal layer, lens sign
consistency is **0.889 / 0.847 / 0.944**.

Thus the measurement can recover a property when the prompt makes the model
express it. This makes R7's lack of meaningful token loadings informative rather
than a simple measurement failure. It does not show reliable task performance:
the final choices remain biased, and the 6.7B result reverses under another
prompt wording.

## R9 — The unique R-lens result, now replicated on both DeepSeek models

### Method

The conserving R-lens divides an answer score among syntactic roles. Only the
sink argument differs between each safe/unsafe pair; the two data-flow chains are
textually identical. The test asks whether relevance moves between those
unchanged chains when a different chain is connected to the sink.

### Result

The chain feeding the sink loses relevance share and the other gains, in both
models, at each model's reported cell (72 held-out clean pairs):

| | 1.3B (layer 0) | 6.7B (layer 11) |
|---|---|---|
| taint chain shifts down | **65/72** | **64/72** |
| trust chain shifts up | **63/72** | **57/72** |
| median taint-chain shift | −0.014 | −0.021 |
| sign-test *p* | 7.0e−13 | 5.8e−12 |
| permutation *p* on the mean | 0.57 — **fails** | **0.000** |
| verdict | `redistribution_consistent_but_not_in_mean` | `redistribution_found` |

Relevance conserves essentially exactly in both — median |ρ−1| of 1e−7 and 2e−7,
worst case 5e−5 and 3e−6, at every layer read — so the role shares are a genuine
partition of the model's own answer and a paired difference is a redistribution
rather than a change of scale.

**The mean-based control that fails on 1.3B fires on 6.7B.** On 1.3B the
per-pair deltas are heavy-tailed enough that the mean and the sign disagree on
several roles, and the statistic that survives is the sign, whose exact null
under the same random-orientation scheme is the sign test in the table above. On
6.7B both statistics agree, and all five declared checks hold.

**The depth does not replicate.** What is being located is the *paired* pattern —
taint chain down and trust chain up in the same cell. On 1.3B it is present at
layers 0 and 3 — 90% and 86% of pairs move the taint chain down while 88% and
85% move the trust chain up. The taint side is gone by layer 11; the trust side
stays elevated to layer 19. On 6.7B layers 0 and 3
do not show it at all: the two chains move together or sit at chance there, and
at layer 0 the trust chain moves *down*. The paired pattern appears at layer 7,
peaks at layer 11, holds at 15, and is gone by 19. In relative depth that is
0–13% against 23–48% — not the same place in the network. It is not an artifact
of which output token the relevance is taken for: within each model both target
tokens give the same profile.

### What this means

R9 shows that the model routes its computation differently depending on which
data-flow chain reaches the sink, on text that is identical at the roles being
measured. It is now a replicated result rather than a single-model clue. Three
limits remain: the effect is small (**1–2%** of the answer score), it depends on
the chosen attribution rules, and it is confined to the DeepSeek family because
the method does not apply to StarCoder2. It is observational — it describes where
an answer is attributed, not what the model uses.

## Bottom line for semantic understanding

- The lens evidence shows a **reliable internal difference** between safe and
  unsafe programs in output coordinates.
- It does **not** show a discrete or human-readable “unsafe” concept. The signal
  is distributed across unrelated token dimensions.
- The J-lens adds no semantic evidence beyond the simple logit lens.
- The R-lens routing result is a confirmed observational finding on the DeepSeek
  family, small in magnitude and not consistent in depth across the two scales.
- None of these observations proves causal use or semantic understanding. The
  intervention experiments in Part IV are required for claims about what the
  model actually uses.


# Part IV — Whether the model uses it

*Instrument 4: DAS interchange
([METHODS §8](METHODS.md#8-instrument-4--das-magnitude-free-interchange-on-a-learned-subspace)).
Everything in Parts I–III is correlational: a representation can be a faithful
shadow of a computation happening somewhere else. This part is the only causal
evidence in the project.*

## R10 — A rank-1 interchange transports which definition is in scope

### Research question

Does a low-rank, **magnitude-free** interchange at the site where a binding is
resolved transport *which definition is in scope* — rather than a token, or an
answer direction?

### Hypothesis and the falsification that identifies it

The design is a 2×2 crossing **binding structure** with **value assignment**, so
that the *same* binding flip demands **opposite token movements** in the two
arms. The alignment is fitted on arm `ab`; the claim is read on arm `ba`.

| account of what the subspace carries | arm `ab` | arm `ba` |
|---|---|---|
| *which definition is in scope* | positive | **positive** |
| *the token `b`* | positive | **negative** |
| *the answer* | positive | **negative** |

This is a **refutation** design, not a support design: a subspace that encodes the
answer is falsified by the held-out arm rather than merely left unsupported.
There is **no arithmetic anywhere** — the model returns a variable — and the
interchange has **no dose parameter**, so "was the edit big enough?" is not a
question this design has to answer (METHODS §8.2).

### Method

Stages 100–108 on **deepseek-coder-6.7b** (site `use`, layer 8, rank 1) and on
**starcoder2-3b** (site `use`, layer 11, rank 1). 400 base programs each (120
calibration / 280 test); site and layer chosen on calibration and recorded before
test numbers were read. Six hard gates, each refusing to run downstream stages.
Seven control arms (METHODS §8.4). Outcome metric is `says_installed`, the
**full-vocabulary argmax**, not the logit margin, for the reason in METHODS §8.6.
Cluster bootstrap over bases.

The two models are different architecture families — Llama-style with RMSNorm and
a gated MLP, against StarCoder2's LayerNorm and non-gated MLP. That matters here
in a way it does not elsewhere in this document: it is the same *design* run on a
genuinely different network, not a rerun at another scale.

### Result — the gates

**All six gates pass in both models.**

| gate | 6.7B (layer 8) | StarCoder2-3B (layer 11) |
|---|---|---|
| **H0** generation and independent ground truth | **PASS** — 400/400 bases, all six invariants at 1.0000 | **PASS** — 400/400, all six at 1.0000 |
| **H1** the model returns the bound variable | **PASS** — 1.000 overall | **PASS** — 0.981 [0.973, 0.988]; weakest cell 0.954 |
| **H2** the binding is decodable at the use anchor | **PASS** — 1.000 vs a measured floor of 0.500 | **PASS** — 1.000 vs the same 0.500 floor |
| **H3** whole-state interchange flips the answer, per arm | **PASS** — ab +4.781, ba +4.799; flip 0.857 / 0.879 | **PASS** — ab +1.731, ba +1.761; flip 0.663 / 0.659 |
| **H4** low-rank beats matched controls on the training arm | **PASS** — 189% of ceiling; edit moved 0.479 of ‖h‖ | **PASS** — 434% of ceiling; edit moved 0.478 of ‖h‖ |
| **H5** the same subspace transfers to the held-out arm | **PASS** — 100.0% installed; `answer_direction` arm ratio 0.154 vs transport's 1.025 | **PASS** — 100.0% installed; `answer_direction` arm ratio **−0.157** vs transport's 1.002 |

H1 and H2 are worth pausing on: with no arithmetic anywhere, both models resolve
these bindings at or near ceiling, and which definition is in scope is perfectly
decodable at the use anchor against a floor pinned to 0.500. That is a cleaner
replication of R1's isolation than R1 itself, on a corpus built for intervention.

> **A bookkeeping note, stated because it matters.** For **6.7B only**, the
> `gates.yaml` and `e13_report.md` on disk still record **H5 as FAIL**, because they were written
> under the superseded `delta_ld` (logit-margin) discriminator and stages 106–107
> have not been re-run since the rule was corrected to `says_installed` on
> 2026-08-13. **The underlying rows are unchanged** — every number in the table
> below is read directly from `interchange_summary.csv` — and the full record of
> the rule change, including the verdicts under both rules, is in
> [ARCHIVE.md §5](ARCHIVE.md). Re-running stages 106–107 would regenerate the
> gate file; nothing else changes. StarCoder2's gate file was written after the
> rule change and records H5 as PASS, so it needs no such note.

### Result — the transport, and every control

A rank-1 subspace at the use anchor (layer 8), fitted on arm `ab` alone, makes the
model emit the value the *installed binding* selects on **100.0% of held-out rows
in both arms** — 280 base programs, 560 rows per cell:

| variant | `ab` emits installed | `ba` emits installed | edit fraction |
|---|---:|---:|---:|
| **`das_binding`** (rank 1, learned) | **100.0%** | **100.0%** | **0.479** |
| `whole_state` (the entire donor state, rank-`d` limit) | 85.7% | 87.9% | 0.805 |
| **`mean_difference`** (rank 1, closed form) | 76.1% | 76.8% | 0.711 |
| `answer_direction` (norm-matched) | 27.9% | 4.3% | 0.479 |
| `random_norm` (dose-matched random) | 2.1% | 1.8% | 0.513 |
| `random_rank` / `noop` / raw unembedding | 0.0% | 0.0% | 0.018 / 0 / 0.479 |

All 14 machinery checks pass: structural zeros exactly 0.00e+00 (`noop` at both
sites, `whole_state` at the pre-mutation site), alignment orthonormal to 4.07e-07,
the ceiling alive in both arms, and the model emits a non-candidate token on
**0.0%** of rows.

### The same result in a different architecture family

StarCoder2-3B, rank 1 at the use anchor (layer 11), fitted on arm `ab` alone,
280 test bases, 560 rows per cell:

| variant | `ab` emits installed | `ba` emits installed | edit fraction |
|---|---:|---:|---:|
| **`das_binding`** (rank 1, learned) | **100.0%** | **100.0%** | **0.478** |
| `whole_state` (the entire donor state) | 68.8% | 67.3% | 0.804 |
| **`mean_difference`** (rank 1, closed form) | 54.6% | 54.5% | 0.717 |
| `answer_direction` (norm-matched) | 44.8% | 18.4% | 0.478 |
| `random_norm` (dose-matched random) | 30.9% | 30.4% | 0.540 |
| `random_rank` / `noop` | 2.3% | 1.4% | 0.019 / 0 |

The learned subspace lands in the same place: **100% in both arms at essentially
the same dose** (0.478 against 6.7B's 0.479), transfer ratio **1.002**. And the
falsification bites harder here than it did on 6.7B — the explicit answer
direction does not merely attenuate across the arms, it **reverses sign** (arm
ratio −0.157), which is the strongest form of the signature the 2×2 was built to
produce.

Two differences from 6.7B are worth stating rather than leaving for a reader to
find. The dose-matched random control is far livelier — **30.9%** against 6.7B's
2.1%, at a dose 13% above the treatment's — so StarCoder2's answer at this site is
easier to disturb in general. It is still separated decisively (the paired
contrast `das_binding − random_norm` is **+7.05 [+6.97, +7.12]**), but the margin
is bought by the treatment being near-perfect, not by the control being dead. And
`answer_direction` pushes the model off-candidate on 10.9% of `ab` rows, where the
treatment never does.

**On the structural zeros.** StarCoder2's four structural-zero checks report
`False`, and this is a threshold artifact rather than a machinery fault. 58.6% of
`noop` rows are exactly zero and every non-zero one is a multiple of **0.03125** —
precisely one fp16 unit-in-last-place at this model's logit scale (~36–38) — with
a maximum of two ulps and a mean of +1.1e−5 whose interval straddles zero. The
check uses an *absolute* 1e−4 bound, which is below what fp16 can represent here.
The comparison that makes this legible: before the layer-keying bug in stage 106
was fixed, the same statistic was **−0.129 [−0.141, −0.119] with a maximum of
0.719** — systematically biased and 23 ulps wide. That was a real fault; this is
quantisation. The bound should be made scale- and dtype-aware, the same correction
already applied to the R-lens R0 check.

### What this refutes, rather than merely fails to support

**Not disruption.** The dose-matched random subspace is *over*-dosed — 0.513 of
‖h‖ against the treatment's 0.479 — and at that larger dose produces the installed
answer on ~2% of rows against 100%, while the model never emits a non-candidate
token.

**Not an answer direction.** The explicit answer direction attenuates **6.9×**
across the arms (27.9% → 4.3%) while the treatment does not attenuate at all, and
it pushes the model off-candidate on 9.1% of rows where the treatment never does.
This is exactly the asymmetry the 2×2 was built to produce, and it is the
falsification earlier designs could not construct.

### The closed-form baseline transports too — and loses

The learned direction sits at |cos| 0.673 from the mean donor−host difference:
substantially aligned, not identical, and no cosine can say whether the optimiser
earned the rest. So the difference-in-means direction was run as its own arm — no
optimiser, no labels, one fixed direction for every example.

It works: **76.1% / 76.8%**, transfer ratio 1.003. A fixed direction really does
carry much of the binding, and that was worth knowing.

It also loses. `das_binding` reaches **100% while moving 0.479 of ‖h‖;
`mean_difference` reaches 76% while moving 0.711** — roughly twice the effect per
unit of dose. And the learned direction captures *less* of the raw state
difference (59.5% against 88.2%), so it is not a better-aligned version of the
same object. It is a different direction that works better while disturbing the
state less.

### What it means

At this site, in **both** models, the downstream computation **reads** a rank-1
subspace whose content is *which definition is in scope*. Not the token. Not the
answer. That is the affirmative causal result in the project, and the design
refutes the two competing accounts rather than merely failing to support them.
Because the two models are different architecture families, the result is no
longer a statement about one network's idiosyncrasy.

### What is still open

**Why a rank-1 edit outperforms the whole-state patch**, which is now the largest
loose end rather than a footnote. On 6.7B the rank-1 edit reaches 100% against the
full donor state's 86%; on StarCoder2 it reaches 100% against **68.8%**, and in
logit-margin terms it is **434% of the whole-state ceiling**. The "ceiling" is
therefore not functioning as one in either model, and more strongly in the second.
The available explanation — that the full patch installs the driving component
*and* components that fight it — is plausible and remains **not** independently
demonstrated. A gate expressed as a fraction of a ceiling that the treatment
exceeds four-fold is a threshold that has stopped doing work, and it should be
reformulated before this is written up.

Also still open: one site, one layer, one construction. What is no longer open is
the model count.

---

# Synthesis: what this says about semantic understanding

Read across the four instruments, the results compose into a single account with
a definite shape.

**Program structure is the right level of description, and it is available in the
model.** The relations tested here are drawn from the code property graph —
binding and def–use from the DFG, source→sink reachability from the PDG. They are
linearly decodable from mid-network states well above what any bounded surface
reader achieves. The models are not merely tracking text; they are tracking a
structure over the text that has an independent, exactly computable definition.
And the criterion excludes things: control dependence, whose surface floor is
0.927, does not qualify and is not reported as a result.

**The representation is built, layered, and shaped like a computation.** Binding
is exactly absent at the input, appears within a few blocks, peaks mid-network and
is partly shed before the output. Interference that makes scope resolution
*harder* destroys it while interference that only makes the program *longer* does
not, and the damage lands on the middle layers rather than on the input
representation. That profile is what a computed relation looks like and is not
what a lookup table looks like.

**What it is made of is control structure.** The single most informative result in
the robustness track is negative in an unusually clean way: two of four
semantics-preserving rewrites cost *exactly nothing*, one costs a little, and one
— control-flow flattening — costs essentially the entire effect, with no
measurable interaction between them. The representation is therefore not anchored
to identifier strings (renaming is survivable mid-network) nor to expression form
(MBA rewriting is free), but to the control skeleton the relation is defined over.
Dissolve the skeleton and the frozen readout goes with it. That is a real limit
and a real prediction: any tool built on these representations degrades on
obfuscated control flow specifically, not on obfuscation in general.

**The security property is not special — and that is the interesting part.** "Is
this dangerous argument attacker-controlled?" behaves exactly like binding and
def–use: ceiling-level decodability over two independently measured lexical
floors, and failure under exactly the same one transformation. It is at least as
robust as the primitives it rests on. So the security result is not a story about
security representations being fragile; it is a story about a general property of
frozen linear readouts of program relations, demonstrated on the case where the
consequences are legible.

**Format and content come apart.** The distinction the probe reads at 1.000 is
present in the model's **own output coordinates** — 72/72 held-out pairs on every
model, over a floor of exactly zero — and yet is carried by **no word**. Where a
readout demonstrably fires on a property these models express, the security
lexicon at that same cell runs *backwards* on two of three models. Interpretability
that looks for concepts by asking "which token lights up" would have concluded the
property is absent. It is not absent; it is distributed.

**There is an order of operations on 1.3B, and it does not survive scale.** On
DeepSeek 1.3B relevance routing differs between the two members at 0–13% depth,
an output-aligned direction appears at 30%, and the answer becomes sayable at
40–65%: the property is *routed* differently before it is *representable* in
output coordinates, and representable long before it is *sayable*. On 6.7B the
first two coincide — routing and the output-aligned direction both begin at layer
7 (23% depth) and both peak at layer 11 (35%). The routing itself replicates; the
claim that routing *precedes* representation is a 1.3B statement and should be
made as one.

**And at least one of these representations is causally used, in two architecture
families.** The DAS interchange settles for binding what probing structurally
cannot: a rank-1, dose-free edit at the resolution site installs *which definition
is in scope*, in both arms of a factorial where a token account and an answer
account each demand the opposite movement. Both models read that subspace, at
essentially the same dose, and on StarCoder2 the answer-direction control reverses
outright rather than merely fading.

**One methodological result about the tools themselves.** Of three
output-basis readouts, the two expensive ones change no conclusion when used as
vocabulary projections — the plain logit lens matches them wherever a result
actually fires. The R-lens pays for itself only in a different role, as a
*conserving attribution* over input positions, and that role produced the only
evidence in this project that no vocabulary projection can reach — including, on
1.3B, the earliest-depth evidence in the depth sequence. A validated instrument
that turns out to be unnecessary for most of the questions at hand, but decisive
for one, is worth reporting as such.

**What the whole thing licenses, stated conservatively.** These models compute
program-structural relations, build them with depth, anchor them to control
structure rather than to surface form, express at least one of them in
distributed output-aligned coordinates without lexicalising it, report it only
when asked and only as a ranking, and — for binding — actually use it. Every one
of those verbs is doing separate work, and the evidence for each comes from a
different instrument. Collapsing them into "the model understands the program"
would discard exactly the distinctions the project exists to draw.

---

# Boundaries: what this project does not claim

- **Not that code models "understand" programs.** Every claim is a decoding,
  format or intervention result at named sites under named controls.
- **Not that binding is causally used *in general*.** R10 shows a rank-1
  interchange transports the binding at one site, one layer, two models, one
  synthetic construction. Four earlier designs failed to establish even that, each
  for a different recorded reason ([ARCHIVE.md](ARCHIVE.md)).
- **Not that the R10 subspace is the *only* direction that transports.** The
  closed-form difference-in-means direction reaches 76% in both arms; the learned
  one reaches 100% at two-thirds the dose. The claim is that it dominates that
  baseline, not that the baseline is inert.
- **Not that the isolation transfers to real code.** The 0.500 floor exists only
  in synthetic programs; a real-code transfer number is an upper bound on what
  transfers semantically, not a representational finding.
- **Not anything about control dependence.** Its surface floor is 0.927; it is
  archived, not reported.
- **Not that the model "represents unsafe".** R7 and R8 replace that sentence with
  a weaker and more precise one about output-aligned distributed format.
- **Not that the security lexicon is silent everywhere.** On 6.7B it separates the
  pair at the answer position of a prompt that asks the question.
- **Not that a failing frozen probe proves the model lost the information.** R4's
  and R5's flattening results are statements about a frozen linear readout at one
  position.
- **Not that anything about the security property is causal.** No intervention of
  any kind was run on it. R10's interchange covers binding.
- **Not that the 0.500 floor is pinned against *every* computable text feature.**
  It is pinned against the stated surface baselines. A cross-position
  string-equality baseline is outside the ±3 window and is an open item.
- **Not that R9's routing effect is large.** Median 1–2% of the answer. It now
  replicates on 6.7B, where the mean-based permutation control fires as well; on
  1.3B that control still does not fire and the effect there rests on the sign
  test alone. The two models also route at different depths.

---

# Open items

Ordered by what would most change what this project can claim.

1. **Re-run stages 106–107 to regenerate R10's gate file** under the
   `says_installed` discriminator, so the on-disk verdict matches the reported
   one. No number changes; this is bookkeeping that a reader will check.
2. **Explain R9's depth shift.** The redistribution replicates on 6.7B and clears
   every declared control there, but it sits at layers 7–15 (23–48% depth) rather
   than 1.3B's layers 0–3 (0–13%), with each model flat where the other fires.
   Both target tokens give the same profile within each model, so this is not a
   target artifact and needs an account. The per-layer CSVs are already on disk;
   no GPU time is required to start.
3. **Explain, or bound, the rank-1 edit beating the whole-state patch** (100% vs
   86% at 60% of the edit norm). The available account is plausible and untested,
   and a reviewer will ask.
4. **A second *site* for R10, and an account of the 434% ceiling.** The second
   model is done — StarCoder2-3B reproduces 100%/100% at layer 11 — so what
   remains is a second anchor and, more urgently, an explanation of why the
   rank-1 edit exceeds the whole-state patch by four-fold there. Related: make
   `verify_structural_zeros` scale- and dtype-aware, so an fp16 run stops
   reporting one-ulp noise as a machinery failure.
5. **Explain the `assign_chain` fragility.** It has replicated in three models
   under renaming *alone* — starcoder2-3b drops to 0.639 there while
   `branch_merge` stays at 1.000. Diagnose on the existing
   `sinkflow_predictions.csv` before spending any GPU.
6. **Make the R-lens architecture-general, or bound it.** It does not apply to
   starcoder2-3b at all, which is what confines R9 to one model family. Extending
   `norm_eps_attr` to LayerNorm and `is_gated_mlp` to non-gated MLPs is the open
   work; the LayerNorm half is harder, since the mean-subtraction term is what the
   current algebra assumes away.
7. **Context-matched pairs on real code** — the highest-value follow-up for the
   foundation, and the one thing that would let R5's floor argument extend beyond
   synthetic programs. Build by mutating real functions.
8. **A cross-position string-equality surface baseline** in the probe stage. The
   current baseline cannot represent "the inner definition's name equals the
   use's name", which is the feature a lexical adversary would use. CPU-only.
9. **Reconcile `configs/models.yaml` with `MODEL_REGISTRY`** so declared
   `probe_layers` are the ones that actually run. It is why the three models sit
   on different layer grids and why every cross-model number must be read at
   matched relative depth.
