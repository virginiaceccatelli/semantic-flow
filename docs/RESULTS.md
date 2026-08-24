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
| **R9** relevance routing | R-lens attribution | ● | ☐ | n/a | the chain feeding the sink loses relevance share, 63–65 of 72 pairs, on token-identical text |
| **R10** binding interchange | DAS | ☐ | ● | ☐ | rank-1 transport at 100% in both arms, beating a closed-form baseline at two-thirds the dose |

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

*Instrument 3: output-basis readouts
([METHODS §6–§7](METHODS.md#6-instrument-3--the-lens-stack-logit-j-lens-r-lens)).
Everything in this part is observational: a projection or an attribution, never
an edit.*

## Which lens is doing the work — the short answer

Three lenses exist in this repository (logit, J-lens, R-lens), and it is worth
saying plainly which of them earns its cost, because the answer is not the one
the track was built expecting.

**As a vocabulary projection, none of them beats the plain logit lens.** At the
one cell where a vocabulary readout actually fires (R8), the three are
interchangeable:

| taint sign consistency, `sink` prompt, clean held-out | logit | J-lens | R-lens |
|---|---:|---:|---:|
| 1.3B (L19) | 0.889 | 0.889 | 0.889 |
| 6.7B (L27) | 0.847 | 0.792 | 0.847 |
| SC2-3B (L29) | 0.944 | 0.944 | 0.944 |

Their mean vocabulary-difference vectors agree at pairwise cosine 0.75–0.97
elsewhere. **The Jacobian correction and the LRP rules change none of the
conclusions in this file that come from projecting onto tokens.** They are
validated instruments — the J-lens is provably exact at the last layer and
recovers next-token content the logit lens cannot (+0.15 to +0.22 top-1
pre-final; METHODS §6.3) — but for *this* task that fidelity buys nothing, and
saying so is more useful than defending the expense.

**The R-lens earns its keep in exactly one role, and it is not as a lens.**
Because its rules make the traversed tail degree-1 homogeneous, `Σ_t R_t = s`
holds, so `R_t / s` is a genuine **partition of the model's own answer across
input positions**. That is an attribution, not a projection, and no logit lens
can produce it. It is what makes **R9** possible at all, and R9 is the only
result in this file that could not have been obtained with a logit lens.

**The J-lens produces no surviving result.** Its validation is correct instrument
work and lives in METHODS §6.3; its two intended uses were archived when their
controls failed, and where it was used as a coordinate system the plain logit
lens was *more* efficient at the same site ([ARCHIVE.md §1.2–§1.3](ARCHIVE.md)).

So Part III contains one instrument result whose *finding* is about LRP itself
(**R6**), two results obtained with an ordinary output-basis projection (**R7**,
**R8**), and one that requires the R-lens as a conserving attribution (**R9**).

## The argument of Part III, in four sentences

1. **A conserving backward pass exists on gated-MLP transformers, and the rule
   that makes it work is not the one that was predicted.** *(R6)*
2. **The safe/unsafe distinction is present in the models' own output
   coordinates** — a label-defined direction generalises to held-out programs in
   72/72 pairs on every model, above a token-identity floor that is exactly zero
   — **but it is carried by no word**: its top loadings are meaningless
   fragments. *(R7)*
3. **The readout is not blind, which is what makes (2)'s second half mean
   something.** Run on a property the models *do* express, the identical readout
   finds it at 0.85–0.94 and converges on the model's own answer margin — while
   the security lexicon at that same cell is significantly *inverted*. *(R8)*
4. **Below vocabulary space entirely, the model routes identical text
   differently.** Whichever data-flow chain feeds the sink loses relevance share
   and the other gains, in 85–90% of matched pairs at early layers. *(R9)*

**The headline the track supports:** *the safe/unsafe distinction is present in
output-aligned coordinates, distributed across the vocabulary, and not carried by
any word for it.* "Decodable" and "verbalised" come apart, and — because the
positive control fired — **the gap is a fact about the models, not a limitation
of the instrument.**

---

## R6 — A conserving backward pass, and the rule that carries it

### Research question

An ordinary gradient lens transports raw autograd through modules that are **not
degree-1 homogeneous**, so relevance is not conserved and a mid-layer reading has
no fixed interpretation. **Is a more faithful backward pass available on real
code models, how would we know, and which algebraic fix actually does the work?**

### Why this is a result and not only instrument validation

Two of its three findings are transferable claims that hold beyond this project:
*which* rule carries the faithfulness gain on a gated-MLP transformer (the
pre-registered prediction was wrong), and a general diagnostic for detecting that
LRP rules silently failed to install. The validation itself is the prerequisite
for R9.

### Hypothesis

Installing the LRP rules — RMSNorm → diagonal, SiLU → elementwise, gate split
50/50, attention pattern frozen — makes the traversed tail degree-1 homogeneous,
so relevance conservation

> `ρ = Σ_t ⟨∂s/∂h_t , h_t⟩ / s`

should hold near 1 at every layer by Euler's identity. The design **predicted the
LN-rule would dominate** the ablation.

### Method

Gate R (stage 110), described in full in
[METHODS §6.4](METHODS.md#64-the-r-lens-what-it-is-what-it-fixes-and-how-we-know).
Four required checks — forward invariance (R0), last-layer equality (R1), LRP
beating raw autograd at every testable layer (R2a), conservation in early layers
(R2b) — plus a **rule ablation** (R2c) that removes one rule at a time and is
reported, not gated.

### Result — gate R passes on both DeepSeek models

| check | 1.3B (float32) | 6.7B (float16) |
|---|---|---|
| **R0** forward invariance — the rules change no activation | 1.62e-06 relative (tol 1e-04) | 1.21e-03 relative (tol 1e-02) |
| **R1** last layer equals the logit lens | cosine **1.0000** | cosine **1.0000** |
| **R2a** LRP beats raw autograd at every testable layer | **7/7** | **9/9** |
| **R2b** conservation, median \|ρ−1\| over early/mid layers | **0.0000** | **0.0001** |

The all-rules arm holds `ρ ≈ 1` to within 1e-4 at *every* layer including the
embedding (measured `median_rho` 1.0003 at layer −1 through 0.99996 at layer 31
on 6.7B). Under raw autograd, by contrast, `ρ` wanders and inverts sign with
depth.

### Result — the ablation falsifies the prediction, and replicates

| rule removed | 1.3B | 6.7B |
|---|---:|---:|
| **`no_half`** (gated-MLP split) | **4.4203** | **4.4628** |
| `no_ln` (RMSNorm → diagonal) | 0.9806 | 0.9885 |
| `no_identity` (SiLU → elementwise) | 0.2265 | 0.3941 |
| `no_attn` (attention hooks) | 0.5128 | 0.3044 |

The **half-rule dominates by ~4.5×**, and the ordering is near-identical across
two models and two dtypes. What makes the traversed tail homogeneous is
overwhelmingly the **gated-MLP split**, not the norm; without it the conservation
error is *larger than the quantity being conserved*.

`no_ln` is the second-order effect and a total one — removing it drives `ρ` to
~0.01, i.e. the relevance essentially vanishes. Attention, deliberately left
unmodified in the published formulation, costs 0.30–0.51: real, bounded, and the
honest answer to "what does the unmodified softmax path cost".

One earlier anomaly resolves here. An fp16 run had reported the *identity-rule
making conservation worse*. In float32 the all-rules arm sits at |ρ−1| = 0.0000
at every layer against 0.2265 without the identity rule — so the rule helps, and
the inversion was fp16 noise rather than a property of SiLU.

### Result — it does not apply to StarCoder2 at all, and the tell is general

Gate R **cannot complete** on starcoder2-3b, for an **architectural rather than
numerical** reason. StarCoder2 uses LayerNorm (deliberately unmatched: it
subtracts the mean, so the rule's algebra differs) and a non-gated MLP, so
`norm_eps_attr` and `is_gated_mlp` both decline and the two homogenising rules
bind to **nothing**. Only the attention hooks register; stage 110 then raises
when its `no_attn` arm removes the only rule that bound.

**The diagnostic is worth carrying elsewhere**: the one file it produced reports a
forward delta of **exactly 0.0**. Value-preserving rules still perturb float
arithmetic; rules that were never installed do not. **An R0 that passes
*perfectly* is the signature of an empty install.** Gate J0 now refuses this case
(`rlens_rules_bound`).

### What it means

For anyone building an attribution method on a gated-MLP transformer, the gate's
**bilinearity — not the norm — is where autograd's double-counting lives**, and
that now has a number attached to it in two models and two dtypes. For this
project, conservation holding to 1e-4 is the precondition that makes R9's
relevance *partition* a partition rather than an uninterpretable ratio, and where
it does not hold the stage refuses to run rather than emitting raw autograd under
the name relevance.

---

## R7 — The distinction is in the output basis, and it is not a word

### Research question

The probe (R5) says what a *fitted* direction can recover, in a basis of its own
choosing. It cannot say whether the model's **output-aligned** coordinates carry
the distinction. So: **is the safe/unsafe distinction present in output-aligned
coordinates at all — and is it carried by any single token?**

An earlier attempt asked this against a hand-picked 196-token security lexicon
and returned a null. That attempt is archived
([ARCHIVE.md §4.4](ARCHIVE.md#44-the-security-lexicon-vocabulary-contrast)),
because at its reported cell the effect does not beat a random direction on one
of three models, and because it can only find a concept that some token *in its
pool* carries. This result removes the pool.

### Hypothesis

Two pre-declared and separable predictions:

- **generalisation** — a direction estimated from the *training* pairs' full
  vocabulary differences will be projected onto positively by *held-out* pairs;
- **dominance** — the label axis will be the largest axis of variation among the
  difference vectors, operationalised as `sv1_ratio ≥ 2.0` against a same-label
  null.

### Method

Stage 128. Form each matched pair's difference over **all ~32k tokens**, z-scored
per member; estimate the mean direction on the training split; project held-out
pairs onto it. Nothing is chosen in advance, so a null cannot be blamed on a
pool.

The primary site is **`last_token`**, not `sink_arg`, because it is the only site
where both members carry the same token id in 100% of pairs — which makes the
layer −1 surface floor **exactly zero**, since identical tokens give identical
embeddings. Concentration is measured with the sign-invariant
`sv1_share = λ_max(UUᵀ)/trace(UUᵀ)` against both a random floor (`1/n`) and a
same-label null.

### Result — the direction generalises, decisively

| clean held-out, `last_token`, mid-depth | 1.3B (L11) | 6.7B (L15) | SC2-3B (L15) |
|---|---:|---:|---:|
| held-out pairs projecting positively | **72/72** | **72/72** | **72/72** |
| mean cosine with the frozen direction | **0.383** | **0.380** | **0.390** |
| cluster-bootstrap 95% CI | [0.360, 0.406] | [0.358, 0.401] | [0.351, 0.429] |
| same-label control, same direction | 0.507 | 0.465 | 0.549 |
| token-identity floor (layer −1) | **exactly 0** | **exactly 0** | **exactly 0** |

Every number that could make this an artifact was measured and came back clean. A
random direction in 32 256 dimensions gives a cosine near 0.006, so 0.38 is
roughly **sixty standard deviations out**. The same-label control — two programs
of the *same* label, everything else varying — sits at chance on the same
direction. And on the **44 of 72 pairs whose two members have identical token
counts**, the last-token position index is identical too, so 100% sign
consistency cannot be a length or position effect: you cannot be right on all 72
without being right on those 44.

### Result — a clear onset depth, identical in all three models

| relative depth | ≈0.0 | ≈0.13 | ≈0.25 | ≈0.35–0.50 | ≈0.75–1.0 |
|---|---:|---:|---:|---:|---:|
| layers | −1, 0 | 3 | 7 | 11–15 | 19–31 |
| sign consistency (1.3B / 6.7B / SC2) | 0.49/0.46/0.46 | 0.46/0.49/0.50 | **1.00**/0.79/0.71 | 1.00/1.00/1.00 | 0.90–1.00 |

Nothing before a quarter of the way through the stack; then it appears and holds
to the output. Two model families, three scales, one curve.

### Result — it degrades exactly as the probe does

Projection sign consistency at the reported layer, across the ten conditions:

| | clean | normalize | rename | opaque | encode | **flatten** | full cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.3B | 1.000 | 1.000 | 0.958 | 0.903 | 1.000 | **0.819** | 0.722 |
| 6.7B | 1.000 | 1.000 | 0.986 | 0.917 | 1.000 | **0.708** | 0.625 |
| SC2-3B | 1.000 | 0.944 | 0.889 | 0.903 | 0.972 | **0.681** | 0.681 |

The mean projection falls by ~93% under flattening alone and by 4–14% under
renaming, opaque predicates or MBA encoding. **This independently replicates R5's
headline with a completely different readout** — nothing is fitted here, no probe
is trained, and the conclusion is the same.

### Result — but the direction is not a concept, and it does not dominate

Two facts keep this from being "the model has an unsafe feature", and both are
reported.

*Its top loadings are meaningless.* The tokens carrying it are `' Lemmon'`,
`'egraphics'`, `'ateral'`, `'uta'` (1.3B); `' mel'`, `'椒'`, `' Jonathan'` (6.7B);
`'bootstrapcdn'`, `'%%%%%%%%%%'`, `'pmatrix'` (SC2). Loadings are **flat** —
0.019 to 0.027 across the top twelve — i.e. the direction is spread thinly over
thousands of tokens rather than concentrated in a few. **It is output-aligned but
not lexicalised**, and this is the direct evidence for that claim: it needs no
hand-picked vocabulary and no null.

*The label axis is not the largest axis of variation.* The pre-declared criterion
`sv1_ratio ≥ 2.0` measured **0.76 / 0.97 / 0.76**. It **failed on all three
models**, and it fails because two programs of the same label already differ along
a dominant shared axis of comparable size. The verdict recorded is
`direction_replicates_but_not_dominant`.

**Those two statements are compatible and both are reported.** The projection asks
*does a label-defined direction generalise to unseen programs* — yes, decisively.
The concentration asks *is the label axis the biggest thing separating these
difference vectors* — no. What settles that they are **different axes** rather
than one: the frozen direction's top-100 loadings overlap what the *same-label*
differences find at a Jaccard index of **0.005 / 0.005 / 0.000**. Nearly disjoint.

### What it means

The distinction lives in the model's own output coordinates from a quarter of the
way up the stack, spread across the vocabulary, and no word carries it. This is a
claim about **representational format**, which is stronger than "a probe can find
it" and weaker than "the model uses it" — it remains observational.

It also carries a methodological warning: an interpretability method that looks
for concepts by asking *which token lights up* would have concluded this property
is absent. It is not absent; it is distributed.

---

## R8 — The positive control: the readout is not blind, and the security words run backwards

### Research question

Any claim of the form "the model does not verbalise X" has an ambiguity that no
negative control can resolve: *the model does not verbalise it* versus *this
readout could not detect verbalisation if it were there*. So: **can this readout
detect verbalisation at all — and when it demonstrably can, what do the security
words do at that same cell?**

### Hypothesis

Four outcomes were declared in advance, including the one that would retire the
track. If the identical readout finds a property the models demonstrably express
and does *not* find the security lexicon at the same cell, the "not verbalised"
reading is about the models (`machinery_validated`). If it finds neither, it is
about the method (`machinery_blind`).

### Method

Stage 129 runs the **identical** measurement — same `pair_contrast` call, same
z-score convention, same orientation, **one** candidate basis carrying both token
sets, with gate J3 refusing the run if the two bases ever differ — on a
forced-choice taint question whose answer is a single token.

The behavioural statistic is **`pair_separation`** (the fraction of bases where
the unsafe member draws a higher yes-margin than its matched safe counterpart),
not accuracy, because a model with a fixed answer bias earns 0.5 accuracy for
free while pair separation's chance level of 0.5 is immune to answer bias. Two
prompt styles are run so prompt sensitivity is measured rather than assumed.

### Result — step 1: can the models answer it at all?

By **accuracy**, no. Every model scores **0.500**, and the reason is visible in
one column: each has a fixed answer bias whose argmax never moves. 1.3B answers
"no" to every program; 6.7B and StarCoder2 answer "yes" to every program.

But the *graded* margin does separate the pair:

| clean held-out, `sink` prompt | 1.3B | 6.7B | SC2-3B |
|---|---:|---:|---:|
| accuracy | 0.500 | 0.500 | 0.500 |
| answers the same token for every program | yes ("no") | yes ("yes") | yes ("yes") |
| **pair separation** | **0.694** | **0.750** | **0.917** |
| sign-test p | 0.0013 | 0.0000 | 0.0000 |

So the property *is* expressed behaviourally, **as a ranking rather than as a
decision**. This is precisely why the design fixed on the paired statistic in
advance: had the verdict read accuracy, every model would have returned
`property_not_verbalised` and the stage would have concluded nothing.

### Result — step 2: the readout finds it, and the security words invert

| best taint cell, R-lens, `sink` prompt | 1.3B (L19) | 6.7B (L27) | SC2-3B (L29) |
|---|---:|---:|---:|
| relative depth | 0.83 | 0.87 | 0.93 |
| **taint sign consistency** | **0.889** | **0.847** | **0.944** |
| permutation p | 0.000 | 0.000 | 0.000 |
| lens tracks the model's own margin | 0.708 | 0.806 | 0.917 |
| **security lexicon at the same cell** | **0.347** | 0.764 | **0.389** |
| its permutation p | 0.000 | 0.000 | 0.006 |

The readout's agreement with the model's own forced-choice margin rises
monotonically with depth and reaches **0.95 / 0.98 / 0.98** at the final layer —
which it must, because there the lens *is* the output head. That is the internal
validity check on the whole instrument, and it passes.

**This is the strongest form the "not lexicalised" claim takes anywhere in the
project**, because it is a within-cell dissociation on one instrument at one
position: on 1.3B and StarCoder2, the taint poles separate the pair at
0.889/0.944 while the security words separate it at 0.347/0.389 — *significantly
in the wrong direction*, both p < 0.01. The readout is demonstrably alive at that
cell. The security vocabulary is not merely silent there; it is inverted.

### Result — 6.7B differs, and it must be reported as such

There the security lexicon *also* fires (0.764, p = 0.000) — the
`both_properties_detected` outcome. Read it precisely: this is at the **answer
position of a prompt that asks the question**, at 87% depth, not at a sink
argument in unprompted code. The two contrasts are only weakly correlated
(r = +0.26), so the security signal is not merely the taint signal wearing another
name; but "vulnerable/safe" moving with "yes/no" where a model is about to answer
a security question is close to what one would expect. **The narrow conclusion:
6.7B's security lexicon separates the pair when the model is asked.**

### Result — prompt sensitivity is large, and measuring it was not optional

Both prompt styles ran. On 6.7B they disagree outright: pair separation is
**0.750 under `sink` and 0.222 under `e6`** (p = 0.000 both ways) — i.e. under the
second wording 78% of pairs give the *safe* member the higher yes-margin. Same
model, same programs, opposite answer. 1.3B (0.694 vs 0.722) and StarCoder2 (0.917
vs 0.806) agree across the two.

### Result — behaviour breaks where everything else does

Pair separation under the `sink` prompt across conditions:

| | clean | rename | opaque | encode | **flatten** | full cumulative |
|---|---:|---:|---:|---:|---:|---:|
| 1.3B | 0.694 | 0.625 | 0.708 | 0.722 | **0.514** | 0.472 |
| 6.7B | 0.750 | 0.597 | 0.653 | 0.486 | **0.597** | 0.556 |
| SC2-3B | 0.917 | 0.708 | 0.764 | 0.833 | **0.694** | 0.653 |

Flattening takes 1.3B to chance (0.514) and costs the other two 0.15–0.22.

### What it means

Three things. **The instrument is not blind**, so R7's "carried by no word" is a
statement about the models. **The security vocabulary is inverted, not inert**, on
two of three models, at a cell where the same readout demonstrably works.
And this is the **third independent readout** — a fitted probe (R5), an
unsupervised full-vocabulary direction (R7) and the model's own forced-choice
margin — showing the same transformation boundary, the last of the three
involving no probe and no lens at all.

Two cautions. Any behavioural claim drawn from a single prompt is worth less than
it looks. And on 6.7B "not lexicalised" holds for the *unprompted* state only.

---

## R9 — Relevance moves, on text that is character-for-character identical

### Research question

R7 and R8 both read the state through the vocabulary, so both can only find a
distinction that some token or combination of tokens carries. **Does the model
route its answer differently through the two members of a matched pair, in a way
that needs no lexicalisation at all?**

### Hypothesis

Under the LRP rules validated in R6 the relevance decomposition conserves, so
`R_t / s` is a genuine **partition** of the model's own answer across input
positions, and a paired difference is a *redistribution* rather than a change of
scale. If the model resolves the flow question by tracing the chain that feeds the
sink, then the relevance share of that chain should differ systematically between
the unsafe and safe members — at positions whose tokens are identical.

### Method

Stage 130 (deepseek-coder-1.3b). Conservation is measured per (pair, layer) and
came back at median |ρ − 1| = **0.0000** with a worst case of 5e-5 at every layer.
Relevance is summed **by AST role**, recomputed from each program's own source.

**The control is free, and it is the strongest one in the project: only
`sink_arg` differs in tokens between the two members.** Measured under the real
tokenizer, every other role matches at 1.000 and `sink_arg` at 0.611 — exactly
the 44/72 length-matched pairs. So a shift among the other roles is the model
routing its answer differently through **identical text**.

### Result

Paired difference in each role's share, clean held-out, 72 pairs:

| layer | role | median Δ share | pairs shifting the same way | sign-test p |
|---|---|---:|---:|---:|
| 0 | `taint_chain` | −0.0136 | **65/72** | 7e-13 |
| 0 | `trust_chain` | +0.0207 | **63/72** | 4e-11 |
| 3 | `taint_chain` | −0.0083 | 62/72 | 3e-10 |
| 3 | `trust_chain` | +0.0179 | 61/72 | 2e-09 |
| 7 | `trust_chain` | +0.0097 | 55/72 | 8e-06 |
| 19 | `trust_chain` | +0.0028 | 56/72 | 4e-06 |

Read across the two rows: **whichever chain feeds the sink loses relevance share,
and the other gains.** In the unsafe program the sink takes the tainted variable
and the *trusted* chain carries more of the answer; in the safe program it is the
other way round. The effect is strongest at the bottom of the stack, decays with
depth, is gone for `taint_chain` by layer 11, and survives for `trust_chain` to
layer 19.

It survives both available controls: it holds separately in both `role_swap`
strata (which identifier name carries the taint) and both `order_swap` strata
(which chain is written first), at p ≤ 2e-2 in every one of the eight cells at
layers 0 and 3. So it is neither an identifier-name effect nor a
position-in-program effect.

### Two honest qualifications

The magnitude is **small** — a median shift of 1–2% of the answer. And the
pre-declared `above_permutation_control` check, which tests the *mean*, **fails**
(p = 0.39–0.62): relevance deltas are heavy-tailed enough that seven outlier pairs
out of seventy-two flip the mean's sign while the median and the sign stay put.
The statistic that survives is the sign, and its exact null under the same
random-orientation scheme is binomial — which is the test reported above. The
verdict records both: `redistribution_consistent_but_not_in_mean`.

This ran on **deepseek-coder-1.3b only**. It is *not applicable* to StarCoder2 —
the homogenising rules bind to nothing there, so there is no conservation to read,
and stage 130 refuses rather than emitting raw autograd under the name relevance
(R6). It has not been run on 6.7B.

### What it means

The distinction shows up in *routing* before it is representable in output
coordinates: the relevance shift is already present at layers 0–3, while the
output-basis direction of R7 does not appear until ~25% depth. This is the one
result in the file that the R-lens makes possible and a logit lens cannot
produce.

---

## What is actually verbalised: three grades

Read together, R7, R8 and R9 answer this in one sentence: **the answer to a
question is verbalised; the property it is an answer about is not.** Three
distinct grades come out of the data and should not be collapsed.

**1. Verbalised — the yes/no lean, late, and as a ranking.** When the model is
*asked*, the lean toward `" yes"` over `" no"` is readable in vocabulary space
(0.889 / 0.847 / 0.944), tracks the model's own margin (0.708 / 0.806 / 0.917),
and first becomes significant at 38–65% depth. It is verbalised as a **ranking,
not a decision**: accuracy is 0.500 on all three models, because each has a fixed
answer bias whose argmax never moves. *(R8)*

**2. Output-aligned but not lexicalised — the distinction itself.** Projected
onto positively by 72/72 held-out pairs on every model at cosine 0.38 over a floor
of exactly zero, appearing at ~25% depth — *earlier* than the verbalised answer.
Its loadings are flat (0.019–0.027) and spread over thousands of meaningless
tokens. *(R7)*

**3. Not lexicalised — and the security words are *inverted*, not inert.** At the
cell where the taint poles separate the pair at 0.889 and 0.944, the security
lexicon separates it at **0.347 and 0.389** — significantly in the wrong
direction, on the same instrument, at the same position, with p < 0.01. The
readout is demonstrably alive there. *(R8)*

**The depth sequence is the cleanest way to say all of this:**

| relative depth | what is happening |
|---|---|
| 0–13% | relevance **routing** already differs (R9: the chain feeding the sink loses share, 65/72 pairs) while vocabulary space is empty |
| ~25% | a **distributed output-aligned direction** appears and holds to the output (R7) |
| 40–65% | the **answer becomes sayable** (R8) |

The property is routed differently before it is representable in output
coordinates, and representable long before it is sayable. It never becomes a
word.

---

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

Stages 100–108 on deepseek-coder-6.7b, 400 base programs (120 calibration / 280
test), site `use`, layer 8, rank 1 — site and layer chosen on calibration and
recorded before test numbers were read. Six hard gates, each refusing to run
downstream stages. Seven control arms (METHODS §8.4). Outcome metric is
`says_installed`, the **full-vocabulary argmax**, not the logit margin, for the
reason in METHODS §8.6. Cluster bootstrap over bases.

### Result — the gates

| gate | result |
|---|---|
| **H0** generation and independent ground truth | **PASS** — 400/400 bases; all six invariant checks at 1.0000, including the arm crossing |
| **H1** the model returns the bound variable | **PASS** — 1.000 overall, 1.000 in the weakest cell |
| **H2** the binding is decodable at the use anchor | **PASS** — 1.000 against a *measured* surface floor of 0.500, selectivity 0.524 |
| **H3** whole-state interchange flips the answer, per arm | **PASS** — ab +4.781 [+4.683, +4.878], ba +4.799 [+4.694, +4.903], flip rates 0.857 / 0.879; both structural zeros exactly 0.00e+00 |
| **H4** low-rank interchange beats matched controls on the training arm | **PASS** — 189% of the whole-state ceiling; all three control contrasts clear zero; edit moved 0.479 of ‖h‖ |
| **H5** the same subspace transfers to the held-out arm | **PASS** — 100.0% of held-out rows emit the installed answer, 114% of that arm's ceiling; `answer_direction` transfers at an arm ratio of 0.154 against transport's 1.025 |

H1 and H2 both at 1.000 are worth pausing on: with no arithmetic anywhere, 6.7B
resolves these bindings perfectly, and which definition is in scope is perfectly
decodable at the use anchor against a floor pinned to 0.500. That is a cleaner
replication of R1's isolation than R1 itself, on a corpus built for intervention.

> **A bookkeeping note, stated because it matters.** The `gates.yaml` and
> `e13_report.md` on disk still record **H5 as FAIL**, because they were written
> under the superseded `delta_ld` (logit-margin) discriminator and stages 106–107
> have not been re-run since the rule was corrected to `says_installed` on
> 2026-08-13. **The underlying rows are unchanged** — every number in the table
> below is read directly from `interchange_summary.csv` — and the full record of
> the rule change, including the verdicts under both rules, is in
> [ARCHIVE.md §5](ARCHIVE.md). Re-running stages 106–107 would regenerate the
> gate file; nothing else changes.

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

At this site, in this model, the model's own downstream computation **reads** a
rank-1 subspace whose content is *which definition is in scope*. Not the token.
Not the answer. That is the first affirmative causal result in the project, and
the design refutes the two competing accounts rather than merely failing to
support them.

### What is still open

A rank-1 edit outperforming the whole-state patch (100% vs 86%) has a plausible
explanation — the full patch installs the driving component *and* components that
fight it — that is **not** independently demonstrated. And this is one site, one
layer, one model, one construction.

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

**There is an order of operations, and it is legible.** Relevance routing differs
between the two members at 0–13% depth, an output-aligned direction appears at
~25%, and the answer becomes sayable at 40–65%. The property is *routed*
differently before it is *representable* in output coordinates, and representable
long before it is *sayable*.

**And at least one of these representations is causally used.** The DAS
interchange settles for binding what probing structurally cannot: a rank-1,
dose-free edit at the resolution site installs *which definition is in scope*, in
both arms of a factorial where a token account and an answer account each demand
the opposite movement. The model reads that subspace.

**One methodological result about the tools themselves.** Of three
output-basis readouts, the two expensive ones change no conclusion when used as
vocabulary projections — the plain logit lens matches them wherever a result
actually fires. The R-lens pays for itself only in a different role, as a
*conserving attribution* over input positions, and that role produced the
earliest-depth evidence in the depth sequence. A validated instrument that turns
out to be unnecessary for the question at hand is worth reporting as such.

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
  interchange transports the binding at one site, one layer, one model, one
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
- **Not that R9's routing effect is large.** Median 1–2% of the answer, one model,
  and its mean-based permutation control does not fire.

---

# Open items

Ordered by what would most change what this project can claim.

1. **Re-run stages 106–107 to regenerate R10's gate file** under the
   `says_installed` discriminator, so the on-disk verdict matches the reported
   one. No number changes; this is bookkeeping that a reader will check.
2. **Replicate R9 on 6.7B.** The relevance redistribution is one model, and it is
   the only result the R-lens makes possible.
   `scripts/130_sinkflow_relevance.py --model deepseek-coder-6.7b`, ~30–90 min.
   It is not applicable to StarCoder2, so 6.7B is the whole replication.
3. **Explain, or bound, the rank-1 edit beating the whole-state patch** (100% vs
   86% at 60% of the edit norm). The available account is plausible and untested,
   and a reviewer will ask.
4. **A second model and a second site for R10.** The causal result is currently
   one cell: `use`, layer 8, rank 1, 6.7B. 1.3B is cheap now that stage 106 runs
   in minutes.
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
