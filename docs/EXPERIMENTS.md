# Experiments

This file is the catalogue: what every experiment asks, how it is built, why it
is built that way, and what it found. It is organised by the **order the
research actually took**, because each step exists because of what the previous
one could not settle.

Two companions: `docs/RESULTS.md` says what is currently established and what is
not; `docs/ARCHIVE.md` holds the experiments whose claims were withdrawn or
whose designs were abandoned, each with the reason. The registry the pipeline
itself reads is `results/STATUS.yaml`.

---

## Contents

- [§0 The research question, and the flow](#0-the-research-question-and-the-flow)
- [§1 Phase I — Is the relation there?](#1-phase-i--is-the-relation-there) — E2, E3, E4
- [§2 Phase II — What is it made of?](#2-phase-ii--what-is-it-made-of) — E5, E9, **E15**, **E15-C**
- [§3 Phase III — Is it causally used?](#3-phase-iii--is-it-causally-used) — E13

This document maps **only the experiments that carry a finding**. Work that was
retired, parked or superseded — E1, E6, E7, E8, E10-2, E10-3, E11, E12 — lives in
`docs/ARCHIVE.md` with the reason for each. All of it is still runnable.
- [§4 Shared conventions](#4-shared-conventions)
- [§5 Models and replication](#5-models-and-replication)

---

# 0. The research question, and the flow

> **Do code language models build representations of program *semantics* that
> are distinct from lexical and syntactic regularity — and are those
> representations causally used in the model's own computation?**

The question splits into three, and they need different evidence. It is easy to
conflate them, and most of this project's retractions come from doing so.

| | Question | What would settle it | Phase |
|---|---|---|---|
| **Representation** | Is the relation *present* in the hidden states, beyond what the text predicts? | a decoder that beats a floor no surface feature can exceed | I |
| **Robustness** | What is the representation made *of*? Which perturbations destroy it? | change the form while fixing the meaning, and vice versa | II |
| **Causal use** | Does the model's own downstream computation *read* it? | an intervention that changes the relation and nothing else | III |

The flow, and why each step followed the last:

```
  Phase I    E2 binding ─────→ E3 def-use
             THE FLOOR         same floor
                                │
                                │  E4 control dependence: floor is 0.927, not 0.500
                                │  → kept only as the CONTRAST that makes E2 mean something
                                ↓
  Phase II   E5 context: distance is cheap, interference is not
             E9 obfuscation: renaming survives mid-layer, flattening does not
                                │
                                │  E15 the same transformations, on the security
                                │  property: does "untrusted data reaches this
                                │  sink" survive them, and WHICH ONE breaks it?
                                │  → 1.000 clean over TWO chance floors; opaque
                                │    predicates and arithmetic rewriting are free;
                                │    FLATTENING ALONE causes the whole collapse,
                                │    with the interaction inside draw noise — at
                                │    1.3B, 6.7B AND starcoder2-3b.
                                │  E15-C the same states in the model's OWN
                                │  vocabulary: a null, and inverted at 1.3B.
                                │  Decodable is not verbalised.
                                ↓
  Phase III  "decodable" is not "used". Four attempts:
             E7  whole-state patch      → transports the tokens too       [claim retired]
             E10 J-lens readout track   → instrument OK, both uses failed [archived]
             E11 rank-2 coordinate swap → below the site's causal dose    [NO-GO, retracted]
             E12 latent store transfer  → bottlenecked on arithmetic      [parked]
             E13 binding interchange    → H0-H5 PASS: the binding transports
```

Phase III is where the work is, and its history is the project's real
methodological content. Each of the first four attempts failed for a
*different, nameable* reason, and each reason narrowed the design space for the
next; E13 is what was left standing. `docs/ARCHIVE.md` records all four in
detail, along with every criterion this project changed after seeing data.

**One idea carries the whole project**, and it is worth stating before any
experiment. Take two programs that are identical except for a single character,
where that character changes what the program *means*:

```python
x = 3                  x = 3
def f():               def f():
    y = 7                  x = 7          ← one character
    return x               return x
#   → 3                    → 7
```

The token we ask about (`x` in `return x`) is at the same index in both. The
words around it are the same. The distance to everything is the same. Only the
answer changes. A predictor that sees only nearby tokens and distances is
therefore right **exactly half the time — 0.500 by construction, not by
estimate**. Everything above that floor is something the model computed.

Most interpretability testbeds must *estimate* how much a shortcut could
explain and subtract it. Code lets us make the shortcut carry no information at
all. That construction is reused in E2, E3, E11 and E13, and its absence is why
E4 was demoted.

---

# 1. Phase I — Is the relation there?

## E2 — variable binding (the foundation)

**Question.** Does the model represent *which definition* an identifier
occurrence refers to — as opposed to merely which characters it is spelled with?

**Why it matters.** This is the cleanest case where surface form and meaning
come apart. Two occurrences of `data` may be the same variable or two different
ones depending on scope, and nothing local tells you which.

**Method.** A pairwise probe on `[h_i; h_j; h_i−h_j; |h_i−h_j|]` asking "do
these two occurrences bind to the same definition?" Negatives are split into
strata, reported separately, from easy to decisive:

| stratum | what it removes |
|---|---|
| `diff_name` | nothing — the trivial baseline (capped at 3× positives) |
| `distance_matched` | the "nearby ⇒ related" shortcut |
| `same_name_diff_binding` | the "same string ⇒ same variable" shortcut |
| **`context_matched`** | **every fixed-offset surface cue at once** — the pair above |

**The control is the experiment.** On `context_matched` the two programs are
token-identical except the one binding-flipping character; the anchor windows
and the token distance are identical while the label flips; and both programs
share a cross-validation group so neither can be memorised through the other.

**Found** (`context_matched`, the only clean headline):

| | 1.3B | 6.7B |
|---|---:|---:|
| surface baseline (token ids + distance, no model) | 0.500 | 0.500 |
| embedding layer (−1, token identity only) | 0.500 | 0.500 |
| first transformer block | 0.570 | 0.531 |
| layer 3 | 0.961 | 0.914 |
| **peak, middle layers** | **0.984** (L7) | **0.984** (L11–15) |
| last layer | 0.930 | 0.914 |

**Reading.** Three things happen in order. *Nothing is there at the input* —
both floors are exactly 0.500, so the relation cannot be looked up and has to be
computed. *It is built in the first few blocks*, reaching ~0.95 by layer 3.
*It is partly shed near the output*, consistent with the final layers
reorganising toward next-token prediction. The two model scales agree on shape
and differ only where a scaling account predicts.

Stage 20 · `binding_strata_*.png`, `layers_accuracy_*.png`.

## E3 — def-use edges (data flow)

**Question.** Is a directed definition→use edge decodable, and how far does it
reach?

**Method.** Same pairwise setup and the same strata as E2, over reaching-
definition edges. Negatives are **distance-matched**, so "nearby ⇒ related"
cannot win. Accuracy is bucketed by token distance (0–10, 10–50, 50–200, 200+).

**Found.** Same profile as E2 — peak ~0.99 at layers 7–11 over the same 0.500
floor — with honest decay by distance: the hardest bucket (50–200 tokens apart)
still reaches **0.96–0.99**.

**Reading.** The model tracks data flow across real distance, not adjacency.

Stage 20 · `defuse_distance_*.png`.

## E4 — control dependence (demoted; the contrast that matters)

**Question.** Does the model represent whether a statement executes under a
given guard?

**Method.** Pairs of (guard-expression anchor, statement anchor). Negatives are
statements in the *same* program outside the guard, including an
`indent_matched` hard stratum — a statement in a *sibling* guard's body at the
same nesting depth. Ground truth from AST nesting with join points resolved
exactly.

**Found.** Decodable at near-ceiling (AUC 0.999 at 6.7B) — **but the model-free
surface baseline is already 0.927 / AUC 0.990.**

**Reading, and why it is demoted.** A statement's guard is usually its nearest
enclosing `if`, so token windows plus indentation recover most of the relation
without any model. Unlike binding and def-use, no construction pins this floor
to chance. E4 is therefore reported as the **syntactic end of a spectrum whose
semantic end is binding**, and no representational conclusion is drawn from it.

This demotion is load-bearing. It shows the project's criterion for "semantic"
excludes things, which is what makes it a criterion rather than a slogan.

*Open caveat:* probing anchors fall on each span's last token, which here are
integer literals. Re-anchoring on the guard variable and statement target is a
CPU-only re-run and remains open.

Stage 20.

## E5 — context degradation: distance versus interference

**Question.** Does the representation degrade because things get *far apart*, or
because the *problem gets harder*?

**Method.** Insert filler between the tracked definition and its use, sized by
real tokenizer counts (0 → 1000 tokens), varying only what the filler *does*:

| filler | what it adds |
|---|---|
| `comment_prose` | inert English |
| `dead_code` | unreachable statements |
| `lexical_decoy` | similar-looking but irrelevant fresh names |
| `competing_update` | genuinely rebinds other variables |
| `scope_shadow` | reuses the *tracked* names in a nested scope |

**Found** (6.7B binding accuracy at 500 inserted tokens):

| filler | acc | reading |
|---|---:|---|
| `comment_prose` | **0.921** | length is almost free |
| `dead_code` | 0.794 | mild |
| `lexical_decoy` | 0.795 | mild |
| `competing_update` | 0.859 | moderate |
| `scope_shadow` | **0.570** | **severe** |

At 1000 tokens `scope_shadow` drives binding to 0.498 — chance — while every
other filler stays above 0.70.

**Reading.** The representation degrades when the *semantic task* gets harder,
not when the context gets longer. A per-layer detail sharpens it: under
`scope_shadow`, block 0 is the most stable part of the network (flat ~0.75)
while the middle layers — the ones doing the binding work — collapse. The
interference lands on the computation, not on a lookup.

Stage 30 · `context_{task}_*.png`.

## E9 — obfuscation: same meaning, harder surface

**Question.** Is the relation carried by the identifiers, or by something that
survives rewriting them?

**Method.** A cumulative, Tigress-inspired ladder implemented natively for
Python. **Every variant is executed and verified observationally equivalent to
its base**, and all levels of a base are kept or dropped together so level
curves compare identical program sets.

| level | transformation |
|---:|---|
| 0 | normalize (`ast.unparse`, a formatting baseline) |
| 1 | + consistent renaming of every local |
| 2 | + dead branches under provably false opaque predicates |
| 3 | + mixed boolean-arithmetic rewriting (`a+b → (a^b)+((a&b)<<1)`) |
| 4 | + control-flow flattening into a dispatch loop |

**Found**, now on all three models (best layer per task, cumulative levels):

| | rename | opaque | encode | flatten |
|---|---:|---:|---:|---:|
| binding — 1.3B / 6.7B / starcoder2-3b | 0.783 / 0.883 / 0.708 | 0.801 / 0.862 / 0.743 | 0.834 / 0.857 / 0.790 | **0.555 / 0.615 / 0.527** |
| def–use — 1.3B / 6.7B / starcoder2-3b | 0.819 / 0.864 / 0.689 | 0.799 / 0.846 / 0.731 | 0.800 / 0.833 / 0.747 | **0.461 / 0.545 / 0.402** |

**This is E15's companion control, and it is now complete.** The same
transformations break binding and def–use the same way they break the security
readout — and the security readout is *at least as robust* as either (0.882–0.986
under renaming, 0.660–0.688 under flattening). So the boundary is a general limit
of frozen linear readouts of program relations, **not** a security-specific
fragility. That distinction was open for a year; it is now closed on evidence.

**Reading — the layer breakdown is the finding.** Renaming pushes the
*embedding and block-0* probes below chance (0.29–0.33): those layers keyed on
identifier strings and renaming actively misleads them. Middle layers 7–15 hold
at 0.85–0.90. Opaque predicates and rewritten arithmetic barely register,
because they do not change which definition reaches which use. **Control-flow
flattening is the real boundary**: the frozen probes encode binding relative to
the surrounding control structure, and dissolving that scaffold breaks transfer.

Taken with E5, the two ladders describe one failure surface: robust to *how far
apart* things are and to *what they are called*, fragile when the scope or
control structure it is a representation *of* becomes harder.

Stage 31 · `obfuscation_levels_*.png`.
## The safety-relevant track: what is actually being audited, and why these metrics

Three sections follow — E15 (the benchmark and the robustness result), E15-C (the
vocabulary contrast) — and they are the only part of this project framed around a
*security* property rather than a graph-theoretic relation. This preamble says
what the safety framing does and does not commit to, because that is the easiest
thing in the repository to overstate.

### The one bit under audit

> Is the value passed to this security-sensitive, **code-bearing** argument
> derived from untrusted input?

That is the whole property. It is **not** vulnerability detection: real
exploitability depends on path feasibility, the environment, sink semantics, and
mitigations that live outside this benchmark. It is **not** malware
classification. What it *is* is the one bit a taint analysis exists to produce,
posed so that a model's internal states can be asked for it directly.

Three sink families, each with a real CWE behind it: command execution
(`os.system`, `subprocess.call(..., shell=True)`), SQL execution
(`cursor.execute`), dynamic execution (`eval`, `exec`).

### The threat model, stated as a threat model

| | |
|---|---|
| **Defender** | an auditor using a fixed code model and a frozen linear readout |
| **Adversary** | controls the program's *surface form*; applies semantics-preserving transformations; does **not** change the weights or the security label |
| **Protected property** | whether the value at a code-bearing sensitive argument derives from untrusted input |
| **Attack success** | reducing the readout's ability to separate a matched safe/unsafe pair — **especially by producing false negatives** |
| **Out of scope** | executable malware, reflection, dynamic loading, heap aliasing, concurrency, and any adaptive white-box attack optimised against this model or this probe |

The adversary here is **the surface form, not the model**. That is what makes the
matched-pair construction possible and what makes a frozen readout the right
instrument: if the probe were refitted on each transformation, it could learn a
*new* shortcut for each one and conceal exactly the representational failure the
experiment is looking for.

### Why "safe" is a trusted literal and not a sanitizer

The generic sanitizer list used elsewhere in this repository (`html.escape`,
`shlex.quote`) is deliberately **not** reused. `html.escape` before `exec` and
`shlex.quote` before `eval` are not mitigations, so a benchmark whose "safe"
class was built from them would be labelling genuinely vulnerable programs safe —
and every accuracy number downstream would be measuring the wrong thing. The safe
member instead passes an **independently trusted literal**: a constant that never
touches the source, through the same propagation, to the same sink.

### No dangerous API is ever executed

Ground truth is recomputed from every program — including every transformed
variant — by two independent readings that must agree with each other *and* with
the intended label. One of them is **instrumented execution**: the module runs
with `__builtins__ = {}` and every sensitive API (`os.system`, `subprocess.*`,
`cursor.execute`, `eval`, `exec`) replaced by a recorder, with a provenance-carrying
`str` subclass standing in for untrusted input. Nothing dangerous is reachable
even in principle, at any transformation level, even if a generated program were
wrong.

### The metrics are safety metrics, and pooled accuracy is not one

This is the methodological commitment the track exists to demonstrate. Accuracy
of 0.5 has at least two very different causes, and the number alone cannot tell
them apart:

* **the information is gone** — the readout gives both members of a pair the
  *same* label, because the position no longer distinguishes them.
  `pairs_same_label` → 1, `frac_predicted_unsafe` collapses toward one class;
* **the information is there and no longer means taint** — the readout still
  splits the pair, but the direction is now arbitrary. `pairs_same_label` stays
  low while accuracy falls to chance.

So every reported cell carries, beside accuracy and its cluster-bootstrap
interval:

| metric | why an auditor needs it |
|---|---|
| `acc_unsafe` / `acc_safe` | a symmetric 0.07 loss and a one-sided 0.24 loss are different failures with the same headline |
| **`false_negative_rate`** | a vulnerable program called safe — the failure direction that matters, named rather than left as `1 − acc_unsafe` |
| `false_positive_rate` | the cost side: an auditor drowning in false alarms stops reading them |
| `frac_predicted_unsafe` | detects collapse onto a class prior |
| **`pairs_same_label`** | the two members differ *only* at the sink argument, so this rising is the sharpest possible evidence the position stopped carrying the distinction |

And the results below show why this is not pedantry: under the full
transformation composition all three models land within 0.08 of each other while
**biasing in opposite directions**, and StarCoder2's entire renaming loss is
false negatives. A pooled number would have reported all of that as "mostly
fine".

---

## E15 — source→sink under obfuscation (the security audit track)

**Question.** E9 asks what the transformations do to *binding* and *def-use*. E15
asks the question they were built for: **is the value that reaches a code-bearing,
security-sensitive argument derived from untrusted input**, does a readout of that
fact, frozen on clean programs, survive obfuscation — and **which transformation
breaks it, on its own**?

**Why it exists.** Binding and def-use are the mechanism; "untrusted data reaches
`os.system`" is the property an auditor actually wants, and the one an adversary
has an interest in hiding.

**Method.** 3 sink families × 4 flow structures × 20 base seeds × 2 labels =
**480 clean programs**. Each base is a matched unsafe/safe pair holding the same
source, propagation, trusted alternative and sink, differing **only at the sink
argument** — checked character-exactly. 14 seeds per cell train the readout, 6 are
held out, and **only held-out programs are transformed**. The readout is fitted
once on clean training programs and frozen.

**Ten conditions — four atomic, four cumulative.** The same four rewrites are
applied one at a time *and* composed:

| kind | conditions |
|---|---|
| baseline | `normalize` (ast round-trip only) |
| **atomic** | `rename_only`, `opaque_only`, `encode_only`, `flatten_only` |
| **cumulative** | `rename_cumulative` → `rename_opaque` → `rename_opaque_encode` → `rename_opaque_encode_flatten` |

No new obfuscation algorithm and no arbitrary combinations — the atomic arms make
the existing rewrites individually addressable. 1296 held-out variants per model.
Each variant's transformations are **read off its own AST** and must equal exactly
what its condition declares; both pair members share one draw; the transformed
pair is re-checked for sink-argument confinement — all inside gate S0.

**Four control arms, all frozen and all transferred**: `local_surface` (±3 token
ids at the anchor), `whole_program_lexical` (token n-grams + char 3–5-grams over
the **entire** program), `embedding` (layer −1) and `hidden_state`. The two lexical
arms answer different questions — the local one bounds "the identifier gives it
away", the whole-program one bounds "the generator left a shortcut in the text".

**Found** (site `sink_arg`, 144 held-out programs / 72 bases per row, cluster
bootstrap, read at the layer nearest 48% depth: 1.3B L11, 6.7B L15, starcoder2-3b
L15). All six gates pass on all three models with no overrides.

| condition | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| local surface floor, every condition | 0.431–0.514 | 0.431–0.514 | 0.431–0.521 |
| **whole-program lexical floor**, every condition | 0.465–0.535 | 0.465–0.535 | 0.465–0.535 |
| embedding layer (−1), clean | 0.482 | 0.482 | 0.482 |
| **clean held-out** | **1.000** | **1.000** | **1.000** |
| `rename_only` | 0.938 | 0.986 | 0.882 |
| `opaque_only` | **1.000** | **1.000** | **1.000** |
| `encode_only` | **1.000** | **1.000** | **1.000** |
| **`flatten_only`** | **0.688** | **0.667** | **0.660** |
| full ladder | 0.729 | 0.653 | 0.674 |

**Reading — the atomic arms are the finding.** Chance at the input against *two*
measured floors, ceiling by half depth, held to the output. Then: opaque
predicates and arithmetic rewriting are **exactly free**; renaming costs 0.01–0.12;
and **flattening alone costs 0.31–0.34**, within 0.03 of what the entire
four-transformation composition costs. The interaction (cumulative − atomic) is
+0.042 / −0.014 / +0.014 against a *measured* draw-noise floor of 0.021 / 0.035 /
0.028 — indistinguishable from noise. **Composition is not the threat; one
transformation is.** The model is reading control structure, and when the
structure is dissolved the readout goes with it.

**The flattening number is worse than it looks.** At `flatten_only` about half the
matched pairs collapse to one label (0.514 / 0.556 / 0.458), and under the full
ladder the class biases run in *opposite* directions across models — 6.7B toward
"unsafe" (0.861/0.444), starcoder2-3b toward "safe" (0.569/0.778). A constant
predictor of either class scores exactly 0.500 on this balanced set.

**The dangerous errors arrive first, and before any structural change.** Under
renaming alone starcoder2-3b is 0.882 pooled but **0.764 on unsafe against 1.000
on safe** — the whole loss is false negatives. By structure the **assignment chain
is the fragile one under renaming** in all three models (0.778 / 0.972 / 0.639)
while `branch_merge` is untouched at 1.000; by sink family nothing reproduces.

**The boundary is general.** The companion E9 run is complete on all three models:
the same transformations take binding from 1.000 to 0.708–0.883 (rename) and
0.527–0.615 (flatten), and def–use to 0.689–0.864 and 0.402–0.545. The security
readout is **at least as robust** as the primitives it rests on. So the supported
claim is "structural obfuscation breaks frozen linear readouts of program
relations, security ones included" — *not* "security representations are
specifically fragile".

**Limitations, stated up front.** The floor is pinned only against *declared*
feature families: both measured floors sit at chance, but a reader that ran the
taint analysis itself would score 1.0. Eight arms, not the full 15-combination
lattice. "Flattening breaks the readout" is a claim about a **frozen linear
readout at one position**, not proof the model lost the information. The embedding
control is one measurement, not three. Nothing causal is claimed.

## E15-C — is the difference in the model's own vocabulary? (observational)

**Question.** The probe says what a *fitted* direction can recover. It cannot say
whether the model's **output-aligned** coordinates carry the distinction, because
a probe chooses its own basis. So: after mapping the sink-site state into the
model's vocabulary space, which vocabulary directions separate an unsafe program
from its matched safe counterfactual?

**Method.** Three readouts on the same states — the logit lens, E10/E11's J-lens
and E14's R-lens — with **R-lens declared primary in code before any result was
produced**. Orientation fixed once: `delta = score_unsafe − score_safe`. Discovery
is two-phase and training-only (a full-vocabulary logit-lens ranking on clean
*training* pairs gives a 196-token candidate pool; each lens then ranks the pool
by its own training delta) and is **frozen to disk before stage 126 reads it
back**. A small security lexicon is fixed in advance and validated per model, with
every omission recorded and nothing substituted.

**Controls.** Permutation of the orientation within bases; mismatched pairs from
different bases; the embedding layer, which at `sink_arg` *is* the token-identity
contrast; the `last_token` site, where both members carry the same token;
identifier-role strata; and random and Gram-matched lenses.

**Found — a null, in all three models.**

| clean held-out, R-lens | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| concept token surviving the tokenizer | `" vulnerable"` | `" vulnerable"` | `" unsafe"` |
| held-out sign consistency | **0.153** | 0.403 | 0.694 |
| permutation p | 0.000 | 0.004 | 0.008 |
| verdict | **inverted** | stable non-security | stable non-security |

The security lexicon carries the contrast in no model, and the direction is not
even consistent: 1.3B is significantly **inverted** — 85% of pairs put *less*
unsafe-pole mass on the unsafe member — while starcoder2-3b leans the hypothesised
way without reaching the pre-declared 0.70 threshold.

**The distribution confound is ruled out.** A systematic difference in the shape
of a member's candidate distribution would shift a z-scored contrast in a fixed
direction with no concept involved. Per pair, it does not: at the reported cells
the contrast correlates with the paired entropy difference at r = −0.29 / +0.16 /
+0.14 and with the score-norm difference at −0.04 / −0.10 / +0.10, with no cell
anywhere above |r| = 0.39 and a mean paired entropy difference of ≈ 0. At most 8%
of the variance is distributional, so **the inverted 1.3B sign is real** — an
unexplained phenomenon rather than an artifact.

**Why this is a real null and not a failed measurement.** (i) The three lenses
**agree** — pairwise cosine of their mean vocabulary-difference vectors is
0.75–0.97 — so the null is not an artifact of the primary-lens choice. (ii) It is
not token identity: the embedding-layer contrast is null (p = 0.71–0.81) and 75%
of pairs share the *same* anchor token at `sink_arg` anyway. (iii) Something does
replicate — frozen training-discovered tokens reappear in the held-out top-k at
0.875 / 0.750 / 0.875 against 0.000–0.031 for random control tokens — but the
tokens are semantically arbitrary (`" ?"`, `"?."`, `"??"`; `" liber"`, `"clean"`,
`"tbl"`; `"OrNull"`, `"displayMode"`, `"fuchsia"`). (iv) Under flattening the
vocabulary contrast degrades alongside the probe (sign consistency 0.389 / 0.472 /
0.583), so **both trained and output-aligned auditability are lost together**.

**What this licenses:** *linear decodability and expression in the model's own
output vocabulary are different properties, and E15 exhibits the first without the
second.* It does not license any sentence containing "the model represents
unsafe".

**Diagnostics, which warn but never block.** R-lens relevance conservation is
1.0001 (1.3B) and 0.9993 (6.7B) — essentially exact — but **0.154 on
starcoder2-3b**, where the LRP rules never installed at all (LayerNorm plus a
non-gated MLP match neither homogenising rule), so that model's `rlens` artifact
is arithmetically a J-lens. Final-layer rank agreement runs
0.18–0.47. Next-token recovery is unmeasurable (a 196-token candidate vocabulary
rarely coincides with the true next token). The experiment is mechanically valid
throughout; the report separates *mechanically invalid* from *mechanically valid
with weak lens fidelity*.

**Not causal.** No intervention of any kind. E13's interchange is the causal
instrument, and it covers binding, not this.

Stages 120–127 · design `docs/design/E15_SINKFLOW_PLAN.md` (§8 results, §11 A,
§12 B, §13 C, §14 commands).

---

## The J-lens / R-lens track, end to end

The lens work is spread over four experiments and it is worth reading as one
line of argument. Methodology is `docs/METHODS.md` §10–§10b; this is what each
step asked and what came back.

| | question | verdict |
|---|---|---|
| **E10-0** J-lens validation (stage 60) | is the Jacobian-lens machinery correct on code models? | **yes** — at the last decoder layer `J` is the identity, so the J-lens must equal the logit lens; measured cosine 1.0000, a closed-form check of the whole gradient path. Next-token recovery beats a norm-matched floor. |
| **E10-2/-3** J-lens on taint / control-dependence | does the "verbalizable workspace" framing explain E6's scale split? | **archived.** The behavioural signal it rested on did not survive its own controls (`docs/ARCHIVE.md`). |
| **E11** J-space binding routing | is the *value* causally reused through J-space coordinates? | **NO-GO**, reported and not claimed. |
| **E14** R-lens gate R (stage 110) | is a more faithful backward pass available, and how would we know? | **yes, and measurably** — on Llama-family architectures. Gate R passes on both deepseek models: `rho` within 1e-4 at every layer, LRP beating raw autograd at 7/7 and 9/9 testable layers. The ablation says the **gated-MLP rule** carries it (4.46 vs 0.99 for the norm rule), falsifying the plan's prediction. **Not applicable to starcoder2-3b**: LayerNorm plus a non-gated MLP means the rules never install. |
| **E15-C** vocabulary contrast (stages 125–127) | is a *security* distinction expressed in the model's own output vocabulary? | **a null, in all three models** — and significantly *inverted* in 1.3B. |

### What the R-lens fixed, where it fails, and which rule does the work

The J-lens is an averaged first-order readout whose backward pass runs through
modules that are not degree-1 homogeneous. Relevance conservation measures the
damage directly: `rho = 1` exactly when the tail above a layer is homogeneous.
Under raw autograd `rho` wanders and **inverts sign** with depth. The LRP rules
make the traversed tail homogeneous and are **value-preserving** — they change no
activation, only the backward graph, which gate R0 / J0 verifies against the
ordinary forward logits.

**Gate R, both deepseek models, every required check passing:**

| | 1.3B (float32) | 6.7B (float16) |
|---|---|---|
| R0 forward invariance | 1.62e-06 relative (tol 1e-04) | 1.21e-03 relative (tol 1e-02) |
| R1 last layer = logit lens | cosine 1.0000 | cosine 1.0000 |
| R2 LRP beats autograd | **7/7** layers | **9/9** layers |
| R2 conservation, median &#124;ρ−1&#124; | **0.0000** | **0.0001** |

**The rule ablation now replicates.** §2.1 of the plan predicted the LN-rule
would dominate; a 1.3B fp16 run already recorded that as half wrong. A second
model and a second dtype settle it:

| rule removed | 1.3B | 6.7B |
|---|---:|---:|
| **`no_half`** (gated-MLP split) | **4.4203** | **4.4628** |
| `no_ln` (RMSNorm → diagonal) | 0.9806 | 0.9885 |
| `no_identity` (SiLU → elementwise) | 0.2265 | 0.3941 |
| `no_attn` (attention, unmodified by design) | 0.5128 | 0.3044 |

The gated-MLP split dominates by ~4.5× and the ordering is near-identical across
both. Attention's cost — the one path the design deliberately leaves alone — is
0.30–0.51, the bounded answer to "what does the unmodified softmax cost". The
August fp16 anomaly where the identity-rule appeared to *hurt* conservation does
not survive float32: `all` sits at 0.0000 against 0.2265 for `no_identity`, so it
was fp16 noise.

**Where it fails: StarCoder2, and not for a numerical reason.** Gate R cannot
complete there. StarCoder2 uses LayerNorm (deliberately unmatched: it subtracts
the mean, so the algebra differs) and a non-gated MLP, so both homogenising rules
bind to **nothing**; only attention hooks register, which satisfies `lrp_rules`'
own strict check, and stage 110 then raises when its `no_attn` arm removes the
only rule that bound. The tell is in the one file it produced: a forward delta of
**exactly 0.0**. Value-preserving rules still perturb float arithmetic; rules that
were never installed do not.

**Consequence for E15-C**: the starcoder2-3b artifact labelled `rlens` is
arithmetically a J-lens, and its 0.154 conservation is simply raw autograd. J0 now
refuses this (`rlens_rules_bound`). The null is unaffected — it rests on logit and
J-lens results there, and on genuine R-lenses in both deepseek models — but
"three lenses agree" is, for that model, two lenses measured three ways.

### What the lens contrast found, and why the null is trustworthy

E15-C is the first time a lens is used on a **matched pair** rather than a single
state, and the null it returned is only worth anything because the instrument was
pinned down first:

* **The primary lens was declared in code before any result** (`PRIMARY_LENS =
  "rlens"`), because the target includes early and middle layers — exactly where
  E14 showed the J-lens backward is least faithful. Choosing afterwards would
  have made every number a selection artifact.
* **All three lenses agree.** Pairwise cosine of their mean vocabulary-difference
  vectors is 0.90 / 0.96 / 0.97 (1.3B), 0.91 / 0.96 / 0.97 (6.7B), 0.75 / 0.96 /
  0.77 (starcoder2-3b). A null on which the logit lens, the J-lens and the R-lens
  concur is a statement about the models, not about the readout.
* **It is not token identity.** At the embedding layer the contrast is null
  (p = 0.71–0.81), and 75% of pairs share the *same* anchor token at `sink_arg`
  anyway — the sink-argument span's last token is frequently identical between
  the two members.
* **Something replicates; it just is not security.** Frozen
  training-discovered tokens reappear in the held-out top-k at 0.875 / 0.750 /
  0.875 against 0.000–0.031 for random control tokens, with per-token sign
  consistency up to 0.99. The tokens are `" ?"`, `"?."`, `"??"` (1.3B),
  `" liber"`, `"clean"`, `"tbl"` (6.7B), `"OrNull"`, `"displayMode"`,
  `"fuchsia"` (starcoder2-3b) — reliable, and semantically arbitrary.
* **Both readouts fail together under flattening.** The vocabulary contrast
  degrades alongside the probe (sign consistency 0.389 / 0.472 / 0.583), so this
  is the design's "loss of both trained and output-aligned auditability" outcome
  rather than a dissociation between them.

**The claim this licenses**, and the reason the null is a contribution rather
than an absence:

> Linear decodability and expression in a model's own output vocabulary are
> **different properties**. E15 exhibits the first at ceiling and the second not
> at all. A probe finding a direction does not mean the model is disposed to say
> the corresponding word.

**What it does not license.** Any sentence containing "the model represents
unsafe". Nor the converse over-reading: a null here does not prove the
information is absent — only that it is not expressed in the candidate
output-aligned coordinates this design can search, and that search is itself
bounded (the J/R candidate pool is logit-lens-selected).

**Nothing here is causal.** E15-C performs no intervention of any kind — no
J-space coordinate edit, no interchange, no swap. E13's interchange is the causal
instrument in this project, and it covers *binding*, not source-to-sink flow.

### What the null actually looks like by depth — and why it is not noise

The headline reads one layer. The full layer sweep (`vocab_summary.csv`) shows the
contrast is **systematic and depth-structured**, which is a stronger statement
than "no effect":

| R-lens sign consistency, `sink_arg`, clean held-out | embedding | ~10% | ~25% | ~35–48% | ~65% | last |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-coder-1.3b | 0.12 (ns) | 0.53 | 0.36 | **0.15** | 0.14 | 0.25 |
| deepseek-coder-6.7b | 0.11 (ns) | 0.54 | 0.53 | **0.40** | 0.21 | 0.26 |
| starcoder2-3b | 0.12 (ns) | 0.56 | 0.58 | **0.62** | 0.39 | 0.50 |

Read down the columns: both deepseek models start at chance and drift
*monotonically below* 0.5 with depth, reaching significance from roughly a third
of the way in and staying there. StarCoder2 does the opposite — it rises above 0.5
through mid-depth, peaks at 0.69, and falls back. The embedding row is
non-significant everywhere, which is the token-identity control passing.

So there **is** a reliable, depth-organised difference between the two members in
output-aligned coordinates. What there is not is a difference whose *sign* matches
the security hypothesis, or even agrees between model families. That is why the
verdict is a null on the security question and not a null on "anything is
happening".

### Roadmap — how to turn this into a significant finding

Ordered by cost, and each step says what result would justify the next.

**Tier 1 — re-analysis of artifacts already on disk. CPU.**

1. **Depth sweep as a first-class result** — ✅ *built and run* (stage 127 now
   writes `results/figures/e15c_depth_{model}.png`).
2. **Calibrate against the random lens rather than against zero** — ✅ *built and
   run* (stage 127 writes `vocab/vocab_specificity.csv`). **This immediately
   qualified the null**, and it is the most important thing Tier 1 produced:

   | primary lens at the reported cell | sign | displacement | best control | **specificity** |
   |---|---:|---:|---:|---:|
   | deepseek-coder-1.3b L11 | 0.153 | 0.347 | 0.167 | **2.08** |
   | deepseek-coder-6.7b L15 | 0.403 | 0.097 | 0.111 | **0.87** |
   | starcoder2-3b L15 | 0.694 | 0.194 | 0.139 | **1.40** |

   The random and Gram-matched lenses **follow the same depth trajectory** as the
   real ones (see the figure). So the permutation null was detecting *that the two
   states differ at all* — which any direction picks up — and the real lenses beat
   a random direction by a factor of only 0.87–2.08. **On 6.7B the effect is not
   specific to the lens at all.** Any future positive result in this track has to
   clear this bar, not the permutation bar.
3. **Test the confound for the inversion** — ✅ *built and run; it came back
   negative*. Stage 126 records each member's candidate-distribution entropy and
   score norm; stage 127 correlates them with the contrast (report table 12).
   |r| ≤ 0.39 in every cell across all three models, ≤ 0.29 at the reported cells,
   with a mean paired entropy difference of ≈ 0. **The contrast is not a
   distribution artifact**, so the inverted 1.3B sign is a real and currently
   unexplained property. **Tier 1 is complete.**

**Tier 2 — ✅ *built and run as E15-D, stages 128–131*, except the positive
control, which is built and has **not** run. Design and pre-declared thresholds:
`docs/design/E15D_LENS_FOLLOWUPS_PLAN.md`; results in `docs/RESULTS.md`.

> **The headline it produced.** Removing the candidate pool turned E15-C's null
> into a positive finding: a direction defined by the safe/unsafe label,
> estimated on the training split, is projected onto by **72 of 72 held-out
> pairs on every model** (cosine 0.38, token-identity floor exactly zero). It
> appears a quarter of the way up the stack, holds to the output, and collapses
> under flattening alone — independently replicating E15's headline with nothing
> fitted. Its top-loading tokens are meaningless fragments. **Output-aligned,
> distributed, not lexicalised.**

4. **A positive control — the single highest-value next step.** ⚠️ *built, and
   still the single highest-value next step, because it has **not been run**.*
   Stage 129, gate J3, unrecorded on all three models. The design could not distinguish "the models do not
   verbalise this" from "this machinery could not detect verbalisation if it were
   there", and no negative control can: they establish that a positive result is
   not an artifact, and are silent about a null. Stage 129 runs the identical
   readout on the E6/E7 forced-choice taint property, whose answer is a single
   token. **Same** function (`sinkflow_vocab.pair_contrast`), **same** z-score
   convention, **same** orientation, and **one** candidate basis carrying both the
   taint poles and the E15-C security lexicon — J3 refuses the run if the two
   bases ever differ, so "the identical pipeline" is checkable rather than
   asserted. Two prompt styles run (`e6` verbatim, and one naming the sink), so
   prompt sensitivity is measured rather than assumed.
5. **Anchor the lens to behaviour.** ✅ *built: the same stage.* Each model's own
   forced-choice margin is recorded per program, and `taint_lens_tracks_model` is
   the fraction of pairs where the lens's paired margin has the same sign as the
   model's. The behavioural statistic the verdict uses is `pair_separation`, not
   raw accuracy: a model that answers "no" to everything scores 0.5 accuracy for
   free, while pair separation has a chance level of 0.5 that no answer bias can
   move.
6. **De-bias the candidate pool.** ✅ *superseded by stage 128, which did better
   than de-biasing it — it removed it, and that is what turned the null into a
   result.* Instead of choosing a better 196 tokens,
   V1 forms each pair's difference over the **whole vocabulary** and asks whether
   those differences agree. A null there cannot be blamed on a pool, because there
   is no pool. Its statistic is *concentration* (`sv1_share`), not the mean —
   which is the distinction E15-C could not make, since a large mean is
   compatible with every pair pointing somewhere different.

**Tier 2b — a defect found in E15-C's own controls while building the above, and
fixed.** ✅ *`sinkflow_vocab.same_label_pairs`, run by stage 126.*
`mismatched_pairs` redraws the **safe** partner from the same safe pool, so the
label difference survives it intact: the arm averages over the very set the main
arm averages over, its expected mean is the main arm's **exactly**, and on the
canonical runs the two agree to four decimal places on all three models. Its
docstring claimed it separated "tracks the safe/unsafe difference" from "tracks
any difference between two programs" — it cannot, and could never have. The
`above_mismatched_pair_control` check passed by margins of 0.014, 0.014 and 0.056
against a comparison with no noise band, and on deepseek-coder-6.7b the control
is *more* sign-consistent than the main arm (0.417 vs 0.403). The replacement arm
takes **both** members from one pole: everything a matched pair differs in is
still there, the label difference is gone, so the expected contrast is zero and
the expected sign consistency 0.5. That is the arm a label claim has to clear.
This overturns no E15-C result; it replaces an uninformative check with an
informative one.

**Tier 2c — a readout that needs no lexicalisation at all.** ✅ *built and run on
deepseek-coder-1.3b: stage 130, gate J4. Not applicable to starcoder2-3b; not yet
run on 6.7B.* It found a consistent shift: whichever data-flow chain feeds the
sink loses relevance share and the other gains, in 63–65 of 72 pairs at layers
0–3 (sign-test p ≤ 4e-11), on spans that are token-identical between the two
members, surviving both role- and order-swap strata. Small — a median 1–2% of the
answer — and the mean-based permutation null does not fire because the deltas are
heavy-tailed. E15-C and V1 both read the state through the vocabulary, so both can
only find a distinction that some token or combination of tokens carries. Under
the LRP rules `Σ_t R_t = s` (E14 gate R: |ρ−1| within 1e-4 at every layer on both
DeepSeeks), so `R_t/s` is a **partition** of the model's own answer and a paired
difference is a genuine redistribution rather than a change of scale. Aggregated
by AST role, recomputed from each variant's own source. The control comes free:
**only `sink_arg` differs in tokens between the two members** — enforced at
generation time for every condition, and verified across all 1440 held-out
programs to give identical per-role token counts within every pair, including
under full cumulative obfuscation. A shift among the token-identical roles
therefore has no surface account. Stage 130 **refuses on StarCoder2** and records
J4 as *not applicable*: LayerNorm plus a non-gated MLP means both homogenising
rules bind to nothing, so there is no conservation to read.

**Tier 3 — instrument work, and the most publishable single result here.**

7. **Diagnose StarCoder2's conservation of 0.154** — ✅ *done, and it was not what
   the number suggested*. Stage 110 on starcoder2-3b **raises**: the rules never
   install there (LayerNorm + non-gated MLP), so 0.154 was raw autograd and the
   `rlens` artifact is arithmetically a J-lens. J0 now refuses that case. The same
   stage on **deepseek-coder-6.7b passes gate R outright** and replicates the
   ablation ordering. No further diagnosis needed.
8. **~~If the gap is attention-dominated, extend the rules to it.~~** Answered and
   dropped: attention's cost is 0.30–0.51 in both models, while the gated-MLP rule
   carries 4.46. AttnLRP is not where the leverage is. **The replacement item is
   architecture generality**: extend `norm_eps_attr` to LayerNorm and
   `is_gated_mlp` to non-gated MLPs, which is what would make the R-lens usable on
   StarCoder2 at all. The LayerNorm half is the harder one — the mean-subtraction
   term is exactly what the current algebra assumes away.
9. **Relax "vocabulary token" to "output-aligned direction".** The sharpest
   limitation of E15-C is that it can only find a concept if some *single token*
   carries it. Fit a probe constrained to the row space of `W_U` — output-aligned
   by construction, but not required to be one token — and ask whether it
   separates the pairs and replicates held out. That closes the gap between the
   probe result (1.000) and the lens result (null), and it would say whether the
   distinction is output-aligned but *distributed*, which is the one hypothesis
   the current design cannot test.

**What would count as a significant finding at the end of this.** Either: the
positive control fires and the security contrast does not, which turns the null
into a claim about what code models verbalise; or the full-vocabulary direction
concentrates above its same-label null, which turns it into a claim that the
property *is* output-aligned but distributed across tokens rather than
lexicalised; or a token-identical AST role's relevance share shifts consistently,
which is a claim about routing that needs no vocabulary at all. All three are
now built and none has been run at canonical scale, so the current state supports
none of them, and says so.

**Where that leaves it.** Two of the three fired. The full-vocabulary direction
is the strongest result in the lens track and it is a *positive* one, resting on
its own controls. The relevance routing is real, small and single-model. What is
still open is what E15-C's **null** means, and only the positive control settles
that: if the models answer the forced choice and the identical readout misses it
(`machinery_blind`), E15-C's null is about the *method*, every number in that
track keeps its caveat, and no claim about what code models represent survives
it. Stage 131 will print exactly that sentence if that is what the data says.

**Tier 3 items 8 and 9 are now differently urgent.** Item 9 — the
`W_U`-constrained probe — asked whether the distinction is "output-aligned but
distributed". **Stage 128 answered that directly and affirmatively**, without
fitting anything, so item 9 is superseded. Item 8 (architecture generality of the
R-lens) is unchanged and is what would make stage 130 runnable on StarCoder2.

---

# 3. Phase III — Is it causally used?

Everything above is correlational. A representation can be a faithful shadow of
a computation happening somewhere else, and probing cannot tell the difference.
Phase III needs an intervention.

The requirement is harder than it sounds. A useful intervention must change
**the relation and nothing else** — which means acting at a position where the
programs are token-identical, editing a nameable part of the state rather than
replacing it, and doing so at a magnitude the site can actually register. Three
designs failed to have all three at once before one succeeded.

## Three attempts that did not survive their own controls

Before E13, three intervention designs were built, run and retired. They are not
in this document any more because none of them carries a claim — but the *reasons*
they failed are the methodological content that produced E13, and they are written
up in full in `docs/ARCHIVE.md`:

| attempt | why it failed | what E13 took from it |
|---|---|---|
| **E7** whole-state activation patching | the patched state transports the *input difference* as well as any semantic state; the design cannot separate them | intervene where the two programs are token-identical |
| **E10-2 / E10-3** J-lens taint and control-dependence | rested on a behavioural signal that did not survive its own controls | the lens is an instrument, not a result — keep E10-0's validation, drop the uses |
| **E11** J-space rank-2 coordinate swap | the edit was below the site's causal dose; a control (`probe_basis`) was silently skipped rather than refused | hard gates that refuse to run, and a dose that is measured rather than assumed |
| **E12** latent store transitions | parked at its behavioural gate (0.418) — the model could not solve the task the instrument needed it to solve | check the model can do the task *before* building an instrument on top of it |

Every one is still runnable and every CSV is preserved. What changed is the
claim, not the data.

## E13 — binding interchange, falsified by the value assignment (current)

**Question.** Does a low-rank, magnitude-free interchange at the site where a
binding is resolved transport *which definition is in scope* — rather than a
token, or an answer direction?

This is the question `paper/main.tex` §Discussion currently declares open:
*"Where the resolution itself happens, the causal question remains open."*

**The design.** Four programs per base — binding × value assignment — all
token-identical except one character:

```python
# ARM ab: (outer, inner) = (a, b)        # ARM ba: (outer, inner) = (b, a)
x = a                                    x = b
def f():                                 def f():
    y = b     # target program: `x = b`      y = a     # target: `x = a`
    return x                                 return x
#   → a  (outer)   /  → b  (inner)       #   → b  (outer)   /  → a  (inner)
```

Install the *target* run's state into the *source* run at the marked use. In arm
`ab` the answer must move **a → b**; in arm `ba` the same intervention must move
it **b → a**. Fit the alignment on `ab`; read the claim on `ba`.

**Why this identifies what the earlier attempts could not:**

| account | arm `ab` | arm `ba` |
|---|---|---|
| the subspace carries *which definition is in scope* | positive | **positive** |
| the subspace carries *the token `b`* | positive | **negative** |
| the subspace carries *the answer* | positive | **negative** |

E11 could not build this table. With an arithmetic operation between the value
and the answer it had to forbid `answer == value` to avoid circularity, and paid
for it with a capability requirement. **Here the answer *is* the bound value —
deliberately — and the arm crossing breaks the circularity instead.** So there
is **no arithmetic anywhere**: the model has to return a variable. That is
exactly the coupling that sank E12.

And the intervention has **no dose parameter**. An interchange installs whatever
the donor run holds in the subspace; magnitude is a measured consequence, not a
choice. That is the E11 failure removed by construction rather than argued away.

**Six gates**, each refusing to run downstream stages until it passes:

| gate | asserts | threshold |
|---|---|---|
| H0 | execution and a scope-aware reference interpreter agree; invariants hold, **including the arm crossing** | ≥ 0.999 of bases |
| H1 | the model returns the correctly bound variable | ≥ 0.85 overall, ≥ 0.75 per cell |
| H2 | which definition is in scope is decodable at the use anchor | ≥ 0.80, ≥ 0.10 over the measured surface baseline |
| H3 | whole-state interchange flips the answer **in both arms** | CI > 0, flip rate ≥ 0.25 |
| H4 | low-rank interchange beats matched controls on the **training** arm | ≥ 50% of the ceiling |
| H5 | the same subspace transfers to the **held-out** arm | ≥ 50% of that arm's ceiling, **and** `answer_direction` fails there |

**Controls.** `whole_state` (the ceiling, per arm — and the proof both arms are
measurable); **`answer_direction`** (an explicit answer direction, norm-matched
to the treatment, which *must* pass `ab` and *must* fail `ba` — the positive
control for the falsification itself); `random_rank`; `random_norm` (matched on
removed norm, since for an orthogonal projector only the span matters); `noop`
(provably the zero edit); and the `def_source` site (a structural zero — the
programs are token-identical before the mutation).

**Found so far** (6.7B, 400 bases):

| gate | result |
|---|---|
| H0 | **PASS** — 400/400 bases, all six invariant checks at 1.0000 |
| H1 | **PASS** — 1.000 overall, 1.000 in the weakest cell |
| H2 | **PASS** — binding decodable at 1.000 against a measured 0.500 surface floor |
| H3 | **PASS** — ab +4.781, ba +4.799, flip rate 0.857; structural zeros exactly 0.00e+00 |
| H4 | **PASS** — +9.029 [8.952, 9.108] on the training arm; all three control contrasts clear zero |
| H5 | **PASS** — +9.009 [8.933, 9.089] on the **held-out** arm, which the subspace was never fitted on |

**The result.** A rank-1 subspace at the use anchor, layer 8, fitted on arm `ab`
alone, makes the model emit the value the *installed binding* selects on
**100.0% of held-out rows in both arms** — 280 base programs, 560 rows per cell,
cluster bootstrap over bases. The metric is the full-vocabulary argmax, not the
logit margin, because `delta_ld` is positively biased when clean accuracy is at
ceiling and any disruption inflates it.

| variant | `ab` emits installed | `ba` emits installed | edit fraction |
|---|---:|---:|---:|
| **`das_binding`** (rank 1, learned) | **100.0%** | **100.0%** | **0.479** |
| `whole_state` (the entire donor state) | 85.7% | 87.9% | 0.805 |
| **`mean_difference`** (rank 1, closed form) | 76.1% | 76.8% | 0.711 |
| `answer_direction` (J-lens, norm-matched) | 27.9% | 4.3% | 0.479 |
| `random_norm` (dose-matched random) | 2.1% | 1.8% | 0.513 |
| `random_rank`, `noop`, raw unembedding | 0.0% | 0.0% | 0.018 / 0 / 0.479 |

Two accounts are refuted rather than merely unsupported. **Not disruption:** the
random control is *over*-dosed — 0.538 of ‖h‖ against the treatment's 0.479 — and
at that larger dose produces the installed answer on 1.1% of rows against 100%,
with the model never emitting a non-candidate token. **Not an answer direction:**
the explicit one attenuates 6.9× across the arms while the treatment does not
attenuate at all, and it pushes the model off-candidate on 9.1% of rows where the
treatment never does.

**The baseline the |cos| 0.673 demanded.** The learned direction is
substantially aligned with the mean donor−host difference without being identical
to it, and no cosine settles whether the optimiser earned the remainder. So the
difference-in-means direction — no optimiser, no labels, one fixed direction for
every example — was run as its own arm. It transports: 76.1% / 76.8%, transfer
ratio 1.003. A fixed direction really does carry much of the binding.

It also loses. `das_binding` gets 100% while moving 0.479 of ‖h‖ against the
baseline's 76% at 0.711 — about twice the effect per unit dose — and captures
*less* of the raw state difference (59.5% vs 88.2%). The learned direction is
therefore not a better-aligned version of the difference in means; it is a
different direction that works better while disturbing the state less.

**Still open:** why a rank-1 edit beats the whole-state patch (100% vs 86%). The
available account — the full patch installs the driving component *and*
components that fight it — is plausible and not independently demonstrated. And
this is one site, one layer, one model, one construction.

Stages 100–108 · design `docs/design/E13_PLAN.md` · commands
`docs/RUNBOOK_E13.md`.

---

# 4. Shared conventions

These apply everywhere and are the reason the numbers mean what they claim.
Full rationale: `docs/METHODS.md`.

| Convention | What it kills |
|---|---|
| **Grouped cross-validation** — folds split by source program, never within | rows from one program share hidden vectors; random folds leak train into test |
| **Selectivity control** — the identical probe on labels shuffled *within* each program | accuracy from class priors or per-program regularities |
| **Negative strata reported separately** | a headline averaged over easy negatives |
| **Measured surface baseline** — a probe on ±3 token ids + bucketed distance, no hidden states | any claim that a hidden-state result beats "the text" without checking |
| **Verified token alignment** — AST spans → offsets checked against the source | string-matching a variable name silently mislabels shadows, the exact thing E2 measures |
| **Cross-validated ground truth** — def-use tested against `beniget`; obfuscation execution-verified; E13 against a scope-aware interpreter | labels wrong in the same way for train and test look like signal |
| **Cluster bootstrap** over source programs; control comparisons paired on the same rows | intervals too narrow, in the direction that makes a null look like a finding |
| **Calibration/test separation**, with layer and site chosen on calibration and recorded before test numbers are read | a site picked after seeing the test split is a maximum, not a site |

One non-obvious hazard: `AutoTokenizer` on transformers 5.x silently
mis-tokenizes deepseek-coder (`def func` → `['de','ff','unc']`), relabelling
*every* example without crashing. `src/models/loader.py::load_tokenizer` rejects
any tokenizer failing an exact code round-trip.

---

# 5. Models and replication

| Role | Model | Where |
|---|---|---|
| Development, smoke, pilots | `deepseek-coder-1.3b` | local MPS or cluster |
| Main results | `deepseek-coder-6.7b` | cluster GPU |
| Architecture replication (optional) | `starcoder2-3b` | cluster GPU |

Base (non-instruct) models on purpose: the object of study is the
representation built during code pretraining, not chat behaviour. Every stage
takes `--model`; probed layers per model live in `configs/models.yaml`, and
canonical per-stage settings in `configs/experiments.yaml`.
