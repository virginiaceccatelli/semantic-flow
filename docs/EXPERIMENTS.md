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
- [§1 Phase I — Is the relation there?](#1-phase-i--is-the-relation-there) — E1, E2, E3, E4, E8
- [§2 Phase II — What is it made of?](#2-phase-ii--what-is-it-made-of) — E5, E9
- [§3 Phase III — Is it causally used?](#3-phase-iii--is-it-causally-used) — E7, E10, E11, E12, E13
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
  Phase I    E1 machinery ─→ E2 binding ─→ E3 def-use ─→ E8 real code
             (sanity)        THE FLOOR      same floor    transfer
                                │
                                │  E4 control dependence: floor is 0.927, not 0.500
                                │  → demoted; kept as the CONTRAST that makes E2 mean something
                                ↓
  Phase II   E5 context: distance is cheap, interference is not
             E9 obfuscation: renaming survives mid-layer, flattening does not
                                │
                                ↓
  Phase III  "decodable" is not "used". Four attempts:
             E7  whole-state patch      → transports the tokens too       [claim retired]
             E10 J-lens readout track   → instrument OK, both uses failed [archived]
             E11 rank-2 coordinate swap → below the site's causal dose    [NO-GO, retracted]
             E12 latent store transfer  → bottlenecked on arithmetic      [parked]
             E13 binding interchange    → running now
```

Phase III is where the work is, and its history is the project's real
methodological content. Each attempt failed for a *different, nameable* reason,
and each reason narrowed the design space for the next. `docs/ARCHIVE.md`
records all four in detail.

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

## E1 — lexical token type (machinery check, not a finding)

**Question.** Can a linear probe recover a token's syntactic class (keyword /
identifier / literal / operator) from a single hidden state?

**Why it exists.** It is a smoke test for the whole extraction chain. Token type
is a pure surface property, so it must be near-ceiling at the embedding layer.
If it is not, the tokenizer, the AST→token alignment or the activation store is
broken, and every other number in the project is meaningless.

**Method.** Multiclass linear probe on single positions, labels from
`classify_token`, grouped cross-validation by source program.

**Found.** Accuracy **1.000 at the embedding layer**, selectivity 0.88–0.90, in
both models. Exactly as designed.

**Reading.** Not a result about code models. It is the contrast that makes E2
interpretable: lexical features are readable *before* any computation happens;
semantic relations are not, and only appear after it.

Stage 20 · rows `task=lexical_token_type` in `static_probes_*.csv`.

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

## E8 — does any of this transfer to real code?

**Question.** Are E2/E3 artifacts of a synthetic generator?

**Method.** Stages 10 and 20 re-run unchanged on ~200 `ast`-parseable
CodeSearchNet Python functions, fixed seed.

**Found** (6.7B, aggregate AUC — read AUC, not accuracy, since accuracy is
threshold-dependent and peaks at the embedding layer here):

| | surface | embedding | peak | last |
|---|---:|---:|---:|---:|
| binding | 0.673 | 0.962 | **0.978** (L7) | 0.913 |
| def-use | 0.590 | 0.958 | **0.979** (L3) | 0.907 |

**Reading, with the limitation stated in the same breath.** Hidden states beat
the surface baseline by +0.31/+0.39 AUC and the layer profile matches synthetic
at the same relative depth, which rules out a pure generator-template
explanation. But **real identifiers are genuinely informative** — `self._cache`
and `result` look different — so the embedding layer starts at 0.96 and no
stratum pins the floor to chance. E8 shows *the decoder transfers to
naturalistic input*; it does **not** show that the semantic component
specifically transfers. E2's isolation still rests on synthetic programs.

Stages 10, 20 on `data/real/csn_python_200.jsonl`.

---

# 2. Phase II — What is it made of?

Phase I says a relation is present. It does not say what the representation is
built from. Phase II breaks it in controlled ways: **the probes are fitted once
on the base programs and then frozen**, never refitted on a perturbed variant,
so any change in accuracy is a change in the model's state rather than in the
probe. Ground truth is recomputed from each variant's own source, so a
perturbation that genuinely changes the answer is scored against the new answer.

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

**Found** (6.7B binding, best layer per level): 1.000 → **0.897** (rename) →
0.857 → 0.846 → **0.750** (flatten).

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

---

# 3. Phase III — Is it causally used?

Everything above is correlational. A representation can be a faithful shadow of
a computation happening somewhere else, and probing cannot tell the difference.
Phase III needs an intervention.

The requirement is harder than it sounds. A useful intervention must change
**the relation and nothing else** — which means acting at a position where the
programs are token-identical, editing a nameable part of the state rather than
replacing it, and doing so at a magnitude the site can actually register. No
attempt so far has had all three at once. What follows is four attempts, in
order, and what each established.

## E7 — activation patching (attempt 1: too coarse)

**Question.** Does patching a clean program's state into a corrupted one move
the answer?

**Method.** Length-matched taint minimal pairs, identical except the sink
argument. Patch the clean run's residual stream into the corrupted run at one
(layer, position) at a time; measure logit-difference recovery.

**Found** (6.7B mean recovery):

| layer | `sink_arg` | `last_token` | `sanitizer_def` |
|---:|---:|---:|---:|
| 0 | **0.99** | −0.01 | 0.00 |
| 15 | 0.24 | 0.31 | 0.00 |
| 31 | 0.00 | **1.00** | 0.00 |

**What survives.** A reproducible description of *where the decision becomes
committed*: the causal locus migrates from the sink-argument token to the
last-token position across the middle of the network, crossing over near where
E2's binding curve plateaus.

**What was retired, and why.** The claim that it *isolates semantic use*.
`sink_arg` is the only place the two programs differ — that is how the pair is
built — so patching there transports the surface difference along with any
semantic state, and ~1.0 recovery at layer 0 is exactly what pure input
restoration would produce. The `sanitizer_def` null has no positive control at
that position, so it is absence of evidence at one hand-picked token. And
late-layer `last_token` recovery forces the answer trivially.

**Lesson carried forward.** *Intervene only where the inputs agree.*

Stage 50 · `docs/ARCHIVE.md` for the full retirement.

## E10 — the J-lens track (attempt 2: instrument fine, uses failed)

**E10-0 (kept, supporting).** Validation of a Jacobian-corrected output-aligned
readout: at the last decoder layer the Jacobian is provably the identity, so the
J-lens must reproduce the logit lens exactly — measured cosine **1.0000**, a
closed-form check of the entire gradient path. Next-token recovery beats a
norm-matched floor, and the correction adds +0.15/+0.18 top-1 over the plain
logit lens pre-final. This is a statement about the *instrument*, and it is the
only part of the track that survives.

**E10-2 and E10-3 (archived).** Two attempts to use that readout — for taint
"verbalizability" and for a control-dependence probe/lens dissociation. Both are
retired: E10-2 inherits a metric that was itself shown to be broken, and E10-3's
positive control was an *identity* control where the test was *relational*, so
"the model cannot report this" and "this readout cannot express relations here"
remain indistinguishable. Full reasoning in `docs/ARCHIVE.md`.

**Lesson carried forward.** *A null needs a positive control matched in kind —
same type of question, same site, same states.*

Stages 60, 61, 62.

## E11 — the J-space coordinate swap (attempt 3: too fine)

**Question.** When the model resolves a binding, does it route the *selected
value* into output-aligned coordinates that downstream computation reuses?

**Method.** Token-aligned counterfactual pairs where a one-token mutation of an
inner definition's name flips which value a marked use selects, with both values
present in both programs and the answer computed by five different downstream
operations. The intervention exchanges just two coordinates,
`h ← h + V(swap(c) − c)`, leaving the orthogonal complement untouched. The
falsification: one edit must produce a *different correct answer* in each
operation family, which answer-steering cannot do.

**Found.** At the readout position (L24) the swap reaches 46% of the efficiency
of an ideal same-norm push while two matched-norm controls reach zero. But:

- **The pre-registered gate reads NO-GO at both positions.** Behavioural
  balanced accuracy 0.706 against a 0.75 threshold, and
  `swap_is_specific_to_the_value_subspace` **fails** — the plain logit lens is
  more efficient than the Jacobian-corrected one (−0.016 [−0.024, −0.009]).
- **The use-position null was retracted.** A dose-matched control showed the
  site's response to small edits is strongly convex: efficiency rises 18× from
  the smallest dose to the largest, and a push along the *known-correct*
  direction at 2% of ‖h‖ yields 0.002 nats with an interval covering zero — the
  same as the value swap at 3.7%. No two-dimensional edit is large enough to
  test the question there.

**Reading.** Without the dose control this would have been reported as a clean
null, with a passing readout positive control, four subspace controls at the
same magnitude, and a site potent enough to flip 22% of answers when replaced
wholesale. It is the most instructive failure in the project.

**Lesson carried forward.** *A positive control must be matched in **scale** as
well as in kind. A low-rank edit may sit below a site's effective causal dose,
and whether it does is a measured property of the site, not an assumption.*

Stages 70–74 · `results/jspace/6.7b-5fam/go_no_go*.md`.

## E12 — latent store transitions (attempt 4: parked)

**Question.** Does the model hold a computed value that appears *nowhere in the
program text*, and apply the program's own transition function to it?

**Why it was tried.** E11's swapped values are literals in the text, so its
surviving claim is about output-aligned *token* directions. Tracking a
text-absent value removes that escape route.

**Why it is parked.** Text-absent-because-computed forces arithmetic, and the
design made *two chained arithmetic steps* the load-bearing capability for a
question about program state. The 1.3B behavioural gate returned **0.418 —
below chance** — with the correct answer as argmax on 6.3% of prompts against a
10% uniform floor. Two of four operation families sat at exactly 0.500, which a
simulation showed is reproduced by a model doing **no computation at all** and
picking whichever candidate is numerically closer to the head literal.

**Reading.** Not a finding about code models; a design error. The published
prediction was available in advance: arithmetic in language models is
implemented by heuristic neurons that do not chain. Code, gates and diagnostics
are kept and still run; nothing is claimed.

**Lesson carried forward.** *Do not couple the semantic question to a capability
that is not the phenomenon of interest.*

Stages 80–89 · `docs/design/archive/E12_PLAN.md`.

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
| H4, H5 | **not yet valid** — see below |

**H4/H5 are pending a re-run.** The first attempt produced three anomalies that
a working apparatus cannot produce: `das_binding` at **189% of the whole-state
ceiling** (a rank-1 subspace cannot out-move installing the entire donor state),
a rank-1 edit moving **48% of ‖h‖**, and the `answer_direction` control reading
**+0.001 on both arms** — i.e. discriminating nothing. The last is a bug in the
control: it was a *unit-norm* unembedding row moving ~1% of ‖h‖ while the
treatment moved 48%, which is the E11 dose error rebuilt inside the control. It
is now norm-matched per row and verified exact. Stage 108 refuses to report a
reading while the machinery is broken, which is why no result is claimed here.

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
