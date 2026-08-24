# Archive — what was retired, and what each failure produced

Everything tried here that did not survive its own controls. **Nothing is
deleted**: every raw CSV is still in `results/tables/`, every figure in
`results/figures/`, every manifest in `results/manifests/`, and every stage
command still runs. What is withdrawn is the **claim**, not the data.

This file exists because the retirements are the project's methodological
content. Four intervention designs were attempted before the one that worked, and
each failed for a *different* reason that constrained the next. Read in order,
they are an argument about what a causal claim in interpretability requires.

| | Attempt | Failed because | Lesson carried into the surviving design |
|---|---|---|---|
| 1 | **Whole-state activation patching** | the informative position is the only place the two programs differ, so the patch transports the *input* | intervene only where the inputs agree |
| 2 | **J-lens readout uses** (taint, control dependence) | the positive control was an *identity* control where the test was *relational* | a null needs a positive control matched **in kind** |
| 3 | **Rank-2 J-space coordinate swap** | the site's dose–response is 18× convex, so the edit was below its effective causal dose | … and matched **in scale** — or use an intervention with no dose parameter |
| 4 | **Latent store transitions** | the design made two-step arithmetic the load-bearing capability for a question about program state | do not couple the semantic question to an unrelated capability |
| — | **Behavioural lead time** | the metric rewarded unreliable readouts | a metric that cannot separate signal from noise is untrustworthy in *either* direction |

Current status per experiment: `results/STATUS.yaml`.
Currently supported findings: [RESULTS.md](RESULTS.md).

**Sections 1–3** are designs that failed their own controls. **Section 4** is
work that is sound but does not carry a claim — including two results that were
in the main document until 2026-08-24 and were demoted on evidence: control
dependence (§4.3, surface floor 0.927) and the security-lexicon vocabulary
contrast (§4.4, does not beat a random direction at its reported cell on one of
three models). **Section 5** records a gate criterion that was changed after
seeing data, with the numbers under both rules.

---

# 1. The four intervention attempts

## 1.1 Whole-state activation patching (claim retired; the stage is not)

**Retired claim.** That patching the sink-argument state isolates *semantic use*
of the taint relation — in particular that "the model represents the sanitization
and does not route it to the answer".

**Why.**

1. **The patched position is where the two programs' tokens differ.** It is the
   only place they differ — that is how the pair is constructed. Patching there
   transports the surface difference along with any semantic state, so the ~1.0
   recovery at early layers is exactly what a pure input-restoration effect would
   produce. The design cannot separate the two.
2. **The null at the sanitizer site is uncontrolled.** Recovery is 0.000 at every
   layer, and the conclusion drawn was that the site is causally inert. But
   nothing establishes that *any* intervention there could have moved the output,
   so the null is absence of evidence at one hand-picked token.
3. **The last-token column is not evidence of anything semantic** — patching the
   readout position at late layers forces the answer trivially.

**What survives.** The routing observation itself: the causal locus of the
decision migrates from the sink-argument token (recovery 0.99 at layer 0) to the
last-token position (1.00 at the final layer), crossing over around the middle of
the network. That is a real and reproducible description of *where* the model's
decision becomes committed. It is preliminary because the intervention is a
whole-state replacement, which changes everything the position encodes at once.

**Preserved:** `causal_patching{,_summary}_{model}.csv`,
`patching_recovery_{model}.png`. Stage 50 still runs.

## 1.2 J-lens readout uses (archived)

Two uses of the validated J-lens, both archived. **The instrument validation
itself survives; it is instrument work and lives in
[METHODS §6.3](METHODS.md#63-the-j-lens-a-first-order-causal-correction), with §4.5
below recording why it is no longer listed as a result.**

**Taint / lead time.** Built to explain a behavioural effect that did not survive
its own floors (§2), and it inherits that metric wholesale. It also produced the
measurement that condemned the metric: across all 40 (layer, readout) cells,
early-warning rate is predicted by readout *unreliability* at Pearson r = −0.905
(p = 1.1e-15), and a norm-matched random direction carrying no information posts a
*higher* mean early-warning rate (0.634) than the J-lens (0.481), the logit lens
(0.373) or the trained probe (0.354). Informative about the method, uninformative
about the model.

**Control dependence — "decodable but not verbalizable".** The measurements held
up and were carefully done: at guard anchors the J-lens ranks the guard's own
variable above another present variable at 0.813 by the last layer, while the
control-dependence comparison sits within ±0.02 of chance everywhere. A
below-chance tail turned out to be a temporal confound (the `indent_matched`
negative precedes the anchor half the time, so recency favours the wrong answer);
conditioning on the matched subset removed it completely, and after Bonferroni
correction with a cluster bootstrap, **no layer differs from chance** — a clean,
well-defended null.

Archived anyway, for two reasons that are not about execution:

1. **The positive control does not license the inference.** `guard_var` shows the
   readout is alive and identifier-sensitive at those anchors. It does *not* show
   that a **relational** fact would be readable there if one were present — it is
   a recency/identity control, not a relational one. The intended relational
   control failed for a mechanical reason (the guard anchor is the trailing
   literal of the guard expression, so the "next" identifier is several tokens
   downstream past the colon and the indent). With no relational positive
   control, "not verbalizable" and "this readout cannot express relations at
   these positions" remain indistinguishable.
2. **The relation is the wrong one to build on.** Control dependence's own surface
   baseline is 0.927 (§4.3).
   A dissociation between "decodable" and "verbalizable" is least interesting
   exactly where decodability is mostly syntactic.

**This is the failure that produced the positive-control discipline of
[METHODS §7.3](METHODS.md#73-three-ways-a-null-could-be-wrong-and-the-measurement-for-each)** — and
[R8](RESULTS.md#r8--the-positive-control-the-readout-is-not-blind-and-the-security-words-run-backwards)
is what a
positive control matched *in kind* looks like when it is done right.

**Preserved:** `jlens_taint{,_summary,_prefixes,_sanity}_{model}.csv`,
`jlens_controldep{,_summary}_{model}.csv`, and their figures.
`scripts/63_controldep_temporal.py` reproduces the temporal-confound analysis.

## 1.3 J-space rank-2 coordinate swap (NO-GO; the use-position null retracted)

**Not archived — reported, and read narrowly.** What is recorded here is that it
did not pass its own pre-registration, which the headline sentence can obscure.
**Both go/no-go files read NO-GO**:

| criterion | use position | answer position |
|---|---|---|
| `behavioural_balanced_accuracy` (≥ 0.75) | FAIL 0.706 | FAIL 0.706 |
| `readout_beats_random_control` | FAIL +0.056 [−0.007, +0.117] | PASS +0.257 |
| `swap_moves_logits_toward_swapped_value` | FAIL +0.001 [−0.002, +0.004] | PASS +0.141 |
| `swap_is_specific_to_the_value_subspace` | FAIL | **FAIL** −0.016 [−0.024, −0.009] |
| cross-operation, all families positive | False | False |

**Retracted: the use-position null.** A dose-matched control added after the run
showed the site's response to small edits is strongly **convex** — efficiency
rises **18×** from the smallest dose to the largest, and a push along the
*known-correct* direction at 2% of ‖h‖ produces 0.002 nats with an interval
covering zero, the same as the value swap at 3.7%. **No two-dimensional edit is
large enough to test the question at that site.**

Without that control this would have been reported as a clean null: a passing
readout positive control, four subspace controls at the same magnitude, and a
site potent enough to flip 22% of answers when replaced wholesale. **It is the
most instructive failure in the project**, and it is why every subsequent design
either has no dose parameter or measures the site's response curve first — see
[METHODS §8.2](METHODS.md#82-the-interchange-operator-and-why-it-has-no-dose-knob).

**Also not attributable to the Jacobian correction.** The plain logit lens is more
efficient at the same site (2.35 vs 1.82), and at the last layer the two are equal
by construction.

**Outstanding, and the origin of the hard-gate mechanism:** the `probe_basis`
control **never ran** — the swap stage was invoked before the readout stage, so no
frozen probes were on disk and the variant was **silently skipped rather than
refused**.

**Preserved:** `results/jspace/`, `jspace_{lens,readout,behaviour,swap}_*.csv`.

## 1.4 Latent store transitions (parked before any claim)

**Never claimed anything.** Built as instrument validation: can a computed,
**text-absent** program value be identified and interchanged such that downstream
computation *transforms* it? Parked at its behavioural gate.

**Why it was tried.** The coordinate-swap design's values are literals in the
program text, so its surviving claim is about output-aligned *token* directions.
Tracking a value that appears nowhere in the text removes that escape route — in
`a = 1; c = a + 4; d = c + 3`, the value of `c` has no token, so a direction
carrying it cannot be a token-presence direction.

**Why it is parked.** Text-absent-because-computed **forces arithmetic**, and the
design thereby made two chained arithmetic steps the load-bearing capability for a
question about program state. On 1.3B:

- balanced accuracy **0.418 — below chance** on a two-alternative forced choice;
- the correct answer was the argmax on **6.3%** of prompts, against **10%** for a
  uniform random digit;
- two of four operation families sat at **exactly 0.500**.

That exact-0.500 pattern is the tell. A simulated model doing **no computation at
all** — picking whichever candidate is numerically closer to the head literal —
reproduces it precisely:

```
proximity to head (no computation)  overall 0.494  add 0.500  double_sub 0.500
observed, deepseek-coder-1.3b       overall 0.418  add 0.500  double_sub 0.500
```

On a monotone operation the two candidates straddle the anchor symmetrically, so a
pure proximity rule scores exactly chance. Two of four families therefore had **no
computational headroom in the metric at all**, independent of model size.

**The prediction was available in advance.** Arithmetic in language models is
implemented by a sparse set of pattern-matching heuristic neurons that do not
chain (Nikankin et al., [arXiv:2410.21272](https://arxiv.org/abs/2410.21272)). A
two-step chain was the wrong thing to require.

**Lesson carried forward, and it is the one that produced the surviving design.**
Do not couple the semantic question to a capability that is not the phenomenon of
interest. The binding interchange requires **no arithmetic anywhere** — the model
returns a variable — and gets its falsification from a value-assignment factorial
instead. Its gate H1 exists to check the model can do the task *before* an
instrument is built on top of it.

**Preserved:** all code, gates and diagnostics still run; `89_store_diagnose.py`
separates a constant responder, a format that elicits no digit, a model answering
the *intermediate* value, and a genuine capability limit.

---

# 2. Behavioural lead time (archived)

**Retired claim.** That latent taint-state corruption precedes behavioural failure
(`lead_time > 0`), and that the effect is scale-dependent.

**What happened, in order.**

**1. The original positive was measured on a constant responder.** Under the bare
prompt, both models answered the same token to every prefix of every program:

| Model | answer | raw accuracy | **balanced accuracy** |
|---|---|---:|---:|
| 1.3b | always "no" | 0.220 | **0.500** |
| 6.7b | always "yes" | **0.780** | **0.500** |

6.7b's 0.780 is exactly the base rate of `tainted=1`, so the behavioural signal
looked healthy under the check anyone would run. The generator always emits the
taint source on line 2, so a model that always says "yes" is wrong only at and
after the sanitizer, which places the failure point mid-program on exactly the
sanitized programs and hands a "lead" to any readout that errs before them. The
entire reported scale split was produced by the two models' opposite constant
biases.

**2. The prompt was fixed, and only one arm survived.** Few-shot demonstrations
*and* naming the variable lift 6.7b to balanced accuracy 0.857
(`scripts/diagnose_taint_prompt.py` sweeps the four variants and documents that
neither ingredient works alone). 1.3b cannot do the task under any prompt —
balanced accuracy 0.471 — so lead time there is **undefined**, not zero.

**3. With floors in place, the metric itself turned out to be broken.** Against an
analytic null — for a readout with per-prefix error rate ε whose errors are
independent of the model's state, P(errs before step *k*) = 1−(1−ε)^(k−1):

| Readout | 6.7b excess over the null |
|---|---:|
| `position` — no model at all, "tainted iff step ≤ 3" | **+0.113** |
| `random` — norm-matched random direction | +0.005 … +0.067 |
| `probe` — trained, ~99% accurate | **−0.010** |

A baseline that knows nothing but how many lines into the program it is scores
higher than a 99%-accurate probe, and not one of the probe's positive cells
survives Bonferroni correction. Within readout families, early-warning rate and
accuracy correlate at r ≈ −0.9: **the metric rewards unreliability, not
anticipation.**

**Why the *negative* is not a finding either.** A metric that cannot distinguish
anticipation from unreliability does not become trustworthy when it returns a
null. The honest summary is that the design cannot answer its question, on either
model.

**Preserved:** `behavioral_leadtime{,_summary,_prefixes}_{model}.csv`,
`behavioral_sanity_{model}.csv` and figures. `scripts/41_leadtime_floors.py`
re-applies the floors to any stage-40 run without a GPU.

---

# 3. The retired interpretive frame

An earlier version of this project closed with a single synthesizing claim:

> The model computes program semantics, causally uses some of them, and reports
> none of them.

with a global-workspace reading layered on top — that the J-lens measures
*verbalizability*, that relations absent from it are computed "outside the
workspace", and that binding is anticipatory in a way the output head does not
reflect.

**Why it was retired.** The claim is a conjunction of three parts, and two rested
on experiments that failed their own controls:

| part | rested on | what happened |
|---|---|---|
| "computes" | binding, def–use | **survives** — it is the foundation in [RESULTS.md Part I](RESULTS.md#part-i--the-relation-is-represented) |
| "causally uses" | whole-state patching | the design cannot separate semantic use from transported surface difference (§1.1) |
| "reports none of them" | lead time, J-lens uses | all three nulls turned out to be uninformative rather than negative (§1.2, §2) |

A conjunction inherits the weakness of its weakest conjunct, and "does not report"
was carrying most of the interpretive weight while being the least supported
piece.

There is a second, structural reason. "Verbalizable" and "reportable" are claims
about what a model *would say if asked*, and a lens does not measure that: it
measures a first-order causal projection of a hidden state onto the output head.
Those coincide at the last layer, where the lens *is* the output head, and are not
the same thing anywhere else. Reading a mid-layer lens null as "the model cannot
report this" imports an interpretation the instrument does not support.

**What replaced it.** The lens is treated as an **output-aligned coordinate
system**, and the questions asked of it are about *format*: is the distinction in
those coordinates at all, is any single token carrying it, and — separately, with
its own positive control — would this readout detect verbalisation if it were
there. Those questions have answers ([RESULTS.md Part III](RESULTS.md#part-iii--what-form-it-is-in)),
and none of them requires the retired frame.

**Smaller claims that went with it:**

- *"Binding is anticipatory."* Never separately tested. It travelled with the
  lead-time result and has no independent support.
- *"The 1.3b/6.7b split is a scale effect."* Retired: the 1.3b arm was
  structurally forced (wrong at the first evaluable prefix on 100% of programs, so
  a positive lead is arithmetically impossible) and the 6.7b arm was uncontrolled.

---

# 4. Retired from the main documents, data untouched

Nothing in this section failed. Each entry is a measurement that is sound and
still reproducible, removed from [RESULTS.md](RESULTS.md) because it does not
carry a claim: either its floor was never pinned (§4.1, §4.2, §4.3), or its own
later calibration showed the effect does not clear a random-direction control
(§4.4), or it is instrument validation with no finding attached (§4.5).

## 4.1 Lexical token type

A probe for token type (identifier / keyword / literal / operator), decodable at
ceiling from the embedding layer onwards. **Never a finding** — its purpose was to
prove the extraction and probing pipeline worked at all, and once binding and
def–use ran with construction-pinned floors, that job was done by better evidence.
Ceiling accuracy on a property the tokenizer already encodes constrains nothing.
Still runnable; rows in `results/tables/static_probes_*.csv`.

## 4.2 Real-code transfer

Binding/def–use probes evaluated on CodeSearchNet Python: accuracy transfers with
the same layer signature. Retired from the main documents because the transfer
number is an **upper bound on what transfers semantically**, not a finding about
representation. Ground truth on real code comes from the same static analysis the
synthetic corpus uses, and real functions have **no context-matched stratum** — so
the surface floor that makes the foundation mean anything is *not pinned there*.
Reporting it beside the pinned results invited exactly the conflation the project
exists to avoid.

**What would revive it:** context-matched pairs built by *mutating* real
functions, which would make it a like-for-like replication of the isolation rather
than a transfer check. That is open item 8 in [RESULTS.md](RESULTS.md#open-items).


## 4.3 Control dependence

**What it was.** A pairwise probe for "does this statement execute under that
guard?" — the third natural CPG relation, read off the CFG with join points
resolved exactly. Negatives were statements in the *same* program outside the
guard, including an `indent_matched` hard stratum (a statement in a *sibling*
guard's body at the same nesting depth).

**The measurement, which is sound.** The hidden state dominates on both classes
at once, so the gap is not a threshold artifact:

| control_dep, best layer | positive recall | hard-negative recall |
|---|---:|---:|
| surface baseline (no model) | 0.959 | 0.676 |
| hidden — 1.3B (L11) | 0.981 | 0.873 |
| hidden — 6.7B (L15) | **0.995** | **0.923** |

Aggregate AUC rises 0.990 → 0.999.

**Why it is not a result.** The model-free surface baseline is already **0.927**.
A statement's guard is usually its nearest enclosing `if`, so token windows plus
indentation recover most of the relation with no model at all. Unlike binding and
def–use, **no construction pins this floor to chance** — there is no
context-matched stratum for control dependence, and building one would require
program pairs that are token-identical while the guard relation flips.

By the criterion in [METHODS §0.2](METHODS.md#02-the-working-definition), the
relation is *mostly syntactic*, so decoding it at ceiling constrains nothing about
semantic representation. It was carried in the main documents for a while as "the
contrast that makes binding mean something" — the evidence that the criterion
excludes things. That argument is worth keeping, and it now lives in METHODS §0.2
as a point about the *definition*, with the numbers here. It does not need to
occupy a slot in the results.

**A second, independent reason not to build on it.** The probing anchors fall on
each span's last token, which in this corpus are integer literals (`> 50`, `+ K`).
Re-anchoring on the guard variable and the statement target is a CPU-only re-run
and was never done, so even the measured numbers describe a slightly odd position.

**What would revive it.** Context-matched control-dependence pairs — two
token-identical programs in which the guard relation flips — which would pin the
floor the way binding's is pinned. Until then there is no honest headline number.

**Still runnable, data preserved.** `scripts/20_run_probes.py`; rows in
`results/tables/static_probes_*.csv` under `task=control_dep`. `results/STATUS.yaml`
records the demotion.

## 4.4 The security-lexicon vocabulary contrast

**What it was.** The first attempt to ask whether the safe/unsafe distinction is
expressed in the models' own output vocabulary. Sink-site states were mapped into
vocabulary space through three readouts (logit lens, J-lens, R-lens, with
`PRIMARY_LENS = "rlens"` declared in code before any result), against a 196-token
candidate pool discovered on the *training* split and frozen to disk, containing a
per-model-validated security lexicon. Orientation fixed once as
`score_unsafe − score_safe`.

**What it returned.** A null, in all three models — and an *inverted* one in 1.3B:

| clean held-out, R-lens, reported cell | 1.3B (L11) | 6.7B (L15) | SC2-3B (L15) |
|---|---:|---:|---:|
| concept token surviving the tokenizer | `" vulnerable"` | `" vulnerable"` | `" unsafe"` |
| held-out sign consistency | 0.153 | 0.403 | 0.694 |
| permutation p | 0.000 | 0.004 | 0.008 |

It was carefully controlled — three lenses agreeing at pairwise cosine 0.75–0.97,
a same-label arm collapsing the contrast to ≈0, the embedding-layer token-identity
contrast null at p = 0.71–0.81, and a measured distribution-shape confound never
exceeding |r| = 0.39.

**Why it was retired as a main result.** Three reasons, in order of weight.

**1. At the reported cell it does not beat a random direction, on one of three
models.** The specificity check added later — displacement of the real lens over
the best random/Gram-matched control — reads, from
`vocab/vocab_specificity.csv`:

| reported cell, clean held-out `sink_arg` | logit | J-lens | R-lens | beats random? |
|---|---:|---:|---:|---|
| 1.3B, L11 | 1.83 | 1.58 | 2.08 | yes |
| **6.7B, L15** | **0.875** | **0.875** | **0.875** | **no, all three** |
| SC2-3B, L15 | 0.50 | 1.30 | 1.40 | logit no; J/R yes |

Random and Gram-matched lenses follow the *same depth trajectory* as the real
ones, so the permutation null was detecting "the two states differ at all", which
any direction picks up. A null whose instrument does not clear a random direction
on a third of the evidence is uninformative in both directions.

**2. Both halves of its reading were superseded by measurements that do clear
their bars.** The "distributed, not lexicalised" half is established directly by
[R7](RESULTS.md#r7--the-distinction-is-in-the-output-basis-and-it-is-not-a-word)'s
loadings — flat, spread over thousands of meaningless tokens — with no pool and no
null involved. The "the security words do not carry it" half is established far
more sharply by [R8](RESULTS.md#r8--the-positive-control-the-readout-is-not-blind-and-the-security-words-run-backwards),
as a *within-cell dissociation*: at the cell where the taint poles separate the
pair at 0.889 / 0.944, the same lexicon in the same basis separates it at
0.347 / 0.389 — significantly in the wrong direction, on an instrument that is
demonstrably alive.

**3. The search it performs cannot be characterised.** The candidate pool is
selected by a full-vocabulary *logit-lens* ranking, so a direction only the J- or
R-lens would surface, on a token outside the pool, cannot be discovered. That
limitation is recorded inside the frozen artifact, and it is exactly what stage
128 removed.

**What was worth learning from it, and is kept.** Three things, all now stated
elsewhere. That the *unprompted* sink-argument state is where the security words
fail, which is what makes 6.7B's prompted 0.764 a narrow rather than a general
result (R8). That something replicates in that pool but is semantically arbitrary
— frozen training-discovered tokens reappear in the held-out top-k at
0.875 / 0.750 / 0.875 against 0.000–0.031 for random controls, and the tokens are
`" ?"`, `" liber"`, `"OrNull"`. And that under flattening the vocabulary contrast
degrades alongside the probe, so trained and output-aligned auditability are lost
together.

**One control it taught the project to stop trusting.** The original
`mismatched_pairs` arm redrew the *safe* partner from the same safe pool, so the
label difference survived it: its expected mean is the main arm's exactly, and on
the canonical runs the two agree to four decimal places on all three models — on
6.7B the *control* is more sign-consistent than the main arm (0.417 vs 0.403). It
falsifies "specific to this pairing", not "about the label". The same-label arm
that replaced it is now standard ([METHODS §7.2](METHODS.md#72-what-the-contrast-is-controlled-against)).

**Still runnable, data preserved.** Stages 125–127; everything under
`results/sinkflow/{model}/vocab/`, including `e15c_report.md`, the depth figures
`results/figures/e15c_depth_{model}.png`, and `vocab_specificity.csv`.

## 4.5 J-lens validation as a reported result

**Not retired — relocated.** The J-lens validation is correct and still gates the
lens track: at the last decoder layer `J` is provably the identity, so the J-lens
must equal the logit lens exactly, and it measures **cosine 1.0000** on all three
models — a closed-form check of the entire gradient path. Next-token top-1
recovery is 0.633–0.650 against chance 0.038, beating the plain logit lens by
+0.15 to +0.22 at pre-final layers.

It stopped being listed as a *result* because it is instrument validation with no
finding attached, and because the instrument it validates turned out to change no
conclusion: at every cell where a vocabulary readout actually fires, the logit
lens, J-lens and R-lens agree to within noise, and where the J-lens was used as a
coordinate system the plain logit lens was *more* efficient at the same site
(§1.3). It now lives in [METHODS §6.3](METHODS.md#63-the-j-lens-a-first-order-causal-correction),
where instrument validation belongs.

One caveat that travelled with it and still applies: StarCoder2 has no R-lens
(§ the architecture note in METHODS §6.4), so that model's vocabulary-contrast
numbers rest on this validation and nothing else.

**Still runnable, data preserved.** `scripts/60_jlens_validate.py`;
`results/tables/jlens_validation{,_checks}_{model}.csv`.

---

# 5. A gate criterion that was changed after seeing data

This is the thing the project is most careful about, so the change, its
justification, and the numbers under **both** rules are recorded in full.

**What changed.** H5's third condition — the explicit `answer_direction` control
must FAIL on the held-out arm — was evaluated on `delta_ld`, the two-way logit
margin. It is now evaluated on `says_installed`, the **full-vocabulary argmax**,
against the arm-to-arm ratio `whole_state` achieves on the same rows.

**Why this implements the pre-registration rather than relaxing it.** The module
docstring of `src/experiments/binding_interchange.py`, written before any 6.7B
run, states: *"`delta_ld` is positively biased and must not be gated on alone…
Every row therefore also records `says_installed`, the full-vocabulary argmax,
which a disruption cannot produce systematically, and the gates read that."* **No
gate evaluator ever read it.** The stated design and the implementation disagreed
from the start, and stages 106 and 108 reported different H5 verdicts on every run
because of it.

**Why the margin is the wrong metric here specifically.** H1 is 1.000 on this
corpus, so the clean distribution is confident and `logP(own)` sits far above
`logP(installed)`. Any edit that merely disrupts the state regresses both toward
the middle and *raises* `delta_ld` with nothing transported. Not hypothetical: the
control also knocked the model off both candidate tokens on 6.4% of held-out rows,
which is disruption, not transport.

**The 6.7B numbers under both rules** (site `use`, layer 8, rank 1, 280 held-out
bases). Only condition 3 differs:

| | margin `delta_ld` | argmax `says_installed` |
|---|---|---|
| `das_binding` transfers to `ba` | +9.009 [8.933, 9.089] = 188% of ceiling — **passes** | 100.0% = 114% of ceiling — **passes** |
| `answer_direction` on `ab` | +2.322 [2.157, 2.482] — works | 27.9% — works |
| `answer_direction` on `ba` | +0.335 [0.208, 0.456], interval clears zero — scored **"did not fail"** | 4.3%; arm ratio 0.154 against transport's 1.025 — **fails** |
| **H5 verdict** | **FAIL** (on condition 3) | **PASS** |

**The new rule, precisely.** A control fails on the held-out arm if its `ba`/`ab`
argmax ratio is below `MIN_TRANSFER_FRACTION` (0.50) of the ratio `whole_state`
achieves. The reference is *measured on the same rows* rather than chosen, and the
coefficient is the threshold already pre-registered for H5's first condition — **no
new free parameter was introduced.**

**What would have made this illegitimate**, and did not happen: inventing a
threshold that the observed 4.3% happens to clear; switching metrics on condition
1 as well, where the verdict is unchanged either way (188% of ceiling on the
margin, 114% on the argmax); or making the change without leaving the old numbers
on the record.

Runs predating `says_installed` still evaluate under the old rule — the fallback
is explicit in `evaluate_gate_h5` and pinned by a test. **The 6.7B `gates.yaml`
and `e13_report.md` on disk are such runs**, which is why they still read H5 FAIL;
re-running stages 106–107 regenerates them under the corrected rule without
changing a single measured row.
