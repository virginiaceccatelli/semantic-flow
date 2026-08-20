# E15-D — three follow-ups to the E15-C null

**Status: 128 run on all three models; 130 run on deepseek-coder-1.3b and not
applicable to starcoder2-3b; 129 (the positive control) NOT RUN.**
Stages 128–131, gates J2/J3/J4. Every threshold in this document is declared in
code (`src/experiments/sinkflow_{align,positive,relevance}.py`) and was written
before any canonical number was produced. Results are in `docs/RESULTS.md`; §7
below records what happened to each pre-declared criterion, including the two
that failed.

---

## 1. What E15-C established, and the hole in it

E15-C asked whether the safe→unsafe difference is expressed in the model's own
vocabulary, and returned a null on all three models: `inverted_security_vocabulary`
on deepseek-coder-1.3b, `stable_non_security_vocabulary` on the other two. The
mechanical gates J0 and J1 passed everywhere, with no overrides. Tier-1
re-analysis then *qualified* that null in two ways that matter here:

* **specificity** — measured against a norm- and Gram-matched random lens rather
  than against zero, the real lenses beat a random direction by a factor of only
  0.87–2.08 at the reported cells, and across all cells the median specificity is
  0.67–1.00 with `beats_random_lens` true in only 24–38%;
* **the distribution confound is ruled out** — |r| ≤ 0.39 anywhere between the
  contrast and the paired difference in candidate-distribution entropy or score
  norm, so the inverted 1.3B sign is real and currently unexplained.

Two problems survive that, and E15-D exists for them.

**Problem 1: the null is unfalsifiable in the direction that matters.** Every
control E15-C runs is *negative* — permutation, mismatched pairs, random lenses,
Gram-matched lenses. Negative controls establish that a positive result is not an
artifact. They are silent about a null. Nothing in E15-C separates

> the models do not verbalise this

from

> this machinery could not detect verbalisation if it were there.

**Problem 2: the readout can only find a lexicalised concept.** The candidate
pool is 196 tokens selected by a full-vocabulary *logit-lens* ranking of the mean
paired delta. Two limitations compound: a direction only the J-lens or R-lens
would surface, on a token outside the pool, cannot be discovered at all; and the
pool is ranked by the **mean**, which is large whenever the two members differ
systematically *in any way*, including when every pair's difference points
somewhere different.

---

## 2. A defect found in E15-C's controls, and the fix

`mismatched_pairs` keeps the unsafe member and redraws the **safe** partner from
the same safe pool at the same (condition, site), preferring one that also
matches family and structure. Its docstring claimed it separated

> "the contrast tracks the safe/unsafe difference" from "the contrast tracks any
> difference between two programs of this kind at all".

It does not, and cannot. The redrawn partner is still a *safe* program, so the
label difference survives the control intact; the arm averages over the very set
the main arm averages over, so its **expected mean is the main arm's exactly**
and only resampling noise separates the two. There is no systematic component for
a real effect to appear in. Measured on the canonical runs over 200 redraws, the
two agree to four decimal places on every model:

| at the reported cell | main mean Δ | cross-label redraw | same-label control |
|---|---:|---:|---:|
| deepseek-coder-1.3b L11 | −0.3041 | −0.3041 | +0.0000 |
| deepseek-coder-6.7b L15 | −0.1998 | −0.1998 | −0.0000 |
| starcoder2-3b L15 | +0.0940 | +0.0940 | +0.0000 |

Only per-pair statistics can move at all, and because the partner also matches
family and structure, in practice they barely do: sign consistency goes
0.153 → 0.167, 0.403 → **0.417** (the control is *more* consistent than the main
arm), 0.694 → 0.639. The check `above_mismatched_pair_control` therefore passed
by margins of 0.014, 0.014 and 0.056 against a comparison with no noise band.

**The fix (`sinkflow_vocab.same_label_pairs`, run by stage 126 as arms
`same_label_unsafe` and `same_label_safe`).** Both members come from the *same*
pole, different bases, same condition and site, with partner selection otherwise
identical. Everything a matched pair differs in — family, identifier draw, flow
structure, program identity — is still present, and the label difference is gone,
so the expected contrast is zero and the expected sign consistency 0.5. That is
the arm a label claim has to clear, and it is now a declared check
(`above_same_label_control`) plus a `pairing_diagnostics` block that states the
`pairing_gain` — what base matching actually buys — as a number rather than
leaving it implicit.

This does not overturn any E15-C result. It replaces one uninformative check with
an informative one, and corrects a docstring that overstated what the old arm
could do.

---

## 3. V1 (stage 128) — is there a shared full-vocabulary direction?

### The question

Drop the basis restriction entirely. For each matched pair at (layer, site,
condition), form

    d_p = z(W_U g · h_unsafe) − z(W_U g · h_safe) ∈ R^V,    u_p = d_p / ‖d_p‖

and ask whether the `u_p` point the same way. Nothing is chosen in advance; the
direction is *discovered* as the leading structure of the training-split
differences, and its top-loading tokens are read off afterwards. **A null here
cannot be blamed on a candidate pool, because there is no candidate pool.**

### The statistic, and why it is not the mean

| | |
|---|---|
| `sv1_share` | largest eigenvalue of the Gram matrix `U Uᵀ` over its trace: the fraction of the pairs' total energy along ONE direction. **Sign-invariant.** 1/n for unrelated differences, 1 for identical ones. Computed from the (n, n) Gram, not an SVD of (n, V). |
| `mean_pairwise_cosine` | the oriented version. Only meaningful for the main arm, and never compared against the same-label null, which has no canonical orientation. |
| `proj_*` | projection of held-out differences onto the direction frozen on the training split, with a cluster bootstrap CI over bases. |

`sv1_share` is the primary statistic precisely because it is sign-invariant: the
same-label null's members are `A − B` or `B − A` with equal probability, so any
oriented statistic averages to zero there and the comparison would be vacuous.

### Declared before the run

* primary site **`last_token`**, not `sink_arg` — measured on the benchmark, both
  members carry the same token id in **100%** of `last_token` pairs and only 75%
  of `sink_arg` pairs, so a difference at `last_token` cannot be token identity;
* `ALIGN_SIGN_CONSISTENCY = 0.70`, `SV1_MARGIN = 2.0`, `MIN_PAIRS_ALIGN = 24`;
* the layer −1 floor is measured at both sites and the reported layer must beat
  it. At `last_token` the floor is *exactly zero* — identical tokens give
  identical embeddings, so every difference is dropped as having no direction,
  and stage 131 treats an empty floor cell as the strongest possible pass rather
  than as a NaN failure.

### The freeze, and why it is weaker than E15-C's on purpose

E15-C freezes its token set across a process boundary, because selecting 196
tokens out of 32k is a discrete choice with many degrees of freedom. A **mean
over training pairs has none** — no token is selected, no threshold tuned — so
the direction cannot overfit the split it is estimated on. Stage 128 therefore
estimates and evaluates in one process, writes the direction to disk with its
provenance before held-out states are scored, and J2 checks that the base sets
are disjoint and that the split is recorded. The reasoning is stated here so it
is not mistaken for an oversight.

### One numerical detail that is not cosmetic

The zero-difference guard is **relative**, not absolute. States are read in
float32, so two genuinely identical members — or two differing only by a positive
scaling, which the z-score convention removes exactly — leave a residue of order
1e-6 after cancellation. An absolute bound would normalise that residue into a
unit vector of pure rounding noise and count it as a measurement. The threshold
is `1e-4 · sqrt(V)`, and the scale is exact rather than estimated: a z-scored
vector over V candidates has sum of squares exactly V. Real differences sit at
~0.3·sqrt(V); noise at ~1e-6·sqrt(V). This is the same lesson E14's R0 bound
records.

---

## 4. The positive control (stage 129) — the highest-value item

### The property

The E6/E7 forced-choice taint question, whose answer is a **single vocabulary
token** — the same constraint that made E15-C possible. `TAINT_QUESTION` and
`choice_token_ids` are already built and tokenizer-validated in
`jlens_validate`.

### Why it is the same measurement, checkably

One candidate basis carries both properties: `{" yes", " no"}` + the E15-C
security lexicon + random controls. `taint` and `security` are two
`VocabCandidates` over the **same `token_ids`**, differing only in which tokens
they name as poles, and both contrasts go through
`sinkflow_vocab.pair_contrast` — the same function, the same z-score convention,
the same unsafe-minus-safe orientation. J3 refuses the run if the two bases ever
differ. The random controls are not decoration: over two tokens alone every
z-score is ±1 and the softmax is a logistic of one margin, so the two properties
would not be on the same scale as each other or as E15-C.

Stage 129 deliberately does **not** require J0. A positive control that inherited
E15-C's candidate pool would inherit the limitation it exists to test.

### Two prompts, because prompt sensitivity is a confound

`e6` is `TAINT_QUESTION` verbatim, so the number is comparable to the E6/E7
track. `sink` names the sink the label is actually about, because "the current
value" is ambiguous in a program with two chains. Both run; both are reported.
Within a matched pair the prompt is *identical* — the sink is a property of the
base — and J3 fails the run if that is ever not true, since a paired contrast
across two different prompts would be measuring the prompt.

### The behavioural statistic the verdict uses

`pair_separation`: the fraction of bases where the unsafe member draws a higher
yes-margin than its matched safe counterpart. Raw accuracy is inflated by answer
bias — a model that says "no" to everything scores 0.5 for free — while pair
separation has a chance level of 0.5 that no answer bias can move. Both are
reported; only the paired one decides.

### What each outcome licenses — declared before the run

| outcome | what E15-C's null then means |
|---|---|
| `property_not_verbalised` | Coherent but weak. The models cannot answer either, so there was nothing to detect and the null does not discriminate between the models and the method. |
| `machinery_validated` | **The strongest available outcome.** The identical readout detects a property these models verbalise and not the security distinction, so "not expressed in output-aligned coordinates" becomes a supported claim about code models. |
| `machinery_blind` | The null is about the **method**. Every E15-C number keeps its caveat and no claim about the models survives that track. |
| `both_properties_detected` | E15-C's **pool** was the limitation, not its readout; the E15-C null should be re-reported as a pool artifact. |

The third row is the one that would retire the track. It is written here in
advance so that it cannot be reinterpreted afterwards.

---

## 5. V3 (stage 130) — relevance redistribution across AST roles

### The property being exploited

Under the LRP rules the tail network above layer `l` is degree-1 homogeneous, so
the Euler identity gives

    R_t = ⟨∂s/∂h_l,t , h_l,t⟩        Σ_t R_t = s

E14 gate R measures |ρ − 1| within 1e-4 at **every** layer on both DeepSeek
models. So `R_t / s` is the fraction of the answer position `t` is responsible
for, the fractions sum to one, and a difference between two members of a matched
pair is a genuine **redistribution** rather than a change of scale. This is the
property the vocabulary readout never had: E15-C's z-score convention exists
because `JLens.scores` drops an unknown positive factor, and here there is
nothing to drop because conservation fixes the total.

**It requires no lexicalisation at all**, which is what makes it independent of
both E15-C and V1.

### Roles, and the control that comes free

Relevance is summed over the syntactic role each token belongs to, recomputed
from **each variant's own source** — the discipline `find_anchors` already
follows — across eight roles in a precedence order that makes the assignment a
partition: `source_expr`, `trusted_expr`, `sink_arg`, `sink_call`, `taint_chain`,
`trust_chain`, `signature`, `other`. The chains are followed *structurally* (an
assignment joins a chain because its right-hand side mentions a name already on
it), so the partition survives alpha renaming.

**Only `sink_arg` differs in tokens between the two members.**
`pair_diff_is_confined_to_sink_arg` enforces that at generation time for every
condition, and both members of a base receive the *same* transformation draw. So
a redistribution measured among the **token-identical roles** cannot be the
differing sink-argument token, cannot be a length effect, and cannot be a
tokenisation artifact.

This was verified rather than assumed: across all 1440 held-out programs in all
ten conditions, the role partition resolves with no problems, no empty role, an
exact token partition, and **identical per-role token counts within every pair**,
including under `rename_opaque_encode_flatten`. `ntok_{role}_match` re-measures
it per pair at run time under the real tokenizer.

### Validity condition, checked and not assumed

Conservation is what licenses the fraction reading, so `rho` is recorded per
(pair, layer) and `relevance_conservation.csv` reports median |ρ − 1| per layer.
Where it exceeds `CONSERVATION_TOLERANCE = 0.25` the redistribution is not read
at that layer. Where the homogenising rules bind to **nothing** —
starcoder2-3b's LayerNorm plus non-gated MLP — stage 130 **refuses**, records J4
as *not applicable* with the rule counts, and says why. That is a fact about the
architecture, not a failed measurement.

Declared thresholds: `REDISTRIBUTION_SIGN_CONSISTENCY = 0.70`,
`PERMUTATION_P = 0.05`, `MIN_PAIRS_RELEVANCE = 24`. J4 also checks that the
per-role deltas **close** — they must sum to `ρ_unsafe − ρ_safe` — which is the
arithmetic that makes a difference of fractions a redistribution at all.

---

## 6. What is still not addressed

* **Causal use.** All three stages are observational, exactly as E15-C is. A
  shared direction, a detected verbalisation and a relevance shift are all
  statements about *format* and *attribution*, not about use. E13's interchange
  remains the causal instrument, and the original E15 extension's prohibition on
  J-space interventions still stands.
* **The inverted 1.3B sign.** Tier 1 ruled out the three cheap explanations
  (distribution shape, token identity, lens choice). V1 may localise it — if the
  shared direction exists and its top loadings are interpretable — but nothing
  here is designed to explain it.
* **Architecture generality of the R-lens.** Extending `norm_eps_attr` to
  LayerNorm and `is_gated_mlp` to non-gated MLPs is what would make stage 130
  runnable on StarCoder2. The LayerNorm half is the harder one: the
  mean-subtraction term is exactly what the current algebra assumes away.
* **The `W_U`-constrained probe** (roadmap item 9). V1 answers a nearby but
  distinct question — V1 asks whether the *observed* differences concentrate,
  the probe would ask whether a *fitted* output-aligned direction separates the
  classes. Both are worth having; only the first is built.

---

## 7. What happened to each pre-declared criterion

Recorded here rather than in `docs/RESULTS.md` because the point is the
*bookkeeping*: which criteria were written before the run, which passed, which
failed, and what was done about the failures.

### V1 (stage 128), all three models

| declared check | outcome |
|---|---|
| `direction_frozen_on_train` | **pass** — J2 passes on all three; train and evaluated bases disjoint |
| `heldout_projection_replicates` (sign ≥ 0.70 **and** bootstrap CI excludes 0) | **pass** — 1.000 / 1.000 / 1.000, CIs [0.360,0.406] / [0.358,0.401] / [0.351,0.429] |
| `above_surface_floor` | **pass** — the layer −1 floor is *exactly* zero at `last_token`; all 72 embedding differences vanish |
| `above_same_label_null` (`sv1_ratio ≥ 2.0`) | **FAIL** — 0.76 / 0.97 / 0.76 |

The declared verdict space had three outcomes and the data landed outside all of
them: the strict criterion for `shared_direction_found` was not met, but calling
the result `no_shared_direction` would misdescribe a direction that 72 of 72
held-out pairs project onto. A fourth label,
`direction_replicates_but_not_dominant`, was added **after** the run, and the
reasoning is worth stating because it is the kind of change that can hide a
moved goalpost:

* the criterion itself was **not** changed, relaxed, or re-tuned — `sv1_ratio ≥
  2.0` still appears in the checks table, still reads *no*, and still blocks
  `shared_direction_found`;
* what was added is a *name* for an outcome the original three-way space could
  not express, because the two statistics answer different questions.
  `proj_sign_consistency` asks whether a label-defined direction **generalises**
  to unseen programs; `sv1_share` asks whether the label axis **dominates** the
  difference vectors. A direction can generalise perfectly while being a modest
  component of a cloud whose largest axis is program-to-program variation;
* that the two are genuinely different axes rather than one is not an assumption:
  the frozen direction's top-100 loadings overlap the same-label direction's at a
  Jaccard index of 0.005 / 0.005 / 0.000.

`sv1_share` was chosen as primary because the same-label null has no orientation,
which makes any oriented comparison against it vacuous. That reasoning was
correct and is unchanged. What it did not anticipate is that the *unoriented*
comparison is dominated by a shared axis both arms possess, so it tests a
stronger claim than the design intended. Both numbers are reported.

### V3 (stage 130), deepseek-coder-1.3b

| declared check | outcome |
|---|---|
| `rules_installed_and_conserving` | **pass** — median \|ρ−1\| = 0.0000, max 5e-5, at every layer |
| `role_token_counts_matched` | **pass** — every token-identical role at 1.000; `sink_arg` at 0.611, the 44/72 length-matched pairs |
| `redistribution_consistent` (sign ≥ 0.70) | **pass** — 0.097 for `taint_chain` and 0.875 for `trust_chain` at layer 0 |
| `above_permutation_control` (mean's null, p < 0.05) | **FAIL** — 0.39–0.62 |

Same shape, same treatment. `permutation_null` tests the **mean**, and relevance
deltas are heavy-tailed enough that seven outlier pairs out of seventy-two flip
the mean's sign while the median and the sign stay put. The statistic that
survives is `sign_consistency`, which was itself pre-declared — and its exact
null under the *same* random-orientation scheme is Binomial(n, ½), because
flipping each base's orientation at random is exactly what makes the positive
count binomial. So `sign_test_p` is the permutation test **for the pre-declared
statistic**, not a second test chosen after seeing the data; it was added, the
mean-based check was kept and still reads *no*, and the verdict label
`redistribution_consistent_but_not_in_mean` says both.

Two defects in the reporting were also found and fixed by the same data:

* at the **last decoder layer** the tail network is the final norm and the
  unembedding at the readout position alone, so the score depends on one position
  and every other role's relevance is identically zero. `0 > 0` is false for
  every pair, so a naive sign consistency read 0.0 — maximal displacement from
  chance — and that structurally empty cell won the "largest effect" search
  outright. Such cells are now marked `degenerate`, excluded from the headline,
  and the last layer is dropped from the default layer list;
* `align_direction.json` serialised a full 32k-float row per (layer, site) cell
  as JSON text: 15–29 MB per model, 63 MB in total, that nothing downstream read.
  The vectors now go to a compressed `.npz` (7.9 MB total) and the JSON keeps
  provenance and per-cell statistics.

### The positive control (stage 129)

Not run. No manifest, no artifacts, J3 unrecorded on all three models. §4's
outcome table stands unused, and every sentence in it is still binding on
whatever it eventually returns.
