# Results

What this project currently establishes, what it does not, and what is still
running. Nothing here is a summary of intent: every row is a measurement that
exists in `results/tables/*.csv`, and every claim is paired with the control
that could have falsified it.

Three rules for reading this file:

- The machine-readable registry is `results/STATUS.yaml`, which
  `scripts/90_make_paper_assets.py` reads to decide which figures to regenerate.
- Claims this project **used to make and has withdrawn** are in
  `docs/ARCHIVE.md`, each with its reason. The data behind them is preserved.
- What each experiment does and why: `docs/EXPERIMENTS.md`.

---

## The one-paragraph version

Variable binding and def–use structure **are** represented in DeepSeek-Coder 1.3B
and 6.7B, above a floor pinned to exactly 0.500 by construction rather than
estimated. That representation is **built** in the first few transformer blocks,
**robust** to distance and to identifier renaming in the middle layers, and
**fragile** exactly where the underlying scope or control structure gets harder.
The same holds for a property an auditor would actually ask for: E15 reads
"is the value at this dangerous argument source-derived?" at 1.000 over *two*
measured floors in three models — and, applying each transformation on its own,
shows that **control-flow flattening alone destroys the readout** while opaque
predicates and arithmetic rewriting cost exactly nothing and composition adds no
measurable interaction. What survives flattening is each model's class prior, not
flow information. Whether the model **causally uses** any of this has its first
affirmative answer in E13, where a rank-1, magnitude-free interchange transports
*which definition is in scope* into both value assignments of a 2×2 — including
the one it was never fitted on. And E15-C establishes a boundary in the other
direction: mapped into the models' own output vocabulary, the security
distinction is **not there** — decodable is not verbalised.

---

## Status at a glance

| Exp | What it tests | 1.3B | 6.7B | SC2 | Status | Finding |
|---|---|:--:|:--:|:--:|---|---|
| **E2** binding | which definition an identifier refers to | ● | ● | ◑ | **foundation** | decodable from mid layers over a construction-pinned 0.500 floor |
| **E3** def–use | directed def→use edges | ● | ● | ◑ | **foundation** | decodable, mild distance decay, same floor |
| **E15** source→sink | which transformation breaks a frozen *security* readout, on its own? | ● | ● | ● | **foundation** | 1.000 over **two** chance floors; **flattening alone** causes the whole collapse; opaque predicates and MBA encoding cost exactly nothing |
| **E13** binding interchange | is the binding *causally* transported? | ☐ | ● | ☐ | **H0–H5 pass** | a rank-1 interchange installs the binding's value in both arms (100%/100%), beating a closed-form baseline at two-thirds the dose |
| **E5** context | robustness to distance vs interference | ● | ● | ☐ | supporting | survives 1000 tokens of filler; collapses under interference |
| **E9** obfuscation | the same transformations on binding and def–use | ● | ● | ● | supporting | **E15's companion control** — same boundary, so the failure is not security-specific |
| **E15-C** vocabulary contrast | is the safe→unsafe difference in the model's own output basis? | ● | ● | ● | supporting | **null in all three**, significantly *inverted* in 1.3B. Decodable ≠ verbalised |
| **E14** R-lens | is a more faithful backward pass available? | ● | ● | n/a | supporting | **gate R passes on both deepseek models** (ρ within 1e-4 at every layer; LRP beats autograd 7/7 and 9/9). The **gated-MLP rule dominates by 4.5×**, falsifying the plan's prediction that the LN-rule would. **Does not apply to starcoder2-3b** — LayerNorm + non-gated MLP, so the rules never install |
| E10-0 J-lens | instrument validation for the lens track | ● | ● | ● | supporting | V1 exact (cosine 1.0000) on all three; the Jacobian correction is real. On starcoder2-3b every required check passes (V2 top-1 0.633 vs 0.000 random), so that model's E15-C **J-lens** numbers have instrument validation behind them even though its R-lens does not exist |
| **E15-D** what the null is about | is E15-C's null about the models or the method? | ☐ | ☐ | ☐ | **built, not run** | stages 128–131, gates J2–J4. A full-vocabulary alignment measurement with no pool to blame, the **positive control**, and a relevance readout needing no lexicalisation. Nothing claimed |
| E4 control dep | guard→statement | ● | ● | ☐ | contrast only | decodable, but its surface floor is already 0.927 — the contrast that makes E2 mean something |

**Retired, parked and superseded** — E1, E6, E7, E8, E10-2, E10-3, E11, E12 — are
not listed above because none of them carries a claim this project stands on.
Their data, code and the reason each was retired are in `docs/ARCHIVE.md`; every
one is still runnable.

Legend: ☐ not run · ◑ partially run · ● run

---

# 1. Established: the representation exists and is built with depth

## E2 — variable binding

The claim rests on one control. A probe can score 100% on "are these two tokens
the same variable?" by reading the token strings, so every binding pair has a
**`context_matched`** partner: a second program that is token-identical except
the single character that flips the binding. The correct answer flips; nothing
observable about the text does.

| `context_matched` accuracy | 1.3B | 6.7B |
|---|---:|---:|
| surface baseline (token ids + distance, no model) | 0.500 | 0.500 |
| embedding layer (−1, token identity only) | 0.500 | 0.500 |
| block 0 (first transformer layer) | 0.570 | 0.531 |
| layer 3 | 0.961 | 0.914 |
| **peak (mid layers)** | **0.984** (L7) | **0.984** (L11–15) |
| last layer | 0.930 (L23) | 0.914 (L31) |

Three phases, each saying something different:

1. **Nothing at the input.** Both floors are *exactly* 0.500 — by construction,
   and confirmed in the data. The binding information is not in the tokens; it
   has to be built.
2. **Built in the first few blocks**, reaching ~0.91–0.96 by layer 3 and
   plateauing near 0.98 through the middle. That is early for a relation
   requiring scope resolution.
3. **Partly shed near the output** (~0.91–0.93), consistent with the final
   layers reorganising toward next-token prediction.

**Only `context_matched` is a clean headline.** The other strata sit at ~0.99
from block 0 because the token strings already separate them — the surface
baseline scores 0.78–0.94 on them too.

**Cross-scale.** The two models agree on shape and differ only where a scaling
account predicts: 6.7B does slightly less work in block 0 and holds its peak
longer — the same relative depth, stretched. The surface-baseline and
embedding rows are numerically identical across models, which they must be
since neither involves the model; that identity doubles as a corpus-integrity
check.

## E3 — def-use edges

Same design, same floors, same profile: peak ~0.99 at layers 7–11 with honest
decay by distance. The hardest bucket (50–200 tokens apart) holds at
**0.96–0.99** against ~0.99 for nearby pairs, so the model tracks def-use links
across real distance rather than adjacency.

---

# 2. Established: what the representation is made of

Both results below use **frozen** probes — fitted once on base programs, never
refitted on a variant — so a change in accuracy is a change in the model's
state, not in the probe.

## E5 — distance is cheap, interference is not

6.7B binding accuracy at 500 inserted filler tokens:

| filler | what it adds | acc | reading |
|---|---|---:|---|
| `comment_prose` | inert English | **0.921** | length is almost free |
| `dead_code` | unreachable statements | 0.794 | mild |
| `lexical_decoy` | similar-looking fresh names | 0.795 | mild |
| `competing_update` | rebinds *other* variables | 0.859 | moderate |
| `scope_shadow` | reuses the *tracked* names | **0.570** | **severe** |

At 1000 tokens `scope_shadow` reaches chance (0.498) while every other filler
stays above 0.70. Per layer: under `scope_shadow`, block 0 is the *most* stable
part of the network while the middle layers — the ones doing the binding work —
collapse. **The interference lands on the computation, not on a lookup.**

## E9 — renaming is survivable mid-layer; flattening is not

6.7B binding, best-layer accuracy per cumulative level:

| level | transform | best layer |
|---:|---|---:|
| 0 | normalize | ~1.000 |
| 1 | + rename every local | **0.897** (L11) |
| 2 | + opaque predicates | 0.857 |
| 3 | + MBA arithmetic | 0.846 |
| 4 | + control-flow flatten | **0.750** |

The layer breakdown is the finding: renaming pushes the *embedding and block-0*
probes **below chance** (0.29–0.33) — those layers keyed on identifier strings
and renaming actively misleads them — while mid layers 7–15 hold at 0.85–0.90.
Opaque predicates and rewritten arithmetic barely register, because they do not
change which definition reaches which use. Control-flow flattening is the true
limit.

**Together, E5 and E9 describe one failure surface.** The representation is
robust to how far apart things are and to what they are called, and it fails
when the scope or control structure it is a representation *of* becomes harder.
That is what one wants from a computed relation rather than a positional
heuristic — and it is a first, coarse map of when a tool built on these
representations should not be trusted.

---

# 3. The instrument track: what we can and cannot read with

## E4 — control dependence is decodable, but largely local syntax

| control_dep, best layer | positive recall | hard-negative recall |
|---|---:|---:|
| surface baseline (no model) | 0.959 | 0.676 |
| hidden — 1.3B (L11) | 0.981 | 0.873 |
| hidden — 6.7B (L15) | **0.995** | **0.923** |

The hidden state dominates on both classes at once, so the gap is not a
threshold artifact (aggregate AUC 0.990 → 0.999). **But the surface floor is
already 0.927**, unlike binding and def-use whose floor is pinned to exactly
0.500. A statement's guard is usually its nearest enclosing `if`.

This is reported as **the contrast that makes E2's isolation meaningful**, not
as a finding about representation. It is also the evidence that the project's
criterion for "semantic" excludes things.

## E10-0 — the J-lens implementation is correct

| check | 1.3B | 6.7B | reading |
|---|---:|---:|---|
| V1 — J-lens vs logit lens at the last layer | **1.0000** | **1.0000** | `J` is provably the identity there, so this must be 1.0 — a closed-form check of the whole gradient path |
| V2 — next-token top-1 (chance 0.038) | 0.633 | 0.650 | the lens reads real content |
| V2 advantage over the logit lens, pre-final | **+0.150** | **+0.183** | the Jacobian correction recovers content the logit lens cannot |

Instrument validation, not a result about the model. *Caveat:* V3 passed at
n=10, too small to carry weight; V1 and V2 are the load-bearing checks.

---

# 4. Causal use: answered for binding, open for flow

This is the project's centre of gravity and it is **not settled**. Four designs
have been attempted. The honest summary of each:

## E14 — the R-lens is more faithful, and the rule that matters is not the predicted one

Gate R passes on **both** DeepSeek models. Every required check, both models:

| check | 1.3B (float32) | 6.7B (float16) |
|---|---|---|
| **R0** forward invariance — the rules change no activation | 1.62e-06 relative (tol 1e-04) | 1.21e-03 relative (tol 1e-02) |
| **R1** last layer equals the logit lens | cosine **1.0000** | cosine **1.0000** |
| **R2** LRP beats raw autograd at every testable layer | **7/7** | **9/9** |
| **R2** conservation in early layers, median &#124;ρ−1&#124; | **0.0000** | **0.0001** |

The `all` arm holds `ρ ≈ 1` to within 1e-4 at *every* layer including the
embedding — so the estimator the R-lens rests on is sound on Llama-family
architectures, and E14's reference-architecture target reproduces on real models.

### The ablation replicates across models and dtypes

`docs/design/E14_RLENS_PLAN.md` §2.1 predicted the **LN-rule** would dominate;
a 1.3B fp16 run in August already recorded that prediction as half wrong. These
runs settle it, in a second model and a second dtype:

| rule removed | 1.3B | 6.7B |
|---|---:|---:|
| **`no_half`** (gated-MLP split) | **4.4203** | **4.4628** |
| `no_ln` (RMSNorm → diagonal) | 0.9806 | 0.9885 |
| `no_identity` (SiLU → elementwise) | 0.2265 | 0.3941 |
| `no_attn` (attention hooks) | 0.5128 | 0.3044 |

The **half-rule dominates by ~4.5×**, and the ordering is near-identical across
two models and two dtypes. What makes the traversed tail homogeneous is
overwhelmingly the gated-MLP split, not the norm; without it, the conservation
error is larger than the quantity being conserved.

One earlier anomaly also resolves. The August fp16 run reported the
*identity-rule making conservation worse*. In float32 the all-rules arm sits at
|ρ−1| = 0.0000 at every layer, against 0.2265 without the identity rule — so the
rule helps, and the earlier inversion looks like fp16 noise rather than a
property of SiLU. `no_identity` is the smallest of the four effects in both
models.

`no_ln` is the second-order effect and a total one — removing it drives `ρ` to
~0.01, i.e. the relevance essentially vanishes. Attention, deliberately left
unmodified, costs 0.30–0.51: real, bounded, and the honest answer to "what does
the unmodified softmax path cost".

### It does not apply to StarCoder2 at all

Gate R **cannot complete** on starcoder2-3b, and the reason is architectural
rather than numerical: StarCoder2 uses LayerNorm (deliberately unmatched — it
subtracts the mean, so the rule's algebra differs) and a non-gated MLP, so
`norm_eps_attr` and `is_gated_mlp` both decline and the two homogenising rules
bind to **nothing**. Stage 110 raises when its `no_attn` arm removes the only
rule that did bind.

The tell is in the one file it did produce: `rlens_r0_forward.csv` reports a
forward delta of **exactly 0.0**. Rules that are value-preserving still perturb
float arithmetic; rules that were never installed do not. An R0 that passes
*perfectly* is the signature of an empty install.

**Consequence for E15-C.** The starcoder2-3b artifact labelled `rlens` was built
with neither homogenising rule and is arithmetically a J-lens; its conservation of
0.154 is simply what raw autograd gives. J0 now refuses this case
(`rlens_rules_bound`). The E15-C null is unaffected — it rests on the logit and
J-lens results there, and on genuine R-lenses in both DeepSeek models — but
"three lenses agree" is, for that model, two lenses measured three ways.
## E13 — H0–H5 all pass (6.7B)

**Gates passed so far** (6.7B, 400 base programs):

| gate | result |
|---|---|
| **H0** generation and independent ground truth | **PASS** — 400/400 bases; all six invariant checks at 1.0000, including the arm crossing |
| **H1** the model returns the bound variable | **PASS** — 1.000 overall, 1.000 in the weakest cell |
| **H2** the binding is decodable at the use anchor | **PASS** — 1.000 against a measured surface floor of 0.500 |
| **H3** whole-state interchange flips the answer, per arm | **PASS** — ab +4.781 [+4.683, +4.878], ba +4.799 [+4.694, +4.903], flip rate 0.857; both structural zeros exactly 0.00e+00 |
| **H4** low-rank interchange beats matched controls on the training arm | **PASS** — +9.029 [+8.952, +9.108]; `das − random_norm` +8.126 [+8.020, +8.225], `das − random_rank` +9.033, `das − noop` +9.029 |
| **H5** the same subspace transfers to the held-out arm | **PASS** — 100.0% of held-out rows emit the installed answer, 114% of that arm's ceiling; the `answer_direction` control transfers at 0.154 against transport's 1.025 |

H1 at 1.000 and H2 at 1.000 are worth pausing on: with no arithmetic anywhere,
6.7B resolves these bindings perfectly, and which definition is in scope is
perfectly decodable at the use anchor against a floor pinned to 0.500. That is a
cleaner replication of E2's isolation than E2 itself, on a corpus built for
intervention.

**H4 and H5 both pass.** A rank-1, magnitude-free interchange at the use anchor
(layer 8), fitted on arm `ab` alone, makes the model emit the value the
*installed binding* selects on **100.0% of held-out rows in both arms** — 280
base programs, 560 rows per cell, cluster bootstrap over bases. The outcome is
the full-vocabulary argmax rather than the logit margin, because `delta_ld` is
positively biased at ceiling accuracy and any disruption inflates it.

| variant | `ab` emits installed | `ba` emits installed | edit fraction |
|---|---:|---:|---:|
| **`das_binding`** (rank 1, learned) | **100.0%** | **100.0%** | **0.479** |
| `whole_state` (the entire donor state) | 85.7% | 87.9% | 0.805 |
| **`mean_difference`** (rank 1, closed form) | 76.1% | 76.8% | 0.711 |
| `answer_direction` (J-lens, norm-matched) | 27.9% | 4.3% | 0.479 |
| `random_norm` (dose-matched random) | 2.1% | 1.8% | 0.513 |
| `random_rank` / `noop` / raw unembedding | 0.0% | 0.0% | 0.018 / 0 / 0.479 |

All 14 machinery checks pass: structural zeros exactly 0.00e+00, alignment
orthonormal to 4.07e-07, ceiling alive in both arms, and the model emits a
non-candidate token on 0.0% of rows.

**What this refutes, rather than merely fails to support.** A *disruption*
account has to explain why the dose-matched random subspace, which is
**over**-dosed at 0.538 of ‖h‖ against the treatment's 0.479, produces the
installed answer on 1.1% of rows against 100%. An *answer-direction* account has
to explain why the explicit answer direction attenuates 6.9× across the arms
while the treatment does not attenuate at all — and why it pushes the model
off-candidate on 9.1% of rows where the treatment never does. This is the
falsification E11 could not construct, because with arithmetic between the value
and the answer it had to forbid `answer == value` to avoid circularity.

**The closed-form baseline transports too — and loses.** The learned direction
sits at |cos| 0.673 from the mean donor−host difference, which is substantially
aligned but not identical, and no cosine can say whether the optimiser earned the
rest. So the difference-in-means direction was run as its own arm: no optimiser,
no labels, one fixed direction for every example. It works — **76.1% / 76.8%**,
transfer ratio 1.003, so a fixed direction does carry much of the binding, and
that was worth knowing.

But it does not explain the result. `das_binding` reaches **100% while moving
0.479 of ‖h‖; `mean_difference` reaches 76% while moving 0.711** — roughly twice
the effect per unit of dose. And the learned direction captures *less* of the raw
state difference (59.5% against 88.2%), so it is not a better-aligned version of
the same object. Whatever the optimiser found, a difference in means does not
recover it.

**What is still open.** A rank-1 edit outperforming the whole-state patch (100%
vs 86%) has a plausible explanation — the full patch installs the driving
component *and* components that fight it — that is **not** independently
demonstrated. And this is one site, one layer, one model, one construction.

---
## E15 — flattening alone breaks the security readout; the other transformations do not

Run at canonical scale on **three** models — `deepseek-coder-1.3b`,
`deepseek-coder-6.7b`, `starcoder2-3b`: 480 clean programs, 336 training / 144
held-out (72 bases) per condition, **ten conditions** (clean, normalize, four
atomic, four cumulative), 1776 programs extracted per model. **All six gates pass
in every run with no overrides.** Site `sink_arg`, cluster bootstrap over bases,
read at the layer nearest 48% of network depth (1.3B L11, 6.7B L15, starcoder2-3b
L15 — in all three also the argmax of clean-training CV).

**The property is there, and it is not the text.** On clean training programs the
local surface baseline (±3 token ids) is 0.488–0.491, the **whole-program lexical
baseline** (token n-grams and char 3–5-grams over the entire file) is 0.464, and
the embedding layer is 0.482 — all chance — while accuracy reaches **1.000**
(both deepseek models) and **0.997** (starcoder2-3b) near half depth. Held out,
across all ten conditions, the floors never leave chance: local surface
0.431–0.521, whole-program lexical 0.465–0.535, embedding 0.451–0.569.

### Flattening alone accounts for the entire collapse

| condition | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| clean held-out / normalize | **1.000** | **1.000** | **1.000** |
| `rename_only` | 0.938 | 0.986 | 0.882 |
| `opaque_only` | **1.000** | **1.000** | **1.000** |
| `encode_only` | **1.000** | **1.000** | **1.000** |
| **`flatten_only`** | **0.688** | **0.667** | **0.660** |
| full ladder (rename+opaque+encode+flatten) | 0.729 | 0.653 | 0.674 |

Three things follow, and this is the strongest result in the track:

1. **Two of the four transformations are free.** Opaque dead branches and
   mixed-boolean-arithmetic rewriting score *exactly* 1.000 on their own, in every
   model. They cost the readout nothing.
2. **Flattening alone costs 0.312 / 0.333 / 0.340**, within 0.03 of what the whole
   four-transformation composition costs.
3. **The interaction is null.** Cumulative minus atomic is +0.042 / −0.014 /
   +0.014, against a *measured* draw-noise floor of 0.021 / 0.035 / 0.028
   (`rename_only` vs `rename_cumulative` — identical transformation, independent
   draws). Composition adds nothing distinguishable from noise.

So the claim is now **"control-flow flattening breaks the readout"**, not "the
cumulative ladder does". The earlier marginal phrasing is retired on evidence.

**What survives flattening is class bias.** At `flatten_only`, unsafe/safe
accuracy is 0.625/0.750, **0.833/0.500** and 0.667/0.653, with 51% / 56% / 46% of
matched pairs receiving the same label. Under the full ladder the biases run in
*opposite* directions — 6.7B toward "unsafe" (0.861/0.444), starcoder2-3b toward
"safe" (0.569/0.778). A constant predictor of either class scores exactly 0.500
here, so residual accuracy that biases oppositely across models is each model's
own prior, not retained flow.

**The dangerous errors arrive before any structural change.** Under renaming
alone, starcoder2-3b is 0.882 pooled but **0.764 on unsafe against 1.000 on
safe** — the entire loss is false negatives, with the control flow untouched. By
structure, the **assignment chain is the fragile one under renaming** in all three
models (0.778 / 0.972 / 0.639) against `branch_merge` at 1.000 everywhere. By sink
family nothing reproduces, which is the null the design wanted: the readout tracks
flow, not which dangerous API sits at the end of it.

### E15-C — the difference is not in the model's own vocabulary

Mapping the same sink-site states into each model's vocabulary through three
readouts (logit lens, J-lens, R-lens; **R-lens declared primary in code before any
result**) returns **a null**:

| clean held-out, R-lens | 1.3B | 6.7B | starcoder2-3b |
|---|---:|---:|---:|
| concept token surviving the tokenizer | `" vulnerable"` | `" vulnerable"` | `" unsafe"` |
| held-out sign consistency | **0.153** | 0.403 | 0.694 |
| permutation p | 0.000 | 0.004 | 0.008 |
| verdict | **inverted** | stable non-security | stable non-security |

The security lexicon carries the contrast in no model, and the direction is not
consistent — 1.3B is significantly **inverted** (85% of pairs put *less*
unsafe-pole mass on the unsafe member).

**The obvious confound is ruled out.** A systematic difference in the *shape* of a
member's candidate distribution — its entropy, or the norm of its score vector —
would shift a z-scored contrast in a fixed direction with no concept involved.
Measured per pair, it does not: at the reported cells the contrast correlates with
the paired entropy difference at r = −0.29 / +0.16 / +0.14 and with the score-norm
difference at −0.04 / −0.10 / +0.10, no cell anywhere exceeds |r| = 0.39, and the
mean paired entropy difference is ≈ 0 (+0.0003 / +0.0045 / +0.0000 nats). At most
8% of the variance is distributional, so **1.3B's inverted sign is a real property
of the contrast** — an unexplained phenomenon rather than a measurement error.

Four further facts make this a real null rather
than a failed measurement: the three lenses **agree** (pairwise cosine 0.75–0.97);
it is not token identity (embedding-layer contrast null at p = 0.71–0.81, and 75%
of pairs share the same anchor token); something *does* replicate but is
semantically arbitrary (frozen discovered tokens reappear in the held-out top-k at
0.875 / 0.750 / 0.875 vs 0.000–0.031 for random controls — the tokens are `" ?"`,
`" liber"`, `"OrNull"`); and under flattening the vocabulary readout degrades
alongside the probe, so both trained and output-aligned auditability are lost
together.

**What this licenses:** linear decodability and expression in a model's own output
vocabulary are **different properties**, and E15 exhibits the first without the
second. It does not license any sentence containing "the model represents unsafe".

**What it does not yet license, and the correction that goes with it.** Every
control above is a *negative* control, and negative controls are silent about a
null: nothing here separates "the models do not verbalise this" from "this readout
could not detect verbalisation if it were there". Until a positive control has run,
the honest form of the claim is **"the security lexicon's contrast does not
replicate under this readout"**, not "the distinction is absent from output-aligned
coordinates".

One of those negative controls is also weaker than it was described as being.
`mismatched_pairs` redraws the *safe* partner from the same safe pool, so the label
difference survives it: the arm averages over the very set the main arm averages
over, its expected mean is the main arm's exactly, and measured over 200 redraws
the two agree to four decimal places on all three models
(−0.3041/−0.3041, −0.1998/−0.1998, +0.0940/+0.0940). It falsifies "specific to this
pairing", not "about the label" — and on 6.7B it is *more* sign-consistent than the
main arm (0.417 vs 0.403). Stage 126 now also runs a **same-label** arm, taking both
members from one pole, whose expected contrast is zero; stage 127 reports
`pairing_gain`, what base matching actually buys, as a number. **No result above
changes** — an uninterpretable check is replaced by an interpretable one.

Three stages now exist to close both gaps (E15-D, stages 128–131; **built,
smoke-tested, not yet run at canonical scale, nothing claimed**): a
full-vocabulary alignment measurement with no candidate pool to blame, the
**positive control** on the E6/E7 forced-choice taint property, and a relevance
readout that needs no lexicalisation at all. Design and the pre-declared outcome
table — including the one that would retire this track — are in
`docs/design/E15D_LENS_FOLLOWUPS_PLAN.md`.

### The boundary is general, not security-specific

The companion E9 run is now complete on all three models, which settles a question
that was open for a year. The same transformations degrade binding and def–use the
same way — binding 1.000 → 0.708–0.883 under renaming and 0.527–0.615 under
flattening; def–use → 0.689–0.864 and 0.402–0.545 — and the security readout is
**at least as robust** as the primitives it rests on (0.882–0.986 / 0.660–0.688).
The supported claim is *"structural obfuscation breaks frozen linear readouts of
program relations, security ones included"*, **not** "security representations are
specifically fragile".

**What it does not establish.** The floor is pinned only against *declared* feature
families; a reader that ran the taint analysis itself would score 1.0, so this is
an audit of a readout's transfer, not an E2-style construction-pinned claim. Eight
arms, not the full 15-combination lattice. "Flattening breaks the readout" is a
statement about a **frozen linear readout at one position** — a failing probe does
not prove the model lost the information, though §8.6's parallel failure is
consistent with real loss. The embedding control is one measurement, not three.
E15-C is observational, its candidate pool is logit-lens-selected, and the LRP
rules never install on starcoder2-3b at all (LayerNorm plus a non-gated
MLP), so its `rlens` artifact is arithmetically a J-lens and that model's E15-C
lens agreement is two lenses, not three. **E15-C has no positive control**, so its
null is not yet falsifiable in the direction that matters. Nothing causal is
claimed or tested for the security property. Full analysis, gates and limitations:
`docs/design/E15_SINKFLOW_PLAN.md` §8–§14.

# 5. What this project does not claim

- Not that code models "understand" programs. Every claim is a decoding or
  intervention result at named sites under named controls.
- Not that binding is causally used *in general*. E13 shows a rank-1 interchange
  transports the binding at one site, in one layer, in one model, on one
  synthetic construction. E7, E10, E11 and E12 each failed to establish even
  that, for a different recorded reason.
- Not that the E13 subspace is the *only* direction that transports. The
  closed-form difference-in-means direction reaches 76% in both arms; the
  learned one reaches 100% at two-thirds the dose. The claim is that it
  dominates that baseline, not that the baseline is inert.
- Not that the isolation transfers to real code. E8 shows the decoder
  transfers; the 0.500 floor exists only in synthetic programs.
- Not that control dependence is a semantic result — its floor is 0.927.
- Not that E11's readout-position effect passed. Both go/no-go files read NO-GO
  and the specificity check failed.
- Not that the 0.500 floor is pinned against *every* computable text feature. It
  is pinned against the stated surface baseline (±3 token ids plus bucketed
  distance). A cross-position string-equality baseline is outside that window
  and is an open item.

Withdrawn claims, with reasons: `docs/ARCHIVE.md`.

---

# 6. Open items

Ordered by what would most change what this project can claim.

1. **Explain, or bound, the rank-1 edit beating the whole-state patch**
   (100% vs 86% at 60% of the edit norm). The available account — the full patch
   installs components that fight the driving one — is plausible and untested. A
   reviewer will ask; better to answer it first.
2. **A second model and a second site for E13.** The causal result is currently
   one cell: `use`, layer 8, rank 1, 6.7B. 1.3b is cheap now that stage 106 runs
   in minutes.
3. **Explain the `assign_chain` fragility** (E15 §8.5). It has now replicated in
   three models under renaming *alone* — starcoder2-3b drops to 0.639 there while
   `branch_merge` stays at 1.000. Diagnose on the existing
   `sinkflow_predictions.csv` before spending any GPU.
4. **Make the R-lens architecture-general, or bound it.** It does not apply to
   starcoder2-3b at all: LayerNorm plus a non-gated MLP means neither
   homogenising rule installs. Extending `norm_eps_attr` to LayerNorm and
   `is_gated_mlp` to non-gated MLPs is the open work; the LayerNorm half is the
   harder one, since the mean-subtraction term is what the current algebra assumes
   away. See the lens roadmap in `docs/EXPERIMENTS.md`.
5. **Context-matched pairs on real code** — the highest-value follow-up for the
   foundation, and the one thing that would let E15's floor argument extend
   beyond synthetic programs. Build by mutating real functions.
6. **Fix `configs/models.yaml` ↔ `MODEL_REGISTRY`** so declared `probe_layers` are
   the ones that actually run. Repo-wide; it is why the three models sit on
   different layer grids and why every cross-model number has to be read at
   matched relative depth.
7. **A cross-position string-equality surface baseline** in stage 20. The current
   baseline cannot represent "the inner definition's name equals the use's name",
   which is the feature a lexical adversary would use. CPU-only.
