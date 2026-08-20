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
the one it was never fitted on. And the lens track locates that
representation in the models' **own output coordinates**: no word for it exists —
E15-C's security lexicon carries nothing — but a direction defined by the label
generalises to held-out programs in **72/72 pairs on every model**, appears a
quarter of the way up the stack, and collapses under flattening exactly as the
probe does. The distinction is output-aligned and **distributed**, not
lexicalised. Decodable is not verbalised, and the gap is not a gap in the
instrument.

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
| **E15-C** security lexicon | does a *word* for the distinction carry it? | ● | ● | ● | supporting | **no, in all three**, significantly *inverted* in 1.3B. The same-label control collapses the contrast to ≈0, so the inversion is genuinely about the label |
| **E14** R-lens | is a more faithful backward pass available? | ● | ● | n/a | supporting | **gate R passes on both deepseek models** (ρ within 1e-4 at every layer; LRP beats autograd 7/7 and 9/9). The **gated-MLP rule dominates by 4.5×**, falsifying the plan's prediction that the LN-rule would. **Does not apply to starcoder2-3b** — LayerNorm + non-gated MLP, so the rules never install |
| E10-0 J-lens | instrument validation for the lens track | ● | ● | ● | supporting | V1 exact (cosine 1.0000) on all three; the Jacobian correction is real. On starcoder2-3b every required check passes (V2 top-1 0.633 vs 0.000 random), so that model's E15-C **J-lens** numbers have instrument validation behind them even though its R-lens does not exist |
| **E15-D** full-vocabulary direction | is the distinction in output coordinates at all? | ● | ● | ● | **foundation** | **yes, and it is not a word.** 72/72 held-out pairs project positively on a label-defined direction (cosine 0.38) over a token-identity floor of *exactly zero*; onset at ~25% depth; collapses under flattening alone. Top loadings are meaningless fragments |
| **E15-D** relevance routing | does the model route identical text differently? | ● | ☐ | n/a | supporting | the chain feeding the sink **loses** relevance share and the other gains, 63–65 of 72 pairs at layers 0–3 (sign p ≤ 4e-11), on token-identical spans; survives role- and order-swap strata. Small (1–2% of the answer), one model |
| **E15-D** positive control | could this readout detect verbalisation at all? | ☐ | ☐ | ☐ | **not run** | the one measurement that would fix what E15-C's *null* means. Built; `scripts/129_sinkflow_positive.py` |
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

## The lens argument in one place

Four experiments read the residual stream through the model's **own output
vocabulary** rather than through a fitted probe. They are easy to conflate, so
here is the whole argument in five sentences, each with the experiment that
establishes it.

1. **The instrument is sound.** The J-lens reproduces the logit lens exactly at
   the last layer and recovers real next-token content before it, on all three
   models. *(E10-0, stage 60.)*
2. **A more faithful backward pass exists, and it is architecture-dependent.**
   Under LRP rules the relevance decomposition conserves to 1e-4 at every layer
   on both DeepSeek models; on StarCoder2 the rules bind to nothing, so there is
   no R-lens there at all. *(E14, stage 110.)*
3. **Reading the state through a hand-picked security vocabulary finds
   nothing.** Null on all three models, and not specific to the lens. *(E15-C,
   stages 125–127.)*
4. **Reading it through the *whole* vocabulary finds something — but it is not a
   word.** A direction defined by the safe/unsafe label generalises to held-out
   programs in 72/72 pairs on every model, above a token-identity floor that is
   exactly zero; its top-loading tokens are meaningless fragments. *(E15-D V1,
   stage 128.)*
5. **And the model routes its answer differently through token-identical
   text.** Whichever data-flow chain feeds the sink loses relevance share and
   the other gains, in 85–90% of matched pairs at early layers. *(E15-D V3,
   stage 130, one model.)*

The one thing not established: whether this readout could detect verbalisation
if it were there. The positive control that would settle it is built and has not
been run — see §"E15-D" and `docs/design/E15D_LENS_FOLLOWUPS_PLAN.md`.

**The headline the track supports:** *the safe/unsafe distinction is present in
output-aligned coordinates, distributed across the vocabulary, and not carried by
any word for it.* "Decodable" and "verbalised" come apart, and the gap is not a
gap in the instrument.

## E10-0 — the J-lens implementation is correct

| check | 1.3B | 6.7B | SC2-3B | reading |
|---|---:|---:|---:|---|
| V1 — J-lens vs logit lens at the last layer | **1.0000** | **1.0000** | **1.0000** | `J` is provably the identity there, so this must be 1.0 — a closed-form check of the whole gradient path |
| V2 — next-token top-1 (chance 0.038) | 0.633 | 0.650 | 0.633 | the lens reads real content |
| V2 advantage over the logit lens, pre-final | **+0.150** | **+0.183** | +0.217 | the Jacobian correction recovers content the logit lens cannot |

Instrument validation, not a result about the model — and it now covers all
three models, which matters because StarCoder2 has no R-lens: **its E15-C and
E15-D lens numbers rest on this validation and nothing else.** *Caveat:* V3
passed at n=10, too small to carry weight; V1 and V2 are load-bearing.

## E14 — the R-lens is more faithful, and the rule that matters is not the predicted one

*(Instrument work. It lives here rather than under "causal use" because nothing in it is causal.)*

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

---

# 4. Causal use: answered for binding, open for flow

This is the project's centre of gravity and it is **not settled**. Four designs
have been attempted. The honest summary of each:

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

### E15-C — no *word* carries the difference

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

Three stages exist to close both gaps, and two of them have now run — see
**E15-D** below.

### E15-D — the distinction *is* in the output basis; it is just not a word

E15-C looked for the safe→unsafe difference in a **hand-picked 196-token
vocabulary** and found nothing. Stage 128 removes the vocabulary: it forms each
matched pair's difference over **all 32k tokens**, estimates the mean direction
on the training split, and projects held-out pairs onto it. Nothing is chosen in
advance, so a null cannot be blamed on a pool.

It is not a null.

| clean held-out, `last_token`, mid-depth | 1.3B (L11) | 6.7B (L15) | SC2-3B (L15) |
|---|---:|---:|---:|
| held-out pairs projecting positively | **72/72** | **72/72** | **72/72** |
| mean cosine with the frozen direction | **0.383** | **0.380** | **0.390** |
| cluster-bootstrap 95% CI | [0.360, 0.406] | [0.358, 0.401] | [0.351, 0.429] |
| same-label control, same direction | 0.507 | 0.465 | 0.549 |
| token-identity floor (layer −1) | **exactly 0** | **exactly 0** | **exactly 0** |

Every number that could make this an artifact was measured and came back clean.
The floor is *exactly* zero because at `last_token` both members carry the same
token id, so their embeddings are identical and all 72 differences vanish. A
random direction in 32 256 dimensions would give a cosine near 0.006, so 0.38 is
roughly sixty standard deviations out. The same-label control — two programs of
the *same* label, everything else varying — sits at chance on the same direction.
And on the **44 of 72 pairs whose two members have identical token counts**, the
last-token position index is identical too, so the 100% sign consistency cannot
be a length or position effect: you cannot be right on all 72 without being right
on those 44.

**It has a clear onset depth, identical in all three models.**

| relative depth | ≈0.0 | ≈0.13 | ≈0.25 | ≈0.35–0.50 | ≈0.75–1.0 |
|---|---:|---:|---:|---:|---:|
| layers | −1, 0 | 3 | 7 | 11–15 | 19–31 |
| sign consistency (1.3B / 6.7B / SC2) | 0.49/0.46/0.46 | 0.46/0.49/0.50 | **1.00**/0.79/0.71 | 1.00/1.00/1.00 | 0.90–1.00 |

Nothing before a quarter of the way through the stack; then it appears and holds
to the output. Two model families, three scales, one curve.

**And it degrades exactly as the probe does.** Projection sign consistency at the
reported layer, across the ten conditions:

| | clean | normalize | rename | opaque | encode | **flatten** | full cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.3B | 1.000 | 1.000 | 0.958 | 0.903 | 1.000 | **0.819** | 0.722 |
| 6.7B | 1.000 | 1.000 | 0.986 | 0.917 | 1.000 | **0.708** | 0.625 |
| SC2-3B | 1.000 | 0.944 | 0.889 | 0.903 | 0.972 | **0.681** | 0.681 |

The mean projection falls by ~93% under flattening alone and by 4–14% under
renaming, opaque predicates or MBA encoding. **This independently replicates E15's
headline with a completely different readout** — nothing is fitted here, no probe
is trained, and the conclusion is the same: flattening alone accounts for the
collapse.

**But the direction is not a concept, and it does not dominate.** Two facts keep
this from being "the model has an unsafe feature":

* *Its top loadings are meaningless.* The tokens carrying it are `' Lemmon'`,
  `'egraphics'`, `'idir'`, `'女儿'` (1.3B); `' mel'`, `'椒'`, `' Jonathan'`
  (6.7B); `'bootstrapcdn'`, `'%%%%%%%%%%'`, `'pmatrix'` (SC2). Loadings are flat
  — 0.019 to 0.027 across the top twelve — i.e. the direction is spread thinly
  over thousands of tokens rather than concentrated in a few. It is
  **output-aligned but not lexicalised**.
* *The label axis is not the largest axis of variation.* The pre-declared
  criterion was `sv1_ratio ≥ 2.0` — the pairs' differences must concentrate on
  one direction at least twice as much as *same-label* differences do. Measured:
  **0.76 / 0.97 / 0.76**. It **failed on all three models**, and it fails because
  two programs of the same label already differ along a dominant shared axis of
  comparable size. So the verdict is `direction_replicates_but_not_dominant`.

Those two statements are compatible and both are reported. The projection asks
*does a label-defined direction generalise to unseen programs* — yes, decisively.
The concentration asks *is the label axis the biggest thing separating these
difference vectors* — no. What settles that they are different axes rather than
one: the frozen direction's top-100 loadings overlap what the **same-label**
differences find by a Jaccard index of **0.005 / 0.005 / 0.000**. Nearly
disjoint.

### E15-D — relevance moves, on text that is character-for-character identical

Stage 130 abandons the vocabulary entirely. Under the LRP rules the relevance
decomposition conserves — measured here at median |ρ − 1| = **0.0000** and a
worst case of 5e-5 at every layer — so `R_t / s` is a genuine *partition* of the
model's own answer across input positions, and a paired difference is a
redistribution rather than a change of scale. Relevance is summed by AST role,
recomputed from each program's own source.

The control is free and it is the strongest one in the project: **only
`sink_arg` differs in tokens between the two members.** Measured under the real
tokenizer, every other role matches at 1.000 and `sink_arg` at 0.611 — exactly
the 44/72 length-matched pairs. So a shift among the other roles is the model
routing its answer differently through *identical text*.

It shifts. Paired difference in each role's share, deepseek-coder-1.3b, clean
held-out, 72 pairs:

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
depth, and is gone for `taint_chain` by layer 11 — while `trust_chain` survives
to layer 19.

It survives both available controls: it holds separately in both `role_swap`
strata (which identifier name carries the taint) and both `order_swap` strata
(which chain is written first), at p ≤ 2e-2 in every one of the eight cells at
layers 0 and 3. So it is neither an identifier-name effect nor a
position-in-program effect.

**Two honest qualifications.** The magnitude is small — a median shift of 1–2% of
the answer. And the pre-declared `above_permutation_control` check, which tests
the *mean*, **fails** (p = 0.39–0.62): relevance deltas are heavy-tailed enough
that seven outlier pairs out of seventy-two flip the mean's sign while the median
and the sign stay put. The statistic that survives is the sign, and its exact null
under the same random-orientation scheme is binomial — which is the test reported
above. The verdict records both: `redistribution_consistent_but_not_in_mean`.

This ran on **deepseek-coder-1.3b only**. It is *not applicable* to StarCoder2 —
LayerNorm plus a non-gated MLP means the homogenising rules bind to nothing, so
there is no conservation to read, and stage 130 refuses rather than emitting raw
autograd under the name relevance. It has not been run on 6.7B.

### E15-D — what has not run, and what it would settle

**The positive control (stage 129) has not been run on any model.** It is the one
measurement that separates

> the models do not verbalise this

from

> this readout could not detect verbalisation if it were there,

and until it runs, the E15-C null keeps that ambiguity. Every control in E15-C is
*negative*, and negative controls establish that a positive result is not an
artifact — they are silent about a null. Note that this does **not** touch the
E15-D results above, which are positive findings and stand on their own controls;
it limits only what may be concluded from E15-C's *absence* of a security
vocabulary.

The four outcomes and what each licenses are written down in advance in
`docs/design/E15D_LENS_FOLLOWUPS_PLAN.md` §4, including the one that would retire
the E15-C track. Run it with
`python scripts/129_sinkflow_positive.py --model M --conditions all`.

### E15-C's mismatched-pair control, corrected

Building E15-D turned up a defect in E15-C's own controls. `mismatched_pairs`
redraws the **safe** partner from the same safe pool, so the label difference
survives it: the arm averages over the very set the main arm averages over, and
its expected mean is the main arm's exactly. On the canonical runs the two agree
to four decimal places, and on 6.7B the *control* is more sign-consistent than the
main arm (0.417 vs 0.403). It falsifies "specific to this pairing" and nothing
more.

Stage 126 now also runs a **same-label** arm — both members from one pole, so the
expected contrast is zero — and it behaves as it should, which retroactively
strengthens E15-C rather than weakening it:

| clean held-out, R-lens, reported cell | 1.3B | 6.7B | SC2-3B |
|---|---:|---:|---:|
| main arm, mean Δ contrast (z) | −0.304 | −0.200 | +0.094 |
| same-label (unsafe pole) | +0.004 | −0.031 | +0.028 |
| same-label (safe pole) | +0.015 | +0.050 | −0.004 |
| `pairing_gain` (what base matching buys) | 0.014 | 0.014 | 0.056 |

The same-label control collapses the contrast to nothing on every model. So
**1.3B's inverted contrast is genuinely about the label** — it is not a
program-to-program artifact. No E15-C number changed; an uninterpretable check
was replaced by an interpretable one.

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
rules never install on starcoder2-3b at all (LayerNorm plus a non-gated MLP), so
its `rlens` artifact is arithmetically a J-lens and that model's lens agreement
is two lenses, not three. **The positive control has not run**, so E15-C's *null*
is not yet falsifiable in the direction that matters — which does not touch
E15-D's positive results. E15-D V1 is observational too: a direction that
generalises is a statement about format, not about use, and its top loadings
being uninterpretable is a fact to report rather than a puzzle solved. E15-D V3
ran on one model and its magnitude is small. Nothing causal is
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

0. **Run the positive control** (`scripts/129_sinkflow_positive.py`, ~1–3 h per
   model on GPU). It is built, gated and unrun, and it is the only thing standing
   between E15-C's null and a claim about what code models verbalise. Every other
   item below is smaller. Run it on all three models with `--conditions all`.
1. **Replicate E15-D V3 on 6.7B.** The relevance redistribution is one model.
   `scripts/130_sinkflow_relevance.py --model deepseek-coder-6.7b`, ~30–90 min.
   It is *not applicable* to StarCoder2, so 6.7B is the whole replication.
2. **Explain, or bound, the rank-1 edit beating the whole-state patch**
   (100% vs 86% at 60% of the edit norm). The available account — the full patch
   installs components that fight the driving one — is plausible and untested. A
   reviewer will ask; better to answer it first.
3. **A second model and a second site for E13.** The causal result is currently
   one cell: `use`, layer 8, rank 1, 6.7B. 1.3b is cheap now that stage 106 runs
   in minutes.
4. **Explain the `assign_chain` fragility** (E15 §8.5). It has now replicated in
   three models under renaming *alone* — starcoder2-3b drops to 0.639 there while
   `branch_merge` stays at 1.000. Diagnose on the existing
   `sinkflow_predictions.csv` before spending any GPU.
5. **Make the R-lens architecture-general, or bound it.** It does not apply to
   starcoder2-3b at all: LayerNorm plus a non-gated MLP means neither
   homogenising rule installs. Extending `norm_eps_attr` to LayerNorm and
   `is_gated_mlp` to non-gated MLPs is the open work; the LayerNorm half is the
   harder one, since the mean-subtraction term is what the current algebra assumes
   away. See the lens roadmap in `docs/EXPERIMENTS.md`.
6. **Context-matched pairs on real code** — the highest-value follow-up for the
   foundation, and the one thing that would let E15's floor argument extend
   beyond synthetic programs. Build by mutating real functions.
7. **Fix `configs/models.yaml` ↔ `MODEL_REGISTRY`** so declared `probe_layers` are
   the ones that actually run. Repo-wide; it is why the three models sit on
   different layer grids and why every cross-model number has to be read at
   matched relative depth.
8. **A cross-position string-equality surface baseline** in stage 20. The current
   baseline cannot represent "the inner definition's name equals the use's name",
   which is the feature a lexical adversary would use. CPU-only.
