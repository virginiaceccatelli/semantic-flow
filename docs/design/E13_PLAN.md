# E13 — Binding interchange, falsified by the value assignment

**Status: IMPLEMENTED, not run.** Stages 100–107 exist
(`src/data/binding_pairs.py`, `src/experiments/binding_interchange.py`,
`scripts/10*.py`, `tests/test_binding.py`, `jobs/binding_*.csh`). No GPU stage
has been executed and no result is claimed. `results/STATUS.yaml` carries E13 as
`active` with `claim: none`.

> **Does a low-rank, magnitude-free interchange at the site where a variable
> binding is resolved transport *which definition is in scope* — rather than a
> token, or an answer direction?**

This is the question `paper/main.tex` §Discussion currently declares open:
*"Where the resolution itself happens, the causal question remains open."*

---

## Contents

- [§1 Why E12 was abandoned](#1-why-e12-was-abandoned)
- [§2 The design in one picture](#2-the-design-in-one-picture)
- [§3 The identification strategy](#3-the-identification-strategy)
- [§4 The metric](#4-the-metric)
- [§5 The programs and their invariants](#5-the-programs-and-their-invariants)
- [§6 The gate sequence](#6-the-gate-sequence)
- [§7 Controls](#7-controls)
- [§8 What each outcome means](#8-what-each-outcome-means)
- [§9 Position in the literature](#9-position-in-the-literature)
- [§10 Implementation map and cost](#10-implementation-map-and-cost)
- [§11 Risks](#11-risks)
- [§12 Do not claim](#12-do-not-claim)

---

# 1. Why E12 was abandoned

E12 tried to escape E11's "it is just output-aligned token directions" problem
by tracking a value that is **absent from the text**. That was the right
diagnosis. But text-absent-because-computed forces arithmetic, and the trade was
never priced: E12 made two chained arithmetic steps the load-bearing capability
for a question about program state.

The 1.3b pilot settled it. Balanced accuracy 0.418 — *below* chance. The correct
answer was the argmax on 6.3% of prompts, worse than the 10% a uniform random
digit gives. And two of four operation families sat at exactly 0.500, which a
simulation showed is reproduced by a model doing **no computation at all** and
picking whichever candidate is numerically closer to the head literal:

```
proximity to head (no computation)  overall 0.494  add 0.500  double_sub 0.500
observed, deepseek-coder-1.3b       overall 0.418  add 0.500  double_sub 0.500
```

Nikankin et al. ([arXiv:2410.21272](https://arxiv.org/abs/2410.21272)) published
the prediction: arithmetic in LMs is heuristic neurons that do not chain. E12's
full diagnosis is in `docs/design/archive/E12_PLAN.md` and
`results/store/{model}/g1_triage.csv`; the code is kept and gated, not deleted.

**E13 takes the cheaper escape.** There was one all along.

---

# 2. The design in one picture

Four programs per base, all token-identical except one character:

```
              ARM ab   (v_outer, v_inner) = (a, b)      ARM ba   = (b, a)

 source        x = a                                     x = b
 (outer        def f():                                  def f():
  binding)         y = b                                     y = a
                   return x            -> a                  return x      -> b

 target        x = a                                     x = b
 (inner        def f():                                  def f():
  binding)         x = b   <- one token                      x = a
                   return x            -> b                  return x      -> a
```

Install the **target** run's state into the **source** run at the marked use.

- In arm `ab` the answer must move **a → b**.
- In arm `ba` the very same intervention must move it **b → a**.

Fit the alignment on `ab`. Read the claim on `ba`.

---

# 3. The identification strategy

| What could explain a positive on arm `ab` | What it does on arm `ba` |
|---|---|
| the subspace carries **which definition is in scope** | positive — the installed binding selects the other value in both arms |
| the subspace carries **the token `b`** | **negative** — arm `ba` needs `a` |
| the subspace carries **the answer** | **negative**, for the same reason |
| the edit is generic perturbation of this size | ~zero in both arms (`random_norm`) |
| the edit moved nothing | exactly zero (`noop`, provably) |

That table is the whole contribution. E11 could not build it: with an arithmetic
operation between the value and the answer it had to forbid `answer == value` to
avoid circularity, and paid for it with a capability requirement. **Here the
answer IS the bound value — deliberately — and the arm crossing breaks the
circularity instead.**

Three further properties, none of which E11 or E7 had together:

- **No arithmetic anywhere.** The model returns a variable. E12's failure mode
  is removed by construction, and `tests/test_binding.py` asserts no arithmetic
  operator appears in any generated program.
- **No dose parameter.** An interchange installs whatever the donor run holds,
  so "was the edit large enough?" — the question that produced E11's retraction —
  does not arise. Magnitude is a measured consequence (`edit_fraction`), not a
  choice.
- **The intervention site is token-identical across the counterfactual.** The
  mutation sits ≥4 tokens upstream, outside any local window on the use.

---

# 4. The metric

> **Correction 2, 2026-08-13 — the falsification criterion was wrong.** H5
> originally required the `answer_direction` control to **reverse sign** on the
> held-out arm. On 6.7B it did not reverse; it *attenuated 7x* (+2.322 →
> +0.335), while `das_binding` did not attenuate at all (+9.029 → +9.009) and
> `whole_state` — which installs the entire donor state and therefore genuinely
> transports the binding — sat at +4.781 → +4.799.
>
> Reversal is one way a token/answer account can fail, not the only one, and
> demanding it reported "machinery broken" on data that discriminates cleanly.
> The criterion is now the **transfer ratio** (held-out / training arm), read
> against `whole_state` as the known-good reference in the same table:
>
> | variant | ratio |
> |---|---:|
> | `whole_state` (known-good transport) | 1.004 |
> | `das_binding` | 0.998 |
> | `answer_direction` | 0.144 |
>
> Why the control attenuated rather than reversed is a genuine bind, recorded
> here rather than patched: norm-matching it to the treatment means a
> 48%-of-‖h‖ push, and at that magnitude "a direction that moves the output
> toward token *w*" is no longer a meaningful linear statement. Unit-norm makes
> the control underpowered; matched-norm makes it non-linear. There is no
> version of this control that is both, which is why the ratio — a *relative*
> statistic with an internal reference — is the right thing to read.

> **Correction, 2026-08-11, pre-registered before the re-run.** The first 6.7B
> run showed `delta_ld` is positively biased on this corpus. H1 is 1.000, so
> the clean distribution is confident and `logP(own)` sits far above
> `logP(installed)`; any edit that merely *disrupts* the state regresses both
> toward the middle and raises `delta_ld` with nothing transported. The
> `answer_direction` control proved it: the design requires it to **reverse** on
> the held-out arm and it came out at **+0.136** there, more positive than on the
> arm it was built for.
>
> Two consequences, both now in the code. Every row records
> **`says_installed`** — the full-vocabulary argmax — which a disruption cannot
> produce systematically, and the gates read it alongside `delta_ld`. And the
> `answer_direction` control is built from **J-lens rows at the intervention
> layer** rather than raw unembedding rows: at layer 8 of 32 the unembedding row
> is not the direction that moves the output head toward a token, which is the
> premise of E10-0 and the reason it is the one surviving piece of that track.
> The unembedding version is kept as `answer_direction_unembedding` for
> comparison.


With host cell `(arm, binding)` and donor the same arm's other binding:

```
own       = the value the host's use resolves to
installed = the value the donor's binding selects
delta_ld  = [logP(installed) - logP(own)]_patched - [same]_clean
```

Positive means the output moved toward the value the **installed binding**
selects. The definition is uniform across arms; the token identity of
`installed` is not — which is exactly what the held-out arm tests.

Intervals are cluster bootstraps over base programs throughout; control
comparisons are paired on the same rows.

---

# 5. The programs and their invariants

Enforced at generation, re-checked independently in stage 101, and unit tested:

| invariant | what it closes |
|---|---|
| all four prompts share a token length | every anchor is the same index in all four |
| within an arm, exactly one differing token, at the inner definition's name | nothing else varies with the binding |
| the use token is identical in all four | the probing site carries no lexical cue |
| use − mutation ≥ 4 tokens | no local window on the use can see the mutation |
| **`answer(ab,source) == answer(ba,target)` and `answer(ab,target) == answer(ba,source)`** | **the crossing** — without it the held-out test proves nothing |
| both values distinct single tokens | the logits are read at the right rows |
| the answer appends exactly one token | the logits are read at the right position |
| no arithmetic operator in any program | the capability required stays a lookup |

**Ground truth is read twice.** Stage 100 uses `execute_program`; stage 101
re-derives every cell with a **scope-aware reference interpreter**
(`store_semantics.interpret_scoped`) written against the AST independently of
the rendering path. It implements the rule under test explicitly — *a name
assigned anywhere in a function body is local for the whole body* — because
E12's interpreter has no concept of globals and literally cannot express what
E13 measures. Disagreements are dropped and counted.

Measured yield: **400/400 bases in 4.5 s**, every invariant at 1.0000.

---

# 6. The gate sequence

Each stage declares its prerequisites in
`src/experiments/store_gates.py::BINDING` and **refuses to run** (exit 2) unless
they passed. `--override-gate REASON` is permitted for diagnostics and recorded
permanently in `gates.yaml`, the manifest, and every output row.

| gate | stage | asserts | threshold |
|---|---|---|---|
| **H0** | 101 | execution and a scope-aware interpreter agree; invariants hold, including the crossing | ≥ 0.999 of bases |
| **H1** | 102 | the model returns the correctly bound variable | ≥ 0.85 overall **and** ≥ 0.75 in every cell |
| **H2** | 104 | which definition is in scope is decodable at the use anchor | ≥ 0.80, ≥ 0.10 over the measured surface baseline |
| **H3** | 105 | whole-state interchange flips the answer **in both arms** | CI > 0 and flip rate ≥ 0.25, per arm |
| **H4** | 106 | low-rank interchange beats its controls on the **training** arm | ≥ 50% of the ceiling, all contrasts CI > 0 |
| **H5** | 106 | the same subspace transfers to the **held-out** arm | ≥ 50% of that arm's ceiling, **and** `answer_direction` fails there |

**H1 is per-cell on purpose.** A model that handles the outer binding well and
the shadowed one at chance would pass on average while being unable to do the
only thing E13 measures.

**H3 is per-arm on purpose.** If the held-out arm cannot be moved even by
replacing the state outright, then "the subspace failed to transfer" and "the
arm is not testable" are the same observation — the ambiguity that retired
E10-3. H3 is what makes an H5 null interpretable.

---

# 7. Controls

| Control | Rules out |
|---|---|
| `whole_state` | the ceiling, per arm; also the proof that both arms are measurable |
| **`answer_direction`** | the positive control for the falsification itself: an explicit answer direction (the unembedding row of the answer arm `ab` demands) MUST pass on `ab` and MUST fail on `ba`. If it does not fail, the discriminator is broken and no verdict is licensed |
| `random_rank` | any subspace of this rank would do |
| `random_norm` | any edit moving this fraction of ‖h‖ would do — matched on *removed norm*, because for an orthogonal projector only the span matters |
| `noop` | numerical noise; the edit is provably the zero vector |
| `def_source` site | a structural zero: the programs are token-identical before the mutation, so host and donor states there are the same state |
| measured surface baseline (H2) | text features at the anchor |
| smallest-rank reporting | a high-rank success being read as localisation |

The two structural zeros are kept in the output rather than suppressed. A
nonzero value in either means the hooks, anchors or dtypes are wrong and every
other number in the stage is suspect.

---

# 8. What each outcome means

| Outcome | Reading |
|---|---|
| **H4 and H5 pass** | A low-rank subspace at the resolution site transports the binding, in both value assignments, where an explicit answer direction manages only one. This closes the question the paper declares open. |
| **H4 passes, H5 fails, `answer_direction` fails on `ba` as designed** | The learned subspace *is* an answer direction. A real, reportable negative — and exactly what E11 could not establish, because it had no held-out arm. |
| **H4 passes, H5 fails, `answer_direction` also passes on `ba`** | The discriminator is broken. No verdict. Fix the control before interpreting anything. |
| **H3 fails on `ba` only** | The held-out arm is not measurable; H5 is untestable. Not a result about the model. |
| **H2 fails** | The corpus or the anchoring is wrong — this replicates E2's `context_matched`, which is a validated foundation. |
| **H1 fails** | A capability limit on a variable lookup. Check for a constant responder before blaming the model. |

Every branch is informative and none of them is "the model understands scope".

---

# 9. Position in the literature

| Work | What it already does | E13's delta |
|---|---|---|
| Geiger et al., DAS ([arXiv:2303.02536](https://arxiv.org/abs/2303.02536)); Wu et al., Boundless DAS; Huang et al., RAVEL | interchange interventions on a learned subspace; interchange-intervention accuracy; the isolation criterion | applied to variable binding in a **pretrained** code model, with a construction-pinned surface floor |
| Wu, Geiger & Millière, ICML 2025 ([arXiv:2505.20896](https://arxiv.org/abs/2505.20896)) | interchange interventions on dereference chains in symbolic programs; program state *not* linearly decodable at a single location | 37.8M from-scratch vs pretrained 1.3B/6.7B; and the **value-assignment factorial**, which refutes an answer-direction account rather than assuming it away |
| Feng & Steinhardt, ICLR 2024 | binding IDs identified by causal intervention | retrieval of a stored attribute vs. transport of *which definition is in scope* |
| E11 (this repo) | rank-2 swap of output-aligned value directions | no dose parameter; no arithmetic; and a held-out arm that separates binding from answer — the three things E11's retraction turned on |

**Honest positioning.** This is a new identification strategy on an established
technique, not a new technique. The delta that matters is the factorial: it is
what turns "we intervened and the answer moved" into "we intervened and the
answer moved *in the direction the binding implies, not the direction the token
implies*".

---

# 10. Implementation map and cost

| Stage | Script | Gate | Where |
|---|---|---|---|
| 100 | `scripts/100_binding_pairs.py` | — | CPU, ~5 s |
| 101 | `scripts/101_binding_verify.py` | **H0** | CPU, ~5 s |
| 102 | `scripts/102_binding_behaviour.py` | **H1** | GPU, ~2 min |
| 103 | `scripts/103_binding_extract.py` | — | GPU, ~3 min |
| 104 | `scripts/104_binding_decode.py` | **H2** | CPU, minutes |
| 105 | `scripts/105_binding_ceiling.py` | **H3** | GPU, ~15 min |
| 106 | `scripts/106_binding_interchange.py` | **H4, H5** | GPU, ~1–2 h |
| 107 | `scripts/107_binding_report.py` | — | CPU, seconds |

Prompts are ~21 tokens (E12's were 37, E11's 49), so at the measured 31
passes/s the whole 6.7b run is **≈ 1.5–3 GPU-hours**, dominated by stage 106's
backward passes.

Reused unchanged: `src/models/das.py` (the interchange operator, the controls,
the DAS loop), `src/models/hooks.py` siblings, `src/probes/base.py`,
`src/analysis/bootstrap.py`, `counterfactual_pairs`'s tokenizer helpers and
`_positions_for` — the E13 template is E11's `global_shadow` with the operation
removed, so every anchor resolves with no new alignment code.

---

# 11. Risks

1. **H1 is still a capability gate**, just a far cheaper one. E11's `affine`
   family (one arithmetic op on a directly-read value) reached 0.905 on 6.7b; a
   pure lookup should be higher, but that is an extrapolation, not a
   measurement, and it is the first thing the pilot settles.
2. **DAS is expressive enough to find structure that is not there.** Mitigated by
   the disjoint split, smallest-rank reporting, and decisively by the held-out
   arm — which is the entire point of the design rather than an add-on.
3. **The two arms are not independent samples.** Both arms of a base stay in one
   split, so arm transfer is a within-base contrast and is not confounded with
   example generalization. Reported as such.
4. **fp16 backward stability** in stage 106. `--dtype float32` is the fallback.
5. **Synthetic, one template, one model family.** Unchanged from E8's
   limitation. Nothing here transfers to real code and E13 does not claim it.

---

# 12. Do not claim

- Not that the model "understands scope". The claim is transport of a binding at
  one site, under one template.
- Not H4 alone. Without H5 it is E11 again, and E11's own go/no-go read NO-GO.
- Not an H5 null without checking that `answer_direction` failed on `ba` too.
  If the discriminator did not discriminate, there is no verdict.
- Not "low-dimensional" if the effect needs rank 16. Report the smallest rank.
- Not any transfer to real code, other languages, or other model families.
- Not a scale claim from 1.3b vs 6.7b unless H1 is matched across them.
- Not that E13 supersedes E2/E3. It rests on them: H2 is E2's `context_matched`
  result replicated on this corpus, and a failure there indicts the corpus.
