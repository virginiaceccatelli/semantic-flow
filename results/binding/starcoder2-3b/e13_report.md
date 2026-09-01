# E13 binding interchange — starcoder2-3b

**Verdict: INCOMPLETE**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 0.981 [0.973, 0.988] against 0.85; weakest cell ab_target 0.954 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 7: binding decodable at 1.000 (selectivity 0.519) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 11 (both chosen on calibration): ab: +1.729 [+1.671, +1.788], flip rate 0.659; ba: +1.762 [+1.704, +1.820], flip rate 0.662 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L11 r1: +7.512 [+7.437, +7.586] = 434% of the whole-state ceiling +1.731 (threshold 50%); controls cleared: True; edit moved 0.478 of ||h||
- **SUPERSEDED (was PASS)** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L11 r1: das_binding installed 100.0% = 149% of the held-out ceiling (threshold 50%); margin +7.528 [+7.454, +7.597] = 427% of it; discriminator — answer_direction ab +1.674 [+1.512, +1.827], installed 44.8% (passes: True); ba/ab argmax ratio 0.410 against transport's 0.979 (bar 0.490) (fails: True)
- **NOT RUN** `H6` (the relevance readout is mechanically sound: the LRP rules actually installed so relevance conserves, the token roles partition every token exactly once, the per-role deltas close to the difference of the two conservation ratios, every binding_flip pair differs at exactly one measured token index, the fixed-target conditions really do score both members at one token, and every declared cell exists) — not recorded
- **NOT RUN** `H7` (the candidate vocabulary is mechanically sound: every declared lexicon pair is kept whole or dropped whole with a reason, enough pairs survive from more than one family, discovery ran on calibration bases only and the frozen file records which, and the candidate set contains the lexicon, the discovered pool and the random controls without duplicates) — not recorded
- **NOT RUN** `H8` (the forced choice is mechanically sound: every declared (base, cell, question) is scored, the rendered question is identical in all four cells of a base, no question names the inner definition, both choices are distinct single tokens, both variants of every word style ran, and the value positive control ran) — not recorded
- **NOT RUN** `H9` (the verbalisation relevance readout is mechanically sound: H6's checks on the verbalisation prompt, plus a positive-score condition — `R_t / s` is a share only when the score is positive, and conservation holds for negative scores too) — not recorded
- **NOT RUN** `H10` (the unprompted vocabulary readout is mechanically sound: every declared lexicon pair kept whole or dropped whole with a reason, the candidate row order the margin arithmetic assumes, all three readouts carrying the same rows with their kinds declared and the random control actually Gram-matched, the scored text being E13's program verbatim with the encodings agreeing through the use position, the use token identical in all four cells, every declared cell present in both arms over the declared layer grid, and the positive control fitted on calibration and read on test) — not recorded

> **The `H5` line above is archived, not current.** H5's recorded verdict was decided by the ARCHIVED `answer_direction` control — the corpus-averaged cotangent readout stage 106 fitted for itself before 2026-09-01. That is a different estimator from the published J-lens (`docs/WORKSPACE_LENS.md` §1), so the discriminator behind this verdict no longer exists in the pipeline. The number is not translated, rescaled or reused: stage 106 must run again against the stage-201 J-lens artifact.

## Diagnostic

results/binding/starcoder2-3b/e16_*.csv — MECHANICAL, so a failure is an apparatus fault and never a negative result. Read the LRP rule counts first: on a LayerNorm model with a non-gated MLP (starcoder2) the homogenising rules bind to nothing, relevance does not conserve, and the readout is NOT APPLICABLE rather than failing. This is the ARCHIVED cotangent method, not the published R-lens — see docs/WORKSPACE_LENS.md §1.

Re-run after fixing: `python scripts/140_binding_relevance.py --model starcoder2-3b`

> **ARCHIVED.** H5's recorded verdict was decided by the ARCHIVED `answer_direction` control — the corpus-averaged cotangent readout stage 106 fitted for itself before 2026-09-01. That is a different estimator from the published J-lens (`docs/WORKSPACE_LENS.md` §1), so the discriminator behind this verdict no longer exists in the pipeline. The number is not translated, rescaled or reused: stage 106 must run again against the stage-201 J-lens artifact.

## Controls

No `interchange_panel.csv`. Stage 106 has not been re-run since the answer-direction control moved to the published J-lens (2026-09-01), so there is no current control table to show. The archived `answer_direction` numbers from earlier runs are **not** substituted here: that arm is a different estimator (`docs/WORKSPACE_LENS.md` §1). Re-run `make binding-interchange` after `make lens-fit`.

## Do not claim

- that the model 'understands' scope — the claim is transport at one site
- anything about real code, other languages, or other model families
- that H4 alone supports the conclusion; without H5 it is E11 again
- a null from H5 without checking that answer_direction_jlens failed on `ba` too
- that the R-lens arm gates anything — it is reported, never gated
- any number from the ARCHIVED `answer_direction` arm; it is a different estimator (docs/WORKSPACE_LENS.md §1) and is not comparable
