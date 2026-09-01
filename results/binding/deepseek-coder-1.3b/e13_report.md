# E13 binding interchange — deepseek-coder-1.3b

**Verdict: INCOMPLETE**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 200 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **FAIL** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 0.809 [0.779, 0.838] against 0.85; weakest cell ab_target 0.571 against 0.75
- **NOT RUN** `H2` (which definition is in scope is decodable at the use anchor) — not recorded
- **NOT RUN** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — not recorded
- **NOT RUN** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — not recorded
- **NOT RUN** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — not recorded
- **PASS** `H6` (the relevance readout is mechanically sound: the LRP rules actually installed so relevance conserves, the token roles partition every token exactly once, the per-role deltas close to the difference of the two conservation ratios, every binding_flip pair differs at exactly one measured token index, the fixed-target conditions really do score both members at one token, and every declared cell exists) — 9600 readings and 19200 paired contrasts over 4 contrasts x 6 layers x 4 target conditions; median |rho-1| 1.56e-07; conserving layers [0, 3, 7, 11, 15, 19]; LRP rules bound {'ln': 49, 'mlp': 24, 'attn': 24}
- **PASS** `H7` (the candidate vocabulary is mechanically sound: every declared lexicon pair is kept whole or dropped whole with a reason, enough pairs survive from more than one family, discovery ran on calibration bases only and the frozen file records which, and the candidate set contains the lexicon, the discovered pool and the random controls without duplicates) — 10 lexicon pairs from 4 families, 15 mechanism words, 225 candidates (160 discovered, 32 random) over 60 calib bases
- **PASS** `H8` (the forced choice is mechanically sound: every declared (base, cell, question) is scored, the rendered question is identical in all four cells of a base, no question names the inner definition, both choices are distinct single tokens, both variants of every word style ran, and the value positive control ran) — 7200 scored choices over 9 questions x 200 bases x 4 cells; report split test
- **PASS** `H9` (the verbalisation relevance readout is mechanically sound: H6's checks on the verbalisation prompt, plus a positive-score condition — `R_t / s` is a share only when the score is positive, and conservation holds for negative scores too) — 14400 readings and 24000 paired contrasts over 4 contrasts x 6 layers x 5 target conditions for scope/direct; median |rho-1| 3.03e-07; readable layers [0, 3, 7, 11, 15, 19]; LRP rules bound {'ln': 49, 'mlp': 24, 'attn': 24}
- **NOT RUN** `H10` (the unprompted vocabulary readout is mechanically sound: every declared lexicon pair kept whole or dropped whole with a reason, the candidate row order the margin arithmetic assumes, all three readouts carrying the same rows with their kinds declared and the random control actually Gram-matched, the scored text being E13's program verbatim with the encodings agreeing through the use position, the use token identical in all four cells, every declared cell present in both arms over the declared layer grid, and the positive control fitted on calibration and read on test) — not recorded

## Diagnostic

The task is a variable lookup with no arithmetic. If this fails, check behaviour.csv for a constant responder (group by argmax_token) before blaming the model, then check the weakest CELL — a model that handles the outer binding but not the shadowed one fails the only thing E13 measures.

Re-run after fixing: `python scripts/103_binding_extract.py --model deepseek-coder-1.3b --layers <layers>`

## Controls

No `interchange_panel.csv`. Stage 106 has not been re-run since the answer-direction control moved to the published J-lens (2026-09-01), so there is no current control table to show. The archived `answer_direction` numbers from earlier runs are **not** substituted here: that arm is a different estimator (`docs/WORKSPACE_LENS.md` §1). Re-run `make binding-interchange` after `make lens-fit`.

## Do not claim

- that the model 'understands' scope — the claim is transport at one site
- anything about real code, other languages, or other model families
- that H4 alone supports the conclusion; without H5 it is E11 again
- a null from H5 without checking that answer_direction_jlens failed on `ba` too
- that the R-lens arm gates anything — it is reported, never gated
- any number from the ARCHIVED `answer_direction` arm; it is a different estimator (docs/WORKSPACE_LENS.md §1) and is not comparable
