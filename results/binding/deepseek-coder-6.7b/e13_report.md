# E13 binding interchange — deepseek-coder-6.7b

**Verdict: BINDING TRANSPORTED**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 1.000 [1.000, 1.000] against 0.85; weakest cell ab_source 1.000 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 8: binding decodable at 1.000 (selectivity 0.524) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 8 (both chosen on calibration): ab: +4.781 [+4.683, +4.878], flip rate 0.857; ba: +4.799 [+4.694, +4.903], flip rate 0.879 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L8 r1: +9.029 [+8.952, +9.108] = 189% of the whole-state ceiling +4.781 (threshold 50%); controls cleared: True; edit moved 0.479 of ||h||
- **PASS** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L8 r1: das_binding installed 100.0% = 114% of the held-out ceiling (threshold 50%); margin +9.009 [+8.933, +9.089] = 188% of it; discriminator — answer_direction ab +2.322 [+2.157, +2.482], installed 27.9% (passes: True); ba/ab argmax ratio 0.154 against transport's 1.025 (bar 0.513) (fails: True)
- **PASS** `H6` (the relevance readout is mechanically sound: the LRP rules actually installed so relevance conserves, the token roles partition every token exactly once, the per-role deltas close to the difference of the two conservation ratios, every binding_flip pair differs at exactly one measured token index, the fixed-target conditions really do score both members at one token, and every declared cell exists) — 25600 readings and 51200 paired contrasts over 4 contrasts x 8 layers x 4 target conditions; median |rho-1| 8.61e-08; conserving layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32}
- **PASS** `H7` (the candidate vocabulary is mechanically sound: every declared lexicon pair is kept whole or dropped whole with a reason, enough pairs survive from more than one family, discovery ran on calibration bases only and the frozen file records which, and the candidate set contains the lexicon, the discovered pool and the random controls without duplicates) — 10 lexicon pairs from 4 families, 15 mechanism words, 221 candidates (160 discovered, 32 random) over 120 calib bases
- **PASS** `H8` (the forced choice is mechanically sound: every declared (base, cell, question) is scored, the rendered question is identical in all four cells of a base, no question names the inner definition, both choices are distinct single tokens, both variants of every word style ran, and the value positive control ran) — 14400 scored choices over 9 questions x 400 bases x 4 cells; report split test
- **PASS** `H9` (the verbalisation relevance readout is mechanically sound: H6's checks on the verbalisation prompt, plus a positive-score condition — `R_t / s` is a share only when the score is positive, and conservation holds for negative scores too) — 38400 readings and 64000 paired contrasts over 4 contrasts x 8 layers x 5 target conditions for scope/direct; median |rho-1| 1.27e-07; readable layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32}

## Do not claim

- that the model 'understands' scope — the claim is transport at one site
- anything about real code, other languages, or other model families
- that H4 alone supports the conclusion; without H5 it is E11 again
- a null from H5 without checking that answer_direction failed on `ba` too
