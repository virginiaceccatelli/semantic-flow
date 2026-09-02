# E13 binding interchange — deepseek-coder-6.7b

**Verdict: BINDING TRANSPORTED**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 1.000 [1.000, 1.000] against 0.85; weakest cell ab_source 1.000 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 6: binding decodable at 1.000 (selectivity 0.521) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 6 (both chosen on calibration): ab: +4.024 [+3.922, +4.134], flip rate 0.738; ba: +4.017 [+3.905, +4.125], flip rate 0.734 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L6 r1: +8.115 [+8.037, +8.195] = 201% of the whole-state ceiling +4.028 (threshold 50%); controls cleared: True; edit |18.480| = 0.416 of ||h||
- **PASS** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L6 r1: das_binding installed 100.0% = 137% of the held-out ceiling (threshold 50%); margin +8.089 [+8.003, +8.173] = 201% of it; discriminator — answer_direction_jlens ab +0.096 [+0.080, +0.113], installed 0.0% (passes: True); ba -0.008 [-0.026, +0.010] (fails: True)
- **PASS** `H6` (the relevance readout is mechanically sound: the LRP rules actually installed so relevance conserves, the token roles partition every token exactly once, the per-role deltas close to the difference of the two conservation ratios, every binding_flip pair differs at exactly one measured token index, the fixed-target conditions really do score both members at one token, and every declared cell exists) — 25600 readings and 51200 paired contrasts over 4 contrasts x 8 layers x 4 target conditions; median |rho-1| 8.61e-08; conserving layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32}
- **PASS** `H7` (the candidate vocabulary is mechanically sound: every declared lexicon pair is kept whole or dropped whole with a reason, enough pairs survive from more than one family, discovery ran on calibration bases only and the frozen file records which, and the candidate set contains the lexicon, the discovered pool and the random controls without duplicates) — 10 lexicon pairs from 4 families, 15 mechanism words, 221 candidates (160 discovered, 32 random) over 120 calib bases
- **PASS** `H8` (the forced choice is mechanically sound: every declared (base, cell, question) is scored, the rendered question is identical in all four cells of a base, no question names the inner definition, both choices are distinct single tokens, both variants of every word style ran, and the value positive control ran) — 14400 scored choices over 9 questions x 400 bases x 4 cells; report split test
- **PASS** `H9` (the verbalisation relevance readout is mechanically sound: H6's checks on the verbalisation prompt, plus a positive-score condition — `R_t / s` is a share only when the score is positive, and conservation holds for negative scores too) — 38400 readings and 64000 paired contrasts over 4 contrasts x 8 layers x 5 target conditions for scope/direct; median |rho-1| 1.27e-07; readable layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32}
- **PASS** `H10` (the unprompted vocabulary readout is mechanically sound: every declared lexicon pair kept whole or dropped whole with a reason, the candidate row order the margin arithmetic assumes, all three readouts carrying the same rows with their kinds declared and the random control actually Gram-matched, the scored text being E13's program verbatim with the encodings agreeing through the use position, the use token identical in all four cells, every declared cell present in both arms over the declared layer grid, and the positive control fitted on calibration and read on test) — 108000 reversal rows over 400 bases x 2 arms x 5 layers x 3 readouts x 9 pairs; probe succeeds at layers [8, 12, 16, 20, 24]; report split test

## Controls — both arms

`ab` is the arm the DAS subspace and every fixed answer direction were built on; `ba` is the crossed arm, where the identical binding flip demands the opposite token. `delta_ld` is the paired logit-difference shift with a 95% cluster-bootstrap interval over base programs; `installed` is the full-vocabulary argmax rate, which the gates read because `delta_ld` is positively biased at ceiling accuracy. `|edit|` and `|edit|/|h|` show the dose: every fixed answer direction is matched to the treatment's own per-row edit norm, so no arm is compared against another at a different size. `vs das` is a paired difference on the *same* rows.

| arm | variant | delta_ld | 95% CI | installed | flip | \|edit\| | \|edit\|/\|h\| | vs das (paired, 95% CI) | n | bases |
|---|---|---|---|---|---|---|---|---|---|---|
| ab | `das_binding` | +8.115 | [+8.037, +8.195] | 100.0% | 100.0% | 18.480 | 0.416 | — | 560 | 280 |
| ab | `answer_direction_jlens` | +0.096 | [+0.080, +0.113] | 0.0% | 0.0% | 18.480 | 0.416 | +8.019 [+7.944, +8.094] | 560 | 280 |
| ab | `answer_direction_rlens` | +0.098 | [+0.081, +0.115] | 0.0% | 0.0% | 18.480 | 0.416 | +8.017 [+7.940, +8.096] | 560 | 280 |
| ab | `answer_direction_unembedding` | +0.003 | [-0.009, +0.016] | 0.0% | 0.0% | 18.480 | 0.416 | +8.112 [+8.032, +8.196] | 560 | 280 |
| ab | `mean_difference` | +3.733 | [+3.624, +3.840] | 68.4% | 68.0% | 25.689 | 0.579 | +4.382 [+4.276, +4.498] | 560 | 280 |
| ab | `random_rank` | -0.004 | [-0.011, +0.002] | 0.0% | 0.0% | 0.590 | 0.013 | +8.120 [+8.041, +8.200] | 560 | 280 |
| ab | `random_norm` | +0.607 | [+0.530, +0.683] | 2.0% | 2.0% | 19.852 | 0.447 | +7.508 [+7.392, +7.624] | 560 | 280 |
| ab | `whole_state` | +4.028 | [+3.925, +4.135] | 72.7% | 72.1% | 30.907 | 0.696 | +4.087 [+3.974, +4.204] | 560 | 280 |
| ba | `das_binding` | +8.089 | [+8.003, +8.173] | 100.0% | 100.0% | 18.477 | 0.416 | — | 560 | 280 |
| ba | `answer_direction_jlens` | -0.008 | [-0.026, +0.010] | 0.0% | 0.0% | 18.477 | 0.416 | +8.097 [+8.010, +8.183] | 560 | 280 |
| ba | `answer_direction_rlens` | +0.042 | [+0.023, +0.061] | 0.0% | 0.0% | 18.477 | 0.416 | +8.047 [+7.963, +8.132] | 560 | 280 |
| ba | `answer_direction_unembedding` | +0.071 | [+0.060, +0.083] | 0.0% | 0.0% | 18.477 | 0.416 | +8.018 [+7.933, +8.104] | 560 | 280 |
| ba | `mean_difference` | +3.725 | [+3.617, +3.838] | 67.5% | 67.0% | 25.693 | 0.579 | +4.364 [+4.259, +4.467] | 560 | 280 |
| ba | `random_rank` | -0.004 | [-0.009, +0.002] | 0.0% | 0.0% | 0.586 | 0.013 | +8.093 [+8.008, +8.176] | 560 | 280 |
| ba | `random_norm` | +0.546 | [+0.477, +0.625] | 2.0% | 2.0% | 19.640 | 0.442 | +7.544 [+7.422, +7.657] | 560 | 280 |
| ba | `whole_state` | +4.017 | [+3.904, +4.125] | 72.9% | 72.1% | 30.931 | 0.696 | +4.072 [+3.966, +4.181] | 560 | 280 |

| variant | what it is |
|---|---|
| `das_binding` | the treatment — a learned low-rank interchange |
| `answer_direction_jlens` | PUBLISHED J-lens answer direction (H5's discriminator) |
| `answer_direction_rlens` | published R-lens answer direction (descriptive) |
| `answer_direction_unembedding` | raw unembedding rows — no transport (floor) |
| `mean_difference` | difference-in-means direction — the no-optimiser baseline |
| `random_rank` | a random subspace of the same rank |
| `random_norm` | a random subspace matched to the treatment's edit fraction |
| `whole_state` | the whole-state ceiling — what transport looks like here |

### How to read it

1. **DAS follows binding** if it succeeds in *both* crossed arms.
2. **A fixed answer direction** should work in the training arm `ab` and attenuate or reverse in the crossed arm `ba` — it was built from `ab`'s required movement and held fixed.
3. If `answer_direction_jlens` **also succeeds like DAS in both arms**, H5 does not distinguish binding transport from a lens-visible answer direction, and the causal verdict must not pass.
4. `answer_direction_rlens` provides the same secondary diagnostic through the published R-lens. It is reported, not gated.

## Do not claim

- that the model 'understands' scope — the claim is transport at one site
- anything about real code, other languages, or other model families
- that H4 alone supports the conclusion; without H5 it is E11 again
- a null from H5 without checking that answer_direction_jlens failed on `ba` too
- that the R-lens arm gates anything — it is reported, never gated
- any number from the ARCHIVED `answer_direction` arm; it is a different estimator (docs/WORKSPACE_LENS.md §1) and is not comparable
