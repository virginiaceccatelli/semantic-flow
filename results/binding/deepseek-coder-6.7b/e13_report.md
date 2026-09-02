# E13 binding interchange — deepseek-coder-6.7b

**Verdict: BINDING TRANSPORTED**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 1.000 [1.000, 1.000] against 0.85; weakest cell ab_source 1.000 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 6: binding decodable at 1.000 (selectivity 0.521) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 6 (both chosen on calibration): ab: +4.025 [+3.923, +4.135], flip rate 0.738; ba: +4.017 [+3.905, +4.125], flip rate 0.734 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L6 r1: +8.119 [+8.040, +8.199] = 202% of the whole-state ceiling +4.025 (threshold 50%); controls cleared: True; edit |18.457| = 0.416 of ||h||
- **PASS** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L6 r1: das_binding installed 100.0% = 136% of the held-out ceiling (threshold 50%); margin +8.096 [+8.014, +8.179] = 202% of it; discriminator — das_answer_control ab +4.773 [+4.580, +4.965], installed 76.8% (passes: True); ba/ab argmax ratio 0.274 against transport's 0.995 (bar 0.498) (fails: True)

## Controls — both arms

`ab` is the arm on which binding DAS and the answer-only control were fitted; `ba` is the crossed arm, where the identical binding flip demands the opposite token. `delta_ld` is the paired logit-difference shift with a 95% cluster-bootstrap interval over base programs; `installed` is the full-vocabulary argmax rate, which the gates read because `delta_ld` is positively biased at ceiling accuracy. `|edit|` and `|edit|/|h|` show the dose: the answer-only control and optional lens directions are matched to the treatment's own per-row edit norm, so no arm is compared at a different size. `vs das` is a paired difference on the *same* rows.

| arm | variant | delta_ld | 95% CI | installed | flip | \|edit\| | \|edit\|/\|h\| | vs das (paired, 95% CI) | n | bases |
|---|---|---|---|---|---|---|---|---|---|---|
| ab | `das_binding` | +8.119 | [+8.040, +8.199] | 100.0% | 100.0% | 18.457 | 0.416 | — | 560 | 280 |
| ab | `das_answer_control` | +4.773 | [+4.580, +4.965] | 76.8% | 76.6% | 18.457 | 0.416 | +3.346 [+3.166, +3.535] | 560 | 280 |
| ab | `mean_difference` | +3.732 | [+3.625, +3.839] | 68.2% | 68.2% | 25.677 | 0.579 | +4.386 [+4.282, +4.500] | 560 | 280 |
| ab | `random_rank` | -0.001 | [-0.001, +0.000] | 0.0% | 0.0% | 0.588 | 0.013 | +8.119 [+8.041, +8.200] | 560 | 280 |
| ab | `random_norm` | +0.542 | [+0.474, +0.614] | 1.6% | 1.4% | 19.682 | 0.444 | +7.577 [+7.472, +7.681] | 560 | 280 |
| ab | `whole_state` | +4.025 | [+3.923, +4.135] | 73.8% | 73.8% | 30.911 | 0.697 | +4.093 [+3.983, +4.211] | 560 | 280 |
| ba | `das_binding` | +8.096 | [+8.014, +8.179] | 100.0% | 100.0% | 18.452 | 0.416 | — | 560 | 280 |
| ba | `das_answer_control` | +1.498 | [+1.320, +1.672] | 21.1% | 20.9% | 18.452 | 0.416 | +6.599 [+6.431, +6.784] | 560 | 280 |
| ba | `mean_difference` | +3.729 | [+3.622, +3.842] | 67.5% | 67.5% | 25.675 | 0.579 | +4.367 [+4.265, +4.470] | 560 | 280 |
| ba | `random_rank` | -0.001 | [-0.002, -0.000] | 0.0% | 0.0% | 0.584 | 0.013 | +8.098 [+8.015, +8.180] | 560 | 280 |
| ba | `random_norm` | +0.532 | [+0.461, +0.604] | 1.8% | 1.8% | 19.673 | 0.443 | +7.565 [+7.450, +7.675] | 560 | 280 |
| ba | `whole_state` | +4.017 | [+3.905, +4.125] | 73.4% | 73.4% | 30.927 | 0.697 | +4.079 [+3.973, +4.186] | 560 | 280 |

| variant | what it is |
|---|---|
| `das_binding` | the treatment — a learned low-rank interchange |
| `das_answer_control` | causally fitted answer-token actuator (H5's discriminator) |
| `mean_difference` | difference-in-means direction — the no-optimiser baseline |
| `random_rank` | a random subspace of the same rank |
| `random_norm` | a random subspace matched to the treatment's edit fraction |
| `whole_state` | the whole-state ceiling — what transport looks like here |

### How to read it

1. **DAS follows binding** if it succeeds in *both* crossed arms.
2. **The trained answer-only control** should work in `ab` and attenuate or reverse in `ba`: its fitted answer orientation is frozen.
3. `das_answer_control` must work above the matched-random floor on `ab` and attenuate or reverse on `ba`. Otherwise H5 does not identify binding transport.
4. Published J/R directions are reported as lens diagnostics; they do not gate the causal experiment.

## Do not claim

- that the model 'understands' scope — the claim is transport at one site
- anything about real code, other languages, or other model families
- that H4 alone supports the conclusion; without H5 it is E11 again
- a null from H5 without checking that das_answer_control was live on `ab`
- that the R-lens arm gates anything — it is reported, never gated
- any number from the ARCHIVED `answer_direction` arm; it is a different estimator (docs/WORKSPACE_LENS.md §1) and is not comparable
