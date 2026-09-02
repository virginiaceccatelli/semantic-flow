# E13 binding interchange — starcoder2-3b

**Verdict: BINDING TRANSPORTED**

Does a low-rank, magnitude-free interchange at the site where a variable binding is resolved transport WHICH DEFINITION IS IN SCOPE, rather than a token or an answer direction?

## Identification

The same binding flip demands opposite token movements in the two value assignments. The alignment is fitted on arm `ab` and the claim is read on arm `ba`, where an answer direction is refuted rather than confounded.

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 0.981 [0.973, 0.988] against 0.85; weakest cell ab_target 0.954 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 7: binding decodable at 1.000 (selectivity 0.519) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 12 (both chosen on calibration): ab: +1.627 [+1.568, +1.683], flip rate 0.596; ba: +1.660 [+1.602, +1.718], flip rate 0.632 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L12 r1: +7.740 [+7.662, +7.816] = 476% of the whole-state ceiling +1.627 (threshold 50%); controls cleared: True; edit |14.884| = 0.466 of ||h||
- **PASS** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L12 r1: das_binding installed 100.0% = 155% of the held-out ceiling (threshold 50%); margin +7.765 [+7.693, +7.834] = 468% of it; discriminator — das_answer_control ab +4.752 [+4.595, +4.900], installed 96.6% (passes: True); ba/ab argmax ratio 0.468 against transport's 1.031 (bar 0.516) (fails: True)

## Controls — both arms

`ab` is the arm on which binding DAS and the answer-only control were fitted; `ba` is the crossed arm, where the identical binding flip demands the opposite token. `delta_ld` is the paired logit-difference shift with a 95% cluster-bootstrap interval over base programs; `installed` is the full-vocabulary argmax rate, which the gates read because `delta_ld` is positively biased at ceiling accuracy. `|edit|` and `|edit|/|h|` show the dose: the answer-only control and optional lens directions are matched to the treatment's own per-row edit norm, so no arm is compared at a different size. `vs das` is a paired difference on the *same* rows.

| arm | variant | delta_ld | 95% CI | installed | flip | \|edit\| | \|edit\|/\|h\| | vs das (paired, 95% CI) | n | bases |
|---|---|---|---|---|---|---|---|---|---|---|
| ab | `das_binding` | +7.740 | [+7.662, +7.816] | 100.0% | 97.7% | 14.884 | 0.466 | — | 560 | 280 |
| ab | `das_answer_control` | +4.752 | [+4.595, +4.900] | 96.6% | 94.3% | 14.884 | 0.466 | +2.989 [+2.844, +3.153] | 560 | 280 |
| ab | `mean_difference` | +1.066 | [+1.025, +1.107] | 47.1% | 44.8% | 22.252 | 0.697 | +6.674 [+6.604, +6.742] | 560 | 280 |
| ab | `random_rank` | +0.003 | [+0.002, +0.005] | 2.3% | 0.0% | 0.422 | 0.013 | +7.737 [+7.658, +7.813] | 560 | 280 |
| ab | `random_norm` | +0.417 | [+0.379, +0.454] | 21.2% | 18.4% | 16.441 | 0.515 | +7.323 [+7.241, +7.402] | 560 | 280 |
| ab | `whole_state` | +1.627 | [+1.568, +1.683] | 62.7% | 59.6% | 24.951 | 0.781 | +6.113 [+6.031, +6.189] | 560 | 280 |
| ba | `das_binding` | +7.765 | [+7.693, +7.834] | 100.0% | 98.8% | 14.910 | 0.467 | — | 560 | 280 |
| ba | `das_answer_control` | +1.105 | [+0.924, +1.277] | 45.2% | 44.3% | 14.910 | 0.467 | +6.660 [+6.500, +6.839] | 560 | 280 |
| ba | `mean_difference` | +1.078 | [+1.039, +1.117] | 45.9% | 44.3% | 22.290 | 0.698 | +6.687 [+6.618, +6.752] | 560 | 280 |
| ba | `random_rank` | +0.004 | [+0.002, +0.005] | 1.4% | 0.0% | 0.419 | 0.013 | +7.761 [+7.689, +7.831] | 560 | 280 |
| ba | `random_norm` | +0.423 | [+0.388, +0.458] | 22.3% | 20.5% | 16.425 | 0.514 | +7.342 [+7.267, +7.415] | 560 | 280 |
| ba | `whole_state` | +1.660 | [+1.602, +1.718] | 64.6% | 63.2% | 24.984 | 0.783 | +6.105 [+6.026, +6.185] | 560 | 280 |

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
