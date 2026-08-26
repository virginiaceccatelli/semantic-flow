# E16 — R-lens attribution on the binding counterfactual (deepseek-coder-6.7b)

## What this experiment asks

When the binding of a variable use changes and **exactly one token** of the program changes with it, does the model's own attribution of its answer move from the definition that just went out of scope to the one that just came into scope?

**Verdict: `binding_shift_found`**

When the binding changes and nothing else does, the model's own attribution of its answer moves from the definition that went out of scope to the one that came into scope. The shift survives the token-identical restriction, replicates across the arms where the scored token moves the other way, and does not appear in the same-binding controls. It remains OBSERVATIONAL: it says where the answer is attributed, not what the model uses.

> **This is observational.** The R-lens decomposes the model's output score over input positions; it intervenes on nothing. E13/R10's DAS interchange is the causal benchmark on this same corpus, and the comparison below reports only what is comparable between them. A relevance shift is not weak causal evidence — it is evidence about a different quantity.

## The construction, and what it rules out for free

```
  z = 2                      z = 2
  def f():                   def f():
      d = 4   <- name            z = 4   <- name
      return z                   return z
  -> 2  (outer binding)      -> 4  (inner binding)
```

Within one arm the two programs differ at **one token index** out of ~21 — the inner definition's name. The outer definition, the inner definition's *value*, the use site, the signature and the answer suffix are token-identical at identical indices, which stage 140 measures on the encoded prompts rather than inheriting from the data file (table 7). So a redistribution among those roles cannot be the differing token, a length effect, a tokenisation artifact, or positional drift.

The headline statistic is **`binding_shift_identical`** = `delta_frac_inner_def_identical - delta_frac_outer_def`: the inner definition's token-identical half gaining share minus the (wholly token-identical) outer definition losing it. Positive means relevance moved toward the newly active definition. Relevance is taken for the model's output score of the **bound value** (`target_condition = bound`).

## Gates

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 400 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **PASS** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 1.000 [1.000, 1.000] against 0.85; weakest cell ab_source 1.000 against 0.75
- **PASS** `H2` (which definition is in scope is decodable at the use anchor) — best layer 8: binding decodable at 1.000 (selectivity 0.524) against a MEASURED surface baseline of 0.500; margin +0.500. Thresholds 0.8 and 0.1. The floor is pinned by construction here: the anchor token is identical across the counterfactual and the mutation is outside the baseline's window.
- **PASS** `H3` (whole-state interchange flips the answer — the ceiling, per arm) — site use, layer 8 (both chosen on calibration): ab: +4.781 [+4.683, +4.878], flip rate 0.857; ba: +4.799 [+4.694, +4.903], flip rate 0.879 (thresholds: CI above 0.0, flip rate 0.25). Both arms must be measurable or an H5 null says nothing.
- **PASS** `H4` (low-rank interchange beats matched controls on the TRAINING arm) — ab @ use L8 r1: +9.029 [+8.952, +9.108] = 189% of the whole-state ceiling +4.781 (threshold 50%); controls cleared: True; edit moved 0.479 of ||h||
- **FAIL** `H5` (the same subspace transfers to the HELD-OUT value assignment, where an answer direction cannot) — ba @ use L8 r1: das_binding +9.009 [+8.933, +9.089] = 188% of the held-out ceiling (threshold 50%); discriminator — answer_direction ab +2.322 [+2.157, +2.482] (passes: True), ba +0.335 [+0.208, +0.456] (fails: False)
- **PASS** `H6` (the relevance readout is mechanically sound: the LRP rules actually installed so relevance conserves, the token roles partition every token exactly once, the per-role deltas close to the difference of the two conservation ratios, every binding_flip pair differs at exactly one measured token index, the fixed-target conditions really do score both members at one token, and every declared cell exists) — 25600 readings and 51200 paired contrasts over 4 contrasts x 8 layers x 4 target conditions; median |rho-1| 8.61e-08; conserving layers [0, 3, 7, 11, 15, 19, 23, 27]; LRP rules bound {'ln': 65, 'mlp': 32, 'attn': 32}

H6 is **mechanical**: a null redistribution passes it. It gates whether the numbers are relevance at all, never whether they are interesting.

## The reported cell

- layer **0**, selected on **calibration** by the rule in `binding_relevance.select_cell`, read on split `test`
- conserving layers: [0, 3, 7, 11, 15, 19, 23, 27] (tolerance |rho-1| <= 0.25)
- declared thresholds: sign consistency 0.7, p < 0.05

| check | holds |
|---|---|
| rules_installed_and_conserving | yes |
| shift_consistent | yes |
| above_permutation_control | yes |
| above_sign_test | yes |
| arms_agree | yes |
| same_binding_controls_quiet | yes |
| statistic_is_token_identical | yes |

### Table 1 — the headline statistic, every contrast and layer

`expect` is declared in `binding_relevance.CONTRASTS` before the run: `shift` for the two binding flips, `null` for the two same-binding controls where the bound token moves the same way and the binding does not. `ci_lo`/`ci_hi` are a cluster bootstrap over bases — the same interval convention stage 106 reports DAS with.

| contrast | expect | layer | n_pairs | n_bases | mean_delta | ci_lo | ci_hi | median_delta | cohens_d | sign_consistency | n_nonzero | sign_test_p | permutation_p | permutation_effect_size | degenerate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flip_ab | shift | 0 | 280 | 280 | 0.12634 | 0.12461 | 0.12805 | 0.12657 | 8.59616 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.56182 | 0 |
| flip_ab | shift | 3 | 280 | 280 | 0.13666 | 0.13474 | 0.13856 | 0.13592 | 8.36277 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.55001 | 0 |
| flip_ab | shift | 7 | 280 | 280 | 0.17181 | 0.16984 | 0.17378 | 0.17305 | 10.17265 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.59853 | 0 |
| flip_ab | shift | 11 | 280 | 280 | 0.20465 | 0.20234 | 0.20709 | 0.20652 | 10.25955 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.60690 | 0 |
| flip_ab | shift | 15 | 280 | 280 | 0.21869 | 0.21584 | 0.22148 | 0.22111 | 9.05740 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.58330 | 0 |
| flip_ab | shift | 19 | 280 | 280 | 0.09059 | 0.08825 | 0.09288 | 0.09213 | 4.61286 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.12036 | 0 |
| flip_ab | shift | 23 | 280 | 280 | 0.07198 | 0.06994 | 0.07404 | 0.07503 | 4.06964 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.16414 | 0 |
| flip_ab | shift | 27 | 280 | 280 | 0.02452 | 0.02337 | 0.02572 | 0.02607 | 2.41314 | 0.97500 | 280 | 0.00000 | 0.00000 | 15.56401 | 0 |
| flip_ba | shift | 0 | 280 | 280 | 0.12705 | 0.12543 | 0.12873 | 0.12645 | 8.69918 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.48079 | 0 |
| flip_ba | shift | 3 | 280 | 280 | 0.13743 | 0.13559 | 0.13927 | 0.13568 | 8.46132 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.43815 | 0 |
| flip_ba | shift | 7 | 280 | 280 | 0.17213 | 0.17015 | 0.17415 | 0.17159 | 9.75892 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.59471 | 0 |
| flip_ba | shift | 11 | 280 | 280 | 0.20494 | 0.20248 | 0.20736 | 0.20592 | 9.89041 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.62122 | 0 |
| flip_ba | shift | 15 | 280 | 280 | 0.21936 | 0.21654 | 0.22219 | 0.22175 | 8.93647 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.62383 | 0 |
| flip_ba | shift | 19 | 280 | 280 | 0.09026 | 0.08807 | 0.09235 | 0.09076 | 4.88242 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.32419 | 0 |
| flip_ba | shift | 23 | 280 | 280 | 0.07134 | 0.06948 | 0.07309 | 0.07396 | 4.51033 | 1.00000 | 280 | 0.00000 | 0.00000 | 16.43382 | 0 |
| flip_ba | shift | 27 | 280 | 280 | 0.02398 | 0.02280 | 0.02513 | 0.02603 | 2.46699 | 0.98214 | 280 | 0.00000 | 0.00000 | 15.59789 | 0 |
| same_outer | no_shift | 0 | 280 | 280 | -0.00136 | -0.00452 | 0.00189 | -0.00128 | -0.05084 | 0.48571 | 280 | 0.67578 | 0.35800 | -1.03696 | 0 |
| same_outer | no_shift | 3 | 280 | 280 | -0.00170 | -0.00506 | 0.00171 | -0.00173 | -0.05973 | 0.47143 | 280 | 0.37006 | 0.29400 | -1.18064 | 0 |
| same_outer | no_shift | 7 | 280 | 280 | -0.00128 | -0.00523 | 0.00270 | -0.00143 | -0.03817 | 0.48929 | 280 | 0.76515 | 0.50800 | -0.77813 | 0 |
| same_outer | no_shift | 11 | 280 | 280 | -0.00138 | -0.00634 | 0.00358 | -0.00090 | -0.03298 | 0.48929 | 280 | 0.76515 | 0.54800 | -0.67966 | 0 |
| same_outer | no_shift | 15 | 280 | 280 | -0.00201 | -0.00715 | 0.00316 | -0.00270 | -0.04596 | 0.46786 | 280 | 0.30965 | 0.41600 | -0.92565 | 0 |
| same_outer | no_shift | 19 | 280 | 280 | -0.00027 | -0.00353 | 0.00313 | -0.00012 | -0.00969 | 0.49643 | 280 | 0.95236 | 0.83600 | -0.27516 | 0 |
| same_outer | no_shift | 23 | 280 | 280 | 0.00021 | -0.00263 | 0.00298 | 0.00100 | 0.00919 | 0.51429 | 280 | 0.67578 | 0.85000 | 0.06778 | 0 |
| same_outer | no_shift | 27 | 280 | 280 | 0.00070 | -0.00052 | 0.00192 | 0.00024 | 0.06870 | 0.51429 | 280 | 0.67578 | 0.24400 | 1.06984 | 0 |
| same_inner | no_shift | 0 | 280 | 280 | 0.00066 | -0.00168 | 0.00302 | -0.00028 | 0.03317 | 0.48929 | 280 | 0.76515 | 0.57400 | 0.66649 | 0 |
| same_inner | no_shift | 3 | 280 | 280 | 0.00094 | -0.00156 | 0.00342 | 0.00116 | 0.04438 | 0.51786 | 280 | 0.59076 | 0.44200 | 0.85606 | 0 |
| same_inner | no_shift | 7 | 280 | 280 | 0.00097 | -0.00335 | 0.00528 | 0.00153 | 0.02667 | 0.51429 | 280 | 0.67578 | 0.64200 | 0.55280 | 0 |
| same_inner | no_shift | 11 | 280 | 280 | 0.00109 | -0.00363 | 0.00567 | 0.00352 | 0.02731 | 0.53214 | 280 | 0.30965 | 0.62400 | 0.56273 | 0 |
| same_inner | no_shift | 15 | 280 | 280 | 0.00135 | -0.00323 | 0.00579 | 0.00184 | 0.03555 | 0.51429 | 280 | 0.67578 | 0.52400 | 0.72344 | 0 |
| same_inner | no_shift | 19 | 280 | 280 | 0.00061 | -0.00336 | 0.00442 | -0.00012 | 0.01944 | 0.50000 | 280 | 1.00000 | 0.72600 | 0.44014 | 0 |
| same_inner | no_shift | 23 | 280 | 280 | 0.00042 | -0.00275 | 0.00354 | 0.00057 | 0.01670 | 0.51429 | 280 | 0.67578 | 0.77200 | 0.36578 | 0 |
| same_inner | no_shift | 27 | 280 | 280 | -0.00016 | -0.00238 | 0.00198 | -0.00088 | -0.00877 | 0.48571 | 280 | 0.67578 | 0.88000 | -0.12041 | 0 |

### Table 2 — the layer profile of the two binding flips

This is where the attribution is redistributed, not where binding is computed. Compare the depth with DAS's chosen layer in the comparison section, not the magnitudes.

| layer | contrast | mean_delta | median_delta | sign_consistency | sign_test_p | permutation_p | n_pairs | median_abs_rho_minus_one |
|---|---|---|---|---|---|---|---|---|
| 0 | flip_ab | 0.12634 | 0.12657 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 0 | flip_ba | 0.12705 | 0.12645 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 3 | flip_ab | 0.13666 | 0.13592 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 3 | flip_ba | 0.13743 | 0.13568 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 7 | flip_ab | 0.17181 | 0.17305 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 7 | flip_ba | 0.17213 | 0.17159 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 11 | flip_ab | 0.20465 | 0.20652 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 11 | flip_ba | 0.20494 | 0.20592 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 15 | flip_ab | 0.21869 | 0.22111 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 15 | flip_ba | 0.21936 | 0.22175 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 19 | flip_ab | 0.09059 | 0.09213 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 19 | flip_ba | 0.09026 | 0.09076 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 23 | flip_ab | 0.07198 | 0.07503 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 23 | flip_ba | 0.07134 | 0.07396 | 1.00000 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 27 | flip_ab | 0.02452 | 0.02607 | 0.97500 | 0.00000 | 0.00000 | 280 | 0.00000 |
| 27 | flip_ba | 0.02398 | 0.02603 | 0.98214 | 0.00000 | 0.00000 | 280 | 0.00000 |

### Table 3 — the output-token control: do the arms agree?

Under `bound` the scored token moves v_a -> v_b in `flip_ab` and v_b -> v_a in `flip_ba`. An artifact of which token the relevance is taken for must **reverse sign** between the arms; a binding effect must not. `arm_ratio` near +1 is agreement, negative is the artifact signature. This is the same crossing stage 106 reads DAS's `answer_direction` control on.

| layer | mean_delta_ab | mean_delta_ba | median_delta_ab | median_delta_ba | sign_consistency_ab | sign_consistency_ba | signs_agree | arm_ratio | both_significant_sign |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.12634 | 0.12705 | 0.12657 | 0.12645 | 1.00000 | 1.00000 | 1 | 1.00559 | 1 |
| 3 | 0.13666 | 0.13743 | 0.13592 | 0.13568 | 1.00000 | 1.00000 | 1 | 1.00558 | 1 |
| 7 | 0.17181 | 0.17213 | 0.17305 | 0.17159 | 1.00000 | 1.00000 | 1 | 1.00183 | 1 |
| 11 | 0.20465 | 0.20494 | 0.20652 | 0.20592 | 1.00000 | 1.00000 | 1 | 1.00139 | 1 |
| 15 | 0.21869 | 0.21936 | 0.22111 | 0.22175 | 1.00000 | 1.00000 | 1 | 1.00305 | 1 |
| 19 | 0.09059 | 0.09026 | 0.09213 | 0.09076 | 1.00000 | 1.00000 | 1 | 0.99631 | 1 |
| 23 | 0.07198 | 0.07134 | 0.07503 | 0.07396 | 1.00000 | 1.00000 | 1 | 0.99113 | 1 |
| 27 | 0.02452 | 0.02398 | 0.02607 | 0.02603 | 0.97500 | 0.98214 | 1 | 0.97819 | 1 |

### Table 4 — the output-token control, part two: the same token in both members

`fixed_a` and `fixed_b` score BOTH members at literally the same token id, so the output token is removed from the contrast entirely. They cost no extra backward pass: each program is already read at both candidate tokens. If the shift under `bound` were about the scored token, these rows would be flat.

| contrast | target_condition | same_target_token | mean_delta | ci_lo | ci_hi | median_delta | sign_consistency | sign_test_p | permutation_p | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| flip_ab | bound | 0 | 0.12634 | 0.12461 | 0.12805 | 0.12657 | 1.00000 | 0.00000 | 0.00000 | 280 |
| flip_ab | fixed_a | 1 | 0.03696 | 0.03479 | 0.03925 | 0.03734 | 0.97500 | 0.00000 | 0.00000 | 280 |
| flip_ab | fixed_b | 1 | 0.03650 | 0.03447 | 0.03853 | 0.03556 | 0.99286 | 0.00000 | 0.00000 | 280 |
| flip_ab | other | 0 | -0.05288 | -0.05619 | -0.04959 | -0.05271 | 0.02857 | 0.00000 | 0.00000 | 280 |
| flip_ba | bound | 0 | 0.12705 | 0.12543 | 0.12873 | 0.12645 | 1.00000 | 0.00000 | 0.00000 | 280 |
| flip_ba | fixed_a | 1 | 0.03734 | 0.03516 | 0.03949 | 0.03630 | 0.99286 | 0.00000 | 0.00000 | 280 |
| flip_ba | fixed_b | 1 | 0.03742 | 0.03517 | 0.03969 | 0.03618 | 0.97143 | 0.00000 | 0.00000 | 280 |
| flip_ba | other | 0 | -0.05229 | -0.05557 | -0.04889 | -0.05308 | 0.02857 | 0.00000 | 0.00000 | 280 |
| same_outer | bound | 0 | -0.00136 | -0.00452 | 0.00189 | -0.00128 | 0.48571 | 0.67578 | 0.35800 | 280 |
| same_outer | fixed_a | 1 | 0.08834 | 0.08530 | 0.09143 | 0.08332 | 1.00000 | 0.00000 | 0.00000 | 280 |
| same_outer | fixed_b | 1 | -0.09121 | -0.09450 | -0.08775 | -0.08937 | 0.00357 | 0.00000 | 0.00000 | 280 |
| same_outer | other | 0 | -0.00150 | -0.00409 | 0.00108 | -0.00187 | 0.46071 | 0.20940 | 0.24600 | 280 |
| same_inner | bound | 0 | 0.00066 | -0.00168 | 0.00302 | -0.00028 | 0.48929 | 0.76515 | 0.57400 | 280 |
| same_inner | fixed_a | 1 | -0.08872 | -0.09220 | -0.08533 | -0.07980 | 0.00000 | 0.00000 | 0.00000 | 280 |
| same_inner | fixed_b | 1 | 0.09029 | 0.08684 | 0.09389 | 0.08217 | 1.00000 | 0.00000 | 0.00000 | 280 |
| same_inner | other | 0 | 0.00091 | -0.00320 | 0.00492 | 0.00268 | 0.52857 | 0.37006 | 0.61800 | 280 |

### Table 5 — every role at the reported cell

`mean_delta` is the paired change in a role's share of the model's answer. The column sums to ~0 by conservation: whatever one role gains, another loses. `token_identical` marks the roles whose tokens do not change; `inner_def_name` is the one that does, and it is reported rather than hidden.

| role | token_identical | n_pairs | mean_frac_from | mean_frac_to | median_delta | mean_delta | ci_lo | ci_hi | sign_consistency | sign_test_p | permutation_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| outer_def_value | 1 | 280 | 0.06564 | 0.02283 | -0.04239 | -0.04280 | -0.04374 | -0.04189 | 0.00000 | 0.00000 | 0.00000 |
| outer_def_name | 1 | 280 | 0.37690 | 0.34205 | -0.03451 | -0.03485 | -0.03586 | -0.03385 | 0.00000 | 0.00000 | 0.00000 |
| use_site | 1 | 280 | 0.02225 | -0.00036 | -0.02197 | -0.02261 | -0.02332 | -0.02191 | 0.00000 | 0.00000 | 0.00000 |
| return_kw | 1 | 280 | 0.02065 | 0.01809 | -0.00236 | -0.00256 | -0.00295 | -0.00216 | 0.22500 | 0.00000 | 0.00000 |
| suffix | 1 | 280 | 0.16584 | 0.16554 | -0.00024 | -0.00029 | -0.00150 | 0.00088 | 0.49643 | 0.95236 | 0.66200 |
| inner_def_name | 0 | 280 | 0.00525 | 0.00708 | 0.00212 | 0.00184 | 0.00129 | 0.00240 | 0.66429 | 0.00000 | 0.00000 |
| signature | 1 | 280 | 0.03012 | 0.04190 | 0.01186 | 0.01178 | 0.01106 | 0.01247 | 0.97500 | 0.00000 | 0.00000 |
| other | 1 | 280 | 0.29060 | 0.33142 | 0.04087 | 0.04082 | 0.03976 | 0.04192 | 1.00000 | 0.00000 | 0.00000 |
| inner_def_value | 1 | 280 | 0.02276 | 0.07144 | 0.04858 | 0.04868 | 0.04781 | 0.04957 | 1.00000 | 0.00000 | 0.00000 |

The composites below are **sums of the rows above**, so they do not add to zero and are listed separately rather than mixed in. `binding_shift` and `binding_shift_identical` are differences of two composites; only the second is made entirely of token-identical spans, which is why it is the headline.

| role | token_identical | n_pairs | median_delta | mean_delta | ci_lo | ci_hi | sign_consistency | sign_test_p | permutation_p |
|---|---|---|---|---|---|---|---|---|---|
| outer_def | 1 | 280 | -0.07703 | -0.07766 | -0.07932 | -0.07598 | 0.00000 | 0.00000 | 0.00000 |
| both_defs | 0 | 280 | -0.02604 | -0.02714 | -0.02911 | -0.02527 | 0.01071 | 0.00000 | 0.00000 |
| inner_def_identical | 1 | 280 | 0.04858 | 0.04868 | 0.04781 | 0.04957 | 1.00000 | 0.00000 | 0.00000 |
| inner_def | 0 | 280 | 0.05139 | 0.05052 | 0.04964 | 0.05146 | 1.00000 | 0.00000 | 0.00000 |
| binding_shift_identical | 1 | 280 | 0.12657 | 0.12634 | 0.12461 | 0.12805 | 1.00000 | 0.00000 | 0.00000 |
| binding_shift | 0 | 280 | 0.12812 | 0.12818 | 0.12631 | 0.13012 | 1.00000 | 0.00000 | 0.00000 |

### Table 6 — the same statistic on pairs the model actually answers

**400 of 400** bases have the model answering BOTH members of `flip_ab` correctly. H1 is not a prerequisite for this stage — it fails on deepseek-coder-1.3b — so the shift is reported on all pairs above and on that subset here.

| contrast | expect | layer | n_pairs | mean_delta | ci_lo | ci_hi | median_delta | sign_consistency | sign_test_p | permutation_p |
|---|---|---|---|---|---|---|---|---|---|---|
| flip_ab | shift | 0 | 280 | 0.12634 | 0.12461 | 0.12805 | 0.12657 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 3 | 280 | 0.13666 | 0.13474 | 0.13856 | 0.13592 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 7 | 280 | 0.17181 | 0.16984 | 0.17378 | 0.17305 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 11 | 280 | 0.20465 | 0.20234 | 0.20709 | 0.20652 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 15 | 280 | 0.21869 | 0.21584 | 0.22148 | 0.22111 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 19 | 280 | 0.09059 | 0.08825 | 0.09288 | 0.09213 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 23 | 280 | 0.07198 | 0.06994 | 0.07404 | 0.07503 | 1.00000 | 0.00000 | 0.00000 |
| flip_ab | shift | 27 | 280 | 0.02452 | 0.02337 | 0.02572 | 0.02607 | 0.97500 | 0.00000 | 0.00000 |
| flip_ba | shift | 0 | 280 | 0.12705 | 0.12543 | 0.12873 | 0.12645 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 3 | 280 | 0.13743 | 0.13559 | 0.13927 | 0.13568 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 7 | 280 | 0.17213 | 0.17015 | 0.17415 | 0.17159 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 11 | 280 | 0.20494 | 0.20248 | 0.20736 | 0.20592 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 15 | 280 | 0.21936 | 0.21654 | 0.22219 | 0.22175 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 19 | 280 | 0.09026 | 0.08807 | 0.09235 | 0.09076 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 23 | 280 | 0.07134 | 0.06948 | 0.07309 | 0.07396 | 1.00000 | 0.00000 | 0.00000 |
| flip_ba | shift | 27 | 280 | 0.02398 | 0.02280 | 0.02513 | 0.02603 | 0.98214 | 0.00000 | 0.00000 |
| same_outer | no_shift | 0 | 280 | -0.00136 | -0.00452 | 0.00189 | -0.00128 | 0.48571 | 0.67578 | 0.35800 |
| same_outer | no_shift | 3 | 280 | -0.00170 | -0.00506 | 0.00171 | -0.00173 | 0.47143 | 0.37006 | 0.29400 |
| same_outer | no_shift | 7 | 280 | -0.00128 | -0.00523 | 0.00270 | -0.00143 | 0.48929 | 0.76515 | 0.50800 |
| same_outer | no_shift | 11 | 280 | -0.00138 | -0.00634 | 0.00358 | -0.00090 | 0.48929 | 0.76515 | 0.54800 |
| same_outer | no_shift | 15 | 280 | -0.00201 | -0.00715 | 0.00316 | -0.00270 | 0.46786 | 0.30965 | 0.41600 |
| same_outer | no_shift | 19 | 280 | -0.00027 | -0.00353 | 0.00313 | -0.00012 | 0.49643 | 0.95236 | 0.83600 |
| same_outer | no_shift | 23 | 280 | 0.00021 | -0.00263 | 0.00298 | 0.00100 | 0.51429 | 0.67578 | 0.85000 |
| same_outer | no_shift | 27 | 280 | 0.00070 | -0.00052 | 0.00192 | 0.00024 | 0.51429 | 0.67578 | 0.24400 |
| same_inner | no_shift | 0 | 280 | 0.00066 | -0.00168 | 0.00302 | -0.00028 | 0.48929 | 0.76515 | 0.57400 |
| same_inner | no_shift | 3 | 280 | 0.00094 | -0.00156 | 0.00342 | 0.00116 | 0.51786 | 0.59076 | 0.44200 |
| same_inner | no_shift | 7 | 280 | 0.00097 | -0.00335 | 0.00528 | 0.00153 | 0.51429 | 0.67578 | 0.64200 |
| same_inner | no_shift | 11 | 280 | 0.00109 | -0.00363 | 0.00567 | 0.00352 | 0.53214 | 0.30965 | 0.62400 |
| same_inner | no_shift | 15 | 280 | 0.00135 | -0.00323 | 0.00579 | 0.00184 | 0.51429 | 0.67578 | 0.52400 |
| same_inner | no_shift | 19 | 280 | 0.00061 | -0.00336 | 0.00442 | -0.00012 | 0.50000 | 1.00000 | 0.72600 |
| same_inner | no_shift | 23 | 280 | 0.00042 | -0.00275 | 0.00354 | 0.00057 | 0.51429 | 0.67578 | 0.77200 |
| same_inner | no_shift | 27 | 280 | -0.00016 | -0.00238 | 0.00198 | -0.00088 | 0.48571 | 0.67578 | 0.88000 |

### Table 7 — the token-identity control, measured

`as_designed` is the fraction of pairs with the expected number of differing token indices (1 for a binding flip, 2 for a same-binding control, since both value literals move). `use_token_identical` must be 1.0 everywhere or a relevance change at the use site could be the token.

| contrast | contrast_kind | n | same_length | mean_differing | as_designed | use_token_identical |
|---|---|---|---|---|---|---|
| flip_ab | binding_flip | 400 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| flip_ba | binding_flip | 400 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| same_inner | same_binding | 400 | 1.00000 | 2.00000 | 1.00000 | 1.00000 |
| same_outer | same_binding | 400 | 1.00000 | 2.00000 | 1.00000 | 1.00000 |

### Table 8 — the mismatched-pair control

Members drawn from **different bases** with the orientation kept. The permutation null keeps the pairing and destroys the orientation; this keeps the orientation and destroys the base matching, so what it can falsify is 'the redistribution is specific to this pairing'.

| contrast | layer | mean_delta | median_delta | sign_consistency | n |
|---|---|---|---|---|---|
| flip_ab_mismatched | 0 | 0.12682 | 0.12635 | 1.00000 | 400 |
| flip_ba_mismatched | 0 | 0.12705 | 0.12725 | 1.00000 | 400 |

### Table 9 — conservation, the validity condition

The fraction reading is licensed only where relevance conserves. This is measured per (layer, target mode) on this run's own programs, not inherited from E14 gate R2.

| layer | target_mode | n_readings | median_rho | median_abs_rho_minus_one | max_abs_rho_minus_one | conserving |
|---|---|---|---|---|---|---|
| 0 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 0 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 3 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 3 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 7 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 7 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 11 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 11 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 15 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 15 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 19 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 19 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 23 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 23 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 27 | bound | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |
| 27 | other | 1600 | 1.00000 | 0.00000 | 0.00000 | 1 |

### Table 10 — per-position deltas at the reported cell

E15-D could not produce this table: its pair members are not token-aligned. Here all four cells share a token length and differ at one index, so this shows whether the role aggregation is hiding a single position doing all the work.

| position | role_to | mean_delta | median_delta | sign_consistency | n |
|---|---|---|---|---|---|
| 0 | outer_def_name | -0.03504 | -0.03456 | 0.00000 | 400 |
| 1 | other | -0.01799 | -0.01779 | 0.00000 | 400 |
| 2 | outer_def_value | -0.04309 | -0.04301 | 0.00000 | 400 |
| 3 | other | 0.03777 | 0.03755 | 1.00000 | 400 |
| 4 | signature | 0.00365 | 0.00368 | 0.83750 | 400 |
| 5 | signature | 0.00261 | 0.00262 | 0.87500 | 400 |
| 6 | signature | 0.00539 | 0.00544 | 0.99750 | 400 |
| 7 | other | 0.00017 | 0.00014 | 0.61500 | 400 |
| 8 | other | 0.00181 | 0.00167 | 0.95500 | 400 |
| 9 | inner_def_name | 0.00208 | 0.00213 | 0.67250 | 400 |
| 10 | other | 0.01838 | 0.01848 | 1.00000 | 400 |
| 11 | inner_def_value | 0.04869 | 0.04845 | 1.00000 | 400 |
| 12 | other | 0.00063 | 0.00063 | 0.79750 | 400 |
| 13 | other | 0.00005 | 0.00004 | 0.53000 | 400 |
| 14 | return_kw | -0.00245 | -0.00234 | 0.24000 | 400 |
| 15 | use_site | -0.02270 | -0.02207 | 0.00000 | 400 |
| 16 | suffix | -0.00120 | -0.00116 | 0.11750 | 400 |
| 17 | suffix | -0.00157 | -0.00178 | 0.31000 | 400 |
| 18 | suffix | 0.00131 | 0.00141 | 0.90000 | 400 |
| 19 | suffix | 0.00316 | 0.00337 | 0.86000 | 400 |
| 20 | suffix | -0.00166 | -0.00144 | 0.35250 | 400 |

## Observational R-lens versus causal DAS on the same corpus

DAS (stage 106) reports at site `use`, layer **8**; this stage reports at layer **0**.

| | R-lens (this stage, E16) | DAS (stage 106, R10) |
|---|---|---|
| what is done to the model | nothing | a rank-1 subspace at the use anchor is replaced with the donor's |
| what is read | how the answer score decomposes over input positions | whether the emitted token becomes the installed binding's value |
| licenses | a statement about **attribution** | a statement about **causal transport** at that site, layer and construction |
| reported layer | 0 | 8 |
| both arms | see table 3 | 100.0% / 100.0% (`says_installed`) |
| effect size units | share of the answer score | rate of answer change |

**The units do not convert.** No ratio between the two is computed anywhere in this pipeline. What the two results can jointly support is a conjunction, not a chain: at this site the binding is causally transportable (DAS) *and* the attribution redistributes with it (R-lens), or it is transportable and the attribution does not move — which would itself be the more interesting finding, because it would show attribution and use coming apart on a corpus where the causal fact is settled.

### Table 11 — E13's causal numbers, as stage 106 wrote them

| arm | variant | site | layer | rank | split | says_installed_rate | edit_fraction | delta_ld | ci_lo | ci_hi | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ab | answer_direction | use | 8 | 1 | test | 0.27857 | 0.47924 | 2.32201 | 2.15674 | 2.48165 | 560 |
| ab | das_binding | use | 8 | 1 | test | 1.00000 | 0.47924 | 9.02910 | 8.95172 | 9.10840 | 560 |
| ab | mean_difference | use | 8 | 1 | test | 0.76071 | 0.71059 | 4.13806 | 4.02547 | 4.25223 | 560 |
| ab | noop | use | 8 | 1 | test | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 560 |
| ab | random_norm | use | 8 | 1536 | test | 0.02143 | 0.51338 | 0.75684 | 0.69129 | 0.82685 | 560 |
| ab | whole_state | use | 8 | 4096 | test | 0.85714 | 0.80541 | 4.78119 | 4.68331 | 4.87765 | 560 |
| ba | answer_direction | use | 8 | 1 | test | 0.04286 | 0.47923 | 0.33493 | 0.20764 | 0.45573 | 560 |
| ba | das_binding | use | 8 | 1 | test | 1.00000 | 0.47923 | 9.00910 | 8.93261 | 9.08937 | 560 |
| ba | mean_difference | use | 8 | 1 | test | 0.76786 | 0.71043 | 4.15042 | 4.03421 | 4.26161 | 560 |
| ba | noop | use | 8 | 1 | test | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 560 |
| ba | random_norm | use | 8 | 1536 | test | 0.01786 | 0.50932 | 0.71710 | 0.65036 | 0.78996 | 560 |
| ba | whole_state | use | 8 | 4096 | test | 0.87857 | 0.80559 | 4.79914 | 4.69366 | 4.90328 | 560 |

## Do not claim

- that a relevance shift shows the model USES the binding — this is an attribution of the model's own score, it intervenes on nothing, and causal use is what E13/R10's DAS interchange tests
- that the size of a relevance shift is comparable to the size of a DAS effect; one is a share of an answer score, the other a rate of answer change under an edit
- that the lens attributes relevance to pattern formation — the attn-rule detaches q and k, so 'attend to the right definition' is precisely the mechanism this instrument cannot see (src/models/lrp.py)
- anything about real code, other languages, or model families outside the two DeepSeeks the R-lens rules match
- a layer profile as a claim about where binding is COMPUTED; it is where the answer's attribution is redistributed

