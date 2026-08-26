# E16 — R-lens attribution on the binding counterfactual (deepseek-coder-1.3b)

## What this experiment asks

When the binding of a variable use changes and **exactly one token** of the program changes with it, does the model's own attribution of its answer move from the definition that just went out of scope to the one that just came into scope?

**Verdict: `output_token_artifact`**

A shift is present but the two arms disagree in sign, which is the signature of an artifact of which output token the relevance was taken for rather than of the binding. The fixed_a/fixed_b conditions are the rows to read next.

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

- **PASS** `H0` (the four-program factorial verifies: invariants, alignment, execution truth) — 1.0000 of 200 bases agree with an independent interpreter and satisfy every invariant, including the arm crossing that makes the held-out test a falsification (threshold 0.999). Per check: semantics_agree 1.0000, arms_crossed 1.0000, mutation_distance_ok 1.0000, anchors_ordered 1.0000, values_distinct 1.0000, tokens_distinct 1.0000
- **FAIL** `H1` (the model returns the correctly bound variable (behavioural accuracy)) — overall 0.809 [0.779, 0.838] against 0.85; weakest cell ab_target 0.571 against 0.75
- **PASS** `H6` (the relevance readout is mechanically sound: the LRP rules actually installed so relevance conserves, the token roles partition every token exactly once, the per-role deltas close to the difference of the two conservation ratios, every binding_flip pair differs at exactly one measured token index, the fixed-target conditions really do score both members at one token, and every declared cell exists) — 9600 readings and 19200 paired contrasts over 4 contrasts x 6 layers x 4 target conditions; median |rho-1| 1.56e-07; conserving layers [0, 3, 7, 11, 15, 19]; LRP rules bound {'ln': 49, 'mlp': 24, 'attn': 24}

Not run for this model: `H2`, `H3`, `H4`, `H5`. A gate that was never run is not a failed gate; stage 140 requires H0 only.

H6 is **mechanical**: a null redistribution passes it. It gates whether the numbers are relevance at all, never whether they are interesting.

## The reported cell

- layer **7**, selected on **calibration** by the rule in `binding_relevance.select_cell`, read on split `test`
- conserving layers: [0, 3, 7, 11, 15, 19] (tolerance |rho-1| <= 0.25)
- declared thresholds: sign consistency 0.7, p < 0.05

| check | holds |
|---|---|
| rules_installed_and_conserving | yes |
| shift_consistent | yes |
| above_permutation_control | no |
| above_sign_test | yes |
| arms_agree | no |
| same_binding_controls_quiet | yes |
| statistic_is_token_identical | yes |

### Table 1 — the headline statistic, every contrast and layer

`expect` is declared in `binding_relevance.CONTRASTS` before the run: `shift` for the two binding flips, `null` for the two same-binding controls where the bound token moves the same way and the binding does not. `ci_lo`/`ci_hi` are a cluster bootstrap over bases — the same interval convention stage 106 reports DAS with.

| contrast | expect | layer | n_pairs | n_bases | mean_delta | ci_lo | ci_hi | median_delta | cohens_d | sign_consistency | n_nonzero | sign_test_p | permutation_p | permutation_effect_size | degenerate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flip_ab | shift | 0 | 140 | 140 | -0.36109 | -1.04422 | 0.12849 | 0.07134 | -0.09794 | 0.72143 | 140 | 0.00000 | 0.43200 | -1.13739 | 0 |
| flip_ab | shift | 3 | 140 | 140 | -0.15483 | -0.66884 | 0.22604 | 0.10407 | -0.05530 | 0.78571 | 140 | 0.00000 | 0.50800 | -0.65749 | 0 |
| flip_ab | shift | 7 | 140 | 140 | -0.15219 | -0.76874 | 0.31283 | 0.12213 | -0.04514 | 0.81429 | 140 | 0.00000 | 0.55200 | -0.54755 | 0 |
| flip_ab | shift | 11 | 140 | 140 | 0.10297 | -0.30086 | 0.51866 | 0.09186 | 0.04254 | 0.81429 | 140 | 0.00000 | 0.67200 | 0.44550 | 0 |
| flip_ab | shift | 15 | 140 | 140 | 0.10864 | -0.35421 | 0.56129 | 0.12446 | 0.03977 | 0.82857 | 140 | 0.00000 | 0.68000 | 0.42104 | 0 |
| flip_ab | shift | 19 | 140 | 140 | 0.39505 | 0.09711 | 0.76101 | 0.02219 | 0.19630 | 0.65000 | 140 | 0.00049 | 0.00400 | 2.21425 | 0 |
| flip_ba | shift | 0 | 140 | 140 | 0.10013 | -0.04469 | 0.26864 | 0.06979 | 0.10781 | 0.77143 | 140 | 0.00000 | 0.21400 | 1.24039 | 0 |
| flip_ba | shift | 3 | 140 | 140 | 0.10832 | 0.02543 | 0.20971 | 0.09171 | 0.19456 | 0.80714 | 140 | 0.00000 | 0.01600 | 2.29259 | 0 |
| flip_ba | shift | 7 | 140 | 140 | 0.10521 | -0.01827 | 0.23419 | 0.12286 | 0.14157 | 0.85714 | 140 | 0.00000 | 0.07800 | 1.66003 | 0 |
| flip_ba | shift | 11 | 140 | 140 | 0.04834 | -0.11443 | 0.19720 | 0.08897 | 0.05264 | 0.81429 | 140 | 0.00000 | 0.54200 | 0.55476 | 0 |
| flip_ba | shift | 15 | 140 | 140 | 0.04166 | -0.13333 | 0.19784 | 0.10433 | 0.04257 | 0.77143 | 140 | 0.00000 | 0.63800 | 0.44011 | 0 |
| flip_ba | shift | 19 | 140 | 140 | 0.00306 | -0.23212 | 0.29089 | 0.00364 | 0.00198 | 0.51429 | 140 | 0.79996 | 0.98800 | 0.01808 | 0 |
| same_outer | no_shift | 0 | 140 | 140 | -0.04678 | -0.14308 | 0.02622 | 0.03301 | -0.09187 | 0.55000 | 140 | 0.27184 | 0.30600 | -1.05101 | 0 |
| same_outer | no_shift | 3 | 140 | 140 | -0.02029 | -0.07197 | 0.01801 | 0.02731 | -0.07610 | 0.54286 | 140 | 0.35258 | 0.36800 | -0.86295 | 0 |
| same_outer | no_shift | 7 | 140 | 140 | -0.01236 | -0.05481 | 0.02062 | 0.02671 | -0.05477 | 0.50714 | 140 | 0.93269 | 0.53800 | -0.61207 | 0 |
| same_outer | no_shift | 11 | 140 | 140 | 0.00298 | -0.02606 | 0.03065 | 0.01018 | 0.01700 | 0.52143 | 140 | 0.67276 | 0.84400 | 0.25115 | 0 |
| same_outer | no_shift | 15 | 140 | 140 | 0.00979 | -0.01570 | 0.03961 | 0.00266 | 0.05774 | 0.50714 | 140 | 0.93269 | 0.49800 | 0.70899 | 0 |
| same_outer | no_shift | 19 | 140 | 140 | 0.02595 | -0.02108 | 0.08373 | 0.00002 | 0.08345 | 0.50000 | 140 | 1.00000 | 0.31800 | 0.96744 | 0 |
| same_inner | no_shift | 0 | 140 | 140 | -0.41443 | -1.11680 | 0.10825 | -0.01442 | -0.10962 | 0.47143 | 140 | 0.55427 | 0.31400 | -1.27057 | 0 |
| same_inner | no_shift | 3 | 140 | 140 | -0.24286 | -0.76505 | 0.15134 | 0.00595 | -0.08483 | 0.50000 | 140 | 1.00000 | 0.42600 | -0.99975 | 0 |
| same_inner | no_shift | 7 | 140 | 140 | -0.24504 | -0.87973 | 0.24677 | -0.01413 | -0.07046 | 0.49286 | 140 | 0.93269 | 0.46000 | -0.83887 | 0 |
| same_inner | no_shift | 11 | 140 | 140 | 0.05165 | -0.36445 | 0.47631 | -0.06055 | 0.01979 | 0.45714 | 140 | 0.35258 | 0.85000 | 0.20416 | 0 |
| same_inner | no_shift | 15 | 140 | 140 | 0.05719 | -0.43182 | 0.52548 | 0.01609 | 0.01952 | 0.51429 | 140 | 0.79996 | 0.84000 | 0.20696 | 0 |
| same_inner | no_shift | 19 | 140 | 140 | 0.36604 | -0.05318 | 0.79700 | 0.02479 | 0.14717 | 0.54286 | 140 | 0.35258 | 0.06800 | 1.67613 | 0 |

### Table 2 — the layer profile of the two binding flips

This is where the attribution is redistributed, not where binding is computed. Compare the depth with DAS's chosen layer in the comparison section, not the magnitudes.

| layer | contrast | mean_delta | median_delta | sign_consistency | sign_test_p | permutation_p | n_pairs | median_abs_rho_minus_one |
|---|---|---|---|---|---|---|---|---|
| 0 | flip_ab | -0.36109 | 0.07134 | 0.72143 | 0.00000 | 0.43200 | 140 | 0.00000 |
| 0 | flip_ba | 0.10013 | 0.06979 | 0.77143 | 0.00000 | 0.21400 | 140 | 0.00000 |
| 3 | flip_ab | -0.15483 | 0.10407 | 0.78571 | 0.00000 | 0.50800 | 140 | 0.00000 |
| 3 | flip_ba | 0.10832 | 0.09171 | 0.80714 | 0.00000 | 0.01600 | 140 | 0.00000 |
| 7 | flip_ab | -0.15219 | 0.12213 | 0.81429 | 0.00000 | 0.55200 | 140 | 0.00000 |
| 7 | flip_ba | 0.10521 | 0.12286 | 0.85714 | 0.00000 | 0.07800 | 140 | 0.00000 |
| 11 | flip_ab | 0.10297 | 0.09186 | 0.81429 | 0.00000 | 0.67200 | 140 | 0.00000 |
| 11 | flip_ba | 0.04834 | 0.08897 | 0.81429 | 0.00000 | 0.54200 | 140 | 0.00000 |
| 15 | flip_ab | 0.10864 | 0.12446 | 0.82857 | 0.00000 | 0.68000 | 140 | 0.00000 |
| 15 | flip_ba | 0.04166 | 0.10433 | 0.77143 | 0.00000 | 0.63800 | 140 | 0.00000 |
| 19 | flip_ab | 0.39505 | 0.02219 | 0.65000 | 0.00049 | 0.00400 | 140 | 0.00000 |
| 19 | flip_ba | 0.00306 | 0.00364 | 0.51429 | 0.79996 | 0.98800 | 140 | 0.00000 |

### Table 3 — the output-token control: do the arms agree?

Under `bound` the scored token moves v_a -> v_b in `flip_ab` and v_b -> v_a in `flip_ba`. An artifact of which token the relevance is taken for must **reverse sign** between the arms; a binding effect must not. `arm_ratio` near +1 is agreement, negative is the artifact signature. This is the same crossing stage 106 reads DAS's `answer_direction` control on.

| layer | mean_delta_ab | mean_delta_ba | median_delta_ab | median_delta_ba | sign_consistency_ab | sign_consistency_ba | signs_agree | arm_ratio | both_significant_sign |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.36109 | 0.10013 | 0.07134 | 0.06979 | 0.72143 | 0.77143 | 0 | -0.27729 | 1 |
| 3 | -0.15483 | 0.10832 | 0.10407 | 0.09171 | 0.78571 | 0.80714 | 0 | -0.69961 | 1 |
| 7 | -0.15219 | 0.10521 | 0.12213 | 0.12286 | 0.81429 | 0.85714 | 0 | -0.69128 | 1 |
| 11 | 0.10297 | 0.04834 | 0.09186 | 0.08897 | 0.81429 | 0.81429 | 1 | 0.46947 | 1 |
| 15 | 0.10864 | 0.04166 | 0.12446 | 0.10433 | 0.82857 | 0.77143 | 1 | 0.38346 | 1 |
| 19 | 0.39505 | 0.00306 | 0.02219 | 0.00364 | 0.65000 | 0.51429 | 1 | 0.00775 | 0 |

### Table 4 — the output-token control, part two: the same token in both members

`fixed_a` and `fixed_b` score BOTH members at literally the same token id, so the output token is removed from the contrast entirely. They cost no extra backward pass: each program is already read at both candidate tokens. If the shift under `bound` were about the scored token, these rows would be flat.

| contrast | target_condition | same_target_token | mean_delta | ci_lo | ci_hi | median_delta | sign_consistency | sign_test_p | permutation_p | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| flip_ab | bound | 0 | -0.15219 | -0.76874 | 0.31283 | 0.12213 | 0.81429 | 0.00000 | 0.55200 | 140 |
| flip_ab | fixed_a | 1 | -0.03906 | -0.18842 | 0.10159 | -0.04451 | 0.32143 | 0.00003 | 0.66400 | 140 |
| flip_ab | fixed_b | 1 | -0.46885 | -1.23773 | 0.11495 | 0.03267 | 0.65000 | 0.00049 | 0.23400 | 140 |
| flip_ab | other | 0 | -0.35572 | -0.84841 | -0.04981 | -0.14165 | 0.12857 | 0.00000 | 0.01400 | 140 |
| flip_ba | bound | 0 | 0.10521 | -0.01827 | 0.23419 | 0.12286 | 0.85714 | 0.00000 | 0.07800 | 140 |
| flip_ba | fixed_a | 1 | 2.12418 | -0.11449 | 6.47066 | 0.01722 | 0.57857 | 0.07555 | 0.58200 | 140 |
| flip_ba | fixed_b | 1 | -4.82199 | -16.91328 | 1.93145 | -0.07287 | 0.27143 | 0.00000 | 0.79000 | 140 |
| flip_ba | other | 0 | -2.80302 | -16.90705 | 6.52579 | -0.18159 | 0.14286 | 0.00000 | 0.88800 | 140 |
| same_outer | bound | 0 | -0.01236 | -0.05481 | 0.02062 | 0.02671 | 0.50714 | 0.93269 | 0.53800 | 140 |
| same_outer | fixed_a | 1 | -2.03134 | -6.33731 | 0.14321 | 0.10738 | 0.68571 | 0.00001 | 1.00000 | 140 |
| same_outer | fixed_b | 1 | -0.32903 | -0.76283 | -0.10350 | -0.11438 | 0.34286 | 0.00025 | 0.00000 | 140 |
| same_outer | other | 0 | -2.34800 | -6.83457 | 0.01578 | 0.02870 | 0.52857 | 0.55427 | 0.22600 | 140 |
| same_inner | bound | 0 | -0.24504 | -0.87973 | 0.24677 | -0.01413 | 0.49286 | 0.93269 | 0.46000 | 140 |
| same_inner | fixed_a | 1 | -0.13190 | -0.33838 | 0.05742 | -0.17605 | 0.27143 | 0.00000 | 0.20400 | 140 |
| same_inner | fixed_b | 1 | 4.68216 | -2.08617 | 17.01424 | 0.24950 | 0.74286 | 0.00000 | 0.86600 | 140 |
| same_inner | other | 0 | 4.79530 | -1.96268 | 16.89362 | 0.03639 | 0.54286 | 0.35258 | 0.81800 | 140 |

### Table 5 — every role at the reported cell

`mean_delta` is the paired change in a role's share of the model's answer. The column sums to ~0 by conservation: whatever one role gains, another loses. `token_identical` marks the roles whose tokens do not change; `inner_def_name` is the one that does, and it is reported rather than hidden.

| role | token_identical | n_pairs | mean_frac_from | mean_frac_to | median_delta | mean_delta | ci_lo | ci_hi | sign_consistency | sign_test_p | permutation_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| suffix | 1 | 140 | 0.41533 | -0.28193 | 0.04500 | -0.69727 | -1.51324 | -0.05301 | 0.67857 | 0.00003 | 0.06400 |
| outer_def_name | 1 | 140 | 0.23073 | 0.09007 | 0.01983 | -0.14066 | -0.32816 | 0.00787 | 0.74286 | 0.00000 | 0.10600 |
| inner_def_value | 1 | 140 | 0.08422 | 0.01483 | 0.07312 | -0.06940 | -0.33756 | 0.12545 | 0.85000 | 0.00000 | 0.51200 |
| use_site | 1 | 140 | 0.02031 | 0.00864 | -0.00695 | -0.01167 | -0.02418 | 0.00086 | 0.35714 | 0.00091 | 0.05800 |
| return_kw | 1 | 140 | 0.00221 | 0.03960 | -0.00278 | 0.03739 | 0.00120 | 0.08402 | 0.40714 | 0.03424 | 0.09000 |
| inner_def_name | 0 | 140 | 0.00437 | 0.05158 | -0.00992 | 0.04721 | -0.01739 | 0.13641 | 0.32143 | 0.00003 | 0.40200 |
| signature | 1 | 140 | 0.07142 | 0.16729 | -0.01659 | 0.09587 | -0.00514 | 0.22556 | 0.24286 | 0.00000 | 0.11200 |
| outer_def_value | 1 | 140 | 0.09642 | 0.31987 | -0.07091 | 0.22345 | -0.13922 | 0.72255 | 0.13571 | 0.00000 | 0.49200 |
| other | 1 | 140 | 0.07498 | 0.59005 | -0.03195 | 0.51507 | 0.02700 | 1.13068 | 0.35714 | 0.00091 | 0.08600 |

The composites below are **sums of the rows above**, so they do not add to zero and are listed separately rather than mixed in. `binding_shift` and `binding_shift_identical` are differences of two composites; only the second is made entirely of token-identical spans, which is why it is the headline.

| role | token_identical | n_pairs | median_delta | mean_delta | ci_lo | ci_hi | sign_consistency | sign_test_p | permutation_p |
|---|---|---|---|---|---|---|---|---|---|
| binding_shift_identical | 1 | 140 | 0.12213 | -0.15219 | -0.76874 | 0.31283 | 0.81429 | 0.00000 | 0.55200 |
| binding_shift | 0 | 140 | 0.11290 | -0.10498 | -0.63181 | 0.31128 | 0.80714 | 0.00000 | 0.64800 |
| inner_def_identical | 1 | 140 | 0.07312 | -0.06940 | -0.33756 | 0.12545 | 0.85000 | 0.00000 | 0.51200 |
| inner_def | 0 | 140 | 0.06406 | -0.02219 | -0.20603 | 0.11738 | 0.84286 | 0.00000 | 0.75600 |
| both_defs | 0 | 140 | 0.01619 | 0.06061 | -0.11419 | 0.24218 | 0.62857 | 0.00296 | 0.49400 |
| outer_def | 1 | 140 | -0.05144 | 0.08280 | -0.20398 | 0.43665 | 0.20714 | 0.00000 | 0.63600 |

### Table 6 — the same statistic on pairs the model actually answers

**123 of 200** bases have the model answering BOTH members of `flip_ab` correctly. H1 is not a prerequisite for this stage — it fails on deepseek-coder-1.3b — so the shift is reported on all pairs above and on that subset here.

| contrast | expect | layer | n_pairs | mean_delta | ci_lo | ci_hi | median_delta | sign_consistency | sign_test_p | permutation_p |
|---|---|---|---|---|---|---|---|---|---|---|
| flip_ab | shift | 0 | 80 | 0.05377 | -0.05879 | 0.21375 | 0.05181 | 0.67500 | 0.00232 | 0.59800 |
| flip_ab | shift | 3 | 80 | 0.18245 | 0.02728 | 0.39258 | 0.08168 | 0.76250 | 0.00000 | 0.01200 |
| flip_ab | shift | 7 | 80 | 0.25042 | 0.02763 | 0.56346 | 0.11133 | 0.81250 | 0.00000 | 0.02400 |
| flip_ab | shift | 11 | 80 | 0.39098 | 0.02463 | 0.95777 | 0.08542 | 0.77500 | 0.00000 | 0.03200 |
| flip_ab | shift | 15 | 80 | 0.43321 | 0.04738 | 1.02115 | 0.10494 | 0.77500 | 0.00000 | 0.00600 |
| flip_ab | shift | 19 | 80 | 0.31799 | 0.01935 | 0.73746 | 0.01464 | 0.56250 | 0.31431 | 0.06200 |
| flip_ba | shift | 0 | 93 | 0.08513 | -0.12316 | 0.30671 | 0.03751 | 0.69892 | 0.00016 | 0.48200 |
| flip_ba | shift | 3 | 93 | 0.07378 | -0.03357 | 0.20795 | 0.06545 | 0.74194 | 0.00000 | 0.32600 |
| flip_ba | shift | 7 | 93 | 0.03662 | -0.12358 | 0.20461 | 0.09107 | 0.80645 | 0.00000 | 0.67200 |
| flip_ba | shift | 11 | 93 | -0.03951 | -0.27286 | 0.17553 | 0.05546 | 0.73118 | 0.00001 | 0.71200 |
| flip_ba | shift | 15 | 93 | -0.05989 | -0.31296 | 0.17060 | 0.06546 | 0.66667 | 0.00171 | 0.63200 |
| flip_ba | shift | 19 | 93 | -0.03313 | -0.36241 | 0.38805 | -0.01207 | 0.44086 | 0.29973 | 0.89600 |
| same_outer | no_shift | 0 | 140 | -0.04678 | -0.14308 | 0.02622 | 0.03301 | 0.55000 | 0.27184 | 0.30600 |
| same_outer | no_shift | 3 | 140 | -0.02029 | -0.07197 | 0.01801 | 0.02731 | 0.54286 | 0.35258 | 0.36800 |
| same_outer | no_shift | 7 | 140 | -0.01236 | -0.05481 | 0.02062 | 0.02671 | 0.50714 | 0.93269 | 0.53800 |
| same_outer | no_shift | 11 | 140 | 0.00298 | -0.02606 | 0.03065 | 0.01018 | 0.52143 | 0.67276 | 0.84400 |
| same_outer | no_shift | 15 | 140 | 0.00979 | -0.01570 | 0.03961 | 0.00266 | 0.50714 | 0.93269 | 0.49800 |
| same_outer | no_shift | 19 | 140 | 0.02595 | -0.02108 | 0.08373 | 0.00002 | 0.50000 | 1.00000 | 0.31800 |
| same_inner | no_shift | 0 | 58 | 0.08855 | -0.01499 | 0.23741 | 0.06188 | 0.58621 | 0.23705 | 0.17600 |
| same_inner | no_shift | 3 | 58 | 0.05137 | -0.01574 | 0.13003 | 0.04500 | 0.55172 | 0.51184 | 0.18800 |
| same_inner | no_shift | 7 | 58 | 0.12137 | -0.01695 | 0.28907 | 0.04028 | 0.58621 | 0.23705 | 0.15800 |
| same_inner | no_shift | 11 | 58 | 0.05576 | -0.18364 | 0.30891 | 0.06218 | 0.58621 | 0.23705 | 0.57400 |
| same_inner | no_shift | 15 | 58 | 0.05189 | -0.19024 | 0.28752 | 0.08011 | 0.58621 | 0.23705 | 0.59200 |
| same_inner | no_shift | 19 | 58 | -0.14019 | -0.71000 | 0.21595 | 0.03488 | 0.60345 | 0.14801 | 0.96400 |

### Table 7 — the token-identity control, measured

`as_designed` is the fraction of pairs with the expected number of differing token indices (1 for a binding flip, 2 for a same-binding control, since both value literals move). `use_token_identical` must be 1.0 everywhere or a relevance change at the use site could be the token.

| contrast | contrast_kind | n | same_length | mean_differing | as_designed | use_token_identical |
|---|---|---|---|---|---|---|
| flip_ab | binding_flip | 200 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| flip_ba | binding_flip | 200 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| same_inner | same_binding | 200 | 1.00000 | 2.00000 | 1.00000 | 1.00000 |
| same_outer | same_binding | 200 | 1.00000 | 2.00000 | 1.00000 | 1.00000 |

### Table 8 — the mismatched-pair control

Members drawn from **different bases** with the orientation kept. The permutation null keeps the pairing and destroys the orientation; this keeps the orientation and destroys the base matching, so what it can falsify is 'the redistribution is specific to this pairing'.

| contrast | layer | mean_delta | median_delta | sign_consistency | n |
|---|---|---|---|---|---|
| flip_ab_mismatched | 7 | -0.10339 | 0.07389 | 0.66000 | 200 |
| flip_ba_mismatched | 7 | 0.06408 | 0.11779 | 0.69000 | 200 |

### Table 9 — conservation, the validity condition

The fraction reading is licensed only where relevance conserves. This is measured per (layer, target mode) on this run's own programs, not inherited from E14 gate R2.

| layer | target_mode | n_readings | median_rho | median_abs_rho_minus_one | max_abs_rho_minus_one | conserving |
|---|---|---|---|---|---|---|
| 0 | bound | 800 | 1.00000 | 0.00000 | 0.00001 | 1 |
| 0 | other | 800 | 1.00000 | 0.00000 | 0.00114 | 1 |
| 3 | bound | 800 | 1.00000 | 0.00000 | 0.00002 | 1 |
| 3 | other | 800 | 1.00000 | 0.00000 | 0.00126 | 1 |
| 7 | bound | 800 | 1.00000 | 0.00000 | 0.00001 | 1 |
| 7 | other | 800 | 1.00000 | 0.00000 | 0.00121 | 1 |
| 11 | bound | 800 | 1.00000 | 0.00000 | 0.00001 | 1 |
| 11 | other | 800 | 1.00000 | 0.00000 | 0.00118 | 1 |
| 15 | bound | 800 | 1.00000 | 0.00000 | 0.00002 | 1 |
| 15 | other | 800 | 1.00000 | 0.00000 | 0.00122 | 1 |
| 19 | bound | 800 | 1.00000 | 0.00000 | 0.00002 | 1 |
| 19 | other | 800 | 1.00000 | 0.00000 | 0.00117 | 1 |

### Table 10 — per-position deltas at the reported cell

E15-D could not produce this table: its pair members are not token-aligned. Here all four cells share a token length and differ at one index, so this shows whether the role aggregation is hiding a single position doing all the work.

| position | role_to | mean_delta | median_delta | sign_consistency | n |
|---|---|---|---|---|---|
| 0 | outer_def_name | -0.08239 | 0.02399 | 0.75500 | 200 |
| 1 | other | 0.12368 | -0.02086 | 0.21500 | 200 |
| 2 | outer_def_value | 0.15258 | -0.07062 | 0.16000 | 200 |
| 3 | other | -0.07502 | 0.04147 | 0.84000 | 200 |
| 4 | signature | 0.20708 | -0.02143 | 0.23000 | 200 |
| 5 | signature | -0.03518 | 0.00771 | 0.81000 | 200 |
| 6 | signature | -0.11775 | -0.00081 | 0.49000 | 200 |
| 7 | other | 0.14397 | -0.01906 | 0.24500 | 200 |
| 8 | other | 0.05598 | -0.00984 | 0.19500 | 200 |
| 9 | inner_def_name | 0.02667 | -0.01120 | 0.29500 | 200 |
| 10 | other | 0.03514 | -0.01424 | 0.31500 | 200 |
| 11 | inner_def_value | -0.03320 | 0.07168 | 0.85000 | 200 |
| 12 | other | 0.03087 | 0.00115 | 0.54500 | 200 |
| 13 | other | 0.02184 | -0.01000 | 0.17000 | 200 |
| 14 | return_kw | 0.02519 | -0.00365 | 0.38000 | 200 |
| 15 | use_site | -0.00918 | -0.00605 | 0.40000 | 200 |
| 16 | suffix | 0.06631 | -0.01574 | 0.24500 | 200 |
| 17 | suffix | 0.06828 | -0.01699 | 0.18000 | 200 |
| 18 | suffix | 0.00726 | -0.00863 | 0.13000 | 200 |
| 19 | suffix | -0.07542 | 0.01731 | 0.78000 | 200 |
| 20 | suffix | -0.53673 | 0.07520 | 0.75500 | 200 |

## Observational R-lens versus causal DAS on the same corpus

DAS (stage 106) reports at site `use`, layer **not recorded**; this stage reports at layer **7**.

| | R-lens (this stage, E16) | DAS (stage 106, R10) |
|---|---|---|
| what is done to the model | nothing | a rank-1 subspace at the use anchor is replaced with the donor's |
| what is read | how the answer score decomposes over input positions | whether the emitted token becomes the installed binding's value |
| licenses | a statement about **attribution** | a statement about **causal transport** at that site, layer and construction |
| reported layer | 7 | n/a |
| both arms | see table 3 | 100.0% / 100.0% (`says_installed`) |
| effect size units | share of the answer score | rate of answer change |

**The units do not convert.** No ratio between the two is computed anywhere in this pipeline. What the two results can jointly support is a conjunction, not a chain: at this site the binding is causally transportable (DAS) *and* the attribution redistributes with it (R-lens), or it is transportable and the attribution does not move — which would itself be the more interesting finding, because it would show attribution and use coming apart on a corpus where the causal fact is settled.

### Table 11 — E13's causal numbers, as stage 106 wrote them

_not run_

## Do not claim

- that a relevance shift shows the model USES the binding — this is an attribution of the model's own score, it intervenes on nothing, and causal use is what E13/R10's DAS interchange tests
- that the size of a relevance shift is comparable to the size of a DAS effect; one is a share of an answer score, the other a rate of answer change under an edit
- that the lens attributes relevance to pattern formation — the attn-rule detaches q and k, so 'attend to the right definition' is precisely the mechanism this instrument cannot see (src/models/lrp.py)
- anything about real code, other languages, or model families outside the two DeepSeeks the R-lens rules match
- a layer profile as a claim about where binding is COMPUTED; it is where the answer's attribution is redistributed

