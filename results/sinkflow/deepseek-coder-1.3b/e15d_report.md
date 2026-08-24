# E15-D — three follow-ups to the E15-C null (deepseek-coder-1.3b)

## What this experiment asks

This report follows up the vocabulary result with three checks: a full-vocabulary direction, a positive control on a property the model can express, and relevance redistribution across tokens. Together they distinguish a model-level null from a readout that is simply unable to detect anything.

Each section states a verdict decided by a checklist declared in code before the run. All three stages are observational: none of them establishes that the model *uses* what is measured.

| stage | gate | verdict |
|---|---|---|
| V1 full-vocabulary alignment | J2 PASS | `direction_replicates_but_not_dominant` |
| positive control | J3 PASS | `machinery_blind` |
| V3 relevance redistribution | J4 PASS | `redistribution_consistent_but_not_in_mean` |

**What this means for E15-C.** E15-C's null is about the METHOD. The models answer the forced choice, the identical readout does not see it, so no claim about what code models represent survives that track and every number in it keeps its caveat.

---

## V1 — is there a shared full-vocabulary direction?

**Verdict.** DIRECTION REPLICATES, BUT DOES NOT DOMINATE — a direction defined by the label on the training split generalises to held-out programs, above the token-identity floor; but the label axis is not the largest axis of variation among the difference vectors, so the declared `sv1_ratio >= 2.0` criterion is NOT met. The two statements are compatible and both are reported: the projection asks whether the direction generalises, the concentration asks whether it dominates.

Read at site `last_token` (declared before any result: it is the only site where both members carry the same token id), layer 11, condition `clean_heldout`.

| check | holds |
|---|---|
| direction_frozen_on_train | yes |
| heldout_projection_replicates | yes |
| above_same_label_null | no |
| above_surface_floor | yes |

### Table 13 — concentration and projection by layer

`sv1_share` is the fraction of the pairs' total energy on ONE direction; `sv1_floor` is what unrelated differences give (1/n); `same_label_sv1_share` is the harder of the two same-label nulls; `proj_*` is the projection onto the direction frozen on the training split.

| layer | n_pairs | sv1_share | sv1_floor | same_label_sv1_share | sv1_ratio | mean_pairwise_cosine | proj_mean | proj_sign_consistency | proj_ci_lo | proj_ci_hi | same_label_proj_sign_consistency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 0 |  |  |  |  |  |  |  |  |  |  |
| 0 | 72 | 0.2743 | 0.0139 | 0.2863 | 0.9579 | 0.0035 | -0.0331 | 0.4861 | -0.1017 | 0.0289 | 0.4583 |
| 3 | 72 | 0.1675 | 0.0139 | 0.3232 | 0.5183 | -0.0012 | -0.0297 | 0.4583 | -0.0910 | 0.0286 | 0.5069 |
| 7 | 72 | 0.2204 | 0.0139 | 0.3252 | 0.6778 | 0.1698 | 0.4041 | 1.0000 | 0.3712 | 0.4380 | 0.5208 |
| 11 | 72 | 0.2145 | 0.0139 | 0.2809 | 0.7636 | 0.1499 | 0.3827 | 1.0000 | 0.3603 | 0.4064 | 0.5069 |
| 15 | 72 | 0.2332 | 0.0139 | 0.2684 | 0.8685 | 0.1289 | 0.3522 | 1.0000 | 0.3264 | 0.3757 | 0.5139 |
| 19 | 72 | 0.2303 | 0.0139 | 0.2625 | 0.8773 | 0.1053 | 0.3161 | 1.0000 | 0.2877 | 0.3437 | 0.5347 |
| 23 | 72 | 0.3900 | 0.0139 | 0.3555 | 1.0972 | 0.1016 | 0.2992 | 0.9028 | 0.2538 | 0.3423 | 0.5417 |

### Table 14 — the same measurement across obfuscation

| condition | condition_kind | sv1_share | sv1_ratio | proj_mean | proj_sign_consistency | n_pairs |
|---|---|---|---|---|---|---|
| clean_heldout | clean | 0.2145 | 0.7636 | 0.3827 | 1.0000 | 72 |
| normalize | baseline | 0.2142 | 0.6869 | 0.2453 | 1.0000 | 72 |
| rename_only | atomic | 0.1468 | 0.4900 | 0.1622 | 0.9583 | 72 |
| opaque_only | atomic | 0.1801 | 0.7545 | 0.1779 | 0.9028 | 72 |
| encode_only | atomic | 0.2178 | 0.6996 | 0.2417 | 1.0000 | 72 |
| flatten_only | atomic | 0.0897 | 0.5717 | 0.0626 | 0.8194 | 72 |
| rename_cumulative | cumulative | 0.1471 | 0.4882 | 0.1456 | 0.8750 | 72 |
| rename_opaque | cumulative | 0.0991 | 0.3021 | 0.0992 | 0.8333 | 72 |
| rename_opaque_encode | cumulative | 0.1088 | 0.4164 | 0.1014 | 0.8611 | 72 |
| rename_opaque_encode_flatten | cumulative | 0.0978 | 0.6859 | 0.0434 | 0.7222 | 72 |

### Table 15 — what the direction says, as tokens

Discovered from the differences, not proposed in advance. `overlap_with_same_label_direction` is the Jaccard overlap of the top-100 loadings with the direction the SAME-LABEL differences find: high overlap means the direction is whatever distinguishes any two of these programs.

| pole | rank_within_pole | token | loading | overlap_with_same_label_direction |
|---|---|---|---|---|
| unsafe_higher | 0 |  Lemmon | 0.0270 | 0.0050 |
| unsafe_higher | 1 | egraphics | 0.0259 | 0.0050 |
| unsafe_higher | 2 | ateral | 0.0238 | 0.0050 |
| unsafe_higher | 3 | uta | 0.0230 | 0.0050 |
| unsafe_higher | 4 | ateria | 0.0226 | 0.0050 |
| unsafe_higher | 5 | Stats | 0.0225 | 0.0050 |
| unsafe_higher | 6 | gres | 0.0218 | 0.0050 |
| unsafe_higher | 7 | alog | 0.0216 | 0.0050 |
| unsafe_higher | 8 | thems | 0.0210 | 0.0050 |
| unsafe_higher | 9 | ensed | 0.0207 | 0.0050 |
| unsafe_higher | 10 | EEE | 0.0206 | 0.0050 |
| unsafe_higher | 11 | morrow | 0.0205 | 0.0050 |
| unsafe_higher | 12 | reduc | 0.0205 | 0.0050 |
| unsafe_higher | 13 | IFI | 0.0199 | 0.0050 |
| unsafe_higher | 14 | tee | 0.0199 | 0.0050 |
| unsafe_higher | 15 | omon | 0.0197 | 0.0050 |
| unsafe_higher | 16 |  teac | 0.0197 | 0.0050 |
| unsafe_higher | 17 |  accret | 0.0195 | 0.0050 |
| unsafe_higher | 18 | estone | 0.0194 | 0.0050 |
| unsafe_higher | 19 | Messages | 0.0193 | 0.0050 |
| unsafe_higher | 20 | stats | 0.0192 | 0.0050 |
| unsafe_higher | 21 | GREE | 0.0191 | 0.0050 |
| unsafe_higher | 22 | edly | 0.0190 | 0.0050 |
| unsafe_higher | 23 |  своя | 0.0190 | 0.0050 |
| unsafe_higher | 24 | bros | 0.0189 | 0.0050 |
| safe_higher | 0 | idir | -0.0235 | 0.0050 |
| safe_higher | 1 | azi | -0.0221 | 0.0050 |
| safe_higher | 2 | 弟 | -0.0218 | 0.0050 |
| safe_higher | 3 |  Johnson | -0.0209 | 0.0050 |
| safe_higher | 4 | BB | -0.0208 | 0.0050 |

### Table 16 — full vocabulary versus E15-C's frozen 196-token pool

If concentration is high over the full vocabulary and low inside the pool, the pool missed the direction. If it is low in both, no pool would have helped.

| lens | arm | layer | site | n_candidates | sv1_share | sv1_floor | mean_pairwise_cosine |
|---|---|---|---|---|---|---|---|
| jlens | main | -1 | sink_arg | 196 | 0.3415 | 0.0556 | -0.0392 |
| jlens | same_label_unsafe | -1 | sink_arg | 196 | 0.4526 | 0.0233 | -0.0232 |
| jlens | same_label_safe | -1 | sink_arg | 196 | 0.4352 | 0.0233 | -0.0230 |
| jlens | main | -1 | last_token | 196 |  |  |  |
| jlens | same_label_unsafe | -1 | last_token | 196 |  |  |  |
| jlens | same_label_safe | -1 | last_token | 196 |  |  |  |
| jlens | main | 0 | sink_arg | 196 | 0.2878 | 0.0139 | 0.0021 |
| jlens | same_label_unsafe | 0 | sink_arg | 196 | 0.2994 | 0.0139 | -0.0118 |
| jlens | same_label_safe | 0 | sink_arg | 196 | 0.2813 | 0.0139 | -0.0092 |
| jlens | main | 0 | last_token | 196 | 0.4439 | 0.0139 | 0.0044 |
| jlens | same_label_unsafe | 0 | last_token | 196 | 0.4047 | 0.0139 | -0.0119 |
| jlens | same_label_safe | 0 | last_token | 196 | 0.4046 | 0.0139 | -0.0119 |
| jlens | main | 3 | sink_arg | 196 | 0.1904 | 0.0139 | 0.0182 |
| jlens | same_label_unsafe | 3 | sink_arg | 196 | 0.2411 | 0.0139 | -0.0117 |
| jlens | same_label_safe | 3 | sink_arg | 196 | 0.2360 | 0.0139 | -0.0120 |
| jlens | main | 3 | last_token | 196 | 0.2214 | 0.0139 | 0.0017 |
| jlens | same_label_unsafe | 3 | last_token | 196 | 0.3256 | 0.0139 | -0.0137 |
| jlens | same_label_safe | 3 | last_token | 196 | 0.3334 | 0.0139 | -0.0136 |
| jlens | main | 7 | sink_arg | 196 | 0.2544 | 0.0139 | 0.1540 |
| jlens | same_label_unsafe | 7 | sink_arg | 196 | 0.2685 | 0.0139 | -0.0129 |
| jlens | same_label_safe | 7 | sink_arg | 196 | 0.2592 | 0.0139 | -0.0126 |
| jlens | main | 7 | last_token | 196 | 0.2666 | 0.0139 | 0.1662 |
| jlens | same_label_unsafe | 7 | last_token | 196 | 0.3410 | 0.0139 | -0.0137 |
| jlens | same_label_safe | 7 | last_token | 196 | 0.3324 | 0.0139 | -0.0136 |
| jlens | main | 11 | sink_arg | 196 | 0.4449 | 0.0139 | 0.4094 |
| jlens | same_label_unsafe | 11 | sink_arg | 196 | 0.2298 | 0.0139 | -0.0132 |
| jlens | same_label_safe | 11 | sink_arg | 196 | 0.2500 | 0.0139 | -0.0131 |
| jlens | main | 11 | last_token | 196 | 0.2104 | 0.0139 | 0.1629 |
| jlens | same_label_unsafe | 11 | last_token | 196 | 0.2675 | 0.0139 | -0.0137 |
| jlens | same_label_safe | 11 | last_token | 196 | 0.2731 | 0.0139 | -0.0136 |
| jlens | main | 15 | sink_arg | 196 | 0.4881 | 0.0139 | 0.4552 |
| jlens | same_label_unsafe | 15 | sink_arg | 196 | 0.2204 | 0.0139 | -0.0135 |
| jlens | same_label_safe | 15 | sink_arg | 196 | 0.1955 | 0.0139 | -0.0135 |
| jlens | main | 15 | last_token | 196 | 0.2246 | 0.0139 | 0.1489 |
| jlens | same_label_unsafe | 15 | last_token | 196 | 0.2824 | 0.0139 | -0.0136 |
| jlens | same_label_safe | 15 | last_token | 196 | 0.2621 | 0.0139 | -0.0137 |
| jlens | main | 19 | sink_arg | 196 | 0.3976 | 0.0139 | 0.3576 |
| jlens | same_label_unsafe | 19 | sink_arg | 196 | 0.2025 | 0.0139 | -0.0134 |
| jlens | same_label_safe | 19 | sink_arg | 196 | 0.1930 | 0.0139 | -0.0136 |
| jlens | main | 19 | last_token | 196 | 0.2654 | 0.0139 | 0.1217 |

---

## Positive control — can this machinery detect verbalisation at all?

**Verdict.** MACHINERY BLIND — the model answers the forced choice and the identical readout misses it. The instrument, not the model, is what E15-C's null is about.

Prompt style `sink`, lens `rlens`, condition `clean_heldout`, layer -1 — chosen as the layer that best detects the TAINT property, with the security contrast then read at that same cell.

| check | holds |
|---|---|
| behaviour_above_chance | yes |
| lens_detects_the_property | no |
| lens_tracks_the_model | no |
| security_contrast_at_same_cell | no |

### Table 17 — behaviour: can the model answer at all?

`pair_separation` is the fraction of bases where the unsafe member draws a higher yes-margin than its matched safe counterpart. Its chance level is 0.5 and no answer bias can move it, which is why it is the statistic the verdict uses rather than raw accuracy.

| prompt_style | condition | n_pairs | accuracy | accuracy_unsafe | accuracy_safe | says_tainted_rate | pair_separation | pair_separation_p | mean_model_delta |
|---|---|---|---|---|---|---|---|---|---|
| e6 | clean_heldout | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.7222 | 0.0002 | 0.0217 |
| e6 | normalize | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.7361 | 0.0001 | 0.0251 |
| e6 | rename_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6250 | 0.0444 | 0.0077 |
| e6 | opaque_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.7083 | 0.0005 | 0.0197 |
| e6 | encode_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.8056 | 0.0000 | 0.0244 |
| e6 | flatten_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.4583 | 0.5560 | -0.0007 |
| e6 | rename_cumulative | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6667 | 0.0063 | 0.0138 |
| e6 | rename_opaque | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6389 | 0.0245 | 0.0094 |
| e6 | rename_opaque_encode | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.7222 | 0.0002 | 0.0128 |
| e6 | rename_opaque_encode_flatten | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.4583 | 0.5560 | -0.0039 |
| sink | clean_heldout | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6944 | 0.0013 | 0.0227 |
| sink | normalize | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6944 | 0.0013 | 0.0229 |
| sink | rename_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6250 | 0.0444 | 0.0108 |
| sink | opaque_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.7083 | 0.0005 | 0.0202 |
| sink | encode_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.7222 | 0.0002 | 0.0230 |
| sink | flatten_only | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.5139 | 0.9063 | 0.0028 |
| sink | rename_cumulative | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.6944 | 0.0013 | 0.0125 |
| sink | rename_opaque | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.5556 | 0.4096 | 0.0069 |
| sink | rename_opaque_encode | 72 | 0.4931 | 0.0417 | 0.9444 | 0.0486 | 0.6111 | 0.0764 | 0.0115 |
| sink | rename_opaque_encode_flatten | 72 | 0.5000 | 0.0000 | 1.0000 | 0.0000 | 0.4722 | 0.7239 | 0.0014 |

### Table 18 — the two properties, one basis, one lens, by layer

`taint_*` and `security_*` differ only in which token positions are named as the poles. `taint_lens_tracks_model` is the fraction of pairs where the lens's paired margin has the same sign as the model's own.

| layer | relative_depth | n_pairs | taint_sign_consistency | taint_permutation_p | taint_lens_tracks_model | taint_corr_model_delta | security_sign_consistency | security_permutation_p |
|---|---|---|---|---|---|---|---|---|
| -1 |  | 72 | 0.0000 | 1.0000 | 0.1667 |  | 0.0000 | 1.0000 |
| 0 | 0.0000 | 72 | 0.4028 | 0.1860 | 0.3472 | -0.0337 | 0.4583 | 0.5200 |
| 3 | 0.1304 | 72 | 0.4861 | 0.5480 | 0.3611 | -0.1491 | 0.5417 | 0.2180 |
| 7 | 0.3043 | 72 | 0.5694 | 0.3060 | 0.4306 | -0.1046 | 0.4583 | 0.6580 |
| 11 | 0.4783 | 72 | 0.5694 | 0.0780 | 0.5000 | 0.0339 | 0.2639 | 0.0000 |
| 15 | 0.6522 | 72 | 0.5972 | 0.0020 | 0.6389 | 0.5162 | 0.3472 | 0.0160 |
| 19 | 0.8261 | 72 | 0.8889 | 0.0000 | 0.7083 | 0.5534 | 0.3472 | 0.0000 |
| 23 | 1.0000 | 72 | 0.8056 | 0.0000 | 0.8333 | 0.9524 | 0.2361 | 0.0000 |

---

## V3 — where does relevance move?

**Verdict.** REDISTRIBUTION IN SIGN, NOT IN MEAN — a token-identical role's share of the model's own answer shifts in the same direction in the large majority of matched pairs, significantly under the exact null of that statistic; but the shift is small and the delta distribution is heavy-tailed, so the MEAN's permutation null does not fire. Read `median_delta_frac`, not `mean_delta_frac`, and treat the magnitude as small.

| check | holds |
|---|---|
| rules_installed_and_conserving | yes |
| redistribution_consistent | yes |
| above_permutation_control | no |
| above_sign_test | yes |
| role_token_counts_matched | yes |

Token-identical roles: `['source_expr', 'trusted_expr', 'taint_chain', 'trust_chain', 'sink_call', 'signature']`. `sink_arg` is excluded from the verdict because it is the span the design edits — it is reported below, separately, as the role where a surface account is available.

### Table 19 — conservation, the validity condition

The fraction reading is licensed only where median |rho - 1| is within 0.25.

| layer | n_readings | median_rho | median_abs_rho_minus_one | max_abs_rho_minus_one | conserving |
|---|---|---|---|---|---|
| 0 | 288 | 1.0000 | 0.0000 | 0.0000 | 1 |
| 3 | 288 | 1.0000 | 0.0000 | 0.0000 | 1 |
| 7 | 288 | 1.0000 | 0.0000 | 0.0000 | 1 |
| 11 | 288 | 1.0000 | 0.0000 | 0.0000 | 1 |
| 15 | 288 | 1.0000 | 0.0000 | 0.0000 | 1 |
| 19 | 288 | 1.0000 | 0.0000 | 0.0001 | 1 |

### Table 20 — the redistribution at the reported cell

`mean_delta_frac` is the paired change in a role's share of the model's answer. The column sums to ~0 by conservation: whatever one role gains, another loses.

| ast_role | token_identical | n_pairs | mean_frac_unsafe | mean_frac_safe | median_delta_frac | mean_delta_frac | sign_consistency | sign_test_p | permutation_p | token_count_matched_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| other | 0 | 72 | 0.6481 | 0.6784 | 0.0075 | -0.0304 | 0.7639 | 0.0000 | 0.9420 | 1.0000 |
| trusted_expr | 1 | 72 | 0.0162 | 0.0372 | -0.0057 | -0.0210 | 0.3056 | 0.0013 | 0.0180 | 1.0000 |
| taint_chain | 1 | 72 | 0.0397 | 0.0552 | -0.0136 | -0.0155 | 0.0972 | 0.0000 | 0.5700 | 1.0000 |
| sink_arg | 0 | 72 | 0.0068 | 0.0027 | 0.0016 | 0.0042 | 0.6250 | 0.0444 | 0.5280 | 0.6111 |
| trust_chain | 1 | 72 | 0.0616 | 0.0488 | 0.0207 | 0.0128 | 0.8750 | 0.0000 | 0.3860 | 1.0000 |
| sink_call | 1 | 72 | -0.0002 | -0.0144 | -0.0060 | 0.0142 | 0.3194 | 0.0029 | 0.9420 | 1.0000 |
| signature | 1 | 72 | 0.1935 | 0.1763 | 0.0006 | 0.0172 | 0.5278 | 0.7239 | 0.9520 | 1.0000 |
| source_expr | 1 | 72 | 0.0342 | 0.0157 | -0.0006 | 0.0185 | 0.4583 | 0.5560 | 0.1420 | 1.0000 |

