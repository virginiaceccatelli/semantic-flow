# E15-D — three follow-ups to the E15-C null (starcoder2-3b)

Each section states a verdict decided by a checklist declared in code before the run. All three stages are observational: none of them establishes that the model *uses* what is measured.

| stage | gate | verdict |
|---|---|---|
| V1 full-vocabulary alignment | J2 PASS | `no_shared_direction` |
| positive control | J3 FAIL | `mechanically_invalid` |
| V3 relevance redistribution | J4 FAIL | `mechanically_invalid` |

**What this means for E15-C.** J3 did not pass; nothing may be read.

---

## V1 — is there a shared full-vocabulary direction?

**Verdict.** NO SHARED DIRECTION — the per-pair differences do not agree, over the WHOLE vocabulary. This is strictly stronger than E15-C's null: no candidate pool was chosen, so no pool can be blamed.

Read at site `last_token` (declared before any result: it is the only site where both members carry the same token id), layer 15, condition `clean_heldout`.

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
| 0 | 72 | 0.1641 | 0.0139 | 0.4516 | 0.3633 | 0.0010 | -0.0250 | 0.4583 | -0.0745 | 0.0217 | 0.4931 |
| 3 | 72 | 0.1437 | 0.0139 | 0.3631 | 0.3957 | 0.0018 | -0.0075 | 0.5000 | -0.0491 | 0.0338 | 0.4861 |
| 7 | 72 | 0.1112 | 0.0139 | 0.2946 | 0.3773 | 0.0198 | 0.1198 | 0.7083 | 0.0769 | 0.1623 | 0.5208 |
| 11 | 72 | 0.2045 | 0.0139 | 0.3059 | 0.6684 | 0.1533 | 0.3878 | 0.9861 | 0.3459 | 0.4312 | 0.5278 |
| 15 | 72 | 0.2200 | 0.0139 | 0.2912 | 0.7554 | 0.1597 | 0.3895 | 1.0000 | 0.3514 | 0.4292 | 0.5486 |
| 19 | 72 | 0.2207 | 0.0139 | 0.2755 | 0.8010 | 0.1345 | 0.3448 | 1.0000 | 0.3137 | 0.3772 | 0.5556 |
| 23 | 72 | 0.2056 | 0.0139 | 0.2618 | 0.7852 | 0.1263 | 0.3378 | 0.9583 | 0.3014 | 0.3745 | 0.5347 |
| 27 | 72 | 0.1964 | 0.0139 | 0.2360 | 0.8321 | 0.1125 | 0.3193 | 0.9583 | 0.2832 | 0.3562 | 0.5208 |
| 29 | 72 | 0.1992 | 0.0139 | 0.2278 | 0.8747 | 0.1155 | 0.3244 | 0.9583 | 0.2837 | 0.3673 | 0.4931 |

### Table 14 — the same measurement across obfuscation

| condition | condition_kind | sv1_share | sv1_ratio | proj_mean | proj_sign_consistency | n_pairs |
|---|---|---|---|---|---|---|
| clean_heldout | clean | 0.2200 | 0.7554 | 0.3895 | 1.0000 | 72 |
| normalize | baseline | 0.1927 | 0.6051 | 0.1127 | 0.9444 | 72 |
| rename_only | atomic | 0.1283 | 0.4118 | 0.0845 | 0.8889 | 72 |
| opaque_only | atomic | 0.1563 | 0.6613 | 0.0908 | 0.9028 | 72 |
| encode_only | atomic | 0.1901 | 0.6236 | 0.1058 | 0.9722 | 72 |
| flatten_only | atomic | 0.0700 | 0.3708 | 0.0281 | 0.6806 | 72 |
| rename_cumulative | cumulative | 0.1390 | 0.4497 | 0.0872 | 0.8611 | 72 |
| rename_opaque | cumulative | 0.0757 | 0.2546 | 0.0587 | 0.8333 | 72 |
| rename_opaque_encode | cumulative | 0.1049 | 0.4395 | 0.0669 | 0.8750 | 72 |
| rename_opaque_encode_flatten | cumulative | 0.0698 | 0.4289 | 0.0265 | 0.6806 | 72 |

### Table 15 — what the direction says, as tokens

Discovered from the differences, not proposed in advance. `overlap_with_same_label_direction` is the Jaccard overlap of the top-100 loadings with the direction the SAME-LABEL differences find: high overlap means the direction is whatever distinguishes any two of these programs.

| pole | rank_within_pole | token | loading | overlap_with_same_label_direction |
|---|---|---|---|---|
| unsafe_higher | 0 | Iam | 0.0236 | 0.0000 |
| unsafe_higher | 1 |  closing | 0.0188 | 0.0000 |
| unsafe_higher | 2 | gn | 0.0182 | 0.0000 |
| unsafe_higher | 3 | ry | 0.0180 | 0.0000 |
| unsafe_higher | 4 | iam | 0.0177 | 0.0000 |
| unsafe_higher | 5 | brtc | 0.0176 | 0.0000 |
| unsafe_higher | 6 | urity | 0.0176 | 0.0000 |
| unsafe_higher | 7 | ús | 0.0175 | 0.0000 |
| unsafe_higher | 8 | 有的 | 0.0174 | 0.0000 |
| unsafe_higher | 9 | Kinds | 0.0170 | 0.0000 |
| unsafe_higher | 10 | uppet | 0.0169 | 0.0000 |
| unsafe_higher | 11 | Preserve | 0.0168 | 0.0000 |
| unsafe_higher | 12 |  MARK | 0.0168 | 0.0000 |
| unsafe_higher | 13 | Lite | 0.0167 | 0.0000 |
| unsafe_higher | 14 | ema | 0.0165 | 0.0000 |
| unsafe_higher | 15 | orse | 0.0165 | 0.0000 |
| unsafe_higher | 16 | rias | 0.0164 | 0.0000 |
| unsafe_higher | 17 |  iy | 0.0163 | 0.0000 |
| unsafe_higher | 18 |  CORS | 0.0160 | 0.0000 |
| unsafe_higher | 19 | bill | 0.0160 | 0.0000 |
| unsafe_higher | 20 | Occurrence | 0.0160 | 0.0000 |
| unsafe_higher | 21 | geo | 0.0159 | 0.0000 |
| unsafe_higher | 22 |  informace | 0.0158 | 0.0000 |
| unsafe_higher | 23 |  Occ | 0.0157 | 0.0000 |
| unsafe_higher | 24 | Panels | 0.0157 | 0.0000 |
| safe_higher | 0 | Wn | -0.0250 | 0.0000 |
| safe_higher | 1 | ceeding | -0.0213 | 0.0000 |
| safe_higher | 2 | <pr_in_reply_to_review_id> | -0.0204 | 0.0000 |
| safe_higher | 3 | PLI | -0.0200 | 0.0000 |
| safe_higher | 4 | bootstrapcdn | -0.0200 | 0.0000 |

### Table 16 — full vocabulary versus E15-C's frozen 196-token pool

If concentration is high over the full vocabulary and low inside the pool, the pool missed the direction. If it is low in both, no pool would have helped.

| lens | arm | layer | site | n_candidates | sv1_share | sv1_floor | mean_pairwise_cosine |
|---|---|---|---|---|---|---|---|
| jlens | main | -1 | sink_arg | 196 | 0.3385 | 0.0556 | -0.0235 |
| jlens | same_label_unsafe | -1 | sink_arg | 196 | 0.5672 | 0.0233 | -0.0226 |
| jlens | same_label_safe | -1 | sink_arg | 196 | 0.5691 | 0.0233 | -0.0234 |
| jlens | main | -1 | last_token | 196 |  |  |  |
| jlens | same_label_unsafe | -1 | last_token | 196 |  |  |  |
| jlens | same_label_safe | -1 | last_token | 196 |  |  |  |
| jlens | main | 0 | sink_arg | 196 | 0.1833 | 0.0139 | -0.0001 |
| jlens | same_label_unsafe | 0 | sink_arg | 196 | 0.3451 | 0.0139 | -0.0111 |
| jlens | same_label_safe | 0 | sink_arg | 196 | 0.3365 | 0.0139 | -0.0118 |
| jlens | main | 0 | last_token | 196 | 0.2296 | 0.0139 | 0.0064 |
| jlens | same_label_unsafe | 0 | last_token | 196 | 0.4023 | 0.0139 | -0.0133 |
| jlens | same_label_safe | 0 | last_token | 196 | 0.3828 | 0.0139 | -0.0134 |
| jlens | main | 3 | sink_arg | 196 | 0.1809 | 0.0139 | 0.0071 |
| jlens | same_label_unsafe | 3 | sink_arg | 196 | 0.2563 | 0.0139 | -0.0123 |
| jlens | same_label_safe | 3 | sink_arg | 196 | 0.2415 | 0.0139 | -0.0130 |
| jlens | main | 3 | last_token | 196 | 0.1594 | 0.0139 | -0.0022 |
| jlens | same_label_unsafe | 3 | last_token | 196 | 0.3483 | 0.0139 | -0.0132 |
| jlens | same_label_safe | 3 | last_token | 196 | 0.3430 | 0.0139 | -0.0134 |
| jlens | main | 7 | sink_arg | 196 | 0.1742 | 0.0139 | 0.0404 |
| jlens | same_label_unsafe | 7 | sink_arg | 196 | 0.2427 | 0.0139 | -0.0133 |
| jlens | same_label_safe | 7 | sink_arg | 196 | 0.2364 | 0.0139 | -0.0130 |
| jlens | main | 7 | last_token | 196 | 0.2509 | 0.0139 | 0.0272 |
| jlens | same_label_unsafe | 7 | last_token | 196 | 0.2946 | 0.0139 | -0.0136 |
| jlens | same_label_safe | 7 | last_token | 196 | 0.2929 | 0.0139 | -0.0136 |
| jlens | main | 11 | sink_arg | 196 | 0.1584 | 0.0139 | 0.1009 |
| jlens | same_label_unsafe | 11 | sink_arg | 196 | 0.2281 | 0.0139 | -0.0130 |
| jlens | same_label_safe | 11 | sink_arg | 196 | 0.2175 | 0.0139 | -0.0133 |
| jlens | main | 11 | last_token | 196 | 0.1823 | 0.0139 | 0.1242 |
| jlens | same_label_unsafe | 11 | last_token | 196 | 0.3065 | 0.0139 | -0.0135 |
| jlens | same_label_safe | 11 | last_token | 196 | 0.2997 | 0.0139 | -0.0132 |
| jlens | main | 15 | sink_arg | 196 | 0.2544 | 0.0139 | 0.1754 |
| jlens | same_label_unsafe | 15 | sink_arg | 196 | 0.2538 | 0.0139 | -0.0130 |
| jlens | same_label_safe | 15 | sink_arg | 196 | 0.2586 | 0.0139 | -0.0133 |
| jlens | main | 15 | last_token | 196 | 0.2636 | 0.0139 | 0.1700 |
| jlens | same_label_unsafe | 15 | last_token | 196 | 0.3126 | 0.0139 | -0.0133 |
| jlens | same_label_safe | 15 | last_token | 196 | 0.3039 | 0.0139 | -0.0130 |
| jlens | main | 19 | sink_arg | 196 | 0.3110 | 0.0139 | 0.2635 |
| jlens | same_label_unsafe | 19 | sink_arg | 196 | 0.2042 | 0.0139 | -0.0132 |
| jlens | same_label_safe | 19 | sink_arg | 196 | 0.2127 | 0.0139 | -0.0132 |
| jlens | main | 19 | last_token | 196 | 0.2265 | 0.0139 | 0.1483 |

---

## Positive control — can this machinery detect verbalisation at all?

**Verdict.** MECHANICALLY INVALID — J3 did not pass.

Prompt style `sink`, lens `rlens`, condition `clean_heldout`, layer None — chosen as the layer that best detects the TAINT property, with the security contrast then read at that same cell.

| check | holds |
|---|---|
| behaviour_above_chance | no |
| lens_detects_the_property | no |
| lens_tracks_the_model | no |
| security_contrast_at_same_cell | no |

### Table 17 — behaviour: can the model answer at all?

`pair_separation` is the fraction of bases where the unsafe member draws a higher yes-margin than its matched safe counterpart. Its chance level is 0.5 and no answer bias can move it, which is why it is the statistic the verdict uses rather than raw accuracy.

_not run_

### Table 18 — the two properties, one basis, one lens, by layer

`taint_*` and `security_*` differ only in which token positions are named as the poles. `taint_lens_tracks_model` is the fraction of pairs where the lens's paired margin has the same sign as the model's own.

_not run_

---

## V3 — where does relevance move?

**Verdict.** MECHANICALLY INVALID — J4 did not pass.

| check | holds |
|---|---|
| rules_installed_and_conserving | no |
| redistribution_consistent | no |
| above_permutation_control | no |
| role_token_counts_matched | no |

Token-identical roles: `['source_expr', 'trusted_expr', 'taint_chain', 'trust_chain', 'sink_call', 'signature']`. `sink_arg` is excluded from the verdict because it is the span the design edits — it is reported below, separately, as the role where a surface account is available.

### Table 19 — conservation, the validity condition

The fraction reading is licensed only where median |rho - 1| is within 0.25.

_not run_

### Table 20 — the redistribution at the reported cell

`mean_delta_frac` is the paired change in a role's share of the model's answer. The column sums to ~0 by conservation: whatever one role gains, another loses.

_not run_

