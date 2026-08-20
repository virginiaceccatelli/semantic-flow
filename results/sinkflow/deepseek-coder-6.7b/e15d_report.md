# E15-D — three follow-ups to the E15-C null (deepseek-coder-6.7b)

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
| 0 | 72 | 0.2505 | 0.0139 | 0.3231 | 0.7753 | -0.0051 | -0.0382 | 0.4583 | -0.1252 | 0.0453 | 0.5139 |
| 3 | 72 | 0.1989 | 0.0139 | 0.2684 | 0.7409 | -0.0026 | -0.0306 | 0.4861 | -0.1029 | 0.0392 | 0.4861 |
| 7 | 72 | 0.2245 | 0.0139 | 0.2806 | 0.8001 | 0.0534 | 0.2155 | 0.7917 | 0.1660 | 0.2652 | 0.5069 |
| 11 | 72 | 0.2145 | 0.0139 | 0.2327 | 0.9215 | 0.1333 | 0.3582 | 1.0000 | 0.3323 | 0.3835 | 0.4931 |
| 15 | 72 | 0.1967 | 0.0139 | 0.2035 | 0.9666 | 0.1475 | 0.3796 | 1.0000 | 0.3577 | 0.4007 | 0.4653 |
| 19 | 72 | 0.2089 | 0.0139 | 0.2003 | 1.0428 | 0.0956 | 0.2998 | 1.0000 | 0.2752 | 0.3242 | 0.5139 |
| 23 | 72 | 0.1946 | 0.0139 | 0.2063 | 0.9433 | 0.0802 | 0.2738 | 0.9861 | 0.2493 | 0.2970 | 0.5208 |
| 27 | 72 | 0.1869 | 0.0139 | 0.2114 | 0.8838 | 0.0734 | 0.2628 | 1.0000 | 0.2400 | 0.2853 | 0.5208 |
| 31 | 72 | 0.1497 | 0.0139 | 0.2201 | 0.6800 | 0.0627 | 0.2464 | 0.9861 | 0.2216 | 0.2707 | 0.5069 |

### Table 14 — the same measurement across obfuscation

| condition | condition_kind | sv1_share | sv1_ratio | proj_mean | proj_sign_consistency | n_pairs |
|---|---|---|---|---|---|---|
| clean_heldout | clean | 0.1967 | 0.9666 | 0.3796 | 1.0000 | 72 |
| normalize | baseline | 0.2280 | 0.7663 | 0.2484 | 1.0000 | 72 |
| rename_only | atomic | 0.1541 | 0.5342 | 0.1852 | 0.9861 | 72 |
| opaque_only | atomic | 0.1861 | 0.8260 | 0.1765 | 0.9167 | 72 |
| encode_only | atomic | 0.2302 | 0.8154 | 0.2297 | 1.0000 | 72 |
| flatten_only | atomic | 0.0688 | 0.4805 | 0.0266 | 0.7083 | 72 |
| rename_cumulative | cumulative | 0.1422 | 0.5017 | 0.1681 | 0.9583 | 72 |
| rename_opaque | cumulative | 0.1028 | 0.3314 | 0.1025 | 0.8333 | 72 |
| rename_opaque_encode | cumulative | 0.1287 | 0.4946 | 0.1154 | 0.9028 | 72 |
| rename_opaque_encode_flatten | cumulative | 0.0742 | 0.5288 | 0.0254 | 0.6250 | 72 |

### Table 15 — what the direction says, as tokens

Discovered from the differences, not proposed in advance. `overlap_with_same_label_direction` is the Jaccard overlap of the top-100 loadings with the direction the SAME-LABEL differences find: high overlap means the direction is whatever distinguishes any two of these programs.

| pole | rank_within_pole | token | loading | overlap_with_same_label_direction |
|---|---|---|---|---|
| unsafe_higher | 0 |  mel | 0.0248 | 0.0050 |
| unsafe_higher | 1 | mel | 0.0218 | 0.0050 |
| unsafe_higher | 2 | eto | 0.0215 | 0.0050 |
| unsafe_higher | 3 | hat | 0.0211 | 0.0050 |
| unsafe_higher | 4 | HO | 0.0208 | 0.0050 |
| unsafe_higher | 5 |  sug | 0.0206 | 0.0050 |
| unsafe_higher | 6 | 脸 | 0.0203 | 0.0050 |
| unsafe_higher | 7 |  breakdown | 0.0199 | 0.0050 |
| unsafe_higher | 8 | Package | 0.0198 | 0.0050 |
| unsafe_higher | 9 | Route | 0.0197 | 0.0050 |
| unsafe_higher | 10 |  break | 0.0195 | 0.0050 |
| unsafe_higher | 11 | clean | 0.0194 | 0.0050 |
| unsafe_higher | 12 | agg | 0.0193 | 0.0050 |
| unsafe_higher | 13 |  Stage | 0.0193 | 0.0050 |
| unsafe_higher | 14 |  Ho | 0.0192 | 0.0050 |
| unsafe_higher | 15 |  pending | 0.0192 | 0.0050 |
| unsafe_higher | 16 | sb | 0.0192 | 0.0050 |
| unsafe_higher | 17 | break | 0.0191 | 0.0050 |
| unsafe_higher | 18 |  bgcolor | 0.0191 | 0.0050 |
| unsafe_higher | 19 | 场 | 0.0190 | 0.0050 |
| unsafe_higher | 20 | nea | 0.0189 | 0.0050 |
| unsafe_higher | 21 | sqrt | 0.0188 | 0.0050 |
| unsafe_higher | 22 |  stage | 0.0188 | 0.0050 |
| unsafe_higher | 23 | Face | 0.0187 | 0.0050 |
| unsafe_higher | 24 |  Suite | 0.0186 | 0.0050 |
| safe_higher | 0 | arguments | -0.0235 | 0.0050 |
| safe_higher | 1 |  Ivan | -0.0233 | 0.0050 |
| safe_higher | 2 | 椒 | -0.0230 | 0.0050 |
| safe_higher | 3 | utch | -0.0227 | 0.0050 |
| safe_higher | 4 |  Lore | -0.0221 | 0.0050 |

### Table 16 — full vocabulary versus E15-C's frozen 196-token pool

If concentration is high over the full vocabulary and low inside the pool, the pool missed the direction. If it is low in both, no pool would have helped.

| lens | arm | layer | site | n_candidates | sv1_share | sv1_floor | mean_pairwise_cosine |
|---|---|---|---|---|---|---|---|
| jlens | main | -1 | sink_arg | 196 | 0.3703 | 0.0556 | -0.0407 |
| jlens | same_label_unsafe | -1 | sink_arg | 196 | 0.4992 | 0.0233 | -0.0230 |
| jlens | same_label_safe | -1 | sink_arg | 196 | 0.4869 | 0.0233 | -0.0232 |
| jlens | main | -1 | last_token | 196 |  |  |  |
| jlens | same_label_unsafe | -1 | last_token | 196 |  |  |  |
| jlens | same_label_safe | -1 | last_token | 196 |  |  |  |
| jlens | main | 0 | sink_arg | 196 | 0.2779 | 0.0139 | -0.0029 |
| jlens | same_label_unsafe | 0 | sink_arg | 196 | 0.2948 | 0.0139 | -0.0093 |
| jlens | same_label_safe | 0 | sink_arg | 196 | 0.2969 | 0.0139 | -0.0106 |
| jlens | main | 0 | last_token | 196 | 0.4006 | 0.0139 | -0.0067 |
| jlens | same_label_unsafe | 0 | last_token | 196 | 0.4181 | 0.0139 | -0.0116 |
| jlens | same_label_safe | 0 | last_token | 196 | 0.4226 | 0.0139 | -0.0126 |
| jlens | main | 3 | sink_arg | 196 | 0.1819 | 0.0139 | 0.0218 |
| jlens | same_label_unsafe | 3 | sink_arg | 196 | 0.3308 | 0.0139 | -0.0117 |
| jlens | same_label_safe | 3 | sink_arg | 196 | 0.3108 | 0.0139 | -0.0120 |
| jlens | main | 3 | last_token | 196 | 0.2656 | 0.0139 | -0.0047 |
| jlens | same_label_unsafe | 3 | last_token | 196 | 0.4296 | 0.0139 | -0.0136 |
| jlens | same_label_safe | 3 | last_token | 196 | 0.4502 | 0.0139 | -0.0134 |
| jlens | main | 7 | sink_arg | 196 | 0.2588 | 0.0139 | 0.0378 |
| jlens | same_label_unsafe | 7 | sink_arg | 196 | 0.2576 | 0.0139 | -0.0124 |
| jlens | same_label_safe | 7 | sink_arg | 196 | 0.2392 | 0.0139 | -0.0129 |
| jlens | main | 7 | last_token | 196 | 0.2298 | 0.0139 | 0.0665 |
| jlens | same_label_unsafe | 7 | last_token | 196 | 0.2608 | 0.0139 | -0.0137 |
| jlens | same_label_safe | 7 | last_token | 196 | 0.2545 | 0.0139 | -0.0133 |
| jlens | main | 11 | sink_arg | 196 | 0.2392 | 0.0139 | 0.1415 |
| jlens | same_label_unsafe | 11 | sink_arg | 196 | 0.2571 | 0.0139 | -0.0131 |
| jlens | same_label_safe | 11 | sink_arg | 196 | 0.2711 | 0.0139 | -0.0132 |
| jlens | main | 11 | last_token | 196 | 0.2550 | 0.0139 | 0.1096 |
| jlens | same_label_unsafe | 11 | last_token | 196 | 0.2606 | 0.0139 | -0.0138 |
| jlens | same_label_safe | 11 | last_token | 196 | 0.2538 | 0.0139 | -0.0135 |
| jlens | main | 15 | sink_arg | 196 | 0.4484 | 0.0139 | 0.4256 |
| jlens | same_label_unsafe | 15 | sink_arg | 196 | 0.2143 | 0.0139 | -0.0135 |
| jlens | same_label_safe | 15 | sink_arg | 196 | 0.2142 | 0.0139 | -0.0136 |
| jlens | main | 15 | last_token | 196 | 0.2109 | 0.0139 | 0.1774 |
| jlens | same_label_unsafe | 15 | last_token | 196 | 0.2097 | 0.0139 | -0.0138 |
| jlens | same_label_safe | 15 | last_token | 196 | 0.1936 | 0.0139 | -0.0137 |
| jlens | main | 19 | sink_arg | 196 | 0.4780 | 0.0139 | 0.4540 |
| jlens | same_label_unsafe | 19 | sink_arg | 196 | 0.1878 | 0.0139 | -0.0138 |
| jlens | same_label_safe | 19 | sink_arg | 196 | 0.2023 | 0.0139 | -0.0138 |
| jlens | main | 19 | last_token | 196 | 0.1930 | 0.0139 | 0.1275 |

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

