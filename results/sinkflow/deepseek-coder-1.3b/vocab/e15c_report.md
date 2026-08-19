# E15-C — vocabulary-space contrast (deepseek-coder-1.3b)

**Verdict.** INVERTED — the security lexicon's contrast is strong and consistent but runs OPPOSITE to the hypothesis: unsafe programs score lower on the unsafe pole than their matched safe counterparts. Report the sign; do not report this as the model representing 'unsafe'.

Primary lens `rlens` (declared before any result was produced); reported at lens `rlens`, site `sink_arg`, layer 11, condition `clean_heldout`.

| check | holds |
|---|---|
| discovery_train_only_and_frozen | yes |
| held_out_replication | no |
| consistent_orientation | yes |
| above_permutation_control | yes |
| above_mismatched_pair_control | yes |
| stable_across_identifier_roles | yes |

This experiment is **observational**. A vocabulary direction that separates the two members is not evidence that the model uses it; E13's interchange is the causal instrument.

### Table 6 — training-discovered vocabulary-difference tokens

Ranked on CLEAN TRAINING pairs only and frozen before any held-out pair was scored. Positive = higher in the unsafe member.

| token | mean_delta_z | rank | sign_consistency | is_concept_unsafe | is_concept_safe | is_random_control |
|---|---|---|---|---|---|---|
|  ? | 0.960 | 0 | 0.964 | 0 | 0 | 0 |
| ?. | 0.887 | 1 | 0.946 | 0 | 0 | 0 |
| ?? | 0.858 | 2 | 0.929 | 0 | 0 | 0 |
| ?! | 0.826 | 3 | 0.964 | 0 | 0 | 0 |
| ected | 0.775 | 4 | 0.940 | 0 | 0 | 0 |
| ?-- | 0.748 | 5 | 0.810 | 0 | 0 | 0 |
| mor | 0.665 | 6 | 0.857 | 0 | 0 | 0 |
|  invested | 0.664 | 7 | 0.952 | 0 | 0 | 0 |
| bey | 0.663 | 8 | 0.911 | 0 | 0 | 0 |
| anas | 0.661 | 9 | 0.946 | 0 | 0 | 0 |
|  lighter | -0.589 | 186 | 0.077 | 0 | 0 | 0 |
|  associate | -0.603 | 187 | 0.071 | 0 | 0 | 0 |
|  involving | -0.607 | 188 | 0.196 | 0 | 0 | 0 |
|  Di | -0.612 | 189 | 0.089 | 0 | 0 | 0 |
| тва | -0.615 | 190 | 0.107 | 0 | 0 | 0 |
| 此时 | -0.636 | 191 | 0.006 | 0 | 0 | 0 |
| ponse | -0.640 | 192 | 0.083 | 0 | 0 | 0 |
| чина | -0.648 | 193 | 0.077 | 0 | 0 | 0 |
|  sensation | -0.654 | 194 | 0.089 | 0 | 0 | 0 |
| 联 | -0.661 | 195 | 0.089 | 0 | 0 | 0 |

### Table 7 — held-out semantic mass and sign consistency

`mean_delta_contrast_prob` is the paired change in (unsafe-token mass − safe-token mass); `..._z` is the scale-invariant companion, which is the one whose sign is exact under the J/R lenses.

| condition | condition_kind | mean_delta_contrast_prob | mean_delta_contrast_z | sign_consistency_z | sign_consistency_prob | permutation_effect_size | permutation_p | topk_enrichment_positive | topk_enrichment_random | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | -0.000 | -0.304 | 0.153 | 0.264 | -5.682 | 0.000 | 0.875 | 0.000 | 72 |
| normalize | baseline | -0.000 | -0.295 | 0.167 | 0.264 | -5.691 | 0.000 | 0.875 | 0.000 | 72 |
| rename_only | atomic | -0.000 | -0.053 | 0.514 | 0.486 | -1.018 | 0.288 | 0.625 | 0.000 | 72 |
| opaque_only | atomic | -0.000 | -0.279 | 0.222 | 0.264 | -5.170 | 0.000 | 0.875 | 0.000 | 72 |
| encode_only | atomic | -0.000 | -0.231 | 0.250 | 0.278 | -4.869 | 0.000 | 0.875 | 0.000 | 72 |
| flatten_only | atomic | -0.000 | -0.140 | 0.389 | 0.431 | -2.327 | 0.014 | 0.750 | 0.000 | 72 |
| rename_cumulative | cumulative | -0.000 | -0.063 | 0.444 | 0.444 | -1.231 | 0.210 | 0.625 | 0.000 | 72 |
| rename_opaque | cumulative | -0.000 | -0.111 | 0.486 | 0.375 | -2.496 | 0.014 | 0.500 | 0.000 | 72 |
| rename_opaque_encode | cumulative | -0.000 | 0.026 | 0.583 | 0.583 | 0.630 | 0.560 | 0.375 | 0.000 | 72 |
| rename_opaque_encode_flatten | cumulative | -0.000 | -0.194 | 0.347 | 0.403 | -2.927 | 0.000 | 0.625 | 0.000 | 72 |

### Table 8 — lens-method comparison by layer

| layer | relative_depth | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | topk_enrichment_positive |
|---|---|---|---|---|---|---|
| -1 |  | jlens | -0.000 | 0.111 | 0.986 | 0.000 |
| -1 |  | logit | 0.076 | 0.125 | 0.422 | 0.000 |
| -1 |  | rlens | 0.008 | 0.125 | 0.814 | 0.000 |
| 0 | 0.000 | jlens | -0.014 | 0.403 | 0.404 | 0.000 |
| 0 | 0.000 | logit | 0.063 | 0.514 | 0.482 | 0.125 |
| 0 | 0.000 | rlens | 0.005 | 0.542 | 0.940 | 0.000 |
| 3 | 0.130 | jlens | 0.038 | 0.556 | 0.298 | 0.000 |
| 3 | 0.130 | logit | 0.080 | 0.458 | 0.212 | 0.250 |
| 3 | 0.130 | rlens | 0.093 | 0.528 | 0.134 | 0.250 |
| 7 | 0.304 | jlens | -0.156 | 0.236 | 0.002 | 0.875 |
| 7 | 0.304 | logit | -0.387 | 0.181 | 0.000 | 0.875 |
| 7 | 0.304 | rlens | -0.058 | 0.361 | 0.208 | 0.625 |
| 11 | 0.478 | jlens | -0.268 | 0.236 | 0.000 | 0.875 |
| 11 | 0.478 | logit | -0.433 | 0.194 | 0.000 | 0.625 |
| 11 | 0.478 | rlens | -0.304 | 0.153 | 0.000 | 0.875 |
| 15 | 0.652 | jlens | -0.283 | 0.167 | 0.000 | 1.000 |
| 15 | 0.652 | logit | -0.348 | 0.153 | 0.000 | 0.500 |
| 15 | 0.652 | rlens | -0.264 | 0.139 | 0.000 | 1.000 |
| 19 | 0.826 | jlens | -0.295 | 0.208 | 0.000 | 0.875 |
| 19 | 0.826 | logit | -0.201 | 0.250 | 0.000 | 0.625 |
| 19 | 0.826 | rlens | -0.204 | 0.250 | 0.000 | 0.875 |
| 23 | 1.000 | jlens | -0.100 | 0.250 | 0.000 | 0.750 |
| 23 | 1.000 | logit | -0.100 | 0.250 | 0.000 | 0.750 |
| 23 | 1.000 | rlens | -0.100 | 0.250 | 0.000 | 0.750 |

Pairwise agreement of the three readouts' mean vocabulary-difference vectors:

| layer | lens_a | lens_b | cosine | spearman | n_tokens |
|---|---|---|---|---|---|
| 11 | jlens | logit | 0.895 | 0.871 | 196 |
| 11 | jlens | rlens | 0.960 | 0.954 | 196 |
| 11 | logit | rlens | 0.966 | 0.946 | 196 |

### Table 9 — semantic contrast across atomic and cumulative obfuscation

`cosine_to_clean` compares each condition's mean vocabulary-difference vector with the clean held-out one: accuracy asks whether a fitted direction still separates the classes, this asks whether the vocabulary-space difference still points the same way.

| condition | condition_kind | cosine_to_clean |
|---|---|---|
| clean_heldout | clean | 1.000 |
| normalize | baseline | 1.000 |
| rename_only | atomic | 0.964 |
| opaque_only | atomic | 0.995 |
| encode_only | atomic | 0.997 |
| flatten_only | atomic | 0.891 |
| rename_cumulative | cumulative | 0.969 |
| rename_opaque | cumulative | 0.969 |
| rename_opaque_encode | cumulative | 0.964 |
| rename_opaque_encode_flatten | cumulative | 0.855 |

### Table 11 — specificity: is the effect better than a random direction?

The permutation null asks whether the safe→unsafe *orientation* carries the effect. It does not ask whether **this** direction in the residual stream is special. `specificity` is the real arm's displacement from chance over the largest displacement any random or Gram-matched lens reaches in the same cell: **at or below 1.0, the result is not specific to the lens.**

| lens | layer | relative_depth | sign_consistency_z | permutation_p | displacement | control_displacement | specificity | beats_random_lens |
|---|---|---|---|---|---|---|---|---|
| jlens | -1 |  | 0.111 | 0.986 | 0.389 | 0.347 | 1.120 | True |
| jlens | 0 | 0.000 | 0.403 | 0.404 | 0.097 | 0.056 | 1.750 | True |
| jlens | 3 | 0.130 | 0.556 | 0.298 | 0.056 | 0.153 | 0.364 | False |
| jlens | 7 | 0.304 | 0.236 | 0.002 | 0.264 | 0.278 | 0.950 | False |
| jlens | 11 | 0.478 | 0.236 | 0.000 | 0.264 | 0.167 | 1.583 | True |
| jlens | 15 | 0.652 | 0.167 | 0.000 | 0.333 | 0.250 | 1.333 | True |
| jlens | 19 | 0.826 | 0.208 | 0.000 | 0.292 | 0.250 | 1.167 | True |
| jlens | 23 | 1.000 | 0.250 | 0.000 | 0.250 | 0.250 | 1.000 | False |
| logit | -1 |  | 0.125 | 0.422 | 0.375 | 0.347 | 1.080 | True |
| logit | 0 | 0.000 | 0.514 | 0.482 | 0.014 | 0.056 | 0.250 | False |
| logit | 3 | 0.130 | 0.458 | 0.212 | 0.042 | 0.153 | 0.273 | False |
| logit | 7 | 0.304 | 0.181 | 0.000 | 0.319 | 0.278 | 1.150 | True |
| logit | 11 | 0.478 | 0.194 | 0.000 | 0.306 | 0.167 | 1.833 | True |
| logit | 15 | 0.652 | 0.153 | 0.000 | 0.347 | 0.250 | 1.389 | True |
| logit | 19 | 0.826 | 0.250 | 0.000 | 0.250 | 0.250 | 1.000 | False |
| logit | 23 | 1.000 | 0.250 | 0.000 | 0.250 | 0.250 | 1.000 | False |
| rlens | -1 |  | 0.125 | 0.814 | 0.375 | 0.347 | 1.080 | True |
| rlens | 0 | 0.000 | 0.542 | 0.940 | 0.042 | 0.056 | 0.750 | False |
| rlens | 3 | 0.130 | 0.528 | 0.134 | 0.028 | 0.153 | 0.182 | False |
| rlens | 7 | 0.304 | 0.361 | 0.208 | 0.139 | 0.278 | 0.500 | False |
| rlens | 11 | 0.478 | 0.153 | 0.000 | 0.347 | 0.167 | 2.083 | True |
| rlens | 15 | 0.652 | 0.139 | 0.000 | 0.361 | 0.250 | 1.444 | True |
| rlens | 19 | 0.826 | 0.250 | 0.000 | 0.250 | 0.250 | 1.000 | False |
| rlens | 23 | 1.000 | 0.250 | 0.000 | 0.250 | 0.250 | 1.000 | False |

### Table 12 — is the contrast a distribution artifact?

`corr_contrast_entropy` and `corr_contrast_norm` correlate the paired contrast against the paired difference in the candidate distribution's entropy and score norm. A large |r| means the contrast tracks the *shape* of the distribution rather than its content, which would explain a consistent sign without any concept being involved.

| condition | mean_delta_contrast_z | corr_contrast_entropy | corr_contrast_norm | mean_delta_entropy |
|---|---|---|---|---|
| clean_heldout | -0.304 | -0.286 | -0.044 | 0.000 |
| normalize | -0.295 | -0.306 | -0.052 | 0.000 |
| rename_only | -0.053 | -0.010 | -0.342 | 0.001 |
| opaque_only | -0.279 | -0.267 | 0.098 | 0.000 |
| encode_only | -0.231 | -0.384 | -0.034 | 0.000 |
| flatten_only | -0.140 | -0.162 | 0.128 | 0.000 |
| rename_cumulative | -0.063 | -0.113 | -0.365 | 0.001 |
| rename_opaque | -0.111 | -0.117 | -0.038 | 0.001 |
| rename_opaque_encode | 0.026 | -0.256 | -0.101 | 0.001 |
| rename_opaque_encode_flatten | -0.194 | 0.243 | -0.322 | 0.000 |

### Table 10 — lens fidelity diagnostics (warnings, never blocking)

A weak row does not invalidate its layer. It is the reason the verdict separates *mechanically valid with weak fidelity* from *mechanically invalid*.

| lens | layer | is_control | next_token_top1 | next_token_mrr | final_layer_rank_agreement | relevance_conservation | weak_fidelity | warnings |
|---|---|---|---|---|---|---|---|---|
| gram_random | -1 | 1 |  |  | 0.047 |  | 1 | final-layer rank agreement 0.047 < 0.3 |
| gram_random | 0 | 1 |  |  | 0.181 |  | 1 | final-layer rank agreement 0.181 < 0.3 |
| gram_random | 3 | 1 |  |  | 0.095 |  | 1 | final-layer rank agreement 0.095 < 0.3 |
| gram_random | 7 | 1 |  |  | 0.044 |  | 1 | final-layer rank agreement 0.044 < 0.3 |
| gram_random | 11 | 1 |  |  | 0.036 |  | 1 | final-layer rank agreement 0.036 < 0.3 |
| gram_random | 15 | 1 |  |  | -0.020 |  | 1 | final-layer rank agreement -0.020 < 0.3 |
| gram_random | 19 | 1 |  |  | 0.098 |  | 1 | final-layer rank agreement 0.098 < 0.3 |
| gram_random | 23 | 1 |  |  | 0.092 |  | 1 | final-layer rank agreement 0.092 < 0.3 |
| jlens | -1 | 0 |  |  | 0.163 |  | 1 | final-layer rank agreement 0.163 < 0.3 |
| jlens | 0 | 0 |  |  | 0.314 |  | 0 |  |
| jlens | 3 | 0 |  |  | 0.281 |  | 1 | final-layer rank agreement 0.281 < 0.3 |
| jlens | 7 | 0 |  |  | 0.258 |  | 1 | final-layer rank agreement 0.258 < 0.3 |
| jlens | 11 | 0 |  |  | 0.358 |  | 0 |  |
| jlens | 15 | 0 |  |  | 0.352 |  | 0 |  |
| jlens | 19 | 0 |  |  | 0.534 |  | 0 |  |
| jlens | 23 | 0 |  |  | 1.000 |  | 0 |  |
| logit | -1 | 0 |  |  | 0.016 |  | 1 | final-layer rank agreement 0.016 < 0.3 |
| logit | 0 | 0 |  |  | 0.162 |  | 1 | final-layer rank agreement 0.162 < 0.3 |
| logit | 3 | 0 |  |  | 0.162 |  | 1 | final-layer rank agreement 0.162 < 0.3 |
| logit | 7 | 0 |  |  | 0.240 |  | 1 | final-layer rank agreement 0.240 < 0.3 |
| logit | 11 | 0 |  |  | 0.360 |  | 0 |  |
| logit | 15 | 0 |  |  | 0.494 |  | 0 |  |
| logit | 19 | 0 |  |  | 0.652 |  | 0 |  |
| logit | 23 | 0 |  |  | 1.000 |  | 0 |  |
| random | -1 | 1 |  |  | 0.017 |  | 1 | final-layer rank agreement 0.017 < 0.3 |
| random | 0 | 1 |  |  | -0.006 |  | 1 | final-layer rank agreement -0.006 < 0.3 |
| random | 3 | 1 |  |  | -0.006 |  | 1 | final-layer rank agreement -0.006 < 0.3 |
| random | 7 | 1 |  |  | 0.003 |  | 1 | final-layer rank agreement 0.003 < 0.3 |
| random | 11 | 1 |  |  | 0.006 |  | 1 | final-layer rank agreement 0.006 < 0.3 |
| random | 15 | 1 |  |  | 0.005 |  | 1 | final-layer rank agreement 0.005 < 0.3 |
| random | 19 | 1 |  |  | 0.008 |  | 1 | final-layer rank agreement 0.008 < 0.3 |
| random | 23 | 1 |  |  | 0.026 |  | 1 | final-layer rank agreement 0.026 < 0.3 |
| rlens | -1 | 0 |  |  | -0.148 | 1.001 | 1 | final-layer rank agreement -0.148 < 0.3 |
| rlens | 0 | 0 |  |  | -0.067 | 1.001 | 1 | final-layer rank agreement -0.067 < 0.3 |
| rlens | 3 | 0 |  |  | 0.104 | 1.001 | 1 | final-layer rank agreement 0.104 < 0.3 |
| rlens | 7 | 0 |  |  | 0.309 | 1.000 | 0 |  |
| rlens | 11 | 0 |  |  | 0.472 | 1.000 | 0 |  |
| rlens | 15 | 0 |  |  | 0.623 | 1.000 | 0 |  |
| rlens | 19 | 0 |  |  | 0.780 | 1.000 | 0 |  |
| rlens | 23 | 0 |  |  | 1.000 | 1.000 | 0 |  |

### Controls at the reported cell

| arm | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | n_pairs |
|---|---|---|---|---|---|
| mismatched_pairs | jlens | -0.256 | 0.250 | 0.000 | 72 |
| mismatched_pairs | logit | -0.402 | 0.153 | 0.000 | 72 |
| mismatched_pairs | rlens | -0.289 | 0.167 | 0.000 | 72 |
| random_lens | random | -0.278 | 0.347 | 0.000 | 72 |
| gram_random_lens | gram_random | 0.327 | 0.667 | 0.000 | 72 |
| role_swap_0 | jlens | -0.215 | 0.351 | 0.002 | 37 |
| role_swap_0 | logit | -0.390 | 0.189 | 0.000 | 37 |
| role_swap_0 | rlens | -0.273 | 0.216 | 0.000 | 37 |
| role_swap_1 | jlens | -0.324 | 0.114 | 0.000 | 35 |
| role_swap_1 | logit | -0.478 | 0.200 | 0.000 | 35 |
| role_swap_1 | rlens | -0.337 | 0.086 | 0.000 | 35 |

### Concept tokens at the reported cell

| token | mean_delta_z | rank | sign_consistency | mean_prob_unsafe | mean_prob_safe |
|---|---|---|---|---|---|
|  vulnerable | -0.105 | 104 | 0.361 | 0.005 | 0.005 |
|  safe | 0.009 | 93 | 0.472 | 0.005 | 0.005 |
|  trusted | 0.348 | 58 | 0.833 | 0.005 | 0.005 |
|  clean | 0.242 | 74 | 0.708 | 0.006 | 0.006 |

