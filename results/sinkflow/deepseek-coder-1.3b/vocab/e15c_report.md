# E15-C — vocabulary-space contrast (deepseek-coder-1.3b)

**Verdict.** MECHANICALLY VALID, WEAK LENS FIDELITY — the numbers stand as measurements, the instrument is the caveat at this layer.

Primary lens `rlens` (declared before any result was produced); reported at lens `rlens`, site `sink_arg`, layer 0, condition `clean_heldout`.

| check | holds |
|---|---|
| discovery_train_only_and_frozen | yes |
| held_out_replication | no |
| consistent_orientation | yes |
| above_permutation_control | no |
| above_mismatched_pair_control | yes |
| stable_across_identifier_roles | yes |

This experiment is **observational**. A vocabulary direction that separates the two members is not evidence that the model uses it; E13's interchange is the causal instrument.

### Table 6 — training-discovered vocabulary-difference tokens

Ranked on CLEAN TRAINING pairs only and frozen before any held-out pair was scored. Positive = higher in the unsafe member.

| token | mean_delta_z | rank | sign_consistency | is_concept_unsafe | is_concept_safe | is_random_control |
|---|---|---|---|---|---|---|
| suff | 0.116 | 0 | 0.554 | 0 | 0 | 0 |
| sel | 0.112 | 1 | 0.536 | 0 | 0 | 0 |
| ux | 0.093 | 2 | 0.500 | 0 | 0 | 0 |
| nd | 0.085 | 3 | 0.536 | 0 | 0 | 0 |
| pled | 0.078 | 4 | 0.476 | 0 | 0 | 0 |
| anas | 0.070 | 5 | 0.530 | 0 | 0 | 0 |
| ser | 0.069 | 6 | 0.530 | 0 | 0 | 0 |
| emor | 0.065 | 7 | 0.565 | 0 | 0 | 0 |
| нд | 0.062 | 8 | 0.542 | 0 | 0 | 0 |
| ra | 0.059 | 9 | 0.494 | 0 | 0 | 0 |
| ponse | -0.055 | 186 | 0.464 | 0 | 0 | 0 |
| 或是 | -0.055 | 187 | 0.464 | 0 | 0 | 0 |
|  Poly | -0.055 | 188 | 0.500 | 0 | 0 | 0 |
| ря | -0.062 | 189 | 0.470 | 0 | 0 | 0 |
| mb | -0.065 | 190 | 0.506 | 0 | 0 | 0 |
| DOCTYPE | -0.066 | 191 | 0.518 | 0 | 0 | 0 |
| чина | -0.069 | 192 | 0.518 | 0 | 0 | 0 |
|  MB | -0.074 | 193 | 0.488 | 0 | 0 | 0 |
| ientes | -0.078 | 194 | 0.446 | 0 | 0 | 0 |
| цо | -0.081 | 195 | 0.470 | 0 | 0 | 0 |

### Table 7 — held-out semantic mass and sign consistency

`mean_delta_contrast_prob` is the paired change in (unsafe-token mass − safe-token mass); `..._z` is the scale-invariant companion, which is the one whose sign is exact under the J/R lenses.

| condition | condition_kind | mean_delta_contrast_prob | mean_delta_contrast_z | sign_consistency_z | sign_consistency_prob | permutation_effect_size | permutation_p | topk_enrichment_positive | topk_enrichment_random | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 0.000 | 0.005 | 0.542 | 0.597 | 0.078 | 0.940 | 0.000 | 0.031 | 72 |
| normalize | baseline | 0.000 | 0.005 | 0.569 | 0.611 | 0.079 | 0.936 | 0.000 | 0.031 | 72 |
| rename_only | atomic | -0.000 | -0.049 | 0.417 | 0.444 | -0.858 | 0.356 | 0.250 | 0.031 | 72 |
| opaque_only | atomic | 0.000 | 0.003 | 0.514 | 0.611 | 0.031 | 0.964 | 0.000 | 0.031 | 72 |
| encode_only | atomic | 0.000 | 0.009 | 0.514 | 0.611 | 0.156 | 0.884 | 0.000 | 0.031 | 72 |
| flatten_only | atomic | 0.000 | 0.004 | 0.528 | 0.583 | 0.048 | 0.954 | 0.000 | 0.031 | 72 |
| rename_cumulative | cumulative | 0.000 | 0.031 | 0.528 | 0.542 | 0.704 | 0.468 | 0.375 | 0.000 | 72 |
| rename_opaque | cumulative | -0.000 | -0.093 | 0.417 | 0.417 | -1.673 | 0.094 | 0.250 | 0.031 | 72 |
| rename_opaque_encode | cumulative | -0.000 | -0.051 | 0.486 | 0.500 | -0.859 | 0.396 | 0.125 | 0.000 | 72 |
| rename_opaque_encode_flatten | cumulative | -0.000 | 0.002 | 0.458 | 0.472 | 0.051 | 0.968 | 0.000 | 0.062 | 72 |

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
| 0 | jlens | logit | 0.252 | 0.182 | 196 |
| 0 | jlens | rlens | 0.482 | 0.442 | 196 |
| 0 | logit | rlens | 0.427 | 0.398 | 196 |

### Table 9 — semantic contrast across atomic and cumulative obfuscation

`cosine_to_clean` compares each condition's mean vocabulary-difference vector with the clean held-out one: accuracy asks whether a fitted direction still separates the classes, this asks whether the vocabulary-space difference still points the same way.

| condition | condition_kind | cosine_to_clean |
|---|---|---|
| clean_heldout | clean | 1.000 |
| normalize | baseline | 1.000 |
| rename_only | atomic | 0.307 |
| opaque_only | atomic | 0.995 |
| encode_only | atomic | 0.995 |
| flatten_only | atomic | 0.990 |
| rename_cumulative | cumulative | -0.023 |
| rename_opaque | cumulative | 0.139 |
| rename_opaque_encode | cumulative | 0.074 |
| rename_opaque_encode_flatten | cumulative | -0.136 |

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
| mismatched_pairs | jlens | -0.020 | 0.389 | 0.300 | 72 |
| mismatched_pairs | logit | 0.133 | 0.542 | 0.170 | 72 |
| mismatched_pairs | rlens | 0.035 | 0.514 | 0.588 | 72 |
| random_lens | random | 0.019 | 0.472 | 0.800 | 72 |
| gram_random_lens | gram_random | 0.004 | 0.556 | 0.972 | 72 |
| role_swap_0 | jlens | 0.006 | 0.378 | 0.808 | 37 |
| role_swap_0 | logit | -0.076 | 0.541 | 0.380 | 37 |
| role_swap_0 | rlens | 0.049 | 0.622 | 0.566 | 37 |
| role_swap_1 | jlens | -0.035 | 0.429 | 0.244 | 35 |
| role_swap_1 | logit | 0.211 | 0.486 | 0.182 | 35 |
| role_swap_1 | rlens | -0.041 | 0.457 | 0.656 | 35 |

### Concept tokens at the reported cell

| token | mean_delta_z | rank | sign_consistency | mean_prob_unsafe | mean_prob_safe |
|---|---|---|---|---|---|
|  vulnerable | -0.046 | 174 | 0.375 | 0.005 | 0.005 |
|  safe | -0.057 | 186 | 0.375 | 0.005 | 0.005 |
|  trusted | -0.043 | 171 | 0.500 | 0.005 | 0.005 |
|  clean | -0.055 | 184 | 0.389 | 0.005 | 0.005 |

