# E15-C — vocabulary-space contrast (deepseek-coder-6.7b)

**Verdict.** MECHANICALLY VALID, WEAK LENS FIDELITY — the numbers stand as measurements, the instrument is the caveat at this layer.

Primary lens `rlens` (declared before any result was produced); reported at lens `rlens`, site `sink_arg`, layer 3, condition `clean_heldout`.

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
| tbl | 0.098 | 0 | 0.488 | 0 | 0 | 0 |
| ses | 0.095 | 1 | 0.536 | 0 | 0 | 0 |
| pty | 0.095 | 2 | 0.500 | 0 | 0 | 0 |
| Clean | 0.082 | 3 | 0.577 | 0 | 0 | 0 |
| enders | 0.076 | 4 | 0.560 | 0 | 0 | 0 |
| clean | 0.075 | 5 | 0.560 | 0 | 0 | 0 |
| hens | 0.069 | 6 | 0.565 | 0 | 0 | 0 |
| 隔 | 0.061 | 7 | 0.571 | 0 | 0 | 0 |
| Pop | 0.059 | 8 | 0.512 | 0 | 0 | 0 |
| tain | 0.057 | 9 | 0.548 | 0 | 0 | 0 |
| inu | -0.054 | 186 | 0.470 | 0 | 0 | 0 |
| renc | -0.055 | 187 | 0.452 | 0 | 0 | 0 |
| amo | -0.056 | 188 | 0.446 | 0 | 0 | 0 |
|  practical | -0.058 | 189 | 0.435 | 0 | 0 | 0 |
|  landed | -0.067 | 190 | 0.429 | 0 | 0 | 0 |
| ward | -0.068 | 191 | 0.423 | 0 | 0 | 0 |
| break | -0.072 | 192 | 0.488 | 0 | 0 | 0 |
|  complement | -0.074 | 193 | 0.458 | 0 | 0 | 0 |
| ait | -0.078 | 194 | 0.470 | 0 | 0 | 0 |
|  WHERE | -0.083 | 195 | 0.387 | 0 | 0 | 0 |

### Table 7 — held-out semantic mass and sign consistency

`mean_delta_contrast_prob` is the paired change in (unsafe-token mass − safe-token mass); `..._z` is the scale-invariant companion, which is the one whose sign is exact under the J/R lenses.

| condition | condition_kind | mean_delta_contrast_prob | mean_delta_contrast_z | sign_consistency_z | sign_consistency_prob | permutation_effect_size | permutation_p | topk_enrichment_positive | topk_enrichment_random | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 0.000 | 0.001 | 0.542 | 0.542 | -0.045 | 0.974 | 0.250 | 0.031 | 72 |
| normalize | baseline | 0.000 | 0.010 | 0.542 | 0.556 | 0.128 | 0.830 | 0.125 | 0.031 | 72 |
| rename_only | atomic | 0.000 | 0.035 | 0.472 | 0.514 | 0.617 | 0.512 | 0.250 | 0.062 | 72 |
| opaque_only | atomic | 0.000 | 0.026 | 0.569 | 0.556 | 0.411 | 0.618 | 0.125 | 0.031 | 72 |
| encode_only | atomic | 0.000 | 0.016 | 0.528 | 0.583 | 0.249 | 0.772 | 0.125 | 0.031 | 72 |
| flatten_only | atomic | 0.000 | 0.024 | 0.542 | 0.611 | 0.318 | 0.730 | 0.125 | 0.000 | 72 |
| rename_cumulative | cumulative | 0.000 | 0.108 | 0.583 | 0.583 | 1.268 | 0.218 | 0.125 | 0.031 | 72 |
| rename_opaque | cumulative | -0.000 | 0.137 | 0.583 | 0.542 | 1.614 | 0.130 | 0.125 | 0.062 | 72 |
| rename_opaque_encode | cumulative | 0.000 | 0.187 | 0.556 | 0.528 | 2.184 | 0.016 | 0.125 | 0.031 | 72 |
| rename_opaque_encode_flatten | cumulative | -0.000 | -0.026 | 0.542 | 0.403 | -0.267 | 0.746 | 0.000 | 0.000 | 72 |

### Table 8 — lens-method comparison by layer

| layer | relative_depth | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | topk_enrichment_positive |
|---|---|---|---|---|---|---|
| -1 |  | jlens | -0.003 | 0.125 | 0.900 | 0.000 |
| -1 |  | logit | 0.041 | 0.167 | 0.684 | 0.000 |
| -1 |  | rlens | 0.010 | 0.111 | 0.800 | 0.000 |
| 0 | 0.000 | jlens | -0.005 | 0.528 | 0.928 | 0.000 |
| 0 | 0.000 | logit | 0.036 | 0.500 | 0.608 | 0.000 |
| 0 | 0.000 | rlens | 0.021 | 0.528 | 0.634 | 0.250 |
| 3 | 0.097 | jlens | 0.054 | 0.597 | 0.148 | 0.000 |
| 3 | 0.097 | logit | 0.081 | 0.556 | 0.442 | 0.000 |
| 3 | 0.097 | rlens | 0.001 | 0.542 | 0.974 | 0.250 |
| 7 | 0.226 | jlens | -0.040 | 0.486 | 0.286 | 0.500 |
| 7 | 0.226 | logit | -0.026 | 0.556 | 0.786 | 0.625 |
| 7 | 0.226 | rlens | 0.023 | 0.528 | 0.726 | 0.750 |
| 11 | 0.355 | jlens | -0.004 | 0.528 | 0.924 | 0.750 |
| 11 | 0.355 | logit | -0.049 | 0.458 | 0.450 | 0.500 |
| 11 | 0.355 | rlens | -0.149 | 0.500 | 0.018 | 0.875 |
| 15 | 0.484 | jlens | -0.098 | 0.403 | 0.118 | 0.875 |
| 15 | 0.484 | logit | -0.175 | 0.403 | 0.012 | 0.750 |
| 15 | 0.484 | rlens | -0.200 | 0.403 | 0.004 | 0.750 |
| 19 | 0.613 | jlens | -0.319 | 0.208 | 0.000 | 0.625 |
| 19 | 0.613 | logit | -0.422 | 0.208 | 0.000 | 0.625 |
| 19 | 0.613 | rlens | -0.273 | 0.208 | 0.000 | 0.750 |
| 23 | 0.742 | jlens | -0.193 | 0.292 | 0.000 | 0.625 |
| 23 | 0.742 | logit | -0.314 | 0.264 | 0.000 | 0.625 |
| 23 | 0.742 | rlens | -0.217 | 0.306 | 0.000 | 0.750 |
| 27 | 0.871 | jlens | -0.209 | 0.333 | 0.000 | 0.750 |
| 27 | 0.871 | logit | -0.235 | 0.278 | 0.000 | 0.750 |
| 27 | 0.871 | rlens | -0.175 | 0.319 | 0.000 | 0.750 |
| 31 | 1.000 | jlens | -0.141 | 0.264 | 0.000 | 0.875 |
| 31 | 1.000 | logit | -0.141 | 0.264 | 0.000 | 0.875 |
| 31 | 1.000 | rlens | -0.141 | 0.264 | 0.000 | 0.875 |

Pairwise agreement of the three readouts' mean vocabulary-difference vectors:

| layer | lens_a | lens_b | cosine | spearman | n_tokens |
|---|---|---|---|---|---|
| 3 | jlens | logit | -0.006 | -0.027 | 196 |
| 3 | jlens | rlens | 0.218 | 0.198 | 196 |
| 3 | logit | rlens | 0.515 | 0.478 | 196 |

### Table 9 — semantic contrast across atomic and cumulative obfuscation

`cosine_to_clean` compares each condition's mean vocabulary-difference vector with the clean held-out one: accuracy asks whether a fitted direction still separates the classes, this asks whether the vocabulary-space difference still points the same way.

| condition | condition_kind | cosine_to_clean |
|---|---|---|
| clean_heldout | clean | 1.000 |
| normalize | baseline | 0.993 |
| rename_only | atomic | 0.395 |
| opaque_only | atomic | 0.985 |
| encode_only | atomic | 0.986 |
| flatten_only | atomic | 0.857 |
| rename_cumulative | cumulative | 0.046 |
| rename_opaque | cumulative | 0.245 |
| rename_opaque_encode | cumulative | 0.284 |
| rename_opaque_encode_flatten | cumulative | -0.019 |

### Table 10 — lens fidelity diagnostics (warnings, never blocking)

A weak row does not invalidate its layer. It is the reason the verdict separates *mechanically valid with weak fidelity* from *mechanically invalid*.

| lens | layer | is_control | next_token_top1 | next_token_mrr | final_layer_rank_agreement | relevance_conservation | weak_fidelity | warnings |
|---|---|---|---|---|---|---|---|---|
| gram_random | -1 | 1 |  |  | 0.069 |  | 1 | final-layer rank agreement 0.069 < 0.3 |
| gram_random | 0 | 1 |  |  | -0.116 |  | 1 | final-layer rank agreement -0.116 < 0.3 |
| gram_random | 3 | 1 |  |  | -0.111 |  | 1 | final-layer rank agreement -0.111 < 0.3 |
| gram_random | 7 | 1 |  |  | -0.026 |  | 1 | final-layer rank agreement -0.026 < 0.3 |
| gram_random | 11 | 1 |  |  | -0.051 |  | 1 | final-layer rank agreement -0.051 < 0.3 |
| gram_random | 15 | 1 |  |  | -0.015 |  | 1 | final-layer rank agreement -0.015 < 0.3 |
| gram_random | 19 | 1 |  |  | 0.018 |  | 1 | final-layer rank agreement 0.018 < 0.3 |
| gram_random | 23 | 1 |  |  | -0.041 |  | 1 | final-layer rank agreement -0.041 < 0.3 |
| gram_random | 27 | 1 |  |  | 0.008 |  | 1 | final-layer rank agreement 0.008 < 0.3 |
| gram_random | 31 | 1 |  |  | 0.038 |  | 1 | final-layer rank agreement 0.038 < 0.3 |
| jlens | -1 | 0 |  |  | -0.002 |  | 1 | final-layer rank agreement -0.002 < 0.3 |
| jlens | 0 | 0 |  |  | 0.073 |  | 1 | final-layer rank agreement 0.073 < 0.3 |
| jlens | 3 | 0 |  |  | 0.057 |  | 1 | final-layer rank agreement 0.057 < 0.3 |
| jlens | 7 | 0 |  |  | 0.133 |  | 1 | final-layer rank agreement 0.133 < 0.3 |
| jlens | 11 | 0 |  |  | 0.210 |  | 1 | final-layer rank agreement 0.210 < 0.3 |
| jlens | 15 | 0 |  |  | 0.253 |  | 1 | final-layer rank agreement 0.253 < 0.3 |
| jlens | 19 | 0 |  |  | 0.312 |  | 0 |  |
| jlens | 23 | 0 |  |  | 0.413 |  | 0 |  |
| jlens | 27 | 0 |  |  | 0.564 |  | 0 |  |
| jlens | 31 | 0 |  |  | 1.000 |  | 0 |  |
| logit | -1 | 0 |  |  | 0.047 |  | 1 | final-layer rank agreement 0.047 < 0.3 |
| logit | 0 | 0 |  |  | 0.034 |  | 1 | final-layer rank agreement 0.034 < 0.3 |
| logit | 3 | 0 |  |  | 0.078 |  | 1 | final-layer rank agreement 0.078 < 0.3 |
| logit | 7 | 0 |  |  | 0.122 |  | 1 | final-layer rank agreement 0.122 < 0.3 |
| logit | 11 | 0 |  |  | 0.165 |  | 1 | final-layer rank agreement 0.165 < 0.3 |
| logit | 15 | 0 |  |  | 0.236 |  | 1 | final-layer rank agreement 0.236 < 0.3 |
| logit | 19 | 0 |  |  | 0.347 |  | 0 |  |
| logit | 23 | 0 |  |  | 0.421 |  | 0 |  |
| logit | 27 | 0 |  |  | 0.518 |  | 0 |  |
| logit | 31 | 0 |  |  | 1.000 |  | 0 |  |
| random | -1 | 1 |  |  | 0.008 |  | 1 | final-layer rank agreement 0.008 < 0.3 |
| random | 0 | 1 |  |  | -0.009 |  | 1 | final-layer rank agreement -0.009 < 0.3 |
| random | 3 | 1 |  |  | -0.021 |  | 1 | final-layer rank agreement -0.021 < 0.3 |
| random | 7 | 1 |  |  | -0.014 |  | 1 | final-layer rank agreement -0.014 < 0.3 |
| random | 11 | 1 |  |  | -0.001 |  | 1 | final-layer rank agreement -0.001 < 0.3 |
| random | 15 | 1 |  |  | 0.001 |  | 1 | final-layer rank agreement 0.001 < 0.3 |
| random | 19 | 1 |  |  | -0.009 |  | 1 | final-layer rank agreement -0.009 < 0.3 |
| random | 23 | 1 |  |  | -0.008 |  | 1 | final-layer rank agreement -0.008 < 0.3 |
| random | 27 | 1 |  |  | -0.008 |  | 1 | final-layer rank agreement -0.008 < 0.3 |
| random | 31 | 1 |  |  | -0.003 |  | 1 | final-layer rank agreement -0.003 < 0.3 |
| rlens | -1 | 0 |  |  | -0.002 | 1.000 | 1 | final-layer rank agreement -0.002 < 0.3 |
| rlens | 0 | 0 |  |  | -0.059 | 1.000 | 1 | final-layer rank agreement -0.059 < 0.3 |
| rlens | 3 | 0 |  |  | 0.029 | 0.999 | 1 | final-layer rank agreement 0.029 < 0.3 |
| rlens | 7 | 0 |  |  | 0.208 | 0.999 | 1 | final-layer rank agreement 0.208 < 0.3 |
| rlens | 11 | 0 |  |  | 0.342 | 0.999 | 0 |  |
| rlens | 15 | 0 |  |  | 0.414 | 0.999 | 0 |  |
| rlens | 19 | 0 |  |  | 0.511 | 0.999 | 0 |  |
| rlens | 23 | 0 |  |  | 0.569 | 1.000 | 0 |  |
| rlens | 27 | 0 |  |  | 0.700 | 1.000 | 0 |  |
| rlens | 31 | 0 |  |  | 1.000 | 1.000 | 0 |  |

### Controls at the reported cell

| arm | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | n_pairs |
|---|---|---|---|---|---|
| mismatched_pairs | jlens | 0.079 | 0.653 | 0.030 | 72 |
| mismatched_pairs | logit | 0.020 | 0.528 | 0.812 | 72 |
| mismatched_pairs | rlens | 0.002 | 0.514 | 0.948 | 72 |
| random_lens | random | -0.012 | 0.472 | 0.868 | 72 |
| gram_random_lens | gram_random | 0.048 | 0.458 | 0.376 | 72 |
| role_swap_0 | jlens | 0.058 | 0.595 | 0.272 | 37 |
| role_swap_0 | logit | 0.023 | 0.514 | 0.822 | 37 |
| role_swap_0 | rlens | 0.031 | 0.486 | 0.634 | 37 |
| role_swap_1 | jlens | 0.050 | 0.600 | 0.334 | 35 |
| role_swap_1 | logit | 0.141 | 0.600 | 0.480 | 35 |
| role_swap_1 | rlens | -0.030 | 0.600 | 0.726 | 35 |

### Concept tokens at the reported cell

| token | mean_delta_z | rank | sign_consistency | mean_prob_unsafe | mean_prob_safe |
|---|---|---|---|---|---|
|  vulnerable | -0.024 | 144 | 0.528 | 0.005 | 0.005 |
|  safe | -0.036 | 159 | 0.417 | 0.005 | 0.005 |
|  trusted | -0.005 | 102 | 0.569 | 0.005 | 0.005 |
|  clean | -0.034 | 155 | 0.486 | 0.005 | 0.005 |

