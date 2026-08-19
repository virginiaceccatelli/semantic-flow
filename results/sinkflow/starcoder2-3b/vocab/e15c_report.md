# E15-C — vocabulary-space contrast (starcoder2-3b)

**Verdict.** STABLE NON-SECURITY VOCABULARY — the training-discovered directions replicate held out and beat the random-token control, but the security lexicon's own contrast does not carry it. Output-aligned flow information WITHOUT explicit verbalisation; do not call this 'the model represents unsafe'.

Primary lens `rlens` (declared before any result was produced); reported at lens `rlens`, site `sink_arg`, layer 15, condition `clean_heldout`.

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
| OrNull | 0.447 | 0 | 0.839 | 0 | 0 | 0 |
| displayMode | 0.434 | 1 | 0.923 | 0 | 0 | 0 |
|  or | 0.409 | 2 | 0.696 | 0 | 0 | 0 |
| fuchsia | 0.350 | 3 | 0.768 | 0 | 0 | 0 |
|  CORS | 0.334 | 4 | 0.821 | 0 | 0 | 1 |
| eu | 0.332 | 5 | 0.899 | 0 | 0 | 0 |
| putc | 0.326 | 6 | 0.815 | 0 | 0 | 0 |
| ásá | 0.326 | 7 | 0.708 | 0 | 0 | 0 |
| ,/ | 0.315 | 8 | 0.679 | 0 | 0 | 0 |
| ogenerated | 0.301 | 9 | 0.839 | 0 | 0 | 0 |
| � | -0.263 | 186 | 0.214 | 0 | 0 | 1 |
| η | -0.270 | 187 | 0.167 | 0 | 0 | 0 |
| Gq | -0.312 | 188 | 0.220 | 0 | 0 | 0 |
| Wq | -0.314 | 189 | 0.262 | 0 | 0 | 0 |
| opunto | -0.314 | 190 | 0.262 | 0 | 0 | 0 |
| Zc | -0.315 | 191 | 0.268 | 0 | 0 | 0 |
| Vg | -0.325 | 192 | 0.131 | 0 | 0 | 0 |
| Qc | -0.336 | 193 | 0.167 | 0 | 0 | 0 |
| ITERAL | -0.346 | 194 | 0.208 | 0 | 0 | 0 |
| dating | -0.361 | 195 | 0.137 | 0 | 0 | 0 |

### Table 7 — held-out semantic mass and sign consistency

`mean_delta_contrast_prob` is the paired change in (unsafe-token mass − safe-token mass); `..._z` is the scale-invariant companion, which is the one whose sign is exact under the J/R lenses.

| condition | condition_kind | mean_delta_contrast_prob | mean_delta_contrast_z | sign_consistency_z | sign_consistency_prob | permutation_effect_size | permutation_p | topk_enrichment_positive | topk_enrichment_random | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | 0.000 | 0.094 | 0.694 | 0.569 | 2.683 | 0.008 | 0.875 | 0.031 | 72 |
| normalize | baseline | 0.000 | 0.095 | 0.694 | 0.556 | 2.643 | 0.006 | 0.750 | 0.031 | 72 |
| rename_only | atomic | -0.000 | -0.008 | 0.514 | 0.444 | -0.220 | 0.822 | 0.375 | 0.000 | 72 |
| opaque_only | atomic | -0.000 | 0.057 | 0.597 | 0.542 | 1.737 | 0.102 | 0.625 | 0.000 | 72 |
| encode_only | atomic | 0.000 | 0.077 | 0.667 | 0.569 | 2.389 | 0.018 | 0.875 | 0.031 | 72 |
| flatten_only | atomic | -0.000 | 0.031 | 0.583 | 0.431 | 0.918 | 0.392 | 0.125 | 0.031 | 72 |
| rename_cumulative | cumulative | -0.000 | -0.029 | 0.514 | 0.458 | -0.917 | 0.374 | 0.500 | 0.000 | 72 |
| rename_opaque | cumulative | -0.000 | -0.022 | 0.431 | 0.403 | -0.612 | 0.502 | 0.375 | 0.000 | 72 |
| rename_opaque_encode | cumulative | -0.000 | -0.010 | 0.458 | 0.417 | -0.295 | 0.784 | 0.500 | 0.031 | 72 |
| rename_opaque_encode_flatten | cumulative | -0.000 | 0.000 | 0.472 | 0.444 | -0.009 | 0.996 | 0.125 | 0.031 | 72 |

### Table 8 — lens-method comparison by layer

| layer | relative_depth | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | topk_enrichment_positive |
|---|---|---|---|---|---|---|
| -1 |  | jlens | -0.006 | 0.111 | 0.946 | 0.000 |
| -1 |  | logit | 0.026 | 0.125 | 0.718 | 0.000 |
| -1 |  | rlens | -0.021 | 0.125 | 0.710 | 0.000 |
| 0 | 0.000 | jlens | 0.008 | 0.486 | 0.862 | 0.000 |
| 0 | 0.000 | logit | -0.009 | 0.542 | 0.874 | 0.000 |
| 0 | 0.000 | rlens | -0.005 | 0.486 | 0.942 | 0.000 |
| 3 | 0.103 | jlens | -0.037 | 0.500 | 0.522 | 0.000 |
| 3 | 0.103 | logit | -0.028 | 0.569 | 0.726 | 0.125 |
| 3 | 0.103 | rlens | -0.060 | 0.556 | 0.238 | 0.250 |
| 7 | 0.241 | jlens | 0.038 | 0.583 | 0.252 | 0.625 |
| 7 | 0.241 | logit | -0.005 | 0.500 | 0.934 | 0.875 |
| 7 | 0.241 | rlens | 0.040 | 0.583 | 0.240 | 0.750 |
| 11 | 0.379 | jlens | 0.100 | 0.681 | 0.000 | 0.750 |
| 11 | 0.379 | logit | -0.082 | 0.389 | 0.118 | 0.875 |
| 11 | 0.379 | rlens | 0.098 | 0.625 | 0.008 | 0.750 |
| 15 | 0.517 | jlens | 0.084 | 0.681 | 0.010 | 0.875 |
| 15 | 0.517 | logit | -0.041 | 0.431 | 0.482 | 0.625 |
| 15 | 0.517 | rlens | 0.094 | 0.694 | 0.008 | 0.875 |
| 19 | 0.655 | jlens | -0.165 | 0.389 | 0.000 | 0.750 |
| 19 | 0.655 | logit | -0.249 | 0.236 | 0.000 | 0.500 |
| 19 | 0.655 | rlens | -0.135 | 0.389 | 0.002 | 0.750 |
| 23 | 0.793 | jlens | -0.035 | 0.403 | 0.288 | 1.000 |
| 23 | 0.793 | logit | -0.214 | 0.222 | 0.000 | 0.625 |
| 23 | 0.793 | rlens | -0.032 | 0.431 | 0.356 | 0.875 |
| 27 | 0.931 | jlens | 0.007 | 0.514 | 0.764 | 0.875 |
| 27 | 0.931 | logit | -0.035 | 0.431 | 0.384 | 0.875 |
| 27 | 0.931 | rlens | 0.006 | 0.500 | 0.802 | 0.875 |
| 29 | 1.000 | jlens | 0.036 | 0.500 | 0.172 | 0.750 |
| 29 | 1.000 | logit | 0.036 | 0.486 | 0.172 | 0.750 |
| 29 | 1.000 | rlens | 0.036 | 0.500 | 0.172 | 0.750 |

Pairwise agreement of the three readouts' mean vocabulary-difference vectors:

| layer | lens_a | lens_b | cosine | spearman | n_tokens |
|---|---|---|---|---|---|
| 15 | jlens | logit | 0.750 | 0.753 | 196 |
| 15 | jlens | rlens | 0.956 | 0.955 | 196 |
| 15 | logit | rlens | 0.774 | 0.776 | 196 |

### Table 9 — semantic contrast across atomic and cumulative obfuscation

`cosine_to_clean` compares each condition's mean vocabulary-difference vector with the clean held-out one: accuracy asks whether a fitted direction still separates the classes, this asks whether the vocabulary-space difference still points the same way.

| condition | condition_kind | cosine_to_clean |
|---|---|---|
| clean_heldout | clean | 1.000 |
| normalize | baseline | 0.999 |
| rename_only | atomic | 0.866 |
| opaque_only | atomic | 0.979 |
| encode_only | atomic | 0.982 |
| flatten_only | atomic | 0.491 |
| rename_cumulative | cumulative | 0.851 |
| rename_opaque | cumulative | 0.847 |
| rename_opaque_encode | cumulative | 0.852 |
| rename_opaque_encode_flatten | cumulative | 0.643 |

### Table 10 — lens fidelity diagnostics (warnings, never blocking)

A weak row does not invalidate its layer. It is the reason the verdict separates *mechanically valid with weak fidelity* from *mechanically invalid*.

| lens | layer | is_control | next_token_top1 | next_token_mrr | final_layer_rank_agreement | relevance_conservation | weak_fidelity | warnings |
|---|---|---|---|---|---|---|---|---|
| gram_random | -1 | 1 |  |  | -0.111 |  | 1 | final-layer rank agreement -0.111 < 0.3 |
| gram_random | 0 | 1 |  |  | 0.109 |  | 1 | final-layer rank agreement 0.109 < 0.3 |
| gram_random | 3 | 1 |  |  | -0.040 |  | 1 | final-layer rank agreement -0.040 < 0.3 |
| gram_random | 7 | 1 |  |  | -0.061 |  | 1 | final-layer rank agreement -0.061 < 0.3 |
| gram_random | 11 | 1 |  |  | -0.111 |  | 1 | final-layer rank agreement -0.111 < 0.3 |
| gram_random | 15 | 1 |  |  | -0.100 |  | 1 | final-layer rank agreement -0.100 < 0.3 |
| gram_random | 19 | 1 |  |  | -0.054 |  | 1 | final-layer rank agreement -0.054 < 0.3 |
| gram_random | 23 | 1 |  |  | -0.063 |  | 1 | final-layer rank agreement -0.063 < 0.3 |
| gram_random | 27 | 1 |  |  | -0.116 |  | 1 | final-layer rank agreement -0.116 < 0.3 |
| gram_random | 29 | 1 |  |  | -0.013 |  | 1 | final-layer rank agreement -0.013 < 0.3 |
| jlens | -1 | 0 |  |  | -0.050 |  | 1 | final-layer rank agreement -0.050 < 0.3 |
| jlens | 0 | 0 |  |  | 0.015 |  | 1 | final-layer rank agreement 0.015 < 0.3 |
| jlens | 3 | 0 |  |  | 0.061 |  | 1 | final-layer rank agreement 0.061 < 0.3 |
| jlens | 7 | 0 |  |  | 0.227 |  | 1 | final-layer rank agreement 0.227 < 0.3 |
| jlens | 11 | 0 |  |  | 0.275 |  | 1 | final-layer rank agreement 0.275 < 0.3 |
| jlens | 15 | 0 |  |  | 0.328 |  | 0 |  |
| jlens | 19 | 0 |  |  | 0.448 |  | 0 |  |
| jlens | 23 | 0 |  |  | 0.592 |  | 0 |  |
| jlens | 27 | 0 |  |  | 0.806 |  | 0 |  |
| jlens | 29 | 0 |  |  | 0.982 |  | 0 |  |
| logit | -1 | 0 |  |  | 0.273 |  | 1 | final-layer rank agreement 0.273 < 0.3 |
| logit | 0 | 0 |  |  | 0.166 |  | 1 | final-layer rank agreement 0.166 < 0.3 |
| logit | 3 | 0 |  |  | 0.078 |  | 1 | final-layer rank agreement 0.078 < 0.3 |
| logit | 7 | 0 |  |  | 0.150 |  | 1 | final-layer rank agreement 0.150 < 0.3 |
| logit | 11 | 0 |  |  | 0.120 |  | 1 | final-layer rank agreement 0.120 < 0.3 |
| logit | 15 | 0 |  |  | 0.183 |  | 1 | final-layer rank agreement 0.183 < 0.3 |
| logit | 19 | 0 |  |  | 0.277 |  | 1 | final-layer rank agreement 0.277 < 0.3 |
| logit | 23 | 0 |  |  | 0.415 |  | 0 |  |
| logit | 27 | 0 |  |  | 0.561 |  | 0 |  |
| logit | 29 | 0 |  |  | 0.982 |  | 0 |  |
| random | -1 | 1 |  |  | -0.003 |  | 1 | final-layer rank agreement -0.003 < 0.3 |
| random | 0 | 1 |  |  | 0.022 |  | 1 | final-layer rank agreement 0.022 < 0.3 |
| random | 3 | 1 |  |  | 0.029 |  | 1 | final-layer rank agreement 0.029 < 0.3 |
| random | 7 | 1 |  |  | 0.025 |  | 1 | final-layer rank agreement 0.025 < 0.3 |
| random | 11 | 1 |  |  | 0.039 |  | 1 | final-layer rank agreement 0.039 < 0.3 |
| random | 15 | 1 |  |  | 0.041 |  | 1 | final-layer rank agreement 0.041 < 0.3 |
| random | 19 | 1 |  |  | 0.042 |  | 1 | final-layer rank agreement 0.042 < 0.3 |
| random | 23 | 1 |  |  | 0.036 |  | 1 | final-layer rank agreement 0.036 < 0.3 |
| random | 27 | 1 |  |  | 0.029 |  | 1 | final-layer rank agreement 0.029 < 0.3 |
| random | 29 | 1 |  |  | 0.029 |  | 1 | final-layer rank agreement 0.029 < 0.3 |
| rlens | -1 | 0 |  |  | -0.003 | -0.012 | 1 | final-layer rank agreement -0.003 < 0.3; relevance conservation -0.012 is 1.012 from 1 |
| rlens | 0 | 0 |  |  | -0.071 | -0.108 | 1 | final-layer rank agreement -0.071 < 0.3; relevance conservation -0.108 is 1.108 from 1 |
| rlens | 3 | 0 |  |  | 0.078 | -0.165 | 1 | final-layer rank agreement 0.078 < 0.3; relevance conservation -0.165 is 1.165 from 1 |
| rlens | 7 | 0 |  |  | 0.139 | -0.014 | 1 | final-layer rank agreement 0.139 < 0.3; relevance conservation -0.014 is 1.014 from 1 |
| rlens | 11 | 0 |  |  | 0.219 | 0.008 | 1 | final-layer rank agreement 0.219 < 0.3; relevance conservation 0.008 is 0.992 from 1 |
| rlens | 15 | 0 |  |  | 0.306 | 0.154 | 1 | relevance conservation 0.154 is 0.846 from 1 |
| rlens | 19 | 0 |  |  | 0.494 | 0.054 | 1 | relevance conservation 0.054 is 0.946 from 1 |
| rlens | 23 | 0 |  |  | 0.626 | 0.070 | 1 | relevance conservation 0.070 is 0.930 from 1 |
| rlens | 27 | 0 |  |  | 0.811 | 0.125 | 1 | relevance conservation 0.125 is 0.875 from 1 |
| rlens | 29 | 0 |  |  | 0.982 | 1.000 | 0 |  |

### Controls at the reported cell

| arm | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | n_pairs |
|---|---|---|---|---|---|
| mismatched_pairs | jlens | 0.082 | 0.597 | 0.010 | 72 |
| mismatched_pairs | logit | -0.037 | 0.375 | 0.490 | 72 |
| mismatched_pairs | rlens | 0.090 | 0.639 | 0.002 | 72 |
| random_lens | random | 0.117 | 0.611 | 0.066 | 72 |
| gram_random_lens | gram_random | 0.098 | 0.639 | 0.022 | 72 |
| role_swap_0 | jlens | 0.047 | 0.622 | 0.296 | 37 |
| role_swap_0 | logit | -0.193 | 0.324 | 0.024 | 37 |
| role_swap_0 | rlens | 0.048 | 0.622 | 0.338 | 37 |
| role_swap_1 | jlens | 0.123 | 0.743 | 0.006 | 35 |
| role_swap_1 | logit | 0.120 | 0.543 | 0.196 | 35 |
| role_swap_1 | rlens | 0.143 | 0.771 | 0.000 | 35 |

### Concept tokens at the reported cell

| token | mean_delta_z | rank | sign_consistency | mean_prob_unsafe | mean_prob_safe |
|---|---|---|---|---|---|
|  unsafe | 0.117 | 52 | 0.611 | 0.005 | 0.005 |
|  safe | 0.084 | 62 | 0.569 | 0.005 | 0.005 |
|  trusted | -0.006 | 96 | 0.431 | 0.005 | 0.005 |
|  clean | -0.010 | 98 | 0.583 | 0.005 | 0.005 |

