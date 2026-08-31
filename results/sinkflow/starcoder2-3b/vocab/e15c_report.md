# E15-C — vocabulary-space contrast (starcoder2-3b)

## What this experiment asks

This report asks whether the safe/unsafe difference appears in vocabulary coordinates, especially in a predeclared security-word set. It is an observational readout: direction, sign, held-out replication, and controls must all be read together, and an inverted sign must not be described as an unsafe concept.

**Verdict.** STABLE NON-SECURITY VOCABULARY — the training-discovered directions replicate held out and beat the random-token control, but the security lexicon's own contrast does not carry it. Output-aligned flow information WITHOUT explicit verbalisation; do not call this 'the model represents unsafe'.

Primary lens `clrp` (declared before any result was produced); reported at lens `clrp`, site `sink_arg`, layer 15, condition `clean_heldout`.

| check | holds |
|---|---|
| discovery_train_only_and_frozen | yes |
| held_out_replication | no |
| consistent_orientation | yes |
| above_permutation_control | yes |
| above_mismatched_pair_control | yes |
| above_same_label_control | yes |
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
| -1 |  | clens | -0.006 | 0.111 | 0.946 | 0.000 |
| -1 |  | logit | 0.026 | 0.125 | 0.718 | 0.000 |
| -1 |  | clrp | -0.021 | 0.125 | 0.710 | 0.000 |
| 0 | 0.000 | clens | 0.008 | 0.486 | 0.862 | 0.000 |
| 0 | 0.000 | logit | -0.009 | 0.542 | 0.874 | 0.000 |
| 0 | 0.000 | clrp | -0.005 | 0.486 | 0.942 | 0.000 |
| 3 | 0.103 | clens | -0.037 | 0.500 | 0.522 | 0.000 |
| 3 | 0.103 | logit | -0.028 | 0.569 | 0.726 | 0.125 |
| 3 | 0.103 | clrp | -0.060 | 0.556 | 0.238 | 0.250 |
| 7 | 0.241 | clens | 0.038 | 0.583 | 0.252 | 0.625 |
| 7 | 0.241 | logit | -0.005 | 0.500 | 0.934 | 0.875 |
| 7 | 0.241 | clrp | 0.040 | 0.583 | 0.240 | 0.750 |
| 11 | 0.379 | clens | 0.100 | 0.681 | 0.000 | 0.750 |
| 11 | 0.379 | logit | -0.082 | 0.389 | 0.118 | 0.875 |
| 11 | 0.379 | clrp | 0.098 | 0.625 | 0.008 | 0.750 |
| 15 | 0.517 | clens | 0.084 | 0.681 | 0.010 | 0.875 |
| 15 | 0.517 | logit | -0.041 | 0.431 | 0.482 | 0.625 |
| 15 | 0.517 | clrp | 0.094 | 0.694 | 0.008 | 0.875 |
| 19 | 0.655 | clens | -0.165 | 0.389 | 0.000 | 0.750 |
| 19 | 0.655 | logit | -0.249 | 0.236 | 0.000 | 0.500 |
| 19 | 0.655 | clrp | -0.135 | 0.389 | 0.002 | 0.750 |
| 23 | 0.793 | clens | -0.035 | 0.403 | 0.288 | 1.000 |
| 23 | 0.793 | logit | -0.214 | 0.222 | 0.000 | 0.625 |
| 23 | 0.793 | clrp | -0.032 | 0.431 | 0.356 | 0.875 |
| 27 | 0.931 | clens | 0.007 | 0.514 | 0.764 | 0.875 |
| 27 | 0.931 | logit | -0.035 | 0.431 | 0.384 | 0.875 |
| 27 | 0.931 | clrp | 0.006 | 0.500 | 0.802 | 0.875 |
| 29 | 1.000 | clens | 0.036 | 0.500 | 0.172 | 0.750 |
| 29 | 1.000 | logit | 0.036 | 0.486 | 0.172 | 0.750 |
| 29 | 1.000 | clrp | 0.036 | 0.500 | 0.172 | 0.750 |

Pairwise agreement of the three readouts' mean vocabulary-difference vectors:

| layer | lens_a | lens_b | cosine | spearman | n_tokens |
|---|---|---|---|---|---|
| 15 | clens | logit | 0.750 | 0.753 | 196 |
| 15 | clens | clrp | 0.956 | 0.955 | 196 |
| 15 | logit | clrp | 0.774 | 0.776 | 196 |

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

### Table 11 — specificity: is the effect better than a random direction?

The permutation null asks whether the safe→unsafe *orientation* carries the effect. It does not ask whether **this** direction in the residual stream is special. `specificity` is the real arm's displacement from chance over the largest displacement any random or Gram-matched lens reaches in the same cell: **at or below 1.0, the result is not specific to the lens.**

| lens | layer | relative_depth | sign_consistency_z | permutation_p | displacement | control_displacement | specificity | beats_random_lens |
|---|---|---|---|---|---|---|---|---|
| clens | -1 |  | 0.111 | 0.946 | 0.389 | 0.375 | 1.037 | True |
| clens | 0 | 0.000 | 0.486 | 0.862 | 0.014 | 0.083 | 0.167 | False |
| clens | 3 | 0.103 | 0.500 | 0.522 | 0.000 | 0.097 | 0.000 | False |
| clens | 7 | 0.241 | 0.583 | 0.252 | 0.083 | 0.194 | 0.429 | False |
| clens | 11 | 0.379 | 0.681 | 0.000 | 0.181 | 0.208 | 0.867 | False |
| clens | 15 | 0.517 | 0.681 | 0.010 | 0.181 | 0.139 | 1.300 | True |
| clens | 19 | 0.655 | 0.389 | 0.000 | 0.111 | 0.194 | 0.571 | False |
| clens | 23 | 0.793 | 0.403 | 0.288 | 0.097 | 0.181 | 0.538 | False |
| clens | 27 | 0.931 | 0.514 | 0.764 | 0.014 | 0.125 | 0.111 | False |
| clens | 29 | 1.000 | 0.500 | 0.172 | 0.000 | 0.028 | 0.000 | False |
| logit | -1 |  | 0.125 | 0.718 | 0.375 | 0.375 | 1.000 | False |
| logit | 0 | 0.000 | 0.542 | 0.874 | 0.042 | 0.083 | 0.500 | False |
| logit | 3 | 0.103 | 0.569 | 0.726 | 0.069 | 0.097 | 0.714 | False |
| logit | 7 | 0.241 | 0.500 | 0.934 | 0.000 | 0.194 | 0.000 | False |
| logit | 11 | 0.379 | 0.389 | 0.118 | 0.111 | 0.208 | 0.533 | False |
| logit | 15 | 0.517 | 0.431 | 0.482 | 0.069 | 0.139 | 0.500 | False |
| logit | 19 | 0.655 | 0.236 | 0.000 | 0.264 | 0.194 | 1.357 | True |
| logit | 23 | 0.793 | 0.222 | 0.000 | 0.278 | 0.181 | 1.538 | True |
| logit | 27 | 0.931 | 0.431 | 0.384 | 0.069 | 0.125 | 0.556 | False |
| logit | 29 | 1.000 | 0.486 | 0.172 | 0.014 | 0.028 | 0.500 | False |
| clrp | -1 |  | 0.125 | 0.710 | 0.375 | 0.375 | 1.000 | False |
| clrp | 0 | 0.000 | 0.486 | 0.942 | 0.014 | 0.083 | 0.167 | False |
| clrp | 3 | 0.103 | 0.556 | 0.238 | 0.056 | 0.097 | 0.571 | False |
| clrp | 7 | 0.241 | 0.583 | 0.240 | 0.083 | 0.194 | 0.429 | False |
| clrp | 11 | 0.379 | 0.625 | 0.008 | 0.125 | 0.208 | 0.600 | False |
| clrp | 15 | 0.517 | 0.694 | 0.008 | 0.194 | 0.139 | 1.400 | True |
| clrp | 19 | 0.655 | 0.389 | 0.002 | 0.111 | 0.194 | 0.571 | False |
| clrp | 23 | 0.793 | 0.431 | 0.356 | 0.069 | 0.181 | 0.385 | False |
| clrp | 27 | 0.931 | 0.500 | 0.802 | 0.000 | 0.125 | 0.000 | False |
| clrp | 29 | 1.000 | 0.500 | 0.172 | 0.000 | 0.028 | 0.000 | False |

### Table 12 — is the contrast a distribution artifact?

`corr_contrast_entropy` and `corr_contrast_norm` correlate the paired contrast against the paired difference in the candidate distribution's entropy and score norm. A large |r| means the contrast tracks the *shape* of the distribution rather than its content, which would explain a consistent sign without any concept being involved.

| condition | mean_delta_contrast_z | corr_contrast_entropy | corr_contrast_norm | mean_delta_entropy |
|---|---|---|---|---|
| clean_heldout | 0.094 | 0.138 | 0.100 | 0.000 |
| normalize | 0.095 | 0.121 | 0.119 | 0.000 |
| rename_only | -0.008 | 0.341 | 0.056 | 0.000 |
| opaque_only | 0.057 | -0.012 | 0.127 | 0.000 |
| encode_only | 0.077 | 0.167 | -0.145 | 0.000 |
| flatten_only | 0.031 | 0.170 | 0.003 | 0.000 |
| rename_cumulative | -0.029 | 0.349 | -0.117 | 0.000 |
| rename_opaque | -0.022 | 0.169 | -0.042 | -0.000 |
| rename_opaque_encode | -0.010 | 0.081 | -0.002 | 0.000 |
| rename_opaque_encode_flatten | 0.000 | -0.025 | -0.276 | -0.000 |

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
| clens | -1 | 0 |  |  | -0.050 |  | 1 | final-layer rank agreement -0.050 < 0.3 |
| clens | 0 | 0 |  |  | 0.015 |  | 1 | final-layer rank agreement 0.015 < 0.3 |
| clens | 3 | 0 |  |  | 0.061 |  | 1 | final-layer rank agreement 0.061 < 0.3 |
| clens | 7 | 0 |  |  | 0.227 |  | 1 | final-layer rank agreement 0.227 < 0.3 |
| clens | 11 | 0 |  |  | 0.275 |  | 1 | final-layer rank agreement 0.275 < 0.3 |
| clens | 15 | 0 |  |  | 0.328 |  | 0 |  |
| clens | 19 | 0 |  |  | 0.448 |  | 0 |  |
| clens | 23 | 0 |  |  | 0.592 |  | 0 |  |
| clens | 27 | 0 |  |  | 0.806 |  | 0 |  |
| clens | 29 | 0 |  |  | 0.982 |  | 0 |  |
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
| clrp | -1 | 0 |  |  | -0.003 | -0.012 | 1 | final-layer rank agreement -0.003 < 0.3; relevance conservation -0.012 is 1.012 from 1 |
| clrp | 0 | 0 |  |  | -0.071 | -0.108 | 1 | final-layer rank agreement -0.071 < 0.3; relevance conservation -0.108 is 1.108 from 1 |
| clrp | 3 | 0 |  |  | 0.078 | -0.165 | 1 | final-layer rank agreement 0.078 < 0.3; relevance conservation -0.165 is 1.165 from 1 |
| clrp | 7 | 0 |  |  | 0.139 | -0.014 | 1 | final-layer rank agreement 0.139 < 0.3; relevance conservation -0.014 is 1.014 from 1 |
| clrp | 11 | 0 |  |  | 0.219 | 0.008 | 1 | final-layer rank agreement 0.219 < 0.3; relevance conservation 0.008 is 0.992 from 1 |
| clrp | 15 | 0 |  |  | 0.306 | 0.154 | 1 | relevance conservation 0.154 is 0.846 from 1 |
| clrp | 19 | 0 |  |  | 0.494 | 0.054 | 1 | relevance conservation 0.054 is 0.946 from 1 |
| clrp | 23 | 0 |  |  | 0.626 | 0.070 | 1 | relevance conservation 0.070 is 0.930 from 1 |
| clrp | 27 | 0 |  |  | 0.811 | 0.125 | 1 | relevance conservation 0.125 is 0.875 from 1 |
| clrp | 29 | 0 |  |  | 0.982 | 1.000 | 0 |  |

### Controls at the reported cell

`mismatched_pairs` redraws the SAFE partner from the safe pool, so the label difference survives it and its mean is invariant by construction; it can only move the per-pair statistics. `same_label_unsafe` and `same_label_safe` take BOTH members from one pole, so the label difference is gone and the expected contrast is zero — that is the arm a label claim has to clear.

| arm | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | n_pairs |
|---|---|---|---|---|---|
| mismatched_pairs | clens | 0.082 | 0.597 | 0.010 | 72 |
| mismatched_pairs | logit | -0.037 | 0.375 | 0.490 | 72 |
| mismatched_pairs | clrp | 0.090 | 0.639 | 0.002 | 72 |
| same_label_unsafe | clens | 0.027 | 0.542 | 0.422 | 72 |
| same_label_unsafe | logit | -0.025 | 0.458 | 0.652 | 72 |
| same_label_unsafe | clrp | 0.028 | 0.528 | 0.424 | 72 |
| same_label_safe | clens | -0.002 | 0.514 | 0.936 | 72 |
| same_label_safe | logit | 0.004 | 0.542 | 0.930 | 72 |
| same_label_safe | clrp | -0.004 | 0.528 | 0.886 | 72 |
| random_lens | random | 0.117 | 0.611 | 0.066 | 72 |
| gram_random_lens | gram_random | 0.098 | 0.639 | 0.022 | 72 |
| role_swap_0 | clens | 0.047 | 0.622 | 0.296 | 37 |
| role_swap_0 | logit | -0.193 | 0.324 | 0.024 | 37 |
| role_swap_0 | clrp | 0.048 | 0.622 | 0.338 | 37 |
| role_swap_1 | clens | 0.123 | 0.743 | 0.006 | 35 |
| role_swap_1 | logit | 0.120 | 0.543 | 0.196 | 35 |
| role_swap_1 | clrp | 0.143 | 0.771 | 0.000 | 35 |

### Concept tokens at the reported cell

| token | mean_delta_z | rank | sign_consistency | mean_prob_unsafe | mean_prob_safe |
|---|---|---|---|---|---|
|  unsafe | 0.117 | 52 | 0.611 | 0.005 | 0.005 |
|  safe | 0.084 | 62 | 0.569 | 0.005 | 0.005 |
|  trusted | -0.006 | 96 | 0.431 | 0.005 | 0.005 |
|  clean | -0.010 | 98 | 0.583 | 0.005 | 0.005 |

