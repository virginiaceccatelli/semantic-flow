# E15-C — vocabulary-space contrast (deepseek-coder-6.7b)

## What this experiment asks

This report asks whether the safe/unsafe difference appears in vocabulary coordinates, especially in a predeclared security-word set. It is an observational readout: direction, sign, held-out replication, and controls must all be read together, and an inverted sign must not be described as an unsafe concept.

**Verdict.** STABLE NON-SECURITY VOCABULARY — the training-discovered directions replicate held out and beat the random-token control, but the security lexicon's own contrast does not carry it. Output-aligned flow information WITHOUT explicit verbalisation; do not call this 'the model represents unsafe'.

Primary lens `rlens` (declared before any result was produced); reported at lens `rlens`, site `sink_arg`, layer 15, condition `clean_heldout`.

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
|  liber | 0.911 | 0 | 0.970 | 0 | 0 | 0 |
| Clean | 0.880 | 1 | 0.976 | 0 | 0 | 0 |
| clean | 0.874 | 2 | 0.952 | 0 | 0 | 0 |
| pty | 0.825 | 3 | 0.940 | 0 | 0 | 0 |
| mart | 0.793 | 4 | 0.875 | 0 | 0 | 0 |
| tbl | 0.777 | 5 | 0.940 | 0 | 0 | 0 |
| shape | 0.768 | 6 | 0.940 | 0 | 0 | 0 |
|  injection | 0.766 | 7 | 0.935 | 0 | 0 | 0 |
|  caut | 0.762 | 8 | 0.946 | 0 | 0 | 0 |
| break | 0.758 | 9 | 0.869 | 0 | 0 | 0 |
|  Dick | -0.758 | 186 | 0.089 | 0 | 0 | 0 |
| whose | -0.761 | 187 | 0.131 | 0 | 0 | 0 |
|  wide | -0.763 | 188 | 0.089 | 0 | 0 | 0 |
|  accompany | -0.776 | 189 | 0.054 | 0 | 0 | 0 |
| plus | -0.786 | 190 | 0.054 | 0 | 0 | 0 |
| ista | -0.791 | 191 | 0.137 | 0 | 0 | 0 |
|  followed | -0.806 | 192 | 0.119 | 0 | 0 | 0 |
|  complement | -0.825 | 193 | 0.083 | 0 | 0 | 0 |
|  plus | -0.832 | 194 | 0.077 | 0 | 0 | 0 |
|  SF | -0.844 | 195 | 0.077 | 0 | 0 | 0 |

### Table 7 — held-out semantic mass and sign consistency

`mean_delta_contrast_prob` is the paired change in (unsafe-token mass − safe-token mass); `..._z` is the scale-invariant companion, which is the one whose sign is exact under the J/R lenses.

| condition | condition_kind | mean_delta_contrast_prob | mean_delta_contrast_z | sign_consistency_z | sign_consistency_prob | permutation_effect_size | permutation_p | topk_enrichment_positive | topk_enrichment_random | n_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_heldout | clean | -0.002 | -0.200 | 0.403 | 0.153 | -3.125 | 0.004 | 0.750 | 0.000 | 72 |
| normalize | baseline | -0.002 | -0.209 | 0.403 | 0.153 | -3.260 | 0.000 | 0.750 | 0.000 | 72 |
| rename_only | atomic | -0.002 | 0.007 | 0.514 | 0.111 | 0.090 | 0.912 | 0.625 | 0.000 | 72 |
| opaque_only | atomic | -0.002 | -0.070 | 0.431 | 0.167 | -1.189 | 0.230 | 0.875 | 0.000 | 72 |
| encode_only | atomic | -0.002 | -0.121 | 0.458 | 0.125 | -2.064 | 0.050 | 0.750 | 0.000 | 72 |
| flatten_only | atomic | -0.000 | 0.019 | 0.472 | 0.444 | 0.169 | 0.782 | 0.125 | 0.000 | 72 |
| rename_cumulative | cumulative | -0.002 | -0.023 | 0.514 | 0.097 | -0.494 | 0.660 | 0.625 | 0.000 | 72 |
| rename_opaque | cumulative | -0.002 | 0.020 | 0.528 | 0.139 | 0.436 | 0.696 | 0.500 | 0.000 | 72 |
| rename_opaque_encode | cumulative | -0.002 | 0.021 | 0.514 | 0.153 | 0.367 | 0.696 | 0.625 | 0.000 | 72 |
| rename_opaque_encode_flatten | cumulative | -0.001 | -0.120 | 0.458 | 0.361 | -1.809 | 0.070 | 0.500 | 0.031 | 72 |

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
| 15 | jlens | logit | 0.913 | 0.905 | 196 |
| 15 | jlens | rlens | 0.958 | 0.959 | 196 |
| 15 | logit | rlens | 0.974 | 0.959 | 196 |

### Table 9 — semantic contrast across atomic and cumulative obfuscation

`cosine_to_clean` compares each condition's mean vocabulary-difference vector with the clean held-out one: accuracy asks whether a fitted direction still separates the classes, this asks whether the vocabulary-space difference still points the same way.

| condition | condition_kind | cosine_to_clean |
|---|---|---|
| clean_heldout | clean | 1.000 |
| normalize | baseline | 1.000 |
| rename_only | atomic | 0.950 |
| opaque_only | atomic | 0.993 |
| encode_only | atomic | 0.996 |
| flatten_only | atomic | 0.846 |
| rename_cumulative | cumulative | 0.944 |
| rename_opaque | cumulative | 0.938 |
| rename_opaque_encode | cumulative | 0.945 |
| rename_opaque_encode_flatten | cumulative | 0.812 |

### Table 11 — specificity: is the effect better than a random direction?

The permutation null asks whether the safe→unsafe *orientation* carries the effect. It does not ask whether **this** direction in the residual stream is special. `specificity` is the real arm's displacement from chance over the largest displacement any random or Gram-matched lens reaches in the same cell: **at or below 1.0, the result is not specific to the lens.**

| lens | layer | relative_depth | sign_consistency_z | permutation_p | displacement | control_displacement | specificity | beats_random_lens |
|---|---|---|---|---|---|---|---|---|
| jlens | -1 |  | 0.125 | 0.900 | 0.375 | 0.375 | 1.000 | False |
| jlens | 0 | 0.000 | 0.528 | 0.928 | 0.028 | 0.153 | 0.182 | False |
| jlens | 3 | 0.097 | 0.597 | 0.148 | 0.097 | 0.042 | 2.333 | True |
| jlens | 7 | 0.226 | 0.486 | 0.286 | 0.014 | 0.097 | 0.143 | False |
| jlens | 11 | 0.355 | 0.528 | 0.924 | 0.028 | 0.208 | 0.133 | False |
| jlens | 15 | 0.484 | 0.403 | 0.118 | 0.097 | 0.111 | 0.875 | False |
| jlens | 19 | 0.613 | 0.208 | 0.000 | 0.292 | 0.167 | 1.750 | True |
| jlens | 23 | 0.742 | 0.292 | 0.000 | 0.208 | 0.361 | 0.577 | False |
| jlens | 27 | 0.871 | 0.333 | 0.000 | 0.167 | 0.264 | 0.632 | False |
| jlens | 31 | 1.000 | 0.264 | 0.000 | 0.236 | 0.333 | 0.708 | False |
| logit | -1 |  | 0.167 | 0.684 | 0.333 | 0.375 | 0.889 | False |
| logit | 0 | 0.000 | 0.500 | 0.608 | 0.000 | 0.153 | 0.000 | False |
| logit | 3 | 0.097 | 0.556 | 0.442 | 0.056 | 0.042 | 1.333 | True |
| logit | 7 | 0.226 | 0.556 | 0.786 | 0.056 | 0.097 | 0.571 | False |
| logit | 11 | 0.355 | 0.458 | 0.450 | 0.042 | 0.208 | 0.200 | False |
| logit | 15 | 0.484 | 0.403 | 0.012 | 0.097 | 0.111 | 0.875 | False |
| logit | 19 | 0.613 | 0.208 | 0.000 | 0.292 | 0.167 | 1.750 | True |
| logit | 23 | 0.742 | 0.264 | 0.000 | 0.236 | 0.361 | 0.654 | False |
| logit | 27 | 0.871 | 0.278 | 0.000 | 0.222 | 0.264 | 0.842 | False |
| logit | 31 | 1.000 | 0.264 | 0.000 | 0.236 | 0.333 | 0.708 | False |
| rlens | -1 |  | 0.111 | 0.800 | 0.389 | 0.375 | 1.037 | True |
| rlens | 0 | 0.000 | 0.528 | 0.634 | 0.028 | 0.153 | 0.182 | False |
| rlens | 3 | 0.097 | 0.542 | 0.974 | 0.042 | 0.042 | 1.000 | False |
| rlens | 7 | 0.226 | 0.528 | 0.726 | 0.028 | 0.097 | 0.286 | False |
| rlens | 11 | 0.355 | 0.500 | 0.018 | 0.000 | 0.208 | 0.000 | False |
| rlens | 15 | 0.484 | 0.403 | 0.004 | 0.097 | 0.111 | 0.875 | False |
| rlens | 19 | 0.613 | 0.208 | 0.000 | 0.292 | 0.167 | 1.750 | True |
| rlens | 23 | 0.742 | 0.306 | 0.000 | 0.194 | 0.361 | 0.538 | False |
| rlens | 27 | 0.871 | 0.319 | 0.000 | 0.181 | 0.264 | 0.684 | False |
| rlens | 31 | 1.000 | 0.264 | 0.000 | 0.236 | 0.333 | 0.708 | False |

### Table 12 — is the contrast a distribution artifact?

`corr_contrast_entropy` and `corr_contrast_norm` correlate the paired contrast against the paired difference in the candidate distribution's entropy and score norm. A large |r| means the contrast tracks the *shape* of the distribution rather than its content, which would explain a consistent sign without any concept being involved.

| condition | mean_delta_contrast_z | corr_contrast_entropy | corr_contrast_norm | mean_delta_entropy |
|---|---|---|---|---|
| clean_heldout | -0.200 | 0.155 | -0.096 | 0.005 |
| normalize | -0.209 | 0.102 | -0.055 | 0.005 |
| rename_only | 0.007 | -0.275 | 0.174 | 0.002 |
| opaque_only | -0.070 | -0.017 | 0.134 | 0.000 |
| encode_only | -0.121 | 0.183 | -0.004 | 0.002 |
| flatten_only | 0.019 | 0.028 | -0.099 | 0.002 |
| rename_cumulative | -0.023 | -0.039 | 0.014 | 0.002 |
| rename_opaque | 0.020 | -0.104 | 0.300 | 0.002 |
| rename_opaque_encode | 0.021 | -0.033 | 0.221 | 0.001 |
| rename_opaque_encode_flatten | -0.120 | 0.056 | -0.191 | 0.000 |

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

`mismatched_pairs` redraws the SAFE partner from the safe pool, so the label difference survives it and its mean is invariant by construction; it can only move the per-pair statistics. `same_label_unsafe` and `same_label_safe` take BOTH members from one pole, so the label difference is gone and the expected contrast is zero — that is the arm a label claim has to clear.

| arm | lens | mean_delta_contrast_z | sign_consistency_z | permutation_p | n_pairs |
|---|---|---|---|---|---|
| mismatched_pairs | jlens | -0.050 | 0.500 | 0.468 | 72 |
| mismatched_pairs | logit | -0.126 | 0.444 | 0.072 | 72 |
| mismatched_pairs | rlens | -0.150 | 0.417 | 0.010 | 72 |
| same_label_unsafe | jlens | -0.037 | 0.486 | 0.522 | 72 |
| same_label_unsafe | logit | -0.034 | 0.500 | 0.568 | 72 |
| same_label_unsafe | rlens | -0.031 | 0.458 | 0.560 | 72 |
| same_label_safe | jlens | 0.048 | 0.597 | 0.472 | 72 |
| same_label_safe | logit | 0.049 | 0.556 | 0.416 | 72 |
| same_label_safe | rlens | 0.050 | 0.583 | 0.360 | 72 |
| random_lens | random | 0.403 | 0.611 | 0.002 | 72 |
| gram_random_lens | gram_random | 0.067 | 0.528 | 0.340 | 72 |
| role_swap_0 | jlens | -0.085 | 0.405 | 0.174 | 37 |
| role_swap_0 | logit | -0.158 | 0.432 | 0.108 | 37 |
| role_swap_0 | rlens | -0.208 | 0.351 | 0.008 | 37 |
| role_swap_1 | jlens | -0.112 | 0.400 | 0.302 | 35 |
| role_swap_1 | logit | -0.194 | 0.371 | 0.070 | 35 |
| role_swap_1 | rlens | -0.192 | 0.457 | 0.034 | 35 |

### Concept tokens at the reported cell

| token | mean_delta_z | rank | sign_consistency | mean_prob_unsafe | mean_prob_safe |
|---|---|---|---|---|---|
|  vulnerable | 0.330 | 72 | 0.792 | 0.006 | 0.005 |
|  safe | 0.565 | 45 | 0.833 | 0.007 | 0.006 |
|  trusted | 0.386 | 66 | 0.750 | 0.005 | 0.005 |
|  clean | 0.637 | 28 | 0.917 | 0.006 | 0.005 |

