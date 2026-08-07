# E11 pilot go/no-go — deepseek-coder-6.7b (position: use)

**Verdict: NO-GO**

- **PASS** `behavioural_balanced_accuracy` — test balanced accuracy 0.771 (all pairs 0.758); per variant {'source': 0.807, 'target': 0.736}; threshold 0.75
- **PASS** `readout_beats_random_control` — layer 16 (chosen on calibration): accuracy advantage +0.118 [+0.064, +0.171]
- **FAIL** `swap_moves_logits_toward_swapped_value` — site L8+16+24 (chosen on calibration): paired logit shift +0.001 [-0.003, +0.004], flip rate 0.000
- **FAIL** `swap_is_specific_to_the_value_subspace` — vs logit_value: +0.001 [-0.002, +0.004] (the Jacobian correction, not the unembedding); vs jlens_offvalue: -0.004 [-0.008, -0.001] (these values, not the digit subspace at large)

## Cross-operation consistency

- families: 2, min per-family shift +0.000, all positive: True, all CIs positive: False

## No-op control

- max |Δ logit-diff| = 0.00e+00 (passes: True)
