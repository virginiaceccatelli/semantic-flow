# E11 pilot go/no-go — deepseek-coder-6.7b (position: use)

**Verdict: NO-GO**

- **FAIL** `behavioural_balanced_accuracy` — test balanced accuracy 0.706 (all pairs 0.709); per variant {'source': 0.721, 'target': 0.691}; threshold 0.75
- **FAIL** `readout_beats_random_control` — layer 16 (chosen on calibration): accuracy advantage +0.056 [-0.007, +0.117]
- **FAIL** `swap_moves_logits_toward_swapped_value` — site L8+16+24 (chosen on calibration): paired logit shift +0.001 [-0.002, +0.004], flip rate 0.000
- **FAIL** `swap_is_specific_to_the_value_subspace` — vs logit_value: +0.001 [-0.002, +0.005] (the Jacobian correction, not the unembedding); vs jlens_offvalue: -0.005 [-0.009, -0.002] (these values, not the digit subspace at large)

## Cross-operation consistency

- families: 5, min per-family shift -0.004, all positive: False, all CIs positive: False

## No-op control

- max |Δ logit-diff| = 0.00e+00 (passes: True)
