# E11 pilot go/no-go — deepseek-coder-6.7b (position: answer)

**Verdict: NO-GO**

- **FAIL** `behavioural_balanced_accuracy` — test balanced accuracy 0.706 (all pairs 0.709); per variant {'source': 0.721, 'target': 0.691}; threshold 0.75
- **PASS** `readout_beats_random_control` — layer 31 (chosen on calibration): accuracy advantage +0.257 [+0.218, +0.297]
- **PASS** `swap_moves_logits_toward_swapped_value` — site L24 (chosen on calibration): paired logit shift +0.141 [+0.111, +0.175], flip rate 0.034
- **FAIL** `swap_is_specific_to_the_value_subspace` — vs logit_value: -0.016 [-0.024, -0.009] (the Jacobian correction, not the unembedding); vs jlens_offvalue: +0.150 [+0.117, +0.189] (these values, not the digit subspace at large)

## Cross-operation consistency

- families: 5, min per-family shift -0.024, all positive: False, all CIs positive: False

## No-op control

- max |Δ logit-diff| = 0.00e+00 (passes: True)
